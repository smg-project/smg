//! Check a local registration against the gateway's settings and workers.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::{model_card::ModelCard, worker::MmProcessingMode};
use smg_worker::{
    identity::{
        basic::{ConnectionMode, Worker},
        endpoint::Endpoint,
        metadata::join_base,
    },
    workflow::{require_host, WorkerKind},
};
use tracing::{info, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    admin::worker_service::{control_character_refusal, logged_strings},
    config::ParserChoice,
    routers::{
        common::snapshot::WorkerRegistry,
        grpc::{
            multimodal::{
                mm_settings, pool_processes_media_refs, registration_lacks_media_processing,
            },
            regular::symphony::{
                check_names, labels_select_nothing, set_aside_by_the_registry, table_label_read,
                tables_by_alias, ByAlias,
            },
            utils::{ModelNames, Names},
            zmq_client::zmq_handshake_address,
        },
    },
    workflow::data::WorkerWorkflowData,
};

/// The id of the gateway's check on the local line.
pub(crate) const CHECK_LOCAL_WORKER_STEP: &str = "check_local_worker";

/// Checks the new workers before they enter the registry and records which
/// table's failure counters registration will publish.
pub struct CheckLocalWorkerStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for CheckLocalWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let app_context = require_host(context.data.host.as_ref())?;
        let workers = context.data.actual_workers.as_deref().unwrap_or_default();
        let first = workers
            .first()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;
        // Creation gives every rank the same card.
        let model_card = first
            .models()
            .into_iter()
            .next()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("model_card".to_string()))?;
        let config = &context.data.config;
        let connection_mode = first.connection_mode();

        // An unknown override would leave the model's output unparsed.
        validate_parser_overrides(
            &model_card,
            &config.url,
            app_context.router_config.parser,
            app_context.tool_parser_factory.as_ref(),
            app_context.reasoning_parser_factory.as_ref(),
        )
        .map_err(|message| WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message,
        })?;

        warn_on_conflicting_parser_overrides(
            &model_card,
            &config.url,
            &app_context.worker_registry,
        );

        let flags = Names {
            tool: app_context.router_config.tool_call_parser.clone(),
            reasoning: app_context.router_config.reasoning_parser.clone(),
        };
        if let Some(message) = registry_reads_over_the_flags(
            &model_card,
            &config.url,
            app_context.router_config.parser,
            &flags,
            &app_context.worker_registry,
        ) {
            warn!(model = %model_card.id, worker = %Endpoint::redacted_str(&config.url), "{message}");
        }
        if let Some((read, set_aside)) = read_by_an_alias(
            &model_card,
            app_context.router_config.parser,
            *connection_mode,
            &flags,
            &app_context.by_alias,
            &app_context.worker_registry,
        ) {
            info!(model = %model_card.id, "{read}");
            if let Some(set_aside) = set_aside {
                warn!(model = %model_card.id, "{set_aside}");
            }
        }

        validate_zmq_handshake_address(
            &app_context.worker_registry,
            first.metadata().base_url(),
            first.metadata().spec.zmq_handshake_address.as_deref(),
            *connection_mode,
        )
        .map_err(|message| WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message,
        })?;

        // Publish the table's counters only after the worker is registered.
        context.data.host_notes.symphony_table = symphony_table_of(
            &model_card,
            app_context.router_config.parser,
            *connection_mode,
            &flags,
            &app_context.by_alias,
        );

        if mm_settings().processing.value == MmProcessingMode::Worker {
            warn_about_media_processing(workers, &app_context.worker_registry);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// A registration that worker-side media placement cannot use is said
/// once, here, with the worker's address: an engine with a vision
/// encoder that does not process media references never receives a
/// media reference under `--mm-processing worker` (a request with
/// per-item hints it still serves, through the router's pipeline, unless
/// every media-processing worker of the model applies the hints). The
/// line says what follows for the model, read from its pool (the
/// registered workers and the ones registering with it): with a capable
/// worker beside this one the references are served (selection picks
/// that worker); without one the model's requests that forward media
/// references are refused (`no_media_ref_capable_worker`) and
/// `/readiness` names the model until such a worker registers.
/// The registration itself is not refused. A model on the
/// encode-prefill-decode pipeline (an encode worker among its
/// registered workers or these) is not warned about: references never
/// travel on it, and selection answers with the fix that applies there.
fn warn_about_media_processing(workers: &[Arc<dyn Worker>], registry: &WorkerRegistry) {
    for worker in workers {
        let registered = registry.get_by_model(worker.model_id());
        let pool = || {
            workers
                .iter()
                .chain(registered.iter())
                .filter(|other| other.model_id() == worker.model_id())
        };
        if !registration_lacks_media_processing(worker.as_ref(), pool()) {
            continue;
        }
        let consequence = if pool_processes_media_refs(pool()) {
            "other workers of the model process them"
        } else {
            "the model's requests that forward media references are refused until a \
             worker of it processes them"
        };
        warn!(
            url = %worker.redacted_url(),
            model = %worker.model_id(),
            "worker reports a vision encoder but no worker-side media processor (no \
             mm_processor label): under --mm-processing worker it receives no media \
             references; {consequence}"
        );
    }
}

/// Warn when a new worker's parser overrides disagree with an already
/// registered worker serving the same model. Resolution stays deterministic
/// (lexicographically smallest name wins), but only one format parses
/// correctly — operators should see the conflict at deploy time.
pub(super) fn warn_on_conflicting_parser_overrides(
    card: &ModelCard,
    worker_url: &str,
    registry: &WorkerRegistry,
) {
    for existing in registry.get_by_model(&card.id).iter() {
        // The worker itself (every rank of it), replaced at its address: its
        // old card is still registered while the new one is checked. The
        // base URL carries no trailing slash; the address as given may.
        if existing.base_url() == join_base(worker_url) {
            continue;
        }
        let Some(existing_card) = existing.metadata().spec.models.find(&card.id) else {
            continue;
        };
        for (kind, new_name, existing_name) in [
            ("tool_parser", &card.tool_parser, &existing_card.tool_parser),
            (
                "reasoning_parser",
                &card.reasoning_parser,
                &existing_card.reasoning_parser,
            ),
        ] {
            if let (Some(new_name), Some(existing_name)) = (new_name, existing_name) {
                if new_name != existing_name {
                    warn!(
                        model = %card.id,
                        new_worker = %Endpoint::redacted_str(worker_url),
                        existing_worker = existing.redacted_url(),
                        %kind,
                        new = %new_name,
                        existing = %existing_name,
                        "Workers for one model declare conflicting parser \
                         overrides; the lexicographically smallest name wins \
                         at request time"
                    );
                }
            }
        }
    }
}

/// Reject a card the registry cannot take as it came. First a control character in its id or
/// parser names, with the door's sentence (`control_character_refusal`): the door checked the
/// strings a request carried, and these may be the worker's own report. Then parser-override
/// names the registries don't know: the legacy parsers', and, under `--parser symphony`,
/// Symphony's tables (`regular::symphony::check_names`). A name whose factory is absent is not
/// checked against a registry (parsers unused in that configuration).
pub(super) fn validate_parser_overrides(
    card: &ModelCard,
    worker_url: &str,
    parser: ParserChoice,
    tool_parser_factory: Option<&tool_parser::ParserFactory>,
    reasoning_parser_factory: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), String> {
    if let Some(refusal) = control_character_refusal(logged_strings(card)) {
        return Err(refusal);
    }
    check_names(
        parser,
        &card_names(card),
        tool_parser_factory,
        reasoning_parser_factory,
    )
    .map_err(|why| {
        format!(
            "worker {} declares a parser override for model '{}' the gateway cannot \
             read with: {why}",
            Endpoint::redacted_str(worker_url),
            card.id
        )
    })
}

/// The overrides creation resolved from the worker's card and labels.
fn card_names(card: &ModelCard) -> Names {
    Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    }
}

/// Under `--parser symphony`, the table whose failure classes the registration step publishes at
/// zero for a gRPC or ZMQ worker of `card`'s model, as this worker's labels and the gateway's flags
/// select it, so `failed`, `malformed` and the rest exist before the first such output
/// (`regular::symphony::telemetry::init_table_series`, called once the worker is in the registry).
/// Every worker of a model resolves one (a later worker whose labels select another table brings
/// that table's series; publishing a table again sets nothing). An HTTP worker resolves none: only
/// the gRPC and ZMQ pipeline reads with Symphony. The legacy parser choice and a model the legacy
/// parsers read resolve none either (the fallback series of such a model, under `none`, exist from
/// start-up).
fn symphony_table_of(
    card: &ModelCard,
    parser: ParserChoice,
    connection_mode: ConnectionMode,
    flags: &Names,
    by_alias: &[ByAlias],
) -> Option<&'static str> {
    if parser != ParserChoice::Symphony || connection_mode == ConnectionMode::Http {
        return None;
    }
    let labels = card_names(card);
    table_label_read(
        &ModelNames {
            labels,
            flags,
            by_alias,
        },
        &card.id,
    )
}

/// Under `--parser symphony`, the gateway's flags read a model ahead of the registry for every
/// worker whose labels select no reader (none, or a reasoning name alone that no table answers to
/// or that gives way), except a `--reasoning-parser` alone that selects a table
/// (`regular::symphony::names::flags_read_first`): the registry's table reads a model it knows then
/// (`regular::symphony::names::reads`). When that name selects another table than the registry's,
/// the registration of the model's first worker says so: `Some(message)` while no worker of the
/// model is registered, naming the model, both tables and the one used; `None` for the model's
/// later workers, a worker whose labels select a reader, a model the registry does not know, flags
/// that read first, select nothing or name the registry's own table, and the legacy parser
/// choice. Workers of one model registering at the same moment may each say it, as the override
/// check above may: the registry gains the worker a step later.
fn registry_reads_over_the_flags(
    card: &ModelCard,
    worker_url: &str,
    parser: ParserChoice,
    flags: &Names,
    registry: &WorkerRegistry,
) -> Option<String> {
    if parser != ParserChoice::Symphony
        || !labels_select_nothing(&card_names(card))
        || !registry.get_by_model(&card.id).is_empty()
    {
        return None;
    }
    let (table, named) = set_aside_by_the_registry(flags, &card.id)?;
    Some(format!(
        "the registry's table '{}' reads model '{}' for worker {}, whose labels select \
         no reader; the gateway's --reasoning-parser '{}', set alone, names {named}, which \
         reads only the models the registry does not know (a --tool-call-parser beside it puts \
         the flags before the registry)",
        table.name(),
        card.id,
        Endpoint::redacted_str(worker_url),
        flags.reasoning.as_deref().unwrap_or_default(),
    ))
}

/// Under `--parser symphony`, a gRPC or ZMQ model that no parser name is set for, the registry
/// does not know and no legacy registry detects a parser for is read by the registry's table for a
/// `--model-alias` of it that the registry knows, the first in sorted order
/// (`regular::symphony::names::read_by_alias`). The registration of the
/// model's first worker says which alias gives which table, and, where another such alias of the
/// model names another table, a second message names each with its table: `None` for the model's
/// later workers, a model read any other way, an HTTP worker and the legacy parser choice. Workers
/// of one model registering at the same moment may each say it, as the checks above may.
fn read_by_an_alias(
    card: &ModelCard,
    parser: ParserChoice,
    connection_mode: ConnectionMode,
    flags: &Names,
    by_alias: &[ByAlias],
    registry: &WorkerRegistry,
) -> Option<(String, Option<String>)> {
    if parser != ParserChoice::Symphony
        || connection_mode == ConnectionMode::Http
        || !registry.get_by_model(&card.id).is_empty()
    {
        return None;
    }
    let labels = card_names(card);
    let tables = tables_by_alias(
        &ModelNames {
            labels,
            flags,
            by_alias,
        },
        &card.id,
    );
    let ((alias, table), rest) = tables.split_first()?;
    let read = format!(
        "the registry's table '{}' reads model '{}' by its --model-alias '{alias}': the registry \
         does not know the served name and no parser name is set for the model",
        table.name(),
        card.id,
    );
    let others: Vec<String> = rest
        .iter()
        .filter(|(_, other)| other != table)
        .map(|(alias, other)| format!("'{alias}' names '{}'", other.name()))
        .collect();
    let set_aside = (!others.is_empty()).then(|| {
        format!(
            "model '{}' has --model-alias names of more than one table: '{alias}', the first in \
             sorted order, gives it '{}'; set aside: {} (a parser label on the model's workers \
             names the table for this model; --tool-call-parser names it for every model)",
            card.id,
            table.name(),
            others.join(", "),
        )
    });
    Some((read, set_aside))
}

/// Reject a ZMQ worker whose TCP handshake address is unusable or already
/// claimed.
///
/// Both failures strand the worker the same way — its handshake bind fails on
/// every connect attempt with a bare transport error, hours after a
/// registration that reported success — so both are named here instead.
///
/// *Unusable*: the base URL is not an `ipc://` path, or the
/// `zmq_handshake_address` override is not a `tcp://` address.
///
/// *Claimed*: the handshake port is a hash of the ipc path, so a collision
/// between two worker URLs is deterministic and permanent.
fn validate_zmq_handshake_address(
    registry: &WorkerRegistry,
    url: &str,
    handshake_override: Option<&str>,
    connection_mode: ConnectionMode,
) -> Result<(), String> {
    if connection_mode != ConnectionMode::Zmq {
        return Ok(());
    }
    let handshake = zmq_handshake_address(url, handshake_override)?;
    for existing in registry.get_all() {
        let metadata = existing.metadata();
        // Sockets are bound against the base URL (a dp-expanded worker carries
        // a `@rank` suffix on `spec.url` but binds the base).
        let existing_url = metadata.base_url();
        if *existing.connection_mode() != ConnectionMode::Zmq || existing_url == url {
            continue;
        }
        // A peer that reached the registry with an unusable address (some other
        // path registered it) has no address to collide with — that is its own
        // problem, not a reason to reject this registration.
        let Ok(existing_handshake) =
            zmq_handshake_address(existing_url, metadata.spec.zmq_handshake_address.as_deref())
        else {
            continue;
        };
        if existing_handshake == handshake {
            return Err(format!(
                "ZMQ worker {} would bind handshake address {handshake}, already claimed by \
                 worker {}: the derived port is a hash of the ipc path, so rename \
                 this worker's ipc path or set zmq_handshake_address to a free tcp:// address",
                Endpoint::redacted_str(url),
                Endpoint::redacted_str(existing_url)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/check_local_worker/tests.rs"]
mod tests;
