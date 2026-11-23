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
use tracing::warn;
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
/// per-item hints it still serves, through the router's pipeline). The
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
) -> Option<&'static str> {
    if parser != ParserChoice::Symphony || connection_mode == ConnectionMode::Http {
        return None;
    }
    let labels = card_names(card);
    table_label_read(&ModelNames { labels, flags }, &card.id)
}

/// Under `--parser symphony`, the registry's table reads a model it knows for every worker whose
/// labels select no reader (none, or a reasoning name alone that no table answers to or that gives
/// way), whatever the
/// gateway's `--tool-call-parser` and `--reasoning-parser` name (`regular::symphony::names::reads`). When
/// they name another reader, the registration of the model's first worker says so: `Some(message)`
/// while no worker of the model is registered, naming the model, both readers and the one used;
/// `None` for the model's later workers, a worker whose labels select a reader, a model the
/// registry does not know, flags that select nothing or the registry's own table, and the legacy
/// parser choice. Workers of one model registering at the same moment may each say it, as the
/// override check above may: the registry gains the worker a step later.
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
    let spell =
        |name: Option<&str>| name.map_or_else(|| "unset".to_string(), |name| format!("'{name}'"));
    Some(format!(
        "the registry's table '{}' reads model '{}' for worker {}, whose labels select \
         no reader; the gateway's --tool-call-parser {} / --reasoning-parser {} name {named}, which \
         reads only the models the registry does not know",
        table.name(),
        card.id,
        Endpoint::redacted_str(worker_url),
        spell(flags.tool.as_deref()),
        spell(flags.reasoning.as_deref()),
    ))
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
mod tests {
    use std::{collections::HashMap, time::Duration};

    use openai_protocol::worker::{RuntimeType, WorkerSpec, WorkerStatus, WorkerType};
    use smg_worker::{
        identity::basic::DEFAULT_BOOTSTRAP_PORT,
        jobs::WorkerJobAsker,
        test_support::{labels_of, RefusedAddress},
        workflow::{build_model_card, CreateLocalWorkerStep},
    };
    use tracing_test::traced_test;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::RouterConfig,
        routers::grpc::backend_client::BasicWorkerBuilder,
        server::http_client,
        workflow::{data::WorkerRegistrationMode, steps::create_worker_workflow_data},
    };

    #[tokio::test]
    async fn the_check_uses_the_created_workers_handshake_address() {
        let host = make_app_context();
        let path = "/tmp/worker-check-step.ipc";
        let mut spec = WorkerSpec::new(path);
        spec.labels.insert("model_id".into(), "m".into());
        let mut data = create_worker_workflow_data(
            spec,
            WorkerRegistrationMode::Upsert,
            Arc::clone(&host),
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Zmq);
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(
            CheckLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert!(host.worker_registry.is_empty());

        let address = zmq_handshake_address("ipc:///tmp/worker-check-step.ipc", None).unwrap();
        host.worker_registry
            .register(zmq_worker("ipc:///tmp/other-worker.ipc", Some(&address)))
            .unwrap();
        let error = CheckLocalWorkerStep.execute(&mut ctx).await.unwrap_err();
        assert_eq!(error.to_string(),
            format!("Step failed: create_worker - ZMQ worker ipc:///tmp/worker-check-step.ipc would bind handshake address {address}, already claimed by worker ipc:///tmp/other-worker.ipc: the derived port is a hash of the ipc path, so rename this worker's ipc path or set zmq_handshake_address to a free tcp:// address"));
        assert!(host
            .worker_registry
            .get_by_url("ipc:///tmp/worker-check-step.ipc")
            .is_none());
    }

    #[traced_test]
    #[test]
    fn the_media_warning_reads_each_new_workers_labels() {
        use smg_worker::labels::SUPPORTS_VISION_LABEL;

        let registry = WorkerRegistry::new();
        let worker = |labels: &[(&str, &str)]| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new("grpc://w:9000")
                    .model(ModelCard::new("m"))
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Vllm)
                    .labels(labels_of(labels))
                    .build(),
            )
        };
        let with_processor = [worker(&[
            (SUPPORTS_VISION_LABEL, "true"),
            ("mm_processor", "inprocess"),
        ])];
        warn_about_media_processing(&with_processor, &registry);
        assert!(!logs_contain(
            "worker reports a vision encoder but no worker-side media processor"
        ));

        let without_processor = [worker(&[(SUPPORTS_VISION_LABEL, "true")])];
        warn_about_media_processing(&without_processor, &registry);
        assert!(logs_contain(
            "worker reports a vision encoder but no worker-side media processor"
        ));
    }

    /// The parser-override warning skips the worker itself when it is replaced
    /// at an address registered with a trailing slash: the registry's key keeps
    /// the slash, the join base does not. Another address with the other
    /// override is still warned about.
    #[traced_test]
    #[test]
    fn the_parser_override_warning_skips_the_worker_replaced_at_a_slashed_address() {
        let mut registered = ModelCard::new("m");
        registered.tool_parser = Some("one".into());
        let registry = WorkerRegistry::new();
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("http://h:8000/")
                    .model(registered)
                    .build(),
            ))
            .unwrap();
        let mut replacement = ModelCard::new("m");
        replacement.tool_parser = Some("two".into());

        warn_on_conflicting_parser_overrides(&replacement, "http://h:8000/", &registry);
        assert!(!logs_contain("conflicting parser"));

        warn_on_conflicting_parser_overrides(&replacement, "http://other:8000", &registry);
        assert!(logs_contain("conflicting parser"));
    }

    /// Under `--parser symphony`, a gRPC worker's registration resolves the table whose failure
    /// classes are published at zero: the registry's table for an unlabeled worker of a model it
    /// knows, and another table for a worker whose labels name one. The legacy parser choice and
    /// an HTTP worker (the pipeline that reads with Symphony never serves it) resolve none.
    #[test]
    fn a_workers_registration_resolves_the_table_whose_failure_classes_are_published() {
        let known = ModelCard::new("Qwen/Qwen3-8B");
        let table = |card: &ModelCard, parser, connection_mode| {
            symphony_table_of(card, parser, connection_mode, &Names::default())
        };
        assert_eq!(
            table(&known, ParserChoice::Legacy, ConnectionMode::Grpc),
            None,
            "the legacy parsers publish no table"
        );
        assert_eq!(
            table(&known, ParserChoice::Symphony, ConnectionMode::Http),
            None,
            "an HTTP worker publishes no table"
        );
        assert_eq!(
            table(&known, ParserChoice::Symphony, ConnectionMode::Grpc),
            Some("qwen3")
        );
        // A later worker of the model whose labels name another table brings
        // that table's series: requests count under it from then on.
        let mut labeled = known.clone();
        labeled.tool_parser = Some("kimi_k2".to_string());
        assert_eq!(
            table(&labeled, ParserChoice::Symphony, ConnectionMode::Zmq),
            Some("kimi_k2")
        );
    }

    /// The table a worker's registration resolves is published by the registration step, once the
    /// worker is in the registry: under `--parser symphony` a gRPC worker of a model the registry
    /// knows resolves the registry's table in the check step, a registration the registration
    /// step refuses leaves no series behind, and an accepted one leaves the table's failure classes
    /// at zero.
    #[tokio::test]
    async fn a_refused_registration_publishes_no_table_and_an_accepted_one_its_failure_classes() {
        use crate::workflow::steps::RegisterWorkersStep;

        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let symphony = || {
            app_context_with(
                RouterConfig::builder()
                    .worker_startup_timeout_secs(1)
                    .parser(ParserChoice::Symphony)
                    .build_unchecked(),
            )
        };

        // The replacement of a worker the registry no longer holds is refused.
        let refused = WorkerRegistrationMode::ReplaceById {
            worker_id: "worker-gone".to_string(),
            expected_revision: 0,
        };
        let mut ctx = grpc_worker_context("Qwen/Qwen3-8B", refused, symphony());
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(
            CheckLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(ctx.data.host_notes.symphony_table, Some("qwen3"));
        assert!(RegisterWorkersStep.execute(&mut ctx).await.is_err());
        assert!(
            !handle.render().contains("smg_symphony_"),
            "{}",
            handle.render()
        );

        let mut ctx =
            grpc_worker_context("Qwen/Qwen3-8B", WorkerRegistrationMode::Upsert, symphony());
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(
            CheckLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(ctx.data.host_notes.symphony_table, Some("qwen3"));
        assert_eq!(
            RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let scrape = handle.render();
        assert!(
            scrape.lines().any(|line| {
                line.starts_with("smg_symphony_outputs_total")
                    && line.contains("table=\"qwen3\"")
                    && line.contains("outcome=\"failed\"")
                    && line.ends_with(" 0")
            }),
            "{scrape}"
        );
    }

    /// A registration context for a gRPC worker of `model` at a loopback URL
    /// under `app_context`'s configuration.
    fn grpc_worker_context(
        model: &str,
        registration_mode: WorkerRegistrationMode,
        app_context: Arc<AppContext>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        use openai_protocol::worker::WorkerModels;
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("grpc://127.0.0.1:18001");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new(model)));
        let mut data =
            create_worker_workflow_data(spec, registration_mode, app_context, WorkerJobAsker::Api);
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Grpc);
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// Under `--parser symphony`, the registration of a model's first worker whose labels select
    /// no reader (none, or a reasoning name no table answers to) says that the registry's table
    /// reads the model, not what the gateway's flags name, naming both; the model's next worker,
    /// a worker whose labels select a reader, a model the registry does not know, flags that name
    /// the registry's own table or nothing, and the legacy parser choice, say nothing.
    #[test]
    fn the_registrys_table_over_the_flags_is_said_once_per_model_at_registration() {
        let flags = Names {
            tool: Some("mistral_v13".to_string()),
            reasoning: None,
        };
        let known = ModelCard::new("Qwen/Qwen3-8B");
        let registry = WorkerRegistry::new();
        let said =
            |card: &ModelCard, url: &str, parser, flags: &Names, registry: &WorkerRegistry| {
                registry_reads_over_the_flags(card, url, parser, flags, registry)
            };

        // The worker is named through the mask: a credential in its address
        // is in no message (the helper masks, so no caller can forget).
        let first = said(
            &known,
            "grpc://user:secret@a:1",
            ParserChoice::Symphony,
            &flags,
            &registry,
        )
        .expect("the model's first worker says so");
        assert!(
            first.contains("'Qwen/Qwen3-8B'")
                && first.contains("'qwen3'")
                && first.contains("the table 'mistral_v13'")
                && first.contains("***@a:1")
                && !first.contains("secret"),
            "{first}"
        );
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://a:1")
                    .model(known.clone())
                    .build(),
            ))
            .expect("register");
        assert_eq!(
            said(
                &known,
                "grpc://b:1",
                ParserChoice::Symphony,
                &flags,
                &registry
            ),
            None,
            "the model's second worker"
        );

        let fresh = WorkerRegistry::new();
        let mut labeled = known.clone();
        labeled.reasoning_parser = Some("qwen3".to_string());
        assert_eq!(
            said(
                &labeled,
                "grpc://a:1",
                ParserChoice::Symphony,
                &flags,
                &fresh
            ),
            None,
            "a worker whose labels select a reader"
        );
        assert_eq!(
            said(
                &ModelCard::new("acme/ministral-3-14b-instruct-w4a16"),
                "grpc://a:1",
                ParserChoice::Symphony,
                &flags,
                &fresh
            ),
            None,
            "a model the registry does not know"
        );
        let own = Names {
            tool: Some("qwen".to_string()),
            reasoning: None,
        };
        assert_eq!(
            said(&known, "grpc://a:1", ParserChoice::Symphony, &own, &fresh),
            None,
            "the registry's own table"
        );
        // A label that selects nothing (a reasoning name no table answers to) falls through to
        // the registry as no label does, so the worker says it too.
        let mut idle_label = known.clone();
        idle_label.reasoning_parser = Some("deepseek_r1".to_string());
        assert!(
            said(
                &idle_label,
                "grpc://a:1",
                ParserChoice::Symphony,
                &flags,
                &fresh
            )
            .is_some_and(|message| {
                message.contains("'qwen3'") && message.contains("whose labels select no reader")
            }),
            "a label that selects nothing"
        );
        assert_eq!(
            said(
                &known,
                "grpc://a:1",
                ParserChoice::Symphony,
                &Names::default(),
                &fresh
            ),
            None,
            "no flags"
        );
        assert_eq!(
            said(&known, "grpc://a:1", ParserChoice::Legacy, &flags, &fresh),
            None,
            "the legacy parsers"
        );
    }

    fn make_app_context() -> Arc<AppContext> {
        app_context_with(
            RouterConfig::builder()
                .worker_startup_timeout_secs(1)
                .build_unchecked(),
        )
    }

    fn app_context_with(router_config: RouterConfig) -> Arc<AppContext> {
        app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()), None)
    }

    /// A registration context for a disaggregated worker at a loopback URL.
    fn worker_context(
        worker_type: WorkerType,
        bootstrap_port: Option<u16>,
        registration_mode: WorkerRegistrationMode,
    ) -> WorkflowContext<WorkerWorkflowData> {
        worker_context_on(
            worker_type,
            bootstrap_port,
            registration_mode,
            RuntimeType::Unspecified,
            None,
        )
    }

    /// A registration context for a disaggregated worker of `runtime` on
    /// `kv_connector`, at a loopback URL.
    fn worker_context_on(
        worker_type: WorkerType,
        bootstrap_port: Option<u16>,
        registration_mode: WorkerRegistrationMode,
        runtime: RuntimeType,
        kv_connector: Option<&str>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        use openai_protocol::worker::WorkerModels;
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("http://127.0.0.1:18000");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.worker_type = worker_type;
        spec.bootstrap_port = bootstrap_port;
        spec.runtime_type = runtime;
        spec.kv_connector = kv_connector.map(str::to_string);
        let mut data = create_worker_workflow_data(
            spec,
            registration_mode,
            make_app_context(),
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(http_client()));
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// The probe's count once it lands: the probe runs off the path, within
    /// its own timeout.
    async fn wait_for_probe_count(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
    ) -> String {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let rendered = handle.render();
            if let Some(line) = rendered
                .lines()
                .find(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            {
                return line.to_string();
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the failed probe is counted within its timeout:\n{rendered}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// This step builds a prefill worker with the bootstrap port and host its
    /// spec carries and probes nothing: the port is probed once the worker is
    /// registered (the register step), so a workflow that fails between the
    /// two is not reported as a registered worker with a dead port.
    #[tokio::test]
    async fn a_prefill_workers_bootstrap_port_is_probed_once_the_worker_is_registered() {
        use crate::workflow::steps::RegisterWorkersStep;

        let closed = RefusedAddress::reserve();
        let closed_port = closed.port();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let mut ctx = worker_context(
            WorkerType::Prefill,
            Some(closed_port),
            WorkerRegistrationMode::Upsert,
        );
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
        assert_eq!(worker.bootstrap_port(), Some(closed_port));
        assert_eq!(worker.bootstrap_host(), "127.0.0.1");
        // Built, not registered: nothing is probed yet.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !handle.render().contains("smg_pd_rendezvous_failures_total"),
            "{}",
            handle.render()
        );

        assert_eq!(
            RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let counted = wait_for_probe_count(&handle).await;
        assert!(
            counted.contains(r#"leg="prefill""#)
                && counted.contains(r#"phase="probe""#)
                && counted.ends_with(" 1"),
            "{counted}"
        );
    }

    /// A worker whose registration the register step refuses never joined the
    /// registry, so its port is not probed and no probe failure is counted
    /// for it; a decode worker without a bootstrap port is not probed either,
    /// since no partner is handed a port for it.
    #[tokio::test]
    async fn a_worker_the_registry_refuses_or_a_decode_worker_without_a_bootstrap_port_is_not_probed(
    ) {
        use crate::workflow::steps::RegisterWorkersStep;

        let closed = RefusedAddress::reserve();
        let closed_port = closed.port();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);

        // The replacement of a worker the registry no longer holds is refused.
        let refused = WorkerRegistrationMode::ReplaceById {
            worker_id: "worker-gone".to_string(),
            expected_revision: 0,
        };
        let mut ctx = worker_context(WorkerType::Prefill, Some(closed_port), refused);
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert!(RegisterWorkersStep.execute(&mut ctx).await.is_err());

        let mut ctx = worker_context(WorkerType::Decode, None, WorkerRegistrationMode::Upsert);
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(
            RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !handle.render().contains("smg_pd_rendezvous_failures_total"),
            "{}",
            handle.render()
        );
    }

    /// Whether something on this host answers on the default bootstrap port:
    /// the tests of a worker registered without a port cannot tell a probe that
    /// found a listener there from one that never ran, so they stand down.
    fn something_listens_on_the_default_bootstrap_port() -> bool {
        use std::net::{SocketAddr, TcpStream};

        let address = SocketAddr::from(([127, 0, 0, 1], DEFAULT_BOOTSTRAP_PORT));
        TcpStream::connect_timeout(&address, Duration::from_millis(500)).is_ok()
    }

    /// A prefill worker registered without a bootstrap port (a `POST /workers`
    /// without the field, a discovered pod without the annotation) is handed
    /// the default port on the request path, so the default port is what is
    /// probed for it: with nothing listening there, the failed probe is
    /// counted once the worker is registered, like a registered port's.
    #[tokio::test]
    async fn a_prefill_worker_registered_without_a_bootstrap_port_is_probed_on_the_default_port() {
        use crate::workflow::steps::RegisterWorkersStep;

        if something_listens_on_the_default_bootstrap_port() {
            return;
        }
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let mut ctx = worker_context(WorkerType::Prefill, None, WorkerRegistrationMode::Upsert);
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
        assert_eq!(
            worker.bootstrap_port(),
            None,
            "the spec keeps what was registered"
        );
        assert_eq!(
            RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let counted = wait_for_probe_count(&handle).await;
        assert!(
            counted.contains(r#"leg="prefill""#)
                && counted.contains(r#"phase="probe""#)
                && counted.ends_with(" 1"),
            "{counted}"
        );
    }

    /// A prefill worker on a KV store registered with the bootstrap port the
    /// rendezvous recipe carries: no partner is ever handed it, so nothing is
    /// probed and no probe failure is counted once the worker is registered.
    #[tokio::test]
    async fn a_store_prefill_workers_registered_bootstrap_port_is_not_probed() {
        use crate::workflow::steps::RegisterWorkersStep;

        let closed = RefusedAddress::reserve();
        let closed_port = closed.port();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);

        let mut ctx = worker_context_on(
            WorkerType::Prefill,
            Some(closed_port),
            WorkerRegistrationMode::Upsert,
            RuntimeType::Vllm,
            Some("MooncakeStoreConnector"),
        );
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
        assert_eq!(
            worker.bootstrap_port(),
            Some(closed_port),
            "the spec keeps what was registered"
        );
        assert_eq!(
            worker.metadata().spec.kv_connector.as_deref(),
            Some("MooncakeStoreConnector")
        );
        assert_eq!(
            RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        // A probe of the closed loopback port is refused and counted at once;
        // wait longer than that before reading nothing.
        tokio::time::sleep(Duration::from_millis(500)).await;
        let rendered = handle.render();
        assert!(
            !rendered.contains("smg_pd_rendezvous_failures_total"),
            "{rendered}"
        );
    }

    /// The text the registration of a worker that reports `labels` fails
    /// with.
    async fn failure_of_a_worker_reporting(labels: &[(&str, &str)]) -> String {
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::{create_worker_workflow_data, local::test_support::labels_of};

        let spec = WorkerSpec::new("http://worker:8000");
        let mut data = create_worker_workflow_data(
            spec,
            WorkerRegistrationMode::Upsert,
            make_app_context(),
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(http_client()));
        data.discovered_labels = labels_of(labels);
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap();
        CheckLocalWorkerStep
            .execute(&mut ctx)
            .await
            .expect_err("the registration fails")
            .to_string()
    }

    /// A model id the worker itself reports is refused like one a request
    /// carries: the registration fails with the door's sentence, which does
    /// not repeat the id.
    #[tokio::test]
    async fn a_reported_model_id_with_a_control_character_fails_the_registration() {
        assert_eq!(
            failure_of_a_worker_reporting(&[("served_model_name", "m\nINJECTED")]).await,
            "Step failed: create_worker - models[].id must not contain control characters"
        );
    }

    /// A tool parser name the worker reports is refused the same way.
    #[tokio::test]
    async fn a_reported_tool_parser_with_a_control_character_fails_the_registration() {
        assert_eq!(
            failure_of_a_worker_reporting(&[
                ("served_model_name", "m"),
                ("tool_parser", "p\nINJECTED"),
            ])
            .await,
            "Step failed: create_worker - models[].tool_parser must not contain control characters"
        );
    }

    /// A reasoning parser name the worker reports is refused the same way.
    #[tokio::test]
    async fn a_reported_reasoning_parser_with_a_control_character_fails_the_registration() {
        assert_eq!(
            failure_of_a_worker_reporting(&[
                ("served_model_name", "m"),
                ("reasoning_parser", "p\nINJECTED"),
            ])
            .await,
            "Step failed: create_worker - models[].reasoning_parser must not contain control characters"
        );
    }

    /// The check refuses a replacement behind its build. The build wrote no
    /// series, so each series of the worker that serves the address reads as
    /// it stood: healthy, and on HTTP/2.
    #[tokio::test]
    async fn a_replacement_the_check_refuses_leaves_the_live_workers_series_as_they_stood() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let url = "http://refused-by-the-check:8000";
        let series = || {
            let label = format!("worker=\"{url}\"");
            let mut lines: Vec<String> = handle
                .render()
                .lines()
                .filter(|line| line.contains(&label))
                .map(str::to_owned)
                .collect();
            lines.sort_unstable();
            lines
        };

        let app_context = make_app_context();
        let live: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .status(WorkerStatus::Ready)
                .http2(true)
                .build(),
        );
        app_context
            .worker_registry
            .register(live)
            .expect("a fresh URL registers");
        let stood = series();
        for gauge in ["smg_worker_health", "smg_worker_http2"] {
            assert!(
                stood.contains(&format!("{gauge}{{worker=\"{url}\"}} 1")),
                "{stood:?}"
            );
        }

        let mut data = create_worker_workflow_data(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            app_context,
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(http_client()));
        data.discovered_labels =
            labels_of(&[("served_model_name", "m"), ("tool_parser", "p\nINJECTED")]);
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap();
        CheckLocalWorkerStep
            .execute(&mut ctx)
            .await
            .expect_err("the check refuses the parser name");
        assert_eq!(series(), stood);
    }

    fn zmq_worker(url: &str, handshake_override: Option<&str>) -> Arc<dyn Worker> {
        let mut spec = WorkerSpec::new(url);
        spec.connection_mode = ConnectionMode::Zmq;
        spec.zmq_handshake_address = handshake_override.map(str::to_string);
        Arc::new(BasicWorkerBuilder::from_spec(spec).build())
    }

    #[test]
    fn zmq_handshake_collision_is_rejected_at_registration() {
        // Two ipc paths hashing to the same derived port would leave the second
        // worker's handshake bind failing forever with a bare transport error.
        let registry = WorkerRegistry::new();
        let first = "ipc:///tmp/smg-zmq/a.ipc";
        let handshake = zmq_handshake_address(first, None).expect("derived handshake address");
        registry.register(zmq_worker(first, None)).unwrap();

        // Same derived address reached through an explicit override.
        let err = validate_zmq_handshake_address(
            &registry,
            "ipc:///tmp/smg-zmq/b.ipc",
            Some(&handshake),
            ConnectionMode::Zmq,
        )
        .expect_err("a colliding handshake address must be rejected");
        assert!(err.contains(&handshake), "message was: {err}");
        assert!(err.contains(first), "message was: {err}");

        // A distinct path derives a distinct address; re-registering the same
        // URL (update path) is not a collision with itself.
        validate_zmq_handshake_address(
            &registry,
            "ipc:///tmp/smg-zmq/b.ipc",
            None,
            ConnectionMode::Zmq,
        )
        .expect("a distinct ipc path must be accepted");
        validate_zmq_handshake_address(&registry, first, None, ConnectionMode::Zmq)
            .expect("re-registering the same worker URL must be accepted");
        validate_zmq_handshake_address(&registry, "grpc://worker:8080", None, ConnectionMode::Grpc)
            .expect("non-ZMQ workers are not affected");
    }

    #[test]
    fn malformed_zmq_handshake_address_is_rejected_at_registration() {
        // A handshake address that can never bind must fail here, with the
        // reason named — not silently register a worker whose every connect
        // attempt dies on a bare transport error.
        let registry = WorkerRegistry::new();

        // The engine dials a TCP handshake, so an ipc:// override is unusable.
        let err = validate_zmq_handshake_address(
            &registry,
            "ipc:///tmp/smg-zmq/a.ipc",
            Some("ipc:///tmp/smg-zmq/handshake.sock"),
            ConnectionMode::Zmq,
        )
        .expect_err("a non-tcp:// handshake override must be rejected");
        assert!(err.contains("tcp://"), "message was: {err}");
        assert!(
            err.contains("ipc:///tmp/smg-zmq/handshake.sock"),
            "message was: {err}"
        );

        // A ZMQ worker whose base URL is not an ipc:// path derives nothing.
        let err = validate_zmq_handshake_address(
            &registry,
            "http://worker:8080",
            None,
            ConnectionMode::Zmq,
        )
        .expect_err("a non-ipc:// ZMQ base URL must be rejected");
        assert!(err.contains("ipc://"), "message was: {err}");

        // A well-formed tcp:// override on an ipc:// base is the supported form.
        validate_zmq_handshake_address(
            &registry,
            "ipc:///tmp/smg-zmq/a.ipc",
            Some("tcp://127.0.0.1:30500"),
            ConnectionMode::Zmq,
        )
        .expect("a tcp:// override must be accepted");

        // A peer that reached the registry with an unusable address is skipped,
        // not inherited: it cannot claim a handshake address to collide with.
        registry
            .register(zmq_worker("http://peer:8080", None))
            .expect("registering the malformed peer");
        validate_zmq_handshake_address(
            &registry,
            "ipc:///tmp/smg-zmq/a.ipc",
            None,
            ConnectionMode::Zmq,
        )
        .expect("an unusable peer address must not block a valid registration");
    }

    #[test]
    fn parser_override_labels_flow_into_card() {
        let spec = WorkerSpec::new("http://worker:8080");
        let labels = HashMap::from([
            ("tool_parser".to_string(), "json".to_string()),
            ("reasoning_parser".to_string(), "basic".to_string()),
            ("model_type".to_string(), "llama".to_string()),
        ]);

        let card = build_model_card("m", &spec, &labels, &HashMap::new());
        assert_eq!(card.tool_parser.as_deref(), Some("json"));
        assert_eq!(card.reasoning_parser.as_deref(), Some("basic"));

        // Empty label values are treated as unset.
        let labels = HashMap::from([("tool_parser".to_string(), String::new())]);
        let card = build_model_card("m", &spec, &labels, &HashMap::new());
        assert_eq!(card.tool_parser, None);
    }

    #[test]
    fn explicit_card_parser_wins_over_labels() {
        // A user-provided WorkerSpec card keeps its parser fields; labels
        // only fill gaps (same precedence as tokenizer_path).
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.models = openai_protocol::worker::WorkerModels::Single(Box::new(
            ModelCard::new("m").with_tool_parser("pythonic"),
        ));
        let labels = HashMap::from([
            ("tool_parser".to_string(), "json".to_string()),
            ("reasoning_parser".to_string(), "basic".to_string()),
        ]);

        let card = build_model_card("m", &spec, &labels, &HashMap::new());
        assert_eq!(card.tool_parser.as_deref(), Some("pythonic"));
        // The explicit card left reasoning unset — the label fills it.
        assert_eq!(card.reasoning_parser.as_deref(), Some("basic"));
    }

    /// An engine's own reasoning parser rides in its own label
    /// (`engine_reasoning_parser`), not in `reasoning_parser`, the per-model
    /// parser override registration validates against the gateway's registry:
    /// the engine's parser names are the engine's (`openai_gptoss`, `kimi_k2`),
    /// which the gateway may not have, and a shared name must not outrank the
    /// configured `--reasoning-parser`. Such a worker registers, its card
    /// carries no override, and its grammar start reads as advertised.
    #[test]
    fn an_engines_advertised_reasoning_parser_is_not_a_parser_override() {
        use openai_protocol::worker::GrammarStart;
        use smg_grpc_client::engine_labels::ServerInfo;

        let tool_factory = tool_parser::ParserFactory::default();
        let reasoning_factory = reasoning_parser::ParserFactory::default();
        let spec = WorkerSpec::new("grpc://engine:50051");
        for name in ["openai_gptoss", "kimi_k2", "qwen3", "glm45"] {
            let info = ServerInfo::Vllm(Box::new(
                smg_grpc_client::vllm_proto::GetServerInfoResponse {
                    engine_reasoning_parser: name.to_string(),
                    constrained_decoding_start: "after_reasoning".to_string(),
                    ..Default::default()
                },
            ));
            let labels = info.to_labels();
            assert_eq!(
                labels.get("engine_reasoning_parser").map(String::as_str),
                Some(name)
            );
            let card = build_model_card("m", &spec, &labels, &HashMap::new());
            assert_eq!(
                card.reasoning_parser, None,
                "{name}: no override from the engine's fact"
            );
            assert_eq!(card.tool_parser, None);
            validate_parser_overrides(
                &card,
                "grpc://engine:50051",
                ParserChoice::Legacy,
                Some(&tool_factory),
                Some(&reasoning_factory),
            )
            .unwrap_or_else(|e| panic!("{name}: the worker must register: {e}"));
            assert_eq!(
                GrammarStart::from_label(labels.get(GrammarStart::LABEL).map(String::as_str)),
                GrammarStart::AfterReasoning,
                "{name}: the grammar start reads as advertised"
            );
        }
    }

    #[test]
    fn a_symphony_table_is_a_parser_override_under_parser_symphony_alone() {
        let tool_factory = tool_parser::ParserFactory::default();
        let reasoning_factory = reasoning_parser::ParserFactory::default();
        let mut card = ModelCard::new("m");
        card.tool_parser = Some("ling".to_string());
        card.reasoning_parser = Some("ling".to_string());
        let check = |card: &ModelCard, parser| {
            validate_parser_overrides(
                card,
                "http://w:1",
                parser,
                Some(&tool_factory),
                Some(&reasoning_factory),
            )
        };
        assert!(check(&card, ParserChoice::Symphony).is_ok());
        let err = check(&card, ParserChoice::Legacy).expect_err("a table is no legacy parser");
        assert!(
            err.contains("'ling'") && err.contains("--parser symphony"),
            "{err}"
        );
        // A table no legacy parser shares binds the other override to the same table.
        card.reasoning_parser = Some("qwen3".to_string());
        let err = check(&card, ParserChoice::Symphony).expect_err("bound to the table");
        assert!(err.contains("same table"), "{err}");
        // A name both registries have passes under either parser.
        card.tool_parser = Some("kimi_k3".to_string());
        card.reasoning_parser = Some("kimi_k3".to_string());
        assert!(check(&card, ParserChoice::Symphony).is_ok());
        assert!(check(&card, ParserChoice::Legacy).is_ok());
    }

    #[test]
    fn unknown_parser_override_fails_validation() {
        let tool_factory = tool_parser::ParserFactory::default();
        let reasoning_factory = reasoning_parser::ParserFactory::default();

        // No overrides → nothing to validate.
        let card = ModelCard::new("m");
        assert!(validate_parser_overrides(
            &card,
            "http://w:1",
            ParserChoice::Legacy,
            Some(&tool_factory),
            Some(&reasoning_factory)
        )
        .is_ok());

        // Known tool parser passes; unknown names fail loudly.
        let card = ModelCard::new("m").with_tool_parser("json");
        assert!(validate_parser_overrides(
            &card,
            "http://w:1",
            ParserChoice::Legacy,
            Some(&tool_factory),
            None
        )
        .is_ok());

        // The refusal names the worker through the mask: a credential in its
        // address is in no refusal (the helper masks, so no caller can forget).
        let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
        let err = validate_parser_overrides(
            &card,
            "http://user:secret@w:1",
            ParserChoice::Legacy,
            Some(&tool_factory),
            None,
        )
        .expect_err("unknown tool parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");
        assert!(
            err.contains("http://***@w:1") && !err.contains("secret"),
            "{err}"
        );

        let card = ModelCard::new("m").with_reasoning_parser("definitely-not-a-parser");
        let err = validate_parser_overrides(
            &card,
            "http://user:secret@w:1",
            ParserChoice::Legacy,
            None,
            Some(&reasoning_factory),
        )
        .expect_err("unknown reasoning parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");
        assert!(
            err.contains("http://***@w:1") && !err.contains("secret"),
            "{err}"
        );

        // Absent factories skip validation (parsers unused in that config).
        let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
        assert!(
            validate_parser_overrides(&card, "http://w:1", ParserChoice::Legacy, None, None)
                .is_ok()
        );
    }

    #[tokio::test]
    async fn tokenspeed_discovery_bounds_each_rank_by_its_local_window() {
        use smg_grpc_client::{engine_labels::ServerInfo, tokenspeed_proto};
        use smg_worker::workflow::dp_info_from_labels;

        let host = app_context_with(RouterConfig::default());
        for (size, total, expected) in [(1, 1024, 1024), (16, 1024, 64), (16, 1031, 64)] {
            for window_key in ["max_num_seqs", "max_running_requests"] {
                let info =
                    ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
                        server_args: Some(prost_types::Struct {
                            fields: [("dp_size", size), (window_key, total)]
                                .into_iter()
                                .map(|(key, value)| {
                                    (
                                        key.to_string(),
                                        prost_types::Value {
                                            kind: Some(prost_types::value::Kind::NumberValue(
                                                value as f64,
                                            )),
                                        },
                                    )
                                })
                                .collect(),
                        }),
                        ..Default::default()
                    }));
                let labels = info.to_labels();
                for (overrides, window) in [
                    (HashMap::new(), expected),
                    (labels_of(&[("max_running_requests", "32")]), 32),
                ] {
                    let mut spec = WorkerSpec::new("grpc://w:30000");
                    spec.labels = overrides;
                    let mut data = create_worker_workflow_data(
                        spec,
                        WorkerRegistrationMode::Upsert,
                        Arc::clone(&host),
                        WorkerJobAsker::Api,
                    );
                    data.worker_kind = Some(WorkerKind::Local);
                    data.connection_mode = Some(ConnectionMode::Grpc);
                    data.dp_info = dp_info_from_labels(&labels).unwrap();
                    data.discovered_labels = labels.clone();
                    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

                    assert_eq!(
                        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
                        StepResult::Success
                    );
                    let workers = ctx.data.actual_workers.unwrap();
                    assert_eq!(workers.len(), size as usize);
                    for worker in workers {
                        assert_eq!(worker.max_running_requests(), Some(window));
                        if window == expected {
                            let window = usize::from(window);
                            assert!(worker.try_admit_pd(window, window));
                            assert!(!worker.try_admit_pd(1, window));
                        }
                    }
                }
            }
        }
    }
}
