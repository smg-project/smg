//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::{
    model_card::ModelCard,
    model_type::ModelType,
    worker::{ConstrainedDecoding, HealthCheckConfig, MmProcessingMode, WorkerSpec, WorkerType},
};
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_dp::DpInfo;
use crate::{
    routers::grpc::{
        is_harmony_card,
        multimodal::{
            mm_settings, pool_processes_media_refs, registration_lacks_media_processing,
            SUPPORTS_VISION_LABEL,
        },
        regular::symphony::{
            check_names, labels_select_nothing, nothing_names_a_table, set_aside_by_the_registry,
            table_label_read, unanswered_reasoning,
        },
        utils::{card_names_with, ModelNames, Names},
        zmq_client::zmq_handshake_address,
    },
    worker::{
        basic::{merge_labels, RuntimeType, CONTEXT_WINDOW_LABELS, WINDOW_LABELS},
        circuit_breaker::CircuitBreakerConfig,
        endpoint::Endpoint,
        metadata::join_base,
        overload::OverloadThresholds,
        resilience::resolve_resilience,
        BasicWorkerBuilder, ConnectionMode, Worker, WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    workflow::data::{WorkerKind, WorkerRegistrationMode, WorkerWorkflowData},
};

/// Step 3: Create worker object(s) with merged configuration + metadata.
pub struct CreateLocalWorkerStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for CreateLocalWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;

        if context.data.registration_mode == WorkerRegistrationMode::CreateOnly
            && app_context
                .worker_registry
                .get_by_url(&config.url)
                .is_some()
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!(
                    "Worker {} already exists",
                    Endpoint::redacted_str(&config.url)
                ),
            });
        }

        // Merge labels: discovered first, then config (config takes precedence;
        // a window set by hand is the worker's only window, see `merge_labels`).
        let mut labels = merge_labels(&context.data.discovered_labels, &config.labels);

        validate_constrained_decoding_label(&labels, &config.url, "create_worker")?;
        let (kv_connector, kv_role, kv_engine_id) = take_kv_transfer_metadata(config, &mut labels);

        let model_id = resolve_model_id(config, &labels);
        // ZMQ EngineCore does not report a served model name over the wire, so a
        // ZMQ worker's model identity must come from config (`--model-path`).
        // Without it the worker would register as UNKNOWN and be unroutable, so
        // fail loudly at registration rather than silently.
        let model_id = if model_id == UNKNOWN_MODEL_ID && *connection_mode == ConnectionMode::Zmq {
            app_context
                .router_config
                .model_path
                .as_deref()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("create_worker"),
                    message: format!(
                        "ZMQ worker {} has no model identity: EngineCore does not report a \
                         served model name, so --model-path (or a model_id label) is required",
                        Endpoint::redacted_str(&config.url)
                    ),
                })?
        } else {
            model_id
        };

        let model_card = build_model_card(
            model_id,
            config,
            &labels,
            &app_context.router_config.model_aliases,
        );

        // A parser override naming a parser the gateway cannot read with would
        // silently ship unparsed output at serve time: fail registration loudly
        // instead (mirrors the fail-fast AppContext applies to the global CLI names).
        validate_parser_overrides(&model_card, &config.url).map_err(|message| {
            WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message,
            }
        })?;

        // Mixed overrides across same-model workers are a misconfiguration
        // (except transiently during rolling upgrades): resolution picks one
        // deterministically, but only one family parses correctly. Warn, don't
        // fail — failing would block rolling upgrades that change the parser.
        warn_on_conflicting_parser_overrides(
            &model_card,
            &config.url,
            &app_context.worker_registry,
        );

        // The registry's table reads a model it knows for a worker whose labels
        // select no reader, whatever the gateway's flags name
        // (`regular::symphony::names::reads`): the model's first such worker
        // says so.
        let flags = Names {
            tool: app_context.router_config.tool_call_parser.clone(),
            reasoning: app_context.router_config.reasoning_parser.clone(),
        };
        if let Some(message) = registry_reads_over_the_flags(
            &model_card,
            &config.url,
            &flags,
            &app_context.worker_registry,
        ) {
            warn!(model = %model_card.id, worker = %Endpoint::redacted_str(&config.url), "{message}");
        }
        // A `reasoning_parser` label that is an older name no table answers to
        // passed the override check and selects nothing: the model's first
        // such worker says so, as startup does for the flags.
        if let Some(message) =
            unanswered_reasoning_label(&model_card, &config.url, &app_context.worker_registry)
        {
            warn!(model = %model_card.id, worker = %Endpoint::redacted_str(&config.url), "{message}");
        }
        // Nothing names a table for the model (no label, no entry of the
        // registry, no flag): `plain` reads it, where the older parsers
        // matched such an id by name. The model's first gRPC or ZMQ worker
        // says so.
        if let Some(message) = plain_reads_a_model_nothing_names(
            &model_card,
            &config.url,
            *connection_mode,
            &flags,
            &app_context.worker_registry,
        ) {
            warn!(model = %model_card.id, worker = %Endpoint::redacted_str(&config.url), "{message}");
        }
        let runtime_type = match context.data.detected_runtime_type.as_deref() {
            Some(s) => s.parse::<RuntimeType>().unwrap_or(config.runtime_type),
            None => config.runtime_type,
        };

        // If runtime is still Unspecified after detection, fall back to Sglang
        // (the most common local backend). This preserves the old default behavior
        // where Sglang was the default RuntimeType.
        let runtime_type = if runtime_type.is_specified() {
            runtime_type
        } else {
            debug!(
                "Runtime type unresolved for {} after detection; defaulting to sglang",
                Endpoint::redacted_str(&config.url)
            );
            RuntimeType::Sglang
        };

        validate_overload_overrides(config).map_err(|message| WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message,
        })?;

        validate_zmq_handshake_override(config, *connection_mode).map_err(|message| {
            WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message,
            }
        })?;

        // Only vLLM EngineCore, TokenSpeed and SGLang speak the ZMQ direct-backend
        // wire. Fail registration here rather than letting the connect-time
        // rejection strand the worker in Pending.
        if *connection_mode == ConnectionMode::Zmq
            && !matches!(
                runtime_type,
                RuntimeType::Vllm | RuntimeType::TokenSpeed | RuntimeType::Sglang
            )
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!(
                    "ZMQ worker {} has unsupported runtime {}: only vllm, tokenspeed and \
                     sglang are supported over the ZMQ direct backend",
                    Endpoint::redacted_str(&config.url),
                    runtime_type
                ),
            });
        }

        validate_zmq_worker_type(
            *connection_mode,
            runtime_type,
            config.worker_type,
            &config.url,
        )?;

        // A grouped ZMQ worker (`dp_size: N` on the spec) awaits N engines on
        // one socket set. Every ZMQ runtime routes per rank: vLLM by in-request
        // DP rank, TokenSpeed and SGLang by per-rank socket identity with the
        // producing rank named on each output batch.
        let zmq_engine_group = config
            .dp_size
            .filter(|&n| n > 1 && *connection_mode == ConnectionMode::Zmq);

        // Normalize URL
        let url = normalize_url(&config.url, *connection_mode);

        validate_zmq_handshake_address(
            &app_context.worker_registry,
            &url,
            config.zmq_handshake_address.as_deref(),
            *connection_mode,
        )
        .map_err(|message| WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message,
        })?;

        // Build workers — resolve per-worker resilience and HTTP client
        let base_retry = app_context.router_config.effective_retry_config();
        let base_cb_cfg = app_context.router_config.effective_circuit_breaker_config();
        let base_cb = CircuitBreakerConfig {
            failure_threshold: base_cb_cfg.failure_threshold,
            success_threshold: base_cb_cfg.success_threshold,
            timeout_duration: Duration::from_secs(base_cb_cfg.timeout_duration_secs),
            window_duration: Duration::from_secs(base_cb_cfg.window_duration_secs),
        };

        let (resolved_resilience, circuit_breaker) = resolve_resilience(
            &base_retry,
            &base_cb,
            !app_context.router_config.disable_retries,
            !app_context.router_config.disable_circuit_breaker,
            &config.resilience,
        );

        let http_client = context.data.http_client("create_worker")?;
        let http2 = context.data.http2.unwrap_or(false);

        let overload_defaults = OverloadThresholds::from_gateway_config(&app_context.router_config);

        let health_base = app_context.router_config.health_check.to_protocol_config();
        let health_config =
            resolve_zmq_health_config(config.health.apply_to(&health_base), *connection_mode, &url);
        let health_endpoint = &app_context.router_config.health_check.endpoint;

        let dp_ranks: Vec<Option<(usize, usize)>> = if app_context.router_config.dp_aware {
            dp_rank_expansion(context.data.dp_info.as_ref(), *connection_mode, &config.url)?
        } else {
            vec![None] // single worker, no DP
        };

        // The failure classes of the table that reads the model exist at zero
        // from the registration of a worker the pipeline reads with, so a
        // dashboard finds `failed` and `malformed` before the first such
        // output. The table is resolved here, where the card and the
        // names are, and published by the registration step once the worker
        // is in the registry, so a registration refused there leaves no
        // series behind.
        context.data.symphony_table = symphony_table_of(&model_card, *connection_mode, &flags);

        // The worker keeps the spec it was registered with, so every field
        // the API accepted rides along: provider, kv_block_size, the
        // resilience overrides (which set the per-model retry config at
        // registration), the pairing protocol, the load-monitor interval,
        // the HTTP pool and multimodal transport overrides. What this step
        // resolves is written over it: the normalized URL, the model card,
        // the merged labels with the KV transfer keys taken out, the KV
        // transfer fields as resolved, and the DP placement per rank.
        let mut spec = config.clone();
        spec.url.clone_from(&url);
        spec.dp_base_url = None;
        spec.dp_rank = None;
        spec.dp_size = None;
        spec.kv_connector = None;
        spec.kv_role = None;
        spec.kv_engine_id = None;

        let workers: Vec<Arc<dyn Worker>> = dp_ranks
            .into_iter()
            .map(|dp| {
                let mut builder = BasicWorkerBuilder::from_spec(spec.clone())
                    .model(model_card.clone())
                    .connection_mode(*connection_mode)
                    .runtime_type(runtime_type)
                    .circuit_breaker_config(circuit_breaker.clone())
                    .http_client(http_client.clone())
                    .http2(http2)
                    .resilience(resolved_resilience.clone())
                    .health_config(health_config.clone())
                    .health_endpoint(health_endpoint)
                    .zmq_startup_timeout(Duration::from_secs(
                        app_context.router_config.zmq_startup_timeout_secs,
                    ))
                    // The request bound of the HTTP worker client, for the
                    // ZMQ link: a stream the engine leaves silent ends at it.
                    .zmq_request_timeout(zmq_request_timeout(
                        &spec,
                        app_context.router_config.request_timeout_secs,
                    ))
                    .overload_defaults(overload_defaults)
                    .labels(labels_for_rank(&labels, dp, &config.labels))
                    // The engine's report in this rank's space, sized by the
                    // width the worker's own labels use, minus the keys the
                    // operator's config sets: those are the operator's whatever
                    // value they carry (the same value as the engine's
                    // included), so a later refresh leaves them alone.
                    .discovered_labels(
                        report_for_rank(&context.data.discovered_labels, dp, &labels)
                            .into_iter()
                            .filter(|(key, _)| !config.labels.contains_key(key))
                            .collect(),
                    );

                if let Some((rank, size)) = dp {
                    builder = builder.dp_config(rank, size);
                }
                if let Some(ref c) = kv_connector {
                    builder = builder.kv_connector(c);
                }
                if let Some(ref r) = kv_role {
                    builder = builder.kv_role(r);
                }
                if let Some(ref e) = kv_engine_id {
                    builder = builder.kv_engine_id(e);
                }
                if let Some(ref address) = config.zmq_handshake_address {
                    builder = builder.zmq_handshake_address(address.clone());
                }
                if let Some(group_size) = zmq_engine_group {
                    builder = builder.zmq_engine_group(group_size);
                }
                // Promotion can be event-driven: a ZMQ worker signals the
                // manager the instant its handshake completes, and any worker
                // demoted by health is signalled on its first successful
                // contact (see `worker::liveness`), so every worker gets the
                // registry's connect signal.
                builder =
                    builder.connect_signal_tx(app_context.worker_registry.connect_signal_sender());

                // Builder sets initial status: Pending if health-checked, Ready if not.
                Arc::new(builder.build()) as Arc<dyn Worker>
            })
            .collect();

        // A registration that worker-side media placement cannot use is said
        // once, here, with the worker's address: an engine with a vision
        // encoder that does not process media references never receives a
        // media reference under `--mm-processing worker` (a request with
        // per-item hints it still serves, through the router's pipeline). The
        // line says what follows for the model, read from its pool (the
        // registered workers and the ones registering with it): with a capable
        // worker beside this one the references are served (selection picks
        // that worker); without one the model's requests that forward media
        // references are refused (`no_media_ref_capable_worker`) and
        // `/readiness` names the model until such a worker registers.
        // The registration itself is not refused. A model on the
        // encode-prefill-decode pipeline (an encode worker among its
        // registered workers or these) is not warned about: references never
        // travel on it, and selection answers with the fix that applies there.
        if mm_settings().processing.value == MmProcessingMode::Worker {
            for worker in &workers {
                let registered = app_context.worker_registry.get_by_model(worker.model_id());
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

        debug!(
            "Created {} worker(s) for {} ({:?}, {} labels)",
            workers.len(),
            Endpoint::redacted_str(&url),
            connection_mode,
            labels.len()
        );

        context.data.actual_workers = Some(workers);
        context.data.final_labels = labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// A `constrained_decoding` label, configured, discovered or sent in a label
/// update, must name a policy; a worker that would silently fall back to
/// `requested` on a typo fails the `step` instead. The typed spec field needs
/// no such check.
pub(super) fn validate_constrained_decoding_label(
    labels: &HashMap<String, String>,
    url: &str,
    step: &str,
) -> WorkflowResult<()> {
    match labels.get(ConstrainedDecoding::LABEL) {
        Some(value) => {
            ConstrainedDecoding::parse(value)
                .map(drop)
                .map_err(|why| WorkflowError::StepFailed {
                    step_id: StepId::new(step),
                    message: format!("Worker {}: {why}", Endpoint::redacted_str(url)),
                })
        }
        None => Ok(()),
    }
}

fn take_kv_transfer_metadata(
    config: &WorkerSpec,
    labels: &mut HashMap<String, String>,
) -> (Option<String>, Option<String>, Option<String>) {
    let connector_label = labels.remove("kv_connector");
    let role_label = labels.remove("kv_role");
    let engine_id_label = labels.remove("kv_engine_id");
    (
        config
            .kv_connector
            .clone()
            .filter(|s| !s.is_empty())
            .or(connector_label),
        config
            .kv_role
            .clone()
            .filter(|s| !s.is_empty())
            .or(role_label),
        config
            .kv_engine_id
            .clone()
            .filter(|s| !s.is_empty())
            .or(engine_id_label)
            .filter(|s| !s.is_empty()),
    )
}

/// Resolve the canonical model ID before aliases are applied.
///
/// Kubernetes service discovery creates a spec without model cards, so its
/// canonical ID comes from the backend's `served_model_name`. Router aliases
/// are deliberately absent from this function and cannot change discovery.
pub(super) fn resolve_model_id<'a>(
    config: &'a WorkerSpec,
    labels: &'a HashMap<String, String>,
) -> &'a str {
    config
        .models
        .primary()
        .map(|model| model.id.as_str())
        .or_else(|| labels.get("served_model_name").map(String::as_str))
        .or_else(|| labels.get("model_id").map(String::as_str))
        .or_else(|| labels.get("model_path").map(String::as_str))
        .unwrap_or(UNKNOWN_MODEL_ID)
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
                    tracing::warn!(
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

/// Reject parser-override names the gateway cannot read with: a name that is no Symphony table's,
/// none a table answers to, and not the response template's (`regular::symphony::check_names`).
pub(super) fn validate_parser_overrides(card: &ModelCard, worker_url: &str) -> Result<(), String> {
    let names = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    check_names(&names).map_err(|why| {
        format!(
            "worker {} declares a parser override for model '{}' the gateway cannot \
             read with: {why}",
            Endpoint::redacted_str(worker_url),
            card.id
        )
    })
}

/// The table whose failure classes the registration step publishes at zero for a gRPC or ZMQ
/// worker of `card`'s model, as this worker's labels and the gateway's flags select it, so
/// `failed`, `malformed` and the rest exist before the first such output
/// (`regular::symphony::telemetry::init_table_series`, called once the worker is in the registry).
/// Every worker of a model resolves one (a later worker whose labels select another table brings
/// that table's series; publishing a table again sets nothing). An HTTP worker resolves none: only
/// the gRPC and ZMQ pipeline reads with Symphony.
fn symphony_table_of(
    card: &ModelCard,
    connection_mode: ConnectionMode,
    flags: &Names,
) -> Option<&'static str> {
    if connection_mode == ConnectionMode::Http {
        return None;
    }
    let labels = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    Some(table_label_read(&ModelNames { labels, flags }, &card.id))
}

/// The word for a worker whose `reasoning_parser` label is an older parser's name no table
/// answers to (`regular::symphony::unanswered_reasoning`): the label passed the override check and
/// selects nothing, so the tool-call label's table, the registry's table or `plain` reads the
/// model. `Some(message)` for the model's first worker, naming the worker, the model, the name and
/// the reason; `None` for the model's later workers and for any other label.
fn unanswered_reasoning_label(
    card: &ModelCard,
    worker_url: &str,
    registry: &WorkerRegistry,
) -> Option<String> {
    if !registry.get_by_model(&card.id).is_empty() {
        return None;
    }
    let labels = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    let (name, reason) = unanswered_reasoning(&labels)?;
    Some(format!(
        "worker {} labels model '{}' with the reasoning parser name '{name}', an older \
         parser's that no Symphony table answers to, so it reads nothing ({reason}); the tool-call \
         label's table, the registry's table or plain reads the model",
        Endpoint::redacted_str(worker_url),
        card.id
    ))
}

/// The registry's table reads a model it knows for every worker whose labels select no reader
/// (none, or a reasoning name alone that no table answers to), whatever the gateway's
/// `--tool-call-parser` and `--reasoning-parser` name (`regular::symphony::names::reads`). When
/// they name another reader, the registration of the model's first worker says so: `Some(message)`
/// while no worker of the model is registered, naming the model, both readers and the one used;
/// `None` for the model's later workers, a worker whose labels select a reader, a model the
/// registry does not know, and flags that select nothing or the registry's own table. Workers of
/// one model registering at the same moment may each say it, as the override check above may: the
/// registry gains the worker a step later.
fn registry_reads_over_the_flags(
    card: &ModelCard,
    worker_url: &str,
    flags: &Names,
    registry: &WorkerRegistry,
) -> Option<String> {
    let labels = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    if !labels_select_nothing(&labels) || !registry.get_by_model(&card.id).is_empty() {
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

/// `plain` reads a model nothing names a table for (`regular::symphony::nothing_names_a_table`:
/// its labels select none, the registry does not list its id, the gateway's flags select none),
/// so its tool calls and its thought reach the client as content. The older parsers matched such
/// an id by name (a quantized or renamed checkpoint of a family they read), so the registration
/// of the model's first gRPC or ZMQ worker says so: `Some(message)` while no gRPC or ZMQ worker of
/// the model is registered (an HTTP worker of it is no earlier reader), naming the model, the
/// worker and what selects a table; `None` for the model's later gRPC and ZMQ workers, a model a
/// label, the registry or a flag gives a table, and an HTTP worker, whose engine parses its own
/// output; a label is one any card of the model carries, an HTTP worker's too, as a request
/// reads them ([`card_names_with`]). `None` as well for a model no table reads: a Harmony
/// model ([`is_harmony_card`]), whose family has no table and stays on the Harmony pipeline, and
/// a model that is no LLM (an embedding or a rerank model), which generates no text.
fn plain_reads_a_model_nothing_names(
    card: &ModelCard,
    worker_url: &str,
    connection_mode: ConnectionMode,
    flags: &Names,
    registry: &WorkerRegistry,
) -> Option<String> {
    let no_table_reads_it = connection_mode == ConnectionMode::Http
        || !card.is_llm()
        || is_harmony_card(&card.architectures, card.hf_model_type.as_deref(), &card.id);
    // An HTTP worker's engine parses its own output: only a gRPC or ZMQ worker of the model is
    // an earlier worker a table read.
    let workers = registry.get_by_model(&card.id);
    let read_before = workers
        .iter()
        .any(|worker| *worker.connection_mode() != ConnectionMode::Http);
    if no_table_reads_it || read_before {
        return None;
    }
    // A request reads the names among every card of the model, an HTTP worker's too
    // (`ParserResolver::names_among`): so does the word, with this worker's card among them.
    let labels = card_names_with(&workers, card);
    if !nothing_names_a_table(&ModelNames { labels, flags }, &card.id) {
        return None;
    }
    Some(format!(
        "no Symphony table is named for model '{}': the registry does not list the id, and \
         neither the labels of worker {} nor --tool-call-parser / --reasoning-parser name one, so \
         its output is read as plain content, tool calls and reasoning included; a tool_parser or \
         reasoning_parser label on the model's workers, or the two flags, select its table",
        card.id,
        Endpoint::redacted_str(worker_url),
    ))
}

/// The card a worker is registered with: the operator's card for `model_id`,
/// if the spec has one, completed from the labels, else one built from the
/// labels alone; with the router's aliases for the model attached.
pub(super) fn build_model_card(
    model_id: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
    model_aliases: &HashMap<String, String>,
) -> ModelCard {
    let mut card = card_from_labels(config.models.find(model_id), model_id, labels);

    // Router-level alias map (`--model-alias alias=canonical`): attach every
    // alias that names this canonical model. This is the only alias entry
    // point for automatically registered workers (startup URLs, Kubernetes
    // service discovery) — the backend reports a single served model name, so
    // it can never declare aliases itself. A user-provided card keeps its own
    // aliases; duplicates are skipped so re-registration stays idempotent.
    for (alias, canonical) in model_aliases {
        if canonical == &card.id && alias != &card.id && !card.aliases.contains(alias) {
            card.aliases.push(alias.clone());
        }
    }

    card
}

/// The fields of a worker's model card that come from its labels, derived
/// again for the labels it has now. A field whose value is what the labels
/// before gave it (or that no label gave) follows the labels now, the way
/// registration derived it; one the operator set by hand, on the card or over
/// a label, stands. The id is the worker's identity in the registry and
/// stays. Also the names of the fields that changed.
pub(super) fn model_card_for_labels(
    card: &ModelCard,
    labels_before: &HashMap<String, String>,
    labels_now: &HashMap<String, String>,
) -> (ModelCard, Vec<&'static str>) {
    let before = card_from_labels(None, &card.id, labels_before);
    let now = card_from_labels(None, &card.id, labels_now);
    let mut next = card.clone();
    let mut changed = Vec::new();
    macro_rules! follow {
        ($($field:ident),+ $(,)?) => {$(
            if next.$field == before.$field && next.$field != now.$field {
                next.$field = now.$field;
                changed.push(stringify!($field));
            }
        )+};
    }
    follow!(
        model_type,
        hf_model_type,
        architectures,
        id2label,
        num_labels,
        context_length,
        tokenizer_path,
        tool_parser,
        reasoning_parser,
    );
    (next, changed)
}

/// `operator_card` completed from the labels, or a card from the labels alone.
fn card_from_labels(
    operator_card: Option<&ModelCard>,
    model_id: &str,
    labels: &HashMap<String, String>,
) -> ModelCard {
    let user_provided = operator_card.is_some();
    let mut card = operator_card
        .cloned()
        .unwrap_or_else(|| ModelCard::new(model_id));

    if let Some(mt) = labels.get("model_type") {
        card = card.with_hf_model_type(mt.clone());
    }
    if let Some(archs_json) = labels.get("architectures") {
        if let Ok(archs) = serde_json::from_str::<Vec<String>>(archs_json) {
            card = card.with_architectures(archs);
        }
    }

    // Classification model id2label
    if let Some(json) = labels.get("id2label_json").filter(|s| !s.is_empty()) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(json) {
            let id2label: HashMap<u32, String> = map
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|i| (i, v)))
                .collect();
            if !id2label.is_empty() {
                card = card.with_id2label(id2label);
            }
        }
    } else if let Some(n) = labels
        .get("num_labels")
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
    {
        let id2label: HashMap<u32, String> = (0..n).map(|i| (i, format!("LABEL_{i}"))).collect();
        card = card.with_id2label(id2label);
    }

    // Fill context_length from whichever backend key is available
    if card.context_length.is_none() {
        card.context_length = CONTEXT_WINDOW_LABELS
            .iter()
            .find_map(|k| labels.get(*k))
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| n > 0);
    }

    // Fill tokenizer_path
    if card.tokenizer_path.is_none() {
        card.tokenizer_path = labels
            .get("tokenizer_path")
            .filter(|s| !s.is_empty())
            .cloned();
    }

    // Per-model parser overrides: `tool_parser` / `reasoning_parser` labels
    // (from backend metadata or WorkerSpec.labels) pin the parser for this
    // model, overriding the process-wide `--tool-call-parser` /
    // `--reasoning-parser` names on the gRPC serving path. An explicit card
    // keeps its own values (same precedence as tokenizer_path above).
    if card.tool_parser.is_none() {
        card.tool_parser = labels.get("tool_parser").filter(|s| !s.is_empty()).cloned();
    }
    if card.reasoning_parser.is_none() {
        card.reasoning_parser = labels
            .get("reasoning_parser")
            .filter(|s| !s.is_empty())
            .cloned();
    }

    // Infer model_type capabilities from discovered signals
    let has_vision = labels
        .get(SUPPORTS_VISION_LABEL)
        .or_else(|| labels.get("has_image_understanding"))
        .map(|s| s == "true")
        .unwrap_or(false);

    if !user_provided {
        let is_embedding = labels.get("is_embedding").is_some_and(|s| s == "true");
        let is_non_generation = labels.get("is_generation").is_some_and(|s| s == "false");

        if is_embedding || is_non_generation {
            card.model_type = infer_non_generation_type(labels);
        } else if has_vision && !card.model_type.supports_vision() {
            card.model_type |= ModelType::VISION;
        }
    } else if has_vision && !card.model_type.supports_vision() {
        card.model_type |= ModelType::VISION;
    }

    card
}

/// Determine embedding vs rerank from architecture/model_type hints.
fn infer_non_generation_type(labels: &HashMap<String, String>) -> ModelType {
    if let Some(archs_json) = labels.get("architectures") {
        if let Ok(archs) = serde_json::from_str::<Vec<String>>(archs_json) {
            let joined = archs.join(" ").to_lowercase();
            if joined.contains("rerank") || joined.contains("crossencoder") {
                return ModelType::RERANK;
            }
        }
    }
    if let Some(mt) = labels.get("model_type") {
        if mt.to_lowercase().contains("rerank") {
            return ModelType::RERANK;
        }
    }
    ModelType::EMBEDDINGS
}

/// Reject degenerate per-worker overload thresholds at registration — same
/// rules and wording as `ConfigValidator` applies to the gateway-level
/// `worker_overload_*` fields. Both comparisons are inclusive, so the excluded
/// ends of these ranges would mark this worker overloaded unconditionally.
fn validate_overload_overrides(config: &WorkerSpec) -> Result<(), String> {
    if config.overload.waiting_requests == Some(0) {
        return Err(format!(
            "worker {} sets overload.waiting_requests to 0: Must be >= 1 \
             (0 would mark every worker overloaded)",
            Endpoint::redacted_str(&config.url)
        ));
    }
    if let Some(threshold) = config.overload.token_usage {
        if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
            return Err(format!(
                "worker {} sets overload.token_usage to {threshold}: Must be a \
                 fraction in (0.0, 1.0]",
                Endpoint::redacted_str(&config.url)
            ));
        }
    }
    Ok(())
}

/// The bound on a request over a ZMQ worker's link: the worker's own
/// `http_pool.timeout_secs` when it sets one, else the router's
/// `request_timeout_secs` (`--request-timeout-secs`), the way the HTTP worker
/// client resolves its per-request timeout, so a worker registered with a
/// longer bound for its long generations keeps it on either link.
fn zmq_request_timeout(spec: &WorkerSpec, router_request_timeout_secs: u64) -> Duration {
    Duration::from_secs(
        spec.http_pool
            .timeout_secs
            .unwrap_or(router_request_timeout_secs),
    )
}

/// `zmq_handshake_address` only steers the ZMQ handshake bind; on any other
/// connection mode it would be silently ignored, so reject the registration
/// loudly instead.
fn validate_zmq_handshake_override(
    config: &WorkerSpec,
    connection_mode: ConnectionMode,
) -> Result<(), String> {
    if config.zmq_handshake_address.is_some() && connection_mode != ConnectionMode::Zmq {
        return Err(format!(
            "worker {} sets zmq_handshake_address but its connection mode is \
             {connection_mode:?}: the field is only meaningful for ZMQ workers",
            Endpoint::redacted_str(&config.url)
        ));
    }
    Ok(())
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

/// Keep health checking on for a ZMQ worker.
///
/// The ZMQ probe is not a network call — it peeks a local liveness flag — and
/// it is the only path that evicts a client whose engine died and rebinds the
/// sockets for a replacement engine to dial into (ZMQ liveness is latched: a
/// dead client never recovers in place). With probing off the first handshake
/// still runs, since a request kicks the background connect driver, but an
/// engine restart leaves the worker pinned to that dead client until the
/// gateway restarts. Honor the transport over the flag and say so.
fn resolve_zmq_health_config(
    health_config: HealthCheckConfig,
    connection_mode: ConnectionMode,
    url: &str,
) -> HealthCheckConfig {
    if connection_mode != ConnectionMode::Zmq || !health_config.disable_health_check {
        return health_config;
    }
    warn!(
        "Ignoring disabled health checks for ZMQ worker {}: the ZMQ probe is a local liveness \
         check and the only path that reconnects a restarted engine, so it stays enabled",
        Endpoint::redacted_str(url)
    );
    HealthCheckConfig {
        disable_health_check: false,
        ..health_config
    }
}

fn normalize_url(url: &str, connection_mode: ConnectionMode) -> String {
    if url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("grpc://")
        || url.starts_with("grpcs://")
        || url.starts_with("ipc://")
        || url.starts_with("tcp://")
    {
        url.to_string()
    } else {
        match connection_mode {
            ConnectionMode::Http => format!("http://{url}"),
            ConnectionMode::Grpc => format!("grpc://{url}"),
            ConnectionMode::Zmq => format!("ipc://{url}"),
        }
    }
}

/// Reject a disaggregated worker the ZMQ path cannot serve.
///
/// TokenSpeed's wire carries the prefill/decode rendezvous on its generate
/// request (the bootstrap slots) and takes an encode server's encode request,
/// so a TokenSpeed ZMQ worker serves every role. vLLM's `EngineCoreRequest`
/// has no field for the bootstrap info PD injects, and the other runtimes'
/// encode dispatch is a gRPC encoder RPC, so for them `Regular` is the only
/// role the ZMQ lane serves. A worker of a role its lane cannot serve is
/// rejected at registration rather than registered and never selected.
fn validate_zmq_worker_type(
    connection_mode: ConnectionMode,
    runtime_type: RuntimeType,
    worker_type: WorkerType,
    url: &str,
) -> Result<(), WorkflowError> {
    if connection_mode != ConnectionMode::Zmq {
        return Ok(());
    }
    let reason = match worker_type {
        WorkerType::Regular => return Ok(()),
        _ if runtime_type == RuntimeType::TokenSpeed => return Ok(()),
        WorkerType::Prefill | WorkerType::Decode => format!(
            "the {runtime_type} direct-ZMQ backend carries no KV-transfer metadata, so \
             prefill/decode disaggregation requires a gRPC worker or a TokenSpeed engine"
        ),
        WorkerType::Encode => format!(
            "the {runtime_type} direct-ZMQ backend has no encode path, so encode \
             disaggregation requires a gRPC worker or a TokenSpeed encode server"
        ),
    };
    Err(WorkflowError::StepFailed {
        step_id: StepId::new("create_worker"),
        message: format!(
            "ZMQ worker {} cannot serve worker type {worker_type}: {reason}",
            Endpoint::redacted_str(url)
        ),
    })
}

/// The labels of one dp-aware rank worker: the engine's labels as they are,
/// except the running window of an engine that splits it across its
/// attention-DP workers (the `enable_dp_attention` / `attn_dp_size` labels:
/// the flag and the servicer's computed figure alike name the whole engine's
/// window), which becomes the rank's share, so the PD admission gate and the
/// capacity sum bound each rank worker by what its rank runs instead of the
/// whole engine's window `size` times over. Engines whose figure is per rank
/// already (e.g. vLLM's `max_num_seqs`) carry no such label and keep theirs; an
/// unexpanded worker stands for the whole engine and keeps the whole window.
/// A window the operator set by hand on the worker (`set_by_hand`, the
/// spec's own labels, which win the merge) is the operator's statement
/// about each rank worker and stays as set (`merge_labels` already made it
/// the worker's only window); only a discovered window is shared. The
/// discovered record and a refresh's fresh labels pass through the same rule
/// (with no operator labels), so a refresh compares like with like in rank
/// space.
pub(super) fn labels_for_rank(
    labels: &HashMap<String, String>,
    dp: Option<(usize, usize)>,
    set_by_hand: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut labels = labels.clone();
    let Some((_, size)) = dp.filter(|(_, size)| *size > 1) else {
        return labels;
    };
    let Some(width) = attention_dp_width(&labels, size) else {
        return labels;
    };
    // A hand-set window is the worker's only window (the merge dropped the
    // discovered one): nothing to share.
    if WINDOW_LABELS
        .iter()
        .any(|key| set_by_hand.contains_key(*key))
    {
        return labels;
    }
    for key in WINDOW_LABELS {
        let Some(window) = labels
            .get(key)
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|window| *window > 0)
        else {
            continue;
        };
        let share = (window / width).max(1);
        labels.insert(key.to_string(), share.to_string());
    }
    labels
}

/// The engine's report `report` in a rank's space, with the attention-DP
/// width read from `width_source` rather than from the report itself. A
/// worker's labels and the record it keeps of the engine's share must agree
/// on the width, and the worker's can come from the operator: a DP flag set
/// in the config over an engine that does not report one, or reports it
/// differently. The width labels themselves stay as the report has them.
pub(super) fn report_for_rank(
    report: &HashMap<String, String>,
    dp: Option<(usize, usize)>,
    width_source: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut sized = report.clone();
    for key in DP_WIDTH_LABELS {
        match width_source.get(key) {
            Some(value) => sized.insert(key.to_string(), value.clone()),
            None => sized.remove(key),
        };
    }
    let mut mapped = labels_for_rank(&sized, dp, &HashMap::new());
    for key in DP_WIDTH_LABELS {
        match report.get(key) {
            Some(value) => mapped.insert(key.to_string(), value.clone()),
            None => mapped.remove(key),
        };
    }
    mapped
}

/// The labels that size a rank's share of the engine's window
/// (`attention_dp_width`).
pub(super) const DP_WIDTH_LABELS: [&str; 2] = ["attn_dp_size", "enable_dp_attention"];

/// How many attention-DP workers the engine's labels say its running window
/// is split across, `None` for an engine that runs the whole window on every
/// rank: `attn_dp_size` above one (the engine's newer releases resolve the
/// flag into it, and `dp_size` then counts replicas, each running the whole
/// window, so the dp-aware width of `dp_size x attn_dp_size` ranks is not the
/// divisor), else the dp-aware width `size` under `enable_dp_attention` (the
/// pinned release, where `dp_size` is the attention-DP width).
fn attention_dp_width(labels: &HashMap<String, String>, size: usize) -> Option<u64> {
    let resolved = labels
        .get("attn_dp_size")
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|width| *width > 1);
    if resolved.is_some() {
        return resolved;
    }
    labels
        .get("enable_dp_attention")
        .is_some_and(|value| value == "true")
        .then_some(size as u64)
}

/// Ranks to register for a dp-aware worker. A width above one expands into
/// one dp-annotated worker per rank. A width-1 advertisement (older
/// servicers emit it as a capability signal) and a missing one both
/// register a plain worker: there is no placement to choose, and pinning
/// rank 0 anyway still alters engine-side scheduling. discover_dp already
/// logged the degradation for the missing-label case.
fn dp_rank_expansion(
    dp_info: Option<&DpInfo>,
    connection_mode: ConnectionMode,
    url: &str,
) -> Result<Vec<Option<(usize, usize)>>, WorkflowError> {
    match dp_info {
        Some(dp_info) if dp_info.dp_size > 1 => {
            validate_zmq_dp(connection_mode, dp_info.dp_size, url)?;
            Ok((0..dp_info.dp_size)
                .map(|r| Some((r, dp_info.dp_size)))
                .collect())
        }
        Some(_) | None => Ok(vec![None]),
    }
}

/// Reject a data-parallel worker the ZMQ path cannot serve.
///
/// dp-aware expansion creates one rank-pinned worker per DP rank, but each
/// ZMQ worker owns its own socket bind — N expanded workers would fight over
/// the same ipc paths. ZMQ DP runs as a *grouped* worker instead (`dp_size: N`
/// on the worker spec: one worker, one socket set, N engines dialing in), so
/// reject the expansion loudly and point at the grouped form. gRPC/HTTP
/// expansion and single-engine ZMQ are fine.
fn validate_zmq_dp(
    connection_mode: ConnectionMode,
    dp_size: usize,
    url: &str,
) -> Result<(), WorkflowError> {
    if connection_mode == ConnectionMode::Zmq && dp_size > 1 {
        return Err(WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message: format!(
                "ZMQ worker {} cannot be dp-aware expanded (dp_size={dp_size}); \
                 configure a grouped ZMQ worker (`dp_size` on the worker spec) instead",
                Endpoint::redacted_str(url)
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::worker::test_support::RefusedAddress;

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

    /// A gRPC worker's registration resolves the table whose failure classes are published at
    /// zero: the registry's table for an unlabeled worker of a model it knows, and another table
    /// for a worker whose labels name one. An HTTP worker (the pipeline that reads with Symphony
    /// never serves it) resolves none.
    #[test]
    fn a_workers_registration_resolves_the_table_whose_failure_classes_are_published() {
        let known = ModelCard::new("Qwen/Qwen3-8B");
        let table = |card: &ModelCard, connection_mode| {
            symphony_table_of(card, connection_mode, &Names::default())
        };
        assert_eq!(
            table(&known, ConnectionMode::Http),
            None,
            "an HTTP worker publishes no table"
        );
        assert_eq!(table(&known, ConnectionMode::Grpc), Some("qwen3"));
        // A later worker of the model whose labels name another table brings
        // that table's series: requests count under it from then on.
        let mut labeled = known.clone();
        labeled.tool_parser = Some("kimi_k2".to_string());
        assert_eq!(table(&labeled, ConnectionMode::Zmq), Some("kimi_k2"));
    }

    /// The table a worker's registration resolves is published by the registration step, once the
    /// worker is in the registry: a gRPC worker of a model the registry knows resolves the
    /// registry's table in the create step, a registration the registration
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
                crate::config::RouterConfig::builder()
                    .worker_startup_timeout_secs(1)
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
        assert_eq!(ctx.data.symphony_table, Some("qwen3"));
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
        assert_eq!(ctx.data.symphony_table, Some("qwen3"));
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
        app_context: Arc<crate::app_context::AppContext>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        use openai_protocol::worker::WorkerModels;
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("grpc://127.0.0.1:18001");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new(model)));
        let mut data = create_worker_workflow_data(spec, registration_mode, app_context);
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Grpc);
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// The registration of a model's first worker whose `reasoning_parser` label is an older
    /// name no table answers to says that the label reads nothing, with its reason; the model's
    /// next worker, a label a table answers to, and no label say nothing.
    #[test]
    fn an_older_reasoning_label_no_table_answers_to_is_said_once_per_model_at_registration() {
        let mut labeled = ModelCard::new("acme/thinker");
        labeled.reasoning_parser = Some("qwen3_thinking".to_string());
        let registry = WorkerRegistry::new();
        // The worker is named through the mask: a credential in its address
        // is in no message (the helper masks, so no caller can forget).
        let said = unanswered_reasoning_label(&labeled, "grpc://user:secret@a:1", &registry)
            .expect("the model's first worker says so");
        assert!(
            said.contains("'acme/thinker'")
                && said.contains("'qwen3_thinking'")
                && said.contains("***@a:1")
                && !said.contains("secret")
                && said.contains("reads nothing"),
            "{said}"
        );
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://a:1")
                    .model(labeled.clone())
                    .build(),
            ))
            .expect("register");
        assert_eq!(
            unanswered_reasoning_label(&labeled, "grpc://b:1", &registry),
            None,
            "the model's second worker"
        );

        let fresh = WorkerRegistry::new();
        let mut answered = ModelCard::new("acme/thinker");
        answered.reasoning_parser = Some("qwen3".to_string());
        assert_eq!(
            unanswered_reasoning_label(&answered, "grpc://a:1", &fresh),
            None,
            "a label a table answers to"
        );
        assert_eq!(
            unanswered_reasoning_label(&ModelCard::new("acme/thinker"), "grpc://a:1", &fresh),
            None,
            "no label"
        );
    }

    /// The registration of a model's first worker whose labels select no reader (none, or a
    /// reasoning name no table answers to) says that the registry's table reads the model, not
    /// what the gateway's flags name, naming both; the model's next worker, a worker whose labels
    /// select a reader, a model the registry does not know, and flags that name the registry's
    /// own table or nothing, say nothing.
    #[test]
    fn the_registrys_table_over_the_flags_is_said_once_per_model_at_registration() {
        let flags = Names {
            tool: Some("mistral_v13".to_string()),
            reasoning: None,
        };
        let known = ModelCard::new("Qwen/Qwen3-8B");
        let registry = WorkerRegistry::new();
        let said = |card: &ModelCard, url: &str, flags: &Names, registry: &WorkerRegistry| {
            registry_reads_over_the_flags(card, url, flags, registry)
        };

        // The worker is named through the mask: a credential in its address
        // is in no message (the helper masks, so no caller can forget).
        let first = said(&known, "grpc://user:secret@a:1", &flags, &registry)
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
            said(&known, "grpc://b:1", &flags, &registry),
            None,
            "the model's second worker"
        );

        let fresh = WorkerRegistry::new();
        let mut labeled = known.clone();
        labeled.reasoning_parser = Some("qwen3".to_string());
        assert_eq!(
            said(&labeled, "grpc://a:1", &flags, &fresh),
            None,
            "a worker whose labels select a reader"
        );
        assert_eq!(
            said(
                &ModelCard::new("acme/ministral-3-14b-instruct-w4a16"),
                "grpc://a:1",
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
            said(&known, "grpc://a:1", &own, &fresh),
            None,
            "the registry's own table"
        );
        // A label that selects nothing (a reasoning name no table answers to) falls through to
        // the registry as no label does, so the worker says it too.
        let mut idle_label = known.clone();
        idle_label.reasoning_parser = Some("qwen3_thinking".to_string());
        assert!(
            said(&idle_label, "grpc://a:1", &flags, &fresh).is_some_and(|message| {
                message.contains("'qwen3'") && message.contains("whose labels select no reader")
            }),
            "a label that selects nothing"
        );
        assert_eq!(
            said(&known, "grpc://a:1", &Names::default(), &fresh),
            None,
            "no flags"
        );
    }

    /// The registration of the first gRPC or ZMQ worker of a model nothing names a table for (no
    /// label, no entry of the registry, no flag) says that `plain` reads the model, naming the
    /// model, the worker through the mask and what selects a table, and the create step warns
    /// with it. The model's next worker, an HTTP worker, a worker whose label names a table, a
    /// flag that names one and a model the registry lists say nothing, and neither does a model
    /// no table reads: a Harmony model, an embedding model or a rerank model.
    #[traced_test]
    #[tokio::test]
    async fn a_model_nothing_names_a_table_for_is_said_once_per_model_at_registration() {
        let unlisted = ModelCard::new("acme/ministral-3-14b-instruct-w4a16");
        let none = Names::default();
        let registry = WorkerRegistry::new();
        let said = |card: &ModelCard,
                    mode: ConnectionMode,
                    flags: &Names,
                    registry: &WorkerRegistry| {
            plain_reads_a_model_nothing_names(card, "grpc://user:secret@a:1", mode, flags, registry)
        };

        let first = said(&unlisted, ConnectionMode::Grpc, &none, &registry)
            .expect("the model's first worker says so");
        assert!(
            first.contains("'acme/ministral-3-14b-instruct-w4a16'")
                && first.contains("plain content")
                && first.contains("tool_parser")
                && first.contains("--tool-call-parser")
                && first.contains("***@a:1")
                && !first.contains("secret"),
            "{first}"
        );
        assert!(said(&unlisted, ConnectionMode::Zmq, &none, &registry).is_some());
        assert_eq!(
            said(&unlisted, ConnectionMode::Http, &none, &registry),
            None,
            "an HTTP worker's engine parses its own output"
        );
        let mut labeled = unlisted.clone();
        labeled.tool_parser = Some("mistral_v13".to_string());
        assert_eq!(
            said(&labeled, ConnectionMode::Grpc, &none, &registry),
            None,
            "a label names a table"
        );
        let flags = Names {
            tool: Some("mistral_v13".to_string()),
            reasoning: None,
        };
        assert_eq!(
            said(&unlisted, ConnectionMode::Grpc, &flags, &registry),
            None,
            "a flag names a table"
        );
        let listed = ModelCard::new("Qwen/Qwen3-8B");
        assert_eq!(
            said(&listed, ConnectionMode::Grpc, &none, &registry),
            None,
            "a model the registry lists"
        );
        // No table reads these, so nothing is said for them: a Harmony model stays on the
        // Harmony pipeline, and an embedding or a rerank model generates no text.
        let unread = [
            ("a Harmony id", ModelCard::new("acme/gpt-oss-20b-w4a16")),
            (
                "a Harmony architecture",
                unlisted
                    .clone()
                    .with_architectures(vec!["GptOssForCausalLM".to_string()]),
            ),
            (
                "an embedding model",
                unlisted.clone().with_model_type(ModelType::EMBEDDINGS),
            ),
            (
                "a rerank model",
                unlisted.clone().with_model_type(ModelType::RERANK),
            ),
        ];
        let told: Vec<&str> = unread
            .iter()
            .filter(|(_, card)| said(card, ConnectionMode::Grpc, &none, &registry).is_some())
            .map(|(which, _)| *which)
            .collect();
        assert!(told.is_empty(), "said for {told:?}, which no table reads");
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://a:1")
                    .model(unlisted.clone())
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            ))
            .expect("register");
        assert_eq!(
            said(&unlisted, ConnectionMode::Grpc, &none, &registry),
            None,
            "the model's second worker"
        );

        // The create step warns with it for the first gRPC worker of such a model, and says
        // nothing for a model the registry lists.
        let symphony = || {
            app_context_with(
                crate::config::RouterConfig::builder()
                    .worker_startup_timeout_secs(1)
                    .build_unchecked(),
            )
        };
        let mut ctx = grpc_worker_context(&listed.id, WorkerRegistrationMode::Upsert, symphony());
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert!(!logs_contain("no Symphony table is named"));
        let mut ctx = grpc_worker_context(&unlisted.id, WorkerRegistrationMode::Upsert, symphony());
        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        assert_eq!(ctx.data.symphony_table, Some("plain"));
        assert!(logs_contain(
            "no Symphony table is named for model 'acme/ministral-3-14b-instruct-w4a16'"
        ));
    }

    /// A model whose first worker is an HTTP worker is told at its first gRPC or ZMQ worker: an
    /// HTTP worker's engine parses its own output, so no table read the model before, and a
    /// model that moves from HTTP to gRPC while its HTTP workers still serve gets the word.
    #[test]
    fn a_model_whose_first_worker_is_http_is_told_at_its_first_grpc_worker() {
        let unlisted = ModelCard::new("acme/ministral-3-14b-instruct-w4a16");
        let none = Names::default();
        let registry = WorkerRegistry::new();
        let said = |mode: ConnectionMode| {
            plain_reads_a_model_nothing_names(&unlisted, "grpc://b:1", mode, &none, &registry)
        };
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("http://a:1")
                    .model(unlisted.clone())
                    .connection_mode(ConnectionMode::Http)
                    .build(),
            ))
            .expect("register");
        assert!(
            said(ConnectionMode::Grpc).is_some(),
            "an HTTP worker of the model is no earlier reader"
        );
        assert!(said(ConnectionMode::Zmq).is_some());
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://b:1")
                    .model(unlisted.clone())
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            ))
            .expect("register");
        assert_eq!(
            said(ConnectionMode::Grpc),
            None,
            "the model's second gRPC worker"
        );
        assert_eq!(
            said(ConnectionMode::Zmq),
            None,
            "a ZMQ worker after the model's gRPC worker"
        );
    }

    /// A request reads the parser names among every card of the model, whatever its worker's
    /// wire, so a table an HTTP worker's label names is the model's: nothing is said for the
    /// model's unlabelled first gRPC worker, which the label's table reads.
    #[test]
    fn a_table_an_http_workers_label_names_is_the_models_at_its_first_grpc_worker() {
        let none = Names::default();
        let registry = WorkerRegistry::new();
        let labelled = ModelCard::new("acme/ministral-3-8b-instruct-w4a16");
        let mut http_card = labelled.clone();
        http_card.tool_parser = Some("mistral_v13".to_string());
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("http://c:1")
                    .model(http_card)
                    .connection_mode(ConnectionMode::Http)
                    .build(),
            ))
            .expect("register");
        assert_eq!(
            plain_reads_a_model_nothing_names(
                &labelled,
                "grpc://d:1",
                ConnectionMode::Grpc,
                &none,
                &registry
            ),
            None,
            "the HTTP worker's label names the model's table"
        );
    }

    fn make_app_context() -> Arc<crate::app_context::AppContext> {
        app_context_with(
            crate::config::RouterConfig::builder()
                .worker_startup_timeout_secs(1)
                .build_unchecked(),
        )
    }

    fn app_context_with(
        router_config: crate::config::RouterConfig,
    ) -> Arc<crate::app_context::AppContext> {
        use crate::app_context::testing::app_context_for_tests;

        app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()))
    }

    /// A worker registered through the API keeps every spec field the
    /// request carried, not only the ones this step copies by hand: the
    /// provider, the KV block size, the resilience overrides (the per-model
    /// retry config is derived from them at registration), the pairing
    /// protocol and the load-monitor interval all have consumers.
    #[tokio::test]
    async fn api_registered_worker_keeps_the_fields_the_spec_accepted() {
        use openai_protocol::worker::{ProviderType, ResilienceUpdate, WorkerModels};
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("http://worker:8000");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.provider = Some(ProviderType::OpenAI);
        spec.kv_block_size = Some(64);
        spec.pairing_protocol = Some("cluster-blue".to_string());
        spec.load_monitor_interval_secs = Some(3);
        spec.resilience = ResilienceUpdate {
            max_retries: Some(7),
            ..Default::default()
        };
        spec.labels = HashMap::from([
            ("kv_role".to_string(), "kv_consumer".to_string()),
            ("tier".to_string(), "gold".to_string()),
        ]);

        let mut data =
            create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, make_app_context());
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(crate::server::http_client()));
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let worker = &ctx.data.actual_workers.as_ref().unwrap()[0];
        let spec = &worker.metadata().spec;
        assert_eq!(spec.url, "http://worker:8000");
        assert_eq!(spec.provider, Some(ProviderType::OpenAI));
        assert_eq!(spec.kv_block_size, Some(64));
        assert_eq!(spec.pairing_protocol.as_deref(), Some("cluster-blue"));
        assert_eq!(spec.load_monitor_interval_secs, Some(3));
        assert_eq!(spec.resilience.max_retries, Some(7));
        assert_eq!(worker.resilience().retry.max_retries, 7);
        // Resolved by this step rather than copied: the KV transfer keys
        // leave the labels for their own fields, and no DP placement is set.
        assert_eq!(spec.kv_role.as_deref(), Some("kv_consumer"));
        assert!(!spec.labels.contains_key("kv_role"));
        assert_eq!(spec.labels["tier"], "gold");
        assert_eq!(spec.dp_rank, None);
        assert_eq!(spec.dp_size, None);
    }

    /// What a registration records as engine-discovered is the engine's
    /// report minus every key the operator's config sets, whatever value the
    /// config carries: a config label equal to the engine's value is still
    /// the operator's, so a later refresh does not rewrite it as the engine's.
    #[tokio::test]
    async fn a_config_label_equal_to_the_engines_value_is_not_recorded_as_discovered() {
        use openai_protocol::worker::WorkerModels;
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("http://worker:8000");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.labels = HashMap::from([
            ("page_size".to_string(), "256".to_string()),
            ("tier".to_string(), "gold".to_string()),
        ]);

        let mut data =
            create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, make_app_context());
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(crate::server::http_client()));
        data.discovered_labels = HashMap::from([
            ("page_size".to_string(), "256".to_string()),
            ("max_num_seqs".to_string(), "8".to_string()),
        ]);
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let worker = &ctx.data.actual_workers.as_ref().unwrap()[0];
        let labels = &worker.metadata().spec.labels;
        assert_eq!(labels["page_size"], "256");
        assert_eq!(labels["tier"], "gold");
        assert_eq!(labels["max_num_seqs"], "8");
        assert_eq!(
            *worker.metadata().discovered_labels,
            HashMap::from([("max_num_seqs".to_string(), "8".to_string())])
        );
    }

    /// A dp-aware registration of one model under the DP label, as a local
    /// HTTP worker with a client, before the engine's report is set on it.
    fn dp_aware_local_http_data() -> WorkerWorkflowData {
        use openai_protocol::worker::WorkerModels;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("http://worker:8000");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.labels = HashMap::from([("enable_dp_attention".to_string(), "true".to_string())]);
        let router_config = crate::config::RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .dp_aware(true)
            .build_unchecked();

        let mut data = create_worker_workflow_data(
            spec,
            WorkerRegistrationMode::Upsert,
            app_context_with(router_config),
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(crate::server::http_client()));
        data
    }

    /// On a dp-aware registration the engine's report is mapped to the rank's
    /// space before the config's keys leave it: a DP label the config also
    /// names still sizes the window share the record carries, so a later
    /// refresh reads that share, not the whole engine window, as the engine's.
    #[tokio::test]
    async fn a_dp_label_the_config_also_sets_still_sizes_the_recorded_window_share() {
        use wfaas::WorkflowInstanceId;

        let mut data = dp_aware_local_http_data();
        data.discovered_labels = HashMap::from([
            ("enable_dp_attention".to_string(), "true".to_string()),
            ("max_running_requests".to_string(), "256".to_string()),
        ]);
        data.dp_info = Some(DpInfo {
            dp_size: 8,
            model_id: "m".to_string(),
        });
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let workers = ctx.data.actual_workers.as_ref().unwrap();
        assert_eq!(workers.len(), 8);
        for worker in workers {
            let labels = &worker.metadata().spec.labels;
            assert_eq!(labels["enable_dp_attention"], "true");
            assert_eq!(labels["max_running_requests"], "32");
            assert_eq!(
                *worker.metadata().discovered_labels,
                HashMap::from([("max_running_requests".to_string(), "32".to_string())])
            );
        }
    }

    /// The width the worker's labels use sizes the record too when only the
    /// config names it: an operator's DP flag over an engine that reports no
    /// such label still leaves the engine's share, not its whole window, in
    /// the record, so a refresh reads the share as the engine's.
    #[tokio::test]
    async fn a_dp_flag_only_the_config_sets_sizes_the_recorded_window_share_too() {
        use wfaas::WorkflowInstanceId;

        let mut data = dp_aware_local_http_data();
        data.discovered_labels =
            HashMap::from([("max_running_requests".to_string(), "256".to_string())]);
        data.dp_info = Some(DpInfo {
            dp_size: 8,
            model_id: "m".to_string(),
        });
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        assert_eq!(
            CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let workers = ctx.data.actual_workers.as_ref().unwrap();
        assert_eq!(workers.len(), 8);
        for worker in workers {
            assert_eq!(worker.metadata().spec.labels["max_running_requests"], "32");
            assert_eq!(
                *worker.metadata().discovered_labels,
                HashMap::from([("max_running_requests".to_string(), "32".to_string())])
            );
        }
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
        let mut data = create_worker_workflow_data(spec, registration_mode, make_app_context());
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(crate::server::http_client()));
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

        let address = SocketAddr::from(([127, 0, 0, 1], crate::worker::DEFAULT_BOOTSTRAP_PORT));
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

    /// The dedicated `kv_role` field follows the same precedence as the
    /// connector and engine id: it wins over the legacy label, and an empty
    /// value falls back to the label. A `POST /workers` carrying a top-level
    /// `kv_role` and no label used to lose the role here.
    #[test]
    fn dedicated_kv_role_wins_over_the_label() {
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.kv_role = Some("kv_producer".to_string());
        let mut labels = HashMap::new();
        let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
        assert_eq!(role.as_deref(), Some("kv_producer"));

        labels.insert("kv_role".to_string(), "kv_consumer".to_string());
        let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
        assert_eq!(role.as_deref(), Some("kv_producer"));
        assert!(!labels.contains_key("kv_role"));

        spec.kv_role = Some(String::new());
        labels.insert("kv_role".to_string(), "kv_consumer".to_string());
        let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
        assert_eq!(role.as_deref(), Some("kv_consumer"));
    }

    /// A registration refused for a worker whose address carries userinfo
    /// names the address through the mask: the credential is in no refusal
    /// text (the job log prints the refusal beside the masked worker).
    #[tokio::test]
    async fn a_refusal_names_the_workers_address_through_the_mask() {
        use openai_protocol::worker::{OverloadUpdate, WorkerModels};
        use wfaas::WorkflowInstanceId;

        use crate::workflow::steps::create_worker_workflow_data;

        let mut spec = WorkerSpec::new("http://user:secret@worker:8000");
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.overload = OverloadUpdate {
            waiting_requests: Some(0),
            ..Default::default()
        };
        let mut data =
            create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, make_app_context());
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(ConnectionMode::Http);
        data.http_client_handle = Some(Arc::new(crate::server::http_client()));
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

        let text = CreateLocalWorkerStep
            .execute(&mut ctx)
            .await
            .expect_err("the overload override is refused")
            .to_string();
        assert!(text.contains("http://***@worker:8000"), "{text}");
        assert!(
            !text.contains("secret") && !text.contains("user:"),
            "{text}"
        );
    }

    #[test]
    fn dedicated_kv_metadata_wins_and_legacy_labels_are_removed() {
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.kv_connector = Some("NixlConnector".to_string());
        spec.kv_engine_id = Some("engine-direct".to_string());
        let mut labels = HashMap::from([
            ("kv_connector".to_string(), "legacy".to_string()),
            ("kv_role".to_string(), "kv_consumer".to_string()),
            ("kv_engine_id".to_string(), "engine-legacy".to_string()),
        ]);

        let metadata = take_kv_transfer_metadata(&spec, &mut labels);
        assert_eq!(
            metadata,
            (
                Some("NixlConnector".to_string()),
                Some("kv_consumer".to_string()),
                Some("engine-direct".to_string()),
            )
        );
        assert!(labels.is_empty());

        spec.kv_connector = Some(String::new());
        spec.kv_engine_id = Some(String::new());
        labels.insert("kv_connector".to_string(), "legacy".to_string());
        labels.insert("kv_engine_id".to_string(), "engine-legacy".to_string());
        let metadata = take_kv_transfer_metadata(&spec, &mut labels);
        assert_eq!(
            (metadata.0.as_deref(), metadata.2.as_deref()),
            (Some("legacy"), Some("engine-legacy"))
        );
    }

    #[test]
    fn dp_expansion_pins_only_multi_rank_widths() {
        let multi = DpInfo {
            dp_size: 3,
            model_id: "m".to_string(),
        };
        assert_eq!(
            dp_rank_expansion(Some(&multi), ConnectionMode::Grpc, "grpc://w:1").unwrap(),
            vec![Some((0, 3)), Some((1, 3)), Some((2, 3))]
        );

        // Width 1 is a capability-only advertisement: no placement exists,
        // and a rank-0 pin would still alter engine scheduling.
        let single = DpInfo {
            dp_size: 1,
            model_id: "m".to_string(),
        };
        assert_eq!(
            dp_rank_expansion(Some(&single), ConnectionMode::Grpc, "grpc://w:1").unwrap(),
            vec![None]
        );
        assert_eq!(
            dp_rank_expansion(None, ConnectionMode::Grpc, "grpc://w:1").unwrap(),
            vec![None]
        );
    }

    #[test]
    fn normalize_url_preserves_existing_schemes() {
        assert_eq!(
            normalize_url("http://localhost:30000", ConnectionMode::Http),
            "http://localhost:30000"
        );
        assert_eq!(
            normalize_url("https://localhost:30000", ConnectionMode::Http),
            "https://localhost:30000"
        );
        assert_eq!(
            normalize_url("grpc://localhost:30001", ConnectionMode::Grpc),
            "grpc://localhost:30001"
        );
        assert_eq!(
            normalize_url("grpcs://localhost:30001", ConnectionMode::Grpc),
            "grpcs://localhost:30001"
        );
    }

    #[test]
    fn normalize_url_adds_scheme_for_bare_urls() {
        assert_eq!(
            normalize_url("localhost:30000", ConnectionMode::Http),
            "http://localhost:30000"
        );
        assert_eq!(
            normalize_url("localhost:30001", ConnectionMode::Grpc),
            "grpc://localhost:30001"
        );
    }

    #[test]
    fn degenerate_overload_overrides_are_rejected_at_registration() {
        // Silent acceptance would strand the worker permanently vetoed; the
        // ConfigValidator rules apply to spec blocks too.
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.overload.waiting_requests = Some(0);
        let err =
            validate_overload_overrides(&spec).expect_err("waiting_requests = 0 must be rejected");
        assert!(err.contains("overload.waiting_requests"), "{err}");
        assert!(err.contains("http://worker:8080"), "{err}");

        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            let mut spec = WorkerSpec::new("http://worker:8080");
            spec.overload.token_usage = Some(bad);
            let err = validate_overload_overrides(&spec)
                .expect_err("degenerate token_usage must be rejected");
            assert!(err.contains("(0.0, 1.0]"), "{err}");
        }

        // The excluded ends' valid neighbors and an empty block pass.
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.overload.waiting_requests = Some(1);
        spec.overload.token_usage = Some(1.0);
        assert!(validate_overload_overrides(&spec).is_ok());
        assert!(validate_overload_overrides(&WorkerSpec::new("x")).is_ok());
    }

    #[test]
    fn zmq_handshake_override_is_rejected_off_the_zmq_path() {
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.zmq_handshake_address = Some("tcp://127.0.0.1:30500".to_string());

        for mode in [ConnectionMode::Http, ConnectionMode::Grpc] {
            let err = validate_zmq_handshake_override(&spec, mode)
                .expect_err("non-ZMQ workers must reject the handshake override");
            assert!(err.contains("zmq_handshake_address"), "{err}");
        }
        // On the ZMQ path the override is legitimate; unset is always fine.
        assert!(validate_zmq_handshake_override(&spec, ConnectionMode::Zmq).is_ok());
        assert!(
            validate_zmq_handshake_override(&WorkerSpec::new("x"), ConnectionMode::Http).is_ok()
        );
    }

    fn alias_map(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect()
    }

    #[test]
    fn model_alias_map_attaches_matching_aliases_to_discovered_card() {
        // The service-discovery path supplies no user card, so the card is
        // built from the discovered model ID alone; the router-level alias
        // map is the only way it can gain aliases.
        let spec = WorkerSpec::new("http://worker:8080");
        let aliases = alias_map(&[
            ("GLM-5.2-Coding", "GLM-5.2"),
            ("other-alias", "other-model"),
        ]);

        let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

        assert_eq!(card.id, "GLM-5.2");
        assert_eq!(card.aliases, vec!["GLM-5.2-Coding".to_string()]);
    }

    #[test]
    fn service_discovery_keeps_served_model_name_canonical() {
        // Service discovery supplies an empty model list. Metadata discovery
        // supplies the backend's served_model_name.
        let spec = WorkerSpec::new("http://worker:8080");
        let labels = HashMap::from([
            ("served_model_name".to_string(), "GLM-5.2".to_string()),
            ("model_id".to_string(), "GLM-5.2-Coding".to_string()),
            ("model_path".to_string(), "unrelated-alias".to_string()),
        ]);
        let aliases = alias_map(&[
            ("GLM-5.2-Coding", "GLM-5.2"),
            ("unrelated-alias", "other-model"),
        ]);

        let model_id = resolve_model_id(&spec, &labels);
        let card = build_model_card(model_id, &spec, &labels, &aliases);
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new(&spec.url).model(card).build());
        let registry = WorkerRegistry::new();
        registry.register(worker.clone()).unwrap();

        assert_eq!(worker.model_id(), "GLM-5.2");
        assert_eq!(registry.get_by_model("GLM-5.2").len(), 1);
        assert_eq!(registry.get_by_model("GLM-5.2-Coding").len(), 1);
        assert_eq!(
            registry.resolve_model_alias("GLM-5.2-Coding").as_deref(),
            Some("GLM-5.2")
        );
        assert!(registry.get_by_model("unrelated-alias").is_empty());
    }

    #[test]
    fn model_alias_map_is_case_sensitive_and_skips_self_reference() {
        let spec = WorkerSpec::new("http://worker:8080");
        // Wrong-case canonical must not match; an alias equal to the model ID
        // must not be attached (it would shadow the canonical entry).
        let aliases = alias_map(&[("glm-5.2-coding", "glm-5.2"), ("GLM-5.2", "GLM-5.2")]);

        let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

        assert!(card.aliases.is_empty());
    }

    fn zmq_worker(url: &str, handshake_override: Option<&str>) -> Arc<dyn Worker> {
        let mut spec = WorkerSpec::new(url);
        spec.connection_mode = ConnectionMode::Zmq;
        spec.zmq_handshake_address = handshake_override.map(str::to_string);
        Arc::new(BasicWorkerBuilder::from_spec(spec).build())
    }

    /// The ZMQ link's request bound resolves as the HTTP worker client's
    /// does: the worker's own `http_pool.timeout_secs` when set, else the
    /// router's `request_timeout_secs`.
    #[test]
    fn zmq_request_timeout_takes_the_workers_pool_timeout_over_the_routers() {
        let mut spec = WorkerSpec::new("ipc:///tmp/smg-zmq/bound.ipc");
        spec.connection_mode = ConnectionMode::Zmq;
        assert_eq!(zmq_request_timeout(&spec, 1800), Duration::from_secs(1800));
        spec.http_pool.timeout_secs = Some(7200);
        assert_eq!(zmq_request_timeout(&spec, 1800), Duration::from_secs(7200));
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
    fn zmq_workers_keep_health_checks_enabled() {
        // The ZMQ probe drives the handshake and evicts dead clients, so
        // disabling it would brick the worker on engine restart.
        let disabled = HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        };

        let resolved = resolve_zmq_health_config(
            disabled.clone(),
            ConnectionMode::Zmq,
            "ipc:///tmp/smg-zmq/a.ipc",
        );
        assert!(!resolved.disable_health_check);
        assert_eq!(
            resolved.check_interval_secs, disabled.check_interval_secs,
            "only the disable flag is overridden"
        );

        assert!(
            resolve_zmq_health_config(disabled, ConnectionMode::Http, "http://w:1")
                .disable_health_check,
            "other transports keep the operator's choice"
        );
    }

    #[test]
    fn zmq_disaggregated_workers_are_rejected_as_a_create_worker_failure() {
        let refused = [
            (RuntimeType::Vllm, WorkerType::Prefill),
            (RuntimeType::Vllm, WorkerType::Decode),
            (RuntimeType::Vllm, WorkerType::Encode),
            (RuntimeType::Sglang, WorkerType::Prefill),
            (RuntimeType::Sglang, WorkerType::Decode),
            (RuntimeType::Sglang, WorkerType::Encode),
        ];
        for (runtime, worker_type) in refused {
            let err = validate_zmq_worker_type(
                ConnectionMode::Zmq,
                runtime,
                worker_type,
                "ipc:///tmp/smg-zmq/ts0.ipc",
            )
            .expect_err("a role the ZMQ lane cannot serve must be rejected");
            match err {
                WorkflowError::StepFailed { step_id, message } => {
                    assert_eq!(step_id, StepId::new("create_worker"));
                    assert!(
                        message.contains(&worker_type.to_string()),
                        "message was: {message}"
                    );
                }
                other => panic!("expected StepFailed, got {other:?}"),
            }
        }

        // Every runtime's ZMQ lane serves Regular; a TokenSpeed engine, whose
        // generate request carries the rendezvous and whose encode server takes
        // the encode request, serves every role; every worker type over the
        // other transports stays accepted.
        for runtime in [RuntimeType::Vllm, RuntimeType::TokenSpeed] {
            validate_zmq_worker_type(
                ConnectionMode::Zmq,
                runtime,
                WorkerType::Regular,
                "ipc:///tmp/smg-zmq/ts0.ipc",
            )
            .expect("ZMQ regular workers are the supported case");
        }
        for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
            validate_zmq_worker_type(
                ConnectionMode::Zmq,
                RuntimeType::TokenSpeed,
                worker_type,
                "ipc:///tmp/smg-zmq/ts0.ipc",
            )
            .expect("a TokenSpeed ZMQ worker serves every role");
        }
        for mode in [ConnectionMode::Http, ConnectionMode::Grpc] {
            for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
                validate_zmq_worker_type(
                    mode,
                    RuntimeType::Sglang,
                    worker_type,
                    "grpc://worker:8080",
                )
                .expect("non-ZMQ transports serve every worker type");
            }
        }
    }

    #[test]
    fn zmq_data_parallel_is_rejected_as_a_create_worker_failure() {
        // dp_size > 1 over ZMQ is the only rejected combination, and it must
        // surface as a create_worker StepFailed.
        let err = validate_zmq_dp(ConnectionMode::Zmq, 2, "ipc:///tmp/smg-zmq/ts0.ipc")
            .expect_err("dp_size > 1 over ZMQ must be rejected");
        match err {
            WorkflowError::StepFailed { step_id, message } => {
                assert_eq!(step_id, StepId::new("create_worker"));
                assert!(message.contains("dp_size=2"), "message was: {message}");
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
    }

    #[test]
    fn single_engine_zmq_and_data_parallel_grpc_are_accepted() {
        // Single-engine ZMQ is the supported ZMQ shape; gRPC/HTTP data
        // parallelism is untouched by the ZMQ guard.
        validate_zmq_dp(ConnectionMode::Zmq, 1, "ipc:///tmp/smg-zmq/ts0.ipc")
            .expect("single-engine ZMQ must be accepted");
        validate_zmq_dp(ConnectionMode::Grpc, 4, "grpc://worker:8080")
            .expect("gRPC data parallelism must be accepted");
        validate_zmq_dp(ConnectionMode::Http, 4, "http://worker:8080")
            .expect("HTTP data parallelism must be accepted");
    }

    #[test]
    fn model_alias_map_does_not_duplicate_user_provided_alias() {
        // POST /workers can already carry aliases in the spec; the router map
        // must merge, not duplicate, so repeated registration stays stable.
        let mut spec = WorkerSpec::new("http://worker:8080");
        let card_with_alias = ModelCard::new("GLM-5.2").with_alias("GLM-5.2-Coding");
        spec.models = vec![card_with_alias].into();
        let aliases = alias_map(&[("GLM-5.2-Coding", "GLM-5.2"), ("glm-5.2", "GLM-5.2")]);

        let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

        assert_eq!(
            card.aliases
                .iter()
                .filter(|a| *a == "GLM-5.2-Coding")
                .count(),
            1
        );
        assert!(card.aliases.contains(&"glm-5.2".to_string()));
        assert_eq!(card.aliases.len(), 2);
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

        use crate::routers::grpc::client::ServerInfo;

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
            validate_parser_overrides(&card, "grpc://engine:50051")
                .unwrap_or_else(|e| panic!("{name}: the worker must register: {e}"));
            assert_eq!(
                GrammarStart::from_label(labels.get(GrammarStart::LABEL).map(String::as_str)),
                GrammarStart::AfterReasoning,
                "{name}: the grammar start reads as advertised"
            );
        }
    }

    /// A parser override is a name the gateway reads with: a table's own name, a name a table
    /// answers to (an older parser's whose format a table reads, a launcher's), or `hf`; the two
    /// names may select different tables (the tool-call name's reads, and startup says so once).
    #[test]
    fn a_tables_name_is_a_parser_override_and_an_older_name_a_table_answers_to_passes() {
        let mut card = ModelCard::new("m");
        card.tool_parser = Some("ling".to_string());
        card.reasoning_parser = Some("ling".to_string());
        let check = |card: &ModelCard| validate_parser_overrides(card, "http://w:1");
        assert!(check(&card).is_ok());
        card.reasoning_parser = Some("qwen3".to_string());
        assert!(check(&card).is_ok());
        card.tool_parser = Some("kimik2".to_string());
        card.reasoning_parser = Some("kimi_k3".to_string());
        assert!(check(&card).is_ok());
        card.tool_parser = Some("hf".to_string());
        card.reasoning_parser = None;
        assert!(check(&card).is_ok());
        // `hf` binds the other name to itself or to nothing.
        card.reasoning_parser = Some("qwen3".to_string());
        let err = check(&card).expect_err("bound to the template");
        assert!(err.contains("'hf'") && err.contains("'qwen3'"), "{err}");
    }

    #[test]
    fn unknown_parser_override_fails_validation() {
        let check = |card: &ModelCard| validate_parser_overrides(card, "http://user:secret@w:1");

        // No overrides: nothing to validate.
        assert!(check(&ModelCard::new("m")).is_ok());

        // A known name passes; unknown names fail loudly, naming the worker, the model and the
        // name. The refusal names the worker through the mask: a credential in its address is in
        // no refusal (the helper masks, so no caller can forget).
        assert!(check(&ModelCard::new("m").with_tool_parser("json")).is_ok());
        let err = check(&ModelCard::new("m").with_tool_parser("definitely-not-a-parser"))
            .expect_err("unknown tool parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");
        assert!(
            err.contains("http://***@w:1") && !err.contains("secret") && err.contains("'m'"),
            "{err}"
        );
        let err = check(&ModelCard::new("m").with_reasoning_parser("definitely-not-a-parser"))
            .expect_err("unknown reasoning parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");
        assert!(
            err.contains("http://***@w:1") && !err.contains("secret"),
            "{err}"
        );

        // An older parser's name no table answers to is refused with its reason.
        let err = check(&ModelCard::new("m").with_tool_parser("pythonic"))
            .expect_err("an older name no table answers to");
        assert!(
            err.contains("'pythonic'") && err.contains("llama4_pythonic"),
            "{err}"
        );
    }
}

#[cfg(test)]
mod rank_window_tests {
    use std::collections::HashMap;

    use smg_grpc_client::tokenspeed_proto;

    use super::{dp_rank_expansion, labels_for_rank};
    use crate::{
        routers::grpc::client::ServerInfo,
        worker::{basic::merge_labels, BasicWorkerBuilder, ConnectionMode, Worker},
        workflow::steps::local::{discover_dp::dp_info_from_labels, test_support::labels_of},
    };

    #[test]
    fn tokenspeed_discovery_bounds_each_rank_by_its_local_window() {
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
                let engine = info.to_labels();
                let dp = dp_info_from_labels(&engine).unwrap();
                let ranks =
                    dp_rank_expansion(Some(&dp), ConnectionMode::Grpc, "grpc://w:30000").unwrap();
                assert_eq!(ranks.len(), size as usize);
                for rank in ranks {
                    let worker = BasicWorkerBuilder::new("grpc://w:30000")
                        .labels(discovered(&engine, rank))
                        .build();
                    assert_eq!(worker.max_running_requests(), Some(expected));
                    let window = usize::from(worker.max_running_requests().unwrap());
                    assert!(worker.try_admit_pd(window, window));
                    assert!(!worker.try_admit_pd(1, window));

                    let override_labels = labels_of(&[("max_running_requests", "32")]);
                    let merged = merge_labels(&engine, &override_labels);
                    let overridden = BasicWorkerBuilder::new("grpc://w:30000")
                        .labels(labels_for_rank(&merged, rank, &override_labels))
                        .build();
                    assert_eq!(overridden.max_running_requests(), Some(32));
                }
            }
        }
    }

    /// The rank's labels when every label was discovered (none set by hand).
    fn discovered(
        engine: &HashMap<String, String>,
        dp: Option<(usize, usize)>,
    ) -> HashMap<String, String> {
        labels_for_rank(engine, dp, &HashMap::new())
    }

    /// A window the operator set by hand on the worker is the worker's only
    /// window, whichever name the operator used: the merge drops a
    /// discovered window under the other name (the reader prefers
    /// `max_num_seqs`, which would otherwise override the operator's
    /// figure), on an attention-DP engine, a plain data-parallel one and an
    /// unexpanded worker alike; the other labels merge as before, the
    /// operator's on top.
    #[test]
    fn a_window_the_operator_set_by_hand_is_the_workers_only_window() {
        let by_hand = labels_of(&[("max_running_requests", "512"), ("served_model_name", "m")]);
        let replicas = labels_of(&[
            ("max_num_seqs", "256"),
            ("dp_size", "4"),
            ("served_model_name", "engine"),
        ]);
        let merged = merge_labels(&replicas, &by_hand);
        assert_eq!(
            merged,
            labels_of(&[
                ("max_running_requests", "512"),
                ("dp_size", "4"),
                ("served_model_name", "m"),
            ])
        );
        assert_eq!(labels_for_rank(&merged, Some((1, 4)), &by_hand), merged);
        assert_eq!(labels_for_rank(&merged, None, &by_hand), merged);

        let by_hand = labels_of(&[("max_num_seqs", "512")]);
        let attention = labels_of(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
        let merged = merge_labels(&attention, &by_hand);
        assert_eq!(merged.get("max_running_requests"), None);
        let rank = labels_for_rank(&merged, Some((0, 8)), &by_hand);
        assert_eq!(rank.get("max_num_seqs").map(String::as_str), Some("512"));

        let nothing_by_hand = labels_of(&[("served_model_name", "m")]);
        let both = labels_of(&[("max_running_requests", "2048"), ("max_num_seqs", "256")]);
        let merged = merge_labels(&both, &nothing_by_hand);
        assert_eq!(
            merged.get("max_running_requests").map(String::as_str),
            Some("2048")
        );
        assert_eq!(merged.get("max_num_seqs").map(String::as_str), Some("256"));

        // The context window is a family of its own under the same rule: a
        // limit set by hand drops the engine's figure under the names the
        // reader prefers to it, and leaves the running window alone.
        let by_hand = labels_of(&[("max_model_len", "2048")]);
        let engine = labels_of(&[
            ("max_context_length", "131072"),
            ("context_length", "131072"),
            ("max_num_seqs", "256"),
        ]);
        assert_eq!(
            merge_labels(&engine, &by_hand),
            labels_of(&[("max_model_len", "2048"), ("max_num_seqs", "256")])
        );
    }

    /// An engine that splits its running window across its attention-DP
    /// workers advertises the whole engine's window; each of its rank
    /// workers gets its share, whichever way the engine's release spells
    /// the split; the other labels ride along. Replicas of attention-DP
    /// groups (`dp_size` next to a resolved `attn_dp_size`) each run the
    /// whole window: the share is by the attention-DP width, not by the
    /// dp-aware width of every rank.
    #[test]
    fn an_attention_dp_engines_window_is_each_ranks_share() {
        let engine = labels_of(&[
            ("max_running_requests", "16384"),
            ("enable_dp_attention", "true"),
            ("dp_size", "8"),
        ]);
        let rank = discovered(&engine, Some((3, 8)));
        assert_eq!(
            rank.get("max_running_requests").map(String::as_str),
            Some("2048")
        );
        assert_eq!(rank.get("dp_size").map(String::as_str), Some("8"));
        let newer = labels_of(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
        assert_eq!(
            discovered(&newer, Some((0, 8)))
                .get("max_running_requests")
                .map(String::as_str),
            Some("2048")
        );
        let replicas_of_groups = labels_of(&[
            ("max_running_requests", "4096"),
            ("dp_size", "2"),
            ("attn_dp_size", "4"),
        ]);
        assert_eq!(
            discovered(&replicas_of_groups, Some((5, 8)))
                .get("max_running_requests")
                .map(String::as_str),
            Some("1024")
        );
    }

    /// Plain data parallelism runs the whole flag on every rank, an engine
    /// whose figure is per rank already keeps it, and a worker that stands
    /// for the whole engine (no expansion, or a width of one) keeps the
    /// whole window.
    #[test]
    fn a_per_rank_window_and_an_unexpanded_worker_keep_their_labels() {
        let replicas = labels_of(&[("max_running_requests", "2048"), ("dp_size", "8")]);
        assert_eq!(discovered(&replicas, Some((1, 8))), replicas);
        let per_rank = labels_of(&[("max_num_seqs", "256"), ("dp_size", "4")]);
        assert_eq!(discovered(&per_rank, Some((0, 4))), per_rank);
        let attention = labels_of(&[
            ("max_running_requests", "16384"),
            ("enable_dp_attention", "true"),
        ]);
        assert_eq!(discovered(&attention, None), attention);
        assert_eq!(discovered(&attention, Some((0, 1))), attention);
    }
}

#[cfg(test)]
mod constrained_decoding_label_tests {
    use std::collections::HashMap;

    use super::validate_constrained_decoding_label;

    /// A configured or discovered `constrained_decoding` label must name a
    /// policy; anything else fails the registration with the reason.
    #[test]
    fn a_constrained_decoding_label_must_name_a_policy() {
        let mut labels = HashMap::new();
        let check = |labels: &HashMap<String, String>, step: &str| {
            validate_constrained_decoding_label(labels, "http://w:8000", step)
        };
        assert!(check(&labels, "create_worker").is_ok());
        labels.insert("constrained_decoding".to_string(), "always".to_string());
        assert!(check(&labels, "create_worker").is_ok());
        labels.insert("constrained_decoding".to_string(), "sometimes".to_string());
        let message = check(&labels, "create_worker").unwrap_err().to_string();
        assert!(message.contains("http://w:8000"), "{message}");
        assert!(message.contains("'requested' or 'always'"), "{message}");
        let update = check(&labels, "update_worker_properties").unwrap_err();
        assert!(
            format!("{update:?}").contains("update_worker_properties"),
            "{update}"
        );
    }
}
