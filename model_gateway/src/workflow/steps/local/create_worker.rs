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
    config::ParserChoice,
    routers::grpc::{
        multimodal::{
            mm_settings, pool_processes_media_refs, registration_lacks_media_processing,
            SUPPORTS_VISION_LABEL,
        },
        regular::symphony::{check_names, labels_select_nothing, set_aside_by_the_registry},
        utils::Names,
        zmq_client::zmq_handshake_address,
    },
    worker::{
        circuit_breaker::CircuitBreakerConfig,
        overload::OverloadThresholds,
        resilience::resolve_resilience,
        worker::{merge_labels, RuntimeType, WINDOW_LABELS},
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
                message: format!("Worker {} already exists", config.url),
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
                        config.url
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

        // A parser override naming an unknown parser would silently ship
        // unparsed output at serve time — fail registration loudly instead
        // (mirrors the fail-fast AppContext applies to the global CLI names).
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

        // Mixed overrides across same-model workers are a misconfiguration
        // (except transiently during rolling upgrades): resolution picks one
        // deterministically, but only one family parses correctly. Warn, don't
        // fail — failing would block rolling upgrades that change the parser.
        warn_on_conflicting_parser_overrides(
            &model_card,
            &config.url,
            &app_context.worker_registry,
        );

        // Under Symphony the registry's table reads a model it knows for a
        // worker whose labels select no reader, whatever the gateway's flags
        // name (`regular::symphony::names::reads`): the model's first such worker
        // says so.
        if let Some(message) = registry_reads_over_the_flags(
            &model_card,
            &config.url,
            app_context.router_config.parser,
            &Names {
                tool: app_context.router_config.tool_call_parser.clone(),
                reasoning: app_context.router_config.reasoning_parser.clone(),
            },
            &app_context.worker_registry,
        ) {
            warn!(model = %model_card.id, worker = %config.url, "{message}");
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
                config.url
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
                    config.url, runtime_type
                ),
            });
        }

        validate_zmq_worker_type(*connection_mode, config.worker_type, &config.url)?;

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
                    url = %worker.url(),
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
            url,
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
                    message: format!("Worker {url}: {why}"),
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
        // old card is still registered while the new one is checked.
        if existing.base_url() == worker_url {
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
                        new_worker = worker_url,
                        existing_worker = existing.url(),
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

/// Reject parser-override names the registries don't know: the legacy parsers', and, under
/// `--parser symphony`, Symphony's tables (`regular::symphony::check_names`). A name whose
/// factory is absent is not checked (parsers unused in that configuration).
pub(super) fn validate_parser_overrides(
    card: &ModelCard,
    worker_url: &str,
    parser: ParserChoice,
    tool_parser_factory: Option<&tool_parser::ParserFactory>,
    reasoning_parser_factory: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), String> {
    let names = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    check_names(
        parser,
        &names,
        tool_parser_factory,
        reasoning_parser_factory,
    )
    .map_err(|why| {
        format!(
            "worker {worker_url} declares a parser override for model '{}' the gateway cannot \
             read with: {why}",
            card.id
        )
    })
}

/// Under `--parser symphony`, the registry's table reads a model it knows for every worker whose
/// labels select no reader (none, or a reasoning name alone that no table answers to), whatever the
/// gateway's `--tool-call-parser` and `--reasoning-parser` name (`regular::symphony::names::reads`). When
/// they name another reader, the registration of the model's first worker says so: `Some(message)`
/// while no worker of the model is registered, naming the model, both readers and the one used;
/// `None` for the model's later workers, a worker whose labels select a reader, a model the
/// registry does not know, flags that select nothing or the registry's own table, and the legacy
/// parser choice. Workers of one model registering at the same moment may each say it, as the
/// override check above may: the registry gains the worker a step later.
pub(super) fn registry_reads_over_the_flags(
    card: &ModelCard,
    worker_url: &str,
    parser: ParserChoice,
    flags: &Names,
    registry: &WorkerRegistry,
) -> Option<String> {
    let labels = Names {
        tool: card.tool_parser.clone(),
        reasoning: card.reasoning_parser.clone(),
    };
    if parser != ParserChoice::Symphony
        || !labels_select_nothing(&labels)
        || !registry.get_by_model(&card.id).is_empty()
    {
        return None;
    }
    let (table, named) = set_aside_by_the_registry(flags, &card.id)?;
    let spell =
        |name: Option<&str>| name.map_or_else(|| "unset".to_string(), |name| format!("'{name}'"));
    Some(format!(
        "the registry's table '{}' reads model '{}' for worker {worker_url}, whose labels select \
         no reader; the gateway's --tool-call-parser {} / --reasoning-parser {} name {named}, which \
         reads only the models the registry does not know",
        table.name(),
        card.id,
        spell(flags.tool.as_deref()),
        spell(flags.reasoning.as_deref()),
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
        card.context_length = [
            "context_length",
            "max_context_length",
            "max_model_len",
            "max_total_tokens",
            "max_seq_len",
        ]
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
            config.url
        ));
    }
    if let Some(threshold) = config.overload.token_usage {
        if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
            return Err(format!(
                "worker {} sets overload.token_usage to {threshold}: Must be a \
                 fraction in (0.0, 1.0]",
                config.url
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
            config.url
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
                "ZMQ worker {url} would bind handshake address {handshake}, already claimed by \
                 worker {existing_url}: the derived port is a hash of the ipc path, so rename \
                 this worker's ipc path or set zmq_handshake_address to a free tcp:// address"
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
        "Ignoring disabled health checks for ZMQ worker {url}: the ZMQ probe is a local liveness \
         check and the only path that reconnects a restarted engine, so it stays enabled"
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
/// The ZMQ wire carries no KV-transfer rendezvous: neither the vLLM
/// `EngineCoreRequest` nor the TokenSpeed `TokenizedGenerateReqInput` has a
/// field for the bootstrap info PD injects, so a ZMQ prefill/decode worker would
/// silently drop it and degrade to a decode-side recompute or a stalled KV
/// wait. Encode fares no better: encode dispatch is a gRPC encoder RPC a
/// direct-ZMQ worker has no path for. `Regular` is therefore the only role the
/// ZMQ lane serves, and every disaggregated worker is rejected at registration —
/// matching the gRPC-only PD/EPD selection folds, which would otherwise leave
/// such a worker registered but never selected.
fn validate_zmq_worker_type(
    connection_mode: ConnectionMode,
    worker_type: WorkerType,
    url: &str,
) -> Result<(), WorkflowError> {
    if connection_mode == ConnectionMode::Zmq
        && matches!(
            worker_type,
            WorkerType::Prefill | WorkerType::Decode | WorkerType::Encode
        )
    {
        return Err(WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message: format!(
                "ZMQ worker {url} cannot serve worker type {worker_type}: the direct-ZMQ \
                 backend carries no KV-transfer metadata and has no encode path, so \
                 encode/prefill/decode disaggregation requires a gRPC worker"
            ),
        });
    }
    Ok(())
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
                "ZMQ worker {url} cannot be dp-aware expanded (dp_size={dp_size}); \
                 configure a grouped ZMQ worker (`dp_size` on the worker spec) instead"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rank_window_tests {
    use std::collections::HashMap;

    use smg_grpc_client::tokenspeed_proto;

    use super::{dp_rank_expansion, labels_for_rank};
    use crate::{
        routers::grpc::client::ServerInfo,
        worker::{worker::merge_labels, BasicWorkerBuilder, ConnectionMode, Worker},
        workflow::steps::local::discover_dp::dp_info_from_labels,
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

                    let override_labels = labels(&[("max_running_requests", "32")]);
                    let merged = merge_labels(&engine, &override_labels);
                    let overridden = BasicWorkerBuilder::new("grpc://w:30000")
                        .labels(labels_for_rank(&merged, rank, &override_labels))
                        .build();
                    assert_eq!(overridden.max_running_requests(), Some(32));
                }
            }
        }
    }

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
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
        let by_hand = labels(&[("max_running_requests", "512"), ("served_model_name", "m")]);
        let replicas = labels(&[
            ("max_num_seqs", "256"),
            ("dp_size", "4"),
            ("served_model_name", "engine"),
        ]);
        let merged = merge_labels(&replicas, &by_hand);
        assert_eq!(
            merged,
            labels(&[
                ("max_running_requests", "512"),
                ("dp_size", "4"),
                ("served_model_name", "m"),
            ])
        );
        assert_eq!(labels_for_rank(&merged, Some((1, 4)), &by_hand), merged);
        assert_eq!(labels_for_rank(&merged, None, &by_hand), merged);

        let by_hand = labels(&[("max_num_seqs", "512")]);
        let attention = labels(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
        let merged = merge_labels(&attention, &by_hand);
        assert_eq!(merged.get("max_running_requests"), None);
        let rank = labels_for_rank(&merged, Some((0, 8)), &by_hand);
        assert_eq!(rank.get("max_num_seqs").map(String::as_str), Some("512"));

        let nothing_by_hand = labels(&[("served_model_name", "m")]);
        let both = labels(&[("max_running_requests", "2048"), ("max_num_seqs", "256")]);
        let merged = merge_labels(&both, &nothing_by_hand);
        assert_eq!(
            merged.get("max_running_requests").map(String::as_str),
            Some("2048")
        );
        assert_eq!(merged.get("max_num_seqs").map(String::as_str), Some("256"));
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
        let engine = labels(&[
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
        let newer = labels(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
        assert_eq!(
            discovered(&newer, Some((0, 8)))
                .get("max_running_requests")
                .map(String::as_str),
            Some("2048")
        );
        let replicas_of_groups = labels(&[
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
        let replicas = labels(&[("max_running_requests", "2048"), ("dp_size", "8")]);
        assert_eq!(discovered(&replicas, Some((1, 8))), replicas);
        let per_rank = labels(&[("max_num_seqs", "256"), ("dp_size", "4")]);
        assert_eq!(discovered(&per_rank, Some((0, 4))), per_rank);
        let attention = labels(&[
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
