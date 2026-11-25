//! Build a local registration's workers from its spec and discovered labels.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{DpInfo, WorkerKind, WorkerRegistrationWorkflowData},
    host::{require_host, WorkflowHost},
    runtimes::{zmq_runtime, zmq_runtime_names, DEFAULT_RUNTIME},
    select::find_workers_by_url,
};
use crate::{
    connection::engine::EngineConnectionCache,
    health::resilience::resolve_resilience,
    identity::{
        basic::{
            merge_labels, ConnectionMode, RuntimeType, Worker, CONTEXT_WINDOW_LABELS, WINDOW_LABELS,
        },
        builder::BasicWorkerBuilder,
        endpoint::Endpoint,
    },
    jobs::WorkerRegistrationMode,
    labels::{has_control_character, SUPPORTS_VISION_LABEL},
    model_card::ModelCard,
    model_type::ModelType,
    registry::WorkerOrigin,
    worker::{ConstrainedDecoding, HealthCheckConfig, WorkerSpec, WorkerType},
    UNKNOWN_MODEL_ID,
};

/// The id of the step that builds a local worker's objects.
pub const CREATE_LOCAL_WORKER_STEP: &str = "create_local_worker";

/// Builds a local registration's workers from the spec and what detection
/// found. It leaves the workers in the run's data for the next step.
pub struct CreateLocalWorkerStep;

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for CreateLocalWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let host = require_host(context.data.host.as_ref())?;
        let registry = host.worker_registry();
        let settings = host.registration_settings();
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;

        let create_only = context.data.registration_mode == WorkerRegistrationMode::CreateOnly;
        if create_only && registry.get_by_url(&config.url).is_some() {
            return Err(already_exists(&config.url));
        }

        // Labels from the spec take precedence over discovered labels.
        let mut labels = merge_labels(&context.data.discovered_labels, &config.labels);

        validate_constrained_decoding_label(&labels, &config.url, "create_worker")?;
        let (kv_connector, kv_role, kv_engine_id) = take_kv_transfer_metadata(config, &mut labels);

        let model_id = resolve_model_id(config, &labels);
        // A ZMQ worker does not report its model before creation. Use the
        // host's model path when the spec and labels give no identity.
        let model_id = if model_id == UNKNOWN_MODEL_ID && *connection_mode == ConnectionMode::Zmq {
            settings
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

        let model_card = build_model_card(model_id, config, &labels, &settings.model_aliases);
        let runtime_type = match context.data.detected_runtime_type.as_deref() {
            Some(s) => s.parse::<RuntimeType>().unwrap_or(config.runtime_type),
            None => config.runtime_type,
        };

        let runtime_type = if runtime_type.is_specified() {
            runtime_type
        } else {
            debug!(
                "Runtime type unresolved for {} after detection; defaulting to {DEFAULT_RUNTIME}",
                Endpoint::redacted_str(&config.url)
            );
            DEFAULT_RUNTIME
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

        if *connection_mode == ConnectionMode::Zmq && zmq_runtime(runtime_type).is_none() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!(
                    "ZMQ worker {} has unsupported runtime {}: only {} are supported over the \
                     ZMQ direct backend",
                    Endpoint::redacted_str(&config.url),
                    runtime_type,
                    zmq_runtime_names()
                ),
            });
        }

        validate_zmq_worker_type(
            *connection_mode,
            runtime_type,
            config.worker_type,
            &config.url,
        )?;

        let url = normalize_url(&config.url, *connection_mode);

        let (resolved_resilience, circuit_breaker) = resolve_resilience(
            &settings.retry,
            &settings.circuit_breaker,
            settings.retries_enabled,
            settings.circuit_breaker_enabled,
            &config.resilience,
        );

        let http_client = context.data.http_client("create_worker")?;
        let http2 = context.data.http2.unwrap_or(false);

        let health_config = resolve_zmq_health_config(
            config.health.apply_to(&settings.health),
            *connection_mode,
            &url,
        );
        let health_endpoint = &settings.health_endpoint;

        let dp_ranks = dp_rank_expansion(context.data.dp_info.as_ref());
        // The workers that stand for this address, read from the registry:
        // the plain one, and the rank workers at `{base}@{rank}`, which the
        // base address checked above does not find.
        let standing = find_workers_by_url(registry, &url);
        // A create is refused by any of them, whatever shape this
        // registration read.
        if create_only && !standing.is_empty() {
            return Err(already_exists(&config.url));
        }

        // Keep the spec's other fields. Set the resolved address, model,
        // labels, KV transfer fields and placement on each worker below.
        let mut spec = config.clone();
        spec.url.clone_from(&url);
        spec.dp_base_url = None;
        spec.dp_rank = None;
        spec.dp_size = None;
        spec.kv_connector = None;
        spec.kv_role = None;
        spec.kv_engine_id = None;

        // The engines of a ZMQ group dial one socket set, whichever worker
        // binds it: the group's rank workers share one connection.
        let zmq_rank_connection = (*connection_mode == ConnectionMode::Zmq && dp_ranks.len() > 1)
            .then(|| Arc::new(EngineConnectionCache::new(OnceCell::new())));

        let workers: Vec<Arc<dyn Worker>> = dp_ranks
            .into_iter()
            .map(|dp| {
                let mut builder: BasicWorkerBuilder = BasicWorkerBuilder::from_spec(spec.clone())
                    .model(model_card.clone())
                    .connection_mode(*connection_mode)
                    .engine_connector(host.engine_connector())
                    .runtime_type(runtime_type)
                    .circuit_breaker_config(circuit_breaker.clone())
                    .http_client(http_client.clone())
                    .http2(http2)
                    .resilience(resolved_resilience.clone())
                    .health_config(health_config.clone())
                    .health_endpoint(health_endpoint)
                    .zmq_startup_timeout(Duration::from_secs(settings.zmq_startup_timeout_secs))
                    // The request bound of the HTTP worker client, for the
                    // ZMQ link: a stream the engine leaves silent ends at it.
                    .zmq_request_timeout(zmq_request_timeout(&spec, settings.request_timeout_secs))
                    .overload_defaults(settings.overload_defaults)
                    .labels(labels_for_rank(&labels, dp, &config.labels))
                    // Keep a copy of the report in this rank's space. A later
                    // refresh leaves every label in the spec as the operator set it.
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
                if let Some(ref slot) = zmq_rank_connection {
                    builder = builder.shared_engine_connection(Arc::clone(slot));
                }
                // A new or recovered connection can wake the health manager.
                builder = builder.connect_signal_tx(registry.connect_signal_sender());

                Arc::new(builder.build()) as Arc<dyn Worker>
            })
            .collect();

        // Any other registration replaces every worker of this node that
        // stands, or changes nothing. The registry step replaces a worker
        // under its own address alone, so a standing worker is replaced where
        // one was built at its address, whatever rank either states. A
        // registration that read another shape would leave a worker beside
        // its own, and the registry would hold the engine twice: a plain
        // worker beside rank workers, where a server info failed to read on
        // either registration, or a rank above the size read now.
        //
        // A peer's worker, imported from its published state, holds nothing
        // back: this node takes the addresses it is configured with, where
        // the registry step claims an import, and a peer's worker of another
        // shape stays the import's to settle. The engine can then stand
        // twice on this node until the peer's state changes.
        let own: Vec<&Arc<dyn Worker>> = standing
            .iter()
            .filter(|stands| {
                registry
                    .get_id_by_url(stands.url())
                    .is_some_and(|id| registry.origin_of(&id) == Some(WorkerOrigin::Local))
            })
            .collect();
        let kept: Vec<&Arc<dyn Worker>> = own
            .iter()
            .copied()
            .filter(|stands| workers.iter().all(|built| built.url() != stands.url()))
            .collect();
        // The refusal names one of them, the first by its address.
        if let Some(first) = kept.iter().min_by_key(|stands| stands.url()) {
            // A rank worker is one whose address ends in a rank, as the
            // finder selected it.
            let standing_ranks = own
                .iter()
                .filter(|stands| {
                    Endpoint::parse_with_rank(stands.url()).is_ok_and(|(_, rank)| rank.is_some())
                })
                .count();
            let stands = match (own.len() - standing_ranks, standing_ranks) {
                (1, 0) => "one worker".to_string(),
                (plain, 0) => format!("{plain} workers"),
                (0, 1) => "one rank worker".to_string(),
                (0, ranks) => format!("{ranks} rank workers"),
                (plain, ranks) => format!("{plain} plain and {ranks} rank workers"),
            };
            let built_ranks = workers
                .iter()
                .filter(|built| built.dp_rank().is_some())
                .count();
            let read = match built_ranks {
                0 => "no data-parallel ranks".to_string(),
                ranks => format!("{ranks} data-parallel ranks"),
            };
            let others = match kept.len() - 1 {
                0 => String::new(),
                more => format!(" and {more} more"),
            };
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!(
                    "Worker {} stands as {stands} and this registration read {read}: it would \
                     leave {}{others} beside its own, so nothing is changed (remove what stands \
                     at the address to register it again)",
                    Endpoint::redacted_str(&config.url),
                    // As it is registered, so that another spelling shows.
                    Endpoint::redacted_url(first.url()),
                ),
            });
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
pub fn validate_constrained_decoding_label(
    labels: &HashMap<String, String>,
    url: &str,
    step: &str,
) -> WorkflowResult<()> {
    match labels.get(ConstrainedDecoding::LABEL) {
        Some(value) => {
            // The value may be the worker's own report, and the refusal of an
            // unknown policy repeats it: a control character is refused first,
            // in a sentence that names the label and not the value.
            let policy = if has_control_character(value) {
                Err(format!(
                    "{} must not contain control characters",
                    ConstrainedDecoding::LABEL
                ))
            } else {
                ConstrainedDecoding::parse(value)
            };
            policy.map(drop).map_err(|why| WorkflowError::StepFailed {
                step_id: StepId::new(step),
                message: format!("Worker {}: {why}", Endpoint::redacted_str(url)),
            })
        }
        None => Ok(()),
    }
}

/// Treat an empty KV transfer value as unset, so a later report can fill it.
pub fn kv_transfer_value(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

/// The worker's KV transfer fields out of the spec and the labels (the
/// three keys leave the labels): the spec's value when it says something,
/// else the label's, read through [`kv_transfer_value`] on both sides.
fn take_kv_transfer_metadata(
    config: &WorkerSpec,
    labels: &mut HashMap<String, String>,
) -> (Option<String>, Option<String>, Option<String>) {
    let connector_label = kv_transfer_value(labels.remove("kv_connector"));
    let role_label = kv_transfer_value(labels.remove("kv_role"));
    let engine_id_label = kv_transfer_value(labels.remove("kv_engine_id"));
    (
        kv_transfer_value(config.kv_connector.clone()).or(connector_label),
        kv_transfer_value(config.kv_role.clone()).or(role_label),
        kv_transfer_value(config.kv_engine_id.clone()).or(engine_id_label),
    )
}

/// Read the canonical model ID from the first model card or from the
/// worker's labels. Aliases do not change it.
pub fn resolve_model_id<'a>(
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

/// The card a worker is registered with: the operator's card for `model_id`,
/// if the spec has one, completed from the labels, else one built from the
/// labels alone. Add the host's aliases for the model to either card.
pub fn build_model_card(
    model_id: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
    model_aliases: &HashMap<String, String>,
) -> ModelCard {
    let mut card = card_from_labels(config.models.find(model_id), model_id, labels);

    // Keep the card's own aliases and attach each new alias once.
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
pub fn model_card_for_labels(
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

    // Labels only fill gaps in the card's overrides.
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

    // A card built from the labels follows an embedding or non-generation
    // signal (which wins over vision); otherwise, the operator's card or a
    // generation card from the labels gains vision.
    let non_generation = !user_provided
        && (labels.get("is_embedding").is_some_and(|s| s == "true")
            || labels.get("is_generation").is_some_and(|s| s == "false"));
    if non_generation {
        card.model_type = infer_non_generation_type(labels);
    } else if has_vision {
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

/// Reject thresholds that would mark a worker overloaded on every sample.
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

/// The ZMQ request timeout uses the same override as the HTTP client.
fn zmq_request_timeout(spec: &WorkerSpec, request_timeout_secs: u64) -> Duration {
    Duration::from_secs(spec.http_pool.timeout_secs.unwrap_or(request_timeout_secs))
}

/// Refuse a ZMQ handshake address on a transport that cannot use it.
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

/// Keep the ZMQ health probe on so it can reconnect a restarted engine.
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

/// Reject roles the runtime cannot serve over its ZMQ link.
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
        _ if zmq_runtime(runtime_type).is_some_and(|row| row.supports_disaggregation) => {
            return Ok(())
        }
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

/// Give an expanded rank its share of a discovered running window when
/// the report says the window is split across attention-DP ranks. Keep a
/// window set by hand, one already sized per rank, or one on a plain worker.
pub fn labels_for_rank(
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

/// Size a rank's discovered window by the same width as its merged labels.
/// Keep the width labels in the report as discovered for later refreshes.
pub fn report_for_rank(
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
pub const DP_WIDTH_LABELS: [&str; 2] = ["attn_dp_size", "enable_dp_attention"];

/// Read the width that shares a window. A reported attention-DP width
/// excludes replicas. With only the attention-DP flag, use the rank count.
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

/// The refusal of a create for an address that already has a worker.
fn already_exists(url: &str) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new("create_worker"),
        message: format!("Worker {} already exists", Endpoint::redacted_str(url)),
    }
}

/// Ranks to register for a worker, from the shape the DP read left. A width
/// above one expands into one dp-annotated worker per rank. A width-1
/// advertisement (older servicers emit it as a capability signal) and a
/// missing one both register a plain worker: there is no placement to
/// choose, and pinning rank 0 anyway still alters engine-side scheduling.
fn dp_rank_expansion(dp_info: Option<&DpInfo>) -> Vec<Option<(usize, usize)>> {
    match dp_info {
        Some(dp_info) if dp_info.dp_size > 1 => (0..dp_info.dp_size)
            .map(|r| Some((r, dp_info.dp_size)))
            .collect(),
        Some(_) | None => vec![None],
    }
}

#[cfg(test)]
mod tests;
