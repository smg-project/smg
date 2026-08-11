//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::{model_card::ModelCard, model_type::ModelType, worker::WorkerSpec};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::{
        circuit_breaker::CircuitBreakerConfig, http_client::build_worker_http_client,
        resilience::resolve_resilience, worker::RuntimeType, BasicWorkerBuilder, ConnectionMode,
        Worker, UNKNOWN_MODEL_ID,
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

        // Merge labels: discovered first, then config (config takes precedence)
        let mut labels = context.data.discovered_labels.clone();
        for (key, value) in &config.labels {
            labels.insert(key.clone(), value.clone());
        }

        // Extract KV transfer config (dedicated metadata fields, not labels)
        let kv_connector = labels.remove("kv_connector");
        let kv_role = labels.remove("kv_role");
        let kv_engine_id = labels.remove("kv_engine_id").filter(|s| !s.is_empty());

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

        validate_zmq_handshake_override(config, *connection_mode).map_err(|message| {
            WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message,
            }
        })?;

        // Only vLLM EngineCore and TokenSpeed speak the ZMQ direct-backend wire.
        // Fail registration here rather than letting the connect-time rejection
        // strand the worker in Pending.
        if *connection_mode == ConnectionMode::Zmq
            && !matches!(runtime_type, RuntimeType::Vllm | RuntimeType::TokenSpeed)
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!(
                    "ZMQ worker {} has unsupported runtime {}: only vllm and tokenspeed \
                     are supported over the ZMQ direct backend",
                    config.url, runtime_type
                ),
            });
        }

        // Normalize URL
        let url = normalize_url(&config.url, *connection_mode);

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

        let http_client = build_worker_http_client(&config.http_pool, &app_context.router_config)
            .map_err(|e| WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message: e,
        })?;

        let health_base = app_context.router_config.health_check.to_protocol_config();
        let health_config = config.health.apply_to(&health_base);
        let health_endpoint = &app_context.router_config.health_check.endpoint;

        let dp_ranks: Vec<Option<(usize, usize)>> = if app_context.router_config.dp_aware {
            let dp_info = context
                .data
                .dp_info
                .as_ref()
                .ok_or_else(|| WorkflowError::ContextValueNotFound("dp_info".to_string()))?;
            validate_zmq_dp(*connection_mode, dp_info.dp_size, &config.url)?;
            (0..dp_info.dp_size)
                .map(|r| Some((r, dp_info.dp_size)))
                .collect()
        } else {
            vec![None] // single worker, no DP
        };

        let workers: Vec<Arc<dyn Worker>> = dp_ranks
            .into_iter()
            .map(|dp| {
                let mut builder = BasicWorkerBuilder::new(url.clone())
                    .model(model_card.clone())
                    .worker_type(config.worker_type)
                    .connection_mode(*connection_mode)
                    .runtime_type(runtime_type)
                    .circuit_breaker_config(circuit_breaker.clone())
                    .http_client(http_client.clone())
                    .resilience(resolved_resilience.clone())
                    .health_config(health_config.clone())
                    .health_endpoint(health_endpoint)
                    .bootstrap_port(config.bootstrap_port)
                    .priority(config.priority)
                    .cost(config.cost);

                if let Some((rank, size)) = dp {
                    builder = builder.dp_config(rank, size);
                }
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key.clone());
                }
                if !labels.is_empty() {
                    builder = builder.labels(labels.clone());
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
                // ZMQ promotion is event-driven: the worker signals the manager
                // the instant its handshake completes, so wire the registry's
                // connect signal. Other transports promote via polling.
                if *connection_mode == ConnectionMode::Zmq {
                    builder = builder
                        .connect_signal_tx(app_context.worker_registry.connect_signal_sender());
                }

                // Builder sets initial status: Pending if health-checked, Ready if not.
                Arc::new(builder.build()) as Arc<dyn Worker>
            })
            .collect();

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

/// Resolve the canonical model ID before aliases are applied.
///
/// Kubernetes service discovery creates a spec without model cards, so its
/// canonical ID comes from the backend's `served_model_name`. Router aliases
/// are deliberately absent from this function and cannot change discovery.
fn resolve_model_id<'a>(config: &'a WorkerSpec, labels: &'a HashMap<String, String>) -> &'a str {
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
fn warn_on_conflicting_parser_overrides(
    card: &ModelCard,
    worker_url: &str,
    registry: &crate::worker::WorkerRegistry,
) {
    for existing in registry.get_by_model(&card.id).iter() {
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

/// Reject parser-override names the registries don't know. Skipped when a
/// factory is absent (parsers unused in that configuration).
fn validate_parser_overrides(
    card: &ModelCard,
    worker_url: &str,
    tool_parser_factory: Option<&tool_parser::ParserFactory>,
    reasoning_parser_factory: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), String> {
    if let (Some(name), Some(factory)) = (card.tool_parser.as_deref(), tool_parser_factory) {
        if !factory.registry().has_parser(name) {
            return Err(format!(
                "worker {} declares unknown tool_parser '{}' for model '{}'",
                worker_url, name, card.id
            ));
        }
    }
    if let (Some(name), Some(factory)) =
        (card.reasoning_parser.as_deref(), reasoning_parser_factory)
    {
        if !factory.registry().has_parser(name) {
            return Err(format!(
                "worker {} declares unknown reasoning_parser '{}' for model '{}'",
                worker_url, name, card.id
            ));
        }
    }
    Ok(())
}

fn build_model_card(
    model_id: &str,
    config: &WorkerSpec,
    labels: &HashMap<String, String>,
    model_aliases: &HashMap<String, String>,
) -> ModelCard {
    let user_provided = config.models.find(model_id).is_some();
    let mut card = config
        .models
        .find(model_id)
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
        .get("supports_vision")
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

/// Reject a data-parallel worker the ZMQ path cannot serve.
///
/// A ZMQ worker binds a single EngineCore connection (engine_count=1); DP>1
/// needs the coordinator + wave protocol (not yet implemented), so fail loudly
/// rather than silently under-connecting. Only ZMQ with `dp_size > 1` is
/// rejected; gRPC/HTTP data parallelism and single-engine ZMQ are fine.
fn validate_zmq_dp(
    connection_mode: ConnectionMode,
    dp_size: usize,
    url: &str,
) -> Result<(), WorkflowError> {
    if connection_mode == ConnectionMode::Zmq && dp_size > 1 {
        return Err(WorkflowError::StepFailed {
            step_id: StepId::new("create_worker"),
            message: format!(
                "ZMQ worker {url} cannot run data-parallel (dp_size={dp_size}); \
                 DP>1 over ZMQ is not yet supported"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRegistry;

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

    #[test]
    fn unknown_parser_override_fails_validation() {
        let tool_factory = tool_parser::ParserFactory::default();
        let reasoning_factory = reasoning_parser::ParserFactory::default();

        // No overrides → nothing to validate.
        let card = ModelCard::new("m");
        assert!(validate_parser_overrides(
            &card,
            "http://w:1",
            Some(&tool_factory),
            Some(&reasoning_factory)
        )
        .is_ok());

        // Known tool parser passes; unknown names fail loudly.
        let card = ModelCard::new("m").with_tool_parser("json");
        assert!(validate_parser_overrides(&card, "http://w:1", Some(&tool_factory), None).is_ok());

        let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
        let err = validate_parser_overrides(&card, "http://w:1", Some(&tool_factory), None)
            .expect_err("unknown tool parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");

        let card = ModelCard::new("m").with_reasoning_parser("definitely-not-a-parser");
        let err = validate_parser_overrides(&card, "http://w:1", None, Some(&reasoning_factory))
            .expect_err("unknown reasoning parser must fail registration");
        assert!(err.contains("definitely-not-a-parser"), "{err}");

        // Absent factories skip validation (parsers unused in that config).
        let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
        assert!(validate_parser_overrides(&card, "http://w:1", None, None).is_ok());
    }
}
