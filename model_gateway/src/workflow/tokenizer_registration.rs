//! Tokenizer registration workflow
//!
//! This module provides a workflow for registering tokenizers asynchronously.
//! Tokenizers can be loaded from local paths or downloaded from HuggingFace.
//!
//! This is the **single source of truth** for tokenizer registration. All paths
//! (startup, worker connection, API) should use this workflow to ensure consistent
//! behavior (validation, caching, deduplication).

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use dashmap::DashMap;
use llm_tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    factory,
    registry::{LoadOutcome, TokenizerRegistry},
    traits::Tokenizer,
};
use serde::{Deserialize, Serialize};
use smg_grpc_client::{tokenizer_bundle, tokenizer_bundle::StreamBundle};
use tracing::{debug, error, info, warn};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::data::TokenizerWorkflowData;
use crate::{
    app_context::AppContext,
    config::{TokenizerCacheConfig, TEMPLATE_PARSER},
    routers::grpc::multimodal::{
        load_image_preprocessor_config, load_video_preprocessor_config, MultimodalModelConfig,
    },
    worker::{ConnectionMode, WorkerRegistry},
};

/// Configuration for adding a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfigRequest {
    /// Pre-generated UUID for this tokenizer
    pub id: String,
    /// User-provided name (what to register under in the registry)
    pub name: String,
    /// Source: either a local path or HuggingFace model ID
    pub source: String,
    /// Optional path to chat template file
    pub chat_template_path: Option<String>,
    /// Optional cache configuration. If provided, wraps tokenizer with CachedTokenizer.
    #[serde(default)]
    pub cache_config: Option<TokenizerCacheConfig>,
    /// If true, the workflow fails when a tokenizer with the same name already exists.
    /// If false (default), the workflow succeeds and returns the existing tokenizer's ID.
    /// API callers should set this to true.
    #[serde(default)]
    pub fail_on_duplicate: bool,
}

/// Configuration for removing a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerRemovalRequest {
    /// UUID of the tokenizer to remove
    pub id: String,
}

/// Why a served model's tokenizer will not register on its own: the terminal
/// outcome of its last load, kept per model so that a request for the model
/// is answered with that fact (503 `tokenizer_load_failed`, no `Retry-After`)
/// instead of the retry hint of a load still in flight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerLoadFailure {
    /// Nothing to load from: neither `tokenizer_path` nor `model_path` among
    /// the worker's labels, and no `--tokenizer-path` or `--model-path` in
    /// the router configuration.
    NoSource,
    /// The registration workflow failed; its error, for the log.
    Failed(String),
    /// The model's tokenizer was removed (`DELETE /v1/tokenizers/{id}`)
    /// while the model is still served; nothing reloads it on its own.
    Removed,
}

impl TokenizerLoadFailure {
    /// The reason as a response names it: the class only, never a path or an
    /// error text (the gateway's log line for the model has those).
    pub fn reason(&self) -> &'static str {
        match self {
            Self::NoSource => {
                "no tokenizer path in the worker's labels or the router configuration"
            }
            Self::Failed(_) => "the load failed; see the gateway log",
            Self::Removed => "the tokenizer was removed; register one again",
        }
    }
}

/// A model's tokenizer loads: how many are in flight (accepted, not ended)
/// and the terminal outcome of the last one, when it ended without a
/// tokenizer.
#[derive(Debug, Default)]
struct ModelLoadState {
    in_flight: u32,
    failure: Option<TokenizerLoadFailure>,
}

/// The tokenizer loads of the served models, by model id: the loads in
/// flight and the last terminal failure. A model with a load in flight is
/// pending, whatever was recorded before (the load decides when it ends); a
/// model without one and without a registered tokenizer is pending too,
/// until a failure is recorded: its load skipped for lack of a source, the
/// last of its loads failed for good, or its tokenizer removed. A load that
/// is accepted or succeeds clears the record (a worker re-registering with a
/// usable path, an operator's `POST /v1/tokenizers`).
#[derive(Debug, Default)]
pub struct TokenizerLoadFailures {
    by_model: DashMap<String, ModelLoadState>,
}

impl TokenizerLoadFailures {
    pub fn new() -> Self {
        Self::default()
    }

    /// A load for the model is handed to the queue: the model is pending
    /// until it ends, whatever was recorded. Counted with the queue slot held
    /// and the hand-over right behind it, so a job that ends at once finds
    /// its count to release.
    pub fn load_started(&self, model: &str) {
        let mut state = self.by_model.entry(model.to_string()).or_default();
        state.in_flight += 1;
        state.failure = None;
    }

    /// A load for the model ended; `true` when it was the last one in flight
    /// (the moment its failure counts). A load that was never counted (none
    /// is, outside [`Self::load_started`]) leaves no state behind.
    pub fn load_ended(&self, model: &str) -> bool {
        let Some(mut state) = self.by_model.get_mut(model) else {
            return true;
        };
        state.in_flight = state.in_flight.saturating_sub(1);
        let last = state.in_flight == 0;
        drop(state);
        self.forget_if_blank(model);
        last
    }

    /// Records the model's failure, unless a load is in flight (that load
    /// decides); `true` when the model was not in that state before (the
    /// moment to log it: once per model and entry).
    pub fn record(&self, model: &str, failure: TokenizerLoadFailure) -> bool {
        let mut state = self.by_model.entry(model.to_string()).or_default();
        if state.in_flight > 0 {
            return false;
        }
        let newly = state.failure.as_ref() != Some(&failure);
        state.failure = Some(failure);
        newly
    }

    /// Forgets the model's failure, a load having succeeded; `true` when
    /// there was one.
    pub fn clear(&self, model: &str) -> bool {
        let Some(mut state) = self.by_model.get_mut(model) else {
            return false;
        };
        let had = state.failure.take().is_some();
        drop(state);
        self.forget_if_blank(model);
        had
    }

    /// The model's recorded failure, unless a load is in flight.
    pub fn get(&self, model: &str) -> Option<TokenizerLoadFailure> {
        self.by_model
            .get(model)
            .filter(|state| state.in_flight == 0)
            .and_then(|state| state.failure.clone())
    }

    fn forget_if_blank(&self, model: &str) {
        self.by_model.remove_if(model, |_, state| {
            state.in_flight == 0 && state.failure.is_none()
        });
    }
}

/// Notes the end of a tokenizer registration job for `model` (the autoload a
/// worker's registration starts, an operator's request and the start-up load
/// alike): a success clears the model's record; a failure is recorded, and
/// logged once per model and entry, when it was the model's last load in
/// flight, the model is served and the registry holds no tokenizer for it (a
/// failed duplicate registration takes nothing away; a name no served model
/// carries is nobody's request). Requests for a recorded model are answered
/// 503 `tokenizer_load_failed` without a retry hint until a tokenizer
/// registers.
pub fn note_tokenizer_load_outcome(
    failures: &TokenizerLoadFailures,
    registry: &TokenizerRegistry,
    workers: &WorkerRegistry,
    model: &str,
    result: &Result<String, String>,
) {
    let last = failures.load_ended(model);
    match result {
        Ok(_) => {
            failures.clear(model);
        }
        Err(_) if !last || registry.contains(model) => {}
        Err(error) if !workers.contains_model(model) => {
            debug!(
                name = %model,
                error = %error,
                "Tokenizer load failed for a name no served model carries; nothing to record"
            );
        }
        Err(error) => {
            if failures.record(model, TokenizerLoadFailure::Failed(error.clone())) {
                warn!(
                    model = %model,
                    error = %error,
                    "Tokenizer load failed for a served model; its requests answer 503 tokenizer_load_failed until a tokenizer is registered"
                );
            }
        }
    }
}

/// Notes the removal of the tokenizer `name` (`DELETE /v1/tokenizers/{id}`):
/// for a model still served nothing reloads it on its own, so the model's
/// requests are answered 503 `tokenizer_load_failed` without a retry hint
/// until a tokenizer registers; logged once per model and entry.
pub fn note_tokenizer_removed(
    failures: &TokenizerLoadFailures,
    workers: &WorkerRegistry,
    name: &str,
) {
    if workers.contains_model(name) && failures.record(name, TokenizerLoadFailure::Removed) {
        warn!(
            model = %name,
            "Tokenizer removed for a served model; its requests answer 503 tokenizer_load_failed until a tokenizer is registered"
        );
    }
}

// ============================================================================
// Workflow Steps
// ============================================================================

/// Load the tokenizer from source (local path or HuggingFace)
///
/// This step handles:
/// - Input validation (via registry.load())
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Optional caching layer wrapping
pub struct LoadTokenizerStep;

#[async_trait]
impl StepExecutor<TokenizerWorkflowData> for LoadTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<TokenizerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        info!(
            "Loading tokenizer '{}' (id: {}) from source: {}{}",
            config.name,
            config.id,
            config.source,
            if config.cache_config.is_some() {
                " with caching"
            } else {
                ""
            }
        );

        // Clone needed values before async move
        let id = config.id.clone();
        let name = config.name.clone();
        let source = config.source.clone();
        let chat_template = config.chat_template_path.clone();
        let cache_config = config.cache_config.clone();

        // Load the tokenizer using the registry's load method
        // This handles: validation, deduplication, and loading
        let result = app_context
            .tokenizer_registry
            .load(&id, &name, &source, || {
                let source = source.clone();
                let chat_template = chat_template.clone();
                let cache_cfg = cache_config.clone();
                let app_context = app_context.clone();
                let name = name.clone();
                let id = id.clone();
                async move {
                    let base_tokenizer = match factory::create_tokenizer_async_with_chat_template(
                        &source,
                        chat_template.as_deref(),
                    )
                    .await
                    {
                        Ok(tok) => tok,
                        Err(local_err) => {
                            debug!(
                                "Local tokenizer load failed for source '{}', attempting to fetch from worker. Error: {:?}",
                                source, local_err
                            );

                            fetch_tokenizer_from_worker(&app_context, &id, &name, chat_template.as_deref()).await.map_err(
                                |worker_err| {
                                    format!(
                                        "Failed to load tokenizer locally ({local_err}) and remotely from worker ({worker_err})"
                                    )
                                },
                            )?
                        }
                    };

                    Ok(with_optional_cache(base_tokenizer, cache_cfg))
                }
            })
            .await;

        match result {
            Ok(outcome) => {
                let loaded_id = outcome.id();

                // Get vocab size for logging
                let vocab_size = app_context
                    .tokenizer_registry
                    .get_by_id(loaded_id)
                    .map(|e| e.tokenizer.vocab_size());

                match &outcome {
                    LoadOutcome::Loaded { id } => {
                        info!(
                            "Successfully loaded tokenizer '{}' (id: {}) with vocab_size: {:?}",
                            name, id, vocab_size
                        );
                        hint_response_template(&app_context, &name, id);
                    }
                    LoadOutcome::AlreadyExists { id } => {
                        if config.fail_on_duplicate {
                            return Err(WorkflowError::StepFailed {
                                step_id: StepId::new("load_tokenizer"),
                                message: format!("Tokenizer '{name}' already exists (id: {id})"),
                            });
                        }
                        info!(
                            "Tokenizer '{}' already exists (id: {}), skipping load",
                            name, id
                        );
                    }
                }

                // Store vocab size in typed data
                if let Some(size) = vocab_size {
                    context.data.vocab_size = Some(size);
                }

                Ok(StepResult::Success)
            }
            Err(e) => {
                error!("Failed to load tokenizer '{}': {}", name, e);
                Err(WorkflowError::StepFailed {
                    step_id: StepId::new("load_tokenizer"),
                    message: e.to_string(),
                })
            }
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Network/IO errors are retryable
    }
}

fn with_optional_cache(
    tokenizer: Arc<dyn Tokenizer>,
    cache_cfg: Option<TokenizerCacheConfig>,
) -> Arc<dyn Tokenizer> {
    match cache_cfg {
        Some(cfg) if cfg.enable_l0 || cfg.enable_l1 => {
            let cache_config = CacheConfig {
                enable_l0: cfg.enable_l0,
                l0_max_entries: cfg.l0_max_entries,
                l0_max_memory: cfg.l0_max_memory,
                enable_l1: cfg.enable_l1,
                l1_max_memory: cfg.l1_max_memory,
            };
            Arc::new(CachedTokenizer::new(tokenizer, cache_config))
        }
        _ => tokenizer,
    }
}

fn load_tokenizer_from_bundle(
    bundle: &StreamBundle,
    chat_template_path: Option<&str>,
) -> Result<(Arc<dyn Tokenizer>, Option<MultimodalModelConfig>), String> {
    tokenizer_bundle::with_extracted_bundle(bundle, |tokenizer_dir| {
        let tokenizer_path = tokenizer_dir.to_string_lossy().into_owned();
        info!(
            "Tokenizer extracted from temporary path: {}",
            tokenizer_path
        );

        let tokenizer =
            factory::create_tokenizer_with_chat_template(&tokenizer_path, chat_template_path)
                .map_err(|e| format!("tokenizer load failed: {e}"))?;

        let mm_config = try_load_multimodal_config(tokenizer_dir);

        Ok((tokenizer, mm_config))
    })
    .map_err(|e| format!("bundle extraction/load failed: {e}"))
}

/// Best-effort read of `config.json` + `preprocessor_config.json` from a
/// tokenizer bundle. Returns `None` only if `config.json` is missing or
/// unparsable; `preprocessor_config.json` is optional because each image
/// processor supplies its own model-specific defaults.
fn try_load_multimodal_config(tokenizer_dir: &std::path::Path) -> Option<MultimodalModelConfig> {
    let config_path = tokenizer_dir.join("config.json");
    if !config_path.exists() {
        debug!("Bundle has no config.json; skipping multimodal config preload");
        return None;
    }

    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            warn!("Failed to read config.json from bundle: {e}");
            return None;
        }
    };
    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse config.json from bundle: {e}");
            return None;
        }
    };

    let preprocessor_config = load_image_preprocessor_config(tokenizer_dir).unwrap_or_else(|| {
        debug!("No image preprocessor config in bundle; using defaults");
        llm_multimodal::PreProcessorConfig::default()
    });
    let video_preprocessor_config = load_video_preprocessor_config(tokenizer_dir);

    Some(MultimodalModelConfig {
        config,
        preprocessor_config,
        video_preprocessor_config,
    })
}

/// Fetch a tokenizer from a healthy gRPC worker when local loading fails.
async fn fetch_tokenizer_from_worker(
    app_context: &AppContext,
    tokenizer_id: &str,
    model_id: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn Tokenizer>, String> {
    let workers = app_context.worker_registry.get_workers_filtered(
        Some(model_id),
        None,
        Some(ConnectionMode::Grpc),
        None,
        true,
    );

    if workers.is_empty() {
        return Err(format!(
            "No healthy gRPC worker available to fetch tokenizer for model '{model_id}'"
        ));
    }

    let mut failures = Vec::new();
    let worker_count = workers.len();

    for worker in workers {
        let runtime = worker.metadata().spec.runtime_type;
        info!(
            "Fetching tokenizer from worker: {} (runtime: {})",
            worker.url(),
            runtime
        );

        let backend_client = match worker.get_backend_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                failures.push(format!(
                    "worker {} (runtime: {runtime}) does not provide a gRPC client",
                    worker.url(),
                ));
                continue;
            }
            Err(e) => {
                failures.push(format!(
                    "failed to create gRPC client for worker {} (runtime: {runtime}): {e}",
                    worker.url(),
                ));
                continue;
            }
        };

        let bundle = match backend_client.get_tokenizer().await {
            Ok(bundle) => bundle,
            Err(e) => {
                let is_unimplemented = e
                    .downcast_ref::<tonic::Status>()
                    .is_some_and(|s| s.code() == tonic::Code::Unimplemented);
                if is_unimplemented {
                    debug!(
                        "Worker {} (runtime: {}) does not support GetTokenizer",
                        worker.url(),
                        runtime
                    );
                } else {
                    warn!(
                        "Worker {} (runtime: {}) get_tokenizer failed: {}",
                        worker.url(),
                        runtime,
                        e
                    );
                }
                failures.push(format!(
                    "worker {} (runtime: {runtime}) get_tokenizer failed: {e}",
                    worker.url(),
                ));
                continue;
            }
        };
        debug!(
            "Tokenizer bundle validation passed for worker {}",
            worker.url()
        );

        match load_tokenizer_from_bundle(&bundle, chat_template_path) {
            Ok((tokenizer, mm_config)) => {
                if let Some(cfg) = mm_config {
                    app_context
                        .multimodal_config_registry
                        .insert(tokenizer_id.to_string(), Arc::new(cfg));
                    info!(
                        tokenizer_id = %tokenizer_id,
                        tokenizer_name = %model_id,
                        "Preloaded multimodal config from GetTokenizer bundle"
                    );
                }
                return Ok(tokenizer);
            }
            Err(e) => failures.push(format!(
                "worker {} (runtime: {runtime}) bundle extraction/load failed: {e}",
                worker.url(),
            )),
        };
    }

    let failures_summary = failures.join("; ");
    Err(format!(
        "Failed to fetch tokenizer for model '{model_id}' from {worker_count} healthy gRPC worker(s): {failures_summary}"
    ))
}

// ============================================================================
// Workflow Builder
// ============================================================================

/// A tokenizer that ships a `response_template` is read with its table only when the parser
/// flags name `hf` (see `routers::grpc::regular::symphony`): says so once per tokenizer when
/// they do not, so the choice is a known one.
fn hint_response_template(app_context: &AppContext, model: &str, tokenizer_id: &str) {
    let ships_template = app_context
        .tokenizer_registry
        .get_by_id(tokenizer_id)
        .is_some_and(|entry| entry.tokenizer.response_template().is_some());
    let names_template = [
        &app_context.configured_tool_parser,
        &app_context.configured_reasoning_parser,
    ]
    .into_iter()
    .flatten()
    .any(|name| name.trim().eq_ignore_ascii_case(TEMPLATE_PARSER));
    if ships_template && !names_template {
        info!(
            model,
            "the tokenizer ships a response_template; --tool-call-parser {TEMPLATE_PARSER} and \
             --reasoning-parser {TEMPLATE_PARSER} read with it under --parser symphony"
        );
    }
}

/// Create the tokenizer registration workflow
///
/// This workflow loads and registers a tokenizer. The single LoadTokenizerStep handles:
/// - Input validation (empty name/source)
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Optional caching layer wrapping
///
/// Configuration:
/// - 3 retries with 2s backoff (for network issues)
/// - 5 minute timeout (HuggingFace downloads can be slow)
pub fn create_tokenizer_registration_workflow() -> WorkflowDefinition<TokenizerWorkflowData> {
    WorkflowDefinition::new("tokenizer_registration", "Tokenizer Registration").add_step(
        StepDefinition::new(
            "load_tokenizer",
            "Load Tokenizer",
            Arc::new(LoadTokenizerStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Fixed(Duration::from_secs(2)),
        })
        .with_timeout(Duration::from_secs(300)) // 5 min for HuggingFace downloads
        .with_failure_action(FailureAction::FailWorkflow),
    )
}

/// Helper to create initial workflow data for tokenizer registration
pub fn create_tokenizer_workflow_data(
    config: TokenizerConfigRequest,
    app_context: Arc<AppContext>,
) -> TokenizerWorkflowData {
    TokenizerWorkflowData {
        config,
        vocab_size: None,
        app_context: Some(app_context),
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, WorkerType},
    };

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    #[test]
    fn worker_tokenizer_bundle_preserves_configured_template() {
        use std::io::{Cursor, Write};

        use sha2::{Digest, Sha256};
        use zip::{write::SimpleFileOptions, ZipWriter};

        let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data) in [
            (
                "tokenizer.json",
                serde_json::json!({
                    "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                    "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
                    "post_processor":null, "decoder":null,
                    "model":{"type":"BPE", "vocab":{"hello":0}, "merges":[]}
                }),
            ),
            (
                "tokenizer_config.json",
                serde_json::json!({"chat_template":"worker_default"}),
            ),
        ] {
            archive
                .start_file(name, SimpleFileOptions::default())
                .unwrap();
            archive.write_all(data.to_string().as_bytes()).unwrap();
        }
        let compressed_data = archive.finish().unwrap().into_inner();
        let bundle = StreamBundle {
            sha256: Sha256::digest(&compressed_data)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            compressed_data,
        };
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("override.jinja");
        std::fs::write(&template, "router_override").unwrap();
        let messages = [serde_json::json!({"role":"user", "content":"hello"})];
        let (default, _) = load_tokenizer_from_bundle(&bundle, None).unwrap();
        assert_eq!(
            default
                .apply_chat_template(&messages, Default::default())
                .unwrap(),
            "worker_default"
        );
        let (overridden, _) = load_tokenizer_from_bundle(&bundle, template.to_str()).unwrap();
        assert_eq!(
            overridden
                .apply_chat_template(&messages, Default::default())
                .unwrap(),
            "router_override"
        );
        let missing = dir.path().join("missing.jinja");
        assert!(
            load_tokenizer_from_bundle(&bundle, missing.to_str()).is_err(),
            "an invalid configured override must not silently use the worker default"
        );
    }

    #[test]
    fn test_tokenizer_config_request_serialization() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            cache_config: None,
            fail_on_duplicate: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-uuid-1234");
        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
        assert!(parsed.chat_template_path.is_none());
        assert!(parsed.cache_config.is_none());
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_defaults_to_false() {
        // Test that fail_on_duplicate defaults to false when not specified in JSON
        let json = r#"{
            "id": "test-uuid",
            "name": "test-model",
            "source": "/path/to/tokenizer"
        }"#;
        let parsed: TokenizerConfigRequest = serde_json::from_str(json).unwrap();
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_true() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            cache_config: None,
            fail_on_duplicate: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();
        assert!(parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_with_cache() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            cache_config: Some(TokenizerCacheConfig {
                enable_l0: true,
                l0_max_entries: 1000,
                l0_max_memory: 256 * 1024 * 1024,
                enable_l1: false,
                l1_max_memory: 0,
            }),
            fail_on_duplicate: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert!(parsed.cache_config.is_some());
        let cache = parsed.cache_config.unwrap();
        assert!(cache.enable_l0);
        assert_eq!(cache.l0_max_entries, 1000);
        assert!(!cache.enable_l1);
    }

    #[test]
    fn test_workflow_creation() {
        let mut workflow = create_tokenizer_registration_workflow();
        assert_eq!(workflow.id.to_string(), "tokenizer_registration");
        // Validate the workflow DAG
        workflow
            .validate()
            .expect("Workflow validation should pass");
    }

    /// A worker registry serving `model`.
    fn serving(model: &str) -> WorkerRegistry {
        let workers = WorkerRegistry::new();
        workers
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://worker:30000")
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .model(ModelCard::new(model))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .expect("register worker");
        workers
    }

    /// A failed load is recorded once per model and entry (the log line's
    /// allowance), a different reason is a new entry, and the next load,
    /// accepted or succeeding, clears the record.
    #[test]
    fn a_load_failure_is_recorded_once_per_entry_and_cleared_by_the_next_load() {
        let failures = TokenizerLoadFailures::new();
        let registry = TokenizerRegistry::new();
        let workers = serving("m");
        let error = "Workflow failed at step load_tokenizer: boom".to_string();
        assert_eq!(failures.get("m"), None);

        failures.load_started("m");
        assert_eq!(failures.get("m"), None);
        note_tokenizer_load_outcome(&failures, &registry, &workers, "m", &Err(error.clone()));
        assert_eq!(
            failures.get("m"),
            Some(TokenizerLoadFailure::Failed(error.clone()))
        );
        assert!(!failures.record("m", TokenizerLoadFailure::Failed(error.clone())));
        assert!(failures.record("m", TokenizerLoadFailure::NoSource));
        assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::NoSource));
        assert_eq!(failures.get("other"), None);

        failures.load_started("m");
        assert_eq!(failures.get("m"), None, "a load accepted: pending again");
        note_tokenizer_load_outcome(
            &failures,
            &registry,
            &workers,
            "m",
            &Ok("m completed".to_string()),
        );
        assert_eq!(failures.get("m"), None);
        assert!(!failures.clear("m"));
    }

    /// While another load of the model is in flight, neither a registration
    /// without a path nor an earlier load's failure is recorded: the last
    /// load decides.
    #[test]
    fn a_failure_is_not_recorded_while_another_load_of_the_model_is_in_flight() {
        let failures = TokenizerLoadFailures::new();
        let registry = TokenizerRegistry::new();
        let workers = serving("m");

        failures.load_started("m");
        assert!(!failures.record("m", TokenizerLoadFailure::NoSource));
        assert_eq!(failures.get("m"), None);

        failures.load_started("m");
        note_tokenizer_load_outcome(
            &failures,
            &registry,
            &workers,
            "m",
            &Err("first".to_string()),
        );
        assert_eq!(failures.get("m"), None, "a load is still in flight");
        note_tokenizer_load_outcome(
            &failures,
            &registry,
            &workers,
            "m",
            &Err("second".to_string()),
        );
        assert_eq!(
            failures.get("m"),
            Some(TokenizerLoadFailure::Failed("second".to_string()))
        );
    }

    /// A load that ended without a count (none is taken outside the queue's
    /// hand-over) leaves no state behind, and the model records as before.
    #[test]
    fn a_load_that_ended_without_a_count_leaves_nothing_behind() {
        let failures = TokenizerLoadFailures::new();
        assert!(failures.load_ended("m"));
        assert_eq!(failures.get("m"), None);
        assert!(failures.record("m", TokenizerLoadFailure::NoSource));
        assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::NoSource));
        failures.load_started("m");
        assert_eq!(
            failures.get("m"),
            None,
            "pending while the load is in flight"
        );
        assert!(failures.load_ended("m"));
        assert_eq!(
            failures.get("m"),
            None,
            "the load that ended left no record"
        );
    }

    /// A failed load of a name no served model carries (the start-up load,
    /// named by its source; an operator's registration of another name) is
    /// nobody's request: nothing is recorded.
    #[test]
    fn a_failed_load_of_a_name_nobody_serves_is_not_recorded() {
        let failures = TokenizerLoadFailures::new();
        let registry = TokenizerRegistry::new();
        let workers = WorkerRegistry::new();
        failures.load_started("/models/x");
        note_tokenizer_load_outcome(
            &failures,
            &registry,
            &workers,
            "/models/x",
            &Err("boom".to_string()),
        );
        assert_eq!(failures.get("/models/x"), None);
    }

    /// A registration that fails for a model whose tokenizer the registry
    /// holds (a duplicate) takes nothing away and is not recorded.
    #[tokio::test]
    async fn a_failed_duplicate_registration_of_a_registered_tokenizer_is_no_failure() {
        let failures = TokenizerLoadFailures::new();
        let registry = TokenizerRegistry::new();
        let workers = serving("m");
        registry
            .load("id", "m", "mock", || async {
                factory::create_tokenizer_with_chat_template("mock", None)
                    .map_err(|e| e.to_string())
            })
            .await
            .expect("the mock tokenizer loads");
        failures.load_started("m");
        note_tokenizer_load_outcome(
            &failures,
            &registry,
            &workers,
            "m",
            &Err("Tokenizer 'm' already exists (id: id)".to_string()),
        );
        assert_eq!(failures.get("m"), None);
    }

    /// A tokenizer removed while its model is served is recorded, once; one
    /// of a name nobody serves is not.
    #[test]
    fn a_removed_tokenizer_is_recorded_only_for_a_served_model() {
        let failures = TokenizerLoadFailures::new();
        let workers = serving("m");
        note_tokenizer_removed(&failures, &workers, "m");
        assert_eq!(failures.get("m"), Some(TokenizerLoadFailure::Removed));
        assert!(!failures.record("m", TokenizerLoadFailure::Removed));
        note_tokenizer_removed(&failures, &workers, "other");
        assert_eq!(failures.get("other"), None);
    }
}
