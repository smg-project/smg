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
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    worker::WorkerStatus,
};
use tracing::{debug, error, info, warn};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::{
    data::{require_app_context, TokenizerWorkflowData},
    tokenizer_shares::{self, FilesRead},
};
use crate::{
    app_context::AppContext,
    config::{TokenizerCacheConfig, TEMPLATE_PARSER},
    routers::{
        common::snapshot::WorkerRegistry,
        grpc::{
            backend_client::get_backend_client,
            multimodal::{
                load_image_preprocessor_config, load_video_preprocessor_config,
                MultimodalModelConfig,
            },
        },
    },
    workflow::steps::local::ReleaseTokenizerStep,
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
    /// Whether a worker's registration asked for the load: an entry it loads
    /// afresh is the worker's (recorded by the loader under the registry's
    /// per-name lock, before the entry is published), released with its
    /// model's last worker, by the load itself when that worker left while
    /// it ran; the startup entry's and an operator's requests leave this
    /// false.
    #[serde(skip)]
    pub for_worker: bool,
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
    /// Nothing to load from: no `tokenizer_path`, `model_path` or `model_id`
    /// among the worker's labels, and no `--tokenizer-path` or
    /// `--model-path` in the router configuration.
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
                "no tokenizer source in the worker's labels (tokenizer_path, model_path, model_id) or the router configuration"
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
        let app_context = require_app_context(context.data.app_context.as_ref())?.clone();

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
        let for_worker = config.for_worker;

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
                    let tokenizer = load_tokenizer(
                        &app_context,
                        &id,
                        &name,
                        &source,
                        chat_template.as_deref(),
                        cache_cfg,
                    )
                    .await?;
                    // Recorded under the registry's per-name lock, before the
                    // entry is published: a reader that sees the entry sees
                    // the record.
                    if for_worker {
                        app_context.worker_loaded_tokenizers.insert(id);
                    }
                    Ok(tokenizer)
                }
            })
            .await;

        match result {
            Ok(outcome) => {
                let vocab_size = note_load_outcome(&app_context, config, &outcome)?;
                // A worker's load looks at the worker registry again once it
                // has published: the removal of the model's last worker while
                // the load ran found no entry to take.
                if for_worker && matches!(outcome, LoadOutcome::Loaded { .. }) {
                    ReleaseTokenizerStep::release_if_unserved(&app_context, &name);
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

/// The tokenizer of `source` with its chat template, the loader the registry
/// runs for [`LoadTokenizerStep`]: the local path or the hub first, a serving
/// worker's bundle when that fails, wrapped in the cache when one is
/// configured.
async fn load_tokenizer(
    app_context: &AppContext,
    tokenizer_id: &str,
    model_id: &str,
    source: &str,
    chat_template: Option<&str>,
    cache_cfg: Option<TokenizerCacheConfig>,
) -> Result<Arc<dyn Tokenizer>, String> {
    let base_tokenizer = match load_from_source(source, chat_template).await {
        Ok(tok) => tok,
        Err(local_err) => {
            debug!(
                    "Local tokenizer load failed for source '{}', attempting to fetch from worker. Error: {:?}",
                    source, local_err
                );

            fetch_tokenizer_from_worker(app_context, tokenizer_id, model_id, chat_template)
                    .await
                    .map_err(|worker_err| {
                        format!(
                            "Failed to load tokenizer locally ({local_err}) and remotely from worker ({worker_err})"
                        )
                    })?
        }
    };

    Ok(with_optional_cache(base_tokenizer, cache_cfg))
}

/// The tokenizer the factory makes of `source` and `chat_template`, or the
/// one another name handed the same source already holds for the same files
/// (see [`tokenizer_shares`]): a name over files a tokenizer was loaded from
/// reads them once more to know that, and parses nothing.
async fn load_from_source(
    source: &str,
    chat_template: Option<&str>,
) -> anyhow::Result<Arc<dyn Tokenizer>> {
    let read = {
        let (source, chat_template) = (source.to_owned(), chat_template.map(str::to_owned));
        move || FilesRead::of(&source, chat_template.as_deref())
    };
    tokenizer_shares::shares()
        .load(read, || {
            factory::create_tokenizer_async_with_chat_template(source, chat_template)
        })
        .await
}

/// What a load that ended means for the step: a tokenizer loaded is logged
/// with the template hint; one the registry already held is logged, or is
/// the failure the request asks for on a duplicate. Returns the vocabulary
/// size of the tokenizer under the outcome's id.
fn note_load_outcome(
    app_context: &AppContext,
    config: &TokenizerConfigRequest,
    outcome: &LoadOutcome,
) -> WorkflowResult<Option<usize>> {
    let name = &config.name;

    // Get vocab size for logging
    let vocab_size = app_context
        .tokenizer_registry
        .get_by_id(outcome.id())
        .map(|e| e.tokenizer.vocab_size());

    match outcome {
        LoadOutcome::Loaded { id } => {
            info!(
                "Successfully loaded tokenizer '{}' (id: {}) with vocab_size: {:?}",
                name, id, vocab_size
            );
            hint_response_template(app_context, name, id);
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

    Ok(vocab_size)
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

    Some(MultimodalModelConfig::new(
        config,
        preprocessor_config,
        video_preprocessor_config,
    ))
}

/// Lists the model's ready gRPC workers, then its pending ones. A newly
/// registered worker can serve its tokenizer before health checks promote it.
fn tokenizer_workers(registry: &WorkerRegistry, model_id: &str) -> Vec<Arc<dyn Worker>> {
    let mut ready = Vec::new();
    let mut pending = Vec::new();
    for worker in registry.get_by_model(model_id).iter() {
        if *worker.connection_mode() != ConnectionMode::Grpc {
            continue;
        }
        match worker.status() {
            WorkerStatus::Ready => ready.push(Arc::clone(worker)),
            WorkerStatus::Pending => pending.push(Arc::clone(worker)),
            WorkerStatus::NotReady | WorkerStatus::Failed | WorkerStatus::Draining => {}
        }
    }
    ready.extend(pending);
    ready
}

/// Fetch a tokenizer from a serving gRPC worker when local loading fails,
/// trying sources in the order returned by [`tokenizer_workers`].
async fn fetch_tokenizer_from_worker(
    app_context: &AppContext,
    tokenizer_id: &str,
    model_id: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn Tokenizer>, String> {
    let workers = tokenizer_workers(&app_context.worker_registry, model_id);

    if workers.is_empty() {
        return Err(format!(
            "No ready or pending gRPC worker available to fetch tokenizer for model '{model_id}'"
        ));
    }

    // A failed job keeps this error, so it names the first failure alone: a
    // fleet that cannot serve the tokenizer would name every worker in each.
    // For the same reason a fetch warns of its first bundle that fails to
    // load, not of each.
    let mut first_failure: Option<String> = None;
    let mut load_warned = false;
    let mut keep_first = |failure: String| {
        if first_failure.is_none() {
            first_failure = Some(failure);
        } else {
            debug!("Tokenizer fetch for model '{model_id}' failed on a further worker: {failure}");
        }
    };
    let worker_count = workers.len();

    for worker in workers {
        let runtime = worker.metadata().spec.runtime_type;
        info!(
            "Fetching tokenizer from worker: {} (runtime: {})",
            worker.redacted_url(),
            runtime
        );

        let bundle = match fetch_bundle(worker.as_ref()).await {
            Ok(bundle) => bundle,
            Err(failure) => {
                keep_first(failure);
                continue;
            }
        };

        match install_bundle(
            app_context,
            tokenizer_id,
            model_id,
            bundle,
            chat_template_path,
        )
        .await
        {
            Ok(tokenizer) => return Ok(tokenizer),
            Err(e) => {
                let failure = format!("worker {} (runtime: {runtime}) {e}", worker.redacted_url());
                if !load_warned {
                    load_warned = true;
                    warn!("Tokenizer fetch for model '{model_id}': {failure}");
                }
                keep_first(failure);
            }
        };
    }

    let first_failure = first_failure.unwrap_or_default();
    Err(format!(
        "Failed to fetch tokenizer for model '{model_id}' from {worker_count} ready or pending gRPC worker(s), first failure: {first_failure}"
    ))
}

/// The tokenizer bundle `worker` serves over its gRPC client, or the failure
/// the fetch reports for it: no gRPC client, a client that fails to build, a
/// worker that does not implement the call or fails it.
async fn fetch_bundle(worker: &dyn Worker) -> Result<StreamBundle, String> {
    let runtime = worker.metadata().spec.runtime_type;

    let backend_client = match get_backend_client(worker).await {
        Ok(Some(client)) => client,
        Ok(None) => {
            return Err(format!(
                "worker {} (runtime: {runtime}) does not provide a gRPC client",
                worker.redacted_url(),
            ));
        }
        Err(e) => {
            return Err(format!(
                "failed to create gRPC client for worker {} (runtime: {runtime}): {e}",
                worker.redacted_url(),
            ));
        }
    };

    let bundle = backend_client.get_tokenizer().await.map_err(|e| {
        let is_unimplemented = e
            .downcast_ref::<tonic::Status>()
            .is_some_and(|s| s.code() == tonic::Code::Unimplemented);
        if is_unimplemented {
            debug!(
                "Worker {} (runtime: {}) does not support GetTokenizer",
                worker.redacted_url(),
                runtime
            );
        } else {
            warn!(
                "Worker {} (runtime: {}) get_tokenizer failed: {}",
                worker.redacted_url(),
                runtime,
                e
            );
        }
        format!(
            "worker {} (runtime: {runtime}) get_tokenizer failed: {e}",
            worker.redacted_url(),
        )
    })?;
    debug!(
        "Tokenizer bundle validation passed for worker {}",
        worker.redacted_url()
    );

    Ok(bundle)
}

/// The tokenizer of a bundle a worker served, loaded with the chat template;
/// a multimodal configuration the bundle carries is registered under the
/// tokenizer's id on the way. The bundle is unpacked and parsed on the
/// blocking pool (see [`on_the_pool`]), as the factory parses a source.
async fn install_bundle(
    app_context: &AppContext,
    tokenizer_id: &str,
    model_id: &str,
    bundle: StreamBundle,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn Tokenizer>, String> {
    let chat_template_path = chat_template_path.map(str::to_owned);
    let (tokenizer, mm_config) =
        on_the_pool(move || load_tokenizer_from_bundle(&bundle, chat_template_path.as_deref()))
            .await?;
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
    Ok(tokenizer)
}

/// `work` on the blocking pool. A panic of it unwinds the caller, as it did
/// when the work ran on the step's task: the engine ends the attempt as a
/// panic, with no retry.
async fn on_the_pool<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|ended| std::panic::resume_unwind(ended.into_panic()))
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
#[path = "../../tests/unit/workflow/tokenizer_registration/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "../../tests/unit/workflow/tokenizer_registration/tests.rs"]
mod tests;
