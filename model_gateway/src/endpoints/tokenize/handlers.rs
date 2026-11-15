//! Tokenize and detokenize handlers
//!
//! Provides tokenization, detokenization, and tokenizer management operations.
//! These handlers use the TokenizerRegistry for tokenizer storage and retrieval.
//! A model named by an alias tokenizes as the model the worker registry
//! resolves it to, as every serving route reads it. The management routes
//! refuse in the control plane's envelope (`{"error", "code"}`), the shape
//! the worker routes use; an empty or whitespace name or source is refused
//! up front, before a load job is queued for it. The tokenize and detokenize
//! routes refuse in the gateway's public envelope, as the model routes do (an
//! unknown model is a `404 model_not_found`); a body that names no model is
//! served by the tokenizer of the one model the workers serve, else by the
//! one loaded tokenizer, and refused as `missing_model` when neither picks
//! one.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use llm_tokenizer::{registry::TokenizerEntry, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::tokenize::{
    AddTokenizerRequest, AddTokenizerResponse, CountResult, DetokenizeRequest, DetokenizeResponse,
    ListTokenizersResponse, RemoveTokenizerResponse, TextResult, TokenizeRequest, TokenizeResponse,
    TokenizerInfo, TokensResult,
};
use smg_auth::AuditResource;
use tracing::{debug, error, warn};

use crate::{
    admin::json::refusal,
    app_context::AppContext,
    routers::{
        error,
        grpc::utils::{encode_blocking_in_batch, InPlaceBudget},
    },
    worker::{WorkerRegistry, UNKNOWN_MODEL_ID},
    workflow::{note_tokenizer_removed, Job, TokenizerConfigRequest},
};

/// The tokenizer that serves a tokenize or detokenize body, and the model it
/// is the tokenizer of (whose workers' context window a tokenize answer
/// reports): the name or id as given, then the model the worker registry
/// resolves an alias to (the name the serving routes use). A body that names
/// no model (absent, empty or the unknown model) is the one served model's
/// ([`WorkerRegistry::sole_served_model`], as the gRPC router serves such a
/// body) when a tokenizer is loaded under its name, beside whatever else is
/// loaded (a single-model gateway holds the startup load under its path and
/// the worker's under the served name); else it is the one loaded
/// tokenizer's. With several loaded and no served model's among them it
/// names none of them, and is refused as `missing_model` rather than read
/// with whichever vocabulary sorts first. A name no tokenizer serves is a
/// `404 model_not_found`.
fn get_tokenizer(
    workers: &WorkerRegistry,
    registry: &TokenizerRegistry,
    model: &str,
) -> Result<(Arc<dyn Tokenizer>, String), Response> {
    // First, try exact match (by name or ID)
    if let Some(tokenizer) = registry.get(model) {
        debug!("Found tokenizer for model: {}", model);
        return Ok((tokenizer, model.to_owned()));
    }

    // An alias tokenizes as the model it resolves to
    if let Some(canonical) = workers.resolve_model_alias(model) {
        if let Some(tokenizer) = registry.get(&canonical) {
            debug!(
                "Found tokenizer for model: {} (alias of {})",
                model, canonical
            );
            return Ok((tokenizer, canonical.to_string()));
        }
    }

    // A body that names no model is the one served model's
    let model_less = model == UNKNOWN_MODEL_ID || model.is_empty();
    if model_less {
        if let Some(served) = workers.sole_served_model() {
            if let Some(tokenizer) = registry.get(&served) {
                debug!("Using the one served model's tokenizer '{}'", served);
                return Ok((tokenizer, served));
            }
        }
    }

    let entries = registry.list();
    let names = || {
        entries
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    // ... else the one loaded tokenizer's
    if model_less {
        match entries.as_slice() {
            [only] => {
                debug!("Using the one loaded tokenizer '{}' as default", only.name);
                return Ok((only.tokenizer.clone(), only.name.clone()));
            }
            [] => {}
            _ => {
                return Err(error::bad_request(
                    "missing_model",
                    format!(
                        "model is required while more than one tokenizer is loaded: name one of {}",
                        names()
                    ),
                ));
            }
        }
    }

    // List available tokenizers for error message
    let message = if entries.is_empty() {
        "No tokenizers available. Use POST /v1/tokenizers to add one.".to_string()
    } else {
        format!(
            "Tokenizer for model '{}' not found. Available: {}",
            model,
            names()
        )
    };
    Err(error::not_found("model_not_found", message))
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

/// Handle POST /v1/tokenize
pub async fn tokenize(context: &Arc<AppContext>, request: TokenizeRequest) -> Response {
    debug!("Tokenize request for model: {}", request.model);

    let (tokenizer, model) = match get_tokenizer(
        &context.worker_registry,
        &context.tokenizer_registry,
        &request.model,
    ) {
        Ok(found) => found,
        Err(refusal) => return refusal,
    };

    let texts = request.prompt.as_strings();
    let is_batch = request.prompt.is_batch();

    // Tokenize each text
    let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
    let mut all_counts: Vec<i32> = Vec::with_capacity(texts.len());
    let mut all_char_counts: Vec<i32> = Vec::with_capacity(texts.len());

    let mut budget = InPlaceBudget::default();
    for text in texts {
        // Don't add special tokens for tokenize API (matches Python behavior)
        let encoding =
            match encode_blocking_in_batch(tokenizer.clone(), text, false, &mut budget).await {
                Ok(enc) => enc,
                Err(e) => {
                    error!("Tokenization failed: {}", e);
                    return error::internal_error(
                        "tokenization_error",
                        format!("Tokenization failed: {e}"),
                    );
                }
            };

        let token_ids: Vec<u32> = encoding.into_token_ids();
        let count = token_ids.len() as i32;

        all_tokens.push(token_ids);
        all_counts.push(count);
        all_char_counts.push(text.chars().count() as i32);
    }

    // Format response based on single vs batch
    let (tokens, count, char_count) = if is_batch {
        (
            TokensResult::Batch(all_tokens),
            CountResult::Batch(all_counts),
            CountResult::Batch(all_char_counts),
        )
    } else {
        (
            TokensResult::Single(all_tokens.into_iter().next().unwrap_or_default()),
            CountResult::Single(all_counts.into_iter().next().unwrap_or(0)),
            CountResult::Single(all_char_counts.into_iter().next().unwrap_or(0)),
        )
    };

    Json(TokenizeResponse {
        tokens,
        count,
        char_count,
        max_model_len: model_context_window(&context.worker_registry, &model),
    })
    .into_response()
}

/// The smallest context window the workers of `model` (an alias read as the
/// model it names) report, as SGLang's `/tokenize` answers its own as
/// `max_model_len`; -1, SGLang's value when there is none, when no worker
/// reports one.
fn model_context_window(workers: &WorkerRegistry, model: &str) -> i64 {
    let canonical = workers.resolve_model_alias(model);
    let model = canonical.as_deref().unwrap_or(model);
    workers
        .get_by_model(model)
        .iter()
        .filter_map(|worker| worker.context_length(model))
        .min()
        .map_or(-1, i64::from)
}

/// Handle POST /v1/detokenize
pub async fn detokenize(context: &Arc<AppContext>, request: DetokenizeRequest) -> Response {
    debug!("Detokenize request for model: {}", request.model);

    let (tokenizer, _) = match get_tokenizer(
        &context.worker_registry,
        &context.tokenizer_registry,
        &request.model,
    ) {
        Ok(found) => found,
        Err(refusal) => return refusal,
    };

    let sequences = request.tokens.sequences();
    let is_batch = request.tokens.is_batch();

    // Detokenize each sequence
    let mut all_texts: Vec<String> = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let text = match tokenizer.decode(seq, request.skip_special_tokens) {
            Ok(t) => t,
            Err(e) => {
                error!("Detokenization failed: {}", e);
                return error::internal_error(
                    "detokenization_error",
                    format!("Detokenization failed: {e}"),
                );
            }
        };
        all_texts.push(text);
    }

    // Format response based on single vs batch
    let text = if is_batch {
        TextResult::Batch(all_texts)
    } else {
        TextResult::Single(all_texts.into_iter().next().unwrap_or_default())
    };

    Json(DetokenizeResponse { text }).into_response()
}

// ============================================================================
// Tokenizer Management Handlers
// ============================================================================

/// Handle POST /v1/tokenizers - async version using job queue
pub async fn add_tokenizer(context: &Arc<AppContext>, request: AddTokenizerRequest) -> Response {
    // An empty or whitespace name is refused up front: the registry's load
    // refuses an empty one only once the job runs, after a 202, and takes a
    // whitespace one as a name.
    if request.name.trim().is_empty() {
        return refusal(
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            "A tokenizer name must not be empty".to_string(),
        );
    }

    // An empty or whitespace source is refused up front too: the registry's
    // load refuses an empty one only once the job runs, after a 202, and a
    // whitespace one fails its local load and falls back to a worker fetch
    // under the request's name.
    if request.source.trim().is_empty() {
        return refusal(
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            "A tokenizer source must not be empty".to_string(),
        );
    }

    // A name in use is refused; the error names the tokenizer that holds it.
    if let Some(entry) = context.tokenizer_registry.get_by_name(&request.name) {
        return refusal(
            StatusCode::CONFLICT,
            "TOKENIZER_ALREADY_EXISTS",
            format!(
                "Tokenizer '{}' already exists with id {}",
                request.name, entry.id
            ),
        );
    }

    let Some(job_queue) = context.worker_job_queue.get() else {
        error!("Job queue not available");
        return refusal(
            StatusCode::SERVICE_UNAVAILABLE,
            "SERVICE_UNAVAILABLE",
            "The job queue is not initialized".to_string(),
        );
    };

    // Generate UUID for this tokenizer
    let tokenizer_id = TokenizerRegistry::generate_id();

    // Create the job with the pre-generated ID
    // Note: API-initiated tokenizer loads don't use caching by default
    // Caching is applied for startup and worker-initiated loads based on router config
    let config = TokenizerConfigRequest {
        id: tokenizer_id.clone(),
        name: request.name.clone(),
        source: request.source.clone(),
        chat_template_path: request.chat_template_path.clone(),
        cache_config: None,
        fail_on_duplicate: true,
        for_worker: false,
    };

    let job = Job::AddTokenizer {
        config: Box::new(config),
    };

    // Submit the job
    match job_queue.submit(job).await {
        Ok(()) => AuditResource::name(
            (
                StatusCode::ACCEPTED,
                Json(AddTokenizerResponse {
                    id: tokenizer_id.clone(),
                    status: "pending".to_string(),
                    message: format!(
                        "Tokenizer '{}' registration job submitted. Loading from: {}",
                        request.name, request.source
                    ),
                    vocab_size: None,
                }),
            )
                .into_response(),
            tokenizer_id,
        ),
        Err(e) => {
            error!("Failed to submit tokenizer job: {}", e);
            refusal(StatusCode::SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE", e)
        }
    }
}

/// Handle GET /v1/tokenizers
pub async fn list_tokenizers(registry: &Arc<TokenizerRegistry>) -> Response {
    debug!("List tokenizers request");

    let entries = registry.list();
    let tokenizers: Vec<TokenizerInfo> = entries
        .into_iter()
        .map(|e| TokenizerInfo {
            id: e.id,
            name: e.name,
            source: e.source,
            vocab_size: e.tokenizer.vocab_size(),
        })
        .collect();

    Json(ListTokenizersResponse { tokenizers }).into_response()
}

/// Handle DELETE /v1/tokenizers/{tokenizer_id}
pub async fn remove_tokenizer(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    // Try to remove by ID first, then by name for backward compatibility
    let removed = context
        .tokenizer_registry
        .remove_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.remove(tokenizer_id));

    if let Some(entry) = removed {
        context.multimodal_config_registry.remove(&entry.id);
        context.worker_loaded_tokenizers.remove(&entry.id);
        note_tokenizer_removed(
            &context.tokenizer_load_failures,
            &context.worker_registry,
            &entry.name,
        );
        debug!("Removed tokenizer '{}' (id: {})", entry.name, entry.id);
        AuditResource::name(
            (
                StatusCode::OK,
                Json(RemoveTokenizerResponse {
                    success: true,
                    message: format!("Tokenizer '{}' removed successfully", entry.name),
                }),
            )
                .into_response(),
            entry.id,
        )
    } else {
        warn!("Tokenizer '{}' not found", tokenizer_id);
        tokenizer_not_found(tokenizer_id)
    }
}

/// 404 in the control plane's envelope: no tokenizer has that id or name.
fn tokenizer_not_found(tokenizer_id: &str) -> Response {
    refusal(
        StatusCode::NOT_FOUND,
        "TOKENIZER_NOT_FOUND",
        format!("Tokenizer '{tokenizer_id}' not found"),
    )
}

/// Handle GET /v1/tokenizers/{tokenizer_id}
pub async fn get_tokenizer_info(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    debug!("Get tokenizer info for '{}'", tokenizer_id);

    // Try by ID first, then by name
    let entry: Option<TokenizerEntry> = context
        .tokenizer_registry
        .get_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.get_by_name(tokenizer_id));

    match entry {
        Some(e) => {
            let info = TokenizerInfo {
                id: e.id,
                name: e.name,
                source: e.source,
                vocab_size: e.tokenizer.vocab_size(),
            };
            Json(info).into_response()
        }
        None => tokenizer_not_found(tokenizer_id),
    }
}

/// Handle GET /v1/tokenizers/{tokenizer_id}/status
pub async fn get_tokenizer_status(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    debug!("Get tokenizer status for '{}'", tokenizer_id);

    // First check if tokenizer is already loaded (by ID or name)
    let entry = context
        .tokenizer_registry
        .get_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.get_by_name(tokenizer_id));

    if let Some(e) = entry {
        return Json(AddTokenizerResponse {
            id: e.id,
            status: "completed".to_string(),
            message: format!("Tokenizer '{}' is loaded and ready", e.name),
            vocab_size: Some(e.tokenizer.vocab_size()),
        })
        .into_response();
    }

    // Check job status (jobs are tracked by ID)
    if let Some(job_queue) = context.worker_job_queue.get() {
        if let Some(job_status) = job_queue.get_status(tokenizer_id) {
            return Json(AddTokenizerResponse {
                id: tokenizer_id.to_string(),
                status: job_status.status.clone(),
                message: job_status
                    .message
                    .unwrap_or_else(|| format!("Tokenizer job is {}", job_status.status)),
                vocab_size: None,
            })
            .into_response();
        }
    }

    // Not found
    refusal(
        StatusCode::NOT_FOUND,
        "TOKENIZER_NOT_FOUND",
        format!("Tokenizer '{tokenizer_id}' not found and no pending job"),
    )
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::MockTokenizer;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, Worker, WorkerType};

    /// The status, code and message of a refusal in the public envelope.
    async fn refusal_of(response: Response) -> (StatusCode, String, String) {
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("the error envelope");
        let field = |key: &str| body["error"][key].as_str().unwrap_or_default().to_owned();
        (status, field("code"), field("message"))
    }

    /// A worker serving `model`, with no health check.
    fn worker_serving(model: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(format!("http://{model}.invalid:8000"))
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new(model))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// Loads the mock tokenizer under `name` (its id too).
    async fn load_mock(registry: &TokenizerRegistry, name: &str) {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        registry
            .load(name, name, "test", || async move { Ok(tokenizer) })
            .await
            .expect("load the mock tokenizer");
    }

    #[tokio::test]
    async fn test_get_tokenizer_empty_registry() {
        let workers = WorkerRegistry::new();
        let registry = TokenizerRegistry::new();
        let Err(refusal) = get_tokenizer(&workers, &registry, "any") else {
            panic!("Expected error");
        };
        let (status, code, message) = refusal_of(refusal).await;
        assert_eq!(
            (status, code.as_str()),
            (StatusCode::NOT_FOUND, "model_not_found")
        );
        assert!(message.contains("No tokenizers available"), "{message}");
    }

    /// A body that names no model (absent, which reads as the unknown model,
    /// or empty) is served by the one loaded tokenizer, under that
    /// tokenizer's name. With a second loaded it names neither, and is
    /// refused as `missing_model`, the message naming both; a body naming
    /// one is served by it.
    #[tokio::test]
    async fn a_body_without_a_model_is_served_only_while_one_tokenizer_is_loaded() {
        let workers = WorkerRegistry::new();
        let registry = TokenizerRegistry::new();
        load_mock(&registry, "alpha").await;
        for model in [UNKNOWN_MODEL_ID, ""] {
            let Ok((_, name)) = get_tokenizer(&workers, &registry, model) else {
                panic!("one tokenizer serves {model:?}");
            };
            assert_eq!(name, "alpha");
        }

        load_mock(&registry, "beta").await;
        for model in [UNKNOWN_MODEL_ID, ""] {
            let Err(refusal) = get_tokenizer(&workers, &registry, model) else {
                panic!("two tokenizers refuse {model:?}");
            };
            let (status, code, message) = refusal_of(refusal).await;
            assert_eq!(
                (status, code.as_str()),
                (StatusCode::BAD_REQUEST, "missing_model")
            );
            assert!(message.contains("alpha, beta"), "{message}");
        }
        let Ok((_, name)) = get_tokenizer(&workers, &registry, "beta") else {
            panic!("a named tokenizer serves its model");
        };
        assert_eq!(name, "beta");
    }

    /// A single-model gateway holds two tokenizers of its one model: the
    /// startup load under its path and the worker's under the served name.
    /// A body that names no model is the served model's, and is served by the
    /// tokenizer under that name; with a second model served it names
    /// neither, and the two tokenizers leave it to the `missing_model`
    /// refusal.
    #[tokio::test]
    async fn a_body_without_a_model_is_the_one_served_models_beside_another_tokenizer() {
        let workers = WorkerRegistry::new();
        let registry = TokenizerRegistry::new();
        load_mock(&registry, "/models/alpha").await;
        load_mock(&registry, "alpha").await;
        workers
            .register(worker_serving("alpha"))
            .expect("register the worker");
        for model in [UNKNOWN_MODEL_ID, ""] {
            let Ok((_, name)) = get_tokenizer(&workers, &registry, model) else {
                panic!("the served model's tokenizer serves {model:?}");
            };
            assert_eq!(name, "alpha");
        }

        workers
            .register(worker_serving("beta"))
            .expect("register the worker");
        let Err(refusal) = get_tokenizer(&workers, &registry, UNKNOWN_MODEL_ID) else {
            panic!("two served models and two tokenizers refuse the body");
        };
        let (status, code, _) = refusal_of(refusal).await;
        assert_eq!(
            (status, code.as_str()),
            (StatusCode::BAD_REQUEST, "missing_model")
        );
    }
}
