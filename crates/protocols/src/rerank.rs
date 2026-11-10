use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use validator::Validate;

use super::common::{default_true, GenerationRequest, StringOrArray, UsageInfo};

fn default_rerank_object() -> String {
    "rerank".to_string()
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rerank_request"))]
pub struct RerankRequest {
    /// The query text to rank documents against
    #[validate(custom(function = "validate_query"))]
    pub query: String,

    /// List of documents to be ranked
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// Model to use for reranking
    pub model: String,

    /// Maximum number of documents to return (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_true")]
    pub return_documents: bool,

    // Engine-specific extensions
    /// Request ID for tracking
    pub rid: Option<StringOrArray>,

    /// User identifier
    pub user: Option<String>,
}

impl GenerationRequest for RerankRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_ref().and_then(StringOrArray::first)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl super::validated::Normalizable for RerankRequest {
    // Use default no-op normalization
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates that the query is not empty
fn validate_query(query: &str) -> Result<(), validator::ValidationError> {
    if query.trim().is_empty() {
        return Err(validator::ValidationError::new("query cannot be empty"));
    }
    Ok(())
}

/// Validates that the documents list is not empty
fn validate_documents(documents: &[String]) -> Result<(), validator::ValidationError> {
    if documents.is_empty() {
        return Err(validator::ValidationError::new(
            "documents list cannot be empty",
        ));
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
#[expect(
    clippy::unnecessary_wraps,
    reason = "validator crate requires Result return type"
)]
fn validate_rerank_request(req: &RerankRequest) -> Result<(), validator::ValidationError> {
    // Validate top_k if specified
    if let Some(k) = req.top_k {
        if k > req.documents.len() {
            tracing::warn!(
                "top_k ({}) is greater than number of documents ({})",
                k,
                req.documents.len()
            );
        }
    }
    Ok(())
}

impl RerankRequest {
    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

/// Individual rerank result
///
/// Written in the gateway's own spelling; read from a worker's answer in
/// every spelling the rerank contracts write: `relevance_score` for the
/// score and `text` for the document (the Cohere-style and the
/// embeddings-server contracts), and a document given as the `{text}`
/// object.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResult {
    /// Relevance score for the document
    #[serde(alias = "relevance_score")]
    pub score: f32,

    /// The document text (if return_documents was true)
    #[serde(default, alias = "text", deserialize_with = "deserialize_document")]
    pub document: Option<String>,

    /// Original index of the document in the request
    pub index: usize,

    /// Additional metadata about the ranking
    pub meta_info: Option<HashMap<String, Value>>,
}

/// A result's document as the contracts write it: the text itself, the
/// `{text}` object of the Cohere-style contract, or nothing.
fn deserialize_document<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Document {
        Text(String),
        Object { text: Option<String> },
    }
    Ok(match Option::<Document>::deserialize(deserializer)? {
        Some(Document::Text(text)) => Some(text),
        Some(Document::Object { text }) => text,
        None => None,
    })
}

/// A rerank worker's answer: the ranked results as the bare list the gateway
/// itself answers with (the embeddings-server contract writes the same list,
/// `text` for the document), or under `results` as the Cohere-style object
/// of the OpenAI-compatible rerank servers (`relevance_score`, the document
/// as text or as `{text}`; the object's `id`, `model` and `usage` are not
/// read). One `from_slice` reads every shape; the results keep the worker's
/// order, which every contract sorts by score, highest first.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RerankWorkerResponse {
    /// The results as a bare list.
    List(Vec<RerankResult>),
    /// The results under `results`.
    Object { results: Vec<RerankResult> },
}

impl RerankWorkerResponse {
    /// The ranked results in the worker's order.
    pub fn into_results(self) -> Vec<RerankResult> {
        match self {
            Self::List(results) | Self::Object { results } => results,
        }
    }
}

/// Rerank response containing sorted results
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResponse {
    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Model used for reranking
    pub model: String,

    /// Usage information
    pub usage: Option<UsageInfo>,

    /// Response object type
    #[serde(default = "default_rerank_object")]
    pub object: String,

    /// Response ID
    pub id: Option<StringOrArray>,

    /// Creation timestamp
    pub created: i64,
}

impl RerankResponse {
    /// Create a new RerankResponse with the given results and model
    pub fn new(
        results: Vec<RerankResult>,
        model: String,
        request_id: Option<StringOrArray>,
    ) -> Self {
        RerankResponse {
            results,
            model,
            usage: None,
            object: default_rerank_object(),
            id: request_id,
            created: current_timestamp(),
        }
    }

    /// Apply top_k limit to results
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
}

/// The body of `/v1/rerank`: the request of the OpenAI-compatible rerank
/// servers, `query` and `documents` with an optional `model`, `top_n` and
/// `return_documents`. A body without `model` is routed as the unknown
/// model, to any worker, the shape a single-model server's client sends;
/// the native `/rerank` route takes [`RerankRequest`], whose `model` is
/// required.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankReqInput {
    pub query: String,
    pub documents: Vec<String>,

    /// The model to rerank with, which selects the worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Maximum number of documents to return; absent and 0 mean all of them,
    /// as the OpenAI-compatible servers read it. `top_k`, the native
    /// spelling, is read as well.
    #[serde(default, alias = "top_k", skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,

    /// Whether the documents are returned with their scores (the default).
    #[serde(default = "default_true")]
    pub return_documents: bool,
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: v1
                .model
                .unwrap_or_else(|| super::UNKNOWN_MODEL_ID.to_string()),
            top_k: v1.top_n.filter(|n| *n > 0),
            return_documents: v1.return_documents,
            rid: None,
            user: None,
        }
    }
}
