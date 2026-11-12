use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{routing_text_of_input, GenerationRequest, UsageInfo};

// ============================================================================
// Embedding API
// ============================================================================

/// A request to the embeddings route: the model, the input (a text, texts,
/// token ids or a batch of them) and the encoding format of the vectors.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct EmbeddingRequest {
    /// ID of the model to use
    pub model: String,

    /// Input can be a string, array of strings, tokens, or batch inputs
    pub input: Value,

    /// Optional encoding format (e.g., "float", "base64")
    pub encoding_format: Option<String>,

    /// Optional user identifier
    pub user: Option<String>,

    /// Optional number of dimensions for the embedding
    pub dimensions: Option<u32>,

    /// Engine extension: request id for tracking
    pub rid: Option<String>,

    /// Additional fields not explicitly defined above (e.g. engine-specific parameters)
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

impl GenerationRequest for EmbeddingRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        // Embeddings are non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        routing_text_of_input(&self.input)
    }
}

/// One embedding of the answer, object `embedding`: the vector and the index of
/// the input it embeds.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingObject {
    pub object: String, // "embedding"
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// The answer of the embeddings route, object `list`: one embedding per input,
/// the model and the usage.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingResponse {
    pub object: String, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: UsageInfo,
}
