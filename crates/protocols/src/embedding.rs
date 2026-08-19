use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{GenerationRequest, UsageInfo};

// ============================================================================
// Embedding API
// ============================================================================

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

    /// SGLang extension: request id for tracking
    pub rid: Option<String>,
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
        // Best effort: extract text content for routing decisions
        match &self.input {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingObject {
    pub object: String, // "embedding"
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingResponse {
    pub object: String, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: UsageInfo,
}
