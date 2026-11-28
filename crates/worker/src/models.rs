//! The entry of an OpenAI-compatible `/v1/models` response.
//!
//! A [`ModelCard`](crate::model_card::ModelCard) converts into one; the
//! response that lists them lives in `openai-protocol`.

use serde::{Deserialize, Serialize};

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelObject {
    /// Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of model creation (0 = unknown).
    pub created: i64,
    /// Who owns/hosts the model.
    pub owned_by: String,
}
