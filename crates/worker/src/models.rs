//! The entry of an OpenAI-compatible `/v1/models` response.
//!
//! A [`ModelCard`](crate::model_card::ModelCard) converts into one; the
//! response that lists them lives in `openai-protocol`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelObject {
    /// Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of model creation (0 = unknown).
    pub created: i64,
    /// Who owns/hosts the model.
    pub owned_by: String,
    /// Engine-provided extension fields, preserved at the top level.
    #[serde(default, flatten)]
    pub extensions: Map<String, Value>,
}
