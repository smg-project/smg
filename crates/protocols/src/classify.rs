//! Classify API protocol definitions.
//!
//! This module defines the request and response types for the `/v1/classify` API,
//! which is compatible with the serving engine's classification endpoint.
//!
//! Classification reuses the embedding backend - the scheduler returns logits as
//! "embeddings", and the classify layer applies softmax + label mapping.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::{invalid_at, routing_text_of_input, GenerationRequest, StringOrArray, UsageInfo},
    embedding::validate_input,
    validated::Normalizable,
};

// ============================================================================
// Classify API
// ============================================================================

/// Classification request - compatible with the serving engine's /v1/classify API.
/// Its rules are the public API's shape rule on `input` (`validate_input`; the
/// embeddings route's array of objects is not among its shapes, as no engine
/// takes one here) and, for a list `rid`, one id per input, none twice
/// (`validate_rid_per_input`):
/// the validated extractor answers a body outside them in the gateway's error
/// envelope before any worker sees it.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rid_per_input"))]
pub struct ClassifyRequest {
    /// ID of the model to use
    pub model: String,

    /// Input can be a string, array of strings, or token IDs
    /// - Single string: "text to classify"
    /// - Array of strings: ["text1", "text2"]
    /// - Token IDs: [1, 2, 3] (advanced usage)
    ///
    /// Nothing empty: the public API's shape rule, shared with the embeddings route.
    #[validate(custom(function = "validate_input"))]
    pub input: Value,

    /// Optional user identifier
    pub user: Option<String>,

    /// Engine extension: request id for tracking, one, or one per input, as
    /// SGLang's `rid` takes it: a list runs the i-th input under the i-th id
    pub rid: Option<StringOrArray>,

    /// Engine extension: request priority, any 64-bit integer, as SGLang's
    /// `priority` takes it
    pub priority: Option<i64>,

    /// Additional fields not explicitly defined above (e.g. engine-specific parameters)
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

impl Normalizable for ClassifyRequest {}

/// A list `rid` names each input's request, as SGLang's classify runs the
/// i-th input under the i-th id and refuses an id named twice: it holds one id
/// per input (a string or an array of token ids is one input; an array of
/// strings or of token id arrays is one per member), none twice. The rule runs
/// after `input`'s shape rule has passed.
fn validate_rid_per_input(request: &ClassifyRequest) -> Result<(), validator::ValidationError> {
    let Some(StringOrArray::Array(ids)) = &request.rid else {
        return Ok(());
    };
    let inputs = match &request.input {
        Value::Array(members)
            if members
                .first()
                .is_some_and(|first| first.is_string() || first.is_array()) =>
        {
            members.len()
        }
        _ => 1,
    };
    if ids.len() != inputs {
        return Err(invalid_at(
            "rid",
            "invalid_value",
            format!(
                "Invalid 'rid': a list of {} ids for {inputs} inputs. A list of ids takes one per input.",
                ids.len()
            ),
        ));
    }
    let mut seen = HashSet::with_capacity(ids.len());
    if let Some(repeated) = ids.iter().find(|id| !seen.insert(id.as_str())) {
        return Err(invalid_at(
            "rid",
            "invalid_value",
            format!("Invalid 'rid': '{repeated}' is named twice. Each input takes its own id."),
        ));
    }
    Ok(())
}

impl GenerationRequest for ClassifyRequest {
    fn rid(&self) -> Option<&str> {
        // A batch's first id, as a rerank request's
        self.rid.as_ref().and_then(StringOrArray::first)
    }

    fn is_stream(&self) -> bool {
        false // Classification is always non-streaming
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        routing_text_of_input(&self.input)
    }
}

// ============================================================================
// Classify Response
// ============================================================================

/// Single classification result
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClassifyData {
    /// Index of this result (for batch requests)
    pub index: u32,
    /// Predicted class label (from id2label mapping)
    pub label: String,
    /// Probability distribution over all classes (softmax of logits)
    pub probs: Vec<f32>,
    /// Number of classes
    pub num_classes: u32,
}

/// Classification response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClassifyResponse {
    /// Unique request ID (format: "classify-{uuid}")
    pub id: String,
    /// Always "list"
    pub object: String,
    /// Unix timestamp (seconds since epoch)
    pub created: u64,
    /// Model name
    pub model: String,
    /// Classification results (one per input in batch)
    pub data: Vec<ClassifyData>,
    /// Token usage info
    pub usage: UsageInfo,
}

impl ClassifyResponse {
    /// Create a new ClassifyResponse with the given data
    pub fn new(
        id: String,
        model: String,
        created: u64,
        data: Vec<ClassifyData>,
        usage: UsageInfo,
    ) -> Self {
        Self {
            id,
            object: "list".to_string(),
            created,
            model,
            data,
            usage,
        }
    }
}
