use std::collections::HashMap;

use serde::{
    de::{self, value::SeqAccessDeserializer, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::{Map, Value};
use validator;

// ============================================================================
// Default value helpers
// ============================================================================

/// Default model for endpoints where model is optional (e.g., /generate).
/// Uses UNKNOWN_MODEL_ID so routers treat it as "any available worker."
pub fn default_unknown_model() -> String {
    super::UNKNOWN_MODEL_ID.to_string()
}

/// Helper function for serde default value (returns true)
pub fn default_true() -> bool {
    true
}

/// Helper for `#[serde(skip_serializing_if = "is_false")]` on default-`false` flags.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
pub fn is_false(v: &bool) -> bool {
    !*v
}

/// Helper for `#[serde(skip_serializing_if = "is_true")]` on default-`true` flags.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
pub fn is_true(v: &bool) -> bool {
    *v
}

/// Deserialize a bool that also accepts JSON `null` (mapped to `false`).
///
/// Use with `#[serde(default, deserialize_with = "deserialize_null_as_false")]`
/// on fields that the OpenAI spec defines as `Optional[bool]` defaulting to `false`.
pub fn deserialize_null_as_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<bool>::deserialize(deserializer).map(|opt| opt.unwrap_or(false))
}

// ============================================================================
// GenerationRequest Trait
// ============================================================================

/// Trait for unified access to generation request properties
/// Implemented by ChatCompletionRequest, CompletionRequest, GenerateRequest,
/// EmbeddingRequest, RerankRequest, and ResponsesRequest
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;

    /// Token IDs for routing when the request is already tokenized.
    /// Some(_) routes on the token radix tree instead of the decimal-string
    /// rendering of the same IDs.
    fn routing_tokens(&self) -> Option<&[i32]> {
        None
    }

    /// Client-provided request id, when the protocol carries one. Routing may
    /// derive a session-affinity key from it; a batch reports its first id.
    fn rid(&self) -> Option<&str> {
        None
    }
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    /// Get the number of items in the StringOrArray
    pub fn len(&self) -> usize {
        match self {
            StringOrArray::String(_) => 1,
            StringOrArray::Array(arr) => arr.len(),
        }
    }

    /// Check if the StringOrArray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrArray::String(s) => s.is_empty(),
            StringOrArray::Array(arr) => arr.is_empty(),
        }
    }

    /// Convert to a vector of strings (clones the data)
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s.clone()],
            StringOrArray::Array(arr) => arr.clone(),
        }
    }

    /// Returns an iterator over string references without cloning.
    /// Use this instead of `to_vec()` when you only need to iterate.
    pub fn iter(&self) -> StringOrArrayIter<'_> {
        StringOrArrayIter {
            inner: self,
            index: 0,
        }
    }

    /// Returns the first string, or None if empty
    pub fn first(&self) -> Option<&str> {
        match self {
            StringOrArray::String(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }
            StringOrArray::Array(arr) => arr.first().map(|s| s.as_str()),
        }
    }
}

/// Iterator over StringOrArray that yields string references without cloning
pub struct StringOrArrayIter<'a> {
    inner: &'a StringOrArray,
    index: usize,
}

impl<'a> Iterator for StringOrArrayIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            StringOrArray::String(s) => {
                if self.index == 0 {
                    self.index = 1;
                    Some(s.as_str())
                } else {
                    None
                }
            }
            StringOrArray::Array(arr) => {
                if self.index < arr.len() {
                    let item = &arr[self.index];
                    self.index += 1;
                    Some(item.as_str())
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match self.inner {
            StringOrArray::String(_) => 1 - self.index,
            StringOrArray::Array(arr) => arr.len() - self.index,
        };
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for StringOrArrayIter<'a> {}

/// Validates stop sequences (max 4, non-empty strings)
/// Used by both ChatCompletionRequest and ResponsesRequest
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    match stop {
        StringOrArray::String(s) => {
            if s.is_empty() {
                return Err(validator::ValidationError::new(
                    "stop sequences cannot be empty",
                ));
            }
        }
        StringOrArray::Array(arr) => {
            if arr.len() > 4 {
                return Err(validator::ValidationError::new(
                    "maximum 4 stop sequences allowed",
                ));
            }
            for s in arr {
                if s.is_empty() {
                    return Err(validator::ValidationError::new(
                        "stop sequences cannot be empty",
                    ));
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "audio_url")]
    AudioUrl { audio_url: AudioUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct AudioUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct InputAudio {
    /// Base64-encoded audio bytes.
    pub data: String,
    /// Encoded audio format. The OpenAI Chat API supports `wav` and `mp3`.
    pub format: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct VideoUrl {
    pub url: String,
}

// ============================================================================
// Response Format (for structured outputs)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StreamOptions {
    /// Chat Completions / Completions: include usage block at end of stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,

    /// Chat Completions / Completions: emit a usage chunk with every streamed
    /// delta instead of only in the final chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuous_usage_stats: Option<bool>,

    /// Responses API: add random chars on `obfuscation` field of delta events
    /// to normalize payload sizes. Defaults to `true` upstream when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,

    /// Additional fields not explicitly defined above (e.g. engine-specific
    /// streaming options). Without this, the gateway silently drops them while
    /// re-serializing the request for the backend.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

/// Tool choice value for simple string options
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for both Chat Completion and Responses APIs
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String, // "allowed_tools"
        mode: String, // "auto" | "required" TODO: need validation
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

impl ToolChoice {
    /// Serialize tool_choice to string for ResponsesResponse
    ///
    /// Returns the JSON-serialized tool_choice or "auto" as default
    pub fn serialize_to_string(tool_choice: Option<&ToolChoice>) -> String {
        tool_choice
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for ToolChoice::AllowedTools
///
/// Represents a reference to a specific tool in the allowed_tools array.
/// Different tool types have different required fields.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    /// Reference to a function tool
    #[serde(rename = "function")]
    Function { name: String },

    /// Reference to an MCP tool
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// File search hosted tool
    #[serde(rename = "file_search")]
    FileSearch,

    /// Web search preview hosted tool
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,

    /// Computer use preview hosted tool
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,

    /// Code interpreter hosted tool
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,

    /// Image generation hosted tool
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{name}"),
            ToolReference::Mcp { server_label, name } => {
                if let Some(n) = name {
                    format!("mcp:{server_label}:{n}")
                } else {
                    format!("mcp:{server_label}")
                }
            }
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }

    /// Get the tool name if this is a function tool
    pub fn function_name(&self) -> Option<&str> {
        match self {
            ToolReference::Function { name } => Some(name.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: Function,
}

/// Per the OpenAI spec, omitting `parameters` defines a function with an
/// empty parameter list, so a missing field deserializes to an empty schema.
fn empty_parameters_schema() -> Value {
    Value::Object(Map::new())
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    #[serde(default = "empty_parameters_schema")]
    pub parameters: Value, // JSON Schema
    /// Whether to enable strict schema adherence (OpenAI structured outputs)
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: FunctionCallResponse,
}

/// Deprecated `function_call` field from the OpenAI API.
/// Can be `"none"`, `"auto"`, or `{"name": "function_name"}`.
#[derive(Debug, Clone)]
pub enum FunctionCall {
    None,
    Auto,
    Function { name: String },
}

impl Serialize for FunctionCall {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            FunctionCall::None => serializer.serialize_str("none"),
            FunctionCall::Auto => serializer.serialize_str("auto"),
            FunctionCall::Function { name } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("name", name)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for FunctionCall {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match &value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(FunctionCall::None),
                "auto" => Ok(FunctionCall::Auto),
                other => Err(de::Error::custom(format!(
                    "unknown function_call value: \"{other}\""
                ))),
            },
            Value::Object(map) => {
                if let Some(Value::String(name)) = map.get("name") {
                    Ok(FunctionCall::Function { name: name.clone() })
                } else {
                    Err(de::Error::custom(
                        "function_call object must have a \"name\" string field",
                    ))
                }
            }
            _ => Err(de::Error::custom(
                "function_call must be a string or object",
            )),
        }
    }
}

impl schemars::JsonSchema for FunctionCall {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "FunctionCall".into()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // FunctionCall is either "none", "auto", or {"name": "..."}
        let name_schema = generator.subschema_for::<String>();
        schemars::json_schema!({
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["none", "auto"]
                },
                {
                    "type": "object",
                    "properties": { "name": name_schema },
                    "required": ["name"]
                }
            ]
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>, // JSON string
}

// ============================================================================
// Usage and Logging
// ============================================================================
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    /// Create a Usage from prompt and completion token counts
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Add cached token details to this Usage
    pub fn with_cached_tokens(mut self, cached_tokens: u32) -> Self {
        // Calling this builder means the backend supplied cache accounting.
        // Zero is therefore evidence of a cold miss, not absence of support,
        // and must remain distinguishable from `prompt_tokens_details: None`.
        self.prompt_tokens_details = Some(PromptTokenUsageInfo { cached_tokens });
        self
    }

    /// Add reasoning token details to this Usage
    pub fn with_reasoning_tokens(mut self, reasoning_tokens: u32) -> Self {
        if reasoning_tokens > 0 {
            self.completion_tokens_details = Some(CompletionTokensDetails {
                reasoning_tokens: Some(reasoning_tokens),
                accepted_prediction_tokens: None,
                rejected_prediction_tokens: None,
            });
        }
        self
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
    pub accepted_prediction_tokens: Option<u32>,
    pub rejected_prediction_tokens: Option<u32>,
}

/// Usage information (used by rerank and other endpoints)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ChatLogProbs {
    Detailed {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ChatLogProbsContent>>,
    },
    Raw(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

/// Shape probe for the first `input_ids` element: it alone picks the variant,
/// so the rest parses in place instead of through untagged-enum buffering.
enum FirstInputId {
    Id(i32),
    Ids(Vec<i32>),
}

impl<'de> Deserialize<'de> for FirstInputId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FirstInputIdVisitor;

        impl<'de> Visitor<'de> for FirstInputIdVisitor {
            type Value = FirstInputId;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a token id or an array of token ids")
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                i32::try_from(v)
                    .map(FirstInputId::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Signed(v), &self))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                i32::try_from(v)
                    .map(FirstInputId::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Unsigned(v), &self))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Deserialize::deserialize(SeqAccessDeserializer::new(seq)).map(FirstInputId::Ids)
            }
        }

        deserializer.deserialize_any(FirstInputIdVisitor)
    }
}

impl<'de> Deserialize<'de> for InputIds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct InputIdsVisitor;

        impl<'de> Visitor<'de> for InputIdsVisitor {
            type Value = InputIds;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an array of token ids or an array of token id arrays")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // An empty array is Single, matching untagged first-match order.
                let Some(first) = seq.next_element::<FirstInputId>()? else {
                    return Ok(InputIds::Single(Vec::new()));
                };
                let remaining = seq.size_hint().unwrap_or(0);
                match first {
                    FirstInputId::Id(id) => {
                        let mut ids = Vec::with_capacity(remaining.saturating_add(1));
                        ids.push(id);
                        while let Some(id) = seq.next_element()? {
                            ids.push(id);
                        }
                        Ok(InputIds::Single(ids))
                    }
                    FirstInputId::Ids(head) => {
                        let mut seqs = Vec::with_capacity(remaining.saturating_add(1));
                        seqs.push(head);
                        while let Some(ids) = seq.next_element()? {
                            seqs.push(ids);
                        }
                        Ok(InputIds::Batch(seqs))
                    }
                }
            }
        }

        deserializer.deserialize_seq(InputIdsVisitor)
    }
}

/// LoRA adapter path - can be single path or batch of paths (SGLang extension)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum LoRAPath {
    Single(Option<String>),
    Batch(Vec<Option<String>>),
}

// ============================================================================
// Redacted Types
// ============================================================================
#[derive(Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Redacted(pub String);

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

// ============================================================================
// Response Prompt
// ============================================================================

/// Reference to a prompt template and its variables.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResponsePrompt {
    pub id: String,
    pub variables: Option<HashMap<String, PromptVariable>>,
    pub version: Option<String>,
}

/// A prompt variable value: plain string or a typed input (text, image, file).
///
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string succeeds
/// as `String`; a JSON object falls through to `Typed`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum PromptVariable {
    String(String),
    Typed(PromptVariableTyped),
}

/// Typed prompt variable input.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match OpenAI API spec"
)]
pub enum PromptVariableTyped {
    #[serde(rename = "input_text")]
    ResponseInputText { text: String },
    #[serde(rename = "input_image")]
    ResponseInputImage {
        detail: Option<Detail>,
        file_id: Option<String>,
        image_url: Option<String>,
    },
    #[serde(rename = "input_file")]
    ResponseInputFile {
        file_data: Option<String>,
        file_id: Option<String>,
        file_url: Option<String>,
        filename: Option<String>,
    },
}

/// Image detail level for [`PromptVariableTyped::ResponseInputImage`] and
/// [`crate::responses::ResponseContentPart::InputImage`]. Spec allows
/// `"low" | "high" | "auto" | "original"`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Low,
    High,
    #[default]
    Auto,
    Original,
}

// ============================================================================
// Responses API: prompt-cache retention & context management
// ============================================================================

/// Retention policy for prompt-cache entries on the Responses API.
///
/// Spec: `prompt_cache_retention: "in-memory" | "24h"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum PromptCacheRetention {
    #[serde(rename = "in-memory")]
    InMemory,
    #[serde(rename = "24h")]
    Duration24h,
}

/// A single entry in the Responses API `context_management` array.
///
/// Spec: each entry has `type` (currently only `"compaction"`) and an optional
/// `compact_threshold` token count.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContextManagementEntry {
    #[serde(rename = "type")]
    pub r#type: ContextManagementType,
    pub compact_threshold: Option<u32>,
}

/// Type tag for [`ContextManagementEntry`]. Currently only `compaction` is
/// defined by the spec; the enum is kept small so unknown values serde-fail
/// (consistent with P5's fail-fast direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContextManagementType {
    Compaction,
}

// ============================================================================
// Responses API: conversation reference
// ============================================================================

/// Reference to a conversation the response belongs to.
///
/// Spec: `conversation: string | ResponseConversationParam { id: string }`.
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string succeeds
/// as `Id`; an object falls through to `Object`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ConversationRef {
    Id(String),
    Object { id: String },
}

impl ConversationRef {
    /// Return the underlying conversation id regardless of the wire shape.
    pub fn as_id(&self) -> &str {
        match self {
            Self::Id(id) | Self::Object { id } => id.as_str(),
        }
    }

    /// `true` when the underlying conversation id is the empty string.
    /// Mirrors `String::is_empty` for callers that previously treated
    /// `Option<String>` empty values as "unset".
    pub fn is_empty(&self) -> bool {
        self.as_id().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[derive(Deserialize)]
    struct NullableBoolTest {
        #[serde(default, deserialize_with = "deserialize_null_as_false")]
        field: bool,
    }

    #[test]
    fn test_deserialize_null_as_false() {
        let cases = [
            (json!({"field": true}), true),
            (json!({"field": false}), false),
            (json!({"field": null}), false),
            (json!({}), false),
        ];
        for (input, expected) in cases {
            let t: NullableBoolTest = serde_json::from_value(input).unwrap();
            assert_eq!(t.field, expected);
        }
    }

    #[test]
    fn test_deserialize_null_as_false_rejects_non_bool() {
        let result = serde_json::from_value::<NullableBoolTest>(json!({"field": "yes"}));
        assert!(result.is_err());
    }

    #[test]
    fn stream_options_preserve_unknown_fields() {
        let raw = json!({
            "include_usage": true,
            "continuous_usage_stats": true,
            "step_usage_chunks": "all",
            "engine_specific": {"nested": 1},
        });

        let opts: StreamOptions = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(opts.include_usage, Some(true));
        assert_eq!(opts.continuous_usage_stats, Some(true));
        assert_eq!(opts.other.get("step_usage_chunks"), Some(&json!("all")));

        // Re-serializing must round-trip the engine-specific keys, otherwise the
        // gateway would strip them on the way to the backend.
        assert_eq!(serde_json::to_value(&opts).unwrap(), raw);
    }

    #[test]
    fn stream_options_without_unknown_fields_stay_compact() {
        let opts: StreamOptions = serde_json::from_value(json!({"include_usage": true})).unwrap();
        assert!(opts.other.is_empty());
        assert_eq!(
            serde_json::to_value(&opts).unwrap(),
            json!({"include_usage": true})
        );
    }

    #[test]
    fn cached_token_builder_preserves_explicit_zero() {
        let usage = Usage::from_counts(16, 1).with_cached_tokens(0);
        assert!(matches!(
            usage.prompt_tokens_details,
            Some(PromptTokenUsageInfo { cached_tokens: 0 })
        ));
    }

    #[test]
    fn content_part_deserializes_audio_url() {
        let value = json!({
            "type": "audio_url",
            "audio_url": {
                "url": "https://example.com/audio.wav"
            }
        });
        let part: ContentPart = serde_json::from_value(value).expect("audio_url content part");
        assert_eq!(
            part,
            ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "https://example.com/audio.wav".to_string(),
                },
            }
        );
    }

    #[test]
    fn content_part_round_trips_input_audio() {
        let value = json!({
            "type": "input_audio",
            "input_audio": {
                "data": "UklGRg==",
                "format": "wav"
            }
        });
        let part: ContentPart =
            serde_json::from_value(value.clone()).expect("input_audio content part");
        assert_eq!(
            part,
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "UklGRg==".to_string(),
                    format: "wav".to_string(),
                },
            }
        );
        assert_eq!(serde_json::to_value(part).unwrap(), value);
    }

    #[test]
    fn conversation_ref_deserializes_bare_string() {
        let v = json!("conv_abc");
        let r: ConversationRef = serde_json::from_value(v).expect("string form");
        assert!(matches!(r, ConversationRef::Id(ref s) if s == "conv_abc"));
        assert_eq!(r.as_id(), "conv_abc");
        // Bare string round-trips back to a JSON string.
        assert_eq!(serde_json::to_value(&r).unwrap(), json!("conv_abc"));
    }

    #[test]
    fn conversation_ref_deserializes_object() {
        let v = json!({"id": "conv_xyz"});
        let r: ConversationRef = serde_json::from_value(v).expect("object form");
        assert!(matches!(r, ConversationRef::Object { ref id } if id == "conv_xyz"));
        assert_eq!(r.as_id(), "conv_xyz");
        // Object round-trips back to an object.
        assert_eq!(serde_json::to_value(&r).unwrap(), json!({"id": "conv_xyz"}));
    }

    #[test]
    fn conversation_ref_is_empty() {
        assert!(ConversationRef::Id(String::new()).is_empty());
        assert!(!ConversationRef::Id("conv_1".to_string()).is_empty());
        assert!(ConversationRef::Object { id: String::new() }.is_empty());
    }

    #[test]
    fn input_ids_deserializes_single() {
        let ids: InputIds = serde_json::from_str("[1, -2, 3]").unwrap();
        assert!(matches!(ids, InputIds::Single(ref v) if v == &[1, -2, 3]));
    }

    #[test]
    fn input_ids_deserializes_batch() {
        let ids: InputIds = serde_json::from_str("[[1, 2], [3], []]").unwrap();
        assert!(matches!(ids, InputIds::Batch(ref v) if v == &[vec![1, 2], vec![3], vec![]]));
    }

    #[test]
    fn input_ids_empty_array_is_single() {
        let ids: InputIds = serde_json::from_str("[]").unwrap();
        assert!(matches!(ids, InputIds::Single(ref v) if v.is_empty()));
    }

    #[test]
    fn input_ids_rejects_invalid_input() {
        for input in [
            "[1, [2]]",
            "[[1], 2]",
            "[\"a\"]",
            "[1.5]",
            "[5000000000]",
            "\"nope\"",
            "null",
            "7",
        ] {
            assert!(
                serde_json::from_str::<InputIds>(input).is_err(),
                "accepted {input}"
            );
        }
    }

    #[test]
    fn input_ids_round_trip() {
        for input in [json!([1, 2, 3]), json!([[1, 2], [3]]), json!([])] {
            let ids: InputIds = serde_json::from_value(input.clone()).unwrap();
            assert_eq!(serde_json::to_value(&ids).unwrap(), input);
        }
    }

    #[test]
    fn function_deserializes_without_parameters() {
        // Per the OpenAI spec, omitting `parameters` defines a function with
        // an empty parameter list.
        let value = json!({"name": "web_search", "description": ""});
        let function: Function = serde_json::from_value(value).expect("parameterless function");
        assert_eq!(function.parameters, json!({}));
    }
}
