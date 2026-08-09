use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use super::{
    common::{
        default_true, deserialize_null_as_false, validate_stop, ChatLogProbs, ContentPart,
        Function, FunctionCall, FunctionChoice, GenerationRequest, ResponseFormat, StreamOptions,
        StringOrArray, Tool, ToolCall, ToolCallDelta, ToolChoice, ToolChoiceValue, ToolReference,
        Usage,
    },
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::{
    builders::{ChatCompletionResponseBuilder, ChatCompletionStreamResponseBuilder},
    validated::Normalizable,
};

// ============================================================================
// Chat Messages
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<MessageContent>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for O1-style models (SGLang extension)
        reasoning_content: Option<String>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
    #[serde(rename = "function")]
    Function { content: String, name: String },
    #[serde(rename = "developer")]
    Developer {
        content: MessageContent,
        tools: Option<Vec<Tool>>,
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Returns the text content, cloning only when necessary.
    /// For simple text, returns a clone of the string.
    /// For parts, concatenates text parts with spaces.
    pub fn to_simple_string(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => {
                let mut result = String::new();
                let mut first = true;
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        if !first {
                            result.push(' ');
                        }
                        result.push_str(text);
                        first = false;
                    }
                }
                result
            }
        }
    }

    /// Appends text content directly to a buffer, avoiding intermediate allocations.
    /// Returns true if any content was appended.
    #[inline]
    pub fn append_text_to(&self, buffer: &mut String) -> bool {
        match self {
            MessageContent::Text(text) => {
                if text.is_empty() {
                    false
                } else {
                    buffer.push_str(text);
                    true
                }
            }
            MessageContent::Parts(parts) => {
                let mut appended = false;
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        if !text.is_empty() {
                            if appended {
                                buffer.push(' ');
                            }
                            buffer.push_str(text);
                            appended = true;
                        }
                    }
                }
                appended
            }
        }
    }

    /// Returns true if this content contains any non-empty text.
    #[inline]
    pub fn has_text(&self) -> bool {
        match self {
            MessageContent::Text(text) => !text.is_empty(),
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|part| matches!(part, ContentPart::Text { text } if !text.is_empty())),
        }
    }
}

// ============================================================================
// Chat Completion Request
// ============================================================================

/// DeepSeek thinking-mode configuration for the OpenAI-compatible API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatThinkingConfig {
    Enabled {},
    Disabled {},
}

impl ChatThinkingConfig {
    #[inline]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled {})
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
    /// A list of messages comprising the conversation so far
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,

    /// ID of the model to use
    pub model: String,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Deprecated: Replaced by tool_choice
    #[deprecated(note = "Use tool_choice instead")]
    pub function_call: Option<FunctionCall>,

    /// Deprecated: Replaced by tools
    #[deprecated(note = "Use tools instead")]
    pub functions: Option<Vec<Function>>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default, deserialize_with = "deserialize_null_as_false")]
    pub logprobs: bool,

    /// Deprecated: Replaced by max_completion_tokens
    #[deprecated(note = "Use max_completion_tokens instead")]
    #[validate(range(min = 1))]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,

    /// Developer-defined tags and values used for filtering completions in the dashboard
    pub metadata: Option<HashMap<String, String>>,

    /// Output types that you would like the model to generate for this request
    pub modalities: Option<Vec<String>>,

    /// Whether to return audio output.
    pub return_audio: Option<bool>,

    /// How many chat completion choices to generate for each input message
    #[validate(range(min = 1, max = 10))]
    pub n: Option<u32>,

    /// Whether to enable parallel function calling during tool use
    pub parallel_tool_calls: Option<bool>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Cache key for prompts (beta feature)
    pub prompt_cache_key: Option<String>,

    /// Enable or disable DeepSeek thinking mode.
    pub thinking: Option<ChatThinkingConfig>,

    /// Effort level for reasoning models.
    ///
    /// OpenAI-compatible callers normally send a named string, while some
    /// model integrations accept a numeric value. Keep the public Rust shape
    /// as a string for compatibility, but accept either JSON representation at
    /// the HTTP boundary; model-specific normalization happens in the gateway.
    #[serde(default, deserialize_with = "deserialize_reasoning_effort")]
    pub reasoning_effort: Option<String>,

    /// An object specifying the format that the model must output
    pub response_format: Option<ResponseFormat>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Deprecated: This feature is in Legacy mode
    #[deprecated(note = "This feature is in Legacy mode")]
    pub seed: Option<i64>,

    /// The service tier to use for this request
    pub service_tier: Option<String>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// If set, partial message deltas will be sent
    #[serde(default, deserialize_with = "deserialize_null_as_false")]
    pub stream: bool,

    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// What sampling temperature to use, between 0 and 2
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// Controls which (if any) tool is called by the model
    pub tool_choice: Option<ToolChoice>,

    /// A list of tools the model may call
    pub tools: Option<Vec<Tool>>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// An alternative to sampling with temperature
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Verbosity level for debugging
    pub verbosity: Option<i32>,

    // =============================================================================
    // Engine-Specific Sampling Parameters
    // =============================================================================
    // These parameters are extensions beyond the OpenAI API specification and
    // control model generation behavior in engine-specific ways.
    // =============================================================================
    /// Top-k sampling parameter (-1 to disable)
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[validate(range(min = 0))]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    pub ebnf: Option<String>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    pub session_params: Option<HashMap<String, Value>>,

    /// Separate reasoning content from final answer (O1-style models)
    #[serde(default = "default_true")]
    pub separate_reasoning: bool,

    /// Stream reasoning tokens during generation
    #[serde(default = "default_true")]
    pub stream_reasoning: bool,

    /// Chat template kwargs
    pub chat_template_kwargs: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Random seed for sampling for deterministic outputs
    pub sampling_seed: Option<u64>,

    /// Request ID forwarded to the backend for log correlation (SGLang extension)
    pub rid: Option<String>,

    /// Additional fields not explicitly defined above (e.g. engine-specific parameters)
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl ChatCompletionRequest {
    /// Resolve the protocol-level thinking preference.
    ///
    /// The explicit DeepSeek `thinking` field wins over compatibility mappings
    /// from OpenAI's `reasoning_effort`.
    #[inline]
    pub fn thinking_preference(&self) -> Option<bool> {
        self.thinking
            .map(ChatThinkingConfig::is_enabled)
            .or_else(|| thinking_from_reasoning_effort(self.reasoning_effort.as_deref()))
    }
}

/// Map an OpenAI `reasoning_effort` to a thinking on/off preference.
///
/// This is the protocol-level interpretation of "does the caller want
/// reasoning?" — independent of any model/template. `reasoning_effort` is a
/// *level* (`"low"`/`"medium"`/`"high"`) plus the vendor-extension `"none"`.
///
/// Both `"none"` and `"minimal"` map to thinking OFF (`Some(false)`).
/// `"minimal"` is treated as an off-signal deliberately: templates that expose
/// only a boolean thinking toggle (GLM/Qwen3) cannot do "a little" reasoning,
/// so the lowest OpenAI level is the closest available "do not reason".
/// Level values return `None` — no opinion, defer to the template default or an
/// explicit thinking kwarg.
pub fn thinking_from_reasoning_effort(reasoning_effort: Option<&str>) -> Option<bool> {
    match reasoning_effort {
        Some("none") | Some("minimal") => Some(false),
        _ => None,
    }
}

fn deserialize_reasoning_effort<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(Value::Number(value)) => Ok(Some(value.to_string())),
        Some(_) => Err(serde::de::Error::custom(
            "reasoning_effort must be a string, number, or null",
        )),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates messages array is not empty and has valid content
fn validate_messages(messages: &[ChatMessage]) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }

    for msg in messages {
        if let ChatMessage::User { content, .. } = msg {
            match content {
                MessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                MessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_chat_cross_parameters(
    req: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    // 1. Validate logprobs dependency
    if req.top_logprobs.is_some() && !req.logprobs {
        let mut e = validator::ValidationError::new("top_logprobs_requires_logprobs");
        e.message = Some("top_logprobs is only allowed when logprobs is enabled".into());
        return Err(e);
    }

    // 2. Validate stream_options dependency
    if req.stream_options.is_some() && !req.stream {
        let mut e = validator::ValidationError::new("stream_options_requires_stream");
        e.message =
            Some("The 'stream_options' parameter is only allowed when 'stream' is enabled".into());
        return Err(e);
    }

    // 3. Validate token limits - min <= max
    if let (Some(min), Some(max)) = (req.min_tokens, req.max_completion_tokens) {
        if min > max {
            let mut e = validator::ValidationError::new("min_tokens_exceeds_max");
            e.message = Some("min_tokens cannot exceed max_tokens/max_completion_tokens".into());
            return Err(e);
        }
    }

    // 4. Validate structured output conflicts
    let has_json_format = matches!(
        req.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    );

    if has_json_format && req.regex.is_some() {
        let mut e = validator::ValidationError::new("regex_conflicts_with_json");
        e.message = Some("cannot use regex constraint with JSON response format".into());
        return Err(e);
    }

    if has_json_format && req.ebnf.is_some() {
        let mut e = validator::ValidationError::new("ebnf_conflicts_with_json");
        e.message = Some("cannot use EBNF constraint with JSON response format".into());
        return Err(e);
    }

    // 5. Validate mutually exclusive structured output constraints
    let constraint_count = [
        req.regex.is_some(),
        req.ebnf.is_some(),
        matches!(req.response_format, Some(ResponseFormat::JsonSchema { .. })),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if constraint_count > 1 {
        let mut e = validator::ValidationError::new("multiple_constraints");
        e.message = Some("only one structured output constraint (regex, ebnf, or json_schema) can be active at a time".into());
        return Err(e);
    }

    // 6. Validate response format JSON schema name
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &req.response_format {
        if json_schema.name.is_empty() {
            let mut e = validator::ValidationError::new("json_schema_name_empty");
            e.message = Some("JSON schema name cannot be empty".into());
            return Err(e);
        }
    }

    // 7. Validate tool_choice requires tools (except for "none")
    if let Some(ref tool_choice) = req.tool_choice {
        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

        // Check if tool_choice is anything other than "none"
        let is_some_choice = !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None));

        if is_some_choice && !has_tools {
            let mut e = validator::ValidationError::new("tool_choice_requires_tools");
            e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
            return Err(e);
        }

        // Additional validation when tools are present
        if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
            match tool_choice {
                ToolChoice::Function { function, .. } => {
                    // Validate that the specified function name exists in tools
                    let function_exists = tools.iter().any(|tool| {
                        tool.tool_type == "function" && tool.function.name == function.name
                    });

                    if !function_exists {
                        let mut e =
                            validator::ValidationError::new("tool_choice_function_not_found");
                        e.message = Some(
                            format!(
                            "Invalid value for 'tool_choice': function '{}' not found in 'tools'.",
                            function.name
                        )
                            .into(),
                        );
                        return Err(e);
                    }
                }
                ToolChoice::AllowedTools {
                    mode,
                    tools: allowed_tools,
                    ..
                } => {
                    // Validate mode is "auto" or "required"
                    if mode != "auto" && mode != "required" {
                        let mut e = validator::ValidationError::new("tool_choice_invalid_mode");
                        e.message = Some(format!(
                            "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{mode}'."
                        ).into());
                        return Err(e);
                    }

                    // Validate that all ToolReferences are Function type (Chat API only supports function tools)
                    for tool_ref in allowed_tools {
                        match tool_ref {
                            ToolReference::Function { name } => {
                                // Validate that the function exists in tools array
                                let tool_exists = tools.iter().any(|tool| {
                                    tool.tool_type == "function" && tool.function.name == *name
                                });

                                if !tool_exists {
                                    let mut e = validator::ValidationError::new(
                                        "tool_choice_tool_not_found",
                                    );
                                    e.message = Some(
                                        format!(
                                            "Invalid value for 'tool_choice.tools': tool '{name}' not found in 'tools'."
                                        )
                                        .into(),
                                    );
                                    return Err(e);
                                }
                            }
                            _ => {
                                // Chat Completion API only supports function tools in tool_choice
                                let mut e = validator::ValidationError::new(
                                    "tool_choice_invalid_tool_type",
                                );
                                e.message = Some(
                                    format!(
                                        "Invalid value for 'tool_choice.tools': Chat Completion API only supports function tools, got '{}'.",
                                        tool_ref.identifier()
                                    )
                                    .into(),
                                );
                                return Err(e);
                            }
                        }
                    }
                }
                ToolChoice::Value(_) => {}
            }
        }
    }

    Ok(())
}

// ============================================================================
// Normalizable Implementation
// ============================================================================

impl Normalizable for ChatCompletionRequest {
    /// Normalize the request by applying migrations and defaults:
    /// 1. Migrate deprecated fields to their replacements
    /// 2. Clear deprecated fields and log warnings
    /// 3. Apply OpenAI defaults for tool_choice
    fn normalize(&mut self) {
        // Migrate deprecated max_tokens → max_completion_tokens
        #[expect(deprecated)]
        if self.max_completion_tokens.is_none() && self.max_tokens.is_some() {
            self.max_completion_tokens = self.max_tokens;
            self.max_tokens = None; // Clear deprecated field
        }

        // Migrate deprecated functions → tools
        #[expect(deprecated)]
        if self.tools.is_none() && self.functions.is_some() {
            tracing::warn!("functions is deprecated, use tools instead");
            self.tools = self.functions.as_ref().map(|functions| {
                functions
                    .iter()
                    .map(|func| Tool {
                        tool_type: "function".to_string(),
                        function: func.clone(),
                    })
                    .collect()
            });
            self.functions = None; // Clear deprecated field
        }

        // Migrate deprecated function_call → tool_choice
        #[expect(deprecated)]
        if self.tool_choice.is_none() && self.function_call.is_some() {
            tracing::warn!("function_call is deprecated, use tool_choice instead");
            self.tool_choice = self.function_call.as_ref().map(|fc| match fc {
                FunctionCall::None => ToolChoice::Value(ToolChoiceValue::None),
                FunctionCall::Auto => ToolChoice::Value(ToolChoiceValue::Auto),
                FunctionCall::Function { name } => ToolChoice::Function {
                    tool_type: "function".to_string(),
                    function: FunctionChoice { name: name.clone() },
                },
            });
            self.function_call = None; // Clear deprecated field
        }

        // Apply tool_choice defaults
        if self.tool_choice.is_none() {
            if let Some(tools) = &self.tools {
                let choice_value = if tools.is_empty() {
                    ToolChoiceValue::None
                } else {
                    ToolChoiceValue::Auto
                };
                self.tool_choice = Some(ToolChoice::Value(choice_value));
            }
            // If tools is None, leave tool_choice as None (don't set it)
        }
    }
}

// ============================================================================
// GenerationRequest Trait Implementation
// ============================================================================

impl GenerationRequest for ChatCompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Extract text from messages for routing decisions
        // Use a single buffer to avoid intermediate Vec<String> allocations
        let mut buffer = String::new();
        let mut has_content = false;

        for msg in &self.messages {
            match msg {
                ChatMessage::System { content, .. }
                | ChatMessage::User { content, .. }
                | ChatMessage::Tool { content, .. }
                | ChatMessage::Developer { content, .. } => {
                    if has_content && content.has_text() {
                        buffer.push(' ');
                    }
                    if content.append_text_to(&mut buffer) {
                        has_content = true;
                    }
                }
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    ..
                } => {
                    // Append main content
                    if let Some(c) = content {
                        if has_content && c.has_text() {
                            buffer.push(' ');
                        }
                        if c.append_text_to(&mut buffer) {
                            has_content = true;
                        }
                    }
                    // Append reasoning content
                    if let Some(reasoning) = reasoning_content {
                        if !reasoning.is_empty() {
                            if has_content {
                                buffer.push(' ');
                            }
                            buffer.push_str(reasoning);
                            has_content = true;
                        }
                    }
                }
                ChatMessage::Function { content, .. } => {
                    if !content.is_empty() {
                        if has_content {
                            buffer.push(' ');
                        }
                        buffer.push_str(content);
                        has_content = true;
                    }
                }
            }
        }

        buffer
    }
}

// ============================================================================
// Response Types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionResponse {
    /// Create a new builder for ChatCompletionResponse
    pub fn builder(
        id: impl Into<String>,
        model: impl Into<String>,
    ) -> ChatCompletionResponseBuilder {
        ChatCompletionResponseBuilder::new(id, model)
    }
}

/// Response message structure for ChatCompletionResponse (different from request ChatMessage)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatCompletionMessage {
    pub role: String, // Always "assistant" for responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning_content: Option<String>,
    // Note: function_call is deprecated and not included
    // Note: refusal, annotations, audio are not added yet
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter", "function_call"
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>, // Can be string or integer
    /// Hidden states from the model (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_states: Option<Vec<f32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatStreamChoice>,
    pub usage: Option<Usage>,
}

impl ChatCompletionStreamResponse {
    /// Create a new builder for ChatCompletionStreamResponse
    pub fn builder(
        id: impl Into<String>,
        model: impl Into<String>,
    ) -> ChatCompletionStreamResponseBuilder {
        ChatCompletionStreamResponseBuilder::new(id, model)
    }
}

/// Delta structure for streaming chat completion responses
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::{thinking_from_reasoning_effort, ChatCompletionRequest, ChatThinkingConfig};

    fn request_with_output_fields(fields: &[(&str, Value)]) -> ChatCompletionRequest {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let object = value.as_object_mut().expect("request must be an object");
        for (name, field_value) in fields {
            object.insert((*name).to_string(), field_value.clone());
        }
        serde_json::from_value(value).expect("request must deserialize")
    }

    #[test]
    fn thinking_from_reasoning_effort_maps_disable_values() {
        // "none"/"minimal" mean do-not-reason -> thinking OFF.
        assert_eq!(thinking_from_reasoning_effort(Some("none")), Some(false));
        assert_eq!(thinking_from_reasoning_effort(Some("minimal")), Some(false));
        // Level values do not toggle thinking on their own.
        assert_eq!(thinking_from_reasoning_effort(Some("low")), None);
        assert_eq!(thinking_from_reasoning_effort(Some("medium")), None);
        assert_eq!(thinking_from_reasoning_effort(Some("high")), None);
        // Unspecified / unknown -> defer.
        assert_eq!(thinking_from_reasoning_effort(None), None);
        assert_eq!(thinking_from_reasoning_effort(Some("bogus")), None);
    }

    #[test]
    fn reasoning_effort_accepts_scalar_json_and_rejects_other_types() {
        for (value, expected) in [
            (json!("high"), Some("high")),
            (json!(0.2), Some("0.2")),
            (json!(0.99), Some("0.99")),
            (Value::Null, None),
        ] {
            let request = request_with_output_fields(&[("reasoning_effort", value)]);
            assert_eq!(request.reasoning_effort.as_deref(), expected);
        }

        for value in [json!(true), json!([]), json!({"level": "high"})] {
            let mut request = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            });
            request["reasoning_effort"] = value;
            let error = serde_json::from_value::<ChatCompletionRequest>(request).unwrap_err();
            assert!(error
                .to_string()
                .contains("reasoning_effort must be a string, number, or null"));
        }
    }

    #[test]
    fn deepseek_thinking_config_is_strict_and_typed() {
        for (value, expected) in [
            (json!({"type": "enabled"}), ChatThinkingConfig::Enabled {}),
            (json!({"type": "disabled"}), ChatThinkingConfig::Disabled {}),
        ] {
            let request = request_with_output_fields(&[("thinking", value.clone())]);
            assert_eq!(request.thinking, Some(expected));
            assert!(!request.other.contains_key("thinking"));
            assert_eq!(
                serde_json::to_value(request).unwrap().get("thinking"),
                Some(&value)
            );
        }

        for value in [
            json!({"type": "auto"}),
            json!({"type": "enabled", "budget_tokens": 1024}),
            json!(true),
        ] {
            let mut request = json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            });
            request["thinking"] = value.clone();
            assert!(
                serde_json::from_value::<ChatCompletionRequest>(request).is_err(),
                "accepted invalid thinking value: {value}"
            );
        }
    }

    #[test]
    fn explicit_deepseek_thinking_wins_over_reasoning_effort() {
        let request = request_with_output_fields(&[
            ("thinking", json!({"type": "enabled"})),
            ("reasoning_effort", json!("none")),
        ]);
        assert_eq!(request.thinking_preference(), Some(true));

        let request = request_with_output_fields(&[
            ("thinking", json!({"type": "disabled"})),
            ("reasoning_effort", json!("max")),
        ]);
        assert_eq!(request.thinking_preference(), Some(false));
    }

    #[test]
    fn return_audio_preserves_explicit_values() {
        for fields in [vec![], vec![("return_audio", Value::Null)]] {
            let request = request_with_output_fields(&fields);
            assert_eq!(request.return_audio, None);
            assert!(!request.other.contains_key("return_audio"));
            let serialized = serde_json::to_value(request).expect("request must serialize");
            assert!(serialized.get("return_audio").is_none());
        }

        for value in [false, true] {
            let request = request_with_output_fields(&[("return_audio", json!(value))]);
            assert_eq!(request.return_audio, Some(value));
            assert!(!request.other.contains_key("return_audio"));
            let serialized = serde_json::to_value(request).expect("request must serialize");
            assert_eq!(serialized.get("return_audio"), Some(&Value::Bool(value)));
        }
    }

    #[test]
    fn chat_request_accepts_function_tool_without_parameters() {
        // https://github.com/smg-project/smg/issues/1974 — omitting
        // `parameters` is spec-legal and must not reject the request.
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "function", "function": {"name": "web_search", "description": ""}}
            ],
        });
        let request: ChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let tools = request.tools.expect("tools must be present");
        assert_eq!(tools[0].function.parameters, json!({}));
    }
}
