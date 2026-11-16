use serde::{Deserialize, Serialize};

use crate::common::Tool;

/// Request to parse function calls from model output text.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ParseFunctionCallRequest {
    /// The text to parse for function calls.
    pub text: String,
    /// The parser type/name to use for parsing (e.g., "json", "qwen").
    pub tool_call_parser: String,
    /// The list of available tools that the model can call.
    pub tools: Vec<Tool>,
}

/// Response from parsing function calls.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ParseFunctionCallResponse {
    /// Remaining text after extracting function calls.
    pub remaining_text: String,
    /// Extracted tool calls.
    pub tool_calls: Vec<serde_json::Value>,
    /// Whether parsing succeeded.
    pub success: bool,
}

/// Request to separate reasoning from normal text in model output.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SeparateReasoningRequest {
    /// The text to parse for reasoning content.
    pub text: String,
    /// The parser type/name to use for reasoning detection (e.g., "qwen3", "deepseek_r1").
    pub reasoning_parser: String,
}

/// Response from separating reasoning.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SeparateReasoningResponse {
    /// Normal (non-reasoning) text.
    pub normal_text: String,
    /// Extracted reasoning text.
    pub reasoning_text: String,
    /// Whether parsing succeeded.
    pub success: bool,
}
