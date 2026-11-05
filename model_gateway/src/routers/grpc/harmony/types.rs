//! Shared types for Harmony pipeline

use openai_protocol::common::ToolCall;
use serde::{Deserialize, Serialize};

/// Output from Harmony encoding process
///
/// Contains the encoded input_ids, stop tokens and the selection text for worker routing.
#[derive(Debug, Clone)]
pub(crate) struct HarmonyBuildOutput {
    /// Encoded token IDs to send to the model
    pub input_ids: Vec<u32>,

    /// Stop token IDs for this model (injected into sampling params)
    pub stop_token_ids: Vec<u32>,

    /// Selection text for worker routing (concise snippet from last user message)
    pub selection_text: String,
}

/// Parsed output from all three Harmony channels
///
/// Represents the complete response after parsing analysis, commentary, and final channels.
#[derive(Debug, Clone)]
pub(crate) struct HarmonyChannelOutput {
    /// Analysis/reasoning content (from analysis channel)
    pub analysis: Option<String>,

    /// Tool calls (from commentary channel)
    pub commentary: Option<Vec<ToolCall>>,

    /// Final text content (from final channel)
    pub final_text: String,

    /// Finish reason
    pub finish_reason: String,

    /// Number of reasoning tokens (from analysis and commentary channels)
    pub reasoning_token_count: u32,
}

/// Streaming delta for SSE responses
///
/// Represents incremental updates as tokens are parsed from the stream.
#[derive(Debug, Clone)]
pub(crate) struct HarmonyChannelDelta {
    /// Delta for analysis/reasoning content
    pub analysis_delta: Option<String>,

    /// Delta for tool calls
    pub commentary_delta: Option<ToolCallDelta>,

    /// Delta for final text content
    pub final_delta: Option<String>,
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<FunctionDelta>,
}

/// Function call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
