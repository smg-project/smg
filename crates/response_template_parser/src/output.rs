use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Fields parsed from a response, or from one `feed` or `finish` call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParseOutput {
    /// Reasoning text.
    pub thinking: String,
    /// Answer text.
    pub content: String,
    /// Parsed tool calls, in output order.
    pub tool_calls: Vec<ToolCall>,
    /// The parser never places matched delimiters or unmatched wire bytes here.
    pub wire_bytes: Vec<u8>,
}

impl ParseOutput {
    /// Whether there is no text, tool call or wire byte.
    pub fn is_empty(&self) -> bool {
        self.thinking.is_empty()
            && self.content.is_empty()
            && self.tool_calls.is_empty()
            && self.wire_bytes.is_empty()
    }

    /// Append `other` to this output, field by field.
    pub fn merge(&mut self, other: Self) {
        self.thinking.push_str(&other.thinking);
        self.content.push_str(&other.content);
        self.tool_calls.extend(other.tool_calls);
        self.wire_bytes.extend(other.wire_bytes);
    }
}

/// One parsed tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Function name captured by the opener.
    pub name: String,
    /// Arguments by key; each value is a string.
    pub arguments: Map<String, Value>,
    /// Result of applying the template's structural transform.
    pub transformed: Value,
}
