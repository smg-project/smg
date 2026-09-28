use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParseOutput {
    pub thinking: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// The parser never places matched delimiters or unmatched wire bytes here.
    pub wire_bytes: Vec<u8>,
}

impl ParseOutput {
    pub fn is_empty(&self) -> bool {
        self.thinking.is_empty()
            && self.content.is_empty()
            && self.tool_calls.is_empty()
            && self.wire_bytes.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.thinking.push_str(&other.thinking);
        self.content.push_str(&other.content);
        self.tool_calls.extend(other.tool_calls);
        self.wire_bytes.extend(other.wire_bytes);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
    /// Result of applying the template's structural transform.
    pub transformed: Value,
}
