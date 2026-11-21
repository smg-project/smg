use serde_json::Value;

use crate::router::topology::error::TopologyError;
/// Extracts the conversation prefix from a JSON request payload.
///
/// This function looks for the `"messages"` array and extracts all messages
/// up to (but excluding) the last user message. If only a single‑turn
/// request exists, it falls back to the full message content.
///
/// # Returns
/// - `Ok(String)` containing the prefix with role tags (`system:`, `user:`, `assistant:`)
/// - `Err(TopologyError)` if no valid content is found
pub fn extract_conversation_prefix(json_bytes: &[u8]) -> Result<String, TopologyError> {
    if json_bytes.is_empty() {
        return Err(TopologyError::MissingContent);
    }

    let value: Value = serde_json::from_slice(json_bytes)
        .map_err(|e| TopologyError::InvalidJson(e.to_string()))?;

    if let Some(content) = value.get("content") {
        if let Some(text) = extract_text_from_content(content) {
            if !text.is_empty() {
                return Ok(text);
            }
        }
    }

    if let Some(messages) = value.get("messages").and_then(|m| m.as_array()) {
        if messages.is_empty() {
            return Err(TopologyError::MissingContent);
        }

        let mut last_user_idx = None;
        for (i, msg) in messages.iter().enumerate() {
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                if role == "user" {
                    last_user_idx = Some(i);
                }
            }
        }

        let prefix_end = match last_user_idx {
            Some(idx) if idx > 0 => idx,
            Some(idx) if idx == 0 => messages.len(),
            _ => messages.len(),
        };

        let prefix_messages = &messages[..prefix_end];
        let messages_to_hash = if prefix_messages.is_empty() {
            messages
        } else {
            prefix_messages
        };

        let mut prefix = String::new();
        for msg in messages_to_hash {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("unknown");
            if let Some(content) = msg.get("content") {
                if let Some(text) = extract_text_from_content(content) {
                    prefix.push_str(role);
                    prefix.push(':');
                    prefix.push_str(&text);
                    prefix.push('\n');
                }
            }
        }

        if prefix.is_empty() {
            return Err(TopologyError::MissingContent);
        }

        return Ok(prefix);
    }

    Err(TopologyError::MissingContent)
}

fn extract_text_from_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }

    if let Some(blocks) = content.as_array() {
        let mut combined = String::new();
        for block in blocks {
            if let Some(block_type) = block.get("type").and_then(|t| t.as_str()) {
                if block_type == "text" {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        if !combined.is_empty() {
                            combined.push(' ');
                        }
                        combined.push_str(text);
                    }
                }
            }
        }
        if !combined.is_empty() {
            return Some(combined);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_single_turn_query() {
        let json = br#"{"messages": [{"role": "user", "content": "What is 2+2?"}]}"#;
        let prefix = extract_conversation_prefix(json).unwrap();
        assert!(prefix.contains("user:What is 2+2?"));
    }

    #[test]
    fn handles_system_and_user() {
        let json = br#"{
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"}
            ]
        }"#;
        let prefix = extract_conversation_prefix(json).unwrap();
        assert!(prefix.contains("system:You are helpful."));
        assert!(!prefix.contains("user:What is 2+2?"));
    }

    #[test]
    fn handles_multi_turn() {
        let json = br#"{
            "messages": [
                {"role": "system", "content": "You are a Rust expert."},
                {"role": "user", "content": "What is ownership?"},
                {"role": "assistant", "content": "Ownership is Rust's memory model."},
                {"role": "user", "content": "Explain borrowing."}
            ]
        }"#;
        let prefix = extract_conversation_prefix(json).unwrap();
        assert!(prefix.contains("system:You are a Rust expert."));
        assert!(prefix.contains("user:What is ownership?"));
        assert!(prefix.contains("assistant:Ownership is Rust's memory model."));
        assert!(!prefix.contains("Explain borrowing."));
    }

    #[test]
    fn handles_empty_messages() {
        let json = br#"{"messages": []}"#;
        assert!(matches!(
            extract_conversation_prefix(json),
            Err(TopologyError::MissingContent)
        ));
    }
}
