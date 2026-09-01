//! MiniMax contract rules (MiniMax-Provider-Verifier m3_format_check).

use std::collections::HashSet;

use crate::chat::{ChatCompletionRequest, ChatMessage};

/// Tool-protocol strictness for conversation history (MPV tests 16_08, 16_09,
/// 16_12): tool messages must answer a pending tool_call id, every tool_call
/// must be answered, and historical `arguments` must be valid JSON.
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    let mut open: HashSet<&str> = HashSet::new();

    for msg in &req.messages {
        match msg {
            ChatMessage::Assistant { tool_calls, .. } => {
                for tc in tool_calls.iter().flatten() {
                    if !open.insert(tc.id.as_str()) {
                        return Err(error(
                            "tool_call_id_duplicate",
                            format!("duplicate tool_call id '{}'", tc.id),
                        ));
                    }
                    let arguments = tc.function.arguments.as_deref().unwrap_or("");
                    if serde_json::from_str::<serde_json::Value>(arguments).is_err() {
                        return Err(error(
                            "tool_call_arguments_invalid_json",
                            format!("tool_call '{}' has non-JSON arguments", tc.id),
                        ));
                    }
                }
            }
            ChatMessage::Tool { tool_call_id, .. } if !open.remove(tool_call_id.as_str()) => {
                return Err(error(
                    "tool_call_id_mismatch",
                    format!("tool message references unknown tool_call id '{tool_call_id}'"),
                ));
            }
            _ => {}
        }
    }

    if let Some(id) = open.iter().next() {
        return Err(error(
            "tool_call_unanswered",
            format!("tool_call '{id}' has no matching tool message"),
        ));
    }

    Ok(())
}

/// Rewrite `root` messages to system messages for dispatch: upstream MiniMax
/// serving stacks take the top-priority instruction as the leading system
/// message, and api.minimax.io itself rejects the literal role.
pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    for msg in &mut req.messages {
        if let ChatMessage::Root { content, name } = msg {
            *msg = ChatMessage::System {
                content: std::mem::take(content),
                name: name.take(),
                ext: Default::default(),
            };
        }
    }
}

fn error(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}
