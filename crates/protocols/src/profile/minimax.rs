//! MiniMax contract rules (MiniMax-Provider-Verifier m3_format_check).

use std::collections::HashSet;

use crate::chat::{ChatCompletionRequest, ChatMessage};

/// Tool-protocol strictness for conversation history (MPV tests 16_08, 16_09,
/// 16_12): tool messages must answer a pending tool_call id, every tool_call
/// must be answered, ids are unique across the whole conversation, and
/// historical `arguments`, when present, must be a JSON object.
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut pending: HashSet<&str> = HashSet::new();
    // Call order, so the reported unanswered id is stable.
    let mut order: Vec<&str> = Vec::new();

    for msg in &req.messages {
        match msg {
            ChatMessage::Assistant { tool_calls, .. } => {
                for tc in tool_calls.iter().flatten() {
                    if !seen.insert(tc.id.as_str()) {
                        return Err(error(
                            "tool_call_id_duplicate",
                            format!("duplicate tool_call id '{}'", tc.id),
                        ));
                    }
                    pending.insert(tc.id.as_str());
                    order.push(tc.id.as_str());
                    // An empty string is how several providers spell a call without arguments.
                    let arguments = tc.function.arguments.as_deref();
                    if let Some(arguments) = arguments.filter(|a| !a.trim().is_empty()) {
                        if serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                            arguments,
                        )
                        .is_err()
                        {
                            return Err(error(
                                "tool_call_arguments_invalid_json",
                                format!("tool_call '{}' arguments are not a JSON object", tc.id),
                            ));
                        }
                    }
                }
            }
            ChatMessage::Tool { tool_call_id, .. } => {
                let answered = pending.remove(tool_call_id.as_str());
                if !answered {
                    return Err(error(
                        "tool_call_id_mismatch",
                        format!("no pending tool_call with id '{tool_call_id}'"),
                    ));
                }
            }
            _ => {}
        }
    }

    if let Some(id) = order.iter().find(|id| pending.contains(*id)) {
        return Err(error(
            "tool_call_unanswered",
            format!("tool_call '{id}' has no matching tool message"),
        ));
    }

    Ok(())
}

/// Rewrite every `root` message to a leading system message for dispatch:
/// upstream MiniMax serving stacks take the top-priority instruction as the
/// leading system message, and api.minimax.io itself rejects the literal
/// role. Roots are hoisted above everything else in their original order.
pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    let is_root = |msg: &ChatMessage| matches!(msg, ChatMessage::Root { .. });
    if !req.messages.iter().any(is_root) {
        return;
    }
    let (roots, rest): (Vec<_>, Vec<_>) = req.messages.drain(..).partition(is_root);
    tracing::debug!(
        model = %req.model,
        count = roots.len(),
        "rewrote role root to leading system messages"
    );
    req.messages = roots
        .into_iter()
        .map(|msg| match msg {
            ChatMessage::Root { content, name } => ChatMessage::System {
                content,
                name,
                ext: Default::default(),
            },
            other => other,
        })
        .chain(rest)
        .collect();
}

fn error(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}
