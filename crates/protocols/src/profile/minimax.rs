//! MiniMax contract rules (MiniMax-Provider-Verifier m3_format_check).

use std::collections::HashSet;

use crate::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::ContentPart,
};

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

/// Hoist every `root` message above everything else and, when there are
/// several, merge them into one leading root (their texts joined by a blank
/// line in their original order, a parts list keeping its parts, under the
/// first one's name), with the role kept. The chat template of the newest
/// model this profile targets reads a leading `root` message into its
/// high-priority system slot and a `system` or `developer` message right
/// after it into its developer slot (the template's own role mapping:
/// "root -> system sp (high priority), system/developer -> developer sp
/// (low priority)"), which is what the engine's own server renders for the
/// request; it reads one root, hence the merge. A root rewritten to a
/// leading `system` message, as this profile did before, landed in the
/// developer slot under the template's default identity, and a system
/// message that followed it was dropped. A renderer whose template has no
/// root branch (the vendor's earlier templates read `system` only) hands the
/// root on as the leading system message itself.
pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    let is_root = |msg: &ChatMessage| matches!(msg, ChatMessage::Root { .. });
    let root_count = req.messages.iter().filter(|msg| is_root(msg)).count();
    if root_count == 0 || (root_count == 1 && req.messages.first().is_some_and(is_root)) {
        return;
    }
    let (roots, rest): (Vec<_>, Vec<_>) = req.messages.drain(..).partition(is_root);
    tracing::debug!(
        model = %req.model,
        count = root_count,
        "merged the root messages into one leading root"
    );
    req.messages = merge_roots(roots).into_iter().chain(rest).collect();
}

/// The root messages as one: the first one's name, the contents joined in
/// order.
fn merge_roots(roots: Vec<ChatMessage>) -> Option<ChatMessage> {
    let mut merged: Option<(MessageContent, Option<String>)> = None;
    for root in roots {
        let ChatMessage::Root { content, name } = root else {
            continue;
        };
        merged = Some(match merged {
            None => (content, name),
            Some((joined, first_name)) => (join_content(joined, content), first_name),
        });
    }
    merged.map(|(content, name)| ChatMessage::Root { content, name })
}

/// Two contents as one: texts joined by a blank line (an empty text adds
/// nothing); when either is a parts list, the parts of both in order, a text
/// as one text part.
fn join_content(first: MessageContent, second: MessageContent) -> MessageContent {
    match (first, second) {
        (MessageContent::Text(first), MessageContent::Text(second)) => {
            MessageContent::Text(match (first.is_empty(), second.is_empty()) {
                (true, _) => second,
                (_, true) => first,
                (false, false) => format!("{first}\n\n{second}"),
            })
        }
        (first, second) => {
            let mut parts = into_parts(first);
            parts.extend(into_parts(second));
            MessageContent::Parts(parts)
        }
    }
}

fn into_parts(content: MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::Parts(parts) => parts,
        MessageContent::Text(text) => vec![ContentPart::Text { text }],
    }
}

fn error(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}
