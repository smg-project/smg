//! Kimi/Moonshot contract rules (Kimi-Vendor-Verifier).

use crate::{
    chat::{ChatCompletionRequest, ChatMessage},
    ext::kimi::DeclaredTools,
};

/// K3 dynamic tools may be declared on system messages, and on developer
/// messages, which the OpenAI spec defines as the successor of `system` and
/// this crate reads the same way; a declaration that is not a list of tools is
/// rejected. A `tools` field set on a user or assistant message is rejected,
/// an empty list included, while null counts as absent: the contract keys on
/// the field being set, not on its contents (KVV test_dynamic_tools; the
/// verifier has no developer case, so that role follows `system`). Tool and
/// function messages capture no such key, so serde drops it there as it
/// always did.
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    for msg in &req.messages {
        let (code, role, message) = match msg {
            ChatMessage::User { ext, .. } if ext.tools.is_some() => {
                ("tools_role_restricted", "user", "is not allowed")
            }
            ChatMessage::Assistant { ext, .. } if ext.tools.is_some() => {
                ("tools_role_restricted", "assistant", "is not allowed")
            }
            ChatMessage::System { ext, .. } if is_malformed(ext.tools.as_ref()) => (
                "tools_malformed",
                "system",
                "must be a list of tool declarations",
            ),
            ChatMessage::Developer { ext, .. } if is_malformed(ext.tools.as_ref()) => (
                "tools_malformed",
                "developer",
                "must be a list of tool declarations",
            ),
            _ => continue,
        };
        let mut e = validator::ValidationError::new(code);
        e.message = Some(format!("'tools' on a message with role '{role}' {message}").into());
        return Err(e);
    }
    Ok(())
}

fn is_malformed(tools: Option<&DeclaredTools>) -> bool {
    matches!(tools, Some(DeclaredTools::Malformed(_)))
}
