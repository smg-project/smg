//! Kimi/Moonshot contract rules (Kimi-Vendor-Verifier).

use crate::chat::{ChatCompletionRequest, ChatMessage};

/// K3 dynamic tools may only be declared on system messages. A `tools` key on
/// a user, assistant or developer message is rejected, an empty list included:
/// the contract keys on the key being declared, not on its contents (KVV
/// test_dynamic_tools). Tool and function messages capture no such key, so
/// serde drops it there as it always did.
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    for msg in &req.messages {
        let role = match msg {
            ChatMessage::User { ext, .. } if ext.tools.is_some() => "user",
            ChatMessage::Assistant { ext, .. } if ext.tools.is_some() => "assistant",
            ChatMessage::Developer { ext, .. } if ext.tools.is_some() => "developer",
            _ => continue,
        };
        let mut e = validator::ValidationError::new("tools_role_restricted");
        e.message = Some(format!("'tools' is not allowed on a message with role '{role}'").into());
        return Err(e);
    }
    Ok(())
}
