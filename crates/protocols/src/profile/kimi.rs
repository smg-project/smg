//! Kimi/Moonshot contract rules (Kimi-Vendor-Verifier).

use crate::chat::{ChatCompletionRequest, ChatMessage};

/// K3 dynamic tools may only be declared on system messages; any other role
/// carrying a `tools` key must be rejected (KVV test_dynamic_tools).
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    for msg in &req.messages {
        let (role, has_tools) = match msg {
            ChatMessage::User { ext, .. } => {
                ("user", ext.tools.as_ref().is_some_and(|t| !t.is_empty()))
            }
            ChatMessage::Assistant { ext, .. } => (
                "assistant",
                ext.tools.as_ref().is_some_and(|t| !t.is_empty()),
            ),
            _ => continue,
        };
        if has_tools {
            let mut e = validator::ValidationError::new("tools_role_restricted");
            e.message =
                Some(format!("'tools' is not allowed on a message with role '{role}'").into());
            return Err(e);
        }
    }
    Ok(())
}
