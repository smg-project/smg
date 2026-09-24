//! Bounded DeepSeek V4 / V4.1 Chat dialect, calibrated by
//! smg-project/deepseek-provider-verifier against deepseek-flash and deepseek-v4-pro.
//! See https://api-docs.deepseek.com/api/create-chat-completion/.
//!
//! Older V3/R1 models and opaque aliases retain the baseline. This profile
//! does not claim the vendor's context window, sampling behavior or JSON
//! constraints for a self-hosted worker, nor enforce documented history and
//! identifier restrictions the official API did not enforce in calibration.

use serde_json::Value;

use crate::{
    chat::{ChatCompletionRequest, ThinkingType},
    common::{ToolChoice, ToolChoiceValue},
};

pub(super) fn matches_model(segment: &str) -> bool {
    [
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4.1-flash",
    ]
    .iter()
    .any(|model| segment.eq_ignore_ascii_case(model))
}

fn is_v41_model(model: &str) -> bool {
    model
        .split('/')
        .any(|segment| segment.eq_ignore_ascii_case("deepseek-v4.1-flash"))
}

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    let is_v41 = is_v41_model(&req.model);
    for effort in [
        req.reasoning_effort.as_mut(),
        req.thinking
            .as_mut()
            .and_then(|thinking| thinking.effort.as_mut()),
    ]
    .into_iter()
    .flatten()
    {
        match effort.as_str() {
            "minimal" => *effort = "low".into(),
            "medium" => *effort = "high".into(),
            // V4.1 has its own native xhigh budget (75 rather than high's 50).
            "xhigh" if !is_v41 => *effort = "high".into(),
            _ => {}
        }
    }
    // Native V4 and V4.1 renderers have different defaults and effort
    // precedence. Resolve the provider contract once and pass the same
    // highest-priority toggle to rendering, parser arming and validation.
    let enabled = thinking_enabled(req);
    let kwargs = req.chat_template_kwargs.get_or_insert_default();
    if kwargs.get("thinking").is_none_or(Value::is_null) {
        kwargs.insert("thinking".into(), Value::Bool(enabled));
    }
}

pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    // Never let a malformed override block projection and make validation
    // disagree with a renderer that ignores that value.
    if let Some(kwargs) = &req.chat_template_kwargs {
        for key in ["thinking", "enable_thinking"] {
            if kwargs
                .get(key)
                .is_some_and(|value| !value.is_null() && !value.is_boolean())
            {
                return Err(error(
                    "invalid_thinking_override",
                    "DeepSeek thinking template overrides must be booleans or null",
                ));
            }
        }
        if let (Some(thinking), Some(alias)) = (
            kwargs.get("thinking").and_then(Value::as_bool),
            kwargs.get("enable_thinking").and_then(Value::as_bool),
        ) {
            if thinking != alias {
                return Err(error(
                    "conflicting_thinking_overrides",
                    "DeepSeek thinking and enable_thinking template overrides must agree",
                ));
            }
        }
    }
    if req
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.r#type == Some(ThinkingType::Adaptive))
    {
        return Err(error(
            "thinking_type_not_supported",
            "thinking.type must be enabled or disabled for DeepSeek V4",
        ));
    }
    // Validate both spellings, including one hidden by the other's precedence.
    for effort in [
        req.reasoning_effort.as_deref(),
        req.thinking
            .as_ref()
            .and_then(|thinking| thinking.effort.as_deref()),
    ]
    .into_iter()
    .flatten()
    {
        // The V4.1 native renderer restores ASCII-digit strings to 1..=100
        // budgets. Other V4 renderers do not support that extension.
        let is_v41 = is_v41_model(&req.model);
        let native_budget = is_v41
            && effort.bytes().all(|byte| byte.is_ascii_digit())
            && effort
                .parse::<u8>()
                .is_ok_and(|budget| (1..=100).contains(&budget));
        if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&effort)
            && !native_budget
        {
            return Err(error(
                "reasoning_effort_not_allowed",
                if is_v41 {
                    "DeepSeek V4.1 reasoning effort must be none, low, high, xhigh, max or an integer budget from 1 to 100 (minimal and medium aliases are accepted)"
                } else {
                    "DeepSeek V4 reasoning effort must be none, low, high or max (minimal, medium and xhigh aliases are accepted)"
                },
            ));
        }
    }
    let forced = match req.tool_choice.as_ref() {
        Some(ToolChoice::Value(ToolChoiceValue::Required) | ToolChoice::Function { .. }) => true,
        Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
        _ => false,
    };
    if thinking_enabled(req) && forced {
        return Err(error(
            "tool_choice_not_supported",
            "DeepSeek V4 requires thinking to be disabled for forced tool choice",
        ));
    }
    Ok(())
}

/// SMG template overrides stay explicit; public `thinking.type` then
/// decides ahead of public effort, with thinking enabled by default.
fn thinking_enabled(req: &ChatCompletionRequest) -> bool {
    let kwargs = req.chat_template_kwargs.as_ref();
    kwargs
        .and_then(|kwargs| {
            kwargs
                .get("thinking")
                .and_then(Value::as_bool)
                .or_else(|| kwargs.get("enable_thinking").and_then(Value::as_bool))
        })
        .or_else(|| {
            kwargs
                .and_then(|kwargs| kwargs.get("reasoning_effort").and_then(Value::as_str))
                .and_then(|effort| match effort {
                    "none" | "minimal" => Some(false),
                    "low" | "high" | "xhigh" | "max" => Some(true),
                    _ => None,
                })
        })
        .or_else(|| req.thinking_toggle())
        .unwrap_or_else(|| req.effective_reasoning_effort() != Some("none"))
}

fn error(code: &'static str, message: &'static str) -> validator::ValidationError {
    let mut error = validator::ValidationError::new(code);
    error.message = Some(message.into());
    error
}
