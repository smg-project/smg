//! Bounded DeepSeek V4 / V4.1 Chat dialect, calibrated by
//! smg-project/deepseek-provider-verifier against deepseek-flash and deepseek-v4-pro.
//! See <https://api-docs.deepseek.com/api/create-chat-completion/>.
//!
//! Older V3/R1 models and opaque aliases retain the baseline. This profile
//! does not claim the vendor's context window, sampling behavior or JSON
//! constraints for a self-hosted worker, nor enforce documented history and
//! identifier restrictions the official API did not enforce in calibration,
//! nor the vendor's refusal of a forced tool choice while thinking is on: a
//! self-hosted engine serves that call (the gateway shapes the call's grammar
//! for where the engine starts it, after the thought or around it), and the
//! vendor's own API refuses it by itself where it still does.

use serde_json::Value;

use crate::{
    chat::{ChatCompletionRequest, ThinkingType},
    common::invalid,
};

const V41_MODELS: [&str; 2] = ["deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"];

pub(super) fn matches_model(segment: &str) -> bool {
    ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro"]
        .iter()
        .chain(&V41_MODELS)
        .any(|model| segment.eq_ignore_ascii_case(model))
}

/// Whether any `/`-separated segment of a model id is a V4.1 name.
fn is_v41_name(model: &str) -> bool {
    model.split('/').any(|segment| {
        V41_MODELS
            .iter()
            .any(|v41| segment.eq_ignore_ascii_case(v41))
    })
}

/// Whether `model` is served as V4.1: an opaque alias of a V4.1 served
/// model is, and so is an opaque served model whose vendor-named aliases
/// are all V4.1 names, whichever name the request carries, and so is one
/// vendor's name for a V4.1 model; an alias named for the vendor it resolves
/// to and an unregistered name are read as they are. Every reader of the
/// version goes through here, so normalization, the thinking toggle and
/// validation agree with each other.
fn is_v41_model(model: &str) -> bool {
    super::served_under(model, is_v41_name)
}

/// The effort names the vendor's API documents as aliases of its levels:
/// `minimal` means `low`; `medium` and `xhigh` mean `high`.
fn vendor_alias(effort: &str) -> Option<&'static str> {
    match effort {
        "minimal" => Some("low"),
        "medium" | "xhigh" => Some("high"),
        _ => None,
    }
}

/// The request's effort spellings, the top-level field and `thinking.effort`,
/// whichever are present.
fn effort_fields(req: &mut ChatCompletionRequest) -> impl Iterator<Item = &mut String> {
    [
        req.reasoning_effort.as_mut(),
        req.thinking
            .as_mut()
            .and_then(|thinking| thinking.effort.as_mut()),
    ]
    .into_iter()
    .flatten()
}

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    let is_v41 = is_v41_model(&req.model);
    for effort in effort_fields(req) {
        // V4.1 accepts medium and xhigh natively (its renderer gives each
        // the budget of the checkpoint's own table).
        if is_v41 && matches!(effort.as_str(), "medium" | "xhigh") {
            continue;
        }
        if let Some(level) = vendor_alias(effort) {
            *effort = level.into();
        }
    }
    // Native V4 and V4.1 renderers have different defaults and effort
    // precedence. Resolve the provider contract once and pass the same
    // highest-priority toggle to rendering, parser arming and validation.
    let enabled = thinking_enabled(req, is_v41);
    let kwargs = req.chat_template_kwargs.get_or_insert_default();
    if kwargs.get("thinking").is_none_or(Value::is_null) {
        kwargs.insert("thinking".into(), Value::Bool(enabled));
    }
}

/// The vendor's reading of `medium` for a request the gateway forwards to an
/// upstream that renders the prompt itself. [`normalize_chat`] keeps `medium`
/// on V4.1 for the gateway's own renderer, which gives it the checkpoint
/// template's budget; the vendor's API reads it as `high`, as the profile did
/// before the renderer knew the level, and a server that renders with the
/// model's own prompt encoder has no `medium` at all. `xhigh` goes out as
/// sent, as it always did (those encoders have it), and a chat-template kwarg
/// is the client's word to the upstream's template and goes out as sent.
pub(super) fn alias_effort_for_upstream(req: &mut ChatCompletionRequest) {
    for effort in effort_fields(req).filter(|effort| effort.as_str() == "medium") {
        if let Some(level) = vendor_alias(effort) {
            *effort = level.into();
        }
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
                return Err(invalid(
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
                return Err(invalid(
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
        return Err(invalid(
            "thinking_type_not_supported",
            "thinking.type must be enabled or disabled for DeepSeek V4",
        ));
    }
    // The V4.1 native renderer restores ASCII-digit strings to 1..=100
    // budgets. Other V4 renderers do not support that extension.
    let is_v41 = is_v41_model(&req.model);
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
        let native_budget = is_v41
            && effort.bytes().all(|byte| byte.is_ascii_digit())
            && effort
                .parse::<u8>()
                .is_ok_and(|budget| (1..=100).contains(&budget));
        if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&effort)
            && !native_budget
        {
            return Err(invalid(
                "reasoning_effort_not_allowed",
                if is_v41 {
                    "DeepSeek V4.1 reasoning effort must be none, low, medium, high, xhigh, max or an integer budget from 1 to 100 (the minimal alias is accepted)"
                } else {
                    "DeepSeek V4 reasoning effort must be none, low, high or max (minimal, medium and xhigh aliases are accepted)"
                },
            ));
        }
    }
    Ok(())
}

/// SMG template overrides stay explicit; public `thinking.type` then
/// decides ahead of public effort, with thinking enabled by default.
/// `is_v41` is the request's version, read once by the caller.
fn thinking_enabled(req: &ChatCompletionRequest, is_v41: bool) -> bool {
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
                    // A level of V4.1's own table; under V4 an alias the
                    // renderer maps, with no opinion on the mode.
                    "medium" if is_v41 => Some(true),
                    _ => None,
                })
        })
        .or_else(|| req.thinking_toggle())
        .unwrap_or_else(|| req.effective_reasoning_effort() != Some("none"))
}
