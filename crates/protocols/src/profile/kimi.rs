//! Kimi/Moonshot contract rules (Kimi-Vendor-Verifier).
//!
//! The sampling and thinking rules are K3's alone; other Kimi and Moonshot
//! models keep OpenAI's ranges. The stream usage default is Kimi-wide, as the
//! verifier expects. Only requests entering through `ValidatedJson` reach
//! these rules; the Responses conversion builds its chat request without them.

use std::collections::HashSet;

use crate::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent, ThinkingType},
    common::Tool,
    ext::kimi::DeclaredTools,
};

/// Tool names longer than this are rejected (KVV pins 257).
const MAX_TOOL_NAME_LEN: usize = 256;

/// Sampling defaults the K3 serving requirements fix to the official values;
/// applied when the client omits the field so a self-hosted engine does not
/// substitute its own. `max_tokens` is left to the engine: the manual's
/// documented default is inconsistent (262144 vs 32768).
const DEFAULT_TEMPERATURE: f32 = TEMPERATURES[2];
const DEFAULT_TOP_P: f32 = TOP_P;

/// Sampling values the verifier requires accepted (KVV tests/params
/// IMMUTABLE_PARAMS, thinking and non-thinking sets combined); anything else
/// must be rejected. The official API pins temperature to 1.0 alone, but the
/// verifier asserts 0.0 and 0.6 accepted too.
const TEMPERATURES: [f32; 3] = [0.0, 0.6, 1.0];
const TOP_P: f32 = 0.95;

/// `thinking.effort` levels the K3 renderer accepts.
const THINKING_EFFORTS: [&str; 3] = ["low", "high", "max"];

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    // KVV tests/prompt_tokens reads usage from streams sent without stream_options.
    if req.stream {
        req.stream_options
            .get_or_insert_default()
            .include_usage
            .get_or_insert(true);
    }
    if !is_k3(&req.model) {
        return;
    }
    req.temperature.get_or_insert(DEFAULT_TEMPERATURE);
    req.top_p.get_or_insert(DEFAULT_TOP_P);
    req.presence_penalty.get_or_insert(0.0);
    req.frequency_penalty.get_or_insert(0.0);
    req.n.get_or_insert(1);
}

pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if is_k3(&req.model) {
        validate_sampling(req)?;
        validate_thinking(req)?;
    }
    validate_message_tools(req)
}

/// K3 dynamic tools: the tools declared on system and developer messages, in
/// message order. They stand next to the request-level `tools` everywhere a
/// tool name is resolved after validation (the response spec, tool-call
/// parsing, the `tool_choice` constraint). Malformed declarations are
/// [`validate_message_tools`]'s business and are skipped here.
pub(super) fn dynamic_tools(req: &ChatCompletionRequest) -> impl Iterator<Item = &Tool> {
    req.messages.iter().flat_map(|message| match message {
        ChatMessage::System { ext, .. } => ext
            .tools
            .as_ref()
            .and_then(DeclaredTools::typed)
            .unwrap_or_default(),
        ChatMessage::Developer { ext, .. } => ext
            .tools
            .as_ref()
            .and_then(DeclaredTools::typed)
            .unwrap_or_default(),
        _ => &[],
    })
}

/// K3 dynamic tools (KVV test_dynamic_tools): rejected on user/assistant; on system
/// and developer they must be unique `function` tools with identifier names and empty content.
fn validate_message_tools(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    // Request-level names only seed the duplicate set; their shape is not judged.
    let mut seen: HashSet<&str> = req
        .tools
        .iter()
        .flatten()
        .map(|t| t.function.name.as_str())
        .collect();
    for (i, msg) in req.messages.iter().enumerate() {
        let (role, content, tools) = match msg {
            ChatMessage::User { ext, .. } if ext.tools.is_some() => {
                return Err(on_role("tools_role_restricted", "user", "is not allowed"));
            }
            ChatMessage::Assistant { ext, .. } if ext.tools.is_some() => {
                return Err(on_role(
                    "tools_role_restricted",
                    "assistant",
                    "is not allowed",
                ));
            }
            ChatMessage::System { content, ext, .. } => ("system", content, ext.tools.as_ref()),
            ChatMessage::Developer { content, ext, .. } => {
                ("developer", content, ext.tools.as_ref())
            }
            _ => continue,
        };
        let tools = match tools {
            None => continue,
            Some(DeclaredTools::Malformed(_)) => {
                return Err(on_role(
                    "tools_malformed",
                    role,
                    "must be a list of tool declarations",
                ));
            }
            Some(DeclaredTools::Tools(tools)) => tools,
        };
        // An empty list renders as a plain system message, so its content stays legal.
        if tools.is_empty() {
            continue;
        }
        if has_content(content) {
            return Err(on_role(
                "tools_content_conflict",
                role,
                "requires empty content",
            ));
        }
        for (j, tool) in tools.iter().enumerate() {
            let at = || format!("messages[{i}].tools[{j}]");
            if tool.tool_type != "function" {
                return Err(error(
                    "tool_type_unsupported",
                    format!("{}: tool type must be 'function'", at()),
                ));
            }
            let name = tool.function.name.as_str();
            if !is_valid_tool_name(name) {
                return Err(error(
                    "tool_name_invalid",
                    format!(
                        "{}: tool name must match [A-Za-z_][A-Za-z0-9_]* and be at most {MAX_TOOL_NAME_LEN} characters",
                        at()
                    ),
                ));
            }
            if !seen.insert(name) {
                return Err(error(
                    "tool_name_duplicate",
                    format!("{}: duplicate tool name '{name}'", at()),
                ));
            }
        }
    }
    Ok(())
}

fn has_content(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(text) => !text.is_empty(),
        MessageContent::Parts(parts) => !parts.is_empty(),
    }
}

/// `[A-Za-z_][A-Za-z0-9_]*`, at most [`MAX_TOOL_NAME_LEN`] bytes.
fn is_valid_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= MAX_TOOL_NAME_LEN
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn on_role(code: &'static str, role: &str, rule: &str) -> validator::ValidationError {
    error(
        code,
        format!("'tools' on a message with role '{role}' {rule}"),
    )
}

fn error(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}

/// Whether a model id names Kimi K3, the only Kimi model with pinned sampling and thinking rules.
fn is_k3(model: &str) -> bool {
    model.split('/').any(|segment| {
        super::starts_with_ignore_ascii_case(segment, "kimi-k3")
            || super::starts_with_ignore_ascii_case(segment, "kimi_k3")
    })
}

/// Immutable sampling parameters: the contract fixes them, so any other value
/// is a 400 rather than a silently different sampling regime.
fn validate_sampling(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if let Some(t) = req.temperature {
        if !TEMPERATURES.iter().any(|allowed| approx_eq(t, *allowed)) {
            return Err(pinned(
                "temperature_not_allowed",
                "temperature",
                "0, 0.6 or 1",
            ));
        }
    }
    if req.top_p.is_some_and(|p| !approx_eq(p, TOP_P)) {
        return Err(pinned("top_p_not_allowed", "top_p", "0.95"));
    }
    if req.presence_penalty.is_some_and(|p| !approx_eq(p, 0.0)) {
        return Err(pinned(
            "presence_penalty_not_allowed",
            "presence_penalty",
            "0",
        ));
    }
    if req.frequency_penalty.is_some_and(|p| !approx_eq(p, 0.0)) {
        return Err(pinned(
            "frequency_penalty_not_allowed",
            "frequency_penalty",
            "0",
        ));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(pinned("n_not_allowed", "n", "1"));
    }
    Ok(())
}

/// K3 takes `effort` in low/high/max and no `adaptive`; `keep` is left to the renderer.
fn validate_thinking(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    let Some(thinking) = &req.thinking else {
        return Ok(());
    };
    if thinking.r#type == Some(ThinkingType::Adaptive) {
        return Err(pinned(
            "thinking_type_not_supported",
            "thinking.type",
            "enabled or disabled",
        ));
    }
    if thinking
        .effort
        .as_deref()
        .is_some_and(|effort| !THINKING_EFFORTS.contains(&effort))
    {
        return Err(pinned(
            "thinking_effort_invalid",
            "thinking.effort",
            "low, high or max",
        ));
    }
    Ok(())
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn pinned(code: &'static str, field: &str, allowed: &str) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(format!("invalid {field}: only {allowed} is allowed for this model").into());
    e
}
