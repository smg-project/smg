//! Native typed Inkling2 prompts. Text segments are never structural tokens.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::RendererProfile;
use crate::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams, ThinkingToggle},
    traits::RendererCapabilities,
};

/// The hydration layer expands only this soft anchor, not the image header.
pub const IMAGE_ANCHOR: &str = "<|unused_200054|>";
pub const IMAGE_ANCHOR_ID: u32 = 200054;
pub const NATIVE_EFFORT_VALUES: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// What the renderer answers around a render. It keeps message parts in the
/// OpenAI wire format and has no thinking switch: effort is a prompt line, not
/// a toggle, and no prompt ends inside a reasoning block. It continues a final
/// assistant message itself and parses tool-call argument strings itself. A
/// JSON response needs its reasoning separated and cannot continue a
/// partial assistant value.
pub(crate) const PROFILE: RendererProfile = RendererProfile {
    content_format: Some(ChatTemplateContentFormat::OpenAI),
    thinking_toggle: ThinkingToggle::None,
    thinking_key_name: None,
    native_effort_values: NATIVE_EFFORT_VALUES,
    native_effort_off_values: &[],
    think_in_prefill: false,
    capabilities: RendererCapabilities {
        native_assistant_continuation: true,
        raw_tool_call_arguments: true,
        json_requires_separate_reasoning: true,
        json_requires_empty_assistant_prefix: true,
        ..RendererProfile::NO_CAPABILITIES
    },
    thinking_mode: None,
};

pub const SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("<|message_user|>", 200000),
    ("<|message_model|>", 200001),
    ("<|message_system|>", 200002),
    ("<|message_tool|>", 200003),
    ("<|content_text|>", 200004),
    ("<|content_image|>", 200005),
    ("<|content_model_end_sampling|>", 200006),
    ("<|content_thinking|>", 200008),
    ("<|end_message|>", 200010),
    ("<|content_tool_error|>", 200022),
    ("<|content_xml|>", 200024),
    ("<|content_invoke_tool_json|>", 200049),
    (IMAGE_ANCHOR, IMAGE_ANCHOR_ID),
];

#[derive(Debug, PartialEq)]
pub enum Segment {
    Special(&'static str, u32),
    Text(String),
}

impl Segment {
    pub fn text(&self) -> &str {
        match self {
            Self::Special(text, _) => text,
            Self::Text(text) => text,
        }
    }
}

fn special(out: &mut Vec<Segment>, id: u32) -> Result<()> {
    let &(text, _) = SPECIAL_TOKENS
        .iter()
        .find(|(_, token)| *token == id)
        .with_context(|| format!("Inkling2 structural token ID {id} is not declared"))?;
    out.push(Segment::Special(text, id));
    Ok(())
}

fn block(
    out: &mut Vec<Segment>,
    role: u32,
    name: &str,
    kind: u32,
    body: Segment,
    close: bool,
) -> Result<()> {
    special(out, role)?;
    if !name.is_empty() {
        out.push(Segment::Text(name.to_owned()));
    }
    special(out, kind)?;
    out.push(body);
    if close {
        special(out, 200010)?;
    }
    Ok(())
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("Inkling2 requires a string field {key:?}"))
}

/// Match the reference renderer's sorted, compact, Unicode JSON dialect.
fn native_json(value: &Value, legacy_floats: bool) -> Result<String> {
    Ok(match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            let fields = keys
                .into_iter()
                .map(|key| {
                    Ok(format!(
                        "{}:{}",
                        serde_json::to_string(key)?,
                        native_json(&map[key], legacy_floats)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| native_json(item, legacy_floats))
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Value::Number(number) if number.is_f64() => {
            let text = number.to_string();
            if let Some((mantissa, exponent)) = text.split_once('e') {
                let exponent: i32 = exponent.parse()?;
                if (-5..16).contains(&exponent) {
                    let negative = mantissa.starts_with('-');
                    let digits = mantissa.trim_start_matches('-').replace('.', "");
                    let decimal = exponent + 1;
                    let body = if decimal <= 0 {
                        format!("0.{}{}", "0".repeat((-decimal) as usize), digits)
                    } else if decimal as usize >= digits.len() {
                        format!(
                            "{}{}.0",
                            digits,
                            "0".repeat(decimal as usize - digits.len())
                        )
                    } else {
                        format!(
                            "{}.{}",
                            &digits[..decimal as usize],
                            &digits[decimal as usize..]
                        )
                    };
                    format!("{}{body}", if negative { "-" } else { "" })
                } else {
                    format!(
                        "{mantissa}e{}{exponent}",
                        if !legacy_floats && exponent >= 0 {
                            "+"
                        } else {
                            ""
                        }
                    )
                }
            } else {
                text
            }
        }
        _ => serde_json::to_string(value)?,
    })
}

fn effort(params: &ChatTemplateParams) -> Result<String> {
    let value = params
        .template_kwargs
        .and_then(|kwargs| kwargs.get("reasoning_effort"));
    let effort = match value {
        None | Some(Value::Null) => 0.9,
        Some(Value::String(name)) => match name.as_str() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            _ => name
                .parse::<f64>()
                .context("Inkling2 reasoning_effort is invalid")?,
        },
        Some(Value::Number(number)) => number
            .as_f64()
            .context("Inkling2 reasoning_effort is invalid")?,
        _ => bail!("Inkling2 reasoning_effort must be a name or a number in [0, 0.99]"),
    };
    if !effort.is_finite() || !(0.0..=0.99).contains(&effort) {
        bail!("Inkling2 reasoning_effort must be in [0, 0.99]");
    }
    let milli = (effort * 1000.0).round_ties_even();
    let rounded = ((milli / 1000.0) * 100.0).round_ties_even() / 100.0;
    Ok(format!("Thinking effort level: {rounded}"))
}

fn tool_call(call: &Value) -> Result<Segment> {
    if call.get("type").is_some_and(|kind| kind != "function") {
        bail!("Inkling2 supports only function tool calls");
    }
    let function = call
        .get("function")
        .context("Inkling2 tool call requires function")?;
    let name = string(function, "name")?;
    let args = match function.get("arguments") {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text)
            .context("Inkling2 tool arguments must be valid JSON")?,
        Some(value) => value.clone(),
        None => json!({}),
    };
    if !args.is_object() {
        bail!("Inkling2 tool arguments must be a JSON object");
    }
    Ok(Segment::Text(format!(
        "{{\"name\":{},\"args\":{}}}",
        native_json(&json!(name), true)?,
        native_json(&args, true)?
    )))
}

fn message(out: &mut Vec<Segment>, value: &Value, continuation: bool) -> Result<()> {
    let role = match string(value, "role")? {
        "user" => 200000,
        "assistant" => 200001,
        "system" | "developer" => 200002,
        "tool" => 200003,
        role => bail!("Inkling2 does not support message role {role:?}"),
    };
    let name = if role == 200003 {
        match value.get("name") {
            None | Some(Value::Null) => "",
            Some(Value::String(name)) => name,
            _ => bail!("Inkling2 tool name must be a string"),
        }
    } else {
        ""
    };
    let content = value.get("content").unwrap_or(&Value::Null);
    let calls = match value.get("tool_calls") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(calls)) => calls.as_slice(),
        _ => bail!("Inkling2 tool_calls must be an array"),
    };
    let reasoning = match value.get("reasoning_content") {
        None | Some(Value::Null) => "",
        Some(Value::String(text)) => text,
        _ => bail!("Inkling2 reasoning_content must be a string"),
    };
    if continuation
        && (role != 200001 || !content.is_string() || !calls.is_empty() || !reasoning.is_empty())
    {
        bail!("Inkling2 continuation requires a final assistant with only string text");
    }
    if role != 200001 && !calls.is_empty() {
        bail!("Inkling2 tool calls require an assistant message");
    }
    let start = out.len();
    if !reasoning.is_empty() {
        block(
            out,
            role,
            name,
            200008,
            Segment::Text(reasoning.to_owned()),
            true,
        )?;
    }
    match content {
        Value::Null => {}
        Value::String(text) => {
            if !text.is_empty() || (calls.is_empty() && reasoning.is_empty()) {
                block(
                    out,
                    role,
                    name,
                    200004,
                    Segment::Text(text.clone()),
                    !continuation,
                )?;
            }
        }
        Value::Array(parts) => {
            for part in parts {
                let (kind, body) = match string(part, "type")? {
                    "text" => (200004, Segment::Text(string(part, "text")?.to_owned())),
                    "thinking" => (200008, Segment::Text(string(part, "thinking")?.to_owned())),
                    "tool_error" => (200022, Segment::Text(string(part, "text")?.to_owned())),
                    "image_url" => {
                        let image = part
                            .get("image_url")
                            .context("Inkling2 image_url is missing")?;
                        let url = image
                            .as_str()
                            .or_else(|| image.get("url").and_then(Value::as_str));
                        if !url.is_some_and(|url| !url.is_empty()) {
                            bail!("Inkling2 image URL must be a nonempty string");
                        }
                        (200005, Segment::Special(IMAGE_ANCHOR, IMAGE_ANCHOR_ID))
                    }
                    // The common gateway has already extracted and validated the URL.
                    "image" => (200005, Segment::Special(IMAGE_ANCHOR, IMAGE_ANCHOR_ID)),
                    "tool_call" if role == 200001 => (
                        200049,
                        tool_call(
                            part.get("tool_call")
                                .context("Inkling2 tool_call is missing")?,
                        )?,
                    ),
                    kind => bail!("Inkling2 does not support content part {kind:?}"),
                };
                block(out, role, name, kind, body, true)?;
            }
        }
        _ => bail!("Inkling2 content must be string, array, or null"),
    }
    for call in calls {
        block(out, role, name, 200049, tool_call(call)?, true)?;
    }
    if role == 200001 && out.len() > start && !continuation {
        special(out, 200006)?;
    }
    Ok(())
}

/// Whether the render leaves the final assistant message open: the
/// `continue_final_message` template argument when the request sends one,
/// else the caller's [`ChatTemplateParams::continue_final_message`]. The
/// renderer writes no generation opener, so `add_generation_prompt` alone
/// does not continue a final assistant message.
pub fn continues_final_message(
    messages: &[Value],
    params: &ChatTemplateParams,
    assistant_prefix: Option<&str>,
) -> Result<bool> {
    let requested = params
        .template_kwargs
        .and_then(|kwargs| kwargs.get("continue_final_message"));
    let continuation = match requested {
        Some(Value::Bool(value)) => *value,
        Some(_) => bail!("Inkling2 continue_final_message must be a boolean"),
        None => params.continue_final_message,
    };
    if continuation
        && (assistant_prefix.is_some()
            || !messages
                .last()
                .is_some_and(|message| message["role"] == "assistant"))
    {
        bail!("Inkling2 continuation requires one final assistant message");
    }
    Ok(continuation)
}

/// Render the reference HTTP request format, without BOS or a generation opener.
/// Returned segments retain the distinction between text and structural IDs.
pub fn render(
    messages: &[Value],
    params: &ChatTemplateParams,
    assistant_prefix: Option<&str>,
) -> Result<Vec<Segment>> {
    if let Some(kwargs) = params.template_kwargs {
        for key in kwargs.keys() {
            // The gateway forwards tool choice and output constraints. They
            // do not change this renderer's prompt, as in the reference HTTP format.
            if !matches!(
                key.as_str(),
                "reasoning_effort" | "continue_final_message" | "tool_choice" | "response_format"
            ) {
                bail!("Inkling2 does not support template argument {key:?}");
            }
        }
    }
    if params
        .documents
        .is_some_and(|documents| !documents.is_empty())
    {
        bail!("Inkling2 document rendering is not supported");
    }
    let continuation = continues_final_message(messages, params, assistant_prefix)?;
    let mut out = Vec::new();
    if let Some(tools) = params.tools.filter(|tools| !tools.is_empty()) {
        let declarations = tools.iter().map(|tool| {
            if tool.get("type").is_some_and(|kind| kind != "function") {
                bail!("Inkling2 supports only function tools");
            }
            let function = tool.get("function")
                .context("Inkling2 tool declaration requires function")?;
            let name = string(function, "name")?;
            let description = match function.get("description").filter(|value| !value.is_null()) {
                Some(value) => value.as_str().context("Inkling2 tool description must be a string")?,
                None => "",
            };
            let parameters = function.get("parameters").filter(|value| !value.is_null())
                .cloned().unwrap_or_else(|| json!({}));
            if !parameters.is_object() {
                bail!("Inkling2 tool parameters must be an object");
            }
            Ok(json!({"type":"function", "name": name, "description": description, "parameters": parameters}))
        }).collect::<Result<Vec<_>>>()?;
        block(
            &mut out,
            200002,
            "tool_declare",
            200024,
            Segment::Text(native_json(&json!(declarations), false)?),
            true,
        )?;
    }
    let leading = messages
        .iter()
        .take_while(|message| matches!(message["role"].as_str(), Some("system" | "developer")))
        .count();
    for value in &messages[..leading] {
        message(&mut out, value, false)?;
    }
    block(
        &mut out,
        200002,
        "",
        200004,
        Segment::Text(effort(params)?),
        true,
    )?;
    for (index, value) in messages.iter().enumerate().skip(leading) {
        message(&mut out, value, continuation && index + 1 == messages.len())?;
    }
    if let Some(prefix) = assistant_prefix {
        block(
            &mut out,
            200001,
            "",
            200004,
            Segment::Text(prefix.to_owned()),
            false,
        )?;
    }
    Ok(out)
}
