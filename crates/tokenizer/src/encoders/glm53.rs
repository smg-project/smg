//! GLM-5.3 / GLM-5.3-Flash native chat rendering, adapted from vLLM PR #62424.
//! Tool-result matching is linear in the result block instead of repeated
//! Jinja macro scans. Tokenization and multimodal preprocessing stay shared.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
};

use anyhow::{bail, Result};
use serde_json::Value;

use crate::chat_template::{self, ChatTemplateParams};

/// The current official templates identify the supported renderer revision.
/// Other templates retain MiniJinja; checkpoint metadata alone cannot identify 5.3.
const TEXT_TEMPLATE: &str = include_str!("../../tests/fixtures/glm53/chat_template.jinja");
const FLASH_TEMPLATE: &str = include_str!("../../tests/fixtures/glm53/chat_template_flash.jinja");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Template {
    Text,
    Flash,
}

impl Template {
    pub(crate) fn detect(source: &str) -> Option<Self> {
        // The shared .jinja loader reads the file as written but expands
        // literal \n; tokenizer_config.json and set_chat_template retain the
        // original. One trailing newline of a source never reaches the prompt
        // (minijinja drops it, as Jinja2 does), so a copy with or without it
        // renders alike and takes the native path alike.
        let body = |text: &str| text.strip_suffix('\n').unwrap_or(text).to_owned();
        let matches = |known: &str| {
            let source = body(source);
            source == body(known) || source == body(&known.replace("\\n", "\n"))
        };
        if matches(TEXT_TEMPLATE) {
            Some(Self::Text)
        } else if matches(FLASH_TEMPLATE) {
            Some(Self::Flash)
        } else {
            None
        }
    }
}

const TOOLS_PREFIX: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_SUFFIX: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

pub(super) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => value.clone(),
        Value::Number(value) if !value.is_f64() => value.to_string(),
        // A float spells as Python's `str`, a list or a map as Python's repr,
        // as the template writes them.
        _ => chat_template::python_repr(value),
    }
}

#[derive(Default)]
struct MediaContext {
    enabled: bool,
}

fn visible_text(value: &Value, media: &mut MediaContext) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(part) = part.as_str() {
                    out.push_str(part);
                    continue;
                }
                match part["type"].as_str().unwrap_or_default() {
                    "text" => out.push_str(&part.get("text").map(text).unwrap_or_default()),
                    kind @ ("image" | "image_url" | "video" | "video_url" | "audio"
                    | "audio_url" | "input_audio") => {
                        let kind = kind.replace("_url", "").replace("input_", "");
                        if media.enabled {
                            let _ = write!(out, "<|begin_of_{kind}|>");
                            if kind != "audio" {
                                let _ = write!(out, "<|{kind}|>");
                            }
                            let _ = write!(out, "<|end_of_{kind}|>");
                        } else {
                            let _ = write!(out, "<reminder>You are unable to process this {kind} because you don't have multi-modal input ability. Try different methods.</reminder>");
                        }
                    }
                    _ => {}
                }
            }
            out
        }
        _ => text(value),
    }
}

fn tool_definition(tool: &Value) -> Result<String> {
    let Some(fields) = tool.as_object() else {
        bail!("Each tool must be a JSON object");
    };
    let mut out = String::from("{");
    for (key, value) in fields {
        if key == "strict" || key == "defer_loading" {
            continue;
        }
        if out.len() > 1 {
            out.push_str(", ");
        }
        let _ = write!(out, "\"{key}\": {}", json(value)?);
    }
    out.push('}');
    Ok(out)
}

fn unwrap_tool(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

fn tool_references(value: &Value) -> bool {
    value
        .as_array()
        .and_then(|parts| parts.first())
        .is_some_and(|part| part["type"] == "tool_reference")
}

fn reference_response(refs: &Value, tools: &[Value]) -> Result<String> {
    if tools.is_empty() {
        bail!("Rendering a tool_reference requires tools");
    }
    let mut out = String::from("<tool_response><tools>\n");
    for reference in refs.as_array().into_iter().flatten() {
        for tool in tools.iter().map(unwrap_tool) {
            if tool.get("name") == reference.get("name") {
                out.push_str(&tool_definition(tool)?);
                out.push('\n');
            }
        }
    }
    out.push_str("</tools></tool_response>");
    Ok(out)
}

fn response(value: &Value, tools: &[Value], media: &mut MediaContext) -> Result<String> {
    if tool_references(value) {
        reference_response(value, tools)
    } else {
        Ok(format!(
            "<tool_response>{}</tool_response>",
            visible_text(value, media)
        ))
    }
}

fn bundled_outputs(value: &Value) -> Result<bool> {
    if !truthy(value) || value.is_string() {
        return Ok(false);
    }
    let Some(parts) = value.as_array() else {
        bail!("A tool message content must be a string or a list");
    };
    Ok(parts
        .first()
        .is_some_and(|part| part.get("output").is_some()))
}

fn tool_response(message: &Value, tools: &[Value], media: &mut MediaContext) -> Result<String> {
    let value = &message["content"];
    if tool_references(value) {
        return reference_response(value, tools);
    }
    if bundled_outputs(value)? {
        let mut out = String::new();
        for part in value.as_array().into_iter().flatten() {
            out.push_str(&response(&part["output"], tools, media)?);
        }
        Ok(out)
    } else {
        response(value, tools, media)
    }
}

fn id(value: &Value) -> String {
    ["tool_call_id", "id"]
        .into_iter()
        .filter_map(|key| value.get(key))
        .find(|value| truthy(value))
        .map(text)
        .unwrap_or_default()
}

fn ordered_tool_block(
    block: &[Value],
    assistant: Option<&Value>,
    tools: &[Value],
    media: &mut MediaContext,
) -> Result<String> {
    let calls = assistant
        .filter(|message| message["role"] == "assistant")
        .and_then(|message| message["tool_calls"].as_array())
        .filter(|calls| !calls.is_empty());
    if let Some(calls) = calls {
        let mut can_sort = true;
        let mut responses = HashMap::new();
        for message in block {
            if bundled_outputs(&message["content"])? {
                for entry in message["content"].as_array().into_iter().flatten() {
                    let key = id(entry);
                    if key.is_empty() || responses.insert(key, &entry["output"]).is_some() {
                        can_sort = false;
                    }
                }
            } else {
                let key = id(message);
                if key.is_empty() || responses.insert(key, &message["content"]).is_some() {
                    can_sort = false;
                }
            }
        }
        let call_ids: Vec<_> = calls.iter().map(id).collect();
        let mut seen = HashSet::new();
        for key in &call_ids {
            if key.is_empty() || !seen.insert(key) {
                can_sort = false;
            }
        }
        if can_sort && responses.keys().all(|key| seen.contains(key)) {
            return call_ids
                .iter()
                .filter_map(|key| responses.get(key))
                .map(|value| response(value, tools, media))
                .collect();
        }
    }
    let mut out = String::new();
    for message in block {
        out.push_str(&tool_response(message, tools, media)?);
    }
    Ok(out)
}

fn tool_call(out: &mut String, call: &Value) -> Result<()> {
    let call = call
        .get("function")
        .filter(|value| truthy(value))
        .unwrap_or(call);
    let Some(name) = call["name"].as_str() else {
        bail!("Each tool_call needs a string name");
    };
    let arguments = &call["arguments"];
    let Some(arguments) = arguments.as_object() else {
        bail!("tool_call arguments must be a JSON object");
    };
    let _ = write!(out, "<tool_call>{name}");
    for (key, value) in arguments {
        let _ = write!(out, "<arg_key>{key}</arg_key><arg_value>");
        if let Some(value) = value.as_str() {
            out.push_str(value);
        } else {
            out.push_str(&json(value)?);
        }
        out.push_str("</arg_value>");
    }
    out.push_str("</tool_call>");
    Ok(())
}

/// Render through the same role normalization and continuation contract as
/// the HF path, while interpreting no Jinja on the request path.
pub(crate) fn render(
    messages: &[Value],
    params: ChatTemplateParams,
    template: Template,
) -> Result<(String, bool)> {
    let wrapped = chat_template::string_content_as_text_parts(messages);
    let messages = wrapped.as_deref().unwrap_or(messages);
    let root = chat_template::root_messages_as_system(messages);
    let messages = root.as_deref().unwrap_or(messages);
    let developer = chat_template::developer_messages_as_system(messages);
    let messages = developer.as_deref().unwrap_or(messages);
    let render = |messages: &[Value], params: ChatTemplateParams| {
        render_inner(
            messages,
            params.tools.unwrap_or_default(),
            &params,
            &mut MediaContext {
                enabled: template == Template::Flash,
            },
        )
    };
    if params.continue_final_message {
        chat_template::continue_final_message(messages, params, render)
    } else {
        render(messages, params).map(|text| (text, false))
    }
}

fn render_inner(
    messages: &[Value],
    tools: &[Value],
    params: &ChatTemplateParams,
    media: &mut MediaContext,
) -> Result<String> {
    let kwargs = params.template_kwargs;
    let effort = match kwargs
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(Value::as_str)
    {
        Some("low") => "Low",
        Some("high") => "High",
        _ => "Max",
    };
    let clear_thinking = kwargs
        .and_then(|k| k.get("clear_thinking"))
        .is_some_and(truthy);
    let mut out = format!("[gMASK]<sop><|system|>Reasoning Effort: {effort}");
    if !tools.is_empty() {
        out.push_str(TOOLS_PREFIX);
        for tool in tools.iter().map(unwrap_tool) {
            if !truthy(&tool["defer_loading"]) {
                out.push_str(&tool_definition(tool)?);
                out.push('\n');
            }
        }
        out.push_str(TOOLS_SUFFIX);
    }
    let last_user = messages
        .iter()
        .rposition(|message| message["role"] == "user");
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        match message["role"].as_str().unwrap_or_default() {
            "user" | "system" => {
                let _ = write!(
                    out,
                    "<|{}|>{}",
                    message["role"].as_str().unwrap_or_default(),
                    visible_text(&message["content"], media)
                );
            }
            "assistant" => {
                out.push_str("<|assistant|><think>");
                let mut content = visible_text(&message["content"], media);
                let mut reasoning = message["reasoning_content"].as_str().map(str::to_owned);
                if reasoning.is_none() && content.contains("</think>") {
                    reasoning = Some(
                        content
                            .split("</think>")
                            .next()
                            .unwrap_or_default()
                            .rsplit("<think>")
                            .next()
                            .unwrap_or_default()
                            .to_owned(),
                    );
                    content = content
                        .rsplit("</think>")
                        .next()
                        .unwrap_or_default()
                        .to_owned();
                }
                if !clear_thinking || last_user.is_none_or(|last| index > last) {
                    if let Some(reasoning) = reasoning {
                        out.push_str(&reasoning);
                    }
                }
                out.push_str("</think>");
                // Match MiniJinja pycompat's str.strip (Rust whitespace); the
                // shared continuation cut separately follows Python whitespace.
                out.push_str(content.trim());
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    tool_call(&mut out, call)?;
                }
            }
            "tool" => {
                let start = index;
                while index + 1 < messages.len() && messages[index + 1]["role"] == "tool" {
                    index += 1;
                }
                out.push_str("<|observation|>");
                out.push_str(&ordered_tool_block(
                    &messages[start..=index],
                    start.checked_sub(1).map(|i| &messages[i]),
                    tools,
                    media,
                )?);
            }
            _ => {}
        }
        index += 1;
    }
    if params.add_generation_prompt {
        out.push_str("<|assistant|><think>");
    }
    Ok(out)
}

fn json(value: &Value) -> Result<String> {
    chat_template::compact_python_json(value)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde::Deserialize;
    use serde_json::{json, Value};

    use super::*;
    use crate::chat_template::ChatTemplateState;

    #[derive(Deserialize)]
    struct Case {
        name: String,
        messages: Vec<Value>,
        tools: Vec<Value>,
        add_generation_prompt: bool,
        #[serde(default)]
        continue_final_message: bool,
        kwargs: HashMap<String, Value>,
    }

    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }

    /// The PR's corpus covers sorting failures, missing outputs, deferred tool
    /// references, embedded reasoning, Unicode and continuation. The existing SMG
    /// MiniJinja path is the oracle, including its role and continuation rules.
    #[test]
    fn native_matches_minijinja_corpus_for_supported_templates() {
        let cases: Cases = serde_json::from_str(include_str!(
            "../../tests/fixtures/glm53/render_fixtures.json"
        ))
        .unwrap();
        let loaded_text = TEXT_TEMPLATE.replace("\\n", "\n");
        let loaded_flash = FLASH_TEMPLATE.replace("\\n", "\n");
        let stripped_text = TEXT_TEMPLATE.strip_suffix('\n').unwrap();
        for (source, template) in [
            (TEXT_TEMPLATE, Template::Text),
            (FLASH_TEMPLATE, Template::Flash),
            (loaded_text.as_str(), Template::Text),
            (loaded_flash.as_str(), Template::Flash),
            (stripped_text, Template::Text),
        ] {
            assert_eq!(Template::detect(source), Some(template));
            let jinja = ChatTemplateState::new(Some(source.into())).unwrap();
            for case in &cases.cases {
                let mut messages = case.messages.clone();
                // SMG's request preparation parses OpenAI string arguments before
                // handing them to either renderer. Invalid arguments still reject.
                for message in &mut messages {
                    if let Some(calls) = message["tool_calls"].as_array_mut() {
                        for call in calls {
                            let function = call.get_mut("function").unwrap();
                            if let Some(raw) = function["arguments"].as_str() {
                                if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
                                    function["arguments"] = parsed;
                                }
                            }
                        }
                    }
                }
                let params = || ChatTemplateParams {
                    add_generation_prompt: case.add_generation_prompt,
                    continue_final_message: case.continue_final_message,
                    tools: Some(&case.tools),
                    template_kwargs: Some(&case.kwargs),
                    ..Default::default()
                };
                let expected = jinja.apply_with_continuation(&messages, params());
                let actual = render(&messages, params(), template);
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(actual, expected, "{} ({template:?})", case.name);
                    }
                    (Err(_), Err(_)) => {}
                    (actual, expected) => panic!(
                        "{} ({template:?}): native={actual:?}, jinja={expected:?}",
                        case.name
                    ),
                }
            }
        }
    }

    #[test]
    fn flash_media_and_reordered_results_match_minijinja() {
        let jinja = ChatTemplateState::new(Some(FLASH_TEMPLATE.into())).unwrap();
        for kind in [
            "image",
            "image_url",
            "video",
            "video_url",
            "audio",
            "audio_url",
            "input_audio",
        ] {
            let messages = vec![
                json!({"role": "user", "content": [{"type": "text", "text": "before"}, {"type": kind}, {"type": "text", "text": "after"}]}),
                json!({"role": "assistant", "tool_calls": [
                    {"id": "a", "function": {"name": "lookup", "arguments": {}}},
                    {"id": "b", "function": {"name": "lookup", "arguments": {}}}
                ]}),
                json!({"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "B"}, {"type": kind}]}),
                json!({"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "A"}, {"type": kind}]}),
            ];
            let params = || ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            };
            let expected = jinja.apply_with_continuation(&messages, params()).unwrap();
            assert_eq!(
                render(&messages, params(), Template::Flash).unwrap(),
                expected,
                "{kind}"
            );
            assert!(
                expected.0.find("<tool_response>A").unwrap()
                    < expected.0.find("<tool_response>B").unwrap()
            );
        }
    }

    #[test]
    fn scalar_and_structured_text_parts_match_minijinja() {
        for source in [TEXT_TEMPLATE, FLASH_TEMPLATE] {
            let template = Template::detect(source).unwrap();
            let jinja = ChatTemplateState::new(Some(source.into())).unwrap();
            for value in [
                json!(1.0),
                json!(1e20),
                json!(false),
                json!({"a": [true, null, "中文"]}),
                json!(null),
            ] {
                for role in ["user", "assistant"] {
                    for content in [value.clone(), json!([{"type": "text", "text": value}])] {
                        let messages = [json!({"role": role, "content": content})];
                        let params = || ChatTemplateParams {
                            add_generation_prompt: true,
                            ..Default::default()
                        };
                        assert_eq!(
                            render(&messages, params(), template).unwrap(),
                            jinja.apply_with_continuation(&messages, params()).unwrap(),
                            "{messages:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn continuation_markers_roles_and_conflicting_flags_match_minijinja() {
        let jinja = ChatTemplateState::new(Some(TEXT_TEMPLATE.into())).unwrap();
        for messages in [
            vec![
                json!({"role": "root", "content": "root"}),
                json!({"role": "system", "content": "system"}),
                json!({"role": "developer", "content": "developer"}),
                json!({"role": "assistant", "content": "partial"}),
            ],
            vec![
                json!({"role": "user", "content": "CONTINUE_FINAL_MESSAGE_TAG___"}),
                json!({"role": "assistant", "content": "CONTINUE_FINAL_MESSAGE_TAG answer"}),
            ],
            vec![
                json!({"role": "assistant", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": " second \n"}]}),
            ],
            vec![json!({"role": "developer", "content": "partial"})],
            vec![],
        ] {
            for add_generation_prompt in [false, true] {
                let params = || ChatTemplateParams {
                    add_generation_prompt,
                    continue_final_message: true,
                    ..Default::default()
                };
                match (
                    render(&messages, params(), Template::Text),
                    jinja.apply_with_continuation(&messages, params()),
                ) {
                    (Ok(actual), Ok(expected)) => assert_eq!(actual, expected, "{messages:?}"),
                    (Err(_), Err(_)) => {}
                    (actual, expected) => panic!("native={actual:?}, jinja={expected:?}"),
                }
            }
        }
    }
}
