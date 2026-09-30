//! The `adapter` glue over the recorded sessions of every template it accepts.
//! A complete output reads the parsed message. A stream, fed by the
//! reasoning parser or else by the tool parser, reads the region events as
//! `transformers serve` does: transformers' own events for one feed, and this
//! crate's events (which the replay test matches to transformers') for every
//! two-way split.

#![expect(clippy::unwrap_used, reason = "a test: a bad fixture should panic")]

mod common;

use common::{canonical, fixtures_dir, read_json, read_jsonl, tag, untag};
use serde_json::{json, Value};
use smg_response_template::{
    adapter::{check, ResponseParserState, ToolCall},
    load_response_template, parse_response, ResponseParser, ResponseTemplate,
};

/// What smg's parsers return: reasoning, content, and each call as canonical
/// JSON of `[name, arguments]`.
#[derive(Debug, Default, PartialEq)]
struct Read {
    reasoning: String,
    content: String,
    calls: Vec<String>,
}

fn call(name: &Value, arguments: &Value) -> String {
    canonical(&tag(&json!([name, arguments])))
}

/// `transformers serve`'s `_normalize_tool_call` over one value or a list;
/// `None` where it fails.
fn serve_calls(value: &Value, calls: &mut Vec<String>) -> Option<()> {
    if let Value::Array(items) = value {
        return items.iter().try_for_each(|item| serve_calls(item, calls));
    }
    let function = value.get("function")?;
    let name = function.get("name").filter(|n| n.is_string())?;
    calls.push(call(name, function.get("arguments")?));
    Some(())
}

/// `transformers serve`'s `response_events_to_chunks`, with `reasoning_content`
/// read as `thinking`, and reasoning into the content when `merge`.
fn serve_events(events: &Value, merge: bool, read: &mut Read) -> Option<()> {
    for event in events.as_array().unwrap() {
        let field = event["field"].as_str().unwrap();
        let reasoning = matches!(field, "thinking" | "reasoning_content");
        match event["type"].as_str().unwrap() {
            "region_chunk" if reasoning || field == "content" => {
                let text = event["text"].as_str().unwrap();
                if reasoning && !merge {
                    read.reasoning.push_str(text);
                } else {
                    read.content.push_str(text);
                }
            }
            "region_close" if field == "tool_calls" => {
                serve_calls(&untag(&event["value"]), &mut read.calls)?;
            }
            _ => {}
        }
    }
    Some(())
}

/// `transformers serve`'s `parse_assistant_message` of a message.
fn serve_message(message: &Value) -> Option<Read> {
    let text = |key: &str| message.get(key).and_then(Value::as_str).unwrap_or_default();
    let mut read = Read {
        reasoning: [text("thinking"), text("reasoning_content")].concat(),
        content: text("content").to_owned(),
        calls: Vec::new(),
    };
    // `parsed.get("tool_calls") or []`
    let falsy = [
        json!(null),
        json!(false),
        json!(0),
        json!(""),
        json!([]),
        json!({}),
        json!(0.0),
    ];
    match message.get("tool_calls") {
        Some(calls) if !falsy.contains(calls) => serve_calls(calls, &mut read.calls)?,
        _ => {}
    }
    Some(read)
}

/// The events of this crate's parser for `chunks`, read as serve reads them.
fn crate_stream(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    merge: bool,
) -> Option<Read> {
    let events = |events| tag(&serde_json::to_value(events).unwrap());
    let mut parser = ResponseParser::new(template, prefix, tools).ok()?;
    let mut read = Read::default();
    for chunk in chunks {
        serve_events(&events(parser.feed(chunk).ok()?), merge, &mut read)?;
    }
    serve_events(&events(parser.finalize().ok()?.1), merge, &mut read)?;
    Some(read)
}

fn new_state(template: &ResponseTemplate, prefix: &str, tools: &[Value]) -> ResponseParserState {
    let tail = template.truncate_past_last_anchor(prefix);
    ResponseParserState::new(template, tail, tools, false)
}

fn push_calls(read: &mut Read, calls: Vec<ToolCall>) {
    read.calls
        .extend(calls.iter().map(|c| call(&json!(c.name), &c.arguments)));
}

/// A stream through a parser state, as the gateway calls the parsers: the
/// reasoning parser feeds each chunk (or, without it, the tool parser) and the
/// tool parser takes the content.
fn state_stream(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    with_reasoning: bool,
) -> Option<Read> {
    let state = new_state(template, prefix, tools);
    let mut read = Read::default();
    for chunk in chunks.iter().copied().map(Some).chain([None]) {
        let content = if with_reasoning {
            let (reasoning, content) = state.reasoning(chunk).ok()?;
            read.reasoning.push_str(&reasoning);
            Some(content)
        } else {
            chunk.map(str::to_owned)
        };
        let (content, calls) = state.tools(content.as_deref()).ok()?;
        read.content.push_str(&content);
        push_calls(&mut read, calls);
    }
    Some(read)
}

fn state_complete(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    text: &str,
) -> Option<Read> {
    let state = new_state(template, prefix, tools);
    let (reasoning, content) = state.reasoning_complete(text).ok()?;
    let (content, calls) = state.tools(Some(&content)).ok()?;
    let mut read = Read {
        reasoning,
        content,
        calls: Vec::new(),
    };
    push_calls(&mut read, calls);
    Some(read)
}

fn replay(name: &str) -> (usize, usize) {
    let dir = fixtures_dir();
    let templates: std::collections::HashMap<String, ResponseTemplate> =
        read_jsonl(&dir, "templates")
            .iter()
            .filter_map(|row| {
                let template = load_response_template(&untag(&row["template"])).ok()?;
                check(&template).ok()?;
                Some((row["id"].as_str().unwrap().to_owned(), template))
            })
            .collect();
    let tool_sets = read_json(&dir, "tools.json");
    let (mut sessions, mut splits) = (0, 0);
    let mut failures = Vec::new();
    for case in read_jsonl(&dir, name) {
        let Some(template) = templates.get(case["template"].as_str().unwrap()) else {
            continue;
        };
        let tools = match case.get("tools") {
            Some(Value::String(set)) => untag(&tool_sets[set]).as_array().unwrap().clone(),
            _ => Vec::new(),
        };
        let (text, prefix) = (
            case["text"].as_str().unwrap(),
            case["prefix"].as_str().unwrap(),
        );
        let id = case["id"].as_str().unwrap();
        sessions += 1;

        // transformers raises from `parse_response` where this crate does (the
        // replay test matches every step); its message otherwise.
        let expected = parse_response(text, template, prefix, &tools)
            .ok()
            .and_then(|message| serve_message(&Value::Object(message)));
        let got = state_complete(template, prefix, &tools, text);
        if got != expected {
            failures.push(format!("{id} complete: {got:?}, expected {expected:?}"));
        }

        // One feed, against transformers' own events.
        if let Some(trace) = case["unary"]["trace"].as_array() {
            for merge in [false, true] {
                let mut expected = Some(Read::default());
                for step in trace.iter().skip(1) {
                    let events = match step[0].as_str().unwrap() {
                        "feed" => &step[1],
                        "final" => &step[2],
                        _ => {
                            expected = None;
                            break;
                        }
                    };
                    expected = expected
                        .and_then(|mut read| serve_events(events, merge, &mut read).map(|()| read));
                }
                if trace[0][0] != "init" {
                    expected = None;
                }
                let got = state_stream(template, prefix, &tools, &[text], !merge);
                if got != expected {
                    failures.push(format!(
                        "{id} stream (merge {merge}): {got:?}, expected {expected:?}"
                    ));
                }
            }
        }

        // Every two-way split, against this crate's events.
        if case.get("split2").is_some() {
            for (b, _) in text.char_indices().skip(1) {
                splits += 1;
                let chunks = [&text[..b], &text[b..]];
                for merge in [false, true] {
                    let expected = crate_stream(template, prefix, &tools, &chunks, merge);
                    let got = state_stream(template, prefix, &tools, &chunks, !merge);
                    if got != expected {
                        failures.push(format!(
                            "{id} split at {b} (merge {merge}): {got:?}, expected {expected:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
    (sessions, splits)
}

#[test]
fn sessions_read_like_transformers_serve() {
    let (sessions, splits) = replay("sessions");
    assert!(
        sessions > 100 && splits > 1000,
        "{sessions} sessions, {splits} splits"
    );
}

#[test]
fn random_sessions_read_like_transformers_serve() {
    let (sessions, _) = replay("random");
    assert!(sessions > 100, "{sessions} sessions");
}

#[test]
fn check_names_what_smg_cannot_use() {
    let unsuitable = |template: Value| check(&load_response_template(&template).unwrap()).is_err();
    let field = json!({"open": "<a>", "close": "</a>"});
    assert!(unsuitable(
        json!({"start_anchor": "S", "fields": {"content": {}}})
    ));
    assert!(unsuitable(json!({"start_anchor": "S", "fields": {
        "thinking": field, "reasoning_content": {"open": "<b>", "close": "</b>"}}})));
    assert!(unsuitable(json!({"start_anchor": "S", "fields": {
        "thinking": {"open": "<a>", "close": "</a>", "repeats": true}}})));
    assert!(unsuitable(json!({"start_anchor": "S", "fields": {
        "thinking": field, "content": {"content": "json"}}})));
    assert!(unsuitable(
        json!({"start_anchor": "S", "defaults": {"content": 0},
        "fields": {"thinking": field}})
    ));
    assert!(!unsuitable(json!({"start_anchor": "S", "fields": {
        "reasoning_content": {"open": "<a>", "close": "</a>", "repeats": true, "join": "\n"},
        "content": {}}})));
}
