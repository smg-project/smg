//! The `adapter` glue over the recorded sessions of every template it accepts.
//! A complete output reads the parsed message. A stream, fed by the
//! reasoning parser or else by the tool parser, reads the region events as
//! `transformers serve` does: transformers' own events for one feed, and this
//! crate's events (which the replay test matches to transformers') for every
//! two-way split. A stream whose tool parser reads calls by name first adds up
//! to the same.

#![expect(clippy::unwrap_used, reason = "a test: a bad fixture should panic")]

mod common;

use common::{canonical, fixtures_dir, read_json, read_jsonl, tag, untag};
use serde_json::{json, Value};
use smg_response_template::{
    adapter::{check, content_needs_opener, CallItem, ResponseParserState, ToolCall},
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

/// [`state_stream`] with the tool parser reading items, and how many calls
/// were named before their region closed. Nothing comes between a call's name
/// and its arguments.
fn items_stream(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    with_reasoning: bool,
) -> Option<(Read, usize)> {
    let state = new_state(template, prefix, tools);
    let mut read = Read::default();
    let mut calls: Vec<(String, Option<Value>)> = Vec::new();
    let (mut named, mut names) = (None, 0);
    for chunk in chunks.iter().copied().map(Some).chain([None]) {
        let (reasoning, content) = if with_reasoning {
            let (reasoning, content) = state.reasoning(chunk).ok()?;
            (reasoning, Some(content))
        } else {
            (String::new(), chunk.map(str::to_owned))
        };
        let (content, items) = state.tool_items(content.as_deref()).ok()?;
        if named.is_some() {
            assert!(
                reasoning.is_empty() && content.is_empty(),
                "{reasoning:?} {content:?}"
            );
            assert!(
                matches!(items.first(), None | Some(CallItem::Arguments(_))),
                "{items:?}"
            );
        }
        read.reasoning.push_str(&reasoning);
        read.content.push_str(&content);
        for item in items {
            match item {
                CallItem::Call(call) => calls.push((call.name, Some(call.arguments))),
                CallItem::Name(name) => {
                    assert!(named.replace(calls.len()).is_none());
                    names += 1;
                    calls.push((name, None));
                }
                CallItem::Arguments(arguments) => calls[named.take().unwrap()].1 = Some(arguments),
            }
        }
    }
    if state.take_error().is_some() {
        return None;
    }
    for (name, arguments) in calls {
        read.calls.push(call(&json!(name), &arguments.unwrap()));
    }
    Some((read, names))
}

/// Whether a template's opener names every call: the rule the adapter
/// applies, restated over the template's JSON.
fn names_calls(template: &Value) -> bool {
    let field = &template["fields"]["tool_calls"];
    let function = &field["transform"]["function"];
    let group = function["name"]
        .as_str()
        .and_then(|name| name.strip_prefix('{')?.strip_suffix('}'));
    let pattern = field["open_pattern"].as_str().unwrap_or_default();
    field["transform_each"] != true
        && field["join"].is_null()
        && function.get("arguments").is_some()
        && group.is_some_and(|group| {
            group != "content" && !group.contains('.') && pattern.contains(&format!("(?P<{group}>"))
        })
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

fn replay(name: &str) -> (usize, usize, usize) {
    let dir = fixtures_dir();
    let templates: std::collections::HashMap<String, (ResponseTemplate, bool)> =
        read_jsonl(&dir, "templates")
            .iter()
            .filter_map(|row| {
                let json = untag(&row["template"]);
                let template = load_response_template(&json).ok()?;
                check(&template).ok()?;
                let id = row["id"].as_str().unwrap().to_owned();
                Some((id, (template, names_calls(&json))))
            })
            .collect();
    let tool_sets = read_json(&dir, "tools.json");
    let (mut sessions, mut splits, mut names) = (0, 0, 0);
    let mut failures = Vec::new();
    for case in read_jsonl(&dir, name) {
        let Some((template, names_calls)) = templates.get(case["template"].as_str().unwrap())
        else {
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
        // The read of `chunks` with the tool parser taking items, which must
        // add up to `expected`; only an opener that names calls names them.
        let by_items = |chunks: &[&str], merge: bool, expected: &Option<Read>| {
            let got = items_stream(template, prefix, &tools, chunks, !merge);
            let (got, named) = got.map_or((None, 0), |(read, named)| (Some(read), named));
            let failure = (got != *expected || (named > 0 && !names_calls)).then(|| {
                format!("{id} items {chunks:?} (merge {merge}): {got:?}, {named} named, expected {expected:?}")
            });
            (named, failure)
        };

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
                let (named, failure) = by_items(&[text], merge, &expected);
                names += named;
                failures.extend(failure);
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
                    let (named, failure) = by_items(&chunks, merge, &expected);
                    names += named;
                    failures.extend(failure);
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
    (sessions, splits, names)
}

#[test]
fn sessions_read_like_transformers_serve() {
    let (sessions, splits, names) = replay("sessions");
    assert!(
        sessions > 100 && splits > 1000 && names > 1000,
        "{sessions} sessions, {splits} splits, {names} calls named"
    );
}

#[test]
fn random_sessions_read_like_transformers_serve() {
    let (sessions, _, _) = replay("random");
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

#[test]
fn content_needs_an_opener_unless_it_is_the_field_without_one() {
    let needs = |fields: Value| {
        let template = json!({"start_anchor": "S", "fields": fields});
        content_needs_opener(&load_response_template(&template).unwrap())
    };
    let thinking = json!({"open": "<t>", "close": "</t>"});
    assert!(!needs(json!({"thinking": thinking, "content": {}})));
    assert!(needs(
        json!({"thinking": thinking, "content": {"open": "<c>", "close": "</c>"}})
    ));
    // Output outside every region is thinking, or nothing.
    assert!(needs(
        json!({"thinking": {"close": "</t>"}, "content": {"open": "<c>"}})
    ));
    assert!(needs(json!({"thinking": thinking})));
}

/// Calls named by the opener, holding JSON arguments.
fn named_calls() -> Value {
    json!({
        "start_anchor": "<|assistant|>",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>"},
            "tool_calls": {
                "open_pattern": "<call name=\"(?P<name>\\w*)\">",
                "close": "</call>",
                "repeats": true,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            },
            "content": {"close": "<|end|>"}
        }
    })
}

fn named_state(template: &Value, prompt_tail: &str) -> ResponseParserState {
    let template = load_response_template(template).unwrap();
    ResponseParserState::new(&template, prompt_tail, &[], false)
}

fn name(name: &str) -> CallItem {
    CallItem::Name(name.to_owned())
}

fn arguments(arguments: Value) -> CallItem {
    CallItem::Arguments(arguments)
}

#[test]
fn a_stream_names_a_call_once_its_region_opens() {
    let state = named_state(&named_calls(), "");
    let items = |text| state.tool_items(text).unwrap();
    // A regex opener at the edge of the text could still grow: the region
    // opens, and the name comes, with the next text.
    assert_eq!(items(Some("Hi <call name=\"f\">")), ("Hi ".into(), vec![]));
    assert_eq!(items(Some("{\"a\": 1")), (String::new(), vec![name("f")]));
    // The arguments come alone; the text after the close waits.
    assert_eq!(
        items(Some("}</call>Bye")),
        (String::new(), vec![arguments(json!({"a": 1}))])
    );
    assert_eq!(items(None), ("Bye".into(), vec![]));

    // A region that opens and closes in one text gives the whole call.
    let state = named_state(&named_calls(), "");
    let (text, items) = state
        .tool_items(Some("<call name=\"f\">{}</call>"))
        .unwrap();
    let call = ToolCall {
        name: "f".into(),
        arguments: json!({}),
    };
    assert_eq!((text, items), (String::new(), vec![CallItem::Call(call)]));
}

#[test]
fn reasoning_after_a_named_call_waits_for_its_arguments() {
    let state = named_state(&named_calls(), "");
    assert_eq!(
        state.reasoning(Some("<call name=\"f\">{")).unwrap(),
        (String::new(), String::new())
    );
    assert_eq!(
        state.tool_items(Some("")).unwrap(),
        (String::new(), vec![name("f")])
    );
    // The region closes and reasoning follows in one chunk: the reasoning
    // waits, and the tool parser runs for the arguments.
    assert_eq!(
        state.reasoning(Some("}</call><think>more")).unwrap(),
        (String::new(), String::new())
    );
    assert!(!state.in_reasoning());
    assert_eq!(
        state.tool_items(Some("")).unwrap(),
        (String::new(), vec![arguments(json!({}))])
    );
    assert_eq!(
        state.reasoning(None).unwrap(),
        ("more".into(), String::new())
    );
}

#[test]
fn an_error_after_deferred_output_comes_with_the_next_call() {
    let named = |state: &ResponseParserState| {
        state.tool_items(Some("<call name=\"f\">{")).unwrap();
        let (text, items) = state
            .tool_items(Some("}</call>Hi <call name=\"g\">{bad"))
            .unwrap();
        assert_eq!((text, items), (String::new(), vec![arguments(json!({}))]));
    };
    // The deferred text and the failing chunk pass through; the next call
    // returns the error.
    let state = named_state(&named_calls(), "");
    named(&state);
    assert_eq!(
        state.tool_items(Some("</call>")).unwrap(),
        ("Hi </call>".into(), vec![])
    );
    assert!(state.tool_items(Some("x")).is_err());
    // At the end of the output there is no next call: `take_error` has it.
    let state = named_state(&named_calls(), "");
    named(&state);
    assert_eq!(state.tool_items(None).unwrap(), ("Hi ".into(), vec![]));
    assert!(state.take_error().is_some());
}

#[test]
fn a_named_call_whose_region_fails_keeps_only_its_name() {
    let state = named_state(&named_calls(), "");
    assert_eq!(
        state.tool_items(Some("<call name=\"f\">{\"a\"")).unwrap(),
        (String::new(), vec![name("f")])
    );
    assert!(state.tool_items(None).is_err());
}

#[test]
fn a_call_is_named_only_by_a_capture_every_call_takes() {
    // A region left open in the prompt is named on the first call.
    let state = named_state(&named_calls(), "<call name=\"f\">{");
    assert_eq!(
        state.tool_items(Some("\"a\"")).unwrap(),
        (String::new(), vec![name("f")])
    );
    // An empty capture names no call.
    let state = named_state(&named_calls(), "");
    let (_, items) = state.tool_items(Some("<call name=\"\">{")).unwrap();
    assert_eq!(items, vec![]);

    let unnamed = |edit: fn(&mut Value)| {
        let mut template = named_calls();
        edit(&mut template["fields"]["tool_calls"]);
        let state = named_state(&template, "");
        let (_, items) = state.tool_items(Some("<call name=\"f\">{")).unwrap();
        assert_eq!(items, vec![], "{template}");
    };
    // The name is in the region's JSON.
    unnamed(|field| field["transform"]["function"] = json!("{content}"));
    // No arguments to read, or a value per item.
    unnamed(|field| field["transform"]["function"] = json!({"name": "{name}"}));
    unnamed(|field| field["transform_each"] = json!(true));
    // A literal opener captures nothing.
    unnamed(|field| {
        field["open"] = json!("<call name=\"f\">");
        field.as_object_mut().unwrap().remove("open_pattern");
    });
}
