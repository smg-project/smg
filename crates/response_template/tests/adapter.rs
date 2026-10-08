//! The `adapter` glue over the recorded sessions of every template it accepts.
//! A complete output reads the parsed message. A stream, fed by the
//! reasoning parser or else by the tool parser, reads the region events as
//! `transformers serve` does: transformers' own events for one feed, and this
//! crate's events (which the replay test matches to transformers') for every
//! two-way split. Where transformers raises, a region failed: the read keeps
//! what the steps before it read, and the rest of the output is content. A
//! stream whose tool parser reads calls by name first adds up to the same.

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
#[derive(Debug, Default, Clone, PartialEq)]
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
            // An empty value holds no calls, as in serve's complete read
            // (`parsed.get("tool_calls") or []`); its stream raises there.
            "region_close" if field == "tool_calls" && !falsy(&event["value"]) => {
                let mut calls = Vec::new();
                serve_calls(&untag(&event["value"]), &mut calls)?;
                read.calls.extend(calls);
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
    match message.get("tool_calls") {
        Some(calls) if !falsy(calls) => serve_calls(calls, &mut read.calls)?,
        _ => {}
    }
    Some(read)
}

/// Python's falsy values, as JSON (a float as the fixtures tag it).
fn falsy(value: &Value) -> bool {
    [
        json!(null),
        json!(false),
        json!(0),
        json!(""),
        json!([]),
        json!({}),
        json!(0.0),
        tag(&json!(0.0)),
        tag(&json!(-0.0)),
    ]
    .contains(value)
}

/// The events of this crate's parser for `chunks`, read as serve reads them,
/// up to the step that raises or whose events serve cannot read; and whether
/// one does.
fn crate_steps(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    merge: bool,
) -> (Read, bool) {
    let events = |events| tag(&serde_json::to_value(events).unwrap());
    let mut read = Read::default();
    let Ok(mut parser) = ResponseParser::new(template, prefix, tools) else {
        return (read, true);
    };
    for chunk in chunks {
        let Ok(fed) = parser.feed(chunk) else {
            return (read, true);
        };
        if serve_events(&events(fed), merge, &mut read).is_none() {
            return (read, true);
        }
    }
    let read_all = match parser.finalize() {
        Ok((_, fed)) => serve_events(&events(fed), merge, &mut read).is_some(),
        Err(_) => false,
    };
    (read, !read_all)
}

/// [`crate_steps`] where no step raises.
fn crate_stream(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    merge: bool,
) -> Option<Read> {
    let (read, failed) = crate_steps(template, prefix, tools, chunks, merge);
    (!failed).then_some(read)
}

/// Where a step raises, a region failed: `got` reads what the steps before
/// it read (but for prompt text, see [`after_prompt`]), then more, and its
/// content ends with output (the failed region from its start and what
/// follows, unparsed).
fn extends(got: &Read, before: &Read, tail: &str, text: &str) -> bool {
    let output = format!("{tail}{text}");
    let rest = |got: &'_ str, before: &str| -> Option<String> {
        after_prompt(before, tail)
            .iter()
            .find_map(|before| got.strip_prefix(before).map(str::to_owned))
    };
    got.calls.starts_with(&before.calls)
        && rest(&got.reasoning, &before.reasoning).is_some()
        && rest(&got.content, &before.content).is_some_and(|rest| {
            let last = rest.chars().last();
            last.is_none_or(|last| output.ends_with(last))
        })
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
) -> Read {
    let state = new_state(template, prefix, tools);
    let mut read = Read::default();
    for chunk in chunks.iter().copied().map(Some).chain([None]) {
        let content = if with_reasoning {
            let (reasoning, content) = state.reasoning(chunk);
            read.reasoning.push_str(&reasoning);
            Some(content)
        } else {
            chunk.map(str::to_owned)
        };
        let (content, calls) = state.tools(content.as_deref());
        read.content.push_str(&content);
        push_calls(&mut read, calls);
    }
    read
}

/// [`state_stream`] with the tool parser reading items, and how many calls
/// were named before their region closed. Nothing comes between a call's name
/// and its arguments, except the content of its region when that fails; the
/// call then keeps only its name, which [`state_stream`] does not read.
fn items_stream(
    template: &ResponseTemplate,
    prefix: &str,
    tools: &[Value],
    chunks: &[&str],
    with_reasoning: bool,
) -> (Read, usize) {
    let state = new_state(template, prefix, tools);
    let mut read = Read::default();
    let mut calls: Vec<(String, Option<Value>)> = Vec::new();
    let (mut named, mut names) = (None, 0);
    for chunk in chunks.iter().copied().map(Some).chain([None]) {
        let (reasoning, content) = if with_reasoning {
            let (reasoning, content) = state.reasoning(chunk);
            (reasoning, Some(content))
        } else {
            (String::new(), chunk.map(str::to_owned))
        };
        let (content, items) = state.tool_items(content.as_deref());
        let arguments_next = matches!(items.first(), None | Some(CallItem::Arguments(_)));
        if named.is_some() && !(reasoning.is_empty() && content.is_empty() && arguments_next) {
            // The named call's region failed and its text is content.
            assert!(
                items
                    .iter()
                    .all(|item| !matches!(item, CallItem::Arguments(_))),
                "{items:?}"
            );
            named = None;
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
    let failed = state.take_error().is_some();
    for (name, arguments) in calls {
        match arguments {
            Some(arguments) => read.calls.push(call(&json!(name), &arguments)),
            None => assert!(failed, "{name} was named, and no region failed"),
        }
    }
    (read, names)
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

/// The reads `expected` can leave after prompt text it starts with: a stream
/// through a parser state shows no prompt text, where `transformers serve`
/// streams the text the prompt held back once the output shows that it began
/// no delimiter. Such text comes from `tail` in order, in pieces when the held
/// text spans delimiters, so it is a subsequence of `tail`.
fn after_prompt<'a>(expected: &'a str, tail: &str) -> Vec<&'a str> {
    let mut tail = tail.chars();
    let mut reads = vec![expected];
    for (i, c) in expected.char_indices() {
        if !tail.any(|t| t == c) {
            break;
        }
        reads.push(&expected[i + c.len_utf8()..]);
    }
    reads
}

/// `got` reads what `expected` does, but for prompt text (see [`after_prompt`]).
fn same_but_prompt(got: &Read, expected: &Read, tail: &str) -> bool {
    got.calls == expected.calls
        && after_prompt(&expected.reasoning, tail).contains(&got.reasoning.as_str())
        && after_prompt(&expected.content, tail).contains(&got.content.as_str())
}

fn state_complete(template: &ResponseTemplate, prefix: &str, tools: &[Value], text: &str) -> Read {
    let state = new_state(template, prefix, tools);
    let (reasoning, content) = state.reasoning_complete(text);
    let (content, calls) = state.tools(Some(&content));
    let mut read = Read {
        reasoning,
        content,
        calls: Vec::new(),
    };
    push_calls(&mut read, calls);
    read
}

/// Sessions read, two-way splits read, and sessions where a region failed.
/// What a replay read.
struct Replayed {
    sessions: usize,
    splits: usize,
    /// Sessions where a region failed.
    failing: usize,
    /// Calls named before their region closed.
    names: usize,
}

fn replay(name: &str) -> Replayed {
    let dir = fixtures_dir();
    // Each template, and the template with every field optional: a missing
    // required field is no error for smg's parsers, so the second gives the
    // events transformers produces up to that error.
    type Loaded = (ResponseTemplate, ResponseTemplate, bool);
    let templates: std::collections::HashMap<String, Loaded> = read_jsonl(&dir, "templates")
        .iter()
        .filter_map(|row| {
            let mut json = untag(&row["template"]);
            let template = load_response_template(&json).ok()?;
            check(&template).ok()?;
            let names = names_calls(&json);
            for field in json["fields"].as_object_mut()?.values_mut() {
                field["optional"] = json!(true);
            }
            let optional = load_response_template(&json).ok()?;
            let id = row["id"].as_str().unwrap().to_owned();
            Some((id, (template, optional, names)))
        })
        .collect();
    let tool_sets = read_json(&dir, "tools.json");
    let (mut sessions, mut splits, mut failing, mut names) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for case in read_jsonl(&dir, name) {
        let Some((template, optional, names_calls)) =
            templates.get(case["template"].as_str().unwrap())
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
        let tail = template.truncate_past_last_anchor(prefix);
        failing += usize::from(crate_stream(optional, prefix, &tools, &[text], false).is_none());
        // The read of `chunks` with the tool parser taking items, which must
        // add up to `expected`; only an opener that names calls names them.
        let by_items = |chunks: &[&str], merge: bool, expected: &Read| {
            let (got, named) = items_stream(template, prefix, &tools, chunks, !merge);
            let failure = (got != *expected || (named > 0 && !names_calls)).then(|| {
                format!("{id} items {chunks:?} (merge {merge}): {got:?}, {named} named, expected {expected:?}")
            });
            (named, failure)
        };

        // transformers raises from `parse_response` where this crate does (the
        // replay test matches every step); its message otherwise.
        let expected = parse_response(text, optional, prefix, &tools)
            .ok()
            .and_then(|message| serve_message(&Value::Object(message)));
        let got = state_complete(template, prefix, &tools, text);
        if expected.as_ref().is_some_and(|expected| got != *expected) {
            failures.push(format!("{id} complete: {got:?}, expected {expected:?}"));
        }

        // One feed, against transformers' own events; where a step raises,
        // against what the steps before it read (see `extends`).
        if let Some(trace) = case["unary"]["trace"].as_array() {
            for merge in [false, true] {
                let mut expected = Read::default();
                let mut failed = trace[0][0] != "init";
                for step in trace.iter().skip(1) {
                    let events = match step[0].as_str().unwrap() {
                        "feed" => &step[1],
                        "final" => &step[2],
                        _ => {
                            failed = true;
                            break;
                        }
                    };
                    if serve_events(events, merge, &mut expected).is_none() {
                        failed = true;
                        break;
                    }
                }
                // Only a missing required field raised.
                if let Some(read) =
                    crate_stream(optional, prefix, &tools, &[text], merge).filter(|_| failed)
                {
                    (expected, failed) = (read, false);
                }
                let got = state_stream(template, prefix, &tools, &[text], !merge);
                let matches = if failed {
                    extends(&got, &expected, tail, text)
                } else {
                    same_but_prompt(&got, &expected, tail)
                };
                if !matches {
                    failures.push(format!(
                        "{id} stream (merge {merge}): {got:?}, expected {expected:?} (failed {failed})"
                    ));
                }
                let (named, failure) = by_items(&[text], merge, &got);
                names += named;
                failures.extend(failure);
            }
        }

        // Every two-way split, against this crate's events; where a step
        // raises, against what the steps before it read.
        if case.get("split2").is_some() {
            for (b, _) in text.char_indices().skip(1) {
                splits += 1;
                let chunks = [&text[..b], &text[b..]];
                for merge in [false, true] {
                    let (expected, failed) = crate_steps(optional, prefix, &tools, &chunks, merge);
                    let got = state_stream(template, prefix, &tools, &chunks, !merge);
                    let matches = if failed {
                        extends(&got, &expected, tail, text)
                    } else {
                        same_but_prompt(&got, &expected, tail)
                    };
                    if !matches {
                        failures.push(format!(
                            "{id} split at {b} (merge {merge}): {got:?}, expected {expected:?} (failed {failed})"
                        ));
                    }
                    let (named, failure) = by_items(&chunks, merge, &got);
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
    Replayed {
        sessions,
        splits,
        failing,
        names,
    }
}

#[test]
fn sessions_read_like_transformers_serve() {
    let Replayed {
        sessions,
        splits,
        failing,
        names,
    } = replay("sessions");
    assert!(
        sessions > 100 && splits > 1000 && failing > 20 && names > 1000,
        "{sessions} sessions, {splits} splits, {failing} with a failed region, {names} calls named"
    );
}

#[test]
fn random_sessions_read_like_transformers_serve() {
    let Replayed {
        sessions, failing, ..
    } = replay("random");
    assert!(
        sessions > 100 && failing > 20,
        "{sessions} sessions, {failing} with a failed region"
    );
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

/// Thinking, calls named inside their JSON, and content.
fn json_calls() -> ResponseTemplate {
    load_response_template(&json!({
        "start_anchor": "<|assistant|>",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>"},
            "tool_calls": {
                "open": "<call>", "close": "</call>", "repeats": true, "content": "json",
                "transform": {"type": "function", "function": "{content}"}
            },
            "content": {"close": "<|end|>", "optional": false}
        }
    }))
    .unwrap()
}

/// What smg reads of `output` as a stream, the same in one feed, at every
/// two-way split and in chunks of four characters; fed by the reasoning
/// parser, or (`merge`) by the tool parser, which reads reasoning as content.
fn stream_read(template: &ResponseTemplate, tail: &str, output: &str, merge: bool) -> Read {
    let read = |chunks: &[&str]| {
        let state = ResponseParserState::new(template, tail, &[], false);
        let mut read = Read::default();
        for chunk in chunks.iter().copied().map(Some).chain([None]) {
            let content = if merge {
                chunk.map(str::to_owned)
            } else {
                let (reasoning, content) = state.reasoning(chunk);
                read.reasoning.push_str(&reasoning);
                Some(content)
            };
            let (content, calls) = state.tools(content.as_deref());
            read.content.push_str(&content);
            push_calls(&mut read, calls);
        }
        read
    };
    let whole = read(&[output]);
    let bounds: Vec<usize> = output.char_indices().map(|(b, _)| b).collect();
    for &b in &bounds[1..] {
        assert_eq!(read(&[&output[..b], &output[b..]]), whole, "split at {b}");
    }
    let mut fours: Vec<&str> = bounds
        .chunks(4)
        .map(|c| c[0])
        .zip(bounds.chunks(4).skip(1).map(|c| c[0]).chain([output.len()]))
        .map(|(a, b)| &output[a..b])
        .collect();
    fours.retain(|c| !c.is_empty());
    assert_eq!(read(&fours), whole, "chunks of four");
    whole
}

fn content_read(reasoning: &str, content: &str, calls: &[Value]) -> Read {
    Read {
        reasoning: reasoning.to_owned(),
        content: content.to_owned(),
        calls: calls.iter().map(|c| call(&c[0], &c[1])).collect(),
    }
}

#[test]
fn a_region_that_fails_is_content_with_the_rest_of_the_output() {
    let template = json_calls();
    let bad = r#"<call>{"name": "f", "arguments": {"city": Paris}}</call>"#;
    let output = format!("<think>plan</think>Checking.{bad}Sorry.");
    let rest = format!("Checking.{bad}Sorry.");
    assert_eq!(
        stream_read(&template, "", &output, false),
        content_read("plan", &rest, &[])
    );
    assert_eq!(
        stream_read(&template, "", &output, true),
        content_read("", &format!("plan{rest}"), &[])
    );
    // A complete output reads the message parsed before the region, then
    // the output from it on.
    let state = ResponseParserState::new(&template, "", &[], false);
    assert_eq!(state.reasoning_complete(&output), ("plan".into(), rest));
    assert!(state.take_error().is_some());
    assert!(state.take_error().is_none());

    // Calls that closed before the region stay; an output cut off inside a
    // call is the same.
    let output = r#"<call>{"name": "f", "arguments": {}}</call>Then <call>{"name": "g"#;
    let f = json!(["f", {}]);
    for merge in [false, true] {
        assert_eq!(
            stream_read(&template, "", output, merge),
            content_read("", r#"Then <call>{"name": "g"#, std::slice::from_ref(&f))
        );
    }
    let state = ResponseParserState::new(&template, "", &[], false);
    let (_, content) = state.reasoning_complete(output);
    assert_eq!(content, r#"Then<call>{"name": "g"#);
    assert_eq!(state.tools(Some(&content)).1.len(), 1);
}

#[test]
fn a_value_json_cannot_hold_or_a_call_without_a_name_fails_its_region() {
    let template = json_calls();
    for bad in [
        r#"<call>{"name": "f", "arguments": {"x": NaN}}</call>"#,
        r#"<call>{"arguments": {}}</call>"#,
    ] {
        let output = format!("Hi {bad} ok");
        assert_eq!(
            stream_read(&template, "", &output, false),
            content_read("", &output, &[])
        );
    }
    // An empty value holds no calls.
    let template = load_response_template(&json!({"start_anchor": "<s>", "fields": {
        "tool_calls": {"open": "<call>", "close": "</call>", "content": "json"},
        "content": {}}}))
    .unwrap();
    assert_eq!(
        stream_read(&template, "", "<call>[]</call>ok", false),
        content_read("", "ok", &[])
    );
}

#[test]
fn a_prompt_that_fails_leaves_the_output_as_content() {
    let template = json_calls();
    let output = "<think>plan</think>Hi";
    assert_eq!(
        stream_read(&template, "<call>{bad</call>", output, false),
        content_read("", output, &[])
    );
    let state = ResponseParserState::new(&template, "<call>{bad</call>", &[], false);
    assert_eq!(
        state.reasoning_complete(output),
        (String::new(), output.into())
    );
    assert!(state.take_error().is_some());
}

#[test]
fn a_missing_required_field_is_not_an_error() {
    // `content` is required, and transformers raises without it.
    let template = json_calls();
    assert!(parse_response("<think>plan</think>", &template, "", &[]).is_err());
    assert_eq!(
        stream_read(&template, "", "<think>plan</think>", false),
        content_read("plan", "", &[])
    );
    let state = ResponseParserState::new(&template, "", &[], false);
    assert_eq!(
        state.reasoning_complete("<think>plan</think>"),
        ("plan".into(), String::new())
    );
    assert!(state.take_error().is_none());
}

/// The `transformers serve` template for qwen3_5 checkpoints.
fn serve_qwen3_5() -> ResponseTemplate {
    load_response_template(&json!({
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>"},
            "tool_calls": {
                "open_pattern": "\\s*<tool_call>\\s*<function=(?P<name>[^>\\n]+)>",
                "close_pattern": "</function>\\s*</tool_call>",
                "repeats": true, "content": "xml-inline",
                "content_args": {"tag_pattern": "<parameter=(?P<key>[^>\\n]+)>\\s*(?P<value>.*?)\\s*</parameter>"},
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
            },
            "content": {"close_pattern": "\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>)"}
        }
    }))
    .unwrap()
}

/// A stream holds back at most 8 KiB that could still begin a delimiter, so
/// each chunk is searched in time bounded by that; transformers holds it all.
#[test]
fn a_stream_holds_back_a_bounded_amount_of_text() {
    const LIMIT: usize = 8 * 1024;
    let template = serve_qwen3_5();
    // An opener that never completes, and whitespace a delimiter can start with.
    for (head, unit) in [("<tool_call>\n<function=", "x"), ("a", " ")] {
        let state = ResponseParserState::new(&template, "", &[], false);
        let (mut fed, mut read) = (0, 0);
        for chunk in [head.to_owned()]
            .into_iter()
            .chain((0..256).map(|_| unit.repeat(256)))
        {
            fed += chunk.len();
            let (reasoning, content) = state.reasoning(Some(&chunk));
            read += reasoning.len() + content.len();
            assert!(fed - read <= LIMIT, "{head:?}: {} bytes held", fed - read);
        }
        let (reasoning, content) = state.reasoning(None);
        assert_eq!(read + reasoning.len() + content.len(), fed, "{head:?}");
    }
    // A delimiter that starts in the text still held back is found.
    let output = format!(
        "<tool_call>\n<function={}<think>plan</think>done",
        "x".repeat(20_000)
    );
    let state = ResponseParserState::new(&template, "", &[], false);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for chunk in output
        .as_bytes()
        .chunks(1000)
        .map(|c| std::str::from_utf8(c).unwrap())
        .map(Some)
        .chain([None])
    {
        let (r, c) = state.reasoning(chunk);
        reasoning.push_str(&r);
        content.push_str(&c);
    }
    assert_eq!(reasoning, "plan");
    assert!(content.starts_with("<tool_call>\n<function=x") && content.ends_with("xdone"));
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
    let items = |text| state.tool_items(text);
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
    let (text, items) = state.tool_items(Some("<call name=\"f\">{}</call>"));
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
        state.reasoning(Some("<call name=\"f\">{")),
        (String::new(), String::new())
    );
    assert_eq!(state.tool_items(Some("")), (String::new(), vec![name("f")]));
    // The region closes and reasoning follows in one chunk: the reasoning
    // waits, and the tool parser runs for the arguments.
    assert_eq!(
        state.reasoning(Some("}</call><think>more")),
        (String::new(), String::new())
    );
    assert!(!state.in_reasoning());
    assert_eq!(
        state.tool_items(Some("")),
        (String::new(), vec![arguments(json!({}))])
    );
    assert_eq!(state.reasoning(None), ("more".into(), String::new()));
}

#[test]
fn a_region_that_fails_after_deferred_output_is_content_after_it() {
    let named = |state: &ResponseParserState| {
        state.tool_items(Some("<call name=\"f\">{"));
        let (text, items) = state.tool_items(Some("}</call>Hi <call name=\"g\">{bad"));
        assert_eq!((text, items), (String::new(), vec![arguments(json!({}))]));
    };
    // The deferred text comes first, then the region from its opener on; the
    // rest of the output passes through.
    let state = named_state(&named_calls(), "");
    named(&state);
    assert_eq!(
        state.tool_items(Some("</call>")),
        ("Hi <call name=\"g\">{bad</call>".into(), vec![])
    );
    assert_eq!(state.tool_items(Some("x")), ("x".into(), vec![]));
    assert!(state.take_error().is_some());
    // So at the end of the output.
    let state = named_state(&named_calls(), "");
    named(&state);
    assert_eq!(
        state.tool_items(None),
        ("Hi <call name=\"g\">{bad".into(), vec![])
    );
    assert!(state.take_error().is_some());
}

#[test]
fn a_named_call_whose_region_fails_keeps_only_its_name() {
    let state = named_state(&named_calls(), "");
    assert_eq!(
        state.tool_items(Some("Hi <call name=\"f\">{\"a\"")),
        ("Hi ".into(), vec![name("f")])
    );
    // The region, opener included, is content after the name.
    assert_eq!(
        state.tool_items(Some(": 1 oops</call>Bye")),
        ("<call name=\"f\">{\"a\": 1 oops</call>Bye".into(), vec![])
    );
    assert_eq!(state.tool_items(None), (String::new(), vec![]));
    assert!(state.take_error().is_some());
}

#[test]
fn a_call_is_named_only_by_a_capture_every_call_takes() {
    // A region left open in the prompt is named on the first call.
    let state = named_state(&named_calls(), "<call name=\"f\">{");
    assert_eq!(
        state.tool_items(Some("\"a\"")),
        (String::new(), vec![name("f")])
    );
    // An empty capture names no call.
    let state = named_state(&named_calls(), "");
    let (_, items) = state.tool_items(Some("<call name=\"\">{"));
    assert_eq!(items, vec![]);

    let unnamed = |edit: fn(&mut Value)| {
        let mut template = named_calls();
        edit(&mut template["fields"]["tool_calls"]);
        let state = named_state(&template, "");
        let (_, items) = state.tool_items(Some("<call name=\"f\">{"));
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

/// Text at the end of the prompt that could begin a delimiter waits for the
/// output; once the output shows that it does not, a stream shows only the
/// output.
#[test]
fn a_stream_shows_no_prompt_text() {
    let template = serve_qwen3_5();
    let cases = [
        ("<think>\n\n</think>\n\n", "It is sunny.", "It is sunny."),
        // A continued message: the held text is where the client's prefix ends.
        (
            "<think>\n\n</think>\n\nThe weather ",
            "is sunny.",
            "is sunny.",
        ),
        ("<think>\n\n</think>\n\nso x <", "= 5", "= 5"),
    ];
    for (tail, output, content) in cases {
        for merge in [false, true] {
            assert_eq!(
                stream_read(&template, tail, output, merge),
                content_read("", content, &[]),
                "{tail:?}"
            );
        }
    }
    // The parsed message holds the prompt's text, as transformers' does.
    let state = ResponseParserState::new(&template, "<think>\n\n</think>\n\nso x <", &[], false);
    assert_eq!(
        state.reasoning_complete("= 5"),
        (String::new(), "so x <= 5".into())
    );
}
