//! Templates compiled into symphony tables.
//!
//! - Every public template of `fixtures/compiled/` either compiles or is refused with the
//!   constructs `corpus.tsv` pins, and every template of `fixtures/hf/` that is read, with those
//!   `to_symphony.tsv` pins (`TO_SYMPHONY_WRITE=1` rewrites them). A refusal lists every construct
//!   no table reads, with its code.
//! - A call's value reads as transformers' `_sub_parse`, `_coerce` and `_coerce_tool_calls` read
//!   it, over every row `fixtures/compiled/values.jsonl` records with transformers' own functions.
//! - Templates written for these tests on abstract tags compile into tables that read random
//!   outputs as the template's reader in this crate does, at every chunking, keeping symphony's
//!   contract.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture or table should panic"
)]

mod common;
mod replay;

use std::{collections::BTreeMap, fmt::Write as _};

use common::{fixtures_dir, read_json, read_jsonl, untag};
use openai_protocol::common::Tool;
use replay::{check_contract, compiled, compiled_dir, pinned, plans, rows, run, Said};
use serde_json::{json, Value};
use smg_response_template::{
    load_response_template, read_response_template,
    to_symphony::{compile, Compiled, Construct, Refusals, TRANSFORMERS},
    Event as RegionEvent, ResponseParser, ResponseTemplate,
};
use symphony::{
    shape::{Dialect, Duplicates, Schema, ValueParser},
    Declared, EngineFinish, ValueRead,
};

/// One line for a pinned list: the table's summary, or every refusal.
fn outcome(id: &str, result: Option<&Result<Compiled, Refusals>>) -> String {
    match result {
        None => format!("{id}\tnot read\n"),
        Some(Ok(compiled)) => format!("{id}\ttable\t{}\n", compiled.summary),
        Some(Err(refusals)) => {
            let mut line = format!("{id}\trefused");
            for (at, refusal) in refusals.refusals().iter().enumerate() {
                let sep = if at == 0 { '\t' } else { ' ' };
                let _ = write!(
                    line,
                    "{sep}{} ({}: {})",
                    refusal.construct, refusal.scope, refusal.detail
                );
            }
            line + "\n"
        }
    }
}

#[test]
fn the_public_templates_compile_or_are_refused_as_pinned() {
    let mut seen = String::new();
    for row in rows(&compiled_dir()) {
        seen.push_str(&outcome(&row.id, compiled(&row).as_ref()));
    }
    pinned("corpus.tsv", &seen);
}

#[test]
fn the_fixture_templates_compile_or_are_refused_as_pinned() {
    let mut seen = String::new();
    for row in rows(&fixtures_dir()) {
        let result = compiled(&row);
        if result.is_some() {
            seen.push_str(&outcome(&row.id, result.as_ref()));
        }
    }
    pinned("to_symphony.tsv", &seen);
}

#[test]
fn a_refusal_is_displayed_as_the_first_and_lists_every_construct() {
    let template = read_response_template(&json!({
        "start_anchor_pattern": "<a>\\s*",
        "fields": {
            "content": {"open": "<c>", "close": "</c>", "repeats": true},
            "tool_calls": {
                "open": "<t>", "close": "</t>", "content": "kv-lines",
                "transform": {"function": {"name": "{name}", "arguments": "{content}"}}
            }
        }
    }))
    .unwrap();
    let refused = compile(&template, &[]).unwrap_err();
    assert_eq!(
        refused.constructs(),
        [
            Construct::AnchorNotFinite,
            Construct::RepeatedTextField,
            Construct::NonRepeatingCalls,
            Construct::KeyValueLines,
            Construct::HeaderName,
        ]
    );
    assert_eq!(
        refused.to_string(),
        "response_template: AnchorNotFinite: a start anchor pattern that is not literal text"
    );
}

/// A template for each construct of a field, a delimiter or a call that no table reads, refused
/// with its code.
#[test]
fn each_construct_no_table_reads_is_refused_by_name() {
    let call = |field: Value| json!({"start_anchor": "<a>", "fields": {"tool_calls": field}});
    let named = json!({"function": {"name": "{name}", "arguments": "{content}"}});
    let text = |fields: Value| json!({"start_anchor": "<a>", "fields": fields});
    let cases = [
        (
            Construct::ImplicitField,
            text(json!({"thinking": {"close": "</t>"}})),
        ),
        (
            Construct::SharedField,
            text(json!({
                "thinking": {"open": "<t>", "close": "</t>"},
                "reasoning_content": {"open": "<r>", "close": "</r>"},
            })),
        ),
        (
            Construct::TextFieldContent,
            text(json!({"content": {"open": "<c>", "close": "</c>", "content": "json"}})),
        ),
        (
            Construct::TextFieldTransform,
            text(
                json!({"content": {"open": "<c>", "close": "</c>", "transform": {"text": "{content}"}}}),
            ),
        ),
        (
            Construct::RepeatedTextField,
            text(json!({"content": {"open": "<c>", "close": "</c>", "repeats": true, "join": ""}})),
        ),
        (
            Construct::Default,
            json!({
                "start_anchor": "<a>",
                "defaults": {"content": "hi"},
                "fields": {"content": {"open": "<c>", "close": "</c>"}},
            }),
        ),
        (
            // transformers' message holds the default whether or not a field writes it.
            Construct::Default,
            json!({
                "start_anchor": "<a>",
                "defaults": {"thinking": "hi"},
                "fields": {"content": {"open": "<c>", "close": "</c>"}},
            }),
        ),
        (
            Construct::TextFieldDelimiter,
            text(json!({"thinking": {"open_pattern": "<t>\\w+", "close": "</t>"}})),
        ),
        (
            Construct::UntrimmedSlack,
            text(json!({"content": {
                "open_pattern": "<c>\\s*", "close": "</c>", "content_args": {"strip": false},
            }})),
        ),
        (
            Construct::SlackBeforeMarker,
            text(json!({
                "thinking": {"open": "<t>", "close_pattern": "</t>\\s*"},
                "content": {"open": " <c>", "close": "</c>"},
            })),
        ),
        (
            // The pattern's match starts in the whitespace before `</t>`, ahead of the newline
            // the table would read first.
            Construct::SlackBeforeMarker,
            text(json!({"thinking": {"open": "<t>", "close_pattern": "(?:\\s*</t>|\\n)"}})),
        ),
        (
            Construct::SlackBeforeMarker,
            text(json!({
                "content": {"close_pattern": "\\s*<e>"},
                "notes": {"open": "\n", "close": "</n>"},
            })),
        ),
        (
            Construct::AlternativeOrder,
            text(json!({"thinking": {"open_pattern": "(?:<t>|<t>x)", "close": "</t>"}})),
        ),
        (
            Construct::NoLiteral,
            text(json!({"thinking": {"open_pattern": "[^<]*<t>", "close": "</t>"}})),
        ),
        (
            Construct::ZeroWidthOpen,
            text(json!({"thinking": {"open_pattern": "(?=[\\s\\S]*?</t>)", "close": "</t>"}})),
        ),
        (
            Construct::SharedOpenMarker,
            text(json!({
                "thinking": {"open": "<x>", "close": "</t>"},
                "notes": {"open": "<x>", "close": "</n>"},
            })),
        ),
        (
            Construct::MarkerInsideMarker,
            text(json!({
                "thinking": {"open": "<t>", "close": "</t>"},
                "content": {"open": "<c>", "close": "z</t>"},
            })),
        ),
        (
            Construct::NonRepeatingCalls,
            call(json!({"open": "<c>", "close": "</c>", "content": "json", "transform": named})),
        ),
        (
            Construct::TransformEach,
            call(json!({
                "open": "<c>", "close": "</c>", "repeats": true, "content": "json",
                "transform": named, "transform_each": true,
            })),
        ),
        (
            Construct::CallTransform,
            call(
                json!({"open": "<c>", "close": "</c>", "repeats": true, "transform": {"function": 5}}),
            ),
        ),
        (
            Construct::DottedPath,
            call(json!({
                "open": "<c>", "close": "</c>", "repeats": true, "content": "json",
                "transform": {"function": {"name": "{content.name}", "arguments": "{content.args}"}},
            })),
        ),
        (
            Construct::CallContent,
            call(
                json!({"open": "<c>", "close": "</c>", "repeats": true, "content": "int", "transform": named}),
            ),
        ),
        (
            Construct::JsonArguments,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "json", "transform": named,
            })),
        ),
        (
            Construct::JsonEnvelope,
            call(json!({
                "open": "<c>", "close": "</c>", "repeats": true, "content": "json",
                "transform": {"function": "{content}"},
            })),
        ),
        (
            Construct::JsonDialect,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "json", "content_args": {"unquoted_keys": true}, "transform": named,
            })),
        ),
        (
            Construct::TaggedParameters,
            call(json!({
                "open": "<c>", "close": "</c>", "repeats": true, "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)</k>"},
                "transform": {"function": "{content}"},
            })),
        ),
        (
            // A tag pattern this crate's reader does not run (a lookbehind).
            Construct::TagPattern,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)(?<=x)</k>"},
                "transform": named,
            })),
        ),
        (
            // A value with nothing after it to end it.
            Construct::TagPattern,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)"},
                "transform": named,
            })),
        ),
        (
            // A greedy value of any character, which only the call's end decides.
            Construct::TagPattern,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*)</k>"},
                "transform": named,
            })),
        ),
        (
            // No literal to find a parameter by.
            Construct::TagPattern,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "(?P<key>\\w+)=(?P<value>.*?);"},
                "transform": named,
            })),
        ),
        (
            // A value that ends where the next tag begins, in a call with text before its
            // closing marker.
            Construct::EndBeforeTrailer,
            call(json!({
                "open": "<c>", "close_pattern": "</f>\\s*</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)(?:</k>|(?=<k=))"},
                "transform": {"function": {"name": "f", "arguments": "{content}"}},
            })),
        ),
        (
            Construct::NestedValueParser,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)</k>",
                    "value_parser": {"name": "kv-lines"},
                },
                "transform": named,
            })),
        ),
        (
            Construct::CallAlternatives,
            call(json!({
                "open_pattern": "(?:<c>(?P<name>\\w+)>|<d>)", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)</k>"},
                "transform": named,
            })),
        ),
        (
            Construct::KeyValueLines,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "kv-lines", "transform": named,
            })),
        ),
        (
            Construct::MergeDuplicates,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)>", "close": "</c>", "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)</k>",
                    "merge_duplicates": true,
                },
                "transform": named,
            })),
        ),
        (
            Construct::HeaderName,
            call(
                json!({"open": "<c>", "close": "</c>", "repeats": true, "content": "json", "transform": named}),
            ),
        ),
        (
            Construct::UnboundedHeaderRun,
            call(json!({
                "open_pattern": "<c>(?P<name>\\w+)[\\s\\S]*<m>", "close": "</c>", "repeats": true,
                "content": "json", "transform": named,
            })),
        ),
    ];
    for (construct, spec) in cases {
        let template = read_response_template(&spec).unwrap_or_else(|e| panic!("{construct}: {e}"));
        let refused = compile(&template, &[]).expect_err(construct.code());
        assert!(
            refused.constructs().contains(&construct),
            "{construct}: {:?}",
            refused.refusals()
        );
    }
}

/// A value parser as the fixture writes it (`value_parser` of `content_args`), as a table's shape
/// holds it.
fn value_parser(spec: &Value) -> ValueParser {
    if spec.is_null() {
        return ValueParser::Raw;
    }
    let args = &spec["args"];
    let flag = |key: &str, default: bool| args.get(key).and_then(Value::as_bool).unwrap_or(default);
    match spec["name"].as_str().unwrap_or("text") {
        "text" => ValueParser::Text {
            strip: flag("strip", true),
        },
        "json" => ValueParser::Json {
            dialect: Dialect {
                bare_keys: flag("unquoted_keys", false),
                string_delims: args["string_delims"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|pair| {
                        (
                            pair[0].as_str().unwrap().into(),
                            pair[1].as_str().unwrap().into(),
                        )
                    })
                    .collect(),
            },
            allow_non_json: flag("allow_non_json", false),
            strip: flag("strip", true),
        },
        "int" => ValueParser::Int,
        "float" => ValueParser::Float,
        "bool" => ValueParser::Bool {
            strip: flag("strip", true),
        },
        other => panic!("no value parser {other}"),
    }
}

/// What the reader writes, read back as JSON.
fn json_of(written: &str) -> Value {
    serde_json::from_str(written).unwrap_or_else(|e| panic!("{written:?} is no JSON: {e}"))
}

/// The value transformers recorded, or, where it raised, the text: the reader writes the text
/// where transformers raises.
fn recorded(row: &Value) -> Value {
    match row.get("output") {
        Some(output) => untag(output),
        None => row["text"].clone(),
    }
}

/// A tagged value reads as `_sub_parse` and then `_coerce` with `_schema_types` read it, a string
/// member of JSON arguments is cast the same way when it closes, and held until then only when a
/// cast could change it; over every schema and text `fixtures/compiled/values.jsonl` records,
/// recorded with transformers' own functions.
#[test]
fn values_are_typed_as_transformers_types_them() {
    let dir = compiled_dir();
    let manifest = read_json(&dir, "manifest.json");
    let schemas = untag(&manifest["value_schemas"]);
    let tools = untag(&manifest["value_tools"]);
    let root = &tools[0]["function"]["parameters"];
    let api: Vec<Tool> = serde_json::from_value(tools.clone()).unwrap();
    let declared = Declared::of(&api);
    let mut casts: BTreeMap<String, bool> = BTreeMap::new();
    let mut counted = BTreeMap::new();
    for row in read_jsonl(&dir, "values") {
        let kind = row["kind"].as_str().unwrap();
        *counted.entry(kind.to_owned()).or_insert(0) += 1;
        match kind {
            "coerce" => {
                let key = row["schema"].as_str().unwrap();
                let text = row["text"].as_str().unwrap();
                let schema = Schema {
                    root,
                    property: &schemas[key],
                };
                let expected = recorded(&row);
                let read = TRANSFORMERS.read(&ValueParser::Raw, text, Some(schema));
                assert_eq!(json_of(&read), expected, "{key} {text:?}");
                let kept = expected == Value::String(text.to_owned());
                let member =
                    TRANSFORMERS.member(&Value::String(text.to_owned()).to_string(), Some(schema));
                assert_eq!(
                    member.as_deref().map(json_of),
                    (!kept).then_some(expected),
                    "{key} {text:?} as a member"
                );
                *casts.entry(key.to_owned()).or_default() |= !kept && row.get("error").is_none();
            }
            "sub_parse" => {
                let parser = value_parser(&row["value_parser"]);
                let text = row["text"].as_str().unwrap();
                let read = TRANSFORMERS.read(&parser, text, None);
                assert_eq!(json_of(&read), recorded(&row), "{parser:?} {text:?}");
            }
            "coerce_tool_calls" => {
                let call = untag(&row["value"]);
                let name = call["function"]["name"].as_str().unwrap();
                let output = untag(&row["output"]);
                let expected = &output["function"]["arguments"];
                for (key, value) in call["function"]["arguments"].as_object().unwrap() {
                    let schema = declared
                        .parameter(name, key)
                        .map(|(root, property)| Schema { root, property });
                    // A tagged value whose text reads as this JSON is typed as the argument is.
                    let read = TRANSFORMERS.read(&ValueParser::json(), &value.to_string(), schema);
                    assert_eq!(json_of(&read), expected[key], "{call} {key} as a value");
                    // A string member of JSON arguments is cast when it closes.
                    if value.is_string() {
                        let member = TRANSFORMERS
                            .member(&value.to_string(), schema)
                            .map_or_else(|| value.clone(), |typed| json_of(&typed));
                        assert_eq!(member, expected[key], "{call} {key} as a member");
                    }
                }
            }
            other => panic!("no kind {other}"),
        }
    }
    // A member waits until it is whole only where a cast could change it.
    for (key, cast) in &casts {
        let schema = Schema {
            root,
            property: &schemas[key.as_str()],
        };
        assert_eq!(
            TRANSFORMERS.streams(&ValueParser::json(), Some(schema)),
            !cast,
            "{key}"
        );
    }
    assert!(TRANSFORMERS.streams(&ValueParser::json(), None));
    assert_eq!(TRANSFORMERS.duplicates(), Duplicates::Last);
    assert_eq!(
        counted,
        BTreeMap::from([
            ("coerce".to_owned(), 810),
            ("coerce_tool_calls".to_owned(), 9),
            ("sub_parse".to_owned(), 588),
        ])
    );
}

/// Templates written for these tests on abstract tags, each with the pieces its outputs are made
/// of: the constructs a compiled table reads, one or two at a time.
fn synthetic() -> Vec<(&'static str, Value, Vec<&'static str>)> {
    let filler = ["hi", " ", "\n", "x<", "<e", "turn", "é", "\u{a0}", "\u{1c}"];
    // The markers three times over, so that most outputs open a field.
    let with = |pieces: &[&'static str]| -> Vec<&'static str> {
        pieces
            .iter()
            .chain(pieces)
            .chain(pieces)
            .chain(filler.iter())
            .copied()
            .collect()
    };
    vec![
        (
            // Nothing outside the fields is kept; a close of two markers; a close list.
            "framing",
            json!({
                "start_anchor": "<turn>",
                "fields": {
                    "thinking": {"open": "<t>", "close": "<e><turn>"},
                    "content": {"open": "<c>", "close": ["<r>", "<e>"]},
                },
            }),
            with(&["<t>", "<e><turn>", "<e>", "<c>", "<turn>"]),
        ),
        (
            // Content outside the thought, trimmed; whitespace the delimiters take; a close of
            // the content that is no stop, dropped; two anchors.
            "slack",
            json!({
                "start_anchor": ["<turn>", "<turn2>"],
                "fields": {
                    "reasoning_content": {"open_pattern": "<t>\\s*", "close_pattern": "\\s*</t>\\s*"},
                    "content": {"close_pattern": "\\s*(?:<end>|<stop>)"},
                },
            }),
            with(&["<t>", "</t>", "<end>", "<turn>", "  ", "\n\n"]),
        ),
        (
            // A field the message has no place for; content kept as written; an anchor pattern
            // of three literals.
            "kept",
            json!({
                "start_anchor_pattern": "<a>(?:<b>|<c>)?",
                "defaults": {"role": "assistant", "note": "x"},
                "fields": {
                    "notes": {"open": "<n>", "close": "</n>"},
                    "content": {"open": "<c>", "close": "</c>", "content_args": {"strip": false}},
                },
            }),
            with(&["<n>", "</n>", "<c>", "</c>", "<a>", "<b>"]),
        ),
        (
            // The text outside the thought is a field the message has no place for, closed by the
            // marker that opens the thought: the open wins.
            "outside-field",
            json!({
                "start_anchor": "<s>",
                "fields": {
                    "log": {"close": "<x>"},
                    "thinking": {"open": "<x>", "close": "</x>"},
                },
            }),
            with(&["<x>", "</x>", "<s>"]),
        ),
    ]
}

/// What the template's reader closes in each field, every value joined: a thought's and the
/// content's text as a stream gives them, which joins a field written twice.
fn read_joined(template: &ResponseTemplate, output: &str) -> Option<(String, String)> {
    let mut parser = ResponseParser::new(template, "", &[]).ok()?;
    let mut events = parser.feed(output).ok()?;
    events.extend(parser.finalize().ok()?.1);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for event in events {
        if let RegionEvent::RegionClose { field, value } = event {
            match (field.as_str(), value.as_str()) {
                ("thinking" | "reasoning_content", Some(text)) => reasoning.push_str(text),
                ("content", Some(text)) => content.push_str(text),
                _ => {}
            }
        }
    }
    Some((reasoning, content))
}

/// Templates on abstract tags compile into tables that read random outputs made of their own
/// literals as the template's reader in this crate does, every field written more than once
/// joined, whole and at every chunking, keeping the contract.
#[test]
fn compiled_tables_read_random_outputs_as_the_template_reader_does() {
    for (name, spec, pieces) in synthetic() {
        let template = load_response_template(&spec).unwrap();
        let format = compile(&template, &["<stop>", "<r>"])
            .unwrap_or_else(|why| panic!("{name}: {why}"))
            .format;
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ name.len() as u64;
        let mut next = |below: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize % below
        };
        for _ in 0..600 {
            let count = next(16);
            let output: String = (0..count).map(|_| pieces[next(pieces.len())]).collect();
            let Some((reasoning, content)) = read_joined(&template, &output) else {
                continue;
            };
            for (plan, cuts) in plans(&output, &pieces, 64).into_iter().take(40) {
                let place = format!("{name}: {output:?} ({plan})");
                let events = run(
                    &format,
                    &Declared::default(),
                    "",
                    &output,
                    &cuts,
                    &EngineFinish::Stop,
                );
                check_contract(&place, &output, &events);
                let said = Said::of(&events);
                assert_eq!(said.reasoning.concat(), reasoning, "{place}: reasoning");
                assert_eq!(said.content, content, "{place}: content");
            }
        }
    }
}

/// A message as the template's reader and a compiled table both give it: the thought and the
/// content, every value of a field joined, and each call's name and arguments.
type Read = (String, String, Vec<(String, Value)>);

/// What the template's reader gives for `output` with `tools`, every value of a text field joined
/// (a stream joins a field written twice), each call's name and arguments as its region closes
/// them; `None` where it raises.
fn read_message(template: &ResponseTemplate, output: &str, tools: &[Value]) -> Option<Read> {
    let mut parser = ResponseParser::new(template, "", tools).ok()?;
    let mut events = parser.feed(output).ok()?;
    events.extend(parser.finalize().ok()?.1);
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for event in events {
        if let RegionEvent::RegionClose { field, value } = event {
            match (field.as_str(), value.as_str()) {
                ("thinking" | "reasoning_content", Some(text)) => reasoning.push_str(text),
                ("content", Some(text)) => content.push_str(text),
                ("tool_calls", None) => {
                    let function = &value["function"];
                    let name = function["name"].as_str().unwrap_or_default().to_owned();
                    calls.push((name, function["arguments"].clone()));
                }
                _ => {}
            }
        }
    }
    Some((reasoning, content, calls))
}

/// What a client assembles from a table's events. With `cut`, a call the stream cut is closed:
/// its arguments' object closed, as the template's reader closes the region at the end of the
/// text. Without it, every call's arguments must be whole.
fn said_message(said: &Said, cut: bool) -> Read {
    let calls = said
        .calls
        .iter()
        .map(|(index, _, name)| {
            let arguments = said
                .arguments
                .iter()
                .find(|(at, _)| at == index)
                .map(|(_, json)| json.as_str())
                .unwrap_or_default();
            let value = serde_json::from_str(arguments)
                .or_else(|error| match arguments {
                    "" if cut => serde_json::from_str("{}"),
                    open if cut => serde_json::from_str(&format!("{open}}}")),
                    _ => Err(error),
                })
                .unwrap_or_else(|e| panic!("{arguments:?}: {e}"));
            (name.clone(), value)
        })
        .collect();
    (said.reasoning.concat(), said.content.clone(), calls)
}

/// The tools the outputs of the call templates call: `f` on abstract tags, and the weather tool
/// of the public templates' sessions.
fn call_tools() -> Vec<Value> {
    let tools = json!([
        {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "b": {"type": "boolean"},
                "s": {"type": "string"},
                "o": {"type": "object"},
            },
        }}},
        {"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        }}},
    ]);
    tools.as_array().cloned().unwrap()
}

/// Templates whose calls compile, each with the texts of its stop tokens and the pieces its
/// outputs are made of: two written for these tests on abstract tags, and the public templates
/// of `fixtures/compiled/` whose calls are tagged parameters.
fn call_templates() -> Vec<(String, Value, Vec<String>, Vec<&'static str>)> {
    let filler = ["hi", " ", "\n", "x<", "é", "\u{a0}"];
    let with = |pieces: &[&'static str]| -> Vec<&'static str> {
        pieces
            .iter()
            .chain(pieces)
            .chain(filler.iter())
            .copied()
            .collect()
    };
    let named =
        json!({"type": "function", "function": {"name": "{name}", "arguments": "{content}"}});
    let mut templates = vec![
        (
            // Nothing outside the fields is kept. A call's name ends its header; a value ends at
            // its close or where the next parameter begins, and reads as JSON where it is JSON; a
            // call closes at its marker or at the stream's stop.
            "abstract/calls-ahead".to_owned(),
            json!({
                "start_anchor": "<turn>",
                "fields": {
                    "thinking": {"open": "<t>", "close": "</t>"},
                    "tool_calls": {
                        "open_pattern": "<c (?P<name>\\w+)",
                        "close": ["</c>", "<stop>"],
                        "repeats": true,
                        "content": "xml-inline",
                        "content_args": {
                            "tag_pattern": "<k=(?P<key>\\w+)>(?P<value>.*?)(?:</k>|(?=<k=))",
                            "value_parser": {"name": "json", "args": {"allow_non_json": true}},
                        },
                        "transform": named,
                    },
                },
            }),
            vec!["<stop>".to_owned()],
            with(&[
                "<t>",
                "</t>",
                "<c f",
                "<c g",
                "<c ",
                "<k=n>",
                "<k=b>",
                "<k=s>",
                "<k=o>",
                "5",
                "true",
                "\"7\"",
                "{\"a\": 1}",
                "</k>",
                "</c>",
                "<turn>",
                "<k=n>5</k>",
                "<k=b>true",
                "<k=s>\"7\" ",
                "<k=o>{\"a\": [1]}</k>",
                "<c f<k=n>06</k></c>",
            ]),
        ),
        (
            // Content outside the call, trimmed, its close dropped. The template writes the
            // call's name; a value is the text as written; text before the call's closing
            // marker is its trailer.
            "abstract/calls-text".to_owned(),
            json!({
                "start_anchor": "<turn>",
                "fields": {
                    "content": {"close": "<end>"},
                    "tool_calls": {
                        "open": "<call>",
                        "close_pattern": "</args>\\s*</call>",
                        "repeats": true,
                        "content": "xml-inline",
                        "content_args": {
                            "tag_pattern": "<v (?P<key>\\w+)>(?P<value>.*?)</v>",
                            "value_parser": {"name": "text", "args": {"strip": false}},
                        },
                        "transform": {"function": {"name": "f", "arguments": "{content}"}},
                    },
                },
            }),
            Vec::new(),
            with(&[
                "<call>",
                "<v n>",
                "<v s>",
                "<v b>",
                " 5 ",
                "1",
                "a b",
                "</v>",
                "</args>",
                "</call>",
                "</args></call>",
                "<end>",
                "<turn>",
                "<v n>7</v>",
                "<v s> x </v>",
                "<v b>1</v>\n",
                "</args>\n</call>",
                "<call><v s>y</v></args></call>",
            ]),
        ),
    ];
    let public: [(&str, &[&'static str]); 5] = [
        (
            "hub/muse-glimmer",
            &[
                "to=self<|message|>",
                "to=user<|message|>",
                "<|eom|>",
                "<atem:invoke",
                " name=\"get_weather\">",
                " x=\"1\" name=\"f\">",
                "<atem:parameter",
                " name=\"city\">",
                " name=\"days\" y>",
                "Paris",
                "\"Paris\"",
                "3",
                "</atem:parameter>",
                "</atem:invoke>",
                "<|start|>assistant",
                "<atem:invoke name=\"get_weather\">",
                "<atem:parameter name=\"city\">Paris</atem:parameter>",
                "<atem:parameter name=\"days\">3</atem:parameter> ",
            ],
        ),
        (
            "serve/qwen3_5",
            &[
                "<think>",
                "</think>",
                "<tool_call>",
                "<function=get_weather>",
                "<function=f>",
                "<parameter=city>",
                "<parameter=days>",
                "Paris",
                "3",
                "</parameter>",
                "</function>",
                "</tool_call>",
                "<|im_start|>assistant\n",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>\nParis\n</parameter>\n",
                "<parameter=days>\n3\n</parameter>\n",
                "</function>\n</tool_call>",
            ],
        ),
        (
            "trl/qwen3_5",
            &[
                "<think>",
                "</think>",
                "<tool_call>",
                "<function=get_weather>",
                "<function=f>",
                "<parameter=city>",
                "<parameter=days>",
                "Paris",
                "3",
                "[1]",
                "</parameter>",
                "</function>",
                "</tool_call>",
                "<|im_start|>assistant\n",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>\nParis\n</parameter>\n",
                "<parameter=days>\n3\n</parameter>\n",
                "</function>\n</tool_call>\n",
            ],
        ),
        (
            "hf-tests/qwen3_template",
            &[
                "<think>",
                "</think>",
                "<tool_call>",
                "<function=get_weather>",
                "<function=f>",
                "<parameter=city>",
                "<parameter=days>",
                "Paris",
                "3",
                "\"3\"",
                "</parameter>",
                "</tool_call>",
                "<|im_start|>assistant\n",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>\nParis\n</parameter>\n",
                "<parameter=days>\n3\n</parameter>\n",
            ],
        ),
        (
            "hf-tests/_XML_STRING_ARGS_TEMPLATE",
            &[
                "<tool_call>",
                "<function=get_weather>",
                "<parameter=city>",
                "<parameter=days>",
                "Paris",
                "3",
                "</parameter>",
                "</tool_call>",
                "<|im_start|>assistant\n",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>\nParis\n</parameter>\n",
                "<parameter=days>\n3\n</parameter>\n",
            ],
        ),
    ];
    let rows = rows(&compiled_dir());
    for (id, pieces) in public {
        let row = rows.iter().find(|row| row.id == id).unwrap();
        templates.push((
            id.to_owned(),
            row.spec.clone(),
            row.stops.clone(),
            with(pieces),
        ));
    }
    templates
}

/// Templates whose calls compile read random outputs made of their own literals as the template's
/// reader in this crate does, every field written more than once joined, whole and at every
/// chunking, keeping the contract: a stream that stops closes every call, as the end of the text
/// closes the reader's region, and one cut for length leaves a call cut, which reads the same once
/// closed.
#[test]
fn compiled_calls_read_random_outputs_as_the_template_reader_does() {
    let tools = call_tools();
    let api: Vec<Tool> = serde_json::from_value(Value::Array(tools.clone())).unwrap();
    let declared = Declared::of(&api);
    let mut read = 0;
    for (name, spec, stops, pieces) in call_templates() {
        let template = load_response_template(&spec).unwrap();
        let stops: Vec<&str> = stops.iter().map(String::as_str).collect();
        let format = compile(&template, &stops)
            .unwrap_or_else(|why| panic!("{name}: {why}"))
            .format;
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ name.len() as u64;
        let mut next = |below: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize % below
        };
        for _ in 0..600 {
            let count = next(16);
            let output: String = (0..count).map(|_| pieces[next(pieces.len())]).collect();
            let Some(expected) = read_message(&template, &output, &tools) else {
                continue;
            };
            read += 1;
            for (plan, cuts) in plans(&output, &pieces, 64).into_iter().take(40) {
                for (finish, cut) in [(EngineFinish::Stop, false), (EngineFinish::Length, true)] {
                    let place = format!("{name}: {output:?} ({plan}, {finish:?})");
                    let events = run(&format, &declared, "", &output, &cuts, &finish);
                    check_contract(&place, &output, &events);
                    assert_eq!(said_message(&Said::of(&events), cut), expected, "{place}");
                }
            }
        }
    }
    assert!(read > 3000, "{read} outputs read");
}
