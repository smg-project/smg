//! Test support the crate's modules share: readers of an event list, and a
//! driver that feeds a parser one turn.

use openai_protocol::common::{Function, Tool};
use serde_json::{json as value, Value};

use crate::{
    event::{Event, Events, FinishReason},
    format::Format,
    grammar::{GrammarStart, Options},
    input::{EngineFinish, Input},
    parser::Parser,
};

/// The output bytes the events account for, in order: content, reasoning,
/// dropped and malformed text, and the source bytes of the tool-call events.
pub(crate) fn bytes(events: &[Event]) -> String {
    events
        .iter()
        .map(|event| match event {
            Event::Content(text) | Event::Reasoning(text) => &*text.text,
            Event::Dropped { text, .. } | Event::Malformed { text, .. } => &*text.text,
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => &*source.text,
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
        })
        .collect()
}

/// The content text, joined.
pub(crate) fn content(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Content(text) => Some(&*text.text),
            _ => None,
        })
        .collect()
}

/// The reasoning text, joined.
pub(crate) fn reasoning(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Reasoning(text) => Some(&*text.text),
            _ => None,
        })
        .collect()
}

/// The argument fragments of call `call`, joined.
pub(crate) fn arguments_of(events: &[Event], call: u32) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCallArguments { index, json, .. } if *index == call => Some(json.as_str()),
            _ => None,
        })
        .collect()
}

/// The names of the calls, in order.
pub(crate) fn names<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

/// The calls as (name, arguments), the fragments of each joined; an
/// arguments event before any call started fails the test.
pub(crate) fn calls(events: &[Event]) -> Vec<(String, String)> {
    let mut calls: Vec<(String, String)> = Vec::new();
    for event in events {
        match event {
            Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
            Event::ToolCallArguments { json, .. } => {
                calls.last_mut().expect("a call started").1.push_str(json);
            }
            _ => {}
        }
    }
    calls
}

/// The engine's finish: its reason as the engine gave it (the adapters refine it to
/// `tool_calls`), and the calls counted.
pub(crate) fn finish(events: &[Event]) -> (FinishReason, u32) {
    events
        .iter()
        .find_map(|event| match event {
            Event::Finish {
                reason, tool_calls, ..
            } => Some((reason.clone(), *tool_calls)),
            _ => None,
        })
        .expect("a finish")
}

/// One turn through `parser`: the prompt, the output in `pieces`, a stop; the
/// events it produced.
pub(crate) fn drive(
    parser: &mut impl Parser,
    prompt: &str,
    pieces: &[&str],
) -> Vec<Event<'static>> {
    drive_to(parser, prompt, pieces, EngineFinish::Stop)
}

/// [`drive`] with the engine's finish reason chosen.
pub(crate) fn drive_to(
    parser: &mut impl Parser,
    prompt: &str,
    pieces: &[&str],
    finish: EngineFinish,
) -> Vec<Event<'static>> {
    let mut out = Events::new();
    parser
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            &mut out,
        )
        .expect("the prompt");
    for piece in pieces {
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: piece,
                    spans: &[],
                },
                &mut out,
            )
            .expect("a delta");
    }
    parser
        .feed(Input::End { finish }, &mut out)
        .expect("the end");
    out.into_owned().drain()
}

/// What a table said of an output: the calls, the finish, the content and the reasoning.
#[derive(Debug, PartialEq)]
pub(crate) struct Said {
    pub(crate) calls: Vec<(String, String)>,
    pub(crate) finish: Option<(FinishReason, u32)>,
    pub(crate) content: String,
    pub(crate) reasoning: String,
}

pub(crate) fn said(events: &[Event]) -> Said {
    let finish = events.iter().find_map(|event| match event {
        Event::Finish {
            reason, tool_calls, ..
        } => Some((reason.clone(), *tool_calls)),
        _ => None,
    });
    Said {
        calls: calls(events),
        finish,
        content: content(events),
        reasoning: reasoning(events),
    }
}

/// `output` read by `read` whole, one character at a time and at every two-way cut says the
/// same; what it said.
pub(crate) fn the_same_at_every_cut(
    read: impl Fn(&[&str]) -> Vec<Event<'static>>,
    output: &str,
) -> Said {
    let whole = said(&read(&[output]));
    let characters: Vec<&str> = output
        .char_indices()
        .map(|(at, character)| &output[at..at + character.len_utf8()])
        .collect();
    assert_eq!(said(&read(&characters)), whole, "one character at a time");
    for cut in 1..output.len() {
        if !output.is_char_boundary(cut) {
            continue;
        }
        assert_eq!(
            said(&read(&[&output[..cut], &output[cut..]])),
            whole,
            "cut at {cut}"
        );
    }
    whole
}

/// The structural tag of a forced call on `format`, a table whose layout writes `after` after
/// the thought's close (`close`), in every configuration a request and an engine present:
/// `tool_choice` naming a function (the first of `tools`) and `required` (all of them), the
/// prompt's thought closed and open, the grammar started where the engine does not say, after
/// its reasoning parser and at the first token. Where the grammar starts after the close, the tag
/// begins with `after` and then the calls, whose every way in begins with `marker`; with the
/// thought closed, where the engine does not say or starts at the first token, the calls at once,
/// as before, and on every start where the prompt wrote `after` itself after the close it wrote
/// (thinking switched off); with the thought open, the thought closed by its marker (at the first
/// token) or the free text that may hold the close (where the engine does not say) comes first,
/// then `after`, then the calls.
pub(crate) fn a_forced_tag_begins_with_the_templates_text(
    format: &Format,
    tools: &[Tool],
    close: &str,
    after: &str,
    marker: &str,
) {
    let text = value!({"type": "const_string", "value": after});
    for (choice, one_call, passed) in [
        ("a named function", true, &tools[..1]),
        ("required", false, tools),
    ] {
        let tag_of = |reasoning_open: bool, thought_closed: bool, start: GrammarStart| -> Value {
            format
                .grammar_for(
                    passed,
                    Options {
                        at_least_one: true,
                        reasoning_open,
                        thought_closed,
                        one_call,
                        start,
                    },
                )
                .unwrap_or_else(|| panic!("{choice}: no tag"))
                .payload()
        };
        let tag = |reasoning_open: bool, start: GrammarStart| tag_of(reasoning_open, false, start);
        // With the thought closed, where the engine does not say: the calls at once.
        let calls = tag(false, GrammarStart::Unknown);
        let calls = &calls["format"];
        assert_eq!(calls["type"], "triggered_tags", "{choice}: {calls}");
        assert_eq!(calls["at_least_one"], true, "{choice}: {calls}");
        assert_eq!(
            calls["stop_after_first"],
            if one_call { value!(true) } else { Value::Null },
            "{choice}: {calls}"
        );
        let tags = calls["tags"].as_array().expect("tags");
        assert!(!tags.is_empty(), "{choice}: {calls}");
        for tag in tags {
            let begin = tag["begin"].as_str().expect("a begin");
            assert!(
                begin.starts_with(marker),
                "{choice}: {begin:?} does not begin with {marker:?}"
            );
        }
        assert_eq!(
            tag(false, GrammarStart::FirstToken)["format"],
            *calls,
            "{choice}: at the first token with the thought closed, the calls at once"
        );
        // After the gate of the engine's reasoning parser, the thought closed or open: `after`,
        // then the calls.
        let gated = tag(false, GrammarStart::AfterReasoning);
        assert_eq!(
            gated["format"],
            value!({"type": "sequence", "elements": [text.clone(), calls.clone()]}),
            "{choice}: after the gate"
        );
        assert_eq!(
            tag(true, GrammarStart::AfterReasoning),
            gated,
            "{choice}: after the gate with the thought open"
        );
        // The prompt closed the thought itself and wrote `after` (thinking switched off): the
        // calls at once on every start, the template's text not written a second time.
        for start in [
            GrammarStart::Unknown,
            GrammarStart::AfterReasoning,
            GrammarStart::FirstToken,
        ] {
            assert_eq!(
                tag_of(false, true, start)["format"],
                *calls,
                "{choice}: the prompt wrote the template's text, {start:?}"
            );
        }
        // The thought open, from the first token: the thought closed by its marker, `after`,
        // the calls.
        let first = tag(true, GrammarStart::FirstToken);
        let elements = first["format"]["elements"]
            .as_array()
            .unwrap_or_else(|| panic!("{choice}: {first}"));
        assert_eq!(first["format"]["type"], "sequence", "{choice}: {first}");
        assert_eq!(elements.len(), 3, "{choice}: {first}");
        assert_eq!(elements[0]["type"], "tag", "{choice}: {first}");
        assert_eq!(elements[0]["begin"], "", "{choice}: {first}");
        assert_eq!(elements[0]["end"], close, "{choice}: {first}");
        assert_eq!(elements[1], text, "{choice}: {first}");
        assert_eq!(elements[2], *calls, "{choice}: {first}");
        // The thought open, where the engine does not say: free text that may hold the close,
        // `after`, the calls.
        let unknown = tag(true, GrammarStart::Unknown);
        let elements = unknown["format"]["elements"]
            .as_array()
            .unwrap_or_else(|| panic!("{choice}: {unknown}"));
        assert_eq!(unknown["format"]["type"], "sequence", "{choice}: {unknown}");
        assert_eq!(elements.len(), 3, "{choice}: {unknown}");
        assert_eq!(elements[0]["type"], "any_text", "{choice}: {unknown}");
        assert_eq!(elements[1], text, "{choice}: {unknown}");
        assert_eq!(elements[2], *calls, "{choice}: {unknown}");
    }
}

/// The structural tag of a forced call on `format` after `prompt`, a rendered generation prompt
/// that closed the thought itself and wrote the template's text after it (a request with
/// thinking switched off), derived as the gateway derives it: `reasoning_open` and
/// `thought_closed` read from the prompt ([`Format::thought_closed`]). On every start the calls
/// at once, whose every way in begins with `marker`: the text the prompt wrote is not written a
/// second time. `on` is the prompt that leaves the thought to the model, which wrote none of it.
pub(crate) fn a_forced_tag_after_a_prompt_that_wrote_the_templates_text(
    format: &Format,
    tools: &[Tool],
    prompt: &str,
    on: &str,
    marker: &str,
) {
    let name = format.name();
    assert!(
        format.thought_closed(prompt),
        "{name}: the prompt wrote the template's text"
    );
    assert!(
        !format.reasoning_open(prompt),
        "{name}: the prompt closed the thought"
    );
    assert!(
        !format.thought_closed(on),
        "{name}: the thinking-on prompt wrote nothing"
    );
    for start in [
        GrammarStart::Unknown,
        GrammarStart::AfterReasoning,
        GrammarStart::FirstToken,
    ] {
        let tag = format
            .grammar_for(
                tools,
                Options {
                    at_least_one: true,
                    reasoning_open: format.reasoning_open(prompt),
                    thought_closed: format.thought_closed(prompt),
                    one_call: false,
                    start,
                },
            )
            .unwrap_or_else(|| panic!("{name}: no tag"))
            .payload();
        let calls = &tag["format"];
        assert_eq!(
            calls["type"], "triggered_tags",
            "{name}: {start:?}: {calls}"
        );
        for tag in calls["tags"].as_array().expect("tags") {
            let begin = tag["begin"].as_str().expect("a begin");
            assert!(
                begin.starts_with(marker),
                "{name}: {start:?}: {begin:?} does not begin with {marker:?}"
            );
        }
    }
}

/// A function tool named `name` with `parameters` as its schema.
pub(crate) fn tool(name: &str, parameters: Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    }
}

/// The grammar tests' pair of tools: `get_weather` with a city, days and
/// units, and `ping` without parameters.
pub(crate) fn weather_tools() -> Vec<Tool> {
    vec![
        tool(
            "get_weather",
            value!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city", "days"],
            }),
        ),
        tool("ping", value!({"type": "object"})),
    ]
}
