//! Test support the crate's modules share: readers of an event list, and a
//! driver that feeds a parser one turn.

use openai_protocol::common::{Function, Tool};
use serde_json::{json as value, Value};

use crate::{
    adapt,
    event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
    format::Format,
    grammar::{GrammarStart, Options},
    input::{EngineFinish, Input},
    parser::Parser,
    Declared,
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
            Event::ToolCallArguments { index, json, .. } if *index == call => Some(json.as_ref()),
            _ => None,
        })
        .collect()
}

/// The argument fragments of every call, joined in order.
pub(crate) fn arguments(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCallArguments { json, .. } => Some(json.as_ref()),
            _ => None,
        })
        .collect()
}

/// The malformed reports, each text with its reason, in order.
pub(crate) fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
            _ => None,
        })
        .collect()
}

/// The text of every malformed report, one entry per event, whatever its reason (a report of no
/// bytes counts here, where the joined texts would not show it); the reasons are [`malformed`]'s.
pub(crate) fn malformed_texts<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Malformed { text, .. } => Some(&*text.text),
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

/// The calls as (index, name, arguments), the fragments of each joined onto the
/// last call started; an arguments event before any call is left out.
pub(crate) fn indexed_calls(events: &[Event]) -> Vec<(u32, String, String)> {
    let mut calls: Vec<(u32, String, String)> = Vec::new();
    for event in events {
        match event {
            Event::ToolCallStart { index, name, .. } => {
                calls.push((*index, name.clone(), String::new()));
            }
            Event::ToolCallArguments { json, .. } => {
                if let Some(last) = calls.last_mut() {
                    last.2.push_str(json);
                }
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

/// Every shape of an invoke a model writes outside a block on a table whose `run` reads an output
/// after a prompt, read whole, one character at a time and at every two-way cut as the serving
/// engines' own parsers read them: `invoke` alone is the call; bare and then again inside a block
/// (`open` to `close`), two calls and the model's text; prose after a bare call is text between
/// calls, shown to a chat client as content; an invoke inside the thought stays there; after a
/// preamble the blank line before a bare invoke is the text's; two bare invokes are two calls.
/// `chat` is the prompt that closed the thought, `thinking` the one that left it open; `invoke`
/// calls `add_integers` with 6 and 8.
pub(crate) fn an_invoke_outside_a_block_reads_as_the_engines_do(
    run: impl Fn(&str, &[&str]) -> Vec<Event<'static>>,
    invoke: &str,
    open: &str,
    close: &str,
    chat: &str,
    thinking: &str,
) {
    let call = (
        "add_integers".to_string(),
        r#"{"a": 6, "b": 8}"#.to_string(),
    );
    let shapes: [(&str, String, &str, usize, &str); 6] = [
        ("the bare invoke alone", invoke.to_string(), chat, 1, ""),
        (
            "bare, then the block",
            format!("\n\n:tool\n{invoke}\n\n{open}\n{invoke}\n{close}"),
            chat,
            2,
            "\n\n:tool\n",
        ),
        (
            "bare, then prose",
            format!("{invoke}\nThe sum is 14."),
            chat,
            1,
            "",
        ),
        (
            "inside the thought",
            format!("Let me add.{invoke}</think>Done."),
            thinking,
            0,
            "Done.",
        ),
        (
            "after a preamble",
            format!("Let me check.\n\n{invoke}"),
            chat,
            1,
            "Let me check.\n\n",
        ),
        (
            "two bare invokes",
            format!("{invoke}\n{invoke}"),
            chat,
            2,
            "",
        ),
    ];
    for (name, output, prompt, count, content) in shapes {
        assert_eq!(bytes(&run(prompt, &[&output])), output, "{name}");
        let said = the_same_at_every_cut(|pieces| run(prompt, pieces), &output);
        assert_eq!(said.content, content, "{name}");
        assert_eq!(said.calls.len(), count, "{name}: {:?}", said.calls);
        assert!(
            said.calls.iter().all(|c| *c == call),
            "{name}: {:?}",
            said.calls
        );
        assert_eq!(
            said.finish.map(|(_, calls)| calls),
            Some(count as u32),
            "{name}"
        );
    }
    // The bare invoke alone: nothing malformed, and a chat client reads the call alone.
    let events = run(chat, &[invoke]);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Malformed { .. })),
        "{events:?}"
    );
    let choice = adapt::chat::message(0, &events);
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(choice.message.content, None);
    // Prose after a bare call is text between calls, shown to a chat client as content.
    let events = run(chat, &[&format!("{invoke}\nThe sum is 14.")]);
    let between: String = malformed(&events)
        .into_iter()
        .filter(|(_, why)| *why == MalformedReason::TextBetweenCalls)
        .map(|(text, _)| text)
        .collect();
    assert_eq!(between, "The sum is 14.");
    assert_eq!(
        adapt::chat::message(0, &events).message.content.as_deref(),
        Some("The sum is 14.")
    );
    // Inside the thought, the invoke's bytes are the thought's.
    let output = format!("Let me add.{invoke}</think>Done.");
    let said = the_same_at_every_cut(|pieces| run(thinking, pieces), &output);
    assert_eq!(said.reasoning, format!("Let me add.{invoke}"));
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
            parameters: Some(parameters),
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

/// Wrapper text the engine dropped: uncounted, dropped as a wrapper, the
/// event the formats' tests expect around a call.
pub(crate) fn dropped(text: &str) -> Event<'static> {
    Event::Dropped {
        text: Text::uncounted(text.to_owned()),
        why: DropReason::Wrapper,
    }
}

/// One declared tool, `get_weather` with a city and days: the fixture the
/// engine's tests and several formats' parse against.
pub(crate) fn weather_declared() -> Declared {
    Declared::of(&[tool(
        "get_weather",
        value!({"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        }}),
    )])
}

/// The events a test writes by hand: the tool-call events with the id and the
/// empty source the adapters do not read, and the engine's finish with no
/// reasoning tokens.
pub(crate) mod events {
    use crate::event::{Event, FinishReason, Text};

    /// A call's start, `call_<index>` as its id.
    pub(crate) fn call(index: u32, name: &str) -> Event<'static> {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.into(),
            source: Text::default(),
        }
    }

    /// A fragment of a call's arguments.
    pub(crate) fn arguments(index: u32, json: &str) -> Event<'static> {
        Event::ToolCallArguments {
            index,
            json: json.to_owned().into(),
            source: Text::default(),
        }
    }

    /// A call's end.
    pub(crate) fn end(index: u32) -> Event<'static> {
        Event::ToolCallEnd {
            index,
            source: Text::default(),
        }
    }

    /// The engine's finish with the calls counted and no reasoning tokens.
    pub(crate) fn finish(reason: FinishReason, tool_calls: u32) -> Event<'static> {
        Event::Finish {
            reason,
            tool_calls,
            reasoning_tokens: 0,
        }
    }
}
