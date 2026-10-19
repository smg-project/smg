//! Test support the crate's modules share: readers of an event list, and a
//! driver that feeds a parser one turn.

use crate::{
    event::{Event, Events},
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
