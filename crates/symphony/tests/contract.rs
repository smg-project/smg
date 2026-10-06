//! The contract every Symphony parser keeps, checked for every format over a corpus of its outputs
//! and every way of chunking them.
//!
//! Checked for each format in [`FORMATS`]:
//!
//! - conservation: every byte of the output lands in exactly one event, in order, with `Finish`
//!   last;
//! - chunking invariance: what the parser says (content, each reasoning region, dropped and
//!   malformed text by reason, the calls with their arguments, and the finish with its reason and
//!   counts) does not depend on how the output was cut into deltas, nor on empty deltas between
//!   them;
//! - well-formedness: arguments and an end only for a call that has started and not ended, one
//!   start per index, reasoning text only inside an open region, exactly one `Finish`, counting the
//!   calls that started, and after every fragment each call's arguments so far a valid JSON prefix,
//!   which is what prefix stability promises a client;
//! - order: calls are numbered from zero as they start, with ids that follow the index;
//! - the lifecycle: one `Prompt`, deltas, one `End`, and an input out of order is a `Lifecycle`
//!   error that emits nothing.
//!
//! The design's "committed stays committed" is not a separate check: `Events` is append-only and
//! conservation forbids saying a byte twice, so nothing a parser pushed can be taken back through
//! the event list. The fifth property, token identity, comes with token attribution. A format joins
//! the contract with one entry in [`FORMATS`], its constructor and its corpus.

mod common;

use common::{bytes_of, chunkings, delta, prompt};
use symphony::{
    json::PartialJson, DropReason, EngineFinish, Event, Events, FinishReason, Input,
    MalformedReason, ParseError, Parser, Qwen3,
};

/// A format under test: how to make its parser, and the outputs it is checked over.
struct Format {
    name: &'static str,
    new: fn() -> Box<dyn Parser>,
    outputs: &'static [&'static str],
}

fn qwen3() -> Box<dyn Parser> {
    Box::new(Qwen3::new())
}

const FORMATS: &[Format] = &[Format {
    name: "qwen3",
    new: qwen3,
    outputs: QWEN3_OUTPUTS,
}];

const QWEN3_OUTPUTS: &[&str] = &[
    "",
    "Hello!",
    "<think>\n\n</think>\n\nHello!",
    "<think>\nplan\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\nAnd again.\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>\nDone.",
    "<tool_call>\n{\"name\": \"save_note\", \"arguments\": {\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\"}}\n</tool_call>",
    "a</think>b</tool_call>c",
    "<think>still thinking",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": [1, 2",
    "<tool_call>junk</tool_call><tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>",
    "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>\n```",
    "<tool_call>{\"arguments\": {\"a\": 1}, \"name\": \"f\"}</tool_call>",
    "<tool_x <thinking> </thinking> <</think>> <tool_call",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"e\": \"\\ud83c\\udf0d\"}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": 12abc}</tool_call>",
    "<think>I could write </think> here but it is text.\n</think>\n\nParis is sunny.",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n x\n</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {}}x y</tool_call>",
];

/// Every format with each of its outputs.
fn corpus() -> impl Iterator<Item = (&'static Format, &'static str)> {
    FORMATS
        .iter()
        .flat_map(|format| format.outputs.iter().map(move |text| (format, *text)))
}

/// What the parser has said, as the streams a client would assemble.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Said {
    content: String,
    /// The text of each reasoning region, in order.
    reasoning: Vec<String>,
    /// Dropped text by reason, each reason's text joined in order.
    dropped: Vec<(DropReason, String)>,
    /// Malformed text by reason, each reason's text joined in order.
    malformed: Vec<(MalformedReason, String)>,
    calls: Vec<(u32, String, String)>,
    arguments: Vec<(u32, String)>,
    ended: Vec<u32>,
    /// The finish's reason, call count and reasoning token count.
    finish: Option<(FinishReason, u32, u32)>,
}

impl Said {
    fn of(events: &[Event]) -> Self {
        let mut said = Self::default();
        let mut region_open = false;
        for event in events {
            match event {
                Event::Content(t) => said.content.push_str(&t.text),
                Event::ReasoningStart => {
                    said.reasoning.push(String::new());
                    region_open = true;
                }
                Event::Reasoning(t) => {
                    if !region_open {
                        said.reasoning.push(String::new());
                        region_open = true;
                    }
                    if let Some(region) = said.reasoning.last_mut() {
                        region.push_str(&t.text);
                    }
                }
                Event::ReasoningEnd => region_open = false,
                Event::Dropped { text, why } => joined_by(&mut said.dropped, why, &text.text),
                Event::Malformed { text, why } => joined_by(&mut said.malformed, why, &text.text),
                Event::ToolCallStart {
                    index, id, name, ..
                } => {
                    said.calls.push((*index, id.clone(), name.clone()));
                    said.arguments.push((*index, String::new()));
                }
                Event::ToolCallArguments { index, json, .. } => {
                    if let Some((_, so_far)) = said.arguments.iter_mut().find(|(i, _)| i == index) {
                        so_far.push_str(json);
                    }
                }
                Event::ToolCallEnd { index, .. } => said.ended.push(*index),
                Event::Finish {
                    reason,
                    tool_calls,
                    reasoning_tokens,
                } => said.finish = Some((reason.clone(), *tool_calls, *reasoning_tokens)),
            }
        }
        said
    }
}

/// Append `text` to the entry for `why`, adding the entry when it is the reason's first text.
fn joined_by<R: Clone + PartialEq>(list: &mut Vec<(R, String)>, why: &R, text: &str) {
    match list.iter_mut().find(|(reason, _)| reason == why) {
        Some((_, joined)) => joined.push_str(text),
        None => list.push((why.clone(), text.to_string())),
    }
}

fn end() -> Input<'static> {
    Input::End {
        finish: EngineFinish::Stop,
    }
}

/// Replay `text` through a new parser of `format`, cut at `cuts`, ending with `stop`.
fn replay(
    format: &Format,
    text: &str,
    cuts: &[usize],
    empty_between: bool,
) -> Result<Vec<Event>, ParseError> {
    common::replay(
        &mut *(format.new)(),
        text,
        cuts,
        &EngineFinish::Stop,
        empty_between,
    )
}

/// Whether `text` is a JSON prefix the prefix parser takes whole.
fn valid_json_prefix(text: &str) -> bool {
    matches!(PartialJson::default().parse(text, true), Ok((_, consumed)) if consumed == text.len())
}

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(name: &str, text: &str, cuts: &[usize], events: &[Event]) {
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    for (at, event) in events.iter().enumerate() {
        let place = || format!("{name}: {text:?} cut at {cuts:?}, event {at}: {event:?}");
        assert!(!finished, "{}: after Finish", place());
        match event {
            Event::ToolCallStart { index, .. } => {
                assert!(!started.contains(index), "{}: a second start", place());
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments { index, json, .. } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: after the call's end", place());
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    so_far.push_str(json);
                    assert!(
                        valid_json_prefix(so_far),
                        "{}: the arguments so far, {so_far:?}, are no valid JSON prefix",
                        place()
                    );
                }
            }
            Event::ToolCallEnd { index, .. } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: a second end", place());
                ended.push(*index);
            }
            Event::ReasoningStart => {
                assert!(!reasoning_open, "{}: reasoning already open", place());
                reasoning_open = true;
            }
            Event::Reasoning(_) => {
                assert!(reasoning_open, "{}: reasoning outside a region", place());
            }
            Event::ReasoningEnd => {
                assert!(reasoning_open, "{}: no reasoning open", place());
                reasoning_open = false;
            }
            Event::Finish { tool_calls, .. } => {
                assert_eq!(
                    *tool_calls as usize,
                    started.len(),
                    "{}: the finish counts the calls that started",
                    place()
                );
                finished = true;
            }
            Event::Content(_) | Event::Dropped { .. } | Event::Malformed { .. } => {}
        }
    }
    let place = format!("{name}: {text:?} cut at {cuts:?}");
    assert!(finished, "{place}: no Finish");
    assert!(!reasoning_open, "{place}: reasoning left open");
    assert_eq!(started.len(), ended.len(), "{place}: a call without an end");
}

#[test]
fn every_byte_of_every_output_lands_in_exactly_one_event_in_order() {
    for (format, text) in corpus() {
        for cuts in chunkings(text) {
            let events = replay(format, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, *text, "{}: cuts {cuts:?}", format.name);
            assert!(
                matches!(events.last(), Some(Event::Finish { .. })),
                "{}: {text:?}: Finish is last",
                format.name
            );
        }
    }
}

#[test]
fn what_the_parser_says_does_not_depend_on_the_chunking() {
    for (format, text) in corpus() {
        let whole = replay(format, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
        let expected = Said::of(&whole);
        for cuts in chunkings(text) {
            let events = replay(format, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
            assert_eq!(
                Said::of(&events),
                expected,
                "{}: {text:?} cut at {cuts:?}",
                format.name
            );
        }
    }
}

#[test]
fn empty_deltas_between_the_pieces_change_nothing() {
    for (format, text) in corpus() {
        let whole = replay(format, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
        let per_char: Vec<usize> = text.char_indices().map(|(i, _)| i).skip(1).collect();
        for cuts in [Vec::new(), per_char] {
            let events = replay(format, text, &cuts, true)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, *text, "{}: cuts {cuts:?}", format.name);
            assert_eq!(
                Said::of(&events),
                Said::of(&whole),
                "{}: {text:?} cut at {cuts:?}",
                format.name
            );
        }
    }
}

#[test]
fn the_event_stream_is_well_formed_under_every_chunking() {
    for (format, text) in corpus() {
        for cuts in chunkings(text) {
            let events = replay(format, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
            check_well_formed(format.name, text, &cuts, &events);
        }
    }
}

#[test]
fn calls_are_numbered_from_zero_in_order_with_ids_that_follow_the_index() {
    // The whole output is enough: chunking invariance above makes every other cut say the same.
    for (format, text) in corpus() {
        let events = replay(format, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
        for (position, (index, id, _)) in Said::of(&events).calls.iter().enumerate() {
            assert_eq!(*index as usize, position, "{}: {text:?}", format.name);
            assert_eq!(id, &format!("call_{index}"), "{}: {text:?}", format.name);
        }
    }
}

/// Feed `input`, which is out of order, and check that it is a `Lifecycle` error and emits nothing.
fn rejected(parser: &mut dyn Parser, input: Input<'_>, out: &mut Events, what: &str) {
    let before = out.len();
    assert!(
        matches!(parser.feed(input, out), Err(ParseError::Lifecycle(_))),
        "{what}: a lifecycle error"
    );
    assert_eq!(out.len(), before, "{what}: nothing emitted");
}

#[test]
fn the_lifecycle_is_one_prompt_then_deltas_then_one_end() {
    for format in FORMATS {
        let mut parser = (format.new)();
        let mut out = Events::new();
        parser.feed(prompt(), &mut out).expect("a prompt first");
        rejected(&mut *parser, prompt(), &mut out, "a second prompt");
        parser.feed(delta("a"), &mut out).expect("a delta");
        rejected(
            &mut *parser,
            prompt(),
            &mut out,
            "a prompt after output began",
        );
        parser.feed(end(), &mut out).expect("the end");
        rejected(&mut *parser, delta("b"), &mut out, "a delta after the end");
        rejected(&mut *parser, end(), &mut out, "a second end");

        let mut without_prompt = (format.new)();
        let mut out = Events::new();
        without_prompt
            .feed(delta("a"), &mut out)
            .expect("a delta first is fine: the prompt is optional");
    }
}
