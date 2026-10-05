//! The contract every Symphony parser keeps, checked over a corpus of outputs and every way of
//! chunking them.
//!
//! Four properties hold today and are checked here for the Qwen3 format: conservation (every byte
//! of the output lands in exactly one event, in order, with `Finish` last); chunking invariance
//! (what the parser says about an output does not depend on how the output was cut into deltas);
//! well-formedness of the event stream (arguments and an end only for a call that has started and
//! not ended, one start per index, reasoning text only inside an open reasoning region, exactly one
//! `Finish`); and order (calls are numbered from zero as they start, with ids that follow the
//! index, every started call ended once, the finish counting them). The design's "committed stays
//! committed" is not a separate check: `Events` is append-only and conservation forbids saying a
//! byte twice, so nothing a parser pushed can be taken back through the event list; what a client
//! relies on beyond that is the well-formedness checked here. The fifth property, token identity,
//! comes with token attribution. Each new format adds its outputs to the corpus below.

mod common;

use common::chunkings;
use symphony::{EngineFinish, Event, Events, Input, ParseError, Parser, Qwen3};

const OUTPUTS: &[&str] = &[
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
];

/// What the parser has said, as the streams a client would assemble.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Said {
    content: String,
    reasoning: String,
    dropped: String,
    malformed: String,
    calls: Vec<(u32, String, String)>,
    arguments: Vec<(u32, String)>,
    ended: Vec<u32>,
    finished: bool,
}

impl Said {
    fn of(events: &[Event]) -> Self {
        let mut said = Self::default();
        for event in events {
            match event {
                Event::Content(t) => said.content.push_str(&t.text),
                Event::Reasoning(t) => said.reasoning.push_str(&t.text),
                Event::Dropped { text, .. } => said.dropped.push_str(&text.text),
                Event::Malformed { text, .. } => said.malformed.push_str(&text.text),
                Event::ToolCallStart {
                    index, id, name, ..
                } => {
                    said.calls.push((*index, id.clone(), name.clone()));
                    said.arguments.push((*index, String::new()));
                }
                Event::ToolCallArguments { index, json, .. } => {
                    match said.arguments.iter_mut().find(|(i, _)| i == index) {
                        Some((_, so_far)) => so_far.push_str(json),
                        None => said.arguments.push((*index, json.clone())),
                    }
                }
                Event::ToolCallEnd { index, .. } => said.ended.push(*index),
                Event::Finish { .. } => said.finished = true,
                Event::ReasoningStart | Event::ReasoningEnd => {}
            }
        }
        said
    }
}

fn bytes_of(event: &Event) -> &str {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.as_str(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// Replay `text` cut at `cuts` and return the events.
fn replay(text: &str, cuts: &[usize]) -> Result<Vec<Event>, ParseError> {
    let mut parser = Qwen3::new();
    let mut out = Events::new();
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            parser.feed(
                Input::Delta {
                    token_ids: &[],
                    text: &text[from..cut],
                    spans: &[],
                },
                &mut out,
            )?;
            from = cut;
        }
    }
    parser.feed(
        Input::End {
            finish: EngineFinish::Stop,
        },
        &mut out,
    )?;
    Ok(out.drain())
}

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(text: &str, cuts: &[usize], events: &[Event]) {
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    for (at, event) in events.iter().enumerate() {
        let place = || format!("{text:?} cut at {cuts:?}, event {at}: {event:?}");
        assert!(!finished, "{}: after Finish", place());
        match event {
            Event::ToolCallStart { index, .. } => {
                assert!(!started.contains(index), "{}: a second start", place());
                started.push(*index);
            }
            Event::ToolCallArguments { index, .. } | Event::ToolCallEnd { index, .. } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: after the call's end", place());
                if let Event::ToolCallEnd { .. } = event {
                    ended.push(*index);
                }
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
            Event::Finish { .. } => finished = true,
            Event::Content(_) | Event::Dropped { .. } | Event::Malformed { .. } => {}
        }
    }
    assert!(finished, "{text:?} cut at {cuts:?}: no Finish");
    assert!(
        !reasoning_open,
        "{text:?} cut at {cuts:?}: reasoning left open"
    );
    assert_eq!(
        started.len(),
        ended.len(),
        "{text:?} cut at {cuts:?}: a call without an end"
    );
}

#[test]
fn every_byte_of_every_output_lands_in_exactly_one_event_in_order() {
    for text in OUTPUTS {
        for cuts in chunkings(text) {
            let events = replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, *text, "cuts {cuts:?}");
            assert!(
                matches!(events.last(), Some(Event::Finish { .. })),
                "{text:?}: Finish is last"
            );
        }
    }
}

#[test]
fn what_the_parser_says_does_not_depend_on_the_chunking() {
    for text in OUTPUTS {
        let whole = replay(text, &[]).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        let expected = Said::of(&whole);
        for cuts in chunkings(text) {
            let events = replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(Said::of(&events), expected, "{text:?} cut at {cuts:?}");
        }
    }
}

#[test]
fn the_event_stream_is_well_formed_under_every_chunking() {
    for text in OUTPUTS {
        for cuts in chunkings(text) {
            let events = replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            check_well_formed(text, &cuts, &events);
        }
    }
}

#[test]
fn calls_are_numbered_from_zero_in_order_with_ids_that_follow_the_index() {
    // The whole output is enough: chunking invariance above makes every other cut say the same.
    for text in OUTPUTS {
        let events = replay(text, &[]).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        let said = Said::of(&events);
        for (position, (index, id, _)) in said.calls.iter().enumerate() {
            assert_eq!(*index as usize, position, "{text:?}");
            assert_eq!(id, &format!("call_{index}"), "{text:?}");
        }
        let mut ended = said.ended.clone();
        ended.sort_unstable();
        assert_eq!(
            ended,
            said.calls
                .iter()
                .map(|(index, ..)| *index)
                .collect::<Vec<_>>(),
            "{text:?}: every call that starts ends exactly once"
        );
        if let Some(Event::Finish { tool_calls, .. }) = events.last() {
            assert_eq!(*tool_calls as usize, said.calls.len(), "{text:?}");
        }
    }
}

#[test]
fn the_lifecycle_is_one_prompt_then_deltas_then_one_end() {
    let mut parser = Qwen3::new();
    let mut out = Events::new();
    let delta = |text| Input::Delta {
        token_ids: &[],
        text,
        spans: &[],
    };
    parser
        .feed(delta("a"), &mut out)
        .expect("a delta first is fine");
    assert!(matches!(
        parser.feed(
            Input::Prompt {
                token_ids: &[],
                text: ""
            },
            &mut out
        ),
        Err(ParseError::Lifecycle(_))
    ));
    parser
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("end");
    assert!(matches!(
        parser.feed(delta("b"), &mut out),
        Err(ParseError::Lifecycle(_))
    ));
    assert!(matches!(
        parser.feed(
            Input::End {
                finish: EngineFinish::Stop
            },
            &mut out
        ),
        Err(ParseError::Lifecycle(_))
    ));
}
