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
//! the event list. The fifth property, token identity (every token counted once, in the event that
//! carries its first byte, whatever the cuts), is checked event by event with a synthetic
//! tokenization of each output. A format joins the contract with one entry in [`FORMATS`], its
//! constructor and its corpus.

mod common;

use common::{bytes_of, chunkings, delta, prompt};
use symphony::{
    json::scan, Choice, Constrained, DropReason, EngineFinish, Event, Events, FinishReason, Input,
    MalformedReason, ParseError, Parser, Qwen3, TokenSpan,
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

fn forced_function() -> Box<dyn Parser> {
    Box::new(Constrained::new(Choice::Function(
        "get_weather".to_string(),
    )))
}

fn required_call() -> Box<dyn Parser> {
    Box::new(Constrained::new(Choice::Required))
}

const FORMATS: &[Format] = &[
    Format {
        name: "qwen3",
        new: qwen3,
        outputs: QWEN3_OUTPUTS,
    },
    Format {
        name: "constrained, one function",
        new: forced_function,
        outputs: FUNCTION_OUTPUTS,
    },
    Format {
        name: "constrained, required",
        new: required_call,
        outputs: REQUIRED_OUTPUTS,
    },
];

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
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": nope}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": truex}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": [1, falsex, 2]}</tool_call>",
];

/// Outputs of a request that forced one function: the arguments of that call, as the grammar
/// shaped them, plus what a grammar never produces, since the parser must survive it anyway, junk
/// that begins like a JSON literal included.
const FUNCTION_OUTPUTS: &[&str] = &[
    "",
    "\n\n",
    "{}",
    "{\"city\": \"Paris\"}",
    "  {\"city\": \"Paris\"}\n",
    "{\"a\": [1, 2",
    "{\"a\": 1} and more",
    "@ not json at all",
    "\"a string\"",
    "{\"e\": \"\\ud83c\\udf0d\"}",
    "{\"nested\": {\"x\": [true, null, 1.5e3]}, \"計画\": \"🌍\"}",
    "{\"a\": 1, #}",
    "nope",
    "not json at all",
    "truex",
    "{\"a\": nope}",
];

/// Outputs of a request that required a call: a list of call objects, shaped and misshapen.
const REQUIRED_OUTPUTS: &[&str] = &[
    "",
    "[]",
    "[{\"name\": \"f\", \"parameters\": {}}]",
    "[{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}, \
     {\"name\": \"get_time\", \"parameters\": {\"zone\": \"CET\"}}]",
    " [ {\"name\": \"f\", \"parameters\": {\"a\": 1}} , {\"name\": \"g\", \"parameters\": {}} ] \n",
    "[{\"name\": \"f\", \"parameters\": {\"a\": [1,",
    "[{\"parameters\": {}, \"name\": \"f\"}]",
    "[{\"parameters\": {}}]",
    "{\"name\": \"f\"}",
    "[{\"name\": \"f\", \"parameters\": {}}] junk",
    "[1, 2]",
    "[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]",
    "[{\"name\": \"f\", \"parameters\": {\"e\": \"\\ud83c\\udf0d\"}}]",
    "[{\"name\": \"f\", \"parameters\": 12abc}]",
    "[{\"name\": \"f\", \"parameters\": truex}]",
    "[{\"name\": \"f\", \"parameters\": {\"a\": nope}}]",
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

/// Whether some JSON value could still continue `text`, by the strict acceptor.
fn valid_json_prefix(text: &str) -> bool {
    scan(text).valid == text.len()
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

/// A synthetic tokenization of `text`: token boundaries at character offsets, one to three
/// characters per token, deterministic in the text.
fn token_boundaries(text: &str) -> Vec<usize> {
    let chars: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut boundaries = vec![0];
    let mut at = 0;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    while at < chars.len() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        at += 1 + (state % 3) as usize;
        if at < chars.len() {
            boundaries.push(chars[at]);
        }
    }
    boundaries.push(text.len());
    boundaries.dedup();
    boundaries
}

/// Replay `text` through a new parser of `format` cut at `cuts`, after the prompt, each delta
/// carrying the spans of the synthetic tokens it holds, a token cut by a delta boundary continuing
/// into the next delta, and a byte-less token at the very end, as an end-of-turn token would be.
fn replay_counted(format: &Format, text: &str, cuts: &[usize]) -> Result<Vec<Event>, ParseError> {
    let boundaries = token_boundaries(text);
    let mut tokens: Vec<(usize, usize)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    tokens.push((text.len(), text.len()));
    let mut parser = (format.new)();
    let mut out = Events::new();
    parser.feed(prompt(), &mut out)?;
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            let spans: Vec<TokenSpan> = tokens
                .iter()
                .filter(|&&(start, end)| {
                    (end > from && start < cut)
                        || (start == end && start == cut && cut == text.len())
                })
                .map(|&(start, end)| TokenSpan {
                    token_id: 0,
                    start: start.max(from) - from,
                    end: end.min(cut) - from,
                    continued: start < from,
                })
                .collect();
            parser.feed(
                Input::Delta {
                    token_ids: &[],
                    text: &text[from..cut],
                    spans: &spans,
                },
                &mut out,
            )?;
            from = cut;
        }
    }
    if text.is_empty() {
        // No delta carried text, so the byte-less token arrives in an empty one.
        parser.feed(
            Input::Delta {
                token_ids: &[],
                text: "",
                spans: &[TokenSpan {
                    token_id: 0,
                    start: 0,
                    end: 0,
                    continued: false,
                }],
            },
            &mut out,
        )?;
    }
    parser.feed(
        Input::End {
            finish: EngineFinish::Stop,
        },
        &mut out,
    )?;
    Ok(out.drain())
}

fn tokens_of(event: &Event) -> Option<u32> {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.tokens,
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.tokens,
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.tokens,
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
    }
}

#[test]
fn every_token_is_counted_in_the_event_that_carries_its_first_byte_whatever_the_cuts() {
    for (format, text) in corpus() {
        let boundaries = token_boundaries(text);
        // Where each synthetic token starts, and the byte-less one at the end.
        let mut starts: Vec<usize> = boundaries[..boundaries.len() - 1].to_vec();
        starts.push(text.len());
        for cuts in chunkings(text) {
            let events = replay_counted(format, text, &cuts)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", format.name));
            let place = || format!("{}: {text:?} cut at {cuts:?}", format.name);
            let mut at = 0;
            let mut total = 0;
            for event in &events {
                if matches!(
                    event,
                    Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. }
                ) {
                    continue;
                }
                let tokens =
                    tokens_of(event).unwrap_or_else(|| panic!("{}: uncounted {event:?}", place()));
                let length = bytes_of(event).len();
                let expected = match event {
                    // The byte-less token at the end has no byte to follow it.
                    Event::Dropped {
                        why: DropReason::ControlToken,
                        ..
                    } if length == 0 => starts.iter().filter(|&&start| start >= text.len()).count(),
                    _ => starts
                        .iter()
                        .filter(|&&start| start >= at && start < at + length)
                        .count(),
                };
                assert_eq!(
                    tokens as usize,
                    expected,
                    "{}: {event:?} at byte {at}",
                    place()
                );
                at += length;
                total += tokens as usize;
            }
            assert_eq!(total, starts.len(), "{}: every token counted once", place());
            let reasoning: u32 = events
                .iter()
                .filter_map(|e| match e {
                    Event::Reasoning(t) => t.tokens,
                    _ => None,
                })
                .sum();
            let Some(Event::Finish {
                reasoning_tokens, ..
            }) = events.last()
            else {
                panic!("{}: Finish is last", place());
            };
            assert_eq!(
                *reasoning_tokens,
                reasoning,
                "{}: Finish counts the reasoning tokens",
                place()
            );
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
