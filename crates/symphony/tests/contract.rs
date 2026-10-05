//! The contract every Symphony parser keeps, checked over a corpus of outputs and every way of
//! chunking them.
//!
//! Four properties hold today and are checked here for the Qwen3 format: conservation (every byte
//! of the output lands in exactly one event, in order); chunking invariance (what the parser says
//! about an output does not depend on how the output was cut into deltas); prefix stability (what
//! the parser has said so far is never taken back: content, reasoning, dropped and malformed text
//! only grow, the calls so far are a prefix of the final calls, and a call's arguments so far are a
//! prefix of its final arguments); and order (calls are numbered from zero in the order they start,
//! with ids that follow the index). The fifth property of the design, token identity, comes with
//! token attribution. Each new format adds its outputs to the corpus below.

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

    /// Whether `self` is what `later` said at an earlier point: nothing taken back since.
    fn is_prefix_of(&self, later: &Self) -> bool {
        later.content.starts_with(&self.content)
            && later.reasoning.starts_with(&self.reasoning)
            && later.dropped.starts_with(&self.dropped)
            && later.malformed.starts_with(&self.malformed)
            && later.calls.starts_with(&self.calls)
            && self.arguments.iter().all(|(index, so_far)| {
                later
                    .arguments
                    .iter()
                    .any(|(i, final_args)| i == index && final_args.starts_with(so_far))
            })
            && later.ended.starts_with(&self.ended)
            && (!self.finished || later.finished)
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

/// The cut offsets to replay `text` with: whole, every two-way split, byte by byte, and thirty seeded
/// plans of one to eight characters; every cut on a character boundary.
fn chunkings(text: &str) -> Vec<Vec<usize>> {
    let chars: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut plans = vec![vec![]];
    plans.extend(chars.iter().skip(1).map(|&cut| vec![cut]));
    plans.push(chars.iter().skip(1).copied().collect());
    for seed in 1..=30u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cuts = Vec::new();
        let mut at = 0;
        while at < chars.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            at += 1 + (state >> 33) as usize % 8;
            if at < chars.len() {
                cuts.push(chars[at]);
            }
        }
        plans.push(cuts);
    }
    plans
}

/// Replay `text` cut at `cuts`: the final events, and what was said after each delta.
fn replay(text: &str, cuts: &[usize]) -> Result<(Vec<Event>, Vec<Said>), ParseError> {
    let mut parser = Qwen3::new();
    let mut out = Events::new();
    let mut said_after_each = Vec::new();
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
            said_after_each.push(Said::of(out.as_slice()));
            from = cut;
        }
    }
    parser.feed(
        Input::End {
            finish: EngineFinish::Stop,
        },
        &mut out,
    )?;
    Ok((out.drain(), said_after_each))
}

#[test]
fn every_byte_of_every_output_lands_in_exactly_one_event_in_order() {
    for text in OUTPUTS {
        for cuts in chunkings(text) {
            let (events, _) = replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
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
        let (whole, _) = replay(text, &[]).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        let expected = Said::of(&whole);
        for cuts in chunkings(text) {
            let (events, _) = replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(Said::of(&events), expected, "{text:?} cut at {cuts:?}");
        }
    }
}

#[test]
fn nothing_said_is_ever_taken_back() {
    for text in OUTPUTS {
        for cuts in chunkings(text) {
            let (events, said_after_each) =
                replay(text, &cuts).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            let final_said = Said::of(&events);
            let mut previous = Said::default();
            for (step, said) in said_after_each.iter().enumerate() {
                assert!(
                    previous.is_prefix_of(said),
                    "{text:?} cut at {cuts:?}: step {step} took back what step {} said:\n  before {previous:?}\n  after  {said:?}",
                    step.saturating_sub(1)
                );
                assert!(
                    said.is_prefix_of(&final_said),
                    "{text:?} cut at {cuts:?}: step {step} said what the end did not keep:\n  step {said:?}\n  final {final_said:?}"
                );
                previous = said.clone();
            }
        }
    }
}

#[test]
fn calls_are_numbered_from_zero_in_order_with_ids_that_follow_the_index() {
    for text in OUTPUTS {
        let (events, _) = replay(text, &[]).unwrap_or_else(|e| panic!("{text:?}: {e}"));
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
