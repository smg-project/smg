//! Qwen3: reasoning between `<think>` and `</think>`, each tool call between `<tool_call>` and
//! `</tool_call>`, everything else content. Inside the call markers the family writes one of two
//! syntaxes ([`CallSyntax`]): Qwen3 a JSON object, `{"name": …, "arguments": {…}}`; Qwen 3.5
//! and later and Qwen3-Coder the tags `<function=NAME>` and `<parameter=KEY>` around each value's
//! text, which the request's tools type, an object or a list written as JSON.
//!
//! The definition is the table below; the [`Engine`](crate::Engine) runs it, and decides
//! everything the module doc of [`engine`](crate::engine) lists. Two things are Qwen3's own:
//!
//! - `calls + call_open = calls`: a new `<tool_call>` before the previous block closed ends the
//!   call that was open and starts the next.
//! - No row leaves reasoning on a call marker: Qwen3 closes its thought with `</think>` before a
//!   call, so a `<tool_call>` inside the thought is reasoning text.
//! - The turn opener is ChatML's `<|im_start|>assistant`.
//!
//! The tests here are the engine's as much as this format's: they were written against the
//! hand-written parser this table replaced, and hold the table to the same events.

use crate::format::{CallSyntax, Emits, Format};
#[cfg(test)]
use crate::tagged::Spelling;

/// The Qwen3 table, with the call syntax the checkpoint writes.
pub fn qwen3(syntax: CallSyntax) -> Format {
    Format::new("qwen3")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .transition("calls", "call_open", "calls")
        .calls(syntax)
        .opens_turn("<|im_start|>assistant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{Engine, BLOCK_WITHOUT_A_COMPLETE_CALL, TEXT_AFTER_THE_OBJECT},
        event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
        input::{EngineFinish, Input, TokenSpan},
        parser::{ParseError, Parser},
        tagged::Declared,
    };

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn delta(text: &str) -> Input<'_> {
        Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        }
    }

    fn run(pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        for piece in pieces {
            parser.feed(delta(piece), &mut out).expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.drain()
    }

    /// Every byte of every event, in order: the proof that nothing was lost or invented.
    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    /// Events with adjacent text of one kind joined and argument fragments joined per call, so two
    /// chunkings compare by what they said.
    fn joined(events: Vec<Event>) -> Vec<Event> {
        let mut out: Vec<Event> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (Some(Event::Content(a)), Event::Content(b))
                | (Some(Event::Reasoning(a)), Event::Reasoning(b)) => a.text.push_str(&b.text),
                (
                    Some(Event::Dropped { text: a, why: wa }),
                    Event::Dropped { text: b, why: wb },
                ) if *wa == wb => a.text.push_str(&b.text),
                (
                    Some(Event::ToolCallArguments {
                        index: i,
                        json,
                        source,
                    }),
                    Event::ToolCallArguments {
                        index: j,
                        json: more,
                        source: more_source,
                    },
                ) if *i == j => {
                    json.push_str(&more);
                    source.text.push_str(&more_source.text);
                }
                (_, event) => out.push(event),
            }
        }
        out
    }

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn thinking_then_content_in_the_fixtures_shape() {
        let output = "<think>\n\n</think>\n\nHello!";
        let events = run(&[output], EngineFinish::Stop);
        assert_eq!(
            events,
            vec![
                dropped("<think>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("\n\n")),
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("\n\nHello!")),
                Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 0,
                    reasoning_tokens: 0,
                },
            ]
        );
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn a_call_after_thinking_streams_the_models_argument_bytes() {
        let output = format!("<think>\nThe user asks.\n</think>\n\n{CALL}");
        let events = run(&[&output], EngineFinish::Stop);
        assert_eq!(
            events,
            vec![
                dropped("<think>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("\nThe user asks.\n")),
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("\n\n")),
                dropped("<tool_call>"),
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "get_weather".into(),
                    source: Text::uncounted("\n{\"name\": \"get_weather\", \"arguments\": "),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "{\"city\": \"Paris\"}".into(),
                    source: Text::uncounted("{\"city\": \"Paris\"}"),
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("}"),
                },
                dropped("\n"),
                dropped("</tool_call>"),
                Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ]
        );
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn prose_before_a_call_in_the_same_chunk_is_content() {
        let output = format!("Let me check.\n{CALL}");
        let events = run(&[&output], EngineFinish::Stop);
        assert_eq!(
            events[0],
            Event::Content(Text::uncounted("Let me check.\n"))
        );
        assert!(matches!(events[2], Event::ToolCallStart { .. }));
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn two_calls_in_one_chunk_are_both_emitted_with_their_own_index_and_id() {
        let second = CALL
            .replace("get_weather", "get_time")
            .replace("Paris", "CET");
        let output = format!("{CALL}\n{second}");
        let events = run(&[&output], EngineFinish::Stop);
        let starts: Vec<(u32, String, String)> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallStart {
                    index, id, name, ..
                } => Some((*index, id.clone(), name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![
                (0, "call_0".into(), "get_weather".into()),
                (1, "call_1".into(), "get_time".into()),
            ]
        );
        assert_eq!(
            events.last(),
            Some(&Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 2,
                reasoning_tokens: 0,
            })
        );
        assert!(
            events.contains(&Event::Content(Text::uncounted("\n"))),
            "the newline between the calls is content"
        );
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let output = format!("<think>\nplan\n</think>\n\nSure.\n{CALL}\nDone.");
        let whole = joined(run(&[&output], EngineFinish::Stop));
        let mut chunkings: Vec<Vec<&str>> = (1..output.len())
            .map(|cut| vec![&output[..cut], &output[cut..]])
            .collect();
        chunkings.push(
            output
                .char_indices()
                .map(|(i, c)| &output[i..i + c.len_utf8()])
                .collect(),
        );
        for pieces in chunkings {
            let events = run(&pieces, EngineFinish::Stop);
            assert_eq!(bytes(&events), output, "{pieces:?}");
            assert_eq!(joined(events), whole, "{pieces:?}");
        }
    }

    #[test]
    fn a_closing_marker_with_nothing_open_is_content_and_a_fenced_marker_is_a_marker() {
        let events = run(&["a</think>b"], EngineFinish::Stop);
        assert_eq!(events[0], Event::Content(Text::uncounted("a</think>b")));
        let fenced = format!("The syntax is:\n```\n{CALL}\n```");
        let events = run(&[&fenced], EngineFinish::Stop);
        assert!(
            events.iter().any(|e| matches!(e, Event::ToolCallStart { .. })),
            "a marker inside a code fence is a marker to this layer (bellwether #16 judges the policy)"
        );
        assert_eq!(bytes(&events), fenced);
    }

    #[test]
    fn a_call_that_never_closes_is_finished_at_the_end() {
        let events = run(
            &["<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1"],
            EngineFinish::Length,
        );
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallEnd { .. })));
        assert_eq!(
            events.last(),
            Some(&Event::Finish {
                reason: FinishReason::Length,
                tool_calls: 1,
                reasoning_tokens: 0,
            })
        );
        let events = run(&["<tool_call>\nnot json"], EngineFinish::Stop);
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        assert_eq!(bytes(&events), "<tool_call>\nnot json");
        assert_eq!(
            events.last(),
            Some(&Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                reasoning_tokens: 0,
            }),
            "a block that held no call is not counted"
        );
    }

    #[test]
    fn text_after_a_calls_object_is_classified_run_by_run_whatever_the_chunking() {
        let output = "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n x\n</tool_call>";
        let whole = run(&[output], EngineFinish::Stop);
        let surplus: Vec<&Event> = whole
            .iter()
            .filter(|e| matches!(e, Event::Dropped { .. } | Event::Malformed { .. }))
            .collect();
        assert_eq!(
            surplus,
            [
                &dropped("<tool_call>"),
                &dropped("\n "),
                &Event::Malformed {
                    text: Text::uncounted("x"),
                    why: MalformedReason::Other(TEXT_AFTER_THE_OBJECT.to_string()),
                },
                &dropped("\n"),
                &dropped("</tool_call>"),
            ]
        );
        let per_byte: Vec<String> = output.chars().map(String::from).collect();
        let per_byte: Vec<&str> = per_byte.iter().map(String::as_str).collect();
        let pieces = run(&per_byte, EngineFinish::Stop);
        let texts = |events: &[Event], dropped: bool| -> String {
            events
                .iter()
                .filter_map(|e| match e {
                    Event::Dropped { text, .. } if dropped => Some(text.text.as_str()),
                    Event::Malformed { text, .. } if !dropped => Some(text.text.as_str()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(texts(&pieces, true), texts(&whole, true));
        assert_eq!(texts(&pieces, false), texts(&whole, false));
    }

    #[test]
    fn a_cut_short_call_keeps_its_events_in_the_outputs_order() {
        let output = "<tool_call>\n{\"name\": \"f\", \"arguments\": {},</tool_call>";
        let events = run(&[output], EngineFinish::Stop);
        assert_eq!(
            events,
            vec![
                dropped("<tool_call>"),
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "f".into(),
                    source: Text::uncounted("\n{\"name\": \"f\", \"arguments\": "),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "{}".into(),
                    source: Text::uncounted("{}"),
                },
                Event::Malformed {
                    text: Text::uncounted(","),
                    why: MalformedReason::Other(BLOCK_WITHOUT_A_COMPLETE_CALL.to_string()),
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::default(),
                },
                dropped("</tool_call>"),
                Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ]
        );
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn a_block_that_held_no_call_takes_no_index_and_is_not_counted() {
        let output = format!("<tool_call>junk</tool_call>{CALL}");
        let events = run(&[&output], EngineFinish::Stop);
        assert_eq!(
            events[1],
            Event::Malformed {
                text: Text::uncounted("junk"),
                why: MalformedReason::Other(BLOCK_WITHOUT_A_COMPLETE_CALL.to_string()),
            },
            "the block closed at its marker, so the stream did not end inside it"
        );
        let starts: Vec<(u32, String)> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallStart { index, id, .. } => Some((*index, id.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![(0, "call_0".into())],
            "the real call takes the first index"
        );
        assert_eq!(
            events.last(),
            Some(&Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                reasoning_tokens: 0,
            })
        );
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn thinking_left_open_is_closed_at_the_end_and_the_engines_reason_is_kept() {
        let events = run(&["<think>still"], EngineFinish::Other("abort".into()));
        assert_eq!(
            events,
            vec![
                dropped("<think>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("still")),
                Event::ReasoningEnd,
                Event::Finish {
                    reason: FinishReason::Other("abort".into()),
                    tool_calls: 0,
                    reasoning_tokens: 0,
                },
            ]
        );
    }

    /// The spans of `tokens`, each a run of bytes, as one delta's spans.
    fn spans_of(tokens: &[&str]) -> Vec<TokenSpan> {
        let mut at = 0;
        tokens
            .iter()
            .map(|token| {
                let span = TokenSpan {
                    token_id: 0,
                    start: at,
                    end: at + token.len(),
                    continued: false,
                };
                at += token.len();
                span
            })
            .collect()
    }

    fn counted(events: &[Event]) -> Vec<(String, Option<u32>)> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => Some((t.text.clone(), t.tokens)),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => {
                    Some((text.text.clone(), text.tokens))
                }
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => Some((source.text.clone(), source.tokens)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_token_is_counted_once_in_the_event_that_carries_its_first_byte() {
        let tokens = [
            "<think>",
            "\nplan",
            "\n</think>",
            "\n\nHi",
            "<tool_call>",
            "{\"name\": \"f\", ",
            "\"arguments\": {}}",
            "</tool_call>",
        ];
        let text: String = tokens.concat();
        let spans = spans_of(&tokens);
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: &text,
                    spans: &spans,
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(
            counted(&events),
            vec![
                ("<think>".into(), Some(1)),
                ("\nplan\n".into(), Some(2)),
                ("</think>".into(), Some(0)),
                ("\n\nHi".into(), Some(1)),
                ("<tool_call>".into(), Some(1)),
                ("{\"name\": \"f\", \"arguments\": ".into(), Some(2)),
                ("{}".into(), Some(0)),
                ("}".into(), Some(0)),
                ("</tool_call>".into(), Some(1)),
            ]
        );
        assert_eq!(
            counted(&events)
                .iter()
                .map(|(_, n)| n.unwrap_or(0))
                .sum::<u32>(),
            tokens.len() as u32,
            "every token exactly once"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reasoning_tokens: 2,
                ..
            })
        ));
    }

    #[test]
    fn held_halves_of_a_character_count_as_reasoning_and_a_trailing_special_token_is_reported_once()
    {
        // The emoji is three tokens: two held halves with no bytes and a third carrying it.
        let pieces: [(&str, &[(usize, usize)]); 8] = [
            ("<think>", &[(0, 7)]),
            ("\n", &[(0, 1)]),
            ("", &[(0, 0)]),
            ("", &[(0, 0)]),
            ("🌍", &[(0, 4)]),
            ("\n", &[(0, 1)]),
            ("</think>", &[(0, 8)]),
            ("", &[(0, 0)]),
        ];
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        for (text, ranges) in pieces {
            let spans: Vec<TokenSpan> = ranges
                .iter()
                .map(|&(start, end)| TokenSpan {
                    token_id: 0,
                    start,
                    end,
                    continued: false,
                })
                .collect();
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text,
                        spans: &spans,
                    },
                    &mut out,
                )
                .expect("delta");
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(
            counted(&events),
            vec![
                ("<think>".into(), Some(1)),
                ("\n🌍\n".into(), Some(5)),
                ("</think>".into(), Some(1)),
                (String::new(), Some(1)),
            ],
            "the held halves count into the character they began; \
             the trailing token is reported at the end"
        );
        assert_eq!(
            events[events.len() - 2],
            Event::Dropped {
                text: Text::new("", 1),
                why: DropReason::ControlToken,
            }
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reasoning_tokens: 5,
                ..
            })
        ));
    }

    #[test]
    fn a_token_cut_by_a_delta_boundary_is_counted_once_and_a_special_token_counts_into_what_follows(
    ) {
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        let first = [TokenSpan {
            token_id: 1,
            start: 0,
            end: 4,
            continued: false,
        }];
        parser
            .feed(
                Input::Delta {
                    token_ids: &[1],
                    text: "<thi",
                    spans: &first,
                },
                &mut out,
            )
            .expect("delta");
        let second = [
            TokenSpan {
                token_id: 1,
                start: 0,
                end: 3,
                continued: true,
            },
            TokenSpan {
                token_id: 2,
                start: 3,
                end: 3,
                continued: false,
            },
            TokenSpan {
                token_id: 3,
                start: 3,
                end: 4,
                continued: false,
            },
        ];
        parser
            .feed(
                Input::Delta {
                    token_ids: &[1, 2, 3],
                    text: "nk>x",
                    spans: &second,
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(
            counted(&events),
            vec![("<think>".into(), Some(1)), ("x".into(), Some(2))],
            "the cut token once, where its first byte landed; \
             the byte-less token into the text after it"
        );
    }

    #[test]
    fn a_stream_that_turns_uncounted_reports_no_reasoning_count() {
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        let spans = [TokenSpan {
            token_id: 0,
            start: 0,
            end: 8,
            continued: false,
        }];
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "<think>a",
                    spans: &spans,
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(delta("b</think>"), &mut out)
            .expect("a delta without spans");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        assert!(matches!(
            out.as_slice().last(),
            Some(Event::Finish {
                reasoning_tokens: 0,
                ..
            })
        ));
    }

    #[test]
    fn deltas_without_spans_leave_the_text_uncounted() {
        let events = run(&["<think>a</think>b"], EngineFinish::Stop);
        assert!(counted(&events).iter().all(|(_, n)| n.is_none()));
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reasoning_tokens: 0,
                ..
            })
        ));
    }

    #[test]
    fn an_empty_output_is_only_its_finish() {
        assert_eq!(
            run(&[], EngineFinish::Stop),
            vec![Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                reasoning_tokens: 0,
            }]
        );
    }

    /// A tool `get_weather` with a string `city` and an integer `days`.
    fn declared() -> Declared {
        use openai_protocol::common::{Function, Tool};
        use serde_json::json as value;
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                }}),
                strict: None,
            },
        }])
    }

    const TAGGED_CALL: &str = concat!(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>",
        "\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    );

    fn run_tagged(pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Engine::new(qwen3(CallSyntax::Tagged(Spelling::Json)), declared());
        let mut out = Events::new();
        for piece in pieces {
            parser.feed(delta(piece), &mut out).expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.drain()
    }

    fn arguments_of(events: &[Event], call: u32) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_tagged_call_after_thinking_streams_its_string_and_writes_its_integer() {
        let output = format!("<think>\nThe user asks.\n</think>\n\n{TAGGED_CALL}");
        let events = run_tagged(&[&output], EngineFinish::Stop);
        assert_eq!(
            events[6..],
            [
                dropped("<tool_call>"),
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "get_weather".into(),
                    source: Text::uncounted("\n<function=get_weather>"),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "{\"city\": \"".into(),
                    source: Text::uncounted("\n<parameter=city>"),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "Paris".into(),
                    source: Text::uncounted("\nParis"),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "\"".into(),
                    source: Text::uncounted("\n</parameter>"),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: ", \"days\": 3".into(),
                    source: Text::uncounted("\n<parameter=days>\n3\n</parameter>"),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "}".into(),
                    source: Text::uncounted("\n"),
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("</function>"),
                },
                dropped("\n"),
                dropped("</tool_call>"),
                Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ]
        );
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);
        assert_eq!(bytes(&events), output);
    }

    #[test]
    fn two_tagged_calls_each_get_their_own_index_and_a_call_to_an_undeclared_tool_is_inferred() {
        let second = concat!(
            "<tool_call>\n<function=lookup>\n<parameter=id>\n42\n</parameter>",
            "\n</function>\n</tool_call>"
        );
        let output = format!("{TAGGED_CALL}\n{second}");
        let events = run_tagged(&[&output], EngineFinish::Stop);
        let starts: Vec<(u32, String, String)> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart {
                    index, id, name, ..
                } => Some((*index, id.clone(), name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![
                (0, "call_0".to_string(), "get_weather".to_string()),
                (1, "call_1".to_string(), "lookup".to_string()),
            ]
        );
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"id": 42}"#);
        assert_eq!(bytes(&events), output);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_tagged_block_its_marker_closes_early_ends_the_call_closed_and_a_cut_stream_leaves_it_open()
    {
        // The block's closing marker arrives before `</function>`: the string and the object close.
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar</tool_call>";
        let events = run_tagged(&[output], EngineFinish::Stop);
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Par"}"#);
        assert_eq!(bytes(&events), output);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "nothing was held when the marker came"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // The stream is cut instead: nothing closes, and the region is unterminated.
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
        let events = run_tagged(&[output], EngineFinish::Length);
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Par"#);
        assert_eq!(bytes(&events), output);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 0, source } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Length,
                tool_calls: 1,
                ..
            })
        ));
    }

    #[test]
    fn every_chunking_of_a_tagged_output_says_the_same_and_accounts_for_every_byte() {
        let output = format!("<think>\nplan\n</think>\n\nSure.\n{TAGGED_CALL}\nDone.");
        let whole = joined(run_tagged(&[&output], EngineFinish::Stop));
        let mut chunkings: Vec<Vec<&str>> = (1..output.len())
            .map(|cut| vec![&output[..cut], &output[cut..]])
            .collect();
        chunkings.push(
            output
                .char_indices()
                .map(|(i, c)| &output[i..i + c.len_utf8()])
                .collect(),
        );
        for pieces in chunkings {
            let events = run_tagged(&pieces, EngineFinish::Stop);
            assert_eq!(bytes(&events), output, "{pieces:?}");
            assert_eq!(joined(events), whole, "{pieces:?}");
        }
        assert_eq!(arguments_of(&whole, 0), r#"{"city": "Paris", "days": 3}"#);
    }

    /// The prompt, then the whole output in one delta, then the engine's stop.
    fn after_prompt(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(qwen3(CallSyntax::Tagged(Spelling::Json)), declared());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        parser.feed(delta(output), &mut out).expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.drain()
    }

    #[test]
    fn a_prompt_that_opens_the_thought_starts_the_output_in_reasoning() {
        // Qwen 3.5's generation prompt ends with `<think>\n`. bellwether's qwen3.5-27b parse
        // cases: an empty thought, closed at once, then the call.
        let output = format!("\n</think>\n\n{TAGGED_CALL}");
        let events = after_prompt("<|im_start|>assistant\n<think>\n", &output);
        assert_eq!(
            events[..5],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("\n")),
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("\n\n")),
            ]
        );
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);

        // A thought the stream ends inside is closed at the end, as one the model opened is.
        let events = after_prompt("<think>\n", "still");
        assert_eq!(
            events[..3],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("still")),
                Event::ReasoningEnd,
            ]
        );

        // Thinking disabled: the prompt closed the thought itself, and the output is content.
        let events = after_prompt("<|im_start|>assistant\n<think>\n\n</think>\n\n", "Hello");
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello")));

        // Qwen3's prompt opens nothing, and the model's own `<think>` is read as before; a
        // thought in an earlier turn of the prompt is closed there and opens nothing either.
        for prompt in [
            "<|im_start|>assistant\n",
            "<think>\nearlier\n</think>\n\nHi<|im_end|>\n<|im_start|>assistant\n",
        ] {
            let events = after_prompt(prompt, "<think>\nplan\n</think>\n\nHi");
            assert_eq!(
                events[..2],
                [dropped("<think>"), Event::ReasoningStart],
                "{prompt:?}"
            );
        }
    }

    /// The prompt, then the whole output in one delta, then the engine's stop, under the JSON
    /// syntax.
    fn after_prompt_json(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        parser.feed(delta(output), &mut out).expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.drain()
    }

    fn reasoning_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_prompt_is_replayed_from_the_turn_the_model_writes_not_from_its_start() {
        // A stray `<tool_call>` in the user's turn, and the generation prompt opens the thought:
        // the output is the thought, not content (smg #2839, Alex's probe against the parser
        // this table replaced, which read the prompt's tail by its last `<think>`).
        let output = "The user asks about the marker.\n</think>\n\nIt opens a tool call.";
        for prompt in [
            "<|im_start|>user\nWhy did you print <tool_call> there?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n",
            "<|im_start|>user\n<tool_response>\nsee <tool_call> in src\n</tool_response>\
             <|im_end|>\n\
             <|im_start|>assistant\n<think>\n",
        ] {
            for events in [
                after_prompt(prompt, output),
                after_prompt_json(prompt, output),
            ] {
                assert_eq!(events[0], Event::ReasoningStart, "{prompt:?}");
                assert_eq!(
                    reasoning_of(&events),
                    "The user asks about the marker.\n",
                    "{prompt:?}"
                );
            }
        }
        // An earlier call whose arguments hold `<think>`, closed in its own turn: the output is
        // content, as the model wrote no thought (the replaced parser read it as reasoning).
        let prompt = "<|im_start|>assistant\n<tool_call>\n\
                      {\"name\": \"write\", \"arguments\": {\"text\": \"<think>\"}}\n\
                      </tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nok\n\
                      </tool_response><|im_end|>\n<|im_start|>assistant\n";
        let events = after_prompt_json(prompt, "Done.");
        assert_eq!(events[0], Event::Content(Text::uncounted("Done.")));
    }

    #[test]
    fn a_table_whose_first_state_emits_arguments_starts_inside_a_call() {
        // No marker before the call: the engine opens it at the start, and the output's first
        // bytes are the call's.
        let bare = Format::new("bare")
            .state("call", Emits::Arguments)
            .calls(CallSyntax::Json);
        let mut parser = Engine::new(bare, Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "{\"name\": \"f\", \"arguments\": {\"a\": 1}}",
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert!(matches!(
            events.first(),
            Some(Event::ToolCallStart { index: 0, name, .. }) if name == "f"
        ));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_model_with_another_chat_template_names_its_own_turn_opener() {
        // K-EXAONE writes Qwen3's markers under its own template: a turn opens with
        // `<|assistant|>`, and the generation prompt ends `<|assistant|>\n<think>\n`. With the
        // table's ChatML opener the whole prompt would be replayed, and a stray `<tool_call>` in
        // the user's turn would leave the engine in content (smg #2841, Alex's probe).
        let prompt = "<|user|>\nWhy did you print <tool_call> there?<|endofturn|>\n\
                      <|assistant|>\n<think>\n";
        let output = "The user asks about the marker.\n</think>\n\nIt opens a tool call.";
        let mut parser = Engine::new(
            qwen3(CallSyntax::Json).opens_turn("<|assistant|>"),
            Declared::default(),
        );
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        let events = out.drain();
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning_of(&events), "The user asks about the marker.\n");
    }

    #[test]
    fn inputs_out_of_order_are_lifecycle_errors() {
        let mut parser = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|im_start|>assistant\n",
                },
                &mut out,
            )
            .expect("a prompt first is fine and says nothing");
        assert!(out.is_empty());
        parser.feed(delta("a"), &mut out).expect("delta");
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
}
