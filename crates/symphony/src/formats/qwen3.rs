//! Qwen3: reasoning between `<think>` and `</think>`, each tool call between `<tool_call>` and
//! `</tool_call>`, everything else content. Inside the call markers the family writes one of two
//! syntaxes ([`CallSyntax`]): Qwen3 a JSON object, `{"name": …, "arguments": {…}}`; Qwen 3.5
//! and later and Qwen3-Coder the tags `<function=NAME>` and `<parameter=KEY>` around each value's
//! text, which the request's tools type, an object or a list written as JSON, one function block
//! per pair of markers as the template writes it and several in one as a model may write a
//! parallel answer. Qwen 3.5 puts the
//! value on a line of its own and MiMo writes it between the tags directly, which the syntax's
//! [`Placement`] says.
//!
//! The table carries the one part of its chat template's layout the tags write: after the
//! thought's close the Qwen3 and Qwen 3.5 templates write two newlines before the content or the
//! first call (`</think>\n\n<tool_call>`), so a forced call written where the close left the
//! engine begins with them, as the template writes it, and the call's marker is not the grammar's
//! first text ([`Format::grammar_for`]). MiMo's template writes what follows right after the
//! close, so the table under the direct placement carries none; the NVIDIA Nemotron 3 groups and
//! Step 3.5 Flash, which read with the tagged table, write one newline there, which their entries
//! in `groups.json` carry in the table's place (`after`). The thought the model opens and closes
//! itself: a `<tool_call>` inside it is thought text, as the template reads it.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

#[cfg(test)]
use crate::tagged::Spelling;
use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::Placement,
};

/// The Qwen3 table, with the call syntax the checkpoint writes; under the direct placement
/// (MiMo) without the thought layout, whose two newlines after the close that template does not
/// write.
pub fn qwen3(syntax: CallSyntax) -> Format {
    let format = Format::new("qwen3")
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
        .opens_turn("<|im_start|>assistant");
    if matches!(syntax, CallSyntax::Tagged(_, Placement::Direct)) {
        return format;
    }
    format.with_layout(super::thought_layout("</think>", "\n\n"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        ids::CallIds,
        testing::{
            a_forced_tag_begins_with_the_templates_text, arguments_of, bytes, content, drive,
            reasoning, the_same_at_every_cut, Said,
        },
    };

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    /// An engine whose ids count from the one nonce, so that the events of two engines compare.
    fn engine(format: Format, declared: Declared) -> Engine {
        Engine::with_call_ids(Arc::new(format), declared, ids())
    }
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
        input::{EngineFinish, Input, TokenSpan},
        parser::{ParseError, Parser},
        Declared,
    };

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn delta(text: &str) -> Input<'_> {
        Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        }
    }

    fn run(pieces: &[&str], finish: EngineFinish) -> Vec<Event<'static>> {
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        for piece in pieces {
            parser.feed(delta(piece), &mut out).expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.into_owned().drain()
    }

    /// Events with adjacent text of one kind joined and argument fragments joined per call, so two
    /// chunkings compare by what they said.
    fn joined(events: Vec<Event<'static>>) -> Vec<Event<'static>> {
        let mut out: Vec<Event<'static>> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (Some(Event::Content(a)), Event::Content(b))
                | (Some(Event::Reasoning(a)), Event::Reasoning(b)) => {
                    a.text.to_mut().push_str(&b.text);
                }
                (
                    Some(Event::Dropped { text: a, why: wa }),
                    Event::Dropped { text: b, why: wb },
                ) if *wa == wb => a.text.to_mut().push_str(&b.text),
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
                    source.text.to_mut().push_str(&more_source.text);
                }
                (_, event) => out.push(event),
            }
        }
        out
    }

    fn dropped(text: &str) -> Event<'static> {
        Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::Wrapper,
        }
    }

    /// A call whose arguments are not JSON carries them as the model wrote them and says once,
    /// with no bytes, that they are no JSON; nothing leaks into content, and the pieces say what
    /// the whole says.
    #[test]
    fn arguments_that_are_not_json_stay_the_calls_own_and_are_said_to_be_no_json() {
        let output = "<tool_call>\n{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}\n</tool_call>";
        let per_char: Vec<&str> = output
            .char_indices()
            .map(|(i, c)| &output[i..i + c.len_utf8()])
            .collect();
        let whole = run(&[output], EngineFinish::Stop);
        let streamed = run(&per_char, EngineFinish::Stop);
        for (how, events) in [("whole", &whole), ("per character", &streamed)] {
            assert_eq!(
                arguments_of(events, 0),
                "{\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}",
                "{how}"
            );
            assert_eq!(content(events), "", "{how}");
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, Event::Malformed { .. }))
                    .collect::<Vec<_>>(),
                [&Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::InvalidArguments,
                }],
                "{how}: the break is said once, with no bytes"
            );
            assert_eq!(bytes(events), output, "{how}");
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "{how}"
            );
        }
        assert_eq!(
            joined(whole),
            joined(streamed),
            "the pieces say what the whole says"
        );
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
                // The template's two newlines before the call: held until the call's opener
                // said they were its spacing, not content.
                Event::Dropped {
                    text: Text::uncounted("\n\n"),
                    why: DropReason::Whitespace,
                },
                dropped("<tool_call>"),
                Event::ToolCallStart {
                    index: 0,
                    id: ids().id(0),
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
                (0, ids().id(0), "get_weather".into()),
                (1, ids().id(1), "get_time".into()),
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
            events.contains(&Event::Dropped {
                text: Text::uncounted("\n"),
                why: DropReason::Whitespace,
            }),
            "the newline between the calls is the template's spacing"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Content(_))),
            "and no content: {events:?}"
        );
        assert_eq!(bytes(&events), output);
    }

    /// The content text of the events, in order.
    /// The whitespace the engine dropped as the template's spacing, event by event.
    fn spacing<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Whitespace,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The content deltas a client of the stream receives when `pieces` arrive one by one after
    /// `prompt`: the content of each feed's events, the end's included, empty ones left out.
    fn content_deltas(format: Format, prompt: &str, pieces: &[&str]) -> Vec<String> {
        let mut parser = Engine::new(format, declared());
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
        let mut deltas = Vec::new();
        for piece in pieces {
            parser.feed(delta(piece), &mut out).expect("delta");
            deltas.push(content(&out.drain()));
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        deltas.push(content(&out.drain()));
        deltas.retain(|delta| !delta.is_empty());
        deltas
    }

    /// `text` cut into the pieces a tokenizer makes of it: at every newline and every marker,
    /// which the family writes as tokens of their own.
    fn tokens_of(text: &str) -> Vec<&str> {
        let mut pieces = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let cut = ["\n", "<think>", "</think>", "<tool_call>", "</tool_call>"]
                .iter()
                .filter_map(|marker| rest.find(marker).map(|at| (at, marker.len())))
                .min()
                .map_or(rest.len(), |(at, len)| if at == 0 { len } else { at });
            pieces.push(&rest[..cut]);
            rest = &rest[cut..];
        }
        pieces
    }

    #[test]
    fn the_spacing_after_the_thought_and_between_two_calls_is_not_content_whole_or_streamed() {
        let second = CALL
            .replace("get_weather", "get_time")
            .replace("Paris", "CET");
        let output = format!("<think>\nplan\n</think>\n\n{CALL}\n{second}");
        let whole = run(&[&output], EngineFinish::Stop);
        assert_eq!(bytes(&whole), output);
        assert_eq!(content(&whole), "");
        assert_eq!(spacing(&whole), ["\n\n", "\n"]);
        assert!(matches!(
            whole.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
        // Byte by byte and token by token the events are the same, and no feed's events carry
        // content: a client of the stream gets no content delta, as the whole answer has none.
        let whole = joined(whole);
        let per_char: Vec<&str> = output
            .char_indices()
            .map(|(i, c)| &output[i..i + c.len_utf8()])
            .collect();
        for pieces in [per_char, tokens_of(&output)] {
            assert_eq!(
                joined(run(&pieces, EngineFinish::Stop)),
                whole,
                "{pieces:?}"
            );
            assert_eq!(
                content_deltas(qwen3(CallSyntax::Json), "", &pieces),
                Vec::<String>::new(),
                "{pieces:?}"
            );
        }
    }

    #[test]
    fn the_tagged_syntax_streams_no_content_for_the_spacing_after_the_thought_or_between_calls() {
        // Qwen 3.5's prompt opens the thought; the model closes it at once and writes two calls.
        let prompt = "<|im_start|>assistant\n<think>\n";
        let output = format!("\n</think>\n\n{TAGGED_CALL}\n{TAGGED_CALL}");
        let whole = after_prompt(prompt, &output);
        assert_eq!(bytes(&whole), output);
        assert_eq!(content(&whole), "");
        assert_eq!(spacing(&whole), ["\n\n", "\n"]);
        assert_eq!(arguments_of(&whole, 1), r#"{"city": "Paris", "days": 3}"#);
        let tagged = || qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine));
        let per_char: Vec<&str> = output
            .char_indices()
            .map(|(i, c)| &output[i..i + c.len_utf8()])
            .collect();
        for pieces in [per_char, tokens_of(&output)] {
            assert_eq!(
                content_deltas(tagged(), prompt, &pieces),
                Vec::<String>::new(),
                "{pieces:?}"
            );
        }
        // Text after the thought keeps the spacing around it as content, whole and streamed:
        // the stream shows the spacing with the text it belongs to.
        let output = format!("\n</think>\n\nSure.\n{TAGGED_CALL}");
        let events = after_prompt(prompt, &output);
        assert_eq!(content(&events), "\n\nSure.\n");
        assert_eq!(spacing(&events), Vec::<&str>::new());
        assert_eq!(
            content_deltas(tagged(), prompt, &tokens_of(&output)),
            ["\n\nSure.", "\n"]
        );
    }

    #[test]
    fn the_tokens_of_a_held_lead_are_counted_where_the_lead_leaves() {
        // The spacing's own tokens are counted in the whitespace event that drops it, and the
        // sum over the events is still the number of tokens.
        let tokens = [
            "<think>",
            "\nplan",
            "\n</think>",
            "\n\n",
            "<tool_call>",
            "{\"name\": \"f\", ",
            "\"arguments\": {}}",
            "</tool_call>",
            "\n",
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
        let whitespace: Vec<(String, Option<u32>)> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Whitespace,
                } => Some((text.text.to_string(), text.tokens)),
                _ => None,
            })
            .collect();
        assert_eq!(
            whitespace,
            [("\n\n".to_string(), Some(1)), ("\n".to_string(), Some(1))]
        );
        assert_eq!(
            counted(&events)
                .iter()
                .map(|(_, n)| n.unwrap_or(0))
                .sum::<u32>(),
            tokens.len() as u32,
            "every token exactly once"
        );
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
        // The close stays in the content where the model put it; the event of no bytes before it
        // says the table read no frame there.
        let events = run(&["a</think>b"], EngineFinish::Stop);
        assert_eq!(
            events[..3],
            [
                Event::Content(Text::uncounted("a")),
                Event::Malformed {
                    text: Text::uncounted(""),
                    why: MalformedReason::MarkerOutOfPlace,
                },
                Event::Content(Text::uncounted("</think>b")),
            ]
        );
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
                    why: MalformedReason::TextAfterTheArguments,
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
                    Event::Dropped { text, .. } if dropped => Some(&*text.text),
                    Event::Malformed { text, .. } if !dropped => Some(&*text.text),
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
                    id: ids().id(0),
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
                    why: MalformedReason::BlockWithoutACompleteCall,
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
                why: MalformedReason::BlockWithoutACompleteCall,
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
            vec![(0, ids().id(0))],
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
                Event::Content(t) | Event::Reasoning(t) => Some((t.text.to_string(), t.tokens)),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => {
                    Some((text.text.to_string(), text.tokens))
                }
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => {
                    Some((source.text.to_string(), source.tokens))
                }
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
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
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
        let spans: Vec<Vec<TokenSpan>> = pieces
            .iter()
            .map(|(_, ranges)| {
                ranges
                    .iter()
                    .map(|&(start, end)| TokenSpan {
                        token_id: 0,
                        start,
                        end,
                        continued: false,
                    })
                    .collect()
            })
            .collect();
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        for ((text, _), spans) in pieces.iter().zip(&spans) {
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text,
                        spans,
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
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
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
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
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
                extra: Default::default(),
            },
        }])
    }

    const TAGGED_CALL: &str = concat!(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>",
        "\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    );

    fn run_tagged(pieces: &[&str], finish: EngineFinish) -> Vec<Event<'static>> {
        let mut parser = engine(
            qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
            declared(),
        );
        let mut out = Events::new();
        for piece in pieces {
            parser.feed(delta(piece), &mut out).expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.into_owned().drain()
    }

    /// A parameter written in XML-attribute style, `<parameter name="city">`, which the tagged
    /// syntax has no tag for: the call is delivered with no arguments, and the model's lines are
    /// refused, byte for byte, the whitespace between the words included, never the chat
    /// message's content nor a delta, at every cut alike.
    #[test]
    fn an_xml_attribute_parameter_is_refused_never_content() {
        let output = concat!(
            "<tool_call>\n<function=get_weather>\n<parameter name=\"city\">Paris</parameter>\n",
            "<parameter name=\"days\">three or four days</parameter>\n</function>\n</tool_call>"
        );
        let lines = concat!(
            "<parameter name=\"city\">Paris</parameter>\n",
            "<parameter name=\"days\">three or four days</parameter>"
        );
        let content = |events: &[Event]| chat::message(0, events).message.content;
        let streamed = |events: &[Event]| -> String {
            chat::deltas(0, events)
                .into_iter()
                .filter_map(|choice| choice.delta.content)
                .collect()
        };
        let refused = |events: &[Event]| -> String {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Dropped {
                        text,
                        why: DropReason::MalformedCall,
                    } => Some(&*text.text),
                    _ => None,
                })
                .collect()
        };
        let whole = run_tagged(&[output], EngineFinish::Stop);
        let message = chat::message(0, &whole).message;
        assert_eq!(message.content, None);
        let calls = message.tool_calls.expect("a call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments.as_deref(), Some("{}"));
        assert_eq!(streamed(&whole), "");
        assert_eq!(refused(&whole), lines);
        for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
            let events = run_tagged(&[&output[..cut], &output[cut..]], EngineFinish::Stop);
            assert_eq!(content(&events), None, "cut at {cut}");
            assert_eq!(streamed(&events), "", "cut at {cut}");
            assert_eq!(refused(&events), lines, "cut at {cut}");
        }
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
                    id: ids().id(0),
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
                (0, ids().id(0), "get_weather".to_string()),
                (1, ids().id(1), "lookup".to_string()),
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
    fn two_function_blocks_in_one_block_are_two_calls_under_either_placement() {
        // The template writes one `<tool_call>` block per call; a parallel answer may put its
        // function blocks in one, which the engines' own parsers read as the calls they are.
        let shapes = [
            (
                Placement::OwnLine,
                concat!(
                    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
                    "</function>\n<function=lookup>\n<parameter=id>\n42\n</parameter>\n</function>\n",
                    "</tool_call>"
                ),
            ),
            (
                Placement::Direct,
                concat!(
                    "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n",
                    "</function>\n<function=lookup>\n<parameter=id>42</parameter>\n</function>\n",
                    "</tool_call>"
                ),
            ),
        ];
        // Both functions declared: the engine announces a call to an undeclared one, and the
        // subject here is the block's shape.
        let two_tools = || {
            use openai_protocol::common::{Function, Tool};
            use serde_json::json as value;
            let tool = |name: &str, parameters: serde_json::Value| Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: name.to_string(),
                    description: None,
                    parameters,
                    strict: None,
                    extra: Default::default(),
                },
            };
            Declared::of(&[
                tool(
                    "get_weather",
                    value!({"type": "object", "properties": {"city": {"type": "string"}}}),
                ),
                tool(
                    "lookup",
                    value!({"type": "object", "properties": {"id": {"type": "integer"}}}),
                ),
            ])
        };
        for (placement, output) in shapes {
            let per_char: Vec<String> = output.chars().map(String::from).collect();
            let per_char: Vec<&str> = per_char.iter().map(String::as_str).collect();
            for pieces in [vec![output], per_char] {
                let mut parser = engine(
                    qwen3(CallSyntax::Tagged(Spelling::Json, placement)),
                    two_tools(),
                );
                let mut out = Events::new();
                for piece in &pieces {
                    parser.feed(delta(piece), &mut out).expect("delta");
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
                        (0, ids().id(0), "get_weather".to_string()),
                        (1, ids().id(1), "lookup".to_string()),
                    ],
                    "{placement:?} in {} pieces",
                    pieces.len()
                );
                assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
                assert_eq!(arguments_of(&events, 1), r#"{"id": 42}"#);
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, Event::Malformed { .. })),
                    "{placement:?}: {events:?}"
                );
                assert_eq!(bytes(&events), output);
                assert!(matches!(
                    events.last(),
                    Some(Event::Finish { tool_calls: 2, .. })
                ));
            }
        }
    }

    #[test]
    fn a_tagged_block_its_marker_closes_inside_a_string_closes_it_there_and_a_cut_stream_does_not()
    {
        // The block's closing marker arrives before `</function>`, inside a string: the model
        // ended the call, so the string is closed with the value's bytes whole, under one
        // `Malformed { OpenAtTheMarker }` event of no bytes right before the closing quote, and
        // the arguments read as JSON.
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar</tool_call>";
        let events = run_tagged(&[output], EngineFinish::Stop);
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Par"}"#);
        assert_eq!(bytes(&events), output);
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))
            .expect("the one word");
        assert!(
            matches!(
                &events[at..at + 2],
                [
                    Event::Malformed { text, why: MalformedReason::OpenAtTheMarker },
                    Event::ToolCallArguments { json, .. }
                ] if text.text.is_empty() && json == "\""
            ),
            "{events:#?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count(),
            1
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
    fn after_prompt(prompt: &str, output: &str) -> Vec<Event<'static>> {
        let mut parser = engine(
            qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
            declared(),
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
        parser.feed(delta(output), &mut out).expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.into_owned().drain()
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
                Event::Dropped {
                    text: Text::uncounted("\n\n"),
                    why: DropReason::Whitespace,
                },
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
    fn after_prompt_json(prompt: &str, output: &str) -> Vec<Event<'static>> {
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
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
        out.into_owned().drain()
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
                    reasoning(&events),
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
        let mut parser = engine(bare, Declared::default());
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
        let mut parser = engine(
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
        assert_eq!(reasoning(&events), "The user asks about the marker.\n");
    }

    /// The table with the tagged syntax at `placement`, for a `submit_patch` tool whose `patch`
    /// is a declared string, as bellwether's swebench call cases declare it.
    fn submit_patch(placement: Placement) -> Engine {
        use openai_protocol::common::{Function, Tool};
        use serde_json::json as value;
        engine(
            qwen3(CallSyntax::Tagged(Spelling::Json, placement)),
            Declared::of(&[Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "submit_patch".to_string(),
                    description: None,
                    parameters: value!({
                        "type": "object",
                        "properties": {"patch": {"type": "string"}},
                    }),
                    strict: None,
                    extra: Default::default(),
                },
            }]),
        )
    }

    #[test]
    fn a_value_written_between_the_tags_directly_keeps_the_newline_it_ends_with() {
        // MiMo writes `<parameter=KEY>`, the value and `</parameter>` with no newline between
        // them, where Qwen 3.5 puts the value on a line of its own: bellwether's
        // mimo-v2.5/parse/swebench-*-call-* cases, whose patch ends in a newline, came back a byte
        // short under the Qwen 3.5 reading (smg-lab #110). Each template's own shape gives the
        // value back whole under its placement, at every cut.
        let patch = "diff --git a/f b/f\n-x\n+y\n";
        let expected = format!(
            "{{\"patch\": {}}}",
            serde_json::Value::String(patch.to_string())
        );
        let shapes = [
            (
                Placement::Direct,
                format!(
                    "<think></think><tool_call>\n<function=submit_patch>\n\
                     <parameter=patch>{patch}</parameter>\n</function>\n</tool_call>"
                ),
            ),
            (
                Placement::OwnLine,
                format!(
                    "<tool_call>\n<function=submit_patch>\n\
                     <parameter=patch>\n{patch}\n</parameter>\n</function>\n</tool_call>"
                ),
            ),
        ];
        for (placement, output) in &shapes {
            for cut in 0..output.len() {
                let pieces: Vec<&str> = if cut == 0 {
                    vec![output]
                } else {
                    vec![&output[..cut], &output[cut..]]
                };
                let mut parser = submit_patch(*placement);
                let mut out = Events::new();
                for piece in &pieces {
                    parser.feed(delta(piece), &mut out).expect("delta");
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
                    arguments_of(&events, 0),
                    expected,
                    "{placement:?} cut at {cut}"
                );
                assert_eq!(bytes(&events), *output, "{placement:?} cut at {cut}");
            }
        }
        // Under the direct placement a value of one newline is one newline, spaces are spaces,
        // and nothing is nothing.
        for (value, expected) in [("\n", "\"\\n\""), ("  ", "\"  \""), ("", "\"\"")] {
            let output = format!(
                "<tool_call>\n<function=submit_patch>\n<parameter=patch>{value}</parameter>\n\
                 </function>\n</tool_call>"
            );
            let mut parser = submit_patch(Placement::Direct);
            let mut out = Events::new();
            parser.feed(delta(&output), &mut out).expect("delta");
            parser
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("end");
            assert_eq!(
                arguments_of(&out.drain(), 0),
                format!("{{\"patch\": {expected}}}"),
                "{value:?}"
            );
        }
    }

    #[test]
    fn inputs_out_of_order_are_lifecycle_errors() {
        let mut parser = engine(qwen3(CallSyntax::Json), Declared::default());
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

    /// The tool `get_weather` of [`declared()`] and one without parameters, as a request lists
    /// them.
    fn weather_tools() -> Vec<openai_protocol::common::Tool> {
        use openai_protocol::common::{Function, Tool};
        use serde_json::json as value;
        let tool = |name: &str, parameters: serde_json::Value| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        };
        vec![
            tool(
                "get_weather",
                value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                }}),
            ),
            tool("ping", value!({"type": "object", "properties": {}})),
        ]
    }

    /// The two tables that carry the thought layout, each with a call as its syntax writes it.
    fn tables_with_a_call() -> [(Format, &'static str); 2] {
        [
            (qwen3(CallSyntax::Json), CALL_WITH_DAYS),
            (
                qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
                TAGGED_CALL,
            ),
        ]
    }

    const CALL_WITH_DAYS: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
        {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";

    /// The events of `pieces` after the turn's opener, read by `format` with the weather tool.
    fn read_with(format: &Format, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut engine(format.clone(), declared()),
            "<|im_start|>assistant\n",
            pieces,
        )
    }

    #[test]
    fn a_prompt_with_thinking_off_has_written_the_templates_two_newlines_itself() {
        // A request with thinking switched off closes an empty thought in the prompt and writes
        // the two newlines after it (`<think>\n\n</think>\n\n` at the prompt's end): the table
        // reads that from the prompt, as the gateway does beside `reasoning_open`, and a forced
        // call written from there is the calls at once on every start, the newlines not written a
        // second time. The thinking-on prompt has written nothing, and MiMo's table carries no
        // layout to have written.
        use crate::testing::a_forced_tag_after_a_prompt_that_wrote_the_templates_text;
        let tools = weather_tools();
        let off = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        for format in [
            qwen3(CallSyntax::Json),
            qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
        ] {
            a_forced_tag_after_a_prompt_that_wrote_the_templates_text(
                &format,
                &tools,
                off,
                "<|im_start|>assistant\n",
                "<tool_call>",
            );
        }
        assert!(!qwen3(CallSyntax::Tagged(Spelling::Json, Placement::Direct)).thought_closed(off));
    }

    #[test]
    fn a_forced_call_after_the_thoughts_close_begins_with_the_templates_two_newlines() {
        // The Qwen3 and Qwen 3.5 templates write two newlines after the thought's close before
        // the content or the first call, and the table carries that part of its layout. Where the
        // grammar starts after the close (after the gate of the engine's reasoning parser, with
        // the prompt's thought closed or open; after the thought a free-text prefix or a tag of
        // the thought wrote), the calls' tag begins with them, so the call's marker is not the
        // grammar's first text: an engine that takes a marker the model rehearses inside its
        // thought for the thought's end cannot start the forced call there once the marker cannot
        // start the grammar. With the thought closed, at the first token or where the engine does
        // not say, the calls at once, as before. MiMo's table, the direct placement, carries no
        // layout: its template writes what follows right after the close.
        use crate::grammar::{GrammarStart, Options};
        let tools = weather_tools();
        for format in [
            qwen3(CallSyntax::Json),
            qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
        ] {
            a_forced_tag_begins_with_the_templates_text(
                &format,
                &tools,
                "</think>",
                "\n\n",
                "<tool_call>",
            );
        }
        let direct = qwen3(CallSyntax::Tagged(Spelling::Json, Placement::Direct));
        assert!(direct.layout().is_none());
        let gated = direct
            .grammar_for(
                &tools[..1],
                Options {
                    at_least_one: true,
                    reasoning_open: true,
                    thought_closed: false,
                    one_call: true,
                    start: GrammarStart::AfterReasoning,
                },
            )
            .expect("a tag")
            .payload();
        assert_eq!(gated["format"]["type"], "triggered_tags", "{gated}");
    }

    #[test]
    fn a_forced_call_written_after_the_thoughts_close_reads_as_the_call_with_the_rehearsal_as_thought(
    ) {
        // The tag's shape on an engine whose grammar starts after the thought's close: the thought
        // with the call's format rehearsed inside it, its close, the template's two newlines and
        // the call. The rehearsal is thought text, the newlines the template's, the call the call;
        // whole, one character at a time and at every cut, under both syntaxes.
        for (format, call) in tables_with_a_call() {
            let rehearsed = format!("\nThe user wants the forecast. The format is {call}. So:\n");
            let output = format!("<think>{rehearsed}</think>\n\n{call}");
            assert_eq!(
                the_same_at_every_cut(|pieces| read_with(&format, pieces), &output),
                Said {
                    calls: vec![(
                        "get_weather".to_string(),
                        "{\"city\": \"Paris\", \"days\": 3}".to_string()
                    )],
                    finish: Some((FinishReason::Stop, 1)),
                    content: String::new(),
                    reasoning: rehearsed,
                },
                "{call}"
            );
        }
    }

    #[test]
    fn a_call_marker_inside_the_thought_stays_thought_text() {
        // An engine whose reasoning gate took the rehearsed marker for the thought's end and
        // started a forced call there wrote the call inside the thought, the output ending there;
        // the table reads it as the template does: thought text, no call, `stop`. The tag no
        // longer lets such an engine start there (its first text after the close is the
        // template's newlines, not the marker); the reading is pinned here, whole, one character
        // at a time and at every cut, under both syntaxes.
        for (format, call) in tables_with_a_call() {
            let thought = format!("\nThe user wants the forecast. The format is {call}");
            let output = format!("<think>{thought}");
            assert_eq!(
                the_same_at_every_cut(|pieces| read_with(&format, pieces), &output),
                Said {
                    calls: Vec::new(),
                    finish: Some((FinishReason::Stop, 0)),
                    content: String::new(),
                    reasoning: thought,
                },
                "{call}"
            );
        }
    }
}
