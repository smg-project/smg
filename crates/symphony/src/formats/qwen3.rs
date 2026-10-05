//! Qwen3: reasoning between `<think>` and `</think>`, each tool call a JSON object between
//! `<tool_call>` and `</tool_call>`, everything else content.
//!
//! [`Qwen3`] is the first format and the first [`Parser`]. The scanner splits the model's text into
//! content and the four markers, holding back a half-arrived marker; a region says what the text
//! between markers is; inside a call region the [`Assembler`] turns the object into the call's
//! events, streaming the model's own argument bytes. Every byte of the output lands in exactly one
//! event: the markers as `Dropped { Wrapper }`, the whitespace between a call's object and its
//! closing marker as `Dropped { Wrapper }` too, and the rest as content, reasoning, or the call's
//! events.
//!
//! What this format decides, and what it leaves open:
//!
//! - Prose before a call, between two calls, and after the last call is content, and every complete
//!   call in a chunk is emitted; the old parser dropped the first and stopped after one (smg #2788).
//! - The template's separator bytes stay where the model put them: the newline after `<think>`, the
//!   two after `</think>`, and so on are reasoning or content, not dropped. Whether they should be
//!   is bellwether #17; `Dropped { Whitespace }` exists for the other answer.
//! - A closing marker with nothing open (`</think>` in content) is content, as the model wrote it.
//!   A marker inside a code fence is a marker, as for every parser that reads markers; bellwether
//!   #16 is where that policy is judged.
//! - A `<tool_call>` that never closes is finished at the end of the stream: a call with what
//!   arrived, or the bytes as `Malformed { UnterminatedRegion }`.
//! - Call ids are `call_<index>` for now; the id scheme is Simo's decision (deterministic or carrying
//!   the conversation's history) and changes only this one line.
//! - Text is `Text::uncounted`: token attribution from the input spans comes with the engine work;
//!   `Finish::reasoning_tokens` is zero until then.
//! - Tool names are not checked against the request's tools; the format has no tool list yet.
//!
//! The prompt is accepted first in the lifecycle and otherwise ignored: Qwen3 writes its own
//! `<think>` into the output, so nothing about the prompt decides where the output starts.

use crate::{
    event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
    input::{EngineFinish, Input},
    json::Assembler,
    markers::{Piece, Scanner},
    parser::{ParseError, Parser},
};

const THINK_OPEN: usize = 0;
const THINK_CLOSE: usize = 1;
const CALL_OPEN: usize = 2;
const CALL_CLOSE: usize = 3;
const MARKERS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

/// The Qwen3 format as a parser. One per generated choice.
#[derive(Debug)]
pub struct Qwen3 {
    scanner: Scanner,
    region: Region,
    calls: u32,
    stage: Stage,
}

#[derive(Debug)]
enum Region {
    Content,
    Reasoning,
    Call(Assembler),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Streaming,
    Ended,
}

impl Default for Qwen3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3 {
    /// A parser at the start of an output.
    pub fn new() -> Self {
        Self {
            scanner: Scanner::new(MARKERS),
            region: Region::Content,
            calls: 0,
            stage: Stage::Fresh,
        }
    }

    fn take(&mut self, piece: Piece, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, out),
            Piece::Marker(index) => self.marker(index, out),
        }
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        match &mut self.region {
            Region::Content => out.push_content(Text::uncounted(text)),
            Region::Reasoning => out.push_reasoning(Text::uncounted(text)),
            Region::Call(assembler) => {
                let taken = assembler.feed(text, out);
                let surplus = &text[taken..];
                if surplus.is_empty() {
                    return;
                }
                if surplus.trim().is_empty() {
                    out.push(Event::Dropped {
                        text: Text::uncounted(surplus),
                        why: DropReason::Wrapper,
                    });
                } else {
                    out.push(Event::Malformed {
                        text: Text::uncounted(surplus),
                        why: MalformedReason::Other(
                            "text between a call's object and its closing marker".to_string(),
                        ),
                    });
                }
            }
        }
    }

    fn marker(&mut self, index: usize, out: &mut Events) {
        let marker = MARKERS[index];
        match (&mut self.region, index) {
            (Region::Content, THINK_OPEN) => {
                Self::drop_marker(marker, out);
                out.push(Event::ReasoningStart);
                self.region = Region::Reasoning;
            }
            (Region::Reasoning, THINK_CLOSE) => {
                Self::drop_marker(marker, out);
                out.push(Event::ReasoningEnd);
                self.region = Region::Content;
            }
            (Region::Content, CALL_OPEN) => {
                Self::drop_marker(marker, out);
                self.open_call();
            }
            (Region::Call(_), CALL_CLOSE) => {
                Self::drop_marker(marker, out);
                self.close_call(out);
            }
            (Region::Call(_), CALL_OPEN) => {
                // A new call before the previous one closed: finish what arrived, then start.
                self.close_call(out);
                Self::drop_marker(marker, out);
                self.open_call();
            }
            // A closing marker with nothing open, or an opener inside a region that does not nest:
            // the model's bytes, where the model put them.
            (Region::Content, _) => out.push_content(Text::uncounted(marker)),
            (Region::Reasoning, _) => out.push_reasoning(Text::uncounted(marker)),
            (Region::Call(_), _) => self.text(marker, out),
        }
    }

    fn open_call(&mut self) {
        let index = self.calls;
        self.calls += 1;
        self.region = Region::Call(Assembler::new(index, format!("call_{index}")));
    }

    fn close_call(&mut self, out: &mut Events) {
        if let Region::Call(assembler) = std::mem::replace(&mut self.region, Region::Content) {
            assembler.finish(out);
        }
    }

    fn drop_marker(marker: &str, out: &mut Events) {
        out.push(Event::Dropped {
            text: Text::uncounted(marker),
            why: DropReason::Wrapper,
        });
    }

    fn end(&mut self, finish: EngineFinish, out: &mut Events) {
        let scanner = std::mem::replace(&mut self.scanner, Scanner::new(MARKERS));
        for piece in scanner.finish() {
            self.take(piece, out);
        }
        match &self.region {
            Region::Reasoning => {
                out.push(Event::ReasoningEnd);
                self.region = Region::Content;
            }
            Region::Call(_) => self.close_call(out),
            Region::Content => {}
        }
        let reason = match finish {
            EngineFinish::Stop => FinishReason::Stop,
            EngineFinish::Length => FinishReason::Length,
            EngineFinish::Abort => FinishReason::Abort,
            EngineFinish::Other(other) => FinishReason::Other(other),
        };
        out.push(Event::Finish {
            reason,
            tool_calls: self.calls,
            reasoning_tokens: 0,
        });
    }
}

impl Parser for Qwen3 {
    fn feed(&mut self, input: Input<'_>, out: &mut Events) -> Result<(), ParseError> {
        match input {
            Input::Prompt { .. } => {
                if self.stage != Stage::Fresh {
                    return Err(ParseError::Lifecycle(
                        "prompt after output began".to_string(),
                    ));
                }
                self.stage = Stage::Streaming;
                Ok(())
            }
            Input::Delta { text, .. } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("delta after end".to_string()));
                }
                self.stage = Stage::Streaming;
                for piece in self.scanner.feed(text) {
                    self.take(piece, out);
                }
                Ok(())
            }
            Input::End { finish } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("end after end".to_string()));
                }
                self.stage = Stage::Ended;
                self.end(finish, out);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn delta(text: &str) -> Input<'_> {
        Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        }
    }

    fn run(pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Qwen3::new();
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

    #[test]
    fn inputs_out_of_order_are_lifecycle_errors() {
        let mut parser = Qwen3::new();
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
