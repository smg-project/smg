//! Qwen: reasoning between `<think>` and `</think>`, each tool call between `<tool_call>` and
//! `</tool_call>`, everything else content. Inside the call markers the family writes one of two
//! syntaxes ([`CallSyntax`]): Qwen3 a JSON object, `{"name": …, "arguments": {…}}`; Qwen 3.5
//! and later and Qwen3-Coder the tags `<function=NAME>` and `<parameter=KEY>` around each value's
//! text, which the request's tools type.
//!
//! [`Qwen3`] is the first format and the first [`Parser`]. The scanner splits the model's text into
//! content and the four markers, holding back a half-arrived marker; a region says what the text
//! between markers is; inside a call region the call syntax's assembler ([`json::Assembler`] or
//! [`tagged::Assembler`]) turns the call into its events, streaming the model's own argument bytes.
//! Every byte of the output lands in exactly one event: the markers as `Dropped { Wrapper }`, the
//! whitespace between a call and its closing marker as `Dropped { Wrapper }` too, any other text
//! there as `Malformed`, run by run so that where the chunks were cut changes nothing, and the
//! rest as content, reasoning, or the call's events.
//!
//! What this format decides, and what it leaves open:
//!
//! - Prose before a call, between two calls, and after the last call is content, and every complete
//!   call in a chunk is emitted; the old parser dropped the first and stopped after one (smg
//!   #2788).
//! - The template's separator bytes stay where the model put them: the newline after `<think>`, the
//!   two after `</think>`, and so on are reasoning or content, not dropped. Whether they should be
//!   is bellwether #17; `Dropped { Whitespace }` exists for the other answer.
//! - A closing marker with nothing open (`</think>` in content) is content, as the model wrote it.
//!   A marker inside a code fence, or inside a string in a call's arguments, is a marker, as for
//!   every parser that reads markers: `</tool_call>` in an argument's text ends the call there.
//!   Bellwether #16 is where that policy is judged.
//! - A `<tool_call>` that never closes is finished at the end of the stream: a call with what
//!   arrived, or the bytes as `Malformed { UnterminatedRegion }`. A block whose closing marker
//!   comes before a complete call gives what is left as `Malformed` with a reason that says so,
//!   since `UnterminatedRegion` means the stream ended inside the region; with the tagged syntax
//!   the call is closed for the client first, as [`tagged::Assembler::close`] says.
//! - With the tagged syntax, a value's type comes from the request's tools ([`Declared`]): a
//!   declared string streams, every other value is written at its close. A call to a function the
//!   request did not declare has every value inferred.
//! - Call ids are `call_<index>` for now; the id scheme is Simo's decision (deterministic or
//!   carrying the conversation's history) and changes only this one line.
//! - Every text event says how many tokens it carries, counted by the [`Ledger`] from the deltas'
//!   spans: a token in the event that carries its first byte, a byte-less span (a held half of a
//!   character, or a hidden special token) into the run that carries the next byte, and the tokens
//!   left without bytes at the end once as `Dropped { ControlToken }`. `Finish::reasoning_tokens`
//!   is the count over the reasoning text. Once a delta lacks spans, or its spans do not partition
//!   its text, the rest of the stream is uncounted, and then `reasoning_tokens` is zero, the one
//!   place where zero does not mean none (rule 7 keeps `Finish` as it is for now).
//! - Tool names are not checked against the request's tools; the format has no tool list yet.
//!
//! The prompt decides one thing: whether the output starts inside the thinking region. Qwen 3.5's
//! template opens `<think>` in the generation prompt, so the output starts with the thought and
//! its first `</think>` closes it; Qwen3 writes its own `<think>`; and in both families a prompt
//! that disables thinking ends with `<think>\n\n</think>\n\n`, which leaves nothing open. So a
//! prompt whose last `<think>` comes after its last `</think>` starts the output in reasoning, with
//! `ReasoningStart` pushed for the prompt, and any other prompt starts it in content.

use crate::{
    event::{DropReason, Event, Events, FinishReason, MalformedReason},
    input::{EngineFinish, Input},
    json,
    markers::{Piece, Scanner},
    parser::{ParseError, Parser},
    tagged::{self, Declared},
    tokens::Ledger,
};

const THINK_OPEN: usize = 0;
const THINK_CLOSE: usize = 1;
const CALL_OPEN: usize = 2;
const CALL_CLOSE: usize = 3;
const MARKERS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];
const BLOCK_WITHOUT_A_COMPLETE_CALL: &str = "a tool-call block that closed without a complete call";
const TEXT_AFTER_THE_OBJECT: &str = "text between a call's object and its closing marker";

/// Why a call region closed: its closing marker (or the next opener) arrived, or the stream ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Closed {
    ByMarker,
    ByEnd,
}

/// How the model writes a call between the call markers.
#[derive(Clone, Debug)]
pub enum CallSyntax {
    /// One JSON object, `{"name": …, "arguments": {…}}`: Qwen3.
    Json,
    /// `<function=NAME>` and then `<parameter=KEY>` around each value's text, typed by the
    /// request's tools: Qwen 3.5 and later, Qwen3-Coder.
    Tagged(Declared),
}

/// The Qwen format as a parser. One per generated choice.
#[derive(Debug)]
pub struct Qwen3 {
    scanner: Scanner,
    syntax: CallSyntax,
    region: Region,
    calls: u32,
    tokens: Ledger,
    reasoning_tokens: u32,
    stage: Stage,
}

#[derive(Debug)]
enum Region {
    Content,
    Reasoning,
    Call(Call),
}

/// The assembler of the call region, one per call syntax.
#[derive(Debug)]
enum Call {
    Json(json::Assembler),
    Tagged(tagged::Assembler),
}

impl Call {
    fn new(syntax: &CallSyntax, index: u32) -> Self {
        let id = format!("call_{index}");
        match syntax {
            CallSyntax::Json => Self::Json(json::Assembler::new(index, id)),
            CallSyntax::Tagged(_) => Self::Tagged(tagged::Assembler::new(index, id)),
        }
    }

    fn feed(&mut self, text: &str, syntax: &CallSyntax, out: &mut Events) -> usize {
        match (self, syntax) {
            (Self::Json(assembler), _) => assembler.feed(text, out),
            (Self::Tagged(assembler), CallSyntax::Tagged(declared)) => {
                assembler.feed(text, declared, out)
            }
            // A tagged assembler exists only under the tagged syntax; the arm is unreachable, and
            // the text is kept as the call's rather than lost.
            (Self::Tagged(assembler), CallSyntax::Json) => {
                assembler.feed(text, &Declared::default(), out)
            }
        }
    }

    fn started(&self) -> bool {
        match self {
            Self::Json(assembler) => assembler.started(),
            Self::Tagged(assembler) => assembler.started(),
        }
    }

    /// Ends the call the way the region closed: the JSON assembler has one ending, and the format
    /// names a block its marker closed (`close_call`); the tagged one closes the call for the
    /// client when the block ended, and leaves it open when the stream was cut.
    fn end(self, closed: Closed, out: &mut Events) {
        match (self, closed) {
            (Self::Json(assembler), _) => assembler.finish(out),
            (Self::Tagged(assembler), Closed::ByMarker) => assembler.close(out),
            (Self::Tagged(assembler), Closed::ByEnd) => assembler.finish(out),
        }
    }
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
    /// A parser at the start of an output, for a model that writes each call as a JSON object.
    pub fn new() -> Self {
        Self::with_calls(CallSyntax::Json)
    }

    /// A parser at the start of an output, for a model that writes each call as tags; `declared`
    /// holds the request's tools, which type the values.
    pub fn with_tagged_calls(declared: Declared) -> Self {
        Self::with_calls(CallSyntax::Tagged(declared))
    }

    fn with_calls(syntax: CallSyntax) -> Self {
        Self {
            scanner: Scanner::new(MARKERS),
            syntax,
            region: Region::Content,
            calls: 0,
            tokens: Ledger::new(),
            reasoning_tokens: 0,
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
            Region::Content => out.push_content(self.tokens.text(text)),
            Region::Reasoning => {
                let counted = self.tokens.text(text);
                self.reasoning_tokens += counted.tokens.unwrap_or(0);
                out.push_reasoning(counted);
            }
            Region::Call(call) => {
                let mut assembled = Events::new();
                let taken = call.feed(text, &self.syntax, &mut assembled);
                for event in assembled.drain() {
                    out.push(self.tokens.relabel(event));
                }
                self.surplus(&text[taken..], out);
            }
        }
    }

    /// Text after a call and before its closing marker: each run of whitespace is the template's
    /// wrapping and is dropped, each run of anything else is malformed. Classifying run by run
    /// keeps the result the same wherever the chunks were cut.
    fn surplus(&mut self, surplus: &str, out: &mut Events) {
        let mut rest = surplus;
        while let Some(first) = rest.chars().next() {
            let space = first.is_whitespace();
            let length = rest
                .char_indices()
                .find(|(_, c)| c.is_whitespace() != space)
                .map_or(rest.len(), |(at, _)| at);
            let run = self.tokens.text(&rest[..length]);
            out.push(if space {
                Event::Dropped {
                    text: run,
                    why: DropReason::Wrapper,
                }
            } else {
                Event::Malformed {
                    text: run,
                    why: MalformedReason::Other(TEXT_AFTER_THE_OBJECT.to_string()),
                }
            });
            rest = &rest[length..];
        }
    }

    fn marker(&mut self, index: usize, out: &mut Events) {
        let marker = MARKERS[index];
        match (&mut self.region, index) {
            (Region::Content, THINK_OPEN) => {
                self.drop_marker(marker, out);
                out.push(Event::ReasoningStart);
                self.region = Region::Reasoning;
            }
            (Region::Reasoning, THINK_CLOSE) => {
                self.drop_marker(marker, out);
                out.push(Event::ReasoningEnd);
                self.region = Region::Content;
            }
            (Region::Content, CALL_OPEN) => {
                self.drop_marker(marker, out);
                self.open_call();
            }
            (Region::Call(_), CALL_CLOSE) => {
                // The call's remaining events come before the marker that closed it, so the events'
                // bytes stay in the output's order.
                self.close_call(Closed::ByMarker, out);
                self.drop_marker(marker, out);
            }
            (Region::Call(_), CALL_OPEN) => {
                // A new call before the previous one closed: finish what arrived, then start.
                self.close_call(Closed::ByMarker, out);
                self.drop_marker(marker, out);
                self.open_call();
            }
            // A closing marker with nothing open, or an opener inside a region that does not nest:
            // the model's bytes, where the model put them.
            (Region::Content | Region::Reasoning, _) | (Region::Call(_), _) => {
                self.text(marker, out);
            }
        }
    }

    /// The next call takes the next free index; the index is spent only if the region produces a
    /// call, so a `<tool_call>` block that held no call does not count and does not leave a gap.
    fn open_call(&mut self) {
        self.region = Region::Call(Call::new(&self.syntax, self.calls));
    }

    /// Ends the call region: the assembler closes what arrived. A region its marker closed reports
    /// leftover bytes as a block without a complete call; `UnterminatedRegion` is kept for a region
    /// the end of the stream cut.
    fn close_call(&mut self, closed: Closed, out: &mut Events) {
        if let Region::Call(call) = std::mem::replace(&mut self.region, Region::Content) {
            if call.started() {
                self.calls += 1;
            }
            let mut finished = Events::new();
            call.end(closed, &mut finished);
            for event in finished.drain() {
                let event = match event {
                    Event::Malformed {
                        text,
                        why: MalformedReason::UnterminatedRegion,
                    } if closed == Closed::ByMarker => Event::Malformed {
                        text,
                        why: MalformedReason::Other(BLOCK_WITHOUT_A_COMPLETE_CALL.to_string()),
                    },
                    event => event,
                };
                out.push(self.tokens.relabel(event));
            }
        }
    }

    fn drop_marker(&mut self, marker: &str, out: &mut Events) {
        out.push(Event::Dropped {
            text: self.tokens.text(marker),
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
            Region::Call(_) => self.close_call(Closed::ByEnd, out),
            Region::Content => {}
        }
        self.tokens.finish(out);
        let reason = match finish {
            EngineFinish::Stop => FinishReason::Stop,
            EngineFinish::Length => FinishReason::Length,
            EngineFinish::Abort => FinishReason::Abort,
            EngineFinish::Other(other) => FinishReason::Other(other),
        };
        out.push(Event::Finish {
            reason,
            tool_calls: self.calls,
            reasoning_tokens: if self.tokens.counting() {
                self.reasoning_tokens
            } else {
                0
            },
        });
    }
}

/// Whether the prompt ends inside the thinking region: its last `<think>` comes after its last
/// `</think>`.
fn leaves_thinking_open(prompt: &str) -> bool {
    match (
        prompt.rfind(MARKERS[THINK_OPEN]),
        prompt.rfind(MARKERS[THINK_CLOSE]),
    ) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

impl Parser for Qwen3 {
    fn feed(&mut self, input: Input<'_>, out: &mut Events) -> Result<(), ParseError> {
        match input {
            Input::Prompt { text, .. } => {
                if self.stage != Stage::Fresh {
                    return Err(ParseError::Lifecycle(
                        "prompt after output began".to_string(),
                    ));
                }
                self.stage = Stage::Streaming;
                if leaves_thinking_open(text) {
                    out.push(Event::ReasoningStart);
                    self.region = Region::Reasoning;
                }
                Ok(())
            }
            Input::Delta { text, spans, .. } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("delta after end".to_string()));
                }
                self.stage = Stage::Streaming;
                self.tokens.note(text, spans);
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
    use crate::{event::Text, input::TokenSpan};

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
        let mut parser = Qwen3::new();
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
        let mut parser = Qwen3::new();
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
        let mut parser = Qwen3::new();
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
        let mut parser = Qwen3::new();
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
        let mut parser = Qwen3::with_tagged_calls(declared());
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
        let mut parser = Qwen3::with_tagged_calls(declared());
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
