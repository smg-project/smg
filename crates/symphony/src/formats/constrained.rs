//! Constrained: the output a grammar forced for the request's `tool_choice`, without markers.
//!
//! When a request names one function (`tool_choice: {"type": "function", ...}`), the gateway
//! constrains the engine to that function's parameter schema, and the whole output is the arguments
//! of that one call. When it requires a call from the request's tools (`tool_choice: "required"`,
//! or an allowed-tools list in `required` mode), the output is a JSON array of call objects, each
//! with the function's `name` and its `parameters`. [`Constrained`] is the [`Parser`] for both. It
//! knows the shape from the request, so it needs no scanner, and it streams the arguments as the
//! model's own bytes, as every format does; the old gateway parsed the whole output once it had
//! ended and, for a forced function, streamed the whole output as one call's arguments without
//! looking at it. Where the gateway constrains a model through a structural tag instead, the output
//! keeps the model's own markers, and the model's format parses it.
//!
//! One function: the bytes before the value are whitespace and dropped as such; the call starts at
//! the first byte of the value, named by the request (its `source` is empty, since no output byte
//! names it), and its argument fragments are the output's bytes for as long as they keep the
//! arguments a valid JSON prefix; from the first byte that does not, the bytes come back as
//! `Malformed` with `InvalidArguments`. Once the value is whole, the whitespace after it is
//! dropped, and from the first byte that is not whitespace everything to the end is `Malformed`
//! with `Other`. The call ends when the output ends, whole or not, so a value cut short is a call
//! with what arrived, as the assembler does for an object that never closed.
//!
//! Required: the list's brackets and commas are `Dropped { Wrapper }`, the whitespace between them
//! `Dropped { Whitespace }`, and each object goes to an [`Assembler`], which emits the call's
//! events and says where the object ended. The list's syntax is checked as JSON has it: a call or
//! the list's end after the opening bracket, a comma or the end after a call, a call after a comma.
//! An output that does not begin with a list, bytes the syntax does not allow where they stand, and
//! bytes after the list that are not whitespace come back as `Malformed` with `Other`, from the
//! first such byte to the end. A list cut short ends with what arrived: the assembler closes a
//! started call, and an object that never named a call comes back as `Malformed`.
//!
//! Every byte of the output lands in exactly one event; every text event carries its token count
//! from the [`Ledger`]; call ids are `call_<index>`, as in every format until the id scheme is
//! decided; the engine's finish reason is kept, with `tool_calls` counting the calls that started.
//! Reasoning cannot occur, since the grammar leaves no room for it, and the prompt is accepted
//! first in the lifecycle and otherwise ignored.

use super::finish_reason;
use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    input::{EngineFinish, Input},
    json::{is_complete, Assembler, PartialJson},
    parser::{ParseError, Parser},
    tokens::Ledger,
};

/// What the request's `tool_choice` forced, and so what shape the output has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    /// One function, named by the request: the whole output is its arguments.
    Function(String),
    /// At least one call from the request's tools (`tool_choice: "required"`, or an allowed-tools
    /// list in `required` mode): the output is a list of call objects.
    Required,
}

/// The parser for a grammar-constrained output. One per generated choice.
#[derive(Debug)]
pub struct Constrained {
    shape: Shape,
    tokens: Ledger,
    stage: Stage,
}

#[derive(Debug)]
enum Shape {
    Function(Single),
    Required(List),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Streaming,
    Ended,
}

/// The JSON whitespace that may separate the tokens of a constrained output.
fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl Constrained {
    /// A parser at the start of an output of the shape `choice` forced.
    pub fn new(choice: Choice) -> Self {
        Self {
            shape: match choice {
                Choice::Function(name) => Shape::Function(Single::new(name)),
                Choice::Required => Shape::Required(List::new()),
            },
            tokens: Ledger::new(),
            stage: Stage::Fresh,
        }
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        let mut made = Events::new();
        match &mut self.shape {
            Shape::Function(single) => single.feed(text, &mut made),
            Shape::Required(list) => list.feed(text, &mut made),
        }
        for event in made.drain() {
            out.push(self.tokens.relabel(event));
        }
    }

    fn end(&mut self, finish: EngineFinish, out: &mut Events) {
        let mut made = Events::new();
        let calls = match &mut self.shape {
            Shape::Function(single) => single.finish(&mut made),
            Shape::Required(list) => list.finish(&mut made),
        };
        for event in made.drain() {
            out.push(self.tokens.relabel(event));
        }
        self.tokens.finish(out);
        out.push(Event::Finish {
            reason: finish_reason(finish),
            tool_calls: calls,
            reasoning_tokens: 0,
        });
    }
}

impl Parser for Constrained {
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
            Input::Delta { text, spans, .. } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("delta after end".to_string()));
                }
                self.stage = Stage::Streaming;
                self.tokens.note(text, spans);
                self.text(text, out);
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

/// One forced function: the output is its arguments.
#[derive(Debug)]
struct Single {
    name: String,
    /// Whether the call has started, that is, whether the value's first byte arrived.
    started: bool,
    /// The value's bytes so far.
    value: String,
    /// Bytes of `value` accounted for so far, as fragments, malformed text or trailing bytes.
    emitted: usize,
    /// Where the value stopped being a valid prefix, if it did.
    invalid_from: Option<usize>,
    /// Where the value ended, if it is whole; bytes after it are no arguments.
    whole_end: Option<usize>,
    /// Whether bytes other than whitespace have followed the whole value; from the first of them to
    /// the end, everything is malformed, whitespace included.
    junk: bool,
}

impl Single {
    fn new(name: String) -> Self {
        Self {
            name,
            started: false,
            value: String::new(),
            emitted: 0,
            invalid_from: None,
            whole_end: None,
            junk: false,
        }
    }

    fn feed(&mut self, mut bytes: &str, out: &mut Events) {
        if !self.started {
            let trimmed = bytes.trim_start_matches(json_space);
            let space = &bytes[..bytes.len() - trimmed.len()];
            if !space.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(space),
                    why: DropReason::Whitespace,
                });
            }
            if trimmed.is_empty() {
                return;
            }
            out.push(Event::ToolCallStart {
                index: 0,
                id: "call_0".to_string(),
                name: self.name.clone(),
                source: Text::default(),
            });
            self.started = true;
            bytes = trimmed;
        }
        self.value.push_str(bytes);
        self.account(out);
    }

    /// Accounts for the value's new bytes: a fragment while they keep the value a valid prefix,
    /// malformed text once the prefix has broken, trailing bytes once the value is whole.
    fn account(&mut self, out: &mut Events) {
        let valid_end = match (self.invalid_from, self.whole_end) {
            (Some(from), _) => from,
            (None, Some(end)) => end,
            (None, None) => match PartialJson::default().parse(&self.value, true) {
                Ok((_, consumed)) if consumed == self.value.len() => consumed,
                Ok((_, consumed)) if is_complete(&self.value[..consumed]) => {
                    *self.whole_end.insert(consumed)
                }
                Ok((_, consumed)) => *self.invalid_from.insert(consumed.max(self.emitted)),
                Err(_) => *self.invalid_from.insert(self.emitted),
            },
        };
        if self.emitted < valid_end {
            let fresh = &self.value[self.emitted..valid_end];
            out.push(Event::ToolCallArguments {
                index: 0,
                json: fresh.to_string(),
                source: Text::uncounted(fresh),
            });
            self.emitted = valid_end;
        }
        if self.emitted == self.value.len() {
            return;
        }
        let rest = &self.value[self.emitted..];
        if self.invalid_from.is_some() {
            out.push(Event::Malformed {
                text: Text::uncounted(rest),
                why: MalformedReason::InvalidArguments,
            });
        } else {
            let trimmed = if self.junk {
                rest
            } else {
                rest.trim_start_matches(json_space)
            };
            let space = &rest[..rest.len() - trimmed.len()];
            if !space.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(space),
                    why: DropReason::Whitespace,
                });
            }
            if !trimmed.is_empty() {
                self.junk = true;
                out.push(Event::Malformed {
                    text: Text::uncounted(trimmed),
                    why: MalformedReason::Other("bytes after the arguments".to_string()),
                });
            }
        }
        self.emitted = self.value.len();
    }

    /// The output ended: a started call ends with what arrived. Returns the calls made.
    fn finish(&mut self, out: &mut Events) -> u32 {
        if !self.started {
            return 0;
        }
        out.push(Event::ToolCallEnd {
            index: 0,
            source: Text::default(),
        });
        1
    }
}

/// A required call: the output is a list of call objects.
#[derive(Debug)]
struct List {
    phase: Phase,
    /// Calls that started, which is also the next call's index.
    calls: u32,
}

#[derive(Debug)]
enum Phase {
    /// Before the opening bracket.
    Before,
    /// Right after the opening bracket: a call or the list's end comes next.
    Opened,
    /// Inside an object.
    Object(Assembler),
    /// After a call: a comma or the list's end comes next.
    AfterCall,
    /// After a comma: a call comes next.
    AfterComma,
    /// After the closing bracket.
    Closed,
    /// The output left the shape; everything from here is malformed, for the reason given.
    Broken(&'static str),
}

impl List {
    fn new() -> Self {
        Self {
            phase: Phase::Before,
            calls: 0,
        }
    }

    fn feed(&mut self, bytes: &str, out: &mut Events) {
        let mut rest = bytes;
        while !rest.is_empty() {
            match &mut self.phase {
                Phase::Object(assembler) => {
                    let taken = assembler.feed(rest, out);
                    if assembler.done() {
                        if assembler.started() {
                            self.calls += 1;
                        }
                        self.phase = Phase::AfterCall;
                    }
                    rest = &rest[taken..];
                }
                Phase::Broken(why) => {
                    out.push(Event::Malformed {
                        text: Text::uncounted(rest),
                        why: MalformedReason::Other((*why).to_string()),
                    });
                    rest = "";
                }
                Phase::Before
                | Phase::Opened
                | Phase::AfterCall
                | Phase::AfterComma
                | Phase::Closed => {
                    let trimmed = rest.trim_start_matches(json_space);
                    if trimmed.len() < rest.len() {
                        out.push(Event::Dropped {
                            text: Text::uncounted(&rest[..rest.len() - trimmed.len()]),
                            why: DropReason::Whitespace,
                        });
                        rest = trimmed;
                        continue;
                    }
                    let next = rest.chars().next();
                    let (phase, wrapper) = match (&self.phase, next) {
                        (Phase::Before, Some('[')) => (Phase::Opened, true),
                        (Phase::Before, _) => {
                            (Phase::Broken("the output is not a list of calls"), false)
                        }
                        (Phase::Opened | Phase::AfterComma, Some('{')) => (
                            Phase::Object(Assembler::new(
                                self.calls,
                                format!("call_{}", self.calls),
                            )),
                            false,
                        ),
                        (Phase::Opened | Phase::AfterCall, Some(']')) => (Phase::Closed, true),
                        (Phase::AfterCall, Some(',')) => (Phase::AfterComma, true),
                        (Phase::Opened, _) => (
                            Phase::Broken("bytes where a call or the list's end should be"),
                            false,
                        ),
                        (Phase::AfterComma, _) => {
                            (Phase::Broken("bytes where a call should be"), false)
                        }
                        (Phase::AfterCall, _) => (
                            Phase::Broken("bytes where a comma or the list's end should be"),
                            false,
                        ),
                        (_, _) => (Phase::Broken("bytes after the list"), false),
                    };
                    if wrapper {
                        out.push(Event::Dropped {
                            text: Text::uncounted(&rest[..1]),
                            why: DropReason::Wrapper,
                        });
                        rest = &rest[1..];
                    }
                    self.phase = phase;
                }
            }
        }
    }

    /// The output ended: an open object ends with what arrived. Returns the calls made.
    fn finish(&mut self, out: &mut Events) -> u32 {
        if let Phase::Object(assembler) = std::mem::replace(&mut self.phase, Phase::Closed) {
            if assembler.started() {
                self.calls += 1;
            }
            assembler.finish(out);
        }
        self.calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::FinishReason;

    fn run(choice: Choice, pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Constrained::new(choice);
        let mut out = Events::new();
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
        out.drain()
    }

    fn function(pieces: &[&str]) -> Vec<Event> {
        run(
            Choice::Function("get_weather".to_string()),
            pieces,
            EngineFinish::Stop,
        )
    }

    fn required(pieces: &[&str]) -> Vec<Event> {
        run(Choice::Required, pieces, EngineFinish::Stop)
    }

    fn start(index: u32, name: &str) -> Event {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.to_string(),
            source: Text::default(),
        }
    }

    fn fragment(index: u32, json: &str) -> Event {
        Event::ToolCallArguments {
            index,
            json: json.to_string(),
            source: Text::uncounted(json),
        }
    }

    fn end(index: u32) -> Event {
        Event::ToolCallEnd {
            index,
            source: Text::default(),
        }
    }

    fn dropped(text: &str, why: DropReason) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why,
        }
    }

    fn finish(tool_calls: u32) -> Event {
        Event::Finish {
            reason: FinishReason::Stop,
            tool_calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn a_forced_function_takes_the_whole_output_as_its_arguments() {
        assert_eq!(
            function(&["{\"city\": ", "\"Paris\"}"]),
            [
                start(0, "get_weather"),
                fragment(0, "{\"city\": "),
                fragment(0, "\"Paris\"}"),
                end(0),
                finish(1),
            ]
        );
    }

    #[test]
    fn whitespace_around_the_value_is_dropped_and_the_call_starts_at_the_value() {
        assert_eq!(
            function(&["  \n", "{\"a\": 1}", "\n"]),
            [
                dropped("  \n", DropReason::Whitespace),
                start(0, "get_weather"),
                fragment(0, "{\"a\": 1}"),
                dropped("\n", DropReason::Whitespace),
                end(0),
                finish(1),
            ]
        );
    }

    #[test]
    fn a_value_cut_short_is_a_call_with_what_arrived() {
        let events = function(&["{\"a\": [1, 2"]);
        assert_eq!(
            events,
            [
                start(0, "get_weather"),
                fragment(0, "{\"a\": [1, 2"),
                end(0),
                finish(1),
            ]
        );
    }

    #[test]
    fn bytes_that_break_the_prefix_or_follow_a_whole_value_are_malformed() {
        let broken = function(&["{\"a\": 1, nope}"]);
        assert_eq!(broken[0], start(0, "get_weather"));
        assert_eq!(broken[1], fragment(0, "{\"a\": 1, "));
        assert_eq!(
            broken[2],
            Event::Malformed {
                text: Text::uncounted("nope}"),
                why: MalformedReason::InvalidArguments,
            }
        );
        let trailing = function(&["{\"a\": 1} and more"]);
        assert_eq!(trailing[1], fragment(0, "{\"a\": 1}"));
        assert_eq!(trailing[2], dropped(" ", DropReason::Whitespace));
        assert_eq!(
            trailing[3],
            Event::Malformed {
                text: Text::uncounted("and more"),
                why: MalformedReason::Other("bytes after the arguments".to_string()),
            }
        );
    }

    #[test]
    fn an_empty_or_blank_output_makes_no_call() {
        assert_eq!(function(&[]), [finish(0)]);
        assert_eq!(
            function(&[" \n"]),
            [dropped(" \n", DropReason::Whitespace), finish(0)]
        );
    }

    #[test]
    fn a_required_list_streams_each_call_and_drops_the_list_syntax() {
        let events = required(&[
            "[{\"name\": \"get_weather\", \"parameters\": {\"ci",
            "ty\": \"Paris\"}}, {\"name\": \"get_time\", \"parameters\": {}}]",
        ]);
        let kinds: Vec<String> = events
            .iter()
            .map(|event| match event {
                Event::ToolCallStart { index, name, .. } => format!("start {index} {name}"),
                Event::ToolCallArguments { index, json, .. } => format!("args {index} {json}"),
                Event::ToolCallEnd { index, .. } => format!("end {index}"),
                Event::Dropped { text, why } => format!("drop {why:?} {:?}", text.text),
                Event::Finish { tool_calls, .. } => format!("finish {tool_calls}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "drop Wrapper \"[\"",
                "start 0 get_weather",
                "args 0 {\"ci",
                "args 0 ty\": \"Paris\"}",
                "end 0",
                "drop Wrapper \",\"",
                "drop Whitespace \" \"",
                "start 1 get_time",
                "args 1 {}",
                "end 1",
                "drop Wrapper \"]\"",
                "finish 2",
            ]
        );
    }

    #[test]
    fn a_list_that_is_not_one_or_goes_on_after_its_end_is_malformed_from_there() {
        let not_a_list = required(&["{\"name\": \"f\"}"]);
        assert_eq!(
            not_a_list[0],
            Event::Malformed {
                text: Text::uncounted("{\"name\": \"f\"}"),
                why: MalformedReason::Other("the output is not a list of calls".to_string()),
            }
        );
        let after = required(&["[] junk"]);
        assert_eq!(after[2], dropped(" ", DropReason::Whitespace));
        assert_eq!(
            after[3],
            Event::Malformed {
                text: Text::uncounted("junk"),
                why: MalformedReason::Other("bytes after the list".to_string()),
            }
        );
        let not_objects = required(&["[1, 2]"]);
        assert_eq!(
            not_objects[1],
            Event::Malformed {
                text: Text::uncounted("1, 2]"),
                why: MalformedReason::Other(
                    "bytes where a call or the list's end should be".to_string()
                ),
            }
        );
    }

    #[test]
    fn a_comma_comes_between_calls_and_nowhere_else() {
        let call = "{\"name\": \"f\", \"parameters\": {}}";
        let malformed = |text: &str, why: &str| Event::Malformed {
            text: Text::uncounted(text),
            why: MalformedReason::Other(why.to_string()),
        };
        let a_call_or_the_end = "bytes where a call or the list's end should be";
        for (output, from, why, calls) in [
            (
                format!("[{call}{call}]"),
                format!("{call}]"),
                "bytes where a comma or the list's end should be",
                1,
            ),
            (
                format!("[,{call}]"),
                format!(",{call}]"),
                a_call_or_the_end,
                0,
            ),
            (
                format!("[{call},]"),
                "]".to_string(),
                "bytes where a call should be",
                1,
            ),
            ("[,,]".to_string(), ",,]".to_string(), a_call_or_the_end, 0),
        ] {
            let events = required(&[&output]);
            assert_eq!(
                events[events.len() - 2],
                malformed(&from, why),
                "{output}: {events:?}"
            );
            assert_eq!(events.last(), Some(&finish(calls)), "{output}");
        }
    }

    #[test]
    fn a_list_cut_short_ends_with_what_arrived() {
        let cut = required(&["[{\"name\": \"f\", \"parameters\": {\"a\": [1,"]);
        assert!(
            cut.iter().any(
                |event| matches!(event, Event::ToolCallStart { index: 0, name, .. } if name == "f")
            ),
            "{cut:?}"
        );
        assert!(
            cut.iter()
                .any(|event| matches!(event, Event::ToolCallEnd { index: 0, .. })),
            "{cut:?}"
        );
        assert_eq!(cut.last(), Some(&finish(1)));
        let nameless = required(&["[{\"parameters\": {}}"]);
        assert!(
            nameless
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{nameless:?}"
        );
        assert_eq!(nameless.last(), Some(&finish(0)));
    }

    #[test]
    fn the_engine_finish_reason_is_kept() {
        let events = run(
            Choice::Required,
            &["[{\"name\": \"f\", \"parameters\": {}}]"],
            EngineFinish::Length,
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Length,
                tool_calls: 1,
                ..
            })
        ));
    }
}
