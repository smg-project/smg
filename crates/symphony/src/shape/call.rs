//! Events from one call given as data, a [`CallShape`], as its bytes arrive.
//!
//! A table whose call syntax is [`CallSyntax::Shaped`] hands this assembler what follows the
//! region's terminal. A Hugging Face response template writes a call's region as patterns: the
//! open (`\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>` in the `transformers serve` template
//! for Qwen 3.5), the arguments, and the close (`</function>\s*</tool_call>`). The table's
//! terminals are the open's first literal and the close's last one; the shape holds the rest: the
//! header after the open's terminal, which may capture the tool's name, the way the arguments are
//! written, and the trailer before the close's terminal.
//!
//! - **The header** is read first, with a [`Matcher`], before the call opens: the engine holds the
//!   terminal and everything after it, and moves nothing, until the header is decided. When it
//!   matches ([`Header::Matched`]), `Assembler::start` starts the call (named by the header, by
//!   the template, or later by the arguments' own object) and gives back the bytes that followed
//!   the header, for the engine to read again in the call. When it cannot match
//!   ([`Header::Failed`]), the terminal opened no call: the engine takes back what the header
//!   read (`Assembler::into_header_bytes`) and reads the terminal and those bytes again from the
//!   terminal's second character, as a regular expression's search goes on from the place after
//!   one where its pattern failed. A header holds at most the shape's cap of bytes undecided; past
//!   it, it is no match.
//! - **The arguments** follow: one JSON value ([`json::Arguments`], the model's bytes streamed as
//!   fragments), one object that names the call ([`json::Assembler`]), or tagged parameters, read
//!   as `finditer` reads them, each written as a JSON member when its value ends. What comes after
//!   a value or an object is the trailer, or text that belongs to no part of the call; between
//!   parameters it is skipped.
//! - **The trailer** is checked when a closing terminal comes ([`Assembler::closes`]): the region
//!   closes there only when the trailer ends right before the terminal, starting no earlier than
//!   the end of what the arguments took, and the earliest such start is the trailer's
//!   ([`Suffix`]). A terminal the trailer does not end at closes nothing: the engine hands it to
//!   the call as its text, as the template's closing pattern reads past it. An empty trailer ends
//!   wherever the terminal comes. Until the region closes, the bytes after the arguments are
//!   held, since they may turn out to be the trailer; with an empty trailer they are classified as
//!   they arrive. The stream's stop, where the table says it closes the call, is checked the same
//!   way for one JSON value or object, whose text the template's reader parses whole; tagged
//!   parameters close there wherever the trailer stands, since that reader closes the region at the
//!   end of its text and `finditer` passes over what no parameter takes.
//!
//! Every byte lands in one event. The header's bytes are the start's source (dropped as wrapping
//! when the arguments' object names the call, since its start carries that object's head); the
//! arguments' bytes are their fragments' sources; the trailer and the closing terminal are the
//! end's source. Whitespace after the arguments that is not the trailer is dropped as wrapping, and
//! anything else there is `Malformed`, run by run, so that where the chunks were cut changes
//! nothing. A region the stream cut is ended with what arrived: the call is not closed for the
//! client, and what was held comes back as `Malformed` with `UnterminatedRegion`, its whitespace
//! dropped.
//!
//! [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped

use super::params::Params;
use crate::{
    engine::TEXT_AFTER_THE_OBJECT,
    event::{DropReason, Event, Events, MalformedReason, Text},
    json::{self, arguments::Strings},
    shape::{
        Arguments, CallShape, Match, Matcher, NameSource, Schema, Slot, Suffix, ValueParser,
        Values, Verdict,
    },
    tagged::Declared,
};

const TEXT_AFTER_THE_ARGUMENTS: &str = "text after a call's arguments";

/// What a call's header says, once the bytes fed after the terminal decide it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// More bytes can still change the answer.
    Pending,
    /// The header matched: `Assembler::start` starts the call.
    Matched,
    /// The header cannot match: the terminal opened no call, and
    /// `Assembler::into_header_bytes` gives back what the header read.
    Failed,
}

/// The events of one shaped call, from the bytes after its region's terminal.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    shape: &'static CallShape,
    /// How the call's values read.
    values: Values,
    /// The terminal's last character, which a `\b` at the header's start reads.
    before: Option<char>,
    stage: Stage,
}

#[derive(Clone, Debug)]
enum Stage {
    /// The header is being read; the matcher holds every byte after the terminal.
    Header(Box<Matcher<'static>>),
    /// The header matched: what it read, where it ended, and the name it gave; the call starts
    /// when the engine has moved into it.
    Matched { held: String, found: Match },
    /// The call has started: its arguments, and what follows them.
    Body(Body),
    /// The call has ended.
    Done,
}

/// A started call: its arguments, and what follows them.
#[derive(Clone, Debug)]
enum Body {
    /// One JSON value or one object.
    Value(Box<Value>),
    /// Tagged parameters, which keep their own trailer's search, since the text after the last
    /// parameter moves with every parameter.
    Params(Box<Params>),
}

/// Arguments written as one JSON value or one object, and what follows them.
#[derive(Clone, Debug)]
struct Value {
    reader: JsonReader,
    /// The call's name, whose tool's schema types a member the table's reader holds; empty while an
    /// object that names the call has not done so.
    name: String,
    /// The last character the call took, which a `\b` at the start of the trailer reads.
    before: Option<char>,
    /// Bytes after the arguments, held until the region closes, since they may be the trailer.
    after: String,
    /// The trailer's search over `after`, from the arguments' end on.
    tail: Option<Suffix<'static>>,
    /// Where the trailer starts in `after`, once a closing terminal was found to end it.
    trailer_at: Option<usize>,
}

/// How the arguments are read.
#[derive(Clone, Debug)]
enum JsonReader {
    /// One JSON value, the call already started.
    Json(json::Arguments),
    /// One object that names the call and holds its arguments.
    Envelope(json::Assembler),
}

impl JsonReader {
    /// Whether the arguments are over: the value is whole or ended where refused, or the object is
    /// done.
    fn over(&self) -> bool {
        match self {
            Self::Json(arguments) => arguments.over(),
            Self::Envelope(assembler) => assembler.done(),
        }
    }

    /// The bytes a refused value or object took before the refusal was known, handed back once
    /// with the character before them: what follows the arguments, for the trailer to read.
    fn refused_tail(&mut self) -> Option<(String, Option<char>)> {
        match self {
            Self::Json(arguments) => arguments.refused_tail(),
            Self::Envelope(assembler) => assembler.refused_tail(),
        }
    }

    /// Why text after the arguments that is not whitespace is malformed: for a value or an object
    /// that ended where its arguments were refused, its invalid arguments, since the text continues
    /// them; else the reason the JSON syntax gives for text between a call's object and its closing
    /// marker, for an object that names the call (one that closed after its arguments broke
    /// included), or for text after the arguments.
    fn surplus(&self) -> MalformedReason {
        let ended_where_refused = match self {
            Self::Json(arguments) => arguments.ended_where_refused(),
            Self::Envelope(assembler) => assembler.ended_where_refused(),
        };
        if ended_where_refused {
            return MalformedReason::InvalidArguments;
        }
        MalformedReason::Other(
            match self {
                Self::Json(_) => TEXT_AFTER_THE_ARGUMENTS,
                Self::Envelope(_) => TEXT_AFTER_THE_OBJECT,
            }
            .to_string(),
        )
    }
}

impl Assembler {
    /// An assembler for the call at `index`, with the id the format minted for it, of `shape`,
    /// whose values read with `values`, right after its region's terminal, whose last character is
    /// `before`.
    pub(crate) fn new(
        index: u32,
        id: impl Into<String>,
        shape: &'static CallShape,
        values: Values,
        before: Option<char>,
    ) -> Self {
        let matcher =
            Matcher::new(&shape.header, &[], &shape.word, before).hold_cap(Some(shape.cap()));
        Self {
            index,
            id: id.into(),
            shape,
            values,
            before,
            stage: Stage::Header(Box::new(matcher)),
        }
    }

    /// Read the next bytes after the terminal into the header, and say what they decide. Once
    /// decided, the answer stands, and more bytes are not read.
    pub fn header(&mut self, bytes: &str) -> Header {
        match &mut self.stage {
            Stage::Header(matcher) => {
                let verdict = matcher.feed(bytes);
                self.decided(verdict)
            }
            Stage::Matched { .. } | Stage::Body(_) | Stage::Done => Header::Matched,
        }
    }

    /// No more bytes come: what the bytes read decide.
    pub fn header_end(&mut self) -> Header {
        match &mut self.stage {
            Stage::Header(matcher) => {
                let verdict = matcher.finish();
                self.decided(verdict)
            }
            Stage::Matched { .. } | Stage::Body(_) | Stage::Done => Header::Matched,
        }
    }

    fn decided(&mut self, verdict: Verdict) -> Header {
        match verdict {
            Verdict::Pending => Header::Pending,
            Verdict::NoMatch => Header::Failed,
            Verdict::Matched(found) => {
                let Stage::Header(matcher) = std::mem::replace(&mut self.stage, Stage::Done) else {
                    return Header::Matched;
                };
                self.stage = Stage::Matched {
                    held: matcher.into_held(),
                    found,
                };
                Header::Matched
            }
        }
    }

    /// What the header read, given back when it did not match.
    pub fn into_header_bytes(self) -> String {
        match self.stage {
            Stage::Header(matcher) => matcher.into_held(),
            Stage::Matched { held, .. } => held,
            Stage::Body(_) | Stage::Done => String::new(),
        }
    }

    /// The header matched and the engine has moved into the call: push its start, named by the
    /// header or the template, with the header's bytes as its source (or the header's bytes as
    /// wrapping, when the arguments' own object names the call and its start comes with that
    /// object's head). Returns the bytes that followed the header, which the engine reads again,
    /// now inside the call.
    pub fn start(&mut self, out: &mut Events) -> String {
        let Stage::Matched { held, found } = std::mem::replace(&mut self.stage, Stage::Done) else {
            return String::new();
        };
        let end = found.end();
        let header = &held[..end];
        let before = header.chars().next_back().or(self.before);
        let shape = self.shape;
        if let Arguments::Envelope { .. } = &shape.arguments {
            if !header.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(header),
                    why: DropReason::Wrapper,
                });
            }
            // With a trailer, an object ends where its bytes stop being a call's object (its
            // arguments refused as JSON, or its outline stopped short of the close) and the trailer
            // reads what follows; without one, the region's terminal closes the call wherever the
            // object stands, and the object keeps its malformed rest as before.
            let assembler = json::Assembler::new(self.index, self.id.clone());
            let reader = JsonReader::Envelope(if shape.trailer.0.is_empty() {
                assembler
            } else {
                assembler.ending_where_refused()
            });
            let value = Value::new(reader, String::new(), before);
            self.stage = Stage::Body(Body::Value(Box::new(value)));
            return held[end..].to_string();
        }
        let name = match &shape.name {
            NameSource::Literal(name) => name.to_string(),
            NameSource::Header | NameSource::Body => found
                .span(Slot::Name)
                .map(|span| held[span].to_string())
                .unwrap_or_default(),
        };
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(header),
        });
        let body = match &shape.arguments {
            Arguments::Params { tag, ends, values } => Body::Params(Box::new(Params::new(
                self.index,
                shape,
                (tag, ends, values),
                (name, self.values.clone()),
                before,
            ))),
            // As for an envelope above: only a trailer needs the value to end where refused.
            Arguments::Json(_) | Arguments::Envelope { .. } => {
                let arguments = json::Arguments::new(self.index);
                let reader = JsonReader::Json(if shape.trailer.0.is_empty() {
                    arguments
                } else {
                    arguments.ending_where_refused()
                });
                Body::Value(Box::new(Value::new(reader, name, before)))
            }
        };
        self.stage = Stage::Body(body);
        held[end..].to_string()
    }

    /// Append the next bytes of the call, after its start, and push the events they complete,
    /// a parameter's value typed by `declared`. Every byte is the call's until the engine closes
    /// the region.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        let shape = self.shape;
        let Stage::Body(body) = &mut self.stage else {
            return;
        };
        match body {
            Body::Value(value) => value.feed(shape, (&self.values, declared), bytes, out),
            Body::Params(params) => params.feed(bytes, declared, out),
        }
    }

    /// Whether the region closes at `terminal`, a closing terminal that just came (empty for the
    /// stream's stop): whether the trailer ends right before it, starting no earlier than where
    /// the arguments' bytes end. When it does not, the terminal closes nothing, and the engine
    /// feeds it to the call as its text. The stream's stop closes tagged parameters wherever the
    /// trailer stands.
    pub fn closes(&mut self, terminal: &str) -> bool {
        let shape = self.shape;
        let Stage::Body(body) = &mut self.stage else {
            return true;
        };
        match body {
            Body::Value(value) => value.closes(shape, terminal),
            Body::Params(params) => params.closes(terminal),
        }
    }

    /// The region closed at `terminal` (empty for the stream's stop), which [`closes`] said it
    /// does: the call ends. Bytes after a value or an object and before the trailer are dropped
    /// where they are whitespace and malformed otherwise, and parameters still undecided are
    /// decided as at the end of the call's text; the trailer and the terminal are the end's
    /// source. Returns whether the terminal was taken so; when the arguments' own object named the
    /// call, its close was the call's end, and the trailer and the terminal are the region's.
    ///
    /// [`closes`]: Self::closes
    pub fn close(mut self, terminal: &str, declared: &Declared, out: &mut Events) -> bool {
        let Stage::Body(body) = std::mem::replace(&mut self.stage, Stage::Done) else {
            return false;
        };
        let trailer = match body {
            Body::Value(value) => match value.close(out) {
                Some(trailer) => trailer,
                None => return false,
            },
            Body::Params(params) => params.close(declared, out),
        };
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(format!("{trailer}{terminal}")),
        });
        true
    }

    /// The stream was cut inside the region: the call ends with what arrived, not closed for the
    /// client. What was held after a value or an object comes back as `Malformed` with
    /// `UnterminatedRegion`, its whitespace dropped; parameters still undecided are decided as at
    /// the end of what arrived, and the bytes no parameter took come back the same way.
    pub fn finish(mut self, declared: &Declared, out: &mut Events) {
        let Stage::Body(body) = std::mem::replace(&mut self.stage, Stage::Done) else {
            return;
        };
        let ends = match body {
            Body::Value(value) => value.finish(out),
            Body::Params(params) => {
                params.finish(declared, out);
                true
            }
        };
        if ends {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
        }
    }
}

/// The table's reader, asked about the members of one call's JSON arguments.
struct Typing<'a> {
    values: &'a Values,
    declared: &'a Declared,
    name: &'a str,
}

impl Typing<'_> {
    fn schema(&self, key: &str) -> Option<Schema<'_>> {
        self.declared
            .parameter(self.name, key)
            .map(|(root, property)| Schema { root, property })
    }
}

impl Strings for Typing<'_> {
    fn holds(&self, key: &str) -> bool {
        !self
            .values
            .reader()
            .streams(&ValueParser::json(), self.schema(key))
    }

    fn typed(&self, key: &str, raw: &str) -> Option<String> {
        self.values.reader().member(raw, self.schema(key))
    }
}

impl Value {
    fn new(reader: JsonReader, name: String, before: Option<char>) -> Self {
        Self {
            reader,
            name,
            before,
            after: String::new(),
            tail: None,
            trailer_at: None,
        }
    }

    fn feed(
        &mut self,
        shape: &'static CallShape,
        (values, declared): (&Values, &Declared),
        bytes: &str,
        out: &mut Events,
    ) {
        let mut rest = bytes;
        let handed_back;
        // What the call took before this piece: the character before a value refused at its first
        // byte, which has none of its own.
        let before_feed = self.before;
        if !self.reader.over() {
            let typing = Typing {
                values,
                declared,
                name: &self.name,
            };
            let taken = match &mut self.reader {
                JsonReader::Json(arguments) => arguments.feed(rest, &typing, out),
                JsonReader::Envelope(assembler) => assembler.feed(rest, out),
            };
            if let Some(last) = rest[..taken].chars().next_back() {
                self.before = Some(last);
            }
            rest = &rest[taken..];
            if !self.reader.over() {
                return;
            }
            // A reader that ends where its bytes stop being a value or an object hands the bytes it
            // took past that point back: they lead what follows, so the trailer reads them too,
            // after the character before them.
            if let Some((tail, before)) = self.reader.refused_tail() {
                self.before = before.or(before_feed);
                handed_back = format!("{tail}{rest}");
                rest = &handed_back;
            }
        }
        if shape.trailer.0.is_empty() {
            wrapping(rest, &self.reader.surplus(), out);
            return;
        }
        let tail = self.tail.get_or_insert_with(|| {
            Suffix::new(&shape.trailer, &shape.word, self.before).hold_cap(Some(shape.cap()))
        });
        tail.feed(rest);
        self.after.push_str(rest);
    }

    fn closes(&mut self, shape: &'static CallShape, terminal: &str) -> bool {
        let at = if shape.trailer.0.is_empty() {
            Some(self.after.len())
        } else {
            let next = terminal.chars().next();
            match &mut self.tail {
                Some(tail) => tail.start(next),
                // Nothing has come after the arguments: only a trailer that matches no text can
                // end here.
                None => Suffix::new(&shape.trailer, &shape.word, self.before)
                    .hold_cap(Some(shape.cap()))
                    .start(next),
            }
        };
        self.trailer_at = at;
        at.is_some()
    }

    /// The region closed: the bytes after the arguments and before the trailer are classified,
    /// and the trailer is returned for the call's end; or, for an object that named the call and
    /// ended it at its close, the trailer is the region's wrapping and nothing is returned.
    fn close(self, out: &mut Events) -> Option<String> {
        let Self {
            reader,
            after,
            trailer_at,
            ..
        } = self;
        let at = trailer_at.unwrap_or(after.len());
        let (between, trailer) = after.split_at(at);
        let surplus = reader.surplus();
        let enveloped = match reader {
            JsonReader::Json(mut arguments) => {
                arguments.finish(out);
                false
            }
            JsonReader::Envelope(assembler) => {
                assembler.finish(out);
                true
            }
        };
        wrapping(between, &surplus, out);
        if enveloped {
            if !trailer.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(trailer),
                    why: DropReason::Wrapper,
                });
            }
            return None;
        }
        Some(trailer.to_string())
    }

    /// The stream was cut: what was held after the arguments is malformed, and the call's end is
    /// the caller's to push, unless the arguments' object ended the call. Returns whether the
    /// caller ends it.
    fn finish(self, out: &mut Events) -> bool {
        match self.reader {
            JsonReader::Json(mut arguments) => {
                arguments.finish(out);
                wrapping(&self.after, &MalformedReason::UnterminatedRegion, out);
                true
            }
            JsonReader::Envelope(assembler) => {
                assembler.finish(out);
                wrapping(&self.after, &MalformedReason::UnterminatedRegion, out);
                false
            }
        }
    }
}

/// Bytes after a call's arguments that are no part of the call: each run of whitespace dropped as
/// wrapping, each run of anything else `Malformed` for `why`. Classifying run by run keeps the
/// result the same wherever the chunks were cut.
fn wrapping(text: &str, why: &MalformedReason, out: &mut Events) {
    let mut rest = text;
    while let Some(first) = rest.chars().next() {
        let space = first.is_whitespace();
        let length = rest
            .char_indices()
            .find(|(_, c)| c.is_whitespace() != space)
            .map_or(rest.len(), |(at, _)| at);
        let run = Text::uncounted(&rest[..length]);
        out.push(if space {
            Event::Dropped {
                text: run,
                why: DropReason::Wrapper,
            }
        } else {
            Event::Malformed {
                text: run,
                why: why.clone(),
            }
        });
        rest = &rest[length..];
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        engine::Engine,
        format::{CallSyntax, Emits, Format},
        input::{EngineFinish, Input, TokenSpan},
        parser::Parser,
        shape::{
            intern, Atom, Class, Dialect, Duplicates, End, Path, Pattern, ShapeError, ValueRead,
        },
    };

    fn word() -> Class {
        Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
    }

    fn space() -> Class {
        Class::new([('\t', '\r'), (' ', ' ')], false)
    }

    fn lit(text: &str) -> Atom {
        Atom::Lit(text.into())
    }

    /// `<n=NAME>` on abstract tags: the header of the shapes here.
    fn named_header() -> Pattern {
        Pattern(
            [
                lit("<n="),
                Atom::Run {
                    class: word(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                lit(">"),
            ]
            .into(),
        )
    }

    /// A call named by `<n=NAME>` after its terminal, its arguments one JSON value, with
    /// `trailer` before its closing terminal.
    fn named(trailer: Pattern) -> &'static CallShape {
        intern(CallShape {
            header: named_header(),
            name: NameSource::Header,
            arguments: Arguments::Json(Dialect::default()),
            trailer,
            word: word(),
            hold_cap: 64,
        })
    }

    /// `</f>` and the whitespace after it: a trailer on abstract tags.
    fn closing_f() -> Pattern {
        Pattern(
            [
                lit("</f>"),
                Atom::Run {
                    class: space(),
                    min: 0,
                    max: None,
                    greedy: true,
                    capture: None,
                },
            ]
            .into(),
        )
    }

    /// A call region between `<c>` and `</c>` beside a thought between `<t>` and `</t>`, the
    /// calls given by `shape`.
    fn table(shape: &'static CallShape) -> Format {
        Format::new("shaped")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "call")
            .transition("reasoning", "call_open", "call")
            .transition("call", "call_close", "content")
            .calls(CallSyntax::Shaped(shape))
    }

    /// Events with adjacent text of one kind joined, so that chunkings compare by what they say.
    fn joined(events: Vec<Event>) -> Vec<Event> {
        let mut out: Vec<Event> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (
                    Some(Event::Dropped { text, why }),
                    Event::Dropped {
                        text: more,
                        why: again,
                    },
                ) if *why == again => text.text.push_str(&more.text),
                (
                    Some(Event::Malformed { text, why }),
                    Event::Malformed {
                        text: more,
                        why: again,
                    },
                ) if *why == again => text.text.push_str(&more.text),
                (
                    Some(Event::ToolCallArguments {
                        index,
                        json,
                        source,
                    }),
                    Event::ToolCallArguments {
                        index: other,
                        json: more,
                        source: bytes,
                    },
                ) if *index == other => {
                    json.push_str(&more);
                    source.text.push_str(&bytes.text);
                }
                (Some(Event::Content(text)), Event::Content(more))
                | (Some(Event::Reasoning(text)), Event::Reasoning(more)) => {
                    text.text.push_str(&more.text);
                }
                (_, event) => out.push(event),
            }
        }
        out
    }

    /// The events of `output` after `prompt`, the same whole, cut once at every place and a
    /// character at a time, every byte accounted for.
    fn run_after(format: &Format, prompt: &str, output: &str, finish: EngineFinish) -> Vec<Event> {
        run_declared(format, &Declared::default(), prompt, output, finish)
    }

    /// [`run_after`] for a request whose tools are `declared`.
    fn run_declared(
        format: &Format,
        declared: &Declared,
        prompt: &str,
        output: &str,
        finish: EngineFinish,
    ) -> Vec<Event> {
        let starts: Vec<usize> = output.char_indices().map(|(at, _)| at).skip(1).collect();
        let mut plans: Vec<Vec<usize>> = vec![vec![], starts.clone()];
        plans.extend(starts.iter().map(|&cut| vec![cut]));
        let mut runs = plans.iter().map(|cuts| {
            let mut engine = Engine::new(format.clone(), declared.clone());
            let mut out = Events::new();
            engine
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: prompt,
                    },
                    &mut out,
                )
                .expect("the prompt");
            let mut from = 0;
            for &cut in cuts.iter().chain([&output.len()]) {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: &output[from..cut],
                            spans: &[],
                        },
                        &mut out,
                    )
                    .expect("a delta");
                from = cut;
            }
            engine
                .feed(
                    Input::End {
                        finish: finish.clone(),
                    },
                    &mut out,
                )
                .expect("the end");
            assert_eq!(out.text(), output, "every byte, cut at {cuts:?}");
            joined(out.drain())
        });
        let whole = runs.next().expect("the whole output is one plan");
        for (events, cuts) in runs.zip(plans.iter().skip(1)) {
            assert_eq!(events, whole, "cut at {cuts:?}");
        }
        whole
    }

    fn run(format: &Format, output: &str) -> Vec<Event> {
        run_after(format, "", output, EngineFinish::Stop)
    }

    fn content(text: &str) -> Event {
        Event::Content(Text::uncounted(text))
    }

    fn wrapper(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    fn start(name: &str, source: &str) -> Event {
        Event::ToolCallStart {
            index: 0,
            id: "call_0".to_string(),
            name: name.to_string(),
            source: Text::uncounted(source),
        }
    }

    fn arguments(json: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(json),
        }
    }

    fn end(source: &str) -> Event {
        Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source),
        }
    }

    fn finish(calls: u32) -> Event {
        Event::Finish {
            reason: crate::event::FinishReason::Stop,
            tool_calls: calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn a_header_names_the_call_and_the_closing_terminal_ends_it() {
        let events = run(
            &table(named(Pattern::default())),
            "Sure.<c><n=get_weather>{\"city\": \"Paris\"}\n</c>Done.",
        );
        assert_eq!(
            events,
            [
                content("Sure."),
                wrapper("<c>"),
                start("get_weather", "<n=get_weather>"),
                arguments("{\"city\": \"Paris\"}"),
                wrapper("\n"),
                end("</c>"),
                content("Done."),
                finish(1),
            ]
        );
    }

    #[test]
    fn a_terminal_whose_header_does_not_match_is_the_text_where_it_came() {
        let format = table(named(Pattern::default()));
        assert_eq!(
            run(&format, "Hi <c><x>bye"),
            [content("Hi <c><x>bye"), finish(0)],
            "the search goes on after the terminal's first character"
        );
        assert_eq!(
            run(&format, "<t>plan <c>oops</t>ok"),
            [
                wrapper("<t>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("plan <c>oops")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                content("ok"),
                finish(0),
            ],
            "in a thought, the thought's text, and its close is still its close"
        );
        assert_eq!(
            run(&format, "<c><c><n=f>{}</c>"),
            [
                content("<c>"),
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "the next terminal after the first character opens the call"
        );
        assert_eq!(
            run(&format, "<c><n=f<c><n=g>{}</c>"),
            [
                content("<c><n=f"),
                wrapper("<c>"),
                start("g", "<n=g>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "a terminal inside a header is the header's text until the header fails"
        );
    }

    #[test]
    fn the_end_of_the_stream_decides_a_header_still_being_read() {
        let format = table(named(Pattern::default()));
        assert_eq!(run(&format, "<c><n=f"), [content("<c><n=f"), finish(0)]);
        let events = run_after(&format, "", "<c><n=f>", EngineFinish::Length);
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                end(""),
                Event::Finish {
                    reason: crate::event::FinishReason::Length,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ],
            "a header that matched at the end starts a call the end cut"
        );
    }

    #[test]
    fn a_given_back_terminal_is_read_by_the_rows_of_the_state_it_came_in() {
        let trimmed = table(named(Pattern::default())).trims("content");
        assert_eq!(
            run(&trimmed, " Hi <c>x "),
            [
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Whitespace,
                },
                content("Hi <c>x"),
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Whitespace,
                },
                finish(0),
            ],
            "the space held before the terminal was inside the text"
        );
        let framed = table(named(Pattern::default())).framing("content");
        assert_eq!(
            run(&framed, "x<c>y<c><n=f>{}</c>"),
            [
                wrapper("x<c>y<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ]
        );
        let starts_in_thought = table(named(Pattern::default()))
            .framing("content")
            .starts_in("reasoning");
        assert_eq!(
            run(&starts_in_thought, "<c>plan</t>"),
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("<c>plan")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                finish(0),
            ],
            "a terminal that opens no call is the turn's first text, in the start row's state"
        );
        assert_eq!(
            run(&starts_in_thought, "<c><n=f>{}</c>"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "a call at the turn's first byte opens from the turn's own state"
        );
    }

    #[test]
    fn a_turns_first_terminal_opens_its_call_from_the_state_the_turn_starts_in() {
        // The call's row leaves the start state only, not the first one.
        let format = Format::new("shaped")
            .terminal("think_close", "</t>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("reasoning", "think_close", "outside")
            .transition("reasoning", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Shaped(named(Pattern::default())))
            .framing("outside")
            .starts_in("reasoning");
        assert_eq!(
            run(&format, "<c><n=f>{}</c>"),
            [
                Event::ReasoningStart,
                wrapper("<c>"),
                Event::ReasoningEnd,
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "the turn starts where its first piece says, and the call opens from there"
        );
        assert_eq!(
            run(&format, "<c>x</t>"),
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("<c>x")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                finish(0),
            ],
            "a header that does not match leaves the terminal to the start state's text"
        );
    }

    #[test]
    fn a_shaped_call_has_no_structural_tag_yet() {
        let tools = [openai_protocol::common::Tool {
            tool_type: "function".to_string(),
            function: openai_protocol::common::Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let format = table(named(Pattern::default()));
        for (at_least_one, reasoning_open) in [(true, false), (false, false), (true, true)] {
            assert_eq!(format.grammar(&tools, at_least_one, reasoning_open), None);
        }
    }

    #[test]
    fn a_closing_terminal_closes_only_where_the_trailer_ends_before_it() {
        let format = table(named(closing_f()));
        assert_eq!(
            run(&format, "<c><n=f>{\"a\": 1} </f>\n</c>Done."),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{\"a\": 1}"),
                wrapper(" "),
                end("</f>\n</c>"),
                content("Done."),
                finish(1),
            ]
        );
        let events = run_after(
            &format,
            "",
            "<c><n=f>{\"a\": 1}</c>x</f></c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{\"a\": 1}"),
                Event::Malformed {
                    text: Text::uncounted("</c>x"),
                    why: MalformedReason::Other(TEXT_AFTER_THE_ARGUMENTS.to_string()),
                },
                end("</f></c>"),
                finish(1),
            ],
            "a close without the trailer is the call's text"
        );
        let events = run_after(&format, "", "<c><n=f>{}</c>", EngineFinish::Length);
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                Event::Malformed {
                    text: Text::uncounted("</c>"),
                    why: MalformedReason::UnterminatedRegion,
                },
                end(""),
                Event::Finish {
                    reason: crate::event::FinishReason::Length,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ],
            "a region the trailer never closed is cut by the end"
        );
    }

    /// A value the JSON reader refuses is over where it was refused: what follows is the
    /// trailer's to read, so the closing terminal the trailer ends before closes the call, and the
    /// text after it is content. Without this, the refused value took every byte to the stream's
    /// stop and the close came back as the call's text (#3145's unresolved finding).
    #[test]
    fn a_refused_value_hands_what_follows_to_the_trailer() {
        let format = table(named(closing_f()));
        assert_eq!(
            run(&format, "<c><n=f>{\"a\": 1</f>\n</c>Done."),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{\"a\": 1"),
                end("</f>\n</c>"),
                content("Done."),
                finish(1),
            ]
        );
        // A value refused at its first byte has no character of its own before the tail: the
        // header's last character, or the whitespace before the value, stands before the trailer,
        // which a `\b` at its start reads.
        let format = table(named(Pattern([Atom::Boundary, lit("END")].into())));
        assert_eq!(
            run(&format, "<c><n=f>END</c>Done."),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                end("END</c>"),
                content("Done."),
                finish(1),
            ]
        );
        assert_eq!(
            run(&format, "<c><n=f> END</c>Done."),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Whitespace,
                },
                end("END</c>"),
                content("Done."),
                finish(1),
            ]
        );
    }

    /// The same for an object that names the call: refused inside its arguments, or its outline
    /// stopped short of its close (the brace left off, text where a member should be, a literal
    /// in another member running into the trailer), the call ends there, the trailer and the
    /// terminal are the region's wrapping, and the text after is content.
    #[test]
    fn a_refused_envelope_hands_what_follows_to_the_trailer() {
        let envelope = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(["name"]),
                arguments: Path::new(["arguments"]),
            },
            trailer: closing_f(),
            word: word(),
            hold_cap: 64,
        });
        let format = table(envelope);
        let events = run(
            &format,
            "<c>{\"name\": \"f\", \"arguments\": {\"a\": 1</f>\n</c>Done.",
        );
        let ended = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { .. }))
            .expect("the call ends");
        let content_at = events
            .iter()
            .position(|event| matches!(event, Event::Content(_)))
            .expect("the text after the call is content");
        assert!(ended < content_at, "{events:#?}");
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{events:#?}"
        );
        let shown: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, "Done.", "{events:#?}");
        assert_eq!(
            run(
                &format,
                "<c>{\"name\": \"f\", \"arguments\": {\"a\": 1}</f>\n</c>Done.",
            ),
            [
                wrapper("<c>"),
                start("f", "{\"name\": \"f\", \"arguments\": "),
                arguments("{\"a\": 1}"),
                end(""),
                wrapper("</f>\n</c>"),
                content("Done."),
                finish(1),
            ],
            "the brace left off"
        );
        assert_eq!(
            run(
                &format,
                "<c>{\"name\": \"f\", \"arguments\": {\"a\": 1}, junk</f>\n</c>Done.",
            ),
            [
                wrapper("<c>"),
                start("f", "{\"name\": \"f\", \"arguments\": "),
                arguments("{\"a\": 1}"),
                end(", "),
                Event::Malformed {
                    text: Text::uncounted("junk"),
                    why: MalformedReason::Other(TEXT_AFTER_THE_OBJECT.to_string()),
                },
                wrapper("</f>\n</c>"),
                content("Done."),
                finish(1),
            ],
            "text where a member should be"
        );
        assert_eq!(
            run(
                &format,
                "<c>{\"name\": \"f\", \"arguments\": {\"a\": 1}, \"strict\": true</f>\n</c>Done.",
            ),
            [
                wrapper("<c>"),
                start("f", "{\"name\": \"f\", \"arguments\": "),
                arguments("{\"a\": 1}"),
                end(", \"strict\": true"),
                wrapper("</f>\n</c>"),
                content("Done."),
                finish(1),
            ],
            "a literal in another member runs into the trailer"
        );
    }

    #[test]
    fn a_stop_closes_the_call_where_the_trailer_ends_with_the_stream() {
        let format = table(named(closing_f())).closes_at_stop("call");
        assert_eq!(
            run(&format, "<c><n=f>{}</f>"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</f>"),
                finish(1),
            ]
        );
        assert_eq!(
            run(&format, "<c><n=f>{}"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end(""),
                finish(1),
            ],
            "with no trailer the stop does not close the call, which is cut"
        );
    }

    #[test]
    fn an_envelope_names_its_call_and_a_literal_name_needs_no_header() {
        let envelope = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(["name"]),
                arguments: Path::new(["arguments"]),
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        assert_eq!(
            run(
                &table(envelope),
                "<c>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</c>"
            ),
            [
                wrapper("<c>"),
                start("f", "\n{\"name\": \"f\", \"arguments\": "),
                arguments("{\"a\": 1}"),
                end("}"),
                wrapper("\n</c>"),
                finish(1),
            ]
        );
        let literal = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("lookup".into()),
            arguments: Arguments::Json(Dialect::default()),
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        assert_eq!(
            run(&table(literal), "<c>[1, 2]</c>"),
            [
                wrapper("<c>"),
                start("lookup", ""),
                arguments("[1, 2]"),
                end("</c>"),
                finish(1),
            ]
        );
    }

    #[test]
    fn an_envelope_after_the_marker_reads_as_the_json_syntax_does() {
        let envelope = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(["name"]),
                arguments: Path::new(["arguments"]),
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        let json = crate::formats::qwen3(CallSyntax::Json);
        let shaped = crate::formats::qwen3(CallSyntax::Shaped(envelope));
        for output in [
            "<think>\nplan\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"get_weather\", \
             \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            "<tool_call>junk</tool_call><tool_call>{\"name\": \"f\", \"arguments\": {}}\
             </tool_call>",
            "<tool_call>{\"arguments\": {\"a\": 1}, \"name\": \"f\"}</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": [1, 2",
            "<tool_call>{\"name\": \"f\", \"arguments\": {}}x y</tool_call>",
            "<tool_call>{\"name\": \"f\"}<tool_call>{\"name\": \"g\", \"arguments\": 12abc}",
            "<tool_call>{\"name\": \"f\", \"arguments\": 12abc} x</tool_call>",
            "<tool_call>{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        ] {
            for finish in [EngineFinish::Stop, EngineFinish::Length] {
                assert_eq!(
                    run_after(&shaped, "", output, finish.clone()),
                    run_after(&json, "", output, finish),
                    "{output:?}"
                );
            }
        }
    }

    fn word_run(capture: Option<Slot>) -> Atom {
        Atom::Run {
            class: word(),
            min: 1,
            max: None,
            greedy: true,
            capture,
        }
    }

    fn value() -> Atom {
        Atom::Run {
            class: Class::any(),
            min: 0,
            max: None,
            greedy: false,
            capture: Some(Slot::Value),
        }
    }

    fn spaces() -> Atom {
        Atom::Run {
            class: space(),
            min: 0,
            max: None,
            greedy: true,
            capture: None,
        }
    }

    fn ends_at(
        consume: impl IntoIterator<Item = Atom>,
        ahead: impl IntoIterator<Item = Atom>,
    ) -> End {
        End {
            consume: Pattern(consume.into_iter().collect()),
            ahead: Pattern(ahead.into_iter().collect()),
        }
    }

    /// Parameters on abstract tags: `<k=KEY>` and the value, ended by `ends`, in a call named
    /// `f` by the template, with `trailer` before its closing terminal.
    fn tagged(ends: Vec<End>, trailer: Pattern) -> &'static CallShape {
        intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("f".into()),
            arguments: Arguments::Params {
                tag: Pattern([lit("<k="), word_run(Some(Slot::Key)), lit(">"), value()].into()),
                ends: ends.into(),
                values: ValueParser::Raw,
            },
            trailer,
            word: word(),
            hold_cap: 64,
        })
    }

    fn closed_by_tag() -> Vec<End> {
        vec![ends_at([lit("</k>")], [])]
    }

    fn member(json: &str, source: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(source),
        }
    }

    fn closed() -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: "}".to_string(),
            source: Text::default(),
        }
    }

    #[test]
    fn each_parameter_is_written_when_its_value_ends_and_text_between_is_skipped() {
        let format = table(tagged(closed_by_tag(), Pattern::default()));
        assert_eq!(
            run(&format, "<c>junk<k=a>1</k> mid <k=b>x y</k>tail</c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                wrapper("junk"),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper(" mid "),
                member(", \"b\": \"x y\"", "<k=b>x y</k>"),
                wrapper("tail"),
                closed(),
                end("</c>"),
                finish(1),
            ]
        );
        assert_eq!(
            run(&format, "<c></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{}", ""),
                end("</c>"),
                finish(1),
            ],
            "a call with no parameter has an empty object"
        );
    }

    #[test]
    fn a_tag_that_does_not_match_is_passed_over_from_its_next_character() {
        let format = table(tagged(closed_by_tag(), Pattern::default()));
        assert_eq!(
            run(&format, "<c><k=a b>1</k><k=c>2</k></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                wrapper("<k=a b>1</k>"),
                member("{\"c\": 2}", "<k=c>2</k>"),
                end("</c>"),
                finish(1),
            ]
        );
        assert_eq!(
            run(&format, "<c><k=a>1<k=b>2</k></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": \"1<k=b>2\"}", "<k=a>1<k=b>2</k>"),
                end("</c>"),
                finish(1),
            ],
            "without a lookahead end, the first tag's value runs to the first close"
        );
    }

    #[test]
    fn a_value_ends_where_the_next_tag_begins_when_an_end_looks_ahead_at_it() {
        let ends = vec![ends_at([lit("</k>")], []), ends_at([], [lit("<k=")])];
        let format = table(tagged(ends, Pattern::default()));
        assert_eq!(
            run(&format, "<c><k=a>1<k=b>2</k></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1, \"b\": 2}", "<k=a>1<k=b>2</k>"),
                end("</c>"),
                finish(1),
            ],
            "the first value ends before the next tag, which the lookahead does not take"
        );
        assert_eq!(
            run(&format, "<c><k=a>1<k=b>2</c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1"),
                wrapper("<k=b>2"),
                closed(),
                end("</c>"),
                finish(1),
            ],
            "the last value never ends: no parameter, as at the end of the call's text"
        );
    }

    #[test]
    fn a_key_written_twice_is_written_again_and_values_are_typed_by_the_tools() {
        let format = table(tagged(closed_by_tag(), Pattern::default()));
        assert_eq!(
            run(&format, "<c><k=a>1</k><k=a>2</k></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1, \"a\": 2}", "<k=a>1</k><k=a>2</k>"),
                end("</c>"),
                finish(1),
            ]
        );
        let declared = Declared::of(&[openai_protocol::common::Tool {
            tool_type: "function".to_string(),
            function: openai_protocol::common::Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {
                    "s": {"type": "string"},
                    "n": {"type": "integer"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]);
        let events = run_declared(
            &format,
            &declared,
            "",
            "<c><k=s>5</k><k=n>5</k><k=x>5</k></c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            events[2],
            member(
                "{\"s\": \"5\", \"n\": 5, \"x\": 5}",
                "<k=s>5</k><k=n>5</k><k=x>5</k>"
            ),
            "a declared string is its text; anything else is read as the tagged tables read it"
        );
    }

    #[test]
    fn a_cut_call_decides_what_it_holds_at_the_end_and_stays_open() {
        let ends = vec![ends_at([lit("</k>"), spaces()], [])];
        let format = table(tagged(ends, Pattern::default()));
        let events = run_after(&format, "", "<c><k=a>1</k>  <k=b>2", EngineFinish::Length);
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>  "),
                Event::Malformed {
                    text: Text::uncounted("<k=b>2"),
                    why: MalformedReason::UnterminatedRegion,
                },
                end(""),
                Event::Finish {
                    reason: crate::event::FinishReason::Length,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ],
            "the parameter whose end waited for more whitespace is whole at the end"
        );
        let stopped = table(tagged(closed_by_tag(), Pattern::default())).closes_at_stop("call");
        assert_eq!(
            run(&stopped, "<c><k=a>1</k>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1}", "<k=a>1</k>"),
                end(""),
                finish(1),
            ],
            "the stream's stop closes the call and its object"
        );
    }

    #[test]
    fn parameters_end_where_the_trailer_begins() {
        let format = table(tagged(closed_by_tag(), closing_f()));
        assert_eq!(
            run(&format, "<c><k=a>1</k>\n</f>\n</c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper("\n</f>\n"),
                closed(),
                end("</c>"),
                finish(1),
            ],
            "the trailer's bytes were passed over as they came"
        );
        assert_eq!(
            run(&format, "<c><k=a>1</c><k=b>2</k></f></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": \"1</c><k=b>2\"", "<k=a>1</c><k=b>2</k>"),
                wrapper("</f>"),
                closed(),
                end("</c>"),
                finish(1),
            ],
            "a close the trailer does not end at is the call's text"
        );
        assert_eq!(
            run(&format, "<c><k=a>1</f></c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                wrapper("<k=a>1"),
                member("{}", ""),
                end("</f></c>"),
                finish(1),
            ],
            "a value the trailer cut short is no parameter; the trailer's bytes are the end's"
        );
    }

    #[test]
    fn a_stop_closes_tagged_parameters_wherever_the_trailer_stands() {
        let format = table(tagged(closed_by_tag(), closing_f())).closes_at_stop("call");
        assert_eq!(
            run(&format, "<c><k=a>1</k>\n</c>"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper("\n</c>"),
                closed(),
                end(""),
                finish(1),
            ],
            "a close the trailer does not end at is the call's text, and the stop closes the call"
        );
        assert_eq!(
            run(&format, "<c><k=a>1</k>\n</f>\n"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper("\n</f>\n"),
                closed(),
                end(""),
                finish(1),
            ],
            "a trailer that ends with the stream closes the call as before"
        );
        assert_eq!(
            run(&format, "<c><k=a>1</k><k=b>2"),
            [
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper("<k=b>2"),
                closed(),
                end(""),
                finish(1),
            ],
            "a parameter the stop came inside is no parameter, as at the end of the text"
        );
        let cut = |reason| {
            vec![
                wrapper("<c>"),
                start("f", ""),
                member("{\"a\": 1", "<k=a>1</k>"),
                wrapper("\n</c>"),
                end(""),
                Event::Finish {
                    reason,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ]
        };
        assert_eq!(
            run(
                &table(tagged(closed_by_tag(), closing_f())),
                "<c><k=a>1</k>\n</c>"
            ),
            cut(crate::event::FinishReason::Stop),
            "without the row the stop does not close the call, which is cut"
        );
        assert_eq!(
            run_after(&format, "", "<c><k=a>1</k>\n</c>", EngineFinish::Length),
            cut(crate::event::FinishReason::Length),
            "a stream cut for length leaves the call cut"
        );
    }

    #[test]
    fn a_parameters_shape_is_checked() {
        let check = |ends: Vec<End>, trailer: Pattern| {
            CallShape {
                header: Pattern::default(),
                name: NameSource::Literal("f".into()),
                arguments: Arguments::Params {
                    tag: Pattern([lit("<k="), word_run(Some(Slot::Key)), lit(">"), value()].into()),
                    ends: ends.into(),
                    values: ValueParser::Raw,
                },
                trailer,
                word: word(),
                hold_cap: 64,
            }
            .check()
        };
        assert_eq!(check(closed_by_tag(), closing_f()), Ok(()));
        assert_eq!(
            check(Vec::new(), Pattern::default()),
            Err(ShapeError::NoEnds)
        );
        let ahead = vec![ends_at([lit("</k>")], []), ends_at([], [lit("<k=")])];
        assert_eq!(check(ahead.clone(), Pattern::default()), Ok(()));
        assert_eq!(
            check(ahead, closing_f()),
            Err(ShapeError::ReadsPastTheText(1)),
            "a lookahead could read the trailer, which ends the call's text"
        );
        let spaced = vec![ends_at([lit("</k>"), spaces()], [])];
        assert_eq!(check(spaced.clone(), Pattern::default()), Ok(()));
        assert_eq!(
            check(spaced, closing_f()),
            Err(ShapeError::ReadsPastTheText(0)),
            "a run after the close reads past it, where the trailer may begin"
        );
        assert_eq!(
            check(closed_by_tag(), Pattern([lit(">!")].into())),
            Err(ShapeError::TrailerReadsTheEnd(0)),
            "the trailer could begin inside the close"
        );
        let skipping = Pattern(
            [
                Atom::Run {
                    class: Class::any(),
                    min: 0,
                    max: None,
                    greedy: false,
                    capture: None,
                },
                lit("</f>"),
            ]
            .into(),
        );
        assert_eq!(
            check(closed_by_tag(), skipping),
            Err(ShapeError::TrailerReadsTheEnd(0)),
            "the trailer could begin before the close and read it whole"
        );
        let shape = |tag: Vec<Atom>| CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("f".into()),
            arguments: Arguments::Params {
                tag: Pattern(tag.into()),
                ends: closed_by_tag().into(),
                values: ValueParser::Raw,
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        };
        assert_eq!(
            shape(vec![word_run(Some(Slot::Key)), lit(">"), value()]).check(),
            Err(ShapeError::NoTrigger)
        );
        assert_eq!(
            shape(vec![lit("<k>"), value()]).check(),
            Err(ShapeError::Uncaptured(Slot::Key))
        );
        assert_eq!(
            shape(vec![lit("<k="), word_run(Some(Slot::Key)), lit(">")]).check(),
            Err(ShapeError::Uncaptured(Slot::Value))
        );
    }

    /// A reader stricter than the default: a `text` value stripped when its parser says, a value
    /// or a JSON member the schema declares an integer written as the integer it spells (the
    /// member held until whole), any other value a string, and the first of two values for a key.
    #[derive(Debug)]
    struct Strict;

    fn integer(schema: Option<Schema<'_>>) -> bool {
        schema.and_then(|schema| schema.property.get("type")) == Some(&serde_json::json!("integer"))
    }

    impl ValueRead for Strict {
        fn read(&self, parser: &ValueParser, text: &str, schema: Option<Schema<'_>>) -> String {
            let text = match parser {
                ValueParser::Text { strip: true } => text.trim(),
                _ => text,
            };
            match text.parse::<i64>() {
                Ok(number) if integer(schema) => number.to_string(),
                _ => serde_json::Value::String(text.to_string()).to_string(),
            }
        }

        fn streams(&self, _parser: &ValueParser, schema: Option<Schema<'_>>) -> bool {
            !integer(schema)
        }

        fn member(&self, raw: &str, schema: Option<Schema<'_>>) -> Option<String> {
            let text: String = serde_json::from_str(raw).ok()?;
            integer(schema)
                .then(|| text.parse::<i64>().ok())
                .flatten()
                .map(|number| number.to_string())
        }

        fn duplicates(&self) -> Duplicates {
            Duplicates::First
        }
    }

    fn tools() -> Declared {
        Declared::of(&[openai_protocol::common::Tool {
            tool_type: "function".to_string(),
            function: openai_protocol::common::Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {
                    "n": {"type": "integer"},
                    "s": {"type": "string"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    #[test]
    fn a_tables_reader_holds_and_types_the_json_members_it_says() {
        let format = table(named(Pattern::default())).reads_values_with(Arc::new(Strict));
        let events = run_declared(
            &format,
            &tools(),
            "",
            "<c><n=f>{\"n\": \"5\", \"s\": \"5\", \"m\": [\"5\"], \"x\": \"1\"}</c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            events[2],
            Event::ToolCallArguments {
                index: 0,
                json: "{\"n\": 5, \"s\": \"5\", \"m\": [\"5\"], \"x\": \"1\"}".to_string(),
                source: Text::uncounted(
                    "{\"n\": \"5\", \"s\": \"5\", \"m\": [\"5\"], \"x\": \"1\"}"
                ),
            },
            "the integer's string is held and written as the integer; the rest stream as written"
        );
        let broken = run_declared(
            &format,
            &tools(),
            "",
            "<c><n=f>{\"n\": \"5\", x}</c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            broken[2..4],
            [
                Event::ToolCallArguments {
                    index: 0,
                    json: "{\"n\": 5, ".to_string(),
                    source: Text::uncounted("{\"n\": \"5\", "),
                },
                Event::Malformed {
                    text: Text::uncounted("x}"),
                    why: MalformedReason::InvalidArguments,
                },
            ],
            "a held string that closed before the bytes broke is typed at every cut"
        );
        let mut engine = Engine::new(format, tools());
        let mut out = Events::new();
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "<c><n=f>{\"s\": \"a\", \"n\": \"12",
                    spans: &[],
                },
                &mut out,
            )
            .expect("a delta");
        let sent: String = out
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json),
                _ => None,
            })
            .collect();
        assert_eq!(
            sent, "{\"s\": \"a\", \"n\": ",
            "a held string waits for its close; the members before it are sent"
        );
    }

    #[test]
    fn a_tables_reader_reads_each_value_by_its_parser_and_may_keep_the_first_of_two() {
        let shape = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("f".into()),
            arguments: Arguments::Params {
                tag: Pattern([lit("<k="), word_run(Some(Slot::Key)), lit(">"), value()].into()),
                ends: closed_by_tag().into(),
                values: ValueParser::Text { strip: true },
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        let format = table(shape).reads_values_with(Arc::new(Strict));
        let events = run_declared(
            &format,
            &tools(),
            "",
            "<c><k=n> 7 </k><k=s> x </k><k=n>8</k></c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            &events[2..],
            [
                member("{\"n\": 7, \"s\": \"x\"", "<k=n> 7 </k><k=s> x </k>"),
                Event::Malformed {
                    text: Text::uncounted("<k=n>8</k>"),
                    why: MalformedReason::Other(
                        super::super::params::KEY_WRITTEN_TWICE.to_string()
                    ),
                },
                closed(),
                end("</c>"),
                finish(1),
            ]
        );
    }

    #[test]
    fn tokens_given_back_are_counted_in_the_outputs_order() {
        let format = table(named(Pattern::default()));
        let text = "a<c>b<c><n=f>{}</c>";
        // One token per character but the markers, each one token.
        let mut spans = Vec::new();
        let mut at = 0;
        for (token_id, piece) in (0..).zip([
            "a", "<c>", "b", "<c>", "<", "n", "=", "f", ">", "{", "}", "</c>",
        ]) {
            spans.push(TokenSpan {
                token_id,
                start: at,
                end: at + piece.len(),
                continued: false,
            });
            at += piece.len();
        }
        let mut engine = Engine::new(format, Declared::default());
        let mut out = Events::new();
        for input in [
            Input::Delta {
                token_ids: &[],
                text,
                spans: &spans,
            },
            Input::End {
                finish: EngineFinish::Stop,
            },
        ] {
            engine.feed(input, &mut out).expect("in order");
        }
        let counted: Vec<(String, Option<u32>)> = out
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Dropped { text, .. } => {
                    Some((text.text, text.tokens))
                }
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => Some((source.text, source.tokens)),
                _ => None,
            })
            .collect();
        let as_said: Vec<(&str, Option<u32>)> = counted
            .iter()
            .map(|(text, tokens)| (text.as_str(), *tokens))
            .collect();
        assert_eq!(
            as_said,
            [
                ("a<c>b", Some(3)),
                ("<c>", Some(1)),
                ("<n=f>", Some(5)),
                ("{}", Some(2)),
                ("</c>", Some(1)),
            ],
            "the given-back terminal's token counts where its first byte is shown"
        );
    }
}
