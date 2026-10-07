//! The engine: one [`Parser`] that runs any [`Format`]. The scanner splits the model's text into
//! text and the format's terminals, holding back a half-arrived terminal; the table says which
//! state the text is in and where a terminal moves the engine; inside an arguments state the call
//! syntax's assembler ([`json::Assembler`] or [`tagged::Assembler`]) turns the call into its
//! events, streaming the model's own argument bytes.
//!
//! Every byte of the output lands in exactly one event: a terminal that moves the engine as
//! `Dropped { Wrapper }`, the whitespace between a call and the terminal that closes it as
//! `Dropped { Wrapper }` too, any other text there as `Malformed`, run by run so that where the
//! chunks were cut changes nothing, and the rest as content, reasoning, or the call's events.
//!
//! What the engine decides for every format, and what it leaves open:
//!
//! - Prose before a call, between two calls, and after the last call is content, and every complete
//!   call in a chunk is emitted; the old parser dropped the first and stopped after one (smg
//!   #2788).
//! - The template's separator bytes stay where the model put them: the newline after `<think>`, the
//!   two after `</think>`, and so on are reasoning or content, not dropped. Whether they should be
//!   is bellwether #17; `Dropped { Whitespace }` exists for the other answer.
//! - A terminal with no transition from the current state is text, as the model wrote it:
//!   `</think>` in content is content. A terminal inside a code fence, or inside a string in a
//!   call's arguments, is a terminal, as for every parser that reads markers: `</tool_call>` in an
//!   argument's text ends the call there. Bellwether #16 is where that policy is judged.
//! - A call region that never closes is finished at the end of the stream: a call with what
//!   arrived, or the bytes as `Malformed { UnterminatedRegion }`. A region whose closing terminal
//!   comes before a complete call gives what is left as `Malformed` with a reason that says so,
//!   since `UnterminatedRegion` means the stream ended inside the region; with the tagged syntax
//!   the call is closed for the client first, as [`tagged::Assembler::close`] says.
//! - With the tagged syntax, a value's type comes from the request's tools ([`Declared`]): a
//!   declared string streams, every other value is written at its close. A call to a function the
//!   request did not declare has every value inferred.
//! - Call ids are `call_<index>` for now; the id scheme is the maintainer's decision
//!   (deterministic or carrying the conversation's history) and changes only this one line.
//! - Every text event says how many tokens it carries, counted by the [`Ledger`] from the deltas'
//!   spans: a token in the event that carries its first byte, a byte-less span (a held half of a
//!   character, or a hidden special token) into the run that carries the next byte, and the tokens
//!   left without bytes at the end once as `Dropped { ControlToken }`. `Finish::reasoning_tokens`
//!   is the count over the reasoning text. Once a delta lacks spans, or its spans do not partition
//!   its text, the rest of the stream is uncounted, and then `reasoning_tokens` is zero, the one
//!   place where zero does not mean none (rule 7 keeps `Finish` as it is for now).
//! - Tool names are not checked against the request's tools; the format has no tool list yet.
//!
//! The prompt decides where the output starts: its terminals are replayed over the same table
//! from the initial state, and the output begins in the state they leave. The replay starts at
//! the last turn opener the table names (`<|im_start|>assistant` for ChatML), so a marker quoted
//! in an earlier turn, a `<tool_call>` in the user's question or a `<think>` inside an earlier
//! call's arguments, moves nothing; a table that names no opener is replayed from the prompt's
//! start. Qwen 3.5's template opens `<think>` in the generation prompt, so the output starts
//! inside the thought, with `ReasoningStart` pushed for the prompt, and its first `</think>`
//! closes it; Qwen3 writes its own `<think>`; and a prompt that disables thinking ends with
//! `<think>\n\n</think>\n\n`, which leaves the engine in content. A prompt that leaves an
//! arguments state open is read as content for now (a prefilled call is a later step).
//!
//! [`Declared`]: crate::tagged::Declared

use crate::{
    event::{DropReason, Event, Events, FinishReason, MalformedReason},
    format::{CallSyntax, Emits, Format},
    input::{EngineFinish, Input},
    json,
    markers::{Piece, Scanner},
    parser::{ParseError, Parser},
    tagged::{self, Declared},
    tokens::Ledger,
};

pub(crate) const BLOCK_WITHOUT_A_COMPLETE_CALL: &str =
    "a tool-call block that closed without a complete call";
pub(crate) const TEXT_AFTER_THE_OBJECT: &str =
    "text between a call's object and its closing marker";
const TEXT_BETWEEN_CALLS: &str = "text between a block's calls";

/// Why a call region closed: its closing terminal (or the next opener) arrived, or the stream
/// ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Closed {
    ByMarker,
    ByEnd,
}

/// A [`Format`] running as a [`Parser`]. One per generated choice.
#[derive(Debug)]
pub struct Engine {
    format: Format,
    /// The request's tools, which type a tagged call's values.
    declared: Declared,
    scanner: Scanner,
    state: usize,
    /// The call being assembled while the engine is in an arguments state.
    call: Option<Call>,
    calls: u32,
    tokens: Ledger,
    reasoning_tokens: u32,
    stage: Stage,
}

/// The assembler of the call region, one per call syntax.
#[derive(Debug)]
enum Call {
    Json(json::Assembler),
    Tagged(tagged::Assembler),
    Dsml(tagged::dsml::Assembler),
}

impl Call {
    fn new(syntax: Option<CallSyntax>, index: u32) -> Self {
        let id = format!("call_{index}");
        match syntax {
            // `None` is unreachable: `Format::validate`, run by `Engine::new`, refuses a table
            // with an arguments state and no call syntax. The arm keeps the match total.
            Some(CallSyntax::Json) | None => Self::Json(json::Assembler::new(index, id)),
            Some(CallSyntax::Tagged) => Self::Tagged(tagged::Assembler::new(index, id)),
            Some(CallSyntax::Dsml) => Self::Dsml(tagged::dsml::Assembler::new(index, id)),
        }
    }

    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        match self {
            Self::Json(assembler) => assembler.feed(text, out),
            Self::Tagged(assembler) => assembler.feed(text, declared, out),
            Self::Dsml(assembler) => {
                assembler.feed(text, out);
                text.len()
            }
        }
    }

    fn started(&self) -> bool {
        match self {
            Self::Json(assembler) => assembler.started(),
            Self::Tagged(assembler) => assembler.started(),
            Self::Dsml(assembler) => assembler.started(),
        }
    }

    /// Ends the call the way the region closed, and says whether the terminal that closed it was
    /// taken as the call's end. The JSON assembler has one ending, and the engine names a block its
    /// terminal closed (`close_call`); the Qwen tagged one closes the call for the client when the
    /// block ended, and leaves it open when the stream was cut; the DSML one takes the invoke's
    /// closing tag, and only that terminal, as `ToolCallEnd`'s bytes.
    fn end(self, closed: Closed, terminal: &str, out: &mut Events) -> bool {
        match (self, closed) {
            (Self::Json(assembler), _) => assembler.finish(out),
            (Self::Tagged(assembler), Closed::ByMarker) => assembler.close(out),
            (Self::Tagged(assembler), Closed::ByEnd) => assembler.finish(out),
            // Only the invoke's closing tag is the call's end; the next invoke's opening and the
            // block's close end the call too, and the engine drops them as the region's. An
            // invoke that named no call takes whichever terminal ended it, so that it is reported
            // the same way however it ended.
            (Self::Dsml(assembler), Closed::ByMarker)
                if terminal == tagged::dsml::INVOKE_CLOSE || !assembler.started() =>
            {
                assembler.close(terminal, out);
                return true;
            }
            (Self::Dsml(assembler), Closed::ByMarker) => assembler.close("", out),
            (Self::Dsml(assembler), Closed::ByEnd) => assembler.finish(out),
        }
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Streaming,
    Ended,
}

impl Engine {
    /// An engine at the start of an output, in the format's initial state, with the request's
    /// tools, which type a tagged call's values (`Declared::default()` for a request without
    /// tools, or a format that types nothing by them).
    ///
    /// # Panics
    ///
    /// A table [`Format::validate`] refuses: no state, a first state that emits reasoning, an
    /// arguments state with no call syntax, a terminal with no text, an empty turn opener.
    /// Definitions are written in the crate, so that is a programming error.
    #[expect(
        clippy::panic,
        reason = "a table the crate wrote that fails its own check is a programming error"
    )]
    pub fn new(format: Format, declared: Declared) -> Self {
        if let Err(why) = format.validate() {
            panic!("{why}");
        }
        let mut engine = Self {
            scanner: Scanner::new(format.terminal_texts()),
            format,
            declared,
            state: 0,
            call: None,
            calls: 0,
            tokens: Ledger::new(),
            reasoning_tokens: 0,
            stage: Stage::Fresh,
        };
        // A table whose output starts inside a call (a bare list of calls, with no marker before
        // it) opens the call here, since no terminal will.
        if engine.emits() == Emits::Arguments {
            engine.open_call();
        }
        engine
    }

    fn emits(&self) -> Emits {
        self.format.emits(self.state)
    }

    fn take(&mut self, piece: Piece, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, out),
            Piece::Marker(index) => self.terminal(index, out),
        }
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        match self.emits() {
            Emits::Content => out.push_content(self.tokens.text(text)),
            Emits::Reasoning => {
                let counted = self.tokens.text(text);
                self.reasoning_tokens += counted.tokens.unwrap_or(0);
                out.push_reasoning(counted);
            }
            Emits::Arguments => {
                let Some(call) = &mut self.call else {
                    // An arguments state always has a call open; kept as content, not lost.
                    out.push_content(self.tokens.text(text));
                    return;
                };
                let mut assembled = Events::new();
                let taken = call.feed(text, &self.declared, &mut assembled);
                for event in assembled.drain() {
                    out.push(self.tokens.relabel(event));
                }
                self.wrapping(&text[taken..], TEXT_AFTER_THE_OBJECT, out);
            }
            Emits::Wrapper => self.wrapping(text, TEXT_BETWEEN_CALLS, out),
        }
    }

    /// Text where the template only wraps: after a call and before its closing terminal, or
    /// between a block's calls. Each run of whitespace is the template's and is dropped, each run
    /// of anything else is malformed with `why`. Classifying run by run keeps the result the same
    /// wherever the chunks were cut.
    fn wrapping(&mut self, surplus: &str, why: &str, out: &mut Events) {
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
                    why: MalformedReason::Other(why.to_string()),
                }
            });
            rest = &rest[length..];
        }
    }

    /// A terminal arrived: move where the table says, or keep it as text where the model put it.
    fn terminal(&mut self, index: usize, out: &mut Events) {
        let Some(next) = self.format.next(self.state, index) else {
            let text = self.format.terminal_text(index).to_string();
            self.text(&text, out);
            return;
        };
        // The call's remaining events come before the terminal that closed it, so the events'
        // bytes stay in the output's order; a new call before the previous one closed finishes
        // what arrived, then starts. A call syntax may take the terminal as the call's end.
        let terminal = self.format.terminal_text(index).to_string();
        let taken =
            self.emits() == Emits::Arguments && self.close_call(Closed::ByMarker, &terminal, out);
        if !taken {
            out.push(Event::Dropped {
                text: self.tokens.text(&terminal),
                why: DropReason::Wrapper,
            });
        }
        self.enter(next, out);
    }

    /// Moves into state `next`, with the reasoning events and the call that come with the move.
    fn enter(&mut self, next: usize, out: &mut Events) {
        let (from, to) = (self.emits(), self.format.emits(next));
        if from == Emits::Reasoning && to != Emits::Reasoning {
            out.push(Event::ReasoningEnd);
        }
        if to == Emits::Reasoning && from != Emits::Reasoning {
            out.push(Event::ReasoningStart);
        }
        if to == Emits::Arguments {
            self.open_call();
        }
        self.state = next;
    }

    /// The next call takes the next free index; the index is spent only if the region produces a
    /// call, so a `<tool_call>` block that held no call does not count and does not leave a gap.
    fn open_call(&mut self) {
        self.call = Some(Call::new(self.format.call_syntax().copied(), self.calls));
    }

    /// Ends the call region: the assembler closes what arrived, and the result says whether it took
    /// `terminal`, the bytes that closed the region, as the call's end. A region its terminal
    /// closed reports leftover bytes as a block without a complete call; `UnterminatedRegion` is
    /// kept for a region the end of the stream cut.
    fn close_call(&mut self, closed: Closed, terminal: &str, out: &mut Events) -> bool {
        let Some(call) = self.call.take() else {
            return false;
        };
        if call.started() {
            self.calls += 1;
        }
        let mut finished = Events::new();
        let taken = call.end(closed, terminal, &mut finished);
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
        taken
    }

    /// Where the prompt leaves the engine: its terminals replayed over the table from the initial
    /// state, starting at the last turn opener the table names. An arguments state is not entered
    /// from the prompt.
    fn seed(&mut self, prompt: &str, out: &mut Events) {
        let turn = match self
            .format
            .turn_opener()
            .and_then(|opener| prompt.rfind(opener))
        {
            Some(at) => &prompt[at..],
            None => prompt,
        };
        let mut scanner = Scanner::new(self.format.terminal_texts());
        let mut state = 0;
        let pieces = scanner.feed(turn).into_iter().chain(scanner.finish());
        for piece in pieces {
            if let Piece::Marker(index) = piece {
                if let Some(to) = self.format.next(state, index) {
                    state = to;
                }
            }
        }
        if self.format.emits(state) != Emits::Arguments {
            self.enter(state, out);
        }
    }

    fn end(&mut self, finish: EngineFinish, out: &mut Events) {
        let scanner = std::mem::replace(
            &mut self.scanner,
            Scanner::new(self.format.terminal_texts()),
        );
        for piece in scanner.finish() {
            self.take(piece, out);
        }
        match self.emits() {
            Emits::Reasoning => out.push(Event::ReasoningEnd),
            Emits::Arguments => {
                self.close_call(Closed::ByEnd, "", out);
            }
            Emits::Content | Emits::Wrapper => {}
        }
        self.state = 0;
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

impl Parser for Engine {
    fn feed(&mut self, input: Input<'_>, out: &mut Events) -> Result<(), ParseError> {
        match input {
            Input::Prompt { text, .. } => {
                if self.stage != Stage::Fresh {
                    return Err(ParseError::Lifecycle(
                        "prompt after output began".to_string(),
                    ));
                }
                self.stage = Stage::Streaming;
                self.seed(text, out);
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
