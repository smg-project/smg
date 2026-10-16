//! The engine: one [`Parser`] that runs any [`Format`]. The scanner splits the model's text into
//! text and the format's terminals, holding back a half-arrived terminal; the table says which
//! state the text is in and where a terminal moves the engine; inside an arguments state the call
//! syntax's assembler ([`json::Assembler`] or [`tagged::Assembler`]) turns the call into its
//! events, streaming the model's own argument bytes.
//!
//! Every byte of the output lands in exactly one event: a terminal that moves the engine as
//! `Dropped { Wrapper }`, the whitespace between a call and the terminal that closes it as
//! `Dropped { Wrapper }` too, any other text there as `Malformed`, run by run so that where the
//! chunks were cut changes nothing, the whitespace a content stay begins with as
//! `Dropped { Whitespace }` when no text follows it in the stay (the template's spacing after a
//! thought's close, between two calls, or after a thought nothing follows, held until what
//! follows says whose it is), and the rest as content, reasoning, or the call's events.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/engine.md`.

use std::collections::VecDeque;

use crate::{
    event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
    format::{CallSyntax, Emits, Format, Replay},
    gemma,
    input::{EngineFinish, Input},
    json,
    markers::{Piece, Scanner},
    parser::{ParseError, Parser},
    pythonic,
    shape::{self, Values},
    tagged::{self, Declared},
    tokens::Ledger,
};

pub(crate) const BLOCK_WITHOUT_A_COMPLETE_CALL: &str =
    "a tool-call block that closed without a complete call";
pub(crate) const TEXT_AFTER_THE_OBJECT: &str =
    "text between a call's object and its closing marker";
const TEXT_BETWEEN_CALLS: &str = "text between a block's calls";

/// Why a call region closed: its closing terminal (or the next opener) arrived, or the stream
/// stopped where the table says a stop closes it; or the stream ended.
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
    call: Option<Box<dyn CallAssembler>>,
    calls: u32,
    tokens: Ledger,
    reasoning_tokens: u32,
    stage: Stage,
    /// Whether the turn's first piece is still to come, for a table with a start row.
    at_turn_start: bool,
    /// The stay in the current state, as far as a trimmed state or a held lead needs it.
    stay: Stay,
    /// A shaped call whose terminal arrived and whose header is still being read.
    pending: Option<Pending>,
    /// The bytes of the call being assembled, and the ignored terminals that arrived among them
    /// while the assembler held bytes.
    region: Region,
}

/// The bytes of the call region being assembled: how many the assembler took, how many its events
/// have carried out, and the table's ignored terminals that arrived while it held bytes, each at
/// the offset it arrived at. A syntax may hold bytes until it can place them (a header until its
/// marker, a word until it is spelled), and an ignored terminal pushed at once would leave before
/// them; so one that arrives while bytes are held is placed where it arrived once the event that
/// carries the bytes around it comes: spliced into a call event's source, which no client reads,
/// or split out of a text event as a `Dropped { Wrapper }` event of its own between the text's
/// pieces ([`Engine::carried_into`]). The output's bytes leave in their order whatever the syntax
/// holds. When nothing is held it is dropped where it stands.
#[derive(Debug, Default)]
struct Region {
    fed: usize,
    out: usize,
    carried: Vec<(usize, String)>,
}

/// A terminal whose row enters a shaped call, pending the call's header: nothing the row does has
/// been done, and every byte after the terminal is the header's until it decides. A header that
/// matches takes the row; one that cannot gives the terminal and its bytes back.
#[derive(Debug)]
struct Pending {
    /// The terminal, by its index.
    terminal: usize,
    /// The state the row leaves: the engine's, or, for the turn's first piece, the state the turn
    /// starts in, which is taken only when the header matches.
    from: usize,
    /// The call's state, where the row leads.
    to: usize,
    call: shape::Assembler,
}

/// The stay in a state, from the row that entered it to the row that leaves it, as trimming and
/// the held lead read it: whether text other than whitespace has arrived, and the run held. A
/// trimmed stay holds the run after its last text that is not whitespace; a content stay holds
/// the run before its first ([`Engine::lead`]).
#[derive(Debug, Default)]
struct Stay {
    begun: bool,
    held: Vec<Held>,
}

/// Bytes a stay holds, in the output's order: a trimmed stay's run after its last text that is not
/// whitespace, or a content stay's lead before its first.
#[derive(Debug)]
enum Held {
    /// Whitespace: the state's text if more text follows in the stay, dropped if the stay ends.
    Space(String),
    /// A terminal the table ignores, inside the run: dropped wherever it stands, in its place.
    Ignored(String),
}

/// A text event of `like`'s kind carrying `text`, uncounted: the piece of a split text event.
fn text_event_like(like: &Event, text: &str) -> Event {
    let text = Text::uncounted(text);
    match like {
        Event::Reasoning(_) => Event::Reasoning(text),
        Event::Dropped { why, .. } => Event::Dropped { text, why: *why },
        Event::Malformed { why, .. } => Event::Malformed {
            text,
            why: why.clone(),
        },
        _ => Event::Content(text),
    }
}

/// The text an event carries, for the events that carry output bytes.
fn event_text(event: &mut Event) -> Option<&mut Text> {
    match event {
        Event::Content(text) | Event::Reasoning(text) => Some(text),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => Some(text),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => Some(source),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
    }
}

/// Whitespace as Python's `str.strip` removes it: Unicode's white space and the information
/// separators U+001C to U+001F, which `str.isspace` takes too.
pub(crate) fn is_strip_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// What the engine asks of a call syntax's assembler, whichever the syntax: the region's bytes as
/// they arrive, the table's ignored terminal inside the region, and the two endings. Each syntax
/// implements it over its own assembler, so the policy of what ends a call lives with the syntax.
/// An engine moves between threads with its request, so the assembler does too.
pub(crate) trait CallAssembler: std::fmt::Debug + Send + Sync {
    /// Takes the next bytes of the region and pushes the events they complete; the number of bytes
    /// taken. A JSON syntax stops at its object's end and leaves the rest to the engine, which
    /// reports it as text after the object; every other syntax takes every byte.
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize;

    /// The table's ignored terminal arrived inside the call. Its bytes are the engine's to place
    /// ([`Region`]): dropped where it stands when the assembler holds nothing, else placed where
    /// it arrived once the bytes around it leave, spliced into a call event's source or split out
    /// of a text event as a `Dropped` event of its own. A syntax that reads the terminal as its
    /// own separator does what it must here (MiniMax M3's XML releases a tag it holds at it) and
    /// pushes nothing for its bytes.
    fn ignored(&mut self, _text: &str, _declared: &Declared, _out: &mut Events) {}

    /// The call the engine entered through a matched header starts: the events of its start, and
    /// the bytes that followed the header, for the engine to read again inside the call. Every
    /// syntax but a shaped one starts with its first bytes and gives nothing back.
    fn start(&mut self, _out: &mut Events) -> String {
        String::new()
    }

    /// Whether the region closes at `terminal`, a terminal with a row out of the call's state:
    /// always, but for a shaped call, whose trailer must end right before it.
    fn closes(&mut self, _terminal: &str) -> bool {
        true
    }

    /// The region closed: `terminal` is the table's terminal that closed it, the call's own closing
    /// marker, the next call's opener or the block's close. The call ends as the syntax has it, and
    /// the answer says whether the terminal's bytes were taken as the call's end (`ToolCallEnd`'s
    /// source); otherwise the engine drops them as the region's.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool;

    /// The stream ended inside the region: nothing is closed for the client, so arguments cut
    /// short never look complete.
    fn finish(self: Box<Self>, out: &mut Events);
}

/// A call given as data ([`CallSyntax::Shaped`]) behind the trait. Its assembler types values by
/// the request's tools at every ending, so the call keeps them.
#[derive(Debug)]
struct ShapedCall {
    call: shape::Assembler,
    declared: Declared,
}

impl CallAssembler for ShapedCall {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.call.feed(text, declared, out);
        text.len()
    }

    fn start(&mut self, out: &mut Events) -> String {
        self.call.start(out)
    }

    fn closes(&mut self, terminal: &str) -> bool {
        self.call.closes(terminal)
    }

    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool {
        let ShapedCall { call, declared } = *self;
        call.close(terminal, &declared, out)
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        let ShapedCall { call, declared } = *self;
        call.finish(&declared, out);
    }
}

/// The assembler for `syntax`'s next call, at `index`; `framed` says whether a row of the table
/// enters the arguments state, so markers frame the region; `values` reads a shaped call's values
/// and `declared` types them.
fn assembler(
    syntax: Option<CallSyntax>,
    index: u32,
    framed: bool,
    values: Values,
    declared: &Declared,
) -> Box<dyn CallAssembler> {
    let id = format!("call_{index}");
    match syntax {
        // `None` is unreachable: `Format::validate`, run by `Engine::new`, refuses a table with
        // an arguments state and no call syntax. The arm keeps the match total.
        Some(CallSyntax::Json) | None => Box::new(json::Assembler::new(index, id)),
        Some(CallSyntax::Tagged(_, placement)) => {
            Box::new(tagged::Assembler::new(index, id, placement))
        }
        Some(CallSyntax::Keyed(tags)) => Box::new(tagged::keyed::Assembler::new(index, id, tags)),
        Some(CallSyntax::Pythonic(_)) => Box::new(pythonic::Assembler::new(index)),
        // The list between markers (Llama Nemotron) takes text in place of it as malformed; the
        // bare list the output starts in (xLAM) takes it for prose.
        Some(CallSyntax::JsonList) if framed => Box::new(json::list::Assembler::framed(index)),
        Some(CallSyntax::JsonList) => Box::new(json::list::Assembler::new(index)),
        Some(CallSyntax::JsonAlone) => Box::new(json::list::Assembler::object_or_list(index)),
        Some(CallSyntax::Xml) => Box::new(tagged::xml::Assembler::new(index, id)),
        Some(CallSyntax::Attributed(tags)) => {
            Box::new(tagged::attributed::Assembler::new(index, id, tags))
        }
        Some(CallSyntax::NamedJson(marker)) => {
            Box::new(json::named::Assembler::new(index, id, marker))
        }
        Some(CallSyntax::NamedArguments(marker)) => {
            Box::new(json::named_arguments::Assembler::new(index, id, marker))
        }
        Some(CallSyntax::ForcedArguments) => Box::new(json::forced::Assembler::new(index, id)),
        Some(CallSyntax::Gemma(spelling)) => Box::new(gemma::Assembler::new(index, id, spelling)),
        // A shaped call opens through its terminal's header, which the engine reads before it
        // enters the call's state with that assembler (`Engine::open_shaped`); a table that
        // validates enters the state no other way.
        Some(CallSyntax::Shaped(shape)) => Box::new(ShapedCall {
            call: shape::Assembler::new(index, id, shape, values, None),
            declared: declared.clone(),
        }),
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
    /// arguments state with no call syntax, a terminal with no text, an empty turn opener or
    /// anchor, an opener beside anchors, a start row where the first state holds a call or an
    /// opener replays the turn, a stop, framing or trim row on a state it does not fit, or
    /// a shaped call a stream cannot run, or one a table cannot open through its terminal (see
    /// [`Format::validate`]). Definitions are written in the crate, so that is a programming
    /// error; a shape built at run time is checked before its table is.
    #[expect(
        clippy::panic,
        reason = "a table the crate wrote that fails its own check is a programming error"
    )]
    pub fn new(format: Format, declared: Declared) -> Self {
        if let Err(why) = format.validate() {
            panic!("{why}");
        }
        let at_turn_start = format.start().is_some();
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
            at_turn_start,
            stay: Stay::default(),
            pending: None,
            region: Region::default(),
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

    /// Takes the pieces in order. A shaped call's header that decides hands back bytes to read
    /// again: what followed the header when it matched, the terminal from its second character on
    /// when it did not. They are scanned afresh, and so are the pieces after them until the new
    /// scan stands where the old one cut the text with nothing held, from which on the two cut it
    /// alike; at the end of the stream everything is scanned to the end, and a header still
    /// undecided is decided once nothing more comes.
    fn run(&mut self, pieces: Vec<Piece>, at_end: bool, out: &mut Events) {
        let mut queue = VecDeque::from(pieces);
        loop {
            while let Some(piece) = queue.pop_front() {
                let again = match piece {
                    Piece::Text(text) if at_end && self.cut_tag(&text) => {
                        self.drop_cut_tag(&text, out);
                        None
                    }
                    piece => self.take(piece, out),
                };
                if let Some(again) = again {
                    self.scan_again(again, &mut queue, at_end);
                }
            }
            if !at_end {
                return;
            }
            let Some(again) = self.decide_at_end(out) else {
                return;
            };
            self.scan_again(again, &mut queue, true);
        }
    }

    /// The bytes a piece stands for.
    fn piece_text<'p>(&'p self, piece: &'p Piece) -> &'p str {
        match piece {
            Piece::Text(text) => text,
            Piece::Marker(index) => self.format.terminal_text(*index),
        }
    }

    /// Scans `text` afresh, before the pieces in `queue`, and puts what it gives at the queue's
    /// front. The queued pieces follow it, scanned again too while the new scan holds a possible
    /// terminal at their edge: once it holds nothing where the old scan began a piece, both cut
    /// what follows alike, since a scan from a piece's start reads nothing before it. When the
    /// queue runs out first, what the old scanner held comes after them, and the new scanner takes
    /// its place; at the end of the stream it is finished.
    fn scan_again(&mut self, text: String, queue: &mut VecDeque<Piece>, at_end: bool) {
        let mut scanner = Scanner::new(self.format.terminal_texts());
        let mut pieces = scanner.feed(&text);
        while !scanner.held().is_empty() {
            let Some(piece) = queue.pop_front() else {
                break;
            };
            pieces.extend(scanner.feed(self.piece_text(&piece)));
        }
        if queue.is_empty() {
            pieces.extend(scanner.feed(self.scanner.held()));
            if at_end {
                pieces.extend(scanner.finish());
                self.scanner = Scanner::new(self.format.terminal_texts());
            } else {
                self.scanner = scanner;
            }
        }
        for piece in pieces.into_iter().rev() {
            queue.push_front(piece);
        }
    }

    /// Takes one piece, and says what to read again when a shaped call's header decided.
    fn take(&mut self, piece: Piece, out: &mut Events) -> Option<String> {
        if self.pending.is_some() {
            let text = self.piece_text(&piece).to_string();
            return self.read_header(&text, out);
        }
        if let Piece::Marker(index) = piece {
            let leaves = self.format.next(self.state, index).is_some();
            if leaves && self.emits() == Emits::Arguments {
                let terminal = self.format.terminal_text(index).to_string();
                if !self.call_closes(&terminal) {
                    // The trailer does not end at the terminal: it closes nothing and is the
                    // call's text, as the template's closing pattern reads past it.
                    self.text(&terminal, out);
                    return None;
                }
            }
            if let Some(pending) = self.pending_call(index) {
                self.pending = Some(pending);
                return self.read_header("", out);
            }
        }
        self.place(piece, out);
        None
    }

    /// Takes one piece where the turn and the table put it: the turn's first piece decides where
    /// the turn starts, then text is the state's and a terminal moves the engine or is text.
    fn place(&mut self, piece: Piece, out: &mut Events) {
        if self.at_turn_start {
            if let Some(state) = self.format.opening(self.state, &piece) {
                self.at_turn_start = false;
                if state != self.state {
                    self.enter(state, out);
                }
            }
        }
        match piece {
            Piece::Text(text) => self.text(&text, out),
            Piece::Marker(index) => self.terminal(index, out),
        }
    }

    /// Whether the call open in this state closes at `terminal`, a terminal with a row out of it:
    /// a shaped call's trailer must end right before the terminal.
    fn call_closes(&mut self, terminal: &str) -> bool {
        self.call.as_mut().is_none_or(|call| call.closes(terminal))
    }

    /// A shaped call pending its header, when terminal `index` has a row into the call's state from
    /// this state, or, as the turn's first piece, from the state the turn starts in: the header is
    /// read before anything the row does, the turn's start included.
    fn pending_call(&self, index: usize) -> Option<Pending> {
        let Some(&CallSyntax::Shaped(shape)) = self.format.call_syntax() else {
            return None;
        };
        let from = if self.at_turn_start {
            self.format.opening(self.state, &Piece::Marker(index))?
        } else {
            self.state
        };
        let to = self.format.next(from, index)?;
        if self.format.emits(to) != Emits::Arguments {
            return None;
        }
        let terminal = self.format.terminal_text(index);
        let id = format!("call_{}", self.calls);
        let values = self.format.value_reader();
        let call =
            shape::Assembler::new(self.calls, id, shape, values, terminal.chars().next_back());
        Some(Pending {
            terminal: index,
            from,
            to,
            call,
        })
    }

    /// Reads the next bytes after a shaped call's terminal into its header, and takes the row or
    /// gives the bytes back once the header decides. Returns what to read again then.
    fn read_header(&mut self, bytes: &str, out: &mut Events) -> Option<String> {
        let pending = self.pending.as_mut()?;
        match pending.call.header(bytes) {
            shape::Header::Pending => None,
            shape::Header::Matched => Some(self.open_shaped(out)),
            shape::Header::Failed => Some(self.give_back(out)),
        }
    }

    /// At the end of the stream: a header still being read is decided by the bytes it has.
    fn decide_at_end(&mut self, out: &mut Events) -> Option<String> {
        let pending = self.pending.as_mut()?;
        match pending.call.header_end() {
            shape::Header::Pending => None,
            shape::Header::Matched => Some(self.open_shaped(out)),
            shape::Header::Failed => Some(self.give_back(out)),
        }
    }

    /// The header matched: the turn starts where its first piece says, if the terminal is that
    /// piece, the row is taken as any terminal's is, the call is entered with its assembler and
    /// started, and what followed the header is handed back to read again inside the call.
    fn open_shaped(&mut self, out: &mut Events) -> String {
        let Some(Pending {
            terminal,
            from,
            to,
            call,
        }) = self.pending.take()
        else {
            return String::new();
        };
        if self.at_turn_start {
            self.at_turn_start = false;
            if from != self.state {
                self.enter(from, out);
            }
        }
        let call = Box::new(ShapedCall {
            call,
            declared: self.declared.clone(),
        });
        self.take_row(terminal, to, Some(call), out);
        let mut started = Events::new();
        let again = self
            .call
            .as_mut()
            .map_or_else(String::new, |call| call.start(&mut started));
        self.push_call_events(started, out);
        // The header's bytes went to the shape, not through `feed`: the start's events carried
        // them out, so the region counts them as taken too.
        self.region.fed = self.region.fed.max(self.region.out);
        again
    }

    /// The header cannot match: the terminal opened no call. Its first character is the text of
    /// the state it came in, as a regular expression's search for the call's opening goes on after
    /// it, and the rest of the terminal and every byte the header read are handed back to read
    /// again. None of them was emitted, so the tokens are still counted in the output's order.
    fn give_back(&mut self, out: &mut Events) -> String {
        let Some(Pending { terminal, call, .. }) = self.pending.take() else {
            return String::new();
        };
        let mut first = self.format.terminal_text(terminal).to_string();
        first.push_str(&call.into_header_bytes());
        let at = first.chars().next().map_or(0, char::len_utf8);
        let again = first.split_off(at);
        self.place(Piece::Text(first), out);
        again
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        if self.format.is_framing(self.state) {
            out.push(Event::Dropped {
                text: self.tokens.text(text),
                why: DropReason::Wrapper,
            });
            return;
        }
        match self.emits() {
            Emits::Content | Emits::Reasoning if self.format.is_trimmed(self.state) => {
                self.trimmed(text, out);
            }
            Emits::Content if !self.stay.begun => self.lead(text, out),
            Emits::Content | Emits::Reasoning => self.show(text, out),
            Emits::Arguments => {
                let Some(call) = &mut self.call else {
                    // An arguments state always has a call open; kept as content, not lost.
                    out.push_content(self.tokens.text(text));
                    return;
                };
                let mut assembled = Events::new();
                let taken = call.feed(text, &self.declared, &mut assembled);
                self.region.fed += taken;
                self.push_call_events(assembled, out);
                self.wrapping(&text[taken..], TEXT_AFTER_THE_OBJECT, out);
            }
            Emits::Wrapper => self.wrapping(text, TEXT_BETWEEN_CALLS, out),
        }
    }

    /// Text of the content or reasoning state the engine is in, as the client sees it.
    fn show(&mut self, text: &str, out: &mut Events) {
        let counted = self.tokens.text(text);
        if self.emits() == Emits::Reasoning {
            self.reasoning_tokens += counted.tokens.unwrap_or(0);
            out.push_reasoning(counted);
        } else {
            out.push_content(counted);
        }
    }

    /// Text in a trimmed stay: whitespace before the stay's first other character is dropped as it
    /// arrives; a run of it after the last one is held, and released as the state's text before
    /// more text that is not whitespace.
    fn trimmed(&mut self, text: &str, out: &mut Events) {
        let mut rest = text;
        if !self.stay.begun {
            let body = rest.trim_start_matches(is_strip_space);
            let lead = &rest[..rest.len() - body.len()];
            if !lead.is_empty() {
                out.push(Event::Dropped {
                    text: self.tokens.text(lead),
                    why: DropReason::Whitespace,
                });
            }
            if body.is_empty() {
                return;
            }
            self.stay.begun = true;
            rest = body;
        }
        let body = rest.trim_end_matches(is_strip_space);
        if !body.is_empty() {
            self.release_stay(out);
            self.show(body, out);
        }
        self.hold(&rest[body.len()..]);
    }

    /// Text at the start of a content stay, before the stay's first character that is not
    /// whitespace: the whitespace is held, since what follows says whose it is. Text makes it
    /// the text's, released as content before it; the stay ending without text makes it the
    /// template's spacing (between a thought's close and the first call, between two calls,
    /// after a thought nothing follows), dropped. The whole answer's message shows no content
    /// for whitespace alone, and a stream that showed it as it arrived carried what the whole
    /// answer does not.
    fn lead(&mut self, text: &str, out: &mut Events) {
        let body = text.trim_start_matches(is_strip_space);
        self.hold(&text[..text.len() - body.len()]);
        if body.is_empty() {
            return;
        }
        self.stay.begun = true;
        self.release_stay(out);
        self.show(body, out);
    }

    /// Holds `space`, a run of whitespace, after what the stay holds.
    fn hold(&mut self, space: &str) {
        if space.is_empty() {
            return;
        }
        match self.stay.held.last_mut() {
            Some(Held::Space(held)) => held.push_str(space),
            _ => self.stay.held.push(Held::Space(space.to_string())),
        }
    }

    /// The run the stay held is the state's text: the whitespace is shown, and an ignored
    /// terminal among it is dropped in its place.
    fn release_stay(&mut self, out: &mut Events) {
        for held in std::mem::take(&mut self.stay.held) {
            match held {
                Held::Space(space) => self.show(&space, out),
                Held::Ignored(terminal) => out.push(Event::Dropped {
                    text: self.tokens.text(&terminal),
                    why: DropReason::Wrapper,
                }),
            }
        }
    }

    /// The stay ends: the run it held is the template's and is dropped, a trimmed stay's run
    /// after its last text as much as a lead no text followed.
    fn end_stay(&mut self, out: &mut Events) {
        for held in std::mem::take(&mut self.stay.held) {
            let (text, why) = match held {
                Held::Space(space) => (space, DropReason::Whitespace),
                Held::Ignored(terminal) => (terminal, DropReason::Wrapper),
            };
            out.push(Event::Dropped {
                text: self.tokens.text(&text),
                why,
            });
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
    /// A terminal the table ignores is dropped where it stands, and moves nothing: inside a call
    /// whose assembler holds bytes it is placed where it arrived once the bytes around it leave
    /// ([`Region`]), and in a stay's held run (a trimmed stay's tail, a content stay's lead) it
    /// keeps its place among them.
    fn terminal(&mut self, index: usize, out: &mut Events) {
        if self.format.ignored(index) {
            let text = self.format.terminal_text(index).to_string();
            if self.emits() == Emits::Arguments {
                if let Some(call) = &mut self.call {
                    let mut events = Events::new();
                    call.ignored(&text, &self.declared, &mut events);
                    self.push_call_events(events, out);
                    if self.region.out < self.region.fed {
                        self.region.carried.push((self.region.fed, text));
                        return;
                    }
                    out.push(Event::Dropped {
                        text: self.tokens.text(&text),
                        why: DropReason::Wrapper,
                    });
                    return;
                }
            }
            // Inside a stay's held run, the terminal keeps its place among the bytes held.
            if !self.stay.held.is_empty() {
                self.stay.held.push(Held::Ignored(text));
                return;
            }
            out.push(Event::Dropped {
                text: self.tokens.text(&text),
                why: DropReason::Wrapper,
            });
            return;
        }
        let Some(next) = self.format.next(self.state, index) else {
            let text = self.format.terminal_text(index).to_string();
            self.text(&text, out);
            return;
        };
        self.take_row(index, next, None, out);
    }

    /// Takes the row that terminal `index` makes from this state into `next`, entering a call
    /// state with `call` when it is given and with a new call of the syntax otherwise. The call's
    /// remaining events come before the terminal that closed it, so the events' bytes stay in the
    /// output's order; a new call before the previous one closed finishes what arrived, then
    /// starts. A call syntax may take the terminal as the call's end.
    fn take_row(
        &mut self,
        index: usize,
        next: usize,
        call: Option<Box<dyn CallAssembler>>,
        out: &mut Events,
    ) {
        let terminal = self.format.terminal_text(index).to_string();
        // A shown marker is text the client sees: a lead held before it is the text's. Any other
        // row ends the stay without text after the held run, which is the template's.
        if self.format.shown(index) && !self.format.is_trimmed(self.state) {
            self.release_stay(out);
        } else {
            self.end_stay(out);
        }
        let taken =
            self.emits() == Emits::Arguments && self.close_call(Closed::ByMarker, &terminal, out);
        let shown = !taken && self.format.shown(index);
        if !taken {
            let text = self.tokens.text(&terminal);
            if shown {
                // A marker of a channel the request does not separate: the client sees it.
                out.push_content(text);
            } else {
                out.push(Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                });
            }
        }
        self.enter_with(next, call, out);
        // The stay a shown marker begins has its text already: whitespace after it is content as
        // it would be after any other text.
        if shown && !self.format.is_trimmed(next) {
            self.stay.begun = true;
        }
    }

    /// Moves into state `next`, with the reasoning events and the call that come with the move.
    fn enter(&mut self, next: usize, out: &mut Events) {
        self.enter_with(next, None, out);
    }

    /// Moves into state `next`, a call state entered with `call` when it is given and with a new
    /// call of the syntax otherwise.
    fn enter_with(&mut self, next: usize, call: Option<Box<dyn CallAssembler>>, out: &mut Events) {
        let (from, to) = (self.emits(), self.format.emits(next));
        if from == Emits::Reasoning && to != Emits::Reasoning {
            out.push(Event::ReasoningEnd);
        }
        if to == Emits::Reasoning && from != Emits::Reasoning {
            out.push(Event::ReasoningStart);
        }
        if to == Emits::Arguments {
            self.region = Region::default();
            self.call = Some(call.unwrap_or_else(|| self.new_call()));
        }
        self.state = next;
        self.stay = Stay::default();
    }

    /// The next call takes the next free index; the index is spent only when a call starts, so a
    /// `<tool_call>` block that held no call does not count and does not leave a gap.
    fn open_call(&mut self) {
        self.region = Region::default();
        self.call = Some(self.new_call());
    }

    /// The assembler of the next call of the syntax: `framed` says whether a row of the table
    /// enters the arguments state, so markers frame the region.
    fn new_call(&self) -> Box<dyn CallAssembler> {
        let framed = self
            .format
            .transitions()
            .any(|(_, _, to)| self.format.emits(to) == Emits::Arguments);
        assembler(
            self.format.call_syntax().copied(),
            self.calls,
            framed,
            self.format.value_reader(),
            &self.declared,
        )
    }

    /// Ends the call region: the assembler closes what arrived, and the result says whether it took
    /// `terminal`, the bytes that closed the region, as the call's end. A region its terminal
    /// closed reports leftover bytes as a block without a complete call; `UnterminatedRegion` is
    /// kept for a region the end of the stream cut.
    fn close_call(&mut self, closed: Closed, terminal: &str, out: &mut Events) -> bool {
        let Some(call) = self.call.take() else {
            return false;
        };
        let mut finished = Events::new();
        let taken = match closed {
            Closed::ByMarker => call.close(terminal, &mut finished),
            Closed::ByEnd => {
                call.finish(&mut finished);
                false
            }
        };
        if taken {
            self.region.fed += terminal.len();
        }
        if closed == Closed::ByMarker {
            let mut renamed = Events::new();
            for event in finished.drain() {
                renamed.push(match event {
                    Event::Malformed {
                        text,
                        why: MalformedReason::UnterminatedRegion,
                    } => Event::Malformed {
                        text,
                        why: MalformedReason::Other(BLOCK_WITHOUT_A_COMPLETE_CALL.to_string()),
                    },
                    event => event,
                });
            }
            finished = renamed;
        }
        self.push_call_events(finished, out);
        // An ignored terminal no event carried out leaves here, before the region's close.
        for (_, marker) in std::mem::take(&mut self.region.carried) {
            out.push(Event::Dropped {
                text: self.tokens.text(&marker),
                why: DropReason::Wrapper,
            });
        }
        self.region = Region::default();
        taken
    }

    /// Pushes a call's events with their tokens counted, and counts each call that starts: the
    /// index the next call takes, and `Finish::tool_calls`. A syntax that writes several calls in
    /// one region starts each of them here. An event's bytes are counted as carried out of the
    /// region, with the ignored terminals that arrived among them placed where they arrived.
    fn push_call_events(&mut self, mut events: Events, out: &mut Events) {
        for event in events.drain() {
            if matches!(event, Event::ToolCallStart { .. }) {
                self.calls += 1;
            }
            for event in self.carried_into(event) {
                out.push(self.tokens.relabel(event));
            }
        }
    }

    /// `event` with the ignored terminals that arrived among the bytes it carries placed where
    /// they arrived, and its bytes counted as carried out of the region. A call's event has them
    /// spliced into its source, which no client reads; a text event is split around them, each
    /// terminal a `Dropped { Wrapper }` event of its own, so that none joins text a client reads
    /// (a syntax that hands held bytes back as content, as the bare list does).
    fn carried_into(&mut self, mut event: Event) -> Vec<Event> {
        let Some(text) = event_text(&mut event) else {
            return vec![event];
        };
        if text.text.is_empty() {
            return vec![event];
        }
        let from = self.region.out;
        let to = from + text.text.len();
        self.region.out = to;
        if self.region.carried.is_empty() {
            return vec![event];
        }
        let (within, after): (Vec<_>, Vec<_>) = std::mem::take(&mut self.region.carried)
            .into_iter()
            .partition(|(at, _)| *at <= to);
        self.region.carried = after;
        if within.is_empty() {
            return vec![event];
        }
        let bytes = std::mem::take(&mut text.text);
        let mut cuts = Vec::with_capacity(within.len());
        let mut done = 0;
        for (at, marker) in within {
            let mut cut = at.saturating_sub(from).clamp(done, bytes.len());
            while !bytes.is_char_boundary(cut) {
                cut -= 1;
            }
            cuts.push((cut, marker));
            done = cut;
        }
        let is_call = matches!(
            event,
            Event::ToolCallStart { .. }
                | Event::ToolCallArguments { .. }
                | Event::ToolCallEnd { .. }
        );
        if is_call {
            let mut spliced = String::with_capacity(bytes.len());
            let mut done = 0;
            for (cut, marker) in cuts {
                spliced.push_str(&bytes[done..cut]);
                spliced.push_str(&marker);
                done = cut;
            }
            spliced.push_str(&bytes[done..]);
            if let Some(text) = event_text(&mut event) {
                text.text = spliced;
            }
            return vec![event];
        }
        let mut events = Vec::with_capacity(cuts.len() * 2 + 1);
        let mut done = 0;
        for (cut, marker) in cuts {
            if cut > done {
                events.push(text_event_like(&event, &bytes[done..cut]));
            }
            events.push(Event::Dropped {
                text: Text::uncounted(marker),
                why: DropReason::Wrapper,
            });
            done = cut;
        }
        if done < bytes.len() {
            events.push(text_event_like(&event, &bytes[done..]));
        }
        events
    }

    /// Where the prompt leaves the engine: the table's replay of it ([`Format::replayed`], which
    /// [`Format::reasoning_open`] reads too), from the last turn opener the table names or after
    /// its last turn anchor, the turn's first piece deciding its start where the table has a start
    /// row. An arguments state is not entered from the prompt: the engine rests where the turn
    /// started.
    fn seed(&mut self, prompt: &str, out: &mut Events) {
        let Replay {
            state,
            at_turn_start,
            begun,
        } = self.format.replayed(prompt);
        self.at_turn_start = at_turn_start;
        if state != self.state {
            self.enter(state, out);
        }
        self.stay.begun = begun;
    }

    /// Whether `text`, a piece at the end of the output, is a tag the end cut short: one that
    /// begins with a whole structural token and is the beginning of a terminal the state has a
    /// row for (`<|close|>message` with no `<|sep|>` after it), which the model cannot have
    /// written as text. A shaped call's header still being read takes every byte.
    fn cut_tag(&self, text: &str) -> bool {
        self.pending.is_none() && self.format.is_cut_tag(self.state, text)
    }

    /// A tag the end cut short is dropped as the whole tag would have been. Inside a call the
    /// assembler still holds the call's bytes, which came before the tail: the call ends first, so
    /// every byte leaves in order.
    fn drop_cut_tag(&mut self, text: &str, out: &mut Events) {
        self.end_stay(out);
        if self.emits() == Emits::Arguments {
            self.close_call(Closed::ByEnd, "", out);
        }
        out.push(Event::Dropped {
            text: self.tokens.text(text),
            why: DropReason::Wrapper,
        });
    }

    fn end(&mut self, finish: EngineFinish, out: &mut Events) {
        let scanner = std::mem::replace(
            &mut self.scanner,
            Scanner::new(self.format.terminal_texts()),
        );
        // What the scanner held back as a possible terminal is text now that no more bytes come,
        // except a tag the end cut short.
        self.run(scanner.finish(), true, out);
        self.end_stay(out);
        match self.emits() {
            Emits::Reasoning => out.push(Event::ReasoningEnd),
            Emits::Arguments => {
                // The stop token that closes the call is hidden, so the call's end has no bytes. A
                // shaped call of one JSON value or object needs its trailer to end where the stream
                // does; tagged parameters close wherever it stands.
                let closed = if matches!(finish, EngineFinish::Stop)
                    && self.format.stop_closes(self.state)
                    && self.call_closes("")
                {
                    Closed::ByMarker
                } else {
                    Closed::ByEnd
                };
                self.close_call(closed, "", out);
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
                let pieces = self.scanner.feed(text);
                self.run(pieces, false, out);
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
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        formats::{glm, inkling, qwen3, xlam},
        input::TokenSpan,
        tagged::{Placement, Spelling},
    };

    fn declared() -> Declared {
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

    /// GLM's last call, its closing marker not written: the stop token that would end it is
    /// hidden by the stop decoder.
    const LAST_CALL: &str = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
    );

    fn run(format: Format, output: &str, finish: EngineFinish) -> Vec<Event> {
        let mut engine = Engine::new(format, declared());
        let mut out = Events::new();
        for input in [
            Input::Delta {
                token_ids: &[],
                text: output,
                spans: &[],
            },
            Input::End { finish },
        ] {
            engine.feed(input, &mut out).expect("in order");
        }
        out.drain()
    }

    /// The events of `output` after `prompt`, ended by a stop.
    fn run_after(format: Format, prompt: &str, output: &str) -> Vec<Event> {
        let mut engine = Engine::new(format, declared());
        let mut out = Events::new();
        for input in [
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            Input::Delta {
                token_ids: &[],
                text: output,
                spans: &[],
            },
            Input::End {
                finish: EngineFinish::Stop,
            },
        ] {
            engine.feed(input, &mut out).expect("in order");
        }
        out.drain()
    }

    /// Each call's arguments as the client joins them, and the bytes of each call's end.
    fn calls(events: &[Event]) -> Vec<(String, Option<String>)> {
        let mut calls: Vec<(String, Option<String>)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { .. } => calls.push((String::new(), None)),
                Event::ToolCallArguments { json, .. } => {
                    if let Some((arguments, _)) = calls.last_mut() {
                        arguments.push_str(json);
                    }
                }
                Event::ToolCallEnd { source, .. } => {
                    if let Some((_, end)) = calls.last_mut() {
                        *end = Some(source.text.clone());
                    }
                }
                _ => {}
            }
        }
        calls
    }

    fn bytes(events: &[Event]) -> String {
        let mut list = Events::new();
        for event in events {
            list.push(event.clone());
        }
        list.text()
    }

    #[test]
    fn a_stop_closes_the_call_where_the_table_says_and_a_cut_stays_cut() {
        let closed = calls(&run(
            glm().closes_at_stop("call"),
            LAST_CALL,
            EngineFinish::Stop,
        ));
        assert_eq!(
            closed,
            [(
                r#"{"city": "Paris", "days": 3}"#.to_string(),
                Some(String::new())
            )]
        );
        for finish in [EngineFinish::Length, EngineFinish::Abort] {
            let events = run(glm().closes_at_stop("call"), LAST_CALL, finish);
            assert_eq!(
                calls(&events),
                [(
                    r#"{"city": "Paris", "days": 3"#.to_string(),
                    Some(String::new())
                )]
            );
            assert_eq!(bytes(&events), LAST_CALL);
        }
    }

    #[test]
    fn without_the_row_a_stop_leaves_the_call_cut() {
        let events = run(glm(), LAST_CALL, EngineFinish::Stop);
        assert_eq!(
            calls(&events),
            [(
                r#"{"city": "Paris", "days": 3"#.to_string(),
                Some(String::new())
            )]
        );
    }

    #[test]
    fn a_closing_marker_the_model_wrote_ends_the_call_with_or_without_the_row() {
        let output = format!("{LAST_CALL}</tool_call><tool_call>get_weather</tool_call>");
        let with_row = run(glm().closes_at_stop("call"), &output, EngineFinish::Stop);
        assert_eq!(with_row, run(glm(), &output, EngineFinish::Stop));
        assert_eq!(
            calls(&with_row),
            [
                (
                    r#"{"city": "Paris", "days": 3}"#.to_string(),
                    Some("</tool_call>".to_string())
                ),
                ("{}".to_string(), Some("</tool_call>".to_string())),
            ]
        );
        assert_eq!(bytes(&with_row), output);
    }

    #[test]
    fn a_stop_closes_the_call_as_its_closing_marker_would_wherever_it_stands() {
        // Inside a string, inside an integer, between two parameters and inside a key: the call
        // ends as `</tool_call>` there would end it, with no bytes for the end.
        for output in [
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key><arg_value>2",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            "<tool_call>get_weather<arg_key>ci",
        ] {
            let stopped = run(glm().closes_at_stop("call"), output, EngineFinish::Stop);
            let marked = run(glm(), &format!("{output}</tool_call>"), EngineFinish::Stop);
            let without_end_bytes = |events: &[Event]| -> Vec<Event> {
                events
                    .iter()
                    .filter(|event| !matches!(event, Event::Dropped { .. }))
                    .map(|event| match event {
                        Event::ToolCallEnd { index, .. } => Event::ToolCallEnd {
                            index: *index,
                            source: Text::uncounted(""),
                        },
                        event => event.clone(),
                    })
                    .collect()
            };
            assert_eq!(
                without_end_bytes(&stopped),
                without_end_bytes(&marked),
                "{output:?}"
            );
            assert_eq!(bytes(&stopped), output);
        }
        assert_eq!(
            calls(&run(
                glm().closes_at_stop("call"),
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
                EngineFinish::Stop,
            )),
            [(r#"{"city": "Par"}"#.to_string(), Some(String::new()))]
        );
    }

    #[test]
    fn under_the_row_any_finish_but_stop_leaves_the_call_cut() {
        // A driver hands a stop sequence that matched inside the call over as `Other`: the row
        // does not take it for the stop token, and the call keeps what arrived, open.
        for finish in [
            EngineFinish::Other("stop".to_string()),
            EngineFinish::Length,
            EngineFinish::Abort,
        ] {
            let events = run(glm().closes_at_stop("call"), LAST_CALL, finish);
            assert_eq!(
                calls(&events),
                [(
                    r#"{"city": "Paris", "days": 3"#.to_string(),
                    Some(String::new())
                )]
            );
        }
    }

    /// Muse Glimmer's messages, spelled from the response template of
    /// `meta-models/Muse-Glimmer-30B` (its `tokenizer_config.json`): a thought addressed to the
    /// model itself, an answer addressed to the user, and nothing kept outside them. The calls are
    /// left out here.
    fn muse(framing: bool) -> Format {
        let format = Format::new("muse")
            .terminal("thought", "to=self<|message|>")
            .terminal("answer", "to=user<|message|>")
            .terminal("eot", "<|eot|>")
            .terminal("eom", "<|eom|>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .transition("outside", "thought", "reasoning")
            .transition("outside", "answer", "content")
            .transition("reasoning", "eom", "outside")
            .transition("content", "eot", "outside")
            .transition("content", "eom", "outside");
        if framing {
            format.framing("outside")
        } else {
            format
        }
    }

    /// The text of the events of one kind: content, reasoning, dropped as wrapping, and malformed.
    fn kinds(events: &[Event]) -> [String; 4] {
        let mut kinds: [String; 4] = Default::default();
        for event in events {
            let (kind, text) = match event {
                Event::Content(text) => (0, text),
                Event::Reasoning(text) => (1, text),
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => (2, text),
                Event::Malformed { text, .. } => (3, text),
                _ => continue,
            };
            kinds[kind].push_str(&text.text);
        }
        kinds
    }

    #[test]
    fn a_framing_state_drops_its_text_and_its_rows_still_move_the_engine() {
        let output = " junk <|eom|>to=self<|message|>Plan.<|eom|><|start|>assistant \
                      to=user<|message|>It is sunny.<|eot|> after";
        for finish in [
            EngineFinish::Stop,
            EngineFinish::Length,
            EngineFinish::Abort,
        ] {
            let events = run(muse(true), output, finish.clone());
            let [content, reasoning, dropped, malformed] = kinds(&events);
            assert_eq!(content, "It is sunny.", "{finish:?}");
            assert_eq!(reasoning, "Plan.", "{finish:?}");
            assert_eq!(
                dropped,
                " junk <|eom|>to=self<|message|><|eom|><|start|>assistant to=user<|message|>\
                 <|eot|> after",
                "{finish:?}: the text outside, a terminal with no row there and the markers"
            );
            assert_eq!(malformed, "", "{finish:?}");
            assert_eq!(bytes(&events), output);
            // Without the row the text outside is the answer's.
            let [content, ..] = kinds(&run(muse(false), output, finish.clone()));
            assert_eq!(
                content, " junk <|eom|><|start|>assistant It is sunny. after",
                "{finish:?}"
            );
        }
    }

    #[test]
    fn what_the_scanner_held_in_a_framing_state_is_dropped_at_every_finish() {
        // A marker the end cut short, and text that only looked like the start of one.
        for (output, held) in [
            (
                "to=self<|message|>Plan.<|eom|><|start|>assistant to=us",
                "to=us",
            ),
            ("to=user<|message|>Hi<|eot|>to=", "to="),
        ] {
            for finish in [
                EngineFinish::Stop,
                EngineFinish::Length,
                EngineFinish::Abort,
                EngineFinish::Other("stop_sequence".to_string()),
            ] {
                let events = run(muse(true), output, finish.clone());
                let [_, _, dropped, malformed] = kinds(&events);
                assert!(
                    dropped.ends_with(held),
                    "{output:?} {finish:?}: {dropped:?}"
                );
                assert_eq!(malformed, "", "{output:?} {finish:?}");
                assert_eq!(bytes(&events), output);
            }
        }
    }

    #[test]
    fn a_framing_wrapper_state_drops_what_it_would_report() {
        let wrapped = |framing: bool| {
            let format = Format::new("wrapped")
                .terminal("open", "<a>")
                .terminal("close", "</a>")
                .state("between", Emits::Wrapper)
                .state("inside", Emits::Content)
                .transition("between", "open", "inside")
                .transition("inside", "close", "between");
            if framing {
                format.framing("between")
            } else {
                format
            }
        };
        let output = "x <a>kept</a> y";
        let [content, _, dropped, malformed] =
            kinds(&run(wrapped(true), output, EngineFinish::Stop));
        assert_eq!(
            [content, dropped, malformed],
            ["kept".to_string(), "x <a></a> y".to_string(), String::new()]
        );
        let [_, _, _, malformed] = kinds(&run(wrapped(false), output, EngineFinish::Stop));
        assert_eq!(
            malformed, "xy",
            "without the row the wrapping's text is reported"
        );
    }

    /// Gemma 4's turn, spelled from the response template of `google/gemma-4-31B-it` (its
    /// `tokenizer_config.json`): two anchors, a thought, the calls, and the content outside them.
    fn gemma_turn() -> Format {
        Format::new("gemma turn")
            .terminal("think_open", "<|channel>thought\n")
            .terminal("think_close", "<channel|>")
            .terminal("call_open", "<|tool_call>")
            .terminal("call_close", "<tool_call|>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "call")
            .transition("call", "call_close", "content")
            .calls(CallSyntax::Gemma(gemma::Spelling::GEMMA4))
            .turn_anchor("<|turn>model\n")
            .turn_anchor("<tool_response|>")
    }

    #[test]
    fn the_turn_starts_after_the_last_anchor() {
        // After a tool's response the template opens a thought: the output starts inside it.
        let prompt = "<|turn>user\nWeather?<turn|>\n<|turn>model\n<|tool_call>call:get_weather{\
                      city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{\
                      temp:20}<tool_response|><|channel>thought\n";
        let events = run_after(gemma_turn(), prompt, "It is warm.<channel|>20 degrees.");
        assert_eq!(events.first(), Some(&Event::ReasoningStart));
        let [content, reasoning, ..] = kinds(&events);
        assert_eq!([content, reasoning], ["20 degrees.", "It is warm."]);
        // A thought opened in an earlier turn moves nothing.
        let later = "<|turn>model\n<|channel>thought\nPlan<turn|>\n<|turn>user\nAgain<turn|>\n\
                     <|turn>model\n";
        let [content, reasoning, ..] = kinds(&run_after(gemma_turn(), later, "Hello."));
        assert_eq!([content, reasoning], ["Hello.", ""]);
    }

    #[test]
    fn a_prompt_without_an_anchor_replays_nothing() {
        // The prompt opens a thought but holds no anchor: the output starts in the first state.
        let [content, reasoning, ..] =
            kinds(&run_after(gemma_turn(), "<|channel>thought\n", "Hello."));
        assert_eq!([content, reasoning], ["Hello.", ""]);
    }

    #[test]
    fn the_output_starts_in_the_state_its_anchor_names() {
        // TRL's DeepSeek R1 distill template: the anchor ends with the thought's opener.
        let distill = || {
            Format::new("r1 distill")
                .terminal("think_close", "</think>")
                .state("content", Emits::Content)
                .state("reasoning", Emits::Reasoning)
                .transition("reasoning", "think_close", "content")
                .turn_anchor_starting_in("<\u{ff5c}Assistant\u{ff5c}><think>\n", "reasoning")
        };
        let prompt = "<\u{ff5c}User\u{ff5c}>2+2?<\u{ff5c}Assistant\u{ff5c}><think>\n";
        let events = run_after(distill(), prompt, "Four.</think>\n\n4");
        assert_eq!(events.first(), Some(&Event::ReasoningStart));
        let [content, reasoning, ..] = kinds(&events);
        assert_eq!([content, reasoning], ["\n\n4", "Four."]);
    }

    /// ERNIE's turn, spelled from transformers v5.18.0's `tests/utils/test_chat_parsing.py`
    /// (`ernie_template`): the thought opens with `(?:^|<think>\s*)`, the answer with
    /// `<response>\n`, a call with `<tool_call>`, and nothing is kept outside them.
    fn ernie() -> Format {
        Format::new("ernie")
            .terminal("think_open", "<think>")
            .terminal("think_close", "</think>")
            .terminal("answer_open", "<response>\n")
            .terminal("answer_close", "</response>")
            .terminal("call_open", "<tool_call>")
            .terminal("call_close", "</tool_call>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Json)
            .framing("outside")
            .turn_anchor("Assistant:")
            .starts_in("reasoning")
    }

    #[test]
    fn a_turn_that_starts_with_text_starts_in_the_start_rows_state() {
        let events = run_after(
            ernie(),
            "User: Weather?\nAssistant:",
            "Let me think.\n</think>\n<response>\nIt is sunny.\n</response>",
        );
        assert_eq!(events.first(), Some(&Event::ReasoningStart));
        let [content, reasoning, ..] = kinds(&events);
        assert_eq!([content, reasoning], ["It is sunny.\n", "Let me think.\n"]);
    }

    #[test]
    fn a_terminal_into_another_state_wins_at_the_first_byte() {
        let [content, reasoning, ..] = kinds(&run_after(
            ernie(),
            "Assistant:",
            "<response>\nHello!\n</response>",
        ));
        assert_eq!([content, reasoning], ["Hello!\n", ""]);
        let events = run_after(
            ernie(),
            "Assistant:",
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>",
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ReasoningStart)));
        assert_eq!(calls(&events).len(), 1);
    }

    #[test]
    fn a_terminal_into_the_start_state_is_its_text_at_the_first_byte() {
        // The `^` alternative comes first, so it matches at the first byte and `<think>` is the
        // thought's text, as Hugging Face's parser reads it; a terminal with no row from the
        // turn's state starts the thought too, and is read there.
        let [_, reasoning, ..] = kinds(&run_after(
            ernie(),
            "Assistant:",
            "<think>\nPlan.\n</think>\n<response>\nHi\n</response>",
        ));
        assert_eq!(reasoning, "<think>\nPlan.\n");
        let events = run_after(
            ernie(),
            "Assistant:",
            "</think>\n<response>\nHi\n</response>",
        );
        assert_eq!(
            events[..3],
            [
                Event::ReasoningStart,
                Event::Dropped {
                    text: Text::uncounted("</think>"),
                    why: DropReason::Wrapper,
                },
                Event::ReasoningEnd,
            ]
        );
        // After the first byte `<think>` opens the thought as any terminal does.
        let [content, reasoning, ..] = kinds(&run_after(
            ernie(),
            "Assistant:",
            "<response>\nHi\n</response><think>More.</think>",
        ));
        assert_eq!([content, reasoning], ["Hi\n", "More."]);
    }

    #[test]
    fn the_prompt_decides_the_start_when_it_writes_the_turns_first_byte() {
        // The generation prompt ends with a space after the anchor: the thought starts there, so
        // the answer's marker in the output is the thought's text.
        let [content, reasoning, ..] = kinds(&run_after(
            ernie(),
            "Assistant: ",
            "<response>\nHi\n</response>",
        ));
        assert_eq!([content, reasoning], ["", "<response>\nHi\n</response>"]);
        // An anchor that names its own state leaves no start to choose.
        let named = || {
            ernie()
                .terminal("thought_anchor", "Thinking:")
                .turn_anchor_starting_in("Answer:", "content")
        };
        let [content, reasoning, ..] = kinds(&run_after(named(), "Answer:", "Hi"));
        assert_eq!([content, reasoning], ["Hi", ""]);
    }

    #[test]
    fn a_held_first_piece_decides_once_it_is_known() {
        for (pieces, expected) in [
            (["<resp", "onse>\nHi"], ["Hi", ""]),
            (["<resp", "ite"], ["", "<respite"]),
        ] {
            let mut engine = Engine::new(ernie(), declared());
            let mut out = Events::new();
            for piece in pieces {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: piece,
                            spans: &[],
                        },
                        &mut out,
                    )
                    .expect("delta");
            }
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("end");
            let [content, reasoning, ..] = kinds(&out.drain());
            assert_eq!([content, reasoning], expected, "{pieces:?}");
        }
    }

    /// Gemma 4's turn with its stays trimmed and the content's non-stop close, `<|tool_response>`,
    /// a row back into the content that drops the marker.
    fn gemma_trimmed() -> Format {
        gemma_turn()
            .terminal("response_open", "<|tool_response>")
            .transition("content", "response_open", "content")
            .trims("content")
            .trims("reasoning")
    }

    /// The text of the whitespace a trimmed stay dropped.
    fn trimmed_space(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Whitespace,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Replays `output` after `prompt` cut at every two-way split and character by character,
    /// and checks that each cut says what the whole does, and that the events account for the
    /// output's bytes.
    fn same_at_every_cut(format: &Format, prompt: &str, output: &str) -> [String; 4] {
        let replay = |cuts: &[usize]| {
            let mut engine = Engine::new(format.clone(), declared());
            let mut out = Events::new();
            engine
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: prompt,
                    },
                    &mut out,
                )
                .expect("prompt");
            let mut from = 0;
            for &cut in cuts.iter().chain(std::iter::once(&output.len())) {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: &output[from..cut],
                            spans: &[],
                        },
                        &mut out,
                    )
                    .expect("delta");
                from = cut;
            }
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("end");
            let events = out.drain();
            assert_eq!(bytes(&events), output, "{cuts:?}");
            (kinds(&events), trimmed_space(&events), bytes(&events))
        };
        let whole = replay(&[]);
        let starts: Vec<usize> = output.char_indices().map(|(at, _)| at).skip(1).collect();
        for cut in &starts {
            assert_eq!(replay(&[*cut]), whole, "{output:?} cut at {cut}");
        }
        assert_eq!(replay(&starts), whole, "{output:?} character by character");
        whole.0
    }

    #[test]
    fn a_trimmed_stay_drops_the_whitespace_at_its_edges_and_keeps_the_rest() {
        let format = gemma_trimmed();
        let [content, reasoning, ..] = same_at_every_cut(
            &format,
            "<|turn>model\n",
            "<|channel>thought\n The user\n wants  the weather. \n<channel|>\n\nIt is\n\nsunny. ",
        );
        assert_eq!(
            [content, reasoning],
            ["It is\n\nsunny.", "The user\n wants  the weather."]
        );
        let events = run_after(format, "<|turn>model\n", "\n\nIt is sunny.\n\n");
        assert_eq!(trimmed_space(&events), "\n\n\n\n");
    }

    #[test]
    fn whitespace_is_what_pythons_strip_removes() {
        // No-break and em spaces, and the separators U+001C to U+001F that `str.isspace` takes,
        // are trimmed; a zero-width space is not whitespace there, and stays.
        let [content, ..] = same_at_every_cut(
            &gemma_trimmed(),
            "",
            "\u{a0}\u{1c}Hi\u{2003}there\u{2003}\u{1f}",
        );
        assert_eq!(content, "Hi\u{2003}there");
        let [content, ..] = kinds(&run(
            gemma_trimmed(),
            "\u{200b}Hi\u{200b}",
            EngineFinish::Stop,
        ));
        assert_eq!(content, "\u{200b}Hi\u{200b}");
        // Python's whitespace, recorded from Python 3.12: the code points `str.isspace` takes.
        let python: Vec<u32> = [
            0x09..=0x0d,
            0x1c..=0x20,
            0x85..=0x85,
            0xa0..=0xa0,
            0x1680..=0x1680,
            0x2000..=0x200a,
            0x2028..=0x2029,
            0x202f..=0x202f,
            0x205f..=0x205f,
            0x3000..=0x3000,
        ]
        .into_iter()
        .flatten()
        .collect();
        let ours: Vec<u32> = (0..=0x10ffff)
            .filter_map(char::from_u32)
            .filter(|&c| is_strip_space(c))
            .map(u32::from)
            .collect();
        assert_eq!(ours, python);
    }

    #[test]
    fn each_stay_is_trimmed_on_its_own() {
        // The content's own close is a row back into it: two stays, each trimmed.
        let [content, ..] = same_at_every_cut(
            &gemma_trimmed(),
            "<|turn>model\n",
            "Hi <|tool_response> there\n",
        );
        assert_eq!(content, "Hithere");
        // A call between two answers ends one stay and the next starts after it.
        let events = run(
            gemma_trimmed(),
            "Let me check.\n<|tool_call>call:get_weather{\
             city:<|\"|>Paris<|\"|>}<tool_call|>\n Done. ",
            EngineFinish::Stop,
        );
        let [content, ..] = kinds(&events);
        assert_eq!(content, "Let me check.Done.");
        assert_eq!(trimmed_space(&events), "\n\n  ");
    }

    #[test]
    fn a_stay_the_prompt_began_keeps_its_leading_whitespace() {
        let prompt = "<|turn>model\n<|channel>thought\nI think";
        let [_, reasoning, ..] = same_at_every_cut(&gemma_trimmed(), prompt, " more. <channel|>");
        assert_eq!(reasoning, " more.");
        let blank = "<|turn>model\n<|channel>thought\n  ";
        let [_, reasoning, ..] = same_at_every_cut(&gemma_trimmed(), blank, " more. <channel|>");
        assert_eq!(reasoning, "more.");
    }

    #[test]
    fn an_ignored_terminal_keeps_its_place_in_a_held_run() {
        let format = || {
            Format::new("separated")
                .terminal("open", "<a>")
                .ignores("separator", "|")
                .state("content", Emits::Content)
                .trims("content")
        };
        for (output, content, dropped) in [
            ("Hi | there |", "Hi  there", "||"),
            ("Hi |", "Hi", "|"),
            ("| Hi", "Hi", "|"),
        ] {
            let events = run(format(), output, EngineFinish::Stop);
            let [shown, _, wrapping, _] = kinds(&events);
            assert_eq!(
                [shown.as_str(), wrapping.as_str()],
                [content, dropped],
                "{output:?}"
            );
            assert_eq!(bytes(&events), output);
        }
    }

    #[test]
    fn a_content_stays_lead_is_content_before_text_and_the_templates_spacing_otherwise() {
        let format = qwen3(CallSyntax::Json);
        let call = |city: &str| {
            format!(
                "<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"{city}\"}}}}\n\
                 </tool_call>"
            )
        };
        // The spacing between the thought's close and the first call, and between two calls,
        // is the template's: dropped as whitespace, the same whole and at every cut.
        let output = format!(
            "<think>\nplan\n</think>\n\n{}\n{}",
            call("Paris"),
            call("Rome")
        );
        let [content, reasoning, ..] = same_at_every_cut(&format, "", &output);
        assert_eq!([content.as_str(), reasoning.as_str()], ["", "\nplan\n"]);
        assert_eq!(
            trimmed_space(&run(format.clone(), &output, EngineFinish::Stop)),
            "\n\n\n"
        );
        // Before text it is the text's, content with it; after text, whitespace is content as
        // it arrives, the newline before the call included.
        let output = format!(
            "<think>\nplan\n</think>\n\nSure.\n{}\n Done.",
            call("Paris")
        );
        let [content, ..] = same_at_every_cut(&format, "", &output);
        assert_eq!(content, "\n\nSure.\n\n Done.");
        assert_eq!(
            trimmed_space(&run(format.clone(), &output, EngineFinish::Stop)),
            ""
        );
        // A stay no text ends, at the end of the output or at a row that opens no call, showed
        // nothing: its whitespace is the template's.
        let output = "<think>\nplan\n</think>\n\n";
        let [content, ..] = same_at_every_cut(&format, "", output);
        assert_eq!(content, "");
        assert_eq!(
            trimmed_space(&run(format.clone(), output, EngineFinish::Stop)),
            "\n\n"
        );
        let output = "<think>\na\n</think>\n\n<think>\nb\n</think>\n";
        let [content, reasoning, ..] = same_at_every_cut(&format, "", output);
        assert_eq!([content.as_str(), reasoning.as_str()], ["", "\na\n\nb\n"]);
        assert_eq!(
            trimmed_space(&run(format.clone(), output, EngineFinish::Stop)),
            "\n\n\n"
        );
        // The stay the prompt began: text in the prompt's tail makes the whitespace content.
        let output = format!("\n{}", call("Paris"));
        let [content, ..] = same_at_every_cut(&format, "<|im_start|>assistant\n", &output);
        assert_eq!(content, "");
        let [content, ..] = same_at_every_cut(&format, "<|im_start|>assistant\nSure,", &output);
        assert_eq!(content, "\n");
    }

    #[test]
    fn a_held_lead_leaves_no_content_delta_before_a_call_and_leaves_with_the_text_after_it() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>";
        // The content of each feed's events, the end's included, empty ones left out: the
        // content deltas a client of the stream receives.
        let deltas = |pieces: &[&str]| -> Vec<String> {
            let mut engine = Engine::new(qwen3(CallSyntax::Json), declared());
            let mut out = Events::new();
            let mut deltas = Vec::new();
            for piece in pieces {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: piece,
                            spans: &[],
                        },
                        &mut out,
                    )
                    .expect("delta");
                let [content, ..] = kinds(&out.drain());
                deltas.push(content);
            }
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("end");
            let [content, ..] = kinds(&out.drain());
            deltas.push(content);
            deltas.retain(|delta| !delta.is_empty());
            deltas
        };
        // Nothing before the first call, nothing between the calls.
        assert_eq!(
            deltas(&["<think>", "\nplan\n", "</think>", "\n", "\n", call, "\n", call]),
            Vec::<String>::new()
        );
        // With text after the thought the spacing leaves with the text; after a call, the same.
        assert_eq!(
            deltas(&[
                "<think>", "\nplan\n", "</think>", "\n\n", "Sure", ".\n", call, "\n", "Done."
            ]),
            ["\n\nSure", ".\n", "\nDone."]
        );
        // Nothing after a thought nothing follows: the whole answer has no content either.
        assert_eq!(
            deltas(&["<think>", "\nplan\n", "</think>", "\n\n"]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_shown_marker_is_text_the_lead_before_it_and_the_whitespace_after_it_belong_to() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>";
        let output = format!("<think>\nplan\n</think>\n\n{call}");
        // The variant that does not separate reasoning shows the thought's markers as content:
        // the spacing after `</think>` follows text the client sees, so it is content, and the
        // content is the model's bytes up to the call.
        let format = qwen3(CallSyntax::Json).without_reasoning();
        let [content, reasoning, ..] = same_at_every_cut(&format, "", &output);
        assert_eq!(
            [content.as_str(), reasoning.as_str()],
            ["<think>\nplan\n</think>\n\n", ""]
        );
        assert_eq!(trimmed_space(&run(format, &output, EngineFinish::Stop)), "");
        // The variant that reads no calls shows a call's markers: the lead held before
        // `<tool_call>` is released as content with it, and the content is every byte after the
        // thought.
        let format = qwen3(CallSyntax::Json).without_calls();
        let [content, reasoning, ..] = same_at_every_cut(&format, "", &output);
        assert_eq!(
            [content.as_str(), reasoning.as_str()],
            [format!("\n\n{call}").as_str(), "\nplan\n"]
        );
        assert_eq!(trimmed_space(&run(format, &output, EngineFinish::Stop)), "");
    }

    #[test]
    fn an_ignored_terminal_in_a_held_lead_keeps_its_place_and_the_leads_tokens_are_counted() {
        let format = || {
            Format::new("led")
                .terminal("open", "<a>")
                .terminal("close", "</a>")
                .ignores("separator", "|")
                .state("content", Emits::Content)
                .state("call", Emits::Arguments)
                .transition("content", "open", "call")
                .transition("call", "close", "content")
                .calls(CallSyntax::Json)
        };
        let call = "<a>{\"name\": \"get_weather\", \"arguments\": {}}</a>";
        for (output, content, dropped) in [
            (format!("\n|\n{call}"), "", "|<a></a>"),
            (format!("\n|\nHi{call}"), "\n\nHi", "|<a></a>"),
        ] {
            let events = run(format(), &output, EngineFinish::Stop);
            let [shown, _, wrapping, _] = kinds(&events);
            assert_eq!(
                [shown.as_str(), wrapping.as_str()],
                [content, dropped],
                "{output:?}"
            );
            assert_eq!(bytes(&events), output, "{output:?}");
        }
        // Each token of the lead is counted in the event that carries its first byte: the
        // whitespace events the lead leaves as, the separator's in its own.
        let tokens = [
            "\n",
            "|",
            "\n",
            "<a>",
            "{\"name\": \"get_weather\", ",
            "\"arguments\": {}}",
            "</a>",
        ];
        let text: String = tokens.concat();
        let mut at = 0;
        let spans: Vec<TokenSpan> = tokens
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
            .collect();
        let mut engine = Engine::new(format(), declared());
        let mut out = Events::new();
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: &text,
                    spans: &spans,
                },
                &mut out,
            )
            .expect("delta");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        let lead: Vec<(String, Option<u32>)> = events
            .iter()
            .take(3)
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some((text.text.clone(), text.tokens)),
                _ => None,
            })
            .collect();
        assert_eq!(
            lead,
            [
                ("\n".to_string(), Some(1)),
                ("|".to_string(), Some(1)),
                ("\n".to_string(), Some(1)),
            ]
        );
        let counted: u32 = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Reasoning(text) => text.tokens,
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.tokens,
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.tokens,
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
            })
            .sum();
        assert_eq!(counted, tokens.len() as u32, "{events:?}");
    }

    #[test]
    fn an_ignored_terminal_inside_a_held_call_keeps_its_place() {
        // Inkling holds a message's header until the marker says what the message is; the turn's
        // end token inside the header leaves after the bytes before it, not before them, and
        // the call still reads. Where nothing is held it is dropped where it stands.
        let prompt = "<|message_model|>";
        let held = concat!(
            "get_<|content_model_end_sampling|>weather<|content_invoke_tool_json|>",
            "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>"
        );
        let whole = run_after(inkling(), prompt, held);
        assert_eq!(bytes(&whole), held);
        assert_eq!(
            calls(&whole),
            [(r#"{"city":"Paris"}"#.to_string(), Some("}".to_string()))]
        );
        let [_, _, wrapping, malformed] = kinds(&whole);
        assert!(
            wrapping.contains("<|content_model_end_sampling|>"),
            "{whole:?}"
        );
        assert_eq!(malformed, "");
        for cut in (1..held.len()).filter(|&cut| held.is_char_boundary(cut)) {
            let mut engine = Engine::new(inkling(), declared());
            let mut out = Events::new();
            for input in [
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                Input::Delta {
                    token_ids: &[],
                    text: &held[..cut],
                    spans: &[],
                },
                Input::Delta {
                    token_ids: &[],
                    text: &held[cut..],
                    spans: &[],
                },
                Input::End {
                    finish: EngineFinish::Stop,
                },
            ] {
                engine.feed(input, &mut out).expect("in order");
            }
            let events = out.drain();
            assert_eq!(bytes(&events), held, "cut at {cut}");
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
        }
        let unheld = "<|content_text|>Hi<|content_model_end_sampling|> there<|end_message|>";
        let events = run_after(inkling(), prompt, unheld);
        assert_eq!(bytes(&events), unheld);
        let [content, _, wrapping, _] = kinds(&events);
        assert_eq!(content, "Hi there");
        assert_eq!(
            wrapping,
            "<|content_text|><|content_model_end_sampling|><|end_message|>"
        );
        // A syntax that hands held bytes back as content (the bare list, which holds `[` until
        // it knows whether a list follows) leaves the terminal in an event of its own, not in
        // the text the client reads.
        let list = xlam().ignores("sep", "|");
        let events = run(list, "[|Sure, done", EngineFinish::Stop);
        assert_eq!(bytes(&events), "[|Sure, done");
        let [content, _, wrapping, _] = kinds(&events);
        assert_eq!(content, "[Sure, done");
        assert_eq!(wrapping, "|");
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::Dropped { text, why: DropReason::Wrapper } if text.text == "|"
            )),
            "{events:?}"
        );
    }

    #[test]
    fn the_row_closes_a_call_of_another_syntax_the_same_way() {
        let tagged = || qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine));
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n";
        let closed = run(tagged().closes_at_stop("calls"), output, EngineFinish::Stop);
        assert_eq!(
            calls(&closed),
            [(r#"{"city": "Paris"}"#.to_string(), Some(String::new()))]
        );
        assert_eq!(bytes(&closed), output);
        let cut = run(tagged(), output, EngineFinish::Stop);
        assert_eq!(
            calls(&cut),
            [(r#"{"city": "Paris""#.to_string(), Some(String::new()))]
        );
    }
}
