//! The engine: one [`Parser`] that runs any [`Format`]. The scanner splits the model's text into
//! text and the format's terminals, holding back a half-arrived terminal; the table says which
//! state the text is in and where a terminal moves the engine; inside an arguments state the call
//! syntax's assembler (`json::Assembler` or `tagged::Assembler`) turns the call into its
//! events, streaming the model's own argument bytes.
//!
//! Every byte of the output lands in exactly one event: a terminal that moves the engine as
//! `Dropped { Wrapper }`, the whitespace between a call and the terminal that closes it as
//! `Dropped { Wrapper }` too except between two runs of other text there, which is the model's and
//! rides in the second run's `Malformed` event, any other text there as `Malformed`, run by run so
//! that where the chunks were cut changes nothing, the whitespace a content stay begins with as
//! `Dropped { Whitespace }` when no text follows it in the stay (the template's spacing after a
//! thought's close, between two calls, or after a thought nothing follows, held until what
//! follows says whose it is) unless the table keeps the state's text as written, and the rest as
//! content, reasoning, or the call's events.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/engine.md`.

use std::{borrow::Cow, collections::VecDeque, ops::Range, sync::Arc};

use crate::{
    event::{
        release_between, runs_at, DropReason, Event, Events, FinishReason, MalformedReason, Text,
    },
    format::{CallId, CallSyntax, Emits, Format, Replay},
    gemma,
    ids::CallIds,
    input::{EngineFinish, Input},
    json,
    markers::{Piece, Scanned, Scanner},
    parser::{ParseError, Parser},
    pythonic,
    shape::{self, Values},
    tagged,
    tokens::Ledger,
    Declared,
};

/// Why a call region closed: its closing terminal (or the next opener) arrived; the stream
/// stopped, the model's own end of output (`EngineFinish::Stop`); or the stream was cut, by the
/// token limit, an abort or a finish the crate does not read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Closed {
    Marker,
    Stop,
    Cut,
}

/// A [`Format`] running as a [`Parser`]. One per generated choice.
#[derive(Debug)]
pub struct Engine {
    /// The table, shared with every engine that runs it.
    format: Arc<Format>,
    /// The request's tools, which type a tagged call's values and say which functions a call may
    /// name.
    declared: Declared,
    /// The request's tools in the order of their names, as positions in `declared`, for a request
    /// with more tools than a call's start searches directly: built at the first call's start
    /// (one allocation, for an output with a call) and searched per call ([`Self::declares`]).
    /// A plain option behind `&mut self`, so the engine stays `Sync`.
    names: Option<Box<[usize]>>,
    scanner: Scanner,
    state: usize,
    /// The call being assembled while the engine is in an arguments state.
    call: Option<Box<dyn CallAssembler>>,
    calls: u32,
    /// Where the output's call ids count from.
    ids: CallIds,
    /// The ids this output's calls were given, in order: a table that carries the model's id
    /// falls back to the minted one where a header repeats an id already given.
    given_ids: Vec<String>,
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
/// pieces ([`Engine::push_carried_out`]). The output's bytes leave in their order whatever the syntax
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

/// The stay in a state, from the row that entered it to the row that leaves it, as trimming, the
/// held lead and the wrapping read it: whether text other than whitespace has arrived, and the
/// run held. A trimmed stay holds the run after its last text that is not whitespace; a content
/// stay holds the run before its first ([`Engine::lead`]); a wrapping stay (between a block's
/// calls, after a call's object) holds the run after its last text ([`Engine::wrapping`]).
#[derive(Debug, Default)]
struct Stay {
    begun: bool,
    held: Vec<Held>,
}

/// Bytes a stay holds, in the output's order: a trimmed stay's run after its last text that is not
/// whitespace, a content stay's lead before its first, or a wrapping stay's run after its last.
#[derive(Debug)]
enum Held {
    /// Whitespace: the state's text if more text follows in the stay, dropped if the stay ends.
    Space(String),
    /// A terminal the table ignores, inside the run: dropped wherever it stands, in its place.
    Ignored(String),
}

/// A text event of `like`'s kind carrying `text`, uncounted: the piece of a split text event.
fn text_event_like<'d>(like: &Event<'_>, text: Cow<'d, str>) -> Event<'d> {
    let text = Text::uncounted(text);
    match like {
        Event::Reasoning(_) => Event::Reasoning(text),
        Event::Dropped { why, .. } => Event::Dropped { text, why: *why },
        Event::Malformed { why, .. } => Event::Malformed { text, why: *why },
        _ => Event::Content(text),
    }
}

/// The text an event carries, for the events that carry output bytes.
fn event_text<'e, 'd>(event: &'e mut Event<'d>) -> Option<&'e mut Text<'d>> {
    match event {
        Event::Content(text) | Event::Reasoning(text) => Some(text),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => Some(text),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => Some(source),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
    }
}

/// `text[range]` as `text` is: a slice of a borrowed run stays borrowed from the delta, a slice
/// of an owned run is copied. The pieces of a run the engine cuts (a lead, a body, a tail, the
/// runs of a wrapper) keep the run's claim on the delta.
fn part<'d>(text: &Cow<'d, str>, range: Range<usize>) -> Cow<'d, str> {
    match text {
        Cow::Borrowed(text) => Cow::Borrowed(&text[range]),
        Cow::Owned(text) => Cow::Owned(text[range].to_owned()),
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

    /// The whitespace the region ends with after its last call, which is the template's wrapping
    /// and nothing of the call's: the engine drops it as it drops wrapping everywhere else, before
    /// the region's close, so a region's common end costs no event of the syntax's. Taken from
    /// the syntax, so neither ending says it again. Every syntax but the tagged one hands back
    /// nothing and reports what it holds at its ending.
    fn trailing_whitespace(&mut self) -> String {
        String::new()
    }

    /// The region closed: `terminal` is the table's terminal that closed it, the call's own closing
    /// marker, the next call's opener or the block's close. The call ends as the syntax has it, and
    /// the answer says whether the terminal's bytes were taken as the call's end (`ToolCallEnd`'s
    /// source); otherwise the engine drops them as the region's.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool;

    /// The stream was cut inside the region: nothing is closed for the client, so arguments cut
    /// short never look complete.
    fn finish(self: Box<Self>, out: &mut Events);

    /// The stream stopped inside the region: the model's own end of output, not a cut.
    /// `as_marker` says that the table's stop row names the state ([`Format::closes_at_stop`]),
    /// so the stop stands for the closing marker the stop decoder hides, and the call ends as
    /// that marker would end it, with no bytes; without the row a syntax whose JSON is the
    /// model's own (JSON, Python, Gemma's tokens, a call given as data) ends the call as a cut
    /// ends it, since the model closed neither its call nor its value. The syntaxes that write
    /// the arguments' JSON for values the model wrote as text between tags have their own
    /// ending here ([`crate::tagged`]): the call ends as its closing marker would, a string
    /// still open is closed first, since the model wrote nothing after its bytes, and one
    /// `Malformed { OpenAtTheStop }` event says so.
    fn stop(self: Box<Self>, as_marker: bool, out: &mut Events) {
        if as_marker {
            self.close("", out);
        } else {
            self.finish(out);
        }
    }
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

/// The assembler for `syntax`'s next call, at `index` with its id from `ids`; `framed` says
/// whether a row of the table enters the arguments state, so markers frame the region; `values`
/// reads a shaped call's values and `declared` types them.
fn assembler(
    syntax: Option<CallSyntax>,
    index: u32,
    ids: CallIds,
    framed: bool,
    values: Values,
    declared: &Declared,
) -> Box<dyn CallAssembler> {
    // The id is minted where a syntax takes one at its making; the tagged and the list syntaxes
    // mint each call's as it starts.
    let id = || ids.id(index);
    match syntax {
        // `None` is unreachable: `Format::validate`, run by `Engine::new`, refuses a table with
        // an arguments state and no call syntax. The arm keeps the match total.
        Some(CallSyntax::Json) | None => Box::new(json::Assembler::new(index, id())),
        Some(CallSyntax::Tagged(_, placement)) => {
            Box::new(tagged::Assembler::new(index, ids, placement))
        }
        Some(CallSyntax::Keyed(tags)) => Box::new(tagged::keyed::Assembler::new(index, id(), tags)),
        // The list between markers (Olmo 3, LFM2.5) reads what the markers frame; the bare list
        // the output starts in (Llama 4) reads the turn's content before it and the wrapping
        // after it.
        Some(CallSyntax::Pythonic(dialect)) if framed => {
            Box::new(pythonic::Assembler::new(index, ids, dialect))
        }
        Some(CallSyntax::Pythonic(dialect)) => {
            Box::new(pythonic::Assembler::bare(index, ids, dialect))
        }
        // The list reader takes what frames the calls as one value: the list between markers
        // (Llama Nemotron) takes text in place of it as malformed; the bare list the output starts
        // in (xLAM) takes it for prose; the `json` table's output is one object, several with `;`
        // or whitespace between them, or a list, behind the family's call marker or not.
        Some(CallSyntax::JsonList) => Box::new(json::list::Assembler::new(
            index,
            ids,
            if framed {
                json::list::Framing::Framed
            } else {
                json::list::Framing::Bare
            },
        )),
        Some(CallSyntax::JsonAlone(opener)) => Box::new(json::list::Assembler::new(
            index,
            ids,
            json::list::Framing::ObjectOrList { opener },
        )),
        Some(CallSyntax::Xml) => Box::new(tagged::xml::Assembler::new(index, id())),
        Some(CallSyntax::Attributed(tags)) => {
            Box::new(tagged::attributed::Assembler::new(index, id(), tags))
        }
        // The named call: a tool's name, a marker, then the call's object or its arguments alone.
        Some(CallSyntax::NamedJson(marker)) => Box::new(json::named::Assembler::new(
            index,
            id(),
            marker,
            json::named::After::Object,
        )),
        Some(CallSyntax::NamedArguments { marker, .. }) => Box::new(json::named::Assembler::new(
            index,
            id(),
            marker,
            json::named::After::Arguments,
        )),
        Some(CallSyntax::ForcedArguments) => Box::new(json::forced::Assembler::new(index, id())),
        Some(CallSyntax::Gemma(spelling)) => Box::new(gemma::Assembler::new(index, id(), spelling)),
        // A shaped call opens through its terminal's header, which the engine reads before it
        // enters the call's state with that assembler (`Engine::open_shaped`); a table that
        // validates enters the state no other way.
        Some(CallSyntax::Shaped(shape)) => Box::new(ShapedCall {
            call: shape::Assembler::new(index, id(), shape, values, None)
                .with_call_id(CallId::Minted, &[]),
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
    /// tools, or a format that types nothing by them). The output's call ids count from a nonce
    /// drawn here ([`CallIds::fresh`]), which no other output shares.
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
    pub fn new(format: Format, declared: Declared) -> Self {
        Self::shared(Arc::new(format), declared)
    }

    /// [`Self::new`] over a table other engines run too, the registry's
    /// ([`Table::shared`](crate::formats::Table::shared)) or one the caller keeps: the engine
    /// holds the one copy. The output's call ids count from a nonce drawn here, as in
    /// [`Self::new`].
    ///
    /// # Panics
    ///
    /// As [`Self::new`]: a table [`Format::validate`] refuses.
    pub fn shared(format: Arc<Format>, declared: Declared) -> Self {
        Self::with_call_ids(format, declared, CallIds::fresh())
    }

    /// [`Self::shared`], with the output's call ids counting from `ids` in place of a nonce drawn
    /// now: for a replay that wants them reproducible.
    ///
    /// # Panics
    ///
    /// As [`Self::new`].
    #[expect(
        clippy::panic,
        reason = "a table the crate wrote that fails its own check is a programming error"
    )]
    pub fn with_call_ids(format: Arc<Format>, declared: Declared, ids: CallIds) -> Self {
        if let Err(why) = format.validate() {
            panic!("{why}");
        }
        let at_turn_start = format.start().is_some();
        let mut engine = Self {
            scanner: format.scanner(),
            format,
            declared,
            names: None,
            state: 0,
            call: None,
            calls: 0,
            ids,
            given_ids: Vec::new(),
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
    fn run(&mut self, scanned: &Scanned<'_>, end: Option<Closed>, out: &mut Events) {
        // The pieces are taken as scanned until a header decides; from then on, what the scan
        // that reads again gives and the pieces not reached yet stand in a queue, which nothing
        // is allocated for before.
        let mut pieces = scanned.pieces();
        let mut queue = VecDeque::new();
        while let Some(piece) = pieces.next() {
            if let Some(again) = self.step(piece, end, out) {
                queue.extend(pieces.by_ref());
                self.scan_again(again, &mut queue, end.is_some());
                break;
            }
        }
        loop {
            while let Some(piece) = queue.pop_front() {
                if let Some(again) = self.step(piece, end, out) {
                    self.scan_again(again, &mut queue, end.is_some());
                }
            }
            if end.is_none() {
                return;
            }
            let Some(again) = self.decide_at_end(out) else {
                return;
            };
            self.scan_again(again, &mut queue, true);
        }
    }

    /// Takes one piece: at the end of the stream (`end` says how it ended) a tag the end cut
    /// short is dropped, anything else is taken as the table says ([`Self::take`]). What a header
    /// that decided hands back to read again.
    fn step(&mut self, piece: Piece<'_>, end: Option<Closed>, out: &mut Events) -> Option<String> {
        match (piece, end) {
            (Piece::Text(text), Some(closed)) if self.cut_tag(&text) => {
                self.drop_cut_tag(text.into_owned(), closed, out);
                None
            }
            (piece, _) => self.take(piece, out),
        }
    }

    /// Scans `text` afresh, before the pieces in `queue`, and puts what it gives at the queue's
    /// front: the scanner reads it with what it held set aside, as a new scanner would. The
    /// queued pieces follow it, scanned again too while the new scan holds a possible terminal at
    /// their edge: once it holds nothing where the old scan began a piece, both cut what follows
    /// alike, since a scan from a piece's start reads nothing before it, and the scanner takes
    /// what it held back. When the queue runs out first, what was held comes after them, and the
    /// new scan stands; at the end of the stream it is finished.
    fn scan_again(&mut self, text: String, queue: &mut VecDeque<Piece<'_>>, at_end: bool) {
        let held = self.scanner.set_aside();
        let mut pieces = owned(&self.scanner.feed(&text));
        while !self.scanner.held().is_empty() {
            let Some(piece) = queue.pop_front() else {
                break;
            };
            let text = piece_text(&self.format, &piece);
            pieces.extend(owned(&self.scanner.feed(text)));
        }
        if queue.is_empty() {
            pieces.extend(owned(&self.scanner.feed(&held)));
            if at_end {
                pieces.extend(owned(&self.scanner.finish()));
            }
        } else {
            self.scanner.resume(held);
        }
        for piece in pieces.into_iter().rev() {
            queue.push_front(piece);
        }
    }

    /// Takes one piece, and says what to read again when a shaped call's header decided.
    fn take(&mut self, piece: Piece<'_>, out: &mut Events) -> Option<String> {
        if self.pending.is_some() {
            let text = piece_text(&self.format, &piece).to_string();
            return self.read_header(&text, out);
        }
        if let Piece::Marker(index) = piece {
            let leaves = self.format.next(self.state, index).is_some();
            if leaves && self.emits() == Emits::Arguments {
                let terminal = self.format.terminal_text(index).to_string();
                if !self.call_closes(&terminal) {
                    // The trailer does not end at the terminal: it closes nothing and is the
                    // call's text, as the template's closing pattern reads past it.
                    self.text(Cow::Owned(terminal), out);
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
    fn place(&mut self, piece: Piece<'_>, out: &mut Events) {
        self.open_turn(&piece, out);
        match piece {
            Piece::Text(text) => self.text(Cow::Owned(text.into_owned()), out),
            Piece::Marker(index) => self.terminal(index, out),
        }
    }

    /// Takes a delta that is text whole ([`Scanner::reads_as_text`]) where [`Self::place`] would
    /// take it as one text piece, without the piece: the common delta of a content or reasoning
    /// stay, and of a call's arguments, is one. A content or reasoning delta's events borrow the
    /// delta, which comes borrowed; an arguments delta's go to the assembler, whose events own
    /// their text; a scan's one piece comes as the scan gives it.
    fn place_text<'d>(&mut self, text: Cow<'d, str>, out: &mut Events<'d>) {
        self.open_turn(&Piece::Text("".into()), out);
        self.text(text, out);
    }

    /// The turn's first piece decides where the turn starts, for a table with a start row.
    fn open_turn(&mut self, piece: &Piece<'_>, out: &mut Events) {
        if self.at_turn_start {
            if let Some(state) = self.format.opening(self.state, piece) {
                self.at_turn_start = false;
                if state != self.state {
                    self.enter(state, out);
                }
            }
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
        let id = self.ids.id(self.calls);
        let values = self.format.value_reader();
        let call =
            shape::Assembler::new(self.calls, id, shape, values, terminal.chars().next_back())
                .with_call_id(self.format.call_id_source(), &self.given_ids);
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
        self.push_call_events(started.drain(), out);
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
        self.place(Piece::Text(first.into()), out);
        again
    }

    /// Text of the state the engine is in: borrowed from the delta when it came whole, owned when
    /// the scan cut it or the engine held it.
    fn text<'d>(&mut self, text: Cow<'d, str>, out: &mut Events<'d>) {
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
            Emits::Content if !self.stay.begun && !self.format.is_kept(self.state) => {
                self.lead(text, out);
            }
            Emits::Content | Emits::Reasoning => self.show(text, out),
            Emits::Arguments => {
                let Some(call) = &mut self.call else {
                    // An arguments state always has a call open; kept as content, not lost.
                    out.push_content(self.tokens.text(text));
                    return;
                };
                let mut assembled = Events::new();
                let taken = call.feed(&text, &self.declared, &mut assembled);
                self.region.fed += taken;
                self.push_call_events(assembled.drain(), out);
                if taken < text.len() {
                    let rest = part(&text, taken..text.len());
                    self.wrapping(rest, MalformedReason::TextAfterTheArguments, out);
                }
            }
            Emits::Wrapper => self.wrapping(text, MalformedReason::TextBetweenCalls, out),
        }
    }

    /// Text of the content or reasoning state the engine is in, as the client sees it.
    fn show<'d>(&mut self, text: Cow<'d, str>, out: &mut Events<'d>) {
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
    fn trimmed<'d>(&mut self, text: Cow<'d, str>, out: &mut Events<'d>) {
        let mut from = 0;
        if !self.stay.begun {
            let body = text.trim_start_matches(is_strip_space);
            let lead = text.len() - body.len();
            if lead > 0 {
                out.push(Event::Dropped {
                    text: self.tokens.text(part(&text, 0..lead)),
                    why: DropReason::Whitespace,
                });
            }
            if body.is_empty() {
                return;
            }
            self.stay.begun = true;
            from = lead;
        }
        let rest = &text[from..];
        let body = rest.trim_end_matches(is_strip_space).len();
        if body > 0 {
            self.release_stay(out);
            self.show(part(&text, from..from + body), out);
        }
        self.hold(&text[from + body..]);
    }

    /// Text at the start of a content stay, before the stay's first character that is not
    /// whitespace: the whitespace is held, since what follows says whose it is. Text makes it
    /// the text's, released as content before it; the stay ending without text makes it the
    /// template's spacing (between a thought's close and the first call, between two calls,
    /// after a thought nothing follows), dropped. The whole answer's message shows no content
    /// for whitespace alone, and a stream that showed it as it arrived carried what the whole
    /// answer does not.
    fn lead<'d>(&mut self, text: Cow<'d, str>, out: &mut Events<'d>) {
        let body = text.trim_start_matches(is_strip_space);
        let lead = text.len() - body.len();
        self.hold(&text[..lead]);
        if body.is_empty() {
            return;
        }
        self.stay.begun = true;
        self.release_stay(out);
        self.show(part(&text, lead..text.len()), out);
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
                Held::Space(space) => self.show(Cow::Owned(space), out),
                Held::Ignored(terminal) => out.push(Event::Dropped {
                    text: self.tokens.text(terminal),
                    why: DropReason::Wrapper,
                }),
            }
        }
    }

    /// The stay ends: the run it held is the template's and is dropped, a trimmed stay's run
    /// after its last text as much as a lead no text followed, and the run a wrapping stay held
    /// after its last text, the template's spacing before the terminal that ended the stay.
    fn end_stay(&mut self, out: &mut Events) {
        // A content or reasoning stay's run is whitespace the format treats as structural; a
        // wrapping stay's is the template's wrapping, as it is before the stay's first text.
        let spacing = match self.emits() {
            Emits::Content | Emits::Reasoning => DropReason::Whitespace,
            Emits::Arguments | Emits::Wrapper => DropReason::Wrapper,
        };
        for held in std::mem::take(&mut self.stay.held) {
            let (text, why) = match held {
                Held::Space(space) => (space, spacing),
                Held::Ignored(terminal) => (terminal, DropReason::Wrapper),
            };
            out.push(Event::Dropped {
                text: self.tokens.text(text),
                why,
            });
        }
    }

    /// Text where the template only wraps: after a call's object and before its closing terminal,
    /// or between a block's calls. Whitespace before the stay's first other text is the template's
    /// and is dropped as it arrives; each run of anything else is malformed with `why`, with the
    /// whitespace held since the last such run in front of it, which is the model's, between two
    /// pieces of its text ([`release_between`], the rule every reader of such text states); the
    /// whitespace after a run is held, the next run's when one follows and the template's spacing
    /// before the terminal that ends the stay otherwise, which [`Self::end_stay`] drops. One event
    /// per run, so a reader that counts the reason counts as before, and classifying run by run
    /// keeps the result the same wherever the chunks were cut.
    fn wrapping<'d>(&mut self, surplus: Cow<'d, str>, why: MalformedReason, out: &mut Events<'d>) {
        for (range, space) in runs_at(&surplus) {
            if space {
                if self.stay.begun {
                    self.hold(&surplus[range]);
                } else {
                    out.push(Event::Dropped {
                        text: self.tokens.text(part(&surplus, range)),
                        why: DropReason::Wrapper,
                    });
                }
                continue;
            }
            let mut before = self.held_before_text(why, out);
            self.release_wrapping(&mut before, part(&surplus, range), why, out);
        }
    }

    /// The run a wrapping stay held, as text follows it: the whitespace is the text's and comes
    /// back to stand in front of it; a terminal the table ignores among it is dropped in its
    /// place, the whitespace before it released on its own, so every byte keeps its order.
    fn held_before_text(&mut self, why: MalformedReason, out: &mut Events) -> String {
        let mut before = String::new();
        for held in std::mem::take(&mut self.stay.held) {
            match held {
                Held::Space(space) => before.push_str(&space),
                Held::Ignored(terminal) => {
                    self.release_wrapping(&mut before, Cow::Borrowed(""), why, out);
                    out.push(Event::Dropped {
                        text: self.tokens.text(terminal),
                        why: DropReason::Wrapper,
                    });
                }
            }
        }
        before
    }

    /// A piece of text where the template only wraps, released as the rule has it
    /// ([`release_between`]) with `before`, the whitespace held in front of it, and counted;
    /// text released begins the stay. The engine carries nothing before the stay's first text,
    /// which it dropped as it arrived, so the piece keeps its borrow on the delta there.
    fn release_wrapping<'d>(
        &mut self,
        before: &mut String,
        text: Cow<'d, str>,
        why: MalformedReason,
        out: &mut Events<'d>,
    ) {
        let mut released = self.stay.begun;
        let mut events = Events::new();
        release_between(before, &mut released, text, why, &mut events);
        for event in events.drain() {
            out.push(self.tokens.relabel(event));
        }
        self.stay.begun = released;
    }

    /// A terminal arrived: move where the table says, or keep it as text where the model put it.
    /// A terminal the table ignores is dropped where it stands, and moves nothing: inside a call
    /// whose assembler holds bytes it is placed where it arrived once the bytes around it leave
    /// ([`Region`]), and in a stay's held run (a trimmed stay's tail, a content stay's lead, the
    /// run a wrapping stay holds after its text) it keeps its place among them.
    fn terminal(&mut self, index: usize, out: &mut Events) {
        if self.format.ignored(index) {
            let text = self.format.terminal_text(index).to_string();
            if self.emits() == Emits::Arguments {
                if let Some(call) = &mut self.call {
                    let mut events = Events::new();
                    call.ignored(&text, &self.declared, &mut events);
                    self.push_call_events(events.drain(), out);
                    if self.region.out < self.region.fed {
                        self.region.carried.push((self.region.fed, text));
                        return;
                    }
                }
            }
            // Inside a stay's held run, the terminal keeps its place among the bytes held.
            if !self.stay.held.is_empty() {
                self.stay.held.push(Held::Ignored(text));
                return;
            }
            out.push(Event::Dropped {
                text: self.tokens.text(text),
                why: DropReason::Wrapper,
            });
            return;
        }
        let Some(next) = self.format.next(self.state, index) else {
            // No row leads out on it: the marker is the state's text, as the model wrote it.
            // Where that text is the table's own content, a `Malformed` event of no bytes stands
            // before it, so a reader counts a frame the table did not read and no byte moves; in
            // a thought the marker is the thought's text alone, as a table may read a thought's
            // opener there (a turn anchor's table), and so it stays in a state a request's
            // variant made content, where the table counted none (a thought's marker is the
            // thought's text, a call's header the call's); where the template discards the
            // state's text it is dropped as the text is. The run the stay holds (a lead, a
            // trimmed tail) is the text's, as the marker's text would release it: released
            // first, so the event stands right before the marker.
            let text = self.format.terminal_text(index).to_string();
            if self.emits() == Emits::Content
                && !self.format.is_framing(self.state)
                && !self.format.is_remade(self.state)
            {
                self.release_stay(out);
                out.push(Event::Malformed {
                    text: self.tokens.text(""),
                    why: MalformedReason::MarkerOutOfPlace,
                });
            }
            self.text(Cow::Owned(text), out);
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
            self.emits() == Emits::Arguments && self.close_call(Closed::Marker, &terminal, out);
        let shown = !taken && self.format.shown(index);
        if !taken {
            let text = self.tokens.text(terminal);
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
            self.ids,
            framed,
            self.format.value_reader(),
            &self.declared,
        )
    }

    /// Ends the call region: the assembler closes what arrived, and the result says whether it took
    /// `terminal`, the bytes that closed the region, as the call's end. A region its terminal
    /// closed, or the stop closed where the table's stop row stands for the marker, reports
    /// leftover bytes as a block without a complete call; `UnterminatedRegion` is kept for a
    /// region the end of the stream cut.
    fn close_call(&mut self, closed: Closed, terminal: &str, out: &mut Events) -> bool {
        let Some(mut call) = self.call.take() else {
            return false;
        };
        // The whitespace after the region's last call is the template's: it leaves as wrapping,
        // counted once where every call event is, and among ignored terminals it keeps its place.
        let trailing = call.trailing_whitespace();
        if !trailing.is_empty() {
            let dropped = Event::Dropped {
                text: Text::uncounted(trailing),
                why: DropReason::Wrapper,
            };
            self.push_carried_out(dropped, out);
        }
        let mut finished = Events::new();
        let mut as_marker = closed == Closed::Marker;
        let taken = match closed {
            Closed::Marker => call.close(terminal, &mut finished),
            Closed::Stop => {
                // The stop token that closes the call is hidden, so the call's end has no bytes. A
                // shaped call of one JSON value or object needs its trailer to end where the
                // stream does; tagged parameters close wherever it stands.
                as_marker = self.format.stop_closes(self.state) && call.closes("");
                call.stop(as_marker, &mut finished);
                false
            }
            Closed::Cut => {
                call.finish(&mut finished);
                false
            }
        };
        if taken {
            self.region.fed += terminal.len();
        }
        let mut finished = finished.drain();
        if as_marker {
            for event in &mut finished {
                if let Event::Malformed { why, .. } = event {
                    if *why == MalformedReason::UnterminatedRegion {
                        *why = MalformedReason::BlockWithoutACompleteCall;
                    }
                }
            }
        }
        self.push_call_events(finished, out);
        // An ignored terminal no event carried out leaves here, before the region's close.
        for (_, marker) in std::mem::take(&mut self.region.carried) {
            out.push(Event::Dropped {
                text: self.tokens.text(marker),
                why: DropReason::Wrapper,
            });
        }
        self.region = Region::default();
        taken
    }

    /// Tools up to this many are searched by name at each call's start as they are; more are
    /// indexed once per output at the first call's start and the index is searched. Measured on
    /// the crate's bench subjects (two tools, two calls), the direct search is free, and the one
    /// allocation an index costs moved the allocator's consolidation into the read.
    const NAMES_SEARCHED_DIRECTLY: usize = 16;

    /// Whether the request declares a function named `name`: up to
    /// [`Self::NAMES_SEARCHED_DIRECTLY`] tools, as they are; more, by the index built once per
    /// output at the first call's start. Asked only when the request declares tools at all.
    fn declares(&mut self, name: &str) -> bool {
        let tools = self.declared.tools();
        if tools.len() <= Self::NAMES_SEARCHED_DIRECTLY {
            return self.declared.declares(name);
        }
        let by_name = self.names.get_or_insert_with(|| {
            let mut positions: Vec<usize> = (0..tools.len()).collect();
            positions
                .sort_unstable_by(|a, b| tools[*a].function.name.cmp(&tools[*b].function.name));
            positions.into_boxed_slice()
        });
        by_name
            .binary_search_by(|at| tools[*at].function.name.as_str().cmp(name))
            .is_ok()
    }

    /// Pushes a call's events with their tokens counted, and counts each call that starts: the
    /// index the next call takes, and `Finish::tool_calls`. A syntax that writes several calls in
    /// one region starts each of them here. A call that names a function the request did not
    /// declare is announced first, by a `Malformed { UnknownTool }` event of no bytes right before
    /// its start, so a reader counts it at its offset; the call goes out as the syntax read it, and
    /// a request that declares no tools is not asked. An event's bytes are counted as carried out
    /// of the region, with the ignored terminals that arrived among them placed where they
    /// arrived.
    fn push_call_events<'d>(&mut self, events: Vec<Event<'d>>, out: &mut Events<'d>) {
        for event in events {
            if let Event::ToolCallStart { id, name, .. } = &event {
                self.calls += 1;
                self.given_ids.push(id.clone());
                if !self.declared.tools().is_empty() && !self.declares(name) {
                    let unknown = Event::Malformed {
                        text: self.tokens.text(""),
                        why: MalformedReason::UnknownTool,
                    };
                    self.push_carried_out(unknown, out);
                }
            }
            self.push_carried_out(event, out);
        }
    }

    /// Pushes `event` with its tokens counted and the ignored terminals that arrived among the
    /// bytes it carries placed where they arrived, and counts its bytes as carried out of the
    /// region. A call's event has them spliced into its source, which no client reads; a text
    /// event is split around them, each terminal a `Dropped { Wrapper }` event of its own, so that
    /// none joins text a client reads (a syntax that hands held bytes back as content, as the bare
    /// list does). Nearly every event has nothing among its bytes and goes out as it is, without
    /// a list of its own.
    fn push_carried_out<'d>(&mut self, mut event: Event<'d>, out: &mut Events<'d>) {
        let Some(text) = event_text(&mut event) else {
            out.push(self.tokens.relabel(event));
            return;
        };
        if text.text.is_empty() {
            out.push(self.tokens.relabel(event));
            return;
        }
        let from = self.region.out;
        let to = from + text.text.len();
        self.region.out = to;
        if self.region.carried.is_empty() {
            out.push(self.tokens.relabel(event));
            return;
        }
        let (within, after): (Vec<_>, Vec<_>) = std::mem::take(&mut self.region.carried)
            .into_iter()
            .partition(|(at, _)| *at <= to);
        self.region.carried = after;
        if within.is_empty() {
            out.push(self.tokens.relabel(event));
            return;
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
                text.text = Cow::Owned(spliced);
            }
            out.push(self.tokens.relabel(event));
            return;
        }
        let mut done = 0;
        for (cut, marker) in cuts {
            if cut > done {
                let piece = text_event_like(&event, part(&bytes, done..cut));
                out.push(self.tokens.relabel(piece));
            }
            out.push(Event::Dropped {
                text: self.tokens.text(marker),
                why: DropReason::Wrapper,
            });
            done = cut;
        }
        if done < bytes.len() {
            let piece = text_event_like(&event, part(&bytes, done..bytes.len()));
            out.push(self.tokens.relabel(piece));
        }
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
    /// assembler still holds the call's bytes, which came before the tail: the call ends first, as
    /// the stream's end ends it (`closed`: the model's own stop, or a cut), so every byte leaves
    /// in order.
    fn drop_cut_tag(&mut self, text: String, closed: Closed, out: &mut Events) {
        self.end_stay(out);
        if self.emits() == Emits::Arguments {
            self.close_call(closed, "", out);
        }
        out.push(Event::Dropped {
            text: self.tokens.text(text),
            why: DropReason::Wrapper,
        });
    }

    fn end(&mut self, finish: EngineFinish, out: &mut Events) {
        // The model's own end of output is a call's third ending; any other finish is a cut. A
        // call still open ends so, whether the tail cut a tag short or not.
        let closed = if matches!(finish, EngineFinish::Stop) {
            Closed::Stop
        } else {
            Closed::Cut
        };
        // What the scanner held back as a possible terminal is text now that no more bytes come,
        // except a tag the end cut short.
        let scanned = self.scanner.finish();
        self.run(&scanned, Some(closed), out);
        self.scanner.reclaim(scanned);
        self.end_stay(out);
        match self.emits() {
            Emits::Reasoning => out.push(Event::ReasoningEnd),
            Emits::Arguments => {
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

/// The bytes a piece stands for.
fn piece_text<'p>(format: &'p Format, piece: &'p Piece<'_>) -> &'p str {
    match piece {
        Piece::Text(text) => text,
        Piece::Marker(index) => format.terminal_text(*index),
    }
}

/// The pieces of a scan, owned: what a scan that reads again gives stands in the queue past the
/// bytes it was read from.
fn owned(scanned: &Scanned<'_>) -> Vec<Piece<'static>> {
    scanned.pieces().map(Piece::into_owned).collect()
}

impl Parser for Engine {
    fn feed<'d>(&mut self, input: Input<'d>, out: &mut Events<'d>) -> Result<(), ParseError> {
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
                // A delta that is text whole, with no header pending, is taken as the one piece
                // the scan would make of it, without the scan and the piece.
                if self.pending.is_none() && self.scanner.reads_as_text(text) {
                    if !text.is_empty() {
                        self.place_text(Cow::Borrowed(text), out);
                    }
                    return Ok(());
                }
                let scanned = self.scanner.feed(text);
                // One piece of text (after a held byte, or before one held now) is taken as
                // [`Self::place`] would take it, without the piece, like a delta read as text.
                match scanned.whole_text() {
                    Some(whole) if self.pending.is_none() => self.place_text(whole, out),
                    _ => self.run(&scanned, None, out),
                }
                self.scanner.reclaim(scanned);
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
        formats::{glm, inkling, llama4, mistral, qwen3, xlam},
        input::TokenSpan,
        tagged::{Placement, Spelling},
        testing::{arguments_of, bytes, weather_declared},
    };

    /// GLM's last call, its closing marker not written: the stop token that would end it is
    /// hidden by the stop decoder.
    const LAST_CALL: &str = concat!(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
    );

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    fn run(format: Format, output: &str, finish: EngineFinish) -> Vec<Event<'static>> {
        let mut engine = Engine::with_call_ids(Arc::new(format), weather_declared(), ids());
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
        out.into_owned().drain()
    }

    /// The events of `output` after `prompt`, ended by a stop.
    fn run_after(format: Format, prompt: &str, output: &str) -> Vec<Event<'static>> {
        let mut engine = Engine::with_call_ids(Arc::new(format), weather_declared(), ids());
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
        out.into_owned().drain()
    }

    /// The events of `output` after `prompt` on `format`, with `declared` the request's tools,
    /// ended by a stop.
    fn read_with(
        format: &Arc<Format>,
        declared: Declared,
        prompt: &str,
        pieces: &[&str],
    ) -> Vec<Event<'static>> {
        let mut engine = Engine::with_call_ids(Arc::clone(format), declared, ids());
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
                .expect("a delta");
        }
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("the end");
        out.into_owned().drain()
    }

    /// Where the `UnknownTool` events stand among `events`.
    fn unknown_tool_at(events: &[Event]) -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter_map(|(at, event)| {
                matches!(
                    event,
                    Event::Malformed {
                        why: MalformedReason::UnknownTool,
                        ..
                    }
                )
                .then_some(at)
            })
            .collect()
    }

    #[test]
    fn a_call_to_a_function_the_request_did_not_declare_is_announced_before_its_start_on_every_syntax(
    ) {
        // Four syntaxes: keyed tags, JSON between tags, the turn's Python list and the JSON list.
        // A declared name: no event. An undeclared or a misspelled one: one `UnknownTool` event of
        // no bytes right before the call's start, the call itself as the syntax read it, at every
        // cut of the output. A request without tools is not asked.
        type Written = fn(&str) -> String;
        let cases: [(&str, Format, &str, Written); 4] = [
            ("glm", glm(), "<|assistant|><think>", |name| {
                format!(
                    "</think>Sure.<tool_call>{name}<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
                )
            }),
            (
                "qwen3",
                qwen3(CallSyntax::Json),
                "<|im_start|>assistant\n",
                |name| {
                    format!(
                        "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {{\"city\": \"Paris\"}}}}\n</tool_call>"
                    )
                },
            ),
            (
                "llama4",
                llama4(),
                "<|header_start|>assistant<|header_end|>\n\n",
                |name| format!("[{name}(city=\"Paris\")]"),
            ),
            ("mistral", mistral(), "[/INST]", |name| {
                format!(
                    "[TOOL_CALLS] [{{\"name\": \"{name}\", \"arguments\": {{\"city\": \"Paris\"}}, \"id\": \"call00000\"}}]"
                )
            }),
        ];
        for (table, format, prompt, written) in cases {
            let format = Arc::new(format);
            let declared_arguments = {
                let events = read_with(
                    &format,
                    weather_declared(),
                    prompt,
                    &[&written("get_weather")],
                );
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, Event::Malformed { .. })),
                    "{table}: a declared name: {events:?}"
                );
                arguments_of(&events, 0)
            };
            for name in ["fetch_weather", "get_wether"] {
                let output = written(name);
                let characters: Vec<&str> = output
                    .char_indices()
                    .map(|(at, character)| &output[at..at + character.len_utf8()])
                    .collect();
                let mut plans: Vec<Vec<&str>> = vec![vec![&output], characters];
                plans.extend(
                    (1..output.len())
                        .filter(|cut| output.is_char_boundary(*cut))
                        .map(|cut| vec![&output[..cut], &output[cut..]]),
                );
                for pieces in plans {
                    let events = read_with(&format, weather_declared(), prompt, &pieces);
                    let place = || format!("{table} {name} in {} pieces: {events:?}", pieces.len());
                    assert_eq!(bytes(&events), output, "{}", place());
                    let unknown = unknown_tool_at(&events);
                    let start = events
                        .iter()
                        .position(|event| {
                            matches!(event, Event::ToolCallStart { name: called, .. } if called == name)
                        })
                        .unwrap_or_else(|| panic!("{}", place()));
                    assert_eq!(unknown, [start - 1], "{}", place());
                    assert!(
                        matches!(&events[start - 1], Event::Malformed { text, .. } if text.text.is_empty()),
                        "{}",
                        place()
                    );
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| matches!(event, Event::Malformed { .. }))
                            .count(),
                        1,
                        "{}",
                        place()
                    );
                    assert_eq!(arguments_of(&events, 0), declared_arguments, "{}", place());
                    assert!(
                        matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                        "{}",
                        place()
                    );
                }
            }
            let events = read_with(
                &format,
                Declared::of(&[]),
                prompt,
                &[&written("fetch_weather")],
            );
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{table}: no tools declared, nothing to announce: {events:?}"
            );
            assert!(matches!(
                events.last(),
                Some(Event::Finish { tool_calls: 1, .. })
            ));
        }
    }

    #[test]
    fn a_request_with_many_tools_has_them_indexed_once_and_every_name_found_by_the_index() {
        // More tools than a call's start searches directly, declared in reverse order of their
        // names: the index built at the first call finds each declared name, and a name between
        // two declared ones, or past the last, is announced.
        let count = Engine::NAMES_SEARCHED_DIRECTLY + 4;
        let tools: Vec<Tool> = (0..count)
            .rev()
            .map(|k| Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: format!("tool_{k:02}"),
                    description: None,
                    parameters: value!({"type": "object", "properties": {"city": {"type": "string"}}}),
                    strict: None,
                    extra: Default::default(),
                },
            })
            .collect();
        let format = Arc::new(glm());
        let prompt = "<|assistant|><think>";
        let written = |name: &str| {
            format!(
                "</think>Sure.<tool_call>{name}<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
            )
        };
        for k in 0..count {
            let name = format!("tool_{k:02}");
            let events = read_with(&format, Declared::of(&tools), prompt, &[&written(&name)]);
            assert!(unknown_tool_at(&events).is_empty(), "{name}: {events:?}");
            assert!(matches!(
                events.last(),
                Some(Event::Finish { tool_calls: 1, .. })
            ));
        }
        for name in ["tool_1", "tool_015", "tool_99", "tool", "tool_000"] {
            let events = read_with(&format, Declared::of(&tools), prompt, &[&written(name)]);
            let start = events
                .iter()
                .position(|event| matches!(event, Event::ToolCallStart { .. }))
                .unwrap_or_else(|| panic!("{name:?}: the call starts: {events:?}"));
            assert_eq!(
                unknown_tool_at(&events),
                [start - 1],
                "{name:?}: {events:?}"
            );
        }
        // Two calls in one output share the index: built once, both names found.
        let output = format!(
            "{}<tool_call>tool_19<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>",
            written("tool_03")
        );
        let events = read_with(&format, Declared::of(&tools), prompt, &[&output]);
        assert!(unknown_tool_at(&events).is_empty(), "{events:?}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
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
                        *end = Some(source.text.to_string());
                    }
                }
                _ => {}
            }
        }
        calls
    }

    /// The malformed events as (text, reason), in order.
    fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
                _ => None,
            })
            .collect()
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
    fn without_the_row_a_stop_ends_the_call_as_the_marker_would_and_says_so() {
        // The model's own end of output inside a call under a table with no stop row: the
        // template writes the closing marker the model skipped, so the call ends as that marker
        // would end it, the object closed, and one `Malformed { OpenAtTheStop }` event of no
        // bytes stands before the closing brace. A cut, by the token limit or an abort, leaves
        // the call open as before and says `UnterminatedRegion`.
        let events = run(glm(), LAST_CALL, EngineFinish::Stop);
        assert_eq!(
            calls(&events),
            [(
                r#"{"city": "Paris", "days": 3}"#.to_string(),
                Some(String::new())
            )]
        );
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::OpenAtTheStop)]
        );
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))
            .expect("the one word");
        assert!(
            matches!(&events[at + 1], Event::ToolCallArguments { json, .. } if json == "}"),
            "{events:?}"
        );
        assert_eq!(bytes(&events), LAST_CALL);
        for finish in [EngineFinish::Length, EngineFinish::Abort] {
            let events = run(glm(), LAST_CALL, finish);
            assert_eq!(
                calls(&events),
                [(
                    r#"{"city": "Paris", "days": 3"#.to_string(),
                    Some(String::new())
                )]
            );
            assert_eq!(
                malformed(&events),
                [(String::new(), MalformedReason::UnterminatedRegion)]
            );
        }
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
        // Inside an integer, between two parameters and inside a key: the call ends as
        // `</tool_call>` there would end it, with no bytes for the end.
        for output in [
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key><arg_value>2",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            "<tool_call>get_weather<arg_key>ci",
        ] {
            let stopped = run(glm().closes_at_stop("call"), output, EngineFinish::Stop);
            let marked = run(glm(), &format!("{output}</tool_call>"), EngineFinish::Stop);
            let without_end_bytes = |events: &[Event<'static>]| -> Vec<Event<'static>> {
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
        // Inside a string the stop closes the string too: the model wrote nothing after the
        // value's bytes, so they are the value whole, and the one word of no bytes before the
        // closing quote says the stop closed it, under the row as without it; the marker closes
        // it the same way, under its own word.
        let output = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par";
        for format in [glm().closes_at_stop("call"), glm()] {
            let events = run(format, output, EngineFinish::Stop);
            assert_eq!(
                calls(&events),
                [(r#"{"city": "Par"}"#.to_string(), Some(String::new()))]
            );
            assert_eq!(
                malformed(&events),
                [(String::new(), MalformedReason::OpenAtTheStop)]
            );
            assert_eq!(bytes(&events), output);
        }
        let marked = run(glm(), &format!("{output}</tool_call>"), EngineFinish::Stop);
        assert_eq!(
            calls(&marked),
            [(
                r#"{"city": "Par"}"#.to_string(),
                Some("</tool_call>".to_string())
            )]
        );
        assert_eq!(
            malformed(&marked),
            [(String::new(), MalformedReason::OpenAtTheMarker)]
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

    /// Prose where the template only wraps, between a block's two calls: the words keep the
    /// model's whitespace between them, one `Malformed { TextBetweenCalls }` event per word with
    /// the whitespace before it; the whitespace right after the first call's close and the run
    /// before the next call's opener are the template's, dropped; the same whole, at every cut
    /// and one character at a time.
    #[test]
    fn prose_between_two_calls_keeps_the_models_whitespace_between_its_words() {
        let format = Format::new("block")
            .terminal("calls_open", "<calls>")
            .terminal("calls_close", "</calls>")
            .terminal("call_open", "<call>")
            .terminal("call_close", "</call>")
            .state("content", Emits::Content)
            .state("calls", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .transition("content", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "content")
            .calls(CallSyntax::Json);
        let call = |city: &str| {
            format!(
                "<call>{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"{city}\"}}}}</call>"
            )
        };
        let output = format!(
            "<calls>\n{}\n and also this one \n{}\n</calls>",
            call("Paris"),
            call("Rome")
        );
        let [content, _, dropped, malformed] = same_at_every_cut(&format, "", &output);
        assert_eq!(
            [content.as_str(), malformed.as_str()],
            ["", "and also this one"]
        );
        assert_eq!(
            dropped,
            "<calls>\n<call></call>\n  \n<call></call>\n</calls>"
        );
        let events = run(format, &output, EngineFinish::Stop);
        assert_eq!(
            wrapping_words(&events, MalformedReason::TextBetweenCalls),
            ["and", " also", " this", " one"]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                .count(),
            2
        );
    }

    /// Text after a call's object, before the call's closing marker: the words keep the model's
    /// whitespace between them, one `Malformed { TextAfterTheArguments }` event per word with the
    /// whitespace before it; the whitespace right after the object and the run before the marker
    /// are the template's; a terminal the table ignores inside the held run keeps its place; the
    /// same whole, at every cut and one character at a time. A stream that ends there drops the
    /// held run as the template's and loses no word.
    #[test]
    fn text_after_a_calls_object_keeps_the_models_whitespace_between_its_words() {
        let format = qwen3(CallSyntax::Json);
        let object = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        let output = format!("<tool_call>\n{object}\nplease note this and that\n</tool_call>");
        let [content, _, dropped, malformed] = same_at_every_cut(&format, "", &output);
        assert_eq!(
            [content.as_str(), malformed.as_str()],
            ["", "please note this and that"]
        );
        assert_eq!(dropped, "<tool_call>\n\n</tool_call>");
        let events = run(format.clone(), &output, EngineFinish::Stop);
        assert_eq!(
            wrapping_words(&events, MalformedReason::TextAfterTheArguments),
            ["please", " note", " this", " and", " that"]
        );
        // A terminal the table ignores inside the whitespace the stay holds keeps its place, the
        // model's whitespace around it in events of its own, and the run before the marker is the
        // template's with the terminal in it.
        let separated = format.clone().ignores("sep", "|");
        let output = format!("<tool_call>\n{object}\nplease | note |\n</tool_call>");
        let [_, _, dropped, malformed] = same_at_every_cut(&separated, "", &output);
        assert_eq!(
            [dropped.as_str(), malformed.as_str()],
            ["<tool_call>\n| |\n</tool_call>", "please  note"]
        );
        let events = run(separated, &output, EngineFinish::Stop);
        let end = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { .. }))
            .expect("the call's end");
        let after_the_call: Vec<(bool, &str)> = events[end + 1..]
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some((false, &*text.text)),
                Event::Malformed { text, .. } => Some((true, &*text.text)),
                _ => None,
            })
            .collect();
        assert_eq!(
            after_the_call,
            [
                (false, "\n"),
                (true, "please"),
                (true, " "),
                (false, "|"),
                (true, " note"),
                (false, " "),
                (false, "|"),
                (false, "\n"),
                (false, "</tool_call>"),
            ]
        );
        // The stream ends after the prose: the held run is the template's, dropped.
        let events = run(
            format,
            &format!("<tool_call>\n{object}\nplease note "),
            EngineFinish::Length,
        );
        assert_eq!(
            wrapping_words(&events, MalformedReason::TextAfterTheArguments),
            ["please", " note"]
        );
        let [_, _, dropped, _] = kinds(&events);
        assert_eq!(dropped, "<tool_call>\n ");
    }

    /// The `Malformed` events of `why`, each one's text: a word with the whitespace before it.
    fn wrapping_words(events: &[Event], why: MalformedReason) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why: reason } if *reason == why => {
                    Some(text.text.to_string())
                }
                _ => None,
            })
            .collect()
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
    fn a_terminal_with_no_row_is_the_states_text_and_content_says_so() {
        // In the answer, a marker the table has no row for stays as the model wrote it, and a
        // `Malformed` event of no bytes before it says a frame was not read; in the thought it is
        // the thought's text alone, as the opener read there is; where the template discards the
        // text it is dropped as the text is.
        let events = run_after(
            ernie(),
            "Assistant:",
            "<think>Plan <think>more</think>\n<response>\nHi<response>\nthere</response></response>",
        );
        let [content, reasoning, dropped, malformed] = kinds(&events);
        assert_eq!(content, "Hi<response>\nthere");
        assert_eq!(reasoning, "<think>Plan <think>more");
        assert_eq!(dropped, "</think>\n<response>\n</response></response>");
        assert_eq!(malformed, "");
        let said: Vec<usize> = events
            .iter()
            .enumerate()
            .filter_map(|(at, event)| match event {
                Event::Malformed { text, why } => {
                    assert_eq!(*why, MalformedReason::MarkerOutOfPlace, "{event:?}");
                    assert!(text.text.is_empty(), "{event:?}");
                    Some(at)
                }
                _ => None,
            })
            .collect();
        let [at] = said[..] else {
            panic!("one event of no bytes, not {said:?}: {events:?}");
        };
        assert!(
            matches!(&events[at + 1], Event::Content(text) if text.text.starts_with("<response>\n")),
            "{events:?}"
        );
        // A request that does not separate reasoning reads the thought as content, and counts
        // the same: the marker written again inside the answer, not the ones in the thought.
        let variant = run_after(
            ernie().without_reasoning(),
            "Assistant:",
            "<think>Plan <think>more</think>\n<response>\nHi<response>\nthere</response></response>",
        );
        assert_eq!(
            variant
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count(),
            1,
            "{variant:?}"
        );
        let variant = run_after(
            ernie().without_reasoning(),
            "Assistant:",
            "<think>\nPlan.\n</think>\n<response>\nHi\n</response>",
        );
        assert!(
            !variant
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{variant:?}"
        );
    }

    #[test]
    fn the_run_a_trimmed_stay_holds_comes_out_before_the_event_for_a_marker_with_no_row() {
        // The thought's close with no thought open, in a trimmed content stay: the tail the stay
        // held is the text's, as the marker's text would make it, released before the event of
        // no bytes, so the event stands right before the marker.
        let events = run_after(gemma_trimmed(), "<|turn>model\n", "Hi \n<channel|>there");
        let [content, reasoning, dropped, malformed] = kinds(&events);
        assert_eq!(content, "Hi \n<channel|>there");
        assert_eq!([reasoning, dropped, malformed], ["", "", ""]);
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))
            .expect("the event of no bytes");
        assert!(
            matches!(&events[at - 1], Event::Content(text) if text.text.ends_with(" \n")),
            "{events:?}"
        );
        assert!(
            matches!(&events[at + 1], Event::Content(text) if text.text.starts_with("<channel|>")),
            "{events:?}"
        );
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
            let mut engine = Engine::new(ernie(), weather_declared());
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
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// Replays `output` after `prompt` cut at every two-way split and character by character,
    /// and checks that each cut says what the whole does, and that the events account for the
    /// output's bytes.
    fn same_at_every_cut(format: &Format, prompt: &str, output: &str) -> [String; 4] {
        let replay = |cuts: &[usize]| {
            let mut engine = Engine::new(format.clone(), weather_declared());
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
            let events = out.into_owned().drain();
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
    fn a_kept_content_state_shows_whitespace_alone_as_written() {
        // A response template's `strip: false` field keeps its value as written: the whitespace a
        // stay holds alone is content, where the held lead drops it as the template's spacing.
        let output = "<think>\nplan\n</think>\n\n";
        let kept = qwen3(CallSyntax::Json).keeps("content");
        let [content, reasoning, ..] = same_at_every_cut(&kept, "", output);
        assert_eq!([content.as_str(), reasoning.as_str()], ["\n\n", "\nplan\n"]);
        assert_eq!(trimmed_space(&run(kept, output, EngineFinish::Stop)), "");
        let [content, ..] = same_at_every_cut(&qwen3(CallSyntax::Json), "", output);
        assert_eq!(content, "");
    }

    #[test]
    fn a_held_lead_leaves_no_content_delta_before_a_call_and_leaves_with_the_text_after_it() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>";
        // The content of each feed's events, the end's included, empty ones left out: the
        // content deltas a client of the stream receives.
        let deltas = |pieces: &[&str]| -> Vec<String> {
            let mut engine = Engine::new(qwen3(CallSyntax::Json), weather_declared());
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
        let mut engine = Engine::new(format(), weather_declared());
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
                Event::Dropped { text, .. } => Some((text.text.to_string(), text.tokens)),
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
            let mut engine = Engine::new(inkling(), weather_declared());
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
    fn the_whitespace_after_a_regions_last_call_keeps_its_place_among_ignored_terminals_and_counts_once(
    ) {
        // The tagged assembler hands the whitespace after its last call back to the engine, which
        // drops it as the region's wrapping: an ignored terminal that arrived inside that
        // whitespace leaves between its bytes, in an event of its own, and every token is counted
        // once, whole and at every cut.
        let output = "<tool_call>\n<function=f>\n</function>\n|\n</tool_call>after";
        let format =
            || qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)).ignores("sep", "|");
        let char_spans = |text: &str| -> Vec<TokenSpan> {
            text.char_indices()
                .map(|(at, c)| TokenSpan {
                    token_id: 0,
                    start: at,
                    end: at + c.len_utf8(),
                    continued: false,
                })
                .collect()
        };
        let read = |pieces: &[&str]| -> Vec<Event<'static>> {
            let spans: Vec<Vec<TokenSpan>> = pieces.iter().map(|piece| char_spans(piece)).collect();
            let mut engine = Engine::new(format(), weather_declared());
            let mut out = Events::new();
            for (piece, spans) in pieces.iter().zip(&spans) {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: piece,
                            spans,
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
            out.into_owned().drain()
        };
        let after_the_call = |events: &[Event]| -> (Vec<(String, Option<u32>)>, String) {
            let end = events
                .iter()
                .position(|event| matches!(event, Event::ToolCallEnd { .. }))
                .expect("the call's end");
            let mut dropped = Vec::new();
            let mut content = String::new();
            for event in &events[end + 1..] {
                match event {
                    Event::Dropped { text, .. } => {
                        dropped.push((text.text.to_string(), text.tokens));
                    }
                    Event::Content(text) => content.push_str(&text.text),
                    Event::Finish { .. } => {}
                    other => panic!("after the call: {other:?}"),
                }
            }
            (dropped, content)
        };
        let counted = |events: &[Event]| -> u32 {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Content(text) | Event::Reasoning(text) => text.tokens,
                    Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.tokens,
                    Event::ToolCallStart { source, .. }
                    | Event::ToolCallArguments { source, .. }
                    | Event::ToolCallEnd { source, .. } => source.tokens,
                    Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
                })
                .sum()
        };
        let whole = read(&[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(
            after_the_call(&whole),
            (
                vec![
                    ("\n".to_string(), Some(1)),
                    ("|".to_string(), Some(1)),
                    ("\n".to_string(), Some(1)),
                    ("</tool_call>".to_string(), Some(12)),
                ],
                "after".to_string()
            )
        );
        assert_eq!(counted(&whole), output.chars().count() as u32, "{whole:?}");
        for cut in 1..output.len() {
            let events = read(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            let (dropped, content) = after_the_call(&events);
            let texts: Vec<&str> = dropped.iter().map(|(text, _)| text.as_str()).collect();
            assert_eq!(texts, ["\n", "|", "\n", "</tool_call>"], "cut at {cut}");
            assert_eq!(content, "after", "cut at {cut}");
            assert_eq!(
                counted(&events),
                output.chars().count() as u32,
                "cut at {cut}: {events:?}"
            );
        }
    }

    #[test]
    fn a_stop_closes_a_call_of_another_syntax_the_same_way_with_or_without_the_row() {
        let tagged = || qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine));
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n";
        let closed = run(tagged().closes_at_stop("calls"), output, EngineFinish::Stop);
        assert_eq!(
            calls(&closed),
            [(r#"{"city": "Paris"}"#.to_string(), Some(String::new()))]
        );
        assert!(malformed(&closed).is_empty());
        assert_eq!(bytes(&closed), output);
        // Without the row the stop ends the call as the marker would too, and says so.
        let stopped = run(tagged(), output, EngineFinish::Stop);
        assert_eq!(
            calls(&stopped),
            [(r#"{"city": "Paris"}"#.to_string(), Some(String::new()))]
        );
        assert_eq!(
            malformed(&stopped),
            [(String::new(), MalformedReason::OpenAtTheStop)]
        );
        assert_eq!(bytes(&stopped), output);
        // A cut leaves it open, with the row or without.
        for format in [tagged().closes_at_stop("calls"), tagged()] {
            let cut = run(format, output, EngineFinish::Length);
            assert_eq!(
                calls(&cut),
                [(r#"{"city": "Paris""#.to_string(), Some(String::new()))]
            );
        }
    }

    /// Two calls of Qwen3's syntax in one output.
    const TWO_CALLS: &str = concat!(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n",
        "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>"
    );

    fn ids_of(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn two_turns_are_two_engines_whose_call_ids_never_meet() {
        // The gateway makes one engine per request, so a conversation's turns never share a
        // nonce: four calls over two turns are four ids, each of the shape the older parsers mint.
        let turn = || {
            let mut engine = Engine::new(qwen3(CallSyntax::Json), weather_declared());
            let mut out = Events::new();
            engine
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: TWO_CALLS,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("a delta");
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("the end");
            ids_of(&out.drain())
        };
        let first = turn();
        let second = turn();
        assert_eq!(first.len(), 2, "{first:?}");
        assert_eq!(second.len(), 2, "{second:?}");
        let mut all: Vec<&String> = first.iter().chain(&second).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 4, "{first:?} {second:?}");
        for id in all {
            assert!(crate::ids::is_minted(id), "{id}");
        }
    }

    #[test]
    fn one_nonce_gives_an_output_the_same_ids_whole_or_cut_character_by_character() {
        let replay = |pieces: Vec<&str>| {
            let mut engine =
                Engine::with_call_ids(Arc::new(qwen3(CallSyntax::Json)), weather_declared(), ids());
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
                    .expect("a delta");
            }
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("the end");
            ids_of(&out.drain())
        };
        let whole = replay(vec![TWO_CALLS]);
        assert_eq!(
            whole,
            [
                "call_0123456789abcdef00000000",
                "call_0123456789abcdef00000001"
            ]
        );
        let characters: Vec<&str> = TWO_CALLS
            .char_indices()
            .map(|(at, c)| &TWO_CALLS[at..at + c.len_utf8()])
            .collect();
        assert_eq!(replay(characters), whole);
    }
}
