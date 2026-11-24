//! A JSON list of calls with no markers around it: the whole output is either
//! `[{"name": …, "arguments": {…}}, …]` or content. xLAM writes this; so does any model whose
//! template puts the calls alone in the turn.
//!
//! The format's one state is an arguments state entered at the output's start, so this assembler
//! sees every byte of the output and decides at the first one that is not whitespace:
//!
//! - `[` opens the list. Each object in it goes to a [`json::Assembler`](super::Assembler) of its
//!   own, with the next free index, which is spent only when the object starts a call, so an
//!   object with no name leaves no gap; every call streams as a Qwen3 call does. The `[`, the
//!   commas between objects, the whitespace around them, the `]` and whitespace that ends the
//!   output after it are the template's and come back as `Dropped { Wrapper }`; any other text
//!   between objects is `Malformed`, one event per word, the whitespace between two words the
//!   model's and in the next word's event, and so is text after the `]`, with the whitespace
//!   before and between its words: once text follows the list, every byte after the `]` is the
//!   model's prose, which the template's reference keeps as written.
//! - Anything else makes the whole output content: that byte and every later one come back as
//!   `Content`, since the model answered in prose. The whitespace held before it comes back as
//!   content too. A `[` that is followed by anything but an object or the `]` is prose as well
//!   (`[Note: draft]`, a markdown link), and the `[` comes back as content with the rest.
//!
//! With [`Framing::ObjectOrList`] (the `json` table), a `{` at the first byte is one call
//! alone: the object goes to a [`json::Assembler`](super::Assembler) as a list's object does, and
//! what follows it reads as what follows a list's `]`: whitespace that ends the output is the
//! template's wrapping, dropped; text, with the whitespace before and between its words, is
//! `Malformed`; but another object is another call, after a `;` (Llama 3.1 writes its parallel
//! calls so, `{…}; {…}`) or after whitespace alone (Llama 3.2 Instruct writes them as consecutive
//! objects with a blank line between, `{…}\n\n{…}`, as the engines' own readers of the family
//! read every object of the sequence), the `;` and the whitespace the template's wrapping; a `;`
//! that no object follows is text after the arguments. Where the table names the family's call
//! marker (Llama 3.1's `<|python_tag|>`), the output may open with it, after whitespace or not:
//! the marker is dropped as wrapping, with the whitespace before it, and what follows decides as
//! the first byte did, so that an output with the marker and one without read the same; a delta
//! that ends inside the marker is held until the next decides, and bytes that turn out not to be
//! it are prose with everything held. The marker is read at the output's start only: later, or
//! after prose, it is text. A list reads as before. [`Framing::Bare`] (the `xlam` table) keeps a
//! `{` as prose.
//!
//! A region with no markers closes by one only before its first byte that is not whitespace,
//! where the variant a forced call makes of a table has the thought's opener
//! ([`Format::with_forced_call`](crate::Format::with_forced_call)): [`Assembler::close`] drops the
//! whitespace held there, and is otherwise for a table that puts the list between markers after
//! all. In that variant the assembler is entered after the thought too, and a marker the table
//! ignores is a call's text inside the call's object. [`Assembler::finish`] is the end of the
//! stream: an object still open is finished as the JSON assembler finishes one (a started call
//! closed with what arrived, an object that never named a call `Malformed`), and a list left
//! open reports what it held.
//!
//! Between markers (Llama Nemotron's `<TOOLCALL>[…]</TOOLCALL>`) a row of the table opens the
//! region and its closing marker ends it, so prose has no place inside: [`Framing::Framed`]
//! reports text where the list should be, a `[` that opens no list of calls included, as
//! `Malformed` where the bare list takes it for content. The engine picks it for a table with a
//! row into the arguments state. [`Framing::FramedObjectOrList`] is that region for a table with
//! no call marker whose model also writes one call as an object alone between the markers
//! (Command): a `{` at the first byte is one call and what follows it reads as under
//! [`Framing::ObjectOrList`], and text in place of the calls is malformed all the same.
//!
//! Under a JSON schema (a required call's list, [`Framing::Forced`]) the output reads as under
//! [`Framing::ObjectOrList`] with no marker and differs at the end of the stream alone: the
//! engine held the output to the list, so no byte of an open one is prose. The separators held
//! after a call, an object cut before its arguments began and what a started call's object held
//! behind its arguments are refused ([`crate::tagged::refuse`]), where the other framings return
//! them as `Malformed`.
//!
//! The five are the one [`Framing`] the assembler takes at construction, the family's call marker
//! inside the one that reads it, and the engine's factory maps the table's call syntax to it; the
//! object reader and the stages around it are the same under every framing. A call written as one
//! object between a table's markers (`CallSyntax::Json`) is the object reader itself
//! ([`json::Assembler`](super::Assembler)), not a framing of this one: the engine reports what
//! follows the object there, where this reader holds the whitespace after a lone object and reads
//! a `;`, another object or prose after it (`docs/assemblers.md`).

use crate::{
    event::{release_between, Event, Events, MalformedReason, Text},
    ids::CallIds,
    json,
};

/// The events of a JSON list of calls, each call with its own index from `first` and its id from
/// the engine's [`CallIds`].
#[derive(Debug)]
pub struct Assembler {
    next_index: u32,
    /// Where the calls' ids count from.
    ids: CallIds,
    stage: Stage,
    /// Bytes no event has accounted for yet; the next event takes them as its source.
    carried: String,
    /// Objects that began, so that prose after the `[` is told from text between objects.
    objects: u32,
    /// Whether a `{` at the first byte is one call alone ([`Framing::ObjectOrList`],
    /// [`Framing::FramedObjectOrList`], [`Framing::Forced`]).
    lone_object: bool,
    /// Whether a `[` opened a list, so that a finished object returns to the list or ends the
    /// output's calls.
    listed: bool,
    /// Whether markers frame the calls ([`Framing::Framed`], and with `lone_object`
    /// [`Framing::FramedObjectOrList`]): text in place of them is malformed, not prose.
    framed: bool,
    /// Whether a schema holds the output to the list ([`Framing::Forced`]): what the end of the
    /// stream leaves of an open list is refused, never text.
    forced: bool,
    /// Whether text followed the lone object or the list's `]`: from there on every byte,
    /// whitespace included, is the model's prose after its calls. Before it, whitespace is held,
    /// the template's wrapping if nothing follows.
    prose_after: bool,
    /// Whether text between the list's objects was released since the `[` or the last object:
    /// the list's wrapping carried after it is the model's, between two pieces of its text, and
    /// leaves with the next piece ([`release_between`]); before any, it is the template's.
    released: bool,
    /// The family's call marker the output may open with ([`Framing::ObjectOrList`]), until the
    /// first byte that is not whitespace has decided whether it does.
    opener: Option<&'static str>,
    /// Whether a `;` followed the lone object: it is held with the whitespace around it until the
    /// next byte decides whether an object follows it, and a `;` nothing follows is text.
    separated: bool,
    /// Which runs of the last piece's bytes were the list's own value, in order: its objects'
    /// bytes, its close from inside it, or the prose of an output that is no list. The JSON-answer
    /// option's quote state steps over these alone
    /// ([`CallAssembler::quote_value`](crate::engine::CallAssembler::quote_value)): text between
    /// calls, a malformed list's bytes and the text after the `]` are no JSON value. The capacity
    /// stays from piece to piece; recorded under `tracked` alone.
    value: Vec<std::ops::Range<usize>>,
    /// Whether `value` is recorded: the JSON-answer option asked for it
    /// ([`CallAssembler::track_value`](crate::engine::CallAssembler::track_value)); off, a feed
    /// records nothing and `value` stays empty, so a request without the option pays nothing
    /// per piece for the runs.
    tracked: bool,
}

/// What frames the calls: how the first byte that is not whitespace reads, and what text in place
/// of a list is. The engine's factory maps the table's call syntax to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The list alone in the output (xLAM): a `[` opens it, and anything else is prose.
    Bare,
    /// The list between the table's markers (Llama Nemotron): a `[` opens it, and anything else
    /// is malformed, since prose has no place inside the markers.
    Framed,
    /// One object, several with `;` or whitespace between them, or the list between the table's
    /// markers (Command): a `{` is one call alone, a `[` opens the list, and anything else is
    /// malformed, as under [`Self::Framed`].
    FramedObjectOrList,
    /// One object, several with `;` or whitespace between them, or a list alone in the output
    /// (the `json` table): a `{` is one call alone, a `[` opens the list, and anything else is
    /// prose.
    /// `opener` is the family's call marker the output may open with (Llama 3.1's
    /// `<|python_tag|>`), dropped as the template's wrapping where it does.
    ObjectOrList { opener: Option<&'static str> },
    /// The JSON a schema holds the output to (a required call's list), read as
    /// [`Self::ObjectOrList`] reads it with no marker. The engine lets the model write nothing
    /// else there, so no byte of an open list is prose: what the end of the stream leaves of it
    /// (the separators after a call, an object cut before its arguments began, a started call's
    /// object behind its arguments) is refused: counted and never content.
    Forced,
}

impl Framing {
    /// Whether a `{` at the first byte is one call alone, and another object after it, behind a
    /// `;` or whitespace, the next.
    fn lone_object(self) -> bool {
        matches!(
            self,
            Self::ObjectOrList { .. } | Self::FramedObjectOrList | Self::Forced
        )
    }

    /// The family's call marker the output may open with, where the framing names one.
    fn opener(self) -> Option<&'static str> {
        match self {
            Self::ObjectOrList { opener } => opener,
            Self::Bare | Self::Framed | Self::FramedObjectOrList | Self::Forced => None,
        }
    }
}

#[derive(Debug)]
enum Stage {
    /// Before the first byte that is not whitespace.
    Deciding,
    /// Inside the family's call marker at the output's start: `matched` bytes of it have come,
    /// held, and the next byte says whether the rest follows.
    Opening { matched: usize },
    /// The output is prose.
    Content,
    /// Text where the list should be, between markers: every byte malformed.
    Malformed,
    /// Inside the list, between objects.
    Between,
    /// Inside an object.
    /// Boxed: the assembler holds the object's scanner and its prefix, far more than the other
    /// stages carry.
    Object(Box<json::Assembler>),
    /// After the `]`.
    After,
}

impl Assembler {
    /// An assembler positioned at the output's start, or after the marker that opens the list's
    /// region, the first call taking index `first` and the ids counting from `ids`; `framing`
    /// says what frames the calls.
    pub fn new(first: u32, ids: CallIds, framing: Framing) -> Self {
        Self {
            next_index: first,
            ids,
            stage: Stage::Deciding,
            carried: String::new(),
            objects: 0,
            lone_object: framing.lone_object(),
            listed: false,
            framed: matches!(framing, Framing::Framed | Framing::FramedObjectOrList),
            forced: framing == Framing::Forced,
            prose_after: false,
            released: false,
            opener: framing.opener(),
            separated: false,
            value: Vec::new(),
            tracked: false,
        }
    }

    /// Append the next bytes and push the events they complete; every byte is taken. An
    /// object's argument fragment that lies within `bytes` borrows them (`'d`, the delta's life).
    pub fn feed<'d>(&mut self, bytes: &'d str, out: &mut Events<'d>) {
        let mut rest = bytes;
        if !self.tracked {
            while !rest.is_empty() {
                rest = self.take(rest, out);
            }
            return;
        }
        self.value.clear();
        while !rest.is_empty() {
            let at = bytes.len() - rest.len();
            let from_object = matches!(self.stage, Stage::Object(_));
            let from_inside = from_object || matches!(self.stage, Stage::Between);
            rest = self.take(rest, out);
            // A take's bytes are the value's by the stage it ends in: an object took them, they are
            // the prose of an output that is no list, or they closed the list from inside it. The
            // `[`, the whitespace and commas between objects, text between calls, a malformed list's
            // bytes and the text after the `]` are not, whichever piece they arrive in.
            let counted = match self.stage {
                Stage::Object(_) | Stage::Content => true,
                Stage::After => from_inside,
                Stage::Between => from_object,
                Stage::Deciding | Stage::Opening { .. } | Stage::Malformed => false,
            };
            if counted {
                let end = bytes.len() - rest.len();
                match self.value.last_mut() {
                    Some(run) if run.end == at => run.end = end,
                    _ => self.value.push(at..end),
                }
            }
        }
    }

    /// The region closed by a marker: a table that puts the list between markers. With no
    /// markers around the calls a marker closes the region only before its first byte that is
    /// not whitespace (the engine asks the assembler whether it closes), and the whitespace held
    /// there is the template's wrapping, not prose.
    pub fn close(mut self, out: &mut Events) {
        if !self.framed && matches!(self.stage, Stage::Deciding) {
            self.drop_carried(out);
        }
        self.finish(out);
    }

    /// The end of the stream: an open object is finished as the JSON assembler finishes one, and
    /// what the list held comes back as `Malformed { UnterminatedRegion }`. Under a schema
    /// ([`Framing::Forced`]) an open list is no prose: an object cut before its arguments began,
    /// what a started call's object held behind its arguments, and the separators held after a
    /// call, are refused ([`crate::tagged::refuse`]), as a failed call's bytes are.
    pub fn finish(mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::After) {
            Stage::Object(object) => {
                self.drop_carried(out);
                if self.forced {
                    object.finish_forced(out);
                } else {
                    object.finish(out);
                }
            }
            // The bytes of a marker the output ends inside are text, as a cut marker is anywhere.
            Stage::Deciding | Stage::Opening { .. } | Stage::Content => {
                if !self.carried.is_empty() {
                    out.push(Event::Content(Text::uncounted(std::mem::take(
                        &mut self.carried,
                    ))));
                }
            }
            // Whitespace held after the object or the list, with nothing after it: the template's
            // wrapping. A `;` held there separates nothing: text after the arguments.
            Stage::After if self.separated => {
                out.push(Event::Malformed {
                    text: Text::uncounted(std::mem::take(&mut self.carried)),
                    why: MalformedReason::TextAfterTheArguments,
                });
            }
            Stage::After => self.drop_carried(out),
            // Every byte was reported as it arrived.
            Stage::Malformed => {}
            // The list's own separators after a call the schema held it to: no text of the
            // model's. The bracket alone, with no object yet, stays what it is below.
            Stage::Between if self.forced && self.objects > 0 => {
                crate::tagged::refuse(
                    std::mem::take(&mut self.carried),
                    MalformedReason::UnterminatedRegion,
                    out,
                );
            }
            Stage::Between => {
                if !self.carried.is_empty() {
                    out.push(Event::Malformed {
                        text: Text::uncounted(std::mem::take(&mut self.carried)),
                        why: MalformedReason::UnterminatedRegion,
                    });
                }
            }
        }
    }

    /// Reads as much of `text` as the stage takes and returns the rest.
    fn take<'d>(&mut self, text: &'d str, out: &mut Events<'d>) -> &'d str {
        match &mut self.stage {
            Stage::Deciding => {
                let Some((at, first)) = text.char_indices().find(|(_, c)| !c.is_whitespace())
                else {
                    self.carried.push_str(text);
                    return "";
                };
                self.carried.push_str(&text[..at]);
                // The marker is read once, at the output's start: whatever the first byte is,
                // a later one is text.
                let opener = self.opener.take();
                if first == '[' {
                    // Held until the first object or the `]` says this is a list: prose after a
                    // `[` takes it back as content.
                    self.carried.push('[');
                    self.listed = true;
                    self.stage = Stage::Between;
                    &text[at + 1..]
                } else if first == '{' && self.lone_object {
                    self.open_object(out);
                    &text[at..]
                } else if opener.is_some_and(|opener| opener.starts_with(first)) {
                    self.opener = opener;
                    self.stage = Stage::Opening { matched: 0 };
                    &text[at..]
                } else {
                    self.not_a_list(&text[at..], out);
                    ""
                }
            }
            Stage::Opening { matched } => {
                let matched = *matched;
                let opener = self.opener.unwrap_or_default();
                let pending = &opener[matched..];
                let common = text
                    .bytes()
                    .zip(pending.bytes())
                    .take_while(|(byte, expected)| byte == expected)
                    .count();
                if common == pending.len() {
                    // The marker is whole: it and the whitespace before it are the template's
                    // wrapping, and the first byte after it decides as the output's first did.
                    self.carried.push_str(&text[..common]);
                    self.drop_carried(out);
                    self.opener = None;
                    self.stage = Stage::Deciding;
                    &text[common..]
                } else if common == text.len() {
                    // The delta ends inside the marker: held until the next byte decides.
                    self.carried.push_str(text);
                    self.stage = Stage::Opening {
                        matched: matched + common,
                    };
                    ""
                } else {
                    // Not the marker after all: the bytes held and the rest are prose.
                    self.opener = None;
                    self.not_a_list(text, out);
                    ""
                }
            }
            Stage::Content => {
                out.push(Event::Content(Text::uncounted(text.to_owned())));
                ""
            }
            Stage::Malformed => {
                out.push(Event::Malformed {
                    text: Text::uncounted(text.to_owned()),
                    why: MalformedReason::TextInPlaceOfTheList,
                });
                ""
            }
            Stage::Between => {
                let Some((at, first)) = text
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace() && *c != ',')
                else {
                    self.carried.push_str(text);
                    return "";
                };
                self.carried.push_str(&text[..at]);
                match first {
                    '{' => {
                        self.open_object(out);
                        &text[at..]
                    }
                    ']' => {
                        self.carried.push(']');
                        self.drop_carried(out);
                        self.stage = Stage::After;
                        &text[at + 1..]
                    }
                    _ if self.objects == 0 => {
                        // Not a list of calls after all: the `[` and everything held are prose,
                        // or malformed between markers.
                        self.not_a_list(&text[at..], out);
                        ""
                    }
                    _ => {
                        // A run of text between objects, reported whole, with the wrapping held
                        // before it once text was released since the last object.
                        let end = text[at..]
                            .char_indices()
                            .find(|(_, c)| c.is_whitespace() || matches!(c, ',' | '{' | ']'))
                            .map_or(text.len(), |(i, _)| at + i);
                        release_between(
                            &mut self.carried,
                            &mut self.released,
                            text[at..end].to_owned().into(),
                            MalformedReason::TextBetweenCalls,
                            out,
                        );
                        &text[end..]
                    }
                }
            }
            Stage::Object(object) => {
                let taken = object.feed(text, out);
                if object.done() {
                    // The index is spent only by an object that started a call.
                    if object.started() {
                        self.next_index += 1;
                    }
                    self.stage = if self.listed {
                        Stage::Between
                    } else {
                        Stage::After
                    };
                }
                &text[taken..]
            }
            Stage::After => {
                // Whitespace after the object or the list is held: the template's wrapping if
                // nothing follows it, the model's prose if text does. After a lone object a `;`
                // is held too, and another object, after the `;` or after the whitespace alone,
                // is another call, the `;` and the whitespace the template's wrapping. Once text
                // follows, every byte is prose after the calls, reported malformed with its
                // whitespace, since the template's reference keeps it as written.
                if !self.prose_after {
                    let Some((at, first)) = text.char_indices().find(|(_, c)| !c.is_whitespace())
                    else {
                        self.carried.push_str(text);
                        return "";
                    };
                    if self.lone_object && !self.listed {
                        if first == ';' && !self.separated {
                            self.separated = true;
                            self.carried.push_str(&text[..=at]);
                            return &text[at + 1..];
                        }
                        // Another object, after a `;` or after whitespace alone: another call,
                        // the bytes held the template's wrapping.
                        if first == '{' {
                            self.separated = false;
                            self.carried.push_str(&text[..at]);
                            self.open_object(out);
                            return &text[at..];
                        }
                    }
                    // A `;` held here is part of the prose's run from now on.
                    self.separated = false;
                    self.prose_after = true;
                }
                self.carried.push_str(text);
                out.push(Event::Malformed {
                    text: Text::uncounted(std::mem::take(&mut self.carried)),
                    why: if self.listed {
                        MalformedReason::TextAfterTheList
                    } else {
                        MalformedReason::TextAfterTheArguments
                    },
                });
                ""
            }
        }
    }

    /// An object opens at the next byte: the bytes held before it are the template's wrapping,
    /// and the object goes to an assembler of its own with the next free index.
    fn open_object(&mut self, out: &mut Events) {
        self.drop_carried(out);
        self.released = false;
        let index = self.next_index;
        self.objects += 1;
        self.stage = Stage::Object(Box::new(json::Assembler::new(index, self.ids.id(index))));
    }

    /// The bytes held and `text` are not a list of calls: content under the bare list, malformed
    /// between markers; every later byte comes back the same way.
    fn not_a_list(&mut self, text: &str, out: &mut Events) {
        self.carried.push_str(text);
        let text = Text::uncounted(std::mem::take(&mut self.carried));
        if self.framed {
            self.stage = Stage::Malformed;
            out.push(Event::Malformed {
                text,
                why: MalformedReason::TextInPlaceOfTheList,
            });
        } else {
            self.stage = Stage::Content;
            out.push(Event::Content(text));
        }
    }

    fn drop_carried(&mut self, out: &mut Events) {
        crate::event::drop_carried(std::mem::take(&mut self.carried), out);
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed<'d>(
        &mut self,
        text: &'d str,
        _declared: &crate::Declared,
        out: &mut Events<'d>,
    ) -> usize {
        self.feed(text, out);
        text.len()
    }

    /// Between markers the region's closing marker closes it, wherever it stands. With no
    /// markers around the calls, a terminal with a row out of the state leaves it only before
    /// the first byte that is not whitespace, where a table that reads a thought before the
    /// calls has the thought's opener
    /// ([`Format::with_forced_call`](crate::Format::with_forced_call)); once that byte decided,
    /// every terminal is the calls' text, as a JSON string may hold any, or the prose's.
    fn closes(&mut self, _terminal: &str) -> bool {
        self.framed || matches!(self.stage, Stage::Deciding)
    }

    /// With no markers around the calls, a terminal the table ignores is a call's text inside
    /// the call's object, where a string may hold it; between the objects and around the list
    /// it is the template's wrapping, which the engine drops.
    fn markers_are_text(&self) -> bool {
        !self.framed && matches!(self.stage, Stage::Object(_))
    }

    /// The value runs are recorded from this call on: the quote state will ask for them.
    fn track_value(&mut self) {
        self.tracked = true;
    }

    /// The list's own value: its objects, its close; nothing between calls, nothing where a
    /// malformed list should be, nothing after its `]`.
    fn quote_value(&self, taken: usize, piece: &str, quoted: &mut crate::engine::Quoted) {
        for run in &self.value {
            quoted.advance(&piece[run.start.min(taken)..run.end.min(taken)]);
        }
    }

    fn close(self: Box<Self>, _terminal: &str, out: &mut Events) -> bool {
        (*self).close(out);
        false
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        event::DropReason,
        testing::{bytes, indexed_calls as calls},
    };

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    const LIST: &str = concat!(
        "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, ",
        "{\"name\": \"get_stock_price\", \"arguments\": {}}]"
    );

    #[test]
    fn the_value_runs_are_recorded_under_the_json_answer_option_alone() {
        // A request without the option pays nothing per piece for the runs: an untracked feed
        // records none, whatever the bytes; told, the same feed records the prose as one run.
        let mut out = Events::new();
        let mut untracked = Assembler::new(0, ids(), Framing::Bare);
        untracked.feed("The forecast ", &mut out);
        untracked.feed("stays mild", &mut out);
        assert!(untracked.value.is_empty());
        let mut tracked = Assembler::new(0, ids(), Framing::Bare);
        crate::engine::CallAssembler::track_value(&mut tracked);
        tracked.feed("The forecast ", &mut out);
        assert_eq!(tracked.value, vec![0..13]);
        tracked.feed("stays mild", &mut out);
        assert_eq!(tracked.value, vec![0..10]);
    }

    fn run(pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, ids(), Framing::Bare);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
        out.into_owned().drain()
    }

    #[test]
    fn an_object_without_arguments_is_a_call_with_an_empty_object_in_a_list_and_alone() {
        // The object's assembler says `{}` for a call whose object has no arguments member, so a
        // list's object and a lone object read as one written `"arguments": {}`, whole and byte
        // by byte.
        let list = "[{\"name\": \"f\"}, {\"name\": \"g\", \"arguments\": {}}]";
        let per_char: Vec<&str> = list
            .char_indices()
            .map(|(i, c)| &list[i..i + c.len_utf8()])
            .collect();
        for events in [run(&[list]), run(&per_char)] {
            assert_eq!(bytes(&events), list);
            assert_eq!(
                calls(&events),
                [
                    (0, "f".to_string(), "{}".to_string()),
                    (1, "g".to_string(), "{}".to_string()),
                ]
            );
        }
        let alone = "{\"name\": \"f\"}";
        let events = run_alone(&[alone]);
        assert_eq!(bytes(&events), alone);
        assert_eq!(calls(&events), [(0, "f".to_string(), "{}".to_string())]);
    }

    #[test]
    fn a_list_gives_each_object_as_a_call_with_the_brackets_and_commas_dropped() {
        let events = run(&[LIST]);
        assert_eq!(bytes(&events), LIST);
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris"}"#.to_string()
                ),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, ["[", ", ", "]"]);
    }

    #[test]
    fn prose_is_content_whole_and_a_cut_list_closes_nothing() {
        let events = run(&["  Hello, ", "[not a list]"]);
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(content, "  Hello, [not a list]");
        assert!(calls(&events).is_empty());

        let events = run(&["[{\"name\": \"get_weather\", \"argu"]);
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        assert_eq!(bytes(&events), "[{\"name\": \"get_weather\", \"argu");
    }

    #[test]
    fn an_object_that_names_no_call_spends_no_index_and_wrapping_after_the_list_is_dropped() {
        let output = "[{\"foo\": 1}, {\"name\": \"f\", \"arguments\": {}}]\n";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(calls(&events), [(0, "f".to_string(), "{}".to_string())]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n"
        )));
        // Text between objects comes back as one run; text after the list too, with the
        // whitespace before it, which is the model's once text follows.
        let output =
            "[{\"name\": \"f\", \"arguments\": {}} junk {\"name\": \"g\", \"arguments\": {}}] tail";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(calls(&events).len(), 2);
        let malformed: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(malformed, ["junk", " tail"]);
    }

    #[test]
    fn text_between_objects_keeps_the_models_whitespace_between_its_words() {
        // Prose between two objects of the list: the whitespace right after the first object is
        // the list's wrapping, the words keep the spaces between them (one event per word, the
        // space before it in the event), and the whitespace before the next object is the
        // list's again; the same whole, at every cut and one character at a time.
        let output =
            "[{\"name\": \"f\", \"arguments\": {}} some words here {\"name\": \"g\", \"arguments\": {}}]";
        let whole = whole_and_per_char(run, output);
        assert_eq!(calls(&whole).len(), 2);
        assert_eq!(
            malformed(&whole, MalformedReason::TextBetweenCalls),
            "some words here"
        );
        let words: Vec<&str> = whole
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(words, ["some", " words", " here"]);
        assert_eq!(dropped(&whole), ["[", " ", " ", "]"]);
        at_every_cut(run, output);
        // A comma between the words rides with them, the one before the next object is wrapping.
        let output = "[{\"name\": \"f\", \"arguments\": {}}, some, words, {\"name\": \"g\"}]";
        let whole = whole_and_per_char(run, output);
        assert_eq!(
            malformed(&whole, MalformedReason::TextBetweenCalls),
            "some, words"
        );
        assert_eq!(dropped(&whole), ["[", ", ", ", ", "]"]);
    }

    #[test]
    fn a_bracket_that_opens_no_list_of_calls_is_prose() {
        for output in [
            "[Note: draft] ok",
            "[1] First, then second.",
            "[docs](https://x) and more",
        ] {
            let events = run(&[output]);
            let content: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Content(text) => Some(&*text.text),
                    _ => None,
                })
                .collect();
            assert_eq!(content, output, "{output:?}");
            assert!(calls(&events).is_empty(), "{output:?}");
        }
    }

    fn run_alone(pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, ids(), Framing::ObjectOrList { opener: None });
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
        out.into_owned().drain()
    }

    #[test]
    fn a_lone_object_is_one_call_and_what_follows_it_is_wrapping() {
        let output = "  {\"name\": \"f\", \"parameters\": {\"a\": 1}}\n";
        let events = run_alone(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(0, "f".to_string(), r#"{"a": 1}"#.to_string())]
        );
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, ["  ", "\n"]);
        let output = "{\"name\": \"f\", \"arguments\": {}} tail";
        let events = run_alone(&[output]);
        assert_eq!(bytes(&events), output);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::TextAfterTheArguments }
                if text.text == " tail"
        )));
        // A list reads as before, prose is content, and a cut object closes nothing.
        assert_eq!(calls(&run_alone(&[LIST])).len(), 2);
        assert!(calls(&run_alone(&["Sure: {\"name\": \"f\"}"])).is_empty());
        let events = run_alone(&["{\"name\": \"f\", \"argu"]);
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
        assert_eq!(bytes(&events), "{\"name\": \"f\", \"argu");
        // The list-only assembler keeps a `{` as prose.
        let events = run(&["{\"name\": \"f\", \"arguments\": {}}"]);
        assert!(calls(&events).is_empty());
        let whole = calls(&run_alone(&[output]));
        for cut in 1..output.len() {
            let events = run_alone(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(calls(&events), whole, "cut at {cut}");
        }
    }

    /// The malformed text of `events` under `why`, joined.
    fn malformed(events: &[Event], why: MalformedReason) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why: reason } if *reason == why => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    fn dropped(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn prose_after_a_lone_object_keeps_its_whitespace_whole_and_in_pieces() {
        // A JSON answer the model explains: the reference keeps the sentence as written.
        let output = "{\n \"a\": true\n} \nA baseball game can be described as exciting.";
        let prose = " \nA baseball game can be described as exciting.";
        let whole = run_alone(&[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(
            malformed(&whole, MalformedReason::TextAfterTheArguments),
            prose
        );
        assert!(dropped(&whole).is_empty(), "{whole:?}");
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let cut = run_alone(&refs);
        assert_eq!(bytes(&cut), output);
        assert_eq!(
            malformed(&cut, MalformedReason::TextAfterTheArguments),
            prose
        );
        assert!(dropped(&cut).is_empty(), "{cut:?}");
        // After a call too, and after a list's `]`.
        let output = "{\"name\": \"f\", \"arguments\": {}}\n\nDone, see above.";
        for events in [
            run_alone(&[output]),
            run_alone(&[
                "{\"name\": \"f\", \"arguments\": {}}\n",
                "\nDone, see above.",
            ]),
        ] {
            assert_eq!(calls(&events).len(), 1);
            assert_eq!(
                malformed(&events, MalformedReason::TextAfterTheArguments),
                "\n\nDone, see above."
            );
            assert!(dropped(&events).is_empty(), "{events:?}");
        }
        let output = "[{\"name\": \"f\", \"arguments\": {}}] and so on";
        for events in [
            run(&[output]),
            run(&["[{\"name\": \"f\", \"arguments\": {}}]", " ", "and so on"]),
        ] {
            assert_eq!(
                malformed(&events, MalformedReason::TextAfterTheList),
                " and so on"
            );
            assert_eq!(dropped(&events), ["[", "]"]);
        }
        // Whitespace alone after the object or the list stays the template's wrapping.
        for events in [
            run_alone(&["{\"name\": \"f\", \"arguments\": {}} \n"]),
            run_alone(&["{\"name\": \"f\", \"arguments\": {}}", " ", "\n"]),
        ] {
            assert_eq!(dropped(&events), [" \n"]);
            assert!(malformed(&events, MalformedReason::TextAfterTheArguments).is_empty());
        }
    }

    /// [`run_alone`] with Llama 3.1's call marker named.
    fn run_opened(pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(
            0,
            ids(),
            Framing::ObjectOrList {
                opener: Some("<|python_tag|>"),
            },
        );
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
        out.into_owned().drain()
    }

    /// `output` through `run`, whole and one character at a time: the same bytes, calls,
    /// dropped text and malformed text either way; the events of the whole.
    fn whole_and_per_char(
        run: fn(&[&str]) -> Vec<Event<'static>>,
        output: &str,
    ) -> Vec<Event<'static>> {
        let whole = run(&[output]);
        assert_eq!(bytes(&whole), output);
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let cut = run(&refs);
        assert_eq!(bytes(&cut), output);
        assert_eq!(calls(&cut), calls(&whole), "{output:?}");
        assert_eq!(
            dropped(&cut).concat(),
            dropped(&whole).concat(),
            "{output:?}"
        );
        for why in [
            MalformedReason::TextAfterTheArguments,
            MalformedReason::TextAfterTheList,
            MalformedReason::TextBetweenCalls,
        ] {
            assert_eq!(malformed(&cut, why), malformed(&whole, why), "{output:?}");
        }
        whole
    }

    #[test]
    fn a_semicolon_after_a_lone_object_separates_the_next_call() {
        let (a, b, c) = (
            "{\"name\": \"a\", \"parameters\": {\"n\": 1}}",
            "{\"name\": \"b\", \"parameters\": {\"n\": 2}}",
            "{\"name\": \"c\"}",
        );
        let expected = [
            (0, "a".to_string(), r#"{"n": 1}"#.to_string()),
            (1, "b".to_string(), r#"{"n": 2}"#.to_string()),
            (2, "c".to_string(), "{}".to_string()),
        ];
        for separator in ["; ", ";", " ;\n "] {
            let output = format!("{a}{separator}{b}{separator}{c}");
            let events = whole_and_per_char(run_alone, &output);
            assert_eq!(calls(&events), expected, "{separator:?}");
            assert_eq!(dropped(&events), [separator, separator], "{separator:?}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{separator:?}: {events:?}"
            );
        }
        // Whitespace after the last object stays wrapping; the list-only assembler reads none
        // of this, a `{` being prose to it.
        let events = whole_and_per_char(run_alone, &format!("{a}; {b}\n"));
        assert_eq!(calls(&events).len(), 2);
        assert_eq!(dropped(&events), ["; ", "\n"]);
        assert!(calls(&run(&[&format!("{a}; {b}")])).is_empty());
    }

    #[test]
    fn a_semicolon_that_no_object_follows_is_text_after_the_arguments() {
        let a = "{\"name\": \"a\", \"parameters\": {\"n\": 1}}";
        for tail in [
            ";",
            "; ",
            ";\n\n",
            "; and then some",
            "; [{\"name\": \"b\"}]",
            ";;",
        ] {
            let output = format!("{a}{tail}");
            let events = whole_and_per_char(run_alone, &output);
            assert_eq!(calls(&events).len(), 1, "{tail:?}");
            assert_eq!(
                malformed(&events, MalformedReason::TextAfterTheArguments),
                tail,
                "{tail:?}"
            );
            assert!(dropped(&events).is_empty(), "{tail:?}: {events:?}");
            // One event per run of text, and never one of no bytes: the held `;` joins the
            // prose's run, so telemetry counts the run once, as it does without the `;`.
            let malformed_events = events
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count();
            assert_eq!(malformed_events, 1, "{tail:?}: {events:?}");
            assert!(
                !events.iter().any(|event| matches!(
                    event,
                    Event::Malformed { text, .. } | Event::Content(text) if text.text.is_empty()
                )),
                "{tail:?}: {events:?}"
            );
        }
        // After a list, a `;` is text after the list, as any text is.
        let events = whole_and_per_char(run_alone, &format!("[{a}]; {a}"));
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(
            malformed(&events, MalformedReason::TextAfterTheList),
            format!("; {a}")
        );
    }

    /// `output` through `run` at every two-way cut: the same bytes, calls and wrapping as whole.
    fn at_every_cut(run: fn(&[&str]) -> Vec<Event<'static>>, output: &str) {
        let whole = run(&[output]);
        for cut in 1..output.len() {
            if !output.is_char_boundary(cut) {
                continue;
            }
            let events = run(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
            assert_eq!(
                dropped(&events).concat(),
                dropped(&whole).concat(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn objects_that_follow_a_lone_object_on_whitespace_alone_are_further_calls() {
        // A checkpoint that writes its parallel calls as consecutive objects, a blank line or a
        // newline between them and no `;` (Llama 3.2 Instruct), or nothing between them: every
        // object is a call, as the engines' own readers of the family read them, the whitespace
        // between them the template's wrapping; two, three and four calls, whole, per character
        // and at every cut.
        let objects = [
            "{\"name\": \"a\", \"parameters\": {\"n\": 1}}",
            "{\n    \"name\": \"b\",\n    \"parameters\": {\n        \"n\": 2\n    }\n}",
            "{\"name\": \"c\"}",
            "{\"name\": \"d\", \"arguments\": {\"n\": 4}}",
        ];
        let expected = [
            (0, "a".to_string(), r#"{"n": 1}"#.to_string()),
            (1, "b".to_string(), "{\n        \"n\": 2\n    }".to_string()),
            (2, "c".to_string(), "{}".to_string()),
            (3, "d".to_string(), r#"{"n": 4}"#.to_string()),
        ];
        for count in 2..=4 {
            for separator in ["\n\n", "\n", " ", "", "\n\n\n", " \n "] {
                let output = objects[..count].join(separator);
                let events = whole_and_per_char(run_alone, &output);
                assert_eq!(calls(&events), expected[..count], "{count} {separator:?}");
                let wrapping: Vec<&str> = if separator.is_empty() {
                    Vec::new()
                } else {
                    vec![separator; count - 1]
                };
                assert_eq!(dropped(&events), wrapping, "{count} {separator:?}");
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, Event::Malformed { .. })),
                    "{count} {separator:?}: {events:?}"
                );
                at_every_cut(run_alone, &output);
            }
        }
        let [a, b, c, _] = objects;
        // A `;` and whitespace alone mix; whitespace after the last object stays wrapping; prose
        // after the sequence is text after the arguments, as after one object, and the wrapping
        // before it is still wrapping.
        let events = whole_and_per_char(run_alone, &format!("{a}; {b}\n\n{c}\n"));
        assert_eq!(calls(&events), expected[..3]);
        assert_eq!(dropped(&events), ["; ", "\n\n", "\n"]);
        let events = whole_and_per_char(run_alone, &format!("{a}\n\n{b}\n\nDone."));
        assert_eq!(calls(&events), expected[..2]);
        assert_eq!(
            malformed(&events, MalformedReason::TextAfterTheArguments),
            "\n\nDone."
        );
        assert_eq!(dropped(&events), ["\n\n"]);
        at_every_cut(run_alone, &format!("{a}\n\n{b}\n\nDone."));
        // Behind the family's call marker too; after a list an object is text after the list, as
        // before; the list-only reader keeps the first `{` as prose, as before.
        let events = whole_and_per_char(run_opened, &format!("<|python_tag|>{a}\n\n{b}"));
        assert_eq!(calls(&events), expected[..2]);
        assert_eq!(dropped(&events), ["<|python_tag|>", "\n\n"]);
        let events = whole_and_per_char(run_alone, &format!("[{a}]\n\n{b}"));
        assert_eq!(calls(&events), expected[..1]);
        assert_eq!(
            malformed(&events, MalformedReason::TextAfterTheList),
            format!("\n\n{b}")
        );
        assert!(calls(&run(&[&format!("{a}\n\n{b}")])).is_empty());
    }

    #[test]
    fn the_call_marker_at_the_start_is_wrapping_and_elsewhere_or_cut_short_it_is_text() {
        let a = "{\"name\": \"a\", \"parameters\": {\"n\": 1}}";
        let one = [(0, "a".to_string(), r#"{"n": 1}"#.to_string())];
        // The marker, with or without whitespace around it, before an object, a list, or a
        // `;`-separated pair.
        for (output, wrapping) in [
            (format!("<|python_tag|>{a}"), vec!["<|python_tag|>"]),
            (
                format!("  <|python_tag|>\n{a}"),
                vec!["  <|python_tag|>", "\n"],
            ),
            (
                format!("<|python_tag|>[{a}]"),
                vec!["<|python_tag|>", "[", "]"],
            ),
        ] {
            let events = whole_and_per_char(run_opened, &output);
            assert_eq!(calls(&events), one, "{output:?}");
            assert_eq!(dropped(&events), wrapping, "{output:?}");
        }
        let events = whole_and_per_char(run_opened, &format!("<|python_tag|>{a}; {a}"));
        assert_eq!(calls(&events).len(), 2);
        assert_eq!(dropped(&events), ["<|python_tag|>", "; "]);
        // Without the marker the opened assembler reads as the plain one.
        assert_eq!(calls(&whole_and_per_char(run_opened, a)), one);
        // Prose after the marker is content, the marker dropped; the marker inside prose, a
        // marker cut short, and text that begins like it are content whole.
        let events = whole_and_per_char(run_opened, "<|python_tag|>no call");
        assert!(calls(&events).is_empty());
        assert_eq!(dropped(&events), ["<|python_tag|>"]);
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Content(text) => Some(&*text.text),
                    _ => None,
                })
                .collect::<String>(),
            "no call"
        );
        for output in [
            format!("Sure: <|python_tag|>{a}"),
            "<|python_".to_string(),
            "<|python_tag|".to_string(),
            "<|pyxx".to_string(),
            "<".to_string(),
            " <x>".to_string(),
        ] {
            let events = whole_and_per_char(run_opened, &output);
            assert!(calls(&events).is_empty(), "{output:?}");
            assert!(dropped(&events).is_empty(), "{output:?}");
            assert!(
                events
                    .iter()
                    .all(|event| matches!(event, Event::Content(_))),
                "{output:?}: {events:?}"
            );
        }
        // After the object the marker is text after the arguments; the plain assembler keeps
        // the marker as prose.
        let events = whole_and_per_char(run_opened, &format!("{a}<|python_tag|>{a}"));
        assert_eq!(calls(&events), one);
        assert_eq!(
            malformed(&events, MalformedReason::TextAfterTheArguments),
            format!("<|python_tag|>{a}")
        );
        let events = whole_and_per_char(run_alone, &format!("<|python_tag|>{a}"));
        assert!(calls(&events).is_empty());
        assert!(events
            .iter()
            .all(|event| matches!(event, Event::Content(_))));
    }

    #[test]
    fn between_markers_text_in_place_of_the_list_is_malformed() {
        // Text in place of the calls is malformed whole under both framings between markers;
        // the object alone is a call only where the framing reads one.
        let alone = "{\"name\": \"f\", \"arguments\": {}}";
        for (framing, output, held) in [
            (Framing::Framed, "not a list", 0),
            (Framing::Framed, "  [Note: draft] ok", 0),
            (Framing::Framed, alone, 0),
            (Framing::Framed, LIST, 2),
            (Framing::FramedObjectOrList, "not a list", 0),
            (Framing::FramedObjectOrList, "  [Note: draft] ok", 0),
            (Framing::FramedObjectOrList, alone, 1),
            (Framing::FramedObjectOrList, LIST, 2),
        ] {
            let mut assembler = Assembler::new(0, ids(), framing);
            let mut out = Events::new();
            assembler.feed(output, &mut out);
            assembler.close(&mut out);
            let events = out.drain();
            assert_eq!(bytes(&events), output, "{framing:?} {output:?}");
            assert_eq!(calls(&events).len(), held, "{framing:?} {output:?}");
            let malformed = |event: &Event| matches!(event, Event::Malformed { .. });
            if held == 0 {
                assert!(
                    events.iter().all(malformed),
                    "{framing:?} {output:?}: {events:?}"
                );
            } else {
                assert!(
                    !events.iter().any(malformed),
                    "{framing:?} {output:?}: {events:?}"
                );
            }
        }
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = calls(&run(&[LIST]));
        for cut in 1..LIST.len() {
            let events = run(&[&LIST[..cut], &LIST[cut..]]);
            assert_eq!(bytes(&events), LIST, "cut at {cut}");
            assert_eq!(calls(&events), whole, "cut at {cut}");
        }
    }
}
