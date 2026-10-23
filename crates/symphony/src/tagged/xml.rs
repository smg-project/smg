//! An XML tree of arguments: the events of one `<invoke name="NAME">` block whose arguments are
//! elements, from its bytes.
//!
//! MiniMax M3 writes a call as
//!
//! ```text
//! <invoke name="change_drink">
//! <drink_id>latte</drink_id>
//! <new_preferences><size>large</size><temperature>hot</temperature></new_preferences>
//! <recipients><item>admin@example.com</item><item>support@example.com</item></recipients>
//! </invoke>
//! ```
//!
//! with its separator token `]<]minimax[>[` before every tag and no newline anywhere (the lines
//! above are for reading). The engine's table owns the invoke markers and the separator
//! ([`formats::minimax_m3`]): it enters this assembler at `<invoke name="`, hands it every
//! separator through [`Assembler::ignored`], and leaves at `</invoke>`, which it hands over as the
//! call's end. Between those the assembler reads the name and the elements itself.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.
//!
//! [`formats::minimax_m3`]: crate::formats::minimax_m3()

use crate::{
    event::{runs, DropReason, Event, Events, MalformedReason, Text},
    json::{PartialJson, PartialJsonError},
    tagged::{
        assembler::{push_escaped, push_quoted},
        end_cut_call,
        value::{json, Kind, ITEM, NULL_WORDS},
    },
    Declared,
};

/// The nesting the arguments may reach, the arguments object counted: the prefix parser's bound
/// ([`PartialJson::DEFAULT_MAX_DEPTH`]), which the JSON syntax's judge applies, so that the
/// fragments of a call stay a prefix that parser takes, whichever syntax wrote them. An element
/// becomes a container at its first child's tag; the tag that would make the element past the
/// bound a container is refused with the rest of the call ([`Assembler::open_tag`]), and so is
/// a leaf whose own JSON (a list or an object the text spells) would nest past it, or an empty
/// element the tool declares a container at the bound ([`Assembler::close_element`]).
const DEEPEST: usize = PartialJson::DEFAULT_MAX_DEPTH;

/// The invoke's closing tag: the one terminal a started call's end carries. The next invoke's
/// opening and the block's close end a call too, but they belong to the region, and the engine
/// drops them.
pub const INVOKE_CLOSE: &str = "</invoke>";

/// The events of one invoke, from the bytes after `<invoke name="`.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    /// The name while it is read, then the open leaf's text.
    value: String,
    /// Where the bytes after the name's quote start in `carried`, while the invoke tag is read.
    name_end: usize,
    stage: Stage,
    /// The function's name once `ToolCallStart` was pushed.
    function: Option<String>,
    /// The elements open below the arguments object, outermost first.
    open: Vec<Element>,
    /// Members written to the arguments object so far.
    written: u32,
    /// A tag being read: the bytes after its `<`, while they still spell a tag.
    tag: Option<String>,
    /// Whether the last bytes taken were the table's ignored terminal: the separator the
    /// template writes before every tag, and never inside a value.
    after_separator: bool,
    /// Whether text where the syntax has tags was refused since the last tag the syntax read (a
    /// word between tags, a tag out of its place, a tag refused with its bytes): the whitespace
    /// carried after it is the model's, between two pieces of its text, and leaves with the next
    /// piece ([`Self::refuse_between`]); before any, the whitespace after a tag is the template's,
    /// dropped before a refusal or carried into the next event's source.
    released: bool,
    /// Whether the arguments would have nested past [`DEEPEST`]: a tag that would have opened a
    /// container past it ([`Self::open_tag`]), a leaf whose own JSON would nest past it, or an
    /// empty element declared a container at the bound ([`Self::close_element`]). The arguments
    /// are no longer read, every byte from the tag of the element that would cross the bound to
    /// the call's end is held and refused at the end as invalid arguments (counted once, the
    /// bytes dropped as a malformed call's), nothing of that element's member is written, and
    /// the containers still open close whole at the end.
    refused: bool,
    done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Inside `name="…"`.
    Name,
    /// After the name's quote, before the tag's `>`.
    Tail,
    /// The elements.
    Body,
}

#[derive(Clone, Debug)]
struct Element {
    key: String,
    shape: Shape,
    /// Children written so far, once the element is a container.
    written: u32,
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// No child yet.
    Open,
    Object,
    List,
    Leaf(Leaf),
}

#[derive(Clone, Copy, Debug)]
struct Leaf {
    kind: Option<Kind>,
    mode: Mode,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// A declared string, its opening fragment out, pushed piece by piece.
    Streaming,
    /// Written at the close with [`json`].
    Whole,
    /// A string that may be null: held until its text rules the null words out.
    Undecided,
}

/// The characters a tag's name is made of: anything but whitespace and the angle brackets, since
/// the template writes any key (BFCL's hold `ñ`, OData's `$filter`, MongoDB's `$gt`); `/` is
/// allowed anywhere here and judged when the tag is whole.
fn is_tag_byte(c: char) -> bool {
    !c.is_whitespace() && !matches!(c, '<' | '>')
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after `<invoke name="`.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            carried: String::new(),
            value: String::new(),
            name_end: 0,
            stage: Stage::Name,
            function: None,
            open: Vec::new(),
            written: 0,
            tag: None,
            after_separator: false,
            released: false,
            refused: false,
            done: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        let mut rest = bytes;
        while !rest.is_empty() {
            rest = self.take(rest, declared, out);
        }
    }

    /// The table's ignored terminal arrived inside the call: a tag held before it is released, as
    /// the separator ends it, and what follows stands after a separator. Its own bytes are the
    /// engine's to place, in their order among the bytes this assembler holds.
    pub fn ignored(&mut self, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        self.release_tag(declared, out);
        self.after_separator = true;
    }

    /// The call's end: `terminal` is the closing marker the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. Every open container is
    /// closed, a leaf never written is refused, the object is closed; inside a streamed string
    /// the string is closed first ([`Self::close_open_string`]), since the model ended the call
    /// and its bytes are the value whole, under one `Malformed { OpenAtTheMarker }` event of no
    /// bytes, and the containers above it close as after any leaf; an invoke whose name never
    /// closed, or that had none, has its bytes refused. After a tag or a value that would have
    /// nested the arguments past [`DEEPEST`], the bytes held from the element that would cross
    /// the bound are refused first ([`refuse`](super::refuse): one counted `Malformed {
    /// InvalidArguments }` of no bytes, the bytes dropped as a malformed call's), and that
    /// element's member is never written.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        self.hold_tag();
        if self.stage != Stage::Body || !self.started() {
            self.carried.push_str(terminal);
            self.report(MalformedReason::CallWithoutName, out);
            return;
        }
        self.close_open_string(MalformedReason::OpenAtTheMarker, out);
        if self.refused {
            self.value.clear();
            super::refuse(
                std::mem::take(&mut self.carried),
                MalformedReason::InvalidArguments,
                out,
            );
        }
        while let Some(element) = self.open.last() {
            match element.shape {
                Shape::Object => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.push_fragment("}".to_string(), source, out);
                }
                Shape::List => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.push_fragment("]".to_string(), source, out);
                }
                // A value never written, or an element the end cut before its first child: its
                // bytes are refused, and its member is never written.
                Shape::Leaf(_) | Shape::Open => {
                    self.value.clear();
                    self.report(MalformedReason::TagCutShort, out);
                }
            }
            self.open.pop();
        }
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.close_object(source, out);
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(terminal.to_owned()),
        });
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A streamed string's open fragment stays open; a call that started still ends, with
    /// no bytes of its own, as every assembler ends one, after the one `Malformed {
    /// UnterminatedRegion }` event that carries everything held and says that its arguments are
    /// no whole value ([`end_cut_call`]); what a block that named no call held is refused as
    /// `UnterminatedRegion` too, as the keyed assembler refuses it, so that the reader's release
    /// of a refused output that made no call takes the whole slice.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.hold_tag();
        if self.started() {
            end_cut_call(self.index, std::mem::take(&mut self.carried), "", out);
        } else {
            self.report(MalformedReason::UnterminatedRegion, out);
        }
    }

    /// The stream's stop inside the invoke: the model's own end of output, not a cut. The call
    /// ends as its closing tag would end it, with no bytes for the end ([`Self::close`]): every
    /// open container is closed, a leaf never written and an element without a child are
    /// refused, the object is closed, an invoke whose name never closed has its bytes refused.
    /// A streamed string still open is closed first ([`Self::close_open_string`]), under the
    /// stop's word. One `Malformed { OpenAtTheStop }` event of no bytes stands before the fragment
    /// that closes what was open ([`open_at_the_stop`](super::open_at_the_stop)): for a string
    /// the stop closed under any table, and for the call itself where the table has no stop row
    /// (`as_marker`), since there the template writes the marker the model skipped.
    pub fn stop(mut self, as_marker: bool, out: &mut Events) {
        if self.done {
            return;
        }
        self.hold_tag();
        let string_closed = self.close_open_string(MalformedReason::OpenAtTheStop, out);
        if !string_closed && self.started() && !as_marker {
            super::open_at_the_stop(out);
        }
        self.close("", out);
    }

    /// A streamed string still open when the call ends, at its closing marker or at the stream's
    /// stop: the model wrote nothing more of the value after the call's end, so its bytes are the
    /// value whole, and the start of a tag held is its text too, since no tag followed. The string
    /// is closed: the one word of no bytes (`why`: the end that closed it,
    /// [`left_open`](super::left_open)), then the closing quote; the containers above it are the
    /// ending's to close. Says whether a string was open; nothing happens otherwise.
    fn close_open_string(&mut self, why: MalformedReason, out: &mut Events) -> bool {
        if !matches!(
            self.open.last(),
            Some(Element {
                shape: Shape::Leaf(Leaf {
                    mode: Mode::Streaming,
                    ..
                }),
                ..
            })
        ) {
            return false;
        }
        let held = std::mem::take(&mut self.carried);
        if !held.is_empty() {
            self.leaf_text(&held, out);
        }
        super::left_open(why, out);
        self.open.pop();
        self.value.clear();
        self.push_fragment("\"".to_string(), Text::default(), out);
        true
    }

    /// Reads as much of `text` as one step takes and returns the rest.
    fn take<'a>(&mut self, text: &'a str, declared: &Declared, out: &mut Events) -> &'a str {
        if self.refused {
            // Nesting past the bound: the rest of the call is held for the report at its end.
            self.carried.push_str(text);
            return "";
        }
        match self.stage {
            Stage::Name => match text.find('"') {
                Some(at) => {
                    self.carried.push_str(&text[..=at]);
                    self.value.push_str(&text[..at]);
                    self.name_end = self.carried.len();
                    self.stage = Stage::Tail;
                    &text[at + 1..]
                }
                None => {
                    self.carried.push_str(text);
                    self.value.push_str(text);
                    ""
                }
            },
            Stage::Tail => match text.find('>') {
                Some(at) => {
                    self.carried.push_str(&text[..at]);
                    self.start_call(out);
                    &text[at + 1..]
                }
                None => {
                    self.carried.push_str(text);
                    ""
                }
            },
            Stage::Body => self.body(text, declared, out),
        }
    }

    /// The invoke tag is whole: the call starts with the name's bytes and its quote as the start's
    /// source; a tail between the quote and `>` is refused; the `>` is carried into the next
    /// source. An invoke with no name starts no call, and the tag's bytes are refused.
    fn start_call(&mut self, out: &mut Events) {
        let name = std::mem::take(&mut self.value);
        self.stage = Stage::Body;
        // The tail is every byte after the name's quote, a separator that arrived meanwhile
        // included; `name_end` is a character boundary, so the split never lands inside one.
        let tail_bytes = self.carried.split_off(self.name_end);
        if name.is_empty() {
            self.carried.push_str(&tail_bytes);
            self.carried.push('>');
            self.report(MalformedReason::EmptyName, out);
            return;
        }
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
        self.function = Some(name);
        self.released = false;
        super::refuse(tail_bytes, MalformedReason::TagTail, out);
        self.carried.push('>');
    }

    /// The elements: a tag being read takes its bytes up to `>`, or stops being a tag at the first
    /// byte a name cannot hold; text runs to the next `<`.
    fn body<'a>(&mut self, text: &'a str, declared: &Declared, out: &mut Events) -> &'a str {
        if let Some(tag) = &mut self.tag {
            let Some((at, byte)) = text.char_indices().find(|(_, c)| !is_tag_byte(*c)) else {
                tag.push_str(text);
                return "";
            };
            tag.push_str(&text[..at]);
            if byte == '>' {
                let tag = self.tag.take().unwrap_or_default();
                self.complete_tag(&tag, declared, out);
                return &text[at + 1..];
            }
            // Not a tag after all: the `<` and what followed it are text, and this byte is read
            // again as text or as the start of a tag.
            self.release_tag(declared, out);
            return &text[at..];
        }
        match text.find('<') {
            Some(at) => {
                if at > 0 {
                    self.text(&text[..at], declared, out);
                }
                self.tag = Some(String::new());
                &text[at + 1..]
            }
            None => {
                self.text(text, declared, out);
                ""
            }
        }
    }

    /// A `<` and the bytes after it were not a tag: they are text where they stand.
    fn release_tag(&mut self, declared: &Declared, out: &mut Events) {
        if let Some(tag) = self.tag.take() {
            let text = format!("<{tag}");
            self.text(&text, declared, out);
        }
    }

    /// A tag the end cut short: its bytes are held with the rest.
    fn hold_tag(&mut self) {
        if let Some(tag) = self.tag.take() {
            self.carried.push('<');
            self.carried.push_str(&tag);
        }
    }

    /// A whole tag: an opening or a closing one, or text when its name is not one.
    fn complete_tag(&mut self, tag: &str, declared: &Declared, out: &mut Events) {
        let bytes = format!("<{tag}>");
        let (closing, name) = match tag.strip_prefix('/') {
            Some(name) => (true, name),
            None => (false, tag),
        };
        if name.is_empty() || name.contains('/') {
            self.text(&bytes, declared, out);
            return;
        }
        if !self.started() {
            self.refuse_between(&bytes, MalformedReason::TagOutOfPlace, out);
            return;
        }
        if closing {
            self.close_tag(name, &bytes, declared, out);
        } else {
            self.open_tag(name, &bytes, declared, out);
        }
        self.after_separator = false;
    }

    /// An opening tag: inside a leaf it is the leaf's text; under an element with no child yet
    /// it makes that element a container or a leaf, by what the tool declares there, or, with no
    /// declaration, by whether the template's separator came before it (the template writes the
    /// separator before every tag it writes and never inside a value); then it opens an element
    /// of its own. An element that would become the container past [`DEEPEST`] (the arguments
    /// object and the containers open above it counted) is refused instead, with every byte
    /// from its own tag to the call's end ([`Self::close`]): the prefix parser would refuse the
    /// nesting, so the fragments stop where they are still a prefix it takes, as the JSON syntax
    /// stops at the bracket it cannot take; nothing of the element's member is written, so the
    /// containers above it close whole at the end without it.
    fn open_tag(&mut self, key: &str, bytes: &str, declared: &Declared, out: &mut Events) {
        match self.open.last().map(|element| element.shape) {
            Some(Shape::Leaf(_)) => {
                self.leaf_text(bytes, out);
                return;
            }
            Some(Shape::Open) => {
                let leaf = match self.kind_here(declared) {
                    Some(Kind::String | Kind::NullableString | Kind::Integer) => true,
                    Some(Kind::Array | Kind::Object) => false,
                    None => !self.after_separator,
                };
                if leaf {
                    self.become_leaf(declared, out);
                    self.leaf_text(bytes, out);
                    return;
                }
                if self.open.len() >= DEEPEST {
                    self.carried.push_str(bytes);
                    self.value.clear();
                    self.refused = true;
                    return;
                }
                self.decide_container(key == ITEM, declared, out);
            }
            Some(Shape::Object | Shape::List) | None => {}
        }
        self.carried.push_str(bytes);
        self.value.clear();
        self.open.push(Element {
            key: key.to_string(),
            shape: Shape::Open,
            written: 0,
        });
        self.released = false;
    }

    /// A closing tag: it closes the innermost element when it names it; inside a leaf, or under
    /// an element with no child yet, any other closing tag is text; elsewhere it is refused.
    fn close_tag(&mut self, name: &str, bytes: &str, declared: &Declared, out: &mut Events) {
        match self.open.last() {
            Some(element) if element.key == name => self.close_element(bytes, declared, out),
            Some(Element {
                shape: Shape::Leaf(_) | Shape::Open,
                ..
            }) => self.text(bytes, declared, out),
            Some(_) | None => self.refuse_between(bytes, MalformedReason::TagOutOfPlace, out),
        }
    }

    /// The element's first child tag says what it is: a list when the tool declares an array there
    /// or, failing that, when the child is an `<item>`; else an object. Its opening fragment goes
    /// out with the bytes held since its own tag.
    fn decide_container(&mut self, item: bool, declared: &Declared, out: &mut Events) {
        let list = match self.kind_here(declared) {
            Some(Kind::Array) => true,
            Some(Kind::Object) => false,
            _ => item,
        };
        let mut fragment = self.prefix();
        fragment.push(if list { '[' } else { '{' });
        if let Some(element) = self.open.last_mut() {
            element.shape = if list { Shape::List } else { Shape::Object };
        }
        self.value.clear();
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(fragment, source, out);
    }

    /// What a member's fragment starts with: the arguments object's `{`, or `, ` after an earlier
    /// member of the same container, and the key unless the container is a list. Counts the member
    /// for its container.
    fn prefix(&mut self) -> String {
        let depth = self.open.len();
        let key = self
            .open
            .last()
            .map(|element| element.key.clone())
            .unwrap_or_default();
        let mut fragment = String::new();
        let list = if depth >= 2 {
            let parent = &mut self.open[depth - 2];
            if parent.written > 0 {
                fragment.push_str(", ");
            }
            parent.written += 1;
            matches!(parent.shape, Shape::List)
        } else {
            fragment.push_str(if self.written == 0 { "{" } else { ", " });
            self.written += 1;
            false
        };
        if !list {
            push_quoted(&mut fragment, &key);
            fragment.push_str(": ");
        }
        fragment
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        self.after_separator = false;
        match self.open.last().map(|element| element.shape) {
            Some(Shape::Leaf(_)) => self.leaf_text(text, out),
            Some(Shape::Open) => {
                // The element's first text: whitespace alone is held, since a child tag may still
                // follow; anything else makes the element a leaf, the whitespace part of it.
                self.value.push_str(text);
                self.carried.push_str(text);
                if !self.value.trim().is_empty() {
                    self.become_leaf(declared, out);
                }
            }
            Some(Shape::Object | Shape::List) | None => self.text_between(text, out),
        }
    }

    /// Text where the syntax has tags: whitespace is carried, since whose it is depends on what
    /// follows, anything else is refused with the whitespace before it when text was refused
    /// since the last tag the syntax read, after it dropped otherwise ([`Self::refuse_between`]).
    fn text_between(&mut self, text: &str, out: &mut Events) {
        for (run, space) in runs(text) {
            if space {
                self.carried.push_str(run);
            } else {
                self.refuse_between(run, MalformedReason::TextBetweenTags, out);
            }
        }
    }

    /// The element holds text: it is a leaf, typed by what the tool declares at its path.
    fn become_leaf(&mut self, declared: &Declared, out: &mut Events) {
        let kind = self.kind_here(declared);
        let mode = match kind {
            Some(Kind::String) => Mode::Streaming,
            Some(Kind::NullableString) => Mode::Undecided,
            Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole,
        };
        if let Some(element) = self.open.last_mut() {
            element.shape = Shape::Leaf(Leaf { kind, mode });
        }
        match mode {
            Mode::Streaming => self.start_streaming(out),
            Mode::Undecided => self.decide_null(out),
            Mode::Whole => {}
        }
    }

    /// The kind the tools declare for the innermost open element.
    fn kind_here(&self, declared: &Declared) -> Option<Kind> {
        let function = self.function.as_deref()?;
        let path: Vec<&str> = self
            .open
            .iter()
            .map(|element| element.key.as_str())
            .collect();
        declared.kind_at(function, &path)
    }

    /// A string that may be null is held while its text could still spell a null word; once it
    /// cannot, it is a string after all, and streams.
    fn decide_null(&mut self, out: &mut Events) {
        let so_far = self.value.as_str();
        if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
            self.start_streaming(out);
        }
    }

    /// The opening fragment of a streamed string, with every byte held so far as its source, and
    /// then the text that arrived before the decision, already accounted for.
    fn start_streaming(&mut self, out: &mut Events) {
        let mut opening = self.prefix();
        opening.push('"');
        if let Some(Element {
            shape: Shape::Leaf(leaf),
            ..
        }) = self.open.last_mut()
        {
            leaf.mode = Mode::Streaming;
        }
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(opening, source, out);
        let arrived = std::mem::take(&mut self.value);
        if !arrived.is_empty() {
            let mut json = String::with_capacity(arrived.len());
            push_escaped(&mut json, &arrived);
            self.push_fragment(json, Text::default(), out);
        }
    }

    fn leaf_text(&mut self, text: &str, out: &mut Events) {
        let Some(Element {
            shape: Shape::Leaf(leaf),
            ..
        }) = self.open.last()
        else {
            return;
        };
        match leaf.mode {
            Mode::Whole => {
                self.value.push_str(text);
                self.carried.push_str(text);
            }
            Mode::Undecided => {
                self.value.push_str(text);
                self.carried.push_str(text);
                self.decide_null(out);
            }
            Mode::Streaming => {
                let mut json = String::with_capacity(text.len());
                push_escaped(&mut json, text);
                let mut source = std::mem::take(&mut self.carried);
                source.push_str(text);
                self.push_fragment(json, Text::uncounted(source), out);
            }
        }
    }

    /// The innermost element's closing tag: a streamed string gets its quote, a held leaf is
    /// written through [`json`], a container closes, an element with no child is empty by its
    /// declared type. A leaf whose own JSON would nest the arguments past [`DEEPEST`], and an
    /// empty element declared a container at the bound, are refused instead, with every byte
    /// from the element's own tag to the call's end ([`Self::close`]), as a tag that would nest
    /// past the bound is ([`Self::open_tag`]): nothing of the member is written.
    fn close_element(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        let Some(shape) = self.open.last().map(|element| element.shape) else {
            return;
        };
        let written = match shape {
            Shape::Leaf(Leaf {
                mode: Mode::Streaming,
                ..
            }) => "\"".to_string(),
            Shape::Leaf(Leaf { kind, .. }) => json(&self.value, kind),
            Shape::Object => "}".to_string(),
            Shape::List => "]".to_string(),
            // Whitespace alone between the tags: a leaf after all.
            Shape::Open if !self.value.is_empty() => json(&self.value, self.kind_here(declared)),
            Shape::Open => {
                // The template writes `None` as an empty element only inside a list; a mapping
                // skips a `None` member, so an empty element anywhere else is an empty string.
                let in_list = matches!(
                    self.open.iter().rev().nth(1),
                    Some(Element {
                        shape: Shape::List,
                        ..
                    })
                );
                match self.kind_here(declared) {
                    Some(Kind::Array) => "[]",
                    Some(Kind::Object) => "{}",
                    Some(_) => "\"\"",
                    None if in_list => "null",
                    None => "\"\"",
                }
                .to_string()
            }
        };
        let fragment = match shape {
            Shape::Leaf(Leaf {
                mode: Mode::Streaming,
                ..
            })
            | Shape::Object
            | Shape::List => written,
            Shape::Leaf(_) | Shape::Open => {
                if self.past_the_bound(&written) {
                    self.carried.push_str(bytes);
                    self.value.clear();
                    self.refused = true;
                    return;
                }
                let mut fragment = self.prefix();
                fragment.push_str(&written);
                fragment
            }
        };
        self.open.pop();
        self.value.clear();
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(bytes);
        self.push_fragment(fragment, Text::uncounted(source), out);
        self.released = false;
    }

    /// Whether `written`, the value of the innermost element, would nest the arguments past
    /// [`DEEPEST`]: the containers open above the element (the arguments object and the element
    /// containers) leave it the rest, and only a list or an object the value spells can use it
    /// up. The reading is the prefix parser's own, at the depth left, so the two agree exactly.
    fn past_the_bound(&self, written: &str) -> bool {
        if !written.starts_with(['[', '{']) {
            return false;
        }
        let left = DEEPEST.saturating_sub(self.open.len());
        matches!(
            PartialJson::new(left, false).parse(written, false),
            Err(PartialJsonError::TooDeep(_))
        )
    }

    /// Refuses the carried bytes with `why`, whitespace included ([`refuse`](super::refuse)): the
    /// failure counted by a `Malformed` event of no bytes, the bytes dropped as a malformed call's,
    /// never shown; released, so the whitespace after them is the model's.
    fn report(&mut self, why: MalformedReason, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        super::refuse(std::mem::take(&mut self.carried), why, out);
        self.released = true;
    }

    /// `text` the syntax has no place for (a word between tags, a tag out of its place), refused
    /// with `why` ([`Self::report`]) under the whitespace rule of text between tags: the whitespace
    /// carried before it is the template's after a tag the syntax read, and is dropped on its own
    /// as the wrapper's, so the refused bytes are the model's text alone; after text refused since
    /// that tag, the whitespace is the model's and leaves with this text.
    fn refuse_between(&mut self, text: &str, why: MalformedReason, out: &mut Events) {
        if !self.released && !self.carried.is_empty() {
            out.push(Event::Dropped {
                text: Text::uncounted(std::mem::take(&mut self.carried)),
                why: DropReason::Wrapper,
            });
        }
        self.carried.push_str(text);
        self.report(why, out);
    }

    /// `}` or `{}`, with `source` as its bytes.
    fn close_object<'d>(&mut self, source: Text<'d>, out: &mut Events<'d>) {
        let json = if self.written == 0 { "{}" } else { "}" }.to_string();
        self.push_fragment(json, source, out);
    }

    fn push_fragment<'d>(&self, json: String, source: Text<'d>, out: &mut Events<'d>) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out);
        text.len()
    }

    fn ignored(&mut self, _text: &str, declared: &Declared, out: &mut Events) {
        self.ignored(declared, out);
    }

    /// Only the invoke's closing tag is a started call's end; the next invoke's opening and the
    /// block's close end the call too, as the region's. An invoke that named no call takes
    /// whichever terminal ended it, so that it is refused the same way however it ended.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool {
        if terminal == INVOKE_CLOSE || !self.started() {
            (*self).close(terminal, out);
            true
        } else {
            (*self).close("", out);
            false
        }
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }

    fn stop(self: Box<Self>, as_marker: bool, out: &mut Events) {
        (*self).stop(as_marker, out);
    }
}
