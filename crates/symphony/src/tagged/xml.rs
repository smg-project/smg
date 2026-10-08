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
//! What the syntax says, and what the assembler does with it:
//!
//! - **An element is a key, and its children say what it holds.** Text makes it a leaf; a child
//!   tag makes it a container: a list when the tool declares an array there or, failing a
//!   declaration, when the child is an `<item>`, else an object (a list's elements are `<item>`
//!   elements, but so is a member named `item`, which is why the declaration decides first); no
//!   child at all makes it empty. A tag under an element the tool declares a string or an
//!   integer is the leaf's text (a file's `<template>`, an e-mail's `<p>`), and with no
//!   declaration a tag opens a child only after the template's separator, since the template
//!   writes the separator before every tag it writes and never inside a value. A list's items
//!   and an object's members are elements in turn, so the tree nests as deep as the arguments
//!   do. Whitespace right after an opening tag is held until the element's next byte says what
//!   it is: a leaf's text when text or the closing tag follows (a value that is one space is one
//!   space), the template's when a child tag follows.
//! - **A value's type comes from the request's tools at every depth** ([`Declared::kind_at`]:
//!   below an object a key selects a property, below an array any element selects its items). A
//!   declared string is its text, streamed as it arrives; every other leaf is written whole at
//!   its closing tag through [`json`], which reads the booleans and numbers the template writes
//!   as text and infers the rest. An empty element is `[]` or `{}` when the tool declares an
//!   array or an object there, `null` when nothing is declared and it is a list's item, and `""`
//!   everywhere else: the template writes `None` inside a list as an empty element, skips a
//!   mapping's `None` member, and leaves a top-level argument that is `None` out altogether.
//!   Nothing is taken from a value: the template writes it directly between the tags, with no
//!   escaping.
//! - **A top-level null is never written.** The template leaves an argument whose value is `None`
//!   out (`for k, v in arguments.items() if v is not none`), so no output carries one; the
//!   parity test counts that as a corpus class.
//! - **A tag is `<`, a name and `>`**, the name any run of characters without whitespace or angle
//!   brackets, with a leading `/` for a closing tag. A `<` followed by anything else is
//!   text, so a comparison in a value stays text; a `<` that does spell a tag is one, inside a
//!   value too, as every marker parser reads it. Inside a leaf, a tag other than the leaf's own
//!   closing tag is the leaf's text; a closing tag that matches nothing open is reported.
//! - **The name ends at its quote**, and the invoke tag at `>`; text between the two is reported,
//!   as for DSML. An invoke with no name starts no call, and everything in it is reported.
//! - **The separator keeps its place.** When nothing is held before it, it is `Dropped { Wrapper }`
//!   at once; when a value's text or an opening tag is held, its bytes go into the next
//!   fragment's source, so every event's bytes stay in the output's order.
//! - **Two endings.** [`Assembler::close`] is the invoke's closing tag ([`INVOKE_CLOSE`], the one
//!   terminal a started call's end carries), or the next invoke's opening or the block's close,
//!   which end the call with no bytes of their own: every open container is closed, a tag the
//!   end cut short is reported and an open streamed string gets its quote, a leaf never written
//!   and an element without a child come back as `Malformed`, and the arguments object is
//!   closed. An invoke that named no call takes whichever terminal ended it, so that it is
//!   reported the same way however it ended. [`Assembler::finish`] is a stream that was
//!   cut: nothing is closed, what was held comes back as `Malformed { UnterminatedRegion }`, and a
//!   call that started still ends, with no bytes of its own, as every assembler ends one.
//! - **Every byte of the invoke lands in exactly one event**, with the tagged assemblers'
//!   accounting: the name's bytes in `ToolCallStart`, each tag's and value's bytes in the
//!   fragments they produce, text where the syntax has tags as `Malformed` run by run with the
//!   whitespace before it as `Dropped { Wrapper }`.
//!
//! This is the fourth tagged assembler. It shares the value module and the escaping with the
//! other three; the tree is its own.
//!
//! [`formats::minimax_m3`]: crate::formats::minimax_m3()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::{json, Declared, Kind, ITEM, NULL_WORDS},
    },
};

/// The invoke's closing tag: the one terminal a started call's end carries. The next invoke's
/// opening and the block's close end a call too, but they belong to the region, and the engine
/// drops them.
pub const INVOKE_CLOSE: &str = "</invoke>";

const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const TAG_TAIL: &str = "text after a tag's name";
const EMPTY_NAME: &str = "an invoke with no name";
const WITHOUT_A_NAME: &str = "an invoke block without a name";

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

    /// The table's ignored terminal arrived inside the call: `Dropped { Wrapper }` at once when
    /// nothing is held before it, otherwise carried into the next fragment's source, so that its
    /// bytes keep their place.
    pub fn ignored(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        self.release_tag(declared, out);
        if self.carried.is_empty() {
            out.push(Event::Dropped {
                text: Text::uncounted(bytes),
                why: DropReason::Wrapper,
            });
        } else {
            self.carried.push_str(bytes);
        }
        self.after_separator = true;
    }

    /// The call's end: `terminal` is the closing marker the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. Every open container is
    /// closed, an open string gets its quote, a leaf never written is reported, the object is
    /// closed; an invoke whose name never closed, or that had none, has its bytes reported.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        self.hold_tag();
        if self.stage != Stage::Body || !self.started() {
            self.carried.push_str(terminal);
            self.report(WITHOUT_A_NAME, out);
            return;
        }
        while let Some(element) = self.open.last() {
            match element.shape {
                Shape::Leaf(Leaf {
                    mode: Mode::Streaming,
                    ..
                }) => {
                    // A tag the end cut short is reported, not hidden in the quote's source.
                    self.report(TAG_CUT_SHORT, out);
                    self.push_fragment("\"".to_string(), Text::default(), out);
                }
                Shape::Object => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.push_fragment("}".to_string(), source, out);
                }
                Shape::List => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.push_fragment("]".to_string(), source, out);
                }
                // A value never written, or an element the end cut before its first child: its
                // bytes are reported, and its member is never written.
                Shape::Leaf(_) | Shape::Open => {
                    self.value.clear();
                    self.report(TAG_CUT_SHORT, out);
                }
            }
            self.open.pop();
        }
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.close_object(source, out);
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(terminal),
        });
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A streamed string's open fragment stays open; everything held comes back as
    /// `Malformed { UnterminatedRegion }`; a call that started still ends, with no bytes of its
    /// own, as every assembler ends one.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.hold_tag();
        if !self.carried.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(std::mem::take(&mut self.carried)),
                why: MalformedReason::UnterminatedRegion,
            });
        }
        if self.started() {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
        }
    }

    /// Reads as much of `text` as one step takes and returns the rest.
    fn take<'a>(&mut self, text: &'a str, declared: &Declared, out: &mut Events) -> &'a str {
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
    /// source; a tail between the quote and `>` is reported; the `>` is carried into the next
    /// source. An invoke with no name starts no call, and the tag's bytes are reported.
    fn start_call(&mut self, out: &mut Events) {
        let name = std::mem::take(&mut self.value);
        self.stage = Stage::Body;
        // The tail is every byte after the name's quote, a separator that arrived meanwhile
        // included; `name_end` is a character boundary, so the split never lands inside one.
        let tail_bytes = self.carried.split_off(self.name_end);
        if name.is_empty() {
            self.carried.push_str(&tail_bytes);
            self.carried.push('>');
            self.report(EMPTY_NAME, out);
            return;
        }
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
        self.function = Some(name);
        if !tail_bytes.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(tail_bytes),
                why: MalformedReason::Other(TAG_TAIL.to_string()),
            });
        }
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
            self.drop_carried(out);
            out.push(Event::Malformed {
                text: Text::uncounted(bytes),
                why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
            });
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
    /// of its own.
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
    }

    /// A closing tag: it closes the innermost element when it names it; inside a leaf, or under
    /// an element with no child yet, any other closing tag is text; elsewhere it is reported.
    fn close_tag(&mut self, name: &str, bytes: &str, declared: &Declared, out: &mut Events) {
        match self.open.last() {
            Some(element) if element.key == name => self.close_element(bytes, declared, out),
            Some(Element {
                shape: Shape::Leaf(_) | Shape::Open,
                ..
            }) => self.text(bytes, declared, out),
            Some(_) | None => {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(bytes),
                    why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
                });
            }
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

    /// Text where the syntax has tags: whitespace is the template's, anything else is reported.
    fn text_between(&mut self, text: &str, out: &mut Events) {
        let mut rest = text;
        while let Some(first) = rest.chars().next() {
            let space = first.is_whitespace();
            let length = rest
                .char_indices()
                .find(|(_, c)| c.is_whitespace() != space)
                .map_or(rest.len(), |(at, _)| at);
            if space {
                self.carried.push_str(&rest[..length]);
            } else {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(&rest[..length]),
                    why: MalformedReason::Other(TEXT_BETWEEN_TAGS.to_string()),
                });
            }
            rest = &rest[length..];
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
    /// declared type.
    fn close_element(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        let Some(shape) = self.open.last().map(|element| element.shape) else {
            return;
        };
        let fragment = match shape {
            Shape::Leaf(Leaf {
                mode: Mode::Streaming,
                ..
            }) => "\"".to_string(),
            Shape::Leaf(Leaf { kind, .. }) => {
                let written = json(&self.value, kind);
                let mut fragment = self.prefix();
                fragment.push_str(&written);
                fragment
            }
            Shape::Object => "}".to_string(),
            Shape::List => "]".to_string(),
            Shape::Open if !self.value.is_empty() => {
                // Whitespace alone between the tags: a leaf after all.
                let written = json(&self.value, self.kind_here(declared));
                let mut fragment = self.prefix();
                fragment.push_str(&written);
                fragment
            }
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
                let empty = match self.kind_here(declared) {
                    Some(Kind::Array) => "[]",
                    Some(Kind::Object) => "{}",
                    Some(_) => "\"\"",
                    None if in_list => "null",
                    None => "\"\"",
                };
                let mut fragment = self.prefix();
                fragment.push_str(empty);
                fragment
            }
        };
        self.open.pop();
        self.value.clear();
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(bytes);
        self.push_fragment(fragment, Text::uncounted(source), out);
    }

    /// Reports the carried bytes as `Malformed` with `why`, whitespace included.
    fn report(&mut self, why: &str, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: MalformedReason::Other(why.to_string()),
        });
    }

    /// Whitespace carried before text that is reported: the template's, dropped so the report
    /// holds only the text.
    fn drop_carried(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: DropReason::Wrapper,
        });
    }

    /// `}` or `{}`, with `source` as its bytes.
    fn close_object(&mut self, source: Text, out: &mut Events) {
        let json = if self.written == 0 { "{}" } else { "}" }.to_string();
        self.push_fragment(json, source, out);
    }

    fn push_fragment(&self, json: String, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}
