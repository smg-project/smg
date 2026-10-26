//! Parameters named by an attribute: the events of one `<invoke name="NAME">` block whose
//! arguments are `<parameter name="KEY">VALUE</parameter>` tags, typed by the request's tools or,
//! in DeepSeek's DSML and Kimi K3's XTML, by an attribute on the tag.
//!
//! MiniMax M2 writes a call as
//!
//! ```text
//! <invoke name="get_weather">
//! <parameter name="city">Paris</parameter>
//! <parameter name="days">3</parameter>
//! </invoke>
//! ```
//!
//! inside a `<minimax:tool_call>` block that holds one invoke or several, each tag on a line of
//! its own. MiniCPM5 writes the same shape with its own tags and no block,
//!
//! ```text
//! <function name="weather"><param name="city">Paris</param><param name="days">3</param></function>
//! ```
//!
//! one call after another on lines of their own; dots3 writes M2's tags with a block per call and
//! every value on a line of its own; DeepSeek V4 and V4.1 write M2's shape in DSML's spelling,
//!
//! ```text
//! <｜DSML｜ invoke name="get_weather">
//! <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
//! <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! ```
//!
//! inside a `<｜DSML｜ calls>` block; Kimi K3 writes it in XTML, every tag an `<|open|>` or
//! `<|close|>` token, a word and the `<|sep|>` token,
//!
//! ```text
//! <|open|>call tool="get_weather" index="1"<|sep|>
//! <|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>
//! <|open|>argument key="days" type="number"<|sep|>3<|close|>argument<|sep|>
//! <|close|>call<|sep|>
//! ```
//!
//! without the newlines, which stand here for the reader. The engine's table owns the block's and
//! the invoke's markers ([`formats::minimax_m2`], [`formats::minicpm5`], [`formats::dots3`],
//! [`formats::deepseek_v4_1`], [`formats::kimi_k3`]): it enters this assembler at `<invoke name="`
//! (`<function name="`, `<|open|>call tool="`) and leaves it at the invoke's closing tag, so the
//! assembler starts inside the function's name and never sees that tag; the engine hands it the
//! tag's bytes as the call's end. Between those two the assembler reads the parameter tags
//! itself, in the family's spelling ([`Tags`]).
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.
//!
//! [`formats::minimax_m2`]: crate::formats::minimax_m2()
//! [`formats::minicpm5`]: crate::formats::minicpm5()
//! [`formats::dots3`]: crate::formats::dots3()
//! [`formats::deepseek_v4_1`]: crate::formats::deepseek_v4_1()
//! [`formats::kimi_k3`]: crate::formats::kimi_k3()

use super::object::{Mode, Object};
use crate::{
    event::{runs, DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        end_cut_call, held_text,
        value::{cut as cut_json, json, Kind},
    },
    Declared,
};

/// A CDATA section's opener and closer, where the family's template writes one around a value.
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

const OPEN: usize = 0;
const CLOSE: usize = 1;
/// The marker that ends a tag's tail, where the family has one ([`TagEnd::Marker`]).
const END: usize = 2;
/// The two tails a DSML parameter tag has after the name's quote: the attribute that types it.
const STRING_TRUE: &str = " string=\"true\"";
const STRING_FALSE: &str = " string=\"false\"";
/// The types an XTML `type` attribute may name; `string` streams, the rest are written whole.
const TYPES: [&str; 6] = ["string", "number", "boolean", "null", "array", "object"];

/// Where a value's type comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Typing {
    /// The request's tools: a declared string streams, every other value is written whole at the
    /// close in the type the tool declares.
    Declared,
    /// A `string="true"` or `string="false"` attribute after the parameter's name: the model says
    /// whether the value is its text or JSON.
    StringAttribute,
    /// A `type="…"` attribute after the parameter's name, one of `string`, `number`, `boolean`,
    /// `null`, `array` and `object`: the model says the type, and only `string` streams.
    TypeAttribute,
}

/// What ends a call tag's or a parameter tag's tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagEnd {
    /// The `>` character, as in `<invoke name="NAME">`.
    Angle,
    /// A marker of its own, which may arrive in pieces: XTML's `<|sep|>`.
    Marker(&'static str),
}

impl TagEnd {
    /// The bytes that end the tail.
    fn text(self) -> &'static str {
        match self {
            Self::Angle => ">",
            Self::Marker(marker) => marker,
        }
    }
}

/// How a family spells the syntax: the parameter tag's opening up to the name's quote and its
/// closing tag, the invoke's closing tag, where a value's type comes from, and how the template
/// frames the tags. The invoke's opening is the table's terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tags {
    pub parameter_open: &'static str,
    pub parameter_close: &'static str,
    /// The invoke's closing tag: the one terminal the call's end carries. The next invoke's
    /// opening and a block's close end a call too, but they belong to the region, and the engine
    /// drops them.
    pub invoke_close: &'static str,
    /// Whether the template wraps a value as `<![CDATA[…]]>` when it holds `<`, `&` or a newline.
    pub cdata: bool,
    /// Whether the template writes a newline after the parameter tag and one before its closing
    /// tag, so a value stands on a line of its own.
    pub value_on_its_own_line: bool,
    /// Whether the template writes each tag on a line of its own: a newline after the invoke's
    /// opening tag, after each parameter's closing tag and after the invoke's closing tag. The
    /// grammar writes them; the assembler reads the newlines between tags as the template's.
    pub tags_on_their_own_lines: bool,
    /// Whether the template opens a block around each call rather than around all of a turn's
    /// calls. The grammar writes the block the table has that way; the assembler reads neither.
    pub block_per_call: bool,
    /// Where a value's type comes from.
    pub typing: Typing,
    /// What the template writes before the opener of a block around a turn's calls: a newline, or
    /// DSML's two. The grammar writes it; the assembler never sees it.
    pub before_block: &'static str,
    /// What ends a tag's tail.
    pub tag_end: TagEnd,
    /// Whether a call tag's tail may be the model's count of the turn's calls, ` index="N"`.
    pub indexed_calls: bool,
    /// The special tokens the family's tags are built of, where they are (XTML's `<|open|>` and
    /// `<|close|>`; the family's table declares them to the engine from here, `kimi_k3()`): a tail
    /// the scanner holds at the model's own stop that begins with one is a tag the end cut short,
    /// never the model's text, and is dropped as the engine drops its own terminals' held
    /// prefixes. Empty where the tags are text the model can write (`</parameter>`).
    pub structural_tokens: &'static [&'static str],
}

impl Tags {
    /// MiniMax M2's: `<parameter name="…">`, `</parameter>` and `</invoke>`, no CDATA.
    pub const MINIMAX_M2: Self = Self {
        parameter_open: "<parameter name=\"",
        parameter_close: "</parameter>",
        invoke_close: "</invoke>",
        cdata: false,
        value_on_its_own_line: false,
        tags_on_their_own_lines: true,
        block_per_call: false,
        typing: Typing::Declared,
        before_block: "\n",
        tag_end: TagEnd::Angle,
        indexed_calls: false,
        structural_tokens: &[],
    };
    /// dots3's: MiniMax M2's tags, a block around each call, every value on a line of its own.
    pub const DOTS3: Self = Self {
        parameter_open: "<parameter name=\"",
        parameter_close: "</parameter>",
        invoke_close: "</invoke>",
        cdata: false,
        value_on_its_own_line: true,
        tags_on_their_own_lines: true,
        block_per_call: true,
        typing: Typing::Declared,
        before_block: "\n",
        tag_end: TagEnd::Angle,
        indexed_calls: false,
        structural_tokens: &[],
    };
    /// MiniCPM5's: `<param name="…">`, `</param>` and `</function>`, a value a CDATA section when
    /// it holds `<`, `&` or a newline.
    pub const MINICPM5: Self = Self {
        parameter_open: "<param name=\"",
        parameter_close: "</param>",
        invoke_close: "</function>",
        cdata: true,
        value_on_its_own_line: false,
        tags_on_their_own_lines: false,
        block_per_call: false,
        typing: Typing::Declared,
        before_block: "\n",
        tag_end: TagEnd::Angle,
        indexed_calls: false,
        structural_tokens: &[],
    };
    /// DeepSeek's DSML: `<｜DSML｜ parameter name="…" string="…">`, `</｜DSML｜ parameter>`
    /// and `</｜DSML｜ invoke>`, each tag on a line of its own, the value typed by its `string`
    /// attribute, the calls block opened after the template's two newlines.
    pub const DSML: Self = Self {
        parameter_open: "<｜DSML｜ parameter name=\"",
        parameter_close: "</｜DSML｜ parameter>",
        invoke_close: "</｜DSML｜ invoke>",
        cdata: false,
        value_on_its_own_line: false,
        tags_on_their_own_lines: true,
        block_per_call: false,
        typing: Typing::StringAttribute,
        before_block: "\n\n",
        tag_end: TagEnd::Angle,
        indexed_calls: false,
        structural_tokens: &[],
    };
    /// DeepSeek's DSML as V4 spells it, before V4.1 put a space after the sentinel:
    /// `<｜DSML｜parameter name="…" string="…">`, `</｜DSML｜parameter>` and `</｜DSML｜invoke>`,
    /// otherwise [`Self::DSML`]: each tag on a line of its own, the value typed by its `string`
    /// attribute, the calls block opened after the template's two newlines.
    pub const DSML_V4: Self = Self {
        parameter_open: "<｜DSML｜parameter name=\"",
        parameter_close: "</｜DSML｜parameter>",
        invoke_close: "</｜DSML｜invoke>",
        cdata: false,
        value_on_its_own_line: false,
        tags_on_their_own_lines: true,
        block_per_call: false,
        typing: Typing::StringAttribute,
        before_block: "\n\n",
        tag_end: TagEnd::Angle,
        indexed_calls: false,
        structural_tokens: &[],
    };
    /// Kimi K3's XTML: `<|open|>argument key="…" type="…"<|sep|>`, `<|close|>argument<|sep|>` and
    /// `<|close|>call<|sep|>`, nothing between the tags, the value typed by its `type` attribute,
    /// the call tag carrying the model's index. The grammar is the `xtml` module's own.
    pub const XTML: Self = Self {
        parameter_open: "<|open|>argument key=\"",
        parameter_close: "<|close|>argument<|sep|>",
        invoke_close: "<|close|>call<|sep|>",
        cdata: false,
        value_on_its_own_line: false,
        tags_on_their_own_lines: false,
        block_per_call: false,
        typing: Typing::TypeAttribute,
        before_block: "",
        tag_end: TagEnd::Marker("<|sep|>"),
        indexed_calls: true,
        structural_tokens: &["<|open|>", "<|close|>"],
    };

    /// Whether `tail`, held by the scanner at the model's own stop, is a tag the end cut short:
    /// it begins with one of the family's structural tokens, which the model cannot have written
    /// as text.
    fn is_cut_tag(&self, tail: &str) -> bool {
        !tail.is_empty()
            && self
                .structural_tokens
                .iter()
                .any(|token| tail.starts_with(token))
    }

    /// The markers the scanner reads inside an invoke: the two parameter tags, and the tail's end
    /// where it is a marker.
    fn markers(&self) -> Vec<&'static str> {
        let mut markers = vec![self.parameter_open, self.parameter_close];
        if let TagEnd::Marker(marker) = self.tag_end {
            markers.push(marker);
        }
        markers
    }

    fn text(&self, tag: usize) -> &'static str {
        match tag {
            OPEN => self.parameter_open,
            CLOSE => self.parameter_close,
            _ => self.tag_end.text(),
        }
    }
}

/// The events of one invoke, from the bytes after `<invoke name="`.
#[derive(Clone, Debug)]
pub struct Assembler {
    id: String,
    tags: Tags,
    scanner: Scanner,
    /// The arguments object as it is written, with the bytes carried for the next event's source.
    object: Object,
    stage: Stage,
    /// The function's name once its tag is whole, which is when `ToolCallStart` was pushed.
    function: Option<String>,
    /// Whether text between tags was refused since the last tag the syntax read: a word where
    /// the syntax has tags, a tag out of its place, a tag refused with its bytes. The
    /// whitespace carried after it is the model's, between two pieces of its text, and leaves with
    /// the next piece; before any, the whitespace after a tag is the template's, dropped before a
    /// report or carried into the next event's source.
    released: bool,
    done: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Inside the invoke tag: the name is `carried[..]` until `"`, then the tail until `>`.
    FunctionName,
    /// After the name's quote, before the tag's `>`: the tail is `carried[start..]`.
    FunctionTail { name: String, start: usize },
    /// After the invoke tag or a `</parameter>`.
    Between,
    /// Inside a parameter tag: the key is `carried[start..]` until `"`.
    ParameterName { start: usize },
    /// After the key's quote, before the tag's `>`: the tail, `carried[start..]`, which the
    /// syntax has empty.
    ParameterTail { key: String, start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Where the value's text starts in `carried`, while it is held.
    start: usize,
    /// Where it ends, once a CDATA section closed; `carried` holds the closer after it.
    end: Option<usize>,
    /// Whether the member left before the closing tag: a value held whole whose section closed
    /// is written at the first word after the closer ([`Assembler::write_held_value`]), so the
    /// closing tag has nothing left to write.
    written: bool,
    mode: Mode,
    cdata: Cdata,
    /// Inside a CDATA section, the trailing bytes that may be the start of its closer; for a value
    /// on its own line, a trailing newline that may be the template's framing: held until the next
    /// bytes or the closing tag say.
    held: String,
    /// For a value on its own line: whether its first byte, the template's newline, is still to
    /// come.
    leading_newline_due: bool,
}

/// Whether a value is a CDATA section, decided from its first bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cdata {
    /// The family writes no CDATA sections, or this value's first bytes are not an opener.
    Plain,
    /// Fewer bytes than the opener has, all of them the opener's so far.
    Undecided,
    /// Inside the section: the text up to `]]>` is the value's.
    Inside,
    /// The section closed: only the closing tag should follow.
    Done,
}

impl Assembler {
    /// An assembler for the invoke at `index` with the id the format minted for it, positioned
    /// just after the invoke tag's `name="`.
    pub fn new(index: u32, id: impl Into<String>, tags: Tags) -> Self {
        Self {
            id: id.into(),
            tags,
            scanner: Scanner::new(tags.markers()),
            object: Object::new(index),
            stage: Stage::FunctionName,
            function: None,
            released: false,
            done: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// The invoke's closing tag in the family's spelling: the one terminal a started call's end
    /// carries.
    pub fn invoke_close(&self) -> &'static str {
        self.tags.invoke_close
    }

    /// Append the next bytes of the invoke and push the events they complete. Every byte is the
    /// invoke's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        let scanned = self.scanner.feed(bytes);
        for piece in scanned.pieces() {
            self.take(piece, declared, out);
        }
        self.scanner.reclaim(scanned);
    }

    /// The invoke's end: `terminal` is the closing tag the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. After the last parameter the
    /// object is closed and the call ends; inside a value held for its close what was held is
    /// refused (the counted event of no bytes, the bytes as `Dropped { MalformedCall }`) and the
    /// object closes without that member; inside a value whose member was already written at its
    /// section's close nothing is reported, and the object closes after that member with the
    /// whitespace carried since the section as the close's source; inside a streamed string the
    /// string is closed first ([`Self::close_open_string`]), since the model ended the call and
    /// its bytes are the value whole, under one `Malformed { OpenAtTheMarker }` event of no bytes;
    /// an invoke that never named a function is refused whole, the terminal included.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        let cut_tag = self.close_open_string(MalformedReason::OpenAtTheMarker, out);
        // The value's held bytes (a trailing newline that may be the template's framing) came
        // before the start of a tag the scanner holds: they leave in that order.
        let held = self.take_held();
        self.object.carried.push_str(&held);
        let held = self.scanner.held().to_string();
        self.object.carried.push_str(&held);
        if !self.started() {
            let why = if self.object.carried.is_empty() && terminal.is_empty() {
                None
            } else if matches!(self.stage, Stage::FunctionName) && self.object.carried.is_empty() {
                Some(MalformedReason::EmptyName)
            } else {
                Some(MalformedReason::ClosedBeforeTheName)
            };
            self.object.carried.push_str(terminal);
            if let Some(why) = why {
                self.object.refuse(why, out);
            }
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between => {}
            // A value whose member left at its section's close: the whitespace carried after the
            // text that followed is the close's source, as between tags.
            Stage::Value(ValueState { written: true, .. }) => {}
            // A name, a tag or a value held for its close that the end cut short: its bytes are
            // refused, its member is never written, and the object closes.
            _ => self.object.refuse(MalformedReason::TagCutShort, out),
        }
        let source = Text::uncounted(std::mem::take(&mut self.object.carried));
        self.object.close(source, out);
        // The tag the marker cut short came before the marker: it leaves before the call's end,
        // which carries the marker's bytes, so every byte keeps its place.
        if let Some(tail) = cut_tag {
            out.push(Event::Dropped {
                text: Text::uncounted(tail),
                why: DropReason::Wrapper,
            });
        }
        out.push(Event::ToolCallEnd {
            index: self.object.index,
            source: Text::uncounted(terminal.to_owned()),
        });
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client, and the call's bytes stay the call's. A streamed string's open fragment stays
    /// open; a value held for its close leaves as the call's own fragment, its key and its bytes
    /// as the model wrote them (`, "prefs": {"size": "large", "temp`), since a client reads the
    /// prefix as written under `length`, never the call's bytes as the answer's prose, and a
    /// section that closed is written whole; a string whose key came and no byte opens as its
    /// first byte would have opened it; a parameter tag the end cut short, the start of a closing
    /// tag held behind the value and the template's whitespace between tags leave as the syntax's
    /// wrapping, as the stop drops a cut tag ([`Self::cut_value`]). A call that started still
    /// ends, with no bytes of its own, as every assembler ends one, after the one `Malformed {
    /// UnterminatedRegion }` event of no bytes that says that its arguments are no whole value
    /// ([`end_cut_call`]); what a block that
    /// named no call held is refused as `UnterminatedRegion`, as the keyed assembler refuses it,
    /// so that the reader's release of a refused output that made no call takes the whole slice
    /// (a `Malformed` event with bytes would end that release).
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        if !self.started() {
            let held = self.take_held();
            self.object.carried.push_str(&held);
            let held = self.scanner.held().to_string();
            self.object.carried.push_str(&held);
            self.object.refuse(MalformedReason::UnterminatedRegion, out);
            return;
        }
        let cut_tag = if matches!(self.stage, Stage::Value(_)) {
            self.cut_value(out)
        } else {
            // Between tags, or inside a tag the end cut short: the whitespace carried and the
            // tag's bytes are the syntax's, never the model's text.
            let tail = self.scanner.set_aside();
            self.object.carried.push_str(&tail);
            self.object.drop_carried(out);
            None
        };
        end_cut_call(self.object.index, String::new(), "", out);
        if let Some(tail) = cut_tag {
            out.push(Event::Dropped {
                text: Text::uncounted(tail),
                why: DropReason::Wrapper,
            });
        }
        self.done = true;
    }

    /// The value the end of the stream cut ([`Self::finish`]), written as far as the model wrote
    /// it. The bytes the value held (the start of a CDATA closer, the opener's first bytes while
    /// the section was undecided) are its text; the start of the closing tag the scanner held is
    /// a tag the end cut short, never the value's text, in the families whose tags are text (a
    /// cut inside a string value's closing tag used to deliver the tag's bytes as the value's,
    /// `"KRW</｜DSML｜ parameter`; the value is `"KRW`), returned to be dropped after the call's
    /// end so that every byte keeps its place, as is a held tail that begins with a structural
    /// token or follows a closed section; a held start of the opening tag is the value's text,
    /// as the whole tag is inside a value, and so is a held strict prefix of a special token
    /// where the tags are built of them, which the model cannot write and no cut splits; a
    /// newline held as the template's framing before a
    /// closing tag that never came stays the template's. Then by the value's mode: a streamed
    /// string's fragments stand; a string whose key came and no byte opens, `{"key": "`; a value
    /// held for its close is written as one fragment, its key and its bytes as the JSON they
    /// stand for so far ([`cut_json`]: as written where they are a strict prefix of a JSON value,
    /// `{"size": "large", "temp`, `12`, `nu`, a Python literal the cut found complete as the JSON
    /// the close makes of it, `True` as `true`, and the opening of a JSON string holding them
    /// where they are neither, `"None]`, `"<![CD`, so that every fragment keeps the arguments a
    /// strict JSON prefix, which is the crate's contract), or whole where its section closed, with
    /// the bytes carried as its source. Whatever is still carried after that (the framing's
    /// newline, the whitespace after a member written at its section's close) is the syntax's
    /// wrapping.
    fn cut_value(&mut self, out: &mut Events) -> Option<String> {
        let Stage::Value(value) = &mut self.stage else {
            return None;
        };
        if value.cdata == Cdata::Undecided {
            // The opener's first bytes, or fewer: no section opens now, so they are the value's.
            value.cdata = Cdata::Plain;
            let arrived = self.object.carried.split_off(value.start);
            self.section_text(&arrived, out);
        }
        let Stage::Value(value) = &self.stage else {
            return None;
        };
        let section_closed = value.cdata == Cdata::Done;
        let mut text = self.take_held();
        // The start of the closing tag the scanner holds is a tag the end cut short, never the
        // value's text, in the families whose tags are text: the limit fell while the model was
        // writing, and the start of the closing tag at the end of its output is that tag (`KRW`
        // and then `</`, up to the tag's last byte). It leaves after the call's end, as does a
        // tail that begins with a structural token or follows a closed section. A held start of
        // the opening tag (`see <p`) is the value's text, as the whole tag is inside a value: the
        // cut is judged as the whole tag would be; and where the tags are special tokens (XTML)
        // a held strict prefix of one (`x <|`) can only be the model's text, since the model
        // cannot write the token and no cut splits it.
        let tail = self.scanner.set_aside();
        let mut cut_tag = None;
        if !tail.is_empty() {
            if section_closed
                || self.tags.is_cut_tag(&tail)
                || (self.tags.structural_tokens.is_empty()
                    && self.tags.parameter_close.starts_with(tail.as_str()))
            {
                cut_tag = Some(tail);
            } else {
                text.push_str(&tail);
            }
        }
        if self.tags.value_on_its_own_line && text == "\n" {
            if let Stage::Value(value) = &mut self.stage {
                if matches!(value.mode, Mode::Whole | Mode::Undecided) && value.end.is_none() {
                    value.end = Some(self.object.carried.len());
                }
            }
            self.object.carried.push('\n');
        } else {
            self.value_bytes(&text, out);
        }
        let Stage::Value(value) = &mut self.stage else {
            return cut_tag;
        };
        if value.written {
            self.object.drop_carried(out);
            return cut_tag;
        }
        let fragment = match value.mode {
            Mode::Streaming { opened: true } => None,
            Mode::Streaming { opened: false } => Some(self.object.member(&value.key, "\"")),
            Mode::Whole | Mode::Undecided => {
                let raw = held_text(&self.object.carried, value.start, value.end);
                let written = if section_closed {
                    json(raw, value.kind)
                } else {
                    cut_json(raw)
                };
                Some(self.object.member(&value.key, &written))
            }
        };
        value.written = true;
        match fragment {
            Some(fragment) => {
                let source = Text::uncounted(std::mem::take(&mut self.object.carried));
                self.object.push_fragment(fragment, source, out);
            }
            None => self.object.drop_carried(out),
        }
        cut_tag
    }

    /// The stream's stop inside the invoke: the model's own end of output, not a cut. The call
    /// ends as its closing tag would end it, with no bytes for the end ([`Self::close`]): after
    /// the last parameter the object is closed, a name, a tag or a value held for its close that
    /// the stop found open is refused and the object closes without that member, an invoke that
    /// never named a function is refused whole. A streamed string still open is closed
    /// first ([`Self::close_open_string`]), under the stop's word. One `Malformed { OpenAtTheStop }`
    /// event of no bytes stands before the fragment that closes what was open
    /// ([`open_at_the_stop`](super::open_at_the_stop)): for a string the stop closed under any
    /// table, and for the call itself where the table has no stop row (`as_marker`), since there
    /// the template writes the marker the model skipped.
    pub fn stop(mut self, as_marker: bool, out: &mut Events) {
        if self.done {
            return;
        }
        let string_open = matches!(
            self.stage,
            Stage::Value(ValueState {
                mode: Mode::Streaming { opened: true },
                ..
            })
        );
        if !string_open && self.started() && !as_marker {
            super::open_at_the_stop(out);
        }
        // The string closes first, under the stop's word; `close` finds it closed, and drops a
        // tag the end cut short after the call.
        let cut_tag = self.close_open_string(MalformedReason::OpenAtTheStop, out);
        self.close("", out);
        if let Some(tail) = cut_tag {
            out.push(Event::Dropped {
                text: Text::uncounted(tail),
                why: DropReason::Wrapper,
            });
        }
    }

    /// A streamed string still open when the call ends, at its closing tag's place (the invoke's
    /// close, the block's end, the next invoke's opener) or at the stream's stop: the model wrote
    /// nothing more of the value after the call's end, so its bytes are the value whole. The
    /// value's held bytes (the start of a CDATA closer that never completed, or the newline that
    /// may be the template's framing) came before the start of a tag the scanner holds: at the
    /// call's end no closer and no tag follows, so both are the value's text, in that order; a
    /// newline alone is the framing, the closing quote's source. After a CDATA section closed the
    /// value ended at its closer: the closer carried is the quote's source, and the start of a tag
    /// after it is the call's end's. A held tail that begins with one of the family's structural
    /// tokens is a tag the end cut short, never the model's text (the engine drops its own
    /// terminals' held prefixes the same way): it is handed back to leave in its place, dropped
    /// as the whole tag would have been: before the call's end at a marker, whose bytes came after
    /// it, and after the call at the stop, whose end carries no bytes, so every byte keeps its
    /// place. The string is closed: the
    /// one word of no bytes (`why`: the end that closed it, [`left_open`](super::left_open)), then
    /// the closing quote. Nothing happens when no string is open.
    fn close_open_string(&mut self, why: MalformedReason, out: &mut Events) -> Option<String> {
        if !matches!(
            self.stage,
            Stage::Value(ValueState {
                mode: Mode::Streaming { opened: true },
                ..
            })
        ) {
            return None;
        }
        let section_closed = matches!(
            &self.stage,
            Stage::Value(ValueState {
                cdata: Cdata::Done,
                ..
            })
        );
        let mut cut_tag = None;
        let mut text = self.take_held();
        if !section_closed {
            let tail = self.scanner.set_aside();
            if self.tags.is_cut_tag(&tail) {
                cut_tag = Some(tail);
            } else {
                text.push_str(&tail);
            }
        }
        if self.tags.value_on_its_own_line && text == "\n" {
            self.object.carried.push('\n');
        } else {
            self.value_bytes(&text, out);
        }
        super::left_open(why, out);
        self.object.close_member("\"".to_string(), "", out);
        self.stage = Stage::Between;
        cut_tag
    }

    fn take(&mut self, piece: Piece<'_>, declared: &Declared, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, declared, out),
            Piece::Marker(tag) => self.tag(tag, declared, out),
        }
    }

    fn tag(&mut self, tag: usize, declared: &Declared, out: &mut Events) {
        let bytes = self.tags.text(tag);
        match &self.stage {
            Stage::Between => self.tag_between(tag, out),
            // The tail of an invoke or parameter tag ends at the family's marker.
            Stage::FunctionTail { .. } | Stage::ParameterTail { .. } if tag == END => {
                self.end_tail(declared, out);
            }
            Stage::Value(_) if tag == CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.text(bytes, declared, out),
            // A tag cuts a name or a tag's tail short: the bytes so far are refused, and the tag
            // is read where they began.
            _ => {
                self.report(MalformedReason::TagCutShort, out);
                self.stage = Stage::Between;
                if self.started() {
                    self.tag_between(tag, out);
                } else {
                    self.release(bytes, MalformedReason::TagOutOfPlace, out);
                }
            }
        }
    }

    /// A tag between parameters: the parameter tag opens a key, once the function is named; the
    /// closing tag has no place, and neither does a parameter before the function's name closed.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = self.tags.text(tag);
        if tag == OPEN && self.started() {
            self.object.carried.push_str(bytes);
            self.stage = Stage::ParameterName {
                start: self.object.carried.len(),
            };
        } else {
            self.release(bytes, MalformedReason::TagOutOfPlace, out);
        }
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        match &self.stage {
            Stage::FunctionName => match text.split_once('"') {
                Some((head, rest)) => {
                    self.object.carried.push_str(head);
                    // The name's bytes stay carried: they are the start's source.
                    let name = self.object.carried.clone();
                    self.object.carried.push('"');
                    if name.is_empty() {
                        self.report(MalformedReason::EmptyName, out);
                        self.stage = Stage::Between;
                    } else {
                        self.stage = Stage::FunctionTail {
                            name,
                            start: self.object.carried.len(),
                        };
                    }
                    self.text(rest, declared, out);
                }
                None => self.object.carried.push_str(text),
            },
            // A tail that ends at a marker is carried whole; the marker arrives as a tag.
            Stage::FunctionTail { .. } | Stage::ParameterTail { .. }
                if matches!(self.tags.tag_end, TagEnd::Marker(_)) =>
            {
                self.object.carried.push_str(text);
            }
            Stage::FunctionTail { name, start } => {
                let (name, start) = (name.clone(), *start);
                match text.split_once('>') {
                    Some((head, rest)) => {
                        self.object.carried.push_str(head);
                        let tail = self.object.carried[start..].to_string();
                        self.object.carried.push('>');
                        self.start_call(name, &tail, out);
                        self.text(rest, declared, out);
                    }
                    None => self.object.carried.push_str(text),
                }
            }
            Stage::Between => self.text_between(text, out),
            Stage::ParameterName { start } => {
                let start = *start;
                match text.split_once('"') {
                    Some((head, rest)) => {
                        self.object.carried.push_str(head);
                        let key = self.object.carried[start..].to_string();
                        self.object.carried.push('"');
                        if key.is_empty() {
                            self.report(MalformedReason::EmptyName, out);
                            self.stage = Stage::Between;
                        } else {
                            self.stage = Stage::ParameterTail {
                                key,
                                start: self.object.carried.len(),
                            };
                        }
                        self.text(rest, declared, out);
                    }
                    None => self.object.carried.push_str(text),
                }
            }
            Stage::ParameterTail { key, start } => {
                let (key, start) = (key.clone(), *start);
                match text.split_once('>') {
                    Some((head, rest)) => {
                        self.object.carried.push_str(head);
                        let tail = self.object.carried[start..].to_string();
                        self.object.carried.push('>');
                        self.open_value(key, &tail, declared, out);
                        self.text(rest, declared, out);
                    }
                    None => self.object.carried.push_str(text),
                }
            }
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// The marker that ends an invoke tag's or a parameter tag's tail.
    fn end_tail(&mut self, declared: &Declared, out: &mut Events) {
        let marker = self.tags.tag_end.text();
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::FunctionTail { name, start } => {
                let tail = self.object.carried[start..].to_string();
                self.object.carried.push_str(marker);
                self.start_call(name, &tail, out);
            }
            Stage::ParameterTail { key, start } => {
                let tail = self.object.carried[start..].to_string();
                self.object.carried.push_str(marker);
                self.open_value(key, &tail, declared, out);
            }
            other => self.stage = other,
        }
    }

    /// Text where the syntax has tags: whitespace is carried, since whose it is depends on what
    /// follows ([`Self::release`]), anything else is released.
    fn text_between(&mut self, text: &str, out: &mut Events) {
        for (run, space) in runs(text) {
            if space {
                self.object.carried.push_str(run);
            } else {
                self.release(run, MalformedReason::TextBetweenTags, out);
            }
        }
    }

    /// `text` the syntax has no place for (a word between tags, a tag out of its place), refused
    /// ([`refuse`](super::refuse)): a `Malformed` event of no bytes with `why`, which a reader
    /// counts at its offset, and the bytes as `Dropped { MalformedCall }`, which no adapter shows.
    /// The bytes are the model's failed call, not its answer: the families' own parsers drop
    /// them, and a client shown them read the format's tags inside the message (a parameter tag
    /// written without its `<`, a call's arguments written as JSON and closed with a region of
    /// the model's own). The whitespace carried before the text is the template's after a tag the
    /// syntax read, and is dropped on its own so the refused bytes hold the model's text alone;
    /// after text refused already it is the model's, between two pieces of its text, and leaves
    /// with this one, so that every byte is accounted for however the bytes were cut.
    fn release(&mut self, text: &str, why: MalformedReason, out: &mut Events) {
        if !self.released {
            self.object.drop_carried(out);
        }
        self.object.carried.push_str(text);
        self.object.refuse(why, out);
        self.released = true;
    }

    /// The carried bytes refused with `why` ([`Object::refuse`]: the counted event of no bytes,
    /// then the bytes as `Dropped { MalformedCall }`), when there are any; released, so the
    /// whitespace after them is the model's.
    fn report(&mut self, why: MalformedReason, out: &mut Events) {
        if self.object.carried.is_empty() {
            return;
        }
        self.object.refuse(why, out);
        self.released = true;
    }

    /// The invoke tag is whole: the call starts. A tail after the name's quote is refused first,
    /// unless it is the model's index of the call and the family's tags carry one.
    fn start_call(&mut self, name: String, tail: &str, out: &mut Events) {
        self.function = Some(name.clone());
        if tail.is_empty() || (self.tags.indexed_calls && is_index(tail)) {
            out.push(Event::ToolCallStart {
                index: self.object.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(std::mem::take(&mut self.object.carried)),
            });
        } else {
            // The tag's bytes up to the tail stay as the start's source; the tail is refused, and
            // the tag's end stays carried.
            let end = self.tags.tag_end.text();
            let tag_end = self.object.carried.len() - tail.len() - end.len();
            let mut after = self.object.carried.split_off(tag_end);
            let close = after.split_off(after.len() - end.len());
            out.push(Event::ToolCallStart {
                index: self.object.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(std::mem::take(&mut self.object.carried)),
            });
            super::refuse(after, MalformedReason::TagTail, out);
            self.object.carried.push_str(&close);
        }
        self.stage = Stage::Between;
        self.released = false;
    }

    /// The parameter tag is whole: the value begins, typed by what the tool declares for the key,
    /// or by the tag's `string` attribute. A tag with any other tail after the name's quote
    /// (`<parameter name="x" type="int">`) is not one the syntax has: the whole tag is refused,
    /// and no parameter opens.
    fn open_value(&mut self, key: String, tail: &str, declared: &Declared, out: &mut Events) {
        let (kind, mode) = match (self.tags.typing, tail) {
            (Typing::Declared, "") => {
                let function = self.function.as_deref().unwrap_or_default();
                let kind = declared.kind(function, &key);
                let mode = match kind {
                    Some(Kind::String) => Mode::Streaming { opened: false },
                    Some(Kind::NullableString) => Mode::Undecided,
                    Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole,
                };
                (kind, mode)
            }
            (Typing::StringAttribute, STRING_TRUE) => (None, Mode::Streaming { opened: false }),
            (Typing::StringAttribute, STRING_FALSE) => (None, Mode::Whole),
            (Typing::TypeAttribute, tail) if typed(tail).is_some_and(|kind| kind == "string") => {
                (None, Mode::Streaming { opened: false })
            }
            (Typing::TypeAttribute, tail) if typed(tail).is_some() => (None, Mode::Whole),
            _ => {
                self.report(MalformedReason::TagTail, out);
                self.stage = Stage::Between;
                return;
            }
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            start: self.object.carried.len(),
            end: None,
            written: false,
            mode,
            cdata: if self.tags.cdata {
                Cdata::Undecided
            } else {
                Cdata::Plain
            },
            held: String::new(),
            leading_newline_due: self.tags.value_on_its_own_line,
        });
        self.released = false;
    }

    /// The next bytes of a value. Where the family writes CDATA sections, the first bytes decide
    /// whether this value is one: they are held until the opener's length has arrived or a byte
    /// rules the opener out, then read as they would have been.
    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        if value.leading_newline_due {
            // The newline after the parameter tag is the template's: into the source, not the
            // value. A value that starts otherwise has no framing to drop.
            value.leading_newline_due = false;
            if let Some(rest) = text.strip_prefix('\n') {
                self.object.carried.push('\n');
                value.start += 1;
                self.value_text(rest, out);
                return;
            }
        }
        match value.cdata {
            Cdata::Undecided => {
                self.object.carried.push_str(text);
                let so_far = &self.object.carried[value.start..];
                if so_far.len() < CDATA_OPEN.len() {
                    if CDATA_OPEN.starts_with(so_far) {
                        return;
                    }
                    value.cdata = Cdata::Plain;
                } else if so_far.starts_with(CDATA_OPEN) {
                    value.cdata = Cdata::Inside;
                    value.start += CDATA_OPEN.len();
                } else {
                    value.cdata = Cdata::Plain;
                }
                let arrived = self.object.carried.split_off(value.start);
                self.section_text(&arrived, out);
            }
            _ => self.section_text(text, out),
        }
    }

    /// The next bytes of a value whose CDATA question is settled: inside a section the text up to
    /// `]]>` is the value's and a trailing start of the closer is held; after the closer, text is
    /// out of place; a plain value's bytes are the value's.
    fn section_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        match value.cdata {
            Cdata::Inside => {
                let mut pending = std::mem::take(&mut value.held);
                pending.push_str(text);
                if let Some(at) = pending.find(CDATA_CLOSE) {
                    let inside = pending[..at].to_string();
                    let after = pending[at + CDATA_CLOSE.len()..].to_string();
                    self.value_bytes(&inside, out);
                    if let Stage::Value(value) = &mut self.stage {
                        value.cdata = Cdata::Done;
                        if matches!(value.mode, Mode::Whole | Mode::Undecided) {
                            value.end = Some(self.object.carried.len());
                        }
                    }
                    self.object.carried.push_str(CDATA_CLOSE);
                    self.section_text(&after, out);
                } else {
                    // A trailing `]]` or `]` may be the closer's start: held. Both are ASCII, so
                    // the split falls on a character boundary.
                    let keep = if pending.ends_with("]]") {
                        2
                    } else if pending.ends_with(']') {
                        1
                    } else {
                        0
                    };
                    let (give, hold) = pending.split_at(pending.len() - keep);
                    let give = give.to_string();
                    value.held = hold.to_string();
                    self.value_bytes(&give, out);
                }
            }
            Cdata::Done => self.after_the_section(text, out),
            Cdata::Plain | Cdata::Undecided if self.tags.value_on_its_own_line => {
                // A trailing newline may be the template's framing before the closing tag: held.
                let mut pending = std::mem::take(&mut value.held);
                pending.push_str(text);
                let (give, hold) = match pending.strip_suffix('\n') {
                    Some(give) => (give.to_string(), "\n".to_string()),
                    None => (pending, String::new()),
                };
                value.held = hold;
                self.value_bytes(&give, out);
            }
            Cdata::Plain | Cdata::Undecided => self.value_bytes(text, out),
        }
    }

    /// Text after a CDATA section closed, where only the closing tag should follow: whitespace
    /// is carried, a word is text between tags ([`Self::text_between`]), refused with the
    /// whitespace before it dropped. A value held whole for its close is written before the
    /// first word ([`Self::write_held_value`]): the refusal would take the carried bytes, and the
    /// value is among them.
    fn after_the_section(&mut self, text: &str, out: &mut Events) {
        if runs(text).any(|(_, space)| !space) {
            self.write_held_value(out);
        }
        self.text_between(text, out);
    }

    /// A value held whole for its close, written once its section closed and text follows the
    /// closer: the member as the section has it, with the bytes through the closer as its
    /// source, so that no report takes the bytes it is held in; the whitespace carried after the
    /// closer stays for the next event, and the closing tag then has nothing left to write
    /// ([`Self::close_value`]). Nothing for a streamed string, whose fragments are out, for a
    /// value with no section, or for one written already.
    fn write_held_value(&mut self, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let Some(end) = value.end else {
            return;
        };
        if value.written || !matches!(value.mode, Mode::Whole | Mode::Undecided) {
            return;
        }
        let closer_end = (end + CDATA_CLOSE.len()).min(self.object.carried.len());
        let after = self.object.carried.split_off(closer_end);
        let written = json(
            held_text(&self.object.carried, value.start, value.end),
            value.kind,
        );
        let fragment = self.object.member(&value.key, &written);
        let source = Text::uncounted(std::mem::take(&mut self.object.carried));
        self.object.push_fragment(fragment, source, out);
        self.object.carried = after;
        value.written = true;
    }

    /// The value's held start of a CDATA closer, if any, taken: value text after all.
    fn take_held(&mut self) -> String {
        match &mut self.stage {
            Stage::Value(value) => std::mem::take(&mut value.held),
            _ => String::new(),
        }
    }

    /// The value's own bytes, by its mode.
    fn value_bytes(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        self.object
            .value_bytes(&value.key, value.start, &mut value.mode, text, out);
    }

    /// `</parameter>`: a streamed string gets its closing quote, any other value is written whole.
    /// Bytes held as a CDATA closer's start that never completed are the value's text.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let held = self.take_held();
        if self.tags.value_on_its_own_line && held == "\n" {
            // The newline before the closing tag: the template's framing.
            if let Stage::Value(value) = &mut self.stage {
                if matches!(value.mode, Mode::Whole | Mode::Undecided) && value.end.is_none() {
                    value.end = Some(self.object.carried.len());
                }
            }
            self.object.carried.push('\n');
        } else {
            self.value_bytes(&held, out);
        }
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = if value.written {
            // The member left at its section's close, before text the syntax had no place for:
            // the closing tag has nothing left to write.
            None
        } else {
            Some(match value.mode {
                Mode::Streaming { opened: true } => "\"".to_string(),
                // An empty string: its opening and closing quotes come together.
                Mode::Streaming { opened: false } => self.object.member(&value.key, "\"\""),
                Mode::Whole | Mode::Undecided => {
                    let written = json(
                        held_text(&self.object.carried, value.start, value.end),
                        value.kind,
                    );
                    self.object.member(&value.key, &written)
                }
            })
        };
        match fragment {
            Some(fragment) => self.object.close_member(fragment, tag, out),
            // The closing tag of a member written already leaves as the syntax's wrapping, with
            // the whitespace carried before it.
            None => {
                self.object.carried.push_str(tag);
                self.object.drop_carried(out);
            }
        }
        self.stage = Stage::Between;
        self.released = false;
    }
}

/// Whether a call tag's tail is the model's index of the call, ` index="N"`.
fn is_index(tail: &str) -> bool {
    tail.strip_prefix(" index=\"")
        .and_then(|rest| rest.strip_suffix('"'))
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// The type an XTML parameter tag's tail names, ` type="…"`, when it is one of the six.
fn typed(tail: &str) -> Option<&str> {
    tail.strip_prefix(" type=\"")
        .and_then(|rest| rest.strip_suffix('"'))
        .filter(|kind| TYPES.contains(kind))
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out);
        text.len()
    }

    /// Only the invoke's closing tag is a started call's end, as for XML.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool {
        if terminal == self.invoke_close() || !self.started() {
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

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        tagged::{dropped, nothing_empty, refused, shown},
        testing::bytes,
    };

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "s": {"type": "string"},
                    "n": {"type": "integer"},
                    "ns": {"type": ["string", "null"]},
                    "o": {"type": "object"},
                    "b": {"type": "boolean"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The events of one invoke fed in `pieces`, closed by its tag or cut.
    fn run(pieces: &[&str], cut: bool) -> Vec<Event<'static>> {
        run_with(Tags::MINIMAX_M2, pieces, cut)
    }

    fn run_with(tags: Tags, pieces: &[&str], cut: bool) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, "call_0", tags);
        let mut out = Events::new();
        let declared = declared();
        for piece in pieces {
            assembler.feed(piece, &declared, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(tags.invoke_close, &mut out);
        }
        out.drain()
    }

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
                _ => None,
            })
            .collect()
    }

    /// A streamed string still open at the invoke's own closing marker, with the start of a cut
    /// tag held before it: the tag the marker cut short leaves before the call's end, as it came
    /// in, so the event bytes read as the input did.
    #[test]
    fn a_tag_the_marker_cut_short_leaves_before_the_calls_end() {
        let call = concat!(
            "f\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"b\" type=\"string\"<|sep|>x<|close|>argument"
        );
        let events = run_with(Tags::XTML, &[call], false);
        assert_eq!(arguments(&events), r#"{"b": "x"}"#);
        assert_eq!(dropped(&events), vec!["<|close|>argument".to_string()]);
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == Tags::XTML.invoke_close
        ));
        assert_eq!(bytes(&events), format!("{call}{}", Tags::XTML.invoke_close));
    }

    /// A recorded Kimi K3 call, from after `<|open|>call tool="`: a string, a number and an object.
    const XTML_CALL: &str = concat!(
        "get_user_info\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"user_id\" type=\"number\"<|sep|>7890<|close|>argument<|sep|>",
        "<|open|>argument key=\"special\" type=\"string\"<|sep|>black<|close|>argument<|sep|>",
        "<|open|>argument key=\"prefs\" type=\"object\"<|sep|>{\"size\": \"large\"}",
        "<|close|>argument<|sep|>"
    );

    #[test]
    fn an_xtml_call_gives_its_name_its_typed_arguments_and_every_byte_however_it_is_cut() {
        let whole = run_with(Tags::XTML, &[XTML_CALL], false);
        nothing_empty(&whole);
        assert!(matches!(
            &whole[0],
            Event::ToolCallStart { name, id, .. } if name == "get_user_info" && id == "call_0"
        ));
        assert_eq!(
            arguments(&whole),
            r#"{"user_id": 7890, "special": "black", "prefs": {"size": "large"}}"#
        );
        assert!(matches!(whole.last(), Some(Event::ToolCallEnd { .. })));
        let all = format!("{XTML_CALL}{}", Tags::XTML.invoke_close);
        assert_eq!(bytes(&whole), all);
        for cut in (1..XTML_CALL.len()).filter(|&cut| XTML_CALL.is_char_boundary(cut)) {
            let events = run_with(Tags::XTML, &[&XTML_CALL[..cut], &XTML_CALL[cut..]], false);
            nothing_empty(&events);
            assert_eq!(arguments(&events), arguments(&whole), "cut at {cut}");
            assert_eq!(bytes(&events), all, "cut at {cut}");
        }
    }

    #[test]
    fn the_six_xtml_types_are_written_as_the_model_wrote_them_and_a_string_streams() {
        let call = concat!(
            "f\" index=\"2\"<|sep|>",
            "<|open|>argument key=\"s\" type=\"string\"<|sep|>12<|close|>argument<|sep|>",
            "<|open|>argument key=\"n\" type=\"number\"<|sep|>1.5<|close|>argument<|sep|>",
            "<|open|>argument key=\"b\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>",
            "<|open|>argument key=\"z\" type=\"null\"<|sep|>null<|close|>argument<|sep|>",
            "<|open|>argument key=\"a\" type=\"array\"<|sep|>[1, \"x\"]<|close|>argument<|sep|>",
            "<|open|>argument key=\"o\" type=\"object\"<|sep|>{\"k\": null}<|close|>argument<|sep|>"
        );
        let events = run_with(Tags::XTML, &[call], false);
        assert_eq!(
            arguments(&events),
            r#"{"s": "12", "n": 1.5, "b": true, "z": null, "a": [1, "x"], "o": {"k": null}}"#
        );
        // The string streamed in pieces, each its own fragment.
        let streamed = run_with(
            Tags::XTML,
            &[
                "f\" index=\"2\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>he",
                "llo<|close|>argument<|sep|>",
            ],
            false,
        );
        let fragments: Vec<&str> = streamed
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(fragments, [r#"{"s": ""#, "he", "llo", "\"", "}"]);
    }

    #[test]
    fn an_xtml_type_the_syntax_lacks_and_a_tail_that_is_no_index_are_refused() {
        let call = concat!(
            "f\" index=\"x\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"date\"<|sep|>2026<|close|>argument<|sep|>",
            "<|open|>argument key=\"b\" type=\"number\"<|sep|>2<|close|>argument<|sep|>"
        );
        let events = run_with(Tags::XTML, &[call], false);
        assert_eq!(arguments(&events), r#"{"b": 2}"#);
        // Both are refused: counted by events of no bytes, the bytes never content.
        assert!(
            malformed(&events).iter().all(|(text, _)| text.is_empty()),
            "{:?}",
            malformed(&events)
        );
        let refused = refused(&events);
        assert!(refused.contains(&" index=\"x\"".to_string()), "{refused:?}");
        // The call tag's separator, carried after its tail was refused, leads the bad tag's bytes.
        assert!(
            refused.contains(&"<|sep|><|open|>argument key=\"a\" type=\"date\"<|sep|>".to_string()),
            "{refused:?}"
        );
        assert_eq!(bytes(&events), format!("{call}{}", Tags::XTML.invoke_close));
    }

    #[test]
    fn an_xtml_call_tag_without_an_index_is_a_call_all_the_same() {
        let call = concat!(
            "f\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"number\"<|sep|>1<|close|>argument<|sep|>"
        );
        let events = run_with(Tags::XTML, &[call], false);
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { name, source, .. } if name == "f" && source.text == "f\"<|sep|>"
        ));
        assert_eq!(arguments(&events), r#"{"a": 1}"#);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn an_xtml_stream_cut_inside_a_value_closes_nothing_and_a_call_cut_short_closes_its_object() {
        let cut = run_with(
            Tags::XTML,
            &["f\" index=\"1\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>par"],
            true,
        );
        assert_eq!(arguments(&cut), r#"{"s": "par"#);
        // The arguments are no whole value: one `Malformed` event of no bytes says so before the
        // end, the only event without bytes.
        assert!(
            matches!(
                &cut[cut.len() - 2..],
                [
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { source, .. }
                ] if text.text.is_empty() && source.text.is_empty()
            ),
            "{cut:?}"
        );
        nothing_empty(&cut[..cut.len() - 2]);

        let mut assembler = Assembler::new(0, "call_0", Tags::XTML);
        let mut out = Events::new();
        assembler.feed(
            "f\" index=\"1\"<|sep|><|open|>argument key=\"n\" type=\"number\"<|sep|>12",
            &declared(),
            &mut out,
        );
        assembler.close("", &mut out);
        let events = out.drain();
        nothing_empty(&events);
        assert_eq!(arguments(&events), "{}");
        assert!(refused(&events).iter().any(|text| text.ends_with("12")));
    }

    #[test]
    fn an_xtml_separator_split_across_pieces_still_ends_the_tag() {
        let whole = arguments(&run_with(Tags::XTML, &[XTML_CALL], false));
        let at = XTML_CALL.find("<|sep|>").expect("a separator") + 3;
        let events = run_with(Tags::XTML, &[&XTML_CALL[..at], &XTML_CALL[at..]], false);
        assert_eq!(arguments(&events), whole);
    }

    #[test]
    fn a_dsml_invoke_types_its_values_by_the_string_attribute_not_by_the_tool() {
        // `s` is declared a string and `n` an integer, but the attribute decides: `n` as text
        // streams, `s` as JSON is written whole, and a false value that is not JSON is the string
        // holding it.
        let pieces = [
            "f\">\n<｜DSML｜ parameter name=\"n\" string=\"true\">Par",
            "is</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"s\" string=\"false\">3",
            "</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"o\" string=\"false\">abc</｜DSML｜ parameter>\n",
        ];
        let events = run_with(Tags::DSML, &pieces, false);
        assert_eq!(bytes(&events), pieces.concat() + Tags::DSML.invoke_close);
        assert_eq!(arguments(&events), r#"{"n": "Paris", "s": 3, "o": "abc"}"#);
        // The streamed value came out in pieces: its opening fragment, `Par`, `is`.
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(&fragments[..3], &[r#"{"n": ""#, "Par", "is"]);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_dsml_parameter_tag_with_another_attribute_or_none_is_no_parameter() {
        for tag in [
            "<｜DSML｜ parameter name=\"n\" kind=\"int\">",
            "<｜DSML｜ parameter name=\"n\">",
        ] {
            let text = format!("f\">\n{tag}1</｜DSML｜ parameter>\n");
            let events = run_with(Tags::DSML, &[&text], false);
            assert_eq!(
                bytes(&events),
                text.clone() + Tags::DSML.invoke_close,
                "{tag}"
            );
            assert_eq!(arguments(&events), "{}", "{tag}");
            let why: Vec<MalformedReason> =
                malformed(&events).into_iter().map(|(_, why)| why).collect();
            assert_eq!(
                why,
                [
                    MalformedReason::TagTail,
                    MalformedReason::TextBetweenTags,
                    MalformedReason::TagOutOfPlace
                ],
                "{tag}"
            );
        }
    }

    /// A parameter tag written with another attribute's name, `<parameter key="s">`, which the
    /// syntax has no tag for, in every family's spelling: the invoke is read with no arguments,
    /// the model's lines never reach the client, and every byte of them is refused, the
    /// whitespace between the words and the newline between the lines included, the same however
    /// the bytes are cut. The template's newline after the invoke tag, where the family writes
    /// one, is dropped, and the one before its close is the object's source.
    #[test]
    fn text_between_tags_never_reaches_the_client_and_is_refused_whole_in_every_spelling() {
        const M2_CALL: &str = concat!(
            "f\">\n<parameter key=\"s\">Paris</parameter>\n",
            "<parameter key=\"n\">three or four</parameter>\n"
        );
        const M2_LINES: &str = concat!(
            "<parameter key=\"s\">Paris</parameter>\n",
            "<parameter key=\"n\">three or four</parameter>"
        );
        const CPM_CALL: &str =
            "f\"><param key=\"s\">Paris</param><param key=\"n\">three or four</param>";
        const DSML_CALL: &str = concat!(
            "f\">\n<｜DSML｜ parameter key=\"s\" string=\"true\">Paris</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter key=\"n\" string=\"true\">three or four</｜DSML｜ parameter>\n"
        );
        const DSML_V4_CALL: &str = concat!(
            "f\">\n<｜DSML｜parameter key=\"s\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "<｜DSML｜parameter key=\"n\" string=\"true\">three or four</｜DSML｜parameter>\n"
        );
        const XTML_CALL: &str = concat!(
            "f\" index=\"1\"<|sep|>",
            "<|open|>argument name=\"s\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|>",
            "<|open|>argument name=\"n\" type=\"string\"<|sep|>three or four<|close|>argument<|sep|>"
        );
        for (tags, call, lines, before) in [
            (Tags::MINIMAX_M2, M2_CALL, M2_LINES, vec!["\n"]),
            (Tags::DOTS3, M2_CALL, M2_LINES, vec!["\n"]),
            (Tags::MINICPM5, CPM_CALL, &CPM_CALL[3..], vec![]),
            (
                Tags::DSML,
                DSML_CALL,
                &DSML_CALL[4..DSML_CALL.len() - 1],
                vec!["\n"],
            ),
            (
                Tags::DSML_V4,
                DSML_V4_CALL,
                &DSML_V4_CALL[4..DSML_V4_CALL.len() - 1],
                vec!["\n"],
            ),
            (Tags::XTML, XTML_CALL, &XTML_CALL[19..], vec![]),
        ] {
            let all = format!("{call}{}", tags.invoke_close);
            let whole = run_with(tags, &[call], false);
            assert!(
                matches!(&whole[0], Event::ToolCallStart { name, .. } if name == "f"),
                "{tags:?}: {whole:?}"
            );
            assert_eq!(arguments(&whole), "{}", "{tags:?}");
            assert_eq!(shown(&whole), "", "{tags:?}");
            assert_eq!(refused(&whole).concat(), lines, "{tags:?}");
            assert_eq!(dropped(&whole), before, "{tags:?}");
            assert_eq!(bytes(&whole), all, "{tags:?}");
            nothing_empty(&whole);
            // At every two-way cut and one character at a time the events differ, the bytes a
            // client reads do not.
            let mut cuts: Vec<Vec<&str>> = (1..call.len())
                .filter(|&cut| call.is_char_boundary(cut))
                .map(|cut| vec![&call[..cut], &call[cut..]])
                .collect();
            cuts.push(
                call.char_indices()
                    .map(|(at, c)| &call[at..at + c.len_utf8()])
                    .collect(),
            );
            for pieces in cuts {
                let events = run_with(tags, &pieces, false);
                assert_eq!(shown(&events), "", "{tags:?}: {pieces:?}");
                assert_eq!(refused(&events).concat(), lines, "{tags:?}: {pieces:?}");
                assert_eq!(arguments(&events), "{}", "{tags:?}: {pieces:?}");
                assert_eq!(dropped(&events), before, "{tags:?}: {pieces:?}");
                assert_eq!(bytes(&events), all, "{tags:?}: {pieces:?}");
                nothing_empty(&events);
            }
        }
    }

    #[test]
    fn a_rendered_invoke_gives_its_call_typed_by_the_tool_and_every_byte() {
        let text = concat!(
            "f\">\n<parameter name=\"s\">Paris</parameter>\n<parameter name=\"n\">3</parameter>\n",
            "<parameter name=\"b\">true</parameter>\n",
            "<parameter name=\"o\">{\"a\": [1, 2]}</parameter>\n"
        );
        let events = run(&[text], false);
        assert_eq!(
            bytes(&events),
            format!("{text}{}", Tags::MINIMAX_M2.invoke_close)
        );
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "f" && source.text == "f\">"
        ));
        assert_eq!(
            arguments(&events),
            r#"{"s": "Paris", "n": 3, "b": true, "o": {"a": [1, 2]}}"#
        );
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == Tags::MINIMAX_M2.invoke_close
        ));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_declared_string_streams_as_it_arrives_and_a_cut_leaves_it_open() {
        let pieces = ["f\">\n<parameter name=\"s\">Par", "is, Fr", "ance"];
        let events = run(&pieces, true);
        assert_eq!(bytes(&events), pieces.concat());
        // The opening fragment, then every piece of the string as it came; no closing quote.
        assert_eq!(arguments(&events), r#"{"s": "Paris, France"#);
        // No closing quote is made up: the one word before the end says the value is no whole one.
        assert!(matches!(
            &events[events.len() - 2..],
            [
                Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                Event::ToolCallEnd { source, .. }
            ] if text.text.is_empty() && source.text.is_empty()
        ));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count(),
            1
        );
    }

    /// An invoke the cut found before it named a call (its name's quote never came): what it
    /// held is refused as `UnterminatedRegion`, as the keyed assembler refuses it, after whatever
    /// its syntax refused inside it; nothing is shown, and no `Malformed` event carries bytes,
    /// since one would end the gateway reader's release of a refused output that made no call.
    #[test]
    fn an_invoke_the_cut_found_unnamed_is_refused_whole() {
        for input in [
            "fo", // codespell:ignore fo
            "f<parameter name=\"s\">x\n",
        ] {
            let events = run(&[input], true);
            assert_eq!(bytes(&events), input, "{input:?}");
            assert_eq!(shown(&events), "", "{input:?}: {events:?}");
            let reported = malformed(&events);
            assert!(
                reported.iter().all(|(text, _)| text.is_empty()),
                "{input:?}: {reported:?}"
            );
            assert_eq!(
                reported.last().map(|(_, why)| *why),
                Some(MalformedReason::UnterminatedRegion),
                "{input:?}: {reported:?}"
            );
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::ToolCallStart { .. })),
                "{input:?}"
            );
            // Every byte is refused or dropped as the syntax's wrapping; none is content.
            let accounted: usize = refused(&events).iter().map(String::len).sum::<usize>()
                + dropped(&events).iter().map(String::len).sum::<usize>();
            assert_eq!(accounted, input.len(), "{input:?}: {events:?}");
        }
    }

    #[test]
    fn a_nullable_string_is_null_or_the_string() {
        let as_null = run(&["f\">\n<parameter name=\"ns\">null</parameter>\n"], false);
        assert_eq!(arguments(&as_null), r#"{"ns": null}"#);
        let as_text = run(
            &[
                "f\">\n<parameter name=\"ns\">nu",
                "ll and void</parameter>\n",
            ],
            false,
        );
        assert_eq!(arguments(&as_text), r#"{"ns": "null and void"}"#);
    }

    #[test]
    fn an_empty_string_and_an_invoke_without_parameters_write_their_objects() {
        let empty = run(&["f\">\n<parameter name=\"s\"></parameter>\n"], false);
        assert_eq!(arguments(&empty), r#"{"s": ""}"#);
        let none = run(&["f\">\n"], false);
        assert_eq!(arguments(&none), "{}");
        assert!(matches!(&none[0], Event::ToolCallStart { name, .. } if name == "f"));
    }

    #[test]
    fn a_parameter_tag_with_a_tail_is_no_parameter_and_the_text_is_refused() {
        let text = "f\">\n<parameter name=\"n\" type=\"int\">1</parameter>\n";
        let events = run(&[text], false);
        assert_eq!(
            bytes(&events),
            format!("{text}{}", Tags::MINIMAX_M2.invoke_close)
        );
        let reported = malformed(&events);
        // The tag with its tail, `1` (text where the syntax has tags) and the stray close are all
        // refused: counted by an event of no bytes each, the bytes never content. The template's
        // newline before the tag rides with the tag's bytes, as every report carried what was held
        // before it.
        assert_eq!(
            reported,
            [
                (String::new(), MalformedReason::TagTail),
                (String::new(), MalformedReason::TextBetweenTags),
                (String::new(), MalformedReason::TagOutOfPlace)
            ]
        );
        assert_eq!(
            refused(&events),
            ["\n<parameter name=\"n\" type=\"int\">", "1", "</parameter>"]
        );
        assert_eq!(shown(&events), "");
        assert_eq!(arguments(&events), "{}");
    }

    /// An invoke that never named a call is refused whole, its closing tag included: the client
    /// reads nothing of it, as the keyed syntax has it for a block that named no call.
    #[test]
    fn an_invoke_that_closes_inside_its_name_is_refused_whole_and_starts_no_call() {
        let events = run(&["get_wea"], false);
        assert_eq!(
            bytes(&events),
            format!("get_wea{}", Tags::MINIMAX_M2.invoke_close)
        );
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::ClosedBeforeTheName)]
        );
        assert_eq!(
            refused(&events),
            [format!("get_wea{}", Tags::MINIMAX_M2.invoke_close)]
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        // A name a tag cuts short, a parameter the unnamed call cannot take and the close: three
        // refused runs, every byte in them, nothing shown.
        let cut = run(&["f<parameter name=\"s\">x</parameter>"], false);
        assert_eq!(
            bytes(&cut),
            format!(
                "f<parameter name=\"s\">x</parameter>{}",
                Tags::MINIMAX_M2.invoke_close
            )
        );
        assert_eq!(shown(&cut), "");
        assert_eq!(refused(&cut).concat(), bytes(&cut));
        assert!(malformed(&cut).iter().all(|(text, _)| text.is_empty()));
        nothing_empty(&cut);
        assert!(!cut.iter().any(|e| matches!(e, Event::ToolCallStart { .. })));
    }

    #[test]
    fn a_tag_inside_a_value_is_the_values_text_and_a_close_inside_a_value_closes_it() {
        let events = run(
            &["f\">\n<parameter name=\"s\">a <parameter name=\"x\"> b</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"s": "a <parameter name=\"x\"> b"}"#);
        let cut = run(&["f\">\n<parameter name=\"n\">12"], false);
        // The close cuts a whole value: the tag and the digits, held for the value's close that
        // never came, are refused together, and the object closes empty.
        assert_eq!(arguments(&cut), "{}");
        assert_eq!(
            malformed(&cut)[0],
            (String::new(), MalformedReason::TagCutShort)
        );
        assert_eq!(refused(&cut), ["\n<parameter name=\"n\">12"]);
    }

    /// The invoke's close inside a streamed string, by its own closing tag or the block's end:
    /// the model ended the call, so the value's bytes are the value whole. The string is closed,
    /// under one `Malformed { OpenAtTheMarker }` event of no bytes right before the closing quote;
    /// the start of a tag held there is the value's text, since no tag followed, and dots3's
    /// framing newline is the quote's source; the object closes, the tag's bytes are still the
    /// call's end, and the arguments read as JSON.
    #[test]
    fn a_close_inside_a_streamed_string_closes_it_there_and_says_so() {
        let invoke_close = Tags::MINIMAX_M2.invoke_close;
        for (tags, pieces, terminal, value) in [
            (
                Tags::MINIMAX_M2,
                &["f\">\n<parameter name=\"s\">Par"][..],
                "",
                "Par",
            ),
            (
                Tags::MINIMAX_M2,
                &["f\">\n<parameter name=\"s\">Pa", "r</par"][..],
                invoke_close,
                "Par</par",
            ),
            (
                Tags::DOTS3,
                &["f\">\n<parameter name=\"s\">\nPar\n"][..],
                "",
                "Par",
            ),
            (
                Tags::XTML,
                &["f\" index=\"1\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>Par"][..],
                "",
                "Par",
            ),
        ] {
            let mut assembler = Assembler::new(0, "call_0", tags);
            let mut out = Events::new();
            for piece in pieces {
                assembler.feed(piece, &declared(), &mut out);
            }
            assembler.close(terminal, &mut out);
            let events = out.drain();
            assert_eq!(
                arguments(&events),
                format!("{{\"s\": \"{value}\"}}"),
                "{pieces:?}"
            );
            let at = events
                .iter()
                .position(|event| matches!(event, Event::Malformed { .. }))
                .expect("the one word");
            assert!(
                matches!(
                    &events[at..at + 2],
                    [
                        Event::Malformed { text, why: MalformedReason::OpenAtTheMarker },
                        Event::ToolCallArguments { json, .. }
                    ] if text.text.is_empty() && json == "\""
                ),
                "{pieces:?}: {events:?}"
            );
            assert!(
                matches!(
                    events.last(),
                    Some(Event::ToolCallEnd { source, .. }) if source.text == terminal
                ),
                "{pieces:?}: {events:?}"
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, Event::Malformed { .. }))
                    .count(),
                1,
                "{pieces:?}"
            );
            assert_eq!(bytes(&events), format!("{}{terminal}", pieces.concat()));
        }
    }

    /// MiniCPM5's spelling: `<param name="…">`, `</param>`, `</function>`, CDATA sections.
    const MINICPM5_CALL: &str = concat!(
        "f\">",
        "<param name=\"s\"><![CDATA[café <b> & \"q\"\nligne]]></param>",
        "<param name=\"o\"><![CDATA[{\"k\": \"<v>\"}]]></param>",
        "<param name=\"n\">3</param>"
    );

    #[test]
    fn a_cdata_section_is_the_value_inside_it_for_a_string_and_for_a_whole_value() {
        let events = run_with(Tags::MINICPM5, &[MINICPM5_CALL], false);
        assert_eq!(bytes(&events), format!("{MINICPM5_CALL}</function>"));
        assert_eq!(
            arguments(&events),
            r#"{"s": "café <b> & \"q\"\nligne", "o": {"k": "<v>"}, "n": 3}"#
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_cdata_section_reads_the_same_however_the_bytes_are_cut() {
        let whole = run_with(Tags::MINICPM5, &[MINICPM5_CALL], false);
        let expected = arguments(&whole);
        // Every split at a character boundary, one byte of UTF-8 at a time where the text allows,
        // the closer's `]]` and `>` apart, the opener in halves.
        let boundaries: Vec<usize> = MINICPM5_CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run_with(
                Tags::MINICPM5,
                &[&MINICPM5_CALL[..at], &MINICPM5_CALL[at..]],
                false,
            );
            assert_eq!(
                bytes(&events),
                format!("{MINICPM5_CALL}</function>"),
                "cut at {at}"
            );
            assert_eq!(arguments(&events), expected, "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        let one_by_one: Vec<&str> = MINICPM5_CALL
            .char_indices()
            .map(|(at, c)| &MINICPM5_CALL[at..at + c.len_utf8()])
            .collect();
        let events = run_with(Tags::MINICPM5, &one_by_one, false);
        assert_eq!(arguments(&events), expected);
    }

    #[test]
    fn a_bracket_inside_a_cdata_value_and_a_short_plain_value_are_read_as_they_are() {
        // `]` and `]]` that do not start the closer are the value's text.
        let call = "f\">\n<param name=\"s\"><![CDATA[a[1]] and b]]></param>";
        let events = run_with(Tags::MINICPM5, &[call], false);
        assert_eq!(arguments(&events), r#"{"s": "a[1]] and b"}"#);
        // A value shorter than the opener, and one that starts like it but is not one.
        let short = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\">Rome</param>"],
            false,
        );
        assert_eq!(arguments(&short), r#"{"s": "Rome"}"#);
        let alike = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\"><![CDATB[x]]></param>"],
            false,
        );
        assert_eq!(arguments(&alike), r#"{"s": "<![CDATB[x]]>"}"#);
    }

    #[test]
    fn text_after_a_cdata_closer_is_refused_and_a_cut_section_keeps_its_held_bytes() {
        let events = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\"><![CDATA[x]]>tail</param>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"s": "x"}"#);
        assert_eq!(
            malformed(&events),
            vec![(String::new(), MalformedReason::TextBetweenTags)]
        );
        assert_eq!(refused(&events), ["tail"]);
        // Cut with a `]` held: the byte is the value's text, in the value, not lost and not
        // content.
        let cut = run_with(Tags::MINICPM5, &["f\"><param name=\"s\"><![CDATA[x]"], true);
        assert_eq!(bytes(&cut), "f\"><param name=\"s\"><![CDATA[x]");
        assert_eq!(arguments(&cut), r#"{"s": "x]"#);
        assert!(shown(&cut).is_empty(), "{cut:?}");
        // Cut inside a tag behind held `]` bytes: the held bytes leave before the tag's start,
        // in the output's order, at the stream's end and at the region's close.
        for text in [
            "f\"><param name=\"s\"><![CDATA[x]</para",
            "f\"><param name=\"s\"><![CDATA[x]]</para",
        ] {
            let cut = run_with(Tags::MINICPM5, &[text], true);
            assert_eq!(bytes(&cut), text, "{text:?} cut");
            let closed = run_with(Tags::MINICPM5, &[text], false);
            assert_eq!(
                bytes(&closed),
                format!("{text}</function>"),
                "{text:?} closed"
            );
        }
    }

    /// A value written whole (an object, an integer, a string that may be null, a key no tool
    /// declares) in a CDATA section with a word after the closer, before the closing tag: the
    /// value is the section's text, written at the word; the word is text between tags; the
    /// closing tag leaves as the syntax's wrapping; every byte lands once; and the same whole, at
    /// every cut and one character at a time. (The word used to drop the carried value as the
    /// template's spacing, and the closing tag then sliced past the carried bytes.)
    #[test]
    fn a_whole_value_in_a_cdata_section_followed_by_a_word_is_written_and_the_word_refused() {
        for (call, expected, word) in [
            (
                "f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]>tail</param>",
                r#"{"o": {"k": 1}}"#,
                "tail",
            ),
            (
                "f\"><param name=\"n\"><![CDATA[3]]>x</param><param name=\"s\">a</param>",
                r#"{"n": 3, "s": "a"}"#,
                "x",
            ),
            (
                "f\"><param name=\"ns\"><![CDATA[null]]>x</param>",
                r#"{"ns": null}"#,
                "x",
            ),
            (
                "f\"><param name=\"ns\"><![CDATA[nu]]>x</param>",
                r#"{"ns": "nu"}"#,
                "x",
            ),
            (
                "f\"><param name=\"u\"><![CDATA[[1, 2]]]>東京</param>",
                r#"{"u": [1, 2]}"#,
                "東京",
            ),
        ] {
            let all = format!("{call}</function>");
            let whole = run_with(Tags::MINICPM5, &[call], false);
            nothing_empty(&whole);
            assert_eq!(arguments(&whole), expected, "{call}");
            assert_eq!(
                malformed(&whole),
                vec![(String::new(), MalformedReason::TextBetweenTags)],
                "{call}"
            );
            assert_eq!(refused(&whole), [word], "{call}");
            assert!(
                whole.iter().any(|event| matches!(
                    event,
                    Event::Dropped { text, why: DropReason::Wrapper } if text.text == "</param>"
                )),
                "{call}: {whole:?}"
            );
            assert!(matches!(whole.last(), Some(Event::ToolCallEnd { .. })));
            assert_eq!(bytes(&whole), all, "{call}");
            let mut cuts: Vec<Vec<&str>> = (1..call.len())
                .filter(|&cut| call.is_char_boundary(cut))
                .map(|cut| vec![&call[..cut], &call[cut..]])
                .collect();
            cuts.push(
                call.char_indices()
                    .map(|(at, c)| &call[at..at + c.len_utf8()])
                    .collect(),
            );
            for pieces in cuts {
                let events = run_with(Tags::MINICPM5, &pieces, false);
                nothing_empty(&events);
                assert_eq!(arguments(&events), expected, "{pieces:?}");
                // A cut inside the word splits its report in two; the bytes refused are the
                // word's, and the client reads none of them.
                let reported = malformed(&events);
                assert!(
                    reported
                        .iter()
                        .all(|(text, why)| text.is_empty()
                            && *why == MalformedReason::TextBetweenTags),
                    "{pieces:?}: {reported:?}"
                );
                assert_eq!(refused(&events).concat(), word, "{pieces:?}");
                assert_eq!(shown(&events), "", "{pieces:?}");
                assert_eq!(bytes(&events), all, "{pieces:?}");
            }
        }
    }

    /// The whitespace after the closer: before a word it is the template's, dropped before the
    /// report as everywhere between tags; before the closing tag with no word it is the member's
    /// source, as before. The same shape ended otherwise: the invoke's close right after the word
    /// closes the object after the written member, the whitespace carried as the close's source;
    /// the stream's end there leaves the arguments no whole value, as every cut does.
    #[test]
    fn a_written_member_followed_by_a_word_outlives_whitespace_the_invokes_close_and_the_end() {
        let spaced = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]> \ntail</param>"],
            false,
        );
        assert_eq!(arguments(&spaced), r#"{"o": {"k": 1}}"#);
        assert_eq!(
            malformed(&spaced),
            vec![(String::new(), MalformedReason::TextBetweenTags)]
        );
        assert_eq!(refused(&spaced), ["tail"]);
        assert_eq!(
            bytes(&spaced),
            "f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]> \ntail</param></function>"
        );
        let quiet = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]> \n</param>"],
            false,
        );
        assert_eq!(arguments(&quiet), r#"{"o": {"k": 1}}"#);
        assert!(malformed(&quiet).is_empty());
        let call = "f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]>tail \n";
        let closed = run_with(Tags::MINICPM5, &[call], false);
        nothing_empty(&closed);
        assert_eq!(arguments(&closed), r#"{"o": {"k": 1}}"#);
        assert_eq!(
            malformed(&closed),
            vec![(String::new(), MalformedReason::TextBetweenTags)]
        );
        assert_eq!(refused(&closed), ["tail"]);
        assert_eq!(bytes(&closed), format!("{call}</function>"));
        let cut = run_with(Tags::MINICPM5, &[call], true);
        assert_eq!(arguments(&cut), r#"{"o": {"k": 1}"#);
        assert!(matches!(
            cut.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text.is_empty()
        ));
        assert_eq!(bytes(&cut), call);
    }

    /// dots3's spelling: MiniMax M2's tags, every value on a line of its own.
    const DOTS3_CALL: &str = concat!(
        "f\">\n<parameter name=\"s\">\ncafé\nà deux\n</parameter>\n",
        "<parameter name=\"n\">\n3\n</parameter>\n",
        "<parameter name=\"o\">\n{\"k\": [1]}\n</parameter>\n"
    );

    #[test]
    fn a_value_on_its_own_line_sheds_its_framing_newlines_however_the_bytes_are_cut() {
        let whole = run_with(Tags::DOTS3, &[DOTS3_CALL], false);
        assert_eq!(bytes(&whole), format!("{DOTS3_CALL}</invoke>"));
        assert_eq!(
            arguments(&whole),
            r#"{"s": "café\nà deux", "n": 3, "o": {"k": [1]}}"#
        );
        assert!(malformed(&whole).is_empty());
        let boundaries: Vec<usize> = DOTS3_CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run_with(Tags::DOTS3, &[&DOTS3_CALL[..at], &DOTS3_CALL[at..]], false);
            assert_eq!(
                bytes(&events),
                format!("{DOTS3_CALL}</invoke>"),
                "cut at {at}"
            );
            assert_eq!(arguments(&events), arguments(&whole), "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        // Without the framing the value is read as it is; an empty value is an empty string.
        let bare = run_with(
            Tags::DOTS3,
            &["f\">\n<parameter name=\"s\">Rome</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&bare), r#"{"s": "Rome"}"#);
        let empty = run_with(
            Tags::DOTS3,
            &["f\">\n<parameter name=\"s\">\n\n</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&empty), r#"{"s": ""}"#);
    }

    /// A call the end of the stream cut inside a value held for its close, right after a
    /// parameter tag, inside a tag or between tags: what the model wrote of the value leaves as
    /// the call's own fragment, its key and its bytes as written (a value that is JSON, a number,
    /// a word that may still be `null`, a CDATA section open or closed), a string whose key came
    /// opens, a cut tag and the template's whitespace leave as wrapping, nothing of the call
    /// reaches a chat client as content, and the one `Malformed` event of no bytes still says the
    /// arguments are no whole value before the call's end; whole and one character at a time.
    #[test]
    fn a_value_the_end_cut_leaves_as_the_calls_own_bytes_and_nothing_as_content() {
        let cases: [(Tags, &str, &str, &[&str]); 30] = [
            // Typed by the tools: an object cut inside a nested string.
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"s\">Paris</parameter>\n<parameter name=\"o\">{\"size\": \"large\", \"temp",
                r#"{"s": "Paris", "o": {"size": "large", "temp"#,
                &[],
            ),
            // An integer: the digits so far.
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"n\">12", r#"{"n": 12"#, &[]),
            // A string that may be null, while the word fits.
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"ns\">nu", r#"{"ns": nu"#, &[]),
            // A key no tool declares, written whole: its text so far where that is the start of
            // a JSON value, a Python literal the cut found complete as the JSON the close makes
            // of it, else the opening of the string it would be written as.
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"x\">fal", r#"{"x": fal"#, &[]),
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"x\">True", r#"{"x": true"#, &[]),
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"x\">None", r#"{"x": null"#, &[]),
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"x\">hel", r#"{"x": "hel"#, &[]),
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"o\">None]",
                r#"{"o": "None]"#,
                &[],
            ),
            // The key came and no byte of its value.
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"o\">", r#"{"o": "#, &[]),
            // The key of a string came and no byte of it: the string opens.
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"s\">", r#"{"s": ""#, &[]),
            // Inside a parameter tag: the tag's bytes and the newline before it are wrapping.
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"s\">x</parameter>\n<param",
                r#"{"s": "x""#,
                &["\n<param"],
            ),
            // Between tags: the template's newline is wrapping.
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"s\">x</parameter>\n",
                r#"{"s": "x""#,
                &["\n"],
            ),
            // DSML, typed by the attribute: a `string="false"` list cut inside.
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"s\" string=\"true\">x</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"o\" string=\"false\">{\"a\": [1, 2",
                r#"{"s": "x", "o": {"a": [1, 2"#,
                &[],
            ),
            // DSML: the key of a JSON value came and no byte of it.
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"o\" string=\"false\">",
                r#"{"o": "#,
                &[],
            ),
            // XTML, typed by the attribute: an object cut inside.
            (
                Tags::XTML,
                "f\" index=\"1\"<|sep|><|open|>argument key=\"o\" type=\"object\"<|sep|>{\"k\": 1",
                r#"{"o": {"k": 1"#,
                &[],
            ),
            // A value on its own line: the newline after the tag is the template's, the value cut.
            (
                Tags::DOTS3,
                "f\">\n<parameter name=\"o\">\n{\"k\": [1",
                r#"{"o": {"k": [1"#,
                &[],
            ),
            // A value on its own line whose newline is held as the framing before a closing tag
            // that never came: the framing stays the template's, in the fragment's source.
            (
                Tags::DOTS3,
                "f\">\n<parameter name=\"n\">\n3\n",
                r#"{"n": 3"#,
                &[],
            ),
            // A CDATA section still open, a `]` held as the closer's start: the value's text.
            (
                Tags::MINICPM5,
                "f\"><param name=\"o\"><![CDATA[{\"k\": [1]",
                r#"{"o": {"k": [1]"#,
                &[],
            ),
            // A CDATA section that closed before the closing tag: the member whole.
            (
                Tags::MINICPM5,
                "f\"><param name=\"o\"><![CDATA[{\"k\": 1}]]>",
                r#"{"o": {"k": 1}"#,
                &[],
            ),
            // The end inside a string value's closing tag, as recorded: the value as far as the
            // model wrote it, the tag's start as wrapping after the call's end; cut after the
            // tag's word and right after its sentinel. (The tag's bytes used to ride inside the
            // value, `"from_currency": "KRW</｜DSML｜ parameter`.)
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"amount\" string=\"false\">2500</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"from_currency\" string=\"true\">KRW</｜DSML｜ parameter",
                r#"{"amount": 2500, "from_currency": "KRW"#,
                &["</｜DSML｜ parameter"],
            ),
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"origin\" string=\"true\">CDG</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"destination\" string=\"true\">JFK</｜DSML｜",
                r#"{"origin": "CDG", "destination": "JFK"#,
                &["</｜DSML｜"],
            ),
            // The same inside the closing tag of an empty string, which opens, and of a JSON
            // value the model wrote whole, which is whole.
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"s\" string=\"true\"></｜DSML｜ param",
                r#"{"s": ""#,
                &["</｜DSML｜ param"],
            ),
            (
                Tags::DSML,
                "f\">\n<｜DSML｜ parameter name=\"o\" string=\"false\">{\"a\": 1}</｜DSML｜ parameter",
                r#"{"o": {"a": 1}"#,
                &["</｜DSML｜ parameter"],
            ),
            // The families whose tags are text the model could write: the tag's start at the
            // end of a cut output is the tag all the same, down to its first byte.
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"s\">Paris</par",
                r#"{"s": "Paris"#,
                &["</par"],
            ),
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"o\">{\"k\": 1}</parameter",
                r#"{"o": {"k": 1}"#,
                &["</parameter"],
            ),
            (Tags::MINIMAX_M2, "f\">\n<parameter name=\"n\">12<", r#"{"n": 12"#, &["<"]),
            // A held start of the OPENING tag is the value's text, as the whole tag is inside a
            // value: the cut is judged as the whole tag would be.
            (
                Tags::MINIMAX_M2,
                "f\">\n<parameter name=\"s\">see <p",
                r#"{"s": "see <p"#,
                &[],
            ),
            // Where the tags are special tokens, a held strict prefix of one is the value's text:
            // the model cannot write the token and no cut splits it.
            (
                Tags::XTML,
                "f\" index=\"1\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>x <|",
                r#"{"s": "x <|"#,
                &[],
            ),
            (
                Tags::MINICPM5,
                "f\"><param name=\"s\">x</par",
                r#"{"s": "x"#,
                &["</par"],
            ),
            // A value on its own line: the newline before the cut tag is the template's framing,
            // in the fragment's source.
            (
                Tags::DOTS3,
                "f\">\n<parameter name=\"n\">\n3\n</par",
                r#"{"n": 3"#,
                &["</par"],
            ),
        ];
        for (tags, call, expected, wrapping) in cases {
            let characters: Vec<&str> = call
                .char_indices()
                .map(|(at, c)| &call[at..at + c.len_utf8()])
                .collect();
            for (how, pieces) in [("whole", vec![call]), ("by character", characters)] {
                let events = run_with(tags, &pieces, true);
                assert_eq!(bytes(&events), call, "{call:?} {how}");
                assert_eq!(arguments(&events), expected, "{call:?} {how}: {events:?}");
                // The contract of every fragment: the arguments so far are a strict prefix of a
                // JSON value, one `serde_json` reads whole or stops reading only at its end.
                let strict_prefix = match serde_json::from_str::<serde_json::Value>(expected) {
                    Ok(_) => true,
                    Err(error) => error.is_eof(),
                };
                assert!(
                    strict_prefix,
                    "{call:?}: no JSON value begins with {expected:?}"
                );
                assert!(shown(&events).is_empty(), "{call:?} {how}: {events:?}");
                assert_eq!(
                    malformed(&events),
                    vec![(String::new(), MalformedReason::UnterminatedRegion)],
                    "{call:?} {how}: {events:?}"
                );
                // The call ends after the one word of no bytes; a tag the end cut short behind
                // the value leaves after the end, as wrapping, where there is one.
                let end_at = events
                    .iter()
                    .position(|event| matches!(event, Event::ToolCallEnd { .. }))
                    .expect("the call's end");
                assert!(
                    matches!(
                        &events[end_at - 1..=end_at],
                        [
                            Event::Malformed { .. },
                            Event::ToolCallEnd { source, .. }
                        ] if source.text.is_empty()
                    ),
                    "{call:?} {how}: {events:?}"
                );
                assert!(
                    events[end_at + 1..].iter().all(|event| matches!(
                        event,
                        Event::Dropped {
                            why: DropReason::Wrapper,
                            ..
                        }
                    )),
                    "{call:?} {how}: {events:?}"
                );
                let dropped_after_the_start: Vec<String> = events
                    .iter()
                    .skip_while(|event| !matches!(event, Event::ToolCallStart { .. }))
                    .filter_map(|event| match event {
                        Event::Dropped { text, .. } => Some(text.text.to_string()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    dropped_after_the_start, wrapping,
                    "{call:?} {how}: {events:?}"
                );
                nothing_empty(&events[..end_at - 1]);
                nothing_empty(&events[end_at + 1..]);
            }
        }
    }

    /// A tag the end cut short behind a held value leaves after the call, as the stop drops it:
    /// the value's fragment, the call's end, then the tag's bytes, so every byte keeps its place.
    #[test]
    fn a_cut_tag_behind_a_held_value_leaves_after_the_calls_end() {
        let call = concat!(
            "f\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"n\" type=\"number\"<|sep|>1.5<|close|>"
        );
        let events = run_with(Tags::XTML, &[call], true);
        assert_eq!(bytes(&events), call);
        assert_eq!(arguments(&events), r#"{"n": 1.5"#);
        assert!(shown(&events).is_empty(), "{events:?}");
        assert!(
            matches!(
                &events[events.len() - 3..],
                [
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { source, .. },
                    Event::Dropped { text: tag, why: DropReason::Wrapper },
                ] if text.text.is_empty() && source.text.is_empty() && tag.text == "<|close|>"
            ),
            "{events:?}"
        );
    }
}
