//! Events from the tagged calls of one block as they arrive.
//!
//! [`Assembler`] takes the calls written as tags, `<function=NAME>` and then `<parameter=KEY>`
//! around each value's text, in pieces as the format cuts them out of the model's output, and
//! turns each into the events of one call: `ToolCallStart` once the function tag is whole,
//! `ToolCallArguments` as the arguments object is built, `ToolCallEnd` at `</function>`. The
//! templates write one function block per call region; a model answering with several calls at
//! once also writes them one after another inside one region
//! (`<function=A>…</function><function=B>…</function>` between one pair of call markers), and the
//! engines' own parsers read every block in the region as a call. So does the assembler: after
//! `</function>` it stays in the block, and the next `<function=` starts the next call, at the
//! next index, with the id that index mints ([`CallIds`]). The model writes no JSON object, so
//! the assembler writes one per call, member by member, in the spelling the references use:
//! `{"city": "Paris", "limit": 5}`. What each value's text means as JSON is [`json`]'s decision,
//! from the type the tool declares for the parameter ([`Declared`]).
//!
//! A value declared a string is streamed: its opening fragment `{"key": "` (or `, "key": "`) is
//! pushed at the parameter tag, each piece of its text as it arrives, and `"` at `</parameter>`.
//! Any other value is pushed whole at its close, because its text has to be read before it is
//! written. A string that may also be null streams from the first byte that rules null out.
//!
//! Where a template puts the value on a line of its own ([`Placement::OwnLine`]: Qwen 3.5,
//! Qwen3-Coder) it writes one newline after `<parameter=KEY>` and one before `</parameter>`;
//! neither is the value's, and the assembler takes exactly those two away, as the engine's parsers do.
//! The second cannot be told from a newline the value ends with until the closing tag follows it,
//! so a newline at the end of a streamed piece is held back until the next piece says which it is.
//! Where a template writes the value between the tags directly ([`Placement::Direct`]: Seed-OSS,
//! MiMo) every byte between them is the value's, a newline it ends with included, and nothing is
//! taken away: a patch keeps its final newline, and a value that is one newline is one newline.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{runs, DropReason, Event, Events, MalformedReason, Text},
    ids::CallIds,
    markers::{Piece, Scanner},
    tagged::{
        end_cut_call,
        value::{json, Kind, NULL_WORDS},
    },
    Declared,
};

const FUNCTION_OPEN: usize = 0;
const PARAMETER_OPEN: usize = 1;
const PARAMETER_CLOSE: usize = 2;
const FUNCTION_CLOSE: usize = 3;
/// The four tags of the syntax: a function's opening up to its name, a parameter's opening up to
/// its key, a parameter's close, a function's close.
pub const TAGS: [&str; 4] = ["<function=", "<parameter=", "</parameter>", "</function>"];

/// How a family's template writes a value that is not a string. Qwen 3.5, Qwen3-Coder and their
/// kin write an object or a list as JSON (`tojson`) and a boolean or null as Python's word
/// (`True`, `None`); Seed-OSS writes every value with Python's `str`, an object or a list as its
/// repr (`{'size': 'large'}`). The assembler reads all of these as the JSON they stand for
/// ([`json`]); the grammar derived from a table pins an object's or a list's
/// shape only where the template writes it as JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// An object or a list as JSON, a boolean or null as Python's word: Qwen 3.5 and later,
    /// Qwen3-Coder.
    Json,
    /// Every value as Python's text, an object or a list as its repr: Seed-OSS.
    Python,
}

/// Where a family's template puts a value between its parameter tags. Qwen 3.5, Qwen3-Coder and
/// their kin write a newline after `<parameter=KEY>` and one before `</parameter>`, so the value
/// stands on a line of its own; Seed-OSS and MiMo write it between the tags directly. The
/// assembler takes the template's two newlines away only where the template writes them, so a
/// value that ends in a newline, a patch or a file's body, reaches the client whole under either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// On a line of its own: a newline after the opening tag and one before the closing tag,
    /// neither of them the value's. Qwen 3.5 and later, Qwen3-Coder.
    OwnLine,
    /// Between the tags directly, every byte between them the value's: Seed-OSS, MiMo.
    Direct,
}

/// The events of one block's tool calls, from the bytes of their tags.
#[derive(Clone, Debug)]
pub struct Assembler {
    /// The index of the call that is open, or of the next one to start; its id is what `ids`
    /// mints for that index.
    index: u32,
    /// Where the output's call ids count from.
    ids: CallIds,
    scanner: Scanner,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// The open call's function name once its tag is whole, which is when `ToolCallStart` was
    /// pushed; nothing before the block's first call and between its calls.
    function: Option<String>,
    /// Members written to the open call's arguments object so far.
    written: u32,
    /// Whether a call of the block has ended, so that the bytes between tags are between calls.
    ended: bool,
    /// Where the template puts a value: whether a newline on either side of it is the template's.
    placement: Placement,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before `<function=`.
    Opening,
    /// Inside `<function=…>`: the name is `carried[start..]` until `>`.
    FunctionName { start: usize },
    /// After the function tag or a `</parameter>`.
    Between,
    /// Inside `<parameter=…>`: the key is `carried[start..]` until `>`.
    ParameterName { start: usize },
    /// Inside a value.
    Value(ValueState),
    /// After a call's `</function>`: the block may hold another call, which `<function=` opens.
    AfterCall,
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Whether the newline the template writes after the tag may still come: no value text yet,
    /// under a template that writes one.
    leading: bool,
    mode: Mode,
}

#[derive(Clone, Debug)]
enum Mode {
    /// Pushed piece by piece; `held_newline` says `carried` ends with a newline that may be the
    /// template's.
    Streaming { held_newline: bool },
    /// Written at the close with [`json`]; the text is `carried[start..]`.
    Whole { start: usize },
    /// A string that may be null: written whole if its text is a null word, streamed from the
    /// first byte that rules the words out; the text is `carried[start..]` until then.
    Undecided { start: usize },
}

impl Assembler {
    /// An assembler for the block's calls, the first at `index` and their ids counting from
    /// `ids`, reading the values where the family's template puts them.
    pub fn new(index: u32, ids: CallIds, placement: Placement) -> Self {
        Self {
            index,
            ids,
            scanner: Scanner::new(TAGS),
            carried: String::new(),
            stage: Stage::Opening,
            function: None,
            written: 0,
            ended: false,
            placement,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// Append the next bytes of the block, push the events they complete, and return how many of
    /// the bytes were taken: all of them, since the block's calls follow one another and its
    /// closing marker is the format's to read.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) -> usize {
        let scanned = self.scanner.feed(bytes);
        for piece in scanned.pieces() {
            self.take(piece, declared, out);
        }
        self.scanner.reclaim(scanned);
        bytes.len()
    }

    /// The block the model ended before `</function>`: after the last value the object is closed;
    /// the bytes of a value held for its close, a name or a tag cut short come back as `Malformed`
    /// and the object closes without the member never written; inside a streamed string nothing
    /// is closed, since nothing is made up for a value the model never finished: the fragments
    /// stand, and one `Malformed` event with what was held, or no bytes, says that the arguments
    /// are no whole value before the call's end ([`end_cut_call`]). A block that never named a
    /// function has had its text reported as it came; what is left of it comes back as
    /// `Malformed` here. A block whose calls all closed at their own `</function>` has no call to
    /// end: the whitespace after the last one is the template's, and a tag cut short is reported.
    pub fn close(mut self, out: &mut Events) {
        let held = self.held_tag();
        if matches!(self.stage, Stage::AfterCall) {
            self.carried.push_str(&held);
            self.report_tag(MalformedReason::TagCutShort, out);
            return;
        }
        if !self.started() {
            self.carried.push_str(&held);
            if matches!(self.stage, Stage::Opening) {
                self.leftover(MalformedReason::CallWithoutFunctionTag, out);
            } else {
                self.report_tag(MalformedReason::ClosedBeforeTheName, out);
            }
            return;
        }
        if matches!(self.stage, Stage::Between) {
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.close_object(source, out);
            self.carried = held;
            self.leftover(MalformedReason::TagCutShort, out);
        } else if self.open_string() {
            // The newline held as the template's and the start of a tag ride in the one word; the
            // marker's bytes are the region's.
            self.carried.push_str(&held);
            end_cut_call(self.index, std::mem::take(&mut self.carried), "", out);
            return;
        } else {
            // The call's name is complete: what the close cut short is the key being written, a
            // value held for its close, or a second function tag, a call that closed before its
            // name closed.
            let why = match self.stage {
                Stage::ParameterName { .. } => MalformedReason::KeyCutShort,
                Stage::Value(_) => MalformedReason::ValueCutShort,
                Stage::FunctionName { .. } => MalformedReason::ClosedBeforeTheName,
                // Not reached once the call started (`Between` took the branch above, `Opening`
                // and `AfterCall` have no call open); the match stays exhaustive.
                Stage::Opening | Stage::Between | Stage::AfterCall => MalformedReason::TagCutShort,
            };
            self.carried.push_str(&held);
            self.report_tag(why, out);
            self.close_object(Text::default(), out);
        }
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }

    /// No more bytes will come, and the call never closed. Nothing is closed for the client: an
    /// open string stays open, and a started call ends with the arguments it sent, which are no
    /// whole JSON since the object never closed; one `Malformed` event before its end says so,
    /// with the bytes of a value, a name or a tag cut short, or none ([`end_cut_call`]). A block
    /// that never named a function has had its text reported as it came; what is left of it comes
    /// back as `Malformed` here.
    pub fn finish(mut self, out: &mut Events) {
        let held = self.held_tag();
        self.carried.push_str(&held);
        if self.started() {
            end_cut_call(self.index, std::mem::take(&mut self.carried), "", out);
        } else if matches!(self.stage, Stage::AfterCall) {
            // The block's calls ended at their own `</function>`: the whitespace after the last
            // is the template's, and a tag cut short is what the region never finished.
            self.report_tag(MalformedReason::UnterminatedRegion, out);
        } else {
            self.leftover(MalformedReason::UnterminatedRegion, out);
        }
    }

    /// The whitespace the block ends with after its last call, which the engine drops as the
    /// region's wrapping: taken from the block, so that neither ending says it again. Nothing while
    /// a call is open or before the block's first; what the scanner holds as the start of a tag
    /// stays for the ending to report.
    pub fn trailing_whitespace(&mut self) -> String {
        if matches!(self.stage, Stage::AfterCall) {
            std::mem::take(&mut self.carried)
        } else {
            String::new()
        }
    }

    /// The bytes the scanner held as the start of a tag. At an ending nothing completes them, and
    /// a whole tag is never held, so they are text no event has used.
    fn held_tag(&mut self) -> String {
        self.scanner.held().to_string()
    }

    /// The stage between tags when no call is open: before the block's first call, or after one.
    fn idle(&self) -> Stage {
        if self.ended {
            Stage::AfterCall
        } else {
            Stage::Opening
        }
    }

    /// `</function>` closed the call: the next `<function=` in the block starts the next call, at
    /// the next index, with nothing of this call's left in it.
    fn ended(&mut self) {
        self.index += 1;
        self.function = None;
        self.written = 0;
        self.ended = true;
        self.stage = Stage::AfterCall;
    }

    fn open_string(&self) -> bool {
        matches!(
            &self.stage,
            Stage::Value(ValueState {
                mode: Mode::Streaming { .. },
                ..
            })
        )
    }

    /// The carried bytes as `Malformed`, when there are any.
    fn leftover(&mut self, why: MalformedReason, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        if !text.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(text),
                why,
            });
        }
    }

    /// A tag that is reported, with the carried bytes: the whitespace before it is the template's
    /// and is dropped, the tag's own bytes come back as `Malformed`.
    fn report_tag(&mut self, why: MalformedReason, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        let tag_at = text
            .char_indices()
            .find(|(_, c)| !c.is_whitespace())
            .map_or(text.len(), |(at, _)| at);
        if tag_at > 0 {
            out.push(Event::Dropped {
                text: Text::uncounted(text[..tag_at].to_owned()),
                why: DropReason::Wrapper,
            });
        }
        if tag_at < text.len() {
            out.push(Event::Malformed {
                text: Text::uncounted(text[tag_at..].to_owned()),
                why,
            });
        }
    }

    /// The carried bytes as the template's wrapper, when there are any: they come before text
    /// that is reported on its own, so they cannot wait for a later event's source.
    fn drop_carried(&mut self, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        if !text.is_empty() {
            out.push(Event::Dropped {
                text: Text::uncounted(text),
                why: DropReason::Wrapper,
            });
        }
    }

    fn take(&mut self, piece: Piece<'_>, declared: &Declared, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, declared, out),
            Piece::Marker(tag) => self.tag(tag, declared, out),
        }
    }

    fn tag(&mut self, tag: usize, declared: &Declared, out: &mut Events) {
        let bytes = TAGS[tag];
        match self.stage {
            Stage::Opening | Stage::Between | Stage::AfterCall => self.tag_between(tag, out),
            // A tag cuts the name short: the tag's bytes so far are reported, and the tag is read
            // where the name began.
            Stage::FunctionName { .. } => {
                self.report_tag(MalformedReason::TagCutShort, out);
                self.stage = if self.started() {
                    Stage::Between
                } else {
                    self.idle()
                };
                self.tag_between(tag, out);
            }
            Stage::ParameterName { .. } => {
                self.report_tag(MalformedReason::TagCutShort, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
            Stage::Value(_) if tag == PARAMETER_CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.text(bytes, declared, out),
        }
    }

    /// A tag where the syntax expects one: before the function tag, between parameters, or after a
    /// call, where the next function tag starts the next call.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = TAGS[tag];
        let between = matches!(self.stage, Stage::Between);
        match tag {
            FUNCTION_OPEN => {
                self.carried.push_str(bytes);
                self.stage = Stage::FunctionName {
                    start: self.carried.len(),
                };
            }
            PARAMETER_OPEN if between => {
                self.carried.push_str(bytes);
                self.stage = Stage::ParameterName {
                    start: self.carried.len(),
                };
            }
            FUNCTION_CLOSE if between => {
                let source = Text::uncounted(std::mem::take(&mut self.carried));
                self.close_object(source, out);
                out.push(Event::ToolCallEnd {
                    index: self.index,
                    source: Text::uncounted(bytes),
                });
                self.ended();
            }
            // A close before the block's first function tag: the call it would end never named a
            // function. After a call, a second close is a tag out of its place, below.
            FUNCTION_CLOSE if matches!(self.stage, Stage::Opening) => {
                self.carried.push_str(bytes);
                self.leftover(MalformedReason::CallWithoutFunctionTag, out);
            }
            _ => {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(bytes),
                    why: MalformedReason::TagOutOfPlace,
                });
            }
        }
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        match self.stage {
            Stage::Opening | Stage::Between => {
                self.text_between(text, MalformedReason::TextBetweenTags, out);
            }
            Stage::AfterCall => self.text_between(text, MalformedReason::TextBetweenCalls, out),
            Stage::FunctionName { start } => match text.split_once('>') {
                Some((head, rest)) => {
                    self.carried.push_str(head);
                    let name = self.carried[start..].to_string();
                    self.carried.push('>');
                    self.function_named(name, out);
                    self.text(rest, declared, out);
                }
                None => self.carried.push_str(text),
            },
            Stage::ParameterName { start } => match text.split_once('>') {
                Some((head, rest)) => {
                    self.carried.push_str(head);
                    let key = self.carried[start..].to_string();
                    self.carried.push('>');
                    self.parameter_named(key, declared, out);
                    self.text(rest, declared, out);
                }
                None => self.carried.push_str(text),
            },
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// Text where the syntax has tags, or the block's next call: whitespace is the template's,
    /// anything else is reported with `why`.
    fn text_between(&mut self, text: &str, why: MalformedReason, out: &mut Events) {
        for (run, space) in runs(text) {
            if space {
                self.carried.push_str(run);
            } else {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(run.to_owned()),
                    why,
                });
            }
        }
    }

    /// The function tag is whole: the call starts, unless the name is empty or a call has started
    /// already, in which case the tag is reported.
    fn function_named(&mut self, name: String, out: &mut Events) {
        if name.is_empty() {
            self.report_tag(MalformedReason::EmptyName, out);
            self.stage = if self.started() {
                Stage::Between
            } else {
                self.idle()
            };
        } else if self.started() {
            self.report_tag(MalformedReason::SecondFunctionTag, out);
            self.stage = Stage::Between;
        } else {
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.ids.id(self.index),
                name: name.clone(),
                source: Text::uncounted(std::mem::take(&mut self.carried)),
            });
            self.function = Some(name);
            self.stage = Stage::Between;
        }
    }

    /// The parameter's tag is whole: the value begins. A declared string opens its fragment now.
    fn parameter_named(&mut self, key: String, declared: &Declared, out: &mut Events) {
        if key.is_empty() {
            // The engine's parser keeps an empty parameter name and writes `"": value`; here it is reported.
            self.report_tag(MalformedReason::EmptyName, out);
            self.stage = Stage::Between;
            return;
        }
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => {
                self.open_string_fragment(&key, out);
                Mode::Streaming {
                    held_newline: false,
                }
            }
            Some(Kind::NullableString) => Mode::Undecided {
                start: self.carried.len(),
            },
            Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole {
                start: self.carried.len(),
            },
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            leading: self.placement == Placement::OwnLine,
            mode,
        });
    }

    /// `{"key": "` or `, "key": "`, with the carried bytes as its source.
    fn open_string_fragment(&mut self, key: &str, out: &mut Events) {
        let mut opening = String::with_capacity(key.len() + 8);
        opening.push_str(self.separator());
        push_quoted(&mut opening, key);
        opening.push_str(": \"");
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(opening, source, out);
        self.written += 1;
    }

    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        let mut text = text;
        if value.leading {
            value.leading = false;
            if let Some(rest) = text.strip_prefix('\n') {
                self.carried.push('\n');
                if let Mode::Whole { start } | Mode::Undecided { start } = &mut value.mode {
                    *start = self.carried.len();
                }
                text = rest;
            }
        }
        if text.is_empty() {
            // The piece was the template's newline alone.
            return;
        }
        match value.mode {
            Mode::Streaming { .. } => self.stream(text, out),
            Mode::Whole { .. } => self.carried.push_str(text),
            Mode::Undecided { start } => {
                self.carried.push_str(text);
                let so_far = self.whole_text(start);
                if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
                    self.open_string_late(start, out);
                }
            }
        }
    }

    /// A string that may have been null is not: open it now, then stream what has arrived.
    fn open_string_late(&mut self, start: usize, out: &mut Events) {
        let arrived = self.carried.split_off(start);
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let key = value.key.clone();
        value.mode = Mode::Streaming {
            held_newline: false,
        };
        self.open_string_fragment(&key, out);
        self.stream(&arrived, out);
    }

    /// A piece of a streamed string. Under a template that writes a newline before the closing
    /// tag, a newline the piece ends with is held, since it may be the template's, and a newline
    /// held before it turns out to be the value's; under one that writes none, every byte is
    /// the value's and goes out at once.
    fn stream(&mut self, text: &str, out: &mut Events) {
        let own_line = self.placement == Placement::OwnLine;
        let Stage::Value(ValueState {
            mode: Mode::Streaming { held_newline },
            ..
        }) = &mut self.stage
        else {
            return;
        };
        let had_newline = *held_newline;
        let (body, ends_with_newline) = match text.strip_suffix('\n') {
            Some(body) if own_line => (body, true),
            _ => (text, false),
        };
        *held_newline = ends_with_newline;
        if body.is_empty() && !had_newline {
            // The piece is one newline; it waits with the carried bytes for the piece that follows.
            self.carried.push('\n');
            return;
        }
        let mut json = String::with_capacity(body.len() + 2);
        if had_newline {
            json.push_str("\\n");
        }
        push_escaped(&mut json, body);
        // The source: the carried bytes (the template's newline after the tag, or the newline that
        // was held and is now the value's) and the piece, less the newline held in its turn.
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(body);
        if ends_with_newline {
            self.carried.push('\n');
        }
        self.push_fragment(json, Text::uncounted(source), out);
    }

    /// `</parameter>`: a streamed string gets its closing quote, any other value is written whole.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = match value.mode {
            Mode::Streaming { .. } => "\"".to_string(),
            Mode::Whole { start } | Mode::Undecided { start } => {
                let written = json(self.whole_text(start), value.kind);
                let mut member = String::with_capacity(value.key.len() + written.len() + 8);
                member.push_str(self.separator());
                push_quoted(&mut member, &value.key);
                member.push_str(": ");
                member.push_str(&written);
                self.written += 1;
                member
            }
        };
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(fragment, Text::uncounted(source), out);
        self.stage = Stage::Between;
    }

    /// The text of a value written whole: what came after the tag and its newline, less the
    /// newline the template writes before `</parameter>`, where the template writes one.
    fn whole_text(&self, start: usize) -> &str {
        let text = &self.carried[start..];
        match self.placement {
            Placement::OwnLine => text.strip_suffix('\n').unwrap_or(text),
            Placement::Direct => text,
        }
    }

    /// `, ` before every member but the first, which opens the object.
    fn separator(&self) -> &'static str {
        if self.written == 0 {
            "{"
        } else {
            ", "
        }
    }

    /// `}` after the members, or `{}` when there were none.
    fn close_object<'d>(&mut self, source: Text<'d>, out: &mut Events<'d>) {
        let closing = if self.written == 0 { "{}" } else { "}" };
        self.push_fragment(closing.to_string(), source, out);
    }

    fn push_fragment<'d>(&self, json: String, source: Text<'d>, out: &mut Events<'d>) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

/// Appends `text` as a JSON string, quotes included.
pub(crate) fn push_quoted(out: &mut String, text: &str) {
    out.push('"');
    push_escaped(out, text);
    out.push('"');
}

/// Appends `text` as the inside of a JSON string, escaped as `serde_json` escapes it: the quote,
/// the backslash and the control characters, nothing else.
pub(crate) fn push_escaped(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => {
                out.push_str("\\u00");
                out.push(char::from_digit(u32::from(c) >> 4, 16).unwrap_or('0'));
                out.push(char::from_digit(u32::from(c) & 0xf, 16).unwrap_or('0'));
            }
            c => out.push(c),
        }
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out)
    }

    fn trailing_whitespace(&mut self) -> String {
        self.trailing_whitespace()
    }

    /// The region's marker closes the call for the client; it belongs to the region, not to the
    /// call.
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
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json as value, Value};

    use super::*;
    use crate::testing::bytes;

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    /// A tool `f` with a string `city`, an integer `limit`, a nullable string `note` and nothing
    /// declared for `extra`.
    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "limit": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    const CALL: &str = concat!(
        "\n<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5\n</parameter>",
        "\n<parameter=note>\nNo rain\n</parameter>\n</function>"
    );
    const CALL_ARGUMENTS: &str = r#"{"city": "Paris", "limit": 5, "note": "No rain"}"#;

    /// Feeds the pieces to an assembler reading values at `placement`, then the ending asked
    /// for, and returns the events with how many bytes each feed took.
    fn run(
        pieces: &[&str],
        placement: Placement,
        ending: fn(Assembler, &mut Events),
    ) -> (Vec<Event<'static>>, Vec<usize>) {
        let declared = declared();
        let mut assembler = Assembler::new(0, ids(), placement);
        let mut out = Events::new();
        let mut taken = Vec::new();
        for piece in pieces {
            taken.push(assembler.feed(piece, &declared, &mut out));
        }
        ending(assembler, &mut out);
        (out.drain(), taken)
    }

    /// The values on a line of their own, Qwen 3.5's way, and the stream cut at the end.
    fn finished(pieces: &[&str]) -> Vec<Event<'static>> {
        run(pieces, Placement::OwnLine, Assembler::finish).0
    }

    /// The values between the tags directly, Seed-OSS's and MiMo's way, and the stream cut.
    fn finished_direct(pieces: &[&str]) -> Vec<Event<'static>> {
        run(pieces, Placement::Direct, Assembler::finish).0
    }

    fn closed(pieces: &[&str]) -> Vec<Event<'static>> {
        run(pieces, Placement::OwnLine, Assembler::close).0
    }

    /// The arguments as the client sees them: every fragment, concatenated.
    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn name_of<'a>(events: &'a [Event<'_>]) -> Option<&'a str> {
        events.iter().find_map(|event| match event {
            Event::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
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

    fn dropped(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.to_string()),
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
            .count()
    }

    fn fragment(json: &str, source: &str) -> Event<'static> {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(source.to_owned()),
        }
    }

    fn end(source: &str) -> Event<'static> {
        Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source.to_owned()),
        }
    }

    /// No fragment is empty and no `Malformed` or `Dropped` text is empty.
    fn nothing_empty(events: &[Event]) {
        for event in events {
            match event {
                Event::ToolCallArguments { json, .. } => assert!(!json.is_empty(), "{event:?}"),
                Event::Malformed { text, .. } | Event::Dropped { text, .. } => {
                    assert!(!text.text.is_empty(), "{event:?}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn a_call_streams_its_strings_and_writes_its_integer_at_the_close() {
        let (events, taken) = run(&[CALL], Placement::OwnLine, Assembler::finish);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: ids().id(0),
                    name: "f".to_string(),
                    source: Text::uncounted("\n<function=f>"),
                },
                fragment("{\"city\": \"", "\n<parameter=city>"),
                fragment("Paris", "\nParis"),
                fragment("\"", "\n</parameter>"),
                fragment(", \"limit\": 5", "\n<parameter=limit>\n5\n</parameter>"),
                fragment(", \"note\": \"", "\n<parameter=note>\n"),
                fragment("No rain", "No rain"),
                fragment("\"", "\n</parameter>"),
                fragment("}", "\n"),
                end("</function>"),
            ]
        );
        assert_eq!(arguments(&events), CALL_ARGUMENTS);
        assert_eq!(bytes(&events), CALL);
        assert_eq!(taken, vec![CALL.len()]);
    }

    #[test]
    fn every_chunking_gives_the_same_arguments_and_accounts_for_every_byte() {
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let (events, taken) = run(
                &[&CALL[..cut], &CALL[cut..]],
                Placement::OwnLine,
                Assembler::finish,
            );
            assert_eq!(arguments(&events), CALL_ARGUMENTS, "cut at {cut}");
            assert_eq!(bytes(&events), CALL, "cut at {cut}");
            assert_eq!(name_of(&events), Some("f"), "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), CALL.len(), "cut at {cut}");
            assert!(malformed(&events).is_empty(), "cut at {cut}");
            nothing_empty(&events);
        }
        // Byte by byte as well.
        let pieces: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let (events, _) = run(&pieces, Placement::OwnLine, Assembler::finish);
        assert_eq!(arguments(&events), CALL_ARGUMENTS);
        assert_eq!(bytes(&events), CALL);
        nothing_empty(&events);
    }

    #[test]
    fn a_string_keeps_its_own_newlines_and_loses_the_templates_two() {
        let call = "<function=f>\n<parameter=city>\n\nTwo\nlines\n\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"city": "\nTwo\nlines\n"}"#);
        assert_eq!(bytes(&events), call);
        // The same with the value's text split after each newline, where the hold-back decides.
        let pieces = [
            "<function=f>\n<parameter=city>",
            "\n",
            "\n",
            "Two\n",
            "lines\n",
            "\n",
            "</parameter>\n</function>",
        ];
        let events = finished(&pieces);
        assert_eq!(arguments(&events), r#"{"city": "\nTwo\nlines\n"}"#);
        assert_eq!(bytes(&events), call);
        nothing_empty(&events);
    }

    #[test]
    fn a_value_without_the_templates_newlines_is_read_as_written() {
        let call = concat!(
            "<function=f><parameter=city>Paris</parameter>",
            "<parameter=limit>5</parameter></function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"city": "Paris", "limit": 5}"#);
        assert_eq!(bytes(&events), call);
        let empty = "<function=f>\n<parameter=city>\n</parameter>\n</function>";
        assert_eq!(arguments(&finished(&[empty])), r#"{"city": ""}"#);
    }

    #[test]
    fn a_type_beside_an_enum_decides_and_an_array_stays_one() {
        // bellwether's qwen3-coder-next/parse/bfcl-live-multiple-146-58-0: BFCL declares the list
        // `{"type": "array", "items": {"type": "string"}, "enum": [...]}`, and the engine's parser reads it as an
        // array; before this the enum made it a string holding `[]`.
        let declared = Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["temperature", "humidity"],
                    },
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]);
        let call = "<function=f>\n<parameter=metrics>\n[]\n</parameter>\n</function>";
        let mut assembler = Assembler::new(0, ids(), Placement::OwnLine);
        let mut out = Events::new();
        assembler.feed(call, &declared, &mut out);
        assembler.finish(&mut out);
        let events = out.drain();
        assert_eq!(arguments(&events), r#"{"metrics": []}"#);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_string_that_may_be_null_streams_once_its_text_rules_null_out() {
        for (text, expected) in [
            ("null", "null"),
            ("None", "null"),
            ("Paris", r#""Paris""#),
            ("Nonesuch", r#""Nonesuch""#),
            ("nul", r#""nul""#),
            ("", r#""""#),
        ] {
            let call = format!("<function=f>\n<parameter=note>\n{text}\n</parameter>\n</function>");
            let events = finished(&[&call]);
            assert_eq!(
                arguments(&events),
                format!(r#"{{"note": {expected}}}"#),
                "{text:?}"
            );
            assert_eq!(bytes(&events), call, "{text:?}");
        }
        // `No` is held as a possible `None`; `tes` settles it, and the string streams from there.
        let events = finished(&[
            "<function=f>\n<parameter=note>\nNo",
            "tes",
            " here\n</parameter>\n</function>",
        ]);
        assert_eq!(
            events[1..5],
            [
                fragment("{\"note\": \"", "\n<parameter=note>\n"),
                fragment("Notes", "Notes"),
                fragment(" here", " here"),
                fragment("\"", "\n</parameter>"),
            ]
        );
    }

    #[test]
    fn an_undeclared_value_is_inferred_and_a_second_member_opens_with_a_comma() {
        let call = concat!(
            "<function=f>\n<parameter=extra>\n{\"a\": [1, 2]}\n</parameter>",
            "\n<parameter=limit>\n007\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"extra": {"a": [1, 2]}, "limit": 7}"#
        );
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_call_without_parameters_is_an_empty_object() {
        let call = "\n<function=f>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(events[1..], [fragment("{}", "\n"), end("</function>")]);
    }

    #[test]
    fn a_function_the_request_did_not_declare_has_every_value_inferred() {
        let call = concat!(
            "<function=g>\n<parameter=city>\nParis\n</parameter>",
            "\n<parameter=n>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("g"));
        assert_eq!(arguments(&events), r#"{"city": "Paris", "n": 5}"#);
    }

    #[test]
    fn keys_and_text_are_escaped_as_json() {
        let call = concat!(
            "<function=f>\n<parameter=a\"b>\nx\n</parameter>",
            "\n<parameter=city>\nsay \"hi\" \\ tab\tend \u{1}\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"a\"b": "x", "city": "say \"hi\" \\ tab\tend \u0001"}"#
        );
        assert!(serde_json::from_str::<Value>(&arguments(&events)).is_ok());
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn the_escaping_is_serde_jsons() {
        let texts = [
            "plain",
            "quote \" backslash \\ slash / tab \t newline \n return \r",
            "\u{8}\u{c}\u{1}\u{1f}\u{7f}",
            "Café 🌍 \u{200d} \u{10ffff}",
            "",
        ];
        for text in texts {
            let mut escaped = String::new();
            push_escaped(&mut escaped, text);
            let expected = serde_json::to_string(text).unwrap_or_default();
            assert_eq!(escaped, expected[1..expected.len() - 1], "{text:?}");
        }
    }

    #[test]
    fn tags_inside_a_value_are_its_text() {
        let call = concat!(
            "<function=f>\n<parameter=city>\na <function=x> b <parameter=y> c </function>",
            "\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"city": "a <function=x> b <parameter=y> c </function>"}"#
        );
        assert_eq!(bytes(&events), call);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_second_function_block_in_the_block_is_the_next_call() {
        // A parallel answer writes its calls one after another between one pair of call markers,
        // and the engines' own parsers read every block as a call. The second starts at its
        // `<function=` with the next index and the id that index mints; the whitespace between
        // the blocks is its source, and the whitespace after the last block is the template's.
        let block = "<function=f>\n</function>\n<function=f>\n<parameter=limit>\n5\n</parameter>\n\
                     </function>\n";
        let start = |index: u32, source: &str| Event::ToolCallStart {
            index,
            id: ids().id(index),
            name: "f".to_string(),
            source: Text::uncounted(source.to_owned()),
        };
        let fragment_of = |index: u32, json: &str, source: &str| Event::ToolCallArguments {
            index,
            json: json.to_string(),
            source: Text::uncounted(source.to_owned()),
        };
        let end_of = |index: u32| Event::ToolCallEnd {
            index,
            source: Text::uncounted("</function>"),
        };
        let expected = vec![
            start(3, "<function=f>"),
            fragment_of(3, "{}", "\n"),
            end_of(3),
            start(4, "\n<function=f>"),
            fragment_of(4, "{\"limit\": 5", "\n<parameter=limit>\n5\n</parameter>"),
            fragment_of(4, "}", "\n"),
            end_of(4),
            Event::Dropped {
                text: Text::uncounted("\n"),
                why: DropReason::Wrapper,
            },
        ];
        let declared = declared();
        let mut cuts: Vec<Vec<&str>> = vec![vec![block]];
        cuts.extend((1..block.len()).map(|cut| vec![&block[..cut], &block[cut..]]));
        cuts.push(
            block
                .char_indices()
                .map(|(at, c)| &block[at..at + c.len_utf8()])
                .collect(),
        );
        for pieces in cuts {
            let mut assembler = Assembler::new(3, ids(), Placement::OwnLine);
            let mut out = Events::new();
            for piece in &pieces {
                assert_eq!(assembler.feed(piece, &declared, &mut out), piece.len());
            }
            assembler.close(&mut out);
            let events = out.drain();
            assert_eq!(arguments(&events), "{}{\"limit\": 5}", "{pieces:?}");
            assert_eq!(bytes(&events), block, "{pieces:?}");
            assert_eq!(ends(&events), 2, "{pieces:?}");
            assert!(malformed(&events).is_empty(), "{pieces:?}");
            nothing_empty(&events);
            if pieces.len() == 1 {
                assert_eq!(events, expected);
            }
        }
    }

    #[test]
    fn after_a_call_text_is_between_calls_and_a_stray_or_cut_tag_is_reported() {
        // Prose after a complete call, and a tag the block's close cut short.
        let block = "<function=f>\n</function>\nstray words\n<fun";
        let events = closed(&[block]);
        assert_eq!(ends(&events), 1);
        assert_eq!(
            malformed(&events),
            vec![
                ("stray".to_string(), MalformedReason::TextBetweenCalls),
                ("words".to_string(), MalformedReason::TextBetweenCalls),
                ("<fun".to_string(), MalformedReason::TagCutShort),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", " ", "\n"]);
        assert_eq!(bytes(&events), block);
        nothing_empty(&events);
        // The stream cut after the last call: the newline is the template's, a cut tag is what
        // the region never finished.
        let events = finished(&["<function=f>\n</function>\n<fun"]);
        assert_eq!(
            malformed(&events),
            vec![("<fun".to_string(), MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        // Whitespace alone after the last call is the template's at either ending, and nothing
        // after `</function>` adds nothing.
        for ending in [finished, closed] {
            let events = ending(&["<function=f>\n</function>\n"]);
            assert!(malformed(&events).is_empty());
            assert_eq!(dropped(&events), vec!["\n"]);
            assert_eq!(ends(&events), 1);
            let events = ending(&["<function=f>\n</function>"]);
            assert_eq!(events.last(), Some(&end("</function>")));
        }
        // A second close and a parameter tag with no call open are out of their place.
        let events =
            closed(&["<function=f>\n</function>\n</function>\n<parameter=x>y</parameter>"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("</function>".to_string(), MalformedReason::TagOutOfPlace),
                ("<parameter=".to_string(), MalformedReason::TagOutOfPlace),
                ("x>y".to_string(), MalformedReason::TextBetweenCalls),
                ("</parameter>".to_string(), MalformedReason::TagOutOfPlace),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", "\n"]);
        assert_eq!(ends(&events), 1);
    }

    #[test]
    fn the_whitespace_after_the_last_call_is_handed_back_and_the_endings_add_nothing_for_it() {
        let declared = declared();
        for ending in [Assembler::close, Assembler::finish] {
            let mut assembler = Assembler::new(0, ids(), Placement::OwnLine);
            let mut out = Events::new();
            assembler.feed("<function=f>\n</function>\n \n", &declared, &mut out);
            assert_eq!(assembler.trailing_whitespace(), "\n \n");
            assert_eq!(assembler.trailing_whitespace(), "");
            let before = out.len();
            ending(assembler, &mut out);
            assert_eq!(out.len(), before, "{:?}", out.as_slice());
        }
        // A tag the end cut short stays for the ending; before the first call, and while a call is
        // open, nothing is handed back.
        let mut assembler = Assembler::new(0, ids(), Placement::OwnLine);
        let mut out = Events::new();
        assembler.feed("<function=f>\n</function>\n<fun", &declared, &mut out);
        assert_eq!(assembler.trailing_whitespace(), "\n");
        assembler.close(&mut out);
        assert_eq!(
            malformed(&out.drain()),
            vec![("<fun".to_string(), MalformedReason::TagCutShort)]
        );
        let mut assembler = Assembler::new(0, ids(), Placement::OwnLine);
        assembler.feed("\n", &declared, &mut out);
        assert_eq!(assembler.trailing_whitespace(), "");
        assembler.feed("<function=f>\n", &declared, &mut out);
        assert_eq!(assembler.trailing_whitespace(), "");
    }

    #[test]
    fn a_close_before_the_first_function_tag_leaves_the_block_open_for_its_call() {
        let block = "</function>\n<function=f>\n</function>";
        let events = finished(&[block]);
        assert_eq!(
            malformed(&events),
            vec![(
                "</function>".to_string(),
                MalformedReason::CallWithoutFunctionTag
            )]
        );
        assert_eq!(name_of(&events), Some("f"));
        assert_eq!(arguments(&events), "{}");
        assert_eq!(ends(&events), 1);
        assert_eq!(bytes(&events), block);
    }

    #[test]
    fn text_between_tags_is_malformed_and_the_whitespace_before_it_is_dropped() {
        // A parameter tag the model misspelt is reported, not read as a parameter and not hidden.
        let call = concat!(
            "<function=f>\n<parameter city>\nParis\n</parameter>",
            "\n<parameter=limit>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter".to_string(), MalformedReason::TextBetweenTags),
                ("city>".to_string(), MalformedReason::TextBetweenTags),
                ("Paris".to_string(), MalformedReason::TextBetweenTags),
                ("</parameter>".to_string(), MalformedReason::TagOutOfPlace),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", " ", "\n", "\n"]);
        assert_eq!(bytes(&events), call);
        nothing_empty(&events);
        // Prose before the function tag is reported too; the whitespace before the tag stays its
        // source.
        let events = finished(&["I will call it now <function=f>\n</function>"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("I".to_string(), MalformedReason::TextBetweenTags),
                ("will".to_string(), MalformedReason::TextBetweenTags),
                ("call".to_string(), MalformedReason::TextBetweenTags),
                ("it".to_string(), MalformedReason::TextBetweenTags),
                ("now".to_string(), MalformedReason::TextBetweenTags),
            ]
        );
        assert_eq!(
            events
                .iter()
                .find(|e| matches!(e, Event::ToolCallStart { .. })),
            Some(&Event::ToolCallStart {
                index: 0,
                id: ids().id(0),
                name: "f".to_string(),
                source: Text::uncounted(" <function=f>"),
            })
        );
        assert_eq!(
            bytes(&events),
            "I will call it now <function=f>\n</function>"
        );
    }

    #[test]
    fn a_tag_inside_a_name_cuts_the_name_short() {
        // `</function>` inside the function's name closes a call that named no function; the
        // block goes on, and what follows is text between its tags.
        let piece = "<function=f</function>rest";
        let (events, taken) = run(&[piece], Placement::OwnLine, Assembler::finish);
        assert_eq!(name_of(&events), None);
        assert_eq!(
            malformed(&events),
            vec![
                ("<function=f".to_string(), MalformedReason::TagCutShort),
                (
                    "</function>".to_string(),
                    MalformedReason::CallWithoutFunctionTag
                ),
                ("rest".to_string(), MalformedReason::TextBetweenTags),
            ]
        );
        assert_eq!(taken, vec![piece.len()]);
        assert_eq!(ends(&events), 0);
        // Inside a parameter's key, the call goes on and `</function>` ends it.
        let call = "<function=f>\n<parameter=a</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![("<parameter=a".to_string(), MalformedReason::TagCutShort)]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(events.last(), Some(&end("</function>")));
        assert_eq!(bytes(&events), call);
        // A parameter tag inside the key starts the next parameter.
        let call = "<function=f>\n<parameter=a<parameter=limit>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn an_empty_name_is_malformed() {
        let events = finished(&["<function=>\n<parameter=limit>\n5\n</parameter>\n</function>"]);
        assert_eq!(name_of(&events), None);
        assert_eq!(
            malformed(&events)[0],
            ("<function=>".to_string(), MalformedReason::EmptyName)
        );
        assert_eq!(arguments(&events), "");
        let call = concat!(
            "<function=f>\n<parameter=>\nx\n</parameter>",
            "\n<parameter=limit>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter=>".to_string(), MalformedReason::EmptyName),
                ("x".to_string(), MalformedReason::TextBetweenTags),
                ("</parameter>".to_string(), MalformedReason::TagOutOfPlace),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", "\n", "\n"]);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_block_that_closed_before_its_function_tag_closed_gets_its_arguments_closed() {
        // Inside a value written whole: its key was never written, so its bytes come back.
        let events =
            closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "<parameter=limit>\n5".to_string(),
                MalformedReason::ValueCutShort
            )]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        // A string that may still be null is a value written whole.
        let events = closed(&["<function=f>\n<parameter=note>\nnul"]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![(
                "<parameter=note>\nnul".to_string(),
                MalformedReason::ValueCutShort
            )]
        );
        // Inside a key, the key the end cut short; a tag the scanner held between parameters, the tag.
        let events = closed(&["<function=f>\n<parameter=ci"]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![("<parameter=ci".to_string(), MalformedReason::KeyCutShort)]
        );
        let events = closed(&["<function=f>\n<param"]);
        assert_eq!(
            events[1..],
            [
                fragment("{}", "\n"),
                Event::Malformed {
                    text: Text::uncounted("<param"),
                    why: MalformedReason::TagCutShort,
                },
                end(""),
            ]
        );
        // Without any parameter: an empty object, with the whitespace as its source.
        let events = closed(&["<function=f>\n"]);
        assert_eq!(events[1..], [fragment("{}", "\n"), end("")]);
        assert_eq!(bytes(&events), "<function=f>\n");
    }

    /// A block the model ended inside a declared string value (its closing marker came, or the
    /// next call began, before `</parameter>`): the fragments sent stand, no closing quote and no
    /// closing brace are made up, and one `Malformed` event before the call's end says that the
    /// arguments are no whole value, carrying what was held (the template's newline, the start
    /// of a tag) or no bytes, under either placement and however the bytes are cut. A call whose
    /// value closed needs no word.
    #[test]
    fn a_block_that_closed_inside_a_streamed_string_leaves_it_open_and_says_so() {
        let said_cut = |events: &[Event], held: &str| {
            matches!(
                &events[events.len() - 2..],
                [
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { source, .. }
                ] if text.text == held && source.text.is_empty()
            )
        };
        for (call, placement, held) in [
            (
                "<function=f>\n<parameter=city>\nPar\n",
                Placement::OwnLine,
                "\n",
            ),
            (
                "<function=f>\n<parameter=city>\nPar",
                Placement::OwnLine,
                "",
            ),
            ("<function=f><parameter=city>Par", Placement::Direct, ""),
        ] {
            let whole = run(&[call], placement, Assembler::close).0;
            assert_eq!(arguments(&whole), r#"{"city": "Par"#, "{call:?}");
            assert!(said_cut(&whole, held), "{call:?}: {whole:?}");
            assert_eq!(malformed(&whole).len(), 1, "{call:?}");
            assert!(dropped(&whole).is_empty(), "{call:?}");
            assert_eq!(bytes(&whole), call, "{call:?}");
            let per_char: Vec<&str> = call
                .char_indices()
                .map(|(i, c)| &call[i..i + c.len_utf8()])
                .collect();
            let streamed = run(&per_char, placement, Assembler::close).0;
            assert_eq!(
                arguments(&streamed),
                r#"{"city": "Par"#,
                "{call:?} per character"
            );
            assert!(
                said_cut(&streamed, held),
                "{call:?} per character: {streamed:?}"
            );
            assert_eq!(bytes(&streamed), call, "{call:?} per character");
        }
        // The start of a tag held at the close rides in the one word, after the newline held
        // before it.
        let held = closed(&["<function=f>\n<parameter=city>\nPar", "is\n</param"]);
        assert_eq!(arguments(&held), r#"{"city": "Paris"#);
        assert!(said_cut(&held, "\n</param"), "{held:?}");
        assert_eq!(
            bytes(&held),
            "<function=f>\n<parameter=city>\nParis\n</param"
        );
        assert_eq!(ends(&held), 1);
        // The value's own close makes it whole: no word, however the block ended.
        let closed = closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n"]);
        assert_eq!(arguments(&closed), r#"{"city": "Paris"}"#);
        assert!(malformed(&closed).is_empty());
    }

    #[test]
    fn a_cut_stream_leaves_an_open_string_open_and_returns_what_was_held() {
        let events = finished(&["<function=f>\n<parameter=city>\nPar", "is\n</param"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"#);
        assert_eq!(
            malformed(&events),
            vec![("\n</param".to_string(), MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(
            bytes(&events),
            "<function=f>\n<parameter=city>\nParis\n</param"
        );
        assert_eq!(events.last(), Some(&end("")));
        // A value written whole, cut short.
        let events = finished(&["<function=f>\n<parameter=limit>\n5"]);
        assert_eq!(arguments(&events), "");
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<parameter=limit>\n5".to_string(),
                MalformedReason::UnterminatedRegion
            )]
        );
        // A started call with no parameter still ends, and only what arrived is reported.
        let events = finished(&["<function=f>\n"]);
        assert_eq!(
            events[1..],
            [
                Event::Malformed {
                    text: Text::uncounted("\n"),
                    why: MalformedReason::UnterminatedRegion
                },
                end(""),
            ]
        );
        // Cut right after the function tag, nothing held: the word before the end has no bytes.
        let events = finished(&["<function=f>"]);
        assert_eq!(
            events[1..],
            [
                Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::UnterminatedRegion
                },
                end(""),
            ]
        );
    }

    /// A call the stream's end cuts inside a declared string value: the fragments sent stand,
    /// nothing is made up, and one `Malformed` event of no bytes before the call's end says that
    /// the arguments are no whole value, under either placement and however the bytes are cut.
    /// Bytes still held at the cut ride in that one event; a call that closed needs no word.
    #[test]
    fn a_call_cut_inside_a_string_value_says_so_before_its_end() {
        let said_cut = |events: &[Event]| {
            matches!(
                &events[events.len() - 2..],
                [
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { .. }
                ] if text.text.is_empty()
            )
        };
        for (call, placement) in [
            ("<function=f>\n<parameter=city>\nPar", Placement::OwnLine),
            ("<function=f><parameter=city>Par", Placement::Direct),
        ] {
            let whole = run(&[call], placement, Assembler::finish).0;
            assert_eq!(arguments(&whole), r#"{"city": "Par"#, "{call:?}");
            assert!(said_cut(&whole), "{call:?}: {whole:?}");
            assert_eq!(malformed(&whole).len(), 1, "{call:?}");
            assert_eq!(bytes(&whole), call, "{call:?}");
            let per_char: Vec<&str> = call
                .char_indices()
                .map(|(i, c)| &call[i..i + c.len_utf8()])
                .collect();
            let streamed = run(&per_char, placement, Assembler::finish).0;
            assert_eq!(
                arguments(&streamed),
                r#"{"city": "Par"#,
                "{call:?} per character"
            );
            assert!(said_cut(&streamed), "{call:?} per character: {streamed:?}");
            assert_eq!(malformed(&streamed).len(), 1, "{call:?} per character");
            assert_eq!(bytes(&streamed), call, "{call:?} per character");
        }
        // A newline held at the cut, which may have been the template's, rides in the one word.
        let held = finished(&["<function=f>\n<parameter=city>\nPar\n"]);
        assert_eq!(arguments(&held), r#"{"city": "Par"#);
        assert_eq!(
            malformed(&held),
            vec![("\n".to_string(), MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(held.last(), Some(&end("")));
        // The call's own close makes the arguments whole: no word.
        let closed = closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n</function>"]);
        assert_eq!(arguments(&closed), r#"{"city": "Paris"}"#);
        assert!(malformed(&closed).is_empty());
    }

    #[test]
    fn a_block_without_a_function_tag_reports_every_byte() {
        // A JSON object where tags were expected: its words are text between tags, the whitespace
        // around them the wrapper, and what is left at the end never closed.
        let events = finished(&["\n{\"name\": \"f\"}\n"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("{\"name\":".to_string(), MalformedReason::TextBetweenTags),
                ("\"f\"}".to_string(), MalformedReason::TextBetweenTags),
                ("\n".to_string(), MalformedReason::UnterminatedRegion),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", " "]);
        assert_eq!(ends(&events), 0);
        assert_eq!(bytes(&events), "\n{\"name\": \"f\"}\n");
        // A function tag cut short by the end of the stream.
        let events = finished(&["\n<function=f"]);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<function=f".to_string(),
                MalformedReason::UnterminatedRegion
            )]
        );
        assert_eq!(events.len(), 1);
        let events = closed(&["\n<parameter=city>\nParis\n"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter=".to_string(), MalformedReason::TagOutOfPlace),
                ("city>".to_string(), MalformedReason::TextBetweenTags),
                ("Paris".to_string(), MalformedReason::TextBetweenTags),
                ("\n".to_string(), MalformedReason::CallWithoutFunctionTag),
            ]
        );
        assert_eq!(bytes(&events), "\n<parameter=city>\nParis\n");
        let events = closed(&["\n"]);
        assert_eq!(
            malformed(&events),
            vec![("\n".to_string(), MalformedReason::CallWithoutFunctionTag)]
        );
        assert!(closed(&[]).is_empty());
        // Held bytes at the ending are reported too, with the whitespace before them.
        let events = closed(&["\n<para"]);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<para".to_string(),
                MalformedReason::CallWithoutFunctionTag
            )]
        );
        let events = closed(&["<function=f</fun"]);
        assert_eq!(
            malformed(&events),
            vec![(
                "<function=f</fun".to_string(),
                MalformedReason::ClosedBeforeTheName
            )]
        );
        assert_eq!(ends(&events), 0);
        // `</function>` with no function open closes a call that named none; the block goes on,
        // and what follows is text between its tags.
        let piece = "\nx</function>rest";
        let (events, taken) = run(&[piece], Placement::OwnLine, Assembler::finish);
        assert_eq!(
            malformed(&events),
            vec![
                ("x".to_string(), MalformedReason::TextBetweenTags),
                (
                    "</function>".to_string(),
                    MalformedReason::CallWithoutFunctionTag
                ),
                ("rest".to_string(), MalformedReason::TextBetweenTags),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(taken, vec![piece.len()]);
    }

    #[test]
    fn a_second_function_tag_inside_a_call_is_reported_and_the_call_goes_on() {
        let call = "<function=f>\n<function=g>\n<parameter=limit>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("f"));
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "<function=g>".to_string(),
                MalformedReason::SecondFunctionTag
            )]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(bytes(&events), call);
        // The block ending inside the second function tag still ends the call, closed.
        let events = closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n<function=g"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "<function=g".to_string(),
                MalformedReason::ClosedBeforeTheName
            )]
        );
        assert_eq!(events.last(), Some(&end("")));
        assert_eq!(ends(&events), 1);
        let events = closed(&["<function=f>\n<function="]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(ends(&events), 1);
    }

    #[test]
    fn a_report_with_no_whitespace_before_it_drops_nothing() {
        let events = finished(&["<function=f>x<function=g>y</function>"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("x".to_string(), MalformedReason::TextBetweenTags),
                (
                    "<function=g>".to_string(),
                    MalformedReason::SecondFunctionTag
                ),
                ("y".to_string(), MalformedReason::TextBetweenTags),
            ]
        );
        assert!(dropped(&events).is_empty());
        nothing_empty(&events);
        // Whitespace outside ASCII counts as the template's too.
        let events = finished(&["<function=f>\u{3000}\u{a0}x</function>"]);
        assert_eq!(dropped(&events), vec!["\u{3000}\u{a0}"]);
        assert_eq!(
            malformed(&events),
            vec![("x".to_string(), MalformedReason::TextBetweenTags)]
        );
    }

    #[test]
    fn a_string_that_may_be_null_is_read_without_the_templates_newlines_too() {
        for (text, expected) in [
            ("null", "null"),
            ("Paris", r#""Paris""#),
            ("nul", r#""nul""#),
        ] {
            let call = format!("<function=f><parameter=note>{text}</parameter></function>");
            let events = finished(&[&call]);
            assert_eq!(
                arguments(&events),
                format!(r#"{{"note": {expected}}}"#),
                "{text:?}"
            );
            assert_eq!(bytes(&events), call, "{text:?}");
        }
    }

    #[test]
    fn a_closing_parameter_tag_between_parameters_is_out_of_place() {
        let call = "<function=f>\n<parameter=limit>\n5\n</parameter>\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![("</parameter>".to_string(), MalformedReason::TagOutOfPlace)]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn whitespace_between_tags_is_carried_into_the_next_events_source() {
        let call = "<function=f>  \n\n<parameter=limit>\n5\n</parameter>  \n</function>";
        let events = finished(&[call]);
        assert_eq!(
            events[1..],
            [
                fragment("{\"limit\": 5", "  \n\n<parameter=limit>\n5\n</parameter>"),
                fragment("}", "  \n"),
                end("</function>"),
            ]
        );
    }

    #[test]
    fn a_value_written_between_the_tags_directly_keeps_every_byte_newlines_included() {
        // Seed-OSS's and MiMo's templates write `<parameter=KEY>`, the value and `</parameter>`
        // with no newline of their own, so a newline the value ends with is the value's:
        // bellwether's swebench call cases, whose patch ends in one, came back a byte short under
        // the Qwen 3.5 reading (smg-lab #110).
        let call = concat!(
            "<function=f>\n<parameter=city>diff --git a/x b/x\n+line\n</parameter>\n",
            "<parameter=note>  spaced  </parameter>\n<parameter=limit>5\n</parameter>\n",
            "<parameter=extra>x\n</parameter>\n</function>"
        );
        let expected = concat!(
            r#"{"city": "diff --git a/x b/x\n+line\n", "note": "  spaced  ", "limit": 5, "#,
            r#""extra": "x\n"}"#
        );
        let events = finished_direct(&[call]);
        assert_eq!(arguments(&events), expected);
        assert_eq!(bytes(&events), call);
        assert!(malformed(&events).is_empty());
        // The same at every two-way cut: no newline is held back, since none is the template's.
        for cut in 1..call.len() {
            let (events, taken) = run(
                &[&call[..cut], &call[cut..]],
                Placement::Direct,
                Assembler::finish,
            );
            assert_eq!(arguments(&events), expected, "cut at {cut}");
            assert_eq!(bytes(&events), call, "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), call.len(), "cut at {cut}");
            nothing_empty(&events);
        }
        // A value of one newline is one newline, of two is two, of nothing is nothing, and a
        // newline at the start is the value's too; under the own-line placement the same values
        // stand between the template's two newlines.
        for (value, json) in [
            ("\n", r#""\n""#),
            ("\n\n", r#""\n\n""#),
            ("", r#""""#),
            ("\nlead", r#""\nlead""#),
            ("trail\n", r#""trail\n""#),
            ("a  ", r#""a  ""#),
        ] {
            let direct = format!("<function=f>\n<parameter=city>{value}</parameter>\n</function>");
            assert_eq!(
                arguments(&finished_direct(&[&direct])),
                format!("{{\"city\": {json}}}"),
                "{value:?} between the tags directly"
            );
            let own_line =
                format!("<function=f>\n<parameter=city>\n{value}\n</parameter>\n</function>");
            assert_eq!(
                arguments(&finished(&[&own_line])),
                format!("{{\"city\": {json}}}"),
                "{value:?} on a line of its own"
            );
        }
    }
}
