//! Gemma 4's call syntax: `call:NAME{key:value,…}` between the table's markers, the events of one
//! call as its bytes arrive.
//!
//! Gemma 4 writes a call as
//!
//! ```text
//! <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>
//! ```
//!
//! The engine's table owns the markers ([`formats::gemma4`]): it enters this assembler after
//! `<|tool_call>` and leaves it at `<tool_call|>`, which it hands over as the call's end. Between
//! them the template writes `call:`, the tool's name and the arguments object in a spelling of its
//! own: keys bare, before `:`; a string between two `<|"|>` tokens, its text as it is, newlines
//! and quotes included, nothing escaped; a number, `true`, `false` and `null` bare; a list between
//! `[` and `]` and an object between `{` and `}`, their members joined by `,` with no space and
//! keys bare at every depth; the keys in sorted order (the template's `dictsort`), whatever order
//! the request gave them. FunctionGemma writes the same shape with `<escape>` around a string and
//! `None` for a null: the table names its [`Spelling`], and the assembler reads by it.
//!
//! The assembler writes the arguments as JSON while they arrive: each key quoted, `": "` after it
//! and `", "` between members, as the engines' parsers spell them; a string's text escaped as JSON
//! requires and streamed as it comes; a bare value as it is. The string token may be cut by a
//! piece's end, so a trailing start of it is held until the next bytes say. Every byte lands in one
//! event: the `call:NAME{` header is the start's source, each fragment's bytes its own, the closing
//! marker the end's. Text after the object closed and before the marker, which the template never
//! writes, is reported word by word with the model's whitespace between the words, the whitespace
//! right after the object and before the marker being the template's; a header that is not
//! `call:NAME{` is reported and no call starts.
//!
//! A member is written from its value's first byte: its key and colon are held until a value is
//! there, since the template writes a member as a key, a colon and a value, never one without a
//! key or without a value, and a comma between two present members only. A member without a key
//! (a colon where a key should stand) or without a value (a comma or a close where the value
//! should start) is reported and not written, every byte of it under the one reason, the comma
//! after it included, and the object closes whole without it, as the tagged assemblers leave such
//! a member; a list item missing before a comma the same. A member the call's marker cuts after
//! its colon is the cut call's: the key held for the value comes back with the end's reason,
//! nothing is made up for the value that never came, and the object closes whole without the
//! member; at the stream's end the key opens the member instead, nothing closes, and the call's
//! bytes stay the call's (`Assembler::finish`). Nesting
//! past the prefix parser's bound (`DEEPEST`) in what is written is reported from the bracket
//! that passes it, with the key held for it, so that the fragments stay a prefix that parser
//! takes, as the JSON syntax keeps them, and the object closes whole without the member.
//!
//! [`formats::gemma4`]: crate::formats::gemma4()

use crate::{
    event::{release_between, runs, DropReason, Event, Events, MalformedReason, Text},
    json::PartialJson,
    tagged::{
        assembler::{push_escaped, push_quoted},
        end_cut_call,
        value::json_prefix,
    },
};

/// How a template spells the syntax: the token around a string's text, and the word for a null
/// value. Gemma 4 writes `<|"|>` and `null`, FunctionGemma `<escape>` and `None`; `true` and
/// `false` are the same in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spelling {
    /// The token around a string's text.
    pub string: &'static str,
    /// The word for a null value.
    pub null: &'static str,
}

impl Spelling {
    /// Gemma 4's: `<|"|>` around a string, `null`.
    pub const GEMMA4: Self = Self {
        string: "<|\"|>",
        null: "null",
    };
    /// FunctionGemma's: `<escape>` around a string, `None` (Python's word for the value, where
    /// `true` and `false` keep JSON's).
    pub const FUNCTION_GEMMA: Self = Self {
        string: "<escape>",
        null: "None",
    };
}
/// The call's header up to the tool's name.
const HEADER: &str = "call:";

/// The tool's name in a header up to its `{`, or why the header is no call's: no `call:` prefix
/// is text where a call should begin, an empty name a call without one, a name with whitespace in
/// it text where the name should be.
fn header_name(head: &str) -> Result<&str, MalformedReason> {
    match head.strip_prefix(HEADER).map(str::trim) {
        None => Err(MalformedReason::TextBetweenCalls),
        Some("") => Err(MalformedReason::CallWithoutName),
        Some(name) if name.contains(char::is_whitespace) => {
            Err(MalformedReason::TextWhereTheNameShouldBe)
        }
        Some(name) => Ok(name),
    }
}

/// The nesting the arguments may reach, the arguments object counted: the prefix parser's bound
/// ([`PartialJson::DEFAULT_MAX_DEPTH`]), which the JSON syntax's judge applies, so that the
/// fragments of a call stay a prefix that parser takes, whichever syntax wrote them. It counts
/// what is written: the containers of a member without a key, which is not written, do not count.
const DEEPEST: usize = PartialJson::DEFAULT_MAX_DEPTH;

/// The events of one call, from the bytes after `<|tool_call>`.
#[derive(Clone, Debug)]
pub(crate) struct Assembler {
    index: u32,
    id: String,
    /// Bytes read into no event yet: the header before its `{`, a key before its `:`, a key and
    /// its colon before the value's first byte, a bare value before its end, or a trailing start
    /// of the string token.
    carried: String,
    /// The length of a member's key and colon at the front of the bytes held, from the colon to
    /// the value's first byte: a member is written only once a value is there, so one that a
    /// comma or a close leaves without a value is reported whole and nothing of it written, and
    /// one the call's end cuts after its colon is reported as the cut it is.
    key: Option<usize>,
    stage: Stage,
    /// The open containers, innermost last, each with whether a member has been written in it.
    containers: Vec<Container>,
    /// The member the template never writes that is being read, while one is: its bytes come
    /// back as `Malformed` under its reason and nothing of it is written, as the tagged
    /// assemblers leave one.
    skipping: Option<Skipped>,
    started: bool,
    done: bool,
    /// Whether text after the arguments object was released as content: the whitespace after it
    /// is the model's, between two pieces of its text, and leaves with the next piece
    /// ([`release_between`]); before any, the whitespace after the object is the template's
    /// spacing before the marker, dropped as it arrives.
    released: bool,
    /// The template's string token and null word.
    spelling: Spelling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Inside `call:NAME{`.
    Header,
    /// Where a key or the object's close stands.
    Key,
    /// Where a value or a list's close stands, after the key held for the value when there is one.
    Value,
    /// Inside a string: the text up to the next string token.
    Text,
    /// Where `,` or the container's close stands.
    AfterValue,
    /// After a bracket that would open a container past [`DEEPEST`]: the arguments are no longer
    /// read, every byte from the refused member's key, or from the refused list element, up to
    /// the call's end is reported, and the containers still open are closed at the end, whole,
    /// since the member whose value the bracket began was held and nothing of it written.
    Refused,
    /// The arguments object closed; only the marker should follow.
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Container {
    list: bool,
    members: u32,
}

/// A member the template never writes, being read: the depth of the container whose member it
/// is, where it ends at a comma or the close, and the reason its bytes carry, a key without a
/// name or a member without a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Skipped {
    depth: usize,
    why: MalformedReason,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after the call's opening marker, reading by `spelling`.
    pub fn new(index: u32, id: impl Into<String>, spelling: Spelling) -> Self {
        Self {
            index,
            id: id.into(),
            carried: String::new(),
            key: None,
            stage: Stage::Header,
            containers: Vec::new(),
            skipping: None,
            started: false,
            done: false,
            released: false,
            spelling,
        }
    }

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done || bytes.is_empty() {
            return;
        }
        let mut text = std::mem::take(&mut self.carried);
        text.push_str(bytes);
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let stage = self.stage;
            let taken = match self.stage {
                Stage::Header => self.header(rest, out),
                Stage::Key => self.key(rest, out),
                Stage::Value => self.value(rest, out),
                Stage::Text => self.text(rest, out),
                Stage::AfterValue => self.after_value(rest, out),
                Stage::Refused => refused(rest, out),
                Stage::Closed => self.after_the_object(rest, out),
            };
            if taken == 0 && self.stage == stage {
                // What is left is held: a header, key or bare value still arriving, a start of
                // the string token, or a key whose value has not begun. A reader that moved on
                // without taking a byte (a key read, held for its value; a member without a
                // value) has the next one read the same bytes.
                self.carried.push_str(rest);
                return;
            }
            rest = &rest[taken..];
        }
    }

    /// The call's end at the table's closing marker, whose bytes go into `ToolCallEnd`. An object
    /// still open is closed for the client, with what was cut short reported, unless a string is
    /// open: nothing is made up for a value the model never finished, so the call ends as it
    /// stands and says that its arguments are no whole value ([`Self::end`]); a call that never
    /// started has its bytes reported.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        self.end(out);
        if self.started {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(terminal.to_owned()),
            });
        } else if !terminal.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(terminal.to_owned()),
                why: MalformedReason::BlockWithoutACompleteCall,
            });
        }
        self.done = true;
    }

    /// The stream's end inside the call: a cut by the token limit, an abort, or the model's own
    /// end of output before the closing marker, which ends a Gemma call as a cut does since its
    /// braces and tokens are the model's own (the table has no stop row; `engine::CallAssembler::stop`).
    /// Nothing is closed, so arguments cut short never look complete to a client, and the
    /// call's bytes stay the call's. A call that started still ends, with no bytes of its own,
    /// as every assembler ends one, after the one `Malformed { UnterminatedRegion }` event of no
    /// bytes that says that its arguments are no whole value (`tagged::end_cut_call`); no
    /// container closes, not even an object the end found between two members, since a closing
    /// brace the model never wrote would hand the client arguments that read as whole. Before
    /// the end, what the cut found open leaves as the call's own: a
    /// string's text so far, the bytes held as a start of its closing token included, as the
    /// marker reads them; the member whose key came, with its bare value as far as it went
    /// (`"n": 1`) or the key alone (`"n": `) when no byte of the value came yet
    /// ([`Self::cut_member`]); a start of the string token held after the key leaves after the
    /// call's end as the syntax's wrapping. A key cut short and the whitespace held after a
    /// value leave as wrapping too, since no member can be written for them. An object the model
    /// closed before the cut is whole: the whitespace held after it is the template's, and the
    /// call ends with no word. A member the template never writes still comes back under its
    /// reason, and a header the cut found before its `{` as `Malformed { UnterminatedRegion }`,
    /// since no call started.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.end_skipped(out);
        if self.stage == Stage::Closed {
            // The object closed before the cut, or a header that was no call's (its text after
            // it released as it came): the whitespace held after them is the template's, before
            // a marker that never came, and a started call is whole and ends with no word.
            self.drop_held(out);
            if self.started {
                out.push(Event::ToolCallEnd {
                    index: self.index,
                    source: Text::default(),
                });
            }
            self.done = true;
            return;
        }
        if !self.started {
            let held = self.take_held();
            if !held.is_empty() {
                out.push(Event::Malformed {
                    text: Text::uncounted(held),
                    why: MalformedReason::UnterminatedRegion,
                });
            }
            self.done = true;
            return;
        }
        let cut_token = match self.stage {
            Stage::Text => {
                let held = self.take_held();
                if !held.is_empty() {
                    let mut json = String::with_capacity(held.len());
                    push_escaped(&mut json, &held);
                    self.push_fragment(json, Text::uncounted(held), out);
                }
                None
            }
            Stage::Value => self.cut_member(out),
            // `Closed` returned above; the others hold a key cut short or whitespace.
            Stage::Closed | Stage::Header | Stage::Key | Stage::AfterValue | Stage::Refused => {
                self.drop_held(out);
                None
            }
        };
        end_cut_call(self.index, String::new(), "", out);
        if let Some(token) = cut_token {
            out.push(Event::Dropped {
                text: Text::uncounted(token),
                why: DropReason::Wrapper,
            });
        }
        self.done = true;
    }

    /// The bytes held, dropped as the syntax's wrapping; nothing when none.
    fn drop_held(&mut self, out: &mut Events) {
        let held = self.take_held();
        if !held.is_empty() {
            out.push(Event::Dropped {
                text: Text::uncounted(held),
                why: DropReason::Wrapper,
            });
        }
    }

    /// The member the end of the stream cut where its value stands ([`Self::finish`]): the key
    /// held for the value, when one is, opens the member, `"key": `, and a bare value's bytes
    /// after it follow as the JSON they stand for so far, written as [`push_bare`] writes the
    /// whole value once the member closes: the digits of a number (its minus sign alone
    /// included) and the start of `true` or `false` as written, the start of the spelling's
    /// null word as the start of `null`, and
    /// any other text (a word that is no literal, Python's `True` or `'abc'`, a JSON quote
    /// where the template writes its own string token) as the opening of the string it is
    /// written as, so that the arguments carry the prefix the model wrote, stay a strict prefix
    /// of a JSON value, and the cut never gives the value a type the closed member would not
    /// have; in a list the bytes are the next element's, after the separator. The bytes held as
    /// a start of the string token are no value yet: they are returned, a token the cut left
    /// short, to leave after the call's end as the syntax's wrapping. Whitespace alone where no
    /// key is held is wrapping too.
    fn cut_member(&mut self, out: &mut Events) -> Option<String> {
        let key = self.key.unwrap_or(0);
        let held = self.take_held();
        let text = held[key..].trim_start();
        let cut_token =
            (!text.is_empty() && self.spelling.string.starts_with(text)).then(|| text.to_owned());
        let value = if cut_token.is_some() {
            ""
        } else {
            text.trim_end()
        };
        let source = &held[..held.len() - cut_token.as_ref().map_or(0, String::len)];
        if key == 0 && value.is_empty() {
            if !source.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(source.to_owned()),
                    why: DropReason::Wrapper,
                });
            }
            return cut_token;
        }
        let mut json = self.separator();
        if key > 0 {
            push_quoted(&mut json, held[..key - 1].trim());
            json.push_str(": ");
        }
        // As [`push_bare`] writes the value once the member closes: a number or `true` or
        // `false` as written, the spelling's null word as `null`, every other text as a string.
        // A lone `-` is the start of a number, as `tr` is the start of a literal: the digits
        // follow it far more often than a member closes on it.
        let number = (value == "-"
            || value
                .strip_prefix('-')
                .unwrap_or(value)
                .starts_with(|c: char| c.is_ascii_digit()))
            && json_prefix(value);
        if !value.is_empty() {
            if self.spelling.null.starts_with(value) {
                json.push_str(&"null"[..value.len()]);
            } else if number || "true".starts_with(value) || "false".starts_with(value) {
                json.push_str(value);
            } else {
                json.push('"');
                push_escaped(&mut json, value);
            }
        }
        self.push_fragment(json, Text::uncounted(source.to_owned()), out);
        cut_token
    }

    /// `call:NAME{`: the start, once the `{` arrives. Anything else up to the `{` is reported.
    fn header(&mut self, rest: &str, out: &mut Events) -> usize {
        let Some(brace) = rest.find('{') else {
            return 0;
        };
        match header_name(&rest[..brace]) {
            Ok(name) => {
                self.started = true;
                out.push(Event::ToolCallStart {
                    index: self.index,
                    id: self.id.clone(),
                    name: name.to_string(),
                    source: Text::uncounted(rest[..=brace].to_owned()),
                });
                self.push_fragment("{".to_string(), Text::default(), out);
                self.containers.push(Container {
                    list: false,
                    members: 0,
                });
                self.stage = Stage::Key;
            }
            Err(why) => {
                out.push(Event::Malformed {
                    text: Text::uncounted(rest[..=brace].to_owned()),
                    why,
                });
                self.stage = Stage::Closed;
            }
        }
        brace + 1
    }

    /// A key up to its `:`, or the object's close. The key is held until its value's first byte
    /// ([`Self::value`]), since the template writes no member without a value.
    fn key(&mut self, rest: &str, out: &mut Events) -> usize {
        let space = rest.len() - rest.trim_start().len();
        if space == rest.len() {
            return 0;
        }
        if rest[space..].starts_with('}') {
            self.close_container(&rest[..=space], out);
            return space + 1;
        }
        let Some(colon) = rest.find(':') else {
            return 0;
        };
        if rest[space..colon].trim_end().is_empty() {
            // The template writes no member without a key: the colon is reported, the value that
            // follows comes back the same way while nothing of it is written, and the object stays
            // whole without the member, as the tagged assemblers leave it. Inside a member already
            // being skipped, the outer member's end is the one that counts.
            out.push(Event::Malformed {
                text: Text::uncounted(rest[..=colon].to_owned()),
                why: MalformedReason::EmptyKey,
            });
            self.skip(MalformedReason::EmptyKey);
            self.stage = Stage::Value;
            return colon + 1;
        }
        // The key and colon stay among the bytes held: the value's reader writes them once the
        // value's first byte is there.
        self.key = Some(colon + 1);
        self.stage = Stage::Value;
        0
    }

    /// A value's start: a string's opening token, a container's opening bracket, a list's close,
    /// or a bare literal up to the byte that ends it, the key held for the value written first. A
    /// comma or a close where the value should start is a member without a value, which the
    /// template never writes: the key held for it and the whitespace come back as `Malformed`
    /// (the comma follows them through [`Self::after_value`]), nothing of the member is written,
    /// and the container closes whole without it, as for a member without a key.
    fn value(&mut self, rest: &str, out: &mut Events) -> usize {
        // The key and colon held for the value stand at the front; the value's bytes follow.
        let key = self.key.unwrap_or(0);
        let text = &rest[key..];
        let space = text.len() - text.trim_start().len();
        if space == text.len() {
            return 0;
        }
        let after = &text[space..];
        let first = after.chars().next().unwrap_or_default();
        let token = self.spelling.string;
        let string = after.starts_with(token);
        // Nothing is taken until the value's start is whole: a start of the string token still
        // arriving, or a bare literal whose end has not come, is held.
        if (!string && token.starts_with(after))
            || (!string && !matches!(first, '{' | '[' | ']') && !after.contains([',', '}', ']']))
        {
            return 0;
        }
        if matches!(first, ']' | '}') && self.in_list() {
            // The list's close, after its `[` or a comma: an object's brace closes it too, as it
            // does after a value. No key is held in a list.
            debug_assert!(self.key.is_none());
            self.close_container(&text[..=space], out);
            return key + space + 1;
        }
        if matches!(first, ',' | '}' | ']') {
            self.key = None;
            self.skip(MalformedReason::KeyWithoutAValue);
            self.push_fragment(
                String::new(),
                Text::uncounted(rest[..key + space].to_owned()),
                out,
            );
            self.stage = Stage::AfterValue;
            return key + space;
        }
        if matches!(first, '{' | '[') && self.skipping.is_none() && self.containers.len() >= DEEPEST
        {
            // Nesting the prefix parser would refuse: the fragments stop here, where they are
            // still a prefix it takes, and the bytes from here on are reported, as the JSON
            // syntax reports the bytes from the bracket it cannot take: a member from its key,
            // held for this value and not written, so that the object closes whole without it,
            // as it does without a member that has no value; a list's element from the
            // whitespace before its bracket. The members written before stay the call's; the end
            // closes what is open. A member being skipped is not written, so its nesting is not
            // counted: nothing of it reaches the parser, and the members after it are written as
            // before.
            self.key = None;
            self.stage = Stage::Refused;
            return refused(rest, out);
        }
        if self.key.take().is_some() {
            let mut json = self.separator();
            push_quoted(&mut json, rest[..key - 1].trim());
            json.push_str(": ");
            self.push_fragment(json, Text::uncounted(rest[..key].to_owned()), out);
        }
        // In a list each value is a member and takes the separator; in an object the key took it.
        let separator = if self.in_list() {
            self.separator()
        } else {
            String::new()
        };
        if string {
            let json = format!("{separator}\"");
            self.push_fragment(
                json,
                Text::uncounted(text[..space + token.len()].to_owned()),
                out,
            );
            self.stage = Stage::Text;
            return key + space + token.len();
        }
        match first {
            '{' | '[' => {
                let json = format!("{separator}{first}");
                self.push_fragment(json, Text::uncounted(text[..=space].to_owned()), out);
                self.containers.push(Container {
                    list: first == '[',
                    members: 0,
                });
                self.stage = if first == '[' {
                    Stage::Value
                } else {
                    Stage::Key
                };
                key + space + 1
            }
            _ => {
                let end = after.find([',', '}', ']']).unwrap_or(after.len());
                let literal = after[..end].trim_end();
                let mut json = separator;
                push_bare(&mut json, literal, self.spelling.null);
                self.push_fragment(json, Text::uncounted(text[..space + end].to_owned()), out);
                self.stage = Stage::AfterValue;
                key + space + end
            }
        }
    }

    /// A string's text up to its closing token, streamed as it arrives; a trailing start of the
    /// token is held.
    fn text(&mut self, rest: &str, out: &mut Events) -> usize {
        let token = self.spelling.string;
        if let Some(at) = rest.find(token) {
            let mut json = String::with_capacity(at + 1);
            push_escaped(&mut json, &rest[..at]);
            json.push('"');
            self.push_fragment(
                json,
                Text::uncounted(rest[..at + token.len()].to_owned()),
                out,
            );
            self.stage = Stage::AfterValue;
            return at + token.len();
        }
        let keep = (1..token.len())
            .rev()
            .find(|&n| {
                rest.len() >= n
                    && rest.is_char_boundary(rest.len() - n)
                    && token.starts_with(&rest[rest.len() - n..])
            })
            .unwrap_or(0);
        let give = &rest[..rest.len() - keep];
        if give.is_empty() {
            return 0;
        }
        let mut json = String::with_capacity(give.len());
        push_escaped(&mut json, give);
        self.push_fragment(json, Text::uncounted(give.to_owned()), out);
        give.len()
    }

    /// After a value: `,` before the next member, or the container's close.
    fn after_value(&mut self, rest: &str, out: &mut Events) -> usize {
        let space = rest.len() - rest.trim_start().len();
        if space == rest.len() {
            return 0;
        }
        let first = rest[space..].chars().next().unwrap_or_default();
        match first {
            ',' => {
                self.push_fragment(
                    String::new(),
                    Text::uncounted(rest[..=space].to_owned()),
                    out,
                );
                self.member_done();
                self.stage = if self.in_list() {
                    Stage::Value
                } else {
                    Stage::Key
                };
            }
            '}' | ']' => self.close_container(&rest[..=space], out),
            _ => {
                // The template writes a separator or a close here; anything else is the next
                // member's start, read as such from its first byte, with the whitespace before
                // it reported.
                if space > 0 {
                    out.push(Event::Malformed {
                        text: Text::uncounted(rest[..space].to_owned()),
                        why: MalformedReason::ValueCutShort,
                    });
                }
                self.member_done();
                let next = &rest[space..];
                let taken = if self.in_list() {
                    self.stage = Stage::Value;
                    self.value(next, out)
                } else {
                    self.stage = Stage::Key;
                    self.key(next, out)
                };
                return space + taken;
            }
        }
        space + 1
    }

    /// The innermost container closes: `}` or `]` written, with `source` as the bytes.
    fn close_container(&mut self, source: &str, out: &mut Events) {
        self.member_done();
        let container = self.containers.pop();
        let json = if container.is_some_and(|c| c.list) {
            "]"
        } else {
            "}"
        };
        self.push_fragment(json.to_string(), Text::uncounted(source.to_owned()), out);
        self.stage = if self.containers.is_empty() {
            Stage::Closed
        } else {
            Stage::AfterValue
        };
    }

    fn in_list(&self) -> bool {
        self.containers.last().is_some_and(|c| c.list)
    }

    /// The member of the innermost container being read is one the template never writes, for
    /// `why`: its bytes come back as `Malformed` until it ends. Inside a member already skipped
    /// the outer one's reason and end stay.
    fn skip(&mut self, why: MalformedReason) {
        if self.skipping.is_none() {
            self.skipping = Some(Skipped {
                depth: self.containers.len(),
                why,
            });
        }
    }

    /// A member of the innermost container ended: a member skipped there is over, and the
    /// fragments resume.
    fn member_done(&mut self) {
        if self
            .skipping
            .is_some_and(|skipped| skipped.depth == self.containers.len())
        {
            self.skipping = None;
        }
    }

    /// The call ends inside a skipped member: what was held is the member's bytes, the
    /// containers opened inside it have nothing to write, and the object closes whole without it.
    fn end_skipped(&mut self, out: &mut Events) {
        let Some(skipped) = self.skipping.take() else {
            return;
        };
        let held = self.take_held();
        if !held.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(held),
                why: skipped.why,
            });
        }
        self.containers.truncate(skipped.depth);
        self.stage = Stage::AfterValue;
    }

    /// The bytes read into no event yet, a key held for its value among them.
    fn take_held(&mut self) -> String {
        self.key = None;
        std::mem::take(&mut self.carried)
    }

    /// `", "` before every member of a container but its first, and the member counted.
    fn separator(&mut self) -> String {
        match self.containers.last_mut() {
            Some(container) => {
                container.members += 1;
                if container.members > 1 {
                    ", ".to_string()
                } else {
                    String::new()
                }
            }
            None => String::new(),
        }
    }

    /// The call's end before its marker: a key or a value cut short is reported, and every open
    /// container is closed, so the arguments are an object. A member cut after its colon is the
    /// key held for its value, reported with the value's reason, so the object closed below is
    /// whole without it: nothing is made up for a value the model never wrote. A string left
    /// open stays open, with the bytes held as its text sent, and nothing closes after it,
    /// neither the string nor the containers above it: a `Malformed` event of no bytes, with the
    /// value's reason, says that the arguments are no whole value.
    fn end(&mut self, out: &mut Events) {
        self.end_skipped(out);
        let held = self.take_held();
        match self.stage {
            Stage::Header if !held.is_empty() => {
                let why = header_name(&held)
                    .map_or_else(|why| why, |_| MalformedReason::NameWithoutACall);
                out.push(Event::Malformed {
                    text: Text::uncounted(held),
                    why,
                });
            }
            Stage::Key if !held.trim().is_empty() => out.push(Event::Malformed {
                text: Text::uncounted(held),
                why: MalformedReason::KeyCutShort,
            }),
            Stage::Value if !held.trim().is_empty() => {
                out.push(Event::Malformed {
                    text: Text::uncounted(held),
                    why: MalformedReason::ValueCutShort,
                });
            }
            Stage::Text => {
                // The held bytes are the string's text, cut before its closing token.
                if !held.is_empty() {
                    let mut json = String::with_capacity(held.len());
                    push_escaped(&mut json, &held);
                    self.push_fragment(json, Text::uncounted(held), out);
                }
                out.push(Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::ValueCutShort,
                });
            }
            _ if !held.is_empty() => out.push(Event::Dropped {
                text: Text::uncounted(held),
                why: DropReason::Wrapper,
            }),
            _ => {}
        }
        if self.started && self.stage != Stage::Text {
            self.close_open_containers(out);
        }
    }

    fn close_open_containers(&mut self, out: &mut Events) {
        while let Some(container) = self.containers.pop() {
            let json = if container.list { "]" } else { "}" };
            self.push_fragment(json.to_string(), Text::default(), out);
        }
        self.stage = Stage::Closed;
    }

    /// Text after the arguments closed, before the call's end. Whitespace right after the object
    /// is the template's spacing and is dropped as it arrives; a word is reported, with the
    /// whitespace before it once a word was reported already, since that whitespace is the
    /// model's, between two pieces of its text ([`release_between`]); the whitespace after the
    /// last word is held, the next word's if one comes and the template's spacing before the
    /// marker otherwise, which the end drops. The same text however the bytes are cut.
    fn after_the_object(&mut self, text: &str, out: &mut Events) -> usize {
        let kept = text.trim_end().len();
        if kept == 0 {
            if self.released {
                return 0;
            }
            out.push(Event::Dropped {
                text: Text::uncounted(text.to_owned()),
                why: DropReason::Wrapper,
            });
            return text.len();
        }
        let mut before = String::new();
        for (run, space) in runs(&text[..kept]) {
            if space {
                before.push_str(run);
            } else {
                release_between(
                    &mut before,
                    &mut self.released,
                    run.to_owned().into(),
                    MalformedReason::TextAfterTheArguments,
                    out,
                );
            }
        }
        kept
    }

    /// A fragment of the arguments; inside a skipped member, its bytes as `Malformed` with the
    /// member's reason instead, since nothing of that member is written.
    fn push_fragment<'d>(&self, json: String, source: Text<'d>, out: &mut Events<'d>) {
        if let Some(skipped) = self.skipping {
            if !source.text.is_empty() {
                out.push(Event::Malformed {
                    text: source,
                    why: skipped.why,
                });
            }
            return;
        }
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: json.into(),
            source,
        });
    }
}

/// Bytes from a bracket that would open a container past [`DEEPEST`], the key held for it before
/// them, up to the call's end: reported as invalid arguments, piece by piece, which joins to the
/// same text however the bytes are cut.
fn refused(text: &str, out: &mut Events) -> usize {
    if !text.is_empty() {
        out.push(Event::Malformed {
            text: Text::uncounted(text.to_owned()),
            why: MalformedReason::InvalidArguments,
        });
    }
    text.len()
}

/// A bare value as JSON: a number, `true` or `false` as it is, the spelling's null word as `null`;
/// any other bare text, which the template never writes, as a JSON string, so the arguments stay
/// an object a client can read.
fn push_bare(json: &mut String, literal: &str, null: &str) {
    let is_number = literal
        .strip_prefix('-')
        .unwrap_or(literal)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
        && serde_json::from_str::<serde_json::Value>(literal).is_ok();
    if is_number || matches!(literal, "true" | "false") {
        json.push_str(literal);
    } else if literal == null {
        json.push_str("null");
    } else {
        push_quoted(json, literal);
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

    /// Gemma 4's closing marker is the call's own, carried by its end.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool {
        (*self).close(terminal, out);
        true
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{arguments, bytes, malformed};

    const CLOSE: &str = "<tool_call|>";

    fn run(pieces: &[&str], cut: bool) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, "call_0", Spelling::GEMMA4);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(CLOSE, &mut out);
        }
        out.drain()
    }

    const CALL: &str = concat!(
        "call:get_weather{city:<|\"|>Paris, \"la\" ville\n<|\"|>,days:3,metric:true,",
        "options:{tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],none:null,ratio:-0.5},empty:[]}"
    );

    #[test]
    fn a_member_that_follows_a_value_without_a_separator_keeps_its_first_byte() {
        // A value followed by the next member where a comma should be: the member is read from
        // its first byte, so every byte of the output lands in an event, whole and in pieces,
        // and the arguments stay a valid object; a member without a key is reported and skipped.
        let cases = [
            (
                "call:f{a:<|\"|>x<|\"|>coconut<|\"|>,b:1}",
                r#"{"a": "x", "coconut<|\"|>,b": 1}"#,
            ),
            ("call:f{a:<|\"|>x<|\"|>  coconut}", r#"{"a": "x"}"#),
            ("call:f{a:<|\"|>x<|\"|>:2}", r#"{"a": "x"}"#),
            ("call:f{:2}", "{}"),
            ("call:f{a:[<|\"|>x<|\"|>y]}", r#"{"a": ["x", "y"]}"#),
        ];
        for (text, expected) in cases {
            let whole = run(&[text], false);
            assert_eq!(bytes(&whole), format!("{text}{CLOSE}"), "{text:?}: whole");
            assert_eq!(arguments(&whole), expected, "{text:?}");
            assert!(
                serde_json::from_str::<serde_json::Value>(&arguments(&whole)).is_ok(),
                "{text:?}: a valid object"
            );
            let per_char: Vec<&str> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            let in_pieces = run(&per_char, false);
            assert_eq!(
                bytes(&in_pieces),
                format!("{text}{CLOSE}"),
                "{text:?}: in pieces"
            );
            assert_eq!(arguments(&in_pieces), expected, "{text:?}: in pieces");
            assert_eq!(
                malformed(&in_pieces),
                malformed(&whole),
                "{text:?}: in pieces"
            );
        }
    }

    #[test]
    fn a_missing_value_before_a_comma_reads_on_however_the_bytes_are_cut() {
        // The reader of a member without a value moves on without taking a byte; the next reader
        // takes the same bytes, whole and in pieces alike, instead of the rest being held.
        let text = "call:f{a:,<|\"|>X<|\"|>}";
        let whole = run(&[text], false);
        let per_char: Vec<&str> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        let in_pieces = run(&per_char, false);
        assert_eq!(bytes(&whole), format!("{text}{CLOSE}"));
        assert_eq!(bytes(&in_pieces), format!("{text}{CLOSE}"));
        assert_eq!(arguments(&whole), "{}");
        assert_eq!(
            malformed(&whole),
            vec![
                ("a:".to_string(), MalformedReason::KeyWithoutAValue),
                (",".to_string(), MalformedReason::KeyWithoutAValue),
                ("<|\"|>X<|\"|>}".to_string(), MalformedReason::KeyCutShort),
            ]
        );
        assert_eq!(whole, in_pieces, "the same events");
    }

    /// A member without a value (`{city:,days:3}`, `{city:}`): the template writes a value after
    /// every colon, so the key and colon held for the value, the whitespace after them and the
    /// comma that follows come back as `Malformed` under one reason, nothing of the member is
    /// written (no empty string made up for it), and the object stays whole without it, its next
    /// member first; a list item missing before a comma the same. A string the model closed as
    /// soon as it opened it is an empty string and needs no word. Whole and character by
    /// character.
    #[test]
    fn a_member_without_a_value_is_reported_and_nothing_of_it_is_written() {
        let reported = |events: &[Event], reason: MalformedReason| -> String {
            malformed(events)
                .iter()
                .filter(|(_, why)| *why == reason)
                .map(|(text, _)| text.as_str())
                .collect()
        };
        let made_up = |events: &[Event]| -> bool {
            events.iter().any(|event| {
                matches!(
                    event,
                    Event::ToolCallArguments { json, source, .. }
                        if json.ends_with("\"\"") && source.text.trim().is_empty()
                )
            })
        };
        for (output, expected, bytes_reported) in [
            ("call:f{city:,days:3}", r#"{"days": 3}"#, "city:,"),
            ("call:f{city:}", "{}", "city:"),
            ("call:f{a:1,city: ,b:2}", r#"{"a": 1, "b": 2}"#, "city: ,"),
            ("call:f{a:1,city:}", r#"{"a": 1}"#, "city:"),
            (
                "call:f{a:{x:},b:[1,,2]}",
                r#"{"a": {}, "b": [1, 2]}"#,
                "x:,",
            ),
            ("call:f{a:[,<|\"|>x<|\"|>]}", r#"{"a": ["x"]}"#, ","),
            ("call:f{a:<|\"|><|\"|>,b:1}", r#"{"a": "", "b": 1}"#, ""),
        ] {
            let per_char: Vec<&str> = output
                .char_indices()
                .map(|(i, c)| &output[i..i + c.len_utf8()])
                .collect();
            for (pieces, how) in [(&[output][..], "whole"), (&per_char[..], "per character")] {
                let events = run(pieces, false);
                assert_eq!(arguments(&events), expected, "{output:?} {how}");
                assert!(
                    serde_json::from_str::<serde_json::Value>(&arguments(&events)).is_ok(),
                    "{output:?} {how}: a valid object"
                );
                assert_eq!(
                    reported(&events, MalformedReason::KeyWithoutAValue),
                    bytes_reported,
                    "{output:?} {how}"
                );
                assert!(
                    malformed(&events)
                        .iter()
                        .all(|(_, why)| *why == MalformedReason::KeyWithoutAValue),
                    "{output:?} {how}: {events:?}"
                );
                assert!(!made_up(&events), "{output:?} {how}: {events:?}");
                assert_eq!(
                    bytes(&events),
                    format!("{output}{CLOSE}"),
                    "{output:?} {how}"
                );
            }
        }
        // Inside a member without a key, a missing value keeps that member's reason, and a
        // member without a key nested in one ends with the outer member, not before it.
        for (output, expected, bytes_skipped) in [
            ("call:f{:,a:1}", r#"{"a": 1}"#, ":,"),
            ("call:f{:{x:},a:1}", r#"{"a": 1}"#, ":{x:},"),
            ("call:f{:{:1},a:2}", r#"{"a": 2}"#, ":{:1},"),
        ] {
            let per_char: Vec<&str> = output
                .char_indices()
                .map(|(i, c)| &output[i..i + c.len_utf8()])
                .collect();
            for (pieces, how) in [(&[output][..], "whole"), (&per_char[..], "per character")] {
                let events = run(pieces, false);
                assert_eq!(arguments(&events), expected, "{output:?} {how}");
                assert_eq!(
                    reported(&events, MalformedReason::EmptyKey),
                    bytes_skipped,
                    "{output:?} {how}"
                );
                assert!(
                    malformed(&events)
                        .iter()
                        .all(|(_, why)| *why == MalformedReason::EmptyKey),
                    "{output:?} {how}: {events:?}"
                );
                assert_eq!(
                    bytes(&events),
                    format!("{output}{CLOSE}"),
                    "{output:?} {how}"
                );
            }
        }
    }

    #[test]
    fn function_gemmas_spelling_reads_escape_strings_and_none_as_null() {
        // functiongemma-270m-it/parse/call-nested-arguments, as the template writes it
        let output =
            "call:save_note{body:<escape>Line one\nLine two with \"quotes\" and a backslash \
                      \\<escape>,meta:{nested:{deep:[1,2.5,true,None]}},tags:[<escape>a<escape>,\
                      <escape>b<escape>],title:<escape>Plan<escape>}";
        let mut assembler = Assembler::new(0, "call_0", Spelling::FUNCTION_GEMMA);
        let mut out = Events::new();
        assembler.feed(output, &mut out);
        assembler.close("<end_function_call>", &mut out);
        let events = out.drain();
        assert_eq!(bytes(&events), format!("{output}<end_function_call>"));
        assert_eq!(
            arguments(&events),
            "{\"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \
             \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}, \"tags\": [\"a\", \"b\"], \
             \"title\": \"Plan\"}"
        );
        // Gemma 4's word is not this spelling's: a bare `null` is text the template never writes,
        // kept as a string; the token cut across pieces is held until it is whole.
        let mut assembler = Assembler::new(0, "call_0", Spelling::FUNCTION_GEMMA);
        let mut out = Events::new();
        for piece in ["call:f{a:null,b:<esc", "ape>x<escape>}"] {
            assembler.feed(piece, &mut out);
        }
        assembler.close("<end_function_call>", &mut out);
        assert_eq!(arguments(&out.drain()), "{\"a\": \"null\", \"b\": \"x\"}");
    }

    #[test]
    fn a_rendered_call_gives_its_name_its_arguments_as_json_and_every_byte() {
        let events = run(&[CALL], false);
        assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"));
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "get_weather" && source.text == "call:get_weather{"
        ));
        assert_eq!(
            arguments(&events),
            concat!(
                r#"{"city": "Paris, \"la\" ville\n", "days": 3, "metric": true, "#,
                r#""options": {"tags": ["a", "b"], "none": null, "ratio": -0.5}, "empty": []}"#
            )
        );
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == CLOSE
        ));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn the_call_reads_the_same_however_the_bytes_are_cut() {
        let whole = run(&[CALL], false);
        let expected = arguments(&whole);
        let boundaries: Vec<usize> = CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run(&[&CALL[..at], &CALL[at..]], false);
            assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"), "cut at {at}");
            assert_eq!(arguments(&events), expected, "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        let one_by_one: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let events = run(&one_by_one, false);
        assert_eq!(arguments(&events), expected);
        assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"));
    }

    #[test]
    fn a_string_streams_as_its_pieces_arrive() {
        let events = run(
            &[
                "call:f{patch:<|\"|>diff --git",
                " a/x\n+if a < b",
                "<|\"|>}",
            ],
            false,
        );
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } if !json.is_empty() => Some(json.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            [
                "{",
                "\"patch\": ",
                "\"",
                "diff --git",
                " a/x\\n+if a < b",
                "\"",
                "}"
            ]
        );
    }

    #[test]
    fn no_arguments_is_an_empty_object_and_a_cut_stream_closes_nothing() {
        let none = run(&["call:ping{}"], false);
        assert_eq!(arguments(&none), "{}");
        assert!(malformed(&none).is_empty());
        // Cut after a list's comma: neither the list nor the object closes, and the one word of
        // no bytes says that the arguments are no whole value.
        let cut = run(&["call:f{tags:[<|\"|>a<|\"|>,"], true);
        assert_eq!(arguments(&cut), r#"{"tags": ["a""#);
        assert_eq!(
            malformed(&cut),
            vec![(String::new(), MalformedReason::UnterminatedRegion)]
        );
        assert!(cut.iter().any(
            |event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty())
        ));
        // Cut inside a key: the key's bytes are the syntax's wrapping, and the object stays open.
        let cut_in_key = run(&["call:f{ci"], true);
        assert_eq!(arguments(&cut_in_key), "{");
        assert_eq!(bytes(&cut_in_key), "call:f{ci");
        assert!(cut_in_key.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "ci"
        )));
    }

    /// A member without a key (`{:1}`): the template writes none, so the colon and the value that
    /// follows come back as `Malformed`, run by run, and nothing of the member is written, as the
    /// tagged assemblers leave one; the object stays whole without it, its next member first.
    /// The same for a string, for a container with members of its own, and for a member the
    /// marker or the end of the stream cuts, whole and character by character.
    #[test]
    fn a_member_without_a_key_is_reported_and_nothing_of_it_is_written() {
        let skipped = |events: &[Event]| -> String {
            malformed(events)
                .iter()
                .filter(|(_, why)| *why == MalformedReason::EmptyKey)
                .map(|(text, _)| text.as_str())
                .collect()
        };
        for (output, expected, bytes_skipped) in [
            ("call:f{:1,a:2}", r#"{"a": 2}"#, ":1,"),
            (
                "call:f{a:2,:<|\"|>x<|\"|>}",
                r#"{"a": 2}"#,
                ":<|\"|>x<|\"|>",
            ),
            (
                "call:f{:{x:[1,<|\"|>y<|\"|>]},a:2}",
                r#"{"a": 2}"#,
                ":{x:[1,<|\"|>y<|\"|>]},",
            ),
            ("call:f{:[1,2]}", "{}", ":[1,2]"),
            // A member without a key inside the skipped member: the outer skip stands.
            ("call:f{:{:1},a:2}", r#"{"a": 2}"#, ":{:1},"),
            ("call:f{:{:1,b:2},a:2}", r#"{"a": 2}"#, ":{:1,b:2},"),
        ] {
            let per_char: Vec<&str> = output
                .char_indices()
                .map(|(i, c)| &output[i..i + c.len_utf8()])
                .collect();
            for (pieces, how) in [(&[output][..], "whole"), (&per_char[..], "per character")] {
                let events = run(pieces, false);
                assert_eq!(arguments(&events), expected, "{output:?} {how}");
                assert_eq!(skipped(&events), bytes_skipped, "{output:?} {how}");
                assert!(
                    malformed(&events)
                        .iter()
                        .all(|(_, why)| *why == MalformedReason::EmptyKey),
                    "{output:?} {how}: {events:?}"
                );
                assert_eq!(
                    bytes(&events),
                    format!("{output}{CLOSE}"),
                    "{output:?} {how}"
                );
            }
        }
        for (output, cut) in [
            ("call:f{:<|\"|>te", false),
            ("call:f{:<|\"|>te", true),
            ("call:f{:{x:", false),
            ("call:f{:{x:", true),
            // A member without a key inside the skipped member, cut: the outer skip ends it.
            ("call:f{:{:1", false),
            ("call:f{:{:1", true),
        ] {
            let events = run(&[output], cut);
            // At the marker the object closes whole without the member; at the stream's end
            // nothing closes, and the one word of no bytes says so.
            assert_eq!(
                arguments(&events),
                if cut { "{" } else { "{}" },
                "{output:?} cut {cut}"
            );
            assert_eq!(
                skipped(&events),
                &output["call:f{".len()..],
                "{output:?} cut {cut}"
            );
            let others: Vec<(String, MalformedReason)> = malformed(&events)
                .into_iter()
                .filter(|(_, why)| *why != MalformedReason::EmptyKey)
                .collect();
            assert_eq!(
                others,
                if cut {
                    vec![(String::new(), MalformedReason::UnterminatedRegion)]
                } else {
                    Vec::new()
                },
                "{output:?} cut {cut}: {events:?}"
            );
        }
    }

    /// A string the closing marker or the end of the stream cuts: the fragments sent stand, no
    /// closing token and no closing brace are made up, and one `Malformed` event of no bytes
    /// before the call's end says that the arguments are no whole value, with the value's reason
    /// at the marker and as the cut region at the end; a string nested in a list the same. Bytes
    /// held as a possible start of the string token are the string's text at either end. A
    /// string the model closed needs no word.
    #[test]
    fn a_string_cut_by_the_calls_end_stays_open_and_nothing_is_made_up() {
        for (output, expected) in [
            ("call:get_weather{city:<|\"|>Par", r#"{"city": "Par"#),
            ("call:f{tags:[<|\"|>a<|\"|>,<|\"|>b", r#"{"tags": ["a", "b"#),
            ("call:f{city:<|\"|>", r#"{"city": ""#),
        ] {
            let closed = run(&[output], false);
            assert_eq!(arguments(&closed), expected, "{output:?}");
            assert_eq!(
                malformed(&closed),
                vec![(String::new(), MalformedReason::ValueCutShort)],
                "{output:?}"
            );
            assert!(
                matches!(
                    &closed[closed.len() - 2..],
                    [Event::Malformed { .. }, Event::ToolCallEnd { source, .. }]
                        if source.text == CLOSE
                ),
                "{output:?}: {closed:?}"
            );
            assert_eq!(bytes(&closed), format!("{output}{CLOSE}"));
            let cut = run(&[output], true);
            assert_eq!(arguments(&cut), expected, "{output:?} cut");
            assert!(
                matches!(
                    &cut[cut.len() - 2..],
                    [
                        Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                        Event::ToolCallEnd { source, .. }
                    ] if text.text.is_empty() && source.text.is_empty()
                ),
                "{output:?} cut: {cut:?}"
            );
            assert_eq!(bytes(&cut), output, "{output:?} cut");
        }
        let held = run(&["call:f{city:<|\"|>Par<|"], false);
        assert_eq!(arguments(&held), r#"{"city": "Par<|"#);
        assert_eq!(
            malformed(&held),
            vec![(String::new(), MalformedReason::ValueCutShort)]
        );
        let held = run(&["call:f{city:<|\"|>Par<|"], true);
        assert_eq!(arguments(&held), r#"{"city": "Par<|"#);
        assert_eq!(
            malformed(&held),
            vec![(String::new(), MalformedReason::UnterminatedRegion)]
        );
        let whole = run(&["call:f{city:<|\"|>Paris<|\"|>"], false);
        assert_eq!(arguments(&whole), r#"{"city": "Paris"}"#);
        assert!(malformed(&whole).is_empty());
    }

    /// A member the call's end cuts between its colon and its value is the cut call's case, not
    /// a member without a value: the key held for the value comes back as `Malformed` before the
    /// call's end at the closing marker, with the value's reason, nothing is made up for the
    /// value, and the object closes whole without the member; at the end of the stream the key
    /// opens the member, nothing closes, and the one word of no bytes says so. A list closed
    /// where a value may come is an empty list, and an object closed after a comma is whole: no
    /// word for those at the marker, and at the stream's end they stay open with the word.
    #[test]
    fn a_member_cut_before_its_value_is_reported_and_nothing_is_made_up() {
        let closed = run(&["call:get_weather{city:"], false);
        assert_eq!(arguments(&closed), "{}");
        assert_eq!(
            malformed(&closed),
            vec![("city:".to_string(), MalformedReason::ValueCutShort)]
        );
        assert!(
            matches!(
                &closed[closed.len() - 3..],
                [
                    Event::Malformed { .. },
                    Event::ToolCallArguments { json, .. },
                    Event::ToolCallEnd { source, .. }
                ] if json == "}" && source.text == CLOSE
            ),
            "{closed:?}"
        );
        assert_eq!(bytes(&closed), format!("call:get_weather{{city:{CLOSE}"));
        let second = run(&["call:get_weather{city:<|\"|>Paris<|\"|>,days:"], false);
        assert_eq!(arguments(&second), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&second),
            vec![("days:".to_string(), MalformedReason::ValueCutShort)]
        );
        let spaced = run(&["call:get_weather{city: "], false);
        assert_eq!(arguments(&spaced), "{}");
        assert_eq!(
            malformed(&spaced),
            vec![("city: ".to_string(), MalformedReason::ValueCutShort)]
        );
        // At the stream's end the key held for the value opens the member, nothing is made up
        // for the value, and the object stays open: the one word of no bytes says so.
        let cut = run(&["call:get_weather{city:"], true);
        assert_eq!(arguments(&cut), r#"{"city": "#);
        assert_eq!(
            malformed(&cut),
            vec![(String::new(), MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(bytes(&cut), "call:get_weather{city:");
        for (output, closed, open) in [
            ("call:f{tags:[", r#"{"tags": []}"#, r#"{"tags": ["#),
            ("call:f{a:1,", r#"{"a": 1}"#, r#"{"a": 1"#),
            ("call:f{", "{}", "{"),
        ] {
            let events = run(&[output], false);
            assert_eq!(arguments(&events), closed, "{output:?}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{output:?}: {events:?}"
            );
            let events = run(&[output], true);
            assert_eq!(arguments(&events), open, "{output:?} cut");
            assert_eq!(
                malformed(&events),
                vec![(String::new(), MalformedReason::UnterminatedRegion)],
                "{output:?} cut"
            );
        }
    }

    /// A container opened past the prefix parser's depth bound: the fragments stop where the
    /// parser still takes them, the bytes from that bracket to the call's end are reported as
    /// invalid arguments, and the marker closes what is open (the stream's end leaves it open),
    /// whole and in pieces alike. Nesting up to the bound reads as before.
    #[test]
    fn nesting_past_the_prefix_parsers_bound_is_reported_from_the_bracket_that_passes_it() {
        let deep = |levels: usize| {
            format!(
                "call:f{{days:{}1{}}}",
                "[".repeat(levels),
                "]".repeat(levels)
            )
        };
        let within = deep(DEEPEST - 1);
        for cut in [false, true] {
            let events = run(&[&within], cut);
            let json = arguments(&events);
            assert!(
                PartialJson::default().parse(&json, false).is_ok(),
                "cut {cut}: {json}"
            );
            assert!(
                !events.iter().any(|e| matches!(e, Event::Malformed { .. })),
                "cut {cut}"
            );
        }
        // The refused bracket as an object member's value: the key held for it is reported with
        // the bytes from the bracket on, nothing of the member is written, and the objects close
        // whole without it; the reported bytes run from the key to the object's own close.
        let objects = |levels: usize| {
            format!(
                "call:f{{{}a:1{}}}",
                "a:{".repeat(levels),
                "}".repeat(levels)
            )
        };
        let within_objects = objects(DEEPEST - 1);
        let events = run(&[&within_objects], false);
        assert!(PartialJson::default()
            .parse(&arguments(&events), false)
            .is_ok());
        assert!(!events.iter().any(|e| matches!(e, Event::Malformed { .. })));
        let past_objects = objects(DEEPEST);
        for cut in [false, true] {
            let events = run(&[&past_objects], cut);
            let json = arguments(&events);
            assert!(
                PartialJson::default().parse(&json, false).is_ok(),
                "objects cut {cut}: a prefix the parser takes: {json}"
            );
            let written = format!("{{{}", "\"a\": {".repeat(DEEPEST - 1));
            assert_eq!(
                json,
                if cut {
                    written.clone()
                } else {
                    format!("{written}{}", "}".repeat(DEEPEST))
                },
                "objects cut {cut}: the refused member not written; the objects close whole at \
                 the marker and stay open at the stream's end"
            );
            let reported: String = events
                .iter()
                .filter_map(|e| match e {
                    Event::Malformed {
                        text,
                        why: MalformedReason::InvalidArguments,
                    } => Some(&*text.text),
                    _ => None,
                })
                .collect();
            assert_eq!(
                reported,
                format!("a:{{a:1{}}}", "}".repeat(DEEPEST)),
                "objects cut {cut}: from the refused member's key to the object's own close"
            );
        }
        let past = deep(DEEPEST);
        let per_char: Vec<&str> = past
            .char_indices()
            .map(|(i, c)| &past[i..i + c.len_utf8()])
            .collect();
        for (how, pieces) in [("whole", vec![past.as_str()]), ("per character", per_char)] {
            for cut in [false, true] {
                let events = run(&pieces, cut);
                let json = arguments(&events);
                assert!(
                    PartialJson::default().parse(&json, false).is_ok(),
                    "{how} cut {cut}: the fragments stay a value the prefix parser takes: {json}"
                );
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&json).is_ok(),
                    !cut,
                    "{how} cut {cut}: the marker closes what is open, the stream's end does \
                     not: {json}"
                );
                let reported: String = events
                    .iter()
                    .filter_map(|e| match e {
                        Event::Malformed {
                            text,
                            why: MalformedReason::InvalidArguments,
                        } => Some(&*text.text),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    reported,
                    format!("[1{}}}", "]".repeat(DEEPEST)),
                    "{how} cut {cut}: from the bracket past the bound to the object's own close"
                );
                assert_eq!(
                    bytes(&events),
                    if cut {
                        past.clone()
                    } else {
                        format!("{past}{CLOSE}")
                    },
                    "{how} cut {cut}"
                );
            }
        }
        // The bound counts what is written: a member without a key is reported and not written,
        // so the containers opened inside it, however deep, refuse nothing, and the members after
        // it are written as before.
        let skipped = format!(
            "call:f{{:{}1{},b:1}}",
            "[".repeat(DEEPEST + 8),
            "]".repeat(DEEPEST + 8)
        );
        for cut in [false, true] {
            let events = run(&[&skipped], cut);
            assert_eq!(arguments(&events), "{\"b\": 1}", "skipped cut {cut}");
            assert!(
                malformed(&events)
                    .iter()
                    .all(|(_, why)| *why == MalformedReason::EmptyKey),
                "skipped cut {cut}: {:?}",
                malformed(&events)
            );
            assert_eq!(
                bytes(&events),
                if cut {
                    skipped.clone()
                } else {
                    format!("{skipped}{CLOSE}")
                },
                "skipped cut {cut}"
            );
        }
    }

    #[test]
    fn what_is_not_a_call_is_reported_and_so_is_text_after_the_object() {
        let not = run(&["hello{x:1}"], false);
        assert!(!not
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
        assert_eq!(bytes(&not), format!("hello{{x:1}}{CLOSE}"));
        assert_eq!(malformed(&not)[0].1, MalformedReason::TextBetweenCalls);
        // The space before the text is the template's wrapping, the text is reported: the same
        // two events however the bytes are cut.
        for pieces in [&["call:f{x:1} trailing"][..], &["call:f{x:1} ", "trailing"]] {
            let tail = run(pieces, false);
            assert_eq!(arguments(&tail), r#"{"x": 1}"#);
            assert_eq!(
                malformed(&tail),
                vec![(
                    "trailing".to_string(),
                    MalformedReason::TextAfterTheArguments
                )]
            );
            assert!(tail.iter().any(|event| matches!(
                event,
                Event::Dropped { text, why: DropReason::Wrapper } if text.text == " "
            )));
            assert_eq!(bytes(&tail), format!("call:f{{x:1}} trailing{CLOSE}"));
        }
    }

    /// Several words after the object: the space right after the object is the template's and
    /// is dropped, the words keep the spaces between them (one event per word, the space before
    /// it in the event), the space before the marker is the template's again; the same text
    /// however the bytes are cut, and at a stream's end the held space is wrapping, not a cut.
    #[test]
    fn text_after_the_object_keeps_the_models_whitespace_between_its_words() {
        let text = "call:f{x:1} please note this and that ";
        let words = |events: &[Event]| -> Vec<(String, MalformedReason)> { malformed(events) };
        let dropped = |events: &[Event]| -> Vec<String> {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Dropped {
                        text,
                        why: DropReason::Wrapper,
                    } => Some(text.text.to_string()),
                    _ => None,
                })
                .collect()
        };
        let expected: Vec<(String, MalformedReason)> =
            ["please", " note", " this", " and", " that"]
                .into_iter()
                .map(|word| (word.to_string(), MalformedReason::TextAfterTheArguments))
                .collect();
        let whole = run(&[text], false);
        assert_eq!(bytes(&whole), format!("{text}{CLOSE}"));
        assert_eq!(arguments(&whole), r#"{"x": 1}"#);
        assert_eq!(words(&whole), expected);
        assert_eq!(dropped(&whole), [" ", " "]);
        for cut in 1..text.len() {
            let events = run(&[&text[..cut], &text[cut..]], false);
            assert_eq!(bytes(&events), format!("{text}{CLOSE}"), "cut at {cut}");
            assert_eq!(
                words(&events)
                    .iter()
                    .map(|(word, _)| word.as_str())
                    .collect::<String>(),
                "please note this and that",
                "cut at {cut}"
            );
            assert_eq!(dropped(&events), [" ", " "], "cut at {cut}");
        }
        let pieces: Vec<String> = text.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let events = run(&refs, false);
        assert_eq!(bytes(&events), format!("{text}{CLOSE}"));
        assert_eq!(dropped(&events), [" ", " "]);
        assert_eq!(
            words(&events)
                .iter()
                .map(|(word, _)| word.as_str())
                .collect::<String>(),
            "please note this and that"
        );
        // The stream cut after the prose: the held space is the template's, nothing is reported
        // as cut, and the call still ends.
        let events = run(&["call:f{x:1} please note "], true);
        assert_eq!(bytes(&events), "call:f{x:1} please note ");
        assert_eq!(words(&events), expected[..2].to_vec());
        assert_eq!(dropped(&events), [" ", " "]);
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
    }

    #[test]
    fn a_header_that_is_no_call_says_which_shape_it_saw() {
        for (text, why) in [
            ("hello{x:1}", MalformedReason::TextBetweenCalls),
            ("call:{x:1}", MalformedReason::CallWithoutName),
            (
                "call:get weather{x:1}",
                MalformedReason::TextWhereTheNameShouldBe,
            ),
            ("call:", MalformedReason::CallWithoutName),
            ("call:get_weather", MalformedReason::NameWithoutACall),
        ] {
            let reasons = malformed(&run(&[text], false));
            assert_eq!(reasons[0].1, why, "{text:?}");
            assert_eq!(
                reasons.last().map(|(_, why)| *why),
                Some(MalformedReason::BlockWithoutACompleteCall),
                "{text:?}: the close with no call started"
            );
        }
    }

    #[test]
    fn a_bare_value_that_is_no_literal_is_kept_as_a_string() {
        let events = run(&["call:f{mode:fast,n:12}"], false);
        assert_eq!(arguments(&events), r#"{"mode": "fast", "n": 12}"#);
    }

    /// A call the end of the stream cut inside a bare value, after a key, inside a key, between
    /// members, inside a nested object or a list, inside a string and right after the opening
    /// brace: what the model wrote of the member leaves as the call's own fragment, its key and
    /// its bytes as written; a key cut short, the whitespace after a value and a start of the
    /// string token after a key leave as wrapping; no container closes; nothing of the call
    /// reaches a chat client as content; and the one `Malformed` event of no bytes still says
    /// that the arguments are no whole value before the call's end; whole and one character at
    /// a time.
    #[test]
    fn a_member_the_stream_end_cuts_leaves_as_the_calls_own_bytes_and_nothing_closes() {
        let cases: [(&str, &str, &[&str]); 25] = [
            // A bare value: the digits so far, a word that may still be a literal.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:1",
                r#"{"artist": "Taylor", "n": 1"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:12",
                r#"{"artist": "Taylor", "n": 12"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:tr",
                r#"{"artist": "Taylor", "n": tr"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:-1",
                r#"{"artist": "Taylor", "n": -1"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:-",
                r#"{"artist": "Taylor", "n": -"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:nu",
                r#"{"artist": "Taylor", "n": nu"#,
                &[],
            ),
            // A word that is no literal: the opening of the string it would be written as, as
            // `push_bare` writes it once the member closes: Python's words and quotes too, and a
            // JSON quote where the template writes its own string token.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:hel",
                r#"{"artist": "Taylor", "n": "hel"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:True",
                r#"{"artist": "Taylor", "n": "True"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:None",
                r#"{"artist": "Taylor", "n": "None"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:'ab",
                r#"{"artist": "Taylor", "n": "'ab"#,
                &[],
            ),
            ("call:play{city:\"Par", r#"{"city": "\"Par"#, &[]),
            // The key and its colon came, and no byte of the value.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n:",
                r#"{"artist": "Taylor", "n": "#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n: ",
                r#"{"artist": "Taylor", "n": "#,
                &[],
            ),
            // A key cut short: wrapping.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,n",
                r#"{"artist": "Taylor""#,
                &["n"],
            ),
            // Between members, after a value, and the whitespace held after one.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,",
                r#"{"artist": "Taylor""#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>",
                r#"{"artist": "Taylor""#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|> ",
                r#"{"artist": "Taylor""#,
                &[" "],
            ),
            // Inside a string, with and without a start of the closing token held.
            ("call:play{artist:<|\"|>Tay", r#"{"artist": "Tay"#, &[]),
            ("call:play{artist:<|\"|>Tay<|", r#"{"artist": "Tay<|"#, &[]),
            // A start of the string token after the key: the member opens, the token leaves
            // after the call's end.
            ("call:play{artist:<|\"", r#"{"artist": "#, &["<|\""]),
            // Nested: a string and a bare value inside an object, a bare value in a list.
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,prefs:{size:<|\"|>lar",
                r#"{"artist": "Taylor", "prefs": {"size": "lar"#,
                &[],
            ),
            (
                "call:play{artist:<|\"|>Taylor<|\"|>,prefs:{size:2",
                r#"{"artist": "Taylor", "prefs": {"size": 2"#,
                &[],
            ),
            (
                "call:play{tags:[<|\"|>a<|\"|>,1",
                r#"{"tags": ["a", 1"#,
                &[],
            ),
            ("call:play{tags:[", r#"{"tags": ["#, &[]),
            // Right after the opening brace.
            ("call:play{", "{", &[]),
        ];
        for (call, expected, wrapping) in cases {
            let characters: Vec<&str> = call
                .char_indices()
                .map(|(at, c)| &call[at..at + c.len_utf8()])
                .collect();
            for (how, pieces) in [("whole", vec![call]), ("by character", characters)] {
                let events = run(&pieces, true);
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
                assert_eq!(
                    malformed(&events),
                    vec![(String::new(), MalformedReason::UnterminatedRegion)],
                    "{call:?} {how}: {events:?}"
                );
                let ended = events
                    .iter()
                    .position(|event| matches!(event, Event::ToolCallEnd { .. }))
                    .expect("the call ends");
                assert!(
                    matches!(
                        &events[ended - 1..],
                        [Event::Malformed { .. }, Event::ToolCallEnd { source, .. }]
                            | [
                                Event::Malformed { .. },
                                Event::ToolCallEnd { source, .. },
                                Event::Dropped { why: DropReason::Wrapper, .. }
                            ] if source.text.is_empty()
                    ),
                    "{call:?} {how}: {events:?}"
                );
                let dropped_after_the_start: Vec<&str> = events
                    .iter()
                    .skip_while(|event| !matches!(event, Event::ToolCallStart { .. }))
                    .filter_map(|event| match event {
                        Event::Dropped { text, .. } => Some(&*text.text),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    dropped_after_the_start, wrapping,
                    "{call:?} {how}: {events:?}"
                );
            }
        }
        // An object the model closed before the cut is whole: no word, and the whitespace after
        // it is the template's.
        let closed = run(&["call:play{artist:<|\"|>Taylor<|\"|>} "], true);
        assert_eq!(arguments(&closed), r#"{"artist": "Taylor"}"#);
        assert!(malformed(&closed).is_empty(), "{closed:?}");
        // The start of the spelling's null word is the start of `null`, under either spelling;
        // the start of the JSON word `null` is one only where it is the spelling's: under
        // FunctionGemma a whole `null` is a string, so its cut opens that string, as `n:hel`
        // does, and the value's type is the one the member closes with.
        let function_gemma = |call: &str| {
            let mut assembler = Assembler::new(0, "call_0", Spelling::FUNCTION_GEMMA);
            let mut out = Events::new();
            assembler.feed(call, &mut out);
            assembler.finish(&mut out);
            arguments(&out.drain())
        };
        assert_eq!(function_gemma("call:f{n:No"), r#"{"n": nu"#);
        assert_eq!(function_gemma("call:f{n:nul"), r#"{"n": "nul"#);
        assert_eq!(function_gemma("call:f{n:null"), r#"{"n": "null"#);
        assert_eq!(function_gemma("call:f{n:True"), r#"{"n": "True"#);
        assert_eq!(function_gemma("call:f{n:tr"), r#"{"n": tr"#);
        assert_eq!(function_gemma("call:f{n:"), r#"{"n": "#);
        let null = run(&["call:f{n:nul"], true);
        assert_eq!(arguments(&null), r#"{"n": nul"#, "{null:?}");
        // A header that was no call's, its text after it released as it came: the whitespace
        // held at the end is the template's, as at the marker, never shown.
        let rejected = run(&["cal:x{ hello  "], true);
        assert_eq!(bytes(&rejected), "cal:x{ hello  ");
        assert!(
            !rejected.iter().any(|event| matches!(
                event,
                Event::Malformed {
                    why: MalformedReason::UnterminatedRegion,
                    ..
                }
            )),
            "{rejected:?}"
        );
        assert!(
            rejected.iter().any(|event| matches!(
                event,
                Event::Dropped { text, why: DropReason::Wrapper } if text.text == "  "
            )),
            "{rejected:?}"
        );
        // A header the cut found before its brace: no call started, and its bytes come back as
        // the cut region's, as every assembler reports a block that named no call.
        let header = run(&["call:pl"], true);
        assert_eq!(
            malformed(&header),
            vec![("call:pl".to_string(), MalformedReason::UnterminatedRegion)]
        );
        assert!(!header
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
    }
}
