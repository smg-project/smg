//! Whether argument bytes are still a prefix of JSON a strict reader accepts, judged as they
//! arrive.
//!
//! [`Prefix`] takes the bytes of one JSON value in pieces and says where, if anywhere, they stop
//! being a prefix of a value a strict reader (`serde_json`, a client's JSON library, the engine's
//! own) accepts. It stops where [`PartialJson`](super::PartialJson) stops, and earlier, at the
//! first byte of what that parser tolerates and JSON lacks, so the promise `ToolCallArguments`
//! makes, that the fragments so far are a prefix of JSON a strict reader accepts, holds at every
//! byte and not only at a call's end; a test runs the judge, the parser and a strict reader over
//! every cut of a corpus of sound and broken values. It keeps its place between pieces and reads
//! each byte once, where the parser read the whole text again on every piece, which made a long
//! arguments string cost the square of its length.
//!
//! What it takes is what a strict reader takes, cut anywhere: a literal's prefix at the end
//! (`nu`), a number cut anywhere (`-`, `1.`, `1e+`), a string cut anywhere, and the one thing JSON
//! lacks that it lets through, a raw control character inside a string, which the fragment that
//! carries it writes as its escape ([`Prefix::escaped`]), so that what a client assembles is JSON
//! again. One place it is wider than `serde_json`: it reads a number's grammar, not its size, so a
//! number out of the reader's range (`1e400`, an integer past `f64`'s range) passes, and a call
//! that carries one closes whole while a client's reader may refuse the number; the templates' own
//! readers differ there too (Python's reads it as infinity), so the judge leaves it to the client.
//! What it refuses, at the first byte a strict reader refuses: a value left out before a
//! comma or a close (`[,,]`, `{"a": }`), a trailing comma before a close (`[1, ]`), a bracket
//! closed by the wrong kind (`{"a": [1}`), whitespace JSON lacks between tokens (a no-break
//! space), a number or a word cut short before a delimiter (`1.}`, `nu}`), a container nested
//! past the depth limit, and anything after the value; and from the first byte of what the
//! readers hold until it is decided, which the strict reader refuses a byte or more later: a word
//! that is no literal, from its first letter (as the parser rolls back to it), and an escape JSON
//! lacks (`\q`), a `\u` escape with fewer than four hex digits and a surrogate escape without its
//! other half, from the escape's backslash. The partial parser tolerates most of these, as the old
//! crate did (it refuses the depth, the bytes after the value, the word that is no literal and
//! the surrogate), which is right for a prefix that is still arriving and wrong at a call's close,
//! where the fragments are handed over as the model's whole arguments and no strict reader takes
//! them; where the judge refuses, the readers say so once, by a `Malformed { InvalidArguments }`
//! event of no bytes, and send the bytes from there as the model wrote them, nothing twice.

use std::{borrow::Cow, cmp::Ordering};

/// The nesting the prefix parser allows, which is [`PartialJson::DEFAULT_MAX_DEPTH`].
///
/// [`PartialJson::DEFAULT_MAX_DEPTH`]: crate::json::PartialJson::DEFAULT_MAX_DEPTH
const DEEPEST: usize = 32;

/// The whitespace JSON allows between tokens and around a value: these four characters and no
/// other, where the old crate took any Unicode whitespace.
pub(crate) fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// A JSON prefix being judged as its bytes arrive.
#[derive(Clone, Debug)]
pub struct Prefix {
    /// The containers open around the position, outermost first.
    open: Vec<Container>,
    /// What the grammar admits next.
    expect: Expect,
    /// The token being read, when a string, a literal or a number is under way.
    token: Token,
    /// Bytes judged so far.
    at: usize,
    /// Where the bytes stopped being a prefix, once they did.
    invalid: Option<usize>,
    /// Where the top-level value ended, once something ended it.
    ended: Option<usize>,
    /// The raw control characters met inside strings, in order: JSON forbids them there, and an
    /// assembler writes the fragment that carries one with its escape ([`Self::escaped`]).
    /// Empty for all but the rare output that has one.
    controls: Vec<Control>,
}

/// A raw control character (U+0000..U+001F) inside a string.
#[derive(Clone, Copy, Debug)]
struct Control {
    /// Its byte offset in the whole text.
    at: usize,
    /// Whether a backslash stood right before it, which then opens the escape written for it.
    escaped: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// A value: at the top, after a colon, or after a comma in an array. A comma or a close here
    /// is a value left out, which JSON lacks.
    Value,
    /// An array's first value, or its close (`[]`).
    ValueOrClose,
    /// An object's key, after a comma. A close here is a trailing comma, which JSON lacks.
    Key,
    /// An object's first key, or its close (`{}`).
    KeyOrClose,
    /// The colon after a key.
    Colon,
    /// A comma or the container's close, after a value.
    CommaOrClose,
    /// Nothing: the top-level value is complete.
    End,
}

#[derive(Clone, Copy, Debug)]
enum Token {
    None,
    /// A string, as a key when `key` is set.
    Str {
        state: Str,
        key: bool,
    },
    /// `true`, `false` or `null` being spelled; `start` is where it began, `len` how many of its
    /// letters have matched so far.
    Lit {
        start: usize,
        expected: &'static str,
        len: usize,
    },
    Num(Num),
}

#[derive(Clone, Copy, Debug)]
enum Str {
    Text,
    /// After a backslash.
    Escape,
    /// Inside `\u`, with the hex digits read so far and their value.
    Unicode {
        digits: u8,
        value: u32,
    },
    /// After a high surrogate's four digits: only `\u` and a low surrogate may follow.
    AfterHigh,
    /// After the backslash that follows a high surrogate: only `u` may follow; `at` is the
    /// backslash's offset, where the parser stops when something else comes.
    AfterHighBackslash {
        at: usize,
    },
    /// Inside the low surrogate's `\u`.
    Low {
        digits: u8,
        value: u32,
    },
}

#[derive(Clone, Copy, Debug)]
enum Num {
    /// After `-`: a digit must come.
    Sign,
    /// After a leading `0`: only a fraction or an exponent may follow.
    Zero,
    Integer,
    /// After `.`: a digit must come.
    Dot,
    /// Inside the fraction's digits.
    Fraction,
    /// After `e` or `E`: a sign or a digit must come.
    Exponent,
    /// After the exponent's sign: a digit must come.
    ExponentSign,
    ExponentDigits,
}

impl Num {
    /// Whether the number is whole as it stands: a strict reader takes it, and would take more
    /// digits.
    fn whole(self) -> bool {
        matches!(
            self,
            Num::Zero | Num::Integer | Num::Fraction | Num::ExponentDigits
        )
    }
}

impl Default for Prefix {
    fn default() -> Self {
        Self {
            open: Vec::new(),
            expect: Expect::Value,
            token: Token::None,
            at: 0,
            invalid: None,
            ended: None,
            controls: Vec::new(),
        }
    }
}

impl Prefix {
    /// Judge the next bytes, which follow every byte judged before, and return where the bytes
    /// stopped being a prefix of a value, if they have: an offset into the whole text, never past
    /// what has been fed, but anywhere at or after the start, bytes judged sound before included,
    /// since the parser rolls back to a word's first letter when the word is no literal, to the
    /// backslash of an escape it refuses (one JSON lacks, a `\u` escape short of its digits, a
    /// surrogate without its other half), and to the value's start when a top-level string
    /// breaks. A caller that has already passed on the sound bytes clamps the
    /// answer to what it passed on, as the assembler does.
    pub fn feed(&mut self, bytes: &str) -> Option<usize> {
        for (offset, c) in bytes.char_indices() {
            if self.invalid.is_some() {
                break;
            }
            let at = self.at + offset;
            self.step(at, c);
            if self.ended.is_none() && self.expect == Expect::End {
                // The character closed the value, or came after a number or a word that it ended
                // and was refused, as nothing may follow the value.
                self.ended = Some(if self.invalid == Some(at) {
                    at
                } else {
                    at + c.len_utf8()
                });
            }
        }
        if self.invalid.is_none() {
            self.at += bytes.len();
        }
        self.invalid
    }

    /// Where the top-level value ended, once something ended it: just past its closing quote or
    /// bracket, or before the character that ended a number or a word. The bytes up to there are
    /// one whole value a strict reader accepts, since the judge ends no other ([`Self::whole`]).
    pub fn ended(&self) -> Option<usize> {
        self.ended
    }

    /// Whether the bytes so far are one whole JSON value a strict reader accepts, as they stand:
    /// the value closed, or a number or a literal whole at the end of the bytes (`12`, `null`),
    /// which more bytes could still extend or refuse. Not where the bytes stopped being a prefix,
    /// nor while a string, a container, a word or a number is open (`"ab`, `[1,`, `nu`, `1.`).
    pub fn whole(&self) -> bool {
        if self.invalid.is_some() || !self.open.is_empty() {
            return false;
        }
        match self.token {
            Token::None => self.expect == Expect::End,
            Token::Lit { expected, len, .. } => len == expected.len(),
            Token::Num(state) => state.whole(),
            Token::Str { .. } => false,
        }
    }

    /// Where the word under way starts, while one is: a literal being spelled (`nu`), which is a
    /// prefix of `null` so far, and which the judge refuses from its first letter if it turns out
    /// to be no literal; at the top, from the value's start, as the parser refuses the whole
    /// value then.
    pub fn word_start(&self) -> Option<usize> {
        match self.token {
            Token::Lit { start, .. } if self.invalid.is_none() => {
                Some(if self.open.is_empty() { 0 } else { start })
            }
            _ => None,
        }
    }

    /// Where the escape under way starts, while one is: the backslash of an escape whose letter
    /// has not come, of a `\u` escape whose four hex digits are not all in, and of the `\u` escape
    /// after a high surrogate until it is the low half. The judge refuses at that backslash when
    /// the escape turns out to be one JSON lacks (`\q`), a `\u` escape short of its four hex
    /// digits, or a surrogate without its other half, so an assembler holds from there, and no
    /// fragment ends inside an escape; a letter JSON knows, four digits that are a character, and
    /// a high surrogate's four digits release the hold.
    pub fn escape_start(&self) -> Option<usize> {
        if self.invalid.is_some() {
            return None;
        }
        match self.token {
            Token::Str {
                state: Str::Escape, ..
            } => Some(self.at - 1),
            Token::Str {
                state: Str::Unicode { digits, .. } | Str::Low { digits, .. },
                ..
            } => Some(self.at - 2 - usize::from(digits)),
            Token::Str {
                state: Str::AfterHighBackslash { at },
                ..
            } => Some(at),
            _ => None,
        }
    }

    /// Where the judge may still go back to, if anywhere: the earliest of the word's start
    /// ([`Self::word_start`]) and the escape's ([`Self::escape_start`]), and the value's start
    /// while nothing but whitespace has come at the top, since a word may yet come and have the
    /// whole value refused, whitespace included. An assembler that passes bytes on as fragments
    /// holds from there, since a fragment passed on cannot be taken back.
    ///
    /// A top-level string is not held, though the parser refuses it whole when a surrogate
    /// escape breaks inside it: a string-valued arguments (a stringified call object) streams
    /// as written, and a surrogate that breaks inside one is said once at the break, after the
    /// string's bytes before it, whole and in pieces alike.
    pub fn held_from(&self) -> Option<usize> {
        let before_the_value = self.open.is_empty()
            && self.expect == Expect::Value
            && matches!(self.token, Token::None)
            && self.invalid.is_none();
        [
            self.word_start(),
            self.escape_start(),
            if before_the_value { Some(0) } else { None },
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// `fragment`, the judged bytes from offset `from` on, with every raw control character the
    /// judge met inside a string there written as its JSON escape: `\n`, `\t`, `\r`, `\b`, `\f`
    /// or `\u00XX`, and after a backslash the escape without its own backslash, which already
    /// stands (`\` and a newline read as a newline, and are written `\n`). The bytes themselves,
    /// borrowed, where it met none, which is nearly every fragment. An assembler sends the result
    /// as the fragment's `json` and keeps the model's bytes as its `source`: what a client
    /// assembles is then JSON a strict reader accepts, which a raw control character is not.
    pub fn escaped<'a>(&self, from: usize, fragment: &'a str) -> Cow<'a, str> {
        let end = from + fragment.len();
        let first = self.controls.partition_point(|control| control.at < from);
        if self
            .controls
            .get(first)
            .is_none_or(|control| control.at >= end)
        {
            return Cow::Borrowed(fragment);
        }
        let mut out = String::with_capacity(fragment.len() + 8);
        let mut copied = 0;
        for control in self.controls[first..]
            .iter()
            .take_while(|control| control.at < end)
        {
            let at = control.at - from;
            out.push_str(&fragment[copied..at]);
            if !control.escaped {
                out.push('\\');
            }
            match fragment.as_bytes()[at] {
                b'\n' => out.push('n'),
                b'\t' => out.push('t'),
                b'\r' => out.push('r'),
                0x08 => out.push('b'),
                0x0c => out.push('f'),
                byte => {
                    out.push_str("u00");
                    out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
                    out.push(char::from_digit(u32::from(byte & 0xf), 16).unwrap_or('0'));
                }
            }
            copied = at + 1;
        }
        out.push_str(&fragment[copied..]);
        Cow::Owned(out)
    }

    /// `text`, judged from its start, closed into the whole value its complete members make:
    /// the bytes up to where they stop being a prefix (or all of them), cut back to the end of
    /// the last member complete at every level, so that a string, a number or a word cut at the
    /// end is left out with the key or the comma it was to follow, and the containers still open
    /// closed. A whole value comes back as it stands, the whitespace before it included and the
    /// whitespace after it, which the judge refuses, left off; `None` when nothing complete stands
    /// in `text`: whitespace alone, or a scalar cut short at the top (`"ab`, `nul`, `1.`). What
    /// comes back is one whole value a strict reader accepts (a number out of the reader's range
    /// excepted, as at [`Self::feed`]).
    ///
    /// This is the value a client's partial JSON reader ends with from the fragments an assembler
    /// sent: the prefix the judge accepted, complete members only. The Messages adapter reads a
    /// call's `input` from it where the fragments are no whole JSON (the output cut, or the model
    /// wrote something else after the prefix), so that the whole message carries what the stream
    /// gave; a rare path, run once per such call over its arguments.
    pub fn closed(text: &str) -> Option<String> {
        let mut judge = Prefix::default();
        // Up to where each level's content is complete members: the top first, then one entry
        // per open container, each starting right after its opening bracket.
        let mut done = vec![0usize];
        for (at, c) in text.char_indices() {
            let depth = judge.open.len();
            let token = judge.token;
            judge.step(at, c);
            // A number or a word that `c` ended, or a string that `c` closed, is a complete
            // member of the innermost container, whatever the judge then said of `c`.
            let end = match (token, judge.token) {
                (Token::Num(_) | Token::Lit { .. }, Token::None) => Some(at),
                (Token::Str { key: false, .. }, Token::None) => Some(at + c.len_utf8()),
                _ => None,
            };
            if let (Some(end), Some(level)) = (end, done.get_mut(depth)) {
                *level = end;
            }
            if judge.invalid.is_some() {
                break;
            }
            match judge.open.len().cmp(&depth) {
                Ordering::Greater => done.push(at + c.len_utf8()),
                Ordering::Less => {
                    done.pop();
                    if let Some(parent) = done.last_mut() {
                        *parent = at + c.len_utf8();
                    }
                }
                Ordering::Equal => {}
            }
        }
        let complete = judge.invalid.is_none()
            && match judge.token {
                Token::None => matches!(judge.expect, Expect::CommaOrClose | Expect::End),
                Token::Num(state) => state.whole(),
                Token::Lit { expected, len, .. } => len == expected.len(),
                Token::Str { .. } => false,
            };
        let cut = if complete { text.len() } else { *done.last()? };
        if cut == 0 {
            return None;
        }
        let mut out = String::with_capacity(cut + judge.open.len());
        out.push_str(&text[..cut]);
        for container in judge.open.iter().rev() {
            out.push(match container {
                Container::Object => '}',
                Container::Array => ']',
            });
        }
        Some(out)
    }

    /// One character, at byte offset `at` of the whole text.
    fn step(&mut self, at: usize, c: char) {
        match self.token {
            Token::Str { state, key } => self.in_string(state, key, at, c),
            Token::Lit {
                start,
                expected,
                len,
            } => self.in_literal(start, expected, len, at, c),
            Token::Num(state) => self.in_number(state, at, c),
            Token::None => self.between(at, c),
        }
    }

    /// A character where no token is under way.
    fn between(&mut self, at: usize, c: char) {
        // Between tokens JSON allows four whitespace characters and no other: a no-break space or
        // an ideographic space, which the partial parser skipped, is refused as a strict reader
        // refuses it.
        if self.expect != Expect::End && json_space(c) {
            return;
        }
        match self.expect {
            Expect::Value | Expect::ValueOrClose => self.value(at, c),
            Expect::Key | Expect::KeyOrClose => match c {
                '}' if self.expect == Expect::KeyOrClose => self.close(Container::Object),
                '"' => {
                    self.token = Token::Str {
                        state: Str::Text,
                        key: true,
                    };
                }
                // A close after a comma (`{"a": 1, }`) is a trailing comma, which the partial
                // parser took and JSON lacks; anything else is no key.
                _ => self.refuse(at),
            },
            Expect::Colon => {
                if c == ':' {
                    self.expect = Expect::Value;
                } else {
                    self.refuse(at);
                }
            }
            Expect::CommaOrClose => match (self.open.last(), c) {
                (Some(Container::Object), ',') => self.expect = Expect::Key,
                (Some(Container::Array), ',') => self.expect = Expect::Value,
                (Some(Container::Object), '}') => self.close(Container::Object),
                (Some(Container::Array), ']') => self.close(Container::Array),
                // A bracket of the wrong kind, or anything else the container cannot take. The
                // partial parser returned the container to its parent as it stood and let the
                // parent judge the character, which made `{"a": [1}` a whole object; a strict
                // reader refuses the character, and so does the judge.
                _ => self.refuse(at),
            },
            Expect::End => self.refuse(at),
        }
    }

    /// A character where a value may start.
    fn value(&mut self, at: usize, c: char) {
        match c {
            '{' => self.open(at, Container::Object),
            '[' => self.open(at, Container::Array),
            '"' => {
                self.token = Token::Str {
                    state: Str::Text,
                    key: false,
                };
            }
            't' | 'f' | 'n' => {
                let expected = match c {
                    't' => "true",
                    'f' => "false",
                    _ => "null",
                };
                self.token = Token::Lit {
                    start: at,
                    expected,
                    len: 1,
                };
            }
            '-' => self.token = Token::Num(Num::Sign),
            '0' => self.token = Token::Num(Num::Zero),
            '1'..='9' => self.token = Token::Num(Num::Integer),
            ']' if self.expect == Expect::ValueOrClose => self.close(Container::Array),
            // No value starts here. A comma or a close where a value must stand (`[,`, `[1,]`,
            // `{"a": }`) is a value left out: the partial parser read `null` for it and left the
            // character to the container, and JSON has no such thing.
            _ => self.refuse(at),
        }
    }

    fn open(&mut self, at: usize, container: Container) {
        if self.open.len() >= DEEPEST {
            self.refuse(at);
            return;
        }
        self.open.push(container);
        self.expect = match container {
            Container::Object => Expect::KeyOrClose,
            Container::Array => Expect::ValueOrClose,
        };
    }

    /// The innermost container, which is `container`, closes.
    fn close(&mut self, container: Container) {
        debug_assert_eq!(self.open.last(), Some(&container));
        self.open.pop();
        self.value_done();
    }

    /// A value is complete: its container wants a comma or its close; the top wants nothing.
    fn value_done(&mut self) {
        self.token = Token::None;
        self.expect = if self.open.is_empty() {
            Expect::End
        } else {
            Expect::CommaOrClose
        };
    }

    fn in_string(&mut self, state: Str, key: bool, at: usize, c: char) {
        let next = match state {
            Str::Text => match c {
                '\\' => Str::Escape,
                '"' => {
                    if key {
                        self.token = Token::None;
                        self.expect = Expect::Colon;
                    } else {
                        self.value_done();
                    }
                    return;
                }
                // JSON forbids a raw control character here; the parser reads it as the string's
                // text, as the old crate did, and the fragment that carries it is written escaped.
                c if c < ' ' => {
                    self.controls.push(Control { at, escaped: false });
                    Str::Text
                }
                _ => Str::Text,
            },
            Str::Escape => match c {
                'u' => Str::Unicode {
                    digits: 0,
                    value: 0,
                },
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => Str::Text,
                // A backslash and a raw newline read as a newline, written `\n` with the
                // backslash that already stands.
                c if c < ' ' => {
                    self.controls.push(Control { at, escaped: true });
                    Str::Text
                }
                // An escape JSON lacks (`\q`): the partial parser read the character itself; a
                // strict reader refuses it, and the judge refuses the escape from its backslash,
                // which the readers hold ([`Self::escape_start`]).
                _ => {
                    self.refuse(at - 1);
                    return;
                }
            },
            Str::Unicode { digits, value } => match c.to_digit(16) {
                Some(digit) => {
                    let value = (value << 4) | digit;
                    if digits + 1 < 4 {
                        Str::Unicode {
                            digits: digits + 1,
                            value,
                        }
                    } else if (0xD800..0xDC00).contains(&value) {
                        Str::AfterHigh
                    } else if (0xDC00..0xE000).contains(&value) {
                        // A low surrogate alone is no character: refused at its backslash, which
                        // the readers hold ([`Self::escape_start`]); the parser stopped after
                        // its digits.
                        self.string_broken(at - 2 - usize::from(digits));
                        return;
                    } else {
                        Str::Text
                    }
                }
                // Fewer than four hex digits (`\u12G`): the partial parser took the escape as
                // unfinished and read this character as the string's text; a strict reader
                // refuses it, and the judge refuses the escape from its backslash, held likewise.
                None => {
                    self.refuse(at - 2 - usize::from(digits));
                    return;
                }
            },
            Str::AfterHigh => match c {
                '\\' => Str::AfterHighBackslash { at },
                _ => {
                    self.string_broken(at);
                    return;
                }
            },
            Str::AfterHighBackslash { at: backslash } => match c {
                'u' => Str::Low {
                    digits: 0,
                    value: 0,
                },
                _ => {
                    self.string_broken(backslash);
                    return;
                }
            },
            Str::Low { digits, value } => match c.to_digit(16) {
                Some(digit) => {
                    let value = (value << 4) | digit;
                    if digits + 1 < 4 {
                        Str::Low {
                            digits: digits + 1,
                            value,
                        }
                    } else if (0xDC00..0xE000).contains(&value) {
                        Str::Text
                    } else {
                        // Not the low half: refused at this escape's backslash, held likewise.
                        self.string_broken(at - 2 - usize::from(digits));
                        return;
                    }
                }
                None => {
                    self.refuse(at - 2 - usize::from(digits));
                    return;
                }
            },
        };
        self.token = Token::Str { state: next, key };
    }

    fn in_literal(&mut self, start: usize, expected: &'static str, len: usize, at: usize, c: char) {
        if c.is_alphabetic() {
            // The parser reads the whole word first and compares it: a word that is not the
            // literal, or more than its prefix, is refused from its first letter.
            if expected.as_bytes().get(len).copied() == Some(c as u8) && c.is_ascii() {
                self.token = Token::Lit {
                    start,
                    expected,
                    len: len + 1,
                };
            } else {
                self.refuse_word(start);
            }
            return;
        }
        // The word ends before this character. Whole, it is the literal; cut short (`nu}`), it is
        // no JSON, and is refused from its first letter like a word that is no literal, which the
        // readers hold it as until it is decided.
        if len < expected.len() {
            self.refuse_word(start);
            return;
        }
        self.value_done();
        self.step(at, c);
    }

    fn in_number(&mut self, state: Num, at: usize, c: char) {
        let next = match (state, c) {
            (Num::Sign, '0') => Some(Num::Zero),
            (Num::Sign | Num::Integer, '1'..='9') | (Num::Integer, '0') => Some(Num::Integer),
            (Num::Zero | Num::Integer, '.') => Some(Num::Dot),
            (Num::Dot | Num::Fraction, '0'..='9') => Some(Num::Fraction),
            (Num::Zero | Num::Integer | Num::Fraction, 'e' | 'E') => Some(Num::Exponent),
            (Num::Exponent, '+' | '-') => Some(Num::ExponentSign),
            (Num::Exponent | Num::ExponentSign | Num::ExponentDigits, '0'..='9') => {
                Some(Num::ExponentDigits)
            }
            _ => None,
        };
        match next {
            Some(state) => self.token = Token::Num(state),
            // The number ends before this character, which the container judges.
            None if state.whole() => {
                self.value_done();
                self.step(at, c);
            }
            // Cut short (`1.}`, `-]`, `1e+,`): the partial parser took it as a number, and no
            // strict reader does; the character that ends it is refused.
            None => self.refuse(at),
        }
    }

    /// The bytes stop being a prefix at `at`.
    fn refuse(&mut self, at: usize) {
        if self.invalid.is_none() {
            self.invalid = Some(at);
        }
    }

    /// A string broken at `at` (a surrogate escape without its other half: the backslash of the
    /// escape that is not the half the other needs, or the character after a high surrogate that
    /// is no backslash): the bytes stop being a prefix there. At the top the error is the value's
    /// own, and nothing of the value counts; the assemblers say so where they choose it
    /// ([`Self::held_from`]).
    fn string_broken(&mut self, at: usize) {
        self.refuse(if self.open.is_empty() { 0 } else { at });
    }

    /// A word that is no literal: refused from its first letter, as the parser rolls back to it;
    /// at the top the error is the value's, and nothing of the value counts.
    fn refuse_word(&mut self, start: usize) {
        self.refuse(if self.open.is_empty() { 0 } else { start });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::PartialJson;

    /// How many of `text`'s bytes are a prefix by the prefix parser's account: the bytes it
    /// consumes, or none when it refuses the value.
    fn parser_says(text: &str) -> usize {
        match PartialJson::default().parse(text, true) {
            Ok((_, consumed)) => consumed,
            Err(_) => 0,
        }
    }

    /// The same by [`Prefix`]: where it refused, or the whole text.
    fn prefix_says(text: &str) -> usize {
        Prefix::default().feed(text).unwrap_or(text.len())
    }

    /// The corpus the judge and the partial parser agree on: sound values, the broken ones both
    /// refuse at the same byte (or from a word's first letter), and the shapes the assembler's
    /// tests feed.
    const CORPUS: &[&str] = &[
        r#"{"city": "Paris", "days": [1, 2]}"#,
        r#"{"name": "test", "value": 42, "flags": [true, false, null], "ratio": -1.5e2}"#,
        r#"{"a": {"b": {"c": [[], {}, [{}]]}}, "d": ""}"#,
        r#""a string value""#,
        "[1, 2, 3]",
        "[-0, 0.5, 1e5, 1E+5, 2e-3, -12.25E-1]",
        "-0.5e-3",
        "true",
        r#"{"q": "say \"hi\"\n\t\\ done", "u": "\u00e9\u00E9", "s": "a\/b\b\f\r"}"#,
        r#"{"pair": "\ud83d\ude00", "high": "\ud83d"}"#,
        r#"{"half": "\ud83dx"}"#,
        r#"{"half": "\ud83d\n"}"#,
        r#"{"half": "\ud83dA"}"#,
        r#""\udc00""#,
        "[1, truex]",
        "[1, falsey, 2]",
        "[1, nullx",
        "[nu",
        "truex",
        "trueé",
        "[1, 2}",
        r#"{"a" 1}"#,
        "{a: 1}",
        r#"{"a": 1 "b": 2}"#,
        "[1 2]",
        "[01]",
        "[1.5.5]",
        "[1x]",
        "[-x]",
        r#"{"a":1} "#,
        r#"{"a":1}x"#,
        r#"{"a": [1, {"b": [2, {"c": 3}]}]}"#,
        r#"{"name": "f", "arguments": {"k": "v"}}"#,
    ];

    /// What the partial parser tolerates and a strict reader refuses, with the byte the judge
    /// refuses it at: the first byte the strict reader refuses, or the first letter of a word or
    /// the backslash of an escape, which the readers hold. One of each kind, in the shapes the
    /// fuzzer found at a call's close.
    const TOLERATED: &[(&str, usize)] = &[
        // A value left out before a comma or a close.
        (r#"{"a": }"#, 6),
        (r#"{"a": , "b": 1}"#, 6),
        (r#"{"city": "Paris", "days": ,}"#, 26),
        ("[,,]", 1),
        ("[1,,2]", 3),
        // A trailing comma before a close.
        (r#"{"a": 1, }"#, 9),
        (r#"{"a": 1,}"#, 8),
        ("[1, 2, ]", 7),
        ("[1,]", 3),
        // An escape JSON lacks, and a `\u` escape with fewer than four hex digits.
        (r#"{"odd": "\q\/"}"#, 9),
        (r#"{"a": "it\'s"}"#, 9),
        (r#"{"short": "\u00"}"#, 11),
        (r#"{"short": "\uZZZZ"}"#, 11),
        (r#"{"a": "x\u12G4"}"#, 8),
        // A surrogate escape without its other half, a low surrogate alone and a high one whose
        // next escape is no low surrogate: refused at that escape's backslash, which is held.
        (r#"{"pair": "\ud83d\ude00", "lone": "\udc00"}"#, 34),
        (r#"{"a": "\ud83d\u0041"}"#, 13),
        // A bracket closed by the wrong kind.
        (r#"{"a": [1}"#, 8),
        (r#"{"a": [1, 2}"#, 11),
        (r#"[{"a": 1]"#, 8),
        // Whitespace JSON lacks, before the value and between tokens.
        ("\u{a0}[1,\u{a0}2]\u{2003}", 0),
        ("[1,\u{a0}2]", 3),
        ("{\"a\":\u{3000}1}", 5),
        // A number cut short before a delimiter.
        ("[-]", 2),
        ("[1.]", 3),
        ("[1e]", 3),
        ("[1e+]", 4),
        (r#"{"a": 1.}"#, 8),
        (r#"{"a": -}"#, 7),
        // A word cut short before a delimiter, from its first letter.
        ("[nu]", 1),
        (r#"{"a": nu}"#, 6),
        (r#"{"a": tr, "b": 1}"#, 6),
    ];

    /// The texts of both tables.
    fn every_text() -> Vec<String> {
        CORPUS
            .iter()
            .copied()
            .chain(TOLERATED.iter().map(|(text, _)| *text))
            .map(str::to_string)
            .collect()
    }

    /// Whether `text` is a prefix of JSON a strict reader accepts: whole, or cut where the reader
    /// wants more bytes rather than refusing one it saw.
    fn strict_prefix(text: &str) -> bool {
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(_) => true,
            Err(error) => error.is_eof(),
        }
    }

    /// The byte a strict reader refuses `text` at, if it refuses one (its column is one-based and
    /// counts bytes; the corpus has no newlines).
    fn strict_refusal(text: &str) -> Option<usize> {
        match serde_json::from_str::<serde_json::Value>(text) {
            Err(error) if !error.is_eof() => {
                assert_eq!(error.line(), 1, "{text:?}");
                Some(error.column() - 1)
            }
            _ => None,
        }
    }

    fn deep(levels: usize) -> String {
        let mut text = "[".repeat(levels);
        text.push('1');
        text.push_str(&"]".repeat(levels));
        text
    }

    #[test]
    fn it_stops_exactly_where_the_prefix_parser_stops_at_every_cut() {
        let mut corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        corpus.push(deep(32));
        corpus.push(deep(33));
        corpus.push(deep(40));
        let mut cuts = 0;
        for text in &corpus {
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let prefix = &text[..cut];
                assert_eq!(prefix_says(prefix), parser_says(prefix), "{prefix:?}");
                cuts += 1;
            }
        }
        assert!(cuts > 700, "{cuts} cuts");
    }

    /// What the partial parser tolerates and JSON lacks is refused at the first byte a strict
    /// reader refuses, or from the first letter of a word or the backslash of an escape, which
    /// the readers hold and which come before it: the bytes before that byte are a prefix the
    /// strict reader takes, the whole text is not, the parser read further, and every cut of the
    /// text gives the same byte.
    #[test]
    fn it_refuses_what_the_partial_parser_tolerates_where_a_strict_reader_does() {
        for &(text, at) in TOLERATED {
            assert_eq!(prefix_says(text), at, "{text:?}");
            assert!(
                strict_prefix(&text[..at]),
                "{text:?}: the bytes before {at}"
            );
            assert!(!strict_prefix(text), "{text:?}: whole");
            if let Some(refused) = strict_refusal(text) {
                assert!(
                    at <= refused,
                    "{text:?}: the judge at {at}, the reader at {refused}"
                );
            }
            assert!(parser_says(text) > at, "{text:?}: the parser tolerated it");
            for cut in (1..text.len()).filter(|&cut| text.is_char_boundary(cut)) {
                let mut judge = Prefix::default();
                let first = judge.feed(&text[..cut]);
                assert_eq!(
                    first.or_else(|| judge.feed(&text[cut..])),
                    Some(at),
                    "{text:?} cut at {cut}"
                );
            }
        }
    }

    /// `whole` says what a strict reader says of the bytes as they stand: a value that closed, or a
    /// number or a literal whole at the end, and nothing open or refused.
    #[test]
    fn whole_says_whether_the_bytes_are_one_value_a_strict_reader_takes() {
        for (text, expected) in [
            ("12", true),
            ("-0", true),
            ("1.5", true),
            ("1e5", true),
            ("1E+5", true),
            ("null", true),
            ("true", true),
            (r#"{"a": 1}"#, true),
            ("[]", true),
            ("{}", true),
            (r#""ab""#, true),
            ("-", false),
            ("1.", false),
            ("1e", false),
            ("1e+", false),
            ("nu", false),
            (r#"{"a": 1"#, false),
            ("[1,", false),
            (r#""ab"#, false),
            ("", false),
            ("  ", false),
            ("[1}", false),
            (r#"{"a": 1, }"#, false),
            (r#"{"a": nu}"#, false),
        ] {
            let mut judge = Prefix::default();
            judge.feed(text);
            assert_eq!(judge.whole(), expected, "{text:?}");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(text).is_ok(),
                expected,
                "{text:?} by the strict reader"
            );
        }
    }

    #[test]
    fn the_judge_says_where_it_may_still_go_back_to() {
        let after = |text: &str| {
            let mut prefix = Prefix::default();
            assert_eq!(prefix.feed(text), None, "{text:?}");
            prefix
        };
        // A word: its first letter inside a container, the value's start at the top; and the
        // value's start while nothing but whitespace has come at the top.
        assert_eq!(after(r#"{"a": nu"#).held_from(), Some(6));
        assert_eq!(after("  nu").held_from(), Some(0));
        assert_eq!(after("  ").held_from(), Some(0));
        assert_eq!(after("").held_from(), Some(0));
        assert_eq!(after("  {").held_from(), None);
        assert_eq!(after("  1").held_from(), None);
        assert_eq!(after("null").held_from(), Some(0));
        assert_eq!(after("[null, ").held_from(), None);
        // An escape being read: its backslash, until a letter JSON knows or four hex digits that
        // are a character have decided it; a high surrogate's four digits are not held, and the
        // `\u` escape after them is, until it is the low half.
        assert_eq!(after(r#"{"e": "\"#).held_from(), Some(7));
        assert_eq!(after(r#"{"e": "\u"#).held_from(), Some(7));
        assert_eq!(after(r#"{"e": "\u12"#).held_from(), Some(7));
        assert_eq!(after(r#"{"e": "\u1234"#).held_from(), None);
        assert_eq!(after(r#"{"e": "\n"#).held_from(), None);
        assert_eq!(after(r#"{"e": "\ud83c\"#).held_from(), Some(13));
        assert_eq!(after(r#"{"e": "\ud83c\u"#).held_from(), Some(13));
        assert_eq!(after(r#"{"e": "\ud83c\ude0"#).held_from(), Some(13));
        assert_eq!(after(r#"{"e": "\ud83c"#).held_from(), None);
        // A string streams on, at the top too; a number and a container: nothing to go back to.
        assert_eq!(after(r#""ab\ud83c"#).held_from(), None);
        assert_eq!(after(r#""ab"#).held_from(), None);
        assert_eq!(after("12.").held_from(), None);
        assert_eq!(after(r#"{"a": [1, "#).held_from(), None);
    }

    #[test]
    fn feeding_a_text_in_pieces_judges_it_as_feeding_it_whole_does() {
        let mut corpus = every_text();
        corpus.push(deep(33));
        for text in &corpus {
            let whole = Prefix::default().feed(text);
            // Character by character.
            let mut by_char = Prefix::default();
            let mut answer = None;
            for (i, c) in text.char_indices() {
                answer = by_char.feed(&text[i..i + c.len_utf8()]);
                if answer.is_some() {
                    break;
                }
            }
            assert_eq!(answer, whole, "{text:?} character by character");
            // Every two-way split.
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let mut split = Prefix::default();
                let first = split.feed(&text[..cut]);
                let answer = first.or_else(|| split.feed(&text[cut..]));
                assert_eq!(answer, whole, "{text:?} split at {cut}");
            }
        }
    }

    /// A raw control character inside a string is written as its escape, in a key or a value,
    /// after a backslash as the escape's own letter; between tokens a newline is whitespace and
    /// stays; a fragment with none is the bytes themselves, borrowed; and the pieces, each
    /// written from its own offset, join to what the whole gives.
    #[test]
    fn a_raw_control_character_inside_a_string_is_written_as_its_escape() {
        let text = "{\"a\nb\": \"x\ty\u{12}\u{7f}\", \"c\": [1,\n2], \"d\": \"\\\n\"}";
        let expected = "{\"a\\nb\": \"x\\ty\\u0012\u{7f}\", \"c\": [1,\n2], \"d\": \"\\n\"}";
        let mut whole = Prefix::default();
        assert_eq!(whole.feed(text), None);
        let escaped = whole.escaped(0, text);
        assert_eq!(escaped, expected);
        assert!(serde_json::from_str::<serde_json::Value>(&escaped).is_ok());
        assert!(matches!(
            Prefix::default().escaped(0, "{\"a\": 1}"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(whole.escaped(0, "{\"a"), Cow::Borrowed(_)));
        let mut by_char = Prefix::default();
        let mut joined = String::new();
        for (at, c) in text.char_indices() {
            let piece = &text[at..at + c.len_utf8()];
            assert_eq!(by_char.feed(piece), None);
            joined.push_str(&by_char.escaped(at, piece));
        }
        assert_eq!(joined, expected);
        let mut halves = Prefix::default();
        let cut = text.find('y').map_or(0, |at| at + 1);
        assert_eq!(halves.feed(&text[..cut]), None);
        assert_eq!(halves.feed(&text[cut..]), None);
        let mut joined = halves.escaped(0, &text[..cut]).into_owned();
        joined.push_str(&halves.escaped(cut, &text[cut..]));
        assert_eq!(joined, expected);
    }

    #[test]
    fn once_refused_it_stays_refused_where_it_was() {
        let mut prefix = Prefix::default();
        assert_eq!(prefix.feed(r#"{"a": 1 "#), None);
        assert_eq!(prefix.feed(r#""b""#), Some(8));
        assert_eq!(prefix.feed(": 2}"), Some(8));
    }

    /// The closure of a text: the prefix a strict reader accepts, cut back to its complete
    /// members and closed. A string, a number or a word cut at the end leaves with the key or the
    /// comma before it; a number or a word whole at the end stays, as does an empty container; the
    /// bytes from the first refused byte on are not read, so arguments that stop being JSON close
    /// where the readers stopped sending them; a whole value is itself, and nothing complete is
    /// `None`.
    #[test]
    fn closed_is_the_prefixs_complete_members_in_their_containers() {
        for (text, expected) in [
            // A call whose arguments stop being JSON: the whole arguments as the model wrote them
            // close where the judge refused (the number ends before the space it refused after),
            // as the prefix the readers sent closes.
            (
                r#"{"start_range": 0, "end_range": 2.0 * Math.PI, "frequency": 5}"#,
                Some(r#"{"start_range": 0, "end_range": 2.0}"#),
            ),
            (
                r#"{"start_range": 0, "end_range": 2.0 "#,
                Some(r#"{"start_range": 0, "end_range": 2.0 }"#),
            ),
            // The output cut inside a string value: the member leaves, the ones before it stay.
            (r#"{"drink_id": "lat"#, Some("{}")),
            (
                r#"{"city": "Paris", "drink_id": "lat"#,
                Some(r#"{"city": "Paris"}"#),
            ),
            (r#"{"q": "say \"hi\" and"#, Some("{}")),
            (
                r#"{"q": "say \"hi\"", "n": 1"#,
                Some(r#"{"q": "say \"hi\"", "n": 1}"#),
            ),
            (r#"{"a": "é", "b": "ü"#, Some(r#"{"a": "é"}"#)),
            // Cut at a key, a colon or a comma: the member under way leaves, no comma dangles.
            (r#"{"city": "Paris", "zip"#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"city": "Paris", "zip""#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"city": "Paris", "zip":"#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"city": "Paris", "zip": "#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"city": "Paris","#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"city": "Paris", "#, Some(r#"{"city": "Paris"}"#)),
            (r#"{"a": 1, "b\u00"#, Some(r#"{"a": 1}"#)),
            ("[1, 2,", Some("[1, 2]")),
            ("[1, 2, ", Some("[1, 2]")),
            // A number or a word: whole at the end it stays, cut short it leaves.
            (
                r#"{"city": "Paris", "zip": 7"#,
                Some(r#"{"city": "Paris", "zip": 7}"#),
            ),
            (
                r#"{"city": "Paris", "zip": 7."#,
                Some(r#"{"city": "Paris"}"#),
            ),
            (r#"{"zip": 1e5"#, Some(r#"{"zip": 1e5}"#)),
            (r#"{"zip": 1e"#, Some("{}")),
            (r#"{"zip": -"#, Some("{}")),
            (r#"{"ok": true"#, Some(r#"{"ok": true}"#)),
            (r#"{"ok": t"#, Some("{}")),
            (r#"{"ok": null"#, Some(r#"{"ok": null}"#)),
            (r#"{"ok": nul"#, Some("{}")),
            ("[1, 2", Some("[1, 2]")),
            // Open containers close at every level; an empty one stays.
            (
                r#"{"a": {"b": 1, "c": [1, 2"#,
                Some(r#"{"a": {"b": 1, "c": [1, 2]}}"#),
            ),
            (
                r#"{"a": {"b": 1, "c": [1, "x"#,
                Some(r#"{"a": {"b": 1, "c": [1]}}"#),
            ),
            (
                r#"{"a": {"b": 1, "c": {"#,
                Some(r#"{"a": {"b": 1, "c": {}}}"#),
            ),
            (r#"{"a": ["#, Some(r#"{"a": []}"#)),
            (
                r#"{"a": [1, [2, 3], [4"#,
                Some(r#"{"a": [1, [2, 3], [4]]}"#),
            ),
            ("[", Some("[]")),
            ("{", Some("{}")),
            (" {", Some(" {}")),
            // Refused bytes end the reading: a bracket of the wrong kind, a word that is no
            // literal, a surrogate without its other half, anything after the value.
            ("[1, 2}", Some("[1, 2]")),
            (r#"{"a": [1}"#, Some(r#"{"a": [1]}"#)),
            (r#"{"a": 1, "b": nulx}"#, Some(r#"{"a": 1}"#)),
            (r#"{"a": 1, "b": "x\ud800y"}"#, Some(r#"{"a": 1}"#)),
            (r#"{"a": "x\ud800y"}"#, Some("{}")),
            (r#"{"a": 1} "#, Some(r#"{"a": 1}"#)),
            (r#"{"a": 1}x"#, Some(r#"{"a": 1}"#)),
            // A whole value is itself.
            (r#"{"a": 1}"#, Some(r#"{"a": 1}"#)),
            (
                r#" {"a": [1, {"b": null}]}"#,
                Some(r#" {"a": [1, {"b": null}]}"#),
            ),
            ("12", Some("12")),
            (r#""abc""#, Some(r#""abc""#)),
            ("true", Some("true")),
            // Nothing complete.
            ("", None),
            ("  ", None),
            (r#""ab"#, None),
            ("t", None),
            ("1.", None),
            ("-", None),
            ("not json", None),
            ("x", None),
            (r#""x\ud800y""#, None),
        ] {
            let closed = Prefix::closed(text);
            assert_eq!(closed.as_deref(), expected, "{text:?}");
            if let Some(closed) = closed {
                assert!(
                    serde_json::from_str::<serde_json::Value>(&closed).is_ok(),
                    "{text:?}: {closed:?} is no JSON"
                );
            }
        }
    }

    /// At every cut of every text of the corpus the closure is one whole value a strict reader
    /// takes, made of the cut's bytes and closing brackets alone, and closed again it is itself;
    /// a text that is one whole value comes back as it stands (the whitespace after the value,
    /// which the judge refuses, left off).
    #[test]
    fn closed_at_every_cut_is_one_whole_value_made_of_the_cuts_bytes() {
        let mut cuts = 0;
        let mut closures = 0;
        for text in every_text() {
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let prefix = &text[..cut];
                cuts += 1;
                let Some(closed) = Prefix::closed(prefix) else {
                    continue;
                };
                closures += 1;
                assert!(
                    serde_json::from_str::<serde_json::Value>(&closed).is_ok(),
                    "{prefix:?}: {closed:?} is no JSON"
                );
                let head = closed.trim_end_matches(['}', ']']);
                assert!(
                    prefix.starts_with(head),
                    "{prefix:?}: {closed:?} is not the cut's bytes closed"
                );
                assert_eq!(Prefix::closed(&closed).as_deref(), Some(closed.as_str()));
                if cut == text.len() && serde_json::from_str::<serde_json::Value>(&text).is_ok() {
                    assert_eq!(closed, text.trim_end(), "a whole value is itself");
                }
            }
        }
        assert!(cuts > 700, "{cuts} cuts");
        assert!(closures > 500, "{closures} closures");
    }
}
