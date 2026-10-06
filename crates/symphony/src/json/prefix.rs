//! A strict acceptor of JSON prefixes: which bytes some JSON value could still continue, and where
//! a whole value ends.
//!
//! The assembler and the constrained format promise that the argument fragments they emit always
//! form a valid JSON prefix. [`scan`] is how they keep it: a byte-by-byte run of the JSON grammar
//! over the value's text so far, which says how many leading bytes some value could still continue
//! (`valid`) and, once the text holds a whole value, where it ends (`complete`). It is a function
//! of the text alone, so where fragments stop and malformed text begins does not depend on how the
//! output was cut into deltas; the ported prefix parser, which decided this before, tolerated a
//! literal's prefix differently whole and in pieces, and the boundary moved with the cuts.
//!
//! The grammar is RFC 8259's, byte for byte: objects with string keys, arrays, strings with their
//! escapes and four hex digits after `\u`, numbers without leading zeros, `true`, `false` and
//! `null`, and the four whitespace bytes between tokens. Two deliberate readings: a number at the
//! very end of the text is a valid prefix but never whole, since more digits may come, so a bare
//! number is whole only once a byte that cannot continue it has arrived; and a raw control
//! character inside a string is taken as the string's content, as the model wrote it, rather than
//! refused. Nesting beyond [`MAX_DEPTH`] is refused, as the ported parser refused it. One
//! malformation the ported parser tolerated is refused here as the grammar refuses it: a trailing
//! comma before a closing bracket, `{"a": 1,}`, which the old gateway healed into `{"a":1}`, stops
//! the fragments before the bracket and makes the bracket malformed. Whether to heal it instead is
//! a policy question for the formats, not for the acceptor.

/// Where a text stands as the prefix of one JSON value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scan {
    /// How many leading bytes some JSON value could continue: all of them, up to the first byte
    /// none could, or up to the end of a whole value when the text holds one; the bytes after a
    /// whole value are not part of it.
    pub valid: usize,
    /// One past the last byte of the value, once the text holds a whole one.
    pub complete: Option<usize>,
}

/// The depth beyond which nesting is refused.
pub const MAX_DEPTH: usize = 32;

/// Read `text` as the prefix of one JSON value.
pub fn scan(text: &str) -> Scan {
    let mut machine = Machine::default();
    for (at, byte) in text.bytes().enumerate() {
        match machine.step(at, byte) {
            Step::Take => {}
            Step::Reject => {
                return Scan {
                    valid: at,
                    complete: None,
                }
            }
            Step::Done(end) => {
                return Scan {
                    valid: end,
                    complete: Some(end),
                }
            }
        }
    }
    Scan {
        valid: text.len(),
        complete: None,
    }
}

/// What one byte did to the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// The byte continues some value.
    Take,
    /// No value continues with this byte.
    Reject,
    /// The value ended before this byte, or with it, at the given offset.
    Done(usize),
}

/// What the machine is reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// A value is expected; inside an array, `close` says whether `]` may come instead.
    Value { close: bool },
    /// An object key is expected; `close` says whether `}` may come instead.
    Key { close: bool },
    /// The colon after a key.
    Colon,
    /// A value ended inside a container; a comma or the container's close is expected.
    After,
    /// Inside a string; `key` says whether it is an object key.
    Str { key: bool, escaped: bool, hex: u8 },
    /// Inside a number.
    Number(Number),
    /// Inside `true`, `false` or `null`; the bytes still expected.
    Literal(&'static [u8]),
}

/// Where a number stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Number {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

impl Number {
    /// The state after `byte`, if the number continues with it.
    fn next(self, byte: u8) -> Option<Self> {
        match (self, byte) {
            (Self::Minus, b'0') => Some(Self::Zero),
            (Self::Minus, b'1'..=b'9') => Some(Self::Int),
            (Self::Int, b'0'..=b'9') => Some(Self::Int),
            (Self::Zero | Self::Int, b'.') => Some(Self::Dot),
            (Self::Zero | Self::Int | Self::Frac, b'e' | b'E') => Some(Self::Exp),
            (Self::Dot | Self::Frac, b'0'..=b'9') => Some(Self::Frac),
            (Self::Exp, b'+' | b'-') => Some(Self::ExpSign),
            (Self::Exp | Self::ExpSign | Self::ExpDigits, b'0'..=b'9') => Some(Self::ExpDigits),
            _ => None,
        }
    }

    /// Whether the number may end here.
    fn whole(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

/// An open container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Object,
    Array,
}

#[derive(Debug)]
struct Machine {
    state: State,
    stack: Vec<Frame>,
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            state: State::Value { close: false },
            stack: Vec::new(),
        }
    }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

impl Machine {
    /// Feed the byte at `at`.
    fn step(&mut self, at: usize, byte: u8) -> Step {
        loop {
            match self.state {
                State::Value { close } => {
                    return match byte {
                        _ if is_space(byte) => Step::Take,
                        b']' if close => self.close(Frame::Array, at),
                        b'{' => self.open(Frame::Object),
                        b'[' => self.open(Frame::Array),
                        b'"' => self.enter(State::Str {
                            key: false,
                            escaped: false,
                            hex: 0,
                        }),
                        b'-' => self.enter(State::Number(Number::Minus)),
                        b'0' => self.enter(State::Number(Number::Zero)),
                        b'1'..=b'9' => self.enter(State::Number(Number::Int)),
                        b't' => self.enter(State::Literal(b"rue")),
                        b'f' => self.enter(State::Literal(b"alse")), // codespell:ignore alse
                        b'n' => self.enter(State::Literal(b"ull")),
                        _ => Step::Reject,
                    };
                }
                State::Key { close } => {
                    return match byte {
                        _ if is_space(byte) => Step::Take,
                        b'}' if close => self.close(Frame::Object, at),
                        b'"' => self.enter(State::Str {
                            key: true,
                            escaped: false,
                            hex: 0,
                        }),
                        _ => Step::Reject,
                    };
                }
                State::Colon => {
                    return match byte {
                        _ if is_space(byte) => Step::Take,
                        b':' => self.enter(State::Value { close: false }),
                        _ => Step::Reject,
                    };
                }
                State::After => {
                    return match (self.stack.last(), byte) {
                        (_, _) if is_space(byte) => Step::Take,
                        (Some(Frame::Object), b',') => self.enter(State::Key { close: false }),
                        (Some(Frame::Array), b',') => self.enter(State::Value { close: false }),
                        (Some(Frame::Object), b'}') => self.close(Frame::Object, at),
                        (Some(Frame::Array), b']') => self.close(Frame::Array, at),
                        _ => Step::Reject,
                    };
                }
                State::Str { key, escaped, hex } => {
                    if hex > 0 {
                        if !byte.is_ascii_hexdigit() {
                            return Step::Reject;
                        }
                        self.state = State::Str {
                            key,
                            escaped: false,
                            hex: hex - 1,
                        };
                        return Step::Take;
                    }
                    if escaped {
                        return match byte {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                                self.enter(State::Str {
                                    key,
                                    escaped: false,
                                    hex: 0,
                                })
                            }
                            b'u' => self.enter(State::Str {
                                key,
                                escaped: false,
                                hex: 4,
                            }),
                            _ => Step::Reject,
                        };
                    }
                    return match byte {
                        b'\\' => self.enter(State::Str {
                            key,
                            escaped: true,
                            hex: 0,
                        }),
                        b'"' if key => self.enter(State::Colon),
                        b'"' => self.value_done(at + 1),
                        _ => Step::Take,
                    };
                }
                State::Literal(rest) => {
                    let Some((&expected, rest)) = rest.split_first() else {
                        return Step::Reject;
                    };
                    if byte != expected {
                        return Step::Reject;
                    }
                    return if rest.is_empty() {
                        self.value_done(at + 1)
                    } else {
                        self.enter(State::Literal(rest))
                    };
                }
                State::Number(number) => {
                    if let Some(next) = number.next(byte) {
                        return self.enter(State::Number(next));
                    }
                    if !number.whole() {
                        return Step::Reject;
                    }
                    // The number ended before this byte, which belongs to what follows it.
                    match self.value_done(at) {
                        Step::Take => continue,
                        done => return done,
                    }
                }
            }
        }
    }

    fn enter(&mut self, state: State) -> Step {
        self.state = state;
        Step::Take
    }

    fn open(&mut self, frame: Frame) -> Step {
        if self.stack.len() >= MAX_DEPTH {
            return Step::Reject;
        }
        self.stack.push(frame);
        self.state = match frame {
            Frame::Object => State::Key { close: true },
            Frame::Array => State::Value { close: true },
        };
        Step::Take
    }

    fn close(&mut self, frame: Frame, at: usize) -> Step {
        debug_assert_eq!(self.stack.last(), Some(&frame));
        self.stack.pop();
        self.value_done(at + 1)
    }

    /// A value ended at `end`: the whole text's value when nothing is open, else the container
    /// reads on.
    fn value_done(&mut self, end: usize) -> Step {
        if self.stack.is_empty() {
            return Step::Done(end);
        }
        self.state = State::After;
        Step::Take
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHOLE: &[&str] = &[
        "{}",
        "[]",
        "\"\"",
        "true",
        "false",
        "null",
        "{\"city\": \"Paris\", \"days\": [1, 2.5, -3e+2, 0.5E-1], \"ok\": true, \"none\": null}",
        "[[[{\"a\": [\"計画 🌍\", \"a \\\"q\\\" \\\\ b\\nc\", \"\\ud83c\\udf0d\"]}]]]",
        " \n\t\r{ \"a\" : [ ] , \"b\" : { } } ",
        "\"a\nb\"",
    ];

    #[test]
    fn every_prefix_of_a_whole_value_is_valid_and_only_the_whole_is_complete() {
        for text in WHOLE {
            let whole = scan(text);
            let end = text.trim_end().len();
            assert_eq!(
                whole,
                Scan {
                    valid: end,
                    complete: Some(end)
                },
                "{text:?}"
            );
            for cut in text.char_indices().map(|(i, _)| i) {
                let prefix = &text[..cut];
                assert_eq!(
                    scan(prefix),
                    Scan {
                        valid: cut.min(end),
                        complete: (cut >= end).then_some(end),
                    },
                    "{text:?} cut at {cut}"
                );
            }
        }
    }

    #[test]
    fn the_first_byte_no_value_could_continue_ends_the_valid_prefix() {
        let cases: &[(&str, usize)] = &[
            ("nope", 1),
            ("not json", 1),
            ("hello", 0),
            ("[1,]", 3),
            ("{\"a\": 1,}", 8),
            ("{,}", 1),
            ("{\"a\" 1}", 5),
            ("{\"a\":}", 5),
            ("{\"a\": 1, nope}", 9),
            ("{1: 2}", 1),
            ("[1 2]", 3),
            ("-", 1),
            ("-x", 1),
            ("1.e", 2),
            ("1e", 2),
            ("1e+", 3),
            ("\"\\x\"", 2),
            ("\"\\u12g4\"", 5),
            ("@", 0),
            ("é", 0),
        ];
        for &(text, valid) in cases {
            assert_eq!(
                scan(text),
                Scan {
                    valid,
                    complete: None
                },
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_whole_value_followed_by_more_ends_where_the_value_does() {
        let cases: &[(&str, usize)] = &[
            ("truex", 4),
            ("12abc", 2),
            ("01", 1),
            ("{} and more", 2),
            ("\"s\"\"t\"", 3),
            ("[1] [2]", 3),
            ("null,", 4),
            ("1.5e3 ", 5),
        ];
        for &(text, end) in cases {
            assert_eq!(
                scan(text),
                Scan {
                    valid: end,
                    complete: Some(end)
                },
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_number_at_the_end_of_the_text_is_a_valid_prefix_but_not_whole() {
        for text in ["0", "12", "-3.5", "1e10", "1.", "1e", "-"] {
            assert_eq!(
                scan(text),
                Scan {
                    valid: text.len(),
                    complete: None
                },
                "{text:?}"
            );
        }
    }

    #[test]
    fn incomplete_strings_literals_and_containers_are_valid_prefixes() {
        for text in [
            "\"abc",
            "\"a\\",
            "\"a\\u00",
            "tru",  // codespell:ignore tru
            "fals", // codespell:ignore fals
            "nul",
            "{\"a\": [1, {\"b\": \"c",
        ] {
            assert_eq!(
                scan(text),
                Scan {
                    valid: text.len(),
                    complete: None
                },
                "{text:?}"
            );
        }
    }

    #[test]
    fn nesting_beyond_the_depth_limit_is_refused() {
        let deep = "[".repeat(MAX_DEPTH);
        assert_eq!(scan(&deep).valid, MAX_DEPTH);
        let deeper = "[".repeat(MAX_DEPTH + 1);
        assert_eq!(
            scan(&deeper),
            Scan {
                valid: MAX_DEPTH,
                complete: None
            }
        );
    }

    #[test]
    fn the_valid_prefix_of_every_cut_is_the_cut_or_the_whole_text_s_boundary() {
        // Stateless, so this holds by construction; the check is that `valid` never exceeds the
        // cut and agrees with the whole text's verdict below it.
        for text in [
            "{\"a\": 1, nope}",
            "truex",
            "[1, falsex, 2]",
            "{\"a\": [1, 2",
            "12abc",
        ] {
            let whole = scan(text);
            for cut in text.char_indices().map(|(i, _)| i) {
                let scanned = scan(&text[..cut]);
                assert_eq!(scanned.valid, cut.min(whole.valid), "{text:?} cut at {cut}");
            }
        }
    }
}
