//! A call's arguments written as one JSON value, the call named elsewhere.
//!
//! Some formats write a call's name outside its arguments: in a header before them
//! (`<|channel|>commentary to=functions.get_weather <|message|>{"city": "Paris"}`), in a template's
//! text, or nowhere at all, as when a grammar forced one function and the whole output is its
//! arguments. What is left is one JSON value, and [`Arguments`] turns its bytes into the call's
//! argument events as they arrive: the whitespace before the value is dropped as such, the value's
//! bytes are fragments for as long as they keep it a JSON prefix (judged by [`Prefix`], so the
//! fragments keep the promise `ToolCallArguments` makes, exactly as strong as for the
//! [`Assembler`](super::Assembler)), and at the first byte that does not, one `Malformed` event of
//! no bytes with `InvalidArguments` says so, the rest of the value staying the call's own
//! fragments, as the model wrote them (as `json::fragments` says it for the object readers). The
//! fragments are the model's own bytes, spacing included,
//! but for a raw control character inside a string, which JSON forbids and the fragment writes as
//! its escape ([`Prefix::escaped`]); only `source` keeps the model's byte there.
//! A word being spelled (`nu`) is held until it is a literal or is refused, which the judge does
//! from its first letter, so that where the chunks were cut does not decide which of its bytes
//! were sent.
//!
//! A reader of the template's values may type a member of an object's top level whose value is a
//! string (a string `"5"` the tool declares an integer is written `5`), which it can do only with
//! the whole string: the caller's [`Strings`] says which members' strings are held from their
//! opening quote until they close, and what is written for them then, the string's bytes the
//! fragment's source either way. [`AsWritten`] holds none.
//!
//! The value ends where it is whole: at its closing bracket or quote, or before the byte after a
//! number or a literal; a number or a literal that the text ends on is whole once the caller says
//! no more bytes come. `feed` returns how many of a piece's bytes the value took, all of them
//! before its end and up to it in the piece that carries it, so the caller routes what follows (a
//! trailer, a closing marker, text that belongs to no value) itself. The call's start and end are
//! the caller's too, since it knows the name and where the call closes: a forced function starts
//! its call at the value's first byte, a format at its header.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    json::{prefix::json_space, Prefix},
};

/// What the reader does with a member of the value's top-level object whose value is a string:
/// sends its bytes as they arrive, or holds them until the string closes and writes what
/// [`typed`](Self::typed) gives for it.
pub trait Strings {
    /// Whether the string value of the member `key` is held until it closes.
    fn holds(&self, key: &str) -> bool;

    /// The JSON written for the held string `raw` (its JSON text, quotes included) of the member
    /// `key`; `None` writes the model's bytes.
    fn typed(&self, key: &str, raw: &str) -> Option<String>;
}

/// Every string sent as the model wrote it.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsWritten;

impl Strings for AsWritten {
    fn holds(&self, _key: &str) -> bool {
        false
    }

    fn typed(&self, _key: &str, _raw: &str) -> Option<String> {
        None
    }
}

/// The argument events of one call, from the bytes of its arguments value.
#[derive(Clone, Debug)]
pub struct Arguments {
    index: u32,
    /// The value's bytes so far, from its first one.
    value: String,
    /// Whether the value's first byte has arrived.
    started: bool,
    /// Bytes of `value` sent as fragments.
    emitted: usize,
    /// Whether the bytes are still a JSON prefix, judged as they arrive.
    prefix: Prefix,
    /// Where the value stopped being a JSON prefix, if it did.
    invalid_from: Option<usize>,
    /// Where the value ended, once it is whole.
    whole: Option<usize>,
    /// Whether the value ends where its bytes stop being a JSON prefix: the bytes from there on
    /// are then the caller's ([`Self::refused_tail`]), for a shaped call's trailer to read, rather
    /// than the call's fragments after the break.
    ends_where_refused: bool,
    /// The last character of the whitespace before the value, if any came: what stands before a
    /// value refused at its first byte ([`Self::refused_tail`]).
    led_by: Option<char>,
    /// The top-level object's members, as far as holding a string needs them.
    members: Members,
}

/// The members of the value's top-level object, read as far as the bytes are a JSON prefix: which
/// key a string value belongs to, and where a held one starts and closes.
#[derive(Clone, Debug, Default)]
struct Members {
    /// Bytes read.
    at: usize,
    /// Containers open around `at`: one inside the top-level object.
    depth: usize,
    /// Whether the value is an object.
    object: bool,
    /// Inside the top-level object, whether a key comes next.
    key_next: bool,
    /// The string being read, if one is.
    string: Option<Quoted>,
    /// The last key read inside the top-level object, as it stands between its quotes.
    key: String,
    /// The held strings that have closed and not been written: their start, end and key.
    closed: Vec<(usize, usize, String)>,
    /// Where a held string that has not closed starts.
    open: Option<usize>,
}

/// A string being read: where its opening quote is, whether the last character was a backslash,
/// and what the string is.
#[derive(Clone, Copy, Debug)]
struct Quoted {
    start: usize,
    escaped: bool,
    role: Role,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A key of the top-level object.
    Key,
    /// A value of the top-level object that is held.
    Held,
    /// Any other string.
    Other,
}

impl Members {
    /// Read `value` on to `to`, asking `strings` whether each top-level string value is held. A
    /// place already read is not read again.
    fn read(&mut self, value: &str, to: usize, strings: &dyn Strings) {
        if to <= self.at {
            return;
        }
        for (offset, c) in value[self.at..to].char_indices() {
            let at = self.at + offset;
            if let Some(quoted) = &mut self.string {
                match c {
                    _ if quoted.escaped => quoted.escaped = false,
                    '\\' => quoted.escaped = true,
                    '"' => {
                        let quoted = *quoted;
                        self.string = None;
                        match quoted.role {
                            Role::Key => self.key = value[quoted.start + 1..at].to_string(),
                            Role::Held => {
                                self.closed.push((quoted.start, at + 1, decoded(&self.key)));
                                self.open = None;
                            }
                            Role::Other => {}
                        }
                    }
                    _ => {}
                }
                continue;
            }
            let top = self.object && self.depth == 1;
            match c {
                '{' | '[' => {
                    self.object |= self.depth == 0 && c == '{';
                    self.depth += 1;
                    self.key_next = self.object && self.depth == 1;
                }
                '}' | ']' => self.depth = self.depth.saturating_sub(1),
                ',' if top => self.key_next = true,
                ':' if top => self.key_next = false,
                '"' => {
                    let role = if !top {
                        Role::Other
                    } else if self.key_next {
                        Role::Key
                    } else if strings.holds(&decoded(&self.key)) {
                        Role::Held
                    } else {
                        Role::Other
                    };
                    if role == Role::Held {
                        self.open = Some(at);
                    }
                    self.string = Some(Quoted {
                        start: at,
                        escaped: false,
                        role,
                    });
                }
                _ => {}
            }
        }
        self.at = to;
    }
}

/// A key as it stands between its quotes, decoded where it is a JSON string's text.
fn decoded(key: &str) -> String {
    serde_json::from_str(&format!("\"{key}\"")).unwrap_or_else(|_| key.to_string())
}

impl Arguments {
    /// A reader for the arguments of the call at `index`.
    pub fn new(index: u32) -> Self {
        Self {
            index,
            value: String::new(),
            started: false,
            emitted: 0,
            prefix: Prefix::default(),
            invalid_from: None,
            whole: None,
            ends_where_refused: false,
            led_by: None,
            members: Members::default(),
        }
    }

    /// The value ends where its bytes stop being a JSON prefix (see [`Self::refused_tail`]).
    pub fn ending_where_refused(mut self) -> Self {
        self.ends_where_refused = true;
        self
    }

    /// Whether the bytes stopped being a JSON prefix.
    #[cfg(test)]
    pub fn refused(&self) -> bool {
        self.invalid_from.is_some()
    }

    /// Whether the value ended where its bytes stopped being a JSON prefix: it ends there
    /// ([`Self::ending_where_refused`]), and they did.
    pub fn ended_where_refused(&self) -> bool {
        self.ends_where_refused && self.invalid_from.is_some()
    }

    /// Whether the value takes no more bytes: it is whole, or it ended where refused.
    pub fn over(&self) -> bool {
        self.whole.is_some() || self.ended_where_refused()
    }

    /// For a value that ends where refused: the bytes from the refusal on, which the value took
    /// before the refusal was known (a held word, the rest of the piece), handed back once with
    /// the character before them (the last of the whitespace before the value, for a value refused
    /// at its first byte; `None` when nothing came before it), for the caller to read as what
    /// follows the value; `None` for a value not refused, or whose tail was already taken. `feed`
    /// takes nothing after this.
    pub fn refused_tail(&mut self) -> Option<(String, Option<char>)> {
        let from = self.invalid_from.filter(|_| self.ends_where_refused)?;
        if from >= self.value.len() && self.emitted >= from {
            return None;
        }
        let tail = self.value.split_off(from);
        let before = self.value.chars().next_back().or(self.led_by);
        self.emitted = self.emitted.min(from);
        Some((tail, before))
    }

    /// Whether the value is whole.
    pub fn whole(&self) -> bool {
        self.whole.is_some()
    }

    /// Append the next bytes, push the events they complete, holding the string values `strings`
    /// holds, and return how many of them the value took: all of them before its end, up to its
    /// end in the piece that carries it, and none once it is whole.
    pub fn feed(&mut self, bytes: &str, strings: &dyn Strings, out: &mut Events) -> usize {
        if self.over() {
            return 0;
        }
        let mut taken = 0;
        let mut rest = bytes;
        if !self.started {
            let value = rest.trim_start_matches(json_space);
            let space = &rest[..rest.len() - value.len()];
            if let Some(last) = space.chars().next_back() {
                self.led_by = Some(last);
                out.push(Event::Dropped {
                    text: Text::uncounted(space.to_owned()),
                    why: DropReason::Whitespace,
                });
            }
            taken = space.len();
            if value.is_empty() {
                return taken;
            }
            self.started = true;
            rest = value;
        }
        let before = self.value.len();
        self.value.push_str(rest);
        let sound_so_far = self.invalid_from.is_none();
        let refused = self.prefix.feed(rest);
        // The judge ends none but a value a strict reader takes whole, so the value is whole where
        // it ended.
        if self.invalid_from.is_none() {
            self.whole = self.prefix.ended();
        }
        let end = self.whole.unwrap_or(self.value.len());
        if self.invalid_from.is_none() {
            // A refusal at or after the value's end is the bytes that follow it, not the value's.
            if let Some(from) = refused.filter(|&from| from < end) {
                self.invalid_from = Some(from.max(self.emitted));
            }
            // The members are read as far as the bytes are a JSON prefix, in the piece that breaks
            // it too, so that a held string that closed before the break is written as `strings`
            // says whichever piece the break came in.
            self.members
                .read(&self.value, self.invalid_from.unwrap_or(end), strings);
        }
        let mut upto = end;
        if self.invalid_from.is_none() && self.whole.is_none() {
            // A word being spelled and an escape being read (from its backslash) are held; a
            // string streams on, as a string parameter would otherwise wait for its closing
            // quote; a surrogate that breaks inside it is said once at the break, whole and in
            // pieces alike (`Prefix::held_from`).
            let held = [self.prefix.held_from(), self.members.open];
            upto = held.into_iter().flatten().fold(end, usize::min);
        }
        // A value that ends where refused keeps the bytes from the refusal on for the caller
        // (`refused_tail`), instead of sending them on as the call's fragments after the break.
        let reported = match self.invalid_from {
            Some(from) if self.ends_where_refused => from.max(self.emitted).min(end),
            _ => end,
        };
        let breaks_here = sound_so_far && self.invalid_from.is_some();
        self.emit(
            upto.max(self.emitted).min(reported),
            reported,
            breaks_here,
            strings,
            out,
        );
        self.value.truncate(end);
        taken + end - before
    }

    /// No more bytes come: a number or a literal the bytes end on is whole now, and a word they
    /// end on is sent as the prefix it is. A value that is no whole one, whether it began (a
    /// string or a container left open, a word that is no literal's prefix) or no byte of it came
    /// (the end right after the header that opened the call, whitespace alone), is said to be
    /// none, by a `Malformed` event of no bytes after its fragments, unless it broke, which was
    /// said at the break: nothing sent is revised, nothing is made up for what the model never
    /// finished, and a call whose value never began is not handed over as a whole one with an
    /// empty string for its arguments. Returns whether the value is whole.
    pub fn finish(&mut self, out: &mut Events) -> bool {
        if self.whole.is_none()
            && self.started
            && self.invalid_from.is_none()
            && self.prefix.whole()
        {
            self.whole = Some(self.value.len());
        }
        let end = self.value.len();
        self.emit(end, end, false, &AsWritten, out);
        // A value that broke said so at the break, whether it sent the rest or handed it back
        // (`ends_where_refused`); one that never broke and is no whole value is said to be none
        // here.
        if self.whole.is_none() && self.invalid_from.is_none() {
            out.push(Event::Malformed {
                text: Text::default(),
                why: MalformedReason::UnterminatedRegion,
            });
        }
        self.whole.is_some()
    }

    /// The value's bytes up to `upto` not yet accounted for, the rest up to `end` held: a fragment
    /// for the part that keeps the value a JSON prefix, a held string that closed written as
    /// `strings` says, the break said once where the prefix broke (`breaks_here`, by a `Malformed`
    /// event of no bytes), and the part after it as the model wrote it. A held string the prefix
    /// broke in, or one the bytes end in, is sent as the model wrote it.
    fn emit(
        &mut self,
        upto: usize,
        end: usize,
        breaks_here: bool,
        strings: &dyn Strings,
        out: &mut Events,
    ) {
        let valid_end = self.invalid_from.unwrap_or(upto).min(upto);
        let ready = self
            .members
            .closed
            .iter()
            .take_while(|&&(_, close, _)| close <= valid_end)
            .count();
        let closed: Vec<(usize, usize, String)> = self.members.closed.drain(..ready).collect();
        for (start, close, key) in closed {
            self.fragment(start, out);
            let raw = &self.value[start..close];
            let typed = strings.typed(&key, raw);
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: typed.unwrap_or_else(|| self.prefix.escaped(start, raw).into_owned()),
                source: Text::uncounted(raw.to_owned()),
            });
            self.emitted = close;
        }
        self.fragment(valid_end, out);
        if breaks_here {
            out.push(Event::Malformed {
                text: Text::default(),
                why: MalformedReason::InvalidArguments,
            });
        }
        if self.invalid_from.is_some() && self.emitted < end {
            let rest = &self.value[self.emitted..end];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: rest.to_string(),
                source: Text::uncounted(rest.to_owned()),
            });
            self.emitted = end;
        }
    }

    /// The value's bytes from what was sent up to `upto`, as a fragment of the model's bytes, a
    /// raw control character inside a string written as its escape.
    fn fragment(&mut self, upto: usize, out: &mut Events) {
        if self.emitted < upto {
            let fresh = &self.value[self.emitted..upto];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: self.prefix.escaped(self.emitted, fresh).into_owned(),
                source: Text::uncounted(fresh.to_owned()),
            });
            self.emitted = upto;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The events for `pieces`, what the value took of each, and whether it is whole after the
    /// last piece and after `finish`.
    fn run(pieces: &[&str]) -> (Vec<Event<'static>>, Vec<usize>, bool, bool) {
        let mut arguments = Arguments::new(0);
        let mut out = Events::new();
        let taken = pieces
            .iter()
            .map(|piece| arguments.feed(piece, &AsWritten, &mut out))
            .collect();
        let fed = arguments.whole();
        let finished = arguments.finish(&mut out);
        (out.drain(), taken, fed, finished)
    }

    fn fragments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The bytes the events account for.
    fn sources(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::ToolCallArguments { source, .. } => &*source.text,
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => &*text.text,
                _ => "",
            })
            .collect()
    }

    #[test]
    fn an_object_streams_as_its_bytes_and_ends_at_its_brace() {
        let (events, taken, whole, _) = run(&[" \n{\"city\": ", "\"Paris\"}", "\n</x>"]);
        assert_eq!(
            events[0],
            Event::Dropped {
                text: Text::uncounted(" \n"),
                why: DropReason::Whitespace,
            }
        );
        assert_eq!(fragments(&events), "{\"city\": \"Paris\"}");
        assert_eq!(taken, [11, 8, 0], "nothing after the brace is the value's");
        assert!(whole);
        assert_eq!(sources(&events), " \n{\"city\": \"Paris\"}");
    }

    #[test]
    fn every_cut_takes_the_same_bytes() {
        let text = " {\"a\": [1, 2.5, \"x\\\"y\"], \"b\": null}  tail";
        let value_end = text.find("  tail").expect("the value's end");
        for cut in (1..text.len()).filter(|&cut| text.is_char_boundary(cut)) {
            let (events, taken, whole, _) = run(&[&text[..cut], &text[cut..]]);
            assert!(whole, "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), value_end, "cut at {cut}");
            assert_eq!(
                fragments(&events),
                text[..value_end].trim_start(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_number_is_whole_at_the_byte_after_it_or_at_the_end() {
        let (events, taken, whole, finished) = run(&["12", "5 }"]);
        assert_eq!(fragments(&events), "125");
        assert_eq!(taken, [2, 1]);
        assert!(whole && finished);
        let (_, _, whole, finished) = run(&["12"]);
        assert!(!whole, "more digits may come");
        assert!(finished, "the end of the bytes ends the number");
        let (_, _, _, finished) = run(&["1."]);
        assert!(!finished, "a number cut short is not whole");
    }

    /// The fragments sent before the first `Malformed` event, joined; `None` when none came.
    fn said_before(events: &[Event]) -> Option<String> {
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))?;
        Some(fragments(&events[..at]))
    }

    #[test]
    fn bytes_that_break_the_prefix_are_said_to_once_and_stay_the_values_own() {
        let broke = Event::Malformed {
            text: Text::default(),
            why: MalformedReason::InvalidArguments,
        };
        let (events, taken, whole, finished) = run(&["{\"a\": trux, \"b\": 1}"]);
        assert_eq!(fragments(&events), "{\"a\": trux, \"b\": 1}");
        assert_eq!(
            said_before(&events),
            Some("{\"a\": ".to_string()),
            "the word is refused from its first letter"
        );
        assert_eq!(
            events.iter().filter(|event| *event == &broke).count(),
            1,
            "the break is said once, with no bytes: {events:#?}"
        );
        assert_eq!(taken, [19], "a broken value takes every byte");
        assert!(!whole && !finished);
        let (events, _, _, _) = run(&["{\"a\": t", "rux, \"b\": 1}"]);
        assert_eq!(fragments(&events), "{\"a\": trux, \"b\": 1}");
        assert_eq!(
            said_before(&events),
            Some("{\"a\": ".to_string()),
            "a word is held until it is decided, so the cut changes nothing"
        );
        assert_eq!(sources(&events), "{\"a\": trux, \"b\": 1}");
        let (events, _, _, finished) = run(&["[nu", "ll, tr"]);
        assert_eq!(
            fragments(&events),
            "[null, tr",
            "a literal is sent once it ends, and a word the bytes end on as the prefix it is"
        );
        assert!(!finished);
        let (events, _, whole, _) = run(&["[1}", " more"]);
        assert_eq!(fragments(&events), "[1} more");
        assert_eq!(said_before(&events), Some("[1".to_string()));
        assert!(!whole, "an array a brace cut short is not whole");
        assert_eq!(sources(&events), "[1} more");
    }

    /// Holds the member `n`'s string and writes it as the integer it spells.
    struct HoldsN;

    impl Strings for HoldsN {
        fn holds(&self, key: &str) -> bool {
            key == "n"
        }

        fn typed(&self, _key: &str, raw: &str) -> Option<String> {
            let text: String = serde_json::from_str(raw).ok()?;
            text.parse::<i64>().ok().map(|number| number.to_string())
        }
    }

    /// The fragments for `pieces`, read with [`HoldsN`], and how many `Malformed` events came.
    fn held(pieces: &[&str]) -> (String, usize) {
        let mut arguments = Arguments::new(0);
        let mut out = Events::new();
        for piece in pieces {
            arguments.feed(piece, &HoldsN, &mut out);
        }
        arguments.finish(&mut out);
        let events = out.drain();
        let malformed = events
            .iter()
            .filter(|event| matches!(event, Event::Malformed { .. }))
            .count();
        (fragments(&events), malformed)
    }

    /// A raw control character inside a string is written as its escape, in a streamed string and
    /// in a held one the typing could not read (a strict reader refuses the raw character, so the
    /// model's bytes are written, escaped), and the value is whole where the model closed it.
    #[test]
    fn a_raw_control_character_inside_a_string_is_written_as_its_escape() {
        let text = "{\"s\": \"a\nb\", \"n\": \"5\t\"}";
        let expected = ("{\"s\": \"a\\nb\", \"n\": \"5\\t\"}".to_string(), 0);
        assert_eq!(held(&[text]), expected);
        for cut in (1..text.len()).filter(|&cut| text.is_char_boundary(cut)) {
            assert_eq!(
                held(&[&text[..cut], &text[cut..]]),
                expected,
                "cut at {cut}"
            );
        }
        let mut arguments = Arguments::new(0);
        let mut out = Events::new();
        assert_eq!(
            arguments.feed("{\"s\": \"a\nb\"} rest", &AsWritten, &mut out),
            "{\"s\": \"a\nb\"}".len(),
            "the value is whole at its brace"
        );
        assert!(arguments.whole());
        assert_eq!(fragments(&out.drain()), "{\"s\": \"a\\nb\"}");
    }

    /// What the partial parser tolerates and a strict reader refuses is refused from the first
    /// such byte: the fragments before it stand, the break is said there once with no bytes, the
    /// rest of the value stays the value's own as written, the value is never whole, and the cut
    /// changes nothing of it.
    #[test]
    fn what_a_strict_reader_refuses_is_said_to_at_the_first_such_byte_and_kept_whole() {
        for (text, expected_fragments) in [
            (
                "{\"city\": \"Paris\", \"days\": ,}",
                "{\"city\": \"Paris\", \"days\": ",
            ),
            (
                "{\"city\": \"Paris\", \"days\": 3, }",
                "{\"city\": \"Paris\", \"days\": 3, ",
            ),
            ("{\"tags\": [,,]}", "{\"tags\": ["),
            ("{\"city\": \"Pa\\yris\"}", "{\"city\": \"Pa"),
            ("{\"city\": \"\\u12G4\"}", "{\"city\": \""),
            ("{\"a\": [1}", "{\"a\": [1"),
            ("{\"n\": 1.}", "{\"n\": 1."),
            ("{\"a\":\u{a0}1}", "{\"a\":"),
            ("{\"flag\": nu}", "{\"flag\": "),
        ] {
            let whole = run(&[text]);
            let pieces: Vec<&str> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            let by_char = run(&pieces);
            for ((events, taken, fed, finished), how) in
                [(whole, "whole"), (by_char, "by the character")]
            {
                assert_eq!(fragments(&events), text, "{text:?} {how}");
                assert_eq!(
                    said_before(&events),
                    Some(expected_fragments.to_string()),
                    "{text:?} {how}"
                );
                assert_eq!(sources(&events), text, "{text:?} {how}");
                assert_eq!(
                    taken.iter().sum::<usize>(),
                    text.len(),
                    "{text:?} {how}: a broken value takes every byte"
                );
                assert!(!fed && !finished, "{text:?} {how}");
                let said: Vec<&Event> = events
                    .iter()
                    .filter(|event| matches!(event, Event::Malformed { .. }))
                    .collect();
                assert_eq!(
                    said,
                    [&Event::Malformed {
                        text: Text::default(),
                        why: MalformedReason::InvalidArguments,
                    }],
                    "{text:?} {how}: the break is said once, with no bytes"
                );
            }
        }
    }

    #[test]
    fn a_held_string_is_written_the_same_whichever_piece_breaks_the_value() {
        let text = "{\"s\": \"a\", \"n\": \"5\", x}";
        let whole = held(&[text]);
        assert_eq!(whole, ("{\"s\": \"a\", \"n\": 5, x}".to_string(), 1));
        for cut in (1..text.len()).filter(|&cut| text.is_char_boundary(cut)) {
            assert_eq!(held(&[&text[..cut], &text[cut..]]), whole, "cut at {cut}");
        }
    }

    /// A value that ends where refused takes the piece that breaks it, sends the prefix before
    /// the break and says the break, in a `Malformed` event of no bytes, and nothing after it:
    /// the bytes from the break on are handed back once with the character before them, and it
    /// takes nothing more; its end adds no word. A value refused at the end of a piece has no
    /// tail.
    #[test]
    fn a_value_ending_where_refused_hands_its_tail_back_once() {
        let mut arguments = Arguments::new(0).ending_where_refused();
        let mut out = Events::new();
        assert_eq!(arguments.feed("{\"a\": 1</f>\n", &AsWritten, &mut out), 12);
        assert!(arguments.refused() && arguments.over() && !arguments.whole());
        let events = out.drain();
        assert_eq!(fragments(&events), "{\"a\": 1");
        assert_eq!(
            events.last(),
            Some(&Event::Malformed {
                text: Text::default(),
                why: MalformedReason::InvalidArguments,
            }),
            "the break is said where it stands: {events:#?}"
        );
        assert_eq!(
            arguments.refused_tail(),
            Some(("</f>\n".to_string(), Some('1')))
        );
        assert_eq!(arguments.refused_tail(), None);
        assert_eq!(arguments.feed("more", &AsWritten, &mut out), 0);
        assert!(!arguments.finish(&mut out));
        assert!(
            out.drain().is_empty(),
            "the break was said with the bytes; the end adds nothing"
        );

        let mut arguments = Arguments::new(0).ending_where_refused();
        let mut out = Events::new();
        assert_eq!(arguments.feed("{\"a\": tr", &AsWritten, &mut out), 8);
        assert!(!arguments.refused(), "a word being spelled is held");
        assert_eq!(arguments.feed("uex</f>", &AsWritten, &mut out), 7);
        assert!(arguments.refused());
        assert_eq!(fragments(&out.drain()), "{\"a\": ");
        assert_eq!(
            arguments.refused_tail(),
            Some(("truex</f>".to_string(), Some(' '))),
            "the held word comes back with the tail"
        );

        // Refused at its first byte, the value has no character of its own before the tail: the
        // last of the whitespace before it stands there, whichever piece it came in, and the
        // caller knows what came before when none did.
        let mut arguments = Arguments::new(0).ending_where_refused();
        let mut out = Events::new();
        assert_eq!(arguments.feed(" \nEND", &AsWritten, &mut out), 5);
        assert!(arguments.ended_where_refused());
        assert_eq!(fragments(&out.drain()), "");
        assert_eq!(
            arguments.refused_tail(),
            Some(("END".to_string(), Some('\n')))
        );
        let mut arguments = Arguments::new(0).ending_where_refused();
        assert_eq!(arguments.feed(" ", &AsWritten, &mut out), 1);
        assert_eq!(arguments.feed("END", &AsWritten, &mut out), 3);
        assert_eq!(
            arguments.refused_tail(),
            Some(("END".to_string(), Some(' ')))
        );
        let mut arguments = Arguments::new(0).ending_where_refused();
        assert_eq!(arguments.feed("END", &AsWritten, &mut out), 3);
        assert_eq!(arguments.refused_tail(), Some(("END".to_string(), None)));

        let mut plain = Arguments::new(0);
        let mut out = Events::new();
        assert_eq!(plain.feed("{\"a\": 1</f>", &AsWritten, &mut out), 11);
        assert!(
            plain.refused() && !plain.over() && !plain.ended_where_refused(),
            "a plain value keeps the rest as its own"
        );
        assert_eq!(plain.refused_tail(), None);
    }

    /// Whitespace alone, or nothing at all, is no value: the end says so, by the one `Malformed`
    /// event of no bytes a value cut short gets, so that a call whose arguments never began is
    /// not handed over as whole with an empty string for them.
    #[test]
    fn whitespace_alone_is_no_value() {
        let none = Event::Malformed {
            text: Text::default(),
            why: MalformedReason::UnterminatedRegion,
        };
        let (events, taken, whole, finished) = run(&[" \t", "\r\n"]);
        assert_eq!(taken, [2, 2]);
        assert!(!whole && !finished);
        assert!(fragments(&events).is_empty());
        assert_eq!(sources(&events), " \t\r\n");
        assert_eq!(events.last(), Some(&none), "{events:#?}");
        let (events, taken, whole, finished) = run(&[]);
        assert!(taken.is_empty() && !whole && !finished);
        assert_eq!(
            events,
            [none],
            "no byte came, and the end still says the value is none"
        );
    }
}
