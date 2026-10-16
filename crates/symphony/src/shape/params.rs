//! A shaped call's arguments written as tagged parameters, read as a template's `xml-inline`
//! content reads them.
//!
//! Hugging Face's parser reads such a call's text with `finditer` over the parameter's pattern
//! (`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>`): from the start, the earliest
//! place where the whole pattern matches is a parameter, the next search goes on from where it
//! ended, and whatever lies between parameters is skipped. [`Params`] does the same as the bytes
//! arrive. A parameter can only begin where its tag's first literal, the trigger, does, so the
//! reader looks for the trigger, runs a [`Matcher`] for the tag and its value's ends from there,
//! and on a match writes the parameter as one JSON member, `{"key": value` and then `, "key":
//! value`, the bytes of the whole parameter its source. On no match the search goes on after the
//! trigger's first character, as `finditer` tries the next place. The text the search passes over
//! is dropped as wrapping. A value is written when its parameter ends, its text read by the
//! template's parser and typed by the table's reader with the parameter's schema; a key written
//! twice is written again, which a JSON reader takes as the last value at the first one's place,
//! as a Python dict does, unless the reader keeps the first.
//!
//! A value has no cap: it may be as long as the call. The call's text ends where its trailer
//! begins, which is known only when a closing terminal comes, so the reader keeps the trailer's
//! search over the bytes after the last parameter; when the region closes, what is still
//! undecided before the trailer is decided as `finditer` decides it at the end of the text, and the
//! arguments' object is closed. The stream's stop, where the table says it closes the call, closes
//! it wherever the trailer stands: the template's reader closes a region at the end of its text
//! whether or not the closing pattern matched, and `finditer` passes over a trailer the stream
//! never finished, or a closing terminal given back to the call, as over any text no parameter
//! takes. When the stream cuts the region, it closes the same way at the end of what arrived, as
//! the template's reader closes a region at the end of its text: what is undecided is decided,
//! the bytes no parameter took (a parameter the cut left unfinished) are dropped as wrapping, and
//! the arguments' object is closed over the parameters that finished.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    shape::{
        Atom, CallShape, Duplicates, End, Match, Matcher, Pattern, Schema, Slot, Suffix,
        ValueParser, Values, Verdict,
    },
    tagged::{assembler::push_quoted, value::Declared},
};

pub(super) const KEY_WRITTEN_TWICE: &str = "a parameter whose key the call already wrote";

/// The parameters of one shaped call, from the bytes after its header.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    index: u32,
    shape: &'static CallShape,
    tag: &'static Pattern,
    ends: &'static [End],
    /// The literal every parameter begins with.
    trigger: &'static str,
    /// How a value's text reads, and the reader that types it.
    parser: &'static ValueParser,
    values: Values,
    /// The call's name, whose tool's schema types its values.
    name: String,
    /// The keys written, for a reader that keeps the first of two.
    keys: Vec<String>,
    search: Search,
    /// The character before the bytes the search holds, which a `\b` at a tag's start reads.
    before: Option<char>,
    /// How many members have been written.
    written: u32,
    /// The trailer's search, over the bytes since the last parameter's end, the floor.
    tail: Option<Suffix<'static>>,
    /// The character before the floor.
    floor_before: Option<char>,
    /// Bytes since the floor: how many arrived, and how many of them were dropped before the
    /// bytes the search holds.
    since_floor: usize,
    dropped: usize,
    /// Where the trailer starts, counted from the floor, once a closing terminal was found to end
    /// it.
    trailer_at: Option<usize>,
}

/// Where the search for the next parameter stands.
#[derive(Clone, Debug)]
enum Search {
    /// No parameter under way: the bytes held may still begin the trigger.
    Scanning(String),
    /// A parameter from its trigger on, the matcher holding every byte since.
    Matching(Box<Matcher<'static>>),
}

impl Params {
    /// The parameters of the call at `index`, named `name`, each written as `tag`, ended by one of
    /// `ends` and its value's text read by `parser` (the shape's
    /// [`Arguments::Params`](super::Arguments::Params)) and typed by `values`, after the header,
    /// whose last character is `before`. A tag that begins with no literal, which a shape that
    /// checks does not have, is looked for at every place.
    pub(crate) fn new(
        index: u32,
        shape: &'static CallShape,
        (tag, ends, parser): (&'static Pattern, &'static [End], &'static ValueParser),
        (name, values): (String, Values),
        before: Option<char>,
    ) -> Self {
        let trigger = match tag.0.first() {
            Some(Atom::Lit(trigger)) => &**trigger,
            _ => "",
        };
        Self {
            index,
            shape,
            tag,
            ends,
            trigger,
            parser,
            values,
            name,
            keys: Vec::new(),
            search: Search::Scanning(String::new()),
            before,
            written: 0,
            tail: None,
            floor_before: before,
            since_floor: 0,
            dropped: 0,
            trailer_at: None,
        }
    }

    /// Append the next bytes of the call and push the parameters they complete.
    pub(crate) fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if !self.shape.trailer.0.is_empty() {
            let (shape, before) = (self.shape, self.floor_before);
            self.tail
                .get_or_insert_with(|| trailer_search(shape, before))
                .feed(bytes);
        }
        self.since_floor += bytes.len();
        match &mut self.search {
            Search::Scanning(tail) => tail.push_str(bytes),
            Search::Matching(matcher) => {
                let verdict = matcher.feed(bytes);
                if !self.decided(verdict, declared, out) {
                    return;
                }
            }
        }
        self.scan(declared, out);
    }

    /// Whether the region closes at `terminal` (empty for the stream's stop): whether the trailer
    /// ends right before it, starting no earlier than the last parameter's end. The stream's stop
    /// closes it either way, as the end of the text closes the template's region: with no trailer
    /// there, every byte after the last parameter is the call's text.
    pub(crate) fn closes(&mut self, terminal: &str) -> bool {
        let at = if self.shape.trailer.0.is_empty() {
            Some(self.since_floor)
        } else {
            let next = terminal.chars().next();
            match &mut self.tail {
                Some(tail) => tail.start(next),
                None => trailer_search(self.shape, self.floor_before).start(next),
            }
        };
        self.trailer_at = at;
        at.is_some() || terminal.is_empty()
    }

    /// The region closed where [`closes`](Self::closes) said: what is undecided before the
    /// trailer is decided as at the end of the text, the arguments' object is closed, and the
    /// trailer's bytes still held are returned, for the call's end.
    pub(crate) fn close(mut self, declared: &Declared, out: &mut Events) -> String {
        let at = self.trailer_at.unwrap_or(self.since_floor);
        let held = self.take_held();
        // The trailer starts at `at` counted from the floor; the held bytes start at `dropped`.
        let split = at.saturating_sub(self.dropped).min(held.len());
        let (text, trailer) = held.split_at(split);
        self.settle(text, declared, out);
        self.close_object(out);
        trailer.to_string()
    }

    /// The stream cut the region: it closes as the template's reader closes a region at the end
    /// of its text. What is undecided is decided as at the end of the text, the bytes no
    /// parameter took are dropped as wrapping, and the arguments' object is closed over the
    /// parameters that finished, `{}` where none did.
    pub(crate) fn finish(mut self, declared: &Declared, out: &mut Events) {
        let held = self.take_held();
        self.settle(&held, declared, out);
        self.close_object(out);
    }

    /// Closes the arguments' object: `{}` where no parameter was written.
    fn close_object(&self, out: &mut Events) {
        let close = if self.written == 0 { "{}" } else { "}" };
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: close.to_string(),
            source: Text::default(),
        });
    }

    /// The bytes the search holds, taken.
    fn take_held(&mut self) -> String {
        match std::mem::replace(&mut self.search, Search::Scanning(String::new())) {
            Search::Scanning(tail) => tail,
            Search::Matching(matcher) => matcher.into_held(),
        }
    }

    /// Looks for the next parameter in what the search holds, and reads each one that the bytes
    /// decide.
    fn scan(&mut self, declared: &Declared, out: &mut Events) {
        loop {
            let Search::Scanning(tail) = &mut self.search else {
                return;
            };
            let Some(at) = tail.find(self.trigger) else {
                // A tail that may still become the trigger is held; the rest is passed over.
                let keep = longest_start_of(tail, self.trigger);
                let passed = tail[..tail.len() - keep].to_string();
                tail.drain(..tail.len() - keep);
                self.skip(&passed, out);
                return;
            };
            let passed = tail[..at].to_string();
            let from = tail.split_off(at);
            tail.clear();
            self.skip(&passed, out);
            let mut matcher =
                Matcher::new(self.tag, self.ends, &self.shape.word, self.before).hold_cap(None);
            let verdict = matcher.feed(&from);
            self.search = Search::Matching(Box::new(matcher));
            if !self.decided(verdict, declared, out) {
                return;
            }
        }
    }

    /// Takes a parameter's verdict: a match is written and the search goes on after it, no match
    /// passes over the trigger's first character. Returns whether the search goes on.
    fn decided(&mut self, verdict: Verdict, declared: &Declared, out: &mut Events) -> bool {
        let found = match verdict {
            Verdict::Pending => return false,
            Verdict::Matched(found) => Some(found),
            Verdict::NoMatch => None,
        };
        let Search::Matching(matcher) =
            std::mem::replace(&mut self.search, Search::Scanning(String::new()))
        else {
            return false;
        };
        let mut held = matcher.into_held();
        match found {
            Some(found) => {
                let rest = held.split_off(found.end());
                self.write(&held, &found, declared, out);
                self.set_floor(&held, &rest);
                self.search = Search::Scanning(rest);
            }
            None => {
                let first = held.chars().next().map_or(0, char::len_utf8);
                let rest = held.split_off(first);
                self.skip(&held, out);
                self.search = Search::Scanning(rest);
            }
        }
        true
    }

    /// Decides `text`, the call's last undecided bytes, as `finditer` does at the end of the text:
    /// each parameter that matches is written, and what lies between is dropped.
    fn settle(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        let mut rest = text;
        while let Some(at) = rest.find(self.trigger) {
            self.skip(&rest[..at], out);
            rest = &rest[at..];
            let mut matcher =
                Matcher::new(self.tag, self.ends, &self.shape.word, self.before).hold_cap(None);
            matcher.feed(rest);
            match matcher.finish() {
                Verdict::Matched(found) => {
                    let (parameter, after) = rest.split_at(found.end());
                    self.write(parameter, &found, declared, out);
                    self.before = parameter.chars().next_back();
                    rest = after;
                }
                Verdict::NoMatch | Verdict::Pending => {
                    let first = rest.chars().next().map_or(0, char::len_utf8);
                    self.skip(&rest[..first], out);
                    rest = &rest[first..];
                }
            }
        }
        self.skip(rest, out);
    }

    /// Writes the parameter `found` matched in `parameter`, its bytes: its key and its value as
    /// the next member of the arguments' object, the value read by the parser and typed by the
    /// reader with the parameter's schema. A key written before is written again, or, for a reader
    /// that keeps the first, the parameter is returned as `Malformed`.
    fn write(&mut self, parameter: &str, found: &Match, declared: &Declared, out: &mut Events) {
        let span = |slot| found.span(slot).map_or("", |span| &parameter[span]);
        let (key, value) = (span(Slot::Key), span(Slot::Value));
        let again = self.keys.iter().any(|written| written == key);
        if again && self.values.reader().duplicates() == Duplicates::First {
            out.push(Event::Malformed {
                text: Text::uncounted(parameter),
                why: MalformedReason::Other(KEY_WRITTEN_TWICE.to_string()),
            });
            return;
        }
        if !again {
            self.keys.push(key.to_string());
        }
        let schema = declared
            .parameter(&self.name, key)
            .map(|(root, property)| Schema { root, property });
        let mut member = String::from(if self.written == 0 { "{" } else { ", " });
        push_quoted(&mut member, key);
        member.push_str(": ");
        member.push_str(&self.values.reader().read(self.parser, value, schema));
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: member,
            source: Text::uncounted(parameter),
        });
        self.written += 1;
    }

    /// The text the search passed over, before the bytes it still holds, dropped as wrapping.
    fn skip(&mut self, text: &str, out: &mut Events) {
        if text.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        });
        self.dropped += text.len();
        self.before = text.chars().next_back().or(self.before);
    }

    /// A parameter ended: the trailer's search starts again after it, over the bytes that already
    /// followed it.
    fn set_floor(&mut self, parameter: &str, rest: &str) {
        self.before = parameter.chars().next_back().or(self.before);
        self.floor_before = self.before;
        self.since_floor = rest.len();
        self.dropped = 0;
        self.trailer_at = None;
        if !self.shape.trailer.0.is_empty() {
            let mut tail = trailer_search(self.shape, self.floor_before);
            tail.feed(rest);
            self.tail = Some(tail);
        }
    }
}

/// The search for `shape`'s trailer over text whose first byte follows `before`.
fn trailer_search(shape: &'static CallShape, before: Option<char>) -> Suffix<'static> {
    Suffix::new(&shape.trailer, &shape.word, before).hold_cap(Some(shape.cap()))
}

/// How many bytes at the end of `text` begin `literal` without being all of it: the longest proper
/// start of the literal that the text ends with.
fn longest_start_of(text: &str, literal: &str) -> usize {
    (1..literal.len())
        .rev()
        .find(|&length| {
            text.len() >= length
                && literal.is_char_boundary(length)
                && text.is_char_boundary(text.len() - length)
                && text.ends_with(&literal[..length])
        })
        .unwrap_or(0)
}
