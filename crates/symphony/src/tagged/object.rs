//! The arguments object a tagged assembler writes, one `ToolCallArguments` fragment at a time.
//!
//! A tagged syntax writes no JSON: the assembler does, and every fragment carries as its source
//! the output bytes it accounts for. [`Object`] keeps that bookkeeping in one place, the bytes
//! carried for the next event's source and the members written so far, and writes what every
//! object needs: `{` before the first member and `, ` before every other, `}` or `{}` at the
//! close, the carried bytes dropped as the template's wrapping before a report or reported
//! themselves. For the assemblers that type a value by the request's tools it also writes the
//! member: a declared string streamed from its first fragment and closed with its quote, a string
//! that may be null held until its text says, any other value written whole at its close.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::NULL_WORDS,
    },
};

/// How a value is written, decided when it opens.
#[derive(Clone, Copy, Debug)]
pub(super) enum Mode {
    /// A declared string, pushed piece by piece once its opening fragment is out.
    Streaming { opened: bool },
    /// Written at the close with [`json`](super::value::json).
    Whole,
    /// A string that may be null: held until its text rules the null words out.
    Undecided,
}

/// The arguments object of the call at `index`, as it is written.
#[derive(Clone, Debug)]
pub(super) struct Object {
    pub(super) index: u32,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    /// An assembler that holds a name or a value by its position in here (a whole value, up to
    /// its close) writes or reports it before any other event takes the bytes: no position
    /// outlives the bytes it points into.
    pub(super) carried: String,
    /// Members written so far.
    written: u32,
}

impl Object {
    /// An empty object for the call at `index`, nothing carried.
    pub(super) fn new(index: u32) -> Self {
        Self {
            index,
            carried: String::new(),
            written: 0,
        }
    }

    /// `{` before the first member, `, ` before every other.
    fn separator(&self) -> &'static str {
        if self.written == 0 {
            "{"
        } else {
            ", "
        }
    }

    /// A fragment of the arguments, with `source` as the bytes it accounts for.
    pub(super) fn push_fragment<'d>(&self, json: String, source: Text<'d>, out: &mut Events<'d>) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }

    /// `}` after the members, or `{}` when there were none, with `source` as its bytes.
    pub(super) fn close<'d>(&mut self, source: Text<'d>, out: &mut Events<'d>) {
        let json = if self.written == 0 { "{}" } else { "}" }.to_string();
        self.push_fragment(json, source, out);
    }

    /// The carried bytes refused ([`refuse`](super::refuse)): a `Malformed` event of no bytes
    /// with `why`, then the bytes as `Dropped { MalformedCall }`; nothing when none.
    pub(super) fn refuse(&mut self, why: MalformedReason, out: &mut Events) {
        super::refuse(std::mem::take(&mut self.carried), why, out);
    }

    /// Whitespace carried before text that is reported: the template's, after a tag the syntax
    /// read, dropped so the report holds only the text (after text released already an assembler
    /// keeps it with the report instead); nothing when none.
    pub(super) fn drop_carried(&mut self, out: &mut Events) {
        crate::event::drop_carried(std::mem::take(&mut self.carried), out);
    }

    /// The next bytes of the value under `key`, whose text starts at `start` in the carried
    /// bytes, by its `mode`: a whole value is carried to its close; a string that may be null is
    /// carried until its text rules the null words out, then opened and streamed; a declared
    /// string is streamed as it arrives.
    pub(super) fn value_bytes(
        &mut self,
        key: &str,
        start: usize,
        mode: &mut Mode,
        text: &str,
        out: &mut Events,
    ) {
        if text.is_empty() {
            return;
        }
        match *mode {
            Mode::Whole => self.carried.push_str(text),
            Mode::Undecided => {
                self.carried.push_str(text);
                let so_far = &self.carried[start..];
                if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
                    // Not null after all: open the string and stream what arrived.
                    let arrived = self.carried.split_off(start);
                    *mode = Mode::Streaming { opened: false };
                    self.stream(key, mode, &arrived, out);
                }
            }
            Mode::Streaming { .. } => self.stream(key, mode, text, out),
        }
    }

    /// A piece of the streamed string under `key`: the member opens on the first, `{"key": "` or
    /// `, "key": "` with the carried bytes as its source; every piece is pushed escaped, as its
    /// own source.
    fn stream(&mut self, key: &str, mode: &mut Mode, text: &str, out: &mut Events) {
        let Mode::Streaming { opened } = mode else {
            return;
        };
        if !*opened {
            *opened = true;
            let mut opening = String::with_capacity(key.len() + 8);
            opening.push_str(self.separator());
            push_quoted(&mut opening, key);
            opening.push_str(": \"");
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.push_fragment(opening, source, out);
            self.written += 1;
        }
        if text.is_empty() {
            return;
        }
        let mut json = String::with_capacity(text.len());
        push_escaped(&mut json, text);
        self.push_fragment(json, Text::uncounted(text.to_owned()), out);
    }

    /// A member written whole: `"key": json` after the separator, counted as written.
    pub(super) fn member(&mut self, key: &str, json: &str) -> String {
        let mut member = String::with_capacity(key.len() + json.len() + 8);
        member.push_str(self.separator());
        push_quoted(&mut member, key);
        member.push_str(": ");
        member.push_str(json);
        self.written += 1;
        member
    }

    /// The value closed: `fragment`, with the carried bytes and the closing `tag` as its source.
    pub(super) fn close_member(&mut self, fragment: String, tag: &str, out: &mut Events) {
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(fragment, Text::uncounted(source), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragments(events: &Events) -> Vec<(String, String)> {
        events
            .as_slice()
            .iter()
            .map(|event| match event {
                Event::ToolCallArguments { json, source, .. } => {
                    (json.clone(), source.text.to_string())
                }
                Event::Dropped { text, .. } => ("dropped".to_string(), text.text.to_string()),
                Event::Malformed { text, .. } => ("malformed".to_string(), text.text.to_string()),
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    #[test]
    fn members_open_the_object_then_follow_a_comma_and_the_close_matches_the_count() {
        let mut object = Object::new(3);
        let mut out = Events::new();
        let first = object.member("a", "1");
        object.close_member(first, "</v>", &mut out);
        let mut mode = Mode::Streaming { opened: false };
        object.carried.push_str("<v>");
        object.value_bytes("b", 0, &mut mode, "x\"y", &mut out);
        object.close_member("\"".to_string(), "</v>", &mut out);
        object.close(Text::uncounted("</f>"), &mut out);
        assert_eq!(
            fragments(&out),
            vec![
                ("{\"a\": 1".to_string(), "</v>".to_string()),
                (", \"b\": \"".to_string(), "<v>".to_string()),
                ("x\\\"y".to_string(), "x\"y".to_string()),
                ("\"".to_string(), "</v>".to_string()),
                ("}".to_string(), "</f>".to_string()),
            ]
        );
        let mut out = Events::new();
        Object::new(0).close(Text::uncounted("</f>"), &mut out);
        assert_eq!(
            fragments(&out),
            vec![("{}".to_string(), "</f>".to_string())]
        );
    }

    #[test]
    fn a_string_that_may_be_null_is_held_while_a_null_word_fits_and_streams_once_it_does_not() {
        let mut object = Object::new(0);
        let mut out = Events::new();
        let mut mode = Mode::Undecided;
        let start = object.carried.len();
        object.value_bytes("k", start, &mut mode, "nu", &mut out);
        assert!(out.is_empty(), "held: {:?}", out.as_slice());
        assert!(matches!(mode, Mode::Undecided));
        object.value_bytes("k", start, &mut mode, "mber", &mut out);
        assert!(matches!(mode, Mode::Streaming { opened: true }));
        assert_eq!(
            fragments(&out),
            vec![
                ("{\"k\": \"".to_string(), String::new()),
                ("number".to_string(), "number".to_string()),
            ]
        );
        let mut whole = Object::new(0);
        let mut mode = Mode::Whole;
        whole.value_bytes("k", 0, &mut mode, "12", &mut out);
        assert_eq!(whole.carried, "12");
    }

    #[test]
    fn carried_bytes_are_refused_or_dropped_once_and_nothing_is_pushed_for_none() {
        let mut object = Object::new(0);
        let mut out = Events::new();
        object.refuse(MalformedReason::TextBetweenTags, &mut out);
        object.drop_carried(&mut out);
        assert!(out.is_empty());
        object.carried.push_str("  ");
        object.drop_carried(&mut out);
        object.carried.push('x');
        object.refuse(MalformedReason::TextBetweenTags, &mut out);
        // Refused: the counted event of no bytes, then the bytes as the model's failed call.
        assert_eq!(
            fragments(&out),
            vec![
                ("dropped".to_string(), "  ".to_string()),
                ("malformed".to_string(), String::new()),
                ("dropped".to_string(), "x".to_string()),
            ]
        );
        assert!(object.carried.is_empty());
    }
}
