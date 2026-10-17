//! The fragments of an arguments value: which of its bytes are sent, which are held, and which
//! are malformed.
//!
//! A value's bytes arrive in pieces cut wherever the model's chunks fell. [`Fragments`] judges
//! them as they come ([`Prefix`]) and accounts for every byte once: a fragment for each run that
//! keeps the value a valid JSON prefix, malformed text from the first byte that breaks it, and,
//! while the value is still open, the bytes the judge may yet go back to held until the next
//! bytes settle them, the value closes, or the stream ends. The object readers of this module
//! ([`Assembler`](super::Assembler), [`named_arguments`](super::named_arguments)) keep their
//! accounting here, so that the two say the same of the same bytes.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::prefix::Prefix,
};

/// The argument bytes of one value, accounted for as they arrive.
#[derive(Clone, Debug, Default)]
pub(super) struct Fragments {
    /// Bytes accounted for so far, as fragments or as malformed text.
    emitted: usize,
    /// Bytes the prefix has judged so far.
    judged: usize,
    /// Whether the bytes are still a JSON prefix, judged as they arrive.
    prefix: Prefix,
    /// The byte from which the value stopped being a valid JSON prefix, if it did.
    invalid_from: Option<usize>,
}

impl Fragments {
    /// Whether the bytes stopped being a JSON prefix.
    pub(super) fn refused(&self) -> bool {
        self.invalid_from.is_some()
    }

    /// The bytes of `value` that arrived since the last call, judged, and the part that keeps the
    /// value a valid JSON prefix sent as a fragment of the call at `index`. Bytes the judge may
    /// still go back to (a word being spelled, `nu`; a surrogate escape awaiting its low half; the
    /// whitespace before a value at the top) are held while the value is `open`: a fragment
    /// passed on cannot be taken back, so the pieces would say `n` where the whole says nothing.
    /// They go once the next bytes settle them, with the value's end, or when no more bytes can
    /// come ([`Self::send_held`]), as the prefix they are then. A string where the value should
    /// be streams as written, the one shape where the pieces may say more than the whole
    /// (`Prefix::held_from`). Where the sent bytes end, the held ones excluded; `None` when every
    /// byte of `value` was accounted for already.
    pub(super) fn send(
        &mut self,
        value: &str,
        open: bool,
        index: u32,
        out: &mut Events,
    ) -> Option<usize> {
        if value.len() <= self.emitted {
            return None;
        }
        // Only the bytes that arrived are judged; the prefix keeps its place between pieces.
        let verdict = self.prefix.feed(&value[self.judged..]);
        self.judged = value.len();
        let valid_end = match self.invalid_from {
            Some(from) => from,
            None => match verdict {
                None => value.len(),
                Some(from) => *self.invalid_from.insert(from.max(self.emitted)),
            },
        };
        let valid_end = match self.prefix.held_from() {
            Some(start) if open && self.invalid_from.is_none() => {
                valid_end.min(start).max(self.emitted)
            }
            _ => valid_end,
        };
        if self.emitted < valid_end {
            let fresh = &value[self.emitted..valid_end];
            out.push(Event::ToolCallArguments {
                index,
                json: fresh.to_string(),
                source: Text::uncounted(fresh.to_owned()),
            });
            self.emitted = valid_end;
        }
        Some(valid_end)
    }

    /// Past the break, the bytes of `value` not yet accounted for are malformed; before any break
    /// nothing is said, since what is not sent is held.
    pub(super) fn send_malformed(&mut self, value: &str, out: &mut Events) {
        if self.invalid_from.is_some() && self.emitted < value.len() {
            out.push(Event::Malformed {
                text: Text::uncounted(value[self.emitted..].to_owned()),
                why: MalformedReason::InvalidArguments,
            });
            self.emitted = value.len();
        }
    }

    /// The bytes held back ([`Self::send`]), sent as the prefix they are when the value can take
    /// no more bytes: at its end, or the stream's.
    pub(super) fn send_held(&mut self, value: &str, index: u32, out: &mut Events) {
        if self.invalid_from.is_none() && self.emitted < value.len() {
            let held = &value[self.emitted..];
            out.push(Event::ToolCallArguments {
                index,
                json: held.to_string(),
                source: Text::uncounted(held.to_owned()),
            });
            self.emitted = value.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .map(|event| match event {
                Event::ToolCallArguments { json, .. } => format!("fragment {json}"),
                Event::Malformed { text, why } => format!("malformed {} ({why:?})", text.text),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn a_word_being_spelled_is_held_while_the_value_is_open_and_sent_when_it_closes() {
        let mut fragments = Fragments::default();
        let mut out = Events::new();
        let value = r#"{"a": nu"#;
        assert_eq!(fragments.send(value, true, 0, &mut out), Some(6));
        assert_eq!(said(&out.drain()), ["fragment {\"a\": "]);
        // Nothing new: nothing said, the word still held.
        assert_eq!(fragments.send(value, true, 0, &mut out), Some(6));
        assert!(out.drain().is_empty());
        let value = r#"{"a": null}"#;
        assert_eq!(fragments.send(value, false, 0, &mut out), Some(value.len()));
        assert_eq!(said(&out.drain()), ["fragment null}"]);
        assert!(!fragments.refused());
    }

    #[test]
    fn the_bytes_from_the_break_on_are_malformed_once_and_the_valid_prefix_is_a_fragment() {
        let mut fragments = Fragments::default();
        let mut out = Events::new();
        let value = r#"{"a": 1} tail"#;
        assert_eq!(fragments.send(value, true, 0, &mut out), Some(8));
        assert!(fragments.refused());
        fragments.send_malformed(value, &mut out);
        assert_eq!(
            said(&out.drain()),
            ["fragment {\"a\": 1}", "malformed  tail (InvalidArguments)"]
        );
        // More bytes after the break: malformed too, and nothing is sent twice.
        let value = r#"{"a": 1} tail more"#;
        assert_eq!(fragments.send(value, true, 0, &mut out), Some(8));
        fragments.send_malformed(value, &mut out);
        assert_eq!(said(&out.drain()), ["malformed  more (InvalidArguments)"]);
        fragments.send_held(value, 0, &mut out);
        assert!(out.drain().is_empty(), "a broken value holds nothing");
    }

    #[test]
    fn what_is_held_goes_as_the_prefix_it_is_when_no_more_bytes_can_come() {
        let mut fragments = Fragments::default();
        let mut out = Events::new();
        let value = r#"{"a": tr"#;
        assert_eq!(fragments.send(value, true, 0, &mut out), Some(6));
        out.drain();
        fragments.send_held(value, 0, &mut out);
        assert_eq!(said(&out.drain()), ["fragment tr"]);
        fragments.send_held(value, 0, &mut out);
        assert!(out.drain().is_empty(), "sent once");
    }
}
