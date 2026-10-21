//! The header of a call written as its tool's name, a marker, then JSON: the bytes before the
//! marker, held until the marker arrives, and what follows it in the piece that completes it.
//! The named reader ([`named`](super::named)) reads its header with it; what the framing becomes
//! and which JSON reader takes the rest is its parameter ([`After`](super::named::After)).

use crate::event::{Event, Events, MalformedReason, Text};

/// The bytes before a call's marker, held until the marker arrives or the region ends.
#[derive(Clone, Debug)]
pub(super) struct Header {
    text: String,
    marker: &'static str,
}

/// A marker found: the header's text through the marker, where the marker starts in it, and the
/// bytes after it from the piece that completed it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Found {
    pub(super) framing: String,
    pub(super) at: usize,
    pub(super) after: String,
}

impl Header {
    /// An empty header before `marker`.
    pub(super) fn new(marker: &'static str) -> Self {
        Self {
            text: String::new(),
            marker,
        }
    }

    /// Append `bytes`, and find the marker if it ends inside them. The marker can only end inside
    /// the new bytes, so the search starts `marker.len() - 1` bytes before them (at a character
    /// boundary): a header that never meets the marker costs each piece its own length, not the
    /// header's.
    pub(super) fn push(&mut self, bytes: &str) -> Option<Found> {
        let before = self.text.len();
        self.text.push_str(bytes);
        let from = before.saturating_sub(self.marker.len() - 1);
        let from = (0..=from)
            .rev()
            .find(|&at| self.text.is_char_boundary(at))
            .unwrap_or(0);
        let at = self.text[from..].find(self.marker).map(|at| from + at)?;
        // The marker ends inside this piece, so what follows it is this piece's too.
        let after = self.text.split_off(at + self.marker.len());
        let framing = std::mem::take(&mut self.text);
        Some(Found { framing, at, after })
    }

    /// The region ended before the marker came: the held bytes come back as `Malformed` for `why`;
    /// nothing was held, nothing is pushed.
    pub(super) fn ended(self, why: MalformedReason, out: &mut Events) {
        if !self.text.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(self.text),
                why,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(framing: &str, at: usize, after: &str) -> Found {
        Found {
            framing: framing.to_string(),
            at,
            after: after.to_string(),
        }
    }

    #[test]
    fn the_marker_is_found_where_it_ends_and_the_rest_of_that_piece_follows_it() {
        let mut header = Header::new("[ARGS]");
        assert_eq!(header.push("get_weather"), None);
        assert_eq!(
            header.push("[ARGS]{\"city\""),
            Some(found("get_weather[ARGS]", 11, "{\"city\""))
        );
        let mut header = Header::new("[ARGS]");
        assert_eq!(header.push("f[ARGS]"), Some(found("f[ARGS]", 1, "")));
    }

    #[test]
    fn a_marker_split_across_pieces_is_found_when_its_last_byte_arrives() {
        let mut header = Header::new("<|content_invoke_tool_json|>");
        assert_eq!(header.push("get_weather<|content_"), None);
        assert_eq!(header.push("invoke_tool_"), None);
        assert_eq!(
            header.push("json|>\n{"),
            Some(found("get_weather<|content_invoke_tool_json|>", 11, "\n{"))
        );
    }

    #[test]
    fn the_search_starts_on_a_character_boundary_inside_multibyte_text() {
        // Five bytes before the new piece is inside the second three-byte character: the search
        // moves back to where that character starts.
        let mut header = Header::new("[ARGS]");
        assert_eq!(header.push("\u{65e5}\u{672c}\u{8a9e}["), None);
        assert_eq!(
            header.push("ARGS]"),
            Some(found("\u{65e5}\u{672c}\u{8a9e}[ARGS]", 9, ""))
        );
    }

    #[test]
    fn a_header_fed_a_byte_at_a_time_finds_what_the_whole_finds() {
        let text = "na\u{e9}me[ARGS]{}";
        let whole = Header::new("[ARGS]").push(text).expect("found whole");
        let mut pieces = Header::new("[ARGS]");
        let mut by_char = None;
        for (at, c) in text.char_indices() {
            if let Some(found) = pieces.push(&text[at..at + c.len_utf8()]) {
                by_char = Some(found);
                break;
            }
        }
        let by_char = by_char.expect("found in pieces");
        assert_eq!((by_char.framing, by_char.at), (whole.framing, whole.at));
        assert_eq!((by_char.after.as_str(), whole.after.as_str()), ("", "{}"));
    }

    #[test]
    fn an_ended_header_reports_what_it_held_and_an_empty_one_nothing() {
        let mut header = Header::new("[ARGS]");
        assert_eq!(header.push("name"), None);
        let mut out = Events::new();
        header.ended(MalformedReason::UnterminatedRegion, &mut out);
        assert_eq!(
            out.as_slice(),
            &[Event::Malformed {
                text: Text::uncounted("name"),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
        let mut out = Events::new();
        Header::new("[ARGS]").ended(MalformedReason::UnterminatedRegion, &mut out);
        assert!(out.is_empty());
    }
}
