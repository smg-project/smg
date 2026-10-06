//! Marker strings, and the hold-back that keeps a half-arrived marker out of the content.
//!
//! A format recognises a model's output by marker strings: `<think>`, `</think>`, `<tool_call>`,
//! `</tool_call>`. A marker can arrive split across chunks, so text that ends in the beginning of a
//! marker cannot be released yet: it may be content, or the first bytes of a marker. [`Scanner`]
//! holds exactly that much back and releases everything else as soon as it is certain, so content
//! reaches the client with the least delay the markers allow and no marker is ever shown as text.
//!
//! The scanner knows nothing about what the markers mean; the format does. At a position where
//! several markers match, the longest wins, and a marker that is a prefix of another (`ab` and
//! `abc`) is told apart only once enough bytes are there, so `ab` at the end of the text is held.

/// What a piece of the model's text turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    /// Text that is not part of any marker.
    Text(String),
    /// A marker, by its position in the list the scanner was given.
    Marker(usize),
}

/// Splits a growing text into content and markers, holding back only what may still be a marker.
#[derive(Clone, Debug)]
pub struct Scanner {
    markers: Vec<String>,
    held: String,
}

impl Scanner {
    /// A scanner for `markers`, each reported by its position in the list as given. With none,
    /// every byte is text. An empty string never matches, so the scan always moves forward, and of
    /// two equal markers the first in the list is the one reported.
    pub fn new(markers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            markers: markers.into_iter().map(Into::into).collect(),
            held: String::new(),
        }
    }

    /// The marker a [`Piece::Marker`] refers to.
    pub fn marker(&self, index: usize) -> &str {
        &self.markers[index]
    }

    /// The bytes held back because they may still become a marker.
    pub fn held(&self) -> &str {
        &self.held
    }

    /// Append the next bytes and return what is now certain, in order: text that cannot be part of
    /// a marker, and the markers found. Whatever may still become a marker stays held.
    pub fn feed(&mut self, bytes: &str) -> Vec<Piece> {
        let text = std::mem::take(&mut self.held) + bytes;
        self.scan(text, true)
    }

    /// No more bytes will come: what was held is a marker if it is one whole, and text otherwise.
    pub fn finish(mut self) -> Vec<Piece> {
        let text = std::mem::take(&mut self.held);
        self.scan(text, false)
    }

    /// Split `text` into pieces. With `more_may_come`, a tail that could still grow into a marker
    /// is held instead of released.
    fn scan(&mut self, text: String, more_may_come: bool) -> Vec<Piece> {
        let mut pieces = Vec::new();
        let mut released = 0;
        let mut at = 0;
        while at < text.len() {
            let rest = &text[at..];
            if more_may_come && self.could_grow_into_marker(rest) {
                Self::release(&mut pieces, &text[released..at]);
                self.held = rest.to_string();
                return pieces;
            }
            if let Some(index) = self.longest_marker_at(rest) {
                Self::release(&mut pieces, &text[released..at]);
                pieces.push(Piece::Marker(index));
                at += self.markers[index].len();
                released = at;
            } else {
                at += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        Self::release(&mut pieces, &text[released..]);
        pieces
    }

    /// Whether `rest`, all of it, is a proper prefix of some marker: more bytes could complete it.
    fn could_grow_into_marker(&self, rest: &str) -> bool {
        self.markers
            .iter()
            .any(|marker| marker.len() > rest.len() && marker.starts_with(rest))
    }

    /// The longest marker that `rest` starts with, by index; of equal ones, the first in the list.
    fn longest_marker_at(&self, rest: &str) -> Option<usize> {
        let mut found: Option<(usize, usize)> = None;
        for (index, marker) in self.markers.iter().enumerate() {
            let longer = found.is_none_or(|(_, length)| marker.len() > length);
            if !marker.is_empty() && longer && rest.starts_with(marker.as_str()) {
                found = Some((index, marker.len()));
            }
        }
        found.map(|(index, _)| index)
    }

    fn release(pieces: &mut Vec<Piece>, text: &str) {
        if !text.is_empty() {
            pieces.push(Piece::Text(text.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];
    const OUTPUT: &str = "<think>plan</think>Sure.<tool_call>{\"name\": \"f\"}</tool_call>";

    fn scan(markers: &[&str], pieces_in: &[&str]) -> Vec<Piece> {
        let mut scanner = Scanner::new(markers.iter().copied());
        let mut out = Vec::new();
        for piece in pieces_in {
            out.extend(scanner.feed(piece));
        }
        out.extend(scanner.finish());
        out
    }

    /// Adjacent text pieces joined, so chunkings compare by what they say, not how often.
    fn joined(pieces: Vec<Piece>) -> Vec<Piece> {
        let mut out: Vec<Piece> = Vec::new();
        for piece in pieces {
            match (out.last_mut(), piece) {
                (Some(Piece::Text(prev)), Piece::Text(text)) => prev.push_str(&text),
                (_, piece) => out.push(piece),
            }
        }
        out
    }

    fn rendered(markers: &[&str], pieces: &[Piece]) -> String {
        pieces
            .iter()
            .map(|p| match p {
                Piece::Text(text) => text.as_str(),
                Piece::Marker(index) => markers[*index],
            })
            .collect()
    }

    #[test]
    fn a_whole_output_splits_into_text_and_markers_in_order() {
        assert_eq!(
            scan(&QWEN, &[OUTPUT]),
            vec![
                Piece::Marker(0),
                Piece::Text("plan".into()),
                Piece::Marker(1),
                Piece::Text("Sure.".into()),
                Piece::Marker(2),
                Piece::Text("{\"name\": \"f\"}".into()),
                Piece::Marker(3),
            ]
        );
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = joined(scan(&QWEN, &[OUTPUT]));
        let mut chunkings: Vec<Vec<&str>> = (1..OUTPUT.len())
            .map(|cut| vec![&OUTPUT[..cut], &OUTPUT[cut..]])
            .collect();
        chunkings.push(
            OUTPUT
                .char_indices()
                .map(|(i, c)| &OUTPUT[i..i + c.len_utf8()])
                .collect(),
        );
        for pieces in chunkings {
            let got = scan(&QWEN, &pieces);
            assert_eq!(rendered(&QWEN, &got), OUTPUT, "{pieces:?}");
            assert_eq!(joined(got), whole, "{pieces:?}");
        }
    }

    #[test]
    fn text_that_may_begin_a_marker_is_held_until_it_is_decided() {
        let mut scanner = Scanner::new(QWEN);
        assert_eq!(
            scanner.feed("Hello <tool_c"),
            vec![Piece::Text("Hello ".into())]
        );
        assert_eq!(scanner.held(), "<tool_c");
        assert_eq!(scanner.feed("all>"), vec![Piece::Marker(2)]);
        assert_eq!(scanner.held(), "");
        assert_eq!(
            scanner.feed("<tool_x"),
            vec![Piece::Text("<tool_x".into())],
            "no longer a prefix, so text"
        );
        assert_eq!(scanner.feed("a<"), vec![Piece::Text("a".into())]);
        assert_eq!(
            scanner.held(),
            "<",
            "a lone bracket may still begin any marker"
        );
    }

    #[test]
    fn finish_releases_what_was_held_as_text_or_as_the_marker_it_turned_out_to_be() {
        let mut scanner = Scanner::new(QWEN);
        assert_eq!(scanner.feed("abc</thi"), vec![Piece::Text("abc".into())]);
        assert_eq!(scanner.finish(), vec![Piece::Text("</thi".into())]);
        assert_eq!(Scanner::new(QWEN).finish(), vec![]);
        let mut scanner = Scanner::new(["ab", "abc"]);
        assert_eq!(scanner.feed("ab"), vec![], "held: `abc` may still come");
        assert_eq!(
            scanner.finish(),
            vec![Piece::Marker(0)],
            "nothing came, so it was `ab`"
        );
    }

    #[test]
    fn a_marker_that_is_a_prefix_of_another_is_told_apart_once_enough_bytes_are_there() {
        let markers = ["ab", "abc"];
        let mut scanner = Scanner::new(markers);
        assert_eq!(
            scanner.feed("xab"),
            vec![Piece::Text("x".into())],
            "`ab` may still become `abc`"
        );
        assert_eq!(
            scanner.feed("d"),
            vec![Piece::Marker(0), Piece::Text("d".into())]
        );
        assert_eq!(scan(&markers, &["abc"]), vec![Piece::Marker(1)]);
        assert_eq!(
            scan(&markers, &["ab"]),
            vec![Piece::Marker(0)],
            "held, then released at the end as the marker it is"
        );
    }

    #[test]
    fn adjacent_markers_and_no_markers_at_all() {
        assert_eq!(
            scan(&QWEN, &["</think><tool_call>"]),
            vec![Piece::Marker(1), Piece::Marker(2)]
        );
        assert_eq!(
            scan(&[], &["<think>any</think>"]),
            vec![Piece::Text("<think>any</think>".into())],
            "with no markers everything is text at once"
        );
        assert_eq!(scan(&QWEN, &[""]), vec![]);
    }

    #[test]
    fn multibyte_markers_hold_back_on_character_boundaries() {
        let markers = ["<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"];
        let text = "ok <｜tool▁calls▁begin｜>body<｜tool▁calls▁end｜>";
        let whole = joined(scan(&markers, &[text]));
        assert_eq!(
            whole,
            vec![
                Piece::Text("ok ".into()),
                Piece::Marker(0),
                Piece::Text("body".into()),
                Piece::Marker(1),
            ]
        );
        for (cut, _) in text.char_indices().skip(1) {
            let got = scan(&markers, &[&text[..cut], &text[cut..]]);
            assert_eq!(rendered(&markers, &got), text, "cut at {cut}");
            assert_eq!(joined(got), whole, "cut at {cut}");
        }
        let mut scanner = Scanner::new(markers);
        assert_eq!(scanner.feed("ok <｜tool"), vec![Piece::Text("ok ".into())]);
        assert_eq!(scanner.held(), "<｜tool");
    }

    #[test]
    fn an_empty_marker_never_matches_and_keeps_its_place_in_the_list() {
        let mut scanner = Scanner::new(["", "<think>", ""]);
        assert_eq!(
            scanner.feed("a<think>b"),
            vec![
                Piece::Text("a".into()),
                Piece::Marker(1),
                Piece::Text("b".into()),
            ]
        );
        assert_eq!(
            scanner.marker(1),
            "<think>",
            "indices are the list's as given"
        );
        assert_eq!(
            Scanner::new([""]).feed("plain"),
            vec![Piece::Text("plain".into())]
        );
        assert_eq!(
            Scanner::new(["<think>", ""]).finish(),
            vec![],
            "finishing with nothing held ends at once"
        );
    }

    #[test]
    fn of_two_equal_markers_the_first_in_the_list_is_reported() {
        assert_eq!(
            Scanner::new(["<x>", "<x>", "<x>y"]).feed("<x>z"),
            vec![Piece::Marker(0), Piece::Text("z".into())]
        );
    }

    #[test]
    fn the_marker_text_is_available_by_index() {
        let scanner = Scanner::new(QWEN);
        assert_eq!(scanner.marker(2), "<tool_call>");
    }
}
