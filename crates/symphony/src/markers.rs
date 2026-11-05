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
//! A marker is looked for only where its first two bytes stand (a newline before a newline, where
//! a marker begins with a blank line; `<` before `t` or `/`, never `<` before a letter no marker
//! continues with), or where the text's last byte alone begins one, so prose with a marker's first
//! byte in it costs nothing but the byte's test.
//!
//! The pieces of a scan are slices: of the bytes fed when nothing was held, else of the scanner's
//! own join of the held bytes and them. The scanner lends the buffers of that join out with the
//! pieces ([`Scanned`]) and takes them back once they are read ([`Scanner::reclaim`]), so a stream
//! that holds a byte at every delta allocates nothing for it after its first.

use std::{borrow::Cow, sync::Arc};

/// What a piece of the model's text turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece<'a> {
    /// Text that is not part of any marker: a slice of what was scanned, or its own copy where
    /// a caller keeps the piece past the bytes it was read from ([`Self::into_owned`]).
    Text(Cow<'a, str>),
    /// A marker, by its position in the list the scanner was given.
    Marker(usize),
}

impl Piece<'_> {
    /// The same piece owning its text, to keep past the bytes it was read from.
    pub fn into_owned(self) -> Piece<'static> {
        match self {
            Piece::Text(text) => Piece::Text(Cow::Owned(text.into_owned())),
            Piece::Marker(index) => Piece::Marker(index),
        }
    }
}

/// Splits a growing text into content and markers, holding back only what may still be a marker.
#[derive(Clone, Debug)]
pub struct Scanner {
    /// The markers, shared with every other scanner of the same list: a format builds its list
    /// once and every request's scanner holds a reference to it, not a copy.
    markers: Arc<[String]>,
    /// The bytes held back because they may still become a marker: fewer than the longest marker.
    held: String,
    /// The bytes the markers start with: a marker can begin at no other byte, so the scan looks
    /// for one only there, a word of the text at a time ([`Self::next_start`]). A multi-byte
    /// character's first byte is among them when a marker starts with it; its other bytes never
    /// are, so the scan never cuts a character.
    starts: Starts,
    /// The longest marker's length: a tail of that many bytes or more holds no half-arrived marker.
    longest: usize,
    /// Where the held bytes and the next ones are joined for a scan, on loan to the [`Scanned`] of
    /// the last feed until it is reclaimed; empty between feeds, its capacity kept.
    joined: String,
    /// The last scan's pieces, on loan and kept the same way.
    spans: Vec<Span>,
}

/// A piece, by its place in the text scanned.
#[derive(Clone, Copy, Debug)]
enum Span {
    Text { start: usize, end: usize },
    Marker(usize),
}

/// What one feed gave: the pieces, read as slices of the bytes fed or of the scanner's join of the
/// held bytes and them. The scanner's buffers are on loan here; [`Scanner::reclaim`] takes them
/// back, so that the next feed allocates nothing for them.
#[derive(Debug)]
pub struct Scanned<'a> {
    /// The bytes fed: what the pieces are slices of when nothing was held before them.
    bytes: &'a str,
    /// The held bytes joined to the bytes fed, when something was held: what the pieces are slices
    /// of then. Empty otherwise.
    joined: String,
    spans: Vec<Span>,
}

impl<'a> Scanned<'a> {
    /// The pieces, in order: text that cannot be part of a marker, and the markers found.
    pub fn pieces(&self) -> impl Iterator<Item = Piece<'_>> {
        let text = self.text();
        self.spans.iter().map(move |span| match *span {
            Span::Text { start, end } => Piece::Text(Cow::Borrowed(&text[start..end])),
            Span::Marker(index) => Piece::Marker(index),
        })
    }

    /// The text, when the pieces are one piece of text and nothing else: a caller takes it as one
    /// without walking the pieces, as it takes a delta that reads as text whole. Borrowed from
    /// the bytes fed when nothing was held before them; a copy of the join when something was.
    pub fn whole_text(&self) -> Option<Cow<'a, str>> {
        match *self.spans.as_slice() {
            [Span::Text { start, end }] if self.joined.is_empty() => {
                Some(Cow::Borrowed(&self.bytes[start..end]))
            }
            [Span::Text { start, end }] => Some(Cow::Owned(self.joined[start..end].to_owned())),
            _ => None,
        }
    }

    /// What the pieces are slices of.
    fn text(&self) -> &str {
        if self.joined.is_empty() {
            self.bytes
        } else {
            &self.joined
        }
    }
}

/// The bytes the markers start with, distinct, in the markers' order: in place for up to sixteen,
/// more than any table's markers begin with, so that a scanner costs no allocation for them; on
/// the heap past that.
#[derive(Clone, Debug)]
enum Starts {
    Few { bytes: [u8; 16], count: u8 },
    Many(Vec<u8>),
}

impl Default for Starts {
    fn default() -> Self {
        Starts::Few {
            bytes: [0; 16],
            count: 0,
        }
    }
}

impl Starts {
    /// Adds `byte`, once.
    fn add(&mut self, byte: u8) {
        match self {
            Starts::Few { bytes, count } => {
                let known = usize::from(*count);
                if bytes[..known].contains(&byte) {
                    return;
                }
                if known < bytes.len() {
                    bytes[known] = byte;
                    *count += 1;
                } else {
                    let mut all = bytes.to_vec();
                    all.push(byte);
                    *self = Starts::Many(all);
                }
            }
            Starts::Many(all) => {
                if !all.contains(&byte) {
                    all.push(byte);
                }
            }
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Starts::Few { bytes, count } => &bytes[..usize::from(*count)],
            Starts::Many(all) => all,
        }
    }
}

impl Scanner {
    /// A scanner for `markers`, each reported by its position in the list as given. With none,
    /// every byte is text. An empty string never matches, so the scan always moves forward, and of
    /// two equal markers the first in the list is the one reported.
    pub fn new(markers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::sharing(markers.into_iter().map(Into::into).collect())
    }

    /// A scanner for a list of markers built once and shared: the same scanner as [`Self::new`]
    /// would make for them, with nothing of the list copied.
    pub fn sharing(markers: Arc<[String]>) -> Self {
        let mut starts = Starts::default();
        for marker in markers.iter() {
            if let Some(&first) = marker.as_bytes().first() {
                starts.add(first);
            }
        }
        let longest = markers.iter().map(String::len).max().unwrap_or(0);
        Self {
            markers,
            held: String::new(),
            starts,
            longest,
            joined: String::new(),
            spans: Vec::new(),
        }
    }

    /// The marker a [`Piece::Marker`] refers to.
    #[cfg(test)]
    pub fn marker(&self, index: usize) -> &str {
        &self.markers[index]
    }

    /// The bytes held back because they may still become a marker.
    pub fn held(&self) -> &str {
        &self.held
    }

    /// Takes the held bytes out, so that the next feed scans afresh, as a new scanner of the same
    /// markers would: for a caller that reads a text again from a point of its own. What was held
    /// comes back through [`Self::resume`], or is fed after the text read again.
    pub fn set_aside(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Holds `held`, set aside before, again: the scanner stands where it stood before, once a
    /// scan from a point of its own holds nothing.
    pub fn resume(&mut self, held: String) {
        debug_assert!(
            self.held.is_empty(),
            "a scan that holds something is not where it was"
        );
        self.held = held;
    }

    /// Whether `bytes`, fed now, would be text whole: nothing is held, and a marker may begin at
    /// none of them, so none can be found in them or held at their end. A caller that knows takes
    /// the text as one piece without a scan.
    #[inline]
    pub fn reads_as_text(&self, bytes: &str) -> bool {
        self.held.is_empty() && !self.may_begin_anywhere(bytes.as_bytes())
    }

    /// Append the next bytes and give what is now certain, in order: text that cannot be part of
    /// a marker, and the markers found. Whatever may still become a marker stays held. Bytes that
    /// arrive while nothing is held are scanned in place, without a copy; otherwise the held bytes
    /// and they are joined in the scanner's buffer, which the pieces are read from until
    /// [`Self::reclaim`] takes it back. One held byte that the bytes' first does not continue a
    /// marker from (a newline before a letter, a bracket before a space) is text before them, and
    /// when no marker may begin in them either, the join is one piece of text without a scan: the
    /// common case of a delta that ended in such a byte.
    pub fn feed<'a>(&mut self, bytes: &'a str) -> Scanned<'a> {
        let mut joined = std::mem::take(&mut self.joined);
        let mut spans = std::mem::take(&mut self.spans);
        if self.held.is_empty() {
            self.scan(bytes, true, &mut spans);
        } else {
            let one_byte = self.held.len() == 1;
            joined.push_str(&self.held);
            joined.push_str(bytes);
            self.held.clear();
            if one_byte
                && !self.continues(joined.as_bytes(), 0)
                && !self.may_begin_anywhere(bytes.as_bytes())
            {
                spans.push(Span::Text {
                    start: 0,
                    end: joined.len(),
                });
            } else {
                self.scan(&joined, true, &mut spans);
            }
        }
        Scanned {
            bytes,
            joined,
            spans,
        }
    }

    /// No more bytes will come: what was held is a marker if it is one whole, and text otherwise.
    /// Nothing is held after.
    pub fn finish(&mut self) -> Scanned<'static> {
        let mut joined = std::mem::take(&mut self.joined);
        let mut spans = std::mem::take(&mut self.spans);
        joined.push_str(&self.held);
        self.held.clear();
        self.scan(&joined, false, &mut spans);
        Scanned {
            bytes: "",
            joined,
            spans,
        }
    }

    /// Takes back the buffers on loan to `scanned`, once its pieces are read, emptied with their
    /// capacity kept: the next feed then allocates nothing for them. A scan not reclaimed costs
    /// the next one the allocation, nothing else.
    pub fn reclaim(&mut self, scanned: Scanned<'_>) {
        self.joined = scanned.joined;
        self.joined.clear();
        self.spans = scanned.spans;
        self.spans.clear();
    }

    /// Split `text` into pieces, as spans of it. With `more_may_come`, a tail that could still
    /// grow into a marker is held instead of released. Only a position where a marker may begin is
    /// looked at: nowhere else can a marker be, or a tail be the beginning of one.
    fn scan(&mut self, text: &str, more_may_come: bool, spans: &mut Vec<Span>) {
        let bytes = text.as_bytes();
        let mut released = 0;
        let mut at = 0;
        // Only where a byte that starts some marker stands, and the next byte continues one of
        // those from it, is there anything to decide.
        while let Some(start) = self.next_start(bytes, at) {
            at = start;
            if !self.continues(bytes, at) {
                at += 1;
                continue;
            }
            let rest = &text[at..];
            if more_may_come && rest.len() < self.longest && self.could_grow_into_marker(rest) {
                Self::release(spans, released, at);
                self.held.push_str(rest);
                return;
            }
            if let Some(index) = self.longest_marker_at(rest) {
                Self::release(spans, released, at);
                spans.push(Span::Marker(index));
                at += self.markers[index].len();
                released = at;
            } else {
                at += 1;
            }
        }
        Self::release(spans, released, bytes.len());
    }

    /// Whether a marker may begin anywhere in `text`: at a byte that starts some marker and that
    /// the next byte continues one of those from ([`Self::continues`]). The start bytes are
    /// sought a word at a time ([`Self::next_start`]), and the pair test runs at the few found,
    /// since nearly every delta has none. Inlined into both callers, always: this is the test
    /// over every delta, the engine's hottest path, and a call around it, which the compiler chose
    /// once the test had a second caller, cost two per cent on every prose delta.
    #[inline(always)]
    fn may_begin_anywhere(&self, text: &[u8]) -> bool {
        let mut from = 0;
        while let Some(at) = self.next_start(text, from) {
            if self.continues(text, at) {
                return true;
            }
            from = at + 1;
        }
        false
    }

    /// The first position at or after `from` where a byte that starts some marker stands.
    #[inline(always)]
    fn next_start(&self, text: &[u8], from: usize) -> Option<usize> {
        let found = find_any(text.get(from..)?, self.starts.as_slice())?;
        Some(from + found)
    }

    /// Whether the byte after `at`, where a byte that starts some marker stands, continues one of
    /// those (a marker of one byte needs none); the text's last byte alone may, since more bytes
    /// may complete what it starts. A first byte the next byte continues no marker from is text,
    /// whatever comes later, so the scan never stops there. Read off the markers' first two bytes
    /// each time: a start byte is rare in text, and a table built per scan would cost more.
    fn continues(&self, text: &[u8], at: usize) -> bool {
        let first = text[at];
        match text.get(at + 1) {
            None => true,
            Some(&second) => self.markers.iter().any(|marker| {
                let bytes = marker.as_bytes();
                bytes.first() == Some(&first) && bytes.get(1).is_none_or(|&next| next == second)
            }),
        }
    }

    /// Whether `rest`, all of it, is a proper prefix of some marker: more bytes could complete it.
    fn could_grow_into_marker(&self, rest: &str) -> bool {
        self.markers.iter().any(|marker| {
            marker.len() > rest.len() && heads_alike(marker, rest) && marker.starts_with(rest)
        })
    }

    /// The longest marker that `rest` starts with, by index; of equal ones, the first in the list.
    fn longest_marker_at(&self, rest: &str) -> Option<usize> {
        let mut found: Option<(usize, usize)> = None;
        for (index, marker) in self.markers.iter().enumerate() {
            let longer = found.is_none_or(|(_, length)| marker.len() > length);
            if !marker.is_empty()
                && longer
                && heads_alike(marker, rest)
                && rest.starts_with(marker.as_str())
            {
                found = Some((index, marker.len()));
            }
        }
        found.map(|(index, _)| index)
    }

    fn release(spans: &mut Vec<Span>, start: usize, end: usize) {
        if start < end {
            spans.push(Span::Text { start, end });
        }
    }
}

/// Whether `a` and `b` agree on their first two bytes as far as both have them: the test before a
/// comparison of the whole, since at a byte that starts some marker most markers differ from the
/// text in their second byte (`<t`, `</`, a sentinel's first byte), and a comparison of the whole
/// costs a call for each.
fn heads_alike(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.first() == b.first() && (a.len() < 2 || b.len() < 2 || a[1] == b[1])
}

/// The lowest position in `text` of any of `bytes`, read a word of eight at a time rather than a
/// byte: the whole words of the text in order, then its last eight bytes once more where some end
/// past the words (the bytes read twice hold none, so a hit is among the new ones), and a text
/// shorter than a word as the one word its bytes make. A delta of a few bytes is one word, with
/// no loop over its bytes and no branch on each; prose fed whole is read eight bytes a step.
#[inline(always)]
fn find_any(text: &[u8], bytes: &[u8]) -> Option<usize> {
    // One or two start bytes, which the tables have, without the loop over them.
    let lanes = |word: u64| match *bytes {
        [byte] => lanes_with(word, byte),
        [first, second] => lanes_with(word, first) | lanes_with(word, second),
        _ => bytes
            .iter()
            .fold(0, |hits, &byte| hits | lanes_with(word, byte)),
    };
    let (words, rest) = text.as_chunks::<8>();
    for (index, word) in words.iter().enumerate() {
        let hits = lanes(u64::from_le_bytes(*word));
        if hits != 0 {
            return Some(index * 8 + lane_of(hits));
        }
    }
    if rest.is_empty() {
        return None;
    }
    let (start, hits) = if text.len() >= 8 {
        let start = text.len() - 8;
        let mut last = [0; 8];
        last.copy_from_slice(&text[start..]);
        (start, lanes(u64::from_le_bytes(last)))
    } else {
        // The lanes past the bytes are zero, and left out: a start byte may be zero too.
        (
            0,
            lanes(small_word(rest)) & (u64::MAX >> (8 * (8 - rest.len()))),
        )
    };
    (hits != 0).then(|| start + lane_of(hits))
}

/// The word that one to seven bytes make, in their order, zero past them: two reads of four bytes
/// that overlap, for four bytes or more, and three reads of one byte for fewer.
fn small_word(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    debug_assert!(
        (1..8).contains(&len),
        "one to seven bytes make a small word"
    );
    if len >= 4 {
        let low = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let high = u32::from_le_bytes([
            bytes[len - 4],
            bytes[len - 3],
            bytes[len - 2],
            bytes[len - 1],
        ]);
        u64::from(low) | u64::from(high) << (8 * (len - 4))
    } else {
        let (first, middle, last) = (bytes[0], bytes[len / 2], bytes[len - 1]);
        u64::from(first) | u64::from(middle) << (8 * (len / 2)) | u64::from(last) << (8 * (len - 1))
    }
}

/// The lanes of `word` that hold `byte`, each marked in its high bit. The lowest marked lane holds
/// the byte; a higher one may be marked by the borrow out of a lower hit alone, which the lowest
/// never is, and none is marked where no lane holds the byte.
fn lanes_with(word: u64, byte: u8) -> u64 {
    const LOW: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let lanes = word ^ (LOW * u64::from(byte));
    lanes.wrapping_sub(LOW) & !lanes & HIGH
}

/// The position of the lowest lane marked in `hits`.
fn lane_of(hits: u64) -> usize {
    (hits.trailing_zeros() / 8) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];
    const OUTPUT: &str = "<think>plan</think>Sure.<tool_call>{\"name\": \"f\"}</tool_call>";

    /// The pieces of a scan, owned, the scanner's buffers given back.
    fn fed(scanner: &mut Scanner, bytes: &str) -> Vec<Piece<'static>> {
        let scanned = scanner.feed(bytes);
        let pieces = scanned.pieces().map(Piece::into_owned).collect();
        scanner.reclaim(scanned);
        pieces
    }

    fn finished(scanner: &mut Scanner) -> Vec<Piece<'static>> {
        scanner.finish().pieces().map(Piece::into_owned).collect()
    }

    fn scan(markers: &[&str], pieces_in: &[&str]) -> Vec<Piece<'static>> {
        let mut scanner = Scanner::new(markers.iter().copied());
        let mut out = Vec::new();
        for piece in pieces_in {
            out.extend(fed(&mut scanner, piece));
        }
        out.extend(finished(&mut scanner));
        out
    }

    /// Adjacent text pieces joined, so chunkings compare by what they say, not how often.
    fn joined(pieces: Vec<Piece<'static>>) -> Vec<Piece<'static>> {
        let mut out: Vec<Piece<'static>> = Vec::new();
        for piece in pieces {
            match (out.last_mut(), piece) {
                (Some(Piece::Text(prev)), Piece::Text(text)) => prev.to_mut().push_str(&text),
                (_, piece) => out.push(piece),
            }
        }
        out
    }

    fn rendered(markers: &[&str], pieces: &[Piece<'_>]) -> String {
        pieces
            .iter()
            .map(|p| match p {
                Piece::Text(text) => text.as_ref(),
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
            fed(&mut scanner, "Hello <tool_c"),
            vec![Piece::Text("Hello ".into())]
        );
        assert_eq!(scanner.held(), "<tool_c");
        assert_eq!(fed(&mut scanner, "all>"), vec![Piece::Marker(2)]);
        assert_eq!(scanner.held(), "");
        assert_eq!(
            fed(&mut scanner, "<tool_x"),
            vec![Piece::Text("<tool_x".into())],
            "no longer a prefix, so text"
        );
        assert_eq!(fed(&mut scanner, "a<"), vec![Piece::Text("a".into())]);
        assert_eq!(
            scanner.held(),
            "<",
            "a lone bracket may still begin any marker"
        );
    }

    #[test]
    fn finish_releases_what_was_held_as_text_or_as_the_marker_it_turned_out_to_be() {
        let mut scanner = Scanner::new(QWEN);
        assert_eq!(
            fed(&mut scanner, "abc</thi"),
            vec![Piece::Text("abc".into())]
        );
        assert_eq!(finished(&mut scanner), vec![Piece::Text("</thi".into())]);
        assert_eq!(scanner.held(), "", "nothing is held after the end");
        assert_eq!(finished(&mut Scanner::new(QWEN)), vec![]);
        let mut scanner = Scanner::new(["ab", "abc"]);
        assert_eq!(
            fed(&mut scanner, "ab"),
            vec![],
            "held: `abc` may still come"
        );
        assert_eq!(
            finished(&mut scanner),
            vec![Piece::Marker(0)],
            "nothing came, so it was `ab`"
        );
    }

    #[test]
    fn a_marker_that_is_a_prefix_of_another_is_told_apart_once_enough_bytes_are_there() {
        let markers = ["ab", "abc"];
        let mut scanner = Scanner::new(markers);
        assert_eq!(
            fed(&mut scanner, "xab"),
            vec![Piece::Text("x".into())],
            "`ab` may still become `abc`"
        );
        assert_eq!(
            fed(&mut scanner, "d"),
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
    fn a_first_byte_the_next_byte_continues_no_marker_from_is_text_at_once() {
        // A marker that begins with a blank line: a newline before anything but a newline is
        // text, held or scanned; a newline last in the text may still begin it; and a bracket
        // before a letter no marker continues with is text too.
        const BLANK_LINE: [&str; 3] = ["\n\n<a>", "<a>", "</a>"];
        let mut scanner = Scanner::new(BLANK_LINE);
        assert!(scanner.reads_as_text("x\ny"));
        assert!(scanner.reads_as_text("a<b"));
        assert!(!scanner.reads_as_text("x\n"));
        assert!(!scanner.reads_as_text("x\n\n"));
        assert!(!scanner.reads_as_text("x<"));
        assert_eq!(
            fed(&mut scanner, "x\ny<b"),
            vec![Piece::Text("x\ny<b".into())]
        );
        assert_eq!(fed(&mut scanner, "x\n"), vec![Piece::Text("x".into())]);
        assert_eq!(scanner.held(), "\n");
        assert_eq!(fed(&mut scanner, "\n<a>"), vec![Piece::Marker(0)]);
        assert_eq!(fed(&mut scanner, "x\n"), vec![Piece::Text("x".into())]);
        assert_eq!(fed(&mut scanner, "y"), vec![Piece::Text("\ny".into())]);
        assert_eq!(scanner.held(), "");
        let mut scanner = Scanner::new(["\n"]);
        assert!(
            !scanner.reads_as_text("x\ny"),
            "a marker of one byte needs no second"
        );
        assert_eq!(
            fed(&mut scanner, "x\ny"),
            vec![
                Piece::Text("x".into()),
                Piece::Marker(0),
                Piece::Text("y".into())
            ]
        );
    }

    #[test]
    fn the_pieces_of_a_feed_after_a_hold_are_read_from_the_join_and_the_buffers_come_back() {
        let mut scanner = Scanner::new(QWEN);
        assert_eq!(fed(&mut scanner, "a<"), vec![Piece::Text("a".into())]);
        let scanned = scanner.feed("b<tool_call>c<");
        let pieces: Vec<Piece<'_>> = scanned.pieces().collect();
        assert_eq!(
            pieces,
            vec![
                Piece::Text("<b".into()),
                Piece::Marker(2),
                Piece::Text("c".into()),
            ],
            "the held bracket and the bytes fed are one piece"
        );
        assert!(
            matches!(&pieces[0], Piece::Text(Cow::Borrowed(_))),
            "a slice, not a copy"
        );
        assert_eq!(scanner.held(), "<");
        assert_eq!(scanned.pieces().count(), 3, "read as often as wanted");
        scanner.reclaim(scanned);
        assert!(
            scanner.joined.capacity() > 0 && scanner.spans.capacity() > 0,
            "the buffers are the scanner's again"
        );
        let (joined, spans) = (scanner.joined.as_ptr(), scanner.spans.as_ptr());
        let scanned = scanner.feed("x");
        assert_eq!(
            scanned.pieces().collect::<Vec<_>>(),
            vec![Piece::Text("<x".into())]
        );
        assert_eq!(
            (scanned.joined.as_ptr(), scanned.spans.as_ptr()),
            (joined, spans),
            "the same buffers serve the next feed: nothing is allocated for it"
        );
        scanner.reclaim(scanned);
        let scanned = scanner.feed("plain");
        assert_eq!(
            scanned.pieces().collect::<Vec<_>>(),
            vec![Piece::Text("plain".into())]
        );
        assert!(
            scanned.joined.is_empty(),
            "nothing held: the pieces are slices of the bytes fed, with no copy"
        );
    }

    #[test]
    fn one_piece_of_text_is_had_whole_and_a_dead_held_byte_joins_it_without_a_scan() {
        const BLANK_LINE: [&str; 3] = ["\n\n<a>", "<a>", "</a>"];
        let mut scanner = Scanner::new(BLANK_LINE);
        let scanned = scanner.feed("x\n");
        assert_eq!(
            scanned.whole_text().as_deref(),
            Some("x"),
            "the text before the held newline"
        );
        assert_eq!(scanner.held(), "\n");
        scanner.reclaim(scanned);
        let scanned = scanner.feed("yz");
        assert_eq!(
            scanned.whole_text().as_deref(),
            Some("\nyz"),
            "the newline was no marker's start"
        );
        assert_eq!(scanned.pieces().count(), 1);
        assert_eq!(scanner.held(), "");
        scanner.reclaim(scanned);
        assert_eq!(fed(&mut scanner, "x\n"), vec![Piece::Text("x".into())]);
        let scanned = scanner.feed("\n<a>y");
        assert_eq!(
            scanned.whole_text().as_deref(),
            None,
            "a marker among the pieces"
        );
        assert_eq!(
            scanned.pieces().collect::<Vec<_>>(),
            vec![Piece::Marker(0), Piece::Text("y".into())]
        );
        scanner.reclaim(scanned);
        assert_eq!(fed(&mut scanner, "x\n"), vec![Piece::Text("x".into())]);
        let scanned = scanner.feed("y<");
        assert_eq!(
            scanned.whole_text().as_deref(),
            Some("\ny"),
            "the bracket at the end is held"
        );
        assert_eq!(scanner.held(), "<");
        scanner.reclaim(scanned);
        let scanned = scanner.feed("");
        assert_eq!(
            scanned.whole_text().as_deref(),
            None,
            "nothing came: the bracket stays held"
        );
        assert_eq!(scanner.held(), "<");
        scanner.reclaim(scanned);
        assert_eq!(fed(&mut scanner, "a>"), vec![Piece::Marker(1)]);
    }

    #[test]
    fn what_is_set_aside_leaves_the_scanner_fresh_and_comes_back_whole() {
        let mut scanner = Scanner::new(QWEN);
        assert_eq!(fed(&mut scanner, "a<tool"), vec![Piece::Text("a".into())]);
        let held = scanner.set_aside();
        assert_eq!((held.as_str(), scanner.held()), ("<tool", ""));
        assert!(scanner.reads_as_text("plain"), "nothing held: a fresh scan");
        assert_eq!(fed(&mut scanner, "<tool_call>"), vec![Piece::Marker(2)]);
        scanner.resume(held);
        assert_eq!(scanner.held(), "<tool");
        assert_eq!(fed(&mut scanner, "_call>"), vec![Piece::Marker(2)]);
    }

    #[test]
    fn a_marker_that_starts_with_a_multibyte_character_is_found_at_every_cut() {
        // "｜x" starts with a three-byte character; "０" shares its first byte and is text.
        let markers = ["｜x"];
        let text = "０｜x｜y０";
        let whole = joined(scan(&markers, &[text]));
        assert_eq!(
            whole,
            vec![
                Piece::Text("０".into()),
                Piece::Marker(0),
                Piece::Text("｜y０".into()),
            ]
        );
        for (cut, _) in text.char_indices().skip(1) {
            let got = scan(&markers, &[&text[..cut], &text[cut..]]);
            assert_eq!(rendered(&markers, &got), text, "cut at {cut}");
            assert_eq!(joined(got), whole, "cut at {cut}");
        }
        let mut scanner = Scanner::new(markers);
        assert_eq!(fed(&mut scanner, "０｜"), vec![Piece::Text("０".into())]);
        assert_eq!(scanner.held(), "｜");
        assert_eq!(fed(&mut scanner, "y"), vec![Piece::Text("｜y".into())]);
    }

    #[test]
    fn text_reads_as_text_when_nothing_is_held_and_no_byte_of_it_starts_a_marker() {
        let mut scanner = Scanner::new(QWEN);
        assert!(scanner.reads_as_text("Hello, world"));
        assert!(scanner.reads_as_text(""));
        // A byte that starts a marker: the scan decides.
        assert!(!scanner.reads_as_text("Hello <"));
        assert!(!scanner.reads_as_text("<"));
        // Something held: the scan decides, whatever the bytes.
        assert_eq!(
            fed(&mut scanner, "Hello <tool_c"),
            vec![Piece::Text("Hello ".into())]
        );
        assert!(!scanner.reads_as_text("Hello"));
        assert_eq!(fed(&mut scanner, "all>"), vec![Piece::Marker(2)]);
        assert!(scanner.reads_as_text("Hello"));
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
        assert_eq!(
            fed(&mut scanner, "ok <｜tool"),
            vec![Piece::Text("ok ".into())]
        );
        assert_eq!(scanner.held(), "<｜tool");
    }

    #[test]
    fn an_empty_marker_never_matches_and_keeps_its_place_in_the_list() {
        let mut scanner = Scanner::new(["", "<think>", ""]);
        assert_eq!(
            fed(&mut scanner, "a<think>b"),
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
            fed(&mut Scanner::new([""]), "plain"),
            vec![Piece::Text("plain".into())]
        );
        assert_eq!(
            finished(&mut Scanner::new(["<think>", ""])),
            vec![],
            "finishing with nothing held ends at once"
        );
    }

    #[test]
    fn of_two_equal_markers_the_first_in_the_list_is_reported() {
        assert_eq!(
            fed(&mut Scanner::new(["<x>", "<x>", "<x>y"]), "<x>z"),
            vec![Piece::Marker(0), Piece::Text("z".into())]
        );
    }

    #[test]
    fn the_marker_text_is_available_by_index() {
        let scanner = Scanner::new(QWEN);
        assert_eq!(scanner.marker(2), "<tool_call>");
    }

    #[test]
    fn a_start_byte_is_found_a_word_at_a_time_where_a_byte_by_byte_search_finds_it() {
        let sets: &[&[u8]] = &[&[], b"<", b"<]", b"<;|", &[0, b'<', 0xff, 0x80, b'a']];
        let fillers = [b'a', 0, 1, 0x7f, 0x80, 0xff, b'<' - 1, b'<' + 1];
        let naive = |text: &[u8], set: &[u8]| text.iter().position(|byte| set.contains(byte));
        for &set in sets {
            for len in 0..=24 {
                for filler in fillers {
                    let plain = vec![filler; len];
                    assert_eq!(
                        find_any(&plain, set),
                        naive(&plain, set),
                        "{plain:?} {set:?}"
                    );
                    for at in 0..len {
                        for &start in set {
                            let mut text = plain.clone();
                            text[at] = start;
                            assert_eq!(find_any(&text, set), naive(&text, set), "{text:?} {set:?}");
                            for later in at + 1..len {
                                for &other in set {
                                    let mut text = text.clone();
                                    text[later] = other;
                                    assert_eq!(find_any(&text, set), naive(&text, set), "{text:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_marker_at_every_position_of_a_long_text_is_found_whatever_the_word_boundaries() {
        let markers = ["<x", "]]"];
        for len in 2..40_usize {
            for at in 0..=len - 2 {
                let mut text = "a".repeat(len);
                text.replace_range(at..at + 2, "<x");
                let pieces = scan(&markers, &[&text]);
                let rendered = rendered(&markers, &pieces);
                assert_eq!(rendered, text);
                assert!(pieces.contains(&Piece::Marker(0)), "{text}: {pieces:?}");
            }
        }
    }

    #[test]
    fn more_distinct_start_bytes_than_fit_in_place_are_all_kept() {
        let markers: Vec<String> = (b'a'..=b'z').map(|b| format!("{}!", b as char)).collect();
        let names: Vec<&str> = markers.iter().map(String::as_str).collect();
        for (index, marker) in names.iter().enumerate() {
            let text = format!("..{marker}..");
            let pieces = scan(&names, &[&text]);
            assert!(pieces.contains(&Piece::Marker(index)), "{text}: {pieces:?}");
            assert_eq!(rendered(&names, &pieces), text);
        }
        let scanner = Scanner::new(names.clone());
        assert!(scanner.reads_as_text("no marker here."));
        assert!(!scanner.reads_as_text("one z! here."));
        // The last byte alone may begin a marker that more bytes complete.
        assert!(!scanner.reads_as_text("no marker here"));
    }
}
