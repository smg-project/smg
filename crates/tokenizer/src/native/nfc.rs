//! The NFC quick check with its two per-character questions answered from
//! bitmaps over the Basic Multilingual Plane: whether the character is the
//! check's Yes, and whether its canonical combining class is zero. The
//! library's check (`unicode_normalization_alignments::is_nfc_quick`) asks
//! its perfect-hash tables both questions for every character, 15-20 ns a
//! character on text that is not ASCII (a Cyrillic prompt spent half of its
//! encode there: measured); here each is one bit of an 8 KiB bitmap (24 KiB
//! for the three) that stays in the first cache levels, and ASCII is skipped
//! eight bytes at a time. Nearly every character is Yes with combining class
//! 0, so the third bitmap holds those and the walk reads that one bit per
//! such character after a lead-byte decode (the two bitmaps and a character
//! decode per character were a third of a short Cyrillic encode: measured).
//! The answers are the library's own,
//! read from its functions once when the tables are built, and what the
//! tables cannot settle (a Maybe, a character outside the plane) is handed
//! back to the library.

use unicode_normalization_alignments::{
    char::canonical_combining_class, is_nfc_quick, IsNormalized,
};

use super::scan::char_beyond_ascii;

/// Code points of the plane.
const PLANE: usize = 0x1_0000;
/// Words of one bitmap over the plane.
const WORDS: usize = PLANE / 64;
/// The high bit of every byte: a word of ASCII has none set.
const NON_ASCII: u64 = 0x8080_8080_8080_8080;

/// The quick check's tables for the plane; see the module documentation.
pub(super) struct NfcQuick {
    /// Characters the quick check calls Yes whose canonical combining class
    /// is zero: the one bit the walk reads for nearly every character.
    plain: Box<[u64]>,
    /// Characters the quick check calls Yes.
    yes: Box<[u64]>,
    /// Characters whose canonical combining class is not zero.
    marked: Box<[u64]>,
}

impl NfcQuick {
    /// The tables, read from the library's functions for every character of
    /// the plane.
    pub(super) fn read() -> Self {
        let mut plain = vec![0u64; WORDS];
        let mut yes = vec![0u64; WORDS];
        let mut marked = vec![0u64; WORDS];
        for code in 0..PLANE as u32 {
            let Some(ch) = char::from_u32(code) else {
                continue;
            };
            let (word, bit) = (code as usize / 64, 1u64 << (code % 64));
            let is_yes = is_nfc_quick(std::iter::once(ch)) == IsNormalized::Yes;
            let is_marked = canonical_combining_class(ch) != 0;
            if is_yes {
                yes[word] |= bit;
            }
            if is_marked {
                marked[word] |= bit;
            }
            if is_yes && !is_marked {
                plain[word] |= bit;
            }
        }
        Self {
            plain: plain.into_boxed_slice(),
            yes: yes.into_boxed_slice(),
            marked: marked.into_boxed_slice(),
        }
    }

    /// Whether `text` is in NFC: the quick check, and the library's own
    /// stretch-wise check from the cut before the character the quick check
    /// could not place.
    pub(super) fn is_nfc(&self, text: &str) -> bool {
        match self.check(text) {
            Ok(in_nfc) => in_nfc,
            Err(cut) => super::is_nfc(&text[cut..]),
        }
    }

    /// The quick check of `text`: `Ok(true)` when it is in NFC, `Ok(false)`
    /// when it is not, `Err(cut)` when it cannot tell, as the library's
    /// answers Yes, No and Maybe. The walk is the library's: a character
    /// whose combining class is below the one before it and not zero makes
    /// the text not NFC; the first character that is not Yes stops the walk
    /// (a Maybe, which the library settles by normalizing; its No would be
    /// No here too). `cut` is the start of the last ASCII character before
    /// it, or 0: the text before the cut is Yes with combining class 0
    /// throughout, an ASCII character is the second element of no
    /// composition and reorders with nothing, so the text is in NFC exactly
    /// when the text from the cut on is, and the library reads that alone.
    pub(super) fn check(&self, text: &str) -> Result<bool, usize> {
        let bytes = text.as_bytes();
        let mut at = 0;
        let mut cut = 0;
        let mut last = 0u8;
        while at < bytes.len() {
            if bytes[at].is_ascii() {
                // ASCII is Yes with combining class 0, every character of
                // it: a byte, then whole words of it.
                at += 1;
                while let Some(word) = bytes.get(at..at + 8) {
                    if u64::from_ne_bytes(word.try_into().map_err(|_| cut)?) & NON_ASCII != 0 {
                        break;
                    }
                    at += 8;
                }
                while at < bytes.len() && bytes[at].is_ascii() {
                    at += 1;
                }
                cut = at - 1;
                last = 0;
                continue;
            }
            // `at` is on the lead byte of a character (a continuation byte
            // follows a lead byte only, and the steps here are whole
            // characters), which gives its length. A character that is Yes
            // with combining class 0, nearly every one, is one bit read; the
            // rest take both tables and the class.
            let (code, len) = char_beyond_ascii(bytes, at, bytes[at]);
            at += len;
            let code = code as usize;
            if code < PLANE && Self::bit(&self.plain, code) {
                last = 0;
                continue;
            }
            let ch = char::from_u32(code as u32).ok_or(cut)?;
            let (yes, marked) = if code < PLANE {
                (Self::bit(&self.yes, code), Self::bit(&self.marked, code))
            } else {
                (
                    is_nfc_quick(std::iter::once(ch)) == IsNormalized::Yes,
                    canonical_combining_class(ch) != 0,
                )
            };
            let class = if marked {
                canonical_combining_class(ch)
            } else {
                0
            };
            if last > class && class != 0 {
                return Ok(false);
            }
            if !yes {
                return Err(cut);
            }
            last = class;
        }
        Ok(true)
    }

    #[inline(always)]
    fn bit(table: &[u64], code: usize) -> bool {
        (table[(code / 64) & (WORDS - 1)] >> (code % 64)) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use unicode_normalization_alignments::{
        char::canonical_combining_class, is_nfc as library_is_nfc, is_nfc_quick, IsNormalized,
    };

    use super::NfcQuick;

    #[test]
    fn the_tables_are_the_librarys_answers_for_every_character_of_the_plane() {
        let quick = NfcQuick::read();
        let (mut yes, mut marked) = (0, 0);
        for code in 0..0x1_0000u32 {
            let Some(ch) = char::from_u32(code) else {
                continue;
            };
            let is_yes = is_nfc_quick(std::iter::once(ch)) == IsNormalized::Yes;
            let is_marked = canonical_combining_class(ch) != 0;
            assert_eq!(
                NfcQuick::bit(&quick.yes, code as usize),
                is_yes,
                "U+{code:04X}"
            );
            assert_eq!(
                NfcQuick::bit(&quick.marked, code as usize),
                is_marked,
                "U+{code:04X}"
            );
            assert_eq!(
                NfcQuick::bit(&quick.plain, code as usize),
                is_yes && !is_marked,
                "U+{code:04X}"
            );
            yes += usize::from(is_yes);
            marked += usize::from(is_marked);
        }
        // The plane less the surrogates is 63,488 characters; the marks are
        // a few hundred of them, the Maybes and Nos about a thousand.
        assert!((61_000..63_000).contains(&yes), "{yes}");
        assert!((500..1_000).contains(&marked), "{marked}");
    }

    #[test]
    fn the_check_answers_what_the_library_answers_and_tells_when_it_cannot() {
        let quick = NfcQuick::read();
        for (text, expected) in [
            ("", Ok(true)),
            ("plain ascii, eight bytes and more of it", Ok(true)),
            ("привет мир", Ok(true)),
            ("中文 and 漢字", Ok(true)),
            ("が", Ok(true)),
            ("\u{AC00}\u{AC01}", Ok(true)),
            ("ก\u{E38}\u{E48}", Ok(true)),
            ("🙂 and 🇫🇷", Ok(true)),
            // A mark after a letter: the library's Maybe, read from the last
            // ASCII character before it.
            ("x\u{301}", Err(0)),
            ("e\u{301}", Err(0)),
            ("abc d\u{301}", Err(4)),
            ("漢字 e\u{301}", Err(7)),
            ("\u{1100}\u{1161}", Err(0)),
            ("中文\u{1100}\u{1161}", Err(0)),
            ("a\u{301}\u{323}", Err(0)),
            // Marks out of canonical order: the library's No.
            ("ก\u{E48}\u{E38}", Ok(false)),
            ("1234567ก\u{E48}\u{E38}", Ok(false)),
        ] {
            assert_eq!(quick.check(text), expected, "{text:?}");
            assert_eq!(quick.is_nfc(text), library_is_nfc(text), "{text:?}");
        }
    }
}
