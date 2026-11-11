//! The role bits of a character, for the hand-written pre-tokenizer scanners.
//!
//! A scanner decides each alternative of its template by a few class tests
//! on the character it stands at and by runs of one class; the classes are
//! the parser's, each in a role of the template. [`Roles`] answers every
//! role's test for one character in one read: a table of role bits per
//! ASCII byte (a byte beyond ASCII reads as no role there), one per code
//! point of the Basic Multilingual Plane built from the classes when the
//! scanner is recognized, and the classes themselves above the plane. The
//! plane table is kept by 256-code-point blocks, blocks with the same bits
//! sharing one row: the ideographs, the kana, a Cyrillic or Thai block each
//! read from one 512-byte row, so the table a text touches stays in the
//! first-level cache (a flat table of the plane is 128 KiB and missed the
//! cache on every character of such a run: measured, a seventh of the
//! encode's instructions on a corpus with such scripts in it). A run of one
//! class is then a loop over bytes with one load per ASCII byte; a character
//! beyond ASCII is decoded from its bytes in place (the text is valid UTF-8
//! and the scanners stand on character boundaries, so the lead byte gives
//! the length and no validation is repeated).

use std::collections::HashMap;

use super::pattern::Class;

/// The role bits of one character: one bit per role of a scanner, up to
/// sixteen.
pub(super) type Bits = u16;

/// Code points of the plane table.
const PLANE: u32 = 0x1_0000;

/// Code points per block of the plane table.
const BLOCK: usize = 256;

/// The code point of the character beyond ASCII whose lead byte `lead`
/// stands at `pos` of `bytes`, and its length. The text is valid UTF-8 and
/// `pos` a character boundary, so the lead byte gives the length and the
/// continuation bytes follow; the reads stay checked all the same.
#[inline(always)]
pub(super) fn char_beyond_ascii(bytes: &[u8], pos: usize, lead: u8) -> (u32, usize) {
    let tail = |i: usize| u32::from(bytes.get(pos + i).copied().unwrap_or(0) & 0x3F);
    if lead < 0xE0 {
        ((u32::from(lead & 0x1F) << 6) | tail(1), 2)
    } else if lead < 0xF0 {
        ((u32::from(lead & 0x0F) << 12) | (tail(1) << 6) | tail(2), 3)
    } else {
        (
            (u32::from(lead & 0x07) << 18) | (tail(1) << 12) | (tail(2) << 6) | tail(3),
            4,
        )
    }
}

/// Bit 7 of each byte of a word.
const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

/// Each byte of a word.
const EACH_BYTE: u64 = 0x0101_0101_0101_0101;

/// The word scans: eight bytes of a run tested at once, where the byte loop
/// reads the table and takes three branches per byte, and leaves after a
/// run length the branch predictor cannot learn. A scan stands for one of
/// the sets the ASCII members of a template's run roles are (the letters,
/// the lower-case letters, the upper-case letters, the white space); the
/// caller of [`Roles::run_end_as`] names the scan of the role whose run it
/// walks, and the scan is taken when the role's members are that set,
/// which [`Roles::build`] reads off the byte table once. Any other role
/// (the punctuation class is everything the other classes are not; a run
/// of line breaks is one or two characters) runs byte by byte.
pub(super) struct WordScan;

impl WordScan {
    /// No word scan: the mark of a role whose members are no scan's set,
    /// which runs byte by byte under any name a caller passes. Not a scan a
    /// caller may name ([`Roles::run_end_as`] refuses it at compile time).
    pub(super) const NONE: u8 = 0;
    /// `A-Z` and `a-z` (tested as `a-z` with the case bit set).
    pub(super) const LETTERS: u8 = 1;
    /// `a-z`.
    pub(super) const LOWER: u8 = 2;
    /// `A-Z`.
    pub(super) const UPPER: u8 = 3;
    /// `\t`-`\r` and the space.
    pub(super) const SPACES: u8 = 4;

    /// The scan whose set the ASCII members `members` of a role are (bit
    /// `code` of the word is set for code point `code`); `NONE` for any
    /// other set.
    fn of(members: u128) -> u8 {
        let letters = ascii_range(b'A', b'Z') | ascii_range(b'a', b'z');
        let spaces = ascii_range(b'\t', b'\r') | ascii_range(b' ', b' ');
        [
            (Self::LETTERS, letters),
            (Self::LOWER, ascii_range(b'a', b'z')),
            (Self::UPPER, ascii_range(b'A', b'Z')),
            (Self::SPACES, spaces),
        ]
        .into_iter()
        .find(|(_, set)| *set == members)
        .map_or(Self::NONE, |(scan, _)| scan)
    }

    /// Bit 7 of each byte of `word` that is outside the set of `SCAN`, or
    /// beyond ASCII. `SCAN` is one of the four scans, checked when the
    /// function is instantiated.
    #[inline(always)]
    fn outside<const SCAN: u8>(word: u64) -> u64 {
        const {
            assert!(
                SCAN >= Self::LETTERS && SCAN <= Self::SPACES,
                "one of the four word scans"
            );
        };
        let low = word & !HIGH_BITS;
        let inside = if SCAN == Self::LETTERS {
            between(low | (EACH_BYTE * 0x20), b'a', b'z')
        } else if SCAN == Self::LOWER {
            between(low, b'a', b'z')
        } else if SCAN == Self::UPPER {
            between(low, b'A', b'Z')
        } else {
            // `SPACES`, the one scan left.
            between(low, b'\t', b'\r') | between(low, b' ', b' ')
        };
        (inside & !word & HIGH_BITS) ^ HIGH_BITS
    }
}

/// The code points `from..=to` as bits of a word over ASCII.
const fn ascii_range(from: u8, to: u8) -> u128 {
    let mut set = 0u128;
    let mut code = from;
    while code <= to {
        set |= 1 << code;
        code += 1;
    }
    set
}

/// Bit 7 of each byte of `low` (every high bit clear) that is in
/// `from..=to`: `byte + (0x80 - from)` sets the high bit of the bytes at or
/// above `from` and `byte + (0x7F - to)` that of the bytes above `to`, no
/// sum reaching the next byte.
#[inline(always)]
const fn between(low: u64, from: u8, to: u8) -> u64 {
    low.wrapping_add(EACH_BYTE * (0x80 - from as u64))
        & !low.wrapping_add(EACH_BYTE * (0x7F - to as u64))
}

/// The role tables of one scanner; see the module documentation.
#[derive(Clone)]
pub(super) struct Roles {
    /// Per byte, the role bits of the ASCII character it is; zero beyond
    /// ASCII, so a run over bytes stops there and reads the character whole.
    bytes: [Bits; 256],
    /// Per role, the word scan its ASCII members are the set of
    /// ([`WordScan`]), `WordScan::NONE` for most.
    scans: [u8; Bits::BITS as usize],
    /// Per block of the plane, the row of `rows` that holds its bits.
    block_rows: [u16; 256],
    /// The distinct rows of the plane table, `BLOCK` bits each.
    rows: Box<[Bits]>,
    /// The class of each role, read above the plane.
    classes: Vec<Class>,
}

impl std::fmt::Debug for Roles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Roles")
            .field("roles", &self.classes.len())
            .field("rows", &(self.rows.len() / BLOCK))
            .finish()
    }
}

impl Roles {
    /// The tables of `classes`, one class per role in the order of their
    /// bits.
    pub(super) fn build(classes: &[&Class]) -> Self {
        debug_assert!(classes.len() <= Bits::BITS as usize);
        let bits_of = |c: char| -> Bits {
            classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.contains(c))
                .fold(0, |bits, (slot, _)| bits | (1 << slot))
        };
        let mut bytes = [0; 256];
        for (code, bits) in bytes.iter_mut().enumerate().take(128) {
            *bits = bits_of(char::from(code as u8));
        }
        let mut scans = [WordScan::NONE; Bits::BITS as usize];
        for (slot, scan) in scans.iter_mut().enumerate().take(classes.len()) {
            let members = bytes
                .iter()
                .enumerate()
                .take(128)
                .filter(|(_, bits)| *bits & (1 << slot) != 0)
                .fold(0u128, |members, (code, _)| members | (1 << code));
            *scan = WordScan::of(members);
        }
        let mut block_rows = [0u16; 256];
        let mut rows: Vec<Bits> = Vec::new();
        let mut seen: HashMap<[Bits; BLOCK], u16> = HashMap::new();
        for (block, row) in block_rows.iter_mut().enumerate() {
            let mut block_bits = [0; BLOCK];
            for (low, bits) in block_bits.iter_mut().enumerate() {
                let code = (block * BLOCK + low) as u32;
                *bits = char::from_u32(code).map_or(0, bits_of);
            }
            *row = *seen.entry(block_bits).or_insert_with(|| {
                rows.extend_from_slice(&block_bits);
                (rows.len() / BLOCK - 1) as u16
            });
        }
        Self {
            bytes,
            scans,
            block_rows,
            rows: rows.into_boxed_slice(),
            classes: classes.iter().map(|class| (*class).clone()).collect(),
        }
    }

    /// The role bits of the character at `pos`, a character boundary, and
    /// its length in bytes; `None` at the end of the text.
    #[inline(always)]
    pub(super) fn at(&self, text: &str, pos: usize) -> Option<(Bits, usize)> {
        let bytes = text.as_bytes();
        let lead = *bytes.get(pos)?;
        if lead < 0x80 {
            Some((self.bytes[usize::from(lead)], 1))
        } else {
            Some(self.beyond(bytes, pos, lead))
        }
    }

    /// The role bits of the character beyond ASCII whose lead byte `lead`
    /// stands at `pos`, and its length.
    #[inline(always)]
    fn beyond(&self, bytes: &[u8], pos: usize, lead: u8) -> (Bits, usize) {
        let (code, len) = char_beyond_ascii(bytes, pos, lead);
        let bits = if code < PLANE {
            let row = usize::from(self.block_rows[(code >> 8) as usize]);
            self.rows[(row << 8) | (code & 0xFF) as usize]
        } else {
            self.above(code)
        };
        (bits, len)
    }

    /// The role bits of a code point above the plane, from the classes.
    #[cold]
    #[inline(never)]
    fn above(&self, code: u32) -> Bits {
        let Some(c) = char::from_u32(code) else {
            return 0;
        };
        self.classes
            .iter()
            .enumerate()
            .filter(|(_, class)| class.contains(c))
            .fold(0, |bits, (slot, _)| bits | (1 << slot))
    }

    /// [`Self::run_end`] for a role whose run the caller expects to be the
    /// word scan `SCAN` (one of the four of [`WordScan`], never `NONE`: a
    /// compile-time error): eight bytes at a time when the role's ASCII
    /// members are that set, else byte by byte. The first byte is read as
    /// the byte loop reads it (a run that ends at once costs what it did,
    /// and so does a run of characters beyond ASCII); after an ASCII member,
    /// whole words of members pass at once up to the first byte outside the
    /// set, which ends the run when it is ASCII and is read by
    /// [`Self::beyond`] when it is not, or the tail under eight bytes. With
    /// `WORDS` off (an input under the encoder's `WORD_SCAN_MIN_BYTES`) it is
    /// [`Self::run_end`] whatever `SCAN` names.
    #[inline(always)]
    pub(super) fn run_end_as<const SCAN: u8, const WORDS: bool>(
        &self,
        text: &str,
        mut pos: usize,
        bit: Bits,
    ) -> usize {
        const {
            assert!(
                SCAN >= WordScan::LETTERS && SCAN <= WordScan::SPACES,
                "one of the four word scans; a role with no scan runs through run_end"
            );
        };
        debug_assert_eq!(bit.count_ones(), 1, "one role's bit");
        if !WORDS || self.scans[bit.trailing_zeros() as usize % Bits::BITS as usize] != SCAN {
            return self.run_end(text, pos, bit);
        }
        let bytes = text.as_bytes();
        loop {
            let Some(&lead) = bytes.get(pos) else {
                return pos;
            };
            if lead >= 0x80 {
                let (bits, len) = self.beyond(bytes, pos, lead);
                if bits & bit == 0 {
                    return pos;
                }
                pos += len;
                continue;
            }
            if self.bytes[usize::from(lead)] & bit == 0 {
                return pos;
            }
            pos += 1;
            for word in bytes[pos..].as_chunks::<8>().0 {
                let outside = WordScan::outside::<SCAN>(u64::from_le_bytes(*word));
                if outside != 0 {
                    // The first lane outside the set (inside this word): an
                    // ASCII byte there is not a member, so the run ends at it.
                    pos += (outside.trailing_zeros() / 8) as usize;
                    if bytes[pos] < 0x80 {
                        return pos;
                    }
                    break;
                }
                pos += 8;
            }
        }
    }

    /// The word scan of the role of `bit` (`WordScan::NONE` when it runs byte
    /// by byte); for the tests that pin the templates' scans.
    #[cfg(test)]
    pub(super) fn scan_of(&self, bit: Bits) -> u8 {
        debug_assert_eq!(bit.count_ones(), 1, "one role's bit");
        self.scans[bit.trailing_zeros() as usize]
    }

    /// The end of the run of characters holding `bit` that starts at `pos`.
    #[inline(always)]
    pub(super) fn run_end(&self, text: &str, mut pos: usize, bit: Bits) -> usize {
        let bytes = text.as_bytes();
        while let Some(&lead) = bytes.get(pos) {
            if lead < 0x80 {
                if self.bytes[usize::from(lead)] & bit == 0 {
                    break;
                }
                pos += 1;
            } else {
                let (bits, len) = self.beyond(bytes, pos, lead);
                if bits & bit == 0 {
                    break;
                }
                pos += len;
            }
        }
        pos
    }

    /// The end of the run of characters holding `bit` that starts at `pos`,
    /// and the end of the last character of that run which also holds
    /// `marked`, if any.
    #[inline(always)]
    pub(super) fn run_end_marked(
        &self,
        text: &str,
        mut pos: usize,
        bit: Bits,
        marked: Bits,
    ) -> (usize, Option<usize>) {
        let mut last_marked_end = None;
        while let Some((bits, len)) = self.at(text, pos) {
            if bits & bit == 0 {
                break;
            }
            if bits & marked != 0 {
                last_marked_end = Some(pos + len);
            }
            pos += len;
        }
        (pos, last_marked_end)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::pattern::Pattern, *};

    /// The role bits read through the tables equal the bits computed from
    /// the classes for every code point, through every byte length, and the
    /// lengths are the characters'.
    #[test]
    fn the_tables_read_every_code_point_as_the_classes_do() {
        // Classes of every kind: a category, a negated set with a category,
        // a range set, whitespace, a single character.
        let pattern =
            Pattern::parse(r"\p{L}|[^\r\n\p{L}\p{N}]|[一-龥぀-ゟ゠-ヿ]|\s|\p{N}|'").expect("parses");
        let classes: Vec<&Class> = pattern.classes().iter().collect();
        let roles = Roles::build(&classes);
        let mut text = String::new();
        for code in (0..0x11_0000u32).filter(|code| code % 7 == 0 || *code < 0x3000) {
            let Some(c) = char::from_u32(code) else {
                continue;
            };
            let expected = classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.contains(c))
                .fold(0, |bits, (slot, _)| bits | (1 << slot));
            text.clear();
            text.push(c);
            text.push('x');
            assert_eq!(
                roles.at(&text, 0),
                Some((expected, c.len_utf8())),
                "U+{code:04X}"
            );
        }
        assert_eq!(roles.at("", 0), None);
        assert_eq!(roles.at("ab", 2), None);
    }

    #[test]
    fn runs_end_where_the_class_ends_whatever_the_byte_lengths() {
        let pattern = Pattern::parse(r"\p{L}|\s").expect("parses");
        let classes: Vec<&Class> = pattern.classes().iter().collect();
        let roles = Roles::build(&classes);
        let letters: Bits = 1;
        let spaces: Bits = 2;
        for (text, end) in [
            ("abc", 3),
            ("abc def", 3),
            ("日本語の文章 x", 18),
            ("éàü1", 6),
            ("😀a", 0),
            ("a😀", 1),
            ("", 0),
            ("   ", 0),
        ] {
            assert_eq!(roles.run_end(text, 0, letters), end, "{text:?}");
        }
        assert_eq!(roles.run_end_marked("abc ", 0, letters, spaces), (3, None));
        assert_eq!(
            roles.run_end_marked("a b", 0, letters | spaces, spaces),
            (3, Some(2))
        );
    }

    /// The roles of the templates: the letters, the lower-case and upper-case
    /// letters and the white space have a word scan, the digits, the
    /// punctuation class (everything else) and the line breaks have none;
    /// and every scan reads each byte of a word as the table does.
    #[test]
    fn the_word_scans_are_the_roles_byte_tables() {
        let pattern =
            Pattern::parse(r"\p{L}|\p{Ll}|\p{Lu}|\s|\p{N}|[^\s\p{L}\p{N}]|[\r\n]").expect("parses");
        let classes: Vec<&Class> = pattern.classes().iter().collect();
        let roles = Roles::build(&classes);
        assert_eq!(
            roles.scans[..7],
            [
                WordScan::LETTERS,
                WordScan::LOWER,
                WordScan::UPPER,
                WordScan::SPACES,
                WordScan::NONE,
                WordScan::NONE,
                WordScan::NONE
            ]
        );
        let outside = |scan: u8, word: u64| match scan {
            WordScan::LETTERS => WordScan::outside::<{ WordScan::LETTERS }>(word),
            WordScan::LOWER => WordScan::outside::<{ WordScan::LOWER }>(word),
            WordScan::UPPER => WordScan::outside::<{ WordScan::UPPER }>(word),
            WordScan::SPACES => WordScan::outside::<{ WordScan::SPACES }>(word),
            other => panic!("no word scan {other}"),
        };
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for slot in 0..4 {
            let bit = 1 << slot;
            for word in (0..=255u64)
                .map(|byte| byte * EACH_BYTE)
                .chain((0..20_000).map(|_| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    seed
                }))
            {
                let expected = word
                    .to_le_bytes()
                    .iter()
                    .enumerate()
                    .filter(|(_, &byte)| byte >= 0x80 || roles.bytes[usize::from(byte)] & bit == 0)
                    .fold(0u64, |outside, (lane, _)| outside | 0x80 << (8 * lane));
                assert_eq!(
                    outside(roles.scans[slot], word),
                    expected,
                    "slot {slot} word {word:#018x}"
                );
            }
        }
    }

    /// A run ends at the same place through a word scan as byte by byte,
    /// whatever the mix of ASCII members, members beyond ASCII, non-members
    /// of both kinds and the offset of the run in the text; and a scan named
    /// for a role whose members are another set is the byte loop. The pieces
    /// longer than a word reach the chunk loop's branches, each hundreds of
    /// times (counted from the byte loop's answer): a whole word of members
    /// passing at once, a second word after it, the tail under eight bytes
    /// after a word, a member beyond ASCII after a word.
    #[test]
    fn runs_end_the_same_through_the_word_scans_and_byte_by_byte() {
        let pattern = Pattern::parse(r"\p{L}|\s|\p{Ll}|[\r\n]").expect("parses");
        let classes: Vec<&Class> = pattern.classes().iter().collect();
        let roles = Roles::build(&classes);
        let pieces = [
            "a",
            "Z",
            " ",
            "\n",
            "\r",
            "1",
            ".",
            "é",
            "日",
            "×",
            "😀",
            "٣",
            "abcdefgh",
            "ABCDEFGHIJK",
            "lowercase",
            "        ",
            "\t\t\t\t\t",
            "abcdefghijé",
        ];
        let ascii = |bytes: &[u8]| bytes.iter().all(|byte| *byte < 0x80);
        let mut reached = [0usize; 4];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut text = String::new();
        for _ in 0..3_000 {
            text.clear();
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = (seed % 40) as usize;
            let mut pick = seed;
            for _ in 0..len {
                pick = pick.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(1);
                text.push_str(pieces[((pick >> 33) % pieces.len() as u64) as usize]);
            }
            for pos in text.char_indices().map(|(pos, _)| pos).chain([text.len()]) {
                let by_bytes = |bit: Bits| roles.run_end(&text, pos, bit);
                for bit in [1, 2, 4] {
                    let run = &text.as_bytes()[pos..by_bytes(bit)];
                    if run.len() >= 9 && ascii(&run[..9]) {
                        reached[0] += 1;
                        reached[1] += usize::from(run.len() >= 17 && ascii(&run[..17]));
                        reached[2] += usize::from(
                            pos + run.len() == text.len()
                                && ascii(run)
                                && !(run.len() - 1).is_multiple_of(8),
                        );
                        reached[3] += usize::from(!ascii(&run[9..]));
                    }
                }
                assert_eq!(
                    roles.run_end_as::<{ WordScan::LETTERS }, true>(&text, pos, 1),
                    by_bytes(1),
                    "{text:?} at {pos} letters"
                );
                assert_eq!(
                    roles.run_end_as::<{ WordScan::SPACES }, true>(&text, pos, 2),
                    by_bytes(2),
                    "{text:?} at {pos} spaces"
                );
                assert_eq!(
                    roles.run_end_as::<{ WordScan::LOWER }, true>(&text, pos, 4),
                    by_bytes(4),
                    "{text:?} at {pos} lower"
                );
                // The line breaks are no scan's set: the byte loop under any name.
                assert_eq!(
                    roles.run_end_as::<{ WordScan::SPACES }, true>(&text, pos, 8),
                    by_bytes(8),
                    "{text:?} at {pos} breaks"
                );
            }
        }
        assert!(
            reached.iter().all(|&times| times >= 100),
            "a word, two words, a tail after a word, beyond ASCII after a word: {reached:?}"
        );
    }
}
