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

/// The role tables of one scanner; see the module documentation.
#[derive(Clone)]
pub(super) struct Roles {
    /// Per byte, the role bits of the ASCII character it is; zero beyond
    /// ASCII, so a run over bytes stops there and reads the character whole.
    bytes: [Bits; 256],
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
    /// stands at `pos`, and its length. The text is valid UTF-8 and `pos` a
    /// character boundary, so the lead byte gives the length and the
    /// continuation bytes follow; the reads stay checked all the same.
    #[inline(always)]
    fn beyond(&self, bytes: &[u8], pos: usize, lead: u8) -> (Bits, usize) {
        let tail = |i: usize| u32::from(bytes.get(pos + i).copied().unwrap_or(0) & 0x3F);
        let (code, len) = if lead < 0xE0 {
            ((u32::from(lead & 0x1F) << 6) | tail(1), 2)
        } else if lead < 0xF0 {
            ((u32::from(lead & 0x0F) << 12) | (tail(1) << 6) | tail(2), 3)
        } else {
            (
                (u32::from(lead & 0x07) << 18) | (tail(1) << 12) | (tail(2) << 6) | tail(3),
                4,
            )
        };
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

    /// The end of the run of characters holding none of `bits` that starts
    /// at `pos`.
    #[inline(always)]
    pub(super) fn run_end_without(&self, text: &str, mut pos: usize, bits: Bits) -> usize {
        let bytes = text.as_bytes();
        while let Some(&lead) = bytes.get(pos) {
            if lead < 0x80 {
                if self.bytes[usize::from(lead)] & bits != 0 {
                    break;
                }
                pos += 1;
            } else {
                let (held, len) = self.beyond(bytes, pos, lead);
                if held & bits != 0 {
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
        assert_eq!(roles.run_end_without("abc  d", 0, spaces), 3);
        assert_eq!(roles.run_end_without("  ab", 0, spaces), 0);
        assert_eq!(roles.run_end_without("中文 x", 0, spaces), 6);
        assert_eq!(roles.run_end_marked("abc ", 0, letters, spaces), (3, None));
        assert_eq!(
            roles.run_end_marked("a b", 0, letters | spaces, spaces),
            (3, Some(2))
        );
    }
}
