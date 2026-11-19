//! The role bits of a character, for the hand-written pre-tokenizer scanners.
//!
//! A scanner decides each alternative of its template by a few class tests
//! on the character it stands at and by runs of one class; the classes are
//! the parser's, each in a role of the template. [`Roles`] answers every
//! role's test for one character in one read: a table of role bits per
//! ASCII byte (a byte beyond ASCII reads as no role there), one per code
//! point of the Basic Multilingual Plane built from the classes when the
//! scanner is recognized, and above the plane the runs of code points the
//! classes read alike: one binary search per character, where asking the
//! classes is a search per category table per class (measured: two thirds
//! of an encode's cycles on a text with one emoji per 134 bytes). The plane
//! table is kept by 64-code-point blocks, blocks with the same bits
//! sharing one row: the ideographs read from one 128-byte row, the kana, a
//! Cyrillic or Thai block from a few, so the table a text touches stays in
//! the first-level cache (a flat table of the plane is 128 KiB and missed
//! the cache on every character of such a run: measured, a seventh of the
//! encode's instructions on a corpus with such scripts in it). Such a block
//! is the characters whose bytes differ in the last one alone: the bytes
//! before it name the block and its six bits the place in the row, so a
//! character of the plane is read from its bytes as they stand and no code
//! point is put together. A run of one class is then a loop over bytes with
//! one load per ASCII byte; a character beyond ASCII is read from its bytes
//! in place (the text is valid UTF-8 and the scanners stand on character
//! boundaries, so the lead byte gives the length and no validation is
//! repeated).

use rustc_hash::FxHashMap;

use super::pattern::Class;

/// The role bits of one character: one bit per role of a scanner, up to
/// sixteen.
pub(super) type Bits = u16;

/// Code points of the plane table.
const PLANE: u32 = 0x1_0000;

/// Code points per block of the plane table: the characters whose UTF-8
/// bytes differ in the last one alone.
const BLOCK: usize = 64;

/// Blocks of the plane table. There are at most as many rows as blocks, so
/// the start of a row fits sixteen bits whatever the classes are.
const BLOCKS: usize = PLANE as usize / BLOCK;
const _: () = assert!(BLOCKS * BLOCK - BLOCK <= u16::MAX as usize);
// A block is one word of a class's bitmap of the plane (`Roles::build`).
const _: () = assert!(BLOCK == u64::BITS as usize);

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
    /// Every ASCII character of any of the four sets (the letters and the
    /// white space), as a bit per code: a scanner whose runs stop at other
    /// classes checks those classes against it (see `chain`).
    pub(super) const MEMBERS: u128 = ascii_range(b'A', b'Z')
        | ascii_range(b'a', b'z')
        | ascii_range(b'\t', b'\r')
        | ascii_range(b' ', b' ');
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
        (Self::inside::<SCAN>(word & !HIGH_BITS) & !word & HIGH_BITS) ^ HIGH_BITS
    }

    /// [`Self::outside`] for a word of an all-ASCII text: no byte has its
    /// high bit set, so the set's test is the whole of it, three operations
    /// and one constant fewer per word.
    #[inline(always)]
    fn outside_ascii<const SCAN: u8>(word: u64) -> u64 {
        debug_assert_eq!(word & HIGH_BITS, 0, "a word of an all-ASCII text");
        !Self::inside::<SCAN>(word) & HIGH_BITS
    }

    /// Bit 7 of each byte of `low` (every high bit clear) that is in the set
    /// of `SCAN`; the other bits of the result are not defined.
    #[inline(always)]
    fn inside<const SCAN: u8>(low: u64) -> u64 {
        const {
            assert!(
                SCAN >= Self::LETTERS && SCAN <= Self::SPACES,
                "one of the four word scans"
            );
        };
        if SCAN == Self::LETTERS {
            between(low | (EACH_BYTE * 0x20), b'a', b'z')
        } else if SCAN == Self::LOWER {
            between(low, b'a', b'z')
        } else if SCAN == Self::UPPER {
            between(low, b'A', b'Z')
        } else {
            // `SPACES`, the one scan left.
            between(low, b'\t', b'\r') | between(low, b' ', b' ')
        }
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
    /// Per block of the plane, where the row that holds its bits starts in
    /// `rows`.
    block_rows: [u16; BLOCKS],
    /// The distinct rows of the plane table, `BLOCK` bits each.
    rows: Box<[Bits]>,
    /// The role bits above the plane as runs in code point order: the first
    /// code point of each run and its bits; a run ends where the next starts.
    above: Box<[(u32, Bits)]>,
}

impl std::fmt::Debug for Roles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Roles")
            .field("rows", &(self.rows.len() / BLOCK))
            .field("runs_above", &self.above.len())
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
        let mut block_rows = [0u16; BLOCKS];
        let mut rows: Vec<Bits> = Vec::new();
        let mut seen: FxHashMap<[Bits; BLOCK], u16> = FxHashMap::default();
        // The row of the block before, when that block was read from the
        // classes' words: a block with the same words shares it. The
        // ideographs, the syllables and the unassigned stretches are long
        // runs of such blocks.
        let mut before: Option<u16> = None;
        for (block, row) in block_rows.iter_mut().enumerate() {
            let mut block_bits = [0; BLOCK];
            let first = (block * BLOCK) as u32;
            let by_words = first >= 0x80 && !(0xD800..0xE000).contains(&first);
            if !by_words {
                // ASCII has a table of its own in each class, and no
                // character is a surrogate: asked one code point at a time.
                for (low, bits) in block_bits.iter_mut().enumerate() {
                    *bits = char::from_u32(first + low as u32).map_or(0, bits_of);
                }
            } else if let Some(same) = before.filter(|_| {
                classes
                    .iter()
                    .all(|class| class.plane_block(block) == class.plane_block(block - 1))
            }) {
                *row = same;
                continue;
            } else {
                // A block is one word of each class's bitmap of the plane,
                // and most blocks hold a class whole or not at all: the
                // classes that hold it whole are set in one pass at the end,
                // the others bit by bit of their word. Asking every class
                // about every code point cost 12 M instructions per scanner
                // at load (counted).
                let mut whole: Bits = 0;
                for (slot, class) in classes.iter().enumerate() {
                    let mut members = class.plane_block(block);
                    if members == u64::MAX {
                        whole |= 1 << slot;
                        continue;
                    }
                    while members != 0 {
                        block_bits[members.trailing_zeros() as usize] |= 1 << slot;
                        members &= members - 1;
                    }
                }
                for bits in &mut block_bits {
                    *bits |= whole;
                }
            }
            *row = *seen.entry(block_bits).or_insert_with(|| {
                rows.extend_from_slice(&block_bits);
                (rows.len() - BLOCK) as u16
            });
            before = by_words.then_some(*row);
        }
        // Above the plane the bits change only where a class's membership
        // can: one run from each such point to the next, equal neighbours
        // as one, and a run without bits past the last character. The
        // points come as a few sorted tables, which the stable sort merges,
        // and each class answers for all of them in one walk along its
        // tables, where asking point by point is a search per table per
        // class.
        let limit = u32::from(char::MAX) + 1;
        let mut points = vec![PLANE];
        for class in classes {
            class.changes_from(PLANE, &mut points);
        }
        #[expect(
            clippy::stable_sort_primitive,
            reason = "the stable sort merges sorted tables"
        )]
        points.sort();
        points.dedup();
        points.pop_if(|code| *code == limit);
        let mut point_bits: Vec<Bits> = vec![0; points.len()];
        let mut members = Vec::new();
        for (slot, class) in classes.iter().enumerate() {
            class.members_among(&points, &mut members);
            for (bits, &member) in point_bits.iter_mut().zip(&members) {
                *bits |= Bits::from(member) << slot;
            }
        }
        let mut above: Vec<(u32, Bits)> = Vec::new();
        for (code, bits) in points.into_iter().zip(point_bits).chain([(limit, 0)]) {
            if above.last().is_none_or(|&(_, last)| last != bits) {
                above.push((code, bits));
            }
        }
        Self {
            bytes,
            scans,
            block_rows,
            rows: rows.into_boxed_slice(),
            above: above.into_boxed_slice(),
        }
    }

    /// The ASCII characters holding any of `bits`, as a bit per code.
    pub(super) fn ascii_members(&self, bits: Bits) -> u128 {
        Self::ascii_members_of(&self.bytes, bits)
    }

    /// [`Self::ascii_members`] of the byte table `bytes`.
    fn ascii_members_of(bytes: &[Bits; 256], bits: Bits) -> u128 {
        bytes
            .iter()
            .take(128)
            .enumerate()
            .filter(|(_, role_bits)| **role_bits & bits != 0)
            .fold(0u128, |members, (code, _)| members | (1 << code))
    }

    /// Whether every character holds one of `bits`. The plane is asked a
    /// character at a time through [`Self::at`] and not row by row, so the
    /// answer does not lean on how the plane's table is laid out; it is
    /// asked once per pattern. Above the plane the runs say it.
    pub(super) fn every_char_holds(&self, bits: Bits) -> bool {
        let mut utf8 = [0; 4];
        let in_plane = (0..PLANE).filter_map(char::from_u32).all(|c| {
            self.at(c.encode_utf8(&mut utf8), 0)
                .is_some_and(|(held, _)| held & bits != 0)
        });
        let above = self
            .above
            .iter()
            .all(|&(start, held)| held & bits != 0 || start > u32::from(char::MAX));
        in_plane && above
    }

    /// Whether some character holds a bit of `one` and a bit of `other`
    /// at once. Every entry of the tables is the bits of a code point, so
    /// they are read as they stand; it is asked once per pattern.
    pub(super) fn some_char_holds_both(&self, one: Bits, other: Bits) -> bool {
        let both = |held: Bits| held & one != 0 && held & other != 0;
        self.bytes.iter().any(|&held| both(held))
            || self.rows.iter().any(|&held| both(held))
            || self.above.iter().any(|&(_, held)| both(held))
    }

    /// Takes `bits` from every character that holds a bit of `of`: a role
    /// such a character never plays in its scanner (see `cased`). The word
    /// scans are named again from the byte table.
    pub(super) fn clear_where(&mut self, of: Bits, bits: Bits) {
        let clear = |held: &mut Bits| {
            if *held & of != 0 {
                *held &= !bits;
            }
        };
        self.bytes.iter_mut().for_each(clear);
        self.rows.iter_mut().for_each(clear);
        self.above.iter_mut().for_each(|(_, held)| clear(held));
        for (slot, scan) in self.scans.iter_mut().enumerate() {
            *scan = WordScan::of(Self::ascii_members_of(&self.bytes, 1 << slot));
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
    /// stands at `pos`, and its length. A character of two or three bytes
    /// is one of the plane, and the bytes before its last name its block;
    /// one of four bytes stands above the plane. The text is valid UTF-8
    /// and `pos` a character boundary, so the bytes the lead announces
    /// follow; the reads stay checked all the same.
    #[inline(always)]
    fn beyond(&self, bytes: &[u8], pos: usize, lead: u8) -> (Bits, usize) {
        let tail = |i: usize| bytes.get(pos + i).copied().unwrap_or(0);
        if lead < 0xE0 {
            (self.in_block(usize::from(lead & 0x1F), tail(1)), 2)
        } else if lead < 0xF0 {
            let block = (usize::from(lead & 0x0F) << 6) | usize::from(tail(1) & 0x3F);
            (self.in_block(block, tail(2)), 3)
        } else {
            self.above(bytes, pos, lead)
        }
    }

    /// The role bits of the character of `block` whose last byte is `last`.
    #[inline(always)]
    fn in_block(&self, block: usize, last: u8) -> Bits {
        let row = usize::from(self.block_rows[block]);
        self.rows[row | usize::from(last & 0x3F)]
    }

    /// The role bits of the character of four bytes whose lead byte `lead`
    /// stands at `pos`, and its length: one above the plane, which reads
    /// the bits of its run.
    #[cold]
    #[inline(never)]
    fn above(&self, bytes: &[u8], pos: usize, lead: u8) -> (Bits, usize) {
        let (code, len) = char_beyond_ascii(bytes, pos, lead);
        let next = self.above.partition_point(|&(start, _)| start <= code);
        (next.checked_sub(1).map_or(0, |run| self.above[run].1), len)
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
        pos: usize,
        bit: Bits,
    ) -> usize {
        const {
            assert!(
                SCAN >= WordScan::LETTERS && SCAN <= WordScan::SPACES,
                "one of the four word scans; a role with no scan runs through the byte loop"
            );
        };
        debug_assert_eq!(bit.count_ones(), 1, "one role's bit");
        if !WORDS || self.scans[bit.trailing_zeros() as usize % Bits::BITS as usize] != SCAN {
            return self.run_end(text, pos, bit);
        }
        self.words_end::<SCAN>(text, pos, bit, 0)
    }

    /// [`Self::run_end_as`] for the chain scanner, whose runs also end at a
    /// character holding one of the `stop` bits (its stage classes): the
    /// same words, [`Self::run_end_in`] as the byte loop. The caller has read
    /// that no stop class holds an ASCII member of the set.
    #[inline(always)]
    pub(super) fn run_end_as_in<const SCAN: u8>(
        &self,
        text: &str,
        pos: usize,
        bit: Bits,
        stop: Bits,
    ) -> usize {
        const {
            assert!(
                SCAN >= WordScan::LETTERS && SCAN <= WordScan::SPACES,
                "one of the four word scans; a role with no scan runs through the byte loop"
            );
        };
        debug_assert_eq!(bit.count_ones(), 1, "one role's bit");
        if self.scans[bit.trailing_zeros() as usize % Bits::BITS as usize] != SCAN {
            return self.run_end_in(text, pos, bit, stop);
        }
        self.words_end::<SCAN>(text, pos, bit, stop)
    }

    /// The word scan itself: the first byte through the table, then whole
    /// words of the set's members, a byte outside the set ending the run when
    /// it is ASCII; a `stop` bit ends it as the byte loops do.
    #[inline(always)]
    fn words_end<const SCAN: u8>(
        &self,
        text: &str,
        mut pos: usize,
        bit: Bits,
        stop: Bits,
    ) -> usize {
        let bytes = text.as_bytes();
        loop {
            let Some(&lead) = bytes.get(pos) else {
                return pos;
            };
            if lead >= 0x80 {
                let (bits, len) = self.beyond(bytes, pos, lead);
                if bits & bit == 0 || bits & stop != 0 {
                    return pos;
                }
                pos += len;
                continue;
            }
            let bits = self.bytes[usize::from(lead)];
            if bits & bit == 0 || bits & stop != 0 {
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

    /// [`Self::run_end_as`] for a text that is all ASCII (the encoder reads
    /// that once per unit): there is no character beyond ASCII to decode
    /// and none for a run to end at, so the word test is the whole scan and
    /// a run costs the same few operations whatever its length up to a
    /// word. The letters' run goes straight to the word test: its caller
    /// has read the run's first letter, and a second mostly follows.
    /// Another role's run often ends at once and reads its first byte
    /// through the table, as the byte loop does, before the word. The first
    /// word is tested in line and ends nine runs in ten of English prose; a
    /// longer run, and the tail under eight bytes, go out of line
    /// ([`Self::run_end_ascii_rest`]): as a loop in here the scan cost every
    /// run the loop's setup, as many operations as the test itself
    /// (counted). A role whose ASCII members are not the set of `SCAN` runs
    /// byte by byte.
    #[inline(always)]
    pub(super) fn run_end_ascii<const SCAN: u8>(
        &self,
        text: &str,
        mut pos: usize,
        bit: Bits,
    ) -> usize {
        const {
            assert!(
                SCAN >= WordScan::LETTERS && SCAN <= WordScan::SPACES,
                "one of the four word scans; a role with no scan runs through the byte loop"
            );
        };
        debug_assert_eq!(bit.count_ones(), 1, "one role's bit");
        let bytes = text.as_bytes();
        if self.scans[bit.trailing_zeros() as usize % Bits::BITS as usize] != SCAN {
            return self.run_end_bytes(bytes, pos, bit);
        }
        if SCAN != WordScan::LETTERS {
            match bytes.get(pos) {
                Some(&lead) if self.bytes[usize::from(lead)] & bit != 0 => pos += 1,
                _ => return pos,
            }
        }
        let Some(word) = bytes.get(pos..).and_then(<[u8]>::first_chunk::<8>) else {
            return self.run_end_ascii_rest::<SCAN>(bytes, pos, bit);
        };
        let outside = WordScan::outside_ascii::<SCAN>(u64::from_le_bytes(*word));
        if outside != 0 {
            return pos + (outside.trailing_zeros() / 8) as usize;
        }
        self.run_end_ascii_rest::<SCAN>(bytes, pos + 8, bit)
    }

    /// The rest of a lean run ([`Self::run_end_ascii`]) from `pos`: whole
    /// words while every byte is in the set, then the tail under eight bytes
    /// byte by byte.
    #[cold]
    #[inline(never)]
    fn run_end_ascii_rest<const SCAN: u8>(&self, bytes: &[u8], mut pos: usize, bit: Bits) -> usize {
        while let Some(word) = bytes.get(pos..).and_then(<[u8]>::first_chunk::<8>) {
            let outside = WordScan::outside_ascii::<SCAN>(u64::from_le_bytes(*word));
            if outside != 0 {
                return pos + (outside.trailing_zeros() / 8) as usize;
            }
            pos += 8;
        }
        self.run_end_bytes(bytes, pos, bit)
    }

    /// [`Self::run_end`] on a text that is all ASCII: a character is its
    /// byte, and the byte table is the whole test.
    #[inline(always)]
    fn run_end_bytes(&self, bytes: &[u8], mut pos: usize, bit: Bits) -> usize {
        while let Some(&byte) = bytes.get(pos) {
            if self.bytes[usize::from(byte)] & bit == 0 {
                break;
            }
            pos += 1;
        }
        pos
    }

    /// [`Self::at`] by the walk's text: on a text that is all ASCII
    /// (`ASCII`) a character is its byte, and the byte table is the read.
    #[inline(always)]
    pub(super) fn at_by<const ASCII: bool>(&self, text: &str, pos: usize) -> Option<(Bits, usize)> {
        if ASCII {
            let byte = *text.as_bytes().get(pos)?;
            Some((self.bytes[usize::from(byte)], 1))
        } else {
            self.at(text, pos)
        }
    }

    /// [`Self::run_end`] by the walk's text: on a text that is all ASCII the
    /// byte table is the whole test.
    #[inline(always)]
    pub(super) fn run_end_of<const ASCII: bool>(&self, text: &str, pos: usize, bit: Bits) -> usize {
        if ASCII {
            self.run_end_bytes(text.as_bytes(), pos, bit)
        } else {
            self.run_end(text, pos, bit)
        }
    }

    /// A run's end by the scans a walk takes: [`Self::run_end_ascii`] when
    /// its text is all ASCII (`ASCII`), else [`Self::run_end_as`].
    #[inline(always)]
    pub(super) fn run_end_by<const SCAN: u8, const WORDS: bool, const ASCII: bool>(
        &self,
        text: &str,
        pos: usize,
        bit: Bits,
    ) -> usize {
        if ASCII {
            self.run_end_ascii::<SCAN>(text, pos, bit)
        } else {
            self.run_end_as::<SCAN, WORDS>(text, pos, bit)
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

    /// [`Self::run_end`] with `stop` bits: a character holding one of them
    /// ends the run as the end of the text would (the chain scanner's
    /// stretches end at the next character of a stage class). The stop test
    /// comes first: with the role's bit tested first the compiler emits one
    /// instruction more per character (measured on the chain shape's short
    /// inputs), and this loop is the chain's byte loop for every run.
    #[inline(always)]
    pub(super) fn run_end_in(&self, text: &str, mut pos: usize, bit: Bits, stop: Bits) -> usize {
        while let Some((bits, len)) = self.at(text, pos).filter(|(bits, _)| bits & stop == 0) {
            if bits & bit == 0 {
                break;
            }
            pos += len;
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
        // a range set, whitespace, a single character, a category with an
        // exclusion.
        let pattern = Pattern::parse(concat!(
            r"\p{L}|[^\r\n\p{L}\p{N}]|[一-龥぀-ゟ゠-ヿ]|\s|\p{N}|'",
            r"|[\p{L}&&[^\p{Lu}]]",
        ))
        .expect("parses");
        let classes: Vec<&Class> = pattern.classes().iter().collect();
        let roles = Roles::build(&classes);
        let mut text = String::new();
        // Above the plane the table is runs: both ends of every run are read
        // beside the sample, and every code point of the block at U+10400,
        // where only the exclusion cuts a run of the last class (capitals
        // out, small letters in).
        let mut changes = vec![PLANE];
        for class in &classes {
            class.changes_from(PLANE, &mut changes);
        }
        let ends = changes.iter().flat_map(|&code| code - 1..=code);
        let cuts = 0x1_0400..=0x1_044F;
        // The plane whole, where a character's bytes name its block, and a
        // sample above it.
        let sample = (0..0x11_0000u32).filter(|code| *code < PLANE || code % 7 == 0);
        for code in sample.chain(ends).chain(cuts) {
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

    /// The lean scan of an all-ASCII text ends every run where the byte loop
    /// ends it: every role at every offset of texts made of ASCII pieces of
    /// every kind, so a run that ends at once, one inside a word, one over
    /// two words and one into the tail under eight bytes after a word are
    /// each reached hundreds of times (counted from the byte loop's answer);
    /// and a scan named for a role whose members are another set is the
    /// byte loop.
    #[test]
    fn the_lean_scan_of_an_ascii_text_ends_runs_as_the_byte_loop_does() {
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
            "abcdefgh",
            "ABCDEFGHIJK",
            "lowercase",
            "        ",
            "\t\t\t\t\t",
        ];
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
            for pos in 0..=text.len() {
                let by_bytes = |bit: Bits| roles.run_end(&text, pos, bit);
                for bit in [1, 2, 4] {
                    let run = by_bytes(bit) - pos;
                    reached[0] += usize::from(run == 0);
                    reached[1] += usize::from((1..8).contains(&run));
                    reached[2] += usize::from(run >= 17);
                    reached[3] += usize::from(
                        run >= 9 && pos + run == text.len() && !(run - 1).is_multiple_of(8),
                    );
                }
                assert_eq!(
                    roles.run_end_ascii::<{ WordScan::LETTERS }>(&text, pos, 1),
                    by_bytes(1),
                    "{text:?} at {pos} letters"
                );
                assert_eq!(
                    roles.run_end_ascii::<{ WordScan::SPACES }>(&text, pos, 2),
                    by_bytes(2),
                    "{text:?} at {pos} spaces"
                );
                assert_eq!(
                    roles.run_end_ascii::<{ WordScan::LOWER }>(&text, pos, 4),
                    by_bytes(4),
                    "{text:?} at {pos} lower"
                );
                // The line breaks are no scan's set: the byte loop under any name.
                assert_eq!(
                    roles.run_end_ascii::<{ WordScan::SPACES }>(&text, pos, 8),
                    by_bytes(8),
                    "{text:?} at {pos} breaks"
                );
            }
        }
        assert!(
            reached.iter().all(|&times| times >= 100),
            "a run at once, one inside a word, one over two words, a tail after a word: {reached:?}"
        );
    }
}
