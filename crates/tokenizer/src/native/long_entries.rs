//! A filter over the vocabulary's entries longer than the seed's keys.
//!
//! Under `ignore_merges` a piece that is an entry is that entry, so a piece
//! neither the thread's cache nor the seed holds is looked up whole in the
//! model's vocabulary before it is merged (see
//! `NativeEncoder::tokenize_piece`): the piece mapped to the byte-level
//! alphabet as a string, hashed, probed in a map of the whole vocabulary and
//! compared, a memory latency and more on a cold piece. The seed spares that
//! for a short piece, as it holds every short entry. A long piece, more than
//! `SHORT_PIECE_MAX_BYTES` bytes, is rarely an entry (one to five entries in
//! a hundred are that long in the vocabularies in use) and this filter
//! answers "no entry" for nearly every such piece from one word: two bits of
//! a word picked by the piece's length and its first and last eight bytes,
//! set at load for every long entry. A piece with either bit clear was never
//! an entry; one with both set is looked up as before, so a false positive
//! costs what every lookup cost (about one piece in twenty at eight bits per
//! entry; the rounding of the word count to a power of two gives up to
//! sixteen bits and a rate down to two in a hundred).

use super::short_pieces::SHORT_PIECE_MAX_BYTES;

/// Filter bits per long entry, at least.
const BITS_PER_ENTRY: usize = 8;
/// The least words of a filter: 512 bytes for a vocabulary with few long
/// entries.
const MIN_WORDS: usize = 64;
/// The filter's hash multiplier.
const MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// The filter; see the module documentation.
pub(super) struct LongEntryFilter {
    words: Box<[u64]>,
    /// Bits a piece's hash is shifted right by to give its word; the filter
    /// holds `1 << (64 - shift)` words.
    shift: u32,
}

impl LongEntryFilter {
    /// A filter sized for `entries` long entries.
    pub(super) fn for_entries(entries: usize) -> Self {
        let words = (entries * BITS_PER_ENTRY)
            .div_ceil(u64::BITS as usize)
            .next_power_of_two()
            .max(MIN_WORDS);
        Self {
            words: vec![0; words].into_boxed_slice(),
            shift: u64::BITS - words.trailing_zeros(),
        }
    }

    /// The word of `piece` and its two bits, from its length and its first
    /// and last eight bytes; `None` for a piece the seed's keys cover, which
    /// the filter never answers for.
    fn slot(&self, piece: &[u8]) -> Option<(usize, u64)> {
        if piece.len() <= SHORT_PIECE_MAX_BYTES {
            return None;
        }
        let (head, _) = piece.split_first_chunk::<8>()?;
        let (_, tail) = piece.split_last_chunk::<8>()?;
        let head = u64::from_le_bytes(*head);
        let tail = u64::from_le_bytes(*tail);
        let hash = (head ^ tail.rotate_left(32) ^ piece.len() as u64).wrapping_mul(MULTIPLIER);
        // The word from the hash's high bits, the two bits from its middle
        // ones folded down (the product's low bits know only the low bytes).
        let word = (hash >> self.shift) as usize;
        let folded = hash ^ (hash >> 32);
        let bits = (1u64 << (folded & 63)) | (1u64 << ((folded >> 6) & 63));
        Some((word, bits))
    }

    /// Records `piece` as an entry.
    pub(super) fn insert(&mut self, piece: &[u8]) {
        if let Some((word, bits)) = self.slot(piece) {
            self.words[word] |= bits;
        }
    }

    /// Whether `piece` may be an entry: true for every piece inserted and
    /// for a piece the seed's keys cover (the seed answers for those);
    /// rarely for any other.
    pub(super) fn may_hold(&self, piece: &[u8]) -> bool {
        match self.slot(piece) {
            Some((word, bits)) => (self.words[word] & bits) == bits,
            None => true,
        }
    }

    /// Bytes the filter holds, for the load log.
    pub(super) fn bytes(&self) -> usize {
        size_of_val(&*self.words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(n: usize) -> Vec<Vec<u8>> {
        // Distinct pieces of 16 to 40 bytes from a small alphabet, the way
        // long entries look: a word with its space, a run of a script.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let len = 16 + (state % 25) as usize;
                (0..len)
                    .map(|i| b'a' + ((state >> (i % 8 * 8)) as u8).wrapping_add(i as u8) % 26)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn every_inserted_piece_is_held() {
        let entries = pieces(3_000);
        let mut filter = LongEntryFilter::for_entries(entries.len());
        for piece in &entries {
            filter.insert(piece);
        }
        assert!(entries.iter().all(|piece| filter.may_hold(piece)));
    }

    #[test]
    fn pieces_never_inserted_are_rarely_held_and_short_ones_always() {
        let entries = pieces(3_000);
        let mut filter = LongEntryFilter::for_entries(entries.len());
        for piece in &entries {
            filter.insert(piece);
        }
        let others: Vec<Vec<u8>> = pieces(13_000).into_iter().skip(3_000).collect();
        let held = others.iter().filter(|piece| filter.may_hold(piece)).count();
        assert!(
            held * 10 < others.len(),
            "{held} of {} other pieces held",
            others.len()
        );
        assert!(filter.may_hold(b"a short piece"));
        let none = LongEntryFilter::for_entries(0);
        assert!(!none.may_hold(b"a piece of sixteen bytes or more"));
        assert_eq!(none.bytes(), MIN_WORDS * 8);
    }
}
