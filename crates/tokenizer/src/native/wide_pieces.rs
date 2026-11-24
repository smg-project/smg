//! The thread's table of the pieces of 16 to 32 bytes whose ids fit an
//! entry: one cache line per piece (the piece as its first and its last
//! sixteen bytes, its length and up to seven ids, all inline) behind one
//! tag byte per slot, probed linearly from the slot the key hashes to. A
//! hit reads a tag and one line, both at addresses the hash gives; a first
//! sight writes them, and nothing is allocated per entry. A longer piece,
//! and a piece of more ids, is the map's (`PieceCache::short`).

/// The longest piece the table keys: two sixteen-byte words cover it.
const WIDE_PIECE_MAX_BYTES: usize = 32;
/// The ids an entry holds: what a line has left beside the key.
const WIDE_IDS: usize = 7;
/// The slots of a table's first allocation, 4 KiB of entries.
const MIN_SLOTS: usize = 64;
/// The slot count up to which the table grows by four; by two above.
const GROW_BY_FOUR_UP_TO: usize = 4_096;
const EMPTY_TAG: u8 = 0;
const LOW_SALT: u64 = 0x2545_F491_4F6C_DD1D;
const HIGH_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// A piece of 16 to 32 bytes: its first sixteen bytes, its last sixteen
/// (the two overlap under 32 bytes) and its length. Every byte of the piece
/// lies in one of the two words, so two keys are equal exactly when the
/// pieces are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct WideKey {
    first: u128,
    last: u128,
    len: u8,
}

impl WideKey {
    /// The key of `piece`; `None` for a piece of fewer than sixteen bytes
    /// (the packed table's) or of more than the table keys.
    pub(super) fn of(piece: &[u8]) -> Option<Self> {
        if piece.len() > WIDE_PIECE_MAX_BYTES {
            return None;
        }
        let first = u128::from_le_bytes(*piece.first_chunk()?);
        let last = u128::from_le_bytes(*piece.last_chunk()?);
        Some(Self {
            first,
            last,
            len: piece.len() as u8,
        })
    }

    /// The key's hash: its four words folded two by two by multiplication,
    /// the length with the last. The low bits pick the slot, the high bits
    /// make the tag.
    fn hash(self) -> u64 {
        let fold = |a: u64, b: u64| {
            let product = u128::from(a) * u128::from(b);
            (product as u64) ^ ((product >> 64) as u64)
        };
        let front = fold(
            self.first as u64 ^ LOW_SALT,
            (self.first >> 64) as u64 ^ HIGH_SALT,
        );
        let back = fold(
            self.last as u64 ^ HIGH_SALT,
            (self.last >> 64) as u64 ^ LOW_SALT ^ u64::from(self.len),
        );
        fold(front ^ LOW_SALT, back ^ HIGH_SALT)
    }
}

/// One slot's piece and ids: 64 bytes, a cache line.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Entry {
    first: u128,
    last: u128,
    ids: [u32; WIDE_IDS],
    len: u8,
    count: u8,
}

const _: () = assert!(size_of::<Entry>() == 64);

impl Entry {
    const EMPTY: Self = Self {
        first: 0,
        last: 0,
        ids: [0; WIDE_IDS],
        len: 0,
        count: 0,
    };

    fn key(&self) -> WideKey {
        WideKey {
            first: self.first,
            last: self.last,
            len: self.len,
        }
    }
}

/// The table. It has no slots until its first piece, so a thread that
/// meets no such piece holds nothing for it.
pub(super) struct WidePieceTable {
    /// One byte per slot: empty, or seven bits of the key's hash under a
    /// set top bit.
    tags: Vec<u8>,
    entries: Vec<Entry>,
    /// Entries stored.
    len: usize,
    /// One less than the slot count, a power of two; zero without slots.
    mask: usize,
}

impl WidePieceTable {
    pub(super) fn new() -> Self {
        Self {
            tags: Vec::new(),
            entries: Vec::new(),
            len: 0,
            mask: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn slots(&self) -> usize {
        self.tags.len()
    }

    /// The slots a table of `entries` pieces grows to at the most: a power
    /// of two that holds them under the fill bound, as the packed table's.
    pub(super) fn slots_for(entries: usize) -> usize {
        (entries * 4 / 3 + 1).next_power_of_two().max(MIN_SLOTS)
    }

    /// The slot `hash` probes first and its tag.
    fn first_slot(&self, hash: u64) -> (usize, u8) {
        (hash as usize & self.mask, (hash >> 57) as u8 | 0x80)
    }

    /// Appends the ids of `key` to `out` and answers whether the table had
    /// the piece. All seven ids of the entry are copied and the length is
    /// set back to the count: a copy of a fixed size, not of the count.
    pub(super) fn ids_into(&self, key: WideKey, out: &mut Vec<u32>) -> bool {
        if self.len == 0 {
            return false;
        }
        let (mut slot, tag) = self.first_slot(key.hash());
        loop {
            let found = self.tags[slot];
            if found == tag {
                let entry = &self.entries[slot];
                if entry.key() == key {
                    out.extend_from_slice(&entry.ids);
                    out.truncate(out.len() - (WIDE_IDS - usize::from(entry.count)));
                    return true;
                }
            } else if found == EMPTY_TAG {
                return false;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Stores `ids` for `key` and answers whether the table holds the piece
    /// now. It refuses no ids and more ids than an entry holds; a piece
    /// already stored keeps its ids. A table at its fill bound, three
    /// quarters of its slots, grows first, up to the slots `bound` entries
    /// take ([`Self::slots_for`]), so a probe always ends on an empty slot
    /// and the table never has more slots than its thread's share allows;
    /// at those slots it refuses the piece, which is the map's then.
    pub(super) fn insert(&mut self, key: WideKey, ids: &[u32], bound: usize) -> bool {
        if ids.is_empty() || ids.len() > WIDE_IDS {
            return false;
        }
        if self.len >= self.slots() / 4 * 3 {
            let most = Self::slots_for(bound);
            if self.slots() >= most {
                return false;
            }
            self.grow(most);
        }
        let (mut slot, tag) = self.first_slot(key.hash());
        while self.tags[slot] != EMPTY_TAG {
            if self.tags[slot] == tag && self.entries[slot].key() == key {
                return true;
            }
            slot = (slot + 1) & self.mask;
        }
        let mut entry = Entry {
            first: key.first,
            last: key.last,
            ids: [0; WIDE_IDS],
            len: key.len,
            count: ids.len() as u8,
        };
        entry.ids[..ids.len()].copy_from_slice(ids);
        self.tags[slot] = tag;
        self.entries[slot] = entry;
        self.len += 1;
        true
    }

    /// Grows the slots (the first ones, then by four up to
    /// `GROW_BY_FOUR_UP_TO`, by two above, to `most` at the most) into new
    /// arrays, every entry placed again by its key. A tag does not depend
    /// on the slot count.
    fn grow(&mut self, most: usize) {
        let slots = match self.slots() {
            0 => MIN_SLOTS,
            old if old < GROW_BY_FOUR_UP_TO => old * 4,
            old => old * 2,
        }
        .min(most);
        let tags = std::mem::replace(&mut self.tags, vec![EMPTY_TAG; slots]);
        let entries = std::mem::replace(&mut self.entries, vec![Entry::EMPTY; slots]);
        self.mask = slots - 1;
        for (tag, entry) in tags.into_iter().zip(entries) {
            if tag == EMPTY_TAG {
                continue;
            }
            let (mut slot, _) = self.first_slot(entry.key().hash());
            while self.tags[slot] != EMPTY_TAG {
                slot = (slot + 1) & self.mask;
            }
            self.tags[slot] = tag;
            self.entries[slot] = entry;
        }
    }

    /// Forgets every piece. The slots stay (only the tags are cleared),
    /// unless the table has more of them than `bound` entries take, a bound
    /// that fell since it grew: then they are given back and the table
    /// grows again from none.
    pub(super) fn clear(&mut self, bound: usize) {
        if self.slots() > Self::slots_for(bound) {
            *self = Self::new();
        } else {
            self.tags.fill(EMPTY_TAG);
            self.len = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bound no table of these tests reaches.
    const ROOMY: usize = 1 << 20;

    /// A piece of `len` bytes, distinct for distinct `i`.
    fn piece(len: usize, i: usize) -> Vec<u8> {
        let mut piece = format!("{i:08}").into_bytes();
        piece.resize(len, b'x');
        piece
    }

    fn key(piece: &[u8]) -> WideKey {
        WideKey::of(piece).expect("a piece of 16 to 32 bytes")
    }

    fn ids_of(table: &WidePieceTable, piece: &[u8]) -> Option<Vec<u32>> {
        let mut ids = Vec::new();
        table.ids_into(key(piece), &mut ids).then_some(ids)
    }

    #[test]
    fn a_piece_outside_sixteen_to_thirty_two_bytes_has_no_key() {
        assert!(WideKey::of(&[]).is_none());
        assert!(WideKey::of(&[b'a'; 15]).is_none());
        assert!(WideKey::of(&[b'a'; 16]).is_some());
        assert!(WideKey::of(&[b'a'; 32]).is_some());
        assert!(WideKey::of(&[b'a'; 33]).is_none());
    }

    #[test]
    fn keys_tell_pieces_apart_by_every_byte_and_by_length() {
        for len in 16..=WIDE_PIECE_MAX_BYTES {
            let piece = vec![b'a'; len];
            assert_eq!(key(&piece), key(&vec![b'a'; len]));
            for at in 0..len {
                let mut other = piece.clone();
                other[at] = b'b';
                assert_ne!(key(&piece), key(&other), "byte {at} of {len}");
            }
            if len < WIDE_PIECE_MAX_BYTES {
                assert_ne!(key(&piece), key(&vec![b'a'; len + 1]), "{len}");
            }
        }
    }

    #[test]
    fn a_table_starts_without_slots_and_answers_what_it_was_given() {
        let mut table = WidePieceTable::new();
        assert_eq!((table.len(), table.slots()), (0, 0));
        assert_eq!(ids_of(&table, &piece(20, 1)), None);
        for count in 1..=WIDE_IDS {
            let ids: Vec<u32> = (0..count as u32)
                .map(|id| 100 * count as u32 + id)
                .collect();
            assert!(table.insert(key(&piece(15 + count, count)), &ids, ROOMY));
            assert_eq!(ids_of(&table, &piece(15 + count, count)), Some(ids));
        }
        assert_eq!((table.len(), table.slots()), (WIDE_IDS, MIN_SLOTS));
        assert_eq!(ids_of(&table, &piece(20, 9)), None);
    }

    #[test]
    fn ids_are_appended_after_what_the_buffer_held() {
        let mut table = WidePieceTable::new();
        assert!(table.insert(key(&piece(24, 1)), &[7, 8, 9], ROOMY));
        let mut ids = vec![1, 2];
        assert!(table.ids_into(key(&piece(24, 1)), &mut ids));
        assert_eq!(ids, [1, 2, 7, 8, 9]);
    }

    #[test]
    fn no_ids_and_more_than_an_entry_holds_are_refused() {
        let mut table = WidePieceTable::new();
        assert!(!table.insert(key(&piece(20, 1)), &[], ROOMY));
        assert!(!table.insert(key(&piece(20, 1)), &[1; WIDE_IDS + 1], ROOMY));
        assert_eq!((table.len(), table.slots()), (0, 0));
        assert_eq!(ids_of(&table, &piece(20, 1)), None);
    }

    #[test]
    fn a_stored_piece_keeps_its_ids_on_a_second_insert() {
        let mut table = WidePieceTable::new();
        assert!(table.insert(key(&piece(20, 1)), &[1, 2], ROOMY));
        assert!(table.insert(key(&piece(20, 1)), &[3], ROOMY));
        assert_eq!(table.len(), 1);
        assert_eq!(ids_of(&table, &piece(20, 1)), Some(vec![1, 2]));
    }

    #[test]
    fn a_table_grows_by_four_then_doubles_and_keeps_its_pieces() {
        let mut table = WidePieceTable::new();
        let mut steps = vec![];
        for i in 0..10_000 {
            let slots = table.slots();
            assert!(table.insert(key(&piece(16 + i % 17, i)), &[i as u32, 1], ROOMY));
            if table.slots() != slots {
                steps.push(table.slots());
            }
            assert!(table.len() <= table.slots() / 4 * 3, "{i}");
        }
        assert_eq!(steps, [64, 256, 1_024, 4_096, 8_192, 16_384]);
        assert_eq!(table.len(), 10_000);
        for i in 0..10_000 {
            assert_eq!(
                ids_of(&table, &piece(16 + i % 17, i)),
                Some(vec![i as u32, 1]),
                "{i}"
            );
        }
    }

    #[test]
    fn a_table_grows_to_the_slots_of_its_bound_and_refuses_past_them() {
        let mut table = WidePieceTable::new();
        let bound = 1_000;
        let most = WidePieceTable::slots_for(bound);
        assert_eq!(most, 2_048);
        let (mut steps, mut kept) = (vec![], 0);
        for i in 0..2_000 {
            let slots = table.slots();
            kept += usize::from(table.insert(key(&piece(20, i)), &[i as u32], bound));
            if table.slots() != slots {
                steps.push(table.slots());
            }
        }
        // By four while small, then to the bound's slots and no further:
        // the step by four from 1,024 slots would pass them.
        assert_eq!(steps, [64, 256, 1_024, most]);
        assert_eq!((kept, table.len()), (most / 4 * 3, most / 4 * 3));
        // The bound's own clear finds nothing to give back.
        table.clear(bound);
        assert_eq!((table.len(), table.slots()), (0, most));
    }

    #[test]
    fn a_clear_forgets_and_keeps_the_slots_until_the_bound_falls_under_them() {
        let mut table = WidePieceTable::new();
        for i in 0..1_000 {
            assert!(table.insert(key(&piece(20, i)), &[i as u32], ROOMY));
        }
        let slots = table.slots();
        assert_eq!(slots, 4_096);
        // A bound the slots serve: they stay, empty.
        table.clear(3_000);
        assert_eq!((table.len(), table.slots()), (0, slots));
        assert_eq!(ids_of(&table, &piece(20, 1)), None);
        assert!(table.insert(key(&piece(20, 1)), &[5], ROOMY));
        assert_eq!(ids_of(&table, &piece(20, 1)), Some(vec![5]));
        // A bound that takes fewer slots: they are given back.
        table.clear(1_000);
        assert_eq!((table.len(), table.slots()), (0, 0));
        assert_eq!(ids_of(&table, &piece(20, 1)), None);
        assert!(table.insert(key(&piece(20, 2)), &[6], ROOMY));
        assert_eq!((table.len(), table.slots()), (1, MIN_SLOTS));
    }
}
