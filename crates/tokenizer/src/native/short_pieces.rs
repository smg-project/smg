//! A packed table from short pieces to their ids.
//!
//! The native encoder answers a piece from a cache before it runs the
//! model's merges. A hash map from boxed bytes to boxed ids pays for a hit
//! with a hash over the bytes, a probe, a pointer chase to the key for the
//! comparison and a second one to the ids, and for a miss with two heap
//! objects. Nearly every piece is short, so this table keys a piece of up to
//! [`SHORT_PIECE_MAX_BYTES`] bytes as one `u128` (the bytes and the length:
//! the comparison is one integer compare) and keeps up to four ids in the
//! same 32-byte entry, two entries to a cache line and none across one, with
//! linear probing over a power-of-two slot count. A byte per slot holds a tag
//! of the key's hash (0 for an empty slot), so a probe walks a dense line of
//! tags and reads an entry only where the tag agrees: a miss touches one
//! line of tags where the entries would be a line each, and a clear is a
//! memset of the tags alone. The encoder builds one table at load from the
//! vocabulary, read by a thread for a piece it has not met; each thread keeps
//! the pieces it met in a table of its own (see `native`), which starts at
//! [`MIN_SLOTS`] and grows as the thread meets pieces (by four up to
//! `GROW_BY_FOUR_UP_TO` slots, by two above), up to the slots for the
//! thread's share, so a thread that encodes little holds little. A thread's
//! table grows in place: its slots are reserved up to
//! [`THREAD_BACKING_SLOTS`] when it is made (reserved, not written: a page
//! of the reservation is touched when a growth step extends the table over
//! it), a growth step extends the table into the reservation and places the
//! stored entries again where they stand, and only a table that outgrows
//! the reservation moves to a new allocation; so a thread that encodes
//! little touches little, and a thread that encodes much pays the pages of
//! one table, not of every table it grew through.

/// Longest piece the table keys: its bytes and length fit a `u128`.
pub(super) const SHORT_PIECE_MAX_BYTES: usize = 15;
/// Ids an entry keeps inline; an encoding with more goes to the spill.
const INLINE_IDS: usize = 4;
/// Bit of a key where the piece's length begins; the bytes are below it.
const LENGTH_SHIFT: u32 = 120;
/// Bit of a stored key where the id count begins; a probe key has zeros
/// there.
const COUNT_SHIFT: u32 = 124;
/// The bytes and the length of a stored key, without its count.
const KEY_MASK: u128 = (1u128 << COUNT_SHIFT) - 1;
/// Most ids an entry stands for: the count has four bits, and a piece of
/// `n` bytes has at most `n` ids on a byte-level vocabulary.
const MAX_IDS: usize = 15;
/// Fewest slots a table has, whatever it was sized for; a thread's table
/// starts here and grows.
pub(super) const MIN_SLOTS: usize = 64;
/// Slots reserved for a thread's table when it is made: the size a thread
/// reaches over its first few thousand pieces. The reservation is not
/// written until a growth step extends the table into it, so a thread of
/// few pieces touches the pages of its small table alone, and a thread that
/// grows to the reservation touches its pages once, where growth through
/// new tables touched every table's (measured: the tables grown through
/// were 97 % of a cold call's page faults under threads).
pub(super) const THREAD_BACKING_SLOTS: usize = 16_384;

/// Slots up to which a table grows by four at a time (the entries placed
/// again over that growth are a fourth of what doublings place); above them
/// it doubles. Above 4,096 slots the sizes are the ones doubling gives;
/// up to it, a table that has just grown holds up to about 5.3 slots per
/// entry (four times the slots it had at its fill bound), at most 2,048
/// slots, about 66 KiB, more than doubling would, per thread.
const GROW_BY_FOUR_UP_TO: usize = 4_096;
/// The tag of an empty slot; a stored slot's tag has its high bit set.
const EMPTY_TAG: u8 = 0;
/// The two halves of a key are salted before they are multiplied, so a
/// short piece (its high half is its length alone) still spreads over the
/// slots.
const LOW_SALT: u64 = 0x2545_F491_4F6C_DD1D;
const HIGH_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
/// `KEEP_LOW[n]` keeps the low `n` bytes of a sixteen-byte word: the mask
/// of a piece of `n` bytes read as one word of the text around it (see
/// [`ShortPieceKey::at`]).
const KEEP_LOW: [u128; 16] = {
    let mut masks = [0u128; 16];
    let mut n = 1;
    while n < 16 {
        masks[n] = (1u128 << (8 * n)) - 1;
        n += 1;
    }
    masks
};

/// A short piece the table does not hold: its key and where the probe that
/// missed ended (the empty slot it stopped on, the key's tag and the table's
/// mask at the time; a table grown or made anew since has other slots). The
/// insert that follows the miss stores there without hashing and probing
/// again (see [`ShortPieceTable::insert_at`]).
#[derive(Clone, Copy)]
pub(super) struct Vacancy {
    key: ShortPieceKey,
    slot: u32,
    mask: u32,
    tag: u8,
}

impl Vacancy {
    /// The key of the piece the probe missed.
    pub(super) fn key(&self) -> ShortPieceKey {
        self.key
    }
}

/// A piece of 1..=15 bytes as one integer: the bytes little-endian in the
/// low 120 bits, the length above them, so two keys are equal exactly when
/// the pieces are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ShortPieceKey(u128);

impl ShortPieceKey {
    /// The key of `piece`; `None` when it is empty or longer than the table
    /// keys. The bytes are read as two overlapping words (eight and eight,
    /// four and four, or byte by byte under four) rather than copied: a
    /// `memcpy` of a length known only at run time is a call, and this runs
    /// per piece.
    #[inline(always)]
    pub(super) fn of(piece: &[u8]) -> Option<Self> {
        let len = piece.len();
        if len == 0 || len > SHORT_PIECE_MAX_BYTES {
            return None;
        }
        let bytes = if len >= 8 {
            let head = u64::from_le_bytes(*piece.split_first_chunk::<8>()?.0);
            let tail = u64::from_le_bytes(*piece.split_last_chunk::<8>()?.1);
            // The tail word ends at `len`; its bytes from offset 8 on are the
            // high half, the ones before 8 are shifted out.
            u128::from(head) | ((u128::from(tail) >> (8 * (16 - len))) << 64)
        } else if len >= 4 {
            let head = u32::from_le_bytes(*piece.split_first_chunk::<4>()?.0);
            let tail = u32::from_le_bytes(*piece.split_last_chunk::<4>()?.1);
            u128::from(u64::from(head) | ((u64::from(tail) >> (8 * (8 - len))) << 32))
        } else {
            piece
                .iter()
                .enumerate()
                .fold(0u128, |bytes, (i, &b)| bytes | (u128::from(b) << (8 * i)))
        };
        Some(Self(bytes | ((len as u128) << LENGTH_SHIFT)))
    }

    /// The key of the piece `text[start..end]`, `None` as for [`Self::of`].
    /// When sixteen bytes of the text remain past `start`, the piece is read
    /// as one sixteen-byte word masked to its length: the length picks a
    /// mask from a table instead of taking a branch and a shift, so the read
    /// costs the same for every length (the length of the next piece is the
    /// one thing the branch predictor cannot learn from the text; measured
    /// on the warm long classes). Near the end of the text the piece is read
    /// on its own.
    #[inline(always)]
    pub(super) fn at(text: &[u8], start: usize, end: usize) -> Option<Self> {
        let len = end.checked_sub(start)?;
        if len == 0 || len > SHORT_PIECE_MAX_BYTES {
            return None;
        }
        match text.get(start..).and_then(|rest| rest.first_chunk::<16>()) {
            Some(window) => {
                let word = u128::from_le_bytes(*window);
                Some(Self(
                    (word & KEEP_LOW[len]) | ((len as u128) << LENGTH_SHIFT),
                ))
            }
            None => Self::of(text.get(start..end)?),
        }
    }

    /// The length of the piece the key stands for.
    #[cfg(test)]
    fn len(self) -> usize {
        (self.0 >> LENGTH_SHIFT) as usize & 0xF
    }
}

/// One slot: a stored key with its id count in the top four bits, and the
/// ids inline, or the offset of the ids in the spill in `ids[0]` when there
/// are more than [`INLINE_IDS`]. Key 0 is an empty slot: a stored key has a
/// length. The ids past the count are zero; a hit copies all four and cuts
/// the length back to the count, whatever they hold.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Entry {
    key: u128,
    ids: [u32; INLINE_IDS],
}

impl Entry {
    const EMPTY: Self = Self {
        key: 0,
        ids: [0; INLINE_IDS],
    };
}

/// The table; see the module documentation.
pub(super) struct ShortPieceTable {
    /// One tag per slot: [`EMPTY_TAG`], or seven bits of the stored key's
    /// hash with the high bit set.
    tags: Vec<u8>,
    entries: Vec<Entry>,
    /// The ids of the entries with more than [`INLINE_IDS`], each run at the
    /// offset its entry holds.
    spill: Vec<u32>,
    /// Entries stored.
    len: usize,
    /// One less than the slot count, a power of two.
    mask: usize,
}

impl ShortPieceTable {
    /// A thread's table: [`MIN_SLOTS`] slots in a reservation of
    /// [`THREAD_BACKING_SLOTS`], grown in place (see [`Self::insert`]).
    pub(super) fn new() -> Self {
        Self::with_slots(MIN_SLOTS, THREAD_BACKING_SLOTS)
    }

    /// A table for about `entries` pieces: the slots that hold them under the
    /// fill bound, and no reservation beyond them.
    pub(super) fn for_entries(entries: usize) -> Self {
        let slots = Self::slots_for(entries);
        Self::with_slots(slots, slots)
    }

    /// The slots a table for about `entries` pieces has: a power of two that
    /// holds them under the fill bound, [`MIN_SLOTS`] at least.
    pub(super) fn slots_for(entries: usize) -> usize {
        (entries * 4 / 3 + 1).next_power_of_two().max(MIN_SLOTS)
    }

    /// A table of `slots` slots with room for `reserved` before it has to
    /// move; the room is capacity, not written.
    fn with_slots(slots: usize, reserved: usize) -> Self {
        let mut tags = Vec::with_capacity(reserved.max(slots));
        tags.resize(slots, EMPTY_TAG);
        let mut entries = Vec::with_capacity(reserved.max(slots));
        entries.resize(slots, Entry::EMPTY);
        Self {
            tags,
            entries,
            spill: Vec::new(),
            len: 0,
            mask: slots - 1,
        }
    }

    /// Slots the table can grow to before it moves.
    #[cfg(test)]
    fn backing(&self) -> usize {
        self.entries.capacity()
    }

    /// Entries stored.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Slots, stored or empty: the table's size is 32 bytes per slot.
    pub(super) fn slots(&self) -> usize {
        self.entries.len()
    }

    /// Entries the table takes before [`Self::insert`] refuses: three
    /// quarters of the slots, so a probe always ends on an empty slot.
    pub(super) fn fill_bound(&self) -> usize {
        self.entries.len() / 4 * 3
    }

    /// The slot a key probes first and its tag, from the folded product of
    /// the key's salted halves: the low bits pick the slot, the high bits
    /// make the tag.
    #[inline(always)]
    fn first_slot(&self, key: ShortPieceKey) -> (usize, u8) {
        let low = key.0 as u64 ^ LOW_SALT;
        let high = (key.0 >> 64) as u64 ^ HIGH_SALT;
        let product = u128::from(low) * u128::from(high);
        let hash = (product as u64) ^ ((product >> 64) as u64);
        (hash as usize & self.mask, (hash >> 57) as u8 | 0x80)
    }

    /// Appends the ids of `key` to `out` and answers whether the table had
    /// the piece.
    #[inline(always)]
    pub(super) fn ids_into(&self, key: ShortPieceKey, out: &mut Vec<u32>) -> bool {
        self.lookup(key, out).is_none()
    }

    /// The probe: a hit appends the piece's ids (all four inline ids, then
    /// the length set to the count; a spilled entry from the spill) and
    /// returns `None`; a miss returns where it ended, for the insert that
    /// follows it ([`Self::ids_into`] is this probe read as a hit test).
    #[inline(always)]
    pub(super) fn lookup(&self, key: ShortPieceKey, out: &mut Vec<u32>) -> Option<Vacancy> {
        let (mut slot, tag) = self.first_slot(key);
        loop {
            let found = self.tags[slot];
            if found == tag {
                let entry = &self.entries[slot];
                if entry.key & KEY_MASK == key.0 {
                    let count = (entry.key >> COUNT_SHIFT) as usize;
                    if count <= INLINE_IDS {
                        out.extend_from_slice(&entry.ids);
                        out.truncate(out.len() - (INLINE_IDS - count));
                    } else {
                        let start = entry.ids[0] as usize;
                        out.extend_from_slice(&self.spill[start..start + count]);
                    }
                    return None;
                }
            } else if found == EMPTY_TAG {
                return Some(Vacancy {
                    key,
                    slot: slot as u32,
                    mask: self.mask as u32,
                    tag,
                });
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Stores `ids` as the piece's encoding and answers true. A table at its
    /// fill bound with fewer slots than `bound` entries take grows first
    /// (by four while small, by two above `GROW_BY_FOUR_UP_TO` slots, up to
    /// the slots for `bound`; every stored entry placed again, the spill
    /// stays), so a table starts small and grows with use; at the bound it
    /// refuses, as it does ids that are not 1..=15 in number, and
    /// the piece is merged again on every encode. A piece already stored
    /// keeps its ids (they are the model's for that piece either way).
    pub(super) fn insert(&mut self, key: ShortPieceKey, ids: &[u32], bound: usize) -> bool {
        if ids.is_empty() || ids.len() > MAX_IDS {
            return false;
        }
        if self.len >= self.fill_bound() {
            if self.slots() >= Self::slots_for(bound) {
                return false;
            }
            self.grow(Self::slots_for(bound));
        }
        let (mut slot, tag) = self.first_slot(key);
        while self.tags[slot] != EMPTY_TAG {
            if self.tags[slot] == tag && self.entries[slot].key & KEY_MASK == key.0 {
                return true;
            }
            slot = (slot + 1) & self.mask;
        }
        self.store(slot, tag, key, ids);
        true
    }

    /// [`Self::insert`] of the piece a probe missed, at the vacancy the probe
    /// ended on: the miss has done the hashing and the walk. The vacancy must
    /// be of this table as it is, with no [`Self::clear`] between the
    /// [`Self::lookup`] that made it and this call (a cleared table keeps its
    /// slots and would take the entry off its probe path; a caller that
    /// cleared inserts by probing). A table grown since, or at its fill
    /// bound, or a slot taken since, sends the insert through the probe.
    pub(super) fn insert_at(&mut self, vacancy: Vacancy, ids: &[u32], bound: usize) -> bool {
        let slot = vacancy.slot as usize;
        if vacancy.mask as usize == self.mask
            && self.len < self.fill_bound()
            && self.tags[slot] == EMPTY_TAG
            && !ids.is_empty()
            && ids.len() <= MAX_IDS
        {
            debug_assert!(
                self.occupied_up_to(vacancy.key, slot),
                "a vacancy of the table as it is: no clear between the probe and the insert"
            );
            self.store(slot, vacancy.tag, vacancy.key, ids);
            return true;
        }
        self.insert(vacancy.key, ids, bound)
    }

    /// Whether every slot from `key`'s home slot up to `slot` is occupied:
    /// the condition under which an entry stored at `slot` is on the probe
    /// path of `key`.
    fn occupied_up_to(&self, key: ShortPieceKey, slot: usize) -> bool {
        let (mut at, _) = self.first_slot(key);
        while at != slot {
            if self.tags[at] == EMPTY_TAG {
                return false;
            }
            at = (at + 1) & self.mask;
        }
        true
    }

    /// Writes the entry of `key` at the empty `slot`: up to four ids in the
    /// entry (a fixed pattern per count, no copy call), more in the spill.
    #[inline(always)]
    fn store(&mut self, slot: usize, tag: u8, key: ShortPieceKey, ids: &[u32]) {
        let inline = match *ids {
            [a] => [a, 0, 0, 0],
            [a, b] => [a, b, 0, 0],
            [a, b, c] => [a, b, c, 0],
            [a, b, c, d] => [a, b, c, d],
            _ => {
                let at = self.spill.len() as u32;
                self.spill.extend_from_slice(ids);
                [at, 0, 0, 0]
            }
        };
        self.tags[slot] = tag;
        self.entries[slot] = Entry {
            key: key.0 | ((ids.len() as u128) << COUNT_SHIFT),
            ids: inline,
        };
        self.len += 1;
    }

    /// Grows the slots (by four up to `GROW_BY_FOUR_UP_TO`, by two above, to
    /// `most` at the most) in place: the table extends into its reservation
    /// (past the reservation it moves, as a `Vec` does) and every stored
    /// entry is placed again where it stands ([`Self::place_again`]). A tag
    /// does not depend on the slot count, so the tags move with their
    /// entries.
    fn grow(&mut self, most: usize) {
        let old = self.entries.len();
        let slots = if old < GROW_BY_FOUR_UP_TO {
            old * 4
        } else {
            old * 2
        }
        .min(most);
        self.tags.resize(slots, EMPTY_TAG);
        self.entries.resize(slots, Entry::EMPTY);
        self.mask = slots - 1;
        self.place_again(old);
    }

    /// Places the entries stored in the first `old` slots again under the
    /// current mask, in the table as it is: no second table, no copy of the
    /// whole. Every stored slot of the old range starts pending. A pending
    /// entry's probe from its new home walks over placed slots and stops at
    /// the first empty or pending one: its own slot means it is placed, an
    /// empty slot takes it (its old slot empties), a pending slot swaps
    /// entries with it (that slot is placed, this one is pending again with
    /// the other entry). Each step places one slot, so the pass ends. A
    /// placed entry's probe path holds placed slots only, which never
    /// change afterwards, and never crosses a pending slot (the probe stops
    /// there), so every path is whole once no slot is pending.
    fn place_again(&mut self, old: usize) {
        let mut pending = vec![0u64; old.div_ceil(64)];
        for slot in 0..old {
            if self.tags[slot] != EMPTY_TAG {
                pending[slot / 64] |= 1u64 << (slot % 64);
            }
        }
        let is_pending = |pending: &[u64], slot: usize| {
            slot < old && pending[slot / 64] & (1u64 << (slot % 64)) != 0
        };
        for slot in 0..old {
            while is_pending(&pending, slot) {
                let key = ShortPieceKey(self.entries[slot].key & KEY_MASK);
                let (mut at, _) = self.first_slot(key);
                while self.tags[at] != EMPTY_TAG && !is_pending(&pending, at) {
                    at = (at + 1) & self.mask;
                }
                if at == slot {
                    pending[slot / 64] &= !(1u64 << (slot % 64));
                } else if self.tags[at] == EMPTY_TAG {
                    self.tags[at] = self.tags[slot];
                    self.entries[at] = self.entries[slot];
                    self.tags[slot] = EMPTY_TAG;
                    pending[slot / 64] &= !(1u64 << (slot % 64));
                } else {
                    self.tags.swap(slot, at);
                    self.entries.swap(slot, at);
                    pending[at / 64] &= !(1u64 << (at % 64));
                }
            }
        }
    }

    /// Forgets every piece. The slots stay allocated, only the tags are
    /// cleared (an empty tag is an empty slot whatever its entry holds),
    /// unless the table has more slots than `bound` entries take: then it is
    /// cut back to the slots for `bound`, which fell since the table grew
    /// (inside a thread's reservation the room stays for the next growth; a
    /// table that had moved past it is made anew, so the large allocation
    /// goes).
    pub(super) fn clear(&mut self, bound: usize) {
        let slots = Self::slots_for(bound);
        if self.slots() > slots {
            if self.entries.capacity() > THREAD_BACKING_SLOTS.max(slots) {
                *self = Self::with_slots(slots, THREAD_BACKING_SLOTS);
                return;
            }
            self.tags.truncate(slots);
            self.entries.truncate(slots);
            self.mask = slots - 1;
        }
        self.tags.fill(EMPTY_TAG);
        self.spill.clear();
        self.len = 0;
    }

    /// Slots read per lookup of a stored piece, on average.
    #[cfg(test)]
    fn mean_probes(&self) -> f64 {
        let mut probes = 0usize;
        for (slot, &tag) in self.tags.iter().enumerate() {
            if tag == EMPTY_TAG {
                continue;
            }
            let (first, _) = self.first_slot(ShortPieceKey(self.entries[slot].key & KEY_MASK));
            probes += (slot.wrapping_sub(first) & self.mask) + 1;
        }
        probes as f64 / self.len.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(piece: &[u8]) -> ShortPieceKey {
        ShortPieceKey::of(piece).expect("a short piece")
    }

    fn ids_of(table: &ShortPieceTable, piece: &[u8]) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        table.ids_into(key(piece), &mut out).then_some(out)
    }

    #[test]
    fn an_empty_piece_and_one_past_the_bound_have_no_key() {
        assert_eq!(ShortPieceKey::of(b""), None);
        assert_eq!(ShortPieceKey::of(&[b'a'; 16]), None);
        assert_eq!(key(&[b'a'; 15]).len(), 15);
        assert_eq!(key(b"a").len(), 1);
    }

    #[test]
    fn keys_tell_pieces_apart_by_bytes_and_by_length() {
        assert_ne!(key(b"a"), key(b"b"));
        assert_ne!(key(b"a"), key(b"a\0"));
        assert_ne!(key(b"\0"), key(b"\0\0"));
        assert_eq!(key(b"hello"), key(b"hello"));
        assert_ne!(key(&[b'a'; 15]), key(&[b'a'; 14]));
    }

    /// The word reads of every length class give the bytes a plain copy
    /// gives, with the length above them.
    #[test]
    fn a_key_is_the_pieces_bytes_and_its_length_for_every_length() {
        let piece: Vec<u8> = (1..=15).map(|b| b * 17).collect();
        for len in 1..=15 {
            let mut copied = [0u8; 16];
            copied[..len].copy_from_slice(&piece[..len]);
            let expected = u128::from_le_bytes(copied) | ((len as u128) << LENGTH_SHIFT);
            assert_eq!(key(&piece[..len]).0, expected, "{len}");
            let last = &piece[15 - len..];
            let mut copied = [0u8; 16];
            copied[..len].copy_from_slice(last);
            let expected = u128::from_le_bytes(copied) | ((len as u128) << LENGTH_SHIFT);
            assert_eq!(key(last).0, expected, "{len} from the end");
        }
    }

    /// The key read from the text around a piece equals the key of the
    /// piece on its own, for every length at every distance from the
    /// text's end, and is `None` exactly where that is.
    #[test]
    fn a_key_read_from_the_text_equals_the_pieces_own_key() {
        let text: Vec<u8> = (0..40u8)
            .map(|b| b.wrapping_mul(37).wrapping_add(11))
            .collect();
        for start in 0..=text.len() {
            for len in 0..=17 {
                let end = (start + len).min(text.len());
                assert_eq!(
                    ShortPieceKey::at(&text, start, end),
                    ShortPieceKey::of(&text[start..end]),
                    "{start}+{len}"
                );
            }
        }
        assert_eq!(ShortPieceKey::at(&text, 5, 4), None);
    }

    #[test]
    fn a_table_starts_empty_and_answers_what_it_was_given() {
        let mut table = ShortPieceTable::new();
        assert_eq!(table.len(), 0);
        assert_eq!(table.slots(), MIN_SLOTS);
        assert_eq!(ids_of(&table, b"a"), None);
        assert!(table.insert(key(b"a"), &[7], 3));
        assert!(table.insert(key(&[b'z'; 15]), &[1, 2], 3));
        assert!(table.insert(key(&[0xFF; 15]), &[3, 4, 5, 6], 3));
        assert_eq!(table.len(), 3);
        assert_eq!(ids_of(&table, b"a"), Some(vec![7]));
        assert_eq!(ids_of(&table, &[b'z'; 15]), Some(vec![1, 2]));
        assert_eq!(ids_of(&table, &[0xFF; 15]), Some(vec![3, 4, 5, 6]));
        assert_eq!(ids_of(&table, b"b"), None);
        assert_eq!(ids_of(&table, &[b'z'; 14]), None);
        assert_eq!(ids_of(&table, b"a\0"), None);
    }

    #[test]
    fn five_ids_and_more_spill_and_read_back_whole() {
        let mut table = ShortPieceTable::for_entries(4);
        let fifteen: Vec<u32> = (100..115).collect();
        assert!(table.insert(key(&[0x80; 15]), &fifteen, 4));
        assert!(table.insert(key(b"abcde"), &[1, 2, 3, 4, 5], 4));
        assert!(table.insert(key(b"abcd"), &[1, 2, 3, 4], 4));
        assert_eq!(ids_of(&table, &[0x80; 15]), Some(fifteen));
        assert_eq!(ids_of(&table, b"abcde"), Some(vec![1, 2, 3, 4, 5]));
        assert_eq!(ids_of(&table, b"abcd"), Some(vec![1, 2, 3, 4]));
        assert_eq!(table.spill.len(), 20);
    }

    #[test]
    fn ids_are_appended_after_what_the_buffer_held() {
        let mut table = ShortPieceTable::for_entries(1);
        table.insert(key(b"x"), &[9, 8], 1);
        let mut out = vec![1];
        assert!(table.ids_into(key(b"x"), &mut out));
        assert_eq!(out, [1, 9, 8]);
    }

    #[test]
    fn no_ids_or_too_many_are_refused() {
        let mut table = ShortPieceTable::for_entries(1);
        assert!(!table.insert(key(b"x"), &[], 1));
        assert!(!table.insert(key(b"y"), &[0; 16], 1));
        assert!(table.insert(key(b"z"), &[0; 15], 1));
        assert_eq!(table.len(), 1);
    }

    /// A thread's table: the fewest slots at first, grown at the fill bound
    /// while `bound` entries take more slots (by four while small, by two
    /// above `GROW_BY_FOUR_UP_TO`), every piece kept across the growths (the
    /// spilled ones too), and at the slots for `bound` the fill bound refuses
    /// and a probe still ends.
    #[test]
    fn a_table_starts_small_grows_by_four_then_doubles_and_refuses_at_its_bound() {
        let mut table = ShortPieceTable::new();
        let bound = 1000;
        let fifteen: Vec<u32> = (100..115).collect();
        assert!(table.insert(key(&[0x80; 15]), &fifteen, bound));
        assert!(table.insert(key(b"abcde"), &[1, 2, 3, 4, 5], bound));
        let mut sizes = vec![table.slots()];
        let mut stored: usize = 2;
        while table.insert(key(&stored.to_le_bytes()), &[stored as u32], bound) {
            stored += 1;
            if table.slots() != *sizes.last().expect("one size") {
                sizes.push(table.slots());
            }
        }
        assert_eq!(sizes, [64, 256, 1024, 2048]);
        assert_eq!(table.slots(), ShortPieceTable::slots_for(bound));
        assert_eq!(stored, table.fill_bound());
        assert_eq!(table.len(), stored);
        assert!(table.fill_bound() >= bound);
        assert!(!table.insert(key(b"one more"), &[1], bound));
        assert_eq!(ids_of(&table, b"one more"), None);
        assert_eq!(ids_of(&table, &[0x80; 15]), Some(fifteen));
        assert_eq!(ids_of(&table, b"abcde"), Some(vec![1, 2, 3, 4, 5]));
        for i in 2..stored {
            assert_eq!(
                ids_of(&table, &i.to_le_bytes()),
                Some(vec![i as u32]),
                "{i}"
            );
        }
        // A larger bound lets it grow again.
        assert!(table.insert(key(b"one more"), &[1], bound * 2));
        assert_eq!(table.slots(), 4096);
        assert_eq!(ids_of(&table, b"one more"), Some(vec![1]));
    }

    #[test]
    fn a_miss_hands_its_vacancy_to_the_insert_and_the_piece_reads_back() {
        let mut table = ShortPieceTable::new();
        let mut out = Vec::new();
        let vacancy = table.lookup(key(b"piece"), &mut out).expect("a miss");
        assert!(out.is_empty());
        assert!(table.insert_at(vacancy, &[7, 8], 1000));
        assert_eq!(ids_of(&table, b"piece"), Some(vec![7, 8]));
        assert!(table.lookup(key(b"piece"), &mut out).is_none());
        assert_eq!(out, [7, 8]);
        // A slot taken since the probe (here by the same piece) sends the
        // insert through the probe, which finds the piece: no duplicate.
        let stale = table.lookup(key(b"other"), &mut out).expect("a miss");
        assert!(table.insert(key(b"other"), &[9], 1000));
        assert!(table.insert_at(stale, &[9], 1000));
        assert_eq!(ids_of(&table, b"other"), Some(vec![9]));
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn a_vacancy_taken_at_the_fill_bound_inserts_by_probing_and_the_table_grows() {
        let mut table = ShortPieceTable::new();
        let bound = 1000;
        let mut stored: usize = 0;
        while table.len() < table.fill_bound() {
            assert!(table.insert(key(&stored.to_le_bytes()), &[stored as u32], bound));
            stored += 1;
        }
        let slots = table.slots();
        let mut out = Vec::new();
        let vacancy = table
            .lookup(key(b"at the bound"), &mut out)
            .expect("a miss");
        assert!(table.insert_at(vacancy, &[1, 2, 3], bound));
        assert_eq!(table.slots(), 4 * slots);
        assert_eq!(ids_of(&table, b"at the bound"), Some(vec![1, 2, 3]));
        for i in 0..stored {
            assert_eq!(
                ids_of(&table, &i.to_le_bytes()),
                Some(vec![i as u32]),
                "{i}"
            );
        }
        assert_eq!(table.len(), stored + 1);
    }

    #[test]
    fn a_table_at_the_regime_boundary_doubles() {
        let mut table = ShortPieceTable::for_entries(3_000);
        assert_eq!(table.slots(), GROW_BY_FOUR_UP_TO);
        let bound = 100_000;
        let mut stored: usize = 0;
        while table.slots() == GROW_BY_FOUR_UP_TO {
            assert!(table.insert(key(&stored.to_le_bytes()), &[stored as u32], bound));
            stored += 1;
        }
        assert_eq!(table.slots(), 2 * GROW_BY_FOUR_UP_TO);
        assert_eq!(table.len(), stored);
        for i in 0..stored {
            assert_eq!(
                ids_of(&table, &i.to_le_bytes()),
                Some(vec![i as u32]),
                "{i}"
            );
        }
    }

    #[test]
    fn a_table_at_its_bound_from_the_start_refuses_at_the_fill_bound() {
        let mut table = ShortPieceTable::for_entries(0);
        let bound = table.fill_bound();
        assert!(bound < table.slots());
        for i in 0..bound {
            assert!(table.insert(key(&i.to_le_bytes()), &[i as u32], 0), "{i}");
        }
        assert!(!table.insert(key(b"one more"), &[1], 0));
        assert_eq!(table.len(), bound);
        assert_eq!(table.slots(), MIN_SLOTS);
        assert_eq!(ids_of(&table, b"one more"), None);
        for i in 0..bound {
            assert_eq!(ids_of(&table, &i.to_le_bytes()), Some(vec![i as u32]));
        }
    }

    #[test]
    fn a_stored_piece_keeps_its_ids_on_a_second_insert() {
        let mut table = ShortPieceTable::for_entries(2);
        assert!(table.insert(key(b"x"), &[1], 2));
        assert!(table.insert(key(b"x"), &[2], 2));
        assert_eq!(table.len(), 1);
        assert_eq!(ids_of(&table, b"x"), Some(vec![1]));
    }

    #[test]
    fn a_cleared_table_forgets_and_takes_pieces_again() {
        let mut table = ShortPieceTable::for_entries(8);
        table.insert(key(b"a"), &[1], 8);
        table.insert(key(b"abcdef"), &[1, 2, 3, 4, 5, 6], 8);
        table.clear(8);
        assert_eq!(table.len(), 0);
        assert_eq!(table.slots(), MIN_SLOTS);
        assert!(table.spill.is_empty());
        assert_eq!(ids_of(&table, b"a"), None);
        assert_eq!(ids_of(&table, b"abcdef"), None);
        assert!(table.insert(key(b"a"), &[2], 8));
        assert_eq!(ids_of(&table, b"a"), Some(vec![2]));
    }

    /// A table keeps its slots through a clear for a bound they fit, and is
    /// cut back to the slots for a bound that fell below them, inside the
    /// reservation it keeps.
    #[test]
    fn a_clear_for_a_smaller_bound_cuts_the_table_back() {
        let mut table = ShortPieceTable::new();
        for i in 0..1000u32 {
            assert!(table.insert(key(&i.to_le_bytes()), &[i], 1000));
        }
        assert_eq!(table.slots(), 2048);
        table.clear(1000);
        assert_eq!(table.slots(), 2048);
        assert_eq!(table.len(), 0);
        table.clear(1536);
        assert_eq!(table.slots(), 2048, "a bound the slots fit keeps them");
        table.clear(100);
        assert_eq!(table.slots(), ShortPieceTable::slots_for(100));
        assert_eq!(
            table.backing(),
            THREAD_BACKING_SLOTS,
            "the reservation stays"
        );
        assert_eq!(table.len(), 0);
        assert!(table.spill.is_empty());
        assert_eq!(ids_of(&table, &7u32.to_le_bytes()), None);
        assert!(table.insert(key(b"a"), &[2], 100));
        assert_eq!(ids_of(&table, b"a"), Some(vec![2]));
    }

    /// A thread's table grows through every step inside its reservation
    /// without moving, every stored piece reads back after every step, and
    /// the step past the reservation moves it.
    #[test]
    fn a_thread_table_grows_in_place_up_to_its_reservation_and_moves_past_it() {
        let mut table = ShortPieceTable::new();
        assert_eq!(table.backing(), THREAD_BACKING_SLOTS);
        let bound = 100_000;
        let mut stored = 0usize;
        let mut sizes = vec![table.slots()];
        while table.slots() <= THREAD_BACKING_SLOTS {
            let before = table.slots();
            assert!(table.insert(key(&stored.to_le_bytes()), &[stored as u32, 1], bound));
            stored += 1;
            if table.slots() != before {
                sizes.push(table.slots());
                if table.slots() <= THREAD_BACKING_SLOTS {
                    assert_eq!(table.backing(), THREAD_BACKING_SLOTS, "grown in place");
                }
                for i in 0..stored {
                    assert_eq!(
                        ids_of(&table, &i.to_le_bytes()),
                        Some(vec![i as u32, 1]),
                        "{i} after growing to {}",
                        table.slots()
                    );
                }
            }
        }
        assert_eq!(sizes, [64, 256, 1024, 4096, 8192, 16_384, 32_768]);
        assert!(table.backing() >= 32_768);
        assert_eq!(table.len(), stored);
        table.clear(100);
        assert_eq!(
            table.backing(),
            THREAD_BACKING_SLOTS,
            "made anew inside the reservation"
        );
        assert_eq!(table.slots(), ShortPieceTable::slots_for(100));
    }

    #[test]
    fn sizing_keeps_the_entries_under_the_fill_bound() {
        for entries in [0, 1, 63, 64, 1000, 16_384, 126_000, 149_000, 200_000] {
            let table = ShortPieceTable::for_entries(entries);
            assert!(table.slots().is_power_of_two());
            assert!(
                table.fill_bound() >= entries,
                "{entries}: {}",
                table.slots()
            );
            assert!(
                entries < MIN_SLOTS || table.slots() < entries * 4,
                "{entries}: {}",
                table.slots()
            );
        }
        assert_eq!(ShortPieceTable::for_entries(149_000).slots(), 262_144);
    }

    /// Words like a vocabulary's: letters with and without a leading space,
    /// distinct for distinct `i`.
    fn word(mut i: usize) -> Vec<u8> {
        let mut word = Vec::new();
        if !i.is_multiple_of(3) {
            word.push(b' ');
        }
        loop {
            word.push(b'a' + (i % 26) as u8);
            i /= 26;
            if i == 0 {
                return word;
            }
        }
    }

    /// Linear probing with a random hash reads (1 + 1 / (1 - load)) / 2
    /// slots per stored piece on average: 1.3 at three eighths full, 2.5 at
    /// the fill bound. A table of words reads no more than that.
    #[test]
    fn a_table_of_words_is_read_in_as_few_probes_as_a_random_hash_would_take() {
        let mut table = ShortPieceTable::for_entries(150_000);
        let bound = table.fill_bound();
        for i in 0..bound / 2 {
            assert!(table.insert(key(&word(i)), &[i as u32], 150_000));
        }
        let probes = table.mean_probes();
        assert!(probes < 1.6, "{probes} probes on average at three eighths");
        for i in bound / 2..bound {
            assert!(table.insert(key(&word(i)), &[i as u32], 150_000));
        }
        let probes = table.mean_probes();
        assert!(probes < 3.0, "{probes} probes on average at the fill bound");
        for i in (0..bound).step_by(997) {
            assert_eq!(ids_of(&table, &word(i)), Some(vec![i as u32]));
        }
    }
}
