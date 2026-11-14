//! The map tiers of a thread's piece cache (`PieceCache` in the parent):
//! the pieces longer than the packed table keys, each with the ids the
//! encoder gave it, in a table whose storage is two slabs. An entry is a
//! 16-byte slot: the low half of the piece's hash, where the piece's bytes
//! lie in the byte slab and how many there are, where its ids lie in the id
//! slab and how many. A remembered piece appends to the slabs and writes a
//! slot, so a first sight costs no allocation of its own (the slabs grow by
//! doubling, the slots too); a clear truncates the slabs and empties the
//! slots in place; and a thread's entries lie dense in three arrays rather
//! than in two heap objects each, as a map of boxed pieces and boxed ids
//! held them (two allocations per first sight, two frees per entry at a
//! clear, the probe reading scattered lines).
//!
//! Open addressing with linear probing at half fill: a probe compares the
//! stored hash half, then the piece in the slab, and ends on an empty slot.
//! The hash is the one the map used (`FxHasher` over the bytes, the length
//! first, as a byte slice hashes), so a lookup costs the hash it did; its
//! low 32 bits pick the slot and, stored, stand as the tag.

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

/// Slots a map starts with (1 KiB), doubled as the thread meets pieces.
const MIN_SLOTS: usize = 64;

/// One entry; see the module documentation. Empty when `piece_len` is 0:
/// no cached piece is.
#[derive(Clone, Copy)]
struct Slot {
    /// The low 32 bits of the piece's hash.
    hash: u32,
    /// Where the piece's bytes start in the byte slab.
    piece_start: u32,
    /// Where its ids start in the id slab.
    ids_start: u32,
    /// The piece's bytes.
    piece_len: u16,
    /// Its ids.
    ids_len: u16,
}

impl Slot {
    const EMPTY: Self = Self {
        hash: 0,
        piece_start: 0,
        ids_start: 0,
        piece_len: 0,
        ids_len: 0,
    };
}

/// What a slot takes: an entry's cost beside its piece and its ids.
pub(super) const SLOT_BYTES: usize = size_of::<Slot>();

/// The table; see the module documentation.
pub(super) struct PieceMap {
    /// A power of two of slots, [`MIN_SLOTS`] at least.
    slots: Vec<Slot>,
    /// Entries stored.
    len: usize,
    /// The pieces' bytes, each entry's run where its slot says.
    bytes: Vec<u8>,
    /// The entries' ids, likewise.
    ids: Vec<u32>,
}

impl PieceMap {
    /// An empty map with the fewest slots and no slab.
    pub(super) fn new() -> Self {
        Self {
            slots: vec![Slot::EMPTY; MIN_SLOTS],
            len: 0,
            bytes: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Entries stored.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Whether no entry is stored.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Every entry, the piece and its ids, in slot order.
    #[cfg(test)]
    pub(super) fn entries(&self) -> impl Iterator<Item = (&[u8], &[u32])> {
        self.slots
            .iter()
            .filter(|slot| slot.piece_len != 0)
            .map(|slot| (self.piece_of(slot), self.ids_of(slot)))
    }

    /// Entries the map has room for before it grows: half its slots.
    pub(super) fn capacity(&self) -> usize {
        self.slots.len() / 2
    }

    /// What the entries take: their slots, their pieces and their ids.
    pub(super) fn bytes(&self) -> usize {
        self.len * SLOT_BYTES + self.bytes.len() + size_of_val(self.ids.as_slice())
    }

    /// What the map's arrays hold, used or not: every slot and the two
    /// slabs' capacity.
    pub(super) fn held_bytes(&self) -> usize {
        self.slots.len() * SLOT_BYTES
            + self.bytes.capacity()
            + self.ids.capacity() * size_of::<u32>()
    }

    /// The ids of `piece`, if the map holds it.
    #[inline(always)]
    pub(super) fn get(&self, piece: &[u8]) -> Option<&[u32]> {
        let hash = hash_of(piece);
        let mask = self.slots.len() - 1;
        let mut at = hash as usize & mask;
        loop {
            let slot = &self.slots[at];
            if slot.piece_len == 0 {
                return None;
            }
            if slot.hash == hash && self.piece_of(slot) == piece {
                return Some(self.ids_of(slot));
            }
            at = (at + 1) & mask;
        }
    }

    /// Stores `ids` as the encoding of `piece`; a piece the map holds keeps
    /// the ids it has (they are the model's for that piece either way). A
    /// map at half fill doubles its slots first. A piece or an id run past
    /// what a slot counts (65,535) is not stored, nor anything once a slab
    /// has 4 GiB behind it; no tier's bound reaches either.
    pub(super) fn insert(&mut self, piece: &[u8], ids: &[u32]) {
        let (Ok(piece_len), Ok(ids_len)) = (u16::try_from(piece.len()), u16::try_from(ids.len()))
        else {
            return;
        };
        let (Ok(piece_start), Ok(ids_start)) = (
            u32::try_from(self.bytes.len()),
            u32::try_from(self.ids.len()),
        ) else {
            return;
        };
        if piece_len == 0 {
            return;
        }
        if self.len >= self.capacity() {
            self.grow();
        }
        let hash = hash_of(piece);
        let mask = self.slots.len() - 1;
        let mut at = hash as usize & mask;
        while self.slots[at].piece_len != 0 {
            if self.slots[at].hash == hash && self.piece_of(&self.slots[at]) == piece {
                return;
            }
            at = (at + 1) & mask;
        }
        self.bytes.extend_from_slice(piece);
        self.ids.extend_from_slice(ids);
        self.slots[at] = Slot {
            hash,
            piece_start,
            ids_start,
            piece_len,
            ids_len,
        };
        self.len += 1;
    }

    /// Forgets every entry: the slabs truncated and the slots emptied in
    /// place, their capacity kept for the pieces the thread meets next.
    pub(super) fn clear(&mut self) {
        self.slots.fill(Slot::EMPTY);
        self.len = 0;
        self.bytes.clear();
        self.ids.clear();
    }

    /// [`Self::clear`], or the map made anew with the fewest slots when it
    /// has room for more than twice `share` entries: the share fell below
    /// what the map was grown for (threads joined). A map that grew under
    /// the share stays under that, so at a steady share the check never
    /// fires and a clear at the bound keeps the slots and the slabs.
    pub(super) fn clear_within(&mut self, share: usize) {
        if self.capacity() > 2 * share {
            *self = Self::new();
        } else {
            self.clear();
        }
    }

    /// [`Self::clear`], or the map made anew with the fewest slots when its
    /// arrays hold more than four times `share` bytes: the share fell below
    /// what the map was grown for (threads joined). A map that grew under a
    /// byte share holds at most four times its live bytes (a slab's capacity
    /// under twice its length, the slots under four times their entries'),
    /// so at a steady share the check never fires and a clear at the bound
    /// keeps the slots and the slabs.
    pub(super) fn clear_within_bytes(&mut self, share: usize) {
        if self.held_bytes() > 4 * share {
            *self = Self::new();
        } else {
            self.clear();
        }
    }

    /// Doubles the slots and places every entry again where its hash now
    /// says; the slabs stay.
    fn grow(&mut self) {
        let doubled = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![Slot::EMPTY; doubled]);
        let mask = doubled - 1;
        for slot in old.into_iter().filter(|slot| slot.piece_len != 0) {
            let mut at = slot.hash as usize & mask;
            while self.slots[at].piece_len != 0 {
                at = (at + 1) & mask;
            }
            self.slots[at] = slot;
        }
    }

    #[inline(always)]
    fn piece_of(&self, slot: &Slot) -> &[u8] {
        let start = slot.piece_start as usize;
        &self.bytes[start..start + usize::from(slot.piece_len)]
    }

    #[inline(always)]
    fn ids_of(&self, slot: &Slot) -> &[u32] {
        let start = slot.ids_start as usize;
        &self.ids[start..start + usize::from(slot.ids_len)]
    }
}

/// The low half of the piece's hash, the length first as a byte slice
/// hashes.
#[inline(always)]
fn hash_of(piece: &[u8]) -> u32 {
    let mut hasher = FxHasher::default();
    piece.hash(&mut hasher);
    hasher.finish() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct piece of `len` bytes for each `i`.
    fn piece(len: usize, i: usize) -> Vec<u8> {
        let mut out = format!("{i:08}").into_bytes();
        out.resize(len, b'x');
        out
    }

    #[test]
    fn an_empty_map_holds_nothing_and_has_the_fewest_slots() {
        let map = PieceMap::new();
        assert_eq!(map.get(b"sixteen bytes!!!"), None);
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), MIN_SLOTS / 2);
        assert_eq!(map.bytes(), 0);
    }

    #[test]
    fn a_stored_piece_is_found_with_its_ids_and_a_second_store_keeps_them() {
        let mut map = PieceMap::new();
        map.insert(b"twenty bytes of text", &[1, 2, 3]);
        assert_eq!(map.get(b"twenty bytes of text"), Some(&[1, 2, 3][..]));
        assert_eq!(map.get(b"twenty bytes of tex"), None);
        map.insert(b"twenty bytes of text", &[9]);
        assert_eq!(map.get(b"twenty bytes of text"), Some(&[1, 2, 3][..]));
        assert_eq!(map.len(), 1);
        assert_eq!(map.bytes(), SLOT_BYTES + 20 + 3 * size_of::<u32>());
    }

    #[test]
    fn an_entry_without_ids_is_found_empty() {
        let mut map = PieceMap::new();
        map.insert(b"a piece with no ids", &[]);
        assert_eq!(map.get(b"a piece with no ids"), Some(&[][..]));
    }

    #[test]
    fn pieces_past_the_fill_bound_grow_the_slots_and_stay_found() {
        let mut map = PieceMap::new();
        for i in 0..5_000 {
            map.insert(&piece(16 + i % 49, i), &[i as u32, 7]);
        }
        assert_eq!(map.len(), 5_000);
        assert!(map.capacity() >= 5_000, "{}", map.capacity());
        assert!(map.slots.len().is_power_of_two());
        for i in 0..5_000 {
            assert_eq!(
                map.get(&piece(16 + i % 49, i)),
                Some(&[i as u32, 7][..]),
                "{i}"
            );
        }
        assert_eq!(map.get(&piece(16, 5_000)), None);
    }

    #[test]
    fn a_clear_forgets_the_entries_and_keeps_the_room() {
        let mut map = PieceMap::new();
        for i in 0..1_000 {
            map.insert(&piece(24, i), &[i as u32]);
        }
        let (slots, bytes, ids) = (map.slots.len(), map.bytes.capacity(), map.ids.capacity());
        map.clear();
        assert_eq!(map.len(), 0);
        assert_eq!(map.bytes(), 0);
        assert_eq!(map.get(&piece(24, 1)), None);
        assert_eq!(
            (map.slots.len(), map.bytes.capacity(), map.ids.capacity()),
            (slots, bytes, ids)
        );
        map.insert(&piece(24, 1), &[5]);
        assert_eq!(map.get(&piece(24, 1)), Some(&[5][..]));
    }

    #[test]
    fn a_clear_within_a_share_the_map_outgrew_makes_it_anew() {
        let mut map = PieceMap::new();
        for i in 0..1_000 {
            map.insert(&piece(24, i), &[i as u32]);
        }
        let grown = map.capacity();
        map.clear_within(grown / 2);
        assert_eq!(
            map.capacity(),
            grown,
            "a clear at a share the slots serve keeps them"
        );
        for i in 0..1_000 {
            map.insert(&piece(24, i), &[i as u32]);
        }
        map.clear_within(grown / 2 - 1);
        assert_eq!(
            map.capacity(),
            MIN_SLOTS / 2,
            "a share under half the room: made anew"
        );
        assert_eq!(map.bytes.capacity(), 0);
        assert_eq!(map.get(&piece(24, 1)), None);
    }

    #[test]
    fn a_clear_within_a_byte_share_the_map_outgrew_makes_it_anew() {
        let share = 64 << 10;
        let mut map = PieceMap::new();
        // The long tier's loop: an entry that would take the tier over its
        // share clears it first.
        for i in 0..10_000 {
            let piece = piece(100, i);
            let ids = [i as u32; 30];
            if map.bytes() + SLOT_BYTES + piece.len() + size_of_val(&ids) > share {
                map.clear_within_bytes(share);
            }
            map.insert(&piece, &ids);
        }
        let held = map.held_bytes();
        assert!(held > share, "the arrays grew past the share: {held}");
        assert!(
            held <= 4 * share,
            "a map grown under the share holds under four times it: {held}"
        );
        map.clear_within_bytes(share);
        assert_eq!(
            map.held_bytes(),
            held,
            "a clear at a steady share keeps the arrays"
        );
        assert_eq!(map.len(), 0);
        map.clear_within_bytes(held / 4 - 1);
        assert_eq!(
            map.held_bytes(),
            MIN_SLOTS * SLOT_BYTES,
            "a share under a quarter: made anew"
        );
        assert_eq!(map.capacity(), MIN_SLOTS / 2);
    }

    #[test]
    fn a_piece_or_an_id_run_past_what_a_slot_counts_is_not_stored() {
        let mut map = PieceMap::new();
        map.insert(&vec![b'a'; 70_000], &[1]);
        map.insert(b"a piece of sixteen", &vec![1; 70_000]);
        map.insert(b"", &[1]);
        assert_eq!(map.len(), 0);
        assert_eq!(map.bytes(), 0);
    }
}
