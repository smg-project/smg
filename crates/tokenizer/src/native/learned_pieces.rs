//! The pieces the threads of one encoder have merged, kept for all of them.
//!
//! A thread keeps the short pieces it has met in a table of its own
//! ([`ShortPieceTable`]) and clears it whole at its share, so a thread that
//! encodes much merges the same words again after every clear, and every
//! thread merges for itself what another has merged already. This table is
//! the encoder's, like the seed, and an encoder has one once enough threads
//! call it (the parent module's `LEARNED_FROM_THREADS`): a thread
//! that has cleared its own table then asks it for a piece that table and
//! the seed do not hold, before the merge, keeps what it finds the way it
//! keeps a seeded piece, and writes what it had to merge. A clear of a
//! thread does not empty it.
//!
//! The table is the packed table of a thread in parts, a key in the part
//! its hash names. A part takes entries behind a reader-writer lock, and
//! nobody waits: a reader that finds a writer in its part merges the piece
//! as it would have, and a writer that finds its part taken leaves the
//! piece to the next thread that merges it. A part that is full takes no
//! more and is set down once for good: from then on it is read with no
//! lock, and a merge of its keys writes nothing. The table therefore keeps
//! the pieces that were merged first, and a piece it has no room for is
//! merged by each thread after each of its clears, as every piece was
//! before this table.

use std::{
    ops::Deref,
    sync::{OnceLock, RwLock, RwLockReadGuard},
};

use super::short_pieces::{ShortPieceKey, ShortPieceTable};

/// The parts of the table: enough that two of sixteen threads seldom meet
/// in one while it still takes entries.
const PARTS: usize = 64;

struct Part {
    /// The part while it takes entries.
    open: RwLock<ShortPieceTable>,
    /// The part once it is full, set down once: read with no lock.
    full: OnceLock<ShortPieceTable>,
}

/// A part to probe: the full one as it stands, or the open one under its
/// read lock.
pub(super) enum PartRead<'a> {
    Full(&'a ShortPieceTable),
    Open(RwLockReadGuard<'a, ShortPieceTable>),
}

impl Deref for PartRead<'_> {
    type Target = ShortPieceTable;

    #[inline(always)]
    fn deref(&self) -> &ShortPieceTable {
        match self {
            Self::Full(table) => table,
            Self::Open(table) => table,
        }
    }
}

/// The table; see the module documentation.
pub(super) struct LearnedPieces {
    parts: Box<[Part]>,
    /// The entries one part is sized for.
    part_bound: usize,
}

impl LearnedPieces {
    /// An empty table for about `entries` pieces, each part
    /// [`ShortPieceTable::slots_for`] its share of them when it is full and
    /// 64 slots until a thread writes to it.
    pub(super) fn for_entries(entries: usize) -> Self {
        Self {
            parts: (0..PARTS)
                .map(|_| Part {
                    open: RwLock::new(ShortPieceTable::for_entries(0)),
                    full: OnceLock::new(),
                })
                .collect(),
            part_bound: entries / PARTS,
        }
    }

    /// The entries the table was made for.
    #[cfg(test)]
    pub(super) fn entries(&self) -> usize {
        self.part_bound * PARTS
    }

    fn part(&self, key: ShortPieceKey) -> &Part {
        &self.parts[key.part(PARTS)]
    }

    /// The part that would hold `key`, to probe: the full part with no
    /// lock; an open one unless a writer is in it (or one died in it: a
    /// poisoned part answers nothing).
    #[inline]
    pub(super) fn read(&self, key: ShortPieceKey) -> Option<PartRead<'_>> {
        let part = self.part(key);
        match part.full.get() {
            Some(table) => Some(PartRead::Full(table)),
            None => part.open.try_read().ok().map(PartRead::Open),
        }
    }

    /// One byte of the tag and of the entry a probe of `key` begins on in
    /// its part, folded ([`ShortPieceTable::touch`]): asked of several keys
    /// one after the other, their far lines are read together. Nothing of a
    /// part a writer is in.
    #[inline(always)]
    pub(super) fn touch(&self, key: ShortPieceKey) -> u8 {
        let part = self.part(key);
        match part.full.get() {
            Some(table) => table.touch(key),
            None => part.open.try_read().map_or(0, |table| table.touch(key)),
        }
    }

    /// Keeps the ids a merge gave for `key` (1 to 15 of them, as a thread's
    /// table takes), unless its part is full or taken by a reader or a
    /// writer: the next thread that merges the piece writes it. The write
    /// that finds the part at its bound sets it down as full.
    pub(super) fn learn(&self, key: ShortPieceKey, ids: &[u32]) {
        let part = self.part(key);
        if part.full.get().is_some() {
            return;
        }
        let Ok(mut open) = part.open.try_write() else {
            return;
        };
        // Read again under the lock: a part set down between the check
        // above and the lock has left an empty open table that is read no
        // more, and nothing is written there.
        if part.full.get().is_some() {
            return;
        }
        // A write refused at the slots of the part's bound: the part is
        // full. (Ids a table takes at no size, none or more than 15, are
        // refused in a part that can still grow and set nothing down.)
        if !open.insert(key, ids, self.part_bound)
            && open.len() >= open.fill_bound()
            && open.slots() >= ShortPieceTable::slots_for(self.part_bound)
        {
            let table = std::mem::replace(&mut *open, ShortPieceTable::for_entries(0));
            // Set once: the part was not full under this lock, and only a
            // holder of the lock sets it down.
            let _ = part.full.set(table);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(piece: &[u8]) -> ShortPieceKey {
        ShortPieceKey::of(piece).unwrap()
    }

    fn ids_of(table: &LearnedPieces, piece: &[u8]) -> Option<Vec<u32>> {
        let part = table.read(key(piece))?;
        let entry = part.probe(key(piece)).ok()?;
        let mut ids = Vec::new();
        part.append(entry, &mut ids);
        Some(ids)
    }

    /// The `i`-th of a run of distinct pieces.
    fn word(i: usize) -> Vec<u8> {
        format!("w{i}").into_bytes()
    }

    #[test]
    fn a_table_answers_what_it_was_taught_and_nothing_else() {
        let table = LearnedPieces::for_entries(PARTS * 64);
        assert_eq!(ids_of(&table, b"hello"), None);
        table.learn(key(b"hello"), &[7, 8]);
        table.learn(key(b"spilled"), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(ids_of(&table, b"hello"), Some(vec![7, 8]));
        assert_eq!(ids_of(&table, b"spilled"), Some(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(ids_of(&table, b"other"), None);
        assert_eq!(table.touch(key(b"other")), table.touch(key(b"other")));
    }

    #[test]
    fn a_full_part_is_set_down_keeps_what_it_held_and_takes_no_more() {
        let table = LearnedPieces::for_entries(PARTS * 64);
        let mut taught = Vec::new();
        for i in 0..PARTS * 4096 {
            let piece = word(i);
            table.learn(key(&piece), &[i as u32 + 1]);
            if ids_of(&table, &piece).is_some() {
                taught.push((piece, i as u32 + 1));
            }
        }
        // Every part met more pieces than it takes: each is full, read with
        // no lock, and holds its fill bound.
        for part in &table.parts {
            let full = part.full.get().expect("a part past its bound is set down");
            assert_eq!(full.len(), full.fill_bound());
            assert_eq!(full.slots(), ShortPieceTable::slots_for(64));
            assert_eq!(part.open.try_read().unwrap().len(), 0);
        }
        assert_eq!(taught.len(), PARTS * ShortPieceTable::slots_for(64) / 4 * 3);
        for (piece, id) in &taught {
            assert_eq!(ids_of(&table, piece), Some(vec![*id]));
        }
        // A full part takes no more, and a reader of it shuts no writer out.
        let held = table.read(key(b"late")).unwrap();
        assert!(matches!(held, PartRead::Full(_)));
        table.learn(key(b"late"), &[9]);
        assert_eq!(ids_of(&table, b"late"), None);
    }

    #[test]
    fn ids_a_table_takes_at_no_size_set_no_part_down() {
        let table = LearnedPieces::for_entries(PARTS * 4096);
        let part = table.part(key(b"late"));
        let at_fill_bound = |part: &Part| {
            let open = part.open.try_read().unwrap();
            open.len() >= open.fill_bound()
        };
        // The part at the fill bound of its first slots: it can still grow.
        for piece in (0..).map(word) {
            if at_fill_bound(part) {
                break;
            }
            if key(&piece).part(PARTS) == key(b"late").part(PARTS) {
                table.learn(key(&piece), &[1]);
            }
        }
        table.learn(key(b"late"), &[0; 16]);
        assert!(part.full.get().is_none());
        table.learn(key(b"late"), &[9]);
        assert_eq!(ids_of(&table, b"late"), Some(vec![9]));
    }

    #[test]
    fn nobody_waits_in_a_part_that_still_takes_entries() {
        let table = LearnedPieces::for_entries(PARTS * 64);
        table.learn(key(b"first"), &[1]);
        // A writer that finds a reader in its part leaves the piece.
        let reader = table.read(key(b"second")).unwrap();
        assert!(matches!(reader, PartRead::Open(_)));
        let same_part = (0..)
            .map(word)
            .find(|piece| key(piece).part(PARTS) == key(b"second").part(PARTS))
            .unwrap();
        table.learn(key(&same_part), &[2]);
        drop(reader);
        assert_eq!(ids_of(&table, &same_part), None);
        // The next merge of the piece writes it.
        table.learn(key(&same_part), &[2]);
        assert_eq!(ids_of(&table, &same_part), Some(vec![2]));
        // A reader that finds a writer in its part is answered nothing.
        let part = table.part(key(&same_part));
        let writer = part.open.try_write().unwrap();
        assert!(table.read(key(&same_part)).is_none());
        assert_eq!(table.touch(key(&same_part)), 0);
        drop(writer);
        assert_eq!(ids_of(&table, &same_part), Some(vec![2]));
    }

    #[test]
    fn the_keys_spread_over_the_parts() {
        let mut counts = [0usize; PARTS];
        for i in 0..PARTS * 1000 {
            counts[key(&word(i)).part(PARTS)] += 1;
        }
        let (least, most) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
        assert!(*least > 800 && *most < 1200, "least {least}, most {most}");
    }
}
