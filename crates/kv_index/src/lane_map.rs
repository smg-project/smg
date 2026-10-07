//! The lane map: one worker's blocks by engine hash, as the event lane that owns the worker
//! keeps them for the chain index.
//!
//! An open-addressing table of 16-byte slots (engine hash, run id, offset) with a one-byte tag
//! per slot beside them: Fibonacci home slot, linear probing over the tags (dense, so a probe
//! run of several slots reads one cache line and touches a slot only when its tag matches), at
//! most three quarters full so every probe run ends at an empty tag, doubling growth, and
//! backward-shift deletion so steady churn (an engine storing and evicting at the same rate for
//! hours) never accumulates tombstones. Removals and insertions come in batches of an event's
//! blocks: the home slots of the next keys are prefetched a few keys ahead so their cache misses
//! overlap instead of serialising.
//!
//! Semantics match a hash map: a key maps to at most one place, `insert` replaces, `remove` of an
//! absent key is a no-op, iteration yields every entry once. A differential test against a hash
//! map model under random operations keeps the backshift right.

use crate::{chain_index::BlockRef, event_tree::SequenceHash};

/// Smallest table a non-empty map allocates.
const MIN_SLOTS: usize = 16;
/// How many keys ahead a batch touches.
const AHEAD: usize = 8;
/// Run id that marks an empty slot (never a live location: run ids are below 2^26).
const EMPTY: u32 = u32::MAX;
/// Tag of an empty slot; a full slot's tag is seven bits of its key with the top bit set.
const VACANT: u8 = 0;

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    run: u32,
    offset: u32,
}

const EMPTY_SLOT: Slot = Slot {
    key: 0,
    run: EMPTY,
    offset: 0,
};

/// One worker's blocks by engine hash: where each lives in the chain index.
pub struct ChainBlockMap {
    /// Power-of-two length, or empty before the first insert.
    slots: Box<[Slot]>,
    /// One tag per slot: `VACANT`, or the key's fingerprint.
    tags: Box<[u8]>,
    len: usize,
    shift: u32,
}

impl Default for ChainBlockMap {
    fn default() -> Self {
        Self {
            slots: Box::default(),
            tags: Box::default(),
            len: 0,
            shift: 64,
        }
    }
}

/// Seven bits of the key the home slot does not use, with the top bit set so it is never
/// `VACANT`.
#[inline]
fn fingerprint(key: u64) -> u8 {
    ((key >> 25) as u8) | 0x80
}

impl ChainBlockMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slots allocated (16 bytes each, plus a tag byte).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        // Fibonacci hashing spreads structured keys; engine hashes are already uniform.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The slot holding `key`, or else the empty slot that ends its probe run. Requires slots.
    /// Walks the tags; a slot is read only when its tag matches.
    #[inline]
    fn probe(&self, key: u64) -> Result<usize, usize> {
        let mask = self.mask();
        let tag = fingerprint(key);
        let mut index = self.home(key);
        loop {
            let found = self.tags[index];
            if found == VACANT {
                return Err(index);
            }
            if found == tag && self.slots[index].key == key {
                return Ok(index);
            }
            index = (index + 1) & mask;
        }
    }

    fn find(&self, key: u64) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        self.probe(key).ok()
    }

    pub fn contains_key(&self, key: SequenceHash) -> bool {
        self.find(key.0).is_some()
    }

    pub fn get(&self, key: SequenceHash) -> Option<BlockRef> {
        self.find(key.0).map(|index| {
            let slot = &self.slots[index];
            BlockRef {
                run: slot.run,
                offset: slot.offset,
            }
        })
    }

    /// Room for `additional` more entries under three-quarters load.
    fn reserve(&mut self, additional: usize) {
        let needed = self.len + additional;
        if needed * 4 <= self.slots.len() * 3 {
            return;
        }
        let mut capacity = self.slots.len().max(MIN_SLOTS);
        while needed * 4 > capacity * 3 {
            capacity *= 2;
        }
        let old = std::mem::replace(&mut self.slots, (0..capacity).map(|_| EMPTY_SLOT).collect());
        self.tags = (0..capacity).map(|_| VACANT).collect();
        self.shift = 64 - capacity.trailing_zeros();
        let mask = capacity - 1;
        for slot in old.iter().filter(|slot| slot.run != EMPTY) {
            let mut index = self.home(slot.key);
            while self.tags[index] != VACANT {
                index = (index + 1) & mask;
            }
            self.slots[index] = *slot;
            self.tags[index] = fingerprint(slot.key);
        }
    }

    /// Insert or replace; the previous location when the key was present.
    pub fn insert(&mut self, key: SequenceHash, at: BlockRef) -> Option<BlockRef> {
        self.reserve(1);
        match self.probe(key.0) {
            Ok(index) => {
                let slot = &mut self.slots[index];
                let previous = BlockRef {
                    run: slot.run,
                    offset: slot.offset,
                };
                slot.run = at.run;
                slot.offset = at.offset;
                Some(previous)
            }
            Err(empty) => {
                self.slots[empty] = Slot {
                    key: key.0,
                    run: at.run,
                    offset: at.offset,
                };
                self.tags[empty] = fingerprint(key.0);
                self.len += 1;
                None
            }
        }
    }

    /// Remove `key`; its location when it was present.
    pub fn remove(&mut self, key: SequenceHash) -> Option<BlockRef> {
        let mut hole = self.find(key.0)?;
        let removed = BlockRef {
            run: self.slots[hole].run,
            offset: self.slots[hole].offset,
        };
        self.slots[hole] = EMPTY_SLOT;
        self.tags[hole] = VACANT;
        self.len -= 1;
        // Backward-shift deletion: pull later entries of the probe run into the hole whenever
        // the hole lies between their home slot and their current slot.
        let mask = self.mask();
        let mut next = hole;
        loop {
            next = (next + 1) & mask;
            if self.tags[next] == VACANT {
                break;
            }
            let slot = self.slots[next];
            let home = self.home(slot.key);
            if (next.wrapping_sub(home) & mask) >= (next.wrapping_sub(hole) & mask) {
                self.slots[hole] = slot;
                self.tags[hole] = self.tags[next];
                self.slots[next] = EMPTY_SLOT;
                self.tags[next] = VACANT;
                hole = next;
            }
        }
        Some(removed)
    }

    /// Ask the cache for the home slot of `key` before the key is used: a prefetch hint, so the
    /// misses of a batch overlap instead of serialising.
    #[inline]
    fn touch(&self, key: u64) {
        if !self.slots.is_empty() {
            let home = self.home(key);
            crate::prefetch::prefetch_read(&self.tags[home]);
            crate::prefetch::prefetch_read(&self.slots[home]);
        }
    }

    /// Remove every key of a batch, reporting each present one's location, with the home slots
    /// touched `AHEAD` keys early.
    pub(crate) fn remove_all(
        &mut self,
        keys: &[SequenceHash],
        mut on_removed: impl FnMut(BlockRef),
    ) {
        if self.len == 0 {
            return;
        }
        for key in &keys[..keys.len().min(AHEAD)] {
            self.touch(key.0);
        }
        for (index, key) in keys.iter().enumerate() {
            if let Some(next) = keys.get(index + AHEAD) {
                self.touch(next.0);
            }
            if let Some(at) = self.remove(*key) {
                on_removed(at);
            }
        }
    }

    /// Insert `count` consecutive blocks starting at `first` for the keys given, with the home
    /// slots touched `AHEAD` keys early: the lane map writes of one store. A key already present
    /// moves to its new place and `on_moved` sees its old and new places (a block the engine
    /// stored again at the same place passes through here unchanged).
    pub(crate) fn insert_run(
        &mut self,
        keys: impl ExactSizeIterator<Item = SequenceHash> + Clone,
        first: BlockRef,
        mut on_moved: impl FnMut(BlockRef, BlockRef),
    ) {
        self.reserve(keys.len());
        let mut ahead = keys.clone();
        for key in ahead.by_ref().take(AHEAD) {
            self.touch(key.0);
        }
        for (index, key) in keys.enumerate() {
            if let Some(next) = ahead.next() {
                self.touch(next.0);
            }
            let at = BlockRef {
                run: first.run,
                offset: first.offset + index as u32,
            };
            match self.probe(key.0) {
                Ok(slot) => {
                    let old = BlockRef {
                        run: self.slots[slot].run,
                        offset: self.slots[slot].offset,
                    };
                    if old != at {
                        on_moved(old, at);
                    }
                    self.slots[slot].run = at.run;
                    self.slots[slot].offset = at.offset;
                }
                Err(empty) => {
                    self.slots[empty] = Slot {
                        key: key.0,
                        run: at.run,
                        offset: at.offset,
                    };
                    self.tags[empty] = fingerprint(key.0);
                    self.len += 1;
                }
            }
        }
    }

    /// Every entry, in table order.
    pub fn iter(&self) -> impl Iterator<Item = (SequenceHash, BlockRef)> + '_ {
        self.slots
            .iter()
            .filter(|slot| slot.run != EMPTY)
            .map(|slot| {
                (
                    SequenceHash(slot.key),
                    BlockRef {
                        run: slot.run,
                        offset: slot.offset,
                    },
                )
            })
    }
}

impl IntoIterator for ChainBlockMap {
    type Item = (SequenceHash, BlockRef);
    type IntoIter = std::vec::IntoIter<(SequenceHash, BlockRef)>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

impl std::fmt::Debug for ChainBlockMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainBlockMap")
            .field("len", &self.len)
            .field("slots", &self.slots.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn at(n: u64) -> BlockRef {
        BlockRef {
            run: (n % 1000) as u32,
            offset: (n % 97) as u32,
        }
    }

    /// Random inserts, replacements, single and batched removals (present and absent keys), and
    /// batched runs of inserts against a hash map model; the table must agree after every step
    /// and at the end, entry for entry.
    #[test]
    fn matches_a_hash_map_under_random_operations() {
        let mut rng = Rng(20261005);
        let mut table = ChainBlockMap::default();
        let mut model: FxHashMap<SequenceHash, BlockRef> = FxHashMap::default();
        // Keys from a small universe so the same ones come back after removal and probe runs
        // overlap; a few structured keys (multiples of a power of two) stress the home hash.
        let key = |r: &mut Rng| {
            let n = r.next() % 6000;
            SequenceHash(if n.is_multiple_of(7) {
                n << 40
            } else {
                n.wrapping_mul(0x9E37_79B9)
            })
        };
        for step in 0..200_000u64 {
            match rng.next() % 10 {
                0..=3 => {
                    let k = key(&mut rng);
                    let v = at(rng.next());
                    assert_eq!(
                        table.insert(k, v),
                        model.insert(k, v),
                        "insert at step {step}"
                    );
                }
                4..=5 => {
                    let k = key(&mut rng);
                    assert_eq!(table.remove(k), model.remove(&k), "remove at step {step}");
                }
                6 => {
                    let keys: Vec<SequenceHash> =
                        (0..(rng.next() % 40)).map(|_| key(&mut rng)).collect();
                    let mut removed = 0usize;
                    table.remove_all(&keys, |_| removed += 1);
                    let expected = keys.iter().filter(|k| model.remove(k).is_some()).count();
                    assert_eq!(removed, expected, "batch removal at step {step}");
                }
                7 => {
                    let keys: Vec<SequenceHash> =
                        (0..(rng.next() % 60)).map(|_| key(&mut rng)).collect();
                    let first = at(rng.next());
                    table.insert_run(keys.iter().copied(), first, |_, _| {});
                    for (index, k) in keys.iter().enumerate() {
                        model.insert(
                            *k,
                            BlockRef {
                                run: first.run,
                                offset: first.offset + index as u32,
                            },
                        );
                    }
                }
                _ => {
                    let k = key(&mut rng);
                    assert_eq!(table.get(k), model.get(&k).copied(), "get at step {step}");
                    assert_eq!(table.contains_key(k), model.contains_key(&k));
                }
            }
            assert_eq!(table.len(), model.len(), "len at step {step}");
            assert!(table.len() * 4 <= table.capacity() * 3 || table.capacity() == 0);
        }
        let mut seen: Vec<(SequenceHash, BlockRef)> = table.iter().collect();
        seen.sort_by_key(|(k, _)| k.0);
        let mut expected: Vec<(SequenceHash, BlockRef)> =
            model.iter().map(|(k, v)| (*k, *v)).collect();
        expected.sort_by_key(|(k, _)| k.0);
        assert_eq!(seen, expected);
        let drained: FxHashMap<SequenceHash, BlockRef> = table.into_iter().collect();
        assert_eq!(drained, model);
    }

    /// Probe lengths at the load the table runs at: how many slots a hit walks and how many an
    /// insert walks to its empty slot, for uniform keys at just under three quarters full.
    #[test]
    fn probe_lengths_at_three_quarters_load() {
        let mut rng = Rng(7);
        let mut table = ChainBlockMap::default();
        let keys: Vec<SequenceHash> = (0..24_000).map(|_| SequenceHash(rng.next())).collect();
        for (index, key) in keys.iter().enumerate() {
            table.insert(*key, at(index as u64));
        }
        assert_eq!(table.capacity(), 32_768);
        let mask = table.mask();
        let mut hit = [0usize; 65];
        let mut miss = [0usize; 65];
        for key in &keys {
            let home = table.home(key.0);
            let found = table.probe(key.0).expect("present");
            hit[(found.wrapping_sub(home) & mask).min(64)] += 1;
        }
        for _ in 0..24_000 {
            let key = rng.next();
            let home = table.home(key);
            let empty = table.probe(key).expect_err("absent");
            miss[(empty.wrapping_sub(home) & mask).min(64)] += 1;
        }
        let mean = |h: &[usize; 65]| {
            h.iter().enumerate().map(|(d, n)| d * n).sum::<usize>() as f64
                / h.iter().sum::<usize>() as f64
        };
        let tail = |h: &[usize; 65], d: usize| {
            h[d..].iter().sum::<usize>() as f64 / h.iter().sum::<usize>() as f64
        };
        // Measured on this layout: hits 1.37 slots, inserts 6.16 with 11% past 16 slots.
        assert!(
            mean(&hit) < 4.0,
            "hit probe mean {:.2} (>4: {:.1}%, >16: {:.2}%)",
            mean(&hit),
            100.0 * tail(&hit, 5),
            100.0 * tail(&hit, 17)
        );
        assert!(
            mean(&miss) < 16.0,
            "insert probe mean {:.2} (>4: {:.1}%, >16: {:.2}%)",
            mean(&miss),
            100.0 * tail(&miss, 5),
            100.0 * tail(&miss, 17)
        );
    }

    #[test]
    fn empty_map_answers_without_slots() {
        let mut table = ChainBlockMap::default();
        assert_eq!(table.capacity(), 0);
        assert!(table.get(SequenceHash(1)).is_none());
        assert!(table.remove(SequenceHash(1)).is_none());
        table.remove_all(&[SequenceHash(1), SequenceHash(2)], |_| {
            panic!("nothing to remove")
        });
        table.insert_run(std::iter::empty(), at(0), |_, _| {});
        assert!(table.is_empty());
        assert!(table.insert(SequenceHash(1), at(1)).is_none());
        assert_eq!(table.capacity(), MIN_SLOTS);
        assert_eq!(table.len(), 1);
    }
}
