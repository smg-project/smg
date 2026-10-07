//! The word arena behind the chain index: hash and engine-hash arrays, child tables,
//! partial-holder tables and forwarding records in one address space of 64-bit words, with
//! per-class free lists.

use super::*;

/// Hash array capacity class: `8, 16, .., 128` in steps of 8, then four classes per octave
/// (`160, 192, 224, 256, 320, ..`), so an array wastes at most a quarter of its words to the
/// class, not half; a run of 140 blocks with its slack lands in 160, not 256.
pub(super) fn array_class(capacity: usize) -> usize {
    if capacity <= 8 * SMALL_ARRAY_CLASSES {
        capacity.div_ceil(8).max(1) - 1
    } else {
        // `capacity` in `(low, 2 * low]` with `low` a power of two from 128 up.
        let bits = usize::BITS - (capacity - 1).leading_zeros();
        let low = 1usize << (bits - 1);
        let quarter = (capacity - low).div_ceil(low / 4).max(1) - 1;
        SMALL_ARRAY_CLASSES + 4 * (bits as usize - 8) + quarter
    }
}

pub(super) fn class_capacity(class: usize) -> usize {
    if class < SMALL_ARRAY_CLASSES {
        8 * (class + 1)
    } else {
        let above = class - SMALL_ARRAY_CLASSES;
        let low = 1usize << (7 + above / 4);
        low + (above % 4 + 1) * (low / 4)
    }
}

/// Room for `len` hashes plus a little for decode extensions.
pub(super) fn capacity_for(len: usize) -> usize {
    len + (len / 8).max(2)
}

pub(super) fn table_class(slots: usize) -> usize {
    (slots.trailing_zeros() as usize).saturating_sub(MIN_TABLE_SLOTS.trailing_zeros() as usize)
}

/// Append-only storage of 64-bit words in fixed chunks with free lists per size class. Holds
/// hash arrays (`used | capacity << 32`, `refs`, then the hashes) and child tables
/// (`slots | used << 32`, `live`, then `(head hash, run id | generation << 32)` slots). An
/// allocation never crosses a chunk, so any array is one slice.
pub(super) struct WordArena {
    pub(super) dir: Box<[OnceLock<Box<[AtomicU64]>>]>,
    pub(super) next: AtomicU64,
    pub(super) free_arrays: Vec<SegQueue<u32>>,
    pub(super) free_tables: Vec<SegQueue<u32>>,
    pub(super) free_partials: Vec<SegQueue<u32>>,
}

impl WordArena {
    pub(super) fn new() -> Self {
        Self {
            dir: (0..WORD_DIR).map(|_| OnceLock::new()).collect(),
            next: AtomicU64::new(1),
            free_arrays: (0..ARRAY_CLASSES).map(|_| SegQueue::new()).collect(),
            free_tables: (0..TABLE_CLASSES).map(|_| SegQueue::new()).collect(),
            free_partials: (0..TABLE_CLASSES).map(|_| SegQueue::new()).collect(),
        }
    }

    #[inline]
    pub(super) fn chunk(&self, index: usize) -> &[AtomicU64] {
        self.dir[index].get_or_init(|| (0..WORD_CHUNK).map(|_| AtomicU64::new(0)).collect())
    }

    /// `count` consecutive words starting at `start` (all within one chunk, as allocated). A
    /// lock-free reader may compute `count` from a header that is being recycled under it; it
    /// confirms the run's version afterwards and discards what it read, so the slice is clamped
    /// to the chunk rather than trusted.
    #[inline]
    pub(super) fn words(&self, start: u32, count: usize) -> &[AtomicU64] {
        let start = start as usize;
        let offset = start & (WORD_CHUNK - 1);
        let end = offset.saturating_add(count).min(WORD_CHUNK);
        &self.chunk(start >> WORD_CHUNK_BITS)[offset..end]
    }

    #[inline]
    pub(super) fn word(&self, at: u32) -> &AtomicU64 {
        &self.words(at, 1)[0]
    }

    /// Fresh words inside one chunk.
    pub(super) fn bump(&self, count: usize) -> u32 {
        debug_assert!(0 < count && count <= WORD_CHUNK);
        loop {
            let current = self.next.load(Ordering::Relaxed);
            let mut start = current as usize;
            if start >> WORD_CHUNK_BITS != (start + count - 1) >> WORD_CHUNK_BITS {
                start = ((start >> WORD_CHUNK_BITS) + 1) << WORD_CHUNK_BITS;
            }
            let end = start + count;
            assert!(
                end <= WORD_DIR * WORD_CHUNK,
                "chain index arena exhausted: more than 2^32 hash words"
            );
            if self
                .next
                .compare_exchange_weak(current, end as u64, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.chunk(start >> WORD_CHUNK_BITS);
                return start as u32;
            }
        }
    }

    pub(super) fn used(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    /// Bytes of chunks taken from the process allocator: what the arena costs in memory.
    pub(super) fn chunk_bytes(&self) -> usize {
        let chunks = self
            .dir
            .iter()
            .filter(|chunk| chunk.get().is_some())
            .count();
        chunks * WORD_CHUNK * size_of::<AtomicU64>()
    }

    /// Words sitting in free lists.
    pub(super) fn free_words(&self) -> usize {
        let arrays: usize = self
            .free_arrays
            .iter()
            .enumerate()
            .map(|(class, list)| list.len() * (class_capacity(class) + 2))
            .sum();
        let tables: usize = self
            .free_tables
            .iter()
            .enumerate()
            .map(|(class, list)| list.len() * (2 + 2 * (MIN_TABLE_SLOTS << class)))
            .sum();
        let partials: usize = self
            .free_partials
            .iter()
            .enumerate()
            .map(|(class, list)| list.len() * (2 + (MIN_TABLE_SLOTS << class)))
            .sum();
        arrays + tables + partials
    }

    // ---- hash arrays: [used | capacity << 32][refs][hash; capacity], data = start + 2 ----

    /// A hash array holding `contents` with room for at least `capacity`; returns the data start.
    pub(super) fn alloc_array(&self, contents: &[u64], capacity: usize) -> u32 {
        let wanted = array_class(capacity.max(contents.len()).max(8));
        // A freed array of the wanted class, else of one up to an octave larger: fresh words
        // are bumped only when nothing in that range is free, so the free lists of neighbouring
        // classes do not fill while others bump (churn moves arrays between classes as runs
        // grow and die).
        let (class, recycled) = (wanted..ARRAY_CLASSES.min(wanted + ARRAY_FIT_SPAN + 1))
            .find_map(|class| self.free_arrays[class].pop().map(|start| (class, start)))
            .unwrap_or((wanted, 0));
        let capacity = class_capacity(class);
        let start = if recycled == 0 {
            self.bump(capacity + 2)
        } else {
            recycled
        };
        let data = start + 2;
        for (slot, &hash) in self.words(data, contents.len()).iter().zip(contents) {
            slot.store(hash, Ordering::Relaxed);
        }
        self.word(start + 1).store(1, Ordering::Relaxed);
        self.word(start).store(
            pack(contents.len() as u32, capacity as u32),
            Ordering::Release,
        );
        data
    }

    #[inline]
    pub(super) fn array_header(&self, data: u32) -> &AtomicU64 {
        self.word(data - 2)
    }

    /// Words an array can hold: the class it was given, which may be larger than asked for.
    pub(super) fn array_capacity(&self, data: u32) -> usize {
        unpack(self.array_header(data).load(Ordering::Relaxed)).1 as usize
    }

    /// One more run shares this array.
    pub(super) fn array_retain(&self, data: u32) {
        self.word(data - 1).fetch_add(1, Ordering::Relaxed);
    }

    /// One run fewer uses this array; the last one frees it.
    pub(super) fn array_release(&self, data: u32) {
        if data == NONE {
            return;
        }
        if self.word(data - 1).fetch_sub(1, Ordering::AcqRel) == 1 {
            let (_, capacity) = unpack(self.array_header(data).load(Ordering::Relaxed));
            self.free_arrays[array_class(capacity as usize)].push(data - 2);
        }
    }

    // ---- child tables: [slots | used << 32][live][(head, run | gen << 32); slots] ----

    pub(super) fn alloc_table(&self, slots: usize) -> u32 {
        let class = table_class(slots);
        let recycled = self.free_tables[class].pop();
        let table = recycled.unwrap_or_else(|| self.bump(2 + 2 * slots));
        for word in self.words(table + 2, 2 * slots) {
            word.store(0, Ordering::Relaxed);
        }
        self.word(table + 1).store(0, Ordering::Relaxed);
        self.word(table).store(slots as u64, Ordering::Release);
        table
    }

    pub(super) fn free_table(&self, table: u32) {
        if table == NONE {
            return;
        }
        let slots = self.table_slots(table);
        self.free_tables[table_class(slots)].push(table);
    }

    #[inline]
    pub(super) fn table_slots(&self, table: u32) -> usize {
        self.word(table).load(Ordering::Relaxed) as u32 as usize
    }

    /// The child continuing with `head`, as `(run, generation)`.
    #[inline]
    /// Live entries of a child table (`NONE` has none).
    pub(super) fn table_live(&self, table: u32) -> usize {
        if table == NONE {
            0
        } else {
            self.word(table + 1).load(Ordering::Relaxed) as usize
        }
    }

    pub(super) fn table_find(&self, table: u32, head: u64) -> Option<(u32, u32)> {
        if table == NONE {
            return None;
        }
        let slots = self.table_slots(table);
        if slots == 0 || !slots.is_power_of_two() {
            // A header being recycled under a lock-free reader; the version check discards this.
            return None;
        }
        let words = self.words(table + 2, 2 * slots);
        let mask = slots - 1;
        let mut index = head as usize & mask;
        for _ in 0..slots {
            let entry = words.get(2 * index + 1)?.load(Ordering::Acquire);
            if entry == 0 {
                return None;
            }
            if entry != TOMB && words.get(2 * index)?.load(Ordering::Relaxed) == head {
                return Some((entry as u32, (entry >> 32) as u32));
            }
            index = (index + 1) & mask;
        }
        None
    }

    /// Claim a slot for `key` without a lock: the head word is taken by compare-and-swap, then the
    /// run word is published. A reader stops at an empty run word, so an entry in flight is simply
    /// not there yet for it; a writer that meets the same head in flight waits for it.
    pub(super) fn table_claim(&self, table: u32, key: u64, child: u32, generation: u32) -> Claim {
        if table == NONE {
            return Claim::Full;
        }
        let slots = self.table_slots(table);
        if slots == 0 || !slots.is_power_of_two() {
            return Claim::Full;
        }
        let words = self.words(table + 2, 2 * slots);
        let mask = slots - 1;
        let mut index = key as usize & mask;
        for _ in 0..slots {
            let (Some(head_word), Some(run_word)) =
                (words.get(2 * index), words.get(2 * index + 1))
            else {
                return Claim::Full;
            };
            let mut head = head_word.load(Ordering::Acquire);
            if head == 0 {
                match head_word.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        run_word.store(
                            u64::from(child) | (u64::from(generation) << 32),
                            Ordering::Release,
                        );
                        self.word(table).fetch_add(1 << 32, Ordering::Relaxed);
                        self.word(table + 1).fetch_add(1, Ordering::Relaxed);
                        return Claim::Inserted;
                    }
                    Err(taken) => head = taken,
                }
            }
            if head == key {
                loop {
                    let run = run_word.load(Ordering::Acquire);
                    if run == 0 {
                        std::hint::spin_loop();
                        continue;
                    }
                    if run == TOMB {
                        break;
                    }
                    return Claim::Exists(run as u32, (run >> 32) as u32);
                }
            }
            index = (index + 1) & mask;
        }
        Claim::Full
    }

    /// Live `(head, run, generation)` entries of a table.
    pub(super) fn table_entries(&self, table: u32) -> Vec<(u64, u32, u32)> {
        if table == NONE {
            return Vec::new();
        }
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        (0..slots)
            .filter_map(|index| {
                let entry = words[2 * index + 1].load(Ordering::Acquire);
                (entry != 0 && entry != TOMB).then(|| {
                    (
                        words[2 * index].load(Ordering::Relaxed),
                        entry as u32,
                        (entry >> 32) as u32,
                    )
                })
            })
            .collect()
    }

    /// Write an entry into a free or tombstoned slot of a table that has room (writer side).
    pub(super) fn table_put(&self, table: u32, head: u64, run: u32, generation: u32) {
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        let mask = slots - 1;
        let mut index = head as usize & mask;
        loop {
            let entry = words[2 * index + 1].load(Ordering::Relaxed);
            if entry == 0 || entry == TOMB {
                words[2 * index].store(head, Ordering::Relaxed);
                words[2 * index + 1].store(
                    u64::from(run) | (u64::from(generation) << 32),
                    Ordering::Release,
                );
                let header = self.word(table);
                let used = (header.load(Ordering::Relaxed) >> 32) + u64::from(entry == 0);
                header.store(slots as u64 | (used << 32), Ordering::Relaxed);
                self.word(table + 1).fetch_add(1, Ordering::Relaxed);
                return;
            }
            index = (index + 1) & mask;
        }
    }

    /// Tombstone the entry of `run`; returns the live count left.
    pub(super) fn table_take(&self, table: u32, run: u32) -> u64 {
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        for index in 0..slots {
            let entry = words[2 * index + 1].load(Ordering::Relaxed);
            if entry != 0 && entry != TOMB && entry as u32 == run {
                words[2 * index + 1].store(TOMB, Ordering::Release);
                return self.word(table + 1).fetch_sub(1, Ordering::Relaxed) - 1;
            }
        }
        self.word(table + 1).load(Ordering::Relaxed)
    }

    /// A new table holding the live entries of `table` plus room for one more.
    pub(super) fn table_grown(&self, table: u32) -> u32 {
        let entries = self.table_entries(table);
        let needed = entries.len() + 1;
        let mut slots = MIN_TABLE_SLOTS;
        while needed * 4 > slots * 3 {
            slots *= 2;
        }
        let grown = self.alloc_table(slots);
        for (head, run, generation) in entries {
            self.table_put(grown, head, run, generation);
        }
        grown
    }

    /// A table holding `entries` (`(key, run, generation)`) with room for one more; `NONE` for
    /// none.
    pub(super) fn table_from(&self, entries: &[(u64, u32, u32)]) -> u32 {
        if entries.is_empty() {
            return NONE;
        }
        let needed = entries.len() + 1;
        let mut slots = MIN_TABLE_SLOTS;
        while needed * 4 > slots * 3 {
            slots *= 2;
        }
        let table = self.alloc_table(slots);
        for &(key, run, generation) in entries {
            self.table_put(table, key, run, generation);
        }
        table
    }

    /// Tombstoned slots of a table: entries taken out since it was built.
    pub(super) fn table_dead(&self, table: u32) -> usize {
        let header = self.word(table).load(Ordering::Relaxed);
        ((header >> 32) as usize)
            .saturating_sub(self.word(table + 1).load(Ordering::Relaxed) as usize)
    }
}

impl WordArena {
    // ---- partial holders: [slots | used << 32][live][(worker | cutoff << 32); slots] ----
    // Entries are appended in slot order; a removed entry is a tombstone. A worker listed here
    // holds the run's blocks `[0, cutoff)` with `0 < cutoff < len`.

    pub(super) fn alloc_partials(&self, slots: usize) -> u32 {
        let class = table_class(slots);
        let recycled = self.free_partials[class].pop();
        let table = recycled.unwrap_or_else(|| self.bump(2 + slots));
        for word in self.words(table + 2, slots) {
            word.store(0, Ordering::Relaxed);
        }
        self.word(table + 1).store(0, Ordering::Relaxed);
        self.word(table).store(slots as u64, Ordering::Release);
        table
    }

    pub(super) fn free_partials(&self, table: u32) {
        if table == NONE {
            return;
        }
        let slots = self.word(table).load(Ordering::Relaxed) as u32 as usize;
        self.free_partials[table_class(slots)].push(table);
    }

    /// `(slots, used)` of a partial table.
    #[inline]
    pub(super) fn partials_shape(&self, table: u32) -> (usize, usize) {
        let header = self.word(table).load(Ordering::Relaxed);
        (header as u32 as usize, (header >> 32) as usize)
    }

    pub(super) fn partials_live(&self, table: u32) -> usize {
        if table == NONE {
            0
        } else {
            self.word(table + 1).load(Ordering::Relaxed) as usize
        }
    }

    /// Live `(worker, cutoff)` entries, in slot order.
    pub(super) fn partial_entries(&self, table: u32) -> Vec<(u32, u32)> {
        if table == NONE {
            return Vec::new();
        }
        let (_, used) = self.partials_shape(table);
        self.words(table + 2, used)
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .filter(|entry| *entry != 0 && *entry != TOMB)
            .map(|entry| (entry as u32, (entry >> 32) as u32))
            .collect()
    }

    /// The slot and cutoff of `worker`, if it is a partial holder.
    pub(super) fn partial_find(&self, table: u32, worker: u32) -> Option<(usize, u32)> {
        if table == NONE {
            return None;
        }
        let (_, used) = self.partials_shape(table);
        self.words(table + 2, used)
            .iter()
            .enumerate()
            .find_map(|(index, slot)| {
                let entry = slot.load(Ordering::Relaxed);
                (entry != 0 && entry != TOMB && entry as u32 == worker)
                    .then_some((index, (entry >> 32) as u32))
            })
    }

    /// The largest cutoff among the partial holders (0 when there are none).
    pub(super) fn partial_max(&self, table: u32) -> usize {
        if table == NONE {
            return 0;
        }
        let (_, used) = self.partials_shape(table);
        self.words(table + 2, used)
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .filter(|entry| *entry != 0 && *entry != TOMB)
            .map(|entry| (entry >> 32) as usize)
            .max()
            .unwrap_or(0)
    }

    pub(super) fn partial_set(&self, table: u32, slot: usize, worker: u32, cutoff: u32) {
        self.word(table + 2 + slot as u32).store(
            u64::from(worker) | (u64::from(cutoff) << 32),
            Ordering::Release,
        );
    }

    /// Append an entry; `false` when the table is full.
    pub(super) fn partial_put(&self, table: u32, worker: u32, cutoff: u32) -> bool {
        if table == NONE {
            return false;
        }
        let (slots, used) = self.partials_shape(table);
        if used >= slots {
            return false;
        }
        self.partial_set(table, used, worker, cutoff);
        self.word(table)
            .store(slots as u64 | ((used as u64 + 1) << 32), Ordering::Relaxed);
        self.word(table + 1).fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Tombstone a slot; returns the live count left.
    pub(super) fn partial_remove(&self, table: u32, slot: usize) -> usize {
        self.word(table + 2 + slot as u32)
            .store(TOMB, Ordering::Release);
        self.word(table + 1).fetch_sub(1, Ordering::Relaxed) as usize - 1
    }

    // ---- forwarding records: [slots | count << 32][0][(at | suffix << 32), generation; slots] ----
    // Appended under the run's lock, read without it: `count` is published with a release store
    // after the record, and a reader checks the run's version around the read. Same word count as
    // a child table of the same slot count, so the two share free lists.

    pub(super) fn alloc_forwards(&self, slots: usize) -> u32 {
        let class = table_class(slots);
        let recycled = self.free_tables[class].pop();
        let table = recycled.unwrap_or_else(|| self.bump(2 + 2 * slots));
        self.word(table + 1).store(0, Ordering::Relaxed);
        self.word(table).store(slots as u64, Ordering::Release);
        table
    }

    #[inline]
    pub(super) fn forwards_shape(&self, table: u32) -> (usize, usize) {
        let header = self.word(table).load(Ordering::Acquire);
        (header as u32 as usize, (header >> 32) as usize)
    }

    /// The records of a table, oldest first.
    pub(super) fn forward_records(&self, table: u32) -> Vec<(u32, u32, u32)> {
        if table == NONE {
            return Vec::new();
        }
        let (_, count) = self.forwards_shape(table);
        let words = self.words(table + 2, 2 * count);
        (0..count)
            .map(|index| {
                let first = words[2 * index].load(Ordering::Relaxed);
                (
                    first as u32,
                    (first >> 32) as u32,
                    words[2 * index + 1].load(Ordering::Relaxed) as u32,
                )
            })
            .collect()
    }

    /// Where a block at `offset` of the run went: the oldest record whose split point is at or
    /// before it (later splits cut the shorter prefix).
    #[inline]
    pub(super) fn forwards_find(&self, table: u32, offset: u32) -> Option<(BlockRef, u32)> {
        if table == NONE {
            return None;
        }
        let (_, count) = self.forwards_shape(table);
        let words = self.words(table + 2, 2 * count);
        (0..count).find_map(|index| {
            let first = words.get(2 * index)?.load(Ordering::Relaxed);
            let at = first as u32;
            if at > offset {
                return None;
            }
            Some((
                BlockRef {
                    run: (first >> 32) as u32,
                    offset: offset - at,
                },
                words.get(2 * index + 1)?.load(Ordering::Relaxed) as u32,
            ))
        })
    }

    /// Append a record; `false` when the table is full.
    pub(super) fn forwards_push(&self, table: u32, at: u32, suffix: u32, generation: u32) -> bool {
        if table == NONE {
            return false;
        }
        let (slots, count) = self.forwards_shape(table);
        if count >= slots {
            return false;
        }
        let words = self.words(table + 2, 2 * slots);
        words[2 * count].store(u64::from(at) | (u64::from(suffix) << 32), Ordering::Relaxed);
        words[2 * count + 1].store(u64::from(generation), Ordering::Relaxed);
        self.word(table)
            .store(slots as u64 | ((count as u64 + 1) << 32), Ordering::Release);
        true
    }

    /// A new table with the records of `table` and room for more.
    pub(super) fn forwards_grown(&self, table: u32) -> u32 {
        let records = self.forward_records(table);
        let mut slots = MIN_TABLE_SLOTS;
        while slots < 2 * (records.len() + 1) {
            slots *= 2;
        }
        let grown = self.alloc_forwards(slots);
        for (at, suffix, generation) in records {
            self.forwards_push(grown, at, suffix, generation);
        }
        grown
    }

    /// A partial table holding `entries`; `NONE` for none.
    pub(super) fn partials_from(&self, entries: &[(u32, u32)]) -> u32 {
        if entries.is_empty() {
            return NONE;
        }
        let mut slots = MIN_TABLE_SLOTS;
        while slots < 2 * entries.len() {
            slots *= 2;
        }
        let table = self.alloc_partials(slots);
        for &(worker, cutoff) in entries {
            self.partial_put(table, worker, cutoff);
        }
        table
    }

    /// A new table with the live entries of `table` and room for `extra` more.
    pub(super) fn partials_grown(&self, table: u32, extra: usize) -> u32 {
        let entries = self.partial_entries(table);
        let needed = entries.len() + extra;
        let mut slots = MIN_TABLE_SLOTS;
        while slots < 2 * needed {
            slots *= 2;
        }
        let grown = self.alloc_partials(slots);
        for (worker, cutoff) in entries {
            self.partial_put(grown, worker, cutoff);
        }
        grown
    }
}

#[inline]
pub(super) fn pack(used: u32, capacity: u32) -> u64 {
    (u64::from(capacity) << 32) | u64::from(used)
}

#[inline]
pub(super) fn unpack(header: u64) -> (u32, u32) {
    (header as u32, (header >> 32) as u32)
}
