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
//! [`MIN_SLOTS`] and grows as the thread meets pieces (to
//! `FIRST_GROWTH_SLOTS` at first, then by four up to `GROW_BY_FOUR_UP_TO`
//! slots, by two above), up to the slots for the
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
pub(super) const INLINE_IDS: usize = 4;
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

/// Slots a thread's table grows to in its first step, inside its
/// reservation: about 25 KiB written sooner, so that no step places the
/// entries again at 192 stored pieces. A table without the reservation
/// grows by four from the start.
const FIRST_GROWTH_SLOTS: usize = 1_024;
/// Slots up to which a table grows by four at a time (the entries placed
/// again over that growth are a fourth of what doublings place); above them
/// it doubles. Above 4,096 slots the sizes are the ones doubling gives;
/// up to it, a table that has just grown by four holds up to about 5.3
/// slots per entry (four times the slots it had at its fill bound), at
/// most 2,048 slots, about 66 KiB, more than doubling would, per thread.
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
            Some(window) => Some(Self::of_window(window, len)),
            None => Self::of(text.get(start..end)?),
        }
    }

    /// [`Self::at`] by its sixteen-byte read alone: `None` also for a piece
    /// with fewer than sixteen bytes of the text from its start on, which
    /// [`Self::at`] reads on its own. For a probe compiled into a loop, which
    /// keeps the one read there and asks [`Self::at`] off the loop.
    #[inline(always)]
    pub(super) fn in_window(text: &[u8], start: usize, end: usize) -> Option<Self> {
        let len = end.checked_sub(start)?;
        if len == 0 || len > SHORT_PIECE_MAX_BYTES {
            return None;
        }
        let window = text.get(start..)?.first_chunk::<16>()?;
        Some(Self::of_window(window, len))
    }

    /// The key of the piece of `len` bytes, `1..=15`, that `window` opens
    /// with: the word masked to the length.
    #[inline(always)]
    fn of_window(window: &[u8; 16], len: usize) -> Self {
        let word = u128::from_le_bytes(*window);
        Self((word & KEEP_LOW[len]) | ((len as u128) << LENGTH_SHIFT))
    }

    /// The bytes of the piece the key stands for, in the low fifteen of the
    /// sixteen, and its length: the piece itself, for a caller that kept
    /// the key and no longer has the text.
    pub(super) fn bytes(self) -> ([u8; 16], usize) {
        (
            self.0.to_le_bytes(),
            (self.0 >> LENGTH_SHIFT) as usize & 0xF,
        )
    }

    /// Which of `parts` tables (a power of two) a key belongs to, by bits
    /// of the hash that neither the slot nor the tag of a table reads (the
    /// low bits and the top seven), so the keys of one part spread over its
    /// slots and tags like the keys of a whole table.
    #[inline]
    pub(super) fn part(self, parts: usize) -> usize {
        let low = self.0 as u64 ^ LOW_SALT;
        let high = (self.0 >> 64) as u64 ^ HIGH_SALT;
        let product = u128::from(low) * u128::from(high);
        let hash = (product as u64) ^ ((product >> 64) as u64);
        (hash >> 36) as usize & (parts - 1)
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
pub(super) struct Entry {
    key: u128,
    ids: [u32; INLINE_IDS],
}

impl Entry {
    const EMPTY: Self = Self {
        key: 0,
        ids: [0; INLINE_IDS],
    };

    /// Appends the entry's ids to `out` when it holds them inline, all four
    /// and then the length set to the count, and answers true; false, with
    /// nothing appended, for an entry whose ids are in the spill.
    #[inline(always)]
    fn append_inline(&self, out: &mut Vec<u32>) -> bool {
        let count = (self.key >> COUNT_SHIFT) as usize;
        if count > INLINE_IDS {
            return false;
        }
        out.extend_from_slice(&self.ids);
        out.truncate(out.len() - (INLINE_IDS - count));
        true
    }
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

    /// The table in the slots [`Self::for_entries`] gives the entries it
    /// holds, and no reservation beyond them. One that inserts filled to its
    /// fill bound stands in half those slots and grows once more. One with
    /// more slots is made anew, every entry stored again in slot order (a
    /// piece keeps its ids; it may lie nearer to its home slot or further
    /// from it than an insert in id order puts it).
    pub(super) fn fitted(mut self) -> Self {
        let slots = Self::slots_for(self.len);
        if slots >= self.slots() {
            if slots > self.slots() {
                self.grow(slots);
            }
            self.tags.shrink_to_fit();
            self.entries.shrink_to_fit();
            return self;
        }
        let mut fitted = Self::for_entries(self.len);
        let mut ids = Vec::with_capacity(MAX_IDS);
        for (slot, entry) in self.entries.iter().enumerate() {
            if self.tags[slot] != EMPTY_TAG {
                ids.clear();
                self.append(entry, &mut ids);
                fitted.insert(ShortPieceKey(entry.key & KEY_MASK), &ids, self.len);
            }
        }
        fitted
    }

    /// The slots a table for about `entries` pieces has: a power of two that
    /// holds them under the fill bound, [`MIN_SLOTS`] at least.
    pub(super) const fn slots_for(entries: usize) -> usize {
        let slots = (entries * 4 / 3 + 1).next_power_of_two();
        if slots < MIN_SLOTS {
            MIN_SLOTS
        } else {
            slots
        }
    }

    /// A table of `slots` slots with room for `reserved` before it has to
    /// move; the room is capacity, not written.
    pub(super) fn with_slots(slots: usize, reserved: usize) -> Self {
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

    /// Reads one tag in 64 and one id word of every second entry, a load
    /// a cache line, and keeps nothing, so that the lines come in as two
    /// streams rather than one dependent probe at a time
    /// (`PieceCache::reads_back` says when).
    #[inline(never)]
    pub(super) fn read_through(&self) {
        const LINE_STEP: usize = 64 / size_of::<Entry>();
        const _: () = assert!(LINE_STEP > 0, "an entry wider than a line reads none");
        let mut sum = 0u64;
        for tag in self.tags.iter().step_by(64) {
            sum = sum.wrapping_add(u64::from(*tag));
        }
        for entry in self.entries.iter().step_by(LINE_STEP) {
            sum = sum.wrapping_add(u64::from(entry.ids[0]));
        }
        // The loads stay: the sum is observed.
        std::hint::black_box(sum);
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
        self.home(Self::hash_of(key))
    }

    /// [`Self::first_slot`] from the key's hash ([`Self::hash_of`]): the one
    /// place that reads a slot and a tag out of a hash.
    #[inline(always)]
    fn home(&self, hash: u64) -> (usize, u8) {
        (hash as usize & self.mask, (hash >> 57) as u8 | 0x80)
    }

    /// The hash a probe of `key` begins from, in every table: the folded
    /// product of the key's salted halves. For a caller that holds a piece
    /// between the read of its lines and its probe.
    #[inline(always)]
    pub(super) fn hash_of(key: ShortPieceKey) -> u64 {
        let low = key.0 as u64 ^ LOW_SALT;
        let high = (key.0 >> 64) as u64 ^ HIGH_SALT;
        let product = u128::from(low) * u128::from(high);
        (product as u64) ^ ((product >> 64) as u64)
    }

    /// [`Self::touch`] from the key's hash ([`Self::hash_of`]).
    #[inline(always)]
    pub(super) fn touch_hashed(&self, hash: u64) -> u8 {
        let (slot, _) = self.home(hash);
        self.tags[slot] ^ (self.entries[slot].key as u8)
    }

    /// [`Self::lookup`] from the key's hash ([`Self::hash_of`]), with the
    /// walk of [`Self::probe`] written out: one walk for both cost every
    /// loop that compiles the probe in 0.5 to 0.9 instructions a token
    /// (counted).
    #[inline(always)]
    pub(super) fn lookup_hashed(
        &self,
        key: ShortPieceKey,
        hash: u64,
        out: &mut Vec<u32>,
    ) -> Option<Vacancy> {
        debug_assert_eq!(hash, Self::hash_of(key));
        let (mut slot, tag) = self.home(hash);
        loop {
            let found = self.tags[slot];
            if found == tag {
                let entry = &self.entries[slot];
                if entry.key & KEY_MASK == key.0 {
                    self.append(entry, out);
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

    /// Appends the ids of `key` to `out` and answers whether the table had
    /// the piece.
    #[cfg(test)]
    pub(super) fn ids_into(&self, key: ShortPieceKey, out: &mut Vec<u32>) -> bool {
        self.lookup(key, out).is_none()
    }

    /// The probe as a lookup: a hit appends the piece's ids
    /// ([`Self::append`]) and returns `None`; a miss returns where the probe
    /// ended, for the insert that follows it.
    #[inline(always)]
    pub(super) fn lookup(&self, key: ShortPieceKey, out: &mut Vec<u32>) -> Option<Vacancy> {
        match self.probe(key) {
            Ok(entry) => {
                self.append(entry, out);
                None
            }
            Err(vacancy) => Some(vacancy),
        }
    }

    /// [`Self::lookup`] for a probe compiled into a loop, which keeps only
    /// the answer of an entry with inline ids there: such a hit appends the
    /// ids. `Err` for every other piece: with the vacancy of a miss, or with
    /// `None` for an entry whose ids are in the spill, which
    /// [`Self::lookup`] appends off the loop.
    #[inline(always)]
    pub(super) fn lookup_inline(
        &self,
        key: ShortPieceKey,
        out: &mut Vec<u32>,
    ) -> Result<(), Option<Vacancy>> {
        match self.probe(key) {
            Ok(entry) if entry.append_inline(out) => Ok(()),
            Ok(_) => Err(None),
            Err(vacancy) => Err(Some(vacancy)),
        }
    }

    /// The probe: the entry of `key`, or where the walk ended on an empty
    /// slot when the table does not hold the piece.
    #[inline(always)]
    pub(super) fn probe(&self, key: ShortPieceKey) -> Result<&Entry, Vacancy> {
        let (mut slot, tag) = self.first_slot(key);
        loop {
            let found = self.tags[slot];
            if found == tag {
                let entry = &self.entries[slot];
                if entry.key & KEY_MASK == key.0 {
                    return Ok(entry);
                }
            } else if found == EMPTY_TAG {
                return Err(Vacancy {
                    key,
                    slot: slot as u32,
                    mask: self.mask as u32,
                    tag,
                });
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// The id of `entry`, an entry of this table, when it stands for exactly
    /// one, as a vocabulary word's does; else `None`, with the entry's ids
    /// appended to `out` ([`Self::append`]).
    #[inline(always)]
    pub(super) fn one_or_append(&self, entry: &Entry, out: &mut Vec<u32>) -> Option<u32> {
        if entry.key >> COUNT_SHIFT == 1 {
            return Some(entry.ids[0]);
        }
        self.append(entry, out);
        None
    }

    /// One byte of the tag and of the entry a probe of `key` begins on,
    /// folded. A caller that asks this of several keys one after the other
    /// has their far lines read together, where whole probes read them one
    /// by one (see `Native::answer_waiting`).
    #[inline(always)]
    pub(super) fn touch(&self, key: ShortPieceKey) -> u8 {
        let (slot, _) = self.first_slot(key);
        self.tags[slot] ^ (self.entries[slot].key as u8)
    }

    /// Appends the ids of `entry`, an entry of this table, to `out`: all
    /// four inline ids, then the length set to the count; a spilled entry's
    /// from the spill.
    #[inline(always)]
    pub(super) fn append(&self, entry: &Entry, out: &mut Vec<u32>) {
        if !entry.append_inline(out) {
            let count = (entry.key >> COUNT_SHIFT) as usize;
            let start = entry.ids[0] as usize;
            out.extend_from_slice(&self.spill[start..start + count]);
        }
    }

    /// Stores a copy of `entry`, another table's entry for the key a probe
    /// of this table has just missed, at the probe's vacancy, and answers
    /// true: the key, the count and the inline ids of an entry read the same
    /// in every table, so the copy is the insert. False, with nothing stored,
    /// for an entry whose ids are in the other table's spill, for a table
    /// grown since the probe or at its fill bound, and for a slot taken
    /// since: [`Self::insert_at`] takes those. As for [`Self::insert_at`],
    /// the vacancy must be of this table as it is, with no [`Self::clear`]
    /// between the probe that made it and this call: a cleared table keeps
    /// its slots, and the copy would sit off its key's probe path.
    #[inline(always)]
    pub(super) fn keep(&mut self, vacancy: Vacancy, entry: &Entry) -> bool {
        let slot = vacancy.slot as usize;
        if (entry.key >> COUNT_SHIFT) as usize > INLINE_IDS
            || vacancy.mask as usize != self.mask
            || self.len >= self.fill_bound()
            || self.tags[slot] != EMPTY_TAG
        {
            return false;
        }
        debug_assert!(
            self.occupied_up_to(vacancy.key, slot),
            "a vacancy of the table as it is: no clear between the probe and the copy"
        );
        self.tags[slot] = vacancy.tag;
        self.entries[slot] = *entry;
        self.len += 1;
        true
    }

    /// Stores `ids` as the piece's encoding and answers true. A table at its
    /// fill bound with fewer slots than `bound` entries take grows first
    /// (in the steps of `grow`: more than doubling while small, up to
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

    /// Grows the slots (in the first step to `FIRST_GROWTH_SLOTS` where the
    /// table reserved them, else by four up to `GROW_BY_FOUR_UP_TO`, by two
    /// above; to `most` at the most) in place, into the table's reservation
    /// (past the reservation it moves, as a `Vec` does) and every stored
    /// entry is placed again where it stands ([`Self::place_again`]). A tag
    /// does not depend on the slot count, so the tags move with their
    /// entries.
    fn grow(&mut self, most: usize) {
        let old = self.entries.len();
        let slots = if old == MIN_SLOTS && self.backing() >= FIRST_GROWTH_SLOTS {
            FIRST_GROWTH_SLOTS
        } else if old < GROW_BY_FOUR_UP_TO {
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
    /// whole. One walk lifts each stored entry and probes it in from its new
    /// home, which is the old home or the same slot of a later block of
    /// `old` slots (the mask grew by whole bits). The walk starts at an
    /// empty slot, so it cuts no run in two and comes to every entry after
    /// its old home and the slots between them. A probe therefore meets
    /// only slots that are final. Among the old slots those are the ones
    /// the walk has passed: the slot just lifted ends the probe at the
    /// latest, or the old slots' end does when the entry's run wrapped
    /// around it, and the probe runs on into the next block. In a later
    /// block only placed entries stand, each at or before the slot that
    /// mirrors its old one, so no probe runs off the table's end before the
    /// walk's last stretch, the slots ahead of its start, and one that does
    /// comes back to the first old slots when the walk has passed them.
    fn place_again(&mut self, old: usize) {
        let start = self.tags[..old]
            .iter()
            .position(|&tag| tag == EMPTY_TAG)
            .unwrap_or(0);
        for step in 0..old {
            let slot = (start + step) & (old - 1);
            let tag = self.tags[slot];
            if tag == EMPTY_TAG {
                continue;
            }
            let entry = self.entries[slot];
            let (mut to, _) = self.first_slot(ShortPieceKey(entry.key & KEY_MASK));
            self.tags[slot] = EMPTY_TAG;
            while self.tags[to] != EMPTY_TAG {
                to = (to + 1) & self.mask;
            }
            self.tags[to] = tag;
            self.entries[to] = entry;
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

    /// The key by the sixteen-byte read alone is the key read from the text
    /// wherever sixteen bytes of the text remain from the piece's start on,
    /// and `None` nearer the end, where [`ShortPieceKey::at`] reads the piece
    /// on its own.
    #[test]
    fn a_key_by_the_one_read_is_the_texts_key_or_none_near_the_end() {
        let text: Vec<u8> = (0..40u8)
            .map(|b| b.wrapping_mul(37).wrapping_add(11))
            .collect();
        for start in 0..=text.len() {
            for len in 0..=17 {
                let end = (start + len).min(text.len());
                let expected = if text.len() - start >= 16 {
                    ShortPieceKey::at(&text, start, end)
                } else {
                    None
                };
                assert_eq!(
                    ShortPieceKey::in_window(&text, start, end),
                    expected,
                    "{start}+{len}"
                );
            }
        }
        assert_eq!(ShortPieceKey::in_window(&text, 5, 4), None);
    }

    #[test]
    fn a_fitted_table_stands_in_the_slots_for_its_pieces_and_reserves_none() {
        // Made for 48 pieces and given one; made small in a reservation and
        // given one; the same given 48, which fill its 64 slots to the
        // bound: it grows to the slots for 48, not into the reservation.
        // A piece has more ids than an entry holds.
        let small = || ShortPieceTable::with_slots(MIN_SLOTS, FIRST_GROWTH_SLOTS);
        let fitted = [
            (ShortPieceTable::for_entries(48), 1),
            (small(), 1),
            (small(), 48),
        ]
        .map(|(mut table, pieces)| {
            for i in 0..pieces {
                table.insert(key(&word(i)), &[1, 2, 3, 4, 5], 48);
            }
            let table = table.fitted();
            let ids = ids_of(&table, &word(0));
            (table.slots(), table.backing(), table.tags.capacity(), ids)
        });
        let ids = Some(vec![1, 2, 3, 4, 5]);
        let one = (MIN_SLOTS, MIN_SLOTS, MIN_SLOTS, ids.clone());
        assert_eq!(fitted, [one.clone(), one, (128, 128, 128, ids)]);
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

    /// An entry of one id answers the id and appends nothing; an entry of
    /// more appends its ids.
    #[test]
    fn an_entry_of_one_id_answers_it_and_one_of_more_appends_its_ids() {
        let mut table = ShortPieceTable::new();
        assert!(table.insert(key(b"a"), &[7], 3));
        assert!(table.insert(key(b"b"), &[1, 2], 3));
        let mut out = vec![9];
        let entry = table.probe(key(b"a")).ok().expect("stored");
        assert_eq!(table.one_or_append(entry, &mut out), Some(7));
        assert_eq!(out, [9]);
        let entry = table.probe(key(b"b")).ok().expect("stored");
        assert_eq!(table.one_or_append(entry, &mut out), None);
        assert_eq!(out, [9, 1, 2]);
    }

    /// A key gives back the bytes and the length of its piece, for every
    /// length.
    #[test]
    fn a_key_gives_back_the_bytes_of_its_piece() {
        let piece: Vec<u8> = (1..=15u8).map(|b| b.wrapping_mul(37)).collect();
        for len in 1..=15 {
            let (bytes, length) = key(&piece[..len]).bytes();
            assert_eq!(&bytes[..length], &piece[..len], "{len}");
        }
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

    /// The lookup a loop keeps answers an entry with inline ids as
    /// [`ShortPieceTable::lookup`] does. An entry with spilled ids comes
    /// back with nothing appended and no vacancy, and a miss with the
    /// vacancy the insert takes.
    #[test]
    fn an_inline_lookup_appends_inline_ids_and_hands_the_other_pieces_back() {
        let mut table = ShortPieceTable::for_entries(4);
        assert!(table.insert(key(b"abcd"), &[1, 2, 3, 4], 4));
        assert!(table.insert(key(b"abcde"), &[1, 2, 3, 4, 5], 4));
        let mut out = vec![9];
        assert!(table.lookup_inline(key(b"abcd"), &mut out).is_ok());
        assert_eq!(out, [9, 1, 2, 3, 4]);
        assert!(matches!(
            table.lookup_inline(key(b"abcde"), &mut out),
            Err(None)
        ));
        assert_eq!(out, [9, 1, 2, 3, 4]);
        let Err(Some(vacancy)) = table.lookup_inline(key(b"new"), &mut out) else {
            panic!("a miss with its vacancy");
        };
        assert_eq!(out, [9, 1, 2, 3, 4]);
        assert!(table.insert_at(vacancy, &[7], 4));
        assert_eq!(ids_of(&table, b"new"), Some(vec![7]));
        assert!(table.lookup(key(b"abcde"), &mut out).is_none());
        assert_eq!(out, [9, 1, 2, 3, 4, 1, 2, 3, 4, 5]);
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

    /// A table: the fewest slots at first, grown at the fill bound
    /// while `bound` entries take more slots (by four while small, by two
    /// above `GROW_BY_FOUR_UP_TO`), every piece kept across the growths (the
    /// spilled ones too), and at the slots for `bound` the fill bound refuses
    /// and a probe still ends.
    #[test]
    fn a_table_starts_small_grows_by_four_then_doubles_and_refuses_at_its_bound() {
        let mut table = ShortPieceTable::for_entries(0);
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
    fn a_missed_piece_is_kept_as_a_copy_of_another_tables_entry() {
        let mut seed = ShortPieceTable::for_entries(8);
        assert!(seed.insert(key(b"word"), &[7, 8], 8));
        assert!(seed.insert(key(b"spilled"), &[1, 2, 3, 4, 5], 8));
        let mut table = ShortPieceTable::new();
        let mut out = vec![9];
        let vacancy = table.lookup(key(b"word"), &mut out).expect("a miss");
        let Ok(entry) = seed.probe(key(b"word")) else {
            panic!("the seed holds the piece");
        };
        assert!(table.keep(vacancy, entry));
        seed.append(entry, &mut out);
        assert_eq!(out, [9, 7, 8]);
        assert_eq!(table.len(), 1);
        assert_eq!(ids_of(&table, b"word"), Some(vec![7, 8]));
        // Ids in the other table's spill are not a copy to keep.
        let vacancy = table.lookup(key(b"spilled"), &mut out).expect("a miss");
        let Ok(spilled) = seed.probe(key(b"spilled")) else {
            panic!("the seed holds the piece");
        };
        assert!(!table.keep(vacancy, spilled));
        assert_eq!(table.len(), 1);
        assert_eq!(ids_of(&table, b"spilled"), None);
    }

    #[test]
    fn a_table_at_its_fill_bound_keeps_no_copy_and_the_insert_grows_it() {
        let mut seed = ShortPieceTable::for_entries(8);
        assert!(seed.insert(key(b"word"), &[7], 8));
        let mut table = ShortPieceTable::new();
        let bound = 1000;
        let mut stored = 0usize;
        while table.len() < table.fill_bound() {
            assert!(table.insert(key(&stored.to_le_bytes()), &[stored as u32], bound));
            stored += 1;
        }
        let slots = table.slots();
        let mut out = Vec::new();
        let vacancy = table.lookup(key(b"word"), &mut out).expect("a miss");
        let Ok(entry) = seed.probe(key(b"word")) else {
            panic!("the seed holds the piece");
        };
        assert!(!table.keep(vacancy, entry));
        assert_eq!(ids_of(&table, b"word"), None);
        assert!(table.insert_at(vacancy, &[7], bound));
        assert!(table.slots() > slots);
        assert_eq!(ids_of(&table, b"word"), Some(vec![7]));
    }

    #[test]
    fn a_vacancy_taken_at_the_fill_bound_inserts_by_probing_and_the_table_grows() {
        let mut table = ShortPieceTable::for_entries(0);
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
        assert_eq!(sizes, [64, 1024, 4096, 8192, 16_384, 32_768]);
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

    /// A run that wrapped around the old slots' end is placed again whole,
    /// in a step by four and in a thread table's first step: for homes that
    /// stay behind that end, for homes at the grown table's end, where the
    /// run wraps again, and for a run that mixes the two, the one a walk
    /// from the first slot loses entries of.
    #[test]
    fn a_run_wrapped_around_the_old_end_reads_back_after_growth() {
        for (slots, homes) in [
            (256, &[62, 63][..]),
            (256, &[254, 255]),
            (256, &[62, 63, 254, 255]),
            (FIRST_GROWTH_SLOTS, &[62, 63]),
            (FIRST_GROWTH_SLOTS, &[1022, 1023]),
            (FIRST_GROWTH_SLOTS, &[62, 63, 1022, 1023]),
        ] {
            let grown = ShortPieceTable::with_slots(slots, slots);
            let slot_of = |i: u32| grown.first_slot(key(&i.to_le_bytes())).0;
            let wrapped: Vec<u32> = (0..40_000)
                .filter(|&i| homes.contains(&slot_of(i)))
                .take(6)
                .collect();
            assert_eq!(wrapped.len(), 6);
            let mut table = if slots == FIRST_GROWTH_SLOTS {
                ShortPieceTable::new()
            } else {
                ShortPieceTable::for_entries(0)
            };
            for &i in &wrapped {
                assert!(table.insert(key(&i.to_le_bytes()), &[i], 1000));
            }
            assert_ne!(table.tags[3], EMPTY_TAG, "the run wrapped to the start");
            let mut stored = wrapped.clone();
            let mut rest = (0..40_000).filter(|&i| slot_of(i) % MIN_SLOTS < 62);
            while table.slots() == MIN_SLOTS {
                let i = rest.next().expect("pieces enough to grow the table");
                assert!(table.insert(key(&i.to_le_bytes()), &[i], 1000));
                stored.push(i);
            }
            assert_eq!(table.slots(), slots);
            for &i in &stored {
                assert_eq!(ids_of(&table, &i.to_le_bytes()), Some(vec![i]), "{i}");
            }
        }
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
