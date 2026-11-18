//! The term's counter in blocks: how a lease's id names its file with no
//! split of the counter.
//!
//! An id is its term over a counter of 32 bits (`Leases::mint`). A file
//! claims the counter a block at a time, [`BLOCKS`] blocks a term of 2^24
//! ids each, and mints from the block it holds under its own turn. One
//! table of a byte a block names each block's file, so a frame finds its
//! lease's file by one byte. A file's bits in every id would spend a
//! term's ids as many times sooner as there are files, on a fleet of one
//! model as well.
//!
//! A claim is one atomic step of the term's block count, made under the
//! claiming file's turn, and the block's byte is written before the
//! block's first id leaves that turn. The blocks are claimed in their
//! order: one file mints the counter from 1 on, as it did before the
//! blocks, and three quarters of the counter, the term's mark, is the
//! claim of block 192. The claim past the last block fails: the term is
//! spent.
//!
//! An id's file never changes while its lease can be on file. A rise of
//! the term starts the blocks again under the new term and keeps the table
//! it leaves while a lease of that term is on file ([`Blocks::rise`]): at
//! a rise the kept tables are at most the terms that hold a lease on
//! file, and a table whose last lease ended goes at the next rise. So an
//! id names its file by one rule ([`Blocks::file_of`]): an id of the
//! leading term by the leading table, an id of a kept term by that term's
//! table, any other id by its block modulo the files.
//!
//! The law covers the ids this replica mints and the ids of terms that
//! led. An id of the leading term that some other door filed stands in a
//! block no file here claimed: it goes by that block's cleared byte, and
//! the claim of the block would name it another file. A gateway's replay
//! brings ids of older terms, so a served controller files none; with one
//! file nothing can tell, and the door that files a lease holds the law
//! once the files are several.

use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc,
};

use arc_swap::ArcSwap;

use crate::lease::LeaseId;

/// The bits of an id's counter that count inside a block.
const BLOCK_BITS: u32 = 24;

/// The ids of one block.
#[cfg(any(test, feature = "controller"))]
const BLOCK: u32 = 1 << BLOCK_BITS;

/// The blocks of one term's counter.
const BLOCKS: u32 = 1 << (u32::BITS - BLOCK_BITS);

/// The bit of the blocks' word that says a table is kept: a read of the
/// word tells whether an id of another term has a table to look for.
const KEPT: u64 = 1 << (u32::BITS - 1);

/// The blocks of the leading term's counter that the files claimed, each
/// block's file, and the tables of the terms that led and still hold a
/// lease on file.
#[derive(Debug)]
pub(super) struct Blocks {
    /// The leading term over the blocks claimed under it, [`KEPT`] between
    /// the two. One word: a claim is one atomic step, and counts under no
    /// term but the leading.
    claimed: AtomicU64,
    /// Each block's file under the leading term, written by the block's
    /// claim and cleared by a rise, in place.
    files: [AtomicU8; BLOCKS as usize],
    /// The tables kept, the newest first: replaced whole by a rise.
    kept: ArcSwap<Vec<Kept>>,
}

/// The table a term left at the rise past it, kept while a lease of the
/// term is on file.
#[derive(Clone, Debug)]
struct Kept {
    term: u64,
    files: [u8; BLOCKS as usize],
}

impl Default for Blocks {
    fn default() -> Self {
        Self {
            claimed: AtomicU64::new(0),
            files: [const { AtomicU8::new(0) }; BLOCKS as usize],
            kept: ArcSwap::from_pointee(Vec::new()),
        }
    }
}

impl Blocks {
    /// The word of `term` with no block claimed and no table kept. A term
    /// past 32 bits takes the greatest word: the mint refuses a term past
    /// 31 bits, so no claim is made under it.
    fn word(term: u64) -> u64 {
        u64::from(u32::try_from(term).unwrap_or(u32::MAX)) << u32::BITS
    }

    /// The term of `lease`, as the mint lays an id out: the half over its
    /// counter. A gateway's own id reads a term no controller leads.
    fn term_of(lease: LeaseId) -> u64 {
        lease.0 >> u32::BITS
    }

    /// The block of `lease`'s counter.
    fn block_of(lease: LeaseId) -> u8 {
        (lease.0 >> BLOCK_BITS) as u8
    }

    /// `named` among `files` files: a byte past them wraps, and no count
    /// of files stops a read.
    fn among(named: u8, files: usize) -> usize {
        usize::from(named).checked_rem(files).unwrap_or(0)
    }

    /// Claims the next block of `term`'s counter for `file`, under that
    /// file's turn: the block's number, its byte written. `None` where
    /// `term` does not lead the blocks or its last block is claimed: the
    /// file mints no further id under it.
    #[cfg(any(test, feature = "controller"))]
    pub(super) fn claim(&self, term: u64, file: u8) -> Option<u8> {
        let leading = Self::word(term);
        let next = |word: u64| {
            let claimed = (word & !KEPT).checked_sub(leading)?;
            (claimed < u64::from(BLOCKS)).then_some(word + 1)
        };
        let claimed = &self.claimed;
        let word = claimed.try_update(Ordering::Relaxed, Ordering::Relaxed, next);
        // The blocks claimed before this one are its number.
        let block = u8::try_from((word.ok()? & !KEPT) - leading).ok()?;
        let named = self.files.get(usize::from(block))?;
        named.store(file, Ordering::Release);
        Some(block)
    }

    /// The term rose to `term`, under every file's turn; `on_file` are the
    /// ids of the leases on file. The blocks start again under `term`. The
    /// table the leaving term wrote is kept where a lease of that term is
    /// on file, and a kept table whose term holds none is dropped. The
    /// kept tables are told first, then the word, and the bytes are
    /// cleared last: a read that meets the rise finds the leaving term's
    /// table among the kept ([`Self::file_of`]).
    pub(super) fn rise(&self, term: u64, on_file: impl Iterator<Item = LeaseId>) {
        let leaving = self.claimed.load(Ordering::Acquire) >> u32::BITS;
        let before = self.kept.load();
        // The terms a table can be kept for, and whether a lease on file
        // is of each: one pass over the leases, no count a term.
        let terms = std::iter::once(leaving).chain(before.iter().map(|table| table.term));
        let mut terms: Vec<(u64, bool)> = terms.map(|term| (term, false)).collect();
        for lease in on_file {
            let of = Self::term_of(lease);
            if let Some((_, held)) = terms.iter_mut().find(|(term, _)| *term == of) {
                *held = true;
            }
        }
        let held = |of: u64| terms.iter().any(|&(term, held)| held && term == of);
        let mut kept = Vec::new();
        if held(leaving) {
            let files = self.files.each_ref();
            let files = files.map(|named| named.load(Ordering::Acquire));
            kept.push(Kept {
                term: leaving,
                files,
            });
        }
        kept.extend(before.iter().filter(|table| held(table.term)).cloned());
        let flag = if kept.is_empty() { 0 } else { KEPT };
        self.kept.store(Arc::new(kept));
        self.claimed
            .store(Self::word(term) | flag, Ordering::Release);
        for named in &self.files {
            named.store(0, Ordering::Release);
        }
    }

    /// The file `lease` names among `files` of them: an id of the leading
    /// term by its block's byte in the leading table, an id of a kept term
    /// by that term's table, any other id by its block modulo the files.
    /// Read with no turn, and of one standing: the leading term is read
    /// again behind the byte, and an id whose term turned between the two
    /// reads goes to the kept tables, where the rise put its table before
    /// it wrote the word. With one file every id names it, and nothing is
    /// read.
    pub(super) fn file_of(&self, lease: LeaseId, files: usize) -> usize {
        if files <= 1 {
            return 0;
        }
        let (term, block) = (Self::term_of(lease), Self::block_of(lease));
        let mut word = self.claimed.load(Ordering::Acquire);
        if word >> u32::BITS == term {
            let named = self.files.get(usize::from(block));
            let named = named.map_or(0, |named| named.load(Ordering::Acquire));
            word = self.claimed.load(Ordering::Acquire);
            if word >> u32::BITS == term {
                return Self::among(named, files);
            }
        }
        if word & KEPT == 0 {
            return Self::among(block, files);
        }
        self.kept_file(term, block, files)
    }

    /// The file an id of `term` in `block` names while a table is kept: by
    /// the table kept for the term, the newest kept table read first, and
    /// by the block where none is kept for it. Out of line, and the last
    /// thing [`Self::file_of`] does: the read of the kept tables saves
    /// registers, and an id of the leading term would pay for them.
    #[inline(never)]
    fn kept_file(&self, term: u64, block: u8, files: usize) -> usize {
        let kept = self.kept.load();
        let table = kept.iter().find(|table| table.term == term);
        let named = table.and_then(|table| table.files.get(usize::from(block)));
        Self::among(named.copied().unwrap_or(block), files)
    }

    /// The terms whose tables are kept, the newest first.
    #[cfg(test)]
    pub(super) fn kept_terms(&self) -> Vec<u64> {
        self.kept.load().iter().map(|table| table.term).collect()
    }

    /// Spends `term`'s counter but for its last `left` ids, all for
    /// `file`: every block before the one that holds the next id is
    /// claimed, and what is returned holds that one from the next id on.
    /// A test's door: no run reaches a term's last ids in under 2^32
    /// issues.
    #[cfg(test)]
    pub(super) fn spend(&self, term: u64, left: u32, file: u8) -> Minted {
        let next = u64::from(u32::MAX - left) + 1;
        let block = next >> BLOCK_BITS;
        let claimed = (block + 1).min(u64::from(BLOCKS));
        let kept = self.claimed.load(Ordering::Acquire) & KEPT;
        let word = Self::word(term) | kept | claimed;
        self.claimed.store(word, Ordering::Release);
        for named in self.files.iter().take(claimed as usize) {
            named.store(file, Ordering::Release);
        }
        let within = next as u32 & (BLOCK - 1);
        Minted {
            term,
            next: next as u32,
            left: if left == 0 { 0 } else { BLOCK - within },
        }
    }
}

/// What one file holds of the counter, under its turn: the block it
/// claimed last, as far as it minted into it.
#[cfg(any(test, feature = "controller"))]
#[derive(Debug, Default)]
pub(super) struct Minted {
    /// The term the block was claimed under.
    term: u64,
    /// The counter of the next id.
    next: u32,
    /// The ids the block still holds: none before the first claim.
    left: u32,
}

#[cfg(any(test, feature = "controller"))]
impl Minted {
    /// The counter of `file`'s next id under `term`: the next of the block
    /// it holds, or the first of the block it claims where that one is
    /// spent or was claimed under an earlier term. Never 0: the term's
    /// first block starts at 1. `None` where no block is left to claim.
    pub(super) fn mint(&mut self, blocks: &Blocks, term: u64, file: u8) -> Option<u32> {
        if self.term != term || self.left == 0 {
            let block = u32::from(blocks.claim(term, file)?);
            // The counter 0 is no id: the first block is one id short.
            let next = (block << BLOCK_BITS).max(1);
            let left = BLOCK - (next & (BLOCK - 1));
            *self = Self { term, next, left };
        }
        let counter = self.next;
        self.next = counter.wrapping_add(1);
        self.left -= 1;
        Some(counter)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::atomic::{AtomicBool, AtomicU64, Ordering},
        thread,
        time::{Duration, Instant},
    };

    use super::{Blocks, Minted, BLOCK, BLOCKS, BLOCK_BITS};
    use crate::lease::LeaseId;

    /// The id of `term` and `counter`, as the mint lays an id out.
    fn id(term: u64, counter: u32) -> LeaseId {
        LeaseId(term << u32::BITS | u64::from(counter))
    }

    /// One file mints the counter from 1 on, as it did before the blocks:
    /// the id behind a block's last is the next block's first, and a block
    /// is claimed where its first id is minted.
    #[test]
    fn one_file_mints_the_counter_from_1_on_across_its_blocks() {
        let blocks = Blocks::default();
        let mut file = Minted::default();
        let first = [(); 3].map(|()| file.mint(&blocks, 0, 0));
        assert_eq!(first, [Some(1), Some(2), Some(3)]);
        // From the first block's last id on.
        let mut file = blocks.spend(0, u32::MAX - (BLOCK - 2), 0);
        let across = [(); 3].map(|()| file.mint(&blocks, 0, 0));
        assert_eq!(across, [Some(BLOCK - 1), Some(BLOCK), Some(BLOCK + 1)]);
        // Two blocks are claimed: another file's first id is the third's.
        assert_eq!(Minted::default().mint(&blocks, 0, 1), Some(2 * BLOCK));
    }

    /// One file mints its first block to its end and two ids on: every
    /// counter once, in its order. The first block is one id short, so
    /// its last id is the one before the second block's first, and no id
    /// is minted twice where the second block is claimed.
    #[test]
    fn one_file_mints_its_first_block_to_its_end() {
        let blocks = Blocks::default();
        let mut file = Minted::default();
        let mut last = 0_u32;
        for _ in 0..=BLOCK {
            let counter = file.mint(&blocks, 0, 0).expect("a block is left");
            assert_eq!(counter, last + 1, "the id behind {last}");
            last = counter;
        }
        assert_eq!(last, BLOCK + 1);
        // Two blocks are claimed: another file's first id is the third's.
        assert_eq!(Minted::default().mint(&blocks, 0, 1), Some(2 * BLOCK));
    }

    /// Files that mint in turn hold a block each, and every id names the
    /// file that minted it by its block's byte. A file whose block is
    /// spent claims the next block of the term, whoever holds the ones
    /// between.
    #[test]
    fn an_id_names_the_file_that_minted_it() {
        let blocks = Blocks::default();
        let mut files = [(); 3].map(|()| Minted::default());
        let mut minted = Vec::new();
        for _ in 0..4 {
            for (at, file) in (0..).zip(&mut files) {
                let counter = file.mint(&blocks, 0, at).expect("a block is left");
                minted.push((usize::from(at), id(0, counter)));
            }
        }
        for &(at, lease) in &minted {
            assert_eq!(blocks.file_of(lease, files.len()), at, "{lease:?}");
        }
        let ids: BTreeSet<u64> = minted.iter().map(|(_, lease)| lease.0).collect();
        assert_eq!(ids.len(), minted.len(), "no id twice");
        // The second file's block is spent: its next id opens the fourth.
        files[1].left = 0;
        let counter = files[1].mint(&blocks, 0, 1).expect("a block is left");
        assert_eq!(counter, 3 * BLOCK);
        assert_eq!(blocks.file_of(id(0, counter), files.len()), 1);
    }

    /// The claim past the last block fails as a spent term does: the last
    /// id of the counter is minted, and no file mints another under the
    /// term. The term's mark, three quarters of the counter, is the first
    /// id of block 192, claimed as any block is.
    #[test]
    fn the_claim_past_the_last_block_fails_as_a_spent_term() {
        let blocks = Blocks::default();
        let mut file = blocks.spend(0, 1, 0);
        assert_eq!(file.mint(&blocks, 0, 0), Some(u32::MAX));
        assert_eq!(file.mint(&blocks, 0, 0), None, "the counter is spent");
        assert_eq!(Minted::default().mint(&blocks, 0, 1), None, "for each file");
        let nothing_left = blocks.spend(0, 0, 0).mint(&blocks, 0, 0);
        assert_eq!(nothing_left, None, "spent whole");

        let (blocks, mark) = (Blocks::default(), 3_u32 << 30);
        assert_eq!(mark, 192 << BLOCK_BITS);
        let mut file = blocks.spend(0, u32::MAX - (mark - 2), 0);
        let at_the_mark = [(); 2].map(|()| file.mint(&blocks, 0, 0));
        assert_eq!(at_the_mark, [Some(mark - 1), Some(mark)]);
        // Block 192 is claimed: another file's first id is block 193's.
        assert_eq!(
            Minted::default().mint(&blocks, 0, 1),
            Some(193 << BLOCK_BITS)
        );
    }

    /// A rise starts the blocks again: the term that led claims nothing
    /// more and the new term's first id is 1 again. With no lease of the
    /// leaving term on file its table is gone: its ids go by their block
    /// modulo the files, and no byte it wrote names a file under the new
    /// term.
    #[test]
    fn a_rise_starts_the_blocks_again_and_drops_a_table_no_lease_holds() {
        let blocks = Blocks::default();
        let mut file = Minted::default();
        assert_eq!(file.mint(&blocks, 0, 1), Some(1));
        file.left = 0;
        assert_eq!(file.mint(&blocks, 0, 1), Some(BLOCK));
        assert_eq!(blocks.file_of(id(0, 1), 2), 1, "by the table");

        blocks.rise(5, std::iter::empty());
        assert_eq!(blocks.claim(0, 1), None, "a term that led claims nothing");
        // The file holds ids of the term that led: under the new term it
        // claims, and the counter starts again.
        assert_eq!(file.mint(&blocks, 5, 1), Some(1));
        assert_eq!(blocks.file_of(id(5, 1), 2), 1, "the new term's claim");
        assert_eq!(blocks.file_of(id(5, BLOCK), 2), 0, "the old byte is gone");
        assert_eq!(blocks.file_of(id(0, 1), 2), 0, "by its block: no table");
        assert_eq!(blocks.file_of(id(0, BLOCK), 2), 1, "by its block as well");
        assert!(blocks.kept.load().is_empty());
    }

    /// An id's file never changes while its lease can be on file: a lease
    /// dispatched under one term and ended by its Completed after the same
    /// replica took the next is found in the file that minted it. A rise
    /// keeps the table it leaves while a lease of that term is on file,
    /// through the rises that follow, and the kept tables are no more than
    /// the terms that hold a lease.
    #[test]
    fn a_rise_keeps_the_table_of_a_term_that_holds_a_lease() {
        let blocks = Blocks::default();
        let first = |term: u64, file: u8| {
            let counter = Minted::default().mint(&blocks, term, file);
            id(term, counter.expect("the term's first block"))
        };
        // By their blocks alone both ids would name the first file.
        let lease = first(0, 2);
        blocks.rise(1, [lease].into_iter());
        assert_eq!(blocks.file_of(lease, 3), 2, "kept at the rise past it");
        let next = first(1, 1);
        assert_eq!((blocks.file_of(lease, 3), blocks.file_of(next, 3)), (2, 1));
        blocks.rise(2, [lease, next].into_iter());
        assert_eq!((blocks.file_of(lease, 3), blocks.file_of(next, 3)), (2, 1));
        assert_eq!(blocks.kept.load().len(), 2, "a table a term that holds");
        // A gateway's own id on file is of no term that led.
        let own = LeaseId(1 << 63 | u64::from(BLOCK));
        blocks.rise(3, [lease, own].into_iter());
        assert_eq!((blocks.file_of(lease, 3), blocks.file_of(next, 3)), (2, 0));
        assert_eq!(blocks.kept.load().len(), 1, "the ended term's is gone");
        // Beside a kept table, an id of a term with none goes by its block.
        assert_eq!(blocks.file_of(id(1, 2 * BLOCK), 3), 2);
    }

    /// A kept table goes at the first rise that finds no lease of its term
    /// on file, and an id of a term whose table is gone is any other id:
    /// it names its file by its block modulo the files. A lease of that
    /// term restored later is filed there, where its frames then go: no
    /// rise brings the table back.
    #[test]
    fn a_kept_table_goes_at_the_first_rise_behind_its_last_lease() {
        let blocks = Blocks::default();
        let counter = Minted::default().mint(&blocks, 0, 2);
        let held = id(0, counter.expect("the term's first block"));
        blocks.rise(1, [held].into_iter());
        assert_eq!(blocks.file_of(held, 3), 2, "kept");
        blocks.rise(2, std::iter::empty());
        assert_eq!(blocks.file_of(held, 3), 0, "gone: by its block");
        blocks.rise(3, [held].into_iter());
        assert_eq!(blocks.file_of(held, 3), 0, "restored: by its block still");
        assert!(blocks.kept.load().is_empty());
    }

    /// An id of a term that neither leads nor has a kept table names its
    /// file by its block modulo the files, a gateway's own id among them;
    /// with one file every id names it.
    #[test]
    fn an_id_of_another_term_names_its_file_by_its_block() {
        let blocks = Blocks::default();
        blocks.rise(9, std::iter::empty());
        let mut file = Minted::default();
        assert_eq!(file.mint(&blocks, 9, 3), Some(1));
        let led = id(3, 5 * BLOCK + 17);
        let own = LeaseId(1 << 63 | u64::from(6 * BLOCK));
        assert_eq!(blocks.file_of(id(9, 1), 4), 3, "the leading term's table");
        assert_eq!(
            blocks.file_of(id(3, 1), 4),
            0,
            "no table for a term that led"
        );
        assert_eq!(blocks.file_of(led, 4), 1);
        assert_eq!(blocks.file_of(own, 4), 2);
        for lease in [led, own, id(9, 1)] {
            assert_eq!(blocks.file_of(lease, 1), 0, "{lease:?}");
        }
    }

    /// A claim is one atomic step of the block count: two files that claim
    /// at once take the 256 blocks of a term between them, each block
    /// once, whichever of them takes it. Every wait ends at the case's
    /// bound and a file claims no more than two terms' blocks: a claim
    /// that never fails, or a claimer that leaves, fails the case.
    #[test]
    fn two_files_that_claim_at_once_take_each_block_once() {
        const TERMS: u64 = 200;
        let end = Instant::now() + Duration::from_secs(60);
        let blocks = Blocks::default();
        let (term, taken) = (AtomicU64::new(0), AtomicU64::new(0));
        let mut totals = Vec::new();
        thread::scope(|scope| {
            for file in [0_u8, 1] {
                let (blocks, term, taken) = (&blocks, &term, &taken);
                scope.spawn(move || {
                    for under in 1..=TERMS {
                        while term.load(Ordering::Acquire) < under {
                            if Instant::now() > end {
                                return;
                            }
                            thread::yield_now();
                        }
                        let claims = std::iter::from_fn(|| blocks.claim(under, file));
                        let claims = claims.take(2 * BLOCKS as usize);
                        // The claims of this file, and its end of the term.
                        let claimed = (claims.count() as u64) << u32::BITS | 1;
                        taken.fetch_add(claimed, Ordering::AcqRel);
                    }
                });
            }
            for under in 1..=TERMS {
                blocks.rise(under, std::iter::empty());
                taken.store(0, Ordering::Release);
                term.store(under, Ordering::Release);
                let both = loop {
                    let taken = taken.load(Ordering::Acquire);
                    if taken & u64::from(u32::MAX) == 2 || Instant::now() > end {
                        break taken >> u32::BITS;
                    }
                    thread::yield_now();
                };
                totals.push(both);
            }
        });
        assert!(
            totals.iter().all(|&both| both == u64::from(BLOCKS)),
            "{totals:?}"
        );
    }

    /// A frame's read is of one standing: a reader beside a rise never
    /// reads the leaving term's id by a cleared byte. Each term's id is
    /// minted by the second file and stays on file through every rise
    /// that follows, so whichever of them the reader holds names that
    /// file whenever it is read. On one core a reader seldom lands inside
    /// a rise, and the case never fails a right read.
    #[test]
    fn a_read_beside_a_rise_is_of_one_standing() {
        /// Ends the reader where the rising thread leaves, by its end or
        /// by an unwind.
        struct Ends<'a>(&'a AtomicBool);
        impl Drop for Ends<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        const TERMS: u64 = 200;
        let blocks = Blocks::default();
        let first = |term: u64| {
            let counter = Minted::default().mint(&blocks, term, 1);
            id(term, counter.expect("the term's first block"))
        };
        let mut on_file = vec![first(0)];
        let (newest, done) = (AtomicU64::new(on_file[0].0), AtomicBool::new(false));
        let wrong = thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut wrong = 0_u64;
                while !done.load(Ordering::Acquire) {
                    let lease = LeaseId(newest.load(Ordering::Acquire));
                    wrong += u64::from(blocks.file_of(lease, 2) != 1);
                }
                wrong
            });
            let ends = Ends(&done);
            for term in 1..=TERMS {
                blocks.rise(term, on_file.iter().copied());
                let lease = first(term);
                on_file.push(lease);
                newest.store(lease.0, Ordering::Release);
            }
            drop(ends);
            reader.join().expect("the reader ends")
        });
        assert_eq!(wrong, 0, "an id read by a table that is not its term's");
    }
}
