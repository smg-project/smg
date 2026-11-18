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
//! The table is the leading term's alone: a rise of the term starts the
//! blocks again and clears it ([`Blocks::rise`]). So an id names its file
//! by one rule ([`Blocks::file_of`]): an id of the leading term by the
//! table, any other id by its block modulo the files.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::lease::LeaseId;

/// The bits of an id's counter that count inside a block.
const BLOCK_BITS: u32 = 24;

/// The ids of one block.
#[cfg(any(test, feature = "controller"))]
const BLOCK: u32 = 1 << BLOCK_BITS;

/// The blocks of one term's counter.
const BLOCKS: u32 = 1 << (u32::BITS - BLOCK_BITS);

/// The blocks of the leading term's counter that the files claimed, and
/// each block's file.
#[derive(Debug)]
pub(super) struct Blocks {
    /// The leading term over the blocks claimed under it. One word: a
    /// claim is one atomic step, and counts under no term but the leading.
    claimed: AtomicU64,
    /// Each block's file, written by the block's claim.
    files: [AtomicU8; BLOCKS as usize],
}

impl Default for Blocks {
    fn default() -> Self {
        Self {
            claimed: AtomicU64::new(0),
            files: [const { AtomicU8::new(0) }; BLOCKS as usize],
        }
    }
}

impl Blocks {
    /// The word of `term` with no block claimed. A term past 32 bits
    /// leads no blocks: no id is minted under a term past 31.
    fn word(term: u64) -> u64 {
        u64::from(u32::try_from(term).unwrap_or(u32::MAX)) << u32::BITS
    }

    /// Claims the next block of `term`'s counter for `file`, under that
    /// file's turn: the block's number, its byte written. `None` where
    /// `term` does not lead the blocks or its last block is claimed: the
    /// file mints no further id under it.
    #[cfg(any(test, feature = "controller"))]
    pub(super) fn claim(&self, term: u64, file: u8) -> Option<u8> {
        let leading = Self::word(term);
        let next = |word: u64| {
            let claimed = word.checked_sub(leading)?;
            (claimed < u64::from(BLOCKS)).then_some(word + 1)
        };
        let claimed = &self.claimed;
        let word = claimed.try_update(Ordering::Relaxed, Ordering::Relaxed, next);
        // The blocks claimed before this one are its number.
        let block = u8::try_from(word.ok()? - leading).ok()?;
        let named = self.files.get(usize::from(block))?;
        named.store(file, Ordering::Release);
        Some(block)
    }

    /// The term rose to `term`, under every file's turn: the blocks start
    /// again under it and the table is cleared.
    pub(super) fn rise(&self, term: u64) {
        self.claimed.store(Self::word(term), Ordering::Release);
        for named in &self.files {
            named.store(0, Ordering::Release);
        }
    }

    /// The file `lease` names among `files` of them: an id of the leading
    /// term by its block's byte, any other id by its block modulo the
    /// files. Read with no turn: the leading term and one byte.
    pub(super) fn file_of(&self, lease: LeaseId, files: usize) -> usize {
        let (term, block) = (lease.0 >> u32::BITS, Self::block_of(lease));
        let leading = self.claimed.load(Ordering::Acquire) >> u32::BITS;
        let named = if term == leading {
            let named = self.files.get(usize::from(block));
            named.map_or(0, |named| named.load(Ordering::Acquire))
        } else {
            block
        };
        usize::from(named).checked_rem(files).unwrap_or(0)
    }

    /// The block of `lease`'s counter.
    fn block_of(lease: LeaseId) -> u8 {
        (lease.0 >> BLOCK_BITS) as u8
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
        self.claimed
            .store(Self::word(term) + claimed, Ordering::Release);
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
        sync::atomic::{AtomicU64, Ordering},
        thread,
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

    /// A rise starts the blocks again and clears the table: the term that
    /// led claims nothing more, the new term's first id is 1 again, and no
    /// byte the old term wrote names a file under the new one.
    #[test]
    fn a_rise_starts_the_blocks_again_and_clears_the_table() {
        let blocks = Blocks::default();
        let mut file = Minted::default();
        assert_eq!(file.mint(&blocks, 0, 1), Some(1));
        file.left = 0;
        assert_eq!(file.mint(&blocks, 0, 1), Some(BLOCK));
        assert_eq!(blocks.file_of(id(0, BLOCK), 2), 1, "by the table");

        blocks.rise(5);
        assert_eq!(blocks.claim(0, 1), None, "a term that led claims nothing");
        // The file holds ids of the term that led: under the new term it
        // claims, and the counter starts again.
        assert_eq!(file.mint(&blocks, 5, 1), Some(1));
        assert_eq!(blocks.file_of(id(5, 1), 2), 1, "the new term's claim");
        assert_eq!(blocks.file_of(id(5, BLOCK), 2), 0, "the old byte is gone");
    }

    /// An id of another term than the leading one names its file by its
    /// block modulo the files, a gateway's own id among them; with one
    /// file every id names it.
    #[test]
    fn an_id_of_another_term_names_its_file_by_its_block() {
        let blocks = Blocks::default();
        blocks.rise(9);
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
    /// once, whichever of them takes it.
    #[test]
    fn two_files_that_claim_at_once_take_each_block_once() {
        const TERMS: u64 = 200;
        let blocks = Blocks::default();
        let (term, taken) = (AtomicU64::new(0), AtomicU64::new(0));
        let mut totals = Vec::new();
        thread::scope(|scope| {
            for file in [0_u8, 1] {
                let (blocks, term, taken) = (&blocks, &term, &taken);
                scope.spawn(move || {
                    for under in 1..=TERMS {
                        while term.load(Ordering::Acquire) < under {
                            thread::yield_now();
                        }
                        let claims = std::iter::from_fn(|| blocks.claim(under, file));
                        // The claims of this file, and its end of the term.
                        let claimed = (claims.count() as u64) << u32::BITS | 1;
                        taken.fetch_add(claimed, Ordering::AcqRel);
                    }
                });
            }
            for under in 1..=TERMS {
                blocks.rise(under);
                taken.store(0, Ordering::Release);
                term.store(under, Ordering::Release);
                let both = loop {
                    let taken = taken.load(Ordering::Acquire);
                    if taken & u64::from(u32::MAX) == 2 {
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
}
