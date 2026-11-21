//! The term's counter in blocks: how a lease's id names its file with no
//! split of the counter.
//!
//! An id is its term over a counter of 32 bits (`Leases::mint`). A file
//! claims the counter a block at a time, [`BLOCKS`] blocks a term of 2^20
//! ids each, and mints from the block it holds under its own turn. One
//! table of a byte a block names each block's file, so a frame finds its
//! lease's file by one byte. A file's bits in every id would spend a
//! term's ids as many times sooner as there are files, on a fleet of one
//! model as well. A block is small beside the counter: each file holds
//! one, so the most files a host names, 256, take a sixteenth of the
//! counter at their first ids and leave the term its span.
//!
//! A claim is made under the claiming file's turn, one claim at a time:
//! the block's byte is written, then the term's block count steps past
//! it. So a count that covers a block finds its byte written, and what
//! asks whether an id's file is settled reads the count first
//! ([`Blocks::settled`]). The blocks are claimed in their order: one file
//! mints the counter from 1 on, as it did before the blocks, and three
//! quarters of the counter, the term's mark, is the claim of block 3,072.
//! The claim past the last block fails: the term is spent.
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
//! A term past 31 bits leads no table ([`Blocks::rise`]): the mint serves
//! no id under it, and the blocks stay as the last term below it left
//! them. So a gateway's own id, whose upper half reads as such a term,
//! names its block's file whatever term the store reaches. From then on
//! no block is claimed and no term leads the blocks anew: every id names
//! for good the file it reads.
//!
//! The law covers the ids this replica mints and the ids of terms that
//! led. An id of the leading term that some other door filed stands in a
//! block no file here claimed: it goes by that block's cleared byte, and
//! the claim of the block would name it another file. A gateway's replay
//! brings ids of older terms, so a served controller files none; with one
//! file nothing can tell, and the door that files a lease holds the law
//! once the files are several.

use std::{
    cmp,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc,
    },
};

use arc_swap::ArcSwap;
use parking_lot::Mutex;

use crate::tracking::lease::LeaseId;

/// The bits of an id's counter that count inside a block: 2^20 ids a
/// block, 4,096 blocks a term.
const BLOCK_BITS: u32 = 20;

/// The ids of one block.
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
    /// the two. One word, written under the claims' lock: a claim counts
    /// under no term but the leading.
    claimed: AtomicU64,
    /// Each block's file under the leading term, written by the block's
    /// claim and cleared by a rise, in place.
    files: [AtomicU8; BLOCKS as usize],
    /// The tables kept, the newest first: replaced whole by a rise.
    kept: ArcSwap<Vec<Kept>>,
    /// Held by a claim from its block's byte to the count's step, and by a
    /// rise: one claim at a time, and none beside a rise.
    claims: Mutex<()>,
    /// Whether the store's term is past 31 bits: set by a rise, under the
    /// claims' lock, and never cleared. From then on no block is claimed
    /// and no term leads the blocks anew, so every id names for good the
    /// file it reads.
    past: AtomicBool,
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
            claims: Mutex::new(()),
            past: AtomicBool::new(false),
        }
    }
}

impl Blocks {
    /// The word of `term` with no block claimed and no table kept. A term
    /// past 32 bits takes the greatest word: no table leads a term past 31
    /// bits, so no claim is made under it.
    fn word(term: u64) -> u64 {
        u64::from(u32::try_from(term).unwrap_or(u32::MAX)) << u32::BITS
    }

    /// The term of `lease`, as the mint lays an id out: the half over its
    /// counter. A gateway's own id reads a term no controller leads.
    fn term_of(lease: LeaseId) -> u64 {
        lease.0 >> u32::BITS
    }

    /// The block of `lease`'s counter.
    fn block_of(lease: LeaseId) -> u16 {
        (lease.0 as u32 >> BLOCK_BITS) as u16
    }

    /// `named` among `files` files, a block's byte or a block's number: one
    /// past them wraps, and no count of files stops a read.
    fn among(named: u16, files: usize) -> usize {
        usize::from(named).checked_rem(files).unwrap_or(0)
    }

    /// Claims the next block of `term`'s counter for `file`, under that
    /// file's turn: the block's number. The block's byte is written and
    /// then the count steps past it, one claim at a time, so a count that
    /// covers a block finds its byte written ([`Self::settled`]). `None`
    /// where `term` does not lead the blocks, its last block is claimed or
    /// the store's term is past 31 bits: the file mints no further id
    /// under it.
    pub(super) fn claim(&self, term: u64, file: u8) -> Option<u16> {
        let _one = self.claims.lock();
        if self.past.load(Ordering::Acquire) {
            return None;
        }
        let word = self.claimed.load(Ordering::Acquire);
        // The blocks claimed before this one are its number.
        let claimed = (word & !KEPT).checked_sub(Self::word(term))?;
        let block = u16::try_from(claimed).ok()?;
        let named = self.files.get(usize::from(block))?;
        named.store(file, Ordering::Release);
        self.claimed.store(word + 1, Ordering::Release);
        Some(block)
    }

    /// The term rose to `term`, under every file's turn; `on_file` are the
    /// ids of the leases on file. The blocks start again under `term`. The
    /// table the leaving term wrote is kept where a lease of that term is
    /// on file, and a kept table whose term holds none is dropped. The
    /// kept tables are told first, then the word, and the bytes are
    /// cleared last: a read that meets the rise finds the leaving term's
    /// table among the kept ([`Self::file_of`]).
    ///
    /// A term past 31 bits leads no table, and the blocks stay as they
    /// stand for good: the mint serves no id under it, and an id whose
    /// upper half reads as that term, a gateway's own, goes on naming its
    /// block's file. No block is claimed from then on and no term leads
    /// the blocks anew, so every id's file is settled ([`Self::settled`]).
    pub(super) fn rise(&self, term: u64, on_file: impl Iterator<Item = LeaseId>) {
        let _one = self.claims.lock();
        if term >= 1 << 31 {
            self.past.store(true, Ordering::Release);
            return;
        }
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
            let mut files = [0; BLOCKS as usize];
            for (kept, named) in files.iter_mut().zip(&self.files) {
                *kept = named.load(Ordering::Acquire);
            }
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
                return Self::among(u16::from(named), files);
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
    fn kept_file(&self, term: u64, block: u16, files: usize) -> usize {
        let kept = self.kept.load();
        let table = kept.iter().find(|table| table.term == term);
        let named = table.and_then(|table| table.files.get(usize::from(block)));
        Self::among(named.map_or(block, |named| u16::from(*named)), files)
    }

    /// The file `lease` names for good among `files` of them: the one it
    /// names for as long as a lease can be on file under it. `None` where
    /// its file is not settled. An id of the leading term is settled from
    /// a block a file claimed: the count is read first and the block's
    /// byte behind it, the other way round from a claim's writes, so a
    /// block the count covers reads its own file's byte. An id of a term
    /// below the leading one is settled wherever it reads its file: no
    /// claim and no rise writes either anew while its lease is on file.
    /// An id of the leading term from a block no file claimed is not,
    /// since the block's claim names its file anew, and neither is an id
    /// of a term that can still lead: until the store's term is past 31
    /// bits, when no claim and no rise follows and both are settled where
    /// they read their file. An id of a term past 31 bits names its
    /// block's file, as no table leads such a term, and with one file
    /// every id names it: both are settled. Asked under a file's turn,
    /// which no rise passes.
    pub(super) fn settled(&self, lease: LeaseId, files: usize) -> Option<usize> {
        let (term, block) = (Self::term_of(lease), Self::block_of(lease));
        if files <= 1 {
            return Some(0);
        }
        if term >= 1 << 31 {
            return Some(Self::among(block, files));
        }
        let word = self.claimed.load(Ordering::Acquire);
        let settled = match term.cmp(&(word >> u32::BITS)) {
            cmp::Ordering::Less => Some(self.file_of(lease, files)),
            cmp::Ordering::Equal => {
                // The blocks claimed under the leading term: the word
                // below the kept bit.
                let claimed = u64::from(block) < word & (KEPT - 1);
                let named = self.files.get(usize::from(block)).filter(|_| claimed);
                named.map(|named| Self::among(named.load(Ordering::Acquire).into(), files))
            }
            cmp::Ordering::Greater => None,
        };
        // Past 31 bits no claim and no rise follows: the file an id reads
        // is its file for good.
        let past = || self.past.load(Ordering::Acquire);
        settled.or_else(|| past().then(|| self.file_of(lease, files)))
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
#[derive(Debug, Default)]
pub(super) struct Minted {
    /// The term the block was claimed under.
    term: u64,
    /// The counter of the next id.
    next: u32,
    /// The ids the block still holds: none before the first claim.
    left: u32,
}

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
    use crate::tracking::lease::LeaseId;

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
    /// id of block 3,072, claimed as any block is.
    #[test]
    fn the_claim_past_the_last_block_fails_as_a_spent_term() {
        let blocks = Blocks::default();
        let mut file = blocks.spend(0, 1, 0);
        assert_eq!(file.mint(&blocks, 0, 0), Some(u32::MAX));
        assert_eq!(file.mint(&blocks, 0, 0), None, "the counter is spent");
        assert_eq!(Minted::default().mint(&blocks, 0, 1), None, "for each file");
        let nothing_left = blocks.spend(0, 0, 0).mint(&blocks, 0, 0);
        assert_eq!(nothing_left, None, "spent whole");

        let (blocks, mark, at) = (Blocks::default(), 3_u32 << 30, 3 * BLOCKS / 4);
        assert_eq!((at, mark), (3_072, at << BLOCK_BITS));
        let mut file = blocks.spend(0, u32::MAX - (mark - 2), 0);
        let at_the_mark = [(); 2].map(|()| file.mint(&blocks, 0, 0));
        assert_eq!(at_the_mark, [Some(mark - 1), Some(mark)]);
        // The mark's block is claimed: another file's first id is the
        // next block's.
        assert_eq!(
            Minted::default().mint(&blocks, 0, 1),
            Some((at + 1) << BLOCK_BITS)
        );
    }

    /// The most files a host names, each at its first id, take a sixteenth
    /// of the counter between them: no first id is near the term's mark,
    /// and the term keeps its span whatever the count of files.
    #[test]
    fn the_first_ids_of_the_most_files_leave_the_term_its_span() {
        let blocks = Blocks::default();
        blocks.rise(1, std::iter::empty());
        let first = |file: u8| Minted::default().mint(&blocks, 1, file);
        let firsts: Vec<_> = (0..=u8::MAX).map(first).collect();
        let highest = firsts.iter().max().copied().flatten();
        assert!(firsts.iter().all(Option::is_some), "a block a file");
        assert_eq!(highest, Some(u32::from(u8::MAX) << BLOCK_BITS));
        assert!(highest < Some(1 << 28), "under a sixteenth of the counter");
        for (file, counter) in (0..=u8::MAX).zip(&firsts) {
            let lease = id(1, counter.expect("a block a file"));
            assert_eq!(blocks.file_of(lease, 256), usize::from(file));
        }
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

    /// An id's file is settled where no claim and no rise can name it
    /// anew: an id of the leading term from a claimed block, an id of a
    /// term below it, a gateway's own id. An id of the leading term from
    /// a block no file claimed is not, and neither is one of a term that
    /// can still lead. With one file every id is settled.
    #[test]
    fn an_ids_file_is_settled_from_a_claimed_block_or_under_a_term_that_led() {
        let blocks = Blocks::default();
        blocks.rise(9, std::iter::empty());
        let mut file = Minted::default();
        assert_eq!(file.mint(&blocks, 9, 3), Some(1));
        let own = LeaseId(1 << 63 | u64::from(6 * BLOCK));
        for settled in [id(9, 1), id(9, BLOCK - 1), id(3, 5 * BLOCK), own] {
            let named = Some(blocks.file_of(settled, 4));
            assert_eq!(blocks.settled(settled, 4), named, "{settled:?}");
        }
        for open in [id(9, BLOCK), id(9, 5 * BLOCK + 17), id(10, 1)] {
            assert_eq!(blocks.settled(open, 4), None, "{open:?}");
            assert_eq!(blocks.settled(open, 1), Some(0), "one file: {open:?}");
        }
        // The claim of the second block settles its ids, and a rise leaves
        // the term below the leading one.
        let mut other = Minted::default();
        assert_eq!(other.mint(&blocks, 9, 2), Some(BLOCK));
        assert_eq!(blocks.settled(id(9, BLOCK), 4), Some(2));
        blocks.rise(10, std::iter::empty());
        assert_eq!(blocks.settled(id(9, 5 * BLOCK + 17), 4), Some(1));
        assert_eq!(blocks.settled(id(10, 1), 4), None);
        // A table kept is no block claimed: the next term's first block
        // is no file's.
        blocks.rise(11, [id(10, 1)].into_iter());
        assert_eq!(blocks.kept_terms(), [10]);
        assert_eq!(blocks.settled(id(11, 1), 4), None);
    }

    /// A block's number is past a byte. A block behind the first 256 names
    /// its file as any block does: by the leading table, by the table a
    /// rise keeps, and by its own number once no table holds its term.
    #[test]
    fn a_block_past_the_first_256_names_its_file_as_any_block_does() {
        let blocks = Blocks::default();
        // The first file holds the blocks up to 300 and the second claims
        // block 301, which a byte would read as block 45, the first's.
        let [held, next] = [300_u32, 301].map(|block| block << BLOCK_BITS);
        let mut first = blocks.spend(0, u32::MAX - held + 1, 0);
        let mut second = Minted::default();
        let minted = [first.mint(&blocks, 0, 0), second.mint(&blocks, 0, 2)];
        assert_eq!(minted, [Some(held), Some(next)]);
        let lease = id(0, next);
        assert_eq!(blocks.file_of(lease, 3), 2, "the leading table");
        blocks.rise(1, [lease].into_iter());
        assert_eq!(blocks.file_of(lease, 3), 2, "the kept table");
        blocks.rise(2, std::iter::empty());
        assert_eq!(blocks.file_of(lease, 3), 1, "its number, 301, among 3");
    }

    /// One claim is made at a time: two files that claim at once take the
    /// blocks of a term between them, each block once, whichever of them
    /// takes it. Every wait ends at the case's bound and a file claims no
    /// more than two terms' blocks: a claim that never fails, or a claimer
    /// that leaves, fails the case.
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

    /// A count that covers a block finds its byte written: a claim writes
    /// the byte and then steps the count, and what asks whether an id's
    /// file is settled reads the count first. Beside a file that claims a
    /// term's blocks a reader asks of the block the count covered last and
    /// of the next one: the first is the claiming file's, the next is not
    /// settled or is the claiming file's, and neither is the one a cleared
    /// byte names. On one core a reader seldom lands inside a claim, and
    /// the case never fails a right read.
    #[test]
    fn a_block_under_claim_is_not_settled_before_its_byte_is_written() {
        const ROUNDS: usize = 32;
        let round = || {
            let blocks = Blocks::default();
            let (reads, done) = (AtomicBool::new(false), AtomicBool::new(false));
            thread::scope(|scope| {
                let reader = scope.spawn(|| {
                    let mut wrong = 0_u64;
                    let file = |block: u32| blocks.settled(id(0, block << BLOCK_BITS | 1), 4);
                    while !done.load(Ordering::Acquire) {
                        let next = blocks.claimed.load(Ordering::Acquire) as u32;
                        if let Some(last) = next.checked_sub(1) {
                            wrong += u64::from(file(last) != Some(3));
                        }
                        if next < BLOCKS {
                            wrong += u64::from(!matches!(file(next), None | Some(3)));
                        }
                        reads.store(true, Ordering::Release);
                    }
                    wrong
                });
                while !reads.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                // A term's blocks and no more: a claim that never fails
                // ends at twice as many.
                let claims = std::iter::from_fn(|| blocks.claim(0, 3));
                let claims = claims.take(2 * BLOCKS as usize).count();
                done.store(true, Ordering::Release);
                let wrong = reader.join().expect("the reader ends");
                assert_eq!(claims, BLOCKS as usize, "a term's blocks, each once");
                wrong
            })
        };
        let wrong: u64 = (0..ROUNDS).map(|_| round()).sum();
        assert_eq!(wrong, 0, "a block read as settled under a cleared byte");
    }

    /// A term past 31 bits leads no table, the least such term as any
    /// above it: the mint serves no id under it and no block is claimed.
    /// A gateway's own id, whose upper half reads as such a term, names
    /// its block's file before the store reaches that term, under it and
    /// behind it, and an id of the last term below it goes on reading
    /// that term's table. From then on every id is settled where it reads
    /// its file: no block is claimed and no term leads anew.
    #[test]
    fn a_term_past_31_bits_leads_no_table() {
        for past in [1 << 31, 1 << 31 | 5] {
            let blocks = Blocks::default();
            blocks.rise(4, std::iter::empty());
            assert_eq!(Minted::default().mint(&blocks, 4, 2), Some(1));
            let (own, held) = (id(past, 7 * BLOCK + 9), id(4, 1));
            let files = || (blocks.file_of(own, 3), blocks.file_of(held, 3));
            assert_eq!(files(), (1, 2), "ahead of {past:#x}");
            // Not settled ahead of it: a block of the leading term that no
            // file claimed, and a term that can still lead.
            let open = [id(4, 5 * BLOCK + 3), id(9, 5 * BLOCK + 3)];
            let settled = || open.map(|open| blocks.settled(open, 3));
            assert_eq!(settled(), [None, None]);
            blocks.rise(past, [own, held].into_iter());
            assert_eq!(files(), (1, 2), "under {past:#x}");
            assert_eq!((blocks.claim(past, 0), blocks.claim(4, 0)), (None, None));
            let named = (blocks.settled(own, 3), blocks.settled(held, 3));
            assert_eq!(named, (Some(1), Some(2)));
            // No claim and no rise follows: each is settled where it reads
            // its file, the cleared byte and the block's number among 3.
            assert_eq!(settled(), [Some(0), Some(2)]);
            assert!(blocks.kept_terms().is_empty());
            blocks.rise(past + 1, [own, held].into_iter());
            assert_eq!(files(), (1, 2), "behind {past:#x}");
            assert_eq!(settled(), [Some(0), Some(2)]);
        }
    }

    /// A claim and a rise never write the word beside each other: a claim
    /// that read the word ahead of a rise would write the leaving term
    /// back over the risen one. A file claims under whatever term leads
    /// while the term rises: each rise finds the term the rise before it
    /// wrote, and the last one stands. On one core a claim seldom meets a
    /// rise, and the case never fails a right write.
    #[test]
    fn a_claim_beside_a_rise_leaves_the_risen_term() {
        const TERMS: u64 = 500;
        let blocks = Blocks::default();
        let leading = || blocks.claimed.load(Ordering::Acquire) >> u32::BITS;
        let done = AtomicBool::new(false);
        let back = thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    blocks.claim(leading(), 1);
                }
            });
            let back = (1..=TERMS).filter(|&term| {
                let back = leading() != term - 1;
                blocks.rise(term, std::iter::empty());
                back
            });
            let back = back.count();
            done.store(true, Ordering::Release);
            back
        });
        assert_eq!((back, leading()), (0, TERMS), "a term written back");
    }
}
