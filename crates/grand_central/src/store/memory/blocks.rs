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
#[path = "../../../tests/unit/store/memory/blocks/tests.rs"]
mod tests;
