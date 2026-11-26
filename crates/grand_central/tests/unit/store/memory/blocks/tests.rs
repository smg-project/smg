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
