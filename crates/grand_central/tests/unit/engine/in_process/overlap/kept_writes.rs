//! The writes a thread keeps where it meets a held record: a write behind a
//! column and its next turn, and what becomes of a write that a leave made
//! stale and of the kept writes beside it.

use super::*;

/// A write behind a column that meets a held record is kept as one behind
/// a pick is: the thread's next column reads it beside what the record
/// holds, another thread's column reads the record alone, and the next
/// turn's write is kept in its turn. Once the record is free, the write
/// behind the thread's next column stands, the kept write is written on
/// behind it, and nothing waits.
#[test]
fn a_column_reads_the_write_its_thread_keeps_and_the_next_turn_is_kept() {
    let (affinity, _, _) = hosted(false);
    let prefills = Bed::new(affinity, paired(), 4);
    let asked = prompt(1_000, 4);
    let longer = prompt(1_000, 6);
    let stood = prompt(2_000, 4);
    // A column of the prompt and the write behind it for the worker `to`.
    let sent = |prompt: &PromptHashes, to: usize| {
        let facts = request(prompt, 7, NOW);
        let (column, read) = prefills.column(&facts, Phase::Prefill);
        let read = read.expect("the record speaks for the prefills");
        record_by_column(
            &prefills.pool,
            Phase::Prefill,
            &facts,
            read,
            &column,
            prefills.ids[to],
        );
        held(&column)
    };
    let read =
        |prompt: &PromptHashes| held(&prefills.column(&request(prompt, 7, NOW), Phase::Prefill).0);
    // What a thread that keeps no write reads.
    let elsewhere = |prompt: &PromptHashes| {
        thread::scope(|scope| {
            let reader = scope.spawn(|| read(prompt));
            reader.join().expect("a reader")
        })
    };
    // The blocks a column reads for one worker of the four.
    let at = |worker: usize, blocks: u32| {
        let mut column = [(0, 0); 4];
        column[worker] = (0, blocks);
        column
    };
    // The record holds another prompt for the fourth worker.
    assert_eq!(sent(&stood, 3), [(0, 0); 4]);
    thread::scope(|scope| {
        let keeper = scope.spawn(|| {
            assert_eq!(
                prefills.affinity.while_held(|| sent(&asked, 1)),
                [(0, 0); 4]
            );
            assert_eq!(prefills.affinity.left_out(), 0);
            assert_eq!(read(&asked), at(1, 4));
            // A thread that keeps a write reads the record as before.
            assert_eq!(read(&stood), at(3, 4));
            assert_eq!(elsewhere(&asked), [(0, 0); 4]);
            // The next turn, the record held again: its column reads the
            // turn before, and its write is kept in its turn.
            assert_eq!(prefills.affinity.while_held(|| sent(&longer, 1)), at(1, 4));
            assert_eq!(read(&longer), at(1, 6));
            assert_eq!(elsewhere(&longer), [(0, 0); 4]);
            // The record free, the turn sent to the third worker: its write
            // stands and the kept write is written on behind it.
            assert_eq!(sent(&longer, 2), at(1, 6));
            assert_eq!(elsewhere(&longer), [(0, 0), (0, 6), (0, 6), (0, 0)]);
        });
        keeper.join().expect("a keeper");
    });
    // Nothing waited as the thread's buffers went.
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A write whose naming a leave made stale takes no kept write of its
/// thread with it. The thread fills a column for one request and a worker
/// leaves. It then fills and writes for a second request, the first one's
/// leading part, for the same worker. That write meets a held record and
/// is kept, named on the record's count as it stands. The first request's
/// write then stops at the record, and the kept one is written on behind
/// it. So a reader on another thread finds the second request's blocks at
/// the worker and no block more, and nothing is left out.
#[test]
fn a_write_a_leave_made_stale_takes_no_kept_write_with_it() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let (whole, leading) = (prompt(1_000, 4), prompt(1_000, 2));
    let (first, second) = (request(&whole, 7, NOW), request(&leading, 7, NOW));
    let (list, id) = (&bed.pool, bed.ids[1]);
    let (column_of, read) = bed.column(&first, Phase::Single);
    let read = read.expect("the record speaks for the list");
    bed.affinity.forget(bed.ids[3]);
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    {
        let (column_of, read) = bed.column(&second, Phase::Single);
        let read = read.expect("the record speaks for the list");
        let _reader = record.predictor.read();
        record_by_column(list, Phase::Single, &second, read, &column_of, id);
    }
    assert!(record.predictor.read().is_empty(), "the write waits");
    record_by_column(list, Phase::Single, &first, read, &column_of, id);
    // What a thread that keeps no write reads, of the leading part and of
    // the whole request: the stale write gave the worker no block.
    let held_of = |facts: &RequestFacts<'_>| held(&bed.column(facts, Phase::Single).0);
    let elsewhere = thread::scope(|scope| {
        let reader = scope.spawn(|| (held_of(&second), held_of(&first)));
        reader.join().expect("a reader")
    });
    assert_eq!(elsewhere.0, [(0, 0), (0, 2), (0, 0), (0, 0)]);
    assert_eq!(elsewhere.1, [(0, 0), (0, 2), (0, 0), (0, 0)]);
    assert_eq!(bed.affinity.left_out(), 0);
}

/// A write whose naming a leave made stale is kept by no thread. Met at a
/// held record it could write nothing, and at the bound of eight it would
/// push a kept write out. A pair's prefill side fills its column and a
/// worker leaves. Behind a held record the thread then keeps the writes of
/// eight more columns, as the decisions a stage asks for inside the pair's
/// own would leave them. The first column's write meets the record held
/// too. The eight stay, the thread's next column still reads the oldest,
/// and nothing is left out. So for a request the record holds no block of,
/// and for one whose leading part stands for the picked worker: that write
/// begins behind the blocks that stand.
#[test]
fn a_stale_write_that_meets_a_held_record_is_not_kept() {
    for leading in [false, true] {
        let (affinity, _, _) = hosted(false);
        let prefills = Bed::new(affinity, paired(), 4);
        let (stale, oldest) = (prompt(9_000, 4), prompt(1_000, 4));
        let (first, part) = (request(&stale, 7, NOW), prompt(9_000, 2));
        let (list, to) = (&prefills.pool, prefills.ids[2]);
        // The write of one column behind its fill, as a pick makes it.
        let write_for = |asked: &PromptHashes, id: WorkerId| {
            let facts = request(asked, 7, NOW);
            let (column, named) = prefills.column(&facts, Phase::Prefill);
            let named = named.expect("the record speaks for the prefills");
            record_by_column(list, Phase::Prefill, &facts, named, &column, id);
        };
        // What the thread's next column reads of the oldest kept write.
        let kept = || held(&prefills.column(&request(&oldest, 7, NOW), Phase::Prefill).0);
        thread::scope(|scope| {
            let keeper = scope.spawn(|| {
                if leading {
                    write_for(&part, prefills.ids[1]);
                }
                let (column_of, read) = prefills.column(&first, Phase::Prefill);
                let read = read.expect("the record speaks for the prefills");
                // The second shape's write begins behind the leading part's
                // two blocks: the fill reads them at the picked worker.
                let at_the_picked = if leading { (0, 2) } else { (0, 0) };
                assert_eq!(held(&column_of)[1], at_the_picked, "leading: {leading}");
                prefills.affinity.forget(prefills.ids[3]);
                prefills.affinity.while_held(|| {
                    for more in 1..9 {
                        write_for(&prompt(1_000 * more, 4), to);
                    }
                    let eight = (prefills.affinity.left_out(), kept());
                    let id = prefills.ids[1];
                    record_by_column(list, Phase::Prefill, &first, read, &column_of, id);
                    (eight, (prefills.affinity.left_out(), kept()))
                })
            });
            let (eight, behind) = keeper.join().expect("a keeper");
            let at_the_third = vec![(0, 0), (0, 0), (0, 4), (0, 0)];
            assert_eq!(eight, (0, at_the_third.clone()), "leading: {leading}");
            assert_eq!(behind, (0, at_the_third), "leading: {leading}");
        });
    }
}
