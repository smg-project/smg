//! The cases of a rank its engine stops naming. An answer that brings a
//! row for one rank of an engine and none for another marks the rank it
//! left out: the rank's rows carry the overload bit while a rank the answer
//! named has a record on file, with no threshold in the store.

use std::time::{Duration, Instant};

use super::{
    ask, block, generation_with, load, member, overloaded, pressure, rank_among, rank_of, view,
};
use crate::{
    store::{LoadSink, LoadSource, MemoryStore, WorkersStore},
    view::WorkerId,
};

const ENGINE: &str = "grpc://e:9000";
const OTHER: &str = "grpc://f:9000";

/// A store with no threshold and the two ranks of one engine, each with a
/// record on file from one answer.
fn two_ranks() -> (MemoryStore, Instant, WorkerId, WorkerId) {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_of(&store, ENGINE, 0);
    let one = rank_of(&store, ENGINE, 1);
    names_both(&store, zero, one, t0);
    (store, t0, zero, one)
}

/// An answer that names both ranks.
fn names_both(store: &MemoryStore, zero: WorkerId, one: WorkerId, at: Instant) {
    store.report(
        &[(zero, 0, Some(load(1, 0))), (one, 1, Some(load(2, 0)))],
        at,
        LoadSource::Poll,
    );
}

/// An answer that names rank one and leaves rank zero out.
fn names_one_alone(store: &MemoryStore, zero: WorkerId, one: WorkerId, at: Instant) {
    store.report(
        &[(zero, 0, None), (one, 1, Some(load(2, 0)))],
        at,
        LoadSource::Poll,
    );
}

#[test]
fn a_rank_an_answer_leaves_out_beside_a_named_one_is_avoided_at_once() {
    let (store, t0, zero, one) = two_ranks();
    assert_eq!(overloaded(&store, t0), [], "both ranks are named");
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero], "with no threshold");
    assert_eq!(store.overload_flips(), 1);
    // The rank keeps what is on file: it is avoided, not forgotten.
    let rows = view(&store, &ask(t0)).0;
    let row = rows.iter().find(|row| row.id == zero).expect("its row");
    assert_eq!(row.load.map(|load| load.running), Some(1));
    // The same answer again moves nothing.
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero]);
    assert_eq!((store.overload_flips(), store.unnamed_ranks()), (1, 1));
}

#[test]
fn an_answer_or_a_record_that_names_the_rank_ends_its_mark() {
    let (store, t0, zero, one) = two_ranks();
    names_one_alone(&store, zero, one, t0);
    names_both(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [], "an answer names it again");
    assert_eq!(store.unnamed_ranks(), 0);
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero]);
    let t1 = t0 + Duration::from_secs(1);
    store.record(zero, 0, load(1, 0), t1, LoadSource::Stream);
    assert_eq!(overloaded(&store, t1), [], "a record names it again");
    assert_eq!(store.unnamed_ranks(), 0);
    // A poll that began before that record says nothing of the rank.
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t1), [], "it pushed since the poll began");
    assert_eq!(store.unnamed_ranks(), 0);
}

#[test]
fn a_rank_that_pushed_since_the_poll_began_takes_no_mark_beside_one_the_answer_leaves_out() {
    // Three ranks of one engine, each with a record on file from one answer.
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [zero, one, two] = [0, 1, 2].map(|rank| rank_among(&store, ENGINE, rank, 3));
    store.report(
        &[
            (zero, 0, Some(load(1, 0))),
            (one, 1, Some(load(2, 0))),
            (two, 2, Some(load(3, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    // Rank zero pushes after a poll began. The poll's answer names rank one
    // and leaves the two others out: it marks the rank that said nothing
    // since, and the rank that pushed named itself.
    let t1 = t0 + Duration::from_secs(1);
    store.record(zero, 0, load(1, 0), t1, LoadSource::Stream);
    store.report(
        &[(zero, 0, None), (one, 1, Some(load(2, 0))), (two, 2, None)],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(
        store.unnamed_ranks(),
        1,
        "the rank that pushed named itself"
    );
    // The answer is dated before the engine's last write, so its mark is
    // shown with the next one.
    let t2 = t1 + Duration::from_secs(1);
    store.record(one, 1, load(2, 0), t2, LoadSource::Stream);
    assert_eq!(overloaded(&store, t2), [two]);
}

#[test]
fn a_rank_left_out_beside_a_named_rank_that_pushed_since_the_poll_began_is_avoided() {
    let (store, t0, zero, one) = two_ranks();
    // Rank one pushes as a poll begins. The poll's answer brings a row for
    // it and none for rank zero: the record stands over the answer's row,
    // and the answer named the rank all the same.
    store.record(one, 1, load(5, 0), t0, LoadSource::Stream);
    names_one_alone(&store, zero, one, t0);
    assert_eq!(store.unnamed_ranks(), 1, "the answer named its peer");
    // No row was written: the mark is judged and shown with the answer.
    assert_eq!(overloaded(&store, t0), [zero]);
    assert_eq!(store.overload_flips(), 1);
    let rows = view(&store, &ask(t0)).0;
    let row = rows.iter().find(|row| row.id == one).expect("its row");
    assert_eq!(row.load.map(|load| load.running), Some(5), "the record");
    // The same answer again moves nothing.
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero]);
    assert_eq!((store.overload_flips(), store.unnamed_ranks()), (1, 1));
}

#[test]
fn a_row_passed_over_for_a_record_ends_no_word_that_no_report_came() {
    let (store, t0, zero, one) = two_ranks();
    let mut both = [zero, one];
    both.sort_unstable();
    store
        .apply(generation_with(1, block(8, 0.8)))
        .expect("applied");
    // Rank one pushes at the count: the engine's verdict holds both.
    let t1 = t0 + Duration::from_secs(1);
    store.record(one, 1, pressure(9, 0), t1, LoadSource::Stream);
    assert_eq!(overloaded(&store, t1), both);
    // A later poll brings no report, and the verdict reads clear.
    let t2 = t1 + Duration::from_secs(1);
    store.report(&[(zero, 0, None), (one, 1, None)], t2, LoadSource::Poll);
    assert_eq!(overloaded(&store, t2), []);
    // The answer of a poll that began before the record lands late. Its row
    // for rank one is passed over: it marks the rank left out and ends no
    // word, so the engine's own verdict still reads clear.
    names_one_alone(&store, zero, one, t0);
    store.publish(t2);
    assert_eq!(overloaded(&store, t2), [zero]);
}

#[test]
fn a_mixed_answer_marks_the_rank_left_out_of_each_engine_it_names() {
    // Two engines in one answer: a written row names the first, and a row
    // passed over for a record names the second.
    let (store, t0, zero, one) = two_ranks();
    let [far_zero, far_one] = [0, 1].map(|rank| rank_of(&store, OTHER, rank));
    names_both(&store, far_zero, far_one, t0);
    store.record(far_one, 1, load(2, 0), t0, LoadSource::Stream);
    store.report(
        &[
            (zero, 0, None),
            (one, 1, Some(load(2, 0))),
            (far_zero, 0, None),
            (far_one, 1, Some(load(2, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    let mut left_out = [zero, far_zero];
    left_out.sort_unstable();
    assert_eq!(overloaded(&store, t0), left_out);
    assert_eq!(store.unnamed_ranks(), 2);
}

#[test]
fn the_word_that_no_report_came_moves_no_mark() {
    let (store, t0, zero, one) = two_ranks();
    let nobody = [(zero, 0, None), (one, 1, None)];
    store.report(&nobody, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), [], "no rank is marked");
    assert_eq!(store.unnamed_ranks(), 0);
    names_one_alone(&store, zero, one, t0);
    store.report(&nobody, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), [zero], "and none is named again");
    assert_eq!(store.unnamed_ranks(), 1);
}

#[test]
fn a_rank_with_no_named_rank_on_file_answers_alone() {
    let (store, t0, zero, one) = two_ranks();
    names_one_alone(&store, zero, one, t0);
    // The named rank's rows go: the rank left out is the one on file.
    store.evict(one);
    store.publish(t0);
    assert_eq!(overloaded(&store, t0), [], "alone it answers");
    assert_eq!(store.unnamed_ranks(), 1, "the mark stands");
    // The named rank answers again: the mark holds its rank again.
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero]);
}

#[test]
fn an_eviction_takes_the_rank_s_mark_with_its_rows() {
    let (store, t0, zero, one) = two_ranks();
    names_one_alone(&store, zero, one, t0);
    store.evict(zero);
    store.publish(t0);
    assert_eq!(overloaded(&store, t0), [], "nothing on file: a rank again");
    assert_eq!(store.unnamed_ranks(), 0);
    // An answer that leaves it out again marks it, record or none.
    names_one_alone(&store, zero, one, t0);
    assert_eq!(overloaded(&store, t0), [zero]);
    // A second eviction finds no record to fold and still lifts the mark.
    store.evict(zero);
    store.publish(t0);
    assert_eq!(overloaded(&store, t0), []);
    assert_eq!(store.unnamed_ranks(), 0);
}

#[test]
fn a_membership_write_keeps_the_mark_and_a_member_that_leaves_takes_its_own() {
    let (store, t0, zero, one) = two_ranks();
    names_one_alone(&store, zero, one, t0);
    // Another worker joins: every row is new, and the mark stands.
    member(&store, "http://other:8000");
    assert_eq!(overloaded(&store, t0), [zero]);
    // The named rank leaves: the rank left out is alone.
    let revision = WorkersStore::get(&store, one).expect("the row").revision;
    store.remove(one, revision).expect("removed");
    assert_eq!(overloaded(&store, t0), []);
    assert_eq!(store.unnamed_ranks(), 1);
    // The rank itself leaves: its mark goes with it.
    let revision = WorkersStore::get(&store, zero).expect("the row").revision;
    store.remove(zero, revision).expect("removed");
    assert_eq!(store.unnamed_ranks(), 0);
}

#[test]
fn a_threshold_s_verdict_and_the_mark_share_the_bit() {
    let (store, t0, zero, one) = two_ranks();
    let mut both = [zero, one];
    both.sort_unstable();
    store
        .apply(generation_with(1, block(8, 0.8)))
        .expect("applied");
    let answer = |first: Option<u32>, second: u32| {
        store.report(
            &[
                (zero, 0, first.map(|queued| pressure(queued, 0))),
                (one, 1, Some(pressure(second, 0))),
            ],
            t0,
            LoadSource::Poll,
        );
        overloaded(&store, t0)
    };
    // Under the count the mark alone holds its rank, and the generation's
    // judgment keeps it.
    assert_eq!(answer(Some(1), 1), []);
    assert_eq!(answer(None, 1), [zero]);
    store
        .apply(generation_with(2, block(8, 0.8)))
        .expect("applied");
    assert_eq!(overloaded(&store, t0), [zero]);
    // At the count the engine's verdict holds both, and the mark's end
    // leaves the verdict standing.
    assert_eq!(answer(None, 7), both, "1 on file and 7");
    assert_eq!(answer(Some(1), 7), both);
    assert_eq!(answer(Some(1), 1), []);
}

#[test]
fn a_write_is_judged_for_a_mark_only_while_one_stands() {
    let (store, t0, zero, one) = two_ranks();
    assert!(!store.judges_writes(), "no threshold and no mark");
    names_one_alone(&store, zero, one, t0);
    assert!(store.judges_writes(), "a rank stands unnamed");
    // A record ends the last mark: its engine is judged once more, and no
    // write after it is.
    let t1 = t0 + Duration::from_secs(1);
    store.record(zero, 0, load(1, 0), t1, LoadSource::Stream);
    assert_eq!(overloaded(&store, t1), []);
    assert_eq!((store.unnamed_ranks(), store.judges_writes()), (0, false));
    // An answer that names the rank again ends it the same, and so does
    // the eviction of the marked member.
    let t2 = t1 + Duration::from_secs(1);
    names_one_alone(&store, zero, one, t2);
    assert_eq!(overloaded(&store, t2), [zero]);
    assert!(store.judges_writes());
    names_both(&store, zero, one, t2);
    assert_eq!(overloaded(&store, t2), []);
    assert_eq!((store.unnamed_ranks(), store.judges_writes()), (0, false));
    // A membership write judges every engine again and the mark stands:
    // a record after it still ends the mark.
    names_one_alone(&store, zero, one, t2);
    member(&store, "http://other:8000");
    assert!(store.judges_writes(), "across a membership write");
    let t3 = t2 + Duration::from_secs(1);
    store.record(zero, 0, load(1, 0), t3, LoadSource::Stream);
    assert_eq!(overloaded(&store, t3), []);
    assert_eq!((store.unnamed_ranks(), store.judges_writes()), (0, false));
    let t4 = t3 + Duration::from_secs(1);
    names_one_alone(&store, zero, one, t4);
    store.evict(zero);
    store.publish(t4);
    assert_eq!(overloaded(&store, t4), []);
    assert_eq!((store.unnamed_ranks(), store.judges_writes()), (0, false));
}
