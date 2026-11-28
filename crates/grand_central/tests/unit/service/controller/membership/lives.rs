//! The cases of how a row's life is numbered: by the term above the store's
//! epoch, from the base a run draws, under a term the log leads alone,
//! inside the bits each part has, apart for two runs under one term by the
//! base each draws, and so that a call which names a life that left moves no
//! later one.

use super::*;

/// The bound of a base and of an epoch by its number, 2^39: a case reads
/// it here and not through the door's constant, so that constant's value
/// stands on a case.
const BOUND: u64 = 1 << 39;

#[test]
fn a_providers_registration_numbers_its_life_under_the_term_the_log_leads() {
    use super::super::{ReplicaTurn, Turn};
    use crate::store::membership::FactPastBound;

    let (store, door) = door_in(TERM + 1);
    let door = Arc::new(door);
    let maker = found("pods/a");
    let stale = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let refusal = Err(MembershipWriteRefused::NotLeading(
        super::super::LEADS_NO_TERM.to_owned(),
    ));
    assert_eq!(stale.found_one(&maker, URL, &|_| found_facts()), refusal);
    assert_eq!(store.epoch(), 0);

    let leading = ReplicaMembershipWriter::new(door, operator(), under(TERM + 1));
    let number = life(TERM + 1, 0);
    assert_eq!(
        leading.found_one(&maker, URL, &|_| found_facts()),
        Ok(MembershipWriteOutcome::Registered(identity(URL, number)))
    );
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(row.facts.incarnation, number);
    let replaced = || WorkerFacts {
        role: Role::Decode,
        ..found_facts()
    };
    assert_eq!(stale.found_one(&maker, URL, &|_| replaced()), refusal);
    assert_eq!(store.by_key(URL), Some(row.clone()));
    let new = life(TERM + 1, 1);
    assert_eq!(
        leading.found_one(&maker, URL, &|_| replaced()),
        Ok(MembershipWriteOutcome::Another {
            old: row.id,
            new: identity(URL, new)
        })
    );
    assert_eq!(
        store
            .by_key(URL)
            .expect("the replaced row")
            .facts
            .incarnation,
        new
    );

    let turn = ReplicaTurn {
        host: &leading,
        term: TERM + 1,
    };
    let heavy = WorkerFacts {
        models: vec!["m".repeat(256); 64],
        ..found_facts()
    };
    assert_eq!(
        turn.room(&heavy, &maker),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "row",
            why: "a row is at most 16384 bytes on the wire",
        }))
    );

    let (store, door) = door_in(TERM);
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let state = |_: Option<&WorkerFacts>| {
        store::LeasesStore::set_term(store.as_ref(), TERM + 1);
        found_facts()
    };
    assert_eq!(hosted.found_one(&maker, URL, &state), refusal);
    assert_eq!(store.all_rows(), (0, Vec::new()));
    // A life is asked a row at a time: once the term has moved the turn
    // makes none, at whichever row of a registration asks it.
    let turn = ReplicaTurn {
        host: &hosted,
        term: TERM,
    };
    assert_eq!(
        turn.life(),
        Err(MembershipWriteRefused::NotLeading(
            super::super::LEADS_NO_TERM.to_owned()
        ))
    );
}

#[test]
fn a_life_is_numbered_by_the_term_above_the_stores_epoch() {
    assert_eq!(life(7, 0), (7 << 40) | 1);
    assert_eq!(life(7, 12), (7 << 40) | 13);
    assert_eq!(
        super::super::life(7, 5, 12),
        (7 << 40) | 18,
        "from its base"
    );
    // The highest base with the highest epoch fills the low bits and
    // carries nothing into the term's.
    let highest = super::super::life(7, BOUND - 1, BOUND - 1);
    assert_eq!(highest, (7 << 40) | ((1 << 40) - 1));
    assert!(
        highest < life(8, 0),
        "a later term stands above every count of an earlier one"
    );

    // Two leaders that take the same URL first number two lives: the
    // pair read at one names no row at the other.
    let (_, earlier) = door_in(7);
    let (store, later) = door_in(8);
    let first = row_of(
        &earlier
            .register(&operator(), under(7), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let second = row_of(
        &later
            .register(&operator(), under(8), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(
        (first.incarnation, second.incarnation),
        (life(7, 0), life(8, 0))
    );
    assert_ne!(first.id, second.id);
    assert_eq!(first.revision, second.revision);
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    let refused = later
        .deregister(&operator(), under(8), &removal)
        .expect_err("another leader's registration");
    assert_eq!(refused.code(), Code::NotFound);
    assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(second.id));
}

#[test]
fn a_life_is_minted_only_under_a_term_its_log_leads() {
    let (store, door) = door();
    let refused = |told: Result<proto::WorkerEvent, Status>| {
        let status = told.expect_err("no life under a term the log does not lead");
        assert_eq!(
            (status.code(), status.message()),
            (Code::Unavailable, super::super::LEADS_NO_TERM)
        );
    };
    // The call was admitted under a term the log has left, or one it
    // has not entered: the two terms are compared at the mint.
    for admitted in [TERM - 1, TERM + 1] {
        let told = door.register(&operator(), under(admitted), &ask(spec(URL), Mode::Add));
        refused(told);
    }
    // The store follows another leader's history at the admitted term:
    // a number under it could be one that leader gave another worker.
    door.log
        .restart(Position {
            seq: 40,
            term: TERM + 1,
        })
        .expect("a place the log takes");
    let told = door.register(&operator(), under(TERM + 1), &ask(spec(URL), Mode::Add));
    refused(told);
    assert_eq!(store.epoch(), 0, "a refused registration writes nothing");
    assert!(store.by_key(URL).is_none());
    // A take above every term the log has seen starts a lead again.
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let row = row_of(
        &door
            .register(&operator(), under(TERM + 2), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(row.incarnation, life(TERM + 2, 0));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a term past its 24 bits")]
fn a_term_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = life(1 << 24, 0);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "an epoch past its 39 bits")]
fn an_epoch_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = life(7, BOUND);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a base past its 39 bits")]
fn a_base_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = super::super::life(7, BOUND, 0);
}

#[test]
fn a_run_draws_its_base_below_the_bound_and_apart_from_another_runs() {
    let drawn: Vec<u64> = (0..64)
        .map(|_| {
            let (store, feed) = journaled(&LeasePlace::Process, 1);
            let objects = holder(&store);
            MembershipCalls::new(store, feed.log, objects).base
        })
        .collect();
    assert!(drawn.iter().all(|&base| base < BOUND), "{drawn:?}");
    let mut apart = drawn.clone();
    apart.sort_unstable();
    apart.dedup();
    assert_eq!(apart.len(), drawn.len(), "{drawn:?}");
}

#[test]
fn two_runs_under_one_term_number_a_urls_lives_apart() {
    // A controller with no shared lease leads under term one at every
    // start: each run's door draws where its count starts.
    let run = || {
        let (store, feed) = journaled(&LeasePlace::Process, 1);
        store::LeasesStore::set_term(store.as_ref(), 1);
        let door = MembershipCalls::new(Arc::clone(&store), feed.log, holder(&store));
        let first = row_of(
            &door
                .register(&operator(), under(1), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        (store, door, first)
    };
    let (_, earlier, removed) = run();
    let (store, later, standing) = run();
    assert_ne!(removed.incarnation, standing.incarnation);
    assert_ne!(removed.id, standing.id);
    // Inside one run the count keeps its order: a later life has the
    // higher number.
    let next = row_of(
        &earlier
            .register(&operator(), under(1), &ask(spec(URL), Mode::Replace))
            .expect("replaced"),
    );
    assert!(next.incarnation > removed.incarnation);
    // A removal read at the earlier run names no row of the later one.
    let removal = proto::DeregisterWorkerRequest {
        worker_id: removed.id,
        revision: removed.revision,
        drain: false,
    };
    let refused = later
        .deregister(&operator(), under(1), &removal)
        .expect_err("the earlier run's registration");
    assert_eq!(refused.code(), Code::NotFound);
    assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(standing.id));
}

#[test]
fn a_removal_that_arrives_twice_never_takes_the_urls_next_registration() {
    let (store, door) = door();
    let first = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    // The worker registers again before the removal's retry lands (its
    // first answer was lost): a new row, at the revision the first had.
    let second = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered again"),
    );
    assert_eq!(second.revision, first.revision);
    assert_ne!(second.id, first.id, "a life of its own");
    assert!(second.incarnation > first.incarnation);

    let retried = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("the retry names a registration that left");
    assert_eq!(retried.code(), Code::NotFound);
    let stands = store.by_key(URL).expect("the second registration");
    assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
}

#[test]
fn a_patch_under_a_registration_that_left_never_writes_the_urls_next_one() {
    let (store, door) = door();
    let first = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    let second = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered again"),
    );
    // A patch its caller made of the first registration, arriving late.
    let late = proto::PatchWorkerRequest {
        worker_id: first.id,
        labels: [("lane".to_owned(), "late".to_owned())].into(),
        revision: first.revision,
        ..Default::default()
    };
    let refused = door
        .patch(&operator(), under(TERM), &late)
        .expect_err("the patch names a registration that left");
    assert_eq!(refused.code(), Code::NotFound);
    let stands = store.by_key(URL).expect("the second registration");
    assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
    assert!(stands.facts.labels.is_empty());
}
