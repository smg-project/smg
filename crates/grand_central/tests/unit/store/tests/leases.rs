//! A lease on file: the term it is minted and kept under, its place in the
//! queue of dispatch deadlines, and the lookups a dispatch, a completion, a
//! decode's take-back and a release make of it.

use super::*;

#[test]
fn a_lease_above_the_stores_term_is_refused_and_one_below_is_kept() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let gateway = GatewayId(1);
    leases.set_term(3);
    leases.set_term(2);
    assert_eq!(leases.term(), 3, "a term never goes back");
    assert_eq!(
        leases.issue(lease(gateway, 1, 4, now)),
        Err(LeaseRefused::TermAhead { lease: 4, store: 3 })
    );
    assert_eq!(leases.issue(lease(gateway, 1, 2, now)), Ok(()));
    assert_eq!(
        leases.issue(lease(gateway, 1, 3, now)),
        Err(LeaseRefused::Duplicate)
    );
    assert_eq!(leases.issue(lease(gateway, 2, 3, now)), Ok(()));
    assert_eq!(leases.live(gateway), 2);
    assert!(leases.dispatched(gateway, LeaseId(1), Dispatch::default(), now));
    assert!(!leases.dispatched(GatewayId(2), LeaseId(1), Dispatch::default(), now));
    let expired = leases.expire(now + Duration::from_secs(2));
    assert_eq!(
        expired.len(),
        1,
        "the undispatched lease passed its deadline"
    );
    assert_eq!(expired[0].record.lease.id, LeaseId(2));
    assert_eq!(
        expired[0].record.state,
        LeaseState::Released(ReleaseReason::Abandoned)
    );
    assert_eq!(leases.live(gateway), 1);
    let outcome = Outcome {
        side: Side::Single,
        served: WorkerId(7),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    assert!(leases.completed(gateway, LeaseId(1), &outcome));
    assert!(!leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(leases.live(gateway), 0);
}

/// No lease's counter: what an issue that ended nothing hands back.
const NONE: [u64; 0] = [];

#[test]
fn the_store_mints_a_decisions_lease_under_its_own_term_alone() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let world = store.snapshot();
    let decided = |term: u64| {
        let record = LeaseRecord {
            term,
            ..booking(gateway, 0, a, now)
        };
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.map(|lease| lease.id.0)
    };
    leases.set_term(3);
    assert_eq!(decided(3), Ok(3 << 32 | 1), "the term over its first count");
    assert_eq!(decided(3), Ok(3 << 32 | 2));
    let ahead = LeaseRefused::TermAhead { lease: 4, store: 3 };
    assert_eq!(decided(4), Err(ahead));
    let behind = LeaseRefused::Unminted { lease: 2, store: 3 };
    assert_eq!(decided(2), Err(behind), "a term the store left mints none");
    assert_eq!(in_flight(&store), [(a, 2)], "a refused lease books nothing");
    let second = LeaseId(3 << 32 | 2);
    let filed = leases.get(gateway, second).expect("under its minted id");
    assert_eq!(
        (filed.lease.id, filed.term, filed.target, filed.state),
        (second, 3, a, LeaseState::ISSUED)
    );
    // A term's counter starts at 1, and an id a restored lease holds is
    // passed over: the store's `Duplicate` is no decision's answer.
    leases.set_term(4);
    let restored = lease(gateway, 4 << 32 | 2, 4, now);
    leases.issue(restored).expect("the key is free");
    assert_eq!(decided(4), Ok(4 << 32 | 1));
    assert_eq!(decided(4), Ok(4 << 32 | 3), "the id on file is passed over");
    // A term's last id is its counter's last value: the next is refused,
    // and the one after it too.
    store.spend_ids(1);
    assert_eq!(decided(4), Ok(4 << 32 | u64::from(u32::MAX)));
    let spent = LeaseRefused::Unminted { lease: 4, store: 4 };
    assert_eq!((decided(4), decided(4)), (Err(spent), Err(spent)));
    leases.set_term(5);
    assert_eq!(decided(5), Ok(5 << 32 | 1), "the next term mints again");
    // A term past the 31 bits an id carries it in mints none.
    leases.set_term(1 << 31);
    let wide = LeaseRefused::Unminted {
        lease: 1 << 31,
        store: 1 << 31,
    };
    assert_eq!(decided(1 << 31), Err(wide));
}

/// A rise of the store's term hands the blocks the ids on file: the term a
/// lease was issued under keeps its table through the rises past it, and
/// the table goes at the first rise behind the lease's end. One file names
/// every id either way, so the kept terms are read through a test's door.
#[test]
fn a_rise_keeps_the_table_of_a_term_whose_lease_is_on_file() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(3);
    let record = LeaseRecord {
        term: 3,
        ..booking(gateway, 0, a, now)
    };
    let booked = store.snapshot().booked_of(a);
    let issued = store.issue_decided(record, booked, |_| ());
    issued.expect("minted under the store's term");
    leases.set_term(4);
    assert_eq!(store.kept_terms(), [3], "a lease of the term is on file");
    leases.set_term(5);
    assert_eq!(store.kept_terms(), [3], "through the rises that follow");
    assert_eq!(leases.drop_gateways(&[gateway]), [1]);
    assert_eq!(store.kept_terms(), [3], "until the next rise");
    leases.set_term(6);
    assert!(
        store.kept_terms().is_empty(),
        "the first rise behind its end"
    );
}

/// A restore that raises the store's term tells the blocks, as the term's
/// own door does, with the ids on file: the table of a term that holds a
/// lease is kept, and the first lease decided behind the restore is the
/// new term's first id.
#[test]
fn a_restore_that_raises_the_term_starts_the_blocks_under_it() {
    let standby = MemoryStore::new();
    let term: &dyn LeasesStore = &standby;
    term.set_term(3);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let held = standby.issue_decided(lease(gateway, 0, 3, now), None, |_| ());
    held.expect("minted under the term the standby stood at");
    let mut cut = MemoryStore::new().checkpoint();
    cut.at.term = 7;
    assert!(standby.restore_from(cut));
    assert_eq!(term.term(), 7);
    assert_eq!(
        standby.kept_terms(),
        [3],
        "the rise is told the ids on file"
    );
    let issued = standby.issue_decided(lease(gateway, 1, 7, now), None, |_| ());
    assert_eq!(issued.map(|lease| lease.id.0), Ok(7 << 32 | 1));
}

#[test]
fn an_issue_ends_two_leases_past_their_dispatch_deadline_at_most() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, t0) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let id = |counter: u64| LeaseId(1 << 32 | counter);
    // Five leases at the start: the second is dispatched, the fourth ends.
    for counter in 1..=5 {
        assert_eq!(issued_at(&store, a, t0), (counter, vec![]));
    }
    assert!(leases.dispatched(gateway, id(2), Dispatch::default(), t0));
    assert!(leases.released(gateway, id(4), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), [(a, 4)]);
    // An issue inside the deadline ends nothing.
    assert_eq!(issued_at(&store, a, t0 + Duration::from_secs(4)).1, NONE);
    // One past it ends the first and the third: two, and the dispatched
    // lease between them stands.
    let late = t0 + Duration::from_secs(6);
    assert_eq!(issued_at(&store, a, late).1, [1, 3]);
    // The next steps over the place of the lease that left and ends the
    // fifth; the one after finds nothing due.
    assert_eq!(issued_at(&store, a, late).1, [5]);
    assert_eq!(issued_at(&store, a, late).1, NONE);
    for ended in [1, 3, 5] {
        assert_eq!(leases.get(gateway, id(ended)), None);
    }
    // On file: the dispatched lease, the one issued at four seconds and
    // the three late ones, each on its worker's count.
    assert_eq!(leases.live(gateway), 5);
    assert_eq!(in_flight(&store), [(a, 5)]);
}

#[test]
fn an_issue_steps_over_four_places_of_the_deadline_queue_at_most() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, t0) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // Seven leases at the start; the first six leave at their gateway's
    // word and keep their places in the queue.
    for counter in 1..=7 {
        assert_eq!(issued_at(&store, a, t0).0, counter);
    }
    for counter in 1..=6 {
        let id = LeaseId(1 << 32 | counter);
        assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));
    }
    // Past the deadline one issue steps over four of those places and ends
    // nothing; the next steps over the last two and ends the seventh.
    let late = t0 + Duration::from_secs(6);
    assert_eq!(issued_at(&store, a, late).1, NONE);
    assert_eq!(leases.live(gateway), 2, "the seventh still stands");
    assert_eq!(issued_at(&store, a, late).1, [7]);
    assert_eq!(in_flight(&store), [(a, 2)]);
}

#[test]
fn a_lease_that_left_or_was_dispatched_leaves_the_deadline_queue_at_the_next_issues() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, t0) = (GatewayId(1), Instant::now());
    let id = |counter: u64| LeaseId(1 << 32 | counter);
    leases.set_term(1);
    // A hundred leases inside one dispatch deadline, each ended before the
    // next is issued: the queue holds the place of the last one ended and
    // of the one just issued, not a hundred places.
    for counter in 1..=100 {
        assert_eq!(issued_at(&store, a, t0), (counter, NONE.to_vec()));
        assert!(store.queued() <= 2, "at lease {counter}");
        assert!(leases.released(gateway, id(counter), None, ReleaseReason::ClientGone));
    }
    // A lease that waits for its dispatch keeps its place, and the places
    // behind it stand until it moves.
    assert_eq!(issued_at(&store, a, t0).0, 101);
    for counter in 102..=105 {
        assert_eq!(issued_at(&store, a, t0).0, counter);
        assert!(leases.released(gateway, id(counter), None, ReleaseReason::ClientGone));
    }
    assert_eq!(store.queued(), 5);
    // Its dispatch frees them: the next issue steps over four.
    assert!(leases.dispatched(gateway, id(101), Dispatch::default(), t0));
    assert_eq!(issued_at(&store, a, t0), (106, NONE.to_vec()));
    assert_eq!(store.queued(), 2);
    assert_eq!(
        in_flight(&store),
        [(a, 2)],
        "the dispatched lease and the last one"
    );
}

#[test]
fn the_dispatch_deadlines_fall_due_in_their_own_order_whatever_order_they_were_filed_in() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // Filed with deadlines of five, one and three seconds, in that order.
    for (id, secs) in [(1, 5), (2, 1), (3, 3)] {
        let record = LeaseRecord {
            dispatch_deadline: now + Duration::from_secs(secs),
            ..lease(gateway, id, 1, now)
        };
        leases.issue(record).expect("the key is free");
    }
    let due = |secs: u64| {
        let ended = leases.expire(now + Duration::from_secs(secs));
        let ids = ended.iter().map(|found| found.record.lease.id.0);
        ids.collect::<Vec<u64>>()
    };
    assert_eq!(due(0), NONE);
    assert_eq!(due(2), [2]);
    assert_eq!(due(2), NONE, "an ended lease is ended once");
    // A lease written again with a later deadline is due at the later one:
    // the place its first deadline left in the queue ends nothing.
    let again = LeaseRecord {
        dispatch_deadline: now + Duration::from_secs(9),
        ..lease(gateway, 3, 1, now)
    };
    assert_eq!(leases.restore(gateway, &[again]), 1);
    assert_eq!(due(8), NONE, "the first lease left at the restore");
    assert_eq!(due(9), [3]);
    assert_eq!(leases.live(gateway), 0);
}

#[test]
fn dispatch_and_completion_each_look_up_the_lease_once() {
    let now = Instant::now();
    let (store, _, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    let hashes = |said| {
        LEASE_HASHES.set(0);
        store.hand(&hand, ids[0], said);
        LEASE_HASHES.get()
    };
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    let complete = Said::Completed(done(Side::Single));
    assert_eq!((hashes(dispatch), hashes(complete)), (1, 1));
}

#[cfg(feature = "controller")]
#[test]
fn a_decode_take_back_looks_up_the_lease_once() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    store.set_term(1);
    issue_pair(&store, chosen_pair(id.0, p, &[d], now)).expect("the key is free");
    assert!(store.dispatched(gateway, id, on(Side::Prefill, 0), now));
    assert_eq!(store.issue_decode(gateway, id, decode_side(d), now), Ok(()));

    LEASE_HASHES.set(0);
    assert!(store.take_back_decode(gateway, id, ReleaseReason::Abandoned));
    assert_eq!(LEASE_HASHES.get(), 1);
}

/// A release on a file no decode call issued into looks the lease up once,
/// for the release itself: the read of what an issue replaced, ahead of
/// it, looks no record up there.
#[cfg(feature = "controller")]
#[test]
fn a_release_on_a_file_no_call_issued_into_looks_up_the_lease_once() {
    let now = Instant::now();
    let (store, _, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    LEASE_HASHES.set(0);
    store.hand(&hand, ids[0], GONE);
    assert_eq!(LEASE_HASHES.get(), 1);
}
