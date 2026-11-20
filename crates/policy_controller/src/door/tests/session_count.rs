//! The session keys a worker carries, at the door. A rule that places a
//! first key by `min_group` reads, beside each row, how many keys have a
//! request in flight on its worker. The store counts them by the leases
//! on file: a key is one of a worker's keys from its first request there
//! to its last.

use std::{sync::Arc, time::Instant};

use super::{dialed, dispatched, fleet, gone, keyed, placed, ready, report, stated, GATEWAY};
use crate::{
    door::Door,
    lease::Side,
    spec::{AppliedSpec, PolicySpec},
    store::{Dispatch, Said, WorkersStore},
    view::{WireModes, WorkerId},
    Assignment,
};

/// The `manual` preset as generation 2, its rule placing a first key on
/// the worker that carries the fewest keys.
fn grouped() -> AppliedSpec {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 2;
    let rule = document.bindings[0].sessions.as_mut();
    rule.expect("the preset's rule").assignment = Assignment::MinGroup;
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The keys each worker of `placements` carries, fewest first.
fn spread(placements: &[WorkerId]) -> Vec<usize> {
    let mut workers = placements.to_vec();
    workers.sort_unstable();
    workers.dedup();
    let keys = |worker: &WorkerId| placements.iter().filter(|held| *held == worker).count();
    let mut spread: Vec<usize> = workers.iter().map(keys).collect();
    spread.sort_unstable();
    spread
}

/// Under `min_group` a first key lands on a worker that carries the
/// fewest keys with a request in flight. Nine keys, one after another,
/// spread three to a worker over a pool of three.
#[test]
fn first_keys_one_after_another_spread_evenly_under_min_group() {
    let store = fleet(1);
    store.apply(grouped()).expect("a later generation");
    let door = Door::new(Arc::clone(&store));
    let mut placements = Vec::new();
    for n in 1..=9_u8 {
        let id = [b'k', b'-', b'0' + n];
        let asked = keyed(&id, 100 + u64::from(n), u64::from(n));
        let (lease, worker, _) = placed(&door, &asked);
        report(&door, lease, dispatched());
        placements.push(worker);
    }
    assert_eq!(spread(&placements), [3, 3, 3], "{placements:?}");
}

/// A gateway that states its wires is handed rows written for it. They
/// carry the keys each worker holds, and first keys spread over them as
/// over the publication's own rows.
#[test]
fn first_keys_spread_evenly_over_the_rows_written_for_a_gateway() {
    let store = fleet(1);
    store.apply(grouped()).expect("a later generation");
    let wires = WireModes::GRPC;
    for name in ["w3", "w4", "w5"] {
        dialed(&store, name, wires);
    }
    let door = Door::new(Arc::clone(&store));
    let mut placements = Vec::new();
    for n in 1..=9_u8 {
        let id = [b'g', b'-', b'0' + n];
        let asked = stated(&id, 200 + u64::from(n), u64::from(n), wires);
        let (lease, worker, _) = placed(&door, &asked);
        report(&door, lease, dispatched());
        placements.push(worker);
    }
    assert_eq!(spread(&placements), [3, 3, 3], "{placements:?}");
}

/// A key is one of a worker's keys from its first request there to its
/// last. A second request of the key counts nothing more, and the
/// worker's count falls at the key's last end alone.
#[test]
fn a_key_counts_once_on_its_worker_from_its_first_request_to_its_last() {
    let store = fleet(1);
    store.apply(grouped()).expect("a later generation");
    let door = Door::new(Arc::clone(&store));
    let (first, worker, _) = placed(&door, &keyed(b"a-1", 7, 1));
    report(&door, first, dispatched());
    assert_eq!(store.keys_on(worker), 1);
    // The key's second request stays on its worker.
    let (second, again, _) = placed(&door, &keyed(b"a-2", 7, 2));
    assert_eq!(again, worker);
    assert_eq!(store.keys_on(worker), 1, "one key, two requests");
    report(&door, first, gone());
    assert_eq!(store.keys_on(worker), 1, "the key's other request stands");
    report(&door, second, gone());
    assert_eq!(store.keys_on(worker), 0);
    assert_eq!(store.keys_in_flight(), 0);
}

/// A worker back under its id stands under the table's next slot. A key
/// with a request on it from before the leave is one of its keys again at
/// the key's next request, and the lease of the registration that left
/// gives back the slot it booked.
#[test]
fn a_key_counts_on_a_worker_back_under_its_id_at_its_next_request() {
    let store = fleet(1);
    store.apply(grouped()).expect("a later generation");
    let door = Door::new(Arc::clone(&store));
    let (before, worker, _) = placed(&door, &keyed(b"b-1", 7, 1));
    report(&door, before, dispatched());
    assert_eq!(store.keys_on(worker), 1);
    // The worker leaves and registers again under its key and incarnation.
    let row = WorkersStore::get(&*store, worker).expect("the row");
    assert!(store.remove(worker, row.revision).is_some());
    assert_eq!(ready(&store, row.facts), worker);
    assert_eq!(store.keys_on(worker), 0, "the next slot holds no key yet");
    let (after, again, _) = placed(&door, &keyed(b"b-2", 7, 2));
    assert_eq!(again, worker);
    assert_eq!(store.keys_on(worker), 1);
    // The lease of the registration that left ends on the slot it booked.
    report(&door, before, gone());
    assert_eq!(store.keys_on(worker), 1);
    report(&door, after, gone());
    assert_eq!(store.keys_on(worker), 0);
    assert_eq!(store.keys_in_flight(), 0);
}

/// A dispatch that names another target of its pick moves the key with
/// its lease: the key leaves the first target's keys and is one of the
/// worker's it was sent to, until the lease's end.
#[test]
fn a_dispatch_on_another_target_moves_the_key_with_its_lease() {
    let store = fleet(1);
    store.apply(grouped()).expect("a later generation");
    let door = Door::new(Arc::clone(&store));
    let decided = door.decide(GATEWAY, &keyed(b"m-1", 7, 1), Instant::now());
    let decided = decided.expect("a decision");
    let pick = decided.decision.pick().expect("a pick");
    let mut targets = pick.targets.iter().map(|target| target.id);
    let first = targets.next().expect("a target");
    let other = targets.next().expect("a second target");
    let lease = decided.lease.expect("a pick's lease").lease.id;
    assert_eq!((store.keys_on(first), store.keys_on(other)), (1, 0));
    let sent = Dispatch {
        target_index: 1,
        side: Side::Single,
        since_decision_micros: 250,
    };
    report(&door, lease, Said::Dispatched(sent, Instant::now()));
    assert_eq!((store.keys_on(first), store.keys_on(other)), (0, 1));
    report(&door, lease, gone());
    assert_eq!((store.keys_on(first), store.keys_on(other)), (0, 0));
}
