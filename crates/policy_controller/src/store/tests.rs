//! The contract tests of the store ingress, run over [`MemoryStore`] through
//! the trait objects: what every implementation of a trait promises.

use std::{
    cell::RefCell,
    sync::{mpsc, Arc, Mutex},
    time::{Duration, Instant},
};

use super::{
    identity, identity_of,
    leases::{Counts, Slot},
    Ask, BatchFacts, BatchOutcome, ContentHash, Dispatch, GatewayId, Hand, HealthSink, Journal,
    Kinds, KvEvent, KvEventSink, LeaseRecord, LeaseRefused, LeaseState, LeasesStore, LoadSample,
    LoadSink, LoadSource, MembershipChange, MembershipTail, MemoryStore, Picked, Pin, PollMode,
    Position, Reachability, ReservationKind, ReservationStore, ResyncReason, Said, SequenceHash,
    SessionsStore, Signals, SnapshotChunk, StoredBlock, VetoWrite, ViewBuffers, WorkerFacts,
    WorkerPatch, WorkerRow, WorkerStatus, WorkersStore, World,
};
use crate::{
    decider::Decider,
    decision::{Target, Targets},
    lease::{Lease, LeaseId, Membership, Outcome, ReleaseReason, Side, TransportClass},
    view::{fingerprint, LoadRow, Role, SignalSource, WorkerId},
};

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation,
        models: vec!["m".to_string()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    }
}

fn ready(store: &MemoryStore, key: &str) -> WorkerId {
    let id = store.upsert(facts(key, 1)).id();
    store.publish_now();
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    store
        .set_status(id, WorkerStatus::Ready, revision)
        .expect("ready");
    id
}

fn row(age_ms: u32) -> LoadRow {
    LoadRow {
        running: 1,
        report_age_ms: age_ms,
        ..LoadRow::default()
    }
}

fn batch(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// A lease of `gateway` on worker 7, issued at `now` under `term`: the one
/// place the store's cases build a record's fields.
fn lease(gateway: GatewayId, id: u64, term: u64, now: Instant) -> LeaseRecord {
    LeaseRecord {
        lease: Lease {
            id: LeaseId(id),
            ..Lease::default()
        },
        gateway,
        request: id,
        target: WorkerId(7),
        picked: Picked::default(),
        decode: None,
        term,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(1),
        state: LeaseState::Issued,
        session_ttl_micros: None,
    }
}

#[test]
fn a_registration_is_keyed_and_a_new_incarnation_replaces_it_under_a_new_id() {
    let store = MemoryStore::new();
    let workers: &dyn WorkersStore = &store;
    let first = workers.upsert(facts("http://w:8000", 1));
    let MembershipChange::Registered { id } = first else {
        panic!("a new key registers: {first:?}");
    };
    assert_eq!(id, identity("http://w:8000", 1));
    assert_eq!(
        workers.upsert(facts("http://w:8000", 1)),
        MembershipChange::Relabelled { id },
        "the same incarnation refreshes in place"
    );
    let mut wider = facts("http://w:8000", 1);
    wider.dp_size = 4;
    assert_eq!(
        workers.upsert(wider),
        MembershipChange::Resized { id },
        "a changed width is a resize"
    );
    let replaced = workers.upsert(facts("http://w:8000", 2));
    let MembershipChange::Replaced { old, new } = replaced else {
        panic!("a new incarnation replaces: {replaced:?}");
    };
    assert_eq!(old, id);
    assert_eq!(new, identity("http://w:8000", 2));
    assert_ne!(old, new);
    assert!(workers.get(old).is_none(), "the old id left");
    assert_eq!(workers.by_key("http://w:8000").map(|row| row.id), Some(new));
    assert_eq!(
        replaced.as_lifecycle(),
        Some(Membership::Replaced { old, new })
    );
}

#[test]
fn a_replacement_carries_a_matching_registrations_status_and_a_changed_one_starts_pending() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://w:8000");
    let same = store.upsert(facts("http://w:8000", 2)).id();
    assert_eq!(
        WorkersStore::get(&store, same).expect("the row").status,
        WorkerStatus::Ready
    );
    let mut other_role = facts("http://w:8000", 3);
    other_role.role = Role::Prefill;
    let changed = store.upsert(other_role).id();
    assert_eq!(
        WorkersStore::get(&store, changed).expect("the row").status,
        WorkerStatus::Pending
    );
    assert_ne!(id, changed);
}

#[test]
fn a_write_under_a_moved_revision_is_refused_and_never_reaches_a_successor() {
    let store = MemoryStore::new();
    let workers: &dyn WorkersStore = &store;
    let id = workers.upsert(facts("http://w:8000", 1)).id();
    let stale = workers.get(id).expect("the row").revision;
    assert!(workers
        .patch(id, WorkerPatch::default(), stale)
        .is_some_and(|change| change == MembershipChange::Relabelled { id }));
    assert!(
        workers.remove(id, stale).is_none(),
        "the patch moved the revision"
    );
    assert!(workers.set_status(id, WorkerStatus::Ready, stale).is_none());
    assert!(workers.get(id).is_some(), "the row stands");
    let current = workers.get(id).expect("the row").revision;
    let status = workers.set_status(id, WorkerStatus::Ready, current);
    assert_eq!(
        status,
        Some(MembershipChange::StatusChanged {
            id,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        })
    );
    assert!(
        status.and_then(|change| change.as_lifecycle()).is_none(),
        "a status change is a health signal, not a membership event"
    );
    let current = workers.get(id).expect("the row").revision;
    assert!(
        workers
            .set_status(id, WorkerStatus::Ready, current)
            .is_none(),
        "a status that already stood emits nothing"
    );
    assert_eq!(
        workers.remove(id, current),
        Some(MembershipChange::Removed { id })
    );
    assert!(workers.get(id).is_none());
}

#[test]
fn a_watcher_tails_the_changes_after_its_epoch_and_snapshots_again_past_the_log() {
    let store = MemoryStore::new();
    let workers: &dyn WorkersStore = &store;
    let (epoch, rows) = workers.snapshot();
    assert_eq!((epoch, rows.len()), (0, 0));
    let a = workers.upsert(facts("http://a:8000", 1)).id();
    let (epoch, rows) = workers.snapshot();
    assert_eq!((epoch, rows.len()), (1, 1));
    let b = workers.upsert(facts("http://b:8000", 1)).id();
    assert_eq!(
        workers.changes_since(epoch),
        MembershipTail::Changes(vec![(2, MembershipChange::Registered { id: b })])
    );
    assert_eq!(
        workers.changes_since(workers.epoch()),
        MembershipTail::Changes(Vec::new()),
        "a watcher at the head has nothing to read"
    );
    assert_eq!(
        workers.changes_since(99),
        MembershipTail::SnapshotRequired,
        "an epoch from another era snapshots"
    );
    for i in 0..1_100u64 {
        let mut facts = facts("http://a:8000", 1);
        facts.priority = i32::try_from(i).expect("small");
        workers.upsert(facts);
    }
    assert_eq!(
        workers.changes_since(epoch),
        MembershipTail::SnapshotRequired,
        "a watcher past the log snapshots again"
    );
    let (head, rows) = workers.snapshot();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, a, "the snapshot is ordered by key");
    assert!(matches!(
        workers.changes_since(head - 1),
        MembershipTail::Changes(changes) if changes.len() == 1
    ));
}

#[test]
fn probe_outcomes_follow_the_rows_thresholds_and_a_stale_outcome_is_dropped() {
    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let id = store.upsert(facts("http://w:8000", 1)).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    assert_eq!(health.probe_outcome(id, true, revision()).moved(), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "two passes promote a pending worker"
    );
    // A patch moves the revision: an outcome read under the one it left is
    // stale.
    let stale = revision();
    WorkersStore::patch(&store, id, WorkerPatch::default(), stale).expect("the patch");
    assert_eq!(
        health.probe_outcome(id, false, stale).moved(),
        None,
        "a stale outcome is dropped"
    );
    assert_eq!(health.probe_outcome(id, false, revision()).moved(), None);
    assert_eq!(health.probe_outcome(id, false, revision()).moved(), None);
    assert_eq!(
        health.probe_outcome(id, false, revision()).moved(),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady)),
        "three failures demote"
    );
    for _ in 0..8 {
        assert_eq!(health.probe_outcome(id, false, revision()).moved(), None);
    }
    assert_eq!(
        health.probe_outcome(id, false, revision()).moved(),
        Some((WorkerStatus::NotReady, WorkerStatus::Failed)),
        "nine more fail it"
    );
    assert_eq!(health.probe_outcome(id, true, revision()).moved(), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()).moved(),
        Some((WorkerStatus::Failed, WorkerStatus::Ready)),
        "two passes bring a failed worker back"
    );
    let fresh = store.upsert(facts("http://f:8000", 1)).id();
    let fresh_revision = WorkersStore::get(&store, fresh).expect("the row").revision;
    assert_eq!(
        health.connected(fresh, fresh_revision).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "the connect signal promotes at once"
    );
    assert_eq!(
        health.connected(fresh, fresh_revision).moved(),
        None,
        "a ready row takes no second promotion"
    );
}

#[test]
fn a_threshold_past_the_counters_range_is_never_met_early() {
    // The first threshold whose tenfold passes 32 bits, and the last one.
    for threshold in [429_496_730, u32::MAX] {
        let store = MemoryStore::new();
        let health: &dyn HealthSink = &store;
        let mut stated = facts("http://w:8000", 1);
        stated.health.failure_threshold = threshold;
        let id = store.upsert(stated).id();
        let revision = || WorkersStore::get(&store, id).expect("the row").revision;
        // A pending worker's cap is ten thresholds: five failed probes are
        // far under it.
        for _ in 0..5 {
            assert_eq!(
                health.probe_outcome(id, false, revision()).moved(),
                None,
                "pending under {threshold}"
            );
        }
        assert_eq!(health.probe_outcome(id, true, revision()).moved(), None);
        assert_eq!(
            health.probe_outcome(id, true, revision()).moved(),
            Some((WorkerStatus::Pending, WorkerStatus::Ready))
        );
        // And a ready one stays ready through failures under the threshold.
        for _ in 0..5 {
            assert_eq!(
                health.probe_outcome(id, false, revision()).moved(),
                None,
                "ready under {threshold}"
            );
        }
    }
}

#[test]
fn a_counter_at_its_ranges_end_stays_there() {
    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let mut stated = facts("http://w:8000", 1);
    stated.health.failure_threshold = u32::MAX;
    let id = store.upsert(stated).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    // A pending worker's cap is held at the counter's end: met there, and
    // one probe short of it not yet.
    store.probed(id, 0, 0, u32::MAX - 2);
    assert_eq!(health.probe_outcome(id, false, revision()).moved(), None);
    assert_eq!(
        health.probe_outcome(id, false, revision()).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Failed))
    );
    // Each counter at its end takes one probe more and stays there.
    store.probed(id, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(health.probe_outcome(id, false, revision()).moved(), None);
    store.probed(id, 0, u32::MAX, u32::MAX);
    assert_eq!(
        health.probe_outcome(id, true, revision()).moved(),
        Some((WorkerStatus::Failed, WorkerStatus::Ready))
    );
}

#[test]
fn a_threshold_whose_triple_passes_the_counters_range_fails_no_row_early() {
    // The first threshold whose triple passes 32 bits: a product that
    // wrapped would read 2.
    let threshold = 1_431_655_766;
    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let mut stated = facts("http://w:8000", 1);
    stated.health.failure_threshold = threshold;
    let id = store.upsert(stated).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    assert_eq!(health.probe_outcome(id, true, revision()).moved(), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    // Ready, the row is not ready at the threshold's own count of failed
    // probes: the count is set one short of it.
    store.probed(id, threshold - 1, 0, 0);
    assert_eq!(
        health.probe_outcome(id, false, revision()).moved(),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady))
    );
    // Not ready, it fails at three thresholds of failed probes: five are
    // far under them.
    for _ in 0..5 {
        assert_eq!(
            health.probe_outcome(id, false, revision()).moved(),
            None,
            "not ready under the triple"
        );
    }
}

#[test]
fn a_connect_signal_read_under_a_revision_the_row_left_is_dropped() {
    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let id = store.upsert(facts("http://p:8000", 1)).id();
    let read = WorkersStore::get(&store, id).expect("the row").revision;
    WorkersStore::patch(&store, id, WorkerPatch::default(), read).expect("the patch");
    assert_eq!(
        health.connected(id, read).moved(),
        None,
        "a stale signal is dropped"
    );
    assert_eq!(
        WorkersStore::get(&store, id).expect("the row").status,
        WorkerStatus::Pending,
        "and the row stays as it stood"
    );
    assert_eq!(
        health.connected(id, read + 1).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "the signal read under the row's revision promotes"
    );
}

#[test]
fn a_health_write_says_what_happened_and_the_status_that_stands() {
    use super::HealthWrite::{Moved, NoRow, Stale, Stood};

    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let id = store.upsert(facts("http://w:8000", 1)).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    // A write that moves nothing is judged, and says the status that stands.
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Stood(WorkerStatus::Pending)
    );
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Moved {
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        }
    );
    // The connect signal on a ready row stands and clears no counter: two
    // failures ahead of it and one behind it are the threshold's three.
    for _ in 0..2 {
        assert_eq!(
            health.probe_outcome(id, false, revision()),
            Stood(WorkerStatus::Ready)
        );
    }
    assert_eq!(health.connected(id, revision()), Stood(WorkerStatus::Ready));
    assert_eq!(
        health.probe_outcome(id, false, revision()),
        Moved {
            from: WorkerStatus::Ready,
            to: WorkerStatus::NotReady
        },
        "the signal cleared no counter of the ready row"
    );
    // A write read under a revision the row left is stale: it says no
    // status and is not counted, so the two passes behind it are the
    // threshold's two.
    let left = revision();
    WorkersStore::patch(&store, id, WorkerPatch::default(), left).expect("the patch");
    assert_eq!(health.probe_outcome(id, true, left), Stale);
    assert_eq!(health.connected(id, left), Stale);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Stood(WorkerStatus::NotReady),
        "the stale pass was not counted"
    );
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Moved {
            from: WorkerStatus::NotReady,
            to: WorkerStatus::Ready
        }
    );
    // A draining row stands at the signal too.
    WorkersStore::set_status(&store, id, WorkerStatus::Draining, revision()).expect("drained");
    assert_eq!(
        health.connected(id, revision()),
        Stood(WorkerStatus::Draining)
    );
    // A write for a worker whose row is gone says no row.
    let last = revision();
    store.remove(id, last).expect("removed");
    assert_eq!(health.probe_outcome(id, true, last), NoRow);
    assert_eq!(health.connected(id, last), NoRow);
}

#[test]
fn a_health_writes_two_readers_say_the_transition_and_the_status_that_stands() {
    use super::HealthWrite::{Moved, NoRow, Stale, Stood};

    let moved = Moved {
        from: WorkerStatus::Pending,
        to: WorkerStatus::Ready,
    };
    assert_eq!(
        moved.moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    assert_eq!(moved.stands(), Some(WorkerStatus::Ready));
    let stood = Stood(WorkerStatus::Draining);
    assert_eq!(stood.moved(), None);
    assert_eq!(stood.stands(), Some(WorkerStatus::Draining));
    for unjudged in [Stale, NoRow] {
        assert_eq!(unjudged.moved(), None);
        assert_eq!(unjudged.stands(), None);
    }
}

#[test]
fn a_gateway_veto_masks_the_worker_for_that_gateway_until_its_end_or_its_clear() {
    let store = MemoryStore::new();
    // A registered worker: the health writers refuse an id without a row.
    let id = ready(&store, "http://v:8000");
    let health: &dyn HealthSink = &store;
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let now = Instant::now();
    health.gateway_veto(g1, id, Some(now + Duration::from_secs(60)));
    assert!(health.vetoed(g1, id, now));
    assert!(!health.vetoed(g2, id, now), "another gateway's view");
    assert!(
        !health.vetoed(g1, id, now + Duration::from_secs(61)),
        "the veto ends"
    );
    health.gateway_veto(g1, id, None);
    assert!(!health.vetoed(g1, id, now), "an explicit clear");
    assert!(!health.completed_ok_since(g2, id, now));
    health.completed(g2, id, TransportClass::ServerError, now);
    assert!(
        !health.completed_ok_since(g2, id, now),
        "only an ok dispatch counts"
    );
    health.completed(g2, id, TransportClass::Ok, now + Duration::from_millis(5));
    assert!(health.completed_ok_since(g2, id, now));
    assert!(!health.completed_ok_since(g2, id, now + Duration::from_millis(6)));
    assert!(
        !health.completed_ok_since(g1, id, now),
        "one gateway's dispatch keeps the worker routable for that gateway alone"
    );
}

/// The health table keeps the latest instant of a gateway's completions on
/// a worker: an earlier one that arrives after it moves nothing back.
#[test]
fn a_gateways_completion_on_a_worker_is_kept_at_its_latest_instant() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://e:8000");
    let health: &dyn HealthSink = &store;
    let gateway = GatewayId(1);
    let first = Instant::now();
    let second = first + Duration::from_millis(5);
    health.completed(gateway, id, TransportClass::Ok, first);
    assert!(!health.completed_ok_since(gateway, id, second));
    health.completed(gateway, id, TransportClass::Ok, second);
    assert!(
        health.completed_ok_since(gateway, id, second),
        "the later completion is the one kept"
    );
    health.completed(gateway, id, TransportClass::Ok, first);
    assert!(
        health.completed_ok_since(gateway, id, second),
        "an earlier one that arrives late moves nothing back"
    );
}

#[test]
fn a_removed_worker_leaves_the_health_tables_and_takes_no_late_write() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://d:8000");
    let health: &dyn HealthSink = &store;
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let now = Instant::now();
    health.gateway_veto(g1, id, Some(now + Duration::from_secs(60)));
    health.gateway_veto(g2, id, Some(now + Duration::from_secs(1)));
    health.completed(g1, id, TransportClass::Ok, now);
    assert_eq!(store.health_entries(), 3);
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
    assert!(
        !health.vetoed(g1, id, now),
        "a standing veto on a removed worker is gone with it"
    );
    assert!(
        !health.completed_ok_since(g1, id, now),
        "and so is its completed dispatch"
    );
    assert_eq!(store.health_entries(), 0);
    // A gateway's in-flight work on the worker ends after the removal: a
    // late completion and a late veto write nothing under the gone id.
    health.completed(g2, id, TransportClass::Ok, now + Duration::from_secs(2));
    health.gateway_veto(g2, id, Some(now + Duration::from_secs(90)));
    assert!(
        !health.completed_ok_since(g2, id, now),
        "a completion after the removal books nothing"
    );
    assert!(
        !health.vetoed(g2, id, now + Duration::from_secs(3)),
        "a veto after the removal masks nothing"
    );
    assert_eq!(store.health_entries(), 0, "and the tables stay empty");
}

#[test]
fn a_replaced_incarnation_leaves_the_health_tables_with_its_row() {
    let store = MemoryStore::new();
    let old = ready(&store, "http://w:8000");
    let health: &dyn HealthSink = &store;
    let g = GatewayId(1);
    let now = Instant::now();
    health.gateway_veto(g, old, Some(now + Duration::from_secs(60)));
    health.completed(g, old, TransportClass::Ok, now);
    assert_eq!(store.health_entries(), 2);
    let new = store.upsert(facts("http://w:8000", 2)).id();
    assert_ne!(old, new, "a new incarnation under the key");
    assert!(
        !health.vetoed(g, old, now),
        "the old incarnation's veto left with its row"
    );
    assert!(!health.completed_ok_since(g, old, now));
    assert_eq!(store.health_entries(), 0, "and the new id starts clean");
}

/// A worker's load rows live exactly as long as its row: the removal deletes
/// them, and a record or a report that lands later (a poll that was out, a
/// stream's last frame) writes nothing.
#[test]
fn a_removed_worker_leaves_no_load_row_and_a_late_record_or_report_writes_none() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://d:8000");
    let loads: &dyn LoadSink = &store;
    let t0 = Instant::now();
    loads.record(id, 0, row(0), t0, LoadSource::Stream);
    loads.report(&[(id, 1, Some(row(0)))], t0, LoadSource::Poll);
    assert_eq!(loads.ranks(id), 2);
    assert_eq!(store.load_entries(), 1);

    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
    assert_eq!(loads.ranks(id), 0, "the rows left with the row");
    assert_eq!(store.load_entries(), 0, "and the entry with them");

    let later = t0 + Duration::from_secs(1);
    loads.record(id, 0, row(0), later, LoadSource::Stream);
    loads.record(id, 0, row(0), later, LoadSource::Relay(GatewayId(4)));
    loads.report(
        &[(id, 0, Some(row(0))), (id, 1, Some(row(0)))],
        later,
        LoadSource::Poll,
    );
    assert_eq!(loads.ranks(id), 0, "a late write re-creates nothing");
    assert!(loads.load(id, 0).is_none());
    assert_eq!(store.load_entries(), 0);
    assert_eq!(
        loads.poll_mode(id, later, Duration::from_secs(10)),
        PollMode::Poll
    );
}

#[test]
fn a_replaced_incarnation_leaves_the_old_id_without_rows_and_the_new_id_starts_empty() {
    let store = MemoryStore::new();
    let old = ready(&store, "http://w:8000");
    let loads: &dyn LoadSink = &store;
    let t0 = Instant::now();
    loads.record(old, 0, row(0), t0, LoadSource::Stream);
    assert_eq!(loads.ranks(old), 1);

    let new = store.upsert(facts("http://w:8000", 2)).id();
    assert_ne!(old, new, "a new incarnation under the key");
    assert_eq!(
        loads.ranks(old),
        0,
        "the old incarnation's rows left with its row"
    );
    assert_eq!(
        loads.ranks(new),
        0,
        "nothing the old one reported is the new one's"
    );
    assert_eq!(store.load_entries(), 1, "one entry, the new member's");

    // The old incarnation's stream is still draining: its record is dropped,
    // the new one's is written.
    loads.record(old, 0, row(0), t0, LoadSource::Stream);
    loads.record(new, 0, row(0), t0, LoadSource::Stream);
    assert_eq!((loads.ranks(old), loads.ranks(new)), (0, 1));
    assert_eq!(store.load_entries(), 1);
}

#[test]
fn a_load_write_for_an_id_that_never_registered_is_dropped() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let stranger = WorkerId(77);
    let t0 = Instant::now();
    loads.record(stranger, 0, row(0), t0, LoadSource::Scrape);
    loads.report(&[(stranger, 0, Some(row(0)))], t0, LoadSource::Poll);
    assert_eq!(loads.ranks(stranger), 0);
    assert_eq!(store.load_entries(), 0);
}

/// The removal and a writer racing it: whichever takes the table first, no
/// row is on file once the removal has returned.
#[test]
fn a_writer_racing_the_removal_finds_no_row_after_it_returns() {
    for _ in 0..200 {
        let store = MemoryStore::new();
        let id = ready(&store, "http://r:8000");
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..50 {
                    store.record(id, 0, row(0), t0, LoadSource::Stream);
                    store.report(&[(id, 1, Some(row(0)))], t0, LoadSource::Poll);
                }
            });
            scope.spawn(|| {
                store.remove(id, revision).expect("removed");
            });
        });
        assert_eq!(LoadSink::ranks(&store, id), 0);
        assert_eq!(store.load_entries(), 0);
    }
}

#[test]
fn an_evicted_member_has_nothing_on_file_and_takes_its_next_report() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://w:8000");
    let loads: &dyn LoadSink = &store;
    let t0 = Instant::now();
    loads.record(id, 0, row(0), t0, LoadSource::Stream);
    loads.evict(id);
    assert_eq!(loads.ranks(id), 0);
    assert_eq!(
        loads.poll_mode(id, t0, Duration::from_secs(10)),
        PollMode::Poll,
        "its receipts went too"
    );
    loads.report(&[(id, 0, Some(row(0)))], t0, LoadSource::Poll);
    assert_eq!(loads.ranks(id), 1, "a member's next report is written");
}

/// A poll's row stands in for a silent rank's sample but never for its
/// receipt: the poll decision keeps asking while that rank stays silent,
/// whatever its neighbour pushes.
#[test]
fn fallback_polling_stays_while_a_silent_rank_stays_silent() {
    let store = MemoryStore::new();
    let sink: &dyn LoadSink = &store;
    let id = ready(&store, "http://w:8000");
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    let row = LoadRow::default();
    sink.record(id, 0, row, t0, LoadSource::Stream);
    sink.record(id, 1, row, t0, LoadSource::Stream);
    assert_eq!(
        sink.poll_mode(id, t0 + Duration::from_secs(1), interval),
        PollMode::Suppressed
    );
    let later = t0 + interval + Duration::from_secs(1);
    sink.record(id, 0, row, later, LoadSource::Stream);
    assert_eq!(
        sink.poll_mode(id, later, interval),
        PollMode::Fallback,
        "rank 1 fell silent"
    );
    sink.report(
        &[(id, 0, Some(row)), (id, 1, Some(row))],
        later,
        LoadSource::Poll,
    );
    assert_eq!(
        sink.poll_mode(id, later, interval),
        PollMode::Fallback,
        "the poll's row for rank 1 is no record of its own"
    );
    assert_eq!(
        sink.load(id, 1).expect("the poll's row").source,
        LoadSource::Poll
    );
    sink.record(
        id,
        0,
        row,
        later + Duration::from_secs(1),
        LoadSource::Stream,
    );
    assert_eq!(
        sink.poll_mode(id, later + Duration::from_secs(1), interval),
        PollMode::Fallback
    );
    sink.record(
        id,
        1,
        row,
        later + Duration::from_secs(2),
        LoadSource::Stream,
    );
    assert_eq!(
        sink.poll_mode(id, later + Duration::from_secs(2), interval),
        PollMode::Suppressed,
        "both ranks push again"
    );
}

/// Formatting the store takes each table's guard once.
#[test]
fn the_store_formats_without_holding_a_table_twice() {
    let store = MemoryStore::new();
    let shown = format!("{store:?}");
    assert!(shown.starts_with("MemoryStore"), "{shown}");
}

#[test]
fn a_record_replaces_its_rank_alone_and_a_poll_fills_only_the_ranks_that_stayed_quiet() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let id = ready(&store, "http://w:8000");
    let t0 = Instant::now();
    loads.record(id, 0, row(10), t0, LoadSource::Stream);
    loads.record(id, 1, row(0), t0, LoadSource::Stream);
    assert_eq!(loads.ranks(id), 2);
    let rank0 = loads.load(id, 0).expect("rank 0");
    assert_eq!(rank0.sampled_at, t0 - Duration::from_millis(10));
    assert_eq!(SignalSource::from(rank0.source), SignalSource::Events);
    let polled_at = t0 + Duration::from_millis(50);
    loads.record(
        id,
        1,
        LoadRow {
            running: 9,
            ..row(0)
        },
        polled_at + Duration::from_millis(1),
        LoadSource::Stream,
    );
    let mut polled = row(0);
    polled.running = 2;
    loads.report(
        &[
            (id, 0, Some(polled)),
            (id, 1, Some(polled)),
            (id, 2, Some(polled)),
        ],
        polled_at,
        LoadSource::Poll,
    );
    assert_eq!(
        loads.load(id, 0).expect("rank 0").row.running,
        2,
        "a quiet rank takes the poll's row"
    );
    assert_eq!(
        loads.load(id, 1).expect("rank 1").row.running,
        9,
        "a rank that pushed since the poll began keeps the stream's"
    );
    assert_eq!(
        loads.load(id, 2).map(|sample| sample.source),
        Some(LoadSource::Poll),
        "a rank the stream never named stands on the poll"
    );
    loads.record(
        id,
        0,
        row(0),
        polled_at + Duration::from_secs(1),
        LoadSource::Stream,
    );
    assert_eq!(loads.load(id, 0).expect("rank 0").row.running, 1);
    loads.evict(id);
    assert_eq!(loads.ranks(id), 0);
}

#[test]
fn a_report_answers_the_positions_of_the_rows_it_passed_over() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let id = ready(&store, "http://w:8000");
    let stranger = WorkerId(77);
    let t0 = Instant::now();
    loads.record(id, 0, row(0), t0, LoadSource::Stream);
    loads.record(id, 1, row(0), t0, LoadSource::Stream);
    let polled = LoadRow {
        running: 2,
        ..row(0)
    };
    let quiet = t0 + Duration::from_millis(50);
    assert_eq!(
        loads.report(
            &[(id, 0, Some(polled)), (id, 1, Some(polled))],
            quiet,
            LoadSource::Poll,
        ),
        Vec::<usize>::new(),
        "no rank pushed since the poll began: both rows are written"
    );
    // Rank 1 pushes while the next poll is out.
    let polled_at = quiet + Duration::from_secs(1);
    let pushed = LoadRow {
        running: 9,
        ..row(0)
    };
    loads.record(
        id,
        1,
        pushed,
        polled_at + Duration::from_millis(1),
        LoadSource::Stream,
    );
    assert_eq!(
        loads.report(
            &[
                (stranger, 0, Some(polled)),
                (id, 0, Some(polled)),
                (id, 1, Some(polled)),
            ],
            polled_at,
            LoadSource::Poll,
        ),
        vec![2],
        "the row of the rank that pushed, by its position: a stranger's row is dropped and rank 0's is written"
    );
    assert_eq!(loads.load(id, 1).expect("rank 1").row.running, 9);
    assert_eq!(
        loads.report(
            &[(id, 0, Some(polled)), (id, 1, None)],
            polled_at,
            LoadSource::Poll,
        ),
        Vec::<usize>::new(),
        "a rank that pushed and brought no row is not on the list"
    );
    assert_eq!(
        loads.report(&[(id, 1, Some(polled))], polled_at, LoadSource::Poll),
        vec![0],
        "an answer of passed-over rows alone says them too"
    );
    // Both ranks push: each row passed over is said, in the answer's order.
    loads.record(
        id,
        0,
        pushed,
        polled_at + Duration::from_millis(2),
        LoadSource::Stream,
    );
    assert_eq!(
        loads.report(
            &[
                (id, 1, Some(polled)),
                (stranger, 0, Some(polled)),
                (id, 0, Some(polled)),
            ],
            polled_at,
            LoadSource::Poll,
        ),
        vec![0, 2]
    );
    assert_eq!(loads.load(id, 0).expect("rank 0").row.running, 9);
}

#[test]
fn the_poll_mode_reads_the_receipts_of_the_pushing_ranks_alone() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let id = ready(&store, "http://w:8000");
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    assert_eq!(loads.poll_mode(id, t0, interval), PollMode::Poll);
    loads.report(&[(id, 0, Some(row(0)))], t0, LoadSource::Poll);
    assert_eq!(
        loads.poll_mode(id, t0, interval),
        PollMode::Poll,
        "a poll's own row is not the stream working"
    );
    loads.record(id, 0, row(0), t0, LoadSource::Stream);
    loads.record(id, 1, row(0), t0, LoadSource::Relay(GatewayId(4)));
    assert_eq!(
        loads.poll_mode(id, t0 + Duration::from_secs(5), interval),
        PollMode::Suppressed
    );
    loads.record(
        id,
        0,
        row(0),
        t0 + Duration::from_secs(20),
        LoadSource::Stream,
    );
    assert_eq!(
        loads.poll_mode(id, t0 + Duration::from_secs(21), interval),
        PollMode::Fallback,
        "one stale pushing rank falls back to the poll"
    );
}

/// A scrape is the gateway's own read, not the engine pushing: its record
/// writes no receipt, so the poll it came from stays.
#[test]
fn a_scrape_writes_no_receipt_so_the_poll_stays() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let id = ready(&store, "http://w:8000");
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    loads.record(id, 0, row(4), t0, LoadSource::Scrape);
    assert_eq!(
        loads.load(id, 0).map(|sample| sample.source),
        Some(LoadSource::Scrape),
        "the row is held"
    );
    assert_eq!(
        loads.poll_mode(id, t0 + Duration::from_secs(1), interval),
        PollMode::Poll
    );
    loads.record(id, 1, row(0), t0, LoadSource::Stream);
    assert_eq!(
        loads.poll_mode(id, t0 + Duration::from_secs(1), interval),
        PollMode::Suppressed,
        "the pushing rank's receipt is the one read"
    );
}

/// A row whose signals ask for events, at `rank`: the store indexes its
/// blocks.
fn publisher(store: &MemoryStore, key: &str, rank: u16) -> WorkerId {
    let facts = WorkerFacts {
        dp_rank: rank,
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..facts(key, 1)
    };
    store.upsert(facts).id()
}

/// What the batch of `sequence` carries in these cases: one block of its
/// own at a chain's start, so the blocks a rank holds count the batches
/// applied since its last clear.
fn one_block(sequence: u64) -> [KvEvent; 1] {
    [KvEvent::Stored {
        parent: None,
        blocks: vec![StoredBlock {
            seq_hash: SequenceHash(1_000 + sequence),
            content_hash: ContentHash(2_000 + sequence),
        }],
    }]
}

#[test]
fn a_reorder_that_moves_the_first_model_moves_the_rows_lane_with_it() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = [KvEvent]> = &store;
    let mut stated = WorkerFacts {
        models: vec!["m".to_owned(), "n".to_owned()],
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..facts("w", 1)
    };
    let id = store.upsert(stated.clone()).id();
    for seq in 1..=3 {
        sink.admit(id, 0, batch(seq), &one_block(seq));
    }
    let held = || (store.kv().held_blocks("m"), store.kv().held_blocks("n"));
    assert_eq!(held(), (3, 0));
    // The same two the other way round, written in place. A row's events
    // are keyed by its first model, and the lane moves with it in the same
    // write: no block stays under the old key, and the lane made anew is
    // unheard, which tells a collector to feed it from a snapshot.
    stated.models.reverse();
    assert_eq!(store.upsert(stated).id(), id);
    assert_eq!(held(), (0, 0));
    assert_eq!(sink.cursor(id, 0), None);
    assert_eq!(
        sink.admit(id, 0, batch(1), &one_block(1)),
        BatchOutcome::Applied
    );
    assert_eq!(held(), (0, 1));
}

#[test]
fn kv_batches_apply_in_sequence_skip_duplicates_and_ask_one_replay_per_gap() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = [KvEvent]> = &store;
    let id = publisher(&store, "w", 0);
    assert_eq!(sink.cursor(id, 0), None, "a rank never heard from");
    assert_eq!(
        sink.admit(id, 0, batch(17), &one_block(17)),
        BatchOutcome::Applied
    );
    assert_eq!(
        sink.admit(id, 0, batch(18), &one_block(18)),
        BatchOutcome::Applied
    );
    assert_eq!(
        sink.admit(id, 0, batch(18), &one_block(18)),
        BatchOutcome::Skipped,
        "a duplicate"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 18);
    assert_eq!(
        sink.admit(id, 0, batch(25), &one_block(25)),
        BatchOutcome::Gap {
            expected: 19,
            received: 25
        }
    );
    assert!(sink.cursor(id, 0).expect("heard").replay_pending);
    for seq in 19..=25 {
        assert_eq!(
            sink.admit(id, 0, batch(seq), &one_block(seq)),
            BatchOutcome::Applied,
            "the replay fills the gap"
        );
    }
    assert_eq!(store.indexed_blocks(id, 0), 9);
    assert_eq!(
        sink.admit(id, 0, batch(40), &one_block(40)),
        BatchOutcome::Gap {
            expected: 26,
            received: 40
        }
    );
    assert_eq!(
        sink.admit(id, 0, batch(50), &one_block(50)),
        BatchOutcome::Applied,
        "a second gap is lost history, applied"
    );
    let cursor = sink.cursor(id, 0).expect("heard");
    assert!(cursor.degraded && !cursor.replay_pending);
    assert_eq!(
        store.indexed_blocks(id, 0),
        10,
        "a small gap keeps the state"
    );
    let second = publisher(&store, "w/1", 1);
    assert_eq!(
        sink.admit(second, 1, batch(3), &one_block(3)),
        BatchOutcome::Applied
    );
    assert_eq!(
        sink.cursor(second, 1).expect("heard").applied,
        3,
        "each rank its own counter"
    );
    assert_eq!(
        sink.admit(id, 1, batch(3), &one_block(3)),
        BatchOutcome::Skipped,
        "a batch of a rank the row is not"
    );
    assert_eq!(
        sink.admit(WorkerId(9), 0, batch(1), &one_block(1)),
        BatchOutcome::Skipped,
        "a batch of a worker the store holds no row for"
    );
}

/// The same sequence below the cursor is a late duplicate on the connection
/// that produced it and a restarted publisher on a fresh one; neither a
/// clear nor the restart window is needed to tell them apart.
#[test]
fn a_fresh_connection_below_the_cursor_is_a_restart_where_a_late_duplicate_is_not() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = [KvEvent]> = &store;
    let id = publisher(&store, "w", 0);
    for seq in 1..=5 {
        assert_eq!(
            sink.admit(id, 0, batch(seq), &one_block(seq)),
            BatchOutcome::Applied
        );
    }
    assert_eq!(
        sink.admit(id, 0, batch(3), &one_block(3)),
        BatchOutcome::Skipped,
        "a late duplicate on the same connection"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 5);
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 0);
    sink.reconnected(id);
    assert_eq!(
        sink.admit(id, 0, batch(3), &one_block(3)),
        BatchOutcome::Applied,
        "the same sequence on a fresh connection is a restarted publisher"
    );
    let cursor = sink.cursor(id, 0).expect("heard");
    assert_eq!((cursor.applied, cursor.degraded), (3, false));
    assert_eq!(store.indexed_blocks(id, 0), 1, "the rank was cleared first");
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    assert_eq!(
        sink.admit(id, 0, batch(4), &one_block(4)),
        BatchOutcome::Applied,
        "the new life continues from there"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 4);
}

#[test]
fn a_publisher_restart_and_a_resync_clear_the_rank_and_a_removal_forgets_it() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = [KvEvent]> = &store;
    let id = publisher(&store, "w", 0);
    for seq in 1..=5 {
        assert_eq!(
            sink.admit(id, 0, batch(seq), &one_block(seq)),
            BatchOutcome::Applied
        );
    }
    let clearing = BatchFacts {
        sequence: 2,
        clears: true,
        snapshot: None,
    };
    assert_eq!(
        sink.admit(id, 0, clearing, &one_block(2)),
        BatchOutcome::Applied,
        "a clear below the cursor is a restarted publisher"
    );
    assert_eq!(store.indexed_blocks(id, 0), 1, "the rank was cleared first");
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    sink.reconnected(id);
    assert_eq!(
        sink.admit(id, 0, batch(1), &one_block(1)),
        BatchOutcome::Applied,
        "below the cursor on a fresh connection: a restart too"
    );
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 2);
    for seq in 2..=4 {
        sink.admit(id, 0, batch(seq), &one_block(seq));
    }
    assert_eq!(store.indexed_blocks(id, 0), 4);
    sink.resync(id, Some(0), ResyncReason::OutOfRange);
    assert_eq!(store.indexed_blocks(id, 0), 0);
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 0);
    assert_eq!(store.resyncs(ResyncReason::OutOfRange), 1);
    let chunk = BatchFacts {
        sequence: 90,
        clears: false,
        snapshot: Some(SnapshotChunk {
            unknown_before: true,
        }),
    };
    sink.resync(id, Some(0), ResyncReason::Snapshot);
    assert_eq!(
        sink.admit(id, 0, chunk, &one_block(90)),
        BatchOutcome::Applied
    );
    let cursor = sink.cursor(id, 0).expect("heard");
    assert_eq!((cursor.applied, cursor.degraded), (90, true));
    assert_eq!(store.resyncs(ResyncReason::Snapshot), 1);
    assert_eq!(
        sink.admit(id, 0, batch(91), &one_block(91)),
        BatchOutcome::Applied,
        "live batches continue after the snapshot's stamp"
    );
    assert_eq!(store.indexed_blocks(id, 0), 2);
    sink.removed(id);
    assert_eq!(sink.cursor(id, 0), None);
    assert_eq!(store.indexed_blocks(id, 0), 0);
    sink.set_block_size("m", 16);
    sink.set_block_size("m", 8);
    assert_eq!(sink.block_size("m"), Some(16), "only ever raised");
    sink.set_block_size("m", 32);
    assert_eq!(sink.block_size("m"), Some(32));
}

#[test]
fn a_reservation_is_all_or_nothing_within_the_window_and_released_by_its_lease() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://d:8000");
    let rooms: &dyn ReservationStore = &store;
    assert_eq!(rooms.window(id, ReservationKind::Room), Some(4));
    let (a, b, c) = (LeaseId(1), LeaseId(2), LeaseId(3));
    assert!(rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 3, a));
    assert!(
        !rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 2, b),
        "five rooms do not fit a window of four"
    );
    assert_eq!(
        rooms.held(id, ReservationKind::Room),
        3,
        "nothing was claimed"
    );
    assert!(rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, b));
    assert!(
        !rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, b),
        "one reservation per kind per lease"
    );
    assert!(rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Slot, 1, b));
    rooms.release(GatewayId::LOCAL, a);
    rooms.release(GatewayId::LOCAL, a);
    assert_eq!(rooms.held(id, ReservationKind::Room), 1);
    assert!(rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 3, c));
    rooms.release(GatewayId::LOCAL, b);
    assert_eq!(rooms.held(id, ReservationKind::Room), 3);
    assert_eq!(rooms.held(id, ReservationKind::Slot), 0);
    // A live worker without a bound has no window and the gate abstains; a
    // worker the store does not know, or dropped, is refused outright.
    let unbounded = store
        .upsert(WorkerFacts {
            max_running: None,
            ..facts("http://p:8000", 1)
        })
        .id();
    assert_eq!(rooms.window(unbounded, ReservationKind::Room), None);
    assert!(
        rooms.try_reserve(
            GatewayId::LOCAL,
            unbounded,
            ReservationKind::Room,
            100,
            LeaseId(9)
        ),
        "no window: the gate abstains"
    );
    let unknown = WorkerId(77);
    assert_eq!(rooms.window(unknown, ReservationKind::Room), None);
    assert!(
        !rooms.try_reserve(
            GatewayId::LOCAL,
            unknown,
            ReservationKind::Room,
            1,
            LeaseId(10)
        ),
        "a worker the store does not know is refused"
    );
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
    assert!(
        !rooms.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, LeaseId(11)),
        "a removed worker is refused whatever its window was"
    );
}

#[test]
fn two_gateways_sharing_a_lease_number_hold_separate_reservations() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://d:8000");
    let rooms: &dyn ReservationStore = &store;
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    assert!(rooms.try_reserve(g1, id, ReservationKind::Room, 2, LeaseId(1)));
    assert!(
        rooms.try_reserve(g2, id, ReservationKind::Room, 2, LeaseId(1)),
        "the other gateway's lease of the same number is its own"
    );
    assert_eq!(rooms.held(id, ReservationKind::Room), 4);
    rooms.release(g1, LeaseId(1));
    assert_eq!(
        rooms.held(id, ReservationKind::Room),
        2,
        "one gateway's release frees its own claim alone"
    );
    rooms.release(g2, LeaseId(1));
    assert_eq!(rooms.held(id, ReservationKind::Room), 0);
}

/// The manual preset as `generation`, its keys pinned for `ttl_s` and swept
/// every `sweep_s`.
#[cfg(feature = "controller")]
fn manual_ruled(generation: u64, ttl_s: u32, sweep_s: u32) -> crate::spec::AppliedSpec {
    use crate::spec::{AppliedSpec, PolicySpec};

    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = generation;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (ttl_s, sweep_s);
    AppliedSpec::validate(&document).expect("the document validates")
}

#[cfg(feature = "controller")]
#[test]
fn the_pins_sweep_drops_each_pin_past_its_own_end_once_an_interval() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let t = |seconds: u64| now + Duration::from_secs(seconds);
    let sessions: &dyn SessionsStore = &store;
    // No sessions rule ever stood: no pass runs, whatever the end of a
    // pin.
    sessions.pin(1, WorkerId(1), now, Some(store.micros(t(60))));
    assert_eq!(store.sweep_pins(t(1_000_000)), 0);
    store
        .apply(manual_ruled(1, 60, 30))
        .expect("a first generation");
    for key in 2..=1_001 {
        sessions.pin(key, WorkerId(1), t(10), Some(store.micros(t(70))));
    }
    // One key its rule keeps with no TTL.
    sessions.pin(1_002, WorkerId(1), t(10), None);
    assert_eq!(sessions.len(), 1_002);
    // The first pass: the pin that ends at second 60 is past it at 61.
    assert_eq!(store.sweep_pins(t(61)), 1);
    // A tick inside the interval walks nothing, though a thousand pins
    // ended.
    assert_eq!(store.sweep_pins(t(80)), 0);
    assert_eq!(sessions.len(), 1_001);
    // The next pass due: the thousand pins that ended at second 70 leave
    // together, and the pin with no end stays.
    assert_eq!(store.sweep_pins(t(91)), 1_000);
    assert_eq!(sessions.len(), 1);
    // A later generation whose one rule keeps its keys for ever sweeps
    // nothing of its own: the earlier interval stands, and a pin an
    // earlier rule ended still leaves.
    store
        .apply(manual_ruled(2, u32::MAX, u32::MAX))
        .expect("a later generation");
    sessions.pin(1, WorkerId(1), t(100), Some(store.micros(t(110))));
    assert_eq!(store.sweep_pins(t(121)), 1);
    assert_eq!(sessions.len(), 1);
}

/// The pins outlive generations, and the interval read at a tick is the
/// standing generation's: an apply that shortens the sweep waits out no
/// earlier interval, however long the one the last pass ran under.
#[cfg(feature = "controller")]
#[test]
fn a_shorter_sweep_applied_waits_out_no_earlier_interval() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let t = |seconds: u64| now + Duration::from_secs(seconds);
    let sessions: &dyn SessionsStore = &store;
    store
        .apply(manual_ruled(1, 7_200, 3_600))
        .expect("a first generation");
    sessions.pin(1, WorkerId(1), now, Some(store.micros(t(60))));
    // The first pass runs at once and finds no pin past its end.
    assert_eq!(store.sweep_pins(t(1)), 0);
    // A generation that sweeps every thirty seconds: its pass is due
    // thirty seconds after the last one, not an hour after it.
    store
        .apply(manual_ruled(2, 60, 30))
        .expect("a later generation");
    assert_eq!(store.sweep_pins(t(30)), 0, "inside the standing interval");
    assert_eq!(store.sweep_pins(t(61)), 1);
    assert!(sessions.is_empty());
}

/// A generation that sweeps, replaced before the sweep ticked once: its
/// interval was learned at its apply, so the pins it ended still leave
/// under a generation that sweeps nothing.
#[cfg(feature = "controller")]
#[test]
fn a_generation_replaced_before_any_pass_still_leaves_its_interval() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let t = |seconds: u64| now + Duration::from_secs(seconds);
    let sessions: &dyn SessionsStore = &store;
    store
        .apply(manual_ruled(1, 60, 30))
        .expect("a first generation");
    sessions.pin(1, WorkerId(1), now, Some(store.micros(t(60))));
    store
        .apply(manual_ruled(2, u32::MAX, u32::MAX))
        .expect("a later generation");
    assert_eq!(store.sweep_pins(t(61)), 1);
    assert!(sessions.is_empty());
}

/// Two standing rules, one that keeps its keys for ever and one that ends
/// them: the sweep runs at the interval of the rule that sweeps, each pin
/// leaves at its own end, and a pin with none stays.
#[cfg(feature = "controller")]
#[test]
fn a_pin_past_its_own_end_leaves_where_another_rule_keeps_its_keys_for_ever() {
    use crate::spec::{AppliedSpec, PolicySpec};

    let store = MemoryStore::new();
    let now = Instant::now();
    let t = |seconds: u64| now + Duration::from_secs(seconds);
    let sessions: &dyn SessionsStore = &store;
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    let mut for_ever = document.bindings[0].clone();
    for_ever.matcher.model = "kept-for-ever".into();
    let rule = for_ever.sessions.as_mut().expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (u32::MAX, u32::MAX);
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (60, 30);
    document.bindings.insert(0, for_ever);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("a first generation");
    sessions.pin(1, WorkerId(1), now, Some(store.micros(t(60))));
    sessions.pin(2, WorkerId(1), now, None);
    assert_eq!(store.sweep_pins(t(61)), 1);
    assert_eq!(sessions.lookup(1), None);
    assert!(sessions.lookup(2).is_some());
}

#[test]
fn a_removed_workers_pins_of_record_leave_with_its_row() {
    let store = MemoryStore::new();
    let (a, b) = (
        ready(&store, "http://a:8000"),
        ready(&store, "http://b:8000"),
    );
    let sessions: &dyn SessionsStore = &store;
    let now = Instant::now();
    sessions.pin(11, a, now, None);
    sessions.pin(12, b, now, None);
    sessions.pin(13, a, now, None);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    // The removal's own write drops them: no sweep ran, and no rule
    // stands.
    assert!(sessions.lookup(11).is_none() && sessions.lookup(13).is_none());
    assert_eq!(sessions.lookup(12).map(|pin| pin.worker), Some(b));
    assert_eq!(sessions.len(), 1);
    // A replaced incarnation's row is gone as well, and so is its pin.
    let replaced = store.upsert(facts("http://b:8000", 2)).id();
    assert_ne!(replaced, b);
    assert!(sessions.is_empty(), "the old incarnation's pin");
}

/// The door of a writer that knows a key's rule by its lease alone: the
/// worker and the last use move as at the plain door, and the end that
/// stood stays where it is the later one. No end is later than any
/// instant.
#[test]
fn a_pin_by_a_leases_own_rule_never_ends_earlier_than_the_pin_that_stood() {
    let store = MemoryStore::new();
    let sessions: &dyn SessionsStore = &store;
    let now = Instant::now();
    let later = now + Duration::from_secs(1);
    let ends = |seconds: u64| Some(store.micros(now + Duration::from_secs(seconds)));
    let pin = |worker, last_used, ends| {
        Some(Pin {
            worker: WorkerId(worker),
            last_used,
            ends,
        })
    };
    sessions.pin_no_earlier(0, WorkerId(1), now, ends(5));
    assert!(sessions.is_empty(), "the absent key is never pinned");
    // No pin stood: the end is the writer's own.
    sessions.pin_no_earlier(11, WorkerId(1), now, ends(5));
    assert_eq!(sessions.lookup(11), pin(1, now, ends(5)));
    // An earlier end leaves the later one, and the pin moves.
    sessions.pin_no_earlier(11, WorkerId(2), later, ends(3));
    assert_eq!(sessions.lookup(11), pin(2, later, ends(5)));
    // A later end is taken.
    sessions.pin_no_earlier(11, WorkerId(3), now, ends(7));
    assert_eq!(sessions.lookup(11), pin(3, now, ends(7)));
    // No end is the latest: it is taken, and then it stays.
    sessions.pin_no_earlier(11, WorkerId(4), now, None);
    assert_eq!(sessions.lookup(11), pin(4, now, None));
    sessions.pin_no_earlier(11, WorkerId(5), now, ends(9));
    assert_eq!(sessions.lookup(11), pin(5, now, None));
    // The plain door moves a pin's end either way.
    sessions.pin(11, WorkerId(6), now, ends(1));
    assert_eq!(sessions.lookup(11), pin(6, now, ends(1)));
}

#[test]
fn a_pin_is_written_at_the_commit_and_leaves_with_its_worker_or_at_its_end() {
    let store = MemoryStore::new();
    let sessions: &dyn SessionsStore = &store;
    let now = Instant::now();
    let ends = |seconds: u64| Some(store.micros(now + Duration::from_secs(seconds)));
    sessions.pin(0, WorkerId(1), now, ends(1));
    assert!(sessions.is_empty(), "the absent key is never pinned");
    sessions.pin(11, WorkerId(1), now, ends(1));
    sessions.pin(12, WorkerId(2), now + Duration::from_secs(5), ends(6));
    sessions.pin(11, WorkerId(3), now + Duration::from_secs(1), ends(2));
    assert_eq!(
        sessions.lookup(11).map(|pin| pin.worker),
        Some(WorkerId(3)),
        "a re-pin moves"
    );
    sessions.merge(
        12,
        Pin {
            worker: WorkerId(9),
            last_used: now,
            ends: None,
        },
    );
    assert_eq!(
        sessions.lookup(12).map(|pin| pin.worker),
        Some(WorkerId(2)),
        "a merge keeps the later use"
    );
    sessions.merge(
        13,
        Pin {
            worker: WorkerId(9),
            last_used: now,
            ends: None,
        },
    );
    assert_eq!(sessions.len(), 3);
    assert_eq!(sessions.drop_worker(WorkerId(9)), 1);
    let two = store.micros(now + Duration::from_secs(2));
    assert_eq!(sessions.expire(two), 1);
    assert_eq!(sessions.lookup(11), None);
    sessions.unpin(12);
    assert!(sessions.is_empty());
}

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
    assert_eq!(expired[0].lease.id, LeaseId(2));
    assert_eq!(
        expired[0].state,
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

#[test]
fn a_pairs_record_stands_until_its_decode_side_completes() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let gateway = GatewayId(1);
    let mut pair = lease(gateway, 5, 0, now);
    pair.decode = Some(WorkerId(8));
    assert_eq!(leases.issue(pair), Ok(()));
    let mut outcome = Outcome {
        side: Side::Prefill,
        served: WorkerId(7),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    assert!(leases.completed(gateway, LeaseId(5), &outcome));
    assert!(
        leases.get(gateway, LeaseId(5)).is_some(),
        "the room lives on"
    );
    outcome.side = Side::Decode;
    assert!(leases.completed(gateway, LeaseId(5), &outcome));
    assert!(leases.get(gateway, LeaseId(5)).is_none());
}

#[test]
fn hello_pages_restore_exactly_what_they_list_and_a_gone_gateway_loses_its_leases() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    leases.set_term(5);
    for id in 1..=3 {
        assert_eq!(leases.issue(lease(g1, id, 5, now)), Ok(()));
    }
    assert_eq!(leases.issue(lease(g2, 1, 5, now)), Ok(()));
    let pages = [
        lease(g1, 2, 4, now),
        lease(g1, 9, 4, now),
        lease(g1, 10, 6, now),
        lease(g2, 5, 4, now),
    ];
    assert_eq!(
        leases.restore(g1, &pages),
        2,
        "what the pages list, under the term, for this gateway"
    );
    assert!(leases.get(g1, LeaseId(1)).is_none(), "omitted: released");
    assert!(leases.get(g1, LeaseId(2)).is_some());
    assert!(
        leases.get(g1, LeaseId(9)).is_some(),
        "an older term is kept"
    );
    assert!(
        leases.get(g1, LeaseId(10)).is_none(),
        "a term ahead is skipped"
    );
    assert!(
        leases.get(g2, LeaseId(5)).is_none(),
        "another gateway's page"
    );
    assert_eq!(leases.live(g2), 1, "the other gateway is untouched");
    assert_eq!(leases.drop_gateways(&[g1]), [2]);
    assert_eq!(leases.live(g1), 0);
    assert_eq!(leases.live(g2), 1);
}

#[test]
fn the_identity_rule_reads_a_key_through_its_fingerprint() {
    assert_eq!(
        identity("http://w:8000", 3),
        identity_of(fingerprint(b"http://w:8000"), 3),
        "the wrapper and the rule agree"
    );
    assert_ne!(identity_of(0, 0), WorkerId(0), "an id is never 0");
}

#[test]
fn a_noted_veto_is_written_only_while_its_note_stands() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let (first, second) = (now + Duration::from_secs(30), now + Duration::from_secs(60));
    let g = GatewayId(1);
    let w = ready(&store, "http://a:8000");

    // No note behind the write: written as the plain door writes it.
    let written = store.gateway_veto_noted(g, w, Some(first), false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    // The note's end stands in the store: the later end is written.
    let written = store.gateway_veto_noted(g, w, Some(second), true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, first), "the later end is shown");

    // The row leaves and comes back under the same id, and its vetoes left
    // with it. A write that says its note stands writes nothing: the row
    // registered again takes no end of the row before it.
    let revision = WorkersStore::get(&store, w).expect("the row").revision;
    WorkersStore::remove(&store, w, revision).expect("removed");
    assert_eq!(ready(&store, "http://a:8000"), w, "the same id");
    for until in [Some(second), None] {
        let written = store.gateway_veto_noted(g, w, until, true, now);
        assert_eq!(written, VetoWrite::Left);
    }
    assert!(!store.vetoed(g, w, now));
    assert_eq!(store.health_entries(), 0);
    // With no note behind it the same end is the gateway's word of now,
    // and a clear behind a note that stands lifts it.
    let written = store.gateway_veto_noted(g, w, Some(second), false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    let written = store.gateway_veto_noted(g, w, None, true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(!store.vetoed(g, w, now));

    // A worker without a row: a veto writes nothing, a clear is taken.
    let none = WorkerId(7);
    let written = store.gateway_veto_noted(g, none, Some(first), false, now);
    assert_eq!(written, VetoWrite::NoRow);
    let written = store.gateway_veto_noted(g, none, None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(store.health_entries(), 0);
}

#[test]
fn a_dropped_gateway_takes_its_vetoes_and_completed_dispatches_and_no_other_gateways() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let until = now + Duration::from_secs(60);
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let a = ready(&store, "http://a:8000");
    let b = ready(&store, "http://b:8000");
    for gateway in [g1, g2] {
        store.gateway_veto(gateway, a, Some(until));
        HealthSink::completed(&store, gateway, b, TransportClass::Ok, now);
    }
    store.gateway_veto(g1, b, Some(until));
    assert_eq!(store.health_entries(), 5);

    assert_eq!(store.drop_gateways(&[g1]), [0], "it held no lease");
    assert!(!store.vetoed(g1, a, now) && !store.vetoed(g1, b, now));
    assert!(!store.completed_ok_since(g1, b, now));
    // The other gateway's stand.
    assert!(store.vetoed(g2, a, now));
    assert!(store.completed_ok_since(g2, b, now));
    assert_eq!(store.health_entries(), 2);
    // A gateway that holds nothing is dropped as cleanly.
    assert_eq!(store.drop_gateways(&[g1]), [0]);
    assert_eq!(store.health_entries(), 2);
}

#[test]
fn gateways_dropped_together_leave_whole_and_say_their_leases_in_the_order_asked() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let until = now + Duration::from_secs(60);
    let (g1, g2, g3) = (GatewayId(1), GatewayId(2), GatewayId(3));
    let a = ready(&store, "http://a:8000");
    let b = ready(&store, "http://b:8000");
    leases.set_term(5);
    let two = [lease(g1, 1, 5, now), lease(g1, 2, 5, now)];
    assert_eq!(leases.restore(g1, &two), 2);
    assert_eq!(leases.restore(g2, &[lease(g2, 3, 5, now)]), 1);
    assert_eq!(leases.restore(g3, &[lease(g3, 4, 5, now)]), 1);
    for gateway in [g1, g2, g3] {
        store.gateway_veto(gateway, a, Some(until));
        HealthSink::completed(&store, gateway, b, TransportClass::Ok, now);
    }
    assert_eq!(store.health_entries(), 6);

    // Asked out of order, and one of them twice: each count is its own
    // gateway's, where the caller named it.
    assert_eq!(leases.drop_gateways(&[g3, g1, g3]), [1, 2, 1]);
    for gateway in [g1, g3] {
        assert_eq!(leases.live(gateway), 0);
        assert!(!store.vetoed(gateway, a, now));
        assert!(!store.completed_ok_since(gateway, b, now));
    }
    // The gateway the list did not name keeps what it held.
    assert_eq!(leases.live(g2), 1);
    assert!(store.vetoed(g2, a, now) && store.completed_ok_since(g2, b, now));
    assert_eq!(store.health_entries(), 2);
    // Asked in falling order the list is found whole all the same: the
    // store sorts what it searches.
    let falling = [GatewayId(6), GatewayId(5), GatewayId(4)];
    for gateway in falling {
        store.gateway_veto(gateway, a, Some(until));
    }
    assert_eq!(leases.drop_gateways(&falling), [0, 0, 0]);
    for gateway in falling {
        assert!(!store.vetoed(gateway, a, now), "{gateway:?} keeps its veto");
    }
    assert!(leases.drop_gateways(&[]).is_empty());
    assert_eq!(store.health_entries(), 2);
}

/// No lease's counter: what an issue that ended nothing hands back.
const NONE: [u64; 0] = [];

/// Each member's requests in flight as a view of the cases' model reads
/// them for `gateway`, in the view's order: by worker id.
fn in_flight_for(store: &MemoryStore, gateway: GatewayId) -> Vec<(WorkerId, u32)> {
    let ask = Ask {
        gateway,
        now: Instant::now(),
        pinned: None,
        affinity: false,
    };
    let mut out = ViewBuffers::default();
    store.deciding(|world| {
        if let Some(built) = world.build("m", Role::Single, &ask, &mut out) {
            world.count_in_flight(&built, &ask, &mut out.candidates);
        }
    });
    let rows = out.candidates.iter();
    let written: Vec<(WorkerId, u32)> = rows.map(|row| (row.id, row.in_flight)).collect();
    // The store's view reads the same counts: beside the rows it lends,
    // each at its row's position, and in the rows it writes.
    let mut viewed = ViewBuffers::default();
    let read: Vec<(WorkerId, u32)> = store.deciding(|world| {
        let view = world.view("m", Role::Single, &ask, false, &mut viewed);
        view.map_or_else(Vec::new, |view| {
            let rows = view.candidates.iter();
            let lent = view.in_flight;
            rows.map(|row| {
                let own = row.in_flight;
                (row.id, lent.map_or(own, |lent| lent.at(row.position)))
            })
            .collect()
        })
    });
    assert_eq!(read, written, "the view's counts are the door's");
    written
}

/// [`in_flight_for`] the gateway the cases issue their leases to.
fn in_flight(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
    in_flight_for(store, GatewayId(1))
}

/// A lease of `gateway` on `target`, issued at `now` under term 1.
fn booking(gateway: GatewayId, id: u64, target: WorkerId, now: Instant) -> LeaseRecord {
    LeaseRecord {
        target,
        ..lease(gateway, id, 1, now)
    }
}

#[test]
fn a_views_row_reads_the_leases_on_file_that_book_its_worker() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (first, second) = (a.min(b), a.max(b));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    assert_eq!(in_flight(&store), [(first, 0), (second, 0)]);
    for (id, target) in [(1, a), (2, a), (3, b)] {
        LeasesStore::issue(&store, booking(gateway, id, target, now)).expect("filed");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(2, 1));
    // A dispatch moves no count: the lease stands on file.
    assert!(LeasesStore::dispatched(
        &store,
        gateway,
        LeaseId(1),
        Dispatch::default(),
        now
    ));
    assert_eq!(in_flight(&store), of(2, 1));
    // A count is its worker's through a membership write and a status.
    let c = ready(&store, "http://c:1");
    let mut rows = in_flight(&store);
    rows.retain(|&(id, _)| id != c);
    assert_eq!(rows, of(2, 1), "a member that joined moved no count");
    assert!(in_flight(&store).contains(&(c, 0)));
    // A lease of a worker that left ends on its own count, and no other
    // member's moves.
    let revision = WorkersStore::get(&store, b).expect("the row").revision;
    assert!(store.remove(b, revision).is_some());
    let leases: &dyn LeasesStore = &store;
    assert!(leases.released(gateway, LeaseId(3), None, ReleaseReason::WorkerRemoved));
    let mut left = [(a, 2), (c, 0)];
    left.sort_unstable();
    assert_eq!(in_flight(&store), left);
}

/// A replica that restores holds no lease. What it filed in an earlier
/// lead is another history's: the file is empty behind the restore, its
/// deadline queue with it, and every count reads zero. Each lease's end
/// is told once, as a frame's release of a removed worker's lease is.
/// When the replica leads again the gateway's `Hello` files what it
/// lists, and each count reads that list once.
#[test]
fn a_restore_empties_the_lease_file_and_a_hello_books_what_it_lists_once() {
    let store = MemoryStore::new();
    store.tell_through(heard);
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    for (id, target) in [(1, a), (2, a), (3, b)] {
        LeasesStore::issue(&store, booking(gateway, id, target, now)).expect("filed");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(2, 1));
    assert_eq!(store.queued(), 3);
    // Another replica led over the same rows, and this one restores from
    // its cut.
    let leader = MemoryStore::new();
    let theirs = (ready(&leader, "http://a:1"), ready(&leader, "http://b:1"));
    assert_eq!(theirs, (a, b));
    drop(told());
    assert!(store.restore_from(leader.cut(Kinds::All)));
    let leases: &dyn LeasesStore = &store;
    assert_eq!(leases.live(gateway), 0, "a lease stood behind the restore");
    assert!(leases.get(gateway, LeaseId(1)).is_none());
    assert_eq!(store.queued(), 0);
    assert_eq!(in_flight(&store), of(0, 0));
    // The three ends, each told once with the record the file held.
    let removed = Said::Released(None, ReleaseReason::WorkerRemoved);
    let mut ended: Vec<u64> = told()
        .into_iter()
        .map(|(record, said)| {
            assert_eq!(said, removed);
            assert_eq!(record.state, LeaseState::Issued);
            record.lease.id.0
        })
        .collect();
    ended.sort_unstable();
    assert_eq!(ended, [1, 2, 3]);
    // The file is empty: a second restore tells no end.
    assert!(store.restore_from(leader.cut(Kinds::All)));
    assert!(told().is_empty(), "a lease was told twice");
    // The lead taken again: the gateway lists the two leases it still
    // holds, one a worker.
    store.set_term(3);
    let listed = [booking(gateway, 1, a, now), booking(gateway, 3, b, now)];
    assert_eq!(leases.restore(gateway, &listed), 2);
    assert_eq!(in_flight(&store), of(1, 1));
    // An end the gateway reports then gives its count back.
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0, 1));
}

/// A restore tells each lease's end under the decider whose file holds
/// the lease, as a frame's end is told, whichever thread restores.
#[test]
fn a_restore_tells_each_end_under_the_decider_whose_file_holds_its_lease() {
    let store = MemoryStore::new().with_deciders(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    let (now, world) = (Instant::now(), store.world());
    let issue = || {
        let record = booking(GatewayId(1), 0, a, now);
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let named = {
        let _acting = Decider::new(3).enter();
        issue()
    };
    let unnamed = issue();
    // Another replica led over the same row, and this one restores from
    // its cut on a thread that names none.
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    assert!(store.restore_from(leader.cut(Kinds::All)));
    let mut ended: Vec<(u64, Option<Decider>)> = told()
        .into_iter()
        .map(|(record, _)| record.lease.id.0)
        .zip(told_by())
        .collect();
    ended.sort_unstable();
    let first = Some(Decider::new(0));
    let mut issued = [(named.0, Some(Decider::new(3))), (unnamed.0, first)];
    issued.sort_unstable();
    assert_eq!(ended, issued);
}

/// A journal that takes every place, and whose restart waits inside the
/// restore's locks: it says that it stands there, then waits for the
/// case's word.
struct WaitsAtRestart {
    reached: mpsc::Sender<()>,
    go: Mutex<mpsc::Receiver<()>>,
}

impl Journal for WaitsAtRestart {
    fn member(
        &self,
        _: Option<Position>,
        _: u64,
        _: MembershipChange,
        _: Option<&WorkerRow>,
        _: Option<&WorkerRow>,
    ) {
    }

    fn load(&self, _: Option<Position>, _: WorkerId, _: u16, _: &LoadSample) {}

    fn term_rose(&self, _: u64) {}

    fn term_followed(&self, _: Position) {}

    fn carries_loads(&self) -> bool {
        false
    }

    fn head(&self, _: Kinds) -> Position {
        Position::default()
    }

    fn followed(&self) -> Option<u64> {
        None
    }

    fn restart(&self, _: Position) -> bool {
        // A case that is gone is waited for no longer.
        let _ = self.reached.send(());
        if let Ok(word) = self.go.lock() {
            let _ = word.recv();
        }
        true
    }
}

/// A take beside a restore stands whole behind it. The take's write waits
/// for its file's turn, and the restore holds every file's turn from its
/// start until it has read the ids on file: the first lease of the term
/// taken is filed behind that read, on the restored row's count, and the
/// restore ends no lease of that term.
#[test]
fn a_take_beside_a_restore_files_its_first_lease_behind_it() {
    let (reached, stands) = mpsc::channel();
    let (go, word) = mpsc::channel();
    let store = MemoryStore::journaled(Arc::new(WaitsAtRestart {
        reached,
        go: Mutex::new(word),
    }));
    let a = ready(&store, "http://a:1");
    let (gateway, now) = (GatewayId(1), Instant::now());
    // The lead before: one lease on file.
    store.set_term(1);
    LeasesStore::issue(&store, booking(gateway, 1, a, now)).expect("filed");
    // Another replica led over the same row, and this one restores from
    // its cut.
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    let cut = leader.cut(Kinds::All);
    let store = &store;
    let (inside, restored, filed) = std::thread::scope(move |scope| {
        let restore = scope.spawn(move || store.restore_from(cut));
        // The restore stands inside its locks, at its journal's restart.
        stands
            .recv_timeout(Duration::from_secs(10))
            .expect("the restore reaches its journal");
        // The replica takes the lease there: the take's write, then the
        // first lease of its term.
        let (landed, lands) = mpsc::channel();
        let take = scope.spawn(move || {
            store.set_term(2);
            let first = LeaseRecord {
                target: a,
                ..lease(gateway, 2, 2, now)
            };
            let filed = LeasesStore::issue(store, first);
            let _ = landed.send(());
            filed
        });
        // Nothing of the take lands while the restore stands there.
        let inside = lands.recv_timeout(Duration::from_millis(200)).is_ok();
        go.send(()).expect("the restore waits at its journal");
        (
            inside,
            restore.join().expect("the restore"),
            take.join().expect("the take"),
        )
    });
    assert!(!inside, "the take landed inside the restore");
    assert!(restored);
    filed.expect("the first lease of the term taken is filed");
    // The file holds that lease, on the restored row's count, and nothing
    // of the lead before.
    let leases: &dyn LeasesStore = store;
    assert!(leases.get(gateway, LeaseId(1)).is_none());
    assert!(leases.get(gateway, LeaseId(2)).is_some());
    assert_eq!(leases.term(), 2);
    assert_eq!(in_flight(store), [(a, 1)]);
}

/// A pool that holds another gateway's host-local row is walked as its
/// build walks it: each row the asking gateway reaches reads its own
/// worker's count, and a build alone writes none.
#[test]
fn a_count_is_written_on_its_own_workers_row_past_another_gateways_local_row() {
    let store = MemoryStore::new();
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let member = |key: &str, reachable_from| {
        let id = store
            .upsert(WorkerFacts {
                reachable_from,
                ..facts(key, 1)
            })
            .id();
        store.publish_now();
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store
            .set_status(id, WorkerStatus::Ready, revision)
            .expect("ready");
        id
    };
    let open = member("http://a:1", Reachability::Any);
    let theirs = member("unix:///b", Reachability::Gateway(GatewayId(2)));
    let ours = member("unix:///c", Reachability::Gateway(gateway));
    let last = member("http://d:1", Reachability::Any);
    // No two workers hold the same count.
    let mut id = 0;
    for (count, target) in [(1, open), (2, theirs), (3, ours), (4, last)] {
        for _ in 0..count {
            id += 1;
            LeasesStore::issue(&store, booking(gateway, id, target, now)).expect("filed");
        }
    }
    let by_id = |mut rows: Vec<(WorkerId, u32)>| {
        rows.sort_unstable();
        rows
    };
    let seen = |asker: u64| in_flight_for(&store, GatewayId(asker));
    assert_eq!(seen(1), by_id(vec![(open, 1), (ours, 3), (last, 4)]));
    assert_eq!(seen(2), by_id(vec![(open, 1), (theirs, 2), (last, 4)]));
    assert_eq!(seen(3), by_id(vec![(open, 1), (last, 4)]));
    let mut built = ViewBuffers::default();
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: false,
    };
    store.build_view("m", Role::Single, &ask, &mut built);
    assert!(built.candidates.iter().all(|row| row.in_flight == 0));
}

/// The counts are one table in pieces that never move: every slot names a
/// cell of its own in whichever piece holds it, a run of slots is read in
/// its order across the pieces, and the slot that names no cell reads zero
/// and books nothing.
#[test]
fn a_count_is_found_by_its_slot_in_whichever_piece_holds_it() {
    let counts = Counts::default();
    // The first piece holds 1,024 cells and the second 2,048: one slot
    // more stands in the third.
    let slots: Vec<Slot> = (0..3_073).map(|_| counts.claim()).collect();
    for &slot in &slots {
        counts.book(slot);
    }
    assert!(counts.of(&slots).all(|count| count == 1), "a cell a slot");
    for &slot in &slots {
        counts.free(slot);
    }
    // Both ends of each piece, a count of its own on each.
    let marked = [0, 1, 1_023, 1_024, 3_071, 3_072];
    for (times, &at) in (1..).zip(&marked) {
        for _ in 0..times {
            counts.book(slots[at]);
        }
    }
    counts.book(Slot::NONE);
    let ends = marked.iter().rev().map(|&at| slots[at]);
    let run: Vec<Slot> = ends.chain([Slot::NONE, slots[2]]).collect();
    assert_eq!(
        counts.of(&run).collect::<Vec<_>>(),
        [6, 5, 4, 3, 2, 1, 0, 0]
    );
    // A lease that leaves gives one back, and a count at zero stays there.
    for slot in [slots[1_024], slots[2], Slot::NONE] {
        counts.free(slot);
    }
    let read: Vec<u32> = counts.of(&[slots[1_024], slots[2]]).collect();
    assert_eq!(read, [3, 0]);
}

/// A member keeps its slot whichever member leaves or joins beside it, and
/// the slot of a worker that left is given to no member after it: a worker
/// that joins reads zero, and the leaver's leases end on the count they
/// booked.
#[test]
fn a_member_keeps_its_slot_and_a_leavers_slot_is_given_to_no_one() {
    let keys = ["http://a:1", "http://b:1", "http://c:1"];
    for leaver in 0..keys.len() {
        let store = MemoryStore::new();
        let ids: Vec<WorkerId> = keys.iter().map(|key| ready(&store, key)).collect();
        let leases: &dyn LeasesStore = &store;
        let (gateway, now) = (GatewayId(1), Instant::now());
        leases.set_term(1);
        // One lease on the first worker, two on the second, three on the
        // third: the leaver's are the ones its turn filed.
        let (mut id, mut its) = (0, Vec::new());
        for (count, &target) in (1..).zip(&ids) {
            for _ in 0..count {
                id += 1;
                let filed = leases.issue(booking(gateway, id, target, now));
                assert_eq!(filed, Ok(()));
                if target == ids[leaver] {
                    its.push(LeaseId(id));
                }
            }
        }
        let revision = WorkersStore::get(&store, ids[leaver]).expect("the row");
        assert!(store.remove(ids[leaver], revision.revision).is_some());
        let joined = ready(&store, "http://d:1");
        let stayers = (1..).zip(&ids).filter(|&(_, &id)| id != ids[leaver]);
        let mut stand: Vec<(WorkerId, u32)> = stayers.map(|(count, &id)| (id, count)).collect();
        stand.push((joined, 0));
        stand.sort_unstable();
        assert_eq!(in_flight(&store), stand, "worker {leaver} left");
        for lease in its {
            assert!(leases.released(gateway, lease, None, ReleaseReason::WorkerRemoved));
        }
        assert_eq!(in_flight(&store), stand, "worker {leaver}'s leases ended");
    }
}

/// Two leases filed through `file`, one on a member that is built and one
/// on a member whose first build is owed to the window. Each books its
/// worker's count: the row the build lists reads its lease, and each lease
/// ends on the count it booked.
fn filed_ahead_of_a_first_build(file: fn(&dyn LeasesStore, GatewayId, &[LeaseRecord; 2])) {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    // No build is asked for the second worker: its registration's is owed
    // to the window when its lease is filed.
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    assert!(store.publish_due().is_some(), "a build is owed");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let records = [
        booking(gateway, 1, built, now),
        booking(gateway, 2, owed, now),
    ];
    file(leases, gateway, &records);
    store.publish_now();
    let of = |count| {
        let mut rows = [(built, count), (owed, count)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(1));
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
    assert_eq!(in_flight(&store), of(0));
}

/// The issue door asks for the build a lease's worker is owed before it
/// reads the standing world.
#[test]
fn a_lease_issued_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, _, records| {
        for record in records {
            assert_eq!(leases.issue(*record), Ok(()));
        }
    });
}

/// The restore door asks for it too: a gateway's pages may list a lease on
/// a member inside its registration window.
#[test]
fn a_lease_restored_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, gateway, records| {
        assert_eq!(leases.restore(gateway, records), 2);
    });
}

/// The issue door asks for no build where the standing world lists the
/// lease's worker: a build owed to a registration beside it stays with its
/// window. A lease on the worker inside that window asks for it, once.
#[test]
fn a_lease_issued_on_a_listed_worker_leaves_an_owed_build_to_its_window() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    assert!(store.publish_due().is_some(), "a build is owed");
    let builds = store.membership_builds();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, built, now)), Ok(()));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "a lease on a listed worker asked for the build"
    );
    assert_eq!(leases.issue(booking(gateway, 2, owed, now)), Ok(()));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds + 1, false),
        "a lease on a worker inside its window asks once"
    );
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
}

/// A member back under its key and incarnation is one that joined: it is
/// given the table's next slot, and the leases on file for the one that
/// left end on the count they booked.
#[test]
fn a_member_back_under_its_id_is_given_the_next_slot() {
    let store = MemoryStore::new();
    let (stays, leaves) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, leaves, now)), Ok(()));
    }
    let row = WorkersStore::get(&store, leaves).expect("the row");
    assert!(store.remove(leaves, row.revision).is_some());
    // The same key and incarnation after the leave: the same id, and the
    // two leases of the registration that left still on file.
    let back = ready(&store, "http://b:1");
    assert_eq!(back, leaves);
    let mut stand = [(stays, 0), (back, 0)];
    stand.sort_unstable();
    assert_eq!(in_flight(&store), stand);
    for id in 1..=2 {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::WorkerRemoved));
    }
    assert_eq!(in_flight(&store), stand);
}

/// A record filed over one on file under its key gives the first's count
/// back: a restore whose page names one lease twice keeps one lease and
/// one count.
#[test]
fn a_record_filed_over_one_on_file_gives_the_firsts_count_back() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let page = booking(gateway, 7, a, now);
    assert_eq!(leases.restore(gateway, &[page, page]), 1);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

/// A sweep's one write ends the leases of every gateway that goes: each
/// gives its count back, and the leases of a gateway that stays still book
/// their workers.
#[test]
fn a_sweeps_one_write_gives_back_the_counts_of_every_gateway_that_goes() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    leases.set_term(1);
    // Gateways 1 and 3 go, gateway 2 stays with its two leases.
    for (gateway, id, target) in [(1, 1, a), (1, 2, b), (2, 3, a), (2, 4, a), (3, 5, b)] {
        let filed = leases.issue(booking(GatewayId(gateway), id, target, now));
        filed.expect("the key is free");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(3, 2));
    assert_eq!(leases.drop_gateways(&[GatewayId(3), GatewayId(1)]), [1, 2]);
    assert_eq!(in_flight(&store), of(2, 0));
    assert_eq!(leases.live(GatewayId(2)), 2);
}

#[test]
fn a_dispatch_on_another_target_of_the_pick_moves_the_leases_count() {
    let store = MemoryStore::new();
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    // A pick of a, then b: the lease books a.
    let mut targets = Targets::new();
    for id in [a, b] {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let picked = |id: u64| LeaseRecord {
        picked: Picked::of(&targets),
        ..booking(gateway, id, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(picked(1)).expect("filed");
    assert_eq!(in_flight(&store), of(1, 0));
    // The gateway dispatched to the pick's second target: the count is b's.
    let on = |target_index| Dispatch {
        target_index,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on(1), now));
    assert_eq!(in_flight(&store), of(0, 1));
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!((filed.target, filed.lease.target_index), (b, 1));
    // A second dispatch of one lease, on the first target, moves nothing.
    assert!(leases.dispatched(gateway, LeaseId(1), on(0), now));
    assert_eq!(in_flight(&store), of(0, 1));
    // The lease ends on the count it books now.
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0, 0));
    // An index the pick holds no target at moves the state, and neither a
    // count nor the record's index.
    leases.issue(picked(2)).expect("filed");
    assert!(leases.dispatched(gateway, LeaseId(2), on(2), now));
    assert_eq!(in_flight(&store), of(1, 0));
    let filed = leases.get(gateway, LeaseId(2)).expect("on file");
    assert_eq!(
        (filed.state, filed.lease.target_index),
        (LeaseState::Dispatched { at: now }, 0)
    );
}

/// A lease moves at its first dispatch alone: a second one that names a
/// third target of the pick, at a later instant, leaves the instant, the
/// target and the counts as the first set them.
#[test]
fn a_second_dispatch_on_a_third_target_of_the_pick_moves_nothing() {
    let store = MemoryStore::new();
    let workers = ["http://a:1", "http://b:1", "http://c:1"].map(|key| ready(&store, key));
    let [a, b, c] = workers;
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let of = |held: [u32; 3]| {
        let mut rows = [(a, held[0]), (b, held[1]), (c, held[2])];
        rows.sort_unstable();
        rows
    };
    let mut targets = Targets::new();
    for id in workers {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: Picked::of(&targets),
        ..booking(gateway, 1, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(record).expect("filed");
    let on = |target_index| Dispatch {
        target_index,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on(1), now));
    assert_eq!(in_flight(&store), of([0, 1, 0]));
    let later = now + Duration::from_secs(1);
    assert!(leases.dispatched(gateway, LeaseId(1), on(2), later));
    assert_eq!(
        in_flight(&store),
        of([0, 1, 0]),
        "the second target's count"
    );
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!(
        (filed.target, filed.lease.target_index, filed.state),
        (b, 1, LeaseState::Dispatched { at: now })
    );
}

/// One worker at two ranks of a pick: a dispatch on the second names the
/// lease's own worker, so no count moves, and the record's index follows
/// the dispatch all the same.
#[test]
fn a_dispatch_on_another_rank_of_the_leases_worker_moves_the_index_alone() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    let mut targets = Targets::new();
    for dp_rank in [0, 3] {
        targets.push(Target {
            id: a,
            dp_rank,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: Picked::of(&targets),
        ..booking(gateway, 1, a, now)
    };
    let leases: &dyn LeasesStore = &store;
    leases.issue(record).expect("filed");
    let on = Dispatch {
        target_index: 1,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on, now));
    assert_eq!(in_flight(&store), [(a, 1)], "the worker's count stands");
    let filed = leases.get(gateway, LeaseId(1)).expect("on file");
    assert_eq!((filed.target, filed.lease.target_index), (a, 1));
    assert_eq!(filed.target_at(1), Some((a, 3)), "the rank dispatched to");
}

#[test]
fn every_door_that_takes_a_lease_off_the_file_gives_its_count_back() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let ok = Outcome {
        side: Side::Single,
        served: a,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    let key = LeaseId(1);
    let doors: [(&str, &dyn Fn()); 5] = [
        ("a completion", &|| {
            assert!(leases.completed(gateway, key, &ok));
        }),
        ("a release", &|| {
            assert!(leases.released(gateway, key, None, ReleaseReason::ClientGone));
        }),
        ("the dispatch deadline", &|| {
            assert_eq!(leases.expire(now + Duration::from_secs(2)).len(), 1);
        }),
        ("a restore that omits it", &|| {
            assert_eq!(leases.restore(gateway, &[]), 0);
        }),
        ("its gateway's end", &|| {
            assert_eq!(leases.drop_gateways(&[gateway]), [1]);
        }),
    ];
    for (door, through) in doors {
        leases
            .issue(booking(gateway, 1, a, now))
            .expect("the key is free");
        assert_eq!(in_flight(&store), [(a, 1)], "{door}: the lease books");
        through();
        assert_eq!(in_flight(&store), [(a, 0)], "{door}: the count is back");
        assert_eq!(leases.get(gateway, key), None, "{door}");
    }
    // A restore lays the pages' leases on file, each on its worker's count,
    // and a lease written again under its key books once.
    let pages = [booking(gateway, 1, a, now), booking(gateway, 2, a, now)];
    assert_eq!(leases.restore(gateway, &pages), 2);
    assert_eq!(in_flight(&store), [(a, 2)]);
    assert_eq!(leases.restore(gateway, &pages[..1]), 1);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

#[test]
fn the_store_mints_a_decisions_lease_under_its_own_term_alone() {
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let world = store.world();
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
        (second, 3, a, LeaseState::Issued)
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
    let booked = store.world().booked_of(a);
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
    let mut cut = MemoryStore::new().cut(Kinds::All);
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

thread_local! {
    /// What the store told of leases on this thread, in the order told.
    static TOLD: RefCell<Vec<(LeaseRecord, Said)>> = const { RefCell::new(Vec::new()) };
    /// The decider each of those tells named, in the same order.
    static TOLD_BY: RefCell<Vec<Option<Decider>>> = const { RefCell::new(Vec::new()) };
}

/// A teller that keeps its word for [`told`]: the record a frame found on
/// file and what the frame said, and for [`told_by`] the decider named.
fn heard(_: &MemoryStore, _: &World, record: &LeaseRecord, said: Said, by: Option<Decider>) {
    TOLD.with(|told| told.borrow_mut().push((*record, said)));
    TOLD_BY.with(|told| told.borrow_mut().push(by));
}

/// The deciders [`heard`] was told since the last call, taken.
fn told_by() -> Vec<Option<Decider>> {
    TOLD_BY.with(|told| std::mem::take(&mut *told.borrow_mut()))
}

/// What [`heard`] kept on this thread since the last call, taken.
fn told() -> Vec<(LeaseRecord, Said)> {
    TOLD.with(|told| std::mem::take(&mut *told.borrow_mut()))
}

/// An issue at `at` on `target` with five seconds to its dispatch: the
/// counter of its id, and the counters of the leases it ended, as the
/// store told their end.
fn issued_at(store: &MemoryStore, target: WorkerId, at: Instant) -> (u64, Vec<u64>) {
    let counter = |id: LeaseId| id.0 & 0xffff_ffff;
    let record = LeaseRecord {
        dispatch_deadline: at + Duration::from_secs(5),
        ..booking(GatewayId(1), 0, target, at)
    };
    let world = store.world();
    store.tell_through(heard);
    let before = told();
    let lease = store
        .issue_decided(record, world.booked_of(target), |_| ())
        .expect("minted under the store's term");
    // What the issue told: the ends of the leases it found past their
    // deadline, taken here, and the frames it applied, kept for the case.
    let abandoned = Said::Released(None, ReleaseReason::Abandoned);
    let (ended, applied): (Vec<_>, Vec<_>) =
        told().into_iter().partition(|(_, said)| *said == abandoned);
    TOLD.with(|told| told.borrow_mut().extend(before.into_iter().chain(applied)));
    let ended = ended.iter().map(|(record, _)| {
        assert_eq!(record.state, LeaseState::Released(ReleaseReason::Abandoned));
        counter(record.lease.id)
    });
    (counter(lease.id), ended.collect())
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
        let ids = ended.iter().map(|record| record.lease.id.0);
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

/// The pins of record sit in shards by key. A worker's removal and a sweep
/// reach every shard, each under that shard's own hold, and the count is
/// the shards' together.
#[test]
fn a_removal_and_a_sweep_reach_the_pins_of_every_shard() {
    let shards = crate::sessions::SHARDS;
    let store = MemoryStore::new();
    let sessions: &dyn SessionsStore = &store;
    let now = Instant::now();
    let later = now + Duration::from_secs(1);
    // Four keys a shard: two of the first worker used now, two of the
    // second used later.
    let keys = 4 * shards;
    let first = |i: usize| (i / shards).is_multiple_of(2);
    for (i, key) in (1_u64..).take(keys).enumerate() {
        if first(i) {
            sessions.pin(key, WorkerId(1), now, Some(store.micros(now)));
        } else {
            sessions.pin(key, WorkerId(2), later, Some(store.micros(later)));
        }
    }
    assert_eq!(sessions.len(), keys);
    assert_eq!(sessions.drop_worker(WorkerId(1)), keys / 2);
    assert_eq!(sessions.len(), keys / 2);
    for (i, key) in (1_u64..).take(keys).enumerate() {
        assert_eq!(sessions.lookup(key).is_some(), !first(i), "key {key}");
    }
    assert_eq!(sessions.expire(store.micros(later) + 1_000_000), keys / 2);
    assert!(sessions.is_empty());
}

/// A gateway's release of a lease: what the frames of these cases say.
const GONE: Said = Said::Released(None, ReleaseReason::ClientGone);

/// How long a threaded case waits for a word of its other side: far over
/// any run, so a wait that ends by it found a fault.
const PATIENCE: Duration = Duration::from_secs(20);

/// A store under term 1 with one ready worker and the teller that keeps
/// its word, and `leases` leases issued on that worker at `now`: the
/// store, the worker and the leases' ids.
fn handed_bed(leases: u64, now: Instant) -> (Arc<MemoryStore>, WorkerId, Vec<LeaseId>) {
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let ids = (0..leases).map(|_| LeaseId(1 << 32 | issued_at(&store, a, now).0));
    let ids = ids.collect();
    drop(told());
    (store, a, ids)
}

/// What [`heard`] was told since the last call: each lease's counter and
/// what was said of it.
fn told_ids() -> Vec<(u64, Said)> {
    let told = told().into_iter();
    told.map(|(record, said)| (record.lease.id.0 & 0xffff_ffff, said))
        .collect()
}

#[test]
fn a_frame_handed_where_nobody_is_at_work_is_applied_by_its_hand() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    // Told before any other take of the turn, with the record as the frame
    // found it.
    let kept = told();
    assert_eq!(kept.len(), 1);
    assert_eq!(
        (kept[0].0.lease.id, kept[0].0.state, kept[0].1),
        (ids[0], LeaseState::Issued, dispatch)
    );
    // A second dispatch, a second release and a lease not on file tell
    // nothing.
    store.hand(&hand, ids[0], dispatch);
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, LeaseId(99), GONE);
    assert_eq!(told_ids(), [(1, GONE)]);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

#[test]
fn a_frame_handed_while_a_thread_is_at_work_is_left_to_that_thread() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    assert_eq!((told_ids(), in_flight(&store)), (vec![], vec![(a, 3)]));
    // The thread at work applies it before it reads the counts,
    at_work.apply();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 2)])
    );
    // at its next issue,
    store.hand(&hand, ids[1], GONE);
    let (_, ended) = issued_at(&store, a, now);
    assert_eq!((ended, told_ids()), (vec![], vec![(2, GONE)]));
    // and once more as it leaves its work.
    store.hand(&hand, ids[2], GONE);
    assert_eq!(told_ids(), []);
    drop(at_work);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(3, GONE)], vec![(a, 1)])
    );
}

/// The frames of one take are told under one world, the store's as it
/// stood at the take's first tell: a take loads it once, however many
/// frames it tells.
#[test]
fn the_frames_of_a_take_are_told_under_one_world() {
    thread_local! {
        static UNDER: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        static LATER: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }
    // Keeps the epoch of the world each tell is handed; the first tell
    // publishes a later world before it returns.
    fn under(store: &MemoryStore, world: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        UNDER.with_borrow_mut(|under| under.push(world.epoch()));
        if LATER.replace(false) {
            ready(store, "http://b:1");
        }
    }
    let now = Instant::now();
    let store = Arc::new(MemoryStore::new());
    store.tell_through(under);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let ids = [0, 1, 2].map(|_| LeaseId(1 << 32 | issued_at(&store, a, now).0));
    let first = store.world().epoch();
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[1], GONE);
    at_work.apply();
    let later = store.world().epoch();
    assert!(later > first);
    assert_eq!(UNDER.take(), [first, first], "one load for the take");
    store.hand(&hand, ids[2], GONE);
    at_work.apply();
    assert_eq!(UNDER.take(), [later], "the next take's own");
}

/// A decider looks at what is handed before a decision reads its rows'
/// counts, and that look never waits: a turn it finds held it reads past.
#[test]
fn a_thread_at_work_does_not_wait_for_a_held_turn_before_its_reads() {
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (t0, world) = (Instant::now(), store.world());
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    let store = &store;
    std::thread::scope(|scope| {
        // A thread is at work: its first step took the turn while it was
        // free. It takes it once more as it leaves, once the holder let go.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked.send(()).expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // Another holder is parked under the turn, in an issue's tell.
        scope.spawn(move || {
            let record = booking(GatewayId(1), 0, a, t0);
            let issued = store.issue_decided(record, world.booked_of(a), |_| {
                reached.send(()).expect("the case hears the tell");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        // The thread at work looks and reads on.
        look.send(())
            .expect("the thread at work waits for the word");
        let early = seen.recv_timeout(Duration::from_secs(2));
        go.send(()).expect("the issuer holds the other end");
        assert!(early.is_ok(), "the look waited for the turn");
    });
    assert!(store.unlocked());
}

#[test]
fn a_dispatch_handed_in_time_saves_its_lease_from_the_deadlines_end() {
    let t0 = Instant::now();
    let (store, a, ids) = handed_bed(2, t0);
    let hand = Hand::new(GatewayId(1));
    let _at_work = store.at_work();
    let in_time = t0 + Duration::from_secs(1);
    store.hand(
        &hand,
        ids[0],
        Said::Dispatched(Dispatch::default(), in_time),
    );
    // An issue past both deadlines applies what is handed before it ends
    // a deadline: the dispatched lease stands, the other is abandoned.
    let late = t0 + Duration::from_secs(6);
    assert_eq!(issued_at(&store, a, late).1, [2]);
    let state = |id| LeasesStore::get(&*store, GatewayId(1), id).map(|record| record.state);
    assert_eq!(state(ids[0]), Some(LeaseState::Dispatched { at: in_time }));
    assert_eq!(state(ids[1]), None);
}

#[test]
fn a_full_ring_makes_its_caller_apply_and_no_frame_is_dropped() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 1, now);
    let hand = Hand::new(GatewayId(1));
    // A thread at work that does not come by: the frames wait in the ring.
    let at_work = store.at_work();
    for id in &ids {
        store.hand(&hand, *id, GONE);
    }
    // The last found the ring full and applied it, in the order the frames
    // were handed; its own frame waits.
    let applied = told_ids();
    assert_eq!(applied.len(), MemoryStore::HANDOVER);
    assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room));
    drop(at_work);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(room + 1, GONE)], vec![(a, 0)])
    );
}

#[test]
fn a_streams_end_applies_what_it_handed() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let at_work = store.at_work();
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    // The stream's end does not wait for the thread at work.
    store.hand_back(hand);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE), (2, GONE)], vec![(a, 0)])
    );
    drop(at_work);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "taken first or not at all")]
fn a_tell_that_takes_a_door_of_the_leases_is_refused_in_a_debug_build() {
    fn takes_a_door(store: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        let _live = LeasesStore::live(store, GatewayId(1));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(takes_a_door);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.world().booked_of(a), |_| ());
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, issued.expect("minted").id, GONE);
}

/// One hand against one thread that leaves its work, the two started
/// together by a spin and moved a little each round. When both have ended
/// no door of the leases is asked: the handed frame is told by then, by the
/// thread that left or by the hand. A leaver that looked before it said it
/// leaves would let a frame wait on a lull.
#[test]
fn a_frame_handed_as_a_thread_leaves_its_work_is_applied_by_one_of_the_two() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        HEARD.fetch_add(1, SeqCst);
    }
    const ROUNDS: usize = 100_000;
    const BOUND: Duration = Duration::from_secs(60);
    /// Tells the case's two threads to end, where the case unwinds too.
    struct Ends<'a>(&'a AtomicUsize);
    impl Drop for Ends<'_> {
        fn drop(&mut self) {
            self.0.store(usize::MAX, SeqCst);
        }
    }
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(counted);
    let (gateway, now, world) = (GatewayId(1), Instant::now(), store.world());
    let marks = [const { AtomicUsize::new(0) }; 4];
    let lease = AtomicU64::new(0);
    let (mut short, mut ran) = (0_usize, 0_usize);
    std::thread::scope(|scope| {
        let (store, [round, set, go, done], lease) = (&store, &marks, &lease);
        let next = move |seen: usize| loop {
            let asked = round.load(SeqCst);
            if asked != seen {
                break asked;
            }
            std::hint::spin_loop();
        };
        let start = move |asked: usize, spins: usize| {
            set.fetch_add(1, SeqCst);
            // The case's end is a word to go, too.
            while go.load(SeqCst) != asked && round.load(SeqCst) != usize::MAX {
                std::hint::spin_loop();
            }
            for _ in 0..spins {
                std::hint::spin_loop();
            }
        };
        scope.spawn(move || {
            let mut asked = next(0);
            while asked != usize::MAX {
                let at_work = store.at_work();
                start(asked, asked / 24 % 24);
                drop(at_work);
                done.fetch_add(1, SeqCst);
                asked = next(asked);
            }
        });
        scope.spawn(move || {
            let hand = Hand::new(gateway);
            let mut asked = next(0);
            while asked != usize::MAX {
                let id = LeaseId(lease.load(SeqCst));
                start(asked, asked % 24);
                store.hand(&hand, id, GONE);
                done.fetch_add(1, SeqCst);
                asked = next(asked);
            }
        });
        // A round's waits are bounded and the two threads end where this
        // one unwinds: a failure ends the case.
        let _ends = Ends(round);
        let wait = |count: &AtomicUsize, asked: usize| {
            while count.load(SeqCst) != 2 * asked {
                assert!(now.elapsed() < BOUND, "a thread of the case stopped");
                std::hint::spin_loop();
            }
        };
        for asked in 1..=ROUNDS {
            if now.elapsed() > PATIENCE {
                break;
            }
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, world.booked_of(a), |_| ());
            lease.store(issued.expect("minted under the store's term").id.0, SeqCst);
            let before = HEARD.load(SeqCst);
            round.store(asked, SeqCst);
            wait(set, asked);
            go.store(asked, SeqCst);
            wait(done, asked);
            short += usize::from(HEARD.load(SeqCst) == before);
            ran = asked;
        }
    });
    assert!(ran > 0, "no round ran");
    assert_eq!(short, 0, "of {ran} rounds, a handed frame waited on a lull");
}

/// A frame handed while another thread holds the turn is left to the
/// thread at work, and that thread waits for the turn as it leaves: the
/// frame is told once the holder lets go, with no door of the leases
/// asked behind it.
#[test]
fn a_thread_that_leaves_its_work_waits_for_a_held_turn() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        HEARD.fetch_add(1, SeqCst);
    }
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(counted);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let first = store
        .issue_decided(
            booking(gateway, 0, a, now),
            store.world().booked_of(a),
            |_| (),
        )
        .expect("minted under the store's term");
    let (began, working) = mpsc::sync_channel(1);
    let (leave, asked) = mpsc::sync_channel::<()>(1);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (left, gone) = mpsc::sync_channel(1);
    let (early, before) = std::thread::scope(|scope| {
        let store = &store;
        // One thread is at work: a frame handed from here on is left to
        // it. It leaves its work at the case's word, and is not through
        // while the turn is held.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case says when to leave");
            drop(at_work);
            left.send(()).expect("the case waits for the leaver");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // Another holder is parked under the turn, in an issue's hook.
        scope.spawn(move || {
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, store.world().booked_of(a), |_| {
                reached.send(()).expect("the case hears the issue");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        let hand = Hand::new(gateway);
        let before = HEARD.load(SeqCst);
        store.hand(&hand, first.id, GONE);
        leave
            .send(())
            .expect("the thread at work waits for the word");
        let early = gone.recv_timeout(Duration::from_millis(250)).is_ok();
        go.send(()).expect("the issuer holds the other end");
        (early, before)
    });
    assert!(!early, "the leaver was through under a held turn");
    let told = HEARD.load(SeqCst) - before;
    assert_eq!(
        told, 1,
        "the leaver applies what was handed under the held turn"
    );
}

/// A line that prints the store from under the turn, a tell's or a panic's,
/// does not take the turn again: it reads no count of the leases there.
#[test]
fn the_store_is_printed_from_under_its_leases_turn() {
    thread_local!(static PRINTED: RefCell<String> = const { RefCell::new(String::new()) });
    fn prints(store: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        PRINTED.with(|printed| *printed.borrow_mut() = format!("{store:?}"));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(prints);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.world().booked_of(a), |_| ());
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, issued.expect("minted").id, GONE);
    let under = PRINTED.with(|printed| printed.borrow().clone());
    assert!(under.contains("leases: under the turn"), "{under}");
    let free = format!("{store:?}");
    assert!(free.contains("leases: 0"), "{free}");
}

#[test]
fn frames_handed_from_other_threads_are_all_applied_whoever_is_at_work() {
    const STREAMS: usize = 4;
    const ROUNDS: usize = 2_000;
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (gateway, t0) = (GatewayId(1), Instant::now());
    std::thread::scope(|scope| {
        let mut streams = Vec::new();
        for _ in 0..STREAMS {
            let (issued, ids) = mpsc::channel::<LeaseId>();
            let store = &store;
            scope.spawn(move || {
                let hand = Hand::new(gateway);
                for id in ids {
                    store.hand(&hand, id, Said::Dispatched(Dispatch::default(), t0));
                    store.hand(&hand, id, GONE);
                }
                store.hand_back(hand);
            });
            streams.push(issued);
        }
        // The deciding thread: at work for two rounds of three, away for
        // the third, so both the thread at work and the hands apply.
        let world = store.world();
        for round in 0..ROUNDS {
            let _at_work = (round % 3 != 0).then(|| store.at_work());
            for stream in &streams {
                let record = booking(gateway, 0, a, t0);
                let lease = store
                    .issue_decided(record, world.booked_of(a), |_| ())
                    .expect("minted under the store's term");
                stream.send(lease.id).expect("its stream reads on");
            }
        }
    });
    // Every stream applied its ring before it left: no lease stands.
    assert_eq!(LeasesStore::live(&*store, gateway), 0);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A lease's events are told under the decider whose file the lease is on,
/// whichever thread applies them: its issuer's, where the store's host
/// named its deciders. A store of one file that no decider is named for
/// tells under no name.
#[test]
fn a_leases_events_are_told_under_its_files_decider() {
    let issue = |store: &MemoryStore, a| {
        let record = booking(GatewayId(1), 0, a, Instant::now());
        let issued = store.issue_decided(record, store.world().booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    let store = MemoryStore::new().with_deciders(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let fourth = {
        let _acting = Decider::new(3).enter();
        issue(&store, a)
    };
    let first = issue(&store, a);
    // Their ends are applied by a thread that names none.
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, fourth, GONE);
    store.hand(&hand, first, GONE);
    assert_eq!(told_by(), [Some(Decider::new(3)), Some(Decider::new(0))]);
    let plain = MemoryStore::new();
    let a = ready(&plain, "http://a:1");
    LeasesStore::set_term(&plain, 1);
    plain.tell_through(heard);
    let named = {
        let _acting = Decider::new(3).enter();
        issue(&plain, a)
    };
    plain.hand(&Hand::new(GatewayId(1)), named, GONE);
    assert_eq!(told_by(), [None]);
}

/// What a store told on any thread, by the store's address, in the order
/// told: a lease's counter, and what was said of it or none for its issue.
static TOLD_ACROSS: Mutex<Vec<(usize, u64, Option<Said>)>> = Mutex::new(Vec::new());

fn told_across(store: &MemoryStore, lease: LeaseId, said: Option<Said>) {
    let entry = (
        std::ptr::from_ref(store).addr(),
        lease.0 & 0xffff_ffff,
        said,
    );
    TOLD_ACROSS
        .lock()
        .expect("no case fails under it")
        .push(entry);
}

#[test]
fn a_lease_is_told_issued_before_any_holder_ends_it() {
    fn ended(store: &MemoryStore, _: &World, record: &LeaseRecord, said: Said, _: Option<Decider>) {
        told_across(store, record.lease.id, Some(said));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(ended);
    let (t0, world) = (Instant::now(), store.world());
    let issue = |at: Instant, parked: &dyn Fn()| {
        let record = LeaseRecord {
            dispatch_deadline: t0 + Duration::from_millis(1),
            ..booking(GatewayId(1), 0, a, at)
        };
        let issued = store.issue_decided(record, world.booked_of(a), |lease| {
            told_across(&store, lease.id, None);
            parked();
        });
        issued.expect("minted under the store's term");
    };
    let issue = &issue;
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (done, swept) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The first issuer is parked in its tell, under the turn.
        scope.spawn(move || {
            issue(t0, &|| {
                reached.send(()).expect("the case hears the tell");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
        });
        parked
            .recv_timeout(PATIENCE)
            .expect("the first issue tells");
        // A second issue past the first lease's deadline would end it: it
        // waits for the turn, so it has swept nothing a moment later.
        scope.spawn(move || {
            issue(t0 + Duration::from_secs(1), &|| ());
            done.send(()).expect("the case waits for the sweep");
        });
        let early = swept.recv_timeout(Duration::from_millis(100));
        assert!(early.is_err(), "the sweep waits for the issue's tell");
        go.send(()).expect("the issuer holds the other end");
        swept.recv_timeout(PATIENCE).expect("the second issue ends");
    });
    let me = std::ptr::from_ref(&store).addr();
    let told = TOLD_ACROSS.lock().expect("no case fails under it");
    let mine = told.iter().filter(|(by, ..)| *by == me);
    let mine: Vec<_> = mine.map(|(_, lease, said)| (*lease, *said)).collect();
    let abandoned = Some(Said::Released(None, ReleaseReason::Abandoned));
    // The second lease is due as it is filed: its own issue ends it, and
    // tells it issued first.
    assert_eq!(mine, [(1, None), (2, None), (1, abandoned), (2, abandoned)]);
}

/// A store made for its deciders holds a file of leases each. A decider
/// issues into its own file under its own turn and the id it mints names
/// that file: what is handed of a lease is applied there, under that
/// decider's name, while another decider's turn is held.
#[test]
fn a_decider_issues_into_its_own_file_and_a_leases_id_names_it() {
    static NAMES: Mutex<Vec<(LeaseId, Option<Decider>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &World, record: &LeaseRecord, _: Said, by: Option<Decider>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_deciders(3));
    assert_eq!(store.deciders(), 3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = Decider::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (second, third) = (issue(1), issue(2));
    assert!(
        third.0.abs_diff(second.0) > 1,
        "a file mints in a block of its own: {second:?} {third:?}"
    );
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the issuer's
        // wait ends, and the case with it.
        let (parked, go) = (parked, go);
        let store = &store;
        // The third decider holds its own turn, parked in an issue's hook.
        scope.spawn(move || {
            let _acting = Decider::new(2).enter();
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, store.world().booked_of(a), |_| {
                reached.send(()).expect("the case hears the issue");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        // The second's lease is read and ended at its own file meanwhile.
        let on_file = LeasesStore::get(&**store, gateway, second);
        assert_eq!(on_file.map(|record| record.lease.id), Some(second));
        store.hand(&Hand::new(gateway), second, GONE);
        assert_eq!(LeasesStore::get(&**store, gateway, second), None);
        go.send(()).expect("the issuer holds the other end");
    });
    store.hand(&Hand::new(gateway), third, GONE);
    let names = NAMES.lock().expect("no case fails under it");
    let (second, third) = ((second, Decider::new(1)), (third, Decider::new(2)));
    assert_eq!(*names, [second, third].map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(LeasesStore::live(&*store, gateway), 1);
}

/// A restore hands each lease's end to its file. Where the file's decider
/// is at work the restore returns with the lease on file and nothing
/// told: that decider tells the end once, at its next take. A frame
/// handed ahead of the restore for a lease is told in its order, and the
/// restore's end of that lease then tells nothing.
#[test]
fn a_restore_leaves_a_working_deciders_leases_to_its_next_take() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    let cut = leader.cut(Kinds::All);
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen) = (working, look, seen);
        let store = &store;
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // A gateway's frame for the first lease, ahead of the restore.
        store.hand(&Hand::new(GatewayId(1)), ids[0], GONE);
        assert!(store.restore_from(cut));
        // The restore is back: both leases stand and nothing is told.
        assert_eq!(LeasesStore::live(&**store, GatewayId(1)), 2);
        assert_eq!(told_ids(), []);
        look.send(())
            .expect("the thread at work waits for the word");
        let removed = Said::Released(None, ReleaseReason::WorkerRemoved);
        assert_eq!(
            seen.recv_timeout(PATIENCE).expect("the look"),
            [(1, GONE), (2, removed)]
        );
    });
    assert_eq!(LeasesStore::live(&*store, GatewayId(1)), 0);
    assert_eq!(store.queued(), 0);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// While a decider is at work no other thread acts at its file: a
/// stream's end, the sweep of the due leases and a door each find it at
/// work under the turn and apply nothing. What waits is the decider's.
#[test]
fn no_thread_acts_at_a_file_beside_the_thread_at_work() {
    let now = Instant::now();
    let (store, _, ids) = handed_bed(2, now);
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen) = (working, look, seen);
        let store = &store;
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        let hand = Hand::new(GatewayId(1));
        store.hand(&hand, ids[0], GONE);
        // The stream's end, the sweep past every deadline and a door's read.
        store.hand_back(hand);
        let late = now + Duration::from_secs(60);
        assert!(LeasesStore::expire(&**store, late).is_empty());
        assert_eq!(LeasesStore::live(&**store, GatewayId(1)), 2);
        assert_eq!(told_ids(), []);
        // The thread at work applies what was handed at its next look.
        look.send(())
            .expect("the thread at work waits for the word");
        assert_eq!(seen.recv_timeout(PATIENCE).expect("the look"), [(1, GONE)]);
    });
    assert_eq!(LeasesStore::live(&*store, GatewayId(1)), 1);
}

/// A decider that takes up its work waits out a hand that applies at its
/// file. The hand found no thread at work, took the file's turn and tells
/// a lease's event under the decider's name: the decider says it works,
/// then takes that turn, so no thread tells under its name beside its
/// first decision.
#[test]
fn a_decider_takes_up_its_work_behind_a_hand_that_tells_at_its_file() {
    type Ends = (mpsc::SyncSender<()>, mpsc::Receiver<()>);
    static ENDS: Mutex<Option<Ends>> = Mutex::new(None);
    fn parks(_: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        let ends = ENDS.lock().expect("no case fails under it").take();
        if let Some((reached, word)) = ends {
            reached.send(()).expect("the case hears the tell");
            word.recv_timeout(PATIENCE)
                .expect("the case lets the teller go");
        }
    }
    let store = Arc::new(MemoryStore::new().with_deciders(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(parks);
    let (gateway, now) = (GatewayId(1), Instant::now());
    // A lease on the second decider's file.
    let lease = {
        let _acting = Decider::new(1).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    *ENDS.lock().expect("no case fails under it") = Some((reached, word));
    let (began, working) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the teller's
        // wait ends, and the case with it.
        let (parked, go, working) = (parked, go, working);
        let store = &store;
        // A hand finds no thread at work at the file: it takes the turn
        // and stands in the tell of the frame it applies.
        scope.spawn(move || store.hand(&Hand::new(gateway), lease, GONE));
        // Bounded: where no tell is made the case fails, and does not
        // wait on an end the teller holds.
        let told = parked.recv_timeout(PATIENCE);
        told.expect("the hand's tell");
        // The file's decider takes up its work behind that turn.
        scope.spawn(move || {
            let _acting = Decider::new(1).enter();
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            drop(at_work);
        });
        let early = working.recv_timeout(Duration::from_millis(250));
        assert!(early.is_err(), "a decider worked beside a hand's tell");
        go.send(()).expect("the teller holds the other end");
        let at_work = working.recv_timeout(PATIENCE);
        at_work.expect("the decider is at work behind the tell");
    });
    assert_eq!(LeasesStore::live(&*store, gateway), 0);
}

/// A full ring stops its hand until a holder applied: the thread at work
/// at the file, which the hand yields to. No frame is dropped, and none is
/// applied by the hand beside that thread.
#[test]
fn a_full_ring_stops_its_hand_until_the_thread_at_work_applied() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 1, now);
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    let (handed, through) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen, through) = (working, look, seen, through);
        let (store, ids) = (&store, &ids);
        // A thread at work that does not come by until it is asked.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            drop(at_work);
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // A hand fills the ring and stops at the frame past it.
        scope.spawn(move || {
            let hand = Hand::new(GatewayId(1));
            for id in ids {
                store.hand(&hand, *id, GONE);
            }
            handed
                .send(told_ids())
                .expect("the case waits for the hand");
        });
        let early = through.recv_timeout(Duration::from_millis(250));
        assert!(early.is_err(), "the hand was through a full ring");
        look.send(())
            .expect("the thread at work waits for the word");
        let by_hand = through.recv_timeout(PATIENCE);
        let by_hand = by_hand.expect("the hand is through");
        // The ring was the thread at work's to apply. The frame past it
        // is that thread's too, or its hand's once the thread had left:
        // every frame is told once, in the order it was handed.
        let mut applied = seen.recv_timeout(PATIENCE).expect("the look");
        assert!(applied.len() >= MemoryStore::HANDOVER);
        applied.extend(by_hand);
        assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room + 1));
    });
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A file whose decider rests ends no lease at an issue: the sweep ends
/// its leases past their dispatch deadline, each told under the file's
/// decider. A file whose decider is at work is left to it.
#[test]
fn the_sweep_ends_a_resting_files_due_leases_under_its_deciders_name() {
    static NAMES: Mutex<Vec<(LeaseId, Option<Decider>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &World, record: &LeaseRecord, _: Said, by: Option<Decider>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_deciders(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = Decider::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (first, second) = (issue(0), issue(1));
    let late = now + Duration::from_secs(60);
    let ended = |store: &MemoryStore| {
        let ended = LeasesStore::expire(store, late);
        ended
            .iter()
            .map(|record| record.lease.id)
            .collect::<Vec<_>>()
    };
    let (began, working) = mpsc::sync_channel(1);
    let (leave, asked) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the decider's
        // wait ends, and the case with it.
        let (working, leave) = (working, leave);
        let store = &store;
        // The first decider is at work: the sweep leaves its file to it.
        scope.spawn(move || {
            let _acting = Decider::new(0).enter();
            let _at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case says when to leave");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the decider is at work");
        assert_eq!(ended(store), [second]);
        leave.send(()).expect("the decider waits for the word");
    });
    assert_eq!(ended(&store), [first]);
    let names = NAMES.lock().expect("no case fails under it");
    let told = [(second, Decider::new(1)), (first, Decider::new(0))];
    assert_eq!(*names, told.map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A gateway's leases stand in the files of the deciders that issued them:
/// its live count, its restore and its end reach every file.
#[test]
fn a_gateways_doors_reach_its_leases_in_every_file() {
    let store = MemoryStore::new().with_deciders(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (ours, theirs, now) = (GatewayId(1), GatewayId(2), Instant::now());
    let issue = |decider: u8, gateway| {
        let _acting = Decider::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let leases = [issue(0, ours), issue(1, ours), issue(2, ours)];
    let other = issue(1, theirs);
    let doors: &dyn LeasesStore = &store;
    assert_eq!((doors.live(ours), doors.live(theirs)), (3, 1));
    assert_eq!(in_flight(&store), [(a, 4)]);
    // A restore keeps the leases its gateway presents, each in the file
    // its id names, and ends the gateway's others in every file.
    let kept = doors.get(ours, leases[2]).expect("on its decider's file");
    assert_eq!(doors.restore(ours, &[kept]), 1);
    let on_file = leases.map(|lease| doors.get(ours, lease).is_some());
    assert_eq!(on_file, [false, false, true]);
    assert_eq!(in_flight(&store), [(a, 2)]);
    // The gateways' end takes each one's leases out of every file.
    assert_eq!(doors.drop_gateways(&[ours, theirs]), [1, 1]);
    assert_eq!(doors.get(theirs, other), None);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// The start of a term the store follows raises the term on every file
/// and tells the blocks, as the term's own door does: the table of a
/// term that holds a lease is kept, and each file mints under the
/// followed term. A start told again, or an older term's, raises nothing.
#[test]
fn a_followed_terms_start_raises_every_file_and_starts_the_blocks_under_it() {
    let standby = MemoryStore::new().with_deciders(2);
    let term: &dyn LeasesStore = &standby;
    term.set_term(3);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, under: u64| {
        let _acting = Decider::new(decider).enter();
        standby.issue_decided(lease(gateway, 0, under, now), None, |_| ())
    };
    issue(1, 3).expect("minted under the term the standby stood at");
    standby.follow_term(Position { seq: 1, term: 7 });
    assert_eq!(term.term(), 7);
    assert_eq!(
        standby.kept_terms(),
        [3],
        "the rise is told the ids on file"
    );
    for decider in [0, 1] {
        let minted = issue(decider, 7).map(|lease| lease.id.0 >> 32);
        assert_eq!(minted, Ok(7), "the file of decider {decider}");
    }
    standby.follow_term(Position { seq: 2, term: 5 });
    assert_eq!(term.term(), 7);
}

/// A lease's id names its file for as long as the lease can be on file. A
/// frame for a lease of the term before reaches the file of the decider
/// that issued it, behind a rise under which another decider claimed the
/// same block.
#[test]
fn a_lease_of_the_term_before_is_found_in_its_file_behind_a_rise() {
    static NAMES: Mutex<Vec<(LeaseId, Option<Decider>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &World, record: &LeaseRecord, _: Said, by: Option<Decider>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((record.lease.id, by));
    }
    let store = MemoryStore::new().with_deciders(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, term: u64| {
        let _acting = Decider::new(decider).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, term, now)
        };
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    // The third decider claims the first block of term 1, the second
    // decider the first block of term 2.
    let before = issue(2, 1);
    LeasesStore::set_term(&store, 2);
    let after = issue(1, 2);
    assert_eq!(before.0 & 0xffff_ffff, after.0 & 0xffff_ffff);
    assert_eq!(store.kept_terms(), [1]);
    let hand = Hand::new(gateway);
    store.hand(&hand, before, GONE);
    store.hand(&hand, after, GONE);
    let names = NAMES.lock().expect("no case fails under it");
    let told = [(before, Decider::new(2)), (after, Decider::new(1))];
    assert_eq!(*names, told.map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(LeasesStore::live(&store, gateway), 0);
}

/// A streak of lifecycle calls that panicked says its first alone: each is
/// counted, and a call that returns ends the streak.
#[test]
fn a_streak_of_lifecycle_faults_says_its_first_alone() {
    let store = MemoryStore::new();
    assert!(store.call_faulted());
    assert!(!store.call_faulted());
    store.call_returned();
    assert_eq!(store.lifecycle_faults(), 2);
    assert!(store.call_faulted());
    assert_eq!(store.lifecycle_faults(), 3);
}

/// An id names its file for as long as its lease can be on file, so the
/// doors that take an id from their caller file none whose file a later
/// claim would name anew: an id of the store's term from a block no
/// decider claimed, or of a term the store can still take. An id of a
/// term before is filed by its block, and one of a claimed block in the
/// file that claimed it. A store of one file takes every id, as it did.
#[test]
fn a_door_files_no_id_whose_file_a_claim_would_name_anew() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let by_hand = |term: u64, counter: u64| lease(gateway, term << 32 | counter, term.min(3), now);
    let store = MemoryStore::new().with_deciders(2);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 3);
    let doors: &dyn LeasesStore = &store;
    // No decider minted under term 3 yet: its first block is no file's.
    assert_eq!(doors.issue(by_hand(3, 9)), Err(LeaseRefused::Unclaimed));
    assert_eq!(doors.issue(by_hand(4, 9)), Err(LeaseRefused::Unclaimed));
    assert_eq!(doors.restore(gateway, &[by_hand(3, 9), by_hand(4, 9)]), 0);
    // An id of a term before names its file by its block: for good.
    assert_eq!(doors.issue(by_hand(2, 9)), Ok(()));
    assert!(doors.get(gateway, LeaseId(2 << 32 | 9)).is_some());
    // The second decider claims the first block of term 3: an id of that
    // block names its file, and the doors file it there.
    let minted = {
        let _acting = Decider::new(1).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, 3, now)
        };
        let issued = store.issue_decided(record, store.world().booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    assert_eq!(minted, LeaseId(3 << 32 | 1));
    assert_eq!(doors.issue(by_hand(3, 9)), Ok(()));
    assert_eq!(doors.restore(gateway, &[by_hand(3, 9), by_hand(2, 9)]), 2);
    let hand = Hand::new(gateway);
    store.hand(&hand, LeaseId(3 << 32 | 9), GONE);
    assert_eq!(doors.get(gateway, LeaseId(3 << 32 | 9)), None);
    assert_eq!(doors.live(gateway), 1);
    // One file: every id names it.
    let one = MemoryStore::new();
    LeasesStore::set_term(&one, 3);
    assert_eq!(LeasesStore::issue(&one, by_hand(3, 9)), Ok(()));
    assert_eq!(LeasesStore::restore(&one, gateway, &[by_hand(4, 9)]), 1);
}

/// A door asks the file an id names once more under the turn it took. A
/// door that waited for the first file's turn while a decider claimed the
/// id's block files the lease in that decider's file, where its frames
/// look for it. No hook says the door reached its wait, so the case gives
/// it time: a door that came late reads the claimed byte and passes too.
#[test]
fn a_door_that_waited_behind_a_claim_files_its_id_where_the_claim_named_it() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let store = MemoryStore::new().with_deciders(2);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 3);
    let (store, booked) = (&store, store.world().booked_of(a));
    let record = LeaseRecord {
        target: a,
        ..lease(gateway, 0, 3, now)
    };
    // An id of the term's second block, which no decider claimed yet: it
    // reads the first file.
    let by_hand = lease(gateway, 3 << 32 | 1 << 20 | 9, 3, now);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let filed = std::thread::scope(|scope| {
        // The first decider claims the first block and is parked under its
        // turn, in its issue's tell.
        scope.spawn(move || {
            let _acting = Decider::new(0).enter();
            let issued = store.issue_decided(record, booked, |_| {
                reached.send(()).expect("the case hears the tell");
                word.recv().expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv().expect("the issue tells");
        // The door reads the first file and waits for its turn.
        let door = scope.spawn(move || LeasesStore::issue(store, by_hand));
        std::thread::sleep(Duration::from_millis(200));
        // The second decider claims the second block, the id's.
        let _acting = Decider::new(1).enter();
        let second = store.issue_decided(record, booked, |_| ());
        go.send(()).expect("the issuer holds the other end");
        let filed = door.join().expect("the door returns");
        (second.map(|lease| lease.id), filed)
    });
    assert_eq!(filed, (Ok(LeaseId(3 << 32 | 1 << 20)), Ok(())));
    assert!(LeasesStore::get(store, gateway, by_hand.lease.id).is_some());
}

/// Once the store's term is past 31 bits no decider mints and no term
/// leads the blocks anew, so the doors file every id where it reads its
/// file: an id of the last term from a block no decider claimed, and an
/// id of a term that never led here. A restore keeps both.
#[test]
fn the_doors_file_every_id_once_the_term_is_past_31_bits() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let by_hand = |term: u64, counter: u64| lease(gateway, term << 32 | counter, term, now);
    let store = MemoryStore::new().with_deciders(2);
    LeasesStore::set_term(&store, 4);
    let doors: &dyn LeasesStore = &store;
    let (unclaimed, never) = (by_hand(4, 9), by_hand(10, 5 << 20 | 1));
    assert_eq!(doors.issue(unclaimed), Err(LeaseRefused::Unclaimed));
    LeasesStore::set_term(&store, 1 << 31);
    assert_eq!(
        (doors.issue(unclaimed), doors.issue(never)),
        (Ok(()), Ok(()))
    );
    let on_file = |record: LeaseRecord| doors.get(gateway, record.lease.id).is_some();
    assert!(on_file(unclaimed) && on_file(never));
    assert_eq!(doors.restore(gateway, &[unclaimed, never]), 2);
    assert!(on_file(unclaimed) && on_file(never));
}

/// The most deciders a host names each issue a first lease: every id is
/// far below its term's mark, three quarters of the counter, so no count
/// of deciders hands a term over at its first leases.
#[test]
fn the_first_leases_of_the_most_deciders_are_far_below_the_terms_mark() {
    let store = MemoryStore::new().with_deciders(256);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let first = |decider: u8| {
        let _acting = Decider::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id.0 & 0xffff_ffff
    };
    let counters: Vec<u64> = (0..=u8::MAX).map(first).collect();
    let mark = 3_u64 << 30;
    let highest = counters.iter().max().copied().unwrap_or(mark);
    assert!(highest * 12 <= mark, "{highest} beside the mark {mark}");
    assert_eq!(LeasesStore::live(&store, gateway), 256);
    assert_eq!(in_flight(&store), [(a, 256)]);
}

/// The files laid for a host's deciders keep the store's word to the
/// engine and its fault count.
#[test]
fn the_files_laid_for_deciders_keep_the_word_and_the_fault_count() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &World, _: &LeaseRecord, _: Said, _: Option<Decider>) {
        HEARD.fetch_add(1, SeqCst);
    }
    let store = MemoryStore::new();
    store.tell_through(counted);
    assert!(store.call_faulted());
    let store = store.with_deciders(2);
    assert_eq!((store.deciders(), store.lifecycle_faults()), (2, 1));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let record = booking(gateway, 0, a, now);
    let lease = store.issue_decided(record, store.world().booked_of(a), |_| ());
    let lease = lease.expect("minted under the store's term").id;
    store.hand(&Hand::new(gateway), lease, GONE);
    assert_eq!(HEARD.load(SeqCst), 1, "told through the word given before");
}

/// A store with a lease on file is not laid for deciders: the lease's
/// file would go from under it.
#[test]
#[should_panic(expected = "a store's files are laid before a lease is filed")]
fn a_store_with_a_lease_on_file_is_not_laid_for_deciders() {
    let store = MemoryStore::new();
    let filed = LeasesStore::issue(&store, lease(GatewayId(1), 1, 0, Instant::now()));
    filed.expect("a lease");
    drop(store.with_deciders(2));
}

/// Nor is a store laid again with a lease on a later file: the guard
/// reads every file, each under its own turn.
#[test]
#[should_panic(expected = "a store's files are laid before a lease is filed")]
fn a_store_with_a_lease_on_a_later_file_is_not_laid_again() {
    let store = MemoryStore::new().with_deciders(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    {
        // A thread that acts as the fourth decider files on the fourth
        // file: the first file stays empty.
        let (_acting, world) = (Decider::new(3).enter(), store.world());
        let record = booking(GatewayId(1), 0, a, Instant::now());
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term");
    }
    drop(store.with_deciders(8));
}
