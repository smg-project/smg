//! The contract tests of the store ingress, run over [`MemoryStore`] through
//! the trait objects: what every implementation of a trait promises.

use std::{
    cell::{Cell, RefCell},
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

use super::{
    identity, identity_of,
    leases::{InFlightCounts, WorkerSlot},
    AtWork, BatchFacts, BatchOutcome, ContentHash, DecodeMark, DecodeSide, GatewayId, Hand,
    HealthSink, HealthWrite, Journal, KvEvent, KvEventSink, LeaseRecord, LeaseRefused, LeaseState,
    LeasesStore, LoadSink, LoadSource, LookOutcome, MembershipChange, MembershipTail, MemoryStore,
    PickRecord, PollMode, Position, ReservationKind, ReservationStore, ResyncReason, SequenceHash,
    Sides, Signals, SnapshotChunk, StoreSnapshot, StoredBlock, VetoWrite, WorkerFacts, WorkerPatch,
    WorkerRow, WorkerStatus, WorkersStore,
};
use crate::{
    document::spec,
    engine::{
        decision::{self, Target, Targets},
        decision_engine::DecodeRefused,
        decision_slot::DecisionSlot,
        demands::DemandFacts,
        sticky_sessions::SessionVerdict,
        view::{fingerprint, LoadRow, Role, Session, SignalSource, WorkerId},
    },
    service::replication::Log,
    tracking::{
        lease::{
            Dispatch, Lease, LeaseId, Membership, Outcome, ReleaseReason, Said, Side,
            TransportClass, DISPATCH_TTL,
        },
        HostCount,
    },
};

mod host_count;

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
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(1),
        state: LeaseState::ISSUED,
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
    // What a registration states for a hard demand is none of the three
    // facts: its next life is handed the status too.
    let mut other_runtime = facts("http://w:8000", 3);
    other_runtime.demand_facts = DemandFacts {
        runtime: 9,
        ..DemandFacts::default()
    };
    let stated = store.upsert(other_runtime).id();
    assert_eq!(
        WorkersStore::get(&store, stated).expect("the row").status,
        WorkerStatus::Ready
    );
    let mut other_role = facts("http://w:8000", 4);
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
    let (epoch, rows) = workers.all_rows();
    assert_eq!((epoch, rows.len()), (0, 0));
    let a = workers.upsert(facts("http://a:8000", 1)).id();
    let (epoch, rows) = workers.all_rows();
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
    let (head, rows) = workers.all_rows();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, a, "the snapshot is ordered by key");
    assert!(matches!(
        workers.changes_since(head - 1),
        MembershipTail::Changes(changes) if changes.len() == 1
    ));
}

#[test]
fn snapshot_sort_does_not_hold_the_workers_lock() {
    let store = Arc::new(MemoryStore::new());
    let id = store.upsert(facts("http://c:8000", 1)).id();
    store.upsert(facts("http://a:8000", 1));
    store.upsert(facts("http://b:8000", 1));
    let revision = WorkersStore::get(store.as_ref(), id)
        .expect("the row")
        .revision;
    let epoch = store.epoch();
    let (cloned, resume) = store.pause_snapshot_after_clone();

    let snapshot_store = Arc::clone(&store);
    let snapshot = std::thread::spawn(move || WorkersStore::all_rows(snapshot_store.as_ref()));
    cloned
        .recv_timeout(PATIENCE)
        .expect("snapshot reached the sort");

    let writer_store = Arc::clone(&store);
    let (written_tx, written_rx) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        let change = writer_store.set_status(id, WorkerStatus::Ready, revision);
        written_tx
            .send(change)
            .expect("the test receives the write");
    });
    let written_while_paused = written_rx.recv_timeout(PATIENCE);

    resume.send(()).expect("resume the snapshot");
    let (snapshot_epoch, rows) = snapshot.join().expect("snapshot thread");
    writer.join().expect("writer thread");
    assert_eq!(snapshot_epoch, epoch, "epoch was read with the rows");
    assert_eq!(
        rows.iter()
            .map(|row| row.facts.key.as_str())
            .collect::<Vec<_>>(),
        ["http://a:8000", "http://b:8000", "http://c:8000"],
        "snapshot rows are ordered by key"
    );
    assert_eq!(
        rows.iter().find(|row| row.id == id).map(|row| row.status),
        Some(WorkerStatus::Pending),
        "the snapshot keeps the row copied before the status write"
    );
    assert!(
        matches!(
            written_while_paused,
            Ok(Some(MembershipChange::StatusChanged {
                id: changed,
                from: WorkerStatus::Pending,
                to: WorkerStatus::Ready,
            })) if changed == id
        ),
        "status write waited for snapshot sorting"
    );
}

#[test]
fn probe_outcomes_follow_the_rows_thresholds_and_a_stale_outcome_is_dropped() {
    let store = MemoryStore::new();
    let health: &dyn HealthSink = &store;
    let id = store.upsert(facts("http://w:8000", 1)).id();
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        None
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "two passes promote a pending worker"
    );
    // A patch moves the revision: an outcome read under the one it left is
    // stale.
    let stale = revision();
    WorkersStore::patch(&store, id, WorkerPatch::default(), stale).expect("the patch");
    assert_eq!(
        health.probe_outcome(id, false, stale, &|| true).moved(),
        None,
        "a stale outcome is dropped"
    );
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        None
    );
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        None
    );
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady)),
        "three failures demote"
    );
    for _ in 0..8 {
        assert_eq!(
            health
                .probe_outcome(id, false, revision(), &|| true)
                .moved(),
            None
        );
    }
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        Some((WorkerStatus::NotReady, WorkerStatus::Failed)),
        "nine more fail it"
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        None
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        Some((WorkerStatus::Failed, WorkerStatus::Ready)),
        "two passes bring a failed worker back"
    );
    let fresh = store.upsert(facts("http://f:8000", 1)).id();
    let fresh_revision = WorkersStore::get(&store, fresh).expect("the row").revision;
    assert_eq!(
        health.connected(fresh, fresh_revision, &|| true).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "the connect signal promotes at once"
    );
    assert_eq!(
        health.connected(fresh, fresh_revision, &|| true).moved(),
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
                health
                    .probe_outcome(id, false, revision(), &|| true)
                    .moved(),
                None,
                "pending under {threshold}"
            );
        }
        assert_eq!(
            health.probe_outcome(id, true, revision(), &|| true).moved(),
            None
        );
        assert_eq!(
            health.probe_outcome(id, true, revision(), &|| true).moved(),
            Some((WorkerStatus::Pending, WorkerStatus::Ready))
        );
        // And a ready one stays ready through failures under the threshold.
        for _ in 0..5 {
            assert_eq!(
                health
                    .probe_outcome(id, false, revision(), &|| true)
                    .moved(),
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
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        None
    );
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Failed))
    );
    // Each counter at its end takes one probe more and stays there.
    store.probed(id, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        None
    );
    store.probed(id, 0, u32::MAX, u32::MAX);
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
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
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        None
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true).moved(),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    // Ready, the row is not ready at the threshold's own count of failed
    // probes: the count is set one short of it.
    store.probed(id, threshold - 1, 0, 0);
    assert_eq!(
        health
            .probe_outcome(id, false, revision(), &|| true)
            .moved(),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady))
    );
    // Not ready, it fails at three thresholds of failed probes: five are
    // far under them.
    for _ in 0..5 {
        assert_eq!(
            health
                .probe_outcome(id, false, revision(), &|| true)
                .moved(),
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
        health.connected(id, read, &|| true).moved(),
        None,
        "a stale signal is dropped"
    );
    assert_eq!(
        WorkersStore::get(&store, id).expect("the row").status,
        WorkerStatus::Pending,
        "and the row stays as it stood"
    );
    assert_eq!(
        health.connected(id, read + 1, &|| true).moved(),
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
        health.probe_outcome(id, true, revision(), &|| true),
        Stood(WorkerStatus::Pending)
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true),
        Moved {
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        }
    );
    // The connect signal on a ready row stands and clears no counter: two
    // failures ahead of it and one behind it are the threshold's three.
    for _ in 0..2 {
        assert_eq!(
            health.probe_outcome(id, false, revision(), &|| true),
            Stood(WorkerStatus::Ready)
        );
    }
    assert_eq!(
        health.connected(id, revision(), &|| true),
        Stood(WorkerStatus::Ready)
    );
    assert_eq!(
        health.probe_outcome(id, false, revision(), &|| true),
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
    assert_eq!(health.probe_outcome(id, true, left, &|| true), Stale);
    assert_eq!(health.connected(id, left, &|| true), Stale);
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true),
        Stood(WorkerStatus::NotReady),
        "the stale pass was not counted"
    );
    assert_eq!(
        health.probe_outcome(id, true, revision(), &|| true),
        Moved {
            from: WorkerStatus::NotReady,
            to: WorkerStatus::Ready
        }
    );
    // A draining row stands at the signal too.
    WorkersStore::set_status(&store, id, WorkerStatus::Draining, revision()).expect("drained");
    assert_eq!(
        health.connected(id, revision(), &|| true),
        Stood(WorkerStatus::Draining)
    );
    // A write for a worker whose row is gone says no row.
    let last = revision();
    store.remove(id, last).expect("removed");
    assert_eq!(health.probe_outcome(id, true, last, &|| true), NoRow);
    assert_eq!(health.connected(id, last, &|| true), NoRow);
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
fn a_new_row_takes_its_maker_and_a_standing_row_keeps_it() {
    use super::Registrant;

    let store = MemoryStore::new();
    let found = Registrant::provider("cluster", "pods/a").expect("a whole maker");
    let id = store
        .upsert_by(facts("http://w:8000", 1), found.clone())
        .id();
    assert_eq!(WorkersStore::get(&store, id).expect("the row").maker, found);
    // A refresh and a new incarnation keep the maker, whoever calls: the
    // store tests no caller.
    let mut relabelled = facts("http://w:8000", 1);
    relabelled.zone = Some("b".to_string());
    assert_eq!(store.upsert(relabelled).id(), id);
    assert_eq!(WorkersStore::get(&store, id).expect("the row").maker, found);
    let next = store.upsert(facts("http://w:8000", 2)).id();
    assert_ne!(next, id);
    assert_eq!(
        WorkersStore::get(&store, next).expect("the row").maker,
        found
    );
    // A row the plain write makes is a control call's.
    let by_hand = store.upsert(facts("http://x:8000", 1)).id();
    let held = WorkersStore::get(&store, by_hand).expect("the row");
    assert_eq!(held.maker, Registrant::Control);
}

#[test]
fn a_providers_maker_is_made_inside_the_rules_of_its_two_parts() {
    use super::{ProviderKeyError, Registrant};

    let Ok(Registrant::Provider(found)) = Registrant::provider("cluster", "pods/a") else {
        panic!("a maker of both parts is made");
    };
    assert_eq!((found.provider(), found.key()), ("cluster", "pods/a"));
    // Each part at its bound is taken, and the key is its provider's own
    // text, blanks and all.
    let name = "p".repeat(Registrant::NAME_BYTES);
    let key = "k ".repeat(Registrant::KEY_BYTES / 2);
    assert!(Registrant::provider(name.as_str(), key.as_str()).is_ok());
    // The part that breaks its rule is named, the name's rules first.
    let unmade =
        |provider: &str, key: &str| Registrant::provider(provider, key).expect_err("no maker");
    // An empty name is refused in its own words, ahead of the token's.
    assert_eq!(
        unmade("", "pods/a"),
        ProviderKeyError::Provider("a provider's row names its provider")
    );
    assert!(matches!(unmade("", ""), ProviderKeyError::Provider(_)));
    assert!(matches!(
        unmade(&format!("{name}p"), "pods/a"),
        ProviderKeyError::Provider(_)
    ));
    // A name is one token: a blank, a control character, a comma, a letter
    // outside ASCII and a character of no width each make none.
    for name in ["a b", "a\nb", "a\u{7}b", "a,b", "\u{e9}", "a\u{200b}b"] {
        let refused = unmade(name, "pods/a");
        assert!(matches!(refused, ProviderKeyError::Provider(_)), "{name:?}");
    }
    assert!(matches!(unmade("cluster", ""), ProviderKeyError::Key(_)));
    assert!(matches!(
        unmade("cluster", &format!("{key}k")),
        ProviderKeyError::Key(_)
    ));
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
fn manual_ruled(generation: u64, ttl_s: u32, sweep_s: u32) -> spec::AppliedSpec {
    use crate::document::spec::{AppliedSpec, PolicySpec};

    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = generation;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (ttl_s, sweep_s);
    AppliedSpec::validate(&document).expect("the document validates")
}

/// Pins `key` to `worker` in the session table of `store`'s standing
/// engine, as a keyed dispatch does.
fn pinned(store: &MemoryStore, key: u64, worker: WorkerId) {
    let world = store.snapshot();
    let table = world.spec().engine().sessions();
    table.dispatched(Session::from_raw(key), SessionVerdict::Pinned, worker);
}

/// The worker `key`'s first pin names in that table.
fn pin_of(store: &MemoryStore, key: u64) -> Option<WorkerId> {
    let world = store.snapshot();
    let table = world.spec().engine().sessions();
    table.pins(Session::from_raw(key)).map(|pins| pins[0])
}

/// How many keys that table holds.
fn pins_held(store: &MemoryStore) -> usize {
    store.snapshot().spec().engine().sessions().len()
}

/// A removed worker's session pins stand through its absence, and the
/// worker it was takes them back where its key is registered again, under
/// the id the registration makes. A new life of the same worker in place
/// keeps its own. Another worker under the key takes none.
#[test]
fn a_key_registered_again_as_the_worker_it_was_takes_its_pins() {
    let store = MemoryStore::new();
    let (a, b) = (
        ready(&store, "http://a:8000"),
        ready(&store, "http://b:8000"),
    );
    pinned(&store, 11, a);
    pinned(&store, 12, b);
    pinned(&store, 13, a);
    let of = |key: u64| pin_of(&store, key);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    // The removal drops none: the key may be registered again.
    assert_eq!((of(11), of(13)), (Some(a), Some(a)));
    let again = store.upsert(facts("http://a:8000", 2)).id();
    assert_ne!(again, a);
    assert_eq!((of(11), of(13)), (Some(again), Some(again)));
    assert_eq!(of(12), Some(b));
    let life = store.upsert(facts("http://b:8000", 2)).id();
    assert_ne!(life, b);
    assert_eq!(of(12), Some(life), "a new life of the same worker");
    // Removed again, then registered in another role: another worker.
    let revision = WorkersStore::get(&store, again).expect("the row").revision;
    store.remove(again, revision).expect("removed");
    store.upsert(WorkerFacts {
        role: Role::Decode,
        ..facts("http://a:8000", 3)
    });
    assert_eq!((of(11), of(13)), (None, None));
    assert_eq!(pins_held(&store), 1);
    // A new life in another role, in place: another worker too.
    store.upsert(WorkerFacts {
        role: Role::Prefill,
        ..facts("http://b:8000", 3)
    });
    assert_eq!(pins_held(&store), 0, "the old worker's pin");
}

/// Two registrations of one key rename its session pins in their order.
/// The second lands where the first has let the workers lock go, and the
/// pin names the last life: in place, and for a key that left.
#[test]
fn registrations_of_one_key_rename_its_pins_in_their_order() {
    use std::{cell::Cell, rc::Rc};

    use super::memory::seam;

    let store = Rc::new(MemoryStore::new());
    let of = |key: u64| pin_of(&store, key);
    // The third life of `key`, registered behind the next registration.
    let third_behind = |key: &'static str| {
        let third = Rc::new(Cell::new(None));
        let (behind, seen) = (Rc::clone(&store), Rc::clone(&third));
        seam::on_release(move || seen.set(Some(behind.upsert(facts(key, 3)).id())));
        third
    };
    // In place: the key's second life is replaced by its third at once.
    let a = ready(&store, "http://a:8000");
    pinned(&store, 11, a);
    let third = third_behind("http://a:8000");
    store.upsert(facts("http://a:8000", 2));
    assert!(third.get().is_some(), "the hook ran");
    assert_eq!(of(11), third.get());
    // A key that left: registered again, and that life replaced at once.
    let b = ready(&store, "http://b:8000");
    pinned(&store, 12, b);
    let revision = WorkersStore::get(&*store, b).expect("the row").revision;
    store.remove(b, revision).expect("removed");
    let third = third_behind("http://b:8000");
    store.upsert(facts("http://b:8000", 2));
    assert!(third.get().is_some(), "the hook ran");
    assert_eq!(of(12), third.get());
}

/// A key that stays away longer than the absence's time takes no pin
/// back: the time is the longest TTL among the document's session rules.
#[test]
fn a_key_registered_again_past_the_absences_time_takes_no_pin() {
    let store = MemoryStore::new();
    let ruled = manual_ruled(1, 1, 1);
    store.apply(ruled).expect("above the unbound start");
    let a = ready(&store, "http://a:8000");
    pinned(&store, 11, a);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    assert_eq!(pin_of(&store, 11), Some(a), "inside its time");
    std::thread::sleep(Duration::from_millis(1_100));
    store.upsert(facts("http://a:8000", 2));
    assert_eq!(pin_of(&store, 11), None);
}

/// A key that stays away is forgotten at the store's tick, in the engine's
/// table: no membership write and no decision has to follow.
#[test]
fn the_tick_ends_the_absence_of_a_key_that_stays_away() {
    let store = MemoryStore::new();
    let ruled = manual_ruled(1, 1, 1);
    store.apply(ruled).expect("above the unbound start");
    let a = ready(&store, "http://a:8000");
    pinned(&store, 21, a);
    let revision = WorkersStore::get(&store, a).expect("the row").revision;
    store.remove(a, revision).expect("removed");
    store.end_gone(Instant::now());
    assert_eq!(pin_of(&store, 21), Some(a), "inside its time");
    store.end_gone(Instant::now() + Duration::from_millis(1_100));
    assert_eq!(pin_of(&store, 21), None);
}

/// `follower` takes `change`, a write `leader` just made, by the door a
/// standby's applier and a gateway's watcher use.
fn followed(leader: &MemoryStore, follower: &MemoryStore, change: MembershipChange) {
    let row = match change {
        MembershipChange::Registered { id } | MembershipChange::Replaced { new: id, .. } => {
            WorkersStore::get(leader, id)
        }
        _ => None,
    };
    let epoch = WorkersStore::epoch(leader);
    follower.follow(Position::default(), epoch, change, row);
    assert_eq!(WorkersStore::epoch(follower), epoch, "the change landed");
}

/// A follower takes its leader's steps for a key's session pins, in its
/// engine's table. A followed removal drops none. The worker it was takes
/// them back at its followed registration, under the id that registration
/// made. A followed new life of the same worker keeps its own. Another
/// worker under the key takes none.
#[test]
fn a_followed_key_registered_again_as_the_worker_it_was_takes_its_pins() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let join = |facts: WorkerFacts| {
        let change = leader.upsert(facts);
        followed(&leader, &follower, change);
        change.id()
    };
    let leave = |id: WorkerId| {
        let revision = WorkersStore::get(&leader, id).expect("the row").revision;
        let change = leader.remove(id, revision).expect("removed");
        followed(&leader, &follower, change);
    };
    let (a, b) = (
        join(facts("http://a:8000", 1)),
        join(facts("http://b:8000", 1)),
    );
    pinned(&follower, 11, a);
    pinned(&follower, 12, b);
    let of = |key: u64| pin_of(&follower, key);
    leave(a);
    assert_eq!(of(11), Some(a), "none is dropped");
    let again = join(facts("http://a:8000", 2));
    assert_ne!(again, a);
    assert_eq!(of(11), Some(again));
    assert_eq!(of(12), Some(b));
    let life = join(facts("http://b:8000", 2));
    assert_ne!(life, b);
    assert_eq!(of(12), Some(life), "a new life of the same worker");
    // Removed again, then registered in another role: another worker.
    leave(again);
    join(WorkerFacts {
        role: Role::Decode,
        ..facts("http://a:8000", 3)
    });
    assert_eq!(of(11), None);
    assert_eq!(pins_held(&follower), 1);
    // A new life in another role, in place: another worker too.
    join(WorkerFacts {
        role: Role::Prefill,
        ..facts("http://b:8000", 3)
    });
    assert_eq!(pins_held(&follower), 0, "the old worker's pin");
}

/// A follower counts a key's absence by its own clock, from the removal
/// it followed. A key registered again past that time takes no pin, at
/// its followed registration and at a snapshot that holds it again.
#[test]
fn a_followed_key_registered_again_past_the_absences_time_takes_no_pin() {
    let leader = MemoryStore::new();
    let followers = [MemoryStore::new(), MemoryStore::new()];
    let joined = leader.upsert(facts("http://a:8000", 1));
    let revision = WorkersStore::get(&leader, joined.id()).expect("the row");
    for follower in &followers {
        follower
            .apply(manual_ruled(1, 1, 1))
            .expect("above the unbound start");
        followed(&leader, follower, joined);
        pinned(follower, 11, joined.id());
    }
    let removed = leader.remove(joined.id(), revision.revision);
    let removed = removed.expect("removed");
    for follower in &followers {
        followed(&leader, follower, removed);
        assert_eq!(pin_of(follower, 11), Some(joined.id()), "inside its time");
    }
    std::thread::sleep(Duration::from_millis(1_100));
    let again = leader.upsert(facts("http://a:8000", 2));
    followed(&leader, &followers[0], again);
    assert!(followers[1].watch_from(leader.checkpoint()));
    for follower in &followers {
        assert_eq!(pin_of(follower, 11), None);
    }
}

/// A follower files the key of a removal it followed. Where it leads
/// next, its own registration of the key takes the pins.
#[test]
fn a_key_a_follower_saw_leave_takes_its_pins_at_the_followers_own_write() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let joined = leader.upsert(facts("http://a:8000", 1));
    followed(&leader, &follower, joined);
    pinned(&follower, 11, joined.id());
    let revision = WorkersStore::get(&leader, joined.id()).expect("the row");
    let removed = leader.remove(joined.id(), revision.revision);
    followed(&leader, &follower, removed.expect("removed"));
    let again = follower.upsert(facts("http://a:8000", 2)).id();
    assert_eq!(pin_of(&follower, 11), Some(again));
}

/// A snapshot's rows move a key's session pins as the writes it stands
/// for would have, in the engine's table, at a gateway's watch and at a
/// standby's restore. A key the cut holds under a new id
/// takes its pins where it is the worker it was. Another worker's pins
/// leave. A key the cut drops keeps its pins on file, and takes them at
/// its followed registration.
#[test]
fn a_snapshot_moves_the_pins_of_the_rows_it_drops() {
    for restore in [false, true] {
        let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
        let lay = || {
            let cut = leader.checkpoint();
            if restore {
                follower.restore_from(cut)
            } else {
                follower.watch_from(cut)
            }
        };
        let keys = [
            "http://a:8000",
            "http://b:8000",
            "http://c:8000",
            "http://d:8000",
        ];
        let [a, b, c, d] = keys.map(|key| leader.upsert(facts(key, 1)).id());
        assert!(lay());
        for (key, worker) in [(11, a), (12, b), (13, c), (14, d)] {
            pinned(&follower, key, worker);
        }
        let of = |key: u64| pin_of(&follower, key);
        // The follower hears none of these writes.
        let leave = |id: WorkerId| {
            let revision = WorkersStore::get(&leader, id).expect("the row").revision;
            leader.remove(id, revision).expect("removed");
        };
        leave(a);
        let again = leader.upsert(facts("http://a:8000", 2)).id();
        leave(b);
        leader.upsert(WorkerFacts {
            role: Role::Decode,
            ..facts("http://b:8000", 2)
        });
        leave(c);
        assert!(lay());
        assert_eq!(of(11), Some(again));
        assert_eq!(of(12), None, "another worker under the key");
        assert_eq!((of(13), of(14)), (Some(c), Some(d)), "none is dropped");
        let back = leader.upsert(facts("http://c:8000", 2));
        followed(&leader, &follower, back);
        assert_eq!(of(13), Some(back.id()), "the dropped key was on file");
    }
}

/// A key on file when a snapshot comes stays there where the cut does not
/// hold it. Where the cut holds it again as the worker it was, it takes
/// its pins.
#[test]
fn a_snapshot_returns_a_key_that_had_left_to_its_pins() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let [a, b] = ["http://a:8000", "http://b:8000"].map(|key| {
        let change = leader.upsert(facts(key, 1));
        followed(&leader, &follower, change);
        change.id()
    });
    pinned(&follower, 11, a);
    pinned(&follower, 12, b);
    let of = |key: u64| pin_of(&follower, key);
    for id in [a, b] {
        let revision = WorkersStore::get(&leader, id).expect("the row").revision;
        let change = leader.remove(id, revision).expect("removed");
        followed(&leader, &follower, change);
    }
    assert!(follower.watch_from(leader.checkpoint()));
    assert_eq!((of(11), of(12)), (Some(a), Some(b)), "both stay on file");
    let again = leader.upsert(facts("http://a:8000", 2)).id();
    assert!(follower.watch_from(leader.checkpoint()));
    assert_eq!((of(11), of(12)), (Some(again), Some(b)));
    let back = leader.upsert(facts("http://b:8000", 2));
    followed(&leader, &follower, back);
    assert_eq!(of(12), Some(back.id()), "still on file behind two cuts");
}

/// The rows a snapshot drops are filed while the list has room. Past it
/// a row's pins leave at once.
#[test]
fn a_snapshot_files_no_more_keys_than_the_list_holds() {
    use super::memory::GONE_KEYS;
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let ids: Vec<WorkerId> = (0..GONE_KEYS + 2)
        .map(|at| leader.upsert(facts(&format!("http://w{at}:8000"), 1)).id())
        .collect();
    assert!(follower.watch_from(leader.checkpoint()));
    for (key, &id) in (1..).zip(&ids) {
        pinned(&follower, key, id);
    }
    assert!(follower.watch_from(MemoryStore::new().checkpoint()));
    assert_eq!(pins_held(&follower), GONE_KEYS);
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

#[test]
fn a_pairs_record_stands_until_its_decode_side_completes() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let gateway = GatewayId(1);
    let pair = lease(gateway, 5, 0, now);
    let decode = decode_side(WorkerId(8));
    assert_eq!(leases.issue_pair(pair, decode), Ok(()));
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

/// What the leases on file book on each of `workers`, in their order.
fn booked(store: &MemoryStore, workers: &[WorkerId]) -> Vec<u32> {
    let rows = in_flight(store);
    let count = |id: &WorkerId| {
        rows.iter()
            .find(|(row, _)| row == id)
            .map(|(_, count)| *count)
    };
    workers.iter().map(|id| count(id).unwrap_or(0)).collect()
}

/// A pair's record under `id`: the prefill side on `prefill`, the decode
/// side on the first of `decodes`, the rest the pick's alternates.
fn pair(
    id: u64,
    prefill: WorkerId,
    decodes: &[WorkerId],
    now: Instant,
) -> (LeaseRecord, DecodeSide) {
    let mut targets = Targets::new();
    for &decode_id in decodes {
        let pair = decision::Pair {
            decode_id,
            ..Default::default()
        };
        targets.push(Target {
            id: prefill,
            pair: Some(pair),
            ..Target::default()
        });
    }
    let mut record = booking(GatewayId(1), id, prefill, now);
    record.lease.side = Side::Prefill;
    record.picked = PickRecord::of(&targets);
    let decode = DecodeSide {
        picked: PickRecord::of_decodes(&targets),
        ..decode_side(decodes[0])
    };
    (record, decode)
}

/// Files a pair's two parts by the pair's door.
fn issue_pair(
    leases: &dyn LeasesStore,
    (record, decode): (LeaseRecord, DecodeSide),
) -> Result<(), LeaseRefused> {
    leases.issue_pair(record, decode)
}

/// A pair's record states no word of a request kept from its profile: the
/// door that files a pair clears it, as it writes the pair's mark, so both
/// sides of a pair are told whatever a caller wrote. A single's record
/// keeps the word as it was filed.
#[test]
fn the_door_that_files_a_pair_clears_the_word_of_a_kept_request() {
    let store = MemoryStore::new();
    let [p, d] = ["p", "d"].map(|name| ready(&store, &format!("http://{name}:1")));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let kept = LeaseRecord {
        untold: true,
        ..booking(gateway, 1, p, now)
    };
    leases.issue(kept).expect("free");
    let (record, decode) = pair(2, p, &[d], now);
    let kept = LeaseRecord {
        untold: true,
        ..record
    };
    issue_pair(leases, (kept, decode)).expect("free");
    let word = |id| leases.get(gateway, LeaseId(id)).expect("on file").untold;
    assert_eq!((word(1), word(2)), (true, false));
}

/// A dispatch of `side` on the pick's target at `index`.
fn on(side: Side, index: u8) -> Dispatch {
    Dispatch {
        target_index: index,
        side,
        ..Dispatch::default()
    }
}

/// A completion of `side`.
fn done(side: Side) -> Outcome {
    Outcome {
        side,
        served: WorkerId(0),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// Where each side of the pair's lease under `id` stands.
fn sides(store: &MemoryStore, id: u64) -> Option<(LeaseState, LeaseState)> {
    let found = LeasesStore::sides(store, GatewayId(1), LeaseId(id))?;
    Some((found.record.state, found.decode?.state))
}

/// A pair filed with its decode targets chosen.
fn chosen_pair(
    id: u64,
    prefill: WorkerId,
    decodes: &[WorkerId],
    now: Instant,
) -> (LeaseRecord, DecodeSide) {
    let (mut record, mut decode) = pair(id, prefill, decodes, now);
    record.dispatch_deadline = now + DISPATCH_TTL;
    decode.state = LeaseState::Chosen;
    (record, decode)
}

#[test]
fn a_chosen_decode_side_books_nothing_and_waits_under_no_deadline() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    let (record, decode) = chosen_pair(1, p, &[d], now);
    issue_pair(leases, (record, decode)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    assert!(decode.state.is_chosen());
    assert!(!decode.state.is_live());
    assert!(!decode.state.is_issued());
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    assert!(leases.expire(now + Duration::from_secs(9)).is_empty());
    assert_eq!(store.queued(), 0);
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: now }, LeaseState::Chosen))
    );
}

#[test]
fn a_call_issues_the_chosen_side_afresh_and_books_it_once() {
    let store = MemoryStore::new().with_decision_slots(2);
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let (_, mut decode) = pair(0, p, &[d1, d0], now);
    decode.session = Session::from_raw(42);
    decode.tokens = 24;
    decode.target_index = 1;

    let id = LeaseId(1);
    issue_pair(leases, chosen_pair(id.0, p, &[d0], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let prefill = leases.get(gateway, id);
    assert_eq!(leases.issue_decode(gateway, id, decode, now), Ok(()));
    let filed = leases.sides(gateway, id).expect("on file");
    assert_eq!(Some(filed.record), prefill);
    assert_eq!(
        filed.decode,
        Some(DecodeSide {
            target_index: 0,
            ..decode
        })
    );
    assert_eq!(filed.target_of(Side::Decode, 0), Some((d1, 0)));
    assert_eq!(filed.target_of(Side::Decode, 1), Some((d0, 0)));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 1]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A lease of an older term in block one names the second file.
    let id = LeaseId(1 << 20 | 2);
    assert_eq!(store.file_of(id), 1);
    issue_pair(leases, chosen_pair(id.0, p, &[d0], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let mut heard = None;
    assert!(store.unlocked());
    let (issued, _) = store
        .issue_decided_decode(gateway, id, (decode, None), now, |lease| {
            assert!(!store.unlocked(), "the issue is told under the turn");
            heard = Some(*lease);
        })
        .expect("the prefill holds the chosen side");
    assert_eq!(heard, Some(issued));
    assert_eq!(
        (
            issued.id,
            issued.side,
            issued.booked,
            issued.target_index,
            issued.session,
            issued.tokens
        ),
        (id, Side::Decode, d1, 0, decode.session, decode.tokens)
    );
    assert_eq!(
        booked(&store, &[p, d0, d1]),
        [1, 0, 0],
        "no slot was supplied"
    );
}

#[test]
fn the_call_starts_the_dispatch_deadline_again() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, t0) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], t0)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), t0));
    let pending = DecodeSide {
        state: LeaseState::Chosen,
        ..decode_side(d)
    };
    let now = t0 + Duration::from_secs(8);
    assert_eq!(leases.issue_decode(gateway, id, pending, now), Ok(()));
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: t0 }, LeaseState::ISSUED))
    );
    assert!(leases.expire(t0 + Duration::from_secs(9)).is_empty());
    assert_eq!(store.queued(), 1);
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    let swept = leases.expire(t0 + Duration::from_secs(14));
    let filed = leases.sides(gateway, id).expect("the prefill still stands");
    assert_eq!(swept, [filed]);
    assert_eq!(filed.record.state, LeaseState::Dispatched { at: t0 });
    assert_eq!(
        filed.decode.map(|side| side.state),
        Some(LeaseState::Released(ReleaseReason::Abandoned))
    );
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
}

#[test]
fn the_call_is_refused_by_name() {
    use DecodeRefused::{DecodeStands, NoLease, NotChosen, PrefillNotHeld};

    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let (mut ended, decode) = chosen_pair(4, p, &[d], now);
    ended.state = LeaseState::Completed;
    issue_pair(leases, (ended, decode)).expect("the key is free");
    leases
        .issue(booking(gateway, 1, p, now))
        .expect("the key is free");
    issue_pair(leases, pair(2, p, &[d], now)).expect("the key is free");
    for id in [3, 5, 6] {
        issue_pair(leases, chosen_pair(id, p, &[d], now)).expect("the key is free");
    }
    for id in [2, 5, 6] {
        assert!(leases.dispatched(gateway, LeaseId(id), on(Side::Prefill, 0), now));
    }
    let decode = decode_side(d);
    assert_eq!(
        leases.issue_decode(gateway, LeaseId(5), decode, now),
        Ok(())
    );
    assert!(leases.dispatched(gateway, LeaseId(6), on(Side::Decode, 0), now));
    let refused = |owner, id, reason| {
        let before = (
            leases.sides(gateway, id),
            booked(&store, &[p, d]),
            store.queued(),
        );
        let later = now + Duration::from_secs(8);
        assert_eq!(
            leases.issue_decode(owner, id, decode, later),
            Err(reason),
            "{owner:?} {id:?}"
        );
        assert_eq!(
            (
                leases.sides(gateway, id),
                booked(&store, &[p, d]),
                store.queued()
            ),
            before
        );
    };
    refused(GatewayId(2), LeaseId(5), NoLease);
    refused(gateway, LeaseId(1), NotChosen);
    refused(gateway, LeaseId(2), NotChosen);
    refused(gateway, LeaseId(3), PrefillNotHeld);
    refused(gateway, LeaseId(4), PrefillNotHeld);
    refused(gateway, LeaseId(5), DecodeStands);
    refused(gateway, LeaseId(6), DecodeStands);
    for id in [2, 5, 6] {
        assert!(leases.completed(gateway, LeaseId(id), &done(Side::Decode)));
    }
    refused(gateway, LeaseId(2), NotChosen);
    refused(gateway, LeaseId(5), DecodeStands);
    refused(gateway, LeaseId(6), DecodeStands);
    assert!(leases.completed(gateway, LeaseId(3), &done(Side::Prefill)));
    refused(gateway, LeaseId(3), NoLease);
}

#[test]
fn a_dispatch_on_a_chosen_side_books_it_once_as_it_lands() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for (index, target, counts) in [(1, d1, [1, 0, 1]), (0, d0, [1, 1, 0])] {
        let id = LeaseId(u64::from(index) + 1);
        let (record, decode) = chosen_pair(id.0, p, &[d0, d1], now);
        issue_pair(leases, (record, decode)).expect("the key is free");
        assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
        assert!(leases.expire(now + DISPATCH_TTL).is_empty());
        assert_eq!(store.queued(), 0);
        let at = now + Duration::from_secs(8);
        assert!(leases.dispatched(gateway, id, on(Side::Decode, index), at));
        let filed = leases.sides(gateway, id).expect("on file");
        assert_eq!(
            filed.decode,
            Some(DecodeSide {
                target,
                target_index: index,
                state: LeaseState::Dispatched { at },
                ..decode
            })
        );
        assert_eq!(booked(&store, &[p, d0, d1]), counts);
        assert_eq!(store.queued(), 0);
        // A second word, on the pick's other target, moves nothing.
        assert!(leases.dispatched(gateway, id, on(Side::Decode, 1 - index), now));
        assert_eq!(leases.sides(gateway, id), Some(filed));
        assert_eq!(booked(&store, &[p, d0, d1]), counts);
        assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));
        assert_eq!(booked(&store, &[p, d0, d1]), [0, 0, 0]);
    }
}

/// A dispatch at an index the pick holds no target at names no worker. A
/// chosen side takes nothing of it: the side stays chosen on its pick's
/// first decode and books none, and the door still answers that the lease
/// is on file. Behind the word a call issues the side, a dispatch that
/// names a target books that target, and the side's release frees no
/// count. The pick is as wide as a record keeps one, four targets, so the
/// first index past it is the first that no record holds.
#[test]
fn a_dispatch_at_an_index_its_pick_holds_no_target_at_leaves_a_chosen_side_as_it_stood() {
    let store = MemoryStore::new();
    let keys = ["p", "d0", "d1", "d2", "d3"].map(|name| format!("http://{name}:1"));
    let workers = keys.map(|key| ready(&store, &key));
    let [p, d0, d1, d2, d3] = workers;
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let at = now + Duration::from_secs(8);
    // A pair of four targets with its prefill sent, and the word said past
    // them: at the first index the pick does not hold, and at a byte's
    // last.
    let past_its_pick = |id: u64| {
        let id = LeaseId(id);
        let (record, decode) = chosen_pair(id.0, p, &[d0, d1, d2, d3], now);
        issue_pair(leases, (record, decode)).expect("the key is free");
        assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
        let filed = leases.sides(gateway, id);
        for index in [4, u8::MAX] {
            assert!(leases.dispatched(gateway, id, on(Side::Decode, index), at));
        }
        assert_eq!(leases.sides(gateway, id), filed);
        assert_eq!(filed.and_then(|sides| sides.decode), Some(decode));
        assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 0]);
        id
    };

    // A call behind the word issues the side, on its own pick's first
    // decode.
    let id = past_its_pick(1);
    let (_, called) = pair(id.0, p, &[d1, d0], now);
    assert_eq!(leases.issue_decode(gateway, id, called, now), Ok(()));
    assert_eq!(booked(&store, &workers), [1, 0, 1, 0, 0]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A dispatch behind it that names a target books that target: here
    // the pick's last.
    let id = past_its_pick(2);
    assert!(leases.dispatched(gateway, id, on(Side::Decode, 3), at));
    let sent = (
        LeaseState::Dispatched { at: now },
        LeaseState::Dispatched { at },
    );
    assert_eq!(sides(&store, id.0), Some(sent));
    assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 1]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A release of the side behind it ends the side and frees no count.
    let id = past_its_pick(3);
    assert!(leases.released(gateway, id, Some(Side::Decode), ReleaseReason::ClientGone));
    let gone = LeaseState::Released(ReleaseReason::ClientGone);
    let ended = (LeaseState::Dispatched { at: now }, gone);
    assert_eq!(sides(&store, id.0), Some(ended));
    assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 0]);
    assert!(leases.completed(gateway, id, &done(Side::Prefill)));
    assert_eq!(booked(&store, &workers), [0, 0, 0, 0, 0]);
}

#[test]
fn a_chosen_side_ends_with_its_prefill() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    // The gateway gives the chosen side up: its end stands and no call
    // issues it.
    assert!(leases.released(gateway, id, Some(Side::Decode), ReleaseReason::ClientGone));
    let gone = LeaseState::Released(ReleaseReason::ClientGone);
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: now }, gone))
    );
    let decode = chosen_pair(1, p, &[d], now).1;
    assert_eq!(
        leases.issue_decode(gateway, id, decode, now),
        Err(DecodeRefused::DecodeStands)
    );
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    assert!(leases.completed(gateway, id, &done(Side::Prefill)));
    assert_eq!(leases.sides(gateway, id), None);
    assert_eq!(store.paired(), [vec![], vec![]]);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
}

#[test]
fn a_chosen_side_dispatched_by_a_handed_frame_is_told_once() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (id, now) = (LeaseId(1), Instant::now());
    leases.set_term(1);
    store.tell_through(heard);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    drop(told_sides());
    let hand = Hand::new(GatewayId(1));
    let sent = |side| Said::Dispatched(on(side, 0), now);
    store.hand(&hand, id, sent(Side::Prefill));
    store.hand(&hand, id, sent(Side::Decode));
    store.hand(&hand, id, sent(Side::Decode));
    // Each side's dispatch is told once, with the lease as its frame found
    // it: the decode side stood chosen.
    let told: Vec<_> = told_sides()
        .into_iter()
        .map(|(found, said)| (found.decode.map(|side| side.state), said))
        .collect();
    let chosen = Some(LeaseState::Chosen);
    assert_eq!(
        told,
        [(chosen, sent(Side::Prefill)), (chosen, sent(Side::Decode))]
    );
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
}

#[test]
fn a_release_of_every_side_frees_a_chosen_side_s_place() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));
    assert_eq!(leases.sides(gateway, id), None);
    assert_eq!(store.paired(), [vec![], vec![]]);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
}

#[test]
fn the_sweep_s_answer_tells_a_decode_side_that_completed_from_one_it_abandoned() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let (p, d) = (WorkerId(7), WorkerId(8));
    leases.set_term(1);
    store.tell_through(heard);
    drop(told_sides());
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), now));
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Decode)));
    let expired = leases.expire(now + Duration::from_secs(2));
    let states = expired.iter().map(|sides| {
        (
            sides.record.lease.id,
            sides.record.state,
            sides.decode.map(|side| side.state),
        )
    });
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    assert_eq!(
        states.collect::<Vec<_>>(),
        [
            (LeaseId(1), abandoned, Some(LeaseState::Completed)),
            (LeaseId(2), abandoned, Some(abandoned)),
        ]
    );
    assert_eq!(leases.live(gateway), 0);
    let told = |id, side| {
        (
            LeaseId(id),
            Said::Released(Some(side), ReleaseReason::Abandoned),
        )
    };
    let ends = told_sides()
        .into_iter()
        .map(|(sides, said)| (sides.record.lease.id, said));
    assert_eq!(
        ends.collect::<Vec<_>>(),
        [
            told(1, Side::Prefill),
            told(2, Side::Prefill),
            told(2, Side::Decode)
        ]
    );
}

#[cfg(feature = "controller")]
#[test]
fn a_pair_s_decode_side_is_told_issued_on_the_decode_worker() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    store.set_term(1);
    let (mut record, mut decode) = pair(0, p, &[d], Instant::now());
    record.lease.booked = p;
    decode.session = Session::from_raw(41);
    decode.tokens = 24;
    let world = store.snapshot();
    let slots = [world.booked_of(p), world.booked_of(d)];
    let heard = RefCell::new(Vec::new());
    let prefill = store
        .issue_decided_pair(record, decode, slots, |lease| {
            heard.borrow_mut().push(*lease);
        })
        .expect("minted under the store's term");
    let decode = Lease {
        side: Side::Decode,
        booked: d,
        session: decode.session,
        tokens: decode.tokens,
        ..prefill
    };
    assert_eq!(heard.into_inner(), [prefill, decode]);
}

#[test]
fn a_pair_with_both_sides_waiting_takes_one_place_of_the_queue() {
    let store = MemoryStore::new();
    store.set_term(1);
    let (p, d) = (WorkerId(7), WorkerId(8));
    issue_pair(&store, pair(1, p, &[d], Instant::now())).expect("the key is free");
    assert_eq!(store.queued(), 1);
}

/// A pair filed with a side that ended books the side that lives, and
/// no count for the one that ended.
#[test]
fn a_pair_filed_with_an_ended_side_books_the_side_that_lives() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let ended = LeaseState::Completed;
    let (record, side) = pair(1, p, &[d], now);
    let half = DecodeSide {
        state: ended,
        ..side
    };
    leases.issue_pair(record, half).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    let (mut half, side) = pair(2, p, &[d], now);
    half.state = ended;
    leases.issue_pair(half, side).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert_eq!(leases.live(gateway), 2);
}

/// The store tells of a lease with the worker its record books. A single's
/// lease and a pair's prefill side name the record's target, a decode side
/// its own, and the lease on file names none.
#[test]
fn a_side_s_lease_names_the_worker_its_record_books() {
    let now = Instant::now();
    let (p, d) = (WorkerId(3), WorkerId(5));
    let booked = |lease: Option<Lease>| lease.map(|lease| lease.booked);
    let told = |sides: &Sides, side| booked(sides.side(side).map(|(lease, _)| lease));
    let record = lease(GatewayId(1), 1, 1, now);
    let single = Sides {
        record,
        decode: None,
    };
    assert_eq!(record.lease.booked, WorkerId(0), "filed with no word");
    assert_eq!(told(&single, Side::Single), Some(record.target));
    assert_eq!(single.ending(None).map(booked), [Some(record.target), None]);
    // A single's end is told off its record, with no list built.
    let mut ended = Vec::new();
    single.each_ending(None, |lease| ended.push(lease.booked));
    assert_eq!(ended, [record.target]);
    let (record, decode) = pair(2, p, &[d], now);
    let both = Sides {
        record,
        decode: Some(decode),
    };
    assert_eq!(told(&both, Side::Prefill), Some(p));
    assert_eq!(told(&both, Side::Decode), Some(d));
    assert_eq!(both.ending(None).map(booked), [Some(p), Some(d)]);
}

#[test]
fn a_pairs_sides_are_dispatched_and_ended_each_by_its_own_frame() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, key, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d], now)).expect("the key is free");
    // One record, and each side books its own worker.
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert_eq!(leases.live(gateway), 1);
    // A side is dispatched by the frame that names it, and no other is.
    let at = now + Duration::from_millis(3);
    let dispatched = LeaseState::Dispatched { at };
    assert!(leases.dispatched(gateway, key, on(Side::Decode, 0), at));
    assert_eq!(sides(&store, 1), Some((LeaseState::ISSUED, dispatched)));
    assert!(leases.dispatched(gateway, key, on(Side::Prefill, 0), at));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    // A frame on a side the pair does not hold changes nothing.
    assert!(leases.dispatched(gateway, key, on(Side::Single, 0), now));
    assert!(leases.completed(gateway, key, &done(Side::Encode)));
    let gone = ReleaseReason::ClientGone;
    assert!(leases.released(gateway, key, Some(Side::Single), gone));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    // The prefill's completion frees its worker's count, and the decode's
    // stands; a second completion of that side changes nothing.
    for _ in 0..2 {
        assert!(leases.completed(gateway, key, &done(Side::Prefill)));
        assert_eq!(booked(&store, &[p, d]), [0, 1]);
        assert_eq!(sides(&store, 1), Some((LeaseState::Completed, dispatched)));
        assert_eq!(leases.live(gateway), 1);
    }
    // The record leaves the file at its last side's end.
    assert!(leases.completed(gateway, key, &done(Side::Decode)));
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    assert_eq!(leases.get(gateway, key), None);
    assert!(!leases.completed(gateway, key, &done(Side::Decode)));
}

#[test]
fn a_decodes_dispatch_on_an_alternate_moves_the_decodes_booking_alone() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, key, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d0, d1], now)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 1, 0]);
    // Every target of a pair's pick names the one prefill: its dispatch on
    // the alternate's index moves no booking.
    assert!(leases.dispatched(gateway, key, on(Side::Prefill, 1), now));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 1, 0]);
    assert!(leases.dispatched(gateway, key, on(Side::Decode, 1), now));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 1]);
    let filed = leases.sides(gateway, key).expect("on file");
    let record = filed.record;
    assert_eq!((record.target, record.lease.target_index), (p, 0));
    let decode = filed.decode.expect("a pair's lease");
    assert_eq!((decode.target, decode.target_index), (d1, 1));
    assert_eq!(filed.target_of(Side::Decode, 1), Some((d1, 0)));
    // The decode's end frees the count of the worker it moved to.
    assert!(leases.completed(gateway, key, &done(Side::Decode)));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 0]);
}

#[test]
fn a_release_ends_the_side_it_names_and_with_no_side_every_side() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now, gone) = (GatewayId(1), Instant::now(), ReleaseReason::ClientGone);
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d], now)).expect("the key is free");
    // One side given up: its count is free and the other side stands.
    assert!(leases.released(gateway, LeaseId(1), Some(Side::Decode), gone));
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    let released = LeaseState::Released(gone);
    assert_eq!(sides(&store, 1), Some((LeaseState::ISSUED, released)));
    // The last side's release takes the record off the file.
    assert!(leases.released(gateway, LeaseId(1), Some(Side::Prefill), gone));
    assert_eq!(leases.get(gateway, LeaseId(1)), None);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    // A release that names no side ends every side at once.
    issue_pair(leases, pair(2, p, &[d], now)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert!(leases.released(gateway, LeaseId(2), None, gone));
    assert_eq!(leases.get(gateway, LeaseId(2)), None);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    assert_eq!(leases.live(gateway), 0);
}

/// A pair filed with its prefill part dispatched and its decode side
/// waiting is due at its deadline all the same: the side that waits ends.
#[test]
fn a_pair_filed_with_its_decode_side_alone_waiting_is_due_at_its_deadline() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let dispatched = LeaseState::Dispatched { at: now };
    let (mut record, decode) = pair(1, p, &[d], now);
    record.state = dispatched;
    issue_pair(leases, (record, decode)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    // An issue past the deadline ends the side that waits.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    assert_eq!(sides(&store, 1), Some((dispatched, abandoned)));
    assert_eq!(booked(&store, &[p, d]), [2, 0]);
}

/// A place let go is the next one laid: a file's list of places is as
/// long as the most pairs that stood at once.
#[test]
fn the_list_of_places_is_as_long_as_the_most_pairs_that_stood_at_once() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=3 {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert_eq!(store.places(), 3);
    // Each pair that ends leaves its place to the next one filed.
    for id in 4..=9 {
        assert!(leases.released(gateway, LeaseId(id - 3), None, ReleaseReason::ClientGone));
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
        assert_eq!(store.places(), 3, "pair {id}");
    }
}

#[test]
fn the_dispatch_deadline_ends_each_side_that_waits_and_a_dispatched_side_stands() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // Two pairs: the first's prefill is dispatched, the second waits whole.
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Prefill, 0), now));
    assert_eq!(booked(&store, &[p, d]), [2, 2]);
    // An issue past the deadline ends what waits and tells of each side
    // it ended, the prefill's first.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store.tell_through(heard);
    drop(told_sides());
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    let ended = told_sides().into_iter().map(|(found, said)| {
        let Said::Released(side, ReleaseReason::Abandoned) = said else {
            panic!("this issue tells the deadline's ends alone: {said:?}");
        };
        // The word reads the side's lease off the lease as the end found
        // it.
        let lease = found.ending(side)[0];
        (found.record.lease.id.0, lease.map(|lease| lease.side))
    });
    assert_eq!(
        ended.collect::<Vec<_>>(),
        [
            (1, Some(Side::Decode)),
            (2, Some(Side::Prefill)),
            (2, Some(Side::Decode)),
        ]
    );
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    let dispatched = LeaseState::Dispatched { at: now };
    // The dispatched side stands with its count; the pair that waited
    // whole left the file.
    assert_eq!(sides(&store, 1), Some((dispatched, abandoned)));
    assert_eq!(leases.get(gateway, LeaseId(2)), None);
    assert_eq!(booked(&store, &[p, d]), [2, 0]);
    // The side that stands ends by its own frame.
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Prefill)));
    assert_eq!(leases.get(gateway, LeaseId(1)), None);
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    // The store's own sweep returns both sides as it left them.
    issue_pair(leases, pair(3, p, &[d], later)).expect("the key is free");
    assert!(leases.dispatched(gateway, LeaseId(3), on(Side::Decode, 0), later));
    let swept = leases.expire(later + Duration::from_secs(9));
    let third =
        |found: &Sides| (found.record.lease.id, found.record.state) == (LeaseId(3), abandoned);
    assert!(swept.iter().any(third));
    let at = LeaseState::Dispatched { at: later };
    assert_eq!(sides(&store, 3), Some((abandoned, at)));
}

/// A pair's sides are dispatched one by one. A decode side that waits
/// behind its dispatched prefill ends at the dispatch deadline, at a later
/// issue and at the store's own sweep, and the dispatched prefill stands.
#[test]
fn the_dispatch_deadline_ends_a_decode_side_that_waits_behind_its_dispatched_prefill() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // The first pair's sides are dispatched one by one, the second's
    // prefill alone.
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    let dispatched = LeaseState::Dispatched { at: now };
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Prefill, 0), now));
    assert_eq!(sides(&store, 1), Some((dispatched, LeaseState::ISSUED)));
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), now));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Prefill, 0), now));
    // An issue past the deadline ends the decode side that waits, and
    // the dispatched sides stand.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert_eq!(sides(&store, 2), Some((dispatched, abandoned)));
    // The store's own sweep returns the prefill as it stood and the
    // decode side as the sweep left it.
    issue_pair(leases, pair(3, p, &[d], later)).expect("the key is free");
    assert!(leases.dispatched(gateway, LeaseId(3), on(Side::Prefill, 0), later));
    let swept = leases.expire(later + Duration::from_secs(9));
    let at = LeaseState::Dispatched { at: later };
    let third = |found: &Sides| {
        (found.record.lease.id, found.record.state) == (LeaseId(3), at)
            && found.decode.is_some_and(|decode| decode.state == abandoned)
    };
    assert!(swept.iter().any(third));
    assert_eq!(sides(&store, 3), Some((at, abandoned)));
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

    // No note behind the write: the veto is written as it is said, here
    // with no hold.
    let written = store.gateway_veto_noted(g, w, Some(first), None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    // The note's end stands in the store: the later end is written.
    let written = store.gateway_veto_noted(g, w, Some(second), None, true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, first), "the later end is shown");

    // The row leaves and comes back under the same id, and its vetoes left
    // with it. A write that says its note stands writes nothing: the row
    // registered again takes no end of the row before it.
    let revision = WorkersStore::get(&store, w).expect("the row").revision;
    WorkersStore::remove(&store, w, revision).expect("removed");
    assert_eq!(ready(&store, "http://a:8000"), w, "the same id");
    for until in [Some(second), None] {
        let written = store.gateway_veto_noted(g, w, until, None, true, now);
        assert_eq!(written, VetoWrite::Left);
    }
    assert!(!store.vetoed(g, w, now));
    assert_eq!(store.health_entries(), 0);
    // With no note behind it the same end is the gateway's word of now,
    // and a clear behind a note that stands lifts it.
    let written = store.gateway_veto_noted(g, w, Some(second), None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    let written = store.gateway_veto_noted(g, w, None, None, true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(!store.vetoed(g, w, now));

    // A worker without a row: a veto writes nothing, a clear is taken.
    let none = WorkerId(7);
    let written = store.gateway_veto_noted(g, none, Some(first), None, false, now);
    assert_eq!(written, VetoWrite::NoRow);
    let written = store.gateway_veto_noted(g, none, None, None, false, now);
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

/// One worker's requests in flight as its host counts them: a number the
/// store steps ([`HostCount`]).
struct Tally(Arc<AtomicU32>);

impl HostCount for Tally {
    fn booked(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn freed(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The count a host keeps of the row `id` from here on: handed to the
/// store, which steps it with each lease that books the row and each that
/// books it no longer ([`WorkersStore::host_count`]). A count handed for a
/// row takes a step for each lease that books the row by then. `None` for
/// a row that left: the store takes no count for it.
fn handed_count(store: &MemoryStore, id: WorkerId) -> Option<Arc<AtomicU32>> {
    let count = Arc::new(AtomicU32::new(0));
    let handed = store.host_count(id, Box::new(Tally(Arc::clone(&count))));
    handed.then_some(count)
}

/// The count a host keeps of the row `id`, which stands in the store.
fn standing(store: &MemoryStore, id: WorkerId) -> Arc<AtomicU32> {
    handed_count(store, id).expect("the row stands in the store")
}

/// What the leases on file book on the row `id`, as a host that asks now
/// counts it ([`handed_count`]).
fn booked_on(store: &MemoryStore, id: WorkerId) -> Option<u32> {
    Some(handed_count(store, id)?.load(Ordering::Relaxed))
}

/// Each member's requests in flight, by worker id: what the leases on
/// file book on each row that serves the cases' model ([`booked_on`]).
pub(super) fn in_flight(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
    let (_, mut rows) = store.all_rows();
    rows.retain(|row| row.facts.models.iter().any(|model| model == "m"));
    rows.sort_by_key(|row| row.id);
    let counted = rows
        .iter()
        .filter_map(|row| Some((row.id, booked_on(store, row.id)?)));
    counted.collect()
}

/// A pair's decode side on `target`: issued, with no alternate behind it.
fn decode_side(target: WorkerId) -> DecodeSide {
    DecodeSide {
        target,
        picked: PickRecord::default(),
        session: Session::NONE,
        state: LeaseState::ISSUED,
        tokens: 0,
        target_index: 0,
    }
}

/// A lease of `gateway` on `target`, issued at `now` under term 1.
pub(super) fn booking(gateway: GatewayId, id: u64, target: WorkerId, now: Instant) -> LeaseRecord {
    LeaseRecord {
        target,
        ..lease(gateway, id, 1, now)
    }
}

#[test]
fn a_rows_count_reads_the_leases_on_file_that_book_its_worker() {
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
    assert!(store.restore_from(leader.checkpoint()));
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
            assert_eq!(record.state, LeaseState::ISSUED);
            record.lease.id.0
        })
        .collect();
    ended.sort_unstable();
    assert_eq!(ended, [1, 2, 3]);
    // The file is empty: a second restore tells no end.
    assert!(store.restore_from(leader.checkpoint()));
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
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    let (now, world) = (Instant::now(), store.snapshot());
    let issue = || {
        let record = booking(GatewayId(1), 0, a, now);
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let named = {
        let _acting = DecisionSlot::new(3).enter();
        issue()
    };
    let unnamed = issue();
    // Another replica led over the same row, and this one restores from
    // its cut on a thread that names none.
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    assert!(store.restore_from(leader.checkpoint()));
    let mut ended: Vec<(u64, Option<DecisionSlot>)> = told()
        .into_iter()
        .map(|(record, _)| record.lease.id.0)
        .zip(told_by())
        .collect();
    ended.sort_unstable();
    let first = Some(DecisionSlot::new(0));
    let mut issued = [(named.0, Some(DecisionSlot::new(3))), (unnamed.0, first)];
    issued.sort_unstable();
    assert_eq!(ended, issued);
}

#[test]
fn a_watch_snapshot_keeps_a_members_local_tables() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://a:1");
    assert_eq!(publisher(&store, "http://a:1", 0), id);
    store.publish_now();
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, id, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");
    let slot = store.snapshot().booked_of(id).expect("the member's slot");
    store.gateway_veto(gateway, id, Some(now + Duration::from_secs(60)));
    HealthSink::completed(&store, gateway, id, TransportClass::Ok, now);
    LoadSink::record(&store, id, 0, row(0), now, LoadSource::Stream);
    let local_load = LoadSink::load(&store, id, 0).expect("the local report");
    assert_eq!(
        store.admit(id, 0, batch(1), &one_block(1)),
        BatchOutcome::Applied
    );
    store.probed(id, 2, 0, 0);

    let mut cut = store.checkpoint();
    cut.epoch += 1;
    cut.at.term = 7;
    cut.rows[0].admitted_at += Duration::from_secs(1);
    cut.rows[0].revision += 1;
    cut.rows[0].facts.priority = 8;
    let mut expected = cut.rows[0].clone();
    expected.admitted_at = WorkersStore::get(&store, id).expect("the row").admitted_at;
    let revision = expected.revision;
    assert!(store.watch_from(cut));

    assert_eq!(leases.live(gateway), 1);
    assert_eq!(leases.get(gateway, lease.lease.id), Some(lease));
    assert_eq!(leases.term(), 1);
    assert_eq!(store.snapshot().booked_of(id), Some(slot));
    assert_eq!(in_flight(&store), [(id, 1)]);
    assert!(store.vetoed(gateway, id, now));
    assert!(store.completed_ok_since(gateway, id, now));
    assert_eq!(LoadSink::load(&store, id, 0), Some(local_load));
    assert_eq!(store.kv().held_blocks("m"), 1);
    assert_eq!(WorkersStore::get(&store, id), Some(expected));
    assert_eq!(
        store.probe_outcome(id, false, revision, &|| true),
        HealthWrite::Moved {
            from: WorkerStatus::Ready,
            to: WorkerStatus::NotReady
        }
    );
}

#[test]
fn a_watch_snapshot_evicts_a_leavers_tables_and_its_lease_moves_no_other_count() {
    let store = MemoryStore::new();
    let left = publisher(&store, "http://a:1", 0);
    store.publish_now();
    let before = store.snapshot();
    let slot = before.booked_of(left).expect("the leaver's slot");
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, left, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");
    store.gateway_veto(gateway, left, Some(now + Duration::from_secs(60)));
    HealthSink::completed(&store, gateway, left, TransportClass::Ok, now);
    LoadSink::record(&store, left, 0, row(0), now, LoadSource::Stream);
    assert_eq!(
        store.admit(left, 0, batch(1), &one_block(1)),
        BatchOutcome::Applied
    );

    let source = MemoryStore::new();
    let joined = ready(&source, "http://b:1");
    LoadSink::record(&source, joined, 0, row(0), now, LoadSource::Stream);
    assert!(store.watch_from(source.checkpoint()));
    assert_eq!(leases.get(gateway, lease.lease.id), Some(lease));
    assert_ne!(store.snapshot().booked_of(joined), Some(slot));
    assert_eq!(in_flight(&store), [(joined, 0)]);
    assert!(WorkersStore::by_key(&store, "http://a:1").is_none());
    assert!(!store.vetoed(gateway, left, now));
    assert!(!store.completed_ok_since(gateway, left, now));
    assert!(LoadSink::load(&store, left, 0).is_none());
    assert!(
        LoadSink::load(&store, joined, 0).is_none(),
        "the cut's load is ignored"
    );
    assert_eq!(store.kv().held_blocks("m"), 0);
    assert!(leases.released(gateway, lease.lease.id, None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), [(joined, 0)]);
    LoadSink::record(&store, joined, 0, row(0), now, LoadSource::Stream);
    assert!(LoadSink::load(&store, joined, 0).is_some());
}

#[test]
fn a_watch_snapshot_gives_a_new_incarnation_its_own_count() {
    let store = MemoryStore::new();
    let key = "http://a:1";
    let old = ready(&store, key);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, old, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");

    let source = MemoryStore::new();
    assert_eq!(ready(&source, key), old);
    let new = source.upsert(facts(key, 2)).id();
    assert!(store.watch_from(source.checkpoint()));
    assert_eq!(store.by_key(key).map(|row| row.id), Some(new));
    assert!(WorkersStore::get(&store, old).is_none());
    assert_eq!(in_flight(&store), [(new, 0)]);
    assert!(leases.released(gateway, lease.lease.id, None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), [(new, 0)]);
}

#[test]
fn a_watch_snapshot_publishes_the_cut_at_once_and_tells_the_members_once() {
    use crate::{
        document::{AppliedSpec, PolicySpec},
        engine::{
            view::{Candidate, RequestFacts},
            Scorer, TieBand,
        },
        stages::{Lifecycle, StageRegistry},
    };

    #[derive(Debug)]
    struct Heard(Arc<Mutex<Vec<String>>>);

    impl Lifecycle for Heard {
        fn on_membership(&self, change: &Membership<'_>) {
            self.0
                .lock()
                .expect("the stage heard")
                .push(format!("{change:?}"));
        }
    }

    impl Scorer for Heard {
        fn name(&self) -> &'static str {
            "heard"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    let store = MemoryStore::new();
    let id = ready(&store, "http://a:1");
    let heard = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let events = Arc::clone(&heard);
    catalog
        .scorer("heard", move || Box::new(Heard(Arc::clone(&events))))
        .expect("a free name");
    let document: PolicySpec = serde_json::from_str(
        r#"{"apiVersion":"smg.policy/v1","generation":1,"view":{"labels":["zone=b"]},"profiles":{"p":{"score":["heard"],"pick":"best"}}}"#,
    ).expect("a policy document");
    let digest = document.digest().expect("the digest");
    let spec =
        AppliedSpec::validate_digested(&document, digest, &catalog, &[]).expect("a generation");
    store.apply(spec).expect("the generation is applied");

    let mut cut = store.checkpoint();
    let earlier = cut.epoch;
    // The same revision can name other facts in another history.
    cut.rows[0]
        .facts
        .labels
        .insert("zone".to_owned(), "b".to_owned());
    cut.rows[0].status = WorkerStatus::Draining;
    cut.epoch += 2;
    let epoch = cut.epoch;
    store.upsert(facts("http://owed:1", 1));
    assert!(store.publish_due().is_some(), "a build is owed");
    heard.lock().expect("the stage heard").clear();
    assert!(store.watch_from(cut));
    assert_eq!(store.snapshot().epoch(), epoch);
    assert_eq!(store.epoch(), epoch);
    assert_eq!(store.history(), 1);
    assert_eq!(
        store.changes_since(earlier),
        MembershipTail::SnapshotRequired
    );
    assert_eq!(store.publish_due(), None);
    assert_eq!(store.filed_members(), 0);
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, id);
    let zone = rows[0].facts.labels.get("zone").map(String::as_str);
    assert_eq!(zone, Some("b"), "the cut's labels");
    assert_eq!(rows[0].status, WorkerStatus::Draining, "the cut's status");
    let snapshot = Membership::Snapshot(&[id]);
    assert_eq!(
        *heard.lock().expect("the stage heard"),
        [format!("{snapshot:?}")]
    );
}

#[test]
fn a_watch_snapshot_the_journal_refuses_changes_nothing() {
    let store = MemoryStore::journaled(Arc::new(Log::new(8)));
    let id = ready(&store, "http://a:1");
    store.set_term(3);
    let before = store.checkpoint();
    let world = store.snapshot();
    let source = MemoryStore::new();
    ready(&source, "http://b:1");
    let mut cut = source.checkpoint();
    cut.at = Position { seq: 1, term: 2 };
    assert!(!store.watch_from(cut));
    let after = store.checkpoint();
    assert_eq!(
        (after.at, after.epoch, after.rows),
        (before.at, before.epoch, before.rows)
    );
    assert_eq!(store.history(), 0);
    assert!(Arc::ptr_eq(&store.snapshot(), &world));
    assert_eq!(in_flight(&store), [(id, 0)]);
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

    fn term_rose(&self, _: u64) {}

    fn generation(&self, _: Option<Position>, _: u64) {}

    fn term_followed(&self, _: Position) {}

    fn head(&self) -> Position {
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
    let cut = leader.checkpoint();
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

/// The counts are one table in pieces that never move: every slot names a
/// cell of its own in whichever piece holds it, a run of slots is read in
/// its order across the pieces, and the slot that names no cell reads zero
/// and books nothing.
#[test]
fn a_count_is_found_by_its_slot_in_whichever_piece_holds_it() {
    let counts = InFlightCounts::default();
    // The first piece holds 1,024 cells and the second 2,048: one slot
    // more stands in the third.
    let slots: Vec<WorkerSlot> = (0..3_073).map(|_| counts.claim()).collect();
    for &slot in &slots {
        counts.book(slot, Session::NONE);
    }
    assert!(counts.of(&slots).all(|count| count == 1), "a cell a slot");
    for &slot in &slots {
        counts.free(slot, Session::NONE);
    }
    // Both ends of each piece, a count of its own on each.
    let marked = [0, 1, 1_023, 1_024, 3_071, 3_072];
    for (times, &at) in (1..).zip(&marked) {
        for _ in 0..times {
            counts.book(slots[at], Session::NONE);
        }
    }
    counts.book(WorkerSlot::NONE, Session::NONE);
    let ends = marked.iter().rev().map(|&at| slots[at]);
    let run: Vec<WorkerSlot> = ends.chain([WorkerSlot::NONE, slots[2]]).collect();
    assert_eq!(
        counts.of(&run).collect::<Vec<_>>(),
        [6, 5, 4, 3, 2, 1, 0, 0]
    );
    // A lease that leaves gives one back, and a count at zero stays there.
    for slot in [slots[1_024], slots[2], WorkerSlot::NONE] {
        counts.free(slot, Session::NONE);
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
/// worker's count and asks for no build: its host's count reads the lease
/// inside the window and behind the window's build, and each lease ends
/// on the count it booked.
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
    let builds = store.membership_builds();
    file(leases, gateway, &records);
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "a lease asked for the build"
    );
    let of = |count| {
        let mut rows = [(built, count), (owed, count)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(1), "inside the window");
    store.publish_now();
    assert_eq!(in_flight(&store), of(1));
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
    assert_eq!(in_flight(&store), of(0));
}

/// The issue door books the slot a member's write gave it, inside the
/// member's registration window too.
#[test]
fn a_lease_issued_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, _, records| {
        for record in records {
            assert_eq!(leases.issue(*record), Ok(()));
        }
    });
}

/// The restore door reads the same slot: a gateway's pages may list a
/// lease on a member inside its registration window.
#[test]
fn a_lease_restored_ahead_of_its_workers_first_build_books_its_count() {
    filed_ahead_of_a_first_build(|leases, gateway, records| {
        assert_eq!(leases.restore(gateway, records), 2);
    });
}

/// No lease asks for a build: one on a worker the standing world lists
/// and one on a worker inside its registration window both leave the owed
/// build to its window.
#[test]
fn a_lease_issued_leaves_an_owed_build_to_its_window() {
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
        (builds, true),
        "a lease on a worker inside its window asked for the build"
    );
    for id in [1, 2] {
        assert!(leases.released(gateway, LeaseId(id), None, ReleaseReason::ClientGone));
    }
}

/// A first dispatch that moves a lease onto a worker inside its
/// registration window books that worker's count and asks for no build:
/// its host's count reads the lease inside the window and behind the
/// window's build, and the lease ends on the count it booked.
#[test]
fn a_lease_dispatched_onto_a_worker_inside_its_window_books_its_count() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    let builds = store.membership_builds();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let mut targets = Targets::new();
    for id in [built, owed] {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let record = LeaseRecord {
        picked: PickRecord::of(&targets),
        ..booking(gateway, 1, built, now)
    };
    assert_eq!(leases.issue(record), Ok(()));
    let on = Dispatch {
        target_index: 1,
        ..Dispatch::default()
    };
    assert!(leases.dispatched(gateway, LeaseId(1), on, now));
    assert_eq!(
        (store.membership_builds(), store.publish_due().is_some()),
        (builds, true),
        "the dispatch asked for the build"
    );
    let of = |count| {
        let mut rows = [(built, 0), (owed, count)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(1), "inside the window");
    store.publish_now();
    assert_eq!(in_flight(&store), of(1));
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0));
}

/// A member that leaves inside its registration window takes its slot
/// with it: the lease filed on it ends on the count it booked, and the
/// member that joins next is given another.
#[test]
fn a_member_that_leaves_inside_its_window_takes_its_slot_with_it() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let left = store.upsert(facts("http://b:1", 1)).id();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, left, now)), Ok(()));
    let revision = WorkersStore::get(&store, left).expect("the row").revision;
    assert!(store.remove(left, revision).is_some());
    let joined = ready(&store, "http://c:1");
    let mut rows = [(built, 0), (joined, 0)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows, "the lease of the one that left");
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::WorkerRemoved));
    assert_eq!(in_flight(&store), rows);
}

/// The table holds a member from the write that files it to the build
/// that lists it: behind a membership build it holds none, however many
/// joined inside the window.
#[test]
fn a_membership_build_leaves_no_member_in_the_table() {
    let store = MemoryStore::new();
    let first = ready(&store, "http://a:1");
    assert_eq!(store.filed_members(), 0, "the first build listed it");
    let second = store.upsert(facts("http://b:1", 1)).id();
    let third = store.upsert(facts("http://c:1", 1)).id();
    assert!(first != second && second != third);
    assert_eq!(store.filed_members(), 2, "two joined inside the window");
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build listed both");
}

/// A generation of the standing pairing mode builds no membership: the
/// table stands behind its publication, and a lease on a member inside
/// its window books its count.
#[cfg(feature = "controller")]
#[test]
fn a_generation_that_builds_no_membership_leaves_a_joiners_slot_on_file() {
    let store = MemoryStore::new();
    store.apply(manual_ruled(1, 60, 30)).expect("generation 1");
    let built = ready(&store, "http://a:1");
    let owed = store.upsert(facts("http://b:1", 1)).id();
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    store.apply(manual_ruled(2, 60, 30)).expect("generation 2");
    assert_eq!(
        (store.publish_due().is_some(), store.filed_members()),
        (true, 1),
        "the generation built no membership"
    );
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A member filed with no model stands in no pool, and the world of the
/// build behind its write names its slot all the same: a lease on it
/// inside the window of the refresh that gives it a model books its
/// count.
#[test]
fn a_member_given_its_first_model_books_a_lease_inside_that_window() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build named it, in no pool");
    // The same incarnation with a model: the row and its slot stay.
    assert_eq!(store.upsert(facts("http://b:1", 1)).id(), owed);
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    assert!(store.publish_due().is_some(), "the refresh owes its build");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A lease on a member that offers no model books its count all the same:
/// its host's count reads the lease while the member stands in no pool,
/// and through the write that gives it its first model.
#[test]
fn a_lease_on_a_member_in_no_pool_books_its_count() {
    let store = MemoryStore::new();
    let built = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    assert_eq!(store.filed_members(), 0, "the build named it, in no pool");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    assert_eq!(in_flight(&store), [(built, 0)], "it serves no model");
    assert_eq!(booked_on(&store, owed), Some(1), "booked, in no pool");
    assert_eq!(store.upsert(facts("http://b:1", 1)).id(), owed);
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    store
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    store.publish_now();
    let mut rows = [(built, 0), (owed, 1)];
    rows.sort_unstable();
    assert_eq!(in_flight(&store), rows);
}

/// A world names the slot of a member in no pool from the build behind
/// its write to the build behind its leave, and of no worker beside the
/// members its build read.
#[test]
fn a_world_names_a_member_in_no_pool_until_it_leaves() {
    let store = MemoryStore::new();
    let pooled = ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    assert!(store.snapshot().booked_of(pooled).is_some());
    assert!(
        store.snapshot().unpooled_of(owed).is_none(),
        "inside its window"
    );
    store.publish_now();
    assert!(
        store.snapshot().unpooled_of(owed).is_some(),
        "named, in no pool"
    );
    assert!(
        store.snapshot().booked_of(owed).is_none(),
        "no pool lists it"
    );
    assert!(
        store.snapshot().unpooled_of(pooled).is_none(),
        "a pool lists it"
    );
    let revision = WorkersStore::get(&store, owed).expect("the row").revision;
    assert!(store.remove(owed, revision).is_some());
    assert!(store.snapshot().unpooled_of(owed).is_none(), "it left");
    assert!(store.snapshot().booked_of(pooled).is_some());
}

/// A generation of the standing pairing mode builds no membership: the
/// world it publishes still names the slot of a member in no pool.
#[cfg(feature = "controller")]
#[test]
fn a_generation_that_builds_no_membership_still_names_a_member_in_no_pool() {
    let store = MemoryStore::new();
    store.apply(manual_ruled(1, 60, 30)).expect("generation 1");
    ready(&store, "http://a:1");
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = store.upsert(bare).id();
    store.publish_now();
    assert!(store.snapshot().unpooled_of(owed).is_some());
    store.apply(manual_ruled(2, 60, 30)).expect("generation 2");
    assert!(
        store.snapshot().unpooled_of(owed).is_some(),
        "the generation built no membership"
    );
}

/// A standby follows the refresh as its leader wrote it: its restored
/// world names the slot of the member given its first model, the
/// followed refresh files nothing, and a lease on the member before the
/// standby's build books its count.
#[test]
fn a_lease_books_on_a_standby_behind_a_followed_refresh() {
    let leader = MemoryStore::new();
    let bare = WorkerFacts {
        models: Vec::new(),
        ..facts("http://b:1", 1)
    };
    let owed = leader.upsert(bare).id();
    let revision = WorkersStore::get(&leader, owed).expect("the row").revision;
    leader
        .set_status(owed, WorkerStatus::Ready, revision)
        .expect("ready");
    leader.publish_now();
    let standby = MemoryStore::new();
    assert!(standby.restore_from(leader.checkpoint()));
    assert_eq!(standby.filed_members(), 0, "the restore built its world");
    // The leader gives the member its first model and the standby follows.
    let unjournaled = Position::default();
    let refresh = leader.upsert(facts("http://b:1", 1));
    let row = WorkersStore::get(&leader, owed);
    standby.follow(unjournaled, WorkersStore::epoch(&leader), refresh, row);
    assert_eq!(
        (standby.publish_due().is_some(), standby.filed_members()),
        (true, 0),
        "the followed refresh owes its build and files nothing"
    );
    let leases: &dyn LeasesStore = &standby;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, owed, now)), Ok(()));
    standby.publish_now();
    assert_eq!(in_flight(&standby), [(owed, 1)]);
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
        picked: PickRecord::of(&targets),
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
        picked: PickRecord::of(&targets),
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
        picked: PickRecord::of(&targets),
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

/// The file's records and its places of decode sides stand and fall
/// together: at every check of a walk over the leases' doors a decode
/// side is on file for each record that carries the mark of a pair's
/// lease and for no other, and both agree with the walk's own plain map
/// of what it filed.
#[test]
fn a_decode_side_is_on_file_for_each_record_that_carries_the_mark_and_for_no_other() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["p", "d0", "d1"].map(|name| ready(&store, &format!("http://{name}:1")));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let gone = ReleaseReason::ClientGone;
    leases.set_term(1);
    // The walk's own word: each id it filed, and whether a pair's lease
    // stands under it; an id that left the file is not a key.
    let mut filed = std::collections::BTreeMap::<u64, bool>::new();
    let mut seen = Vec::new();
    let mut read = |door: &str, filed: &std::collections::BTreeMap<u64, bool>| {
        let [sides, marks] = store.paired();
        assert_eq!(sides, marks, "{door}: a decode side for each mark");
        let pairs = filed.iter().filter(|(_, pair)| **pair);
        let pairs: Vec<_> = pairs.map(|(id, _)| (gateway, *id)).collect();
        assert_eq!(sides, pairs, "{door}: the pairs on file");
        seen.extend(filed.keys().copied());
        for id in &seen {
            let found = leases.sides(gateway, LeaseId(*id));
            let pair = found.map(|found| found.decode.is_some());
            assert_eq!(pair, filed.get(id).copied(), "{door}: lease {id}");
        }
    };
    // Filed by each door that files: a single's, a pair's, and a record
    // alone that carries a mark, which is a single's.
    leases.issue(booking(gateway, 1, p, now)).expect("free");
    issue_pair(leases, pair(2, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(3, p, &[d0, d1], now)).expect("free");
    let alone = LeaseRecord {
        decode: Some(DecodeMark),
        ..booking(gateway, 4, p, now)
    };
    leases.issue(alone).expect("free");
    filed.extend([(1, false), (2, true), (3, true), (4, false)]);
    read("an issue", &filed);
    assert_eq!(
        leases.get(gateway, LeaseId(4)).expect("on file").decode,
        None
    );
    assert_eq!(
        leases.get(gateway, LeaseId(2)).expect("on file").decode,
        Some(DecodeMark)
    );
    // A side's dispatch and a side's end leave both tables as they stand;
    // the last side's end takes both.
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Prefill, 0), now));
    assert!(leases.dispatched(gateway, LeaseId(3), on(Side::Decode, 1), now));
    read("a side's dispatch", &filed);
    assert!(leases.completed(gateway, LeaseId(2), &done(Side::Prefill)));
    read("a side's completion", &filed);
    assert!(leases.completed(gateway, LeaseId(2), &done(Side::Decode)));
    filed.remove(&2);
    read("the last side's completion", &filed);
    assert!(leases.released(gateway, LeaseId(3), Some(Side::Decode), gone));
    read("a side's release", &filed);
    assert!(leases.released(gateway, LeaseId(3), Some(Side::Prefill), gone));
    filed.remove(&3);
    read("the last side's release", &filed);
    // A decision's issue of a single, under an id the store mints, beside
    // a pair; a release of the pair whole.
    let slot = store.snapshot().booked_of(p);
    let single = store.issue_decided(booking(gateway, 0, p, now), slot, |_| ());
    let single = single.expect("minted under the store's term").id;
    issue_pair(leases, pair(10, p, &[d0], now)).expect("free");
    filed.extend([(single.0, false), (10, true)]);
    read("a decision's issue", &filed);
    assert!(leases.released(gateway, LeaseId(10), None, gone));
    filed.remove(&10);
    read("a pair's release whole", &filed);
    // The dispatch deadline: a pair that waits whole leaves, and one with
    // a dispatched side stands with its decode side ended.
    issue_pair(leases, pair(5, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(6, p, &[d0], now)).expect("free");
    assert!(leases.dispatched(gateway, LeaseId(6), on(Side::Prefill, 0), now));
    filed.extend([(5, true), (6, true)]);
    read("two pairs more", &filed);
    leases.expire(now + Duration::from_secs(2));
    filed.retain(|id, _| *id == 6);
    read("the dispatch deadline", &filed);
    // A frame applied through a hand, by whoever holds the turn next.
    let hand = Hand::new(gateway);
    store.hand(&hand, LeaseId(6), Said::Completed(done(Side::Prefill)));
    store.hand_back(hand);
    filed.remove(&6);
    read("a handed frame", &filed);
    // A restore over a key on file: its pages list records alone, so the
    // pair under that key leaves whole and a single's lease stands there.
    issue_pair(leases, pair(7, p, &[d0], now)).expect("free");
    issue_pair(leases, pair(8, p, &[d0], now)).expect("free");
    filed.extend([(7, true), (8, true)]);
    read("two pairs for a restore", &filed);
    assert_eq!(leases.restore(gateway, &[pair(7, p, &[d0], now).0]), 1);
    filed = [(7, false)].into();
    read("a restore", &filed);
    // The gateway's end takes every lease, a pair's two parts together.
    issue_pair(leases, pair(9, p, &[d1], now)).expect("free");
    filed.insert(9, true);
    read("a pair beside a restored lease", &filed);
    assert_eq!(leases.drop_gateways(&[gateway]), [2]);
    filed.clear();
    read("the gateway's end", &filed);
    assert_eq!(booked(&store, &[p, d0, d1]), [0, 0, 0]);
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

thread_local! {
    /// What the store told of leases on this thread, in the order told:
    /// both parts of each lease.
    static TOLD: RefCell<Vec<(Sides, Said)>> = const { RefCell::new(Vec::new()) };
    /// The decider each of those tells named, in the same order.
    static TOLD_BY: RefCell<Vec<Option<DecisionSlot>>> = const { RefCell::new(Vec::new()) };
}

/// A teller that keeps its word for [`told`]: the record a frame found on
/// file and what the frame said, and for [`told_by`] the decider named.
fn heard(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, said: Said, by: Option<DecisionSlot>) {
    TOLD.with(|told| told.borrow_mut().push((*found, said)));
    TOLD_BY.with(|told| told.borrow_mut().push(by));
}

/// The deciders [`heard`] was told since the last call, taken.
fn told_by() -> Vec<Option<DecisionSlot>> {
    TOLD_BY.with(|told| std::mem::take(&mut *told.borrow_mut()))
}

/// What [`heard`] kept on this thread since the last call, taken: each
/// lease's record.
fn told() -> Vec<(LeaseRecord, Said)> {
    let record = |(found, said): (Sides, Said)| (found.record, said);
    told_sides().into_iter().map(record).collect()
}

/// What [`heard`] was told since the last call, both parts of each lease,
/// taken.
fn told_sides() -> Vec<(Sides, Said)> {
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
    let world = store.snapshot();
    store.tell_through(heard);
    let before = told_sides();
    let lease = store
        .issue_decided(record, world.booked_of(target), |_| ())
        .expect("minted under the store's term");
    // What the issue told: the ends of the leases it found past their
    // deadline, taken here, and the frames it applied, kept for the case.
    let abandoned = Said::Released(None, ReleaseReason::Abandoned);
    let (ended, applied): (Vec<_>, Vec<_>) = told_sides()
        .into_iter()
        .partition(|(_, said)| *said == abandoned);
    TOLD.with(|told| told.borrow_mut().extend(before.into_iter().chain(applied)));
    let ended = ended.iter().map(|(found, _)| {
        let record = found.record;
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

thread_local! {
    static LEASE_HASHES: Cell<usize> = const { Cell::new(0) };
}

#[derive(Default)]
pub(super) struct LeaseHashes;

impl std::hash::BuildHasher for LeaseHashes {
    type Hasher = rustc_hash::FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        LEASE_HASHES.set(LEASE_HASHES.get() + 1);
        Self::Hasher::default()
    }
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

/// The pair a decode call reads ahead of its decision is the lease's own:
/// its prefill's worker, then the worker and the rank of the decode its
/// first decision chose for the target the prefill's dispatch took, the
/// pick's first or the one that dispatch named. A side a call issued and a
/// take-back put back names the first decision's decode again. The read is
/// refused under the name its issue is refused under, it gives the file's
/// turn back with its answer, and it moves nothing.
#[cfg(feature = "controller")]
#[test]
fn the_held_pair_names_the_prefill_and_the_decode_of_the_target_its_dispatch_took() {
    use DecodeRefused::{DecodeStands, NoLease, NotChosen, PrefillNotHeld};

    let store = MemoryStore::new();
    let keys = ["http://p:1", "http://d0:1", "http://d1:1", "http://d2:1"];
    let [p, d0, d1, d2] = keys.map(|key| ready(&store, key));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    // A pick of three targets on the one prefill, each decode at a rank of
    // its own: the pair filed with its decode side chosen where `later`.
    let filed = |id: u64, later: bool| {
        let (mut record, mut decode) = pair(id, p, &[d0, d1, d2], now);
        let mut targets = Targets::new();
        for (decode_id, decode_dp_rank) in [(d0, 2), (d1, 3), (d2, 5)] {
            let pair = decision::Pair {
                decode_id,
                decode_dp_rank,
                ..Default::default()
            };
            targets.push(Target {
                id: p,
                pair: Some(pair),
                ..Target::default()
            });
        }
        (record.picked, decode.picked) =
            (PickRecord::of(&targets), PickRecord::of_decodes(&targets));
        if later {
            decode.state = LeaseState::Chosen;
        }
        issue_pair(&store, (record, decode)).expect("the key is free");
    };
    for id in 1..=3 {
        filed(id, true);
    }
    filed(4, false);
    store
        .issue(booking(gateway, 5, p, now))
        .expect("the key is free");
    let file = || {
        let sides = [1, 2, 3, 4, 5].map(|id| LeasesStore::sides(&store, gateway, LeaseId(id)));
        (sides, booked(&store, &[p, d0, d1, d2]), store.queued())
    };
    let held = |owner: GatewayId, id: u64| {
        let before = file();
        let held = store.held_pair(owner, LeaseId(id));
        assert!(store.unlocked(), "the turn is given back");
        assert_eq!(file(), before, "the read moves nothing");
        held
    };
    // No prefill was sent yet.
    assert_eq!(held(gateway, 1), Err(PrefillNotHeld));
    // The prefill's dispatch names the pick's first target, then another.
    assert!(store.dispatched(gateway, LeaseId(1), on(Side::Prefill, 0), now));
    assert!(store.dispatched(gateway, LeaseId(2), on(Side::Prefill, 2), now));
    assert!(store.dispatched(gateway, LeaseId(4), on(Side::Prefill, 0), now));
    assert_eq!(held(gateway, 1), Ok((p, Some((d0, 2)))));
    assert_eq!(held(gateway, 2), Ok((p, Some((d2, 5)))));
    // Each refusal by its name: another gateway's id, a prefill not sent,
    // a pair filed whole and a single's lease.
    assert_eq!(held(GatewayId(2), 1), Err(NoLease));
    assert_eq!(held(gateway, 3), Err(PrefillNotHeld));
    assert_eq!(held(gateway, 4), Err(NotChosen));
    assert_eq!(held(gateway, 5), Err(NotChosen));
    // A call issues the side on another decode: the side stands, and the
    // read is refused as the next issue is.
    let moved = DecodeSide {
        picked: PickRecord::default(),
        ..decode_side(d1)
    };
    assert_eq!(store.issue_decode(gateway, LeaseId(2), moved, now), Ok(()));
    assert_eq!(held(gateway, 2), Err(DecodeStands));
    assert_eq!(
        store.issue_decode(gateway, LeaseId(2), moved, now),
        Err(DecodeStands)
    );
    // Taken back, the side names the first decision's decode again, at the
    // target the prefill's dispatch took.
    assert!(store.take_back_decode(gateway, LeaseId(2), ReleaseReason::Abandoned));
    assert_eq!(held(gateway, 2), Ok((p, Some((d2, 5)))));
}

/// An undo takes back its own issue of a decode side and no other. Where a
/// take-back ended that issue, and where a later call issued the side
/// since, the first issue's undo moves nothing and frees nothing: the place
/// counts the side's issues, and an undo names its own. The undo of the
/// issue that stands puts the side back as that issue found it and frees
/// its booking. Behind the prefill's end it takes the lease off the file:
/// the side would stand chosen beside a prefill that ended, and no later
/// event ends a side that is not live.
#[cfg(feature = "controller")]
#[test]
fn an_undo_takes_back_its_own_issue_of_a_decode_side_and_no_other() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    store.set_term(1);
    issue_pair(&store, chosen_pair(id.0, p, &[d0, d1], now)).expect("the key is free");
    assert!(store.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let file = || {
        let sides = LeasesStore::sides(&store, gateway, id);
        (sides, booked(&store, &[p, d0, d1]))
    };
    let chosen = file();
    assert_eq!(chosen.1, [1, 0, 0]);
    // One call's issue of the side on the second decode, with the slot of
    // the count it books.
    let call = || {
        let slot = store.snapshot().booked_of(d1);
        let issued = store.issue_decided_decode(gateway, id, (decode_side(d1), slot), now, |_| ());
        issued.expect("the prefill holds the chosen side").1
    };
    let first = call();
    assert_eq!(file().1, [1, 0, 1]);
    assert!(store.take_back_decode(gateway, id, ReleaseReason::Abandoned));
    assert_eq!(file(), chosen);
    assert!(!store.unissue_decode(gateway, id, first), "its issue ended");
    assert_eq!(file(), chosen);
    let second = call();
    let issued = file();
    assert_eq!(issued.1, [1, 0, 1]);
    assert!(!store.unissue_decode(gateway, id, first), "another's issue");
    assert_eq!(file(), issued);
    assert!(store.unissue_decode(gateway, id, second));
    assert_eq!(file(), chosen);
    assert!(!store.unissue_decode(gateway, id, second), "undone once");
    // Behind the prefill's end.
    let third = call();
    let gone = ReleaseReason::ClientGone;
    assert!(store.released(gateway, id, Some(Side::Prefill), gone));
    assert!(file().0.is_some(), "the issued decode stands");
    assert!(store.unissue_decode(gateway, id, third));
    assert_eq!(file(), (None, vec![0, 0, 0]));
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
        (ids[0], LeaseState::ISSUED, dispatch)
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

thread_local! {
    /// The epoch of the world each tell of [`under`] was handed.
    static UNDER: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    /// Whether the next tell of [`under`] publishes a later world.
    static LATER: Cell<bool> = const { Cell::new(true) };
}

/// A teller that keeps the epoch of the world each tell is handed. Its
/// first tell publishes a later world before it returns, so a word told
/// behind it under the same load reads the earlier epoch.
fn under(store: &MemoryStore, world: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
    UNDER.with_borrow_mut(|under| under.push(world.epoch()));
    if LATER.replace(false) {
        ready(store, "http://b:1");
    }
}

/// A store under term 1 that tells through [`under`], with one ready
/// worker and `leases` leases issued on it at `now`, five seconds to their
/// dispatch: the store, the worker, the leases' ids and the epoch of the
/// world that stands.
fn under_bed(leases: u64, now: Instant) -> (Arc<MemoryStore>, WorkerId, Vec<LeaseId>, u64) {
    let store = Arc::new(MemoryStore::new());
    store.tell_through(under);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let ids = (0..leases).map(|_| LeaseId(1 << 32 | issued_at(&store, a, now).0));
    let ids = ids.collect();
    let first = store.snapshot().epoch();
    (store, a, ids, first)
}

/// A thread with no job at hand keeps its watch while a lease stands on
/// its file: a hand leaves its frame in the ring, and the thread's next
/// look applies it, or the first step of its next job.
#[test]
fn a_frame_handed_under_a_watch_is_left_to_the_watching_thread() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "three leases stand on the file"
    );
    assert_eq!(store.watches(), 1);
    store.hand(&hand, ids[0], GONE);
    assert_eq!((told_ids(), in_flight(&store)), (vec![], vec![(a, 3)]));
    // The thread's next look applies the frame,
    assert_eq!(place.look(), LookOutcome::Fed);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 2)])
    );
    // and so does the first step of its next job.
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    place.work();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(2, GONE)], vec![(a, 1)])
    );
}

/// A watch starts at a look that finds a lease on file, never at a job.
/// A thread that rests behind each job shows none, as a caller of a door
/// in process: its hands read who is at work, and its rest applies what
/// they left. A thread that looked keeps its watch through the jobs
/// behind the look.
#[test]
fn a_watch_starts_at_a_look_and_stands_through_a_job() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(store.watches(), 0, "at work, and no look yet");
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), [], "left to the thread at work");
    place.rest();
    assert_eq!((told_ids(), store.watches()), (vec![(1, GONE)], 0));
    place.work();
    assert_eq!(store.watches(), 0, "a job starts no watch");
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "two leases stand on the file"
    );
    assert_eq!(store.watches(), 1);
    place.work();
    assert_eq!(store.watches(), 1, "the watch stands through the job");
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    place.apply();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(2, GONE)], vec![(a, 1)])
    );
}

/// A look says whether it applied a frame, for its host to set the time
/// to its next look by: a frame of a lease on file, and one that names no
/// lease on file, since its gateway spoke all the same. A frame the
/// thread applied as it took up a job is behind it at its next look.
#[test]
fn a_look_says_whether_it_applied_a_frame() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet, "nothing was handed");
    store.hand(&hand, ids[0], GONE);
    assert_eq!(place.look(), LookOutcome::Fed);
    assert_eq!(place.look(), LookOutcome::Quiet, "the ring is empty again");
    // A second release of the lease that left: a frame all the same.
    store.hand(&hand, ids[0], GONE);
    assert_eq!(place.look(), LookOutcome::Fed);
    // What the thread applies at a job's first step is not its look's.
    store.hand(&hand, ids[1], GONE);
    place.work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

/// A look that finds no lease on file ends the watch: the thread owes the
/// file no further look, and a hand applies in place again.
#[test]
fn a_look_that_finds_no_lease_on_file_ends_the_watch() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    // The look applies the release of the file's last lease.
    assert_eq!(place.look(), LookOutcome::Ended);
    assert_eq!(store.watches(), 0);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 0)])
    );
    // A lease laid on a file nobody watches: its frame is applied by its
    // hand.
    let (second, _) = issued_at(&store, a, now);
    store.hand(&hand, LeaseId(1 << 32 | second), GONE);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(second, GONE)], vec![(a, 0)])
    );
    assert_eq!(
        place.look(),
        LookOutcome::Ended,
        "no lease stands, and no watch"
    );
}

/// A watch ends by its word first and one more look behind it. A frame a
/// hand left under the old word is applied by that look, and a lease it
/// finds on file keeps the watch: the cases lay the lease and the frame
/// that reach a file between a look that found none and the word's end.
#[test]
fn a_watch_ends_behind_one_more_look() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), []);
    // The end's own look applies the frame and finds the second lease:
    // the watch stands, and the next frame is left to it.
    assert_eq!(place.end_watch(), LookOutcome::Fed);
    assert_eq!(
        (told_ids(), in_flight(&store), store.watches()),
        (vec![(1, GONE)], vec![(a, 1)], 1)
    );
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), [], "left to the watch that stands");
    // With the last lease's release in the ring the end applies it, and
    // the watch ends: the next frame is applied by its hand.
    assert_eq!(place.end_watch(), LookOutcome::Ended);
    assert_eq!(
        (told_ids(), in_flight(&store), store.watches()),
        (vec![(2, GONE)], vec![(a, 0)], 0)
    );
    let (third, _) = issued_at(&store, a, now);
    store.hand(&hand, LeaseId(1 << 32 | third), GONE);
    assert_eq!(told_ids(), [(third, GONE)]);
}

/// A thread that rests, or drops its place, ends its watch whatever
/// stands on its file: the hands apply in place from there.
#[test]
fn a_place_that_rests_or_drops_ends_its_watch() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(4, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), []);
    // Its rest applies what was handed, and ends the watch over the three
    // leases that stand.
    place.rest();
    assert_eq!((told_ids(), store.watches()), (vec![(1, GONE)], 0));
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), [(2, GONE)]);
    // A place dropped on its watch does the same.
    place.work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[2], GONE);
    assert_eq!(told_ids(), []);
    drop(place);
    assert_eq!((told_ids(), store.watches()), (vec![(3, GONE)], 0));
    store.hand(&hand, ids[3], GONE);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(4, GONE)], vec![(a, 0)])
    );
}

/// The sweep applies at a watched file while the file's thread waits: a
/// dispatch left in the ring is applied before the sweep reads a
/// deadline, so its lease is not ended as one no gateway dispatched.
#[test]
fn the_sweep_applies_at_a_watched_file_whose_thread_waits() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    assert_eq!(told_ids(), []);
    // Past both leases' dispatch deadline.
    let ended = LeasesStore::expire(&*store, now + Duration::from_secs(6));
    let ended: Vec<_> = ended.iter().map(|found| found.record.lease.id).collect();
    assert_eq!(ended, [ids[1]]);
    let abandoned = Said::Released(None, ReleaseReason::Abandoned);
    assert_eq!(told_ids(), [(1, dispatch), (2, abandoned)]);
    assert_eq!(in_flight(&store), [(a, 1)]);
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "the dispatched lease stands"
    );
}

/// The sweep applies a release left by a thread that unwound even where
/// the dispatched lease has no place left in the deadline queue.
#[test]
fn the_sweep_applies_a_release_left_before_an_unwind() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    assert_eq!(told_ids(), [(1, dispatch)]);
    assert!(LeasesStore::expire(&*store, now + Duration::from_secs(6)).is_empty());
    assert_eq!(store.queued(), 0, "the dispatched lease's place is popped");

    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut place = store.at_work();
        assert_eq!(place.look(), LookOutcome::Quiet);
        store.hand(&hand, ids[0], GONE);
        assert_eq!(told_ids(), []);
        panic!("leave the release for the sweep");
    }));
    assert!(unwound.is_err());

    assert!(LeasesStore::expire(&*store, now + Duration::from_secs(6)).is_empty());
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 0)])
    );
}

/// A full ring under a watch: the hand that finds it applies where the
/// watching thread waits, in the order the frames were handed, and no
/// frame is dropped.
#[test]
fn a_full_ring_under_a_watch_is_applied_by_its_hand() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    for id in &ids[..MemoryStore::HANDOVER] {
        store.hand(&hand, *id, GONE);
    }
    assert_eq!(told_ids(), [], "the ring holds its room");
    // The next hand finds the ring full and applies it; its own frame
    // waits for the watch.
    store.hand(&hand, ids[MemoryStore::HANDOVER], GONE);
    let applied = told_ids();
    assert_eq!(applied.len(), MemoryStore::HANDOVER);
    assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room));
    assert_eq!(
        place.look(),
        LookOutcome::Fed,
        "one lease stands behind the look"
    );
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(room + 1, GONE)], vec![(a, 1)])
    );
}

/// A place applies what waits at each other file whose thread waits, for a
/// decision under a stage that may refuse by a report of leases. A file
/// whose thread is at work is left to that thread, as a hand leaves it.
/// The second file's thread is the case's own, on its watch and then at
/// work.
#[test]
fn a_place_applies_what_waits_at_the_other_files_whose_threads_wait() {
    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (said, heard_of) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the other
        // thread's waits end, and the case with it.
        let (heard_of, go) = (heard_of, go);
        let store = &store;
        scope.spawn(move || {
            let _acting = DecisionSlot::new(1).enter();
            let ids: Vec<_> = (0..2).map(|_| issued_at(store, a, now).0).collect();
            let mut place = store.at_work();
            assert_eq!(
                place.look(),
                LookOutcome::Quiet,
                "two leases stand on its file"
            );
            let second = ids[1];
            said.send(ids).expect("the case hears the ids");
            word.recv_timeout(PATIENCE).expect("the case's word");
            place.work();
            said.send(Vec::new()).expect("the case hears the work");
            word.recv_timeout(PATIENCE).expect("the case's word");
            drop(place);
            assert_eq!(
                told_ids(),
                [(second, GONE)],
                "what waited is its own as it leaves"
            );
        });
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let ids = heard_of.recv_timeout(PATIENCE).expect("the watch stands");
        let hand = Hand::new(GatewayId(1));
        drop(told());
        // The other file's thread waits on its watch: the frame a hand
        // left there is applied by this place.
        store.hand(&hand, LeaseId(1 << 32 | ids[0]), GONE);
        assert_eq!(told_ids(), [], "left to the watch");
        place.apply_waiting();
        assert_eq!(told_ids(), [(ids[0], GONE)]);
        // That thread is at work: what waits is its own.
        go.send(()).expect("the other thread waits");
        heard_of.recv_timeout(PATIENCE).expect("the work began");
        store.hand(&hand, LeaseId(1 << 32 | ids[1]), GONE);
        place.apply_waiting();
        assert_eq!(told_ids(), [], "left to the thread at work");
        go.send(()).expect("the other thread waits");
    });
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A file that shows no watch holds nothing for a look, so a place's apply
/// passes it by its word and takes no turn there. The case's tell runs
/// under the second file's turn, where a hand applies in place, and
/// applies through the place at the first file: a take of the held turn
/// would be refused.
#[test]
fn a_places_apply_takes_no_turn_at_a_file_that_shows_no_watch() {
    thread_local! {
        static PLACE: RefCell<Option<AtWork>> = const { RefCell::new(None) };
        static TOLD: Cell<u32> = const { Cell::new(0) };
    }
    fn under(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        PLACE.with_borrow(|place| place.as_ref().expect("the case's place").apply_waiting());
        TOLD.set(TOLD.get() + 1);
    }
    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    store.tell_through(under);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    // A lease on the second file, issued as its decider.
    let lease = {
        let _acting = DecisionSlot::new(1).enter();
        issued_at(&store, a, now).0
    };
    let _acting = DecisionSlot::new(0).enter();
    PLACE.set(Some(store.at_work()));
    // The second file shows no watch and no thread works there: the hand
    // applies in place, and the tell runs under that file's turn.
    store.hand(&Hand::new(GatewayId(1)), LeaseId(1 << 32 | lease), GONE);
    assert_eq!(TOLD.get(), 1, "the release was told under its file's turn");
    PLACE.set(None);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A place's apply takes its own file's turn by a wait and applies what
/// was handed while its thread worked. The thread's look at a request is
/// one try of that turn, which another file's decision may hold for a
/// read: a decision under a stage that may refuse by a report of leases
/// does not read by that try alone.
#[test]
fn a_places_apply_applies_what_was_handed_at_its_own_file() {
    let (store, a, ids) = handed_bed(1, Instant::now());
    let place = store.at_work();
    // Handed while the thread works: left to it.
    store.hand(&Hand::new(GatewayId(1)), ids[0], GONE);
    assert_eq!(told_ids(), [], "left to the thread at work");
    place.apply_waiting();
    assert_eq!(told_ids(), [(ids[0].0 & 0xffff_ffff, GONE)]);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A place's apply reads behind what a watch kept, whatever the watching
/// thread does meanwhile. A hand leaves a release under the second
/// file's watch and returns; this place then applies what waits at the
/// other files and reads the worker's count once, as a decision over
/// worker lists does: the count its host keeps, with no lock taken. In
/// every other round the second file's thread takes up a job as that
/// read is made: it says it works under its turn, with its apply behind
/// the mark, so the read finds the release applied, by this place or by
/// the job's first step. A hand at a file with no watch would have
/// applied it before it returned.
#[test]
fn a_places_apply_reads_behind_a_watchs_frames_beside_a_job_that_begins() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

    const ROUNDS: u64 = 20_000;
    // The widest pause, in spins, between the word that lets the other
    // thread take up its job and this place's apply: the rounds sweep it.
    const SPINS: u64 = 64;
    // The words of a round, said by each side in turn: the other thread's
    // watch stands over the round's lease, it may take up its job, it is
    // at work, the round is over.
    const WATCHES: u64 = 1;
    const TAKE_UP: u64 = 2;
    const WORKS: u64 = 3;
    const OVER: u64 = 4;

    // Says the case left, by its end or by an unwind: the other thread
    // waits for no word behind it.
    struct Leaves<'a>(&'a AtomicBool);
    impl Drop for Leaves<'_> {
        fn drop(&mut self) {
            self.0.store(true, SeqCst);
        }
    }

    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (word, lease, left) = (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false));
    std::thread::scope(|scope| {
        let (store, word, lease, left) = (&store, &word, &lease, &left);
        scope.spawn(move || {
            let _acting = DecisionSlot::new(1).enter();
            let mut place = store.at_work();
            // Waits for the case's word, with `pause` between its reads;
            // `false` where the case left. The wait for its job is one
            // side of the race and spins. The wait for the round's end
            // is outside it and yields, as the case's own waits do: on a
            // host with no core to spare a spin can cost a time slice.
            let heard = |wanted: u64, pause: fn()| {
                while word.load(SeqCst) != wanted {
                    if left.load(SeqCst) {
                        return false;
                    }
                    pause();
                }
                true
            };
            loop {
                // At work: the round's lease, then the watch over it.
                let id = issued_at(store, a, now).0;
                assert_ne!(
                    place.look(),
                    LookOutcome::Ended,
                    "a lease stands on the file"
                );
                lease.store(id, SeqCst);
                word.store(WATCHES, SeqCst);
                if !heard(TAKE_UP, std::hint::spin_loop) {
                    return;
                }
                // A job: the first step of its work applies what waits.
                place.work();
                drop(told());
                word.store(WORKS, SeqCst);
                if !heard(OVER, std::thread::yield_now) {
                    return;
                }
            }
        });
        let _leaves = Leaves(left);
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let hand = Hand::new(GatewayId(1));
        let wait_for = |wanted: u64| {
            let since = Instant::now();
            while word.load(SeqCst) != wanted {
                assert!(
                    since.elapsed() < PATIENCE,
                    "the other thread's word {wanted}"
                );
                std::thread::yield_now();
            }
        };
        let counted = standing(store, a);
        for round in 1..=ROUNDS {
            wait_for(WATCHES);
            drop(told());
            store.hand(&hand, LeaseId(1 << 32 | lease.load(SeqCst)), GONE);
            // Every other round the other thread takes up its job as this
            // place applies and reads, a swept pause apart.
            let begins = round % 2 == 0;
            if begins {
                word.store(TAKE_UP, SeqCst);
                for _ in 0..round / 2 % SPINS {
                    std::hint::spin_loop();
                }
            }
            place.apply_waiting();
            assert_eq!(
                counted.load(Ordering::Relaxed),
                0,
                "round {round}: the read is behind the release's apply"
            );
            if !begins {
                assert!(!told().is_empty(), "round {round}: this place applied it");
                word.store(TAKE_UP, SeqCst);
            }
            wait_for(WORKS);
            word.store(OVER, SeqCst);
        }
    });
}

/// The floor of the price of a place's apply at the other files, measured:
/// a million calls with nothing waiting and no other thread, at one, two
/// and eight files that show no watch, then at two and eight where each
/// other file shows one. Its own file's turn is in each call.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn a_places_apply_at_the_other_files_measured() {
    use std::io::Write as _;

    const CALLS: u32 = 1_000_000;
    let mut out = std::io::stdout();
    let now = Instant::now();
    for (files, watched) in [(1_u8, false), (2, false), (8, false), (2, true), (8, true)] {
        let store = Arc::new(MemoryStore::new().with_decision_slots(usize::from(files)));
        let a = ready(&store, "http://a:1");
        LeasesStore::set_term(&*store, 1);
        // A watch at each other file: a lease on it and its place's look.
        let others = if watched { 1..files } else { 0..0 };
        let _watches: Vec<AtWork> = others
            .map(|other| {
                let _acting = DecisionSlot::new(other).enter();
                let mut place = store.at_work();
                issued_at(&store, a, now);
                assert_ne!(
                    place.look(),
                    LookOutcome::Ended,
                    "a lease stands on the file"
                );
                place
            })
            .collect();
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let since = Instant::now();
        for _ in 0..CALLS {
            place.apply_waiting();
        }
        let nanos = since.elapsed().as_nanos() / u128::from(CALLS);
        let shown = if watched {
            "a watch at each other"
        } else {
            "no watch"
        };
        writeln!(
            out,
            "a place's apply, {files} files, {shown}, nothing waiting: {nanos} ns a call"
        )
        .expect("stdout");
    }
}

/// The frames of one take are told under one world, the store's as it
/// stood at the take's first tell: a take loads it once, however many
/// frames it tells.
#[test]
fn the_frames_of_a_take_are_told_under_one_world() {
    let (store, _, ids, first) = under_bed(3, Instant::now());
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[1], GONE);
    at_work.apply();
    let later = store.snapshot().epoch();
    assert!(later > first);
    assert_eq!(UNDER.take(), [first, first], "one load for the take");
    store.hand(&hand, ids[2], GONE);
    at_work.apply();
    assert_eq!(UNDER.take(), [later], "the next take's own");
}

/// The leases one issue ends are told under one world, the store's as it
/// stood at the first end's tell: an issue loads it once for its ends.
/// The next issue's ends are told under its own.
#[test]
fn the_leases_an_issue_ends_are_told_under_one_world() {
    let now = Instant::now();
    let (store, a, _, first) = under_bed(3, now);
    // Six seconds on, all three are past their dispatch deadline: this
    // issue ends two, and the one behind it the third.
    let late = now + Duration::from_secs(6);
    issued_at(&store, a, late);
    let later = store.snapshot().epoch();
    assert!(later > first);
    assert_eq!(
        UNDER.take(),
        [first, first],
        "one load for the issue's ends"
    );
    issued_at(&store, a, late);
    assert_eq!(UNDER.take(), [later], "the next issue's own");
}

/// The leases a sweep ends in a file are told under one world, the
/// store's as it stood at the first end's tell.
#[test]
fn the_leases_a_sweep_ends_in_a_file_are_told_under_one_world() {
    let now = Instant::now();
    let (store, _, ids, first) = under_bed(3, now);
    let ended = LeasesStore::expire(&*store, now + Duration::from_secs(6));
    assert_eq!(ended.len(), ids.len());
    assert!(store.snapshot().epoch() > first);
    assert_eq!(
        UNDER.take(),
        [first, first, first],
        "one load for the sweep"
    );
}

/// An issue's frames and its ends are two runs, told under a load each:
/// the frames under the world that stood at the apply's first tell, the
/// ends under the one that stands at the first end's. This holds the
/// split as a known limit, not as an aim: where one load serves both
/// runs, the ends read the first world and this case turns with it.
#[test]
fn an_issues_frames_and_its_ends_are_told_under_a_load_each() {
    let now = Instant::now();
    let (store, a, ids, first) = under_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    // Six seconds on the other lease is past its dispatch deadline: this
    // issue applies the handed frame, then ends that lease.
    issued_at(&store, a, now + Duration::from_secs(6));
    drop(at_work);
    let later = store.snapshot().epoch();
    assert!(later > first);
    assert_eq!(UNDER.take(), [first, later], "a load for each run");
}

/// A decider looks at what is handed before a decision reads its rows'
/// counts, and that look never waits: a turn it finds held it reads past.
#[test]
fn a_thread_at_work_does_not_wait_for_a_held_turn_before_its_reads() {
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (t0, world) = (Instant::now(), store.snapshot());
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
    fn takes_a_door(
        store: &MemoryStore,
        _: &StoreSnapshot,
        _: &Sides,
        _: Said,
        _: Option<DecisionSlot>,
    ) {
        let _live = LeasesStore::live(store, GatewayId(1));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(takes_a_door);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        HEARD.fetch_add(1, SeqCst);
    }
    const ROUNDS: usize = 100_000;
    /// How long the rounds run at most: past it the case asks no more.
    const RUN: Duration = Duration::from_secs(20);
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
    let (gateway, now, world) = (GatewayId(1), Instant::now(), store.snapshot());
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
            if now.elapsed() > RUN {
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
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
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
            store.snapshot().booked_of(a),
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
            let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| {
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
    fn prints(store: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        PRINTED.with(|printed| *printed.borrow_mut() = format!("{store:?}"));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(prints);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
        let world = store.snapshot();
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
        let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let fourth = {
        let _acting = DecisionSlot::new(3).enter();
        issue(&store, a)
    };
    let first = issue(&store, a);
    // Their ends are applied by a thread that names none.
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, fourth, GONE);
    store.hand(&hand, first, GONE);
    assert_eq!(
        told_by(),
        [Some(DecisionSlot::new(3)), Some(DecisionSlot::new(0))]
    );
    let plain = MemoryStore::new();
    let a = ready(&plain, "http://a:1");
    LeasesStore::set_term(&plain, 1);
    plain.tell_through(heard);
    let named = {
        let _acting = DecisionSlot::new(3).enter();
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
    fn ended(
        store: &MemoryStore,
        _: &StoreSnapshot,
        found: &Sides,
        said: Said,
        _: Option<DecisionSlot>,
    ) {
        told_across(store, found.record.lease.id, Some(said));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(ended);
    let (t0, world) = (Instant::now(), store.snapshot());
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
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(3));
    assert_eq!(store.decision_slots(), 3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
            let _acting = DecisionSlot::new(2).enter();
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| {
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
    let (second, third) = (
        (second, DecisionSlot::new(1)),
        (third, DecisionSlot::new(2)),
    );
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
    let cut = leader.checkpoint();
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
    fn parks(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        let ends = ENDS.lock().expect("no case fails under it").take();
        if let Some((reached, word)) = ends {
            reached.send(()).expect("the case hears the tell");
            word.recv_timeout(PATIENCE)
                .expect("the case lets the teller go");
        }
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(parks);
    let (gateway, now) = (GatewayId(1), Instant::now());
    // A lease on the second decider's file.
    let lease = {
        let _acting = DecisionSlot::new(1).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
            let _acting = DecisionSlot::new(1).enter();
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
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (first, second) = (issue(0), issue(1));
    let late = now + Duration::from_secs(60);
    let ended = |store: &MemoryStore| {
        let ended = LeasesStore::expire(store, late);
        ended
            .iter()
            .map(|found| found.record.lease.id)
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
            let _acting = DecisionSlot::new(0).enter();
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
    let told = [
        (second, DecisionSlot::new(1)),
        (first, DecisionSlot::new(0)),
    ];
    assert_eq!(*names, told.map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A gateway's leases stand in the files of the deciders that issued them:
/// its live count, its restore and its end reach every file.
#[test]
fn a_gateways_doors_reach_its_leases_in_every_file() {
    let store = MemoryStore::new().with_decision_slots(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (ours, theirs, now) = (GatewayId(1), GatewayId(2), Instant::now());
    let issue = |decider: u8, gateway| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    let standby = MemoryStore::new().with_decision_slots(2);
    let term: &dyn LeasesStore = &standby;
    term.set_term(3);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, under: u64| {
        let _acting = DecisionSlot::new(decider).enter();
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
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = MemoryStore::new().with_decision_slots(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, term: u64| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, term, now)
        };
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    let told = [
        (before, DecisionSlot::new(2)),
        (after, DecisionSlot::new(1)),
    ];
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
    let store = MemoryStore::new().with_decision_slots(2);
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
        let _acting = DecisionSlot::new(1).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, 3, now)
        };
        let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    let store = MemoryStore::new().with_decision_slots(2);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 3);
    let (store, booked) = (&store, store.snapshot().booked_of(a));
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
            let _acting = DecisionSlot::new(0).enter();
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
        let _acting = DecisionSlot::new(1).enter();
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
    let store = MemoryStore::new().with_decision_slots(2);
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
    let store = MemoryStore::new().with_decision_slots(256);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let first = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        HEARD.fetch_add(1, SeqCst);
    }
    let store = MemoryStore::new();
    store.tell_through(counted);
    assert!(store.call_faulted());
    let store = store.with_decision_slots(2);
    assert_eq!((store.decision_slots(), store.lifecycle_faults()), (2, 1));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let record = booking(gateway, 0, a, now);
    let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
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
    drop(store.with_decision_slots(2));
}

/// Nor is a store laid again with a lease on a later file: the guard
/// reads every file, each under its own turn.
#[test]
#[should_panic(expected = "a store's files are laid before a lease is filed")]
fn a_store_with_a_lease_on_a_later_file_is_not_laid_again() {
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    {
        // A thread that acts as the fourth decider files on the fourth
        // file: the first file stays empty.
        let (_acting, world) = (DecisionSlot::new(3).enter(), store.snapshot());
        let record = booking(GatewayId(1), 0, a, Instant::now());
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term");
    }
    drop(store.with_decision_slots(8));
}
