//! The contract tests of the store ingress, run over [`MemoryStore`] through
//! the trait objects: what every implementation of a trait promises.

use std::time::{Duration, Instant};

use super::{
    identity, identity_of, BatchFacts, BatchOutcome, GatewayId, HealthSink, KvEventSink,
    LeaseRecord, LeaseRefused, LeaseState, LeasesStore, LoadSink, LoadSource, MembershipChange,
    MembershipTail, MemoryStore, Pin, PollMode, ReservationKind, ReservationStore, ResyncReason,
    SessionsStore, SnapshotChunk, WorkerFacts, WorkerPatch, WorkerStatus, WorkersStore,
};
use crate::{
    lease::{Lease, LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    view::{fingerprint, LoadRow, SignalSource, WorkerId},
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

fn lease(gateway: GatewayId, id: u64, term: u64, now: Instant) -> LeaseRecord {
    LeaseRecord {
        lease: Lease {
            id: LeaseId(id),
            ..Lease::default()
        },
        gateway,
        request: id,
        target: WorkerId(7),
        decode: None,
        term,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(1),
        state: LeaseState::Issued,
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
        Some(crate::lease::Membership::Replaced { old, new })
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
    other_role.role = crate::view::Role::Prefill;
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
    assert_eq!(health.probe_outcome(id, true, revision()), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "two passes promote a pending worker"
    );
    let stale = revision() - 1;
    assert_eq!(
        health.probe_outcome(id, false, stale),
        None,
        "a stale outcome is dropped"
    );
    assert_eq!(health.probe_outcome(id, false, revision()), None);
    assert_eq!(health.probe_outcome(id, false, revision()), None);
    assert_eq!(
        health.probe_outcome(id, false, revision()),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady)),
        "three failures demote"
    );
    for _ in 0..8 {
        assert_eq!(health.probe_outcome(id, false, revision()), None);
    }
    assert_eq!(
        health.probe_outcome(id, false, revision()),
        Some((WorkerStatus::NotReady, WorkerStatus::Failed)),
        "nine more fail it"
    );
    assert_eq!(health.probe_outcome(id, true, revision()), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Some((WorkerStatus::Failed, WorkerStatus::Ready)),
        "two passes bring a failed worker back"
    );
    let fresh = store.upsert(facts("http://f:8000", 1)).id();
    let fresh_revision = WorkersStore::get(&store, fresh).expect("the row").revision;
    assert_eq!(
        health.connected(fresh, fresh_revision),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "the connect signal promotes at once"
    );
    assert_eq!(
        health.connected(fresh, fresh_revision),
        None,
        "the promotion moved the revision"
    );
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

/// A poll's row stands in for a silent rank's sample but never for its
/// receipt: the poll decision keeps asking while that rank stays silent,
/// whatever its neighbour pushes.
#[test]
fn fallback_polling_stays_while_a_silent_rank_stays_silent() {
    let store = MemoryStore::new();
    let sink: &dyn LoadSink = &store;
    let id = WorkerId(12);
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
    sink.report(id, &[(0, row), (1, row)], later, LoadSource::Poll);
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
    let id = WorkerId(3);
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
        id,
        &[(0, polled), (1, polled), (2, polled)],
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
fn the_poll_mode_reads_the_receipts_of_the_pushing_ranks_alone() {
    let store = MemoryStore::new();
    let loads: &dyn LoadSink = &store;
    let id = WorkerId(3);
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    assert_eq!(loads.poll_mode(id, t0, interval), PollMode::Poll);
    loads.report(id, &[(0, row(0))], t0, LoadSource::Poll);
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
    let id = WorkerId(3);
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

#[test]
fn kv_batches_apply_in_sequence_skip_duplicates_and_ask_one_replay_per_gap() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = ()> = &store;
    let id = WorkerId(9);
    assert_eq!(sink.cursor(id, 0), None, "a rank never heard from");
    assert_eq!(sink.admit(id, 0, batch(17), &()), BatchOutcome::Applied);
    assert_eq!(sink.admit(id, 0, batch(18), &()), BatchOutcome::Applied);
    assert_eq!(
        sink.admit(id, 0, batch(18), &()),
        BatchOutcome::Skipped,
        "a duplicate"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 18);
    assert_eq!(
        sink.admit(id, 0, batch(25), &()),
        BatchOutcome::Gap {
            expected: 19,
            received: 25
        }
    );
    assert!(sink.cursor(id, 0).expect("heard").replay_pending);
    for seq in 19..=25 {
        assert_eq!(
            sink.admit(id, 0, batch(seq), &()),
            BatchOutcome::Applied,
            "the replay fills the gap"
        );
    }
    assert_eq!(store.applied_batches(id, 0), 9);
    assert_eq!(
        sink.admit(id, 0, batch(40), &()),
        BatchOutcome::Gap {
            expected: 26,
            received: 40
        }
    );
    assert_eq!(
        sink.admit(id, 0, batch(50), &()),
        BatchOutcome::Applied,
        "a second gap is lost history, applied"
    );
    let cursor = sink.cursor(id, 0).expect("heard");
    assert!(cursor.degraded && !cursor.replay_pending);
    assert_eq!(
        store.applied_batches(id, 0),
        10,
        "a small gap keeps the state"
    );
    assert_eq!(sink.admit(id, 1, batch(3), &()), BatchOutcome::Applied);
    assert_eq!(
        sink.cursor(id, 1).expect("heard").applied,
        3,
        "each rank its own counter"
    );
}

/// The same sequence below the cursor is a late duplicate on the connection
/// that produced it and a restarted publisher on a fresh one; neither a
/// clear nor the restart window is needed to tell them apart.
#[test]
fn a_fresh_connection_below_the_cursor_is_a_restart_where_a_late_duplicate_is_not() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = ()> = &store;
    let id = WorkerId(11);
    for seq in 1..=5 {
        assert_eq!(sink.admit(id, 0, batch(seq), &()), BatchOutcome::Applied);
    }
    assert_eq!(
        sink.admit(id, 0, batch(3), &()),
        BatchOutcome::Skipped,
        "a late duplicate on the same connection"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 5);
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 0);
    sink.reconnected(id);
    assert_eq!(
        sink.admit(id, 0, batch(3), &()),
        BatchOutcome::Applied,
        "the same sequence on a fresh connection is a restarted publisher"
    );
    let cursor = sink.cursor(id, 0).expect("heard");
    assert_eq!((cursor.applied, cursor.degraded), (3, false));
    assert_eq!(
        store.applied_batches(id, 0),
        1,
        "the rank was cleared first"
    );
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    assert_eq!(
        sink.admit(id, 0, batch(4), &()),
        BatchOutcome::Applied,
        "the new life continues from there"
    );
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 4);
}

#[test]
fn a_publisher_restart_and_a_resync_clear_the_rank_and_a_removal_forgets_it() {
    let store = MemoryStore::new();
    let sink: &dyn KvEventSink<Batch = ()> = &store;
    let id = WorkerId(9);
    for seq in 1..=5 {
        assert_eq!(sink.admit(id, 0, batch(seq), &()), BatchOutcome::Applied);
    }
    let clearing = BatchFacts {
        sequence: 2,
        clears: true,
        snapshot: None,
    };
    assert_eq!(
        sink.admit(id, 0, clearing, &()),
        BatchOutcome::Applied,
        "a clear below the cursor is a restarted publisher"
    );
    assert_eq!(
        store.applied_batches(id, 0),
        1,
        "the rank was cleared first"
    );
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    sink.reconnected(id);
    assert_eq!(
        sink.admit(id, 0, batch(1), &()),
        BatchOutcome::Applied,
        "below the cursor on a fresh connection: a restart too"
    );
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 2);
    for seq in 2..=4 {
        sink.admit(id, 0, batch(seq), &());
    }
    sink.resync(id, Some(0), ResyncReason::OutOfRange);
    assert_eq!(store.applied_batches(id, 0), 0);
    assert_eq!(sink.cursor(id, 0).expect("heard").applied, 0);
    assert_eq!(store.resyncs(ResyncReason::OutOfRange), 1);
    let chunk = BatchFacts {
        sequence: 90,
        clears: false,
        snapshot: Some(SnapshotChunk {
            index: 0,
            count: 2,
            unknown_before: true,
        }),
    };
    assert_eq!(sink.admit(id, 0, chunk, &()), BatchOutcome::Applied);
    let cursor = sink.cursor(id, 0).expect("heard");
    assert_eq!((cursor.applied, cursor.degraded), (90, true));
    assert_eq!(store.resyncs(ResyncReason::Snapshot), 1);
    assert_eq!(
        sink.admit(id, 0, batch(91), &()),
        BatchOutcome::Applied,
        "live batches continue after the snapshot's stamp"
    );
    sink.removed(id);
    assert_eq!(sink.cursor(id, 0), None);
    assert_eq!(store.applied_batches(id, 0), 0);
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

#[test]
fn a_pin_is_written_at_the_commit_and_leaves_with_its_worker_or_its_age() {
    let store = MemoryStore::new();
    let sessions: &dyn SessionsStore = &store;
    let now = Instant::now();
    sessions.pin(0, WorkerId(1), now);
    assert!(sessions.is_empty(), "the absent key is never pinned");
    sessions.pin(11, WorkerId(1), now);
    sessions.pin(12, WorkerId(2), now + Duration::from_secs(5));
    sessions.pin(11, WorkerId(3), now + Duration::from_secs(1));
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
        },
    );
    assert_eq!(sessions.len(), 3);
    assert_eq!(sessions.drop_worker(WorkerId(9)), 1);
    assert_eq!(sessions.expire(now + Duration::from_secs(2)), 1);
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
    assert!(leases.dispatched(gateway, LeaseId(1), Side::Single, now));
    assert!(!leases.dispatched(GatewayId(2), LeaseId(1), Side::Single, now));
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
    assert!(!leases.released(gateway, LeaseId(1), ReleaseReason::ClientGone));
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
    assert_eq!(leases.drop_gateway(g1), 2);
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
