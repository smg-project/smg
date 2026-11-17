//! The contract tests of the store ingress, run over [`MemoryStore`] through
//! the trait objects: what every implementation of a trait promises.

use std::time::{Duration, Instant};

use super::{
    identity, identity_of,
    leases::{Counts, Slot},
    Ask, BatchFacts, BatchOutcome, ContentHash, Dispatch, GatewayId, HealthSink, KvEvent,
    KvEventSink, LeaseRecord, LeaseRefused, LeaseState, LeasesStore, LoadSink, LoadSource,
    MembershipChange, MembershipTail, MemoryStore, Picked, Pin, PollMode, Reachability,
    ReservationKind, ReservationStore, ResyncReason, SequenceHash, SessionsStore, Signals,
    SnapshotChunk, StoredBlock, VetoWrite, ViewBuffers, WorkerFacts, WorkerPatch, WorkerStatus,
    WorkersStore,
};
use crate::{
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
        picked: Picked::default(),
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
    assert_eq!(health.probe_outcome(id, true, revision()), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "two passes promote a pending worker"
    );
    // A patch moves the revision: an outcome read under the one it left is
    // stale.
    let stale = revision();
    WorkersStore::patch(&store, id, WorkerPatch::default(), stale).expect("the patch");
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
                health.probe_outcome(id, false, revision()),
                None,
                "pending under {threshold}"
            );
        }
        assert_eq!(health.probe_outcome(id, true, revision()), None);
        assert_eq!(
            health.probe_outcome(id, true, revision()),
            Some((WorkerStatus::Pending, WorkerStatus::Ready))
        );
        // And a ready one stays ready through failures under the threshold.
        for _ in 0..5 {
            assert_eq!(
                health.probe_outcome(id, false, revision()),
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
    assert_eq!(health.probe_outcome(id, false, revision()), None);
    assert_eq!(
        health.probe_outcome(id, false, revision()),
        Some((WorkerStatus::Pending, WorkerStatus::Failed))
    );
    // Each counter at its end takes one probe more and stays there.
    store.probed(id, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(health.probe_outcome(id, false, revision()), None);
    store.probed(id, 0, u32::MAX, u32::MAX);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
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
    assert_eq!(health.probe_outcome(id, true, revision()), None);
    assert_eq!(
        health.probe_outcome(id, true, revision()),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    // Ready, the row is not ready at the threshold's own count of failed
    // probes: the count is set one short of it.
    store.probed(id, threshold - 1, 0, 0);
    assert_eq!(
        health.probe_outcome(id, false, revision()),
        Some((WorkerStatus::Ready, WorkerStatus::NotReady))
    );
    // Not ready, it fails at three thresholds of failed probes: five are
    // far under them.
    for _ in 0..5 {
        assert_eq!(
            health.probe_outcome(id, false, revision()),
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
        health.connected(id, read),
        None,
        "a stale signal is dropped"
    );
    assert_eq!(
        WorkersStore::get(&store, id).expect("the row").status,
        WorkerStatus::Pending,
        "and the row stays as it stood"
    );
    assert_eq!(
        health.connected(id, read + 1),
        Some((WorkerStatus::Pending, WorkerStatus::Ready)),
        "the signal read under the row's revision promotes"
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
    rows.map(|row| (row.id, row.in_flight)).collect()
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
    // An index the pick holds no target at moves the state and no count.
    leases.issue(picked(2)).expect("filed");
    assert!(leases.dispatched(gateway, LeaseId(2), on(2), now));
    assert_eq!(in_flight(&store), of(1, 0));
    let state = leases.get(gateway, LeaseId(2)).map(|record| record.state);
    assert_eq!(state, Some(LeaseState::Dispatched { at: now }));
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
        let issued = store.issue_decided(record, world.booked_of(a));
        issued.map(|(lease, _)| lease.id.0)
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

/// An issue at `at` on `target` with five seconds to its dispatch: the
/// counter of its id, and the counters of the leases it ended.
fn issued_at(store: &MemoryStore, target: WorkerId, at: Instant) -> (u64, Vec<u64>) {
    let counter = |id: LeaseId| id.0 & 0xffff_ffff;
    let record = LeaseRecord {
        dispatch_deadline: at + Duration::from_secs(5),
        ..booking(GatewayId(1), 0, target, at)
    };
    let world = store.world();
    let (lease, ended) = store
        .issue_decided(record, world.booked_of(target))
        .expect("minted under the store's term");
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    let ended = ended.iter().flatten().map(|record| {
        assert_eq!(record.state, abandoned);
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
