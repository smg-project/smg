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
    AtWork, DecodeMark, DecodeSide, GatewayId, Hand, HealthSink, Journal, LeaseRecord,
    LeaseRefused, LeaseState, LeasesStore, LookOutcome, MembershipChange, MemoryStore, PickRecord,
    Position, Sides, StoreSnapshot, VetoWrite, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus,
    WorkersStore,
};
use crate::{
    document::applied::AppliedSpec,
    engine::{
        decision::{self, Target, Targets},
        decision_engine::DecodeRefused,
        decision_slot::DecisionSlot,
        demands::DemandFacts,
        sticky_sessions::SessionVerdict,
        view::{fingerprint, Role, Session, WorkerId},
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

mod counts;
mod deciders;
mod gateways;
mod host_count;
mod leases;
mod pairs;
mod pins;
mod snapshots;
mod turn;
mod workers;

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

/// The manual preset as `generation`, its keys pinned for `ttl_s` and swept
/// every `sweep_s`.
fn manual_ruled(generation: u64, ttl_s: u32, sweep_s: u32) -> AppliedSpec {
    use crate::document::spec::PolicySpec;

    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = generation;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (ttl_s, sweep_s);
    AppliedSpec::validate(&document).expect("the document validates")
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
        session_verdict: SessionVerdict::None,
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
