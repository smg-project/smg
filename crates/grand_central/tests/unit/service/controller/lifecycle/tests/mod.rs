// What an applied document keeps of the stages, and who hears a lease
// of the document before it.
mod kept_stages;
// The rule a guard's lease is told by.
mod lease_on_file;
mod pair_on_file;

use std::{
    cell::RefCell,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tonic::{Code, Status};

use super::{
    class, completed, dispatch, dispatched, outcome, reason, release, released, side, tells, told,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Pair, Target, Targets},
        framework::{Picker, TieBand},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Health, Phase, RequestFacts, Role, Session, WireModes, WorkerId},
        Constraints, PoolFacts, Scratch,
    },
    policies::random::Random,
    service::{
        controller::{
            decision_slots::tests::{holder, ready},
            gateways::tests::lease,
        },
        proto::{
            self,
            completed::Class,
            released::{Reason, What},
        },
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        memory::seam, DecodeSide, GatewayId, Hand, LeaseRecord, LeaseState, LeasesStore,
        MemoryStore, PickRecord, Registrant, WorkerFacts, WorkersStore,
    },
    tracking::{
        lease::{
            Dispatch, EventKind, EventSet, Lease, LeaseId, Outcome, ReleaseReason, Side,
            TransportClass,
        },
        DispatchCredit,
    },
};

const A: GatewayId = GatewayId(11);
const B: GatewayId = GatewayId(12);
/// The session of the keyed leases.
const KEY: Session = Session::from_raw(41);
/// The lease whose release the hearing stage panics at.
const BURSTS: u64 = 99;

thread_local! {
    /// A case's own step inside a stage's call: the hearing stage's
    /// next release on this thread runs it once, before it returns.
    static AT_RELEASE: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

/// A hand at the leases' files of a case's store, which no deciders
/// were started over. Where the store holds no word to the engine it
/// is given the engine's own ([`told`]) with no worker object to ask:
/// a count takes a send at a rank it does not know. The first word a
/// store is given stays.
pub(crate) fn open(store: &MemoryStore, gateway: GatewayId) -> Hand {
    store.tell_through(|store, world, found, said, by| {
        told(store, world, found, said, by, |_, _| true);
    });
    Hand::new(gateway)
}

/// What a worker of the cases states under `key`.
fn stated(key: &str) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        models: vec!["m".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// A worker the store holds a row for, in a store no holder of worker
/// objects stands over.
pub(crate) fn worker(store: &MemoryStore, key: &str) -> WorkerId {
    store.upsert(stated(key)).id()
}

pub(crate) fn a_dispatched(lease: u64) -> proto::Dispatched {
    proto::Dispatched {
        lease_id: lease,
        target_index: 0,
        side: proto::Side::Single.into(),
        since_decision_micros: 250,
    }
}

pub(crate) fn a_completed(lease: u64, served: WorkerId) -> proto::Completed {
    proto::Completed {
        lease_id: lease,
        side: proto::Side::Single.into(),
        served: served.0,
        status: 200,
        class: Class::Ok.into(),
        ttft_micros: 1_000,
        tokens_in: 12,
        tokens_out: 34,
        duration_micros: 5_000,
    }
}

pub(crate) fn a_released(lease: u64) -> proto::Released {
    proto::Released {
        what: Some(What::LeaseId(lease)),
        reason: Reason::ClientGone.into(),
        side: proto::Side::Unspecified.into(),
    }
}

fn state(store: &MemoryStore, gateway: GatewayId, lease: u64) -> Option<LeaseState> {
    LeasesStore::get(store, gateway, LeaseId(lease)).map(|record| record.state)
}

/// What the engine told a stage of leases: the call, the lease's id and
/// the worker the call named. A release names no worker: a pair's is
/// kept by its side.
type Heard = Vec<(&'static str, u64, WorkerId)>;

/// The index each dispatch's lease named and the rank of the target it
/// was told on, as a stage reads them off what it is handed.
type Indexes = Arc<Mutex<Vec<(u8, u16)>>>;

/// The `random` picker, keeping what the engine tells it of leases.
#[derive(Debug)]
struct Hears(Arc<Mutex<Heard>>, Indexes);

impl Lifecycle for Hears {
    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        self.0.lock().push(("dispatched", lease.id.0, target.id));
        self.1.lock().push((lease.target_index, target.dp_rank));
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.0
            .lock()
            .push(("completed", lease.id.0, outcome.served));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        let call = match lease.side {
            Side::Prefill => "released prefill",
            Side::Decode => "released decode",
            _ => "released",
        };
        self.0.lock().push((call, lease.id.0, WorkerId(0)));
        if let Some(step) = AT_RELEASE.take() {
            step();
        }
        assert_ne!(lease.id.0, BURSTS, "the stage's call panics");
    }

    fn hears(&self) -> EventSet {
        EventSet::of([
            EventKind::Dispatched,
            EventKind::Completed,
            EventKind::Released,
        ])
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// Applies `generation` of a document whose picker keeps what it hears,
/// and hands back what that generation's stage heard.
fn hearing(store: &MemoryStore, generation: u64) -> Arc<Mutex<Heard>> {
    hearing_indexes(store, generation).0
}

/// [`hearing`], with the index each dispatch's lease named and the rank
/// it was told.
fn hearing_indexes(store: &MemoryStore, generation: u64) -> (Arc<Mutex<Heard>>, Indexes) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let indexes = Indexes::default();
    let kept = (Arc::clone(&heard), Arc::clone(&indexes));
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || {
            Box::new(Hears(Arc::clone(&kept.0), Arc::clone(&kept.1)))
        })
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = generation;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    store.apply(spec).expect("a later generation");
    (heard, indexes)
}

/// Files the lease `id` of [`A`] as `generation` issued it.
fn issued(store: &MemoryStore, id: u64, generation: u64, now: Instant) {
    let mut record = lease(A, id, now);
    record.lease.generation = generation;
    store.issue(record).expect("a lease");
}

#[test]
fn a_frame_tells_the_engine_that_issued_its_lease() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let heard = hearing(&store, 1);
    let served = worker(&store, "http://a:8000");
    for id in 1..=2 {
        issued(&store, id, 1, t0);
    }
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    // The dispatch names the worker the lease books.
    assert_eq!(*heard.lock(), [("dispatched", 1, WorkerId(7))]);
    completed(&store, &open(&store, A), &a_completed(1, served)).expect("the frame reads");
    released(&store, &open(&store, A), &a_released(2)).expect("the frame reads");
    assert_eq!(
        heard.lock()[1..],
        [("completed", 1, served), ("released", 2, WorkerId(0))]
    );
}

/// A stage's call that panics is held where the engine is told,
/// whoever applies the frame: here the thread that hands it, as a
/// session does with no decider at work. The lease left the file
/// ahead of the call, the fault is counted, and the next frame is
/// told.
#[test]
fn a_stages_call_that_panics_is_held_and_the_next_frame_is_told() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let heard = hearing(&store, 1);
    for id in [BURSTS, 2] {
        issued(&store, id, 1, t0);
    }
    let hand = open(&store, A);
    released(&store, &hand, &a_released(BURSTS)).expect("the frame reads");
    assert_eq!(state(&store, A, BURSTS), None);
    assert_eq!(store.lifecycle_faults(), 1);
    released(&store, &hand, &a_released(2)).expect("the frame reads");
    let told = [BURSTS, 2].map(|lease| ("released", lease, WorkerId(0)));
    assert_eq!(*heard.lock(), told);
    assert_eq!(store.lifecycle_faults(), 1);
    assert_eq!(store.live(A), 0);
}

#[test]
fn a_lease_is_told_dispatched_once_and_nothing_once_it_left_the_file() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let heard = hearing(&store, 1);
    issued(&store, 1, 1, t0);
    for _ in 0..2 {
        dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    }
    assert_eq!(heard.lock().len(), 1, "a second dispatch of one lease");
    // A lease that was never on file, and one that left it.
    dispatched(&store, &open(&store, A), &a_dispatched(2), t0).expect("no error");
    released(&store, &open(&store, A), &a_released(2)).expect("no error");
    released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
    released(&store, &open(&store, A), &a_released(1)).expect("no error");
    assert_eq!(heard.lock()[1..], [("released", 1, WorkerId(0))]);
}

/// A pair's lease is a record with a decode side beside it: the engine
/// is told of each side once, by that side's own frame and on that
/// side's worker.
#[test]
fn a_pairs_frames_tell_the_engine_once_a_side_on_that_sides_worker() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["p", "d0", "d1"].map(|key| worker(&store, key));
    let heard = hearing(&store, 1);
    let now = Instant::now();
    let mut targets = Targets::new();
    for decode_id in [d0, d1] {
        let pair = Pair {
            decode_id,
            ..Pair::default()
        };
        targets.push(Target {
            id: p,
            pair: Some(pair),
            ..Target::default()
        });
    }
    let side = DecodeSide {
        target: d0,
        picked: PickRecord::of_decodes(&targets),
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        state: LeaseState::ISSUED,
        tokens: 0,
        target_index: 0,
    };
    let pair = |id: u64| LeaseRecord {
        lease: Lease {
            id: LeaseId(id),
            side: Side::Prefill,
            generation: 1,
            ..Lease::default()
        },
        gateway: A,
        request: id,
        target: p,
        picked: PickRecord::of(&targets),
        decode: None,
        untold: false,
        term: 0,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(5),
        state: LeaseState::ISSUED,
    };
    LeasesStore::issue_pair(&store, pair(1), side).expect("the key is free");
    let frame = |side: proto::Side, index: u32| proto::Dispatched {
        side: side.into(),
        target_index: index,
        ..a_dispatched(1)
    };
    // The decode's dispatch on the alternate is told on that decode,
    // the prefill's on the prefill. A second dispatch of a side and a
    // frame on a side the pair does not hold tell nothing.
    for (side, index) in [
        (proto::Side::Decode, 1),
        (proto::Side::Prefill, 0),
        (proto::Side::Decode, 0),
        (proto::Side::Single, 0),
    ] {
        dispatched(&store, &open(&store, A), &frame(side, index), now).expect("the frame reads");
    }
    assert_eq!(*heard.lock(), [("dispatched", 1, d1), ("dispatched", 1, p)]);
    // A side's completion is told once, and the other side stands.
    let ended = |side: proto::Side, served: WorkerId| proto::Completed {
        side: side.into(),
        ..a_completed(1, served)
    };
    for _ in 0..2 {
        completed(&store, &open(&store, A), &ended(proto::Side::Prefill, p))
            .expect("the frame reads");
    }
    assert_eq!(heard.lock()[2..], [("completed", 1, p)]);
    // A release with no side ends what stands: the decode alone.
    released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
    assert_eq!(heard.lock()[3..], [("released decode", 1, WorkerId(0))]);
    assert_eq!(LeasesStore::get(&store, A, LeaseId(1)), None);
    // A pair given up whole is told once a side.
    LeasesStore::issue_pair(&store, pair(2), side).expect("the key is free");
    released(&store, &open(&store, A), &a_released(2)).expect("the frame reads");
    let whole = ["released prefill", "released decode"].map(|call| (call, 2, WorkerId(0)));
    assert_eq!(heard.lock()[4..], whole);
}

#[test]
fn a_lease_of_a_generation_that_left_tells_no_engine() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let first = hearing(&store, 1);
    issued(&store, 1, 1, t0);
    // A later generation's stages never saw the issue.
    let second = hearing(&store, 2);
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
    assert!(first.lock().is_empty() && second.lock().is_empty());
    assert_eq!(
        state(&store, A, 1),
        None,
        "the file is written all the same"
    );
}

/// Files the pair `id` of [`A`] as `generation` issued it: the prefill
/// side on `prefill`, and the decode side on `decode`, standing as
/// `stands`.
fn issued_pair(
    store: &MemoryStore,
    (id, generation): (u64, u64),
    (prefill, decode): (WorkerId, WorkerId),
    stands: LeaseState,
    now: Instant,
) {
    let mut record = lease(A, id, now);
    (record.lease.side, record.lease.generation) = (Side::Prefill, generation);
    record.target = prefill;
    let side = DecodeSide {
        target: decode,
        picked: PickRecord::default(),
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        state: stands,
        tokens: 0,
        target_index: 0,
    };
    LeasesStore::issue_pair(store, record, side).expect("the key is free");
}

/// A pair's words reach the stages of the generation that issued its
/// lease and of no later one, as a single's do: no dispatch and no end
/// of a side, also of a decode side its dispatch issues.
#[test]
fn a_pairs_lease_of_a_generation_that_left_tells_no_engine() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let first = hearing(&store, 1);
    let workers = (WorkerId(7), WorkerId(8));
    issued_pair(&store, (1, 1), workers, LeaseState::ISSUED, t0);
    issued_pair(&store, (2, 1), workers, LeaseState::Chosen, t0);
    // A later generation's stages never saw the issue.
    let second = hearing(&store, 2);
    for id in [1, 2] {
        let sent = proto::Dispatched {
            side: proto::Side::Decode.into(),
            ..a_dispatched(id)
        };
        dispatched(&store, &open(&store, A), &sent, t0).expect("the frame reads");
        let ended = proto::Completed {
            side: proto::Side::Decode.into(),
            ..a_completed(id, workers.1)
        };
        completed(&store, &open(&store, A), &ended).expect("the frame reads");
        released(&store, &open(&store, A), &a_released(id)).expect("the frame reads");
    }
    assert!(first.lock().is_empty() && second.lock().is_empty());
    assert_eq!(
        state(&store, A, 1),
        None,
        "the file is written all the same"
    );
}

/// Two frames of one take are handed one world, the take's: a
/// generation that starts inside the first frame's call starts behind
/// that load. The second frame is told by the generation that stands
/// at its word all the same, and so is the next take's frame. The
/// stage that left with the generation before hears neither, and the
/// stage built for the later one never saw the issues.
#[test]
fn a_takes_second_frame_is_told_by_the_generation_that_started_inside_its_first() {
    let store = Arc::new(MemoryStore::new());
    let t0 = Instant::now();
    let first = hearing(&store, 1);
    for id in 1..=3 {
        issued(&store, id, 1, t0);
    }
    let hand = open(&store, A);
    // The first release's call applies generation 2 before it returns.
    let second = Arc::new(Mutex::new(None));
    let (applies, kept) = (Arc::clone(&store), Arc::clone(&second));
    AT_RELEASE.set(Some(Box::new(move || {
        *kept.lock() = Some(hearing(&applies, 2));
    })));
    let at_work = store.at_work();
    for id in 1..=2 {
        released(&store, &hand, &a_released(id)).expect("the frame reads");
    }
    at_work.apply();
    // The first frame's word was under way when the generation started.
    let told = [("released", 1, WorkerId(0))];
    assert_eq!(*first.lock(), told);
    released(&store, &hand, &a_released(3)).expect("the frame reads");
    at_work.apply();
    let second = second.lock().take();
    let second = second.expect("the first release applied generation 2");
    assert_eq!((first.lock().len(), second.lock().len()), (1, 0));
    assert_eq!(
        state(&store, A, 2),
        None,
        "the file is written all the same"
    );
}

#[test]
fn a_dispatch_that_names_no_target_of_the_pick_is_not_told() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let heard = hearing(&store, 1);
    issued(&store, 1, 1, t0);
    let mut second_target = a_dispatched(1);
    second_target.target_index = 1;
    dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
    assert!(heard.lock().is_empty());
    assert_eq!(state(&store, A, 1), Some(LeaseState::Dispatched { at: t0 }));
}

/// A pick of the workers 7 and 9, the second at rank 2.
fn two_targets() -> PickRecord {
    let mut targets = Targets::new();
    for (id, dp_rank) in [(7, 0), (9, 2)] {
        targets.push(Target {
            id: WorkerId(id),
            dp_rank,
            ..Target::default()
        });
    }
    PickRecord::of(&targets)
}

#[test]
fn a_dispatch_on_another_target_of_the_pick_is_told_on_that_target() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let (heard, indexes) = hearing_indexes(&store, 1);
    let session = Session::from_raw(41);
    let mut record = lease(A, 1, t0);
    record.lease.generation = 1;
    record.lease.session = session;
    record.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
    record.picked = two_targets();
    store.issue(record).expect("a lease");
    let mut second_target = a_dispatched(1);
    second_target.target_index = 1;
    dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
    assert_eq!(*heard.lock(), [("dispatched", 1, WorkerId(9))]);
    // The lease told names the dispatched index, which the door wrote
    // after the record was read: a stage that reads the index off the
    // lease is told the second target, at that target's rank.
    assert_eq!(*indexes.lock(), [(1, 2)]);
    // The key's pin follows the dispatch, and so does the lease.
    let pins = store.with_snapshot(|world| world.spec().engine().sessions().pins(session));
    assert_eq!(pins, Some([WorkerId(9), WorkerId(0)]));
    let filed = LeasesStore::get(&store, A, LeaseId(1)).expect("on file");
    assert_eq!((filed.target, filed.lease.target_index), (WorkerId(9), 1));
    assert_eq!(filed.target_at(1), Some((WorkerId(9), 2)));
}

/// The worker the bucket stage leads a request of `tokens` with, beside
/// the two workers of its view as a first pick ordered them, after a
/// lease of that size was issued on that pick's first target and its
/// gateway dispatched on the pick's second: the other worker, or
/// `another_rank` of the first target's own.
fn led_after_a_dispatch_on_the_second_target(
    tokens: u32,
    another_rank: bool,
) -> (WorkerId, [WorkerId; 2]) {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut document = PolicySpec::preset("bucket").expect("a preset of this build");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("a later generation");
    // Two workers of one model, as a view shows them, and a prefill
    // request decided by the standing engine.
    let view: Vec<Candidate> = [7, 9]
        .iter()
        .zip(0..)
        .map(|(&id, position)| Candidate {
            position,
            id: WorkerId(id),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect();
    let mut scratch = Scratch::new();
    let mut decide = |seed: u64| {
        let facts = RequestFacts {
            model: "m",
            phase: Phase::Prefill,
            prompt_tokens: tokens,
            seed,
            now_micros: 1_000_000,
            ..RequestFacts::default()
        };
        let pool = PoolFacts {
            pool_size: 2,
            generation: 1,
            ..PoolFacts::default()
        };
        store.with_snapshot(|world| {
            let engine = world.spec().engine();
            engine.decide(&facts, &pool, &view, &Constraints::default(), &mut scratch)
        })
    };
    let decision = decide(1);
    let pick = decision.pick().expect("a pick");
    let [owner, other] = [0, 1].map(|at| pick.targets.as_slice()[at].id);
    assert_ne!(owner, other);
    // What the record keeps of the pick: its second target as the
    // engine listed it, or the first target's worker again at its
    // second rank, as a pick over a worker of two ranks lists it.
    let mut targets = pick.targets;
    if another_rank {
        targets = Targets::new();
        for dp_rank in 0..2 {
            targets.push(Target {
                id: owner,
                dp_rank,
                ..Target::default()
            });
        }
    }
    // The lease is issued on the pick's first target, which its record
    // names, and the engine that decided is told: the stage books that
    // worker. The lease on file names no booked worker: the store names
    // it from the record where it tells of the dispatch.
    let mut record = lease(A, 1, t0);
    record.lease.generation = 1;
    record.lease.tokens = tokens;
    record.lease.issued_micros = 1_000_000;
    record.target = owner;
    record.picked = PickRecord::of(&targets);
    let issued = record.lease;
    store.with_snapshot(|world| {
        let engine = world.spec().engine();
        engine.on_decision_issued(&issued, &decision);
    });
    store.issue(record).expect("a lease");
    // Its gateway dispatches on the pick's second target.
    let mut second_target = a_dispatched(1);
    second_target.target_index = 1;
    dispatched(&store, &open(&store, A), &second_target, t0).expect("the frame reads");
    let next = decide(2);
    let led = next.pick().expect("a pick").targets.as_slice()[0].id;
    (led, [owner, other])
}

/// A stage that books at the issue books the worker that serves: the
/// bucket stage hears a `Dispatched` frame that names its pick's second
/// target and books the lease's tokens on that worker too.
#[test]
fn a_dispatch_on_another_target_is_booked_by_a_stage_that_books_at_the_issue() {
    // Both workers carry the lease's tokens: the next request of that
    // size goes to its range's owner again. With the first target alone
    // booked, the balance rule would hand it to the other.
    let (led, [owner, _]) = led_after_a_dispatch_on_the_second_target(3_000, false);
    assert_eq!(led, owner);
}

/// A dispatch that names the worker the issue booked books nothing
/// more: the frame names another rank of the lease's own worker.
#[test]
fn a_dispatch_on_another_rank_of_the_leases_worker_is_booked_once() {
    // Booked once, the worker carries 50 tokens, under the balance
    // rule's 64: the next request of that size goes to its range's
    // owner again. Booked twice it would carry 100, and the rule would
    // hand the request to the other worker.
    let (led, [owner, _]) = led_after_a_dispatch_on_the_second_target(50, true);
    assert_eq!(led, owner);
}

#[test]
fn a_dispatched_frame_books_a_keyed_leases_pin_in_its_engines_table() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let _heard = hearing(&store, 1);
    let session = Session::from_raw(41);
    let pins = || store.with_snapshot(|world| world.spec().engine().sessions().pins(session));
    let mut record = lease(A, 1, t0);
    record.lease.generation = 1;
    record.lease.session = session;
    record.lease.session_verdict = SessionVerdict::Claimed(WorkerId(7));
    store.issue(record).expect("a lease");
    assert_eq!(pins(), None, "an issue books no pin");
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(pins(), Some([WorkerId(7), WorkerId(0)]));
}

/// Files the keyed lease `id` of [`A`] on `target`, as generation 1
/// issued it under `verdict`.
fn keyed(store: &MemoryStore, id: u64, target: u64, verdict: SessionVerdict, now: Instant) {
    let mut record = lease(A, id, now);
    record.lease.generation = 1;
    record.lease.session = KEY;
    record.lease.session_verdict = verdict;
    record.target = WorkerId(target);
    store.issue(record).expect("a lease");
}

/// A lease whose generation left tells its dispatch to the session
/// table, which the generation that stands reads though it kept none
/// of that generation's stages: the key is pinned there on the worker
/// the request went to. The table is the store's engine's, one for
/// every generation. The lease here is filed by hand, so no decision
/// claimed its key: the dispatch's pin is the key's first.
#[test]
fn a_dispatch_of_a_lease_whose_generation_left_pins_its_key_in_the_standing_table() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let first = hearing(&store, 1);
    keyed(&store, 1, 7, SessionVerdict::Claimed(WorkerId(7)), t0);
    let second = hearing(&store, 2);
    let pins = || store.with_snapshot(|world| world.spec().engine().sessions().pins(KEY));
    assert_eq!(pins(), None, "no decision claimed the key");
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(pins(), Some([WorkerId(7), WorkerId(0)]));
    assert!(first.lock().is_empty() && second.lock().is_empty());
}

#[test]
fn a_dispatched_frame_moves_its_gateways_issued_lease_to_dispatched() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.issue(lease(A, 1, t0)).expect("a lease");
    let at = t0 + Duration::from_millis(3);
    dispatched(&store, &open(&store, A), &a_dispatched(1), at).expect("the frame reads");
    assert_eq!(state(&store, A, 1), Some(LeaseState::Dispatched { at }));
}

#[test]
fn a_completed_frame_ends_the_lease() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let served = worker(&store, "http://a:8000");
    store.issue(lease(A, 1, t0)).expect("a lease");
    completed(&store, &open(&store, A), &a_completed(1, served)).expect("the frame reads");
    assert_eq!(state(&store, A, 1), None, "the lease ended");
}

#[test]
fn a_released_frame_ends_the_lease() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.issue(lease(A, 1, t0)).expect("a lease");
    released(&store, &open(&store, A), &a_released(1)).expect("the frame reads");
    assert_eq!(state(&store, A, 1), None);
}

#[test]
fn a_lease_not_on_file_is_no_error_and_changes_nothing() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.issue(lease(A, 1, t0)).expect("a lease");
    dispatched(&store, &open(&store, A), &a_dispatched(2), t0).expect("no error");
    completed(&store, &open(&store, A), &a_completed(2, WorkerId(7))).expect("no error");
    released(&store, &open(&store, A), &a_released(2)).expect("no error");
    assert_eq!(state(&store, A, 1), Some(LeaseState::ISSUED));
    assert_eq!(store.live(A), 1);
}

/// A ticket this build never issued is a ticket not on file: no error
/// and nothing written, a lease under the ticket's number included.
/// The frame is read whole all the same.
#[test]
fn a_released_ticket_is_no_error_and_writes_nothing() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.issue(lease(A, 1, t0)).expect("a lease");
    let ticket = proto::Released {
        what: Some(What::Ticket(1)),
        ..a_released(1)
    };
    assert_eq!(
        release(&ticket).expect("the frame reads"),
        (None, None, ReleaseReason::ClientGone)
    );
    released(&store, &open(&store, A), &ticket).expect("no error");
    assert_eq!(state(&store, A, 1), Some(LeaseState::ISSUED));

    let no_reason = proto::Released {
        reason: 0,
        ..ticket
    };
    let status =
        released(&store, &open(&store, A), &no_reason).expect_err("the frame is read whole");
    assert_eq!(
        (status.code(), status.message()),
        (
            Code::InvalidArgument,
            "Released.reason is 0, which names none this version knows"
        )
    );
}

#[test]
fn a_gateway_reaches_no_lease_of_another() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.issue(lease(B, 1, t0)).expect("a lease");
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("no error");
    completed(&store, &open(&store, A), &a_completed(1, WorkerId(7))).expect("no error");
    released(&store, &open(&store, A), &a_released(1)).expect("no error");
    assert_eq!(state(&store, B, 1), Some(LeaseState::ISSUED));
}

#[test]
fn a_frame_reads_whole_into_what_the_store_is_handed() {
    let read = dispatch(&proto::Dispatched {
        lease_id: 5,
        target_index: 255,
        side: proto::Side::Decode.into(),
        since_decision_micros: 250,
    })
    .expect("the frame reads");
    assert_eq!(
        read,
        (
            LeaseId(5),
            Dispatch {
                target_index: 255,
                side: Side::Decode,
                since_decision_micros: 250,
            }
        )
    );
    let read = outcome(&proto::Completed {
        lease_id: 6,
        side: proto::Side::Prefill.into(),
        served: 7,
        status: 65_535,
        class: Class::Timeout.into(),
        ttft_micros: 1,
        tokens_in: 2,
        tokens_out: 3,
        duration_micros: 4,
    })
    .expect("the frame reads");
    assert_eq!(
        read,
        (
            LeaseId(6),
            Outcome {
                side: Side::Prefill,
                served: WorkerId(7),
                status: 65_535,
                class: TransportClass::Timeout,
                ttft_micros: 1,
                tokens_in: 2,
                tokens_out: 3,
                duration_micros: 4,
            }
        )
    );
    // A release of no side names every side; a named side, its own.
    assert_eq!(
        release(&a_released(8)).expect("the frame reads"),
        (Some(LeaseId(8)), None, ReleaseReason::ClientGone)
    );
    let one = proto::Released {
        side: proto::Side::Prefill.into(),
        reason: Reason::Expired.into(),
        ..a_released(8)
    };
    assert_eq!(
        release(&one).expect("the frame reads"),
        (
            Some(LeaseId(8)),
            Some(Side::Prefill),
            ReleaseReason::Expired
        )
    );
}

#[test]
fn every_side_class_and_reason_reads_as_its_own() {
    for (wire, read) in [
        (proto::Side::Single, Side::Single),
        (proto::Side::Prefill, Side::Prefill),
        (proto::Side::Decode, Side::Decode),
        (proto::Side::Encode, Side::Encode),
    ] {
        assert_eq!(side("a side", wire.into()).expect("a side"), read);
    }
    for (wire, read) in [
        (Class::Ok, TransportClass::Ok),
        (Class::ClientError, TransportClass::ClientError),
        (Class::ServerError, TransportClass::ServerError),
        (Class::ConnectFailed, TransportClass::ConnectFailed),
        (Class::Timeout, TransportClass::Timeout),
        (Class::Cancelled, TransportClass::Cancelled),
    ] {
        assert_eq!(class(wire.into()).expect("a class"), read);
    }
    for (wire, read) in [
        (Reason::Abandoned, ReleaseReason::Abandoned),
        (Reason::LateDecision, ReleaseReason::LateDecision),
        (Reason::ClientGone, ReleaseReason::ClientGone),
        (Reason::CommitFailed, ReleaseReason::CommitFailed),
        (Reason::Expired, ReleaseReason::Expired),
        (Reason::WorkerRemoved, ReleaseReason::WorkerRemoved),
    ] {
        assert_eq!(reason(wire.into()).expect("a reason"), read);
    }
}

#[test]
fn a_frame_that_reads_wrong_is_refused_by_its_fields_name_and_writes_nothing() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let served = worker(&store, "http://a:8000");
    store.issue(lease(A, 1, t0)).expect("a lease");
    let refused = |result: Result<(), Status>, code: Code, message: &str| {
        let status = result.expect_err("the frame is refused");
        assert_eq!((status.code(), status.message()), (code, message));
    };
    let none = |field: &str, number: i32| {
        format!("{field} is {number}, which names none this version knows")
    };
    for number in [0, 99] {
        let frame = proto::Dispatched {
            side: number,
            ..a_dispatched(1)
        };
        refused(
            dispatched(&store, &open(&store, A), &frame, t0),
            Code::InvalidArgument,
            &none("Dispatched.side", number),
        );
        let frame = proto::Completed {
            side: number,
            ..a_completed(1, served)
        };
        refused(
            completed(&store, &open(&store, A), &frame),
            Code::InvalidArgument,
            &none("Completed.side", number),
        );
        let frame = proto::Completed {
            class: number,
            ..a_completed(1, served)
        };
        refused(
            completed(&store, &open(&store, A), &frame),
            Code::InvalidArgument,
            &none("Completed.class", number),
        );
        let frame = proto::Released {
            reason: number,
            ..a_released(1)
        };
        refused(
            released(&store, &open(&store, A), &frame),
            Code::InvalidArgument,
            &none("Released.reason", number),
        );
    }
    let frame = proto::Released {
        side: 99,
        ..a_released(1)
    };
    refused(
        released(&store, &open(&store, A), &frame),
        Code::InvalidArgument,
        &none("Released.side", 99),
    );
    let frame = proto::Dispatched {
        target_index: 256,
        ..a_dispatched(1)
    };
    refused(
        dispatched(&store, &open(&store, A), &frame, t0),
        Code::InvalidArgument,
        "Dispatched.target_index is 256: 255 is the most it holds",
    );
    let frame = proto::Completed {
        status: 65_536,
        ..a_completed(1, served)
    };
    refused(
        completed(&store, &open(&store, A), &frame),
        Code::InvalidArgument,
        "Completed.status is 65536: 65535 is the most it holds",
    );
    let frame = proto::Released {
        what: None,
        ..a_released(1)
    };
    refused(
        released(&store, &open(&store, A), &frame),
        Code::InvalidArgument,
        "Released.what names neither a lease nor a ticket",
    );
    // Nothing was written: the lease stands as issued.
    assert_eq!(state(&store, A, 1), Some(LeaseState::ISSUED));
}

/// The model of the cases of a profile's count of sends.
const COUNTED: &str = "counted";

/// `generation` of the least-load preset, whose scorer reads its
/// profile's count of sends.
fn counted(generation: u64) -> AppliedSpec {
    counted_under(generation, None)
}

/// [`counted`] with `limit` as its profile's time limit on a send. A
/// document that states another limit than the standing one builds the
/// profile and its count anew.
fn counted_under(generation: u64, limit: Option<u32>) -> AppliedSpec {
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.generation = generation;
    let profile = document.profiles.get_mut("least_load");
    profile.expect("the preset's profile").sent_ttl_ms = limit;
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, &StageRegistry::builtin(), &[])
        .expect("a preset of this build validates")
}

/// Applies [`counted`] of `generation` and hands back that generation's
/// count of sends.
fn counting(store: &MemoryStore, generation: u64) -> Arc<DispatchCredit> {
    counting_under(store, generation, None)
}

/// [`counting`] for [`counted_under`].
fn counting_under(store: &MemoryStore, generation: u64, limit: Option<u32>) -> Arc<DispatchCredit> {
    store
        .apply(counted_under(generation, limit))
        .expect("a later generation");
    store.deciding(|world| Arc::clone(&world.spec().engine().sent_credits[0]))
}

/// Files the lease `id` of [`A`] on `target`, as `generation` issued it.
fn issued_on(store: &MemoryStore, id: u64, generation: u64, target: WorkerId, now: Instant) {
    let mut record = lease(A, id, now);
    record.lease.generation = generation;
    record.target = target;
    store.issue(record).expect("a lease");
}

/// The sends `count` holds on the first rank of `worker`, read under no
/// report and at the clock's start: none is ended or aged.
fn sends(count: &DispatchCredit, worker: WorkerId) -> u32 {
    let row = Candidate {
        id: worker,
        ..Candidate::default()
    };
    count.read(&RequestFacts::default(), &row).requests
}

/// A dispatch enters the count of sends of the generation that stands,
/// whichever generation issued its lease: the store tells the standing
/// engine of every send, and a generation that left counts no more.
/// The standing generation states another limit, so its count is built
/// new.
#[test]
fn a_dispatch_of_an_older_generations_lease_enters_the_standing_generations_sends() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let first = counting(&store, 1);
    issued_on(&store, 1, 1, served, t0);
    let standing = counting_under(&store, 2, Some(60_000));
    assert!(!Arc::ptr_eq(&first, &standing), "a count built new");
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(sends(&standing, served), 1);
    assert_eq!(sends(&first, served), 0, "the generation that left");
}

/// A generation that states the profile as it stood keeps the
/// profile's count of sends: the standing count is the one object, with
/// the sends it holds. A dispatch of an older generation's lease is
/// counted in it once.
#[test]
fn an_apply_keeps_the_count_of_an_unchanged_profile_with_its_sends() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let first = counting(&store, 1);
    for id in 1..=2 {
        issued_on(&store, id, 1, served, t0);
    }
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    let kept = counting(&store, 2);
    assert!(Arc::ptr_eq(&first, &kept), "the count stands as one object");
    assert_eq!(sends(&kept, served), 1);
    dispatched(&store, &open(&store, A), &a_dispatched(2), t0).expect("the frame reads");
    assert_eq!(sends(&kept, served), 2);
}

/// Applies `generation` of the least-load preset under the picker that
/// keeps what it hears: that generation's count of sends, and what its
/// stage heard.
fn counting_and_hearing(
    store: &MemoryStore,
    generation: u64,
) -> (Arc<DispatchCredit>, Arc<Mutex<Heard>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    let stage =
        move || -> Box<dyn Picker> { Box::new(Hears(Arc::clone(&kept), Indexes::default())) };
    catalog.picker("hears", stage).expect("a free name");
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.generation = generation;
    let profile = document.profiles.get_mut("least_load");
    profile.expect("the preset's profile").pick = "hears".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    store.apply(spec).expect("a later generation");
    let count = store.deciding(|world| Arc::clone(&world.spec().engine().sent_credits[0]));
    (count, heard)
}

/// A lease of a generation above the one in force tells no stage and
/// moves no pin: no stage of this replica booked it. Its dispatch is a
/// send of the host's all the same, and the counts of sends in force
/// take it. No decision files such a lease on one replica: a decision
/// reads its generation off a published world, and the generations
/// only rise. The case files one by hand.
#[test]
fn a_lease_above_the_generation_in_force_tells_no_stage_and_its_send_is_counted() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let (count, heard) = counting_and_hearing(&store, 1);
    let mut record = lease(A, 1, t0);
    record.lease.generation = 2;
    record.lease.session = KEY;
    record.lease.session_verdict = SessionVerdict::Claimed(served);
    record.target = served;
    store.issue(record).expect("a lease");
    let hand = open(&store, A);
    dispatched(&store, &hand, &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(sends(&count, served), 1, "a send of the host's");
    completed(&store, &hand, &a_completed(1, served)).expect("the frame reads");
    assert_eq!(state(&store, A, 1), None, "the lease ended on file");
    assert!(heard.lock().is_empty(), "no stage booked the lease");
    let pins = store.with_snapshot(|world| world.spec().engine().sessions().pins(KEY));
    assert_eq!(pins, None, "and the session table took no pin");
    assert_eq!(store.lifecycle_faults(), 0);
}

/// Two sends said inside an apply, between the engine's turn and the
/// publication of the generation's world: each enters the count of the
/// generation the standing world holds. `apply` puts the second
/// generation in force and hands back its count. The answer is what
/// that count reads, and then what the count before reads, on a worker
/// sent to before the apply and on one first sent to inside it. No
/// word of a lease takes a send out of a count, so the ends said
/// behind the publication leave both counts as they stood.
fn said_inside_an_apply(apply: impl FnOnce(&MemoryStore) -> Arc<DispatchCredit>) -> [[u32; 2]; 2] {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let known = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let fresh = ready(
        &store,
        &objects,
        "b",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let first = counting(&store, 1);
    for (id, target) in [(1, known), (2, known), (3, fresh)] {
        issued_on(&store, id, 1, target, t0);
    }
    // A send before the apply: the count holds an entry of the rank.
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    let inside = Arc::clone(&store);
    seam::on_turn(move || {
        let turned = inside.deciding(|world| !world.front().still_in_force());
        assert!(
            turned,
            "the engine holds the generation, the world does not"
        );
        for id in 2..=3 {
            let said = dispatched(&inside, &open(&inside, A), &a_dispatched(id), t0);
            said.expect("the frame reads");
        }
    });
    let standing = apply(&store);
    assert!(!Arc::ptr_eq(&first, &standing), "a count built new");
    let reads = |count: &DispatchCredit| [sends(count, known), sends(count, fresh)];
    let read = [reads(&standing), reads(&first)];
    for (id, served) in [(2, known), (3, fresh)] {
        completed(&store, &open(&store, A), &a_completed(id, served)).expect("the frame reads");
    }
    let behind_the_ends = [reads(&standing), reads(&first)];
    assert_eq!(behind_the_ends, read, "an end takes no send out");
    read
}

/// A count built new under the settings that stood took the entries of
/// the count before over at the engine's turn and shares them: it
/// reads a send said inside the apply to a rank that stood there, as
/// it reads the sends said before the apply. A rank first heard in the
/// window stands in the count before alone, which no decision reads
/// behind the publication: the new count is one send short on it.
#[test]
fn a_send_said_inside_an_apply_is_read_by_the_count_built_for_its_generation() {
    let read = said_inside_an_apply(|store| counting_and_hearing(store, 2).0);
    assert_eq!(read, [[2, 0], [2, 1]]);
}

/// A count built new under settings no standing count has shares no
/// entry with the count before: at the engine's turn it took a copy of
/// the sends that count held. It holds the send said before the apply
/// and neither of the two said inside it, on a rank that stood as on a
/// rank first heard there: the count before holds those alone.
#[test]
fn a_send_said_inside_an_apply_is_in_no_copy_of_a_count_under_other_settings() {
    let read = said_inside_an_apply(|store| counting_under(store, 2, Some(60_000)));
    assert_eq!(read, [[1, 0], [2, 1]]);
}

/// [`counted`] with a second profile `late` of the same stages, bound
/// to a model no case asks for. `limits` are the milliseconds a send
/// counts for under the preset's profile and under `late`: two limits,
/// so each profile reads a count of its own.
fn counted_twice(generation: u64, limits: [u32; 2]) -> AppliedSpec {
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.generation = generation;
    let mut late = document.profiles["least_load"].clone();
    late.sent_ttl_ms = Some(limits[1]);
    document.profiles.insert("late".into(), late);
    let stock = document.profiles.get_mut("least_load");
    stock.expect("the preset's profile").sent_ttl_ms = Some(limits[0]);
    let elsewhere = r#"{"match":{"model":"elsewhere"},"profile":"late"}"#;
    let elsewhere = serde_json::from_str(elsewhere).expect("a binding");
    document.bindings.insert(0, elsewhere);
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, &StageRegistry::builtin(), &[])
        .expect("the document validates")
}

/// The sends the standing generation counts on the first rank of
/// `worker` under the time limit `limit_ms`, as a decision at
/// `now_micros` reads them.
fn sends_under(store: &MemoryStore, limit_ms: u32, worker: WorkerId, now_micros: u64) -> u32 {
    let row = Candidate {
        id: worker,
        ..Candidate::default()
    };
    let facts = RequestFacts {
        now_micros,
        ..RequestFacts::default()
    };
    store.deciding(|world| {
        let mut counts = world.spec().engine().sent_credits.iter();
        let count = counts.find(|count| count.settings.ttl_ms == Some(limit_ms));
        let count = count.expect("a count under the limit");
        count.read(&facts, &row).requests
    })
}

/// A count of sends a generation builds new holds the sends its own
/// time limit still counts, whichever standing count let them go. The
/// first generation's two profiles count a send for one millisecond
/// and for nine. Two milliseconds behind a send, the count under one
/// has let it go and the count under nine holds it. The second
/// generation writes nine for the one and a longer limit for the nine:
/// the send is inside both, and both of its counts hold it.
#[test]
fn a_generation_under_other_limits_keeps_the_sends_those_limits_count() {
    const ISSUED: u64 = 91_300;
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let applied = store.apply(counted_twice(1, [1, 9]));
    applied.expect("a later generation");
    let mut record = lease(A, 1, t0);
    record.lease.generation = 1;
    record.lease.issued_micros = ISSUED;
    record.target = served;
    store.issue(record).expect("a lease");
    let frame = a_dispatched(1);
    dispatched(&store, &open(&store, A), &frame, t0).expect("the frame reads");
    let later = ISSUED + u64::from(frame.since_decision_micros) + 2_000;
    assert_eq!(sends_under(&store, 1, served, later), 0, "past one");
    assert_eq!(sends_under(&store, 9, served, later), 1, "inside nine");

    let applied = store.apply(counted_twice(2, [9, 43]));
    applied.expect("a later generation");
    let held = [9, 43].map(|limit| sends_under(&store, limit, served, later));
    assert_eq!(held, [1, 1], "under the limit that stood and a longer one");
}

/// The `random` picker, whose call at a dispatch panics.
#[derive(Debug)]
struct Bursts;

impl Lifecycle for Bursts {
    fn on_dispatched(&self, lease: &Lease, _: &Target, _: Side, _: u32) {
        assert_eq!(lease.id.0, 0, "the stage's call panics");
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Dispatched])
    }
}

impl Picker for Bursts {
    fn name(&self) -> &'static str {
        "bursts"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A single's dispatch is told in one order: the count of sends, the
/// session table, then the stages. A stage whose call panics at the
/// dispatch is held and counted as a fault, and the send is counted and
/// the session's pin moved all the same.
#[test]
fn a_stage_that_panics_at_a_dispatch_leaves_the_send_counted_and_the_pin_moved() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("bursts", || Box::new(Bursts))
        .expect("a free name");
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.generation = 1;
    let profile = document.profiles.get_mut("least_load");
    profile.expect("the preset's profile").pick = "bursts".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    store.apply(spec).expect("a later generation");
    let count = store.deciding(|world| Arc::clone(&world.spec().engine().sent_credits[0]));
    let mut record = lease(A, 1, t0);
    record.lease.generation = 1;
    record.lease.session = KEY;
    record.lease.session_verdict = SessionVerdict::Claimed(served);
    record.target = served;
    store.issue(record).expect("a lease");
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(store.lifecycle_faults(), 1, "the stage's call panicked");
    assert_eq!(sends(&count, served), 1);
    let pins = store.with_snapshot(|world| world.spec().engine().sessions().pins(KEY));
    assert_eq!(pins.map(|pins| pins[0]), Some(served));
}

/// A dispatch that comes late enters no rank of a worker that left: a
/// count asks the worker objects the store's word holds whether the
/// rank is listed, and the removal took the worker's object out
/// before the store was told. The lease reads dispatched all the
/// same, and a worker whose object stands is counted.
#[test]
fn a_late_dispatch_enters_no_rank_of_a_worker_that_left() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    tells(&store, &objects);
    let t0 = Instant::now();
    let left = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let stays = ready(
        &store,
        &objects,
        "b",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let count = counting(&store, 1);
    issued_on(&store, 1, 1, left, t0);
    issued_on(&store, 2, 1, stays, t0);
    let row = WorkersStore::get(&*store, left).expect("the row");
    let removed = objects.remove(&store, left, row.revision);
    assert!(removed.is_some(), "the row leaves");
    let hand = Hand::new(A);
    dispatched(&store, &hand, &a_dispatched(1), t0).expect("the frame reads");
    dispatched(&store, &hand, &a_dispatched(2), t0).expect("the frame reads");
    assert!(matches!(
        state(&store, A, 1),
        Some(LeaseState::Dispatched { .. })
    ));
    assert_eq!(sends(&count, left), 0);
    assert_eq!(sends(&count, stays), 1);
}

/// A pair's dispatch is counted as a single's is: on the side's own
/// worker, by the generation that stands, whichever issued the lease,
/// and at no rank of a worker that left. The standing generation states
/// another limit, so its count is built new.
#[test]
fn a_pairs_dispatch_is_counted_by_the_standing_generation_where_its_worker_stands() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    tells(&store, &objects);
    let t0 = Instant::now();
    let [prefill, stays, left] = ["p", "a", "b"].map(|key| {
        ready(
            &store,
            &objects,
            key,
            COUNTED,
            Role::Single,
            WireModes::HTTP,
        )
    });
    let first = counting(&store, 1);
    // Two pairs of the first generation, each on a decode of its own.
    for (id, decode) in [(1, stays), (2, left)] {
        issued_pair(&store, (id, 1), (prefill, decode), LeaseState::ISSUED, t0);
    }
    let standing = counting_under(&store, 2, Some(60_000));
    assert!(!Arc::ptr_eq(&first, &standing), "a count built new");
    let row = WorkersStore::get(&*store, left).expect("the row");
    let removed = objects.remove(&store, left, row.revision);
    assert!(removed.is_some(), "the row leaves");
    let hand = Hand::new(A);
    for id in [1, 2] {
        let frame = proto::Dispatched {
            side: proto::Side::Decode.into(),
            ..a_dispatched(id)
        };
        dispatched(&store, &hand, &frame, t0).expect("the frame reads");
    }
    assert_eq!(sends(&standing, stays), 1);
    assert_eq!(sends(&standing, left), 0);
    assert_eq!(sends(&standing, prefill), 0, "no prefill was sent");
    assert_eq!(sends(&first, stays), 0, "the generation that left");
}

/// A dispatch on a worker whose object stands is counted while the
/// store's world lists the worker not yet: a registration's object
/// stands from its write, and the world lists the worker at the build
/// its publication window ends with. The decision that sent the
/// request read the worker objects, and so does its count.
#[test]
fn a_dispatch_inside_a_registrations_window_is_counted() {
    let store = Arc::new(MemoryStore::with_publish_window(Duration::from_secs(3600)));
    let objects = holder(&store);
    tells(&store, &objects);
    let t0 = Instant::now();
    let count = counting(&store, 1);
    let written = objects.upsert(&store, stated("late"), Registrant::Control);
    let joined = written.expect("the store takes the write").id();
    issued_on(&store, 1, 1, joined, t0);
    let listed = store.deciding(|world| world.booked_of(joined).is_some());
    assert!(!listed, "the world lists the worker inside its window");
    dispatched(&store, &Hand::new(A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(sends(&count, joined), 1);
}

/// An apply hands the standing generation's sends to the generation it
/// takes, under the profile's name. A generation the store refuses
/// takes none.
#[test]
fn an_apply_hands_the_standing_sends_to_the_generation_it_takes() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let t0 = Instant::now();
    let served = ready(
        &store,
        &objects,
        "a",
        COUNTED,
        Role::Single,
        WireModes::HTTP,
    );
    let first = counting(&store, 1);
    issued_on(&store, 1, 1, served, t0);
    dispatched(&store, &open(&store, A), &a_dispatched(1), t0).expect("the frame reads");
    assert_eq!(sends(&first, served), 1);
    // A generation no newer than the one that stands is refused.
    let stale = counted(1);
    let refused = Arc::clone(&stale.engine().sent_credits[0]);
    assert!(store.apply(stale).is_err());
    assert_eq!(sends(&refused, served), 0, "a refused generation");
    let next = counting(&store, 2);
    assert_eq!(sends(&next, served), 1);
}
