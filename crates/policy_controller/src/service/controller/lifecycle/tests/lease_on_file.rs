//! What each word of a single's lease on file carries to the stages, told
//! two ways and written down whole: the lease, the target, the side and
//! the outcome.
//!
//! One way is the store-reading engine's own word, the one a served
//! controller gives its store, here with no worker object to ask
//! ([`open`]). The other is the rule a guard's
//! lease is told by ([`lease_events::single`]), over the lease as the file
//! lends it ([`Sides::with_held_single`]). A case runs one order of frames
//! over a store told each way: the stages are handed the same.
//!
//! Some orders are a host's that holds its decision: a dispatch and then
//! an end, or a release before any dispatch. Others only a file of leases
//! meets: an end with no dispatch, a release behind one, a dispatch that
//! names no target of the pick.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    super::{completed, dispatched, outcome, released},
    a_completed, a_dispatched, a_released, open,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Target, Targets},
        decision_engine::HeldDecode,
        decision_slot::DecisionSlot,
        framework::{Picker, TieBand},
        lease_events,
        sticky_sessions::SessionVerdict,
        view::{Candidate, RequestFacts, Session, WorkerId},
    },
    policies::random::Random,
    service::proto,
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        DecodeSide, GatewayId, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord,
        Sides, StoreSnapshot,
    },
    tracking::lease::{Dispatch, Lease, LeaseId, LeaseKind, Outcome, ReleaseReason, Said, Side},
};

const GATEWAY: GatewayId = GatewayId(11);
/// The session every lease here was issued under.
const KEY: Session = Session::from_raw(41);
/// The generation every lease here was issued under.
const GENERATION: u64 = 1;
/// The instant every lease here was issued at, on the decision clock.
const ISSUED_MICROS: u64 = 1_000;
/// The prompt size every lease here states.
const TOKENS: u32 = 100;
/// The span every dispatch here names.
const SINCE: u32 = 250;
/// How long a lease here waits for its dispatch.
const DISPATCH_TTL: Duration = Duration::from_secs(5);

/// One call of a stage about a lease, with all it was handed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Dispatched(Lease, Target, Side, u32),
    Completed(Lease, Outcome),
    Released(Lease, ReleaseReason),
}

/// The `random` picker, keeping every word it is told of a lease.
#[derive(Debug)]
struct Records(Arc<Mutex<Vec<Event>>>);

impl Lifecycle for Records {
    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
        let event = Event::Dispatched(*lease, *target, side, since);
        self.0.lock().push(event);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.0.lock().push(Event::Completed(*lease, *outcome));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push(Event::Released(*lease, reason));
    }
}

impl Picker for Records {
    fn name(&self) -> &'static str {
        "records"
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

/// Who tells the stages of a frame applied to a lease.
#[derive(Clone, Copy, Debug)]
enum Way {
    /// The store-reading engine's own word, as a served controller gives
    /// it to its store.
    Engine,
    /// The rule a guard's lease is told by, over the lease as the file
    /// lends it.
    Lent,
}

/// The word of [`Way::Lent`]: the lease as it stood before the frame, lent
/// as a guard lends its decision, and told by the rule a guard's lease is
/// told by, to the engine of the standing generation.
fn told_as_lent(
    _: &MemoryStore,
    world: &StoreSnapshot,
    found: &Sides,
    said: Said,
    _: Option<DecisionSlot>,
) {
    let engine = world.spec().engine();
    found.with_held_single(said, |held| lease_events::single(engine, held, said));
}

/// A store whose standing generation keeps every word its stages are told
/// of a lease.
struct Bed {
    store: MemoryStore,
    events: Arc<Mutex<Vec<Event>>>,
    now: Instant,
}

impl Bed {
    fn new(way: Way) -> Self {
        let store = MemoryStore::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&events);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("records", move || Box::new(Records(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = GENERATION;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "records".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage");
        store.apply(spec).expect("a later generation");
        // The first word a store is given stays: a hand opened later gives
        // the engine's own where the store holds none, and changes nothing
        // where it holds one.
        match way {
            Way::Engine => {}
            Way::Lent => store.tell_through(told_as_lent),
        }
        Self {
            store,
            events,
            now: Instant::now(),
        }
    }

    /// What the stages were told since the last read.
    fn told(&self) -> Vec<Event> {
        mem::take(&mut *self.events.lock())
    }

    /// A `Dispatched` frame that names the target at `index`.
    fn dispatched(&self, lease: u64, index: u32) {
        let frame = proto::Dispatched {
            target_index: index,
            ..a_dispatched(lease)
        };
        let hand = open(&self.store, GATEWAY);
        dispatched(&self.store, &hand, &frame, self.now).expect("the frame reads");
    }

    /// A `Completed` frame, served by `served`.
    fn completed(&self, lease: u64, served: WorkerId) {
        let (hand, frame) = (open(&self.store, GATEWAY), a_completed(lease, served));
        completed(&self.store, &hand, &frame, self.now).expect("the frame reads");
    }

    /// A `Released` frame that names no side. Its reason is
    /// [`ReleaseReason::ClientGone`].
    fn released(&self, lease: u64) {
        let hand = open(&self.store, GATEWAY);
        released(&self.store, &hand, &a_released(lease)).expect("the frame reads");
    }

    /// Files the single's lease `id` ([`single`]) over its pick
    /// ([`single_targets`]), as a decider files it for the request `id`.
    fn file_single(&self, id: u64) {
        self.file(id, false);
    }

    /// [`Self::file_single`] for a request its sessions rule keeps from
    /// the profile: the record says so ([`LeaseRecord::untold`]).
    fn file_kept(&self, id: u64) {
        self.file(id, true);
    }

    /// Files the single's lease `id` with `untold` as its record's word.
    fn file(&self, id: u64, untold: bool) {
        let lease = single(id);
        let record = LeaseRecord {
            lease,
            gateway: GATEWAY,
            request: id,
            target: lease.booked,
            picked: PickRecord::of(&single_targets()),
            decode: None,
            untold,
            term: 0,
            issued_at: self.now,
            dispatch_deadline: self.now + DISPATCH_TTL,
            state: LeaseState::ISSUED,
        };
        LeasesStore::issue(&self.store, record).expect("the key is free");
    }

    /// The worker the session of every lease here is pinned on first, in
    /// the table of the standing generation's engine.
    fn pinned(&self) -> Option<WorkerId> {
        let pins = |world: &StoreSnapshot| world.spec().engine().sessions().pins(KEY);
        self.store.deciding(pins).map(|pins| pins[0])
    }

    /// Both parts of the lease `id` as the file holds them.
    fn sides(&self, id: u64) -> Sides {
        LeasesStore::sides(&self.store, GATEWAY, LeaseId(id)).expect("a lease on file")
    }
}

/// The single's lease `id` as it was issued: a pick of worker 7 and, behind
/// it, worker 9 at its rank 2.
fn single(id: u64) -> Lease {
    Lease {
        id: LeaseId(id),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        booked: WorkerId(7),
        generation: GENERATION,
        session: KEY,
        session_verdict: SessionVerdict::Claimed(WorkerId(7)),
        issued_micros: ISSUED_MICROS,
        tokens: TOKENS,
        chars: 0,
        bytes: 0,
    }
}

/// The pick of a single's lease: worker 7 and, behind it, worker 9 at its
/// rank 2.
fn single_targets() -> Targets {
    let mut targets = Targets::new();
    for (worker, dp_rank) in [(7, 0), (9, 2)] {
        targets.push(target(WorkerId(worker), dp_rank));
    }
    targets
}

/// A target as a word names it: a worker and its rank, at no position.
fn target(id: WorkerId, dp_rank: u16) -> Target {
    Target {
        id,
        dp_rank,
        ..Target::default()
    }
}

/// The outcome a `Completed` frame of `lease` served by `served` states,
/// as the frame's own reader reads it and as a stage is handed it.
fn ended(lease: u64, served: WorkerId) -> Outcome {
    let (_, stated) = outcome(&a_completed(lease, served)).expect("the frame reads");
    stated
}

/// Runs `frames` over a store told each way and answers what the stages
/// were handed, which is the same both ways.
fn told_both_ways(frames: impl Fn(&Bed)) -> Vec<Event> {
    let (engine, lent) = (Bed::new(Way::Engine), Bed::new(Way::Lent));
    frames(&engine);
    frames(&lent);
    let told = engine.told();
    assert_eq!(lent.told(), told, "the lease as lent tells the same");
    told
}

/// A single's dispatch is told under the lease as issued with the index
/// the frame names, on the target at that index. Its end is told under the
/// lease moved to that target.
#[test]
fn a_singles_dispatch_and_end_name_the_target_the_dispatch_took() {
    let told = told_both_ways(|bed| {
        bed.file_single(1);
        bed.dispatched(1, 0);
        bed.completed(1, WorkerId(7));
        // On the pick's second target.
        bed.file_single(2);
        bed.dispatched(2, 1);
        bed.completed(2, WorkerId(9));
    });
    let issued = single(1);
    let sent = Lease {
        target_index: 1,
        ..single(2)
    };
    let moved = Lease {
        booked: WorkerId(9),
        ..sent
    };
    assert_eq!(
        told,
        [
            Event::Dispatched(issued, target(WorkerId(7), 0), Side::Single, SINCE),
            Event::Completed(issued, ended(1, WorkerId(7))),
            // The dispatch still names the worker the issue booked, and
            // the end names the worker it went to.
            Event::Dispatched(sent, target(WorkerId(9), 2), Side::Single, SINCE),
            Event::Completed(moved, ended(2, WorkerId(9))),
        ]
    );
}

/// A release before any dispatch, and an end no dispatch came before, are
/// told under the lease as issued. A dispatch that names no target of the
/// pick tells nothing and moves nothing.
#[test]
fn a_singles_words_with_no_dispatch_carry_the_lease_as_issued() {
    let told = told_both_ways(|bed| {
        bed.file_single(1);
        bed.released(1);
        bed.file_single(2);
        bed.completed(2, WorkerId(7));
        bed.file_single(3);
        bed.dispatched(3, 3);
        bed.completed(3, WorkerId(7));
    });
    assert_eq!(
        told,
        [
            Event::Released(single(1), ReleaseReason::ClientGone),
            Event::Completed(single(2), ended(2, WorkerId(7))),
            Event::Completed(single(3), ended(3, WorkerId(7))),
        ]
    );
}

/// A single's lease whose request its sessions rule keeps from the profile
/// tells no stage its dispatch, told either way: the store's own word reads
/// the record ([`LeaseRecord::untold`]), and the lease as lent carries a
/// pick that says the same. The session table hears the dispatch, and the
/// session's pin follows the worker it took. The end and the release are
/// told as any lease's.
#[test]
fn a_kept_singles_dispatch_moves_its_pin_and_reaches_no_stage() {
    let told = told_both_ways(|bed| {
        bed.file_kept(1);
        assert_eq!(bed.pinned(), None);
        bed.dispatched(1, 1);
        assert_eq!(bed.pinned(), Some(WorkerId(9)));
        bed.completed(1, WorkerId(9));
        bed.file_kept(2);
        bed.dispatched(2, 0);
        bed.released(2);
    });
    let moved = Lease {
        target_index: 1,
        booked: WorkerId(9),
        ..single(1)
    };
    assert_eq!(
        told,
        [
            Event::Completed(moved, ended(1, WorkerId(9))),
            Event::Released(single(2), ReleaseReason::ClientGone),
        ]
    );
}

/// A release behind a dispatch is a word a file of leases alone meets: it
/// names the index and the worker the dispatch took.
#[test]
fn a_singles_release_behind_its_dispatch_names_the_target_it_went_to() {
    let told = told_both_ways(|bed| {
        bed.file_single(1);
        bed.dispatched(1, 1);
        bed.released(1);
    });
    let moved = Lease {
        target_index: 1,
        booked: WorkerId(9),
        ..single(1)
    };
    assert_eq!(
        told[1..],
        [Event::Released(moved, ReleaseReason::ClientGone)]
    );
}

/// What a single's lease lends with a word, part by part, as it stood
/// before the word: the record's own lease, the index and the target the
/// word is about, a pick of no target, and a decode side that is a
/// single's. A dispatch that names no target of the pick lends nothing.
/// A lease that states its text's two lengths lends them with its decode
/// side, as it lends its token count.
#[test]
fn a_singles_lease_is_lent_as_it_stood_before_the_word() {
    let bed = Bed::new(Way::Engine);
    bed.file_single(1);
    let lent = |sides: Sides, said| {
        sides.with_held_single(said, |held| {
            let listed = held.decision.pick().map(|pick| pick.targets.len());
            assert_eq!(listed, Some(0), "a pick of no target");
            (*held.lease, held.index, *held.target, held.decode)
        })
    };
    let dispatch = |index| {
        let dispatch = Dispatch {
            target_index: index,
            side: Side::Single,
            since_decision_micros: SINCE,
        };
        Said::Dispatched(dispatch, bed.now)
    };
    let ends = [
        Said::Completed(ended(1, WorkerId(9))),
        Said::Released(None, ReleaseReason::ClientGone),
    ];
    let a_singles = HeldDecode {
        session: Session::NONE,
        tokens: TOKENS,
        chars: 0,
        bytes: 0,
        chosen: false,
    };
    let (first, second) = (target(WorkerId(7), 0), target(WorkerId(9), 2));

    // As issued: a dispatch lends the target its index names, and an end
    // the one the lease books.
    let issued = bed.sides(1);
    assert_eq!(
        lent(issued, dispatch(1)),
        Some((single(1), 1, second, a_singles))
    );
    assert_eq!(lent(issued, dispatch(3)), None);
    for end in ends {
        assert_eq!(lent(issued, end), Some((single(1), 0, first, a_singles)));
    }

    // Behind a dispatch on the second target: every word is lent the lease
    // on that index and that worker.
    bed.dispatched(1, 1);
    let moved = Lease {
        target_index: 1,
        booked: WorkerId(9),
        ..single(1)
    };
    for end in ends {
        let sent = bed.sides(1);
        assert_eq!(lent(sent, end), Some((moved, 1, second, a_singles)));
    }

    // A lease of a request of text: 400 characters in 800 bytes.
    let (chars, bytes) = (400, 800);
    let mut of_text = bed.sides(1);
    (of_text.record.lease.chars, of_text.record.lease.bytes) = (chars, bytes);
    let own = Lease {
        chars,
        bytes,
        ..moved
    };
    let stated = HeldDecode {
        chars,
        bytes,
        ..a_singles
    };
    assert_eq!(lent(of_text, ends[0]), Some((own, 1, second, stated)));
}

/// A pair's lease is no single's: with its decode side beside the record
/// it lends nothing here, whatever the word.
#[test]
fn a_pairs_lease_lends_nothing_as_a_singles() {
    let bed = Bed::new(Way::Engine);
    bed.file_single(1);
    let record = bed.sides(1).record;
    let decode = DecodeSide {
        target: WorkerId(8),
        picked: PickRecord::default(),
        session: Session::NONE,
        state: LeaseState::ISSUED,
        tokens: TOKENS,
        target_index: 0,
    };
    let pair = Sides {
        record,
        decode: Some(decode),
    };
    let dispatch = Dispatch {
        target_index: 0,
        side: Side::Prefill,
        since_decision_micros: SINCE,
    };
    let words = [
        Said::Dispatched(dispatch, bed.now),
        Said::Completed(ended(1, WorkerId(7))),
        Said::Released(None, ReleaseReason::ClientGone),
    ];
    for said in words {
        assert_eq!(pair.with_held_single(said, |_| ()), None);
    }
}
