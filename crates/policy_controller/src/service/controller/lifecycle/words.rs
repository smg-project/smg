//! What each word of a lease carries to the stages, written down whole: the
//! lease, the target, the side, the outcome and, at an issue, the decision.
//!
//! The words are a session's `Dispatched`, `Completed` and `Released`
//! frames, for a single's lease and for a pair's. Some come in the order a
//! host that holds its decision says them: a dispatch, then an end, or a
//! release before any dispatch. Others only a file of leases meets: an end
//! with no dispatch, a release behind one, a word on a side the lease does
//! not hold.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    completed, dispatched, open, released,
    tests::{a_completed, a_dispatched, a_released, worker},
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, Pair, Pick, Target, Targets, Trace},
        framework::{Picker, TieBand},
        sticky_sessions::SessionVerdict,
        view::{Candidate, RequestFacts, Session, WorkerId},
    },
    policies::random::Random,
    service::proto,
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{DecodeSide, GatewayId, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord},
    tracking::lease::{
        EventSet, Lease, LeaseId, LeaseKind, Outcome, ReleaseReason, Side, TransportClass,
    },
};

const GATEWAY: GatewayId = GatewayId(11);
/// The session of a single's lease and of a pair's prefill side.
const KEY: Session = Session::from_raw(41);
/// The session of a pair's decode side.
const DECODE_KEY: Session = Session::from_raw(43);
/// The generation every lease here was issued under.
const GENERATION: u64 = 1;
/// The instant every lease here was issued at, on the decision clock.
const ISSUED_MICROS: u64 = 1_000;
/// The prompt size every lease here states, but a decode side filed under
/// a size of its own.
const TOKENS: u32 = 100;
/// The span every dispatch here names.
const SINCE: u32 = 250;

/// One call of a stage about a lease, with all it was handed.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "a case compares what a stage was handed whole, the decision of an issue too"
)]
enum Word {
    Issued(Lease, Decision),
    Dispatched(Lease, Target, Side, u32),
    Completed(Lease, Outcome),
    Released(Lease, ReleaseReason),
}

/// The `random` picker, keeping every word it is told of a lease.
#[derive(Debug)]
struct Records(Arc<Mutex<Vec<Word>>>);

impl Lifecycle for Records {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.0.lock().push(Word::Issued(*lease, *decision));
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
        let word = Word::Dispatched(*lease, *target, side, since);
        self.0.lock().push(word);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.0.lock().push(Word::Completed(*lease, *outcome));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push(Word::Released(*lease, reason));
    }

    fn hears(&self) -> EventSet {
        EventSet::LEASE
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

/// A store whose standing generation keeps every word its stages are told,
/// with the workers of a pair: one prefill and two decodes.
struct Bed {
    store: MemoryStore,
    words: Arc<Mutex<Vec<Word>>>,
    now: Instant,
    prefill: WorkerId,
    decodes: [WorkerId; 2],
}

impl Bed {
    fn new() -> Self {
        let store = MemoryStore::new();
        let words = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&words);
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
        let [prefill, first, second] = ["p", "d0", "d1"].map(|key| worker(&store, key));
        Self {
            store,
            words,
            now: Instant::now(),
            prefill,
            decodes: [first, second],
        }
    }

    /// What the stages were told since the last read.
    fn told(&self) -> Vec<Word> {
        mem::take(&mut *self.words.lock())
    }

    /// A `Dispatched` frame on `side` that names the target at `index`.
    fn dispatched(&self, lease: u64, side: proto::Side, index: u32) {
        let frame = proto::Dispatched {
            side: side.into(),
            target_index: index,
            ..a_dispatched(lease)
        };
        let hand = open(&self.store, GATEWAY);
        dispatched(&self.store, &hand, &frame, self.now).expect("the frame reads");
    }

    /// A `Completed` frame on `side`, served by `served`: the outcome it
    /// states, as a stage is handed it.
    fn completed(&self, lease: u64, side: proto::Side, served: WorkerId) -> Outcome {
        let frame = proto::Completed {
            side: side.into(),
            ..a_completed(lease, served)
        };
        let hand = open(&self.store, GATEWAY);
        completed(&self.store, &hand, &frame, self.now).expect("the frame reads");
        let side = match side {
            proto::Side::Prefill => Side::Prefill,
            proto::Side::Decode => Side::Decode,
            proto::Side::Encode => Side::Encode,
            _ => Side::Single,
        };
        Outcome {
            side,
            served,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        }
    }

    /// A `Released` frame on `side`; the unspecified side gives every side
    /// up. Its reason is [`ReleaseReason::ClientGone`].
    fn released(&self, lease: u64, side: proto::Side) {
        let frame = proto::Released {
            side: side.into(),
            ..a_released(lease)
        };
        let hand = open(&self.store, GATEWAY);
        released(&self.store, &hand, &frame).expect("the frame reads");
    }

    /// The single's lease `id` as it was issued: a pick of worker 7 and,
    /// behind it, worker 9 at its rank 2.
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
        }
    }

    /// Files the single's lease `id` ([`Self::single`]).
    fn file_single(&self, id: u64) {
        let mut targets = Targets::new();
        for (worker, dp_rank) in [(7, 0), (9, 2)] {
            targets.push(Target {
                id: WorkerId(worker),
                dp_rank,
                ..Target::default()
            });
        }
        let record = self.record(Self::single(id), &targets);
        LeasesStore::issue(&self.store, record).expect("the key is free");
    }

    /// The record of `lease` over the pick `targets`, as a decider files it.
    fn record(&self, lease: Lease, targets: &Targets) -> LeaseRecord {
        LeaseRecord {
            lease,
            gateway: GATEWAY,
            request: lease.id.0,
            target: lease.booked,
            picked: PickRecord::of(targets),
            decode: None,
            term: 0,
            issued_at: self.now,
            dispatch_deadline: self.now + Duration::from_secs(5),
            state: LeaseState::ISSUED,
        }
    }

    /// The pick of a pair's lease: the prefill at its rank 1 with the
    /// first decode at its rank 2, and with the second at its rank 3.
    fn pair_targets(&self) -> Targets {
        let mut targets = Targets::new();
        for (decode_id, decode_dp_rank) in self.decodes.into_iter().zip([2, 3]) {
            targets.push(Target {
                id: self.prefill,
                dp_rank: 1,
                pair: Some(Pair {
                    decode_id,
                    decode_dp_rank,
                    ..Pair::default()
                }),
                ..Target::default()
            });
        }
        targets
    }

    /// The prefill side's lease of the pair `id`, as it was issued.
    fn prefill(&self, id: u64) -> Lease {
        Lease {
            id: LeaseId(id),
            kind: LeaseKind::Credit,
            side: Side::Prefill,
            target_index: 0,
            booked: self.prefill,
            generation: GENERATION,
            session: KEY,
            session_verdict: SessionVerdict::None,
            issued_micros: ISSUED_MICROS,
            tokens: TOKENS,
        }
    }

    /// The decode side's lease of the pair `id`, as it was issued beside
    /// its prefill's: on the pick's first decode, under the decode side's
    /// own session.
    fn decode(&self, id: u64) -> Lease {
        Lease {
            side: Side::Decode,
            booked: self.decodes[0],
            session: DECODE_KEY,
            ..self.prefill(id)
        }
    }

    /// Files the pair's lease `id` with its decode side in `state`, issued
    /// or chosen, stating `tokens`.
    fn file_pair(&self, id: u64, state: LeaseState, tokens: u32) {
        let targets = self.pair_targets();
        let side = DecodeSide {
            target: self.decodes[0],
            picked: PickRecord::of_decodes(&targets),
            session: DECODE_KEY,
            state,
            tokens,
            target_index: 0,
        };
        let record = self.record(self.prefill(id), &targets);
        LeasesStore::issue_pair(&self.store, record, side).expect("the key is free");
    }

    /// A target as a dispatch's word names it: a worker and its rank.
    fn target(id: WorkerId, dp_rank: u16) -> Target {
        Target {
            id,
            dp_rank,
            ..Target::default()
        }
    }
}

const UNSPECIFIED: proto::Side = proto::Side::Unspecified;

/// A single's dispatch is told under the lease as issued with the index
/// the frame names, on the target at that index. Its end is told under the
/// lease moved to that target.
#[test]
fn a_singles_dispatch_and_end_name_the_target_the_dispatch_took() {
    let bed = Bed::new();
    bed.file_single(1);
    bed.dispatched(1, proto::Side::Single, 0);
    let ended = bed.completed(1, proto::Side::Single, WorkerId(7));
    let issued = Bed::single(1);
    let first = Bed::target(WorkerId(7), 0);
    assert_eq!(
        bed.told(),
        [
            Word::Dispatched(issued, first, Side::Single, SINCE),
            Word::Completed(issued, ended),
        ]
    );
    // On the pick's second target: the dispatch still names the worker the
    // issue booked, and the end names the worker it went to.
    bed.file_single(2);
    bed.dispatched(2, proto::Side::Single, 1);
    let ended = bed.completed(2, proto::Side::Single, WorkerId(9));
    let issued = Bed::single(2);
    let sent = Lease {
        target_index: 1,
        ..issued
    };
    let moved = Lease {
        booked: WorkerId(9),
        ..sent
    };
    let second = Bed::target(WorkerId(9), 2);
    assert_eq!(
        bed.told(),
        [
            Word::Dispatched(sent, second, Side::Single, SINCE),
            Word::Completed(moved, ended),
        ]
    );
}

/// A release before any dispatch, and an end no dispatch came before, are
/// told under the lease as issued. A dispatch that names no target of the
/// pick tells nothing and moves nothing.
#[test]
fn a_singles_words_with_no_dispatch_carry_the_lease_as_issued() {
    let bed = Bed::new();
    bed.file_single(1);
    bed.released(1, UNSPECIFIED);
    let gone = ReleaseReason::ClientGone;
    assert_eq!(bed.told(), [Word::Released(Bed::single(1), gone)]);
    bed.file_single(2);
    let ended = bed.completed(2, proto::Side::Single, WorkerId(7));
    assert_eq!(bed.told(), [Word::Completed(Bed::single(2), ended)]);
    bed.file_single(3);
    bed.dispatched(3, proto::Side::Single, 3);
    assert_eq!(bed.told(), []);
    let ended = bed.completed(3, proto::Side::Single, WorkerId(7));
    assert_eq!(bed.told(), [Word::Completed(Bed::single(3), ended)]);
}

/// A release behind a dispatch is a word a file of leases alone meets: it
/// names the index and the worker the dispatch took.
#[test]
fn a_singles_release_behind_its_dispatch_names_the_target_it_went_to() {
    let bed = Bed::new();
    bed.file_single(1);
    bed.dispatched(1, proto::Side::Single, 1);
    bed.released(1, UNSPECIFIED);
    let moved = Lease {
        target_index: 1,
        booked: WorkerId(9),
        ..Bed::single(1)
    };
    let gone = ReleaseReason::ClientGone;
    assert_eq!(bed.told()[1..], [Word::Released(moved, gone)]);
}

/// A pair's words are told a side at a time, each under that side's own
/// lease. The prefill's never moves: every target of the pick names the one
/// prefill. The decode's dispatch names the index its frame took, on that
/// target's decode, and its end the worker it went to.
#[test]
fn a_pairs_words_are_told_under_the_lease_of_the_side_they_name() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED, TOKENS);
    bed.dispatched(1, proto::Side::Prefill, 0);
    bed.dispatched(1, proto::Side::Decode, 1);
    let [first, second] = bed.decodes;
    let prefill_ended = bed.completed(1, proto::Side::Prefill, bed.prefill);
    let decode_ended = bed.completed(1, proto::Side::Decode, second);
    let (prefill, decode) = (bed.prefill(1), bed.decode(1));
    assert_eq!(decode.booked, first);
    let sent = Lease {
        target_index: 1,
        ..decode
    };
    let moved = Lease {
        booked: second,
        ..sent
    };
    assert_eq!(
        bed.told(),
        [
            Word::Dispatched(prefill, Bed::target(bed.prefill, 1), Side::Prefill, SINCE),
            Word::Dispatched(sent, Bed::target(second, 3), Side::Decode, SINCE),
            Word::Completed(prefill, prefill_ended),
            Word::Completed(moved, decode_ended),
        ]
    );
}

/// A release with no side ends each side that stands, the prefill's first.
/// A release on a side ends that side alone. An end no dispatch came before
/// is told under its side's lease as issued. A word on a side the pair does
/// not hold tells nothing.
#[test]
fn a_pairs_release_and_end_reach_the_sides_that_stand() {
    let bed = Bed::new();
    let gone = ReleaseReason::ClientGone;
    bed.file_pair(1, LeaseState::ISSUED, TOKENS);
    bed.released(1, UNSPECIFIED);
    let (prefill, decode) = (bed.prefill(1), bed.decode(1));
    let whole = [Word::Released(prefill, gone), Word::Released(decode, gone)];
    assert_eq!(bed.told(), whole);
    bed.file_pair(2, LeaseState::ISSUED, TOKENS);
    let (prefill, decode) = (bed.prefill(2), bed.decode(2));
    bed.released(2, proto::Side::Decode);
    assert_eq!(bed.told(), [Word::Released(decode, gone)]);
    bed.released(2, proto::Side::Decode);
    bed.released(2, proto::Side::Prefill);
    assert_eq!(bed.told(), [Word::Released(prefill, gone)]);
    bed.file_pair(3, LeaseState::ISSUED, TOKENS);
    let (prefill, decode) = (bed.prefill(3), bed.decode(3));
    bed.dispatched(3, proto::Side::Single, 0);
    let _ = bed.completed(3, proto::Side::Encode, bed.prefill);
    bed.released(3, proto::Side::Encode);
    assert_eq!(bed.told(), []);
    let decode_ended = bed.completed(3, proto::Side::Decode, bed.decodes[0]);
    let prefill_ended = bed.completed(3, proto::Side::Prefill, bed.prefill);
    assert_eq!(
        bed.told(),
        [
            Word::Completed(decode, decode_ended),
            Word::Completed(prefill, prefill_ended),
        ]
    );
}

/// A decode side that stood chosen goes out with no decode call: its
/// dispatch is its issue too. The stages hear the issue first, dated at the
/// frame's read, on the worker the dispatch names and with the pick as the
/// record keeps it, then the dispatch under the lease's own instant. A
/// chosen side's end, and its release, tell nothing: no stage heard of it.
#[test]
fn a_chosen_decodes_dispatch_tells_its_issue_first_on_the_worker_it_goes_to() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::Chosen, TOKENS);
    let second = bed.decodes[1];
    let _ = bed.completed(1, proto::Side::Decode, second);
    bed.released(2, proto::Side::Decode);
    assert_eq!(bed.told(), []);
    bed.file_pair(2, LeaseState::Chosen, TOKENS);
    bed.dispatched(2, proto::Side::Prefill, 0);
    bed.dispatched(2, proto::Side::Decode, 1);
    let decode_ended = bed.completed(2, proto::Side::Decode, second);
    let (prefill, decode) = (bed.prefill(2), bed.decode(2));
    let sent = Lease {
        target_index: 1,
        booked: second,
        ..decode
    };
    let issued = Lease {
        issued_micros: bed.store.micros(bed.now),
        ..sent
    };
    let pick = Decision::Pick(Pick {
        targets: bed.pair_targets(),
        trace: Trace {
            generation: GENERATION,
            ..Trace::default()
        },
        ..Pick::default()
    });
    let at_second = Bed::target(second, 3);
    assert_eq!(
        bed.told(),
        [
            Word::Dispatched(prefill, Bed::target(bed.prefill, 1), Side::Prefill, SINCE),
            Word::Issued(issued, pick),
            Word::Dispatched(sent, at_second, Side::Decode, SINCE),
            Word::Completed(sent, decode_ended),
        ]
    );
}

/// A release behind the decode's dispatch on an alternate, a word a file of
/// leases alone meets: the decode side's release names the index and the
/// worker its dispatch took.
#[test]
fn a_pairs_moved_decode_released_behind_its_dispatch() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED, TOKENS);
    bed.dispatched(1, proto::Side::Decode, 1);
    bed.released(1, UNSPECIFIED);
    let moved = Lease {
        target_index: 1,
        booked: bed.decodes[1],
        ..bed.decode(1)
    };
    let gone = ReleaseReason::ClientGone;
    assert_eq!(
        bed.told()[1..],
        [
            Word::Released(bed.prefill(1), gone),
            Word::Released(moved, gone),
        ]
    );
}

/// A decode side filed under another prompt size than its prefill's, as a
/// decode call that states its own files it: each word of the side carries
/// the side's own size.
#[test]
fn a_decode_side_filed_under_another_size_than_its_prefill() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED, 3 * TOKENS);
    bed.dispatched(1, proto::Side::Decode, 0);
    let first = bed.decodes[0];
    let ended = bed.completed(1, proto::Side::Decode, first);
    let decode = Lease {
        tokens: 3 * TOKENS,
        ..bed.decode(1)
    };
    assert_eq!(
        bed.told(),
        [
            Word::Dispatched(decode, Bed::target(first, 2), Side::Decode, SINCE),
            Word::Completed(decode, ended),
        ]
    );
}
