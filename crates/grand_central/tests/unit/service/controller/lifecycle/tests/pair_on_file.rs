//! What each word of a pair's lease on file carries to the stages, beside
//! what the same rule tells from a guard's view of the pair.
//!
//! A served controller tells a pair's words by the rule a guard's lease is
//! told by (`lease_events::pair`), over the lease as the file lends it
//! ([`Sides::with_held_pair`]). A file meets words no guard says, and the
//! lender turns each into words the rule knows. A case here is one such
//! place. It runs frames over a store and reads what the stages were
//! handed ([`Bed::told`]), then tells the rule the same word from the view
//! a guard would lend of the same pair ([`Bed::held_tells`]).

use std::collections::BTreeMap;

use parking_lot::Mutex;

use super::{
    super::{completed, dispatched, outcome, released},
    a_completed, a_dispatched, a_released,
    lease_on_file::{
        target, Bed, Event, DISPATCH_TTL, GATEWAY, GENERATION, ISSUED_MICROS, KEY, SINCE, TOKENS,
    },
    open,
};
use crate::{
    engine::{
        decision::{Decision, Pair, Pick, Target, Targets, Trace},
        decision_engine::{HeldDecision, HeldDecode},
        lease_events,
        sticky_sessions::{Claim, DecodeVerdict, SessionVerdict},
        view::{Session, WorkerId},
    },
    service::proto,
    stages::lifecycle::Lifecycle,
    store::{DecodeSide, LeaseRecord, LeaseState, LeasesStore, PickRecord, Sides, StoreSnapshot},
    tracking::lease::{Dispatch, Lease, LeaseId, LeaseKind, Outcome, ReleaseReason, Said, Side},
};

/// The prefill of every pair here, at its rank 1.
const P: WorkerId = WorkerId(3);
/// The decode of the pick's first target, at its rank 0.
const D0: WorkerId = WorkerId(7);
/// The decode of the pick's second target, at its rank 2.
const D1: WorkerId = WorkerId(9);
/// The session every pair here states under the decode phase.
const DECODE_KEY: Session = Session::from_raw(43);
/// The prompt size every decode side here states.
const DECODE_TOKENS: u32 = 60;
/// The reason of every release here, a frame's and a guard's.
const GONE: ReleaseReason = ReleaseReason::ClientGone;

/// The pick of every pair here as its decision made it: the prefill with
/// the first decode, then with the second, each at a position of its
/// host's lists. The first pair states the decode side's verdict, a claim,
/// and the pick states the prefill side's.
fn picked() -> Pick {
    let mut targets = Targets::new();
    let decodes = [
        (5, D0, 0, DecodeVerdict::CLAIMED),
        (6, D1, 2, DecodeVerdict::NONE),
    ];
    for (decode_position, decode_id, decode_dp_rank, decode_verdict) in decodes {
        targets.push(Target {
            position: 4,
            id: P,
            dp_rank: 1,
            pair: Some(Pair {
                decode_position,
                decode_id,
                decode_dp_rank,
                decode_verdict,
                ..Pair::default()
            }),
        });
    }
    Pick {
        targets,
        trace: Trace {
            seed: 77,
            generation: GENERATION,
            ..Trace::default()
        },
        session_verdict: SessionVerdict::Pinned,
    }
}

/// The pick as a file of leases lends it with the issue of a decode side
/// no call issued: every target's two workers and their ranks at no
/// position, the first pair's verdict, and no trace beside the generation.
fn picked_on_file() -> Pick {
    let own = picked();
    let mut targets = Targets::new();
    for target in own.targets.iter() {
        let pair = target.pair.map(|pair| Pair {
            decode_position: 0,
            ..pair
        });
        targets.push(Target {
            position: 0,
            pair,
            ..*target
        });
    }
    Pick {
        targets,
        trace: Trace {
            generation: GENERATION,
            ..Trace::default()
        },
        session_verdict: SessionVerdict::None,
    }
}

/// The prefill side's lease of the pair `id` as it was issued.
fn prefill(id: u64) -> Lease {
    Lease {
        id: LeaseId(id),
        kind: LeaseKind::Credit,
        side: Side::Prefill,
        target_index: 0,
        booked: P,
        generation: GENERATION,
        session: KEY,
        session_verdict: SessionVerdict::Pinned,
        issued_micros: ISSUED_MICROS,
        tokens: TOKENS,
        chars: 0,
        bytes: 0,
    }
}

/// The decode side's lease of the pair `id` as it was issued: on the first
/// decode, under the side's own session, verdict and size.
fn decode(id: u64) -> Lease {
    Lease {
        side: Side::Decode,
        booked: D0,
        session: DECODE_KEY,
        session_verdict: SessionVerdict::Claimed(D0),
        tokens: DECODE_TOKENS,
        ..prefill(id)
    }
}

/// The decode side's lease of the pair `id` on the pick's second target,
/// as its dispatch there is told: still on the worker its issue booked.
fn decode_sent(id: u64) -> Lease {
    Lease {
        target_index: 1,
        ..decode(id)
    }
}

/// A dispatch of `side` on the target at `index`, as a word.
fn dispatch(side: Side, index: u8) -> Dispatch {
    Dispatch {
        target_index: index,
        side,
        since_decision_micros: SINCE,
    }
}

/// The side a frame names for `side`; none is the frame's unspecified
/// side.
fn framed(side: Option<Side>) -> proto::Side {
    match side {
        Some(Side::Prefill) => proto::Side::Prefill,
        Some(Side::Decode) => proto::Side::Decode,
        Some(_) => proto::Side::Single,
        None => proto::Side::Unspecified,
    }
}

/// The outcome a `Completed` frame of `side` of the pair `id` served by
/// `served` states, as the frame's own reader reads it.
fn ended(id: u64, side: Side, served: WorkerId) -> Outcome {
    let frame = proto::Completed {
        side: framed(Some(side)).into(),
        ..a_completed(id, served)
    };
    let (_, stated) = outcome(&frame).expect("the frame reads");
    stated
}

impl Bed {
    /// Files the pair `id` over its pick ([`picked`]), its decode side
    /// standing as `state`: issued with the prefill, or chosen for a
    /// later decode.
    fn file_pair(&self, id: u64, state: LeaseState) {
        let pick = picked();
        let record = LeaseRecord {
            lease: prefill(id),
            gateway: GATEWAY,
            request: id,
            target: P,
            picked: PickRecord::of(&pick.targets),
            decode: None,
            untold: false,
            term: 0,
            issued_at: self.now,
            dispatch_deadline: self.now + DISPATCH_TTL,
            state: LeaseState::ISSUED,
        };
        let side = DecodeSide {
            target: D0,
            picked: PickRecord::of_decodes(&pick.targets),
            session: DECODE_KEY,
            session_verdict: SessionVerdict::Claimed(D0),
            state,
            tokens: DECODE_TOKENS,
            target_index: 0,
        };
        LeasesStore::issue_pair(&self.store, record, side).expect("the key is free");
    }

    /// Hands the file `said` of the pair `id` as its gateway's frame.
    fn frame(&self, id: u64, said: Said) {
        let hand = open(&self.store, GATEWAY);
        match said {
            Said::Dispatched(dispatch, _) => {
                let frame = proto::Dispatched {
                    side: framed(Some(dispatch.side)).into(),
                    target_index: dispatch.target_index.into(),
                    ..a_dispatched(id)
                };
                dispatched(&self.store, &hand, &frame, self.now).expect("the frame reads");
            }
            Said::Completed(outcome) => {
                let frame = proto::Completed {
                    side: framed(Some(outcome.side)).into(),
                    ..a_completed(id, outcome.served)
                };
                completed(&self.store, &hand, &frame).expect("the frame reads");
            }
            Said::Released(side, _) => {
                let frame = proto::Released {
                    side: framed(side).into(),
                    ..a_released(id)
                };
                released(&self.store, &hand, &frame).expect("the frame reads");
            }
        }
    }

    /// What the rule tells of `said` from a guard's view of the pair `id`:
    /// a host that holds the pair's decision and took the pick's target at
    /// `index`, its decode side `chosen` or issued. The host has the one
    /// engine and dates a chosen side's issue by the store's clock, as the
    /// file does.
    fn held_tells(&self, id: u64, index: u8, chosen: bool, said: Said) -> Vec<Event> {
        let (pick, lease) = (picked(), prefill(id));
        let decision = Decision::Pick(pick);
        let held = HeldDecision {
            decision: &decision,
            lease: &lease,
            index,
            target: &pick.targets.as_slice()[usize::from(index)],
            decode: HeldDecode {
                session: DECODE_KEY,
                tokens: DECODE_TOKENS,
                chars: 0,
                bytes: 0,
                chosen,
            },
        };
        self.store.deciding(|world: &StoreSnapshot| {
            let engine = world.spec().engine();
            let hearers = lease_events::of_host(engine, engine);
            lease_events::pair(&hearers, &held, said, |at| world.micros(at));
        });
        self.told()
    }

    /// The words `said` under both hosts: handed to the file of the pair
    /// `id` as frames, then told by the rule from a guard's view of the
    /// same pair on the target at `index`.
    fn both(&self, id: u64, index: u8, chosen: bool, said: &[Said]) -> [Vec<Event>; 2] {
        for &word in said {
            self.frame(id, word);
        }
        let on_file = self.told();
        let mut held = Vec::new();
        for (at, &word) in said.iter().enumerate() {
            // A side that stood chosen is issued by its dispatch, the
            // first word of it.
            held.extend(self.held_tells(id, index, chosen && at == 0, word));
        }
        [on_file, held]
    }

    /// The instant of this bed's frames on the store's clock, which dates
    /// the issue of a side its dispatch issues.
    fn frame_micros(&self) -> u64 {
        self.store.micros(self.now)
    }
}

/// One call of a pair's lender: its word, its index, its target, and the
/// first target of its pick as the rule reads it, the decode and the
/// verdict.
type Lent = (Said, u8, Target, Option<(WorkerId, SessionVerdict)>);

/// A stage that books each side it is told of on its worker, and undoes
/// by the worker a word names: one more at an issue, moved at a dispatch
/// on another worker, one fewer at an end.
#[derive(Debug, Default)]
struct Books(Mutex<BTreeMap<WorkerId, i32>>);

impl Books {
    fn add(&self, worker: WorkerId, sides: i32) {
        *self.0.lock().entry(worker).or_default() += sides;
    }

    /// What the stage still holds once it heard `events` in their order:
    /// each worker it holds a count on that is not zero.
    fn left_by(events: &[Event]) -> Vec<(WorkerId, i32)> {
        let stage = Self::default();
        for event in events {
            match event {
                Event::Issued(lease, decision) => stage.on_decision_issued(lease, decision),
                Event::Dispatched(lease, target, side, since) => {
                    stage.on_dispatched(lease, target, *side, *since);
                }
                Event::Completed(lease, outcome) => stage.on_completed(lease, outcome),
                Event::Released(lease, reason) => stage.on_released(lease, *reason),
            }
        }
        let held = stage.0.into_inner();
        held.into_iter().filter(|&(_, sides)| sides != 0).collect()
    }
}

impl Lifecycle for Books {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.add(lease.booked, 1);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        self.add(lease.booked, -1);
        self.add(target.id, 1);
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        self.add(lease.booked, -1);
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.add(lease.booked, -1);
    }
}

/// Each side of a pair is told under its own lease: the decode's states
/// the side's own session, size and session verdict, which the file keeps
/// with the side. A dispatch names the worker the side's issue booked and
/// the target it took, and the side's end names the worker it went to. A
/// claim still names the decode the side was issued on. A guard's words
/// are the same.
#[test]
fn each_side_of_a_pair_on_file_is_told_under_its_own_lease() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED);
    let words = [
        Said::Dispatched(dispatch(Side::Prefill, 1), bed.now),
        Said::Dispatched(dispatch(Side::Decode, 1), bed.now),
        Said::Completed(ended(1, Side::Prefill, P)),
        Said::Completed(ended(1, Side::Decode, D1)),
    ];
    let [on_file, held] = bed.both(1, 1, false, &words);
    let sent = Lease {
        target_index: 1,
        ..prefill(1)
    };
    let moved = Lease {
        booked: D1,
        ..decode_sent(1)
    };
    assert_eq!(moved.session_verdict, SessionVerdict::Claimed(D0));
    assert_eq!(
        on_file,
        [
            Event::Dispatched(sent, target(P, 1), Side::Prefill, SINCE),
            Event::Dispatched(decode_sent(1), target(D1, 2), Side::Decode, SINCE),
            // No dispatch moves a pair's prefill.
            Event::Completed(prefill(1), ended(1, Side::Prefill, P)),
            Event::Completed(moved, ended(1, Side::Decode, D1)),
        ]
    );
    assert_eq!(on_file, held);
}

/// A release of a decode side behind its dispatch is a word a file of
/// leases alone meets: a guard ends a side it sent by a completion. The
/// file's names the worker the side went to, under index 0. It states no
/// claim there: the side claimed the first decode, and a pair states a
/// claim on its own decode alone. A stage that books by the worker
/// holds nothing behind it. The rule would state a guard's on the first
/// target's decode, which that stage no longer holds: the file keeps the
/// side's target and not the first.
#[test]
fn a_decodes_release_behind_its_dispatch_names_the_worker_it_went_to() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED);
    let issue = Event::Issued(decode(1), Box::new(Decision::Pick(picked())));
    let words = [
        Said::Dispatched(dispatch(Side::Decode, 1), bed.now),
        Said::Released(Some(Side::Decode), GONE),
    ];
    let [on_file, held] = bed.both(1, 1, false, &words);
    let went_to = Lease {
        booked: D1,
        session_verdict: SessionVerdict::None,
        ..decode(1)
    };
    let sent = Event::Dispatched(decode_sent(1), target(D1, 2), Side::Decode, SINCE);
    assert_eq!(on_file, [sent.clone(), Event::Released(went_to, GONE)]);
    assert_eq!(held, [sent, Event::Released(decode(1), GONE)]);
    // The stage booked the first decode at the side's issue, and the
    // dispatch moved its booking.
    let heard = |words: &[Event]| [&[issue.clone()][..], words].concat();
    assert!(Books::left_by(&heard(&on_file)).is_empty());
    assert_eq!(Books::left_by(&heard(&held)), [(D0, -1), (D1, 1)]);

    // Behind a dispatch on the first target the side stands where it was
    // issued, and the two hosts say one word, the side's claim in it.
    bed.file_pair(2, LeaseState::ISSUED);
    let words = [
        Said::Dispatched(dispatch(Side::Decode, 0), bed.now),
        Said::Released(Some(Side::Decode), GONE),
    ];
    let [on_file, held] = bed.both(2, 0, false, &words);
    assert_eq!(on_file[1..], [Event::Released(decode(2), GONE)]);
    assert_eq!(on_file, held);
}

/// A release for a commit that failed takes back the claim its lease
/// states (`Engine::on_released`), and a frame may state that reason
/// behind a dispatch, which no guard does. Behind a dispatch that moved a
/// pair's decode side the file states no claim: the side's own stands no
/// more, so a claim that stands on the key is a later decision's. It
/// stands, on the first decode as on the worker the side went to. A side
/// that waits where it was issued states its claim, and the release takes
/// it back.
#[test]
fn a_decodes_release_behind_a_dispatch_elsewhere_takes_back_no_claim() {
    let bed = Bed::new();
    let none = WorkerId(0);
    let pins = || {
        let read = |world: &StoreSnapshot| world.spec().engine().sessions().pins(DECODE_KEY);
        bed.store.deciding(read).unwrap_or_default()
    };
    // A decision of the key that read its pins as they stand claims `on`.
    let claims = |on: WorkerId| {
        let seen = pins();
        let claim = |world: &StoreSnapshot| {
            let table = world.spec().engine().sessions();
            table.claim(DECODE_KEY, seen, on)
        };
        assert_eq!(bed.store.deciding(claim), Claim::Made(on));
    };
    // The workers leave for good, and the key's pins on them go.
    let forgets = |workers: &[WorkerId]| {
        let forget = |world: &StoreSnapshot| {
            world.spec().engine().sessions().forget_workers(workers);
        };
        bed.store.deciding(forget);
    };
    // The decode side of the pair `id` is released as a commit that failed.
    let fails = |id: u64| {
        let frame = proto::Released {
            reason: proto::released::Reason::CommitFailed.into(),
            side: proto::Side::Decode.into(),
            ..a_released(id)
        };
        released(&bed.store, &open(&bed.store, GATEWAY), &frame).expect("the frame reads");
    };
    let sent = Said::Dispatched(dispatch(Side::Decode, 1), bed.now);

    // The side's decision claims the first decode, and its dispatch on the
    // second target withdraws that claim: the key follows the send. A
    // later decision of the key claims the first decode again.
    claims(D0);
    bed.file_pair(1, LeaseState::ISSUED);
    bed.frame(1, sent);
    assert_eq!(pins(), [D1, none]);
    claims(D0);
    fails(1);
    assert_eq!(pins(), [D1, D0], "a later claim on the first decode");

    // The same where the later decision claims the worker the side went
    // to: the key lost its pin with that worker meanwhile.
    forgets(&[D0, D1]);
    claims(D0);
    bed.file_pair(2, LeaseState::ISSUED);
    bed.frame(2, sent);
    forgets(&[D1]);
    claims(D1);
    fails(2);
    assert_eq!(pins(), [D1, none], "a later claim on the worker it went to");

    // A side that waits where it was issued states its claim.
    forgets(&[D1]);
    claims(D0);
    bed.file_pair(3, LeaseState::ISSUED);
    fails(3);
    assert_eq!(pins(), [none, none], "the side's own claim");
}

/// A release that names no side ends each side that is issued or
/// dispatched, the prefill's first, and no other. A guard's rule ends the
/// prefill whenever it is told so: a guard says the word of a lease it
/// holds whole. A file meets it behind a side's own end, and tells the
/// side that stands alone. A second end of a side tells nothing.
#[test]
fn a_release_of_every_side_is_told_of_each_side_that_still_stands() {
    let bed = Bed::new();
    let whole = [Said::Released(None, GONE)];
    let both_sides = |id| {
        [
            Event::Released(prefill(id), GONE),
            Event::Released(decode(id), GONE),
        ]
    };
    // A pair that waits whole: both hosts end both sides.
    bed.file_pair(1, LeaseState::ISSUED);
    let [on_file, held] = bed.both(1, 0, false, &whole);
    assert_eq!(on_file, both_sides(1));
    assert_eq!(on_file, held);

    // Behind the prefill's end, said twice: the file ends the decode
    // alone, and the rule from a guard's view would end the prefill again.
    bed.file_pair(2, LeaseState::ISSUED);
    let done = Said::Completed(ended(2, Side::Prefill, P));
    let [on_file, held] = bed.both(2, 0, false, &[done, done, whole[0]]);
    let ended_once = Event::Completed(prefill(2), ended(2, Side::Prefill, P));
    assert_eq!(
        on_file,
        [ended_once.clone(), Event::Released(decode(2), GONE)]
    );
    let [prefill_again, decode_too] = both_sides(2);
    assert_eq!(
        held,
        [ended_once.clone(), ended_once, prefill_again, decode_too]
    );

    // A decode side still chosen was told to no stage: neither host ends
    // it.
    bed.file_pair(3, LeaseState::Chosen);
    let [on_file, held] = bed.both(3, 0, true, &whole);
    assert_eq!(on_file, [Event::Released(prefill(3), GONE)]);
    assert_eq!(on_file, held);
}

/// The dispatch of a decode side no call issued is the side's issue too.
/// The stages hear the issue first, dated at the dispatch's frame, under
/// the worker the side goes to, and then the dispatch under the lease's
/// own instant. The issue carries the pair's pick: a guard hands its own,
/// and a file the one its record keeps, every target's workers and ranks
/// with the first pair's verdict, at no position and under no trace beside
/// the generation.
#[test]
fn a_chosen_decodes_dispatch_is_its_issue_under_the_pick_on_file() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::Chosen);
    let words = [
        Said::Dispatched(dispatch(Side::Decode, 1), bed.now),
        Said::Completed(ended(1, Side::Decode, D1)),
    ];
    let [on_file, held] = bed.both(1, 1, true, &words);
    let sent = Lease {
        booked: D1,
        ..decode_sent(1)
    };
    let issued = Lease {
        issued_micros: bed.frame_micros(),
        ..sent
    };
    let behind = [
        Event::Dispatched(sent, target(D1, 2), Side::Decode, SINCE),
        Event::Completed(sent, ended(1, Side::Decode, D1)),
    ];
    let under = |pick: Pick| Event::Issued(issued, Box::new(Decision::Pick(pick)));
    assert_eq!(on_file[0], under(picked_on_file()));
    assert_eq!(held[0], under(picked()));
    assert_eq!(on_file[1..], behind);
    assert_eq!(held[1..], behind);
    // A stage that books at the issue books the worker that serves, once.
    assert!(Books::left_by(&on_file).is_empty());

    // A dispatch that names no target of the pick is a frame's alone: it
    // tells nothing. Neither does a word on a side a pair does not hold.
    bed.file_pair(2, LeaseState::Chosen);
    bed.frame(2, Said::Dispatched(dispatch(Side::Decode, 3), bed.now));
    bed.frame(2, Said::Dispatched(dispatch(Side::Single, 0), bed.now));
    assert!(bed.told().is_empty());
}

/// What a pair's lease lends with a word, part by part, as it stood before
/// the word: the prefill side's lease, the index and the target the word
/// is about in a pair's form, the decode side as the file keeps it, and a
/// pick. A word of the prefill side is lent a pick of no target where no
/// decode stands chosen, as none does here, and a word
/// of the decode side the pick's first target alone: the decode the side's
/// lease books, with the side's verdict. A single's lease lends nothing
/// here.
#[test]
fn a_pairs_lease_is_lent_as_it_stood_before_the_word() {
    let bed = Bed::new();
    bed.file_pair(1, LeaseState::ISSUED);
    let sides = |id: u64| -> Sides {
        LeasesStore::sides(&bed.store, GATEWAY, LeaseId(id)).expect("a lease on file")
    };
    let lent = |found: Sides, said: Said| {
        let mut calls: Vec<Lent> = Vec::new();
        found.with_held_pair(said, |held, said| {
            assert_eq!(*held.lease, prefill(1));
            let own = HeldDecode {
                session: DECODE_KEY,
                tokens: DECODE_TOKENS,
                chars: found.record.lease.chars,
                bytes: found.record.lease.bytes,
                chosen: false,
            };
            assert_eq!(held.decode, own);
            let pick = held.decision.pick().expect("a pick");
            assert!(pick.targets.len() <= 1, "no pick but the first target");
            let first = pick.targets.first().and_then(|first| first.pair);
            let first = first.map(|pair| (pair.decode_id, pair.decode_verdict.on(pair.decode_id)));
            calls.push((said, held.index, *held.target, first));
        });
        calls
    };
    let paired = |decode: WorkerId, rank: u16| Target {
        pair: Some(Pair {
            decode_id: decode,
            decode_dp_rank: rank,
            ..Pair::default()
        }),
        ..target(P, 1)
    };
    let claimed = |on: WorkerId| Some((on, SessionVerdict::Claimed(on)));
    let of = |side| Said::Released(Some(side), GONE);
    let done = |side| Said::Completed(ended(1, side, D1));

    // As issued: a dispatch lends the target its index names on both
    // sides, and an end the workers the sides book.
    let issued = sides(1);
    let sent = Said::Dispatched(dispatch(Side::Prefill, 1), bed.now);
    assert_eq!(lent(issued, sent), [(sent, 1, paired(D1, 2), None)]);
    let sent = Said::Dispatched(dispatch(Side::Decode, 1), bed.now);
    assert_eq!(lent(issued, sent), [(sent, 1, paired(D1, 2), claimed(D0))]);
    for side in [Side::Prefill, Side::Decode] {
        let past = Said::Dispatched(dispatch(side, 3), bed.now);
        assert!(lent(issued, past).is_empty());
    }
    assert_eq!(
        lent(issued, Said::Released(None, GONE)),
        [
            (of(Side::Prefill), 0, paired(D0, 0), None),
            (of(Side::Decode), 0, paired(D0, 0), claimed(D0)),
        ]
    );

    // Behind the decode's dispatch on the second target: its completion
    // is lent the claim's worker as the first, and its release the worker
    // the side went to, under no claim. The prefill's words stand as they
    // were.
    bed.frame(1, sent);
    let moved = sides(1);
    assert_eq!(
        lent(moved, done(Side::Decode)),
        [(done(Side::Decode), 1, paired(D1, 2), claimed(D0))]
    );
    let unclaimed = Some((D1, SessionVerdict::None));
    assert_eq!(
        lent(moved, of(Side::Decode)),
        [(of(Side::Decode), 1, paired(D1, 2), unclaimed)]
    );
    assert_eq!(
        lent(moved, done(Side::Prefill)),
        [(done(Side::Prefill), 0, paired(D1, 2), None)]
    );

    // A side that ended is lent no end, and a word on a side a pair does
    // not hold is lent nothing.
    bed.frame(1, done(Side::Decode));
    let half = sides(1);
    for none in [done(Side::Decode), of(Side::Decode), done(Side::Single)] {
        assert!(lent(half, none).is_empty());
    }
    assert_eq!(
        lent(half, Said::Released(None, GONE)),
        [(of(Side::Prefill), 0, paired(D1, 2), None)]
    );

    // A frame states a count and no text's length, so a controller files a
    // lease with neither. A record that states them lends them with the
    // decode side, as the file keeps no second pair of them.
    let mut of_text = issued;
    (of_text.record.lease.chars, of_text.record.lease.bytes) = (400, 800);
    let mut sizes = Vec::new();
    of_text.with_held_pair(of(Side::Decode), |held, _| {
        sizes.push((held.decode.chars, held.decode.bytes));
    });
    assert_eq!(sizes, [(400, 800)]);

    // A single's lease is no pair's.
    let single = Sides {
        decode: None,
        ..issued
    };
    let mut calls = 0;
    single.with_held_pair(Said::Released(None, GONE), |_, _| calls += 1);
    assert_eq!(calls, 0);
}
