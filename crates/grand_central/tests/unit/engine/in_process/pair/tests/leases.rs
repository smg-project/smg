//! The words of a pair's two sides, each under its own lease: a side's
//! issue, dispatch, end and release, what a guard and a placement say of
//! them, the entries that issue no lease, a stage that panics at an issue,
//! what the stages of a later document hear, and what a count of sends
//! takes of a side's dispatch.

use super::*;

/// The sizes the stages heard on this thread since the last read.
fn sizes() -> Vec<(&'static str, Side, u32)> {
    SIZES.with_borrow_mut(std::mem::take)
}

/// [`hearing`]'s document and the same text under the next generation,
/// both validated over one catalog: an engine keeps the stages of the
/// first under the second.
fn hearing_and_its_next() -> ([AppliedSpec; 2], Arc<Mutex<Vec<Word>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || Box::new(Hears(Arc::clone(&kept))))
        .expect("a free picker name");
    let under = |number: u64| {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        document.generation = number;
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "hears".into();
        let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
        spec.expect("a valid document")
    };
    ([under(3), under(4)], heard)
}

#[test]
fn a_pairs_issue_tells_each_side_once_under_its_own_lease() {
    let pair = decided(1);
    let facts = keyed();
    let issued = issue(
        pair.spec.engine(),
        &facts,
        &pair.decision,
        LeaseId(9),
        false,
    );
    let (lease, session) = issued.expect("a pair's lease");
    assert_eq!(
        (lease.id, lease.side, lease.booked, lease.target_index),
        (LeaseId(9), Side::Prefill, WorkerId(1), 0)
    );
    assert_eq!(
        (lease.generation, lease.tokens, lease.issued_micros),
        (3, 40, 7_000)
    );
    assert_eq!(lease.session, facts.session());
    assert_ne!(session, lease.session, "each side keeps its own session");
    assert_eq!(
        pair.words(),
        [
            ("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1),
            ("issued", Side::Decode, 101, 0, session, 7_000, 1),
        ]
    );
}

#[test]
fn a_decode_sent_later_is_not_told_with_its_prefill() {
    let pair = decided(1);
    let issued = issue(
        pair.spec.engine(),
        &keyed(),
        &pair.decision,
        LeaseId(9),
        true,
    );
    let (lease, _) = issued.expect("a pair's lease");
    assert_eq!(
        pair.words(),
        [("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1)]
    );
}

#[test]
fn a_decision_that_picks_no_pair_issues_nothing() {
    let pair = decided(1);
    let none = shed(ShedReason::NoEndpoints, WIRES, 0, 3);
    assert!(issue(pair.spec.engine(), &facts(), &none, LeaseId(1), false).is_none());
    assert!(pair.words().is_empty());
}

/// A placement says a pair's words as a guard driven by hand says
/// them, each side's under its own lease and the index of the target
/// taken. It writes into each end the worker that served, from that
/// target, whatever its host wrote there. Its two workers are the
/// lists' own objects at the target's places. An encode side's send
/// and end say nothing. A pair told such an end behind its two sends
/// still owes both sides. A pair whose host said those two alone and
/// let go is released whole, as a guard dropped unsent.
#[test]
fn a_placement_says_a_pairs_words_and_names_the_workers_that_served() {
    let host: &'static Host = Box::leak(Box::new(Host::of(1, 2)));
    let classes = every(1, 2);
    let asks = host.asks(&classes, false);
    let whole = || {
        let now = Instant::now();
        let answer = host.engine.decide_under(GatewayId::LOCAL, &asks, now);
        let (decided, later) = answer
            .expect("an answer")
            .whole()
            .expect("a whole decision");
        let taken = target_at(&decided.decision, 1);
        (decided, later, now, taken, taken.pair.expect("a pair"))
    };
    // What the stages heard since the last call: each word with its
    // side and its lease's index, and the worker each completion names.
    let words = || {
        let heard = std::mem::take(&mut *host.heard.lock());
        let told: Vec<_> = heard.iter().map(|word| (word.0, word.1, word.3)).collect();
        let ended = heard.iter().filter(|word| word.0 == "completed");
        let served: Vec<_> = ended.map(|word| (word.1, WorkerId(word.6))).collect();
        (told, served)
    };

    let (decided, later, now, taken, pair) = whole();
    let profile = decided.decision.pick().expect("a pick").trace.profile;
    assert_ne!(profile, 0, "the document's second profile picked");
    let prefill = &host.prefills[taken.position as usize];
    let decode = &host.decodes[pair.decode_position as usize];
    let placement = crate::tracking::Placement::taken(
        host.engine.loaded(),
        decided,
        later,
        now,
        1,
        Arc::clone(prefill),
        Some(Arc::clone(decode)),
    );
    let mut placement = placement.expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, prefill));
    assert!(Arc::ptr_eq(
        placement.decode.as_ref().expect("a decode"),
        decode
    ));
    let ids = (placement.id(), placement.decode_id());
    assert_eq!(ids, (taken.id, Some(pair.decode_id)));
    assert_eq!(placement.profile, "prefill");
    placement.sent(Side::Prefill);
    placement.sent(Side::Decode);
    placement.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
    // Its host names no worker and another side in each outcome.
    placement.ended(Side::Prefill, outcome(Side::Single, WorkerId(0)));
    assert!(placement.owes_events(), "the decode side has not ended");
    placement.ended(Side::Decode, outcome(Side::Single, WorkerId(0)));
    assert!(!placement.owes_events());
    drop(placement);
    let (decided, later, now, _, _) = whole();
    let unsent = crate::tracking::Placement::taken(
        host.engine.loaded(),
        decided,
        later,
        now,
        1,
        Arc::clone(prefill),
        Some(Arc::clone(decode)),
    );
    let mut unsent = unsent.expect("a placement");
    unsent.sent(Side::Encode);
    unsent.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
    drop(unsent);
    let (through_placements, served) = words();
    let named = [(Side::Prefill, taken.id), (Side::Decode, pair.decode_id)];
    assert_eq!(served, named);

    let (decided, later, now, _, _) = whole();
    let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
    let taken = guard.take(1).expect("a target");
    let decode = taken.pair.expect("a pair").decode_id;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, taken.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    let (decided, later, now, _, _) = whole();
    let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
    assert!(guard.take(1).is_some());
    drop(guard);
    assert_eq!(through_placements, words().0);
    let told: Vec<_> = through_placements.iter().map(|word| word.0).collect();
    // Each word is told of both sides: the pair sent and ended, then
    // the pair let go.
    let sent_and_ended = ["issued", "dispatched", "completed"].map(|word| [word; 2]);
    let let_go = ["issued", "released"].map(|word| [word; 2]);
    assert_eq!(told, [sent_and_ended.concat(), let_go.concat()].concat());
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
}

fn completed(side: Side, served: u64) -> Said {
    Said::Completed(outcome(side, WorkerId(served)))
}

/// A pick of one prefill (worker 1) with two decodes (workers 101 and
/// 102): the first target and an alternate, as the pair entry orders
/// them.
fn two_decodes() -> Decision {
    let mut targets = Targets::new();
    for (at, decode) in [(0, 101), (1, 102)] {
        targets.push(Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: at,
                decode_id: WorkerId(decode),
                ..Pair::default()
            }),
        });
    }
    Decision::Pick(Pick {
        targets,
        ..Pick::default()
    })
}

/// One word of a held pair told to the stages, the clock reading 9,900
/// and every worker still listed.
fn tell(engine: &Engine, held: &HeldDecision<'_>, word: Said) {
    lease_events::pair(&lease_events::of_host(engine, engine), held, word, |_| {
        9_900
    });
}

/// Each word of a pair, both sides issued with the prefill, the host
/// taking the alternate decode. A side's dispatch names the worker it
/// booked at its issue and the worker it goes to, and its end names
/// the worker it went to.
#[test]
fn each_word_of_a_pair_is_told_under_its_sides_lease() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let decision = two_decodes();
    let issued = issue(engine, &keyed(), &decision, LeaseId(9), false);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&decision, 1);
    let held = HeldDecision {
        decision: &decision,
        lease: &lease,
        index: 1,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    tell(engine, &held, dispatched(Side::Prefill, 1));
    tell(engine, &held, dispatched(Side::Decode, 1));
    assert_eq!(
        pair.words(),
        [
            ("dispatched", Side::Prefill, 1, 1, own, 7_000, 1),
            ("dispatched", Side::Decode, 101, 1, session, 7_000, 102),
        ]
    );
    tell(engine, &held, completed(Side::Prefill, 1));
    tell(engine, &held, completed(Side::Decode, 102));
    assert_eq!(
        pair.words(),
        [
            ("completed", Side::Prefill, 1, 0, own, 7_000, 1),
            ("completed", Side::Decode, 102, 1, session, 7_000, 102),
        ]
    );
    // A side a pair's lease does not hold is told to nobody.
    tell(engine, &held, completed(Side::Encode, 1));
    tell(engine, &held, dispatched(Side::Single, 0));
    assert!(pair.words().is_empty());
}

/// Each side's lease states its own side's session verdict. The
/// prefill's is the pick's. The decode's is the one the pair of the
/// pick's first target keeps, whatever the prefill's is, and a claim
/// names that pair's decode.
#[test]
fn each_side_of_a_pair_states_its_own_session_verdict() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let stated = |decode_verdict| {
        let mut targets = Targets::new();
        targets.push(Target {
            id: WorkerId(1),
            pair: Some(Pair {
                decode_id: WorkerId(101),
                decode_verdict,
                ..Pair::default()
            }),
            ..Target::default()
        });
        let decision = Decision::Pick(Pick {
            targets,
            session_verdict: SessionVerdict::Pinned,
            ..Pick::default()
        });
        let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
        let (prefill, _) = issued.expect("a pair's lease");
        let decode = issue_decode(engine, &keyed(), &decision, &prefill, 9_900);
        let decode = decode.expect("the decode side's lease");
        (prefill.session_verdict, decode.session_verdict)
    };
    let claimed = SessionVerdict::Claimed(WorkerId(101));
    for (kept, whole) in [
        (DecodeVerdict::NONE, SessionVerdict::None),
        (DecodeVerdict::PINNED, SessionVerdict::Pinned),
        (DecodeVerdict::RESPILL, SessionVerdict::Respill),
        (DecodeVerdict::CLAIMED, claimed),
    ] {
        assert_eq!(stated(kept), (SessionVerdict::Pinned, whole));
    }
}

#[test]
fn a_release_ends_the_side_it_names_and_with_none_both() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), false);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&pair.decision, 0);
    let held = HeldDecision {
        decision: &pair.decision,
        lease: &lease,
        index: 0,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    let prefill = ("released", Side::Prefill, 1, 0, own, 7_000, 0);
    let decode = ("released", Side::Decode, 101, 0, session, 7_000, 0);
    let gone = ReleaseReason::ClientGone;
    tell(
        engine,
        &held,
        Said::Released(None, ReleaseReason::CommitFailed),
    );
    assert_eq!(pair.words(), [prefill, decode]);
    tell(engine, &held, Said::Released(Some(Side::Decode), gone));
    assert_eq!(pair.words(), [decode]);
    tell(engine, &held, Said::Released(Some(Side::Prefill), gone));
    assert_eq!(pair.words(), [prefill]);
}

/// A decode that stood chosen goes out as its pick's with no call. Its
/// dispatch is its issue too: the stages hear the issue first, dated at
/// the dispatch, and both words name the decode it goes to.
#[test]
fn a_chosen_decode_sent_with_no_call_is_told_issued_at_its_dispatch() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let decision = two_decodes();
    let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let taken = target_at(&decision, 1);
    let held = HeldDecision {
        decision: &decision,
        lease: &lease,
        index: 1,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    tell(engine, &held, dispatched(Side::Decode, 1));
    assert_eq!(
        pair.words(),
        [
            ("issued", Side::Decode, 102, 1, session, 9_900, 1),
            ("dispatched", Side::Decode, 102, 1, session, 7_000, 102),
        ]
    );
}

/// A decode that stands chosen was issued to no stage, so its end with
/// its prefill tells the prefill's alone.
#[test]
fn a_decode_that_stands_chosen_ends_with_its_prefill_and_is_told_to_nobody() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&pair.decision, 0);
    let held = HeldDecision {
        decision: &pair.decision,
        lease: &lease,
        index: 0,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    tell(
        engine,
        &held,
        Said::Released(None, ReleaseReason::Abandoned),
    );
    tell(
        engine,
        &held,
        Said::Released(Some(Side::Decode), ReleaseReason::Abandoned),
    );
    tell(engine, &held, completed(Side::Decode, 101));
    assert_eq!(
        pair.words(),
        [("released", Side::Prefill, 1, 0, own, 7_000, 0)]
    );
}

/// The engine decides a pair whole for the guard that asks. A pick is
/// issued with both sides told, and each side's dispatch and end reach
/// the stages from what the guard lends. A request held to a wire no
/// worker of the lists is dialed on is declined under the wires' name.
#[test]
fn the_engine_decides_a_pair_whole_and_hears_each_sides_words_from_its_guard() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    let mut undialed = host.asks(&classes, false);
    undialed.0.constraints.wire_modes = WireModes::ZMQ;
    let decided = host.engine.decide_under(GatewayId::LOCAL, &undialed, now);
    match decided
        .expect("an answer")
        .whole()
        .expect("a whole decision")
        .0
        .decision
    {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, WIRES)
        ),
        other => panic!("not shed: {other:?}"),
    }
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
}

/// A pair's decision alone ([`decide`]) is the decision its leasing form
/// makes ([`decided`]): of two engines made alike, the one asked for the
/// decision alone names, call by call, the pair the other issues a lease
/// for, and behind a held target the decode the other issues. It takes
/// no lease id and tells no stage. A request held to a wire no worker
/// of the lists is dialed on is declined under the wires' name by the
/// decision alone, and the leasing form answers that decline with no
/// lease, takes no id for it and tells no stage.
#[test]
fn a_pairs_decision_alone_is_its_leasing_forms_and_takes_no_id_and_tells_no_stage() {
    let (alone, leasing) = (Host::of(2, 2), Host::of(2, 2));
    let classes = every(2, 2);
    let now = Instant::now();
    for _ in 0..4 {
        let mut asked = alone.asks(&classes, false);
        let mut leased = leasing.asks(&classes, false);
        let decision = decide(
            alone.engine.loaded(),
            &asked.1,
            &mut asked.0,
            None,
            None,
            now,
            |_| true,
            SessionCountedUnder::default,
        );
        let outcome = super::super::decided(
            leasing.engine.loaded(),
            &leased.1,
            &mut leased.0,
            None,
            None,
            now,
            |_| true,
            SessionCountedUnder::default,
            &|_| None,
        );
        assert!(outcome.lease.is_some(), "the leasing form issues its pick");
        assert_eq!(alone.picked(&decision), leasing.picked(&outcome.decision));
    }
    // A pair whose decode is sent later, and that decode decided again
    // behind the target each engine's first pick names.
    let mut asked = alone.asks(&classes, true);
    let mut leased = leasing.asks(&classes, true);
    let first = decide(
        alone.engine.loaded(),
        &asked.1,
        &mut asked.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
    );
    let outcome = super::super::decided(
        leasing.engine.loaded(),
        &leased.1,
        &mut leased.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
        &|_| None,
    );
    assert_eq!(alone.picked(&first), leasing.picked(&outcome.decision));
    let (ours, theirs) = (target_at(&first, 0), target_at(&outcome.decision, 0));
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &theirs,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let again = decide(
        alone.engine.loaded(),
        &asked.1,
        &mut asked.0,
        Some(&ours),
        None,
        call,
        |_| true,
        SessionCountedUnder::default,
    );
    let issued = super::super::decided(
        leasing.engine.loaded(),
        &leased.1,
        &mut leased.0,
        Some(&held),
        None,
        call,
        |_| true,
        SessionCountedUnder::default,
        &|_| None,
    );
    assert!(issued.lease.is_some(), "the leasing form issues the decode");
    assert_eq!(alone.picked(&again), leasing.picked(&issued.decision));
    assert_eq!(alone.picked(&again).0, alone.picked(&first).0);
    // A request held to a wire no worker of the lists is dialed on.
    let mut undialed = alone.asks(&classes, false);
    undialed.0.constraints.wire_modes = WireModes::ZMQ;
    match decide(
        alone.engine.loaded(),
        &undialed.1,
        &mut undialed.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
    ) {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, WIRES)
        ),
        other => panic!("not shed: {other:?}"),
    }
    // The leasing form answers that request the same decline, first and
    // behind the held decision, with no lease and no session.
    let mut refused = leasing.asks(&classes, true);
    refused.0.constraints.wire_modes = WireModes::ZMQ;
    for behind in [None, Some(&held)] {
        let declined = super::super::decided(
            leasing.engine.loaded(),
            &refused.1,
            &mut refused.0,
            behind,
            None,
            call,
            |_| true,
            SessionCountedUnder::default,
            &|_| None,
        );
        assert!(matches!(&declined.decision, Decision::Shed(shed) if shed.emptied_by == WIRES));
        assert!(declined.lease.is_none());
        assert_eq!(declined.decode_session, Session::NONE);
    }
    // The leasing form took an id a pair and told each side's issue, and
    // neither for a decline: the decision alone took none and told nobody.
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 5);
    assert_eq!(leasing.words().len(), 10);
    assert_eq!(alone.engine.leases.0.load(Ordering::Relaxed), 0);
    assert!(alone.words().is_empty(), "no stage heard a word");
}

/// The entry that issues no lease decides a pair as the leasing entry
/// does ([`InProcess::decide_without_lease`]). Of two engines made
/// alike it names, call by call, the targets the other issues a lease
/// for, and it takes no lease id and tells no stage. The decode asked
/// again behind the target a first decision names
/// ([`InProcess::decide_decode_without_lease`]) is the decode the
/// leasing entry answers a guard, over the lists as they stand at the
/// call: the decode the first decision named is not ready by then.
#[test]
fn a_pair_without_a_lease_is_decided_as_the_leasing_entry_decides_it() {
    let (bare, leasing) = (Host::of(2, 3), Host::of(2, 3));
    let classes = every(2, 3);
    let now = Instant::now();
    for call in 0..6 {
        let asks = bare.asks(&classes, false);
        let decision = bare.engine.decide_without_lease(&asks, now);
        let (decision, _) = decision.expect("a decision");
        let outcome = leased(&leasing, &leasing.asks(&classes, false), now);
        assert!(outcome.lease.is_some(), "the leasing entry issues its pick");
        assert!(!named(&decision).is_empty(), "call {call}: a pick");
        assert_eq!(named(&decision), named(&outcome.decision), "call {call}");
    }
    // A pair whose decode is sent later, then that decode decided again
    // behind the target each engine's first decision names.
    let asks = bare.asks(&classes, true);
    let first = bare.engine.decide_without_lease(&asks, now);
    let (first, _) = first.expect("a decision");
    let theirs = leasing.asks(&classes, true);
    let outcome = leased(&leasing, &theirs, now);
    assert_eq!(named(&first), named(&outcome.decision));
    let (prefill, chosen) = bare.picked(&first);
    for host in [&bare, &leasing] {
        host.decodes[chosen].set_status(WorkerStatus::NotReady);
    }
    let (ours, target) = (target_at(&first, 0), target_at(&outcome.decision, 0));
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let again = bare.engine.decide_decode_without_lease(&asks, &ours, call);
    let again = again.expect("a decision");
    let issued = leasing
        .engine
        .decide_decode_under(GatewayId::LOCAL, &theirs, &held, call);
    let issued = issued.expect("a decision");
    assert!(
        issued.lease.is_some(),
        "the leasing entry issues the decode"
    );
    assert_eq!(named(&again), named(&issued.decision));
    let (kept, decode) = bare.picked(&again);
    assert_eq!(kept, prefill, "the held prefill");
    assert_ne!(decode, chosen, "a decode that is ready at the call");
    // The leasing entry took an id a pair and told each side's issue.
    // The other took none and told nobody.
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 7);
    assert_eq!(leasing.words().len(), 14);
    assert_eq!(bare.engine.leases.0.load(Ordering::Relaxed), 0);
    assert!(bare.words().is_empty(), "no stage heard a word");
}

/// The entry that issues no lease keeps the test of a request's wires
/// that a pair's decision makes ([`decide`]), at the pair's first
/// decision and at its decode's behind a held target. The prefills
/// here are dialed on one wire and the decodes on another. A request
/// that states one of the two is declined under the wires' name: a
/// worker of the other list is dialed on a wire it does not state. A
/// request that states both is decided, and so is one that states a
/// third beside them and one that states none. The leasing entry
/// answers each request alike.
#[test]
fn a_pair_without_a_lease_keeps_the_test_of_its_wires() {
    let dialed = |name: &str, worker_type, mode| -> Arc<[Arc<dyn Worker>]> {
        let spec = |at: usize| WorkerSpec {
            worker_type,
            connection_mode: mode,
            ..WorkerSpec::new(format!("http://{name}-{at}:9000"))
        };
        let ready = |at| {
            let worker = <BasicWorkerBuilder>::from_spec(spec(at))
                .health_config(no_health_check())
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker) as Arc<dyn Worker>
        };
        (0..2).map(ready).collect()
    };
    let host = Host::over(
        hearing(),
        dialed("prefill", WorkerType::Prefill, ConnectionMode::Http),
        dialed("decode", WorkerType::Decode, ConnectionMode::Grpc),
    );
    let [prefills, decodes] = &host.lists;
    let wires = |list: &PublishedList| host.engine.members_of(list.lent()).wires();
    assert_eq!(
        (wires(prefills), wires(decodes)),
        (WireModes::HTTP, WireModes::GRPC)
    );
    let classes = every(2, 2);
    let now = Instant::now();
    let stating = |wires: WireModes| {
        let mut asks = host.asks(&classes, true);
        asks.0.constraints.wire_modes = wires;
        asks
    };
    let both = WireModes::HTTP.union(WireModes::GRPC);
    let first = host.engine.decide_without_lease(&stating(both), now);
    let held = target_at(&first.expect("a decision").0, 0);
    for wires in [WireModes::NONE, both, both.union(WireModes::ZMQ)] {
        let asks = stating(wires);
        let decision = host.engine.decide_without_lease(&asks, now);
        let (decision, _) = decision.expect("a decision");
        assert!(decision.pick().is_some(), "{wires:?}: {decision:?}");
        let again = host.engine.decide_decode_without_lease(&asks, &held, now);
        let again = again.expect("a decision");
        assert!(again.pick().is_some(), "{wires:?} again: {again:?}");
        assert!(leased(&host, &asks, now).decision.pick().is_some());
    }
    let declined = (ShedReason::PolicyDeclined, WIRES);
    for wires in [WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ] {
        let asks = stating(wires);
        let decision = host.engine.decide_without_lease(&asks, now);
        assert_eq!(shed_for(&decision.expect("a decision").0), declined);
        let again = host.engine.decide_decode_without_lease(&asks, &held, now);
        assert_eq!(shed_for(&again.expect("a decision")), declined);
        assert_eq!(shed_for(&leased(&host, &asks, now).decision), declined);
    }
}

/// A request that hands no pair's lists is decided over no worker by
/// the entries that issue no lease, as the leasing entries answer it: a
/// pair's prefill, and its decode asked again behind a held target.
#[test]
fn a_pair_without_a_lease_that_hands_no_lists_is_decided_over_no_worker() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let now = Instant::now();
    let asks = host.asks(&classes, true);
    let outcome = leased(&host, &asks, now);
    let target = target_at(&outcome.decision, 0);
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    // The request by itself is a maker that hands no list.
    let alone: &Request<'_> = &asks.0;
    let none = (ShedReason::NoEndpoints, "none");
    let prefill = host.engine.decide_without_lease(alone, now);
    assert_eq!(shed_for(&prefill.expect("a decision").0), none);
    assert_eq!(shed_for(&leased(&host, alone, now).decision), none);
    let again = host.engine.decide_decode_without_lease(alone, &target, now);
    assert_eq!(shed_for(&again.expect("a decision")), none);
    let issued = host
        .engine
        .decide_decode_under(GatewayId::LOCAL, alone, &held, now);
    let issued = issued.expect("an answer");
    assert_eq!(shed_for(&issued.decision), none);
    assert!(issued.lease.is_none(), "a shed issues nothing");
}

/// A guard lends its pair's session of the decode phase with the decode
/// side ([`HeldDecode::session`]), and every word of that side is told
/// under it: the session the side's issue stated, which is not the
/// prefill side's.
#[test]
fn a_guards_decode_side_is_told_under_the_session_of_its_issue() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    let decode = target_at(guard.decision(), 0).pair.expect("a pair");
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Decode, decode.decode_id), now);
    drop(guard);
    let heard = std::mem::take(&mut *host.heard.lock());
    let told = |side: Side| -> Vec<(&'static str, Session)> {
        let of_side = heard.iter().filter(|word| word.1 == side);
        of_side.map(|word| (word.0, word.4)).collect()
    };
    let own = asks.0.facts.session();
    let decode_side = told(Side::Decode);
    let issued = decode_side.first().expect("the decode's issue").1;
    assert_ne!(issued, own, "each side keeps its own session");
    assert_eq!(
        decode_side,
        [
            ("issued", issued),
            ("dispatched", issued),
            ("completed", issued),
        ]
    );
    assert!(told(Side::Prefill).iter().all(|word| word.1 == own));
}

/// A decode sent later stands chosen in its guard. Behind the prefill's
/// dispatch the engine decides it again over the host's lists of that
/// call. The side is issued then under the pair's lease, dated at the
/// call, and its later words bear the first decision's instant.
#[test]
fn the_engine_decides_a_chosen_decode_again_for_its_guard() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, true), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(guard.decode_chosen());
    let first = target_at(guard.decision(), 0);
    let chosen = host.picked(guard.decision()).1;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    // The decode the first pick named stops being ready behind it.
    host.decodes[chosen].set_status(WorkerStatus::NotReady);
    let call = now + Duration::from_millis(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    assert_eq!(host.picked(&again), (0, 1 - chosen));
    assert!(!guard.decode_chosen(), "the call's pick issued the side");
    let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
    guard.dispatched_decode(call);
    guard.completed(&outcome(Side::Prefill, first.id), call);
    guard.completed(&outcome(Side::Decode, decode), call);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
}

/// [`hearing`] with a prefill side that picks through [`StopsAt`]: at
/// the decode side's issue, or at the prefill side's where `prefill`.
/// The decode profile's stage, which keeps what it hears, is told each
/// word ahead of it.
fn stopping(prefill: bool) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    let stage: OwnPicker = if prefill {
        ("stops-at", || Box::new(StopsAt(Side::Prefill)))
    } else {
        ("stops-at", || Box::new(StopsAt(Side::Decode)))
    };
    hearing_with(Some(stage))
}

/// A stage that panics at the issue of a decode decided again is held
/// and counted with the words a stage did not hear. The call answers
/// its pick and its guard holds the side as issued, so the stage that
/// heard the issue ahead of the panic hears the side's dispatch and its
/// end. Before, the call answered a fault, its guard kept the first
/// decode as chosen, and no word ended the issue that stage had heard.
///
/// The gateway's own routing picks a pair's decode with its prefill and
/// has no stage that hears an issue
/// (`model_gateway/src/routers/common/placement.rs:1227`).
#[test]
fn a_stage_that_panics_at_a_decode_calls_issue_costs_the_call_nothing() {
    let host = Host::under(stopping(false), 1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, true), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(guard.decode_chosen());
    let first = target_at(guard.decision(), 0);
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    let call = now + Duration::from_millis(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
    assert!(!guard.decode_chosen(), "the call's pick issued the side");
    guard.dispatched_decode(call);
    guard.completed(&outcome(Side::Prefill, first.id), call);
    guard.completed(&outcome(Side::Decode, decode), call);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    // The held panic alone is counted, and no later word of the side.
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// A stage that panics at an issue of a pair's first decision is held
/// and counted with the words a stage did not hear. The decision
/// answers its pick and its lease, so the guard holds both sides as
/// issued: the stage that heard each issue ahead of the panic hears
/// each side's dispatch and its end. Before, the request was answered
/// a fault and no word ended the two issues that stage had heard.
///
/// The classic pass tells a policy nothing where a pick is issued: it
/// asks each side's policy for one pick
/// (`model_gateway/src/routers/common/placement.rs:2758`, `:2820`).
#[test]
fn a_stage_that_panics_at_a_pairs_first_issue_costs_the_request_nothing() {
    let host = Host::under(stopping(false), 1, 2);
    let classes = every(1, 2);
    // The decode is issued with its prefill: its issue is the one the
    // stage panics at, behind the prefill side's.
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(!guard.decode_chosen(), "the pick issued both sides");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    // The held panic alone is counted: no decision faulted.
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// Where the stage panics at the prefill side's issue, the one catch
/// holds both issues: the stage ahead of it heard the prefill's, and
/// no stage hears the decode side's. The decision answers its pick and
/// its lease all the same, so the guard holds both sides as issued,
/// and a stage hears the decode side's dispatch and its end with no
/// issue ahead of them, as the lifecycle's doc says a stage may.
#[test]
fn a_stage_that_panics_at_a_pairs_prefill_issue_leaves_the_decode_side_untold() {
    let host = Host::under(stopping(true), 1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(!guard.decode_chosen(), "the pick issued both sides");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// The prompt size each word of a pair states to a stage, through a
/// guard of `door`: the prefill side's words, then the decode side's.
/// The first decision is `first`'s. Behind the prefill's dispatch the
/// decode is decided again by `again`, or with none goes out as the
/// first decision left it. The decode side is then sent and ends, or is
/// `released` with no dispatch. The guard is dropped with its prefill
/// sent, so that side ends as cancelled.
fn sizes_by_side<D: DecisionEngine, M: RequestMaker>(
    door: &D,
    gateway: GatewayId,
    first: &M,
    again: Option<&M>,
    released: bool,
) -> [Vec<(&'static str, u32)>; 2] {
    // Words an earlier guard said on this thread are not this one's.
    SIZES.with_borrow_mut(Vec::clear);
    let now = Instant::now();
    let mut guard = DecisionGuard::decide_under(door, gateway, first, now).expect("a decision");
    guard.dispatched(now);
    if let Some(again) = again {
        let picked = guard.decide_decode_under(again, now);
        assert!(picked.is_ok_and(|decision| decision.pick().is_some()));
    }
    if released {
        guard.released_decode(ReleaseReason::ClientGone, now);
    } else {
        let decode = target_at(guard.decision(), 0).pair.expect("a pair");
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Decode, decode.decode_id), now);
    }
    drop(guard);
    let heard = sizes();
    let of = |side: Side| {
        let words = heard.iter().filter(|word| word.1 == side);
        words.map(|&(word, _, size)| (word, size)).collect()
    };
    [of(Side::Prefill), of(Side::Decode)]
}

/// A decode call may state another prompt size than the first decision
/// did. The side is issued under the call's size, and its dispatch, its
/// end and its release state that size too: a stage that books a size
/// at the issue reads the same size at every later word of the side.
/// The prefill side states the first decision's size behind the call
/// as before it. A decode that goes out with no call states the first
/// decision's size, issued at its own dispatch or with its prefill.
#[test]
fn a_decode_calls_side_states_the_calls_size_at_every_word() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let first = host.asks(&classes, true);
    let stated = first.0.facts.prompt_tokens;
    let mut again = host.asks(&classes, true);
    again.0.facts.prompt_tokens = stated + 7;
    let (door, local, called) = (&host.engine, GatewayId::LOCAL, stated + 7);
    let own = [
        ("issued", stated),
        ("dispatched", stated),
        ("completed", stated),
    ];
    let [prefill, decode] = sizes_by_side(door, local, &first, Some(&again), false);
    assert_eq!(prefill, own);
    assert_eq!(
        decode,
        [
            ("issued", called),
            ("dispatched", called),
            ("completed", called),
        ]
    );
    let [_, decode] = sizes_by_side(door, local, &first, Some(&again), true);
    assert_eq!(decode, [("issued", called), ("released", called)]);
    let [_, decode] = sizes_by_side(door, local, &first, None, false);
    assert_eq!(decode, own);
    let with = host.asks(&classes, false);
    let [_, decode] = sizes_by_side(door, local, &with, None, false);
    assert_eq!(decode, own);
}

/// A decode call may state another text than the first decision did.
/// The side is issued under the call's three sizes, and its dispatch,
/// its end and its release state them too: a stage that books a
/// request of text by its characters reads one length at every word of
/// the side, and a plugin is shown the size the call's own request
/// showed. Neither request here states a token count: the first is a
/// text of 400 characters in 800 bytes, the call's one of 900 in 1,800.
/// The prefill side states the first decision's lengths behind the
/// call as before it, and so does a decode that goes out with no call.
#[test]
fn a_decode_calls_side_states_the_calls_text_lengths_at_every_word() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let of_text = |chars, bytes| {
        let mut asks = host.asks(&classes, true);
        let facts = &mut asks.0.facts;
        (facts.prompt_tokens, facts.prompt_chars, facts.prompt_bytes) = (0, chars, bytes);
        asks
    };
    let (first, again) = (of_text(400, 800), of_text(900, 1_800));
    let (door, local) = (&host.engine, GatewayId::LOCAL);
    // What each word of a side stated: the prefill's, then the decode's.
    let heard = |again, released| {
        LENGTHS.with_borrow_mut(Vec::clear);
        let _ = sizes_by_side(door, local, &first, again, released);
        let heard = LENGTHS.with_borrow_mut(std::mem::take);
        let of = |side| {
            let words = heard.iter().filter(|word| word.0 == side);
            words.map(|word| word.1).collect::<Vec<_>>()
        };
        [of(Side::Prefill), of(Side::Decode)]
    };
    let (own, called) = ([400, 800, 200], [900, 1_800, 450]);
    let [prefill, decode] = heard(Some(&again), false);
    assert_eq!(prefill, [own; 3], "issued, sent and ended");
    assert_eq!(decode, [called; 3], "issued, sent and ended");
    let [_, decode] = heard(Some(&again), true);
    assert_eq!(decode, [called; 2], "issued and released");
    let [_, decode] = heard(None, false);
    assert_eq!(decode, [own; 3], "with no call");
}

/// A pair's decode asked again behind a new document is decided under
/// that document, and its side is leased as the pair's first lease
/// states it. No stage hears the issue: the pair's lease is the first
/// document's, and no later word of it reaches a stage, so a stage
/// that heard the issue would never hear the side end.
#[test]
fn a_decode_asked_again_behind_a_new_document_tells_no_stage_its_issue() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, true);
    let first = leased(&host, &asks, Instant::now());
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    host.words();
    let (next, later) = hearing_under(4, None);
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");

    let local = Default::default();
    let again = host
        .engine
        .decide_decode_under(local, &asks, &held, Instant::now());
    let again = again.expect("a decode decision");
    let side = again.lease.expect("the side's lease all the same").lease;
    assert_eq!((side.side, side.generation), (Side::Decode, 3));
    let decided_under = again.decision.trace().generation;
    assert_eq!(decided_under, 4, "the document in force decides");
    assert!(host.words().is_empty(), "the first document hears nothing");
    assert!(later.lock().is_empty(), "no stage is told the issue");
}

/// A pair's decode asked again behind a later document that kept the
/// first one's stages tells them the side's issue: they heard the
/// pair's lease issued, and they hear the side's dispatch. A stage
/// hears a lease's issue where it hears that lease's later words, and
/// nowhere else.
#[test]
fn a_decode_asked_again_behind_a_document_that_kept_its_stages_tells_them_its_issue() {
    let ([first, next], heard) = hearing_and_its_next();
    let host = Host::under((first, heard), 1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, true);
    let first = leased(&host, &asks, Instant::now());
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    host.words();
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");

    let local = Default::default();
    let now = Instant::now();
    let again = host.engine.decide_decode_under(local, &asks, &held, now);
    let again = again.expect("a decode decision");
    let side = again.lease.expect("the side's lease").lease;
    assert_eq!((side.side, side.generation), (Side::Decode, 3));
    let decided_under = again.decision.trace().generation;
    assert_eq!(decided_under, 4, "the document in force decides");
    host.engine
        .say(local, &held, dispatched(Side::Decode, 0), now);
    let told: Vec<_> = host.words().iter().map(|word| (word.0, word.1)).collect();
    let both = [("issued", Side::Decode), ("dispatched", Side::Decode)];
    assert_eq!(told, both, "the stages kept hear the side whole");
}

/// A pair's dispatch said behind a later document, for a pair the
/// document before placed, is counted by the counts of sends in force.
/// The later document built its counts new: a send is a fact of its
/// worker, whatever document decided it.
#[test]
fn a_pairs_older_send_is_counted_by_the_counts_in_force() {
    // The stock pair document whose sides count their sends. Under a
    // later number its filters stand in another order: built new.
    let counting = |number: u64| {
        let document = PolicySpec::pair_presets("least_load", "least_load");
        let mut document = document.expect("the pair presets");
        document.generation = number;
        if number > 3 {
            for profile in document.profiles.values_mut() {
                profile.filter.reverse();
            }
        }
        AppliedSpec::validate(&document).expect("a valid document")
    };
    let host = Host::under((counting(3), Arc::default()), 1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, false);
    let now = Instant::now();
    let first = leased(&host, &asks, now);
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    let applied = host.engine.apply(counting(4));
    applied.expect("a number above the standing one");

    // The prefill's sends, as each count in force reads them.
    let counted = || -> Vec<u32> {
        let row = Candidate {
            id: target.id,
            dp_rank: target.dp_rank,
            ..Default::default()
        };
        let facts = RequestFacts::default();
        let counts = &host.engine.standing_spec().engine().sent_credits;
        let counts = counts.iter();
        counts
            .map(|count| count.read(&facts, &row).requests)
            .collect()
    };
    assert_eq!(counted(), [0], "not said yet");
    let local = Default::default();
    host.engine
        .say(local, &held, dispatched(Side::Prefill, 0), now);
    assert_eq!(counted(), [1], "the count built new takes it");
}

/// A pair's pick of one prefill (worker 1) with two decodes (workers
/// 101 and 102), as the forming makes it of its two sides' lists: each
/// side placed by its stages (`None`), or by the sessions rule alone
/// under the rule's word.
fn placed(prefill: Option<RulePlaced>, decode: Option<RulePlaced>) -> Decision {
    let side = |kept: Option<RulePlaced>, ids: &[u64]| {
        let mut targets = Targets::new();
        for (at, &id) in ids.iter().enumerate() {
            targets.push(Target {
                position: at as u32,
                id: WorkerId(id),
                ..Target::default()
            });
        }
        if let Some(kept) = kept {
            targets.by_rule(kept);
        }
        targets
    };
    let prefill = side(prefill, &[1]);
    let mut targets = side(decode, &[101, 102]);
    let first = *prefill.first().expect("a prefill");
    targets.pair_under(&first, SessionVerdict::None, prefill.placing());
    Decision::Pick(Pick {
        targets,
        ..Pick::default()
    })
}

/// A count of sends takes the dispatch of a side by the side's own
/// word: of a side the stages placed, and of one the rule placed and
/// tells its profile of, whatever the other side states, and of no
/// side the rule keeps from the profile. The counts are read as the
/// engine prints them. No lease here states a session, so the session
/// table stands.
///
/// A gateway's policy counts a request it placed, and none its sticky
/// map placed (`model_gateway/src/policies/registry.rs:794-797`).
#[test]
fn a_count_of_sends_takes_a_sides_dispatch_by_the_sides_own_word() {
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let document = document.expect("the pair presets");
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = spec.engine();
    let counts = || format!("{engine:?}");
    let words = [None, Some(RulePlaced::Told), Some(RulePlaced::Untold)];
    for (prefill, decode) in words.into_iter().flat_map(|p| words.map(|d| (p, d))) {
        let decision = placed(prefill, decode);
        let sides = first_sides(&facts(), &decision, LeaseId(9));
        let (lease, _, session) = sides.expect("a pair's lease");
        let taken = first_target(&decision);
        let held = HeldDecision {
            decision: &decision,
            lease: &lease,
            index: 0,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: false,
            },
        };
        for (side, kept) in [(Side::Prefill, prefill), (Side::Decode, decode)] {
            let before = counts();
            tell(engine, &held, dispatched(side, 0));
            let counted = counts() != before;
            let told = kept != Some(RulePlaced::Untold);
            assert_eq!(counted, told, "{prefill:?}, {decode:?}: {side:?}");
        }
    }
}
