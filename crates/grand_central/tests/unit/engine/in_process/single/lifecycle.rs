//! The cases of the lifecycle: the sizes a lease states, the entry that
//! decides without a lease, the words the stages hear and the words a
//! placement says, a stage or a decision that panics, and a worker that
//! joins or leaves.

use super::*;

/// A whole pass's lease states the three sizes its request states: the
/// count of its token ids and its text's two lengths. A reader that
/// sizes a held decision of text reads the bytes over four off the
/// lease, the number its request's facts read: here a text of 400
/// two-byte characters.
#[test]
fn a_whole_passs_lease_states_the_requests_three_sizes() {
    // The stock sessions rule takes a word on a lease.
    let engine = preset("manual");
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let mut keyed = request();
    keyed.facts.session_key = Identity::hashed(9);
    (keyed.facts.prompt_chars, keyed.facts.prompt_bytes) = (400, 800);
    let size = keyed.facts.prompt_size();
    let asks = Asks::of(keyed, Some(&members));
    let lease = decided(&engine, &asks).lease.expect("a lease").lease;
    assert_eq!((lease.tokens, lease.chars, lease.bytes), (0, 400, 800));
    assert_eq!((lease.prompt_size(), size), (200, 200));
}

/// The entry for a host that files leases itself answers the decision
/// alone. Under a document whose stages take every word of a lease and
/// whose picker places a plain request by itself, the entry still makes
/// each request once and decides it whole: its pick names its
/// alternates, where the leasing entry answers the one place. No id is
/// taken and no stage hears an issue: the host tells the stages, under
/// the lease it files.
#[test]
fn a_decision_without_a_lease_takes_no_id_and_tells_no_stage() {
    let (engine, heard) = hearing_as(false, true, EventSet::LEASE);
    let document = engine.spec();
    let place = document.place_for(MODEL, Phase::Single);
    assert!(matches!(place, Place::Alone(_)), "{place:?}");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    assert!(asks.plain().is_some(), "a plain request");
    let mut last = None;
    for _ in 0..8 {
        let decision = engine.decide_without_lease(&asks, Instant::now());
        let (decision, _) = decision.expect("a decision");
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.targets.len(), 4, "the pick and its alternates");
        for target in pick.targets.iter() {
            let worker = &list[target.position as usize];
            assert_eq!(target.id, worker_id(worker.as_ref()));
        }
        last = Some(decision);
    }
    assert_eq!(asks.made.get(), 8, "each request is made once");
    assert!(heard.lock().is_empty(), "no stage heard an issue");
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");

    // The host files the pick's lease and tells the stages under it.
    let decision = last.expect("a decision");
    let pick = decision.pick().expect("a pick");
    let first = *pick.targets.first().expect("the target");
    let lease = Lease {
        id: LeaseId(77),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        booked: first.id,
        generation: pick.trace.generation,
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        issued_micros: 0,
        tokens: 0,
        chars: 0,
        bytes: 0,
    };
    engine.spec().engine().on_decision_issued(&lease, &decision);
    assert_eq!(heard.lock().as_slice(), [("issued", first.id, 0)]);
}

/// The entry makes the leasing entry's whole decision of the same
/// request, under every stock policy: two engines made alike, the
/// leasing one asked by a maker that states no plain request, name the
/// same workers in the same order, request after request, by the
/// request's seed and by a rotation's turns. With a load snapshot lent
/// both read it: a policy that scores by load hands the worker with the
/// shortest queue at every seed.
#[test]
fn a_decision_without_a_lease_is_the_leasing_entrys_whole_decision() {
    use crate::document::spec::PRESETS;

    let places = |decision: &Decision| -> Vec<u32> {
        let targets = decision.pick().map(|pick| pick.targets.iter());
        targets.map_or_else(Vec::new, |targets| {
            targets.map(|target| target.position).collect()
        })
    };
    let reports = (0..8)
        .map(|at| {
            let queued = if at == 5 { 0 } else { 90_000 };
            let url = format!("http://worker-{at}:8000");
            (url, one_rank_load(queued, 0.1, 1_000.0))
        })
        .collect();
    let snapshot = LoadSnapshot::from_loads_for_test(reports);
    for policy in PRESETS {
        for loads in [None, Some(Arc::clone(&snapshot))] {
            let (leasing, bare) = (preset(policy), preset(policy));
            let list = pool(8);
            let (theirs, ours) = (
                PublishedList::new(Arc::clone(&list)),
                PublishedList::new(Arc::clone(&list)),
            );
            for seed in 0..16 {
                let asked = |members| {
                    let mut seeded = request();
                    seeded.facts.seed = seed;
                    Asks {
                        loads: loads.clone(),
                        ..Asks::of(seeded, Some(members))
                    }
                };
                let whole = Whole(asked(&theirs));
                let decided = leasing.decide_under(GatewayId::LOCAL, &whole, Instant::now());
                let decided = decided.expect("an answer").whole();
                let whole = places(&decided.expect("a whole decision").0.decision);
                assert!(!whole.is_empty(), "{policy}, seed {seed}: a pick");
                let alone = bare.decide_without_lease(&asked(&ours), Instant::now());
                let alone = places(&alone.expect("a decision").0);
                assert_eq!(alone, whole, "{policy}, seed {seed}");
                if policy == "least_load" && loads.is_some() {
                    assert_eq!(alone[0], 5, "seed {seed}: the shortest queue");
                }
            }
        }
    }
}

/// A request that hands no list of its kind is decided over no worker,
/// as the one entry decides it: a single with no list, a pair's prefill
/// with none, and a pair's prefill over a single's list. A single over a
/// pair's two lists or an encode request's three is shed too. So is a
/// decode or an encode request that stands alone and hands no list. A
/// pair's decode sent on its own is refused.
#[test]
fn a_decision_without_a_lease_over_no_list_names_no_worker() {
    use smg_worker::{test_support::worker_with_model_aliases, worker::WorkerType};

    use crate::engine::{EncodeLists, EncoderFacts, PairAdmission, PairLists, PartnerClasses};

    #[derive(Debug)]
    struct Open;

    impl PairAdmission for Open {
        fn has_capacity(&self, _: usize) -> bool {
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    let engine = preset("round_robin");
    let document =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
    let paired = InProcess::new(AppliedSpec::validate(&document).expect("a valid document"));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let side = |worker_type| {
        let url = format!("http://{worker_type}:8000");
        let worker = worker_with_model_aliases(&url, MODEL, &[], worker_type);
        worker.set_status(WorkerStatus::Ready);
        PublishedList::new(Arc::from([worker]))
    };
    let [prefill, decode, encoders] =
        [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode].map(side);
    let classes = PartnerClasses::number(1, [&[0_u32][..]]);
    let pair = PairLists {
        prefill: prefill.lent(),
        decode: decode.lent(),
        classes: &classes,
        admission: &Open,
        prefill_places: None,
        decode_places: None,
        prefill_stalled: &[],
        prefill_held: &[],
        decode_stalled: &[],
        decode_held: &[],
    };
    let encode = EncodeLists {
        pair,
        encoders: encoders.lent(),
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: &[EncoderFacts {
            class: 1,
            ..EncoderFacts::default()
        }],
        prefill_classes: &[1],
    };
    let stating = |phase, members| {
        let mut stated = request();
        stated.facts.phase = phase;
        Asks::over(stated, members)
    };
    let none = (ShedReason::NoEndpoints, "none");
    let single = WorkerLists::Single(members.lent());
    let (both, three) = (WorkerLists::Pair(pair), WorkerLists::Encode(encode));
    for (phase, over, members, engine) in [
        (Phase::Single, "no list", WorkerLists::None, &engine),
        (Phase::Prefill, "no list", WorkerLists::None, &engine),
        (Phase::Decode, "no list", WorkerLists::None, &engine),
        (Phase::Encode, "no list", WorkerLists::None, &engine),
        (Phase::Prefill, "a single's list", single, &paired),
        (Phase::Single, "a pair's lists", both, &paired),
        (Phase::Single, "an encode request's lists", three, &paired),
    ] {
        let shed = engine.decide_without_lease(&stating(phase, members), Instant::now());
        let Ok((Decision::Shed(shed), None)) = shed else {
            panic!("{phase:?} over {over}: a shed, not {shed:?}");
        };
        assert_eq!(
            (shed.reason, shed.emptied_by),
            none,
            "{phase:?} over {over}"
        );
        let whole = Whole(stating(phase, members));
        let leased = engine.decide_under(GatewayId::LOCAL, &whole, Instant::now());
        let leased = leased.expect("an answer").whole().expect("no place").0;
        let Decision::Shed(theirs) = leased.decision else {
            panic!("{phase:?} over {over}: the one entry picks");
        };
        assert_eq!((theirs.reason, theirs.emptied_by), none);
    }
    let asks = stating(Phase::Decode, WorkerLists::None);
    let refused = paired.decide_without_lease(&asks, Instant::now());
    assert_eq!(refused.err(), Some(DecisionRefused::PairSide));
}

/// [`Hears`] as a stage of a host's own build that says it needs the
/// lifecycle's calls and names no word of its own.
#[derive(Debug)]
struct Declares(Hears);

impl Lifecycle for Declares {
    fn needs_lifecycle(&self) -> bool {
        true
    }

    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.0.on_decision_issued(lease, decision);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
        self.0.on_dispatched(lease, target, side, since);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.0.on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.on_released(lease, reason);
    }
}

impl Picker for Declares {
    fn name(&self) -> &'static str {
        "declares"
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
        self.0.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The words a stage of `engine` put into `heard` of two requests: one
/// sent to its first target and completed, and one given up unsent. A
/// stage panicked at `unheard` of the words, and no decision faulted.
fn told(engine: &InProcess, heard: &Heard, unheard: u64) -> Vec<&'static str> {
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let mut handle =
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let first = handle.take(0).expect("the first target");
    handle.dispatched(now);
    handle.completed(
        &Outcome {
            side: Side::Single,
            served: first.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
        now,
    );
    drop(handle);
    let unsent =
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    drop(unsent);
    assert_eq!((engine.faults(), engine.unheard()), (0, unheard));
    let words = heard.lock().iter().map(|heard| heard.0).collect();
    words
}

/// A stage hears a single's words from what the request's guard lends,
/// with no lease on file: the issue on the pick's first target, the
/// dispatch on the target taken under the index the word names, the
/// end under the lease moved there, and a release under the lease as
/// issued.
#[test]
fn the_stages_hear_a_singles_words_from_what_its_guard_lends() {
    let (engine, heard) = hearing(false);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let mut handle =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let first = handle.take(0).expect("the first target");
    let taken = handle.take(1).expect("an alternate");
    handle.dispatched(now);
    handle.completed(
        &Outcome {
            side: Side::Single,
            served: taken.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
        now,
    );
    drop(handle);
    let unsent =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let given_up = unsent
        .decision()
        .pick()
        .and_then(|pick| pick.targets.first().copied())
        .expect("the first target");
    drop(unsent);
    assert_eq!(
        *heard.lock(),
        [
            ("issued", first.id, 0),
            ("dispatched", taken.id, 1),
            ("completed", taken.id, 1),
            ("issued", given_up.id, 0),
            ("released", given_up.id, 0),
        ]
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A placement says a single's words as a guard driven by hand says
/// them, and names the list's own worker at the place of the target
/// taken. Four requests, each once through a placement and once
/// through a guard: one sent to an alternate and ended, one sent and
/// dropped, one dropped unsent, and one whose pick has no target at the
/// index taken, which makes no placement and gives its lease up.
#[test]
fn a_placement_says_a_singles_words_as_its_guard_says_them() {
    let (engine, heard) = hearing(false);
    let engine: &'static InProcess = Box::leak(Box::new(engine));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let ended = |side: Side, served: WorkerId| Outcome {
        side,
        served,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    let whole = || {
        let answer = answered(engine, &asks).whole();
        let (decided, later) = answer.expect("a whole decision");
        (decided, later, Instant::now())
    };
    // One request as a placement with the target at `index` taken, and
    // that target.
    let taken_at = |index: usize| {
        let (decided, later, now) = whole();
        let targets = decided.decision.pick().expect("a pick").targets;
        let taken = targets.as_slice().get(index).copied();
        let worker = Arc::clone(&list[taken.map_or(0, |taken| taken.position as usize)]);
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            now,
            index,
            worker,
            None,
        );
        (placement, taken)
    };
    let by_hand = || {
        let (decided, later, now) = whole();
        (
            DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now),
            now,
        )
    };
    // What the stage heard since the last call: each word with its
    // worker, and each word with its lease's index.
    let words = || {
        let heard = std::mem::take(&mut *heard.lock());
        let told: Vec<_> = heard.iter().map(|word| (word.0, word.2)).collect();
        (heard, told)
    };

    // Sent to an alternate and ended. Its host wrote another side and no
    // worker into the outcome.
    let (placement, alternate) = taken_at(1);
    let mut placement = placement.expect("a placement");
    let alternate = alternate.expect("a target");
    let at = alternate.position as usize;
    assert!(Arc::ptr_eq(&placement.worker, &list[at]));
    assert_eq!(
        (placement.id(), placement.decode_id()),
        (alternate.id, None)
    );
    assert!(placement.decode.is_none() && placement.routing_key.is_none());
    placement.routing_key = Some("a-key");
    let mut placement = placement.kept();
    assert_eq!(
        placement.routing_key, None,
        "a kept placement borrows nothing"
    );
    assert!(placement.owes_events());
    placement.sent(Side::Single);
    placement.sent(Side::Single);
    assert!(placement.owes_events(), "sent, and not ended");
    placement.ended(Side::Single, ended(Side::Decode, WorkerId(0)));
    assert!(!placement.owes_events());
    drop(placement);
    // Sent and dropped: a cancelled end.
    let mut placement = taken_at(0).0.expect("a placement");
    placement.sent(Side::Single);
    drop(placement);
    // Dropped unsent: a release.
    drop(taken_at(0).0.expect("a placement"));
    // No target at the index taken: no placement, and the lease is given
    // up.
    assert!(taken_at(9).0.is_none());
    let (heard_of_placements, through_placements) = words();

    let (mut guard, now) = by_hand();
    let taken = guard.take(1).expect("a target");
    guard.dispatched(now);
    guard.completed(&ended(Side::Single, taken.id), now);
    drop(guard);
    let (mut guard, now) = by_hand();
    guard.dispatched(now);
    drop(guard);
    drop(by_hand());
    let (mut guard, _) = by_hand();
    assert!(guard.take(9).is_none());
    drop(guard);
    assert_eq!(through_placements, words().1);
    assert_eq!(
        through_placements,
        [
            ("issued", 0),
            ("dispatched", 1),
            ("completed", 1),
            ("issued", 0),
            ("dispatched", 0),
            ("completed", 0),
            ("issued", 0),
            ("released", 0),
            ("issued", 0),
            ("released", 0),
        ]
    );
    // The dispatch names the target taken, and the end is told under
    // the lease moved there.
    let on_the_alternate = [heard_of_placements[1].1, heard_of_placements[2].1];
    assert_eq!(on_the_alternate, [alternate.id; 2]);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// [`Hears`] as a stage that also keeps the outcome of each end it is
/// told.
#[derive(Debug)]
struct Ends(Hears, Arc<Mutex<Vec<Outcome>>>);

impl Lifecycle for Ends {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.0.on_decision_issued(lease, decision);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
        self.0.on_dispatched(lease, target, side, since);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.0.on_completed(lease, outcome);
        self.1.lock().push(*outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.on_released(lease, reason);
    }
}

impl Picker for Ends {
    fn name(&self) -> &'static str {
        "ends"
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
        self.0.pick(facts, candidates, kept, scores, band, out);
    }
}

/// An engine whose one stage is [`Ends`], with what the stage hears
/// and the outcomes it keeps.
fn hearing_ends() -> (&'static InProcess, Heard, Arc<Mutex<Vec<Outcome>>>) {
    let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
    let kept = (Arc::clone(&heard), Arc::clone(&ends));
    let engine = picked_by("ends", move || {
        let hears = Hears {
            heard: Arc::clone(&kept.0),
            panics: false,
            alone: false,
            takes: EventSet::LEASE,
        };
        Box::new(Ends(hears, Arc::clone(&kept.1)))
    });
    (Box::leak(Box::new(engine)), heard, ends)
}

/// A single request's placement goes out under any side its host names
/// but an encode side, as it ends under any. A host that names its one
/// call a decode side or a prefill side says the request's dispatch:
/// the stage counts the request as sent while it runs, and a drop
/// behind the send is an end of class cancelled, not a release.
#[test]
fn a_single_requests_placement_is_sent_under_the_side_its_host_names() {
    let (engine, heard, ends) = hearing_ends();
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let words = || std::mem::take(&mut *heard.lock());
    for side in [Side::Decode, Side::Prefill, Side::Single] {
        let asks = Asks::of(request(), Some(&members));
        let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first().copied());
        let first = first.expect("a target");
        let worker = Arc::clone(&list[first.position as usize]);
        let now = Instant::now();
        let placement = crate::tracking::Placement::taken(
            engine.loaded(),
            decided,
            later,
            now,
            0,
            worker,
            None,
        );
        let mut placement = placement.expect("a placement");
        assert_eq!(words(), [("issued", first.id, 0)]);
        placement.sent(side);
        assert_eq!(words(), [("dispatched", first.id, 0)], "{side:?}");
        drop(placement);
        assert_eq!(words(), [("completed", first.id, 0)], "{side:?}");
        let cancelled = std::mem::take(&mut *ends.lock());
        let classes: Vec<TransportClass> = cancelled.iter().map(|end| end.class).collect();
        assert_eq!(classes, [TransportClass::Cancelled], "{side:?}");
    }
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode side's end says nothing through a placement, as its send
/// says nothing: no lease stands for such a side. A single request's
/// placement is told one before the request is sent and one behind its
/// send. No stage hears a word of either, and the request is still
/// owed its end. Its own end behind them is said once, under the
/// request's side and the worker of the target taken, whatever its
/// host wrote there.
#[test]
fn an_encode_sides_end_through_a_placement_says_nothing() {
    let (heard, ends) = (Heard::default(), Arc::new(Mutex::new(Vec::new())));
    let kept = (Arc::clone(&heard), Arc::clone(&ends));
    let engine = picked_by("ends", move || {
        let hears = Hears {
            heard: Arc::clone(&kept.0),
            panics: false,
            alone: false,
            takes: EventSet::LEASE,
        };
        Box::new(Ends(hears, Arc::clone(&kept.1)))
    });
    let engine: &'static InProcess = Box::leak(Box::new(engine));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
    let first = decided
        .decision
        .pick()
        .and_then(|pick| pick.targets.first().copied());
    let first = first.expect("a target");
    let worker = Arc::clone(&list[first.position as usize]);
    let now = Instant::now();
    let placement =
        crate::tracking::Placement::taken(engine.loaded(), decided, later, now, 0, worker, None);
    let mut placement = placement.expect("a placement");
    let ended = |side: Side| Outcome {
        side,
        served: WorkerId(77),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    let words = || std::mem::take(&mut *heard.lock());
    assert_eq!(words(), [("issued", first.id, 0)]);
    // Unsent: no dispatch is said for it, and no end.
    placement.ended(Side::Encode, ended(Side::Encode));
    assert!(placement.owes_events(), "the request has not ended");
    assert!(words().is_empty(), "an encode side's end is no word");
    placement.sent(Side::Single);
    assert_eq!(words(), [("dispatched", first.id, 0)]);
    // Sent: the request's lease stands.
    placement.ended(Side::Encode, ended(Side::Encode));
    assert!(placement.owes_events(), "the request has not ended");
    assert!(words().is_empty(), "an encode side's end is no word");
    assert!(ends.lock().is_empty());
    // The request's own end. Its host wrote another side there.
    placement.ended(Side::Single, ended(Side::Decode));
    assert!(!placement.owes_events());
    assert_eq!(words(), [("completed", first.id, 0)]);
    let told = Outcome {
        side: Side::Single,
        served: first.id,
        ..ended(Side::Decode)
    };
    assert_eq!(*ends.lock(), [told]);
    drop(placement);
    assert!(words().is_empty(), "the drop says nothing more");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A placement under no lease owes nothing and says nothing: a place's,
/// and a whole decision's under a document no stage of which takes a
/// word. It names the list's own worker, names the profile that picked
/// by its label, and takes its worker's id when it is asked, and no
/// lease's. The profile that picks is the document's second: a spare
/// one of another name stands ahead of it.
#[test]
fn a_placement_under_no_lease_owes_nothing_and_says_nothing() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    let spare = document.profiles["round_robin"].clone();
    document.profiles.insert("a_spare".into(), spare);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(spec)));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let ended = Outcome {
        side: Side::Single,
        served: WorkerId(0),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };

    let pick = placed(engine, &Asks::of(request(), Some(&members)));
    assert_eq!(pick.profile, 1, "the document's second profile picked");
    let worker = &list[pick.place as usize];
    let label = engine.loaded().label_at(pick.profile);
    let mut placement = crate::tracking::Placement::alone(Arc::clone(worker), label);
    assert!(!placement.owes_events());
    placement.sent(Side::Single);
    placement.ended(Side::Single, ended);
    assert!(Arc::ptr_eq(&placement.worker, worker));
    assert_eq!(placement.id(), worker_id(worker.as_ref()));
    assert_eq!(
        (placement.decode_id(), placement.profile),
        (None, "round_robin")
    );
    assert!(placement.decode.is_none() && placement.routing_key.is_none());

    // A retry is decided whole, and under this document with no lease.
    let tried = [worker_id(list[0].as_ref())];
    let mut retry = request();
    retry.facts.tried = &tried;
    let asks = Asks::of(retry, Some(&members));
    let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
    let picked = decided.decision.pick().expect("a pick");
    let first = picked.targets.first().expect("a target");
    let (profile, target) = (picked.trace.profile, *first);
    assert_eq!(profile, 1, "the document's second profile picked");
    let worker = &list[target.position as usize];
    let now = Instant::now();
    let placement = crate::tracking::Placement::taken(
        engine.loaded(),
        decided,
        later,
        now,
        0,
        Arc::clone(worker),
        None,
    );
    let mut placement = placement.expect("a placement");
    assert!(!placement.owes_events());
    placement.sent(Side::Single);
    placement.ended(Side::Single, ended);
    assert!(Arc::ptr_eq(&placement.worker, worker));
    let named = (placement.id(), placement.profile);
    assert_eq!(named, (target.id, "round_robin"));
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// [`Hears`] as a stage that keeps the span each dispatch states since
/// its decision.
#[derive(Debug)]
struct Spans(Hears, Arc<Mutex<Vec<u32>>>);

impl Lifecycle for Spans {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, since: u32) {
        self.1.lock().push(since);
    }
}

impl Picker for Spans {
    fn name(&self) -> &'static str {
        "spans"
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
        self.0.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A placement dates its lease at the instant its host hands, the
/// decision's, and reads the clock itself at a send: the span the
/// dispatch states counts from the one to the other.
#[test]
fn a_placements_dispatch_counts_from_the_decisions_instant() {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&spans);
    let engine = picked_by("spans", move || {
        let hears = Hears {
            heard: Heard::default(),
            panics: false,
            alone: false,
            takes: EventSet::LEASE,
        };
        Box::new(Spans(hears, Arc::clone(&kept)))
    });
    let engine: &'static InProcess = Box::leak(Box::new(engine));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
    let first = decided
        .decision
        .pick()
        .and_then(|pick| pick.targets.first());
    let worker = Arc::clone(&list[first.expect("a target").position as usize]);
    // The decision stood a second before its request went out.
    let second = Duration::from_secs(1);
    let decided_at = Instant::now().checked_sub(second);
    let decided_at = decided_at.expect("an earlier instant");
    let placement = crate::tracking::Placement::taken(
        engine.loaded(),
        decided,
        later,
        decided_at,
        0,
        worker,
        None,
    );
    let mut placement = placement.expect("a placement");
    placement.sent(Side::Single);
    drop(placement);
    let stated = spans.lock().clone();
    assert_eq!(stated.len(), 1, "one dispatch");
    assert!(stated[0] >= 1_000_000, "{stated:?}");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A caller ends its request with an outcome that names no side and no
/// worker ([`Outcome::reported`]), and adds what it measured. The
/// placement writes the worker of the target it took, and what the
/// caller measured stands.
#[test]
fn a_placement_ends_under_an_outcome_that_names_no_worker() {
    let (engine, _, ends) = hearing_ends();
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let (decided, later) = answered(engine, &asks).whole().expect("a whole decision");
    let picked = decided.decision.pick().expect("a pick");
    let first = *picked.targets.first().expect("a target");
    let worker = Arc::clone(&list[first.position as usize]);
    let now = Instant::now();
    let placement =
        crate::tracking::Placement::taken(engine.loaded(), decided, later, now, 0, worker, None);
    let mut placement = placement.expect("a placement");
    let measured = Outcome {
        tokens_out: 34,
        duration_micros: 5_000,
        ..Outcome::reported(200, TransportClass::Ok)
    };
    assert_eq!(measured.served, WorkerId(0), "no worker named");
    placement.ended(Side::Single, measured);
    let told = Outcome {
        served: first.id,
        ..measured
    };
    assert_eq!(*ends.lock(), [told]);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A word of a lease is told where a stage takes it and to nobody
/// where none does. One request is sent and completed and one is given
/// up unsent: a document whose one stage takes one kind of word hears
/// that kind of the two leases and no other. A completion and a release
/// are told apart.
#[test]
fn a_word_no_stage_takes_is_told_to_nobody() {
    let told_of = |kind: EventKind| {
        let (engine, heard) = hearing_as(false, false, EventSet::NONE.with(kind));
        told(&engine, &heard, 0)
    };
    assert_eq!(told_of(EventKind::Issued), ["issued", "issued"]);
    assert_eq!(told_of(EventKind::Dispatched), ["dispatched"]);
    assert_eq!(told_of(EventKind::Completed), ["completed"]);
    assert_eq!(told_of(EventKind::Released), ["released"]);
}

/// A stage of a host's own build that says it needs the lifecycle's
/// calls and names no word of its own is told every word on a lease:
/// the issue, the dispatch and the completion of a request sent, and
/// the issue and the release of one given up.
#[test]
fn a_stage_that_needs_the_calls_and_names_no_word_is_told_each() {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let engine = picked_by("declares", move || {
        Box::new(Declares(Hears {
            heard: Arc::clone(&kept),
            panics: false,
            alone: false,
            takes: EventSet::NONE,
        }))
    });
    let document = engine.in_force();
    assert_eq!(document.spec.engine().hears(), EventSet::LEASE);
    for &kind in EventKind::ALL {
        let named = EventSet::LEASE.contains(kind);
        assert_eq!(document.takes(kind), named, "{kind:?}");
    }
    assert_eq!(
        told(&engine, &heard, 0),
        ["issued", "dispatched", "completed", "issued", "released"]
    );
}

/// [`Hears`] as a stage that panics at one word of a lease, the kind it
/// names, and keeps each other word.
#[derive(Debug)]
struct StopsAt(Hears, EventKind);

impl StopsAt {
    /// Panics where `word` is the stage's own.
    fn stop(&self, word: EventKind) {
        assert!(self.1 != word, "this stage panics at that word");
    }
}

impl Lifecycle for StopsAt {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.stop(EventKind::Issued);
        self.0.on_decision_issued(lease, decision);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, since: u32) {
        self.stop(EventKind::Dispatched);
        self.0.on_dispatched(lease, target, side, since);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.stop(EventKind::Completed);
        self.0.on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.stop(EventKind::Released);
        self.0.on_released(lease, reason);
    }
}

impl Picker for StopsAt {
    fn name(&self) -> &'static str {
        "stops-at"
    }

    fn refuses_by_reports(&self) -> bool {
        self.0.refuses_by_reports()
    }

    fn picks_alone(&self) -> bool {
        self.0.picks_alone()
    }

    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        self.0.pick_alone(pool, keep)
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
        self.0.pick(facts, candidates, kept, scores, band, out);
    }
}

/// What [`told`] reads of an engine whose stage panics at `word`, with
/// `unheard` held panics counted and no fault. The stage's picker
/// places by itself where `alone`.
fn told_past(word: EventKind, alone: bool, unheard: u64) -> Vec<&'static str> {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let engine = picked_by("stops-at", move || {
        let hears = Hears {
            heard: Arc::clone(&kept),
            panics: false,
            alone,
            takes: EventSet::LEASE,
        };
        Box::new(StopsAt(hears, word))
    });
    told(&engine, &heard, unheard)
}

/// A stage that panics at a first decision's issue is held and counted
/// with the words a stage did not hear, on the whole pass and on the
/// short path of a picker that places by itself. The decision answers
/// its pick and its lease, so the guard tells the lease's dispatch and
/// its end as any lease's: of a request sent and one given up, the
/// stage hears every word but the two issues. Before, each request was
/// answered a fault.
///
/// The classic pass tells a policy nothing where a pick is issued: it
/// asks the policy for one pick
/// (`model_gateway/src/routers/common/placement.rs:1391`).
#[test]
fn a_stage_that_panics_at_a_first_issue_costs_the_request_nothing() {
    for alone in [false, true] {
        // The two held panics alone are counted.
        assert_eq!(
            told_past(EventKind::Issued, alone, 2),
            ["dispatched", "completed", "released"],
            "a picker that places by itself: {alone}"
        );
    }
}

/// A stage that panics at a later word of a lease is held under the
/// same catch and counted the same way, as before this catch was the
/// issue's too: the request goes on, and the stage hears every other
/// word of the request sent and of the one given up.
#[test]
fn a_stage_that_panics_at_a_later_word_of_a_lease_is_held_and_counted() {
    let past = |word| told_past(word, false, 1);
    assert_eq!(
        past(EventKind::Dispatched),
        ["issued", "completed", "issued", "released"]
    );
    assert_eq!(
        past(EventKind::Completed),
        ["issued", "dispatched", "issued", "released"]
    );
    assert_eq!(
        past(EventKind::Released),
        ["issued", "dispatched", "completed", "issued"]
    );
}

/// A decision that panics costs its request alone and is counted.
#[test]
fn a_decision_that_panics_is_refused_and_counted() {
    let (engine, heard) = hearing(true);
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    for faults in 1..=2 {
        assert_eq!(
            engine
                .decide_under(GatewayId::LOCAL, &asks, Instant::now())
                .err(),
            Some(DecisionRefused::Faulted)
        );
        assert_eq!(engine.faults(), faults);
    }
    assert!(heard.lock().is_empty());
}

/// A worker that joins or leaves is told to the document's stages under
/// the id a decision names it by, in the order its host says the two. A
/// stage that panics at the word is held and counted: the call returns.
#[test]
fn a_worker_that_joins_or_leaves_is_told_to_the_documents_stages() {
    let list = pool(2);
    let id = |at: usize| worker_id(list[at].as_ref());
    let (engine, heard) = hearing(false);
    engine.joined(list[0].url());
    engine.forget(list[1].url());
    assert_eq!(*heard.lock(), [("joined", id(0), 0), ("left", id(1), 0)]);
    assert_eq!(engine.unheard(), 0);
    let (engine, heard) = hearing(true);
    engine.joined(list[0].url());
    engine.forget(list[1].url());
    assert!(heard.lock().is_empty());
    assert_eq!(engine.unheard(), 2);
}
