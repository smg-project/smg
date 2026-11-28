//! The decode asked again behind a held prefill: the lists, the places and
//! the demands its call is decided over, the decode it keeps, the ranks its
//! retry leaves, what it is shed under, and the document its request is
//! made from.

use super::*;

/// The second ask runs over the records of its own instant: the held
/// prefill's partners, with the room each has then. Its pick names the
/// held prefill, and its issue is told dated at the call under the
/// lease's own id.
#[test]
fn the_decode_asked_again_is_decided_over_the_held_prefills_partners() {
    let pair = decided(2);
    let facts = keyed();
    let issued = issue(pair.spec.engine(), &facts, &pair.decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    // At the call the first decode has no room and the second has one.
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |at| at == 1),
    );
    let again = decide_decode_records(
        pair.spec.engine(),
        &facts,
        (&Constraints::default(), unvetoed(None)),
        &pool(&decodes),
        pools(&prefills, &decodes, &classes),
        &prefill_alone(WorkerId(1)),
        &mut Scratch::default(),
    );
    assert_eq!(picked(&again), (1, 102));
    let decode = issue_decode(pair.spec.engine(), &facts, &again, &lease, 9_500);
    let decode = decode.expect("the decode side's lease");
    assert_eq!(
        (decode.id, decode.side, decode.booked, decode.session),
        (LeaseId(9), Side::Decode, WorkerId(102), session)
    );
    assert_eq!(
        decode.issued_micros, 7_000,
        "the lease keeps its first instant"
    );
    assert_eq!(
        pair.words(),
        [("issued", Side::Decode, 102, 0, session, 9_500, 1)]
    );
}

#[test]
fn a_prefill_that_left_its_list_sheds_the_second_ask() {
    let pair = decided(1);
    let (mut prefills, mut decodes) = lists(1, 1);
    let classes = every(1, 1);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let again = decide_decode_records(
        pair.spec.engine(),
        &facts(),
        (&Constraints::default(), unvetoed(None)),
        &pool(&decodes),
        pools(&prefills, &decodes, &classes),
        &prefill_alone(WorkerId(5)),
        &mut Scratch::default(),
    );
    match again {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
        other => panic!("not shed: {other:?}"),
    }
}

/// The decode asked again is held to the decode places that partner its
/// held prefill. Two prefills, each with its own decode, and the second
/// prefill held, which is not the first of its list: among both decode
/// places, and at the second alone, its decode is the second at every
/// turn. At the first place alone the places and the prefill's partners
/// meet in no decode, and the call is shed under the partners' name. So
/// it is over no place at all. The gateway's classic router asks no
/// decode again: it selects a request's pair whole, once, the decode
/// among the picked prefill's own partners
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`,
/// `:1548-1553`, at main `b8c15d40a`).
#[test]
fn the_decode_asked_again_is_held_to_the_places_that_partner_its_prefill() {
    let pair = decided(2);
    let (mut prefills, mut decodes) = lists(2, 2);
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let free = Constraints::default();
    let again = |seed: u64, places: &[u32]| {
        let facts = RequestFacts { seed, ..facts() };
        decide_decode_records(
            pair.spec.engine(),
            &facts,
            (&free, unvetoed(Some(places))),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(2)),
            &mut Scratch::default(),
        )
    };
    for seed in 0..24 {
        assert_eq!(picked(&again(seed, &[0, 1])), (2, 102), "seed {seed}");
        assert_eq!(picked(&again(seed, &[1])), (2, 102), "seed {seed}");
    }
    let nobody = (ShedReason::NoEndpoints, "partners");
    assert_eq!(shed_for(&again(1, &[0])), nobody);
    assert_eq!(shed_for(&again(1, &[])), nobody);
}

/// The decode asked again keeps the decode its first decision named
/// while that decode stands among the decodes' places of the call, and
/// picks among the places where it does not: no decode outside them is
/// handed, kept or not. One prefill and three decodes under round
/// robin, which keeps its pick. The gateway's classic router asks no
/// decode again: it selects a request's pair whole, once
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
/// `b8c15d40a`).
#[test]
fn the_decode_asked_again_keeps_its_first_decode_among_the_places() {
    let host = Host::under(presets("round_robin"), 1, 3);
    for seed in 0..12 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode").decode_position;
        let other = (named + 1) % 3;
        // A host states its places ascending.
        let (with, without) = ([named.min(other), named.max(other)], [other]);
        let among = [None, Some(&with[..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), among);
        assert_eq!(host.picked(&again).1, named as usize, "seed {seed}");
        let outside = [None, Some(&without[..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), outside);
        assert_eq!(host.picked(&again).1, other as usize, "seed {seed}");
    }
}

/// The decode asked again reads both lists at its own call, and keeps
/// the prefill that was sent.
#[test]
fn the_decode_asked_again_reads_the_hosts_lists_of_its_own_instant() {
    let host = Host::of(2, 2);
    let (sent, chosen) = host.picked(&host.decide(&OPEN, 3, None));
    let held = worker_id(host.prefills[sent].as_ref());
    // The decode the first pick named stops being ready behind it.
    host.decodes[chosen].set_status(WorkerStatus::NotReady);
    let again = host.decide(&OPEN, 4, Some(held));
    assert_eq!(host.picked(&again), (sent, 1 - chosen));
    // A prefill no list holds any more sheds the call.
    match host.decide(&OPEN, 5, Some(WorkerId(held.0 ^ 1))) {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
        other => panic!("not shed: {other:?}"),
    }
}

/// The decode asked again reads one prefill, the held one: its host's
/// prefill limit is asked of that prefill's place alone.
#[test]
fn the_decode_asked_again_asks_the_limit_of_its_held_prefill_alone() {
    /// A host with no limit that keeps the places its prefill limit
    /// was asked of.
    #[derive(Debug, Default)]
    struct Asked(RefCell<Vec<usize>>);

    impl PairAdmission for Asked {
        fn has_capacity(&self, prefill: usize) -> bool {
            self.0.borrow_mut().push(prefill);
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    let (host, asked) = (Host::of(3, 2), Asked::default());
    let held = worker_id(host.prefills[1].as_ref());
    let again = host.decide(&asked, 4, Some(held));
    assert_eq!(host.picked(&again).0, 1);
    assert_eq!(asked.0.take(), [1]);
    // The thread's buffer holds the one prefill record written.
    let written = BUFFERS.with(|buffers| buffers.borrow().rows[0].len());
    assert_eq!(written, 1);
}

/// A decode asked again where its host lists no decode is refused for
/// the partners it lacks: the held prefill stands in its list. So it is
/// behind a target that names the decode of its first decision. A
/// prefill that left is still named where a list holds no record.
///
/// The classic router places both sides of a pair in one call
/// (`placement.rs:1227`) and decides no decode again, so this answer
/// is the engine's own. Each refusal names its side: the decodes' for
/// the partners the call lacks, and the prefills' for the held prefill
/// that left its list.
#[test]
fn a_decode_asked_again_over_no_decode_is_not_shed_as_a_prefill_that_left() {
    let shed_under = |host: &Host, held: &Target| match host.decide_held(&OPEN, 4, Some(held)) {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by, shed.side),
        other => panic!("not shed: {other:?}"),
    };
    let host = Host::of(1, 0);
    let held = worker_id(host.prefills[0].as_ref());
    let no_partner = (ShedReason::NoEndpoints, "partners", Some(PairSide::Decode));
    assert_eq!(shed_under(&host, &prefill_alone(held)), no_partner);
    let first = first_target(&Host::of(1, 1).decide(&OPEN, 3, None));
    assert_eq!((first.id, first.pair.is_some()), (held, true));
    assert_eq!(shed_under(&host, &first), no_partner);
    let gone = (
        ShedReason::NoEndpoints,
        "prefill-gone",
        Some(PairSide::Prefill),
    );
    let left = prefill_alone(WorkerId(held.0 ^ 1));
    assert_eq!(shed_under(&host, &left), gone);
    assert_eq!(shed_under(&Host::of(0, 1), &prefill_alone(held)), gone);
}

/// Over no decode, as over two lists with records, a request that
/// states a demand is shed under the demand's stage. Over no prefill
/// the call is shed for the prefill that left, before a demand is read.
/// The classic router decides no decode again (`placement.rs:1227`):
/// this answer is the engine's own too.
#[test]
fn a_decode_asked_again_over_no_decode_sheds_a_stated_demand_under_its_stage() {
    let pair = decided(1);
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..Constraints::default()
    };
    let shed_under = |prefills: u8, decodes: u8| {
        let classes = every(usize::from(prefills), u32::from(decodes));
        let (mut prefills, mut decodes) = lists(u64::from(prefills), u64::from(decodes));
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let again = decide_decode_records(
            pair.spec.engine(),
            &facts(),
            (&demanded, unvetoed(None)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(1)),
            &mut Scratch::default(),
        );
        match again {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    };
    assert_eq!(shed_under(1, 0), (ShedReason::NoEndpoints, WIRES));
    assert_eq!(shed_under(0, 1), (ShedReason::NoEndpoints, "prefill-gone"));
}

/// The decode asked again behind a held prefill is held to the decodes
/// that meet the demand, and judges no prefill: the held one ran
/// already, whatever it states. Where no decode meets the demand the
/// call is shed under that demand's stage.
///
/// The classic router decides no decode again (`placement.rs:1227`, at
/// main `b8c15d40a`): this answer is the engine's own.
#[test]
fn a_decode_asked_again_is_held_to_the_decodes_that_meet_the_demand() {
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    // No prefill takes images, and the second decode alone does.
    let host = stating(1, 2, |side, at| FixedFacts {
        vision: (side, at) == (1, 1),
        ..FixedFacts::default()
    });
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    for seed in 0..6 {
        let again = host.decide_among(&OPEN, seed, Some(&held), pixels, [None, None]);
        assert_eq!(host.picked(&again), (0, 1));
    }
    // The pair's first decision judges its prefills.
    let first = host.decide_among(&OPEN, 0, None, pixels, [None, None]);
    assert_eq!(shed_of(&first), (ShedReason::NoEndpoints, "vision"));
    assert_eq!(side_of(&first), Some(PairSide::Prefill));
    // No decode takes images.
    let blind = stating(1, 2, |_, _| FixedFacts::default());
    let held = prefill_alone(worker_id(blind.prefills[0].as_ref()));
    let again = blind.decide_among(&OPEN, 0, Some(&held), pixels, [None, None]);
    assert_eq!(shed_of(&again), (ShedReason::NoEndpoints, "vision"));
    assert_eq!(
        side_of(&again),
        Some(PairSide::Decode),
        "no prefill is judged"
    );
}

/// Under round robin on both sides a pair's first decision takes the
/// decode rotation's turn, and the decode decided again is handed the
/// decode that decision named and keeps it. So requests that come one
/// at a time reach both decodes evenly, and each second answer is its
/// first. Handed the prefill alone, every second answer was the other
/// decode. A first decode that stopped being ready is left for another.
#[test]
fn the_decode_asked_again_keeps_its_first_decode_under_round_robin() {
    let host = Host::under(presets("round_robin"), 1, 2);
    let mut sent = [0; 2];
    for seed in 0..8 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode").decode_position as usize;
        let again = host.decide_held(&OPEN, seed, Some(&held));
        assert_eq!(host.picked(&again).1, named, "seed {seed}");
        sent[named] += 1;
    }
    assert_eq!(sent, [4; 2]);
    let held = first_target(&host.decide(&OPEN, 8, None));
    let named = held.pair.expect("a decode").decode_position as usize;
    let alone = host.decide(&OPEN, 8, Some(held.id));
    assert_eq!(host.picked(&alone).1, 1 - named, "no decode handed");
    let held = first_target(&host.decide(&OPEN, 9, None));
    let named = held.pair.expect("a decode").decode_position as usize;
    host.decodes[named].set_status(WorkerStatus::NotReady);
    let again = host.decide_held(&OPEN, 9, Some(&held));
    assert_eq!(host.picked(&again).1, 1 - named);
}

/// The decode asked again through the guard is handed the decode of the
/// target the guard holds. Under round robin each second answer is its
/// first, and requests one at a time reach both decodes evenly.
#[test]
fn the_guards_decode_call_keeps_its_first_decode() {
    let host = Host::under(presets("round_robin"), 1, 2);
    let classes = every(1, 2);
    let mut sent = [0; 2];
    for _ in 0..4 {
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        let named = host.picked(guard.decision()).1;
        guard.dispatched(now);
        let again = *guard.decide_decode_under(&asks, now).expect("a decision");
        assert!(!guard.decode_chosen(), "the call picked");
        assert_eq!(host.picked(&again).1, named);
        sent[named] += 1;
    }
    assert_eq!(sent, [2; 2]);
}

/// A host that takes another target of its pick holds that target, and
/// the guard hands it to the decode asked again. Under round robin the
/// answer's first decode is then the taken target's: for a target taken
/// before its prefill went out, and for a target's decode taken behind
/// the prefill. The pick's third target is taken: its decode is neither
/// the first target's nor the one the rotation's next turn names.
#[test]
fn the_guards_decode_call_keeps_the_decode_its_host_took() {
    let host = Host::under(presets("round_robin"), 1, 3);
    let classes = every(1, 3);
    for behind_the_prefill in [false, true] {
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        let first = host.picked(guard.decision()).1;
        let taken = if behind_the_prefill {
            guard.dispatched(now);
            guard.take_decode(2)
        } else {
            let taken = guard.take(2);
            guard.dispatched(now);
            taken
        };
        let taken = taken.and_then(|target| target.pair);
        let taken = taken.expect("a third target").decode_position as usize;
        assert_ne!(taken, first, "another decode than the first target's");
        let again = *guard.decide_decode_under(&asks, now).expect("a decision");
        assert_eq!(host.picked(&again).1, taken, "{behind_the_prefill}");
    }
}

/// A decode that is one rank of a data-parallel engine is kept by its
/// worker and its rank. Over an engine's two ranks each second answer is
/// the rank its first named, and both ranks are named.
#[test]
fn a_ranked_decode_is_kept_by_its_rank() {
    let rank = |rank: usize| -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            worker_type: WorkerType::Decode,
            ..WorkerSpec::new("grpc://decode:9000".to_string())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec)
            .dp_config(rank, 2)
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    };
    let prefill = worker("grpc://prefill:9000".to_string(), WorkerType::Prefill);
    let ranks = Arc::from(vec![rank(0), rank(1)]);
    let host = Host::over(presets("round_robin"), Arc::from(vec![prefill]), ranks);
    let mut named_ranks = [0; 2];
    for seed in 0..4 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode");
        let again = first_target(&host.decide_held(&OPEN, seed, Some(&held)));
        assert_eq!(again.pair, Some(named), "seed {seed}");
        named_ranks[usize::from(named.decode_dp_rank)] += 1;
    }
    assert_eq!(named_ranks, [2; 2], "both ranks are named");
}

/// The presets' own filters stand ahead of a kept decode. A first decode
/// the request excludes is left for another decode while one is kept,
/// and is handed again where it is the only decode.
#[test]
fn an_excluded_first_decode_is_kept_only_where_it_is_alone() {
    for (decodes, kept) in [(2, false), (1, true)] {
        let host = Host::under(presets("round_robin"), 1, decodes);
        let held = first_target(&host.decide(&OPEN, 0, None));
        let first = held.pair.expect("a decode");
        let excluded = [first.decode_id];
        let again = host.decide_excluding(&OPEN, 0, Some(&held), &excluded);
        let named = first.decode_position as usize;
        assert_eq!(host.picked(&again).1 == named, kept, "{decodes}");
    }
}

/// `passthrough` takes no turn, so its decode asked again is decided
/// again, whatever the call is handed. A pair's first decision names
/// the first decode of the list that has a room. A decode ahead of that
/// one has its room by the second call, and the answer is that decode.
#[test]
fn the_decode_asked_again_under_passthrough_is_the_lists_first() {
    let host = Host::under(presets("passthrough"), 1, 3);
    let no_room_ahead = Gates(|_| true, |at| at != 0);
    let held = first_target(&host.decide(&no_room_ahead, 0, None));
    assert_eq!(held.pair.expect("a decode").decode_position, 1);
    let again = host.decide_held(&OPEN, 0, Some(&held));
    assert_eq!(host.picked(&again).1, 0);
    let alone = host.decide(&OPEN, 0, Some(held.id));
    assert_eq!(host.picked(&alone).1, 0, "no decode handed");
}

/// The decode asked again looks its held prefill up before it judges
/// what the call states. A prefill that is in no list reads
/// `prefill-gone` under a demand, a subset hint, places of the request's
/// own, and decode places of the list. With the prefill listed, the
/// first three read their own stage and the fourth is decided among its
/// places. The gateway's classic router asks no decode again: it selects
/// a request's pair whole, once
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
/// `c738c8d3b`).
#[test]
fn a_decode_asked_for_a_prefill_that_left_reads_prefill_gone_whatever_it_states() {
    let pair = decided(2);
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let again = |held: u64, cx: &Constraints<'_>, places: Option<&[u32]>| {
        decide_decode_records(
            pair.spec.engine(),
            &facts(),
            (cx, unvetoed(places)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(held)),
            &mut Scratch::default(),
        )
    };
    let shed_of = |decision: Decision| match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    };
    let free = Constraints::default();
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..free
    };
    let hinted = Constraints {
        subset_hint: Some(2),
        ..free
    };
    let held_to = Constraints {
        places: Some(&[0]),
        ..free
    };
    let gone = (ShedReason::NoEndpoints, "prefill-gone");
    for (cx, own) in [
        (&demanded, (ShedReason::NoEndpoints, WIRES)),
        (&hinted, (ShedReason::PolicyDeclined, SUBSET_HINT)),
        (&held_to, (ShedReason::PolicyDeclined, single::PLACES)),
    ] {
        assert_eq!(shed_of(again(5, cx, None)), gone);
        assert_eq!(shed_of(again(1, cx, None)), own);
    }
    assert_eq!(shed_of(again(5, &free, Some(&[1]))), gone);
    assert_eq!(picked(&again(1, &free, Some(&[1]))), (1, 102));
}

/// The later call picks a decode behind one prefill.
/// Its filter scope opens once with a pool size of two and two decode rows.
#[test]
fn a_decode_call_shows_its_profile_the_decode_pools_facts() {
    use crate::engine::framework::DecisionScope;

    #[derive(Debug)]
    struct ObservePool(Arc<Mutex<Vec<(u32, usize)>>>);

    impl Filter for ObservePool {
        fn name(&self) -> &'static str {
            "observe-pool"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    impl DecisionScope for ObservePool {
        fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], pool: &PoolFacts<'_>) {
            self.0.lock().push((pool.pool_size, rows.len()));
        }

        fn close(&self) {}
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let opens = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("observe-pool", move || {
            Box::new(ObservePool(Arc::clone(&opens)))
        })
        .expect("a free filter name");
    let mut document =
        PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    document
        .profiles
        .get_mut("decode")
        .expect("the decode profile")
        .filter
        .push("observe-pool".into());
    let spec = AppliedSpec::validate_in(&document, &catalog, &[]).expect("a valid document");
    let host = Host::under((spec, Arc::default()), 1, 2);
    let classes = every(1, 2);
    let asks = host.asks(&classes, true);
    let now = Instant::now();
    let mut guard = DecisionGuard::decide_under(&host.engine, Default::default(), &asks, now)
        .expect("a pair decision");
    guard.dispatched(now);
    seen.lock().clear();

    let later = guard
        .decide_decode_under(&asks, now)
        .expect("a decode decision");
    assert!(later.pick().is_some(), "{later:?}");
    assert!(!guard.decode_chosen(), "the later call issued its pick");
    assert_eq!(seen.lock().as_slice(), [(2, 2)]);
}

/// A shed at the later call leaves the original decode chosen.
/// Only the prefill has a booking until dispatch books the decode.
#[test]
fn a_shed_at_the_call_leaves_the_side_chosen() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let mut asks = host.asks(&classes, true);
    let now = Instant::now();
    let local = Default::default();
    let mut guard =
        DecisionGuard::decide_under(&host.engine, local, &asks, now).expect("a pair decision");
    let original = *guard.decision();
    let target = first_target(&original);
    let decode = target.pair.expect("the chosen decode").decode_id;
    let before = *guard.lease().expect("the prefill lease");
    guard.dispatched(now);

    let call = now + Duration::from_millis(1);
    for (hint, reason) in [
        (0, ShedReason::NoEndpoints),
        (1, ShedReason::PolicyDeclined),
    ] {
        asks.0.constraints.subset_hint = Some(hint);

        // The raw answer exposes the shed and its lease field.
        let held = HeldDecision {
            decision: guard.decision(),
            lease: &before.lease,
            index: 0,
            target: guard.target(),
            decode: HeldDecode {
                session: decode_session(&asks.0.facts),
                tokens: before.lease.tokens,
                chars: before.lease.chars,
                bytes: before.lease.bytes,
                chosen: guard.decode_chosen(),
            },
        };
        let answer = host
            .engine
            .decide_decode_under(local, &asks, &held, call)
            .expect("a decode decision");
        assert_eq!(
            shed_for(&answer.decision),
            (reason, "subset-hint"),
            "hint {hint}"
        );
        assert!(answer.lease.is_none(), "hint {hint}");

        // The guard returns the pick it still holds.
        let standing = guard
            .decide_decode_under(&asks, call)
            .expect("the held pick");
        assert_eq!(standing, &original);
        assert_eq!(guard.target(), &target);
        assert!(guard.decode_chosen());
    }

    let after = guard.lease().expect("the held prefill lease");
    assert_eq!((after.lease, after.due), (before.lease, before.due));
    let at = before.lease.issued_micros;
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, target.id, at),
            ("dispatched", Side::Prefill, target.id, at),
        ]
    );

    guard.dispatched_decode(call);
    assert!(!guard.decode_chosen());
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
        ]
    );
}

/// The later decode call makes its request from the document it decides under.
/// It issues a decode lease and records that document's generation in the trace.
#[test]
fn a_decode_calls_request_is_made_from_the_document_the_call_decides_under() {
    struct FromDocument<'a> {
        asks: Asks<'a>,
        document: &'a AppliedSpec,
    }

    impl RequestMaker for FromDocument<'_> {
        fn model(&self) -> &str {
            self.asks.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            assert!(
                std::ptr::eq(spec, self.document),
                "the maker receives the deciding document"
            );
            let mut request = self.asks.make(spec);
            request.facts.seed = spec.generation();
            request
        }
    }

    impl LendsLists for FromDocument<'_> {
        fn workers(&self) -> WorkerLists<'_> {
            self.asks.workers()
        }
    }

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
    let local = Default::default();
    host.engine
        .say(local, &held, dispatched(Side::Prefill, 0), Instant::now());

    let maker = FromDocument {
        asks,
        document: host.engine.standing_spec(),
    };
    let answer = host
        .engine
        .decide_decode_under(local, &maker, &held, Instant::now())
        .expect("a decode decision");
    assert_eq!(
        answer.lease.expect("a decode lease").lease.side,
        Side::Decode
    );
    let trace = answer.decision.trace();
    assert_eq!(
        trace.seed,
        maker.document.generation(),
        "the call uses the seed its maker set"
    );
    assert_eq!(
        trace.generation, trace.seed,
        "the trace names the generation handed to the maker"
    );
}

/// The decode asked again behind a held prefill leaves every rank of
/// the engine its retry tried, as the pair's first decision does.
#[test]
fn the_decode_asked_again_leaves_every_rank_of_the_engine_it_tried() {
    use std::collections::BTreeSet;

    let (prefills, _) = ranked("prefill", WorkerType::Prefill, &[1]);
    let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
    let classes = every(prefills.len(), decodes.len() as u32);
    let held = prefill_alone(worker_id(prefills[0].as_ref()));
    // The first engine's second rank was tried.
    let tried = [worker_id(decodes[1].as_ref())];
    let host = Host::over(hearing(), prefills, decodes);
    let landed = |tried: &[WorkerId]| {
        let mut on_decodes = BTreeSet::new();
        for seed in 0..32 {
            let Asks(mut request, lists, _) = host.asks(&classes, true);
            request.facts.seed = seed;
            request.facts.attempt = 1;
            request.facts.tried = tried;
            let now = Instant::now();
            let again = lists.decide(&host.engine, None, now, &mut request, Some(&held));
            for target in named(&again) {
                let pair = target.pair.expect("a pair");
                on_decodes.insert(decode_engine[pair.decode_position as usize]);
            }
        }
        on_decodes
    };
    assert_eq!(landed(&[]), BTreeSet::from([0, 1]), "nobody tried");
    assert_eq!(
        landed(&tried),
        BTreeSet::from([1]),
        "the tried decode's engine"
    );
}
