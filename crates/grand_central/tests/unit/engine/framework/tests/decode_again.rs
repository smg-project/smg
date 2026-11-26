use super::*;

fn first_decode(decision: &Decision) -> Option<u32> {
    Some(decision.pick()?.targets.first()?.pair?.decode_position)
}

#[test]
fn a_decode_decided_alone_is_picked_among_the_partners_of_its_prefill() {
    let engine = strict_pair_engine();
    let mut prefills = classed_rows(&[1, 2]);
    prefills[1].dp_rank = 5;
    let mut decodes = classed_rows(&[1, 2, 1]);
    decodes[2].dp_rank = 3;
    let (ends, classes) = table_of(&[&[2], &[1]]);
    let decision = engine.decide_decode(
        &request(7),
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            table: PairTable::new(&ends, &classes),
            ..PairPools::default()
        },
        prefills[1].id,
        None,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let targets = &decision
        .pick()
        .expect("partners of the held prefill")
        .targets;
    assert_eq!(decodes_of(&decision), [0, 2]);
    for target in targets.iter() {
        assert_eq!(
            (target.id, target.position, target.dp_rank),
            (prefills[1].id, 1, 5)
        );
        let pair = target.pair.expect("a decode at each target");
        let row = &decodes[pair.decode_position as usize];
        assert_eq!((pair.decode_id, pair.decode_dp_rank), (row.id, row.dp_rank));
    }
}

#[test]
fn a_decode_that_joined_since_is_picked() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let prefills = classed_rows(&[1]);
    let mut decodes = classed_rows(&[1, 1]);
    let mut scratch = Scratch::new();
    let facts = request(7);
    let pool = PoolFacts::default();
    let cx = Constraints::default();
    let old = engine.decide_pair(
        &facts,
        &pool,
        PairPools {
            prefills: &prefills,
            decodes: &decodes[..1],
            ..PairPools::default()
        },
        &cx,
        &cx,
        &mut scratch,
    );
    assert_eq!(decodes_of(&old), [0]);
    decodes[0].in_flight = 7;
    let new = engine.decide_decode(
        &facts,
        &pool,
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        prefills[0].id,
        None,
        &cx,
        &mut scratch,
    );
    assert_eq!(first_decode(&new), Some(1));
    assert_eq!(new.trace().seed, facts.seed);
}

/// One request's two decisions over one prefill and `decodes`: the
/// pair's first, then its decode decided again, handed what `held`
/// makes of the worker and the rank the first named. `meanwhile` is
/// what happens to the decodes between the two, told that decode's
/// position. The position the first names, then those the second does.
fn decided_twice(
    engine: &Engine,
    decodes: &mut [Candidate],
    held: impl FnOnce((WorkerId, u16)) -> Option<(WorkerId, u16)>,
    meanwhile: impl FnOnce(&mut [Candidate], usize),
) -> (u32, Vec<u32>) {
    let prefills = classed_rows(&[1]);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let (facts, mut scratch) = (request(7), Scratch::new());
    let pools = PairPools {
        prefills: &prefills,
        decodes,
        ..PairPools::default()
    };
    let first = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
    let target = first.pick().and_then(|pick| pick.targets.first().copied());
    let target = target.expect("a pair");
    let named = target.pair.expect("a decode");
    meanwhile(&mut *decodes, named.decode_position as usize);
    let pools = PairPools {
        prefills: &prefills,
        decodes,
        ..PairPools::default()
    };
    let held = held((named.decode_id, named.decode_dp_rank));
    let again = engine.decide_decode(&facts, &pool, pools, target.id, held, &cx, &mut scratch);
    let again = again.pick().expect("a decode decided again");
    let again = again.targets.iter().filter_map(|target| target.pair);
    let again = again.map(|pair| pair.decode_position).collect();
    (named.decode_position, again)
}

/// A rotation is turned once a request where the decode decided again
/// is handed the decode of its first decision. Requests one at a time
/// over two decodes: the first decisions reach both evenly and each
/// second answer is its first. With none handed the call turns the
/// rotation its first decision turned: every first decision names one
/// decode, and every second the other, which then serves every request.
/// The two decodes are two workers, or one worker's two ranks: a first
/// decode is kept at its rank, and one at a rank no row holds is not.
/// A scorer beside the picker changes none of it: the picker reads no
/// score. The scorer still reads the decodes of a kept answer: behind a
/// busy first decode the idle one tops the trace.
#[test]
fn a_decode_decided_again_under_round_robin_keeps_its_first_decode() {
    type Held = fn((WorkerId, u16)) -> Option<(WorkerId, u16)>;
    let cases: [(bool, Held, bool); 4] = [
        (false, Some, true),
        (true, Some, true),
        (false, |_| None, false),
        (true, |(worker, rank)| Some((worker, rank + 2)), false),
    ];
    let cases = cases.map(|case| [(false, case), (true, case)]);
    for (scored, (ranks, held, kept)) in cases.into_iter().flatten() {
        let mut profile = round_robin_profile();
        if scored {
            profile.scorers = least_requests_profile().scorers;
        }
        let engine = pair_engine(profile);
        let mut decodes = classed_rows(&[1, 1]);
        if ranks {
            (decodes[1].id, decodes[1].dp_rank) = (decodes[0].id, 1);
        }
        let mut sent = [0; 2];
        for _ in 0..8 {
            let (first, again) = decided_twice(&engine, &mut decodes, held, |_, _| {});
            assert_eq!(again[0] == first, kept, "{ranks} {scored}");
            sent[again[0] as usize] += 1;
        }
        sent.sort_unstable();
        assert_eq!(sent, if kept { [4, 4] } else { [0, 8] }, "{ranks} {scored}");
    }
    let engine = pair_engine(Profile {
        scorers: least_requests_profile().scorers,
        ..round_robin_profile()
    });
    let (prefills, mut decodes) = (classed_rows(&[1]), classed_rows(&[1, 1]));
    decodes[0].in_flight = 5;
    let kept = engine.decide_decode(
        &request(7),
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        prefills[0].id,
        Some((decodes[0].id, 0)),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert_eq!(first_decode(&kept), Some(0));
    assert_eq!(kept.trace().top[0].0, decodes[1].id);
}

/// The kept answer names what a pick names over the same survivors:
/// the first decode leads, and the survivors behind it follow in their
/// order, wrapping, to a pick's count. Five decodes and a request at
/// each: four decodes from the first decision's on. Where no other
/// decode stays healthy, the first is the whole answer.
#[test]
fn a_kept_decode_leads_the_survivors_behind_it() {
    let engine = pair_engine(round_robin_profile());
    let mut decodes = classed_rows(&[1; 5]);
    for _ in 0..5 {
        let (first, again) = decided_twice(&engine, &mut decodes, Some, |_, _| {});
        let behind = (0..MAX_TARGETS as u32).map(|k| (first + k) % 5);
        assert_eq!(again, behind.collect::<Vec<_>>());
    }
    let alone = |rows: &mut [Candidate], at: usize| {
        for (i, row) in rows.iter_mut().enumerate() {
            row.health.healthy = i == at;
        }
    };
    let (first, again) = decided_twice(&engine, &mut decodes, Some, alone);
    assert_eq!(again, [first]);
}

/// A first decode that a filter refuses at the second call is not
/// kept: the picker picks among the decodes the filters keep, by one
/// turn of their rotation. Each request's first decode stops being
/// healthy behind its first decision and is healthy again behind its
/// second: both decisions still reach the three decodes evenly.
#[test]
fn a_first_decode_a_filter_refuses_is_picked_again_by_one_turn() {
    let engine = pair_engine(round_robin_profile());
    let mut decodes = classed_rows(&[1, 1, 1]);
    let (mut firsts, mut seconds) = ([0; 3], [0; 3]);
    for _ in 0..6 {
        let unhealthy = |rows: &mut [Candidate], at: usize| rows[at].health.healthy = false;
        let (first, again) = decided_twice(&engine, &mut decodes, Some, unhealthy);
        assert_ne!(again[0], first);
        decodes[first as usize].health.healthy = true;
        firsts[first as usize] += 1;
        seconds[again[0] as usize] += 1;
    }
    assert_eq!((firsts, seconds), ([2; 3], [2; 3]));
}

/// A profile whose picker reads its scores decides again, whatever it is
/// handed. Under least requests the second answer is the first while
/// nothing moved, and it follows the counts of its own instant away
/// from a first decode that is still healthy.
#[test]
fn a_decode_decided_again_under_a_scorer_follows_the_loads() {
    let engine = pair_engine(least_requests_profile());
    let mut decodes = classed_rows(&[1, 1]);
    decodes[1].in_flight = 3;
    let (first, again) = decided_twice(&engine, &mut decodes, Some, |_, _| {});
    assert_eq!((first, again[0]), (0, 0));
    let busy = |rows: &mut [Candidate], at: usize| rows[at].in_flight = 7;
    let (first, again) = decided_twice(&engine, &mut decodes, Some, busy);
    assert_eq!((first, again[0]), (0, 1));
}

/// `passthrough` takes no turn, so it decides again whatever it is
/// handed: the first healthy decode of the list as it stands at the
/// second call, with the healthy decodes behind it in the list's order.
/// The first decode of the list is not healthy at the pair's first
/// decision, which names the second, and is healthy by the second call.
#[test]
fn a_decode_decided_again_under_passthrough_is_the_lists_first() {
    type Held = fn((WorkerId, u16)) -> Option<(WorkerId, u16)>;
    let handed: [Held; 2] = [Some, |_| None];
    for held in handed {
        let engine = pair_engine(Profile {
            picker: Box::new(Passthrough),
            ..round_robin_profile()
        });
        let mut decodes = classed_rows(&[1, 1, 1]);
        decodes[0].health.healthy = false;
        let ahead = |rows: &mut [Candidate], _: usize| rows[0].health.healthy = true;
        let (first, again) = decided_twice(&engine, &mut decodes, held, ahead);
        assert_eq!((first, again), (1, vec![0, 1, 2]));
    }
}

#[test]
fn a_prefill_that_left_is_shed_by_name() {
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1]);
    let decodes = classed_rows(&[1]);
    let facts = request(7);
    let decision = engine.decide_decode(
        &facts,
        &PoolFacts {
            generation: 8,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        WorkerId(987),
        None,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert_eq!(
        shed_of(&decision),
        Some((ShedReason::NoEndpoints, "prefill-gone"))
    );
    assert_eq!(decision.trace().made_under(), Some(0));
    assert_eq!(
        (decision.trace().seed, decision.trace().generation),
        (facts.seed, 8)
    );
}

#[test]
fn a_decode_decided_alone_runs_no_prefill_stage() {
    let engine = Engine::new(
        vec![strict_profile(), least_requests_profile()],
        vec![Binding {
            decode_profile: Some(1),
            ..pair_binding()
        }],
        1,
        Needs::default(),
    );
    let mut prefills = classed_rows(&[1]);
    prefills[0].health.healthy = false;
    let mut decodes = classed_rows(&[1, 1]);
    decodes[0].in_flight = 7;
    let facts = request(7);
    assert_eq!(
        shed_of(&decide_pair(&engine, &prefills, &decodes)),
        Some((ShedReason::NoEndpoints, "healthy"))
    );
    let decision = engine.decide_decode(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            // This column is only read by prefill stages.
            prefill_affinity: &held(&[0, 0]),
            ..PairPools::default()
        },
        prefills[0].id,
        None,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("the decode runs on its own");
    assert_eq!(first_decode(&decision), Some(1));
    assert_eq!(pick.session_verdict, SessionVerdict::None);
}

#[test]
fn a_decode_call_checks_the_decode_view_and_binding() {
    #[derive(Debug)]
    struct Shown(Arc<Mutex<Vec<Phase>>>);

    impl Lifecycle for Shown {}

    impl DecisionScope for Shown {
        fn open(&self, facts: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
            if let Ok(mut phases) = self.0.lock() {
                phases.push(facts.phase);
            }
        }

        fn close(&self) {}
    }

    impl Filter for Shown {
        fn name(&self) -> &'static str {
            "shown"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    let phases = Arc::new(Mutex::new(Vec::new()));
    let engine = Engine::new(
        vec![Profile {
            filters: vec![Box::new(Shown(Arc::clone(&phases)))],
            ..least_requests_profile()
        }],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let prefills = classed_rows(&[1]);
    let decodes = classed_rows(&[1]);
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let facts = request(3);
    let cx = Constraints::default();
    let decide =
        |engine: &Engine, facts: &RequestFacts<'_>, pools: PairPools<'_>, cx: &Constraints<'_>| {
            engine.decide_decode(
                facts,
                &PoolFacts::default(),
                pools,
                prefills[0].id,
                None,
                cx,
                &mut Scratch::new(),
            )
        };
    let column = held(&[1, 2]);
    let bad_pool = PairPools {
        decode_affinity: &column,
        ..pools
    };
    let bad_req = RequestFacts {
        affinity: &column[..1],
        ..facts
    };
    let mask = Constraints {
        stalled: &[0],
        ..cx
    };
    for (decision, stage) in [
        (decide(&engine, &bad_req, pools, &cx), "affinity-column"),
        (decide(&engine, &facts, bad_pool, &cx), "affinity-column"),
        (decide(&strict_engine(), &facts, pools, &cx), "binding"),
        (decide(&engine, &facts, pools, &mask), "written-rows"),
    ] {
        assert_eq!(
            shed_of(&decision),
            Some((ShedReason::PolicyDeclined, stage))
        );
        // A decline ahead of the decode profile names none. One past it
        // names it.
        let under = (stage == "written-rows").then_some(0);
        assert_eq!(decision.trace().made_under(), under, "{stage}");
    }
    // A call whose prefill left the lists says so ahead of its mask: the
    // prefill is looked for before a constraint is read. The gateway's
    // classic router asks no decode again
    // (`model_gateway/src/routers/common/placement.rs:1227-1275`, at
    // main `c738c8d3b`).
    let left = engine.decide_decode(
        &facts,
        &PoolFacts::default(),
        pools,
        WorkerId(u64::MAX),
        None,
        &mask,
        &mut Scratch::new(),
    );
    assert_eq!(
        shed_of(&left),
        Some((ShedReason::NoEndpoints, "prefill-gone"))
    );
    assert!(phases.lock().expect("the phases shown").is_empty());
    assert!(decide(&engine, &facts, pools, &cx).pick().is_some());
    assert_eq!(*phases.lock().expect("the phases shown"), [Phase::Decode]);
}

/// The partner question asks the hard filters alone: a class whose one
/// decode a retry just tried keeps its prefill, and the soft filter then
/// steps aside.
#[test]
fn a_decode_a_retry_just_tried_still_partners_its_prefill() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let prefills = classed_rows(&[1]);
    let decodes = classed_rows(&[1]);
    let tried = [decodes[0].id];
    let decode_cx = Constraints {
        excluded: &tried,
        ..Constraints::default()
    };
    for table in [PairTable::default(), PairTable::new(&[1], &[1])] {
        let decision = engine.decide_pair(
            &RequestFacts {
                model: "m",
                ..RequestFacts::default()
            },
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                table,
                ..PairPools::default()
            },
            &Constraints::default(),
            &decode_cx,
            &mut Scratch::new(),
        );
        assert_eq!(decodes_of(&decision), [0], "{decision:?}");
    }
}
