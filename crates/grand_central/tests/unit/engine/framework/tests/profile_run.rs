use super::*;

/// One profile whose two ring scorers cut a request by two rules has no
/// one list of heads' places: the engine states no rule for it, and a
/// request under it is lent no ring. Two scorers of one rule state that
/// rule.
#[test]
fn a_profile_whose_ring_scorers_name_two_head_rules_states_none() {
    use crate::policies::{cache_aware::Gates, consistent_hashing::ConsistentHash};

    let depth_of = |depths: [u32; 2]| {
        let scorers = depths.map(|depth| {
            let ring = ConsistentHash::new(None);
            WeightedScorer {
                scorer: Box::new(ring.keyed_by_prefix(depth, Gates::default(), &[])),
                weight: 1.0,
            }
        });
        let profile = Profile {
            name: "ring".into(),
            scorers: scorers.into(),
            ..least_requests_profile()
        };
        let engine = Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
        let rule = engine.states_head_places("m", Phase::Single);
        rule.map(|rule| rule.depth)
    };
    assert_eq!(depth_of([256, 256]), Some(256));
    assert_eq!(depth_of([256, 128]), None);
}

#[test]
fn the_pick_orders_targets_best_first_and_names_the_survivors() {
    let engine = engine();
    let pool = [candidate(0, 5), candidate(1, 1), candidate(2, 3)];
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("a pick");
    let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(order, vec![1, 2, 0]);
    assert_eq!(pick.trace.survivors.get(), Some(3));
    assert_eq!(pick.trace.top[0].0, WorkerId(101));
    assert_eq!(pick.trace.top_len, 3);
}

#[test]
fn the_trace_lists_equal_scores_in_the_pools_order_and_keeps_the_three_best() {
    // Four idle workers behind one busy one: the three idle rows the
    // trace names come in the pool's order, and the busy row stays out.
    let engine = engine();
    let pool = [
        candidate(0, 3),
        candidate(1, 0),
        candidate(2, 0),
        candidate(3, 0),
        candidate(4, 0),
    ];
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    );
    let trace = decision.trace();
    assert_eq!(trace.top_len, 3);
    let named: Vec<WorkerId> = trace.top.iter().map(|(id, _)| *id).collect();
    assert_eq!(named, vec![WorkerId(101), WorkerId(102), WorkerId(103)]);
    assert!(trace.top.iter().all(|(_, score)| *score == 1.0));
}

#[test]
fn a_soft_filter_steps_aside_when_it_would_empty_the_pool() {
    let engine = engine();
    let pool = [candidate(0, 1)];
    let tried = [WorkerId(100)];
    let facts = RequestFacts {
        model: "m",
        tried: &tried,
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    );
    assert_eq!(
        decision
            .pick()
            .and_then(|p| p.targets.first().map(|t| t.position)),
        Some(0)
    );
}

#[test]
fn a_hard_filter_that_empties_the_pool_sheds_with_its_name() {
    let engine = engine();
    let mut pool = [candidate(0, 1)];
    pool[0].health.healthy = false;
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    match engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    ) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
            assert_eq!(shed.emptied_by, "healthy");
            assert_eq!(shed.trace.made_under(), Some(0));
        }
        other => panic!("an unhealthy pool answered {other:?}"),
    }
}

/// A stage that reads the affinity column as a filter alone, or as the
/// picker alone, counts in the profile's needs as one that scores does:
/// a host that fills the column where it is declared leaves it empty
/// only where nothing reads it.
#[test]
fn a_profiles_needs_fold_its_filters_and_its_picker() {
    #[derive(Debug)]
    struct Reader;

    impl Lifecycle for Reader {}

    impl Filter for Reader {
        fn name(&self) -> &'static str {
            "reader"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> Needs {
            Needs {
                affinity: true,
                ..Needs::default()
            }
        }
    }

    impl Picker for Reader {
        fn name(&self) -> &'static str {
            "reader"
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
            Best.pick(facts, candidates, kept, scores, band, out);
        }

        fn needs(&self) -> Needs {
            Needs {
                affinity: true,
                ..Needs::default()
            }
        }
    }

    let as_filter = Profile {
        filters: vec![Box::new(Reader)],
        ..least_requests_profile()
    };
    let as_picker = Profile {
        picker: Box::new(Reader),
        ..least_requests_profile()
    };
    assert!(!least_requests_profile().needs().affinity);
    assert!(as_filter.needs().affinity, "a filter's needs");
    assert!(as_picker.needs().affinity, "the picker's needs");
    // The engine's summary holds them however the engine was built: a
    // caller that hands no needs of its own still gets its stages'.
    for (profile, bound_as) in [(as_filter, "a filter"), (as_picker, "the picker")] {
        let engine = Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
        assert!(engine.needs().affinity, "the summary, bound as {bound_as}");
        assert!(
            engine.needs_for("m", Phase::Single).affinity,
            "the decision's needs, bound as {bound_as}"
        );
    }
}

#[test]
fn a_retry_passes_over_the_tried_worker_while_others_remain() {
    let engine = engine();
    let pool = [candidate(0, 1), candidate(1, 5)];
    let tried = [WorkerId(100)];
    let facts = RequestFacts {
        model: "m",
        attempt: 1,
        tried: &tried,
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("the untried worker is picked");
    let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(
        order,
        vec![1],
        "the tried worker is dropped while another remains"
    );
    assert_eq!(pick.trace.survivors.get(), Some(1));
    assert_eq!(pick.trace.made_under(), Some(0));
}

#[test]
fn a_request_no_binding_covers_is_declined_by_the_binding() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![Binding {
            model: "pd/*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let facts = RequestFacts {
        model: "chat",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    match engine.decide(
        &facts,
        &PoolFacts::default(),
        &[candidate(0, 0)],
        &Constraints::default(),
        &mut scratch,
    ) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::PolicyDeclined);
            assert_eq!(shed.emptied_by, "binding");
            assert_eq!(shed.trace.survivors.get(), Some(0), "no pool was read");
            assert_eq!(shed.trace.made_under(), None);
        }
        other => panic!("an unbound model answered {other:?}"),
    }
}

/// The scorers are handed the set the picker compares and no other
/// survivor: the two drawn under a two-draw picker, the fewest in flight
/// under count-first, every survivor under a picker that does not
/// narrow. The trace counts the filters' survivors all the same, and its
/// top rows are of the set compared.
#[test]
fn the_scorers_are_handed_the_set_the_picker_compares() {
    // Five workers up, the first two holding the fewest, and one down.
    let mut pool = [
        candidate(0, 1),
        candidate(1, 1),
        candidate(2, 3),
        candidate(3, 3),
        candidate(4, 3),
        candidate(5, 0),
    ];
    pool[5].health.healthy = false;
    let run = |picker: Box<dyn Picker>, seed: u64| {
        let (engine, handed) = probed(picker, None);
        let facts = RequestFacts {
            model: "m",
            seed,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let pick = *decision.pick().expect("a pick");
        let rows = handed.lock().expect("the probe's rows").clone();
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        let top: Vec<u64> = pick.trace.top[..usize::from(pick.trace.top_len)]
            .iter()
            .map(|(id, _)| id.0 - 100)
            .collect();
        assert_eq!(pick.trace.survivors.get(), Some(5), "the filters' count");
        (rows, order, top)
    };
    let mut pairs = std::collections::BTreeSet::new();
    for seed in 0..64u64 {
        let (rows, order, top) = run(Box::new(PowerOfTwo), seed);
        assert_eq!(rows.len(), 2, "seed {seed}: the two drawn, {rows:?}");
        assert!(rows[0] != rows[1] && rows.iter().all(|&row| row < 5));
        let (better, other) = (rows[0].max(rows[1]), rows[0].min(rows[1]));
        assert_eq!(order, [better, other], "seed {seed}: the later row leads");
        assert_eq!(top, [u64::from(better), u64::from(other)], "seed {seed}");
        pairs.insert((rows[0], rows[1]));

        let (rows, order, top) = run(Box::new(PowerOfTwoCountFirst), seed);
        assert_eq!(rows.len(), 2, "seed {seed}: the two drawn, {rows:?}");
        let count = |row: u32| pool[row as usize].in_flight;
        let lead = *rows
            .iter()
            .min_by_key(|&&row| (count(row), std::cmp::Reverse(row)))
            .expect("two rows");
        assert_eq!(
            order[0], lead,
            "seed {seed}: the fewer in flight, then the later"
        );
        assert_eq!(top.len(), 2, "seed {seed}");

        let (rows, order, top) = run(Box::new(CountFirst), seed);
        assert_eq!(rows.len(), 2, "seed {seed}: {rows:?}");
        assert!(
            rows.contains(&0) && rows.contains(&1),
            "the two that hold one"
        );
        assert_eq!(
            order[..2],
            [1, 0],
            "seed {seed}: the later of the fewest leads"
        );
        assert_eq!(order.len(), 4, "seed {seed}: two passed over follow");
        assert!(order[2] != order[3] && order[2..].iter().all(|row| (2..5).contains(row)));
        assert_eq!(top, [1, 0], "seed {seed}");

        let (rows, order, top) = run(Box::new(Best), seed);
        assert_eq!(rows, [0, 1, 2, 3, 4], "seed {seed}: every survivor");
        assert_eq!(order, [4, 3, 2, 1], "seed {seed}");
        assert_eq!(top, [4, 3, 2], "seed {seed}");
    }
    assert!(pairs.len() > 4, "the draw moves the pair: {pairs:?}");
}

/// A picker whose narrowing names more rows than the survivors hold, of
/// either kind, costs no decision its process: the engine compares what
/// there is and lets what there is follow.
#[test]
fn a_picker_that_claims_more_than_the_survivors_is_held_to_them() {
    /// The first row compared leads; the narrowing claims what it holds.
    #[derive(Debug)]
    struct Claims(Shortlist);
    impl Lifecycle for Claims {}
    impl Picker for Claims {
        fn name(&self) -> &'static str {
            "claims"
        }
        fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &mut [u32]) -> Shortlist {
            self.0
        }
        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(&row) = kept.first() {
                out.push(target_of(&candidates[row as usize]));
            }
        }
    }
    let pool = [candidate(0, 1), candidate(1, 2), candidate(2, 3)];
    let all = Shortlist {
        compared: usize::MAX,
        alternates: usize::MAX,
    };
    let one = Shortlist {
        compared: 1,
        alternates: usize::MAX,
    };
    for (claim, compared, order) in [(all, vec![0, 1, 2], vec![0]), (one, vec![0], vec![0, 1, 2])] {
        let (engine, handed) = probed(Box::new(Claims(claim)), None);
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let pick = decision.pick().expect("a pick");
        assert_eq!(
            *handed.lock().expect("the probe's rows"),
            compared,
            "{claim:?}"
        );
        let targets: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(targets, order, "{claim:?}");
    }
}

/// Alternates follow a pick and never stand in for one. A picker that
/// compares one worker, keeps the rest as alternates and pushes no
/// target has declined: the decision is a shed that names the picker,
/// under the profile run and under a sessions rule that delegates, and
/// no worker it passed over is picked in its place.
#[test]
fn a_narrowing_picker_that_declines_sheds_whatever_it_passed_over() {
    /// Compares the first survivor, keeps the rest and declines.
    #[derive(Debug)]
    struct Declines;
    impl Lifecycle for Declines {}
    impl Picker for Declines {
        fn name(&self) -> &'static str {
            "declines"
        }
        fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], kept: &mut [u32]) -> Shortlist {
            Shortlist {
                compared: 1,
                alternates: kept.len().saturating_sub(1),
            }
        }
        fn pick(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            _: &mut Targets,
        ) {
        }
    }
    let pool = [candidate(0, 1), candidate(1, 2), candidate(2, 3)];
    for sessions in [None, Some(SESSIONS)] {
        let keyed = sessions.is_some();
        let site = if keyed {
            "a delegating rule"
        } else {
            "the profile run"
        };
        let (engine, handed) = probed(Box::new(Declines), sessions);
        let decision = if keyed {
            decide_keyed(&engine, &pool, 41, 1_000)
        } else {
            let facts = RequestFacts {
                model: "m",
                ..RequestFacts::default()
            };
            engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        assert_eq!(*handed.lock().expect("the probe's rows"), [0], "{site}");
        match decision {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined, "{site}");
                assert_eq!(shed.emptied_by, "declines", "{site}");
            }
            picked => panic!("{site}: a decline answered {picked:?}"),
        }
    }
}

#[test]
fn bindings_match_globs_and_phases_in_order() {
    let b = Binding {
        model: "pd/*".into(),
        phases: vec![Phase::Prefill],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: None,
    };
    assert!(b.matches("pd/a", Phase::Prefill));
    assert!(!b.matches("pd/a", Phase::Single));
    assert!(!b.matches("x/a", Phase::Prefill));
    let any = Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: None,
    };
    assert!(any.matches("anything", Phase::Decode));
    let suffix = Binding {
        model: "*-chat".into(),
        ..any
    };
    assert!(suffix.matches("x-chat", Phase::Single));
    assert!(!suffix.matches("x-code", Phase::Single));
}

#[test]
fn a_profile_band_is_the_weighted_sum_of_its_scorers_bands() {
    /// A scorer that writes equal scores and publishes a wide band.
    #[derive(Debug)]
    struct Wide;
    impl Lifecycle for Wide {}
    impl Scorer for Wide {
        fn name(&self) -> &'static str {
            "wide"
        }
        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.5);
            TieBand(0.2)
        }
    }
    let profile = Profile {
        name: "mixed".into(),
        filters: vec![],
        scorers: vec![
            WeightedScorer {
                scorer: Box::new(Wide),
                weight: 3.0,
            },
            WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            },
        ],
        picker: Box::new(Best),
    };
    let pool = [candidate(0, 1), candidate(1, 1)];
    let kept = [0u32, 1];
    let (mut scores, mut stage) = (Vec::new(), Vec::new());
    let band = combine(
        &profile,
        &RequestFacts::default(),
        &pool,
        &kept,
        &mut scores,
        &mut stage,
    );
    let expected = 0.75 * 0.2 + 0.25 * TieBand::NORMALIZED.0;
    assert!((band.0 - expected).abs() < 1e-7, "{band:?}");
    assert!(scores
        .iter()
        .all(|&s| (s - (0.75 * 0.5 + 0.25)).abs() < 1e-6));
}

/// A profile of one stage whose share is the whole sum has that stage
/// write the scores where they stand, and each value and the band are
/// still the sum's own terms, bit for bit: a value past either end is
/// brought in, a negative zero reads as zero, and a weight whose share
/// of itself does not round to one scores through the sum.
#[test]
fn a_profile_of_one_stage_scores_as_the_sum_scores_it() {
    /// The values a stage may write, some outside a score's range.
    const GIVEN: [f32; 5] = [-0.0, 1.5, -3.0, 0.25, 1.0];
    /// A scorer that writes [`GIVEN`] under a band outside the range.
    #[derive(Debug)]
    struct Given;
    impl Lifecycle for Given {}
    impl Scorer for Given {
        fn name(&self) -> &'static str {
            "given"
        }
        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.copy_from_slice(&GIVEN);
            TieBand(2.0)
        }
    }
    let pool: Vec<Candidate> = (0..5).map(|id| candidate(id, 1)).collect();
    let kept = [0u32, 1, 2, 3, 4];
    for weight in [1.0f32, 3.0, 41.0] {
        let profile = Profile {
            name: "one".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(Given),
                weight,
            }],
            picker: Box::new(Best),
        };
        let (mut scores, mut stage) = (Vec::new(), Vec::new());
        let band = combine(
            &profile,
            &RequestFacts::default(),
            &pool,
            &kept,
            &mut scores,
            &mut stage,
        );
        // The sum's own terms: a zero plus the stage's share of the
        // value brought into range. The share is the one a decision in
        // front of the whole pass writes its scores by.
        let share = profile.scorers[0].sole_share();
        assert_eq!(share.to_bits(), (weight * (1.0 / weight)).to_bits());
        let term = |value: f32| (0.0 + share * value.clamp(0.0, 1.0)).to_bits();
        let written: Vec<u32> = scores.iter().map(|score| score.to_bits()).collect();
        let summed: Vec<u32> = GIVEN.iter().map(|&value| term(value)).collect();
        assert_eq!(written, summed, "weight {weight}");
        assert_eq!(band.0.to_bits(), term(2.0), "weight {weight}");
    }
    // The weights stand on both sides: the share of 41 is not one.
    assert_eq!(3.0f32 * (1.0 / 3.0f32), 1.0);
    assert_ne!(41.0f32 * (1.0 / 41.0f32), 1.0);
}

/// A profile may hold as many scorers as a place can name: the walk
/// that lists the stages that keep reports ends with the list.
#[test]
fn an_engine_of_as_many_scorers_as_a_place_names_is_built() {
    let scorers = (0..u16::MAX)
        .map(|_| WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        })
        .collect();
    let engine = Engine::new(
        vec![Profile {
            name: "wide".into(),
            filters: Vec::new(),
            scorers,
            picker: Box::new(Best),
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    );
    assert!(!engine.keeps_loads(), "no stage of it keeps a report");
}
