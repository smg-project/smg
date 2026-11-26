use super::*;

/// A scorer for the affinity column's cases: each kept row scores the
/// overlap its own entry of the column holds, and declares that it reads
/// the column.
#[derive(Debug)]
struct ByOverlap;

impl Lifecycle for ByOverlap {}

impl Scorer for ByOverlap {
    fn name(&self) -> &'static str {
        "by-overlap"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (score, &i) in out.iter_mut().zip(kept) {
            *score = facts.affinity_at(i).overlap_blocks as f32 / 100.0;
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            affinity: true,
            ..Needs::default()
        }
    }
}

/// One profile that keeps the healthy rows and leads with the deepest
/// overlap, under `binding`.
fn overlap_engine(binding: Binding) -> Engine {
    let profile = Profile {
        name: "overlap".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(ByOverlap),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    };
    Engine::new(vec![profile], vec![binding], 1, Needs::default())
}

fn refused_for_its_column(decision: &Decision) -> bool {
    matches!(
        decision,
        Decision::Shed(shed)
            if (shed.reason, shed.emptied_by) == (ShedReason::PolicyDeclined, "affinity-column")
    )
}

#[test]
fn a_column_of_another_length_is_refused_at_the_regular_entry() {
    let engine = overlap_engine(Binding {
        model: "m".into(),
        ..single_binding()
    });
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let decide = |model: &'static str, column: &[AffinityFacts]| {
        let facts = RequestFacts {
            model,
            affinity: column,
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
    // One entry a candidate: the holder of the deepest overlap leads.
    assert_eq!(lead(&decide("m", &held(&[10, 90, 20]))), Some(1));
    // An empty column decides too: nothing held, every row alike.
    assert!(decide("m", &[]).pick().is_some());
    // Shorter or longer: refused, under the column's own stage name.
    for column in [held(&[10, 90]), held(&[10, 90, 20, 30])] {
        let decision = decide("m", &column);
        assert!(
            refused_for_its_column(&decision),
            "a column of {} for three candidates answered {decision:?}",
            column.len()
        );
        assert_eq!(decision.trace().made_under(), None);
    }
    // A decision's inputs are checked before anything is resolved from
    // them: a model no binding covers is refused for its column when the
    // column is another pool's, and for its binding only beside a column
    // that fits.
    assert!(refused_for_its_column(&decide("unbound", &held(&[10, 90]))));
    assert!(matches!(
        decide("unbound", &held(&[10, 90, 20])),
        Decision::Shed(shed) if shed.emptied_by == "binding"
    ));
}

/// A holder a host named in the column's place leads as the deepest
/// overlap of a column does, and a request that names nothing reads its
/// column.
#[test]
fn a_holder_a_host_named_leads_as_the_deepest_overlap_of_a_column_does() {
    use crate::engine::{AffinityHolder, AffinityRows};
    let engine = overlap_engine(Binding {
        model: "m".into(),
        ..single_binding()
    });
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let decide = |column: &[AffinityFacts], named: Option<AffinityRows<'_>>| {
        let facts = RequestFacts {
            model: "m",
            affinity: column,
            named,
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
    let holders = [
        AffinityHolder {
            row: 0,
            overlap_blocks: 10,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 1,
            overlap_blocks: 90,
            predicted_blocks: 0,
        },
    ];
    let named = AffinityRows {
        holders: &holders,
        ..AffinityRows::default()
    };
    assert_eq!(lead(&decide(&[], Some(named))), Some(1));
    // Nobody named decides too.
    assert!(decide(&[], Some(AffinityRows::default())).pick().is_some());
    // A request that names nothing reads its column.
    assert_eq!(lead(&decide(&held(&[10, 90, 20]), None)), Some(1));
}

/// The regular entry takes the named rows under an assertion, as it
/// takes the rows' positions: a debug build stops at a list out of
/// order, at a row named twice, at a row past the pool and at rows
/// named beside a column, each by the assertion's own words. A release
/// build does not check, so the case is a debug build's alone.
#[cfg(debug_assertions)]
#[test]
fn rows_named_out_of_order_past_the_pool_or_beside_a_column_stop_a_debug_build() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::engine::{AffinityHolder, AffinityRows};
    let engine = overlap_engine(Binding {
        model: "m".into(),
        ..single_binding()
    });
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let column = held(&[10, 90, 20]);
    let shapes = [
        ("holders out of order", &[][..], &[2, 1][..], &[][..]),
        ("a holder named twice", &[][..], &[1, 1][..], &[][..]),
        ("a holder past the pool", &[][..], &[1, 3][..], &[][..]),
        ("warming rows out of order", &[][..], &[][..], &[1, 0][..]),
        ("a warming row named twice", &[][..], &[][..], &[1, 1][..]),
        ("a warming row past the pool", &[][..], &[][..], &[3][..]),
        ("rows named beside a column", &column[..], &[1][..], &[][..]),
    ];
    for (what, column, holders, warming) in shapes {
        let holders: Vec<AffinityHolder> = holders
            .iter()
            .map(|&row| AffinityHolder {
                row,
                overlap_blocks: 90,
                predicted_blocks: 0,
            })
            .collect();
        let stopped = catch_unwind(AssertUnwindSafe(|| {
            let facts = RequestFacts {
                model: "m",
                affinity: column,
                named: Some(AffinityRows {
                    holders: &holders,
                    warming,
                    thin: false,
                    mean_in_flight: None,
                    tree: None,
                }),
                ..RequestFacts::default()
            };
            engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        }))
        .expect_err(what);
        let said = stopped
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| stopped.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            said.contains("rows named in the column's place"),
            "{what} stopped by another word: {said}"
        );
    }
}

/// The regular entry takes its rows under the assertion the pair entry
/// takes each side's under: a debug build stops at a row whose position
/// is not its index, the second one here. A release build does not
/// check, so the case is a debug build's alone.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row's position is its index")]
fn a_misplaced_row_stops_a_debug_build_at_the_regular_entry() {
    let engine = overlap_engine(single_binding());
    let mut pool = [candidate(0, 0), candidate(1, 0)];
    pool[1].position = 5;
    let decision = engine.decide(
        &RequestFacts::default(),
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    panic!("a misplaced row answered {decision:?}");
}

/// The pair entry of the cases below: `prefills` and `decodes` beside the
/// two columns as the call hands them.
fn decide_pair_beside(
    engine: &Engine,
    prefills: &[Candidate],
    decodes: &[Candidate],
    prefill_affinity: &[AffinityFacts],
    decode_affinity: &[AffinityFacts],
) -> Decision {
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills,
            decodes,
            prefill_affinity,
            decode_affinity,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

#[test]
fn a_column_of_another_length_is_refused_at_each_side_of_the_pair_entry() {
    let engine = overlap_engine(pair_binding());
    let (prefills, decodes) = (paired(2), paired(3));
    let (two, three) = (held(&[10, 90]), held(&[10, 90, 20]));
    let refused = |prefill_affinity: &[AffinityFacts], decode_affinity: &[AffinityFacts]| {
        refused_for_its_column(&decide_pair_beside(
            &engine,
            &prefills,
            &decodes,
            prefill_affinity,
            decode_affinity,
        ))
    };
    // Each pool beside its own column, or beside none, is decided.
    assert!(!refused(&two, &three));
    assert!(!refused(&[], &[]));
    assert!(!refused(&two, &[]));
    // The prefills beside a column of the decodes' length, then the
    // decodes beside one of the prefills': each side is refused alone.
    assert!(refused(&three, &three));
    assert!(refused(&two, &two));
}

/// A pair's columns ride its pools. A column on the request itself names
/// neither pool, whatever its length: refused, never dropped in silence.
#[test]
fn a_column_on_the_request_is_refused_at_the_pair_entry() {
    let engine = overlap_engine(pair_binding());
    let (prefills, decodes) = (paired(2), paired(2));
    let decide = |on_the_request: &[AffinityFacts]| {
        let facts = RequestFacts {
            model: "m",
            affinity: on_the_request,
            ..RequestFacts::default()
        };
        engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    assert!(decide(&[]).pick().is_some());
    // As long as either pool: its length fits both, and it is refused.
    assert!(refused_for_its_column(&decide(&held(&[10, 90]))));
}

/// Rows named on a pair's request name neither pool, as a column on it
/// does: refused.
#[test]
fn rows_named_on_the_request_are_refused_at_the_pair_entry() {
    let engine = overlap_engine(pair_binding());
    let (prefills, decodes) = (paired(2), paired(2));
    let facts = RequestFacts {
        model: "m",
        named: Some(AffinityRows::default()),
        ..RequestFacts::default()
    };
    let decision = engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(refused_for_its_column(&decision), "{decision:?}");
}

/// A pair's side carries no rows named, whatever its request carries:
/// named rows index neither pool.
#[test]
fn a_pair_side_reads_no_rows_named_on_the_request() {
    let (prefills, decodes) = (paired(2), paired(2));
    let holders = [crate::engine::AffinityHolder {
        row: 1,
        overlap_blocks: 8,
        predicted_blocks: 0,
    }];
    let facts = RequestFacts {
        model: "m",
        named: Some(AffinityRows {
            holders: &holders,
            ..AffinityRows::default()
        }),
        ..RequestFacts::default()
    };
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    for (side, name) in [(PairSide::Prefill, "prefill"), (PairSide::Decode, "decode")] {
        let (_, read) = pools.side(side, &facts).expect("an empty column fits");
        assert!(read.named.is_none(), "the {name} side");
    }
}

/// Each side of a pair reads its own places on a lent ring in the one
/// list a stage reads: the decode side's own where its request states
/// them apart, else the request's one list. Neither side's facts carry
/// the decode side's list on, so no stage is shown the other side's. A
/// decode side stated an empty list reads none: its stage's own ring
/// places it.
#[test]
fn each_side_of_a_pair_reads_its_own_places_on_a_lent_ring() {
    let (prefills, decodes) = (paired(2), paired(2));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let read = |decode_ring_positions: Option<&[u64]>| {
        let facts = RequestFacts {
            model: "m",
            ring_positions: &[7],
            decode_ring_positions,
            ..RequestFacts::default()
        };
        [(PairSide::Prefill, "prefill"), (PairSide::Decode, "decode")].map(|(side, name)| {
            let (_, read) = pools.side(side, &facts).expect("an empty column fits");
            assert!(read.decode_ring_positions.is_none(), "the {name} side");
            read.ring_positions.to_vec()
        })
    };
    assert_eq!(read(None), [vec![7], vec![7]]);
    assert_eq!(read(Some(&[3, 5])), [vec![7], vec![3, 5]]);
    assert_eq!(read(Some(&[])), [vec![7], vec![]]);
}

/// Each side of a pair reads its own pool's column: with the two columns
/// swapped between pools of equal length, the pair moves.
#[test]
fn each_side_of_a_pair_reads_its_own_pools_column() {
    let engine = overlap_engine(pair_binding());
    let (prefills, decodes) = (paired(2), paired(2));
    let (rising, falling) = (held(&[10, 90]), held(&[90, 10]));
    let pair = |prefill_affinity: &[AffinityFacts], decode_affinity: &[AffinityFacts]| {
        let decision = decide_pair_beside(
            &engine,
            &prefills,
            &decodes,
            prefill_affinity,
            decode_affinity,
        );
        let target = *decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .expect("a pair");
        (
            target.position,
            target.pair.expect("a decode").decode_position,
        )
    };
    assert_eq!(pair(&rising, &falling), (1, 0));
    assert_eq!(pair(&falling, &rising), (0, 1));
}

/// One index, one worker: a pool whose filters drop its FIRST candidates
/// reads each survivor's own entry of the column, not the entry at the
/// survivor's place among the kept.
#[test]
fn a_narrowed_pool_reads_each_survivors_own_entry_of_the_column() {
    let engine = overlap_engine(single_binding());
    let mut pool = [
        candidate(0, 0),
        candidate(1, 0),
        candidate(2, 0),
        candidate(3, 0),
    ];
    pool[0].health.healthy = false;
    pool[1].health.healthy = false;
    let column = held(&[90, 80, 10, 70]);
    let facts = RequestFacts {
        model: "m",
        affinity: &column,
        ..RequestFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert_eq!(lead(&decision), Some(3));
}

/// A profile says by the decision's one lookup whether a stage of it
/// may refuse by what its gateways report of leases. A stage that
/// says nothing is taken as one that may, and is no plugin's by
/// that: its profile shows no plugin its rows. A profile of this
/// crate's stages says it holds none.
#[test]
fn a_profile_whose_stage_may_refuse_by_reports_says_so_at_the_lookup() {
    #[derive(Debug)]
    struct Counted;

    impl Filter for Counted {
        fn name(&self) -> &'static str {
            "counted"
        }

        fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
            facts.in_flight(c) == 0
        }
    }

    #[derive(Debug)]
    struct Scored;

    impl Lifecycle for Scored {}

    impl Scorer for Scored {
        fn name(&self) -> &'static str {
            "scored"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    let engine_of =
        |profile| Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
    let plain = engine_of(least_requests_profile());
    assert!(!plain.reads_for("m", Phase::Single).2);
    // A filter that says nothing, and a scorer that says nothing: each
    // is taken as a stage that may refuse, and its word is the
    // profile's.
    let filtered = || Profile {
        filters: vec![Box::new(Counted)],
        ..least_requests_profile()
    };
    let scored = Profile {
        scorers: vec![WeightedScorer {
            scorer: Box::new(Scored),
            weight: 1.0,
        }],
        ..least_requests_profile()
    };
    let (_, shown, by_reports) = engine_of(filtered()).reads_for("m", Phase::Single);
    assert!(by_reports, "the filter's word is the profile's");
    assert!(!shown, "no plugin is shown its rows");
    let scored = engine_of(scored).reads_for("m", Phase::Single).2;
    assert!(scored, "the scorer's word is the profile's");
    // A decode binding with no pair carries its own profile's word.
    let decode = || Binding {
        phases: vec![Phase::Decode],
        ..single_binding()
    };
    let alone = Engine::new(vec![filtered()], vec![decode()], 1, Needs::default());
    assert!(alone.reads_for("m", Phase::Decode).2);
    // A decode binding of a profile with no such stage stands ahead of
    // a pair binding whose decode side holds one: the decode decision
    // carries the pair's word, as it carries the pair's needs.
    let ahead = Engine::new(
        vec![least_requests_profile(), filtered()],
        vec![
            decode(),
            Binding {
                decode_profile: Some(1),
                ..pair_binding()
            },
        ],
        1,
        Needs::default(),
    );
    assert!(!ahead.reads_for("m", Phase::Prefill).2);
    assert!(ahead.reads_for("m", Phase::Decode).2);
    // An encode binding of a profile with no such stage stands ahead of
    // an `epd` binding whose encode side holds one: the encode decision
    // carries the pair's word the same way. A pair of two sides names
    // no encode side and adds none.
    let encode = || Binding {
        phases: vec![Phase::Encode],
        ..single_binding()
    };
    let behind = |plan: Plan| {
        let pair = Binding {
            plan,
            ..pair_binding()
        };
        Engine::new(
            vec![least_requests_profile(), filtered()],
            vec![encode(), pair],
            1,
            Needs::default(),
        )
    };
    let three = behind(Plan::Epd { encode: 1 });
    assert!(three.reads_for("m", Phase::Encode).2);
    assert!(!three.reads_for("m", Phase::Decode).2);
    assert!(!behind(Plan::PdPair).reads_for("m", Phase::Encode).2);
}

/// A profile that shows its rows to a plugin says so by the engine's one
/// lookup, and its decision over rows that carry a mask beside them is
/// declined under a name of its own, in every build.
#[test]
fn a_profile_that_shows_its_rows_is_told_and_declines_rows_not_written_for_it() {
    #[derive(Debug)]
    struct Shown;

    impl Lifecycle for Shown {}

    impl DecisionScope for Shown {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

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

    let engine_of =
        |profile| Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
    let plain = engine_of(least_requests_profile());
    let needs = plain.needs_for("m", Phase::Single);
    assert_eq!(plain.reads_for("m", Phase::Single), (needs, false, false));
    let engine = engine_of(Profile {
        filters: vec![Box::new(Shown)],
        ..least_requests_profile()
    });
    assert!(engine.reads_for("m", Phase::Single).1);
    let pool = [candidate(0, 0), candidate(1, 0)];
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let masked = Constraints {
        stalled: &[0],
        ..Constraints::default()
    };
    let written = Constraints::default();
    let decide = |facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
        engine.decide(facts, &PoolFacts::default(), &pool, cx, &mut Scratch::new())
    };
    assert!(lead(&decide(&facts, &written)).is_some());
    let Decision::Shed(shed) = decide(&facts, &masked) else {
        panic!("a mask beside the rows is declined");
    };
    assert_eq!(
        (shed.reason, shed.emptied_by),
        (ShedReason::PolicyDeclined, WRITTEN_ROWS)
    );
    assert_eq!(shed.trace.made_under(), Some(0));
    // A profile that shows nothing reads the mask.
    assert!(lead(&plain.decide(
        &facts,
        &PoolFacts::default(),
        &pool,
        &masked,
        &mut Scratch::new()
    ))
    .is_some());
    // Under a pair binding the word is the phase's own profile's: the
    // decode side shows its rows here, and the prefill side does not.
    let pair = Engine::new(
        vec![
            least_requests_profile(),
            Profile {
                filters: vec![Box::new(Shown)],
                ..least_requests_profile()
            },
        ],
        vec![Binding {
            decode_profile: Some(1),
            ..pair_binding()
        }],
        1,
        Needs::default(),
    );
    assert!(!pair.reads_for("m", Phase::Prefill).1);
    assert!(pair.reads_for("m", Phase::Decode).1);
}

/// The pair entry declines as the single entry does. A side whose
/// profile shows its rows is handed rows written for the decision: a
/// mask beside that side's rows is declined before either scope opens,
/// and rows that carry the veto in their own field are shown with it. A
/// mask beside a side that shows nothing is read by that side's filters.
/// The host is told of the pair's decode side where a decode binding of
/// another profile stands ahead of it, as it is told that side's needs,
/// and of an `epd` pair's encode side the same way.
#[test]
fn a_pair_side_that_shows_its_rows_declines_a_mask_beside_them() {
    /// The scopes opened, and the rows they were shown stalled.
    #[derive(Debug)]
    struct Shown(Arc<Mutex<(u32, u32)>>);

    impl Lifecycle for Shown {}

    impl DecisionScope for Shown {
        fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], _: &PoolFacts<'_>) {
            let mut seen = self.0.lock().expect("the lock");
            seen.0 += 1;
            seen.1 += rows.iter().filter(|row| row.health.stalled).count() as u32;
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

    let seen = Arc::new(Mutex::new((0u32, 0u32)));
    let shows = || Profile {
        filters: vec![Box::new(Healthy), Box::new(Shown(Arc::clone(&seen)))],
        ..least_requests_profile()
    };
    let read = || *seen.lock().expect("the lock");
    let pair_of = |prefill: Profile, decode: Profile| {
        let binding = Binding {
            decode_profile: Some(1),
            ..pair_binding()
        };
        Engine::new(vec![prefill, decode], vec![binding], 1, Needs::default())
    };
    let (prefills, mut decodes) = (paired(2), paired(2));
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let masked = Constraints {
        stalled: &[0],
        ..Constraints::default()
    };
    let written = Constraints::default();
    let decide = |engine: &Engine, decodes: &[Candidate], sides: [&Constraints<'_>; 2]| {
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            ..PairPools::default()
        };
        let pool = PoolFacts::default();
        engine.decide_pair(
            &facts,
            &pool,
            pools,
            sides[0],
            sides[1],
            &mut Scratch::new(),
        )
    };
    let declined = |decision: &Decision| {
        matches!(decision, Decision::Shed(shed)
                if (shed.reason, shed.emptied_by) == (ShedReason::PolicyDeclined, WRITTEN_ROWS))
    };
    // The decode side shows its rows, the prefill side shows nothing.
    let engine = pair_of(least_requests_profile(), shows());
    assert!(lead(&decide(&engine, &decodes, [&written, &written])).is_some());
    assert_eq!(read(), (1, 0));
    // A mask beside the decode side's rows: declined, and no scope opens.
    let decision = decide(&engine, &decodes, [&written, &masked]);
    assert!(declined(&decision), "{decision:?}");
    assert_eq!(decision.trace().made_under(), Some(1));
    assert_eq!(read(), (1, 0), "no scope opens for a declined decision");
    // A mask beside the prefill side, which shows nothing: its filters
    // read it, and the masked prefill is passed over.
    assert_eq!(
        lead(&decide(&engine, &decodes, [&masked, &written])),
        Some(1)
    );
    assert_eq!(read(), (2, 0));
    // Rows written for the decision carry the veto in their own field:
    // the plugin is shown it, and the built-in filter reads it there.
    decodes[0].health.stalled = true;
    let decision = decide(&engine, &decodes, [&written, &written]);
    let decode = decision
        .pick()
        .and_then(|pick| pick.targets.first().and_then(|target| target.pair));
    assert_eq!(
        decode.map(|pair| pair.decode_position),
        Some(1),
        "{decision:?}"
    );
    assert_eq!(read(), (3, 1));
    decodes[0].health.stalled = false;

    // The other way round: the prefill side shows its rows.
    let turned = pair_of(shows(), least_requests_profile());
    let decision = decide(&turned, &decodes, [&masked, &written]);
    assert!(declined(&decision), "{decision:?}");
    let decision = decide(&turned, &decodes, [&written, &masked]);
    let decode = decision
        .pick()
        .and_then(|pick| pick.targets.first().and_then(|target| target.pair));
    assert_eq!(
        decode.map(|pair| pair.decode_position),
        Some(1),
        "{decision:?}"
    );

    // A decode binding of a profile that shows nothing stands ahead of
    // the pair binding: the pair's decode side still runs under the pair
    // binding, and the host is told so.
    let ahead = Engine::new(
        vec![least_requests_profile(), shows()],
        vec![
            Binding {
                phases: vec![Phase::Decode],
                ..single_binding()
            },
            Binding {
                decode_profile: Some(1),
                ..pair_binding()
            },
        ],
        1,
        Needs::default(),
    );
    assert!(!ahead.reads_for("m", Phase::Prefill).1);
    assert!(ahead.reads_for("m", Phase::Decode).1);
    // An encode binding of a profile that shows nothing stands ahead of
    // an `epd` binding whose encode profile shows its rows: the encode
    // side of the pair's one call runs under the pair binding, and the
    // host is told so too.
    let encoders = Engine::new(
        vec![least_requests_profile(), shows()],
        vec![
            Binding {
                phases: vec![Phase::Encode],
                ..single_binding()
            },
            Binding {
                plan: Plan::Epd { encode: 1 },
                ..pair_binding()
            },
        ],
        1,
        Needs::default(),
    );
    assert!(!encoders.reads_for("m", Phase::Decode).1);
    assert!(encoders.reads_for("m", Phase::Encode).1);
}
