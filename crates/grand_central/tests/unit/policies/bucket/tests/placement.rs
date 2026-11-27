//! Where a prefill request goes: to its size's owner, or to the least
//! booked survivor where the most booked stands too far ahead of it.
//! The preset and a host that makes no lifecycle call stand here too.

use super::*;

#[test]
fn a_fresh_listing_splits_the_span_evenly_and_a_size_goes_to_its_range_s_owner() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(4);
    let placed: Vec<Option<u32>> = [0, 1_023, 1_024, 2_047, 2_048, 3_071, 3_072, 4_096, u32::MAX]
        .iter()
        .map(|&tokens| led(&scorer, &pool, tokens, SECOND))
        .collect();
    assert_eq!(
        placed,
        [0, 0, 1, 1, 2, 2, 3, 3, 3].map(Some),
        "a quarter of the span each, the last range open-ended"
    );
    assert_eq!(
        ranges(&scorer),
        [
            (0, 0, 1_023),
            (1, 1_024, 2_047),
            (2, 2_048, 3_071),
            (3, 3_072, u32::MAX)
        ]
    );
    // A span that does not divide leaves the rest to the open end.
    let thirds = self::scorer(NEVER, 1.5, 5);
    let three = self::pool(3);
    led(&thirds, &three, 0, SECOND);
    assert_eq!(
        ranges(&thirds),
        [(0, 0, 1_364), (1, 1_365, 2_729), (2, 2_730, u32::MAX)]
    );
}

/// A request of text alone is placed and booked by its text's length in
/// characters, as the classic `bucket` sizes one
/// (`model_gateway/src/policies/bucket.rs:104-110`) and books it
/// (`:178-181`). A request that states token ids is placed and booked
/// by their count, whatever length stands beside it. A dispatch on
/// another worker books the same size there.
#[test]
fn a_request_of_text_alone_is_placed_and_booked_by_its_characters() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(4);
    let led = |tokens, chars| {
        let facts = RequestFacts {
            prompt_chars: chars,
            ..prefill("m", tokens, SECOND)
        };
        led_by(&scorer, &facts, &pool, &all(&pool))
    };
    // 2,500 characters and no token id: the third quarter's owner.
    assert_eq!(led(0, 2_500), Some(2));
    // Token ids win: 100 of them go to the first quarter's owner.
    assert_eq!(led(100, 2_500), Some(0));
    let sized = |tokens, chars, index| Lease {
        tokens,
        chars,
        target_index: index,
        issued_micros: SECOND,
        ..Lease::default()
    };
    issue(&scorer, &pool, 2, &sized(0, 2_500, 0));
    issue(&scorer, &pool, 0, &sized(100, 2_500, 0));
    assert_eq!(loads(&scorer), (vec![100, 0, 2_500, 0], 2_600));
    // The text's request is sent to its pick's second target, the last
    // row: that worker is booked its characters too.
    let target = Target {
        id: pool[3].id,
        ..Target::default()
    };
    scorer.on_dispatched(&sized(0, 2_500, 1), &target, Side::Single, 250);
    assert_eq!(loads(&scorer), (vec![100, 0, 2_500, 2_500], 5_100));
}

#[test]
fn an_owner_that_is_no_survivor_hands_its_size_down_then_up() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(4);
    let over = |tokens, kept: &[u32]| led_by(&scorer, &prefill("m", tokens, SECOND), &pool, kept);
    // The first decision lists every row of its view, survivor or not:
    // row 1 owns its range from the start.
    assert_eq!(over(1_500, &[0, 2, 3]), Some(0), "the nearest below");
    assert_eq!(ranges(&scorer).len(), 4);
    assert_eq!(
        over(1_500, &[2, 3]),
        Some(2),
        "none below: the nearest above"
    );
    assert_eq!(
        over(3_500, &[0, 1]),
        Some(1),
        "the nearest below the last range"
    );
    assert_eq!(
        over(100, &[2, 3]),
        Some(2),
        "the nearest above the first range"
    );
    assert_eq!(over(100, &[3]), Some(3));
    assert_eq!(
        over(1_500, &[0, 1, 2, 3]),
        Some(1),
        "the owner when it survives"
    );
}

#[test]
fn only_a_prefill_decision_is_placed() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(4);
    for phase in [Phase::Single, Phase::Decode, Phase::Encode] {
        let facts = RequestFacts {
            phase,
            ..prefill("m", 1_500, SECOND)
        };
        assert_eq!(
            led_by(&scorer, &facts, &pool, &all(&pool)),
            None,
            "{phase:?}"
        );
    }
    assert!(
        scorer.partitions.load().is_empty(),
        "another phase lists nothing and starts no cadence"
    );
    assert_eq!(
        led_by(&scorer, &prefill("m", 1_500, SECOND), &pool, &[]),
        None
    );
    assert!(
        scorer.partitions.load().is_empty(),
        "a decision with no survivor"
    );
}

#[test]
fn the_balance_rule_spills_to_the_least_booked_survivor_when_both_its_parts_hold() {
    // Size 0 is the first row's in an even split of three.
    let spilled = |abs, rel, tokens: &[u32]| {
        let (scorer, pool) = booked(abs, rel, tokens);
        led(&scorer, &pool, 0, SECOND)
    };
    assert_eq!(
        spilled(32, 1.5, &[100, 60, 80]),
        Some(1),
        "40 apart, 100 over 90"
    );
    assert_eq!(
        spilled(32, 1.5, &[100, 70, 80]),
        Some(0),
        "30 apart: the owner"
    );
    assert_eq!(
        spilled(32, 1.5, &[92, 60, 80]),
        Some(0),
        "32 apart is not more than 32"
    );
    assert_eq!(
        spilled(32, 1.5, &[100, 67, 80]),
        Some(0),
        "100 is not over 100.5"
    );
    assert_eq!(spilled(32, 1.5, &[100, 66, 80]), Some(1), "100 over 99");
    assert_eq!(spilled(0, 1.5, &[90, 60, 80]), Some(0), "90 is not over 90");
    assert_eq!(spilled(32, 1.5, &[50, 0, 10]), Some(1), "an idle worker");
    assert_eq!(
        spilled(32, 1.5, &[100, 60, 60]),
        Some(1),
        "the first of two least"
    );
    assert_eq!(spilled(32, 1.5, &[0, 0, 0]), Some(0), "nothing booked");
    // The rule weighs the survivors alone: the least booked worker of the
    // listing draws no spill while it is no survivor.
    let (scorer, pool) = booked(32, 1.5, &[100, 10, 80]);
    let facts = prefill("m", 0, SECOND);
    assert_eq!(led_by(&scorer, &facts, &pool, &[0, 2]), Some(0), "20 apart");
    let (scorer, pool) = booked(32, 1.5, &[100, 10, 60]);
    assert_eq!(led_by(&scorer, &facts, &pool, &[0, 2]), Some(2));
}

#[test]
fn the_balance_rule_uses_the_stored_relative_threshold_at_an_integer_boundary() {
    let (scorer, pool) = booked(64, 1.3, &[286, 220]);
    assert_eq!(led(&scorer, &pool, 0, SECOND), Some(1));
}

#[test]
fn large_exact_loads_still_cross_the_balance_gate() {
    let scorer = scorer(64, 1.0, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    for row in 0..2 {
        for _ in 0..256 {
            book(&scorer, &pool, row, u32::MAX, SECOND);
        }
    }
    book(&scorer, &pool, 0, 321, SECOND);
    book(&scorer, &pool, 1, 256, SECOND);

    assert_eq!(loads(&scorer).0, vec![(1 << 40) + 65, 1 << 40]);
    assert_eq!(led(&scorer, &pool, 0, SECOND), Some(1));
}

#[test]
fn all_rows_of_the_bucket_owner_score_alike() {
    let pool = pool(2);
    let mut candidates = [
        pool[0],
        Candidate {
            position: 1,
            dp_rank: 1,
            ..pool[0]
        },
        Candidate {
            position: 2,
            ..pool[1]
        },
    ];
    let scores = |candidates: &[Candidate]| {
        let scorer = scorer(NEVER, 1.5, 5);
        let mut out = vec![0.5; candidates.len()];
        assert_eq!(
            scorer.score(
                &prefill("m", 0, SECOND),
                candidates,
                &all(candidates),
                &mut out,
            ),
            TieBand::NORMALIZED
        );
        let score = |id, rank| {
            candidates
                .iter()
                .zip(&out)
                .find(|(candidate, _)| candidate.id == id && candidate.dp_rank == rank)
                .map(|(_, &score)| score)
        };
        [
            score(pool[0].id, 0),
            score(pool[0].id, 1),
            score(pool[1].id, 0),
        ]
    };
    assert_eq!(scores(&candidates), [Some(1.0), Some(1.0), Some(0.0)]);
    candidates.swap(0, 1);
    for (position, candidate) in candidates.iter_mut().enumerate() {
        candidate.position = position as u32;
    }
    assert_eq!(scores(&candidates), [Some(1.0), Some(1.0), Some(0.0)]);
}

#[test]
fn the_balance_rule_spills_to_every_row_of_the_least_booked_worker() {
    let pool = pool(2);
    let candidates = [
        pool[0],
        Candidate {
            position: 1,
            dp_rank: 1,
            ..pool[0]
        },
        Candidate {
            position: 2,
            ..pool[1]
        },
    ];
    let scorer = scorer(32, 1.5, 5);
    let mut out = vec![0.5; candidates.len()];
    scorer.score(
        &prefill("m", 0, SECOND),
        &candidates,
        &all(&candidates),
        &mut out,
    );
    book(&scorer, &candidates, 2, 100, SECOND);

    // 3,000 is worker 101's size in the even split of two: worker 100's
    // rows lead only because 100 booked against 0 spills.
    scorer.score(
        &prefill("m", 3_000, SECOND),
        &candidates,
        &all(&candidates),
        &mut out,
    );
    assert_eq!(out, [1.0, 1.0, 0.0]);
}

#[test]
fn the_preset_places_a_prefill_request_by_its_size_and_books_what_is_issued() {
    let engine = PolicySpec::preset("bucket")
        .expect("the preset's form")
        .validate()
        .expect("the preset validates");
    let view = pool(4);
    let facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    let mut placed = Vec::new();
    for seed in 1..=3 {
        let request = RequestFacts {
            seed,
            ..prefill("m", 3_000, SECOND)
        };
        let decision = engine.decide(
            &request,
            &facts,
            &view,
            &Constraints::default(),
            &mut scratch,
        );
        let lease = Lease {
            tokens: request.prompt_tokens,
            issued_micros: request.now_micros,
            ..Lease::default()
        };
        engine.on_decision_issued(&lease, &decision);
        let pick = decision.pick().expect("a pick");
        placed.push(pick.targets.first().expect("a target").position);
    }
    // The owner of 3,000 tokens in an even split of four, then the
    // preset's balance rule (64 tokens and 1.5): the first idle worker,
    // then the next.
    assert_eq!(placed, [2, 0, 1]);
}

#[test]
fn a_host_that_makes_no_lifecycle_call_refuses_the_stage_and_the_preset() {
    let served = StageRegistry::builtin();
    let unserved = StageRegistry::builtin().without_lifecycle();
    let named: PolicySpec = serde_json::from_str(
        r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["bucket"],"pick":"best"}}}"#,
    )
    .expect("a well-formed document");
    let preset = PolicySpec::preset("bucket").expect("the preset's form");
    for (spec, path) in [
        (&named, "profiles.p.score[0].bucket"),
        (&preset, "profiles.bucket.score[0].bucket"),
    ] {
        let engine = spec
            .validate_in(&served, &[])
            .expect("the host makes the calls");
        assert!(engine.needs_lifecycle(), "{path}");
        assert_eq!(
            spec.validate_in(&unserved, &[]).err(),
            Some(SpecError::NotInBuild {
                path: path.to_string(),
                what: "the lifecycle this stage needs",
            }),
            "{path}"
        );
    }
}
