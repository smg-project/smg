//! Report age at the decision, across the load readers.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use smg_policy_controller::{
    document::{PolicySpec, SpecError, API_VERSION},
    engine::{
        AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Decision, Engine,
        Health, LoadRow, PairPools, Phase, PoolFacts, RequestFacts, ScoreFront, Scratch, Target,
        WorkerId,
    },
    stages::Lifecycle,
    tracking::{Lease, Membership, Side},
};

const LIMIT_MS: u64 = 27;
const MAX_LIMIT_MS: u64 = 3_599_999;
const NOW: u64 = 1_000_000;
const EDGE: u64 = NOW - LIMIT_MS * 1_000;

fn document(stages: &str) -> PolicySpec {
    serde_json::from_str(&format!(
        r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{{"score":[{stages}],"pick":"best"}}}}}}"#
    ))
    .expect("a policy document")
}

fn policy(stages: &str, limit: Option<u64>) -> Engine {
    let mut spec = document(stages);
    spec.view.max_report_age_ms = limit;
    spec.validate().expect("a valid policy")
}

fn candidate(position: u32, sample: u64) -> Candidate {
    Candidate {
        position,
        id: WorkerId(u64::from(position) + 1),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        load_sampled_at_micros: sample,
        load: Some(LoadRow {
            running: position + 1,
            live_rate: 1_000,
            resident_tokens: 320 * (position + 1),
            resident_counted: true,
            ..LoadRow::default()
        }),
        ..Candidate::default()
    }
}

fn scores(policy: &Engine, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Vec<f32> {
    let kept: Vec<u32> = rows.iter().map(|row| row.position).collect();
    let mut out = vec![0.0; rows.len()];
    policy.profiles()[0].scorers[0]
        .scorer
        .score(facts, rows, &kept, &mut out);
    out
}

fn request() -> RequestFacts<'static> {
    RequestFacts {
        now_micros: NOW,
        block_hashes: &[0; 16],
        block_size: 16,
        prompt_tokens: 256,
        ..RequestFacts::default()
    }
}

#[test]
fn the_report_age_limit_is_optional_and_validated_by_name() {
    let base = document(r#""least-load""#);
    assert_eq!(base.view.max_report_age_ms, None);
    let printed = serde_json::to_value(&base).expect("JSON");
    assert!(printed.get("view").is_none());
    let mut with_view = base.clone();
    with_view.view.block_size = Some(16);
    let view = serde_json::to_value(&with_view).expect("JSON");
    assert!(view["view"].get("max_report_age_ms").is_none());
    for name in smg_policy_controller::document::PRESETS {
        assert_eq!(
            PolicySpec::preset(name)
                .expect("a preset")
                .view
                .max_report_age_ms,
            None
        );
    }
    let mut written = printed;
    for bound in [0, MAX_LIMIT_MS + 1, 1, MAX_LIMIT_MS] {
        written["view"] = serde_json::json!({ "max_report_age_ms": bound });
        let spec: PolicySpec = serde_json::from_value(written.clone()).expect("a numerical limit");
        assert_eq!(spec.view.max_report_age_ms, Some(bound));
        if bound == 0 || bound > MAX_LIMIT_MS {
            let error = spec.validate().expect_err("an out-of-range limit");
            assert!(
                matches!(&error, SpecError::BadField { value, .. } if value == &bound.to_string())
            );
            assert_eq!(error.field_path(), "view.max_report_age_ms");
        } else {
            spec.validate().expect("an in-range limit");
            assert_eq!(
                serde_json::to_value(&spec).expect("JSON")["view"]["max_report_age_ms"],
                bound
            );
            assert_ne!(
                spec.digest().expect("the configured digest"),
                base.digest().expect("the bare digest")
            );
        }
    }
}

#[test]
fn least_load_uses_the_decision_instant_and_keeps_the_exact_limit() {
    let stages = r#""least-load""#;
    let (limited, bare) = (policy(stages, Some(LIMIT_MS)), policy(stages, None));
    let mut rows = [candidate(0, EDGE), candidate(1, NOW), candidate(2, NOW)];
    rows[0].load = Some(LoadRow {
        running: 17,
        live_rate: 400_000,
        report_age_ms: u32::MAX,
        ..LoadRow::default()
    });
    rows[1].load = rows[1].load.map(|row| LoadRow {
        running: 7,
        live_rate: 0,
        ..row
    });
    let reported = scores(&bare, &request(), &rows);
    assert_eq!(scores(&limited, &request(), &rows), reported);

    rows[0].load_sampled_at_micros -= 1;
    assert_eq!(scores(&bare, &request(), &rows), reported);
    let stale = scores(&limited, &request(), &rows);
    assert_ne!(stale, reported);
    rows[0].load = None;
    assert_eq!(stale, scores(&bare, &request(), &rows));

    let steep = policy(
        r#"{"least-load":{"kv_pressure_weight":1e38}}"#,
        Some(LIMIT_MS),
    );
    let missing = scores(&steep, &request(), &rows);
    // This pressure would overflow the cost of a reported row. It has
    // no effect on a row already marked as unreported.
    rows[0].load = Some(LoadRow {
        active_kv_permille: 999,
        ..LoadRow::default()
    });
    assert_eq!(scores(&steep, &request(), &rows), missing);

    rows[0] = candidate(0, 0);
    let near_origin = RequestFacts {
        now_micros: 1,
        ..request()
    };
    assert_eq!(
        scores(&limited, &near_origin, &rows),
        scores(&bare, &near_origin, &rows),
    );
}

#[test]
fn least_load_asks_for_the_whole_pool_when_the_front_has_a_stale_report() {
    let engine = policy(r#""least-load""#, Some(LIMIT_MS));
    let stage = &engine.profiles()[0].scorers[0].scorer;
    let mut rows = [candidate(0, EDGE - 1), candidate(1, NOW), candidate(2, NOW)];
    rows[1].load = rows[1].load.map(|row| LoadRow {
        live_rate: 0,
        ..row
    });
    let mut out = [0.0; 2];
    let kept = [0, 1, 2];
    assert_eq!(
        stage
            .score_front(&request(), &rows, &kept, &[0, 1], &mut out)
            .1,
        ScoreFront::Whole
    );
    let with_stale = stage.score_front(&request(), &rows, &kept, &[1, 2], &mut out);
    let scores = out;
    rows[0].load = None;
    assert_eq!(
        stage.score_front(&request(), &rows, &kept, &[1, 2], &mut out),
        with_stale
    );
    assert_eq!(out, scores);
}

#[test]
fn resident_work_scores_stale_reports_as_unreported() {
    let stages = r#"{"resident-work":{"request_tokens":80}}"#;
    let (limited, bare) = (policy(stages, Some(LIMIT_MS)), policy(stages, None));
    let mut rows = [candidate(0, EDGE - 1), candidate(1, NOW)];
    rows[0].in_flight = 1;
    let reported = scores(&bare, &request(), &rows);
    let stale = scores(&limited, &request(), &rows);
    rows[0].load = None;
    assert_ne!(stale, reported);
    assert_eq!(stale, scores(&bare, &request(), &rows));
}

fn hot_pool() -> [Candidate; 3] {
    let mut rows = [candidate(0, EDGE - 1), candidate(1, NOW), candidate(2, NOW)];
    rows[0].load = rows[0].load.map(|row| LoadRow {
        active_kv_permille: 950,
        ..row
    });
    rows
}

#[test]
fn prefix_affinity_ignores_stale_reports_in_the_kv_gate() {
    let stages = r#""prefix-affinity",{"kv-pressure":{"ceiling":0.8}}"#;
    let (limited, bare) = (policy(stages, Some(LIMIT_MS)), policy(stages, None));
    let mut rows = hot_pool();
    let facts = RequestFacts {
        affinity: &[
            AffinityFacts {
                overlap_blocks: 8,
                ..AffinityFacts::default()
            },
            AffinityFacts::default(),
            AffinityFacts::default(),
        ],
        ..request()
    };
    assert_eq!(scores(&bare, &facts, &rows), [1.0; 3]);
    let stale = scores(&limited, &facts, &rows);
    assert_eq!(stale, [1.0, 0.5, 0.5]);
    let kept = [0, 1, 2];
    let mut front = [0.0; 3];
    let stage = &limited.profiles()[0].scorers[0].scorer;
    assert_eq!(
        stage.score_front(&facts, &rows, &kept, &kept, &mut front).1,
        ScoreFront::At(1)
    );
    assert_eq!(front, stale.as_slice());
    let named = RequestFacts {
        named: Some(AffinityRows {
            holders: &[AffinityHolder {
                row: 0,
                overlap_blocks: 8,
                predicted_blocks: 0,
            }],
            ..AffinityRows::default()
        }),
        affinity: &[],
        ..facts
    };
    assert_eq!(
        stage.score_front(&named, &rows, &kept, &kept, &mut front).1,
        ScoreFront::Unsaid
    );
    assert_eq!(front, stale.as_slice());
    rows[0].load = None;
    assert_eq!(stale, scores(&bare, &facts, &rows));
}

#[test]
fn the_warmup_slice_ignores_stale_reports_in_the_kv_gate() {
    let stages = r#"{"warmup-slice":{"share":1.0,"blocks":8}},{"kv-pressure":{"ceiling":0.8}}"#;
    let (limited, bare) = (policy(stages, Some(LIMIT_MS)), policy(stages, None));
    let mut rows = hot_pool();
    let mature = AffinityFacts {
        admitted_age_ms: u32::MAX,
        indexed_blocks: 64,
        indexed_growth: 64,
        ..AffinityFacts::default()
    };
    let facts = RequestFacts {
        affinity: &[mature, mature, AffinityFacts::default()],
        ..request()
    };
    assert_eq!(scores(&bare, &facts, &rows), [1.0; 3]);
    let stale = scores(&limited, &facts, &rows);
    assert_eq!(stale, [0.0, 0.0, 1.0]);
    let kept = [0, 1, 2];
    let mut front = [0.0; 3];
    let stage = &limited.profiles()[0].scorers[0].scorer;
    assert_eq!(
        stage.score_front(&facts, &rows, &kept, &kept, &mut front).1,
        ScoreFront::At(1)
    );
    assert_eq!(front, stale.as_slice());
    rows[0].load = None;
    assert_eq!(stale, scores(&bare, &facts, &rows));
    assert!(limited.warm_up_rule_for("", Phase::Single).is_some());
}

#[test]
fn a_report_age_limit_preserves_the_built_in_scorers_refusal_flags() {
    for stage in [
        r#""least-load""#,
        r#"{"resident-work":{"request_tokens":63}}"#,
        r#"{"least-cost":{"terms":{"candidate.in_flight":3}}}"#,
        r#""prefix-affinity""#,
        r#""warmup-slice""#,
    ] {
        let bare = policy(stage, None);
        let limited = policy(stage, Some(41));
        assert_eq!(
            limited.profiles()[0].scorers[0].scorer.refuses_by_reports(),
            bare.profiles()[0].scorers[0].scorer.refuses_by_reports(),
            "{stage}"
        );
    }
}

#[test]
fn a_report_age_limit_keeps_dispatch_credit_until_membership_removes_it() {
    let engine = policy(r#""least-load""#, Some(LIMIT_MS));
    assert!(engine.needs().loads);
    let rows = [candidate(0, EDGE), candidate(1, EDGE)];
    let lease = Lease {
        issued_micros: EDGE + 1,
        tokens: 9_000,
        ..Lease::default()
    };
    let target = Target {
        id: rows[0].id,
        ..Target::default()
    };
    assert_eq!(scores(&engine, &request(), &rows), [1.0, 0.0]);
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    assert_eq!(scores(&engine, &request(), &rows), [0.0, 1.0]);
    engine.on_membership(&Membership::Removed(rows[0].id));
    assert_eq!(scores(&engine, &request(), &rows), [1.0, 0.0]);
}

#[test]
fn a_stale_report_behind_the_front_sends_the_pool_through_whole_scoring() {
    let stages = r#"{"prefix-affinity":{"weight":4.0}},"least-load""#;
    let (limited, bare) = (policy(stages, Some(LIMIT_MS)), policy(stages, None));
    let mut rows: Vec<_> = (0..16).map(|i| candidate(i, EDGE)).collect();
    let mut column = [AffinityFacts::default(); 16];
    for entry in &mut column[..4] {
        entry.overlap_blocks = 8;
    }
    let facts = RequestFacts {
        affinity: &column,
        ..request()
    };
    let decide = |policy: &Engine, rows: &[Candidate], reported| {
        policy.decide(
            &facts,
            &PoolFacts {
                reported,
                ..PoolFacts::default()
            },
            rows,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    let by_rank = decide(&bare, &rows, true);
    assert_ne!(by_rank, decide(&bare, &rows, false));
    assert_eq!(by_rank, decide(&limited, &rows, true));
    rows[15].load_sampled_at_micros -= 1;
    rows[15].in_flight = 48;
    let stale = decide(&limited, &rows, true);
    rows[15].load = None;
    assert_eq!(stale, decide(&bare, &rows, false));
}

#[test]
fn both_sides_of_a_pair_use_the_report_age_limit() {
    let mut spec = PolicySpec::pair_presets("least_load", "least_load").expect("two presets");
    spec.view.max_report_age_ms = Some(LIMIT_MS);
    let engine = spec.validate().expect("the pair policy");
    let side = |base| {
        let mut stale = candidate(0, EDGE - 1);
        stale.load = stale.load.map(|row| LoadRow {
            live_rate: 1_000_000,
            ..row
        });
        stale.in_flight = 3;
        let mut rows = [stale, candidate(1, NOW)];
        for row in &mut rows {
            row.id.0 += base;
            row.pair_class = 1;
            row.has_capacity = true;
            row.has_room = true;
        }
        rows
    };
    let (prefills, decodes) = (side(0), side(2));
    let cx = Constraints::default();
    let decide = |engine: &Engine| {
        let Decision::Pick(pick) = engine.decide_pair(
            &request(),
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &cx,
            &cx,
            &mut Scratch::new(),
        ) else {
            panic!("a pair was available");
        };
        let first = pick.targets.iter().next().expect("the first pair");
        (first.id, first.pair.expect("a decode").decode_id)
    };
    assert_eq!(decide(&engine), (prefills[1].id, decodes[1].id));
    spec.view.max_report_age_ms = None;
    assert_eq!(
        decide(&spec.validate().expect("the bare pair policy")),
        (prefills[0].id, decodes[0].id)
    );
}
