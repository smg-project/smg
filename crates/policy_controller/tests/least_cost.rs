//! The document's costs and margin through a validated decision.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "cases stop at a broken setup or an unexpected answer"
)]

use smg_policy_controller::{
    document::{CostMargin, PolicySpec, SpecError, API_VERSION},
    engine::{
        Candidate, Constraints, Engine, Health, LoadRow, Pick, PoolFacts, RequestFacts, Scorer,
        Scratch, Target, TieBand, WorkerId,
    },
    stages::{score_costs, Lifecycle, StageRegistry},
    tracking::{Lease, Side},
};

fn parse(score: &str, pick: &str) -> Result<PolicySpec, serde_yaml::Error> {
    serde_yaml::from_str(&format!(
        "apiVersion: {API_VERSION}\nprofiles:\n  p:\n    filter: [healthy]\n    score: [{score}]\n    pick: {pick}\n"
    ))
}

fn document(params: &str) -> PolicySpec {
    parse(&format!("{{least-cost: {{{params}}}}}"), "best").expect("a policy document")
}

fn engine(params: &str) -> Engine {
    document(params).validate().expect("a least-cost profile")
}

fn row(position: u32, in_flight: u32, load: Option<LoadRow>) -> Candidate {
    Candidate {
        position,
        id: WorkerId(u64::from(position) + 1),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        load,
        ..Candidate::default()
    }
}

fn reports<const N: usize>(loads: [(u32, u32); N]) -> [Candidate; N] {
    let mut positions = 0..;
    loads.map(|(running, queued)| {
        row(
            positions.next().expect("a position"),
            0,
            Some(LoadRow {
                running,
                queued,
                ..LoadRow::default()
            }),
        )
    })
}

fn decide(engine: &Engine, rows: &[Candidate], seed: u64, scratch: &mut Scratch) -> Pick {
    let facts = RequestFacts {
        seed,
        ..RequestFacts::default()
    };
    decide_with(engine, rows, &facts, scratch)
}

fn decide_with(
    engine: &Engine,
    rows: &[Candidate],
    facts: &RequestFacts<'_>,
    scratch: &mut Scratch,
) -> Pick {
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    *engine
        .decide(facts, &pool, rows, &Constraints::default(), scratch)
        .pick()
        .expect("the rows route")
}

fn scores(params: &str, rows: &[Candidate]) -> Vec<(u64, f32)> {
    scores_at(&engine(params), rows, 0)
}

/// The scores of a decision at `now_micros` on the host's clock.
fn scores_at(engine: &Engine, rows: &[Candidate], now_micros: u64) -> Vec<(u64, f32)> {
    let facts = RequestFacts {
        now_micros,
        ..RequestFacts::default()
    };
    let pick = decide_with(engine, rows, &facts, &mut Scratch::new());
    pick.trace.top[..usize::from(pick.trace.top_len)]
        .iter()
        .map(|&(id, score)| (id.0, score))
        .collect()
}

const SENT_AT: u64 = 85_000;

/// Tells the engine of one send to the row at `SENT_AT`, of `tokens` prompt tokens.
fn send(engine: &Engine, row: &Candidate, tokens: u32) {
    engine.on_dispatched(
        &Lease {
            issued_micros: SENT_AT,
            tokens,
            ..Lease::default()
        },
        &Target {
            id: row.id,
            dp_rank: row.dp_rank,
            ..Target::default()
        },
        Side::Single,
        0,
    );
}

#[test]
fn a_held_send_costs_a_row_the_request_coefficient() {
    let engine = engine("terms: {candidate.in_flight: 250, sent.requests: 1000}");
    let rows = [row(0, 0, None), row(1, 0, None), row(2, 8, None)];
    send(&engine, &rows[1], 317);
    // The costs are 0, 1000 and 2000: one send weighs four requests in flight.
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 1),
        [(1, 1.0), (2, 0.5), (3, 0.0)]
    );
}

#[test]
fn a_new_documents_engine_keeps_the_sends_of_the_standing_one() {
    let params = "terms: {candidate.in_flight: 250, sent.requests: 1000}";
    let standing = engine(params);
    let rows = [row(0, 0, None), row(1, 0, None), row(2, 8, None)];
    send(&standing, &rows[1], 317);
    let next = engine(params);
    next.succeed(&standing);
    assert_eq!(
        scores_at(&next, &rows, SENT_AT + 1),
        [(1, 1.0), (2, 0.5), (3, 0.0)]
    );
}

#[test]
fn sent_terms_alone_need_the_reports_and_no_rule_for_a_row_without_one() {
    let engine = engine("terms: {sent.requests: 3, sent.tokens: 1}");
    assert!(engine.needs().loads);
    assert!(!engine.profiles()[0].scoped());
}

/// A term that prices the sends by their tokens asks its host for each
/// request's size: the send of a request of text alone is counted by its
/// text's bytes, and a host that can send a request on without reading its
/// prompt states them only where a stage of the document asks. A term that
/// counts the sends asks for no size, and neither does a document that
/// names no send.
#[test]
fn a_term_that_prices_the_sends_by_their_tokens_asks_for_each_requests_size() {
    for (terms, sized) in [
        ("terms: {sent.requests: 3, sent.tokens: 1}", true),
        ("terms: {candidate.in_flight: 100, sent.tokens: 2}", true),
        ("terms: {sent.requests: 1000}", false),
        ("terms: {candidate.in_flight: 3}", false),
    ] {
        let engine = engine(terms);
        assert_eq!(engine.needs().prompt_size, sized, "{terms}");
        let stage = &engine.profiles()[0].scorers[0].scorer;
        assert_eq!(stage.needs().prompt_size, sized, "{terms}, the stage");
    }
}

#[test]
fn sent_tokens_adds_the_tokens_the_sends_stated() {
    let engine = engine("terms: {candidate.in_flight: 100, sent.tokens: 2}");
    let rows = [row(0, 0, None), row(1, 0, None), row(2, 16, None)];
    // The third send states no size and adds no token.
    for tokens in [317, 83, 0] {
        send(&engine, &rows[1], tokens);
    }
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 1),
        [(1, 1.0), (2, 0.5), (3, 0.0)]
    );
}

#[test]
fn missing_as_zero_keeps_the_sends_of_a_row_with_no_report() {
    let engine =
        engine("terms: {load.running: 10, sent.requests: 100}, unreported: missing-as-zero");
    let mut rows = reports([(0, 0), (0, 0), (20, 0)]);
    rows[1].load = None;
    send(&engine, &rows[1], 317);
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 1),
        [(1, 1.0), (2, 0.5), (3, 0.0)]
    );
}

#[test]
fn a_report_sampled_after_a_send_ends_its_charge() {
    let engine = engine("terms: {sent.requests: 1000}");
    let mut rows = reports([(0, 0), (0, 0)]);
    rows[1].load_sampled_at_micros = SENT_AT - 1;
    send(&engine, &rows[1], 317);
    assert_eq!(scores_at(&engine, &rows, SENT_AT + 2), [(1, 1.0), (2, 0.0)]);

    rows[1].load_sampled_at_micros = SENT_AT + 1;
    let mut tied = scores_at(&engine, &rows, SENT_AT + 2);
    tied.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(tied, [(1, 1.0), (2, 1.0)]);
}

#[test]
fn a_report_sampled_sooner_than_the_profiles_delay_leaves_a_sends_charge() {
    let mut spec = document("terms: {sent.requests: 1000}");
    spec.profiles
        .get_mut("p")
        .expect("the profile")
        .sent_report_delay_ms = Some(5);
    let engine = spec.validate().expect("a reader of sends under a delay");
    let mut rows = reports([(0, 0), (0, 0)]);
    send(&engine, &rows[1], 317);
    // Three milliseconds behind the send: the charge stands.
    rows[1].load_sampled_at_micros = SENT_AT + 3_000;
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 9_000),
        [(1, 1.0), (2, 0.0)]
    );
    // Five behind it: the report ends the charge.
    rows[1].load_sampled_at_micros = SENT_AT + 5_000;
    let mut tied = scores_at(&engine, &rows, SENT_AT + 9_000);
    tied.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(tied, [(1, 1.0), (2, 1.0)]);
}

#[test]
fn a_report_past_the_age_limit_still_ends_the_sends_before_it() {
    const NOW: u64 = SENT_AT + 2_000_000;
    let mut spec =
        document("terms: {load.running: 5, sent.requests: 100}, unreported: missing-as-zero");
    spec.view.max_report_age_ms = Some(1_000);
    let engine = spec
        .validate()
        .expect("a least-cost profile under an age limit");
    // The first two reports are too old for the terms: their running
    // requests add nothing. The first was sampled ahead of its row's send
    // and the second behind it.
    let mut rows = reports([(7, 0), (7, 0), (10, 0)]);
    rows[0].load_sampled_at_micros = SENT_AT - 1;
    rows[1].load_sampled_at_micros = SENT_AT + 1;
    rows[2].load_sampled_at_micros = NOW;
    for sent in [0, 1] {
        send(&engine, &rows[sent], 317);
    }
    // The costs are 100, 0 and 50.
    assert_eq!(
        scores_at(&engine, &rows, NOW),
        [(2, 1.0), (3, 0.5), (1, 0.0)]
    );
}

#[test]
fn the_profiles_time_limit_ends_a_charge_with_no_report() {
    let mut spec = document("terms: {candidate.in_flight: 1, sent.requests: 1000}");
    spec.profiles.get_mut("p").expect("the profile").sent_ttl_ms = Some(43);
    let engine = spec.validate().expect("a reader of sends under a limit");
    let rows = [row(0, 11, None), row(1, 0, None)];
    send(&engine, &rows[1], 317);
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 42_000),
        [(1, 1.0), (2, 0.0)]
    );
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 44_000),
        [(2, 1.0), (1, 0.0)]
    );
}

#[test]
fn over_least_prices_an_unreported_row_over_the_least_cost_with_its_sends() {
    let engine = engine(
        "terms: {load.running: 1, sent.requests: 24}, unreported: {over-least: {request_cost: 4}}",
    );
    let [first, second] = reports([(4, 0), (12, 0)]);
    let rows = [first, second, row(2, 2, None)];
    for sent in [0, 2] {
        send(&engine, &rows[sent], 317);
    }
    // The send turns the reported costs to 28 and 12. The third row has no
    // report: it costs 12 and its two requests in flight, without its send.
    assert_eq!(
        scores_at(&engine, &rows, SENT_AT + 1),
        [(2, 1.0), (3, 0.5), (1, 0.0)]
    );
}

#[test]
fn the_margin_ties_rows_by_their_costs_with_the_sends() {
    let engine = engine("terms: {candidate.in_flight: 1, sent.requests: 8}, margin: {absolute: 2}");
    let rows = [row(0, 10, None), row(1, 16, None), row(2, 20, None)];
    send(&engine, &rows[0], 317);
    // The costs are 18, 16 and 20: the first row ties the second by its send.
    let mut scored = scores_at(&engine, &rows, SENT_AT + 1);
    scored.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(scored, [(1, 1.0), (2, 1.0), (3, 0.0)]);
}

#[test]
fn the_sends_close_the_sum_with_the_requests_before_the_tokens() {
    let engine =
        engine("terms: {candidate.in_flight: 1e15, sent.requests: -1e15, sent.tokens: 0.03125}");
    let rows = [row(0, 0, None), row(1, 1, None)];
    send(&engine, &rows[1], 1);
    // The request term cancels the count in flight, and the token term then
    // stands alone. Added to either large number first, it would round away.
    assert_eq!(scores_at(&engine, &rows, SENT_AT + 1), [(1, 1.0), (2, 0.0)]);
}

#[test]
fn a_weighted_sum_orders_the_reports_and_puts_the_least_at_one() {
    let rows = reports([(7, 3), (3, 7), (5, 5)]);
    assert_eq!(
        scores(
            "terms: {load.running: 3, load.queued: 7}, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (3, 0.5), (2, 0.0)]
    );
}

#[test]
fn changing_coefficients_changes_the_order() {
    let rows = reports([(7, 3), (3, 7), (5, 5)]);
    assert_eq!(
        scores(
            "terms: {load.running: 7, load.queued: 3}, unreported: missing-as-zero",
            &rows
        ),
        [(2, 1.0), (3, 0.5), (1, 0.0)]
    );
}

#[test]
fn each_load_term_reads_its_own_field_in_the_reports_units() {
    for (name, set) in [
        (
            "load.running",
            (|r: &mut LoadRow, n| r.running = n) as fn(&mut LoadRow, u32),
        ),
        ("load.queued", |r, n| r.queued = n),
        ("load.waiting_uncached_tokens", |r, n| {
            r.waiting_uncached_tokens = n;
        }),
        ("load.kv_usage_permille", |r, n| {
            r.kv_usage_permille = n as u16;
        }),
        ("load.active_kv_permille", |r, n| {
            r.active_kv_permille = n as u16;
        }),
        ("load.resident_tokens", |r, n| r.resident_tokens = n),
    ] {
        let make = |(position, n)| {
            let mut report = LoadRow {
                resident_counted: true,
                ..LoadRow::default()
            };
            set(&mut report, n);
            row(position, 0, Some(report))
        };
        let rows = [(0, 57), (1, 54), (2, 63)].map(make);
        assert_eq!(
            scores(
                &format!(
                    "terms: {{{name}: 3}}, margin: {{absolute: 9}}, unreported: missing-as-zero"
                ),
                &rows
            ),
            [(1, 1.0), (2, 1.0), (3, 0.0)],
            "{name}"
        );
    }
}

#[test]
fn costs_add_in_field_name_order() {
    let mut rows = reports([(7, u32::MAX), (0, 0)]);
    rows[0].in_flight = u32::MAX;
    assert_eq!(
        scores(
            "terms: {load.running: 0.25, load.queued: -9.9e14, candidate.in_flight: 9.9e14}, unreported: missing-as-zero",
            &rows
        ),
        [(2, 1.0), (1, 0.0)]
    );
}

#[test]
fn the_absolute_margin_includes_its_edge() {
    let rows = reports([(113, 0), (118, 0), (120, 0)]);
    assert_eq!(
        scores(
            "terms: {load.running: 2}, margin: {absolute: 10}, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (2, 1.0), (3, 0.0)]
    );
}

/// A bare number under `margin` is the absolute margin: the document of
/// the map, read from YAML and from JSON, and written back as the map.
#[test]
fn a_bare_margin_is_the_absolute_margin() {
    let bare = document("terms: {load.running: 2}, margin: 10, unreported: missing-as-zero");
    let map =
        document("terms: {load.running: 2}, margin: {absolute: 10}, unreported: missing-as-zero");
    assert_eq!(bare, map);
    assert_eq!(
        document("terms: {candidate.in_flight: 1}, margin: 2.5"),
        document("terms: {candidate.in_flight: 1}, margin: {absolute: 2.5}")
    );
    let rows = reports([(113, 0), (118, 0), (120, 0)]);
    assert_eq!(
        scores(
            "terms: {load.running: 2}, margin: 10, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (2, 1.0), (3, 0.0)]
    );
    let json = serde_json::to_string(&bare).expect("the document writes as JSON");
    let parts = r#""margin":{"absolute":10.0,"relative":0.0}"#;
    assert!(json.contains(parts), "{json}");
    for number in ["10", "10.0"] {
        let written = json.replace(parts, &format!(r#""margin":{number}"#));
        let read: PolicySpec = serde_json::from_str(&written).expect("the JSON reads");
        assert_eq!(read, bare, "{number}");
    }
}

/// A bare margin holds the absolute margin's bounds, and a margin that is
/// neither a number nor the map is refused by the two forms it takes.
#[test]
fn a_bare_margin_is_bounded_and_no_third_form_reads() {
    for value in ["-1", "-0.25", ".nan", ".inf", "1.2e15"] {
        refused(
            &format!("terms: {{candidate.in_flight: 3}}, margin: {value}"),
            "margin.absolute",
            "a finite number from 0 to 1e15",
        );
    }
    for margin in ["wide", "true", "[10, 0]"] {
        let err = parse(
            &format!("{{least-cost: {{terms: {{candidate.in_flight: 3}}, margin: {margin}}}}}"),
            "best",
        )
        .expect_err("a margin of a third form");
        assert!(
            err.to_string()
                .contains("a number, the absolute margin, or a map of `absolute` and `relative`"),
            "{margin}: {err}"
        );
    }
}

#[test]
fn costs_one_apart_stay_distinct_at_large_magnitudes() {
    let rows = reports([(17_200_012, 0), (17_200_012, 2)]);
    assert_eq!(
        scores(
            "terms: {load.running: 2, load.queued: 0.5}, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (2, 0.0)]
    );
}

#[test]
fn a_score_rounded_up_outside_the_tie_stays_below_one() {
    let rows = reports([(0, 0), (1, 0), (268_435_456, 0)]);
    assert_eq!(
        scores(
            "terms: {load.running: 2}, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (2, 1.0_f32.next_down()), (3, 0.0)]
    );
}

#[test]
fn the_relative_margin_uses_the_magnitude_of_the_least_cost() {
    for (coefficient, running) in [(2, [80, 90, 91]), (-2, [80, 70, 69])] {
        let rows = reports(running.map(|n| (n, 0)));
        assert_eq!(
            scores(
                &format!("terms: {{load.running: {coefficient}}}, margin: {{relative: 0.125}}, unreported: missing-as-zero"),
                &rows
            ),
            [(1, 1.0), (2, 1.0), (3, 0.0)]
        );
    }
    let rows = reports([(80, 0), (93, 0), (94, 0)]);
    assert_eq!(
        scores("terms: {load.running: 2}, margin: {absolute: 6, relative: 0.125}, unreported: missing-as-zero", &rows),
        [(1, 1.0), (2, 1.0), (3, 0.0)]
    );
}

#[test]
fn best_draws_each_tied_row_about_as_often_and_never_draws_behind_them_first() {
    let engine =
        engine("terms: {load.running: 2}, margin: {absolute: 16}, unreported: missing-as-zero");
    let rows = reports([(18, 0), (22, 0), (27, 0), (26, 0)]);
    let mut scratch = Scratch::new();
    let mut chosen = [0; 4];
    for seed in 0..3_072 {
        let pick = decide(&engine, &rows, seed, &mut scratch);
        chosen[pick.targets.first().expect("a target").position as usize] += 1;
    }
    assert_eq!(chosen[2], 0, "{chosen:?}");
    for index in [0, 1, 3] {
        assert!((914..=1_134).contains(&chosen[index]), "{chosen:?}");
    }
}

#[test]
fn best_lists_the_rest_of_the_tie_then_the_remaining_rows_by_cost() {
    let engine =
        engine("terms: {load.running: 2}, margin: {absolute: 6}, unreported: missing-as-zero");
    let rows = reports([(21, 0), (12, 0), (18, 0), (15, 0)]);
    let mut scratch = Scratch::new();
    let mut both = [false; 2];
    for seed in 0..48 {
        let pick = decide(&engine, &rows, seed, &mut scratch);
        let positions: Vec<_> = pick.targets.iter().map(|t| t.position).collect();
        match positions.as_slice() {
            [1, 3, 2, 0] => both[0] = true,
            [3, 1, 2, 0] => both[1] = true,
            _ => panic!("unexpected target order {positions:?}"),
        }
    }
    assert_eq!(both, [true; 2]);
}

fn mixed_reports() -> [Candidate; 3] {
    [
        row(
            0,
            2,
            Some(LoadRow {
                running: 7,
                ..LoadRow::default()
            }),
        ),
        row(1, 3, None),
        row(
            2,
            3,
            Some(LoadRow {
                running: 16,
                ..LoadRow::default()
            }),
        ),
    ]
}

#[test]
fn over_least_prices_missing_rows_from_the_least_reported_cost() {
    // The reported rows cost 29 and 71. The missing row costs 29 + 3 * 7.
    let params = "terms: {candidate.in_flight: -3, load.running: 5}, unreported: {over-least: {request_cost: 7}}";
    let mut rows = mixed_reports();
    assert_eq!(scores(params, &rows), [(1, 1.0), (2, 0.5), (3, 0.0)]);
    rows[0].health.healthy = false;
    assert_eq!(scores(params, &rows), [(3, 1.0), (2, 0.0)]);
}

#[test]
fn over_least_prices_in_flight_from_zero_when_no_row_reports() {
    let rows = [row(0, 2, None), row(1, 3, None), row(2, 7, None)];
    for (price, want) in [
        (7, [(1, 1.0), (2, 0.8), (3, 0.0)]),
        (0, [(1, 1.0), (2, 1.0), (3, 1.0)]),
    ] {
        assert_eq!(
            scores(&format!("terms: {{load.running: 5}}, unreported: {{over-least: {{request_cost: {price}}}}}"), &rows),
            want
        );
    }
}

#[test]
fn missing_as_zero_keeps_the_terms_a_row_can_supply() {
    assert_eq!(
        scores(
            "terms: {candidate.in_flight: -3, load.running: 5}, unreported: missing-as-zero",
            &mixed_reports()
        ),
        [(2, 1.0), (1, 0.525), (3, 0.0)]
    );
}

#[test]
fn uncounted_resident_tokens_make_a_row_unreported_only_when_named() {
    let mut rows = reports([(7, 0), (10, 0), (4, 0)]);
    rows[0].load.as_mut().expect("a report").resident_counted = true;
    rows[1].load.as_mut().expect("a report").resident_counted = true;
    rows[2].load.as_mut().expect("a report").resident_tokens = 70;
    rows[2].in_flight = 2;
    let over_least = "unreported: {over-least: {request_cost: 3}}";
    assert_eq!(
        scores(&format!("terms: {{load.running: 2}}, {over_least}"), &rows),
        [(3, 1.0), (1, 0.5), (2, 0.0)]
    );
    assert_eq!(
        scores(
            &format!("terms: {{load.running: 2, load.resident_tokens: 3}}, {over_least}"),
            &rows
        ),
        [(1, 1.0), (2, 0.0), (3, 0.0)]
    );
}

/// A report of a queue's depth alone writes zero waiting tokens. Beside a
/// queue that zero is a missing value, priced by the document's rule. Over an
/// empty queue it is a count.
#[test]
fn zero_waiting_tokens_beside_a_queue_are_a_missing_value() {
    let report = |queued, waiting_uncached_tokens| {
        Some(LoadRow {
            queued,
            waiting_uncached_tokens,
            ..LoadRow::default()
        })
    };
    // Tokens over a queue, a depth alone, and an empty queue.
    let rows = [
        row(0, 0, report(8, 40)),
        row(1, 4, report(16, 0)),
        row(2, 1, report(0, 0)),
    ];
    // The depth alone costs the least reported cost, 0, plus 4 * 5.
    assert_eq!(
        scores(
            "terms: {load.waiting_uncached_tokens: 1}, unreported: {over-least: {request_cost: 5}}",
            &rows
        ),
        [(3, 1.0), (2, 0.5), (1, 0.0)]
    );
    // Its tokens add nothing and its queue still counts: 64, 48 and 0.
    assert_eq!(
        scores(
            "terms: {load.queued: 3, load.waiting_uncached_tokens: 1}, unreported: missing-as-zero",
            &rows
        ),
        [(3, 1.0), (2, 0.25), (1, 0.0)]
    );
    // With no such term the same report is a reported row: 8, 16 and 0.
    assert_eq!(
        scores(
            "terms: {load.queued: 1}, unreported: {over-least: {request_cost: 5}}",
            &rows
        ),
        [(3, 1.0), (1, 0.5), (2, 0.0)]
    );
}

/// A report past the document's age limit is no report to this scorer: its
/// row is costed as an unreported one. A report at the limit is read.
#[test]
fn a_report_past_the_documents_age_limit_costs_as_no_report() {
    const NOW: u64 = 9_000_000;
    const LIMIT_MS: u64 = 1_000;
    let mut spec =
        document("terms: {load.running: 5}, unreported: {over-least: {request_cost: 5}}");
    spec.view.max_report_age_ms = Some(LIMIT_MS);
    let engine = spec.validate().expect("a least-cost profile under a limit");
    let facts = RequestFacts {
        now_micros: NOW,
        ..RequestFacts::default()
    };
    let scored = |rows: &[Candidate]| {
        let pool = PoolFacts {
            pool_size: rows.len() as u32,
            ..PoolFacts::default()
        };
        let mut scratch = Scratch::new();
        let pick = *engine
            .decide(&facts, &pool, rows, &Constraints::default(), &mut scratch)
            .pick()
            .expect("the rows route");
        pick.trace.top[..usize::from(pick.trace.top_len)]
            .iter()
            .map(|&(id, score)| (id.0, score))
            .collect::<Vec<_>>()
    };
    // Read, the rows cost 35, 65 and 95.
    let mut rows = reports([(7, 0), (13, 0), (19, 0)]);
    for row in &mut rows {
        row.load_sampled_at_micros = NOW;
    }
    rows[1].in_flight = 3;
    rows[1].load_sampled_at_micros = NOW - LIMIT_MS * 1_000;
    assert_eq!(
        scored(&rows),
        [(1, 1.0), (2, 0.5), (3, 0.0)],
        "at the limit"
    );
    // A microsecond older, the second row has no report: it costs the
    // least reported cost and its three requests in flight at 5 each, 50.
    rows[1].load_sampled_at_micros -= 1;
    assert_eq!(scored(&rows), [(1, 1.0), (2, 0.75), (3, 0.0)], "past it");
    // With no limit the same report is read whatever its age.
    spec.view.max_report_age_ms = None;
    let unlimited = spec.validate().expect("a least-cost profile");
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    let pick = *unlimited
        .decide(
            &facts,
            &pool,
            &rows,
            &Constraints::default(),
            &mut Scratch::new(),
        )
        .pick()
        .expect("the rows route");
    assert_eq!(pick.trace.top[1], (WorkerId(2), 0.5), "no limit");
}

#[test]
fn all_rows_at_one_cost_score_one() {
    let rows = reports([(12, 0), (12, 0), (12, 0)]);
    assert_eq!(
        scores(
            "terms: {load.running: 2}, unreported: missing-as-zero",
            &rows
        ),
        [(1, 1.0), (2, 1.0), (3, 1.0)]
    );
}

#[test]
fn a_count_term_asks_for_no_reports_and_a_load_term_asks_for_reports() {
    assert!(!engine("terms: {candidate.in_flight: 3}").needs().loads);
    assert!(
        engine("terms: {load.running: 2}, unreported: missing-as-zero")
            .needs()
            .loads
    );
    let rows = [row(0, 10, None), row(1, 5, None), row(2, 15, None)];
    assert_eq!(
        scores("terms: {candidate.in_flight: 3}", &rows),
        [(2, 1.0), (1, 0.5), (3, 0.0)]
    );
}

#[test]
fn a_picker_can_draw_before_least_cost_scores() {
    let engine = parse(
        "{least-cost: {terms: {candidate.in_flight: 3}}}",
        "power-of-two",
    )
    .expect("a policy document")
    .validate()
    .expect("a least-cost profile");
    let rows: Vec<_> = (0..34).map(|i| row(i, i, None)).collect();
    assert!(decide(&engine, &rows, 7, &mut Scratch::new())
        .trace
        .drew_first());
}

fn refused(params: &str, field: &str, why: &str) {
    let err = document(params).validate().expect_err("a bad parameter");
    let SpecError::BadField { path, reason, .. } = err else {
        panic!("{params}: {err}");
    };
    assert_eq!(
        path,
        format!("profiles.p.score[0].least-cost.{field}"),
        "{params}"
    );
    assert_eq!(reason, why, "{params}");
}

#[test]
fn terms_need_a_known_field_and_a_bounded_nonzero_coefficient() {
    refused(
        "terms: {}",
        "terms",
        "at least one field with its coefficient",
    );
    for name in [
        "load.quue",
        "load.live_rate",
        "load.max_running",
        "sent.queued",
    ] {
        refused(
            &format!("terms: {{{name}: 3}}"),
            "terms",
            "a field this build reads",
        );
    }
    for value in ["0", "-0.0", ".nan", ".inf", "-.inf", "1.1e15", "-1.1e15"] {
        refused(
            &format!("terms: {{candidate.in_flight: {value}}}"),
            "terms.candidate.in_flight",
            "a finite number other than zero, of magnitude at most 1e15",
        );
    }
}

#[test]
fn both_margin_parts_are_bounded_and_nonnegative() {
    for (part, over, why) in [
        ("absolute", "1.2e15", "a finite number from 0 to 1e15"),
        ("relative", "1.2e6", "a finite number from 0 to 1e6"),
    ] {
        for value in ["-0.25", ".nan", ".inf", over] {
            refused(
                &format!("terms: {{candidate.in_flight: 3}}, margin: {{{part}: {value}}}"),
                &format!("margin.{part}"),
                why,
            );
        }
    }
}

#[test]
fn a_report_term_requires_a_rule_for_missing_values() {
    refused(
        "terms: {load.queued: 3}",
        "unreported",
        "what a row with no report, or with no fresh value of a signal, costs, which a term that reads a report or a signal needs: over-least or missing-as-zero",
    );
}

#[test]
fn the_missing_report_price_is_bounded_and_nonnegative() {
    for value in ["-0.25", ".nan", ".inf", "1.2e15"] {
        refused(
            &format!(
                "terms: {{load.running: 5}}, unreported: {{over-least: {{request_cost: {value}}}}}"
            ),
            "unreported.over-least.request_cost",
            "a finite number from 0 to 1e15",
        );
    }
}

#[test]
fn least_cost_weight_is_bounded_and_weights_the_combined_score() {
    refused(
        "terms: {candidate.in_flight: 3}, weight: 1.2e15",
        "weight",
        "a finite number above zero and at most 1e15",
    );
    let engine = parse(
        "{least-cost: {terms: {candidate.in_flight: -3}, weight: 3}}, least-requests",
        "best",
    )
    .expect("a policy document")
    .validate()
    .expect("both scorers validate");
    let rows = [row(0, 7, None), row(1, 2, None)];
    assert_eq!(
        decide(&engine, &rows, 3, &mut Scratch::new()).trace.top[..2],
        [(WorkerId(1), 0.75), (WorkerId(2), 0.25)]
    );
}

#[test]
fn a_bare_least_cost_is_refused_by_name_with_its_required_terms() {
    let err = parse("least-cost", "best")
        .expect("the bare form parses")
        .validate()
        .expect_err("the scorer requires terms");
    let SpecError::BadField {
        path,
        value,
        reason,
    } = err
    else {
        panic!("{err}");
    };
    assert_eq!(path, "profiles.p.score[0]");
    assert_eq!(value, "least-cost");
    assert_eq!(reason, "a scorer with its terms: { least-cost: { terms } }");
}

#[test]
fn the_document_round_trips_in_canonical_order_and_in_yaml() {
    let written = document("terms: {load.queued: 23, load.running: 3}, margin: {absolute: 19}, weight: 0.75, unreported: {over-least: {request_cost: 17}}");
    written.validate().expect("the document validates");
    let json = serde_json::to_string(&written).expect("the document writes as JSON");
    assert!(
        json.contains(r#""terms":{"load.queued":23.0,"load.running":3.0}"#),
        "{json}"
    );
    let read: PolicySpec = serde_json::from_str(&json).expect("the JSON reads back");
    assert_eq!(written, read);
    assert_eq!(
        written.digest().expect("a digest"),
        read.digest().expect("a digest")
    );
    let yaml = serde_yaml::to_string(&written).expect("the document writes as YAML");
    assert!(yaml.contains("over-least:"), "{yaml}");
    assert_eq!(
        written,
        serde_yaml::from_str(&yaml).expect("the YAML reads back")
    );
}

/// A document whose terms read no report says nothing of a missing value,
/// and is written back with no such key.
#[test]
fn a_document_with_no_rule_for_missing_values_writes_none_back() {
    let written = document("terms: {candidate.in_flight: 3}");
    written.validate().expect("the document validates");
    let json = serde_json::to_string(&written).expect("the document writes as JSON");
    assert!(!json.contains("unreported"), "{json}");
    let read: PolicySpec = serde_json::from_str(&json).expect("the JSON reads back");
    assert_eq!(written, read);
    let yaml = serde_yaml::to_string(&written).expect("the document writes as YAML");
    assert!(!yaml.contains("unreported"), "{yaml}");
    assert_eq!(
        written,
        serde_yaml::from_str(&yaml).expect("the YAML reads back")
    );
}

#[test]
fn a_term_written_twice_is_refused() {
    let err = parse(
        "{least-cost: {terms: {load.running: 3, load.running: 5}}}",
        "best",
    )
    .expect_err("the repeated term is refused");
    assert!(
        err.to_string()
            .contains("the key `load.running` is written twice"),
        "{err}"
    );
}

#[test]
fn the_least_cost_document_refuses_unknown_parameters() {
    for params in [
        "terms: {candidate.in_flight: 3}, offset: 5",
        "terms: {candidate.in_flight: 3}, margin: {relative: 0.2, other: 9}",
        "terms: {load.running: 3}, unreported: {over-least: {request_cost: 3, other: 9}}",
    ] {
        assert!(
            parse(&format!("{{least-cost: {{{params}}}}}"), "best").is_err(),
            "{params}"
        );
    }
}

#[derive(Debug)]
struct OwnCost;

impl Lifecycle for OwnCost {}

impl Scorer for OwnCost {
    fn name(&self) -> &'static str {
        "own-cost"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let mut next = 0;
        let band = score_costs(
            out,
            CostMargin {
                absolute: 12.0,
                ..CostMargin::default()
            },
            |i| {
                assert_eq!(i, next);
                next += 1;
                f64::from(rows[kept[i] as usize].in_flight) * 4.0
            },
        );
        assert_eq!(next, out.len());
        band
    }
}

#[test]
fn a_registered_scorer_uses_the_same_tie_and_a_zero_band() {
    let rows = [row(0, 9, None), row(1, 12, None), row(2, 16, None)];
    let mut out = [0.0; 3];
    let band = OwnCost.score(&RequestFacts::default(), &rows, &[0, 1, 2], &mut out);
    assert_eq!(band, TieBand(0.0));
    assert_eq!(out, [1.0, 1.0, 0.0]);

    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("own-cost", || Box::new(OwnCost))
        .expect("a new name");
    let own = parse("own-cost", "best")
        .expect("a policy document")
        .validate_in(&catalog, &[])
        .expect("the scorer is registered");
    let builtin = engine("terms: {candidate.in_flight: 4}, margin: {absolute: 12}");
    let mut scratch = Scratch::new();
    assert_eq!(
        decide(&own, &rows, 19, &mut scratch),
        decide(&builtin, &rows, 19, &mut scratch)
    );
}

#[test]
fn a_cost_callback_can_score_another_slice() {
    let mut outer = [0.0; 2];
    score_costs(&mut outer, CostMargin::default(), |i| {
        let mut inner = [0.0; 3];
        score_costs(&mut inner, CostMargin::default(), |j| [14.0, 27.0, 14.0][j]);
        assert_eq!(inner, [1.0, 0.0, 1.0]);
        [12.0, 9.0][i]
    });
    assert_eq!(outer, [0.0, 1.0]);
    assert_eq!(
        score_costs(&mut [], CostMargin::default(), |_| panic!("no positions")),
        TieBand(0.0)
    );
}
