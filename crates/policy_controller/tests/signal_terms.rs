//! A document prices a signal in `least-cost` and bounds one in `has-room`:
//! the term `signal.<name>`, and a capacity that is a signal's name.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use smg_policy_controller::{
    document::{PolicySpec, SpecError, API_VERSION},
    engine::{
        Candidate, Constraints, Decision, Engine, Health, LoadRow, Pair, PairPools, Phase,
        PoolFacts, RequestFacts, Scratch, Target, WorkerId,
    },
    load::{SignalCell, SignalCells},
    stages::Lifecycle,
    tracking::{Lease, Side},
};

/// The decisions' instant on the host's clock.
const NOW: u64 = 10_000_000;
/// The signals the documents declare. In name order `PENDING` is column 0,
/// `CAPACITY` column 1 and `USED` column 2. `PENDING` ends at half a
/// second. The other two are the capacity names a worker's report offers.
const PENDING: &str = "kv_load_pending_tokens";
const CAPACITY: &str = "kv_tokens_capacity";
const USED: &str = "kv_tokens_used";
const SIGNALS: usize = 3;

/// A document that declares the three signals, with one profile of
/// `filters` and `score`.
fn document(filters: &str, score: &str) -> PolicySpec {
    serde_yaml::from_str(&format!(
        "apiVersion: {API_VERSION}
signals:
  {PENDING}: {{ unit: tokens, max_age_ms: 500 }}
  {CAPACITY}: {{ unit: tokens }}
  {USED}: {{ unit: tokens }}
profiles:
  p:
    filter: [{filters}]
    score: [{score}]
    pick: best
"
    ))
    .expect("a policy document")
}

fn least_cost(params: &str) -> Engine {
    document("healthy", &format!("{{least-cost: {{{params}}}}}"))
        .validate()
        .expect("a least-cost profile")
}

fn has_room(rule: &str) -> Engine {
    document(
        &format!("healthy, {{has-room: {{{rule}}}}}"),
        "least-requests",
    )
    .validate()
    .expect("a has-room profile")
}

fn refused(filters: &str, score: &str) -> SpecError {
    document(filters, score)
        .validate()
        .expect_err("a refused document")
}

fn row(position: u32, in_flight: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(u64::from(position) + 1),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        ..Candidate::default()
    }
}

/// A value sampled `age_ms` before the decisions' instant.
fn aged(value: f64, age_ms: u64) -> SignalCell {
    SignalCell {
        value,
        sampled_at_micros: NOW - age_ms * 1_000,
    }
}

fn fresh(value: f64) -> SignalCell {
    aged(value, 0)
}

/// The cells of rows that each hold `PENDING`, `CAPACITY` and `USED`, in
/// that order.
fn cells(rows: &[[SignalCell; SIGNALS]]) -> SignalCells {
    SignalCells::new(rows.iter().flatten().copied().collect(), SIGNALS)
}

/// The cells of rows that hold `PENDING` alone.
fn pending(values: &[SignalCell]) -> SignalCells {
    let rows: Vec<_> = values
        .iter()
        .map(|&pending| [pending, SignalCell::NONE, SignalCell::NONE])
        .collect();
    cells(&rows)
}

/// The cells of rows that hold a capacity and what is used of it.
fn rooms(values: &[(SignalCell, SignalCell)]) -> SignalCells {
    let rows: Vec<_> = values
        .iter()
        .map(|&(capacity, used)| [SignalCell::NONE, capacity, used])
        .collect();
    cells(&rows)
}

fn facts<'a>(cells: Option<&'a SignalCells>, prompt_tokens: u32) -> RequestFacts<'a> {
    RequestFacts {
        now_micros: NOW,
        prompt_tokens,
        signal_cells: cells,
        ..RequestFacts::default()
    }
}

fn decide(engine: &Engine, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Decision {
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    engine.decide(
        facts,
        &pool,
        rows,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// The rows a decision scored highest, each worker's id with its score.
fn scores(engine: &Engine, cells: &SignalCells, rows: &[Candidate]) -> Vec<(u64, f32)> {
    scores_of(engine, &facts(Some(cells), 0), rows)
}

fn scores_of(engine: &Engine, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Vec<(u64, f32)> {
    let decision = decide(engine, facts, rows);
    let pick = decision.pick().expect("the rows route");
    pick.trace.top[..usize::from(pick.trace.top_len)]
        .iter()
        .map(|&(id, score)| (id.0, score))
        .collect()
}

/// The rows the profile's rule keeps, by their positions: the filter
/// behind `healthy`.
fn kept(engine: &Engine, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Vec<u32> {
    kept_at(engine, 1, facts, rows)
}

/// The rows the profile's filter at `at` keeps, by their positions. The
/// filter's pass and its word on each row alone must agree.
fn kept_at(engine: &Engine, at: usize, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Vec<u32> {
    let rule = &engine.profiles()[0].filters[at];
    let cx = Constraints::default();
    let mut kept: Vec<u32> = rows.iter().map(|row| row.position).collect();
    rule.retain(facts, rows, &cx, &mut kept);
    let alone: Vec<u32> = rows
        .iter()
        .filter(|row| rule.keep(facts, row, &cx))
        .map(|row| row.position)
        .collect();
    assert_eq!(kept, alone, "the pass and the row's own word");
    kept
}

const FIRST_SECOND_THIRD: [(u64, f32); 3] = [(1, 1.0), (2, 0.5), (3, 0.0)];
const SECOND_FIRST_THIRD: [(u64, f32); 3] = [(2, 1.0), (1, 0.5), (3, 0.0)];

#[test]
fn a_signal_term_makes_its_row_dearer_by_the_coefficient_times_the_value() {
    let engine = least_cost(&format!(
        "terms: {{candidate.in_flight: 100, signal.{PENDING}: 0.5}}, unreported: missing-as-zero"
    ));
    let rows = [row(0, 5), row(1, 0), row(2, 15)];
    // The costs are 500, 1000 and 1500: two thousand pending tokens weigh
    // ten requests in flight.
    let cells = pending(&[fresh(0.0), fresh(2_000.0), fresh(0.0)]);
    assert_eq!(scores(&engine, &cells, &rows), FIRST_SECOND_THIRD);
}

#[test]
fn a_signal_term_closes_the_sum_behind_the_sends() {
    let engine = least_cost(&format!(
        "terms: {{sent.requests: 1000, signal.{PENDING}: 1}}, unreported: missing-as-zero"
    ));
    let rows = [row(0, 0), row(1, 0), row(2, 0)];
    engine.on_dispatched(
        &Lease {
            issued_micros: NOW - 7,
            tokens: 317,
            ..Lease::default()
        },
        &Target {
            id: rows[1].id,
            ..Target::default()
        },
        Side::Single,
        0,
    );
    // The costs are 500, 1000 and 1500: the second row holds one send.
    let cells = pending(&[fresh(500.0), fresh(0.0), fresh(1_500.0)]);
    assert_eq!(scores(&engine, &cells, &rows), FIRST_SECOND_THIRD);
    assert!(engine.needs().loads, "the sends need the reports");
}

/// The two stages that name a signal ask for each request's size where a
/// term prices the sends by their tokens, as the stages that name no signal
/// do: a send of text alone is counted by its text's bytes, which a host
/// that can send a request on without reading its prompt states only where
/// a stage asks. A term that counts the sends asks for no size.
#[test]
fn a_stage_that_names_a_signal_asks_for_each_requests_size_where_it_prices_the_sends() {
    let priced = |sent: &str| {
        least_cost(&format!(
            "terms: {{{sent}: 1, signal.{PENDING}: 1}}, unreported: missing-as-zero"
        ))
    };
    assert!(priced("sent.tokens").needs().prompt_size);
    assert!(!priced("sent.requests").needs().prompt_size);
    let bounded = |sent: &str| {
        has_room(&format!(
            "used: {{{sent}: 1, signal.{PENDING}: 1}}, capacity: 1000"
        ))
    };
    assert!(bounded("sent.tokens").needs().prompt_size);
    assert!(!bounded("sent.requests").needs().prompt_size);
}

#[test]
fn a_row_with_no_value_for_a_named_signal_is_unreported_for_the_scorer() {
    let rows = [row(0, 0), row(1, 1), row(2, 0)];
    let cells = pending(&[fresh(350.0), SignalCell::NONE, fresh(700.0)]);
    // Over the least: the second row costs the least reported sum and one
    // request in flight, 350 and 175.
    let over_least = least_cost(&format!(
        "terms: {{signal.{PENDING}: 1}}, unreported: {{over-least: {{request_cost: 175}}}}"
    ));
    assert_eq!(scores(&over_least, &cells, &rows), FIRST_SECOND_THIRD);
    // Missing as zero: its term adds nothing, and it costs the least.
    let as_zero = least_cost(&format!(
        "terms: {{signal.{PENDING}: 1}}, unreported: missing-as-zero"
    ));
    assert_eq!(scores(&as_zero, &cells, &rows), SECOND_FIRST_THIRD);
}

#[test]
fn where_every_row_lacks_the_signal_over_least_picks_the_least_requests_in_flight() {
    let engine = least_cost(&format!(
        "terms: {{signal.{PENDING}: 1}}, unreported: {{over-least: {{request_cost: 175}}}}"
    ));
    let rows = [row(0, 3), row(1, 1), row(2, 2)];
    // No cells on the facts, no value in a cell, and a value past its age:
    // each row is priced by its requests in flight over a least of zero.
    for cells in [
        None,
        Some(pending(&[SignalCell::NONE; 3])),
        Some(pending(&[aged(1.0, 501); 3])),
    ] {
        let mut scores = scores_of(&engine, &facts(cells.as_ref(), 0), &rows);
        scores.sort_by_key(|&(id, _)| id);
        assert_eq!(scores, [(1, 0.0), (2, 1.0), (3, 0.5)]);
    }
}

#[test]
fn a_value_past_its_maximum_age_is_no_value() {
    let engine = least_cost(&format!(
        "terms: {{signal.{PENDING}: 1}}, unreported: {{over-least: {{request_cost: 175}}}}"
    ));
    let rows = [row(0, 0), row(1, 1), row(2, 0)];
    // At the signal's maximum age the second row's value is read: it
    // costs nothing.
    let at_the_age = pending(&[fresh(350.0), aged(0.0, 500), fresh(700.0)]);
    assert_eq!(scores(&engine, &at_the_age, &rows), SECOND_FIRST_THIRD);
    // A millisecond past it the row has no value: it is priced over the
    // least.
    let past_it = pending(&[fresh(350.0), aged(0.0, 501), fresh(700.0)]);
    assert_eq!(scores(&engine, &past_it, &rows), FIRST_SECOND_THIRD);
}

#[test]
fn the_least_reported_cost_is_taken_with_the_signal_terms_in_it() {
    let engine = least_cost(&format!(
        "terms: {{load.running: 100, signal.{PENDING}: 1}}, \
         unreported: {{over-least: {{request_cost: 50}}}}"
    ));
    let report = |running| {
        Some(LoadRow {
            running,
            ..LoadRow::default()
        })
    };
    let mut rows = [row(0, 0), row(1, 0), row(2, 5)];
    rows[0].load = report(1);
    rows[1].load = report(4);
    // By its fields the first row is the least, at 100 against 400. The
    // signal turns them: 1000 against 500. The third row has no report
    // and follows the new least: 500 and five requests in flight, 750.
    let turned = [(2, 1.0), (3, 0.5), (1, 0.0)];
    let cells = pending(&[fresh(900.0), fresh(100.0), fresh(0.0)]);
    assert_eq!(scores(&engine, &cells, &rows), turned);
    // A third row that reports and holds no value of the signal is left
    // open the same way: the sum its report made is dropped.
    rows[2].load = report(9);
    let cells = pending(&[fresh(900.0), fresh(100.0), SignalCell::NONE]);
    assert_eq!(scores(&engine, &cells, &rows), turned);
}

#[test]
fn a_signals_capacity_keeps_the_rows_under_the_bound() {
    let engine = has_room(&format!(
        "used: {{signal.{USED}: 1, request.prompt_tokens: 1}}, \
         capacity: signal.{CAPACITY}, headroom: 0.25"
    ));
    let rows: Vec<_> = (0..8).map(|position| row(position, 0)).collect();
    let cells = rooms(&[
        // At the bound of a capacity of 1000, 750, with the request's 100.
        (fresh(1_000.0), fresh(650.0)),
        // One token over it.
        (fresh(1_000.0), fresh(651.0)),
        // A larger capacity holds what the smaller did not.
        (fresh(2_000.0), fresh(1_400.0)),
        // No value of the capacity, and no value of what is used.
        (SignalCell::NONE, fresh(1.0)),
        (fresh(1_000.0), SignalCell::NONE),
        // A capacity of zero and one under zero hold nothing.
        (fresh(0.0), fresh(0.0)),
        (fresh(-5.0), fresh(-900.0)),
        // A small capacity holds what fits it, with a token to spare.
        (fresh(400.0), fresh(199.0)),
    ]);
    assert_eq!(kept(&engine, &facts(Some(&cells), 100), &rows), [0, 2, 7]);
    // A request one token larger fits the last row alone.
    assert_eq!(kept(&engine, &facts(Some(&cells), 101), &rows), [7]);
}

#[test]
fn a_rule_sums_a_signal_under_a_written_capacity() {
    let engine = has_room(&format!(
        "used: {{candidate.in_flight: 100, signal.{PENDING}: 1}}, capacity: 1000"
    ));
    let rows = [row(0, 2), row(1, 2), row(2, 0), row(3, 0)];
    // 800 pending tokens and two requests in flight are at the bound. A
    // token more is over it. A value past its age is no value.
    let cells = pending(&[
        fresh(800.0),
        fresh(801.0),
        aged(1.0, 501),
        aged(1_000.0, 500),
    ]);
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [0, 3]);
}

#[test]
fn a_rule_that_names_a_signal_counts_the_profiles_sends() {
    let engine = has_room(&format!(
        "used: {{sent.tokens: 1, signal.{PENDING}: 1}}, capacity: 1000"
    ));
    let rows = [row(0, 0), row(1, 0)];
    let cells = pending(&[fresh(700.0), fresh(700.0)]);
    // No send is on file: each row holds its 700 pending tokens.
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [0, 1]);
    engine.on_dispatched(
        &Lease {
            issued_micros: NOW - 7,
            tokens: 301,
            ..Lease::default()
        },
        &Target {
            id: rows[0].id,
            ..Target::default()
        },
        Side::Single,
        0,
    );
    // A send of 301 tokens puts the first row a token over the bound.
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [1]);
}

#[test]
fn the_requests_size_alone_under_a_signals_capacity_keeps_the_rows_it_fits() {
    let engine = has_room(&format!(
        "used: {{request.prompt_tokens: 1}}, capacity: signal.{CAPACITY}"
    ));
    let rows = [row(0, 0), row(1, 0), row(2, 0)];
    let holds = |capacity: f64| (fresh(capacity), SignalCell::NONE);
    let cells = rooms(&[holds(4_096.0), holds(8_192.0), holds(16_384.0)]);
    assert_eq!(kept(&engine, &facts(Some(&cells), 4_096), &rows), [0, 1, 2]);
    // A larger request keeps fewer.
    assert_eq!(kept(&engine, &facts(Some(&cells), 4_097), &rows), [1, 2]);
    assert_eq!(
        kept(&engine, &facts(Some(&cells), 16_385), &rows),
        [0u32; 0]
    );
}

/// A rule that names a signal reserves room for the answer as a rule that
/// names none does: the request's output limit times its coefficient
/// counts beside what the signal says a worker uses, under a capacity a
/// signal holds.
#[test]
fn the_output_limit_reserves_room_in_a_rule_that_names_a_signal() {
    let engine = has_room(&format!(
        "used: {{signal.{USED}: 1, request.max_output_tokens: 0.25, request.prompt_tokens: 1}}, \
         capacity: signal.{CAPACITY}"
    ));
    let rows = [row(0, 0), row(1, 0)];
    let cells = rooms(&[
        (fresh(1_000.0), fresh(700.0)),
        (fresh(1_000.0), fresh(701.0)),
    ]);
    let with_limit = |limit| RequestFacts {
        max_output_tokens: limit,
        ..facts(Some(&cells), 100)
    };
    // A prompt of 100 and a quarter of an answer of 796 put the second row
    // on its capacity.
    assert_eq!(kept(&engine, &with_limit(796), &rows), [0, 1]);
    // Four tokens more of answer are one token more: the second row is
    // over its capacity, and the first on it.
    assert_eq!(kept(&engine, &with_limit(800), &rows), [0]);
}

/// The request's output limit alone is a rule under a signal's capacity,
/// as its size alone is: each worker's capacity bounds the answer.
#[test]
fn the_output_limit_alone_under_a_signals_capacity_keeps_the_rows_it_fits() {
    let engine = has_room(&format!(
        "used: {{request.max_output_tokens: 1}}, capacity: signal.{CAPACITY}"
    ));
    let rows = [row(0, 0), row(1, 0)];
    let holds = |capacity: f64| (fresh(capacity), SignalCell::NONE);
    let cells = rooms(&[holds(4_096.0), holds(8_192.0)]);
    let with_limit = |limit| RequestFacts {
        max_output_tokens: limit,
        ..facts(Some(&cells), 0)
    };
    assert_eq!(kept(&engine, &with_limit(4_096), &rows), [0, 1]);
    // An answer one token longer fits the larger capacity alone.
    assert_eq!(kept(&engine, &with_limit(4_097), &rows), [1]);
}

#[test]
fn a_signal_alone_under_a_number_is_a_rule_and_the_requests_size_alone_is_none() {
    // A signal reads a worker: summed alone under a number it is a rule.
    let engine = has_room(&format!("used: {{signal.{PENDING}: 1}}, capacity: 1000"));
    let rows = [row(0, 0), row(1, 0)];
    let cells = pending(&[fresh(1_000.0), fresh(1_001.0)]);
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [0]);
    // The request's size alone reads nothing of a worker there, and its
    // path refuses it. Under a signal's capacity it is a rule, as the case
    // above holds.
    let error = refused(
        "healthy, {has-room: {used: {request.prompt_tokens: 1}, capacity: 1000}}",
        "least-requests",
    );
    assert_eq!(error.field_path(), "profiles.p.filter[1].has-room.used");
    assert!(
        matches!(&error, SpecError::BadField { value, reason, .. } if value == "request.prompt_tokens" && *reason == "at least one field or signal of the worker beside the request's own values"),
        "{error}"
    );
}

#[test]
fn a_rule_that_names_a_signal_and_one_that_names_none_each_keep_what_they_kept_alone() {
    let by_report = "{has-room: {used: {load.resident_tokens: 1}, capacity: 1000}}";
    let by_signal =
        format!("{{has-room: {{used: {{signal.{USED}: 1}}, capacity: signal.{CAPACITY}}}}}");
    let engine = |filters: &str| {
        document(filters, "least-requests")
            .validate()
            .expect("a valid document")
    };
    let both = engine(&format!("healthy, {by_report}, {by_signal}"));
    let report_alone = engine(&format!("healthy, {by_report}"));
    let signal_alone = engine(&format!("healthy, {by_signal}"));
    // The report's rule keeps the first and third rows, the signal's the
    // first and second.
    let mut rows: Vec<_> = (0..4).map(|position| row(position, 0)).collect();
    for (row, resident) in rows.iter_mut().zip([900, 1_100, 900, 1_100]) {
        row.load_sampled_at_micros = NOW;
        row.load = Some(LoadRow {
            resident_tokens: resident,
            resident_counted: true,
            ..LoadRow::default()
        });
    }
    let room = |used: f64| (fresh(100.0), fresh(used));
    let cells = rooms(&[room(50.0), room(50.0), room(150.0), room(150.0)]);
    let request = facts(Some(&cells), 0);
    assert_eq!(kept_at(&both, 1, &request, &rows), [0, 2]);
    assert_eq!(kept_at(&both, 2, &request, &rows), [0, 1]);
    assert_eq!(kept_at(&report_alone, 1, &request, &rows), [0, 2]);
    assert_eq!(kept_at(&signal_alone, 1, &request, &rows), [0, 1]);
    // The rule that names none asks for no cells.
    assert!(!report_alone.needs().signal_cells);
    assert!(both.needs().signal_cells);
}

#[test]
fn a_capacity_past_its_signals_maximum_age_is_no_capacity() {
    // The signal that ends at half a second holds the capacity here.
    let engine = has_room(&format!(
        "used: {{candidate.in_flight: 1}}, capacity: signal.{PENDING}"
    ));
    let rows = [row(0, 1), row(1, 1), row(2, 1)];
    let cells = pending(&[aged(8.0, 500), aged(8.0, 501), fresh(8.0)]);
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [0, 2]);
}

#[test]
fn under_a_capacity_of_zero_no_row_fits_and_the_decision_stands() {
    let engine = has_room(&format!(
        "used: {{candidate.in_flight: 1}}, capacity: signal.{CAPACITY}"
    ));
    // The second row holds nothing: a plain compare would pass it against
    // a capacity of zero.
    let rows = [row(0, 3), row(1, 0), row(2, 2)];
    let cells = rooms(&[(fresh(0.0), SignalCell::NONE); 3]);
    let request = facts(Some(&cells), 0);
    assert_eq!(kept(&engine, &request, &rows), [0u32; 0]);
    let decision = decide(&engine, &request, &rows);
    assert!(
        !matches!(decision, Decision::Shed(_)),
        "a soft filter sheds nothing"
    );
    let pick = decision.pick().expect("the decision stands");
    assert_eq!(
        pick.targets.first().map(|target| target.id),
        Some(WorkerId(2))
    );
}

#[test]
fn the_views_report_age_and_a_signals_age_each_reach_their_own() {
    let engine = serde_yaml::from_str::<PolicySpec>(&format!(
        "apiVersion: {API_VERSION}
view: {{ max_report_age_ms: 100 }}
signals:
  {PENDING}: {{ unit: tokens, max_age_ms: 500 }}
profiles:
  p:
    filter:
      - healthy
      - has-room: {{used: {{load.queued: 1, signal.{PENDING}: 1}}, capacity: 10}}
    score: [least-requests]
    pick: best
"
    ))
    .expect("a policy document")
    .validate()
    .expect("a rule under the view's report age");
    // Each row's report and its signal's value, by their ages in
    // milliseconds.
    let ages = [(50, 300), (150, 0), (0, 501), (0, 500)];
    let mut rows: Vec<_> = (0..4).map(|position| row(position, 0)).collect();
    for (row, (report, _)) in rows.iter_mut().zip(ages) {
        row.load_sampled_at_micros = NOW - report * 1_000;
        row.load = Some(LoadRow {
            queued: 1,
            ..LoadRow::default()
        });
    }
    let cells = ages.map(|(_, signal)| aged(1.0, signal)).to_vec();
    let cells = SignalCells::new(cells, 1);
    // The first row's value is older than the view's limit and within its
    // own: it is read. The second row's report is past the view's limit,
    // and the third row's value is past its own.
    assert_eq!(kept(&engine, &facts(Some(&cells), 0), &rows), [0, 3]);
}

#[test]
fn at_a_pair_each_sides_stage_reads_its_own_sides_cells() {
    let engine = serde_yaml::from_str::<PolicySpec>(&format!(
        "apiVersion: {API_VERSION}
signals:
  {PENDING}: {{ unit: tokens }}
profiles:
  side:
    score:
      - least-cost: {{terms: {{signal.{PENDING}: 1}}, unreported: missing-as-zero}}
    pick: best
bindings:
  - match: {{ phase: [prefill, decode] }}
    plan: pd-pair
    profiles: {{ prefill: side, decode: side }}
"
    ))
    .expect("a policy document")
    .validate()
    .expect("a pair's document");
    let side = |first: u64| -> Vec<Candidate> {
        (0..2)
            .map(|position| Candidate {
                position,
                id: WorkerId(first + u64::from(position)),
                pair_class: 1,
                ..Candidate::default()
            })
            .collect()
    };
    let (prefills, decodes) = (side(31), side(41));
    let one = |values: [f64; 2]| SignalCells::new(values.map(fresh).to_vec(), 1);
    // The prefills' second row is the cheaper and the decodes' first. The
    // request's cells name neither pool's rows.
    let (prefill, decode, request) = (one([900.0, 100.0]), one([100.0, 900.0]), one([5.0, 5.0]));
    let facts = RequestFacts {
        phase: Phase::Prefill,
        now_micros: NOW,
        signal_cells: Some(&request),
        ..RequestFacts::default()
    };
    let cx = Constraints::default();
    let decision = engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            prefill_signal_cells: Some(&prefill),
            decode_signal_cells: Some(&decode),
            ..PairPools::default()
        },
        &cx,
        &cx,
        &mut Scratch::new(),
    );
    let first = decision
        .pick()
        .and_then(|pick| pick.targets.first().copied());
    let first = first.expect("a pair was available");
    assert_eq!(first.id, WorkerId(32));
    assert_eq!(
        first.pair.map(|Pair { decode_id, .. }| decode_id),
        Some(WorkerId(41))
    );
}

#[test]
fn where_every_row_lacks_the_value_the_filter_steps_aside() {
    let engine = has_room(&format!(
        "used: {{signal.{USED}: 1}}, capacity: signal.{CAPACITY}"
    ));
    let rows = [row(0, 3), row(1, 1), row(2, 2)];
    let picks_the_least = |facts: &RequestFacts<'_>| {
        assert_eq!(kept(&engine, facts, &rows), [0u32; 0]);
        let decision = decide(&engine, facts, &rows);
        assert!(
            !matches!(decision, Decision::Shed(_)),
            "a soft filter sheds nothing"
        );
        let pick = decision.pick().expect("the decision stands");
        assert_eq!(
            pick.targets.first().map(|target| target.id),
            Some(WorkerId(2))
        );
    };
    // No row holds a capacity.
    let cells = rooms(&[(SignalCell::NONE, fresh(1.0)); 3]);
    picks_the_least(&facts(Some(&cells), 0));
    // The facts hand no cells at all.
    picks_the_least(&facts(None, 0));
}

#[test]
fn an_undeclared_signal_is_refused_by_the_path_of_its_term() {
    const WHY: &str = "a signal the document declares";
    let score = |terms: &str| {
        format!("{{least-cost: {{terms: {{{terms}}}, unreported: missing-as-zero}}}}")
    };
    let rule = |rule: &str| format!("healthy, {{has-room: {{{rule}}}}}");
    for (error, path, written) in [
        (
            refused("healthy", &score("signal.kv_nothing: 1")),
            "profiles.p.score[0].least-cost.terms",
            "signal.kv_nothing",
        ),
        // The family's name alone names no signal.
        (
            refused("healthy", &score("signal.: 1")),
            "profiles.p.score[0].least-cost.terms",
            "signal.",
        ),
        (
            refused(
                &rule("used: {signal.kv_nothing: 1}, capacity: 10"),
                "least-requests",
            ),
            "profiles.p.filter[1].has-room.used",
            "signal.kv_nothing",
        ),
        (
            refused(
                &rule(&format!(
                    "used: {{signal.{USED}: 1}}, capacity: signal.kv_nothing"
                )),
                "least-requests",
            ),
            "profiles.p.filter[1].has-room.capacity",
            "signal.kv_nothing",
        ),
    ] {
        assert_eq!(error.field_path(), path, "{written}");
        assert!(
            matches!(&error, SpecError::BadField { value, reason, .. } if value == written && *reason == WHY),
            "{written}: {error}"
        );
    }
    // A signal's name written bare is no field of the build.
    let bare = refused("healthy", &score(&format!("{PENDING}: 1")));
    assert!(
        matches!(&bare, SpecError::BadField { value, reason, .. } if value == PENDING && *reason == "a field this build reads"),
        "{bare}"
    );
}

#[test]
fn a_signal_term_needs_the_rule_of_an_unreported_row() {
    let error = refused(
        "healthy",
        &format!("{{least-cost: {{terms: {{signal.{PENDING}: 1}}}}}}"),
    );
    assert_eq!(
        error.field_path(),
        "profiles.p.score[0].least-cost.unreported"
    );
    // A document that prices no signal and reads no report needs none.
    let plain = document("healthy", "{least-cost: {terms: {candidate.in_flight: 1}}}");
    assert!(plain.validate().is_ok());
}

#[test]
fn a_profile_whose_stage_names_a_signal_declares_the_cells_and_no_other_does() {
    let priced = least_cost(&format!(
        "terms: {{signal.{PENDING}: 1}}, unreported: missing-as-zero"
    ));
    assert!(priced.needs().signal_cells);
    assert!(priced.needs_for("m", Phase::Single).signal_cells);
    assert!(!priced.needs().loads, "a signal is no report");
    let bounded = has_room(&format!(
        "used: {{candidate.in_flight: 1}}, capacity: signal.{CAPACITY}"
    ));
    assert!(bounded.needs().signal_cells);
    let summed = has_room(&format!("used: {{signal.{USED}: 1}}, capacity: 10"));
    assert!(summed.needs().signal_cells);
    // The signals are declared and no stage names one.
    let unread = least_cost("terms: {candidate.in_flight: 1}");
    assert!(!unread.needs().signal_cells);
    let plain = has_room("used: {candidate.in_flight: 1}, capacity: 10");
    assert!(!plain.needs().signal_cells);
}

/// A document with no `signals` section, as every document stood before
/// one could declare a signal. It asks for no cells, and cells a host
/// hands all the same turn no answer of its rule or of its scorer.
#[test]
fn a_document_that_declares_no_signals_reads_no_cell_and_decides_as_it_did() {
    let spec: PolicySpec = serde_yaml::from_str(&format!(
        "apiVersion: {API_VERSION}
profiles:
  p:
    filter: [healthy, {{has-room: {{used: {{candidate.in_flight: 1}}, capacity: 12}}}}]
    score: [{{least-cost: {{terms: {{candidate.in_flight: 100}}}}}}]
    pick: best
"
    ))
    .expect("a policy document");
    let engine = spec.validate().expect("a document with no signals");
    assert!(!engine.needs().signal_cells);
    // The rule keeps the rows of at most twelve in flight, and their costs
    // are 500, 0 and 1000.
    let rows = [row(0, 5), row(1, 0), row(2, 15), row(3, 10)];
    let quiet = [fresh(0.0), fresh(1e9), fresh(0.0)];
    let loud = [fresh(1e9), fresh(1.0), fresh(1e9)];
    let handed = cells(&[quiet, loud, quiet, quiet]);
    for request in [facts(None, 0), facts(Some(&handed), 0)] {
        assert_eq!(kept(&engine, &request, &rows), [0, 1, 3]);
        assert_eq!(
            scores_of(&engine, &request, &rows),
            [(2, 1.0), (1, 0.5), (4, 0.0)]
        );
    }
}

/// The two capacity names a worker's report offers, as a document bounds a
/// row by them. A host hands the cells here.
#[test]
fn the_reports_two_capacity_names_bound_a_row() {
    let engine = has_room(&format!(
        "used: {{signal.{USED}: 1, request.prompt_tokens: 1}}, \
         capacity: signal.{CAPACITY}, headroom: 0.5"
    ));
    let rows = [row(0, 0), row(1, 0), row(2, 0)];
    let cells = rooms(&[
        (fresh(131_072.0), fresh(60_000.0)),
        (fresh(131_072.0), fresh(65_000.0)),
        (fresh(262_144.0), fresh(65_000.0)),
    ]);
    let request = facts(Some(&cells), 4_096);
    assert_eq!(kept(&engine, &request, &rows), [0, 2]);
    let decision = decide(&engine, &request, &rows);
    let pick = decision.pick().expect("the rows route");
    assert_ne!(
        pick.targets.first().map(|target| target.id),
        Some(WorkerId(2))
    );
}
