//! `has-room` by a rule the document writes: what a row holds against its
//! capacity.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use smg_policy_controller::{
    document::{
        Capacity, FilterSpec, FilterStage, HasRoomParams, PolicySpec, SpecError, API_VERSION,
    },
    engine::{
        Candidate, Constraints, Decision, Engine, Health, LoadRow, PoolFacts, RequestFacts,
        Scratch, Target, WorkerId,
    },
    stages::Lifecycle,
    tracking::{Lease, Side},
};

const NOW: u64 = 1_000_000;
/// The rule most cases read: a row holds its resident tokens, and can hold
/// a thousand.
const RESIDENT: &str = r#"{"has-room":{"used":{"load.resident_tokens":1},"capacity":1000}}"#;

/// A document whose one profile holds `filters` and draws at random.
fn document(filters: &str) -> PolicySpec {
    serde_json::from_str(&format!(
        r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{{"filter":[{filters}],"pick":"random"}}}}}}"#
    ))
    .expect("a policy document")
}

fn policy(filters: &str) -> Engine {
    document(filters).validate().expect("a valid policy")
}

/// A healthy row whose report, sampled at the request's instant, counts
/// `resident` tokens.
fn row(position: u32, resident: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(u64::from(position) + 1),
        health: Health {
            healthy: true,
            ..Health::default()
        },
        load_sampled_at_micros: NOW,
        load: Some(LoadRow {
            resident_tokens: resident,
            resident_counted: true,
            ..LoadRow::default()
        }),
        ..Candidate::default()
    }
}

fn request(prompt_tokens: u32) -> RequestFacts<'static> {
    RequestFacts {
        now_micros: NOW,
        prompt_tokens,
        ..RequestFacts::default()
    }
}

/// A request of `prompt_tokens` whose answer may hold `limit` tokens.
fn with_limit(prompt_tokens: u32, limit: u32) -> RequestFacts<'static> {
    RequestFacts {
        max_output_tokens: limit,
        ..request(prompt_tokens)
    }
}

/// The rows the profile's filter at `at` keeps, by their positions. The
/// filter's pass and its word on each row alone must agree.
fn kept(policy: &Engine, at: usize, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Vec<u32> {
    let filter = &policy.profiles()[0].filters[at];
    let cx = Constraints::default();
    let mut kept: Vec<u32> = rows.iter().map(|row| row.position).collect();
    filter.retain(facts, rows, &cx, &mut kept);
    let alone: Vec<u32> = rows
        .iter()
        .filter(|row| filter.keep(facts, row, &cx))
        .map(|row| row.position)
        .collect();
    assert_eq!(kept, alone, "the pass and the row's own word");
    kept
}

/// The worker a decision over `rows` picks.
fn picked(policy: &Engine, facts: &RequestFacts<'_>, rows: &[Candidate]) -> Option<WorkerId> {
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    let decision = policy.decide(
        facts,
        &pool,
        rows,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        !matches!(decision, Decision::Shed(_)),
        "a soft filter sheds nothing"
    );
    let pick = decision.pick()?;
    pick.targets.first().map(|target| target.id)
}

#[test]
fn a_row_at_or_under_the_bound_is_kept_and_a_row_over_it_is_dropped() {
    let policy = policy(RESIDENT);
    let rows = [row(0, 999), row(1, 1_000), row(2, 1_001)];
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0, 1]);
    // The filter answers the name of the host's bit.
    assert_eq!(policy.profiles()[0].filters[0].name(), "has-room");
}

/// The pass writes the rows that fit over the list's front: rows that fit
/// behind a dropped row keep the list's order, and the list ends at the
/// last row that fits.
#[test]
fn a_pass_keeps_the_rows_that_fit_in_the_lists_order() {
    let policy = policy(RESIDENT);
    let rows = [
        row(0, 1_001),
        row(1, 999),
        row(2, 2_000),
        row(3, 1_000),
        row(4, 1),
        row(5, 1_001),
    ];
    assert_eq!(kept(&policy, 0, &request(0), &rows), [1, 3, 4]);
}

#[test]
fn a_headroom_keeps_its_share_of_the_capacity_free() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1},"capacity":200000,"headroom":0.05}}"#,
    );
    let rows = [row(0, 189_999), row(1, 190_000), row(2, 190_001)];
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0, 1]);
}

#[test]
fn the_requests_size_counts_where_the_rule_names_it() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"request.prompt_tokens":1},"capacity":1000}}"#,
    );
    let rows = [row(0, 700), row(1, 900)];
    assert_eq!(kept(&policy, 0, &request(100), &rows), [0, 1]);
    assert_eq!(kept(&policy, 0, &request(101), &rows), [0]);
    assert_eq!(kept(&policy, 0, &request(301), &rows), [0u32; 0]);
    // A request of text alone counts at its text's bytes over four: 404
    // bytes are the 101 tokens that leave the second row no room.
    let worded = |prompt_bytes| RequestFacts {
        prompt_bytes,
        ..request(0)
    };
    assert_eq!(kept(&policy, 0, &worded(400), &rows), [0, 1]);
    assert_eq!(kept(&policy, 0, &worded(404), &rows), [0]);
    // A rule that names no value of the request reads the same rows for any
    // request.
    let unsized_rule = self::policy(RESIDENT);
    assert_eq!(kept(&unsized_rule, 0, &request(301), &rows), [0, 1]);
}

/// A rule that names the request's output limit reserves room for the
/// answer: the limit times its coefficient counts on every row, beside the
/// request's size. A request that states no limit reserves nothing.
#[test]
fn the_requests_output_limit_reserves_room_where_the_rule_names_it() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"request.max_output_tokens":0.5,"request.prompt_tokens":1},"capacity":1000}}"#,
    );
    let rows = [row(0, 700), row(1, 800)];
    // A prompt of 100 and half of an answer of 200 put the fuller row on
    // the bound.
    assert_eq!(kept(&policy, 0, &with_limit(100, 200), &rows), [0, 1]);
    // Two tokens more of answer are one token over it.
    assert_eq!(kept(&policy, 0, &with_limit(100, 202), &rows), [0]);
    // With no limit stated the size alone counts: a prompt of 200 puts the
    // fuller row on the same bound.
    assert_eq!(kept(&policy, 0, &with_limit(200, 0), &rows), [0, 1]);
}

/// The limit's coefficient counts with its sign, as every coefficient of
/// the rule does: under one below zero a larger limit leaves a row more
/// room.
#[test]
fn the_output_limits_coefficient_counts_with_its_sign() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"request.max_output_tokens":-1},"capacity":1000}}"#,
    );
    let rows = [row(0, 1_100)];
    assert_eq!(kept(&policy, 0, &with_limit(0, 100), &rows), [0]);
    assert_eq!(kept(&policy, 0, &with_limit(0, 99), &rows), [0u32; 0]);
}

/// A rule that names no output limit reserves nothing for an answer: a row
/// on the bound stands, and a row a token over it goes, whatever limit the
/// request states.
#[test]
fn a_rule_that_names_no_output_limit_keeps_its_rows_under_any_limit() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"request.prompt_tokens":1},"capacity":1000}}"#,
    );
    // With a prompt of 100 the second row sits on the bound, and the third
    // a token over it.
    let rows = [row(0, 899), row(1, 900), row(2, 901)];
    for limit in [0, 1, 4_096, u32::MAX] {
        let asks = with_limit(100, limit);
        assert_eq!(kept(&policy, 0, &asks, &rows), [0, 1], "limit {limit}");
    }
}

/// Under the view's report age limit the rule reads its reports through
/// the limit, and reserves the answer's room as it does with none: a row
/// whose report is past the limit has no room, and of the two fresh rows
/// the one the answer puts over the bound goes.
#[test]
fn the_output_limit_reserves_room_under_the_views_report_age_limit() {
    const LIMIT_MS: u64 = 27;
    let mut spec = document(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"request.max_output_tokens":2},"capacity":1000}}"#,
    );
    spec.view.max_report_age_ms = Some(LIMIT_MS);
    let limited = spec.validate().expect("a valid policy");
    let stale = Candidate {
        load_sampled_at_micros: NOW - LIMIT_MS * 1_000 - 1,
        ..row(2, 10)
    };
    let rows = [row(0, 600), row(1, 601), stale];
    // Twice an answer of 199 leaves both fresh rows under the bound.
    assert_eq!(kept(&limited, 0, &with_limit(0, 199), &rows), [0, 1]);
    // Twice an answer of 200 puts the first row on the bound and the
    // second a token over it.
    assert_eq!(kept(&limited, 0, &with_limit(0, 200), &rows), [0]);
}

#[test]
fn where_no_row_has_room_the_filter_steps_aside_and_every_row_stands() {
    let policy = policy(&format!(r#""healthy",{RESIDENT}"#));
    let mut rows = [row(0, 1_001), row(1, 2_000), row(2, 3_000)];
    assert_eq!(kept(&policy, 1, &request(0), &rows), [0u32; 0]);
    // Each row in turn is the one healthy row: the rule would drop it, and
    // it is the pick.
    for healthy in 0..rows.len() {
        for (at, row) in rows.iter_mut().enumerate() {
            row.health.healthy = at == healthy;
        }
        assert_eq!(
            picked(&policy, &request(0), &rows),
            Some(rows[healthy].id),
            "row {healthy} stands"
        );
    }
}

#[test]
fn a_row_that_lacks_a_named_value_has_no_room() {
    let policy = policy(RESIDENT);
    let reported = row(0, 10);
    let unreported = Candidate {
        load: None,
        ..row(1, 0)
    };
    let mut uncounted = row(2, 10);
    if let Some(load) = uncounted.load.as_mut() {
        load.resident_counted = false;
    }
    let rows = [reported, unreported, uncounted];
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0]);
    // A value the rule does not name is never missed: a rule of the running
    // count alone keeps the row whose resident tokens are not counted.
    let running = self::policy(r#"{"has-room":{"used":{"load.running":1},"capacity":2}}"#);
    assert_eq!(kept(&running, 0, &request(0), &rows), [0, 2]);
    // Where no row reports, the filter keeps nobody and steps aside.
    let silent = [
        Candidate {
            position: 0,
            id: WorkerId(7),
            ..unreported
        },
        unreported,
    ];
    assert_eq!(kept(&policy, 0, &request(0), &silent), [0u32; 0]);
    assert!(picked(&policy, &request(0), &silent).is_some());
    // A rule that reads no report keeps a row with none.
    let counted = self::policy(r#"{"has-room":{"used":{"candidate.in_flight":1},"capacity":2}}"#);
    let busy = |in_flight| Candidate {
        in_flight,
        ..unreported
    };
    assert!(counted.profiles()[0].filters[0].keep(&request(0), &busy(2), &Constraints::default()));
    assert!(!counted.profiles()[0].filters[0].keep(&request(0), &busy(3), &Constraints::default()));
}

/// The rule asks for load reports where a term reads one, or where it reads
/// the sends a report ends. It asks for the request's size where it names
/// it, and where a term prices the sends by their tokens: a send of text
/// alone is counted by its text's bytes, which a host that can send a
/// request on without reading its prompt states only where a stage asks.
/// It asks for nothing where it names the output limit, which a request
/// states. It is soft and row-wise, and it refuses by nothing a late report
/// writes.
#[test]
fn the_rule_needs_reports_where_a_term_reads_them_or_the_sends() {
    for (used, loads) in [
        (r#"{"load.resident_tokens":1}"#, true),
        (r#"{"sent.requests":1}"#, true),
        (r#"{"sent.tokens":1}"#, true),
        (r#"{"load.resident_tokens":1,"sent.tokens":1}"#, true),
        (
            r#"{"candidate.in_flight":1,"request.prompt_tokens":1}"#,
            false,
        ),
        (
            r#"{"candidate.in_flight":1,"request.max_output_tokens":1}"#,
            false,
        ),
    ] {
        let policy = policy(&format!(
            r#"{{"has-room":{{"used":{used},"capacity":10}}}}"#
        ));
        let filter = &policy.profiles()[0].filters[0];
        assert_eq!(filter.needs().loads, loads, "{used}");
        let sized = used.contains("request.prompt_tokens") || used.contains("sent.tokens");
        assert_eq!(filter.needs().prompt_size, sized, "{used}");
        assert_eq!(policy.needs().prompt_size, sized, "{used}, the document");
        assert!(filter.soft() && filter.row_wise() && !filter.refuses_by_reports());
    }
}

#[test]
fn a_report_past_the_views_limit_reads_as_no_report() {
    const LIMIT_MS: u64 = 27;
    const EDGE: u64 = NOW - LIMIT_MS * 1_000;
    let sampled = |position, sample| Candidate {
        load_sampled_at_micros: sample,
        ..row(position, 10)
    };
    let rows = [sampled(0, EDGE), sampled(1, EDGE - 1)];
    let mut spec = document(RESIDENT);
    spec.view.max_report_age_ms = Some(LIMIT_MS);
    let limited = spec.validate().expect("a valid policy");
    assert_eq!(kept(&limited, 0, &request(0), &rows), [0]);
    // A document with no limit reads a report of any age.
    assert_eq!(kept(&policy(RESIDENT), 0, &request(0), &rows), [0, 1]);
}

#[test]
fn a_count_that_holds_no_send_leaves_the_other_terms_of_the_rule() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"sent.tokens":1},"capacity":1000}}"#,
    );
    let rows = [row(0, 999), row(1, 1_001)];
    // No row holds an entry: the pass reads no count, and the report's
    // term still drops the second row.
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0]);
    // A send behind the first row's sample enters the count, and the pass
    // reads it.
    let lease = Lease {
        issued_micros: NOW + 1,
        tokens: 2,
        ..Lease::default()
    };
    let target = Target {
        id: rows[0].id,
        ..Target::default()
    };
    policy.on_dispatched(&lease, &target, Side::Single, 0);
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0u32; 0]);
}

#[test]
fn a_send_takes_room_until_a_report_or_the_profiles_limit_ends_it() {
    const START: u64 = 91_300;
    const DELAY: u32 = 47;
    const TOKENS: u32 = 317;
    let document: PolicySpec = serde_json::from_str(&format!(
        r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{{"sent_ttl_ms":17,
            "filter":[{{"has-room":{{"used":{{"sent.tokens":1}},"capacity":500}}}}],"pick":"random"}}}}}}"#
    ))
    .expect("a policy document");
    let policy = document
        .validate()
        .expect("the rule reads the sends the limit ages");
    let send = |row: &Candidate| {
        let lease = Lease {
            issued_micros: START,
            tokens: TOKENS,
            ..Lease::default()
        };
        let target = Target {
            id: row.id,
            dp_rank: row.dp_rank,
            ..Target::default()
        };
        policy.on_dispatched(&lease, &target, Side::Single, DELAY);
    };
    let at = START + u64::from(DELAY);
    let facts = |now_micros| RequestFacts {
        now_micros,
        ..RequestFacts::default()
    };
    // The row's report was sampled ahead of both sends.
    let mut a = row(0, 0);
    a.load_sampled_at_micros = at - 1;
    let b = Candidate {
        load_sampled_at_micros: at - 1,
        ..row(1, 0)
    };
    send(&a);
    assert_eq!(kept(&policy, 0, &facts(at + 400), &[a, b]), [0, 1]);
    send(&a);
    assert_eq!(
        kept(&policy, 0, &facts(at + 400), &[a, b]),
        [1],
        "two sends of 317 tokens are over 500"
    );
    // The profile's limit ends both sends.
    assert_eq!(kept(&policy, 0, &facts(at + 17_000), &[a, b]), [0, 1]);
    // So does a report sampled at the sends' instant.
    a.load_sampled_at_micros = at;
    assert_eq!(kept(&policy, 0, &facts(at + 400), &[a, b]), [0, 1]);
}

#[test]
fn a_send_takes_room_past_a_report_sampled_sooner_than_the_profiles_delay() {
    const SENT_AT: u64 = 91_347;
    let document: PolicySpec = serde_json::from_str(&format!(
        r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{{"sent_report_delay_ms":5,
            "filter":[{{"has-room":{{"used":{{"sent.tokens":1}},"capacity":300}}}}],"pick":"random"}}}}}}"#
    ))
    .expect("a policy document");
    let policy = document
        .validate()
        .expect("the rule reads the sends the delay holds");
    let mut a = row(0, 0);
    let b = row(1, 0);
    let lease = Lease {
        issued_micros: SENT_AT,
        tokens: 317,
        ..Lease::default()
    };
    let target = Target {
        id: a.id,
        dp_rank: a.dp_rank,
        ..Target::default()
    };
    policy.on_dispatched(&lease, &target, Side::Single, 0);
    let facts = RequestFacts {
        now_micros: SENT_AT + 9_000,
        ..RequestFacts::default()
    };
    // Three milliseconds behind the send: its 317 tokens are over 300.
    a.load_sampled_at_micros = SENT_AT + 3_000;
    assert_eq!(kept(&policy, 0, &facts, &[a, b]), [1]);
    // Five behind it: the report ends the send.
    a.load_sampled_at_micros = SENT_AT + 5_000;
    assert_eq!(kept(&policy, 0, &facts, &[a, b]), [0, 1]);
}

#[test]
fn a_sends_request_and_its_tokens_each_take_room_by_their_own_coefficient() {
    let policy =
        policy(r#"{"has-room":{"used":{"sent.requests":100,"sent.tokens":1},"capacity":250}}"#);
    let rows = [row(0, 0), row(1, 0)];
    let send = || {
        let lease = Lease {
            issued_micros: NOW + 1,
            tokens: 50,
            ..Lease::default()
        };
        let target = Target {
            id: rows[0].id,
            ..Target::default()
        };
        policy.on_dispatched(&lease, &target, Side::Single, 0);
    };
    // One send of 50 tokens holds 100 and 50 of the 250. Under the
    // request's coefficient its tokens alone would hold 5,000.
    send();
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0, 1]);
    // Two hold 200 and 100. Under the tokens' coefficient the two requests
    // would hold 2, and the row would stay.
    send();
    assert_eq!(kept(&policy, 0, &request(0), &rows), [1]);
}

/// A send of text alone takes room by its bytes over four, the size the
/// request's own term reads of such a request. The rule is the guide's:
/// what the row holds, what was sent to it, and what the request brings.
#[test]
fn a_send_of_text_alone_takes_room_by_its_bytes_over_four() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"sent.tokens":1,
            "request.prompt_tokens":1},"capacity":1000}}"#,
    );
    let rows = [row(0, 800), row(1, 0)];
    // A request of text alone states no token count: its 404 bytes read
    // as 101, and 800 resident with them are under the thousand.
    let text = RequestFacts {
        prompt_bytes: 404,
        ..request(0)
    };
    assert_eq!(kept(&policy, 0, &text, &rows), [0, 1]);
    // The same text is sent to the first row, behind its report's sample.
    let lease = Lease {
        issued_micros: NOW + 1,
        bytes: 404,
        ..Lease::default()
    };
    let target = Target {
        id: rows[0].id,
        ..Target::default()
    };
    policy.on_dispatched(&lease, &target, Side::Single, 0);
    // 800 resident, the send's 101 and the request's 101 are over it.
    assert_eq!(kept(&policy, 0, &text, &rows), [1]);
}

/// A rule that prices the sends by their tokens and names no size of the
/// request still asks its host for each request's size: the send of a
/// request of text alone is counted by its text's bytes, and a host that
/// can send a request on without reading its prompt states them only where
/// a stage of the document asks. Under such a rule a send of text takes
/// room, and a rule that counts the sends alone asks for no size.
#[test]
fn a_rule_that_prices_the_sends_by_their_tokens_asks_for_each_requests_size() {
    let policy = policy(
        r#"{"has-room":{"used":{"load.resident_tokens":1,"sent.tokens":1},"capacity":1000}}"#,
    );
    assert!(policy.needs().prompt_size, "the document asks for the size");
    let counted = self::policy(r#"{"has-room":{"used":{"sent.requests":1},"capacity":10}}"#);
    assert!(
        !counted.needs().prompt_size,
        "a count of sends reads no size"
    );
    let rows = [row(0, 900), row(1, 0)];
    assert_eq!(kept(&policy, 0, &request(0), &rows), [0, 1]);
    // 404 bytes of text are sent to the first row: 900 resident and the
    // send's 101 are over the thousand.
    let lease = Lease {
        issued_micros: NOW + 1,
        bytes: 404,
        ..Lease::default()
    };
    let target = Target {
        id: rows[0].id,
        ..Target::default()
    };
    policy.on_dispatched(&lease, &target, Side::Single, 0);
    assert_eq!(kept(&policy, 0, &request(0), &rows), [1]);
}

#[test]
fn the_rule_ahead_of_a_hard_filter_decides_as_the_rule_behind_it() {
    let ahead = policy(&format!(r#"{RESIDENT},"healthy""#));
    let behind = policy(&format!(r#""healthy",{RESIDENT}"#));
    let sick = |position, resident| {
        let mut row = row(position, resident);
        row.health.healthy = false;
        row
    };
    // The one healthy row with room is the pick, whatever the order.
    let rows = [sick(0, 10), row(1, 1_001), row(2, 10)];
    for policy in [&ahead, &behind] {
        assert_eq!(picked(policy, &request(0), &rows), Some(rows[2].id));
    }
    // With no healthy row under the bound the rule steps aside, behind the
    // hard filter in both orders: the unhealthy row with room is not the
    // pick.
    let rows = [sick(0, 10), row(1, 1_001)];
    for policy in [&ahead, &behind] {
        assert_eq!(picked(policy, &request(0), &rows), Some(rows[1].id));
    }
}

#[test]
fn each_part_of_the_rule_is_refused_at_its_path() {
    let refused = |rule: &str| {
        document(&format!(r#""healthy",{{"has-room":{rule}}}"#))
            .validate()
            .expect_err("a refused rule")
    };
    const AT: &str = "profiles.p.filter[1].has-room";
    const COEFFICIENT: &str = "a finite number other than zero, of magnitude at most 1e15";
    const OF_A_WORKER: &str =
        "at least one field or signal of the worker beside the request's own values";
    for (rule, field, written, why) in [
        (
            r#"{"used":{},"capacity":10}"#,
            "used",
            "{}",
            "at least one field with its coefficient",
        ),
        // Under a capacity that is a number, the request's size alone
        // asks nothing of a worker.
        (
            r#"{"used":{"request.prompt_tokens":1},"capacity":10}"#,
            "used",
            "request.prompt_tokens",
            OF_A_WORKER,
        ),
        // The limit's coefficient is a coefficient of the map: zero is
        // refused at its own path.
        (
            r#"{"used":{"load.running":1,"request.max_output_tokens":0},"capacity":10}"#,
            "used.request.max_output_tokens",
            "0",
            COEFFICIENT,
        ),
        // The request's output limit alone asks nothing of a worker
        // either, at any coefficient, nor do the two together. The refusal
        // names the first of them in name order, whichever the document
        // writes first.
        (
            r#"{"used":{"request.max_output_tokens":-1},"capacity":10}"#,
            "used",
            "request.max_output_tokens",
            OF_A_WORKER,
        ),
        (
            r#"{"used":{"request.prompt_tokens":1,"request.max_output_tokens":1},"capacity":10}"#,
            "used",
            "request.max_output_tokens",
            OF_A_WORKER,
        ),
        // The capacity is settled first: under a name that is no signal's
        // the refusal is the capacity's.
        (
            r#"{"used":{"request.prompt_tokens":1},"capacity":"load.capacity"}"#,
            "capacity",
            "load.capacity",
            "a declared signal, written signal.<name>",
        ),
        (
            r#"{"used":{"load.nothing":1},"capacity":10}"#,
            "used",
            "load.nothing",
            "a field this build reads",
        ),
        (
            r#"{"used":{"load.running":0},"capacity":10}"#,
            "used.load.running",
            "0",
            COEFFICIENT,
        ),
        (
            r#"{"used":{"load.running":1e16},"capacity":10}"#,
            "used.load.running",
            "10000000000000000",
            COEFFICIENT,
        ),
        (
            r#"{"used":{"load.running":1},"capacity":0}"#,
            "capacity",
            "0",
            "a finite number above zero",
        ),
        (
            r#"{"used":{"load.running":1},"capacity":-3}"#,
            "capacity",
            "-3",
            "a finite number above zero",
        ),
        // No field of this build holds a capacity: a name is a signal's.
        (
            r#"{"used":{"load.running":1},"capacity":"load.capacity"}"#,
            "capacity",
            "load.capacity",
            "a declared signal, written signal.<name>",
        ),
        (
            r#"{"used":{"load.running":1},"capacity":10,"headroom":1}"#,
            "headroom",
            "1",
            "a number from 0 to under 1",
        ),
        (
            r#"{"used":{"load.running":1},"capacity":10,"headroom":-0.1}"#,
            "headroom",
            "-0.1",
            "a number from 0 to under 1",
        ),
    ] {
        let error = refused(rule);
        let path = format!("{AT}.{field}");
        assert_eq!(error.field_path(), path, "{rule}");
        assert!(
            matches!(&error, SpecError::BadField { value, reason, .. } if value == written && *reason == why),
            "{rule}: {error}"
        );
    }
}

/// An empty map is no rule: only the bare name is the host's bit. A null
/// under the name is the slip of a bare name written with a colon, and its
/// author is told both forms at the rule's path. YAML's own typed read
/// would take that null for an empty map: the entry catches it first.
#[test]
fn a_rule_that_leaves_a_part_out_is_refused_at_the_parse() {
    let slip = "expected a rule with `used` and `capacity`, or the bare name `has-room`";
    let yaml = format!(
        "apiVersion: {API_VERSION}\nprofiles:\n  p:\n    filter:\n      - has-room:\n    pick: random\n"
    );
    let slipped = serde_yaml::from_str::<PolicySpec>(&yaml)
        .expect_err("an empty rule")
        .to_string();
    assert!(
        slipped.starts_with("profiles.p.filter[0].has-room: ") && slipped.contains(slip),
        "{slipped}"
    );
    for (rule, message) in [
        ("null", slip),
        ("{}", "missing field `used`"),
        (r#"{"capacity":10}"#, "missing field `used`"),
        (r#"{"used":{"load.running":1}}"#, "missing field `capacity`"),
        (
            r#"{"used":{"load.running":1},"capacity":10,"room":2}"#,
            "unknown field `room`",
        ),
        (
            r#"{"used":{"load.running":1,"load.running":2},"capacity":10}"#,
            "the key `load.running` is written twice",
        ),
        (
            r#"{"used":{"load.running":1},"capacity":true}"#,
            "a number, or a declared signal written signal.<name>",
        ),
    ] {
        let error = serde_json::from_str::<PolicySpec>(&format!(
            r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{{"filter":[{{"has-room":{rule}}}],"pick":"random"}}}}}}"#
        ))
        .expect_err("a refused rule")
        .to_string();
        assert!(error.contains(message), "{rule}: {error}");
    }
}

/// The rule's read is its type's own, whole: its parts written as a list,
/// in the type's order, read as the map does.
#[test]
fn a_rule_written_as_a_list_reads_as_its_map() {
    let listed = document(r#"{"has-room":[{"load.running":1},10,0.25]}"#);
    let mapped =
        document(r#"{"has-room":{"used":{"load.running":1},"capacity":10,"headroom":0.25}}"#);
    assert_eq!(listed, mapped);
    listed.validate().expect("the listed rule builds");
}

#[test]
fn a_document_with_the_rule_round_trips_and_its_digest_ignores_the_key_order() {
    let rule = r#"{"has-room":{"headroom":0.05,"capacity":200000,"used":{"sent.tokens":1,"load.resident_tokens":1,"request.prompt_tokens":1}}}"#;
    let spec = document(&format!(r#""healthy",{rule}"#));
    assert_eq!(
        spec.profiles["p"].filter[1],
        FilterSpec::Params(FilterStage::HasRoom(HasRoomParams {
            used: [
                ("load.resident_tokens".to_string(), 1.0),
                ("request.prompt_tokens".to_string(), 1.0),
                ("sent.tokens".to_string(), 1.0),
            ]
            .into(),
            capacity: Capacity::Fixed(200_000.0),
            headroom: 0.05,
        }))
    );
    let written = serde_json::to_string(&spec).expect("serializes");
    assert!(
        written.contains(
            r#""filter":["healthy",{"has-room":{"used":{"load.resident_tokens":1.0,"request.prompt_tokens":1.0,"sent.tokens":1.0},"capacity":200000.0,"headroom":0.05}}]"#
        ),
        "{written}"
    );
    let read: PolicySpec = serde_json::from_str(&written).expect("reads its own form");
    assert_eq!(read, spec);
    let yaml = serde_yaml::to_string(&spec).expect("yaml");
    let from_yaml: PolicySpec = serde_yaml::from_str(&yaml).expect("yaml round trip");
    assert_eq!(from_yaml, spec);
    // Another order of the same keys is the same document.
    let reordered = document(
        r#""healthy",{"has-room":{"used":{"request.prompt_tokens":1,"load.resident_tokens":1,"sent.tokens":1},"capacity":200000,"headroom":0.05}}"#,
    );
    assert_eq!(
        reordered.digest().expect("a digest"),
        spec.digest().expect("a digest")
    );
    // The bare name stays the string it was.
    let bare = document(r#""healthy","has-room""#);
    assert!(serde_json::to_string(&bare)
        .expect("serializes")
        .contains(r#""filter":["healthy","has-room"]"#));
    spec.validate().expect("the rule validates");
}
