//! `has-room` by a rule the document writes: what a row holds against its
//! capacity.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use smg_policy_controller::{
    spec::{Capacity, FilterSpec, FilterStage, HasRoomParams},
    Candidate, Constraints, Decision, Engine, Health, Lease, Lifecycle, LoadRow, PolicySpec,
    PoolFacts, RequestFacts, Scratch, Side, SpecError, Target, WorkerId, API_VERSION,
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
    // A rule that does not name the size reads the same rows for any request.
    let unsized_rule = self::policy(RESIDENT);
    assert_eq!(kept(&unsized_rule, 0, &request(301), &rows), [0, 1]);
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
/// the sends a report ends. It is soft and row-wise, and it refuses by
/// nothing a late report writes.
#[test]
fn the_rule_needs_reports_where_a_term_reads_them_or_the_sends() {
    for (used, loads) in [
        (r#"{"load.resident_tokens":1}"#, true),
        (r#"{"sent.requests":1}"#, true),
        (
            r#"{"candidate.in_flight":1,"request.prompt_tokens":1}"#,
            false,
        ),
    ] {
        let policy = policy(&format!(
            r#"{{"has-room":{{"used":{used},"capacity":10}}}}"#
        ));
        let filter = &policy.profiles()[0].filters[0];
        assert_eq!(filter.needs().loads, loads, "{used}");
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
    for (rule, field, written, why) in [
        (
            r#"{"used":{},"capacity":10}"#,
            "used",
            "{}",
            "at least one field with its coefficient",
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
        // No field of this build holds a capacity.
        (
            r#"{"used":{"load.running":1},"capacity":"load.capacity"}"#,
            "capacity",
            "load.capacity",
            "a field this build reads",
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

/// An empty map is no rule: only the bare name is the host's bit.
#[test]
fn a_rule_that_leaves_a_part_out_is_refused_at_the_parse() {
    for (rule, message) in [
        ("{}", "missing field `used`"),
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
            "a number, or the name of a field that holds one",
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
