use std::time::Duration;

use smg_worker::worker::SchedulerLoadSnapshot;

use super::*;

fn rank(
    running: i32,
    waiting: i32,
    usage: f64,
    gen: f64,
    prefill: Option<f64>,
) -> SchedulerLoadSnapshot {
    SchedulerLoadSnapshot {
        num_running_reqs: running,
        num_waiting_reqs: waiting,
        token_usage: usage,
        gen_throughput: gen,
        prefill_throughput: prefill,
        max_running_requests: 256,
        num_resident_tokens: Some(1_000),
        ..Default::default()
    }
}

fn report(loads: Vec<SchedulerLoadSnapshot>) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        ..Default::default()
    }
}

#[test]
fn counts_add_over_the_ranks_and_ratios_average() {
    let now = Instant::now();
    let two = report(vec![
        rank(3, 1, 0.2, 100.0, None),
        rank(5, 2, 0.6, 50.0, None),
    ]);
    let row = load_row(&two, Role::Single, now, now, SignalSource::Poll);
    assert_eq!((row.running, row.queued), (8, 3));
    assert_eq!(row.kv_usage_permille, 400);
    assert_eq!(row.active_kv_permille, 400);
    assert_eq!(row.live_rate, 150);
    assert_eq!(row.resident_tokens, 2_000);
    assert!(row.resident_counted);
    assert_eq!(row.max_running, 512);
    assert_eq!(row.source, SignalSource::Poll);
    assert_eq!(row.report_age_ms, 0);
}

/// One rank without the count, a report that lists fewer ranks than it
/// states, or a report with no rank, is a report that does not count
/// resident tokens, not a count of zero: so reads the gateway's own
/// total of the same report.
#[test]
fn resident_tokens_are_counted_only_when_every_rank_sends_them() {
    let now = Instant::now();
    let mut silent = rank(1, 0, 0.1, 40.0, None);
    silent.num_resident_tokens = None;
    let one_silent = report(vec![rank(1, 0, 0.1, 40.0, None), silent]);
    let row = load_row(&one_silent, Role::Single, now, now, SignalSource::Poll);
    assert_eq!((row.resident_tokens, row.resident_counted), (0, false));
    let mut one_missing = report(vec![rank(1, 0, 0.1, 40.0, None)]);
    one_missing.dp_rank_count = 2;
    let row = load_row(&one_missing, Role::Single, now, now, SignalSource::Poll);
    assert_eq!((row.resident_tokens, row.resident_counted), (0, false));
    assert_eq!(one_missing.total_resident_tokens(), None);
    let no_rank = load_row(&report(vec![]), Role::Single, now, now, SignalSource::Poll);
    assert_eq!(
        (no_rank.resident_tokens, no_rank.resident_counted),
        (0, false)
    );
}

/// Ranks that state counts past what one sum of their width holds make
/// a counted row of the most its field states, and under the least
/// count a row that reads none: nothing wraps. The sum is exact in any
/// order of the ranks, so a rank of the other sign gives back what the
/// ranks ahead of it took the sum past.
#[test]
fn resident_tokens_past_what_one_sum_holds_saturate() {
    let now = Instant::now();
    let stating = |count: i64| {
        let mut stated = rank(1, 0, 0.1, 40.0, None);
        stated.num_resident_tokens = Some(count);
        stated
    };
    let row_of = |counts: &[i64]| {
        let stated = report(counts.iter().copied().map(stating).collect());
        let row = load_row(&stated, Role::Single, now, now, SignalSource::Poll);
        (row.resident_tokens, row.resident_counted)
    };
    assert_eq!(row_of(&[i64::MAX, i64::MAX]), (u32::MAX, true));
    assert_eq!(row_of(&[i64::MIN, i64::MIN]), (0, true));
    // Mixed signs: a whole sum that one count of the ranks' width holds,
    // past steps that one such count does not.
    for counts in [
        [i64::MAX, i64::MAX, i64::MIN],
        [i64::MAX, i64::MIN, i64::MAX],
        [i64::MIN, i64::MAX, i64::MAX],
    ] {
        assert_eq!(row_of(&counts), (u32::MAX, true), "{counts:?}");
    }
    assert_eq!(
        row_of(&[i64::MIN, i64::MIN, i64::MAX, i64::MAX, 7]),
        (5, true)
    );
}

#[test]
fn an_idle_rank_adds_no_generation_rate() {
    let now = Instant::now();
    let one_idle = report(vec![
        rank(0, 0, 0.0, 999.0, None),
        rank(1, 0, 0.1, 40.0, None),
    ]);
    let row = load_row(&one_idle, Role::Decode, now, now, SignalSource::Events);
    assert_eq!(row.live_rate, 40);
}

/// Zero is the row's word for no rate, and a scorer drains such a row at
/// its peers' mean rate: a busy worker's rate under one must not read so,
/// and a rate stated as zero still must.
#[test]
fn a_rate_under_one_token_a_second_is_a_rate_of_one() {
    let now = Instant::now();
    let rate = |gen, prefill, role| {
        let busy = report(vec![rank(1, 0, 0.1, gen, Some(prefill))]);
        load_row(&busy, role, now, now, SignalSource::Poll).live_rate
    };
    assert_eq!(rate(0.5, 0.0, Role::Decode), 1);
    assert_eq!(rate(0.0, 0.5, Role::Prefill), 1);
    assert_eq!(rate(0.0, 0.5, Role::Decode), 0, "a stated zero");
}

#[test]
fn a_prefill_drain_needs_every_rank_rate_of_a_complete_report() {
    let now = Instant::now();
    let complete = report(vec![
        rank(1, 0, 0.1, 0.0, Some(2_000.0)),
        rank(1, 0, 0.1, 0.0, Some(3_000.0)),
    ]);
    assert_eq!(
        load_row(&complete, Role::Prefill, now, now, SignalSource::Poll).live_rate,
        5_000
    );
    let one_rate_missing = report(vec![
        rank(1, 0, 0.1, 0.0, Some(2_000.0)),
        rank(1, 0, 0.1, 0.0, None),
    ]);
    assert_eq!(
        load_row(
            &one_rate_missing,
            Role::Prefill,
            now,
            now,
            SignalSource::Poll
        )
        .live_rate,
        0
    );
    let mut partial = report(vec![rank(1, 0, 0.1, 0.0, Some(2_000.0))]);
    partial.dp_rank_count = 2;
    assert_eq!(
        load_row(&partial, Role::Prefill, now, now, SignalSource::Poll).live_rate,
        0
    );
}

#[test]
fn the_age_is_the_decision_instant_less_the_sample_instant() {
    let sampled = Instant::now();
    let now = sampled + Duration::from_millis(1_500);
    let row = load_row(
        &report(vec![rank(0, 0, 0.0, 0.0, None)]),
        Role::Single,
        now,
        sampled,
        SignalSource::Scrape,
    );
    assert_eq!(row.report_age_ms, 1_500);
    assert_eq!(row.source, SignalSource::Scrape);
}

#[test]
fn negative_counts_and_ratios_over_one_are_clamped() {
    let now = Instant::now();
    let odd = report(vec![rank(-4, -1, 1.7, -5.0, None)]);
    let row = load_row(&odd, Role::Single, now, now, SignalSource::Poll);
    assert_eq!((row.running, row.queued), (0, 0));
    assert_eq!(row.kv_usage_permille, 1_000);
    assert_eq!(row.live_rate, 0);
}

/// The declared signals `names`, each under `unit` and its reduction,
/// in name order as the applied document lists them.
fn columns(names: &[(&str, &str, AcrossRanks)]) -> Vec<SignalColumn> {
    let mut names = names.to_vec();
    names.sort_by_key(|&(name, ..)| name);
    let column = |((name, unit, across), at): ((&str, &str, AcrossRanks), u16)| {
        SignalColumn::new(name, unit, at, None).across(across)
    };
    names.into_iter().zip(0..).map(column).collect()
}

fn stated(name: &str, unit: &str, value: f64, age_ms: Option<u32>) -> NamedValue {
    NamedValue {
        name: name.into(),
        unit: unit.into(),
        value,
        age_ms,
    }
}

/// One rank that states `values` by name and no fixed count.
fn stating(values: Vec<NamedValue>) -> SchedulerLoadSnapshot {
    SchedulerLoadSnapshot {
        values,
        ..Default::default()
    }
}

/// What the conversion hands of `report` under `columns`, received at
/// `received` from a record sampled 50 ms before: each value by its
/// name and unit, with its age at the receipt in milliseconds. A
/// stated age counts back from the record's sample.
fn made(
    report: &WorkerLoadResponse,
    columns: &[SignalColumn],
    received: Instant,
) -> Vec<(String, String, f64, u128)> {
    let sampled_at = received - Duration::from_millis(50);
    let mut made = Vec::new();
    signal_values(report, columns, sampled_at, |column, unit, value, at| {
        let age = received.duration_since(at).as_millis();
        made.push((column.name().to_owned(), unit.to_owned(), value, age));
    });
    made
}

fn value(name: &str, unit: &str, value: f64, age: u128) -> (String, String, f64, u128) {
    (name.to_owned(), unit.to_owned(), value, age)
}

#[test]
fn several_ranks_make_one_rows_value_as_the_signal_declares_under_the_oldest_sample() {
    let received = Instant::now() + Duration::from_secs(5);
    let columns = columns(&[
        ("largest", "ms", AcrossRanks::Max),
        ("middle", "ms", AcrossRanks::Mean),
        ("smallest", "ms", AcrossRanks::Min),
        ("total", "ms", AcrossRanks::Sum),
    ]);
    let names = ["largest", "middle", "smallest", "total"];
    // Three ranks state 2, 5 and 11 of each name. The first states no
    // age, so its value is as old as its record: 50 ms. The second's
    // is 300 ms older than the record.
    let rank = |value, age| stating(names.map(|name| stated(name, "ms", value, age)).into());
    let whole = report(vec![
        rank(2.0, None),
        rank(5.0, Some(300)),
        rank(11.0, Some(120)),
    ]);
    assert_eq!(
        made(&whole, &columns, received),
        [
            value("largest", "ms", 11.0, 350),
            value("middle", "ms", 6.0, 350),
            value("smallest", "ms", 2.0, 350),
            value("total", "ms", 18.0, 350),
        ]
    );
    // Scoped to one rank, a report hands that rank's values as they are.
    let scoped = whole.for_dp_rank(None).map(|mut all| {
        all.loads.truncate(1);
        all
    });
    let scoped = scoped.expect("a report with ranks");
    let alone = made(&scoped, &columns, received);
    assert_eq!(alone, names.map(|name| value(name, "ms", 2.0, 50)));
}

/// Each rank's values are dated by the rank's own sample where its
/// entry states one, and by the report's where it states none: the
/// report here is sampled 50 ms before the receipt. A value's own age
/// counts back from its entry's sample. The worker's value stands
/// under the oldest of its ranks' samples. An entry's sample is not
/// held to its report's: one that is newer is read as it is.
#[test]
fn a_ranks_values_are_dated_by_its_own_sample_where_its_entry_states_one() {
    let received = Instant::now() + Duration::from_secs(5);
    let columns = columns(&[("total", "ms", AcrossRanks::Sum)]);
    let own = |value, sampled_ago: Option<u64>, age| {
        let mut rank = stating(vec![stated("total", "ms", value, age)]);
        rank.sampled_at = sampled_ago.map(|ms| received - Duration::from_millis(ms));
        rank
    };
    let total = |sum, age| [value("total", "ms", sum, age)];
    // One rank, sampled 400 ms ago, its value 30 ms older.
    let alone = report(vec![own(2.0, Some(400), Some(30))]);
    assert_eq!(made(&alone, &columns, received), total(2.0, 430));
    // One rank whose own sample is newer than its report's.
    let newer = report(vec![own(2.0, Some(10), None)]);
    assert_eq!(made(&newer, &columns, received), total(2.0, 10));
    // Two ranks: the first sampled 400 ms ago, the second as old as the
    // report. Then the second states the older sample.
    let both = report(vec![own(2.0, Some(400), None), own(5.0, None, None)]);
    assert_eq!(made(&both, &columns, received), total(7.0, 400));
    let turned = report(vec![own(2.0, Some(10), None), own(5.0, Some(900), None)]);
    assert_eq!(made(&turned, &columns, received), total(7.0, 900));
}

#[test]
fn a_row_has_a_value_only_where_every_rank_states_the_name_under_one_unit() {
    let received = Instant::now() + Duration::from_secs(5);
    let columns = columns(&[
        ("mixed", "ms", AcrossRanks::Sum),
        ("partial", "ms", AcrossRanks::Sum),
        ("turned", "ms", AcrossRanks::Sum),
        ("twice", "ms", AcrossRanks::Sum),
        ("whole", "ms", AcrossRanks::Sum),
    ]);
    let first = stating(vec![
        stated("mixed", "ms", 1.0, None),
        stated("partial", "ms", 1.0, None),
        stated("turned", "ms", 1.0, None),
        stated("twice", "ms", 1.0, None),
        // A second statement by one rank: its first one is read.
        stated("twice", "ms", 64.0, None),
        stated("whole", "ms", 1.0, None),
        // A name the document does not declare is not offered.
        stated("other", "ms", 1.0, None),
    ]);
    let second = stating(vec![
        stated("mixed", "seconds", 2.0, None),
        // A rank's first statement rules: under another unit the row
        // has no value, whatever the rank states of the name next.
        stated("turned", "seconds", 9.0, None),
        stated("turned", "ms", 2.0, None),
        stated("twice", "ms", 2.0, None),
        stated("whole", "ms", 2.0, None),
    ]);
    assert_eq!(
        made(&report(vec![first.clone(), second]), &columns, received),
        [value("twice", "ms", 3.0, 50), value("whole", "ms", 3.0, 50)]
    );
    // The ranks in the other order: the names the first one leaves out
    // are none of the row's, and the one both state is.
    let late = stating(vec![stated("partial", "ms", 2.0, None)]);
    assert_eq!(
        made(&report(vec![late, first]), &columns, received),
        [value("partial", "ms", 3.0, 50)]
    );
    // No declared signal and no rank hand nothing.
    let one = report(vec![stating(vec![stated("whole", "ms", 1.0, None)])]);
    assert_eq!(made(&one, &[], received), []);
    assert_eq!(made(&report(vec![]), &columns, received), []);
}

#[test]
fn a_report_of_one_rank_hands_each_declared_name_at_its_first_statement() {
    let received = Instant::now() + Duration::from_secs(5);
    let columns = columns(&[
        ("depth", "items", AcrossRanks::Mean),
        ("wait", "ms", AcrossRanks::Max),
    ]);
    let alone = report(vec![stating(vec![
        stated("wait", "ms", 7.0, Some(20)),
        stated("other", "ms", 1.0, None),
        stated("depth", "items", 3.0, None),
        stated("wait", "ms", 64.0, None),
    ])]);
    // In the report's order: the one rank's values are handed as they
    // are read, each under its own age or its record's.
    assert_eq!(
        made(&alone, &columns, received),
        [
            value("wait", "ms", 7.0, 70),
            value("depth", "items", 3.0, 50)
        ]
    );
    // A value that is no number is handed on: the cell's write refuses it.
    let odd = report(vec![stating(vec![stated("wait", "ms", f64::NAN, None)])]);
    let made = made(&odd, &columns, received);
    assert!(matches!(made.as_slice(), [(_, _, value, _)] if value.is_nan()));
}

#[test]
fn a_ranks_value_that_is_no_number_makes_the_rows_none_under_every_reduction() {
    let received = Instant::now() + Duration::from_secs(5);
    for across in [
        AcrossRanks::Sum,
        AcrossRanks::Max,
        AcrossRanks::Min,
        AcrossRanks::Mean,
    ] {
        let columns = columns(&[("depth", "items", across)]);
        for odd in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let rank = |value| stating(vec![stated("depth", "items", value, None)]);
            let made = made(
                &report(vec![rank(4.0), rank(odd), rank(9.0)]),
                &columns,
                received,
            );
            let [(_, _, value, _)] = made.as_slice() else {
                panic!("one value, {made:?}");
            };
            assert!(!value.is_finite(), "{across:?} of {odd}: {value}");
        }
    }
}

#[test]
fn the_fixed_token_counts_are_offered_under_their_names() {
    let received = Instant::now() + Duration::from_secs(5);
    let columns = columns(&[("kv_tokens_used", "tokens", AcrossRanks::Sum)]);
    let rank = |used| SchedulerLoadSnapshot {
        num_used_tokens: used,
        max_total_num_tokens: 8_192,
        ..Default::default()
    };
    assert_eq!(
        made(&report(vec![rank(1_000), rank(24)]), &columns, received),
        [value("kv_tokens_used", "tokens", 1_024.0, 50)]
    );
    // A rank that drains states zero: the row's sum still stands, and a
    // drained rank alone reads zero where its last count stood.
    assert_eq!(
        made(&report(vec![rank(1_000), rank(0)]), &columns, received),
        [value("kv_tokens_used", "tokens", 1_000.0, 50)]
    );
    assert_eq!(
        made(&report(vec![rank(0)]), &columns, received),
        [value("kv_tokens_used", "tokens", 0.0, 50)]
    );
}
