//! A worker's load row from its report, for a worker list's records.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::worker::{NamedValue, WorkerLoadResponse};

use crate::{
    document::spec::AcrossRanks,
    engine::view::{permille, LoadRow, Role, SignalSource},
    load::signals::SignalColumn,
};

/// One worker's load row from its report, in one pass over the ranks: the
/// request and token counts summed, the KV ratios averaged, the drain rate
/// as the policies read it today. A prefill's drain is its measured input
/// capacity, which only a complete report with every rank's rate states: a
/// partial report or a rank without a rate leaves it unknown (zero), and
/// the scorer prices the worker at the fleet's nominal rate. Any other
/// role's drain is the generation rate of the ranks with a request running.
/// A known rate under one token a second reads as one, zero being unknown.
/// The resident tokens are counted by such a complete report alone: a
/// rank the report leaves out holds tokens no sum of the others states.
/// `source` is where the report came from; `report_age_ms` is `now` less
/// `sampled_at`.
pub fn load_row(
    report: &WorkerLoadResponse,
    role: Role,
    now: Instant,
    sampled_at: Instant,
    source: SignalSource,
) -> LoadRow {
    let mut running: i64 = 0;
    let mut queued: i64 = 0;
    let mut uncached: i64 = 0;
    let mut max_running: i64 = 0;
    let mut kv_usage = 0.0f64;
    let mut active_kv = 0.0f64;
    let mut live_rate = 0.0f64;
    let complete = report.lists_every_rank();
    let mut prefill_rate: Option<f64> = complete.then_some(0.0);
    let mut resident: Option<i64> = complete.then_some(0);
    for rank in &report.loads {
        running += i64::from(rank.num_running_reqs);
        queued += i64::from(rank.num_waiting_reqs);
        uncached += i64::from(rank.num_waiting_uncached_tokens);
        max_running += i64::from(rank.max_running_requests);
        kv_usage += rank.token_usage;
        active_kv += rank.active_token_usage.unwrap_or(rank.token_usage);
        match role {
            Role::Prefill => {
                prefill_rate = prefill_rate
                    .zip(
                        rank.prefill_throughput
                            .filter(|rate| rate.is_finite() && *rate > 0.0),
                    )
                    .map(|(sum, rate)| sum + rate);
            }
            _ if rank.num_running_reqs > 0 => live_rate += rank.gen_throughput,
            _ => {}
        }
        resident = resident.zip(rank.num_resident_tokens).map(|(a, b)| a + b);
    }
    if role == Role::Prefill {
        live_rate = prefill_rate.unwrap_or(0.0);
    }
    let ranks = report.loads.len().max(1) as f64;
    LoadRow {
        running: narrow(running),
        queued: narrow(queued),
        waiting_uncached_tokens: narrow(uncached),
        kv_usage_permille: permille(kv_usage / ranks),
        active_kv_permille: permille(active_kv / ranks),
        live_rate: if live_rate.is_finite() && live_rate > 0.0 {
            // Zero says no rate: cut to it, a rate under one token a second
            // would drain its row at the mean rate of the row's peers.
            live_rate.clamp(1.0, f64::from(u32::MAX)) as u32
        } else {
            0
        },
        resident_tokens: resident.map_or(0, narrow),
        // Counted when the report is complete and every rank it lists sends it.
        resident_counted: resident.is_some(),
        report_age_ms: u32::try_from(now.saturating_duration_since(sampled_at).as_millis())
            .unwrap_or(u32::MAX),
        source,
        max_running: narrow(max_running),
    }
}

/// A summed engine count as the row's field: negative reads as none.
fn narrow(count: i64) -> u32 {
    u32::try_from(count.max(0)).unwrap_or(u32::MAX)
}

/// One row's values of the declared signals from its report, each handed
/// to `each` under its column, with the unit the report states, the value
/// and its sample instant: what every source of a report writes to the
/// signal sink behind the report's row ([`load_row`]), so each makes the
/// same values of the same report. The column is the one of `columns` the
/// value was made under, and the sink's write names it.
///
/// `columns` are the declared signals in name order
/// ([`SignalSink::signal_columns`](crate::store::SignalSink::signal_columns)):
/// a name the document does not declare is not offered. A row has a value
/// only where every rank of the report states the name, under one unit: a
/// sum over some of the ranks would understate the row, as the resident
/// tokens are counted. The ranks' values are made one as the signal
/// declares ([`AcrossRanks`]), under the oldest sample among them. A rank
/// that states a name twice is read at its first statement. A report of
/// one rank, as a rank's own record and a scoped answer are, hands that
/// rank's values as they are and allocates nothing.
///
/// Each rank's entry is read as one sample. An entry that states its own
/// sample instant is read at it
/// ([`SchedulerLoadSnapshot::sampled_at`](smg_worker::worker::SchedulerLoadSnapshot::sampled_at)):
/// the report a worker over several ranks pushes lists each rank at its
/// latest record, and its one instant is its last record's, of whatever
/// rank. An entry that states none is as old as its report.
///
/// `sampled_at` is the sample instant of the report: a record's own, a
/// polled answer's stamp, else the poll's start. A value is sampled at
/// its entry's instant, less its own age where the report states one:
/// the age is the engine's, as of the report it made. A polled value
/// with no stamp is so never dated after its poll began, as its row is
/// not. A stamped one is dated no later than its answer's receipt.
pub fn signal_values(
    report: &WorkerLoadResponse,
    columns: &[SignalColumn],
    sampled_at: Instant,
    mut each: impl FnMut(&SignalColumn, &str, f64, Instant),
) {
    if columns.is_empty() {
        return;
    }
    let ranks = match report.loads.as_slice() {
        [] => return,
        // One rank: nothing is kept. Each declared name has a bit, set at
        // the name's first statement.
        [rank] if columns.len() <= u64::BITS as usize => {
            let own = rank.sampled_at.unwrap_or(sampled_at);
            let mut read = 0u64;
            for stated in rank.named_values() {
                let Some(at) = column_of(columns, &stated.name) else {
                    continue;
                };
                let bit = 1u64 << at;
                if read & bit == 0 {
                    read |= bit;
                    let sampled = sampled(&stated, own);
                    each(&columns[at], &stated.unit, stated.value, sampled);
                }
            }
            return;
        }
        ranks => ranks,
    };
    let mut made: Vec<Option<Made>> = vec![None; columns.len()];
    for (nth, rank) in (0u32..).zip(ranks) {
        let own = rank.sampled_at.unwrap_or(sampled_at);
        for stated in rank.named_values() {
            let Some(at) = column_of(columns, &stated.name) else {
                continue;
            };
            let sampled = sampled(&stated, own);
            let across = columns[at].across_ranks();
            Made::join(
                &mut made[at],
                nth,
                across,
                &stated.unit,
                stated.value,
                sampled,
            );
        }
    }
    hand(columns, made, ranks.len() as u32, each);
}

/// The place of the declared signal `name` among `columns`, which stand in
/// name order.
fn column_of(columns: &[SignalColumn], name: &str) -> Option<usize> {
    let found = columns.binary_search_by(|column| column.name().cmp(name));
    found.ok()
}

/// When `stated` was sampled, by an entry sampled at `sampled_at`: then,
/// less the value's own age where the report states one.
fn sampled(stated: &NamedValue, sampled_at: Instant) -> Instant {
    stated.age_ms.map_or(sampled_at, |age| {
        let age = Duration::from_millis(u64::from(age));
        sampled_at.checked_sub(age).unwrap_or(sampled_at)
    })
}

/// Hands `each` the value of every column that all `count` ranks joined
/// ([`Made::join`]), made one as the column's signal declares.
fn hand(
    columns: &[SignalColumn],
    made: impl IntoIterator<Item = Option<Made>>,
    count: u32,
    mut each: impl FnMut(&SignalColumn, &str, f64, Instant),
) {
    for (column, made) in columns.iter().zip(made) {
        let Some(made) = made.filter(|made| made.ranks == count) else {
            continue;
        };
        let value = match column.across_ranks() {
            AcrossRanks::Mean => made.value / f64::from(count),
            AcrossRanks::Sum | AcrossRanks::Max | AcrossRanks::Min => made.value,
        };
        each(column, &made.unit, value, made.oldest);
    }
}

/// One declared signal's value over the ranks read so far.
#[derive(Clone, Debug)]
struct Made {
    /// How many ranks stated the name, each rank from the first.
    ranks: u32,
    unit: Arc<str>,
    value: f64,
    oldest: Instant,
}

impl Made {
    /// Joins the `nth` rank's statement of a name to the name's `slot`:
    /// its `value` under its `unit`, sampled at `sampled`. A rank's later
    /// statement of the same name is passed over.
    fn join(
        slot: &mut Option<Self>,
        nth: u32,
        across: AcrossRanks,
        unit: &Arc<str>,
        value: f64,
        sampled: Instant,
    ) {
        match slot {
            None if nth == 0 => {
                *slot = Some(Self {
                    ranks: 1,
                    unit: Arc::clone(unit),
                    value,
                    oldest: sampled,
                });
            }
            // Every rank before this one stated the name, and this is
            // the rank's first statement of it. Under another unit the
            // row has no value: its slot is emptied, and a slot is
            // filled at the first rank alone, so no later statement
            // is read.
            Some(row) if row.ranks == nth => {
                if row.unit == *unit {
                    row.ranks += 1;
                    row.value = joined(across, row.value, value);
                    row.oldest = row.oldest.min(sampled);
                } else {
                    *slot = None;
                }
            }
            // A rank before this one left the name out or stated
            // another unit, or this rank states the name once more.
            _ => {}
        }
    }
}

/// The value over one more rank. A rank's value that is no finite number
/// makes the row's none, under every reduction: the sink then counts it,
/// where a maximum would pass it over.
fn joined(across: AcrossRanks, so_far: f64, next: f64) -> f64 {
    match across {
        AcrossRanks::Sum | AcrossRanks::Mean => so_far + next,
        AcrossRanks::Max | AcrossRanks::Min if !(so_far.is_finite() && next.is_finite()) => {
            f64::NAN
        }
        AcrossRanks::Max => so_far.max(next),
        AcrossRanks::Min => so_far.min(next),
    }
}

#[cfg(test)]
mod tests {
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
        // A value that is no number is handed on: the sink counts it.
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
}
