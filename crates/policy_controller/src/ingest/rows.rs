//! A worker's load row from its report: the conversion the gateway's
//! policies have read since the engine host took the view, moved here so
//! the controller's drivers and the gateway's embedded path produce the
//! same row from the same report.

use std::time::Instant;

use openai_protocol::worker::WorkerLoadResponse;

use crate::view::{LoadRow, Role, SignalSource};

/// One worker's load row from its report, in one pass over the ranks: the
/// request and token counts summed, the KV ratios averaged, the drain rate
/// as the policies read it today. A prefill's drain is its measured input
/// capacity, which only a complete report with every rank's rate states: a
/// partial report or a rank without a rate leaves it unknown (zero), and
/// the scorer prices the worker at the fleet's nominal rate. Any other
/// role's drain is the generation rate of the ranks with a request running.
/// `source` is where the report came from; `report_age_ms` is `now` less
/// `sampled_at`.
///
/// `None` when a rank's KV usage, or the active usage it reports, is not a
/// finite number: the report says nothing of the worker's pressure, and its
/// reader has no row, as for a report that never came.
pub fn load_row(
    report: &WorkerLoadResponse,
    role: Role,
    now: Instant,
    sampled_at: Instant,
    source: SignalSource,
) -> Option<LoadRow> {
    let mut running: i64 = 0;
    let mut queued: i64 = 0;
    let mut uncached: i64 = 0;
    let mut max_running: i64 = 0;
    let mut kv_usage = 0.0f64;
    let mut active_kv = 0.0f64;
    let mut live_rate = 0.0f64;
    let complete = !report.loads.is_empty()
        && report.loads.len() >= usize::try_from(report.dp_rank_count).unwrap_or(0);
    let mut prefill_rate: Option<f64> = complete.then_some(0.0);
    let mut resident: Option<i64> = Some(0);
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
    // One figure that is not a finite number makes its sum one. As a ratio
    // it would convert to zero (a NaN, the negative infinity), and the
    // worker whose pressure nobody knows would read as the idle one.
    if !(kv_usage.is_finite() && active_kv.is_finite()) {
        return None;
    }
    if role == Role::Prefill {
        live_rate = prefill_rate.unwrap_or(0.0);
    }
    let ranks = report.loads.len().max(1) as f64;
    Some(LoadRow {
        running: narrow(running),
        queued: narrow(queued),
        waiting_uncached_tokens: narrow(uncached),
        kv_usage_permille: permille(kv_usage / ranks),
        active_kv_permille: permille(active_kv / ranks),
        live_rate: if live_rate.is_finite() && live_rate > 0.0 {
            live_rate.min(f64::from(u32::MAX)) as u32
        } else {
            0
        },
        resident_tokens: resident.map_or(0, narrow),
        report_age_ms: u32::try_from(now.saturating_duration_since(sampled_at).as_millis())
            .unwrap_or(u32::MAX),
        source,
        max_running: narrow(max_running),
    })
}

/// A summed engine count as the row's field: negative reads as none.
fn narrow(count: i64) -> u32 {
    u32::try_from(count.max(0)).unwrap_or(u32::MAX)
}

/// A ratio as the row's permille field: below zero reads zero, over one
/// reads one.
fn permille(ratio: f64) -> u16 {
    (ratio * 1000.0).round().clamp(0.0, 1000.0) as u16
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use openai_protocol::worker::SchedulerLoadSnapshot;

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

    /// The row of a report whose usage is a number.
    fn row_of(
        report: &WorkerLoadResponse,
        role: Role,
        now: Instant,
        sampled_at: Instant,
        source: SignalSource,
    ) -> LoadRow {
        load_row(report, role, now, sampled_at, source).expect("a usage that is a number")
    }

    #[test]
    fn counts_add_over_the_ranks_and_ratios_average() {
        let now = Instant::now();
        let two = report(vec![
            rank(3, 1, 0.2, 100.0, None),
            rank(5, 2, 0.6, 50.0, None),
        ]);
        let row = row_of(&two, Role::Single, now, now, SignalSource::Poll);
        assert_eq!((row.running, row.queued), (8, 3));
        assert_eq!(row.kv_usage_permille, 400);
        assert_eq!(row.active_kv_permille, 400);
        assert_eq!(row.live_rate, 150);
        assert_eq!(row.resident_tokens, 2_000);
        assert_eq!(row.max_running, 512);
        assert_eq!(row.source, SignalSource::Poll);
        assert_eq!(row.report_age_ms, 0);
    }

    #[test]
    fn an_idle_rank_adds_no_generation_rate() {
        let now = Instant::now();
        let one_idle = report(vec![
            rank(0, 0, 0.0, 999.0, None),
            rank(1, 0, 0.1, 40.0, None),
        ]);
        let row = row_of(&one_idle, Role::Decode, now, now, SignalSource::Events);
        assert_eq!(row.live_rate, 40);
    }

    #[test]
    fn a_prefill_drain_needs_every_rank_rate_of_a_complete_report() {
        let now = Instant::now();
        let complete = report(vec![
            rank(1, 0, 0.1, 0.0, Some(2_000.0)),
            rank(1, 0, 0.1, 0.0, Some(3_000.0)),
        ]);
        assert_eq!(
            row_of(&complete, Role::Prefill, now, now, SignalSource::Poll).live_rate,
            5_000
        );
        let one_rate_missing = report(vec![
            rank(1, 0, 0.1, 0.0, Some(2_000.0)),
            rank(1, 0, 0.1, 0.0, None),
        ]);
        assert_eq!(
            row_of(
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
            row_of(&partial, Role::Prefill, now, now, SignalSource::Poll).live_rate,
            0
        );
    }

    #[test]
    fn the_age_is_the_decision_instant_less_the_sample_instant() {
        let sampled = Instant::now();
        let now = sampled + Duration::from_millis(1_500);
        let row = row_of(
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
        let row = row_of(&odd, Role::Single, now, now, SignalSource::Poll);
        assert_eq!((row.running, row.queued), (0, 0));
        assert_eq!(row.kv_usage_permille, 1_000);
        assert_eq!(row.live_rate, 0);
    }

    /// Converted, a NaN and the negative infinity are zero and the positive
    /// infinity is one: two of the three read as an idle worker.
    #[test]
    fn a_usage_that_is_not_a_finite_number_is_no_row() {
        let now = Instant::now();
        for usage in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let one = report(vec![rank(3, 1, usage, 100.0, None)]);
            assert_eq!(
                load_row(&one, Role::Single, now, now, SignalSource::Poll),
                None,
                "{usage}"
            );
            // One rank of a report is enough: a mean over the others would be
            // a figure nobody measured.
            let two = report(vec![
                rank(3, 1, 0.2, 100.0, None),
                rank(5, 2, usage, 50.0, None),
            ]);
            assert_eq!(
                load_row(&two, Role::Decode, now, now, SignalSource::Events),
                None,
                "{usage} beside a rank with a figure"
            );
            // The usage alone is enough, beside an active usage the rank
            // states as a number: without one the active usage falls back
            // to the usage, and the first two reports refuse on either.
            let mut stated = rank(3, 1, usage, 100.0, None);
            stated.active_token_usage = Some(0.25);
            assert_eq!(
                load_row(
                    &report(vec![stated]),
                    Role::Single,
                    now,
                    now,
                    SignalSource::Poll
                ),
                None,
                "{usage} beside a stated active usage"
            );
        }
    }

    /// The active usage is the figure the pressure reads: one that is not a
    /// finite number is no row either, beside a usage that is a number.
    #[test]
    fn an_active_usage_that_is_not_a_finite_number_is_no_row() {
        let now = Instant::now();
        for active in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let mut one = rank(3, 1, 0.6, 100.0, None);
            one.active_token_usage = Some(active);
            assert_eq!(
                load_row(
                    &report(vec![one]),
                    Role::Single,
                    now,
                    now,
                    SignalSource::Poll
                ),
                None,
                "{active}"
            );
        }
        let mut stated = rank(3, 1, 0.6, 100.0, None);
        stated.active_token_usage = Some(0.25);
        let row = row_of(
            &report(vec![stated]),
            Role::Single,
            now,
            now,
            SignalSource::Poll,
        );
        assert_eq!((row.kv_usage_permille, row.active_kv_permille), (600, 250));
    }

    /// A negative usage is a number: the row is kept and the field reads
    /// zero, as a count below zero reads none.
    #[test]
    fn a_negative_usage_is_kept_and_reads_zero() {
        let now = Instant::now();
        let mut below = rank(1, 0, -0.3, 100.0, None);
        below.active_token_usage = Some(-2.0);
        let row = row_of(
            &report(vec![below]),
            Role::Single,
            now,
            now,
            SignalSource::Poll,
        );
        assert_eq!((row.kv_usage_permille, row.active_kv_permille), (0, 0));
        assert_eq!(row.running, 1);
    }
}
