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
/// Their sum over the ranks is exact in any order of the ranks, in a count
/// wider than a rank's: the row's field reads a sum past it as the most it
/// states and one under zero as none, and nothing wraps.
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
    let mut resident: Option<i128> = complete.then_some(0);
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
        resident = resident
            .zip(rank.num_resident_tokens)
            .map(|(sum, count)| sum + i128::from(count));
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

/// A summed engine count as the row's field: negative reads as none, and a
/// sum past the field as the most it states.
fn narrow(count: impl Into<i128>) -> u32 {
    u32::try_from(count.into().max(0)).unwrap_or(u32::MAX)
}

/// One row's values of the declared signals from its report, each handed
/// to `each` under its column, with the unit the report states, the value
/// and its sample instant: the crate's one conversion of a report's named
/// values, beside the report's row ([`load_row`]), so every reader makes
/// the same values of the same report. The column is the one of `columns`
/// the value was made under.
///
/// `columns` are the declared signals in name order
/// ([`AppliedSpec::signal_columns`](crate::document::AppliedSpec::signal_columns)):
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
/// makes the row's none, under every reduction: the write of its cell then
/// refuses it, where a maximum would pass it over.
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
#[path = "../../tests/unit/load/load_report/tests.rs"]
mod tests;
