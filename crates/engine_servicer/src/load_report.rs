//! The arithmetic of a `GetLoads` answer every wire shares: one rank's
//! figures from the engine's load snapshot and the report's capacity, the
//! ranks of a zero-filled report, the aggregate over the ranks reported, and
//! the per-rank entries in the order every wire lists them
//! ([`rank_entries`]). A wire builds its own proto types from these; what
//! differs per wire (the fields it adds, how it names its version) stays in
//! the wire's module.

use openai_protocol::worker::SchedulerLoadSnapshot;

use crate::servicer::Shared;

/// One rank's load as the report states it.
pub(crate) struct RankFigures {
    pub(crate) dp_rank: i32,
    pub(crate) running: i32,
    pub(crate) waiting: i32,
    /// Running plus waiting, saturating.
    pub(crate) total: i32,
    /// The KV tokens in use: the snapshot's usage ratio over the report's
    /// capacity, rounded and clamped to the proto's range.
    pub(crate) used_tokens: i32,
    pub(crate) usage: f64,
}

impl RankFigures {
    pub(crate) fn of(load: &SchedulerLoadSnapshot, kv_capacity_tokens: i32) -> Self {
        let used_tokens = (load.token_usage * f64::from(kv_capacity_tokens)).round();
        Self {
            dp_rank: load.dp_rank,
            running: load.num_running_reqs,
            waiting: load.num_waiting_reqs,
            total: load.num_running_reqs.saturating_add(load.num_waiting_reqs),
            used_tokens: used_tokens.clamp(0.0, f64::from(i32::MAX)) as i32,
            usage: load.token_usage,
        }
    }
}

/// The capacity figures every rank's entry carries: the running window and
/// the KV capacity the servicer knows.
pub(crate) struct Capacity {
    pub(crate) max_running_requests: i32,
    pub(crate) max_total_num_tokens: i32,
}

impl Capacity {
    /// The figures from the shared state; `launcher_running_window` is the
    /// launcher's running-window figure, zero when it reported none.
    pub(crate) fn of(shared: &Shared, launcher_running_window: i32) -> Self {
        Self {
            max_running_requests: shared.running_window(launcher_running_window),
            max_total_num_tokens: shared.kv_capacity_tokens(),
        }
    }
}

/// The proto's rank numbers of the `count` ranks a servicer serves: `0..count`.
pub(crate) fn rank_ids(count: usize) -> impl Iterator<Item = i32> {
    (0..count).map(|rank| i32::try_from(rank).unwrap_or(i32::MAX))
}

/// The ranks of `0..count` with no entry in `reported`: a connected scheduler
/// before its first output batch (an idle one, whether or not a sibling rank
/// has reported). A report gives each a zero-filled entry, as the Python
/// servicer reports an idle scheduler, so it lists every connected rank from
/// boot on and a records-only stream reads an idle rank as idle, never as
/// gone, through its siblings' first batches.
pub(crate) fn unreported_ranks(count: usize, reported: &[i32]) -> impl Iterator<Item = i32> + '_ {
    rank_ids(count).filter(move |rank| !reported.contains(rank))
}

/// One rank's share of the aggregate: the fields of a reported rank the
/// totals are over, named so a caller cannot swap them unnoticed.
pub(crate) struct RankLoad {
    pub(crate) running: i32,
    pub(crate) waiting: i32,
    pub(crate) usage: f64,
}

/// The totals over the ranks reported.
pub(crate) struct Aggregate {
    pub(crate) running: i32,
    pub(crate) waiting: i32,
    /// Running plus waiting, saturating.
    pub(crate) total: i32,
    /// The mean usage ratio; 0 with no rank.
    pub(crate) avg_usage: f64,
}

impl Aggregate {
    /// Over the ranks reported, in one pass.
    pub(crate) fn of(ranks: impl IntoIterator<Item = RankLoad>) -> Self {
        let (mut running, mut waiting, mut usage_sum, mut count) = (0_i32, 0_i32, 0.0_f64, 0_usize);
        for rank in ranks {
            running = running.saturating_add(rank.running);
            waiting = waiting.saturating_add(rank.waiting);
            usage_sum += rank.usage;
            count += 1;
        }
        let avg_usage = if count == 0 {
            0.0
        } else {
            usage_sum / count as f64
        };
        Self {
            running,
            waiting,
            total: running.saturating_add(waiting),
            avg_usage,
        }
    }
}

/// The mean of one per-rank figure over the entries a wire built, 0 with no
/// entry: the aggregate's average throughput and utilization, which the
/// entries carry from the servicer's own estimates and the wire's fill.
pub(crate) fn mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let count = values.len();
    if count == 0 {
        return 0.0;
    }
    values.sum::<f64>() / count as f64
}

/// The per-rank entries of a `GetLoads` answer, as every wire lists them:
/// the snapshot's ranks (the one the request names, when it names one), each
/// built by `reported` from its snapshot and its figures, then a zero-filled
/// entry from `unreported` for every connected rank without one (see
/// [`unreported_ranks`]; a rank past the connected ones gets none), sorted
/// by rank; with the aggregate over all of them, the zero-filled ranks
/// counted as idle.
pub(crate) fn rank_entries<L>(
    snapshot: Vec<SchedulerLoadSnapshot>,
    kv_capacity_tokens: i32,
    ranks: usize,
    dp_rank: Option<i32>,
    mut reported: impl FnMut(&SchedulerLoadSnapshot, &RankFigures) -> L,
    mut unreported: impl FnMut(i32) -> L,
) -> (Vec<L>, Aggregate) {
    let asked = |rank: i32| dp_rank.is_none_or(|wanted| wanted == rank);
    let mut entries: Vec<(i32, L)> = Vec::with_capacity(ranks);
    let mut shares = Vec::with_capacity(ranks);
    for load in snapshot.into_iter().filter(|load| asked(load.dp_rank)) {
        let figures = RankFigures::of(&load, kv_capacity_tokens);
        shares.push(RankLoad {
            running: figures.running,
            waiting: figures.waiting,
            usage: figures.usage,
        });
        entries.push((figures.dp_rank, reported(&load, &figures)));
    }
    let seen: Vec<i32> = entries.iter().map(|(rank, _)| *rank).collect();
    for rank in unreported_ranks(ranks, &seen).filter(|&rank| asked(rank)) {
        shares.push(RankLoad {
            running: 0,
            waiting: 0,
            usage: 0.0,
        });
        entries.push((rank, unreported(rank)));
    }
    entries.sort_by_key(|(rank, _)| *rank);
    (
        entries.into_iter().map(|(_, entry)| entry).collect(),
        Aggregate::of(shares),
    )
}

/// One rank's entry of [`rank_entries`], read alone: the rank `dp_rank`
/// names (`None`: rank 0) as `load` reports it now, built by `reported`
/// from its snapshot and its figures; a connected rank (below `ranks`)
/// without a report as `unreported` builds it, zero-filled; a rank past the
/// connected ones gets none. The relay's per-rank record reads this, so it
/// stays the rank's entry of the full answer by construction.
pub(crate) fn rank_entry<L>(
    load: impl FnOnce(u32) -> Option<SchedulerLoadSnapshot>,
    kv_capacity_tokens: i32,
    ranks: usize,
    dp_rank: Option<i32>,
    reported: impl FnOnce(&SchedulerLoadSnapshot, &RankFigures) -> L,
    unreported: impl FnOnce(i32) -> L,
) -> Option<L> {
    let rank = dp_rank.unwrap_or(0);
    let index = u32::try_from(rank).ok()?;
    match load(index) {
        Some(load) => Some(reported(&load, &RankFigures::of(&load, kv_capacity_tokens))),
        None if usize::try_from(index).is_ok_and(|index| index < ranks) => Some(unreported(rank)),
        None => None,
    }
}

#[cfg(test)]
#[path = "../tests/unit/load_report/tests.rs"]
mod tests;
