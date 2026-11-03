//! The arithmetic of a `GetLoads` answer every wire shares: one rank's
//! figures from the engine's load snapshot and the report's capacity, the
//! ranks of a zero-filled report, and the aggregate over the ranks reported.
//! A wire builds its own proto types from these; what differs per wire (the
//! fields it adds, how it names its version) stays in the wire's module.

use openai_protocol::worker::SchedulerLoadSnapshot;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn load(dp_rank: i32, running: i32, waiting: i32, usage: f64) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            dp_rank,
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            token_usage: usage,
            ..Default::default()
        }
    }

    /// Used tokens are the usage ratio over the capacity, rounded; the sums
    /// saturate; the proto range bounds the figure.
    #[test]
    fn a_ranks_figures_follow_the_snapshot_and_the_capacity() {
        let figures = RankFigures::of(&load(2, 3, 4, 0.25), 1_000);
        assert_eq!(
            (
                figures.dp_rank,
                figures.running,
                figures.waiting,
                figures.total
            ),
            (2, 3, 4, 7)
        );
        assert_eq!(figures.used_tokens, 250);
        assert_eq!(figures.usage, 0.25);
        assert_eq!(RankFigures::of(&load(0, 1, 1, 0.3334), 10).used_tokens, 3);
        assert_eq!(
            RankFigures::of(&load(0, i32::MAX, 1, 0.0), 10).total,
            i32::MAX
        );
        assert_eq!(
            RankFigures::of(&load(0, 0, 0, 2.0), i32::MAX).used_tokens,
            i32::MAX
        );
        assert_eq!(RankFigures::of(&load(0, 0, 0, -1.0), 10).used_tokens, 0);
        assert_eq!(RankFigures::of(&load(0, 0, 0, 0.5), 0).used_tokens, 0);
    }

    /// A zero-filled report numbers its ranks from zero.
    #[test]
    fn rank_ids_count_from_zero() {
        assert_eq!(rank_ids(3).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(rank_ids(0).count(), 0);
    }

    /// The ranks a report zero-fills are the connected ones without an
    /// entry, whichever reported first; none past the count.
    #[test]
    fn the_unreported_ranks_are_the_connected_ones_without_an_entry() {
        assert_eq!(unreported_ranks(3, &[]).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(unreported_ranks(3, &[1]).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(unreported_ranks(1, &[0, 7]).count(), 0);
    }

    /// The aggregate sums the ranks (saturating) and averages their usage;
    /// with no rank every figure is zero.
    #[test]
    fn the_aggregate_sums_the_ranks_and_averages_their_usage() {
        let aggregate = Aggregate::of([
            RankLoad {
                running: 1,
                waiting: 2,
                usage: 0.5,
            },
            RankLoad {
                running: 3,
                waiting: 4,
                usage: 0.25,
            },
        ]);
        assert_eq!(
            (aggregate.running, aggregate.waiting, aggregate.total),
            (4, 6, 10)
        );
        assert_eq!(aggregate.avg_usage, 0.375);
        let empty = Aggregate::of(std::iter::empty::<RankLoad>());
        let saturated = Aggregate::of([
            RankLoad {
                running: i32::MAX,
                waiting: i32::MAX,
                usage: 0.0,
            },
            RankLoad {
                running: 1,
                waiting: 1,
                usage: 0.0,
            },
        ]);
        assert_eq!(
            (saturated.running, saturated.waiting, saturated.total),
            (i32::MAX, i32::MAX, i32::MAX)
        );
        assert_eq!((empty.running, empty.waiting, empty.total), (0, 0, 0));
        assert_eq!(empty.avg_usage, 0.0);
    }
}
