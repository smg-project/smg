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

    /// The entries are the asked ranks of the snapshot plus a zero-filled one
    /// per silent connected rank, sorted by rank; the aggregate counts the
    /// zero-filled ranks as idle.
    #[test]
    fn the_entries_list_the_asked_ranks_zero_fill_the_rest_in_order_and_aggregate_all() {
        // Rank 2 reported before rank 0; rank 1 connected but silent.
        let snapshot = vec![load(2, 3, 1, 0.5), load(0, 1, 0, 0.25)];
        let (entries, aggregate) = rank_entries(
            snapshot.clone(),
            1000,
            3,
            None,
            |load, figures| {
                (
                    figures.dp_rank,
                    figures.running,
                    load.num_waiting_reqs,
                    false,
                )
            },
            |rank| (rank, 0, 0, true),
        );
        assert_eq!(
            entries,
            vec![(0, 1, 0, false), (1, 0, 0, true), (2, 3, 1, false)]
        );
        assert_eq!(
            (aggregate.running, aggregate.waiting, aggregate.total),
            (4, 1, 5)
        );
        assert!(
            (aggregate.avg_usage - 0.25).abs() < 1e-9,
            "{}",
            aggregate.avg_usage
        );
        // Asking for one rank keeps that rank's entry alone, reported or zero-filled.
        let (one, _) = rank_entries(snapshot.clone(), 1000, 3, Some(2), |_, f| f.dp_rank, |r| -r);
        assert_eq!(one, vec![2]);
        let (silent, _) = rank_entries(snapshot, 1000, 3, Some(1), |_, f| f.dp_rank, |r| -r);
        assert_eq!(silent, vec![-1]);
    }

    /// The mean of a per-rank figure over the entries built; zero with none.
    #[test]
    fn the_mean_of_a_figure_over_the_entries_is_zero_with_none() {
        assert_eq!(mean(std::iter::empty()), 0.0);
        assert_eq!(mean([1.0, 2.0, 6.0].into_iter()), 3.0);
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

    /// One rank read alone is that rank's entry of the full answer: a
    /// reported rank from its snapshot, a connected rank without a report
    /// zero-filled, a rank past the connected ones absent, `None` naming
    /// rank 0.
    #[test]
    fn a_rank_read_alone_is_its_entry_of_the_full_answer() {
        let snapshot = vec![load(0, 1, 2, 0.5), load(2, 3, 4, 0.25)];
        let reported = |load: &SchedulerLoadSnapshot, figures: &RankFigures| {
            (
                figures.dp_rank,
                Some((load.num_running_reqs, figures.waiting)),
            )
        };
        let unreported = |rank: i32| (rank, None);
        for dp_rank in [None, Some(0), Some(1), Some(2), Some(3)] {
            let alone = rank_entry(
                |index| {
                    snapshot
                        .iter()
                        .find(|load| u32::try_from(load.dp_rank) == Ok(index))
                        .cloned()
                },
                1_000,
                3,
                dp_rank,
                reported,
                unreported,
            );
            let (entries, _) =
                rank_entries(snapshot.clone(), 1_000, 3, dp_rank, reported, unreported);
            assert_eq!(alone, entries.first().copied(), "rank {dp_rank:?}");
        }
    }
}
