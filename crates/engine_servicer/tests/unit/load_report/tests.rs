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
        let (entries, _) = rank_entries(snapshot.clone(), 1_000, 3, dp_rank, reported, unreported);
        assert_eq!(alone, entries.first().copied(), "rank {dp_rank:?}");
    }
}
