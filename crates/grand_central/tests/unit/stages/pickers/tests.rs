use super::*;
use crate::stages::scorers::tests::{candidate, kept, picked, picked_within};

#[test]
fn equal_scores_spread_across_seeds() {
    let pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
    let mut picks = [0usize; 3];
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Best.pick(
            &facts,
            &pool,
            &kept(3),
            &[1.0; 3],
            TieBand::NORMALIZED,
            &mut out,
        );
        picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
    }
    assert!(picks.iter().all(|&n| n > 50), "{picks:?}");
}

/// Equal counts fall to the score; an exact tie on both is drawn from
/// the seed, as every tie within the band is; the alternates keep the
/// order.
#[test]
fn count_first_breaks_equal_counts_by_the_score_and_exact_ties_by_the_seed() {
    let pool = [
        candidate(0, 3),
        candidate(1, 3),
        candidate(2, 3),
        candidate(3, 7),
    ];
    let by_position = |rows: &[u32]| -> Vec<f32> {
        rows.iter()
            .map(|&row| [0.2f32, 0.9, 0.9, 1.0][row as usize])
            .collect()
    };
    let mut firsts = std::collections::BTreeSet::new();
    for seed in 0..40u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let order = picked(&CountFirst, &facts, &pool, &by_position);
        assert!(order[0] == 1 || order[0] == 2, "{order:?}");
        assert_eq!((order[2], order[3]), (0, 3), "{order:?}");
        firsts.insert(order[0]);
    }
    assert_eq!(firsts.len(), 2, "an exact tie is drawn from the seed");
}

/// The fewest in flight are the set count-first compares: a scorer never
/// sees a worker that holds more, whatever its score would be, and the
/// workers passed over still follow as alternates by count.
#[test]
fn count_first_scores_the_fewest_in_flight_alone() {
    let pool = [
        candidate(0, 9),
        candidate(1, 3),
        candidate(2, 5),
        candidate(3, 3),
        candidate(4, 4),
    ];
    let seen = std::cell::RefCell::new(Vec::new());
    let record = |rows: &[u32]| -> Vec<f32> {
        seen.borrow_mut().push(rows.to_vec());
        rows.iter()
            .map(|&row| [0.0f32, 0.1, 0.2, 0.3, 0.4][row as usize])
            .collect()
    };
    let order = picked(&CountFirst, &RequestFacts::default(), &pool, &record);
    assert_eq!(*seen.borrow(), [vec![1, 3]], "the two that hold three");
    assert_eq!(
        order,
        [3, 1, 4, 2],
        "the better score of the two, then the next by count, wherever they sit"
    );
}

/// The engine has one band rule: the rows within the band of the best
/// score are drawn from, whatever order the pool lists them in. Three
/// near neighbours whose ends lie more than the band apart are listed in
/// each of their six orders: under the best picker and under count-first
/// the lowest is no part of the draw, and the other two share it. (A tie
/// walked against a running best would chain: the lowest would be drawn
/// wherever a neighbour of it is listed before the best.)
#[test]
fn a_tie_is_drawn_within_the_band_of_the_best_whatever_the_order_of_the_rows() {
    let chain = [candidate(0, 2), candidate(1, 2), candidate(2, 2)];
    let steps = [0.50f32, 0.58, 0.66];
    for order in [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let listed = |rows: &[u32]| -> Vec<f32> {
            rows.iter().map(|&row| steps[order[row as usize]]).collect()
        };
        for picker in [&Best as &dyn Picker, &CountFirst] {
            // How often each step of the chain leads, the lowest first.
            let mut leads = [0u32; 3];
            for seed in 0..2_000u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let ranked = picked_within(picker, &facts, &chain, &listed, TieBand(0.1));
                leads[order[ranked[0] as usize]] += 1;
            }
            assert_eq!(leads[0], 0, "{}, {order:?}: {leads:?}", picker.name());
            assert!(
                leads[1..].iter().all(|n| (850..=1150).contains(n)),
                "{}, {order:?}: {leads:?}",
                picker.name()
            );
        }
    }
}

/// A slot's draw reads one value of the seed's stream for a band of
/// several rows, however many, and none for a row alone in its band. A
/// band of many is drawn evenly, and of two tied rows a zero takes the
/// later.
#[test]
fn a_band_is_drawn_with_one_value_of_the_stream_and_evenly() {
    // The row drawn, and how many values of the stream the draw read:
    // none, one, or more than one.
    let drawn = |scores: &[f32], seed: u64| {
        let mut draw = Draw::new(seed);
        let row = draw_in_band(scores, TieBand::NORMALIZED, &mut draw, |_| true);
        // Where the stream stands past its seed, by its next value: no
        // value read, one, or none of the two.
        let next = draw.next_u64();
        let mut twin = Draw::new(seed);
        (row, (0..=1usize).find(|_| twin.next_u64() == next))
    };
    assert_eq!(drawn(&[], 7), (None, Some(0)));
    assert_eq!(drawn(&[0.2, 0.9, 0.4], 7), (Some(1), Some(0)));
    let mut led = [0u32; 64];
    for seed in 0..6_400u64 {
        let (row, read) = drawn(&[0.5; 64], seed);
        assert_eq!(read, Some(1), "seed {seed}");
        led[row.expect("a band holds a row")] += 1;
    }
    assert!(led.iter().all(|n| (50..=150).contains(n)), "{led:?}");
    // Two tied rows share the stream's first value below two: a zero is
    // the later row.
    for seed in 0..64u64 {
        let later = Draw::new(seed).below(2) == 0;
        assert_eq!(
            drawn(&[0.1, 0.7, 0.7], seed),
            (Some(1 + usize::from(later)), Some(1)),
            "seed {seed}"
        );
    }
}

/// A band whose rows score alike is counted once and drawn by rank: the
/// targets are the ones the slots draw one by one, seed for seed,
/// whatever stands above the band and under it. A band whose rows differ
/// within it stays with the slots.
#[test]
fn a_flat_band_is_drawn_by_rank_as_the_slots_draw_it_one_by_one() {
    let band = TieBand::NORMALIZED;
    // The slots one by one: each draws in its band among the rows left.
    let one_by_one = |scores: &[f32], seed: u64| {
        let (mut draw, mut taken) = (Draw::new(seed), Vec::new());
        while taken.len() < MAX_TARGETS.min(scores.len()) {
            let open = |i: usize| !taken.contains(&i);
            let Some(row) = draw_in_band(scores, band, &mut draw, open) else {
                break;
            };
            taken.push(row);
        }
        taken
    };
    // Rows above the band, the band's rows (alike, or two values within
    // the band), and rows under it that tie again.
    let near: f32 = 0.5 - 4e-7;
    let shapes: [(&[f32], &[f32], &[f32]); 8] = [
        (&[], &[0.5, 0.5], &[]),
        (&[], &[0.5, 0.5, 0.5], &[0.3, 0.2]),
        (&[0.9], &[0.5, 0.5], &[0.3, 0.3, 0.3]),
        (&[0.9, 0.8], &[0.5; 5], &[0.1]),
        (&[], &[0.5; 9], &[]),
        (&[0.9, 0.8, 0.7], &[0.5; 4], &[0.2, 0.2]),
        (&[], &[0.5, near, 0.5], &[0.3, 0.3]),
        (&[0.9], &[near, 0.5, near, 0.5], &[]),
    ];
    for (above, tied, under) in shapes {
        let mut scores: Vec<f32> = [above, tied, under].concat();
        let pool: Vec<Candidate> = (0..scores.len() as u32).map(|i| candidate(i, 0)).collect();
        let rows = kept(pool.len() as u32);
        for turn in 0..scores.len() {
            scores.rotate_left(1);
            for seed in 0..100u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let mut out = Targets::new();
                Best.pick(&facts, &pool, &rows, &scores, band, &mut out);
                let got: Vec<usize> = out.iter().map(|t| t.position as usize).collect();
                assert_eq!(
                    got,
                    one_by_one(&scores, seed),
                    "{scores:?} at turn {turn}, seed {seed}"
                );
            }
        }
    }
}

/// A band that ends among the five best is read off them, and the rows
/// and the draws are the ones the passes over the pool give. The pick
/// kept in this case reads by passes: single bands off the five, one
/// flat band by its count, then each slot left by passes of its own.
/// The pools tie whole, differ row by row or tie in bands of every
/// length around the five, some with rows that differ inside a band and
/// some with values no score should be; the bands are narrow, none,
/// wide, below zero and no number; the slots are the list's room or
/// fewer; the five are read by a pass and behind a draw.
#[test]
fn a_band_that_ends_among_the_five_best_is_drawn_as_the_passes_over_the_pool_draw_it() {
    fn by_passes<const DRAWN: bool>(
        draw: &mut Draw,
        scores: &[f32],
        band: TieBand,
        slots: usize,
    ) -> Vec<usize> {
        let mut picked = Vec::new();
        let mut best_five = [(usize::MAX, f32::NEG_INFINITY); MAX_TARGETS + 1];
        let len = if DRAWN {
            top_drawn(scores, &mut best_five)
        } else {
            top_pass(scores, &mut best_five)
        };
        let mut taken = [usize::MAX; MAX_TARGETS];
        let mut filled = 0usize;
        while filled < slots
            && (filled + 1 >= len || below_band(best_five[filled + 1].1, best_five[filled].1, band))
        {
            let (i, _) = best_five[filled];
            taken[filled] = i;
            picked.push(i);
            filled += 1;
        }
        if filled < slots {
            let mut picks = [usize::MAX; MAX_TARGETS];
            let top = best_five[filled].1;
            let fills = draw_flat(draw, scores, top, band, &mut picks[..slots - filled]);
            for &i in &picks[..fills] {
                taken[filled] = i;
                picked.push(i);
                filled += 1;
            }
        }
        for slot in filled..slots {
            let left = |i: usize| !taken[..slot].contains(&i);
            let Some(b) = draw_in_band(scores, band, draw, left) else {
                break;
            };
            taken[slot] = b;
            picked.push(b);
        }
        picked
    }
    // Both readings from one seed: the rows, and where the stream
    // stands behind them.
    fn held<const DRAWN: bool>(case: u32, scores: &[f32], band: TieBand, slots: usize, seed: u64) {
        let (mut read, mut passed) = (Draw::new(seed), Draw::new(seed));
        let mut rows = Vec::new();
        draw_ranked::<DRAWN>(&mut read, scores, band, slots, |row| rows.push(row));
        let by = by_passes::<DRAWN>(&mut passed, scores, band, slots);
        assert_eq!(
            (rows, read.state()),
            (by, passed.state()),
            "case {case}: {slots} of {scores:?} under {band:?}, seed {seed}, drawn {DRAWN}"
        );
    }
    // Under the last of the two bands below zero a score of one lies in
    // its own band and a lower score lies under its own.
    let bands = [1e-6, 0.0, 0.2, -0.2, -3e-8, f32::NAN, f32::INFINITY];
    let odd = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
    let mut pools = Draw::new(0x5eed);
    for case in 0..60_000u32 {
        let rows = pools.below(40);
        // The levels the rows fall on: one ties the pool whole, and
        // sixty-four leave few rows a level.
        let levels = 1usize << pools.below(7);
        let apart = pools.below(8) == 0;
        let (near, wild) = (pools.below(4) == 0, pools.below(4) == 0);
        let scores: Vec<f32> = (0..rows)
            .map(|row| {
                let level = if apart {
                    (row * 37 + case as usize) % 64
                } else {
                    pools.below(levels) * (64 / levels)
                };
                let score = 1.0 - level as f32 / 64.0;
                if wild && pools.below(8) == 0 {
                    odd[pools.below(odd.len())]
                } else if near && pools.below(4) == 0 {
                    score - 4e-7
                } else {
                    score
                }
            })
            .collect();
        let band = TieBand(bands[pools.below(bands.len())]);
        // The list's room, and now and then less of it.
        let room = MAX_TARGETS.min(rows);
        let slots = if pools.below(4) == 0 {
            pools.below(room + 1)
        } else {
            room
        };
        let seed = pools.next_u64();
        held::<false>(case, &scores, band, slots, seed);
        held::<true>(case, &scores, band, slots, seed);
    }
}

/// A ranked draw asked for a slot its scores do not reach is its
/// caller's fault: a debug build stops there, before a place past the
/// scores is handed out.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "no more slots than its list and its scores hold")]
fn a_slot_past_the_scores_stops_a_ranked_draw_in_a_debug_build() {
    draw_ranked::<false>(&mut Draw::new(1), &[0.3, 0.2, 0.1], TieBand(0.0), 4, |_| {});
}

/// And so is a slot past a target list's room, whatever the scores hold.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "no more slots than its list and its scores hold")]
fn a_slot_past_the_list_stops_a_ranked_draw_in_a_debug_build() {
    let scores = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    draw_ranked::<false>(
        &mut Draw::new(1),
        &scores,
        TieBand(0.0),
        MAX_TARGETS + 1,
        |_| {},
    );
}

/// The best rows of an engine are the slots of the ranked push drawn
/// one by one, seed for seed and for every count of slots asked: among
/// few rows and many, where rows tie and where each stands alone in
/// its band, under a narrow band and a wide one.
#[test]
fn the_best_rows_of_an_engine_are_the_ranked_slots_one_by_one() {
    let one_by_one = |scores: &[f32], band: TieBand, seed: u64, slots: usize| {
        let (mut draw, mut taken) = (Draw::new(seed), Vec::new());
        while taken.len() < slots.min(scores.len()) {
            let open = |i: usize| !taken.contains(&i);
            let Some(row) = draw_in_band(scores, band, &mut draw, open) else {
                break;
            };
            taken.push(row);
        }
        taken
    };
    let mut levels = Draw::new(7);
    let mut tied = 0;
    for rows in [0usize, 1, 2, 3, 4, 5, 8, 13, 63, 64, 65, 90] {
        // Scores on a few levels, where rows tie, and on many, where
        // few do; and on the last turn one score that is no number,
        // which stands in every tie.
        for spread in [1usize, 2, 3, 40, 41] {
            let mut scores: Vec<f32> = (0..rows)
                .map(|_| levels.below(spread) as f32 / spread as f32)
                .collect();
            if let (41, Some(score)) = (spread, scores.get_mut(rows / 2)) {
                *score = f32::NAN;
            }
            for band in [TieBand::NORMALIZED, TieBand(0.3)] {
                for seed in 0..40u64 {
                    for slots in 0..=MAX_TARGETS {
                        let mut out = [usize::MAX; MAX_TARGETS];
                        let filled = draw_best(Draw::new(seed), &scores, band, &mut out[..slots]);
                        let ruled = one_by_one(&scores, band, seed, slots);
                        assert_eq!(
                            &out[..filled],
                            &ruled[..],
                            "{rows} rows on {spread} levels, seed {seed}, {slots} slots"
                        );
                        assert_eq!(filled, slots.min(rows));
                    }
                    // The draw decides somewhere: the slots of one seed
                    // are not the slots of every seed.
                    let first = |seed| one_by_one(&scores, band, seed, 1);
                    tied += usize::from(first(seed) != first(seed + 1));
                }
            }
        }
    }
    assert!(tied > 500, "{tied}");
}

/// Count-first orders no more of the workers it passed over than the
/// target list has room for behind the fewest in flight: the narrowing
/// costs the list's room times the pool, never the pool's square. The
/// ones it orders go by count, and equal counts are drawn.
#[test]
fn count_first_orders_no_more_alternates_than_the_target_list_has_room_for() {
    let narrowed = |pool: &[Candidate]| {
        let rows = &mut kept(pool.len() as u32);
        let narrowed = CountFirst.narrow(&RequestFacts::default(), pool, rows);
        (narrowed.compared, narrowed.alternates)
    };
    // Two hold one request and eight hold more: two of the eight follow.
    let spread: Vec<Candidate> = (0..10).map(|i| candidate(i, i.max(1))).collect();
    assert_eq!(narrowed(&spread), (2, MAX_TARGETS - 2));
    // The fewest fill the list alone: no worker behind them is ordered.
    let level: Vec<Candidate> = (0..10).map(|i| candidate(i, 3)).collect();
    assert_eq!(narrowed(&level), (10, 0));
    // Fewer behind the fewest than the list has room for: all of them.
    let short = [candidate(0, 5), candidate(1, 2), candidate(2, 7)];
    assert_eq!(narrowed(&short), (1, 2));
    // Four workers hold the next count and two of them follow, drawn:
    // over the seeds each of the four stands first among the alternates,
    // so a share does not follow where a host lists a worker.
    let counts = [1u32, 1, 2, 2, 2, 2, 5, 6, 7, 8];
    let tied: Vec<Candidate> = (0..10).map(|i| candidate(i, counts[i as usize])).collect();
    let mut firsts = std::collections::BTreeSet::new();
    for seed in 0..64u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut rows = kept(10);
        let narrowed = CountFirst.narrow(&facts, &tied, &mut rows);
        assert_eq!(narrowed.alternates, 2, "seed {seed}");
        assert!(
            rows[2] != rows[3] && rows[2..4].iter().all(|row| (2..6).contains(row)),
            "seed {seed}: {rows:?}"
        );
        firsts.insert(rows[2]);
    }
    assert_eq!(
        firsts,
        std::collections::BTreeSet::from([2, 3, 4, 5]),
        "equal counts are drawn, never taken as listed"
    );
}

/// Count-first handed rows that were never narrowed applies its rule a
/// slot: the fewest in flight of the rows left, and among those the tie
/// within the band of the best score. A plugin picker's fallback is
/// handed every survivor with their scores: it picks as the engine's
/// order does over the same scores, whatever the busier rows score. The
/// two orders are one past the fewest too because this pool holds each
/// higher count once: among equal counts there the pick goes by the
/// scores it was handed, where the narrowing draws them unscored.
#[test]
fn count_first_over_rows_that_were_not_narrowed_picks_as_the_engines_order_does() {
    let pool = [
        candidate(0, 4),
        candidate(1, 2),
        candidate(2, 5),
        candidate(3, 2),
        candidate(4, 3),
        candidate(5, 2),
    ];
    // Of the three that hold two requests, two tie within the band and
    // one lies below it; the busier rows hold the best scores.
    let scores = [0.9f32, 0.52, 1.0, 0.30, 0.95, 0.56];
    let by_position =
        |rows: &[u32]| -> Vec<f32> { rows.iter().map(|&row| scores[row as usize]).collect() };
    let band = TieBand(0.1);
    let mut leads = std::collections::BTreeSet::new();
    for seed in 0..200u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        CountFirst.pick(&facts, &pool, &kept(6), &scores, band, &mut out);
        let whole: Vec<u32> = out.iter().map(|t| t.position).collect();
        assert_eq!(
            whole,
            picked_within(&CountFirst, &facts, &pool, &by_position, band),
            "seed {seed}"
        );
        assert_eq!(
            (whole[2], whole[3]),
            (3, 4),
            "seed {seed}: the last of the fewest, then the next count"
        );
        leads.insert(whole[0]);
    }
    assert_eq!(
        leads,
        std::collections::BTreeSet::from([1, 5]),
        "the two within the band are drawn"
    );
}

/// Count-first draws from two streams of the decision's seed that share
/// no value: the tie among the fewest from the seed's own stream, as the
/// best picker draws it, and the order of equal counts among the workers
/// passed over from the seed under the alternates' mask. Two tied
/// leaders over two tied followers so fall four ways, each about as
/// often. (One stream begun twice would tie the two draws: two ways of
/// the four.)
#[test]
fn count_first_draws_its_tie_and_its_alternates_from_streams_that_share_no_value() {
    let pool = [
        candidate(0, 1),
        candidate(1, 1),
        candidate(2, 4),
        candidate(3, 4),
    ];
    let level = |rows: &[u32]| vec![0.5f32; rows.len()];
    let mut ways = [0u32; 4];
    for seed in 0..4_000u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let order = picked(&CountFirst, &facts, &pool, &level);
        assert_eq!(order.len(), 4);
        assert!(order[0] < 2 && order[1] < 2, "the fewest lead: {order:?}");
        // A later row of a tie takes the lead when its draw is zero.
        let follower = if Draw::new(seed ^ ALTERNATES_STREAM).below(2) == 0 {
            3
        } else {
            2
        };
        let leader = if Draw::new(seed).below(2) == 0 { 1 } else { 0 };
        assert_eq!((order[0], order[2]), (leader, follower), "seed {seed}");
        ways[(order[0] * 2 + order[2] - 2) as usize] += 1;
    }
    assert!(ways.iter().all(|n| (850..=1150).contains(n)), "{ways:?}");
}

#[test]
fn single_bands_are_taken_in_order_and_a_shared_band_behind_them_is_drawn_from() {
    // Two rows alone in their bands lead every pick in order; the three
    // rows that share the next band take the third slot by the seed, and
    // the fourth slot is another of them.
    let pool = [
        candidate(0, 0),
        candidate(1, 0),
        candidate(2, 0),
        candidate(3, 0),
        candidate(4, 0),
    ];
    let scores = [0.9f32, 1.0, 0.5, 0.5, 0.5];
    let mut thirds = [0u32; 5];
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Best.pick(
            &facts,
            &pool,
            &kept(5),
            &scores,
            TieBand::NORMALIZED,
            &mut out,
        );
        let order: Vec<u32> = out.iter().map(|t| t.position).collect();
        assert_eq!(&order[..2], &[1, 0], "seed {seed}: {order:?}");
        assert_eq!(order.len(), 4, "seed {seed}");
        assert!(
            order[2] >= 2 && order[3] >= 2 && order[2] != order[3],
            "seed {seed}: {order:?}"
        );
        thirds[order[2] as usize] += 1;
    }
    assert!(
        thirds[2] > 50 && thirds[3] > 50 && thirds[4] > 50,
        "the shared band is drawn from: {thirds:?}"
    );
}

#[test]
fn the_four_best_of_sixty_four_rows_are_picked_in_order_wherever_they_sit() {
    // The five-best array fills on the first rows and shifts for every
    // better one after it; the four best must come out in order whether
    // they sit scattered in a rising pool or arrive last, rising.
    let pool: Vec<Candidate> = (0..64).map(|i| candidate(i, 0)).collect();
    let kept = kept(64);
    let mut scattered: Vec<f32> = (0..64).map(|i| 0.1 + 0.004 * i as f32).collect();
    for (position, score) in [(7usize, 1.0f32), (40, 0.99), (63, 0.98), (22, 0.97)] {
        scattered[position] = score;
    }
    let mut last_rising: Vec<f32> = (0..64).map(|i| 0.1 + 0.004 * i as f32).collect();
    for (position, score) in [(60usize, 0.96f32), (61, 0.97), (62, 0.98), (63, 0.99)] {
        last_rising[position] = score;
    }
    for (scores, expected) in [
        (scattered, [7u32, 40, 63, 22]),
        (last_rising, [63, 62, 61, 60]),
    ] {
        for seed in 0..8u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept, &scores, TieBand::NORMALIZED, &mut out);
            let order: Vec<u32> = out.iter().map(|t| t.position).collect();
            assert_eq!(order, expected, "seed {seed}");
        }
    }
}

#[test]
fn a_wide_band_ties_against_the_best_score_and_not_the_running_best() {
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let scores = [0.95f32, 1.0, 0.92];
    let mut firsts = [0u32; 3];
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Best.pick(&facts, &pool, &[0, 1, 2], &scores, TieBand(0.06), &mut out);
        firsts[out.first().expect("a pick").position as usize] += 1;
    }
    assert_eq!(
        firsts[2], 0,
        "0.92 is outside the band of the best score 1.0"
    );
    assert!(
        firsts[0] > 100 && firsts[1] > 100,
        "the two tied rows share the draws: {firsts:?}"
    );
}

/// A sampled `best` draws over a pool of more than twice its sample and
/// over no smaller one: it keeps `sample` rows the filters keep, none
/// twice, and counts no refused rows: its draws are its bound, and a
/// sample that holds no row hands the decision to the whole pass.
/// Whether the host named rows is the engine's to ask.
#[test]
fn a_sampled_best_draws_its_sample_from_a_pool_over_twice_it() {
    let picker = SampledBest::new(8);
    assert_eq!(picker.needs().sample, 8);
    assert_eq!(SampledBest::new(0).needs().sample, 1);
    let pool: Vec<Candidate> = (0..64).map(|i| candidate(i, 0)).collect();
    let request = RequestFacts {
        seed: 9,
        ..RequestFacts::default()
    };
    // What a draw answers: it drew, the rows it kept, and whether it
    // read a value of the stream.
    let draws =
        |facts: &RequestFacts<'_>, rows: usize, keep: &mut dyn FnMut(&Candidate) -> bool| {
            let (mut kept, mut draw) = (Vec::new(), Draw::new(facts.seed));
            let drew = picker.draw_first(
                facts,
                &PoolFacts::default(),
                &pool[..rows],
                keep,
                &mut kept,
                &mut draw,
            );
            (drew, kept, draw.state() != facts.seed)
        };
    let (drew, kept, read) = draws(&request, 64, &mut |_| true);
    assert!(drew && read);
    let mut once = kept.clone();
    once.sort_unstable();
    once.dedup();
    assert_eq!((kept.len(), once.len()), (8, 8), "{kept:?}");
    // A pool of at most twice the sample: no draw, and no value of the
    // stream read. One row more is drawn from.
    assert_eq!(
        draws(&request, 16, &mut |_| true),
        (false, Vec::new(), false)
    );
    assert!(draws(&request, 17, &mut |_| true).0);
    // Every row refused: each of the sixteen draws asks, and the sample
    // that holds no row hands the decision to the whole pass.
    let mut asked = 0u32;
    let (drew, ..) = draws(&request, 64, &mut |_| {
        asked += 1;
        false
    });
    assert!(!drew);
    assert_eq!(asked, 16);
}

/// Under a filter that refuses a third and a half of the pool, the rows
/// a sampled `best` keeps hold even shares: each row the filter keeps
/// is kept as often as any other, within six deviations of a count that
/// size, and no refused row is kept.
#[test]
fn a_sampled_best_keeps_even_shares_under_a_filter_that_refuses() {
    let picker = SampledBest::new(8);
    let pool: Vec<Candidate> = (0..64).map(|i| candidate(i, 0)).collect();
    for one_in in [3u32, 2] {
        let mut counts = [0u32; 64];
        for seed in 0..4_000u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let (mut kept, mut draw) = (Vec::new(), Draw::new(seed));
            picker.draw_first(
                &facts,
                &PoolFacts::default(),
                &pool,
                &mut |row| row.position % one_in != 0,
                &mut kept,
                &mut draw,
            );
            for row in kept {
                counts[row as usize] += 1;
            }
        }
        let open = 64 - 64u32.div_ceil(one_in);
        let even = f64::from(counts.iter().sum::<u32>()) / f64::from(open);
        for (row, &count) in (0u32..).zip(&counts) {
            if row % one_in == 0 {
                assert_eq!(count, 0, "one in {one_in} refused: row {row} was kept");
                continue;
            }
            let off = (f64::from(count) - even).abs();
            assert!(
                off < 6.0 * even.sqrt(),
                "one in {one_in} refused: row {row} is {off:.0} off an even {even:.0}"
            );
        }
    }
}

/// A sampled `best` reads at most twice its sample of the stream's
/// values. Where repeats use them up it hands out the rows it kept,
/// fewer than the sample, and reads no value more.
#[test]
fn a_sampled_best_reads_at_most_twice_its_sample_of_the_stream() {
    let picker = SampledBest::new(2);
    // One row past twice the sample: one seed in about a hundred draws
    // one row four times.
    let pool: Vec<Candidate> = (0..5).map(|i| candidate(i, 0)).collect();
    let mut short = 0u32;
    for seed in 0..2_000u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let (mut kept, mut draw) = (Vec::new(), Draw::new(seed));
        let pool_facts = PoolFacts::default();
        assert!(picker.draw_first(
            &facts,
            &pool_facts,
            &pool,
            &mut |_| true,
            &mut kept,
            &mut draw
        ));
        // The values read: the stream's steps from the seed to its state.
        let (mut twin, mut read) = (Draw::new(seed), 0usize);
        while twin.state() != draw.state() {
            twin.next_u64();
            read += 1;
            assert!(read <= 4, "seed {seed}: more than twice the sample read");
        }
        assert!(
            read >= kept.len(),
            "seed {seed}: {kept:?} from {read} values"
        );
        if kept.len() < 2 {
            assert_eq!(read, 4, "seed {seed}: {kept:?}");
            short += 1;
        }
    }
    assert!(short > 0, "no seed spent its draws on one row");
}
