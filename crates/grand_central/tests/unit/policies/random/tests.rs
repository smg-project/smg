use super::*;
use crate::stages::scorers::tests::{candidate, kept};

#[test]
fn random_draws_distinct_alternates_and_spreads_across_seeds() {
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let mut picks = [0usize; 3];
    for seed in 0..300u64 {
        let facts = RequestFacts {
            seed,
            ..RequestFacts::default()
        };
        let mut out = Targets::new();
        Random.pick(
            &facts,
            &pool,
            &kept(3),
            &[0.0; 3],
            TieBand::NORMALIZED,
            &mut out,
        );
        let positions: Vec<u32> = out.iter().map(|t| t.position).collect();
        let mut distinct = positions.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "{positions:?}");
        picks[positions[0] as usize] += 1;
    }
    assert!(picks.iter().all(|&n| n > 50), "{picks:?}");
}

/// Random's pick and its draw name the row the loops they replace
/// named: the j-th position not taken yet, found by stepping over the
/// positions taken, which stand ascending behind each step. Every
/// pool size the whole pass and the draw meet, over many seeds.
#[test]
fn a_random_draw_steps_over_the_positions_taken_as_the_loop_did() {
    for n in 1..=40usize {
        for seed in 0..400u64 {
            let mut draw = Draw::new(seed);
            let mut taken = [usize::MAX; MAX_TARGETS];
            let mut looped = [usize::MAX; MAX_TARGETS];
            for slot in 0..MAX_TARGETS.min(n) {
                let j = draw.below(n - slot);
                let stepped = step_over(&mut taken, slot, j);
                // The loop as it stood: the j-th position not taken.
                let (mut seen, mut chosen) = (0usize, 0usize);
                for i in 0..n {
                    if looped[..slot].contains(&i) {
                        continue;
                    }
                    if seen == j {
                        chosen = i;
                        break;
                    }
                    seen += 1;
                }
                looped[slot] = chosen;
                assert_eq!(stepped, chosen, "{n} rows, seed {seed}, slot {slot}");
                assert!(taken[..=slot].is_sorted(), "{n} rows, seed {seed}");
                let mut ascending = looped;
                ascending[..=slot].sort_unstable();
                assert_eq!(taken, ascending, "{n} rows, seed {seed}");
            }
        }
    }
}
