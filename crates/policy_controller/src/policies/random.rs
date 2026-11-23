//! The `random` policy: how it is composed, and its picker. The engine draws
//! with the picker too, under a sessions rule whose assignment is random.

use crate::{
    document::spec::ProfileSpec,
    engine::{
        decision::{Targets, MAX_TARGETS},
        framework::{Picker, PoolWord, TieBand, DRAW_BOUND},
        view::{Candidate, PoolFacts, RequestFacts},
    },
    fresh_seed,
    policies::standard_filters,
    stages::{
        lifecycle::Lifecycle,
        pickers::{draw_kept, push_drawn, target_of},
    },
    Draw,
};

/// How the `random` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    ProfileSpec {
        filter: standard_filters(),
        score: vec![],
        pick: "random".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// A uniform draw over the survivors from the decision's seed, then further
/// distinct draws as alternates. A place's pick is one draw of the calling
/// thread's own and names no alternate ([`Picker::pick_alone`]).
#[derive(Debug, Default)]
pub struct Random;

impl Lifecycle for Random {}

impl Picker for Random {
    fn name(&self) -> &'static str {
        "random"
    }

    // Picks by its draw.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn draws_first(&self) -> bool {
        true
    }

    // Its draw is its whole rule: a pick alone reads no row.
    fn picks_alone(&self) -> bool {
        true
    }

    /// One uniform draw over the pool, drawn again while its worker is
    /// refused, [`DRAW_BOUND`] draws at most: a uniform draw over the
    /// workers `keep` keeps. The draws are the calling thread's own
    /// (`fresh_seed`), as the seed of a request that states none is: no
    /// word is shared, and two processes draw apart. Past the bound the
    /// pool is left to its caller's pass ([`Self::pick_among`]).
    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        if pool.rows == 0 {
            return None;
        }
        let mut draw = Draw::new(fresh_seed());
        (0..DRAW_BOUND)
            .map(|_| draw.below(pool.rows))
            .find(|&at| keep(at))
    }

    /// A uniform draw among the workers a pass kept.
    fn pick_among(&self, _: u64, passed: usize) -> usize {
        Draw::new(fresh_seed()).below(passed)
    }

    /// The pick is a uniform draw over the pool, drawn again while its row
    /// is refused: a uniform draw over the rows the whole pass keeps. Each
    /// alternate is one further draw among the rows not drawn yet, handed
    /// out when every filter keeps it and dropped when one refuses. The
    /// rows handed out are distinct and uniform over the rows the pass
    /// keeps, and a pool with rows refused hands out fewer alternates than
    /// the pass would: the bound is spent on the pick alone.
    fn draw_first(
        &self,
        _: &RequestFacts<'_>,
        _: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        draw: &mut Draw,
    ) -> bool {
        let Some(pick) = draw_kept(draw, candidates, None, keep, &mut 0) else {
            return false;
        };
        kept.push(pick as u32);
        // The rows drawn so far, ascending: a draw below the count of the
        // rows left names one of them by stepping over these.
        let mut drawn = [pick; MAX_TARGETS];
        for len in 1..MAX_TARGETS.min(candidates.len()) {
            let row = step_over(&mut drawn, len, draw.below(candidates.len() - len));
            if keep(&candidates[row]) {
                kept.push(row as u32);
            }
        }
        true
    }

    fn order_drawn(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        push_drawn(candidates, kept, out);
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let mut draw = Draw::new(facts.seed);
        let n = kept.len();
        // The positions taken so far, ascending.
        let mut taken = [usize::MAX; MAX_TARGETS];
        for slot in 0..MAX_TARGETS.min(n) {
            // The j-th position among those not taken yet: a partial
            // Fisher-Yates without the copy of the survivor list.
            let chosen = step_over(&mut taken, slot, draw.below(n - slot));
            out.push(target_of(&candidates[kept[chosen] as usize]));
        }
    }
}

/// The `j`-th position not among `taken[..len]`, which stand ascending,
/// entered among them in its place. Neither the step over a position
/// taken nor the entry branches on `j`: a drawn value is no pattern a
/// branch can learn, and a branch missed costs more than every step here.
#[inline]
fn step_over(taken: &mut [usize; MAX_TARGETS], len: usize, j: usize) -> usize {
    let mut row = j;
    for &at in &taken[..len] {
        row += usize::from(row >= at);
    }
    // Entered by least and greatest: each position above the row moves up
    // one place.
    let mut held = row;
    for at in &mut taken[..len] {
        let (least, greatest) = ((*at).min(held), (*at).max(held));
        *at = least;
        held = greatest;
    }
    taken[len] = held;
    row
}

#[cfg(test)]
mod tests {
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
}
