//! The engine as a pick's unit. A pool of data-parallel engines holds a row
//! a rank, so a decision that reads its pool whole pays for every rank of
//! the fleet, and one that draws rows compares ranks of engines it knows
//! nothing else of. A picker whose unit is the engine ([`Picker::unit`])
//! decides over the engines the host lists beside the rows ([`Engines`]):
//! it draws two engines, compares one row of each, and reads the rows of
//! the engine it takes. The rows of an engine it passes by are not read, so
//! what the decision costs follows the ranks of one engine and not the
//! fleet.
//!
//! [`Picker::unit`]: super::Picker::unit

use super::{combine, run_single, trace_ordered, Buffers, Engine, Profile, Set, DRAW_BOUND};
use crate::{
    builtins::{draw_best, target_of},
    decision::{Decision, Pick, Targets, MAX_TARGETS},
    sessions::SessionVerdict,
    view::{Candidate, Constraints, Engines, PoolFacts, RequestFacts},
    Draw,
};

/// The targets a pick by engines hands out: the taken engine's least rank
/// and one alternate.
const ENGINE_TARGETS: usize = 2;

/// A single-plan decision by a picker whose unit is the engine
/// ([`Picker::unit`]) over a pool whose host lists its engines
/// ([`PoolFacts::engines`]), in front of the draw and at any pool size.
///
/// Two engines are drawn by the seed's stream, and a row of each stands
/// for it: the engine's one row, or one of its rows by one more value.
/// Every filter of the profile is asked about that row, as a draw asks,
/// and a row refused is drawn again with its engine, [`DRAW_BOUND`] rows
/// refused in the decision at the most. The profile's stages cost the two
/// rows among themselves and the picker orders them: the engine of the row
/// it puts first is taken. Then the taken engine's rows are read, and no
/// other engine's: the filters are asked about each, the stages cost the
/// rows kept among themselves, and the least of them is handed out
/// first, equal scores drawn. The alternate is the other engine's row:
/// the ranks of one engine are lost together, so a commit that loses the
/// taken engine goes to another. A pool of one engine draws nothing and
/// hands that engine's least rank and its next ([`ENGINE_TARGETS`]).
///
/// The stream is read in this order and no value twice: an engine and its
/// row for each of the two, again for each row refused; then one value,
/// the seed of the stream that draws between equal rows of the taken
/// engine; and the picker's own compare reads the stream past those.
///
/// Where no row is kept the draw says nothing of the pool: the decision
/// widens to the whole pass on the stream continued, which alone sheds,
/// names a stage and asks the all-vetoed rule. The trace keeps the
/// request's seed, from which the whole decision replays.
///
/// Out of line, so a decision of any other profile keeps the engine's
/// decision path as it was.
///
/// [`Picker::unit`]: super::Picker::unit
#[inline(never)]
pub(super) fn run_engines(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    let at = usize::from(index);
    let (profile, rows) = (&engine.profiles[at], engine.row_tests[at]);
    let candidates = set.candidates;
    buf.reserve(candidates.len());
    let mut draw = Draw::new(facts.seed);
    let decided = match rows {
        Some(tests) if tests.masked(cx) => {
            let keep = |c: &Candidate| tests.keeps(facts, c, cx) && !cx.masks(c.position);
            by_engines(
                profile, index, facts, pool, candidates, buf, &mut draw, keep,
            )
        }
        Some(tests) => {
            let keep = |c: &Candidate| tests.keeps(facts, c, cx);
            by_engines(
                profile, index, facts, pool, candidates, buf, &mut draw, keep,
            )
        }
        None => {
            let keep = |c: &Candidate| profile.filters.iter().all(|f| f.keep(facts, c, cx));
            by_engines(
                profile, index, facts, pool, candidates, buf, &mut draw, keep,
            )
        }
    };
    if let Some(decision) = decided {
        return decision;
    }
    let resumed = RequestFacts {
        seed: draw.state(),
        ..*facts
    };
    let mut decision = run_single(profile, index, &resumed, pool, set, cx, buf);
    match &mut decision {
        Decision::Pick(pick) => pick.trace.seed = facts.seed,
        Decision::Shed(shed) => shed.trace.seed = facts.seed,
        Decision::Queue(queue) => queue.trace.seed = facts.seed,
    }
    decision
}

/// A keyed single-plan decision by a profile whose one scorer places a
/// request by itself ([`Scorer::own`]), in front of the profile's run and
/// at any pool size, over a pool whose host lists its engines. The scorer
/// finds the request's rows with no pool read whole and asks every filter
/// of the profile about each row it reads, as a draw asks; its rows are
/// the targets in its order, the ones the profile's run would hand out,
/// and the trace names them best first. `None` where the scorer places
/// nothing: the profile's run decides, on the request's seed as it stands,
/// since the scorer reads no value of its stream.
///
/// Out of line, as [`run_engines`] is.
///
/// [`Scorer::own`]: super::Scorer::own
#[inline(never)]
pub(super) fn run_owned(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Decision> {
    let at = usize::from(index);
    let (profile, rows) = (&engine.profiles[at], engine.row_tests[at]);
    let stage = &profile.scorers.first()?.scorer;
    buf.reserve(candidates.len());
    buf.kept.clear();
    let kept = &mut buf.kept;
    let owned = match rows {
        Some(tests) if tests.masked(cx) => {
            let mut keep = |c: &Candidate| tests.keeps(facts, c, cx) && !cx.masks(c.position);
            stage.own(facts, pool, candidates, &mut keep, kept)
        }
        Some(tests) => {
            let mut keep = |c: &Candidate| tests.keeps(facts, c, cx);
            stage.own(facts, pool, candidates, &mut keep, kept)
        }
        None => {
            let mut keep = |c: &Candidate| profile.filters.iter().all(|f| f.keep(facts, c, cx));
            stage.own(facts, pool, candidates, &mut keep, kept)
        }
    };
    if !owned || kept.is_empty() {
        return None;
    }
    // The scorer's order is the targets': each row a step under the one
    // before it, as the scores of a ring's walk stand.
    let mut targets = Targets::new();
    kept.truncate(MAX_TARGETS);
    for &row in kept.iter() {
        targets.push(target_of(&candidates[row as usize]));
    }
    let steps = kept.iter().enumerate().map(|(place, &row)| {
        let step = 1.0 - place as f32 / MAX_TARGETS as f32;
        (candidates[row as usize].id, step)
    });
    let trace = trace_ordered(index, facts, pool, steps);
    Some(Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    }))
}

/// The decision of [`run_engines`] under one form of the filters' test, or
/// `None` where no row was kept. One body, built once for each form.
#[expect(
    clippy::too_many_arguments,
    reason = "the profile run's inputs, with the stream and the filters' test"
)]
#[inline(always)]
fn by_engines(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
    draw: &mut Draw,
    mut keep: impl FnMut(&Candidate) -> bool,
) -> Option<Decision> {
    let engines = &pool.engines;
    // The list names rows by their place among those handed.
    debug_assert!(candidates
        .iter()
        .enumerate()
        .all(|(place, row)| row.position as usize == place));
    let mut refused = 0;
    let first = if engines.len() > 1 {
        Some(stand_in(
            draw,
            engines,
            candidates,
            None,
            &mut keep,
            &mut refused,
        )?)
    } else {
        None
    };
    let second = first.and_then(|(engine, _)| {
        let but = Some(engine);
        stand_in(draw, engines, candidates, but, &mut keep, &mut refused)
    });
    let ties = Draw::new(draw.next_u64());
    let after = RequestFacts {
        seed: draw.state(),
        ..*facts
    };
    // The engine taken, its row already kept, and the other engine's row.
    let (taken, held, other) = match (first, second) {
        (None, _) => (0, None, None),
        (Some((engine, row)), None) => (engine, Some(row), None),
        (Some((a_engine, a)), Some((b_engine, b))) => {
            buf.kept.clear();
            buf.kept.extend([a, b]);
            let (kept, scores) = (&buf.kept, &mut buf.scores);
            let band = combine(profile, &after, candidates, kept, scores, &mut buf.stage);
            let mut two = Targets::new();
            let picker = &profile.picker;
            picker.order_drawn(&after, candidates, kept, scores, band, &mut two);
            match two.first() {
                Some(better) if better.position == b => (b_engine, Some(b), Some(a)),
                Some(_) => (a_engine, Some(a), Some(b)),
                None => return None,
            }
        }
    };
    buf.kept.clear();
    for &row in engines.rows(taken as usize) {
        if held == Some(row) || candidates.get(row as usize).is_some_and(&mut keep) {
            buf.kept.push(row);
        }
    }
    if buf.kept.is_empty() {
        return None;
    }
    let (kept, scores) = (&buf.kept, &mut buf.scores);
    let band = combine(profile, &after, candidates, kept, scores, &mut buf.stage);
    // The slots the answer carries, and no more: the taken engine's least
    // row, and its next where no other engine's row was drawn. The trace
    // names the taken engine's rows as they are handed out; the other
    // engine's row was costed beside another and is not named.
    let other = other.and_then(|row| candidates.get(row as usize));
    let mut best = [0; ENGINE_TARGETS];
    let slots = ENGINE_TARGETS - usize::from(other.is_some());
    let filled = draw_best(ties, scores, band, &mut best[..slots]);
    let best = &best[..filled];
    let mut targets = Targets::new();
    for &at in best {
        targets.push(target_of(&candidates[kept[at] as usize]));
    }
    if let Some(other) = other {
        targets.push(target_of(other));
    }
    let handed = best
        .iter()
        .map(|&at| (candidates[kept[at] as usize].id, scores[at]));
    let trace = trace_ordered(index, facts, pool, handed);
    Some(Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    }))
}

/// One engine drawn among the pool's, or among those but the engine `but`,
/// and the row that stands for it: its one row, or one of its rows by one
/// more value of the stream. A row `keep` refuses is drawn again with its
/// engine, each refusal counted into `refused`. `None` once [`DRAW_BOUND`]
/// rows were refused, in the decision and not in this draw alone, and
/// where no engine is left to draw.
fn stand_in(
    draw: &mut Draw,
    engines: &Engines<'_>,
    candidates: &[Candidate],
    but: Option<u32>,
    keep: &mut impl FnMut(&Candidate) -> bool,
    refused: &mut u32,
) -> Option<(u32, u32)> {
    let left = engines.len() - usize::from(but.is_some());
    if left == 0 {
        return None;
    }
    while *refused < DRAW_BOUND {
        let mut engine = draw.below(left) as u32;
        if but.is_some_and(|taken| engine >= taken) {
            engine += 1;
        }
        let row = match engines.rows(engine as usize) {
            [] => None,
            [row] => Some(*row),
            rows => rows.get(draw.below(rows.len())).copied(),
        };
        if let Some(row) = row {
            if candidates.get(row as usize).is_some_and(&mut *keep) {
                return Some((engine, row));
            }
        }
        *refused += 1;
    }
    None
}

#[cfg(test)]
mod tests;
