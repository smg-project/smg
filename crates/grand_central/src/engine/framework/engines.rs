//! The engine as a pick's unit. A pool of data-parallel engines holds a row
//! a rank, so a decision that reads its pool whole pays for every rank of
//! the fleet, and one that draws rows compares ranks of engines it knows
//! nothing else of. A picker whose unit is the engine ([`Picker::unit`])
//! decides over the engines the host lists beside the rows ([`Engines`]):
//! it draws two engines, compares one row of each, and reads the rows of
//! the engine it takes. The rows of an engine it passes by are not read, so
//! what the decision costs follows the ranks of one engine and not the
//! fleet. A draw that finds no two engines inside its bound decides
//! nothing, and the whole pass reads the pool.
//!
//! [`Picker::unit`]: super::Picker::unit

use super::{
    combine, keep_folded, places, run_single, score_survivors, trace_ordered, Buffers, Engine,
    Profile, Set, DRAW_BOUND,
};
use crate::{
    engine::{
        decision::{Decision, Pick, Targets, MAX_TARGETS},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Engines, PoolFacts, RequestFacts},
    },
    stages::pickers::{draw_best, target_of},
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
/// except a test the build folded that keeps every row under the
/// request's facts ([`RowTests::asked`]),
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
/// Where no row is kept, or where a pool of several engines yields one
/// engine's row and no second's, the draw says nothing of the pool: the
/// decision widens to the whole pass on the stream continued, which alone
/// sheds, names a stage and asks the all-vetoed rule. One engine standing
/// alone among several is such a pool: the whole pass compares two of its
/// rows, as the picker draws over rows, and does not seek its least rank.
/// The trace keeps the request's seed, from which the whole decision
/// replays. No engine is drawn for a request its host holds to places
/// ([`places`]): the staged pass decides it.
///
/// Out of line, so a decision of any other profile keeps the engine's
/// decision path as it was.
///
/// [`Picker::unit`]: super::Picker::unit
/// [`RowTests::asked`]: crate::stages::filters::RowTests::asked
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
    // The fold as this decision asks it: a test that keeps every row under
    // these facts is left out of the draw's rows too.
    let rows = rows.map(|tests| tests.asked(facts, cx));
    let decided = match rows {
        // A request held to places is the staged pass's, which opens its
        // set with them: no engine is drawn, and the pass below decides.
        _ if places::stated(cx) => Err(false),
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
    let stood = match decided {
        Ok(decision) => return decision,
        Err(stood) => stood,
    };
    let resumed = RequestFacts {
        seed: draw.state(),
        ..*facts
    };
    let mut decision = match rows {
        // A row the draw kept passes the fold, so the fold's one pass over
        // the pool keeps a row and leaves the rows the stages would
        // ([`keep_folded`]): the stages are not asked one by one. Under a
        // standing veto they are: the fold asks its tests of a row before
        // the host's mask, the stages behind it.
        Some(tests)
            if stood
                && set.plain()
                && !tests.masked(cx)
                && keep_folded(tests, &resumed, candidates, cx, &mut buf.kept) =>
        {
            score_survivors(profile, index, &resumed, pool, candidates, buf, None)
        }
        _ => run_single(profile, index, &resumed, pool, set, cx, buf),
    };
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
/// nothing, and for a request its host holds to places ([`places`]): the
/// profile's run decides, on the request's seed as it stands, since the
/// scorer reads no value of its stream.
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
    // A request held to places is the profile's run's, which opens its
    // set with them: the scorer's walk would name rows of its own.
    if places::stated(cx) {
        return None;
    }
    let at = usize::from(index);
    let (profile, rows) = (&engine.profiles[at], engine.row_tests[at]);
    let weighted = profile.scorers.first()?;
    let stage = &weighted.scorer;
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
    // before it, as the scores of a ring's walk stand, by the stage's
    // share of the profile's weights. The whole pass folds that share
    // into a score, and a weight's share of itself need not be one.
    let mut targets = Targets::new();
    kept.truncate(MAX_TARGETS);
    for &row in kept.iter() {
        targets.push(target_of(&candidates[row as usize]));
    }
    let share = weighted.sole_share();
    let steps = kept.iter().enumerate().map(|(place, &row)| {
        let step = 1.0 - place as f32 / MAX_TARGETS as f32;
        (candidates[row as usize].id, share * step)
    });
    let trace = trace_ordered(index, facts, pool, steps);
    Some(Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    }))
}

/// The decision of [`run_engines`] under one form of the filters' test, or
/// `Err` where the draw leaves the pool to the whole pass: with `true`
/// where a row it drew stood. One body, built once for each form.
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
) -> Result<Decision, bool> {
    let engines = &pool.engines;
    // The list names rows by their place among those handed.
    debug_assert!(candidates
        .iter()
        .enumerate()
        .all(|(place, row)| row.position as usize == place));
    let mut refused = 0;
    let first = if engines.len() > 1 {
        let first = stand_in(draw, engines, candidates, None, &mut keep, &mut refused);
        Some(first.ok_or(false)?)
    } else {
        None
    };
    // One row compared with none says nothing of the pool: where the
    // second engine is not found the whole pass decides, as it does for
    // a draw over rows.
    let drawn = match first {
        Some(a) => {
            let but = Some(a.0);
            let second = stand_in(draw, engines, candidates, but, &mut keep, &mut refused);
            Some((a, second.ok_or(true)?))
        }
        None => None,
    };
    let ties = Draw::new(draw.next_u64());
    let after = RequestFacts {
        seed: draw.state(),
        ..*facts
    };
    // The engine taken, its row already kept, and the other engine's row.
    let (taken, held, other) = match drawn {
        None => (0, None, None),
        Some(((a_engine, a), (b_engine, b))) => {
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
                None => return Err(true),
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
        return Err(false);
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
    Ok(Decision::Pick(Pick {
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
