//! The pickers written for no one policy: each picks a decision's targets
//! among the rows the filters kept. None allocates per decision.
//!
//! The draw helpers of this file (`draw_kept`, `push_drawn`, `target_of`,
//! `below_band`) also serve the pickers of the named policies.

use crate::{
    document::spec::Needs,
    engine::{
        decision::{Target, Targets, MAX_TARGETS},
        framework::{top_drawn, top_pass, Picker, Shortlist, TieBand, DRAW_BOUND},
        view::{Candidate, PoolFacts, RequestFacts},
    },
    stages::lifecycle::Lifecycle,
    Draw,
};

/// The best score first, ties drawn uniformly from the decision's seed, then
/// the next best as alternates.
#[derive(Debug, Default)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct Best;

impl Lifecycle for Best {}

/// Whether `score` lies below the band of `top`, the best score it is
/// compared with. It is the one test of a tie: a score that is not below the
/// band is drawn with the best.
pub(crate) fn below_band(score: f32, top: f32, band: TieBand) -> bool {
    score < top - band.0
}

/// One slot of a ranked pick, over the rows `open` admits: the best score
/// among them is found first, the rows within the band of it are counted,
/// and one of them is drawn: one value of `draw` a slot, and none where a
/// row stands alone in its band. The band lies around the best score, so
/// the rows of a tie are the same whatever order they are listed in.
fn draw_in_band(
    scores: &[f32],
    band: TieBand,
    draw: &mut Draw,
    open: impl Fn(usize) -> bool,
) -> Option<usize> {
    let mut top = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if open(i) && s > top {
            top = s;
        }
    }
    let tied = |&(i, &s): &(usize, &f32)| open(i) && !below_band(s, top, band);
    let rows = scores.iter().enumerate().filter(tied).count();
    // Counted from the band's last row: of two tied rows a zero takes the
    // later one.
    let back = if rows > 1 { draw.below(rows) } else { 0 };
    let at = rows.checked_sub(1 + back)?;
    scores
        .iter()
        .enumerate()
        .filter(tied)
        .nth(at)
        .map(|(i, _)| i)
}

/// The slots a flat band fills, by rank. Where every row within the band of
/// `top`, the best score left, scores `top` itself, a row taken leaves the
/// band's best where it was: the band is counted once, each slot draws its
/// row among the band's rows left as [`draw_in_band`] would (one value of
/// `draw` while two rows are left, none for the last), and one more pass
/// finds the rows drawn, or none where every row is of the band: a rank is
/// its row there. Returns how many of `picks` it filled, in slot order:
/// none where the band is not flat, and the slots it leaves are drawn one
/// by one.
///
/// Inlined into the ranked push's two bodies, a pass's and a draw's, as it
/// was into the one body before a draw had its own.
#[inline(always)]
fn draw_flat(
    draw: &mut Draw,
    scores: &[f32],
    top: f32,
    band: TieBand,
    picks: &mut [usize],
) -> usize {
    // No row scores above `top` but the rows taken before the band.
    let mut rows = 0usize;
    for &s in scores {
        if s > top || below_band(s, top, band) {
            continue;
        }
        if s != top {
            return 0;
        }
        rows += 1;
    }
    let fills = picks.len().min(rows);
    // A slot's rank among the band's rows left, counted from the last as a
    // slot's draw counts it, becomes its rank among all of them: past each
    // row taken at or before it. `ranks` ascends, and `slots` names the slot
    // of each.
    let mut ranks = [usize::MAX; MAX_TARGETS];
    let mut slots = [0usize; MAX_TARGETS];
    for slot in 0..fills {
        let left = rows - slot;
        let back = if left > 1 { draw.below(left) } else { 0 };
        let mut rank = left - 1 - back;
        let mut at = 0;
        while at < slot && ranks[at] <= rank {
            rank += 1;
            at += 1;
        }
        ranks.copy_within(at..slot, at + 1);
        slots.copy_within(at..slot, at + 1);
        ranks[at] = rank;
        slots[at] = slot;
    }
    // Where every row is of the band, a row's rank is its place.
    if rows == scores.len() {
        for (&rank, &slot) in ranks.iter().zip(&slots).take(fills) {
            picks[slot] = rank;
        }
        return fills;
    }
    let (mut next, mut seen) = (0usize, 0usize);
    for (i, &s) in scores.iter().enumerate() {
        if next == fills {
            break;
        }
        if s == top {
            if seen == ranks[next] {
                picks[slots[next]] = i;
                next += 1;
            }
            seen += 1;
        }
    }
    fills
}

/// Whether the entry at `at` of the five best stands alone in its band: no
/// entry stands behind it, or the next one lies below its band.
fn alone(
    best_five: &[(usize, f32); MAX_TARGETS + 1],
    len: usize,
    at: usize,
    band: TieBand,
) -> bool {
    at + 1 >= len || below_band(best_five[at + 1].1, best_five[at].1, band)
}

/// Whether the five best stand for a pool: no score is a NaN (one ties in
/// every band, and the pass that keeps the five may have dropped it) and the
/// band is a width of zero or more (under a band below zero a row lies below
/// its own band). One read of the scores.
fn five_stand(scores: &[f32], band: TieBand) -> bool {
    band.0 >= 0.0 && !scores.iter().fold(false, |nan, s| nan | s.is_nan())
}

/// The rows of a ranked pick, by their places in `scores`, handed to `pick`
/// in slot order: `slots` of them at most, and `slots` no more than a
/// target list holds or `scores` holds. Per slot the rows within the band
/// of the best remaining score are one tie, drawn from uniformly with the
/// values of `draw`: one value a slot while two rows of its band are left,
/// none for a row alone, the pick counted from the band's last row in scan
/// order ([`draw_in_band`]).
///
/// One pass keeps the five best scores (equal scores in scan order), and a
/// band that ends among them is read off them: an entry below the band
/// stands behind its rows, or the five are every row. A row alone in its
/// band is its slot's pick. A band of several rows that score alike is
/// drawn slot by slot among its rows left, and once it is spent the entry
/// behind it is the best score left. So a pool whose bands end among the
/// five costs that pass and, at its first band of several rows, the read
/// below: no slot costs a pass of its own.
///
/// The five stand for the pool only where no score is a NaN (one ties in
/// every band, and the pass may have dropped it) and the band is a width of
/// zero or more (under a band below zero a row lies below its own band): one
/// read of the scores, made at the first band of several rows. Where that
/// fails, where a band reaches the fifth entry (rows behind the five may
/// stand in it) and where a band's rows differ within it, the slots left
/// are read over the pool: a flat band is counted once and drawn by rank
/// ([`draw_flat`]), and each slot it leaves is drawn as [`draw_in_band`]
/// draws it ([`draw_left`]).
///
/// `DRAWN` is the order of a draw: the five best are read with no branch on
/// a score ([`top_drawn`]). Inlined into the ranked push's two bodies, where
/// `pick` pushes a row's target: the rows drawn live in a fixed array and
/// nothing is allocated.
#[inline(always)]
fn draw_ranked<const DRAWN: bool>(
    draw: &mut Draw,
    scores: &[f32],
    band: TieBand,
    slots: usize,
    mut pick: impl FnMut(usize),
) {
    // Past either bound a slot would be handed a place no row stands at.
    // The push asks for its list's room over the kept rows, so a debug
    // build alone asserts it.
    debug_assert!(
        slots <= MAX_TARGETS.min(scores.len()),
        "a ranked draw fills no more slots than its list and its scores hold"
    );
    let mut best_five = [(usize::MAX, f32::NEG_INFINITY); MAX_TARGETS + 1];
    let len = if DRAWN {
        top_drawn(scores, &mut best_five)
    } else {
        top_pass(scores, &mut best_five)
    };
    let mut taken = [usize::MAX; MAX_TARGETS];
    let mut filled = 0usize;
    while filled < slots && alone(&best_five, len, filled, band) {
        let (row, _) = best_five[filled];
        taken[filled] = row;
        pick(row);
        filled += 1;
    }
    // Whether the five stand for the pool: unread until a band asks.
    let mut sound = None;
    // Several rows share the band of the best score left, the next entry
    // among them.
    while filled < slots {
        let top = best_five[filled].1;
        // Rows behind the five may stand in a band that reaches the fifth
        // entry.
        if len > MAX_TARGETS && !below_band(best_five[MAX_TARGETS].1, top, band) {
            break;
        }
        let mut end = filled + 2;
        while end < len && !below_band(best_five[end].1, top, band) {
            end += 1;
        }
        // A row that differs within the band moves the band as it is taken.
        if best_five[filled + 1..end]
            .iter()
            .any(|&(_, score)| score != top)
        {
            break;
        }
        if sound.is_none() {
            let stands = five_stand(scores, band);
            sound = Some(stands);
            if !stands {
                break;
            }
        }
        // The band's rows left, in scan order: a slot's draw counts from
        // the last of them.
        let mut rows = [usize::MAX; MAX_TARGETS];
        for (row, &(i, _)) in rows.iter_mut().zip(&best_five[filled..end]) {
            *row = i;
        }
        let mut left = end - filled;
        while left > 0 && filled < slots {
            let back = if left > 1 { draw.below(left) } else { 0 };
            let at = left - 1 - back;
            taken[filled] = rows[at];
            pick(rows[at]);
            // The rows behind it close the gap.
            for gap in at..MAX_TARGETS - 1 {
                rows[gap] = rows[gap + 1];
            }
            filled += 1;
            left -= 1;
        }
        // The rows alone behind the band.
        while filled < slots && alone(&best_five, len, filled, band) {
            let (row, _) = best_five[filled];
            taken[filled] = row;
            pick(row);
            filled += 1;
        }
    }
    if filled < slots {
        let top = best_five[filled].1;
        let fills = draw_flat(draw, scores, top, band, &mut taken[filled..slots]);
        for &row in &taken[filled..filled + fills] {
            pick(row);
        }
        filled += fills;
    }
    if filled < slots {
        let end = draw_left(
            draw,
            scores,
            band,
            &best_five[..len],
            sound,
            &mut taken[..slots],
            filled,
        );
        for &row in &taken[filled..end] {
            pick(row);
        }
    }
}

/// The slots from `filled` on of a ranked pick, as many as `taken` holds,
/// each drawn as [`draw_in_band`] draws it: the same row for the same values
/// of `draw`. `taken` holds the rows of the slots before `filled` and takes
/// the rows drawn here. Returns the end of the slots filled.
///
/// In a pool the five best stand for ([`five_stand`]: `sound`, read here
/// where no band of the five asked) the best score left is the
/// first of `best`, the five, whose row is not taken. The rows within its
/// band are counted over the whole pool, once while that score stands, and
/// one pass finds the row drawn. In any other pool a slot costs the passes
/// of [`draw_in_band`], behind that one read where the band is zero or more.
///
/// Out of line: the ranked push's two bodies hold none of these passes, and
/// a pick that leaves no such slot pays nothing for them.
#[inline(never)]
fn draw_left(
    draw: &mut Draw,
    scores: &[f32],
    band: TieBand,
    best: &[(usize, f32)],
    sound: Option<bool>,
    taken: &mut [usize],
    filled: usize,
) -> usize {
    let sound = sound.unwrap_or_else(|| five_stand(scores, band));
    // The best score the band was counted under last, and its rows over
    // the whole pool.
    let mut counted = (f32::NAN, 0usize);
    for slot in filled..taken.len() {
        let open = |i: usize| !taken[..slot].contains(&i);
        let row = if sound {
            // Fewer rows are taken than the five hold.
            let Some(&(_, top)) = best.iter().find(|&&(row, _)| open(row)) else {
                return slot;
            };
            if counted.0 != top {
                let within = scores.iter().filter(|&&s| !below_band(s, top, band));
                counted = (top, within.count());
            }
            // A row taken scores no less than the band's lower end, which
            // only falls: each of them is in the count.
            let rows = counted.1.saturating_sub(slot);
            let back = if rows > 1 { draw.below(rows) } else { 0 };
            let Some(at) = rows.checked_sub(1 + back) else {
                return slot;
            };
            let tied = |&(i, &s): &(usize, &f32)| !below_band(s, top, band) && open(i);
            scores
                .iter()
                .enumerate()
                .filter(tied)
                .nth(at)
                .map(|(i, _)| i)
        } else {
            draw_in_band(scores, band, draw, open)
        };
        let Some(row) = row else {
            return slot;
        };
        taken[slot] = row;
    }
    taken.len()
}

/// Push the kept positions in descending score order, ties by draw
/// ([`draw_ranked`]). The rows drawn live in a fixed array: nothing is
/// allocated.
fn push_ranked<const DRAWN: bool>(
    mut draw: Draw,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
    band: TieBand,
    out: &mut Targets,
) {
    let slots = MAX_TARGETS.min(kept.len());
    draw_ranked::<DRAWN>(&mut draw, scores, band, slots, |row| {
        out.push(target_of(&candidates[kept[row] as usize]));
    });
}

/// The best rows of `scores` into `out`, best first: as many as `out`
/// holds or the scores number. The order is the ranked push's, by its own
/// rule a slot ([`draw_in_band`]): the best score left is found, the rows
/// within the band of it stand equal, and one of them is drawn, by one
/// value of `draw` where they are several. Returns the slots filled.
///
/// For the rows of the engine a picker took ([`DrawUnit::Engine`]): a few rows
/// in no order of their scores, of which the answer carries fewer than a
/// ranked push fills. A slot's tie is marked in one word by two passes
/// over the rows. Where a later slot follows, a third pass notes the rows
/// that score the best itself, and the tie stands for that slot too while
/// one of them is left: the level ranks of one engine are read once. More
/// rows than a word marks are drawn by [`draw_in_band`] itself.
///
/// [`DrawUnit::Engine`]: crate::engine::framework::DrawUnit::Engine
pub(crate) fn draw_best(mut draw: Draw, scores: &[f32], band: TieBand, out: &mut [usize]) -> usize {
    let slots = out.len().min(scores.len());
    // A bit a row, set while the row is not taken.
    let word = u32::try_from(scores.len())
        .ok()
        .and_then(|rows| u64::BITS.checked_sub(rows))
        .and_then(|spare| u64::MAX.checked_shr(spare));
    let Some(mut open) = word else {
        for slot in 0..slots {
            let (taken, _) = out.split_at(slot);
            let Some(row) = draw_in_band(scores, band, &mut draw, |i| !taken.contains(&i)) else {
                return slot;
            };
            out[slot] = row;
        }
        return slots;
    };
    // The rows within the band of the best score left, and those of them
    // that score it.
    let (mut tied, mut at_best) = (0u64, 0u64);
    let mut filled = 0;
    let all = open;
    for (n, slot) in out[..slots].iter_mut().enumerate() {
        if at_best & open == 0 {
            // The best score left: over every row while none is taken.
            let mut top = f32::NEG_INFINITY;
            if open == all {
                for &score in scores {
                    if score > top {
                        top = score;
                    }
                }
            } else {
                for (row, &score) in scores.iter().enumerate() {
                    if (open >> row) & 1 != 0 && score > top {
                        top = score;
                    }
                }
            }
            (tied, at_best) = (0, 0);
            for (row, &score) in scores.iter().enumerate() {
                tied |= u64::from(!below_band(score, top, band)) << row;
            }
            // The rows that score the best itself spare a later slot its
            // passes: the last slot notes none.
            if n + 1 < slots {
                for (row, &score) in scores.iter().enumerate() {
                    at_best |= u64::from(score == top) << row;
                }
            }
        }
        tied &= open;
        // Counted from the tie's last row, as a slot of the ranked push
        // counts: of two tied rows a zero takes the later one.
        let rows = tied.count_ones() as usize;
        let back = if rows > 1 { draw.below(rows) } else { 0 };
        let Some(at) = rows.checked_sub(1 + back) else {
            return filled;
        };
        let mut from = tied;
        for _ in 0..at {
            from &= from - 1;
        }
        let row = from.trailing_zeros() as usize;
        open &= !(1 << row);
        *slot = row;
        filled += 1;
    }
    filled
}

/// The target that names `c`, on no pair ([`Target::of`]).
pub(crate) fn target_of(c: &Candidate) -> Target {
    Target::of(c)
}

/// The rows a draw left, handed out as it drew them.
pub(crate) fn push_drawn(candidates: &[Candidate], kept: &[u32], out: &mut Targets) {
    for &i in kept {
        out.push(target_of(&candidates[i as usize]));
    }
}

/// The targets of a rotation's turn over `kept`, which holds one row or
/// more: the row at `turn`, counted round the rows kept, then the rows kept
/// behind it in their order, wrapping, as far as a decision names targets.
pub(crate) fn push_turn(candidates: &[Candidate], kept: &[u32], turn: usize, out: &mut Targets) {
    for k in 0..MAX_TARGETS.min(kept.len()) {
        out.push(target_of(
            &candidates[kept[(turn + k) % kept.len()] as usize],
        ));
    }
}

/// One uniform draw over the rows of the pool `keep` keeps, or over those
/// but the row `except`: a row is drawn again while it is refused, and each
/// refusal counts into `refused`. `None` once [`DRAW_BOUND`] rows were
/// refused, in the decision and not in this draw alone.
pub(crate) fn draw_kept(
    draw: &mut Draw,
    candidates: &[Candidate],
    except: Option<usize>,
    keep: &mut dyn FnMut(&Candidate) -> bool,
    refused: &mut u32,
) -> Option<usize> {
    let rows = candidates.len() - usize::from(except.is_some());
    if rows == 0 {
        return None;
    }
    while *refused < DRAW_BOUND {
        let mut row = draw.below(rows);
        if except.is_some_and(|taken| row >= taken) {
            row += 1;
        }
        if keep(&candidates[row]) {
            return Some(row);
        }
        *refused += 1;
    }
    None
}

impl Picker for Best {
    fn name(&self) -> &'static str {
        "best"
    }

    // Picks the best score of the rows kept.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The targets are the best of the rows handed: a rank may put rows in
    /// front of this picker.
    fn front_wise(&self) -> bool {
        true
    }

    /// Every row handed is compared and the best scores are taken in
    /// their order: a scorer that places a request by itself is followed.
    fn takes_best(&self) -> bool {
        true
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        push_ranked::<false>(Draw::new(facts.seed), candidates, kept, scores, band, out);
    }
}

/// The best of a sample: `best` over the rows it draws first. Where the
/// host named the request's holders and the pool's warming rows
/// ([`AffinityRows`](crate::engine::AffinityRows)) and the pool holds more than
/// twice the sample, and at least the sixteen rows the engine asks a draw
/// from, it draws `sample` rows the filters keep, uniformly and with
/// replacement (a repeat or a refused row costs one of its `2 * sample`
/// draws), the named rows join them, and the highest combined score among
/// those wins. Any other pool is read whole, as `best` reads it, and so is
/// a pool whose sample holds no row.
#[derive(Debug)]
pub struct SampledBest {
    sample: u32,
}

impl SampledBest {
    /// The picker over a sample of `sample` rows, at least one.
    pub fn new(sample: u32) -> Self {
        Self {
            sample: sample.max(1),
        }
    }
}

impl Lifecycle for SampledBest {}

impl Picker for SampledBest {
    fn name(&self) -> &'static str {
        "best"
    }

    // Picks the best score of the rows it draws.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The targets are the best of the rows handed: a rank may put rows in
    /// front of this picker.
    fn front_wise(&self) -> bool {
        true
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        push_ranked::<false>(Draw::new(facts.seed), candidates, kept, scores, band, out);
    }

    /// The pick over the rows drawn and joined, which stand in the draw's
    /// order: their five best are read as a draw's.
    fn order_drawn(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        push_ranked::<true>(Draw::new(facts.seed), candidates, kept, scores, band, out);
    }

    fn draws_first(&self) -> bool {
        true
    }

    fn joined_by_named(&self) -> bool {
        true
    }

    /// At most `2 * sample` uniform draws until `sample` rows are kept; a
    /// repeat and a row the filters refuse are dropped, each at the cost
    /// of its draw. The draws are the picker's own bound: it counts no
    /// refused rows against [`DRAW_BOUND`], which is sized for a draw of
    /// one or two. A pool of at most twice the sample, and a sample that
    /// holds no row, leave the decision to the whole pass. Whether the
    /// host named rows is the engine's to ask
    /// ([`Picker::joined_by_named`]).
    fn draw_first(
        &self,
        _: &RequestFacts<'_>,
        _: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        draw: &mut Draw,
    ) -> bool {
        let sample = self.sample as usize;
        if candidates.len() <= 2 * sample {
            return false;
        }
        for _ in 0..2 * sample {
            if kept.len() == sample {
                break;
            }
            let row = draw.below(candidates.len()) as u32;
            if !kept.contains(&row) && keep(&candidates[row as usize]) {
                kept.push(row);
            }
        }
        !kept.is_empty()
    }

    fn needs(&self) -> Needs {
        Needs {
            sample: self.sample,
            ..Needs::default()
        }
    }
}

/// The mask of the stream count-first orders its alternates by: the order of
/// equal counts among the workers passed over is drawn from the decision's
/// seed under it. No other site derives that stream: the pick's ties read
/// the seed's own stream, and a pair plan's decode side opens the stream of
/// the seed's first value. [`Draw`] lists every stream a decision opens.
const ALTERNATES_STREAM: u64 = 0xa54f_f53a_5f1d_36f1;

/// The fewest requests in flight on the host first, then the profile's score
/// among equal counts as the `best` picker ranks it: the scores within the
/// profile's tie band of the best are one tie, drawn from the decision's
/// seed. No weight or band stands between the two stages: a strictly lower
/// count wins whatever the scores. The fewest in flight are the set
/// compared: the scorers are handed them alone, so a worker holding more
/// moves no input of a built-in's estimate. The workers passed over follow
/// as alternates by count, equal counts drawn from a stream of their own
/// (the seed under a mask no other site uses), and no scorer orders them. A
/// host that wants the estimate after a failed dispatch decides again with
/// its tried list: the `not-tried` filter leaves the rest, compared among
/// themselves.
#[derive(Debug)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct CountFirst;

impl Lifecycle for CountFirst {}

impl Picker for CountFirst {
    fn name(&self) -> &'static str {
        "count-first"
    }

    // Picks the fewest in flight: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The targets are the best of the rows handed: a rank may put rows in
    /// front of this picker.
    fn front_wise(&self) -> bool {
        true
    }

    fn narrow(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut [u32],
    ) -> Shortlist {
        let count = |row: u32| facts.in_flight(&candidates[row as usize]);
        let Some(fewest) = kept.iter().map(|&row| count(row)).min() else {
            return Shortlist::all(0);
        };
        let mut compared = 0usize;
        for i in 0..kept.len() {
            if count(kept[i]) == fewest {
                kept.swap(compared, i);
                compared += 1;
            }
        }
        // The alternates the target list still has room for once the
        // compared set is ordered: the next by count, equal counts drawn.
        let alternates = MAX_TARGETS
            .saturating_sub(compared)
            .min(kept.len() - compared);
        // The order of equal counts is drawn from a stream of its own, so it
        // says nothing of the tie among the fewest (the seed's stream, which
        // `pick` reads) nor of the draws a pair plan's decode side makes.
        let mut draw = Draw::new(facts.seed ^ ALTERNATES_STREAM);
        for slot in compared..compared + alternates {
            let mut next = slot;
            let mut tied = 1usize;
            for i in slot + 1..kept.len() {
                match count(kept[i]).cmp(&count(kept[next])) {
                    std::cmp::Ordering::Less => {
                        next = i;
                        tied = 1;
                    }
                    std::cmp::Ordering::Equal => {
                        tied += 1;
                        if draw.below(tied) == 0 {
                            next = i;
                        }
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
            kept.swap(slot, next);
        }
        Shortlist {
            compared,
            alternates,
        }
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        let count = |row: u32| facts.in_flight(&candidates[row as usize]);
        // The ties are drawn from the seed's own stream, as the best picker
        // draws them: the alternates' order has a stream of its own.
        let mut draw = Draw::new(facts.seed);
        // The narrowing leaves one count, and among equal counts the pick is
        // the best picker's.
        if kept.windows(2).all(|pair| count(pair[0]) == count(pair[1])) {
            return push_ranked::<false>(draw, candidates, kept, scores, band, out);
        }
        // Rows that were never narrowed: the same rule a slot, among the
        // fewest in flight of the rows left. Every row here came with a
        // score, so equal counts past the fewest go by score too, where the
        // narrowing draws them unscored.
        let mut taken = [usize::MAX; MAX_TARGETS];
        for slot in 0..MAX_TARGETS.min(kept.len()) {
            let left = |i: usize| !taken[..slot].contains(&i);
            let rows = kept.iter().enumerate();
            let fewest = rows
                .filter(|&(i, _)| left(i))
                .map(|(_, &row)| count(row))
                .min();
            let open = |i: usize| left(i) && Some(count(kept[i])) == fewest;
            let Some(b) = draw_in_band(scores, band, &mut draw, open) else {
                break;
            };
            taken[slot] = b;
            out.push(target_of(&candidates[kept[b] as usize]));
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/stages/pickers/tests.rs"]
mod tests;
