//! The `power_of_two` policy: how it is composed, its four forms, and the
//! pickers only it uses.

use crate::{
    document::spec::{PolicySpec, ProfileSpec, ResidentWorkParams, ScoreStage, SpecError},
    engine::{
        decision::Targets,
        framework::{DrawUnit, Picker, Shortlist, TieBand},
        view::{Candidate, PoolFacts, RequestFacts},
    },
    policies::standard_filters,
    stages::{
        lifecycle::Lifecycle,
        pickers::{below_band, draw_kept, target_of},
    },
    Draw,
};

/// How the `power_of_two` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    // The expected-wait form: two draws compared by least-load.
    ProfileSpec {
        filter: standard_filters(),
        score: vec!["least-load".into()],
        pick: "power-of-two".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The load metric the `power_of_two` preset compares its candidates by:
/// one form of the preset each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PowerOfTwoMetric {
    /// Two draws, the lower expected wait: the preset under its bare name.
    ExpectedWait,
    /// Two draws, the fewer requests in flight, the expected wait between
    /// equal counts.
    Requests,
    /// The fewest requests in flight over every candidate, the expected wait
    /// between equal counts.
    LeastRequests,
    /// The least resident work over every candidate, each request priced at
    /// `request_tokens` context tokens.
    ResidentTokens { request_tokens: u32 },
}

impl PolicySpec {
    /// The `power_of_two` preset in the form its load metric names, bound to
    /// every model and phase: the count forms compare the requests in flight
    /// before the score through the count-first pickers, and the resident
    /// form scores every candidate and takes the best, with no draw of two.
    pub fn power_of_two(metric: PowerOfTwoMetric) -> Result<Self, SpecError> {
        const NAME: &str = "power_of_two";
        let mut spec = Self::preset(NAME)?;
        let profile = spec.profiles.get_mut(NAME).ok_or(SpecError::Empty)?;
        match metric {
            PowerOfTwoMetric::ExpectedWait => {}
            PowerOfTwoMetric::Requests => profile.pick = "power-of-two-count-first".into(),
            PowerOfTwoMetric::LeastRequests => profile.pick = "count-first".into(),
            PowerOfTwoMetric::ResidentTokens { request_tokens } => {
                profile.score = vec![ScoreStage::ResidentWork(ResidentWorkParams {
                    weight: 1.0,
                    request_tokens,
                })
                .into()];
                profile.pick = "best".into();
            }
        }
        Ok(spec)
    }
}

/// The two rows the two-draw pickers compare, drawn first: two distinct
/// uniform draws over the rows the whole pass keeps, each drawn again while
/// its row is refused. `false` once [`DRAW_BOUND`] rows were refused before
/// both were held, a pool with one row kept among them: the whole pass
/// decides.
///
/// [`DRAW_BOUND`]: crate::engine::framework::DRAW_BOUND
fn draw_two_kept(
    draw: &mut Draw,
    candidates: &[Candidate],
    keep: &mut dyn FnMut(&Candidate) -> bool,
    kept: &mut Vec<u32>,
) -> bool {
    let mut refused = 0;
    let Some(first) = draw_kept(draw, candidates, None, keep, &mut refused) else {
        return false;
    };
    let Some(second) = draw_kept(draw, candidates, Some(first), keep, &mut refused) else {
        return false;
    };
    kept.extend([first as u32, second as u32]);
    true
}

/// Two distinct uniform positions below `n` (two at least), from the next two
/// values of `draw`.
fn two_below(draw: &mut Draw, n: usize) -> (usize, usize) {
    let a = draw.below(n);
    let mut b = draw.below(n - 1);
    if b >= a {
        b += 1;
    }
    (a, b)
}

/// The narrowing of the two-draw pickers: the two rows the first two values
/// of the seed's stream draw move to the front of `kept`. The number compared
/// comes back: two, or the whole of a set that holds no more, where nothing
/// is passed over and nothing is drawn.
fn draw_two(seed: u64, kept: &mut [u32]) -> usize {
    let n = kept.len();
    if n <= 2 {
        return n;
    }
    let (a, b) = two_below(&mut Draw::new(seed), n);
    kept.swap(0, a);
    // The second draw's row moved with the swap if it led the list.
    let b = if b == 0 { a } else { b };
    kept.swap(1, b);
    2
}

/// `a` before `b` by score, the higher first, for a pick between two rows.
/// The lower of the two is equal to the higher unless it lies below its
/// band: the band is the scorers' word on what they cannot tell apart, and a
/// picker that compares scores reads it. It is the order of a pair alone.
/// Walked over more rows against a running best it would chain ties by the
/// order of the walk, so a pick over more rows draws within the band of the
/// best (`draw_in_band`).
fn score_order(scores: &[f32], band: TieBand, a: usize, b: usize) -> std::cmp::Ordering {
    let order = scores[b].total_cmp(&scores[a]);
    let (high, low) = if order.is_gt() {
        (scores[b], scores[a])
    } else {
        (scores[a], scores[b])
    };
    if below_band(low, high, band) {
        order
    } else {
        std::cmp::Ordering::Equal
    }
}

/// The pick of the two-draw pickers over `kept`: the two rows the first two
/// values of the seed's stream draw, the one `order` puts first leading. Of
/// two rows the draw takes both, so a pair the narrowing left and rows that
/// were never narrowed (a plugin picker's fallback is handed every survivor)
/// go through one rule and one stream, each value deciding once. A tie goes
/// by the stream's third value, between the two rows as the host lists them:
/// an even draw wherever the rows sit, the same for a narrowed pair and not.
fn push_two(
    seed: u64,
    candidates: &[Candidate],
    kept: &[u32],
    order: impl Fn(usize, usize) -> std::cmp::Ordering,
    out: &mut Targets,
) {
    let (first, second) = match kept.len() {
        0 => return,
        1 => return out.push(target_of(&candidates[kept[0] as usize])),
        n => {
            let mut draw = Draw::new(seed);
            let (a, b) = two_below(&mut draw, n);
            match order(a, b) {
                std::cmp::Ordering::Less => (a, b),
                std::cmp::Ordering::Greater => (b, a),
                std::cmp::Ordering::Equal => {
                    let (low, high) = if kept[a] < kept[b] { (a, b) } else { (b, a) };
                    if draw.below(2) == 0 {
                        (low, high)
                    } else {
                        (high, low)
                    }
                }
            }
        }
    };
    out.push(target_of(&candidates[kept[first] as usize]));
    out.push(target_of(&candidates[kept[second] as usize]));
}

/// Two uniform draws, the better score taken; the other is the alternate.
/// Scores within the profile's tie band are a tie, drawn. The profile's
/// scorers decide which of the two is better, so a profile with this picker
/// names at least one scorer. The two drawn are the set compared: the
/// scorers are handed them alone, so a worker that was not drawn moves no
/// input of a built-in's estimate.
///
/// Between two workers a score normalized over them is all or nothing, so
/// several weighted scorers under this picker vote, each with its whole
/// weight, where over the pool they would blend. A profile that wants the
/// blend names one scorer here, or the `best` picker over the pool.
#[derive(Debug)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct PowerOfTwo;

impl Lifecycle for PowerOfTwo {}

impl Picker for PowerOfTwo {
    fn name(&self) -> &'static str {
        "power-of-two"
    }

    // Picks the better score of two draws.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn draws_first(&self) -> bool {
        true
    }

    /// Two engines, where the host lists them: the two rows that stand for
    /// them are compared as two rows drawn are.
    fn unit(&self) -> DrawUnit {
        DrawUnit::Engine
    }

    fn draw_first(
        &self,
        _: &RequestFacts<'_>,
        _: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        draw: &mut Draw,
    ) -> bool {
        draw_two_kept(draw, candidates, keep, kept)
    }

    fn narrow(&self, facts: &RequestFacts<'_>, _: &[Candidate], kept: &mut [u32]) -> Shortlist {
        Shortlist {
            compared: draw_two(facts.seed, kept),
            alternates: 0,
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
        let by_score = |a: usize, b: usize| score_order(scores, band, a, b);
        push_two(facts.seed, candidates, kept, by_score, out);
    }
}

/// Two uniform draws, the one with fewer requests in flight taken and the
/// profile's score deciding equal counts, scores within the tie band a tie
/// and drawn; the other is the alternate. The two drawn are the set
/// compared, as under [`PowerOfTwo`], and several weighted scorers vote
/// between them as they do there.
#[derive(Debug)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct PowerOfTwoCountFirst;

impl Lifecycle for PowerOfTwoCountFirst {}

impl Picker for PowerOfTwoCountFirst {
    fn name(&self) -> &'static str {
        "power-of-two-count-first"
    }

    // Picks the fewer in flight of two draws: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn draws_first(&self) -> bool {
        true
    }

    fn draw_first(
        &self,
        _: &RequestFacts<'_>,
        _: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        draw: &mut Draw,
    ) -> bool {
        draw_two_kept(draw, candidates, keep, kept)
    }

    fn narrow(&self, facts: &RequestFacts<'_>, _: &[Candidate], kept: &mut [u32]) -> Shortlist {
        Shortlist {
            compared: draw_two(facts.seed, kept),
            alternates: 0,
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
        let count = |i: usize| facts.in_flight(&candidates[kept[i] as usize]);
        let by_count = |a: usize, b: usize| {
            count(a)
                .cmp(&count(b))
                .then_with(|| score_order(scores, band, a, b))
        };
        push_two(facts.seed, candidates, kept, by_count, out);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/power_of_two/tests.rs"]
mod tests;
