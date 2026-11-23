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
mod tests {
    use super::*;
    use crate::{
        document::spec::ScoreSpec,
        engine::framework::Scorer,
        stages::{
            pickers::CountFirst,
            scorers::{
                tests::{candidate, kept, picked, picked_within},
                LeastRequests,
            },
        },
    };

    #[test]
    fn power_of_two_takes_the_better_scored_of_two_draws() {
        let pool = [candidate(0, 9), candidate(1, 9), candidate(2, 0)];
        let least_requests = |rows: &[u32]| {
            let mut scores = vec![0.0f32; rows.len()];
            LeastRequests.score(&RequestFacts::default(), &pool, rows, &mut scores);
            scores
        };
        let mut hits = 0;
        for seed in 0..100u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let order = picked(&PowerOfTwo, &facts, &pool, &least_requests);
            assert_eq!(order.len(), 2);
            if order[0] == 2 {
                hits += 1;
            }
        }
        // Worker 2 is drawn in two of three pairs and wins each time.
        assert!((50..=85).contains(&hits), "{hits}");
    }

    /// The review's case: counts 0, 1 and 2000 with the idle worker's wait
    /// the worst of the three. The count decides before any score: the idle
    /// worker leads, the next by count follows; a drawn pair is led by its
    /// fewer in flight, so the busiest worker never leads one.
    #[test]
    fn count_first_takes_the_fewest_in_flight_before_any_score() {
        let pool = [candidate(0, 0), candidate(1, 1), candidate(2, 2000)];
        let by_position = |rows: &[u32]| -> Vec<f32> {
            rows.iter()
                .map(|&row| [0.0f32, 1.0, 0.49][row as usize])
                .collect()
        };
        for seed in 0..20u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let order = picked(&CountFirst, &facts, &pool, &by_position);
            assert_eq!(order, [0, 1, 2], "seed {seed}");
            let drawn = picked(&PowerOfTwoCountFirst, &facts, &pool, &by_position);
            assert_eq!(drawn.len(), 2, "seed {seed}");
            assert!(
                pool[drawn[0] as usize].in_flight <= pool[drawn[1] as usize].in_flight,
                "seed {seed}: {drawn:?}"
            );
            assert_ne!(drawn[0], 2, "seed {seed}: the busiest worker never leads");
        }
    }

    /// A drawn pair of equal scores is led by an even draw: the third value
    /// of the seed's stream, between the two rows as the pool lists them.
    /// Over many seeds every worker leads as often as another, wherever it
    /// sits in the pool. (The stream's first value read again over the two
    /// rows would favor a row by where it sits.)
    #[test]
    fn a_tie_of_two_draws_is_broken_evenly_wherever_the_rows_sit() {
        let pool = [
            candidate(0, 1),
            candidate(1, 1),
            candidate(2, 1),
            candidate(3, 1),
        ];
        let level = |rows: &[u32]| vec![0.5f32; rows.len()];
        for picker in [&PowerOfTwo as &dyn Picker, &PowerOfTwoCountFirst] {
            let mut leads = [0u32; 4];
            for seed in 0..4_000u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let order = picked(picker, &facts, &pool, &level);
                let mut stream = Draw::new(seed);
                let a = stream.below(4) as u32;
                let mut b = stream.below(3) as u32;
                if b >= a {
                    b += 1;
                }
                let lead = if stream.below(2) == 0 {
                    a.min(b)
                } else {
                    a.max(b)
                };
                assert_eq!(order, [lead, a + b - lead], "seed {seed}");
                leads[order[0] as usize] += 1;
            }
            assert!(
                leads.iter().all(|n| (850..=1150).contains(n)),
                "{}: {leads:?}",
                picker.name()
            );
        }
    }

    /// The band is the scorers' word on what they cannot tell apart, and
    /// every picker that compares scores reads it: two workers of one count
    /// whose scores lie within the band are a tie and drawn evenly, and two
    /// further apart are not.
    #[test]
    fn scores_within_the_band_are_a_tie_for_every_picker_that_compares_them() {
        let pair = [candidate(0, 2), candidate(1, 2)];
        let near = |rows: &[u32]| -> Vec<f32> {
            rows.iter()
                .map(|&row| [0.50f32, 0.56][row as usize])
                .collect()
        };
        for picker in [
            &PowerOfTwo as &dyn Picker,
            &PowerOfTwoCountFirst,
            &CountFirst,
        ] {
            let mut leads = [0u32; 2];
            for seed in 0..2_000u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let within = picked_within(picker, &facts, &pair, &near, TieBand(0.1));
                leads[within[0] as usize] += 1;
                let apart = picked_within(picker, &facts, &pair, &near, TieBand(0.05));
                assert_eq!(
                    apart,
                    [1, 0],
                    "{}, seed {seed}: the higher score",
                    picker.name()
                );
            }
            assert!(
                leads.iter().all(|n| (850..=1150).contains(n)),
                "{}: {leads:?}",
                picker.name()
            );
        }
    }

    /// A two-draw pick handed rows that were never narrowed draws its own
    /// two, the two its narrowing would have put in front. A plugin picker's
    /// fallback is handed every survivor with their scores: it picks as the
    /// engine's order does over the same scores.
    #[test]
    fn a_two_draw_pick_over_rows_that_were_not_narrowed_draws_its_own_two() {
        let pool = [
            candidate(0, 4),
            candidate(1, 2),
            candidate(2, 4),
            candidate(3, 2),
            candidate(4, 3),
        ];
        // The scores tie two by two, so the tie's draw is in the case.
        let scores = [0.3f32, 0.3, 0.7, 0.7, 0.5];
        let by_position =
            |rows: &[u32]| -> Vec<f32> { rows.iter().map(|&row| scores[row as usize]).collect() };
        let mut leads = std::collections::BTreeSet::new();
        for picker in [&PowerOfTwo as &dyn Picker, &PowerOfTwoCountFirst] {
            for seed in 0..200u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let mut out = Targets::new();
                picker.pick(
                    &facts,
                    &pool,
                    &kept(5),
                    &scores,
                    TieBand::NORMALIZED,
                    &mut out,
                );
                let whole: Vec<u32> = out.iter().map(|t| t.position).collect();
                assert_eq!(
                    whole,
                    picked(picker, &facts, &pool, &by_position),
                    "{}, seed {seed}",
                    picker.name()
                );
                leads.insert(whole[0]);
            }
        }
        assert_eq!(leads.len(), 5, "the draw reaches every row: {leads:?}");
    }

    #[test]
    fn the_power_of_two_preset_has_four_forms_by_its_load_metric() {
        let form = |metric| {
            let spec = PolicySpec::power_of_two(metric).expect("a preset");
            let engine = spec.validate().expect("validates");
            assert!(engine.needs().loads, "{metric:?}");
            let profile = spec.profiles["power_of_two"].clone();
            (profile.score, profile.pick.name().to_string())
        };
        let least_load: Vec<ScoreSpec> = vec!["least-load".into()];
        assert_eq!(
            form(PowerOfTwoMetric::ExpectedWait),
            (least_load.clone(), "power-of-two".to_string())
        );
        assert_eq!(
            form(PowerOfTwoMetric::Requests),
            (least_load.clone(), "power-of-two-count-first".to_string())
        );
        assert_eq!(
            form(PowerOfTwoMetric::LeastRequests),
            (least_load, "count-first".to_string())
        );
        let resident: Vec<ScoreSpec> = vec![ScoreStage::ResidentWork(ResidentWorkParams {
            weight: 1.0,
            request_tokens: 2048,
        })
        .into()];
        assert_eq!(
            form(PowerOfTwoMetric::ResidentTokens {
                request_tokens: 2048
            }),
            (resident, "best".to_string())
        );
        // A request priced at nothing is a form too: the tokens alone.
        PolicySpec::power_of_two(PowerOfTwoMetric::ResidentTokens { request_tokens: 0 })
            .expect("a preset")
            .validate()
            .expect("the tokens alone");
    }
}
