use super::*;

/// A decision over a pool whose host says that every row holds a
/// report ([`PoolFacts::reported`]): what a profile is asked by rank
/// under.
fn reported(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
    let said = PoolFacts {
        reported: true,
        ..PoolFacts::default()
    };
    engine.decide(
        facts,
        &said,
        pool,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// A rank of the tests: a row reads the level named for its position,
/// whichever rows it is asked of.
/// Where it `says`, it says its spread and its front, as a rank does;
/// where it does not it is a stage of any values, and its profile's
/// decision is the weighted sum's.
#[derive(Debug)]
struct Levelled {
    levels: Vec<f32>,
    step: f32,
    says: bool,
}

impl Lifecycle for Levelled {}

impl Scorer for Levelled {
    fn name(&self) -> &'static str {
        "levelled"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.score_front(facts, candidates, kept, kept, out).0
    }

    fn spread(&self) -> ScoreSpread {
        if self.says {
            ScoreSpread::Levels(self.step)
        } else {
            ScoreSpread::Any
        }
    }

    fn front_wise(&self) -> bool {
        true
    }

    fn score_front(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        let mut top = 0;
        for (o, &row) in out.iter_mut().zip(asked) {
            *o = self.levels[candidates[row as usize].position as usize];
            top += u32::from(*o >= 1.0);
        }
        let front = if self.says {
            ScoreFront::At(top)
        } else {
            ScoreFront::Unsaid
        };
        (TieBand(0.0), front)
    }
}

/// A stage of any values: a row reads the value named for its position,
/// under the tie band the stage was given.
#[derive(Debug)]
struct Valued(Vec<f32>, f32);

impl Lifecycle for Valued {}

impl Scorer for Valued {
    fn name(&self) -> &'static str {
        "valued"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = self.0[candidates[row as usize].position as usize];
        }
        TieBand(self.1)
    }

    fn front_wise(&self) -> bool {
        true
    }
}

/// The probe of the rows a stage is handed, answering row by row: a
/// later position scores higher, and the rows handed last are kept.
#[derive(Debug)]
struct Asked(Arc<Mutex<Vec<u32>>>);

impl Lifecycle for Asked {}

impl Scorer for Asked {
    fn name(&self) -> &'static str {
        "asked"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        *self.0.lock().expect("the probe's rows") = kept.to_vec();
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = candidates[row as usize].position as f32 / 8.0;
        }
        TieBand::NORMALIZED
    }

    fn front_wise(&self) -> bool {
        true
    }
}

/// A stage that orders nothing, counting the times it is asked.
#[derive(Debug)]
struct Flat(Arc<Mutex<u32>>);

impl Lifecycle for Flat {}

impl Scorer for Flat {
    fn name(&self) -> &'static str {
        "flat"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        *self.0.lock().expect("the count") += 1;
        out.fill(1.0);
        TieBand(0.0)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Alike
    }
}

/// A picker that reads every score: the lowest first.
#[derive(Debug)]
struct Lowest;

impl Lifecycle for Lowest {}

impl Picker for Lowest {
    fn name(&self) -> &'static str {
        "lowest"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let mut order: Vec<usize> = (0..kept.len()).collect();
        order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        for at in order.into_iter().take(MAX_TARGETS) {
            out.push(target_of(&candidates[kept[at] as usize]));
        }
    }
}

/// A host's picker that draws for every request and says that it takes
/// its targets from a front.
#[derive(Debug)]
struct Eager;

impl Lifecycle for Eager {}

impl Picker for Eager {
    fn name(&self) -> &'static str {
        "eager"
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
        Lowest.pick(facts, candidates, kept, scores, band, out);
    }

    fn draws_first(&self) -> bool {
        true
    }

    fn front_wise(&self) -> bool {
        true
    }
}

/// One profile over the healthy rows: `stages` under `picker`.
fn staged(stages: Vec<WeightedScorer>, picker: Box<dyn Picker>) -> Engine {
    Engine::new(
        vec![Profile {
            name: "staged".into(),
            filters: vec![Box::new(Healthy)],
            scorers: stages,
            picker,
        }],
        vec![single_binding()],
        1,
        Needs::default(),
    )
}

/// A rank of `levels` with a step of a half, and a stage of any values.
fn rank(levels: &[f32], weight: f32, says: bool) -> WeightedScorer {
    WeightedScorer {
        scorer: Box::new(Levelled {
            levels: levels.to_vec(),
            step: 0.5,
            says,
        }),
        weight,
    }
}

fn valued(values: &[f32], band: f32) -> WeightedScorer {
    WeightedScorer {
        scorer: Box::new(Valued(values.to_vec(), band)),
        weight: 1.0,
    }
}

/// `rows` ready candidates, then candidates that are not ready up to
/// the count a decision is taken by rank from ([`RANK_FROM`]): the
/// filters leave the `rows`.
fn ranked_pool(rows: u32) -> Vec<Candidate> {
    (0..rows.max(RANK_FROM as u32))
        .map(|i| Candidate {
            health: Health {
                healthy: i < rows,
                ..Health::default()
            },
            ..candidate(i, 0)
        })
        .collect()
}

/// What the build reads of a profile's stages: which lead as ranks, the
/// band's room under them, and the mark, which is the front pickers'
/// alone.
#[test]
fn the_build_reads_which_leading_stages_are_ranks() {
    let order = |stages: Vec<WeightedScorer>| staged(stages, Box::new(Best)).orders[0];
    let level = |step: f32, weight: f32| WeightedScorer {
        scorer: Box::new(Levelled {
            levels: Vec::new(),
            step,
            says: true,
        }),
        weight,
    };
    let any = |weight: f32| WeightedScorer {
        scorer: Box::new(LeastRequests),
        weight,
    };
    let flat = |weight: f32| WeightedScorer {
        scorer: Box::new(Flat(Arc::default())),
        weight,
    };
    let wise = |weight: f32| WeightedScorer {
        weight,
        ..valued(&[], 0.0)
    };
    let ranks = |ranks: u8, alike: u8, fronts: u8, room: f32| Order {
        ranks,
        alike,
        fronts,
        room,
    };
    let none = ranks(0, 0, 0, 0.0);
    // A level that outweighs what stands behind it is a rank, and the
    // band's room is half of what it outweighs it by; one that only
    // equals it is none.
    assert_eq!(order(vec![level(1.0, 3.0), any(1.0)]), ranks(1, 0, 0, 0.25));
    // It may put rows in front where every stage behind it answers row
    // by row, or orders nothing.
    assert_eq!(
        order(vec![level(1.0, 3.0), wise(1.0)]),
        ranks(1, 0, 0b1, 0.25)
    );
    assert_eq!(
        order(vec![level(0.5, 4.0), flat(1.0), wise(1.0)]),
        ranks(2, 0b10, 0b01, 1.0 / 6.0 / 2.0)
    );
    // Two stages behind it that each order the rows: every row is asked.
    assert_eq!(
        order(vec![level(1.0, 4.0), wise(1.0), wise(1.0)]),
        ranks(1, 0, 0, 1.0 / 3.0 / 2.0)
    );
    assert_eq!(order(vec![level(1.0, 1.0), any(1.0)]), none);
    assert_eq!(order(vec![level(0.5, 2.0), any(1.0)]), none);
    // One that outweighs it by less than the least share is none.
    assert_eq!(order(vec![level(1.0, 1.000_01), any(1.0)]), none);
    // A stage that orders nothing is a rank where it leads, and weighs
    // nothing behind one.
    assert_eq!(
        order(vec![level(0.5, 4.0), flat(1.0), any(1.0)]),
        ranks(2, 0b10, 0, 1.0 / 6.0 / 2.0)
    );
    assert_eq!(order(vec![flat(1.0), any(1.0)]), ranks(1, 0b1, 0, 0.0));
    // Behind the first stage that is no rank, none is.
    assert_eq!(order(vec![any(1.0), level(1.0, 8.0)]), none);
    // A weight that is not above zero leaves the profile its sum.
    assert_eq!(order(vec![level(1.0, 2.0), any(0.0)]), none);
    // The mark is the front pickers': one that may read every score
    // has its profile asked whole, and so has one that draws first.
    let marked = |picker: Box<dyn Picker>| {
        let engine = staged(vec![level(1.0, 3.0), wise(1.0)], picker);
        assert_eq!(engine.orders[0].ranks, 1);
        engine.marks[0].0 & Marks::RANKED != 0
    };
    // A profile whose ranks leave nothing unasked decides by its sum.
    let plain = staged(vec![level(1.0, 3.0), any(1.0)], Box::new(Best));
    assert_eq!(plain.marks[0].0 & Marks::RANKED, 0);
    assert!(marked(Box::new(Best)) && marked(Box::new(CountFirst)));
    assert!(!marked(Box::new(Lowest)) && !marked(Box::new(Random)));
    // A profile whose picker draws for every request holds no mark
    // either, whatever that picker says of a front: the draw is asked
    // first, and a decision knows the profile by one compare of its
    // marks ([`Marks::ahead`]).
    let eager = Profile {
        name: "eager".into(),
        filters: Vec::new(),
        scorers: Vec::new(),
        picker: Box::new(Eager),
    };
    let gains = staged(vec![level(1.0, 3.0), wise(1.0)], Box::new(Best)).orders[0];
    assert_eq!(Marks::of(&eager, gains).ahead(), Marks::DRAWS_FIRST);
    // The cache-aware preset: the warm-up slice and the holders' tiers
    // are ranks, the gate's stage orders nothing, and the load is asked
    // of the rows they put in front.
    let engine = preset("cache_aware");
    let order = engine.orders[0];
    assert_eq!((order.ranks, order.alike, order.fronts), (3, 0b100, 0b010));
    assert!(order.room > 0.0);
    assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
    // A profile of one scorer of any values, or of none, holds no rank.
    for name in ["least_load", "round_robin", "consistent_hashing"] {
        let engine = preset(name);
        assert_eq!(engine.orders, vec![none], "{name}");
        assert_eq!(engine.marks[0].0 & Marks::RANKED, 0, "{name}");
    }
}

/// Who a stage behind a rank is asked of. The rows at the rank's
/// ceiling stand in front where they fill a decision's four targets and
/// are at most half of the rows: the stage behind is asked of them
/// alone. Fewer than the targets, or more than half, and every row is
/// asked. The probe scores a later position higher.
#[test]
fn a_stage_behind_a_rank_is_asked_of_the_rows_in_front_alone() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let pool = ranked_pool(9);
    // The targets by position, and the rows the stage behind was handed.
    let asked = |levels: &[f32]| {
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        let engine = staged(vec![rank(levels, 4.0, true), probe], Box::new(Best));
        assert_eq!(engine.orders[0].ranks, 1);
        let decision = reported(&engine, &request(7), &pool);
        let rows = handed.lock().expect("the probe's rows").clone();
        (positions(&decision), rows)
    };
    let every: Vec<u32> = (0..9).collect();
    // Four of nine fill the targets and are under half: the four alone.
    let (order, rows) = asked(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
    assert_eq!(rows, vec![0, 2, 4, 6]);
    assert_eq!(order, vec![6, 4, 2, 0]);
    // Five of nine are more than half: every row.
    let (order, rows) = asked(&[1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
    assert_eq!(rows, every);
    assert_eq!(order, vec![6, 4, 2, 1]);
    // Three of nine are fewer than the targets: every row, and the
    // fourth target is the best of the rows a level under.
    let (order, rows) = asked(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
    assert_eq!(rows, every);
    assert_eq!(order, vec![4, 2, 0, 8]);
    // No row at the ceiling: every row.
    let (order, rows) = asked(&[0.5; 9]);
    assert_eq!(rows, every);
    assert_eq!(order, vec![8, 7, 6, 5]);
    // Behind a stage that does not answer for a front alone no row
    // stands in front: four of nine, and every row is asked.
    let whole = WeightedScorer {
        scorer: Box::new(Handed(Arc::clone(&handed))),
        weight: 1.0,
    };
    let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
    let engine = staged(vec![rank(&levels, 4.0, true), whole], Box::new(Best));
    assert_eq!((engine.orders[0].ranks, engine.orders[0].fronts), (1, 0));
    reported(&engine, &request(7), &pool);
    assert_eq!(*handed.lock().expect("the probe's rows"), every);
}

/// A request that names rows is the draw's only under a picker that
/// draws for it ([`Marks::JOINED`]). Under one that never draws, a
/// profile of ranks is asked by rank: the stage behind the rank of the
/// rows in front alone.
#[test]
fn a_request_that_names_rows_is_asked_by_rank_where_the_picker_never_draws() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let probe = WeightedScorer {
        scorer: Box::new(Asked(Arc::clone(&handed))),
        weight: 1.0,
    };
    let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
    let engine = staged(vec![rank(&levels, 4.0, true), probe], Box::new(Best));
    assert_eq!(engine.marks[0].ahead(), Marks::RANKED);
    let pool = ranked_pool(9);
    let facts = RequestFacts {
        named: Some(AffinityRows::default()),
        ..request(7)
    };
    let decision = reported(&engine, &facts, &pool);
    assert_eq!(*handed.lock().expect("the probe's rows"), vec![0, 2, 4, 6]);
    assert_eq!(positions(&decision), vec![6, 4, 2, 0]);
}

/// A stage of the tests as a draw may be handed it: the stage's own
/// answers, and the word that it reads each row alone
/// ([`Scorer::row_wise`]).
#[derive(Debug)]
struct RowWise<S>(S);

impl<S: Scorer> Lifecycle for RowWise<S> {}

impl<S: Scorer> Scorer for RowWise<S> {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.score(facts, candidates, kept, out)
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn front_wise(&self) -> bool {
        self.0.front_wise()
    }

    fn spread(&self) -> ScoreSpread {
        self.0.spread()
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        self.0.score_front(facts, candidates, kept, asked, out)
    }

    fn needs(&self) -> Needs {
        self.0.needs()
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        self.0.scope()
    }

    fn warm_up_rule(&self) -> Option<WarmUpRule> {
        self.0.warm_up_rule()
    }
}

/// A host's picker the rows named join, which reads every score: it
/// says nothing of a front, so its profile holds no rank mark.
#[derive(Debug)]
struct Joining;

impl Lifecycle for Joining {}

impl Picker for Joining {
    fn name(&self) -> &'static str {
        "joining"
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
        Lowest.pick(facts, candidates, kept, scores, band, out);
    }

    fn draws_first(&self) -> bool {
        true
    }

    fn joined_by_named(&self) -> bool {
        true
    }
}

/// Under a picker the rows named join ([`Marks::JOINED`]), over a rated
/// pool of [`DRAW_FROM`] rows: a request that names rows is the draw's
/// ahead of the ranks, and one that names none is the ranks' by the
/// profile's rank mark alone. Without the mark every row is asked.
#[test]
fn a_joined_picker_draws_for_the_rows_named_and_ranks_by_the_mark_alone() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let rows = || handed.lock().expect("the probe's rows").clone();
    // A rank that puts four of sixteen rows in front, and the probe of
    // the rows the stage behind it is handed.
    let stages = || {
        let ranks = Levelled {
            levels: (0..16).map(|row| if row < 4 { 1.0 } else { 0.5 }).collect(),
            step: 0.5,
            says: true,
        };
        vec![
            WeightedScorer {
                scorer: Box::new(RowWise(ranks)),
                weight: 4.0,
            },
            WeightedScorer {
                scorer: Box::new(RowWise(Asked(Arc::clone(&handed)))),
                weight: 1.0,
            },
        ]
    };
    let pool = ranked_pool(16);
    let names = RequestFacts {
        named: Some(AffinityRows::default()),
        ..request(7)
    };
    // Under the built-in picker of that kind the profile holds the
    // three marks. A request that names none is asked by rank: the
    // stage behind the rank of the four rows in front.
    let sampled = pickers::SampledBest::new(2);
    let engine = staged(stages(), Box::new(sampled));
    assert_eq!(
        engine.marks[0].ahead(),
        Marks::DRAWS_FIRST | Marks::JOINED | Marks::RANKED
    );
    let ranked = reported(&engine, &request(7), &pool);
    assert!(!ranked.trace().drew_first());
    assert_eq!(rows(), vec![0, 1, 2, 3]);
    // One that names rows is the draw's, whatever the ranks would put in
    // front: the stage is handed the sample's two rows.
    let drew = reported(&engine, &names, &pool);
    assert!(drew.trace().drew_first());
    assert_eq!(rows().len(), 2);
    // The same stages under a picker that reads every score hold no
    // rank mark, though the first of them is a rank: a request that
    // names none is asked whole, every row.
    let engine = staged(stages(), Box::new(Joining));
    assert_eq!(engine.orders[0].fronts, 0b1);
    assert_eq!(engine.marks[0].ahead(), Marks::DRAWS_FIRST | Marks::JOINED);
    let whole = reported(&engine, &request(7), &pool);
    assert!(!whole.trace().drew_first());
    assert_eq!(rows(), (0..16).collect::<Vec<u32>>());
}

/// A leading stage that orders nothing is not asked by a ranked
/// decision: its share stands in every row's sum, and the decision is
/// the one the weighted sum reads, which asks it once.
#[test]
fn a_stage_that_orders_nothing_is_not_asked() {
    let asks = Arc::new(Mutex::new(0));
    let stages = vec![
        rank(&[1.0, 0.5, 0.0, 0.5, 1.0], 4.0, true),
        WeightedScorer {
            scorer: Box::new(Flat(Arc::clone(&asks))),
            weight: 1.0,
        },
        WeightedScorer {
            scorer: Box::new(Asked(Arc::default())),
            weight: 1.0,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    let order = engine.orders[0];
    assert_eq!((order.ranks, order.alike), (2, 0b10));
    let pool = ranked_pool(5);
    let facts = request(3);
    let ranked = reported(&engine, &facts, &pool);
    assert_eq!(*asks.lock().expect("the count"), 0);
    let summed = whole_pass(&engine, &facts, &pool);
    assert_eq!(*asks.lock().expect("the count"), 1);
    assert_eq!(ranked, summed);
}

/// The word a stage's scores read as under its spread ([`ScoreFront::of`]):
/// alike where every row stands at the ceiling, a count where every
/// other row stands a whole level or more under it, and nothing of any
/// other scores.
#[test]
fn a_word_is_read_off_the_scores_by_the_spread() {
    let levels = ScoreSpread::Levels(0.5);
    assert_eq!(ScoreFront::of(levels, &[1.0, 1.0]), ScoreFront::Alike);
    assert_eq!(ScoreFront::of(levels, &[]), ScoreFront::Alike);
    assert_eq!(
        ScoreFront::of(levels, &[1.0, 0.5, 0.0, 1.0]),
        ScoreFront::At(2)
    );
    assert_eq!(ScoreFront::of(levels, &[0.5, 0.0]), ScoreFront::At(0));
    // A row less than a whole level under the ceiling, or no number.
    assert_eq!(ScoreFront::of(levels, &[1.0, 0.75]), ScoreFront::Unsaid);
    assert_eq!(ScoreFront::of(levels, &[1.0, f32::NAN]), ScoreFront::Unsaid);
    assert_eq!(
        ScoreFront::of(ScoreSpread::Any, &[1.0, 0.0]),
        ScoreFront::Unsaid
    );
    assert_eq!(
        ScoreFront::of(ScoreSpread::Alike, &[1.0, 1.0]),
        ScoreFront::Alike
    );
}

/// A pool under the count a decision is taken by rank from is asked as
/// a profile without ranks asks: the stage that orders nothing is asked
/// at fifteen rows and is not at sixteen.
#[test]
fn a_pool_under_sixteen_rows_is_not_decided_by_rank() {
    let asks = Arc::new(Mutex::new(0));
    let levels: Vec<f32> = (0..16).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
    let stages = vec![
        rank(&levels, 4.0, true),
        WeightedScorer {
            scorer: Box::new(Flat(Arc::clone(&asks))),
            weight: 1.0,
        },
        WeightedScorer {
            scorer: Box::new(Asked(Arc::default())),
            weight: 1.0,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let facts = request(3);
    let under = reported(&engine, &facts, &pool[..RANK_FROM - 1]);
    assert_eq!(*asks.lock().expect("the count"), 1);
    reported(&engine, &facts, &pool);
    assert_eq!(*asks.lock().expect("the count"), 1);
    assert_eq!(under, whole_pass(&engine, &facts, &pool[..RANK_FROM - 1]));
}

/// Under each picker of the catalog, and under one that reads every
/// score, a decision whose leading stage is a rank is the decision of
/// the weighted sum of every stage over every row: the targets, the
/// alternates and the trace, over random rows, levels, values, weights,
/// tie bands and seeds. The twin's stages score alike and say nothing,
/// so its profile holds no rank. A picker the validator builds from its
/// parameters is not of this build yet.
#[test]
fn a_ranked_decision_is_the_weighted_sums_under_every_picker() {
    use crate::stages::registry::{Found, StageRegistry};
    let catalog = StageRegistry::builtin();
    let build = |name: &str| match catalog.find_picker(name) {
        Some(Found::Built(picker)) => Some(picker),
        Some(Found::Typed | Found::With(_)) => None,
        None => Some(Box::new(Lowest) as Box<dyn Picker>),
    };
    let mut marked = Vec::new();
    for name in catalog.pickers().chain(["lowest"]) {
        let mut draw = Draw::new(0x5eed);
        for round in 0..300u64 {
            let (Some(by_rank), Some(by_sum)) = (build(name), build(name)) else {
                break;
            };
            let rows = 1 + draw.below(40) as u32;
            let levels: Vec<f32> = (0..rows)
                .map(|_| [0.0, 0.5, 0.5, 1.0][draw.below(4)])
                .collect();
            let values: Vec<f32> = (0..rows).map(|_| draw.below(9) as f32 / 8.0).collect();
            // A band of none, a small one, and two a rank's room may
            // not hold.
            let band = [0.0, 1e-6, 0.3, 1.0][draw.below(4)];
            // A lead that is no rank, and two that are.
            let lead = [1.5, 3.0, 8.0][draw.below(3)];
            let stages = |says| vec![rank(&levels, lead, says), valued(&values, band)];
            let (ranked, summed) = (staged(stages(true), by_rank), staged(stages(false), by_sum));
            assert_eq!(summed.marks[0].0 & Marks::RANKED, 0, "{name}");
            if ranked.marks[0].0 & Marks::RANKED != 0 && !marked.contains(&name) {
                marked.push(name);
            }
            let pool: Vec<Candidate> = (0..rows)
                .map(|i| candidate(i, draw.below(3) as u32))
                .collect();
            for seed in 0..4 {
                let facts = request(round * 4 + seed);
                assert_eq!(
                    reported(&ranked, &facts, &pool),
                    reported(&summed, &facts, &pool),
                    "{name}, round {round}, seed {seed}: levels {levels:?}, values \
                         {values:?}, band {band}, lead {lead}"
                );
            }
        }
    }
    // The front pickers decided by rank: the case is not of sums alone.
    assert_eq!(marked, ["best", "count-first"]);
}

/// A tie band wider than a rank's room reaches from the rows in front
/// to a row behind them, and the decision asks every row every stage.
/// Four rows stand at the ceiling with the stage behind at nothing, and
/// one a level under them with it at its most: 0.75 and 0.625 of the
/// sum, one tie under a band of 0.25. Over the seeds the row behind
/// leads some decisions, as it does by the weighted sum.
#[test]
fn a_tie_band_wider_than_a_ranks_room_asks_every_row_every_stage() {
    let levels = [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    let values = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let engine = |says| {
        let stages = vec![rank(&levels, 3.0, says), valued(&values, 1.0)];
        staged(stages, Box::new(Best))
    };
    let (ranked, summed) = (engine(true), engine(false));
    assert_eq!(
        ranked.orders[0],
        Order {
            ranks: 1,
            alike: 0,
            fronts: 0b1,
            room: 0.0625
        }
    );
    let pool = ranked_pool(10);
    let mut led = Vec::new();
    for seed in 0..64 {
        let facts = request(seed);
        let decision = reported(&ranked, &facts, &pool);
        assert_eq!(decision, reported(&summed, &facts, &pool), "seed {seed}");
        led.push(positions(&decision)[0]);
    }
    assert!(led.contains(&4), "{led:?}");
}

/// A rank of the tests that says `1` of the scores `0`, which its
/// first rows read; every other row reads nothing.
#[derive(Debug)]
struct Says(Vec<f32>, ScoreFront);

impl Lifecycle for Says {}

impl Scorer for Says {
    fn name(&self) -> &'static str {
        "says"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.score_front(facts, candidates, kept, kept, out).0
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Levels(0.5)
    }

    fn score_front(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        out.fill(0.0);
        out[..self.0.len()].copy_from_slice(&self.0);
        (TieBand(0.0), self.1)
    }
}

/// A decision over twelve rows under a rank that says `said` of
/// `scores`, ahead of a stage that answers for a front alone: the
/// profile is decided by rank, so the engine reads the word.
fn told(scores: &[f32], said: ScoreFront) {
    let stages = vec![
        WeightedScorer {
            scorer: Box::new(Says(scores.to_vec(), said)),
            weight: 4.0,
        },
        valued(&[0.5; 12], 0.0),
    ];
    let engine = staged(stages, Box::new(Best));
    let pool = ranked_pool(12);
    reported(&engine, &request(1), &pool);
}

/// A rank that says a count of ceilings its scores do not hold fails a
/// debug build at the engine's check of the word: its author's case,
/// never a wrong decision in the field.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "says said At(5) of scores that read otherwise")]
fn a_rank_that_miscounts_its_front_fails_a_debug_build() {
    told(&[1.0, 1.0, 1.0, 1.0], ScoreFront::At(5));
}

/// And so does one whose row stands within a step of the ceiling: the
/// rows behind a front lie a step of the spread under it.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "says said At(4) of scores that read otherwise")]
fn a_rank_with_a_row_within_a_step_of_its_ceiling_fails_a_debug_build() {
    told(&[1.0, 1.0, 1.0, 1.0, 0.75], ScoreFront::At(4));
}

/// The cache-aware preset by rank is its weighted sum. Over pools of 4
/// to 64 rows where every row holds the request's prefix, one in two,
/// one in eight or none, under random loads and seeds, the decision is
/// the whole pass's: its targets, its alternates and the rows its trace
/// names. The whole pass asks each stage by its `score`, the decision by
/// rank asks the ranks by their `score_front`: a row scores alike by
/// either. The load is asked of the holders in front alone and scores
/// them among themselves, so their scores are not the whole pass's.
#[test]
fn the_cache_aware_preset_decides_by_rank_what_its_weighted_sum_decides() {
    let engine = preset("cache_aware");
    let mut draw = Draw::new(0xace);
    let mut fronts = 0;
    for round in 0..300u64 {
        let rows = 4 + draw.below(61) as u32;
        let pool: Vec<Candidate> = (0..rows)
            .map(|i| candidate(i, draw.below(4) as u32))
            .collect();
        let odds = [1, 2, 8, 1 << 20][draw.below(4)];
        let overlaps: Vec<u32> = (0..rows)
            .map(|_| if draw.below(odds) == 0 { 4 } else { 0 })
            .collect();
        let column = held(&overlaps);
        let facts = RequestFacts {
            prompt_tokens: 512,
            now_micros: 1_000_000,
            affinity: &column,
            ..request(round)
        };
        let holders = overlaps.iter().filter(|&&blocks| blocks > 0).count();
        fronts += usize::from(holders >= MAX_TARGETS && holders * 2 <= pool.len());
        let (ranked, summed) = (
            reported(&engine, &facts, &pool),
            whole_pass(&engine, &facts, &pool),
        );
        assert_eq!(
            positions(&ranked),
            positions(&summed),
            "round {round}: {overlaps:?}"
        );
        let named = |decision: &Decision| {
            let trace = decision.pick().expect("a pick").trace;
            (trace.top.map(|(id, _)| id), trace.top_len, trace.survivors)
        };
        assert_eq!(named(&ranked), named(&summed), "round {round}");
    }
    // In some rounds the holders fill the targets and are under half.
    assert!(fronts > 20, "{fronts}");
}

/// The cache-aware preset over fleets whose load reports are mixed: by
/// rank it decides what its whole pass decides, in every round of four
/// mixes. One row in eight is not ready and reports the pool's fastest
/// rate: a filter drops it, and no cost reads it. Where every row in
/// front holds a report the load is asked of the holders in front
/// alone, and a report without a rate drains at the mean rate of the
/// rows the filters kept. A row without a report is costed from its
/// reporting peers: in front it makes the load say that it must be
/// asked whole, and it is.
#[test]
fn the_cache_aware_preset_decides_by_rank_its_whole_pass_under_mixed_reports() {
    let engine = preset("cache_aware");
    let mut draw = Draw::new(0xace);
    // The rounds whose holders stand in front, and those of them the
    // front decided alone: the trace's scores are not the whole pass's.
    let (mut fronts, mut alone) = ([0u32; 4], [0u32; 4]);
    for round in 0..40_000u64 {
        // Every row rated; a third of the reports without a rate; a
        // third of the rows without a report; a third of the reports
        // without a rate and one row without a report.
        let mix = (round % 4) as usize;
        let rows = 8 + draw.below(57) as u32;
        let dark = draw.below(rows as usize) as u32;
        let pool: Vec<Candidate> = (0..rows)
            .map(|i| {
                let mut row = candidate(i, draw.below(4) as u32);
                row.health.healthy = draw.below(8) != 0;
                let reports = match mix {
                    2 => draw.below(3) != 0,
                    3 => i != dark,
                    _ => true,
                };
                let rated = !matches!(mix, 1 | 3) || draw.below(3) != 0;
                if reports {
                    row.load = Some(view::LoadRow {
                        running: draw.below(8) as u32,
                        queued: draw.below(4) as u32,
                        waiting_uncached_tokens: draw.below(8192) as u32,
                        live_rate: match (rated, row.health.healthy) {
                            (false, _) => 0,
                            (true, true) => 500 + draw.below(4000) as u32,
                            (true, false) => 60_000,
                        },
                        ..Default::default()
                    });
                }
                row
            })
            .collect();
        let odds = [2, 4, 8][draw.below(3)];
        let overlaps: Vec<u32> = (0..rows)
            .map(|_| if draw.below(odds) == 0 { 4 } else { 0 })
            .collect();
        let column = held(&overlaps);
        let facts = RequestFacts {
            prompt_tokens: 512,
            now_micros: 1_000_000,
            affinity: &column,
            ..request(round)
        };
        let kept = pool.iter().filter(|row| row.health.healthy).count();
        let holders = pool
            .iter()
            .zip(&overlaps)
            .filter(|(row, blocks)| row.health.healthy && **blocks > 0)
            .count();
        let front = holders >= MAX_TARGETS && holders * 2 <= kept;
        let (ranked, summed) = (
            reported(&engine, &facts, &pool),
            whole_pass(&engine, &facts, &pool),
        );
        assert_eq!(
            positions(&ranked),
            positions(&summed),
            "round {round}, mix {mix}: {overlaps:?}"
        );
        let top = |decision: &Decision| decision.pick().expect("a pick").trace.top;
        let named = |decision: &Decision| {
            let trace = decision.pick().expect("a pick").trace;
            (trace.top.map(|(id, _)| id), trace.top_len, trace.survivors)
        };
        assert_eq!(named(&ranked), named(&summed), "round {round}, mix {mix}");
        fronts[mix] += u32::from(front);
        alone[mix] += u32::from(front && top(&ranked) != top(&summed));
    }
    // In each mix the holders fill the targets and are under half in
    // most rounds.
    assert!(fronts.iter().all(|&rounds| rounds > 5_000), "{fronts:?}");
    // A front whose rows each hold a report decides alone, with a rate
    // or without one. A row without a report in it has it asked whole:
    // one such row stands in some fronts, and a third of the rows in
    // nearly every one.
    assert!(alone[0] > 5_000 && alone[1] > 5_000, "{alone:?}");
    assert!(alone[3] > 3_500 && alone[3] < alone[1], "{alone:?}");
    assert!(alone[2] < 1_000, "{alone:?}");
}

/// The cache-aware preset where the whole pass cannot tell the rows in
/// front apart. Eight of thirty-two rows hold the request's prefix. Each
/// runs one request and their rates step by one token a second from
/// 20,000, so their waits lie 2.56 microseconds apart: outside the
/// load's tie of one microsecond. With no queue behind them the
/// decision by rank is the whole pass's under every seed. With a queue
/// of 200 on one row behind, the pool's costs span a hundred seconds:
/// the whole pass spreads the load over that span, the eight's sums
/// meet in one value and its picker draws among them. By rank the load
/// is asked of the eight alone and orders them by their wait: the
/// targets are the four fastest under every seed, and each is a row the
/// whole pass draws among.
#[test]
fn the_rows_in_front_are_ordered_where_the_whole_pass_draws_among_them() {
    let engine = preset("cache_aware");
    let pool_of = |queued: u32| -> Vec<Candidate> {
        (0..32u32)
            .map(|at| {
                let mut row = candidate(at, 0);
                row.load = Some(view::LoadRow {
                    running: 1,
                    queued: if at == 31 { queued } else { 0 },
                    live_rate: if at < 8 { 20_000 + at } else { 2_000 },
                    ..Default::default()
                });
                row
            })
            .collect()
    };
    let overlaps: Vec<u32> = (0..32).map(|at| if at < 8 { 4 } else { 0 }).collect();
    let column = held(&overlaps);
    // The targets of 64 seeds, each list of four once: by rank, and by
    // the whole pass.
    let read = |pool: &[Candidate]| {
        let (mut by_rank, mut by_sum) = (Vec::new(), Vec::new());
        for seed in 0..64u64 {
            let facts = RequestFacts {
                prompt_tokens: 512,
                now_micros: 1_000_000,
                affinity: &column,
                ..request(seed)
            };
            by_rank.push(positions(&reported(&engine, &facts, pool)));
            by_sum.push(positions(&whole_pass(&engine, &facts, pool)));
        }
        by_rank.sort_unstable();
        by_rank.dedup();
        by_sum.sort_unstable();
        by_sum.dedup();
        (by_rank, by_sum)
    };
    let fastest = [vec![7, 6, 5, 4]];
    let (by_rank, by_sum) = read(&pool_of(0));
    assert_eq!(by_rank, fastest);
    assert_eq!(by_sum, fastest);
    let (by_rank, by_sum) = read(&pool_of(200));
    assert_eq!(by_rank, fastest);
    assert!(by_sum.len() > 1, "{by_sum:?}");
    assert!(by_sum.iter().flatten().all(|&at| at < 8), "{by_sum:?}");
}

/// A pool said to hold a report on every row, and a host that says it
/// where one row holds none. A report without a rate among the rows in
/// front drains at the mean rate of the rows the filters kept, as in
/// the whole pass: the load stage folds it over them and the rows in
/// front decide alone, so the targets are the whole pass's and the
/// trace's scores are the front's own. That mean is 2,000 tokens a
/// second here. Over the rows in front alone it is 1,000, and over the
/// pool, with the rows a filter dropped, 6,000: either would put that
/// row elsewhere among the targets. A row without a report among the
/// rows in front makes the stage say that it must be asked whole: the
/// decision is the whole pass's, and so are its trace's scores. Behind
/// the rows in front neither is read: they are costed alone, each from
/// its own report.
#[test]
fn a_row_in_front_without_a_rate_drains_at_the_kept_rows_mean_rate() {
    let engine = preset("cache_aware");
    // Thirty-two rows. The first eight hold the request's prefix and
    // run at 1,000 tokens a second, the next twenty at 2,350, and the
    // last four at 33,000 and are not ready. The row at `bare` reports
    // no rate, or does not report.
    let work = [1, 2, 4, 3, 6, 7, 8, 9];
    let pool_of = |bare: u32, reports: bool| -> Vec<Candidate> {
        (0..32u32)
            .map(|at| {
                let mut row = candidate(at, 0);
                row.health.healthy = at < 28;
                let rate = [1_000, 2_350, 33_000][usize::from(at >= 8) + usize::from(at >= 28)];
                if at != bare || reports {
                    row.load = Some(view::LoadRow {
                        running: work.get(at as usize).copied().unwrap_or(1 + at % 4),
                        live_rate: if at == bare { 0 } else { rate },
                        ..Default::default()
                    });
                }
                row
            })
            .collect()
    };
    let overlaps: Vec<u32> = (0..32).map(|at| if at < 8 { 4 } else { 0 }).collect();
    let column = held(&overlaps);
    let facts = RequestFacts {
        prompt_tokens: 512,
        now_micros: 1_000_000,
        affinity: &column,
        ..request(11)
    };
    // The targets by rank, which are the whole pass's, and whether the
    // trace's scores by rank are the whole pass's too.
    let decided = |pool: &[Candidate]| {
        let (ranked, summed) = (
            reported(&engine, &facts, pool),
            whole_pass(&engine, &facts, pool),
        );
        assert_eq!(positions(&ranked), positions(&summed));
        let top = |decision: &Decision| decision.pick().expect("a pick").trace.top;
        (positions(&ranked), top(&ranked) == top(&summed))
    };
    // The fourth row runs three requests. Without a rate it drains at
    // the kept rows' mean and waits behind the first row alone: the
    // front decides. At its own 1,000 it waits behind the second too.
    assert_eq!(decided(&pool_of(3, true)), (vec![0, 3, 1, 2], false));
    // Without a report it takes its reporting peers' least cost, a row
    // behind the front's: the whole pass's decision, and its scores.
    assert_eq!(decided(&pool_of(3, false)), (vec![3, 0, 1, 2], true));
    // Behind the rows in front, either way: the front decides alone.
    assert_eq!(decided(&pool_of(20, true)), (vec![0, 1, 3, 2], false));
    assert_eq!(decided(&pool_of(20, false)), (vec![0, 1, 3, 2], false));
}

/// A later rank puts fewer rows in front where they still fill the
/// targets and are at most half of the rows the rank before left. Of
/// sixteen rows the first rank leaves eight and the second four of
/// those: the stage behind both is asked of the four alone. Five of the
/// eight are more than half, and the eight are asked.
#[test]
fn a_later_rank_puts_fewer_rows_in_front() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let asked = |second: &[f32]| {
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        let stages = vec![rank(&first, 12.0, true), rank(second, 3.0, true), probe];
        let engine = staged(stages, Box::new(Best));
        assert_eq!(engine.orders[0].ranks, 2);
        let decision = reported(&engine, &request(9), &pool);
        let rows = handed.lock().expect("the probe's rows").clone();
        (positions(&decision), rows)
    };
    // Three of the eight are fewer than the targets: the eight are
    // asked, and the fourth target is the best of the other five.
    let mut second = vec![0.5; 16];
    second[..3].fill(1.0);
    let (order, rows) = asked(&second);
    assert_eq!(rows, (0..8).collect::<Vec<u32>>());
    assert_eq!(order, vec![2, 1, 0, 7]);
    second[3] = 1.0;
    let (order, rows) = asked(&second);
    assert_eq!(rows, vec![0, 1, 2, 3]);
    assert_eq!(order, vec![3, 2, 1, 0]);
    second[4] = 1.0;
    let (order, rows) = asked(&second);
    assert_eq!(rows, (0..8).collect::<Vec<u32>>());
    assert_eq!(order, vec![4, 3, 2, 1]);
}

/// A rank that puts fewer rows in front keeps what each of them holds.
/// The second rank leaves the last four of the eight in front, above
/// rows it wrote two other levels: the stage behind orders the four.
#[test]
fn a_narrowing_rank_keeps_the_sums_of_the_rows_it_leaves_in_front() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let mut second = vec![0.5; 16];
    second[2..4].fill(0.0);
    second[4..8].fill(1.0);
    let probe = WeightedScorer {
        scorer: Box::new(Asked(Arc::clone(&handed))),
        weight: 1.0,
    };
    let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), probe];
    let engine = staged(stages, Box::new(Best));
    let decision = reported(&engine, &request(9), &pool);
    assert_eq!(*handed.lock().expect("the probe's rows"), vec![4, 5, 6, 7]);
    assert_eq!(positions(&decision), vec![7, 6, 5, 4]);
}

/// A stage that writes only the rows it has a score for, and keeps the
/// rows it was handed last: a negative entry leaves its row unwritten.
#[derive(Debug)]
struct Sparse(Vec<f32>, Arc<Mutex<Vec<u32>>>);

impl Lifecycle for Sparse {}

impl Scorer for Sparse {
    fn name(&self) -> &'static str {
        "sparse"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let mut handed = self.1.lock().expect("the probe's rows");
        handed.clear();
        for (o, &row) in out.iter_mut().zip(kept) {
            let position = candidates[row as usize].position;
            handed.push(position);
            let value = self.0[position as usize];
            if value >= 0.0 {
                *o = value;
            }
        }
        TieBand(0.0)
    }

    fn front_wise(&self) -> bool {
        true
    }
}

/// A row a stage leaves unwritten reads nothing by rank too. The first
/// rank puts eight rows in front, and the second, asked of them, leaves
/// four there, each written its ceiling. The stage behind is handed
/// those four and writes three: the fourth reads nothing, as in the
/// zeroed buffer the whole pass hands, and stands last. A front that
/// kept the rank's output would hand it that ceiling and stand it first.
#[test]
fn a_row_a_stage_leaves_unwritten_reads_nothing_among_the_rows_in_front() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let second: Vec<f32> = (0..16).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
    let mut third = vec![-1.0; 16];
    third[..3].copy_from_slice(&[0.25, 0.5, 0.75]);
    let sparse = WeightedScorer {
        scorer: Box::new(Sparse(third, Arc::clone(&handed))),
        weight: 1.0,
    };
    let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), sparse];
    let engine = staged(stages, Box::new(Best));
    let by_rank = positions(&reported(&engine, &request(9), &pool));
    assert_eq!(*handed.lock().expect("the probe's rows"), vec![0, 1, 2, 3]);
    assert_eq!(by_rank, vec![2, 1, 0, 3]);
    let by_sum = positions(&whole_pass(&engine, &request(9), &pool));
    assert_eq!(by_rank, by_sum);
}

/// A rank parts the rows in front without leaving fewer there, the
/// three at its ceiling being under the targets. The rank behind it
/// then leaves no fewer either, although four stand at its own ceiling:
/// the stage behind is asked of the eight, and the targets are the
/// weighted sum's.
#[test]
fn a_rank_behind_a_parted_front_leaves_no_fewer_in_front() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let mut second = vec![0.5; 16];
    second[..3].fill(1.0);
    let mut third = vec![0.5; 16];
    third[2..6].fill(1.0);
    let stages = |says| {
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        vec![
            rank(&first, 48.0, says),
            rank(&second, 12.0, says),
            rank(&third, 3.0, says),
            probe,
        ]
    };
    let (ranked, summed) = (
        staged(stages(true), Box::new(Best)),
        staged(stages(false), Box::new(Best)),
    );
    assert_eq!(ranked.orders[0].ranks, 3);
    let facts = request(9);
    let decision = reported(&ranked, &facts, &pool);
    assert_eq!(
        *handed.lock().expect("the probe's rows"),
        (0..8).collect::<Vec<u32>>()
    );
    assert_eq!(positions(&decision), vec![2, 1, 0, 5]);
    assert_eq!(
        positions(&decision),
        positions(&reported(&summed, &facts, &pool))
    );
}

/// A profile is asked by rank only over a pool whose host says that
/// every row holds a report ([`PoolFacts::reported`]): the stage behind
/// the rank is handed the four rows in front there, and every row where
/// the host does not say it. The turn is read in both directions over
/// one engine, and the targets are the same either way.
#[test]
fn a_pool_not_said_to_hold_a_report_on_every_row_is_asked_whole_at_once() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let probe = WeightedScorer {
        scorer: Box::new(Asked(Arc::clone(&handed))),
        weight: 1.0,
    };
    let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
    let engine = staged(vec![rank(&levels, 4.0, true), probe], Box::new(Best));
    let pool = ranked_pool(9);
    // The targets by position, and the rows the stage behind was handed.
    let asked = |reported: bool| {
        let facts = PoolFacts {
            reported,
            ..PoolFacts::default()
        };
        let decision = engine.decide(
            &request(7),
            &facts,
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let rows = handed.lock().expect("the probe's rows").clone();
        (positions(&decision), rows)
    };
    let targets = vec![6, 4, 2, 0];
    let front = (targets.clone(), vec![0, 2, 4, 6]);
    assert_eq!(asked(true), front);
    // One row without a rate, a fresh worker before its first sample
    // among them, and the host stops saying it: every row.
    assert_eq!(asked(false), (targets, (0..9).collect::<Vec<u32>>()));
    // It reports: the four in front again.
    assert_eq!(asked(true), front);
}

/// The list of the rows in front keeps its length from one decision to
/// the next over one scratch, and is filled only as it grows. A front
/// of four, then of six, then of four again: each decision hands the
/// stage behind the rank its own front and decides as the whole pass,
/// whatever the list held before.
#[test]
fn a_front_is_listed_anew_over_a_scratch_that_held_another() {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let front_of = |levels: &[f32]| {
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        staged(vec![rank(levels, 4.0, true), probe], Box::new(Best))
    };
    let four: Vec<f32> = (0..16)
        .map(|i| if i < 8 && i % 2 == 0 { 1.0 } else { 0.5 })
        .collect();
    let six: Vec<f32> = (0..16).map(|i| if i < 6 { 1.0 } else { 0.5 }).collect();
    let (four, six) = (front_of(&four), front_of(&six));
    let pool = ranked_pool(16);
    let said = PoolFacts {
        reported: true,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    // The targets by position, and the rows the stage behind was handed.
    let mut asked = |engine: &Engine| {
        let facts = request(7);
        let decision = engine.decide(&facts, &said, &pool, &Constraints::default(), &mut scratch);
        let rows = handed.lock().expect("the probe's rows").clone();
        assert_eq!(
            positions(&decision),
            positions(&whole_pass(engine, &facts, &pool))
        );
        (positions(&decision), rows)
    };
    assert_eq!(asked(&four), (vec![6, 4, 2, 0], vec![0, 2, 4, 6]));
    assert_eq!(asked(&six), (vec![5, 4, 3, 2], vec![0, 1, 2, 3, 4, 5]));
    assert_eq!(asked(&four), (vec![6, 4, 2, 0], vec![0, 2, 4, 6]));
}

/// A stage of the tests that answers for a front alone and notes the
/// two lists each ask by the front entry hands it: the kept rows, and
/// the rows asked.
#[derive(Debug)]
struct Shown(Arc<Mutex<Vec<[Vec<u32>; 2]>>>);

impl Lifecycle for Shown {}

impl Scorer for Shown {
    fn name(&self) -> &'static str {
        "shown"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = candidates[row as usize].position as f32 / 16.0;
        }
        TieBand::NORMALIZED
    }

    fn front_wise(&self) -> bool {
        true
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        let lists = [kept.to_vec(), asked.to_vec()];
        self.0.lock().expect("the asks").push(lists);
        (
            self.score(facts, candidates, asked, out),
            ScoreFront::Unsaid,
        )
    }
}

/// A stage behind a rank is shown the rows the filters kept beside the
/// rows in front, and no row a filter dropped. Of twenty rows sixteen
/// are ready: the first rank leaves eight of them in front and the
/// second four of those, and the stage behind both is asked of the
/// four beside the sixteen.
#[test]
fn a_stage_behind_a_rank_is_shown_the_kept_rows_beside_the_rows_in_front() {
    let shown = Arc::new(Mutex::new(Vec::new()));
    let pool: Vec<Candidate> = (0..20)
        .map(|i| Candidate {
            health: Health {
                healthy: i < 16,
                ..Health::default()
            },
            ..candidate(i, 0)
        })
        .collect();
    let first: Vec<f32> = (0..20).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let second: Vec<f32> = (0..20).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
    let probe = WeightedScorer {
        scorer: Box::new(Shown(Arc::clone(&shown))),
        weight: 1.0,
    };
    let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), probe];
    let engine = staged(stages, Box::new(Best));
    let decision = reported(&engine, &request(9), &pool);
    let kept: Vec<u32> = (0..16).collect();
    assert_eq!(
        *shown.lock().expect("the asks"),
        vec![[kept, vec![0, 1, 2, 3]]]
    );
    assert_eq!(positions(&decision), vec![3, 2, 1, 0]);
}

/// A stage of the tests that answers row by row, and says it must be
/// asked whole where it is handed fewer rows than its count. It notes
/// how many rows each ask handed it.
#[derive(Debug)]
struct Wholly(Arc<Mutex<Vec<usize>>>, usize);

impl Lifecycle for Wholly {}

impl Scorer for Wholly {
    fn name(&self) -> &'static str {
        "wholly"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.lock().expect("the asks").push(kept.len());
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = candidates[row as usize].position as f32 / 16.0;
        }
        TieBand::NORMALIZED
    }

    fn front_wise(&self) -> bool {
        true
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        if asked.len() < self.1 {
            self.0.lock().expect("the asks").push(asked.len());
            return (TieBand::NORMALIZED, ScoreFront::Whole);
        }
        (
            self.score(facts, candidates, asked, out),
            ScoreFront::Unsaid,
        )
    }
}

/// A stage behind a rank that says it must be asked whole is asked
/// again of every row, and the decision is the weighted sum's: the rows
/// in front decide alone only where the stage answers for them alone.
#[test]
fn a_stage_that_must_be_asked_whole_is_asked_of_every_row() {
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let levels: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let decided = |least: usize| {
        let asks = Arc::new(Mutex::new(Vec::new()));
        let stage = WeightedScorer {
            scorer: Box::new(Wholly(Arc::clone(&asks), least)),
            weight: 1.0,
        };
        let engine = staged(vec![rank(&levels, 4.0, true), stage], Box::new(Best));
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        let decision = reported(&engine, &request(3), &pool);
        let asks = asks.lock().expect("the asks").clone();
        (positions(&decision), asks)
    };
    // It answers for the eight in front alone: asked once, of them.
    let (alone, asks) = decided(8);
    assert_eq!(asks, vec![8]);
    // It must be asked whole: asked of the eight, then of every row,
    // and the targets are the same four.
    let (whole, asks) = decided(16);
    assert_eq!(asks, vec![8, 16]);
    assert_eq!(whole, alone);
    assert_eq!(whole, vec![7, 6, 5, 4]);
}

/// A rank of the tests that says it must be asked whole and writes
/// nothing then: its plain answer is the level named for a row's
/// position.
#[derive(Debug)]
struct Shy(Vec<f32>);

impl Lifecycle for Shy {}

impl Scorer for Shy {
    fn name(&self) -> &'static str {
        "shy"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = self.0[candidates[row as usize].position as usize];
        }
        TieBand(0.0)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Levels(0.5)
    }

    fn score_front(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[u32],
        _: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        (TieBand(0.0), ScoreFront::Whole)
    }
}

/// A pool the overload veto empties is put to the binding's all-vetoed
/// rule by a decision asked by rank, as by the whole pass: the filter
/// half is one, and so is the plan's home for an emptied pool. The rule
/// steers the request by default, and sheds it where the binding says so.
#[test]
fn a_decision_by_rank_puts_an_all_vetoed_pool_to_the_bindings_rule() {
    let mut engine = preset("cache_aware");
    assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
    let mut pool: Vec<Candidate> = (0..32).map(|i| candidate(i, i % 5)).collect();
    for c in &mut pool {
        c.health.overloaded = true;
    }
    let facts = RequestFacts {
        prompt_tokens: 512,
        now_micros: 1_000_000,
        ..request(7)
    };
    let steered = reported(&engine, &facts, &pool);
    assert!(steered.pick().is_some(), "{steered:?}");
    assert_eq!(steered, whole_pass(&engine, &facts, &pool));
    engine.shed_when_all_vetoed(0);
    let shed = reported(&engine, &facts, &pool);
    assert!(shed.pick().is_none(), "{shed:?}");
    assert_eq!(shed, whole_pass(&engine, &facts, &pool));
}

/// A stage of the tests that orders nothing and notes how many rows
/// each ask handed it.
#[derive(Debug)]
struct Still(Arc<Mutex<Vec<usize>>>);

impl Lifecycle for Still {}

impl Scorer for Still {
    fn name(&self) -> &'static str {
        "still"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.lock().expect("the asks").push(kept.len());
        out.fill(1.0);
        TieBand(0.0)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Alike
    }
}

/// A profile of ten stages: a rank, seven stages that order nothing, a
/// ninth that says its front and a tenth that orders nothing. The build
/// reads eight stages as ranks, so the ninth is none and its word
/// leaves no fewer rows in front: the tenth is asked of the eight rows
/// the leading rank put there, and the decision is the weighted sum's.
/// A stage past the eighth has no bit of the build's to be read by.
/// Read by its index, the ninth stopped a debug build, and a release
/// build read the leading rank's bit and asked the tenth of four rows.
#[test]
fn a_stage_past_the_eighth_that_says_its_front_is_no_rank() {
    let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let last: Vec<f32> = (0..16)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.5 })
        .collect();
    let handed = Arc::new(Mutex::new(Vec::new()));
    let mut stages = vec![rank(&lead, 8.0, true)];
    stages.extend((0..7).map(|_| WeightedScorer {
        scorer: Box::new(Flat(Arc::default())),
        weight: 1.0,
    }));
    stages.push(rank(&last, 1.0, true));
    stages.push(WeightedScorer {
        scorer: Box::new(Still(Arc::clone(&handed))),
        weight: 1.0,
    });
    let engine = staged(stages, Box::new(Best));
    assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
    assert_eq!(usize::from(engine.orders[0].ranks), Order::MOST);
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let facts = request(5);
    let ranked = reported(&engine, &facts, &pool);
    assert_eq!(*handed.lock().expect("the asks"), vec![8]);
    // The four rows at both ceilings tie, and the seed draws their order.
    let mut targets = positions(&ranked);
    targets.sort_unstable();
    assert_eq!(targets, vec![0, 2, 4, 6]);
    assert_eq!(
        positions(&ranked),
        positions(&whole_pass(&engine, &facts, &pool))
    );
}

/// A level stage of the tests that answers among the rows it is handed:
/// the last of them reads the ceiling. It does not say that it may be
/// asked of a front alone, and notes how many rows each ask handed it.
#[derive(Debug)]
struct Among(Arc<Mutex<Vec<usize>>>);

impl Lifecycle for Among {}

impl Scorer for Among {
    fn name(&self) -> &'static str {
        "among"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.lock().expect("the asks").push(kept.len());
        out.fill(0.5);
        if let Some(last) = out.last_mut() {
            *last = 1.0;
        }
        TieBand(0.0)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Levels(0.5)
    }
}

/// A rank that parts the rows and may not put them in front, since a
/// rank behind it answers among the rows it is handed: every row is
/// asked on, and the decision is the weighted sum's. The build's word is
/// read where the rows part.
#[test]
fn a_rank_that_may_not_put_rows_in_front_asks_every_row_on() {
    let asks = Arc::new(Mutex::new(Vec::new()));
    let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let early: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 16.0).collect();
    let stages = vec![
        rank(&lead, 8.0, true),
        WeightedScorer {
            scorer: Box::new(Among(Arc::clone(&asks))),
            weight: 2.0,
        },
        WeightedScorer {
            scorer: Box::new(Valued(early, 0.0)),
            weight: 0.5,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
    // The second rank may put rows in front, the first may not.
    assert_eq!((engine.orders[0].ranks, engine.orders[0].fronts), (2, 0b10));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let facts = request(11);
    let ranked = reported(&engine, &facts, &pool);
    assert_eq!(*asks.lock().expect("the asks"), vec![16]);
    assert_eq!(positions(&ranked), vec![0, 1, 2, 3]);
    assert_eq!(
        positions(&ranked),
        positions(&whole_pass(&engine, &facts, &pool))
    );
}

/// A host's picker that answers a built-in's name and reads every score:
/// the lowest first.
#[derive(Debug)]
struct Namesake;

impl Lifecycle for Namesake {}

impl Picker for Namesake {
    fn name(&self) -> &'static str {
        "best"
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
        Lowest.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A picker is handed a front alone by its own word
/// ([`Picker::front_wise`]), never by its name: a host's picker that
/// answers `best` and reads every score is handed every row, and its
/// profile is not decided by rank.
#[test]
fn a_picker_is_handed_a_front_by_its_word_and_not_by_its_name() {
    let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let stages = |probe: &Arc<Mutex<Vec<u32>>>| {
        vec![
            rank(&lead, 4.0, true),
            WeightedScorer {
                scorer: Box::new(Asked(Arc::clone(probe))),
                weight: 1.0,
            },
        ]
    };
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    let facts = request(13);
    let handed = Arc::new(Mutex::new(Vec::new()));
    let host = staged(stages(&handed), Box::new(Namesake));
    assert_eq!(host.marks[0].0 & Marks::RANKED, 0);
    // The lowest sums are the rows behind the rank's ceiling.
    assert_eq!(
        positions(&reported(&host, &facts, &pool)),
        vec![8, 9, 10, 11]
    );
    assert_eq!(handed.lock().expect("the probe's rows").len(), 16);
    let built = staged(stages(&handed), Box::new(Best));
    assert_ne!(built.marks[0].0 & Marks::RANKED, 0);
    assert_eq!(
        positions(&reported(&built, &facts, &pool)),
        vec![7, 6, 5, 4]
    );
    assert_eq!(handed.lock().expect("the probe's rows").len(), 8);
}

/// A stage of the tests that says it orders nothing and returns a band.
#[derive(Debug)]
struct Banded;

impl Lifecycle for Banded {}

impl Scorer for Banded {
    fn name(&self) -> &'static str {
        "banded"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand(0.5)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Alike
    }
}

/// A stage that says it orders nothing returns no band
/// ([`ScoreSpread::Alike`]): a decision by rank does not ask it, so a band
/// of its own would widen the whole pass's ties and no other's. One that
/// returns a band fails a debug build where it is asked: its author's case,
/// never two decisions in the field.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "banded orders nothing and returned a band")]
fn a_stage_that_orders_nothing_and_returns_a_band_fails_a_debug_build() {
    let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
    let stages = vec![
        rank(&lead, 4.0, true),
        WeightedScorer {
            scorer: Box::new(Banded),
            weight: 1.0,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
    whole_pass(&engine, &request(17), &pool);
}

/// The stage of the case above behind a stage that orders the rows: by
/// rank it is asked, and the decision ends at its band.
#[cfg(debug_assertions)]
fn banded_behind(levels: &[f32]) {
    let stages = vec![
        rank(levels, 8.0, true),
        WeightedScorer {
            scorer: Box::new(Asked(Arc::default())),
            weight: 1.0,
        },
        WeightedScorer {
            scorer: Box::new(Banded),
            weight: 1.0,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    assert_eq!(engine.orders[0].ranks, 1);
    reported(&engine, &request(7), &ranked_pool(9));
}

/// Behind a stage that orders the rows, a stage that says it orders
/// nothing is asked by rank: of the rows in front where four of nine
/// stand at the rank's ceiling. A band of its own fails a debug build
/// there too.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "banded orders nothing and returned a band")]
fn a_band_from_a_stage_that_orders_nothing_fails_a_debug_build_behind_a_front() {
    banded_behind(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
}

/// And of every row where three of nine stand at the rank's ceiling,
/// fewer than a decision's targets.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "banded orders nothing and returned a band")]
fn a_band_from_a_stage_that_orders_nothing_fails_a_debug_build_where_every_row_is_asked_on() {
    banded_behind(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
}

/// A stage of the tests that says it orders nothing and returns no band
/// where every row is asked. Handed a front, it returns a band beside
/// the word that it must be asked whole.
#[derive(Debug)]
struct BandedWhole;

impl Lifecycle for BandedWhole {}

impl Scorer for BandedWhole {
    fn name(&self) -> &'static str {
        "banded-whole"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand(0.0)
    }

    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Alike
    }

    fn score_front(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        out.fill(1.0);
        (TieBand(0.5), ScoreFront::Whole)
    }
}

/// The check stands ahead of the way out of the rows in front. A stage
/// that says it orders nothing and answers that it must be asked whole
/// fails a debug build for the band beside that word, although that
/// band is not read and its answer where every row is asked holds none.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "banded-whole orders nothing and returned a band")]
fn a_band_beside_the_word_to_be_asked_whole_fails_a_debug_build() {
    let stages = vec![
        rank(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5], 8.0, true),
        WeightedScorer {
            scorer: Box::new(Asked(Arc::default())),
            weight: 1.0,
        },
        WeightedScorer {
            scorer: Box::new(BandedWhole),
            weight: 1.0,
        },
    ];
    let engine = staged(stages, Box::new(Best));
    assert_eq!(engine.orders[0].ranks, 1);
    reported(&engine, &request(7), &ranked_pool(9));
}

/// A leading rank that says it must be asked whole was handed every
/// row: it is asked again plainly, and the decision is the weighted
/// sum's.
#[test]
fn a_leading_rank_that_must_be_asked_whole_is_asked_plainly() {
    let levels = [1.0, 0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.5];
    let values = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.0];
    let shy = WeightedScorer {
        scorer: Box::new(Shy(levels.to_vec())),
        weight: 3.0,
    };
    let ranked = staged(vec![shy, valued(&values, 0.0)], Box::new(Best));
    assert_ne!(ranked.marks[0].0 & Marks::RANKED, 0);
    let stages = vec![rank(&levels, 3.0, false), valued(&values, 0.0)];
    let summed = staged(stages, Box::new(Best));
    let pool = ranked_pool(10);
    for seed in 0..16 {
        let facts = request(seed);
        assert_eq!(
            positions(&reported(&ranked, &facts, &pool)),
            positions(&reported(&summed, &facts, &pool)),
            "seed {seed}"
        );
    }
    // The five at the ceiling in the order of the stage behind.
    let led = positions(&reported(&ranked, &request(0), &pool));
    assert_eq!(led, vec![5, 8, 6, 2]);
}
