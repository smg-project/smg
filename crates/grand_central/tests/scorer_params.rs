//! A scorer registered by its host: the parameters a document supplies it,
//! and the stream it draws from.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use smg_grand_central::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Health, PoolFacts, RequestFacts, Scorer, Scratch, TieBand, WorkerId,
    },
    stages::{Lifecycle, StageBuild, StageRegistry},
    Draw,
};

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreferWorker {
    preferred: u64,
}

impl Lifecycle for PreferWorker {}

impl Scorer for PreferWorker {
    fn name(&self) -> &'static str {
        "prefer-worker"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&row, score) in kept.iter().zip(out) {
            *score = if candidates[row as usize].id.0 == self.preferred {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

#[test]
fn a_yaml_parameter_moves_the_pick() {
    let engine = {
        let mut catalog = StageRegistry::builtin();
        catalog
            .scorer_with("prefer-worker", |build: &mut StageBuild<'_>| {
                Ok(Box::new(build.params::<PreferWorker>()?))
            })
            .expect("a free name");
        let document: PolicySpec = serde_yaml::from_str(&format!(
            "apiVersion: {API_VERSION}
profiles:
  default:
    score:
      - least-requests
      - prefer-worker: {{ preferred: 89, weight: 7 }}
    pick: best
"
        ))
        .expect("a well-formed document");
        document.validate_in(&catalog, &[]).expect("validates")
    };
    let mut health = Health::default();
    health.healthy = true;
    let rows = [
        {
            let mut row = Candidate::default();
            row.id = WorkerId(31);
            row.health = health;
            row
        },
        {
            let mut row = Candidate::default();
            row.position = 1;
            row.id = WorkerId(89);
            row.in_flight = 19;
            row.health = health;
            row
        },
    ];
    let mut pool = PoolFacts::default();
    pool.pool_size = rows.len() as u32;
    let decision = engine.decide(
        &RequestFacts::default(),
        &pool,
        &rows,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let target = decision
        .pick()
        .expect("a pick")
        .targets
        .first()
        .expect("a target");
    // The configured preference outweighs the busier row's load.
    assert_eq!(target.id, WorkerId(89));
}

/// A scorer that scores every row alike, draws one of two from the stream
/// of its own name and keeps what it drew.
#[derive(Debug)]
struct Coin {
    drew: Arc<AtomicUsize>,
}

impl Lifecycle for Coin {}

impl Scorer for Coin {
    fn name(&self) -> &'static str {
        "coin"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let mut draw = Draw::for_stage(facts.seed, self.name());
        self.drew.store(draw.below(2), Ordering::Relaxed);
        out[..kept.len()].fill(0.5);
        TieBand::NORMALIZED
    }
}

/// A registered scorer that draws from the stream of its own name: what it
/// draws does not follow the picker's draw from the same seed. The scorer
/// draws one of two, and the `best` picker draws between two rows the
/// scorer leaves tied: over the seeds the two draws fall four ways, each
/// about as often. Drawn from the seed's own stream, the scorer's value
/// would be the picker's, and the draws would fall two ways.
#[test]
fn a_registered_scorer_draws_apart_from_the_picker() {
    let drew = Arc::new(AtomicUsize::new(0));
    let engine = {
        let mut catalog = StageRegistry::builtin();
        let kept = Arc::clone(&drew);
        catalog
            .scorer("coin", move || {
                Box::new(Coin {
                    drew: Arc::clone(&kept),
                })
            })
            .expect("a free name");
        let document: PolicySpec = serde_yaml::from_str(&format!(
            "apiVersion: {API_VERSION}
profiles:
  default:
    score: [coin]
    pick: best
"
        ))
        .expect("a well-formed document");
        document.validate_in(&catalog, &[]).expect("validates")
    };
    let mut health = Health::default();
    health.healthy = true;
    let rows = [0, 1].map(|position| {
        let mut row = Candidate::default();
        row.position = position;
        row.id = WorkerId(u64::from(position) + 1);
        row.health = health;
        row
    });
    let mut pool = PoolFacts::default();
    pool.pool_size = rows.len() as u32;
    let mut ways = [0u32; 4];
    for seed in 0..4_000u64 {
        let mut facts = RequestFacts::default();
        facts.seed = seed;
        let decision = engine.decide(
            &facts,
            &pool,
            &rows,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let picked = decision
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target")
            .position as usize;
        ways[drew.load(Ordering::Relaxed) * 2 + picked] += 1;
    }
    assert!(ways.iter().all(|n| (850..=1150).contains(n)), "{ways:?}");
}
