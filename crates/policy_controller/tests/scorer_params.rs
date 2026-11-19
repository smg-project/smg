//! A document supplies the parameters of a scorer registered by its host.

use smg_policy_controller::{
    Candidate, Catalog, Constraints, Health, Lifecycle, PolicySpec, PoolFacts, RequestFacts,
    Scorer, Scratch, StageBuild, TieBand, WorkerId, API_VERSION,
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
        let mut catalog = Catalog::builtin();
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
    let health = Health {
        healthy: true,
        ..Health::default()
    };
    let rows = [
        Candidate {
            id: WorkerId(31),
            health,
            ..Candidate::default()
        },
        Candidate {
            position: 1,
            id: WorkerId(89),
            in_flight: 19,
            health,
            ..Candidate::default()
        },
    ];
    let decision = engine.decide(
        &RequestFacts::default(),
        &PoolFacts {
            pool_size: rows.len() as u32,
            ..PoolFacts::default()
        },
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
