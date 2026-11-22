//! A document supplies the parameters of a picker registered by its host.

use smg_policy_controller::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Health, Picker, PoolFacts, RequestFacts, Scratch, Targets, TieBand,
        WorkerId,
    },
    stages::{Best, Lifecycle, StageBuild, StageError, StageRegistry},
};

/// Passes over the first `rows` of the workers the filters left and picks
/// the best of the rest.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PastFirst {
    rows: u32,
}

impl Lifecycle for PastFirst {}

impl Picker for PastFirst {
    fn name(&self) -> &'static str {
        "past-first"
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
        let from = (self.rows as usize).min(kept.len());
        Best.pick(facts, candidates, &kept[from..], &scores[from..], band, out);
    }
}

#[test]
fn a_yaml_parameter_moves_the_pick() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("past-first", |build: &mut StageBuild<'_>| {
            let rule: PastFirst = build.params()?;
            if rule.rows > 2 {
                return Err(StageError::bad("rows", rule.rows, "at most 2"));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    // Three healthy rows, each with one more request in flight than the
    // row before it: the first is the best by the scorer.
    let rows: Vec<Candidate> = (0..3)
        .map(|position| Candidate {
            position,
            id: WorkerId(u64::from(position) + 70),
            in_flight: position,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect();
    let document = |past: u32| -> PolicySpec {
        serde_yaml::from_str(&format!(
            "apiVersion: {API_VERSION}
profiles:
  default:
    filter:
      - healthy
    score:
      - least-requests
    pick:
      past-first: {{ rows: {past} }}
"
        ))
        .expect("a well-formed document")
    };
    for past in 0..3 {
        let engine = document(past)
            .validate_in(&catalog, &[])
            .expect("validates");
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
        // The document passes over `past` rows: the best of the rest is
        // the row at that place.
        assert_eq!(target.id, WorkerId(u64::from(past) + 70), "{past} rows");
    }
    // What the factory refuses is refused at the parameter's own path.
    let refused = document(3)
        .validate_in(&catalog, &[])
        .expect_err("the factory takes two rows at most");
    assert_eq!(
        refused.field_path(),
        "profiles.default.pick.past-first.rows"
    );
    assert_eq!(
        refused.to_string(),
        "profiles.default.pick.past-first.rows = 3 is not at most 2"
    );
}
