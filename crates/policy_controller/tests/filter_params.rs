//! A document supplies the parameters of a filter registered by its host.

use smg_policy_controller::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Decision, Filter, Health, PoolFacts, RequestFacts, Scratch,
        WorkerId,
    },
    stages::{StageBuild, StageError, StageRegistry},
};

/// Keeps the workers with `slots` of their eight request slots free.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MinFree {
    slots: u32,
}

impl Filter for MinFree {
    fn name(&self) -> &'static str {
        "min-free"
    }

    fn keep(&self, _: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        candidate.in_flight.saturating_add(self.slots) <= 8
    }
}

#[test]
fn a_yaml_parameter_moves_the_rows_kept() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("min-free", |build: &mut StageBuild<'_>| {
            let rule: MinFree = build.params()?;
            if rule.slots == 0 {
                return Err(StageError::bad("slots", rule.slots, "above zero"));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    // One healthy row with six of its eight slots taken.
    let rows = [Candidate {
        id: WorkerId(89),
        in_flight: 6,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        ..Candidate::default()
    }];
    let decide = |slots: u32| {
        let document: PolicySpec = serde_yaml::from_str(&format!(
            "apiVersion: {API_VERSION}
profiles:
  default:
    filter:
      - healthy
      - min-free: {{ slots: {slots} }}
    pick: random
"
        ))
        .expect("a well-formed document");
        let engine = document.validate_in(&catalog, &[]).expect("validates");
        engine.decide(
            &RequestFacts::default(),
            &PoolFacts {
                pool_size: rows.len() as u32,
                ..PoolFacts::default()
            },
            &rows,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    // The document asks two free slots: the row has them and is the pick.
    let kept = decide(2);
    let target = kept
        .pick()
        .expect("a pick")
        .targets
        .first()
        .expect("a target");
    assert_eq!(target.id, WorkerId(89));
    // It asks three: the filter leaves nobody, and the shed names it.
    let Decision::Shed(shed) = decide(3) else {
        panic!("the row lacks three free slots");
    };
    assert_eq!(shed.emptied_by, "min-free");
}
