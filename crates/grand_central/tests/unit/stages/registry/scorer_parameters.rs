//! The stage registry's cases of a scorer added in code: its factory, and
//! the parameters a document writes under its name.

use super::*;

const PRICE_PATH: &str = "profiles.default.score[0].queue-price";
const PRICE_FIELD: &str = "profiles.default.score[0].queue-price.per_request";
const BAD_PRICE: &str = "[{ queue-price: { per_request: -11 } }]";

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QueuePrice {
    per_request: i32,
}

impl Lifecycle for QueuePrice {}

impl Scorer for QueuePrice {
    fn name(&self) -> &'static str {
        "queue-price"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&row, score) in kept.iter().zip(out) {
            let cost = candidates[row as usize].in_flight as f32 * self.per_request as f32;
            *score = 1.0 / (1.0 + cost);
        }
        TieBand::NORMALIZED
    }
}

fn price_catalog() -> StageRegistry {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", |build| {
            let price: QueuePrice = build.params()?;
            if price.per_request <= 0 {
                return Err(crate::stages::StageError::bad(
                    "per_request",
                    price.per_request,
                    "above zero",
                ));
            }
            Ok(Box::new(price))
        })
        .expect("a free name");
    catalog
}

fn score_document(score: &str) -> PolicySpec {
    let version = crate::document::API_VERSION;
    serde_yaml::from_str(&format!(
        "apiVersion: {version}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n"
    ))
    .expect("a well-formed document")
}

#[test]
fn a_scorer_keeps_its_typed_parameter_and_the_profile_keeps_its_weight() {
    let engine = score_document("[{ queue-price: { weight: 3.5, per_request: 137 } }]")
        .validate_in(&price_catalog(), &[])
        .expect("the scorer reads its price");
    let weighted = &engine.profiles()[0].scorers[0];
    assert_eq!(weighted.weight, 3.5);
    let row = Candidate {
        in_flight: 1,
        ..Candidate::default()
    };
    let mut out = [0.0];
    weighted
        .scorer
        .score(&RequestFacts::default(), &[row], &[0], &mut out);
    assert_eq!(out, [1.0 / 138.0]);
}

#[test]
fn weight_is_absent_from_both_views_of_the_parameters() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Price {
        per_request: i32,
        flat: u32,
    }

    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", |build| {
            let written = build.written();
            assert!(written
                .keys()
                .map(String::as_str)
                .eq(["flat", "per_request"]));
            let price: Price = build.params()?;
            assert_eq!((price.per_request, price.flat), (41, 13));
            Ok(Box::new(QueuePrice {
                per_request: price.per_request,
            }))
        })
        .expect("a free name");
    score_document("[{ queue-price: { weight: 6.25, per_request: 41, flat: 13 } }]")
        .validate_in(&catalog, &[])
        .expect("the weight is outside the parameters");
}

/// A factory reads its parameters in one order whatever order the
/// document wrote their keys in, at every depth: the order of the
/// canonical bytes, which the digest reads. Two files that are one
/// document by their digest hand a factory the same map.
#[test]
fn a_factory_reads_its_parameters_in_key_order() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let heard = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("queue-price", move |build| {
            let written = serde_json::to_string(build.written()).expect("serializes");
            heard.lock().expect("no panic under the lock").push(written);
            Ok(Box::new(QueuePrice { per_request: 0 }))
        })
        .expect("a free name");
    let one = score_document(
        "[{ queue-price: { weight: 2, per_request: 500, costs: [{ unit: 17, flat: 3 }] } }]",
    );
    let other = score_document(
        "[{ queue-price: { costs: [{ flat: 3, unit: 17 }], per_request: 500, weight: 2 } }]",
    );
    assert_eq!(
        one.digest().expect("digests"),
        other.digest().expect("digests")
    );
    one.validate_in(&catalog, &[]).expect("validates");
    other.validate_in(&catalog, &[]).expect("validates");
    let seen = seen.lock().expect("no panic under the lock");
    assert_eq!(seen[0], seen[1], "two orders of one map");
    assert_eq!(
        seen[0],
        r#"{"costs":[{"flat":3,"unit":17}],"per_request":500}"#
    );
}

#[test]
fn a_bare_name_builds_from_an_empty_map_at_weight_one() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("high-id", |build| {
            assert!(build.written().is_empty());
            let typed: serde_json::Map<String, serde_json::Value> = build.params()?;
            assert!(typed.is_empty());
            Ok(Box::new(HighId))
        })
        .expect("a free name");
    for score in ["[high-id]", "[{ high-id: {} }]"] {
        let engine = score_document(score)
            .validate_in(&catalog, &[])
            .expect("the scorer needs no parameters");
        assert_eq!(engine.profiles()[0].scorers[0].weight, 1.0);
    }
}

#[test]
fn an_unknown_or_mistyped_parameter_is_refused_at_the_stage() {
    let catalog = price_catalog();
    for (score, message) in [
        ("[{ queue-price: { spare: 17 } }]", "unknown field `spare`"),
        ("[{ queue-price: { per_request: 'many' } }]", "invalid type"),
    ] {
        let error = score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("the parameter is refused");
        let SpecError::BadStage { path, reason } = error else {
            panic!("{error}");
        };
        assert_eq!(path, PRICE_PATH);
        assert!(reason.contains(message), "{reason}");
    }
}

#[test]
fn a_factory_can_refuse_a_value_at_its_field() {
    let error = score_document(BAD_PRICE)
        .validate_in(&price_catalog(), &[])
        .expect_err("a negative price is refused");
    assert_eq!(
        error,
        SpecError::BadField {
            path: PRICE_FIELD.into(),
            value: "-11".into(),
            reason: "above zero",
        }
    );
    assert_eq!(
        error.to_string(),
        format!("{PRICE_FIELD} = -11 is not above zero")
    );
}

#[test]
fn a_factory_refusal_names_the_field_to_edit() {
    let catalog = price_catalog();
    for (score, path) in [("[queue-price]", PRICE_PATH), (BAD_PRICE, PRICE_FIELD)] {
        let error = score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("the factory refuses the parameters");
        assert_eq!(error.field_path(), path);
    }
}

#[test]
fn a_refusal_displays_without_the_stages_path() {
    let field = crate::stages::StageError::bad("per_request", 0, "above zero");
    assert_eq!(field.to_string(), "per_request = 0 is not above zero");
    let other = crate::stages::StageError::other("no price list");
    assert_eq!(other.to_string(), "no price list");
    let error: &dyn std::error::Error = &other;
    assert!(error.source().is_none());
}

#[test]
fn a_held_name_is_refused_without_running_the_factory() {
    let mut catalog = price_catalog();
    let called = Arc::new(AtomicUsize::new(0));
    for (name, held) in [
        ("healthy", StageKind::Filter),
        ("least-requests", StageKind::Scorer),
        ("best", StageKind::Picker),
        ("queue-price", StageKind::Scorer),
    ] {
        let calls = Arc::clone(&called);
        assert_eq!(
            catalog
                .scorer_with(name, move |_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(Box::new(HighId))
                })
                .map(|_| ()),
            Err(StageRegistryError::Held { name, held })
        );
        assert_eq!(called.load(Ordering::Relaxed), 0);
    }
    assert_eq!(catalog.scorers().count(), SCORERS.len() + 1);
}

#[test]
fn a_factory_runs_only_for_valid_entries_that_name_it() {
    let mut catalog = StageRegistry::builtin();
    let called = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&called);
    catalog
        .scorer_with("high-id", move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(HighId))
        })
        .expect("a free name");
    assert_eq!(called.load(Ordering::Relaxed), 0);
    assert!(catalog.find_scorer("high-id").is_some());
    assert_eq!(called.load(Ordering::Relaxed), 0);
    score_document("[least-requests]")
        .validate_in(&catalog, &[])
        .expect("the document names another scorer");
    assert_eq!(called.load(Ordering::Relaxed), 0);

    let refused = |score: &str| {
        score_document(score)
            .validate_in(&catalog, &[])
            .expect_err("refused")
    };
    assert_eq!(
        refused("[{ absent: [17] }]"),
        SpecError::UnknownScorer {
            profile: "default".into(),
            name: "absent".into(),
        }
    );
    let path = "profiles.default.score[0].high-id";
    for (score, value) in [
        ("[{ high-id: [17] }]", "[17]"),
        ("[{ high-id: null }]", "null"),
    ] {
        assert_eq!(
            refused(score),
            SpecError::BadField {
                path: path.into(),
                value: value.into(),
                reason: "a map of the scorer's weight: { weight }",
            }
        );
    }
    for (weight, value) in [("-7", "-7"), ("9e70", "9e+70"), ("'many'", "\"many\"")] {
        let error = refused(&format!("[{{ high-id: {{ weight: {weight} }} }}]"));
        assert_eq!(
            error,
            SpecError::BadField {
                path: format!("{path}.weight"),
                value: value.into(),
                reason: "a finite number above zero",
            }
        );
    }
    assert_eq!(called.load(Ordering::Relaxed), 0);

    let named = score_document("[high-id, high-id, high-id]");
    for expected in [3, 6] {
        named.validate_in(&catalog, &[]).expect("validates again");
        assert_eq!(called.load(Ordering::Relaxed), expected);
    }
}

#[test]
fn each_build_checks_the_scorers_name() {
    let mut catalog = StageRegistry::builtin();
    let called = AtomicUsize::new(0);
    catalog
        .scorer_with(
            "high-id",
            move |_| -> Result<Box<dyn Scorer>, crate::stages::StageError> {
                if called.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(Box::new(HighId))
                } else {
                    Ok(Box::new(LowId))
                }
            },
        )
        .expect("a free name");
    let document = score_document("[high-id]");
    document
        .validate_in(&catalog, &[])
        .expect("the name agrees");
    let error = document
        .validate_in(&catalog, &[])
        .expect_err("the name disagrees");
    assert_eq!(
        error,
        SpecError::BadStage {
            path: "profiles.default.score[0].high-id".into(),
            reason: r#"scorer registered as "high-id" answers the name "low-id""#.into(),
        }
    );
}

#[test]
fn a_refused_profile_drops_the_stages_built_for_the_document() {
    #[derive(Debug)]
    struct CountDrop(Arc<AtomicUsize>);

    impl Lifecycle for CountDrop {}

    impl Scorer for CountDrop {
        fn name(&self) -> &'static str {
            "count-drop"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            HighId.score(facts, candidates, kept, out)
        }
    }

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let mut catalog = StageRegistry::builtin();
    let dropped = Arc::new(AtomicUsize::new(0));
    let drops = Arc::clone(&dropped);
    catalog
        .scorer_with("count-drop", move |_| {
            Ok(Box::new(CountDrop(Arc::clone(&drops))))
        })
        .and_then(|catalog| {
            catalog.scorer_with("refused", |_| {
                Err(crate::stages::StageError::other("no price"))
            })
        })
        .expect("two free names");
    let standing = score_document("[count-drop]")
        .validate_in(&catalog, &[])
        .expect("a valid document");
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    let version = crate::document::API_VERSION;
    let invalid: PolicySpec = serde_yaml::from_str(&format!(
        "apiVersion: {version}
profiles:
  before: {{ score: [count-drop], pick: best }}
  later: {{ score: [refused], pick: best }}
bindings: [{{ profile: before }}]
"
    ))
    .expect("a well-formed document");
    assert_eq!(
        invalid
            .validate_in(&catalog, &[])
            .expect_err("a later profile is refused"),
        SpecError::BadStage {
            path: "profiles.later.score[0].refused".into(),
            reason: "no price".into(),
        }
    );
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(
        standing.profiles()[0].scorers[0].scorer.name(),
        "count-drop"
    );
    drop(standing);
    assert_eq!(dropped.load(Ordering::Relaxed), 2);
}

#[test]
fn a_plain_scorer_reports_that_it_takes_only_weight() {
    let error = score_document("[{ high-id: { weight: 3.5, spread: 17 } }]")
        .validate_in(&catalog(), &[])
        .expect_err("the plain scorer accepts only weight");
    assert!(matches!(&error, SpecError::BadField { .. }));
    assert_eq!(
            error.to_string(),
            "profiles.default.score[0].high-id.spread = 17 is not a parameter of a scorer added in code, which takes its weight alone"
        );
}
