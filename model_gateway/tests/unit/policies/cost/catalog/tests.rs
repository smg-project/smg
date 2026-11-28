use super::*;

#[test]
fn every_name_builds_with_defaults() {
    for name in POLICY_NAMES {
        let policy = build(name, 0.0).expect("builds");
        assert_eq!(policy.name(), *name);
    }
}

#[test]
fn unknown_names_are_rejected() {
    for name in ["nope", "not-a-policy"] {
        assert!(
            matches!(build(name, 0.0), Err(CatalogError::Unknown(_))),
            "{name}"
        );
    }
}

/// No picker of the catalog acts on a request's end, and each of its
/// policies says so. A policy built over any other picker says it hears, so
/// that picker is told of every end as before.
#[test]
fn only_a_catalog_policy_says_its_picker_ignores_a_request_end() {
    use crate::policies::cost::{CandidateInputs, Needs, Pick, RequestInputs, WorkerPicker};

    #[derive(Debug)]
    struct AnyPicker;

    impl WorkerPicker for AnyPicker {
        fn pick(&self, _: &RequestInputs<'_>, _: &[CandidateInputs<'_>], _: &[f64]) -> Pick {
            Pick::None
        }
    }

    for name in POLICY_NAMES {
        let policy = build(name, 0.0).expect("builds");
        assert!(!policy.hears_request_complete(), "{name}");
    }

    let outside = WorkerSelectionPolicy::new(
        "outside",
        Needs::default(),
        Vec::new(),
        Vec::new(),
        Box::new(AnyPicker),
    );
    assert!(outside.hears_request_complete());
}
