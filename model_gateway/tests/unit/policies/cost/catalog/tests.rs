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
