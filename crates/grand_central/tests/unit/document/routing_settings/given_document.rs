//! Cases of a document a gateway is given beside its routing settings:
//! which document is in force, and where the two disagree. The generation
//! numbers the document and is no part of the comparison. Beside settings
//! that state a policy, a kind in an otherwise equal document is refused
//! by name. A given document is compared with the document of many
//! routers and with a pair mode's two bindings too.

use super::*;

#[test]
fn with_no_document_the_flags_document_is_in_force() {
    let config = config(PolicySettings::RoundRobin);
    assert_eq!(
        config.policy_document_in_force(&defaults()),
        Ok(preset("round_robin"))
    );
}

#[test]
fn a_given_document_wins_where_no_routing_setting_is_set() {
    for (mode, document) in [
        (ModeSettings::Regular, preset("round_robin")),
        (
            ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            PolicySpec::pair_presets("round_robin", "random").expect("paired presets"),
        ),
    ] {
        let config = RoutingSettings {
            mode,
            policy_spec: Some(document.clone()),
            ..defaults()
        };
        assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
    }
}

#[test]
fn a_given_document_that_the_flags_mean_too_is_in_force() {
    let document = preset("round_robin");
    let config = RoutingSettings {
        policy_spec: Some(document.clone()),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
}

#[test]
fn the_generation_numbers_the_document_and_is_no_part_of_the_comparison() {
    for (written, runs) in [(0, 1), (1, 1), (7, 7)] {
        let mut document = preset("round_robin");
        document.generation = written;
        let mut in_force = document.clone();
        in_force.generation = runs;
        for settings in [defaults(), config(PolicySettings::RoundRobin)] {
            let config = RoutingSettings {
                policy_spec: Some(document.clone()),
                ..settings
            };
            assert_eq!(
                config.policy_document_in_force(&defaults()).as_ref(),
                Ok(&in_force),
                "generation {written}"
            );
        }
    }
}

#[test]
fn a_routing_flag_that_says_something_else_is_refused_with_both_values() {
    let mut document = preset("bucket");
    let profile = document.profiles.remove("bucket").expect("the profile");
    document.profiles.insert("power_of_two".to_owned(), profile);
    document.bindings[0].profile = Some("power_of_two".to_owned());
    let config = RoutingSettings {
        policy_spec: Some(document),
        ..config(at_flag_defaults("power_of_two"))
    };
    let refusal = config
        .policy_document_in_force(&defaults())
        .expect_err("different stages");
    assert_eq!(
        refusal.to_string(),
        r#"the routing flags and the policy document disagree at profiles.power_of_two.score[0]: the flags mean "least-load", the document says "bucket""#
    );
}

#[test]
fn a_side_policy_that_says_something_else_is_refused_with_both_values() {
    let config = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(PolicySettings::RoundRobin),
            decode_policy: None,
        },
        policy_spec: Some(PolicySpec::pair_presets("random", "random").expect("paired presets")),
        ..defaults()
    };
    assert_eq!(
        config.policy_document_in_force(&defaults()),
        Err(RoutingSettingsError::Disagreement {
            at: "profiles.prefill.pick".to_owned(),
            flags: r#""round-robin""#.to_owned(),
            document: r#""random""#.to_owned(),
        }),
    );
}

#[test]
fn a_part_one_side_lacks_reads_nothing() {
    let document = edited(preset("round_robin"), |form| {
        form["bindings"][0]["shadow"] = json!("round_robin");
    });
    let mut config = RoutingSettings {
        policy_spec: Some(document),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        config.policy_document_in_force(&defaults()),
        Err(RoutingSettingsError::Disagreement {
            at: "bindings[0].shadow".to_owned(),
            flags: "nothing".to_owned(),
            document: r#""round_robin""#.to_owned(),
        }),
    );

    let mut document = preset("round_robin");
    let binding = serde_json::to_string(&document.bindings[0]).expect("the binding serializes");
    document.bindings.clear();
    config.policy_spec = Some(document);
    assert_eq!(
        config.policy_document_in_force(&defaults()),
        Err(RoutingSettingsError::Disagreement {
            at: "bindings[0]".to_owned(),
            flags: binding,
            document: "nothing".to_owned(),
        }),
    );
}

#[test]
fn a_kind_in_an_otherwise_equal_document_is_refused_by_name() {
    let document = edited(preset("round_robin"), |form| {
        form["kind"] = json!("PolicySpec");
    });
    let config = RoutingSettings {
        policy_spec: Some(document),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        config.policy_document_in_force(&defaults()),
        Err(RoutingSettingsError::Document(SpecError::Kind(
            "PolicySpec".to_owned()
        )))
    );
}

#[test]
fn a_given_document_is_compared_with_the_document_of_many_routers() {
    let given = preset("round_robin");
    for mode in [ModeSettings::Regular, pair_mode(None, None)] {
        // No routing setting is set: the given document is in force.
        let unset = many_routers(RoutingSettings {
            mode: mode.clone(),
            policy_spec: Some(given.clone()),
            ..defaults()
        });
        assert_eq!(
            unset.policy_document_in_force(&defaults()),
            Ok(given.clone())
        );

        // A setting that says another document is refused, and one that
        // says the given document is not.
        let set = many_routers(RoutingSettings {
            mode,
            ..config(PolicySettings::RoundRobin)
        });
        let said = set.policy_document().expect("the flags' document");
        let other = RoutingSettings {
            policy_spec: Some(given.clone()),
            ..set.clone()
        };
        assert!(matches!(
            other.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement { .. })
        ));
        let same = RoutingSettings {
            policy_spec: Some(said.clone()),
            ..set
        };
        assert_eq!(same.policy_document_in_force(&defaults()), Ok(said));
    }
}

/// A pair mode of one router says two bindings, so a given document is
/// compared with both. Beside a stated policy, the pair's own document
/// is refused at the single request's profile, which it lacks, and the
/// document of two bindings is in force. Where no routing setting is
/// set, the pair's own document is in force.
#[test]
fn a_given_document_is_compared_with_the_two_bindings_of_a_pair_mode() {
    let stated = round_robin_pairs_under(at_flag_defaults("least_load"));
    let said = stated.policy_document().expect("the mode's document");
    let alone = pair_alone(&stated).expect("the pair's own document");
    assert_eq!((said.bindings.len(), alone.bindings.len()), (2, 1));
    let given = |document: &PolicySpec, settings: &RoutingSettings| RoutingSettings {
        policy_spec: Some(document.clone()),
        ..settings.clone()
    };

    let refused = given(&alone, &stated).policy_document_in_force(&defaults());
    assert!(
        matches!(
            &refused,
            Err(RoutingSettingsError::Disagreement { at, document, .. })
                if at == "profiles.least_load" && document == "nothing"
        ),
        "{refused:?}"
    );
    assert_eq!(
        given(&said, &stated).policy_document_in_force(&defaults()),
        Ok(said)
    );
    let unset = RoutingSettings {
        mode: pair_mode(None, None),
        ..defaults()
    };
    assert_eq!(
        given(&alone, &unset).policy_document_in_force(&defaults()),
        Ok(alone)
    );
}
