//! Cases of a pair mode: the pair of presets it means, two sides that
//! name two values of one place, and the single request behind its pairs.

use super::*;

#[test]
fn a_pair_mode_is_the_pair_of_presets_under_its_pairing_rule() {
    let pair = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(PolicySettings::RoundRobin),
            decode_policy: None,
        },
        pd_pairing_mode: Pairing::Strict,
        ..config(at_flag_defaults("least_load"))
    };
    let mut expected =
        PolicySpec::pair_presets("round_robin", "least_load").expect("a pair of presets");
    expected.plans.pd_pair.pairing = Pairing::Strict;
    assert_eq!(
        pair.policy_document(),
        Ok(with_single(expected, preset("least_load"))),
        "a side with no policy of its own takes the main one, and the pairing mode is the plan's"
    );
}

#[test]
fn two_sides_that_name_two_values_of_one_place_are_refused() {
    let sides = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(cache_aware(0.3, 64, 1.5, 120, 67_108_864)),
            decode_policy: Some(cache_aware(0.3, 64, 1.5, 60, 67_108_864)),
        },
        ..config(PolicySettings::RoundRobin)
    };
    let (setting, value) = refusal(&sides);
    assert_eq!(
            setting, "cache_index, cache_ttl_secs, max_tree_size, eviction_interval_secs",
            "the index settings stand once under a model, so two sweep intervals have no document: {value}"
        );
    // Each side's value reads as the configuration spells it.
    assert_eq!(
        value,
        r#"["tree",180,67108864,120] and ["tree",180,67108864,60]"#
    );

    // Two block sizes: the view holds one.
    let sized = |blocks: usize| {
        let mut side = cache_aware(0.3, 64, 1.5, 120, 67_108_864);
        if let PolicySettings::CacheAware { block_size, .. } = &mut side {
            *block_size = blocks;
        }
        side
    };
    let blocks = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(sized(16)),
            decode_policy: Some(sized(32)),
        },
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        refusal(&blocks),
        ("block_size", "16 and 32".to_string()),
        "the view holds one block size, so two sides with two have no document"
    );
}

/// A gateway in the pair mode places one single request beside its
/// pairs: a token count takes one prefill worker, and the gateway's own
/// routing places it by the main policy. So the mode's document binds
/// the pair over its two phases and every other request behind it under
/// the main policy's profile, of one router as of many, and the
/// controller's check takes it. The pair's own document alone holds no
/// side for a single request.
#[test]
fn a_pair_mode_binds_a_single_request_behind_its_pairs_under_the_main_policy() {
    let settings = round_robin_pairs_under(at_flag_defaults("least_load"));
    let document = settings
        .policy_document()
        .expect("the pair mode's document");
    assert_eq!(
        serde_json::to_value(&document.bindings).expect("a written form"),
        json!([
            {
                "match": { "model": "*", "phase": ["prefill", "decode"] },
                "plan": "pd-pair",
                "profiles": { "decode": "decode", "prefill": "prefill" },
            },
            { "match": { "model": "*", "phase": [] }, "profile": "least_load" },
        ]),
    );
    // A gateway of many routers plans its encode side ahead of the
    // same two families.
    assert_eq!(
        many_routers(settings.clone()).policy_document(),
        Ok(with_encode_side(document.clone()))
    );

    // Whether a single request has no side, and whether its place is
    // the main policy's profile.
    let single_request = |document: &PolicySpec| {
        let applied = validated(document).expect("the controller takes the document");
        let place = applied.engine().place_for("", Phase::Single);
        let main = Place::NotAlone {
            profile: "least_load",
            in_the_way: InTheWay::Scorer("least-load"),
        };
        (place == Place::NoSide, place == main)
    };
    let alone = pair_alone(&settings).expect("the pair's own document");
    assert_eq!(single_request(&alone), (true, false));
    assert_eq!(single_request(&document), (false, true));
}

/// A pair mode whose main policy states a setting no document says
/// plans its pairs alone: the single binding is left out, of one router
/// as of many, and a single request has no side there.
#[test]
fn a_pair_mode_plans_pairs_alone_where_its_main_policy_has_no_document() {
    // Under the override the hash ring on both sides of the pair has a
    // document, and a main policy of `cache_aware` that names a
    // selection policy has none.
    let mut unsaid = at_flag_defaults("cache_aware");
    if let PolicySettings::CacheAware {
        selection_policy, ..
    } = &mut unsaid
    {
        *selection_policy = Some("another".to_string());
    }
    let mut keyed = RoutingSettings {
        mode: pair_mode(
            Some(PolicySettings::ConsistentHashing),
            Some(PolicySettings::ConsistentHashing),
        ),
        ..config(unsaid)
    };
    keyed.routing_key_override.enabled = true;
    let alone = pair_alone(&keyed).expect("the pair's own document");
    assert_eq!(keyed.policy_document().as_ref(), Ok(&alone));
    assert_eq!(alone.bindings.len(), 1);
    let applied = validated(&alone).expect("the controller takes the document");
    assert_eq!(applied.engine().place_for("", Phase::Single), Place::NoSide);

    // Under a main policy the override has a rule for, the same pair
    // binds its single request, and the rule is the single binding's.
    // The prefix ring is one of them.
    for main in ["random", "prefix_hash"] {
        keyed.policy = at_flag_defaults(main);
        let both = keyed
            .policy_document()
            .expect("the document of two families");
        let rules: Vec<bool> = both
            .bindings
            .iter()
            .map(|binding| binding.sessions.is_some())
            .collect();
        assert_eq!(rules, [false, true], "{main}");
        assert_eq!(validated(&both).err(), None, "{main}");
    }
}

/// The view holds one block size, and a prefix ring counts its key in
/// it without naming one. A main policy that names another size beside a
/// pair of prefix rings would end their key at another token (blocks of
/// 96 under a depth of 256) or leave it no whole block (blocks of 512),
/// and a pair that names the size does the same to a main policy's
/// ring. Either way a gateway of one router plans the pair alone, as the
/// pair's own document says it, and the controller's check takes it. A
/// gateway of many routers serves each family as a fleet of its own and
/// keeps the two bindings it had. At the ring's own size the single
/// request is bound.
#[test]
fn a_pair_mode_plans_pairs_alone_where_a_block_size_would_move_a_rings_key() {
    let ring = || at_flag_defaults("prefix_hash");
    let both_ways = |blocks: usize| {
        [
            RoutingSettings {
                mode: pair_mode(Some(ring()), Some(ring())),
                ..config(cache_aware_over(blocks))
            },
            RoutingSettings {
                mode: pair_mode(
                    Some(cache_aware_over(blocks)),
                    Some(cache_aware_over(blocks)),
                ),
                ..config(ring())
            },
        ]
    };
    for blocks in [96, 512] {
        for settings in both_ways(blocks) {
            let alone = pair_alone(&settings).expect("the pair's own document");
            assert_eq!(validated(&alone).err(), None, "{blocks}");
            assert_eq!(settings.policy_document().as_ref(), Ok(&alone), "{blocks}");
            assert_eq!(
                many_routers(settings)
                    .policy_document()
                    .map(|both| both.bindings.len()),
                Ok(2),
                "{blocks}"
            );
        }
    }
    for settings in both_ways(16) {
        let both = settings
            .policy_document()
            .expect("the document of two families");
        assert_eq!(both.bindings.len(), 2);
        assert_eq!(validated(&both).err(), None);
    }
    // The sides of one binding say one document together: a ring beside
    // a side that names the size counts in it, as it did, and the
    // single request is bound under a main policy that names none.
    let mixed = RoutingSettings {
        mode: pair_mode(Some(ring()), Some(cache_aware_over(96))),
        ..config(PolicySettings::RoundRobin)
    };
    let both = mixed
        .policy_document()
        .expect("the document of two families");
    assert_eq!((both.bindings.len(), both.view.block_size), (2, Some(96)));
    assert_eq!(validated(&both).err(), None);
}

/// Over gRPC or ZMQ a gateway in the pair mode counts tokens with no
/// worker. Its document of one router is then the pair's own: it binds
/// no single request, and reads nothing for the main policy. A gateway
/// of many routers serves single workers however it reaches them, and
/// over HTTP the single request is bound. A given document beside no
/// routing setting is in force over each.
#[test]
fn a_pair_mode_of_one_router_binds_a_single_request_over_http_alone() {
    let over = |connection_mode| RoutingSettings {
        connection_mode,
        ..round_robin_pairs_under(at_flag_defaults("cache_aware"))
    };
    let events = |document: &PolicySpec| {
        validated(document)
            .map(|applied| applied.needs().events)
            .ok()
    };
    let both = over(ConnectionModeSetting::Http)
        .policy_document()
        .expect("the document over HTTP");
    assert_eq!((both.bindings.len(), events(&both)), (2, Some(true)));

    let given = preset("round_robin");
    for connection_mode in [ConnectionModeSetting::Grpc, ConnectionModeSetting::Zmq] {
        let settings = over(connection_mode);
        let alone = pair_alone(&settings).expect("the pair's own document");
        assert_eq!(
            (alone.bindings.len(), events(&alone)),
            (1, Some(false)),
            "{connection_mode:?}"
        );
        assert_eq!(
            settings.policy_document().as_ref(),
            Ok(&alone),
            "{connection_mode:?}"
        );
        assert_eq!(
            many_routers(settings).policy_document(),
            Ok(with_encode_side(both.clone())),
            "{connection_mode:?}"
        );
        let unset = RoutingSettings {
            mode: pair_mode(None, None),
            connection_mode,
            policy_spec: Some(given.clone()),
            ..defaults()
        };
        assert_eq!(
            unset.policy_document_in_force(&defaults()),
            Ok(given.clone()),
            "{connection_mode:?}"
        );
    }
}
