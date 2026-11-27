//! Cases of a gateway of many routers. Its default document binds three
//! sides ahead of single workers. The others hold what its two families,
//! pairs and single workers, share in one document or keep apart.

use super::*;

/// A gateway of many routers mounts its three-sided router in every
/// mode, and a model with an encode pool goes to it. So its default
/// document binds three sides over their phases, the encode side under
/// its default policy, and every other request behind them.
#[test]
fn the_default_document_of_many_routers_binds_three_sides_ahead_of_single_workers() {
    let document = many_routers(defaults())
        .policy_document()
        .expect("the default document");
    assert_eq!(
        serde_json::to_value(&document.bindings).expect("a written form"),
        json!([
            {
                "match": { "model": "*", "phase": ["encode", "prefill", "decode"] },
                "plan": "epd",
                "profiles": { "decode": "decode", "encode": "encode", "prefill": "prefill" },
            },
            { "match": { "model": "*", "phase": [] }, "profile": "random" },
        ]),
    );
    assert_eq!(
        document.profiles.keys().collect::<Vec<_>>(),
        ["decode", "encode", "prefill", "random"]
    );
    assert_eq!(
        document.profiles.get("encode"),
        preset("consistent_hashing")
            .profiles
            .get("consistent_hashing")
    );
    let applied = validated(&document).expect("the controller takes the document");
    let plan = |phase| {
        applied
            .engine()
            .binding("m", phase)
            .map(|binding| binding.plan)
    };
    for phase in [Phase::Encode, Phase::Prefill, Phase::Decode] {
        assert!(matches!(plan(phase), Some(Plan::Epd { .. })), "{phase:?}");
    }
    assert_eq!(plan(Phase::Single), Some(Plan::Single));
}

#[test]
fn many_routers_say_each_family_as_its_own_configuration_does() {
    for name in PRESETS.iter().copied().filter(|name| *name != "manual") {
        let regular = config(at_flag_defaults(name));
        let pair = RoutingSettings {
            mode: pair_mode(None, None),
            ..regular.clone()
        };
        let single = regular.policy_document().expect("the regular document");
        let paired = pair_alone(&pair).expect("the pair's own document");
        let both = pair
            .policy_document()
            .expect("the pair configuration's document");

        // The pair binding leads over its two phases. The single binding
        // stands behind it as the regular configuration writes it.
        let mut bindings = paired.bindings.clone();
        for binding in &mut bindings {
            binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
        }
        bindings.extend(single.bindings.clone());
        assert_eq!(both.bindings, bindings, "{name}");
        let mut profiles = paired.profiles.clone();
        profiles.extend(single.profiles.clone());
        assert_eq!(both.profiles, profiles, "{name}");
        assert_eq!(
            (&both.view, &both.index),
            (&single.view, &single.index),
            "{name}"
        );

        // A gateway of many routers says those two families with its
        // encode side ahead of them, under a regular configuration as
        // under a pair one with no side policy of its own. The pair's
        // two profiles are the pair plan's, and the controller takes
        // both documents.
        let three = with_encode_side(both.clone());
        for settings in [regular, pair] {
            assert_eq!(
                many_routers(settings).policy_document().as_ref(),
                Ok(&three),
                "{name}"
            );
        }
        assert_eq!(validated(&both).err(), None, "{name}");
        assert_eq!(validated(&three).err(), None, "{name}");
    }
}

#[test]
fn a_setting_of_the_binding_lands_on_both_bindings_of_many_routers() {
    let mut keyed = RoutingSettings {
        worker_overload_waiting_requests: Some(32),
        worker_overload_token_usage: Some(0.9),
        worker_overload_shed: true,
        pd_pairing_mode: Pairing::Strict,
        ..config(PolicySettings::ConsistentHashing)
    };
    keyed.routing_key_override.headers = vec!["x-user".to_string()];
    let single = keyed.policy_document().expect("the regular document");
    let (overload, key) = single
        .bindings
        .first()
        .map(|binding| (binding.overload.clone(), binding.routing_key.clone()))
        .expect("the regular binding");
    // The block states its shed word and no number: a threshold off its
    // default lands on no binding.
    let sheds = OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    };
    assert_eq!(overload, Some(sheds));
    assert!(key.is_some());

    let both = many_routers(keyed)
        .policy_document()
        .expect("the document of many routers");
    assert_eq!(both.bindings.len(), 2);
    for binding in &both.bindings {
        // One switch and one shed word stand for the pools of each
        // family, and each family reads a request's key from the same
        // sources.
        assert_eq!(binding.overload, overload);
        assert_eq!(binding.routing_key, key);
    }
    // The pairing mode is the pair plan's under a regular configuration too.
    assert_eq!(both.plans.pd_pair.pairing, Pairing::Strict);
}

#[test]
fn many_routers_write_the_view_and_index_of_a_main_policy_no_side_runs() {
    // Round robin places both sides of a pair and names no view and no
    // index. A tuned cache-aware policy places single workers: what it
    // shares with every side lands in the document all the same.
    let mut main = cache_aware(0.3, 64, 1.5, 60, 16_777_216);
    if let PolicySettings::CacheAware { block_size, .. } = &mut main {
        *block_size = 256;
    }
    let single = config(main.clone())
        .policy_document()
        .expect("the regular document");
    assert!(single.view.block_size.is_some() && !single.index.is_empty());
    let pairs = round_robin_pairs_under(main);
    let paired = pair_alone(&pairs).expect("the pair's own document");
    let both = many_routers(pairs.clone())
        .policy_document()
        .expect("the document of many routers");

    assert_eq!(both.bindings.len(), 2);
    assert_eq!(both.bindings.last(), single.bindings.first());
    assert_eq!(
        both.profiles.get("cache_aware"),
        single.profiles.get("cache_aware")
    );
    for side in ["prefill", "decode"] {
        assert_eq!(both.profiles.get(side), paired.profiles.get(side), "{side}");
    }
    assert_eq!((&both.view, &both.index), (&single.view, &single.index));
    assert_eq!(validated(&both).err(), None);
}

#[test]
fn many_routers_keep_the_manual_rule_on_the_single_binding_alone() {
    // The manual policy places single workers while round robin places
    // both sides of a pair. The session rule and the view's header are
    // the single binding's, and the pair binding holds no rule.
    let manual = at_flag_defaults("manual");
    let single = config(manual.clone())
        .policy_document()
        .expect("the regular document");
    let pairs = round_robin_pairs_under(manual);
    let both = many_routers(pairs.clone())
        .policy_document()
        .expect("the document of many routers");

    assert_eq!(both.bindings.len(), 2);
    let pair = both.bindings.first().expect("the pair binding");
    assert_eq!((&pair.sessions, &pair.routing_key), (&None, &None));
    let rule = both.bindings.last().expect("the single binding");
    assert!(rule.sessions.is_some());
    assert_eq!(Some(rule), single.bindings.first());
    assert_eq!(both.view, single.view);
    assert!(!both.view.headers.is_empty());
    assert_eq!(validated(&both).err(), None);
}

#[test]
fn many_routers_plan_pairs_alone_where_the_main_policy_names_another_shared_value() {
    // The view holds one block size. The sides agree on theirs and the
    // main policy names another, so single workers have no document and
    // the pair configuration plans its own family alone.
    let clash = RoutingSettings {
        mode: pair_mode(Some(cache_aware_over(16)), Some(cache_aware_over(16))),
        ..config(cache_aware_over(32))
    };
    let pairs_alone = pair_alone(&clash);
    assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
    assert_eq!(clash.policy_document(), pairs_alone);
    assert_eq!(many_routers(clash).policy_document(), pairs_alone);
}
