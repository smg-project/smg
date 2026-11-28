//! Cases of a label, the policy a model's workers name for it. They hold
//! a label's own key and rule, where it adds no binding, its index, and
//! what it changes in the view. Refusals stand here too: the settings'
//! own under a label, a label that cannot stand beside the settings, and
//! a parting of the presets, which is no label's refusal.

use super::*;

/// The session rule and the key sources are each policy's own, either
/// way round. Under a main policy of `manual` a round robin label's
/// binding names no rule and no key, so its model's keyed requests take
/// turns. Under round robin a `manual` label brings the manual rule and
/// its header to the view, and a hash ring label its own sources.
#[test]
fn a_label_reads_its_own_key_and_rule_and_not_the_main_bindings() {
    let manual = config(at_flag_defaults("manual"));
    let own = manual.policy_document().expect("the manual document");
    let document = with_labels(&manual, &[("m", "round_robin")]).expect("a labelled one");
    assert_eq!(document.bindings[0].sessions, None);
    assert_eq!(document.bindings[0].routing_key, None);
    assert!(own.bindings[0].sessions.is_some());
    assert_eq!(document.bindings[1], own.bindings[0]);
    assert_eq!(document.view, own.view);

    let turns = config(at_flag_defaults("round_robin"));
    let own = turns.policy_document().expect("the round robin document");
    assert!(own.view.headers.is_empty());
    let document = with_labels(&turns, &[("m", "manual")]).expect("a labelled one");
    assert_eq!(
        document.bindings[0].sessions,
        preset("manual").bindings[0].sessions
    );
    assert_eq!(document.view.headers, ["x-smg-routing-key"]);
    assert_eq!(document.bindings[1], own.bindings[0]);
    let document = with_labels(&turns, &[("m", "consistent_hashing")]).expect("a labelled one");
    assert_eq!(
        document.bindings[0].routing_key,
        preset("consistent_hashing").bindings[0].routing_key
    );
    assert_eq!(document.bindings[0].sessions, None);
    assert_eq!(document.view, own.view);
}

/// Where the settings' document binds no single request, no label is
/// bound and the model's single requests stay refused: a pair mode of
/// one router over gRPC, the encode mode of one router, and a pair
/// planned alone for a main policy's setting no document says.
#[test]
fn a_label_adds_no_binding_where_no_single_request_is_bound() {
    let pairs = || round_robin_pairs_under(at_flag_defaults("least_load"));
    let mut unsaid = at_flag_defaults("cache_aware");
    if let PolicySettings::CacheAware {
        selection_policy, ..
    } = &mut unsaid
    {
        *selection_policy = Some("another".to_owned());
    }
    for (settings, sides) in [
        (
            RoutingSettings {
                connection_mode: ConnectionModeSetting::Grpc,
                ..pairs()
            },
            2,
        ),
        (
            RoutingSettings {
                mode: ModeSettings::EncodePrefillDecode {
                    encode_policy: None,
                    prefill_policy: None,
                    decode_policy: None,
                },
                ..config(at_flag_defaults("round_robin"))
            },
            3,
        ),
        (round_robin_pairs_under(unsaid), 2),
    ] {
        let own = settings
            .policy_document()
            .expect("a document of the mode alone");
        assert_eq!(own.bindings.len(), 1);
        assert_eq!(own.bindings[0].profiles.len(), sides);
        assert_eq!(with_labels(&settings, &[("m", "random")]), Ok(own));
        // A label's own id and name are checked here as everywhere.
        assert_eq!(
            with_labels(&settings, &[("m", "none")]),
            Err(label_error(
                "m",
                "none",
                &SpecError::UnknownPreset("none".to_owned()).to_string()
            ))
        );
        assert!(matches!(
            with_labels(&settings, &[("m*", "random")]),
            Err(RoutingSettingsError::Label { model, .. }) if model == "m*"
        ));
    }
    // What a label's binding would cost the document is checked only
    // where the binding is written: a ring another block size refuses
    // beside a single binding is passed over where there is none.
    let blocks = round_robin_pairs_under(cache_aware_over(32));
    assert!(matches!(
        with_labels(&blocks, &[("m", "prefix_hash")]),
        Err(RoutingSettingsError::Label { reason, .. }) if reason.starts_with("block_size = 16 and 32: ")
    ));
    let unbound = RoutingSettings {
        connection_mode: ConnectionModeSetting::Grpc,
        ..blocks
    };
    let own = unbound.policy_document().expect("the pair's document");
    assert_eq!(own.bindings.len(), 1);
    assert_eq!(with_labels(&unbound, &[("m", "prefix_hash")]), Ok(own));
}

/// Settings no document says are refused as before, whatever the
/// labels: the settings' own refusal comes first.
#[test]
fn settings_no_document_says_are_refused_as_before_under_a_label() {
    let rank = RoutingSettings {
        dp_minimum_tokens_scheduler: true,
        ..config(at_flag_defaults("round_robin"))
    };
    let mut sticky_ring = config(at_flag_defaults("consistent_hashing"));
    sticky_ring.routing_key_override.enabled = true;
    sticky_ring.routing_key_override.headers = vec!["x-session".to_owned()];
    for settings in [rank, sticky_ring] {
        let refusal = settings.policy_document().expect_err("no document");
        for named in [
            &[("m", "random")][..],
            &[("m*", "random")],
            &[("m", "none")],
        ] {
            assert_eq!(with_labels(&settings, named).as_ref(), Err(&refusal));
        }
    }
}

/// The builder's refusal of a setting under a label's policy is the
/// label's, by its model and its policy. A parting of the presets and
/// this module is not: it passes as it is, so a host that leaves a
/// refused label out does not leave it unsaid.
#[test]
fn a_parting_of_the_presets_is_no_labels_refusal() {
    let policy = PolicySettings::Random;
    let parted = || {
        [
            RoutingSettingsError::NoStage {
                policy: "random",
                stage: "picker",
            },
            RoutingSettingsError::NoPart {
                policy: "random",
                part: "profile",
            },
            RoutingSettingsError::Document(SpecError::UnknownPreset("random".to_owned())),
        ]
    };
    for (passed, parted) in parted().into_iter().zip(parted()) {
        assert_eq!(label_refused("m", &policy, passed), parted);
    }
    let setting = || unsupported("worker_warmup_secs", 7, "no document says it");
    assert_eq!(
        label_refused("m", &policy, setting()),
        label_error("m", "random", &setting().to_string())
    );
}

/// A label's policy that cannot stand beside the settings' own is
/// refused as two sides of the settings are, by the builder's own
/// sentence. The settings' document keeps its families: the refusal is
/// the label's, and no family is left out for it. A prefix ring keeps
/// the blocks of its own document, and the builder holds one list of
/// cache boundaries for the document.
///
/// A label's ring is held to the document's block size in a gateway of
/// many routers too, where the settings' own document stands as it
/// stood. Unrefused, that label's ring would have no whole block. The
/// controller's check would then refuse the document of every model,
/// with no label named.
#[test]
fn a_label_that_cannot_stand_beside_the_settings_is_refused_by_name() {
    let blocks = config(cache_aware_over(32));
    assert_eq!(
            with_labels(&blocks, &[("m", "prefix_hash")]),
            Err(label_error(
                "m",
                "prefix_hash",
                "block_size = 16 and 32: a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own"
            )),
        );
    let fleet = many_routers(config(cache_aware_over(512)));
    let own = fleet.policy_document().expect("the fleet's own document");
    assert_eq!(own.bindings.len(), 2);
    assert_eq!(
            with_labels(&fleet, &[("m", "prefix_hash")]),
            Err(label_error(
                "m",
                "prefix_hash",
                "block_size = 16 and 512: a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own"
            )),
        );
    assert_eq!(
        with_labels(&fleet, &[("m", "random")]).map(|document| document.bindings.len()),
        Ok(3)
    );

    let mut cut = at_flag_defaults("prefix_hash");
    if let PolicySettings::PrefixHash {
        cache_boundaries, ..
    } = &mut cut
    {
        *cache_boundaries = vec![64];
    }
    let cut = config(cut);
    assert!(cut.policy_document().is_ok());
    match with_labels(&cut, &[("m", "cache_aware")]) {
        Err(RoutingSettingsError::Label {
            model,
            policy,
            reason,
        }) => {
            assert_eq!((model.as_str(), policy.as_str()), ("m", "cache_aware"));
            assert!(reason.starts_with("cache_boundaries = "), "{reason}");
        }
        other => panic!("expected the label refused by its boundaries, got {other:?}"),
    }
}

/// A label's index stands under its model's own id, where it says
/// another thing than the document answers for that model without it.
/// So a `cache_aware` label builds beside a pair's `cache_aware` side
/// that names another index, and each model reads its own. Two sides of
/// the settings that name two indexes are refused, as before.
///
/// A label whose index is the one the settings wrote adds no entry, and
/// neither does a label of a policy that names no index. Under settings
/// that write none, each `cache_aware` label writes its model's entry.
#[test]
fn a_labels_index_stands_under_its_models_id() {
    let of_label = IndexSpec {
        max_size: 10_000,
        sweep_s: 30,
        ..IndexSpec::default()
    };
    let indexed =
        |prefill: Option<PolicySettings>, decode: Option<PolicySettings>| RoutingSettings {
            mode: pair_mode(prefill, decode),
            ..config(at_flag_defaults("round_robin"))
        };
    let at_flags = || Some(at_flag_defaults("cache_aware"));
    let as_a_label = || PolicySettings::of_label("cache_aware");
    let two_sides = indexed(at_flags(), as_a_label()).policy_document();
    assert!(two_sides.is_err(), "two sides of two indexes");

    let settings = indexed(at_flags(), None);
    let own = settings.policy_document().expect("the settings' document");
    assert_ne!(
        index_of(&own, "m"),
        of_label,
        "the pair's side names another"
    );
    let labels = [("m", "cache_aware"), ("n", "random")];
    let document = with_labels(&settings, &labels).expect("each model its own index");
    assert_eq!(document.index.get("m"), Some(&of_label));
    assert_eq!(document.index.get("n"), None, "`random` names no index");
    assert_eq!(document.index.get("*"), own.index.get("*"));
    assert_eq!(document.index.len(), own.index.len() + 1);
    assert_eq!(index_of(&document, "m"), of_label);
    assert_eq!(index_of(&document, "n"), index_of(&own, "n"));

    // A pair's side at a size of its own writes a `*` of another value.
    // The label's entry stands beside it, and each model reads its own.
    let sized = indexed(Some(cache_aware(0.3, 64, 1.5, 60, 1024)), None);
    let own = sized.policy_document().expect("the settings' document");
    let star = own.index.get("*").cloned().expect("a `*` of the settings");
    assert_ne!(star, of_label, "the pair's side names another");
    let document = with_labels(&sized, &[("m", "cache_aware")]).expect("beside a `*`");
    assert_eq!(document.index.get("*"), Some(&star));
    assert_eq!(document.index.get("m"), Some(&of_label));
    assert_eq!(index_of(&document, "n"), star);

    let same = indexed(as_a_label(), None);
    let own = same.policy_document().expect("the settings' document");
    assert_eq!(
        index_of(&own, "m"),
        of_label,
        "the settings wrote the label's"
    );
    let document = with_labels(&same, &[("m", "cache_aware")]).expect("a labelled one");
    assert_eq!(document.index, own.index);

    let none = config(at_flag_defaults("round_robin"));
    let labels = [("m", "cache_aware"), ("n", "cache_aware")];
    let document = with_labels(&none, &labels).expect("two labelled ones");
    assert_eq!(
        document.index.into_iter().collect::<Vec<_>>(),
        [
            ("m".to_owned(), of_label.clone()),
            ("n".to_owned(), of_label)
        ]
    );
}

/// What a label changes outside its own profile and binding. Under
/// settings that set nothing, `manual` alone changes the view, by its
/// header. `cache_aware` alone writes the index block: its model's own
/// entry, at the size and sweep a label builds it with. No label writes
/// a signal. Under the sticky override beside a main policy of the hash
/// ring, whose binding lists no header, a label the override places adds
/// the override's headers to the view.
#[test]
fn a_label_changes_the_view_for_its_headers_alone() {
    let mut changed = Vec::new();
    for name in PRESETS {
        let settings = config(main_beside(name));
        let own = settings.policy_document().expect("the settings' document");
        let document = with_labels(&settings, &[("m", name)]).expect("a labelled one");
        assert_eq!(document.signals, own.signals, "{name}");
        assert_eq!(document.plans, own.plans, "{name}");
        if document.view != own.view {
            let mut view = own.view.clone();
            view.headers.clone_from(&document.view.headers);
            assert_eq!(document.view, view, "{name}: the headers alone");
            changed.push(("view", name));
        }
        if document.index != own.index {
            changed.push(("index", name));
        }
    }
    assert_eq!(changed, [("view", "manual"), ("index", "cache_aware")]);
    let labelled = with_labels(
        &config(at_flag_defaults("round_robin")),
        &[("m", "cache_aware")],
    )
    .expect("a labelled one");
    let index = IndexSpec {
        max_size: 10_000,
        sweep_s: 30,
        ..IndexSpec::default()
    };
    assert_eq!(
        labelled.index.into_iter().collect::<Vec<_>>(),
        [("m".to_owned(), index)]
    );

    let mut ring = config(at_flag_defaults("consistent_hashing"));
    ring.routing_key_override.enabled = true;
    let own = ring.policy_document().expect("the ring's document");
    assert!(own.view.headers.is_empty());
    for name in PLACED.into_iter().chain(["prefix_hash"]) {
        let document = with_labels(&ring, &[("m", name)]).expect("a labelled one");
        assert_eq!(document.view.headers, ["x-smg-routing-key"], "{name}");
    }
}
