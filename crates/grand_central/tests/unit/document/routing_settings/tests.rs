use serde_json::json;

use super::{translation::label_refused, *};
use crate::{
    document::{
        applied::AppliedSpec,
        spec::{BindingSpec, IndexSpec, OverloadSpec, PlanName, Recovery, PRESETS},
    },
    engine::{
        framework::{InTheWay, Place, Plan},
        view::Phase,
    },
    policies::cache_aware::Gates,
    stages::StageRegistry,
};

mod given_document;
mod labels;
mod many_routers;
mod overload;
mod pair_mode;
mod policy_settings;
mod routing_key_override;

/// The settings of a gateway that sets none, as its configuration
/// defaults them. The document they mean is the `random` preset.
fn defaults() -> RoutingSettings {
    RoutingSettings::default()
}

/// The cache-aware policy with these five settings, and every other at
/// the value a gateway's configuration file gets where it leaves the
/// setting out.
fn cache_aware(
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
) -> PolicySettings {
    PolicySettings::CacheAware {
        cache_threshold,
        balance_abs_threshold,
        balance_rel_threshold,
        balance_prefix_margin_tokens: 0,
        balance_uncached_credit: false,
        eviction_interval_secs,
        max_tree_size,
        block_size: 16,
        balance_token_usage_threshold: 1.0,
        overload_token_usage_threshold: 1.0,
        overlap_decay: 0.0,
        selection_temperature: 0.0,
        cache_index: IndexKind::Tree,
        cache_ttl_secs: 180,
        cache_boundaries: Vec::new(),
        selection_policy: None,
        selection_accounting_ttl_ms: 0,
    }
}

/// The check a gateway passes a document through before it runs it, with
/// the header a gateway keeps to itself denied to the view.
fn validated(document: &PolicySpec) -> Result<AppliedSpec, SpecError> {
    AppliedSpec::validate_with(document, &["x-smg-tenant-id"])
}

fn config(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        policy,
        ..defaults()
    }
}

/// The policy of a preset's name with every setting at its flag's
/// default.
fn at_flag_defaults(name: &str) -> PolicySettings {
    match name {
        "random" => PolicySettings::Random,
        "round_robin" => PolicySettings::RoundRobin,
        "passthrough" => PolicySettings::Passthrough,
        "consistent_hashing" => PolicySettings::ConsistentHashing,
        "cache_aware" => cache_aware(0.3, 64, 1.5, 120, 67_108_864),
        "power_of_two" => PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::ExpectedWait,
        },
        "least_load" => PolicySettings::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        },
        "bucket" => PolicySettings::Bucket {
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        },
        "manual" => PolicySettings::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode: Assignment::Random,
        },
        "prefix_hash" => PolicySettings::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        },
        other => {
            panic!("the controller has a preset this module converts no policy to: {other}")
        }
    }
}

fn preset(name: &str) -> PolicySpec {
    PolicySpec::preset(name).expect("a preset of this build")
}

/// A document with places of its JSON form written over.
fn edited(document: PolicySpec, edit: impl FnOnce(&mut Value)) -> PolicySpec {
    let mut form = serde_json::to_value(document).expect("a document serializes");
    edit(&mut form);
    serde_json::from_value(form).expect("the edited document")
}

fn refusal(config: &RoutingSettings) -> (&'static str, String) {
    match config.policy_document() {
        Err(RoutingSettingsError::UnsupportedSetting { setting, value, .. }) => (setting, value),
        other => panic!("expected a setting refused by name, got {other:?}"),
    }
}

#[test]
fn an_encode_side_with_no_policy_is_placed_by_consistent_hashing() {
    let encode_pair = RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: None,
            prefill_policy: None,
            decode_policy: Some(PolicySettings::Random),
        },
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        encode_pair.policy_document(),
        Ok(
            PolicySpec::epd_presets("consistent_hashing", "round_robin", "random")
                .expect("the encode pair's presets")
        ),
    );
}

#[test]
fn the_manual_policy_writes_the_bindings_session_rule() {
    let manual = PolicySettings::Manual {
        eviction_interval_secs: 30,
        max_idle_secs: 3600,
        assignment_mode: Assignment::MinGroup,
    };
    let rule = |document: &mut Value| {
        let sessions = &mut document["bindings"][0]["sessions"];
        sessions["ttl_s"] = json!(3600);
        sessions["sweep_s"] = json!(30);
        sessions["assignment"] = json!("min_group");
    };
    assert_eq!(
        config(manual.clone()).policy_document(),
        Ok(edited(preset("manual"), rule))
    );

    // Either time at zero sweeps nothing: a key stays pinned for ever.
    let pinned = config(PolicySettings::Manual {
        eviction_interval_secs: 30,
        max_idle_secs: 0,
        assignment_mode: Assignment::Random,
    })
    .policy_document()
    .expect("a document");
    let for_ever = |document: &mut Value| {
        let sessions = &mut document["bindings"][0]["sessions"];
        sessions["ttl_s"] = json!(u32::MAX);
        sessions["sweep_s"] = json!(u32::MAX);
    };
    assert_eq!(pinned, edited(preset("manual"), for_ever));
    assert!(pinned.validate().is_ok(), "the controller takes the rule");

    // The sweep at zero beside an idle time: the gateway starts no
    // sweep, so the key stays pinned for ever too.
    let unswept = config(PolicySettings::Manual {
        eviction_interval_secs: 0,
        max_idle_secs: 3600,
        assignment_mode: Assignment::Random,
    })
    .policy_document()
    .expect("a document");
    assert_eq!(unswept, edited(preset("manual"), for_ever));

    // A pair of manual sides: the pair of presets names no rule, and the
    // manual preset's own is the one written. The single binding behind
    // the pair is the regular document's, under the same rule.
    let mut pair = PolicySpec::pair_presets("manual", "manual").expect("a pair of presets");
    pair.bindings[0].sessions = preset("manual").bindings[0].sessions.clone();
    let both = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: None,
        },
        ..config(manual.clone())
    };
    assert_eq!(
        both.policy_document(),
        Ok(with_single(
            edited(pair, rule),
            edited(preset("manual"), rule)
        ))
    );

    let one_side = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: Some(PolicySettings::RoundRobin),
        },
        ..config(manual)
    };
    assert_eq!(
            refusal(&one_side),
            ("policy", "manual".to_string()),
            "one session rule serves the whole binding, so a manual side beside another policy is refused"
        );
}

/// The policies under which the sticky override lands as a session
/// rule: each places a single request by nothing its key hands it. The
/// gateway's answers with the override are recorded under the first
/// two.
const PLACED: [&str; 7] = [
    "random",
    "power_of_two",
    "round_robin",
    "least_load",
    "passthrough",
    "bucket",
    "cache_aware",
];

/// The policy of a preset's name at its flags' defaults, under the
/// sticky override.
fn sticky(policy: &str) -> RoutingSettings {
    let mut sticky = config(at_flag_defaults(policy));
    sticky.routing_key_override.enabled = true;
    sticky
}

/// The override's own settings land in its rule. A key's first worker
/// is drawn by the assignment, and the pins' order follows it: the
/// latest worker leads under `delegate`, the first under the sticky
/// map's own three. Either time at zero pins a key for ever. Each
/// configured header is a source, lowercase and once, ahead of the
/// request's id, and with none configured the id is the one source.
#[test]
fn the_overrides_settings_land_in_its_session_rule() {
    let written = |set: &dyn Fn(&mut RoutingKeyOverrideSettings)| {
        let mut sticky = sticky("random");
        set(&mut sticky.routing_key_override);
        let document = sticky.policy_document().expect("a document");
        assert!(validated(&document).is_ok(), "{document:?}");
        let binding = document.bindings.into_iter().next().expect("the binding");
        let rule = binding.sessions.expect("the override's rule");
        let key = binding.routing_key.expect("the key's sources");
        (rule, key, document.view.headers)
    };
    for (assignment, recovery) in [
        (Assignment::Delegate, Recovery::Latest),
        (Assignment::Random, Recovery::Original),
        (Assignment::MinLoad, Recovery::Original),
        (Assignment::MinGroup, Recovery::Original),
    ] {
        let (rule, ..) = written(&|sticky| sticky.assignment_mode = assignment);
        assert_eq!((rule.assignment, rule.recovery), (assignment, recovery));
        // The rule reads nothing of the policy beside it.
        let mut other = sticky("power_of_two");
        other.routing_key_override.assignment_mode = assignment;
        let document = other.policy_document().expect("a document");
        assert!(validated(&document).is_ok(), "{document:?}");
        let beside = document.bindings[0].sessions.as_ref();
        assert_eq!(beside, Some(&rule), "{assignment:?} beside power_of_two");
    }

    let times = |idle, interval| {
        let (rule, ..) = written(&|sticky| {
            sticky.max_idle_secs = idle;
            sticky.eviction_interval_secs = interval;
        });
        (rule.ttl_s, rule.sweep_s)
    };
    assert_eq!(times(600, 30), (600, 30));
    assert_eq!(times(0, 30), (u32::MAX, u32::MAX));
    assert_eq!(times(600, 0), (u32::MAX, u32::MAX));
    assert_eq!(times(0, 0), (u32::MAX, u32::MAX));
    let mut past = sticky("random");
    past.routing_key_override.max_idle_secs = u64::from(u32::MAX);
    assert_eq!(
        refusal(&past),
        ("routing_key_override.max_idle_secs", u32::MAX.to_string())
    );
    past.routing_key_override.max_idle_secs = 600;
    past.routing_key_override.eviction_interval_secs = u64::from(u32::MAX);
    assert_eq!(
        refusal(&past).0,
        "routing_key_override.eviction_interval_secs"
    );

    let sources = |headers: &[&str]| {
        let (_, key, view) = written(&|sticky| {
            sticky.headers = headers.iter().map(|name| (*name).to_string()).collect();
        });
        assert!(key.strip_lineage);
        let read: Vec<&str> = key
            .sources
            .iter()
            .map(|source| source.header.as_deref().unwrap_or("rid"))
            .collect();
        (read.join(" "), view.join(" "))
    };
    assert_eq!(
        sources(&["X-Session-Id", "x-smg-routing-key", "x-session-id"]),
        (
            "x-session-id x-smg-routing-key rid".to_string(),
            "x-session-id x-smg-routing-key".to_string()
        )
    );
    assert_eq!(sources(&[]), ("rid".to_string(), String::new()));
}

/// The override's rule has no document in two places. Each is refused
/// by the setting's name, in a sentence that says which it is: a plan
/// with an encode side, and a gateway of many routers. So under every
/// policy the rule lands beside, the prefix ring among them.
#[test]
fn the_override_is_refused_where_its_rule_has_no_document() {
    let why = |config: &RoutingSettings| match config.policy_document() {
        Err(RoutingSettingsError::UnsupportedSetting {
            setting: "routing_key_override.enabled",
            reason,
            ..
        }) => reason,
        other => panic!("expected the override refused by name, got {other:?}"),
    };
    for name in PLACED.into_iter().chain(["prefix_hash"]) {
        let encode = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            ..sticky(name)
        };
        assert_eq!(
            why(&encode),
            "the sticky sessions' rule places a single request or the two sides of a pair, \
                 and this plan has an encode side",
            "{name}"
        );
        assert_eq!(
            why(&many_routers(sticky(name))),
            "the sticky sessions' rule has no document in a gateway of many routers yet",
            "{name}"
        );
    }
}

/// A pair may hold a side under `consistent_hashing`, which keeps a
/// routing key's handling to itself. The override then lands as the
/// session rule of the other side alone (`sessions.sides`), and the hash
/// side stays its own policy's.
/// The rule is the one the binding of single requests holds, with the
/// side named. A gateway's check takes the document, and its engine's
/// rule covers the one side.
///
/// The gateway's sticky map places a role at a time. Beside a side
/// under the hash policy it places the other side alone
/// (`model_gateway/src/policies/registry.rs`, `sticky_for`).
#[test]
fn the_overrides_rule_names_the_side_it_places_beside_a_side_that_reads_the_key() {
    use crate::engine::SessionSides;

    let ring = || Some(PolicySettings::ConsistentHashing);
    for name in PLACED.into_iter().chain(["prefix_hash"]) {
        let mixed = [
            (pair_mode(ring(), None), "decode", SessionSides::Decode),
            (pair_mode(None, ring()), "prefill", SessionSides::Prefill),
        ];
        for (mode, placed, covered) in mixed {
            let pair = RoutingSettings {
                mode,
                ..sticky(name)
            };
            let document = pair.policy_document().expect("the pair's document");
            let form = serde_json::to_value(&document).expect("a document serializes");
            // The pair's binding, then the binding of single requests.
            let bindings = form["bindings"].as_array().expect("the bindings");
            let [paired, single] = bindings.as_slice() else {
                panic!("{name}: a pair's binding and a single's, got {bindings:?}");
            };
            assert_eq!(paired["plan"], "pd-pair", "{name}");
            assert_eq!(single["sessions"].get("sides"), None, "{name}");
            let mut named = single["sessions"].clone();
            named["sides"] = json!([placed]);
            assert_eq!(paired["sessions"], named, "{name}: the {placed} side");
            let applied = validated(&document).expect("a gateway's check takes the document");
            let engine = applied.engine();
            let rule = engine.binding("", Phase::Prefill);
            let rule = rule.and_then(|binding| binding.sessions);
            assert_eq!(rule.map(|rule| rule.sides), Some(covered), "{name}");
            assert!(engine.needs().session_in_flight, "{name}");
        }
    }
}

/// A mixed pair's document stands where no binding of the override is
/// written behind the pair's. Two configurations have none: the hash
/// policy as the main policy, and a gateway of one router over gRPC or
/// ZMQ, which plans the pair alone. The view lists the session's headers
/// alone, as under the override on a single binding. The hash side's
/// client sources stay in the binding's own list, where its ring reads
/// them, and a gateway's check takes the document.
#[test]
fn a_mixed_pairs_view_lists_no_client_header_where_no_single_rule_stands_behind_it() {
    let placed = || Some(at_flag_defaults("round_robin"));
    // The hash policy is the main one, and the prefill side's with it:
    // the binding of single requests holds no rule.
    let under_the_ring = RoutingSettings {
        mode: pair_mode(None, placed()),
        ..sticky("consistent_hashing")
    };
    // One router over gRPC or ZMQ binds no single request.
    let alone = |connection_mode| RoutingSettings {
        connection_mode,
        mode: pair_mode(Some(PolicySettings::ConsistentHashing), None),
        ..sticky("round_robin")
    };
    let single = sticky("round_robin").policy_document();
    let headers = single.expect("the single binding's document").view.headers;
    assert!(!headers.is_empty());
    for (settings, bindings) in [
        (under_the_ring, 2),
        (alone(ConnectionModeSetting::Grpc), 1),
        (alone(ConnectionModeSetting::Zmq), 1),
    ] {
        let document = settings.policy_document().expect("the pair's document");
        assert_eq!(document.bindings.len(), bindings);
        let paired = &document.bindings[0];
        let rule = paired.sessions.as_ref().expect("the pair's rule");
        assert_eq!(rule.sides, Some(vec!["decode".to_owned()]));
        let rules = document.bindings.iter().filter(|b| b.sessions.is_some());
        assert_eq!(rules.count(), 1, "the pair's rule alone");
        assert_eq!(document.view.headers, headers);
        let key = paired.routing_key.as_ref().expect("the pair's key");
        assert!(key.sources.iter().any(|source| source.client));
        validated(&document).expect("a gateway's check takes the document");
    }
}

/// The settings name a side in a rule for that pair alone. Every other
/// document they write names none: no rule of it holds a `sides` key. The
/// case covers each preset's policy at its flags' defaults, alone and on
/// both sides of a pair, with the override on and with it off. The
/// settings write each of them: one they refused would fail the case.
#[test]
fn the_settings_name_no_side_in_any_other_document() {
    for name in PRESETS {
        for (paired, overridden) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut settings = config(at_flag_defaults(name));
            settings.routing_key_override.enabled = overridden;
            if paired {
                settings.mode = pair_mode(None, None);
            }
            let case = format!("{name}, a pair {paired}, the override {overridden}");
            let document = settings.policy_document().expect(&case);
            let rules = document.bindings.iter().filter_map(|b| b.sessions.as_ref());
            assert!(rules.clone().all(|rule| rule.sides.is_none()), "{case}");
            let text = serde_json::to_string(&document).expect("a document serializes");
            assert!(!text.contains("\"sides\""), "{case}: {text}");
        }
    }
}

/// Beside a pair whose two sides read no key the override lands as the
/// same session rule on the pair's binding, over the same sources, and
/// on the binding of single requests behind it. Nothing else of the
/// pair mode's document moves. The gateway's sticky map places each
/// role of a pair under the role's own key
/// (`model_gateway/src/policies/registry.rs`, `select_sticky`), and the
/// binding's rule places each side by its own session. A gateway's
/// check takes the document, where each side of a pick states its own
/// word of who placed it, and its engine asks its host for the count of
/// a key's requests on a worker.
#[test]
fn the_override_lands_as_a_session_rule_on_a_pair_whose_sides_read_no_key() {
    let paired = |settings: RoutingSettings| RoutingSettings {
        mode: pair_mode(None, None),
        ..settings
    };
    for name in PLACED.into_iter().chain(["prefix_hash"]) {
        let plain = paired(config(at_flag_defaults(name))).policy_document();
        let expected = edited(plain.expect("the pair mode's document"), |document| {
            document["view"]["headers"] = json!(["x-smg-routing-key"]);
            for binding in document["bindings"].as_array_mut().expect("the bindings") {
                binding["routing_key"] = json!({
                    "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                    "strip_lineage": true
                });
                binding["sessions"] = json!({
                    "ttl_s": 14_400,
                    "sweep_s": 60,
                    "assignment": "delegate",
                    "inflight_cap": 2,
                    "recovery": "latest",
                    "rule_placed": "untold"
                });
            }
        });
        assert_eq!(expected.bindings.len(), 2, "{name}");
        assert_eq!(
            paired(sticky(name)).policy_document().as_ref(),
            Ok(&expected),
            "{name}"
        );
        let applied = validated(&expected).expect("a gateway's check takes the document");
        assert!(applied.engine().needs().session_in_flight, "{name}");
    }
}

#[test]
fn the_sides_of_a_plan_share_one_list_of_key_sources() {
    let pair = |prefill: &str, decode: &str| RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(at_flag_defaults(prefill)),
            decode_policy: Some(at_flag_defaults(decode)),
        },
        pd_pairing_mode: Pairing::Strict,
        ..config(PolicySettings::RoundRobin)
    };

    // One keyed side names the binding's sources, whichever side it is.
    for (prefill, decode) in [
        ("consistent_hashing", "least_load"),
        ("least_load", "consistent_hashing"),
        ("prefix_hash", "prefix_hash"),
        ("prefix_hash", "consistent_hashing"),
    ] {
        let mut expected = PolicySpec::pair_presets(prefill, decode).expect("a pair of presets");
        expected.plans.pd_pair.pairing = Pairing::Strict;
        assert_eq!(
            pair(prefill, decode).policy_document(),
            Ok(with_single(expected, preset("round_robin"))),
            "{prefill} and {decode}"
        );
    }

    // Two sides that read two lists are refused by their policies: with
    // no header configured the prefix ring reads none, and the hash ring
    // still reads the routing key header.
    let mut bare = pair("prefix_hash", "consistent_hashing");
    bare.routing_key_override.headers = Vec::new();
    assert_eq!(
        refusal(&bare),
        ("policy", "prefix_hash and consistent_hashing".to_string())
    );

    // The manual policy's own refusal comes first.
    assert_eq!(
        refusal(&pair("manual", "consistent_hashing")),
        ("policy", "manual".to_string())
    );
}

#[test]
fn a_setting_no_document_says_is_refused_by_its_name() {
    let round_robin = || config(PolicySettings::RoundRobin);
    // The override's rule has no document in a gateway of many routers
    // yet.
    let many = many_routers(sticky("random"));
    let mut selection = config(at_flag_defaults("cache_aware"));
    if let PolicySettings::CacheAware {
        selection_policy, ..
    } = &mut selection.policy
    {
        *selection_policy = Some("another".to_string());
    }
    for (refused, setting, value) in [
        (many, "routing_key_override.enabled", "true"),
        (
            RoutingSettings {
                dp_minimum_tokens_scheduler: true,
                ..round_robin()
            },
            "dp_minimum_tokens_scheduler",
            "true",
        ),
        (
            RoutingSettings {
                mode: ModeSettings::Provider("a_provider"),
                ..round_robin()
            },
            "mode",
            "a_provider",
        ),
        (selection, "selection_policy", "another"),
    ] {
        assert_eq!(refusal(&refused), (setting, value.to_string()));
    }
}

/// What a host asks ahead of the document: a provider mode of a gateway
/// of one router is named, and no other configuration is.
#[test]
fn a_provider_mode_of_one_router_is_named_alone() {
    let named = |mode: ModeSettings, enable_igw: bool| {
        RoutingSettings {
            mode,
            enable_igw,
            ..config(PolicySettings::RoundRobin)
        }
        .provider_alone()
    };
    let provider = || ModeSettings::Provider("a_provider");
    assert_eq!(named(provider(), false), Some("a_provider"));
    // Under the multi-router setting the document plans the gateway's
    // self-hosted workers.
    assert_eq!(named(provider(), true), None);
    for many in [false, true] {
        let pair = ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: None,
        };
        let encode = ModeSettings::EncodePrefillDecode {
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        };
        for mode in [ModeSettings::Regular, pair, encode] {
            assert_eq!(named(mode, many), None);
        }
    }
}

#[test]
fn the_pin_to_an_engine_of_a_grouped_worker_moves_no_document() {
    // The dispatch pins a sticky key behind the decision, on the worker
    // the decision names: the document is the same with the pin on.
    let pinned = RoutingSettings {
        grouped_dp_sticky: Some("hash"),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(pinned.policy_document(), Ok(preset("round_robin")));
    assert_eq!(
        pinned.policy_document_in_force(&defaults()),
        Ok(preset("round_robin"))
    );
    // The pin reads the gateway's own key, and a supplied document names
    // its own key sources: beside one the pin is refused by its name.
    let given = RoutingSettings {
        grouped_dp_sticky: Some("hash"),
        policy_spec: Some(preset("least_load")),
        ..defaults()
    };
    assert!(matches!(
        given.policy_document_in_force(&defaults()),
        Err(RoutingSettingsError::UnsupportedSetting {
            setting: "grouped_dp_sticky",
            ..
        })
    ));
}

/// `config` as a gateway of many routers runs it.
fn many_routers(config: RoutingSettings) -> RoutingSettings {
    RoutingSettings {
        enable_igw: true,
        ..config
    }
}

fn pair_mode(prefill: Option<PolicySettings>, decode: Option<PolicySettings>) -> ModeSettings {
    ModeSettings::PrefillDecode {
        prefill_policy: prefill,
        decode_policy: decode,
    }
}

/// `both`, a document of pairs and single requests, with the encode
/// side a gateway of many routers plans ahead of them: the side's
/// default profile, and the pair's binding under the plan of three
/// sides, over the encode phase too. Nothing else of the document
/// differs: the pair's two profiles are the pair plan's.
fn with_encode_side(mut both: PolicySpec) -> PolicySpec {
    let ring = preset("consistent_hashing")
        .profiles
        .remove("consistent_hashing")
        .expect("the preset's profile");
    both.profiles.insert("encode".to_string(), ring);
    let sides = both.bindings.first_mut().expect("the pair's binding");
    sides.plan = Some(PlanName::Epd);
    sides
        .profiles
        .insert("encode".to_string(), "encode".to_string());
    sides.matcher.phase.insert(0, Phase::Encode);
    both
}

/// The document `settings` said before a gateway of many routers
/// planned three sides: the plan of two families where it stands, and
/// the mode's own behind it.
fn before_three_sides(settings: &RoutingSettings) -> Result<PolicySpec, RoutingSettingsError> {
    refuse_unsupported(settings)?;
    if let Some(both) = both_families(settings)? {
        let encode_mode = matches!(settings.mode, ModeSettings::EncodePrefillDecode { .. });
        match written(settings, both) {
            Err(RoutingSettingsError::UnsupportedSetting { .. }) => {}
            Ok(both) if encode_mode && both.validate().is_err() => {}
            said => return said,
        }
    }
    written(settings, presets(settings)?)
}

/// The pair's own document alone, as `presets` plans it: what a pair
/// mode says where its main policy's single binding has no document.
fn pair_alone(config: &RoutingSettings) -> Result<PolicySpec, RoutingSettingsError> {
    presets(config).and_then(|planned| written(config, planned))
}

/// A document of two families: `pairs`, a pair's own document, bound over
/// the two phases of a pair, and behind it the binding and the profile
/// of `single`, the main policy's regular document.
fn with_single(mut pairs: PolicySpec, single: PolicySpec) -> PolicySpec {
    for binding in &mut pairs.bindings {
        binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
    }
    pairs.profiles.extend(single.profiles);
    pairs.bindings.extend(single.bindings);
    pairs
}

/// A gateway of one router and the encode mode plan no third side of
/// this kind: each says, to the byte, the document it said before a
/// gateway of many routers planned three sides. One router mounts the
/// three-sided router under the encode mode alone, and the encode
/// mode's plan holds its three sides already.
#[test]
fn one_router_and_the_encode_mode_say_the_document_they_said() {
    let named = |name: &str| Some(at_flag_defaults(name));
    let (mut said, mut manual_pairs) = (0, 0);
    for name in PRESETS {
        let modes = [
            ModeSettings::Regular,
            pair_mode(None, None),
            pair_mode(named(name), named("round_robin")),
            ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            ModeSettings::EncodePrefillDecode {
                encode_policy: named("round_robin"),
                prefill_policy: named(name),
                decode_policy: named("round_robin"),
            },
        ];
        for mode in modes {
            let encode_mode = matches!(mode, ModeSettings::EncodePrefillDecode { .. });
            for connection_mode in [
                ConnectionModeSetting::Http,
                ConnectionModeSetting::Grpc,
                ConnectionModeSetting::Zmq,
            ] {
                let one_router = RoutingSettings {
                    mode: mode.clone(),
                    connection_mode,
                    ..config(at_flag_defaults(name))
                };
                let mut kinds = vec![one_router.clone()];
                if encode_mode {
                    kinds.push(many_routers(one_router));
                }
                for settings in kinds {
                    let row = format!("{name}, {:?}, {connection_mode:?}", settings.mode);
                    let (mode, main) = (&settings.mode, &settings.policy);
                    let three = three_sided(mode, main, settings.enable_igw);
                    assert!(matches!(three, Ok(None)), "{row}");
                    let document = settings.policy_document();
                    assert_eq!(document, before_three_sides(&settings), "{row}");
                    said += usize::from(document.is_ok());
                    let manual = encode_mode && name == "manual";
                    manual_pairs += usize::from(manual && document.is_ok());
                }
            }
        }
    }
    assert_eq!(said, 201, "documents were said");
    // The encode mode under the manual policy on its prefill and its
    // decode says six of them: of one router and of many, over each
    // connection mode.
    assert_eq!(manual_pairs, 6);
}

/// The plan of three sides falls back to the plan that stood before it,
/// setting by setting: such a gateway says the document it said, and an
/// encode request meets no binding with an encode side. Three settings
/// fall back so. The manual policy places by its binding's one rule, so
/// both sides of a pair are manual or neither is. The sticky override's
/// rule has no document on a plan of more sides than one. And a main
/// policy the controller's check refuses leaves the plan as it leaves
/// the encode mode's.
#[test]
fn many_routers_keep_the_plan_they_had_where_three_sides_have_no_document() {
    let manual = || Some(at_flag_defaults("manual"));
    let manual_side = many_routers(RoutingSettings {
        mode: pair_mode(manual(), Some(PolicySettings::RoundRobin)),
        ..config(PolicySettings::RoundRobin)
    });
    let mut overridden = many_routers(config(PolicySettings::RoundRobin));
    overridden.routing_key_override.enabled = true;
    let mut ring = at_flag_defaults("prefix_hash");
    if let PolicySettings::PrefixHash {
        prefix_token_count, ..
    } = &mut ring
    {
        *prefix_token_count = 15;
    }
    let unchecked = many_routers(RoutingSettings {
        mode: pair_mode(
            Some(PolicySettings::RoundRobin),
            Some(PolicySettings::RoundRobin),
        ),
        ..config(ring)
    });
    // What the plan of three sides says alone: the setting it is
    // refused by, or a document the check refuses.
    let three = |settings: &RoutingSettings| {
        let plan = three_sided(&settings.mode, &settings.policy, settings.enable_igw);
        let plan = plan.expect("the plan's presets");
        written(settings, plan.expect("a plan of three sides"))
    };
    let refused_by = |settings: &RoutingSettings| match three(settings) {
        Err(RoutingSettingsError::UnsupportedSetting { setting, value, .. }) => {
            Some((setting, value))
        }
        _ => None,
    };
    assert_eq!(
        refused_by(&manual_side),
        Some(("policy", "manual".to_string()))
    );
    assert_eq!(
        refused_by(&overridden),
        Some(("routing_key_override.enabled", "true".to_string()))
    );
    let checked = three(&unchecked).expect("a written document");
    assert!(checked.validate().is_err());

    for (setting, settings) in [
        ("one manual side of a pair's two", &manual_side),
        ("the sticky override", &overridden),
        ("a main ring under one block", &unchecked),
    ] {
        let said = settings.policy_document();
        assert_eq!(said, before_three_sides(settings), "{setting}");
        let leads = said
            .ok()
            .and_then(|document| document.bindings.first().map(|binding| binding.plan));
        assert_ne!(leads, Some(Some(PlanName::Epd)), "{setting}");
    }
    // One manual side and the override under a policy it places keep
    // their refusals by the setting's name. The main ring keeps its two
    // families, which the controller's check refuses where it is applied.
    assert_eq!(refusal(&manual_side), ("policy", "manual".to_string()));
    assert_eq!(
        refusal(&overridden),
        ("routing_key_override.enabled", "true".to_string())
    );
    let pairs = unchecked
        .policy_document()
        .expect("the plan of two families");
    assert_eq!(pairs.bindings.len(), 2);
    assert_eq!(pairs.bindings[0].plan, Some(PlanName::PdPair));
    // A label is written into the plan that stands: the pair's binding
    // stays ahead of the label's, and no refusal of three sides is its.
    let labelled = with_labels(&unchecked, &[("m", "random")]).expect("a labelled one");
    assert_eq!(labelled.bindings.len(), 3);
    assert_eq!(labelled.bindings[0], pairs.bindings[0]);
    assert_eq!(labelled.bindings[2], pairs.bindings[1]);
}

/// The manual policy on both sides of a pair runs under a plan of three
/// sides. Its rule is the binding's one, and places the pair's two
/// sides: the encode side places an item by its own policy. So a gateway
/// of many routers in a pair mode plans its three sides under that policy
/// as under any other. It does so where both sides name it, and where
/// neither names a policy and the main one is manual. The encode mode
/// says a document, of one router as of many.
///
/// Both sides of the pair are manual or neither is, in the words that
/// refusal has. A manual encode side is refused by its setting's name
/// ahead of the document's check, whatever the pair's two sides run. A
/// regular configuration under the manual policy plans its single
/// requests alone, and so does a provider's in a gateway of many routers.
///
/// The gateway's own check starts the encode mode under any policy on
/// its prefill and its decode side: `validate_mode`
/// (`model_gateway/src/config/validation.rs`). It holds the encode side
/// to three policies, the manual one not among them:
/// `validate_encode_policy`, in the same file.
#[test]
fn the_manual_policy_on_both_sides_of_a_pair_runs_under_three_sides() {
    let manual = || Some(at_flag_defaults("manual"));
    let rule = preset("manual").bindings[0].sessions.clone();
    assert!(rule.is_some(), "the manual preset's rule");
    let encode_mode = |encode: Option<PolicySettings>, decode: Option<PolicySettings>| {
        ModeSettings::EncodePrefillDecode {
            encode_policy: encode,
            prefill_policy: manual(),
            decode_policy: decode,
        }
    };
    // The stock controller checks a document in a catalog with no stage
    // that needs the lifecycle's calls. It takes each plan of three sides
    // below: the manual policy's rule tells its stages.
    let served = StageRegistry::builtin().without_lifecycle();
    let on_file = |document: &PolicySpec| document.validate_in(&served, &[]).err();

    // A pair mode of many routers: the document it said under two
    // families, with the encode side ahead of the pair's two.
    let pairs = many_routers(RoutingSettings {
        mode: pair_mode(manual(), manual()),
        ..config(PolicySettings::RoundRobin)
    });
    let two_families = before_three_sides(&pairs).expect("the plan of two families");
    assert_eq!(two_families.bindings[0].plan, Some(PlanName::PdPair));
    let three = pairs.policy_document().expect("the plan of three sides");
    assert_eq!(three, with_encode_side(two_families));
    assert_eq!(three.bindings[0].sessions, rule);
    assert_eq!(validated(&three).err(), None);
    assert_eq!(on_file(&three), None);
    // A label is written into that plan, between its two bindings.
    let labelled = with_labels(&pairs, &[("m", "random")]).expect("a labelled one");
    assert_eq!(labelled.bindings.len(), 3);
    assert_eq!(labelled.bindings[0], three.bindings[0]);
    assert_eq!(labelled.bindings[2], three.bindings[1]);

    // The same plan under a manual main policy whose pair mode names no
    // side. Each side takes the main policy, and the main policy's single
    // binding behind the plan runs under the rule too.
    let unnamed = many_routers(RoutingSettings {
        mode: pair_mode(None, None),
        ..config(at_flag_defaults("manual"))
    });
    let two_families = before_three_sides(&unnamed).expect("the plan of two families");
    assert_eq!(two_families.bindings[0].plan, Some(PlanName::PdPair));
    let three = unnamed.policy_document().expect("the plan of three sides");
    assert_eq!(three, with_encode_side(two_families));
    assert_eq!(three.bindings[0].plan, Some(PlanName::Epd));
    assert_eq!(three.bindings[0].sessions, rule);
    assert_eq!(three.bindings[1].sessions, rule);
    assert_eq!(validated(&three).err(), None);
    assert_eq!(on_file(&three), None);

    // The encode mode of one router: the mode's three presets, with the
    // manual preset's rule on their one binding.
    let one_router = RoutingSettings {
        mode: encode_mode(None, manual()),
        ..config(PolicySettings::RoundRobin)
    };
    let mut own = PolicySpec::epd_presets("consistent_hashing", "manual", "manual")
        .expect("the mode's presets");
    own.bindings[0].sessions.clone_from(&rule);
    assert_eq!(one_router.policy_document().as_ref(), Ok(&own));
    assert_eq!(validated(&own).err(), None);
    assert_eq!(on_file(&own), None);
    // Of many routers the main policy's single binding stands behind it.
    let many = many_routers(one_router.clone())
        .policy_document()
        .expect("the document of both families");
    assert_eq!(many.bindings.len(), 2);
    assert_eq!(many.bindings[0].plan, Some(PlanName::Epd));
    assert_eq!(many.bindings[0].sessions, rule);
    assert_eq!(many.bindings[1].sessions, None);
    assert_eq!(validated(&many).err(), None);
    assert_eq!(on_file(&many), None);

    // One manual side of the two, in each mode that plans three sides.
    let one_side = [
        RoutingSettings {
            mode: encode_mode(None, Some(PolicySettings::RoundRobin)),
            ..config(PolicySettings::RoundRobin)
        },
        many_routers(RoutingSettings {
            mode: pair_mode(manual(), Some(PolicySettings::RoundRobin)),
            ..config(PolicySettings::RoundRobin)
        }),
    ];
    for settings in &one_side {
        assert_eq!(
                settings.policy_document(),
                Err(RoutingSettingsError::UnsupportedSetting {
                    setting: "policy",
                    value: "manual".to_string(),
                    reason: "the settings write the manual policy's session rule over both sides of a pair, so both sides are manual or neither is",
                }),
                "{:?}",
                settings.mode
            );
    }

    // A manual encode side, beside two manual sides, beside one and
    // beside none: its own setting is named whatever the pair's sides run.
    let other = || Some(PolicySettings::RoundRobin);
    for (prefill, decode) in [
        (manual(), manual()),
        (manual(), other()),
        (other(), other()),
    ] {
        let encode = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: manual(),
                prefill_policy: prefill,
                decode_policy: decode,
            },
            ..config(PolicySettings::RoundRobin)
        };
        for settings in [encode.clone(), many_routers(encode)] {
            assert_eq!(
                settings.policy_document(),
                Err(RoutingSettingsError::UnsupportedSetting {
                    setting: "encode_policy",
                    value: "manual".to_string(),
                    reason:
                        "a session rule places no item: an item goes to its encoder by its side's policy",
                }),
                "{:?}",
                settings.mode
            );
        }
    }

    // A regular configuration under the manual policy: no plan of three
    // sides, and the document of its single requests alone.
    let regular = config(at_flag_defaults("manual"));
    let many = many_routers(regular.clone());
    let three = three_sided(&many.mode, &many.policy, many.enable_igw);
    assert!(matches!(three, Ok(None)));
    assert_eq!(many.policy_document(), regular.policy_document());
    // A provider's configuration under it, the same: of many routers it
    // plans the gateway's own workers, their single requests alone.
    let provider = many_routers(RoutingSettings {
        mode: ModeSettings::Provider("a_provider"),
        ..config(at_flag_defaults("manual"))
    });
    let three = three_sided(&provider.mode, &provider.policy, provider.enable_igw);
    assert!(matches!(three, Ok(None)));
    assert_eq!(provider.policy_document(), regular.policy_document());
}

#[test]
fn many_routers_plan_one_family_where_the_other_has_no_document() {
    // A regular configuration under the manual policy plans its single
    // requests alone.
    let manual = config(at_flag_defaults("manual"));
    assert!(manual.policy_document().is_ok());
    assert_eq!(
        many_routers(manual.clone()).policy_document(),
        manual.policy_document()
    );

    // Under the override the main policy's keyed requests have no
    // document yet, while the hash ring on both sides of the pair has
    // one. The pair configuration's document then plans pairs alone.
    let mut keyed = RoutingSettings {
        mode: pair_mode(
            Some(PolicySettings::ConsistentHashing),
            Some(PolicySettings::ConsistentHashing),
        ),
        ..config(PolicySettings::RoundRobin)
    };
    keyed.routing_key_override.enabled = true;
    keyed.routing_key_override.headers = vec![MANUAL_SESSION_HEADER.to_string()];
    let pairs_alone = pair_alone(&keyed);
    assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
    assert_eq!(many_routers(keyed.clone()).policy_document(), pairs_alone);
    // The same main policy alone is refused by the setting's name.
    keyed.mode = ModeSettings::Regular;
    assert_eq!(
        refusal(&many_routers(keyed)),
        ("routing_key_override.enabled", "true".to_string())
    );
}

/// A gateway of many routers in the encode mode serves single workers
/// beside its encoders and pairs: a model of single workers is placed
/// by the main policy. So its document binds the mode's three sides
/// over their phases and every other request behind them, in that
/// order: the first binding that matches a request decides it.
#[test]
fn the_encode_mode_of_many_routers_binds_a_single_request_behind_its_sides() {
    let encode_mode = |encode: Option<PolicySettings>, sides: Option<PolicySettings>| {
        ModeSettings::EncodePrefillDecode {
            encode_policy: encode,
            decode_policy: sides.clone(),
            prefill_policy: sides,
        }
    };
    let settings = many_routers(RoutingSettings {
        mode: encode_mode(None, None),
        ..config(PolicySettings::RoundRobin)
    });
    let document = settings.policy_document().expect("the mode's document");
    assert_eq!(
        serde_json::to_value(&document.bindings).expect("a written form"),
        json!([
            {
                "match": { "model": "*", "phase": ["encode", "prefill", "decode"] },
                "plan": "epd",
                "profiles": { "decode": "decode", "encode": "encode", "prefill": "prefill" },
            },
            { "match": { "model": "*", "phase": [] }, "profile": "round_robin" },
        ]),
    );
    let applied = validated(&document).expect("the controller takes the document");
    assert_ne!(applied.engine().place_for("", Phase::Single), Place::NoSide);

    // A prefix ring counts its key in its own binding's blocks. Where
    // the main policy names another size, the mode's own document
    // stands, as it does under one router, and the controller takes it.
    let ringed = RoutingSettings {
        mode: encode_mode(None, Some(at_flag_defaults("prefix_hash"))),
        ..config(cache_aware_over(512))
    };
    let own = ringed.policy_document().expect("the mode's own document");
    assert!(validated(&own).is_ok());
    assert_eq!(many_routers(ringed).policy_document().as_ref(), Ok(&own));

    // A main policy the controller's check refuses is no document of
    // the single family: a prefix ring the view carries no block for
    // (its depth under one block or past the blocks a request carries,
    // or a boundary under one block), or a cache-aware setting the
    // build does not run. The mode's own document stands there too, the
    // controller takes it, and a single request has no side under it.
    let mut mains = Vec::new();
    for (depth, boundaries, alone) in [
        (15, vec![], true),
        (16, vec![], false),
        (4111, vec![], false),
        (4112, vec![], true),
        (256, vec![8], true),
    ] {
        let named = format!("a main ring of {depth} tokens over {boundaries:?}");
        let mut main = at_flag_defaults("prefix_hash");
        if let PolicySettings::PrefixHash {
            prefix_token_count,
            cache_boundaries,
            ..
        } = &mut main
        {
            (*prefix_token_count, *cache_boundaries) = (depth, boundaries);
        }
        mains.push((named, main, alone));
    }
    for setting in [
        "",
        "overlap_decay",
        "selection_temperature",
        "selection_accounting_ttl_ms",
        "balance_uncached_credit",
    ] {
        let mut main = at_flag_defaults("cache_aware");
        if let PolicySettings::CacheAware {
            overlap_decay,
            selection_temperature,
            selection_accounting_ttl_ms,
            balance_uncached_credit,
            ..
        } = &mut main
        {
            match setting {
                "overlap_decay" => *overlap_decay = 0.5,
                "selection_temperature" => *selection_temperature = 0.05,
                "selection_accounting_ttl_ms" => *selection_accounting_ttl_ms = 200,
                "balance_uncached_credit" => *balance_uncached_credit = true,
                _ => {}
            }
        }
        let named = format!("a cache-aware main policy that sets {setting:?}");
        mains.push((named, main, !setting.is_empty()));
    }
    for (named, main, alone) in mains {
        let round_robin = Some(PolicySettings::RoundRobin);
        let uncarried = RoutingSettings {
            mode: encode_mode(round_robin.clone(), round_robin),
            ..config(main)
        };
        let own = uncarried
            .policy_document()
            .expect("the mode's own document");
        let said = many_routers(uncarried)
            .policy_document()
            .expect("the settings' document");
        let applied = validated(&said).unwrap_or_else(|refusal| panic!("{named}: {refusal}"));
        let unplaced = applied.engine().place_for("", Phase::Single) == Place::NoSide;
        assert_eq!((said == own, unplaced), (alone, alone), "{named}");
    }

    // No configuration of the mode loses its document to its single
    // request. Every policy is the main one in turn, the encode side
    // and the pair's sides are unnamed or of one policy, and the sticky
    // override is off and on. Where the mode's own document, as
    // `presets` plans it, is refused, the settings' is refused by the
    // same sentence. Where the controller's check takes the mode's
    // own, it takes the settings', which binds a single request behind
    // the mode's sides in all but these: under the override, beside
    // sides that place by the key themselves, the hash ring's or the
    // manual policy's, every main policy but `consistent_hashing` and
    // `manual` has no document for its keyed requests, and the mode is
    // planned alone, whatever its encode side.
    let unnamed_or_each = || std::iter::once(None).chain(PRESETS.iter().copied().map(Some));
    let mut alone_still = Vec::new();
    for main in PRESETS {
        for encode in unnamed_or_each() {
            for sides in unnamed_or_each() {
                for sticky in [false, true] {
                    let named = format!("{main} over {encode:?} and {sides:?}, {sticky}");
                    let mut swept = many_routers(RoutingSettings {
                        mode: encode_mode(
                            encode.map(at_flag_defaults),
                            sides.map(at_flag_defaults),
                        ),
                        ..config(at_flag_defaults(main))
                    });
                    swept.routing_key_override.enabled = sticky;
                    let alone = presets(&swept).and_then(|planned| written(&swept, planned));
                    let said = swept.policy_document();
                    let (Ok(one), Ok(both)) = (&alone, &said) else {
                        assert_eq!(said, alone, "{named}");
                        continue;
                    };
                    let taken = validated(one).is_ok();
                    assert_eq!(validated(both).is_ok(), taken, "{named}");
                    if taken && both.bindings.len() == 1 {
                        assert_eq!(both, one, "{named}");
                        if !alone_still.contains(&(main, sides, sticky)) {
                            alone_still.push((main, sides, sticky));
                        }
                    }
                }
            }
        }
    }
    let keyed_sides = [Some("consistent_hashing"), Some("manual")];
    let keyless = [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
        "prefix_hash",
        "cache_aware",
        "bucket",
    ];
    let planned_alone = |main| keyed_sides.map(|sides| (main, sides, true));
    assert_eq!(alone_still, keyless.map(planned_alone).concat());
}

/// The labels of these models, each with the preset name its label
/// resolves to.
fn labels(named: &[(&str, &str)]) -> BTreeMap<String, String> {
    named
        .iter()
        .map(|&(model, policy)| (model.to_owned(), policy.to_owned()))
        .collect()
}

/// The settings' document with these labels.
fn with_labels(
    settings: &RoutingSettings,
    named: &[(&str, &str)],
) -> Result<PolicySpec, RoutingSettingsError> {
    settings.policy_document_with_labels(&labels(named))
}

/// A label's refusal, in its three parts.
fn label_error(model: &str, policy: &str, reason: &str) -> RoutingSettingsError {
    RoutingSettingsError::Label {
        model: model.to_owned(),
        policy: policy.to_owned(),
        reason: reason.to_owned(),
    }
}

/// A main policy that is not the label's: round robin, and `random`
/// beside a label of round robin.
fn main_beside(label: &str) -> PolicySettings {
    at_flag_defaults(if label == "round_robin" {
        "random"
    } else {
        "round_robin"
    })
}

/// The profile the document binds a single request of `model` to, as
/// the controller's check builds the document.
fn single_profile(document: &PolicySpec, model: &str) -> String {
    let engine = document
        .validate_with(&["x-smg-tenant-id"])
        .expect("the controller's check takes the document");
    let binding = engine
        .binding(model, Phase::Single)
        .expect("a binding for the model's single request");
    engine.profiles()[usize::from(binding.profile)].name.clone()
}

/// The index the document answers for `model`: its most specific entry,
/// or the defaults where no entry names the model.
fn index_of(document: &PolicySpec, model: &str) -> IndexSpec {
    let entry = document.index_for(model);
    entry.map_or_else(IndexSpec::default, |(_, index)| index.clone())
}

/// A label builds each policy as the gateway's registry builds it from
/// a name alone. Eight of the ten are the policy at its flags' defaults,
/// the preset. `cache_aware` and `bucket` are not: seven numbers are the
/// registry's own, and tidying them to the presets' turns this case red.
/// A name's spelling is its host's to resolve: only a preset's name
/// names a policy here.
#[test]
fn a_label_builds_two_policies_at_other_numbers_than_their_presets() {
    for name in PRESETS {
        let by_label = PolicySettings::of_label(name).expect("a policy of each preset's name");
        assert_eq!(by_label.name(), name);
        let preset = by_label == at_flag_defaults(name);
        assert_eq!(preset, !["cache_aware", "bucket"].contains(&name), "{name}");
    }
    assert_eq!(
        PolicySettings::of_label("cache_aware"),
        Some(cache_aware(0.5, 32, 1.1, 30, 10_000))
    );
    assert_eq!(
        PolicySettings::of_label("bucket"),
        Some(PolicySettings::Bucket {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        })
    );
    for unnamed in ["cacheaware", "Bucket", "least_loaded", ""] {
        assert_eq!(PolicySettings::of_label(unnamed), None, "{unnamed:?}");
    }
    // A label's `cache_aware` names the block size a silent ring counts
    // in: beside it no ring of the settings' sides loses its blocks.
    let silent = usize::try_from(Gates::default().block_size).expect("a block size");
    assert!(matches!(
        PolicySettings::of_label("cache_aware"),
        Some(PolicySettings::CacheAware { block_size, .. }) if block_size == silent
    ));
}

/// With no label the labelled document is the settings' own, and a
/// label that names the main policy changes nothing: its model already
/// runs that profile, with the settings' options.
#[test]
fn no_label_and_a_label_of_the_main_policy_leave_the_settings_document() {
    for name in PRESETS {
        for settings in [
            config(at_flag_defaults(name)),
            round_robin_pairs_under(at_flag_defaults(name)),
            many_routers(config(at_flag_defaults(name))),
        ] {
            let own = settings.policy_document();
            assert_eq!(with_labels(&settings, &[]), own, "{name}");
            assert_eq!(with_labels(&settings, &[("m", name)]), own, "{name}");
        }
    }
}

/// Each stock policy as a label, beside another main policy: the
/// label's binding stands ahead of the settings' own, which stays as it
/// was with its profile. The controller's check takes the document, and
/// its engine binds a single request of the labelled model to the
/// label's profile and another model's to the main policy's.
#[test]
fn each_stock_policy_as_a_label_binds_its_model_ahead_of_every_model() {
    for name in PRESETS {
        let settings = config(main_beside(name));
        let main = settings.policy.name();
        let own = settings.policy_document().expect("the settings' document");
        let document = with_labels(&settings, &[("labelled", name)]).expect("a labelled one");
        assert_eq!(document.bindings.len(), 2, "{name}");
        assert_eq!(document.bindings[0].matcher.model, "labelled", "{name}");
        assert!(document.bindings[0].matcher.phase.is_empty(), "{name}");
        assert_eq!(document.bindings[0].profile.as_deref(), Some(name));
        assert_eq!(document.bindings[1], own.bindings[0], "{name}");
        assert_eq!(document.profiles[main], own.profiles[main], "{name}");
        assert_eq!(single_profile(&document, "labelled"), name);
        assert_eq!(single_profile(&document, "another"), main);
    }
}

/// One way to set a configuration, by its name in a case's messages.
type Way<'a> = (&'a str, &'a dyn Fn(&mut RoutingSettings));

/// In a regular gateway of one router, a label's binding is the one its
/// policy would get as the main policy of the same settings: its own
/// profile, key sources and session rule, by the code that writes a main
/// policy's, and none of the main binding's. So the warm-up settings
/// land in a labelled `cache_aware` profile, configured headers lead a
/// keyed label's sources, and under the sticky override a label carries
/// the override's rule where the override places its policy. Where the
/// policy has no document as the main policy there, the label is refused
/// by that sentence. A mode whose side takes the main policy is no such
/// comparison: as the main policy the label's would place that side too.
#[test]
fn a_labels_binding_is_the_one_its_policy_gets_as_the_main_policy() {
    let warm = |settings: &mut RoutingSettings| settings.worker_warmup_secs = 7;
    let headed = |settings: &mut RoutingSettings| {
        settings.routing_key_override.headers = vec!["x-session".to_owned()];
    };
    let sticky = |settings: &mut RoutingSettings| settings.routing_key_override.enabled = true;
    let both = |settings: &mut RoutingSettings| {
        settings.routing_key_override.enabled = true;
        settings.routing_key_override.headers = vec!["x-session".to_owned()];
    };
    let plain = |_: &mut RoutingSettings| {};
    let unprotected = |settings: &mut RoutingSettings| settings.worker_overload_protection = false;
    let ways: [Way<'_>; 6] = [
        ("as set by none", &plain),
        ("the overload protection off", &unprotected),
        ("a warm-up of seven seconds", &warm),
        ("a configured header", &headed),
        ("the sticky override", &sticky),
        ("the override over a configured header", &both),
    ];
    let mut refused = Vec::new();
    for name in PRESETS {
        for (way, set) in ways {
            let mut settings = config(main_beside(name));
            set(&mut settings);
            let main = settings.policy.name();
            let own = settings
                .policy_document()
                .expect("the main policy's document");
            let as_main = RoutingSettings {
                policy: PolicySettings::of_label(name).expect("the label's policy"),
                ..settings.clone()
            }
            .policy_document();
            let said = with_labels(&settings, &[("m", name)]);
            let as_main = match as_main {
                Ok(document) => document,
                Err(refusal) => {
                    let expected = label_error("m", name, &refusal.to_string());
                    assert_eq!(said, Err(expected), "{name}, {way}");
                    refused.push((name, way));
                    continue;
                }
            };
            let document = said.unwrap_or_else(|refusal| panic!("{name}, {way}: {refusal}"));
            let mut expected = as_main.bindings[0].clone();
            expected.matcher.model = "m".to_owned();
            assert_eq!(document.bindings[0], expected, "{name}, {way}");
            assert_eq!(document.bindings[1], own.bindings[0], "{name}, {way}");
            assert_eq!(
                document.profiles[name], as_main.profiles[name],
                "{name}, {way}"
            );
            assert_eq!(document.profiles[main], own.profiles[main], "{name}, {way}");
            // The label's model reads the index its policy gets as the
            // main policy, and every other model the settings' own.
            let index = index_of(&document, "m");
            assert_eq!(index, index_of(&as_main, "m"), "{name}, {way}");
            let index = index_of(&document, "another");
            assert_eq!(index, index_of(&own, "another"), "{name}, {way}");
            // The view's headers: the main binding's, then the label's.
            let mut headers = own.view.headers.clone();
            for header in &as_main.view.headers {
                if !headers.contains(header) {
                    headers.push(header.clone());
                }
            }
            assert_eq!(document.view.headers, headers, "{name}, {way}");
            assert!(validated(&document).is_ok(), "{name}, {way}");
        }
    }
    // The settings' overload block stands on a label's binding.
    let mut exposed = config(main_beside("random"));
    unprotected(&mut exposed);
    let document = with_labels(&exposed, &[("m", "random")]).expect("a labelled one");
    let switched_off = Some(OverloadSpec {
        enabled: false,
        ..OverloadSpec::default()
    });
    assert_eq!(document.bindings[0].overload, switched_off);
    assert_eq!(document.bindings[1].overload, switched_off);
    // The warm-up setting is in the labelled profile, not the preset's.
    let mut warmed = config(main_beside("cache_aware"));
    warm(&mut warmed);
    let cold = config(main_beside("cache_aware"));
    let profile = |settings: &RoutingSettings| {
        let document = with_labels(settings, &[("m", "cache_aware")]).expect("a labelled one");
        document.profiles["cache_aware"].clone()
    };
    assert_ne!(profile(&warmed), profile(&cold));
    // Under the override a policy that places by the key itself reads
    // the routing key header as sent, behind a configured header.
    assert_eq!(
        refused,
        [
            (
                "consistent_hashing",
                "the override over a configured header"
            ),
            ("manual", "the override over a configured header"),
        ]
    );
}

/// A pair mode keeps its pair's binding under a label: the label's
/// stands behind it and ahead of the binding of every model's single
/// requests. The engine binds a prefill of the labelled model to the
/// pair plan and its single request to the label's profile.
#[test]
fn a_pair_mode_keeps_its_pair_binding_ahead_of_a_labels() {
    for (settings, three_sides) in [
        (
            round_robin_pairs_under(at_flag_defaults("least_load")),
            false,
        ),
        (many_routers(config(at_flag_defaults("least_load"))), true),
    ] {
        let own = settings
            .policy_document()
            .expect("a document of two families");
        assert_eq!(own.bindings.len(), 2);
        let document = with_labels(&settings, &[("m", "random")]).expect("a labelled one");
        assert_eq!(document.bindings.len(), 3);
        assert_eq!(document.bindings[0], own.bindings[0]);
        assert_eq!(document.bindings[1].matcher.model, "m");
        assert_eq!(document.bindings[1].profile.as_deref(), Some("random"));
        assert_eq!(document.bindings[1].overload, own.bindings[1].overload);
        assert_eq!(document.bindings[2], own.bindings[1]);
        let engine = document
            .validate_with(&["x-smg-tenant-id"])
            .expect("the controller's check takes the document");
        let plan = |phase| engine.binding("m", phase).map(|binding| binding.plan);
        // A gateway of many routers plans the pair among three sides.
        for phase in [Phase::Prefill, Phase::Decode] {
            let pairs = match plan(phase) {
                Some(Plan::PdPair) => !three_sides,
                Some(Plan::Epd { .. }) => three_sides,
                _ => false,
            };
            assert!(pairs, "{phase:?}");
        }
        assert_eq!(plan(Phase::Single), Some(Plan::Single));
        assert_eq!(single_profile(&document, "m"), "random");
        assert_eq!(single_profile(&document, "another"), "least_load");
    }
}

/// A label the document has no binding for is refused by its model and
/// its policy: a model's id that holds a `*` or is empty, and a name
/// with no preset. A label of the main policy adds no binding, so its
/// model's id refuses nothing. That label is the main policy's name as
/// a preset spells it: another letter case is a name with no preset.
#[test]
fn a_label_is_refused_by_its_model_and_its_policy() {
    let settings = config(at_flag_defaults("round_robin"));
    let own = settings.policy_document();
    let pattern = "a binding reads a `*` at the start or the end of a model's name as a pattern and the document refuses one anywhere else, so no binding names this model alone";
    for model in ["*", "m*", "*m", "a*b"] {
        assert_eq!(
            with_labels(&settings, &[(model, "random")]),
            Err(label_error(model, "random", pattern)),
        );
        assert_eq!(with_labels(&settings, &[(model, "round_robin")]), own);
    }
    assert_eq!(
        with_labels(&settings, &[("", "random")]),
        Err(label_error(
            "",
            "random",
            "a binding names its model by the model's id, and this id is empty"
        )),
    );
    assert_eq!(with_labels(&settings, &[("", "round_robin")]), own);
    let unknown = |name: &str| SpecError::UnknownPreset(name.to_owned()).to_string();
    assert_eq!(
        with_labels(&settings, &[("a", "random"), ("m", "least_loaded")]),
        Err(label_error("m", "least_loaded", &unknown("least_loaded"))),
    );
    assert_eq!(
        with_labels(&settings, &[("m", "Round_Robin")]),
        Err(label_error("m", "Round_Robin", &unknown("Round_Robin"))),
    );
    // Every label's name and id is read before a side is written, so
    // the refusal need not be the first label's in the ids' order.
    let blocks = config(cache_aware_over(32));
    assert!(with_labels(&blocks, &[("a", "prefix_hash")]).is_err());
    assert_eq!(
        with_labels(&blocks, &[("a", "prefix_hash"), ("z", "least_loaded")]),
        Err(label_error("z", "least_loaded", &unknown("least_loaded"))),
    );
}

/// A refused label fails alone. Two labels never refuse one another, so
/// each stands or falls beside the settings alone: the refusal names the
/// one that falls, and every subset of the labels that build builds. A
/// subset's document holds its labels' bindings as the whole one holds
/// them, in their order, ahead of the settings' own, and no other.
#[test]
fn labels_stand_or_fall_alone_beside_the_settings() {
    let settings = config(cache_aware_over(32));
    let named = [
        ("a", "random"),
        ("b", "prefix_hash"),
        ("c", "manual"),
        ("d", "cache_aware"),
        ("e", "bucket"),
    ];
    let falls = |label: &(&str, &str)| with_labels(&settings, &[*label]).is_err();
    let fallen: Vec<_> = named.iter().filter(|label| falls(label)).collect();
    assert_eq!(fallen, [&("b", "prefix_hash")]);
    match with_labels(&settings, &named) {
        Err(RoutingSettingsError::Label { model, policy, .. }) => {
            assert_eq!((model.as_str(), policy.as_str()), ("b", "prefix_hash"));
        }
        other => panic!("expected the ring's label refused, got {other:?}"),
    }
    // The main policy is `cache_aware`: its own label adds nothing.
    let built = [named[0], named[2], named[3], named[4]];
    let whole = with_labels(&settings, &built).expect("the labels that build");
    let models = |document: &PolicySpec| -> Vec<String> {
        document
            .bindings
            .iter()
            .map(|binding| binding.matcher.model.clone())
            .collect()
    };
    assert_eq!(models(&whole), ["a", "c", "e", "*"]);
    let own = settings.policy_document().expect("the settings' document");
    for picked in 0..(1_u32 << built.len()) {
        let subset: Vec<_> = built
            .iter()
            .enumerate()
            .filter(|(at, _)| picked & (1 << at) != 0)
            .map(|(_, label)| *label)
            .collect();
        let document = with_labels(&settings, &subset).expect("a subset builds");
        let of_subset = |binding: &&BindingSpec| {
            subset
                .iter()
                .any(|(model, _)| binding.matcher.model == *model)
        };
        let mut expected: Vec<_> = whole.bindings.iter().filter(of_subset).cloned().collect();
        expected.extend(own.bindings.iter().cloned());
        assert_eq!(document.bindings, expected, "{subset:?}");
        for (name, profile) in &document.profiles {
            assert_eq!(whole.profiles.get(name), Some(profile), "{subset:?}");
        }
    }
}

/// A label whose binding would pass a bound of the document is refused
/// by its model and its policy, in the words of that bound. A key block
/// names eight sources at most: the builder's own sentence, as for a
/// main policy. The view lists 256 headers at most, which the
/// controller's check holds for the whole document and for no model by
/// name: a label that would list the 257th is refused here, and the
/// settings' own document and every other label stand.
#[test]
fn a_label_past_a_bound_of_the_document_is_refused_by_name() {
    let headed = |count: usize| {
        let mut settings = config(at_flag_defaults("round_robin"));
        settings.routing_key_override.headers = (0..count).map(|at| format!("x-k{at}")).collect();
        settings
    };
    let eight = headed(8);
    let as_main = RoutingSettings {
        policy: at_flag_defaults("consistent_hashing"),
        ..eight.clone()
    }
    .policy_document()
    .expect_err("a ninth source of the ring's key");
    assert_eq!(
        with_labels(&eight, &[("m", "consistent_hashing")]),
        Err(label_error("m", "consistent_hashing", &as_main.to_string())),
    );
    assert!(with_labels(&headed(7), &[("m", "consistent_hashing")]).is_ok());

    // The manual rule names its headers by the rule's own fields, which
    // no key block bounds: its headers are the view's.
    let full = headed(255);
    let document = with_labels(&full, &[("b", "manual"), ("m", "manual")]).expect("256 headers");
    assert_eq!(document.view.headers.len(), 256);
    assert!(validated(&document).is_ok());
    let over = headed(256);
    let own = over.policy_document().expect("the settings' own document");
    assert!(own.view.headers.is_empty());
    let too_many = SpecError::TooManyHeaders {
        count: 257,
        max: 256,
    };
    assert_eq!(
        with_labels(&over, &[("a", "random"), ("m", "manual")]),
        Err(label_error("m", "manual", &too_many.to_string())),
    );
    assert_eq!(
        with_labels(&over, &[("a", "random")]).map(|document| document.bindings.len()),
        Ok(2)
    );
    // A list the settings' own policy took past the bound is the
    // settings': a label that adds no header is not refused for it.
    let own_over = RoutingSettings {
        policy: at_flag_defaults("manual"),
        ..headed(256)
    };
    let headers = |document: PolicySpec| document.view.headers.len();
    assert_eq!(own_over.policy_document().map(headers), Ok(257));
    assert_eq!(
        with_labels(&own_over, &[("m", "random")]).map(headers),
        Ok(257)
    );
}

/// Models in the order of their ids, and one profile for each policy:
/// two models that name one policy share its profile, and the same
/// labels give the same document.
#[test]
fn labelled_models_stand_in_id_order_and_share_a_policys_profile() {
    let settings = config(at_flag_defaults("round_robin"));
    let named = [
        ("zeta", "random"),
        ("alpha", "least_load"),
        ("mid", "random"),
    ];
    let document = with_labels(&settings, &named).expect("a labelled document");
    let bound: Vec<_> = document
        .bindings
        .iter()
        .map(|binding| (binding.matcher.model.as_str(), binding.profile.as_deref()))
        .collect();
    assert_eq!(
        bound,
        [
            ("alpha", Some("least_load")),
            ("mid", Some("random")),
            ("zeta", Some("random")),
            ("*", Some("round_robin")),
        ]
    );
    let profiles: Vec<_> = document.profiles.keys().map(String::as_str).collect();
    assert_eq!(profiles, ["least_load", "random", "round_robin"]);
    let again = [named[2], named[0], named[1]];
    assert_eq!(with_labels(&settings, &again), Ok(document.clone()));
    assert!(validated(&document).is_ok());
    assert_eq!(single_profile(&document, "mid"), "random");
    assert_eq!(single_profile(&document, "alpha"), "least_load");
    // A policy's profile is written once for each model that names it,
    // and a second write leaves it as the first left it.
    for name in PRESETS {
        let settings = config(main_beside(name));
        let once = with_labels(&settings, &[("a", name)]).expect("one model");
        let twice = with_labels(&settings, &[("a", name), ("b", name)]).expect("two models");
        assert_eq!(twice.bindings.len(), 3, "{name}");
        assert_eq!(twice.profiles, once.profiles, "{name}");
        assert_eq!(twice.view, once.view, "{name}");
        // Each labelled model reads the same index, and no other model
        // reads another than it read.
        for model in ["a", "b"] {
            assert_eq!(index_of(&twice, model), index_of(&once, "a"), "{name}");
        }
        let index = index_of(&twice, "another");
        assert_eq!(index, index_of(&once, "another"), "{name}");
        // The second model's binding is the first's but for its model:
        // its key sources and its session rule are written again.
        assert_eq!(twice.bindings[0], once.bindings[0], "{name}");
        let mut second = twice.bindings[1].clone();
        assert_eq!(second.matcher.model, "b", "{name}");
        second.matcher.model = "a".to_owned();
        assert_eq!(second, twice.bindings[0], "{name}");
    }
}

/// Two models that name one policy by their labels are bound to one
/// profile, so a stage that keeps a history of its own keeps one for
/// both. Round robin draws the start of a list it sees first from a
/// count of its own. In one labelled document the second model's first
/// turn starts where that count stands behind the first model's. In a
/// document that labels it alone, it starts where the first model's
/// did: a gateway's own registry builds a policy for each labelled
/// model, so there every labelled model starts as the first.
#[test]
fn labelled_models_of_one_policy_share_what_its_stages_count() {
    use crate::engine::{
        framework::{Engine, Scratch},
        view::{Candidate, Constraints, Health, PoolFacts, RequestFacts, WorkerId},
    };
    let settings = config(at_flag_defaults("random"));
    let engine = |named: &[(&str, &str)]| {
        with_labels(&settings, named)
            .expect("a labelled document")
            .validate_with(&["x-smg-tenant-id"])
            .expect("the controller's check takes the document")
    };
    let pool: Vec<Candidate> = (0..32)
        .map(|position| Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
            key: 100 + position,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect();
    let first_turn = |engine: &Engine, model: &'static str| {
        let facts = RequestFacts {
            model,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let pick = decision.pick().expect("a pick");
        pick.targets.iter().next().expect("a target").position
    };
    let both = engine(&[("a", "round_robin"), ("b", "round_robin")]);
    let a_first = first_turn(&both, "a");
    let b_behind_a = first_turn(&both, "b");
    let alone = engine(&[("b", "round_robin")]);
    let b_alone = first_turn(&alone, "b");
    assert_eq!(b_alone, a_first);
    assert_ne!(b_behind_a, b_alone);
}

/// No name holds two profiles. A pair's sides are named for the side,
/// so a `bucket` label beside a pair whose prefill policy is `bucket`
/// adds a profile of its own name, at the numbers a label builds it
/// with, and leaves the side's at the settings'.
#[test]
fn a_labels_profile_stands_beside_a_side_of_the_same_policy() {
    let settings = RoutingSettings {
        mode: pair_mode(Some(at_flag_defaults("bucket")), None),
        ..config(at_flag_defaults("round_robin"))
    };
    let own = settings
        .policy_document()
        .expect("the pair mode's document");
    let document = with_labels(&settings, &[("m", "bucket")]).expect("a labelled one");
    let profiles: Vec<_> = document.profiles.keys().map(String::as_str).collect();
    assert_eq!(profiles, ["bucket", "decode", "prefill", "round_robin"]);
    for side in ["decode", "prefill", "round_robin"] {
        assert_eq!(document.profiles[side], own.profiles[side], "{side}");
    }
    let by_label = config(PolicySettings::of_label("bucket").expect("the label's policy"))
        .policy_document()
        .expect("the label's policy as a main policy");
    assert_eq!(document.profiles["bucket"], by_label.profiles["bucket"]);
    assert_ne!(
        document.profiles["bucket"].score,
        own.profiles["prefill"].score
    );
}

/// A pair configuration with round robin on both sides, under `main`.
fn round_robin_pairs_under(main: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        mode: pair_mode(
            Some(PolicySettings::RoundRobin),
            Some(PolicySettings::RoundRobin),
        ),
        ..config(main)
    }
}

/// The cache-aware policy at its flags' defaults, over blocks of `blocks`.
fn cache_aware_over(blocks: usize) -> PolicySettings {
    let mut policy = at_flag_defaults("cache_aware");
    if let PolicySettings::CacheAware { block_size, .. } = &mut policy {
        *block_size = blocks;
    }
    policy
}

/// No pair mode loses its document to its single request. Every policy
/// is the main one in turn, the sides are unnamed or both of one policy,
/// and the sticky override is off and on. Where the pair's own document
/// is refused, the mode's is refused by the same sentence. Where the
/// controller's check takes the pair's own document, it takes the
/// mode's, which binds a single request behind the pair in every one.
#[test]
fn no_pair_mode_loses_its_document_to_its_single_request() {
    for main in PRESETS {
        for sides in std::iter::once(None).chain(PRESETS.iter().copied().map(Some)) {
            for sticky in [false, true] {
                let mut settings = RoutingSettings {
                    mode: pair_mode(sides.map(at_flag_defaults), sides.map(at_flag_defaults)),
                    ..config(at_flag_defaults(main))
                };
                settings.routing_key_override.enabled = sticky;
                let named = format!("{main} over {sides:?}, the override {sticky}");
                let (alone, said) = (pair_alone(&settings), settings.policy_document());
                let (Ok(own), Ok(both)) = (&alone, &said) else {
                    assert_eq!(said, alone, "{named}");
                    continue;
                };
                let taken = validated(own).is_ok();
                assert_eq!(validated(both).is_ok(), taken, "{named}");
                assert!(!taken || both.bindings.len() == 2, "{named}");
            }
        }
    }
}
