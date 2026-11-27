use serde_json::json;

use super::{translation::label_refused, *};
use crate::{
    document::spec::{
        AppliedSpec, BindingSpec, IndexSpec, OverloadSpec, PlanName, Recovery, PRESETS,
    },
    engine::{
        framework::{InTheWay, Place, Plan},
        view::Phase,
    },
    policies::cache_aware::Gates,
};

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

/// The settings a configuration leaves out are those of a gateway that
/// sets none: the `random` preset, with nothing written over it.
#[test]
fn the_settings_of_a_gateway_that_sets_none_mean_the_random_preset() {
    assert_eq!(
        RoutingSettings::default().policy_document(),
        Ok(preset("random"))
    );
}

#[test]
fn each_policy_name_at_its_flags_defaults_is_its_preset() {
    for name in PRESETS {
        let policy = at_flag_defaults(name);
        assert_eq!(
            policy.name(),
            name,
            "the policy and the preset share one name"
        );
        assert_eq!(
            config(policy).policy_document(),
            Ok(preset(name)),
            "{name} at its flags' defaults is the preset, with nothing written over it"
        );
    }
}

/// A threshold at none switches its signal off in the gateway, on its
/// worker objects, and no document carries a threshold. With the veto
/// on, one signal off leaves the block switched on, whatever the other
/// is: the settings mean the preset, with nothing written over it. Both
/// off is the block switched off, and the shed word stays where it is
/// written.
#[test]
fn one_overload_signal_off_leaves_the_block_on_and_both_off_switch_it_off() {
    let thresholds = |waiting: Option<usize>, usage: Option<f64>| {
        let mut config = config(at_flag_defaults("round_robin"));
        config.worker_overload_waiting_requests = waiting;
        config.worker_overload_token_usage = usage;
        config
    };
    for (waiting, usage) in [
        (None, Some(0.8)),
        (None, Some(0.5)),
        (Some(8), None),
        (Some(32), None),
    ] {
        assert_eq!(
            thresholds(waiting, usage).policy_document(),
            Ok(preset("round_robin")),
            "waiting {waiting:?}, usage {usage:?}"
        );
    }
    let switched_off = edited(preset("round_robin"), |document| {
        document["bindings"][0]["overload"] = json!({ "enabled": false });
    });
    assert_eq!(
        thresholds(None, None).policy_document(),
        Ok(switched_off.clone())
    );
    // With the protection off the thresholds decide nothing: the block
    // is off whatever they are.
    let mut unprotected = thresholds(None, Some(0.8));
    unprotected.worker_overload_protection = false;
    assert_eq!(unprotected.policy_document(), Ok(switched_off));
    // The shed word does not switch the block on.
    let mut shedding = thresholds(None, None);
    shedding.worker_overload_shed = true;
    assert_eq!(
        shedding.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] =
                json!({ "enabled": false, "shed_when_all_vetoed": true });
        }))
    );
}

/// The converter puts a written stage in the place of a preset's stage
/// at the preset's weight, and carries no other parameter of it. So a
/// preset's scorer that has parameters has a weight alone. A preset that
/// gains another parameter fails here first, and the converter carries
/// that parameter from then on. `consistent-hash` is the one stage the
/// converter writes whole from the configuration.
#[test]
fn every_presets_scorer_with_parameters_has_a_weight_alone() {
    for name in PRESETS {
        let form = serde_json::to_value(preset(name)).expect("a document serializes");
        let profiles = form["profiles"].as_object().expect("the profiles");
        for (profile, stages) in profiles {
            for scorer in stages["score"].as_array().into_iter().flatten() {
                let Some(with_parameters) = scorer.as_object() else {
                    continue;
                };
                for (stage, parameters) in with_parameters {
                    if stage == "consistent-hash" {
                        continue;
                    }
                    let named: Vec<&String> = parameters
                        .as_object()
                        .into_iter()
                        .flatten()
                        .map(|(name, _)| name)
                        .collect();
                    assert_eq!(named, ["weight"], "{name}: {profile}: {stage}");
                }
            }
        }
    }
}

#[test]
fn power_of_two_lands_on_the_form_of_its_metric() {
    for (load_metric, form) in [
        (
            LoadMetricSetting::ExpectedWait,
            PowerOfTwoMetric::ExpectedWait,
        ),
        (LoadMetricSetting::Requests, PowerOfTwoMetric::Requests),
        (
            LoadMetricSetting::LeastRequests,
            PowerOfTwoMetric::LeastRequests,
        ),
        (
            LoadMetricSetting::ResidentTokens {
                request_tokens: 512,
            },
            PowerOfTwoMetric::ResidentTokens {
                request_tokens: 512,
            },
        ),
    ] {
        assert_eq!(
            config(PolicySettings::PowerOfTwo { load_metric }).policy_document(),
            Ok(PolicySpec::power_of_two(form).expect("a form of the preset")),
            "{load_metric:?} is the preset in that metric's form"
        );
    }
}

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
fn a_cache_aware_setting_lands_in_the_stage_that_reads_it() {
    let tuned = RoutingSettings {
        policy: PolicySettings::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            balance_prefix_margin_tokens: 1024,
            balance_uncached_credit: true,
            eviction_interval_secs: 60,
            max_tree_size: 16_777_216,
            block_size: 256,
            balance_token_usage_threshold: 0.3,
            overload_token_usage_threshold: 0.9,
            overlap_decay: 0.5,
            selection_temperature: 0.05,
            cache_index: IndexKind::Hash,
            cache_ttl_secs: 300,
            cache_boundaries: vec![512, 2048],
            selection_policy: None,
            selection_accounting_ttl_ms: 200,
        },
        worker_warmup_secs: 0,
        worker_warmup_blocks: 2048,
        ..defaults()
    };
    let expected = edited(preset("cache_aware"), |document| {
        document["view"] = json!({ "block_size": 256, "boundaries": [512, 2048] });
        document["index"] = json!({
            "*": { "kind": "hash", "ttl_s": 300, "max_size": 16_777_216, "sweep_s": 60 }
        });
        let profile = &mut document["profiles"]["cache_aware"];
        profile["score"][0] = json!({
            "warmup-slice": { "weight": 8.0, "secs": 0, "blocks": 2048, "miss_share": 0.5 }
        });
        profile["score"][1] = json!({ "prefix-affinity": {
            "weight": 4.0,
            "threshold": 0.5,
            "decay": 0.5,
            "accounting_ttl_ms": 200,
            "spill": {
                "abs": 32, "rel": 1.1, "prefix_margin_tokens": 1024, "uncached_credit": true
            },
        }});
        profile["score"][2] = json!({ "kv-pressure": { "spread": 0.3, "ceiling": 0.9 } });
        profile["pick"] = json!({ "best-with-temperature": { "temperature": 0.05 } });
    });
    assert_eq!(tuned.policy_document(), Ok(expected));
}

#[test]
fn the_settings_of_the_other_policies_land_in_their_stages() {
    let least_load = config(PolicySettings::LeastLoad {
        kv_pressure_weight: 0.3,
        mean_prefill_tokens: 1024,
        default_throughput: 2000.0,
    });
    assert_eq!(
        least_load.policy_document(),
        Ok(edited(preset("least_load"), |document| {
            document["profiles"]["least_load"]["score"][0] =
                json!({ "least-load": { "kv_pressure_weight": 0.3 } });
        })),
        "one setting off its default is the one parameter written"
    );
    let bucket = config(PolicySettings::Bucket {
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.5,
        bucket_adjust_interval_secs: 10,
    });
    assert_eq!(
        bucket.policy_document(),
        Ok(edited(preset("bucket"), |document| {
            document["profiles"]["bucket"]["score"][0] =
                json!({ "bucket": { "abs_threshold": 32, "adjust_interval_s": 10 } });
        })),
    );
    let prefix_hash = config(PolicySettings::PrefixHash {
        prefix_token_count: 512,
        load_factor: 1.5,
        balance_abs_threshold: 4,
        cache_boundaries: vec![256, 1024, 4096],
    });
    assert_eq!(
        prefix_hash.policy_document(),
        Ok(edited(preset("prefix_hash"), |document| {
            document["view"] = json!({ "boundaries": [256, 1024, 4096] });
            document["profiles"]["prefix_hash"]["score"][0] = json!({ "consistent-hash": {
                "weight": 8.0,
                "prefix_tokens": 512,
                "bounded_load": { "factor": 1.5, "abs_margin": 4 },
            }});
        })),
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

#[test]
fn the_headers_of_a_routing_key_land_where_the_policy_reads_its_key() {
    let keyed = |policy: &str, headers: &[&str]| {
        let mut keyed = config(at_flag_defaults(policy));
        keyed.routing_key_override.headers =
            headers.iter().map(|name| (*name).to_string()).collect();
        keyed
    };

    // The manual policy reads the configured headers, then the routing
    // key header. Its rule names them: the first is the sticky one, and
    // the rule lists them where there is more than one, each once and
    // lowercase.
    let both = |document: &mut Value| {
        document["view"]["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
        let sessions = &mut document["bindings"][0]["sessions"];
        sessions["sticky_header"] = json!("x-session-id");
        sessions["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
    };
    let listed = ["X-Session-Id", "x-smg-routing-key", "x-session-id"];
    assert_eq!(
        keyed("manual", &listed).policy_document(),
        Ok(edited(preset("manual"), both))
    );
    assert_eq!(
        keyed("manual", &["x-session-id"]).policy_document(),
        Ok(edited(preset("manual"), both)),
        "the routing key header is read behind a configured one that is not it"
    );
    assert_eq!(
        keyed("manual", &[]).policy_document(),
        Ok(preset("manual")),
        "with no header configured the manual policy still reads the routing key header"
    );

    // The prefix ring reads the configured headers alone, and none is no
    // block. The hash ring reads the routing key header behind them.
    let sources = |name: &str, sources: Value| {
        edited(preset(name), |document| {
            document["bindings"][0]["routing_key"] = json!({ "sources": sources });
        })
    };
    assert_eq!(
        keyed("prefix_hash", &["x-session-id", "X-Other-Key"]).policy_document(),
        Ok(sources(
            "prefix_hash",
            json!([{ "header": "x-session-id" }, { "header": "x-other-key" }])
        ))
    );
    assert_eq!(
        keyed("prefix_hash", &[]).policy_document(),
        Ok(edited(preset("prefix_hash"), |document| {
            document["bindings"][0]["routing_key"] = Value::Null;
        }))
    );
    assert_eq!(
        keyed("consistent_hashing", &["x-session-id"]).policy_document(),
        Ok(sources(
            "consistent_hashing",
            json!([
                { "header": "x-session-id" },
                { "header": "x-smg-routing-key" },
                { "header": "x-forwarded-for", "client": true },
                { "header": "cookie", "client": true },
                { "header": "x-smg-target-worker", "target": true }
            ])
        ))
    );

    // Eight sources bound a block. The manual policy's rule holds the
    // configured headers, bounded by the view: 256 with the routing key
    // header.
    let eight: Vec<String> = (0..8).map(|n| format!("x-key-{n}")).collect();
    let eight: Vec<&str> = eight.iter().map(String::as_str).collect();
    assert_eq!(
        refusal(&keyed("consistent_hashing", &eight)).0,
        "routing_key_override.headers"
    );
    assert!(keyed("manual", &eight).policy_document().is_ok());
    // The hash ring's two client sources are counted apart: seven
    // configured headers and the routing key header are its eight.
    let seven = keyed("consistent_hashing", &eight[..7]);
    let written = seven.policy_document().expect("eight sources");
    assert!(validated(&written).is_ok());

    // The hash ring reads a configured `x-forwarded-for` both ways,
    // within the cap as a session's key and whole as a client's. The
    // document names it twice, and the controller's check takes it.
    let both_ways = keyed("consistent_hashing", &["x-forwarded-for"]);
    let document = sources(
        "consistent_hashing",
        json!([
            { "header": "x-forwarded-for" },
            { "header": "x-smg-routing-key" },
            { "header": "x-forwarded-for", "client": true },
            { "header": "cookie", "client": true },
            { "header": "x-smg-target-worker", "target": true }
        ]),
    );
    assert_eq!(both_ways.policy_document().as_ref(), Ok(&document));
    assert!(validated(&document).is_ok());

    // No other policy reads a key, so its document names no source.
    let keyed_policies = ["manual", "prefix_hash", "consistent_hashing"];
    for name in PRESETS
        .iter()
        .filter(|name| !keyed_policies.contains(*name))
    {
        assert_eq!(
            keyed(name, &["x-session-id"]).policy_document(),
            Ok(preset(name)),
            "{name}"
        );
    }
}

#[test]
fn the_override_adds_the_requests_id_where_a_policy_keeps_its_own_key() {
    let overridden = |policy: &str| {
        let mut overridden = config(at_flag_defaults(policy));
        overridden.routing_key_override.enabled = true;
        overridden
    };

    // The hash ring keeps its own placement under the override. Its
    // sources gain the request's id behind the headers, and every source
    // loses its turn and retry suffixes.
    assert_eq!(
        overridden("consistent_hashing").policy_document(),
        Ok(edited(preset("consistent_hashing"), |document| {
            document["bindings"][0]["routing_key"] = json!({
                "sources": [
                    { "header": "x-smg-routing-key" },
                    { "rid": true },
                    { "header": "x-forwarded-for", "client": true },
                    { "header": "cookie", "client": true },
                    { "header": "x-smg-target-worker", "target": true }
                ],
                "strip_lineage": true
            });
        }))
    );

    // The manual policy's rule says headers alone. The id and the strip
    // take the binding's block, and the rule then names no header.
    assert_eq!(
        overridden("manual").policy_document(),
        Ok(edited(preset("manual"), |document| {
            let binding = &mut document["bindings"][0];
            binding["routing_key"] = json!({
                "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                "strip_lineage": true
            });
            binding["sessions"]["sticky_header"] = json!("");
        }))
    );

    // Behind other configured headers the gateway reads the routing key
    // header as sent, and a block strips every source or none.
    let mut behind = overridden("consistent_hashing");
    behind.routing_key_override.headers = vec!["x-session-id".to_string()];
    assert_eq!(
        refusal(&behind),
        (
            "routing_key_override.headers",
            r#"["x-session-id"]"#.to_string()
        )
    );

    // The encode side reads no key and the override never places it.
    let mut plan = RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: Some(PolicySettings::RoundRobin),
            prefill_policy: Some(PolicySettings::ConsistentHashing),
            decode_policy: Some(PolicySettings::ConsistentHashing),
        },
        ..config(PolicySettings::RoundRobin)
    };
    plan.routing_key_override.enabled = true;
    let sources = plan.policy_document().map(|document| {
        document
            .bindings
            .first()
            .and_then(|binding| binding.routing_key.as_ref())
            .map(|key| key.sources.len())
    });
    assert_eq!(sources, Ok(Some(5)));
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

/// Beside a policy that reads no key the override lands as a session
/// rule on the policy's own binding: the gateway's sticky map in the
/// document's words. The key is the configured headers, then the
/// request's id, each without its turn and retry suffixes, and the view
/// lists the headers, so the rule's cap reads the key's text. The
/// controller's check takes the document, and its engine asks its host
/// for the count of a key's requests on a worker.
#[test]
fn the_override_lands_as_a_session_rule_beside_a_policy_that_reads_no_key() {
    for name in PLACED {
        let expected = edited(preset(name), |document| {
            document["view"]["headers"] = json!(["x-smg-routing-key"]);
            let binding = &mut document["bindings"][0];
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
        });
        assert_eq!(sticky(name).policy_document().as_ref(), Ok(&expected));
        let applied = validated(&expected).expect("the controller takes the document");
        assert!(applied.engine().needs().session_in_flight, "{name}");
    }
}

/// Beside the prefix ring the override lands as the same session rule
/// over the same sources: the configured headers, then the request's
/// own id, each without its turn and retry suffixes. The ring reads a
/// key too. A header's key leads the prompt, as under the policy alone:
/// the gateway hands its policy that key without its suffixes
/// (`model_gateway/src/policies/registry.rs:646`). The key the
/// request's own id hands names the session alone: the gateway's policy
/// never hashes it (`model_gateway/src/policies/prefix_hash.rs:384`).
/// With no header configured the id is the one source.
#[test]
fn the_override_lands_as_a_session_rule_beside_the_prefix_ring() {
    let expected = edited(preset("prefix_hash"), |document| {
        document["view"]["headers"] = json!(["x-smg-routing-key"]);
        let binding = &mut document["bindings"][0];
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
    });
    let settings = sticky("prefix_hash");
    assert_eq!(settings.policy_document().as_ref(), Ok(&expected));
    let applied = validated(&expected).expect("the controller takes the document");
    assert!(applied.engine().needs().session_in_flight);

    let mut bare = sticky("prefix_hash");
    bare.routing_key_override.headers.clear();
    let document = bare.policy_document().expect("a document");
    assert!(validated(&document).is_ok(), "{document:?}");
    let binding = document.bindings.into_iter().next().expect("the binding");
    let key = binding.routing_key.expect("the key's sources");
    assert_eq!(key.sources.len(), 1);
    assert!(key.sources[0].rid && key.strip_lineage);
    assert!(document.view.headers.is_empty());
}

/// The override's rule stands in the way of one pick. `passthrough`
/// picks alone, so the document's answer for its place is one pick
/// under the settings as they default. Under the override the answer
/// names the rule: no caller is told to pick by itself past a pinned
/// key.
#[test]
fn the_overrides_rule_keeps_a_place_from_one_pick() {
    let place_of = |settings: RoutingSettings| {
        let document = settings.policy_document().expect("a document");
        let applied = validated(&document).expect("the controller takes the document");
        let place = applied.engine().place_for("", Phase::Single);
        let ruled = Place::NotAlone {
            profile: "passthrough",
            in_the_way: InTheWay::Sessions,
        };
        (matches!(place, Place::Alone(_)), place == ruled)
    };
    let plain = config(at_flag_defaults("passthrough"));
    assert_eq!(place_of(plain), (true, false));
    assert_eq!(place_of(sticky("passthrough")), (false, true));
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

/// The override's rule has no document in three places. Each is refused
/// by the setting's name, in a sentence that says which it is: a pair of
/// which one side's policy reads the routing key itself, by that side,
/// a plan with an encode side, and a gateway of many routers. So under
/// every policy the rule lands beside, the prefix ring among them.
///
/// The gateway's sticky map places a role at a time, so beside a side
/// under the hash policy it places the other side alone
/// (`model_gateway/src/policies/registry.rs`, `sticky_for`). A binding's
/// rule places every side of its plan.
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
    let ring = || Some(PolicySettings::ConsistentHashing);
    for name in PLACED.into_iter().chain(["prefix_hash"]) {
        let mixed = [
            (pair_mode(ring(), None), "prefill"),
            (pair_mode(None, ring()), "decode"),
        ];
        for (mode, side) in mixed {
            let pair = RoutingSettings {
                mode,
                ..sticky(name)
            };
            assert_eq!(
                why(&pair),
                format!(
                    "the sticky sessions' rule places both sides of a pair, and the {side} \
                         side's policy reads the routing key itself"
                ),
                "{name}"
            );
        }
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

/// No document carries an overload threshold: one off its default writes
/// nothing, alone and beside the shed word. The switch and the shed word
/// land on the binding, each only off its default.
#[test]
fn an_overload_threshold_lands_on_no_binding_and_the_switch_and_the_shed_word_do() {
    let raised = RoutingSettings {
        worker_overload_waiting_requests: Some(32),
        worker_overload_token_usage: Some(0.9),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(raised.policy_document(), Ok(preset("round_robin")));
    let shedding = RoutingSettings {
        worker_overload_shed: true,
        ..raised
    };
    assert_eq!(
        shedding.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "shed_when_all_vetoed": true });
        })),
    );
    let off = RoutingSettings {
        worker_overload_protection: false,
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        off.policy_document(),
        Ok(edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        })),
    );
}

/// A threshold is the gateway's own beside a supplied document too: one
/// off its default says no other document than the settings without it,
/// so the supplied document is in force wherever it is without that
/// threshold.
#[test]
fn an_overload_threshold_beside_a_given_document_is_no_disagreement() {
    let given = preset("least_load");
    for settings in [defaults(), config(at_flag_defaults("least_load"))] {
        let raised = RoutingSettings {
            policy_spec: Some(given.clone()),
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            ..settings
        };
        assert_eq!(
            raised.policy_document_in_force(&defaults()),
            Ok(given.clone())
        );
    }
}

/// The two thresholds are no part of the comparison with a supplied
/// document. Beside settings that state nothing else, a document is in
/// force with one threshold at none and with both at none, whether its
/// block is on or off: what a block that is on means over no threshold
/// is its host's to say. The settings' own document still switches the
/// block off for both at none.
#[test]
fn overload_thresholds_at_none_beside_a_given_document_are_no_disagreement() {
    let on = preset("least_load");
    let off = edited(preset("least_load"), |document| {
        document["bindings"][0]["overload"] = json!({ "enabled": false });
    });
    for (waiting, usage) in [(None, Some(0.9)), (Some(32), None), (None, None)] {
        for given in [&on, &off] {
            let beside = RoutingSettings {
                policy_spec: Some(given.clone()),
                worker_overload_waiting_requests: waiting,
                worker_overload_token_usage: usage,
                ..defaults()
            };
            assert_eq!(
                beside.policy_document_in_force(&defaults()),
                Ok(given.clone()),
                "{waiting:?} and {usage:?}"
            );
        }
    }
    let own = RoutingSettings {
        worker_overload_waiting_requests: None,
        worker_overload_token_usage: None,
        ..defaults()
    };
    // A host whose own defaults state no threshold: the settings' two
    // stand beside a supplied document all the same.
    for (waiting, usage) in [(Some(8), None), (None, Some(0.8)), (Some(8), Some(0.8))] {
        let beside = RoutingSettings {
            policy_spec: Some(on.clone()),
            worker_overload_waiting_requests: waiting,
            worker_overload_token_usage: usage,
            ..defaults()
        };
        assert_eq!(
            beside.policy_document_in_force(&own),
            Ok(on.clone()),
            "{waiting:?} and {usage:?} over a host's defaults at none"
        );
    }
    let block = |document: PolicySpec| {
        document
            .bindings
            .first()
            .and_then(|binding| binding.overload.clone())
    };
    let switched_off = own.policy_document_in_force(&defaults());
    assert_eq!(switched_off, own.policy_document());
    assert_eq!(switched_off.map(block), Ok(block(off)));
}

#[test]
fn a_setting_no_document_says_is_refused_by_its_name() {
    let round_robin = || config(PolicySettings::RoundRobin);
    // The override's rule places both sides of a pair, and the hash
    // policy on the prefill side reads the routing key itself.
    let paired = RoutingSettings {
        mode: pair_mode(Some(PolicySettings::ConsistentHashing), None),
        ..sticky("random")
    };
    let mut selection = config(at_flag_defaults("cache_aware"));
    if let PolicySettings::CacheAware {
        selection_policy, ..
    } = &mut selection.policy
    {
        *selection_policy = Some("another".to_string());
    }
    for (refused, setting, value) in [
        (paired, "routing_key_override.enabled", "true"),
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

#[test]
fn a_number_past_the_documents_range_is_refused() {
    let for_ever = config(PolicySettings::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: u64::from(u32::MAX),
        assignment_mode: Assignment::Random,
    });
    assert_eq!(refusal(&for_ever).0, "max_idle_secs");
    let wide = config(PolicySettings::PowerOfTwo {
        load_metric: LoadMetricSetting::ResidentTokens {
            request_tokens: u64::from(u32::MAX) + 1,
        },
    });
    assert_eq!(refusal(&wide).0, "load_metric.request_tokens");
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

#[test]
fn a_balance_threshold_past_the_documents_width_lands_as_its_largest_number() {
    // A threshold no load reaches: the way a gateway's own policy
    // switches a spill off. The document's largest number is one too,
    // and the controller takes it.
    let never = usize::MAX;
    let bucket = config(PolicySettings::Bucket {
        balance_abs_threshold: never,
        balance_rel_threshold: 1.5,
        bucket_adjust_interval_secs: 5,
    });
    let prefix_hash = config(PolicySettings::PrefixHash {
        prefix_token_count: 256,
        load_factor: 1.25,
        balance_abs_threshold: never,
        cache_boundaries: Vec::new(),
    });
    let cache_aware = config(cache_aware(0.3, never, 1.5, 120, 67_108_864));
    for (config, at) in [
        (bucket, "/profiles/bucket/score/0/bucket/abs_threshold"),
        (
            prefix_hash,
            "/profiles/prefix_hash/score/0/consistent-hash/bounded_load/abs_margin",
        ),
        (
            cache_aware,
            "/profiles/cache_aware/score/1/prefix-affinity/spill/abs",
        ),
    ] {
        let document = config.policy_document().expect("a document");
        let written = serde_json::to_value(&document).expect("a written form");
        assert_eq!(written.pointer(at), Some(&json!(u32::MAX)), "{at}");
        assert_eq!(validated(&document).err(), None, "{at}");
    }
}

#[test]
fn an_index_the_configuration_never_sweeps_is_never_trimmed() {
    // The gateway's own policy trims its index at its sweep alone, and
    // sweeps nothing at an interval of zero: the configured size bounds
    // nothing. The document says so with its longest cadence and its
    // largest size, which no write of the controller's record meets.
    let swept_every = |secs| cache_aware(0.3, 64, 1.5, secs, 67_108_864);
    let unswept = config(swept_every(0));
    let document = unswept.policy_document().expect("a document");
    assert_eq!(
        document.index.get("*"),
        Some(&IndexSpec {
            max_size: u64::MAX,
            sweep_s: u32::MAX,
            ..IndexSpec::default()
        })
    );
    assert_eq!(validated(&document).err(), None);
    // A swept index keeps the size it is trimmed to.
    let swept = config(cache_aware(0.3, 64, 1.5, 60, 1024))
        .policy_document()
        .expect("a document");
    assert_eq!(
        swept.index.get("*"),
        Some(&IndexSpec {
            max_size: 1024,
            sweep_s: 60,
            ..IndexSpec::default()
        })
    );

    // Two sides are still compared as the configuration spells them.
    let sides = RoutingSettings {
        mode: pair_mode(Some(swept_every(0)), Some(swept_every(60))),
        ..config(PolicySettings::RoundRobin)
    };
    assert_eq!(
        refusal(&sides).1,
        r#"["tree",180,67108864,0] and ["tree",180,67108864,60]"#
    );
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
    // Of many routers the main policy's single binding stands behind it.
    let many = many_routers(one_router.clone())
        .policy_document()
        .expect("the document of both families");
    assert_eq!(many.bindings.len(), 2);
    assert_eq!(many.bindings[0].plan, Some(PlanName::Epd));
    assert_eq!(many.bindings[0].sessions, rule);
    assert_eq!(many.bindings[1].sessions, None);
    assert_eq!(validated(&many).err(), None);

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
                    reason: "the manual policy places by the binding's one session rule, so both sides of a pair are manual or neither is",
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
