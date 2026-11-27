//! Cases of a policy's own settings. A gateway that sets none means the
//! `random` preset, and each policy at its flags' defaults is its preset.
//! A setting off its default lands in the document or is refused. One
//! case reads no setting: it holds the presets to the form the converter
//! writes over.

use super::*;

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
        // The hash index calls no warm-up slice: its stage keeps nobody
        // young and nobody thin.
        profile["score"][0] = json!({
            "warmup-slice": {
                "weight": 8.0, "secs": 0, "blocks": 2048, "thin_ratio": 0.0, "miss_share": 0.5
            }
        });
        // The hash index reads no match threshold: its stage counts a
        // holder at every share, and the slice alone keeps the setting.
        profile["score"][1] = json!({ "prefix-affinity": {
            "weight": 4.0,
            "threshold": 0.0,
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
fn a_hash_index_the_configuration_never_sweeps_keeps_its_size() {
    // Under a hash index the controller keeps a record of sent prefixes
    // and trims it at its own write, so the document keeps the configured
    // size at an interval of zero and names its longest cadence alone.
    let mut policy = cache_aware(0.3, 64, 1.5, 0, 67_108_864);
    if let PolicySettings::CacheAware {
        cache_index,
        cache_boundaries,
        ..
    } = &mut policy
    {
        *cache_index = IndexKind::Hash;
        *cache_boundaries = vec![512, 2048];
    }
    let document = config(policy).policy_document().expect("a document");
    assert_eq!(
        document.index.get("*"),
        Some(&IndexSpec {
            kind: IndexKind::Hash,
            ttl_s: 180,
            max_size: 67_108_864,
            sweep_s: u32::MAX,
        })
    );
    assert_eq!(validated(&document).err(), None);
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
