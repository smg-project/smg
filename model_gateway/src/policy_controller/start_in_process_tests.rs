//! The start of a gateway that decides in its own process: what it refuses
//! of a document its operator supplied, and its answer for a document the
//! gateway does not run.

use std::{error::Error, sync::Arc};

use smg_policy_controller::{
    document::{
        DecodePick, IndexKind, IndexSpec, OverloadSpec, Pairing, PlanName, PolicySpec, ShadowSpec,
        PRESETS,
    },
    engine::Phase,
    stages::StageRegistry,
};

use super::{fed, start_in_process, would_start, InProcess, StartError};
use crate::config::{
    CacheIndexKind, ManualAssignmentMode, PdPairingMode, PolicyConfig, PowerOfTwoLoadMetric,
    RouterConfig, RoutingKeyOverrideConfig, RoutingMode,
};

type Case = Result<(), Box<dyn Error>>;

/// A configuration at its defaults but the setting, with `document`
/// supplied.
fn supplied(document: PolicySpec) -> RouterConfig {
    RouterConfig {
        policy_controller: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    }
}

/// The words the start refuses `config` in.
fn refusal(config: &RouterConfig) -> Result<String, Box<dyn Error>> {
    let refused = start_in_process(config, None)
        .err()
        .ok_or("the start took the document")?;
    Ok(refused.to_string())
}

/// The refusal of a rule of the binding at `at` with the pattern `model`.
fn not_run(at: usize, model: &str, field: &str, because: &str) -> String {
    format!(
        "the policy document's binding {at} (`{model}`) states `{field}`, \
         which is not run in this process yet: {because}"
    )
}

const BY_SETTINGS: &str = "the gateway's placement runs the routing settings' own";

/// A pair document whose binding holds its sides' phases alone, as a
/// gateway that places single requests asks of it.
fn pair_sides() -> Result<PolicySpec, Box<dyn Error>> {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin")?;
    document.bindings[0].matcher.phase = vec![Phase::Prefill, Phase::Decode];
    Ok(document)
}

/// The gateway's own placement runs a binding's overload thresholds, its
/// key sources and its sessions rule from the routing settings, and runs no
/// shadow. A supplied document that states one of them otherwise would
/// start with a rule no code runs: the start refuses it by the binding and
/// the field.
#[test]
fn a_supplied_bindings_own_rules_stop_the_start_by_binding_and_field() -> Case {
    let round_robin = || PolicySpec::preset("round_robin");

    let mut document = round_robin()?;
    document.bindings[0].overload = Some(OverloadSpec {
        waiting_requests: Some(3),
        ..OverloadSpec::default()
    });
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(1, "*", "overload", BY_SETTINGS)
    );

    // An overload block that states the block's own defaults states nothing.
    let mut document = round_robin()?;
    document.bindings[0].overload = Some(OverloadSpec::default());
    assert!(start_in_process(&supplied(document), None).is_ok());

    let mut document = round_robin()?;
    let hashed = PolicySpec::preset("consistent_hashing")?;
    document.bindings[0]
        .routing_key
        .clone_from(&hashed.bindings[0].routing_key);
    assert!(document.bindings[0].routing_key.is_some());
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(1, "*", "routing_key", BY_SETTINGS)
    );

    let mut document = round_robin()?;
    let manual = PolicySpec::preset("manual")?;
    document.view = manual.view.clone();
    document.bindings[0]
        .sessions
        .clone_from(&manual.bindings[0].sessions);
    assert!(document.bindings[0].sessions.is_some());
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(1, "*", "sessions", BY_SETTINGS)
    );

    // The second binding states the rule: the refusal names it.
    let mut document = round_robin()?;
    let second = document.profiles["round_robin"].clone();
    document.profiles.insert("second".to_owned(), second);
    let mut named = document.bindings[0].clone();
    named.matcher.model = "alpha-*".to_owned();
    document.bindings.insert(0, named);
    document.bindings[1].shadow = Some(ShadowSpec::Profile("second".to_owned()));
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(2, "*", "shadow", "no shadow profile runs here")
    );
    Ok(())
}

/// A rule the routing settings state too is the one the placement runs: a
/// supplied document that says what the settings say starts.
#[test]
fn a_supplied_document_that_says_what_the_settings_run_starts() -> Case {
    let mut config = RouterConfig {
        policy_controller: true,
        worker_overload_waiting_requests: Some(3),
        policy: PolicyConfig::RoundRobin,
        ..RouterConfig::default()
    };
    let said = config.policy_document()?;
    assert!(said.bindings[0].overload.is_some());
    config.policy_spec = Some(said);
    assert!(start_in_process(&config, None)?.is_some());
    Ok(())
}

/// A gateway is asked of a document it does not run, as the control plane's
/// dry run asks. The answer is what a start under that document answers:
/// each refusal is the start's own, word for word, and a document the start
/// takes is taken. So a document that passes the dry run does not stop the
/// next start.
#[test]
fn a_document_the_gateway_does_not_run_is_answered_as_a_start_under_it() -> Case {
    // The gateway asked runs the document its settings mean.
    let running = RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    };
    let stages = Arc::new(StageRegistry::builtin());
    let answer = |gateway: &RouterConfig, document: &PolicySpec| {
        would_start(gateway, &stages, document)
            .err()
            .map(|refused| refused.to_string())
    };

    // A place the engine does not decide in this process yet: one that
    // places by a hash ring.
    let ringed = PolicySpec::preset("consistent_hashing")?;
    let words = answer(&running, &ringed).ok_or("a hash ring was taken")?;
    assert!(
        words.starts_with(
            "the policy document's profile `consistent_hashing` is not decided in this process yet"
        ),
        "{words}"
    );
    assert_eq!(words, refusal(&supplied(ringed))?);

    // A rule nothing runs.
    let mut ruled = PolicySpec::preset("round_robin")?;
    ruled.bindings[0].overload = Some(OverloadSpec {
        waiting_requests: Some(3),
        ..OverloadSpec::default()
    });
    let words = not_run(1, "*", "overload", BY_SETTINGS);
    assert_eq!(answer(&running, &ruled), Some(words.clone()));
    assert_eq!(words, refusal(&supplied(ruled))?);

    // A second pair binding.
    let mut twice = pair_sides()?;
    let again = twice.bindings[0].clone();
    twice.bindings.push(again);
    let words = not_run(2, "*", "plan", "an earlier binding plans the pair");
    assert_eq!(answer(&running, &twice), Some(words.clone()));
    assert_eq!(words, refusal(&supplied(twice))?);

    // A document the start takes.
    let taken = PolicySpec::preset("round_robin")?;
    assert_eq!(answer(&running, &taken), None);
    assert!(start_in_process(&supplied(taken), None)?.is_some());

    // A gateway whose routing settings say another document than the one
    // asked: a start under both stops at their first difference, and the
    // answer is that refusal.
    let other = RouterConfig {
        policy: PolicyConfig::RoundRobin,
        ..running.clone()
    };
    let asked = PolicySpec::preset("random")?;
    let beside = RouterConfig {
        policy_spec: Some(asked.clone()),
        ..other.clone()
    };
    let words = answer(&other, &asked).ok_or("a document beside other settings was taken")?;
    assert_eq!(words, refusal(&beside)?);
    Ok(())
}

/// The registry keeps one policy for each side of a pair, for every model.
/// A pair binding that names some models would never be found, and a
/// second pair binding never asked: the start refuses each by name.
#[test]
fn a_supplied_pair_binding_names_every_model_and_stands_alone() -> Case {
    let pair = pair_sides;
    assert!(start_in_process(&supplied(pair()?), None)?.is_some());

    let mut document = pair()?;
    document.bindings[0].matcher.model = "alpha-*".to_owned();
    assert_eq!(document.bindings[0].plan, Some(PlanName::PdPair));
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(
            1,
            "alpha-*",
            "match.model",
            "one policy places each side of a pair, for every model"
        )
    );

    let mut document = pair()?;
    let again = document.bindings[0].clone();
    document.bindings.push(again);
    assert_eq!(
        refusal(&supplied(document))?,
        not_run(2, "*", "plan", "an earlier binding plans the pair")
    );
    Ok(())
}

/// A pair binding plans no single request. One that covers the single
/// phase, by a list that names no phase or by name, takes its models'
/// single requests from every binding behind it: the classic policy would
/// place them and no line would say so. In a gateway that places single
/// requests the start refuses it by `match.phase`, and takes the binding
/// that holds its sides' phases alone. A pair mode places no single
/// request, and starts under either.
#[test]
fn a_pair_binding_holds_its_sides_phases_in_a_gateway_that_places_single_requests() -> Case {
    let pair = || PolicySpec::pair_presets("round_robin", "round_robin");
    let words = not_run(
        1,
        "*",
        "match.phase",
        "this gateway places single requests and a pair binding plans none: \
         a pair binding holds its sides' phases alone (`phase: [prefill, decode]`)",
    );
    // A regular gateway places single requests alone.
    assert_eq!(refusal(&supplied(pair()?))?, words);
    let mut named = pair()?;
    named.bindings[0].matcher.phase = vec![Phase::Single, Phase::Prefill, Phase::Decode];
    assert_eq!(refusal(&supplied(named))?, words);
    assert!(start_in_process(&supplied(pair_sides()?), None)?.is_some());

    // A gateway of many routers places them beside its pairs.
    let many_routers = |document: PolicySpec| RouterConfig {
        enable_igw: true,
        ..supplied(document)
    };
    assert_eq!(refusal(&many_routers(pair()?))?, words);
    assert!(start_in_process(&many_routers(pair_sides()?), None)?.is_some());

    // The encode plan's binding names its three sides.
    let encode = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin")?;
    assert_eq!(
        refusal(&supplied(encode))?,
        not_run(
            1,
            "*",
            "match.phase",
            "this gateway places single requests and a pair binding plans none: \
             a pair binding holds its sides' phases alone (`phase: [encode, prefill, decode]`)",
        )
    );

    // A pair mode places no single request.
    let pair_mode = RouterConfig {
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        ..supplied(pair()?)
    };
    assert!(start_in_process(&pair_mode, None)?.is_some());
    Ok(())
}

/// The document's own rules, apart from its bindings. The placement pairs
/// by the routing settings' mode and picks a pair's decode with its
/// prefill, and no decision is kept for explain: a supplied document that
/// states one of them otherwise is refused by the field. A pair plan's rule
/// in a document that plans no pair has no request, and states nothing.
#[test]
fn a_supplied_documents_own_rules_stop_the_start_by_field() -> Case {
    let pair = pair_sides;
    // Each refusal names the field and says what the operator can do.
    let stated = |field: &str, because: &str, instead: &str| {
        format!(
            "the policy document states `{field}`, which is not run in this process yet: {because}. \
             {instead}, or run the gateway under a controller's address"
        )
    };
    let by_mode = "the gateway's placement pairs by the routing settings' mode";

    let mut document = pair()?;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    assert_eq!(
        refusal(&supplied(document))?,
        stated(
            "plans.pd-pair.decode",
            "a pair's decode is picked with its prefill",
            "Remove the field"
        )
    );

    let mut document = pair()?;
    document.plans.pd_pair.pairing = Pairing::Strict;
    assert_eq!(
        refusal(&supplied(document.clone()))?,
        stated("plans.pd-pair.pairing", by_mode, "Remove the field")
    );
    // The mode the routing settings state is the one the placement runs:
    // a document that states it starts, and one that leaves it out is told
    // the mode to state.
    let strict = |document: PolicySpec| RouterConfig {
        pd_pairing_mode: PdPairingMode::Strict,
        ..supplied(document)
    };
    assert!(start_in_process(&strict(document), None)?.is_some());
    assert_eq!(
        refusal(&strict(pair()?))?,
        stated(
            "plans.pd-pair.pairing",
            by_mode,
            "State `strict`, the routing settings' mode"
        )
    );

    // No binding plans a pair: the pair plan's rules have no request.
    let mut document = PolicySpec::preset("round_robin")?;
    document.plans.pd_pair.pairing = Pairing::Strict;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    assert!(start_in_process(&supplied(document), None)?.is_some());

    let mut document = PolicySpec::preset("round_robin")?;
    document.explain.sample = 8;
    assert_eq!(
        refusal(&supplied(document))?,
        stated(
            "explain.sample",
            "no decision is kept for explain",
            "Remove the field"
        )
    );
    Ok(())
}

/// Every routing policy with each setting at its flag's default, in the
/// order of the policy crate's presets.
fn policies() -> Vec<PolicyConfig> {
    vec![
        PolicyConfig::Random,
        PolicyConfig::RoundRobin,
        PolicyConfig::Passthrough,
        PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ExpectedWait,
        },
        PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        },
        PolicyConfig::ConsistentHashing,
        PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        },
        PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode: ManualAssignmentMode::Random,
        },
        PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864),
        PolicyConfig::Bucket {
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        },
    ]
}

/// The routing settings the settings' translator is held to, each with the
/// policy controller picked: every policy as the main one of a regular
/// gateway, of many routers, of a pair mode and of an encode pair mode;
/// every pair of side policies; each pairing mode; and the settings that
/// land on a binding or in a stage.
fn settings() -> Vec<(String, RouterConfig)> {
    let policies = policies();
    let pair =
        |prefill: Option<PolicyConfig>, decode: Option<PolicyConfig>| RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: prefill,
            decode_policy: decode,
        };
    let round_robin = || RouterConfig {
        policy_controller: true,
        policy: PolicyConfig::RoundRobin,
        ..RouterConfig::default()
    };
    let mut settings = Vec::new();
    for policy in &policies {
        let name = policy.name();
        let main = RouterConfig {
            policy: policy.clone(),
            ..round_robin()
        };
        settings.push((
            format!("{name}, many routers"),
            RouterConfig {
                enable_igw: true,
                ..main.clone()
            },
        ));
        settings.push((
            format!("{name}, a pair mode"),
            RouterConfig {
                mode: pair(None, None),
                ..main.clone()
            },
        ));
        settings.push((
            format!("{name}, an encode pair mode"),
            RouterConfig {
                mode: RoutingMode::EncodePrefillDecode {
                    encode_urls: vec![],
                    prefill_urls: vec![],
                    decode_urls: vec![],
                    encode_policy: None,
                    prefill_policy: None,
                    decode_policy: None,
                },
                ..main.clone()
            },
        ));
        settings.push((
            format!("{name}, the routing-key override"),
            RouterConfig {
                routing_key_override: RoutingKeyOverrideConfig {
                    enabled: true,
                    ..RoutingKeyOverrideConfig::default()
                },
                ..main.clone()
            },
        ));
        settings.push((
            format!("{name}, overload thresholds"),
            RouterConfig {
                worker_overload_waiting_requests: Some(3),
                worker_overload_token_usage: Some(0.5),
                ..main.clone()
            },
        ));
        for decode in &policies {
            let sides = format!("{name} prefills, {} decodes", decode.name());
            settings.push((
                sides.clone(),
                RouterConfig {
                    mode: pair(Some(policy.clone()), Some(decode.clone())),
                    ..round_robin()
                },
            ));
            settings.push((
                format!("{sides}, many routers"),
                RouterConfig {
                    mode: pair(Some(policy.clone()), Some(decode.clone())),
                    enable_igw: true,
                    ..round_robin()
                },
            ));
        }
        settings.push((name.to_owned(), main));
    }
    for mode in [
        PdPairingMode::Off,
        PdPairingMode::Lenient,
        PdPairingMode::Strict,
    ] {
        settings.push((
            format!("pairs under {}", mode.as_str()),
            RouterConfig {
                mode: pair(None, None),
                pd_pairing_mode: mode,
                ..round_robin()
            },
        ));
    }
    for load_metric in [
        PowerOfTwoLoadMetric::Requests,
        PowerOfTwoLoadMetric::LeastRequests,
    ] {
        settings.push((
            format!("power_of_two by {load_metric:?}"),
            RouterConfig {
                policy: PolicyConfig::PowerOfTwo { load_metric },
                ..round_robin()
            },
        ));
    }
    for assignment_mode in [
        ManualAssignmentMode::MinLoad,
        ManualAssignmentMode::MinGroup,
    ] {
        settings.push((
            format!("manual by {assignment_mode:?}"),
            RouterConfig {
                policy: PolicyConfig::Manual {
                    eviction_interval_secs: 60,
                    max_idle_secs: 14_400,
                    assignment_mode,
                },
                ..round_robin()
            },
        ));
    }
    settings
}

/// The settings' own document always starts. A gateway writes the document
/// its routing settings mean, and no refusal of the start's own meets it:
/// the engine decides the places it picks alone and the classic policy of
/// the setting decides the rest. The same document handed back beside the
/// same settings, as a dry run of the document in force hands it, is taken
/// too.
///
/// Two refusals stand as they do under a controller's address, and neither
/// is the start's own. A setting no document says stops the start in the
/// translator's words. A document the policy crate's check refuses stops
/// it in that check's words: a session rule under a pair plan, which the
/// manual policy writes on a side of a pair.
#[test]
fn the_settings_own_document_always_starts() {
    let names: Vec<_> = policies().iter().map(PolicyConfig::name).collect();
    assert_eq!(names, PRESETS, "a policy for each preset");

    let stages = Arc::new(StageRegistry::builtin());
    let mut said = 0;
    // The settings the check refuses, and every setting that breaks the
    // rule, each with what the start answered.
    let (mut unrun, mut wrong) = (Vec::new(), Vec::new());
    for (which, config) in settings() {
        let started = start_in_process(&config, None);
        let Ok(document) = config.policy_document() else {
            if !matches!(started, Err(StartError::Document(_))) {
                wrong.push(format!("{which}, no document: {:?}", started.err()));
            }
            continue;
        };
        if document
            .validate_in(&stages, &config.denied_view_headers())
            .is_err()
        {
            if !matches!(started, Err(StartError::Spec(_))) {
                wrong.push(format!("{which}, unchecked: {:?}", started.err()));
            }
            unrun.push(which);
            continue;
        }
        if !matches!(started, Ok(Some(_))) {
            wrong.push(format!("{which}: {:?}", started.err()));
        }
        if let Err(refused) = would_start(&config, &stages, &document) {
            wrong.push(format!("{which}, handed back: {refused}"));
        }
        said += 1;
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert!(
        unrun.iter().all(|which| which.contains("manual")),
        "{unrun:#?}"
    );
    assert!(said > unrun.len(), "{said} settings start");
}

/// A side's policy places the pair requests of every model. In a gateway
/// that places pairs, a binding of some models would be passed by for
/// those models' pair requests: the start refuses it unless it holds
/// single requests alone. A gateway that places no pair starts under it.
#[test]
fn a_binding_of_some_models_holds_single_requests_in_a_gateway_that_places_pairs() -> Case {
    let two_bindings = |phase: Vec<Phase>| -> Result<PolicySpec, Box<dyn Error>> {
        let mut document = PolicySpec::preset("round_robin")?;
        let mut named = document.bindings[0].clone();
        named.matcher.model = "alpha-*".to_owned();
        named.matcher.phase = phase;
        document.bindings.insert(0, named);
        Ok(document)
    };
    let pair_mode = |document: PolicySpec| RouterConfig {
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        ..supplied(document)
    };
    let many_routers = |document: PolicySpec| RouterConfig {
        enable_igw: true,
        ..supplied(document)
    };
    let held = "one policy places each side of a pair, for every model: \
                a binding of some models holds single requests alone (`phase: [single]`)";

    // No pair is placed: the binding decides its models' requests.
    assert!(start_in_process(&supplied(two_bindings(vec![])?), None)?.is_some());
    let gateways: [&dyn Fn(PolicySpec) -> RouterConfig; 2] = [&pair_mode, &many_routers];
    for places_pairs in gateways {
        assert_eq!(
            refusal(&places_pairs(two_bindings(vec![])?))?,
            not_run(1, "alpha-*", "match.phase", held)
        );
        let single = two_bindings(vec![Phase::Single])?;
        assert!(start_in_process(&places_pairs(single), None)?.is_some());
    }
    Ok(())
}

/// A prefix index kept as a placement map (`kind: hash`) is a rule nothing
/// in this process runs: the engine keeps a model's sent prefixes as a
/// tree. The start refuses it by the index's key and says which setting
/// starts, for the document the routing settings mean and for a supplied
/// one alike. The tree kind starts.
#[test]
fn a_hash_index_is_refused_by_its_key_whoever_wrote_the_document() -> Case {
    let refused = |model: &str| {
        format!(
            "the policy document states `index.{model}.kind: hash`, which is not run in this \
             process yet: no placement map by a prompt's head is kept here. Start with \
             `--cache-index tree` or leave the flag out (in a document, state `kind: tree` \
             or remove the field)"
        )
    };
    let by_settings = |kind: CacheIndexKind| {
        let mut policy = PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864);
        if let PolicyConfig::CacheAware {
            cache_index,
            cache_boundaries,
            ..
        } = &mut policy
        {
            *cache_index = kind;
            if kind == CacheIndexKind::Hash {
                *cache_boundaries = vec![16, 64];
            }
        }
        RouterConfig {
            policy_controller: true,
            policy,
            ..RouterConfig::default()
        }
    };
    assert!(start_in_process(&by_settings(CacheIndexKind::Tree), None)?.is_some());
    assert_eq!(refusal(&by_settings(CacheIndexKind::Hash))?, refused("*"));

    // A supplied document, by the key of the index that states the kind.
    let mut document = PolicySpec::preset("round_robin")?;
    document.view.boundaries = vec![16, 64];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("chat-*".to_owned(), hash);
    assert_eq!(refusal(&supplied(document))?, refused("chat-*"));
    Ok(())
}

/// The routers ask the engine for every request of a document it decides
/// whole from what this process feeds. Round robin's is one: each of its
/// places is picked alone. The default policy's is one: its profile reads
/// who holds the prompt and the workers' load reports, and this process
/// feeds a whole decision both. So is every policy whose stages read a
/// worker's record, its report or the prompt's size. A policy that places
/// by a hash ring or by a header of the request is not yet: the routers
/// keep their classic pass under it, and a supplied document of it is
/// refused by its first stage that asks more than a pick.
#[test]
fn the_engine_decides_whole_what_this_process_feeds() -> Case {
    let started = |policy: PolicyConfig| -> Result<InProcess, Box<dyn Error>> {
        let config = RouterConfig {
            policy_controller: true,
            policy,
            ..RouterConfig::default()
        };
        Ok(start_in_process(&config, None)?.ok_or("the setting starts an engine")?)
    };
    assert!(started(PolicyConfig::RoundRobin)?.decides_whole());
    let default = started(PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864))?;
    assert!(default.decides_whole());
    let declared = default
        .engine
        .spec()
        .engine()
        .needs_for("a-model", Phase::Single);
    assert!(declared.affinity && declared.loads);
    assert!(declared.within(fed()));
    // Of every routing policy, the engine decides whole all but the three
    // that place by a hash ring or by a header of the request.
    for policy in policies() {
        let name = policy.name();
        let classic = matches!(name, "consistent_hashing" | "prefix_hash" | "manual");
        assert_eq!(started(policy)?.decides_whole(), !classic, "{name}");
    }
    // The bucket stage places by the prompt's size and says so, and this
    // process counts the size: a route that would send a body on unread
    // reads it under this document.
    let bucket = policies()
        .into_iter()
        .find(|policy| policy.name() == "bucket");
    let bucket = started(bucket.ok_or("the bucket policy")?)?;
    let declared = bucket
        .engine
        .spec()
        .engine()
        .needs_for("a-model", Phase::Single);
    assert!(declared.prompt_size && declared.within(fed()));

    // A supplied document whose profile reads who holds the prompt is
    // decided whole, so it starts: the prefix stage by itself, and the
    // warm-up slice beside it, which reads a report. One that places by a
    // hash ring is refused by the first stage that asks more than a pick.
    for kept in [
        &["prefix-affinity"][..],
        &["warmup-slice", "prefix-affinity"][..],
    ] {
        let mut held = PolicySpec::preset("cache_aware")?;
        let profile = held.profiles.get_mut("cache_aware").ok_or("the profile")?;
        profile.score.retain(|score| kept.contains(&score.name()));
        let started = start_in_process(&supplied(held), None)?.ok_or("an engine")?;
        assert!(started.decides_whole(), "{kept:?}");
    }
    let refused = refusal(&supplied(PolicySpec::preset("consistent_hashing")?))?;
    assert!(
        refused.starts_with("the policy document's profile `consistent_hashing` is not decided"),
        "{refused}"
    );
    Ok(())
}
