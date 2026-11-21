//! The start of a gateway that decides in its own process: what it refuses
//! of a document its operator supplied, and its answer for a document the
//! gateway does not run.

use std::{error::Error, sync::Arc};

use smg_policy_controller::{
    document::{DecodePick, OverloadSpec, Pairing, PlanName, PolicySpec, ShadowSpec},
    engine::Phase,
    stages::Catalog,
};

use super::{start_in_process, would_start};
use crate::config::{PdPairingMode, PolicyConfig, RouterConfig, RoutingMode};

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
    let stages = Arc::new(Catalog::builtin());
    let answer = |gateway: &RouterConfig, document: &PolicySpec| {
        would_start(gateway, &stages, document)
            .err()
            .map(|refused| refused.to_string())
    };

    // A place the engine does not decide in this process yet.
    let scored = PolicySpec::preset("least_load")?;
    let words = answer(&running, &scored).ok_or("a scorer was taken")?;
    assert!(
        words.starts_with(
            "the policy document's profile `least_load` is not decided in this process yet"
        ),
        "{words}"
    );
    assert_eq!(words, refusal(&supplied(scored))?);

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
    let mut twice = PolicySpec::pair_presets("round_robin", "round_robin")?;
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
    let pair = || PolicySpec::pair_presets("round_robin", "round_robin");
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

/// The document's own rules, apart from its bindings. The placement pairs
/// by the routing settings' mode and picks a pair's decode with its
/// prefill, and no decision is kept for explain: a supplied document that
/// states one of them otherwise is refused by the field. A pair plan's rule
/// in a document that plans no pair has no request, and states nothing.
#[test]
fn a_supplied_documents_own_rules_stop_the_start_by_field() -> Case {
    let pair = || PolicySpec::pair_presets("round_robin", "round_robin");
    let stated = |field: &str, because: &str| {
        format!(
            "the policy document states `{field}`, which is not run in this process yet: {because}"
        )
    };

    let mut document = pair()?;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    assert_eq!(
        refusal(&supplied(document))?,
        stated(
            "plans.pd-pair.decode",
            "a pair's decode is picked with its prefill"
        )
    );

    let mut document = pair()?;
    document.plans.pd_pair.pairing = Pairing::Strict;
    assert_eq!(
        refusal(&supplied(document.clone()))?,
        stated("plans.pd-pair.pairing", BY_SETTINGS)
    );
    // The mode the routing settings state is the one the placement runs.
    let strict = RouterConfig {
        pd_pairing_mode: PdPairingMode::Strict,
        ..supplied(document)
    };
    assert!(start_in_process(&strict, None)?.is_some());

    // No binding plans a pair: the pair plan's rules have no request.
    let mut document = PolicySpec::preset("round_robin")?;
    document.plans.pd_pair.pairing = Pairing::Strict;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    assert!(start_in_process(&supplied(document), None)?.is_some());

    let mut document = PolicySpec::preset("round_robin")?;
    document.explain.sample = 8;
    assert_eq!(
        refusal(&supplied(document))?,
        stated("explain.sample", "no decision is kept for explain")
    );
    Ok(())
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
