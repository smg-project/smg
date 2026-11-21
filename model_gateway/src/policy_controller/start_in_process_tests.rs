//! The start of a gateway that decides in its own process: what it refuses
//! of a document its operator supplied.

use std::error::Error;

use smg_policy_controller::document::{OverloadSpec, PlanName, PolicySpec, ShadowSpec};

use super::start_in_process;
use crate::config::RouterConfig;

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
        policy: crate::config::PolicyConfig::RoundRobin,
        ..RouterConfig::default()
    };
    let said = config.policy_document()?;
    assert!(said.bindings[0].overload.is_some());
    config.policy_spec = Some(said);
    assert!(start_in_process(&config, None)?.is_some());
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
