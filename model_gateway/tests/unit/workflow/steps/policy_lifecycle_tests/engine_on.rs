//! The lifecycle's cases with Grand Central on.
//!
//! Each case of the parent runs here once more, on a context whose engine
//! decides under the spec the routing settings mean. A worker's `policy`
//! label then builds no policy object. The model's first registration
//! decides the policy by it, and the spec that names it is put in force on
//! the running engine. So what a case reads as the model's policy is the
//! policy of the profile that places the model's single requests there,
//! and `None` once the settings' own profile places them again.

use smg_grand_central::document::PRESETS;

use super::*;
use crate::{
    app_context::AppContextBuilder, policies::PolicyRegistry,
    routers::common::snapshot::WorkerMonitor,
};

thread_local! {
    /// The contexts this thread's case made with the engine on: a case
    /// that made none ran under the classic policies and proves nothing
    /// here ([`with_the_engine_on`]).
    static STARTED: Cell<u32> = const { Cell::new(0) };
}

/// An app context over `config` with Grand Central on, started as a
/// test's context starts it: over the stock stages, with the maker of the
/// specs a label builds.
pub(super) fn app_context(mut config: RouterConfig) -> Arc<AppContext> {
    STARTED.set(STARTED.get() + 1);
    config.grand_central = true;
    let mut context = AppContext::clone(&app_context_for_tests(
        config.clone(),
        Arc::new(WorkerRegistry::new()),
        None,
    ));
    context.policy_registry = Arc::new(PolicyRegistry::of_config(&config));
    let monitor = WorkerMonitor::new(
        Arc::clone(&context.worker_registry),
        Arc::clone(&context.policy_registry),
        1,
        false,
        true,
    );
    AppContextBuilder::start_engine_by_hand(
        &config,
        None,
        &context.policy_registry,
        &context.worker_registry,
        &monitor,
    )
    .expect("the settings' own spec");
    Arc::new(context)
}

/// The policy a label gave `model`: the policy that the profile which
/// places the model's single requests is named for, read by the crate's
/// rule. `None` where the settings' own profile places them, as before the
/// model's first label and behind its last worker.
pub(super) fn policy_of(app_context: &AppContext, model: &str) -> Option<&'static str> {
    let policies = &app_context.policy_registry;
    assert_eq!(policies.classic_made_or_asked(), 0, "no policy object");
    let placing = policies.placing_profile(model)?;
    let policy = smg_grand_central::document::RoutingSettings::label_policy(&placing, model)?;
    PRESETS.into_iter().find(|name| *name == policy)
}

/// Runs `case`, a case of the parent, with the engine on. The case must
/// make its context here: one that makes it on another thread would run
/// under the classic policies.
fn with_the_engine_on(case: fn()) {
    ENGINE_ON.set(true);
    case();
    ENGINE_ON.set(false);
    assert!(
        STARTED.get() > 0,
        "the case made no context with the engine on"
    );
}

#[test]
fn the_policy_leaves_with_the_last_worker_however_often_it_was_updated() {
    with_the_engine_on(super::the_policy_leaves_with_the_last_worker_however_often_it_was_updated);
}

#[test]
fn an_update_that_lands_after_the_last_removal_builds_no_policy() {
    with_the_engine_on(super::an_update_that_lands_after_the_last_removal_builds_no_policy);
}

#[test]
fn a_registration_of_a_known_url_is_the_same_worker() {
    with_the_engine_on(super::a_registration_of_a_known_url_is_the_same_worker);
}

#[test]
fn the_policy_stays_while_another_worker_serves_the_model() {
    with_the_engine_on(super::the_policy_stays_while_another_worker_serves_the_model);
}

#[test]
fn a_replacement_on_another_model_takes_the_old_models_policy() {
    with_the_engine_on(super::a_replacement_on_another_model_takes_the_old_models_policy);
}

#[test]
fn the_policy_leaves_with_a_last_worker_that_serves_the_model_among_others() {
    with_the_engine_on(
        super::the_policy_leaves_with_a_last_worker_that_serves_the_model_among_others,
    );
}

#[test]
fn a_replacement_that_lists_a_model_fewer_takes_its_policy() {
    with_the_engine_on(super::a_replacement_that_lists_a_model_fewer_takes_its_policy);
}

#[test]
fn a_model_an_endpoint_stopped_listing_leaves_with_its_last_worker() {
    with_the_engine_on(super::a_model_an_endpoint_stopped_listing_leaves_with_its_last_worker);
}

#[test]
fn a_removal_asks_after_every_model_its_worker_was_registered_with() {
    with_the_engine_on(super::a_removal_asks_after_every_model_its_worker_was_registered_with);
}

#[test]
fn a_replacement_asks_after_a_model_its_worker_stopped_listing() {
    with_the_engine_on(super::a_replacement_asks_after_a_model_its_worker_stopped_listing);
}

#[test]
fn a_replacement_that_renames_the_model_takes_the_old_ids_policy() {
    with_the_engine_on(super::a_replacement_that_renames_the_model_takes_the_old_ids_policy);
}

#[test]
fn a_removal_takes_the_policy_of_the_worker_a_replacement_put_in_its_place() {
    with_the_engine_on(
        super::a_removal_takes_the_policy_of_the_worker_a_replacement_put_in_its_place,
    );
}

#[test]
fn a_mesh_import_removed_here_leaves_the_local_workers_policy() {
    with_the_engine_on(super::a_mesh_import_removed_here_leaves_the_local_workers_policy);
}
