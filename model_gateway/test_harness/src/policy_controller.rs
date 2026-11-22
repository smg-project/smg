//! The switch that turns the policy controller on in every context this
//! harness builds.
//!
//! A test builds its context from a `RouterConfig` and does not name
//! `--policy-controller`. With `SMG_TEST_POLICY_CONTROLLER=on` in a test
//! run's environment, each context is built as a gateway started with that
//! flag and no controller's address builds its own: the configuration
//! carries the flag, and the policy registry, the policy engine and the
//! engine object are the ones the gateway's start builds under it. The
//! run's requests are then placed as that gateway places them. Unset, or
//! set to anything else, every context is built as before.
//!
//! The pull-request tests run the gateway's integration tests a second
//! time with the switch on (`scripts/ci_rust_tests.sh policy-controller`).

use std::{env, sync::Arc};

use smg::{
    app_context::{AppContext, AppContextBuilder},
    config::RouterConfig,
    policies::PolicyRegistry,
    routers::common::snapshot::WorkerRegistry,
};

/// The environment variable of the switch.
pub const SWITCH: &str = "SMG_TEST_POLICY_CONTROLLER";

/// Whether this test run turns the policy controller on.
pub fn on() -> bool {
    env::var_os(SWITCH).is_some_and(|value| value == "on")
}

/// The policy parts of a context: the builder it is built on, its policy
/// registry and the configuration it keeps.
///
/// With the switch off they are a new builder, the registry `classic`
/// builds from `config`, and `config` as given. With it on they are the
/// gateway's own: `config` with `policy_controller` set and no controller's
/// address, and over `workers` what the start's two policy steps build
/// under it ([`AppContextBuilder::with_policies_started`]). A configuration
/// the start refuses under the flag fails the case here, in the start's
/// words.
#[expect(
    clippy::panic,
    reason = "test helper - panicking on failure is intentional"
)]
pub(crate) fn parts(
    config: RouterConfig,
    workers: &Arc<WorkerRegistry>,
    classic: impl FnOnce(&RouterConfig) -> PolicyRegistry,
) -> (AppContextBuilder, Arc<PolicyRegistry>, RouterConfig) {
    if !on() {
        let policies = Arc::new(classic(&config));
        return (AppContext::builder(), policies, config);
    }
    let config = RouterConfig {
        policy_controller: true,
        policy_controller_address: None,
        ..config
    };
    match AppContextBuilder::with_policies_started(&config, Arc::clone(workers)) {
        Ok((builder, policies)) => (builder, policies, config),
        Err(refusal) => {
            panic!("with {SWITCH}=on the gateway's start refuses this configuration: {refusal}")
        }
    }
}
