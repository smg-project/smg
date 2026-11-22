//! The test harness builds its contexts with the policy controller on
//! where the test run's switch says so (`SMG_TEST_POLICY_CONTROLLER=on`),
//! and as before where it does not. The pull-request tests run this binary
//! both ways, and this case holds each run to its side: a run whose switch
//! was no longer read would pass every other case, as a plain run does.

use smg::{
    config::RouterConfig,
    policies::{EnginePolicy, RoundRobinPolicy},
};

use crate::common::{
    create_test_context, policy_controller, test_app::create_test_app_context_with_config,
};

#[tokio::test]
async fn a_context_asks_the_policy_engine_exactly_where_the_switch_is_on() {
    let config = RouterConfig::builder()
        .regular_mode(vec![])
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .build_unchecked();
    let on = policy_controller::on();

    for context in [
        create_test_context(config.clone()).await,
        create_test_app_context_with_config(&config).await,
    ] {
        assert_eq!(context.router_config.policy_controller, on);
        // Round robin is a policy the engine picks by itself: with the
        // flag on, the registry hands out the object that asks the engine.
        let policy = context.policy_registry.get_default_policy();
        assert_eq!(policy.as_any().is::<EnginePolicy>(), on, "{policy:?}");
        assert_eq!(policy.as_any().is::<RoundRobinPolicy>(), !on, "{policy:?}");
    }
}
