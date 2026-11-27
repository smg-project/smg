use std::sync::Arc;

use smg_worker::{jobs::WorkerJobAsker, test_support::ready_worker};
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    app_context::{testing::app_context_for_tests, AppContext},
    config::{PolicyConfig, RouterConfig},
    routers::common::snapshot::WorkerRegistry,
    workflow::steps::local::create_worker_removal_workflow_data,
};

const MODEL: &str = "llama-3";

/// An app context whose policy for [`MODEL`] is cache-aware with an empty
/// tree: the registry made the policy, nothing routed a worker through it.
fn context_with_cache_aware_model() -> Arc<AppContext> {
    let config = RouterConfig::builder()
        .policy(PolicyConfig::RoundRobin)
        .build_unchecked();
    let app_context = app_context_for_tests(config, Arc::new(WorkerRegistry::new()), None);
    app_context
        .policy_registry
        .on_worker_added(MODEL, Some("cache_aware"));
    app_context
}

/// The worker URLs the model's cache-aware tree routes to.
fn tree(app_context: &AppContext) -> Vec<String> {
    let policy = app_context
        .policy_registry
        .get_policy(MODEL)
        .expect("the model's policy");
    as_cache_aware(policy.as_ref())
        .expect("a cache-aware policy")
        .tree_tenants_for_test(MODEL)
}

/// The step routes the model's remaining workers through its cache-aware
/// tree: with one worker gone from the registry, the tree holds the one
/// that stays and never learns the one that left.
#[tokio::test]
async fn the_step_routes_the_remaining_workers_through_the_cache_aware_tree() {
    let app_context = context_with_cache_aware_model();
    for url in ["http://w1:8000", "http://w2:8000"] {
        app_context
            .worker_registry
            .register(ready_worker(url, MODEL))
            .unwrap();
    }
    app_context
        .worker_registry
        .remove_by_url("http://w2:8000")
        .expect("the leaving worker");
    assert!(
        tree(&app_context).is_empty(),
        "nothing routed before the step"
    );

    let mut data = create_worker_removal_workflow_data(
        "http://w2:8000".to_string(),
        None,
        Arc::clone(&app_context),
        WorkerJobAsker::Api,
    );
    data.worker_urls = vec!["http://w2:8000".to_string()];
    data.affected_models = [MODEL.to_string()].into_iter().collect();
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    UpdateRemainingPoliciesStep
        .execute(&mut context)
        .await
        .unwrap();

    assert_eq!(tree(&app_context), ["http://w1:8000"]);
}
