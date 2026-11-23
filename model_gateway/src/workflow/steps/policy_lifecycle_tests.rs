//! A model's policy across the worker workflows: built from the `policy`
//! label of the model's first worker, kept through registrations, updates
//! and replacements of its workers, and dropped with the last worker this
//! node owns, so the model's next registration builds the policy its own
//! labels name.

use std::sync::Arc;

use openai_protocol::{
    model_card::ModelCard,
    worker::{
        ConnectionMode, HealthCheckConfig, RuntimeType, WorkerSpec, WorkerType, WorkerUpdateRequest,
    },
};
use smg_worker::{
    identity::basic::Worker,
    jobs::WorkerJobAsker,
    kv_events::{SubscriptionKind, MAX_CONCURRENT_SINK_REMOVALS, SINK_REMOVAL_PERMITS},
    workflow::RemoveFromWorkerRegistryStep,
};
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::{
    create_worker_removal_workflow_data, create_worker_update_workflow_data,
    create_worker_workflow_data, RegisterWorkersStep, RemoveFromPolicyRegistryStep,
    UpdatePoliciesForWorkerStep, UpdatePoliciesStep,
};
use crate::{
    app_context::{testing::app_context_for_tests, AppContext},
    config::RouterConfig,
    policies::{kv_index::event_monitor::KvEventMonitor, CacheAwarePolicy},
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
    workflow::data::WorkerRegistrationMode,
};

fn app_context() -> Arc<AppContext> {
    app_context_for_tests(
        RouterConfig::builder()
            .round_robin_policy()
            .build_unchecked(),
        Arc::new(WorkerRegistry::new()),
        None,
    )
}

fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    )
}

/// A worker serving several models, as an external endpoint's worker lists
/// every model its provider serves; the first is its primary model.
fn endpoint(url: &str, models: &[&str]) -> Arc<dyn Worker> {
    let cards: Vec<ModelCard> = models.iter().map(|model| ModelCard::new(*model)).collect();
    Arc::new(BasicWorkerBuilder::new(url).models(cards).build())
}

/// The model's own policy entry, `None` once it is dropped: routing would
/// fall back to the default policy, this does not.
fn policy_of(app_context: &AppContext, model: &str) -> Option<&'static str> {
    app_context
        .policy_registry
        .get_policy(model)
        .map(|policy| policy.name())
}

/// Register `worker` with `policy` as its `policy` label the way the
/// registration workflow does: the registry write under `mode`, then the
/// policies.
async fn register_as(
    app_context: &Arc<AppContext>,
    worker: Arc<dyn Worker>,
    policy: &str,
    mode: WorkerRegistrationMode,
) {
    let mut data = create_worker_workflow_data(
        WorkerSpec::new(worker.url()),
        mode,
        Arc::clone(app_context),
        WorkerJobAsker::Api,
    );
    data.final_labels
        .insert("policy".to_string(), policy.to_string());
    data.actual_workers = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    RegisterWorkersStep.execute(&mut context).await.unwrap();
    UpdatePoliciesStep.execute(&mut context).await.unwrap();
}

/// `POST /workers`, a startup or discovery registration of a known URL.
async fn register(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>, policy: &str) {
    register_as(app_context, worker, policy, WorkerRegistrationMode::Upsert).await;
}

/// A `PATCH` or a label refresh: both end in the update workflow's
/// policies step.
async fn update(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
    let request = WorkerUpdateRequest {
        priority: None,
        cost: None,
        labels: None,
        api_key: None,
        health: None,
    };
    let mut data = create_worker_update_workflow_data(
        worker.url().to_string(),
        request,
        Arc::clone(app_context),
        WorkerJobAsker::Api,
    );
    data.updated_workers = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    UpdatePoliciesForWorkerStep
        .execute(&mut context)
        .await
        .unwrap();
}

/// `DELETE /workers/{id}`: out of the worker registry, then the policies.
async fn remove(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
    let url = worker.url().to_string();
    let mut data = create_worker_removal_workflow_data(
        url.clone(),
        None,
        Arc::clone(app_context),
        WorkerJobAsker::Api,
    );
    data.worker_urls = vec![url];
    data.actual_workers_to_remove = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    RemoveFromWorkerRegistryStep
        .execute(&mut context)
        .await
        .unwrap();
    RemoveFromPolicyRegistryStep
        .execute(&mut context)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_batch_removal_starts_all_its_kv_cleanups_in_one_poll() {
    let monitor = Arc::new(KvEventMonitor::new());
    let mut context_with_monitor = (*app_context()).clone();
    context_with_monitor.kv_event_monitor = Some(Arc::clone(&monitor));
    let app_context = Arc::new(context_with_monitor);
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|rank| {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", rank + 10))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .model(ModelCard::new(format!("model-{rank}")))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    for worker in &workers {
        monitor
            .on_worker_added(worker, SubscriptionKind::Events)
            .await;
    }

    // Hold the monitor's purge permits, so every cleanup the step's one
    // poll starts waits at that bound.
    let permits = SINK_REMOVAL_PERMITS
        .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
        .await
        .expect("the sink-removal semaphore is open");
    let mut data = create_worker_removal_workflow_data(
        workers[0].url().to_string(),
        None,
        Arc::clone(&app_context),
        WorkerJobAsker::Api,
    );
    data.actual_workers_to_remove = Some(workers);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    let remaining = || {
        (0..8)
            .filter(|rank| monitor.get_indexer(&format!("model-{rank}")).is_some())
            .count()
    };
    let removal = RemoveFromPolicyRegistryStep.execute(&mut context);
    tokio::pin!(removal);
    assert!(
        futures::poll!(removal.as_mut()).is_pending(),
        "the cleanups wait at the monitor's bound"
    );
    drop(permits);

    // The step is not polled again: a cleanup it had not started stays.
    for _ in 0..200 {
        if remaining() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(remaining(), 0, "the one poll started every cleanup");
    removal.await.expect("the removal step succeeds");
    monitor.stop().await;
}

#[tokio::test]
async fn the_policy_leaves_with_the_last_worker_however_often_it_was_updated() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    update(&app_context, Arc::clone(&llama)).await;
    update(&app_context, Arc::clone(&llama)).await;

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), None);

    // The model's next first worker names its own policy.
    register(&app_context, worker("http://w1:8000", "llama"), "random").await;
    assert_eq!(policy_of(&app_context, "llama"), Some("random"));
}

#[tokio::test]
async fn an_update_that_lands_after_the_last_removal_builds_no_policy() {
    // A PATCH found llama's only worker, then a DELETE removed it and the
    // policy before the update's policies step ran. That step leaves llama
    // without a policy, so the next registration builds its own label's.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    remove(&app_context, Arc::clone(&llama)).await;

    update(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
    register(&app_context, worker("http://w2:8000", "llama"), "random").await;
    assert_eq!(policy_of(&app_context, "llama"), Some("random"));
}

#[tokio::test]
async fn a_registration_of_a_known_url_is_the_same_worker() {
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama"),
        "cache_aware",
    )
    .await;
    let again = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&again), "cache_aware").await;

    remove(&app_context, again).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn the_policy_stays_while_another_worker_serves_the_model() {
    let app_context = app_context();
    let first = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&first), "cache_aware").await;
    let second = worker("http://w2:8000", "llama");
    register(&app_context, Arc::clone(&second), "cache_aware").await;

    remove(&app_context, first).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
    remove(&app_context, second).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn a_replacement_on_another_model_takes_the_old_models_policy() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;

    // `PUT /workers/{id}` keeps the URL; this body serves another model.
    let replaces = WorkerRegistrationMode::ReplaceById {
        worker_id: app_context
            .worker_registry
            .get_id_by_url(llama.url())
            .expect("the worker is registered")
            .as_str()
            .to_string(),
        expected_revision: llama.revision(),
    };
    register_as(
        &app_context,
        worker("http://w1:8000", "qwen"),
        "random",
        replaces,
    )
    .await;

    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn the_policy_leaves_with_a_last_worker_that_serves_the_model_among_others() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    // Its registration builds qwen's policy; it serves llama besides.
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
    remove(&app_context, both).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn a_replacement_that_lists_a_model_fewer_takes_its_policy() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    register(
        &app_context,
        endpoint("http://w2:8000", &["qwen", "llama"]),
        "random",
    )
    .await;
    remove(&app_context, llama).await;

    // The endpoint registered again, its provider serving qwen alone: no
    // worker of this node serves llama any more.
    register(
        &app_context,
        endpoint("http://w2:8000", &["qwen"]),
        "random",
    )
    .await;
    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn a_model_an_endpoint_stopped_listing_leaves_with_its_last_worker() {
    // A refresh of the endpoint's model list on a routing miss dropped
    // llama, and left the model index as the registration wrote it: llama's
    // own worker is its last.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;
    both.set_models(vec![ModelCard::new("qwen")]);

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn a_removal_asks_after_every_model_its_worker_was_registered_with() {
    // The endpoint outlived llama's own worker, then a refresh dropped
    // llama from its list: its removal still asks after llama.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;
    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));

    both.set_models(vec![ModelCard::new("qwen")]);
    remove(&app_context, both).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn a_removal_purges_its_worker_from_the_trees_of_the_models_it_was_registered_with() {
    // As above, with llama's policy kept by llama's own worker: the URL
    // leaves llama's tree, though the refreshed list no longer names llama.
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama"),
        "cache_aware",
    )
    .await;
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;
    let policy = app_context.policy_registry.get_policy("llama").unwrap();
    let tree = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .and_then(|policy| policy.string_tree("llama"))
        .expect("llama's cache-aware tree");
    tree.insert_text("a prompt the endpoint served", "http://w2:8000");

    both.set_models(vec![ModelCard::new("qwen")]);
    remove(&app_context, both).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
    assert!(!tree.get_tenant_char_count().contains_key("http://w2:8000"));
}

#[tokio::test]
async fn a_replacement_asks_after_a_model_its_worker_stopped_listing() {
    // The endpoint outlived llama's own worker, a refresh dropped llama from
    // its list, then the endpoint registered again listing qwen alone.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;
    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));

    both.set_models(vec![ModelCard::new("qwen")]);
    register(
        &app_context,
        endpoint("http://w2:8000", &["qwen"]),
        "random",
    )
    .await;
    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn a_replacement_that_renames_the_model_takes_the_old_ids_policy() {
    // The replacement serves the model under a new id and keeps the old one
    // as an alias. The routers resolve an alias before any policy lookup, so
    // no request reaches the old id's policy any more.
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama-3"),
        "cache_aware",
    )
    .await;
    let renamed: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8000")
            .model(ModelCard::new("meta-llama/Llama-3").with_alias("llama-3"))
            .build(),
    );
    register(&app_context, renamed, "random").await;
    assert_eq!(policy_of(&app_context, "llama-3"), None);
    assert_eq!(
        policy_of(&app_context, "meta-llama/Llama-3"),
        Some("random")
    );
}

#[tokio::test]
async fn a_removal_takes_the_policy_of_the_worker_a_replacement_put_in_its_place() {
    // The removal found llama's worker, then a registration replaced it at
    // its URL during the drain: the registry step removes the replacement,
    // and qwen's policy goes with qwen's last worker.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    register(&app_context, worker("http://w1:8000", "qwen"), "random").await;

    remove(&app_context, llama).await;
    assert!(app_context
        .worker_registry
        .get_by_url("http://w1:8000")
        .is_none());
    assert_eq!(policy_of(&app_context, "qwen"), None);
}

#[tokio::test]
async fn a_removal_purges_the_cache_tenant_of_the_worker_a_replacement_put_in_its_place() {
    // As above, with qwen's policy kept by qwen's other worker: the URL the
    // registry step removed leaves qwen's tree, not only llama's.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "random").await;
    register(
        &app_context,
        worker("http://w2:8000", "qwen"),
        "cache_aware",
    )
    .await;
    register(
        &app_context,
        worker("http://w1:8000", "qwen"),
        "cache_aware",
    )
    .await;
    let policy = app_context.policy_registry.get_policy("qwen").unwrap();
    let tree = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .and_then(|policy| policy.string_tree("qwen"))
        .expect("qwen's cache-aware tree");
    tree.insert_text("a prompt the replacement served", "http://w1:8000");

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "qwen"), Some("cache_aware"));
    assert!(!tree.get_tenant_char_count().contains_key("http://w1:8000"));
}

#[tokio::test]
async fn a_mesh_import_removed_here_leaves_the_local_workers_policy() {
    // The import never added to the model's workers here, so its removal
    // must not end a policy a local worker still routes by.
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama"),
        "cache_aware",
    )
    .await;
    app_context
        .worker_registry
        .on_remote_worker_state(&smg_worker::registry::RemoteWorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "llama".to_string(),
            url: "http://remote:8080".to_string(),
            health: true,
            version: 1,
            spec: vec![],
        });
    let import = app_context
        .worker_registry
        .get_by_url("http://remote:8080")
        .expect("the import");

    remove(&app_context, import).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
}

#[tokio::test]
async fn a_removal_closes_the_encode_connection_pooled_to_its_worker() {
    use std::time::Duration;

    use smg_grpc_client::tokenspeed_encoder::tokenspeed_encoder_proto::EncodeRequest;
    use tokio::io::AsyncReadExt;

    use crate::routers::grpc::backend_client::send_tokenspeed_encode_rpc;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback listener");
    let url = format!("grpc://{}", listener.local_addr().expect("its address"));
    let app_context = app_context();
    let encoder = worker(&url, "vision");
    register(&app_context, Arc::clone(&encoder), "random").await;
    // The peer answers nothing: the send ends at its bound and leaves the
    // connection it dialled in the pool.
    let send = tokio::time::timeout(
        Duration::from_millis(200),
        send_tokenspeed_encode_rpc(url.clone(), EncodeRequest::default()),
    );
    let (_, dialled) = tokio::join!(send, listener.accept());
    let (mut pooled, _) = dialled.expect("the dialled connection");
    // Open before the removal (past the client preface), so the close below
    // is the removal's.
    let mut preface = Vec::new();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), pooled.read_to_end(&mut preface))
            .await
            .is_err(),
        "the pooled connection is open before the removal"
    );

    remove(&app_context, encoder).await;

    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), pooled.read_to_end(&mut received))
        .await
        .expect("the removal closes the pooled connection")
        .expect("a read to its end");
}
