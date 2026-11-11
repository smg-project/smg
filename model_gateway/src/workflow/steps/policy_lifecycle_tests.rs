//! A model's policy across the worker workflows: built from the `policy`
//! label of the model's first worker, kept through registrations, updates
//! and replacements of its workers, and dropped with the last worker this
//! node owns, so the model's next registration builds the policy its own
//! labels name.

use std::sync::Arc;

use openai_protocol::worker::{WorkerSpec, WorkerStatus, WorkerUpdateRequest};
use smg_policy_controller::Side;
use tokio::sync::watch;
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::{
    create_worker_removal_workflow_data, create_worker_update_workflow_data,
    create_worker_workflow_data, RegisterWorkersStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdatePoliciesForWorkerStep, UpdatePoliciesStep,
    UpdateWorkerPropertiesStep,
};
use crate::{
    app_context::{testing::app_context_for_tests, AppContext},
    config::{PolicyConfig, PolicyEngineMode, RouterConfig},
    policies::{
        engine_host::{Decider, HostRequest},
        CacheAwarePolicy,
    },
    worker::{
        load_state::{one_rank_load, LoadSnapshot},
        BasicWorkerBuilder, ModelCard, Worker, WorkerRegistry,
    },
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
    let mut data =
        create_worker_workflow_data(WorkerSpec::new(worker.url()), mode, Arc::clone(app_context));
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
    let mut data = create_worker_removal_workflow_data(url.clone(), None, Arc::clone(app_context));
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

/// Under `--policy-engine engine` a removal prunes the engine's rows of
/// every handle the registry removed: the worker found and a same-URL
/// replacement that landed during the drain, each a worker id of its own.
#[tokio::test]
async fn a_removal_prunes_the_engines_rows_of_the_worker_a_replacement_put_in_its_place() {
    let app_context = app_context_for_tests(
        RouterConfig::builder()
            .policy(PolicyConfig::LeastLoad {
                kv_pressure_weight: 0.0,
                mean_prefill_tokens: 100,
                default_throughput: 1000.0,
            })
            .policy_engine(PolicyEngineMode::Engine)
            .build_unchecked(),
        Arc::new(WorkerRegistry::new()),
        None,
    );
    // A registration of the same URL for another model replaces the handle
    // (a registration for the same model is the same worker).
    let found = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&found), "least_load").await;
    register(&app_context, worker("http://w1:8000", "qwen"), "least_load").await;
    register(
        &app_context,
        worker("http://w2:8000", "llama"),
        "least_load",
    )
    .await;
    // The handles the registry holds are the ones the engine's ids name,
    // in the view and in the removal step alike.
    let handle = |url: &str| {
        app_context
            .worker_registry
            .get_by_url(url)
            .expect("registered")
    };
    let replacement = handle("http://w1:8000");
    let other = handle("http://w2:8000");
    assert!(
        !Arc::ptr_eq(&found, &replacement),
        "the second registration replaced the handle"
    );
    for worker in [&found, &replacement, &other] {
        worker.set_status(WorkerStatus::Ready);
    }

    // Both URLs idle in the load feed, so only the engine's own bookings
    // tell the two handles apart.
    let decider = Arc::clone(app_context.policy_registry.decider().expect("the engine"));
    let (_reports, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
        ("http://w1:8000".to_string(), one_rank_load(0, 0.1, 2000.0)),
        ("http://w2:8000".to_string(), one_rank_load(0, 0.1, 2000.0)),
    ]));
    decider.set_load_receiver(Some(rx));
    let view: Vec<Arc<dyn Worker>> = vec![Arc::clone(&replacement), Arc::clone(&other)];
    let request = HostRequest {
        model: "llama",
        prompt_tokens: decider.prompt_tokens(None),
        ..HostRequest::default()
    };
    let book = |position: u32| {
        let (decision, lease) = decider.decide(&request, &view, &[]);
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == position)
            .expect("the wanted worker is booked");
    };
    let leader = || {
        let (decision, _) = decider.decide(&request, &view, &[]);
        decision
            .pick()
            .expect("a pick")
            .targets
            .iter()
            .next()
            .map(|t| t.position)
    };
    book(0);
    book(0);
    book(1);
    assert_eq!(
        leader(),
        Some(1),
        "the replacement carries more booked work"
    );

    // The removal found the first handle; the registry step removes the
    // replacement standing at the URL, and its rows go with it.
    remove(&app_context, found).await;
    assert!(app_context
        .worker_registry
        .get_by_url("http://w1:8000")
        .is_none());
    // The registry demotes a handle it removes; the probe wants it back in
    // the view to read what the engine still holds against it.
    replacement.set_status(WorkerStatus::Ready);
    assert_eq!(
        leader(),
        Some(0),
        "the replacement's rows went with it, so it reads idle again"
    );
}

/// Two ready workers of one model under the engine, the found one booked
/// twice and the other once: both URLs idle in the load feed, so only the
/// engine's own rows tell the handles apart, and the other leads. The
/// engine keys its rows by handle, so a handle replaced at its URL with no
/// removal (a registration of the URL for another model, a label refresh
/// that rebuilds the worker) must tell the engine as a removal would, or
/// the retired handle's rows stay for good under churn.
struct BookedPair {
    app_context: Arc<AppContext>,
    found: Arc<dyn Worker>,
    decider: Arc<Decider>,
    view: Vec<Arc<dyn Worker>>,
    _reports: watch::Sender<Arc<LoadSnapshot>>,
}

impl BookedPair {
    async fn new() -> Self {
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .policy(PolicyConfig::LeastLoad {
                    kv_pressure_weight: 0.0,
                    mean_prefill_tokens: 100,
                    default_throughput: 1000.0,
                })
                .policy_engine(PolicyEngineMode::Engine)
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        let found = worker("http://w1:8000", "llama");
        register(&app_context, Arc::clone(&found), "least_load").await;
        register(
            &app_context,
            worker("http://w2:8000", "llama"),
            "least_load",
        )
        .await;
        let other = app_context
            .worker_registry
            .get_by_url("http://w2:8000")
            .expect("registered");
        for worker in [&found, &other] {
            worker.set_status(WorkerStatus::Ready);
        }
        let decider = Arc::clone(app_context.policy_registry.decider().expect("the engine"));
        let (reports, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            ("http://w1:8000".to_string(), one_rank_load(0, 0.1, 2000.0)),
            ("http://w2:8000".to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        decider.set_load_receiver(Some(rx));
        let pair = Self {
            app_context,
            view: vec![Arc::clone(&found), other],
            found,
            decider,
            _reports: reports,
        };
        pair.book(0);
        pair.book(0);
        pair.book(1);
        assert_eq!(
            pair.leader(),
            Some(1),
            "the found handle carries more booked work"
        );
        pair
    }

    fn request(&self) -> HostRequest<'static> {
        HostRequest {
            model: "llama",
            prompt_tokens: self.decider.prompt_tokens(None),
            ..HostRequest::default()
        }
    }

    /// Books one request on the worker at `position` of the view.
    fn book(&self, position: u32) {
        let (decision, lease) = self.decider.decide(&self.request(), &self.view, &[]);
        self.decider
            .commit(&lease, &decision, Side::Single, |t| t.position == position)
            .expect("the wanted worker is booked");
    }

    /// The position the engine would pick next, read through the same
    /// view: what it still holds against the found handle shows here.
    fn leader(&self) -> Option<u32> {
        let (decision, _) = self.decider.decide(&self.request(), &self.view, &[]);
        decision
            .pick()
            .expect("a pick")
            .targets
            .iter()
            .next()
            .map(|t| t.position)
    }

    /// Proves another handle now stands at the found worker's URL, and
    /// readies the retired one for the probe through the view it sits in.
    fn replaced(&self) {
        let replacement = self
            .app_context
            .worker_registry
            .get_by_url("http://w1:8000")
            .expect("registered");
        assert!(
            !Arc::ptr_eq(&self.found, &replacement),
            "the handle was replaced"
        );
        self.found.set_status(WorkerStatus::Ready);
    }
}

/// A registration of a URL the registry holds for another model replaces
/// the handle with no removal; the retired handle's rows go with it.
#[tokio::test]
async fn a_replacement_without_a_removal_prunes_the_engines_rows_of_the_retired_handle() {
    let pair = BookedPair::new().await;
    register(
        &pair.app_context,
        worker("http://w1:8000", "qwen"),
        "least_load",
    )
    .await;
    pair.replaced();
    assert_eq!(
        pair.leader(),
        Some(0),
        "the retired handle's rows went with it, so it reads idle again"
    );
}

/// A label refresh that changes the record rebuilds the worker at its URL,
/// the property-update step's replacement with no removal; the retired
/// handle's rows go with it.
#[tokio::test]
async fn a_rebuild_on_a_label_refresh_prunes_the_engines_rows_of_the_retired_handle() {
    let pair = BookedPair::new().await;
    let mut data = create_worker_update_workflow_data(
        pair.found.url().to_string(),
        WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: None,
            api_key: None,
            health: None,
        },
        Arc::clone(&pair.app_context),
    );
    let id = pair
        .app_context
        .worker_registry
        .get_id_by_url(pair.found.url())
        .expect("registered");
    data.snapshot = Some(vec![(id, pair.found.revision())]);
    data.workers_to_update = Some(vec![Arc::clone(&pair.found)]);
    data.refresh_discovered_labels = true;
    // A key the record never had: the refresh rebuilds the worker.
    data.fresh_engine_labels = Some(
        [("page_size".to_string(), "128".to_string())]
            .into_iter()
            .collect(),
    );
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    UpdateWorkerPropertiesStep
        .execute(&mut context)
        .await
        .expect("the refresh goes through");
    pair.replaced();
    assert_eq!(
        pair.leader(),
        Some(0),
        "the retired handle's rows went with it, so it reads idle again"
    );
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
        .on_remote_worker_state(&smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "llama".to_string(),
            url: "http://remote:8080".to_string(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
            owner: "peer".to_string(),
        });
    let import = app_context
        .worker_registry
        .get_by_url("http://remote:8080")
        .expect("the import");

    remove(&app_context, import).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
}
