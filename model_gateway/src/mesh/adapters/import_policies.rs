//! Mesh imports admitted to the policies and released from them.
//!
//! The worker sync adapter writes a peer's workers into the registry outside
//! the registration workflow (`WorkerRegistry::on_remote_worker_state`) and
//! removes them outside the removal workflow (`remove_remote`), so the policy
//! side of both ([`PolicyRegistration`]) never ran for an import: a model
//! served by imports alone had no policy of its own and lost the one it had
//! when this node's last local worker of the model left; an import reported
//! no completions and streamed nothing into the KV index; a tombstoned import
//! left its dispatch tally and its tree tenants behind. This task follows the
//! registry's events the way the worker manager does and runs the two steps
//! for every worker of origin `Mesh`: `Registered` admits it, `Removed`
//! releases it, and `Replaced` tells an owner's in-place rebuild (still an
//! import: released as the old worker, admitted as the new one) from a local
//! claim of its address (the worker is local now and the registration
//! workflow admits it under the same URL: the URL's stream is restarted for
//! the local worker, the import's share of its model's count is released,
//! the dispatch tallies and the tree tenants of the URL stay). A lagged
//! channel resyncs the set it tracks against the registry.

use std::{collections::HashMap, sync::Arc};

use tokio::{sync::broadcast, task::JoinHandle};
use tracing::{debug, warn};

use crate::{
    policies::PolicyRegistration,
    worker::{event::WorkerEvent, registry::WorkerId, Worker, WorkerOrigin, WorkerRegistry},
};

/// The task that admits mesh imports to the policies and releases them.
#[derive(Debug)]
pub struct ImportPolicySync {
    task: JoinHandle<()>,
}

impl ImportPolicySync {
    /// Start following the registry's events. The subscription is taken
    /// before the registry is read, as the worker manager does: an import
    /// registered before this call is admitted from the snapshot, one
    /// registered after it arrives as an event, and an admission is
    /// idempotent per worker id.
    pub fn start(registration: PolicyRegistration) -> Self {
        let registry = Arc::clone(&registration.worker_registry);
        let events = registry.subscribe_events();
        let mut sync = Sync {
            registry,
            registration,
            tracked: HashMap::new(),
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "the task runs for the mesh adapters' lifetime, which is the process lifetime; the handle is kept so a test can stop it"
        )]
        let task = tokio::spawn(async move { sync.run(events).await });
        Self { task }
    }

    /// End the task; the imports it admitted stay admitted.
    pub fn stop(&self) {
        self.task.abort();
    }
}

struct Sync {
    registry: Arc<WorkerRegistry>,
    registration: PolicyRegistration,
    /// The imports admitted, by id, with the worker admitted: what a release
    /// acts on once the registry no longer holds the worker.
    tracked: HashMap<WorkerId, Arc<dyn Worker>>,
}

impl Sync {
    async fn run(&mut self, mut events: broadcast::Receiver<WorkerEvent>) {
        self.resync().await;
        loop {
            match events.recv().await {
                Ok(event) => self.handle(event).await,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        lagged = n,
                        "The import policy sync lagged behind the registry's events; resyncing from the registry"
                    );
                    self.resync().await;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    debug!("WorkerEvent channel closed; the import policy sync exits");
                    return;
                }
            }
        }
    }

    /// One registry event. A `Replaced` of a tracked import is read by what
    /// the registry holds under the id now: still an import, an owner's
    /// in-place rebuild; a local worker, a claim of the import's address;
    /// nothing, an import gone before this task saw its rebuild.
    async fn handle(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Registered { worker_id, worker } => {
                if self.is_import(&worker_id) {
                    self.admit(worker_id, worker).await;
                }
            }
            WorkerEvent::Replaced { worker_id, new, .. } => {
                if !self.tracked.contains_key(&worker_id) {
                    return;
                }
                match self.registry.origin_of(&worker_id) {
                    Some(WorkerOrigin::Mesh) => {
                        self.release(&worker_id).await;
                        self.admit(worker_id, new).await;
                    }
                    Some(WorkerOrigin::Local) => self.release_claimed(&worker_id).await,
                    None => self.release(&worker_id).await,
                }
            }
            WorkerEvent::Removed { worker_id, .. } => self.release(&worker_id).await,
            WorkerEvent::StatusChanged { .. } => {}
        }
    }

    fn is_import(&self, worker_id: &WorkerId) -> bool {
        self.registry.origin_of(worker_id) == Some(WorkerOrigin::Mesh)
    }

    /// Admit the import, with the `policy` label its spec carries as the
    /// hint a local registration's label gives.
    async fn admit(&mut self, worker_id: WorkerId, worker: Arc<dyn Worker>) {
        if self.tracked.contains_key(&worker_id) {
            return;
        }
        let hint = worker.metadata().spec.labels.get("policy").cloned();
        self.registration.admit(&worker, hint.as_deref()).await;
        debug!(
            worker_id = %worker_id.as_str(),
            url = worker.redacted_url(),
            model = worker.model_id(),
            "Mesh import admitted to the policies"
        );
        self.tracked.insert(worker_id, worker);
    }

    /// Release the worker admitted under `worker_id`: what was admitted is
    /// what a release undoes, whatever the registry holds now.
    async fn release(&mut self, worker_id: &WorkerId) {
        let Some(worker) = self.tracked.remove(worker_id) else {
            return;
        };
        self.registration.release(&worker).await;
        debug!(
            worker_id = %worker_id.as_str(),
            url = worker.redacted_url(),
            model = worker.model_id(),
            "Mesh import released from the policies"
        );
    }

    /// A local registration claimed the import's address
    /// (`WorkerRegistry::register_or_replace`): the worker is local now, and
    /// the registration workflow admits it under the same URL in the step
    /// after the claim. The URL's stream was started for the import and
    /// holds its worker object, so it is restarted for the local worker
    /// (`PolicyRegistration::restart_stream`, in either order with the
    /// workflow's admission). The dispatch tallies and the tree tenants of
    /// the URL are the same engine's and stay. The import's share of its
    /// model's count is released last, so the model's policy still decides
    /// the restarted stream's kind. A release keyed by the URL, run after
    /// the workflow's admission, would stop the local worker's stream for
    /// good.
    async fn release_claimed(&mut self, worker_id: &WorkerId) {
        let Some(worker) = self.tracked.remove(worker_id) else {
            return;
        };
        if let Some(local) = self.registry.get(worker_id) {
            self.registration.restart_stream(&local).await;
        }
        self.registration
            .policy_registry
            .on_worker_removed(worker.model_id());
        debug!(
            worker_id = %worker_id.as_str(),
            url = worker.redacted_url(),
            model = worker.model_id(),
            "Mesh import claimed by a local registration; the URL's stream restarted for the local worker, the import's share of the model's count released"
        );
    }

    /// Bring the tracked set to the registry's imports: a tracked worker the
    /// registry no longer holds is released, one it holds as a local worker
    /// was claimed, one it holds as an import under another worker object
    /// was rebuilt (released as the old worker, admitted as the new), and an
    /// import this task has not seen is admitted.
    async fn resync(&mut self) {
        let imports: HashMap<WorkerId, Arc<dyn Worker>> = self
            .registry
            .get_all_with_ids()
            .into_iter()
            .filter(|(worker_id, _)| self.is_import(worker_id))
            .collect();
        let tracked: Vec<(WorkerId, Arc<dyn Worker>)> = self
            .tracked
            .iter()
            .map(|(worker_id, worker)| (worker_id.clone(), Arc::clone(worker)))
            .collect();
        for (worker_id, worker) in tracked {
            match imports.get(&worker_id) {
                Some(current) if Arc::ptr_eq(current, &worker) => {}
                Some(_) => self.release(&worker_id).await,
                None if self.registry.origin_of(&worker_id) == Some(WorkerOrigin::Local) => {
                    self.release_claimed(&worker_id).await;
                }
                None => self.release(&worker_id).await,
            }
        }
        for (worker_id, worker) in imports {
            self.admit(worker_id, worker).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use openai_protocol::{
        model_card::ModelCard,
        worker::{ConnectionMode, WorkerModels, WorkerSpec},
    };
    use smg_mesh::WorkerState;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyRegistry,
        worker::{test_support::no_health_check, BasicWorkerBuilder, KvEventMonitor},
    };

    fn registration(
        worker_registry: &Arc<WorkerRegistry>,
        policy_registry: &Arc<PolicyRegistry>,
    ) -> PolicyRegistration {
        PolicyRegistration {
            policy_registry: Arc::clone(policy_registry),
            worker_registry: Arc::clone(worker_registry),
            kv_event_monitor: None,
            worker_monitor: None,
        }
    }

    /// A cache-aware policy with a KV event monitor: every gRPC worker
    /// admitted streams its KV events under its URL, the state a claim must
    /// leave to the local worker.
    fn streaming_registration(
        worker_registry: &Arc<WorkerRegistry>,
    ) -> (PolicyRegistration, Arc<PolicyRegistry>, Arc<KvEventMonitor>) {
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::cache_aware(
            0.5, 32, 1.1, 0, 4096,
        )));
        let monitor = Arc::new(KvEventMonitor::new());
        policy_registry.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        let registration = PolicyRegistration {
            kv_event_monitor: Some(Arc::clone(&monitor)),
            ..registration(worker_registry, &policy_registry)
        };
        (registration, policy_registry, monitor)
    }

    fn remote_state(worker_id: &str, url: &str, model_id: &str) -> WorkerState {
        WorkerState {
            worker_id: worker_id.into(),
            model_id: model_id.into(),
            url: url.into(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
            owner: "peer".into(),
        }
    }

    /// A peer's gRPC worker: the spec names the transport, so the import
    /// streams into the monitor as the peer's own registration does.
    fn grpc_remote_state(worker_id: &str, url: &str, model_id: &str) -> WorkerState {
        let mut spec = WorkerSpec::new(url);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new(model_id)));
        spec.connection_mode = ConnectionMode::Grpc;
        WorkerState {
            spec: serde_json::to_vec(&spec).expect("a worker spec serializes"),
            ..remote_state(worker_id, url, model_id)
        }
    }

    fn local_worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model_id))
                .health_config(no_health_check())
                .build(),
        )
    }

    /// A local gRPC worker at a port that refuses connections: its stream
    /// task loops on reconnects and answers a shutdown, no engine involved.
    fn grpc_local_worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model_id))
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        )
    }

    /// The sync driven by hand, so a test orders its steps against the
    /// registry's events itself.
    fn sync_by_hand(registry: &Arc<WorkerRegistry>, registration: &PolicyRegistration) -> Sync {
        Sync {
            registry: Arc::clone(registry),
            registration: registration.clone(),
            tracked: HashMap::new(),
        }
    }

    /// Every event the registry has queued, in order.
    async fn handle_queued(sync: &mut Sync, events: &mut broadcast::Receiver<WorkerEvent>) {
        while let Ok(event) = events.try_recv() {
            sync.handle(event).await;
        }
    }

    /// The sync runs on the test's runtime: it acts between the test's
    /// awaits, so an expectation is polled.
    async fn until(what: &str, mut holds: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !holds() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("not within 5 s: {what}"));
    }

    fn count(policies: &PolicyRegistry, model_id: &str) -> Option<usize> {
        policies.get_worker_counts().get(model_id).copied()
    }

    #[tokio::test]
    async fn an_import_is_admitted_to_the_policies_and_released_by_its_tombstone() {
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let sync = ImportPolicySync::start(registration(&registry, &policies));

        registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", "m1"));
        let id = registry.get_id_by_url("http://remote:8080").unwrap();
        until("the import's model has a policy and a count", || {
            count(&policies, "m1") == Some(1) && policies.get_policy("m1").is_some()
        })
        .await;
        assert!(
            registry.get(&id).unwrap().completion_sink().is_some(),
            "the import reports its completions"
        );

        registry.remove_remote(&id);
        until("the tombstone released the import", || {
            count(&policies, "m1").is_none() && policies.get_policy("m1").is_none()
        })
        .await;
        sync.stop();
    }

    #[tokio::test]
    async fn an_import_registered_before_the_sync_started_is_admitted_from_the_snapshot() {
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", "m1"));

        let sync = ImportPolicySync::start(registration(&registry, &policies));
        until("the earlier import is admitted", || {
            count(&policies, "m1") == Some(1)
        })
        .await;
        sync.stop();
    }

    #[tokio::test]
    async fn a_claim_hands_the_worker_to_the_local_registration() {
        // The local registration that claims an import's address runs the
        // registration workflow, which admits the worker as local: the
        // import's share of the count is released, and nothing else here.
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let sync = ImportPolicySync::start(registration(&registry, &policies));
        registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", "m1"));
        until("the import is admitted", || {
            count(&policies, "m1") == Some(1)
        })
        .await;

        let id = registry.register_or_replace(local_worker("http://remote:8080", "m1"));
        assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Local));
        until("the import's share is released", || {
            count(&policies, "m1").is_none()
        })
        .await;
        sync.stop();
    }

    /// The registration workflow claims an import's address in its register
    /// step and admits the local worker in the step after; this task sees
    /// the `Replaced` event on its own schedule, so its release can run
    /// after that admission. The model's index, fed by the stream under the
    /// URL the two workers share, and the model's count both hold.
    #[tokio::test]
    async fn a_claim_leaves_the_local_workers_stream_and_count_to_the_workflow() {
        let registry = Arc::new(WorkerRegistry::new());
        let (registration, policies, monitor) = streaming_registration(&registry);
        let mut sync = sync_by_hand(&registry, &registration);
        let mut events = registry.subscribe_events();

        registry.on_remote_worker_state(&grpc_remote_state("peer-w1", "grpc://127.0.0.1:1", "m1"));
        handle_queued(&mut sync, &mut events).await;
        assert_eq!(count(&policies, "m1"), Some(1));
        assert!(
            monitor.get_indexer("m1").is_some(),
            "the import's stream opened the model's index"
        );

        // The workflow's register step claims the address, its policy step
        // admits the local worker, then this task sees the event.
        let local = grpc_local_worker("grpc://127.0.0.1:1", "m1");
        let id = registry.register_or_replace(Arc::clone(&local));
        assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Local));
        registration.admit(&local, None).await;
        handle_queued(&mut sync, &mut events).await;

        assert_eq!(
            count(&policies, "m1"),
            Some(1),
            "the local worker's share alone"
        );
        assert!(policies.get_policy("m1").is_some());
        assert!(
            monitor.get_indexer("m1").is_some(),
            "the local worker's stream holds"
        );
        assert!(sync.tracked.is_empty(), "the worker is the workflow's now");
    }

    /// The stream under the URL was started for the import and holds its
    /// worker object: the pushed loads, the overload verdict and the liveness
    /// it carries would go to the object the registry dropped at the claim,
    /// the load state drops such a report, and the records' receipts
    /// suppress the poll. The claim restarts the stream for the local
    /// worker, here after the workflow's admission found the import's
    /// stream active and kept it.
    #[tokio::test]
    async fn a_claim_restarts_the_urls_stream_for_the_local_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let (registration, policies, monitor) = streaming_registration(&registry);
        let mut sync = sync_by_hand(&registry, &registration);
        let mut events = registry.subscribe_events();

        registry.on_remote_worker_state(&grpc_remote_state("peer-w1", "grpc://127.0.0.1:1", "m1"));
        handle_queued(&mut sync, &mut events).await;
        let id = registry.get_id_by_url("grpc://127.0.0.1:1").unwrap();
        let import = Arc::downgrade(&registry.get(&id).unwrap());

        let local = grpc_local_worker("grpc://127.0.0.1:1", "m1");
        registry.register_or_replace(Arc::clone(&local));
        registration.admit(&local, None).await;
        handle_queued(&mut sync, &mut events).await;

        until(
            "the import's worker object is released by its stream",
            || import.upgrade().is_none(),
        )
        .await;
        assert_eq!(count(&policies, "m1"), Some(1));
        assert!(
            monitor.get_indexer("m1").is_some(),
            "the restarted stream feeds the model's index"
        );
        assert!(sync.tracked.is_empty());
        // The restarted stream's task holds the local worker's object:
        // ending the stream releases exactly that hold.
        let held = Arc::strong_count(&local);
        monitor.on_worker_removed("grpc://127.0.0.1:1").await;
        assert_eq!(
            Arc::strong_count(&local),
            held - 1,
            "the restarted stream's task held the local worker's object"
        );
    }

    /// A claim this task did not see (a lagged channel) is found at the
    /// resync, and the stream under the URL is restarted the same way.
    #[tokio::test]
    async fn a_resync_after_a_missed_claim_restarts_the_urls_stream_for_the_local_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let (registration, policies, monitor) = streaming_registration(&registry);
        let mut sync = sync_by_hand(&registry, &registration);

        registry.on_remote_worker_state(&grpc_remote_state("peer-w1", "grpc://127.0.0.1:1", "m1"));
        sync.resync().await;
        let id = registry.get_id_by_url("grpc://127.0.0.1:1").unwrap();
        let import = Arc::downgrade(&registry.get(&id).unwrap());

        let local = grpc_local_worker("grpc://127.0.0.1:1", "m1");
        registry.register_or_replace(Arc::clone(&local));
        registration.admit(&local, None).await;
        sync.resync().await;

        until(
            "the import's worker object is released by its stream",
            || import.upgrade().is_none(),
        )
        .await;
        assert_eq!(count(&policies, "m1"), Some(1));
        assert!(monitor.get_indexer("m1").is_some());
        assert!(sync.tracked.is_empty());
        let held = Arc::strong_count(&local);
        monitor.on_worker_removed("grpc://127.0.0.1:1").await;
        assert_eq!(Arc::strong_count(&local), held - 1);
    }

    /// A rebuild of an import that is gone by the time this task sees the
    /// event is a release, not a claim: the tombstone that follows finds
    /// nothing tracked, so the stream must stop here.
    #[tokio::test]
    async fn a_rebuild_of_an_import_gone_before_the_task_saw_it_is_a_release() {
        let registry = Arc::new(WorkerRegistry::new());
        let (registration, policies, monitor) = streaming_registration(&registry);
        let mut sync = sync_by_hand(&registry, &registration);
        let mut events = registry.subscribe_events();

        registry.on_remote_worker_state(&grpc_remote_state("peer-w1", "grpc://127.0.0.1:1", "m1"));
        handle_queued(&mut sync, &mut events).await;
        assert!(monitor.get_indexer("m1").is_some());

        let id = registry.get_id_by_url("grpc://127.0.0.1:1").unwrap();
        assert!(registry.replace(&id, grpc_local_worker("grpc://127.0.0.1:1", "m2")));
        assert!(registry.remove_remote(&id).is_some());
        handle_queued(&mut sync, &mut events).await;

        assert_eq!(count(&policies, "m1"), None);
        assert_eq!(count(&policies, "m2"), None);
        assert!(
            monitor.get_indexer("m1").is_none(),
            "the import's stream stopped with its release"
        );
        assert!(sync.tracked.is_empty());
    }

    /// A claim this task did not see (a lagged channel) is told from a
    /// removal at the resync: the registry still holds the id, as a local
    /// worker.
    #[tokio::test]
    async fn a_resync_after_a_missed_claim_releases_the_count_alone() {
        let registry = Arc::new(WorkerRegistry::new());
        let (registration, policies, monitor) = streaming_registration(&registry);
        let mut sync = sync_by_hand(&registry, &registration);

        registry.on_remote_worker_state(&grpc_remote_state("peer-w1", "grpc://127.0.0.1:1", "m1"));
        sync.resync().await;
        assert_eq!(count(&policies, "m1"), Some(1));

        let local = grpc_local_worker("grpc://127.0.0.1:1", "m1");
        registry.register_or_replace(Arc::clone(&local));
        registration.admit(&local, None).await;
        sync.resync().await;

        assert_eq!(count(&policies, "m1"), Some(1));
        assert!(
            monitor.get_indexer("m1").is_some(),
            "the local worker's stream holds"
        );
        assert!(sync.tracked.is_empty());
    }

    /// A rebuild this task did not see (a lagged channel) is found at the
    /// resync by the worker object: the old one released, the new admitted.
    #[tokio::test]
    async fn a_resync_after_a_missed_rebuild_moves_the_import_to_its_new_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let registration = registration(&registry, &policies);
        let mut sync = sync_by_hand(&registry, &registration);

        registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", "m1"));
        sync.resync().await;
        assert_eq!(count(&policies, "m1"), Some(1));

        let id = registry.get_id_by_url("http://remote:8080").unwrap();
        let rebuilt = local_worker("http://remote:8080", "m2");
        assert!(registry.replace(&id, Arc::clone(&rebuilt)));
        assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Mesh));
        sync.resync().await;

        assert_eq!(count(&policies, "m1"), None);
        assert_eq!(count(&policies, "m2"), Some(1));
        assert!(
            sync.tracked
                .get(&id)
                .is_some_and(|tracked| Arc::ptr_eq(tracked, &rebuilt)),
            "the new worker is the one tracked"
        );
    }

    #[tokio::test]
    async fn an_in_place_rebuild_moves_the_import_to_its_new_model() {
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let sync = ImportPolicySync::start(registration(&registry, &policies));
        registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", "m1"));
        let id = registry.get_id_by_url("http://remote:8080").unwrap();
        until("the import is admitted", || {
            count(&policies, "m1") == Some(1)
        })
        .await;

        assert!(registry.replace(&id, local_worker("http://remote:8080", "m2")));
        assert_eq!(
            registry.origin_of(&id),
            Some(WorkerOrigin::Mesh),
            "a plain replacement keeps the import an import"
        );
        until("the count moved to the new model", || {
            count(&policies, "m1").is_none() && count(&policies, "m2") == Some(1)
        })
        .await;
        sync.stop();
    }

    #[tokio::test]
    async fn a_local_registration_is_left_to_the_workflow() {
        let registry = Arc::new(WorkerRegistry::new());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let sync = ImportPolicySync::start(registration(&registry, &policies));
        registry.register(local_worker("http://local:8080", "m1"));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            policies.get_worker_counts().is_empty(),
            "the sync admits imports only; the workflow admits local workers"
        );
        sync.stop();
    }
}
