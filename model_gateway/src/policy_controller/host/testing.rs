//! Test support of the host: the store a start makes with one ready row a
//! URL, a registry with one worker a URL, and the host over both. A router's
//! cases build their rows this way, so one fixture stands.

use std::sync::Arc;

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::{
    store::{MemoryStore, WorkerFacts, WorkerStatus as RowStatus, WorkersStore},
    view::Role,
};
use smg_worker::identity::basic::Worker;

use super::Host;
use crate::{
    config::RouterConfig,
    policy_controller::start,
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
};

/// The model every row and worker of [`bed`] serves.
pub(crate) const MODEL: &str = "model-a";

/// The store a start makes under the controller with one ready row a URL,
/// a registry with one worker a URL, and the host over the two.
pub(crate) fn bed(urls: &[&str]) -> (Host, Arc<MemoryStore>, Vec<Arc<dyn Worker>>) {
    let config = RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the default routing")
        .expect("the controller was picked");
    let registry = Arc::new(WorkerRegistry::new());
    let mut workers = Vec::new();
    for url in urls {
        let facts = WorkerFacts {
            key: (*url).to_string(),
            engine: (*url).to_string(),
            incarnation: 1,
            role: Role::Single,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            max_running: Some(64),
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        store.publish_now();
        let row = WorkersStore::get(&*store, id).expect("the row");
        if row.status != RowStatus::Ready {
            store
                .set_status(id, RowStatus::Ready, row.revision)
                .expect("the row stands");
        }
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(*url)
                .model(ModelCard::new(MODEL))
                .status(WorkerStatus::Ready)
                .build(),
        );
        registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        workers.push(worker);
    }
    store.publish_now();
    (Host::new(Arc::clone(&store), registry), store, workers)
}
