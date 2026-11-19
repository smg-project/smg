//! Test support of the host: the store a start makes with one ready row a
//! URL, a registry with one worker a URL, and the host over both. A router's
//! cases build their rows this way, so one fixture stands.

use std::sync::Arc;

use kv_index::{chain_prefix_hash, ContentHash, SequenceHash, StoredBlock};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::{
    store::{
        BatchFacts, BatchOutcome, KvEvent, KvEventSink, MemoryStore, Signals, WorkerFacts,
        WorkerStatus as RowStatus, WorkersStore,
    },
    view::{Role, WorkerId},
    AppliedSpec, PolicySpec,
};
use smg_worker::identity::basic::Worker;

use super::Host;
use crate::{
    config::{PolicyConfig, RouterConfig},
    policy_controller::start,
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
};

/// The model every row and worker of [`bed`] serves.
pub(crate) const MODEL: &str = "model-a";

/// A host, its store, and the registry's workers in the URLs' order.
type Bed = (Host, Arc<MemoryStore>, Vec<Arc<dyn Worker>>);

/// The store a start makes under the controller with one ready row a URL,
/// a registry with one worker a URL, and the host over the two.
pub(crate) fn bed(urls: &[&str]) -> Bed {
    bed_under(RouterConfig::default().policy, urls)
}

/// [`bed`] where the gateway routes by `policy`.
pub(crate) fn bed_under(policy: PolicyConfig, urls: &[&str]) -> Bed {
    started(policy, Arc::new(WorkerRegistry::new()), urls)
}

/// The same store and host over workers registered in `registry`.
pub(crate) fn bed_over(registry: Arc<WorkerRegistry>, urls: &[&str]) -> Bed {
    started(RouterConfig::default().policy, registry, urls)
}

/// The store a start makes for `policy`, with its rows and its workers in
/// `registry`.
fn started(policy: PolicyConfig, registry: Arc<WorkerRegistry>, urls: &[&str]) -> Bed {
    let config = RouterConfig {
        policy_controller: true,
        policy,
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the routing")
        .expect("the controller was picked");
    bed_with(store, registry, urls)
}

/// Three ready rows under the cache-aware preset, with the second row's id.
pub(crate) fn cache_bed() -> (Host, Arc<MemoryStore>, WorkerId) {
    let store = Arc::new(MemoryStore::new());
    let document = PolicySpec::preset("cache_aware").expect("the preset");
    let applied = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(applied).expect("the store takes the preset");
    let urls = ["http://w0:8000", "http://w1:8000", "http://w2:8000"];
    let (host, store, _) = bed_with(store, Arc::new(WorkerRegistry::new()), &urls);
    let holder = store.by_key(urls[1]).expect("the second worker's row").id;
    (host, store, holder)
}

/// [`bed`] over `store` and `registry`: a row publishes cache events where
/// the store's generation reads them.
fn bed_with(store: Arc<MemoryStore>, registry: Arc<WorkerRegistry>, urls: &[&str]) -> Bed {
    let events = store.deciding(|world| world.spec().needs().events);
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
            signals: Signals {
                events,
                ..Signals::default()
            },
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

/// Stores a chain of content hashes in one batch on `id`, through the
/// store's own sink.
pub(crate) fn store_blocks(
    store: &MemoryStore,
    id: WorkerId,
    contents: &[ContentHash],
) -> BatchOutcome {
    let mut blocks = Vec::with_capacity(contents.len());
    let mut parent = None;
    for &content_hash in contents {
        let seq_hash = parent.map_or(SequenceHash(content_hash.0), |previous| {
            chain_prefix_hash(previous, content_hash)
        });
        blocks.push(StoredBlock {
            seq_hash,
            content_hash,
        });
        parent = Some(seq_hash);
    }
    let facts = BatchFacts {
        sequence: 1,
        ..BatchFacts::default()
    };
    store.admit(
        id,
        0,
        facts,
        &[KvEvent::Stored {
            parent: None,
            blocks,
        }],
    )
}
