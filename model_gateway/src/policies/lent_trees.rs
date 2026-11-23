//! The token trees a gateway lends the policy engine.
//!
//! Where the policy engine places every request no classic policy object is
//! made, so no policy keeps the trees of the prompts this process sent. The
//! engine's start makes an owner of them in a policy's place
//! ([`LentTrees`]), and only where the document in force reads the prefixes
//! sent to each worker. The owner keeps the trees in the store type the
//! classic policy keeps its own in, under the document's own numbers: its
//! block size is a tree's page, and a model's `index` entry states the size
//! its tree is held to and how often. The classic policy's own pass holds a
//! tree to that size. A worker the registry lost is taken out of the trees
//! by the owner's own task, which follows the registry's events
//! ([`LentTrees::follow_removals`]): the registry sends one for every
//! removal, so a worker a mesh peer's word took away leaves the trees as
//! one this process removed does. The engine calls a tree's match and
//! nothing else.
//!
//! The engine is handed one object for what this process knows of its
//! workers' caches ([`EngineCaches`]): the cache-event monitor's indexes and
//! these trees.

use std::sync::Arc;

use kv_index::{ModelTrees, TenantId, TokenTree};
use parking_lot::Mutex;
use smg_policy_controller::{
    document::{IndexSpec, PolicySpec},
    engine::{CacheIndex, CacheIndexes, Phase},
};
use smg_worker::registry::event::WorkerEvent;
use tokio::{
    runtime::Handle,
    sync::broadcast::error::{RecvError, TryRecvError},
};
use tracing::error;

use super::{
    cache_aware::evict_trees, kv_index::event_monitor::KvEventMonitor, normalize_model_key,
    utils::PeriodicTask,
};
use crate::routers::common::snapshot::WorkerRegistry;

/// The owner of the token trees the policy engine is lent, one a model.
#[derive(Debug)]
pub(crate) struct LentTrees {
    /// The document the engine runs: a model's `index` entry sizes its
    /// tree ([`PolicySpec::index_for`]).
    document: PolicySpec,
    /// A tree's page in tokens: the document's block size.
    page: usize,
    /// One store for each size and cadence the models asked so far take.
    stores: Mutex<Vec<SizedTrees>>,
}

/// The trees of the models whose `index` entries state one size and one
/// cadence, with the pass that holds them to it.
#[derive(Debug)]
struct SizedTrees {
    max_size: u64,
    sweep_s: u32,
    trees: Arc<ModelTrees>,
    /// The size pass on its timer. It ends with the owner.
    _size_pass: Option<PeriodicTask>,
}

impl SizedTrees {
    /// A store of trees of `page` tokens a page, held to `max_size` tokens
    /// by the classic policy's pass every `sweep_s` seconds. The document's
    /// longest cadence spawns no pass: it says a tree that is never trimmed,
    /// which the routing settings' interval of zero is written as, and the
    /// classic policy runs no pass at that interval.
    fn new(page: usize, max_size: u64, sweep_s: u32) -> Self {
        let trees = Arc::new(ModelTrees::new(page));
        let size_pass = (sweep_s != u32::MAX).then(|| {
            let trees = Arc::clone(&trees);
            let max_size = usize::try_from(max_size).unwrap_or(usize::MAX);
            PeriodicTask::spawn(u64::from(sweep_s), "Eviction", move || {
                evict_trees(&trees, max_size);
            })
        });
        Self {
            max_size,
            sweep_s,
            trees,
            _size_pass: size_pass,
        }
    }
}

impl LentTrees {
    /// The owner under the numbers of `document`, the document the engine
    /// runs, whose block size is `page` tokens. The document the routing
    /// settings mean states the settings' own numbers, so the trees take
    /// them as the classic policy's do. A supplied document states its own.
    pub(crate) fn of_document(document: PolicySpec, page: u32) -> Self {
        Self {
            document,
            page: usize::try_from(page).unwrap_or(usize::MAX),
            stores: Mutex::new(Vec::new()),
        }
    }

    /// The token tree of `model`, made at its first ask: every later ask is
    /// handed the same tree. It stands with the trees of its size and its
    /// cadence: the model's `index` entry of the document, or the entry's
    /// defaults where the document writes none for it. The engine asks when
    /// it builds a model's table, not for a request.
    pub(crate) fn token_tree(&self, model: &str) -> Arc<TokenTree> {
        let entry = self.document.index_for(model);
        let (max_size, sweep_s) = entry.map_or_else(
            || {
                let unwritten = IndexSpec::default();
                (unwritten.max_size, unwritten.sweep_s)
            },
            |(_, entry)| (entry.max_size, entry.sweep_s),
        );
        let mut stores = self.stores.lock();
        let found = stores
            .iter()
            .position(|store| store.max_size == max_size && store.sweep_s == sweep_s);
        let at = found.unwrap_or_else(|| {
            stores.push(SizedTrees::new(self.page, max_size, sweep_s));
            stores.len() - 1
        });
        stores[at]
            .trees
            .token_tree_or_new(normalize_model_key(model))
    }

    /// Starts the task that takes a worker the registry lost out of the
    /// trees, on `runtime`. It wakes on the registry's events, and at a
    /// removal it purges ([`Self::forget_the_removed`]). The registry sends
    /// the event for every removal, whichever way the worker left: a
    /// removal of this process's, or a mesh peer's word, which runs no step
    /// of a removal here. The events that wait together are one purge.
    /// Events the task did not hear may hold a removal, so they are one
    /// too. A registration, a replacement and a status change purge
    /// nothing. A purge walks a tree for each worker that left, so it runs
    /// off the runtime's own threads.
    ///
    /// The task holds the registry and the trees weakly. It ends with the
    /// registry, at the first event after the trees are gone, or with its
    /// runtime.
    pub(crate) fn follow_removals(
        self: &Arc<Self>,
        registry: &Arc<WorkerRegistry>,
        runtime: &Handle,
    ) {
        let mut events = registry.subscribe_events();
        let trees = Arc::downgrade(self);
        let registry = Arc::downgrade(registry);
        runtime.spawn(async move {
            loop {
                let mut left = match events.recv().await {
                    Ok(event) => is_removal(&event),
                    Err(RecvError::Lagged(_)) => true,
                    Err(RecvError::Closed) => return,
                };
                loop {
                    match events.try_recv() {
                        Ok(event) => left |= is_removal(&event),
                        Err(TryRecvError::Lagged(_)) => left = true,
                        Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                    }
                }
                let (Some(trees), Some(registry)) = (trees.upgrade(), registry.upgrade()) else {
                    return;
                };
                if !left {
                    continue;
                }
                let purge =
                    tokio::task::spawn_blocking(move || trees.forget_the_removed(&registry));
                // A purge that panicked is this task's last: the size pass
                // alone bounds the trees from here on.
                if let Err(ended) = purge.await {
                    if ended.is_panic() {
                        error!("A purge of the lent token trees panicked: no worker that leaves is taken out of them from here on");
                    }
                    return;
                }
            }
        });
    }

    /// Takes out of every tree each address no worker of `registry` has. A
    /// worker that left is never written again, so the size pass alone
    /// would keep its prompts for good. The registry is asked after each
    /// address a tree holds, so the purge needs no word of which worker
    /// left or which models it served. An address a worker is registered
    /// under again by then keeps what the trees hold for it. An address
    /// written behind its purge, by a decision made on the list of a moment
    /// before, leaves at the next purge.
    fn forget_the_removed(&self, registry: &WorkerRegistry) {
        // The stores are copied out and the lock is let go ahead of the
        // purge: a purge walks a tree once for every worker that left, and
        // a decision that builds a model's table asks for its tree under
        // this lock.
        let stores: Vec<Arc<ModelTrees>> = self
            .stores
            .lock()
            .iter()
            .map(|store| Arc::clone(&store.trees))
            .collect();
        for tree in stores.iter().flat_map(|trees| trees.token_trees()) {
            for address in tree.get_tenant_token_counts().into_keys() {
                if registry.get_by_url(&address).is_none() {
                    tree.remove_tenant_all(&TenantId::from(address));
                }
            }
        }
    }
}

/// Whether an event of the registry's took a worker away. A registration, a
/// replacement and a status change take none.
fn is_removal(event: &WorkerEvent) -> bool {
    matches!(event, WorkerEvent::Removed { .. })
}

/// What a gateway answers the policy engine of its workers' caches, as one
/// object: the indexes of the cache-event monitor, and the token trees it
/// lends.
pub(crate) struct EngineCaches {
    /// The cache-event monitor, where the process keeps one.
    events: Option<Arc<KvEventMonitor>>,
    /// The lent trees, where the document in force reads sent prefixes.
    trees: Option<Arc<LentTrees>>,
}

impl EngineCaches {
    /// The one object over `events` and `trees`: `None` where the process
    /// keeps neither, so the engine is handed nothing.
    pub(crate) fn of(
        events: Option<Arc<KvEventMonitor>>,
        trees: Option<Arc<LentTrees>>,
    ) -> Option<Self> {
        (events.is_some() || trees.is_some()).then_some(Self { events, trees })
    }
}

impl CacheIndexes for EngineCaches {
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        CacheIndexes::index(self.events.as_deref()?, model)
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        CacheIndexes::block_size(self.events.as_deref()?, model)
    }

    /// The tree of a single request's model. A pair's sides are lent none:
    /// the engine reads no tree for a prefill or a decode side yet.
    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        match phase {
            Phase::Single => Some(self.trees.as_ref()?.token_tree(model)),
            Phase::Prefill | Phase::Decode | Phase::Encode => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use openai_protocol::{worker::WorkerStatus, UNKNOWN_MODEL_ID};
    use smg_worker::{registry::RemoteWorkerState, test_support::ready_worker};

    use super::*;
    use crate::{
        config::{types::PolicyConfig, RouterConfig},
        policies::kv_index::index_backend::KvIndex,
    };

    const MODEL: &str = "llama";

    /// The document the routing settings mean under `cache_aware` with a
    /// tree held to `max_tree_size` tokens by a pass every `every_secs`
    /// seconds.
    fn document(max_tree_size: usize, every_secs: u64) -> PolicySpec {
        let config = RouterConfig {
            policy: PolicyConfig::cache_aware(0.5, 32, 1.1, every_secs, max_tree_size),
            ..RouterConfig::default()
        };
        config.policy_document().expect("the settings' document")
    }

    /// The owner under `document`, at pages of 16 tokens.
    fn owner(document: PolicySpec) -> LentTrees {
        LentTrees::of_document(document, 16)
    }

    /// An `index` entry of `max_size` tokens trimmed every `sweep_s`
    /// seconds.
    fn entry(max_size: u64, sweep_s: u32) -> IndexSpec {
        IndexSpec {
            max_size,
            sweep_s,
            ..IndexSpec::default()
        }
    }

    /// The workers `model`'s tree holds a prompt for, sorted.
    fn held(trees: &LentTrees, model: &str) -> Vec<String> {
        let tree = trees.token_tree(model);
        let mut workers: Vec<String> = tree.get_tenant_token_counts().into_keys().collect();
        workers.sort();
        workers
    }

    /// Eight prompts of 64 tokens for one worker: 512 tokens in `model`'s
    /// tree.
    fn filled(trees: &LentTrees, model: &str) -> Arc<TokenTree> {
        let tree = trees.token_tree(model);
        for prompt in 0..8_u32 {
            let tokens: Vec<u32> = (prompt * 1_000..prompt * 1_000 + 64).collect();
            tree.insert_tokens(&tokens, "http://a:1");
        }
        assert_eq!(tree.total_token_size(), 512);
        tree
    }

    /// A tree's page is the block size the owner was made with: the
    /// document's, as the start reads it.
    #[test]
    fn a_trees_page_is_the_documents_block_size() {
        let trees = LentTrees::of_document(document(1_000, 0), 8);
        assert_eq!(trees.token_tree(MODEL).page_size(), 8);
        assert_eq!(trees.token_tree("another").page_size(), 8);
    }

    /// A model is handed one tree, at its first ask and at every later one,
    /// and another model another.
    #[test]
    fn a_model_is_handed_one_tree() {
        let trees = owner(document(1_000, 0));
        let first = trees.token_tree(MODEL);
        assert!(Arc::ptr_eq(&first, &trees.token_tree(MODEL)));
        assert!(!Arc::ptr_eq(&first, &trees.token_tree("another")));
    }

    /// A model named by the empty name is kept under the name the classic
    /// policy keeps it under, so the two names are one tree.
    #[test]
    fn the_empty_model_name_is_the_unknown_models_tree() {
        let trees = owner(document(1_000, 0));
        let unnamed = trees.token_tree("");
        assert!(Arc::ptr_eq(&unnamed, &trees.token_tree(UNKNOWN_MODEL_ID)));
        let stores = trees.stores.lock();
        let kept = stores[0].trees.token_tree(UNKNOWN_MODEL_ID);
        assert!(Arc::ptr_eq(&unnamed, &kept.expect("under the normal name")));
        assert!(stores[0].trees.token_tree("").is_none());
    }

    /// A model the document writes no `index` entry for takes the entry's
    /// own defaults, and a store is one size and one cadence: two entries
    /// of one size and two cadences are two stores, each with its pass.
    #[test]
    fn a_store_is_one_size_and_one_cadence() {
        let mut stated = document(128, 1);
        stated.index.clear();
        let trees = owner(stated.clone());
        trees.token_tree(MODEL);
        let unwritten = IndexSpec::default();
        {
            let stores = trees.stores.lock();
            assert_eq!(
                (stores[0].max_size, stores[0].sweep_s),
                (unwritten.max_size, unwritten.sweep_s),
                "no entry: the entry's defaults"
            );
        }

        stated.index.insert("slow*".to_owned(), entry(128, 2));
        stated.index.insert("fast*".to_owned(), entry(128, 1));
        let trees = owner(stated);
        let (slow, fast) = (trees.token_tree("slow-a"), trees.token_tree("fast-a"));
        assert!(!Arc::ptr_eq(&slow, &fast));
        trees.token_tree("slow-b");
        let stores = trees.stores.lock();
        let numbers: Vec<(u64, u32)> = stores
            .iter()
            .map(|store| (store.max_size, store.sweep_s))
            .collect();
        assert_eq!(numbers, [(128, 2), (128, 1)]);
        assert_eq!(stores[0].trees.token_tree_count(), 2, "both slow models");
    }

    /// Registers a worker at `url` that serves [`MODEL`].
    fn worker_at(registry: &WorkerRegistry, url: &str) {
        registry
            .register(ready_worker(url, MODEL))
            .expect("a worker");
    }

    /// Of the registry's events a removal alone asks for a purge: a
    /// registration, a replacement and a status change take no worker away.
    #[test]
    fn a_removal_alone_asks_for_a_purge() {
        let registry = WorkerRegistry::new();
        let mut events = registry.subscribe_events();
        worker_at(&registry, "http://a:1");
        let id = registry.register_or_replace(ready_worker("http://a:1", MODEL));
        // One of the two differs from the status the worker stands at.
        for status in [WorkerStatus::NotReady, WorkerStatus::Ready] {
            registry.transition_status(&id, status);
        }
        registry.remove(&id).expect("the removal");

        let mut heard: Vec<(&str, bool)> = std::iter::from_fn(|| events.try_recv().ok())
            .map(|event| {
                let kind = match event {
                    WorkerEvent::Registered { .. } => "a registration",
                    WorkerEvent::Replaced { .. } => "a replacement",
                    WorkerEvent::StatusChanged { .. } => "a status change",
                    WorkerEvent::Removed { .. } => "a removal",
                };
                (kind, is_removal(&event))
            })
            .collect();
        heard.sort_unstable();
        heard.dedup();
        let asked = [
            ("a registration", false),
            ("a removal", true),
            ("a replacement", false),
            ("a status change", false),
        ];
        assert_eq!(heard, asked);
    }

    /// One prompt of 64 tokens sent to each of `workers`, in `model`'s tree.
    fn sent(trees: &LentTrees, model: &str, workers: &[&str]) {
        let tokens: Vec<u32> = (0..64).collect();
        let tree = trees.token_tree(model);
        for worker in workers {
            tree.insert_tokens(&tokens, worker);
        }
    }

    /// Waits for the task that follows the registry to get to `holds`.
    async fn until(what: &str, mut holds: impl FnMut() -> bool) {
        let bound = Instant::now() + Duration::from_secs(60);
        while !holds() {
            assert!(Instant::now() < bound, "the task never got to: {what}");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// A worker the registry lost is taken out of every tree, whichever
    /// model's it is and whichever size its store is held to. It leaves the
    /// trees whichever way it left the registry: by a mesh peer's word,
    /// which runs no step of a removal here, or by this process's removal.
    /// The purge asks the registry after each address, so it takes out an
    /// address no worker has.
    #[tokio::test]
    async fn a_worker_the_registry_lost_is_taken_out_of_every_tree() {
        let mut stated = document(1_000, 0);
        let sized = entry(4_096, u32::MAX);
        stated.index.insert("another".to_owned(), sized);
        let trees = Arc::new(owner(stated));
        let registry = Arc::new(WorkerRegistry::new());
        worker_at(&registry, "http://a:1");
        worker_at(&registry, "http://b:1");
        registry.on_remote_worker_state(&RemoteWorkerState {
            worker_id: "a-peers".to_owned(),
            model_id: MODEL.to_owned(),
            url: "http://peer:1".to_owned(),
            health: true,
            version: 1,
            spec: Vec::new(),
        });
        let imported = registry.get_id_by_url("http://peer:1").expect("the import");
        let all = ["http://a:1", "http://b:1", "http://gone:1", "http://peer:1"];
        for model in [MODEL, "another"] {
            sent(&trees, model, &all);
        }
        assert_eq!(trees.stores.lock().len(), 2, "two sizes, two stores");
        trees.follow_removals(&registry, &Handle::current());

        let tombstone = registry.remove_remote(&imported);
        tombstone.expect("the removal a peer's tombstone makes");
        let left = ["http://a:1", "http://b:1"];
        until("the purge behind a peer's tombstone", || {
            held(&trees, MODEL) == left && held(&trees, "another") == left
        })
        .await;

        registry.remove_by_url("http://b:1").expect("the removal");
        until("the purge behind a removal of this process's", || {
            held(&trees, MODEL) == ["http://a:1"] && held(&trees, "another") == ["http://a:1"]
        })
        .await;
    }

    /// A removal the task did not hear is purged too. More events than the
    /// registry's channel holds, the task never run between them, lose the
    /// removal's own event to every listener, and no event that is kept is
    /// a removal.
    #[tokio::test]
    async fn a_removal_that_went_unheard_is_purged_too() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let registry = Arc::new(WorkerRegistry::new());
        worker_at(&registry, "http://a:1");
        worker_at(&registry, "http://b:1");
        sent(&trees, MODEL, &["http://a:1", "http://b:1"]);
        trees.follow_removals(&registry, &Handle::current());

        let mut witness = registry.subscribe_events();
        registry.remove_by_url("http://b:1").expect("the removal");
        for wave in 0..1_500 {
            worker_at(&registry, &format!("http://wave-{wave}:1"));
        }
        assert!(
            matches!(witness.try_recv(), Err(TryRecvError::Lagged(_))),
            "the burst must overflow the channel, or this case holds nothing"
        );
        let kept = std::iter::from_fn(|| witness.try_recv().ok());
        assert_eq!(
            kept.filter(|event| matches!(event, WorkerEvent::Removed { .. }))
                .count(),
            0,
            "a removal that is heard would purge by itself"
        );
        until("the purge of the worker that left unheard", || {
            held(&trees, MODEL) == ["http://a:1"]
        })
        .await;
    }

    /// The task holds neither the registry nor the trees: it ends with the
    /// registry, and at the first event after the trees are gone.
    #[tokio::test]
    async fn the_task_ends_with_the_registry_or_the_trees() {
        let registry = Arc::new(WorkerRegistry::new());
        let trees = Arc::new(owner(document(1_000, 0)));
        trees.follow_removals(&registry, &Handle::current());
        assert_eq!(Arc::weak_count(&trees), 1, "the task's own handle");
        drop(registry);
        until("its end with the registry", || Arc::weak_count(&trees) == 0).await;

        let registry = Arc::new(WorkerRegistry::new());
        let handles = Arc::weak_count(&registry);
        trees.follow_removals(&registry, &Handle::current());
        assert_eq!(Arc::weak_count(&registry), handles + 1);
        drop(trees);
        worker_at(&registry, "http://a:1");
        until("its end with the trees", || {
            Arc::weak_count(&registry) == handles
        })
        .await;
    }

    /// The size pass holds a model's tree to the size of the model's own
    /// `index` entry, on that entry's timer: the settings' numbers under the
    /// document the settings mean, and a narrower entry's for the models it
    /// names. The settings' interval of zero runs no pass.
    #[test]
    fn the_size_pass_holds_a_tree_to_its_index_entrys_size() {
        let mut stated = document(128, 1);
        let roomier = entry(256, 1);
        stated.index.insert("roomy*".to_owned(), roomier);
        let trees = owner(stated);
        let (small, large) = (filled(&trees, MODEL), filled(&trees, "roomy-a"));
        let until = Instant::now() + Duration::from_secs(20);
        let over = || small.total_token_size() > 128 || large.total_token_size() > 256;
        while over() && Instant::now() < until {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(small.total_token_size() <= 128, "the pass ran on its timer");
        // The other entry's pass ran too, and cut its tree to its own size.
        let kept = large.total_token_size();
        assert!(kept <= 256 && kept > 128, "its own entry's size: {kept}");

        let unheld = owner(document(128, 0));
        unheld.token_tree(MODEL);
        let stores = unheld.stores.lock();
        assert!(
            stores[0]._size_pass.is_none(),
            "no pass at an interval of zero"
        );
    }

    /// The engine is lent the tree of a single request's model, the owner's
    /// own, and none for a pair's side or an encode request. Where no
    /// owner stands it is lent none.
    #[test]
    fn the_engine_is_lent_a_single_requests_tree_and_no_sides() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let caches = EngineCaches::of(None, Some(Arc::clone(&trees))).expect("the trees");
        let lent = caches.token_tree(MODEL, Phase::Single).expect("a tree");
        assert!(Arc::ptr_eq(&lent, &trees.token_tree(MODEL)));
        for side in [Phase::Prefill, Phase::Decode, Phase::Encode] {
            assert!(caches.token_tree(MODEL, side).is_none(), "{side:?}");
        }
        // No monitor stands behind this object: no index and no block size.
        assert!(caches.index(MODEL).is_none() && caches.block_size(MODEL).is_none());

        let monitor = Arc::new(KvEventMonitor::new());
        let caches = EngineCaches::of(Some(monitor), None).expect("the monitor");
        assert!(caches.token_tree(MODEL, Phase::Single).is_none());
        assert!(EngineCaches::of(None, None).is_none());
    }

    /// The object answers a model's index and its block size as the
    /// cache-event monitor behind it does.
    #[test]
    fn the_monitors_index_and_block_size_are_read_through_the_one_object() {
        let monitor = Arc::new(KvEventMonitor::new());
        let index = Arc::new(KvIndex::chain());
        monitor.set_index(MODEL, Arc::clone(&index));
        monitor.set_block_size(MODEL, 4);
        let trees = Arc::new(owner(document(1_000, 0)));
        let caches = EngineCaches::of(Some(monitor), Some(trees)).expect("both");
        assert_eq!(caches.block_size(MODEL), Some(4));
        let lent = caches.index(MODEL).expect("the model's index");
        assert!(std::ptr::addr_eq(Arc::as_ptr(&lent), Arc::as_ptr(&index)));
        assert!(caches.index("another").is_none());
        assert_eq!(caches.block_size("another"), None);
    }
}
