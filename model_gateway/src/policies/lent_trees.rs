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
use tracing::{error, warn};

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
    /// too, of every address. A registration, a replacement and a status
    /// change purge nothing. A purge walks a tree for each worker that
    /// left, so it runs off the runtime's own threads, and the task reads
    /// no event until it ends.
    ///
    /// The task holds the trees weakly, and of the registry its events
    /// alone. It ends with the registry, at the first event after the
    /// trees are gone, or with its runtime.
    pub(crate) fn follow_removals(self: &Arc<Self>, registry: &WorkerRegistry, runtime: &Handle) {
        let mut events = registry.subscribe_events();
        let trees = Arc::downgrade(self);
        runtime.spawn(async move {
            loop {
                let mut left = Left::default();
                match events.recv().await {
                    Ok(event) => left.note(&event),
                    Err(RecvError::Lagged(lost)) => left.unheard += lost,
                    Err(RecvError::Closed) => return,
                }
                loop {
                    match events.try_recv() {
                        Ok(event) => left.note(&event),
                        Err(TryRecvError::Lagged(lost)) => left.unheard += lost,
                        Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                    }
                }
                let Some(trees) = trees.upgrade() else {
                    return;
                };
                if !left.asks_for_a_purge() {
                    continue;
                }
                let purge = tokio::task::spawn_blocking(move || trees.forget_the_removed(&left));
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

    /// Takes the workers `left` says left out of every tree. A worker that
    /// left is never written again, so the size pass alone would keep its
    /// prompts for good.
    ///
    /// An address a removal named leaves, and nobody is asked: it leaves
    /// whether or not a worker is registered under it again by now. The
    /// worker that rejoins starts with no prompt, as its cache may not have
    /// outlived it.
    ///
    /// Where events went unheard, every prompt leaves every tree, and a
    /// line at warn says so. A removal may be among those events, and a
    /// registration under its address behind it. The worker that stands
    /// there is then not the one the prompts were sent to, and no event
    /// that was heard tells the two apart: the registry's word keeps no
    /// address. The trees fill again from the requests that follow.
    ///
    /// The trees are emptied by the size pass's own call, at no size
    /// ([`TokenTree::evict_tenant_by_size`]): one pass, where a purge by
    /// name walks a tree once for each address. The call takes one leaf
    /// out at a time and debits it from its worker. A prompt a request
    /// writes meanwhile is never left uncounted. It stays, with the prompt
    /// it extends, as it was sent to the worker that stands. Or it leaves
    /// with its count kept, where the pass takes it between its attach
    /// and its count. The call ends where the tree's total reads zero.
    /// Every address stays listed at no tokens, but one a removal named:
    /// that one is purged by name behind the pass.
    fn forget_the_removed(&self, left: &Left) {
        if left.unheard > 0 {
            warn!(
                "{} events of the worker registry went unheard: the lent token trees are emptied, and cache_aware remembers no prompt until they fill again",
                left.unheard
            );
        }
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
            if left.unheard > 0 {
                // A tree's own `clear` asks for no writer beside it. This
                // call may run beside the size pass's own: the second of
                // the two to take a worker out of a node debits nothing.
                tree.evict_tenant_by_size(0);
            }
            // Behind an eviction a named address's count reads zero, and
            // the purge drops its entries alone.
            for address in &left.addresses {
                tree.remove_tenant_all(&TenantId::from(address.as_str()));
            }
        }
    }
}

/// What the registry's events of one wake ask of the trees.
#[derive(Debug, Default)]
struct Left {
    /// The addresses of the workers a removal took away.
    addresses: Vec<String>,
    /// How many events went unheard: a removal may be among them.
    unheard: u64,
}

impl Left {
    /// Notes `event`: a removal names the address that left. A
    /// registration, a replacement and a status change take no worker
    /// away.
    fn note(&mut self, event: &WorkerEvent) {
        if let WorkerEvent::Removed { worker, .. } = event {
            self.addresses.push(worker.url().to_owned());
        }
    }

    /// Whether the events ask for a purge: a removal was heard, or events
    /// went unheard.
    fn asks_for_a_purge(&self) -> bool {
        self.unheard > 0 || !self.addresses.is_empty()
    }
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
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
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

    /// The workers `model`'s tree holds a prompt for, sorted: a worker it
    /// lists at no tokens holds none.
    fn held(trees: &LentTrees, model: &str) -> Vec<String> {
        let counts = trees.token_tree(model).get_tenant_token_counts();
        let mut workers: Vec<String> = counts
            .into_iter()
            .filter_map(|(worker, tokens)| (tokens > 0).then_some(worker))
            .collect();
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

    /// Of the registry's events a removal alone asks for a purge, and it
    /// names the address that left: a registration, a replacement and a
    /// status change take no worker away.
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

        let mut heard: Vec<(&str, Vec<String>)> = std::iter::from_fn(|| events.try_recv().ok())
            .map(|event| {
                let kind = match event {
                    WorkerEvent::Registered { .. } => "a registration",
                    WorkerEvent::Replaced { .. } => "a replacement",
                    WorkerEvent::StatusChanged { .. } => "a status change",
                    WorkerEvent::Removed { .. } => "a removal",
                };
                let mut left = Left::default();
                left.note(&event);
                assert_eq!(left.asks_for_a_purge(), !left.addresses.is_empty());
                (kind, left.addresses)
            })
            .collect();
        heard.sort_unstable();
        heard.dedup();
        let named = [
            ("a registration", vec![]),
            ("a removal", vec!["http://a:1".to_owned()]),
            ("a replacement", vec![]),
            ("a status change", vec![]),
        ];
        assert_eq!(heard, named);
        let unheard = Left {
            unheard: 1,
            ..Left::default()
        };
        assert!(unheard.asks_for_a_purge(), "events that went unheard");
        assert!(!Left::default().asks_for_a_purge());
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
    #[tokio::test]
    async fn a_worker_the_registry_lost_is_taken_out_of_every_tree() {
        let mut stated = document(1_000, 0);
        let sized = entry(4_096, u32::MAX);
        stated.index.insert("another".to_owned(), sized);
        let trees = Arc::new(owner(stated));
        let registry = WorkerRegistry::new();
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
        let all = ["http://a:1", "http://b:1", "http://peer:1"];
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

    /// A worker a removal named leaves the trees whether or not a worker is
    /// registered under its address again by the purge: the one that
    /// rejoins starts with no prompt. The three events wait together here.
    /// The wake's first read hears another worker's registration, a later
    /// read of the same wake hears the removal, and the purge finds the
    /// address registered.
    #[tokio::test]
    async fn a_worker_that_rejoins_under_its_address_starts_with_no_prompt() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let registry = WorkerRegistry::new();
        worker_at(&registry, "http://a:1");
        worker_at(&registry, "http://b:1");
        sent(&trees, MODEL, &["http://a:1", "http://b:1"]);
        trees.follow_removals(&registry, &Handle::current());

        worker_at(&registry, "http://c:1");
        registry.remove_by_url("http://b:1").expect("the removal");
        worker_at(&registry, "http://b:1");
        until("the purge of the address that rejoined", || {
            held(&trees, MODEL) == ["http://a:1"]
        })
        .await;
        assert!(registry.get_by_url("http://b:1").is_some(), "it rejoined");
    }

    /// Events the task did not hear empty every tree. More events than the
    /// registry's channel holds, the task never run between them, lose a
    /// removal's own event to every listener, and the registration under
    /// its address behind it: no event that is kept names that address.
    /// The worker that rejoined starts with no prompt, and so does the one
    /// that stood throughout, as nothing heard tells the two apart. Each
    /// is debited what left and stays listed: no count is dropped whole.
    /// A third worker's removal is the last event, so it is kept and heard
    /// in the same wake: that address leaves the tree's list too.
    #[tokio::test]
    async fn events_that_went_unheard_empty_every_tree() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let registry = WorkerRegistry::new();
        let all = ["http://a:1", "http://b:1", "http://c:1"];
        for worker in all {
            worker_at(&registry, worker);
        }
        sent(&trees, MODEL, &all);
        trees.follow_removals(&registry, &Handle::current());

        let mut witness = registry.subscribe_events();
        registry.remove_by_url("http://b:1").expect("the removal");
        worker_at(&registry, "http://b:1");
        for wave in 0..1_500 {
            worker_at(&registry, &format!("http://wave-{wave}:1"));
        }
        registry.remove_by_url("http://c:1").expect("the removal");
        assert!(
            matches!(witness.try_recv(), Err(TryRecvError::Lagged(_))),
            "the burst must overflow the channel, or this case holds nothing"
        );
        let kept: Vec<WorkerEvent> = std::iter::from_fn(|| witness.try_recv().ok()).collect();
        let (last, waves) = kept.split_last().expect("the events that are kept");
        assert!(
            matches!(last, WorkerEvent::Removed { worker, .. } if worker.url() == "http://c:1"),
            "the third worker's removal is kept, and heard"
        );
        let of_a_wave = |event: &WorkerEvent| {
            let WorkerEvent::Registered { worker, .. } = event else {
                return false;
            };
            worker.url().starts_with("http://wave-")
        };
        assert!(
            waves.iter().all(of_a_wave),
            "an event that names the address would be heard, and purge it by its name"
        );
        // The purge by name runs behind the eviction: the list is waited
        // for, and the prompts are gone by then.
        let tree = trees.token_tree(MODEL);
        let listed = || {
            let mut listed: Vec<(String, usize)> =
                tree.get_tenant_token_counts().into_iter().collect();
            listed.sort();
            listed
        };
        let debited = ["http://a:1", "http://b:1"].map(|stands| (stands.to_owned(), 0));
        until("the trees emptied behind unheard events", || {
            listed() == debited
        })
        .await;
        // The root's own entry lists the addresses, and is no prompt.
        let prompts = tree.iter_entries().filter(|(prompt, _)| !prompt.is_empty());
        assert_eq!(prompts.count(), 0, "no prompt is left");
        for stands in ["http://a:1", "http://b:1"] {
            assert!(registry.get_by_url(stands).is_some(), "{stands} stands");
        }
    }

    /// A prompt a request writes while the trees are emptied stays counted
    /// to its worker. Two threads write prompts by the engine's own call,
    /// four to a worker, and the trees are emptied again and again until
    /// 2,000 are written. No prompt that is left then is one its worker's
    /// count lacks. A purge by name left such a prompt while it dropped a
    /// worker's whole count behind its walk, with a prompt written between
    /// the two.
    #[test]
    fn a_prompt_written_while_the_trees_are_emptied_stays_counted() {
        let trees = owner(document(1_000, 0));
        let tree = trees.token_tree(MODEL);
        let unheard = Left {
            unheard: 1,
            ..Left::default()
        };
        let written = AtomicUsize::new(0);
        thread::scope(|scope| {
            for writer in 0..2_u32 {
                let (tree, written) = (&tree, &written);
                scope.spawn(move || {
                    for prompt in 0..20_000_u32 {
                        if written.load(Ordering::Relaxed) >= 2_000 {
                            break;
                        }
                        // 64 tokens no other prompt begins with.
                        let first = writer * 2_000_000 + prompt * 64;
                        let tokens: Vec<u32> = (first..first + 64).collect();
                        let worker = format!("http://{writer}-{}:1", prompt / 4);
                        tree.match_and_insert(&tokens, &worker);
                        written.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            while written.load(Ordering::Relaxed) < 2_000 {
                trees.forget_the_removed(&unheard);
            }
        });

        let counted = tree.get_tenant_token_counts();
        let mut left: HashMap<String, usize> = HashMap::new();
        // The root's own entry lists the addresses, and is no prompt.
        let prompts = tree.iter_entries().filter(|(prompt, _)| !prompt.is_empty());
        for (prompt, workers) in prompts {
            assert_eq!(prompt.len(), 64, "a whole prompt, of one worker");
            assert_eq!(workers.len(), 1, "a whole prompt, of one worker");
            *left.entry(workers[0].0.to_string()).or_default() += 64;
        }
        for (worker, tokens) in left {
            let count = counted.get(&worker).copied().unwrap_or(0);
            assert!(
                count >= tokens,
                "{worker} holds {tokens} tokens, counted {count}"
            );
        }
    }

    /// The task holds the trees weakly, and nothing of the registry but its
    /// events: it ends with the registry, and at the first event after the
    /// trees are gone.
    #[tokio::test]
    async fn the_task_ends_with_the_registry_or_the_trees() {
        let tasks = || Handle::current().metrics().num_alive_tasks();
        let registry = WorkerRegistry::new();
        let trees = Arc::new(owner(document(1_000, 0)));
        trees.follow_removals(&registry, &Handle::current());
        let alive = (tasks(), Arc::weak_count(&trees));
        assert_eq!(alive, (1, 1), "the task, and its own handle on the trees");
        drop(registry);
        until("its end with the registry", || tasks() == 0).await;
        assert_eq!(Arc::weak_count(&trees), 0);

        let registry = WorkerRegistry::new();
        trees.follow_removals(&registry, &Handle::current());
        drop(trees);
        worker_at(&registry, "http://a:1");
        assert_eq!(tasks(), 1, "it has yet to read the event");
        until("its end with the trees", || tasks() == 0).await;
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
