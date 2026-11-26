//! The trees a gateway lends Grand Central: a model's token tree, and
//! its string tree for the prompts sent as text.
//!
//! Where Grand Central places every request no classic policy object is
//! made, so no policy keeps the trees of the prompts this process sent. An
//! owner of them stands in a policy's place ([`LentTrees`]). It is made
//! where a document put in force first reads the prefixes sent to each
//! worker: at the engine's start, or behind a worker's `policy` label.
//!
//! The owner keeps the trees in the store type the classic policy keeps its
//! own in, under the numbers of the document in force. Its block size is a
//! token tree's page, and a model's `index` entry states the size its trees
//! are held to and how often. The one size holds a token tree in tokens and
//! a string tree in its own unit: its chars and a weight for each node a
//! worker stands on. The classic policy's one setting does the same, and
//! the classic policy's own pass holds a tree to it.
//!
//! A worker the registry lost is taken out of the trees by the owner's own
//! task, which follows the registry's events
//! ([`LentTrees::follow_removals`]). The registry sends one for every
//! removal. So a worker a mesh peer's word took away leaves the trees as
//! one this process removed does. The engine calls a tree's match and
//! nothing else.
//!
//! The engine is handed one object for what this process knows of its
//! workers' caches ([`EngineCaches`]): the cache-event monitor's indexes and
//! these trees, each to a single request's decision alone.

use std::sync::{Arc, OnceLock};

use kv_index::{ModelTrees, TenantId, TokenTree, Tree as StringTree};
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use smg_grand_central::{
    document::{IndexKind, IndexSpec, PolicySpec},
    engine::{CacheIndex, CacheIndexes, Phase},
};
use smg_worker::registry::event::WorkerEvent;
use tokio::{
    runtime::Handle,
    sync::broadcast::error::{RecvError, TryRecvError},
};
use tracing::{debug, error, warn};

use super::{
    cache_aware::evict_trees, kv_index::event_monitor::KvEventMonitor, normalize_model_key,
    utils::PeriodicTask,
};
use crate::routers::common::snapshot::WorkerRegistry;

/// The owner of the trees Grand Central is lent: a token tree and a
/// string tree a model.
#[derive(Debug)]
pub(crate) struct LentTrees {
    /// The document the engine runs: a model's `index` entry sizes its
    /// trees ([`PolicySpec::index_for`]). A document a worker's label puts
    /// in force takes its place ([`Self::follow`]).
    document: Mutex<PolicySpec>,
    /// A token tree's page in tokens: the block size of the document the
    /// owner was made under. Every later document that reads the trees
    /// states the same size, so the page never moves. The settings'
    /// builder holds one block size for the sides of a document, a
    /// label's among them, and refuses a label that names another.
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
    /// Whether the line of a size no pass holds is still owed: the store's
    /// first token tree takes it ([`LentTrees::token_tree`]).
    unsaid: bool,
    /// The size pass on its timer. It ends with the owner.
    _size_pass: Option<PeriodicTask>,
}

impl SizedTrees {
    /// A store of trees, a token tree of `page` tokens a page, each held to
    /// `max_size` by the classic policy's pass every `sweep_s` seconds: a
    /// token tree to that many tokens, a string tree to that size in its
    /// own unit (its chars and a weight for each node). The document's
    /// longest cadence spawns no pass: the routing settings' interval of
    /// zero is written as that cadence, and the classic policy runs no pass
    /// at that interval. A store with no pass holds its trees to no size,
    /// whatever `max_size` reads.
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
            unsaid: size_pass.is_none() && max_size != u64::MAX,
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
            document: Mutex::new(document),
            page: usize::try_from(page).unwrap_or(usize::MAX),
            stores: Mutex::new(Vec::new()),
        }
    }

    /// The owner in `cell`. The first call makes it under the numbers of
    /// `document`, whose block size is `page` tokens. Every later call is
    /// handed that owner, whatever it names.
    ///
    /// The owner is made with the task that takes a worker `workers` lost
    /// out of the trees ([`Self::follow_removals`]), on the runtime the
    /// call is made on. The engine's start runs on one, and so does a
    /// registration. Where none is entered no task is started, nothing
    /// takes a worker out of the trees, and a debug line says so.
    pub(crate) fn lend(
        cell: &OnceLock<Arc<Self>>,
        document: &PolicySpec,
        page: u32,
        workers: &WorkerRegistry,
    ) -> Arc<Self> {
        let made = || {
            let trees = Arc::new(Self::of_document(document.clone(), page));
            match Handle::try_current() {
                Ok(runtime) => trees.follow_removals(workers, &runtime),
                Err(_) => debug!("No runtime is entered: no task takes a worker that leaves out of the lent trees"),
            }
            trees
        };
        Arc::clone(cell.get_or_init(made))
    }

    /// Puts the trees of `cell` under `document`, a document a worker's
    /// label is about to put in force. An owner that stands follows it
    /// ([`Self::follow`]), and the document it followed before is answered
    /// for a caller whose apply is refused. Where none stands and the
    /// document reads the prefixes sent to each worker (`reads`), the owner
    /// is made under it ([`Self::lend`]). The first document in force to
    /// read them is then a label's. A document that reads none makes no
    /// owner.
    pub(crate) fn under(
        cell: &OnceLock<Arc<Self>>,
        document: PolicySpec,
        reads: bool,
        page: u32,
        workers: &WorkerRegistry,
    ) -> Option<PolicySpec> {
        match cell.get() {
            Some(trees) => Some(trees.follow(document)),
            None => {
                if reads {
                    Self::lend(cell, &document, page, workers);
                }
                None
            }
        }
    }

    /// Sizes the trees asked from here on by `document`, the document in
    /// force, and answers the one they were sized by. A model whose `index`
    /// entry stands as it stood is handed the tree it had. A model whose
    /// entry states another size or cadence under `document` is handed the
    /// tree of that entry's store at its next ask. The engine asks where it
    /// builds the model's table again.
    pub(crate) fn follow(&self, document: PolicySpec) -> PolicySpec {
        std::mem::replace(&mut *self.document.lock(), document)
    }

    /// The number of the document the trees are sized by: for a case that
    /// reads which spec they follow.
    #[cfg(test)]
    pub(crate) fn follows(&self) -> u64 {
        self.document.lock().generation
    }

    /// The token tree of `model`, made at its first ask: every later ask
    /// under the same `index` entry is handed the same tree. It stands with
    /// the trees of its size and its cadence ([`Self::store_of`]). The
    /// engine asks when it builds a model's table, not for a request.
    ///
    /// The first token tree of a size under the largest that no pass holds
    /// ([`SizedTrees::new`]) is lent with one line at warn that says so and
    /// names the `index` entries of the tree kind that state that size and
    /// that cadence in the document in force when the line is written:
    /// under an entry of the hash kind no tree is asked for.
    pub(crate) fn token_tree(&self, model: &str) -> Arc<TokenTree> {
        let mut store = self.store_of(model);
        let (max_size, sweep_s) = (store.max_size, store.sweep_s);
        // The store's first token tree takes the line the store owes.
        let unheld = std::mem::take(&mut store.unsaid);
        let trees = Arc::clone(&store.trees);
        // The tree is looked up and the line is written with the lock let
        // go: a decision that builds a model's table asks for its tree
        // under it.
        drop(store);
        let tree = trees.token_tree_or_new(normalize_model_key(model));
        if unheld {
            // Entries of one size and one cadence share the store, and a
            // tree is asked for under the tree kind alone: an entry of the
            // hash kind is not named. The document is read behind the drop
            // of the store: no caller holds the two locks.
            let index: Vec<String> = self
                .document
                .lock()
                .index
                .iter()
                .filter(|(_, entry)| {
                    entry.kind == IndexKind::Tree
                        && entry.max_size == max_size
                        && entry.sweep_s == sweep_s
                })
                .map(|(key, _)| key.clone())
                .collect();
            warn!(
                ?index,
                max_size,
                sweep_s,
                "The lent token trees of this max_size are held to no size: this sweep_s is the longest cadence, at which no size pass starts"
            );
        }
        tree
    }

    /// The string tree of `model`, for the prompts sent as text: made at
    /// its first ask, handed again at every later one, and kept in the
    /// store that keeps the model's token tree. The engine asks when it
    /// builds a model's table, whatever the model's index holds.
    pub(crate) fn string_tree(&self, model: &str) -> Arc<StringTree> {
        let trees = Arc::clone(&self.store_of(model).trees);
        trees.string_tree_or_new(normalize_model_key(model))
    }

    /// The store of the trees of `model`'s size and cadence, made at the
    /// first ask for a tree of that size and cadence. The two are the
    /// model's `index` entry of the document in force at the ask
    /// ([`Self::follow`]), or the entry's defaults where that document
    /// writes none for it. It is handed under the lock of the stores: a
    /// caller takes what it reads of the store and lets the lock go before
    /// it looks a tree up.
    fn store_of(&self, model: &str) -> MappedMutexGuard<'_, SizedTrees> {
        let document = self.document.lock();
        let (max_size, sweep_s) = document.index_for(model).map_or_else(
            || {
                let unwritten = IndexSpec::default();
                (unwritten.max_size, unwritten.sweep_s)
            },
            |(_, entry)| (entry.max_size, entry.sweep_s),
        );
        // Let go ahead of the stores' lock: no caller holds the two.
        drop(document);
        let mut stores = self.stores.lock();
        let found = stores
            .iter()
            .position(|store| store.max_size == max_size && store.sweep_s == sweep_s);
        let at = found.unwrap_or_else(|| {
            stores.push(SizedTrees::new(self.page, max_size, sweep_s));
            stores.len() - 1
        });
        MutexGuard::map(stores, |stores| &mut stores[at])
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
                        error!("A purge of the lent trees panicked: no worker that leaves is taken out of them from here on");
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
    /// The trees are emptied by the size pass's own call, at no size, of
    /// one name on a token tree and on a string tree
    /// ([`TokenTree::evict_tenant_by_size`]): one pass, where a purge by
    /// name makes up to three for each address. The call takes one leaf
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
                "{} events of the worker registry went unheard: the lent trees are emptied, and cache_aware remembers no prompt until they fill again",
                left.unheard
            );
        }
        // The stores are copied out and the lock is let go ahead of the
        // purge: a purge makes up to three passes over a tree for every
        // worker that left, and a decision that builds a model's table asks
        // for its tree under this lock.
        let stores: Vec<Arc<ModelTrees>> = self
            .stores
            .lock()
            .iter()
            .map(|store| Arc::clone(&store.trees))
            .collect();
        for tree in stores.iter().flat_map(|trees| trees.token_trees()) {
            left.purge(
                || tree.evict_tenant_by_size(0),
                |worker| tree.remove_tenant_all(worker),
            );
        }
        for tree in stores.iter().flat_map(|trees| trees.string_trees()) {
            left.purge(
                || tree.evict_tenant_by_size(0),
                |worker| tree.remove_tenant_all(worker),
            );
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

    /// The purge of one tree, of either kind: `empty` is the tree's size
    /// pass at no size, and `forget` its purge of one worker by name.
    fn purge(&self, empty: impl FnOnce(), forget: impl Fn(&TenantId)) {
        if self.unheard > 0 {
            // A tree's own `clear` asks for no writer beside it. This
            // call may run beside the size pass's own: the second of
            // the two to take a worker out of a node debits nothing.
            empty();
        }
        // Behind an eviction a named address's count reads zero, and
        // the purge drops its entries alone.
        for address in &self.addresses {
            forget(&TenantId::from(address.as_str()));
        }
    }
}

/// What a gateway answers Grand Central of its workers' caches, as one
/// object: the indexes of the cache-event monitor, and the trees it lends.
pub(crate) struct EngineCaches {
    /// The cache-event monitor, where the process keeps one.
    events: Option<Arc<KvEventMonitor>>,
    /// The cell of the lent trees. It is filled where a document put in
    /// force first reads sent prefixes: at the start, or behind a worker's
    /// label. The engine asks where it builds a model's table, so trees
    /// lent behind a label are read at the table's next build.
    trees: Arc<OnceLock<Arc<LentTrees>>>,
}

impl EngineCaches {
    /// The one object over `events` and the cell `trees`. `later` says
    /// that a later document can fill the cell: the registry holds a maker
    /// of later specs. `None` where the process keeps no monitor, the cell
    /// holds no trees and no later document can fill it: the engine is then
    /// handed nothing.
    pub(crate) fn of(
        events: Option<Arc<KvEventMonitor>>,
        trees: Arc<OnceLock<Arc<LentTrees>>>,
        later: bool,
    ) -> Option<Self> {
        (events.is_some() || trees.get().is_some() || later).then_some(Self { events, trees })
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
            Phase::Single => Some(self.trees.get()?.token_tree(model)),
            Phase::Prefill | Phase::Decode | Phase::Encode => None,
        }
    }

    /// The string tree of a single request's model, for a request this
    /// gateway holds as text. A pair's sides are lent none, as they are
    /// lent no token tree.
    fn string_tree(&self, model: &str, phase: Phase) -> Option<Arc<StringTree>> {
        match phase {
            Phase::Single => Some(self.trees.get()?.string_tree(model)),
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

    use kv_index::{ContentHash, SequenceHash, StoredBlock};
    use openai_protocol::{worker::WorkerStatus, UNKNOWN_MODEL_ID};
    use smg_worker::{registry::RemoteWorkerState, test_support::ready_worker};

    use super::*;
    use crate::{
        config::{types::PolicyConfig, RouterConfig},
        policies::kv_index::index_backend::{KvIndex, WorkerBlocks},
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

    /// A cell of lent trees that holds `trees`, or nothing.
    fn cell(trees: Option<Arc<LentTrees>>) -> Arc<OnceLock<Arc<LentTrees>>> {
        Arc::new(trees.map_or_else(OnceLock::new, OnceLock::from))
    }

    /// An `index` entry of `max_size` tokens trimmed every `sweep_s`
    /// seconds.
    fn entry(max_size: u64, sweep_s: u32) -> IndexSpec {
        let mut index = IndexSpec::default();
        index.max_size = max_size;
        index.sweep_s = sweep_s;
        index
    }

    /// Whether each of `model`'s two trees holds a prompt for `workers` and
    /// for no other worker: one a tree lists at no size holds none.
    fn holds(trees: &LentTrees, model: &str, workers: &[&str]) -> bool {
        let named = |sizes: HashMap<String, usize>| {
            let mut held: Vec<String> = sizes
                .into_iter()
                .filter_map(|(worker, size)| (size > 0).then_some(worker))
                .collect();
            held.sort();
            held == workers
        };
        named(trees.token_tree(model).get_tenant_token_counts())
            && named(trees.string_tree(model).get_tenant_char_count())
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

    /// What one of `worded`'s texts counts in a string tree: its 64 chars
    /// and 1,024 for the node it makes.
    const TEXT_NODE: usize = 1_088;

    /// Eight texts of 64 chars for one worker, none beginning as another:
    /// eight nodes in `model`'s string tree.
    fn worded(trees: &LentTrees, model: &str) -> Arc<StringTree> {
        let tree = trees.string_tree(model);
        for text in 0..8_u32 {
            tree.insert_text(&format!("{text}{}", "x".repeat(63)), "http://a:1");
        }
        assert_eq!(tree.total_char_size(), 8 * TEXT_NODE);
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

    /// A model is handed one tree of each kind, at its first ask and at
    /// every later one, and another model another. The two trees of a model
    /// stand in one store.
    #[test]
    fn a_model_is_handed_one_tree() {
        let trees = owner(document(1_000, 0));
        let first = trees.token_tree(MODEL);
        assert!(Arc::ptr_eq(&first, &trees.token_tree(MODEL)));
        assert!(!Arc::ptr_eq(&first, &trees.token_tree("another")));
        let words = trees.string_tree(MODEL);
        assert!(Arc::ptr_eq(&words, &trees.string_tree(MODEL)));
        assert!(!Arc::ptr_eq(&words, &trees.string_tree("another")));
        let stores = trees.stores.lock();
        assert_eq!(stores.len(), 1, "one size and one cadence");
        let counts = &stores[0].trees;
        assert_eq!(
            (counts.token_tree_count(), counts.string_tree_count()),
            (2, 2)
        );
    }

    /// A model named by the empty name is kept under the name the classic
    /// policy keeps it under, so the two names are one tree.
    #[test]
    fn the_empty_model_name_is_the_unknown_models_tree() {
        let trees = owner(document(1_000, 0));
        let unnamed = trees.token_tree("");
        assert!(Arc::ptr_eq(&unnamed, &trees.token_tree(UNKNOWN_MODEL_ID)));
        let words = trees.string_tree("");
        assert!(Arc::ptr_eq(&words, &trees.string_tree(UNKNOWN_MODEL_ID)));
        let stores = trees.stores.lock();
        let kept = stores[0].trees.token_tree(UNKNOWN_MODEL_ID);
        assert!(Arc::ptr_eq(&unnamed, &kept.expect("under the normal name")));
        assert!(stores[0].trees.token_tree("").is_none());
        let kept = stores[0].trees.string_tree(UNKNOWN_MODEL_ID);
        assert!(Arc::ptr_eq(&words, &kept.expect("under the normal name")));
        assert!(stores[0].trees.string_tree("").is_none());
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

    /// A text of 64 chars, as a prompt sent as text.
    const TEXT: &str = "a prompt of sixty-four chars, sent as text to the workers named.";

    /// One prompt of 64 tokens sent to each of `workers` in `model`'s token
    /// tree, and one text of 64 chars in its string tree.
    fn sent(trees: &LentTrees, model: &str, workers: &[&str]) {
        let tokens: Vec<u32> = (0..64).collect();
        let (tree, words) = (trees.token_tree(model), trees.string_tree(model));
        for worker in workers {
            tree.insert_tokens(&tokens, worker);
            words.insert_text(TEXT, worker);
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
        assert!(holds(&trees, MODEL, &all) && holds(&trees, "another", &all));
        let left = ["http://a:1", "http://b:1"];
        until("the purge behind a peer's tombstone", || {
            holds(&trees, MODEL, &left) && holds(&trees, "another", &left)
        })
        .await;

        registry.remove_by_url("http://b:1").expect("the removal");
        until("the purge behind a removal of this process's", || {
            holds(&trees, MODEL, &["http://a:1"]) && holds(&trees, "another", &["http://a:1"])
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
            holds(&trees, MODEL, &["http://a:1"])
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
        // The string tree is emptied by the same purge, behind the token
        // tree: each worker that stands is listed at no chars.
        let words = trees.string_tree(MODEL);
        let listed = || {
            let mut listed: Vec<(String, usize)> =
                words.get_tenant_char_count().into_iter().collect();
            listed.sort();
            listed
        };
        until("the string trees emptied behind unheard events", || {
            listed() == debited
        })
        .await;
        let texts = words.iter_entries().filter(|(text, _)| !text.is_empty());
        assert_eq!(texts.count(), 0, "no text is left");
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
    /// names. The same pass holds a model's string tree to its entry's size,
    /// in the string tree's own unit. A node counts 1,024 beside its chars
    /// there, so the string trees stand under a document of their own, whose
    /// two sizes lie some nodes apart. The settings' interval of zero runs
    /// no pass.
    #[test]
    fn the_size_pass_holds_a_tree_to_its_index_entrys_size() {
        let mut stated = document(128, 1);
        let roomier = entry(256, 1);
        stated.index.insert("roomy*".to_owned(), roomier);
        let trees = owner(stated);
        let (small, large) = (filled(&trees, MODEL), filled(&trees, "roomy-a"));
        let mut worded_by = document(3_000, 1);
        worded_by.index.insert("roomy*".to_owned(), entry(6_000, 1));
        let texts = owner(worded_by);
        let (few, more) = (worded(&texts, MODEL), worded(&texts, "roomy-a"));
        let until = Instant::now() + Duration::from_secs(20);
        let over = || {
            small.total_token_size() > 128
                || large.total_token_size() > 256
                || few.total_char_size() > 3_000
                || more.total_char_size() > 6_000
        };
        while over() && Instant::now() < until {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(small.total_token_size() <= 128, "the pass ran on its timer");
        // The other entry's pass ran too, and cut its tree to its own size.
        let kept = large.total_token_size();
        assert!(kept <= 256 && kept > 128, "its own entry's size: {kept}");
        // Each string tree is held to its own entry's size: as many nodes
        // as fit under it, and no node fewer.
        let (held, roomy) = (few.total_char_size(), more.total_char_size());
        let fits = |chars: usize, size: usize| chars <= size && chars + TEXT_NODE > size;
        assert!(fits(held, 3_000), "the settings' size: {held}");
        assert!(fits(roomy, 6_000), "its own entry's size: {roomy}");

        let unheld = owner(document(128, 0));
        unheld.token_tree(MODEL);
        let stores = unheld.stores.lock();
        assert!(
            stores[0]._size_pass.is_none(),
            "no pass at an interval of zero"
        );
    }

    /// A size under the largest beside the longest cadence is held by no
    /// pass, and one line at warn says so when the first token tree of that
    /// size is lent: a string tree lent before it writes none. The line
    /// names every `index` entry of the tree kind that states the two
    /// numbers and no other: an entry of the hash kind is lent no tree.
    /// The largest size beside that cadence asks for no pass, and writes
    /// none.
    #[test]
    #[tracing_test::traced_test]
    fn a_size_no_pass_holds_is_said_when_its_trees_are_lent() {
        const LINE: &str = "are held to no size";
        let lender = |max_size: u64| {
            let mut stated = document(128, 1);
            let unheld = entry(max_size, u32::MAX);
            stated.index.insert("*".to_owned(), unheld.clone());
            stated.index.insert("roomy*".to_owned(), unheld);
            stated.index.insert("held*".to_owned(), entry(max_size, 1));
            let hashed = IndexSpec {
                kind: IndexKind::Hash,
                ..entry(max_size, u32::MAX)
            };
            stated.index.insert("hashed*".to_owned(), hashed);
            owner(stated)
        };
        lender(u64::MAX).token_tree(MODEL);
        assert!(
            !logs_contain(LINE),
            "the largest size leaves nothing to say"
        );

        let trees = lender(4_096);
        trees.string_tree(MODEL);
        assert!(!logs_contain(LINE), "a string tree takes no line");
        trees.token_tree(MODEL);
        assert!(logs_contain(LINE), "the first token tree takes the line");
        trees.token_tree("another");
        trees.token_tree("roomy-a");
        logs_assert(|lines: &[&str]| {
            let said: Vec<_> = lines.iter().filter(|line| line.contains(LINE)).collect();
            let warns = |line: &&&str| {
                line.contains("WARN")
                    && line.contains(r#"index=["*", "roomy*"] max_size=4096 sweep_s=4294967295"#)
            };
            match said.as_slice() {
                [line] if warns(line) => Ok(()),
                _ => Err(format!("one line at warn for the four trees: {said:?}")),
            }
        });
    }

    /// The engine is lent the two trees of a single request's model, the
    /// owner's own, and none for a pair's side or an encode request. Where
    /// no owner stands it is lent none.
    #[test]
    fn the_engine_is_lent_a_single_requests_tree_and_no_sides() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let lent = cell(Some(Arc::clone(&trees)));
        let caches = EngineCaches::of(None, lent, false).expect("the trees");
        let lent = caches.token_tree(MODEL, Phase::Single).expect("a tree");
        assert!(Arc::ptr_eq(&lent, &trees.token_tree(MODEL)));
        let lent = caches.string_tree(MODEL, Phase::Single).expect("a tree");
        assert!(Arc::ptr_eq(&lent, &trees.string_tree(MODEL)));
        for side in [Phase::Prefill, Phase::Decode, Phase::Encode] {
            assert!(caches.token_tree(MODEL, side).is_none(), "{side:?}");
            assert!(caches.string_tree(MODEL, side).is_none(), "{side:?}");
        }
        // No monitor stands behind this object: no index and no block size.
        assert!(caches.index(MODEL).is_none() && caches.block_size(MODEL).is_none());

        let monitor = Arc::new(KvEventMonitor::new());
        let caches = EngineCaches::of(Some(monitor), cell(None), false).expect("the monitor");
        assert!(caches.token_tree(MODEL, Phase::Single).is_none());
        assert!(caches.string_tree(MODEL, Phase::Single).is_none());
        assert!(EngineCaches::of(None, cell(None), false).is_none());
    }

    /// The engine's object reads the trees through their cell: none while
    /// the cell is empty, and the owner's once a document put in force
    /// filled it. Over an empty cell the object is made only where a later
    /// document can fill it. A document that reads no sent prefix makes no
    /// owner, and the first that reads them makes it. An owner that stands
    /// follows the next document and answers the one before.
    #[test]
    fn the_engines_object_reads_the_trees_a_later_document_lends() {
        let lent = cell(None);
        assert!(EngineCaches::of(None, Arc::clone(&lent), false).is_none());
        let caches = EngineCaches::of(None, Arc::clone(&lent), true).expect("a later document");
        assert!(caches.token_tree(MODEL, Phase::Single).is_none());
        assert!(caches.string_tree(MODEL, Phase::Single).is_none());

        let workers = WorkerRegistry::new();
        let first = document(1_000, 0);
        assert!(LentTrees::under(&lent, first.clone(), false, 16, &workers).is_none());
        assert!(lent.get().is_none(), "a document that reads none");
        assert!(LentTrees::under(&lent, first.clone(), true, 16, &workers).is_none());
        let trees = Arc::clone(lent.get().expect("the first document that reads them"));
        let asked = caches.token_tree(MODEL, Phase::Single).expect("a tree");
        assert!(Arc::ptr_eq(&asked, &trees.token_tree(MODEL)));
        let asked = caches.string_tree(MODEL, Phase::Single).expect("a tree");
        assert!(Arc::ptr_eq(&asked, &trees.string_tree(MODEL)));

        let mut second = first.clone();
        second.generation = first.generation + 1;
        let before = LentTrees::under(&lent, second.clone(), false, 16, &workers);
        assert_eq!(before, Some(first), "the document followed before");
        assert!(Arc::ptr_eq(lent.get().expect("one owner"), &trees));
        assert_eq!(trees.follows(), second.generation);
    }

    /// The owner sizes a model's trees by the model's `index` entry of the
    /// document in force at the ask. Behind a document that follows, a model
    /// whose entry stands as it stood is handed the tree it had. A model
    /// the new document writes an entry of its own for is handed the tree
    /// of that entry's store.
    #[test]
    fn a_models_trees_are_sized_by_the_document_in_force_at_the_ask() {
        let start = document(128, 1);
        let trees = owner(start.clone());
        let (plain, labelled) = (trees.token_tree("plain"), trees.token_tree(MODEL));
        let mut later = start.clone();
        later.index.insert(MODEL.to_owned(), entry(64, 2));
        assert_eq!(trees.follow(later), start, "the document followed before");
        assert!(
            Arc::ptr_eq(&trees.token_tree("plain"), &plain),
            "its entry stands"
        );
        assert!(
            !Arc::ptr_eq(&trees.token_tree(MODEL), &labelled),
            "another store"
        );
        let stores = trees.stores.lock();
        let numbers: Vec<(u64, u32)> = stores
            .iter()
            .map(|store| (store.max_size, store.sweep_s))
            .collect();
        assert_eq!(numbers, [(128, 1), (64, 2)]);
    }

    /// The object answers a model's index and its block size as the
    /// cache-event monitor behind it does: the monitor's own index, which
    /// says it holds no block until an event stores one.
    #[test]
    fn the_monitors_index_and_block_size_are_read_through_the_one_object() {
        let monitor = Arc::new(KvEventMonitor::new());
        let index = Arc::new(KvIndex::chain());
        monitor.set_index(MODEL, Arc::clone(&index));
        monitor.set_block_size(MODEL, 4);
        let trees = Arc::new(owner(document(1_000, 0)));
        let caches = EngineCaches::of(Some(monitor), cell(Some(trees)), false).expect("both");
        assert_eq!(caches.block_size(MODEL), Some(4));
        let empty = caches.index(MODEL).expect("the model's index");
        assert!(!empty.holds_a_block(), "no block yet");
        let worker = index.intern_worker("grpc://w1:9000").expect("a number");
        let block = StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: ContentHash(101),
        };
        let stored = index.apply_stored(worker, &[block], None, &mut WorkerBlocks::default());
        stored.expect("a block stored");
        let lent = caches.index(MODEL).expect("the model's index");
        assert!(std::ptr::addr_eq(Arc::as_ptr(&lent), Arc::as_ptr(&index)));
        assert!(caches.index("another").is_none());
        assert_eq!(caches.block_size("another"), None);
    }
}
