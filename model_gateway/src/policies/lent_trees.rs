//! The trees a gateway lends Grand Central: a model's token tree, and
//! its string tree for the prompts sent as text.
//!
//! Where Grand Central places every request no classic policy object is
//! made, so no policy keeps the trees of the prompts this process sent. The
//! engine's start makes an owner of them in a policy's place
//! ([`LentTrees`]), and only where the document in force reads the prefixes
//! sent to each worker. The owner keeps the trees in the store type the
//! classic policy keeps its own in, under the document's own numbers: its
//! block size is a token tree's page, and a model's `index` entry states
//! the size its trees are held to and how often. The one size holds a token
//! tree in tokens and a string tree in its own unit, its chars and a weight
//! for each node a worker stands on, as the classic policy's one setting
//! does, and the classic policy's own pass holds a tree to it. A
//! worker the registry lost is taken out of the trees
//! by the owner's own task, which follows the registry's events
//! ([`LentTrees::follow_removals`]): the registry sends one for every
//! removal, so a worker a mesh peer's word took away leaves the trees as
//! one this process removed does. The engine calls a tree's match and
//! nothing else.
//!
//! The engine is handed one object for what this process knows of its
//! workers' caches ([`EngineCaches`]): the cache-event monitor's indexes and
//! these trees, each to a single request's decision alone.

use std::{
    collections::HashSet,
    sync::{Arc, Weak},
    time::Duration,
};

use kv_index::{ModelTrees, TenantId, TokenTree, Tree as StringTree};
use openai_protocol::worker::WorkerStatus;
use parking_lot::Mutex;
use smg_grand_central::{
    document::{IndexSpec, PolicySpec},
    engine::{CacheIndex, CacheIndexes, Phase},
};
use smg_worker::registry::event::WorkerEvent;
use tokio::{
    runtime::Handle,
    sync::broadcast::error::{RecvError, TryRecvError},
    time::{timeout_at, Instant},
};
use tracing::{error, warn};

use super::{
    cache_aware::evict_trees, kv_index::event_monitor::KvEventMonitor, normalize_model_key,
    utils::PeriodicTask,
};
use crate::routers::common::snapshot::WorkerRegistry;

/// The settle window of a worker whose own window is zero: how long behind
/// its removal its address is owed a purge. The look behind events that
/// went unheard is owed as long behind them: the window of a worker whose
/// removal was among them is not known.
const SETTLE_WHERE_NONE_IS_STATED: Duration = Duration::from_secs(5);

/// The owner of the trees Grand Central is lent: a token tree and a
/// string tree a model.
#[derive(Debug)]
pub(crate) struct LentTrees {
    /// The document the engine runs: a model's `index` entry sizes its
    /// trees ([`PolicySpec::index_for`]).
    document: PolicySpec,
    /// A token tree's page in tokens: the document's block size.
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
    /// A store of trees, a token tree of `page` tokens a page, each held to
    /// `max_size` by the classic policy's pass every `sweep_s` seconds: a
    /// token tree to that many tokens, a string tree to that size in its
    /// own unit (its chars and a weight for each node). The document's
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
    /// cadence ([`Self::store_of`]). The engine asks when it builds a
    /// model's table, not for a request.
    pub(crate) fn token_tree(&self, model: &str) -> Arc<TokenTree> {
        self.store_of(model)
            .token_tree_or_new(normalize_model_key(model))
    }

    /// The string tree of `model`, for the prompts sent as text: made at
    /// its first ask, handed again at every later one, and kept in the
    /// store that keeps the model's token tree. The engine asks when it
    /// builds a model's table, whatever the model's index holds.
    pub(crate) fn string_tree(&self, model: &str) -> Arc<StringTree> {
        self.store_of(model)
            .string_tree_or_new(normalize_model_key(model))
    }

    /// The store of the trees of `model`'s size and cadence, made at the
    /// first ask for a tree of that size and cadence: the model's `index`
    /// entry of the document, or the entry's defaults where the document
    /// writes none for it.
    fn store_of(&self, model: &str) -> Arc<ModelTrees> {
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
        Arc::clone(&stores[at].trees)
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
    /// no event until it ends. A worker that took requests up to its
    /// removal is purged once more a settle window later
    /// ([`follow_removals`]).
    ///
    /// The task holds the trees weakly, and the registry weakly too, for
    /// the addresses it lists where a purge is owed across events that
    /// went unheard. It ends with the registry, at the first wake after the
    /// trees are gone, or with its runtime.
    pub(crate) fn follow_removals(
        self: &Arc<Self>,
        registry: &Arc<WorkerRegistry>,
        runtime: &Handle,
    ) {
        let listed = Arc::downgrade(registry);
        follow_removals(
            self,
            "the lent trees",
            move |trees: &Self, left: &Left| trees.forget_the_removed(left, &listed),
            registry,
            runtime,
        );
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
    /// that was heard tells the two apart: at such a wake the registry's
    /// list keeps no address. The trees fill again from the requests that
    /// follow.
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
    /// Every address stays listed at no tokens, but one a removal named
    /// and one owed a purge that the registry does not list: those are
    /// purged by name behind the pass.
    ///
    /// A purge owed behind a settle window stays owed across events that
    /// went unheard, and a registration under its address may be among
    /// them. When it is due it runs unless `workers` lists the address: a
    /// listed address is not purged by name, as what stands under it is
    /// the prompts of the worker that came. A wake with unheard events of
    /// its own empties every address all the same: the registry's list
    /// keeps an address from an owed purge alone. The list is read at a
    /// wake where such a purge is due alone, once, ahead of the first
    /// tree's walk and under no lock of the stores. A worker registered
    /// behind that read loses what was written under its address ahead of
    /// its tree's walk, and the requests that follow write it again.
    fn forget_the_removed(&self, left: &Left, workers: &Weak<WorkerRegistry>) {
        if left.unheard > 0 {
            warn!(
                "{} events of the worker registry went unheard: the lent trees are emptied, and cache_aware remembers no prompt until they fill again",
                left.unheard
            );
        }
        let mut owed: Vec<&str> = left.again_if_unlisted.iter().map(String::as_str).collect();
        // With the registry gone no worker stands under any address.
        if let Some(workers) = workers.upgrade().filter(|_| !owed.is_empty()) {
            let listed: HashSet<String> = workers.get_all_urls().into_iter().collect();
            owed.retain(|address| !listed.contains(*address));
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
                &owed,
                || tree.evict_tenant_by_size(0),
                |worker| tree.remove_tenant_all(worker),
            );
        }
        for tree in stores.iter().flat_map(|trees| trees.string_trees()) {
            left.purge(
                &owed,
                || tree.evict_tenant_by_size(0),
                |worker| tree.remove_tenant_all(worker),
            );
        }
    }
}

/// Starts the task that takes a worker the registry lost out of `stores`,
/// named `what` in its one log line, on `runtime`. It wakes on the
/// registry's events and hands `forget` what they ask of the stores
/// ([`Left`]), off the runtime's own threads, and reads no event until
/// `forget` ends.
///
/// A decision that read a worker ahead of its removal may write a prompt
/// under the worker's address behind the removal. A worker that was being
/// drained when it left took no new request for its settle window, so no
/// such decision is under way where that window is not zero. Any other
/// worker's address is owed a purge one settle window behind its removal,
/// the worker's own window or five seconds where that is zero, at an event
/// or with none. An address a worker was registered under meanwhile is
/// owed none: what stands under it is that worker's own.
///
/// A purge that is owed stays owed across events that went unheard. A
/// registration under its address may be among them, so when it is due
/// the store asks the registry, and what stands under an address the
/// registry lists stays. A removal may be among them too, and no event
/// that was heard names its address: `forget` is asked once more, five
/// seconds behind them ([`Left::looks_again`]), and a store that keeps
/// by the registry's list looks then for what a decision under way wrote
/// behind the purge of that wake.
///
/// The task holds `stores` weakly, and of the registry its events alone:
/// the handle a store asks the registry by is `forget`'s own, and weak
/// too. It ends with the registry, at the first wake after `stores` are
/// gone, or with its runtime.
pub(super) fn follow_removals<S: Send + Sync + 'static>(
    stores: &Arc<S>,
    what: &'static str,
    forget: impl Fn(&S, &Left) + Send + Sync + 'static,
    registry: &WorkerRegistry,
    runtime: &Handle,
) {
    let mut events = registry.subscribe_events();
    let stores = Arc::downgrade(stores);
    let forget = Arc::new(forget);
    runtime.spawn(async move {
        // The addresses owed a purge behind a settle window, each with the
        // instant it is due and whether events went unheard inside its
        // window.
        let mut owed: Vec<(Instant, String, bool)> = Vec::new();
        // The instant the look behind events that went unheard is due.
        let mut look: Option<Instant> = None;
        loop {
            let mut left = Left::default();
            let heard = match owed.iter().map(|(due, ..)| *due).chain(look).min() {
                Some(due) => timeout_at(due, events.recv()).await.ok(),
                None => Some(events.recv().await),
            };
            match heard {
                Some(Ok(event)) => left.note(&event),
                Some(Err(RecvError::Lagged(lost))) => left.unheard += lost,
                Some(Err(RecvError::Closed)) => return,
                // A purge behind a settle window is due, or the look
                // behind unheard events.
                None => {}
            }
            loop {
                match events.try_recv() {
                    Ok(event) => left.note(&event),
                    Err(TryRecvError::Lagged(lost)) => left.unheard += lost,
                    Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                }
            }
            let now = Instant::now();
            if left.unheard > 0 {
                // This wake's purge answers the unheard events, and what is
                // written behind it is looked for a window from now. A
                // look that was owed already is answered by it too.
                look = now.checked_add(SETTLE_WHERE_NONE_IS_STATED);
                // A registration may be among them, under an address that
                // is owed a purge.
                for (_, _, unheard) in &mut owed {
                    *unheard = true;
                }
            } else if look.is_some_and(|due| due <= now) {
                left.looks_again = true;
                look = None;
            }
            owed.retain(|(due, address, unheard)| {
                if left.listed.contains(address) {
                    return false;
                }
                if *due <= now {
                    let due_now = if *unheard {
                        &mut left.again_if_unlisted
                    } else {
                        &mut left.again
                    };
                    due_now.push(address.clone());
                }
                *due > now
            });
            // A window that ends past what the clock can say never ends: its
            // address is owed nothing.
            // A removal heard in a wake with unheard events is owed its
            // purge the same way.
            let unheard = left.unheard > 0;
            owed.extend(left.settles.drain(..).filter_map(|(address, settle)| {
                Some((now.checked_add(settle)?, address, unheard))
            }));
            let Some(stores) = stores.upgrade() else {
                return;
            };
            if !left.asks_for_a_purge() {
                continue;
            }
            let forget = Arc::clone(&forget);
            let purge = tokio::task::spawn_blocking(move || forget(&stores, &left));
            // A purge that panicked is this task's last: nothing but a
            // store's own size pass bounds it from here on.
            if let Err(ended) = purge.await {
                if ended.is_panic() {
                    error!("A purge of {what} panicked: no worker that leaves is taken out of them from here on");
                }
                return;
            }
        }
    });
}

/// What the registry's events of one wake ask of the stores.
#[derive(Debug, Default)]
pub(super) struct Left {
    /// The addresses of the workers a removal took away.
    addresses: Vec<String>,
    /// Of those, the ones whose worker was being drained when it left, but
    /// those a worker was registered under behind the removal.
    drained: Vec<String>,
    /// Of those, the ones whose worker took requests up to its removal,
    /// each with its settle window: owed a purge behind it.
    settles: Vec<(String, Duration)>,
    /// The addresses a worker was registered under.
    listed: Vec<String>,
    /// The addresses whose purge behind a settle window is due, every
    /// event inside the window heard.
    again: Vec<String>,
    /// The addresses whose purge behind a settle window is due, where
    /// events went unheard inside the window: a registration under one may
    /// be among them.
    again_if_unlisted: Vec<String>,
    /// How many events went unheard: a removal may be among them.
    unheard: u64,
    /// Whether the look behind events that went unheard is due.
    looks_again: bool,
}

impl Left {
    /// Notes `event`: a removal names the address that left, whether its
    /// worker was being drained, and its settle window where the worker
    /// took requests up to its removal: one that was not being drained, or
    /// was at a window of zero. A registration and a replacement take no
    /// worker away: they name an address a worker stands under again,
    /// which is owed no purge. A status change says nothing.
    fn note(&mut self, event: &WorkerEvent) {
        match event {
            WorkerEvent::Removed { worker, .. } => {
                let address = worker.url().to_owned();
                let drained = worker.status() == WorkerStatus::Draining;
                let window = worker.metadata().health_config.drain_settle_secs;
                if !drained || window == 0 {
                    let settle = match window {
                        0 => SETTLE_WHERE_NONE_IS_STATED,
                        secs => Duration::from_secs(secs),
                    };
                    self.settles.push((address.clone(), settle));
                }
                if drained {
                    self.drained.push(address.clone());
                }
                self.addresses.push(address);
            }
            WorkerEvent::Registered { worker, .. } | WorkerEvent::Replaced { new: worker, .. } => {
                self.drained.retain(|address| address != worker.url());
                self.settles.retain(|(address, _)| address != worker.url());
                self.listed.push(worker.url().to_owned());
            }
            WorkerEvent::StatusChanged { .. } => {}
        }
    }

    /// The addresses to purge at this wake in a store where a worker that
    /// rejoins under its address starts with no prompt: those a removal
    /// named, then those whose purge behind a settle window is due, every
    /// event inside the window heard. Where events went unheard inside it,
    /// a worker may stand under the address by one of them: this list
    /// does not name it, and such a store asks the registry for it
    /// ([`LentTrees::forget_the_removed`]).
    pub(super) fn purged(&self) -> impl Iterator<Item = &str> {
        self.addresses.iter().chain(&self.again).map(String::as_str)
    }

    /// The addresses to purge at this wake in a store where an address a
    /// worker is registered under again keeps its prompts: those whose
    /// worker was being drained when it left and that no worker was
    /// registered under behind the removal, then those whose purge behind
    /// a settle window is due, whether or not events went unheard inside
    /// the window. A registration that went unheard is not told by either:
    /// such a store asks the registry for the addresses it lists.
    pub(super) fn settled(&self) -> impl Iterator<Item = &str> {
        self.drained
            .iter()
            .chain(&self.again)
            .chain(&self.again_if_unlisted)
            .map(String::as_str)
    }

    /// How many events went unheard.
    pub(super) fn unheard(&self) -> u64 {
        self.unheard
    }

    /// Whether this wake is the look behind events that went unheard, five
    /// seconds behind them: a decision that read a worker whose removal
    /// was among them may have written under its address behind the purge
    /// of their wake, and no event that was heard names that address.
    pub(super) fn looks_again(&self) -> bool {
        self.looks_again
    }

    /// Whether the events ask for a purge: a removal was heard, a purge
    /// behind a settle window is due, events went unheard, or the look
    /// behind them is due.
    fn asks_for_a_purge(&self) -> bool {
        self.unheard > 0
            || self.looks_again
            || !self.addresses.is_empty()
            || !self.again.is_empty()
            || !self.again_if_unlisted.is_empty()
    }

    /// The purge of one tree, of either kind: `empty` is the tree's size
    /// pass at no size, and `forget` its purge of one worker by name,
    /// asked for every address [`Self::purged`] names and for each of
    /// `owed`: the addresses whose purge is due across events that went
    /// unheard and that the registry does not list.
    fn purge(&self, owed: &[&str], empty: impl FnOnce(), forget: impl Fn(&TenantId)) {
        if self.unheard > 0 {
            // A tree's own `clear` asks for no writer beside it. This
            // call may run beside the size pass's own: the second of
            // the two to take a worker out of a node debits nothing.
            empty();
        }
        // Behind an eviction a named address's count reads zero, and
        // the purge drops its entries alone.
        for address in self.purged().chain(owed.iter().copied()) {
            forget(&TenantId::from(address));
        }
    }
}

/// What a gateway answers Grand Central of its workers' caches, as one
/// object: the indexes of the cache-event monitor, and the trees it lends.
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

    /// The string tree of a single request's model, for a request this
    /// gateway holds as text. A pair's sides are lent none, as they are
    /// lent no token tree.
    fn string_tree(&self, model: &str, phase: Phase) -> Option<Arc<StringTree>> {
        match phase {
            Phase::Single => Some(self.trees.as_ref()?.string_tree(model)),
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

    /// An `index` entry of `max_size` tokens trimmed every `sweep_s`
    /// seconds.
    fn entry(max_size: u64, sweep_s: u32) -> IndexSpec {
        IndexSpec {
            max_size,
            sweep_s,
            ..IndexSpec::default()
        }
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
        let look = Left {
            looks_again: true,
            ..Left::default()
        };
        assert!(look.asks_for_a_purge(), "the look behind them");
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
        let registry = Arc::new(WorkerRegistry::new());
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

    /// A worker that left with no drain is purged once more a settle window
    /// behind its removal: a prompt a decision under way wrote under its
    /// address behind the first purge leaves too.
    #[tokio::test(start_paused = true)]
    async fn a_prompt_written_behind_the_purge_leaves_a_settle_window_later() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let registry = Arc::new(WorkerRegistry::new());
        worker_at(&registry, "http://a:1");
        worker_at(&registry, "http://b:1");
        sent(&trees, MODEL, &["http://a:1", "http://b:1"]);
        trees.follow_removals(&registry, &Handle::current());

        registry.remove_by_url("http://b:1").expect("the removal");
        until("the purge behind the removal", || {
            holds(&trees, MODEL, &["http://a:1"])
        })
        .await;
        sent(&trees, MODEL, &["http://b:1"]);
        until("the purge a settle window later", || {
            holds(&trees, MODEL, &["http://a:1"])
        })
        .await;
    }

    /// A purge owed across events that went unheard, up to the instant the
    /// trees stand emptied behind them. The removal of `http://b:1` is
    /// heard and purged. More events than the registry's channel holds
    /// follow, the task never run between them: where `rejoins`, a
    /// registration under that address is the first of them. The registry
    /// is handed back, as the task ends with it.
    async fn a_purge_owed_across_unheard_events(
        rejoins: bool,
    ) -> (Arc<LentTrees>, Arc<WorkerRegistry>) {
        let trees = Arc::new(owner(document(1_000, 0)));
        let registry = Arc::new(WorkerRegistry::new());
        worker_at(&registry, "http://a:1");
        worker_at(&registry, "http://b:1");
        sent(&trees, MODEL, &["http://a:1", "http://b:1"]);
        trees.follow_removals(&registry, &Handle::current());

        registry.remove_by_url("http://b:1").expect("the removal");
        until("the purge behind the removal", || {
            holds(&trees, MODEL, &["http://a:1"])
        })
        .await;

        let mut witness = registry.subscribe_events();
        if rejoins {
            worker_at(&registry, "http://b:1");
        }
        for wave in 0..1_500 {
            worker_at(&registry, &format!("http://wave-{wave}:1"));
        }
        assert!(
            matches!(witness.try_recv(), Err(TryRecvError::Lagged(_))),
            "the burst must overflow the channel, or this case holds nothing"
        );
        until("the trees emptied behind unheard events", || {
            holds(&trees, MODEL, &[])
        })
        .await;
        (trees, registry)
    }

    /// A purge that is owed runs across events that went unheard inside
    /// its window, where the registry lists no worker under the address. A
    /// prompt a decision under way writes under the address behind the
    /// emptying leaves each tree when the purge is due.
    #[tokio::test(start_paused = true)]
    async fn a_purge_owed_across_unheard_events_runs_where_no_worker_stands() {
        let (trees, _registry) = a_purge_owed_across_unheard_events(false).await;
        sent(&trees, MODEL, &["http://b:1"]);
        until("the purge a settle window behind the removal", || {
            holds(&trees, MODEL, &[])
        })
        .await;
    }

    /// A purge that is owed is let go where a worker stands under the
    /// address by one of the events that went unheard inside its window:
    /// the registry lists it. A prompt sent to the worker that came stands
    /// past the instant the purge was due, and past the look.
    #[tokio::test(start_paused = true)]
    async fn a_purge_owed_across_unheard_events_is_let_go_where_a_worker_stands() {
        let (trees, _registry) = a_purge_owed_across_unheard_events(true).await;
        sent(&trees, MODEL, &["http://b:1"]);
        tokio::time::sleep(SETTLE_WHERE_NONE_IS_STATED * 2 + Duration::from_secs(1)).await;
        assert!(
            holds(&trees, MODEL, &["http://b:1"]),
            "the worker that came keeps its prompt"
        );
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
        let registry = Arc::new(WorkerRegistry::new());
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
                trees.forget_the_removed(&unheard, &Weak::new());
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
        let registry = Arc::new(WorkerRegistry::new());
        let trees = Arc::new(owner(document(1_000, 0)));
        trees.follow_removals(&registry, &Handle::current());
        let alive = (tasks(), Arc::weak_count(&trees));
        assert_eq!(alive, (1, 1), "the task, and its own handle on the trees");
        drop(registry);
        until("its end with the registry", || tasks() == 0).await;
        assert_eq!(Arc::weak_count(&trees), 0);

        let registry = Arc::new(WorkerRegistry::new());
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

    /// The engine is lent the two trees of a single request's model, the
    /// owner's own, and none for a pair's side or an encode request. Where
    /// no owner stands it is lent none.
    #[test]
    fn the_engine_is_lent_a_single_requests_tree_and_no_sides() {
        let trees = Arc::new(owner(document(1_000, 0)));
        let caches = EngineCaches::of(None, Some(Arc::clone(&trees))).expect("the trees");
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
        let caches = EngineCaches::of(Some(monitor), None).expect("the monitor");
        assert!(caches.token_tree(MODEL, Phase::Single).is_none());
        assert!(caches.string_tree(MODEL, Phase::Single).is_none());
        assert!(EngineCaches::of(None, None).is_none());
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
        let caches = EngineCaches::of(Some(monitor), Some(trees)).expect("both");
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
