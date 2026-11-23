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
//! tree to that size, and a worker that left is taken out of the trees. The
//! engine calls a tree's match and nothing else.
//!
//! The engine is handed one object for what this process knows of its
//! workers' caches ([`EngineCaches`]): the cache-event monitor's indexes and
//! these trees.

use std::{collections::HashSet, sync::Arc};

use kv_index::{ModelTrees, TenantId, TokenTree};
use parking_lot::Mutex;
use smg_policy_controller::{
    document::{IndexSpec, PolicySpec},
    engine::{CacheIndex, CacheIndexes, Phase},
};

use super::{
    cache_aware::evict_trees, kv_index::event_monitor::KvEventMonitor, normalize_model_key,
    utils::PeriodicTask,
};

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

    /// Takes the workers at `worker_urls` out of `model_id`'s tree. A
    /// worker that left is never written again, so the size pass alone
    /// would keep its prompts for good.
    pub(crate) fn remove_workers_from_model(&self, model_id: &str, worker_urls: &HashSet<String>) {
        let tenants: Vec<TenantId> = worker_urls
            .iter()
            .map(|url| TenantId::from(url.as_str()))
            .collect();
        let model = normalize_model_key(model_id);
        // The stores are copied out and the lock is let go ahead of the
        // purge: a purge walks a tree once for every worker, and a decision
        // that builds a model's table asks for its tree under this lock.
        let stores: Vec<Arc<ModelTrees>> = self
            .stores
            .lock()
            .iter()
            .map(|store| Arc::clone(&store.trees))
            .collect();
        // A model's tree stands in one store, and the others hold none of
        // its name.
        for trees in stores {
            trees.remove_tenants_from_model(model, &tenants);
        }
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
        thread,
        time::{Duration, Instant},
    };

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

    /// A worker that left is taken out of its model's tree, and out of no
    /// other model's, whichever sizes the two models' entries state.
    #[test]
    fn a_worker_that_left_is_taken_out_of_its_models_tree() {
        let mut stated = document(1_000, 0);
        let sized = entry(4_096, u32::MAX);
        stated.index.insert("another".to_owned(), sized);
        let trees = owner(stated);
        let tokens: Vec<u32> = (0..64).collect();
        for model in [MODEL, "another"] {
            let tree = trees.token_tree(model);
            tree.insert_tokens(&tokens, "http://a:1");
            tree.insert_tokens(&tokens, "http://b:1");
        }
        assert_eq!(trees.stores.lock().len(), 2, "two sizes, two stores");
        let left = HashSet::from(["http://a:1".to_owned()]);
        trees.remove_workers_from_model("another", &left);
        assert_eq!(held(&trees, "another"), ["http://b:1"]);
        assert_eq!(held(&trees, MODEL), ["http://a:1", "http://b:1"]);
        trees.remove_workers_from_model(MODEL, &left);
        assert_eq!(held(&trees, MODEL), ["http://b:1"]);
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
