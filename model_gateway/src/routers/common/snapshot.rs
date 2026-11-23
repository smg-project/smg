//! The gateway's value of a membership generation ([`Generation`]): the
//! members, and the cells that keep what the routers derive from them (the
//! routing pools, the prefill and decode pairs, the window floors, the
//! labels the workers say together, the hash ring). The registry builds
//! one at every publish; the routing readers hand it to a request.

use std::{
    ops::Deref,
    sync::{Arc, OnceLock},
};

use openai_protocol::{worker::TagVocabulary, UNKNOWN_MODEL_ID};
use smg_policy_controller::engine::{LentList, ListCell};
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    load,
    pd::pairing::PdPairingMode,
    registry::{self, Generation},
};
use symphony::grammar::GrammarStart;

use crate::{
    policies::hash_ring::HashRing,
    routers::{
        common::pair_index::{PdPairIndex, PdWire},
        grpc::multimodal::MediaFacts,
    },
};

/// Static routing pool inside a model snapshot.
///
/// Availability is deliberately not encoded here: health, circuit-breaker and
/// overload state change independently of membership and remain lock-free on
/// each worker. Policies validate that live state while selecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingPool {
    HttpRegular,
    GrpcPipelineRegular,
    /// gRPC workers, and direct-ZMQ engines whose wire carries the
    /// KV-transfer rendezvous, unlike [`Self::GrpcPipelineRegular`]'s every
    /// ZMQ worker: the other ZMQ wires carry none, so such a worker must
    /// never join a PD pool even though it rides the gRPC pipeline elsewhere.
    GrpcPrefill,
    /// The members [`Self::GrpcPrefill`] admits.
    GrpcDecode,
    /// The members [`Self::GrpcPrefill`] admits: a direct-ZMQ encode server
    /// takes its encode request on the ZMQ lane; a gRPC one's encode
    /// dispatch is a gRPC encoder RPC.
    GrpcEncode,
    /// HTTP-transport prefill pool for the HTTP PD router: it proxies plain
    /// HTTP to the selected worker's URL, so a gRPC or ZMQ worker must never
    /// be selectable (the same rule `select_first_worker` follows).
    HttpPrefill,
    /// HTTP-transport decode pool (same contract as [`Self::HttpPrefill`]).
    HttpDecode,
    /// Workers that front a third-party provider, whatever their type.
    External,
}

type WorkerSnapshot = Arc<[Arc<dyn Worker>]>;
type LazyRoutingPool = OnceLock<WorkerSnapshot>;

impl RoutingPool {
    const COUNT: usize = 8;

    const fn index(self) -> usize {
        match self {
            Self::HttpRegular => 0,
            Self::GrpcPipelineRegular => 1,
            Self::GrpcPrefill => 2,
            Self::GrpcDecode => 3,
            Self::GrpcEncode => 4,
            Self::HttpPrefill => 5,
            Self::HttpDecode => 6,
            Self::External => 7,
        }
    }

    /// Whether `worker` is a member of this pool: the one rule of
    /// membership. The snapshot projects its pools by it.
    pub(crate) fn matches(self, worker: &dyn Worker) -> bool {
        match self {
            Self::HttpRegular => {
                *worker.worker_type() == WorkerType::Regular
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::GrpcPipelineRegular => {
                *worker.worker_type() == WorkerType::Regular
                    && worker.connection_mode().uses_grpc_pipeline()
            }
            Self::GrpcPrefill => {
                *worker.worker_type() == WorkerType::Prefill && serves_disaggregated_role(worker)
            }
            Self::GrpcDecode => {
                *worker.worker_type() == WorkerType::Decode && serves_disaggregated_role(worker)
            }
            Self::GrpcEncode => {
                *worker.worker_type() == WorkerType::Encode && serves_disaggregated_role(worker)
            }
            Self::HttpPrefill => {
                *worker.worker_type() == WorkerType::Prefill
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::HttpDecode => {
                *worker.worker_type() == WorkerType::Decode
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::External => worker.metadata().spec.runtime_type == RuntimeType::External,
        }
    }
}

/// Whether a worker of a disaggregated role rides the gRPC pipeline's
/// encode/prefill/decode dispatch: a gRPC worker, or a direct-ZMQ
/// engine whose wire carries the rendezvous and the encode request (the
/// other ZMQ wires carry neither, and registration refuses them a role).
fn serves_disaggregated_role(worker: &dyn Worker) -> bool {
    match worker.connection_mode() {
        ConnectionMode::Grpc => true,
        ConnectionMode::Zmq => worker.metadata().spec.runtime_type == RuntimeType::TokenSpeed,
        ConnectionMode::Http => false,
    }
}

/// The narrowest known context windows of a pool for one model id, as a
/// membership snapshot keeps them ([`ModelWorkerSnapshot::window_floor`]):
/// `budget` over the workers that hold a request's completion budget,
/// `prefill` over the prefill workers, which are held to the input alone.
/// `None` where no worker of the class is bound by a window.
///
/// The figure depends on the pool and on the model id as spelled, never on
/// a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WindowFloor {
    pub(crate) budget: Option<u32>,
    pub(crate) prefill: Option<u32>,
}

/// What a model's workers say together to a request, before one of them is
/// selected: the parser names their cards set for the model, where their
/// engines start a grammar, and the structural-tag vocabulary their grammar
/// compilers take. The stage that prepares a request's reading holds the
/// rule that reads each from the workers; a snapshot keeps the answers
/// ([`ModelWorkerSnapshot::labels`]).
///
/// The answers depend on the membership and on the model id as spelled,
/// never on a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModelLabels {
    /// The tool-call parser the cards name, when one does.
    pub(crate) tool_parser: Option<String>,
    /// The reasoning parser the cards name, when one does.
    pub(crate) reasoning_parser: Option<String>,
    /// Where the workers' engines start a grammar, taken together.
    pub(crate) grammar_start: GrammarStart,
    /// The structural-tag vocabulary every worker's compiler takes.
    pub(crate) vocabulary: TagVocabulary,
}

/// Immutable membership snapshot. Route-specific projections are populated
/// lazily once and then shared by every request until membership changes
/// publish a replacement snapshot.
///
/// Multi-role selection (PD/EPD) must derive every role from ONE snapshot: two
/// separate registry lookups can straddle a concurrent membership change and
/// pair workers that never coexisted.
///
/// A snapshot stands on cache lines of its own, apart from the count of its
/// handles. Every request takes a handle of its model's snapshot and drops
/// it, so the count's line is one every caller writes, and a selection
/// borrows its pool from the snapshot: a field beside the count would be
/// read from that line all through the selection.
#[derive(Debug)]
#[repr(align(128))]
pub struct ModelWorkerSnapshot {
    all: WorkerSnapshot,
    pools: [LazyRoutingPool; RoutingPool::COUNT],
    /// The cell beside each pool's list, empty until the policy engine is
    /// asked over that list (see [`Self::lent`]). A gateway that starts no
    /// engine leaves every cell empty.
    cells: [ListCell; RoutingPool::COUNT],
    /// The compatible prefill/decode pairs per wire and pairing mode, built
    /// once from the pools above (see [`PdPairIndex`]).
    pd_pairs: [[OnceLock<Arc<PdPairIndex>>; PdPairingMode::COUNT]; PdWire::COUNT],
    /// Whether a worker of this generation serves a Harmony model, kept from
    /// the first request's reading (see [`Self::harmony`]).
    harmony: OnceLock<bool>,
    /// The pools' window floors, each kept with the model id it was read
    /// for (see [`Self::window_floor`]).
    window_floors: [OnceLock<(Box<str>, WindowFloor)>; RoutingPool::COUNT],
    /// What the workers of this generation say together, kept with the
    /// model id it was read for (see [`Self::labels`]).
    labels: OnceLock<(Box<str>, ModelLabels)>,
    /// What the media gates read from the workers of this generation
    /// together (see [`Self::media`]). `MediaFacts` stays with the gates in
    /// `routers::grpc::multimodal`, so this module depends on that one.
    media: OnceLock<MediaFacts>,
    /// The consistent-hash ring over the URLs of this generation's members
    /// (see [`Self::hash_ring`]).
    hash_ring: OnceLock<Arc<HashRing>>,
}

impl ModelWorkerSnapshot {
    fn new(all: WorkerSnapshot) -> Self {
        Self {
            all,
            pools: std::array::from_fn(|_| OnceLock::new()),
            cells: std::array::from_fn(|_| ListCell::default()),
            pd_pairs: std::array::from_fn(|_| std::array::from_fn(|_| OnceLock::new())),
            harmony: OnceLock::new(),
            window_floors: std::array::from_fn(|_| OnceLock::new()),
            labels: OnceLock::new(),
            media: OnceLock::new(),
            hash_ring: OnceLock::new(),
        }
    }

    /// Whether a worker of this membership generation serves a Harmony
    /// model: `decide` reads the snapshot's workers on the first request
    /// after the membership change, every request after it reads the kept
    /// answer. A registration, a removal and a property rebuild each publish
    /// a new snapshot, and so does a model refresh that changes a worker's
    /// cards in place ([`WorkerRegistry::refresh_models`]), so the
    /// per-worker card scans run once, not per request. The rule itself
    /// stays with the router's detector.
    pub(crate) fn harmony(&self, decide: impl FnOnce(&[Arc<dyn Worker>]) -> bool) -> bool {
        *self.harmony.get_or_init(|| decide(&self.all))
    }

    /// The window floor of `pool` for `model_id`: `read` reads the pool's
    /// workers on the first request after the membership change, every
    /// request after it reads the kept figure. Whatever changes a card
    /// publishes a new snapshot (see [`Self::harmony`]). The rule that reads
    /// a worker's window stays with the pipeline's context check, as
    /// `harmony`'s stays with the router's detector.
    ///
    /// The figure is kept with the model id it was read for, as spelled: a
    /// worker answers an id by the card that matches it and, with none, by
    /// its primary card, so another spelling can read other windows. `None`
    /// for another spelling than the kept one: the caller then asks the
    /// workers themselves. The gRPC router resolves an alias to its
    /// canonical id before it selects, so a model's snapshot is read under
    /// one spelling.
    pub(crate) fn window_floor(
        &self,
        pool: RoutingPool,
        model_id: &str,
        read: impl FnOnce(&[Arc<dyn Worker>]) -> WindowFloor,
    ) -> Option<WindowFloor> {
        let workers = self.pool(pool);
        if workers.is_empty() {
            // Nothing to keep, whatever the spelling: the snapshot shared by
            // every unknown model keeps no client's model id.
            return Some(WindowFloor::default());
        }
        let (spelling, floor) =
            self.window_floors[pool.index()].get_or_init(|| (model_id.into(), read(workers)));
        (**spelling == *model_id).then_some(*floor)
    }

    /// What the snapshot's workers say together for `model_id`: `read` reads
    /// them on the first request after the membership change, every request
    /// after it reads the kept answers, so preparing a request walks no
    /// worker, whatever the size of the fleet. Whatever changes a card
    /// publishes a new snapshot (see [`Self::harmony`]), and a worker's
    /// labels are read once, at its construction. The rules stay with the
    /// stage that prepares a request's reading, as `harmony`'s stays with
    /// the router's detector.
    ///
    /// The answers are kept with the model id they were read for, as
    /// spelled, like a window floor ([`Self::window_floor`]). `None` for
    /// another spelling than the kept one, and for a snapshot without a
    /// worker (the one every unknown model shares keeps no client's model
    /// id): the caller then reads the workers itself.
    pub(crate) fn labels(
        &self,
        model_id: &str,
        read: impl FnOnce(&[Arc<dyn Worker>]) -> ModelLabels,
    ) -> Option<&ModelLabels> {
        if self.all.is_empty() {
            return None;
        }
        let (spelling, labels) = self
            .labels
            .get_or_init(|| (model_id.into(), read(&self.all)));
        (**spelling == *model_id).then_some(labels)
    }

    /// What the media gates read from the snapshot's workers together:
    /// `read` reads them on the first media request after the membership
    /// change, every request after it reads the kept answers. They hang on
    /// the workers' labels, connection modes and engines, which a worker
    /// holds from its construction (see [`Self::labels`]), and on no model
    /// id as spelled, so one snapshot keeps one value. The rules stay with
    /// the media gates.
    pub(crate) fn media(&self, read: impl FnOnce(&[Arc<dyn Worker>]) -> MediaFacts) -> &MediaFacts {
        self.media.get_or_init(|| read(&self.all))
    }

    /// The consistent-hash ring over the URLs of this generation's members,
    /// once it is filled ([`Self::fill_hash_ring`], [`Self::put_hash_ring`]).
    pub(crate) fn hash_ring(&self) -> Option<&Arc<HashRing>> {
        self.hash_ring.get()
    }

    /// Fills the ring cell, where it is empty, and answers its ring. The
    /// fill reconciles `kept`, the ring the gateway kept for the model, to
    /// these members, hashing only the URLs it does not hold; with none
    /// kept it builds the ring whole. A kept ring over these very members
    /// is taken as it is. A call that arrives during another's fill waits
    /// for it. Whatever ring `kept` is, the cell holds the ring of these
    /// members: `kept` saves hashing and decides nothing of the ring filled.
    pub(crate) fn fill_hash_ring(&self, kept: Option<Arc<HashRing>>) -> &Arc<HashRing> {
        self.hash_ring.get_or_init(|| {
            let urls = || self.all.iter().map(|worker| worker.url());
            match kept {
                Some(kept) if kept.is_over(urls()) => kept,
                Some(kept) => Arc::new(kept.updated(urls())),
                None => Arc::new(HashRing::new(urls())),
            }
        })
    }

    /// Puts `ring` into the ring cell where `listed` are these members'
    /// URLs in this list's order, as after a generation is published again
    /// over the same list. `ring` is the ring over `listed`
    /// ([`Self::listed_urls`] of the generation it was filled for), so it
    /// is then the ring of these members: a compare of the URLs, with
    /// nothing hashed. Any other ring is left out, and so is every ring
    /// once the cell is filled.
    pub(crate) fn put_hash_ring(&self, ring: &Arc<HashRing>, listed: &[Arc<str>]) {
        let same_list = listed.len() == self.all.len()
            && self
                .all
                .iter()
                .zip(listed)
                .all(|(worker, url)| &**url == worker.url());
        if same_list {
            // A fill that got there first put in a ring of the same content.
            let _ = self.hash_ring.set(Arc::clone(ring));
        }
    }

    /// These members' URLs in this list's order, kept beside `ring`, this
    /// generation's ring, for a later generation's [`Self::put_hash_ring`].
    /// A ring keeps an order of its own, so they are the ring's own list
    /// only where that is in this order, and a copy elsewhere.
    pub(crate) fn listed_urls(&self, ring: &HashRing) -> Arc<[Arc<str>]> {
        let urls = || self.all.iter().map(|worker| worker.url());
        if ring.is_over(urls()) {
            Arc::clone(ring.urls())
        } else {
            urls().map(Arc::<str>::from).collect()
        }
    }

    /// Whether the ring cell is filled, for the cases that say who filled it.
    #[cfg(test)]
    pub(crate) fn hash_ring_is_filled(&self) -> bool {
        self.hash_ring.get().is_some()
    }

    /// The compatible prefill/decode pairs of `wire` under `mode`: descriptor
    /// comparisons happen here, once per membership snapshot, never on the
    /// request path.
    pub(crate) fn pd_pairs(&self, wire: PdWire, mode: PdPairingMode) -> Arc<PdPairIndex> {
        self.pd_pairs[wire.index()][mode.index()]
            .get_or_init(|| {
                let (prefill, decode) = wire.pools();
                Arc::new(PdPairIndex::build(
                    Arc::clone(self.pool(prefill)),
                    Arc::clone(self.pool(decode)),
                    wire,
                    mode,
                ))
            })
            .clone()
    }

    /// The workers of `pool`, projected once per snapshot and borrowed from
    /// it: a length or membership read costs no refcount step. A caller that
    /// keeps the pool past the snapshot clones the `Arc` itself.
    pub(crate) fn pool(&self, pool: RoutingPool) -> &WorkerSnapshot {
        self.pools[pool.index()].get_or_init(|| {
            if self.all.iter().all(|worker| pool.matches(worker.as_ref())) {
                return Arc::clone(&self.all);
            }
            self.all
                .iter()
                .filter(|worker| pool.matches(worker.as_ref()))
                .cloned()
                .collect()
        })
    }

    /// The list of `pool` as the policy engine is lent it: the pool's own
    /// list ([`Self::pool`]) with the cell that stands beside it. The engine
    /// keeps its value of the list in that cell from the list's first
    /// request on, and the value falls with this snapshot, as the list
    /// does: a membership change publishes the next one with empty cells.
    /// The gateway states no fact of a worker beside its object.
    pub(crate) fn lent(&self, pool: RoutingPool) -> LentList<'_> {
        LentList {
            workers: self.pool(pool),
            cell: &self.cells[pool.index()],
            facts: None,
        }
    }
}

impl Deref for ModelWorkerSnapshot {
    type Target = [Arc<dyn Worker>];

    fn deref(&self) -> &Self::Target {
        &self.all
    }
}

/// The gateway's value of a generation: the members, and the cells that
/// keep what its routers derive from them.
impl Generation for ModelWorkerSnapshot {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self::new(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.all
    }
}

/// The registry as the gateway names it: its value of a membership
/// generation is the model snapshot that keeps the routers' projections.
/// The load monitor holds that registry. The registry is the worker crate's;
/// this name for it is the gateway's.
pub type WorkerRegistry = registry::WorkerRegistry<ModelWorkerSnapshot>;
/// The load monitor over that registry.
pub type WorkerMonitor = load::monitor::WorkerMonitor<ModelWorkerSnapshot>;

/// The routing readers. They are the gateway's, over the registry's readers
/// of a generation: a trait, because the registry is the worker crate's type
/// and an inherent impl stands only where its type is defined.
pub(crate) trait WorkerRegistryExt {
    /// Return a shared route-specific candidate snapshot for a model.
    ///
    /// The first lookup after a membership change builds the projection once;
    /// subsequent requests only clone the outer `Arc`. Dynamic availability is
    /// intentionally checked by the policy so health, circuit-breaker and
    /// overload changes never require rebuilding this snapshot. The wildcard
    /// model ID uses the global registry snapshot.
    fn get_routing_pool(&self, model_id: &str, pool: RoutingPool) -> Arc<[Arc<dyn Worker>]>;

    /// Like [`Self::get_routing_pool`] but never widens to the global
    /// snapshot: the wildcard id resolves through the literal `"unknown"`
    /// model-index entry (untagged workers index there), yielding an empty
    /// slice when no such entry exists. The HTTP PD router uses this to keep
    /// its conditional wildcard fallback — untagged workers win when present,
    /// and only an empty entry widens to the global pool.
    fn get_model_routing_pool(&self, model_id: &str, pool: RoutingPool) -> Arc<[Arc<dyn Worker>]>;

    /// One membership snapshot for a model (alias-aware), from which several
    /// role pools can be derived atomically: a PD/EPD selection must draw all
    /// its roles from a single generation, and two separate pool lookups could
    /// straddle a concurrent membership change.
    fn model_routing_snapshot(&self, model_id: &str) -> Option<Arc<ModelWorkerSnapshot>>;

    /// One membership snapshot for multi-role selection, with the wildcard
    /// model mapping to the global snapshot and an unknown model to a shared
    /// empty one.
    fn get_routing_snapshot(&self, model_id: &str) -> Arc<ModelWorkerSnapshot>;

    /// Return the immutable global membership snapshot used by routing paths
    /// whose model/provider filters are evaluated dynamically, and by the
    /// readiness recompute, which visits every worker: one pointer load while
    /// the fleet is stable, where an iteration of the worker map locks every
    /// shard.
    fn get_routing_workers(&self) -> Arc<[Arc<dyn Worker>]>;
}

/// No worker, for a model the registry does not hold.
const EMPTY_WORKERS: &[Arc<dyn Worker>] = &[];

impl WorkerRegistryExt for WorkerRegistry {
    fn get_routing_pool(&self, model_id: &str, pool: RoutingPool) -> Arc<[Arc<dyn Worker>]> {
        if model_id == UNKNOWN_MODEL_ID {
            return Arc::clone(self.global_generation().pool(pool));
        }
        self.get_model_routing_pool(model_id, pool)
    }

    fn get_model_routing_pool(&self, model_id: &str, pool: RoutingPool) -> Arc<[Arc<dyn Worker>]> {
        self.model_routing_snapshot(model_id).map_or_else(
            || Arc::from(EMPTY_WORKERS),
            |snapshot| Arc::clone(snapshot.pool(pool)),
        )
    }

    fn model_routing_snapshot(&self, model_id: &str) -> Option<Arc<ModelWorkerSnapshot>> {
        self.model_generation(model_id)
    }

    fn get_routing_snapshot(&self, model_id: &str) -> Arc<ModelWorkerSnapshot> {
        self.generation(model_id)
    }

    fn get_routing_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        Arc::clone(self.global_generation().members())
    }
}

#[cfg(any(test, feature = "test-util"))]
pub mod test_support;
#[cfg(test)]
mod tests;
