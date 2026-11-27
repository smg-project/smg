//! Per-worker KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per gRPC worker that subscribes
//! to KV cache events and feeds them into a shared [`KvIndex`] (one per model:
//! the chain index).
//! This enables event-driven cache-aware routing as an alternative to the approximate
//! radix tree approach.
//!
//! Lifecycle, kept by the worker crate's KV event collector, whose host is
//! the monitor's index side:
//! - `on_worker_added` — spawns streaming task, creates indexer if needed
//! - `on_worker_removed` — signals graceful shutdown, task cleans up indexer
//! - `stop` — signals shutdown to all tasks, clears state
//!
//! The work of a subscription is split by stage: the collector keeps one
//! task a worker and runs it (the slots, connecting, reconnecting, the
//! stream read loop, the pushed load records, the worker's departure);
//! `admission.rs` keeps one stream's state (per-rank cursors, gap recovery,
//! relay snapshots, and the metrics around an applied batch);
//! [`smg_worker::kv_events::apply`] takes one event into the index (tiers,
//! cache groups, namespaces, and the physical copies of a block, counted per
//! worker).
//!
//! The task and the index meet at the worker crate's port: the task says
//! what its stream brought, and `index_sink.rs`, the sink over the models'
//! indexes, keeps each worker's stream state and applies what its cursor
//! admits.

use std::{fmt, sync::Arc};

use dashmap::{mapref::entry::Entry, DashMap};
use smg_grand_central::engine::{CacheIndex, CacheIndexes};
use smg_worker::{
    identity::{basic::Worker, endpoint::Endpoint},
    load::monitor::WorkerMonitor,
    metrics::writers::{KvIndexChainStats, Metrics},
    periodic_task::PeriodicTask,
    registry::{Generation, WorkerRegistry},
};
use tracing::{error, warn};

use super::index_backend::KvIndex;
use crate::policies::normalize_model_key;

mod admission;
#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/event_monitor/gateway_tests.rs"]
mod gateway_tests;
mod index_sink;
mod recovery;

use index_sink::{IndexSink, OpenRow};
// The worker crate's port, at the path it had here.
use smg_worker::kv_events::port;
pub use smg_worker::kv_events::SubscriptionKind;
use smg_worker::kv_events::{port::KvEventSink, KvEventCollector, KvEventFeed, KvEventHost};

/// Interval between publications of each model's index size and, for the chain
/// index, its shape and memory (`smg_kv_index_*` gauges by model).
const STATS_INTERVAL_SECS: u64 = 30;

/// Manages per-worker KV cache event subscriptions.
///
/// Each gRPC worker gets a dedicated tokio task that subscribes to the backend's
/// KV cache event stream and feeds events into a shared [`KvIndex`]
/// (one per `model_id`). Workers serving the same model share the same indexer.
/// A model's KV block size as the monitor knows it: provisional until a
/// stored event names it. A provisional size (a `WorkerSpec` seed or
/// `set_block_size`) is replaced by the first stored block; a size the
/// events named is only ever raised. Hybrid engines publish, beside their
/// full blocks, partial entries at their hash granularity (a fraction of
/// the block), and a request hashed at that size would never match a full
/// block, so the size of record is the largest block the events carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvBlockSize {
    pub(crate) tokens: usize,
    pub(crate) from_events: bool,
}

pub struct KvEventMonitor {
    /// Per-model KV indexes: model_id → shared indexer.
    /// Arc-wrapped so the stats task can share the map WITHOUT holding (even
    /// weakly) the monitor itself: `PeriodicTask` joins its thread on drop, so
    /// a task that could ever own the last monitor reference would run the
    /// monitor's drop — and thus its own join — on its own thread.
    pub(crate) indexers: Arc<DashMap<String, Arc<KvIndex>>>,
    /// Per-model block sizes learned from KV events or set via WorkerSpec.
    /// Used by CacheAwarePolicy to chunk request tokens at query time.
    /// Arc-wrapped so subscription tasks can update it from events.
    block_sizes: Arc<DashMap<String, KvBlockSize>>,
    /// The subscriptions' lifecycle: one task a worker, started and ended
    /// under the slot lock. Its host is the index side, which shares the two
    /// maps above.
    collector: KvEventCollector<IndexHost>,
    /// Periodic index-shape publication, held so it aborts when the monitor
    /// drops. Set once by [`start_stats_task`](Self::start_stats_task); sync
    /// mutex because it is touched only at startup, never on event paths.
    stats_task: parking_lot::Mutex<Option<PeriodicTask>>,
}

/// The index side of the monitor, as the collector's host: the models'
/// indexes, their block sizes, and the sink over them.
struct IndexHost {
    indexers: Arc<DashMap<String, Arc<KvIndex>>>,
    block_sizes: Arc<DashMap<String, KvBlockSize>>,
    /// Where the events subscriptions' batches go: the sink over the models'
    /// indexes, which keeps each worker's stream state by its row.
    sink: Arc<IndexSink>,
}

impl KvEventHost for IndexHost {
    type Sink = OpenRow;
    /// The model's index as the add found or made it: the one its task feeds.
    type Ticket = Arc<KvIndex>;

    fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) -> Arc<KvIndex> {
        let (indexer, created) = match self.indexers.entry(model.to_string()) {
            Entry::Occupied(entry) => (Arc::clone(entry.get()), false),
            Entry::Vacant(entry) => {
                let index = Arc::new(KvIndex::chain());
                entry.insert(Arc::clone(&index));
                (index, true)
            }
        };
        if created {
            KvEventMonitor::publish_index(model, &indexer);
        }
        // Seed block_size provisionally from WorkerSpec. The event stream will
        // overwrite this with the backend's actual page size once received.
        if let Some(bs) = worker.metadata().spec.kv_block_size {
            if bs > 0 {
                self.sink
                    .set_block_size(model, u32::try_from(bs).unwrap_or(u32::MAX));
            } else {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker.url()),
                    "Worker reports kv_block_size=0, ignoring"
                );
            }
        }
        indexer
    }

    fn open(
        &self,
        ticket: Arc<KvIndex>,
        worker: &Arc<dyn Worker>,
        model: &str,
    ) -> Option<KvEventFeed<OpenRow>> {
        match self
            .sink
            .open(ticket, model, worker.url(), worker.dp_rank())
        {
            Ok(row) => Some(row.into_feed()),
            Err(e) => {
                error!(
                    worker_url = %Endpoint::redacted_str(worker.url()),
                    error = %e,
                    "Failed to intern worker; KV events from this worker will \
                     not feed cache-aware routing"
                );
                Metrics::record_kv_event_subscription_failure(worker.url(), "intern_failed");
                None
            }
        }
    }

    fn model_left(&self, model: &str) {
        self.indexers.remove(model);
        self.block_sizes.remove(model);
        KvEventMonitor::publish_dropped_index(model);
    }
}

impl KvEventMonitor {
    /// A monitor whose models get chain indexes.
    pub fn new() -> Self {
        let indexers = Arc::new(DashMap::new());
        let block_sizes = Arc::new(DashMap::new());
        let host = Arc::new(IndexHost {
            indexers: Arc::clone(&indexers),
            sink: Arc::new(IndexSink::new(Arc::clone(&block_sizes))),
            block_sizes: Arc::clone(&block_sizes),
        });
        Self {
            indexers,
            block_sizes,
            collector: KvEventCollector::new(host),
            stats_task: parking_lot::Mutex::new(None),
        }
    }

    /// Route the load records on every worker's event stream to the worker
    /// monitor (once; a later call is ignored).
    pub fn set_load_sink<G: Generation>(&self, monitor: &Arc<WorkerMonitor<G>>) {
        self.collector.set_load_sink(monitor);
    }

    /// Have the add read the registry before it stops or starts a stream
    /// (once; a later call is ignored): a worker the registry no longer
    /// holds under its URL gets none.
    pub fn set_worker_registry<G: Generation>(&self, registry: &Arc<WorkerRegistry<G>>) {
        self.collector.set_worker_registry(registry);
    }

    /// Publish every model's index size, and the chain index's shape and
    /// memory, as gauges: what a soak reads to tell fragmentation or growth
    /// from load. `current_size` and `entry_count` are counter reads; the chain
    /// index's `stats` walks its run headers, which is why this runs on a
    /// 30 s cadence and never on a request.
    fn publish_stats(indexers: &DashMap<String, Arc<KvIndex>>) {
        for entry in indexers {
            Self::publish_index(entry.key(), entry.value());
        }
    }

    /// The chain index's counters its gauges publish, as the worker's
    /// writers take them.
    fn chain_counters(stats: &kv_index::ChainIndexStats) -> KvIndexChainStats {
        KvIndexChainStats {
            runs_live: stats.runs_live,
            blocks_live: stats.blocks_live,
            arena_bytes: stats.arena_bytes,
            arena_free_bytes: stats.arena_free_bytes,
            arena_chunk_bytes: stats.arena_chunk_bytes,
            slab_bytes: stats.slab_bytes,
            moved_hashes: stats.moved_hashes,
            engine_conflicts: stats.engine_conflicts,
            cap_refusals: stats.cap_refusals,
        }
    }

    /// One model's index size and chain shape as gauges: the periodic
    /// publication's unit, and what a new index publishes at once, so the
    /// model's series exist (at zero) from its first worker rather than
    /// from the first tick.
    fn publish_index(model_id: &str, index: &KvIndex) {
        Metrics::set_kv_index_size(model_id, index.current_size(), index.entry_count());
        if let Some(stats) = index.chain_stats() {
            Metrics::set_kv_index_chain_stats(model_id, &Self::chain_counters(&stats));
        }
    }

    /// Zero a model's index-shape gauges once its index is dropped. The
    /// periodic publication only sees the indexes that exist, and a series
    /// keeps its last value, so without this the gauges would report the
    /// last worker's blocks for as long as no worker of the model joins
    /// again (an engine replaced behind a one-engine gateway, a model scaled
    /// to zero).
    fn publish_dropped_index(model_id: &str) {
        Metrics::set_kv_index_size(model_id, 0, 0);
        Metrics::set_kv_index_chain_stats(model_id, &KvIndexChainStats::default());
        // The size went with the model's last subscription: 0, a size no
        // block has, says none is known.
        Metrics::set_kv_block_size(model_id, 0);
    }

    /// Start the periodic index-shape publication. The task shares only the
    /// indexer map, never a reference to the monitor itself, so it can never
    /// be the one to run the monitor's drop (and with it its own thread's
    /// join; see the `indexers` field docs); the monitor holds the handle, so
    /// the task stops when the monitor drops.
    pub fn start_stats_task(&self) {
        let indexers = Arc::clone(&self.indexers);
        let task = PeriodicTask::spawn(STATS_INTERVAL_SECS, "KvIndexStats", move || {
            Self::publish_stats(&indexers);
        });
        *self.stats_task.lock() = Some(task);
    }

    /// Start a worker's stream: its KV events into the model's `KvIndex`
    /// (`Events`), or its load records alone (`Loads`). A second call for the
    /// object already streaming is a no-op, and a call with another object
    /// under an active URL restarts the stream for it. The collector holds
    /// the rules.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>, kind: SubscriptionKind) {
        self.collector.on_worker_added(worker, kind).await;
    }

    /// Stop the KV event subscription for a worker, unless the stream under
    /// the URL is the one of the object the registry holds there now.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        self.collector.on_worker_removed(worker_url).await;
    }

    /// A worker that streams nothing, in place of one that did: the stream a
    /// former object has under the URL ends here.
    pub async fn on_worker_added_without_stream(&self, worker: &Arc<dyn Worker>) {
        self.collector.on_worker_added_without_stream(worker).await;
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        self.collector.stop().await;
        self.indexers.clear();
        self.block_sizes.clear();
    }

    /// The ids under a URL's slot: the active subscription's, or the
    /// reservation's of a removal under way.
    #[cfg(test)]
    async fn slot_ids(&self, worker_url: &str) -> (Option<u64>, Option<u64>) {
        self.collector.slot_ids(worker_url).await
    }

    /// Whether the task under a URL's slot has ended.
    #[cfg(test)]
    async fn task_ended(&self, worker_url: &str) -> bool {
        self.collector.task_ended(worker_url).await
    }

    /// Leave the URL's running stream as a reservation whose task is gone.
    #[cfg(test)]
    async fn strand_reservation(&self, worker_url: &str) {
        self.collector.strand_reservation(worker_url).await;
    }

    /// Get the indexer for a model (used by `CacheAwarePolicy` for queries).
    pub fn get_indexer(&self, model_id: &str) -> Option<Arc<KvIndex>> {
        self.indexers.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the block size for a model (learned from events or set via `set_block_size`).
    pub fn block_size(&self, model_id: &str) -> Option<usize> {
        self.block_sizes.get(model_id).map(|v| v.tokens)
    }

    /// Set the block size for a model (e.g. from WorkerSpec during registration).
    /// Does not overwrite a value already learned from events.
    pub fn set_block_size(&self, model_id: &str, block_size: usize) {
        self.block_sizes
            .entry(model_id.to_string())
            .or_insert(KvBlockSize {
                tokens: block_size,
                from_events: false,
            });
    }
}

/// Grand Central's reads of the monitor, where it decides in the
/// gateway's own process: a model's index and its block size, each read
/// where the engine reads a list of workers, never once a request.
impl CacheIndexes for KvEventMonitor {
    /// The model's index, whenever the model has one: while it holds no
    /// block too, from the enrolment of the model's first worker to its
    /// first event, and again once its last holder left. The index says
    /// itself whether it holds a block ([`CacheIndex::holds_a_block`]),
    /// and the engine reads one that holds none as no index: it decides
    /// the request in the token tree the gateway lends it, as the classic
    /// policy does in its own (`CacheAwarePolicy::event_index_for`). The
    /// engine asks that word at each decision, so a model's first event
    /// and its last holder's leaving decide its next request.
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        let index = self.get_indexer(normalize_model_key(model))?;
        Some(index as Arc<dyn CacheIndex>)
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        let tokens = Self::block_size(self, normalize_model_key(model))?;
        u32::try_from(tokens).ok().filter(|&tokens| tokens > 0)
    }
}

/// Hooks for benchmarks and integration tests that put an index in front of
/// the policy and feed it events the way a subscription does, without a
/// stream: the subscriber's per-worker state and the same apply path.
#[cfg(any(test, feature = "test-util"))]
pub mod bench_support {
    use std::sync::Arc;

    use kv_index::WorkerIdExhausted;
    use smg_grpc_client::common_proto::KvEventBatch;
    use smg_worker::kv_events::apply::{apply_events, WorkerIndexState};

    use super::KvEventMonitor;
    use crate::policies::kv_index::index_backend::{KvIndex, WorkerBlocks};

    impl KvEventMonitor {
        /// Make `index` the model's index, as a worker's first subscription
        /// would; a later subscription for the model shares it.
        pub fn set_index(&self, model_id: &str, index: Arc<KvIndex>) {
            self.indexers.insert(model_id.to_string(), index);
        }
    }

    /// One worker's feed into an index.
    pub struct IndexFeed {
        worker: u32,
        state: WorkerIndexState<WorkerBlocks>,
    }

    impl IndexFeed {
        /// Intern `worker_url` in `index` and start with empty state.
        pub fn new(index: &KvIndex, worker_url: &str) -> Result<Self, WorkerIdExhausted> {
            Ok(Self {
                worker: index.intern_worker(worker_url)?,
                state: WorkerIndexState::default(),
            })
        }

        pub fn worker_id(&self) -> u32 {
            self.worker
        }

        /// Apply every event of `batch`, as an admitted batch is applied.
        pub fn apply(&mut self, index: &KvIndex, batch: &KvEventBatch) {
            apply_events(&batch.events, self.worker, index, &mut self.state);
        }

        /// The worker leaves: its blocks go with it.
        pub fn remove(self, index: &KvIndex) {
            index.remove_worker(self.worker, self.state.blocks);
        }
    }
}

impl Default for KvEventMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for KvEventMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvEventMonitor")
            .field("models", &self.indexers.len())
            .field("block_sizes", &self.block_sizes.len())
            .finish()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/event_monitor/tests.rs"]
mod tests;
