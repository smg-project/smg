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
use smg_policy_controller::engine::{CacheIndex, CacheIndexes};
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

/// The policy engine's reads of the monitor, where it decides in the
/// gateway's own process: a model's index and its block size, each read
/// where the engine reads a list of workers, never once a request.
impl CacheIndexes for KvEventMonitor {
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
mod tests {
    use std::time::Duration;

    use kv_index::{ContentHash, SequenceHash, StoredBlock};
    use openai_protocol::{
        worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerSpec, WorkerType},
        UNKNOWN_MODEL_ID,
    };
    use smg_worker::kv_events::{MAX_CONCURRENT_SINK_REMOVALS, SINK_REMOVAL_PERMITS};
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        policies::kv_index::index_backend::WorkerBlocks,
        routers::grpc::backend_client::BasicWorkerBuilder,
    };

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        for _ in 0..200 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    /// The cases' worker address: a closed port, so a task loops on
    /// reconnects and answers shutdown with no engine involved.
    const REFUSING_URL: &str = "grpc://127.0.0.1:2";

    /// A gRPC worker at [`REFUSING_URL`].
    fn refusing_grpc_worker() -> Arc<dyn Worker> {
        refusing_grpc_worker_at(REFUSING_URL)
    }

    /// A gRPC worker at `url`, a closed port.
    fn refusing_grpc_worker_at(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// A worker removed and added again under the same URL while the old
    /// subscription's cleanup is still running: the index hands the URL its
    /// old id until that cleanup has released it, so the re-add must wait
    /// for the removal to finish rather than intern into the old id and
    /// lose its blocks and its id to the cleanup.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_same_url_re_add_waits_for_the_old_subscriptions_cleanup() {
        // The chain index releases a removed worker's id and name, which is
        // what makes the window observable.
        let monitor = Arc::new(KvEventMonitor::new());
        let worker = refusing_grpc_worker();
        let url = worker.url().to_string();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("the subscription interns its worker", || {
            index.worker_id(&url).is_some()
        })
        .await;

        // Hold every cleanup permit: the old task's cleanup cannot finish.
        let permits = SINK_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
            .await
            .expect("the removal semaphore is open");
        #[expect(
            clippy::disallowed_methods,
            reason = "the removal and the re-add are awaited below"
        )]
        let removal = tokio::spawn({
            let monitor = Arc::clone(&monitor);
            let url = url.clone();
            async move { monitor.on_worker_removed(&url).await }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!removal.is_finished(), "the removal waits for the cleanup");
        #[expect(
            clippy::disallowed_methods,
            reason = "the removal and the re-add are awaited below"
        )]
        let re_add = tokio::spawn({
            let monitor = Arc::clone(&monitor);
            let worker = Arc::clone(&worker);
            async move {
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !re_add.is_finished(),
            "the re-add waits for the old subscription's cleanup"
        );

        drop(permits);
        removal.await.expect("removal task");
        assert!(
            index.worker_id(&url).is_none(),
            "the removal returns once the old subscription has released its id"
        );
        re_add.await.expect("re-add task");
        // The worker was the model's last, so the removal dropped the model's
        // index and the re-add created a new one; the new subscription
        // interned the URL there, after the release, with an id of its own.
        let current = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the re-added worker's model has an index");
        wait_until("the new subscription interns its worker", || {
            current.worker_id(&url).is_some()
        })
        .await;
        monitor.stop().await;
        assert!(
            current.worker_id(&url).is_none(),
            "stop takes the new subscription out of the index"
        );
    }

    /// A caller may drop `on_worker_removed` before the old task has ended:
    /// the registry's removal step runs under a timeout, and a cleanup
    /// waiting on the removal permits can outlast it. The reservation the
    /// removal left must not outlive the task: once the task is gone, the
    /// model's last index is gone with it and an add of the URL goes
    /// through, with nobody awaiting the removal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_removal_leaves_no_reservation_behind() {
        let monitor = Arc::new(KvEventMonitor::new());
        let worker = refusing_grpc_worker();
        let url = worker.url().to_string();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("the subscription interns its worker", || {
            index.worker_id(&url).is_some()
        })
        .await;

        // Hold every cleanup permit: the old task's cleanup cannot finish.
        let permits = SINK_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
            .await
            .expect("the removal semaphore is open");
        // The removal is dropped while it awaits the old task, as a timeout
        // around it would drop it.
        let removal =
            tokio::time::timeout(Duration::from_millis(200), monitor.on_worker_removed(&url)).await;
        assert!(
            removal.is_err(),
            "the removal was still waiting for the old task's cleanup"
        );
        // The cleanup is still held: a re-add waits, and nothing has moved.
        let re_add = tokio::time::timeout(
            Duration::from_millis(200),
            monitor.on_worker_added(&worker, SubscriptionKind::Events),
        )
        .await;
        assert!(
            re_add.is_err(),
            "the re-add waits while the old task's cleanup is held"
        );
        assert!(
            monitor.get_indexer(UNKNOWN_MODEL_ID).is_some(),
            "the model's index stays until the old task has finished"
        );

        drop(permits);
        // Nobody awaits the removal any more: the old task finishes it on
        // its own, releasing its id and dropping the model's last index.
        wait_until("the old task drops the model's index", || {
            monitor.get_indexer(UNKNOWN_MODEL_ID).is_none()
        })
        .await;
        assert!(
            index.worker_id(&url).is_none(),
            "the old task released its id"
        );
        tokio::time::timeout(
            Duration::from_secs(5),
            monitor.on_worker_added(&worker, SubscriptionKind::Events),
        )
        .await
        .expect("the re-add completes once the old task has ended");
        let current = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the re-added worker's model has an index");
        wait_until("the new subscription interns its worker", || {
            current.worker_id(&url).is_some()
        })
        .await;
        monitor.stop().await;
    }

    /// An add can find a reservation whose task is gone and that nobody
    /// lifted. It lifts it before it starts the worker's stream: behind the
    /// model's last events subscription the model's index is dropped and the
    /// add makes a fresh one, and the worker's gauge is lowered.
    #[test]
    fn an_add_behind_a_dead_reservation_builds_the_model_a_fresh_index() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn gauge(handle: &PrometheusHandle, worker_url: &str) -> Option<f64> {
            let prefix = format!("smg_kv_events_unavailable{{worker=\"{worker_url}\"}}");
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&prefix))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        // The add publishes from the runtime's thread, which must be the one
        // holding the local recorder.
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let monitor = KvEventMonitor::new();
                let worker = refusing_grpc_worker_at("grpc://127.0.0.1:4");
                let url = worker.url();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                let before = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                // Raised as an engine's refusal of the stream raises it.
                Metrics::set_kv_events_unavailable(url, true);
                assert_eq!(gauge(&handle, url), Some(1.0));
                let (stranded, _) = monitor.slot_ids(url).await;
                monitor.strand_reservation(url).await;
                assert_eq!(monitor.slot_ids(url).await, (None, stranded));

                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                let after = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the re-added worker's model has an index");
                assert!(
                    !Arc::ptr_eq(&before, &after),
                    "the lift dropped the index of the model's last events subscription; the \
                     add made a fresh one"
                );
                let (active, removing) = monitor.slot_ids(url).await;
                assert!(active.is_some_and(|id| Some(id) != stranded));
                assert_eq!(removing, None, "the add lifted the dead reservation");
                assert_eq!(gauge(&handle, url), Some(0.0), "the lift lowers the gauge");
                monitor.stop().await;
            });
        });
    }

    /// The same add while another worker of the model still feeds the index:
    /// the model keeps its index.
    #[tokio::test]
    async fn an_add_behind_a_dead_reservation_keeps_an_index_another_worker_feeds() {
        let monitor = KvEventMonitor::new();
        let stays = refusing_grpc_worker_at("grpc://127.0.0.1:5");
        let worker = refusing_grpc_worker_at("grpc://127.0.0.1:6");
        let url = worker.url();
        monitor
            .on_worker_added(&stays, SubscriptionKind::Events)
            .await;
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let before = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        monitor.strand_reservation(url).await;

        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let (active, removing) = monitor.slot_ids(url).await;
        assert!(active.is_some(), "the add starts a stream of its own");
        assert_eq!(removing, None, "the add lifted the dead reservation");
        let after = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        assert!(
            Arc::ptr_eq(&before, &after),
            "another events worker of the model is active: its index stays"
        );
        monitor.stop().await;
    }

    #[test]
    fn a_new_monitor_holds_no_index() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.indexers.is_empty());
    }

    /// The `Events` restart takes the whole removal path: the old task
    /// purges the URL's blocks and, as the model's only events subscription,
    /// drops the model's index; the new subscription builds a fresh one and
    /// interns its worker there.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_re_registration_of_an_events_worker_rebuilds_the_models_index() {
        let monitor = Arc::new(KvEventMonitor::new());
        let first = refusing_grpc_worker();
        let url = first.url().to_string();
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        let before = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("the first subscription interns its worker", || {
            before.worker_id(&url).is_some()
        })
        .await;

        let second = refusing_grpc_worker();
        monitor
            .on_worker_added(&second, SubscriptionKind::Events)
            .await;
        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));
        let after = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the re-added worker's model has an index");
        assert!(
            !Arc::ptr_eq(&before, &after),
            "the old task dropped the model's index with its last events subscription; the \
             restart built a fresh one"
        );
        assert!(
            before.worker_id(&url).is_none(),
            "the old index released the URL's id before it was dropped"
        );
        wait_until(
            "the new subscription interns its worker in the fresh index",
            || after.worker_id(&url).is_some(),
        )
        .await;
        monitor.stop().await;
    }

    #[tokio::test]
    async fn an_unknown_model_has_no_index() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.get_indexer("nonexistent").is_none());
    }

    /// The policy engine's reads of the monitor. A block that a worker's
    /// event stored in a model's index is what the engine's lookup answers,
    /// under the index's own number of the worker, and the model's block
    /// size is the one the monitor holds. The index the engine is handed is
    /// the monitor's own, not a copy. A model with no index answers none.
    #[test]
    fn the_engine_reads_the_index_an_event_wrote() {
        use kv_index::compute_request_content_hashes;
        use smg_grpc_client::common_proto::{
            kv_cache_event::Data, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch,
        };

        let monitor = KvEventMonitor::new();
        let index = Arc::new(KvIndex::chain());
        monitor.set_index("llama", Arc::clone(&index));
        monitor.set_block_size("llama", 4);
        let tokens = [10, 20, 30, 40];
        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: tokens.to_vec(),
                block_size: 4,
                ..Default::default()
            }],
            ..Default::default()
        };
        let batch = KvEventBatch {
            events: vec![KvCacheEvent {
                event_id: 1,
                data: Some(Data::Stored(stored)),
            }],
            ..Default::default()
        };
        let mut feed = bench_support::IndexFeed::new(&index, "grpc://w1:9000").unwrap();
        feed.apply(&index, &batch);

        let read: &dyn CacheIndexes = &monitor;
        assert_eq!(read.block_size("llama"), Some(4));
        let lent = read.index("llama").expect("the model's index");
        assert!(
            std::ptr::addr_eq(Arc::as_ptr(&lent), Arc::as_ptr(&index)),
            "the monitor's own index"
        );
        let number = feed.worker_id();
        assert_eq!(lent.worker_number("grpc://w1:9000"), Some(number));
        assert_eq!(lent.worker_blocks(number), 1);
        let hashes = compute_request_content_hashes(&tokens, 4);
        let request: Vec<u64> = hashes.iter().map(|hash| hash.0).collect();
        let mut kept = Vec::new();
        lent.deepest_into(&request, 4, &[1 << number], 16, &mut |_| 0, &mut kept);
        assert_eq!(kept, [(number, 1)]);

        assert!(read.index("nonexistent").is_none());
        assert_eq!(read.block_size("nonexistent"), None);
    }

    #[test]
    fn set_block_size_keeps_the_first_value() {
        let monitor = KvEventMonitor::new();

        // Initially no block_size
        assert!(monitor.block_size("llama").is_none());

        // Set it
        monitor.set_block_size("llama", 32);
        assert_eq!(monitor.block_size("llama"), Some(32));

        // set_block_size doesn't overwrite existing value
        monitor.set_block_size("llama", 64);
        assert_eq!(monitor.block_size("llama"), Some(32));
    }

    /// The add of an events worker seeds the model's block size from the
    /// worker's spec, and the size leaves with the model's last events
    /// subscription: the model's next worker seeds or learns its own.
    #[tokio::test]
    async fn the_block_size_comes_with_the_add_and_leaves_with_the_model() {
        let monitor = KvEventMonitor::new();
        let mut spec = WorkerSpec::new("grpc://127.0.0.1:7");
        spec.kv_block_size = Some(32);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::from_spec(spec)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        assert_eq!(
            monitor.block_size(UNKNOWN_MODEL_ID),
            Some(32),
            "the add seeds the size from the spec"
        );
        monitor.on_worker_removed(worker.url()).await;
        assert!(
            monitor.block_size(UNKNOWN_MODEL_ID).is_none(),
            "the size left with the model's last events subscription"
        );
    }

    #[tokio::test]
    async fn stop_forgets_the_block_sizes() {
        let monitor = KvEventMonitor::new();
        monitor.set_block_size("llama", 16);
        assert_eq!(monitor.block_size("llama"), Some(16));

        monitor.stop().await;
        assert!(monitor.block_size("llama").is_none());
    }

    /// The index-shape gauges come from the index's own counters, by model:
    /// memberships and entries, runs, blocks, arena and slab bytes, moved
    /// hashes and engine conflicts.
    #[test]
    fn index_shape_gauges_follow_the_index() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn gauge(handle: &PrometheusHandle, name: &str, model: &str) -> Option<f64> {
            let prefix = format!("{name}{{model=\"{model}\"}}");
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&prefix))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let monitor = KvEventMonitor::new();
            let index = Arc::new(KvIndex::chain());
            let worker = index.intern_worker("grpc://w1:9000").unwrap();
            let mut held = WorkerBlocks::default();
            let blocks: Vec<StoredBlock> = (1..=3u64)
                .map(|i| StoredBlock {
                    seq_hash: SequenceHash(i),
                    content_hash: ContentHash(100 + i),
                })
                .collect();
            index
                .apply_stored(worker, &blocks, None, &mut held)
                .unwrap();
            monitor.indexers.insert("chain".to_string(), index);
            KvEventMonitor::publish_stats(&monitor.indexers);
            assert_eq!(
                gauge(&handle, "smg_kv_index_memberships", "chain"),
                Some(3.0)
            );
            assert_eq!(gauge(&handle, "smg_kv_index_entries", "chain"), Some(3.0));
            assert_eq!(gauge(&handle, "smg_kv_index_runs_live", "chain"), Some(1.0));
            assert_eq!(
                gauge(&handle, "smg_kv_index_blocks_live", "chain"),
                Some(3.0)
            );
            assert_eq!(
                gauge(&handle, "smg_kv_index_moved_hashes", "chain"),
                Some(0.0)
            );
            assert_eq!(
                gauge(&handle, "smg_kv_index_engine_conflicts", "chain"),
                Some(0.0)
            );
            assert_eq!(
                gauge(&handle, "smg_kv_index_cap_refusals", "chain"),
                Some(0.0)
            );
            assert!(gauge(&handle, "smg_kv_index_arena_bytes", "chain").is_some_and(|b| b > 0.0));
            // The arena holds whole chunks: what it holds is at least what it handed out.
            let handed_out = gauge(&handle, "smg_kv_index_arena_bytes", "chain").expect("arena");
            assert!(gauge(&handle, "smg_kv_index_arena_chunk_bytes", "chain")
                .is_some_and(|held| held >= handed_out));
            assert!(gauge(&handle, "smg_kv_index_slab_bytes", "chain").is_some_and(|b| b > 0.0));
        });
    }

    /// A worker's integrity series and its model's index gauges exist from
    /// the subscription's start, at zero: a scrape of a fresh gateway tells
    /// "nothing happened" from "nothing exported".
    #[test]
    fn a_new_subscription_publishes_its_series_at_zero() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        fn sample(body: &str, family: &str, labels: &[(&str, &str)]) -> Option<f64> {
            let prefix = format!("{family}{{");
            body.lines()
                .find(|line| {
                    line.starts_with(&prefix)
                        && labels
                            .iter()
                            .all(|(key, value)| line.contains(&format!("{key}=\"{value}\"")))
                })
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async {
                let monitor = KvEventMonitor::new();
                let worker = refusing_grpc_worker();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                let body = handle.render();
                let url = worker.url();
                for (family, labels) in [
                    (
                        "smg_kv_event_gaps_total",
                        vec![("worker", url), ("outcome", "replay_requested")],
                    ),
                    (
                        "smg_kv_event_gaps_total",
                        vec![("worker", url), ("outcome", "unrecovered_cleared")],
                    ),
                    ("smg_kv_event_missed_batches_total", vec![("worker", url)]),
                    (
                        "smg_kv_event_resyncs_total",
                        vec![("worker", url), ("reason", "snapshot")],
                    ),
                    (
                        "smg_kv_event_resyncs_total",
                        vec![("worker", url), ("reason", "publisher_restart")],
                    ),
                    (
                        "smg_kv_event_subscription_failures_total",
                        vec![("worker", url), ("reason", "panic")],
                    ),
                    (
                        "smg_kv_event_parentless_stores_total",
                        vec![("worker", url)],
                    ),
                    (
                        "smg_kv_event_parentless_blocks_total",
                        vec![("worker", url)],
                    ),
                    ("smg_kv_event_sub_block_stores_total", vec![("worker", url)]),
                    ("smg_kv_event_sub_block_blocks_total", vec![("worker", url)]),
                    (
                        "smg_kv_event_cap_refused_stores_total",
                        vec![("worker", url)],
                    ),
                    (
                        "smg_kv_event_cap_refused_blocks_total",
                        vec![("worker", url)],
                    ),
                    ("smg_kv_event_degraded_ranks", vec![("worker", url)]),
                    (
                        "smg_kv_index_moved_hashes",
                        vec![("model", UNKNOWN_MODEL_ID)],
                    ),
                    (
                        "smg_kv_index_engine_conflicts",
                        vec![("model", UNKNOWN_MODEL_ID)],
                    ),
                    (
                        "smg_kv_index_cap_refusals",
                        vec![("model", UNKNOWN_MODEL_ID)],
                    ),
                    (
                        "smg_kv_index_memberships",
                        vec![("model", UNKNOWN_MODEL_ID)],
                    ),
                ] {
                    assert_eq!(
                        sample(&body, family, &labels),
                        Some(0.0),
                        "{family} {labels:?} on a fresh gateway:\n{body}"
                    );
                }
                monitor.stop().await;
            });
        });
    }

    /// A model's index goes with its last subscription, and nothing publishes
    /// the model's gauges after that: the removal itself zeroes them, or they
    /// keep the removed worker's count until a worker of the model joins
    /// again (an engine replaced behind a one-engine gateway).
    #[test]
    fn the_models_last_removal_zeroes_its_index_gauges() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn gauge(handle: &PrometheusHandle, name: &str, model: &str) -> Option<f64> {
            let prefix = format!("{name}{{model=\"{model}\"}}");
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&prefix))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        // The removal task publishes from the runtime's thread, which must
        // be the one holding the local recorder.
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let monitor = Arc::new(KvEventMonitor::new());
                let worker = refusing_grpc_worker();
                let url = worker.url().to_string();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                let index = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                wait_until("the subscription interns its worker", || {
                    index.worker_id(&url).is_some()
                })
                .await;
                let worker_id = index.worker_id(&url).unwrap();
                let blocks: Vec<StoredBlock> = (1..=3u64)
                    .map(|i| StoredBlock {
                        seq_hash: SequenceHash(i),
                        content_hash: ContentHash(100 + i),
                    })
                    .collect();
                index
                    .apply_stored(worker_id, &blocks, None, &mut WorkerBlocks::default())
                    .unwrap();
                KvEventMonitor::publish_stats(&monitor.indexers);
                assert_eq!(
                    gauge(&handle, "smg_kv_index_memberships", UNKNOWN_MODEL_ID),
                    Some(3.0)
                );
                assert_eq!(
                    gauge(&handle, "smg_kv_index_blocks_live", UNKNOWN_MODEL_ID),
                    Some(3.0)
                );

                monitor.on_worker_removed(&url).await;
                assert!(
                    monitor.get_indexer(UNKNOWN_MODEL_ID).is_none(),
                    "the model's last worker takes its index with it"
                );
                for name in [
                    "smg_kv_index_memberships",
                    "smg_kv_index_entries",
                    "smg_kv_index_blocks_live",
                    "smg_kv_index_runs_live",
                    "smg_kv_index_arena_bytes",
                    "smg_kv_index_arena_chunk_bytes",
                    "smg_kv_index_slab_bytes",
                    "smg_kv_block_size_tokens",
                ] {
                    assert_eq!(gauge(&handle, name, UNKNOWN_MODEL_ID), Some(0.0), "{name}");
                }
            });
        });
    }

    #[test]
    fn start_stats_task_holds_the_task() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.stats_task.lock().is_none());
        monitor.start_stats_task();
        assert!(monitor.stats_task.lock().is_some());
    }
}
