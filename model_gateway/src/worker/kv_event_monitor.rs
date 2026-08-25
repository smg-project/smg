//! Per-worker KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per gRPC worker that subscribes
//! to KV cache events and feeds them into a shared `PositionalIndexer` (one per model).
//! This enables event-driven cache-aware routing as an alternative to the approximate
//! radix tree approach.
//!
//! Lifecycle:
//! - `on_worker_added` — spawns streaming task, creates indexer if needed
//! - `on_worker_removed` — signals graceful shutdown, task cleans up indexer
//! - `stop` — signals shutdown to all tasks, clears state

use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use dashmap::DashMap;
use futures::FutureExt as _;
use kv_index::{
    compute_content_hash, ApplyError, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap,
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvEventBatch,
};
use tokio::{
    sync::{oneshot, Mutex, Semaphore},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    observability::metrics::Metrics,
    policies::utils::PeriodicTask,
    worker::{ConnectionMode, Worker, UNKNOWN_MODEL_ID},
};

/// Default jump size for new `PositionalIndexer` instances.
const DEFAULT_JUMP_SIZE: usize = 64;

/// Interval between positional-indexer prune cycles (matches the routing
/// policies' default eviction cadence).
const PRUNE_INTERVAL_SECS: u64 = 30;

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Maximum reconnection delay (caps exponential backoff).
const MAX_RECONNECT_DELAY_MS: u64 = 30_000;

/// Positional-index cleanup is CPU-bound and can touch many blocks. Keep it
/// off Tokio workers and bound concurrent purges during fleet-wide drains.
const MAX_CONCURRENT_INDEX_REMOVALS: usize = 4;
static INDEX_REMOVAL_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_INDEX_REMOVALS);

/// Manages per-worker KV cache event subscriptions.
///
/// Each gRPC worker gets a dedicated tokio task that subscribes to the backend's
/// KV cache event stream and feeds events into a shared `PositionalIndexer`
/// (one per `model_id`). Workers serving the same model share the same indexer.
pub struct KvEventMonitor {
    /// Per-model positional indexers: model_id → shared indexer.
    /// Arc-wrapped so the prune task can share the map WITHOUT holding (even
    /// weakly) the monitor itself: `PeriodicTask` joins its thread on drop, so
    /// a task that could ever own the last monitor reference would run the
    /// monitor's drop — and thus its own join — on its own thread.
    pub(crate) indexers: Arc<DashMap<String, Arc<PositionalIndexer>>>,
    /// Per-model block sizes learned from KV events or set via WorkerSpec.
    /// Used by CacheAwarePolicy to chunk request tokens at query time.
    /// Arc-wrapped so subscription tasks can update it from events.
    block_sizes: Arc<DashMap<String, usize>>,
    /// Per-worker subscription handles: worker_url → subscription info.
    /// Mutex matches LoadMonitor pattern for atomic abort + remove.
    worker_handles: Mutex<HashMap<String, WorkerSubscription>>,
    /// Jump size for new PositionalIndexer instances.
    jump_size: usize,
    /// Periodic indexer prune, held so it aborts when the monitor drops.
    /// Set once by [`start_prune_task`](Self::start_prune_task); sync mutex
    /// because it is touched only at startup, never on event paths.
    prune_task: parking_lot::Mutex<Option<PeriodicTask>>,
}

/// Tracks a single worker's subscription state.
struct WorkerSubscription {
    handle: JoinHandle<()>,
    model_id: String,
    /// Signals the subscription task to shut down gracefully.
    /// The task owns its `WorkerBlockMap` and cleans up the indexer on exit.
    shutdown_tx: oneshot::Sender<()>,
}

/// Result of processing a stream connection to completion.
enum StreamResult {
    /// Stream closed normally (server-side).
    Ended,
    /// Stream produced an error.
    Error(tonic::Status),
    /// Detected a gap in sequence numbers.
    GapDetected { expected: u64, received: u64 },
}

impl KvEventMonitor {
    /// Create a new `KvEventMonitor`.
    ///
    /// `jump_size` controls the `PositionalIndexer` jump search stride.
    /// Pass `None` for the default (64).
    pub fn new(jump_size: Option<usize>) -> Self {
        let jump_size = jump_size.unwrap_or(DEFAULT_JUMP_SIZE).max(1);
        Self {
            indexers: Arc::new(DashMap::new()),
            block_sizes: Arc::new(DashMap::new()),
            worker_handles: Mutex::new(HashMap::new()),
            jump_size,
            prune_task: parking_lot::Mutex::new(None),
        }
    }

    /// Prune every model's positional indexer with the given bounds.
    /// `ttl_secs`/`max_entries` of 0 disable the respective pass — see
    /// [`PositionalIndexer::prune`].
    pub fn prune_all(&self, ttl_secs: u64, max_entries: usize) {
        Self::prune_indexers(&self.indexers, ttl_secs, max_entries);
    }

    fn prune_indexers(
        indexers: &DashMap<String, Arc<PositionalIndexer>>,
        ttl_secs: u64,
        max_entries: usize,
    ) {
        let ttl = u32::try_from(ttl_secs).unwrap_or(u32::MAX - 1);
        let ttl = (ttl > 0).then_some(ttl);
        let max = (max_entries > 0).then_some(max_entries);
        if ttl.is_none() && max.is_none() {
            return;
        }
        for entry in indexers {
            let stats = entry.value().prune(ttl, max);
            if stats.evicted_ttl + stats.evicted_capacity > 0 {
                info!(
                    model_id = %entry.key(),
                    evicted_ttl = stats.evicted_ttl,
                    evicted_capacity = stats.evicted_capacity,
                    remaining = stats.remaining,
                    "Pruned positional indexer"
                );
            }
        }
    }

    /// Start the periodic indexer prune. No-op when both bounds are 0/unset.
    /// The task shares only the indexer map — never a reference to the monitor
    /// itself — so it can never be the one to run the monitor's drop (and with
    /// it its own thread's join; see the `indexers` field docs). The handle is
    /// held by the monitor, so the task stops when the monitor drops.
    pub fn start_prune_task(&self, ttl_secs: u64, max_entries: usize) {
        if ttl_secs == 0 && max_entries == 0 {
            return;
        }
        let indexers = Arc::clone(&self.indexers);
        let task = PeriodicTask::spawn(PRUNE_INTERVAL_SECS, "KvIndexerPrune", move || {
            Self::prune_indexers(&indexers, ttl_secs, max_entries);
        });
        *self.prune_task.lock() = Some(task);
        info!(
            ttl_secs,
            max_entries,
            interval_secs = PRUNE_INTERVAL_SECS,
            "Started positional-indexer prune task"
        );
    }

    /// Start a KV event subscription for a worker.
    ///
    /// Spawns a background tokio task that subscribes to KV cache events via
    /// server-streaming gRPC and applies them to the model's `PositionalIndexer`.
    /// Duplicate calls for the same worker URL are no-ops.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url().to_string();
        // Normalize model_id to match routing's normalize_model_key — empty → "unknown".
        let model_id = Self::normalize_model_id(worker.model_id());

        // Only gRPC workers stream KV events. HTTP proxies don't, and a ZMQ
        // EngineCore returns `unimplemented` for SubscribeKvEvents — subscribing
        // there would just be a wasted handshake + round-trip per registration.
        if *worker.connection_mode() != ConnectionMode::Grpc {
            debug!(worker_url = %url, mode = %worker.connection_mode(), "non-gRPC worker, skipping KV event subscription");
            return;
        }

        let mut handles = self.worker_handles.lock().await;
        if handles.contains_key(&url) {
            debug!(worker_url = %url, "KV event subscription already active, skipping");
            return;
        }

        let indexer = self
            .indexers
            .entry(model_id.clone())
            .or_insert_with(|| Arc::new(PositionalIndexer::new(self.jump_size)))
            .clone();
        // Seed block_size provisionally from WorkerSpec. The event stream will
        // overwrite this with the backend's actual page size once received.
        if let Some(bs) = worker.metadata().spec.kv_block_size {
            if bs > 0 {
                self.block_sizes.entry(model_id.clone()).or_insert(bs);
            } else {
                warn!(worker_url = %url, "Worker reports kv_block_size=0, ignoring");
            }
        }

        let worker = Arc::clone(worker);
        let worker_url = url.clone();
        let block_sizes = Arc::clone(&self.block_sizes);

        info!(
            worker_url = %url,
            model_id = %model_id,
            "Starting KV event subscription"
        );

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let loop_model_id = model_id.clone();
        let task_url = url.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "KV event monitor: runs for the lifetime of the worker, \
                      handle is stored and graceful shutdown is sent on removal"
        )]
        let handle = tokio::spawn(async move {
            // Catch panics here so they surface when they happen — a bare
            // JoinError would only be observed at worker removal, leaving the
            // index silently frozen for this worker until then.
            let result = std::panic::AssertUnwindSafe(Self::subscription_loop(
                worker,
                worker_url,
                indexer,
                block_sizes,
                loop_model_id,
                shutdown_rx,
            ))
            .catch_unwind()
            .await;
            if let Err(payload) = result {
                let msg = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .map(String::from)
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "(non-string panic)".into());
                error!(
                    worker_url = %task_url,
                    panic.message = %msg,
                    "KV event subscription task panicked; KV events from this \
                     worker no longer feed cache-aware routing"
                );
                Metrics::record_kv_event_subscription_failure(&task_url, "panic");
            }
        });

        handles.insert(
            url,
            WorkerSubscription {
                handle,
                model_id,
                shutdown_tx,
            },
        );
    }

    /// Stop the KV event subscription for a worker.
    ///
    /// Sends a graceful shutdown signal. The subscription task cleans up its
    /// own `WorkerBlockMap` using the indexer's per-worker reverse map; that
    /// CPU-bound cleanup runs on the bounded blocking pool rather than a Tokio
    /// runtime worker.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        let subscription = {
            let mut handles = self.worker_handles.lock().await;
            handles.remove(worker_url)
        };

        let Some(sub) = subscription else {
            return;
        };
        info!(worker_url = %worker_url, "Stopping KV event subscription");
        // Signal graceful shutdown — task cleans up its worker_blocks in the indexer.
        let _ = sub.shutdown_tx.send(());
        // Panics are caught inside the task; a JoinError here (abort or a
        // panic that escaped the guard) must still be surfaced, not discarded.
        if let Err(e) = sub.handle.await {
            error!(
                worker_url = %worker_url,
                error = %e,
                "KV event subscription task failed"
            );
            Metrics::record_kv_event_subscription_failure(worker_url, "join_error");
        }

        // Re-check under lock whether this was the last worker for the model.
        // Must re-acquire lock after shutdown to avoid TOCTOU with concurrent
        // on_worker_added that may have added a new worker for the same model
        // between our first lock release and this point.
        let should_remove_indexer = {
            let handles = self.worker_handles.lock().await;
            !handles.values().any(|other| other.model_id == sub.model_id)
        };

        if should_remove_indexer {
            self.indexers.remove(&sub.model_id);
            self.block_sizes.remove(&sub.model_id);
        }
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        let subscriptions: HashMap<String, WorkerSubscription> = {
            let mut handles = self.worker_handles.lock().await;
            std::mem::take(&mut *handles)
        };

        if !subscriptions.is_empty() {
            info!(
                count = subscriptions.len(),
                "Stopping all KV event subscriptions"
            );
            for (url, sub) in subscriptions {
                debug!(worker_url = %url, "Stopping KV event subscription");
                let _ = sub.shutdown_tx.send(());
                if let Err(e) = sub.handle.await {
                    error!(
                        worker_url = %url,
                        error = %e,
                        "KV event subscription task failed"
                    );
                    Metrics::record_kv_event_subscription_failure(&url, "join_error");
                }
            }
        }

        self.indexers.clear();
        self.block_sizes.clear();
    }

    /// Get the indexer for a model (used by `CacheAwarePolicy` for queries).
    pub fn get_indexer(&self, model_id: &str) -> Option<Arc<PositionalIndexer>> {
        self.indexers.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the block size for a model (learned from events or set via `set_block_size`).
    pub fn block_size(&self, model_id: &str) -> Option<usize> {
        self.block_sizes.get(model_id).map(|v| *v)
    }

    /// Set the block size for a model (e.g. from WorkerSpec during registration).
    /// Does not overwrite a value already learned from events.
    pub fn set_block_size(&self, model_id: &str, block_size: usize) {
        self.block_sizes
            .entry(model_id.to_string())
            .or_insert(block_size);
    }

    /// Check if any subscription is running.
    pub async fn is_running(&self) -> bool {
        !self.worker_handles.lock().await.is_empty()
    }

    /// Normalize model_id to match routing's `normalize_model_key`.
    /// Empty model IDs map to UNKNOWN_MODEL_ID for consistent keying.
    fn normalize_model_id(model_id: &str) -> String {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            model_id.to_string()
        }
    }

    // -----------------------------------------------------------------------
    // Subscription loop
    // -----------------------------------------------------------------------

    /// Learn `block_size` from the first `KvBlock` in a stored event.
    ///
    /// Called once per model when the first stored event arrives, providing
    /// ground truth from the backend. `CacheAwarePolicy` uses this to chunk
    /// request tokens into blocks for overlap scoring.
    ///
    /// Overwrites any provisional value seeded from `WorkerSpec` since the
    /// event stream reflects the backend's actual page size.
    fn learn_block_size(
        block_sizes: &DashMap<String, usize>,
        model_id: &str,
        learned: &mut bool,
        batch: &KvEventBatch,
    ) {
        if *learned {
            return;
        }
        for event in &batch.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if let Some(block) = stored.blocks.first() {
                    if block.block_size > 0 {
                        let bs = block.block_size as usize;
                        block_sizes.insert(model_id.to_string(), bs);
                        info!(
                            model_id = %model_id,
                            block_size = bs,
                            "Learned block_size from KV event"
                        );
                        *learned = true;
                        return;
                    }
                }
            }
        }
    }

    /// Main subscription loop for a single worker.
    ///
    /// Owns the `WorkerBlockMap` for this worker and cleans it up on exit.
    /// Exits when `shutdown_rx` fires or the backend returns `Unimplemented`.
    async fn remove_indexer_worker(
        indexer: Arc<PositionalIndexer>,
        worker_id: u32,
        worker_blocks: WorkerBlockMap,
    ) {
        let Ok(permit) = INDEX_REMOVAL_PERMITS.acquire().await else {
            error!(worker_id, "Positional-index cleanup semaphore closed");
            return;
        };
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            indexer.remove_worker(worker_id, worker_blocks);
        })
        .await;

        if let Err(error) = result {
            error!(worker_id, %error, "Positional-index worker cleanup task failed");
        }
    }

    async fn subscription_loop(
        worker: Arc<dyn Worker>,
        worker_url: String,
        indexer: Arc<PositionalIndexer>,
        block_sizes: Arc<DashMap<String, usize>>,
        model_id: String,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        let worker_id = match indexer.intern_worker(&worker_url) {
            Ok(id) => id,
            Err(e) => {
                error!(
                    worker_url = %worker_url,
                    error = %e,
                    "Failed to intern worker; KV events from this worker will \
                     not feed cache-aware routing"
                );
                Metrics::record_kv_event_subscription_failure(&worker_url, "intern_failed");
                return;
            }
        };
        let mut worker_blocks = WorkerBlockMap::default();
        let mut last_seq: u64 = 0;
        let mut reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
        let mut block_size_learned = false;

        /// Sleep with shutdown check. Returns `true` if shutdown was signaled.
        macro_rules! sleep_or_shutdown {
            ($delay:expr, $rx:expr) => {
                tokio::select! {
                    _ = tokio::time::sleep($delay) => false,
                    _ = &mut *$rx => true,
                }
            };
        }

        loop {
            let backend_client = match worker.get_backend_client().await {
                Ok(Some(client)) => client,
                Ok(None) => {
                    // HTTP workers are filtered in on_worker_added, so this should
                    // be unreachable. Retry defensively rather than exiting and
                    // leaving a stale entry in worker_handles.
                    warn!(
                        worker_url = %worker_url,
                        delay_ms = reconnect_delay_ms,
                        "Worker has no backend client yet, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to get backend client, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let stream = match backend_client.subscribe_kv_events(last_seq).await {
                Ok(stream) => {
                    info!(
                        worker_url = %worker_url,
                        start_seq = last_seq,
                        "KV event stream connected"
                    );
                    reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                    stream
                }
                Err(e) => {
                    // If the backend doesn't implement SubscribeKvEvents (e.g. vLLM),
                    // stop retrying — this RPC will never succeed.
                    if e.code() == tonic::Code::Unimplemented {
                        warn!(
                            worker_url = %worker_url,
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this worker"
                        );
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    if e.code() == tonic::Code::OutOfRange {
                        warn!(
                            worker_url = %worker_url,
                            last_seq = last_seq,
                            "KV event replay cursor expired; clearing worker state and requesting a current snapshot"
                        );
                        indexer.apply_cleared(worker_id, &mut worker_blocks);
                        last_seq = 0;
                        reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                        continue;
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to subscribe to KV events, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let on_batch = |batch: &KvEventBatch| {
                Self::learn_block_size(&block_sizes, &model_id, &mut block_size_learned, batch);
            };
            let stream_result = tokio::select! {
                result = Self::process_stream(
                    stream, &worker_url, worker_id, &indexer,
                    &mut worker_blocks, &mut last_seq, on_batch,
                ) => result,
                _ = &mut shutdown_rx => {
                    Self::remove_indexer_worker(
                        Arc::clone(&indexer),
                        worker_id,
                        worker_blocks,
                    )
                    .await;
                    return;
                }
            };

            match stream_result {
                StreamResult::Ended => {
                    info!(
                        worker_url = %worker_url,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream ended, reconnecting"
                    );
                    // Backoff to avoid tight reconnect loop if server keeps
                    // closing the stream cleanly (e.g., rolling connections).
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
                StreamResult::Error(e) => {
                    if e.code() == tonic::Code::DataLoss {
                        warn!(
                            worker_url = %worker_url,
                            error = %e,
                            last_seq = last_seq,
                            "KV event subscriber fell behind; clearing worker state and requesting a current snapshot"
                        );
                        indexer.apply_cleared(worker_id, &mut worker_blocks);
                        last_seq = 0;
                        reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                        continue;
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream error, reconnecting"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks)
                            .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
                StreamResult::GapDetected { expected, received } => {
                    warn!(
                        worker_url = %worker_url,
                        expected = expected,
                        received = received,
                        "Sequence gap detected, reconnecting for replay from seq {last_seq}"
                    );
                    // No backoff — gap replay is a normal recovery path.
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stream processing + proto conversion
    // -----------------------------------------------------------------------

    /// Process batches from a single stream connection.
    async fn process_stream(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
        last_seq: &mut u64,
        mut on_batch: impl FnMut(&KvEventBatch),
    ) -> StreamResult {
        use tokio_stream::StreamExt;

        while let Some(result) = stream.next().await {
            let batch = match result {
                Ok(batch) => batch,
                Err(e) => return StreamResult::Error(e),
            };

            // Skip stale/duplicate batches (can occur after reconnect replay).
            if *last_seq > 0 && batch.sequence_number <= *last_seq {
                debug!(
                    worker_url = %worker_url,
                    last_seq = *last_seq,
                    received = batch.sequence_number,
                    "Skipping stale KV event batch"
                );
                continue;
            }

            // Gap detection.
            if *last_seq > 0 && batch.sequence_number > *last_seq + 1 {
                return StreamResult::GapDetected {
                    expected: *last_seq + 1,
                    received: batch.sequence_number,
                };
            }

            on_batch(&batch);

            for event in &batch.events {
                Self::apply_event(event, worker_id, indexer, worker_blocks);
            }

            *last_seq = batch.sequence_number;
        }

        StreamResult::Ended
    }

    /// Apply a single KV cache event to the indexer.
    fn apply_event(
        event: &KvCacheEvent,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let Some(ref data) = event.data else {
            return;
        };

        match data {
            kv_cache_event::Data::Stored(stored) => {
                Self::apply_stored(stored, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Removed(removed) => {
                Self::apply_removed(removed, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Cleared(_) => {
                indexer.apply_cleared(worker_id, worker_blocks);
            }
        }
    }

    /// Convert proto `KvBlocksStored` and apply to the indexer.
    fn apply_stored(
        stored: &KvBlocksStored,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let blocks: Vec<StoredBlock> = stored.blocks.iter().map(convert_kv_block).collect();

        let parent_seq_hash = stored.parent_block_hash.map(SequenceHash::from);

        match indexer.apply_stored(worker_id, &blocks, parent_seq_hash, worker_blocks) {
            Ok(()) => {}
            Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) => {
                // Cold start or parent evicted — retry without parent to start a new chain.
                if let Err(e) = indexer.apply_stored(worker_id, &blocks, None, worker_blocks) {
                    warn!(
                        worker_id = worker_id,
                        error = %e,
                        "Failed to apply stored event after fallback"
                    );
                }
            }
        }
    }

    /// Convert proto `KvBlocksRemoved` and apply to the indexer.
    fn apply_removed(
        removed: &KvBlocksRemoved,
        worker_id: u32,
        indexer: &PositionalIndexer,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let seq_hashes: Vec<SequenceHash> = removed
            .block_hashes
            .iter()
            .map(|&h| SequenceHash::from(h))
            .collect();

        indexer.apply_removed(worker_id, &seq_hashes, worker_blocks);
    }
}

/// Convert a proto `KvBlock` to a kv-index `StoredBlock`.
fn convert_kv_block(block: &KvBlock) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: compute_content_hash(&block.token_ids),
    }
}

impl Drop for KvEventMonitor {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.worker_handles.try_lock() {
            for (_, sub) in handles.drain() {
                let _ = sub.shutdown_tx.send(());
                sub.handle.abort(); // Can't await in Drop, abort as fallback
            }
        }
    }
}

impl fmt::Debug for KvEventMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvEventMonitor")
            .field("models", &self.indexers.len())
            .field("block_sizes", &self.block_sizes.len())
            .field("jump_size", &self.jump_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Proto → kv-index conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_kv_block() {
        let block = KvBlock {
            block_hash: 42,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_convert_kv_block_negative_hash() {
        let block = KvBlock {
            block_hash: -1,
            token_ids: vec![10, 20],
            block_size: 2,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
    }

    #[test]
    fn test_convert_kv_block_empty_tokens() {
        let block = KvBlock {
            block_hash: 100,
            token_ids: vec![],
            block_size: 0,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash::from(100i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[]));
    }

    // -----------------------------------------------------------------------
    // apply_event integration with PositionalIndexer
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_stored_no_parent() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
            ],
            parent_block_hash: None,
        };

        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_apply_stored_with_parent() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored1 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored1, w1, &indexer, &mut wb);

        let stored2 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 2,
                token_ids: vec![50, 60, 70, 80],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: Some(1),
        };
        KvEventMonitor::apply_stored(&stored2, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_apply_stored_fallback_on_worker_not_tracked() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://new-worker:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        // Pass parent_block_hash for an untracked worker — should fallback to no parent.
        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: Some(999),
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_removed() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                },
            ],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);

        let removed = KvBlocksRemoved {
            block_hashes: vec![2],
            cache_level: None,
        };
        KvEventMonitor::apply_removed(&removed, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_cleared_event() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
            }],
            parent_block_hash: None,
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        indexer.apply_cleared(w1, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_apply_event_dispatch_stored() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 42,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                }],
                parent_block_hash: None,
            })),
        };

        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_event_dispatch_removed() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        let stored_event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 1,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                }],
                parent_block_hash: None,
            })),
        };
        KvEventMonitor::apply_event(&stored_event, w1, &indexer, &mut wb);

        let removed_event = KvCacheEvent {
            event_id: 2,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: vec![1],
                cache_level: None,
            })),
        };
        KvEventMonitor::apply_event(&removed_event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_apply_event_dispatch_cleared() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();

        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 1,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 1,
                        token_ids: vec![1, 2, 3, 4],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: None,
                })),
            },
            w1,
            &indexer,
            &mut wb,
        );

        // Clear
        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 2,
                data: Some(kv_cache_event::Data::Cleared(
                    smg_grpc_client::common_proto::KvCacheCleared {},
                )),
            },
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_apply_event_no_data() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: None,
        };
        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_monitor_new() {
        let monitor = KvEventMonitor::new(None);
        assert!(!monitor.is_running().await);
    }

    #[tokio::test]
    async fn test_monitor_new_clamps_zero_jump_size() {
        let monitor = KvEventMonitor::new(Some(0));
        assert_eq!(monitor.jump_size, 1);
    }

    #[tokio::test]
    async fn test_get_indexer_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        assert!(monitor.get_indexer("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_stop_empty_monitor() {
        let monitor = KvEventMonitor::new(None);
        monitor.stop().await;
    }

    #[tokio::test]
    async fn test_on_worker_removed_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        monitor.on_worker_removed("http://nonexistent:8000").await;
    }

    #[tokio::test]
    async fn test_remove_indexer_worker_runs_cleanup_off_runtime() {
        let indexer = Arc::new(PositionalIndexer::new(64));
        let worker_id = indexer.intern_worker("http://w1:8000").unwrap();
        let mut worker_blocks = WorkerBlockMap::default();
        indexer
            .apply_stored(
                worker_id,
                &[StoredBlock {
                    seq_hash: SequenceHash(1),
                    content_hash: compute_content_hash(&[1, 2, 3]),
                }],
                None,
                &mut worker_blocks,
            )
            .unwrap();

        KvEventMonitor::remove_indexer_worker(Arc::clone(&indexer), worker_id, worker_blocks).await;

        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // block_size learning
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_block_size() {
        let monitor = KvEventMonitor::new(None);

        // Initially no block_size
        assert!(monitor.block_size("llama").is_none());

        // Set it
        monitor.set_block_size("llama", 32);
        assert_eq!(monitor.block_size("llama"), Some(32));

        // set_block_size doesn't overwrite existing value
        monitor.set_block_size("llama", 64);
        assert_eq!(monitor.block_size("llama"), Some(32));
    }

    #[tokio::test]
    async fn test_stop_clears_block_sizes() {
        let monitor = KvEventMonitor::new(None);
        monitor.set_block_size("llama", 16);
        assert_eq!(monitor.block_size("llama"), Some(16));

        monitor.stop().await;
        assert!(monitor.block_size("llama").is_none());
    }

    #[test]
    fn test_prune_all_enforces_capacity_ceiling() {
        let monitor = KvEventMonitor::new(None);
        let indexer = monitor
            .indexers
            .entry("llama".to_string())
            .or_insert_with(|| Arc::new(PositionalIndexer::new(64)))
            .clone();

        let worker = indexer.intern_worker("http://w1:8000").unwrap();
        let mut worker_blocks = WorkerBlockMap::default();
        // Ten independent single-block chains → ten index entries.
        for i in 0u64..10 {
            let block = StoredBlock {
                seq_hash: SequenceHash(1000 + i),
                content_hash: kv_index::ContentHash(2000 + i),
            };
            indexer
                .apply_stored(worker, &[block], None, &mut worker_blocks)
                .unwrap();
        }
        assert_eq!(indexer.entry_count(), 10);

        // Disabled bounds → no-op.
        monitor.prune_all(0, 0);
        assert_eq!(indexer.entry_count(), 10);

        // Freshly stored entries sit inside the capacity-eviction grace and
        // are spared — a prune must not race a store batch's accounting.
        monitor.prune_all(0, 5);
        assert_eq!(indexer.entry_count(), 10);

        // Age them past the grace (the indexer's test clock is crate-private
        // to kv_index, so this test uses the real clock) and the ceiling is
        // enforced down to the low-water mark (5 - 5/10 = 5).
        std::thread::sleep(Duration::from_secs(3));
        monitor.prune_all(0, 5);
        assert_eq!(indexer.entry_count(), 5);
    }

    #[tokio::test]
    async fn test_start_prune_task_noop_when_disabled() {
        let monitor = Arc::new(KvEventMonitor::new(None));
        monitor.start_prune_task(0, 0);
        assert!(monitor.prune_task.lock().is_none());

        monitor.start_prune_task(60, 0);
        assert!(monitor.prune_task.lock().is_some());
    }
}
