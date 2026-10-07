//! Per-worker KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per gRPC worker that subscribes
//! to KV cache events and feeds them into a shared [`KvIndex`] (one per model:
//! the chain index).
//! This enables event-driven cache-aware routing as an alternative to the approximate
//! radix tree approach.
//!
//! Lifecycle:
//! - `on_worker_added` — spawns streaming task, creates indexer if needed
//! - `on_worker_removed` — signals graceful shutdown, task cleans up indexer
//! - `stop` — signals shutdown to all tasks, clears state
//!
//! The work of a subscription is split by stage: `subscription.rs` runs the
//! task per worker (connecting, reconnecting, the stream read loop, the
//! pushed load records, the worker's departure); `admission.rs` keeps one
//! stream's state (per-rank cursors, gap recovery, relay snapshots, and the
//! metrics around an applied batch); `apply.rs` takes one event into the
//! index (tiers, cache groups, namespaces, and the physical copies of a
//! block, counted per worker).

use std::{
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock, Weak,
    },
};

use dashmap::DashMap;
use futures::FutureExt as _;
use tokio::{
    sync::{oneshot, watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use super::{kv_index_backend::KvIndex, monitor::WorkerMonitor};
use crate::{
    observability::metrics::Metrics,
    policies::utils::PeriodicTask,
    worker::{ConnectionMode, Worker, UNKNOWN_MODEL_ID},
};

mod admission;
mod apply;
mod subscription;

pub(crate) use apply::WorkerIndexState;

/// Interval between publications of each model's index size and, for the chain
/// index, its shape and memory (`smg_kv_index_*` gauges by model).
const STATS_INTERVAL_SECS: u64 = 30;

/// Manages per-worker KV cache event subscriptions.
///
/// Each gRPC worker gets a dedicated tokio task that subscribes to the backend's
/// KV cache event stream and feeds events into a shared [`KvIndex`]
/// (one per `model_id`). Workers serving the same model share the same indexer.
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
    block_sizes: Arc<DashMap<String, usize>>,
    /// Per-worker subscription slots: worker_url → the live subscription, or
    /// the reservation a removal leaves until the old task has cleaned up.
    /// Mutex matches LoadMonitor pattern for atomic abort + remove; Arc so a
    /// task can lift its own reservation at the end of its exit path.
    worker_handles: Arc<Mutex<HashMap<String, Slot>>>,
    /// Periodic index-shape publication, held so it aborts when the monitor
    /// drops. Set once by [`start_stats_task`](Self::start_stats_task); sync
    /// mutex because it is touched only at startup, never on event paths.
    stats_task: parking_lot::Mutex<Option<PeriodicTask>>,
    /// Where the load records on the event streams go (`KvEventBatch.load`):
    /// the worker monitor, which treats them as polls. Weak: the monitors
    /// are peers in the app context, neither owns the other.
    load_sink: OnceLock<Weak<WorkerMonitor>>,
}

/// A worker URL's slot in the subscription map.
enum Slot {
    Active(WorkerSubscription),
    /// The URL's old task has been told to shut down and is taking its
    /// blocks out of the index. The index hands a URL its existing id until
    /// that cleanup has released it, so a subscription interned in this
    /// window would write under the old id and lose its blocks to the
    /// cleanup; `on_worker_added` waits for the old task to end instead.
    Removing(Reservation),
}

/// What a removal leaves in a URL's slot until the old task has ended.
///
/// The task lifts it itself at the end of its exit path
/// ([`KvEventMonitor::complete_removal`]), and `done` closes when the task
/// is gone for any reason (an abort, a panic past its guard). Nothing about
/// it depends on the `on_worker_removed` future living on: the registry's
/// removal step runs it under a timeout, and a cleanup waiting on the
/// removal permits can outlast that; a reservation only the remover could
/// lift would then hold every later add of the URL for good.
struct Reservation {
    /// The subscription the reservation belongs to; a task lifts only its own.
    id: u64,
    /// Closes when the task has ended.
    done: watch::Receiver<()>,
}

/// Tracks a single worker's subscription state.
struct WorkerSubscription {
    /// Tells this subscription from a later one under the same URL.
    id: u64,
    handle: JoinHandle<()>,
    model_id: String,
    /// Signals the subscription task to shut down gracefully. The task owns
    /// the worker's index state and takes its blocks out of the index on exit.
    shutdown_tx: oneshot::Sender<()>,
    /// Closes when the task has ended; a removal moves it into the
    /// reservation it leaves in the slot.
    done: watch::Receiver<()>,
}

/// Ids for [`WorkerSubscription`]s, unique within the process.
static NEXT_SUBSCRIPTION_ID: AtomicU64 = AtomicU64::new(1);

impl KvEventMonitor {
    /// A monitor whose models get chain indexes.
    pub fn new() -> Self {
        Self {
            indexers: Arc::new(DashMap::new()),
            block_sizes: Arc::new(DashMap::new()),
            worker_handles: Arc::new(Mutex::new(HashMap::new())),
            stats_task: parking_lot::Mutex::new(None),
            load_sink: OnceLock::new(),
        }
    }

    /// Route the load records on every worker's event stream to the worker
    /// monitor (once; a later call is ignored).
    pub fn set_load_sink(&self, monitor: &Arc<WorkerMonitor>) {
        let _ = self.load_sink.set(Arc::downgrade(monitor));
    }

    /// Publish every model's index size, and the chain index's shape and
    /// memory, as gauges: what a soak reads to tell fragmentation or growth
    /// from load. `current_size` and `entry_count` are counter reads; the chain
    /// index's `stats` walks its run headers, which is why this runs on a
    /// 30 s cadence and never on a request.
    fn publish_stats(indexers: &DashMap<String, Arc<KvIndex>>) {
        for entry in indexers {
            let index = entry.value();
            Metrics::set_kv_index_size(entry.key(), index.current_size(), index.entry_count());
            if let Some(stats) = index.chain_stats() {
                Metrics::set_kv_index_chain_stats(entry.key(), &stats);
            }
        }
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

    /// Start a KV event subscription for a worker.
    ///
    /// Spawns a background tokio task that subscribes to KV cache events via
    /// server-streaming gRPC and applies them to the model's `KvIndex`.
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

        let mut handles = loop {
            let mut handles = self.worker_handles.lock().await;
            let mut done = match handles.get(&url) {
                Some(Slot::Active(_)) => {
                    debug!(worker_url = %url, "KV event subscription already active, skipping");
                    return;
                }
                Some(Slot::Removing(reservation)) => reservation.done.clone(),
                None => break handles,
            };
            // The previous subscription for this URL is still being taken
            // out of the index: intern it again only once its task has ended.
            if done.has_changed().is_err() {
                // The task is gone without lifting its reservation (aborted,
                // or a panic past its guard): nothing is left to wait for.
                handles.remove(&url);
                break handles;
            }
            drop(handles);
            // Nothing is ever sent on the channel; this returns at its close.
            let _ = done.changed().await;
        };

        let indexer = self
            .indexers
            .entry(model_id.clone())
            .or_insert_with(|| Arc::new(KvIndex::chain()))
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
        let (done_tx, done) = watch::channel(());
        let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
        let load_sink = self.load_sink.get().cloned();
        let loop_model_id = model_id.clone();
        let task_url = url.clone();
        let task_model_id = model_id.clone();
        let slots = Arc::clone(&self.worker_handles);
        let indexers = Arc::clone(&self.indexers);
        let model_block_sizes = Arc::clone(&self.block_sizes);

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
                load_sink,
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
            Self::complete_removal(
                &slots,
                &indexers,
                &model_block_sizes,
                &task_url,
                id,
                &task_model_id,
            )
            .await;
            // Closes `done` last: an add waiting on it finds the slot free.
            drop(done_tx);
        });

        handles.insert(
            url,
            Slot::Active(WorkerSubscription {
                id,
                handle,
                model_id,
                shutdown_tx,
                done,
            }),
        );
    }

    /// The end of a removal, run by the subscription task once its blocks
    /// are out of the index: lift the URL's reservation if it is this task's
    /// (a later subscription under the URL has its own), and drop the
    /// model's index with the model's last subscription. The task does this
    /// rather than `on_worker_removed`, whose future a caller may drop
    /// before the task has ended; a slot that is `Active` (the task ended on
    /// its own, or `stop` took the subscription) is left as it is.
    async fn complete_removal(
        slots: &Mutex<HashMap<String, Slot>>,
        indexers: &DashMap<String, Arc<KvIndex>>,
        block_sizes: &DashMap<String, usize>,
        worker_url: &str,
        id: u64,
        model_id: &str,
    ) {
        let mut handles = slots.lock().await;
        if !matches!(handles.get(worker_url), Some(Slot::Removing(held)) if held.id == id) {
            return;
        }
        handles.remove(worker_url);
        // Under the slot lock, which `on_worker_added` holds from its lookup
        // of the model's index to its insert: an add of the model racing
        // with this either keeps the index (its slot is already in) or
        // creates the next one.
        let last_of_model = !handles
            .values()
            .any(|slot| matches!(slot, Slot::Active(other) if other.model_id == model_id));
        if last_of_model {
            indexers.remove(model_id);
            block_sizes.remove(model_id);
        }
    }

    /// Stop the KV event subscription for a worker.
    ///
    /// Sends a graceful shutdown signal. The subscription task takes the
    /// worker's blocks out of the index from its own per-worker map; that
    /// CPU-bound cleanup runs on the bounded blocking pool rather than a Tokio
    /// runtime worker.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        let subscription = {
            let mut handles = self.worker_handles.lock().await;
            match handles.remove(worker_url) {
                Some(Slot::Active(sub)) => {
                    handles.insert(
                        worker_url.to_string(),
                        Slot::Removing(Reservation {
                            id: sub.id,
                            done: sub.done.clone(),
                        }),
                    );
                    Some(sub)
                }
                Some(removing @ Slot::Removing(_)) => {
                    // Another removal of the same URL is already under way.
                    handles.insert(worker_url.to_string(), removing);
                    None
                }
                None => None,
            }
        };

        let Some(sub) = subscription else {
            return;
        };
        info!(worker_url = %worker_url, "Stopping KV event subscription");
        // Signal graceful shutdown — task cleans up its worker_blocks in the indexer.
        let _ = sub.shutdown_tx.send(());
        // The task lifts the reservation and drops the model's last index on
        // its own (`complete_removal`); this wait is for the caller, who has
        // the worker out of the index on return. A caller that drops the
        // future here loses nothing but the join error below.
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
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        let subscriptions: Vec<(String, WorkerSubscription)> = {
            let mut handles = self.worker_handles.lock().await;
            let mut active = Vec::new();
            for (url, slot) in std::mem::take(&mut *handles) {
                match slot {
                    Slot::Active(sub) => active.push((url, sub)),
                    // The task behind a removal in flight lifts its own
                    // reservation.
                    reserved @ Slot::Removing(_) => {
                        handles.insert(url, reserved);
                    }
                }
            }
            active
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
    pub fn get_indexer(&self, model_id: &str) -> Option<Arc<KvIndex>> {
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

    /// Normalize model_id to match routing's `normalize_model_key`.
    /// Empty model IDs map to UNKNOWN_MODEL_ID for consistent keying.
    fn normalize_model_id(model_id: &str) -> String {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            model_id.to_string()
        }
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

    use super::{KvEventMonitor, KvIndex, WorkerIndexState};

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
        state: WorkerIndexState,
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
            for event in &batch.events {
                KvEventMonitor::apply_event(event, self.worker, index, &mut self.state);
            }
        }

        /// The worker leaves: its blocks go with it.
        pub fn remove(self, index: &KvIndex) {
            index.remove_worker(self.worker, self.state.blocks);
        }
    }
}

impl Drop for KvEventMonitor {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.worker_handles.try_lock() {
            for (_, slot) in handles.drain() {
                if let Slot::Active(sub) = slot {
                    let _ = sub.shutdown_tx.send(());
                    sub.handle.abort(); // Can't await in Drop, abort as fallback
                }
            }
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
    use openai_protocol::worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerType};

    use super::{
        subscription::{INDEX_REMOVAL_PERMITS, MAX_CONCURRENT_INDEX_REMOVALS},
        *,
    };
    use crate::worker::{kv_index_backend::WorkerBlocks, BasicWorkerBuilder};

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        for _ in 0..200 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    /// A gRPC worker at a port that refuses connections at once: its task
    /// loops on reconnects and answers shutdown, with no engine involved.
    fn refusing_grpc_worker() -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
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
        monitor.on_worker_added(&worker).await;
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("the subscription interns its worker", || {
            index.worker_id(&url).is_some()
        })
        .await;

        // Hold every cleanup permit: the old task's cleanup cannot finish.
        let permits = INDEX_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_INDEX_REMOVALS as u32)
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
            async move { monitor.on_worker_added(&worker).await }
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
        monitor.on_worker_added(&worker).await;
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("the subscription interns its worker", || {
            index.worker_id(&url).is_some()
        })
        .await;

        // Hold every cleanup permit: the old task's cleanup cannot finish.
        let permits = INDEX_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_INDEX_REMOVALS as u32)
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
        let re_add =
            tokio::time::timeout(Duration::from_millis(200), monitor.on_worker_added(&worker))
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
        tokio::time::timeout(Duration::from_secs(5), monitor.on_worker_added(&worker))
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

    #[test]
    fn a_new_monitor_holds_no_index() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.indexers.is_empty());
    }

    #[tokio::test]
    async fn an_unknown_model_has_no_index() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.get_indexer("nonexistent").is_none());
    }

    #[tokio::test]
    async fn stopping_an_empty_monitor_is_a_no_op() {
        let monitor = KvEventMonitor::new();
        monitor.stop().await;
    }

    #[tokio::test]
    async fn removing_an_unknown_worker_is_a_no_op() {
        let monitor = KvEventMonitor::new();
        monitor.on_worker_removed("http://nonexistent:8000").await;
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
            assert!(gauge(&handle, "smg_kv_index_arena_bytes", "chain").is_some_and(|b| b > 0.0));
            assert!(gauge(&handle, "smg_kv_index_slab_bytes", "chain").is_some_and(|b| b > 0.0));
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
