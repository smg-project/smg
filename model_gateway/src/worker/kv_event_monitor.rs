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

use dashmap::{mapref::entry::Entry, DashMap};
use futures::FutureExt as _;
use tokio::{
    sync::{oneshot, watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use super::{kv_index_backend::KvIndex, monitor::WorkerMonitor, registry::WorkerRegistry};
use crate::{
    observability::metrics::Metrics,
    policies::utils::PeriodicTask,
    worker::{endpoint::Endpoint, ConnectionMode, Worker, UNKNOWN_MODEL_ID},
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

/// What a worker's stream is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubscriptionKind {
    /// The worker's KV events into its model's index, with the load records
    /// that ride them: a worker under a cache-aware policy.
    Events,
    /// The worker's load records alone (`SubscribeKvEventsRequest.
    /// loads_only`): a worker priced by its load with no KV index. A
    /// servicer that predates the field streams its events too; they are
    /// dropped.
    Loads,
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
    /// The registry whose object under a URL an add starts a stream for:
    /// `on_worker_added` reads it under the slot lock. Weak like the load
    /// sink, a peer in the app context. Unset in a monitor built alone (the
    /// tests), which holds every object.
    worker_registry: OnceLock<Weak<WorkerRegistry>>,
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
    kind: SubscriptionKind,
    /// The object the stream was started for. A re-registration replaces
    /// the object under the URL, and the records on the stream reach the
    /// one the task holds, so an add with another object restarts it.
    started_for: Weak<dyn Worker>,
    /// Signals the subscription task to shut down gracefully. The task owns
    /// the worker's index state and takes its blocks out of the index on exit.
    shutdown_tx: oneshot::Sender<()>,
    /// Closes when the task has ended; a removal moves it into the
    /// reservation it leaves in the slot.
    done: watch::Receiver<()>,
}

impl WorkerSubscription {
    /// Whether the stream was started for this worker object (the address
    /// alone: the object's vtable is no part of its identity).
    fn streams_for(&self, worker: &Arc<dyn Worker>) -> bool {
        std::ptr::addr_eq(self.started_for.as_ptr(), Arc::as_ptr(worker))
    }
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
            worker_registry: OnceLock::new(),
        }
    }

    /// Route the load records on every worker's event stream to the worker
    /// monitor (once; a later call is ignored).
    pub fn set_load_sink(&self, monitor: &Arc<WorkerMonitor>) {
        let _ = self.load_sink.set(Arc::downgrade(monitor));
    }

    /// Have the add read the registry before it stops or starts a stream
    /// (once; a later call is ignored): a worker the registry no longer
    /// holds under its URL gets none. The app sets it where it wires the
    /// load sink; a monitor without one holds every object.
    pub fn set_worker_registry(&self, registry: &Arc<WorkerRegistry>) {
        let _ = self.worker_registry.set(Arc::downgrade(registry));
    }

    /// Whether the registry still holds this object under its URL, by
    /// address (the vtable is no part of the identity). With no registry
    /// to ask, every object is held.
    fn still_held(&self, worker: &Arc<dyn Worker>) -> bool {
        let Some(registry) = self.worker_registry.get().and_then(Weak::upgrade) else {
            return true;
        };
        registry
            .get_by_url(worker.url())
            .is_some_and(|held| std::ptr::addr_eq(Arc::as_ptr(&held), Arc::as_ptr(worker)))
    }

    /// The object the registry holds under `url`. `None` with no registry
    /// to ask, which leaves a removal unconditional.
    fn held_under(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.worker_registry
            .get()
            .and_then(Weak::upgrade)?
            .get_by_url(url)
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

    /// One model's index size and chain shape as gauges: the periodic
    /// publication's unit, and what a new index publishes at once, so the
    /// model's series exist (at zero) from its first worker rather than
    /// from the first tick.
    fn publish_index(model_id: &str, index: &KvIndex) {
        Metrics::set_kv_index_size(model_id, index.current_size(), index.entry_count());
        if let Some(stats) = index.chain_stats() {
            Metrics::set_kv_index_chain_stats(model_id, &stats);
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
        Metrics::set_kv_index_chain_stats(model_id, &kv_index::ChainIndexStats::default());
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
    /// (`Events`), or its load records alone (`Loads`), on a background task
    /// that subscribes over server-streaming gRPC. Either way the records on
    /// the stream reach the load sink. A second call for the object already
    /// streaming is a no-op. A call with another object under an active URL
    /// restarts the stream for it: a re-registration replaced the worker in
    /// place (an upsert of a changed spec, a replacement by id, a label
    /// refresh; the registration and update workflows' policy steps both
    /// call this for the object the registry holds), and the records on the
    /// stream must reach that object, since the load snapshot fences the
    /// pushed loads of any other object out and its receipts suppress the
    /// poll of the URL. The restart waits for the old task to end; a caller
    /// whose timeout drops this call in that wait leaves the removal to the
    /// old task's own exit path and the URL without a stream until the next
    /// add, the worker's next registration or label refresh.
    ///
    /// The add reads the registry it was given (`set_worker_registry`)
    /// under the slot lock on every turn, before it stops a stream for
    /// another object and before it inserts the new one, so a worker the
    /// registry no longer holds (removed, or replaced again while the add
    /// waited for an old task to end) gets no stream, and a stream another
    /// object has under the URL is left to it. The registry's removal
    /// changes the registry before it reaches this lock: a read here sees
    /// the removal, or the removal finds the slot this add inserts. A
    /// monitor without a registry holds every object.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>, kind: SubscriptionKind) {
        let url = worker.url().to_string();
        // Normalize model_id to match routing's normalize_model_key — empty → "unknown".
        let model_id = Self::normalize_model_id(worker.model_id());

        // Only gRPC workers stream KV events. HTTP proxies don't, and a ZMQ
        // EngineCore returns `unimplemented` for SubscribeKvEvents — subscribing
        // there would just be a wasted handshake + round-trip per registration.
        if *worker.connection_mode() != ConnectionMode::Grpc {
            debug!(worker_url = %Endpoint::redacted_str(&url), mode = %worker.connection_mode(), "non-gRPC worker, skipping KV event subscription");
            return;
        }

        let mut handles = loop {
            let mut handles = self.worker_handles.lock().await;
            // Under the slot lock, on every turn: an object the registry no
            // longer holds gets no stream, and a stream another object has
            // under the URL is not stopped for it.
            if !self.still_held(worker) {
                debug!(worker_url = %Endpoint::redacted_str(&url), "no KV event stream for a worker the registry no longer holds");
                return;
            }
            // A stream started for another object under this URL, which the
            // registry has replaced: take it out of the slot on this same turn
            // of the lock (a removal by URL after the lock could take a stream
            // a concurrent add started in between; the reservation left here
            // makes that add wait instead), stop it and await its end outside
            // the lock, and start one for the object given, on the next turn.
            // Dropped in that wait, this add leaves the slot to the old task
            // (the reservation lifts itself) and starts nothing.
            if matches!(handles.get(&url), Some(Slot::Active(sub)) if !sub.streams_for(worker)) {
                let stale = Self::begin_removal(&mut handles, &url);
                drop(handles);
                info!(worker_url = %Endpoint::redacted_str(&url), "Restarting KV event subscription for a re-registered worker");
                if let Some(sub) = stale {
                    self.finish_removal(&url, sub).await;
                }
                continue;
            }
            let mut done = match handles.get(&url) {
                Some(Slot::Active(_)) => {
                    debug!(worker_url = %Endpoint::redacted_str(&url), "KV event subscription already active, skipping");
                    return;
                }
                Some(Slot::Removing(reservation)) => reservation.done.clone(),
                None => break handles,
            };
            // The previous subscription for this URL is still being taken
            // out of the index: intern it again only once its task has ended.
            if done.has_changed().is_err() {
                // The task is gone without lifting its reservation (it ended
                // before the removal began, was aborted, or panicked past its
                // guard): nothing is left to wait for. Whatever it raised on
                // the gauge is lowered before its successor starts, whose
                // answer owns the gauge from here (the removal's own reset
                // skips a slot that is no longer its reservation).
                handles.remove(&url);
                Metrics::set_kv_events_unavailable(&url, false);
                break handles;
            }
            drop(handles);
            // Nothing is ever sent on the channel; this returns at its close.
            let _ = done.changed().await;
        };

        // A loads-only stream feeds no index: the model gets one with its
        // first events subscription.
        let indexer = match kind {
            SubscriptionKind::Events => {
                let (indexer, created) = match self.indexers.entry(model_id.clone()) {
                    Entry::Occupied(entry) => (Arc::clone(entry.get()), false),
                    Entry::Vacant(entry) => {
                        let index = Arc::new(KvIndex::chain());
                        entry.insert(Arc::clone(&index));
                        (index, true)
                    }
                };
                if created {
                    Self::publish_index(&model_id, &indexer);
                }
                Some(indexer)
            }
            SubscriptionKind::Loads => None,
        };
        // The worker's integrity series exist before its stream does.
        Metrics::init_kv_event_series(&url);
        // Seed block_size provisionally from WorkerSpec. The event stream will
        // overwrite this with the backend's actual page size once received.
        if let Some(bs) = worker
            .metadata()
            .spec
            .kv_block_size
            .filter(|_| kind == SubscriptionKind::Events)
        {
            if bs > 0 {
                let known = self
                    .block_sizes
                    .entry(model_id.clone())
                    .or_insert(KvBlockSize {
                        tokens: bs,
                        from_events: false,
                    });
                // Under the entry's lock, as the learn site writes it, so a
                // size a stream learns meanwhile is not overwritten on the
                // gauge by the seed.
                Metrics::set_kv_block_size(&model_id, known.tokens);
            } else {
                warn!(worker_url = %Endpoint::redacted_str(&url), "Worker reports kv_block_size=0, ignoring");
            }
        }

        let started_for = Arc::downgrade(worker);
        let worker = Arc::clone(worker);
        let worker_url = url.clone();
        let block_sizes = Arc::clone(&self.block_sizes);

        info!(
            worker_url = %Endpoint::redacted_str(&url),
            model_id = %model_id,
            ?kind,
            "Starting KV event subscription"
        );

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (done_tx, done) = watch::channel(());
        let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
        let load_sink = self.load_sink.get().cloned();
        let loop_model_id = model_id.clone();
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
            // The task writes for the worker after the loop has returned it
            // (the panic counter, the gauge `complete_removal` lowers). It
            // keeps a handle on the worker through those writes: the series
            // store forgets a removed address once no handle on its workers
            // is left, and a write for the address by name alone after that
            // would recreate its series.
            let writer = Arc::clone(&worker);
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
                    worker_url = %Endpoint::redacted_str(writer.url()),
                    panic.message = %msg,
                    ?kind,
                    "KV event subscription task panicked; this worker's stream \
                     no longer feeds routing"
                );
                Metrics::record_kv_event_subscription_failure(writer.url(), "panic");
            }
            Self::complete_removal(
                &slots,
                &indexers,
                &model_block_sizes,
                &writer,
                id,
                &task_model_id,
                kind,
            )
            .await;
            // The task's last write is made: the worker may go.
            drop(writer);
            // Closes `done` last: an add waiting on it finds the slot free.
            drop(done_tx);
        });

        handles.insert(
            url,
            Slot::Active(WorkerSubscription {
                id,
                handle,
                model_id,
                kind,
                started_for,
                shutdown_tx,
                done,
            }),
        );
    }

    /// The end of a removal, run by the subscription task once its blocks
    /// are out of the index: lift the URL's reservation if it is this task's
    /// (a later subscription under the URL has its own), and drop the
    /// model's index with the model's last events subscription, zeroing the
    /// gauges that index fed (a loads-only subscription had none to drop).
    /// The task does this rather than `on_worker_removed`,
    /// whose future a caller may drop before the task has ended; a slot that
    /// is `Active` (the task ended on its own, or `stop` took the
    /// subscription) is left as it is. `worker` is the removed worker itself,
    /// which the task holds through the write made here (see the task in
    /// [`Self::on_worker_added`]).
    async fn complete_removal(
        slots: &Mutex<HashMap<String, Slot>>,
        indexers: &DashMap<String, Arc<KvIndex>>,
        block_sizes: &DashMap<String, KvBlockSize>,
        worker: &Arc<dyn Worker>,
        id: u64,
        model_id: &str,
        kind: SubscriptionKind,
    ) {
        let worker_url = worker.url();
        let mut handles = slots.lock().await;
        if !matches!(handles.get(worker_url), Some(Slot::Removing(held)) if held.id == id) {
            return;
        }
        // A backend without KV events is no longer a blind spot once its
        // worker has left: lowered here, by the task that owned the
        // subscription and under the slot lock, so a re-add of the URL that
        // follows the removal cannot see its fresh gauge reset by a stale
        // caller, and a caller that dropped `on_worker_removed` early does
        // not leave it raised.
        Metrics::set_kv_events_unavailable(worker_url, false);
        handles.remove(worker_url);
        // Under the slot lock, which `on_worker_added` holds from its lookup
        // of the model's index to its insert: an add of the model racing
        // with this either keeps the index (its slot is already in) or
        // creates the next one.
        if kind != SubscriptionKind::Events {
            return;
        }
        let last_of_model = !handles.values().any(|slot| {
            matches!(slot, Slot::Active(other)
                if other.kind == SubscriptionKind::Events && other.model_id == model_id)
        });
        if last_of_model {
            indexers.remove(model_id);
            block_sizes.remove(model_id);
            Self::publish_dropped_index(model_id);
        }
    }

    /// Stop the KV event subscription for a worker.
    ///
    /// Sends a graceful shutdown signal. The subscription task takes the
    /// worker's blocks out of the index from its own per-worker map; that
    /// CPU-bound cleanup runs on the bounded blocking pool rather than a Tokio
    /// runtime worker. A write made here once the task is gone (the join
    /// error, the gauge of a task that ended before the removal) is made
    /// under the caller's handle on the removed worker, which the removal
    /// step holds through this call.
    ///
    /// The step runs after the registry let the worker go. A registration
    /// under the same URL in between holds a new object, and its add may
    /// have restarted the stream for it: that stream is the new object's
    /// and stays, read under the slot lock as the add side reads the
    /// registry; its own removal ends it. The stream of any other object,
    /// the removed one's or an older one's, ends here.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        let subscription = {
            let mut handles = self.worker_handles.lock().await;
            if let Some(Slot::Active(sub)) = handles.get(worker_url) {
                if self
                    .held_under(worker_url)
                    .is_some_and(|held| sub.streams_for(&held))
                {
                    debug!(worker_url = %Endpoint::redacted_str(worker_url), "KV event stream left to the worker the registry holds under the URL");
                    return;
                }
            }
            Self::begin_removal(&mut handles, worker_url)
        };

        let Some(sub) = subscription else {
            return;
        };
        self.finish_removal(worker_url, sub).await;
    }

    /// A worker that streams nothing, in place of one that did: the stream
    /// a former object has under the URL ends here, as `on_worker_added`
    /// restarts it for an object with a kind. Nothing else on the replace
    /// path stops it, and its records would keep reaching the replaced
    /// object. The checks are the restart's, under the slot lock: a worker
    /// the registry no longer holds changes nothing (the URL's stream is the
    /// business of whatever holds the URL), and a stream started for this
    /// object stays, as it does for an add with another kind. The stop and
    /// its wait run outside the lock, and only the removal's own reservation
    /// is left, so an add that follows starts afresh.
    pub async fn on_worker_added_without_stream(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        let stale = {
            let mut handles = self.worker_handles.lock().await;
            if !self.still_held(worker) {
                debug!(worker_url = %Endpoint::redacted_str(url), "no KV event stream to end for a worker the registry no longer holds");
                return;
            }
            if !matches!(handles.get(url), Some(Slot::Active(sub)) if !sub.streams_for(worker)) {
                return;
            }
            Self::begin_removal(&mut handles, url)
        };
        if let Some(sub) = stale {
            info!(worker_url = %Endpoint::redacted_str(url), "Ending the KV event subscription of a replaced worker: its successor streams nothing");
            self.finish_removal(url, sub).await;
        }
    }

    /// Under the slot lock: the URL's active subscription taken out, its
    /// reservation left in the slot for the adds that come before its task
    /// has ended; `None` for an empty slot or a removal already under way.
    fn begin_removal(
        handles: &mut HashMap<String, Slot>,
        worker_url: &str,
    ) -> Option<WorkerSubscription> {
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
    }

    /// Outside the lock: the taken subscription told to stop and awaited to
    /// its end; the gauge lowered while the slot still holds its reservation.
    async fn finish_removal(&self, worker_url: &str, sub: WorkerSubscription) {
        info!(worker_url = %Endpoint::redacted_str(worker_url), "Stopping KV event subscription");
        // Signal graceful shutdown — task cleans up its worker_blocks in the indexer.
        let _ = sub.shutdown_tx.send(());
        // The task lifts the reservation and drops the model's last index on
        // its own (`complete_removal`); this wait is for the caller, who has
        // the worker out of the index on return. A caller that drops the
        // future here loses nothing but the join error below.
        // Panics are caught inside the task; a JoinError here (abort or a
        // panic that escaped the guard) must still be surfaced, not discarded.
        let id = sub.id;
        if let Err(e) = sub.handle.await {
            error!(
                worker_url = %Endpoint::redacted_str(worker_url),
                error = %e,
                "KV event subscription task failed"
            );
            Metrics::record_kv_event_subscription_failure(worker_url, "join_error");
        }
        // A task that ends on the shutdown signal lowers the gauge in its own
        // cleanup (`complete_removal`). One that had ended before this removal
        // began (the answer that raises the gauge ends the subscription), or
        // that did not reach its cleanup, left the reservation in place and
        // the gauge with it: lowered here, under the slot lock and only while
        // the reservation is still this subscription's. A successor under the
        // URL owns the gauge from its add on.
        let handles = self.worker_handles.lock().await;
        if matches!(handles.get(worker_url), Some(Slot::Removing(held)) if held.id == id) {
            Metrics::set_kv_events_unavailable(worker_url, false);
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
                debug!(worker_url = %Endpoint::redacted_str(&url), "Stopping KV event subscription");
                let _ = sub.shutdown_tx.send(());
                if let Err(e) = sub.handle.await {
                    error!(
                        worker_url = %Endpoint::redacted_str(&url),
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

    /// The ids under a URL's slot: the active subscription's, or the
    /// reservation's of a removal under way.
    #[cfg(test)]
    async fn slot_ids(&self, worker_url: &str) -> (Option<u64>, Option<u64>) {
        let handles = self.worker_handles.lock().await;
        Self::ids_under(&handles, worker_url)
    }

    /// [`Self::slot_ids`] under a lock the caller holds.
    #[cfg(test)]
    fn ids_under(handles: &HashMap<String, Slot>, worker_url: &str) -> (Option<u64>, Option<u64>) {
        match handles.get(worker_url) {
            Some(Slot::Active(sub)) => (Some(sub.id), None),
            Some(Slot::Removing(held)) => (None, Some(held.id)),
            None => (None, None),
        }
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
            KvEventMonitor::apply_events(&batch.events, self.worker, index, &mut self.state);
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
    use tracing_test::traced_test;

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

    /// The late-write test's worker address: a closed port, so its task loops
    /// on reconnects and answers shutdown with no engine involved, and one no
    /// other test of this binary registers. The series store is the
    /// process's, and a registration of the same address by a test on another
    /// thread would clear its retirement under this test's assertions.
    const REFUSING_URL: &str = "grpc://127.0.0.1:2";

    /// A gRPC worker at [`REFUSING_URL`].
    fn refusing_grpc_worker() -> Arc<dyn Worker> {
        refusing_grpc_worker_at(REFUSING_URL)
    }

    /// The replacement test's address, another closed port: its registry
    /// registrations publish the address's series, which the late-write
    /// test's retirement assertions on [`REFUSING_URL`] must not see.
    const REPLACED_URL: &str = "grpc://127.0.0.1:3";

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

    /// The subscription task writes for its worker after the loop has
    /// returned it (the gauge `complete_removal` lowers), and it keeps a
    /// handle on the worker through that write: the series store forgets a
    /// removed address once no handle on its workers is left, so a late write
    /// by name alone would recreate the address's series for good.
    #[test]
    fn a_removals_late_write_recreates_no_series_for_the_address() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::observability::worker_metrics::{self, WorkerSeriesRecorder};

        let series = worker_metrics::installed_or_default();
        let recorder = WorkerSeriesRecorder::new(
            PrometheusBuilder::new().build_recorder(),
            Arc::clone(series),
        );
        let listed = || {
            let mut rendered = String::new();
            series.render_into(&mut rendered);
            rendered
                .lines()
                .any(|line| line.contains(&format!("worker=\"{REFUSING_URL}\"")))
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // The spawned task runs on this thread, under the local recorder.
        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let monitor = KvEventMonitor::new();
                let worker = refusing_grpc_worker();
                let url = worker.url().to_string();
                let witness = Arc::downgrade(&worker);
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                // The task starts at this thread's first yield. From then on it
                // holds the worker twice, in its loop and in the handle it keeps
                // for the removal writes: three strong counts with this test's.
                wait_until("the task holds the worker", || witness.strong_count() >= 3).await;
                assert!(listed(), "the subscription publishes the worker's series");

                // The registry's removal retires the address with the worker's
                // witness; from here the task alone holds the worker.
                Metrics::remove_worker_metrics(&url, Arc::downgrade(&worker));
                drop(worker);
                // The removal up to its shutdown signal, as `on_worker_removed`
                // does it, under a slot lock this test keeps: the task's
                // cleanup waits at that lock once its loop has returned.
                let mut handles = monitor.worker_handles.lock().await;
                let Some(Slot::Active(sub)) = handles.remove(&url) else {
                    panic!("an active subscription");
                };
                let mut done = sub.done.clone();
                handles.insert(
                    url.clone(),
                    Slot::Removing(Reservation {
                        id: sub.id,
                        done: sub.done.clone(),
                    }),
                );
                let _ = sub.shutdown_tx.send(());
                wait_until("the loop has returned its worker", || {
                    witness.strong_count() <= 1
                })
                .await;
                assert_eq!(
                    witness.strong_count(),
                    1,
                    "the task holds the worker through its removal writes"
                );
                series.run_upkeep();
                assert!(
                    series.is_retired(&url),
                    "the address stays retired while the task can still write"
                );

                drop(handles);
                let _ = done.changed().await;
                assert_eq!(
                    witness.strong_count(),
                    0,
                    "the task is gone with its handle"
                );
                series.run_upkeep();
                assert!(
                    !series.is_retired(&url),
                    "the address is forgotten once the task is gone"
                );
                assert!(!listed(), "the removal's late write recreated no series");
            });
        });
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

    #[test]
    fn a_new_monitor_holds_no_index() {
        let monitor = KvEventMonitor::new();
        assert!(monitor.indexers.is_empty());
    }

    /// A re-registration replaces the worker object under its URL (an
    /// upsert of a changed spec, a replacement by id, a label refresh) and
    /// runs the policy step again. The stream started for the old object
    /// must not stay: its records would reach an object the registry no
    /// longer holds, whose pushed loads the load snapshot fences out and
    /// whose receipts suppress the URL's poll. The add with the new object
    /// restarts the stream for it; a second add of the same object changes
    /// nothing.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_re_registration_under_the_url_restarts_the_stream_for_the_new_object() {
        let monitor = Arc::new(KvEventMonitor::new());
        let first = refusing_grpc_worker();
        let second = refusing_grpc_worker();
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        assert!(
            !logs_contain("Restarting KV event subscription"),
            "the same object's second add keeps its stream"
        );

        let old = Arc::downgrade(&first);
        drop(first);
        monitor
            .on_worker_added(&second, SubscriptionKind::Loads)
            .await;
        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));
        assert!(
            old.upgrade().is_none(),
            "the old stream's task has ended and released the old object"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                2 => Ok(()),
                n => Err(format!("{n} starts for two objects under one URL")),
            }
        });
        monitor.stop().await;
    }

    /// A replacement that streams nothing (its policy reads no index and
    /// the worker monitor wants no loads) ends the stream the object it
    /// replaced had under the URL: the records on that stream would keep
    /// reaching an object the registry no longer holds, and nothing else on
    /// the replace path stops it.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_that_streams_nothing_ends_the_stream_of_the_object_it_replaced() {
        let monitor = Arc::new(KvEventMonitor::new());
        let first = refusing_grpc_worker();
        let url = first.url().to_string();
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let old = Arc::downgrade(&first);
        drop(first);

        let second = refusing_grpc_worker();
        monitor.on_worker_added_without_stream(&second).await;
        assert!(logs_contain(
            "Ending the KV event subscription of a replaced worker"
        ));
        assert!(
            old.upgrade().is_none(),
            "the old stream's task has ended and released the old object"
        );
        assert_eq!(
            monitor.slot_ids(&url).await,
            (None, None),
            "the URL's slot is empty: nothing streams and nothing is reserved"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                1 => Ok(()),
                n => Err(format!("{n} starts: the replacement streams nothing")),
            }
        });
        monitor.stop().await;
    }

    /// The registry replaced the worker again before its call ran, and the
    /// add for the object it holds now has restarted the stream: the call
    /// for the object nobody holds leaves that stream alone. A removal by
    /// URL here would take it.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_that_streams_nothing_leaves_the_stream_the_registry_holds() {
        let monitor = Arc::new(KvEventMonitor::new());
        let registry = Arc::new(WorkerRegistry::new());
        monitor.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = refusing_grpc_worker_at(url);
        registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        drop(first);

        let second = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        let third = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&third));
        monitor
            .on_worker_added(&third, SubscriptionKind::Loads)
            .await;
        let (active, removing) = monitor.slot_ids(url).await;
        let held = active.expect("the add of the third restarted the stream");
        assert_eq!(removing, None);

        monitor.on_worker_added_without_stream(&second).await;
        assert_eq!(
            monitor.slot_ids(url).await,
            (Some(held), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "no KV event stream to end for a worker the registry no longer holds"
        ));
        monitor.stop().await;
    }

    /// A worker that streams nothing now, with the stream its add under a
    /// kind started: that stream stays. The monitor never ends or restarts a
    /// stream for the object it was started for; an add with another kind
    /// leaves it too.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_that_streams_nothing_keeps_the_stream_started_for_it() {
        let monitor = Arc::new(KvEventMonitor::new());
        let worker = refusing_grpc_worker();
        let url = worker.url().to_string();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        let (active, removing) = monitor.slot_ids(&url).await;
        let started = active.expect("the add started a stream");
        assert_eq!(removing, None);

        monitor.on_worker_added_without_stream(&worker).await;
        assert_eq!(
            monitor.slot_ids(&url).await,
            (Some(started), None),
            "the stream started for the object stays"
        );
        assert!(!logs_contain("Ending the KV event subscription"));
        monitor.stop().await;
    }

    /// The removal step runs after the registry let the worker go, and a
    /// registration under the same URL in between may hold a new object
    /// whose add restarted the stream for it. The removal of the old object
    /// leaves that stream: a removal by URL alone would take it and leave
    /// the new object registered with no stream until its next add.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_leaves_the_stream_of_the_object_the_registry_holds() {
        let monitor = Arc::new(KvEventMonitor::new());
        let registry = Arc::new(WorkerRegistry::new());
        monitor.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = refusing_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;

        // The removal took the first out; a registration put the second in
        // and its add restarted the stream before the removal's call here.
        assert!(
            registry.remove(&id).is_some(),
            "the first leaves the registry"
        );
        let second = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        monitor
            .on_worker_added(&second, SubscriptionKind::Loads)
            .await;
        let (active, removing) = monitor.slot_ids(url).await;
        let held = active.expect("the add of the second restarted the stream");
        assert_eq!(removing, None);

        monitor.on_worker_removed(url).await;
        assert_eq!(
            monitor.slot_ids(url).await,
            (Some(held), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "KV event stream left to the worker the registry holds under the URL"
        ));
        monitor.stop().await;
    }

    /// The registry holds a new object whose add has not run yet: the
    /// stream under the URL is still the removed object's, and the removal
    /// ends it. The new object's add then starts afresh.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_ends_an_old_stream_under_an_object_not_yet_added() {
        let monitor = Arc::new(KvEventMonitor::new());
        let registry = Arc::new(WorkerRegistry::new());
        monitor.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = refusing_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        assert!(
            registry.remove(&id).is_some(),
            "the first leaves the registry"
        );
        let second = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));

        monitor.on_worker_removed(url).await;
        assert_eq!(
            monitor.slot_ids(url).await,
            (None, None),
            "the first's stream ended and its reservation was lifted"
        );
        assert!(logs_contain("Stopping KV event subscription"));
        monitor.stop().await;
    }

    /// With nobody under the URL the removal ends the stream: the registry
    /// is read, not assumed, and an unheld address keeps no stream.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_with_nobody_under_the_url_ends_the_stream() {
        let monitor = Arc::new(KvEventMonitor::new());
        let registry = Arc::new(WorkerRegistry::new());
        monitor.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let worker = refusing_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&worker))
            .expect("the registration under the address");
        monitor
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        assert!(
            registry.remove(&id).is_some(),
            "the worker leaves the registry"
        );

        monitor.on_worker_removed(url).await;
        assert_eq!(
            monitor.slot_ids(url).await,
            (None, None),
            "the stream ended and its reservation was lifted"
        );
        assert!(!logs_contain("left to the worker the registry holds"));
        monitor.stop().await;
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

    /// The restart takes the stream it found under the URL on the lock turn
    /// that found it. Made on a later turn, by URL, the removal would take
    /// whatever stream sat there by then, and between the two turns another
    /// add can run. The test holds the lock when the restart is first polled
    /// and queues a read of the slot behind the restart's turn: the lock is
    /// fair, so the read runs the moment that turn ends, before any later
    /// turn, and finds the reservation. The rest is the race run to its end:
    /// a third object replaces the second and starts its stream while the
    /// restart waits for the old task, and the resumed restart leaves that
    /// stream to it.
    #[traced_test]
    #[tokio::test]
    async fn a_restart_reserves_the_slot_on_its_own_lock_turn() {
        let monitor = Arc::new(KvEventMonitor::new());
        let registry = Arc::new(WorkerRegistry::new());
        monitor.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = refusing_grpc_worker_at(url);
        registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let (active, removing) = monitor.slot_ids(url).await;
        let stopped = active.expect("the first add started a stream");
        assert_eq!(removing, None);
        drop(first);

        // The second replaces the first in place. Polled against a held
        // lock, its restart queues for its first turn, and a read of the
        // slot queues right behind it.
        let second = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        let mut restart = std::pin::pin!(monitor.on_worker_added(&second, SubscriptionKind::Loads));
        let mut read = std::pin::pin!(monitor.worker_handles.lock());
        let held = monitor.worker_handles.lock().await;
        assert!(
            std::future::Future::poll(restart.as_mut(), &mut cx).is_pending(),
            "the restart queues for the held lock"
        );
        assert!(
            std::future::Future::poll(read.as_mut(), &mut cx).is_pending(),
            "the read queues behind the restart"
        );
        drop(held);
        assert!(
            std::future::Future::poll(restart.as_mut(), &mut cx).is_pending(),
            "the restart's first turn ends in its wait for the old task"
        );
        {
            let handles = read.await;
            assert_eq!(
                KvEventMonitor::ids_under(&handles, url),
                (None, Some(stopped)),
                "the turn that found the stream left its reservation"
            );
        }
        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));

        // A third object replaces the second while the restart waits, and
        // its add runs to its end: the old task ends on the stop it was
        // sent, and the third's stream starts.
        let third = refusing_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&third));
        tokio::time::timeout(
            Duration::from_secs(30),
            monitor.on_worker_added(&third, SubscriptionKind::Loads),
        )
        .await
        .expect("the third add ends once the old task has");
        let (active, removing) = monitor.slot_ids(url).await;
        let started = active.expect("the third add started a stream");
        assert_ne!(started, stopped);
        assert_eq!(removing, None);

        // The restart resumes behind a task that has ended, for an object
        // the registry no longer holds: the third's stream is not its to
        // take.
        tokio::time::timeout(Duration::from_secs(30), restart)
            .await
            .expect("the restart ends behind the ended task");
        assert_eq!(
            monitor.slot_ids(url).await,
            (Some(started), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "no KV event stream for a worker the registry no longer holds"
        ));
        tokio::time::timeout(Duration::from_secs(30), monitor.stop())
            .await
            .expect("the stop ends the stream left");
    }

    /// The add that restarts a stream waits for the old task to end. A
    /// caller's timeout that drops it in that wait (the policy steps run
    /// under one) leaves the removal to the old task's own exit path and the
    /// URL without a stream; the next add of the new object starts one
    /// afresh, with no second restart.
    #[traced_test]
    #[tokio::test]
    async fn an_add_dropped_while_it_waits_for_the_old_task_leaves_the_next_add_a_fresh_start() {
        let monitor = Arc::new(KvEventMonitor::new());
        let first = refusing_grpc_worker();
        monitor
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let old = Arc::downgrade(&first);
        drop(first);

        let second = refusing_grpc_worker();
        {
            // One poll: the slot is found streaming for another object, the
            // old task is told to stop, and the add waits for its end. Then
            // the add is dropped, as a timeout around it drops it.
            let mut add = std::pin::pin!(monitor.on_worker_added(&second, SubscriptionKind::Loads));
            let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
            assert!(
                std::future::Future::poll(add.as_mut(), &mut cx).is_pending(),
                "the restart waits for the old task to end"
            );
        }
        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));
        wait_until(
            "the old task ends on its own and releases its object",
            || old.upgrade().is_none(),
        )
        .await;

        monitor
            .on_worker_added(&second, SubscriptionKind::Loads)
            .await;
        logs_assert(|lines: &[&str]| {
            let count = |needle: &str| lines.iter().filter(|line| line.contains(needle)).count();
            match (
                count("Restarting KV event subscription"),
                count("Starting KV event subscription"),
            ) {
                (1, 2) => Ok(()),
                other => Err(format!(
                    "{other:?} restarts and starts; one restart and two starts expected"
                )),
            }
        });
        monitor.stop().await;
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
