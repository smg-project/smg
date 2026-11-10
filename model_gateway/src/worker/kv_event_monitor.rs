//! Per-engine KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per gRPC engine address
//! that subscribes to the engine's KV cache events and feeds them into a
//! shared [`KvIndex`] (one per model: the chain index). A data-parallel
//! engine registered as one worker per rank has one subscription: each rank
//! worker joins it and takes the batches and load records of its own rank.
//! This enables event-driven cache-aware routing as an alternative to the
//! approximate radix tree approach.
//!
//! Lifecycle:
//! - `on_worker_added` — the worker joins its engine's subscription, which
//!   is started for the engine's first worker (and the model's indexer with it)
//! - `on_worker_removed` — the worker leaves; the task takes its blocks out
//!   of the index; the engine's last worker ends the task
//! - `stop` — signals shutdown to all tasks, clears state
//!
//! The work of a subscription is split by stage: `subscription.rs` runs the
//! task per engine (connecting, reconnecting, the stream read loop, the
//! pushed load records, the workers joining and leaving); `admission.rs`
//! keeps one engine's state (per-rank cursors advanced once per engine, gap
//! recovery, relay snapshots, each rank worker's index state, and the
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
    sync::{mpsc, oneshot, watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use self::subscription::{EngineTask, MemberChange};
use super::{kv_index_backend::KvIndex, monitor::WorkerMonitor};
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

/// Manages per-engine KV cache event subscriptions.
///
/// Each gRPC engine address gets a dedicated tokio task that subscribes to the
/// backend's KV cache event stream and feeds events into a shared [`KvIndex`]
/// (one per `model_id`). Workers serving the same model share the same indexer;
/// the rank workers of one data-parallel engine share its one stream.
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
    /// The engines' subscriptions and every worker's slot on them.
    /// Mutex matches LoadMonitor pattern for atomic add + remove; Arc so a
    /// task can complete its members' removals and lift its own entry at
    /// the end of its exit path.
    subscriptions: Arc<Mutex<Subscriptions>>,
    /// Periodic index-shape publication, held so it aborts when the monitor
    /// drops. Set once by [`start_stats_task`](Self::start_stats_task); sync
    /// mutex because it is touched only at startup, never on event paths.
    stats_task: parking_lot::Mutex<Option<PeriodicTask>>,
    /// Where the load records on the event streams go (`KvEventBatch.load`):
    /// the worker monitor, which treats them as polls. Weak: the monitors
    /// are peers in the app context, neither owns the other.
    load_sink: OnceLock<Weak<WorkerMonitor>>,
}

/// Every subscription of the monitor: one per engine address, and the slot
/// of every worker on one.
#[derive(Default)]
struct Subscriptions {
    /// engine address → the engine's one subscription, for as long as its
    /// task runs.
    engines: HashMap<String, EngineSubscription>,
    /// worker URL → its place on its engine's subscription, or the
    /// reservation a removal leaves until the task has taken its blocks out.
    workers: HashMap<String, Slot>,
}

/// A worker URL's slot in the subscription map.
enum Slot {
    /// The worker is on its engine's subscription (the one with `engine`'s
    /// id: a slot whose task has ended is stale, and the next add or removal
    /// of the URL retires it instead of touching the engine's next task).
    Member {
        address: String,
        model_id: String,
        kind: SubscriptionKind,
        engine: u64,
    },
    /// The worker has been told to leave and the task is taking its blocks
    /// out of the index. The index hands a URL its existing id until that
    /// cleanup has released it, so a worker interned in this window would
    /// write under the old id and lose its blocks to the cleanup;
    /// `on_worker_added` waits for the cleanup to end instead.
    Removing(Reservation),
}

/// What a removal leaves in a URL's slot until its blocks are out.
///
/// The engine's task lifts it itself at the end of the removal
/// ([`KvEventMonitor::complete_member_removal`]), and `done` closes when the
/// removal is complete or the task is gone for any reason (an abort, a panic
/// past its guard). Nothing about it depends on the `on_worker_removed`
/// future living on: the registry's removal step runs it under a timeout,
/// and a cleanup waiting on the removal permits can outlast that; a
/// reservation only the remover could lift would then hold every later add
/// of the URL for good.
struct Reservation {
    /// The removal the reservation belongs to; a task lifts only that one.
    id: u64,
    /// Closes when the removal is complete.
    done: watch::Receiver<()>,
}

/// One engine's subscription: the task that reads the engine's one stream
/// for every worker registered under its address.
struct EngineSubscription {
    /// Tells this subscription from a later one under the same address.
    id: u64,
    handle: JoinHandle<()>,
    kind: SubscriptionKind,
    /// Workers join and leave through the task's channel; the stream is not
    /// restarted for them.
    changes: mpsc::UnboundedSender<MemberChange>,
    /// Workers on the subscription, joined and not yet told to leave. The
    /// last one's removal ends the task; a worker that arrives meanwhile
    /// waits for its end and starts the next one.
    members: usize,
    /// Signals the task to shut down gracefully. The task owns its members'
    /// index states and takes their blocks out of the index on exit.
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Closes when the task has ended.
    done: watch::Receiver<()>,
}

/// Ids for [`EngineSubscription`]s and removals, unique within the process.
static NEXT_SUBSCRIPTION_ID: AtomicU64 = AtomicU64::new(1);

impl KvEventMonitor {
    /// A monitor whose models get chain indexes.
    pub fn new() -> Self {
        Self {
            indexers: Arc::new(DashMap::new()),
            block_sizes: Arc::new(DashMap::new()),
            subscriptions: Arc::new(Mutex::new(Subscriptions::default())),
            stats_task: parking_lot::Mutex::new(None),
            load_sink: OnceLock::new(),
        }
    }

    /// Route the load records on every engine's event stream to the worker
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

    /// Put a worker on its engine's stream: its KV events into the model's
    /// `KvIndex` (`Events`), or its load records alone (`Loads`), on the
    /// background task that subscribes to the engine's address over
    /// server-streaming gRPC. The engine's first worker starts the task; a
    /// rank worker of a data-parallel engine joins the task its siblings
    /// share and takes its own rank's batches and records. Either way the
    /// records on the stream reach the load sink. Duplicate calls for the
    /// same worker URL are no-ops.
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
        // The engine's address: the URL without a rank suffix, which the
        // rank workers of one engine share.
        let address = worker.base_url().to_string();

        let slot = |engine: u64| Slot::Member {
            address: address.clone(),
            model_id: model_id.clone(),
            kind,
            engine,
        };
        let mut subs = loop {
            let mut subs = self.subscriptions.lock().await;
            // What the add waits for: the URL's old worker leaving the index
            // (`false`), or the engine's old task ending (`true`).
            // A member slot whose task still runs is the worker on its
            // stream; one whose task ended without the worker leaving (an
            // engine without KV events, a panic) is stale, and the worker
            // goes on the engine's next subscription.
            let member_of = match subs.workers.get(&url) {
                Some(Slot::Member { engine, .. }) => Some(*engine),
                _ => None,
            };
            if let Some(engine) = member_of {
                if matches!(subs.engines.get(&address), Some(live) if live.id == engine) {
                    debug!(worker_url = %Endpoint::redacted_str(&url), "KV event subscription already active, skipping");
                    return;
                }
                subs.workers.remove(&url);
                Metrics::set_kv_events_unavailable(&url, false);
                continue;
            }
            let (mut done, engine_ending) = match subs.workers.get(&url) {
                Some(Slot::Member { .. }) => return,
                Some(Slot::Removing(reservation)) => (reservation.done.clone(), false),
                None => match subs.engines.get_mut(&address) {
                    None => break subs,
                    // The engine's last worker is leaving and its task is
                    // ending: the next worker waits for the end and starts
                    // a subscription of its own.
                    Some(engine) if engine.members == 0 => (engine.done.clone(), true),
                    Some(engine) => {
                        // The engine streams already: the worker joins. The
                        // task takes the join at its next turn (between
                        // batches, or while a call or a backoff is pending)
                        // without restarting the stream.
                        if engine.kind != kind {
                            warn!(
                                worker_url = %Endpoint::redacted_str(&url),
                                engine = ?engine.kind,
                                worker = ?kind,
                                "Worker joins its engine's KV event subscription of another kind"
                            );
                        }
                        if engine
                            .changes
                            .send(MemberChange::Join(Arc::clone(worker)))
                            .is_ok()
                        {
                            engine.members += 1;
                            let (engine_id, members) = (engine.id, engine.members);
                            info!(
                                worker_url = %Endpoint::redacted_str(&url),
                                engine = %Endpoint::redacted_str(&address),
                                model_id = %model_id,
                                members,
                                "Joining the engine's KV event subscription"
                            );
                            self.prepare_worker(worker, &url, &model_id, kind);
                            subs.workers.insert(url, slot(engine_id));
                            return;
                        }
                        // The task is at its end (its channel is closed) and
                        // its entry goes with it: wait for that, as below.
                        (engine.done.clone(), true)
                    }
                },
            };
            // The URL's previous worker is still being taken out of the
            // index, or the engine's old task is still ending: intern again
            // only once that has ended.
            if done.has_changed().is_err() {
                // The task is gone without lifting its reservation or its
                // entry (it was aborted, or panicked past its guard): nothing
                // is left to wait for. Whatever it raised on the gauge is
                // lowered before its successor starts, whose answer owns the
                // gauge from here.
                if engine_ending {
                    subs.engines.remove(&address);
                } else {
                    subs.workers.remove(&url);
                    Metrics::set_kv_events_unavailable(&url, false);
                }
                continue;
            }
            drop(subs);
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
        self.prepare_worker(worker, &url, &model_id, kind);

        info!(
            worker_url = %Endpoint::redacted_str(&url),
            engine = %Endpoint::redacted_str(&address),
            model_id = %model_id,
            ?kind,
            "Starting KV event subscription"
        );

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (done_tx, done) = watch::channel(());
        let (changes_tx, changes) = mpsc::unbounded_channel();
        let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
        let task = EngineTask {
            address: address.clone(),
            model_id: model_id.clone(),
            indexer,
            block_sizes: Arc::clone(&self.block_sizes),
            load_sink: self.load_sink.get().cloned(),
            first: Arc::clone(worker),
            changes,
            shutdown_rx,
            slots: Arc::clone(&self.subscriptions),
            indexers: Arc::clone(&self.indexers),
        };
        let task_address = address.clone();
        let task_url = url.clone();
        let slots = Arc::clone(&self.subscriptions);

        #[expect(
            clippy::disallowed_methods,
            reason = "KV event monitor: runs for the lifetime of the engine's workers, \
                      handle is stored and graceful shutdown is sent on the last removal"
        )]
        let handle = tokio::spawn(async move {
            // Catch panics here so they surface when they happen — a bare
            // JoinError would only be observed at worker removal, leaving the
            // index silently frozen for this engine until then.
            let result = std::panic::AssertUnwindSafe(Self::subscription_loop(task))
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
                    engine = %Endpoint::redacted_str(&task_address),
                    panic.message = %msg,
                    ?kind,
                    "KV event subscription task panicked; this engine's stream \
                     no longer feeds routing"
                );
                // Counted under the engine's workers (the series the
                // dashboards key on), or under its first worker when none
                // is left on the task.
                let members = {
                    let subs = slots.lock().await;
                    Self::members_of(&subs, &task_address, id)
                };
                if members.is_empty() {
                    Metrics::record_kv_event_subscription_failure(&task_url, "panic");
                }
                for member in &members {
                    Metrics::record_kv_event_subscription_failure(member, "panic");
                }
            }
            Self::complete_engine_removal(&slots, &task_address, id).await;
            // Closes `done` last: an add waiting on it finds the engine free.
            drop(done_tx);
        });

        let member = slot(id);
        subs.engines.insert(
            address,
            EngineSubscription {
                id,
                handle,
                kind,
                changes: changes_tx,
                members: 1,
                shutdown_tx: Some(shutdown_tx),
                done,
            },
        );
        subs.workers.insert(url, member);
    }

    /// What every worker gets as it goes on a stream: its integrity series,
    /// which exist before its stream does, and the model's provisional
    /// block size from its spec (an events subscription; the event stream
    /// overwrites it with the backend's actual page size once received).
    fn prepare_worker(
        &self,
        worker: &Arc<dyn Worker>,
        url: &str,
        model_id: &str,
        kind: SubscriptionKind,
    ) {
        Metrics::init_kv_event_series(url);
        if let Some(bs) = worker
            .metadata()
            .spec
            .kv_block_size
            .filter(|_| kind == SubscriptionKind::Events)
        {
            if bs > 0 {
                let known = self
                    .block_sizes
                    .entry(model_id.to_string())
                    .or_insert(KvBlockSize {
                        tokens: bs,
                        from_events: false,
                    });
                // Under the entry's lock, as the learn site writes it, so a
                // size a stream learns meanwhile is not overwritten on the
                // gauge by the seed.
                Metrics::set_kv_block_size(model_id, known.tokens);
            } else {
                warn!(worker_url = %Endpoint::redacted_str(url), "Worker reports kv_block_size=0, ignoring");
            }
        }
    }

    /// The end of a worker's removal, run by its engine's task once the
    /// worker's blocks are out of the index (or by the remover when the task
    /// is gone): lift the URL's reservation if it is this removal's (a later
    /// worker under the URL has its own), and drop the model's index with
    /// the model's last events member, zeroing the gauges that index fed (a
    /// loads-only member had none to drop). The task does this rather than
    /// `on_worker_removed`, whose future a caller may drop before the
    /// cleanup has ended; a slot that is a member (the worker was never told
    /// to leave: the task ended on its own, or `stop` took the subscription)
    /// is left as it is. `worker` is the removed worker itself, held through
    /// the write made here: the series store forgets a removed address once
    /// no handle on its workers is left, and a write for the address by name
    /// alone after that would recreate its series.
    async fn complete_member_removal(
        slots: &Mutex<Subscriptions>,
        indexers: &DashMap<String, Arc<KvIndex>>,
        block_sizes: &DashMap<String, KvBlockSize>,
        worker: &Arc<dyn Worker>,
        id: u64,
        model_id: &str,
        kind: SubscriptionKind,
    ) {
        let worker_url = worker.url();
        let mut subs = slots.lock().await;
        Self::complete_member_removal_locked(
            &mut subs,
            indexers,
            block_sizes,
            worker_url,
            id,
            model_id,
            kind,
        );
    }

    /// [`Self::complete_member_removal`] under the lock the caller holds.
    fn complete_member_removal_locked(
        subs: &mut Subscriptions,
        indexers: &DashMap<String, Arc<KvIndex>>,
        block_sizes: &DashMap<String, KvBlockSize>,
        worker_url: &str,
        id: u64,
        model_id: &str,
        kind: SubscriptionKind,
    ) {
        if !matches!(subs.workers.get(worker_url), Some(Slot::Removing(held)) if held.id == id) {
            return;
        }
        // A backend without KV events is no longer a blind spot once its
        // worker has left: lowered here, under the slot lock, so a re-add of
        // the URL that follows the removal cannot see its fresh gauge reset
        // by a stale caller, and a caller that dropped `on_worker_removed`
        // early does not leave it raised.
        Metrics::set_kv_events_unavailable(worker_url, false);
        subs.workers.remove(worker_url);
        // Under the slot lock, which `on_worker_added` holds from its lookup
        // of the model's index to its insert: an add of the model racing
        // with this either keeps the index (its slot is already in) or
        // creates the next one.
        if kind != SubscriptionKind::Events {
            return;
        }
        let last_of_model = !subs.workers.values().any(|slot| {
            matches!(slot, Slot::Member { model_id: other, kind: SubscriptionKind::Events, .. }
                if other == model_id)
        });
        if last_of_model {
            indexers.remove(model_id);
            block_sizes.remove(model_id);
            Self::publish_dropped_index(model_id);
        }
    }

    /// The worker URLs on the engine subscription `id` at `address`: the
    /// series a failure of the task is counted under.
    fn members_of(subs: &Subscriptions, address: &str, id: u64) -> Vec<String> {
        subs.workers
            .iter()
            .filter_map(|(url, slot)| match slot {
                Slot::Member {
                    address: slot_address,
                    engine,
                    ..
                } if slot_address == address && *engine == id => Some(url.clone()),
                _ => None,
            })
            .collect()
    }

    /// The end of an engine's task: its entry leaves the map if it is still
    /// this task's (a later subscription under the address has its own).
    async fn complete_engine_removal(slots: &Mutex<Subscriptions>, address: &str, id: u64) {
        let mut subs = slots.lock().await;
        if matches!(subs.engines.get(address), Some(engine) if engine.id == id) {
            subs.engines.remove(address);
        }
    }

    /// Take a worker off its engine's KV event subscription.
    ///
    /// The engine's task takes the worker's blocks out of the index from the
    /// worker's own per-worker map; that CPU-bound cleanup runs on the
    /// bounded blocking pool rather than a Tokio runtime worker. The engine's
    /// other workers keep the stream; its last worker's removal signals the
    /// task's graceful shutdown.
    /// A write made here once the engine's task is gone (the gauge of a
    /// member that left before the removal) is made under the caller's
    /// handle on the removed worker, which the removal step holds through
    /// this call.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        let mut done = {
            let mut subs = self.subscriptions.lock().await;
            let (address, model_id, kind, on_engine) = match subs.workers.remove(worker_url) {
                Some(Slot::Member {
                    address,
                    model_id,
                    kind,
                    engine,
                }) => (address, model_id, kind, engine),
                Some(removing @ Slot::Removing(_)) => {
                    // Another removal of the same URL is already under way.
                    subs.workers.insert(worker_url.to_string(), removing);
                    return;
                }
                None => return,
            };
            let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
            let (done_tx, done) = watch::channel(());
            subs.workers.insert(
                worker_url.to_string(),
                Slot::Removing(Reservation {
                    id,
                    done: done.clone(),
                }),
            );
            // The member leaves the task it is on; a slot of an ended task
            // (its successor has the address now) is retired below without
            // touching the successor's count.
            let sent = match subs.engines.get_mut(&address) {
                Some(engine) if engine.id == on_engine && engine.members > 0 => {
                    engine.members -= 1;
                    let sent = engine
                        .changes
                        .send(MemberChange::Leave {
                            url: worker_url.to_string(),
                            id,
                            done: done_tx,
                        })
                        .is_ok();
                    if engine.members == 0 {
                        info!(engine = %Endpoint::redacted_str(&address), "Stopping KV event subscription");
                        if let Some(shutdown_tx) = engine.shutdown_tx.take() {
                            let _ = shutdown_tx.send(());
                        }
                    }
                    sent
                }
                _ => false,
            };
            if !sent {
                // The engine's task is gone (the backend serves no KV
                // events, or the task failed): nothing holds the worker's
                // blocks, so its removal completes here.
                Self::complete_member_removal_locked(
                    &mut subs,
                    &self.indexers,
                    &self.block_sizes,
                    worker_url,
                    id,
                    &model_id,
                    kind,
                );
                return;
            }
            info!(worker_url = %Endpoint::redacted_str(worker_url), "Worker leaves its engine's KV event subscription");
            done
        };
        // The task lifts the reservation and drops the model's last index on
        // its own (`complete_member_removal`); this wait is for the caller,
        // who has the worker out of the index on return. A caller that drops
        // the future here loses nothing. Nothing is ever sent on the
        // channel; this returns at its close.
        let _ = done.changed().await;
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        let (engines, members) = {
            let mut subs = self.subscriptions.lock().await;
            // The workers of each engine, the series a join error is counted
            // under once the slots are gone.
            let members: HashMap<String, Vec<String>> = subs
                .engines
                .iter()
                .map(|(address, engine)| {
                    (address.clone(), Self::members_of(&subs, address, engine.id))
                })
                .collect();
            let Subscriptions { engines, workers } = &mut *subs;
            // A removal in flight lifts its own reservation; the members
            // leave with their engines' tasks.
            workers.retain(|_, slot| matches!(slot, Slot::Removing(_)));
            let engines: Vec<(String, EngineSubscription)> = engines.drain().collect();
            (engines, members)
        };

        if !engines.is_empty() {
            info!(count = engines.len(), "Stopping all KV event subscriptions");
            for (address, mut engine) in engines {
                debug!(engine = %Endpoint::redacted_str(&address), "Stopping KV event subscription");
                if let Some(shutdown_tx) = engine.shutdown_tx.take() {
                    let _ = shutdown_tx.send(());
                }
                if let Err(e) = engine.handle.await {
                    error!(
                        engine = %Endpoint::redacted_str(&address),
                        error = %e,
                        "KV event subscription task failed"
                    );
                    // Counted under the engine's workers (the series the
                    // dashboards key on), or under the address when none
                    // was left on it.
                    let urls = members.get(&address).map(Vec::as_slice).unwrap_or_default();
                    if urls.is_empty() {
                        Metrics::record_kv_event_subscription_failure(&address, "join_error");
                    }
                    for url in urls {
                        Metrics::record_kv_event_subscription_failure(url, "join_error");
                    }
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
        if let Ok(mut subs) = self.subscriptions.try_lock() {
            for (_, mut engine) in subs.engines.drain() {
                if let Some(shutdown_tx) = engine.shutdown_tx.take() {
                    let _ = shutdown_tx.send(());
                }
                engine.handle.abort(); // Can't await in Drop, abort as fallback
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

    /// The late-write test's worker address: a closed port, so its task loops
    /// on reconnects and answers shutdown with no engine involved, and one no
    /// other test of this binary registers. The series store is the
    /// process's, and a registration of the same address by a test on another
    /// thread would clear its retirement under this test's assertions.
    const REFUSING_URL: &str = "grpc://127.0.0.1:2";

    /// A gRPC worker at [`REFUSING_URL`].
    fn refusing_grpc_worker() -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(REFUSING_URL)
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

    /// A member's cleanup writes for its worker after the worker has left
    /// the engine's stream (the gauge `complete_member_removal` lowers), and
    /// it keeps a handle on the worker through that write: the series store
    /// forgets a removed address once no handle on its workers is left, so a
    /// late write by name alone would recreate the address's series for good.
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
                // The engine's task starts at this thread's first yield. From
                // then on it holds the worker as its member: two strong counts
                // with this test's.
                wait_until("the task holds the worker", || witness.strong_count() >= 2).await;
                assert!(listed(), "the subscription publishes the worker's series");

                // The registry's removal retires the address with the worker's
                // witness; from here the task alone holds the worker.
                Metrics::remove_worker_metrics(&url, Arc::downgrade(&worker));
                drop(worker);
                // The removal up to the member's leave, as `on_worker_removed`
                // does it, under the slot lock this test keeps: the member's
                // cleanup waits at that lock once its handle is the last.
                let mut subs = monitor.subscriptions.lock().await;
                let Some(Slot::Member {
                    address, engine, ..
                }) = subs.workers.remove(&url)
                else {
                    panic!("a member of its engine's subscription");
                };
                let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
                let (done_tx, mut done) = watch::channel(());
                subs.workers.insert(
                    url.clone(),
                    Slot::Removing(Reservation {
                        id,
                        done: done.clone(),
                    }),
                );
                let sub = subs
                    .engines
                    .get_mut(&address)
                    .filter(|sub| sub.id == engine)
                    .expect("the engine's subscription");
                sub.members -= 1;
                sub.changes
                    .send(MemberChange::Leave {
                        url: url.clone(),
                        id,
                        done: done_tx,
                    })
                    .expect("the engine's task takes the change");
                if sub.members == 0 {
                    if let Some(shutdown_tx) = sub.shutdown_tx.take() {
                        let _ = shutdown_tx.send(());
                    }
                }
                wait_until("the member's cleanup holds the worker alone", || {
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

                drop(subs);
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
