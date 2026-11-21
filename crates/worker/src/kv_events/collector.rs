//! The lifecycle of the workers' KV event subscriptions: one task a worker,
//! started by an add and ended by a removal, each under the slot lock. What
//! an events subscription feeds is the host's ([`KvEventHost`]).

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock, Weak,
    },
};

use futures::FutureExt as _;
use tokio::{
    sync::{oneshot, watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info};

use super::{
    port::KvEventSink,
    subscription::{KvEventFeed, PushedLoadSink, Rounds, SubscriptionKind},
};
use crate::{
    common_proto::KvEventBatch,
    identity::{basic::Worker, endpoint::Endpoint},
    load::monitor::WorkerMonitor,
    metrics::writers::Metrics,
    registry::{Generation, WorkerRegistry},
    worker::ConnectionMode,
    UNKNOWN_MODEL_ID,
};

/// What the collector asks of the side its events subscriptions feed. A
/// loads-only subscription calls none of it.
pub trait KvEventHost: Send + Sync + 'static {
    /// The sink a worker's events subscription feeds.
    type Sink: KvEventSink<Batch = KvEventBatch> + 'static;
    /// What an add hands the task it starts.
    type Ticket: Send + 'static;

    /// An events subscription of `model` is added for `worker`.
    ///
    /// Called under the slot lock, which orders it against
    /// [`model_left`](Self::model_left): it must not wait, and it must not
    /// call the collector.
    fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) -> Self::Ticket;

    /// The task opens the worker's place in the sink, ahead of its first
    /// round. `None` ends the task: the host has said why and counted it.
    /// The slot stays taken, so the worker is not offered again before its
    /// removal or its next registration.
    fn open(
        &self,
        ticket: Self::Ticket,
        worker: &Arc<dyn Worker>,
        model: &str,
    ) -> Option<KvEventFeed<Self::Sink>>;

    /// The model's last events subscription is gone: no slot of the model
    /// is left, active or held by a removal still under way. Told once each
    /// time the model's slots go to none.
    ///
    /// Called under the slot lock like [`enroll`](Self::enroll), by whoever
    /// frees the subscription's slot: it must not wait, and it must not
    /// call the collector. Neither a stop nor a dropped collector calls
    /// it: what the host keeps for a model is then the host's to clear.
    fn model_left(&self, _model: &str) {}
}

/// The lifecycle of the workers' KV event subscriptions: one task a worker,
/// started by an add and ended by a removal, each under the slot lock. What
/// an events subscription feeds is the host's.
pub struct KvEventCollector<H: KvEventHost> {
    host: Arc<H>,
    /// Per-worker subscription slots: worker_url → the live subscription, or
    /// the reservation a removal leaves until the old task has cleaned up.
    /// One mutex over the map, so a slot is read and replaced on one turn;
    /// Arc so a task can lift its own reservation at the end of its exit
    /// path.
    worker_handles: Arc<Mutex<HashMap<String, Slot>>>,
    /// Where the load records on the event streams go (`KvEventBatch.load`):
    /// the load monitor, which treats them as polls. Weak: the two are
    /// peers in their host, neither owns the other.
    load_sink: OnceLock<Weak<dyn PushedLoadSink>>,
    /// The registry whose object under a URL an add starts a stream for:
    /// `on_worker_added` reads it under the slot lock. Weak like the load
    /// sink, a peer in the host. Unset in a collector built alone (the
    /// tests), which holds every object.
    worker_registry: OnceLock<Weak<dyn WorkerLookup>>,
    /// Set by [`Self::stop`] and never cleared: a stopped collector starts
    /// no subscription. Written and read under the slot lock, so an add
    /// finds it on the turn that would have started its task.
    stopped: AtomicBool,
}

/// The registry as the collector holds it: it asks one thing of it, the
/// worker registered under a URL. Such a holder reads no generation, so it
/// names no value of the registry's host.
trait WorkerLookup: Send + Sync {
    /// The worker registered under `url`, as [`WorkerRegistry::get_by_url`]
    /// reads it.
    fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>>;
}

impl<G: Generation> WorkerLookup for WorkerRegistry<G> {
    fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        Self::get_by_url(self, url)
    }
}

/// A worker URL's slot in the subscription map.
enum Slot {
    Active(WorkerSubscription),
    /// The URL's old task has been told to shut down and is telling the sink
    /// that the worker left. A sink may hand a URL its existing place until
    /// that cleanup has released it, so a subscription opened in this window
    /// would write under the old place and lose its blocks to the cleanup;
    /// `on_worker_added` waits for the old task to end instead.
    Removing(Reservation),
}

/// What a removal leaves in a URL's slot until the old task has ended.
///
/// The task lifts it itself at the end of its exit path
/// ([`KvEventCollector::lift_turn`]), and `done` closes when the task
/// is gone for any reason (an abort, a panic past its guard). Nothing about
/// it depends on the `on_worker_removed` future living on: the registry's
/// removal step runs it under a timeout, and a cleanup waiting on the
/// removal permits can outlast that; a reservation only the remover could
/// lift would then hold every later add of the URL for good. A task that
/// had made its own lift turn before its removal began leaves no
/// reservation: the removal frees the slot on the turn that finds it
/// ([`KvEventCollector::begin_removal`]). One that never reaches its lift
/// (an abort, a panic past its guard) leaves the reservation to the
/// remover after its wait, and to the URL's next add where that remover
/// went first ([`KvEventCollector::lift_reservation`]).
struct Reservation {
    /// The subscription the reservation belongs to; a task lifts only its own.
    id: u64,
    /// Closes when the task has ended.
    done: watch::Receiver<()>,
    /// The subscription's model and kind: whoever lifts the reservation
    /// tells the host when the model's last events subscription went with
    /// it.
    model_id: String,
    kind: SubscriptionKind,
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
    /// Signals the subscription task to shut down gracefully. On exit the
    /// task tells the sink the worker left, and the sink drops what it
    /// holds of the worker.
    shutdown_tx: oneshot::Sender<()>,
    /// Closes when the task has ended; a removal moves it into the
    /// reservation it leaves in the slot.
    done: watch::Receiver<()>,
    /// The task has made its own lift turn and found this slot active: it
    /// writes nothing more, and only its channel is still to close. Set
    /// under the slot lock, where a removal's first turn reads it.
    ended: bool,
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

impl<H: KvEventHost> KvEventCollector<H> {
    /// A collector with no subscription, for the host given.
    pub fn new(host: Arc<H>) -> Self {
        Self {
            host,
            worker_handles: Arc::new(Mutex::new(HashMap::new())),
            load_sink: OnceLock::new(),
            worker_registry: OnceLock::new(),
            stopped: AtomicBool::new(false),
        }
    }

    /// Route the load records on every worker's event stream to the worker
    /// monitor (once; a later call is ignored).
    pub fn set_load_sink<G: Generation>(&self, monitor: &Arc<WorkerMonitor<G>>) {
        let sink = Arc::downgrade(monitor);
        let _ = self.load_sink.set(sink);
    }

    /// Have the add read the registry before it stops or starts a stream
    /// (once; a later call is ignored): a worker the registry no longer
    /// holds under its URL gets none. The host sets it where it wires the
    /// load sink; a collector without one holds every object.
    pub fn set_worker_registry<G: Generation>(&self, registry: &Arc<WorkerRegistry<G>>) {
        let registry = Arc::downgrade(registry);
        let _ = self.worker_registry.set(registry);
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

    /// Start a worker's stream: its KV events into the host's sink
    /// (`Events`), or its load records alone (`Loads`), on a background task
    /// that subscribes over server-streaming gRPC. Either way the records on
    /// the stream reach the load sink. A second call for the object already
    /// streaming is a no-op. A call with another object under an active URL
    /// restarts the stream for it: a re-registration replaced the worker in
    /// place (an upsert of a changed spec, a replacement by id, a label
    /// refresh; the host calls this at each, for the object the registry
    /// holds), and the records on the stream must reach that object, since
    /// the load snapshot fences the pushed loads of any other object out and
    /// its receipts suppress the poll of the URL. The restart waits for the
    /// old task to end; a caller whose timeout drops this call in that wait
    /// leaves the removal to the old task's own exit path and the URL
    /// without a stream until the next add, the worker's next registration
    /// or label refresh.
    ///
    /// The add reads the registry it was given (`set_worker_registry`)
    /// under the slot lock on every turn, before it stops a stream for
    /// another object and before it inserts the new one, so a worker the
    /// registry no longer holds (removed, or replaced again while the add
    /// waited for an old task to end) gets no stream, and a stream another
    /// object has under the URL is left to it. The registry's removal
    /// changes the registry before it reaches this lock: a read here sees
    /// the removal, or the removal finds the slot this add inserts. A
    /// collector without a registry holds every object. A stopped
    /// collector starts nothing: the add reads that on the same turns.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>, kind: SubscriptionKind) {
        let url = worker.url().to_string();
        // An empty model id is keyed as `UNKNOWN_MODEL_ID`.
        let model_id = Self::normalize_model_id(worker.model_id());

        // Only gRPC workers stream KV events. HTTP workers don't, and a ZMQ
        // connection answers `unimplemented` to the subscribe call:
        // subscribing there would be a wasted handshake and round trip per
        // registration.
        if *worker.connection_mode() != ConnectionMode::Grpc {
            debug!(worker_url = %Endpoint::redacted_str(&url), mode = %worker.connection_mode(), "non-gRPC worker, skipping KV event subscription");
            return;
        }

        let mut handles = loop {
            let mut handles = self.worker_handles.lock().await;
            // Under the slot lock, on every turn: a stopped collector starts
            // nothing, so no task begins behind the stop that would have
            // ended it.
            if self.stopped.load(Ordering::Relaxed) {
                debug!(worker_url = %Endpoint::redacted_str(&url), "no KV event stream from a stopped collector");
                return;
            }
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
                let stale = Self::begin_removal(&mut handles, &self.host, &url);
                drop(handles);
                info!(worker_url = %Endpoint::redacted_str(&url), "Restarting KV event subscription for a re-registered worker");
                if let Some(sub) = stale {
                    self.finish_removal(&url, sub).await;
                }
                continue;
            }
            let (reserved, mut done) = match handles.get(&url) {
                Some(Slot::Active(_)) => {
                    debug!(worker_url = %Endpoint::redacted_str(&url), "KV event subscription already active, skipping");
                    return;
                }
                Some(Slot::Removing(reservation)) => (reservation.id, reservation.done.clone()),
                None => break handles,
            };
            // The previous subscription for this URL is still leaving the
            // sink: open it again only once its task has ended.
            if done.has_changed().is_err() {
                // The task is gone without lifting its reservation (it was
                // aborted or panicked past its guard, and no remover lifted
                // behind it): nothing is left to wait for. The reservation
                // is lifted here as its task would have lifted it, the gauge
                // lowered and the host told of the model's last events
                // subscription, before its successor starts, whose answer
                // owns the gauge from here.
                Self::lift_reservation(&mut handles, &self.host, &url, reserved);
                break handles;
            }
            drop(handles);
            // Nothing is ever sent on the channel; this returns at its close.
            let _ = done.changed().await;
        };

        // A loads-only stream feeds nothing of the host's: the host hears of
        // the model with its first events subscription.
        let ticket = match kind {
            SubscriptionKind::Events => Some(self.host.enroll(worker, &model_id)),
            SubscriptionKind::Loads => None,
        };
        // The worker's integrity series exist before its stream does.
        Metrics::init_kv_event_series(&url);

        let started_for = Arc::downgrade(worker);
        let worker = Arc::clone(worker);
        let worker_url = url.clone();
        let host = Arc::clone(&self.host);

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
        let slots = Arc::clone(&self.worker_handles);

        #[expect(
            clippy::disallowed_methods,
            reason = "KV event collector: runs for the lifetime of the worker, \
                      handle is stored and graceful shutdown is sent on removal"
        )]
        let handle = tokio::spawn(async move {
            // The task writes for the worker after the loop has returned it
            // (the panic counter, the gauge `lift_turn` lowers). It
            // keeps a handle on the worker through those writes: the series
            // store forgets a removed address once no handle on its workers
            // is left, and a write for the address by name alone after that
            // would recreate its series.
            let writer = Arc::clone(&worker);
            // Catch panics here so they surface when they happen — a bare
            // JoinError would only be observed at worker removal, leaving the
            // sink silently without this worker's events until then.
            let opener = &host;
            let result = std::panic::AssertUnwindSafe(async move {
                // An events subscription opens its place in the host's sink
                // first. The place is the collector's from here on: it goes
                // with the loop, or with the job that says the worker left.
                let feed = match ticket {
                    Some(ticket) => {
                        // `None`: the host has said why and counted it.
                        let Some(feed) = opener.open(ticket, &worker, &loop_model_id) else {
                            return;
                        };
                        Some(feed)
                    }
                    None => None,
                };
                Rounds::subscription_loop(
                    worker,
                    worker_url,
                    feed,
                    loop_model_id,
                    shutdown_rx,
                    load_sink,
                )
                .await;
            })
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
            Self::lift_turn(&slots, &host, &writer, id).await;
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
                ended: false,
            }),
        );
    }

    /// The task's own lift turn: its last turn of the slot lock, at the end
    /// of its exit path. Behind a removal, once the sink has heard that the
    /// worker left, it lifts the URL's reservation if it is this task's (a
    /// later subscription under the URL has its own) and tells the host
    /// when the model's last events subscription went with it. The task
    /// does this itself, since a caller may drop `on_worker_removed` before
    /// the task has ended. A task that ended on its own finds its slot
    /// still `Active` and marks it ended: the removal that comes frees the
    /// slot on its first turn, whether or not the task's channel has closed
    /// by then. A slot `stop` took is gone, and a later subscription's is
    /// left as it is. `worker` is the removed worker itself, which the task
    /// holds through the write made here (see the task in
    /// [`Self::on_worker_added`]).
    async fn lift_turn(
        slots: &Mutex<HashMap<String, Slot>>,
        host: &H,
        worker: &Arc<dyn Worker>,
        id: u64,
    ) {
        let mut handles = slots.lock().await;
        match handles.get_mut(worker.url()) {
            Some(Slot::Active(sub)) if sub.id == id => sub.ended = true,
            _ => Self::lift_reservation(&mut handles, host, worker.url(), id),
        }
    }

    /// Lift the reservation `id` under `worker_url`, if it still stands
    /// there: free the slot, lower the worker's gauge, and tell the host
    /// when the model's last events subscription went with it
    /// ([`KvEventHost::model_left`]). The caller holds the slot lock.
    /// Whoever finds the subscription's task ended behind its reservation
    /// calls it: the task at its own end, the remover after its wait (an
    /// aborted task, or one that panicked past its guard, lifted nothing),
    /// and an add that finds the reservation's task gone.
    fn lift_reservation(handles: &mut HashMap<String, Slot>, host: &H, worker_url: &str, id: u64) {
        if !matches!(handles.get(worker_url), Some(Slot::Removing(held)) if held.id == id) {
            return;
        }
        // The slot was read as this reservation just above, under the lock
        // the caller holds: the `else` is never taken.
        let Some(Slot::Removing(held)) = handles.remove(worker_url) else {
            return;
        };
        Self::subscription_gone(handles, host, worker_url, &held.model_id, held.kind);
    }

    /// A subscription's task has ended and its slot is free: lower the
    /// worker's gauge, and tell the host when the model's last events
    /// subscription went with it. The caller holds the slot lock.
    fn subscription_gone(
        handles: &HashMap<String, Slot>,
        host: &H,
        worker_url: &str,
        model_id: &str,
        kind: SubscriptionKind,
    ) {
        // A backend without KV events is no longer a blind spot once its
        // worker has left: lowered here, under the slot lock and only by
        // whoever freed this subscription's slot, so a re-add of the URL
        // that follows the removal cannot see its fresh gauge reset by a
        // stale caller, and a caller that dropped `on_worker_removed` early
        // does not leave it raised.
        Metrics::set_kv_events_unavailable(worker_url, false);
        // Under the slot lock, which `on_worker_added` holds from its
        // `enroll` to its insert: an add of the model racing with this is
        // either counted below (its slot is already in) or enrolls after the
        // host has heard that the model left.
        if kind != SubscriptionKind::Events {
            return;
        }
        // A reservation counts until it is lifted: its task may still feed
        // the sink, and its own lift tells the host if it is the last.
        let last_of_model = !handles.values().any(|slot| {
            let (other_kind, other_model) = match slot {
                Slot::Active(other) => (other.kind, &other.model_id),
                Slot::Removing(other) => (other.kind, &other.model_id),
            };
            other_kind == SubscriptionKind::Events && other_model == model_id
        });
        if last_of_model {
            host.model_left(model_id);
        }
    }

    /// Stop the KV event subscription for a worker.
    ///
    /// Sends a graceful shutdown signal. The subscription task tells the
    /// sink the worker left, and the sink drops what it holds of the worker;
    /// that cleanup can be CPU-bound, so it runs on the bounded blocking pool
    /// rather than a Tokio runtime worker. A write made here once the task is
    /// gone (the join error, the gauge of a task that ended before the
    /// removal) is made under the caller's handle on the removed worker,
    /// which the host's removal holds through this call.
    ///
    /// A host calls it after the registry let the worker go. A registration
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
            Self::begin_removal(&mut handles, &self.host, worker_url)
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
            Self::begin_removal(&mut handles, &self.host, url)
        };
        if let Some(sub) = stale {
            info!(worker_url = %Endpoint::redacted_str(url), "Ending the KV event subscription of a replaced worker: its successor streams nothing");
            self.finish_removal(url, sub).await;
        }
    }

    /// Under the slot lock: the URL's active subscription taken out. One
    /// whose task still runs leaves its reservation in the slot, for the
    /// adds that come before the task has ended. One whose task has made
    /// its own lift turn (the slot is marked ended), or is gone with its
    /// channel closed, leaves none: that task lifts nothing more, so the
    /// slot is freed on this turn, as a lift frees it, and a caller that
    /// goes away after this turn leaves nothing behind. `None` for an
    /// empty slot or a removal already under way.
    fn begin_removal(
        handles: &mut HashMap<String, Slot>,
        host: &H,
        worker_url: &str,
    ) -> Option<WorkerSubscription> {
        match handles.remove(worker_url) {
            Some(Slot::Active(sub)) => {
                if sub.ended || sub.done.has_changed().is_err() {
                    Self::subscription_gone(handles, host, worker_url, &sub.model_id, sub.kind);
                    return Some(sub);
                }
                handles.insert(
                    worker_url.to_string(),
                    Slot::Removing(Reservation {
                        id: sub.id,
                        done: sub.done.clone(),
                        model_id: sub.model_id.clone(),
                        kind: sub.kind,
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
    /// its end, then its reservation lifted where it still stands.
    async fn finish_removal(&self, worker_url: &str, sub: WorkerSubscription) {
        info!(worker_url = %Endpoint::redacted_str(worker_url), "Stopping KV event subscription");
        // Signal graceful shutdown: the task tells the sink the worker left.
        let _ = sub.shutdown_tx.send(());
        // The task lifts the reservation on its own (`lift_turn`); this
        // wait is for the caller, who has the worker out of the sink on
        // return. A caller that drops the future here leaves the lift to
        // the task and loses the join error below. Panics are caught inside
        // the task; a JoinError here (abort or a panic that escaped the
        // guard) must still be surfaced, not discarded.
        let id = sub.id;
        if let Err(e) = sub.handle.await {
            error!(
                worker_url = %Endpoint::redacted_str(worker_url),
                error = %e,
                "KV event subscription task failed"
            );
            Metrics::record_kv_event_subscription_failure(worker_url, "join_error");
        }
        // A task that ends on the shutdown signal lifts its reservation in
        // its own cleanup (`lift_turn`). One that did not reach
        // that step (an abort, a panic past its guard) left the
        // reservation in place: lifted here, under the slot lock and only
        // while the reservation is still this subscription's. A caller
        // dropped at this lock leaves it to the URL's next add. Behind a
        // task that had made its own lift turn before the removal began
        // nothing is left to lift: the first turn freed the slot. A
        // successor under the URL owns the gauge from its add on.
        let mut handles = self.worker_handles.lock().await;
        Self::lift_reservation(&mut handles, &self.host, worker_url, id);
    }

    /// Stop every subscription, for good. The active tasks are told to stop
    /// and awaited, and the task of a removal under way is awaited too: on
    /// return no task of the collector runs, and an add that comes later,
    /// or that was waiting for a turn of the slot lock, starts none. What
    /// the host keeps for a model is the host's to clear: a stop tells it
    /// of no model.
    pub async fn stop(&self) {
        let (subscriptions, leaving) = {
            let mut handles = self.worker_handles.lock().await;
            // Under the lock, ahead of the slots taken below: an add on a
            // later turn finds the mark.
            self.stopped.store(true, Ordering::Relaxed);
            let mut active = Vec::new();
            let mut leaving = Vec::new();
            for (url, slot) in std::mem::take(&mut *handles) {
                match slot {
                    Slot::Active(sub) => active.push((url, sub)),
                    // The task behind a removal under way lifts its own
                    // reservation: the stop waits for its end below.
                    Slot::Removing(reserved) => {
                        leaving.push(reserved.done.clone());
                        handles.insert(url, Slot::Removing(reserved));
                    }
                }
            }
            (active, leaving)
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
        // Nothing is ever sent on a reservation's channel: each wait returns
        // when its task has ended.
        for mut done in leaving {
            let _ = done.changed().await;
        }
    }

    /// The ids under a URL's slot: the active subscription's, or the
    /// reservation's of a removal under way.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn slot_ids(&self, worker_url: &str) -> (Option<u64>, Option<u64>) {
        let handles = self.worker_handles.lock().await;
        Self::ids_under(&handles, worker_url)
    }

    /// Whether the task under a URL's slot has ended: `true` too for an
    /// empty slot, `false` while a removal is under way.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn task_ended(&self, worker_url: &str) -> bool {
        match self.worker_handles.lock().await.get(worker_url) {
            Some(Slot::Active(sub)) => sub.handle.is_finished(),
            Some(Slot::Removing(_)) => false,
            None => true,
        }
    }

    /// Leave the URL's running stream as a reservation whose task is gone:
    /// a removal's first turn, then its task ended short of its own lift
    /// and no remover behind it. For the cases of the add that finds such a
    /// reservation.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn strand_reservation(&self, worker_url: &str) {
        let taken = {
            let mut handles = self.worker_handles.lock().await;
            Self::begin_removal(&mut handles, &self.host, worker_url)
        };
        if let Some(sub) = taken {
            sub.handle.abort();
            let _ = sub.handle.await;
        }
    }

    /// [`Self::slot_ids`] under a lock the caller holds.
    #[cfg(any(test, feature = "test-util"))]
    fn ids_under(handles: &HashMap<String, Slot>, worker_url: &str) -> (Option<u64>, Option<u64>) {
        match handles.get(worker_url) {
            Some(Slot::Active(sub)) => (Some(sub.id), None),
            Some(Slot::Removing(held)) => (None, Some(held.id)),
            None => (None, None),
        }
    }

    /// The key a model's subscriptions are held under: an empty model id
    /// maps to `UNKNOWN_MODEL_ID`.
    fn normalize_model_id(model_id: &str) -> String {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            model_id.to_string()
        }
    }
}

impl<H: KvEventHost> Drop for KvEventCollector<H> {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tracing_test::traced_test;

    use super::*;
    use crate::{
        kv_events::{
            port::{BatchFacts, BatchOutcome, KvCursor, ResyncReason},
            MAX_CONCURRENT_SINK_REMOVALS, SINK_REMOVAL_PERMITS,
        },
        test_support::{BasicWorkerBuilder, WorkerRegistry},
        worker::{HealthCheckConfig, RuntimeType, WorkerType},
    };

    /// A host that records what the collector asks of it, over a sink that
    /// takes nothing in.
    #[derive(Default)]
    struct RecordingHost {
        calls: parking_lot::Mutex<Vec<String>>,
        /// Whether `open` refuses: the task then ends by itself, at once.
        refuses_open: bool,
    }

    impl RecordingHost {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }
    }

    impl KvEventHost for RecordingHost {
        type Sink = NullSink;
        type Ticket = ();

        fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) {
            self.calls
                .lock()
                .push(format!("enroll {model} {}", worker.url()));
        }

        fn open(
            &self,
            (): (),
            worker: &Arc<dyn Worker>,
            model: &str,
        ) -> Option<KvEventFeed<NullSink>> {
            self.calls
                .lock()
                .push(format!("open {model} {}", worker.url()));
            (!self.refuses_open).then(|| (Arc::new(NullSink), ()))
        }

        fn model_left(&self, model: &str) {
            self.calls.lock().push(format!("model_left {model}"));
        }
    }

    /// A sink that admits nothing and keeps nothing.
    struct NullSink;

    impl KvEventSink for NullSink {
        type Batch = KvEventBatch;
        type Id = ();

        fn admit(&self, (): (), _: u16, _: BatchFacts, _: &KvEventBatch) -> BatchOutcome {
            BatchOutcome::Skipped
        }

        fn reconnected(&self, (): ()) {}

        fn resync(&self, (): (), _: Option<u16>, _: ResyncReason) {}

        fn cursor(&self, (): (), _: u16) -> Option<KvCursor> {
            None
        }

        fn removed(&self, (): ()) {}

        fn block_size(&self, _: &str) -> Option<u32> {
            None
        }

        fn set_block_size(&self, _: &str, _: u32) {}
    }

    /// A collector over a recording host.
    fn new_collector() -> KvEventCollector<RecordingHost> {
        KvEventCollector::new(Arc::new(RecordingHost::default()))
    }

    /// An events subscription asks the host in order: `enroll` at the add,
    /// `open` in the task, and `model_left` once the model's last events
    /// subscription is gone. A loads-only one asks it nothing.
    #[tokio::test]
    async fn an_events_subscription_asks_the_host_and_a_loads_only_one_does_not() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let loads = unconnected_grpc_worker_at("grpc://127.0.0.1:4");
        collector
            .on_worker_added(&loads, SubscriptionKind::Loads)
            .await;
        assert!(host.calls().is_empty(), "a loads-only one asks nothing");

        // Both serve one model. The loads-only subscription does not keep
        // the model for the host, and its own removal tells the host
        // nothing more.
        let events = unconnected_grpc_worker_at("grpc://127.0.0.1:5");
        collector
            .on_worker_added(&events, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;
        collector.on_worker_removed(events.url()).await;
        let asked = [
            "enroll unknown grpc://127.0.0.1:5",
            "open unknown grpc://127.0.0.1:5",
            "model_left unknown",
        ];
        assert_eq!(host.calls(), asked, "the loads-only one keeps no model");
        collector.on_worker_removed(loads.url()).await;
        assert_eq!(host.calls(), asked, "its removal asks nothing");
    }

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        for _ in 0..200 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    /// The late-write test's worker address, one no other test of this binary
    /// registers. The series store is the process's, and a registration of
    /// the same address by a test on another thread would clear its
    /// retirement under this test's assertions.
    const LATE_WRITE_URL: &str = "grpc://127.0.0.1:2";

    /// A gRPC worker at [`LATE_WRITE_URL`].
    fn unconnected_grpc_worker() -> Arc<dyn Worker> {
        unconnected_grpc_worker_at(LATE_WRITE_URL)
    }

    /// The replacement test's address: its registry
    /// registrations publish the address's series, which the late-write
    /// test's retirement assertions on [`LATE_WRITE_URL`] must not see.
    const REPLACED_URL: &str = "grpc://127.0.0.1:3";

    /// A gRPC worker at `url` that never connects: it has no engine handle,
    /// so every round of its task backs off, and the task lives until its
    /// shutdown with no engine involved.
    fn unconnected_grpc_worker_at(url: &str) -> Arc<dyn Worker> {
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
    /// returned it (the gauge `lift_turn` lowers), and it keeps a
    /// handle on the worker through that write: the series store forgets a
    /// removed address once no handle on its workers is left, so a late write
    /// by name alone would recreate the address's series for good.
    #[test]
    fn a_removals_late_write_recreates_no_series_for_the_address() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::metrics::series::{self as worker_series, WorkerSeriesRecorder};

        let series = worker_series::installed_or_default();
        let recorder = WorkerSeriesRecorder::new(
            PrometheusBuilder::new().build_recorder(),
            Arc::clone(series),
        );
        let listed = || {
            let mut rendered = String::new();
            series.render_into(&mut rendered);
            rendered
                .lines()
                .any(|line| line.contains(&format!("worker=\"{LATE_WRITE_URL}\"")))
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // The spawned task runs on this thread, under the local recorder.
        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let collector = new_collector();
                let worker = unconnected_grpc_worker();
                let url = worker.url().to_string();
                let witness = Arc::downgrade(&worker);
                collector
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
                let mut handles = collector.worker_handles.lock().await;
                let Some(Slot::Active(sub)) = handles.remove(&url) else {
                    panic!("an active subscription");
                };
                let mut done = sub.done.clone();
                handles.insert(
                    url.clone(),
                    Slot::Removing(Reservation {
                        id: sub.id,
                        done: sub.done.clone(),
                        model_id: sub.model_id.clone(),
                        kind: sub.kind,
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

    /// A re-registration replaces the worker object under its URL (an
    /// upsert of a changed spec, a replacement by id, a label refresh) and
    /// the host adds it again. The stream started for the old object
    /// must not stay: its records would reach an object the registry no
    /// longer holds, whose pushed loads the load snapshot fences out and
    /// whose receipts suppress the URL's poll. The add with the new object
    /// restarts the stream for it; a second add of the same object changes
    /// nothing.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_re_registration_under_the_url_restarts_the_stream_for_the_new_object() {
        let collector = Arc::new(new_collector());
        let first = unconnected_grpc_worker();
        let second = unconnected_grpc_worker();
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        assert!(
            !logs_contain("Restarting KV event subscription"),
            "the same object's second add keeps its stream"
        );

        let old = Arc::downgrade(&first);
        drop(first);
        collector
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
        collector.stop().await;
    }

    /// A replacement that streams nothing (its host wants neither its events
    /// nor its loads) ends the stream the object it replaced had under the
    /// URL: the records on that stream would keep reaching an object the
    /// registry no longer holds, and nothing else on the replace path stops
    /// it.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_that_streams_nothing_ends_the_stream_of_the_object_it_replaced() {
        let collector = Arc::new(new_collector());
        let first = unconnected_grpc_worker();
        let url = first.url().to_string();
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let old = Arc::downgrade(&first);
        drop(first);

        let second = unconnected_grpc_worker();
        collector.on_worker_added_without_stream(&second).await;
        assert!(logs_contain(
            "Ending the KV event subscription of a replaced worker"
        ));
        assert!(
            old.upgrade().is_none(),
            "the old stream's task has ended and released the old object"
        );
        assert_eq!(
            collector.slot_ids(&url).await,
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
        collector.stop().await;
    }

    /// The registry replaced the worker again before its call ran, and the
    /// add for the object it holds now has restarted the stream: the call
    /// for the object nobody holds leaves that stream alone. A removal by
    /// URL here would take it.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_that_streams_nothing_leaves_the_stream_the_registry_holds() {
        let collector = Arc::new(new_collector());
        let registry = Arc::new(WorkerRegistry::new());
        collector.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = unconnected_grpc_worker_at(url);
        registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        drop(first);

        let second = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        let third = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&third));
        collector
            .on_worker_added(&third, SubscriptionKind::Loads)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        let held = active.expect("the add of the third restarted the stream");
        assert_eq!(removing, None);

        collector.on_worker_added_without_stream(&second).await;
        assert_eq!(
            collector.slot_ids(url).await,
            (Some(held), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "no KV event stream to end for a worker the registry no longer holds"
        ));
        collector.stop().await;
    }

    /// A worker that streams nothing now, with the stream its add under a
    /// kind started: that stream stays. The collector never ends or restarts a
    /// stream for the object it was started for; an add with another kind
    /// leaves it too.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_that_streams_nothing_keeps_the_stream_started_for_it() {
        let collector = Arc::new(new_collector());
        let worker = unconnected_grpc_worker();
        let url = worker.url().to_string();
        collector
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        let (active, removing) = collector.slot_ids(&url).await;
        let started = active.expect("the add started a stream");
        assert_eq!(removing, None);

        collector.on_worker_added_without_stream(&worker).await;
        assert_eq!(
            collector.slot_ids(&url).await,
            (Some(started), None),
            "the stream started for the object stays"
        );
        assert!(!logs_contain("Ending the KV event subscription"));
        collector.stop().await;
    }

    /// The removal step runs after the registry let the worker go, and a
    /// registration under the same URL in between may hold a new object
    /// whose add restarted the stream for it. The removal of the old object
    /// leaves that stream: a removal by URL alone would take it and leave
    /// the new object registered with no stream until its next add.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_leaves_the_stream_of_the_object_the_registry_holds() {
        let collector = Arc::new(new_collector());
        let registry = Arc::new(WorkerRegistry::new());
        collector.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = unconnected_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;

        // The removal took the first out; a registration put the second in
        // and its add restarted the stream before the removal's call here.
        assert!(
            registry.remove(&id).is_some(),
            "the first leaves the registry"
        );
        let second = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        collector
            .on_worker_added(&second, SubscriptionKind::Loads)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        let held = active.expect("the add of the second restarted the stream");
        assert_eq!(removing, None);

        collector.on_worker_removed(url).await;
        assert_eq!(
            collector.slot_ids(url).await,
            (Some(held), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "KV event stream left to the worker the registry holds under the URL"
        ));
        collector.stop().await;
    }

    /// The registry holds a new object whose add has not run yet: the
    /// stream under the URL is still the removed object's, and the removal
    /// ends it. The new object's add then starts afresh.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_ends_an_old_stream_under_an_object_not_yet_added() {
        let collector = Arc::new(new_collector());
        let registry = Arc::new(WorkerRegistry::new());
        collector.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = unconnected_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        assert!(
            registry.remove(&id).is_some(),
            "the first leaves the registry"
        );
        let second = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));

        collector.on_worker_removed(url).await;
        assert_eq!(
            collector.slot_ids(url).await,
            (None, None),
            "the first's stream ended and its reservation was lifted"
        );
        assert!(logs_contain("Stopping KV event subscription"));
        collector.stop().await;
    }

    /// With nobody under the URL the removal ends the stream: the registry
    /// is read, not assumed, and an unheld address keeps no stream.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_with_nobody_under_the_url_ends_the_stream() {
        let collector = Arc::new(new_collector());
        let registry = Arc::new(WorkerRegistry::new());
        collector.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let worker = unconnected_grpc_worker_at(url);
        let id = registry
            .register(Arc::clone(&worker))
            .expect("the registration under the address");
        collector
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        assert!(
            registry.remove(&id).is_some(),
            "the worker leaves the registry"
        );

        collector.on_worker_removed(url).await;
        assert_eq!(
            collector.slot_ids(url).await,
            (None, None),
            "the stream ended and its reservation was lifted"
        );
        assert!(!logs_contain("left to the worker the registry holds"));
        collector.stop().await;
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
        let collector = Arc::new(new_collector());
        let registry = Arc::new(WorkerRegistry::new());
        collector.set_worker_registry(&registry);
        let url = REPLACED_URL;
        let first = unconnected_grpc_worker_at(url);
        registry
            .register(Arc::clone(&first))
            .expect("the first registration under the address");
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        let stopped = active.expect("the first add started a stream");
        assert_eq!(removing, None);
        drop(first);

        // The second replaces the first in place. Polled against a held
        // lock, its restart queues for its first turn, and a read of the
        // slot queues right behind it.
        let second = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&second));
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        let mut restart =
            std::pin::pin!(collector.on_worker_added(&second, SubscriptionKind::Loads));
        let mut read = std::pin::pin!(collector.worker_handles.lock());
        let held = collector.worker_handles.lock().await;
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
                KvEventCollector::<RecordingHost>::ids_under(&handles, url),
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
        let third = unconnected_grpc_worker_at(url);
        registry.register_or_replace(Arc::clone(&third));
        tokio::time::timeout(
            Duration::from_secs(30),
            collector.on_worker_added(&third, SubscriptionKind::Loads),
        )
        .await
        .expect("the third add ends once the old task has");
        let (active, removing) = collector.slot_ids(url).await;
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
            collector.slot_ids(url).await,
            (Some(started), None),
            "the stream of the object the registry holds stays"
        );
        assert!(logs_contain(
            "no KV event stream for a worker the registry no longer holds"
        ));
        tokio::time::timeout(Duration::from_secs(30), collector.stop())
            .await
            .expect("the stop ends the stream left");
    }

    /// A task that ended by itself leaves its slot active (a host that
    /// refuses the open, an engine that serves no events). The removal that
    /// finds it frees the slot on that same turn of the lock and tells the
    /// host, so a caller that goes away right after that turn leaves no
    /// reservation behind. The removal queues for a held lock with a read of
    /// the slot right behind it: the lock is fair, so the read runs the
    /// moment the removal's first turn ends.
    #[tokio::test]
    async fn a_removal_behind_an_ended_task_frees_the_slot_on_its_own_lock_turn() {
        let host = Arc::new(RecordingHost {
            refuses_open: true,
            ..Default::default()
        });
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:6");
        let url = worker.url();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let mut ended = false;
        for _ in 0..200 {
            ended = collector.task_ended(url).await;
            if ended {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(ended, "the task ends behind the refused open");
        let (active, removing) = collector.slot_ids(url).await;
        assert!(
            active.is_some(),
            "a task that ended by itself keeps its slot"
        );
        assert_eq!(removing, None);

        {
            let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
            let mut removal = std::pin::pin!(collector.on_worker_removed(url));
            let mut read = std::pin::pin!(collector.worker_handles.lock());
            let held = collector.worker_handles.lock().await;
            assert!(
                std::future::Future::poll(removal.as_mut(), &mut cx).is_pending(),
                "the removal queues for the held lock"
            );
            assert!(
                std::future::Future::poll(read.as_mut(), &mut cx).is_pending(),
                "the read queues behind the removal"
            );
            drop(held);
            assert!(
                std::future::Future::poll(removal.as_mut(), &mut cx).is_pending(),
                "the removal's first turn ends in its wait for the lock the read has"
            );
            let handles = read.await;
            assert_eq!(
                KvEventCollector::<RecordingHost>::ids_under(&handles, url),
                (None, None),
                "the turn that found the ended task freed the slot"
            );
            // The removal is dropped here in its wait, as a timeout drops it.
        }
        assert_eq!(
            host.calls(),
            [
                "enroll unknown grpc://127.0.0.1:6",
                "open unknown grpc://127.0.0.1:6",
                "model_left unknown",
            ]
        );
        assert_eq!(collector.slot_ids(url).await, (None, None));

        // Nothing is left to wait for: the address's next add starts afresh.
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        assert!(active.is_some(), "the next add starts a stream at once");
        assert_eq!(removing, None);
        collector.stop().await;
    }

    /// The add that restarts a stream waits for the old task to end. A
    /// caller's timeout that drops it in that wait (a host may run the add
    /// under one) leaves the removal to the old task's own exit path and the
    /// URL without a stream; the next add of the new object starts one
    /// afresh, with no second restart.
    #[traced_test]
    #[tokio::test]
    async fn an_add_dropped_while_it_waits_for_the_old_task_leaves_the_next_add_a_fresh_start() {
        let collector = Arc::new(new_collector());
        let first = unconnected_grpc_worker();
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let old = Arc::downgrade(&first);
        drop(first);

        let second = unconnected_grpc_worker();
        {
            // One poll: the slot is found streaming for another object, the
            // old task is told to stop, and the add waits for its end. Then
            // the add is dropped, as a timeout around it drops it.
            let mut add =
                std::pin::pin!(collector.on_worker_added(&second, SubscriptionKind::Loads));
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

        collector
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
        collector.stop().await;
    }

    /// A worker removed and added again under its URL while the old task
    /// still waits to tell the sink that the worker left: the add waits for
    /// that task's end, so the host hears that the model left before the new
    /// subscription enrolls. The removal permits are the process's: held
    /// here, they hold the old task at its last word to the sink.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_same_url_re_add_waits_for_the_old_tasks_end() {
        let host = Arc::new(RecordingHost::default());
        let collector = Arc::new(KvEventCollector::new(Arc::clone(&host)));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:10");
        let url = worker.url().to_string();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;

        let permits = SINK_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
            .await
            .expect("the removal semaphore is open");
        #[expect(
            clippy::disallowed_methods,
            reason = "the removal and the re-add are awaited below"
        )]
        let removal = tokio::spawn({
            let collector = Arc::clone(&collector);
            let url = url.clone();
            async move { collector.on_worker_removed(&url).await }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!removal.is_finished(), "the removal waits for the old task");
        #[expect(
            clippy::disallowed_methods,
            reason = "the removal and the re-add are awaited below"
        )]
        let re_add = tokio::spawn({
            let collector = Arc::clone(&collector);
            let worker = Arc::clone(&worker);
            async move {
                collector
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!re_add.is_finished(), "the re-add waits for the old task");
        assert_eq!(host.calls().len(), 2, "the host hears nothing meanwhile");

        drop(permits);
        removal.await.expect("removal task");
        re_add.await.expect("re-add task");
        assert_eq!(
            host.calls()[..4],
            [
                "enroll unknown grpc://127.0.0.1:10",
                "open unknown grpc://127.0.0.1:10",
                "model_left unknown",
                "enroll unknown grpc://127.0.0.1:10",
            ]
        );
        collector.stop().await;
    }

    /// A caller may drop the removal before the old task has ended (a
    /// registry's removal step runs under a timeout). The reservation it
    /// left does not outlive the task: the task lifts it at its own end,
    /// with nobody awaiting the removal, and the host hears that the model
    /// left.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_removal_leaves_its_reservation_to_the_task() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:11");
        let url = worker.url();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;
        let (reserved, _) = collector.slot_ids(url).await;

        let permits = SINK_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
            .await
            .expect("the removal semaphore is open");
        let removal =
            tokio::time::timeout(Duration::from_millis(200), collector.on_worker_removed(url))
                .await;
        assert!(
            removal.is_err(),
            "the removal was still waiting for the old task"
        );
        assert_eq!(
            collector.slot_ids(url).await,
            (None, reserved),
            "the reservation stands while the old task has not ended"
        );
        assert_eq!(host.calls().len(), 2, "the host hears nothing meanwhile");

        drop(permits);
        wait_until("the old task lifts its reservation", || {
            host.calls().len() == 3
        })
        .await;
        assert_eq!(host.calls()[2], "model_left unknown");
        assert_eq!(collector.slot_ids(url).await, (None, None));
    }

    /// An add can find a reservation whose task is gone and that nobody
    /// lifted. It lifts it as the task would have, before it starts the
    /// worker's stream: the host hears that the model's last events
    /// subscription left, ahead of the new one's `enroll`.
    #[tokio::test]
    async fn an_add_behind_a_dead_reservation_tells_the_host_the_model_left() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:7");
        let url = worker.url();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;
        let (stranded, _) = collector.slot_ids(url).await;
        collector.strand_reservation(url).await;
        assert_eq!(collector.slot_ids(url).await, (None, stranded));

        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        assert!(
            active.is_some_and(|id| Some(id) != stranded),
            "the add starts a stream of its own"
        );
        assert_eq!(removing, None, "the add lifted the dead reservation");
        assert_eq!(
            host.calls()[..4],
            [
                "enroll unknown grpc://127.0.0.1:7",
                "open unknown grpc://127.0.0.1:7",
                "model_left unknown",
                "enroll unknown grpc://127.0.0.1:7",
            ]
        );
        collector.stop().await;
    }

    /// The same add while another worker of the model still streams its
    /// events: the model has not left, and the host hears nothing of it.
    #[tokio::test]
    async fn an_add_behind_a_dead_reservation_keeps_a_model_another_worker_feeds() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let stays = unconnected_grpc_worker_at("grpc://127.0.0.1:8");
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:9");
        let url = worker.url();
        collector
            .on_worker_added(&stays, SubscriptionKind::Events)
            .await;
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("both tasks open their places", || host.calls().len() == 4).await;
        collector.strand_reservation(url).await;

        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let (active, removing) = collector.slot_ids(url).await;
        assert!(active.is_some(), "the add starts a stream of its own");
        assert_eq!(removing, None, "the add lifted the dead reservation");
        let calls = host.calls();
        assert!(
            !calls.iter().any(|call| call.starts_with("model_left")),
            "another worker still feeds the model: {calls:?}"
        );
        collector.stop().await;
    }

    /// A task that ends by itself makes its own lift turn: it finds its
    /// slot still active and marks it ended there, under the slot lock.
    #[tokio::test]
    async fn a_task_that_ends_by_itself_marks_its_slot_ended() {
        let host = Arc::new(RecordingHost {
            refuses_open: true,
            ..Default::default()
        });
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:15");
        let url = worker.url();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        let mut marked = false;
        for _ in 0..200 {
            marked = matches!(
                collector.worker_handles.lock().await.get(url),
                Some(Slot::Active(sub)) if sub.ended
            );
            if marked {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            marked,
            "the task marks its active slot on its own lift turn"
        );
        collector.on_worker_removed(url).await;
        assert_eq!(collector.slot_ids(url).await, (None, None));
    }

    /// Between its own lift turn and its end a task has marked its slot
    /// ended while its channel is still open. The removal that begins then
    /// frees the slot on its first turn and tells the host: no reservation
    /// is left behind a task that lifts nothing more. The slot is laid here
    /// as such a task leaves it, since the two steps of a real task are a
    /// few instructions apart.
    #[tokio::test]
    async fn a_removal_behind_a_task_past_its_lift_frees_the_slot_on_its_first_turn() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:16");
        let url = worker.url();
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();
        // The sender is held to the case's end: the channel stays open.
        let (_done_tx, done) = watch::channel(());
        #[expect(
            clippy::disallowed_methods,
            reason = "the case's stand-in for a subscription's task ends at once"
        )]
        let handle = tokio::spawn(async {});
        let mut handles = collector.worker_handles.lock().await;
        handles.insert(
            url.to_string(),
            Slot::Active(WorkerSubscription {
                id: NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed),
                handle,
                model_id: "m".to_string(),
                kind: SubscriptionKind::Events,
                started_for: Arc::downgrade(&worker),
                shutdown_tx,
                done,
                ended: true,
            }),
        );
        let taken = KvEventCollector::<RecordingHost>::begin_removal(&mut handles, &host, url);
        assert!(taken.is_some(), "the removal takes the subscription out");
        assert!(
            !handles.contains_key(url),
            "no reservation is left behind a task past its lift"
        );
        assert_eq!(host.calls(), ["model_left m"]);
    }

    /// A task that never reaches its own lift turn (an abort, a panic past
    /// its guard) leaves its reservation to the remover: the removal lifts
    /// it after its wait, frees the slot and tells the host.
    #[tokio::test]
    async fn a_removal_behind_a_task_short_of_its_lift_lifts_after_its_wait() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:20");
        let url = worker.url();
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;

        // The removal's first turn finds the task alive and leaves its
        // reservation; then the task is gone short of its lift turn.
        let sub = {
            let mut handles = collector.worker_handles.lock().await;
            KvEventCollector::<RecordingHost>::begin_removal(&mut handles, &host, url)
        }
        .expect("the removal takes the subscription out");
        assert_eq!(collector.slot_ids(url).await, (None, Some(sub.id)));
        sub.handle.abort();
        collector.finish_removal(url, sub).await;

        assert_eq!(collector.slot_ids(url).await, (None, None));
        assert_eq!(
            host.calls().last().map(String::as_str),
            Some("model_left unknown"),
            "the remover's lift tells the host"
        );
    }

    /// The host hears that a model left once, when the model's last slot is
    /// freed: a reservation counts until it is lifted, since its task may
    /// still feed the sink. Two reservations of one model are laid here as
    /// two removals' first turns leave them.
    #[tokio::test]
    async fn a_model_leaves_with_its_last_reservation() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let urls = ["grpc://127.0.0.1:21", "grpc://127.0.0.1:22"];
        // The senders are held to the case's end: both tasks still run.
        let (_first_tx, first_done) = watch::channel(());
        let (_second_tx, second_done) = watch::channel(());
        let mut handles = collector.worker_handles.lock().await;
        let mut ids = Vec::new();
        for (url, done) in urls.into_iter().zip([first_done, second_done]) {
            let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
            ids.push(id);
            handles.insert(
                url.to_string(),
                Slot::Removing(Reservation {
                    id,
                    done,
                    model_id: "m".to_string(),
                    kind: SubscriptionKind::Events,
                }),
            );
        }

        KvEventCollector::<RecordingHost>::lift_reservation(&mut handles, &host, urls[0], ids[0]);
        assert!(
            host.calls().is_empty(),
            "the model keeps a subscription that is still leaving"
        );
        KvEventCollector::<RecordingHost>::lift_reservation(&mut handles, &host, urls[1], ids[1]);
        assert_eq!(host.calls(), ["model_left m"]);
    }

    /// A task's lift turn marks its own slot alone: one that makes it after
    /// a later subscription stands under its address leaves that slot as it
    /// is. Marked by the old task, the later subscription's removal would
    /// free the slot at once while its task still runs.
    #[tokio::test]
    async fn a_lift_turn_behind_a_later_subscription_marks_nothing() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:23");
        let url = worker.url();
        // The id of an older subscription of the address, whose slot is gone.
        let old = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed);
        collector
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        let (later, _) = collector.slot_ids(url).await;
        assert!(later.is_some(), "the later subscription is active");

        KvEventCollector::<RecordingHost>::lift_turn(
            &collector.worker_handles,
            &host,
            &worker,
            old,
        )
        .await;

        assert!(
            matches!(
                collector.worker_handles.lock().await.get(url),
                Some(Slot::Active(sub)) if !sub.ended
            ),
            "the old task leaves the later subscription unmarked"
        );
        assert_eq!(collector.slot_ids(url).await, (later, None));
        collector.on_worker_removed(url).await;
    }

    /// A stop ends the tasks and tells the host of no model: what the host
    /// keeps for a model is the host's to clear.
    #[tokio::test]
    async fn a_stop_tells_the_host_of_no_model() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:12");
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;
        collector.stop().await;
        assert_eq!(collector.slot_ids(worker.url()).await, (None, None));
        assert_eq!(
            host.calls(),
            [
                "enroll unknown grpc://127.0.0.1:12",
                "open unknown grpc://127.0.0.1:12",
            ],
            "a stop asks the host nothing"
        );
    }

    /// A stop is for good: an add that comes after it starts no task, takes
    /// no slot and asks the host nothing.
    #[tokio::test]
    async fn an_add_after_a_stop_starts_nothing() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        collector.stop().await;
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:17");
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        assert_eq!(collector.slot_ids(worker.url()).await, (None, None));
        assert!(
            host.calls().is_empty(),
            "a stopped collector asks the host nothing"
        );
    }

    /// The add reads the stop on each of its turns: one that waited for an
    /// old task to end while the stop came starts nothing on its next turn,
    /// where it would have started a task the stop never ends.
    #[tokio::test]
    async fn an_add_that_waited_through_a_stop_starts_nothing() {
        let collector = new_collector();
        let url = "grpc://127.0.0.1:18";
        let first = unconnected_grpc_worker_at(url);
        collector
            .on_worker_added(&first, SubscriptionKind::Loads)
            .await;
        let (old, _) = collector.slot_ids(url).await;
        assert!(old.is_some(), "the first object's stream is active");

        // Another object under the address: its add takes the old stream
        // out and waits for the old task to end.
        let second = unconnected_grpc_worker_at(url);
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        let mut restart =
            std::pin::pin!(collector.on_worker_added(&second, SubscriptionKind::Loads));
        assert!(
            std::future::Future::poll(restart.as_mut(), &mut cx).is_pending(),
            "the restart's first turn ends in its wait for the old task"
        );
        assert_eq!(collector.slot_ids(url).await, (None, old));

        collector.stop().await;
        restart.await;
        assert_eq!(
            collector.slot_ids(url).await,
            (None, None),
            "the add starts nothing behind the stop"
        );
    }

    /// A stop returns once no task of the collector runs: it waits for the
    /// task of a removal under way too, which lifts its own reservation. The
    /// reservation is laid here as a removal's first turn leaves it, and the
    /// case holds the channel's end that such a task holds.
    #[tokio::test]
    async fn a_stop_waits_for_the_task_of_a_removal_under_way() {
        let collector = new_collector();
        let (done_tx, done) = watch::channel(());
        collector.worker_handles.lock().await.insert(
            "grpc://127.0.0.1:19".to_string(),
            Slot::Removing(Reservation {
                id: NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::Relaxed),
                done,
                model_id: "m".to_string(),
                kind: SubscriptionKind::Loads,
            }),
        );

        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        let mut stop = std::pin::pin!(collector.stop());
        assert!(
            std::future::Future::poll(stop.as_mut(), &mut cx).is_pending(),
            "the stop waits for the task of the removal under way"
        );
        // The task ends: its channel closes.
        drop(done_tx);
        stop.await;
    }

    /// A dropped collector aborts its tasks, and an aborted task tells the
    /// host of no model either.
    #[tokio::test]
    async fn a_dropped_collector_tells_the_host_of_no_model() {
        let host = Arc::new(RecordingHost::default());
        let collector = KvEventCollector::new(Arc::clone(&host));
        let worker = unconnected_grpc_worker_at("grpc://127.0.0.1:13");
        collector
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        wait_until("the task opens its place", || host.calls().len() == 2).await;
        drop(collector);
        wait_until("the aborted task lets go of the host", || {
            Arc::strong_count(&host) == 1
        })
        .await;
        assert_eq!(
            host.calls(),
            [
                "enroll unknown grpc://127.0.0.1:13",
                "open unknown grpc://127.0.0.1:13",
            ],
            "a drop asks the host nothing"
        );
    }

    /// A refused open is final for the worker object: the slot stays taken
    /// and a second add of the same object asks the host nothing. A new
    /// object under the URL is a new registration: the model leaves with
    /// the old subscription, then the new one enrolls and asks to open.
    #[tokio::test]
    async fn a_refused_open_is_final_for_the_worker_object() {
        let host = Arc::new(RecordingHost {
            refuses_open: true,
            ..Default::default()
        });
        let collector = KvEventCollector::new(Arc::clone(&host));
        let url = "grpc://127.0.0.1:14";
        let refused = unconnected_grpc_worker_at(url);
        collector
            .on_worker_added(&refused, SubscriptionKind::Events)
            .await;
        wait_until("the host refuses the open", || host.calls().len() == 2).await;
        collector
            .on_worker_added(&refused, SubscriptionKind::Events)
            .await;
        assert_eq!(
            host.calls().len(),
            2,
            "the same object is not offered again"
        );

        let next = unconnected_grpc_worker_at(url);
        collector
            .on_worker_added(&next, SubscriptionKind::Events)
            .await;
        wait_until("the new object asks to open", || host.calls().len() == 5).await;
        assert_eq!(
            host.calls(),
            [
                "enroll unknown grpc://127.0.0.1:14",
                "open unknown grpc://127.0.0.1:14",
                "model_left unknown",
                "enroll unknown grpc://127.0.0.1:14",
                "open unknown grpc://127.0.0.1:14",
            ]
        );
    }

    #[tokio::test]
    async fn stopping_an_empty_collector_is_a_no_op() {
        let collector = new_collector();
        collector.stop().await;
    }

    #[tokio::test]
    async fn removing_an_unknown_worker_is_a_no_op() {
        let collector = new_collector();
        collector.on_worker_removed("http://nonexistent:8000").await;
    }
}
