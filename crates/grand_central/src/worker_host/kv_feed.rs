//! A host reads its workers' cache events into its own KV tables: every
//! replica of the controller, and a gateway that decides in its own process.
//! A host that hands the follow a sink ([`KvFeed::set_load_sink`]) reads
//! the load records the streams carry as well. Under a host that hands it
//! the monitor that says whose loads are wanted
//! ([`KvFeed::set_loads_wanted`]), a gRPC worker whose row is not indexed
//! streams its load records alone, and so does a rank above 0 of a
//! data-parallel worker, which opens no events feed.
//!
//! # Limits
//!
//! A row written over the controller's wire states its own signal, so such
//! a worker keeps streaming when the active generation reads no index. A
//! row found at a host's door is stated by the generation in force at that
//! write ([`asks_events_under`]). The follow is told of a worker once for each
//! object laid for it, so a later generation moves a standing worker's
//! stream when its row is laid again at the door, and not before.
//!
//! [`asks_events_under`]: super::membership_writer::asks_events_under

use std::{
    collections::HashSet,
    sync::{Arc, OnceLock, Weak},
};

use smg_worker::{
    identity::basic::Worker,
    kv_events::{KvEventCollector, PushedLoadSink, SubscriptionKind},
    load::monitor::WorkerMonitor,
    registry::{event::WorkerEvent, Generation, WorkerId, WorkerRegistry},
    worker::ConnectionMode,
};
use tokio::{
    sync::{broadcast::error::RecvError, watch},
    task::{JoinError, JoinSet},
};

use super::{kv_tables::indexed_under, KvHost, Objects};

/// A host's cache-event collector and the worker objects it follows.
pub struct KvFeed<G: Generation> {
    /// Shared with the tasks the follow hands its removals to.
    collector: Arc<KvEventCollector<KvHost<G>>>,
    registry: Arc<WorkerRegistry<G>>,
    objects: Arc<Objects<G>>,
    /// The host's worker monitor, once handed: it says whether a worker's
    /// loads are wanted. The collector keeps its own weak hold of the
    /// records' sink.
    loads: OnceLock<Weak<WorkerMonitor<G>>>,
}

impl<G: Generation> KvFeed<G> {
    /// The feed over a host's worker objects. [`Self::run`] starts the
    /// streams. It writes the KV tables the holder was handed
    /// ([`Objects::with_kv_tables`]), and reads no store: over a holder
    /// handed none it indexes nothing.
    pub fn new(objects: Arc<Objects<G>>) -> Self {
        let registry = Arc::clone(objects.registry());
        let collector = Arc::new(KvEventCollector::new(Arc::new(KvHost::new(Arc::clone(
            &objects,
        )))));
        collector.set_worker_registry(&registry);
        Self {
            collector,
            registry,
            objects,
            loads: OnceLock::new(),
        }
    }

    /// Hand the follow where the load records on its streams go (once; a
    /// later call is ignored): the host's worker monitor, or a sink of the
    /// host's own, which hands a record on to its monitor or drops it. No
    /// stream is opened for the records' sake: they ride the streams the
    /// follow opens. The follow holds the sink weakly, and the caller keeps
    /// it alive for as long as records should reach it ([`PushedLoadSink`]).
    pub fn set_load_sink<S: PushedLoadSink + 'static>(&self, sink: &Arc<S>) {
        self.collector.set_load_sink(sink);
    }

    /// Hand the follow the monitor that says whose loads are wanted (once;
    /// a later call is ignored). A gRPC worker whose row is not indexed
    /// then gets a stream for its load records alone where the monitor
    /// wants them ([`WorkerMonitor::loads_wanted`]). A worker already told
    /// keeps the stream it has, so a host calls this before [`Self::run`].
    /// The stock controller's serve calls it at its bind, on every replica:
    /// a standby holds such a stream too, and its sink drops the records
    /// until the replica leads.
    pub fn set_loads_wanted(&self, monitor: &Arc<WorkerMonitor<G>>) {
        let _ = self.loads.set(Arc::downgrade(monitor));
    }

    /// Follow the registry until `stop` reads true, then end every stream.
    ///
    /// A removal runs on a task of its own, so one that is slow holds no
    /// add behind it, and not the stop's word. The removals under way end
    /// before the streams are stopped. An add is told in the loop: one
    /// whose address has a removal under way waits there for the old
    /// task's end, and holds the events behind it. A removal that panics
    /// takes the follow down, as a panic in the loop does.
    pub async fn run(&self, mut stop: watch::Receiver<bool>) {
        let mut told = HashSet::new();
        let mut removals = JoinSet::new();
        // Take events before the whole read so a worker laid during it is heard.
        let mut events = self.registry.subscribe_events();
        self.reconcile(&mut told, &mut removals).await;
        loop {
            let event = tokio::select! {
                _ = stop.wait_for(|stop| *stop) => break,
                event = events.recv() => event,
                // An ended removal is reaped as the loop turns.
                Some(ended) = removals.join_next() => {
                    reaped(ended);
                    continue;
                }
            };
            match event {
                Ok(
                    WorkerEvent::Registered { worker_id, worker }
                    | WorkerEvent::Replaced {
                        worker_id,
                        new: worker,
                        ..
                    },
                ) => {
                    self.tell(&worker_id, &worker).await;
                    told.insert(worker.url().to_owned());
                }
                Ok(WorkerEvent::Removed { worker, .. }) => {
                    self.removed(worker.url(), &mut told, &mut removals);
                }
                Ok(WorkerEvent::StatusChanged { .. }) => {}
                Err(RecvError::Lagged(_)) => self.reconcile(&mut told, &mut removals).await,
                Err(RecvError::Closed) => break,
            }
        }
        while let Some(ended) = removals.join_next().await {
            reaped(ended);
        }
        self.collector.stop().await;
    }

    /// Tell the collector of each worker in the read and each URL missing
    /// from it. Keep the read's URLs for the next turn.
    async fn reconcile(&self, told: &mut HashSet<String>, removals: &mut JoinSet<()>) {
        let mut present = HashSet::new();
        for (id, worker) in self.registry.get_all_with_ids() {
            self.tell(&id, &worker).await;
            present.insert(worker.url().to_owned());
        }
        for url in told.difference(&present) {
            self.leave(url, removals);
        }
        *told = present;
    }

    pub(crate) async fn tell(&self, id: &WorkerId, worker: &Arc<dyn Worker>) {
        match self.stream_of(id, worker) {
            Some(kind) => self.collector.on_worker_added(worker, kind).await,
            None => self.collector.on_worker_added_without_stream(worker).await,
        }
    }

    /// The stream a worker gets: its cache events where its row is indexed,
    /// its load records alone where the host's monitor wants them, and none
    /// otherwise. Only a gRPC worker streams. A rank above 0 of a
    /// data-parallel worker opens no events feed, since the host refuses it
    /// one, so it streams as a row that is not indexed.
    pub(crate) fn stream_of(
        &self,
        id: &WorkerId,
        worker: &Arc<dyn Worker>,
    ) -> Option<SubscriptionKind> {
        if *worker.connection_mode() != ConnectionMode::Grpc {
            return None;
        }
        // The record beside the object answers: the facts the holder made
        // the row's lane from. No row of the store is read.
        let indexed = worker.dp_rank().is_none_or(|rank| rank == 0)
            && self
                .objects
                .row_of(id)
                .is_some_and(|row| indexed_under(&row).is_some());
        if indexed {
            return Some(SubscriptionKind::Events);
        }
        self.loads
            .get()
            .and_then(Weak::upgrade)
            .is_some_and(|monitor| monitor.loads_wanted(std::slice::from_ref(worker)))
            .then_some(SubscriptionKind::Loads)
    }

    pub(crate) fn removed(
        &self,
        url: &str,
        told: &mut HashSet<String>,
        removals: &mut JoinSet<()>,
    ) {
        // A whole read may already have told the collector of the next worker
        // under this URL. Keep tracking it while it stands.
        if self.registry.get_by_url(url).is_none() {
            told.remove(url);
        }
        self.leave(url, removals);
    }

    /// Hand `url`'s removal to a task of its own. The collector keeps one
    /// address's calls in order by itself: each takes a turn of the
    /// address's slot under the slot lock. An add that comes behind the
    /// removal's turn waits for the old task's end. One that comes ahead
    /// of it restarts the stream for the object the registry holds, and
    /// the removal then leaves that stream alone.
    fn leave(&self, url: &str, removals: &mut JoinSet<()>) {
        let collector = Arc::clone(&self.collector);
        let url = url.to_owned();
        removals.spawn(async move { collector.on_worker_removed(&url).await });
    }
}

/// A removal's task that panicked takes the follow down, as a panic in the
/// loop does: the panic is raised again where the task is reaped.
fn reaped(ended: Result<(), JoinError>) {
    if let Some(panic) = ended.err().and_then(|error| error.try_into_panic().ok()) {
        std::panic::resume_unwind(panic);
    }
}

#[cfg(all(test, feature = "controller"))]
#[path = "../../tests/unit/worker_host/kv_feed/tests.rs"]
mod tests;
