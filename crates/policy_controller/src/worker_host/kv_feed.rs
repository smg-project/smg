//! A host reads its workers' cache events into its own store's index: every
//! replica of the controller, and a gateway that decides in its own process.
//! A host that hands the follow its worker monitor
//! ([`KvFeed::set_load_sink`]) reads the load records the streams carry as
//! well. A gRPC worker whose row is not indexed then streams its load
//! records alone, and so does a rank above 0 of a data-parallel worker,
//! which opens no events feed.
//!
//! # Limits
//!
//! A row written over the controller's wire states its own signal, so such
//! a worker keeps streaming when the active generation reads no index. A
//! row found at a host's door is stated by the generation in force at that
//! write ([`asks_events`]). The follow is told of a worker once for each
//! object laid for it, so a later generation moves a standing worker's
//! stream when its row is laid again at the door, and not before.

use std::{
    collections::HashSet,
    sync::{Arc, OnceLock, Weak},
};

use smg_worker::{
    identity::basic::Worker,
    kv_events::{KvEventCollector, SubscriptionKind},
    load::monitor::WorkerMonitor,
    registry::{event::WorkerEvent, Generation, WorkerId, WorkerRegistry},
    worker::ConnectionMode,
};
use tokio::sync::{broadcast::error::RecvError, watch};

use super::{KvHost, Objects};
use crate::{
    store::{index::indexed_under, world::phase_of, MemoryStore, WorkerFacts, WorkersStore},
    view::WireModes,
    Engine,
};

/// A host's cache-event collector and the worker objects it follows.
pub struct KvFeed<G: Generation> {
    collector: KvEventCollector<KvHost<G, MemoryStore>>,
    registry: Arc<WorkerRegistry<G>>,
    store: Arc<MemoryStore>,
    objects: Arc<Objects<G>>,
    /// The host's worker monitor, once handed: it says whether a worker's
    /// loads are wanted. The collector keeps its own hold of it, for the
    /// records.
    loads: OnceLock<Weak<WorkerMonitor<G>>>,
}

impl<G: Generation> KvFeed<G> {
    /// The feed over `store` and its worker objects. [`Self::run`] starts
    /// the streams.
    pub fn new(store: Arc<MemoryStore>, objects: Arc<Objects<G>>) -> Self {
        let registry = Arc::clone(objects.registry());
        let collector = KvEventCollector::new(Arc::new(KvHost::new(
            Arc::clone(&objects),
            Arc::clone(&store),
        )));
        collector.set_worker_registry(&registry);
        Self {
            collector,
            registry,
            store,
            objects,
            loads: OnceLock::new(),
        }
    }

    /// Hand the follow the host's worker monitor (once; a later call is
    /// ignored). The load records every stream carries then reach the
    /// monitor, and a gRPC worker whose row is not indexed gets a stream
    /// for its load records alone where the monitor wants them
    /// ([`WorkerMonitor::loads_wanted`]). A worker already told keeps the
    /// stream it has, so a host calls this before [`Self::run`].
    pub fn set_load_sink(&self, monitor: &Arc<WorkerMonitor<G>>) {
        self.collector.set_load_sink(monitor);
        let _ = self.loads.set(Arc::downgrade(monitor));
    }

    /// The collector, for the stock controller's cases of the follow.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn collector(&self) -> &KvEventCollector<KvHost<G, MemoryStore>> {
        &self.collector
    }

    /// Follow the registry until `stop` reads true, then end every stream.
    pub async fn run(&self, mut stop: watch::Receiver<bool>) {
        let mut told = HashSet::new();
        // Take events before the whole read so a worker laid during it is heard.
        let mut events = self.registry.subscribe_events();
        self.reconcile(&mut told).await;
        loop {
            let event = tokio::select! {
                _ = stop.wait_for(|stop| *stop) => break,
                event = events.recv() => event,
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
                    self.removed(worker.url(), &mut told).await;
                }
                Ok(WorkerEvent::StatusChanged { .. }) => {}
                Err(RecvError::Lagged(_)) => self.reconcile(&mut told).await,
                Err(RecvError::Closed) => break,
            }
        }
        self.collector.stop().await;
    }

    /// Tell the collector of each worker in the read and each URL missing
    /// from it. Keep the read's URLs for the next turn.
    async fn reconcile(&self, told: &mut HashSet<String>) {
        let mut present = HashSet::new();
        for (id, worker) in self.registry.get_all_with_ids() {
            self.tell(&id, &worker).await;
            present.insert(worker.url().to_owned());
        }
        for url in told.difference(&present) {
            self.collector.on_worker_removed(url).await;
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
        let indexed = worker.dp_rank().is_none_or(|rank| rank == 0)
            && self
                .objects
                .standing(id)
                .and_then(|standing| self.store.get(standing.row))
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

    pub(crate) async fn removed(&self, url: &str, told: &mut HashSet<String>) {
        // A whole read may already have told the collector of the next worker
        // under this URL. Keep tracking it while it stands.
        if self.registry.get_by_url(url).is_none() {
            told.remove(url);
        }
        self.collector.on_worker_removed(url).await;
    }
}

/// Whether a found row asks for its engine's cache events under `engine`:
/// whether a decision reads the index for the row's first model, the one
/// the store indexes the row under, in the phase of the row's role. Only a
/// gRPC worker streams.
pub(super) fn asks_events(facts: &WorkerFacts, engine: &Engine) -> bool {
    facts.wire_modes.contains(WireModes::GRPC)
        && facts
            .models
            .first()
            .is_some_and(|model| engine.needs_for(model, phase_of(facts.role)).events)
}
