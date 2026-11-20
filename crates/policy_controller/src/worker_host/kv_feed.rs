//! A host reads its workers' cache events into its own store's index: every
//! replica of the controller, and a gateway that decides in its own process.
//!
//! # Limits
//!
//! Eligible workers keep streaming when the active generation reads no index.

use std::{collections::HashSet, sync::Arc};

use smg_worker::{
    identity::basic::Worker,
    kv_events::{KvEventCollector, SubscriptionKind},
    registry::{event::WorkerEvent, Generation, WorkerId, WorkerRegistry},
    worker::ConnectionMode,
};
use tokio::sync::{broadcast::error::RecvError, watch};

use super::{KvHost, Objects};
use crate::store::{index::indexed_under, MemoryStore, WorkersStore};

/// A host's cache-event collector and the worker objects it follows.
pub struct KvFeed<G: Generation> {
    collector: KvEventCollector<KvHost<G, MemoryStore>>,
    registry: Arc<WorkerRegistry<G>>,
    store: Arc<MemoryStore>,
    objects: Arc<Objects<G>>,
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
        }
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
        if *worker.connection_mode() == ConnectionMode::Grpc
            && self
                .objects
                .standing(id)
                .and_then(|standing| self.store.get(standing.row))
                .is_some_and(|row| indexed_under(&row).is_some())
        {
            self.collector
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        } else {
            self.collector.on_worker_added_without_stream(worker).await;
        }
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
