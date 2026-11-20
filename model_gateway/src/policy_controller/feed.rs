//! The membership door writes the store. Its listener brings the worker
//! registry up to date and publishes membership for decisions in that order.
//! The gateway has no publication loop, so each write asks for publication
//! before it returns. A row holds no credential, so its worker object
//! carries none.

use std::sync::Arc;

use parking_lot::Mutex;
use smg_policy_controller::{
    store::{Maker, MemoryStore, WorkerFacts},
    worker_host::{
        Expect, HealthAdapter, Membership, Objects, Refused, Remover, SoleLead, Standing,
        StoreDoor, Wrote,
    },
};
use smg_worker::{
    connection::engine::EngineConnector,
    health::port::{HealthSink, ProbeAnswer},
    registry::WorkerId,
    worker::WorkerStatus,
};

use crate::routers::common::snapshot::{ModelWorkerSnapshot, WorkerRegistry};

/// The feed of a gateway that decides in its own process: the door its
/// worker workflows write at, and the holder that keeps the registry as
/// the store's rows say.
#[derive(Clone)]
pub(crate) struct Feed {
    /// The membership door: the store's own, for a host with no election.
    pub door: Arc<dyn Membership>,
    /// The one writer of the objects of rows where the controller was picked.
    pub objects: Arc<Objects<ModelWorkerSnapshot>>,
    /// The store the door writes and the holder follows.
    store: Arc<MemoryStore>,
    /// The feed's turn: one write with its listener, or one settle, at a
    /// time.
    turn: Arc<Mutex<()>>,
}

impl Feed {
    /// The health sink of a gateway that keeps its workers in the store.
    pub(crate) fn health_sink(&self) -> Arc<dyn HealthSink> {
        Arc::new(FeedHealth {
            objects: Arc::clone(&self.objects),
            adapter: HealthAdapter::new(
                Arc::clone(&self.objects),
                Arc::clone(&self.store),
                Arc::new(SoleLead),
                SoleLead::TERM,
            ),
            registry: Arc::clone(self.objects.registry()),
        })
    }

    /// Bring the registry to the store's rows, then publish them: the two
    /// calls the door's listener makes after each write. It is here for a
    /// follower that applies a served controller's events; no product line
    /// calls it yet. It takes the feed's turn, so it never runs between a
    /// write at the door and that write's listener.
    pub(crate) fn settle(&self) {
        let _turn = self.turn.lock();
        settle(&self.objects, &self.store);
    }
}

/// The adapter writes a row's health and mirrors its status on the worker.
/// The registry writes the health of workers without a row.
/// The adapter answers stale for those workers, so it cannot promote them.
struct FeedHealth {
    objects: Arc<Objects<ModelWorkerSnapshot>>,
    adapter: HealthAdapter<ModelWorkerSnapshot, MemoryStore>,
    registry: Arc<WorkerRegistry>,
}

impl FeedHealth {
    fn sink(&self, worker_id: &WorkerId) -> &dyn HealthSink {
        if self.objects.standing(worker_id).is_some() {
            &self.adapter
        } else {
            self.registry.as_ref()
        }
    }
}

impl HealthSink for FeedHealth {
    fn probe_outcome(
        &self,
        worker_id: &WorkerId,
        passed: bool,
        revision: u64,
        launched: WorkerStatus,
    ) -> ProbeAnswer {
        self.sink(worker_id)
            .probe_outcome(worker_id, passed, revision, launched)
    }

    fn connected(
        &self,
        worker_id: &WorkerId,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.sink(worker_id).connected(worker_id, revision)
    }
}

/// The order is the rule: the registry holds a row's object before a
/// decision can read the row.
fn settle(objects: &Objects<ModelWorkerSnapshot>, store: &MemoryStore) {
    objects.follow(store);
    store.publish_now();
}

/// The feed over `store`, for the workers of `registry`. The holder lays
/// each object with `connector`, the one the gateway was started with.
pub(crate) fn feed(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    let objects = Arc::new(Objects::new(Arc::clone(registry)).with_connector(connector));
    let door = StoreDoor::new(Arc::clone(store)).telling({
        let objects = Arc::clone(&objects);
        let store = Arc::clone(store);
        move || settle(&objects, &store)
    });
    let turn = Arc::new(Mutex::new(()));
    Feed {
        door: Arc::new(FeedDoor {
            door,
            turn: Arc::clone(&turn),
        }),
        objects,
        store: Arc::clone(store),
        turn,
    }
}

/// The store door releases its turn before calling the listener. Hold the
/// feed's turn through that call, so neither a later write nor a settle
/// can publish a row the holder has yet to read.
struct FeedDoor {
    door: StoreDoor,
    turn: Arc<Mutex<()>>,
}

impl FeedDoor {
    fn write(
        &self,
        write: impl FnOnce(&StoreDoor) -> Result<Wrote, Refused>,
    ) -> Result<Wrote, Refused> {
        let _turn = self.turn.lock();
        write(&self.door)
    }
}

impl Membership for FeedDoor {
    fn found(
        &self,
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.write(|door| door.found(maker, expect, url, state))
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.drained(by, url, seen))
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.gone(by, url, seen))
    }
}

#[cfg(test)]
mod tests;
