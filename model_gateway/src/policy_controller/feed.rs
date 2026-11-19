//! The membership door writes the store. Its listener brings the worker
//! registry up to date and publishes membership for decisions in that order.
//! The gateway has no publication loop, so each write asks for publication
//! before it returns. A row holds no credential, so its worker object
//! carries none.

use std::sync::Arc;

use parking_lot::Mutex;
use smg_policy_controller::{
    store::{Maker, MemoryStore, WorkerFacts},
    worker_host::{Membership, Objects, Refused, StoreDoor, Wrote},
};

use crate::routers::{
    common::snapshot::{ModelWorkerSnapshot, WorkerRegistry},
    grpc::backend_client::BackendConnector,
};

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
}

impl Feed {
    /// Bring the registry to the store's rows, then publish them: the two
    /// calls the door's listener makes after each write. A follower that
    /// applies a served controller's events calls this after each event.
    pub(crate) fn settle(&self) {
        settle(&self.objects, &self.store);
    }
}

/// The order is the rule: the registry holds a row's object before a
/// decision can read the row.
fn settle(objects: &Objects<ModelWorkerSnapshot>, store: &MemoryStore) {
    objects.follow(store);
    store.publish_now();
}

/// The feed over `store`, for the workers of `registry`.
pub(crate) fn feed(store: &Arc<MemoryStore>, registry: &Arc<WorkerRegistry>) -> Feed {
    let objects =
        Arc::new(Objects::new(Arc::clone(registry)).with_connector(BackendConnector::shared()));
    let door = StoreDoor::new(Arc::clone(store)).telling({
        let objects = Arc::clone(&objects);
        let store = Arc::clone(store);
        move || settle(&objects, &store)
    });
    Feed {
        door: Arc::new(FeedDoor {
            door,
            turn: Mutex::new(()),
        }),
        objects,
        store: Arc::clone(store),
    }
}

/// The store door releases its turn before calling the listener. Hold the
/// feed's turn through that call so the listener cannot publish a later
/// write whose row the holder has yet to read.
struct FeedDoor {
    door: StoreDoor,
    turn: Mutex<()>,
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
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.write(|door| door.found(maker, url, state))
    }

    fn drained(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
        self.write(|door| door.drained(maker, url))
    }

    fn gone(&self, maker: &Maker, url: &str) -> Result<Wrote, Refused> {
        self.write(|door| door.gone(maker, url))
    }
}

#[cfg(test)]
mod tests;
