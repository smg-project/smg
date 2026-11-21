//! The membership door writes the store. Its listener brings the worker
//! registry up to date and publishes membership for decisions in that order.
//! Each membership write asks for publication before it returns.
//!
//! A registration of this process hands the door the workers it built, and
//! the holder keeps each as its row's object ([`FeedDoor`]). An object
//! holds what no row states: the worker's credential, its connection
//! settings and its model cards.
//!
//! Under a controller's address the follower writes the store in the door's
//! place, and settles the feed after each event it takes ([`following`]).
//! No registration runs there, so the holder builds each object from its
//! row, and such an object holds no credential.

use std::{sync::Arc, time::Duration};

use parking_lot::Mutex;
use smg_policy_controller::{
    store::{Maker, MemoryStore, WorkerFacts},
    worker_host::{
        Expect, LoadAdapter, Membership, Objects, Over, Refused, Remover, SoleLead, Standing,
        StoreDoor, Wrote,
    },
};
use smg_worker::{
    connection::engine::EngineConnector, identity::basic::Worker, load::port::LoadSink,
};
use tracing::warn;

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
    /// Bring the registry to the store's rows, publish them, then write the
    /// published overload verdicts on the objects: the three calls the
    /// door's listener makes after each write. A follower that applies a
    /// served controller's events calls this after each event. It takes the
    /// feed's turn, so it never runs between a write at the door and that
    /// write's listener.
    pub(crate) fn settle(&self) {
        let _turn = self.turn.lock();
        settle(&self.objects, &self.store);
    }

    /// The store the door writes and the holder follows.
    pub(crate) fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// The load sink of a gateway that keeps its workers in the store: the
    /// crate's adapter under the sole lead. `period` is the interval in
    /// which the collector hears every ready rank. A worker no row stands
    /// for is passed over: nothing of it reaches the store.
    pub(crate) fn load_sink(&self, period: Duration) -> Arc<dyn LoadSink> {
        let adapter = LoadAdapter::new(
            Arc::clone(&self.objects),
            Arc::clone(&self.store),
            Arc::new(SoleLead),
            SoleLead::TERM,
            period,
        );
        if !adapter.period_taken() {
            warn!(
                period_secs = period.as_secs(),
                "The policy store keeps an earlier load report period, so this one is not in force"
            );
        }
        Arc::new(adapter)
    }
}

/// The order is the rule: the registry holds a row's object before a
/// decision can read the row. The store pays the build the membership
/// writes left owed, at once. The objects then take the overload verdicts
/// of the world that stands.
fn settle(objects: &Objects<ModelWorkerSnapshot>, store: &MemoryStore) {
    objects.follow(store);
    store.publish_now();
    objects.mirror_overload(store);
}

/// The feed over `store`, for the workers of `registry`. The holder keeps
/// the object each registration hands the door. A row laid with no object
/// handed in gets one built from it with `connector`, the one the gateway
/// was started with, and the holder warns.
pub(crate) fn feed(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    over(store, holder(registry, connector).with_built_kept())
}

/// The feed of a gateway that follows a controller at an address.
///
/// Such a gateway probes no worker of its own. The controller's verdict is
/// each row's status, and the holder copies it onto the row's object. The
/// follower writes the store and settles this feed after each event.
/// Nothing writes at its door.
pub(crate) fn following(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    over(store, holder(registry, connector).with_status_followed())
}

/// A holder that writes `registry` and lays each object with `connector`.
fn holder(
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Objects<ModelWorkerSnapshot> {
    Objects::new(Arc::clone(registry)).with_connector(connector)
}

fn over(store: &Arc<MemoryStore>, objects: Objects<ModelWorkerSnapshot>) -> Feed {
    let objects = Arc::new(objects);
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
            objects: Arc::clone(&objects),
            store: Arc::clone(store),
        }),
        objects,
        store: Arc::clone(store),
        turn,
    }
}

/// The store door releases its turn before calling the listener. Hold the
/// feed's turn through that call, so neither a later write nor a settle
/// can publish a row the holder has yet to read.
///
/// A registration's built workers are on offer at the holder through its
/// write, inside the same turn: the listener's read lays each row with its
/// own, and a row the write left unmoved has the holder lay it too. The
/// offer ends whole with the turn.
/// A guarded write names the object it replaces, and the holder refuses it
/// in that turn where another object stands.
struct FeedDoor {
    door: StoreDoor,
    turn: Arc<Mutex<()>>,
    objects: Arc<Objects<ModelWorkerSnapshot>>,
    store: Arc<MemoryStore>,
}

impl FeedDoor {
    fn write<T>(&self, write: impl FnOnce(&StoreDoor) -> Result<T, Refused>) -> Result<T, Refused> {
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

    fn found_rows(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> Result<Vec<Wrote>, Refused> {
        self.write(|door| {
            let seen = over.map(Over::object);
            self.objects.handing(self.store.as_ref(), built, seen, || {
                door.found_rows(maker, expect, built, over)
            })
        })
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.drained(by, url, seen))
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.gone(by, url, seen))
    }

    fn refreshed(
        &self,
        seen: Standing,
        built: &Arc<dyn Worker>,
        over: &Arc<dyn Worker>,
    ) -> Result<Wrote, Refused> {
        self.write(|door| {
            self.objects
                .handing_one(self.store.as_ref(), built, Some(over), || {
                    door.refreshed(seen, built, over)
                })
        })
    }
}

#[cfg(test)]
pub(crate) mod signal_tests;
#[cfg(test)]
mod tests;
