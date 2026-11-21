//! The worker registry of a gateway that follows a controller, kept as the
//! policy store's rows say.
//!
//! The follower writes the store as the controller's watch tells it, and
//! settles the feed after each event it takes ([`following`]). A settle
//! brings the registry up to date, then publishes membership for
//! decisions, in that order.
//!
//! No registration runs in such a gateway, so the holder builds each
//! worker object from its row. Such an object holds what the row states
//! and no credential.

use std::sync::Arc;

use parking_lot::Mutex;
use smg_policy_controller::{store::MemoryStore, worker_host::Objects};
use smg_worker::connection::engine::EngineConnector;

use crate::routers::common::snapshot::{ModelWorkerSnapshot, WorkerRegistry};

/// The feed of a gateway that follows a controller: the holder that keeps
/// the registry as the store's rows say.
#[derive(Clone)]
pub(crate) struct Feed {
    /// The one writer of the objects of the store's rows.
    pub objects: Arc<Objects<ModelWorkerSnapshot>>,
    /// The store the follower writes and the holder follows.
    store: Arc<MemoryStore>,
    /// The feed's turn: one settle at a time.
    turn: Arc<Mutex<()>>,
}

impl Feed {
    /// Bring the registry to the store's rows, publish them, then write the
    /// published overload verdicts on the objects. The follower calls this
    /// after each event it applies.
    ///
    /// The order is the rule: the registry holds a row's object before a
    /// decision can read the row. The store pays the build the membership
    /// writes left owed, at once. The objects then take the overload
    /// verdicts of the world that stands.
    pub(crate) fn settle(&self) {
        let _turn = self.turn.lock();
        self.objects.follow(&*self.store);
        self.store.publish_now();
        self.objects.mirror_overload(&self.store);
    }

    /// The store the follower writes and the holder follows.
    pub(crate) fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }
}

/// The feed of a gateway that follows a controller at an address, over
/// `store`, for the workers of `registry`.
///
/// Such a gateway probes no worker of its own. The controller's verdict is
/// each row's status, and the holder copies it onto the row's object. It
/// builds that object from the row with `connector`, the one the gateway
/// was started with. The follower writes the store and settles this feed
/// after each event.
pub(crate) fn following(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    let objects = Objects::new(Arc::clone(registry))
        .with_connector(connector)
        .with_status_followed();
    Feed {
        objects: Arc::new(objects),
        store: Arc::clone(store),
        turn: Arc::new(Mutex::new(())),
    }
}

#[cfg(test)]
mod tests;
