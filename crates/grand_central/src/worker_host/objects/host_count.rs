//! A worker object's own count of requests in flight, kept by the leases
//! the store holds on file.
//!
//! The engine over worker lists reads each worker's count off its object.
//! In a gateway's own process a request's load guard moves that count. A
//! controller serves no request itself: what it knows of a request is the
//! lease it filed. So the holder hands the store each row's object as that
//! row's count ([`HostCount`]), and the store steps the object's count
//! where it books and frees the worker for a lease.
//!
//! A host that serves requests itself counts each by its load guard and
//! files no lease for it: a lease filed for a request a guard counts is
//! counted twice on the object.
//!
//! The object's count of one session key's requests moves the same way.
//! A guard counts a request under its key's text. A lease carries its
//! session as a number, so the store's step counts it under that number
//! written as text. One object stands in one process, which counts by
//! guards or by leases: the two texts never meet on it.

use std::sync::Arc;

use smg_worker::{identity::basic::Worker, registry::Generation};

use super::Objects;
use crate::{
    engine::view::{Session, WorkerId},
    store::WorkersStore,
    tracking::HostCount,
};

/// The count of the object laid for a row: the object itself.
///
/// The store keeps it while the row stands. An object laid in another's
/// place takes the standing object's running state, its count with it:
/// the store hands the new object the leases that stand and takes them
/// off the old one, which leaves the number the two share as it stood.
/// An object with a running state of its own reads them from there on.
struct ObjectCount(Arc<dyn Worker>);

impl HostCount for ObjectCount {
    fn booked(&self) {
        self.0.increment_load();
    }

    fn freed(&self) {
        self.0.decrement_load();
    }

    /// The object counts a key's requests under a text. A lease names its
    /// session by a number, so the count stands under that number as
    /// [`Session::text`] writes it, the text the engine's reader asks
    /// under for a request whose held decisions count it.
    fn session_booked(&self, session: Session) {
        let mut text = [0; Session::TEXT_LEN];
        self.0.increment_routing_key_load(session.text(&mut text));
    }

    fn session_freed(&self, session: Session) {
        let mut text = [0; Session::TEXT_LEN];
        self.0.decrement_routing_key_load(session.text(&mut text));
    }
}

impl<G: Generation> Objects<G> {
    /// Hands `store` the object held for `row` as the row's count, where
    /// one is held. The holder calls it behind each lay of an object and
    /// for each row a whole read finds standing: a store that took
    /// another's rows gives each a new place in its counts.
    pub(super) fn counted<S: WorkersStore + ?Sized>(&self, store: &S, row: WorkerId) {
        let object = {
            let held = self.held.read();
            let laid = held.objects.get(&row);
            laid.map(|(_, laid)| Arc::clone(&laid.object))
        };
        if let Some(object) = object {
            store.host_count(row, Box::new(ObjectCount(object)));
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/host_count/tests.rs"]
mod tests;
