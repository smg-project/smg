//! The worker health sink of a replica that keeps its workers in the store:
//! a worker object's status leads, and the store is told it.
//!
//! A probe's outcome goes to the registry's own state machine, which counts
//! on the object and moves its status under the object's mutation lock. The
//! adapter then tells the store the status that stands
//! ([`Objects::tell_status`]). A connect signal is the holder's
//! ([`Objects::connect`]): ready at once, unless the object drains or is
//! ready. No status is decided over the store's row.
//!
//! The adapter reads the lead before it hands a verdict on. The tell reads
//! it again under the object's mutation lock, and the store asks it once
//! more behind its own lock, as an admitted apply asks its own: a term that
//! ended while a write waited there writes nothing.
//!
//! A move the store takes no word of is taken back by the tell itself: in
//! the turn of the object's lock where the store refused it, the object
//! takes the status its row holds, and the outcome answers as a stale
//! one. The row's status, not the one the probe saw the object leave: a
//! new leader's told status can have landed since the move. Where no row
//! stands for the object, it takes back the status it left. So a verdict
//! moves the object and its row, or neither. The registry's
//! state machine moves the object in its own turn of the lock, so between
//! that turn and the tell a reader of the object can see the status that
//! is taken back, and the registry's subscribers hear both moves.
//!
//! The store guards a write by the row's id and revision, and a row's id is
//! its key and incarnation. A URL's next life under another incarnation is
//! another row: a status of the earlier life's object finds no row under
//! the id it names and writes nothing. So the guard tells lives apart where
//! the host numbers each life apart, as a registration door does with
//! [`life`](crate::store::membership::life).

use std::sync::Arc;

use smg_worker::{
    health::port::{HealthSink, ProbeAnswer},
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{Lead, Objects};
use crate::store::MemoryStore;

/// The health sink for a replica that keeps its workers in the store.
pub struct HealthAdapter<G> {
    objects: Arc<Objects<G>>,
    store: Arc<MemoryStore>,
    lead: Arc<dyn Lead>,
    term: u64,
}

impl<G: Generation> HealthAdapter<G> {
    /// An adapter for `term`. Each call checks that `lead` still names it
    /// before it hands a verdict on. A status is told under the object's
    /// lock where `lead` names the term still, and the store asks it once
    /// more behind its own.
    pub fn new(
        objects: Arc<Objects<G>>,
        store: Arc<MemoryStore>,
        lead: Arc<dyn Lead>,
        term: u64,
    ) -> Self {
        Self {
            objects,
            store,
            lead,
            term,
        }
    }

    fn leads(&self) -> bool {
        self.lead.term() == Some(self.term)
    }
}

impl<G: Generation> HealthSink for HealthAdapter<G> {
    /// The registry's state machine judges the probe over the object, and
    /// a move is told to the store.
    fn probe_outcome(
        &self,
        worker_id: &ObjectId,
        passed: bool,
        revision: u64,
        launched: ObjectStatus,
    ) -> ProbeAnswer {
        if !self.leads() {
            return ProbeAnswer::Stale;
        }
        let registry = self.objects.registry().as_ref();
        let answer = HealthSink::probe_outcome(registry, worker_id, passed, revision, launched);
        let ProbeAnswer::Moved { old, new } = answer else {
            return answer;
        };
        let leads = || self.leads();
        let told = self
            .objects
            .tell_status(&self.store, worker_id, revision, (old, new), &leads);
        if told.stands().is_some() {
            return answer;
        }
        // The store took no word of the move: its term ended, the row
        // left the revision the object was built from, or the holder
        // laid no such object. The tell took the move back, to the
        // status the object's row holds or, where none stands, to the
        // one it left. The outcome is stale.
        ProbeAnswer::Stale
    }

    fn connected(
        &self,
        worker_id: &ObjectId,
        revision: u64,
    ) -> Option<(ObjectStatus, ObjectStatus)> {
        if !self.leads() {
            return None;
        }
        let leads = || self.leads();
        self.objects
            .connect(&self.store, worker_id, revision, &leads)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/health/tests.rs"]
mod tests;
