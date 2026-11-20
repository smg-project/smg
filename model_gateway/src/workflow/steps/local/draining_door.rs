//! The door a removal writes at, where the holder leaves a worker's status
//! to the gateway.
//!
//! The feed's holder copies no row's status onto its object: the gateway's
//! own health checks write an object's status. A drain is not a health
//! check's word. The store turns the row `Draining`, and the object would
//! read `Ready` in the registry through the whole window, where every
//! reader of the registry still sends to it. The registry's own drain marks
//! a `Ready` worker `Draining`. This door makes that mark where the store
//! answered a drain.

use std::sync::Arc;

use openai_protocol::worker::WorkerStatus;
use smg_policy_controller::{
    store::{Maker, WorkerFacts},
    view::WorkerId,
    worker_host::{Expect, Membership, Objects, Refused, Remover, Standing, Wrote},
};

use crate::routers::common::snapshot::ModelWorkerSnapshot;

/// A removal's door: the feed's, and a drain the store took is marked on
/// the row's worker in the registry.
pub(crate) struct DrainingDoor {
    door: Arc<dyn Membership>,
    rows: Arc<Objects<ModelWorkerSnapshot>>,
}

impl DrainingDoor {
    /// `door`, over the objects `rows` holds.
    pub(crate) fn new(door: Arc<dyn Membership>, rows: Arc<Objects<ModelWorkerSnapshot>>) -> Self {
        Self { door, rows }
    }

    /// Mark the worker that stands for `row` `Draining` where it reads
    /// `Ready`. The check and the write share the registry's lock, as in
    /// the registry's own drain: a probe's demotion stands, and a worker
    /// rebuilt since the read is another life and keeps its status.
    fn mark(&self, row: WorkerId) {
        let registry = self.rows.registry();
        let Some(id) = self.rows.object(row) else {
            return;
        };
        let Some(worker) = registry.get(&id) else {
            return;
        };
        registry.apply_if_revision(&id, worker.revision(), |worker| {
            let ready = worker.status() == WorkerStatus::Ready;
            ((), ready.then_some(WorkerStatus::Draining))
        });
    }
}

impl Membership for DrainingDoor {
    fn found(
        &self,
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.door.found(maker, expect, url, state)
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        let wrote = self.door.drained(by, url, seen)?;
        // `Unmoved` is a row that was draining already: the removal that
        // marked it may have ended before its mark reached the worker.
        if let Wrote::Draining(row) | Wrote::Unmoved(row) = wrote {
            self.mark(row);
        }
        Ok(wrote)
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.door.gone(by, url, seen)
    }
}
