//! The jobs the worker stack asks of its host. One port, [`WorkerJobs`], that
//! the host's job queue implements: the health manager, a discovery source
//! and the registration steps each ask through it, and none names a job of
//! the host.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{registry::WorkerId, worker::WorkerSpec};

/// Semantics of the final worker registry write.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRegistrationMode {
    /// `POST /workers`: fail when the URL is already registered.
    CreateOnly,
    /// Startup and service discovery: register the URL, replacing whatever
    /// holds it. Both reconcile declared state against the registry.
    #[default]
    Upsert,
    /// `PUT /workers/{id}`: replace the worker only while `worker_id` still
    /// holds the URL at `expected_revision`.
    ///
    /// `PUT` answers 202 and registers later, so other writes for the same URL
    /// can land in between. Binding the write to what the request validated
    /// makes the late write fail instead of resurrecting a worker a `DELETE`
    /// removed, overwriting one a `POST` created, or restoring the older
    /// specification over a second `PUT` that finished first.
    ReplaceById {
        worker_id: String,
        expected_revision: u64,
    },
}

/// What the worker stack asks of its host's job queue for a worker. The
/// health manager asks for the removal of one that failed and for a re-read
/// of the labels of one that came into service. A discovery source asks for
/// a registration, for the removal of the workers at an address, and whether
/// a job of an address is in flight. The host's job queue implements it;
/// no caller names a job.
#[async_trait]
pub trait WorkerJobs: Send + Sync {
    /// Queue the removal of workers at `url`: each one a guard names by its
    /// id, and only at the revision the guard holds. `url` may be a base
    /// address other workers share: the removal takes the named workers
    /// alone.
    async fn remove_workers_at(
        &self,
        url: String,
        guards: Vec<(WorkerId, u64)>,
    ) -> Result<(), String>;

    /// Queue the removal of the one worker `worker_id` at `url`, as it stood
    /// at `revision`: [`Self::remove_workers_at`] with that one guard.
    async fn remove_worker(
        &self,
        url: String,
        worker_id: &WorkerId,
        revision: u64,
    ) -> Result<(), String> {
        self.remove_workers_at(url, vec![(worker_id.clone(), revision)])
            .await
    }

    /// Queue a re-read of the labels of the worker at `url`. `Ok(false)` when
    /// another job for the worker is in flight: the refresh is kept and runs
    /// when that job ends.
    async fn refresh_worker_labels(&self, url: String) -> Result<bool, String>;

    /// Queue the registration of the worker `spec` describes, under `mode`.
    /// `Ok(false)` when a registration of that address is in flight: nothing
    /// is queued, and the caller asks again later. The check and the claim of
    /// the address are one step: of two callers at once one is queued and the
    /// other refused, and a caller that waits for a queue slot holds its
    /// claim while it waits.
    async fn add_worker(
        &self,
        spec: WorkerSpec,
        mode: WorkerRegistrationMode,
    ) -> Result<bool, String>;

    /// Whether the host's record of the address `url`, as written, shows a
    /// job in flight. A host may write a job's record ahead of its queue
    /// slot (a registration's and a label re-read's, at the claim) or once
    /// the slot is held (a removal's), and may keep one record an address,
    /// which a job that ends clears. So the answer is one to plan by, not a
    /// claim: `false` reserves nothing, and [`Self::add_worker`] is the
    /// claim. A job is recorded under the address it was asked with, so a
    /// caller reads in that form: a registration's as its spec writes the
    /// address, a removal's as the removal was asked. A discovery source
    /// registers an address [as published] and asks its removal at the
    /// address's [lookup form].
    ///
    /// [as published]: crate::identity::endpoint::Endpoint::render
    /// [lookup form]: crate::identity::endpoint::Endpoint::lookup_form
    fn job_in_flight(&self, url: &str) -> bool;
}
