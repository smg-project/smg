//! The jobs the worker stack asks of its host. The health manager, a discovery
//! source and the registration steps ask through [`WorkerJobs`]. The queue
//! takes those requests for any host whose jobs can carry a [`WorkerJob`].

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use self::queue::{JobHost, JobQueue, QueueJob};
use crate::{
    identity::endpoint::Endpoint,
    registry::WorkerId,
    worker::{WorkerSpec, WorkerUpdateRequest},
};

pub mod queue;

/// Who asked for a worker job. The caller states it when it asks. The host
/// carries it on the job, and a host's step reads it from the workflow data.
///
/// An asker is always stated, so the type has no default:
///
/// ```compile_fail
/// use smg_worker::jobs::WorkerJobAsker;
/// let _ = WorkerJobAsker::default();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkerJobAsker {
    /// A call of the worker API.
    Api,
    /// An address the host was started with.
    Startup,
    /// A discovery source. It names no source today. The day a host runs
    /// two it carries the source's name: the provider word the reconciler
    /// writes on the worker's label.
    Discovery,
    /// The health manager.
    Health,
}

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
/// no caller names a job. Each caller states who it is when it asks.
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
        asker: WorkerJobAsker,
    ) -> Result<(), String>;

    /// Queue the removal of the one worker `worker_id` at `url`, as it stood
    /// at `revision`: [`Self::remove_workers_at`] with that one guard.
    async fn remove_worker(
        &self,
        url: String,
        worker_id: &WorkerId,
        revision: u64,
        asker: WorkerJobAsker,
    ) -> Result<(), String> {
        self.remove_workers_at(url, vec![(worker_id.clone(), revision)], asker)
            .await
    }

    /// Queue a re-read of the labels of the worker at `url`. `Ok(false)` when
    /// another job for the worker is in flight: the refresh is kept and runs
    /// when that job ends.
    async fn refresh_worker_labels(
        &self,
        url: String,
        asker: WorkerJobAsker,
    ) -> Result<bool, String>;

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
        asker: WorkerJobAsker,
    ) -> Result<bool, String>;

    /// Whether the host's record of the address `url`, as written, shows a
    /// job in flight. The queue keeps one record per address. It writes the
    /// record for a registration or a label re-read at the claim and a plain
    /// submission's once a slot is held. Success clears the job's record.
    /// Failure leaves a terminal record. [`Self::add_worker`] claims the
    /// address when it admits a registration. A discovery source registers
    /// an address [as published] and asks its removal at the address's
    /// [lookup form].
    ///
    /// [as published]: crate::identity::endpoint::Endpoint::render
    /// [lookup form]: crate::identity::endpoint::Endpoint::lookup_form
    fn job_in_flight(&self, url: &str) -> bool;
}

/// A job the worker stack asks of a host's queue.
#[derive(Debug, Clone)]
pub enum WorkerJob {
    Add {
        spec: Box<WorkerSpec>,
        mode: WorkerRegistrationMode,
        asker: WorkerJobAsker,
    },
    Update {
        url: String,
        update: Box<WorkerUpdateRequest>,
        asker: WorkerJobAsker,
    },
    RefreshLabels {
        url: String,
        asker: WorkerJobAsker,
    },
    Remove {
        url: String,
        /// Revisions by worker id. A pin with no revision accepts any
        /// revision of that id. A removal with no pins takes the whole address.
        expected_revisions: Option<HashMap<String, Option<u64>>>,
        asker: WorkerJobAsker,
    },
}

impl QueueJob for WorkerJob {
    const REGISTRATION_TYPE: &'static str = "AddWorker";

    fn job_type(&self) -> &'static str {
        match self {
            Self::Add { .. } => Self::REGISTRATION_TYPE,
            Self::Update { .. } => "UpdateWorker",
            Self::RefreshLabels { .. } => "RefreshWorkerLabels",
            Self::Remove { .. } => "RemoveWorker",
        }
    }

    fn key(&self) -> &str {
        match self {
            Self::Add { spec, .. } => &spec.url,
            Self::Update { url, .. }
            | Self::RefreshLabels { url, .. }
            | Self::Remove { url, .. } => url,
        }
    }

    fn shown_key(_job_type: &str, key: &str) -> String {
        Endpoint::redacted_str(key)
    }

    fn is_removal(&self) -> bool {
        matches!(self, Self::Remove { .. })
    }

    fn merge_removals(mut held: Self, mut newer: Self) -> Result<Self, (Self, Self)> {
        match (&mut held, &mut newer) {
            (
                Self::Remove {
                    expected_revisions: pins,
                    asker,
                    ..
                },
                Self::Remove {
                    expected_revisions: newer_pins,
                    asker: newer_asker,
                    ..
                },
            ) if asker == newer_asker => {
                *pins = merge_pins(pins.take(), newer_pins.take());
                Ok(held)
            }
            _ => Err((held, newer)),
        }
    }

    fn asker(&self) -> Option<&WorkerJobAsker> {
        match self {
            Self::Add { asker, .. }
            | Self::Update { asker, .. }
            | Self::RefreshLabels { asker, .. }
            | Self::Remove { asker, .. } => Some(asker),
        }
    }
}

/// Merged removals take the union of their pins. A pin that accepts any
/// revision takes precedence. An unguarded removal takes the whole address.
fn merge_pins(
    held: Option<HashMap<String, Option<u64>>>,
    newer: Option<HashMap<String, Option<u64>>>,
) -> Option<HashMap<String, Option<u64>>> {
    let (Some(mut pins), Some(newer)) = (held, newer) else {
        return None;
    };
    for (id, revision) in newer {
        let pin = pins.entry(id).or_insert(revision);
        if revision.is_none() {
            *pin = None;
        }
    }
    Some(pins)
}

#[async_trait]
impl<H: JobHost> WorkerJobs for JobQueue<H>
where
    H::Job: From<WorkerJob>,
{
    async fn refresh_worker_labels(
        &self,
        url: String,
        asker: WorkerJobAsker,
    ) -> Result<bool, String> {
        self.submit_or_defer(WorkerJob::RefreshLabels { url, asker }.into())
            .await
    }

    async fn add_worker(
        &self,
        spec: WorkerSpec,
        mode: WorkerRegistrationMode,
        asker: WorkerJobAsker,
    ) -> Result<bool, String> {
        let job = WorkerJob::Add {
            spec: Box::new(spec),
            mode,
            asker,
        };
        self.submit_unless_registering(job.into()).await
    }

    async fn remove_workers_at(
        &self,
        url: String,
        guards: Vec<(WorkerId, u64)>,
        asker: WorkerJobAsker,
    ) -> Result<(), String> {
        // Each pin names one worker. Workers sharing a base address can
        // have the same revision, so the revision alone cannot name one.
        let expected_revisions = guards
            .into_iter()
            .map(|(id, revision)| (id.as_str().to_string(), Some(revision)))
            .collect();
        self.submit(
            WorkerJob::Remove {
                url,
                expected_revisions: Some(expected_revisions),
                asker,
            }
            .into(),
        )
        .await
    }

    fn job_in_flight(&self, url: &str) -> bool {
        self.get_status(url)
            .is_some_and(|status| status.status == "pending" || status.status == "processing")
    }
}
