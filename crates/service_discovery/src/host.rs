//! What the providers ask of the process that hosts them.

use std::sync::Arc;

use openai_protocol::worker::WorkerSpec;
use smg_worker::{
    identity::basic::Worker,
    jobs::WorkerJobs,
    registry::{WorkerId, WorkerOrigin},
};

/// The host of worker discovery: the process whose registry a provider's
/// workers are reconciled against, and whose job queue takes what a pass
/// finds to register and to remove. A host implements it once, for the
/// state its tasks share.
pub trait DiscoveryHost: Send + Sync + 'static {
    /// Every registered worker, with its id.
    fn workers(&self) -> Vec<(WorkerId, Arc<dyn Worker>)>;

    /// Where the registration of `worker_id` came from. `None` when no such
    /// worker is registered.
    fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin>;

    /// Write onto `spec` what the host sets on every worker discovery
    /// registers: the credential its workers are called with, and the budget
    /// of connection attempts.
    fn stamp(&self, spec: &mut WorkerSpec);

    /// The host's job queue. `None` until it stands: a pass then defers what
    /// it found.
    fn jobs(&self) -> Option<Arc<dyn WorkerJobs>>;
}
