//! The registry's removal steps for a worker no row stands for, in the
//! removal line that writes at the store's door.
//!
//! The door's steps leave a worker with no row alone, and a provider's
//! worker has none: the registry is its one home. The registry's own drain
//! and removal take it, as in the classic line. A removal that names a
//! row's worker is the door's: both steps skip it, whatever the door
//! answered.

use std::sync::Arc;

use async_trait::async_trait;
use smg_policy_controller::worker_host::Objects;
use smg_worker::workflow::{require_host, WorkflowHost};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::common::snapshot::ModelWorkerSnapshot, workflow::data::WorkerRemovalWorkflowData,
};

/// The id of the step that drains the workers no row stands for.
pub(crate) const DRAIN_PROVIDER_WORKERS_STEP: &str = "drain_provider_workers";

/// The id of the step that removes the workers no row stands for.
pub(crate) const REMOVE_PROVIDER_WORKERS_STEP: &str = "remove_provider_workers";

/// A step of the registry's removal, run for the workers no row stands for.
pub(crate) struct ForProviderWorkers<S> {
    rows: Arc<Objects<ModelWorkerSnapshot>>,
    step: S,
}

impl<S> ForProviderWorkers<S> {
    /// `step`, beside the rows whose objects `rows` holds.
    pub(crate) fn new(rows: Arc<Objects<ModelWorkerSnapshot>>, step: S) -> Self {
        Self { rows, step }
    }
}

#[async_trait]
impl<S> StepExecutor<WorkerRemovalWorkflowData> for ForProviderWorkers<S>
where
    S: StepExecutor<WorkerRemovalWorkflowData>,
{
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let registry = require_host(context.data.host.as_ref())?.worker_registry();
        let names_row = context.data.worker_urls.iter().any(|url| {
            registry
                .get_id_by_url(url)
                .is_some_and(|id| self.rows.standing(&id).is_some())
        });
        // A row the door removed left the registry with it: the door's
        // answer names it.
        if names_row || !context.data.removed_workers.is_empty() {
            return Ok(StepResult::Skip);
        }
        self.step.execute(context).await
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        self.step.is_retryable(error)
    }
}
