//! Step to take a provider's workers out of the registry, in the removal
//! line that writes at the store's door.
//!
//! The door's steps leave a worker with no row alone, and a provider's
//! worker has none: the registry is its one home. This step removes it as
//! the classic line does. A removal that names a row's worker is the
//! door's: the step skips it, whatever the door answered.

use std::sync::Arc;

use async_trait::async_trait;
use smg_policy_controller::worker_host::Objects;
use smg_worker::workflow::{require_host, RemoveFromWorkerRegistryStep, WorkflowHost};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::common::snapshot::ModelWorkerSnapshot, workflow::data::WorkerRemovalWorkflowData,
};

/// The step's id in the removal line that writes at the store's door.
pub(crate) const REMOVE_PROVIDER_WORKERS_STEP: &str = "remove_provider_workers";

/// Step to remove the workers no row stands for.
pub(crate) struct RemoveProviderWorkersStep {
    rows: Arc<Objects<ModelWorkerSnapshot>>,
}

impl RemoveProviderWorkersStep {
    /// The step beside the rows whose objects `rows` holds.
    pub(crate) fn new(rows: Arc<Objects<ModelWorkerSnapshot>>) -> Self {
        Self { rows }
    }
}

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveProviderWorkersStep {
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
        RemoveFromWorkerRegistryStep.execute(context).await
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
