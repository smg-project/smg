//! Step to bring the pool-size gauge down after a removal at the store's
//! door.
//!
//! The registry's removal step refreshes the gauge itself. A row's worker
//! leaves through the holder, which writes no gauge, so this step
//! refreshes the groups of the workers the removal took.

use async_trait::async_trait;
use smg_worker::workflow::{
    pool_group, refresh_worker_pool_sizes, require_host, PoolGroups, WorkflowHost,
};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::data::WorkerRemovalWorkflowData;

/// The step's id in the removal line that writes at the store's door.
pub(crate) const REFRESH_POOL_SIZES_STEP: &str = "refresh_pool_sizes";

/// Step to refresh the pool sizes of the groups the removed workers left.
pub(crate) struct RefreshPoolSizesStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RefreshPoolSizesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let registry = require_host(context.data.host.as_ref())?.worker_registry();
        let groups: PoolGroups = context
            .data
            .removed_workers
            .iter()
            .map(|worker| pool_group(worker.as_ref()))
            .collect();
        refresh_worker_pool_sizes(registry, &groups);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
