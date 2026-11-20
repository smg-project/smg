//! The registry's removal steps for the workers no row stands for, in the
//! removal line that writes at the store's door.
//!
//! The door's steps leave a worker with no row alone, and a provider's
//! worker has none: the registry is its one home. The registry's own drain
//! and removal take it, as in the classic line. One removal can name both
//! kinds: an address selects every rank and every scheme registered under
//! it. Each step here runs for the workers no row stands for and leaves
//! the rest of the selection to the door. A row the door refused reaches
//! neither step: the door's drain fails the job where nothing drained, and
//! takes the row's worker off the job's lists otherwise.

use std::{mem, sync::Arc};

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
        let host = Arc::clone(require_host(context.data.host.as_ref())?);
        let registry = host.worker_registry();
        // The selection's workers that stand in the registry with no row.
        // A row the door drained still stands with its row, and a row the
        // door removed left the registry with it: neither is this step's.
        let selected = context
            .data
            .actual_workers_to_remove
            .clone()
            .unwrap_or_default();
        let (rowless, rest): (Vec<_>, Vec<_>) = selected.into_iter().partition(|worker| {
            registry
                .get_id_by_url(worker.url())
                .is_some_and(|id| registry.get(&id).is_some() && self.rows.standing(&id).is_none())
        });
        if rowless.is_empty() {
            return Ok(StepResult::Skip);
        }

        // The step reads the job's lists and writes its answer over the
        // door's. Hand it the rowless workers alone, then lay the door's
        // lists back in front of what it left. The engine writes a failed
        // step's data back as well, so the lists are whole on every way
        // out.
        let data = &mut context.data;
        let drained = mem::take(&mut data.drained_workers);
        let removed = mem::take(&mut data.removed_workers);
        data.keep_only(rowless);
        let ended = self.step.execute(context).await;
        let data = &mut context.data;
        let left = data.actual_workers_to_remove.take().unwrap_or_default();
        data.keep_only(rest.into_iter().chain(left).collect());
        data.drained_workers.splice(0..0, drained);
        data.removed_workers.splice(0..0, removed);
        ended
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        self.step.is_retryable(error)
    }
}
