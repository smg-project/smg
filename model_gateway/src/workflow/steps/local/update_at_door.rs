//! The update step of a line whose local workers are rows of the policy
//! controller's store.
//!
//! The holder is the one writer of a row's object in the registry, so the
//! step writes a row's worker at the store's door: the row takes the
//! worker's labels and health checks. The door hands the rebuilt worker to
//! the holder, which lays it over the one that stands, under its id. A
//! provider's worker stays outside the store, and the step writes it in
//! the registry as the classic line does.
//!
//! A row holds no credential, no priority and no cost. The rebuilt worker
//! holds them, so an update that states one moves the object the holder
//! keeps, though it moves no fact of the row. A label refresh leaves a
//! row's worker as it is: the door takes the engine's labels from no
//! writer yet.

use std::sync::Arc;

use async_trait::async_trait;
use smg_policy_controller::worker_host::{
    maker_of, Expect, Membership, Objects, Refused, Standing,
};
use smg_worker::{identity::basic::Worker, jobs::WorkerJobAsker, registry::WorkerId};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::update_worker_properties::UpdateWorkerPropertiesStep;
use crate::{
    routers::common::snapshot::ModelWorkerSnapshot, workflow::data::WorkerUpdateWorkflowData,
};

/// The update step beside the rows whose objects `rows` holds, writing at
/// `door`.
pub(crate) struct UpdateAtDoorStep {
    door: Arc<dyn Membership>,
    rows: Arc<Objects<ModelWorkerSnapshot>>,
}

fn step_failed(message: String) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new("update_worker_properties"),
        message,
    }
}

impl UpdateAtDoorStep {
    pub(crate) fn new(door: Arc<dyn Membership>, rows: Arc<Objects<ModelWorkerSnapshot>>) -> Self {
        Self { door, rows }
    }

    /// The row behind each worker the find step selected, under the id it
    /// captured: none for a worker the holder laid no object for, which the
    /// registry holds alone.
    pub(super) fn rows_behind(
        &self,
        snapshot: Option<&[(WorkerId, u64)]>,
        workers: &[Arc<dyn Worker>],
    ) -> Vec<Option<Standing>> {
        workers
            .iter()
            .enumerate()
            .map(|(index, worker)| {
                let (id, _) = snapshot?.get(index)?;
                self.rows.standing_of(id, worker)
            })
            .collect()
    }

    /// Lay the labels and the health checks of `updated` over the row
    /// `standing` names, and hand `updated` in as the row's object in the
    /// place of `found`, the worker it was rebuilt from. An update that
    /// moves no fact of the row still lays its worker. The job writes as
    /// its asker's maker, and the door is the judge: another maker's row
    /// answers by the store's sentence, and a worker replaced since the
    /// update found it drops the update.
    pub(super) fn lay(
        &self,
        asker: &WorkerJobAsker,
        standing: Standing,
        updated: &Arc<dyn Worker>,
        found: &Arc<dyn Worker>,
    ) -> WorkflowResult<()> {
        let maker = maker_of(asker, updated.url()).map_err(|why| step_failed(why.to_string()))?;
        self.door
            .found_built(&maker, Expect::Seen(standing), updated, Some(found))
            .map(drop)
            .map_err(|refusal| step_failed(dropped(updated.as_ref(), refusal)))
    }
}

/// Why the door dropped the update of `worker`.
fn dropped(worker: &dyn Worker, refusal: Refused) -> String {
    let url = worker.redacted_url();
    match refusal {
        // The store's sentence, word for word.
        Refused::NotMaker(why) => why,
        Refused::Past(past) => format!("{}: {}", past.fact, past.why),
        Refused::Moved | Refused::Absent => format!(
            "worker {url} was replaced or removed since this update found it, so the \
             update was dropped"
        ),
        Refused::Draining => format!("worker {url} is draining, so the update was dropped"),
        other => format!("the membership door refused the update of worker {url}: {other:?}"),
    }
}

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateAtDoorStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        UpdateWorkerPropertiesStep::update(context, Some(self))
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
