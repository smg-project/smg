//! The update step of a line whose local workers are rows of the policy
//! controller's store.
//!
//! The holder is the one writer of a row's object in the registry, so the
//! step writes a row's worker at the store's door: the row takes the
//! worker's labels and health checks, and the holder lays its object over
//! the one that stands, under its id. A provider's worker stays outside
//! the store, and the step writes it in the registry as the classic line
//! does.
//!
//! A row holds no credential, no priority and no cost yet. The worker API
//! refuses an update that states one before it queues a job. A job that
//! reaches a row all the same is refused here by the same names, before
//! any worker is written. A label refresh leaves a row's worker as it is:
//! the door takes the engine's labels from no writer yet.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerUpdateRequest;
use smg_policy_controller::worker_host::{
    facts, maker_of, Expect, Membership, Objects, Refused, Standing,
};
use smg_worker::{identity::basic::Worker, jobs::WorkerJobAsker, registry::WorkerId};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    refuse_worker_key::WORKER_KEY_UNSUPPORTED, update_worker_properties::UpdateWorkerPropertiesStep,
};
use crate::{
    routers::common::snapshot::ModelWorkerSnapshot, workflow::data::WorkerUpdateWorkflowData,
};

/// The name of the refusal of a priority or a cost, in the step's error and
/// in the worker API's answer.
pub(crate) const WORKER_FIELD_UNSUPPORTED: &str = "POLICY_CONTROLLER_WORKER_FIELD_UNSUPPORTED";

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

    /// The refusal of an update of `worker`, a row's, that states what no
    /// row holds yet, by the name the worker API answers with: the
    /// credential first, then the priority, then the cost.
    pub(super) fn refuse_unheld(
        request: &WorkerUpdateRequest,
        worker: &dyn Worker,
    ) -> WorkflowResult<()> {
        let (name, stated) = if request.api_key.is_some() {
            (WORKER_KEY_UNSUPPORTED, "an API key")
        } else if request.priority.is_some() {
            (WORKER_FIELD_UNSUPPORTED, "a priority")
        } else if request.cost.is_some() {
            (WORKER_FIELD_UNSUPPORTED, "a cost")
        } else {
            return Ok(());
        };
        Err(step_failed(format!(
            "{name}: the update of worker {} states {stated}, and the worker API sets none \
             on a worker of the policy controller's store yet",
            worker.redacted_url()
        )))
    }

    /// Lay the labels and the health checks of `updated` over the row
    /// `standing` names. The job writes as its asker's maker, and the door
    /// is the judge: another maker's row answers by the store's sentence.
    pub(super) fn lay(
        &self,
        asker: &WorkerJobAsker,
        standing: Standing,
        updated: &dyn Worker,
    ) -> WorkflowResult<()> {
        let url = updated.url();
        let maker = maker_of(asker, url).map_err(|why| step_failed(why.to_string()))?;
        self.door
            .found(&maker, Expect::Seen(standing), url, &|row| {
                facts(updated, row)
            })
            .map(drop)
            .map_err(|refusal| step_failed(dropped(updated, refusal)))
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
