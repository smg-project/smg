//! Step to tell the cache-event monitor which workers left.

use async_trait::async_trait;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    workflow::data::{require_app_context, WorkerRemovalWorkflowData},
};

/// The step's id in the removal line that writes at the store's door.
pub(crate) const FORGET_CACHE_EVENTS_STEP: &str = "forget_cache_events";

/// Tell the cache-event monitor the worker at `url` left: its subscription
/// stops and its blocks leave the index.
pub(super) async fn forget_cache_events(app_context: &AppContext, url: &str) {
    if let Some(monitor) = &app_context.kv_event_monitor {
        monitor.on_worker_removed(url).await;
    }
}

/// Step to forget the cache events of the workers a removal took.
///
/// It reads the workers the removal's write answered for, so a worker the
/// write left standing keeps its subscription.
pub(crate) struct ForgetCacheEventsStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for ForgetCacheEventsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.host.as_ref())?;
        for worker in &context.data.removed_workers {
            forget_cache_events(app_context, worker.url()).await;
        }
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
