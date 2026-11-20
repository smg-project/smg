//! Step to refuse a local worker that states an API key, where the gateway's
//! local workers are rows of the policy controller's store.
//!
//! A row holds no credential, so the object built from it carries none and
//! the worker would be asked without its key. The registration is refused
//! by name instead. A provider's worker stays outside the store and keeps
//! its key.

use async_trait::async_trait;
use smg_worker::{identity::endpoint::Endpoint, workflow::WorkerKind};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::data::WorkerWorkflowData;

/// The step's id in the registration line that writes at the store's door.
pub(crate) const REFUSE_WORKER_KEY_STEP: &str = "refuse_worker_key";

/// The name of the refusal, in the step's error and in the worker API's
/// answer.
pub(crate) const WORKER_KEY_UNSUPPORTED: &str = "POLICY_CONTROLLER_WORKER_KEY_UNSUPPORTED";

/// Step to refuse a local registration that states an API key.
pub struct RefuseWorkerKeyStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for RefuseWorkerKeyStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }
        let config = &context.data.config;
        if config.api_key.is_none() {
            return Ok(StepResult::Skip);
        }
        Err(WorkflowError::StepFailed {
            step_id: StepId::new(REFUSE_WORKER_KEY_STEP),
            message: format!(
                "{WORKER_KEY_UNSUPPORTED}: worker {} states an API key, and a worker of the \
                 policy controller's store holds none yet",
                Endpoint::redacted_str(&config.url)
            ),
        })
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::WorkerSpec;
    use smg_worker::jobs::WorkerJobAsker;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        routers::common::snapshot::WorkerRegistry,
        workflow::{data::WorkerRegistrationMode, steps::create_worker_workflow_data},
    };

    const URL: &str = "http://worker:8000";

    async fn refusal(kind: Option<WorkerKind>, key: Option<&str>) -> Result<StepResult, String> {
        let host = app_context_for_tests(
            RouterConfig::default(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        let mut spec = WorkerSpec::new(URL);
        spec.api_key = key.map(str::to_owned);
        let mut data = create_worker_workflow_data(
            spec,
            WorkerRegistrationMode::Upsert,
            host,
            WorkerJobAsker::Startup,
        );
        data.worker_kind = kind;
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        RefuseWorkerKeyStep
            .execute(&mut context)
            .await
            .map_err(|error| error.to_string())
    }

    #[tokio::test]
    async fn a_local_registration_with_a_key_fails_by_name_and_keeps_the_key_out_of_its_error() {
        let error = refusal(Some(WorkerKind::Local), Some("secret-key"))
            .await
            .expect_err("a keyed local registration is refused");
        assert!(error.contains(WORKER_KEY_UNSUPPORTED), "{error}");
        assert!(error.contains(URL), "{error}");
        assert!(!error.contains("secret-key"), "{error}");
    }

    #[tokio::test]
    async fn a_registration_without_a_key_and_a_providers_with_one_pass() {
        assert_eq!(
            refusal(Some(WorkerKind::Local), None).await,
            Ok(StepResult::Skip)
        );
        assert_eq!(
            refusal(Some(WorkerKind::External), Some("secret-key")).await,
            Ok(StepResult::Skip)
        );
    }
}
