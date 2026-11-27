use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError,
    WorkflowInstanceId, WorkflowResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestData {
    value: i32,
}

impl WorkflowData for TestData {
    fn workflow_type() -> &'static str {
        "test"
    }
}

struct TestStep {
    should_succeed: bool,
}

#[async_trait]
impl StepExecutor<TestData> for TestStep {
    async fn execute(
        &self,
        _context: &mut WorkflowContext<TestData>,
    ) -> WorkflowResult<StepResult> {
        if self.should_succeed {
            Ok(StepResult::Success)
        } else {
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("test"),
                message: "test error".to_string(),
            })
        }
    }
}

#[tokio::test]
async fn test_step_executor_success() {
    let step = TestStep {
        should_succeed: true,
    };
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), TestData { value: 42 });

    let result = step.execute(&mut context).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), StepResult::Success);
}

#[tokio::test]
async fn test_step_executor_failure() {
    let step = TestStep {
        should_succeed: false,
    };
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), TestData { value: 42 });

    let result = step.execute(&mut context).await;
    assert!(result.is_err());
}
