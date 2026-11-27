//! Step executor trait and implementations

use async_trait::async_trait;

use crate::types::{StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult};

/// Trait for executing individual workflow steps
#[async_trait]
pub trait StepExecutor<D: WorkflowData>: Send + Sync {
    /// Execute the step with the given context
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult>;

    /// Check if an error is retry-able
    ///
    /// Override this method to customize which errors should trigger retries.
    /// By default, all errors are considered retry-able.
    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }

    /// Called when the step succeeds
    ///
    /// This hook allows steps to perform cleanup or additional actions
    /// after successful execution.
    async fn on_success(&self, _context: &WorkflowContext<D>) -> WorkflowResult<()> {
        Ok(())
    }

    /// Called when the step fails after all retries
    ///
    /// This hook allows steps to perform cleanup or compensation logic
    /// when the step cannot complete successfully.
    async fn on_failure(
        &self,
        _context: &WorkflowContext<D>,
        _error: &WorkflowError,
    ) -> WorkflowResult<()> {
        Ok(())
    }
}

/// Simple function-based step executor
pub struct FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    func: F,
    _phantom: std::marker::PhantomData<D>,
}

impl<D, F> FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<D, F> StepExecutor<D> for FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        (self.func)(context).await
    }
}

#[cfg(test)]
#[path = "../tests/unit/executor/tests.rs"]
mod tests;
