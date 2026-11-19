//! What the steps of a worker workflow ask of the process that runs them.

use std::sync::Arc;

use wfaas::{WorkflowError, WorkflowResult};

use crate::registry::{Generation, WorkerRegistry};

/// The host of the worker workflows: the process that keeps the worker
/// registry. The steps of this crate read their host through this trait
/// alone.
pub trait WorkflowHost: Send + Sync + 'static {
    /// What the host's registry keeps of a membership generation.
    type Generation: Generation;

    /// The registry the workflows register workers in and remove them from.
    fn worker_registry(&self) -> &Arc<WorkerRegistry<Self::Generation>>;
}

/// The host a workflow's data was given, or the workflow's error for a run
/// that was started without one.
pub fn require_host<H>(host: Option<&Arc<H>>) -> WorkflowResult<&Arc<H>> {
    host.ok_or_else(|| WorkflowError::ContextValueNotFound("host".to_string()))
}

/// A host that keeps a registry and nothing else, for the steps' own cases.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Arc;

    use super::WorkflowHost;
    use crate::test_support::{Members, WorkerRegistry};

    #[derive(Debug)]
    pub(crate) struct RegistryHost {
        pub(crate) worker_registry: Arc<WorkerRegistry>,
    }

    impl WorkflowHost for RegistryHost {
        type Generation = Members;

        fn worker_registry(&self) -> &Arc<WorkerRegistry> {
            &self.worker_registry
        }
    }
}
