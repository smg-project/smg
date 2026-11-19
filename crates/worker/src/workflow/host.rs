//! What the steps of a worker workflow ask of the process that runs them.

use std::{fmt, sync::Arc};

use wfaas::{WorkflowError, WorkflowResult};

use crate::{
    connection::http_client::WorkerHttpClientCache,
    registry::{Generation, WorkerRegistry},
};

/// The host of the worker workflows: the process that keeps the worker
/// registry. The steps of this crate read their host through this trait
/// alone.
pub trait WorkflowHost: Send + Sync + 'static {
    /// What the host's registry keeps of a membership generation.
    type Generation: Generation;

    /// What the host's own steps keep on a registration between them. No step
    /// of this crate reads or writes it, and it is not saved with the
    /// registration.
    type RegistrationNotes: Clone + fmt::Debug + Default + Send + Sync + 'static;

    /// The registry the workflows register workers in and remove them from.
    fn worker_registry(&self) -> &Arc<WorkerRegistry<Self::Generation>>;

    /// The clients a registration reaches its worker with.
    fn worker_client_cache(&self) -> &WorkerHttpClientCache;
}

/// The host a workflow's data was given, or the workflow's error for a run
/// that was started without one.
pub fn require_host<H>(host: Option<&Arc<H>>) -> WorkflowResult<&Arc<H>> {
    host.ok_or_else(|| WorkflowError::ContextValueNotFound("host".to_string()))
}

/// A host that keeps a registry and nothing else, for the steps' own cases.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::{Arc, OnceLock};

    use super::WorkflowHost;
    use crate::{
        connection::http_client::{WorkerHttpClientCache, WorkerHttpClientSettings},
        test_support::{Members, WorkerRegistry},
    };

    #[derive(Debug)]
    pub(crate) struct RegistryHost {
        pub(crate) worker_registry: Arc<WorkerRegistry>,
    }

    impl WorkflowHost for RegistryHost {
        type Generation = Members;
        type RegistrationNotes = ();

        fn worker_registry(&self) -> &Arc<WorkerRegistry> {
            &self.worker_registry
        }

        fn worker_client_cache(&self) -> &WorkerHttpClientCache {
            // One cache serves every case: this host keeps a registry alone.
            static CACHE: OnceLock<WorkerHttpClientCache> = OnceLock::new();
            CACHE.get_or_init(|| {
                WorkerHttpClientCache::new(WorkerHttpClientSettings {
                    client_identity: None,
                    ca_certificates: Vec::new(),
                    upstream_http2: false,
                    request_timeout_secs: 30,
                    pool_idle_timeout_secs: 0,
                })
            })
        }
    }
}
