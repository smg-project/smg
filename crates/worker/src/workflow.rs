//! The worker side of the workflows a host runs over its workers, on the
//! workflow engine (`wfaas`): the steps, the data they pass from one to the
//! next, and what they ask of their host.
//!
//! A host is the process that keeps the worker registry: a gateway, or a
//! controller that registers the workers it finds. A step reads its host
//! through [`WorkflowHost`] and nothing else of it, so every host runs the
//! same steps. A workflow's data is generic over the host: the steps a host
//! adds to the workflow for what only it keeps of a worker read the host as
//! the type it is.
//!
//! Here stands the removal of a worker ([`worker_removal_workflow`]): the
//! selection of the workers an address names ([`FindWorkersToRemoveStep`]),
//! their drain ([`DrainWorkersStep`]) and their removal from the registry
//! ([`RemoveFromWorkerRegistryStep`]).
//!
//! Of a worker's registration it holds the data
//! ([`WorkerRegistrationWorkflowData`]): what the steps pass along, with one
//! slot for what the host's own steps keep between them.
//!
//! A workflow's order is this crate's: its step ids, timeouts and retries
//! are written once, here. A host adds its own steps to a workflow by the
//! engine's `depends_on`, on a step id this crate names.

mod data;
mod drain_workers;
mod find_workers_to_remove;
mod host;
mod pool;
mod registration;
mod removal;
mod remove_from_worker_registry;
mod retry;
mod select;

pub use data::{
    DpInfo, WorkerKind, WorkerList, WorkerRegistrationWorkflowData, WorkerRemovalWorkflowData,
};
pub use drain_workers::DrainWorkersStep;
pub use find_workers_to_remove::{FindWorkersToRemoveStep, WorkerRemovalRequest};
pub use host::{require_host, WorkflowHost};
pub use pool::{pool_group, refresh_worker_pool_sizes, PoolGroup, PoolGroups};
pub use registration::WORKER_REGISTRATION_WORKFLOW;
pub use removal::{
    worker_removal_workflow, DRAIN_WORKERS_STEP, FIND_WORKERS_TO_REMOVE_STEP,
    REMOVE_FROM_WORKER_REGISTRY_STEP, WORKER_REMOVAL_WORKFLOW,
};
pub use remove_from_worker_registry::RemoveFromWorkerRegistryStep;
pub use retry::single_attempt;
pub use select::find_workers_by_url;
