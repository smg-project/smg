//! The worker side of the workflows a host runs over its workers. This
//! module holds the steps, the data they pass along and what they ask of
//! their host.
//!
//! A host is the process that keeps the worker registry: a gateway, or a
//! controller that registers the workers it finds. A step reads its host
//! through [`WorkflowHost`]. A workflow's data is generic over the host, so
//! the host's own steps can use its concrete type.
//!
//! With [`RemovalOrder::Registry`], a removal selects workers, drains them
//! and removes them from the registry.
//! With [`RemovalOrder::Door`], it selects workers and waits for them to
//! settle between the host's drain and removal.
//!
//! Of a worker's registration it holds the data
//! ([`WorkerRegistrationWorkflowData`]: what the steps pass along, with one
//! slot for what the host's own steps keep between them) and the detection
//! of how the worker is spoken to ([`DetectConnectionModeStep`]) and of
//! the runtime it runs ([`DetectRuntimeStep`]), and the reads of what its
//! engine reports about itself ([`DiscoverMetadataStep`]) and of its
//! data-parallel shape ([`DiscoverDpInfoStep`]). [`CreateLocalWorkerStep`]
//! builds workers from the spec and what detection found. These steps run
//! in the order of [`worker_registration_workflow`]. A host uses
//! [`RegistrationGates`] to name the steps of its own that detection or
//! creation waits on.
//!
//! A workflow's order is this crate's: its step ids, timeouts and retries
//! are written once, here. A host adds its own steps to a workflow by the
//! engine's `depends_on`, on a step id this crate names.

mod create_worker;
mod data;
mod detect_connection;
mod detect_runtime;
mod discover_dp;
mod discover_metadata;
mod drain_workers;
mod find_workers_to_remove;
mod host;
mod metadata;
mod model_list;
mod reach;
mod registration;
mod removal;
mod remove_from_worker_registry;
mod retry;
mod runtimes;
mod select;

pub use create_worker::{
    build_model_card, kv_transfer_value, labels_for_rank, model_card_for_labels, report_for_rank,
    resolve_model_id, validate_constrained_decoding_label, CreateLocalWorkerStep,
    CREATE_LOCAL_WORKER_STEP, DP_WIDTH_LABELS,
};
pub use data::{
    DpInfo, WorkerKind, WorkerList, WorkerRegistrationWorkflowData, WorkerRemovalWorkflowData,
};
pub use detect_connection::DetectConnectionModeStep;
pub use detect_runtime::DetectRuntimeStep;
pub use discover_dp::{dp_info_from_labels, DiscoverDpInfoStep};
pub use discover_metadata::DiscoverMetadataStep;
pub use drain_workers::{pin, DrainWorkersStep, Pin, SettleDrainedWorkersStep};
pub use find_workers_to_remove::{FindWorkersToRemoveStep, WorkerRemovalRequest};
pub use host::{require_host, WorkflowHost};
pub use metadata::discover_engine_labels;
pub use reach::{
    grpc_health_answer, grpc_reachable_url, http_base_url, http_health_url, try_grpc_reachable,
    try_http_reachable,
};
pub use registration::{
    startup_attempts, worker_registration_workflow, RegistrationGates, DETECT_CONNECTION_MODE_STEP,
    DETECT_RUNTIME_STEP, DISCOVER_DP_STEP, DISCOVER_METADATA_STEP, WORKER_REGISTRATION_WORKFLOW,
};
pub use removal::{
    worker_removal_workflow, RemovalOrder, DRAIN_WORKERS_STEP, FIND_WORKERS_TO_REMOVE_STEP,
    REMOVE_FROM_WORKER_REGISTRY_STEP, SETTLE_DRAINED_WORKERS_STEP, SETTLE_STEP_TIMEOUT,
    WORKER_REMOVAL_WORKFLOW,
};
pub use remove_from_worker_registry::RemoveFromWorkerRegistryStep;
pub use retry::single_attempt;
pub use select::find_workers_by_url;
