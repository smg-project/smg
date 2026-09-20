//! The node-local SMG Worker: the process that fronts one colocated engine and
//! serves `WorkerControl`, `grpc.health.v1`, and `WorkerInference` to Routers.
//!
//! [`crate::worker`] is the Router's model of remote workers; this module is
//! the Worker itself.

pub mod control;
pub mod engine_transport;

pub use control::{
    init_tracing, parse_health_state, WorkerHealthState, WorkerNodeConfig, WorkerNodeError,
    WorkerNodeServer,
};
pub use engine_transport::ZmqWorkerTransport;
