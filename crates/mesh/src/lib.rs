//! Mesh Gossip Protocol and Distributed State Synchronization
//!
//! This crate provides mesh networking capabilities for distributed cluster state management:
//! - Gossip protocol for node discovery and failure detection
//! - CRDT-based state synchronization across cluster nodes
//! - Partition detection and recovery

mod crdt_kv;
mod gossip_controller;
mod gossip_service;
pub mod kv;
mod metrics;
mod mtls;
mod partition;
mod service;
mod transport;
mod types;

// Internal tests module with full access to private types
#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use crdt_kv::{
    decode as decode_epoch_count, encode as encode_epoch_count, CrdtChange, CrdtOrMap, EpochCount,
    MergeStrategy, OperationLog, EPOCH_MAX_WINS_ENCODED_LEN,
};
pub use kv::{
    CrdtNamespace, DrainHandle, MeshKV, StreamConfig, StreamDrainFn, StreamNamespace,
    StreamRouting, Subscription,
};
pub use metrics::init_mesh_metrics;
pub use mtls::{MTLSConfig, MTLSManager};
pub use partition::PartitionDetector;
pub use service::{
    gossip, ClusterState, MeshServerBuilder, MeshServerConfig, MeshServerHandler,
    PreparedMeshServer,
};
pub use transport::limits::MAX_STREAM_CHUNK_BYTES;
pub use types::WorkerState;
