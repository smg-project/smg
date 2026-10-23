//! Mesh transport layer: wire-size limits, idle timeouts, chunking,
//! and reassembly. Subsystems that build or consume sync_stream frames
//! depend on this module.

pub mod chunk_assembler;
pub mod chunking;
pub mod crdt_batch;
pub mod limits;
pub mod peer_stream;
pub mod sync_stream;
