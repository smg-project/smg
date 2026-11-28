//! Engine-neutral multimodal pixel RDMA (NIXL) transport for the SMG gateway.
//!
//! The gateway stages each preprocessed image's serialized pixel buffer into a
//! pre-registered host-DRAM arena and hands the worker a small wire descriptor;
//! the worker PULLs the pixels with a one-sided RDMA READ instead of receiving
//! them inline in the gRPC frame. This crate owns only the NIXL *mechanics* and
//! the wire format — **all policy stays in the gateway**: it never reads env vars
//! or router globals; the gateway decides whether RDMA is on, builds a
//! [`RdmaConfig`], and only then constructs a [`RdmaExporter`].
//!
//! The default build compiles a no-op [`RdmaExporter`] (the `nixl` feature off),
//! so ordinary gateway builds need no NIXL headers / clang / bindgen or libnixl.
//! Enable `nixl` to compile the real one-sided-READ implementation.

use std::time::Duration;

/// Descriptor prefix (version tag) for the wire format this crate ships. The
/// worker-side parser mirrors it (grpc_servicer encoder_servicer.py).
pub const DESCRIPTOR_MAGIC: &[u8; 8] = b"SMGRDMA1";
/// Per-lease generation stamp width. Each leased slot is framed as
/// `[gen u64 LE][payload][gen u64 LE]` and the descriptor carries the same `gen`
/// so the worker can reject a slot the gateway recycled under its READ.
pub const GEN_BYTES: usize = 8;
/// Header + trailer generation stamps bracketing the payload.
pub const FRAME_OVERHEAD: usize = 2 * GEN_BYTES;

/// Whether this build stages pixels over RDMA at all: `true` under the `nixl`
/// feature, `false` for the no-op [`RdmaExporter`] of the default build, which
/// constructs but exports nothing, so every caller ships its bytes inline. A
/// reader that plans around the lane asks this before it asks for an exporter.
pub const STAGES_OVER_RDMA: bool = cfg!(feature = "nixl");

/// Everything the exporter needs, injected by the gateway. The crate reads no
/// environment and no globals: the gateway parses env / flags and fills this in.
#[derive(Debug, Clone)]
pub struct RdmaConfig {
    /// The gateway's RDMA listener IP (its RoCE address) the worker dials for the
    /// NIXL metadata exchange. Empty => the exporter is unavailable (caller stays
    /// on the inline path).
    pub listen_ip: String,
    /// The gateway's NIXL listener port.
    pub listen_port: u16,
    /// Fixed agent name the worker passes to `fetch_remote_metadata`.
    pub agent_name: String,
    /// Number of slots in the pre-registered arena.
    pub pool_slots: usize,
    /// Per-slot byte capacity (must hold one framed image).
    pub slot_bytes: usize,
    /// How long a leased slot may live without a free-notif before the reaper
    /// force-reclaims it. MUST exceed the worker's max hold (the gateway derives it
    /// so); the per-lease gen framing makes correctness independent of the value.
    pub slot_ttl: Duration,
}

// Pure slot bookkeeping (free-list + lease/reclaim/reuse policy + memcpy). Needs
// no NIXL agent or hardware, so it is unit-tested on its own. Compiled for the
// real impl and for `cargo test`; the no-op stub build skips it.
#[cfg(any(feature = "nixl", test))]
mod slot_pool;

#[cfg(feature = "nixl")]
mod nixl;
#[cfg(feature = "nixl")]
pub use nixl::{RdmaError, RdmaExporter};

#[cfg(not(feature = "nixl"))]
mod stub;
#[cfg(not(feature = "nixl"))]
pub use stub::{RdmaError, RdmaExporter};
