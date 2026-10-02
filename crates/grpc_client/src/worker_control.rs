//! Generated tonic bindings for the Router-to-Worker control plane.
//!
//! `worker_control.proto` and `worker_inference.proto` share the
//! `smg.worker.v1` package, so `tonic-prost-build` emits a single
//! `smg.worker.v1.rs` carrying both services and every message. This module
//! owns the one `include_proto!` for that package; [`crate::worker_inference`]
//! re-exports it rather than including it a second time, which would compile
//! two mutually-incompatible copies of each message type.

/// `WorkerCapabilities.api_major` advertised by every Worker built from this
/// crate; a Router refuses a Worker whose major differs from its own.
pub const WORKER_CONTROL_API_MAJOR: u32 = 1;
/// `WorkerCapabilities.api_minor` advertised by every Worker built from this
/// crate; bumped for additive changes within a major.
pub const WORKER_CONTROL_API_MINOR: u32 = 0;

#[expect(
    clippy::allow_attributes,
    reason = "generated code needs a blanket allow, which cannot be an expect"
)]
pub mod proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("smg.worker.v1");
}
