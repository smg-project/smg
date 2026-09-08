//! Generated tonic bindings for the Router-to-Worker control plane.
//!
//! `worker_control.proto` and `worker_inference.proto` share the
//! `smg.worker.v1` package, so `tonic-prost-build` emits a single
//! `smg.worker.v1.rs` carrying both services and every message. This module
//! owns the one `include_proto!` for that package; [`crate::worker_inference`]
//! re-exports it rather than including it a second time, which would compile
//! two mutually-incompatible copies of each message type.

#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("smg.worker.v1");
}
