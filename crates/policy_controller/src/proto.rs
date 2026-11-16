//! The controller's protocol as Rust (package `smg.policy.v1`): the messages
//! of the two contract files, the service ([`policy_controller_server`]) and
//! its client ([`policy_controller_client`]), written by the build script
//! from `contracts/` in one step. The engine wire's messages a relay forwards
//! are the gRPC client crate's own (`smg_grpc_client::common_proto`).

#![allow(
    clippy::all,
    clippy::absolute_paths,
    clippy::allow_attributes,
    clippy::trivially_copy_pass_by_ref,
    missing_docs,
    unused_qualifications,
    reason = "the generator decides which lints fire, so these allows cannot be expects"
)]

tonic::include_proto!("smg.policy.v1");
