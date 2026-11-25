//! The controller's protocol as Rust (package `smg.policy.v1`): the messages
//! of the two contract files, the service ([`grand_central_server`]) and
//! its client ([`grand_central_client`]), written by the build script
//! from `contracts/` in one step.

#![allow(
    clippy::all,
    clippy::absolute_paths,
    clippy::allow_attributes,
    clippy::exhaustive_enums,
    clippy::exhaustive_structs,
    clippy::trivially_copy_pass_by_ref,
    missing_docs,
    unused_qualifications,
    reason = "the generator decides which lints fire, so these allows cannot be expects"
)]

tonic::include_proto!("smg.policy.v1");
