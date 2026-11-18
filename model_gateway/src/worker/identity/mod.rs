// Open to the worker module alone. The gateway names the builder by its
// alias in `worker`, which pins the gateway's connector; a type position on
// this path would take the module's own default, no connector, silently.
pub(in crate::worker) use smg_worker::identity::builder;

#[cfg(test)]
mod gateway_tests;

// The worker crate's, at the paths they had here.
pub use smg_worker::identity::{basic, endpoint, error, metadata, sampling_defaults};
