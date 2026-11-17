pub mod basic;
// Open to the worker module alone. The gateway names the builder by its
// alias in `worker`, which pins the gateway's connector; a type position on
// this path would take the module's own default, no connector, silently.
pub(in crate::worker) mod builder;
pub mod endpoint;
pub mod error;
pub mod metadata;
pub mod sampling_defaults;
