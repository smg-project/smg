//! The retry policy shape shared by every router family and by the gateway
//! configuration that sets it. The worker crate defines it, since a worker's
//! resolved resilience and the registry's per-model store hold it.

pub use smg_worker::settings::RetryConfig;
