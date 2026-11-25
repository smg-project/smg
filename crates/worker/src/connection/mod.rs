//! How the runtime reaches a worker: the engine handle a host implements,
//! and over HTTP the client factory and the cache of worker-directed clients.

pub mod client_factory;
pub mod engine;
pub(crate) mod engine_admin;
pub mod http_client;
