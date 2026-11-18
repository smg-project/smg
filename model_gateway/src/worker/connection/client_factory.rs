//! The HTTP client factory: the worker crate's, at the path it had here.
//! [`crate::server`] re-exports the constructors, and callers read them as
//! `server::http_client()` and its siblings.

pub use smg_worker::connection::client_factory::{
    blocking_http_client, http_client, http_client_builder, install_crypto_provider,
};
