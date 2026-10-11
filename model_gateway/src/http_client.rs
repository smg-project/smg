//! The one place this crate builds HTTP clients.
//!
//! The HTTP client takes the process-level TLS crypto provider (reqwest's
//! `rustls-no-provider` feature) and panics at build time when there is
//! none, so every client of this crate, of its bindings and of its tests
//! comes from the functions here, which install the provider first (see
//! [`crate::tls`]). The direct constructors are disallowed by the workspace's
//! clippy configuration (`disallowed-methods` in `clippy.toml`) everywhere
//! but here.

use crate::tls::install_crypto_provider;

/// A client with reqwest's defaults, the process provider installed.
///
/// Like `reqwest::Client::new`, panics when the client cannot be built.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn client() -> reqwest::Client {
    install_crypto_provider();
    reqwest::Client::new()
}

/// A client builder, the process provider installed.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn client_builder() -> reqwest::ClientBuilder {
    install_crypto_provider();
    reqwest::Client::builder()
}

/// A blocking client with reqwest's defaults, the process provider installed.
///
/// Like `reqwest::blocking::Client::new`, panics when the client cannot be
/// built.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn blocking_client() -> reqwest::blocking::Client {
    install_crypto_provider();
    reqwest::blocking::Client::new()
}
