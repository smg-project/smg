//! The one place this crate builds HTTP clients.
//!
//! The HTTP client takes the process-level TLS crypto provider (reqwest's
//! `rustls-no-provider` feature) and panics at build time when there is
//! none, so every client of this crate, of its bindings and of its tests
//! comes from the functions here, which install the provider first (see
//! [`install_crypto_provider`]). The direct constructors are disallowed by the workspace's
//! clippy configuration (`disallowed-methods` in `clippy.toml`) everywhere
//! but here.

use rustls::crypto::{ring, CryptoProvider};

/// Installs `ring` as the process-level rustls crypto provider, unless one
/// is installed already.
///
/// `ring` is the only backend compiled into this crate's binaries. The bare
/// rustls builders (the HTTPS listener, the mesh listener) pick it up on
/// their own, but the HTTP client is built with reqwest's
/// `rustls-no-provider` feature and uses whatever provider the process
/// installed, panicking when there is none. Installing is idempotent: a
/// provider installed earlier, by an embedding process or a previous start in
/// the same process, stays. [`crate::server::startup`] calls it first thing,
/// and so do the client functions of this module, through which this
/// crate, its bindings and its tests build every HTTP client; nothing else
/// needs to.
pub fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}

/// A client with reqwest's defaults, the process provider installed.
///
/// Like `reqwest::Client::new`, panics when the client cannot be built.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn http_client() -> reqwest::Client {
    install_crypto_provider();
    reqwest::Client::new()
}

/// A client builder, the process provider installed.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn http_client_builder() -> reqwest::ClientBuilder {
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
pub fn blocking_http_client() -> reqwest::blocking::Client {
    install_crypto_provider();
    reqwest::blocking::Client::new()
}
