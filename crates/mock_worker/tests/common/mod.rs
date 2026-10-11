//! Common test utilities for the mock worker's integration tests.

/// The tests' HTTP client. The client takes the process-level TLS crypto
/// provider (reqwest's `rustls-no-provider` feature) and panics at build
/// time when there is none, so the crate's installer runs first; a provider
/// installed earlier in the process stays.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub fn http_client() -> reqwest::Client {
    mock_worker::tls::install_crypto_provider();
    reqwest::Client::new()
}
