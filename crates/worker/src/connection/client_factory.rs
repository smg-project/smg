//! The HTTP client factory: the process-level rustls crypto provider's
//! install, and the client constructors that need it.
//!
//! The constructors live beside the provider because they cannot be built
//! without it: the HTTP client takes the process-level TLS crypto provider
//! (reqwest's `rustls-no-provider` feature) and panics at build time when
//! there is none, so every client of the gateway, of its bindings and of its
//! tests comes from the functions here, which install the provider first
//! (see [`install_crypto_provider`]). They sit in the worker crate, which
//! builds the worker-directed clients with them; the gateway's server module
//! re-exports them, and callers read them as `server::http_client()` and its
//! siblings. The direct constructors are disallowed by the workspace's clippy
//! configuration (`disallowed-methods` in `clippy.toml`) everywhere but here.

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT_ENCODING};
use rustls::crypto::{ring, CryptoProvider};

/// Installs `ring` as the process-level rustls crypto provider, unless one
/// is installed already.
///
/// `ring` is the only backend compiled into the gateway's binaries. The bare
/// rustls builders (the HTTPS listener, the mesh listener) pick it up on
/// their own, but the HTTP client is built with reqwest's
/// `rustls-no-provider` feature and uses whatever provider the process
/// installed, panicking when there is none. Installing is idempotent: a
/// provider installed earlier, by an embedding process or a previous start in
/// the same process, stays. The gateway's start calls it first thing, and so
/// do the client functions of this module, through which the gateway, its
/// bindings and its tests build every HTTP client; nothing else needs to.
pub fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}

/// The headers every client of the factory sends with every request: the
/// gateway names `identity` as the one content coding it accepts. The relays
/// pass an upstream body through as it came and drop `content-encoding` with
/// the other headers that must not cross, and the client is built without
/// decoders, so a coded body would reach the client unlabelled; an upstream
/// that honours the header never sends one (RFC 9110, section 12.5.3).
fn default_request_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    headers
}

/// A client with reqwest's defaults and the factory's request headers, the
/// process provider installed.
///
/// Like `reqwest::Client::new`, panics when the client cannot be built.
#[expect(
    clippy::panic,
    reason = "the same panic as `reqwest::Client::new`: a client with the defaults always builds"
)]
pub fn http_client() -> reqwest::Client {
    match http_client_builder().build() {
        Ok(client) => client,
        Err(error) => panic!("the HTTP client with the defaults did not build: {error}"),
    }
}

/// A client builder with the factory's request headers, the process provider
/// installed.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub fn http_client_builder() -> reqwest::ClientBuilder {
    install_crypto_provider();
    reqwest::Client::builder().default_headers(default_request_headers())
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

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use super::*;

    /// Both constructors send `accept-encoding: identity` with every request.
    #[tokio::test]
    async fn every_client_names_identity_as_its_accepted_content_coding() {
        let (tx, mut rx) = mpsc::unbounded_channel::<HeaderMap>();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move |headers: HeaderMap| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(headers);
                    "ok"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let with_timeout = http_client_builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        for client in [http_client(), with_timeout] {
            let response = client.get(format!("http://{addr}/")).send().await.unwrap();
            assert_eq!(response.status(), 200);
            let seen = rx.recv().await.unwrap();
            assert_eq!(
                seen.get(ACCEPT_ENCODING).map(|v| v.to_str().unwrap()),
                Some("identity")
            );
        }
    }
}
