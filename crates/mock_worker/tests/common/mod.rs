//! Common test utilities for the mock worker's integration tests.
#![allow(
    dead_code,
    clippy::allow_attributes,
    reason = "each test binary reads the helpers it needs; the others are unused there"
)]

use std::sync::Arc;

use anyhow::Context as _;
use futures::StreamExt;
use mock_worker::{admin, config::Config, grpc::serve_with_listener};
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::net::TcpListener;

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

/// A port nothing listens on right now (bound and released).
pub fn free_port() -> anyhow::Result<u16> {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .context("bind a free port")?
        .local_addr()
        .context("listener address")?
        .port();
    Ok(port)
}

/// Serve `cfg` on a free port and connect the gateway's client to it.
pub async fn start(cfg: Config) -> anyhow::Result<TokenSpeedSchedulerClient> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind a free port")?;
    let port = listener.local_addr().context("listener address")?.port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
    let client = TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .map_err(anyhow::Error::from_boxed)
        .context("connect to the mock worker")?;
    Ok(client)
}

/// Serve the admin API beside the process's workers on a port read back from
/// a listener bound here: the server holds the port before the test reads
/// it, so a neighbour of the test binary (or of the host) cannot take it
/// between a probe and the bind.
pub async fn start_admin(cfg: Config) -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind the admin listener")?;
    let port = listener.local_addr().context("listener address")?.port();
    #[expect(
        clippy::disallowed_methods,
        reason = "the admin server serves until the test process exits"
    )]
    let _admin = tokio::spawn(admin::serve_with_listener(Arc::new(cfg), listener));
    Ok(port)
}

/// Send `req` and read every frame of the response.
pub async fn generate(
    client: &TokenSpeedSchedulerClient,
    req: ts::GenerateRequest,
) -> anyhow::Result<Vec<ts::GenerateResponse>> {
    let mut stream = client.generate(req).await.context("generate")?;
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(*frame.context("a response frame")?);
    }
    stream.mark_completed();
    Ok(frames)
}
