//! Engines for the tests of the gRPC routers, on loopback ports: the
//! `mock_worker` crate's gRPC server, so a test's client has an engine to
//! connect to, and the serving of a test's own scripted engine. Both bind the
//! port before they hand it out: a port picked first and bound later is
//! another test's in between on a loaded host, and the stub then fails with
//! "address in use". The test supplies the frames it reads itself, or sends
//! the engine nothing, and aborts the server task when it is done or leaves
//! it to end with the test process.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use tokio::{net::TcpListener, task::JoinHandle};
use tonic::transport::server::{Router, TcpIncoming};

/// A mock engine serving `model_id` on a loopback port, as its URL and its
/// server task.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; server task is explicitly aborted"
)]
pub(crate) async fn mock_engine(model_id: &str) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock worker");
    let port = listener.local_addr().expect("mock worker address").port();
    let config = Arc::new(mock_worker::config::Config {
        admin_port: None,
        context_length: 32768,
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: model_id.to_string(),
        tokenizer_path: model_id.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 0,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        ..mock_worker::config::Config::default()
    });
    // A port an earlier test's engine served on keeps that engine's
    // overrides; this engine starts from fresh ones.
    mock_worker::overrides::fresh_for_port(port);
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    (format!("http://127.0.0.1:{port}"), server)
}

/// `router` (a tonic server with its services added) served on a loopback
/// port that is bound before this returns, as the address and the server
/// task. A connection made right after this returns waits in the listener's
/// backlog until the task accepts it, so no caller polls for the port. The
/// accepted connections carry `TCP_NODELAY`, as `serve(addr)` sets it through
/// the builder's default; tonic leaves a caller-provided stream as given.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; the caller aborts the server task or lets it end with the test"
)]
pub(crate) async fn serve_on_loopback(router: Router) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback port");
    let addr = listener.local_addr().expect("the bound address");
    let server = tokio::spawn(async move {
        router
            .serve_with_incoming(TcpIncoming::from(listener).with_nodelay(Some(true)))
            .await
            .expect("the test's engine server");
    });
    (addr, server)
}
