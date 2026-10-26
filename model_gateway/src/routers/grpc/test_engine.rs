//! A mock engine for the tests of the gRPC router: the `mock_worker` crate's
//! gRPC server on a loopback port, so a test's client has an engine to
//! connect to. The test supplies the frames it reads itself, or sends the
//! engine nothing, and aborts the server task when it is done.

use std::{sync::Arc, time::Duration};

use tokio::{net::TcpListener, task::JoinHandle};

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
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    (format!("http://127.0.0.1:{port}"), server)
}

/// A loopback port nothing listens on.
pub(crate) fn closed_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    drop(listener);
    port
}
