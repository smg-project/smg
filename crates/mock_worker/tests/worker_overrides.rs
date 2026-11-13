//! Per-worker engine overrides on the HTTP wire: the worker an override names
//! reports its own KV pool in `/v1/loads`, and a worker it does not name
//! reports the fleet's.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use mock_worker::{config::Config, engine::EngineParams, http};
use serde_json::{json, Value};
use tokio::net::TcpListener;

mod common;

/// Serve one realistic HTTP worker with `worker_kv_tokens` naming
/// `named(port)`, and read its `/v1/loads` capacity.
async fn pool_of(named: impl Fn(u16) -> String) -> Value {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let port = listener.local_addr().expect("listener address").port();
    let cfg = Config {
        realistic: true,
        http_base_port: port,
        http_count: 1,
        engine: EngineParams {
            kv_capacity_tokens: 131_072,
            ..EngineParams::default()
        },
        worker_kv_tokens: [(named(port), 65_536)].into(),
        ..Config::default()
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the server serves until the test process exits"
    )]
    let _server = tokio::spawn(http::serve_with_listener(Arc::new(cfg), listener));
    let client = common::http_client();
    let deadline = Instant::now() + Duration::from_secs(10);
    let loads: Value = loop {
        if let Ok(response) = client
            .get(format!("http://127.0.0.1:{port}/v1/loads"))
            .send()
            .await
        {
            break response.json().await.expect("a loads answer");
        }
        assert!(Instant::now() < deadline, "the HTTP worker did not come up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    loads["loads"][0]["max_total_num_tokens"].clone()
}

#[tokio::test]
async fn the_http_worker_an_override_names_reports_its_own_pool() {
    assert_eq!(pool_of(|port| format!("http:{port}")).await, json!(65_536));
    let neighbour = |port: u16| format!("http:{}", port.wrapping_add(1));
    assert_eq!(pool_of(neighbour).await, json!(131_072));
}
