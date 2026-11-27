use std::sync::Arc;

use openai_protocol::worker::ConnectionMode;

use super::*;
use crate::{
    config::RouterConfig, routers::grpc::backend_client::BasicWorkerBuilder, server::http_client,
};

fn view_with(config: &RouterConfig, workers: Vec<Arc<dyn Worker>>) -> RegistryRlView {
    let registry = Arc::new(WorkerRegistry::new());
    for w in workers {
        registry.register(w);
    }
    let cache = Arc::new(WorkerHttpClientCache::new(
        config.worker_http_client_settings(),
    ));
    RegistryRlView::new(registry, cache)
}

fn view_over(workers: Vec<Arc<dyn Worker>>) -> RegistryRlView {
    view_with(&RouterConfig::default(), workers)
}

/// A gRPC worker whose engine advertised a wildcard-bound control app.
fn grpc_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Grpc)
            .label("rl.control_url", "http://0.0.0.0:40100")
            .build(),
    )
}

/// Control calls to an HTTP worker use the client the gateway negotiated
/// for it (HTTP version, TLS identity, pool tuning), not a second one.
#[test]
fn http_worker_controls_itself_with_its_negotiated_client() {
    let client = Arc::new(http_client());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://engine:30000")
            .connection_mode(ConnectionMode::Http)
            .http_client(Arc::clone(&client))
            .build(),
    );
    let info = view_over(vec![worker]).list().pop().expect("one worker");
    assert_eq!(info.control_url.as_deref(), Some("http://engine:30000"));
    assert!(Arc::ptr_eq(&info.control_client.expect("client"), &client));
}

#[test]
fn grpc_worker_with_a_label_gets_a_control_endpoint_and_a_client() {
    let info = view_over(vec![grpc_worker("grpc://10.0.0.5:30000")])
        .list()
        .pop()
        .expect("one worker");
    assert_eq!(
        info.control_url.as_deref(),
        Some("http://10.0.0.5:40100"),
        "wildcard bind host resolves to the worker host"
    );
    assert!(info.control_client.is_ok());
}

#[test]
fn grpc_worker_without_a_label_has_no_control_endpoint() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://engine:30000")
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    );
    let info = view_over(vec![worker]).list().pop().expect("one worker");
    assert!(info.control_url.is_none());
}

#[test]
fn grpc_worker_with_a_blank_label_has_no_control_endpoint() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://engine:30000")
            .connection_mode(ConnectionMode::Grpc)
            .label("rl.control_url", "  ")
            .build(),
    );
    let info = view_over(vec![worker]).list().pop().expect("one worker");
    assert!(info.control_url.is_none());
}

/// A TokenSpeed engine on the direct ZMQ lane is controlled over its own
/// connection (one still handshaking is taken at its word; a connected one
/// that declared no control reply is not, which needs a live handshake to
/// show: the adapter's own test covers that refusal); another runtime's
/// ZMQ worker and a gRPC worker are not.
#[test]
fn a_tokenspeed_zmq_worker_is_controlled_over_its_engine_connection() {
    let zmq = |url: &str, runtime: RuntimeType| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(ConnectionMode::Zmq)
                .runtime_type(runtime)
                .build(),
        )
    };
    let infos = view_over(vec![
        zmq("ipc:///tmp/smg-zmq/ts", RuntimeType::TokenSpeed),
        zmq("ipc:///tmp/smg-zmq/vllm", RuntimeType::Vllm),
        grpc_worker("grpc://a:1"),
    ])
    .list();
    let calls = |url: &str| {
        infos
            .iter()
            .find(|info| info.url == url)
            .expect("registered")
            .control_calls
            .is_some()
    };
    assert!(calls("ipc:///tmp/smg-zmq/ts"));
    assert!(!calls("ipc:///tmp/smg-zmq/vllm"));
    assert!(!calls("grpc://a:1"));
}

#[test]
fn control_clients_are_shared_across_workers_with_the_same_pool_config() {
    let infos = view_over(vec![grpc_worker("grpc://a:1"), grpc_worker("grpc://b:1")]).list();
    let a = infos[0].control_client.as_ref().expect("a");
    let b = infos[1].control_client.as_ref().expect("b");
    assert!(Arc::ptr_eq(a, b));
}

/// The gateway cache holds only weak handles and a gRPC worker holds
/// none, so the view keeps the strong one: a client is built once, not
/// on every discovery or control call.
#[test]
fn the_view_holds_the_control_client_between_calls() {
    let view = view_over(vec![grpc_worker("grpc://a:1")]);
    let client = view
        .list()
        .pop()
        .expect("one worker")
        .control_client
        .expect("client");
    assert_eq!(
        Arc::strong_count(&client),
        2,
        "the view holds the other handle"
    );
    view.registry.remove_by_url("grpc://a:1");
    assert!(view.list().is_empty());
    assert_eq!(
        Arc::strong_count(&client),
        1,
        "no registered worker needs the handle anymore"
    );
}

/// A gateway whose TLS settings cannot produce a client is the worker's
/// problem to report, not a silent `None`: the error rides on the info
/// and ends up in the 422. (An unparsable client identity fails the
/// build eagerly; CA bundles are only read when a connection is made.)
#[test]
fn a_failed_client_build_is_reported_on_the_worker() {
    let config = RouterConfig {
        client_identity: Some(b"not a pem".to_vec()),
        ..RouterConfig::default()
    };
    let info = view_with(&config, vec![grpc_worker("grpc://a:1")])
        .list()
        .pop()
        .expect("one worker");
    assert_eq!(info.control_url.as_deref(), Some("http://a:40100"));
    let err = info.control_client.expect_err("no client");
    assert!(err.contains("client identity"), "{err}");
}
