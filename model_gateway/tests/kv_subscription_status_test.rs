//! Verify subscription status under actual registry-owned metric retention.
use std::{sync::Arc, time::Duration};

use openai_protocol::worker::HealthCheckConfig;
use smg::{
    observability::metrics::{start_prometheus, Metrics, MetricsHandle, PrometheusConfig},
    worker::{
        BasicWorkerBuilder, ConnectionMode, KvEventMonitor, RuntimeType, Worker, WorkerRegistry,
    },
};

async fn wait_for_state(handle: &MetricsHandle, url: &str, state: i8) {
    let expected = format!("smg_kv_event_subscription_state{{worker=\"{url}\"}} {state}");
    let result = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if handle.render().lines().any(|line| line == expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(
        result.is_ok(),
        "subscription status must become scrape-visible"
    );
}

fn has_worker(handle: &MetricsHandle, url: &str) -> bool {
    handle
        .render()
        .lines()
        .any(|line| line.contains(&format!("worker=\"{url}\"")))
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "mock server is aborted and joined after lifecycle assertions"
)]
async fn stopped_subscription_is_visible_only_while_registry_owns_worker() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        ..Default::default()
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(
        Arc::new(mock_worker::config::Config {
            realistic: false,
            ..Default::default()
        }),
        listener,
    ));
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(&url)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let registry = WorkerRegistry::new();
    let id = registry.register(worker.clone()).unwrap();
    let monitor = KvEventMonitor::new(None);
    monitor.on_worker_added(&worker).await;
    wait_for_state(&handle, &url, -1).await;
    assert!(
        worker.is_healthy(),
        "unsupported KV events must not remove generation health"
    );
    let help = handle
        .render()
        .lines()
        .find(|line| line.starts_with("# HELP smg_kv_event_subscription_state "))
        .unwrap()
        .to_string();
    assert!(help.contains("-3=stopped"));
    assert!(help.contains("absent after final worker deregistration"));

    // Stopping just the monitor leaves registry ownership and a truthful stopped state.
    monitor.stop().await;
    wait_for_state(&handle, &url, -3).await;
    monitor.on_worker_added(&worker).await;
    wait_for_state(&handle, &url, -1).await;

    // Production removal releases the lease before the monitor's cleanup writes.
    registry.remove(&id).unwrap();
    assert!(!has_worker(&handle, &url));
    monitor.on_worker_removed(&url).await;
    Metrics::record_kv_event_lag(&url, 2.0);
    assert!(
        !has_worker(&handle, &url),
        "late cleanup recreated retired samples"
    );

    let id = registry.register(worker.clone()).unwrap();
    monitor.on_worker_added(&worker).await;
    wait_for_state(&handle, &url, -1).await;
    registry.remove(&id).unwrap();
    monitor.stop().await;
    assert!(!has_worker(&handle, &url));
    server.abort();
    let _ = server.await;
}
