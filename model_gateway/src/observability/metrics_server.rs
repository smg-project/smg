//! HTTP server for the Prometheus metrics endpoint (default port 29000).
//! Serves `GET /metrics` (Prometheus).

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::task::JoinHandle;
use tracing::{error, info};

use super::metrics::UPKEEP_INTERVAL_SECS;

#[derive(Clone)]
struct MetricsState {
    handle: PrometheusHandle,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    (
        [(
            http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.handle.render(),
    )
}

async fn bind_metrics_listener(addr: SocketAddr) -> Result<tokio::net::TcpListener, String> {
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind metrics server on {addr}: {e}"))
}

/// Start the metrics HTTP/WS server. Binds eagerly so callers fail fast on
/// port conflicts or bad addresses; port 0 binds an OS-assigned ephemeral
/// port. Returns the bound address and the server task handle.
pub async fn start_metrics_server(
    handle: PrometheusHandle,
    host: String,
    port: u16,
) -> Result<(SocketAddr, JoinHandle<()>), String> {
    let ip_addr: IpAddr = host.parse().unwrap_or_else(|e| {
        error!("Failed to parse metrics host '{host}': {e}, falling back to 0.0.0.0");
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    });
    let addr = SocketAddr::new(ip_addr, port);

    let listener = bind_metrics_listener(addr).await?;
    let bound_addr = listener
        .local_addr()
        .map_err(|e| format!("failed to read metrics server local address: {e}"))?;

    info!("Metrics server listening on {bound_addr} (/metrics)");

    // Spawn upkeep task — required by install_recorder() for histogram maintenance.
    let upkeep_handle = handle.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "upkeep task runs for the lifetime of the process"
    )]
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(UPKEEP_INTERVAL_SECS)).await;
            upkeep_handle.run_upkeep();
        }
    });

    let prom_state = MetricsState { handle };

    let app = Router::new().route("/metrics", get(prometheus_handler).with_state(prom_state));

    #[expect(
        clippy::disallowed_methods,
        reason = "metrics server runs for the lifetime of the process"
    )]
    let server_task = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            error!("Metrics server error: {e}");
        }
    });

    Ok((bound_addr, server_task))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bind_metrics_listener_error_includes_addr() {
        let pre =
            tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .await
                .unwrap();
        let busy = pre.local_addr().unwrap();
        let err = bind_metrics_listener(busy).await.unwrap_err();
        assert!(err.contains(&busy.to_string()), "got: {err}");
        assert!(err.contains("failed to bind metrics server"), "got: {err}");
    }

    #[tokio::test]
    async fn port_zero_binds_ephemeral_and_serves_metrics() {
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
            .await
            .unwrap();
        assert_ne!(addr.port(), 0);

        let resp = reqwest::get(format!("http://{addr}/metrics"))
            .await
            .unwrap();
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn busy_port_returns_error_instead_of_panicking() {
        let pre =
            tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .await
                .unwrap();
        let busy = pre.local_addr().unwrap();

        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let err = start_metrics_server(handle, "127.0.0.1".to_string(), busy.port())
            .await
            .unwrap_err();
        assert!(err.contains("failed to bind metrics server"), "got: {err}");
    }
}
