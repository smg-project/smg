//! HTTP server for the Prometheus metrics endpoint (default port 29000).
//! Serves `GET /metrics` (Prometheus).

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use smg_grand_central::engine::InProcess;
use smg_worker::metrics::series as worker_series;
use tokio::task::JoinHandle;
use tracing::{error, info};

use super::metrics::{ShadowsPublished, UPKEEP_INTERVAL_SECS};
use crate::config::bind_socket_addr;

#[derive(Clone)]
struct MetricsState {
    handle: PrometheusHandle,
    /// Grand Central, where the gateway started one, and what the
    /// scrapes so far published of its shadow counts. A scrape reads the
    /// engine's fault counts too, and keeps nothing of them.
    engine: Option<(&'static InProcess, Arc<ShadowsPublished>)>,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    super::metrics::record_tokenizer_cache_activity();
    super::metrics::record_detokenize_activity();
    super::metrics::record_tokenizer_encode_activity();
    if let Some((engine, published)) = &state.engine {
        super::metrics::record_policy_shadow_activity(engine, published);
        super::metrics::record_policy_fault_activity(engine);
    }
    let mut body = state.handle.render();
    worker_series::render_into(&mut body);
    (
        [(
            http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
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
///
/// `engine` is the Grand Central the gateway started, where it started one:
/// each scrape publishes that engine's shadow counts too
/// (`smg_policy_shadow_decisions_total`), and the panics it held and counted
/// (`smg_policy_faults_total`). With no engine neither family is published.
pub async fn start_metrics_server(
    handle: PrometheusHandle,
    host: String,
    port: u16,
    engine: Option<&'static InProcess>,
) -> Result<(SocketAddr, JoinHandle<()>), String> {
    // The same spellings as `--host` (`::` and `[::]` both bind IPv6), and a
    // bad host is a startup error: a silent fallback to 0.0.0.0 would hide a
    // typo and leave an IPv6-only network without metrics.
    let addr = bind_socket_addr(&host, port).map_err(|e| format!("metrics server: {e}"))?;

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
            worker_series::run_upkeep();
        }
    });

    let prom_state = MetricsState {
        handle,
        engine: engine.map(|engine| (engine, Arc::default())),
    };

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
#[path = "../../tests/unit/observability/metrics_server/tests.rs"]
mod tests;
