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
    shadows: Option<(&'static InProcess, Arc<ShadowsPublished>)>,
}

async fn prometheus_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    super::metrics::record_tokenizer_cache_activity();
    super::metrics::record_detokenize_activity();
    super::metrics::record_tokenizer_encode_activity();
    if let Some((engine, published)) = &state.shadows {
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
        shadows: engine.map(|engine| (engine, Arc::default())),
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
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use smg_grand_central::document::{AppliedSpec, PolicySpec};

    use super::*;
    use crate::{
        observability::metrics::test_support::{fault_kinds, shadow_kinds, shadowing_spec},
        server::http_client,
    };

    /// One scrape of an endpoint started with `engine`, under a recorder of
    /// its own.
    async fn scraped(engine: Option<&'static InProcess>) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let (handle, host) = (recorder.handle(), "127.0.0.1".to_string());
        let (addr, _server) = start_metrics_server(handle, host, 0, engine).await.unwrap();
        let url = format!("http://{addr}/metrics");
        let scrape = http_client().get(url).send().await.unwrap();
        scrape.text().await.unwrap()
    }

    /// A scrape of a gateway that started Grand Central publishes that
    /// engine's shadow counts: a zero series of each kind for the binding
    /// that names a shadow, before any request. A gateway that started no
    /// engine publishes no such family, and neither does an engine whose
    /// policy document names no shadow.
    #[tokio::test]
    async fn a_scrape_publishes_the_shadow_counts_of_the_engine_it_was_started_with() {
        let engine = |spec| -> &'static InProcess { Box::leak(Box::new(InProcess::new(spec))) };
        let shadowing = scraped(Some(engine(shadowing_spec()))).await;
        assert_eq!(shadow_kinds(&shadowing), [Some(0); 4], "{shadowing}");

        let preset = PolicySpec::preset("round_robin").expect("a stock preset");
        let plain = AppliedSpec::validate(&preset).expect("a valid policy document");
        for unshadowed in [None, Some(engine(plain))] {
            let scrape = scraped(unshadowed).await;
            assert!(
                !scrape.contains("smg_policy_shadow_decisions_total"),
                "{scrape}"
            );
        }
    }

    /// A scrape of a gateway that started Grand Central publishes the
    /// panics that engine held and counted: a series of each kind, at zero
    /// before any panic. A gateway that started no engine publishes no such
    /// family.
    #[tokio::test]
    async fn a_scrape_publishes_the_fault_counts_of_the_engine_it_was_started_with() {
        let preset = PolicySpec::preset("round_robin").expect("a stock preset");
        let plain = AppliedSpec::validate(&preset).expect("a valid policy document");
        let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(plain)));
        let started = scraped(Some(engine)).await;
        assert_eq!(fault_kinds(&started), [Some(0); 3], "{started}");

        let classic = scraped(None).await;
        assert!(!classic.contains("smg_policy_faults_total"), "{classic}");
    }

    #[tokio::test]
    async fn ipv6_host_binds_ipv6_with_or_without_brackets() {
        // Nothing to assert on a host without an IPv6 loopback.
        if std::net::TcpListener::bind("[::1]:0").is_err() {
            return;
        }
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        for host in ["[::1]", "::1"] {
            let (addr, _server) = start_metrics_server(handle.clone(), host.to_string(), 0, None)
                .await
                .unwrap();
            assert!(addr.ip().is_ipv6(), "host {host} bound {addr}");
            let resp = http_client()
                .get(format!("http://{addr}/metrics"))
                .send()
                .await
                .unwrap();
            assert!(resp.status().is_success(), "host {host}");
        }
    }

    #[tokio::test]
    async fn unparsable_host_is_an_error_not_an_ipv4_fallback() {
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let err = start_metrics_server(handle, "metrics-host".to_string(), 0, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("invalid bind host 'metrics-host'"),
            "got: {err}"
        );
    }

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
        let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0, None)
            .await
            .unwrap();
        assert_ne!(addr.port(), 0);

        let resp = http_client()
            .get(format!("http://{addr}/metrics"))
            .send()
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
        let err = start_metrics_server(handle, "127.0.0.1".to_string(), busy.port(), None)
            .await
            .unwrap_err();
        assert!(err.contains("failed to bind metrics server"), "got: {err}");
    }
}
