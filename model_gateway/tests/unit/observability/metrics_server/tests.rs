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
    let pre = tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
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
    let pre = tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
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
