//! A gateway that serves its own scrape records its histogram families in the
//! per-worker store's fixed cells: the scrape still reads every line of such
//! a family, once, and the exporter's handle alone no longer holds it.
//!
//! Drives the recorder `start_prometheus` installs, the call the gateway's
//! start makes right after it, and the metrics HTTP server. This file is its
//! own test binary (process): the call holds for the life of the process, and
//! the suites that read the exporter's handle alone run in other processes.

use metrics::histogram;
use smg::observability::{
    metrics::{start_prometheus, PrometheusConfig},
    metrics_server::start_metrics_server,
};
use smg_test_harness::test_app::scrape;
use smg_worker::metrics::series as worker_series;

const FAMILY: &str = "smg_http_request_duration_seconds";

#[tokio::test]
async fn the_scrape_reads_a_family_the_cells_keep_and_the_handle_alone_does_not() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    assert!(worker_series::keep_histograms());
    let (addr, _server) = start_metrics_server(handle.clone(), "127.0.0.1".to_string(), 0, None)
        .await
        .expect("metrics server binds an ephemeral port");

    let series = histogram!(FAMILY, "method" => "POST", "path" => "/v1/chat/completions");
    series.record(0.02);
    series.record(0.2);

    let body = scrape(addr).await;
    let lines: Vec<&str> = body.lines().filter(|line| line.contains(FAMILY)).collect();
    let listing = lines.join("\n");
    let kind = format!("# TYPE {FAMILY} histogram");
    assert_eq!(
        lines.iter().filter(|line| **line == kind).count(),
        1,
        "{listing}"
    );
    let bucket = |le: &str| -> Option<f64> {
        lines
            .iter()
            .find(|line| {
                line.starts_with(&format!("{FAMILY}_bucket{{"))
                    && line.contains(&format!("le=\"{le}\""))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    };
    assert_eq!(bucket("0.01"), Some(0.0), "{listing}");
    assert_eq!(bucket("0.025"), Some(1.0), "{listing}");
    assert_eq!(bucket("0.25"), Some(2.0), "{listing}");
    assert_eq!(bucket("+Inf"), Some(2.0), "{listing}");
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with(&format!("{FAMILY}_count{{")) && line.ends_with(" 2")),
        "{listing}"
    );

    assert!(
        !handle.render().contains(&format!("{FAMILY}_bucket")),
        "the store did not take the family: the exporter's text still holds it"
    );
}
