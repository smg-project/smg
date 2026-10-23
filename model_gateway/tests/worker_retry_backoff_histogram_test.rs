//! `smg_worker_retry_backoff_seconds` is exported as a histogram, not a summary.
//!
//! Drives the recorder `start_prometheus` installs, so the bucket matchers
//! under test are the production ones. This file is its own test binary
//! (process), so installing the global Prometheus recorder here does not
//! collide with other suites.

use std::time::Duration;

use smg::observability::metrics::{start_prometheus, Metrics, PrometheusConfig};

#[test]
fn worker_retry_backoff_renders_histogram_buckets() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });

    Metrics::record_worker_retry_backoff(1, Duration::from_millis(60));
    Metrics::record_worker_retry_backoff(4, Duration::from_secs(2));

    let body = handle.render();
    let backoff_lines: Vec<&str> = body
        .lines()
        .filter(|line| line.contains("smg_worker_retry_backoff_seconds"))
        .collect();

    assert!(
        backoff_lines.contains(&"# TYPE smg_worker_retry_backoff_seconds histogram"),
        "backoff family is not a histogram:\n{}",
        backoff_lines.join("\n")
    );
    assert!(
        !backoff_lines.iter().any(|line| line.contains("quantile=")),
        "backoff family still renders summary quantiles:\n{}",
        backoff_lines.join("\n")
    );

    // A 60 ms backoff lands in the 100 ms bucket; a 2 s backoff only from the
    // 2.5 s bucket upward: the edges span tens of milliseconds to seconds.
    let bucket = |attempt: &str, le: &str| -> f64 {
        backoff_lines
            .iter()
            .find(|line| {
                line.starts_with("smg_worker_retry_backoff_seconds_bucket{")
                    && line.contains(&format!("attempt=\"{attempt}\""))
                    && line.contains(&format!("le=\"{le}\""))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "no le=\"{le}\" bucket for attempt {attempt}:\n{}",
                    backoff_lines.join("\n")
                )
            })
    };
    assert_eq!(bucket("1", "0.05"), 0.0);
    assert_eq!(bucket("1", "0.1"), 1.0);
    assert_eq!(bucket("4", "1"), 0.0);
    assert_eq!(bucket("4", "2.5"), 1.0);
    assert_eq!(bucket("4", "+Inf"), 1.0);
    assert!(
        backoff_lines
            .iter()
            .any(|line| line.starts_with("smg_worker_retry_backoff_seconds_count{")),
        "no _count series:\n{}",
        backoff_lines.join("\n")
    );
}
