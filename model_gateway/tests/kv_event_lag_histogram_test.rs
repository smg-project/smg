//! `smg_kv_event_lag_seconds` is exported as a histogram, not a summary.
//!
//! Drives the recorder `start_prometheus` installs, so the bucket matchers
//! under test are the production ones. This file is its own test binary
//! (process), so installing the global Prometheus recorder here does not
//! collide with other suites.

use smg::observability::metrics::{start_prometheus, Metrics, PrometheusConfig};

#[test]
fn kv_event_lag_renders_histogram_buckets() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });

    Metrics::record_kv_event_lag("grpc://worker-a:50051", 0.004);
    Metrics::record_kv_event_lag("grpc://worker-b:50051", 2.0);

    let body = handle.render();
    let lag_lines: Vec<&str> = body
        .lines()
        .filter(|line| line.contains("smg_kv_event_lag_seconds"))
        .collect();

    assert!(
        lag_lines.contains(&"# TYPE smg_kv_event_lag_seconds histogram"),
        "lag family is not a histogram:\n{}",
        lag_lines.join("\n")
    );
    assert!(
        !lag_lines.iter().any(|line| line.contains("quantile=")),
        "lag family still renders summary quantiles:\n{}",
        lag_lines.join("\n")
    );

    // A 4 ms lag lands in the 5 ms bucket; a 2 s lag only from the 2.5 s
    // bucket upward: the edges span sub-millisecond to seconds.
    let bucket = |worker: &str, le: &str| -> f64 {
        lag_lines
            .iter()
            .find(|line| {
                line.starts_with("smg_kv_event_lag_seconds_bucket{")
                    && line.contains(&format!("worker=\"{worker}\""))
                    && line.contains(&format!("le=\"{le}\""))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "no le=\"{le}\" bucket for {worker}:\n{}",
                    lag_lines.join("\n")
                )
            })
    };
    assert_eq!(bucket("grpc://worker-a:50051", "0.0025"), 0.0);
    assert_eq!(bucket("grpc://worker-a:50051", "0.005"), 1.0);
    assert_eq!(bucket("grpc://worker-b:50051", "1"), 0.0);
    assert_eq!(bucket("grpc://worker-b:50051", "2.5"), 1.0);
    assert_eq!(bucket("grpc://worker-b:50051", "+Inf"), 1.0);
    assert!(
        lag_lines
            .iter()
            .any(|line| line.starts_with("smg_kv_event_lag_seconds_count{")),
        "no _count series:\n{}",
        lag_lines.join("\n")
    );
}
