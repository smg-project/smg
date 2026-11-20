//! The other half of the final-artifact contract for jemalloc metrics.
//!
//! This integration test is a separate executable that declares no
//! allocator, so Rust's default serves it while the library's jemalloc is
//! linked and unused. The gateway finds that out and publishes none of the
//! unused allocator's statistics.

#![cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use smg::observability::metrics::{start_prometheus, PrometheusConfig};

#[test]
fn a_binary_on_another_allocator_publishes_no_allocator_gauge() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let body = handle.render();

    let gauges: Vec<&str> = body
        .lines()
        .filter(|line| line.contains("smg_allocator_"))
        .collect();
    assert!(
        gauges.is_empty(),
        "an unused jemalloc published its statistics: {gauges:?}"
    );
}
