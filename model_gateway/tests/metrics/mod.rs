//! Metrics integration tests that drive the recorder `start_prometheus` installs: one binary, the
//! recorder installed once per process (a second install panics), every test taking the handle from
//! `recorder()` and reading series of its own.

use std::sync::OnceLock;

use metrics_exporter_prometheus::PrometheusHandle;
use smg::observability::metrics::{start_prometheus, PrometheusConfig};

/// The process's recorder, installed on the first call and shared by every test of this binary:
/// `start_prometheus` installs the global recorder, the worker-series store and the metric
/// descriptions once, and panics on a second install. The configuration is the one every test
/// here used in a process of its own: an ephemeral port, the default histogram buckets.
pub fn recorder() -> PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            start_prometheus(PrometheusConfig {
                port: 0,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            })
        })
        .clone()
}

mod engine_metrics_test;
mod kv_event_lag_histogram_test;
mod metrics_help_text_test;
mod multimodal_series_test;
mod pushed_loads_test;
mod worker_request_counter_test;
mod worker_retry_backoff_histogram_test;
mod worker_series_retention_test;
mod worker_url_credentials_test;
