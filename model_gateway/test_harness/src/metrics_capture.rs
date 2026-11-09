//! The metrics one test case emits, rendered: a Prometheus recorder local to
//! the case, around a current-thread runtime of its own.

use std::future::Future;

use metrics_exporter_prometheus::PrometheusBuilder;

/// Run `case` on a current-thread runtime under a recorder local to the call and
/// return the rendered `/metrics` text.
#[expect(clippy::expect_used, reason = "test helper")]
pub fn render_after<F, Fut>(case: F) -> String
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        rt.block_on(case());
    });
    handle.render()
}

/// The `_sum` or `_count` sample of the histogram `family` whose label set
/// holds every fragment of `labels` (`endpoint="chat"`, say), parsed; None
/// when the family was not emitted with those labels.
pub fn sample(rendered: &str, family: &str, field: &str, labels: &[&str]) -> Option<f64> {
    let prefix = format!("{family}_{field}{{");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && labels.iter().all(|label| line.contains(label)))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}
