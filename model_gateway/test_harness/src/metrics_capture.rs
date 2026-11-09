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
