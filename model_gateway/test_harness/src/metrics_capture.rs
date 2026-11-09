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

#[cfg(test)]
mod tests {
    use super::sample;

    /// Two series of one family, apart by their endpoint label.
    const RENDERED: &str = "# TYPE smg_router_ttft_seconds histogram\n\
        smg_router_ttft_seconds_sum{backend_type=\"regular\",endpoint=\"chat\"} 0.5\n\
        smg_router_ttft_seconds_count{backend_type=\"regular\",endpoint=\"chat\"} 1\n\
        smg_router_ttft_seconds_sum{backend_type=\"regular\",endpoint=\"completions\"} 2.5\n\
        smg_router_ttft_seconds_count{backend_type=\"regular\",endpoint=\"completions\"} 2\n";

    #[test]
    fn a_sample_is_read_from_the_series_carrying_every_label() {
        let family = "smg_router_ttft_seconds";
        assert_eq!(
            sample(RENDERED, family, "sum", &["endpoint=\"completions\""]),
            Some(2.5)
        );
        assert_eq!(
            sample(
                RENDERED,
                family,
                "count",
                &["backend_type=\"regular\"", "endpoint=\"chat\""]
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(RENDERED, family, "sum", &["endpoint=\"generate\""]),
            None
        );
        assert_eq!(
            sample(
                RENDERED,
                family,
                "sum",
                &["backend_type=\"pd\"", "endpoint=\"chat\""]
            ),
            None
        );
    }
}
