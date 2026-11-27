use metrics_exporter_prometheus::PrometheusHandle;

/// The recorder's rendered line of `metric` carrying `model`, if any.
pub(crate) fn series_line(handle: &PrometheusHandle, metric: &str, model: &str) -> Option<String> {
    let rendered = handle.render();
    let model = format!("model=\"{model}\"");
    rendered
        .lines()
        .find(|line| line.starts_with(&format!("{metric}{{")) && line.contains(&model))
        .map(str::to_string)
}

/// The error series the failed request is counted under for `model`, as
/// `(error_type, reason, count)`, with the number of duration samples.
pub(crate) fn counted_failure(
    handle: &PrometheusHandle,
    model: &str,
) -> (String, String, u64, u64) {
    let errors = series_line(handle, "smg_router_request_errors_total", model)
        .expect("the failed request is counted");
    let label = |name: &str| {
        let key = format!("{name}=\"");
        let start = errors.find(&key).expect(name) + key.len();
        let end = errors[start..].find('"').expect(name) + start;
        errors[start..end].to_string()
    };
    let count = |line: &str| -> u64 {
        line.rsplit(' ')
            .next()
            .expect("a value")
            .parse()
            .expect("a count")
    };
    let durations = series_line(handle, "smg_router_request_duration_seconds_count", model)
        .expect("the failed request has a duration");
    (
        label("error_type"),
        label("reason"),
        count(&errors),
        count(&durations),
    )
}
