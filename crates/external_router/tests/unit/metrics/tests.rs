use std::collections::HashMap;

use metrics_exporter_prometheus::PrometheusBuilder;

use super::*;

/// The scrape of a recorder of its own after `records`, as series lines
/// (no `# HELP` or `# TYPE`), sorted.
fn series_after(records: &dyn Fn()) -> Vec<String> {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, records);
    let mut series: Vec<String> = handle
        .render()
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(str::to_owned)
        .collect();
    series.sort_unstable();
    series
}

/// The request counter and the duration histogram recorded through
/// kept handles render exactly as with a key built per record, and one
/// handle is kept per label set and model.
#[test]
fn router_records_render_the_same_through_kept_handles() {
    let records = || {
        for streaming in [STREAMING_TRUE, STREAMING_FALSE, STREAMING_TRUE] {
            Metrics::record_router_request(
                metrics_labels::ROUTER_GRPC,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_GRPC,
                "m1",
                metrics_labels::ENDPOINT_CHAT,
                streaming,
            );
        }
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            "m2",
            metrics_labels::ENDPOINT_GENERATE,
            STREAMING_FALSE,
        );
        for millis in [5, 7] {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_GRPC,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_GRPC,
                "m1",
                metrics_labels::ENDPOINT_CHAT,
                Duration::from_millis(millis),
            );
        }
    };
    let built = series_after(&records);

    handles::force_on_this_thread(Some(true));
    let kept = series_after(&records);
    let kept_counters = handles::ROUTER_REQUESTS
        .with_borrow(|by_labels| by_labels.values().map(HashMap::len).sum::<usize>());
    let kept_histograms = handles::ROUTER_DURATIONS
        .with_borrow(|by_labels| by_labels.values().map(HashMap::len).sum::<usize>());
    handles::force_on_this_thread(None);

    assert_eq!(kept, built);
    assert_eq!(kept_counters, 3, "a counter per label set and model");
    assert_eq!(kept_histograms, 1, "a histogram per label set and model");
    for expected in [
        "smg_router_requests_total{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\",streaming=\"true\"} 2",
        "smg_router_requests_total{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\",streaming=\"false\"} 1",
        "smg_router_requests_total{router_type=\"http\",backend_type=\"regular\",connection_mode=\"http\",model=\"m2\",endpoint=\"generate\",streaming=\"false\"} 1",
        "smg_router_request_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\"} 2",
    ] {
        assert!(
            kept.iter().any(|line| line == expected),
            "missing {expected} in:\n{}",
            kept.join("\n")
        );
    }
}

/// A router error known by its status alone reads as the worker's: the
/// status classes an operator asks about first (a deadline, a dropped
/// connection, capacity pushback, the worker's own failure) each get
/// their reason, and a status outside them is `other`.
#[test]
fn a_status_alone_names_the_reason_of_the_workers_answer() {
    use metrics_labels::*;
    for (status, reason) in [
        (StatusCode::BAD_REQUEST, REASON_INVALID_REQUEST),
        (StatusCode::NOT_FOUND, REASON_MODEL_NOT_FOUND),
        (StatusCode::REQUEST_TIMEOUT, REASON_DEADLINE),
        (StatusCode::GATEWAY_TIMEOUT, REASON_DEADLINE),
        (StatusCode::TOO_MANY_REQUESTS, REASON_OVERLOADED),
        (StatusCode::INTERNAL_SERVER_ERROR, REASON_ENGINE_INTERNAL),
        (StatusCode::BAD_GATEWAY, REASON_TRANSPORT_DISCONNECT),
        (StatusCode::SERVICE_UNAVAILABLE, REASON_ENGINE_UNAVAILABLE),
        (StatusCode::UNAUTHORIZED, REASON_OTHER),
        (StatusCode::CONFLICT, REASON_OTHER),
        (StatusCode::NOT_IMPLEMENTED, REASON_OTHER),
    ] {
        assert_eq!(error_reason_from_status(status), reason, "{status}");
    }
    // The pool's two refusals of a request it could not place.
    assert_eq!(
        error_reason_for_refused_selection(StatusCode::NOT_FOUND),
        REASON_MODEL_NOT_FOUND
    );
    assert_eq!(
        error_reason_for_refused_selection(StatusCode::SERVICE_UNAVAILABLE),
        REASON_NO_AVAILABLE_WORKERS
    );
}

/// Every sample of the router error family carries its `reason` next to
/// its `error_type`, so a sum over `reason` is the count the family had
/// before the label existed, and two causes under one class are two
/// series.
#[test]
fn a_router_error_is_counted_under_its_class_and_its_reason() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        for reason in [
            metrics_labels::REASON_TRANSPORT_DISCONNECT,
            metrics_labels::REASON_ENGINE_INTERNAL,
            metrics_labels::REASON_ENGINE_INTERNAL,
        ] {
            Metrics::record_router_error(
                metrics_labels::ROUTER_GRPC,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_GRPC,
                "reason-test",
                metrics_labels::ENDPOINT_CHAT,
                metrics_labels::ERROR_BACKEND,
                reason,
            );
        }
    });
    let rendered = handle.render();
    let samples: Vec<&str> = rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_router_request_errors_total{")
                && line.contains(r#"model="reason-test""#)
        })
        .collect();
    assert_eq!(samples.len(), 2, "{rendered}");
    let value = |reason: &str| -> u64 {
        let line = samples
            .iter()
            .find(|line| line.contains(&format!(r#"reason="{reason}""#)))
            .unwrap_or_else(|| panic!("a {reason} series:\n{rendered}"));
        assert!(line.contains(r#"error_type="backend_error""#), "{line}");
        line.rsplit(' ').next().unwrap().parse().unwrap()
    };
    assert_eq!(value("transport_disconnect"), 1);
    assert_eq!(value("engine_internal"), 2);
}
