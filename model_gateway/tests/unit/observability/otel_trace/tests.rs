use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use axum::http::{HeaderMap, HeaderValue};
use tracing::{Level, Subscriber};
use tracing_subscriber::{filter::LevelFilter, layer::SubscriberExt, Layer};

use super::{is_otel_enabled, with_trace_context_http, CustomOtelFilter};

/// While tracing is off (no test of this binary turns it on) nothing is
/// written into the headers, so the caller's map is lent, not copied.
#[test]
fn the_callers_headers_are_lent_while_tracing_is_off() {
    assert!(!is_otel_enabled());
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", HeaderValue::from_static("r-1"));

    let lent = with_trace_context_http(Some(&headers));
    assert!(matches!(lent, Cow::Borrowed(_)));
    assert_eq!(*lent, headers);
    assert!(with_trace_context_http(None).is_empty());
}

/// The trace layer's level is the more verbose of INFO and the log level: the spans export
/// at `--log-level warn` and `error`, and a finer log level reaches the trace targets' events.
#[test]
fn the_trace_filter_sits_at_info_or_finer_with_the_log_level() {
    for (log_level, expected) in [
        (LevelFilter::ERROR, LevelFilter::INFO),
        (LevelFilter::WARN, LevelFilter::INFO),
        (LevelFilter::INFO, LevelFilter::INFO),
        (LevelFilter::DEBUG, LevelFilter::DEBUG),
        (LevelFilter::TRACE, LevelFilter::TRACE),
    ] {
        assert_eq!(
            CustomOtelFilter::with_log_level(log_level).level,
            expected,
            "log level {log_level}"
        );
    }
}

/// A layer that records the level of every event it is handed.
struct Recorder(Arc<Mutex<Vec<Level>>>);

impl<S: Subscriber> Layer<S> for Recorder {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.0.lock().unwrap().push(*event.metadata().level());
    }
}

/// Behind the filter at `--log-level warn`, an INFO event of a trace target reaches the
/// layer and a DEBUG one does not; at `--log-level debug` the DEBUG one does too.
#[test]
fn the_filter_admits_the_trace_targets_at_its_own_level() {
    let seen = |log_level: LevelFilter| {
        let levels = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(
            Recorder(Arc::clone(&levels)).with_filter(CustomOtelFilter::with_log_level(log_level)),
        );
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "smg::otel-trace", "an info event of a trace target");
            tracing::debug!(target: "smg::otel-trace", "a debug event of a trace target");
            tracing::info!(target: "smg::elsewhere", "an info event of another target");
        });
        let levels = levels.lock().unwrap().clone();
        levels
    };
    assert_eq!(seen(LevelFilter::WARN), vec![Level::INFO]);
    assert_eq!(seen(LevelFilter::DEBUG), vec![Level::INFO, Level::DEBUG]);
}
