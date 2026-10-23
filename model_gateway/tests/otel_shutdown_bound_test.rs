//! The tracer's shutdown waits no longer than its bound. A collector that
//! accepts the connection and never answers holds the export; the shutdown
//! returns at `OTEL_SHUTDOWN_TIMEOUT`, the tracer off, so a stop or a
//! refused start-up with such a collector costs the bound once and no more.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    net::TcpListener,
    time::{Duration, Instant},
};

use smg::observability::otel_trace::{self, OTEL_SHUTDOWN_TIMEOUT};
use tracing::info_span;
use tracing_subscriber::prelude::*;

/// The tracer is process-wide and initialised once: this binary runs the one
/// shutdown that needs it.
#[test]
fn the_shutdown_wait_is_bounded_when_the_collector_never_answers() {
    // A listener nobody accepts from: the handshake completes in the kernel
    // and nothing ever answers the export.
    let unanswering = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let endpoint = format!("http://{}", unanswering.local_addr().unwrap());

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime
        .block_on(async { otel_trace::otel_tracing_init(true, Some(&endpoint)) })
        .expect("the tracer initialises against a bound listener");

    // One span queued, so the shutdown has an export to wait for.
    let layer = otel_trace::get_otel_layer().expect("the layer follows the tracer");
    tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), || {
        info_span!(target: "smg::otel-trace", "unanswered").in_scope(|| {});
    });

    let started = Instant::now();
    otel_trace::shutdown_otel();
    let waited = started.elapsed();

    assert!(
        waited < OTEL_SHUTDOWN_TIMEOUT + Duration::from_secs(2),
        "the shutdown returned after {waited:?}, past the bound of {OTEL_SHUTDOWN_TIMEOUT:?}"
    );
    assert!(
        waited >= OTEL_SHUTDOWN_TIMEOUT - Duration::from_secs(1),
        "the shutdown returned after {waited:?}: the collector answered, or the export failed \
         at once, and the bound was not what ended the wait"
    );
    assert!(
        !otel_trace::is_otel_enabled(),
        "the tracer is off once the bound has passed"
    );
    drop(unanswering);
}
