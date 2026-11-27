use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::Duration,
};

use axum::http::header::CONTENT_TYPE;
use openai_protocol::common::GenerationRequest;

/// Typed request carrying a drop probe; tests watch the `Arc` count to
/// observe exactly when a router frees the parsed body.
#[derive(serde::Serialize)]
pub(crate) struct DropProbeRequest {
    pub text: String,
    #[serde(skip)]
    pub _probe: Arc<()>,
}

impl GenerationRequest for DropProbeRequest {
    fn is_stream(&self) -> bool {
        false
    }

    fn get_model(&self) -> Option<&str> {
        None
    }

    fn extract_text_for_routing(&self) -> String {
        self.text.clone()
    }
}

/// Loopback POST /generate stub that answers `{}` only after every probe
/// clone outside the test is gone (or after five seconds, leaving
/// `released` false).
pub(crate) async fn spawn_release_gated_stub(probe: Weak<()>) -> (String, Arc<AtomicBool>) {
    spawn_release_gated_stub_with_deadline(probe, Duration::from_secs(5)).await
}

/// [`spawn_release_gated_stub`] with the stub's own answer `deadline` away
/// instead of five seconds: far out, the stub answers only once the test
/// lets go of the probe, so a verdict can wait for that answer under a
/// hang guard instead of a clock.
#[expect(
    clippy::disallowed_methods,
    reason = "test stub server lives for the duration of the test process"
)]
pub(crate) async fn spawn_release_gated_stub_with_deadline(
    probe: Weak<()>,
    deadline: Duration,
) -> (String, Arc<AtomicBool>) {
    let released = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&released);
    let app = axum::Router::new().route(
        "/generate",
        axum::routing::post(move || {
            let probe = probe.clone();
            let flag = Arc::clone(&flag);
            async move {
                let answer_by = tokio::time::Instant::now() + deadline;
                while probe.strong_count() > 1 && tokio::time::Instant::now() < answer_by {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                flag.store(probe.strong_count() <= 1, Ordering::SeqCst);
                ([(CONTENT_TYPE, "application/json")], "{}")
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), released)
}

/// Loopback POST /generate stub answering `{}` immediately.
#[expect(
    clippy::disallowed_methods,
    reason = "test stub server lives for the duration of the test process"
)]
pub(crate) async fn spawn_immediate_stub() -> String {
    let app = axum::Router::new().route(
        "/generate",
        axum::routing::post(|| async { ([(CONTENT_TYPE, "application/json")], "{}") }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}
