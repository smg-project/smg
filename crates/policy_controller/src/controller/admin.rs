//! The HTTP admin listener: the probes, the version and the metrics. The
//! probes keep the gateway's paths and bodies (`/liveness`, `/readiness`,
//! `/health`; `OK`, `{"status": "ready"}`, `503 {"status": "not ready",
//! "reason": ...}`) and the scrape keeps its path (`/metrics`), so one probe
//! set and one scrape configuration serve both processes; `/version` names
//! the binary, its version and the protocol package it speaks.

use axum::{
    extract::State as AxumState,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::Serialize;
use serde_json::json;

use super::{readiness::Readiness, PROTOCOL_PACKAGE};

/// What `/version` answers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Version {
    /// The binary's name.
    pub name: &'static str,
    /// The crate's version.
    pub version: &'static str,
    /// The protobuf package of the API the listeners speak.
    pub protocol: &'static str,
}

/// This build's version.
pub const VERSION: Version = Version {
    name: "smg-policy-controller",
    version: env!("CARGO_PKG_VERSION"),
    protocol: PROTOCOL_PACKAGE,
};

/// What the handlers read.
#[derive(Clone)]
pub(crate) struct Admin {
    pub(crate) readiness: Readiness,
    pub(crate) metrics: PrometheusHandle,
}

/// The admin routes over their state.
pub(crate) fn router(admin: Admin) -> Router {
    Router::new()
        .route("/liveness", get(liveness))
        .route("/health", get(liveness))
        .route("/readiness", get(readiness))
        .route("/version", get(version))
        .route("/metrics", get(metrics))
        .with_state(admin)
}

/// Liveness stays `OK` while draining: the process is healthy, it is just
/// not taking new work.
async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(AxumState(admin): AxumState<Admin>) -> Response {
    let state = admin.readiness.state();
    if state.is_ready() {
        (StatusCode::OK, Json(json!({ "status": "ready" }))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "not ready", "reason": state.as_str() })),
        )
            .into_response()
    }
}

async fn version() -> Json<Version> {
    Json(VERSION)
}

async fn metrics(AxumState(admin): AxumState<Admin>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        admin.metrics.render(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    use super::{router, Admin, VERSION};
    use crate::controller::{metrics::Metrics, readiness, readiness::State};

    async fn get(app: axum::Router, path: &str) -> (StatusCode, String, String) {
        let response = app
            .oneshot(Request::get(path).body(Body::empty()).expect("a request"))
            .await
            .expect("a response");
        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let body = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("a body");
        (
            status,
            content_type,
            String::from_utf8_lossy(&body).into_owned(),
        )
    }

    #[tokio::test]
    async fn the_probes_follow_the_state_and_liveness_never_fails() {
        let (writer, reader) = readiness::channel();
        let metrics = Metrics::new();
        let app = router(Admin {
            readiness: reader,
            metrics: metrics.handle(),
        });

        let (status, _, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body, r#"{"status":"not ready","reason":"starting"}"#);

        writer.send_replace(State::Ready);
        let (status, content_type, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "application/json");
        assert_eq!(body, r#"{"status":"ready"}"#);

        writer.send_replace(State::Draining);
        let (status, _, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body, r#"{"status":"not ready","reason":"draining"}"#);

        for path in ["/liveness", "/health"] {
            let (status, _, body) = get(app.clone(), path).await;
            assert_eq!((status, body.as_str()), (StatusCode::OK, "OK"), "{path}");
        }
    }

    #[tokio::test]
    async fn version_and_metrics_render_the_controllers_own() {
        let (_writer, reader) = readiness::channel();
        let metrics = Metrics::new();
        let app = router(Admin {
            readiness: reader,
            metrics: metrics.handle(),
        });

        let (status, content_type, body) = get(app.clone(), "/version").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "application/json");
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(parsed["name"], VERSION.name);
        assert_eq!(parsed["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(parsed["protocol"], "smg.policy.v1");

        let (status, content_type, body) = get(app.clone(), "/metrics").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/plain; version=0.0.4; charset=utf-8");
        assert!(body.contains("smg_policy_controller_state{state=\"starting\"} 1"));
        assert!(body.contains("smg_policy_controller_start_time_seconds "));

        let (status, _, _) = get(app, "/nowhere").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
