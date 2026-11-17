//! The HTTP admin listener: the probes, the version and the metrics. The
//! probes keep the gateway's paths and bodies (`/liveness`, `/readiness`,
//! `/health`; `OK`, `{"status": "ready", ...}`, `503 {"status": "not ready",
//! "reason": ...}`) and the scrape keeps its path (`/metrics`), so one probe
//! set and one scrape configuration serve both processes; a ready controller
//! adds its role (`"role": "leader"` or `"follower"`), read from the fence at
//! the probe; `/version` names the binary, its version and the protocol
//! package it speaks.

use std::sync::Arc;

use axum::{
    extract::State as AxumState,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use serde::Serialize;
use serde_json::json;

use super::{
    deciders::Counts,
    metrics::{Metrics, StoreCounts},
    readiness::Readiness,
    PROTOCOL_PACKAGE,
};
use crate::{ingest::loads::Stats, store::MemoryStore};

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
    pub(crate) metrics: Arc<Metrics>,
    /// The load driver's counts, rendered with the metrics.
    pub(crate) loads: Arc<Stats>,
    /// The decide call's counts, rendered with the metrics.
    pub(crate) decides: Arc<Counts>,
    /// The store, whose counts the metrics render reads.
    pub(crate) store: Arc<MemoryStore>,
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
        (
            StatusCode::OK,
            Json(json!({ "status": "ready", "role": state.as_str() })),
        )
            .into_response()
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
        admin.metrics.render(
            || admin.readiness.sample(),
            &admin.loads,
            &admin.decides,
            || StoreCounts::of(&admin.store),
        ),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    use super::{router, Admin, VERSION};
    use crate::{
        controller::{
            metrics::Metrics,
            readiness::{tests::fenced, Phase},
        },
        ingest::loads::Stats,
        store::{MemoryStore, WorkerFacts, WorkersStore},
        view::Role,
        AppliedSpec, PolicySpec,
    };

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
        let (mut elector, phase, _steps, reader) = fenced();
        let metrics = Metrics::new();
        let app = router(Admin {
            readiness: reader,
            metrics: Arc::new(metrics),
            loads: Arc::new(Stats::default()),
            decides: Arc::default(),
            store: Arc::new(MemoryStore::new()),
        });

        let (status, _, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body, r#"{"status":"not ready","reason":"starting"}"#);

        // Serving behind a closed fence: ready, and a follower.
        phase.send_replace(Phase::Serving);
        let (status, content_type, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "application/json");
        assert_eq!(body, r#"{"status":"ready","role":"follower"}"#);
        let (_, _, scrape) = get(app.clone(), "/metrics").await;
        assert!(scrape.contains("smg_policy_controller_state{state=\"follower\"} 1"));
        assert!(scrape.contains("smg_policy_controller_state{state=\"leader\"} 0"));

        // The candidate, alone over its own lease, takes it and leads: the
        // probe and the gauge say leader with no message between the
        // election and the admin listener.
        elector.lead_at(Instant::now());
        let (status, _, body) = get(app.clone(), "/readiness").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, r#"{"status":"ready","role":"leader"}"#);
        let (_, _, scrape) = get(app.clone(), "/metrics").await;
        assert!(scrape.contains("smg_policy_controller_state{state=\"leader\"} 1"));
        assert!(scrape.contains("smg_policy_controller_state{state=\"follower\"} 0"));

        phase.send_replace(Phase::Draining);
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
        let (_elector, _phase, _steps, reader) = fenced();
        let metrics = Metrics::new();
        let app = router(Admin {
            readiness: reader,
            metrics: Arc::new(metrics),
            loads: Arc::new(Stats::default()),
            decides: Arc::default(),
            store: Arc::new(MemoryStore::new()),
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
        assert!(body.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 0"));

        let (status, _, _) = get(app, "/nowhere").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn the_scrape_reads_the_stores_counts_at_each_render() {
        let (_elector, _phase, _steps, reader) = fenced();
        let store = Arc::new(MemoryStore::new());
        let app = router(Admin {
            readiness: reader,
            metrics: Arc::new(Metrics::new()),
            loads: Arc::new(Stats::default()),
            decides: Arc::default(),
            store: Arc::clone(&store),
        });
        let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
        document.generation = 1;
        document.view.engines = vec!["alpha".to_owned()];
        document.view.zones = vec!["east".to_owned()];
        let lists = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(lists).expect("applied");
        let member = |key: &str, engine: &str, zone: &str| WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation: 1,
            role: Role::Single,
            models: ["m".to_owned()].into_iter().collect(),
            dp_size: 1,
            engine_kind: engine.to_owned(),
            zone: Some(zone.to_owned()),
            ..WorkerFacts::default()
        };

        store.upsert(member("listed", "alpha", "east"));
        let (_, _, body) = get(app.clone(), "/metrics").await;
        assert!(body.contains("smg_policy_controller_members_outside_list{list=\"engines\"} 0"));
        assert!(body.contains("smg_policy_controller_members_outside_list{list=\"zones\"} 0"));
        assert!(body.contains("smg_policy_controller_overload_flips_total 0"));

        // A member the lists do not name is counted at the next scrape.
        store.upsert(member("stray", "beta", "west"));
        store.upsert(member("far", "alpha", "north"));
        let (_, _, body) = get(app, "/metrics").await;
        assert!(body.contains("smg_policy_controller_members_outside_list{list=\"engines\"} 1"));
        assert!(body.contains("smg_policy_controller_members_outside_list{list=\"zones\"} 2"));
    }
}
