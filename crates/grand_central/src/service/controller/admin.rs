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
    decision_slots::DecideStats,
    metrics::{LoadTotals, Metrics},
    readiness::Readiness,
    PROTOCOL_PACKAGE,
};
use crate::store::MemoryStore;

/// What `/version` answers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[non_exhaustive]
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
    name: "smg-grand-central",
    version: env!("CARGO_PKG_VERSION"),
    protocol: PROTOCOL_PACKAGE,
};

/// What the handlers read.
#[derive(Clone)]
pub(crate) struct Admin {
    pub(crate) readiness: Readiness,
    pub(crate) metrics: Arc<Metrics>,
    /// The reader of the load monitor's counts, read at each render.
    pub(crate) loads: Arc<dyn Fn() -> LoadTotals + Send + Sync>,
    /// The decide call's counts, rendered with the metrics.
    pub(crate) decides: Arc<DecideStats>,
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
            &(admin.loads)(),
            &admin.decides,
            || admin.store.stats(),
        ),
    )
        .into_response()
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/admin/tests.rs"]
mod tests;
