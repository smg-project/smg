//! Explicit version and control-state writes: local table updates, no engine call.

use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::rl::{
    RlSetControlRequest, RlSetVersionRequest, RlWorkerEntry, RlWorkersResponse, RL_PROTOCOL_VERSION,
};

use crate::{
    discovery::{dp_ranks, entry},
    error::RlError,
    fanout::{parse_selector, resolve_targets, FanoutQuery},
    state::RlState,
    table::{ControlState, VersionSource},
    version::Version,
    view::RlWorkerInfo,
};

/// The API write shares one validator with the passthrough observer, so a
/// version SMG would refuse here cannot slip in through a proxied refit body.
fn parse_version(raw: &str) -> Result<Version, RlError> {
    Version::validated(raw).map_err(|e| RlError::InvalidVersion(e.to_string()))
}

fn body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, RlError> {
    body.map(|Json(v)| v)
        .map_err(|e| RlError::InvalidBody(e.body_text()))
}

fn row(state: &RlState, worker: &RlWorkerInfo) -> RlWorkerEntry {
    entry(&state.table, worker, dp_ranks(state.view.as_ref(), worker))
}

fn rows(state: &RlState, workers: &[RlWorkerInfo]) -> Response {
    let workers: Vec<RlWorkerEntry> = workers.iter().map(|w| row(state, w)).collect();
    let total = workers.len();
    (
        StatusCode::OK,
        Json(RlWorkersResponse {
            protocol_version: RL_PROTOCOL_VERSION,
            workers,
            total,
        }),
    )
        .into_response()
}

fn targets(state: &RlState, selector: Option<String>) -> Result<Vec<RlWorkerInfo>, RlError> {
    let selector = parse_selector(selector)?;
    let targets = resolve_targets(state.view.as_ref(), &state.table, &selector);
    if targets.is_empty() {
        return Err(RlError::NoWorkersMatch(selector.source().to_string()));
    }
    Ok(targets)
}

pub(crate) async fn set_worker_version(
    State(state): State<Arc<RlState>>,
    Path(id): Path<String>,
    payload: Result<Json<RlSetVersionRequest>, JsonRejection>,
) -> Response {
    let version = match body(payload).and_then(|b| parse_version(&b.weight_version)) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let Some(worker) = state.view.get(&id) else {
        return RlError::WorkerNotFound(id).into_response();
    };
    state.apply_version(&worker, version, VersionSource::Api);
    (StatusCode::OK, Json(row(&state, &worker))).into_response()
}

pub(crate) async fn set_worker_state(
    State(state): State<Arc<RlState>>,
    Path(id): Path<String>,
    payload: Result<Json<RlSetControlRequest>, JsonRejection>,
) -> Response {
    let control: ControlState = match body(payload) {
        Ok(b) => b.control.into(),
        Err(e) => return e.into_response(),
    };
    let Some(worker) = state.view.get(&id) else {
        return RlError::WorkerNotFound(id).into_response();
    };
    state.apply_control(&worker, control);
    (StatusCode::OK, Json(row(&state, &worker))).into_response()
}

pub(crate) async fn set_fleet_version(
    State(state): State<Arc<RlState>>,
    Query(FanoutQuery { selector }): Query<FanoutQuery>,
    payload: Result<Json<RlSetVersionRequest>, JsonRejection>,
) -> Response {
    let version = match body(payload).and_then(|b| parse_version(&b.weight_version)) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let workers = match targets(&state, selector) {
        Ok(w) => w,
        Err(e) => return e.into_response(),
    };
    for worker in &workers {
        state.apply_version(worker, version.clone(), VersionSource::Api);
    }
    rows(&state, &workers)
}

pub(crate) async fn set_fleet_state(
    State(state): State<Arc<RlState>>,
    Query(FanoutQuery { selector }): Query<FanoutQuery>,
    payload: Result<Json<RlSetControlRequest>, JsonRejection>,
) -> Response {
    let control: ControlState = match body(payload) {
        Ok(b) => b.control.into(),
        Err(e) => return e.into_response(),
    };
    let workers = match targets(&state, selector) {
        Ok(w) => w,
        Err(e) => return e.into_response(),
    };
    for worker in &workers {
        state.apply_control(worker, control);
    }
    rows(&state, &workers)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
        response::Response,
    };
    use http_body_util::BodyExt;
    use openai_protocol::worker::RuntimeType;
    use serde_json::Value;
    use tower::ServiceExt;

    use crate::{
        config::RlConfig,
        state::RlState,
        testing::{worker, FakeView},
        view::RlWorkerInfo,
    };

    fn state(workers: Vec<RlWorkerInfo>) -> Arc<RlState> {
        Arc::new(RlState::new(
            Arc::new(FakeView(workers)),
            RlConfig::default(),
        ))
    }

    async fn json_body(resp: Response) -> Value {
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[tokio::test]
    async fn set_worker_version_updates_the_row() {
        let app = crate::router::<()>(state(vec![worker("w1", "http://a:1", RuntimeType::Sglang)]));
        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/version")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"weight_version": "42"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let row = json_body(resp).await;
        assert_eq!(row["id"], "w1");
        assert_eq!(row["weight_version"], "42");
        assert_eq!(row["version_source"], "api");
        let listed = json_body(
            app.oneshot(Request::get("/workers").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(listed["workers"][0]["weight_version"], "42");
    }

    #[tokio::test]
    async fn set_worker_state_and_errors() {
        let app = crate::router::<()>(state(vec![worker("w1", "http://a:1", RuntimeType::Sglang)]));
        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/state")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"control": "asleep"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["control"], "asleep");

        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/nope/state")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"control": "paused"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/state")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"control": "napping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["error"], "invalid_body");

        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/version")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"weight_version": "  "}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["error"], "invalid_version");

        let long = "x".repeat(129);
        let resp = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/version")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(r#"{{"weight_version": "{long}"}}"#)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert_eq!(body["error"], "invalid_version");
        assert_eq!(body["message"], "weight_version exceeds 128 bytes");

        // The shared validator also refuses anything outside printable ASCII,
        // so a version that could never be stamped into a header is refused at
        // the write rather than silently dropped at stamp time.
        let resp = app
            .oneshot(
                Request::post("/workers/w1/version")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"weight_version": "v1\u0007"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["error"], "invalid_version");
    }

    #[tokio::test]
    async fn fleet_forms_write_one_entry_per_engine_and_need_a_selector() {
        let app = crate::router::<()>(state(vec![
            worker("r0", "http://b:1@0", RuntimeType::Sglang),
            worker("r1", "http://b:1@1", RuntimeType::Sglang),
            worker("v1", "http://c:1", RuntimeType::Vllm),
        ]));
        let resp = app
            .clone()
            .oneshot(
                Request::post("/version?selector=engine%3Dsglang")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"weight_version": "3"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["total"], 1, "DP ranks collapse to one engine row");
        assert_eq!(body["workers"][0]["weight_version"], "3");
        assert_eq!(body["workers"][0]["dp_ranks"], 2);
        let listed = json_body(
            app.clone()
                .oneshot(Request::get("/workers/v1").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(listed["weight_version"], "default", "vllm worker untouched");

        let resp = app
            .clone()
            .oneshot(
                Request::post("/state")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"control": "paused"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["error"], "selector_required");

        let resp = app
            .oneshot(
                Request::post("/state?selector=engine%3Dtrtllm")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"control": "paused"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["error"], "no_workers_match");
    }
}
