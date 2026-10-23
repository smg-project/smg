//! The weight version each successful refit relayed through the control plane
//! stamped. A worker's labels carry the version its engine reported at
//! registration and are never refreshed, so discovery and selectors report
//! this one over it.

use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, PoisonError},
};

use openai_protocol::rl::RlCallOutcome;
use serde_json::Value;

use crate::{proxy::ProxyRequest, view::RlWorkerInfo};

const WEIGHT_VERSION: &str = "weight_version";

/// The version a successful call to the engine route `path` with `body`
/// stamps, if it names one.
pub(crate) fn stamped_version(path: &str, body: &[u8]) -> Option<String> {
    let field = match path {
        "update_weights_from_disk"
        | "update_weights_from_distributed"
        | "update_weights_from_tensor" => WEIGHT_VERSION,
        "update_weight_version" => "new_version",
        _ => return None,
    };
    let body: Value = serde_json::from_slice(body).ok()?;
    match body.get(field)? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// The version `outcome` stamped: the call succeeded (an engine can answer
/// 200 with `success: false`) and `req` named one.
pub(crate) fn stamped_by(req: &ProxyRequest, outcome: &RlCallOutcome) -> Option<String> {
    let refused = outcome.body.get("success") == Some(&Value::Bool(false));
    if !outcome.is_success() || refused {
        return None;
    }
    stamped_version(&req.path, &req.body)
}

/// Whether `req` names a version a successful call would stamp, so the caller
/// takes the registry snapshot to record against before it calls.
pub(crate) fn names_a_version(req: &ProxyRequest) -> bool {
    stamped_version(&req.path, &req.body).is_some()
}

// Keyed by worker id: a worker that registers again gets a new id and reports
// its own metadata. Each version holds while the worker's `downs` and
// `revision` stay what they were before the refit was sent: an engine that went
// down may have restarted on other weights, and a same-URL replacement brings
// its own spec.
#[derive(Default)]
pub(crate) struct ReportedVersions(Mutex<HashMap<String, (String, u64, u64)>>);

impl ReportedVersions {
    /// Record each called worker's stamped version for every rank of its engine
    /// in `registered`, the snapshot taken before the calls, and drop entries
    /// of workers not in it.
    pub(crate) fn record(&self, registered: &[RlWorkerInfo], stamped: &[(&RlWorkerInfo, String)]) {
        let by_engine: HashMap<&str, &str> = stamped
            .iter()
            .map(|(w, v)| (w.base_url.as_str(), v.as_str()))
            .collect();
        let ids: HashSet<&str> = registered.iter().map(|w| w.id.as_str()).collect();
        let mut versions = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        versions.retain(|id, _| ids.contains(id.as_str()));
        for w in registered {
            if let Some(v) = by_engine.get(w.base_url.as_str()) {
                versions.insert(w.id.clone(), ((*v).to_string(), w.downs, w.revision));
            }
        }
    }

    pub(crate) fn apply(&self, mut workers: Vec<RlWorkerInfo>) -> Vec<RlWorkerInfo> {
        let versions = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        for w in &mut workers {
            let current = |(_, downs, revision): &&(String, u64, u64)| {
                *downs == w.downs && *revision == w.revision
            };
            if let Some((v, _, _)) = versions.get(&w.id).filter(current) {
                w.labels.insert(WEIGHT_VERSION.to_string(), v.clone());
            }
        }
        workers
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{
        body::Body,
        http::{Request, StatusCode},
        Router,
    };
    use http_body_util::BodyExt;
    use openai_protocol::worker::RuntimeType;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use crate::{
        config::RlConfig,
        state::RlState,
        testing::{worker, FakeEngine},
        view::{RlWorkerInfo, RlWorkerView},
    };

    struct SwapView(Mutex<Vec<RlWorkerInfo>>);

    impl SwapView {
        fn new(workers: Vec<RlWorkerInfo>) -> Arc<Self> {
            Arc::new(Self(Mutex::new(workers)))
        }
        fn replace(&self, workers: Vec<RlWorkerInfo>) {
            *self.0.lock().unwrap() = workers;
        }
    }

    impl RlWorkerView for SwapView {
        fn list(&self) -> Vec<RlWorkerInfo> {
            self.0.lock().unwrap().clone()
        }
        fn get(&self, id: &str) -> Option<RlWorkerInfo> {
            self.0.lock().unwrap().iter().find(|w| w.id == id).cloned()
        }
    }

    fn app(view: Arc<SwapView>) -> Router {
        let cfg = RlConfig {
            enabled: true,
            control_timeout_secs: 5,
            fanout_concurrency: 8,
        };
        crate::router::<()>(Arc::new(RlState::new(view, cfg)))
    }

    async fn send(
        app: &Router,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn reported(app: &Router, id: &str) -> Value {
        send(app, "GET", &format!("/workers/{id}"), None).await.1["weight_version"].clone()
    }

    #[test]
    fn a_worker_no_longer_registered_is_not_recorded() {
        let gone = worker("w1", "http://a:1", RuntimeType::Sglang);
        let still = worker("w2", "http://b:1", RuntimeType::Sglang);
        let versions = super::ReportedVersions::default();

        versions.record(std::slice::from_ref(&still), &[(&gone, "4".to_string())]);

        assert_eq!(
            versions.apply(vec![gone])[0].labels["weight_version"],
            "default"
        );
    }

    #[test]
    fn a_worker_that_went_down_since_the_refit_reports_its_registration_metadata() {
        let w = worker("w1", "http://a:1", RuntimeType::Sglang);
        let versions = super::ReportedVersions::default();
        versions.record(std::slice::from_ref(&w), &[(&w, "7".to_string())]);
        let mut back = w.clone();
        back.downs = 1;

        assert_eq!(versions.apply(vec![w])[0].labels["weight_version"], "7");
        assert_eq!(
            versions.apply(vec![back])[0].labels["weight_version"],
            "default"
        );
    }

    #[test]
    fn a_worker_replaced_since_the_refit_reports_its_new_metadata() {
        let w = worker("w1", "http://a:1", RuntimeType::Sglang);
        let versions = super::ReportedVersions::default();
        versions.record(std::slice::from_ref(&w), &[(&w, "7".to_string())]);
        let mut replaced = w.clone();
        replaced.revision = 1;

        assert_eq!(
            versions.apply(vec![replaced])[0].labels["weight_version"],
            "default"
        );
    }

    #[tokio::test]
    async fn an_engine_that_goes_down_during_its_refit_does_not_report_it() {
        let e = FakeEngine::start(StatusCode::OK, json!({"success": true}), 300).await;
        let view = SwapView::new(vec![worker("w1", &e.url, RuntimeType::Sglang)]);
        let app = app(Arc::clone(&view));
        let mut down = worker("w1", &e.url, RuntimeType::Sglang);
        down.downs = 1;

        let refit = send(
            &app,
            "POST",
            "/workers/w1/engine/update_weights_from_disk",
            Some(json!({"model_path": "/m", "weight_version": "7"})),
        );
        let restart = async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            view.replace(vec![down]);
        };
        let ((status, _), ()) = tokio::join!(refit, restart);

        assert_eq!(status, StatusCode::OK);
        assert_eq!(reported(&app, "w1").await, "default");
    }

    #[tokio::test]
    async fn a_refit_through_the_fanout_is_what_discovery_reports() {
        let e1 = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let e2 = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let app = app(SwapView::new(vec![
            worker("w1", &e1.url, RuntimeType::Sglang),
            worker("w2", &e2.url, RuntimeType::Sglang),
        ]));

        let body = json!({"names": ["w"], "flush_cache": true, "weight_version": "1"});
        let (status, fan) = send(
            &app,
            "POST",
            "/engine/update_weights_from_distributed?selector=engine%3Dsglang",
            Some(body),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{fan}");

        let (_, list) = send(&app, "GET", "/workers", None).await;
        for w in list["workers"].as_array().unwrap() {
            assert_eq!(w["weight_version"], "1", "{w}");
        }
        assert_eq!(reported(&app, "w1").await, "1");

        let (_, fan) = send(
            &app,
            "POST",
            "/engine/flush_cache?selector=weight_version%3D1",
            None,
        )
        .await;
        assert_eq!(
            fan["total"], 2,
            "a selector on the new version matches: {fan}"
        );
    }

    #[tokio::test]
    async fn update_weight_version_through_the_proxy_stamps_its_new_version() {
        let e = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let app = app(SwapView::new(vec![worker(
            "w1",
            &e.url,
            RuntimeType::Sglang,
        )]));

        let (status, _) = send(
            &app,
            "POST",
            "/workers/w1/engine/update_weight_version",
            Some(json!({"new_version": "7"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reported(&app, "w1").await, "7");
    }

    #[tokio::test]
    async fn a_call_that_stamps_nothing_keeps_the_reported_version() {
        let ok = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let refused =
            FakeEngine::start(StatusCode::OK, json!({"success": false, "message": "x"}), 0).await;
        let failed = FakeEngine::start(StatusCode::INTERNAL_SERVER_ERROR, json!({}), 0).await;
        let app = app(SwapView::new(vec![
            worker("ok", &ok.url, RuntimeType::Sglang),
            worker("refused", &refused.url, RuntimeType::Sglang),
            worker("failed", &failed.url, RuntimeType::Sglang),
        ]));

        // A chunk before the last carries no version.
        send(
            &app,
            "POST",
            "/workers/ok/engine/update_weights_from_distributed",
            Some(json!({"names": ["w"]})),
        )
        .await;
        // A version on a route that does not stamp one.
        send(
            &app,
            "POST",
            "/workers/ok/engine/flush_cache",
            Some(json!({"weight_version": "9"})),
        )
        .await;
        for id in ["refused", "failed"] {
            send(
                &app,
                "POST",
                &format!("/workers/{id}/engine/update_weights_from_disk"),
                Some(json!({"model_path": "/m", "weight_version": "2"})),
            )
            .await;
        }

        for id in ["ok", "refused", "failed"] {
            assert_eq!(reported(&app, id).await, "default", "{id}");
        }
    }

    #[tokio::test]
    async fn every_rank_of_the_refitted_engine_reports_the_version() {
        let e = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let app = app(SwapView::new(vec![
            worker("r0", &format!("{}@0", e.url), RuntimeType::Sglang),
            worker("r1", &format!("{}@1", e.url), RuntimeType::Sglang),
        ]));

        let (status, fan) = send(
            &app,
            "POST",
            "/engine/update_weights_from_disk?selector=engine%3Dsglang",
            Some(json!({"model_path": "/m", "weight_version": "3"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{fan}");
        assert_eq!(e.seen().len(), 1, "the ranks share one engine call");
        assert_eq!(reported(&app, "r0").await, "3");
        assert_eq!(reported(&app, "r1").await, "3");
    }

    #[tokio::test]
    async fn a_worker_that_registers_again_reports_its_own_metadata() {
        let e = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let view = SwapView::new(vec![worker("w1", &e.url, RuntimeType::Sglang)]);
        let app = app(Arc::clone(&view));

        send(
            &app,
            "POST",
            "/workers/w1/engine/update_weights_from_disk",
            Some(json!({"model_path": "/m", "weight_version": "5"})),
        )
        .await;
        assert_eq!(reported(&app, "w1").await, "5");

        view.replace(vec![worker("w9", &e.url, RuntimeType::Sglang)]);
        assert_eq!(reported(&app, "w9").await, "default");
    }
}
