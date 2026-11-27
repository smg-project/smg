use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use super::{router, Admin, VERSION};
use crate::{
    document::{AppliedSpec, PolicySpec},
    engine::view::Role,
    service::controller::{
        metrics::{LoadTotals, Metrics},
        readiness::{tests::fenced, Phase},
    },
    store::{MemoryStore, WorkerFacts},
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
        loads: Arc::new(LoadTotals::default),
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
    assert!(scrape.contains("smg_grand_central_state{state=\"follower\"} 1"));
    assert!(scrape.contains("smg_grand_central_state{state=\"leader\"} 0"));

    // The candidate, alone over its own lease, takes it and leads: the
    // probe and the gauge say leader with no message between the
    // election and the admin listener.
    elector.lead_at(Instant::now());
    let (status, _, body) = get(app.clone(), "/readiness").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"status":"ready","role":"leader"}"#);
    let (_, _, scrape) = get(app.clone(), "/metrics").await;
    assert!(scrape.contains("smg_grand_central_state{state=\"leader\"} 1"));
    assert!(scrape.contains("smg_grand_central_state{state=\"follower\"} 0"));

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
        loads: Arc::new(LoadTotals::default),
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
    assert!(body.contains("smg_grand_central_state{state=\"starting\"} 1"));
    assert!(body.contains("smg_grand_central_start_time_seconds "));
    assert!(body.contains("smg_grand_central_load_polls_total{mode=\"poll\"} 0"));

    let (status, _, _) = get(app, "/nowhere").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_scrape_reads_the_stores_counts_at_each_render() {
    let (_elector, _phase, _steps, reader) = fenced();
    let store = Arc::new(MemoryStore::new());
    // What the load monitor's reader answers for the flips: the case
    // moves it.
    let flips = Arc::new(AtomicU64::new(0));
    let read = Arc::clone(&flips);
    let app = router(Admin {
        readiness: reader,
        metrics: Arc::new(Metrics::new()),
        loads: Arc::new(move || LoadTotals {
            overload_flips: read.load(Ordering::Relaxed),
            ..LoadTotals::default()
        }),
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
    store.publish_now();
    let (_, _, body) = get(app.clone(), "/metrics").await;
    assert!(body.contains("smg_grand_central_members_outside_list{list=\"engines\"} 0"));
    assert!(body.contains("smg_grand_central_members_outside_list{list=\"zones\"} 0"));
    assert!(body.contains("smg_grand_central_overload_flips_total 0"));

    // A member the lists do not name is counted at the scrape after the
    // build its registration owes.
    store.upsert(member("stray", "beta", "west"));
    store.upsert(member("far", "alpha", "north"));
    store.publish_now();
    let (_, _, body) = get(app.clone(), "/metrics").await;
    assert!(body.contains("smg_grand_central_members_outside_list{list=\"engines\"} 1"));
    assert!(body.contains("smg_grand_central_members_outside_list{list=\"zones\"} 2"));
    assert!(body.contains("smg_grand_central_overload_flips_total 0"));

    // The flips are the load monitor's, read at each scrape.
    flips.store(3, Ordering::Relaxed);
    let (_, _, body) = get(app, "/metrics").await;
    assert!(body.contains("smg_grand_central_overload_flips_total 3"));
}
