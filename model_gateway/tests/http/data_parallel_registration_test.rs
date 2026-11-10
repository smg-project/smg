//! A data-parallel HTTP engine registers as one worker per rank with no
//! configuration flag: the registration's metadata read takes the rank count
//! from the engine's `/server_info` and the registry holds one rank worker per
//! rank, each at the engine's address with its own rank; a single-rank engine
//! is one plain worker. The retired `--dp-aware` flag changes nothing either
//! way.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::worker::WorkerStatus;
use serde_json::json;
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};
use tower::ServiceExt;

use crate::common::{self, test_app::create_test_app_with_context};

const MODEL: &str = "data-parallel-http-registration-model";
/// How long the registration gets to put every rank's row into the registry.
const REGISTRATION_BOUND: Duration = Duration::from_secs(30);

/// An in-process mock HTTP engine advertising `ranks` data-parallel ranks in
/// its `/server_info`, as an engine does.
struct MockEngine {
    port: u16,
    stop: oneshot::Sender<()>,
    served: JoinHandle<()>,
}

impl MockEngine {
    #[expect(
        clippy::expect_used,
        reason = "test helper: a mock engine that does not bind fails the fixture"
    )]
    #[expect(
        clippy::disallowed_methods,
        reason = "test helper: the mock engine's serve task ends with its stop signal, \
                  which the test sends before it returns"
    )]
    async fn start(ranks: u16) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind the mock engine");
        let port = listener
            .local_addr()
            .expect("the mock engine's address")
            .port();
        let cfg = Arc::new(mock_worker::config::Config {
            http_dp_size: ranks,
            model_id: MODEL.to_string(),
            tokenizer_path: MODEL.to_string(),
            ..mock_worker::config::Config::default()
        });
        let (stop, stopped) = oneshot::channel();
        let served = tokio::spawn(mock_worker::http::serve_with_listener_until(
            cfg,
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        Self { port, stop, served }
    }

    async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.served.await;
    }
}

/// One row of the registry after the engine's registration: its rank, its
/// group size and the address the requests go to.
type Row = (Option<usize>, Option<usize>, String);

/// Registers an HTTP engine of `ranks` ranks through `POST /workers` on a
/// gateway configured with `dp_aware` as given (the retired flag), waits for
/// its rows to be routable and returns them sorted by rank.
#[expect(
    clippy::expect_used,
    reason = "test helper: a gateway or request that does not build fails the fixture"
)]
async fn registered_rows(ranks: u16, dp_aware: bool) -> Vec<Row> {
    let engine = MockEngine::start(ranks).await;
    let config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .dp_aware(dp_aware)
        .build_unchecked();
    let app_context = common::create_test_context(config).await;
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("HTTP router");
    let app = create_test_app_with_context(Arc::from(router), Arc::clone(&app_context));

    let body = json!({
        "url": format!("http://127.0.0.1:{}", engine.port),
        "connection_mode": "http",
        "runtime": "sglang",
        "models": [{"id": MODEL}],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/workers")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("a registration request");
    let response = app.oneshot(request).await.expect("a registration answer");
    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let registry = Arc::clone(&app_context.worker_registry);
    let want = usize::from(ranks);
    common::wait_until("the engine's rows are routable", REGISTRATION_BOUND, || {
        registry
            .get_all()
            .iter()
            .filter(|worker| worker.status() == WorkerStatus::Ready)
            .count()
            >= want
    })
    .await;
    let mut rows: Vec<Row> = registry
        .get_all()
        .iter()
        .map(|worker| {
            (
                worker.dp_rank(),
                worker.dp_size(),
                worker.base_url().to_string(),
            )
        })
        .collect();
    rows.sort();
    engine.stop().await;
    rows
}

/// Four ranks in `/server_info`, no flag: four rows, ranks 0 to 3 of a group
/// of four, all at the engine's address. With the retired flag set: the same
/// four.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_four_rank_http_engine_registers_as_four_rows_without_any_flag() {
    for dp_aware in [false, true] {
        let rows = registered_rows(4, dp_aware).await;
        let ranks: Vec<(Option<usize>, Option<usize>)> =
            rows.iter().map(|(rank, size, _)| (*rank, *size)).collect();
        let expected: Vec<(Option<usize>, Option<usize>)> =
            (0..4).map(|rank| (Some(rank), Some(4))).collect();
        assert_eq!(ranks, expected, "dp_aware {dp_aware}: {rows:?}");
        assert!(
            rows.iter().all(|(_, _, address)| *address == rows[0].2),
            "one address for every rank: {rows:?}"
        );
    }
}

/// One rank: one plain worker, with or without the retired flag.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_single_rank_http_engine_registers_as_one_row() {
    for dp_aware in [false, true] {
        let rows = registered_rows(1, dp_aware).await;
        assert_eq!(rows.len(), 1, "dp_aware {dp_aware}: {rows:?}");
        assert_eq!((rows[0].0, rows[0].1), (None, None), "{rows:?}");
    }
}
