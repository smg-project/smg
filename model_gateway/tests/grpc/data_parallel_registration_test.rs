//! A data-parallel gRPC engine registers as one worker per rank with no
//! configuration flag: the registration workflow reads the rank count the
//! engine advertises and the registry holds one rank worker per rank, each at
//! the engine's address with its own rank; a single-rank engine is one plain
//! worker. The retired `--dp-aware` flag changes nothing either way.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{CONTENT_TYPE, LOCATION},
        StatusCode,
    },
};
use openai_protocol::worker::WorkerStatus;
use serde_json::json;
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};
use tower::ServiceExt;

use crate::common::{self, grpc_harness::free_port, test_app::create_test_app_with_context};

const MODEL: &str = "data-parallel-registration-model";
/// How long the registration gets to put every rank's row into the registry.
const REGISTRATION_BOUND: Duration = Duration::from_secs(30);

/// An in-process mock engine of `ranks` data-parallel ranks on `port`, each
/// rank a realistic engine, advertising its rank count as a servicer does.
struct MockEngine {
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
    async fn start(port: u16, ranks: u16) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", port))
            .await
            .expect("bind the mock engine");
        let cfg = Arc::new(mock_worker::config::Config {
            admin_port: None,
            context_length: 32768,
            host: "127.0.0.1".to_string(),
            http_base_port: 0,
            http_count: 0,
            grpc_base_port: port,
            grpc_count: 1,
            grpc_dp_size: ranks,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: MODEL.to_string(),
            tokenizer_path: MODEL.to_string(),
            gen_delay: Duration::ZERO,
            output_tokens: 3,
            realistic: true,
            engine: mock_worker::engine::EngineParams::default(),
            ..mock_worker::config::Config::default()
        });
        let (stop, stopped) = oneshot::channel();
        let served = tokio::spawn(mock_worker::grpc::serve_with_listener_until(
            cfg,
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        Self { stop, served }
    }

    async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.served.await;
    }
}

/// One row of the registry after the engine's registration: its rank, its
/// group size and the address the requests go to.
type Row = (Option<usize>, Option<usize>, String);

/// Registers an engine of `ranks` ranks through `POST /workers` on a gateway
/// configured with `dp_aware` as given (the retired flag), waits for its
/// rows to be routable, follows the POST's Location to the registered
/// engine and returns the rows sorted by rank.
#[expect(
    clippy::expect_used,
    reason = "test helper: a gateway or request that does not build fails the fixture"
)]
async fn registered_rows(ranks: u16, dp_aware: bool) -> Vec<Row> {
    let port = free_port().await;
    let engine = MockEngine::start(port, ranks).await;
    let config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .dp_aware(dp_aware)
        .build_unchecked();
    let app_context = common::create_test_context_with_mock_tokenizer(config, MODEL).await;
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router");
    let app = create_test_app_with_context(Arc::from(router), Arc::clone(&app_context));

    let body = json!({
        "url": format!("grpc://127.0.0.1:{port}"),
        "connection_mode": "grpc",
        "runtime": "tokenspeed",
        "models": [{"id": MODEL}],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/workers")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("a registration request");
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("a registration answer");
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let location = response
        .headers()
        .get(LOCATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .expect("the registration's Location");

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

    // The Location the POST answered with resolves to the registered engine:
    // for a data-parallel one, its first rank with the group size.
    let request = Request::builder()
        .uri(&location)
        .body(Body::empty())
        .expect("a lookup request");
    let response = app.oneshot(request).await.expect("a lookup answer");
    assert_eq!(response.status(), StatusCode::OK, "{location}");
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the lookup body");
    let info: serde_json::Value = serde_json::from_slice(&body).expect("worker info");
    assert_eq!(
        info["id"].as_str(),
        location.rsplit('/').next(),
        "the Location's id: {info}"
    );
    assert_eq!(info["status"], "ready", "{info}");
    if ranks > 1 {
        assert_eq!(info["dp_size"], usize::from(ranks), "{info}");
        assert_eq!(info["dp_rank"], 0, "{info}");
    }
    engine.stop().await;
    rows
}

/// Four ranks, no flag: four rows, ranks 0 to 3 of a group of four, all at
/// the engine's address. With the retired flag set: the same four.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_four_rank_engine_registers_as_four_rows_without_any_flag() {
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
async fn a_single_rank_engine_registers_as_one_row() {
    for dp_aware in [false, true] {
        let rows = registered_rows(1, dp_aware).await;
        assert_eq!(rows.len(), 1, "dp_aware {dp_aware}: {rows:?}");
        assert_eq!((rows[0].0, rows[0].1), (None, None), "{rows:?}");
    }
}

/// A second POST of the same data-parallel engine, once its registration is
/// done, is refused with 409 by the group (its rows hold their own IDs; the
/// group is the conflict for the bare URL), and the first POST's ID keeps
/// resolving to the group afterwards; DELETE by that ID removes every row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_post_of_a_data_parallel_engine_is_refused_with_409() {
    let port = free_port().await;
    let engine = MockEngine::start(port, 4).await;
    let config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    let app_context = common::create_test_context_with_mock_tokenizer(config, MODEL).await;
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router");
    let app = create_test_app_with_context(Arc::from(router), Arc::clone(&app_context));
    let body = json!({
        "url": format!("grpc://127.0.0.1:{port}"),
        "connection_mode": "grpc",
        "runtime": "tokenspeed",
        "models": [{"id": MODEL}],
    });
    let post = || {
        Request::builder()
            .method("POST")
            .uri("/workers")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("a registration request")
    };
    let response = app
        .clone()
        .oneshot(post())
        .await
        .expect("a registration answer");
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let location = response
        .headers()
        .get(LOCATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .expect("the registration's Location");
    let registry = Arc::clone(&app_context.worker_registry);
    common::wait_until("the engine's rows are routable", REGISTRATION_BOUND, || {
        registry
            .get_all()
            .iter()
            .filter(|worker| worker.status() == WorkerStatus::Ready)
            .count()
            >= 4
    })
    .await;

    // While the first registration's job is pending or processing, the
    // in-flight guard refuses a second POST (409 as well) and releases
    // nothing; the group is the conflict once that job is done, so the
    // second POST waits for it. Without the group conflict it would be
    // accepted, fail against the rank rows and release the reservation the
    // Location resolves through.
    let job_queue = Arc::clone(app_context.worker_job_queue.get().expect("the job queue"));
    let engine_url = format!("grpc://127.0.0.1:{port}");
    common::wait_until(
        "the first registration's job is done",
        REGISTRATION_BOUND,
        || {
            job_queue
                .get_status(&engine_url)
                .is_none_or(|status| !matches!(status.status.as_str(), "pending" | "processing"))
        },
    )
    .await;

    let response = app.clone().oneshot(post()).await.expect("a second answer");
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the refusal body");
    let refusal: serde_json::Value = serde_json::from_slice(&body).expect("the refusal");
    assert_eq!(
        refusal["code"], "WORKER_ALREADY_EXISTS",
        "the group, not an in-flight job, is the conflict: {refusal}"
    );
    assert_eq!(registry.get_all().len(), 4, "the rows stay");

    let request = Request::builder()
        .uri(&location)
        .body(Body::empty())
        .expect("a lookup request");
    let response = app.clone().oneshot(request).await.expect("a lookup answer");
    assert_eq!(response.status(), StatusCode::OK, "{location}");

    let request = Request::builder()
        .method("DELETE")
        .uri(&location)
        .body(Body::empty())
        .expect("a removal request");
    let response = app.oneshot(request).await.expect("a removal answer");
    assert!(response.status().is_success(), "{}", response.status());
    common::wait_until("every row is gone", REGISTRATION_BOUND, || {
        registry.get_all().is_empty()
    })
    .await;
    engine.stop().await;
}
