//! A prefill worker joins a PD deployment whose prefill policy partitions
//! requests by size across the prefill workers (`bucket`). The registration
//! step hands the bucket every prefill worker again; afterwards the sizes
//! have to be shared among all of them as the first split shared them, the
//! newcomer included. A re-split that pins every size to the first worker
//! turns the partition into a single-worker queue.

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "pd-prefill-addition-test-model";
/// The bucket's first split hands 4096 size units out evenly: two prefill
/// workers own [0, 2047] and [2048, ..), three own [0, 1364], [1365, 2729]
/// and [2730, ..). The mock tokenizer turns every `test` of a prompt into
/// one token, so a prompt's word count is the size the policy sees.
const SMALL: usize = 100;
const MID: usize = 2000;
const LARGE: usize = 3000;

/// Spawn a canned mock gRPC worker in-process.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        admin_port: None,
        context_length: 32768,
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A worker of `worker_type` for the mock on `port`, as the rendezvous-room
/// runtime the canned mock serves.
fn worker(port: u16, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                drain_settle_secs: 0,
                ..Default::default()
            })
            .build(),
    )
}

struct Bed {
    app_context: Arc<AppContext>,
    router: Box<dyn RouterTrait>,
    prefill_workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    fn prefill_url(&self, index: usize) -> String {
        self.prefill_workers[index].url().to_string()
    }

    /// Register one more prefill worker, then do what the registration
    /// workflow's policy step does once it is in: hand the bucket policy
    /// every prefill worker again. Returns the newcomer's URL.
    ///
    /// This mirrors `UpdatePoliciesStep` rather than running it: the bed
    /// registers its mock workers directly, where the removal test drives the
    /// removal job. Driving the registration job here, so that the step's
    /// own listing is what the test exercises, is owed, together with the
    /// helpers this file shares with the removal test.
    #[expect(
        clippy::unwrap_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    fn add_prefill_worker(&mut self, port: u16) -> String {
        let newcomer = worker(port, WorkerType::Prefill);
        self.app_context
            .worker_registry
            .register(Arc::clone(&newcomer))
            .unwrap();
        self.prefill_workers.push(newcomer);
        self.app_context
            .policy_registry
            .init_pd_bucket_policies(&self.prefill_workers);
        self.prefill_url(self.prefill_workers.len() - 1)
    }
}

/// A PD gRPC router with the bucket prefill policy over `prefill_ports` mock
/// prefill workers and one mock decode worker.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_bed(prefill_ports: &[u16], decode_port: u16) -> Bed {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: Some(PolicyConfig::Bucket {
                // Selection follows the size partition alone: no load spill
                // and no boundary adjustment within the test.
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: 1.0001,
                bucket_adjust_interval_secs: 3600,
            }),
            decode_policy: None,
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;

    let mut prefill_workers: Vec<Arc<dyn Worker>> = Vec::new();
    let roles = prefill_ports
        .iter()
        .map(|port| (*port, WorkerType::Prefill))
        .chain([(decode_port, WorkerType::Decode)]);
    for (port, worker_type) in roles {
        let worker = worker(port, worker_type);
        app_context
            .worker_registry
            .register(Arc::clone(&worker))
            .unwrap();
        if matches!(worker.worker_type(), WorkerType::Prefill) {
            prefill_workers.push(worker);
        }
    }
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build");
    // What the registration workflow's policy step does once prefill workers
    // are in: the bucket policy learns their URLs and splits the sizes.
    app_context
        .policy_registry
        .init_pd_bucket_policies(&prefill_workers);
    Bed {
        app_context,
        router,
        prefill_workers,
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request(words: usize) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "test ".repeat(words),
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-prefill-addition-test-tenant"))
}

/// Route one completion of `words` tokens, asking for its decision record,
/// and return the prefill worker the bucket policy chose for it.
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn prefill_worker_for(bed: &Bed, words: usize) -> String {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
    let response = bed
        .router
        .route_completion(Some(&headers), &tenant(), completion_request(words), MODEL)
        .await;
    let status = response.status();
    let trace = response
        .headers()
        .get("x-smg-cache-trace")
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    assert_eq!(
        status,
        http::StatusCode::OK,
        "{words} tokens: {}",
        String::from_utf8_lossy(&bytes)
    );
    let trace: serde_json::Value =
        serde_json::from_str(&trace.expect("the request asked for its decision record")).unwrap();
    trace["selections"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|selection| selection["policy"] == "bucket")
        .and_then(|selection| selection["worker"].as_str())
        .map(str::to_owned)
        .expect("the record names the prefill worker the bucket policy chose")
}

/// Which prefill worker took each of `rounds` requests of every size, as
/// (size, worker) in request order.
async fn placements(bed: &Bed, rounds: usize) -> Vec<(usize, String)> {
    let mut placements = Vec::new();
    for _ in 0..rounds {
        for words in [SMALL, MID, LARGE] {
            placements.push((words, prefill_worker_for(bed, words).await));
        }
    }
    placements
}

/// Two prefill workers take requests of every size; a third joins while
/// they flow. Afterwards the mid-size requests reach the second worker and
/// the large ones the newcomer: the sizes are shared among three as the
/// first split shared them among two, not all handed to the first.
#[tokio::test]
async fn a_prefill_worker_that_joins_during_traffic_takes_its_share_of_the_sizes() {
    let prefill = [
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    ];
    let decode = start_mock_grpc_worker().await;
    let mut bed = build_bed(&prefill, decode).await;
    let (first, second) = (bed.prefill_url(0), bed.prefill_url(1));

    assert_eq!(
        placements(&bed, 1).await,
        [
            (SMALL, first.clone()),
            (MID, first.clone()),
            (LARGE, second.clone())
        ],
        "before the join: two workers split the sizes at 2048"
    );

    let third = bed.add_prefill_worker(start_mock_grpc_worker().await);

    let after = placements(&bed, 3).await;
    let expected: Vec<(usize, String)> = (0..3)
        .flat_map(|_| {
            [
                (SMALL, first.clone()),
                (MID, second.clone()),
                (LARGE, third.clone()),
            ]
        })
        .collect();
    let tally = |url: &str| after.iter().filter(|(_, worker)| worker == url).count();
    assert_eq!(
        after,
        expected,
        "after the join: three workers split the sizes at 1365 and 2730; \
         requests per worker: first {}, second {}, third {}",
        tally(&first),
        tally(&second),
        tally(&third)
    );
}
