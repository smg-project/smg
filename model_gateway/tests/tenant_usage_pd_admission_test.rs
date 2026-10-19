//! The usage row of a disaggregated request that waited for admission: its
//! time runs from the dispatch, past the wait, as the streams' does.
//!
//! The same mock prefill/decode pair as the admission test (a decode window
//! of one, a mock that holds each request for a while), with an admission
//! wait long enough for the second request to be admitted once the first
//! frees the room, and the usage table installed: the second request's row
//! books the hold, not the hold plus its wait. Its own binary: the admission
//! wait is a process-wide latch, and the admission test latches it to zero.

#[path = "common/mod.rs"]
mod common;

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, Canned,
};
use openai_protocol::{completion::CompletionRequest, model_card::ModelCard};
use smg::{
    middleware::TenantRequestMeta,
    observability::tenant_usage::{self, TenantUsage},
    routers::{common::pd_admission, RouterTrait},
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

const MODEL: &str = "pd-admission-usage-test-model";
/// How long the canned mock holds each request before answering.
const HOLD: Duration = Duration::from_secs(2);
/// Head start the first request gets before the second is sent: the second
/// then waits about `HOLD - HEAD_START` for the room, and the two holds run
/// back to back.
const HEAD_START: Duration = Duration::from_millis(200);
/// The admission wait: long enough for the second request to be admitted
/// once the first completes, far from a shed.
const ADMISSION_WAIT_SECS: u64 = 10;
const FIRST_TENANT: &str = "pd-admission-usage-first";
const SECOND_TENANT: &str = "pd-admission-usage-second";

/// The canned mock worker: it holds each request for `HOLD`.
async fn start_mock_grpc_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: HOLD,
        output_tokens: 3,
    })
    .await
}

/// A PD gRPC router over one mock prefill and one mock decode worker, the
/// decode reporting a window of one, with an admission wait that outlasts
/// the first request's hold.
async fn build_waiting_pd_router(prefill_port: u16, decode_port: u16) -> Box<dyn RouterTrait> {
    // The test context skips the start-up path that latches the config's
    // wait, so latch it here.
    pd_admission::set_pd_admission_wait_secs(ADMISSION_WAIT_SECS);
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .pd_admission_wait_secs(ADMISSION_WAIT_SECS)
        .build_unchecked();
    let prefill = grpc_worker(
        prefill_port,
        WorkerType::Prefill,
        RuntimeType::TokenSpeed,
        ModelCard::new(MODEL),
    );
    let decode = grpc_worker(
        decode_port,
        WorkerType::Decode,
        RuntimeType::TokenSpeed,
        ModelCard::new(MODEL),
    )
    .labels(HashMap::from([(
        "max_num_seqs".to_string(),
        "1".to_string(),
    )]));
    build_grpc_router(
        config,
        MODEL,
        vec![Arc::new(prefill.build()), Arc::new(decode.build())],
    )
    .await
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 8,
    }))
    .unwrap()
}

/// One buffered completion under `tenant`: its status.
async fn complete(router: &dyn RouterTrait, tenant: &str) -> http::StatusCode {
    let meta = TenantRequestMeta::new(TenantKey::new(tenant));
    router
        .route_completion(None, &meta, completion_request(), MODEL)
        .await
        .status()
}

/// The usage table of the process: installed by this test, the first call
/// winning as in the gateway.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn usage_table() -> &'static Arc<TenantUsage> {
    tenant_usage::install(Arc::new(TenantUsage::new(8)));
    tenant_usage::installed().expect("the usage table is installed")
}

/// A tenant's row: its request count and its summed request seconds.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn booked(table: &TenantUsage, tenant: &str) -> (u64, f64) {
    let snapshot = table.snapshot();
    let row = snapshot
        .tenants
        .iter()
        .find(|row| row.tenant == tenant)
        .expect("a usage row for the tenant");
    (row.totals.requests, row.totals.request_seconds)
}

/// The request that waited for the room books the hold, not the hold plus
/// its wait, so a buffered request's row reads as a streamed one's would.
#[tokio::test]
async fn a_request_that_waited_for_admission_books_its_time_from_the_dispatch() {
    let table = usage_table();
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_waiting_pd_router(prefill, decode).await;

    let first = complete(router.as_ref(), FIRST_TENANT);
    let second = async {
        tokio::time::sleep(HEAD_START).await;
        complete(router.as_ref(), SECOND_TENANT).await
    };
    let started = Instant::now();
    let (first, second) = tokio::join!(first, second);
    let wall = started.elapsed();
    assert_eq!(
        (first, second),
        (http::StatusCode::OK, http::StatusCode::OK),
        "both requests are admitted, the second after its wait"
    );
    // The two holds ran back to back: the second request waited for the
    // first's room, so its row below is a waiting request's.
    assert!(
        wall >= HOLD * 2 - HEAD_START / 2,
        "the second request waited for the first's room: {wall:?}"
    );

    let hold = HOLD.as_secs_f64();
    let (requests, seconds) = booked(table, SECOND_TENANT);
    assert_eq!(requests, 1, "the waiting request is booked once");
    assert!(
        seconds >= hold * 0.9 && seconds < hold * 1.5,
        "the waiting request books the hold ({hold} s), not the hold plus its wait: {seconds} s"
    );
    let (requests, seconds) = booked(table, FIRST_TENANT);
    assert_eq!(requests, 1, "the first request is booked once");
    assert!(
        seconds >= hold * 0.9 && seconds < hold * 1.5,
        "the first request, admitted at once, books the hold too: {seconds} s"
    );
}
