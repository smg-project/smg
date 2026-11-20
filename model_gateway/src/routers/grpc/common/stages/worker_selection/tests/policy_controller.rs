//! Regular selection and re-selection over a store with worker rows.

use axum::body::to_bytes;
use llm_multimodal::MediaContentPart;
use serde_json::Value;
use smg_policy_controller::{
    store::{GatewayId, LeasesStore, MemoryStore, WorkersStore},
    view::fingerprint,
};
use smg_worker::health::circuit_breaker::CircuitBreakerConfig;

use super::*;
use crate::{middleware::request_id::RequestId, routers::grpc::context::LoadGuards};

const MODEL: &str = "model-a";
const URLS: [&str; 2] = ["http://w0:8000", "http://w1:8000"];
const WINDOW: u32 = 16;

fn stage_with_host(urls: &[&str]) -> (WorkerSelectionStage, Arc<MemoryStore>) {
    let workers: Vec<_> = urls
        .iter()
        .map(|url| {
            (
                *url,
                WorkerType::Regular,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            )
        })
        .collect();
    stage_with_workers(&workers)
}

/// The stage over a host whose rows have one worker a URL, each of the
/// type, on the connection and of the runtime beside it.
fn stage_with_workers(
    workers: &[(&str, WorkerType, ConnectionMode, RuntimeType)],
) -> (WorkerSelectionStage, Arc<MemoryStore>) {
    let urls: Vec<&str> = workers.iter().map(|(url, ..)| *url).collect();
    let (_, store, rows) = host::testing::bed(&urls);
    let registry = Arc::new(WorkerRegistry::new());
    for (row, (_, worker_type, connection, runtime)) in rows.iter().zip(workers) {
        let worker = BasicWorkerBuilder::carrying(row)
            .worker_type(*worker_type)
            .connection_mode(*connection)
            .runtime_type(*runtime)
            .model(ModelCard::new(MODEL).with_context_length(WINDOW))
            .build();
        assert!(registry.register(Arc::new(worker)).is_some());
    }
    let host = Host::new(Arc::clone(&store), Arc::clone(&registry));
    (
        regular_stage(registry).with_host(Some(Arc::new(host))),
        store,
    )
}

fn ingress(registry: &Arc<WorkerRegistry>) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.input.model_id = MODEL.to_owned();
    ctx
}

async fn assert_named_refusal(
    response: Response,
    shape: &str,
    store: &MemoryStore,
) -> anyhow::Result<()> {
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = to_bytes(response.into_body(), 4096).await?;
    let value: Value = serde_json::from_slice(&body)?;
    assert_eq!(value["error"]["code"], "not_decided_by_policy_controller");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains(shape)),
        "{value}"
    );
    assert_eq!(LeasesStore::live(store, GatewayId::LOCAL), 0);
    Ok(())
}

/// `refused` is the host's answer for an attempt no target holds, it is not
/// retried in process, and no lease is left.
fn assert_no_target_holds(refused: &Response, store: &MemoryStore) {
    let answer = host::Unplaced::NoTargetHolds.answer(MODEL);
    assert_eq!(refused.status(), answer.status());
    assert_eq!(
        error::extract_error_code_from_response(refused),
        error::extract_error_code_from_response(&answer),
    );
    assert!(!is_retryable_response(refused));
    assert_eq!(LeasesStore::live(store, GatewayId::LOCAL), 0);
}

/// The one live lease is named by none of `ids`: an ask carries the
/// gateway's own mark and nothing a client sent.
fn assert_the_lease_is_named_by_none_of(store: &MemoryStore, ids: &[&str]) {
    let leases = store.expire(Instant::now() + Duration::from_secs(60));
    assert_eq!(leases.len(), 1);
    for id in ids {
        assert_ne!(leases[0].record.request, fingerprint(id.as_bytes()), "{id}");
    }
}

#[tokio::test]
async fn a_stage_built_with_no_host_holds_none() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = regular_worker(URLS[0], MODEL, UNLABELED);
    assert!(registry.register(Arc::clone(&worker)).is_some());
    let stage = regular_stage(Arc::clone(&registry));
    let mut ctx = ingress(&registry);
    stage.execute(&mut ctx).await.expect("classic selection");
    assert!(ctx.state.placed.is_none());
    assert!(matches!(
        ctx.state.workers,
        Some(WorkerSelection::Single { worker: selected }) if Arc::ptr_eq(&selected, &worker)
    ));

    let wire = WireConstraint {
        runtime: worker.metadata().spec.runtime_type,
        ..grpc_wire(RuntimeType::Vllm)
    };
    let mut ctx = dispatch_ctx(MODEL, wire);
    stage
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect("classic re-selection");
    assert!(ctx.placed.is_none());
}

#[tokio::test]
async fn a_first_attempt_under_the_host_lands_on_the_placed_worker() {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut ctx = ingress(&stage.worker_registry);
    let RequestType::Generate(request) = &mut ctx.input.request_type else {
        panic!("a generate request");
    };
    Arc::make_mut(request).rid = Some("request-rid".to_owned());
    ctx.input.tenant_request_meta = Some(
        TenantRequestMeta::new(TenantKey::new("test-tenant"))
            .with_extension(RequestId("middleware-id".to_owned())),
    );
    ctx.input.streaming = true;
    let id = host::mark();
    let ask = RegularAttempt::First(&ctx).ask(MODEL, &id);
    assert_eq!(ask.id, id);
    assert!(ask.streaming);
    assert!(ask.tried.is_empty());
    stage.execute(&mut ctx).await.expect("a placed worker");
    let ctx = Box::new(ctx).into_dispatch().expect("a dispatch context");
    let placed = ctx
        .placed
        .as_ref()
        .expect("the placement followed the worker");
    let row = WorkersStore::get(&*store, placed.id()).expect("the placement's row");
    assert_eq!(row.facts.key, URLS[0]);
    let Some(WorkerSelection::Single { worker }) = ctx.workers.as_ref() else {
        panic!("a regular worker");
    };
    assert!(Arc::ptr_eq(worker, placed.worker()));
    assert_eq!(worker.url(), row.facts.key);
    assert_the_lease_is_named_by_none_of(&store, &["request-rid", "middleware-id", ""]);
}

#[tokio::test]
async fn each_ask_carries_a_mark_of_its_own() {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut first = ingress(&stage.worker_registry);
    stage.execute(&mut first).await.expect("a placed worker");
    let mut second = ingress(&stage.worker_registry);
    stage.execute(&mut second).await.expect("a placed worker");
    let leases = store.expire(Instant::now() + Duration::from_secs(60));
    assert_eq!(leases.len(), 2);
    assert_ne!(leases[0].record.request, leases[1].record.request);
}

#[tokio::test]
async fn a_worker_with_an_open_breaker_is_passed_over() {
    let (stage, _) = stage_with_host(&URLS);
    let mut ctx = ingress(&stage.worker_registry);
    stage.execute(&mut ctx).await.expect("an initial pick");
    let initial = ctx.state.placed.as_ref().expect("a placement");
    let blocked = Arc::clone(initial.worker());
    drop(ctx);
    for _ in 0..CircuitBreakerConfig::default().failure_threshold {
        blocked.record_outcome(503);
    }
    assert!(!blocked.circuit_breaker_can_execute());

    let mut ctx = ingress(&stage.worker_registry);
    stage.execute(&mut ctx).await.expect("the other worker");
    let placed = ctx.state.placed.as_ref().expect("a placement");
    assert_ne!(placed.worker().url(), blocked.url());
}

#[tokio::test]
async fn an_attempt_no_target_holds_reads_the_hosts_answer() {
    use openai_protocol::worker::WorkerStatus;

    let (stage, store) = stage_with_host(&URLS[..1]);
    let worker = stage
        .worker_registry
        .get_by_url(URLS[0])
        .expect("the worker");
    worker.set_status(WorkerStatus::NotReady);
    let mut ctx = ingress(&stage.worker_registry);
    let refused = stage.execute(&mut ctx).await.expect_err("no target holds");
    assert_no_target_holds(&refused, &store);
}

#[tokio::test]
async fn media_under_the_host_is_refused_by_name() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_intermediate = Some(image_intermediate(None));
    let refused = stage.execute(&mut ctx).await.expect_err("pixel input");
    assert_named_refusal(refused, "media input", &store).await?;

    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_refs = Some(multimodal::MediaPlan::new([MediaContentPart::AudioUrl {
        url: "https://example.test/audio.wav".to_owned().into(),
        uuid: None,
    }]));
    let refused = stage.execute(&mut ctx).await.expect_err("audio reference");
    assert_named_refusal(refused, "media input", &store).await?;

    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    ctx.wire.requires_media_refs = true;
    let refused = stage
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect_err("retained media reference");
    assert_named_refusal(refused, "media input", &store).await
}

#[tokio::test]
async fn a_bound_context_window_under_the_host_is_refused_by_name() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: None,
        token_ids: vec![1; WINDOW as usize + 1],
    });
    let refused = stage.execute(&mut ctx).await.expect_err("the window binds");
    assert_named_refusal(refused, "a context demand a worker's window binds", &store).await
}

#[tokio::test]
async fn a_retry_under_a_decoding_rule_is_refused_by_name() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    ctx.constraint_decided_on = Some(utils::WorkerDecoding::of(None));
    let refused = stage
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect_err("the retry carries a decoding rule");
    assert_named_refusal(refused, "a retry under a decoding rule", &store).await
}

#[tokio::test]
async fn a_target_off_the_requests_wire_is_passed_over() {
    let pool = [
        (
            URLS[0],
            WorkerType::Regular,
            ConnectionMode::Grpc,
            RuntimeType::Vllm,
        ),
        (
            URLS[1],
            WorkerType::Regular,
            ConnectionMode::Grpc,
            RuntimeType::Sglang,
        ),
    ];
    let (stage, store) = stage_with_workers(&pool);
    let plan = retained_plan(&[1]);
    // Each wire admits one worker of the two. Every attempt under it lands
    // there, whichever target the controller names first.
    for (url, _, _, runtime) in pool {
        for _ in 0..4 {
            let mut ctx = dispatch_ctx(MODEL, grpc_wire(runtime));
            stage
                .reselect(&mut ctx, &plan)
                .await
                .expect("a worker on the wire");
            let placed = ctx.placed.as_ref().expect("a placement");
            assert_eq!(placed.worker().url(), url);
        }
    }

    // A first attempt pins a wire for an adapter.
    for _ in 0..4 {
        let mut ctx = ingress(&stage.worker_registry);
        let RequestType::Generate(request) = &mut ctx.input.request_type else {
            panic!("a generate request");
        };
        Arc::make_mut(request).lora_path = Some("adapter".to_owned());
        stage.execute(&mut ctx).await.expect("a worker on the wire");
        let placed = ctx.state.placed.as_ref().expect("a placement");
        assert_eq!(placed.worker().url(), URLS[1]);
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

#[tokio::test]
async fn a_worker_outside_the_regular_pool_is_passed_over() {
    const REGULAR: &str = "http://w2:8000";
    let (stage, store) = stage_with_workers(&[
        (
            URLS[0],
            WorkerType::Regular,
            ConnectionMode::Http,
            RuntimeType::Vllm,
        ),
        (
            URLS[1],
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            RuntimeType::Vllm,
        ),
        (
            REGULAR,
            WorkerType::Regular,
            ConnectionMode::Grpc,
            RuntimeType::Vllm,
        ),
    ]);
    // No wire is pinned: the regular pool's own match is the rule. One
    // worker is on another connection, one is of another type.
    for _ in 0..8 {
        let mut ctx = ingress(&stage.worker_registry);
        stage
            .execute(&mut ctx)
            .await
            .expect("the worker of the regular pool");
        let placed = ctx.state.placed.as_ref().expect("a placement");
        assert_eq!(placed.worker().url(), REGULAR);
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

#[tokio::test]
async fn an_attempt_with_no_worker_on_its_wire_reads_the_hosts_answer() {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let off_the_pool = WireConstraint {
        connection: ConnectionMode::Zmq,
        ..grpc_wire(RuntimeType::Vllm)
    };
    let mut ctx = dispatch_ctx(MODEL, off_the_pool);
    let refused = stage
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect_err("no worker is on the wire");
    assert_no_target_holds(&refused, &store);

    // A first attempt pins a wire for an adapter.
    let mut ctx = ingress(&stage.worker_registry);
    let RequestType::Generate(request) = &mut ctx.input.request_type else {
        panic!("a generate request");
    };
    Arc::make_mut(request).lora_path = Some("adapter".to_owned());
    let refused = stage
        .execute(&mut ctx)
        .await
        .expect_err("no worker is on the adapter's wire");
    assert_no_target_holds(&refused, &store);
}

#[tokio::test]
async fn a_retry_names_the_rows_of_its_failed_attempts() {
    let (stage, store) = stage_with_host(&URLS);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    ctx.request_id = Some("retry-id".to_owned());
    ctx.streaming = true;
    let id = host::mark();
    assert!(RegularAttempt::Retry(&ctx).ask(MODEL, &id).streaming);
    let plan = retained_plan(&[1]);
    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("the first pick");
    let first = ctx.placed.as_ref().expect("the first placement");
    let first_row = first.id();
    let other_url = URLS
        .into_iter()
        .find(|url| *url != first.worker().url())
        .expect("another worker");
    let other = stage
        .worker_registry
        .get_by_url(other_url)
        .expect("the other worker");
    // The attempt fails ahead of its send, and its row is named all the same.
    ctx.release_attempt();
    assert_eq!(*ctx.tried_rows, [first_row]);

    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("the other row");
    let second = ctx.placed.as_ref().expect("the second placement");
    assert_ne!(second.id(), first_row);
    assert!(Arc::ptr_eq(second.worker(), &other));
    assert!(WireConstraint::admits(
        Some(ctx.wire),
        second.worker().as_ref()
    ));
    let Some(WorkerSelection::Single { worker }) = ctx.workers.as_ref() else {
        panic!("a regular worker");
    };
    assert!(Arc::ptr_eq(worker, second.worker()));
    assert_the_lease_is_named_by_none_of(&store, &["retry-id", ""]);
}

#[tokio::test]
async fn a_retry_never_returns_to_a_worker_that_answered_definitively() {
    let (stage, store) = stage_with_host(&URLS);
    let plan = retained_plan(&[1]);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));

    // The first attempt's worker answers the request definitively: the
    // ledger closes its window, and its row is named as tried.
    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("the first pick");
    let closed = Arc::clone(ctx.placed.as_ref().expect("a placement").worker());
    let mut failure = error::internal_error("upstream", "Definitive failure");
    mark_upstream_answer(&mut failure);
    ctx.ledger.settle(&mut failure, [&closed], []);
    assert!(!ctx.ledger.admits(closed.as_ref()));
    ctx.release_attempt();

    // The second attempt's worker fails in passing: its row is tried too,
    // and its window stays open.
    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("the other worker");
    let open = Arc::clone(ctx.placed.as_ref().expect("a placement").worker());
    assert!(!Arc::ptr_eq(&open, &closed));
    ctx.release_attempt();
    assert_eq!(ctx.tried_rows.len(), 2);

    // Every row is tried now, so the controller's soft filter hands both
    // back. The commit still passes over the worker the ledger closed.
    for _ in 0..8 {
        stage
            .reselect(&mut ctx, &plan)
            .await
            .expect("the open worker");
        let placed = ctx.placed.take().expect("a placement");
        assert!(Arc::ptr_eq(placed.worker(), &open));
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

#[tokio::test]
async fn a_failed_attempt_lets_its_worker_go_and_names_its_row() {
    let (stage, store) = stage_with_host(&URLS);
    let plan = retained_plan(&[1]);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));

    // A dispatched attempt: the row is read from the guards before they go.
    stage.reselect(&mut ctx, &plan).await.expect("a placement");
    let placed = ctx.placed.take().expect("the placement");
    let sent = placed.id();
    let worker = Arc::clone(placed.worker());
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker),
    };
    ctx.load_guards = Some(LoadGuards::at_dispatch(
        &selection,
        None,
        1,
        Some(placed),
        Instant::now(),
    ));
    assert_eq!(worker.load(), 1);
    ctx.release_attempt();
    assert!(ctx.load_guards.is_none());
    assert_eq!(*ctx.tried_rows, [sent]);
    assert_eq!(
        (LeasesStore::live(&*store, GatewayId::LOCAL), worker.load()),
        (0, 0)
    );

    // An attempt that failed ahead of its send: its placement goes too, and
    // its row is named from the context.
    stage.reselect(&mut ctx, &plan).await.expect("a placement");
    let unsent = ctx.placed.as_ref().expect("the placement").id();
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    ctx.release_attempt();
    assert!(ctx.placed.is_none());
    assert_eq!(*ctx.tried_rows, [sent, unsent]);
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

#[tokio::test]
async fn an_attempt_the_controller_does_not_place_reads_the_hosts_answer() {
    let (stage, store) = stage_with_host(&[]);
    let mut ctx = ingress(&stage.worker_registry);
    let unplaced = stage.execute(&mut ctx).await.expect_err("no store rows");
    assert_eq!(unplaced.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&unplaced),
        "no_available_workers",
    );
    assert!(!is_retryable_response(&unplaced));
    assert!(ctx.state.placed.is_none());
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}
