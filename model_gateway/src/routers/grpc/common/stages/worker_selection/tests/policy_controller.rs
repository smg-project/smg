//! Regular selection and re-selection over a store with worker rows.

use anyhow::Context;
use axum::body::to_bytes;
use kv_index::{salt::request_content_hashes_with_seed, XXH3_SEED};
use llm_multimodal::MediaContentPart;
use serde_json::Value;
use smg_policy_controller::{
    spec::{KeySourceSpec, RoutingKeySpec},
    store::{
        BatchOutcome, EncodeHandoff as OfferedHandoff, EncodeOffer, GatewayId, KvEventSink,
        LeasesStore, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus as RowStatus,
        WorkersStore,
    },
    view::{fingerprint, LoadRow, Role, WireModes, WorkerId},
    worker_host, AppliedSpec, PolicySpec,
};

use super::*;
use crate::{
    config::{
        ManualAssignmentMode, PolicyConfig, RouterConfig, RoutingKeyOverrideConfig, RoutingMode,
    },
    middleware::request_id::RequestId,
    policy_controller::host::testing::MODEL,
    routers::grpc::context::LoadGuards,
};

const URLS: [&str; 2] = ["http://w0:8000", "http://w1:8000"];
const WINDOW: u32 = 16;
const BLOCK_SIZE: u32 = 4;
const TOKENS: [u32; BLOCK_SIZE as usize] = [1, 2, 3, 4];
const PLACEMENTS: usize = 8;

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
    (stage_on_rows(workers, &store, &rows), store)
}

fn stage_on_rows(
    workers: &[(&str, WorkerType, ConnectionMode, RuntimeType)],
    store: &Arc<MemoryStore>,
    rows: &[Arc<dyn Worker>],
) -> WorkerSelectionStage {
    let registry = Arc::new(WorkerRegistry::new());
    for (row, (_, worker_type, connection, runtime)) in rows.iter().zip(workers) {
        let worker = BasicWorkerBuilder::carrying(row)
            .worker_type(*worker_type)
            .connection_mode(*connection)
            .runtime_type(*runtime)
            .model(ModelCard::new(MODEL).with_context_length(WINDOW))
            .build();
        state_wire(store, &worker);
        assert!(registry.register(Arc::new(worker)).is_some());
    }
    let host = Host::at_keys(Arc::clone(store), Arc::clone(&registry));
    regular_stage(registry).with_host(Some(Arc::new(host)))
}

/// The stage over a host whose document places by consistent hashing, with
/// the body's `rid` as the one source of the key.
fn stage_keyed_by_the_rid(
    urls: &[&str],
) -> anyhow::Result<(WorkerSelectionStage, Arc<MemoryStore>)> {
    let config = RouterConfig {
        policy_controller: true,
        policy: PolicyConfig::ConsistentHashing,
        ..RouterConfig::default()
    };
    let (_, store, rows) = host::testing::bed_under(PolicyConfig::ConsistentHashing, urls);
    let mut document = config.policy_document()?;
    document.generation += 1;
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        }],
        strip_lineage: false,
    });
    store.apply(config.hosted_policy(&document)?)?;
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
    Ok((stage_on_rows(&workers, &store, &rows), store))
}

fn stage_with_cached_prompt(
    seed: u64,
    runtime: RuntimeType,
) -> anyhow::Result<(WorkerSelectionStage, WorkerId)> {
    let store = Arc::new(MemoryStore::with_publish_window(Duration::ZERO));
    let document = PolicySpec::preset("cache_aware")?;
    store.apply(AppliedSpec::validate(&document)?)?;
    let rows = URLS.map(|url| (url, Some(MODEL)));
    let (_, store, rows) =
        host::testing::bed_with(store, Arc::new(WorkerRegistry::new()), rows, None);
    let workers = URLS.map(|url| (url, WorkerType::Regular, ConnectionMode::Grpc, runtime));
    let stage = stage_on_rows(&workers, &store, &rows);

    let other = store.by_key(URLS[0]).context("the other row")?.id;
    let holder = store.by_key(URLS[1]).context("the holder's row")?.id;
    store.set_block_size(MODEL, BLOCK_SIZE);
    // Both workers hold a block so neither is thin. A miss goes to the less
    // loaded worker, and only a match can place on the holder.
    for (id, tokens, running) in [(other, [2; BLOCK_SIZE as usize], 0), (holder, TOKENS, 1)] {
        let blocks = request_content_hashes_with_seed(&tokens, BLOCK_SIZE as usize, seed);
        assert_eq!(
            host::testing::store_blocks(&store, id, &blocks),
            BatchOutcome::Applied
        );
        store.record(
            id,
            0,
            LoadRow {
                running,
                ..LoadRow::default()
            },
            Instant::now(),
            LoadSource::Stream,
        );
    }
    Ok((stage, holder))
}

fn ingress(registry: &Arc<WorkerRegistry>) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.input.model_id = MODEL.to_owned();
    ctx
}

fn ingress_with_tokens(registry: &Arc<WorkerRegistry>, tokens: &[u32]) -> RequestContext {
    let mut ctx = ingress(registry);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: None,
        token_ids: tokens.to_vec(),
    });
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

/// The row of `worker` states the wire of its connection, by the rule a
/// store's writer states it.
fn state_wire(store: &MemoryStore, worker: &dyn Worker) {
    let mut facts = store.by_key(worker.url()).expect("the worker's row").facts;
    facts.wire_modes = worker_host::facts(worker, None).wire_modes;
    let _row = store.upsert(facts);
    store.publish_now();
}

/// The row of `worker` states its runtime, by the rule a store's writer
/// states it.
fn state_runtime(store: &MemoryStore, worker: &dyn Worker) {
    let mut facts = store.by_key(worker.url()).expect("the worker's row").facts;
    facts.engine_kind = worker_host::facts(worker, None).engine_kind;
    let _row = store.upsert(facts);
    store.publish_now();
}

/// `refused` is the host's answer for an ask no row serves on its wires:
/// the controller shed it at its wires, the answer is final, and no lease
/// was issued.
fn assert_shed_by_wires(refused: &Response, store: &MemoryStore) {
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        placement::refusal_reason(refused),
        Some("the policy controller's wires stage left no worker")
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
    let mut headers = HeaderMap::new();
    headers.insert("x-session", HeaderValue::from_static("s1"));
    ctx.input.headers = Some(headers);
    let id = host::mark();
    let hashes = host::Hashes::default();
    let ask = Attempt::First(&ctx).ask(MODEL, &id, &hashes);
    assert_eq!(ask.id, id);
    assert!(ask.streaming);
    assert!(ask.tried.is_empty());
    // The sources of its routing key: its headers and its own `rid`.
    assert_eq!(ask.headers, ctx.input.headers.as_ref());
    assert_eq!(ask.rid, Some("request-rid"));
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
async fn a_first_attempt_lands_on_the_worker_that_holds_its_prompt() -> anyhow::Result<()> {
    let (stage, holder) = stage_with_cached_prompt(XXH3_SEED, RuntimeType::Vllm)?;
    for _ in 0..PLACEMENTS {
        let mut ctx = ingress_with_tokens(&stage.worker_registry, &TOKENS);
        stage.execute(&mut ctx).await.expect("a placement");
        assert_eq!(ctx.state.placed.expect("a placed worker").id(), holder);
    }
    Ok(())
}

#[tokio::test]
async fn a_retry_states_the_retained_prompt() -> anyhow::Result<()> {
    let (stage, holder) = stage_with_cached_prompt(XXH3_SEED, RuntimeType::Vllm)?;
    let plan = retained_plan(&TOKENS);
    for _ in 0..PLACEMENTS {
        let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
        stage.reselect(&mut ctx, &plan).await.expect("a placement");
        assert_eq!(ctx.placed.expect("a placed worker").id(), holder);
    }
    Ok(())
}

#[tokio::test]
async fn a_prompt_under_a_cache_partition_hashes_under_its_seed() -> anyhow::Result<()> {
    let adapter = "adapter";
    let namespace = CacheNamespace::derive(&CachePartition {
        lora_path: Some(adapter),
        ..CachePartition::default()
    })
    .expect("the adapter's namespace");
    let (stage, holder) = stage_with_cached_prompt(namespace.event_seed(), RuntimeType::Sglang)?;

    let mut scoped = ingress_with_tokens(&stage.worker_registry, &TOKENS);
    let RequestType::Generate(request) = &mut scoped.input.request_type else {
        panic!("a generate request");
    };
    Arc::make_mut(request).lora_path = Some(adapter.to_owned());
    stage
        .execute(&mut scoped)
        .await
        .expect("a scoped placement");
    assert_eq!(scoped.state.placed.expect("a placed worker").id(), holder);

    let mut plain = ingress_with_tokens(&stage.worker_registry, &TOKENS);
    stage.execute(&mut plain).await.expect("a plain placement");
    assert_eq!(
        plain.state.placed.expect("a placed worker").worker().url(),
        URLS[0]
    );
    Ok(())
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

/// A worker of `worker_type` on gRPC in `registry` that states no window,
/// so no prompt is past it. No row stands for it: the stage reads a window
/// off the registry.
fn register_unbound(registry: &WorkerRegistry, url: &str, worker_type: WorkerType) {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::Vllm)
        .model(ModelCard::new(MODEL))
        .build();
    assert!(registry.register(Arc::new(worker)).is_some());
}

/// A request whose prompt is one token past [`WINDOW`].
fn past_the_window(registry: &Arc<WorkerRegistry>) -> RequestContext {
    ingress_with_tokens(registry, &[1; WINDOW as usize + 1])
}

/// The status and the body of an answer.
async fn answer_of(response: Response) -> anyhow::Result<(StatusCode, Value)> {
    let status = response.status();
    let body = to_bytes(response.into_body(), 4096).await?;
    Ok((status, serde_json::from_slice(&body)?))
}

#[tokio::test]
async fn a_bound_context_window_under_the_host_is_refused_by_name() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    // A second worker states no window, so it may hold the prompt. The
    // demand binds one window of two, and the controller cannot place by it
    // yet.
    register_unbound(
        &stage.worker_registry,
        "http://unbound:8000",
        WorkerType::Regular,
    );
    let mut ctx = past_the_window(&stage.worker_registry);
    let refused = stage.execute(&mut ctx).await.expect_err("the window binds");
    assert_named_refusal(refused, "a context demand a worker's window binds", &store).await
}

/// `stage`'s workers and host under consistent hashing, the classic policy
/// that reads an explicit target.
fn under_consistent_hashing(stage: &WorkerSelectionStage) -> WorkerSelectionStage {
    WorkerSelectionStage::new(
        Arc::clone(&stage.worker_registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHashing)),
        stage.mode,
        None,
    )
    .with_host(stage.host.clone())
}

/// A request that names the first candidate as its target.
fn naming_its_worker(registry: &Arc<WorkerRegistry>) -> RequestContext {
    let mut ctx = ingress(registry);
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-target-worker", HeaderValue::from_static("0"));
    ctx.input.headers = Some(headers);
    ctx
}

/// A pair's settings keep an explicit target where the prefill or the
/// decode policy, each the main one where unset, is consistent hashing.
#[test]
fn a_pairs_settings_keep_a_target_under_consistent_hashing_on_a_role() {
    let pair = |prefill_policy: Option<PolicyConfig>, decode_policy: Option<PolicyConfig>| {
        RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy,
            decode_policy,
        }
    };
    let hashing = || Some(PolicyConfig::ConsistentHashing);
    let other = || Some(PolicyConfig::Random);
    // A role alone, under another main policy.
    let main = PolicyConfig::RoundRobin;
    assert!(placement::pair_keeps_target(&pair(hashing(), None), &main));
    assert!(placement::pair_keeps_target(&pair(None, hashing()), &main));
    assert!(!placement::pair_keeps_target(&pair(None, None), &main));
    // The main policy stands for a role with no setting of its own, and
    // for no role that has one.
    let main = PolicyConfig::ConsistentHashing;
    assert!(placement::pair_keeps_target(&pair(None, other()), &main));
    assert!(!placement::pair_keeps_target(
        &pair(other(), other()),
        &main
    ));
}

/// Under consistent hashing the classic holds a request to the worker its
/// header names. The controller reads no target yet and would place the
/// request on another worker in silence, so both lines refuse it by name.
/// Under a policy that reads no target the request is placed as before.
#[tokio::test]
async fn an_explicit_target_the_classic_keeps_is_refused_by_name() -> anyhow::Result<()> {
    let (plain, store) = stage_with_host(&URLS);
    let hashing = under_consistent_hashing(&plain);
    let mut ctx = naming_its_worker(&hashing.worker_registry);
    let refused = hashing
        .execute(&mut ctx)
        .await
        .expect_err("the request names its worker");
    assert_named_refusal(refused, "an explicit target worker", &store).await?;
    // A retry of it has no next attempt.
    let mut retry = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    retry.headers.clone_from(&ctx.input.headers);
    assert!(!hashing.places_retry(&retry));
    assert!(plain.places_retry(&retry));

    // A pair asks its settings. No role policy is built under the host, so
    // the registry's main policy says nothing of a pair.
    let (pair, pair_store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let unkept = under_consistent_hashing(&pair);
    let mut ctx = naming_its_worker(&unkept.worker_registry);
    assert!(unkept.execute(&mut ctx).await.is_ok());
    drop(ctx);
    let kept = pair.with_pair_keeps_target(true);
    let mut ctx = naming_its_worker(&kept.worker_registry);
    let refused = kept
        .execute(&mut ctx)
        .await
        .expect_err("the request names its worker");
    assert_named_refusal(refused, "an explicit target worker", &pair_store).await?;

    let mut ctx = naming_its_worker(&plain.worker_registry);
    assert!(plain.execute(&mut ctx).await.is_ok());
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    Ok(())
}

/// A prompt no worker's window holds is the client's to fix. Under the host
/// it reads the classic's 400, the answer of the classic stage over the
/// same workers, and no ask is made.
#[tokio::test]
async fn a_prompt_no_window_holds_reads_the_classic_400_under_the_host() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    let mut ctx = past_the_window(&stage.worker_registry);
    let refused = stage
        .execute(&mut ctx)
        .await
        .expect_err("no window holds the prompt");
    let hosted = answer_of(refused).await?;
    assert_eq!(hosted.0, StatusCode::BAD_REQUEST);
    assert_eq!(hosted.1["error"]["code"], "context_length_exceeded");
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);

    let classic = regular_stage(Arc::clone(&stage.worker_registry));
    let mut ctx = past_the_window(&classic.worker_registry);
    let refused = classic
        .execute(&mut ctx)
        .await
        .expect_err("no window holds the prompt");
    assert_eq!(answer_of(refused).await?, hosted);
    Ok(())
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
    // No wire is pinned. The ask states the pipeline's wires, so the worker
    // on another connection is never named. The one of another type is
    // named, and the regular pool's own match passes it over.
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
    // The retry states its plan's one wire, and no row serves it: the
    // controller sheds the ask, and the route passes over nothing.
    assert_shed_by_wires(&refused, &store);

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

/// The stage's ask states the wires its pipeline dials, so the controller
/// places among the rows that serve one. A fleet on another wire holds no
/// such row: the ask is shed at its wires and no lease is issued. With no
/// wire stated the controller named a target the route then passed over.
#[tokio::test]
async fn an_ask_states_the_wires_its_pipeline_dials() {
    let (stage, store) = stage_with_workers(&[(
        URLS[0],
        WorkerType::Regular,
        ConnectionMode::Http,
        RuntimeType::Vllm,
    )]);
    let mut ctx = ingress(&stage.worker_registry);
    let refused = stage
        .execute(&mut ctx)
        .await
        .expect_err("no row serves the pipeline's wires");
    assert_shed_by_wires(&refused, &store);
}

/// ZMQ is a wire of the pipeline too. With no pin the ask states it beside
/// gRPC, so a worker on ZMQ is placed and the one on HTTP is never named.
#[tokio::test]
async fn an_unpinned_ask_places_a_worker_on_zmq() {
    const ON_ZMQ: &str = "http://w2:8000";
    let (stage, store) = stage_with_workers(&[
        (
            URLS[0],
            WorkerType::Regular,
            ConnectionMode::Http,
            RuntimeType::Vllm,
        ),
        (
            ON_ZMQ,
            WorkerType::Regular,
            ConnectionMode::Zmq,
            RuntimeType::TokenSpeed,
        ),
    ]);
    for _ in 0..8 {
        let mut ctx = ingress(&stage.worker_registry);
        stage.execute(&mut ctx).await.expect("the worker on ZMQ");
        let placed = ctx.state.placed.as_ref().expect("a placement");
        assert_eq!(placed.worker().url(), ON_ZMQ);
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

/// With no pin the ask states the wire of every connection that uses the
/// gRPC pipeline, and of no other. A pinned ask states its connection's
/// wire by a match over every connection, so one that joins is visited.
#[test]
fn an_unpinned_ask_states_the_wires_of_the_pipelines_connections() {
    let unpinned = route_wires(None);
    for connection in [
        ConnectionMode::Http,
        ConnectionMode::Grpc,
        ConnectionMode::Zmq,
    ] {
        let pinned = route_wires(Some(WireConstraint {
            connection,
            ..grpc_wire(RuntimeType::Vllm)
        }));
        assert_eq!(
            unpinned.intersects(pinned),
            connection.uses_grpc_pipeline(),
            "{connection:?}"
        );
    }
}

#[tokio::test]
async fn a_retry_of_a_shape_refused_by_name_has_no_next_attempt() -> anyhow::Result<()> {
    let (stage, store) = stage_with_host(&URLS[..1]);
    // A plain retry is placed, so a next attempt may follow.
    let plain = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    assert!(stage.places_retry(&plain));

    // Each shape the stage refuses by name ends the attempts.
    let mut under_rule = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    under_rule.constraint_decided_on = Some(utils::WorkerDecoding::of(None));
    assert!(!stage.places_retry(&under_rule));
    let mut with_media = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    with_media.wire.requires_media_refs = true;
    assert!(!stage.places_retry(&with_media));

    // Asked all the same, the stage still refuses by name.
    let refused = stage
        .reselect(&mut under_rule, &retained_plan(&[1]))
        .await
        .expect_err("the retry carries a decoding rule");
    assert_named_refusal(refused, "a retry under a decoding rule", &store).await?;

    // The classic selection answers every retry itself.
    let classic = regular_stage(Arc::clone(&stage.worker_registry));
    assert!(classic.places_retry(&under_rule));
    Ok(())
}

#[tokio::test]
async fn a_retry_names_the_rows_of_its_failed_attempts() {
    let (stage, store) = stage_with_host(&URLS);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    ctx.request_id = Some("retry-id".to_owned());
    ctx.streaming = true;
    ctx.headers = Some(HeaderMap::new());
    ctx.routing.rid = Some("retained-rid".to_owned());
    let id = host::mark();
    let hashes = host::Hashes::default();
    let ask = Attempt::Retry(&ctx).ask(MODEL, &id, &hashes);
    assert!(ask.streaming);
    // A retry states the headers and the `rid` the request left behind.
    assert_eq!(ask.headers, ctx.headers.as_ref());
    assert_eq!(ask.rid, Some("retained-rid"));
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
    let failure = error::internal_error("attempt_failed", "Attempt failed");
    ctx.release_attempt(&failure, Instant::now());
    assert_eq!(*ctx.tried_rows, [first_row]);
    // The retry's ask states that row as tried.
    let ask = Attempt::Retry(&ctx).ask(MODEL, &id, &hashes);
    assert_eq!(ask.tried, [first_row]);

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

/// A binding keyed by the body's `rid` keeps its key across a retry. The
/// first selection retains the `rid` and the retry's ask states it, so each
/// retry lands where the host places an ask of the same key with the same
/// tried row. An ask with no key does not land there for every key.
#[tokio::test]
async fn a_retry_under_a_binding_keyed_by_the_rid_is_placed_by_the_key() -> anyhow::Result<()> {
    const WORKERS: [&str; 4] = [
        "http://w0:8000",
        "http://w1:8000",
        "http://w2:8000",
        "http://w3:8000",
    ];
    let (stage, store) = stage_keyed_by_the_rid(&WORKERS)?;
    let host = stage.host.as_ref().expect("the stage's host");
    let plan = retained_plan(&[1]);
    let mut elsewhere_with_no_key = 0;
    for turn in 0..16 {
        let rid = format!("conv-{turn}");
        let mut ctx = ingress(&stage.worker_registry);
        let RequestType::Generate(request) = &mut ctx.input.request_type else {
            panic!("a generate request");
        };
        Arc::make_mut(request).rid = Some(rid.clone());
        stage.execute(&mut ctx).await.expect("the first attempt");
        // The request is gone at the dispatch: a retry reads what was kept.
        let mut ctx = Box::new(ctx).into_dispatch().expect("a dispatch context");
        let first_row = ctx.placed.as_ref().expect("the first placement").id();
        // The attempt fails ahead of its send, and its row is named as tried.
        let failure = error::internal_error("attempt_failed", "Attempt failed");
        ctx.release_attempt(&failure, Instant::now());
        assert_eq!(*ctx.tried_rows, [first_row]);

        // Where the host places the key with that row tried.
        let id = host::mark();
        let keyed = Ask {
            rid: Some(&rid),
            tried: &ctx.tried_rows,
            ..Ask::of(&id, MODEL)
        };
        let by_key = host
            .place(&keyed, Instant::now())
            .await
            .expect("the key's next worker")
            .id();

        // Where it places the same ask with no key.
        let mark = host::mark();
        let keyless = Ask {
            tried: &ctx.tried_rows,
            ..Ask::of(&mark, MODEL)
        };
        let by_none = host
            .place(&keyless, Instant::now())
            .await
            .expect("a worker with no key")
            .id();
        elsewhere_with_no_key += usize::from(by_none != by_key);

        stage.reselect(&mut ctx, &plan).await.expect("the retry");
        let retried = ctx.placed.take().expect("the retry's placement").id();
        assert_eq!(retried, by_key, "{rid}");
    }
    // The key is what places a retry: with none, an ask lands elsewhere.
    assert!(
        elsewhere_with_no_key > 0,
        "an ask with no key landed by the key at every turn"
    );
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    Ok(())
}

/// The routing snapshot keeps the body's `rid` only where a retry reads it.
/// Under a host the retry's ask states it. On the classic line it stays
/// where the sticky key derives from it, and a retry derives that key from
/// it again. Elsewhere the classic line keeps no copy.
#[tokio::test]
async fn the_snapshot_keeps_the_rid_only_where_a_retry_reads_it() {
    const RID: &str = "conv_t2_r1";
    async fn kept(stage: &WorkerSelectionStage) -> Option<String> {
        let mut ctx = ingress(&stage.worker_registry);
        let RequestType::Generate(request) = &mut ctx.input.request_type else {
            panic!("a generate request");
        };
        Arc::make_mut(request).rid = Some(RID.to_owned());
        stage.execute(&mut ctx).await.expect("a selection");
        ctx.state.routing_snapshot.expect("the snapshot").rid
    }
    let (hosted, _store) = stage_with_host(&URLS[..1]);
    assert_eq!(kept(&hosted).await.as_deref(), Some(RID));

    let registry = Arc::clone(&hosted.worker_registry);
    let classic = regular_stage(Arc::clone(&registry));
    assert_eq!(classic.policy_registry.derive_rid_key(Some(RID)), None);
    assert_eq!(kept(&classic).await, None, "no retry reads it");

    let sticky = WorkerSelectionStage::new(
        registry,
        Arc::new(PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..RoutingKeyOverrideConfig::default()
            },
        )),
        WorkerSelectionMode::Regular,
        None,
    );
    let rid = kept(&sticky).await;
    assert_eq!(rid.as_deref(), Some(RID));
    // A retry derives the first selection's key from it.
    assert_eq!(
        sticky.policy_registry.derive_rid_key(rid.as_deref()),
        Some("conv")
    );
}

/// A classic retry derives its sticky key from the `rid` the first
/// selection kept, so it lands on that selection's worker. With no key the
/// rotation under the override would move on.
#[tokio::test]
async fn a_classic_retry_lands_by_the_key_of_the_kept_rid() {
    let (hosted, _store) = stage_with_host(&URLS);
    let sticky = WorkerSelectionStage::new(
        Arc::clone(&hosted.worker_registry),
        Arc::new(PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::Delegate,
                ..RoutingKeyOverrideConfig::default()
            },
        )),
        WorkerSelectionMode::Regular,
        None,
    );
    let mut ctx = ingress(&sticky.worker_registry);
    let RequestType::Generate(request) = &mut ctx.input.request_type else {
        panic!("a generate request");
    };
    Arc::make_mut(request).rid = Some("conv_t1".to_owned());
    sticky.execute(&mut ctx).await.expect("the first selection");
    let Some(WorkerSelection::Single { worker }) = ctx.state.workers.as_ref() else {
        panic!("a regular worker");
    };
    let first = worker.url().to_owned();
    // The request is gone at the dispatch: a retry reads what was kept.
    let mut ctx = Box::new(ctx).into_dispatch().expect("a dispatch context");
    assert_eq!(ctx.routing.rid.as_deref(), Some("conv_t1"));
    for _ in 0..2 {
        sticky
            .reselect(&mut ctx, &retained_plan(&[1]))
            .await
            .expect("the retry");
        let Some(WorkerSelection::Single { worker }) = ctx.workers.as_ref() else {
            panic!("a regular worker");
        };
        assert_eq!(worker.url(), first);
    }
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
    ctx.release_attempt(&failure, Instant::now());

    // The second attempt's worker fails in passing: its row is tried too,
    // and its window stays open.
    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("the other worker");
    let open = Arc::clone(ctx.placed.as_ref().expect("a placement").worker());
    assert!(!Arc::ptr_eq(&open, &closed));
    let passing = error::service_unavailable("upstream", "Passing failure");
    ctx.release_attempt(&passing, Instant::now());
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
    let failure = error::internal_error("attempt_failed", "Attempt failed");
    ctx.release_attempt(&failure, Instant::now());
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
    ctx.release_attempt(&failure, Instant::now());
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

const PREFILL: &str = "http://p0:8000";
const DECODE: &str = "http://d0:8000";

fn urls_of<'a>(side: &[(&'a str, ConnectionMode, RuntimeType)]) -> Vec<&'a str> {
    side.iter().map(|(url, ..)| *url).collect()
}

/// `side` with every worker on gRPC.
fn on_grpc<'a>(side: &[(&'a str, RuntimeType)]) -> Vec<(&'a str, ConnectionMode, RuntimeType)> {
    side.iter()
        .map(|(url, runtime)| (*url, ConnectionMode::Grpc, *runtime))
        .collect()
}

/// [`pair_stage_on`] with every worker on gRPC.
fn pair_stage(
    mode: WorkerSelectionMode,
    prefills: &[(&str, RuntimeType)],
    decodes: &[(&str, RuntimeType)],
) -> (WorkerSelectionStage, Arc<MemoryStore>) {
    pair_stage_on(mode, &on_grpc(prefills), &on_grpc(decodes))
}

/// The stage in `mode` over a host whose rows are a prefill pool and a
/// decode pool under a pair document: one worker a URL, on the connection
/// and of the runtime beside it. Each row states its worker's wire.
fn pair_stage_on(
    mode: WorkerSelectionMode,
    prefills: &[(&str, ConnectionMode, RuntimeType)],
    decodes: &[(&str, ConnectionMode, RuntimeType)],
) -> (WorkerSelectionStage, Arc<MemoryStore>) {
    let (_, store, rows) = host::testing::pair_bed(&urls_of(prefills), &urls_of(decodes));
    let registry = Arc::new(WorkerRegistry::new());
    for (row, (_, connection, runtime)) in rows.iter().zip(prefills.iter().chain(decodes)) {
        let worker = BasicWorkerBuilder::carrying(row)
            .connection_mode(*connection)
            .runtime_type(*runtime)
            .model(ModelCard::new(MODEL).with_context_length(WINDOW))
            .build();
        state_wire(&store, &worker);
        assert!(registry.register(Arc::new(worker)).is_some());
    }
    let host = Host::at_keys(Arc::clone(&store), Arc::clone(&registry));
    let stage = WorkerSelectionStage::new(
        registry,
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        mode,
        None,
    )
    .with_host(Some(Arc::new(host)));
    (stage, store)
}

/// [`pair_stage`] over one prefill and one decode of one runtime.
fn one_pair_stage(mode: WorkerSelectionMode) -> (WorkerSelectionStage, Arc<MemoryStore>) {
    pair_stage(
        mode,
        &[(PREFILL, RuntimeType::Vllm)],
        &[(DECODE, RuntimeType::Vllm)],
    )
}

#[tokio::test]
async fn a_hosted_pair_selection_is_the_controllers_prefill_and_decode() {
    let (stage, store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let mut ctx = ingress(&stage.worker_registry);
    stage.execute(&mut ctx).await.expect("a placed pair");
    // The classic placement leaves no placement and asks for no lease.
    let placed = ctx
        .state
        .placed
        .as_ref()
        .expect("the placement followed the pair");
    assert!(ctx.state.pd_prefill_guard.is_some());
    let Some(WorkerSelection::Disaggregated {
        encode_assignments,
        prefill,
        decode,
        runtime_type,
    }) = ctx.state.workers.as_ref()
    else {
        panic!("a pair");
    };
    assert!(encode_assignments.is_none());
    assert!(Arc::ptr_eq(prefill, placed.worker()));
    assert!(Arc::ptr_eq(
        decode,
        placed.decode().expect("the pair's decode")
    ));
    assert_eq!((prefill.url(), decode.url()), (PREFILL, DECODE));
    assert_eq!(*runtime_type, RuntimeType::Vllm);
    let key = |id: WorkerId| {
        WorkersStore::get(&*store, id)
            .expect("a row")
            .facts
            .key
            .clone()
    };
    assert_eq!(key(placed.id()), PREFILL);
    assert_eq!(placed.decode_id().map(key).as_deref(), Some(DECODE));
    // One lease for the pair, asked at its prefill with the prompt's size.
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    let leases = store.expire(Instant::now() + Duration::from_secs(60));
    assert!(!leases.is_empty());
    assert!(leases.iter().all(|sides| sides.record.lease.tokens == 3));
}

#[tokio::test]
async fn a_pair_shape_the_controller_does_not_place_is_refused_by_name() -> anyhow::Result<()> {
    let (stage, store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_intermediate = Some(image_intermediate(None));
    let refused = stage.execute(&mut ctx).await.expect_err("pixel input");
    assert_named_refusal(refused, "media input", &store).await?;

    // A second pair states no window, so it may hold the prompt. The demand
    // binds one pair of two, and the controller cannot place by it yet.
    let registry = &stage.worker_registry;
    register_unbound(registry, "http://unbound-p:8000", WorkerType::Prefill);
    register_unbound(registry, "http://unbound-d:8000", WorkerType::Decode);
    let mut ctx = past_the_window(registry);
    let refused = stage.execute(&mut ctx).await.expect_err("the window binds");
    assert_named_refusal(refused, "a context demand a worker's window binds", &store).await?;

    // A retry under a decoding rule has no next attempt, and is refused by
    // name when asked all the same. A plain retry is placed.
    let mut under_rule = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    under_rule.constraint_decided_on = Some(utils::WorkerDecoding::of(None));
    assert!(!stage.places_retry(&under_rule));
    assert!(stage.places_retry(&dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm))));
    let refused = stage
        .reselect(&mut under_rule, &retained_plan(&[1]))
        .await
        .expect_err("the retry carries a decoding rule");
    assert_named_refusal(refused, "a retry under a decoding rule", &store).await
}

/// A fleet of mixed wires holds four pairs, and one of them is on the
/// pipeline's wires: on gRPC, then on ZMQ. The pair's ask states those
/// wires, so every attempt is placed on that pair. With no wire stated the
/// controller named the other three as well, and each of them was released
/// and refused.
#[tokio::test]
async fn a_pair_ask_states_the_wires_its_pipeline_dials() {
    const HTTP_PREFILL: &str = "http://p1:8000";
    const HTTP_DECODE: &str = "http://d1:8000";
    for (connection, runtime) in [
        (ConnectionMode::Grpc, RuntimeType::Vllm),
        (ConnectionMode::Zmq, RuntimeType::TokenSpeed),
    ] {
        let (stage, store) = pair_stage_on(
            WorkerSelectionMode::PrefillDecode,
            &[
                (HTTP_PREFILL, ConnectionMode::Http, RuntimeType::Vllm),
                (PREFILL, connection, runtime),
            ],
            &[
                (HTTP_DECODE, ConnectionMode::Http, RuntimeType::Vllm),
                (DECODE, connection, runtime),
            ],
        );
        for _ in 0..16 {
            let mut ctx = ingress(&stage.worker_registry);
            stage
                .execute(&mut ctx)
                .await
                .expect("the pair on the pipeline's wires");
            let placed = ctx.state.placed.as_ref().expect("a placement");
            assert_eq!(placed.worker().url(), PREFILL);
            assert_eq!(placed.decode().map(|decode| decode.url()), Some(DECODE));
        }
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    }
}

/// A prompt no pair's windows hold reads the classic's 400 under the host,
/// the answer of the classic pair stage over the same workers, and no ask
/// is made.
#[tokio::test]
async fn a_pair_prompt_no_window_holds_reads_the_classic_400_under_the_host() -> anyhow::Result<()>
{
    let (stage, store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let mut ctx = past_the_window(&stage.worker_registry);
    let refused = stage
        .execute(&mut ctx)
        .await
        .expect_err("no pair holds the prompt");
    let hosted = answer_of(refused).await?;
    assert_eq!(hosted.0, StatusCode::BAD_REQUEST);
    assert_eq!(hosted.1["error"]["code"], "context_length_exceeded");
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);

    let classic = round_robin_pd_stage(Arc::clone(&stage.worker_registry));
    let mut ctx = past_the_window(&classic.worker_registry);
    let refused = classic
        .execute(&mut ctx)
        .await
        .expect_err("no pair holds the prompt");
    assert_eq!(answer_of(refused).await?, hosted);
    Ok(())
}

#[tokio::test]
async fn a_pair_on_another_wire_than_the_pinned_one_is_released_and_refused() {
    let (stage, store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let prefill = stage
        .worker_registry
        .get_by_url(PREFILL)
        .expect("the prefill");
    let plan = retained_plan(&[1]);
    // A retry pinned to the wire of another runtime: the pair the
    // controller names is not on it. Its lease and its slot are given up.
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Sglang));
    let refused = stage
        .reselect(&mut ctx, &plan)
        .await
        .expect_err("no pair is on the wire");
    assert_no_target_holds(&refused, &store);
    assert!(ctx.placed.is_none() && ctx.pd_prefill_guard.is_none());
    assert_eq!(prefill.load(), 0);

    // On the pair's own wire the same retry is placed.
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    stage
        .reselect(&mut ctx, &plan)
        .await
        .expect("a pair on the wire");
    assert!(ctx.placed.is_some() && ctx.pd_prefill_guard.is_some());
    assert_eq!(prefill.load(), 1);
    drop(ctx);

    // A pair of two runtimes cannot go out as one request: refused the
    // same way, with no wire pinned.
    let (stage, store) = pair_stage(
        WorkerSelectionMode::PrefillDecode,
        &[(PREFILL, RuntimeType::Vllm)],
        &[(DECODE, RuntimeType::Sglang)],
    );
    let mut ctx = ingress(&stage.worker_registry);
    let refused = stage
        .execute(&mut ctx)
        .await
        .expect_err("the pair spans two runtimes");
    assert_no_target_holds(&refused, &store);
}

/// A retry keeps its plan's runtime. In a fleet of two runtimes the
/// controller names pairs of either, and the pair's commit passes over a
/// pair the route's rule refuses, so every retry lands on its own runtime's
/// pair. With no rule at the commit such a pair was released and refused.
#[tokio::test]
async fn a_pinned_retry_passes_over_a_pair_of_another_runtime() {
    const OTHER_PREFILL: &str = "http://p1:8000";
    const OTHER_DECODE: &str = "http://d1:8000";
    let (stage, store) = pair_stage(
        WorkerSelectionMode::PrefillDecode,
        &[
            (PREFILL, RuntimeType::Vllm),
            (OTHER_PREFILL, RuntimeType::Sglang),
        ],
        &[
            (DECODE, RuntimeType::Vllm),
            (OTHER_DECODE, RuntimeType::Sglang),
        ],
    );
    let plan = retained_plan(&[1]);
    for _ in 0..16 {
        let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Sglang));
        stage
            .reselect(&mut ctx, &plan)
            .await
            .expect("the pair of the plan's runtime");
        let placed = ctx.placed.as_ref().expect("a placement");
        assert_eq!(placed.worker().url(), OTHER_PREFILL);
        assert_eq!(
            placed.decode().map(|decode| decode.url()),
            Some(OTHER_DECODE)
        );
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

/// A document whose plan places no encoder leaves a request's encode items
/// with no worker: the request is refused by name, and its pair is let go.
#[tokio::test]
async fn encode_items_under_a_plan_that_places_no_encoder_are_refused_by_name() -> anyhow::Result<()>
{
    let (stage, store) = one_pair_stage(WorkerSelectionMode::EncodePrefillDecode);
    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_intermediate = Some(image_intermediate(None));
    let refused = stage.execute(&mut ctx).await.expect_err("processed items");
    assert_named_refusal(refused, "encode items", &store).await?;

    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_refs = Some(multimodal::MediaPlan::new([MediaContentPart::AudioUrl {
        url: "https://example.test/audio.wav".to_owned().into(),
        uuid: None,
    }]));
    let refused = stage.execute(&mut ctx).await.expect_err("item references");
    assert_named_refusal(refused, "encode items", &store).await?;

    // A request with none is a pair, placed in the same mode, and so is
    // its retry.
    let mut ctx = ingress(&stage.worker_registry);
    stage.execute(&mut ctx).await.expect("a placed pair");
    assert!(ctx
        .state
        .placed
        .as_ref()
        .is_some_and(|placed| placed.decode().is_some()));
    assert!(matches!(
        ctx.state.workers,
        Some(WorkerSelection::Disaggregated {
            encode_assignments: None,
            ..
        })
    ));
    drop(ctx);
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    assert!(stage.places_retry(&ctx));
    stage
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect("the pair again");
    assert!(ctx.placed.is_some());
    Ok(())
}

/// A prefill and a decode of two runtimes pair nothing on this wire: the
/// store keeps known runtimes apart, as the classic pair index does. Both
/// judge the fleet from membership alone, and under the host the answer is
/// the classic stage's own, its code and its message whole.
#[tokio::test]
async fn a_fleet_that_pairs_nothing_reads_the_classic_refusal_under_the_host() -> anyhow::Result<()>
{
    let (stage, store) = pair_stage(
        WorkerSelectionMode::PrefillDecode,
        &[(PREFILL, RuntimeType::Vllm)],
        &[(DECODE, RuntimeType::Sglang)],
    );
    for url in [PREFILL, DECODE] {
        let worker = stage
            .worker_registry
            .get_by_url(url)
            .context("the worker")?;
        state_runtime(&store, worker.as_ref());
    }
    let mut ctx = ingress(&stage.worker_registry);
    let hosted = stage
        .execute(&mut ctx)
        .await
        .expect_err("the fleet pairs nothing");
    assert!(!is_retryable_response(&hosted));
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);

    let classic = round_robin_pd_stage(Arc::clone(&stage.worker_registry));
    let mut ctx = ingress(&classic.worker_registry);
    let refused = classic
        .execute(&mut ctx)
        .await
        .expect_err("the fleet pairs nothing");
    assert_eq!(hosted.status(), refused.status());
    let hosted: Value = serde_json::from_slice(&to_bytes(hosted.into_body(), 4096).await?)?;
    let refused: Value = serde_json::from_slice(&to_bytes(refused.into_body(), 4096).await?)?;
    assert_eq!(hosted["error"]["code"], "no_compatible_pd_pair");
    assert_eq!(hosted, refused);
    Ok(())
}

#[test]
fn a_pair_attempt_states_the_retry_loops_number() {
    let first = ingress(&Arc::new(WorkerRegistry::new()));
    assert_eq!(Attempt::First(&first).number(), 0);
    // Two failed pairs named four rows, and the retry is the second.
    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    ctx.tried_rows = Box::from([WorkerId(1), WorkerId(2), WorkerId(3), WorkerId(4)]);
    ctx.attempt = 2;
    assert_eq!(Attempt::Retry(&ctx).number(), 2);
    ctx.attempt = 300;
    assert_eq!(Attempt::Retry(&ctx).number(), u8::MAX);
}

#[tokio::test]
async fn the_classic_pair_stage_is_unmoved_without_a_host() {
    let (hosted, store) = one_pair_stage(WorkerSelectionMode::PrefillDecode);
    let classic = round_robin_pd_stage(Arc::clone(&hosted.worker_registry));
    let mut ctx = ingress(&classic.worker_registry);
    classic.execute(&mut ctx).await.expect("the classic pair");
    assert!(ctx.state.placed.is_none());
    assert!(ctx.state.pd_prefill_guard.is_some());
    assert!(matches!(
        &ctx.state.workers,
        Some(WorkerSelection::Disaggregated { prefill, decode, .. })
            if prefill.url() == PREFILL && decode.url() == DECODE
    ));

    let mut ctx = dispatch_ctx(MODEL, grpc_wire(RuntimeType::Vllm));
    classic
        .reselect(&mut ctx, &retained_plan(&[1]))
        .await
        .expect("the classic pair again");
    assert!(ctx.placed.is_none() && ctx.pd_prefill_guard.is_some());
    assert!(classic.places_retry(&ctx));
    // No decision was asked of the controller.
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
}

const ENCODER: &str = "http://e0:8000";
/// The runtime whose encoder pushes an encoded item to its prefill.
const PUSH: RuntimeType = RuntimeType::Vllm;
/// The runtime whose encoder and prefill meet at a rendezvous.
const MEET: RuntimeType = RuntimeType::TokenSpeed;
/// The demand of a request whose items are processed images.
const IMAGES: EpdDemands = EpdDemands {
    wire: None,
    encode: MediaDemands {
        media_refs: false,
        vision: true,
        pipeline: PipelinePin::Any,
    },
    decode_vision: false,
    image_fanout: false,
};

/// One worker of an encode fleet: its URL, its role and the runtime its
/// registry object runs, and what its row states: its end of the encode
/// handoff, that it takes images, that it processes references itself.
#[derive(Clone, Copy)]
struct Member {
    url: &'static str,
    role: WorkerType,
    runtime: RuntimeType,
    states_end: bool,
    vision: bool,
    references: bool,
}

impl Member {
    /// A worker of `runtime` that states its end and takes images.
    const fn of(url: &'static str, role: WorkerType, runtime: RuntimeType) -> Self {
        Self {
            url,
            role,
            runtime,
            states_end: true,
            vision: true,
            references: false,
        }
    }
}

/// One prefill and one decode of `runtime` around `encoders`.
fn fleet_around(runtime: RuntimeType, encoders: &[Member]) -> Vec<Member> {
    let mut fleet = vec![
        Member::of(PREFILL, WorkerType::Prefill, runtime),
        Member::of(DECODE, WorkerType::Decode, runtime),
    ];
    fleet.extend_from_slice(encoders);
    fleet
}

/// [`fleet_around`] with four encoders of `runtime`.
fn fleet_of(runtime: RuntimeType) -> Vec<Member> {
    let urls = [
        ENCODER,
        "http://e1:8000",
        "http://e2:8000",
        "http://e3:8000",
    ];
    let encoders = urls.map(|url| Member::of(url, WorkerType::Encode, runtime));
    fleet_around(runtime, &encoders)
}

/// The stage in the encode mode over a host whose rows are `fleet` under a
/// document of the encode plan: one gRPC worker a row. The rows report no
/// capacity and no room, so each side of the pair keeps the two filters
/// that read neither.
fn encode_stage(fleet: &[Member]) -> anyhow::Result<(WorkerSelectionStage, Arc<MemoryStore>)> {
    let store = Arc::new(MemoryStore::new());
    let mut document = PolicySpec::epd_presets("consistent_hashing", "least_load", "least_load")?;
    for side in ["prefill", "decode"] {
        let profile = document
            .profiles
            .get_mut(side)
            .context("the side's profile")?;
        profile.filter = vec!["healthy".into(), "not-tried".into()];
    }
    store.apply(AppliedSpec::validate(&document)?)?;
    let registry = Arc::new(WorkerRegistry::new());
    for member in fleet {
        let pushes = member.runtime == PUSH;
        let stated = pushes && member.states_end;
        let role = match member.role {
            WorkerType::Encode => Role::Encode,
            WorkerType::Prefill => Role::Prefill,
            _ => Role::Decode,
        };
        let facts = WorkerFacts {
            key: member.url.to_owned(),
            engine: member.url.to_owned(),
            engine_kind: format!("{:?}", member.runtime),
            incarnation: 1,
            role,
            wire_modes: WireModes::GRPC,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(64),
            vision: member.vision,
            encode: EncodeOffer {
                handoff: if pushes {
                    OfferedHandoff::Push
                } else {
                    OfferedHandoff::Rendezvous
                },
                producer: stated && role == Role::Encode,
                consumer: stated && role == Role::Prefill,
                connector: stated.then(|| "c0".to_owned()),
                references: member.references,
            },
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        let row = WorkersStore::get(&*store, id).context("the row")?;
        if row.status != RowStatus::Ready {
            store
                .set_status(id, RowStatus::Ready, row.revision)
                .context("the row stands")?;
        }
        let mut builder = BasicWorkerBuilder::new(member.url)
            .model(ModelCard::new(MODEL).with_context_length(WINDOW))
            .worker_type(member.role)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(member.runtime)
            .health_config(no_health_check());
        if !member.vision {
            builder = builder.label("supports_vision", "false");
        }
        if member.references {
            builder = builder.label("mm_processor", "inprocess");
        }
        assert!(registry.register(Arc::new(builder.build())).is_some());
    }
    store.publish_now();
    let host = Host::at_keys(Arc::clone(&store), Arc::clone(&registry));
    let stage = WorkerSelectionStage::new(
        registry,
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        WorkerSelectionMode::EncodePrefillDecode,
        None,
    )
    .with_host(Some(Arc::new(host)));
    Ok((stage, store))
}

/// Turns the row of `url` not ready, where a decision reads it.
fn close(store: &MemoryStore, url: &str) -> anyhow::Result<()> {
    let id = store.by_key(url).context("the worker's row")?.id;
    let row = WorkersStore::get(store, id).context("the row")?;
    store
        .set_status(id, RowStatus::NotReady, row.revision)
        .context("the row turns")?;
    store.publish_now();
    Ok(())
}

/// A first attempt's encode pair for `items` under `demand`, as the stage's
/// helper answers it: the selection and any placement of the controller.
async fn encode_attempt(
    stage: &WorkerSelectionStage,
    items: &[Vec<u8>],
    demand: EpdDemands,
) -> Result<EpdAttempt, Response> {
    let ctx = ingress(&stage.worker_registry);
    let inputs = PlacementInputs::default();
    let first = Attempt::First(&ctx);
    let attempt = stage.select_epd_attempt(MODEL, inputs, None, items, demand, first);
    attempt.await.map(|(attempt, _guard)| attempt)
}

/// A response's status and its body.
async fn status_and_body(response: Response) -> anyhow::Result<(StatusCode, Value)> {
    let status = response.status();
    let body = to_bytes(response.into_body(), 4096).await?;
    Ok((status, serde_json::from_slice(&body)?))
}

/// A request with items under the host is one decision: its pair, and an
/// encoder for each item in the items' order. The classic placement is not
/// asked: the selection is the controller's placement, under one lease.
#[tokio::test]
async fn a_hosted_encode_request_gets_its_pair_and_one_assignment_an_item() -> anyhow::Result<()> {
    let (stage, store) = encode_stage(&fleet_of(MEET))?;
    let items = [vec![1], vec![2], vec![3]];
    let ((assignments, prefill, decode, runtime), placed) = encode_attempt(&stage, &items, IMAGES)
        .await
        .ok()
        .context("a placed encode pair")?;
    let placed = placed.context("the controller's placement")?;
    assert_eq!(
        (prefill.url(), decode.url(), runtime),
        (PREFILL, DECODE, MEET)
    );
    assert!(Arc::ptr_eq(&prefill, placed.worker()));
    assert!(!placed.one_key());
    assert_eq!(assignments.len(), items.len());
    for (at, (assignment, encoder)) in assignments.iter().zip(placed.encoders()).enumerate() {
        assert_eq!(assignment.item_index, at, "the items' order");
        assert!(Arc::ptr_eq(&assignment.worker, &encoder.worker));
        assert_eq!(assignment.worker.worker_type(), &WorkerType::Encode);
    }
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    drop(placed);

    // Through the stage: a request with one processed image.
    let (stage, store) = encode_stage(&fleet_of(PUSH))?;
    let mut ctx = ingress(&stage.worker_registry);
    ctx.state.multimodal_intermediate = Some(image_intermediate(None));
    stage.execute(&mut ctx).await.expect("a placed encode pair");
    let placed = ctx.state.placed.as_ref().context("the placement")?;
    assert_eq!(placed.encoders().len(), 1);
    assert!(ctx.state.pd_prefill_guard.is_some());
    let Some(WorkerSelection::Disaggregated {
        encode_assignments: Some(assignments),
        prefill,
        ..
    }) = ctx.state.workers.as_ref()
    else {
        panic!("a pair with its assignments");
    };
    assert!(Arc::ptr_eq(prefill, placed.worker()));
    assert_eq!(assignments.len(), 1);
    assert!(Arc::ptr_eq(
        &assignments[0].worker,
        &placed.encoders()[0].worker
    ));
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    Ok(())
}

/// Equal items carry one key, so one encoder takes them, and the keys of a
/// fleet spread over more encoders than one.
#[tokio::test]
async fn equal_items_are_assigned_one_encoder() -> anyhow::Result<()> {
    let (stage, _store) = encode_stage(&fleet_of(MEET))?;
    let mut taken = Vec::new();
    for key in 0..16_u8 {
        let items = [vec![key], vec![key, key], vec![key]];
        let ((assignments, ..), _placed) = encode_attempt(&stage, &items, IMAGES)
            .await
            .ok()
            .context("a placed encode pair")?;
        assert_eq!(assignments.len(), items.len());
        let first = assignments[0].worker.url();
        assert_eq!(first, assignments[2].worker.url(), "{key}");
        if !taken.iter().any(|url| url == first) {
            taken.push(first.to_owned());
        }
    }
    assert!(taken.len() > 1, "every key landed on {taken:?}");
    Ok(())
}

/// Under a class that pushes the one key is placed in the items' stead:
/// one encoder stands for every item, in the form the classic hands on.
/// Under an image fan-out no one key is stated, and each item has its own.
#[tokio::test]
async fn under_a_class_that_pushes_one_encoder_stands_for_every_item() -> anyhow::Result<()> {
    let (stage, _store) = encode_stage(&fleet_of(PUSH))?;
    let items = [vec![1], vec![2], vec![3]];
    let ((assignments, _, _, runtime), placed) = encode_attempt(&stage, &items, IMAGES)
        .await
        .ok()
        .context("a placed encode pair")?;
    assert_eq!(runtime, PUSH);
    assert!(placed.as_ref().is_some_and(Placed::one_key));
    let places: Vec<usize> = assignments.iter().map(|one| one.item_index).collect();
    assert_eq!(places, [0]);
    drop(placed);

    let fanout = EpdDemands {
        image_fanout: true,
        ..IMAGES
    };
    let ((assignments, ..), placed) = encode_attempt(&stage, &items, fanout)
        .await
        .ok()
        .context("a placed encode pair")?;
    assert!(placed.as_ref().is_some_and(|placed| !placed.one_key()));
    let places: Vec<usize> = assignments.iter().map(|one| one.item_index).collect();
    assert_eq!(places, [0, 1, 2]);
    Ok(())
}

/// A shed of the controller's encode side is answered under the classic's
/// own name: the status and the body of the classic's answer for it. A
/// shed of the pair's own reads the host's answer. No lease is left.
#[tokio::test]
async fn each_shed_of_the_encode_side_is_answered_under_the_classics_name() -> anyhow::Result<()> {
    type Name = fn() -> EpdSelectionFailure;
    let references = EpdDemands {
        encode: MediaDemands {
            media_refs: true,
            vision: false,
            pipeline: PipelinePin::Any,
        },
        ..IMAGES
    };
    let decode_images = EpdDemands {
        decode_vision: true,
        ..IMAGES
    };
    let encoder = Member::of(ENCODER, WorkerType::Encode, MEET);
    let mute = Member {
        states_end: false,
        ..Member::of(ENCODER, WorkerType::Encode, PUSH)
    };
    let blind_decode = vec![
        Member::of(PREFILL, WorkerType::Prefill, MEET),
        Member {
            vision: false,
            ..Member::of(DECODE, WorkerType::Decode, MEET)
        },
        encoder,
    ];
    let rows: [(&str, Vec<Member>, EpdDemands, Name); 3] = [
        // No registered encoder processes a reference.
        (
            "encode-media",
            fleet_around(MEET, &[encoder]),
            references,
            || EpdSelectionFailure::EncodeNotSupported,
        ),
        // A pushing encoder states no role.
        ("ec-compatible", fleet_around(PUSH, &[mute]), IMAGES, || {
            EpdSelectionFailure::EcIncompatible
        }),
        // No decode of the pair takes the images asked of it.
        ("decode-vision", blind_decode, decode_images, || {
            EpdSelectionFailure::DecodeVisionUnsupported
        }),
    ];
    for (shed, fleet, demand, name) in rows {
        let (stage, store) = encode_stage(&fleet)?;
        let refused = encode_attempt(&stage, &[vec![1]], demand)
            .await
            .err()
            .with_context(|| format!("{shed}: the request was placed"))?;
        let classic = stage.epd_selection_failure(MODEL, name(), true);
        assert_eq!(
            status_and_body(refused).await?,
            status_and_body(classic).await?,
            "{shed}"
        );
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0, "{shed}");
    }

    // The engine names `encode-vision` from the request's images, which no
    // ask states yet: the name is held at the table.
    let unseen = host::Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "encode-vision",
        retry_after_secs: 0,
    };
    assert!(matches!(
        epd_shed(&unseen),
        Some(EpdSelectionFailure::EncodeVisionUnsupported)
    ));

    // An encoder is registered and none is open: the classic's verdict by
    // role.
    let (stage, store) = encode_stage(&fleet_around(MEET, &[encoder]))?;
    close(&store, ENCODER)?;
    let refused = encode_attempt(&stage, &[vec![1]], IMAGES)
        .await
        .err()
        .context("encoders: the request was placed")?;
    let classic = stage.epd_selection_failure(MODEL, EpdSelectionFailure::Unavailable, true);
    assert_eq!(
        status_and_body(refused).await?,
        status_and_body(classic).await?,
        "encoders"
    );

    // A shed of the pair's own is the host's answer, as the pair stage reads
    // it.
    let (stage, store) = encode_stage(&fleet_around(MEET, &[encoder]))?;
    close(&store, PREFILL)?;
    let refused = encode_attempt(&stage, &[vec![1]], IMAGES)
        .await
        .err()
        .context("the pair: the request was placed")?;
    let hosts = host::Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 0,
    }
    .answer(MODEL);
    assert_eq!(refused.status(), hosts.status());
    assert_eq!(
        error::extract_error_code_from_response(&refused),
        error::extract_error_code_from_response(&hosts)
    );
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    Ok(())
}

/// The ask states no pixel media yet, so an encoder that takes no image can
/// be placed for a request with images. The gateway's own read of the
/// encoder refuses it: the placement is let go as a failed commit, and the
/// request reads the host's answer for an attempt no target holds.
#[tokio::test]
async fn an_encoder_the_route_cannot_send_to_is_given_up_as_a_failed_commit() -> anyhow::Result<()>
{
    let blind = Member {
        vision: false,
        ..Member::of(ENCODER, WorkerType::Encode, MEET)
    };
    let (stage, store) = encode_stage(&fleet_around(MEET, &[blind]))?;
    let refused = encode_attempt(&stage, &[vec![1]], IMAGES)
        .await
        .err()
        .context("the request was placed")?;
    assert_no_target_holds(&refused, &store);
    Ok(())
}

/// A request without items under the encode plan is its pair alone: no
/// encoder is placed and no assignment is handed on.
#[tokio::test]
async fn a_request_without_items_under_the_encode_plan_is_placed_as_its_pair() -> anyhow::Result<()>
{
    let (stage, store) = encode_stage(&fleet_of(PUSH))?;
    let mut ctx = ingress(&stage.worker_registry);
    stage.execute(&mut ctx).await.expect("a placed pair");
    let placed = ctx.state.placed.as_ref().context("the placement")?;
    assert!(placed.decode().is_some() && placed.encoders().is_empty());
    assert!(ctx.state.pd_prefill_guard.is_some());
    assert!(matches!(
        &ctx.state.workers,
        Some(WorkerSelection::Disaggregated { encode_assignments: None, prefill, decode, .. })
            if prefill.url() == PREFILL && decode.url() == DECODE
    ));
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);
    Ok(())
}

/// Without a host the stage's encode mode is the classic selection: it
/// places the items itself, leaves no placement and asks no decision.
#[tokio::test]
async fn the_classic_encode_stage_is_unmoved_without_a_host() -> anyhow::Result<()> {
    let (hosted, store) = encode_stage(&fleet_of(MEET))?;
    let classic = WorkerSelectionStage::new(
        Arc::clone(&hosted.worker_registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        WorkerSelectionMode::EncodePrefillDecode,
        None,
    );
    let items = [vec![1], vec![2]];
    let ((assignments, prefill, decode, _), placed) = encode_attempt(&classic, &items, IMAGES)
        .await
        .ok()
        .context("the classic encode pair")?;
    assert!(placed.is_none());
    assert_eq!(assignments.len(), items.len());
    assert_eq!((prefill.url(), decode.url()), (PREFILL, DECODE));
    // No decision was asked of the controller.
    assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    Ok(())
}
