//! Regular selection and re-selection over a store with worker rows.

use anyhow::Context;
use axum::body::to_bytes;
use kv_index::{salt::request_content_hashes_with_seed, XXH3_SEED};
use llm_multimodal::MediaContentPart;
use serde_json::Value;
use smg_policy_controller::{
    spec::{KeySourceSpec, RoutingKeySpec},
    store::{
        BatchOutcome, GatewayId, KvEventSink, LeasesStore, LoadSink, LoadSource, MemoryStore,
        WorkersStore,
    },
    view::{fingerprint, LoadRow, WorkerId},
    AppliedSpec, PolicySpec,
};

use super::*;
use crate::{
    config::{PolicyConfig, RouterConfig, RoutingKeyOverrideConfig},
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
    let (_, store, rows) = host::testing::bed_with(store, Arc::new(WorkerRegistry::new()), rows);
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
    let ask = RegularAttempt::First(&ctx).ask(MODEL, &id, &hashes);
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
    let ask = RegularAttempt::Retry(&ctx).ask(MODEL, &id, &hashes);
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
    let ask = RegularAttempt::Retry(&ctx).ask(MODEL, &id, &hashes);
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
    for turn in 0..8 {
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

        stage.reselect(&mut ctx, &plan).await.expect("the retry");
        let retried = ctx.placed.take().expect("the retry's placement").id();
        assert_eq!(retried, by_key, "{rid}");
        assert_ne!(retried, first_row, "{rid}");
    }
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
