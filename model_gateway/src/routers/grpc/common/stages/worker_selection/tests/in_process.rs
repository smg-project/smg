//! Regular selection and re-selection where the policy engine of this
//! process decides. A demand of the request narrows the workers the engine
//! decides over, as it narrows the candidates of the selection with no
//! policy engine, and no shape of request is refused.

use axum::body::to_bytes;
use openai_protocol::worker::WorkerStatus;
use serde_json::Value;
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::PolicyEngine as OwnEngine,
};

use super::*;
use crate::{
    config::{ManualAssignmentMode, RouterConfig},
    policies::DecidedByEngine,
    policy_controller::start_in_process,
};

const MODEL: &str = "test-model";
const WINDOW: u32 = 16;
const PLACEMENTS: usize = 8;

/// The labels of a worker that processes media references itself.
const TAKES_REFERENCES: &[(&str, &str)] = &[
    ("mm_processor", "inprocess"),
    ("mm_media_ref_schemes", "http,https,data"),
];

/// The regular stage of `registry` with no policy engine, the model's policy
/// being `policy`.
fn classic_stage(registry: &Arc<WorkerRegistry>, policy: PolicyConfig) -> WorkerSelectionStage {
    WorkerSelectionStage::new(
        Arc::clone(registry),
        Arc::new(PolicyRegistry::new(policy)),
        WorkerSelectionMode::Regular,
        None,
    )
}

/// The same stage over the round robin engine of this process. Its attempts
/// ask that engine, and the policy registry only says what the model's own
/// policy keeps of a request.
fn own_stage(registry: &Arc<WorkerRegistry>, policy: PolicyConfig) -> WorkerSelectionStage {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(registry));
    classic_stage(registry, policy).with_policy_engine(Some(Arc::new(engine)))
}

/// A ready regular worker of [`MODEL`] on gRPC in `registry`, of `runtime`,
/// with `labels` and the `window` it states.
fn listed(
    registry: &WorkerRegistry,
    url: &str,
    runtime: RuntimeType,
    window: Option<u32>,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    let card = ModelCard::new(MODEL);
    let mut builder = BasicWorkerBuilder::new(url)
        .model(match window {
            Some(window) => card.with_context_length(window),
            None => card,
        })
        .worker_type(WorkerType::Regular)
        .runtime_type(runtime)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(no_health_check());
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    worker.set_status(WorkerStatus::Ready);
    assert!(registry.register(Arc::clone(&worker)).is_some());
    worker
}

/// A first attempt whose prompt is `tokens` tokens long.
fn prompt_of(registry: &Arc<WorkerRegistry>, tokens: usize) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: None,
        token_ids: vec![1; tokens],
    });
    ctx
}

/// A first attempt that names the worker at place `target` in its header.
fn naming(registry: &Arc<WorkerRegistry>, target: &'static str) -> RequestContext {
    let mut ctx = prompt_of(registry, 3);
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-target-worker", HeaderValue::from_static(target));
    ctx.input.headers = Some(headers);
    ctx
}

/// A first attempt that carries the routing key `key` in its header.
fn keyed(registry: &Arc<WorkerRegistry>, key: &str) -> RequestContext {
    let mut ctx = prompt_of(registry, 3);
    let mut headers = HeaderMap::new();
    let key = HeaderValue::from_str(key).expect("a header's value");
    headers.insert("x-smg-routing-key", key);
    ctx.input.headers = Some(headers);
    ctx
}

/// The one worker a selection left in `workers`.
fn single(workers: Option<&WorkerSelection>) -> &Arc<dyn Worker> {
    match workers {
        Some(WorkerSelection::Single { worker }) => worker,
        _ => panic!("a regular selection"),
    }
}

/// The worker a retry of `ctx` is placed on by `stage`.
async fn retried(stage: &WorkerSelectionStage, ctx: &mut DispatchContext) -> Arc<dyn Worker> {
    let reselected = stage.reselect(ctx, &retained_plan(&[1, 2, 3])).await;
    reselected.expect("a worker for the retry");
    assert!(ctx.placed.is_some(), "the engine placed the retry");
    Arc::clone(single(ctx.workers.as_ref()))
}

/// The status and the body of an answer.
async fn answer_of(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = to_bytes(response.into_body(), 4096).await;
    let body = body.expect("the answer's body");
    (status, serde_json::from_slice(&body).expect("a JSON body"))
}

/// A prompt past one worker's window is placed by the engine on the worker
/// whose window holds it, every time. A prompt every window holds narrows
/// nothing: the engine decides over the registry's own list, and each
/// worker takes its turn.
#[tokio::test]
async fn a_context_demand_narrows_the_workers_the_engine_decides_over() {
    let registry = Arc::new(WorkerRegistry::new());
    let vllm = RuntimeType::Vllm;
    let _small = listed(
        &registry,
        "grpc://small:8000",
        vllm,
        Some(WINDOW),
        UNLABELED,
    );
    let large = listed(&registry, "grpc://large:8000", vllm, None, UNLABELED);
    let stage = own_stage(&registry, PolicyConfig::RoundRobin);
    for _ in 0..PLACEMENTS {
        let mut ctx = prompt_of(&registry, WINDOW as usize + 1);
        stage.execute(&mut ctx).await.expect("a window holds it");
        assert!(ctx.state.placed.is_some(), "the engine placed it");
        assert!(Arc::ptr_eq(single(ctx.state.workers.as_ref()), &large));
    }
    let mut turns = [0; 2];
    for _ in 0..PLACEMENTS {
        let mut ctx = prompt_of(&registry, 3);
        stage
            .execute(&mut ctx)
            .await
            .expect("every window holds it");
        let on_large = Arc::ptr_eq(single(ctx.state.workers.as_ref()), &large);
        turns[usize::from(on_large)] += 1;
    }
    assert!(turns.iter().all(|&taken| taken > 0), "{turns:?}");
}

/// A prompt no window holds is the client's to fix: the answer is that of
/// the selection with no policy engine over the same workers, whole. Where
/// the one worker that holds the prompt is not ready, the request reads a
/// 503 and no worker's window is passed over.
#[tokio::test]
async fn a_demand_no_worker_holds_reads_the_answer_with_no_policy_engine() {
    let registry = Arc::new(WorkerRegistry::new());
    let vllm = RuntimeType::Vllm;
    let _small = listed(
        &registry,
        "grpc://small:8000",
        vllm,
        Some(WINDOW),
        UNLABELED,
    );
    let stage = own_stage(&registry, PolicyConfig::RoundRobin);
    let classic = classic_stage(&registry, PolicyConfig::RoundRobin);
    let past = || prompt_of(&registry, WINDOW as usize + 1);

    let refused = stage.execute(&mut past()).await;
    let own = answer_of(refused.expect_err("no window holds the prompt")).await;
    assert_eq!(own.0, StatusCode::BAD_REQUEST);
    assert_eq!(own.1["error"]["code"], "context_length_exceeded");
    let refused = classic.execute(&mut past()).await;
    assert_eq!(answer_of(refused.expect_err("no window")).await, own);

    let large = listed(&registry, "grpc://large:8000", vllm, None, UNLABELED);
    large.set_status(WorkerStatus::NotReady);
    let refused = stage.execute(&mut past()).await;
    let refused = refused.expect_err("the worker that holds it is not ready");
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// Pixel input is placed on the worker with a vision encoder, and a retry
/// whose retained plan forwards media references on the worker that
/// processes them.
#[tokio::test]
async fn a_media_input_narrows_the_workers_the_engine_decides_over() {
    let registry = Arc::new(WorkerRegistry::new());
    let vllm = RuntimeType::Vllm;
    let _blind = listed(
        &registry,
        "grpc://blind:8000",
        vllm,
        None,
        NO_VISION_ENCODER,
    );
    let sees = listed(&registry, "grpc://sees:8000", vllm, None, VISION_ENCODER);
    let stage = own_stage(&registry, PolicyConfig::RoundRobin);
    for _ in 0..PLACEMENTS {
        let mut ctx = prompt_of(&registry, 3);
        ctx.state.multimodal_intermediate = Some(image_intermediate(None));
        stage.execute(&mut ctx).await.expect("a vision encoder");
        assert!(ctx.state.placed.is_some(), "the engine placed it");
        assert!(Arc::ptr_eq(single(ctx.state.workers.as_ref()), &sees));
    }

    let registry = Arc::new(WorkerRegistry::new());
    let _plain = listed(&registry, "grpc://plain:8000", vllm, None, UNLABELED);
    let takes = listed(&registry, "grpc://takes:8000", vllm, None, TAKES_REFERENCES);
    let stage = own_stage(&registry, PolicyConfig::RoundRobin);
    for _ in 0..PLACEMENTS {
        let mut retry = dispatch_ctx(MODEL, grpc_wire(vllm));
        retry.wire.requires_media_refs = true;
        assert!(stage.places_retry(&retry));
        assert!(Arc::ptr_eq(&retried(&stage, &mut retry).await, &takes));
    }
}

/// A retry under a decoding rule stays on the workers that decide the
/// request's constraint as its first attempt's worker did, and each of them
/// takes its turn.
#[tokio::test]
async fn a_retry_under_a_decoding_rule_stays_on_the_workers_that_decide_alike() {
    let registry = Arc::new(WorkerRegistry::new());
    let runtime = RuntimeType::TokenSpeed;
    let always = ("constrained_decoding", "always");
    let first_token = ("constrained_decoding_start", "first_token");
    let after_reasoning = ("constrained_decoding_start", "after_reasoning");
    let workers = [
        ("grpc://w0:8000", &[always, first_token][..]),
        ("grpc://w1:8000", &[][..]),
        ("grpc://w2:8000", &[always, after_reasoning][..]),
        ("grpc://w3:8000", &[always, first_token][..]),
    ]
    .map(|(url, labels)| listed(&registry, url, runtime, None, labels));
    let stage = own_stage(&registry, PolicyConfig::RoundRobin);
    let decided_on = |worker: &Arc<dyn Worker>| {
        utils::WorkerDecoding::of(Some(&WorkerSelection::Single {
            worker: Arc::clone(worker),
        }))
    };

    let mut retry = dispatch_ctx(MODEL, grpc_wire(runtime));
    retry.constraint_decided_on = Some(decided_on(&workers[0]));
    assert!(stage.places_retry(&retry));
    let mut turns = [0; 4];
    for _ in 0..4 * PLACEMENTS {
        let on = retried(&stage, &mut retry).await;
        let place = workers.iter().position(|worker| Arc::ptr_eq(worker, &on));
        turns[place.expect("a worker of the pool")] += 1;
    }
    assert_eq!(turns, [2 * PLACEMENTS, 0, 0, 2 * PLACEMENTS]);

    retry.constraint_decided_on = Some(decided_on(&workers[2]));
    for _ in 0..PLACEMENTS {
        assert!(Arc::ptr_eq(&retried(&stage, &mut retry).await, &workers[2]));
    }
}

/// Where the document in force places the model by the plain hash ring,
/// the request is held to the worker its header names, and the engine
/// decides over that worker alone. The document alone says it: with the
/// setting on under `consistent_hashing` the registry builds no classic
/// policy, and none is asked. The number counts the workers that are
/// available, as the consistent hashing policy counts them, so both name
/// one worker. A number past them is no worker, with one answer. Under a
/// document that reads no target the header holds the request to nobody:
/// under round robin's, where the stage's classic registry holds a
/// consistent hashing policy for the model, and under `prefix_hash`.
#[tokio::test]
async fn an_explicit_target_holds_the_request_to_the_worker_it_names() {
    let registry = Arc::new(WorkerRegistry::new());
    let vllm = RuntimeType::Vllm;
    let workers = ["grpc://w0:8000", "grpc://w1:8000", "grpc://w2:8000"]
        .map(|url| listed(&registry, url, vllm, None, UNLABELED));
    workers[0].set_status(WorkerStatus::NotReady);
    let (stage, policies) = started_stage(PolicyConfig::ConsistentHashing, &registry);
    let classic = classic_stage(&registry, PolicyConfig::ConsistentHashing);
    for _ in 0..PLACEMENTS {
        let mut ctx = naming(&registry, "1");
        stage.execute(&mut ctx).await.expect("the worker it names");
        assert!(ctx.state.placed.is_some(), "the engine placed it");
        // The second of the two workers that are available.
        assert!(Arc::ptr_eq(single(ctx.state.workers.as_ref()), &workers[2]));
        let mut ctx = naming(&registry, "1");
        classic
            .execute(&mut ctx)
            .await
            .expect("the worker it names");
        assert!(Arc::ptr_eq(single(ctx.state.workers.as_ref()), &workers[2]));
    }
    let mut retry = dispatch_ctx(MODEL, grpc_wire(vllm));
    retry.headers = naming(&registry, "1").input.headers;
    assert!(stage.places_retry(&retry));

    let missed = stage.execute(&mut naming(&registry, "2")).await;
    let own = answer_of(missed.expect_err("no worker at that place")).await;
    assert_eq!(own.0, StatusCode::SERVICE_UNAVAILABLE);
    let missed = classic.execute(&mut naming(&registry, "2")).await;
    assert_eq!(answer_of(missed.expect_err("no worker")).await, own);
    assert!(!classic_was_asked(&policies));

    let reads_none = own_stage(&registry, PolicyConfig::ConsistentHashing);
    let mut turns = [0; 3];
    for _ in 0..PLACEMENTS {
        let mut ctx = naming(&registry, "1");
        reads_none.execute(&mut ctx).await.expect("a worker");
        let on = single(ctx.state.workers.as_ref());
        let place = workers.iter().position(|worker| Arc::ptr_eq(worker, on));
        turns[place.expect("a worker of the pool")] += 1;
    }
    assert!(turns[0] == 0 && turns[1] > 0 && turns[2] > 0, "{turns:?}");

    let prefix_hash = PolicyConfig::PrefixHash {
        prefix_token_count: 256,
        load_factor: 1.25,
        balance_abs_threshold: 10,
        cache_boundaries: Vec::new(),
    };
    let (reads_none, policies) = started_stage(prefix_hash, &registry);
    for _ in 0..PLACEMENTS {
        // A number that names no worker: placed all the same.
        let mut ctx = naming(&registry, "2");
        reads_none.execute(&mut ctx).await.expect("a worker");
        assert!(ctx.state.placed.is_some(), "the engine placed it");
        let on = single(ctx.state.workers.as_ref());
        assert!(!Arc::ptr_eq(on, &workers[0]), "a worker that is not ready");
    }
    assert!(!classic_was_asked(&policies));
}

/// The regular stage as a gateway's start builds it under the manual policy
/// with the setting on: over the engine of the manual document, which
/// decides every request, with the policy registry the start builds beside
/// that engine. The registry too, for what it holds.
fn manual_stage(registry: &Arc<WorkerRegistry>) -> (WorkerSelectionStage, Arc<PolicyRegistry>) {
    let manual = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    started_stage(manual, registry)
}

/// [`manual_stage`] under `policy`, a policy whose every request the engine
/// decides. The engine object is handed the hash rings the registry keeps,
/// as the start hands them, and no task fills them here: a request fills
/// its membership's ring.
fn started_stage(
    policy: PolicyConfig,
    registry: &Arc<WorkerRegistry>,
) -> (WorkerSelectionStage, Arc<PolicyRegistry>) {
    let config = RouterConfig {
        policy_controller: true,
        policy,
        ..RouterConfig::default()
    };
    let started = start_in_process(&config, None)
        .expect("the document starts")
        .expect("the setting starts an engine");
    assert!(started.decides_whole());
    let policies = Arc::new(PolicyRegistry::with_policy_engine(
        config.policy.clone(),
        config.routing_key_override.clone(),
        Arc::clone(&started.engine),
        started.origin,
        true,
    ));
    let engine = PolicyEngine::own(started.engine, Arc::clone(registry));
    policies.share_rings_with(&engine);
    let stage = WorkerSelectionStage::new(
        Arc::clone(registry),
        Arc::clone(&policies),
        WorkerSelectionMode::Regular,
        None,
    )
    .with_policy_engine(Some(Arc::new(engine)));
    (stage, policies)
}

/// Whether a placement asked the classic policy of the registry's default
/// place, of which the registry holds the name alone.
fn classic_was_asked(policies: &PolicyRegistry) -> bool {
    let default = policies.get_default_policy();
    let held = default.as_any().downcast_ref::<DecidedByEngine>();
    let held = held.expect("the registry holds the policy's name alone");
    held.was_asked()
}

/// The worker `stage` places a first attempt under the routing key `key`
/// on, the placement told its dispatch: a key's pin is made there.
async fn dispatched_under(
    stage: &WorkerSelectionStage,
    registry: &Arc<WorkerRegistry>,
    key: &str,
) -> Arc<dyn Worker> {
    let mut ctx = prompt_of(registry, 3);
    let mut headers = HeaderMap::new();
    let key = HeaderValue::from_str(key).expect("a header value");
    headers.insert("x-smg-routing-key", key);
    ctx.input.headers = Some(headers);
    stage.execute(&mut ctx).await.expect("a worker");
    let placed = ctx.state.placed.as_mut();
    placed
        .expect("the engine placed it")
        .dispatched(Instant::now());
    Arc::clone(single(ctx.state.workers.as_ref()))
}

/// Under the manual policy the regular stage of the gRPC wire asks the
/// engine for every attempt. A key's requests follow the pin its first
/// dispatch made, and many keys reach every worker. The registry holds the
/// policy's name alone, with no classic policy behind it, and the stage
/// never asks it for a worker.
#[tokio::test]
async fn under_the_manual_policy_the_engine_places_and_no_classic_policy_is_asked() {
    let registry = Arc::new(WorkerRegistry::new());
    let workers = ["grpc://w0:8000", "grpc://w1:8000", "grpc://w2:8000"]
        .map(|url| listed(&registry, url, RuntimeType::Vllm, None, UNLABELED));
    let (stage, policies) = manual_stage(&registry);

    let pinned = dispatched_under(&stage, &registry, "conversation-0").await;
    for _ in 0..PLACEMENTS {
        let again = dispatched_under(&stage, &registry, "conversation-0").await;
        assert!(Arc::ptr_eq(&again, &pinned), "a key left its pin");
    }
    let mut keys = [0; 3];
    for key in 1..=48 {
        let on = dispatched_under(&stage, &registry, &format!("conversation-{key}")).await;
        let place = workers.iter().position(|worker| Arc::ptr_eq(worker, &on));
        keys[place.expect("a worker of the pool")] += 1;
    }
    assert!(keys.iter().all(|&taken| taken > 0), "{keys:?}");

    let default = policies.get_default_policy();
    let held = default.as_any().downcast_ref::<DecidedByEngine>();
    let held = held.expect("the registry holds the policy's name alone");
    assert!(!held.was_asked(), "the stage asked the classic policy");
}

/// On the gRPC pipeline's regular stage a routing key has one worker by the
/// engine of this process and by the classic consistent hashing policy:
/// both walk the one ring the gateway keeps for the membership. A worker
/// that is not ready is passed over by both, to the same next worker. The
/// registry holds the policy's name alone, and the stage never asks it.
#[tokio::test]
async fn a_routing_key_goes_to_the_worker_the_classic_ring_policy_gives_it() {
    let registry = Arc::new(WorkerRegistry::new());
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|at| {
            let url = format!("grpc://w{at}:8000");
            listed(&registry, &url, RuntimeType::Vllm, None, UNLABELED)
        })
        .collect();
    let (stage, policies) = started_stage(PolicyConfig::ConsistentHashing, &registry);
    let classic = classic_stage(&registry, PolicyConfig::ConsistentHashing);
    let mut taken = [0_u32; 8];
    for down in [None, Some(3), Some(5)] {
        for (at, worker) in workers.iter().enumerate() {
            worker.set_status(if Some(at) == down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
        }
        for key in 0..256 {
            let key = format!("conversation-{key}");
            let mut ctx = keyed(&registry, &key);
            stage.execute(&mut ctx).await.expect("a worker");
            assert!(ctx.state.placed.is_some(), "the engine placed it");
            let own = Arc::clone(single(ctx.state.workers.as_ref()));
            let mut ctx = keyed(&registry, &key);
            classic.execute(&mut ctx).await.expect("a worker");
            let by_classic = single(ctx.state.workers.as_ref());
            assert!(Arc::ptr_eq(&own, by_classic), "{key}, down {down:?}");
            let place = workers.iter().position(|worker| Arc::ptr_eq(worker, &own));
            taken[place.expect("a worker of the pool")] += 1;
        }
    }
    assert!(taken.iter().all(|&keys| keys > 0), "{taken:?}");
    assert!(!classic_was_asked(&policies));
}
