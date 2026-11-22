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
    own_stage_under("round_robin", registry, policy)
}

/// [`own_stage`] over the engine of the stock document `preset`.
fn own_stage_under(
    preset: &str,
    registry: &Arc<WorkerRegistry>,
    policy: PolicyConfig,
) -> WorkerSelectionStage {
    let mut document = PolicySpec::preset(preset).expect("a built-in preset");
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
    asking(registry, vec![1; tokens])
}

/// A first attempt whose prompt is the token ids `tokens`.
fn asking(registry: &Arc<WorkerRegistry>, tokens: Vec<u32>) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: None,
        token_ids: tokens,
    });
    ctx
}

/// A first attempt whose prompt is `text`, with no token ids.
fn asking_by_text(registry: &Arc<WorkerRegistry>, text: &str) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: Some(text.to_owned()),
        token_ids: Vec::new(),
    });
    ctx
}

/// The place in `workers` of the worker `stage` selects for a first attempt
/// whose prompt is the token ids `tokens`.
async fn placed_at(
    stage: &WorkerSelectionStage,
    registry: &Arc<WorkerRegistry>,
    workers: &[Arc<dyn Worker>],
    tokens: Vec<u32>,
) -> usize {
    placed(stage, workers, asking(registry, tokens)).await
}

/// The place in `workers` of the worker `stage` selects for the attempt
/// `ctx`.
async fn placed(
    stage: &WorkerSelectionStage,
    workers: &[Arc<dyn Worker>],
    mut ctx: RequestContext,
) -> usize {
    stage.execute(&mut ctx).await.expect("a worker");
    let chosen = single(ctx.state.workers.as_ref());
    let at = workers
        .iter()
        .position(|worker| Arc::ptr_eq(worker, chosen));
    at.expect("a worker the registry lists")
}

/// A first attempt that names the worker at place `target` in its header.
fn naming(registry: &Arc<WorkerRegistry>, target: &'static str) -> RequestContext {
    let mut ctx = prompt_of(registry, 3);
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-target-worker", HeaderValue::from_static(target));
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

/// Where the model's own policy keeps an explicit target, the request is
/// held to the worker its header names, and the engine decides over that
/// worker alone. The number counts the workers that are available, as the
/// consistent hashing policy counts them, so both name one worker. A number
/// past them is no worker, with one answer. Under a policy that reads no
/// target the header holds the request to nobody.
#[tokio::test]
async fn an_explicit_target_holds_the_request_to_the_worker_it_names() {
    let registry = Arc::new(WorkerRegistry::new());
    let vllm = RuntimeType::Vllm;
    let workers = ["grpc://w0:8000", "grpc://w1:8000", "grpc://w2:8000"]
        .map(|url| listed(&registry, url, vllm, None, UNLABELED));
    workers[0].set_status(WorkerStatus::NotReady);
    let stage = own_stage(&registry, PolicyConfig::ConsistentHashing);
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

    let reads_none = own_stage(&registry, PolicyConfig::RoundRobin);
    let mut turns = [0; 3];
    for _ in 0..PLACEMENTS {
        let mut ctx = naming(&registry, "1");
        reads_none.execute(&mut ctx).await.expect("a worker");
        let on = single(ctx.state.workers.as_ref());
        let place = workers.iter().position(|worker| Arc::ptr_eq(worker, on));
        turns[place.expect("a worker of the pool")] += 1;
    }
    assert!(turns[0] == 0 && turns[1] > 0 && turns[2] > 0, "{turns:?}");
}

/// A conversation's next turn goes back to the worker its last turn went
/// to, where the engine of this process decides under the default policy.
/// The stage states the prompt's token ids on its ask, and the engine walks
/// and writes its record of sent prefixes with them: an ask that stated no
/// prompt would leave that record empty, and the turns would scatter. Other
/// conversations are not held to that worker.
#[tokio::test]
async fn a_conversations_next_turn_goes_back_to_its_worker() {
    let registry = Arc::new(WorkerRegistry::new());
    let workers: Vec<Arc<dyn Worker>> = (0..4)
        .map(|at| {
            let url = format!("grpc://worker-{at}:8000");
            listed(&registry, &url, RuntimeType::Vllm, None, UNLABELED)
        })
        .collect();
    let stage = own_stage_under("cache_aware", &registry, PolicyConfig::RoundRobin);
    let mut conversation: Vec<u32> = (0..1_024).collect();
    let first = placed_at(&stage, &registry, &workers, conversation.clone()).await;
    for turn in 1..=16_u32 {
        let said = 1_000_000 * turn;
        conversation.extend(said..said + 64);
        let next = placed_at(&stage, &registry, &workers, conversation.clone()).await;
        assert_eq!(next, first, "turn {turn}");
    }
    let mut others = Vec::new();
    for other in 1..=16_u32 {
        let begins = 100_000_000 * other;
        let prompt: Vec<u32> = (begins..begins + 1_024).collect();
        others.push(placed_at(&stage, &registry, &workers, prompt).await);
    }
    assert!(others.iter().any(|&at| at != others[0]), "{others:?}");
}

/// A request with no token ids is held to its worker by its text. The stage
/// states the text beside the ids on its ask, and the engine walks and
/// writes its record of sent prefixes by the text, as the classic policy
/// does where a request has no ids: an ask that stated the ids alone would
/// read no record, and the turns would scatter. Other texts are not held to
/// that worker.
#[tokio::test]
async fn a_conversation_with_no_token_ids_goes_back_to_its_worker() {
    // A text of `bytes` bytes that differs from another speaker's within
    // its first eight.
    let said = |speaker: u32, bytes: usize| -> String {
        let word = format!("{speaker:07}:");
        word.chars().cycle().take(bytes).collect()
    };
    let registry = Arc::new(WorkerRegistry::new());
    let workers: Vec<Arc<dyn Worker>> = (0..4)
        .map(|at| {
            let url = format!("grpc://worker-{at}:8000");
            listed(&registry, &url, RuntimeType::Vllm, None, UNLABELED)
        })
        .collect();
    let stage = own_stage_under("cache_aware", &registry, PolicyConfig::RoundRobin);
    let mut conversation = said(0, 4_096);
    let first = placed(&stage, &workers, asking_by_text(&registry, &conversation)).await;
    for turn in 1..=16_u32 {
        conversation.push_str(&said(turn, 256));
        let next = placed(&stage, &workers, asking_by_text(&registry, &conversation)).await;
        assert_eq!(next, first, "turn {turn}");
    }
    let mut others = Vec::new();
    for other in 1..=16_u32 {
        let prompt = said(1_000 * other, 4_096);
        others.push(placed(&stage, &workers, asking_by_text(&registry, &prompt)).await);
    }
    assert!(others.iter().any(|&at| at != others[0]), "{others:?}");
}
