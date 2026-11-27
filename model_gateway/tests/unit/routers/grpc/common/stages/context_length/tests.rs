use std::sync::Arc;

use axum::http::StatusCode;
use openai_protocol::{
    chat::ChatCompletionRequest, model_card::ModelCard, validated::Normalizable,
    worker::HealthCheckConfig,
};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType};

use super::*;
use crate::routers::{
    common::snapshot::{WorkerRegistry, WorkerRegistryExt},
    error::HEADER_X_SMG_ERROR_CODE,
    grpc::{backend_client::BasicWorkerBuilder, context::CompletionItem},
};

const MODEL: &str = "ctx-test-model";

fn worker(worker_type: WorkerType, context_length: Option<u32>) -> Arc<dyn Worker> {
    kind(worker_type, ConnectionMode::Grpc, context_length)
}

fn kind(
    worker_type: WorkerType,
    mode: ConnectionMode,
    context_length: Option<u32>,
) -> Arc<dyn Worker> {
    let mut card = ModelCard::new(MODEL);
    if let Some(len) = context_length {
        card = card.with_context_length(len);
    }
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:1/{worker_type:?}"))
            .worker_type(worker_type)
            .connection_mode(mode)
            .runtime_type(RuntimeType::Sglang)
            .model(card)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// What the kept floors of a snapshot's pools leave of a demand:
/// nothing where every window of every pool holds it; the demand whole
/// where a window of one pool refuses it, and where a floor was kept for
/// another spelling of the model id.
#[test]
fn a_demand_is_left_whole_unless_every_pools_floor_holds_it() {
    let registry = WorkerRegistry::new();
    for (port, worker_type, window) in [
        (8861, WorkerType::Prefill, 64),
        (8862, WorkerType::Decode, 256),
    ] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                    .worker_type(worker_type)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Sglang)
                    .model(
                        ModelCard::new(MODEL)
                            .with_alias("ctx-test-alias")
                            .with_context_length(window),
                    )
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .unwrap();
    }
    let snapshot = registry.get_routing_snapshot(MODEL);
    let demand = |input_tokens, budget| ContextRequirements {
        input_tokens,
        scoring_position: false,
        output_tokens: Some(budget),
    };
    let nothing = ContextRequirements::default();
    let (prefill, decode) = (RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode);

    // Both floors hold it: the prefill window the input, the decode
    // window the input and the budget.
    let held = demand(64, 256);
    assert_eq!(held.left_by(&snapshot, &[prefill, decode], MODEL), nothing);
    // The prefill window refuses the input, which the decode window holds.
    let long = demand(65, 256);
    assert_eq!(long.left_by(&snapshot, &[prefill, decode], MODEL), long);
    assert_eq!(long.left_by(&snapshot, &[decode, prefill], MODEL), long);
    assert_eq!(long.left_by(&snapshot, &[decode], MODEL), nothing);
    // The decode window refuses the budget, which binds no prefill worker.
    let wide = demand(64, 257);
    assert_eq!(wide.left_by(&snapshot, &[prefill, decode], MODEL), wide);
    assert_eq!(wide.left_by(&snapshot, &[prefill], MODEL), nothing);
    // A floor kept for another spelling answers nothing: the workers
    // are asked.
    assert_eq!(
        held.left_by(&snapshot, &[prefill, decode], "ctx-test-alias"),
        held
    );
    // A pool with no worker binds nothing, and a demand that binds
    // nobody is left as it is.
    assert_eq!(
        long.left_by(&snapshot, &[RoutingPool::GrpcEncode], MODEL),
        nothing
    );
    assert_eq!(
        nothing.left_by(&snapshot, &[prefill, decode], "ctx-test-alias"),
        nothing
    );
}

fn single(context_length: Option<u32>) -> WorkerSelection {
    WorkerSelection::Single {
        worker: worker(WorkerType::Regular, context_length).into(),
    }
}

fn pd(prefill: Option<u32>, decode: Option<u32>) -> WorkerSelection {
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: worker(WorkerType::Prefill, prefill),
        decode: worker(WorkerType::Decode, decode).into(),
        runtime_type: RuntimeType::Sglang,
    }
}

fn input(tokens: usize) -> PreparationOutput {
    PreparationOutput::Generate {
        original_text: None,
        token_ids: vec![7; tokens],
    }
}

fn assert_rejected(result: Result<(), Response>) {
    let response = result.expect_err("over-long input must be rejected");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .and_then(|v| v.to_str().ok()),
        Some(CONTEXT_LENGTH_EXCEEDED)
    );
}

fn chat(model: &str, max_tokens: Option<u32>) -> RequestType {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    });
    if let Some(max_tokens) = max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    RequestType::Chat(Arc::new(
        serde_json::from_value(body).expect("chat request deserializes"),
    ))
}

fn completion(max_tokens: u32) -> RequestType {
    RequestType::Completion(Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "prompt": "hi", "max_tokens": max_tokens
        }))
        .expect("completion request deserializes"),
    ))
}

fn messages(max_tokens: u32) -> RequestType {
    RequestType::Messages(Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("messages request deserializes"),
    ))
}

fn responses(max_output_tokens: u32) -> RequestType {
    RequestType::Responses(Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "input": "hi", "max_output_tokens": max_output_tokens
        }))
        .expect("responses request deserializes"),
    ))
}

fn generate(max_new_tokens: u32) -> RequestType {
    RequestType::Generate(Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "text": "hi", "sampling_params": {"max_new_tokens": max_new_tokens}
        }))
        .expect("generate request deserializes"),
    ))
}

fn embedding() -> RequestType {
    RequestType::Embedding(Arc::new(
        serde_json::from_value(serde_json::json!({"model": MODEL, "input": "hi"}))
            .expect("embedding request deserializes"),
    ))
}

/// Selection shares both the budget bound and the overflow-safe scoring
/// boundary with the existing post-selection checks.
#[test]
fn context_requirements_preserve_budget_and_scoring_boundaries() {
    for budget in [None, Some(0), Some(16)] {
        let request = chat("ordinary-model", budget);
        let requirement = ContextRequirements::new(&input(16), &request);
        assert!(requirement.enforce(16, MODEL).is_ok());
    }
    let request = chat("ordinary-model", Some(17));
    assert_rejected(ContextRequirements::new(&input(16), &request).enforce(16, MODEL));
    let requirement = ContextRequirements {
        input_tokens: usize::MAX,
        scoring_position: true,
        output_tokens: Some(u32::MAX),
    };
    assert_rejected(requirement.enforce(u32::MAX, MODEL));
    let requirement = ContextRequirements {
        input_tokens: 16,
        scoring_position: true,
        output_tokens: None,
    };
    assert_rejected(requirement.enforce(16, MODEL));
    assert!(requirement.enforce(17, MODEL).is_ok());
}

/// A budget the window could never hold is refused whatever the model's
/// provider profile: the check the z.ai profile had is every profile's.
#[test]
fn a_budget_beyond_the_window_is_a_400_context_length_exceeded() {
    for model in [
        "glm-5.3-flash",
        "zai-org/GLM-5.3-Flash",
        "gpt-4o",
        "kimi-k3",
        "MiniMax-M3",
    ] {
        assert_rejected(enforce_output_budget(
            &chat(model, Some(10_000_000)),
            &single(Some(131_072)),
            MODEL,
        ));
        assert_rejected(enforce_output_budget(
            &chat(model, Some(131_073)),
            &pd(Some(1_000_000), Some(131_072)),
            MODEL,
        ));
    }
}

/// An omitted Flash budget leaves a smaller worker window available.
#[test]
fn glm53_flash_chat_omitted_budget_preserves_the_worker_window() {
    let mut request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    request.normalize();
    assert_eq!(request.output_token_cap(), None);
    let request = RequestType::Chat(Arc::new(request));
    assert!(enforce_output_budget(&request, &single(Some(32_768)), MODEL).is_ok());
}

#[test]
fn a_budget_within_the_window_or_without_a_window_passes() {
    for model in ["glm-5.3-flash", "gpt-4o"] {
        assert!(
            enforce_output_budget(&chat(model, Some(131_072)), &single(Some(131_072)), MODEL)
                .is_ok(),
            "{model}"
        );
        assert!(
            enforce_output_budget(&chat(model, None), &single(Some(131_072)), MODEL).is_ok(),
            "{model}"
        );
        assert!(
            enforce_output_budget(&chat(model, Some(10_000_000)), &single(None), MODEL).is_ok(),
            "{model}"
        );
    }
}

/// A route's request with the budget it names.
type Budgeted = fn(u32) -> RequestType;

/// Every route that names a completion budget is bounded by the window
/// (the budget alone: the input's share stays with the engine); a route
/// without one names nothing to bound.
#[test]
fn every_route_that_names_a_budget_is_bounded_by_the_window() {
    let routes: [(&str, Budgeted); 4] = [
        ("completion", completion),
        ("messages", messages),
        ("responses", responses),
        ("generate", generate),
    ];
    for (route, request) in routes {
        assert_rejected(enforce_output_budget(
            &request(17),
            &single(Some(16)),
            MODEL,
        ));
        assert!(
            enforce_output_budget(&request(16), &single(Some(16)), MODEL).is_ok(),
            "{route}: a budget that fits"
        );
        assert!(
            enforce_output_budget(&request(17), &single(None), MODEL).is_ok(),
            "{route}: an unknown window"
        );
    }
    assert!(enforce_output_budget(&embedding(), &single(Some(16)), MODEL).is_ok());
}

#[tokio::test]
async fn budget_rejection_body_names_the_limit_and_the_budget() {
    let response = enforce_output_budget(&chat("any-model", Some(40)), &single(Some(16)), MODEL)
        .expect_err("a budget beyond the window must be rejected");
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
    assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED);
    let message = json["error"]["message"].as_str().expect("message");
    assert!(message.contains("16 tokens"), "{message}");
    assert!(
        message.contains("40 tokens for the completion"),
        "{message}"
    );
    assert!(message.contains("max_tokens"), "{message}");
}

#[test]
fn input_within_the_window_passes() {
    assert!(enforce_context_length(&input(8), &single(Some(16)), MODEL).is_ok());
}

/// The window is a capacity, not a strict bound: a prompt that exactly
/// fills it is not "longer than" it, and whether the engine can still
/// generate from it is the engine's call (embeddings, for one, can).
#[test]
fn input_exactly_at_the_window_passes() {
    assert!(enforce_context_length(&input(16), &single(Some(16)), MODEL).is_ok());
}

#[test]
fn input_over_the_window_is_a_400_context_length_exceeded() {
    assert_rejected(enforce_context_length(&input(17), &single(Some(16)), MODEL));
}

#[tokio::test]
async fn rejection_body_names_the_limit_and_the_actual_count() {
    let response = enforce_context_length(&input(40), &single(Some(16)), MODEL)
        .expect_err("over-long input must be rejected");
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
    assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED);
    let message = json["error"]["message"].as_str().expect("message");
    assert!(message.contains("16 tokens"), "{message}");
    assert!(message.contains("40 input tokens"), "{message}");
    assert!(
        !message.contains("worker"),
        "the client mistake must not be blamed on a worker: {message}"
    );
}

/// No advertised window means no gateway opinion: the engine keeps
/// enforcing its own limit exactly as before this check existed.
#[test]
fn unknown_window_is_not_enforced() {
    assert!(enforce_context_length(&input(1_000_000), &single(None), MODEL).is_ok());
    assert!(enforce_context_length(&input(1_000_000), &pd(None, None), MODEL).is_ok());
}

/// A disaggregated prompt has to fit both engines; the tighter one wins.
#[test]
fn disaggregated_uses_the_tighter_side() {
    assert!(enforce_context_length(&input(10), &pd(Some(16), Some(16)), MODEL).is_ok());
    assert_rejected(enforce_context_length(
        &input(10),
        &pd(Some(16), Some(8)),
        MODEL,
    ));
    assert_rejected(enforce_context_length(
        &input(10),
        &pd(Some(8), Some(16)),
        MODEL,
    ));
}

#[test]
fn disaggregated_with_one_unknown_side_uses_the_known_one() {
    assert!(enforce_context_length(&input(10), &pd(None, Some(16)), MODEL).is_ok());
    assert_rejected(enforce_context_length(
        &input(10),
        &pd(Some(8), None),
        MODEL,
    ));
    assert_rejected(enforce_context_length(
        &input(10),
        &pd(None, Some(8)),
        MODEL,
    ));
}

/// Each batched prompt is its own engine request: a batch of short prompts
/// whose *sum* exceeds the window is fine, one long prompt anywhere in the
/// batch is not.
#[test]
fn batched_completion_is_bounded_per_item() {
    let batch = |lens: &[usize]| PreparationOutput::Completion {
        items: lens
            .iter()
            .map(|&len| CompletionItem {
                text: String::new(),
                token_ids: vec![7; len],
            })
            .collect(),
        joined_routing_text: None,
    };
    assert!(enforce_context_length(&batch(&[10, 10, 10]), &single(Some(16)), MODEL).is_ok());
    assert_rejected(enforce_context_length(
        &batch(&[4, 17]),
        &single(Some(16)),
        MODEL,
    ));
}

/// The window is looked up for the model actually requested, so a worker
/// serving several cards with different windows is bounded by the right one.
#[test]
fn window_is_resolved_for_the_requested_model_card() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:1/multi")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Sglang)
            .models(vec![
                ModelCard::new("wide").with_context_length(64),
                ModelCard::new("narrow").with_context_length(8),
            ])
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let selection = WorkerSelection::Single {
        worker: worker.into(),
    };
    assert!(enforce_context_length(&input(10), &selection, "wide").is_ok());
    assert_rejected(enforce_context_length(&input(10), &selection, "narrow"));
    // An id no card matches falls back to the primary (first) card.
    assert!(enforce_context_length(&input(10), &selection, "unlisted").is_ok());
}

/// A prefill worker is held to the input alone: its decode partner holds
/// the completion budget. The same budget on a regular or a decode worker
/// with the same window is refused, and the input still binds the
/// prefill side.
#[test]
fn a_prefill_worker_is_held_to_the_input_alone() {
    let requirements = ContextRequirements::new(&input(1024), &completion(8192));
    assert!(requirements.admits(worker(WorkerType::Prefill, Some(4096)).as_ref(), MODEL));
    assert!(!requirements.admits(worker(WorkerType::Decode, Some(4096)).as_ref(), MODEL));
    assert!(!requirements.admits(worker(WorkerType::Regular, Some(4096)).as_ref(), MODEL));
    assert!(requirements.admits(worker(WorkerType::Decode, Some(32768)).as_ref(), MODEL));
    let over_long = ContextRequirements::new(&input(5000), &completion(1));
    assert!(
        !over_long.admits(worker(WorkerType::Prefill, Some(4096)).as_ref(), MODEL),
        "the input still binds the prefill side"
    );
}

/// On a disaggregated pair the budget is bound by the decode side's
/// window and the input by the tighter of the two.
#[test]
fn the_pair_budget_is_bound_by_the_decode_window() {
    let pair = pd(Some(4096), Some(32768));
    assert!(enforce_output_budget(&completion(8192), &pair, MODEL).is_ok());
    assert_rejected(enforce_output_budget(&completion(40000), &pair, MODEL));
    assert!(enforce_context_length(&input(1024), &pair, MODEL).is_ok());
    assert_rejected(enforce_context_length(&input(5000), &pair, MODEL));
    // The selection-time refusal of a pair nothing fits names the same windows.
    assert_rejected(
        ContextRequirements::new(&input(1024), &completion(40000)).enforce_for_pair(
            Some(4096),
            Some(32768),
            MODEL,
        ),
    );
    assert!(ContextRequirements::new(&input(1024), &completion(8192))
        .enforce_for_pair(Some(4096), Some(32768), MODEL)
        .is_ok());
    assert_rejected(
        ContextRequirements::new(&input(5000), &completion(1)).enforce_for_pair(
            Some(4096),
            None,
            MODEL,
        ),
    );
}

#[test]
fn a_lists_floor_answers_as_its_workers_do() {
    // Every kind of worker the rule tells apart, alone, in pairs (a
    // minimum over a class is decided between two) and in triples of
    // the two classes, against every demand around the windows' edges:
    // each worker answers as the rule spelled out below, and the
    // floor's answer is the workers' own, both ways.
    //
    // The rule: only a gRPC worker with a known window can refuse. It
    // refuses an input that does not fit, and a budget above its
    // window unless it is a prefill worker.
    let refuses = |demand: &ContextRequirements, worker: &dyn Worker| {
        let Some(window) = worker.context_length(MODEL) else {
            return false;
        };
        if *worker.connection_mode() != ConnectionMode::Grpc {
            return false;
        }
        let input_over = if demand.scoring_position {
            demand.input_tokens >= window as usize
        } else {
            demand.input_tokens > window as usize
        };
        let budget_over = *worker.worker_type() != WorkerType::Prefill
            && demand.output_tokens.is_some_and(|tokens| tokens > window);
        input_over || budget_over
    };
    let windows = [None, Some(32), Some(64), Some(65), Some(256), Some(257)];
    let modes = [
        ConnectionMode::Grpc,
        ConnectionMode::Http,
        ConnectionMode::Zmq,
    ];
    let types = [
        WorkerType::Regular,
        WorkerType::Prefill,
        WorkerType::Decode,
        WorkerType::Encode,
    ];
    let mut kinds = Vec::new();
    for window in windows {
        for mode in modes {
            for worker_type in types {
                kinds.push(kind(worker_type, mode, window));
            }
        }
    }
    let mut pools: Vec<Vec<Arc<dyn Worker>>> = Vec::new();
    for (i, first) in kinds.iter().enumerate() {
        pools.push(vec![Arc::clone(first)]);
        for second in &kinds[i + 1..] {
            pools.push(vec![Arc::clone(first), Arc::clone(second)]);
        }
    }
    // Three workers: a floor that kept the last one alone would pass
    // some pairs. The gRPC kinds of both classes, in every order.
    let bound: Vec<Arc<dyn Worker>> = [None, Some(32), Some(64), Some(257)]
        .into_iter()
        .flat_map(|window| {
            [WorkerType::Regular, WorkerType::Prefill]
                .map(|worker_type| kind(worker_type, ConnectionMode::Grpc, window))
        })
        .collect();
    for first in &bound {
        for second in &bound {
            for third in &bound {
                pools.push(vec![
                    Arc::clone(first),
                    Arc::clone(second),
                    Arc::clone(third),
                ]);
            }
        }
    }
    let mut demands = Vec::new();
    for input_tokens in [0, 63, 64, 65, 256, 300] {
        for scoring_position in [false, true] {
            for output_tokens in [None, Some(1), Some(255), Some(256), Some(257)] {
                demands.push(ContextRequirements {
                    input_tokens,
                    scoring_position,
                    output_tokens,
                });
            }
        }
    }
    let (mut held, mut refused) = (0usize, 0usize);
    for pool in &pools {
        let floor = window_floor_of(pool, MODEL);
        for demand in &demands {
            for worker in pool {
                assert_eq!(
                    demand.admits(worker.as_ref(), MODEL),
                    !refuses(demand, worker.as_ref()),
                    "{demand:?} at {:?}",
                    (
                        worker.worker_type(),
                        worker.connection_mode(),
                        worker.context_length(MODEL)
                    )
                );
            }
            let every_worker_admits = pool
                .iter()
                .all(|worker| demand.admits(worker.as_ref(), MODEL));
            assert_eq!(
                demand.held_by_all(floor),
                every_worker_admits,
                "{demand:?} over {floor:?}: {:?}",
                pool.iter()
                    .map(|worker| (
                        worker.worker_type(),
                        worker.connection_mode(),
                        worker.context_length(MODEL)
                    ))
                    .collect::<Vec<_>>()
            );
            if every_worker_admits {
                held += 1;
            } else {
                refused += 1;
            }
        }
    }
    // The cross product reaches both answers many times over.
    assert!(
        held > 1_000 && refused > 1_000,
        "{held} held, {refused} refused"
    );
}
