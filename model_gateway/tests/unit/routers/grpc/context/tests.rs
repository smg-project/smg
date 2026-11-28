use smg_grpc_client::VllmEngineClient;

use super::*;
use crate::routers::{
    common::prefill_admission::test_support::select_worker,
    grpc::{client::GrpcClient, test_engine::mock_engine},
};

/// Only multimodal vLLM EPD owns EC state absent from its retained plan.
/// The plan's kind, its media and the handoff kind each decide it
/// alone: a text-only plan encoded nothing, a rendezvous handoff keeps
/// its rooms across a re-selected pair, and no handoff has nothing to
/// replay.
#[test]
fn only_an_epd_plan_with_media_under_a_push_handoff_replays_a_replica_bound_handoff() {
    use smg_grpc_client::{tokenspeed_proto, vllm_proto};
    let push = Some(EncodeHandoff::Push);
    let media = ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
        media_refs: Some(vllm_proto::MediaRefs::default()),
        ..Default::default()
    }));
    for kind in [
        ExecutionPlanKind::Single,
        ExecutionPlanKind::PrefillDecode,
        ExecutionPlanKind::EncodePrefillDecode,
    ] {
        assert_eq!(
            ExecutionPlan::generate(kind, media.clone()).replays_replica_bound_handoff(push),
            kind == ExecutionPlanKind::EncodePrefillDecode
        );
    }
    assert!(ExecutionPlan::Batch {
        kind: ExecutionPlanKind::EncodePrefillDecode,
        shared_request_id: "batch".to_string(),
        requests: vec![media],
    }
    .replays_replica_bound_handoff(push));
    assert!(!ExecutionPlan::EncodePrefillDecode {
        request: ProtoGenerateRequest::Vllm(Box::default()),
    }
    .replays_replica_bound_handoff(push));
    let items = ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
        mm_inputs: Some(tokenspeed_proto::MultimodalInputs::default()),
        ..Default::default()
    }));
    for handoff in [Some(EncodeHandoff::Rendezvous), None] {
        assert!(!ExecutionPlan::EncodePrefillDecode {
            request: items.clone(),
        }
        .replays_replica_bound_handoff(handoff));
    }
}

/// The status of a failure a stage reports and the code of its JSON
/// error object.
async fn failure_shape(failure: axum::response::Response) -> (http::StatusCode, String) {
    let status = failure.status();
    let body = axum::body::to_bytes(failure.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
    let code = body["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("an error code: {body}"))
        .to_owned();
    (status, code)
}

/// The failure a read of a missing output reports (the outputs have no
/// `Debug` to `expect_err` against).
fn failure_of<T>(read: Result<T, axum::response::Response>) -> axum::response::Response {
    match read {
        Ok(_) => panic!("the output was there"),
        Err(failure) => failure,
    }
}

/// A disaggregated selection's wire is the prefill side's, which builds
/// the request: a direct-ZMQ prefill worker pins the plan to the ZMQ wire
/// of the pair's runtime whatever the decode side's transport, and a gRPC
/// prefill worker to gRPC.
#[test]
fn a_disaggregated_selections_wire_is_the_prefill_sides() {
    use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType};

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    let worker = |url: &str, mode: ConnectionMode, role: WorkerType| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(mode)
                .runtime_type(RuntimeType::TokenSpeed)
                .worker_type(role)
                .build(),
        )
    };
    let pair = |prefill: Arc<dyn Worker>, decode: Arc<dyn Worker>| WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode: decode.into(),
        runtime_type: RuntimeType::TokenSpeed,
    };

    let zmq_prefill = pair(
        worker(
            "ipc:///tmp/prefill.ipc",
            ConnectionMode::Zmq,
            WorkerType::Prefill,
        ),
        worker("grpc://decode:1", ConnectionMode::Grpc, WorkerType::Decode),
    );
    let wire = WireConstraint::of(&zmq_prefill, false, PipelinePin::Any);
    assert_eq!(wire.connection, ConnectionMode::Zmq);
    assert_eq!(wire.runtime, RuntimeType::TokenSpeed);
    assert!(!wire.requires_media_refs);

    let grpc_prefill = pair(
        worker(
            "grpc://prefill:1",
            ConnectionMode::Grpc,
            WorkerType::Prefill,
        ),
        worker(
            "ipc:///tmp/decode.ipc",
            ConnectionMode::Zmq,
            WorkerType::Decode,
        ),
    );
    let wire = WireConstraint::of(&grpc_prefill, true, PipelinePin::Any);
    assert_eq!(wire.connection, ConnectionMode::Grpc);
    assert!(wire.requires_media_refs);
}

#[tokio::test]
async fn a_missing_preparation_output_is_one_failure_whichever_stage_reads_it() {
    let mut slot: Option<PreparationOutput> = None;
    for failure in [
        failure_of(stage_output(slot.as_ref(), "a_stage")),
        failure_of(stage_output(slot.take(), "another_stage")),
    ] {
        assert_eq!(
            failure_shape(failure).await,
            (
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "preparation_not_completed".to_owned()
            )
        );
    }

    slot = Some(completion_prep(&["a prompt"], None));
    assert!(matches!(
        stage_output(slot.as_ref(), "a_stage"),
        Ok(PreparationOutput::Completion { .. })
    ));
    assert!(matches!(
        stage_output(slot.take(), "the_last_reader"),
        Ok(PreparationOutput::Completion { .. })
    ));
    assert!(slot.is_none(), "the taking reader consumes the output");
}

#[tokio::test]
async fn a_missing_worker_selection_is_one_failure_whichever_stage_reads_it() {
    use smg_worker::identity::basic::WorkerType;

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    let mut slot: Option<WorkerSelection> = None;
    let failure = failure_of(stage_output(slot.as_ref(), "a_stage"));
    assert_eq!(
        failure_shape(failure).await,
        (
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "worker_selection_not_completed".to_owned()
        )
    );

    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill-selected")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode-selected")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    slot = Some(pd_selection(&prefill, &decode));
    assert!(matches!(
        stage_output(slot.as_ref(), "a_stage"),
        Ok(WorkerSelection::Disaggregated { .. })
    ));
}

#[tokio::test]
async fn a_missing_client_selection_is_one_failure_whichever_stage_reads_it() {
    let mut slot: Option<ClientSelection> = None;
    for failure in [
        failure_of(stage_output(slot.as_ref(), "a_stage")),
        failure_of(stage_output(slot.as_mut(), "the_dispatch")),
    ] {
        assert_eq!(
            failure_shape(failure).await,
            (
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "client_acquisition_not_completed".to_owned()
            )
        );
    }
}

#[test]
fn the_dispatch_phase_outputs_keep_their_codes() {
    assert_eq!(
        <ExecutionResult as StageOutput>::NOT_COMPLETED.0,
        "no_execution_result"
    );
    assert_eq!(
        <DispatchMetadata as StageOutput>::NOT_COMPLETED.0,
        "dispatch_metadata_not_set"
    );
    assert_eq!(
        <Arc<dyn Tokenizer> as StageOutput>::NOT_COMPLETED.0,
        "tokenizer_not_cached"
    );
    assert_eq!(
        <StopSequenceDecoder as StageOutput>::NOT_COMPLETED.0,
        "stop_decoder_not_initialized"
    );
}

fn completion_prep(texts: &[&str], joined: Option<&str>) -> PreparationOutput {
    PreparationOutput::Completion {
        items: texts
            .iter()
            .enumerate()
            .map(|(i, text)| CompletionItem {
                text: (*text).to_string(),
                token_ids: vec![i as u32],
            })
            .collect(),
        joined_routing_text: joined.map(str::to_string),
    }
}

fn pd_selection(prefill: &Arc<dyn Worker>, decode: &Arc<dyn Worker>) -> WorkerSelection {
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: Arc::clone(prefill),
        decode: Arc::clone(decode).into(),
        runtime_type: RuntimeType::Sglang,
    }
}

#[test]
fn disaggregated_load_guards_hold_decode_only() {
    use smg_worker::identity::basic::WorkerType;

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill-load")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode-load")
            .worker_type(WorkerType::Decode)
            .build(),
    );

    let prefill_guard = PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), None),
    };
    assert_eq!(prefill.load(), 1);
    drop(prefill_guard);

    let guards = LoadGuards::new(&pd_selection(&prefill, &decode), None);
    assert_eq!(prefill.load(), 0);
    assert_eq!(decode.load(), 1);
    drop(guards);
    assert_eq!(decode.load(), 0);
}

#[tokio::test]
async fn batch_prompts_preserve_phase_load_semantics() {
    use smg_worker::identity::basic::WorkerType;

    use crate::routers::{
        common::prefill_admission::PrefillAdmission, grpc::backend_client::BasicWorkerBuilder,
    };

    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:30000")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:30000")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    let routing_key = "batch-key";

    let admission = PrefillAdmission::new(1, 0, std::time::Duration::from_secs(1));
    let admitted = match admission
        .admit(Some(routing_key), select_worker(&prefill))
        .await
    {
        Ok(admitted) => admitted,
        Err(_) => panic!("initial admission should succeed"),
    };
    let prefill_guard = PrefillLoadGuard::Admission {
        _reservation: Arc::new(admitted.reservation),
    };
    let mut prefill_children = vec![prefill_guard.replicate(), prefill_guard.replicate()];
    prefill_children.push(prefill_guard);
    let mut decode_children =
        match LoadGuards::scaled(&pd_selection(&prefill, &decode), Some(routing_key), 3) {
            LoadGuards::Batch { _guards: guards } => guards,
            _ => panic!("count > 1 should create batch guards"),
        };

    assert_eq!(prefill.load(), 1);
    assert_eq!(decode.load(), 3);
    assert_eq!(prefill.routing_key_load(), 1);
    assert_eq!(decode.routing_key_load(), 1);

    drop(prefill_children.pop());
    drop(decode_children.pop());
    assert_eq!(prefill.load(), 1);
    assert_eq!(decode.load(), 2);
    assert_eq!(prefill.routing_key_load(), 1);
    assert_eq!(decode.routing_key_load(), 1);

    drop(prefill_children);
    drop(decode_children);
    assert_eq!(prefill.load(), 0);
    assert_eq!(decode.load(), 0);
    assert_eq!(prefill.routing_key_load(), 0);
    assert_eq!(decode.routing_key_load(), 0);

    let prefill_guard = PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), Some(routing_key)),
    };
    let mut prefill_children = vec![prefill_guard.replicate(), prefill_guard.replicate()];
    prefill_children.push(prefill_guard);
    let mut decode_children =
        match LoadGuards::scaled(&pd_selection(&prefill, &decode), Some(routing_key), 3) {
            LoadGuards::Batch { _guards: guards } => guards,
            _ => panic!("count > 1 should create batch guards"),
        };

    assert_eq!(prefill.load(), 3);
    assert_eq!(decode.load(), 3);
    assert_eq!(prefill.routing_key_load(), 1);
    assert_eq!(decode.routing_key_load(), 1);

    drop(prefill_children.pop());
    drop(decode_children.pop());
    assert_eq!(prefill.load(), 2);
    assert_eq!(decode.load(), 2);
    assert_eq!(prefill.routing_key_load(), 1);
    assert_eq!(decode.routing_key_load(), 1);

    drop(prefill_children);
    drop(decode_children);
    assert_eq!(prefill.load(), 0);
    assert_eq!(decode.load(), 0);
    assert_eq!(prefill.routing_key_load(), 0);
    assert_eq!(decode.routing_key_load(), 0);
}

#[test]
fn completion_preparation_routes_by_first_item_or_joined_text() {
    let scalar = completion_prep(&["hello"], None);
    assert_eq!(scalar.routing_text(), Some("hello"));
    assert_eq!(scalar.token_ids(), &[0]);

    let batch = completion_prep(&["a", "b"], Some("a b"));
    assert_eq!(batch.routing_text(), Some("a b"));
    assert_eq!(batch.token_ids(), &[0]);
}

/// `token_ids()` is deliberately a single-item routing-affinity proxy
/// (see the test above), but a batched Completion's real input cost is
/// every prompt in the batch -- `total_input_token_count()` must sum
/// them all, not just echo the first item like `token_ids()` does.
#[test]
fn total_input_token_count_sums_every_batched_completion_item() {
    let batch = PreparationOutput::Completion {
        items: vec![
            CompletionItem {
                text: "short".to_string(),
                token_ids: vec![1],
            },
            CompletionItem {
                text: "much longer prompt".to_string(),
                token_ids: vec![2, 3, 4, 5, 6],
            },
        ],
        joined_routing_text: Some("short much longer prompt".to_string()),
    };

    // The routing proxy only ever sees the first item...
    assert_eq!(batch.token_ids().len(), 1);
    // ...but reservation accounting must see the whole batch's cost.
    assert_eq!(batch.total_input_token_count(), 6);

    let scalar = completion_prep(&["hello"], None);
    assert_eq!(scalar.total_input_token_count(), scalar.token_ids().len());
}

/// The context window bounds each engine request, and a batched
/// Completion dispatches one per prompt: the length check must see the
/// longest item, not the batch total (which would reject a batch of
/// short prompts) nor the first item (which would miss a long later one).
#[test]
fn max_input_token_count_is_the_longest_batched_completion_item() {
    let batch = PreparationOutput::Completion {
        items: vec![
            CompletionItem {
                text: "short".to_string(),
                token_ids: vec![1],
            },
            CompletionItem {
                text: "much longer prompt".to_string(),
                token_ids: vec![2, 3, 4, 5, 6],
            },
        ],
        joined_routing_text: Some("short much longer prompt".to_string()),
    };

    assert_eq!(batch.max_input_token_count(), 5);
    assert_ne!(batch.max_input_token_count(), batch.token_ids().len());
    assert_ne!(
        batch.max_input_token_count(),
        batch.total_input_token_count()
    );

    let scalar = completion_prep(&["hello"], None);
    assert_eq!(scalar.max_input_token_count(), scalar.token_ids().len());

    let empty = PreparationOutput::Completion {
        items: vec![],
        joined_routing_text: None,
    };
    assert_eq!(empty.max_input_token_count(), 0);
}

/// An embedding batch goes to the engine one input per call, so the
/// window applies to its longest input: neither the first alone (which
/// would let a long later input through) nor their sum.
#[test]
fn max_input_token_count_is_the_longest_embedding_input() {
    let batch = PreparationOutput::Embedding {
        inputs: vec![
            EmbeddingInput {
                text: Some("short".to_string()),
                token_ids: vec![1],
            },
            EmbeddingInput {
                text: None,
                token_ids: vec![2, 3, 4, 5, 6],
            },
        ],
    };

    assert_eq!(batch.max_input_token_count(), 5);
    assert_eq!(batch.token_ids().len(), 1);
    assert_eq!(batch.total_input_token_count(), 6);
}

#[test]
fn batch_execution_plan_reports_shared_id_and_kind_label() {
    let plan = ExecutionPlan::Batch {
        kind: ExecutionPlanKind::PrefillDecode,
        shared_request_id: "cmpl_shared".to_string(),
        requests: vec![],
    };
    assert_eq!(plan.request_id(), "cmpl_shared");
    assert_eq!(plan.request_type(), "generate");
    assert_eq!(plan.mode_label(), "prefill_decode");
}

/// The ids a retry routes by, per plan shape: what the first selection
/// used (a batch's first prompt), and none for a prompt that went as text.
#[test]
fn a_retry_routes_by_the_plans_prompt_ids_whatever_the_plans_shape() {
    use smg_grpc_client::vllm_proto as vllm;
    let tokenized = |ids: &[u32]| {
        ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "r".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: ids.to_vec(),
                },
            )),
            ..Default::default()
        }))
    };
    let as_text = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        request_id: "r".to_string(),
        input: Some(vllm::generate_request::Input::Text("as text".to_string())),
        ..Default::default()
    }));
    let embed = ProtoEmbedRequest::Vllm(Box::new(vllm::EmbedRequest {
        tokenized: Some(vllm::TokenizedInput {
            original_text: String::new(),
            input_ids: vec![4, 5],
        }),
        ..Default::default()
    }));
    let batch = |requests: Vec<ProtoGenerateRequest>| ExecutionPlan::Batch {
        kind: ExecutionPlanKind::Single,
        shared_request_id: "b".to_string(),
        requests,
    };
    let cases: [(&str, ExecutionPlan, &[u32]); 7] = [
        (
            "a single request",
            ExecutionPlan::generate(ExecutionPlanKind::Single, tokenized(&[1, 2, 3])),
            &[1, 2, 3],
        ),
        (
            "a single request that went as text",
            ExecutionPlan::generate(ExecutionPlanKind::Single, as_text),
            &[],
        ),
        (
            "prefill-decode",
            ExecutionPlan::generate(ExecutionPlanKind::PrefillDecode, tokenized(&[6])),
            &[6],
        ),
        (
            "encode-prefill-decode",
            ExecutionPlan::generate(ExecutionPlanKind::EncodePrefillDecode, tokenized(&[7])),
            &[7],
        ),
        ("an embed request", ExecutionPlan::embed(embed), &[4, 5]),
        (
            "a batch: its first prompt",
            batch(vec![tokenized(&[8, 9]), tokenized(&[10])]),
            &[8, 9],
        ),
        ("an empty batch", batch(Vec::new()), &[]),
    ];
    for (shape, plan, expected) in cases {
        assert_eq!(plan.routing_token_ids(), expected, "{shape}");
    }
}

#[test]
fn load_guards_report_completion_for_each_side_they_hold() {
    use smg_worker::identity::basic::{RequestCompletionSink, WorkerType};

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    #[derive(Debug, Default)]
    struct CompletionSpy(std::sync::Mutex<Vec<String>>);
    impl RequestCompletionSink for CompletionSpy {
        fn request_completed(&self, worker: &dyn Worker) {
            self.0.lock().unwrap().push(worker.url().to_string());
        }
    }

    let spy = Arc::new(CompletionSpy::default());
    let sink: Arc<dyn RequestCompletionSink> = spy.clone();
    let single: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://single")
            .worker_type(WorkerType::Regular)
            .build(),
    );
    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    for worker in [&single, &prefill, &decode] {
        worker.set_completion_sink(Some(Arc::clone(&sink)));
    }

    // The regular gRPC path: one guard, one completion when the stream ends.
    drop(LoadGuards::new(
        &WorkerSelection::Single {
            worker: Arc::clone(&single).into(),
        },
        None,
    ));
    assert_eq!(spy.0.lock().unwrap().as_slice(), ["grpc://single"]);

    // The disaggregated path: the decode side completes with the guards,
    // the prefill side with its own guard when the prefill phase ends.
    let guards = LoadGuards::new(&pd_selection(&prefill, &decode), None);
    let prefill_guard = PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), None),
    };
    drop(prefill_guard);
    assert_eq!(
        spy.0.lock().unwrap().last().map(String::as_str),
        Some("grpc://prefill")
    );
    drop(guards);
    assert_eq!(
        spy.0.lock().unwrap().last().map(String::as_str),
        Some("grpc://decode")
    );

    // A batched fan-out reports once per sub-request.
    let before = spy.0.lock().unwrap().len();
    drop(LoadGuards::scaled(
        &WorkerSelection::Single {
            worker: Arc::clone(&single).into(),
        },
        None,
        3,
    ));
    assert_eq!(spy.0.lock().unwrap().len(), before + 3);
}

/// The handle on a request's end is asked of its guard under whatever
/// rides with the guards: the dispatch's upstream span, an admission
/// claim. A request told through that handle hears its outcome.
#[test]
fn the_guards_hand_a_requests_end_under_what_rides_with_them() {
    use smg_worker::identity::basic::WorkerType;

    use crate::routers::{
        common::{
            pd_admission::{try_admit_decode, Tried},
            placement::tests::heard_request,
        },
        grpc::backend_client::BasicWorkerBuilder,
    };

    // A ride takes the guards and hands them back under itself.
    type Ride = Box<dyn FnOnce(LoadGuards) -> LoadGuards>;

    let window = [("max_running_requests".to_string(), "1".to_string())];
    let windowed: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://windowed")
            .worker_type(WorkerType::Decode)
            .labels(window.into_iter().collect())
            .build(),
    );
    let Tried::Claimed(claim) = try_admit_decode(&windowed, 1) else {
        panic!("a free room is claimed");
    };
    let rides: [(&str, Ride); 2] = [
        (
            "the upstream span",
            Box::new(|guards| LoadGuards::traced(tracing::Span::none(), guards)),
        ),
        (
            "an admission claim",
            Box::new(move |guards| LoadGuards::admitted(Some(claim), guards)),
        ),
    ];
    for (ride, under) in rides {
        let (heard, guard, end) = heard_request();
        drop(end);
        let guards = under(LoadGuards::Single { _guard: guard });
        assert!(
            matches!(
                guards,
                LoadGuards::Traced { .. } | LoadGuards::Admitted { .. }
            ),
            "{ride}"
        );
        guards.end().report(200, || true);
        drop(guards);
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            ["issued", "dispatched Single", "completed Ok 200"],
            "{ride}"
        );
    }
}

/// The client-shape mismatch answers an executor takes from the
/// selection: one status and code per shape, whichever executor asks;
/// a selection of the asked shape hands its clients out.
#[tokio::test]
async fn a_selection_of_the_other_shape_answers_the_mismatch() {
    let (url, server) = mock_engine("test-model").await;
    let client = VllmEngineClient::connect(&url)
        .await
        .expect("connect mock engine");
    let client = BackendClient::Grpc(GrpcClient::Vllm(client));
    let mut single = ClientSelection::Single {
        client: client.clone(),
    };
    assert!(single.single_or_mismatch("an_executor").is_ok());
    assert_eq!(
        failure_shape(failure_of(single.disaggregated_or_mismatch("an_executor"))).await,
        (
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "expected_disaggregated_clients_got_single".to_owned()
        )
    );
    let mut disaggregated = ClientSelection::Disaggregated {
        prefill: client.clone(),
        decode: client,
    };
    assert!(disaggregated
        .disaggregated_or_mismatch("an_executor")
        .is_ok());
    assert_eq!(
        failure_shape(failure_of(disaggregated.single_or_mismatch("an_executor"))).await,
        (
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "expected_single_client_got_disaggregated".to_owned()
        )
    );
    server.abort();
}

/// The Harmony iteration result rides behind a pointer, and the dispatch
/// context, boxed once per request, fits the allocator's 2,560-byte class
/// on 64-bit targets: inline, the result's 208 bytes had taken the box
/// from 2,520 to 2,584 bytes, one class up, when the usage object it and
/// the final response carry grew by 32 bytes each. A field that grows
/// the context past the class needs its own read.
#[test]
fn the_iteration_result_rides_behind_a_pointer_and_the_dispatch_context_fits_its_class() {
    let state = ResponseState::default();
    assert_eq!(
        size_of_val(&state.responses_iteration_result),
        size_of::<usize>()
    );
    assert!(
        size_of::<DispatchContext>() <= 2560,
        "the dispatch context is {} bytes",
        size_of::<DispatchContext>()
    );
}
