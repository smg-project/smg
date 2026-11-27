use super::{
    RuntimeType::{Sglang, Unspecified, Vllm},
    WorkerType::{Decode, Encode, Prefill, Regular},
    *,
};

const MODEL: &str = "test-model";
const CAPABLE: &[(&str, &str)] = &[(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "true")];
const DISABLED: &[(&str, &str)] = &[(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "false")];
const MODES: &[(WorkerSelectionMode, &[WorkerType])] = &[
    (WorkerSelectionMode::Regular, &[Regular]),
    (WorkerSelectionMode::PrefillDecode, &[Prefill, Decode]),
    (WorkerSelectionMode::EncodePrefillDecode, &[Prefill, Decode]),
];

fn register(
    registry: &WorkerRegistry,
    port: usize,
    role: WorkerType,
    runtime: RuntimeType,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    let url = format!("grpc://127.0.0.1:{port}");
    let worker = discovered_grpc_worker(&url, MODEL, role, runtime, labels);
    registry.register(Arc::clone(&worker)).unwrap();
    worker
}

fn stage_for(registry: &Arc<WorkerRegistry>, mode: WorkerSelectionMode) -> WorkerSelectionStage {
    WorkerSelectionStage::new(
        Arc::clone(registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        mode,
        None,
    )
}

fn chat_context(registry: &Arc<WorkerRegistry>, budget: i32) -> RequestContext {
    let mut ctx = prepared_ctx(Arc::clone(registry), None);
    ctx.input.request_type = RequestType::Chat(Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": budget,
        }))
        .unwrap(),
    ));
    ctx
}

#[test]
fn thinking_token_budget_requires_a_worker_declaration_and_an_enabled_label() {
    for (discovered, effective) in [(UNLABELED, "true"), (DISABLED, "true"), (CAPABLE, "false")] {
        let worker = BasicWorkerBuilder::new("grpc://127.0.0.1:9100")
            .runtime_type(Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .discovered_labels(smg_worker::test_support::labels_of(discovered))
            .label(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, effective)
            .build();
        assert!(!supports_thinking_budget(&worker));
    }
}

#[tokio::test]
async fn thinking_token_budget_selects_capable_workers_on_first_and_retry_attempts() {
    for &(mode, roles) in MODES {
        let registry = Arc::new(WorkerRegistry::new());
        let stage = stage_for(&registry, mode);
        for (i, &role) in roles.iter().enumerate() {
            for (j, labels) in [UNLABELED, CAPABLE].into_iter().enumerate() {
                register(&registry, 9100 + 2 * i + j, role, Vllm, labels);
            }
        }
        let assert_capable = |selection: &WorkerSelection| {
            assert!(selection
                .workers()
                .all(|w| supports_thinking_budget(w.as_ref())));
        };
        for _ in 0..4 {
            let mut ctx = chat_context(&registry, 0);
            stage.execute(&mut ctx).await.unwrap();
            assert_capable(ctx.state.workers.as_ref().unwrap());
            let mut dispatch = Box::new(ctx).into_dispatch().unwrap();
            assert!(dispatch.routing.needs_thinking_budget);
            stage
                .reselect(&mut dispatch, &retained_plan(&[1, 2, 3]))
                .await
                .unwrap();
            assert_capable(dispatch.workers.as_ref().unwrap());
        }
    }
}

#[tokio::test]
async fn thinking_token_budget_distinguishes_unsupported_and_unavailable_workers() {
    use openai_protocol::worker::WorkerStatus;

    for &(mode, roles) in MODES {
        for &missing in roles {
            let registry = Arc::new(WorkerRegistry::new());
            let stage = stage_for(&registry, mode);
            let mut budget = chat_context(&registry, 8);
            assert_eq!(
                stage.execute(&mut budget).await.unwrap_err().status(),
                StatusCode::NOT_FOUND
            );

            for (index, &role) in roles.iter().enumerate() {
                let labels = if role == missing { UNLABELED } else { CAPABLE };
                register(&registry, 9200 + index, role, Vllm, labels);
            }

            let refusal = stage.execute(&mut budget).await.unwrap_err();
            assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                error::extract_error_code_from_response(&refusal),
                "unsupported_thinking_token_budget"
            );
            let mut default_budget = chat_context(&registry, -1);
            stage.execute(&mut default_budget).await.unwrap();

            let capable = register(&registry, 9210, missing, Vllm, CAPABLE);
            capable.set_status(WorkerStatus::NotReady);
            let refusal = stage.execute(&mut budget).await.unwrap_err();
            assert_eq!(refusal.status(), StatusCode::SERVICE_UNAVAILABLE);
        }
    }
}

#[test]
fn thinking_token_budget_on_epd_only_requires_the_generation_workers_to_advertise_it() {
    let registry = ec_registry(
        MODEL,
        &[
            (9300, Encode, "ECCPUConnector"),
            (9301, Prefill, "ECCPUConnector"),
            (9302, Decode, "ECCPUConnector"),
        ],
    );
    let consumer = &[
        (SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "true"),
        ("supports_ec_transfer", "true"),
        ("ec_role", "ec_consumer"),
        ("ec_connector", "ECCPUConnector"),
        ("kv_connector", "NixlConnector"),
    ];
    register(&registry, 9303, Prefill, Vllm, consumer);
    register(&registry, 9304, Decode, Vllm, consumer);
    let stage = stage_for(&registry, WorkerSelectionMode::EncodePrefillDecode);
    let (encoders, prefill, decode, _) = stage
        .select_encode_prefill_decode_workers(
            MODEL,
            PlacementInputs::default(),
            &[vec![1]],
            EpdDemands {
                needs_thinking_budget: true,
                wire: None,
                encode: MediaDemands::NONE,
                decode_vision: false,
                image_fanout: false,
            },
            None,
        )
        .unwrap_or_else(|_| panic!("a compatible E/P/D set exists"));
    assert_eq!(encoders.len(), 1);
    assert_eq!(encoders[0].worker.url(), "grpc://127.0.0.1:9300");
    assert_eq!(prefill.url(), "grpc://127.0.0.1:9303");
    assert_eq!(decode.url(), "grpc://127.0.0.1:9304");
}

#[tokio::test]
async fn thinking_token_budget_keeps_prefill_admission_on_the_capable_pool() {
    let registry = Arc::new(WorkerRegistry::new());
    let prefill = register(&registry, 9400, Prefill, Vllm, CAPABLE);
    let legacy = register(&registry, 9401, Prefill, Vllm, UNLABELED);
    register(&registry, 9402, Decode, Vllm, CAPABLE);
    let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
    let occupied = occupy(&admission, &prefill).await;
    let stage = pd_stage(
        Arc::clone(&registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        admission,
    );
    let mut ctx = chat_context(&registry, 8);
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(refusal.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        error::extract_error_code_from_response(&refusal),
        PD_PREFILL_QUEUE_FULL
    );
    assert_eq!(legacy.load(), 0);
    drop(occupied);
    stage.execute(&mut ctx).await.unwrap();
    drop(ctx);
    assert_eq!(prefill.load(), 0);
}

#[tokio::test]
async fn thinking_token_budget_requires_both_capable_workers_to_form_a_pair() {
    let registry = Arc::new(WorkerRegistry::new());
    let stage = stage_for(&registry, WorkerSelectionMode::PrefillDecode);
    register(&registry, 9501, Prefill, Sglang, CAPABLE);
    register(&registry, 9502, Decode, Vllm, CAPABLE);
    let mut ctx = chat_context(&registry, 32);
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(
        error::extract_error_code_from_response(&refusal),
        "no_compatible_pd_pair"
    );

    register(&registry, 9503, Decode, Unspecified, DISABLED);
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(
        error::extract_error_code_from_response(&refusal),
        "unsupported_thinking_token_budget"
    );
    register(&registry, 9504, Prefill, Vllm, CAPABLE);
    stage.execute(&mut ctx).await.unwrap();
}
