use super::*;

const MODEL: &str = "test-model";
const CAPABLE: &[(&str, &str)] = &[(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "true")];
const MODES: [WorkerSelectionMode; 3] = [
    WorkerSelectionMode::Regular,
    WorkerSelectionMode::PrefillDecode,
    WorkerSelectionMode::EncodePrefillDecode,
];

fn generation_roles(mode: WorkerSelectionMode) -> &'static [WorkerType] {
    match mode {
        WorkerSelectionMode::Regular => &[WorkerType::Regular],
        _ => &[WorkerType::Prefill, WorkerType::Decode],
    }
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

#[tokio::test]
async fn thinking_token_budget_selects_capable_workers_on_first_and_retry_attempts() {
    for mode in MODES {
        let registry = Arc::new(WorkerRegistry::new());
        let stage = stage_for(&registry, mode);
        for (i, &role) in generation_roles(mode).iter().enumerate() {
            for (j, labels) in [UNLABELED, CAPABLE].into_iter().enumerate() {
                let url = format!("grpc://127.0.0.1:{}", 9100 + 2 * i + j);
                registry
                    .register(labeled_worker(&url, MODEL, role, labels))
                    .unwrap();
            }
        }
        for _ in 0..4 {
            let mut ctx = chat_context(&registry, 0);
            stage.execute(&mut ctx).await.unwrap();
            assert!(ctx
                .state
                .workers
                .as_ref()
                .unwrap()
                .workers()
                .all(|w| supports_thinking_budget(w.as_ref())));
            let mut dispatch = Box::new(ctx).into_dispatch().unwrap();
            assert!(dispatch.routing.needs_thinking_budget);
            stage
                .reselect(&mut dispatch, &retained_plan(&[1, 2, 3]))
                .await
                .unwrap();
            assert!(dispatch
                .workers
                .as_ref()
                .unwrap()
                .workers()
                .all(|w| supports_thinking_budget(w.as_ref())));
        }
    }
}

#[tokio::test]
async fn thinking_token_budget_distinguishes_unsupported_and_unavailable_workers() {
    use openai_protocol::worker::WorkerStatus;

    for mode in MODES {
        for &missing in generation_roles(mode) {
            let registry = Arc::new(WorkerRegistry::new());
            let stage = stage_for(&registry, mode);
            let mut budget = chat_context(&registry, 8);
            assert_eq!(
                stage.execute(&mut budget).await.unwrap_err().status(),
                StatusCode::NOT_FOUND
            );

            for (index, &role) in generation_roles(mode).iter().enumerate() {
                let url = format!("grpc://127.0.0.1:{}", 9200 + index);
                let labels = if role == missing { UNLABELED } else { CAPABLE };
                registry
                    .register(labeled_worker(&url, MODEL, role, labels))
                    .unwrap();
            }

            let refusal = stage.execute(&mut budget).await.unwrap_err();
            assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                error::extract_error_code_from_response(&refusal),
                "unsupported_thinking_token_budget"
            );
            let mut default_budget = chat_context(&registry, -1);
            stage.execute(&mut default_budget).await.unwrap();

            let capable = labeled_worker("grpc://127.0.0.1:9210", MODEL, missing, CAPABLE);
            registry.register(Arc::clone(&capable)).unwrap();
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
            (9300, WorkerType::Encode, "ECCPUConnector"),
            (9301, WorkerType::Prefill, "ECCPUConnector"),
            (9302, WorkerType::Decode, "ECCPUConnector"),
        ],
    );
    for (url, role) in [
        ("grpc://127.0.0.1:9303", WorkerType::Prefill),
        ("grpc://127.0.0.1:9304", WorkerType::Decode),
    ] {
        registry
            .register(labeled_worker(
                url,
                MODEL,
                role,
                &[
                    (SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "true"),
                    ("supports_ec_transfer", "true"),
                    ("ec_role", "ec_consumer"),
                    ("ec_connector", "ECCPUConnector"),
                    ("kv_connector", "NixlConnector"),
                ],
            ))
            .unwrap();
    }
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
    let prefill = labeled_worker("grpc://127.0.0.1:9400", MODEL, WorkerType::Prefill, CAPABLE);
    let legacy = labeled_worker(
        "grpc://127.0.0.1:9401",
        MODEL,
        WorkerType::Prefill,
        UNLABELED,
    );
    let decode = labeled_worker("grpc://127.0.0.1:9402", MODEL, WorkerType::Decode, CAPABLE);
    for worker in [&prefill, &legacy, &decode] {
        registry.register(Arc::clone(worker)).unwrap();
    }
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
    let register = |url, role, runtime, supports| {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(MODEL))
                    .worker_type(role)
                    .runtime_type(runtime)
                    .connection_mode(ConnectionMode::Grpc)
                    .health_config(no_health_check())
                    .label(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, supports)
                    .build(),
            ))
            .unwrap();
    };
    register(
        "grpc://127.0.0.1:9501",
        WorkerType::Prefill,
        RuntimeType::Sglang,
        "true",
    );
    register(
        "grpc://127.0.0.1:9502",
        WorkerType::Decode,
        RuntimeType::Vllm,
        "true",
    );
    let mut ctx = chat_context(&registry, 32);
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(
        error::extract_error_code_from_response(&refusal),
        "no_compatible_pd_pair"
    );

    register(
        "grpc://127.0.0.1:9503",
        WorkerType::Decode,
        RuntimeType::Unspecified,
        "false",
    );
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(
        error::extract_error_code_from_response(&refusal),
        "unsupported_thinking_token_budget"
    );
    register(
        "grpc://127.0.0.1:9504",
        WorkerType::Prefill,
        RuntimeType::Vllm,
        "true",
    );
    stage.execute(&mut ctx).await.unwrap();
}
