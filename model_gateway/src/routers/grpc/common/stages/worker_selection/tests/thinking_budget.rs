use super::{RuntimeType::Vllm, WorkerType::Regular, *};

const MODEL: &str = "test-model";
const CAPABLE: &[(&str, &str)] = &[(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "true")];
const DISABLED: &[(&str, &str)] = &[(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL, "false")];
const MODES: &[(WorkerSelectionMode, &[WorkerType])] =
    &[(WorkerSelectionMode::Regular, &[Regular])];

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
