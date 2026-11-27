use openai_protocol::{common::CachePartition, worker::WorkerStatus};
use smg_worker::labels::{CACHE_PARTITION_FIELDS_LABEL, SUPPORTS_VISION_LABEL};

use super::*;

fn grpc_worker(
    port: u16,
    role: WorkerType,
    runtime: RuntimeType,
    fields: Option<&str>,
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .model(ModelCard::new("test-model"))
        .worker_type(role)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(runtime)
        .health_config(no_health_check());
    if let Some(fields) = fields {
        builder = builder.label(CACHE_PARTITION_FIELDS_LABEL, fields);
    }
    Arc::new(builder.build())
}

fn salt() -> CachePartitionDemand {
    CachePartitionDemand::of(&CachePartition {
        cache_salt: Some("s"),
        ..Default::default()
    })
}

fn registry_of(workers: &[Arc<dyn Worker>]) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    for worker in workers {
        registry.register(Arc::clone(worker)).unwrap();
    }
    registry
}

/// A generate request with the given partition fields.
fn partitioned(registry: &Arc<WorkerRegistry>, fields: serde_json::Value) -> RequestContext {
    let mut body = serde_json::json!({"model": "test-model", "text": "hi"});
    for (key, value) in fields.as_object().unwrap() {
        body[key] = value.clone();
    }
    prepared_ctx_of(Arc::clone(registry), None, body)
}

fn stage(registry: &Arc<WorkerRegistry>, mode: WorkerSelectionMode) -> WorkerSelectionStage {
    WorkerSelectionStage::new(
        Arc::clone(registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        mode,
        None,
    )
}

/// The URLs of the workers that build or run the selected request.
fn selected(workers: &WorkerSelection) -> Vec<String> {
    match workers {
        WorkerSelection::Single { worker } => vec![worker.url().to_string()],
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => vec![prefill.url().to_string(), decode.url().to_string()],
    }
}

async fn refusal_code(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    (status, body["error"]["code"].as_str().unwrap().to_string())
}

#[tokio::test]
async fn a_salted_request_goes_only_to_workers_that_apply_its_fields() {
    let registry = registry_of(&[
        grpc_worker(9601, WorkerType::Regular, RuntimeType::Vllm, None),
        grpc_worker(
            9602,
            WorkerType::Regular,
            RuntimeType::Vllm,
            Some("cache_salt"),
        ),
        grpc_worker(
            9603,
            WorkerType::Regular,
            RuntimeType::Sglang,
            Some("cache_salt,extra_key"),
        ),
        grpc_worker(9604, WorkerType::Regular, RuntimeType::TokenSpeed, None),
    ]);
    let stage = stage(&registry, WorkerSelectionMode::Regular);
    for (fields, takers) in [
        (
            serde_json::json!({"cache_salt": "tenant-a"}),
            &["grpc://127.0.0.1:9602", "grpc://127.0.0.1:9603"][..],
        ),
        (
            serde_json::json!({"extra_key": "k"}),
            &["grpc://127.0.0.1:9603"][..],
        ),
    ] {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..8 {
            let mut ctx = partitioned(&registry, fields.clone());
            stage.execute(&mut ctx).await.unwrap();
            let url = selected(ctx.state.workers.as_ref().unwrap()).remove(0);
            assert!(takers.contains(&url.as_str()), "{fields} went to {url}");
            seen.insert(url);

            // A retry keeps the rule: it never moves to a worker that drops a field.
            let mut dispatch = Box::new(ctx).into_dispatch().unwrap();
            stage
                .reselect(&mut dispatch, &retained_plan(&[1, 2, 3]))
                .await
                .unwrap();
            let url = selected(dispatch.workers.as_ref().unwrap()).remove(0);
            assert!(takers.contains(&url.as_str()), "{fields} retried on {url}");
        }
        assert_eq!(seen.len(), takers.len(), "round robin reaches each taker");
    }
}

#[tokio::test]
async fn two_salts_key_two_namespaces() {
    let registry = registry_of(&[grpc_worker(
        9611,
        WorkerType::Regular,
        RuntimeType::Vllm,
        Some("cache_salt"),
    )]);
    let stage = stage(&registry, WorkerSelectionMode::Regular);
    let mut namespaces = Vec::new();
    for salt in ["tenant-a", "tenant-b"] {
        let mut ctx = partitioned(&registry, serde_json::json!({"cache_salt": salt}));
        stage.execute(&mut ctx).await.unwrap();
        let namespace = ctx.state.routing_snapshot.as_ref().unwrap().cache_namespace;
        assert_eq!(
            namespace,
            CacheNamespace::derive(&CachePartition {
                cache_salt: Some(salt),
                extra_key: None,
                lora_path: None,
            })
        );
        namespaces.push(namespace);
    }
    assert!(namespaces[0].is_some());
    assert_ne!(namespaces[0], namespaces[1]);
}

#[tokio::test]
async fn a_regular_pool_that_drops_a_field_refuses_the_request() {
    let registry = registry_of(&[
        grpc_worker(9621, WorkerType::Regular, RuntimeType::Vllm, None),
        grpc_worker(
            9622,
            WorkerType::Regular,
            RuntimeType::Vllm,
            Some("cache_salt"),
        ),
    ]);
    let stage = stage(&registry, WorkerSelectionMode::Regular);
    // vLLM has no extra key: no worker of the pool applies it.
    let mut ctx = partitioned(&registry, serde_json::json!({"extra_key": "k"}));
    let (status, code) = refusal_code(stage.execute(&mut ctx).await.unwrap_err()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "cache_partition_not_supported");

    // A request without a field still goes to any worker.
    let mut ctx = partitioned(&registry, serde_json::json!({"cache_salt": ""}));
    stage.execute(&mut ctx).await.unwrap();
}

#[tokio::test]
async fn a_pool_whose_applying_worker_is_down_answers_503_not_400() {
    let down = grpc_worker(
        9631,
        WorkerType::Regular,
        RuntimeType::Vllm,
        Some("cache_salt"),
    );
    let registry = registry_of(&[
        Arc::clone(&down),
        grpc_worker(9632, WorkerType::Regular, RuntimeType::Vllm, None),
    ]);
    down.set_status(WorkerStatus::NotReady);
    let stage = stage(&registry, WorkerSelectionMode::Regular);
    let mut ctx = partitioned(&registry, serde_json::json!({"cache_salt": "tenant-a"}));
    let refusal = stage.execute(&mut ctx).await.unwrap_err();
    assert_eq!(refusal.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn a_disaggregated_request_needs_a_pair_whose_sides_both_apply_its_fields() {
    for mode in [
        WorkerSelectionMode::PrefillDecode,
        WorkerSelectionMode::EncodePrefillDecode,
    ] {
        let prefill = grpc_worker(
            9641,
            WorkerType::Prefill,
            RuntimeType::Vllm,
            Some("cache_salt"),
        );
        let silent_decode = grpc_worker(9642, WorkerType::Decode, RuntimeType::Vllm, None);
        let registry = registry_of(&[Arc::clone(&prefill), Arc::clone(&silent_decode)]);
        let salted = serde_json::json!({"cache_salt": "tenant-a"});

        let mut ctx = partitioned(&registry, salted.clone());
        let (status, code) =
            refusal_code(stage(&registry, mode).execute(&mut ctx).await.unwrap_err()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mode:?}");
        assert_eq!(code, "cache_partition_not_supported", "{mode:?}");

        let decode = grpc_worker(
            9643,
            WorkerType::Decode,
            RuntimeType::Vllm,
            Some("cache_salt"),
        );
        registry.register(Arc::clone(&decode)).unwrap();
        let stage = stage(&registry, mode);
        for _ in 0..4 {
            let mut ctx = partitioned(&registry, salted.clone());
            stage.execute(&mut ctx).await.unwrap();
            assert_eq!(
                selected(ctx.state.workers.as_ref().unwrap()),
                vec![prefill.url().to_string(), decode.url().to_string()],
                "{mode:?}"
            );
            ctx.state.pd_prefill_guard = None;
        }
    }
}

/// Under Grand Central the engine places the request, and the same rule
/// holds: it places it only on workers that list its fields.
#[tokio::test]
async fn under_grand_central_a_salted_request_goes_only_to_workers_that_apply_it() {
    use crate::{
        config::types::RoutingKeyOverrideConfig,
        routers::common::placement::tests::{hearing_engine_on, hearing_pairs_on},
    };

    // The engine runs under a document: one profile for a single worker,
    // the stock pair presets for a pair.
    let engine_stage = |registry: &Arc<WorkerRegistry>, mode| {
        let policies = PolicyRegistry::for_engine(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig::default(),
        );
        if mode == WorkerSelectionMode::Regular {
            hearing_engine_on(&policies);
        } else {
            hearing_pairs_on(&policies, false);
        }
        WorkerSelectionStage::new(Arc::clone(registry), Arc::new(policies), mode, None)
    };
    let salted = serde_json::json!({"cache_salt": "tenant-a"});

    let applying = grpc_worker(
        9652,
        WorkerType::Regular,
        RuntimeType::Vllm,
        Some("cache_salt"),
    );
    let registry = registry_of(&[
        grpc_worker(9651, WorkerType::Regular, RuntimeType::Vllm, None),
        Arc::clone(&applying),
    ]);
    let stage = engine_stage(&registry, WorkerSelectionMode::Regular);
    for _ in 0..6 {
        let mut ctx = partitioned(&registry, salted.clone());
        stage.execute(&mut ctx).await.unwrap();
        assert_eq!(
            selected(ctx.state.workers.as_ref().unwrap()),
            vec![applying.url().to_string()]
        );
    }

    let prefill = grpc_worker(
        9653,
        WorkerType::Prefill,
        RuntimeType::Vllm,
        Some("cache_salt"),
    );
    let decode = grpc_worker(
        9655,
        WorkerType::Decode,
        RuntimeType::Vllm,
        Some("cache_salt"),
    );
    let registry = registry_of(&[
        Arc::clone(&prefill),
        grpc_worker(9654, WorkerType::Decode, RuntimeType::Vllm, None),
        Arc::clone(&decode),
    ]);
    let stage = engine_stage(&registry, WorkerSelectionMode::PrefillDecode);
    for _ in 0..6 {
        let mut ctx = partitioned(&registry, salted.clone());
        stage.execute(&mut ctx).await.unwrap();
        assert_eq!(
            selected(ctx.state.workers.as_ref().unwrap()),
            vec![prefill.url().to_string(), decode.url().to_string()]
        );
        ctx.state.pd_prefill_guard = None;
    }
}

/// A PD pair must meet the vision demand and the fields together. One
/// prefill worker has the encoder. The other prefill worker lists the salt.
/// No pair takes a salted image request, so the answer is 400, not 503.
#[tokio::test]
async fn a_pd_pair_needs_a_prefill_with_the_encoder_and_the_fields() {
    let model_id = "test-model";
    let registry = registry_of(&[
        labeled_worker(
            "grpc://127.0.0.1:9661",
            model_id,
            WorkerType::Prefill,
            &[(SUPPORTS_VISION_LABEL, "true")],
        ),
        labeled_worker(
            "grpc://127.0.0.1:9662",
            model_id,
            WorkerType::Prefill,
            &[
                (SUPPORTS_VISION_LABEL, "false"),
                (CACHE_PARTITION_FIELDS_LABEL, "cache_salt"),
            ],
        ),
        labeled_worker(
            "grpc://127.0.0.1:9663",
            model_id,
            WorkerType::Decode,
            &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt")],
        ),
    ]);
    let stage = round_robin_pd_stage(Arc::clone(&registry));
    let pick =
        |demand| stage.select_pd_pair(model_id, PlacementInputs::default(), None, None, demand);
    // Each demand alone finds a pair.
    assert!(pick(PairDemands::new(true, false)).is_ok());
    assert!(pick(PairDemands::new(false, false).with_partition(salt())).is_ok());

    let failure = pick(PairDemands::new(true, false).with_partition(salt()))
        .expect_err("no pair meets both demands");
    let media = MediaDemands {
        media_refs: false,
        vision: true,
        pipeline: PipelinePin::Any,
    };
    let (status, code) = refusal_code(stage.pd_selection_failure(model_id, failure, media)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "cache_partition_not_supported");
}

/// The same rule holds for an EPD pair on its decode side. One decode
/// worker has vision. The other decode worker lists the salt.
#[test]
fn an_epd_pair_needs_a_decode_with_vision_and_the_fields() {
    let model_id = "test-model";
    let applying_decode = labeled_worker(
        "grpc://127.0.0.1:9673",
        model_id,
        WorkerType::Decode,
        &[
            (SUPPORTS_VISION_LABEL, "false"),
            (CACHE_PARTITION_FIELDS_LABEL, "cache_salt"),
        ],
    );
    let registry = registry_of(&[
        labeled_worker(
            "grpc://127.0.0.1:9671",
            model_id,
            WorkerType::Prefill,
            &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt")],
        ),
        labeled_worker(
            "grpc://127.0.0.1:9672",
            model_id,
            WorkerType::Decode,
            &[(SUPPORTS_VISION_LABEL, "true")],
        ),
        Arc::clone(&applying_decode),
    ]);
    let stage = stage(&registry, WorkerSelectionMode::EncodePrefillDecode);
    let pick = |encode, decode_vision| {
        stage.select_encode_prefill_decode_workers(
            model_id,
            PlacementInputs::default(),
            &[],
            EpdDemands {
                wire: None,
                encode,
                decode_vision,
                image_fanout: false,
                partition: salt(),
            },
            None,
        )
    };
    // The decode side needs vision: no pair has it with the salt.
    assert!(matches!(
        pick(MediaDemands::NONE, true),
        Err(EpdSelectionFailure::CachePartitionUnsupported(_))
    ));
    // Media references only prefer vision: the pair that lists the salt
    // takes the request.
    let refs = MediaDemands {
        media_refs: true,
        vision: true,
        pipeline: PipelinePin::Any,
    };
    let Ok((_, _, decode, _)) = pick(refs, false) else {
        panic!("the pair that lists the salt takes the request");
    };
    assert_eq!(decode.url(), applying_decode.url());
}

/// The decode asked again behind its prefill, under Grand Central. The
/// first decode lists the salt and goes down. The other decode is ready and
/// holds a client. It does not list the salt. Gives the URL of the decode
/// that the request moved to, if it moved.
#[expect(
    clippy::disallowed_methods,
    reason = "test listener task; it ends with the test's runtime"
)]
async fn decode_asked_again(partition: CachePartitionDemand) -> Option<String> {
    use crate::routers::{common::placement::tests::hearing_pairs_on, grpc::client::GrpcClient};

    let model_id = "test-model";
    // A loopback listener stands in for the second decode's engine: the
    // rule needs only a client for it.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
    let client = GrpcClient::connect(&format!("grpc://127.0.0.1:{port}"), "vllm")
        .await
        .expect("a client");
    let worker = |url: &str, role: WorkerType, lists: bool, client: Option<BackendClient>| {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(role)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("NixlConnector")
            .health_config(no_health_check());
        if lists {
            builder = builder.label(CACHE_PARTITION_FIELDS_LABEL, "cache_salt");
        }
        if let Some(client) = client {
            builder = builder.engine_connection(client);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        worker
    };
    let first = worker("grpc://127.0.0.1:9681", WorkerType::Decode, true, None);
    let registry = registry_of(&[
        worker("grpc://127.0.0.1:9680", WorkerType::Prefill, true, None),
        Arc::clone(&first),
        worker(
            &format!("grpc://127.0.0.1:{port}"),
            WorkerType::Decode,
            false,
            Some(BackendClient::Grpc(client)),
        ),
    ]);
    let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
    hearing_pairs_on(&policies, true);
    let stage = WorkerSelectionStage::new(
        Arc::clone(&registry),
        Arc::new(policies),
        WorkerSelectionMode::PrefillDecode,
        None,
    );
    let (prefill, decode, _) = stage
        .placed_pd_pair(
            model_id,
            PlacementInputs::default(),
            None,
            None,
            PairDemands::new(false, false).with_partition(salt()),
            placement::Stated::nothing,
        )
        .expect("a pair");
    assert_eq!(decode.url(), first.url());
    drop(prefill.guard(None));
    let ahead = decode.ahead(Some("key"));
    first.set_status(WorkerStatus::NotReady);

    let routing = RoutingSnapshot {
        context_requirements: ContextRequirements::default(),
        routing_text: None,
        rid: None,
        cache_namespace: None,
        needs_vision: false,
        decode_needs_vision: false,
        language_model_reads_grids: true,
        cache_partition: partition,
    };
    let ledger = AttemptLedger::default();
    let request = RequestAgain {
        model_id,
        routing: &routing,
        headers: None,
        tokens: &[1, 2, 3],
        wire: grpc_wire(RuntimeType::Vllm),
        tried: &[],
        ledger: &ledger,
        claimed: false,
    };
    let (moved, guard) = stage.decode_again(&request, &decode, ahead);
    drop(guard);
    moved.map(|moved| moved.decode.url().to_string())
}

#[tokio::test]
async fn a_decode_asked_again_moves_only_to_a_decode_that_lists_the_fields() {
    assert!(
        decode_asked_again(CachePartitionDemand::default())
            .await
            .is_some(),
        "a request without a field moves to the other decode"
    );
    assert_eq!(
        decode_asked_again(salt()).await,
        None,
        "a salted request does not move to a decode that drops the salt"
    );
}
