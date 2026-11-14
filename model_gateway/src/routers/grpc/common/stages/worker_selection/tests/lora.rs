use super::*;

fn disaggregated_registry(sglang: bool) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    let mut port = 9520;
    for runtime in [RuntimeType::Vllm, RuntimeType::Sglang] {
        if runtime == RuntimeType::Sglang && !sglang {
            continue;
        }
        for role in [WorkerType::Encode, WorkerType::Prefill, WorkerType::Decode] {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                        .model(ModelCard::new("test-model"))
                        .worker_type(role)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(runtime)
                        .health_config(no_health_check())
                        .label(
                            "supports_ec_transfer",
                            (runtime == RuntimeType::Vllm).to_string(),
                        )
                        .label("ec_connector", "MooncakeConnector")
                        .label(
                            "ec_role",
                            if role == WorkerType::Encode {
                                "ec_producer"
                            } else {
                                "ec_consumer"
                            },
                        )
                        .build(),
                ))
                .unwrap();
            port += 1;
        }
    }
    registry
}

#[tokio::test]
async fn lora_initial_pd_and_epd_selection_cannot_choose_a_base_model_pair() {
    for mode in [
        WorkerSelectionMode::PrefillDecode,
        WorkerSelectionMode::EncodePrefillDecode,
    ] {
        for has_sglang in [true, false] {
            let registry = disaggregated_registry(has_sglang);
            let stage = WorkerSelectionStage::new(
                Arc::clone(&registry),
                Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
                mode,
                None,
            );
            for _ in 0..4 {
                let mut ctx = prepared_ctx(Arc::clone(&registry), None);
                let RequestType::Generate(req) = &mut ctx.input.request_type else {
                    panic!("generate request expected");
                };
                Arc::make_mut(req).lora_path = Some("adapter".into());
                let outcome = stage.execute(&mut ctx).await;
                if !has_sglang {
                    assert!(
                        outcome.is_err(),
                        "{mode:?} must refuse a base-model-only pool"
                    );
                    continue;
                }
                outcome.unwrap();
                match ctx.state.workers.as_ref().unwrap() {
                    WorkerSelection::Disaggregated {
                        prefill,
                        decode,
                        runtime_type,
                        ..
                    } => {
                        assert_eq!(*runtime_type, RuntimeType::Sglang);
                        assert_eq!(prefill.metadata().spec.runtime_type, RuntimeType::Sglang);
                        assert_eq!(decode.metadata().spec.runtime_type, RuntimeType::Sglang);
                    }
                    WorkerSelection::Single { .. } => panic!("expected disaggregated selection"),
                }
                let mut dispatch = ctx.into_dispatch().unwrap();
                stage
                    .reselect(&mut dispatch, &retained_plan(&[1, 2, 3]))
                    .await
                    .unwrap();
                assert_eq!(dispatch.wire.runtime, RuntimeType::Sglang);
                match dispatch.workers.as_ref().unwrap() {
                    WorkerSelection::Disaggregated {
                        prefill, decode, ..
                    } => {
                        assert_eq!(prefill.metadata().spec.runtime_type, RuntimeType::Sglang);
                        assert_eq!(decode.metadata().spec.runtime_type, RuntimeType::Sglang);
                    }
                    WorkerSelection::Single { .. } => {
                        panic!("expected disaggregated retry selection")
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn lora_epd_media_refuses_an_unsupported_handoff_instead_of_running_vllm() {
    let stage = WorkerSelectionStage::new(
        disaggregated_registry(true),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        WorkerSelectionMode::EncodePrefillDecode,
        None,
    );
    for wire in [None, Some(grpc_wire(RuntimeType::Sglang))] {
        let result = stage.select_encode_prefill_decode_workers(
            "test-model",
            PlacementInputs::default(),
            &[vec![1]],
            EpdDemands {
                wire,
                encode: MediaDemands::NONE,
                decode_vision: false,
                image_fanout: false,
            },
            None,
        );
        if wire.is_none() {
            let (_, _, _, runtime) =
                result.unwrap_or_else(|_| panic!("the pool has a compatible vLLM EPD set"));
            assert_eq!(runtime, RuntimeType::Vllm);
        } else {
            // SGLang currently has no gateway encode handoff. LoRA must
            // fail here, never fall back to the vLLM base-model set.
            assert!(
                result.is_err(),
                "LoRA EPD must refuse the unsupported handoff"
            );
        }
    }
}

#[tokio::test]
async fn lora_initial_selection_and_retry_stay_on_sglang_in_a_mixed_pool() {
    let registry = Arc::new(WorkerRegistry::new());
    for (port, runtime) in [(9501, RuntimeType::Vllm), (9502, RuntimeType::Sglang)] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                    .model(ModelCard::new("test-model"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(runtime)
                    .health_config(no_health_check())
                    .build(),
            ))
            .unwrap();
    }
    let stage = regular_stage(Arc::clone(&registry));
    for (request, selector) in [
        (RequestType::Generate(Arc::new(serde_json::from_value(serde_json::json!({
            "model": "test-model", "text": "hi", "lora_path": "adapter", "cache_salt": "unforwarded"
        })).unwrap())), "adapter"),
        (RequestType::Generate(Arc::new(serde_json::from_value(serde_json::json!({
            "model": "test-model", "text": "hi", "lora_id": "engine-id"
        })).unwrap())), "engine-id"),
        (RequestType::Chat(Arc::new(serde_json::from_value(serde_json::json!({
            "model": "test-model", "messages": [], "lora_path": "adapter"
        })).unwrap())), "adapter"),
        (RequestType::Completion(Arc::new(serde_json::from_value(serde_json::json!({
            "model": "test-model", "prompt": "hi", "lora_path": "adapter"
        })).unwrap())), "adapter"),
    ] {
        let mut ctx = prepared_ctx(Arc::clone(&registry), None);
        let openai = matches!(&request, RequestType::Chat(_) | RequestType::Completion(_));
        ctx.input.request_type = request;
        stage.execute(&mut ctx).await.unwrap();
        let namespace = ctx.state.routing_snapshot.as_ref().unwrap().cache_namespace;
        assert_eq!(namespace, CacheNamespace::derive(&CachePartition {
            cache_salt: None, extra_key: None, lora_path: Some(selector),
        }));
        assert!(namespace.is_some(), "adapter affinity must be separate from the base model");
        match ctx.state.workers.as_ref().unwrap() {
            WorkerSelection::Single { worker } => assert_eq!(worker.metadata().spec.runtime_type, RuntimeType::Sglang),
            WorkerSelection::Disaggregated { .. } => panic!("expected regular selection"),
        }
        let mut dispatch = ctx.into_dispatch().unwrap();
        assert_eq!(dispatch.dispatch_model,
            if openai { format!("test-model:{selector}") } else { "test-model".to_string() });
        assert_eq!(dispatch.wire.runtime, RuntimeType::Sglang);
        for _ in 0..4 {
            stage.reselect(&mut dispatch, &retained_plan(&[1, 2, 3])).await.unwrap();
            match dispatch.workers.as_ref().unwrap() {
                WorkerSelection::Single { worker } => assert_eq!(worker.metadata().spec.runtime_type, RuntimeType::Sglang),
                WorkerSelection::Disaggregated { .. } => panic!("expected regular selection"),
            }
        }
    }
}

#[tokio::test]
async fn lora_requests_cannot_fall_back_to_a_base_model_only_runtime() {
    let registry = Arc::new(WorkerRegistry::new());
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:9510")
                .model(ModelCard::new("test-model"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();
    let stage = regular_stage(Arc::clone(&registry));
    let mut ctx = prepared_ctx(registry, None);
    let RequestType::Generate(req) = &mut ctx.input.request_type else {
        panic!("generate request expected");
    };
    Arc::make_mut(req).lora_path = Some("adapter".into());
    assert!(stage.execute(&mut ctx).await.is_err());
}
