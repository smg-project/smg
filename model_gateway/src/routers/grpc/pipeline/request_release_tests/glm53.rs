//! GLM-5.3 media reminders through the normal gRPC pipelines.

use super::*;

/// Build a checkpoint-backed tokenizer without any multimodal components.
async fn glm53_components(worker_registry: Arc<WorkerRegistry>) -> Arc<SharedComponents> {
    let components = components(worker_registry).await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    let mut vocab = serde_json::Map::new();
    for id in 0..=100 {
        vocab.insert(format!("token{id}"), serde_json::json!(id));
    }
    std::fs::write(
        &path,
        serde_json::json!({
            "version": "1.0", "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "token0"}
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/tokenizer/tests/fixtures/glm53/chat_template.jinja"
        )),
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(llm_tokenizer::CachedTokenizer::new(
        Arc::new(tokenizer),
        Default::default(),
    ));
    assert_eq!(
        tokenizer.renderer_capabilities().media_handling,
        llm_tokenizer::traits::MediaHandling::TemplateText
    );
    components.tokenizer_registry.remove_by_id("tokenizer-id");
    components
        .tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    components
}

/// Deliberately unusable URLs/data prove that the router does not fetch,
/// decode, or preprocess media. The serving alias intentionally is not GLM.
fn glm53_media_request(stream: bool) -> Arc<ChatCompletionRequest> {
    Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "stream": stream, "max_completion_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/never-fetch"}},
                {"type": "text", "text": "between"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,invalid"}},
                {"type": "input_audio", "input_audio": {"data": "invalid", "format": "wav"}},
                {"type": "text", "text": "after"}
            ]}]
        }))
        .unwrap(),
    )
}

/// Media reminders reach both streaming and non-streaming text-only workers.
#[tokio::test]
async fn glm53_media_returns_normal_completion_and_stream_without_media_processing() {
    for stream in [false, true] {
        let seen_mm = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_mm_placeholders: Arc::clone(&seen_mm),
            ..Default::default()
        })
        .await;
        let registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(&registry, port, WorkerType::Regular, LANGUAGE_MODEL_ONLY);
        let deps = PipelineDeps::pair(
            registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).unwrap();
        let components = glm53_components(registry).await;
        assert!(components.multimodal.is_none());
        let response = pipeline
            .execute_chat(
                glm53_media_request(stream),
                None,
                MODEL.into(),
                components,
                None,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        if stream {
            assert!(body.contains("data: [DONE]"), "{body}");
            assert!(body.contains("token100"), "{body}");
        } else {
            let body: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(body["choices"][0]["message"]["content"], "token100");
        }
        assert!(!body.contains("\"error\""), "{body}");
        assert_eq!(*seen_mm.lock().unwrap(), [Vec::<(u32, u32)>::new()]);
    }
}

/// Chat, Messages, and token counting agree on text-only PD inputs.
#[tokio::test]
async fn glm53_sglang_pd_builds_text_only_for_chat_and_messages() {
    use crate::routers::grpc::{
        backend_client::BackendClient, client::GrpcClient, proto_wrapper::ProtoGenerateRequest,
    };
    // A transport listener is enough: construction never calls an engine RPC.
    let port = spawn_stub(GatedScheduler::default()).await;
    for endpoint in [Endpoint::Chat, Endpoint::Messages] {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type) in [
            ("grpc://prefill:30000", WorkerType::Prefill),
            ("grpc://decode:30000", WorkerType::Decode),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Sglang)
                .model(ModelCard::new(MODEL))
                .label(SUPPORTS_VISION_LABEL, "false")
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            registry.register(Arc::new(worker)).unwrap();
        }
        let deps = PipelineDeps::pair(
            registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline = RequestPipeline::build(endpoint, Mode::PrefillDecode, &deps).unwrap();
        let components = glm53_components(registry).await;
        let mut counted_tokens = None;
        let mut ctx = if matches!(endpoint, Endpoint::Chat) {
            RequestContext::for_chat(glm53_media_request(false), None, MODEL.into(), components)
        } else {
            let request: CreateMessageRequest = serde_json::from_value(serde_json::json!({
                "model": MODEL, "max_tokens": 32, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "invalid"}},
                    {"type": "text", "text": "after"}
                ]}]
            })).unwrap();
            // count_tokens shares this media resolution; no MM bundle is required.
            let tokenizer = components
                .tokenizer_registry
                .get_by_name(MODEL)
                .unwrap()
                .tokenizer;
            let media = utils::message_utils::resolve_messages_media(
                &components,
                MODEL,
                tokenizer.as_ref(),
                &request.messages,
            )
            .await
            .unwrap();
            assert!(media.media.is_none());
            assert_eq!(media.media_order, llm_multimodal::MediaPartOrder::Authored);
            counted_tokens = Some(
                utils::message_utils::count_input_tokens(
                    (&request).into(),
                    tokenizer,
                    &[],
                    media.placeholders(),
                    media.media_order,
                )
                .await
                .unwrap(),
            );
            RequestContext::for_messages(Arc::new(request), None, MODEL.into(), components)
        };
        pipeline.stages.preparation.execute(&mut ctx).await.unwrap();
        assert!(ctx.state.multimodal_intermediate.is_none());
        assert!(ctx.state.multimodal_refs.is_none());
        let (token_ids, prompt) = match ctx.state.preparation.as_ref().unwrap() {
            PreparationOutput::Chat {
                token_ids,
                processed_messages,
                ..
            }
            | PreparationOutput::Messages {
                token_ids,
                processed_messages,
                ..
            } => (token_ids.clone(), &processed_messages.text),
            _ => panic!("chat-shaped preparation"),
        };
        if let Some(counted) = counted_tokens {
            assert_eq!(counted as usize, token_ids.len());
        }
        let reminder = "<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>";
        assert!(prompt.contains(&format!("before{reminder}")), "{prompt}");
        assert!(
            prompt.find(reminder).unwrap() < prompt.find("after").unwrap(),
            "{prompt}"
        );
        assert!(
            !prompt.contains("never-fetch") && !prompt.contains("invalid"),
            "{prompt}"
        );
        if matches!(endpoint, Endpoint::Chat) {
            assert!(
                prompt.contains("between<reminder>You are unable to process this video"),
                "{prompt}"
            );
            assert!(
                prompt.contains("<reminder>You are unable to process this audio"),
                "{prompt}"
            );
        }
        pipeline
            .stages
            .worker_selection
            .execute(&mut ctx)
            .await
            .unwrap();
        assert!(matches!(
            ctx.state.workers,
            Some(WorkerSelection::Disaggregated { .. })
        ));
        let client = BackendClient::Grpc(
            GrpcClient::connect(&format!("http://127.0.0.1:{port}"), "sglang")
                .await
                .unwrap(),
        );
        ctx.state.clients = Some(ClientSelection::Disaggregated {
            prefill: client.clone(),
            decode: client,
        });
        let mut built = pipeline
            .stages
            .request_building
            .build(&mut ctx)
            .await
            .unwrap();
        assert!(matches!(built.plan, ExecutionPlan::PrefillDecode(_)));
        for request in built.plan.generate_requests_mut() {
            let ProtoGenerateRequest::Sglang(request) = request else {
                panic!("SGLang proto")
            };
            assert!(request.mm_inputs.is_none());
            assert_eq!(request.tokenized.as_ref().unwrap().input_ids, token_ids);
            assert!(request.tokenized.as_ref().unwrap().original_text.is_empty());
        }
        assert!(ctx.state.multimodal_inflight.is_none());
    }
}
