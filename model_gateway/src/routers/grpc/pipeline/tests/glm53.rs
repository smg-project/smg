//! GLM-5.3 checkpoint rendering through the normal gRPC pipelines.

use super::*;
use crate::{
    routers::grpc::common::stages::selection_context_length, worker::test_support::no_health_check,
};

/// Build a checkpoint-backed tokenizer without any multimodal components.
async fn glm53_components(worker_registry: Arc<WorkerRegistry>) -> Arc<SharedComponents> {
    let components = glm53_checkpoint_components(
        worker_registry,
        serde_json::json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]}),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/tokenizer/tests/fixtures/glm53/chat_template.jinja"
        )),
    )
    .await;
    assert_eq!(
        components
            .tokenizer_registry
            .get(MODEL)
            .unwrap()
            .renderer_capabilities()
            .media_handling,
        llm_tokenizer::traits::MediaHandling::TemplateText
    );
    components
}

/// Load a minimal tokenizer with the selected GLM checkpoint's config and template.
async fn glm53_checkpoint_components(
    worker_registry: Arc<WorkerRegistry>,
    config: serde_json::Value,
    template: &str,
) -> Arc<SharedComponents> {
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
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), template).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(llm_tokenizer::CachedTokenizer::new(
        Arc::new(tokenizer),
        Default::default(),
    ));
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
        spec::ResponseSpec,
    };
    // A transport listener is enough: construction never calls an engine RPC.
    let port = spawn_stub(GatedScheduler::default()).await;
    // The window the fixture's prefill and decode workers advertise.
    const WINDOW: u32 = 8_192;
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
                .model(ModelCard::new(MODEL).with_context_length(WINDOW))
                .label(SUPPORTS_VISION_LABEL, "false")
                .health_config(no_health_check())
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
                false,
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
            RequestContext::for_messages_carrying_its_tools(
                Arc::new(request),
                None,
                MODEL.into(),
                components,
            )
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
        if let ResponseSpec::Messages(spec) = &built.spec {
            assert_eq!(
                spec.prompt_tokens as usize,
                token_ids.len(),
                "the Messages spec carries the rendered prompt's count"
            );
            let window = selection_context_length(
                ctx.state.workers.as_ref().expect("selected workers"),
                &ctx.input.model_id,
            );
            assert_eq!(window, Some(WINDOW), "the pair advertises its window");
            assert_eq!(
                spec.context_window, window,
                "the Messages spec carries the selected workers' window"
            );
        }
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

/// A renderer that writes media as template text has no plan to build, and
/// the request's media is recorded all the same: the per-request series
/// count what a request carries.
#[test]
fn glm53_media_is_recorded_before_the_template_renders_it_as_text() {
    fn sample(body: &str, name: &str, labels: &[&str]) -> Option<f64> {
        body.lines()
            .find(|line| line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
            .and_then(|line| line.rsplit(' ').next()?.parse().ok())
    }

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    ::metrics::with_local_recorder(&recorder, || {
        runtime.block_on(async {
            let components = glm53_components(Arc::new(WorkerRegistry::new())).await;
            let request: CreateMessageRequest = serde_json::from_value(serde_json::json!({
                "model": MODEL, "max_tokens": 32, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "invalid"}},
                    {"type": "text", "text": "after"}
                ]}]
            }))
            .unwrap();
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
                true,
            )
            .await
            .unwrap();
            assert!(media.media.is_none());
            assert_eq!(media.media_order, llm_multimodal::MediaPartOrder::Authored);
        });
    });
    let body = handle.render();
    let model = format!("model=\"{MODEL}\"");
    assert_eq!(
        sample(
            &body,
            "smg_mm_media_per_request_count",
            &[model.as_str(), "modality=\"image\""]
        ),
        Some(1.0),
        "{body}"
    );
    assert_eq!(
        sample(
            &body,
            "smg_mm_media_items_total",
            &[model.as_str(), "modality=\"image\"", "source=\"data_url\""]
        ),
        Some(1.0),
        "{body}"
    );
}

/// Chat preparation renders a `medium` effort on a GLM-5.3 model as
/// `High`, the level z.ai maps it to (issue #5181).
#[tokio::test]
async fn glm53_medium_effort_is_rendered_as_high() {
    use openai_protocol::validated::Normalizable;

    let port = spawn_stub(GatedScheduler::default()).await;
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
    let mut request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "zai-org/GLM-5.3",
        "max_tokens": 8,
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    request.normalize();
    assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
    let mut ctx = RequestContext::for_chat(Arc::new(request), None, MODEL.into(), components);
    pipeline.stages.preparation.execute(&mut ctx).await.unwrap();
    let Some(PreparationOutput::Chat {
        processed_messages, ..
    }) = ctx.state.preparation.as_ref()
    else {
        panic!("chat preparation")
    };
    let prompt = &processed_messages.text;
    assert!(prompt.contains("Reasoning Effort: High"), "{prompt}");
}

/// Chat preparation expands a Flash tool reference to its declared function schema.
#[tokio::test]
async fn glm53_flash_tool_reference_reaches_the_prompt() {
    use axum::{body::Body, extract::FromRequest, http::Request};
    use openai_protocol::{profile::ProviderProfile, validated::ValidatedJson};
    use serde_json::json;

    const MODEL_ID: &str = "zai-org/GLM-5.3-Flash";
    const ALIAS: &str = "weather-reference-test-alias";
    const TEXT_ALIAS: &str = "weather-reference-text-test-alias";
    ProviderProfile::register_model_aliases([(ALIAS, MODEL_ID), (TEXT_ALIAS, "glm-5.3")]);
    let registry = Arc::new(WorkerRegistry::new());
    let deps = PipelineDeps::pair(
        registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).unwrap();
    let components = glm53_checkpoint_components(
        registry,
        json!({"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"]}),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/tokenizer/tests/fixtures/glm53/chat_template_flash.jinja"
        )),
    )
    .await;

    let reference = json!({"type": "tool_reference", "name": "weather"});
    let request = json!({
        "messages": [
            {"role": "user", "content": "Find a weather tool"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_search", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_search", "content": [reference]}
        ],
        "tools": [
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}
        ]
    });
    let undeclared = json!({"type": "tool_reference", "name": "undeclared"});
    let missing_name = json!({"type": "tool_reference"});
    let text = json!({"type": "text", "text": "extra"});
    let cases = [
        (MODEL_ID, "tool", json!([reference]), true),
        ("glm-5.3-flash", "tool", json!([reference]), true),
        ("GLM-5.3-Flash-BF16", "tool", json!([reference]), true),
        (ALIAS, "tool", json!([reference]), true),
        (TEXT_ALIAS, "tool", json!([reference]), false),
        ("glm-5.3", "tool", json!([reference]), false),
        ("glm-5.3-flashlight", "tool", json!([reference]), false),
        ("local-model", "tool", json!([reference]), false),
        (MODEL_ID, "user", json!([reference]), false),
        (MODEL_ID, "tool", json!([undeclared]), false),
        (MODEL_ID, "tool", json!([missing_name]), false),
        (
            MODEL_ID,
            "tool",
            json!([{"type": "tool_reference", "name": 42}]),
            false,
        ),
        (MODEL_ID, "tool", json!([reference, text]), false),
        (MODEL_ID, "tool", json!([text, reference]), false),
    ];
    let filtered_cases = ["weather", "search"].map(|name| {
        (
            MODEL_ID,
            "tool",
            json!([reference]),
            name == "weather",
            json!({"type": "allowed_tools", "mode": "auto",
                "tools": [{"type": "function", "name": name}]}),
        )
    });
    for (model, role, content, accepted, tool_choice) in cases
        .into_iter()
        .map(|(model, role, content, accepted)| (model, role, content, accepted, json!(null)))
        .chain(filtered_cases)
    {
        let mut wire = request.clone();
        wire["model"] = json!(model);
        wire["tool_choice"] = tool_choice;
        wire["messages"][2]["role"] = json!(role);
        wire["messages"][2]["content"] = content;
        let http_request = Request::builder()
            .header("content-type", "application/json")
            .body(Body::from(wire.to_string()))
            .unwrap();
        let ValidatedJson(request) =
            ValidatedJson::<ChatCompletionRequest>::from_request(http_request, &())
                .await
                .unwrap();
        let mut ctx = RequestContext::for_chat(
            Arc::new(request),
            None,
            MODEL.into(),
            Arc::clone(&components),
        );
        let result = pipeline.stages.preparation.execute(&mut ctx).await;
        if accepted {
            if let Err(response) = result {
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                panic!("{model}: {}", String::from_utf8_lossy(&body));
            }
            let Some(PreparationOutput::Chat {
                processed_messages, ..
            }) = ctx.state.preparation.as_ref()
            else {
                panic!("chat preparation");
            };
            // Observation and generation suffix from the official template at 84c6a6a.
            assert!(
                processed_messages.text.ends_with(concat!(
                    "<|observation|><tool_response><tools>\n",
                    "{\"name\": \"weather\", \"parameters\": {\"type\": \"object\"}}\n",
                    "</tools></tool_response><|assistant|><think>"
                )),
                "{model}: {}",
                processed_messages.text
            );
        } else {
            let response = result.unwrap_err();
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body["error"]["code"], "unsupported_content_part",
                "{model}: {body}"
            );
        }
    }
}
