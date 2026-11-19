//! The lines a stage's answer writes at the levels an operator reads by default. A refusal that
//! is the request's to fix is never an ERROR line: the stage that names the cause writes one WARN
//! and marks the answer, so the pipeline's stage line and the access log's response line stay at
//! DEBUG; a refusal no stage writes a line for is left to the access log's WARN line. A stage
//! writes that WARN only for a cause typed as the request's own. Every other failure keeps an
//! ERROR line: the stage's own where it answers 400 for a fault of the deployment (a model
//! configuration that does not load, a vocabulary without the placeholder token, a tokenizer
//! with no chat template), the pipeline's too in the server's class.

use tracing::Level;

use super::*;
use crate::observability::logging::test_support::{test_subscriber, LinesAtWarnOrAbove};

fn marked(response: &Response) -> bool {
    response.extensions().get::<ReportedRefusal>().is_some()
}

/// A refusal no stage writes a line for (an image to a pool whose engines run without a vision
/// encoder) is the client's mistake, not the pipeline's ERROR: nothing is written at WARN or
/// above here and the answer is left unmarked, so the access log's WARN line is its one line.
#[tokio::test]
async fn a_refusal_no_stage_reports_is_left_to_the_access_logs_line() {
    let model_dir = unreadable_model_dir();
    let (pipeline, components, _) = chat_pipeline_over(
        RuntimeType::TokenSpeed,
        LANGUAGE_MODEL_ONLY,
        model_dir.path(),
    )
    .await;
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = chat_response(&pipeline, components, one_pixel_image_chat()).await;

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "multimodal_not_supported"
    );
    assert!(!marked(&response));
    assert!(lines.lines().is_empty(), "{:?}", lines.lines());
}

/// The answer to one image for a model whose directory holds `config` as its `config.json`,
/// under router-side processing over the mock tokenizer, with the lines at WARN or above the
/// request wrote. The directory is a local one and lives to the end of the awaited call: the
/// configuration is read on this host, with no fetch.
async fn image_answer_over_model_config(config: &str) -> (Response, Vec<(Level, String)>) {
    let model_dir = tempfile::tempdir().expect("a model directory");
    std::fs::write(model_dir.path().join("config.json"), config).expect("the model's config");
    let (pipeline, pooled, _) =
        chat_pipeline_over(RuntimeType::Sglang, &[], model_dir.path()).await;
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            model_dir.path().to_str().expect("a UTF-8 path"),
            || async move { Ok(tokenizer) },
        )
        .await
        .expect("load the tokenizer");
    let mut multimodal = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .expect("multimodal components");
    multimodal.processing = openai_protocol::worker::MmProcessingMode::Router;
    let mut components =
        shared_components(tokenizer_registry, pooled.worker_registry.clone(), false);
    Arc::get_mut(&mut components)
        .expect("the components are not shared yet")
        .multimodal = Some(Arc::new(multimodal));
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = chat_response(&pipeline, components, one_pixel_image_chat()).await;

    (response, lines.lines())
}

/// Media to a model whose family has no media spec, under router-side processing, is the
/// request's to fix: the stage writes one WARN with the cause and marks the answer, and neither
/// it nor the pipeline writes an ERROR.
#[tokio::test]
async fn media_the_preparation_refuses_is_one_warn_and_marked() {
    let (response, lines) = image_answer_over_model_config(r#"{"model_type":"llama"}"#).await;

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "invalid_multimodal_request"
    );
    assert!(marked(&response));
    assert_eq!(
        lines,
        [(
            Level::WARN,
            "Failed to prepare the multimodal prompt plan".to_string()
        )]
    );
}

/// A fault of the deployment in the same preparation, whose 400 names `cause`: the stage's line
/// stays ERROR and the answer unmarked, so an operator still reads it.
async fn a_deployment_fault_keeps_the_stages_error_line(config: &str, cause: &str) {
    let (response, lines) = image_answer_over_model_config(config).await;

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "invalid_multimodal_request"
    );
    assert!(!marked(&response));
    assert_eq!(
        lines,
        [(
            Level::ERROR,
            "Failed to prepare the multimodal prompt plan".to_string()
        )]
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the answer's body");
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains(cause), "{body}");
}

/// Media to a model whose configuration this host cannot load (a `config.json` that does not
/// parse) is the deployment's to fix, though its answer is a 400.
#[tokio::test]
async fn media_of_a_model_whose_config_cannot_be_loaded_keeps_the_stages_error_line() {
    a_deployment_fault_keeps_the_stages_error_line("{", "Failed to parse config.json").await;
}

/// So is media to a model whose placeholder token the tokenizer's vocabulary lacks (the
/// tokenizer of another checkpoint than the configuration's).
#[tokio::test]
async fn media_for_a_vocabulary_without_the_placeholder_token_keeps_the_stages_error_line() {
    a_deployment_fault_keeps_the_stages_error_line(
        r#"{"model_type":"llava"}"#,
        "placeholder token '<image>' is missing from the tokenizer vocabulary",
    )
    .await;
}

/// The chat and the Messages pipelines over `tokenizer` and no worker: both render the
/// conversation through the tokenizer's chat template in their preparation stage.
async fn both_doors_over(
    tokenizer: Arc<dyn Tokenizer>,
) -> (RequestPipeline, RequestPipeline, Arc<SharedComponents>) {
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .expect("load the tokenizer");
    let worker_registry = Arc::new(WorkerRegistry::new());
    let components = shared_components(tokenizer_registry, worker_registry.clone(), false);
    let deps = PipelineDeps::pair(
        worker_registry,
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let chat = RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
    let messages = RequestPipeline::build(Endpoint::Messages, Mode::Regular, &deps)
        .expect("messages pipeline");
    (chat, messages, components)
}

/// One user turn through both doors of [`both_doors_over`]: the chat answer, then the Messages
/// one.
async fn one_turn_through_both_doors(
    chat: &RequestPipeline,
    messages: &RequestPipeline,
    components: Arc<SharedComponents>,
) -> [Response; 2] {
    let from_chat = chat_response(
        chat,
        components.clone(),
        serde_json::json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let request: Arc<CreateMessageRequest> = Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("messages request"),
    );
    let tools = MessagesTools::from(&*request);
    let from_messages = messages
        .execute_messages(
            request,
            tools,
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            None,
        )
        .await;
    [from_chat, from_messages]
}

/// A conversation the model's chat template refuses through its own `raise_exception` is the
/// client's on both doors that render through it: one WARN each, the answer marked, no ERROR.
#[tokio::test]
async fn a_conversation_the_template_refuses_is_one_warn_and_marked_on_both_doors() {
    let tokenizer = Arc::new(
        MockTokenizer::new()
            .with_chat_template("{{ raise_exception('the roles must alternate') }}")
            .expect("a template that refuses"),
    ) as Arc<dyn Tokenizer>;
    let (chat, messages, components) = both_doors_over(tokenizer).await;
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let answers = one_turn_through_both_doors(&chat, &messages, components).await;

    for response in &answers {
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(response),
            "process_messages_failed"
        );
        assert!(marked(response));
    }
    assert_eq!(
        lines.lines(),
        [
            (Level::WARN, "Failed to process chat messages".to_string()),
            (Level::WARN, "Failed to process messages".to_string()),
        ]
    );
}

/// A tokenizer with no chat template fails every request at the render, whatever the request
/// holds: the deployment's to fix. Both doors keep the stage's ERROR line, leave the answer
/// unmarked and say in it what the model lacks and that no request can supply it.
#[tokio::test]
async fn a_tokenizer_with_no_chat_template_keeps_the_error_line_on_both_doors() {
    let tokenizer_dir = tempfile::tempdir().expect("a tokenizer directory");
    let tokenizer_file = tokenizer_dir.path().join("tokenizer.json");
    std::fs::write(
        &tokenizer_file,
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"hi":1},"unk_token":"[UNK]"}}"#,
    )
    .expect("the tokenizer file");
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_file.to_str().expect("a UTF-8 path"))
            .expect("a tokenizer with no chat template"),
    ) as Arc<dyn Tokenizer>;
    let (chat, messages, components) = both_doors_over(tokenizer).await;
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let answers = one_turn_through_both_doors(&chat, &messages, components).await;

    for response in &answers {
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(response),
            "process_messages_failed"
        );
        assert!(!marked(response));
    }
    assert_eq!(
        lines.lines(),
        [
            (Level::ERROR, "Failed to process chat messages".to_string()),
            (Level::ERROR, "Failed to process messages".to_string()),
        ]
    );
    for response in answers {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains(
                "Failed to apply chat template: The model has no chat template. One must be \
                 configured for the model; a request cannot supply it."
            ),
            "{body}"
        );
    }
}

/// A Responses input the chat pipeline has no form for is refused by the conversion: one WARN
/// with the cause, the answer marked, no ERROR.
#[tokio::test]
async fn a_responses_input_the_conversion_refuses_is_one_warn_and_marked() {
    let ctx = responses_context(&Arc::new(WorkerRegistry::new()), 0).await;
    let request: Arc<ResponsesRequest> = Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_file", "file_data": "JVBERi0xLjQK", "filename": "a.pdf"}
            ]}],
        }))
        .expect("responses request"),
    );
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = route_responses(&ctx, request, None, anonymous(), MODEL.to_string()).await;

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "unsupported_content_part"
    );
    assert!(marked(&response));
    assert_eq!(
        lines.lines(),
        [(
            Level::WARN,
            "Failed to convert ResponsesRequest to ChatCompletionRequest".to_string()
        )]
    );
}

/// A failure in the server's class keeps its ERROR lines, the pipeline's among them: a served
/// model with no tokenizer on a gateway that loads none itself.
#[tokio::test]
async fn a_failure_in_the_server_class_keeps_the_pipelines_error_line() {
    let port = spawn_stub(GatedScheduler::default()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(Arc::new(TokenizerRegistry::new()))),
        tokenizer_registry: Arc::new(TokenizerRegistry::new()),
        worker_registry,
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: false,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    });
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = pipeline
        .execute_completion(
            completion_request(false),
            None,
            MODEL.to_string(),
            components,
            None,
            None,
            Some(&fast_retry_config(1)),
        )
        .await;

    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!marked(&response));
    let lines = lines.lines();
    assert!(
        lines
            .iter()
            .any(|(level, line)| *level == Level::ERROR && line.starts_with("Stage ")),
        "{lines:?}"
    );
}
