//! The lines a stage's answer writes at the levels an operator reads by default. A refusal that
//! is the request's to fix is never an ERROR line: the stage that names the cause writes one WARN
//! and marks the answer, so the pipeline's stage line and the access log's response line stay at
//! DEBUG; a refusal no stage writes a line for is left to the access log's WARN line. A stage
//! writes that WARN only for a cause typed as the request's own. Every other failure keeps an
//! ERROR line: the stage's own where it answers 400 (a model configuration found nowhere, a
//! tokenizer with no chat template), the pipeline's too in the server's class (a model
//! configuration this host holds and cannot load, a vocabulary without the placeholder token).

use tracing::Level;

use super::*;
use crate::{
    observability::logging::test_support::{test_subscriber, LinesAtWarnOrAbove},
    routers::grpc::multimodal::{hub_answering, HubEnv},
};

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

/// The answer to one image for a model whose directory holds `config` as its `config.json`
/// ([`image_answer_over_model_dir`]). The directory is a local one and lives to the end of the
/// awaited call: the configuration is read on this host, with no fetch.
async fn image_answer_over_model_config(config: &str) -> (Response, Vec<(Level, String)>) {
    let model_dir = tempfile::tempdir().expect("a model directory");
    std::fs::write(model_dir.path().join("config.json"), config).expect("the model's config");
    image_answer_over_model_dir(model_dir.path()).await
}

/// The answer to one image for a model whose source is `model_dir` (a local directory, or a
/// repository id for a test's hub), under router-side processing over the mock tokenizer, with
/// the lines at WARN or above the request wrote.
async fn image_answer_over_model_dir(model_dir: &Path) -> (Response, Vec<(Level, String)>) {
    let (pipeline, pooled, _) = chat_pipeline_over(RuntimeType::Sglang, &[], model_dir).await;
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            model_dir.to_str().expect("a UTF-8 path"),
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

/// A fault of the deployment in the same preparation is answered in the server's class: a 500
/// under the pipeline's own code, whose sentence does not carry `cause`. The stage's line
/// stays ERROR and the answer unmarked, so an operator still reads it, and the pipeline adds
/// its own line as for every failure of that class.
async fn a_deployment_fault_keeps_the_stages_error_line(config: &str, cause: &str) {
    let (response, lines) = image_answer_over_model_config(config).await;

    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "multimodal_processing_failed"
    );
    assert!(!marked(&response));
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(
        lines[0],
        (
            Level::ERROR,
            "Failed to prepare the multimodal prompt plan".to_string()
        )
    );
    assert!(
        lines[1].0 == Level::ERROR && lines[1].1.starts_with("Stage "),
        "{lines:?}"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the answer's body");
    let body = String::from_utf8_lossy(&body);
    assert!(!body.contains(cause), "{body}");
    assert!(body.contains("the request is not at fault"), "{body}");
}

/// Media to a model whose configuration this host cannot load (a `config.json` that does not
/// parse) is the deployment's to fix, and its answer is a 500.
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

/// Media to a model whose configuration is found nowhere keeps the answer it had: a directory
/// of this host with no `config.json`, which the hub is not asked for, or a repository the hub
/// answers that it has none for (404) or does not open (401, 403). The gateway cannot tell
/// whether such a model takes media, so the request may be the one at fault: a 400 that names
/// the cause, unmarked, with the stage's ERROR line and none of the pipeline's. A hub that
/// fails to serve a repository's file (503) is the hub's fault, a 502. No network: the hub is a
/// loopback listener.
#[tokio::test]
async fn media_of_a_model_whose_config_is_found_nowhere_keeps_the_answer_it_had() {
    use std::sync::atomic::{AtomicU16, Ordering};

    let home = tempfile::tempdir().expect("a hub cache");
    let model_dir = tempfile::tempdir().expect("a model directory with no config.json");
    let repository = Path::new("cell/absent");
    let hub_answer = Arc::new(AtomicU16::new(404));
    let endpoint = hub_answering(hub_answer.clone()).await;
    let _env = HubEnv::set(home.path(), &endpoint).await;
    let stage_line = [(
        Level::ERROR,
        "Failed to prepare the multimodal prompt plan".to_string(),
    )];

    // The directory's turn runs under the hub's 503: a lookup would make its answer a 502.
    for (source, status) in [
        (model_dir.path(), 503),
        (repository, 404),
        (repository, 401),
        (repository, 403),
    ] {
        hub_answer.store(status, Ordering::SeqCst);
        let (response, lines) = image_answer_over_model_dir(source).await;

        assert_eq!(
            response.status(),
            http::StatusCode::BAD_REQUEST,
            "the hub's {status}"
        );
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "invalid_multimodal_request",
            "the hub's {status}"
        );
        assert!(!marked(&response), "the hub's {status}");
        assert_eq!(lines, stage_line, "the hub's {status}");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains("Failed to resolve model config directory"),
            "the hub's {status}: {body}"
        );
    }

    hub_answer.store(503, Ordering::SeqCst);
    let (response, lines) = image_answer_over_model_dir(repository).await;
    assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "multimodal_config_unavailable"
    );
    assert!(!marked(&response));
    assert_eq!(lines.first(), stage_line.first(), "{lines:?}");
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

/// A chat template that fails on its own account whatever the request holds (here it asks for
/// another template, and the model has this one alone) is the deployment's to fix, and answered
/// in the server's class so that it alerts. Both doors answer a 500 `chat_template_render_failed`
/// under the render failure's text, unmarked, with the stage's ERROR line and the pipeline's own
/// after it.
#[tokio::test]
async fn a_template_that_fails_on_its_own_is_a_500_with_the_error_lines_on_both_doors() {
    let tokenizer = Arc::new(
        MockTokenizer::new()
            .with_chat_template("{% include 'tools.jinja' %}")
            .expect("a template that compiles and cannot render"),
    ) as Arc<dyn Tokenizer>;
    let (chat, messages, components) = both_doors_over(tokenizer).await;
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let answers = one_turn_through_both_doors(&chat, &messages, components).await;

    for response in &answers {
        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            error::extract_error_code_from_response(response),
            "chat_template_render_failed"
        );
        assert!(!marked(response));
    }
    let lines = lines.lines();
    assert_eq!(lines.len(), 4, "{lines:?}");
    assert_eq!(
        [&lines[0], &lines[2]],
        [
            &(Level::ERROR, "Failed to process chat messages".to_string()),
            &(Level::ERROR, "Failed to process messages".to_string()),
        ]
    );
    for line in [&lines[1], &lines[3]] {
        assert!(
            line.0 == Level::ERROR && line.1.starts_with("Stage "),
            "{lines:?}"
        );
    }
    for response in answers {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains(
                "Failed to apply chat template: Failed to render template: template not found: "
            ),
            "{body}"
        );
    }
}

/// A render that stops on a value stays the 400 it was on both doors, unmarked, with the stage's
/// ERROR line and none of the pipeline's: the request may be the cause. The first template reads
/// a second message, which a conversation of one turn does not have. The second iterates over a
/// number and fails on every conversation: the template is at fault there, but its error is of
/// the kind a request's value brings about, which the render cannot tell apart, so it is not
/// answered as the deployment's. The third calls a macro it declares only for a conversation
/// that opens with `hello`: the name is unknown for this one, and the conversation decided that.
#[tokio::test]
async fn a_render_that_stops_on_a_value_stays_a_400_on_both_doors() {
    for template in [
        "{{ messages[1]['content'] }}",
        "{% for part in 42 %}{{ part }}{% endfor %}",
        "{% if messages[0].content == 'hello' %}{% macro greet() %}hi there{% endmacro %}\
         {% endif %}{{ greet() }}",
    ] {
        let tokenizer = Arc::new(
            MockTokenizer::new()
                .with_chat_template(template)
                .expect("a template that compiles"),
        ) as Arc<dyn Tokenizer>;
        let (chat, messages, components) = both_doors_over(tokenizer).await;
        let lines = LinesAtWarnOrAbove::default();
        let _subscriber = test_subscriber(lines.clone());

        let answers = one_turn_through_both_doors(&chat, &messages, components).await;

        for response in &answers {
            assert_eq!(
                response.status(),
                http::StatusCode::BAD_REQUEST,
                "{template}"
            );
            assert_eq!(
                error::extract_error_code_from_response(response),
                "process_messages_failed",
                "{template}"
            );
            assert!(!marked(response), "{template}");
        }
        assert_eq!(
            lines.lines(),
            [
                (Level::ERROR, "Failed to process chat messages".to_string()),
                (Level::ERROR, "Failed to process messages".to_string()),
            ],
            "{template}"
        );
    }
}

/// A `chat_template_kwargs` entry can break a sound template's render: one under the name of a
/// function the template calls (its own `raise_exception` among them), one of a kind the
/// template does not expect, one that names a template the model does not have. The request is
/// the cause, and the chat door answers the 400 `process_messages_failed`, not the deployment's
/// 500.
#[tokio::test]
async fn a_template_argument_that_breaks_the_render_stays_a_400() {
    for (template, kwargs, kind) in [
        (
            "{% set ns = namespace(found=false) %}{{ ns.found }}",
            serde_json::json!({"namespace": 0}),
            "invalid operation",
        ),
        (
            "{{ raise_exception('the roles must alternate') }}",
            serde_json::json!({"raise_exception": 0}),
            "invalid operation",
        ),
        (
            "{{ 'Today Date: ' + date_string }}",
            serde_json::json!({"date_string": 20_261_120}),
            "invalid operation",
        ),
        (
            "{% include fragments %}",
            serde_json::json!({"fragments": ["missing.jinja"]}),
            "template not found",
        ),
    ] {
        let tokenizer = Arc::new(
            MockTokenizer::new()
                .with_chat_template(template)
                .expect("a template that compiles"),
        ) as Arc<dyn Tokenizer>;
        let (chat, _, components) = both_doors_over(tokenizer).await;
        let lines = LinesAtWarnOrAbove::default();
        let _subscriber = test_subscriber(lines.clone());

        let response = chat_response(
            &chat,
            components,
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": kwargs
            }),
        )
        .await;

        assert_eq!(
            response.status(),
            http::StatusCode::BAD_REQUEST,
            "{template}"
        );
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "process_messages_failed",
            "{template}"
        );
        assert!(!marked(&response), "{template}");
        assert_eq!(
            lines.lines(),
            [(Level::ERROR, "Failed to process chat messages".to_string())],
            "{template}"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the answer's body");
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains(&format!(
                "Failed to apply chat template: Failed to render template: {kind}: "
            )),
            "{template}: {body}"
        );
    }
}

/// The chat door's answer to a request that continues a final assistant message with reasoning,
/// over a model whose chat template is `template`, and the lines at WARN or above it wrote. The
/// render cannot fall back on the message's text after a generation prompt: the reasoning would
/// be lost.
async fn a_message_with_reasoning_continued_over(
    template: &str,
) -> (Response, Vec<(Level, String)>) {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
            .expect("a tokenizer");
    tokenizer
        .set_chat_template(template.to_string())
        .expect("a template that compiles");
    let (chat, _, components) = both_doors_over(Arc::new(tokenizer)).await;
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = chat_response(
        &chat,
        components,
        serde_json::json!({
            "model": MODEL,
            "continue_final_message": true,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Partial", "reasoning_content": "still thinking"}
            ]
        }),
    )
    .await;

    (response, lines.lines())
}

/// A continuation the render refuses is the request's and stays a 400: a final assistant message
/// the template does not write cannot be continued, and its reasoning would be lost after the
/// generation prompt. Nobody types that check, so the stage keeps its ERROR line and the answer
/// is unmarked.
#[tokio::test]
async fn a_continuation_the_render_refuses_stays_a_400() {
    let (response, lines) = a_message_with_reasoning_continued_over(
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}\n{% endif %}{% endfor %}",
    )
    .await;

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "process_messages_failed"
    );
    assert!(!marked(&response));
    assert_eq!(
        lines,
        [(Level::ERROR, "Failed to process chat messages".to_string())]
    );
}

/// The same continuation over a template that fails on its own account is the deployment's 500
/// `chat_template_render_failed`: the template failed before the render could continue anything,
/// and the reasoning that would be lost is said after that failure, not in its place.
#[tokio::test]
async fn a_continuation_over_a_template_that_fails_on_its_own_is_a_500() {
    let (response, lines) =
        a_message_with_reasoning_continued_over("{% include 'tools.jinja' %}").await;

    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "chat_template_render_failed"
    );
    assert!(!marked(&response));
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(
        lines[0],
        (Level::ERROR, "Failed to process chat messages".to_string())
    );
    assert!(
        lines[1].0 == Level::ERROR && lines[1].1.starts_with("Stage "),
        "{lines:?}"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the answer's body");
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("Failed to render template: template not found: ")
            && body.contains("its reasoning_content would be lost"),
        "{body}"
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

/// A generate input only the request can put right is refused with one WARN that carries the
/// cause, the answer marked, no ERROR.
#[tokio::test]
async fn a_generate_input_the_request_gets_wrong_is_one_warn_and_marked() {
    // Past the mock's vocabulary and none of its added tokens; no worker reports a table.
    let unknown = MockTokenizer::new().vocab_size() as u32 + 7;
    for body in [
        serde_json::json!({"model": MODEL, "input_ids": [[1, 2], [3]]}),
        serde_json::json!({"model": MODEL, "input_ids": [1, -2, 3]}),
        serde_json::json!({"model": MODEL, "input_ids": [1, unknown]}),
        serde_json::json!({"model": MODEL}),
    ] {
        let input = body.to_string();
        // The chat door's pipeline is the generate route's too.
        let (pipeline, _, components) = both_doors_over(Arc::new(MockTokenizer::new())).await;
        let request: Arc<GenerateRequest> =
            Arc::new(serde_json::from_value(body).expect("generate request"));
        let lines = LinesAtWarnOrAbove::default();
        let _subscriber = test_subscriber(lines.clone());

        let response = pipeline
            .execute_generate(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST, "{input}");
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "resolve_input_failed",
            "{input}"
        );
        assert!(marked(&response), "{input}");
        assert_eq!(
            lines.lines(),
            [(Level::WARN, "Failed to resolve generate input".to_string())],
            "{input}"
        );
    }
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
