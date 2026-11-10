//! The pieces every pipeline shares after dispatch: the media references attached once, the missing
//! stage reported by name.

use std::sync::Arc;

use llm_multimodal::MediaContentPart;
use llm_tokenizer::TokenizerRegistry;
use openai_protocol::generate::GenerateRequest;
use smg_grpc_client::VllmEngineClient;
use tokio::task::JoinHandle;

use super::*;
use crate::{
    routers::grpc::{
        client::GrpcClient, context::SharedComponents, multimodal::MediaPlan,
        regular::symphony::template::Templates, test_engine::mock_engine,
    },
    worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, WorkerRegistry, WorkerType},
};

/// A request context after worker selection and client acquisition: one
/// selected worker, a builder client on the mock engine, and the media
/// references of worker-side processing waiting to be attached.
async fn context_with_media_refs() -> (RequestContext, JoinHandle<()>) {
    let (url, server) = mock_engine("test-model").await;
    let client = VllmEngineClient::connect(&url)
        .await
        .expect("connect mock worker");
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(Arc::new(TokenizerRegistry::new()))),
        tokenizer_registry: Arc::new(TokenizerRegistry::new()),
        worker_registry: Arc::new(WorkerRegistry::new()),
        tool_parser_factory: Default::default(),
        reasoning_parser_factory: Default::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: crate::config::ParserChoice::Legacy,
    });
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "text": "Hello"
    }))
    .expect("a generate request");
    let mut ctx = RequestContext::for_generate(
        Arc::new(request),
        None,
        "test-model".to_string(),
        components,
    );
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url.replace("http://", "grpc://"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("test-model"))
            .build(),
    );
    ctx.state.workers = Some(WorkerSelection::Single { worker });
    ctx.state.clients = Some(ClientSelection::Single {
        client: BackendClient::Grpc(GrpcClient::Vllm(client)),
    });
    ctx.state.multimodal_refs = Some(MediaPlan::new([MediaContentPart::ImageUrl {
        url: "data:image/png;base64,AAAA".to_string(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }]));
    (ctx, server)
}

/// The media references are attached by the finishing every request
/// builder shares, once: the request built first carries them and the
/// context records the forwarding; the plan is taken, so a request
/// finished after it carries none.
#[tokio::test]
async fn finishing_attaches_the_media_references_once() {
    let (mut ctx, server) = context_with_media_refs().await;

    let mut request = ProtoGenerateRequest::Vllm(Box::default());
    finish_generate_request(&mut ctx, &mut request, false, "a_builder").expect("finished");
    assert!(
        request.has_vllm_media_refs(),
        "the first request carries the references"
    );
    assert!(ctx.state.media_refs_forwarded);
    assert!(ctx.state.multimodal_refs.is_none(), "the plan is taken");

    let mut again = ProtoGenerateRequest::Vllm(Box::default());
    finish_generate_request(&mut ctx, &mut again, false, "a_builder").expect("finished");
    assert!(!again.has_vllm_media_refs(), "attached once");
    server.abort();
}

/// A request whose preparation left no multimodal intermediate assembles
/// nothing: the helper answers before it needs the builder client or the
/// budget behind it, so it answers the same with the clients taken away.
/// (The media references the fixture carries are worker-side work for the
/// finishing, not an intermediate.)
#[tokio::test]
async fn without_an_intermediate_nothing_is_assembled() {
    let (mut ctx, server) = context_with_media_refs().await;
    ctx.state.clients = None;
    let assembled = assemble_prompt_media(&mut ctx, "a_builder")
        .await
        .expect("assembled");
    assert!(assembled.is_none());
    server.abort();
}

/// Before the clients are acquired, the finishing answers the one failure
/// of a missing client selection, a 500 under its code.
#[tokio::test]
async fn without_acquired_clients_the_finishing_reports_the_missing_stage() {
    let (mut ctx, server) = context_with_media_refs().await;
    ctx.state.clients = None;
    let mut request = ProtoGenerateRequest::Vllm(Box::default());
    let failure = match finish_generate_request(&mut ctx, &mut request, false, "a_builder") {
        Ok(()) => panic!("finished without clients"),
        Err(failure) => failure,
    };
    assert_eq!(failure.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = axum::body::to_bytes(failure.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
    assert_eq!(
        body["error"]["code"], "client_acquisition_not_completed",
        "{body}"
    );
    server.abort();
}
