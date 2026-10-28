//! Test support of the regular stages: the shared components of a gateway
//! with `--parser symphony` and no parser names, and a builder client on a
//! mock engine.

use std::sync::Arc;

use llm_tokenizer::TokenizerRegistry;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg_grpc_client::{vllm_proto, VllmEngineClient};
use tokio::task::JoinHandle;
use tool_parser::ParserFactory as ToolParserFactory;

use crate::{
    config::ParserChoice,
    routers::grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::helpers::{IdStamp, SamplingBaseline, ToolConstraintStamp},
        context::{AttemptStamp, ExecutionPlan, SharedComponents},
        proto_wrapper::{ProtoGenerateRequest, ProtoRequest},
        regular::symphony::template::Templates,
        test_engine::mock_engine,
        utils::ParserResolver,
    },
    worker::WorkerRegistry,
};

/// The gateway with `--parser symphony`, no parser names and no worker.
pub(crate) fn components() -> Arc<SharedComponents> {
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(Arc::clone(&tokenizer_registry))),
        tokenizer_registry,
        worker_registry: Arc::new(WorkerRegistry::new()),
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Symphony,
    })
}

/// A builder client connected to a mock engine serving `model_id`, with the
/// engine's task, aborted by the test when it is done.
pub(crate) async fn mock_backend_client(model_id: &str) -> (BackendClient, JoinHandle<()>) {
    let (url, engine) = mock_engine(model_id).await;
    let client = VllmEngineClient::connect(&url)
        .await
        .expect("connect the mock engine");
    (BackendClient::Grpc(GrpcClient::Vllm(client)), engine)
}

/// A forced call's structural tag as preparation spells one. With no worker
/// selected the shaping leaves it as it is, so the engine's request carries
/// it unchanged and the attempt stamp carries it unshaped.
pub(crate) const FORCED_CALL_TAG: &str = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<tool_call>"],"tags":[]}}"#;

/// What the sequence the chat and Messages builders share makes of a fixed
/// request on the mock engine's wire: `temperature` 0.5 and `max_tokens` 7
/// asked, the ids `[1, 2, 3]` for the prompt, [`FORCED_CALL_TAG`] as the
/// constraint, no media, no selected worker, no PD. The plan carries the id
/// minted under `prefix`, the ids as the prompt with no text beside them, the
/// client's sampling knobs and the tag as the request carried it; the attempt
/// stamp carries the minted id, the mask of the knobs the client left unset,
/// the pre-default baseline and the unshaped constraint with no parser and no
/// first-token lead.
pub(crate) fn assert_fixed_request_plan(
    plan: &ExecutionPlan,
    stamp: &AttemptStamp,
    prefix: &'static str,
) {
    let ExecutionPlan::Single(ProtoRequest::Generate(request)) = plan else {
        panic!("a single generate plan")
    };
    assert_eq!(request.structural_tag(), Some(FORCED_CALL_TAG));
    let ProtoGenerateRequest::Vllm(request) = request else {
        panic!("the mock engine's wire")
    };
    let minted = request
        .request_id
        .strip_prefix(prefix)
        .expect("the id minted under the prefix");
    uuid::Uuid::parse_str(minted).expect("a uuid after the prefix");
    let Some(vllm_proto::generate_request::Input::Tokenized(tokenized)) = request.input.as_ref()
    else {
        panic!("the ids are the prompt")
    };
    assert_eq!(tokenized.input_ids, [1, 2, 3]);
    assert_eq!(
        tokenized.original_text, "",
        "no media: the text stays off the wire"
    );
    let params = request.sampling_params.as_ref().expect("sampling params");
    assert_eq!(
        (params.temperature, params.max_tokens),
        (Some(0.5), Some(7))
    );

    assert!(matches!(stamp.id, IdStamp::Minted { prefix: minted } if minted == prefix));
    assert_eq!(
        format!("{:?}", stamp.sampling_mask),
        "Some(SamplingDefaultsMask { temperature: false, top_p: true, top_k: true, min_p: true, repetition_penalty: true })"
    );
    assert!(matches!(
        stamp.sampling_baseline,
        Some(SamplingBaseline::Vllm(_))
    ));
    assert!(!stamp.inject_pd_metadata);
    let Some(ToolConstraintStamp::Registry {
        constraint,
        parser,
        first_token,
    }) = &stamp.tool_constraint
    else {
        panic!("the unshaped constraint on the stamp")
    };
    assert_eq!(
        *constraint,
        ("structural_tag".to_string(), FORCED_CALL_TAG.to_string())
    );
    assert_eq!(*parser, None);
    assert!(!first_token);
}
