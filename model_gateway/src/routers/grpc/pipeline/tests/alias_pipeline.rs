//! A request sent under a model alias reaches preparation as the canonical model.

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    generate::GenerateRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::json;

use super::*;
use crate::{
    config::types::PolicyConfig,
    routers::grpc::regular::symphony::template::Templates,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};

const CANONICAL_MODEL: &str = "canonical-model";
const MODEL_ALIAS: &str = "model-alias";

fn register_pd_worker(registry: &WorkerRegistry, url: &str, worker_type: WorkerType) {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::Sglang)
        .model(ModelCard::new(CANONICAL_MODEL).with_alias(MODEL_ALIAS))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    registry.register(Arc::new(worker)).unwrap();
}

#[tokio::test]
async fn pd_generate_alias_is_canonical_before_preparation() {
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_pd_worker(
        &worker_registry,
        "grpc://prefill:30000",
        WorkerType::Prefill,
    );
    register_pd_worker(&worker_registry, "grpc://decode:30000", WorkerType::Decode);

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load("tokenizer-id", CANONICAL_MODEL, "test", || async move {
            Ok(tokenizer)
        })
        .await
        .unwrap();

    let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        policy_registry,
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::PrefillDecode, &deps).unwrap();
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        tokenizer_registry,
        worker_registry,
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    });
    let request: GenerateRequest = serde_json::from_value(json!({
        "model": MODEL_ALIAS,
        "text": "Hello"
    }))
    .unwrap();
    let mut ctx =
        RequestContext::for_generate(Arc::new(request), None, MODEL_ALIAS.to_string(), components);

    assert_eq!(ctx.input.model_id, CANONICAL_MODEL);
    assert_eq!(ctx.generate_request_arc().model, CANONICAL_MODEL);

    // Ingress up to worker selection: the canonical id must already be in
    // place when selection runs.
    pipeline.stages.preparation.execute(&mut ctx).await.unwrap();
    pipeline
        .stages
        .worker_selection
        .execute(&mut ctx)
        .await
        .unwrap();

    assert_eq!(ctx.input.model_id, CANONICAL_MODEL);
    assert_eq!(ctx.generate_request_arc().model, CANONICAL_MODEL);
    assert!(ctx.state.tokenizer.is_some());
    match ctx.state.workers.as_ref().unwrap() {
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => {
            assert_eq!(prefill.url(), "grpc://prefill:30000");
            assert_eq!(decode.url(), "grpc://decode:30000");
        }
        WorkerSelection::Single { .. } => panic!("expected PD worker selection"),
    }
}
