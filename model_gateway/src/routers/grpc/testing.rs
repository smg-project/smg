//! Test support of the gRPC router: the `AppContext` a test constructs its
//! router over, with the components `GrpcRouter::new` needs (the parser
//! factories always; an initialized MCP orchestrator, which the Responses
//! contexts of the regular and the disaggregated modes read), and the
//! labelled worker selection the stages' tests read a decode worker's
//! labels from, and the forced call a thinking prompt takes under a parser,
//! which the restamp and the parser-utility tests shape a constraint from.

use std::{
    sync::{Arc, Mutex, OnceLock},
    thread::{self, ThreadId},
};

use llm_tokenizer::{
    registry::TokenizerRegistry,
    traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
    MockTokenizer,
};
use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::{json, Value};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{McpConfig, McpOrchestrator};
use tool_parser::{ParserFactory as ToolParserFactory, ToolConstraint};

use crate::{
    app_context::AppContext,
    config::{ParserChoice, PolicyConfig, RouterConfig, RouterConfigBuilder, RoutingMode},
    policies::PolicyRegistry,
    routers::grpc::context::WorkerSelection,
    worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
};

/// Minimal `AppContext` for constructing a gRPC router in `mode`. Regular and
/// PD both serve /v1/responses, so the MCP orchestrator must be initialized.
pub(crate) async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
    grpc_ctx_with_parsers(mode, ParserChoice::default(), None).await
}

/// [`grpc_ctx`] with the gateway's `--parser` choice and its
/// `--tool-call-parser` name, which under Symphony pins the table that
/// answers to it for every model the registry does not know.
pub(crate) async fn grpc_ctx_with_parsers(
    mode: RoutingMode,
    parser: ParserChoice,
    tool_parser: Option<&str>,
) -> Arc<AppContext> {
    let mcp_orchestrator = mcp_orchestrator().await;
    let mut builder = grpc_config(mode).parser(parser);
    if let Some(name) = tool_parser {
        builder = builder.tool_call_parser(name);
    }
    grpc_ctx_from(&builder.build_unchecked(), mcp_orchestrator)
}

/// The gRPC router configuration the tests start from: `mode` over the gRPC
/// connection mode, the random policy, the HTTP request bound at 60 s and the
/// gRPC one at its default.
pub(crate) fn grpc_config(mode: RoutingMode) -> RouterConfigBuilder {
    RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(3001)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(10)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
}

/// The initialized MCP orchestrator every test context carries: PD serves
/// /v1/responses, which needs it. Awaited before the configuration is built,
/// so no context future holds a `RouterConfig` across the await.
pub(crate) async fn mcp_orchestrator() -> Arc<OnceLock<Arc<McpOrchestrator>>> {
    let mcp_orchestrator = Arc::new(OnceLock::new());
    mcp_orchestrator
        .set(Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("mcp orchestrator"),
        ))
        .ok();
    mcp_orchestrator
}

/// Minimal `AppContext` over `config` for constructing a gRPC router.
pub(crate) fn grpc_ctx_from(
    config: &RouterConfig,
    mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
) -> Arc<AppContext> {
    Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(crate::server::http_client())
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
            .tool_parser_factory(Some(ToolParserFactory::new()))
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(mcp_orchestrator)
            .build()
            .expect("app context"),
    )
}

/// One regular gRPC worker carrying `labels`, as the selection a stage or a
/// parser utility reads the decode worker's labels from.
pub(crate) fn worker(labels: &[(&str, &str)]) -> WorkerSelection {
    let mut builder =
        BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    WorkerSelection::Single {
        worker: Arc::new(builder.build()),
    }
}

/// The forced call a thinking prompt takes under `parser`: the constraint's
/// kind and the structural tag the registry builds for one `get_weather` tool
/// under `tool_choice: required`, with the reasoning prefix armed.
pub(crate) fn forced_call_on_a_thinking_prompt(
    factory: &ToolParserFactory,
    parser: &str,
) -> (String, String) {
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
    }]))
    .unwrap();
    let required = ToolChoice::Value(ToolChoiceValue::Required);
    match factory
        .registry()
        .generate_tool_constraint(Some(parser), &tools, &required, true)
        .unwrap()
    {
        Some(ToolConstraint::StructuralTag(tag)) => ("structural_tag".to_string(), tag),
        other => panic!("expected a structural tag, got {other:?}"),
    }
}

/// A chat tool declaration named `name`, with an empty object schema: what
/// the tool-reading tests put in a request's or a message's `tools`.
pub(crate) fn tool(name: &str) -> Value {
    json!({
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": {}}}
    })
}

/// The threads a [`ThreadRecordingTokenizer`]'s encodes and decodes ran on,
/// in order.
pub(crate) type ThreadRecord = Arc<Mutex<Vec<ThreadId>>>;

/// The mock tokenizer behind a record of the thread each of its encodes and
/// decodes ran on: the probe of the offload policy, under which an input
/// below the bound runs on the calling thread and one at or above it on the
/// blocking pool.
pub(crate) struct ThreadRecordingTokenizer {
    inner: MockTokenizer,
    ran_on: ThreadRecord,
}

/// A [`ThreadRecordingTokenizer`] over the plain mock, with its record.
pub(crate) fn thread_recording_tokenizer() -> (Arc<dyn Tokenizer>, ThreadRecord) {
    let ran_on: ThreadRecord = Arc::new(Mutex::new(Vec::new()));
    let tokenizer = Arc::new(ThreadRecordingTokenizer {
        inner: MockTokenizer::new(),
        ran_on: ran_on.clone(),
    });
    (tokenizer, ran_on)
}

impl ThreadRecordingTokenizer {
    fn record(&self) {
        self.ran_on.lock().unwrap().push(thread::current().id());
    }
}

impl Encoder for ThreadRecordingTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.record();
        self.inner.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.record();
        self.inner.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for ThreadRecordingTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.record();
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for ThreadRecordingTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
