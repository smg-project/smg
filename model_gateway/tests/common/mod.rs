// These modules are used by tests and benchmarks
#![allow(dead_code, clippy::allow_attributes, clippy::large_futures)]

pub mod mock_mcp_server;
pub mod mock_openai_server;
pub mod mock_worker;
pub mod streaming_helpers;
pub mod test_app;
pub mod test_certs;
pub mod test_config;
pub mod tls_mock_worker;

// Re-export commonly used test builders
use std::{
    fs,
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex, OnceLock},
};

use llm_tokenizer::registry::TokenizerRegistry;
use mock_worker::{MockWorker, MockWorkerConfig};
use openai_protocol::common::{Function, Tool};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::json;
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    middleware::TokenBucket,
    policies::PolicyRegistry,
    routers::{router_manager::RouterManager, RouterFactory, RouterTrait},
    worker::{
        BasicWorkerBuilder, ModelCard, RuntimeType, Worker, WorkerMonitor, WorkerRegistry,
        WorkerType,
    },
    workflow::Job,
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
#[allow(unused_imports)]
pub use test_config::{TestRouterConfig, TestWorkerConfig};
use tool_parser::ParserFactory as ToolParserFactory;

/// Test context for directly testing mock workers without full router setup.
pub struct WorkerTestContext {
    pub workers: Vec<MockWorker>,
    pub worker_urls: Vec<String>,
}

impl WorkerTestContext {
    #[expect(
        clippy::unwrap_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        Self {
            workers,
            worker_urls,
        }
    }

    pub fn first_worker_url(&self) -> Option<&str> {
        self.worker_urls.first().map(|s| s.as_str())
    }

    pub async fn make_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let client = reqwest::Client::new();
        let worker_url = self
            .first_worker_url()
            .ok_or_else(|| "No workers available".to_string())?;

        let response = client
            .post(format!("{worker_url}{endpoint}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!("Request failed with status: {}", response.status()));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| format!("Failed to parse response: {e}"))
    }

    pub async fn make_streaming_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<Vec<String>, String> {
        use futures_util::StreamExt;

        let client = reqwest::Client::new();
        let worker_url = self
            .first_worker_url()
            .ok_or_else(|| "No workers available".to_string())?;

        let response = client
            .post(format!("{worker_url}{endpoint}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!("Request failed with status: {}", response.status()));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("text/event-stream") {
            return Err("Response is not a stream".to_string());
        }

        let mut stream = response.bytes_stream();
        let mut events = Vec::new();

        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if let Some(stripped) = line.strip_prefix("data: ") {
                        events.push(stripped.to_string());
                    }
                }
            }
        }

        Ok(events)
    }

    pub async fn shutdown(mut self) {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        for worker in &mut self.workers {
            worker.stop().await;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

/// Test context for integration tests that go through the full axum app stack.
pub struct AppTestContext {
    pub workers: Vec<MockWorker>,
    pub router: Arc<dyn RouterTrait>,
    pub config: RouterConfig,
    pub app_context: Arc<AppContext>,
}

impl AppTestContext {
    pub fn new(
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Pin<Box<dyn Future<Output = Self> + Send>> {
        Box::pin(async move {
            let mut config = RouterConfig::builder()
                .regular_mode(vec![])
                .random_policy()
                .host("127.0.0.1")
                .port(3002)
                .max_payload_size(256 * 1024 * 1024)
                .request_timeout_secs(600)
                .worker_startup_timeout_secs(1)
                .worker_startup_check_interval_secs(1)
                .max_concurrent_requests(64)
                .queue_timeout_secs(60)
                .build_unchecked();
            // Test mock workers don't need health checks — start Ready immediately.
            // Tests that need to exercise Pending/Failed semantics should use
            // `new_with_config()` with an explicit config.
            config.health_check.disable_health_check = true;

            Self::new_with_config(config, worker_configs).await
        })
    }

    #[expect(
        clippy::unwrap_used,
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub fn new_with_config(
        mut config: RouterConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Pin<Box<dyn Future<Output = Self> + Send>> {
        Box::pin(async move {
            let mut workers = Vec::new();
            let mut worker_urls = Vec::new();

            for worker_config in worker_configs {
                let mut worker = MockWorker::new(worker_config);
                let url = worker.start().await.unwrap();
                worker_urls.push(url);
                workers.push(worker);
            }

            if !workers.is_empty() {
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }

            match &mut config.mode {
                RoutingMode::Regular {
                    worker_urls: ref mut urls,
                }
                | RoutingMode::OpenAI {
                    worker_urls: ref mut urls,
                } if urls.is_empty() => {
                    urls.clone_from(&worker_urls);
                }
                _ => {}
            }

            let app_context = create_test_context(config.clone()).await;

            if !worker_urls.is_empty() {
                let job_queue = app_context
                    .worker_job_queue
                    .get()
                    .expect("JobQueue should be initialized");
                let job = Job::InitializeWorkersFromConfig {
                    router_config: Box::new(config.clone()),
                };
                job_queue
                    .submit(job)
                    .await
                    .expect("Failed to submit worker initialization job");

                let expected_count = worker_urls.len();
                let start = tokio::time::Instant::now();
                let timeout_duration = tokio::time::Duration::from_secs(10);
                loop {
                    let healthy_workers = app_context
                        .worker_registry
                        .get_all()
                        .iter()
                        .filter(|w| w.is_healthy())
                        .count();

                    if healthy_workers >= expected_count {
                        break;
                    }

                    assert!(
                        start.elapsed() <= timeout_duration,
                        "Timeout waiting for {expected_count} workers to become healthy (only {healthy_workers} ready)"
                    );

                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }

            let inner_router = RouterFactory::create_router(&app_context).await.unwrap();
            let manager = RouterManager::new(
                app_context.worker_registry.clone(),
                app_context.client.clone(),
            );
            let router_id =
                RouterManager::determine_router_id(&config.mode, config.connection_mode);
            let manager = Arc::new(manager);
            manager.register_router(router_id, Arc::from(inner_router));
            let router: Arc<dyn RouterTrait> = manager;

            Self {
                workers,
                router,
                config,
                app_context,
            }
        })
    }

    pub fn create_app(&self) -> axum::Router {
        test_app::create_test_app_with_context(
            Arc::clone(&self.router),
            Arc::clone(&self.app_context),
        )
    }

    pub async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

/// Shared `AppContext` construction tail for the `create_test_context_*`
/// family below: registries, storage backends, worker monitor, job queue,
/// workflow engines, OpenAI-mode external worker registration, and MCP
/// orchestrator startup are identical across all of them. Only the
/// tokenizer registry, parser factories, rate-limit manager, and MCP config
/// vary by caller, so those are the only parameters.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_test_app_context(
    config: RouterConfig,
    tokenizer_registry: Arc<TokenizerRegistry>,
    reasoning_parser_factory: Option<ReasoningParserFactory>,
    tool_parser_factory: Option<ToolParserFactory>,
    rate_limit_manager: Option<Arc<smg::rate_limit::RateLimitManager>>,
    mcp_config: smg_mcp::McpConfig,
) -> Arc<AppContext> {
    use smg_mcp::McpOrchestrator;

    let client = reqwest::Client::new();

    // Initialize rate limiter
    let rate_limiter = match config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));

    // Initialize storage backends (Memory for tests).
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    // Initialize load monitor
    let worker_monitor = Some(Arc::new(WorkerMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.load_monitor_interval_secs,
        config.engine_metrics,
    )));

    // Create empty OnceLock for worker job queue, workflow engines, and mcp orchestrator
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());
    let mcp_orchestrator_lock = Arc::new(OnceLock::new());

    let app_context = Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .rate_limit_manager(rate_limit_manager)
            .tokenizer_registry(tokenizer_registry)
            .reasoning_parser_factory(reasoning_parser_factory)
            .tool_parser_factory(tool_parser_factory)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .worker_monitor(worker_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_orchestrator(mcp_orchestrator_lock)
            .build()
            .unwrap(),
    );

    // Mirror production wiring: start the WorkerMonitor event
    // loop so tests exercise the event-driven group reconciliation
    // path instead of an inert monitor.
    if let Some(monitor) = &app_context.worker_monitor {
        monitor.start_event_loop();
    }

    // Initialize JobQueue after AppContext is created
    let weak_context = Arc::downgrade(&app_context);
    let job_queue =
        smg::workflow::JobQueue::new(smg::workflow::JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    use smg::workflow::WorkflowEngines;
    let engines = WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

    // Register external workers for OpenAI mode
    if let RoutingMode::OpenAI { worker_urls, .. } = &config.mode {
        for url in worker_urls {
            // Create a worker that supports common test models
            let models = vec![
                ModelCard::new("mock-model"),
                ModelCard::new("gpt-4"),
                ModelCard::new("gpt-3.5-turbo"),
            ];
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Regular)
                    .runtime_type(RuntimeType::External)
                    .models(models)
                    .health_config(openai_protocol::worker::HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            );
            app_context.worker_registry.register(worker);
        }
    }

    let mcp_orchestrator = McpOrchestrator::new(mcp_config)
        .await
        .expect("Failed to create MCP orchestrator");
    app_context
        .mcp_orchestrator
        .set(Arc::new(mcp_orchestrator))
        .ok()
        .expect("McpOrchestrator should only be initialized once");

    app_context
}

/// An empty MCP config -- no servers, defaults everywhere -- for the
/// `create_test_context_*` variants that don't exercise MCP themselves.
fn empty_mcp_config() -> smg_mcp::McpConfig {
    smg_mcp::McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
        policy: Default::default(),
    }
}

/// Helper function to create AppContext for tests
pub fn create_test_context(
    config: RouterConfig,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    create_test_context_with_tokenizer_registry(config, Arc::new(TokenizerRegistry::new()))
}

/// Same as [`create_test_context`], but with a caller-supplied tokenizer
/// registry instead of an always-empty one, and real (not `None`) parser
/// factories -- `GrpcRouter::new` requires both to be `Some`. For tests that
/// build a real gRPC router (e.g. against a gRPC mock worker, which has no
/// tokenizer artifacts of its own to autoload from).
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn create_test_context_with_tokenizer_registry(
    config: RouterConfig,
    tokenizer_registry: Arc<TokenizerRegistry>,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    Box::pin(async move {
        // Build the tenant rate limiter from `config` the same way
        // `AppContext::from_config` does (that path's own
        // `maybe_rate_limit_manager` is private, so it's replicated here via
        // the public direct setter added for this purpose).
        let tenant_rate_limit_manager = smg::rate_limit::RateLimitManager::from_config(&config)
            .expect("tenant rate limit config should be valid");
        build_test_app_context(
            config,
            tokenizer_registry,
            Some(ReasoningParserFactory::new()),
            Some(ToolParserFactory::new()),
            tenant_rate_limit_manager,
            empty_mcp_config(),
        )
        .await
    })
}

/// Helper function to create AppContext for tests with parser factories initialized
pub fn create_test_context_with_parsers(
    config: RouterConfig,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    Box::pin(build_test_app_context(
        config,
        Arc::new(TokenizerRegistry::new()),
        Some(ReasoningParserFactory::new()),
        Some(ToolParserFactory::new()),
        None,
        empty_mcp_config(),
    ))
}

/// Helper function to create AppContext for tests with MCP config from file
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn create_test_context_with_mcp_config(
    config: RouterConfig,
    mcp_config_path: &str,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    let mcp_config_path = mcp_config_path.to_owned();
    Box::pin(async move {
        let mcp_config = smg_mcp::McpConfig::from_file(&mcp_config_path)
            .await
            .expect("Failed to load MCP config from file");
        build_test_app_context(
            config,
            Arc::new(TokenizerRegistry::new()),
            None,
            None,
            None,
            mcp_config,
        )
        .await
    })
}

// Tokenizer download configuration
const TINYLLAMA_TOKENIZER_URL: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
const CACHE_DIR: &str = ".tokenizer_cache";
const TINYLLAMA_TOKENIZER_FILENAME: &str = "tinyllama_tokenizer.json";

// Global mutex to prevent concurrent downloads
static DOWNLOAD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

/// Downloads the TinyLlama tokenizer from HuggingFace if not already cached.
/// Returns the path to the cached tokenizer file.
///
/// This function is thread-safe and will only download the tokenizer once
/// even if called from multiple threads concurrently.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn ensure_tokenizer_cached() -> PathBuf {
    // Get or initialize the mutex
    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));

    // Lock to ensure only one thread downloads at a time
    let _guard = mutex.lock().unwrap();

    let cache_dir = PathBuf::from(CACHE_DIR);
    let tokenizer_path = cache_dir.join(TINYLLAMA_TOKENIZER_FILENAME);

    // Create cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    // Download tokenizer if not already cached
    if !tokenizer_path.exists() {
        println!("Downloading TinyLlama tokenizer from HuggingFace...");

        // Use blocking reqwest client since we're in tests/benchmarks
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(TINYLLAMA_TOKENIZER_URL)
            .send()
            .expect("Failed to download tokenizer");

        assert!(
            response.status().is_success(),
            "Failed to download tokenizer: HTTP {}",
            response.status()
        );

        let content = response.bytes().expect("Failed to read tokenizer content");

        assert!(
            content.len() >= 100,
            "Downloaded content too small: {} bytes",
            content.len()
        );

        fs::write(&tokenizer_path, content).expect("Failed to write tokenizer to cache");
        println!(
            "Tokenizer downloaded and cached successfully ({} bytes)",
            tokenizer_path.metadata().unwrap().len()
        );
    }

    tokenizer_path
}

/// Common test prompts for consistency across tests
pub const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

/// Pre-computed hashes for verification
pub const EXPECTED_HASHES: [u64; 4] = [
    1209591529327510910,
    4181375434596349981,
    6245658446118930933,
    5097285695902185237,
];

/// Create a comprehensive set of test tools covering all parser test scenarios
pub fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "search".to_string(),
                description: Some("Search for information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "location": {"type": "string"},
                        "date": {"type": "string"},
                        "units": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: Some("Perform calculations".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "translate".to_string(),
                description: Some("Translate text".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "to": {"type": "string"},
                        "target_lang": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_current_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "update_settings".to_string(),
                description: Some("Update settings".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "preferences": {"type": "object"},
                        "notifications": {"type": "boolean"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "ping".to_string(),
                description: Some("Ping service".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "process".to_string(),
                description: Some("Process data".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "count": {"type": "number"},
                        "rate": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "data": {"type": "object"},
                        "text": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "web_search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "number"},
                        "search_type": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_tourist_attractions".to_string(),
                description: Some("Get tourist attractions".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "config".to_string(),
                description: Some("Configuration function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "debug": {"type": "boolean"},
                        "verbose": {"type": "boolean"},
                        "optional": {"type": "null"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test_func".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "bool_true": {"type": "boolean"},
                        "bool_false": {"type": "boolean"},
                        "none_val": {"type": "null"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "create".to_string(),
                description: Some("Create resource".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "add".to_string(),
                description: Some("Add operation".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calc".to_string(),
                description: Some("Calculate".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func1".to_string(),
                description: Some("Function 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func2".to_string(),
                description: Some("Function 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool1".to_string(),
                description: Some("Tool 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool2".to_string(),
                description: Some("Tool 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
    ]
}
