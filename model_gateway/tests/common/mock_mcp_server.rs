// tests/common/mock_mcp_server.rs - Mock MCP server for testing
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService,
    },
    ErrorData as McpError, RoleServer, ServerHandler,
};
use tokio::{
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
    time::{timeout, Duration},
};

struct MockServerHarness {
    port: u16,
    shutdown_tx: Option<oneshot::Sender<()>>,
    server_handle: Option<JoinHandle<Result<(), std::io::Error>>>,
}

impl MockServerHarness {
    #[expect(
        clippy::disallowed_methods,
        reason = "test infrastructure uses a background server task"
    )]
    async fn start(app: axum::Router) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();

        let server_handle = tokio::spawn(async move {
            let _ = ready_tx.send(Ok(()));
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        match ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err.into()),
            Err(_) => return Err("mock server readiness channel dropped unexpectedly".into()),
        }

        Ok(Self {
            port,
            shutdown_tx: Some(shutdown_tx),
            server_handle: Some(server_handle),
        })
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(handle) = self.server_handle.take() {
            let mut handle = handle;
            match timeout(Duration::from_secs(2), &mut handle).await {
                Ok(join_result) => {
                    let _ = join_result;
                }
                Err(_) => {
                    handle.abort();
                    let _ = handle.await;
                }
            }
        }
    }
}

impl Drop for MockServerHarness {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

/// Mock MCP server that returns hardcoded responses for testing
pub struct MockMCPServer {
    harness: MockServerHarness,
    calls: Arc<AtomicUsize>,
}

/// Mock MCP server that always fails tool execution with a caller-provided marker.
pub struct MockFailingMCPServer {
    harness: MockServerHarness,
}

/// Mock MCP server that returns configurable web search response formats.
pub struct MockSearchResponseMCPServer {
    harness: MockServerHarness,
}

/// Simple test server with mock search tools
#[derive(Clone)]
pub struct MockSearchServer {
    calls: Arc<AtomicUsize>,
    tool_router: ToolRouter<MockSearchServer>,
}

impl Default for MockSearchServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Test server with a tool that always returns an MCP internal error.
#[derive(Clone)]
pub struct MockFailingSearchServer {
    error_marker: String,
    tool_router: ToolRouter<MockFailingSearchServer>,
}

#[derive(Clone, Copy)]
pub enum MockSearchResponseMode {
    Brave,
    OpenAi,
}

/// Test server that returns configurable web search payloads.
#[derive(Clone)]
pub struct MockSearchResponseServer {
    mode: MockSearchResponseMode,
    tool_router: ToolRouter<MockSearchResponseServer>,
}

impl MockFailingSearchServer {
    pub fn new(error_marker: impl Into<String>) -> Self {
        Self {
            error_marker: error_marker.into(),
            tool_router: Self::tool_router(),
        }
    }
}

impl MockSearchResponseServer {
    pub fn new(mode: MockSearchResponseMode) -> Self {
        Self {
            mode,
            tool_router: Self::tool_router(),
        }
    }
}

#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "proc macro generated"
)]
#[tool_router]
impl MockSearchServer {
    pub fn new() -> Self {
        Self {
            calls: Arc::default(),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Mock web search tool")]
    fn brave_web_search(
        &self,
        Parameters(params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("test");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Mock search results for: {query}"
        ))]))
    }

    #[tool(description = "Mock local search tool")]
    fn brave_local_search(
        &self,
        Parameters(_params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CallToolResult::success(vec![Content::text(
            "Mock local search results",
        )]))
    }
}

#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "proc macro generated"
)]
#[tool_router]
impl MockSearchResponseServer {
    #[tool(description = "Mock web search tool with configurable response shape")]
    fn brave_web_search(
        &self,
        Parameters(params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("test");

        match self.mode {
            MockSearchResponseMode::Brave => Ok(CallToolResult::structured(serde_json::json!({
                "results": [
                    {
                        "type": "url",
                        "url": "https://example.com/brave-result"
                    }
                ]
            }))),
            MockSearchResponseMode::OpenAi => {
                let embedded_payload = serde_json::json!({
                    "execution_id": "1234",
                    "brave_search_response": null,
                    "openai_response": {
                        "content": {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Example citation",
                                    "url": "https://example.com/openai-result",
                                    "start_index": 0,
                                    "end_index": 10
                                }
                            ],
                            "logprobs": [],
                            "text": format!("OpenAI search results for: {query}")
                        },
                        "sources": [
                            {
                                "type": "url",
                                "url": "https://example.com/openai-result"
                            }
                        ]
                    }
                });

                Ok(CallToolResult::success(vec![Content::text(
                    embedded_payload.to_string(),
                )]))
            }
        }
    }
}

#[tool_handler]
impl ServerHandler for MockSearchServer {
    fn get_info(&self) -> ServerInfo {
        // `ServerInfo`/`InitializeResult` is `#[non_exhaustive]` in rmcp 1.7;
        // build via the constructor instead of a struct literal.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions("Mock server for testing")
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

impl MockMCPServer {
    fn router(calls: Arc<AtomicUsize>) -> axum::Router {
        let service = StreamableHttpService::new(
            move || {
                Ok(MockSearchServer {
                    calls: calls.clone(),
                    ..MockSearchServer::new()
                })
            },
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }

    /// Start a mock MCP server on an available port
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let calls = Arc::default();
        Ok(Self {
            harness: MockServerHarness::start(Self::router(Arc::clone(&calls))).await?,
            calls,
        })
    }

    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn port(&self) -> u16 {
        self.harness.port()
    }

    /// Get the full URL for this mock server
    pub fn url(&self) -> String {
        self.harness.url()
    }

    /// Stop the mock server
    pub async fn stop(&mut self) {
        self.harness.stop().await;
    }
}

#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "proc macro generated"
)]
#[tool_router]
impl MockFailingSearchServer {
    #[tool(description = "Mock web search tool that always fails")]
    fn brave_web_search(
        &self,
        Parameters(_params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        Err(McpError::internal_error(
            format!("mock internal MCP failure: {}", self.error_marker),
            None,
        ))
    }
}

#[tool_handler]
impl ServerHandler for MockFailingSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions("Mock failing server for testing")
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

#[tool_handler]
impl ServerHandler for MockSearchResponseServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions("Mock search response server for testing")
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

impl MockFailingMCPServer {
    fn router(error_marker: String) -> axum::Router {
        let service = StreamableHttpService::new(
            move || Ok(MockFailingSearchServer::new(error_marker.clone())),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }

    pub async fn start(
        error_marker: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            harness: MockServerHarness::start(Self::router(error_marker.to_string())).await?,
        })
    }

    pub fn port(&self) -> u16 {
        self.harness.port()
    }

    pub fn url(&self) -> String {
        self.harness.url()
    }

    pub async fn stop(&mut self) {
        self.harness.stop().await;
    }
}

impl MockSearchResponseMCPServer {
    fn router(mode: MockSearchResponseMode) -> axum::Router {
        let service = StreamableHttpService::new(
            move || Ok(MockSearchResponseServer::new(mode)),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }

    pub async fn start(
        mode: MockSearchResponseMode,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            harness: MockServerHarness::start(Self::router(mode)).await?,
        })
    }

    pub fn port(&self) -> u16 {
        self.harness.port()
    }

    pub fn url(&self) -> String {
        self.harness.url()
    }

    pub async fn stop(&mut self) {
        self.harness.stop().await;
    }
}

#[cfg(test)]
mod tests {
    use super::{MockFailingMCPServer, MockMCPServer};

    #[tokio::test]
    async fn test_mock_server_startup() {
        let mut server = MockMCPServer::start().await.unwrap();
        assert!(server.port() > 0);
        assert!(server.url().contains(&server.port().to_string()));
        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_server_with_rmcp_client() {
        let mut server = MockMCPServer::start().await.unwrap();

        use rmcp::{transport::StreamableHttpClientTransport, ServiceExt};

        let transport = StreamableHttpClientTransport::from_uri(server.url().as_str());
        let client = ().serve(transport).await;

        assert!(client.is_ok(), "Should be able to connect to mock server");

        if let Ok(client) = client {
            let tools = client.peer().list_all_tools().await;
            assert!(tools.is_ok(), "Should be able to list tools");

            if let Ok(tools) = tools {
                assert_eq!(tools.len(), 2, "Should have 2 tools");
                assert!(tools.iter().any(|t| t.name == "brave_web_search"));
                assert!(tools.iter().any(|t| t.name == "brave_local_search"));
            }

            // Shutdown by dropping the client
            drop(client);
        }

        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_failing_server_startup() {
        use rmcp::{
            model::CallToolRequestParams, transport::StreamableHttpClientTransport, ServiceExt,
        };

        let mut server = MockFailingMCPServer::start("marker").await.unwrap();
        assert!(server.port() > 0);
        assert!(server.url().contains(&server.port().to_string()));

        let transport = StreamableHttpClientTransport::from_uri(server.url().as_str());
        let client = ().serve(transport).await.expect("connect failing mock server");

        let err = client
            .call_tool(
                CallToolRequestParams::new("brave_web_search").with_arguments(
                    serde_json::json!({ "query": "smoke" })
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            )
            .await
            .expect_err("failing mock tool call should error");
        assert!(err.to_string().contains("marker"));

        server.stop().await;
    }
}
