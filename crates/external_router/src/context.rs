//! Everything an external router may borrow from the gateway.

use std::{
    fmt,
    net::IpAddr,
    sync::{Arc, OnceLock},
    time::Duration,
};

use smg_data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use smg_mcp::McpOrchestrator;

use crate::{
    openai_bridge::FormatRegistry, realtime::RealtimeRegistry, worker::WorkerSource, RetryConfig,
};

/// The gateway, as an external router sees it. The gateway builds one per
/// router; nothing in it is specific to a provider.
#[derive(Clone)]
pub struct ExternalContext {
    pub client: reqwest::Client,
    pub request_timeout: Duration,
    pub retry: RetryConfig,
    pub workers: Arc<dyn WorkerSource>,
    /// The MCP orchestrator, once the gateway has one.
    pub mcp: Arc<OnceLock<Arc<McpOrchestrator>>>,
    pub mcp_formats: FormatRegistry,
    pub responses: Arc<dyn ResponseStorage>,
    pub conversations: Arc<dyn ConversationStorage>,
    pub conversation_items: Arc<dyn ConversationItemStorage>,
    pub realtime: Arc<RealtimeRegistry>,
    pub webrtc_bind_addr: Option<IpAddr>,
    pub webrtc_stun_server: Option<String>,
}

impl ExternalContext {
    /// The MCP orchestrator, or the reason a router cannot start without one.
    pub fn mcp_orchestrator(&self, router: &str) -> Result<Arc<McpOrchestrator>, String> {
        self.mcp
            .get()
            .cloned()
            .ok_or_else(|| format!("{router} router requires the MCP orchestrator"))
    }
}

impl fmt::Debug for ExternalContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalContext")
            .field("request_timeout", &self.request_timeout)
            .field("retry", &self.retry)
            .field("workers", &self.workers)
            .field("mcp", &self.mcp.get().is_some())
            .finish_non_exhaustive()
    }
}
