//! SMG client handler for MCP server notifications and elicitation.
//!
//! Implements RMCP's `ClientHandler` trait to handle:
//! - Elicitation requests (approval flow)
//! - Tool/resource/prompt list change notifications
//! - Progress and logging notifications

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;
use rmcp::{
    model::{
        CancelledNotificationParam, ClientInfo, CreateElicitationRequestParams,
        CreateElicitationResult, ElicitationAction, LoggingLevel, LoggingMessageNotificationParam,
        ProgressNotificationParam, ResourceUpdatedNotificationParam,
    },
    service::{NotificationContext, RequestContext},
    ClientHandler, RoleClient,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::{
    approval::{ApprovalManager, ApprovalMode, ApprovalOutcome, ApprovalParams},
    tenant::TenantContext,
};

/// Request to refresh server inventory.
#[derive(Debug, Clone)]
pub struct RefreshRequest {
    pub server_key: String,
}

/// Per-request context set before tool execution, cleared after.
#[derive(Debug, Clone)]
pub struct HandlerRequestContext {
    pub request_id: String,
    pub approval_mode: ApprovalMode,
    pub tenant_ctx: TenantContext,
    pub forwarded_headers: HashMap<String, String>,
}

impl HandlerRequestContext {
    pub fn new(
        request_id: impl Into<String>,
        approval_mode: ApprovalMode,
        tenant_ctx: TenantContext,
        forwarded_headers: HashMap<String, String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            approval_mode,
            tenant_ctx,
            forwarded_headers,
        }
    }
}

#[derive(Clone)]
pub struct SmgClientHandler {
    server_key: Arc<str>,
    approval_manager: Arc<ApprovalManager>,
    client_info: ClientInfo,
    request_ctx: Arc<RwLock<Option<HandlerRequestContext>>>,
    refresh_tx: Option<mpsc::Sender<RefreshRequest>>,
}

impl SmgClientHandler {
    pub fn new(server_key: impl AsRef<str>, approval_manager: Arc<ApprovalManager>) -> Self {
        let mut client_info = ClientInfo::default();
        client_info.client_info.name = "smg".to_string();
        client_info.client_info.version = env!("CARGO_PKG_VERSION").to_string();

        Self {
            server_key: Arc::from(server_key.as_ref()),
            approval_manager,
            client_info,
            request_ctx: Arc::new(RwLock::new(None)),
            refresh_tx: None,
        }
    }

    #[must_use]
    pub fn with_refresh_channel(mut self, tx: mpsc::Sender<RefreshRequest>) -> Self {
        self.refresh_tx = Some(tx);
        self
    }

    #[must_use]
    pub fn with_client_info(mut self, info: ClientInfo) -> Self {
        self.client_info = info;
        self
    }

    pub fn set_request_context(&self, ctx: HandlerRequestContext) {
        *self.request_ctx.write() = Some(ctx);
    }

    pub fn clear_request_context(&self) {
        *self.request_ctx.write() = None;
    }

    pub fn request_context(&self) -> Option<HandlerRequestContext> {
        self.request_ctx.read().clone()
    }

    pub fn server_key(&self) -> &str {
        &self.server_key
    }

    fn send_refresh(&self) {
        if let Some(tx) = &self.refresh_tx {
            let _ = tx
                .try_send(RefreshRequest {
                    server_key: self.server_key.to_string(),
                })
                .map_err(|e| {
                    warn!(
                        server_key = %self.server_key,
                        error = %e,
                        "Failed to send refresh request"
                    );
                });
        }
    }
}

impl ClientHandler for SmgClientHandler {
    async fn create_elicitation(
        &self,
        request: CreateElicitationRequestParams,
        context: RequestContext<RoleClient>,
    ) -> Result<CreateElicitationResult, rmcp::ErrorData> {
        use crate::annotations::ToolAnnotations;

        let elicitation_id = match &context.id {
            rmcp::model::RequestId::String(s) => s.to_string(),
            rmcp::model::RequestId::Number(n) => n.to_string(),
        };

        // Get request context
        let req_ctx = self.request_ctx.read().clone().ok_or_else(|| {
            rmcp::ErrorData::internal_error("No request context set for elicitation", None)
        })?;

        // rmcp 1.7 split elicitation params into form/url variants; both carry a
        // user-facing message, which we use as the tool identifier (elicitation
        // doesn't have a tool name directly).
        let message = match &request {
            CreateElicitationRequestParams::FormElicitationParams { message, .. }
            | CreateElicitationRequestParams::UrlElicitationParams { message, .. } => message,
        };

        // Default annotations (conservative - not read-only, potentially destructive)
        let hints = ToolAnnotations::default();

        let params = ApprovalParams {
            request_id: &req_ctx.request_id,
            server_key: &self.server_key,
            elicitation_id: &elicitation_id,
            tool_name: "elicitation",
            hints: &hints,
            message,
            tenant_ctx: &req_ctx.tenant_ctx,
        };

        let outcome = self
            .approval_manager
            .handle_approval(req_ctx.approval_mode, params)
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let result_for = |allowed: bool| {
            CreateElicitationResult::new(if allowed {
                ElicitationAction::Accept
            } else {
                ElicitationAction::Decline
            })
        };

        match outcome {
            ApprovalOutcome::Decided(decision) => Ok(result_for(decision.is_allowed())),
            ApprovalOutcome::Pending { rx, .. } => {
                // Wait for user response
                match rx.await {
                    Ok(decision) => Ok(result_for(decision.is_approved())),
                    Err(_) => Err(rmcp::ErrorData::internal_error(
                        "Approval channel closed",
                        None,
                    )),
                }
            }
        }
    }

    async fn on_cancelled(
        &self,
        params: CancelledNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            server_key = %self.server_key,
            request_id = %params.request_id,
            reason = ?params.reason,
            "MCP server cancelled request"
        );
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        debug!(
            server_key = %self.server_key,
            token = ?params.progress_token,
            progress = %params.progress,
            total = ?params.total,
            message = ?params.message,
            "MCP server progress"
        );
    }

    async fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            server_key = %self.server_key,
            uri = %params.uri,
            "MCP server resource updated"
        );
    }

    async fn on_resource_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server resource list changed");
        self.send_refresh();
    }

    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server tool list changed");
        self.send_refresh();
    }

    async fn on_prompt_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server prompt list changed");
        self.send_refresh();
    }

    fn get_info(&self) -> ClientInfo {
        self.client_info.clone()
    }

    async fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        let logger = params.logger.as_deref().unwrap_or("mcp");

        match params.level {
            LoggingLevel::Emergency | LoggingLevel::Alert | LoggingLevel::Critical => {
                error!(
                    server_key = %self.server_key,
                    logger = %logger,
                    level = ?params.level,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Error => {
                error!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Warning => {
                warn!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Notice | LoggingLevel::Info => {
                info!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Debug => {
                debug!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{audit::AuditLog, policy::PolicyEngine};

    fn test_handler() -> SmgClientHandler {
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        SmgClientHandler::new("test-server", approval_manager)
    }

    #[test]
    fn test_handler_creation() {
        let handler = test_handler();
        assert_eq!(handler.server_key(), "test-server");
        assert!(handler.request_context().is_none());
    }

    #[test]
    fn test_request_context() {
        let handler = test_handler();

        let ctx = HandlerRequestContext::new(
            "req-1",
            ApprovalMode::PolicyOnly,
            TenantContext::new("tenant-1"),
            HashMap::new(),
        );

        handler.set_request_context(ctx.clone());
        assert!(handler.request_context().is_some());

        let retrieved = handler.request_context().unwrap();
        assert_eq!(retrieved.request_id, "req-1");

        handler.clear_request_context();
        assert!(handler.request_context().is_none());
    }

    #[test]
    fn test_client_info() {
        let handler = test_handler();
        let info = handler.get_info();
        assert_eq!(info.client_info.name, "smg");
    }

    #[test]
    fn test_with_refresh_channel() {
        let (tx, _rx) = mpsc::channel(10);
        let handler = test_handler().with_refresh_channel(tx);
        assert!(handler.refresh_tx.is_some());
    }
}
