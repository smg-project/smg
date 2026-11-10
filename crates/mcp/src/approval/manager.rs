//! Approval manager for MCP tool execution.
//!
//! Provides a dual-mode approval system:
//! - **Interactive mode**: Returns approval requests to the caller for user confirmation
//! - **Policy-only mode**: Auto-decides using the PolicyEngine

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::info;

use super::{
    audit::{AuditLog, DecisionResult, DecisionSource},
    policy::{PolicyDecision, PolicyEngine},
};
use crate::{
    annotations::ToolAnnotations,
    error::{ApprovalError, McpResult},
    inventory::QualifiedToolName,
    tenant::TenantContext,
};

const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ApprovalKey {
    pub request_id: String,
    pub server_key: String,
    pub elicitation_id: String,
}

impl ApprovalKey {
    pub fn new(
        request_id: impl Into<String>,
        server_key: impl Into<String>,
        elicitation_id: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            server_key: server_key.into(),
            elicitation_id: elicitation_id.into(),
        }
    }
}

impl std::fmt::Display for ApprovalKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.request_id, self.server_key, self.elicitation_id
        )
    }
}

/// A pending approval awaiting user response.
#[derive(Debug)]
pub struct PendingApproval {
    pub key: ApprovalKey,
    pub tool_name: String,
    pub hints: ToolAnnotations,
    pub message: String,
    pub created_at: Instant,
    response_tx: oneshot::Sender<ApprovalDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approved,
    Denied { reason: String },
}

impl ApprovalDecision {
    pub fn approved() -> Self {
        Self::Approved
    }

    pub fn denied(reason: impl Into<String>) -> Self {
        Self::Denied {
            reason: reason.into(),
        }
    }

    pub fn is_approved(&self) -> bool {
        matches!(self, ApprovalDecision::Approved)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalMode {
    Interactive,
    #[default]
    PolicyOnly,
}

#[derive(Debug, Clone)]
pub struct ApprovalParams<'a> {
    pub request_id: &'a str,
    pub server_key: &'a str,
    pub elicitation_id: &'a str,
    pub tool_name: &'a str,
    pub hints: &'a ToolAnnotations,
    pub message: &'a str,
    pub tenant_ctx: &'a TenantContext,
}

/// Approval request sent to the client (matches OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpApprovalRequest {
    pub server_key: String,
    pub tool_name: String,
    pub message: String,
    pub elicitation_id: String,
}

/// Approval response from the client (matches OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpApprovalResponse {
    pub approve: bool,
    pub reason: Option<String>,
}

#[derive(Debug)]
pub enum ApprovalOutcome {
    Decided(PolicyDecision),
    Pending {
        key: ApprovalKey,
        rx: oneshot::Receiver<ApprovalDecision>,
        approval_request: McpApprovalRequest,
    },
}

/// Coordinates approval flow for interactive and policy-only modes.
pub struct ApprovalManager {
    policy_engine: Arc<PolicyEngine>,
    pending: DashMap<ApprovalKey, PendingApproval>,
    audit_log: Arc<AuditLog>,
    approval_timeout: Duration,
}

impl ApprovalManager {
    pub fn new(policy_engine: Arc<PolicyEngine>, audit_log: Arc<AuditLog>) -> Self {
        Self {
            policy_engine,
            pending: DashMap::new(),
            audit_log,
            approval_timeout: DEFAULT_APPROVAL_TIMEOUT,
        }
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.approval_timeout = timeout;
        self
    }

    /// Evicts expired pending approvals on each call.
    pub async fn handle_approval(
        &self,
        mode: ApprovalMode,
        params: ApprovalParams<'_>,
    ) -> McpResult<ApprovalOutcome> {
        // Lazy cleanup of expired approvals
        self.evict_expired();

        match mode {
            ApprovalMode::Interactive => self.request_interactive(&params).await,
            ApprovalMode::PolicyOnly => {
                let decision = self.policy_engine.evaluate(
                    params.server_key,
                    params.tool_name,
                    params.hints,
                    params.tenant_ctx,
                    params.request_id,
                );
                Ok(ApprovalOutcome::Decided(decision))
            }
        }
    }

    #[expect(
        clippy::unused_async,
        reason = "async kept for API consistency and future async backends"
    )]
    async fn request_interactive(&self, params: &ApprovalParams<'_>) -> McpResult<ApprovalOutcome> {
        let key = ApprovalKey::new(params.request_id, params.server_key, params.elicitation_id);

        if self.pending.contains_key(&key) {
            return Err(ApprovalError::AlreadyPending(key.to_string()).into());
        }

        let (tx, rx) = oneshot::channel();

        let pending = PendingApproval {
            key: key.clone(),
            tool_name: params.tool_name.to_string(),
            hints: params.hints.clone(),
            message: params.message.to_string(),
            created_at: Instant::now(),
            response_tx: tx,
        };

        self.audit_log.record_decision(
            &QualifiedToolName::new(params.server_key, params.tool_name),
            &params.tenant_ctx.tenant_id,
            params.request_id,
            DecisionResult::Pending,
            DecisionSource::UserInteractive,
        );

        self.pending.insert(key.clone(), pending);

        let approval_request = McpApprovalRequest {
            server_key: params.server_key.to_string(),
            tool_name: params.tool_name.to_string(),
            message: params.message.to_string(),
            elicitation_id: params.elicitation_id.to_string(),
        };

        Ok(ApprovalOutcome::Pending {
            key,
            rx,
            approval_request,
        })
    }

    #[expect(
        clippy::unused_async,
        reason = "public async API kept for future async resolution backends"
    )]
    pub async fn resolve(
        &self,
        request_id: &str,
        server_key: &str,
        elicitation_id: &str,
        approved: bool,
        reason: Option<String>,
        tenant_ctx: &TenantContext,
    ) -> McpResult<()> {
        let key = ApprovalKey::new(request_id, server_key, elicitation_id);

        let (_, pending) = self
            .pending
            .remove(&key)
            .ok_or_else(|| ApprovalError::NotFound(key.to_string()))?;

        let decision = if approved {
            ApprovalDecision::Approved
        } else {
            ApprovalDecision::Denied {
                reason: reason.unwrap_or_else(|| "User denied".to_string()),
            }
        };

        // Log the decision
        let result = if approved {
            DecisionResult::Approved
        } else {
            DecisionResult::Denied {
                reason: match &decision {
                    ApprovalDecision::Denied { reason } => reason.clone(),
                    ApprovalDecision::Approved => "User denied".to_string(),
                },
            }
        };

        self.audit_log.record_decision(
            &QualifiedToolName::new(server_key, &pending.tool_name),
            &tenant_ctx.tenant_id,
            request_id,
            result,
            DecisionSource::UserInteractive,
        );

        // Send response to waiting handler
        pending
            .response_tx
            .send(decision)
            .map_err(|_| ApprovalError::ChannelClosed)?;

        Ok(())
    }

    pub fn has_pending(&self, key: &ApprovalKey) -> bool {
        self.pending.contains_key(key)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn evict_expired(&self) {
        let now = Instant::now();
        let timeout = self.approval_timeout;
        self.pending
            .retain(|_, pending| now.duration_since(pending.created_at) < timeout);
    }

    pub fn policy_engine(&self) -> &Arc<PolicyEngine> {
        &self.policy_engine
    }

    pub fn audit_log(&self) -> &Arc<AuditLog> {
        &self.audit_log
    }
    /// Cancels all pending approvals, sending a denial to all waiting handlers.
    pub fn cancel_all_pending(&self) {
        info!(
            "Cancelling {} pending approvals due to shutdown",
            self.pending.len()
        );
        let keys: Vec<_> = self
            .pending
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            if let Some((_, pending)) = self.pending.remove(&key) {
                let _ = pending
                    .response_tx
                    .send(ApprovalDecision::denied("System shutdown"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> ApprovalManager {
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
        ApprovalManager::new(policy_engine, audit_log)
    }

    /// The approval parameters of the request `req-1` for `tool_name` on
    /// `server` (the elicitation `elicit-1`), with `message` for the user,
    /// under `hints` and `tenant`.
    fn approval_params<'a>(
        tool_name: &'a str,
        message: &'a str,
        hints: &'a ToolAnnotations,
        tenant: &'a TenantContext,
    ) -> ApprovalParams<'a> {
        ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name,
            hints,
            message,
            tenant_ctx: tenant,
        }
    }

    #[tokio::test]
    async fn test_policy_only_mode() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new().with_read_only(true);

        let params = approval_params("read_tool", "Allow read?", &hints, &tenant);

        let outcome = manager
            .handle_approval(ApprovalMode::PolicyOnly, params)
            .await
            .unwrap();

        match outcome {
            ApprovalOutcome::Decided(decision) => assert!(decision.is_allowed()),
            ApprovalOutcome::Pending { .. } => panic!("Expected decided outcome"),
        }
    }

    #[tokio::test]
    async fn test_interactive_mode_pending() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new().with_destructive(true);

        let params = approval_params("delete_tool", "Allow delete?", &hints, &tenant);

        let outcome = manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        match outcome {
            ApprovalOutcome::Pending {
                key,
                approval_request,
                ..
            } => {
                assert_eq!(key.request_id, "req-1");
                assert_eq!(approval_request.tool_name, "delete_tool");
                assert!(manager.has_pending(&key));
            }
            ApprovalOutcome::Decided(_) => panic!("Expected pending outcome"),
        }
    }

    #[tokio::test]
    async fn test_interactive_resolve() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let params = approval_params("tool", "Allow?", &hints, &tenant);

        let outcome = manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        let rx = match outcome {
            ApprovalOutcome::Pending { rx, .. } => rx,
            ApprovalOutcome::Decided(_) => panic!("Expected pending"),
        };

        manager
            .resolve("req-1", "server", "elicit-1", true, None, &tenant)
            .await
            .unwrap();

        let decision = rx.await.unwrap();
        assert!(decision.is_approved());
        assert_eq!(manager.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_resolve_not_found() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");

        let result = manager
            .resolve("nonexistent", "server", "elicit", true, None, &tenant)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_duplicate_pending() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let params = approval_params("tool", "Allow?", &hints, &tenant);

        manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        let params2 = approval_params("tool", "Allow?", &hints, &tenant);

        let result = manager
            .handle_approval(ApprovalMode::Interactive, params2)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_evict_expired() {
        let manager = test_manager().with_timeout(Duration::from_millis(1));
        manager.evict_expired();
        assert_eq!(manager.pending_count(), 0);
    }
}
