use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use smg_mcp::McpServerConfig;
use tracing::{debug, error, info, warn};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::data::{require_app_context, McpWorkflowData};
use crate::{
    app_context::AppContext, observability::metrics::Metrics, routers::common::openai_bridge,
};

/// MCP server connection configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpServerConfigRequest {
    /// Server name (unique identifier)
    pub name: String,
    /// Server configuration (transport, proxy, etc.)
    pub config: McpServerConfig,
}

impl McpServerConfigRequest {
    /// Check if this server is required for router startup
    pub fn is_required(&self) -> bool {
        self.config.required
    }
}

/// Step 1: Connect and register MCP server
///
/// This step establishes a connection to the MCP server using McpOrchestrator.
/// The orchestrator handles:
/// 1. Establishing the connection
/// 2. Loading tools, prompts, and resources
/// 3. Applying tool configurations (aliases, response formats)
/// 4. Registering the server as a static server
///
/// The connection is retried aggressively (100 attempts) with a long timeout (2 hours)
/// to handle slow-starting servers or network issues.
pub struct ConnectMcpServerStep;

#[async_trait]
impl StepExecutor<McpWorkflowData> for ConnectMcpServerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let app_context = require_app_context(context.data.app_context.as_ref())?;

        debug!("Connecting to MCP server: {}", config_request.name);

        // Get MCP orchestrator from app context
        let mcp_orchestrator =
            app_context
                .mcp_orchestrator
                .get()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("connect_mcp_server"),
                    message: "MCP orchestrator not initialized".to_string(),
                })?;

        openai_bridge::connect_static_server(
            mcp_orchestrator,
            &app_context.mcp_format_registry,
            &config_request.config,
        )
        .await
        .map_err(|e| WorkflowError::StepFailed {
            step_id: StepId::new("connect_mcp_server"),
            message: format!(
                "Failed to connect to MCP server {}: {}",
                config_request.name, e
            ),
        })?;

        // Update active MCP servers metric
        Metrics::set_mcp_servers_active(mcp_orchestrator.list_servers().len());

        info!(
            "Successfully connected and registered MCP server: {}",
            config_request.name
        );

        // Mark as connected for validation step
        context.data.connected = true;

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Connection failures are retryable
    }
}

/// Step 2: Validate registration based on required flag
///
/// This step checks if the server is marked as required. If the server is required
/// but wasn't successfully connected, this step fails the workflow.
/// For optional servers, this step always succeeds, allowing the workflow to complete
/// even if earlier steps failed.
pub struct ValidateRegistrationStep;

#[async_trait]
impl StepExecutor<McpWorkflowData> for ValidateRegistrationStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let server_connected = context.data.connected;

        if server_connected {
            info!(
                "MCP server '{}' registered successfully",
                config_request.name
            );

            // Mark as validated
            context.data.validated = true;

            return Ok(StepResult::Success);
        }

        if config_request.is_required() {
            error!(
                "Required MCP server '{}' failed to register",
                config_request.name
            );
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_registration"),
                message: format!(
                    "Required MCP server '{}' failed to register",
                    config_request.name
                ),
            })
        } else {
            warn!(
                "Optional MCP server '{}' failed to register, continuing workflow",
                config_request.name
            );
            Ok(StepResult::Success)
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Create MCP server registration workflow
///
/// This workflow adapts its failure behavior based on the `required` field in the server config:
/// - If `required == true`: Uses FailWorkflow - router startup fails if server cannot be reached
/// - If `required == false` (default): Uses ContinueNextStep - logs warning but continues
///
/// Workflow configuration:
/// - ConnectMcpServer: 100 retries, 2hr timeout (aggressive retry for slow servers)
///   - Handles connection, inventory discovery, and registration in one step
/// - ValidateRegistration: Final validation step
pub fn create_mcp_registration_workflow() -> WorkflowDefinition<McpWorkflowData> {
    WorkflowDefinition::new("mcp_registration", "MCP Server Registration")
        .add_step(
            StepDefinition::new(
                "connect_mcp_server",
                "Connect and Register MCP Server",
                Arc::new(ConnectMcpServerStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 100,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(Duration::from_secs(7200)) // 2 hours
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "validate_registration",
                "Validate MCP Registration",
                Arc::new(ValidateRegistrationStep),
            )
            .with_timeout(Duration::from_secs(1))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["connect_mcp_server"]),
        )
}

/// Helper to create initial workflow data for MCP registration
pub fn create_mcp_workflow_data(
    config: McpServerConfigRequest,
    app_context: Arc<AppContext>,
) -> McpWorkflowData {
    McpWorkflowData {
        config,
        validated: false,
        connected: false,
        app_context: Some(app_context),
    }
}
