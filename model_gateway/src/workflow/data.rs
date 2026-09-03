//! Typed workflow data structures
//!
//! This module defines the typed data structures for all workflows, enabling
//! compile-time type safety and state persistence. Each workflow has its own
//! strongly-typed data structure, and steps are typed to their specific workflow.
//!
//! # Shared Step Trait
//!
//! For steps that are shared between local and external worker workflows,
//! we use the `WorkerRegistrationData` trait. This trait provides a common
//! interface for accessing worker data while maintaining full type safety.

use std::{collections::HashMap, sync::Arc};

/// Re-export the protocol types for convenience
pub use openai_protocol::worker::{WorkerSpec, WorkerUpdateRequest as ProtocolUpdateRequest};
use openai_protocol::{
    model_card::ModelCard, worker::WorkerUpdateRequest as ProtocolWorkerUpdateRequest,
};
use serde::{Deserialize, Serialize};
use wfaas::{StepId, WorkflowData, WorkflowError, WorkflowResult};

// ============================================================================
// Worker kind classification
// ============================================================================

/// Classification of worker type: local inference backend or external cloud API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerKind {
    /// Self-hosted inference backend (sglang, vllm, trtllm).
    Local,
    /// Cloud API endpoint (OpenAI, Anthropic, etc.).
    External,
}

use super::{
    mcp_registration::McpServerConfigRequest, steps::local::WorkerRemovalRequest,
    tokenizer_registration::TokenizerConfigRequest,
    wasm_module_registration::WasmModuleConfigRequest,
    wasm_module_removal::WasmModuleRemovalRequest,
};
use crate::{app_context::AppContext, worker::Worker};

// ============================================================================
// Shared trait for worker registration workflows
// ============================================================================

/// Trait for workflow data that supports worker registration operations.
///
/// Implemented by `WorkerWorkflowData`, allowing shared steps
/// (register, activate, update policies) to work generically.
pub trait WorkerRegistrationData: WorkflowData {
    /// Get the application context (transient, not serialized).
    fn get_app_context(&self) -> Option<&Arc<AppContext>>;

    /// Get the actual worker objects (transient, not serialized).
    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>>;

    /// Get the labels for policy registration.
    fn get_labels(&self) -> Option<&HashMap<String, String>>;

    /// Select the semantics of the final registry write.
    ///
    /// Defaults to upsert, which is what every caller did before this method
    /// existed.
    fn registration_mode(&self) -> WorkerRegistrationMode {
        WorkerRegistrationMode::Upsert
    }
}

/// Semantics of the final worker registry write.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRegistrationMode {
    /// `POST /workers`: fail when the URL is already registered.
    CreateOnly,
    /// Startup and service discovery: register the URL, replacing whatever
    /// holds it. Both reconcile declared state against the registry.
    #[default]
    Upsert,
    /// `PUT /workers/{id}`: replace the worker only while `worker_id` still
    /// holds the URL at `expected_revision`.
    ///
    /// `PUT` answers 202 and registers later, so other writes for the same URL
    /// can land in between. Binding the write to what the request validated
    /// makes the late write fail instead of resurrecting a worker a `DELETE`
    /// removed, overwriting one a `POST` created, or restoring the older
    /// specification over a second `PUT` that finished first.
    ReplaceById {
        worker_id: String,
        expected_revision: u64,
    },
}

/// Wrapper for worker list that can be serialized
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerList {
    /// Worker URLs (we can't serialize Arc<dyn Worker>, so we store URLs)
    pub worker_urls: Vec<String>,
}

impl WorkerList {
    pub fn new() -> Self {
        Self {
            worker_urls: Vec::new(),
        }
    }

    pub fn from_workers(workers: &[Arc<dyn Worker>]) -> Self {
        Self {
            worker_urls: workers.iter().map(|w| w.url().to_string()).collect(),
        }
    }
}

// ============================================================================
// Unified worker registration workflow data
// ============================================================================

/// Unified data for worker registration workflows (both local and external).
///
/// A `ClassifyWorkerTypeStep` runs first to populate `worker_kind`, then
/// branch-specific steps check `worker_kind` and return `StepResult::Skip`
/// when they don't apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerWorkflowData {
    pub config: WorkerSpec,
    /// Defaults to upsert so persisted workflows from older versions preserve
    /// their historical registration behavior when deserialized.
    #[serde(default)]
    pub registration_mode: WorkerRegistrationMode,
    /// Determined by ClassifyWorkerTypeStep (Local or External).
    pub worker_kind: Option<WorkerKind>,
    // -- Local-only fields --
    pub connection_mode: Option<crate::worker::ConnectionMode>,
    /// HTTP workers: whether the router speaks HTTP/2 prior knowledge,
    /// resolved by DetectConnectionModeStep.
    #[serde(default)]
    pub http2: Option<bool>,
    /// The worker-directed client for `http2` (transient; see
    /// [`Self::http_client`]).
    #[serde(skip, default)]
    pub http_client_handle: Option<Arc<reqwest::Client>>,
    pub detected_runtime_type: Option<String>,
    pub discovered_labels: HashMap<String, String>,
    pub dp_info: Option<super::steps::local::DpInfo>,
    // -- External-only fields --
    pub model_cards: Vec<ModelCard>,
    // -- Shared fields --
    pub workers: Option<WorkerList>,
    pub final_labels: HashMap<String, String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Actual worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for WorkerWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_registration"
    }
}

impl WorkerWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }

    /// The worker-directed client for the resolved HTTP version, re-fetched
    /// from the cache when a deserialize dropped the transient handle.
    pub fn http_client(&self, step_id: &str) -> WorkflowResult<Arc<reqwest::Client>> {
        if let Some(client) = &self.http_client_handle {
            return Ok(Arc::clone(client));
        }
        let app_context = self
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        app_context
            .worker_client_cache
            .get(&self.config.http_pool, self.http2.unwrap_or(false))
            .map_err(|message| WorkflowError::StepFailed {
                step_id: StepId::new(step_id),
                message,
            })
    }
}

impl WorkerRegistrationData for WorkerWorkflowData {
    fn get_app_context(&self) -> Option<&Arc<AppContext>> {
        self.app_context.as_ref()
    }

    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>> {
        self.actual_workers.as_ref()
    }

    fn get_labels(&self) -> Option<&HashMap<String, String>> {
        Some(&self.final_labels)
    }

    fn registration_mode(&self) -> WorkerRegistrationMode {
        self.registration_mode.clone()
    }
}

// ============================================================================
// Workflow-specific data types
// ============================================================================

/// Data for tokenizer registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerWorkflowData {
    pub config: TokenizerConfigRequest,
    pub vocab_size: Option<usize>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for TokenizerWorkflowData {
    fn workflow_type() -> &'static str {
        "tokenizer_registration"
    }
}

impl TokenizerWorkflowData {
    /// Validate that all transient fields are properly initialized.
    ///
    /// Call this after deserializing workflow state to ensure runtime fields
    /// have been repopulated.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for worker removal workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRemovalWorkflowData {
    pub config: WorkerRemovalRequest,
    pub workers_to_remove: Option<WorkerList>,
    /// URLs of workers being removed
    pub worker_urls: Vec<String>,
    /// Model IDs affected by the removal
    pub affected_models: std::collections::HashSet<String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Actual worker objects to remove (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers_to_remove: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for WorkerRemovalWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_removal"
    }
}

impl WorkerRemovalWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for worker update workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdateWorkflowData {
    pub config: ProtocolWorkerUpdateRequest,
    /// URL of worker(s) to update
    pub worker_url: String,
    /// Whether to update all DP-aware workers with matching prefix
    pub dp_aware: bool,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Workers to update (transient, not serialized)
    #[serde(skip, default)]
    pub workers_to_update: Option<Vec<Arc<dyn Worker>>>,
    /// Updated worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub updated_workers: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for WorkerUpdateWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_update"
    }
}

impl WorkerUpdateWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for MCP server registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpWorkflowData {
    pub config: McpServerConfigRequest,
    pub validated: bool,
    /// Whether the server was successfully connected
    #[serde(default)]
    pub connected: bool,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for McpWorkflowData {
    fn workflow_type() -> &'static str {
        "mcp_registration"
    }
}

impl McpWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for WASM module registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRegistrationWorkflowData {
    pub config: WasmModuleConfigRequest,
    #[serde(skip, default)]
    pub wasm_bytes: Option<Arc<Vec<u8>>>,
    /// SHA256 hash of the module file (32 bytes)
    pub sha256_hash: Option<[u8; 32]>,
    /// File size in bytes
    pub file_size_bytes: Option<u64>,
    /// UUID assigned to the registered module
    pub module_uuid: Option<uuid::Uuid>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for WasmRegistrationWorkflowData {
    fn workflow_type() -> &'static str {
        "wasm_module_registration"
    }
}

impl WasmRegistrationWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for WASM module removal workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRemovalWorkflowData {
    pub config: WasmModuleRemovalRequest,
    pub module_id: Option<String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for WasmRemovalWorkflowData {
    fn workflow_type() -> &'static str {
        "wasm_module_removal"
    }
}

impl WasmRemovalWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}
