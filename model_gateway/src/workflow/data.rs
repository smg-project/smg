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

use openai_protocol::worker::WorkerUpdateRequest as ProtocolWorkerUpdateRequest;
/// Re-export the protocol types for convenience
pub use openai_protocol::worker::{WorkerSpec, WorkerUpdateRequest as ProtocolUpdateRequest};
use serde::{Deserialize, Serialize};
pub use smg_worker::jobs::WorkerRegistrationMode;
use smg_worker::{
    identity::basic::Worker, jobs::WorkerJobAsker, registry::WorkerId, workflow as worker_workflow,
};
use wfaas::{WorkflowData, WorkflowError, WorkflowResult};

use super::{
    mcp_registration::McpServerConfigRequest, tokenizer_registration::TokenizerConfigRequest,
    wasm_module_registration::WasmModuleConfigRequest,
    wasm_module_removal::WasmModuleRemovalRequest,
};
use crate::app_context::AppContext;

/// The application context a workflow's data was given, or the workflow's
/// error for a run that was started without one.
pub(crate) fn require_app_context(
    app_context: Option<&Arc<AppContext>>,
) -> WorkflowResult<&Arc<AppContext>> {
    app_context.ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))
}

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

    /// The Symphony table whose failure classes exist at zero once the
    /// workers are registered, when the workflow resolved one.
    fn symphony_table(&self) -> Option<&'static str> {
        None
    }

    /// Select the semantics of the final registry write.
    ///
    /// Defaults to upsert, which is what every caller did before this method
    /// existed.
    fn registration_mode(&self) -> WorkerRegistrationMode {
        WorkerRegistrationMode::Upsert
    }
}

// ============================================================================
// Worker registration workflow data
// ============================================================================

/// What the gateway's own steps keep on a registration between them.
#[derive(Debug, Clone, Default)]
pub struct RegistrationNotes {
    /// Under Symphony, the table whose failure classes the registration step
    /// publishes at zero once the worker is in the registry (the create step
    /// resolves it from the model card and the names; none for an HTTP
    /// worker, the legacy parsers or a model no table reads).
    pub symphony_table: Option<&'static str>,
}

/// Unified data for worker registration workflows (both local and external),
/// with the gateway as their host.
///
/// A `ClassifyWorkerTypeStep` runs first to populate `worker_kind`, then
/// branch-specific steps check `worker_kind` and return `StepResult::Skip`
/// when they don't apply.
pub type WorkerWorkflowData = worker_workflow::WorkerRegistrationWorkflowData<AppContext>;

impl WorkerRegistrationData for WorkerWorkflowData {
    fn get_app_context(&self) -> Option<&Arc<AppContext>> {
        self.host.as_ref()
    }

    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>> {
        self.actual_workers.as_ref()
    }

    fn get_labels(&self) -> Option<&HashMap<String, String>> {
        Some(&self.final_labels)
    }

    fn symphony_table(&self) -> Option<&'static str> {
        self.host_notes.symphony_table
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

/// Data for worker removal workflow: the worker crate's, with this gateway's
/// context as the host its steps read.
pub type WorkerRemovalWorkflowData = worker_workflow::WorkerRemovalWorkflowData<AppContext>;

/// Data for worker update workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdateWorkflowData {
    pub config: ProtocolWorkerUpdateRequest,
    /// Who asked for the update or label refresh. A host's own step reads it.
    pub asker: WorkerJobAsker,
    /// URL of worker(s) to update
    pub worker_url: String,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Workers to update (transient, not serialized)
    #[serde(skip, default)]
    pub workers_to_update: Option<Vec<Arc<dyn Worker>>>,
    /// Updated worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub updated_workers: Option<Vec<Arc<dyn Worker>>>,
    /// Set by a label refresh (the worker's engine came back into service or
    /// finished loading): the labels are re-read from the engine and replace
    /// the discovered set instead of being merged from `config`.
    #[serde(default)]
    pub refresh_discovered_labels: bool,
    /// The labels the engine reports now, read by the re-discovery step for
    /// the update step (transient).
    #[serde(skip, default)]
    pub fresh_engine_labels: Option<HashMap<String, String>>,
    /// Each worker's registry id and revision as the find step found it, in
    /// `workers_to_update` order, captured once so the write-back targets
    /// exactly that registration whatever happens meanwhile (transient).
    #[serde(skip, default)]
    pub snapshot: Option<Vec<(WorkerId, u64)>>,
}

impl WorkflowData for WorkerUpdateWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_update"
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
