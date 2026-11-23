//! Typed workflow engines collection
//!
//! This module provides a collection of typed workflow engines for different workflow types.
//! Each workflow type has its own engine with compile-time type safety.

use std::{sync::Arc, time::Duration};

use wfaas::{EventSubscriber, InMemoryStore, WorkflowEngine};

use super::{
    create_mcp_registration_workflow, create_tokenizer_registration_workflow,
    create_wasm_module_registration_workflow, create_wasm_module_removal_workflow,
    create_worker_registration_workflow, create_worker_removal_workflow,
    create_worker_update_workflow, McpWorkflowData, TokenizerWorkflowData,
    WasmRegistrationWorkflowData, WasmRemovalWorkflowData, WorkerRemovalWorkflowData,
    WorkerUpdateWorkflowData,
};
use crate::{config::RouterConfig, workflow::data::WorkerWorkflowData};

/// Retention for terminal workflow state after its last update. Waiters poll
/// at sub-second cadence and observe a terminal state within seconds, so this
/// only has to outlive that window while bounding state leaked by waiters
/// that timed out before their workflow terminated.
const TERMINAL_STATE_TTL: Duration = Duration::from_secs(300);

/// How often the background sweep reclaims expired terminal state.
const STATE_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Type alias for the unified worker registration workflow engine
pub type WorkerRegistrationEngine =
    WorkflowEngine<WorkerWorkflowData, InMemoryStore<WorkerWorkflowData>>;

/// Type alias for worker removal workflow engine
pub type WorkerRemovalEngine =
    WorkflowEngine<WorkerRemovalWorkflowData, InMemoryStore<WorkerRemovalWorkflowData>>;

/// Type alias for worker update workflow engine
pub type WorkerUpdateEngine =
    WorkflowEngine<WorkerUpdateWorkflowData, InMemoryStore<WorkerUpdateWorkflowData>>;

/// Type alias for MCP registration workflow engine
pub type McpEngine = WorkflowEngine<McpWorkflowData, InMemoryStore<McpWorkflowData>>;

/// Type alias for tokenizer registration workflow engine
pub type TokenizerEngine =
    WorkflowEngine<TokenizerWorkflowData, InMemoryStore<TokenizerWorkflowData>>;

/// Type alias for WASM registration workflow engine
pub type WasmRegistrationEngine =
    WorkflowEngine<WasmRegistrationWorkflowData, InMemoryStore<WasmRegistrationWorkflowData>>;

/// Type alias for WASM removal workflow engine
pub type WasmRemovalEngine =
    WorkflowEngine<WasmRemovalWorkflowData, InMemoryStore<WasmRemovalWorkflowData>>;

/// Collection of typed workflow engines
///
/// Each workflow type has its own engine with compile-time type safety.
/// This replaces the old `WorkflowEngine<AnyWorkflowData, ...>` approach.
#[derive(Clone, Debug)]
pub struct WorkflowEngines {
    /// Engine for unified worker registration workflows (local + external)
    pub worker_registration: Arc<WorkerRegistrationEngine>,
    /// Engine for worker removal workflows
    pub worker_removal: Arc<WorkerRemovalEngine>,
    /// Engine for worker update workflows
    pub worker_update: Arc<WorkerUpdateEngine>,
    /// Engine for MCP server registration workflows
    pub mcp: Arc<McpEngine>,
    /// Engine for tokenizer registration workflows
    pub tokenizer: Arc<TokenizerEngine>,
    /// Engine for WASM module registration workflows
    pub wasm_registration: Arc<WasmRegistrationEngine>,
    /// Engine for WASM module removal workflows
    pub wasm_removal: Arc<WasmRemovalEngine>,
}

impl WorkflowEngines {
    /// Create and initialize all workflow engines with their workflow definitions.
    ///
    /// Must be called within a Tokio runtime: it spawns the periodic state sweepers.
    #[expect(
        clippy::expect_used,
        reason = "Workflow registration uses compile-time-known step/transition definitions that cannot fail at runtime — a failure here indicates a programming error in workflow construction"
    )]
    pub fn new(router_config: &RouterConfig) -> Self {
        // Create unified worker registration engine
        let worker_registration = WorkflowEngine::new();
        worker_registration
            .register_workflow(create_worker_registration_workflow(router_config))
            .expect("worker_registration workflow should be valid");

        // Create worker removal engine
        let worker_removal = WorkflowEngine::new();
        worker_removal
            .register_workflow(create_worker_removal_workflow())
            .expect("worker_removal workflow should be valid");

        // Create worker update engine
        let worker_update = WorkflowEngine::new();
        worker_update
            .register_workflow(create_worker_update_workflow())
            .expect("worker_update workflow should be valid");

        // Create MCP engine
        let mcp = WorkflowEngine::new();
        mcp.register_workflow(create_mcp_registration_workflow())
            .expect("mcp_registration workflow should be valid");

        // Create tokenizer engine
        let tokenizer = WorkflowEngine::new();
        tokenizer
            .register_workflow(create_tokenizer_registration_workflow())
            .expect("tokenizer_registration workflow should be valid");

        // Create WASM registration engine
        let wasm_registration = WorkflowEngine::new();
        wasm_registration
            .register_workflow(create_wasm_module_registration_workflow())
            .expect("wasm_module_registration workflow should be valid");

        // Create WASM removal engine
        let wasm_removal = WorkflowEngine::new();
        wasm_removal
            .register_workflow(create_wasm_module_removal_workflow())
            .expect("wasm_module_removal workflow should be valid");

        let engines = Self {
            worker_registration: Arc::new(worker_registration),
            worker_removal: Arc::new(worker_removal),
            worker_update: Arc::new(worker_update),
            mcp: Arc::new(mcp),
            tokenizer: Arc::new(tokenizer),
            wasm_registration: Arc::new(wasm_registration),
            wasm_removal: Arc::new(wasm_removal),
        };
        engines.start_state_sweepers();
        engines
    }

    /// Reclaim terminal workflow state whose waiter timed out before the
    /// workflow terminated; waiter-side cleanup runs only when the waiter
    /// observes the terminal state. The tasks stop on engine shutdown.
    fn start_state_sweepers(&self) {
        self.worker_registration
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.worker_removal
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.worker_update
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.mcp
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.tokenizer
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.wasm_registration
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
        self.wasm_removal
            .start_cleanup_task(Some(TERMINAL_STATE_TTL), Some(STATE_SWEEP_INTERVAL));
    }

    /// Subscribe an event subscriber to all workflow engines
    pub async fn subscribe_all<S: EventSubscriber + 'static>(&self, subscriber: Arc<S>) {
        self.worker_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_removal
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_update
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.mcp.event_bus().subscribe(subscriber.clone()).await;
        self.tokenizer
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_removal.event_bus().subscribe(subscriber).await;
    }
}
