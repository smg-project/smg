//! Workflow engine, step definitions, and workflow data types.

pub mod data;
pub mod engines;
pub mod job_queue;
pub mod mcp_registration;
pub mod steps;
pub mod tokenizer_registration;
pub mod wasm_module_registration;
pub mod wasm_module_removal;

// Worker management (registration, removal)
// Typed workflow data structures
pub use data::{
    McpWorkflowData, ProtocolUpdateRequest, TokenizerWorkflowData, WasmRegistrationWorkflowData,
    WasmRemovalWorkflowData, WorkerKind, WorkerList as WorkflowWorkerList, WorkerRegistrationData,
    WorkerRegistrationMode, WorkerRemovalWorkflowData, WorkerSpec, WorkerUpdateWorkflowData,
    WorkerWorkflowData,
};
// Typed workflow engines
pub use engines::WorkflowEngines;
pub use job_queue::{Job, JobQueue, JobQueueConfig};
pub use mcp_registration::{
    create_mcp_registration_workflow, create_mcp_workflow_data, ConnectMcpServerStep,
    McpServerConfigRequest, ValidateRegistrationStep,
};
pub use steps::{
    // Unified workflow builder + data factory
    create_worker_registration_workflow,
    // Removal/update workflow builders
    create_worker_removal_workflow,
    create_worker_removal_workflow_data,
    create_worker_update_workflow,
    create_worker_update_workflow_data,
    create_worker_workflow_data,
    // Utility functions
    group_models_into_cards,
    infer_model_type_from_id,
    // Shared steps
    ActivateWorkersStep,
    // Classification step
    ClassifyWorkerTypeStep,
    // External registration steps
    CreateExternalWorkersStep,
    // Local registration steps
    CreateLocalWorkerStep,
    DetectConnectionModeStep,
    DiscoverDPInfoStep,
    DiscoverMetadataStep,
    DiscoverModelsStep,
    DpInfo,
    // Update steps
    FindWorkerToUpdateStep,
    // Removal steps
    FindWorkersToRemoveStep,
    ModelInfo,
    ModelsResponse,
    RegisterWorkersStep,
    RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep,
    UpdatePoliciesForWorkerStep,
    UpdatePoliciesStep,
    UpdateRemainingPoliciesStep,
    UpdateWorkerPropertiesStep,
    WorkerList,
    WorkerRemovalRequest,
};
pub use tokenizer_registration::{
    create_tokenizer_registration_workflow, create_tokenizer_workflow_data,
    note_tokenizer_load_outcome, note_tokenizer_removed, LoadTokenizerStep, TokenizerConfigRequest,
    TokenizerLoadFailure, TokenizerLoadFailures, TokenizerRemovalRequest,
};
pub use wasm_module_registration::{
    create_wasm_module_registration_workflow, create_wasm_registration_workflow_data,
    CalculateHashStep, CheckDuplicateStep, LoadWasmBytesStep, RegisterModuleStep,
    ValidateDescriptorStep, ValidateWasmComponentStep, WasmModuleConfigRequest,
};
pub use wasm_module_removal::{
    create_wasm_module_removal_workflow, create_wasm_removal_workflow_data, FindModuleToRemoveStep,
    RemoveModuleStep, WasmModuleRemovalRequest,
};

pub use crate::config::TokenizerCacheConfig;
