//! The data the worker workflows pass from step to step.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use wfaas::{StepId, WorkflowData, WorkflowError, WorkflowResult};

use super::{
    find_workers_to_remove::WorkerRemovalRequest,
    host::{require_host, WorkflowHost},
    registration::WORKER_REGISTRATION_WORKFLOW,
    removal::WORKER_REMOVAL_WORKFLOW,
};
use crate::{
    identity::basic::{ConnectionMode, Worker},
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    model_card::ModelCard,
    worker::WorkerSpec,
};

/// Wrapper for worker list that can be serialized
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerList {
    /// Worker URLs (we can't serialize `Arc<dyn Worker>`, so we store URLs)
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

/// Data for worker removal workflow, over the host `H` that runs it.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WorkerRemovalWorkflowData<H> {
    pub config: WorkerRemovalRequest,
    /// Who asked for the removal. A host's own step reads it.
    pub asker: WorkerJobAsker,
    pub workers_to_remove: Option<WorkerList>,
    /// URLs of workers being removed
    pub worker_urls: Vec<String>,
    /// Model IDs affected by the removal. The selection writes it; no step of
    /// this crate reads it.
    pub affected_models: HashSet<String>,
    /// The host (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub host: Option<Arc<H>>,
    /// Actual worker objects to remove (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers_to_remove: Option<Vec<Arc<dyn Worker>>>,
    /// The workers marked `Draining` by the registry drain or the host's
    /// drain. The settle reads its window from them. The list stays in memory.
    #[serde(skip, default)]
    pub drained_workers: Vec<Arc<dyn Worker>>,
    /// The workers the registry step removed at those URLs, a same-URL
    /// replacement that landed during the drain rather than the worker found
    /// (transient, not serialized)
    #[serde(skip, default)]
    pub removed_workers: Vec<Arc<dyn Worker>>,
}

// By hand: the derive would ask `H: Clone`, and the host is shared, not copied.
impl<H> Clone for WorkerRemovalWorkflowData<H> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            asker: self.asker.clone(),
            workers_to_remove: self.workers_to_remove.clone(),
            worker_urls: self.worker_urls.clone(),
            affected_models: self.affected_models.clone(),
            host: self.host.clone(),
            actual_workers_to_remove: self.actual_workers_to_remove.clone(),
            drained_workers: self.drained_workers.clone(),
            removed_workers: self.removed_workers.clone(),
        }
    }
}

impl<H> WorkerRemovalWorkflowData<H> {
    /// Narrows the job to `kept`, the workers it still holds: the selection's
    /// result, or what a step left after it found some of them standing on
    /// their own (a pin that moved, a registration under the URL since the
    /// selection). The worker list, the URLs and the affected models move
    /// together, so the steps after it touch nothing of the others.
    pub fn keep_only(&mut self, kept: Vec<Arc<dyn Worker>>) {
        self.worker_urls = kept.iter().map(|worker| worker.url().to_string()).collect();
        self.affected_models = kept
            .iter()
            .map(|worker| worker.model_id().to_string())
            .collect();
        self.workers_to_remove = Some(WorkerList::from_workers(&kept));
        self.actual_workers_to_remove = Some(kept);
    }
}

impl<H: Send + Sync + 'static> WorkflowData for WorkerRemovalWorkflowData<H> {
    fn workflow_type() -> &'static str {
        WORKER_REMOVAL_WORKFLOW
    }
}

/// What a registration adds: a worker this deployment runs, or an API another
/// party serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerKind {
    /// An inference engine this deployment runs.
    Local,
    /// An API another party serves.
    External,
}

/// What a worker reports of its data-parallel ranks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Data for the worker registration workflow, over the host `H` that runs it.
///
/// One workflow registers both kinds of worker. The host's first step sets
/// `worker_kind`, and a step of one kind skips a worker of the other.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WorkerRegistrationWorkflowData<H: WorkflowHost> {
    pub config: WorkerSpec,
    /// Defaults to upsert so persisted workflows from older versions preserve
    /// their historical registration behavior when deserialized.
    #[serde(default)]
    pub registration_mode: WorkerRegistrationMode,
    /// Who asked for the registration. A host's own step reads it.
    pub asker: WorkerJobAsker,
    /// Which kind of worker this is, set by the host's first step.
    pub worker_kind: Option<WorkerKind>,
    // -- Local-only fields --
    pub connection_mode: Option<ConnectionMode>,
    /// HTTP workers: whether the host speaks HTTP/2 with prior knowledge to
    /// the worker, resolved by the connection detection.
    #[serde(default)]
    pub http2: Option<bool>,
    /// The worker-directed client for `http2` (transient; see
    /// [`Self::http_client`]).
    #[serde(skip, default)]
    pub http_client_handle: Option<Arc<reqwest::Client>>,
    pub detected_runtime_type: Option<String>,
    /// What the worker's engine reports about itself, as labels. The metadata
    /// read writes it. For a worker whose engine reports nothing ahead of its
    /// handshake, a step of the host's own writes what the host knows.
    pub discovered_labels: HashMap<String, String>,
    /// The worker's data-parallel shape. The DP read writes it, where the
    /// worker reports a shape.
    pub dp_info: Option<DpInfo>,
    /// What the host's own steps keep on this registration between them
    /// (transient). No step of this crate reads or writes it.
    #[serde(skip, default)]
    pub host_notes: H::RegistrationNotes,
    // -- External-only fields --
    pub model_cards: Vec<ModelCard>,
    // -- Shared fields --
    pub workers: Option<WorkerList>,
    pub final_labels: HashMap<String, String>,
    /// The host (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub host: Option<Arc<H>>,
    /// Actual worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers: Option<Vec<Arc<dyn Worker>>>,
}

// By hand: the derive would ask `H: Clone`, and the host is shared, not copied.
impl<H: WorkflowHost> Clone for WorkerRegistrationWorkflowData<H> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            registration_mode: self.registration_mode.clone(),
            asker: self.asker.clone(),
            worker_kind: self.worker_kind,
            connection_mode: self.connection_mode,
            http2: self.http2,
            http_client_handle: self.http_client_handle.clone(),
            detected_runtime_type: self.detected_runtime_type.clone(),
            discovered_labels: self.discovered_labels.clone(),
            dp_info: self.dp_info.clone(),
            host_notes: self.host_notes.clone(),
            model_cards: self.model_cards.clone(),
            workers: self.workers.clone(),
            final_labels: self.final_labels.clone(),
            host: self.host.clone(),
            actual_workers: self.actual_workers.clone(),
        }
    }
}

impl<H: WorkflowHost> WorkerRegistrationWorkflowData<H> {
    /// A registration of `config` as it enters the workflow, asked by
    /// `asker`: nothing found out about the worker yet, and the host's notes
    /// at their default.
    pub fn new(
        config: WorkerSpec,
        registration_mode: WorkerRegistrationMode,
        host: Arc<H>,
        asker: WorkerJobAsker,
    ) -> Self {
        Self {
            config,
            registration_mode,
            asker,
            worker_kind: None,
            connection_mode: None,
            http2: None,
            http_client_handle: None,
            detected_runtime_type: None,
            discovered_labels: HashMap::new(),
            dp_info: None,
            host_notes: H::RegistrationNotes::default(),
            model_cards: Vec::new(),
            workers: None,
            final_labels: HashMap::new(),
            host: Some(host),
            actual_workers: None,
        }
    }

    /// The worker-directed client for the resolved HTTP version, re-fetched
    /// from the host's cache when a deserialize dropped the transient handle.
    pub fn http_client(&self, step_id: &str) -> WorkflowResult<Arc<reqwest::Client>> {
        if let Some(client) = &self.http_client_handle {
            return Ok(Arc::clone(client));
        }
        require_host(self.host.as_ref())?
            .worker_client_cache()
            .get(&self.config.http_pool, self.http2.unwrap_or(false))
            .map_err(|message| WorkflowError::StepFailed {
                step_id: StepId::new(step_id),
                message,
            })
    }
}

impl<H: WorkflowHost> WorkflowData for WorkerRegistrationWorkflowData<H> {
    fn workflow_type() -> &'static str {
        WORKER_REGISTRATION_WORKFLOW
    }
}

#[cfg(test)]
mod tests;
