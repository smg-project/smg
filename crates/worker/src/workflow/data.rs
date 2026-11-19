//! The data the worker workflows pass from step to step.

use std::{collections::HashSet, sync::Arc};

use serde::{Deserialize, Serialize};
use wfaas::WorkflowData;

use super::{find_workers_to_remove::WorkerRemovalRequest, removal::WORKER_REMOVAL_WORKFLOW};
use crate::identity::basic::Worker;

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
    pub workers_to_remove: Option<WorkerList>,
    /// URLs of workers being removed
    pub worker_urls: Vec<String>,
    /// Model IDs affected by the removal
    pub affected_models: HashSet<String>,
    /// The host (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub host: Option<Arc<H>>,
    /// Actual worker objects to remove (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers_to_remove: Option<Vec<Arc<dyn Worker>>>,
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
            workers_to_remove: self.workers_to_remove.clone(),
            worker_urls: self.worker_urls.clone(),
            affected_models: self.affected_models.clone(),
            host: self.host.clone(),
            actual_workers_to_remove: self.actual_workers_to_remove.clone(),
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
