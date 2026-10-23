//! Process-wide state for the RL control plane: the registry view and the
//! configuration.

use std::sync::Arc;

use crate::{
    config::RlConfig,
    versions::ReportedVersions,
    view::{RlWorkerInfo, RlWorkerView},
};

pub struct RlState {
    pub(crate) view: Arc<dyn RlWorkerView>,
    pub(crate) config: RlConfig,
    pub(crate) versions: ReportedVersions,
}

impl RlState {
    /// Build the state. The control plane owns no HTTP client: every call
    /// goes through the client the gateway resolved for the target worker's
    /// control endpoint (see [`crate::RlWorkerInfo::control_client`]), with
    /// the control deadline applied per request. Breakers and load counters
    /// live on the gateway's worker objects, not on the client, so control
    /// calls leave them untouched.
    pub fn new(view: Arc<dyn RlWorkerView>, config: RlConfig) -> Self {
        Self {
            view,
            config,
            versions: ReportedVersions::default(),
        }
    }
}

/// The registry as discovery and selectors see it: each worker with the
/// weight version the control plane last stamped on it.
impl RlWorkerView for RlState {
    fn list(&self) -> Vec<RlWorkerInfo> {
        self.versions.apply(self.view.list())
    }
    fn get(&self, id: &str) -> Option<RlWorkerInfo> {
        let worker = self.view.get(id)?;
        self.versions.apply(vec![worker]).pop()
    }
}
