//! Process-wide state for the RL control plane: the registry view and the
//! configuration.

use std::sync::Arc;

use crate::{config::RlConfig, view::RlWorkerView};

pub struct RlState {
    pub(crate) view: Arc<dyn RlWorkerView>,
    pub(crate) config: RlConfig,
}

impl RlState {
    /// Build the state. The control plane owns no HTTP client: every call
    /// goes through the client the gateway negotiated for the target worker
    /// (see [`crate::RlWorkerInfo::http_client`]), with the control deadline
    /// applied per request. Breakers and load counters live on the gateway's
    /// worker objects, not on the client, so control calls leave them
    /// untouched.
    pub fn new(view: Arc<dyn RlWorkerView>, config: RlConfig) -> Self {
        Self { view, config }
    }
}
