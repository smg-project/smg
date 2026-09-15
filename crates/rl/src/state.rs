//! Process-wide state for the RL control plane: the registry view, the
//! configuration, and the version/control-state table.

use std::{borrow::Cow, sync::Arc};

use http::HeaderMap;
use tracing::info;

use crate::{
    config::RlConfig,
    metrics,
    observe::Observation,
    policy::VersionPolicy,
    table::{ControlState, NoopEvictionSink, RlTable, VersionEvictionSink, VersionSource},
    version::Version,
    view::{RlWorkerInfo, RlWorkerView},
};

pub struct RlState {
    pub(crate) view: Arc<dyn RlWorkerView>,
    pub(crate) config: RlConfig,
    pub(crate) table: RlTable,
}

impl RlState {
    /// Build the state with a table that evicts nothing on version changes
    /// (tests and crate-only use). The control plane owns no HTTP client:
    /// every call goes through the client the gateway negotiated for the
    /// target worker (see [`RlWorkerInfo::http_client`]), with the control
    /// deadline applied per request. Breakers and load counters live on the
    /// gateway's worker objects, not on the client, so control calls leave
    /// them untouched.
    pub fn new(view: Arc<dyn RlWorkerView>, config: RlConfig) -> Self {
        Self::with_sink(view, config, Arc::new(NoopEvictionSink))
    }

    /// Build the state with the gateway's eviction sink.
    pub fn with_sink(
        view: Arc<dyn RlWorkerView>,
        config: RlConfig,
        sink: Arc<dyn VersionEvictionSink>,
    ) -> Self {
        Self {
            view,
            config,
            table: RlTable::new(sink),
        }
    }

    pub fn table(&self) -> &RlTable {
        &self.table
    }

    pub fn config(&self) -> &RlConfig {
        &self.config
    }

    /// The policy for one request: a valid header override, else the
    /// configured policy. (The middleware already rejected invalid headers.)
    pub fn policy_for(&self, headers: Option<&HeaderMap>) -> Cow<'_, VersionPolicy> {
        match headers.and_then(|h| VersionPolicy::from_headers(h).ok().flatten()) {
            Some(policy) => Cow::Owned(policy),
            None => Cow::Borrowed(&self.config.version_policy),
        }
    }

    /// Record a version for `worker`'s engine, with the inflight accounting.
    pub fn apply_version(
        &self,
        worker: &RlWorkerInfo,
        version: Version,
        source: VersionSource,
    ) -> bool {
        let changed =
            self.table
                .set_version(&worker.base_url, &worker.model_id, version.clone(), source);
        if changed {
            if self.view.inflight(&worker.base_url) > 0 {
                metrics::record_version_change_with_inflight();
            }
            info!(
                target: "smg_rl",
                base_url = %worker.base_url, version = %version, source = source.as_str(),
                "rl.observe version"
            );
        }
        changed
    }

    /// Apply what a successful proxied call revealed about `worker`'s engine.
    pub fn apply_observation(
        &self,
        worker: &RlWorkerInfo,
        observation: Observation,
        source: VersionSource,
    ) {
        match observation {
            Observation::Version(version) => {
                self.apply_version(worker, version, source);
            }
            Observation::Control(control) => {
                self.apply_control(worker, control);
            }
        }
    }

    /// Record a control state for `worker`'s engine.
    pub fn apply_control(&self, worker: &RlWorkerInfo, control: ControlState) -> bool {
        let changed = self
            .table
            .set_control(&worker.base_url, &worker.model_id, control);
        if changed {
            info!(
                target: "smg_rl",
                base_url = %worker.base_url, control = control.as_str(),
                "rl.observe control"
            );
        }
        changed
    }
}
