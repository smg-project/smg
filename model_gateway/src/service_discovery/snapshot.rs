//! The snapshot a provider that reads its whole source at once keeps between
//! reads, shared by the file, Slurm and Consul providers, and what such a read
//! left out.

use std::{collections::HashSet, fmt::Display, sync::Arc};

use tokio::time::Instant;
use tracing::{info, warn};

use super::{
    provider::DiscoveryKind,
    reconciler::{self, DesiredState},
};
use crate::{app_context::AppContext, observability::metrics::Metrics};

/// A record a read left out of an answer it otherwise applied.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Skip {
    /// The `reason` of `smg_discovery_skipped_total`.
    pub(super) reason: &'static str,
    pub(super) message: String,
}

/// What reads leave out: counted on every read, and logged when it first
/// appears, so a skip that lasts is logged once.
pub(super) struct SkipLog {
    kind: DiscoveryKind,
    /// How logs name the provider.
    provider: &'static str,
    /// What the last read left out.
    last: HashSet<String>,
}

impl SkipLog {
    pub(super) fn new(kind: DiscoveryKind, provider: &'static str) -> Self {
        Self {
            kind,
            provider,
            last: HashSet::new(),
        }
    }

    /// Take what one read left out.
    pub(super) fn record(&mut self, skipped: Vec<Skip>) {
        let mut last = HashSet::new();
        for skip in skipped {
            Metrics::record_discovery_skip(self.kind.metric_label(), skip.reason);
            if !self.last.contains(&skip.message) {
                warn!("{} left out {}", self.provider, skip.message);
            }
            last.insert(skip.message);
        }
        self.last = last;
    }
}

/// Why a read produced no snapshot.
pub(super) trait SnapshotError: Display {
    /// The `reason` of `smg_discovery_snapshot_errors_total`.
    fn metric_reason(&self) -> &'static str;
}

/// The last snapshot a provider read successfully, which every pass
/// reconciles. A read that fails is reported and leaves it in place, so a
/// broken source never removes a worker.
pub(super) struct LastGood {
    kind: DiscoveryKind,
    /// What the snapshot was read from, as logs name it.
    source: String,
    /// What a failed read keeps, as logs name it.
    noun: &'static str,
    desired: Option<DesiredState>,
}

impl LastGood {
    pub(super) fn new(kind: DiscoveryKind, source: String, noun: &'static str) -> Self {
        Self {
            kind,
            source,
            noun,
            desired: None,
        }
    }

    /// Take the outcome of a read.
    pub(super) fn update<E: SnapshotError>(&mut self, read: Result<DesiredState, E>) {
        match read {
            Ok(desired) => {
                if self.desired.as_ref() != Some(&desired) {
                    info!("{} lists {} worker(s)", self.source, desired.workers.len());
                }
                self.desired = Some(desired);
            }
            Err(error) => {
                Metrics::record_discovery_snapshot_error(
                    self.kind.metric_label(),
                    error.metric_reason(),
                );
                if self.desired.is_some() {
                    warn!("{error}; keeping the last good {}", self.noun);
                } else {
                    warn!("{error}; no workers discovered yet");
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn desired(&self) -> Option<&DesiredState> {
        self.desired.as_ref()
    }

    /// Reconcile the last good snapshot, once there is one.
    pub(super) async fn reconcile(&self, app_context: &Arc<AppContext>, started_at: Instant) {
        if let Some(desired) = &self.desired {
            reconciler::reconcile(desired, self.kind, app_context, started_at).await;
        }
    }
}
