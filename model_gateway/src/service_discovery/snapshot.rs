//! The snapshot a provider that reads its whole source at once keeps between
//! reads, shared by the file and Slurm providers.

use std::{fmt::Display, sync::Arc};

use tokio::time::Instant;
use tracing::{info, warn};

use super::{
    provider::DiscoveryKind,
    reconciler::{self, DesiredState},
};
use crate::{app_context::AppContext, observability::metrics::Metrics};

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
