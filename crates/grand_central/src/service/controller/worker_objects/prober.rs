//! The health probes of the replica's workers while it leads.

use std::sync::Arc;

use smg_worker::{
    health::manager::{WorkerManager, WorkerManagerConfig},
    settings::UnhealthyRemovalScope,
    worker::HealthCheckConfig,
};
use tokio::sync::watch;

use super::{ElectedLead, WorkerObjects};
use crate::{
    service::elector::Fence,
    worker_host::{HealthAdapter, Lead, Members},
};

/// The probes of the replica's workers while it leads: the worker crate's
/// health loop over the holder's registry. Each outcome moves the
/// worker's object, and the health adapter tells the store the status
/// that stands, under the term this replica leads.
///
/// The loop probes what the holder holds. The registry tells it of each
/// object laid, replaced and ended, and nothing here reads a row. An object
/// carries its row's health policy: the row sets its own interval, timeout
/// and thresholds, and a row whose policy turns probes off is not probed.
/// A probe removes no row: a row leaves by its maker.
pub(crate) struct Prober {
    workers: Arc<WorkerObjects>,
    fence: Arc<Fence>,
    config: WorkerManagerConfig,
}

impl Prober {
    /// The probes of `workers` under `fence`. None runs before
    /// [`Self::run`].
    pub(crate) fn new(workers: Arc<WorkerObjects>, fence: Arc<Fence>) -> Self {
        Self {
            workers,
            fence,
            config: WorkerManagerConfig {
                // Every object carries its row's interval, so no object
                // reads this one.
                default_check_interval_secs: HealthCheckConfig::default().check_interval_secs,
                remove_unhealthy: UnhealthyRemovalScope::Disabled,
            },
        }
    }

    /// Runs one loop a term this replica leads and none while it leads
    /// none, until `stop` reads true. `led` is the role's word: the term
    /// this replica leads, if it leads one.
    ///
    /// A term is one value of the word, so a take under another term is
    /// another word however soon it follows a release. The loop of a term
    /// that ended is told to end, and its end is awaited before the next
    /// starts: two loops never run at once.
    ///
    /// An outcome that comes back behind its term's end moves nothing.
    /// The term's adapter asks the lead ([`ElectedLead`]) before it hands
    /// the outcome on and again under the object's lock, and the store
    /// asks it once more behind its own lock. A move the store takes no
    /// word of is taken back in the tell's own turn of the object's lock:
    /// the object takes the status its row holds.
    ///
    /// The task ends at the stop alone. Once the word's teller is gone
    /// nothing leads here: the loop ends and the task waits for the stop.
    pub(crate) async fn run(
        &self,
        mut stop: watch::Receiver<bool>,
        mut led: watch::Receiver<Option<u64>>,
    ) {
        let lead: Arc<dyn Lead> = Arc::new(ElectedLead::new(Arc::clone(&self.fence), led.clone()));
        let stopped = async {
            let _ = stop.wait_for(|stop| *stop).await;
        };
        tokio::pin!(stopped);
        loop {
            let term = tokio::select! {
                biased;
                () = &mut stopped => return,
                told = led.wait_for(Option::is_some) => told.ok().and_then(|term| *term),
            };
            // No term: the word's teller is gone.
            let Some(term) = term else { break };
            let mut probes = self.start(Arc::clone(&lead), term);
            // `None` at the stop, else whether the word still has its
            // teller.
            let told = tokio::select! {
                biased;
                () = &mut stopped => None,
                told = led.wait_for(|led| *led != Some(term)) => Some(told.is_ok()),
            };
            probes.shutdown().await;
            match told {
                None => return,
                Some(true) => {}
                Some(false) => break,
            }
        }
        stopped.await;
    }

    /// Starts the loop of `term`: the worker crate's health loop over the
    /// holder's registry, with the term's adapter as its sink. The loop
    /// reads the registry as it stands and probes at once each object that
    /// has no verdict yet.
    fn start(&self, lead: Arc<dyn Lead>, term: u64) -> WorkerManager {
        let adapter = self.adapter(lead, term);
        let registry = Arc::clone(self.workers.holder().registry());
        WorkerManager::start_with_sink(registry, adapter, self.config.clone(), None)
    }

    /// The health adapter of `term`: what writes an outcome judged under
    /// the term onto its row.
    fn adapter(&self, lead: Arc<dyn Lead>, term: u64) -> Arc<HealthAdapter<Members>> {
        Arc::new(HealthAdapter::new(
            self.workers.holder(),
            self.workers.store(),
            lead,
            term,
        ))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/service/controller/worker_objects/prober/tests.rs"]
mod tests;
