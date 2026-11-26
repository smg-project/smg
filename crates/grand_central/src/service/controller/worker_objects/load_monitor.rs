//! The load reports of the replica's workers while it leads.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use smg_worker::{
    common_proto::EngineLoad,
    identity::basic::Worker,
    kv_events::PushedLoadSink,
    load::{monitor::WorkerMonitor, state::LoadReceiver},
    registry::WorkerRegistry,
};
use tokio::sync::watch;

use super::WorkerObjects;
use crate::{
    service::controller::metrics::LoadTotals,
    worker_host::{load_monitor, Members},
};

/// The load reports of the replica's workers while it leads: the worker
/// crate's own load monitor over the holder's registry, one for the
/// replica's life, started at each take of the lead and stopped at each
/// loss of it.
///
/// While the replica leads the monitor polls every `Ready` object the
/// replica dials and publishes the reports as one snapshot. It takes the
/// load records on the workers' streams as polls ([`PushedRecords`]): a
/// worker's cache-event stream carries them, and a gRPC worker that is not
/// indexed streams them alone ([`Self::wanted`]). Its own sink writes the
/// overload verdict on each object by the thresholds the object states. It
/// writes no row: a report is read where the monitor publishes it. A row
/// that one gateway alone reaches is asked for no load, and has no stream
/// for its load records.
///
/// The serve runs this task, and every deciding thread holds the monitor's
/// channel from the bind: a single request's decision is given the newest
/// snapshot on it, where its document reads reports and keeps none.
pub(crate) struct LoadMonitor {
    monitor: Arc<WorkerMonitor<Members>>,
    /// The holder's registry, which writes the monitor's overload verdicts
    /// on the objects and counts each one that changes.
    registry: Arc<WorkerRegistry<Members>>,
    /// Where the load records on the workers' streams go: the serve hands
    /// it to its feed of those streams.
    pushed: Arc<PushedRecords>,
}

/// Where the load records on the workers' streams go: the replica's load
/// monitor while this replica leads, and nowhere while it leads none.
///
/// The streams stand on every replica: a worker's cache-event stream for
/// the events it carries, and the stream of a gRPC worker that is not
/// indexed for its load records alone, so that a take finds it open. The
/// worker crate's monitor takes a pushed record whether it runs or not. So
/// the replica's word is kept here. A record that comes while the monitor
/// runs is handed on, and the monitor takes it as a poll: a worker whose
/// records are fresh is polled no more. A record that comes while it is
/// stopped is dropped: a standby files no report, writes no verdict and
/// starts no task of the monitor's for it.
///
/// A stream's task hands a record on under a shared hold of the word.
/// [`LoadMonitor::run`] turns the word under the exclusive hold: it sets
/// it behind the monitor's start and clears it ahead of the monitor's
/// stop, so no record lands behind a stop. Nothing else takes either
/// hold, and no decision does.
pub(crate) struct PushedRecords {
    monitor: Arc<WorkerMonitor<Members>>,
    /// Whether the monitor runs.
    runs: RwLock<bool>,
}

impl PushedLoadSink for PushedRecords {
    fn push_load(
        self: Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &EngineLoad,
        received_at: Instant,
    ) {
        let runs = self.runs.read();
        if *runs {
            self.monitor
                .apply_pushed_load(worker, dp_rank, record, received_at);
        }
    }
}

impl LoadMonitor {
    /// The monitor of `workers`, which polls each worker once an `every`
    /// while this replica leads. It is built stopped: [`Self::run`] starts
    /// it.
    pub(crate) fn new(workers: &WorkerObjects, every: Duration) -> Self {
        let holder = workers.holder();
        let monitor = load_monitor(&holder, every);
        Self {
            pushed: Arc::new(PushedRecords {
                monitor: Arc::clone(&monitor),
                runs: RwLock::new(false),
            }),
            monitor,
            registry: Arc::clone(holder.registry()),
        }
    }

    /// The sink of the load records on the workers' streams, for the
    /// serve's feed of those streams: one for the replica's life, as the
    /// monitor is.
    pub(crate) fn pushed(&self) -> &Arc<PushedRecords> {
        &self.pushed
    }

    /// The monitor itself, which says whose loads are wanted
    /// ([`WorkerMonitor::loads_wanted`]): those of every worker the replica
    /// dials. The serve hands it to its feed of the workers' streams, which
    /// then opens a stream for the load records alone of a gRPC worker that
    /// is not indexed. The answer does not follow the role, so a standby
    /// opens that stream too, and [`PushedRecords`] drops its records.
    pub(crate) fn wanted(&self) -> &Arc<WorkerMonitor<Members>> {
        &self.monitor
    }

    /// The monitor's channel, for the reader of its snapshots: one channel
    /// for the replica's life. The monitor publishes on it under every
    /// lead, and publishes it empty at each lead's end.
    pub(crate) fn loads(&self) -> LoadReceiver {
        self.monitor.subscribe()
    }

    /// The reader of what the monitor's polls did and of how often an
    /// overload verdict of a worker changed, for the metrics render: the
    /// worker crate's own counts, read at each call. The counts are the
    /// replica's: they stand through every stop and start of the monitor,
    /// and a lead's end, which takes every verdict off, counts each one it
    /// takes.
    pub(crate) fn counts(&self) -> impl Fn() -> LoadTotals + Send + Sync + 'static {
        let (monitor, registry) = (Arc::clone(&self.monitor), Arc::clone(&self.registry));
        move || LoadTotals {
            polls: monitor.poll_totals(),
            overload_flips: registry.overload_edges(),
        }
    }

    /// Runs the load monitor while this replica leads and stops it while it
    /// leads none, until `stop` reads true. `led` is the role's word: the
    /// term this replica leads, if it leads one. The monitor runs for the
    /// whole lead, whatever the documents say: a decision over the objects
    /// reads the monitor's verdicts off them under every policy.
    ///
    /// At the word's first term the task starts the monitor
    /// ([`WorkerMonitor::start_event_loop`]) and then lets the streams'
    /// load records in ([`PushedRecords`]). At the word that nobody leads
    /// here it shuts them out, then stops the monitor and waits for the
    /// stop's end ([`WorkerMonitor::stop`]). Every task of the monitor
    /// that stands has ended then, a poll in flight with its task. The
    /// snapshot is published empty and every overload verdict is off the
    /// objects. The monitor keeps nothing that a stream's records filed,
    /// no rank of a report and no publish that waited for its window, and
    /// no object stays marked as a worker whose stream pushes: the
    /// next lead starts each worker from no report. So a replica that does
    /// not lead polls no engine for its load, and the next lead's start
    /// comes behind the last poll. This task alone starts and stops the
    /// monitor, each in its turn.
    ///
    /// One loop is not awaited: that of a group which a registration, a
    /// removal or a change of its interval ended just ahead of the stop. A
    /// tick of it that is past its fetch can still publish a report and
    /// write a verdict behind the stop. They stand until the next start,
    /// which clears the snapshot and takes every verdict off ahead of its
    /// first poll: a lead starts from no report, and a decision reads one of
    /// the lead before only in the instant between a take and that start.
    ///
    /// A word that goes from one term to the next leaves the monitor
    /// running. It writes nothing under a term, so a term's end needs no
    /// end of its own: an engine's report stands whoever leads.
    ///
    /// The task ends at the stop alone, behind the same stop of its
    /// monitor. Once the word's teller is gone nothing leads here: the
    /// monitor is stopped the same way and the task waits for the stop. A
    /// task that is dropped ahead of its stop stops nothing: its monitor
    /// runs on until it is dropped.
    pub(crate) async fn run(
        &self,
        mut stop: watch::Receiver<bool>,
        mut led: watch::Receiver<Option<u64>>,
    ) {
        let stopped = async {
            let _ = stop.wait_for(|stop| *stop).await;
        };
        tokio::pin!(stopped);
        loop {
            let leads = tokio::select! {
                biased;
                () = &mut stopped => return,
                told = led.wait_for(Option::is_some) => told.is_ok(),
            };
            // No word: its teller is gone.
            if !leads {
                break;
            }
            self.monitor.start_event_loop();
            *self.pushed.runs.write() = true;
            // `None` at the stop, else whether the word still has its
            // teller.
            let told = tokio::select! {
                biased;
                () = &mut stopped => None,
                told = led.wait_for(Option::is_none) => Some(told.is_ok()),
            };
            // Ahead of the stop: a record that came behind it would file a
            // report and start a task the stop no longer ends.
            *self.pushed.runs.write() = false;
            self.monitor.stop().await;
            match told {
                None => return,
                Some(true) => {}
                Some(false) => break,
            }
        }
        stopped.await;
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/service/controller/worker_objects/load_monitor/tests.rs"]
mod tests;
