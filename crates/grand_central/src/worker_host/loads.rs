//! A host's worker loads: the worker crate's own load monitor, for an
//! engine that decides over its host's lists ([`load_monitor`]).

use std::{sync::Arc, time::Duration};

use smg_worker::{
    identity::basic::Worker,
    load::monitor::{LoadConsumer, WorkerMonitor},
    registry::Generation,
};

use super::Objects;

/// The readers of a [`load_monitor`]'s loads: the decisions of an engine
/// over the host's lists, which read the snapshot the monitor publishes.
/// They hold the holder of the objects, which says the workers its host
/// dials.
struct SnapshotReaders<G>(Arc<Objects<G>>);

impl<G: Generation> LoadConsumer for SnapshotReaders<G> {
    /// Always. The host runs the monitor only while a decision reads its
    /// snapshot, and stops it for the rest. The monitor asks under polling
    /// by need alone, which [`load_monitor`] leaves off.
    fn needs_loads(&self) -> bool {
        true
    }

    /// Nothing is trimmed. A decision books its dispatch in the engine's
    /// own stages, which are told the request's end, and none books through
    /// the monitor.
    fn reconcile_in_flight(&self, _worker: &dyn Worker) {}

    /// Whether the host dials the worker ([`Objects::dials`]): a host
    /// that is no gateway asks a worker one gateway alone reaches for no
    /// load. The answer stands for the object's life, so a worker left
    /// out leaves no report and no verdict of the monitor behind.
    fn polls(&self, worker: &dyn Worker) -> bool {
        self.0.dials(worker)
    }
}

/// The worker crate's own load monitor over the holder's registry, for a
/// host whose decisions read the monitor's snapshot.
///
/// It polls every `Ready` object its host dials each `every`, cut to whole
/// seconds: the monitor takes one second for less. An object whose load
/// stream pushed a record within that time on every rank that pushes is
/// not polled. An object built for a
/// row that one gateway alone reaches is asked for no load, where the
/// host asked for such objects with their checks off
/// ([`Objects::with_host_local_unchecked`]).
/// It publishes the reports as one
/// snapshot ([`WorkerMonitor::subscribe`]), which the host hands its
/// decisions. Its own sink judges each worker's overload by the thresholds
/// its object states ([`Objects::with_overload_thresholds`]) and writes
/// the verdict on the object. It writes no store and polls every group,
/// whatever its objects state. Each report also feeds the worker crate's
/// engine gauges, as under every monitor.
///
/// It is built stopped, and a host keeps the one it built. The host starts
/// it ([`WorkerMonitor::start_event_loop`]) where its decisions read the
/// snapshot, and stops it where they do not ([`WorkerMonitor::stop`]). The
/// stop ends every task of the monitor that stands and waits for each end.
/// Then it publishes the snapshot empty and takes every overload verdict
/// off the objects. The loop of a group that ended just ahead of the stop
/// is not awaited: a tick of it past its fetch can still publish a report
/// and write a verdict, which stand until the next start clears both. The
/// channel stands through a stop: a reader handed it once reads the next
/// start's reports on it.
///
/// A drop is no such end. It ends the monitor's tasks and its channel,
/// whose last snapshot a reader still holds, awaits nothing and
/// says nothing to the sink: the verdicts stand on the objects until the
/// host takes them off ([`clear_all_overload_flags`]). Another monitor's
/// snapshot starts empty, and its start fails every verdict open.
///
/// [`clear_all_overload_flags`]: smg_worker::registry::WorkerRegistry::clear_all_overload_flags
pub fn load_monitor<G: Generation>(
    objects: &Arc<Objects<G>>,
    every: Duration,
) -> Arc<WorkerMonitor<G>> {
    Arc::new(WorkerMonitor::new(
        Arc::clone(objects.registry()),
        Arc::new(SnapshotReaders(Arc::clone(objects))),
        every.as_secs(),
        // No poll is forced for the gauges' sake and none is left out by
        // need: every group is polled.
        false,
        false,
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/loads/tests.rs"]
mod tests;
