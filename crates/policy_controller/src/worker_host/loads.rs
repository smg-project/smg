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
mod tests {
    use std::{slice, time::Instant};

    use smg_worker::{
        common_proto::EngineLoad, health::overload::OverloadThresholds, registry::WorkerRegistry,
        worker::WorkerLoadResponse,
    };

    use super::*;
    use crate::{
        engine::view::{WireModes, WorkerId},
        store::{
            GatewayId, MemoryStore, Overload, Reachability, WorkerFacts, WorkerStatus, WorkersStore,
        },
        worker_host::Members,
    };

    const URL: &str = "http://loads:8000";
    const MODEL: &str = "m";
    const PERIOD: Duration = Duration::from_secs(2);

    /// A holder that follows a store of its own.
    struct Replica {
        objects: Arc<Objects<Members>>,
        store: MemoryStore,
    }

    impl Replica {
        fn new() -> Self {
            Self {
                objects: Arc::new(Objects::new(Arc::new(WorkerRegistry::new()))),
                store: MemoryStore::with_publish_window(Duration::ZERO),
            }
        }

        /// A ready worker of `facts`, with the object the holder lays for
        /// it.
        fn add(&self, facts: WorkerFacts) -> (WorkerId, Arc<dyn Worker>) {
            let id = self.store.upsert(facts).id();
            let stored = WorkersStore::get(&self.store, id).expect("the registered record");
            self.store
                .set_status(id, WorkerStatus::Ready, stored.revision);
            self.store.publish_now();
            self.objects.follow(&self.store);
            let registry = self.objects.registry();
            let object = registry.get_by_url(&stored.facts.key);
            (id, object.expect("an object"))
        }
    }

    fn facts(url: &str) -> WorkerFacts {
        WorkerFacts {
            key: url.to_string(),
            engine: url.to_string(),
            incarnation: 1,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            wire_modes: WireModes::GRPC,
            ..Default::default()
        }
    }

    /// A host's load monitor asks no worker its host cannot dial. A holder
    /// that builds the object of a row one gateway alone reaches with its
    /// checks off says no of that worker, and the monitor wants no load
    /// stream for it. The holder says yes of a row every gateway reaches,
    /// of a worker it laid no object for, and of every worker where its
    /// host asked for no object with its checks off.
    #[test]
    fn a_hosts_load_monitor_asks_no_worker_its_host_cannot_dial() {
        const LOCAL: &str = "http://loads-local:8000";
        const STRANGER: &str = "http://loads-stranger:8000";
        let local = |url: &str| WorkerFacts {
            reachable_from: Reachability::Gateway(GatewayId(1)),
            ..facts(url)
        };
        let mut replica = Replica::new();
        replica.objects =
            Arc::new(Objects::new(Arc::new(WorkerRegistry::new())).with_host_local_unchecked());
        let (_, reached) = replica.add(facts(URL));
        let (_, unreached) = replica.add(local(LOCAL));
        let monitor = load_monitor(&replica.objects, PERIOD);
        assert!(monitor.loads_wanted(slice::from_ref(&reached)));
        assert!(!monitor.loads_wanted(slice::from_ref(&unreached)));

        // A host that asked for no object with its checks off dials every
        // worker, and the first holder laid no object at this one's address.
        let checking = Replica::new();
        let (_, stranger) = checking.add(local(STRANGER));
        let dialed = load_monitor(&checking.objects, PERIOD);
        assert!(dialed.loads_wanted(slice::from_ref(&stranger)));
        assert!(monitor.loads_wanted(slice::from_ref(&stranger)));
    }

    /// A host's load monitor judges a worker where its report lands, by the
    /// thresholds the holder built the object with, and the verdict is the
    /// object's. A usage of 0.8 as an engine reports it is over a threshold
    /// of 0.8. Its reports stand in the snapshot its channel publishes.
    /// Its drop says nothing to the objects: the verdicts stand until the
    /// host takes them off or another monitor starts, and a reader still
    /// holds the last snapshot. Another monitor's snapshot starts empty.
    /// It takes the host's period, and it wants the loads of every worker:
    /// of one that states no threshold too.
    ///
    /// The reports are handed as a stream's records, the shortest way to
    /// the monitor's sink: it judges a poll's answer by the same rule.
    #[tokio::test]
    async fn a_hosts_load_monitor_judges_by_the_objects_thresholds_and_publishes_its_reports() {
        const OTHER: &str = "http://loads-b:8000";
        let mut replica = Replica::new();
        let defaults = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        };
        replica.objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new())).with_overload_thresholds(defaults),
        );
        let (_, strict) = replica.add(WorkerFacts {
            overload: Overload {
                waiting_requests: Some(2),
                ..Default::default()
            },
            ..facts(URL)
        });
        let (_, plain) = replica.add(facts(OTHER));
        let monitor = load_monitor(&replica.objects, PERIOD);
        let published = monitor.subscribe();
        // The host's period reaches the monitor, whose debug text alone
        // shows its interval.
        let period = format!("default_interval: {PERIOD:?}");
        assert!(format!("{monitor:?}").contains(&period), "{monitor:?}");
        // Every group is polled, whatever its objects state: the loads of a
        // worker that states no threshold are wanted too.
        let bare = Replica::new();
        let (_, unjudged) = bare.add(facts(URL));
        let unjudging = load_monitor(&bare.objects, PERIOD);
        assert!(unjudging.loads_wanted(slice::from_ref(&unjudged)));

        let waiting = |waiting_requests| EngineLoad {
            waiting_requests,
            ..Default::default()
        };
        let now = Instant::now();
        // Three wait: over the first row's own two, under the host's eight.
        monitor.apply_pushed_load(&strict, 0, &waiting(3), now);
        monitor.apply_pushed_load(&plain, 0, &waiting(3), now);
        assert!(strict.is_overloaded());
        assert!(!plain.is_overloaded());
        // Nine wait: over the host's eight, for the row that names none.
        monitor.apply_pushed_load(&plain, 0, &waiting(9), now);
        assert!(plain.is_overloaded());
        // The first row's queue is empty again, and its usage moves.
        let used = |token_usage| EngineLoad {
            token_usage,
            ..Default::default()
        };
        monitor.apply_pushed_load(&strict, 0, &used(0.79), now);
        assert!(!strict.is_overloaded());
        monitor.apply_pushed_load(&strict, 0, &used(0.8), now);
        assert!(strict.is_overloaded());
        monitor.apply_pushed_load(&strict, 0, &waiting(3), now);

        monitor.flush_pushed_for_test();
        let waits = |url| {
            let snapshot = published.borrow();
            snapshot
                .get(url)
                .map(WorkerLoadResponse::total_waiting_reqs)
        };
        assert_eq!((waits(URL), waits(OTHER)), (Some(3), Some(9)));

        drop(monitor);
        assert!(strict.is_overloaded() && plain.is_overloaded());
        assert_eq!((waits(URL), waits(OTHER)), (Some(3), Some(9)));

        // Another monitor over the same objects starts with an empty
        // snapshot, and its start fails every standing verdict open.
        let next = load_monitor(&replica.objects, PERIOD);
        let again = next.subscribe();
        let empty = |url| again.borrow().get(url).is_none();
        assert!(empty(URL) && empty(OTHER));
        next.start_event_loop();
        assert!(!strict.is_overloaded() && !plain.is_overloaded());
        // Its own verdict stands behind its drop, until the host takes it
        // off.
        next.apply_pushed_load(&plain, 0, &waiting(9), now);
        drop(next);
        assert!(plain.is_overloaded());
        let registry = replica.objects.registry();
        assert_eq!(registry.clear_all_overload_flags(), 1);
        assert!(!plain.is_overloaded());
    }
}
