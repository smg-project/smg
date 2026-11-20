//! The worker health sink over the store.
//!
//! The adapter reads the lead before it takes the object's mutation lock and
//! again under it. With the lock held, it reads the holder's map, releases
//! the read lock, and writes the store. The holder writes its map and the
//! registry under separate locks. Its status mirror reads the row and
//! writes the object under the same mutation lock as the adapter.

use std::sync::Arc;

use smg_worker::{
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{object::status, Lead, Objects, Standing};
use crate::store::{HealthSink as StoreHealthSink, HealthWrite};

/// The health sink for a replica that keeps its workers in the store.
pub struct HealthAdapter<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    store: Arc<S>,
    lead: Arc<dyn Lead>,
    term: u64,
}

impl<G: Generation, S: StoreHealthSink + ?Sized> HealthAdapter<G, S> {
    /// An adapter for `term`. Each call checks that `lead` still names it
    /// before taking the object's lock, and again under the lock.
    pub fn new(objects: Arc<Objects<G>>, store: Arc<S>, lead: Arc<dyn Lead>, term: u64) -> Self {
        Self {
            objects,
            store,
            lead,
            term,
        }
    }

    fn leads(&self) -> bool {
        self.lead.term() == Some(self.term)
    }

    fn write(
        &self,
        worker_id: &ObjectId,
        revision: u64,
        write_row: impl FnOnce(&S, Standing) -> HealthWrite,
    ) -> HealthWrite {
        if !self.leads() {
            return HealthWrite::Stale;
        }
        let apply = |object: &Arc<dyn Worker>| {
            // A call can wait here for the object's lock through the end of
            // its term: it writes nothing.
            if !self.leads() {
                return (HealthWrite::Stale, None);
            }
            let Some(standing) = self.objects.standing_of(worker_id, object) else {
                return (HealthWrite::Stale, None);
            };
            let write = write_row(self.store.as_ref(), standing);
            (write, write.stands().map(status))
        };
        self.objects
            .registry()
            .apply_if_revision(worker_id, revision, apply)
            .map_or(HealthWrite::Stale, |(write, _)| write)
    }
}

impl<G: Generation, S: StoreHealthSink + ?Sized + 'static> HealthSink for HealthAdapter<G, S> {
    /// The store counts every probe of a pending row, whatever status the
    /// probe was launched at.
    fn probe_outcome(
        &self,
        worker_id: &ObjectId,
        passed: bool,
        revision: u64,
        _launched: ObjectStatus,
    ) -> ProbeAnswer {
        match self.write(worker_id, revision, |store, standing| {
            store.probe_outcome(standing.row, passed, standing.revision)
        }) {
            HealthWrite::Moved { from, to } => ProbeAnswer::Moved {
                old: status(from),
                new: status(to),
            },
            HealthWrite::Stood(standing) => ProbeAnswer::Stood(status(standing)),
            HealthWrite::Stale | HealthWrite::NoRow => ProbeAnswer::Stale,
        }
    }

    fn connected(
        &self,
        worker_id: &ObjectId,
        revision: u64,
    ) -> Option<(ObjectStatus, ObjectStatus)> {
        self.write(worker_id, revision, |store, standing| {
            store.connected(standing.row, standing.revision)
        })
        .moved()
        .map(|(from, to)| (status(from), status(to)))
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use smg_worker::registry::{event::WorkerEvent, WorkerRegistry};
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;
    use crate::{
        store::{
            HealthPolicy, LeasesStore, MemoryStore, WorkerFacts, WorkerPatch,
            WorkerStatus as RowStatus, WorkersStore,
        },
        worker_host::{object, Members, SoleLead},
    };

    const URL: &str = "http://a:8000";

    struct Replica {
        store: Arc<MemoryStore>,
        objects: Arc<Objects<Members>>,
        id: ObjectId,
        revision: u64,
    }

    impl Replica {
        fn new(success_threshold: u32, probes: bool) -> Option<Self> {
            let store = Arc::new(MemoryStore::new());
            let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
            let row = store
                .upsert(WorkerFacts {
                    key: URL.to_string(),
                    health: HealthPolicy {
                        success_threshold,
                        probes,
                        ..HealthPolicy::default()
                    },
                    ..WorkerFacts::default()
                })
                .id();
            objects.follow(store.as_ref());
            let id = objects.object(row)?;
            let revision = objects.registry().get(&id)?.revision();
            Some(Self {
                store,
                objects,
                id,
                revision,
            })
        }

        fn adapter(&self) -> HealthAdapter<Members, dyn StoreHealthSink> {
            let store: Arc<dyn StoreHealthSink> = self.store.clone();
            HealthAdapter::new(
                Arc::clone(&self.objects),
                store,
                Arc::new(SoleLead),
                SoleLead::TERM,
            )
        }

        #[track_caller]
        fn assert_statuses(&self, row: Option<RowStatus>, object: ObjectStatus) {
            assert_eq!(self.store.by_key(URL).map(|row| row.status), row);
            assert_eq!(
                self.objects.registry().get(&self.id).map(|w| w.status()),
                Some(object)
            );
        }
    }

    struct TestLead(Mutex<Option<u64>>);

    impl Lead for TestLead {
        fn term(&self) -> Option<u64> {
            *self.0.lock()
        }
    }

    /// A lead whose term ends between its first reading and its second.
    struct TurningLead(Mutex<u32>);

    impl Lead for TurningLead {
        fn term(&self) -> Option<u64> {
            let mut readings = self.0.lock();
            *readings += 1;
            Some(if *readings == 1 { 1 } else { 2 })
        }
    }

    #[test]
    fn a_probe_that_moves_the_row_answers_moved_and_sets_the_object() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Moved {
                old: ObjectStatus::Pending,
                new: ObjectStatus::Ready,
            }
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
    }

    #[test]
    fn a_probe_short_of_its_threshold_answers_the_status_that_stands() {
        let replica = Replica::new(2, true).expect("a laid worker");
        let adapter = replica.adapter();
        assert_eq!(
            adapter.probe_outcome(&replica.id, false, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stood(ObjectStatus::Pending)
        );
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Ready),
            ProbeAnswer::Stood(ObjectStatus::Pending)
        );
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Moved {
                old: ObjectStatus::Pending,
                new: ObjectStatus::Ready,
            }
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
    }

    #[test]
    fn an_object_whose_status_lagged_is_set_to_the_stores() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let row = replica.store.by_key(URL).expect("a row");
        assert!(
            StoreHealthSink::connected(replica.store.as_ref(), row.id, row.revision)
                .moved()
                .is_some()
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Pending);
        let mut events = replica.objects.registry().subscribe_events();

        assert_eq!(
            replica.adapter().probe_outcome(
                &replica.id,
                true,
                replica.revision,
                ObjectStatus::Pending
            ),
            ProbeAnswer::Stood(ObjectStatus::Ready)
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
        let WorkerEvent::StatusChanged {
            worker_id,
            old_status,
            new_status,
            ..
        } = events.try_recv().expect("a status change")
        else {
            panic!("expected a status change");
        };
        assert_eq!(
            (worker_id, old_status, new_status),
            (replica.id, ObjectStatus::Pending, ObjectStatus::Ready)
        );
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn an_outcome_at_a_revision_the_object_left_is_stale() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        let row = replica.store.by_key(URL).expect("a row");
        assert!(replica
            .store
            .patch(
                row.id,
                WorkerPatch {
                    priority: Some(1),
                    ..WorkerPatch::default()
                },
                row.revision,
            )
            .is_some());
        let later = replica.store.by_key(URL).expect("a row");
        assert_ne!(later.revision, row.revision);

        // The object still names the row's earlier revision.
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);

        replica.objects.follow(replica.store.as_ref());
        let object = replica
            .objects
            .registry()
            .get(&replica.id)
            .expect("a worker");
        assert_ne!(object.revision(), replica.revision);
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }

    #[test]
    fn an_outcome_of_an_object_the_holder_did_not_lay_is_stale() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let row = replica.store.by_key(URL).expect("a row");
        let foreign = object(&row);
        let registry = replica.objects.registry();
        assert!(registry.replace(&replica.id, Arc::clone(&foreign)));
        let revision = foreign.revision();
        let adapter = replica.adapter();
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }

    #[test]
    fn a_row_that_went_answers_stale() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let row = replica.store.by_key(URL).expect("a row");
        assert!(replica.store.remove(row.id, row.revision).is_some());
        let adapter = replica.adapter();
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(None, ObjectStatus::Pending);
    }

    #[test]
    fn a_connect_signal_readies_a_pending_row_and_its_object() {
        let replica = Replica::new(2, false).expect("a laid worker");
        assert_eq!(
            replica.adapter().connected(&replica.id, replica.revision),
            Some((ObjectStatus::Pending, ObjectStatus::Ready))
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
    }

    #[test]
    fn a_connect_signal_on_a_ready_row_moves_nothing() {
        let replica = Replica::new(2, false).expect("a laid worker");
        let row = replica.store.by_key(URL).expect("a row");
        assert_eq!(
            StoreHealthSink::connected(replica.store.as_ref(), row.id, row.revision),
            HealthWrite::Moved {
                from: RowStatus::Pending,
                to: RowStatus::Ready,
            }
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Pending);
        assert_eq!(
            replica.adapter().connected(&replica.id, replica.revision),
            None
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
    }

    #[test]
    fn a_replica_that_follows_writes_nothing() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            Arc::new(TestLead(Mutex::new(None))),
            SoleLead::TERM,
        );
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }

    #[test]
    fn a_call_whose_term_ended_while_it_waited_for_the_object_writes_nothing() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let under_term_one = || {
            HealthAdapter::new(
                Arc::clone(&replica.objects),
                Arc::clone(&replica.store),
                Arc::new(TurningLead(Mutex::new(0))),
                1,
            )
        };
        assert_eq!(
            under_term_one().probe_outcome(
                &replica.id,
                true,
                replica.revision,
                ObjectStatus::Pending
            ),
            ProbeAnswer::Stale
        );
        assert_eq!(
            under_term_one().connected(&replica.id, replica.revision),
            None
        );
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }

    #[test]
    fn an_adapter_of_an_earlier_term_writes_nothing() {
        let replica = Replica::new(1, true).expect("a laid worker");
        replica.store.set_term(1);
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            lead.clone(),
            1,
        );
        replica.store.set_term(2);
        *lead.0.lock() = Some(2);
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }
}
