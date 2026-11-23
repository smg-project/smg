//! The worker health sink over the store.
//!
//! The adapter reads the lead before it takes the object's mutation lock and
//! again under it. With the lock held, it reads the holder's map, releases
//! the read lock, and writes the store. The store asks the lead once more
//! behind its own locks, as an admitted apply asks its own: a term that
//! ended while a write waited there writes nothing. The holder writes its
//! map and the registry under separate locks. Its status mirror reads the
//! row under the same mutation lock as the adapter and writes through the
//! registry as the adapter does.
//!
//! The store guards a write by the row's id and revision, and a row's id is
//! its key and incarnation. A URL's next life under another incarnation is
//! another row: an outcome of the earlier life's object finds no row under
//! the id it names and writes nothing. So the guard tells lives apart where
//! the host numbers each life apart, as a registration door does with
//! [`life`](crate::store::membership::life).

use std::sync::Arc;

use smg_worker::{
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{object::status, Lead, Objects, RowRevision};
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
    /// before taking the object's lock and again under the lock, and the
    /// store asks it once more behind its own.
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

    /// A verdict's write, run under the fences of a health write: the lead,
    /// read before the object's mutation lock and again under it, the
    /// object the holder laid, and the revision that object was built from.
    /// The status that stands on the row goes back to the registry, which
    /// sets a lagging object to it.
    fn write(
        &self,
        worker_id: &ObjectId,
        revision: u64,
        write_row: impl FnOnce(&S, RowRevision) -> HealthWrite,
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
            store.probe_outcome(standing.row, passed, standing.revision, &|| self.leads())
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
            store.connected(standing.row, standing.revision, &|| self.leads())
        })
        .moved()
        .map(|(from, to)| (status(from), status(to)))
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use smg_worker::{
        registry::{event::WorkerEvent, WorkerRegistry},
        runtime::StallReason,
    };
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;
    use crate::{
        store::{
            membership::life, HealthPolicy, MemoryStore, WorkerFacts, WorkerPatch,
            WorkerStatus as RowStatus, WorkersStore,
        },
        worker_host::{object, Members, SoleLead},
    };

    const URL: &str = "http://a:8000";
    const MODEL: &str = "m";

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
                    models: vec![MODEL.to_string()],
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

        fn object(&self) -> Arc<dyn Worker> {
            let registry = self.objects.registry();
            registry.get(&self.id).expect("a worker")
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

    /// A lead whose term ends behind a count of its readings. The adapter
    /// reads it before the object's lock and again under it.
    struct TurningLead {
        readings: Mutex<u32>,
        leads_for: u32,
    }

    impl TurningLead {
        fn after(leads_for: u32) -> Arc<Self> {
            Arc::new(Self {
                readings: Mutex::new(0),
                leads_for,
            })
        }
    }

    impl Lead for TurningLead {
        fn term(&self) -> Option<u64> {
            let mut readings = self.readings.lock();
            *readings += 1;
            Some(if *readings <= self.leads_for { 1 } else { 2 })
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
            StoreHealthSink::connected(replica.store.as_ref(), row.id, row.revision, &|| true)
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
    fn an_object_laid_for_a_rows_next_revision_carries_the_veto() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let row = replica.store.by_key(URL).expect("a row");
        let patch = WorkerPatch {
            priority: Some(1),
            ..WorkerPatch::default()
        };
        assert!(replica.store.patch(row.id, patch, row.revision).is_some());
        assert!(replica.object().set_stall(Some(StallReason::Wedged)));
        // The holder lays the row's next revision in the object's place.
        // The replacement takes its running state, the veto with the rest.
        replica.objects.follow(replica.store.as_ref());
        assert_ne!(replica.object().revision(), replica.revision);
        assert_eq!(replica.object().stall_reason(), Some(StallReason::Wedged));
    }

    /// The store has the term's last word at a verdict too. A probe's
    /// outcome and a connect signal that passed the adapter's two readings
    /// and waited at the store's lock through their term's end write
    /// nothing there: no status moves, no probe is counted and the journal
    /// is told nothing.
    #[test]
    fn a_verdict_whose_term_ended_at_the_stores_lock_writes_nothing() {
        let replica = Replica::new(2, true).expect("a laid worker");
        let turning = || {
            HealthAdapter::new(
                Arc::clone(&replica.objects),
                Arc::clone(&replica.store),
                TurningLead::after(2),
                1,
            )
        };
        let epoch = replica.store.epoch();
        assert_eq!(
            turning().probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(turning().connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        assert_eq!(
            replica.store.epoch(),
            epoch,
            "a refused verdict moved the store"
        );
        // The refused pass was not counted: under a term that stands the
        // row still owes the two passes its policy asks.
        let adapter = replica.adapter();
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stood(ObjectStatus::Pending)
        );
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

    /// A URL's next life, numbered as a registration door numbers one, is
    /// another row under another id. An outcome of the earlier life's
    /// object, which the holder has not ended yet, finds no row under the id
    /// it names and writes nothing on the new one.
    #[test]
    fn an_outcome_of_a_urls_earlier_life_writes_nothing_on_its_next() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        let earlier = replica.store.by_key(URL).expect("a row");
        assert!(replica.store.remove(earlier.id, earlier.revision).is_some());
        let next = replica
            .store
            .upsert(WorkerFacts {
                incarnation: life(1, 0, replica.store.epoch()),
                ..earlier.facts.clone()
            })
            .id();
        assert_ne!(next, earlier.id, "another life takes another id");

        // The holder has not followed: its object is the earlier life's.
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
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
            StoreHealthSink::connected(replica.store.as_ref(), row.id, row.revision, &|| true),
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
                TurningLead::after(1),
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
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            lead.clone(),
            1,
        );
        *lead.0.lock() = Some(2);
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }
}
