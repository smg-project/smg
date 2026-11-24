//! The worker health sink of a replica that keeps its workers in the store:
//! a worker object's status leads, and the store is told it.
//!
//! A probe's outcome goes to the registry's own state machine, which counts
//! on the object and moves its status under the object's mutation lock. The
//! adapter then tells the store the status that stands
//! ([`Objects::tell_status`]). A connect signal is the holder's
//! ([`Objects::connect`]): ready at once, unless the object drains or is
//! ready. No status is decided over the store's row.
//!
//! The adapter reads the lead before it hands a verdict on. The tell reads
//! it again under the object's mutation lock, and the store asks it once
//! more behind its own lock, as an admitted apply asks its own: a term that
//! ended while a write waited there writes nothing.
//!
//! A move the store takes no word of is taken back by the tell itself: in
//! the turn of the object's lock where the store refused it, the object
//! takes the status its row holds, and the outcome answers as a stale
//! one. The row's status, not the one the probe saw the object leave: a
//! new leader's told status can have landed since the move. Where no row
//! stands for the object, it takes back the status it left. So a verdict
//! moves the object and its row, or neither. The registry's
//! state machine moves the object in its own turn of the lock, so between
//! that turn and the tell a reader of the object can see the status that
//! is taken back, and the registry's subscribers hear both moves.
//!
//! The store guards a write by the row's id and revision, and a row's id is
//! its key and incarnation. A URL's next life under another incarnation is
//! another row: a status of the earlier life's object finds no row under
//! the id it names and writes nothing. So the guard tells lives apart where
//! the host numbers each life apart, as a registration door does with
//! [`life`](crate::store::membership::life).

use std::sync::Arc;

use smg_worker::{
    health::port::{HealthSink, ProbeAnswer},
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{Lead, Objects};
use crate::store::MemoryStore;

/// The health sink for a replica that keeps its workers in the store.
pub struct HealthAdapter<G> {
    objects: Arc<Objects<G>>,
    store: Arc<MemoryStore>,
    lead: Arc<dyn Lead>,
    term: u64,
}

impl<G: Generation> HealthAdapter<G> {
    /// An adapter for `term`. Each call checks that `lead` still names it
    /// before it hands a verdict on. A status is told under the object's
    /// lock where `lead` names the term still, and the store asks it once
    /// more behind its own.
    pub fn new(
        objects: Arc<Objects<G>>,
        store: Arc<MemoryStore>,
        lead: Arc<dyn Lead>,
        term: u64,
    ) -> Self {
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
}

impl<G: Generation> HealthSink for HealthAdapter<G> {
    /// The registry's state machine judges the probe over the object, and
    /// a move is told to the store.
    fn probe_outcome(
        &self,
        worker_id: &ObjectId,
        passed: bool,
        revision: u64,
        launched: ObjectStatus,
    ) -> ProbeAnswer {
        if !self.leads() {
            return ProbeAnswer::Stale;
        }
        let registry = self.objects.registry().as_ref();
        let answer = HealthSink::probe_outcome(registry, worker_id, passed, revision, launched);
        let ProbeAnswer::Moved { old, new } = answer else {
            return answer;
        };
        let leads = || self.leads();
        let told = self
            .objects
            .tell_status(&self.store, worker_id, revision, (old, new), &leads);
        if told.stands().is_some() {
            return answer;
        }
        // The store took no word of the move: its term ended, the row
        // left the revision the object was built from, or the holder
        // laid no such object. The tell took the move back, to the
        // status the object's row holds or, where none stands, to the
        // one it left. The outcome is stale.
        ProbeAnswer::Stale
    }

    fn connected(
        &self,
        worker_id: &ObjectId,
        revision: u64,
    ) -> Option<(ObjectStatus, ObjectStatus)> {
        if !self.leads() {
            return None;
        }
        let leads = || self.leads();
        self.objects
            .connect(&self.store, worker_id, revision, &leads)
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use smg_worker::{
        identity::basic::Worker,
        registry::{event::WorkerEvent, WorkerRegistry},
        runtime::StallReason,
    };
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;
    use crate::{
        store::{
            membership::life, HealthPolicy, WorkerFacts, WorkerPatch, WorkerStatus as RowStatus,
            WorkersStore,
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

        fn adapter(&self) -> HealthAdapter<Members> {
            HealthAdapter::new(
                Arc::clone(&self.objects),
                Arc::clone(&self.store),
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
    /// reads it before it hands a verdict on. A told status reads it under
    /// the object's lock, and the store reads it under its own.
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
    fn a_probe_that_moves_the_object_answers_moved_and_tells_the_row() {
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

    /// A probe's move is one status event of the registry and one told
    /// status of the store: the registry's state machine moves the object,
    /// and the row is told the status that stands.
    #[test]
    fn a_probes_move_is_one_status_event_and_one_told_status() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let mut events = replica.objects.registry().subscribe_events();
        let epoch = replica.store.epoch();
        assert_eq!(
            replica.adapter().probe_outcome(
                &replica.id,
                true,
                replica.revision,
                ObjectStatus::Pending
            ),
            ProbeAnswer::Moved {
                old: ObjectStatus::Pending,
                new: ObjectStatus::Ready,
            }
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
        assert_eq!(replica.store.epoch(), epoch + 1, "one told status");
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

    /// The store has the term's last word at a told status. A probe's move
    /// and a connect signal pass the adapter's reading and the one under
    /// the object's lock, and wait at the store's lock through their
    /// term's end. They write nothing there: no status moves, the journal
    /// is told nothing, and the object takes the status its row holds.
    #[test]
    fn a_status_whose_term_ended_at_the_stores_lock_writes_nothing() {
        let replica = Replica::new(2, true).expect("a laid worker");
        let turning = || {
            HealthAdapter::new(
                Arc::clone(&replica.objects),
                Arc::clone(&replica.store),
                TurningLead::after(2),
                1,
            )
        };
        let pass = |adapter: &HealthAdapter<Members>| {
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending)
        };
        // One pass of the two the policy asks, under a term that stands.
        let standing = replica.adapter();
        assert_eq!(pass(&standing), ProbeAnswer::Stood(ObjectStatus::Pending));
        let epoch = replica.store.epoch();
        // The second moves the object, and its term ends at the store's
        // lock.
        assert_eq!(pass(&turning()), ProbeAnswer::Stale);
        assert_eq!(turning().connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        assert_eq!(
            replica.store.epoch(),
            epoch,
            "a refused status moved the store"
        );
        // The move that was taken back owes its two passes again.
        assert_eq!(pass(&standing), ProbeAnswer::Stood(ObjectStatus::Pending));
        assert_eq!(
            pass(&standing),
            ProbeAnswer::Moved {
                old: ObjectStatus::Pending,
                new: ObjectStatus::Ready,
            }
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
    }

    /// A lead that names its term at two readings: the adapter's, and the
    /// one under the object's lock. Behind the second it runs `then`, and
    /// its term is over.
    struct LeadUntil<F>(Mutex<u32>, F);

    impl<F: Fn() + Send + Sync + 'static> Lead for LeadUntil<F> {
        fn term(&self) -> Option<u64> {
            let mut readings = self.0.lock();
            *readings += 1;
            match *readings {
                1 => Some(1),
                2 => {
                    (self.1)();
                    Some(1)
                }
                _ => Some(2),
            }
        }
    }

    /// A status that waits at the store's lock while a restore lays
    /// another leader's snapshot is not written onto the restored table.
    /// The row stands there under the same id and revision, so the
    /// revision's guard takes the status: the term's word refuses it. The
    /// object then takes the status the restored row holds, the new
    /// leader's word, not the one the probe saw it leave.
    #[test]
    fn a_status_that_waited_through_a_restore_is_not_written_on_the_restored_table() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let stated = replica.store.by_key(URL).expect("a row");
        // Another leader's store: the same row, not ready there.
        let other = MemoryStore::new();
        let row = other.upsert(stated.facts).id();
        assert_eq!(row, stated.id, "the same row");
        other
            .set_status(row, RowStatus::NotReady, 1)
            .expect("a status");
        let store = Arc::clone(&replica.store);
        let restore = move || assert!(store.restore_from(other.checkpoint()));
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            Arc::new(LeadUntil(Mutex::new(0), restore)),
            1,
        );
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        let restored = replica.store.by_key(URL).expect("the restored row");
        assert_eq!(
            (restored.id, restored.revision, restored.status),
            (row, 1, RowStatus::NotReady)
        );
        replica.assert_statuses(Some(RowStatus::NotReady), ObjectStatus::NotReady);
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
    fn a_connect_signal_on_a_ready_object_moves_nothing() {
        let replica = Replica::new(2, false).expect("a laid worker");
        let adapter = replica.adapter();
        assert!(adapter.connected(&replica.id, replica.revision).is_some());
        let epoch = replica.store.epoch();
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Ready);
        assert_eq!(replica.store.epoch(), epoch, "no status is told");
    }

    /// A replica that follows hands no probe to the registry's state
    /// machine: the object does not move and come back, so the registry's
    /// subscribers hear nothing.
    #[test]
    fn a_replica_that_follows_writes_nothing() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            Arc::new(TestLead(Mutex::new(None))),
            SoleLead::TERM,
        );
        let mut events = replica.objects.registry().subscribe_events();
        assert_eq!(
            adapter.probe_outcome(&replica.id, true, replica.revision, ObjectStatus::Pending),
            ProbeAnswer::Stale
        );
        assert_eq!(adapter.connected(&replica.id, replica.revision), None);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
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
