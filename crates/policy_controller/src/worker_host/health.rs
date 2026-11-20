//! The worker health sink over the store.
//!
//! The adapter reads the lead before it takes the object's mutation lock and
//! again under it. With the lock held, it reads the holder's map, releases
//! the read lock, and writes the store. The store asks the lead once more
//! behind its own locks, as an admitted apply asks its own: a term that
//! ended while a write waited there writes nothing, a verdict's and a
//! projection's alike. The holder writes its map and the registry under
//! separate locks. Its status mirror reads the row under the same mutation
//! lock as the adapter and writes through the registry as the adapter does.
//!
//! The store guards a write by the row's id and revision, and a row's id is
//! its key and incarnation. A URL's next life under another incarnation is
//! another row: an outcome of the earlier life's object finds no row under
//! the id it names and writes nothing. So the guard tells lives apart where
//! the host numbers each life apart, as a registration door does with
//! [`life`](crate::store::membership::life).
//!
//! The projection ([`HealthAdapter::project`]) writes what the object knows
//! of its worker onto the row, under the same lock and the same fences. It
//! reads the object at its write and is handed no reading, so of two
//! projections the later one's reading stands on the row. A projection
//! moves no status: the object's follows its row through the holder's
//! mirror and through a verdict's write. The worker crate's collector
//! drives it: the adapter answers the port's tell of a liveness move with
//! the projection, so no host calls it.

use std::sync::Arc;

use smg_worker::{
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{object::status, Lead, Objects, Standing};
use crate::store::{HealthSink as StoreHealthSink, HealthWrite, LeaderBits};

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

    /// Write what the object under `worker_id` knows of its worker onto
    /// the row it stands for ([`StoreHealthSink::leader_bits`]): its
    /// liveness veto ([`Worker::stall_reason`]) is the row's stalled bit.
    /// The fences are a health write's: the term, the object the holder
    /// laid, and the revision that object was built from. The term has the
    /// last word: the store asks it once more behind its own locks. The
    /// object is read under its mutation lock, at the write. The
    /// collector's tell calls it ([`HealthSink::liveness_moved`]): at each
    /// move of the object's veto, for an object laid over another, which
    /// takes the running state of the object it replaces, its veto with
    /// the rest, and for every object where a collector starts.
    ///
    /// A projection moves no status, of the row or of the object. Its
    /// answer says the status that stands on the row
    /// ([`HealthWrite::Stood`]), and the object keeps its own.
    ///
    /// Stale where the adapter's term ended, where no object stands under
    /// the id, where the holder laid another, and where the row left the
    /// revision the object was built from. No row where the row left while
    /// its object still stands: removed, or replaced by its key's next
    /// incarnation.
    pub fn project(&self, worker_id: &ObjectId) -> HealthWrite {
        // The revision of the object that stands now: the guard under the
        // lock refuses the call where another took its place since.
        let Some(object) = self.objects.registry().get(worker_id) else {
            return HealthWrite::Stale;
        };
        self.project_at(worker_id, object.revision())
    }

    /// [`Self::project`] for the object at `revision`: the collector's
    /// tell names the revision it read on the object.
    fn project_at(&self, worker_id: &ObjectId, revision: u64) -> HealthWrite {
        self.fenced(worker_id, revision, |store, standing, object| {
            let bits = LeaderBits {
                stalled: object.stall_reason().is_some(),
            };
            let write = store.leader_bits(standing.row, standing.revision, bits, &|| self.leads());
            // No status goes back to the registry: a verdict's write moves
            // the object, and this is none.
            (write, None)
        })
    }

    fn leads(&self) -> bool {
        self.lead.term() == Some(self.term)
    }

    /// A verdict's write, under the fences. The status that stands on the
    /// row goes back to the registry, which sets a lagging object to it.
    fn write(
        &self,
        worker_id: &ObjectId,
        revision: u64,
        write_row: impl FnOnce(&S, Standing) -> HealthWrite,
    ) -> HealthWrite {
        self.fenced(worker_id, revision, |store, standing, _| {
            let write = write_row(store, standing);
            (write, write.stands().map(status))
        })
    }

    /// Runs `write_row` under the fences of a health write: the lead, read
    /// before the object's mutation lock and again under it, the object
    /// the holder laid, and the revision that object was built from. The
    /// closure answers its write, and the status the registry sets the
    /// object to where it sets one.
    fn fenced(
        &self,
        worker_id: &ObjectId,
        revision: u64,
        write_row: impl FnOnce(&S, Standing, &dyn Worker) -> (HealthWrite, Option<ObjectStatus>),
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
            write_row(self.store.as_ref(), standing, object.as_ref())
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

    /// The projection ([`HealthAdapter::project`]), at the revision the
    /// collector read on the object: the leader's doubt of the worker.
    /// The collector says it at each move of the object's veto, where the
    /// holder lays an object in another's place, and for every object at
    /// its start, so a host that runs a collector for each term it leads
    /// calls no projection of its own.
    fn liveness_moved(&self, worker_id: &ObjectId, revision: u64) {
        self.project_at(worker_id, revision);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;
    use smg_worker::{
        health::manager::{WorkerManager, WorkerManagerConfig},
        registry::{event::WorkerEvent, WorkerRegistry},
        runtime::StallReason,
        settings::UnhealthyRemovalScope,
    };
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;
    use crate::{
        store::{
            membership::life, Ask, GatewayId, HealthPolicy, MemoryStore, ViewBuffers, WorkerFacts,
            WorkerPatch, WorkerStatus as RowStatus, WorkersStore,
        },
        view::Role,
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

        /// Whether a decision reads the worker as doubted: the stalled bit
        /// of its row in the model's pool, as any gateway is handed it.
        fn doubted(&self) -> bool {
            let ask = Ask {
                gateway: GatewayId(1),
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let mut out = ViewBuffers::default();
            // A registration is in force at its window's publication: the
            // case asks for one.
            self.store.publish_now();
            let built = self.store.build_view(MODEL, Role::Single, &ask, &mut out);
            built.is_some() && out.candidates.iter().any(|row| row.health.stalled)
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
    fn a_projection_writes_the_objects_veto_onto_its_row_and_clears_it() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        let stood = HealthWrite::Stood(RowStatus::Pending);
        // No veto stands on the object: its projection writes no doubt.
        assert_eq!(adapter.project(&replica.id), stood);
        assert!(!replica.doubted());
        // The veto alone moves no row. The projection writes it, and moves
        // no status of the row or of the object.
        assert!(replica.object().set_stall(Some(StallReason::Unreachable)));
        assert!(!replica.doubted());
        assert_eq!(adapter.project(&replica.id), stood);
        assert!(replica.doubted());
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
        // The object is read at the write: the veto it dropped is gone
        // from the row at the next projection.
        assert!(replica.object().set_stall(None));
        assert_eq!(adapter.project(&replica.id), stood);
        assert!(!replica.doubted());
    }

    #[test]
    fn a_projection_the_row_outran_is_stale_and_the_object_laid_next_carries_the_veto() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        let row = replica.store.by_key(URL).expect("a row");
        let patch = WorkerPatch {
            priority: Some(1),
            ..WorkerPatch::default()
        };
        assert!(replica.store.patch(row.id, patch, row.revision).is_some());
        assert!(replica.object().set_stall(Some(StallReason::Wedged)));
        // The object still names the row's earlier revision.
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
        assert!(!replica.doubted());
        // The holder lays the row's next revision in the object's place.
        // The replacement takes its running state, the veto with the rest,
        // and its projection writes the doubt.
        replica.objects.follow(replica.store.as_ref());
        assert_ne!(replica.object().revision(), replica.revision);
        assert_eq!(
            adapter.project(&replica.id),
            HealthWrite::Stood(RowStatus::Pending)
        );
        assert!(replica.doubted());
    }

    /// The worker crate's collector over the holder's registry, with an
    /// adapter of the replica as its sink. It probes nothing: the row's
    /// policy turns its probes off.
    fn collector(replica: &Replica) -> WorkerManager {
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            Arc::new(SoleLead),
            SoleLead::TERM,
        );
        WorkerManager::start_with_sink(
            Arc::clone(replica.objects.registry()),
            Arc::new(adapter),
            WorkerManagerConfig {
                default_check_interval_secs: 3_600,
                remove_unhealthy: UnhealthyRemovalScope::Disabled,
            },
            None,
        )
    }

    /// Whether `holds` comes true within five seconds.
    async fn comes(holds: impl Fn() -> bool) -> bool {
        for _ in 0..500 {
            if holds() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        holds()
    }

    /// The collector drives the projection, and no host calls it. A move
    /// of the object's veto reaches the row through the object's signal,
    /// the collector and the adapter, and a decision's view reads it.
    #[tokio::test]
    async fn the_collectors_tell_writes_the_leaders_doubt_and_clears_it() {
        let replica = Replica::new(1, false).expect("a laid worker");
        let _collector = collector(&replica);
        assert!(!replica.doubted());

        assert!(replica.object().set_stall(Some(StallReason::Unreachable)));
        assert!(comes(|| replica.doubted()).await, "the doubt is written");
        assert!(replica.object().set_stall(None));
        assert!(comes(|| !replica.doubted()).await, "and cleared");
    }

    /// A collector tells its sink of every worker at its start, so the
    /// first loop of a term writes what the objects know with no call of
    /// the host: a veto that moved while no loop ran, and the clearing of
    /// a doubt the row kept from the loop before.
    #[tokio::test]
    async fn a_collector_started_behind_a_move_writes_what_the_objects_know() {
        let replica = Replica::new(1, false).expect("a laid worker");
        assert!(replica.object().set_stall(Some(StallReason::Wedged)));
        assert!(!replica.doubted());
        let first = collector(&replica);
        assert!(comes(|| replica.doubted()).await, "the start writes it");

        drop(first);
        assert!(replica.object().set_stall(None));
        let _second = collector(&replica);
        assert!(comes(|| !replica.doubted()).await, "the next clears it");
    }

    /// A tell the row outran is dropped: the object still names the row's
    /// earlier revision. The holder lays the row's next revision in the
    /// object's place, the collector tells its sink of the replacement,
    /// and the doubt is written with no call of the host.
    #[tokio::test]
    async fn a_replacement_laid_behind_a_dropped_tell_carries_the_doubt_to_its_row() {
        let replica = Replica::new(1, false).expect("a laid worker");
        let _collector = collector(&replica);
        let row = replica.store.by_key(URL).expect("a row");
        let patch = WorkerPatch {
            priority: Some(1),
            ..WorkerPatch::default()
        };
        assert!(replica.store.patch(row.id, patch, row.revision).is_some());
        assert!(replica.object().set_stall(Some(StallReason::Wedged)));
        // The loop takes the signal and its tell is dropped.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!replica.doubted());

        replica.objects.follow(replica.store.as_ref());
        assert!(comes(|| replica.doubted()).await, "the lay writes it");
    }

    /// The store has the term's last word. A projection that passed the
    /// adapter's two readings and waited at the store's locks through its
    /// term's end writes nothing there.
    #[test]
    fn a_projection_whose_term_ended_at_the_stores_locks_writes_nothing() {
        let replica = Replica::new(1, true).expect("a laid worker");
        assert!(replica.object().set_stall(Some(StallReason::Wedged)));
        let adapter = HealthAdapter::new(
            Arc::clone(&replica.objects),
            Arc::clone(&replica.store),
            TurningLead::after(2),
            1,
        );
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
        assert!(!replica.doubted());
        // Under a term that stands the same projection writes the doubt.
        assert_eq!(
            replica.adapter().project(&replica.id),
            HealthWrite::Stood(RowStatus::Pending)
        );
        assert!(replica.doubted());
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

    /// A projection moves no status. Its answer says the row's, and an
    /// object that lags its row keeps its own: the holder's mirror and a
    /// verdict's write move it, never a projection.
    #[test]
    fn a_projection_moves_no_status_of_its_object() {
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
            replica.adapter().project(&replica.id),
            HealthWrite::Stood(RowStatus::Ready)
        );
        replica.assert_statuses(Some(RowStatus::Ready), ObjectStatus::Pending);
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    /// The object is read under its lock, at the write. A projection that
    /// waited for the lock while the veto rose and another projection
    /// wrote it writes the veto too: no reading from before its wait.
    #[test]
    fn a_projection_that_waited_for_the_lock_writes_the_reading_under_it() {
        let replica = Replica::new(1, true).expect("a laid worker");
        let adapter = replica.adapter();
        let stood = HealthWrite::Stood(RowStatus::Pending);
        let waited = replica
            .objects
            .registry()
            .with_another_write_before_the_lock(
                |_| adapter.project(&replica.id),
                |_| {
                    assert!(replica.object().set_stall(Some(StallReason::Unreachable)));
                    assert_eq!(adapter.project(&replica.id), stood);
                    assert!(replica.doubted());
                },
            );
        assert_eq!(waited, stood);
        assert!(replica.doubted());
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
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
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
        // The projection is answered by the store while the object stands,
        // and finds no object once the holder followed the removal.
        assert_eq!(adapter.project(&replica.id), HealthWrite::NoRow);
        replica.objects.follow(replica.store.as_ref());
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
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
        assert_eq!(adapter.project(&replica.id), HealthWrite::NoRow);
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
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
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
        assert_eq!(under_term_one().project(&replica.id), HealthWrite::Stale);
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
        assert_eq!(adapter.project(&replica.id), HealthWrite::Stale);
        replica.assert_statuses(Some(RowStatus::Pending), ObjectStatus::Pending);
    }
}
