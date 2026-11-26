//! The holder's status side: a worker object's status leads its row's.
//!
//! On a replica that leads, a status is written on the worker object and
//! the store is told it. A probe's verdict moves the object by the
//! registry's own state machine, and the health adapter then tells the
//! store the status that stands ([`Objects::tell_status`]). A connect
//! signal and a drain are decided here, over the object
//! ([`Objects::connect`], [`Objects::drain_row`]): the store is told
//! first, and the registry moves the object where the store took the
//! status.
//!
//! **The lock order.** Every entry here runs under the object's mutation
//! lock, the registry's lock of one worker, and takes the store's
//! membership lock inside it: the object's lock, then the store's. No
//! holder of the store's lock takes an object's. So the read of an
//! object's status and its telling are one critical section. Of two tells
//! the later read stands, and the table ends where the object does. No
//! decision takes an object's mutation lock. Nor does a first copy's cut
//! ([`mod@super::read`]): it holds the holder's turn and, inside it, the
//! store's lock. A holder's write of a row holds that turn, then an
//! object's lock, then the store's: one order for the three.
//!
//! A tell the store refuses writes nothing, and the call does not try it
//! again: the row left the revision the object was built from, no row
//! stands, or the writer's term ended. The row's next lay or the object's
//! next move settles it. A status the object took ahead of a tell the
//! store takes no word of is taken back in the tell's own turn: the
//! object takes the status its row holds, and where no row stands for it
//! the status it left.
//!
//! From the table to an object there are two ways, both of a told status.
//! A standby's applier and a gateway's watcher lay the status their
//! leader's log tells ([`mod@super::standby`]). And a tell the store
//! refuses gives the object the status its row holds, as said above.

use std::sync::Arc;

use smg_worker::{
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{super::object::row_status, Objects};
use crate::{
    engine::view::WorkerId,
    store::{HealthWrite, MemoryStore},
};

/// An object's move: the status it left, then the one that stands.
type Moved = Option<(ObjectStatus, ObjectStatus)>;

/// What a status write does to its object, read under the object's lock.
#[derive(Clone, Copy)]
enum Write {
    /// Nothing: the status that stands is told. With it, the move the
    /// telling follows where a driver made one ahead of it: the status
    /// the object left, then the one it took.
    Tell(Moved),
    /// Ready at once, with its probes counted anew. A ready or a draining
    /// object stands, and its counters with it.
    Connect,
    /// Draining. A draining object stands.
    Drain,
}

impl Write {
    /// The status an object that stands in `from` moves to, and none where
    /// it stands.
    fn next(self, from: ObjectStatus) -> Option<ObjectStatus> {
        let to = match self {
            Self::Tell(_) => return None,
            Self::Connect if from == ObjectStatus::Draining => return None,
            Self::Connect => ObjectStatus::Ready,
            Self::Drain => ObjectStatus::Draining,
        };
        (to != from).then_some(to)
    }
}

impl<G: Generation> Objects<G> {
    /// Tell `store` the status that stands on the object under `id`, where
    /// the object stands at `revision`, its own. A health driver calls it
    /// behind a move the registry's state machine made, `moved`: the
    /// status the object left, then the one it took. `leads` is the
    /// driver's last word, asked under the object's lock and again under
    /// the store's. Where the store takes no word, the move is taken back
    /// in this turn ([`Self::status_under`]). What the store made of the
    /// status.
    pub(crate) fn tell_status(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        moved: (ObjectStatus, ObjectStatus),
        leads: &dyn Fn() -> bool,
    ) -> HealthWrite {
        let write = Write::Tell(Some(moved));
        self.status_under(store, id, revision, None, leads, write).0
    }

    /// The connect signal of the object under `id`, as
    /// [`Self::tell_status`] names one: ready at once, with its probes
    /// counted anew. A ready or a draining object stands, and its counters
    /// with it. The object's move, where it made one.
    pub(crate) fn connect(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        leads: &dyn Fn() -> bool,
    ) -> Moved {
        self.status_under(store, id, revision, None, leads, Write::Connect)
            .1
    }

    /// The connect signal of the row `row`, read at `revision`: its object
    /// is ready at once, with its probes counted anew, and `store` is told.
    /// A ready or a draining object stands, and its counters with it. A row
    /// that left the revision, or left, moves nothing. One of a host's
    /// write calls ([`Objects`]): its caller asks its term itself, so the
    /// call hands the store no last word.
    pub fn connect_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) {
        self.row_status(store, row, revision, &|| true, Write::Connect);
    }

    /// The drain of the row `row`, read at `revision`, as
    /// [`Self::connect_row`] writes a status: its object drains, and
    /// `store` is told. Whether the row's object moved: a row that left
    /// the revision, or left, moves nothing, and a draining object stands.
    pub fn drain_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) -> bool {
        self.row_status(store, row, revision, &|| true, Write::Drain)
            .is_some()
    }

    /// Tell `store` the status of the object that stands for `row`, behind
    /// a write that laid the object in another's place: it took the
    /// standing object's status there, and the table holds its own.
    pub(super) fn tell_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) {
        self.row_status(store, row, revision, &|| true, Write::Tell(None));
    }

    /// Settle the status of the object that stands for `row` with the
    /// table's, behind a lay outside a write call that put the object in
    /// another's place: it took the standing object's status there, and a
    /// move a health driver made on that one ahead of its telling tells
    /// nothing ([`Self::status_under`]). `leads` is the caller's own word
    /// of its lead. Where it holds, the table is told the object's status,
    /// as [`Self::tell_row`] tells it. Where it does not, the table is not
    /// written, and the object takes the status its row holds.
    pub(super) fn settle_row(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        leads: &dyn Fn() -> bool,
    ) {
        self.row_status(store, row, revision, leads, Write::Tell(None));
    }

    /// A status write that names its row: on the object that stands for
    /// `row`, where it was built from `revision`, under its writer's word
    /// of its lead.
    fn row_status(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        leads: &dyn Fn() -> bool,
        write: Write,
    ) -> Moved {
        let id = self.object(row)?;
        let object = self.registry.get(&id)?;
        self.status_under(store, &id, object.revision(), Some(revision), leads, write)
            .1
    }

    /// One status write, under the object's mutation lock: the writer's
    /// word, the row the holder built this object from, the store's
    /// answer, and the object's move where the store took the status or
    /// holds it already. `seen` is the row's revision where the caller
    /// read one. What the store made of the status, and the object's move.
    ///
    /// A tell the store takes no word of takes the object's move back, in
    /// this same turn, whether the term ended at the turn's start or at
    /// the store's lock. The registry's state machine moved the object
    /// ahead of the telling. Where a row stands at the revision the object
    /// was built from, the object takes the status that row holds, by the
    /// read a told status is laid by ([`Self::table_status`]): a leader's
    /// told status can have landed since the move, so the table's word
    /// stands, never the status the driver saw the object leave. Where no
    /// such row stands (the row left the revision, the row left, or the
    /// holder laid no such object), the object takes back the status its
    /// driver saw it leave, where it stands in the moved one still. A told
    /// status is laid only on an object whose row stands at that revision,
    /// so none is written over there. And a move left on the object would
    /// outlive it: an object laid in its place takes the status it stands
    /// in, with no word to the table.
    ///
    /// A tell that finds the object replaced or gone takes nothing back:
    /// the registry answers none under the revision the driver names, and
    /// the object laid in its place took the status the standing one
    /// stood in, the move with it. What laid it settles the two. A door
    /// tells the store its object's status behind the put
    /// ([`Self::tell_row`]). A told row's put and a cut are followed by
    /// the table's status ([`mod@super::standby`]).
    fn status_under(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        seen: Option<u64>,
        leads: &dyn Fn() -> bool,
        write: Write,
    ) -> (HealthWrite, Moved) {
        let under = |object: &Arc<dyn Worker>| {
            let from = object.status();
            // What a driver's move goes back to where no row stands for
            // the object: the status it left, where the object stands in
            // the one it took still.
            let left = match write {
                Write::Tell(Some((left, took))) if took == from => Some(left),
                _ => None,
            };
            let standing = self.standing_of(id, object);
            let read = |revision: u64| seen.is_none_or(|seen| seen == revision);
            let Some(standing) = standing.filter(|standing| read(standing.revision)) else {
                return (HealthWrite::Stale, left);
            };
            let next = write.next(from);
            let status = row_status(next.unwrap_or(from));
            // A call can wait here for the object's lock through the end of
            // its term: it tells nothing.
            let told = if leads() {
                store.status_told(standing.row, status, standing.revision, leads)
            } else {
                HealthWrite::Stale
            };
            if told.stands().is_none() {
                // The store took no word. A status the object took ahead
                // of its telling goes back in this turn: to the row's,
                // and where no row stands to the one it left.
                let back = match write {
                    Write::Tell(_) => Self::table_status(store, standing).or(left),
                    Write::Connect | Write::Drain => None,
                };
                return (told, back);
            }
            if next.is_some() && matches!(write, Write::Connect) {
                if let Some(state) = object.runtime_state() {
                    state.consecutive_failures_reset();
                    state.consecutive_successes_reset();
                    state.total_pending_probes_reset();
                }
            }
            (told, next)
        };
        self.registry
            .apply_if_revision(id, revision, under)
            .unwrap_or((HealthWrite::Stale, None))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use smg_worker::{
        health::port::{HealthSink, ProbeAnswer},
        registry::WorkerRegistry,
    };

    use super::{
        super::{
            super::object::status,
            write::tests::{facts, A},
        },
        *,
    };
    use crate::{
        store::{
            identity, HealthPolicy, MembershipChange, MembershipTail, Registrant, WorkerFacts,
            WorkerStatus, WorkersStore,
        },
        worker_host::Members,
    };

    /// A leader's holder over a store: its objects lead their rows' status.
    /// One row, registered through the holder.
    struct Leader {
        store: MemoryStore,
        objects: Objects<Members>,
        row: WorkerId,
        id: ObjectId,
        revision: u64,
    }

    /// The registration of [`Leader`]'s row under `tier`, with the two
    /// thresholds of its checks.
    fn stated(incarnation: u64, tier: &str, failure: u32, success: u32) -> WorkerFacts {
        WorkerFacts {
            health: HealthPolicy {
                failure_threshold: failure,
                success_threshold: success,
                ..HealthPolicy::default()
            },
            ..facts(incarnation, tier)
        }
    }

    impl Leader {
        /// A row each probe moves.
        fn new() -> Self {
            Self::with(1, 1)
        }

        fn with(failure: u32, success: u32) -> Self {
            let store = MemoryStore::default();
            let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
            let facts = stated(1, "gold", failure, success);
            let change = objects.upsert(&store, facts, Registrant::Control);
            let row = change.expect("registered").id();
            let id = objects.object(row).expect("the object");
            let revision = objects.registry().get(&id).expect("it stands").revision();
            Self {
                store,
                objects,
                row,
                id,
                revision,
            }
        }

        /// One probe, judged by the registry's own state machine as the
        /// health adapter hands it on: the object alone moves.
        fn probe(&self, passed: bool) -> ProbeAnswer {
            let registry = self.objects.registry().as_ref();
            let launched = registry.get(&self.id).expect("it stands").status();
            HealthSink::probe_outcome(registry, &self.id, passed, self.revision, launched)
        }

        /// The tell behind a probe that moved the object: the table still
        /// holds the status the object left.
        fn tell(&self, leads: bool) -> HealthWrite {
            let table = self.store.by_key(A).expect("the row stands");
            let object = self.objects.registry().get(&self.id).expect("it stands");
            let moved = (status(table.status), object.status());
            self.objects
                .tell_status(&self.store, &self.id, self.revision, moved, &|| leads)
        }

        /// The status the table holds under the key, then its object's.
        fn statuses(&self) -> (Option<WorkerStatus>, Option<ObjectStatus>) {
            let table = self.store.by_key(A).map(|row| row.status);
            let object = self.objects.registry().get_by_url(A);
            (table, object.map(|object| object.status()))
        }
    }

    /// Stand the worker object of `row` in `to` and tell `store`, as a
    /// probe's verdict leaves the two: a case's own hand on a worker's
    /// health where it runs no probe.
    pub(crate) fn stand<G: Generation>(
        objects: &Objects<G>,
        store: &MemoryStore,
        row: WorkerId,
        to: WorkerStatus,
    ) {
        let id = objects.object(row).expect("the row's object");
        let registry = objects.registry();
        let revision = registry.get(&id).expect("it stands").revision();
        let moved = registry.apply_if_revision(&id, revision, |_| ((), Some(status(to))));
        let moved = moved.and_then(|((), moved)| moved);
        let moved = moved.unwrap_or_else(|| (status(to), status(to)));
        let told = objects.tell_status(store, &id, revision, moved, &|| true);
        assert!(told.stands().is_some(), "the store took the status");
    }

    /// The table and the object in one status.
    fn both(stands: WorkerStatus) -> (Option<WorkerStatus>, Option<ObjectStatus>) {
        (Some(stands), Some(status(stands)))
    }

    /// A tell under a word that says no writes nothing to the table, and
    /// the object takes the status its row holds again. One that finds the
    /// table at the object's status tells nothing.
    #[test]
    fn a_tell_writes_where_the_writer_leads_and_the_table_differs() {
        let leader = Leader::new();
        assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
        let epoch = leader.store.epoch();
        assert_eq!(leader.tell(false), HealthWrite::Stale);
        let back = both(WorkerStatus::Pending);
        assert_eq!((leader.statuses(), leader.store.epoch()), (back, epoch));
        assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
        assert!(leader.tell(true).moved().is_some());
        assert_eq!(leader.store.epoch(), epoch + 1);
        assert_eq!(leader.tell(true), HealthWrite::Stood(WorkerStatus::Ready));
        assert_eq!(leader.store.epoch(), epoch + 1, "a status that stands");
    }

    /// A door's connect signal and its drain are written on the object,
    /// and the store is told in the same turn of the object's lock: the
    /// two say one status when the write returns. A ready
    /// object stands at a connect signal, and so does a draining one. A
    /// drain moves any other.
    #[test]
    fn a_doors_connect_signal_and_its_drain_write_the_object_and_tell_the_store() {
        let leader = Leader::new();
        let (store, objects, row) = (&leader.store, &leader.objects, leader.row);
        objects.connect_row(store, row, 1);
        assert_eq!(leader.statuses(), both(WorkerStatus::Ready));
        let epoch = store.epoch();
        objects.connect_row(store, row, 1);
        assert_eq!(store.epoch(), epoch, "a ready object stands");
        assert!(objects.drain_row(store, row, 1));
        assert_eq!(leader.statuses(), both(WorkerStatus::Draining));
        let epoch = store.epoch();
        objects.connect_row(store, row, 1);
        assert!(
            !objects.drain_row(store, row, 1),
            "a draining object stands"
        );
        assert_eq!(leader.statuses(), both(WorkerStatus::Draining));
        assert_eq!(store.epoch(), epoch);
    }

    /// A door's status write names the revision it read the row at. Where
    /// the row left it, or no object stands for the row, nothing moves and
    /// the store is told nothing.
    #[test]
    fn a_status_write_at_a_revision_the_row_left_moves_nothing() {
        let leader = Leader::new();
        let (store, objects, row) = (&leader.store, &leader.objects, leader.row);
        let epoch = store.epoch();
        objects.connect_row(store, row, 2);
        assert!(!objects.drain_row(store, row, 2));
        let absent = identity(A, 9);
        objects.connect_row(store, absent, 1);
        assert!(!objects.drain_row(store, absent, 1));
        assert_eq!(leader.statuses(), both(WorkerStatus::Pending));
        assert_eq!(store.epoch(), epoch);
    }

    /// A connect signal counts the object's probes anew as it readies it:
    /// a failure counted before the signal is not counted behind it.
    #[test]
    fn a_connect_signal_counts_the_objects_probes_anew() {
        let leader = Leader::with(2, 2);
        let pending = ProbeAnswer::Stood(ObjectStatus::Pending);
        assert_eq!(leader.probe(false), pending);
        leader.objects.connect_row(&leader.store, leader.row, 1);
        let ready = ProbeAnswer::Stood(ObjectStatus::Ready);
        assert_eq!(leader.probe(false), ready, "one failure of two");
        assert!(matches!(leader.probe(false), ProbeAnswer::Moved { .. }));
    }

    /// A connect signal on an object that stands leaves its counters with
    /// it: a failure counted before the signal still counts behind it.
    #[test]
    fn a_connect_signal_on_an_object_that_stands_keeps_its_count() {
        let leader = Leader::with(2, 2);
        let (store, objects, row) = (&leader.store, &leader.objects, leader.row);
        objects.connect_row(store, row, 1);
        let ready = ProbeAnswer::Stood(ObjectStatus::Ready);
        assert_eq!(leader.probe(false), ready, "one failure of two");
        // The object stands ready: the signal moves nothing and counts on.
        objects.connect_row(store, row, 1);
        let second = leader.probe(false);
        assert!(matches!(second, ProbeAnswer::Moved { .. }), "{second:?}");
    }

    /// Where no row stands for it, a tell the store takes no word of takes
    /// its driver's move back only where the object stands in the moved
    /// status still. Here the object moved again ahead of the tell, and its
    /// row left: nothing is taken back, and the later move stands.
    #[test]
    fn a_refused_tell_behind_a_second_move_takes_nothing_back() {
        let leader = Leader::new();
        assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
        let first = (ObjectStatus::Pending, ObjectStatus::Ready);
        assert!(matches!(leader.probe(false), ProbeAnswer::Moved { .. }));
        // The store's own removal: the holder's object stands a while yet.
        WorkersStore::remove(&leader.store, leader.row, 1).expect("removed");
        let (store, objects) = (&leader.store, &leader.objects);
        let told = objects.tell_status(store, &leader.id, leader.revision, first, &|| true);
        assert_eq!(told, HealthWrite::NoRow);
        let stands = objects
            .registry()
            .get(&leader.id)
            .map(|object| object.status());
        assert_eq!(stands, Some(ObjectStatus::NotReady), "the later move");
    }

    /// A row changed in place keeps its object's status. The registry lays
    /// the new object in the standing one's place with its running state,
    /// and the holder tells the store that status behind the write, by an
    /// event of its own: here a move the table was not told.
    #[test]
    fn a_row_laid_in_place_tells_the_status_its_object_took() {
        let leader = Leader::new();
        assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
        let epoch = leader.store.epoch();
        let relabelled = stated(1, "iron", 1, 1);
        let change = leader
            .objects
            .upsert(&leader.store, relabelled, Registrant::Control);
        assert_eq!(
            change,
            Some(MembershipChange::Relabelled { id: leader.row })
        );
        assert_eq!(leader.statuses(), both(WorkerStatus::Ready));
        let MembershipTail::Changes(tail) = leader.store.changes_since(epoch) else {
            panic!("the store's tail reaches back to the write");
        };
        let told = MembershipChange::StatusChanged {
            id: leader.row,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready,
        };
        let tail: Vec<_> = tail.into_iter().map(|(_, change)| change).collect();
        assert_eq!(
            tail,
            [MembershipChange::Relabelled { id: leader.row }, told]
        );
    }

    /// A key's next life starts in the status its worker object stood in,
    /// by the rule every writer decides by, and the new object takes it:
    /// here a move the table was not told.
    #[test]
    fn a_keys_next_life_starts_in_the_status_its_object_stood_in() {
        let leader = Leader::new();
        assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
        let trails = (Some(WorkerStatus::Pending), Some(ObjectStatus::Ready));
        assert_eq!(leader.statuses(), trails);
        let next = stated(2, "gold", 1, 1);
        let change = leader
            .objects
            .upsert(&leader.store, next, Registrant::Control);
        let replaced = MembershipChange::Replaced {
            old: leader.row,
            new: identity(A, 2),
        };
        assert_eq!(change, Some(replaced));
        assert_eq!(leader.statuses(), both(WorkerStatus::Ready));
    }
}
