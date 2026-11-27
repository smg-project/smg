use std::cell::Cell;

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
    service::replication::{Event, Log},
    store::{
        identity, HealthPolicy, Journal, LeadWord, MembershipChange, Registrant, WorkerFacts,
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
        Self::over(MemoryStore::default(), failure, success)
    }

    /// The same over `store`: a case that reads the store's journal
    /// hands a journaled one.
    fn over(store: MemoryStore, failure: u32, success: u32) -> Self {
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

/// A writer's word of its lead that holds for `asks` asks, and says no
/// from the next one.
fn holds_for(asks: usize) -> impl Fn() -> bool {
    let left = Cell::new(asks);
    move || left.replace(left.get().saturating_sub(1)) > 0
}

/// A connect signal and a drain under their writer's word. A word that
/// says no under the object's lock, its first ask, writes no status,
/// moves no object and leaves the store's epoch. So does one that
/// turns under the store's lock, its second. A word that holds for
/// both writes as the two public entries do.
#[test]
fn a_connect_signal_and_a_drain_write_nothing_once_their_writers_word_turns() {
    let leader = Leader::new();
    let (store, objects, row) = (&leader.store, &leader.objects, leader.row);
    let epoch = store.epoch();
    for asks in [0, 1] {
        objects.connect_row_under(store, row, 1, LeadWord::of(&holds_for(asks)));
        let drained = objects.drain_row_under(store, row, 1, LeadWord::of(&holds_for(asks)));
        assert!(!drained, "a word that holds for {asks} asks");
        let stands = (leader.statuses(), store.epoch());
        assert_eq!(stands, (both(WorkerStatus::Pending), epoch), "{asks}");
    }
    objects.connect_row_under(store, row, 1, LeadWord::of(&holds_for(2)));
    assert_eq!(leader.statuses(), both(WorkerStatus::Ready));
    assert!(objects.drain_row_under(store, row, 1, LeadWord::of(&holds_for(2))));
    assert_eq!(leader.statuses(), both(WorkerStatus::Draining));
    assert_eq!(store.epoch(), epoch + 2);
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
    leader.store.remove(leader.row, 1).expect("removed");
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
    let log = Arc::new(Log::new(16));
    let journal: Arc<dyn Journal> = Arc::<Log>::clone(&log);
    let leader = Leader::over(MemoryStore::journaled(journal), 1, 1);
    assert!(matches!(leader.probe(true), ProbeAnswer::Moved { .. }));
    let mark = log.head();
    let relabelled = stated(1, "iron", 1, 1);
    let change = leader
        .objects
        .upsert(&leader.store, relabelled, Registrant::Control);
    assert_eq!(
        change,
        Some(MembershipChange::Relabelled { id: leader.row })
    );
    assert_eq!(leader.statuses(), both(WorkerStatus::Ready));
    let mut tail = Vec::new();
    log.read(mark, Log::BATCH, &mut tail)
        .expect("the journal's log holds the mark");
    let told = MembershipChange::StatusChanged {
        id: leader.row,
        from: WorkerStatus::Pending,
        to: WorkerStatus::Ready,
    };
    let tail: Vec<_> = tail
        .into_iter()
        .filter_map(|entry| match entry.event {
            Event::Member { change, .. } => Some(change),
            Event::Status { id, from, to, .. } => {
                Some(MembershipChange::StatusChanged { id, from, to })
            }
            Event::Term | Event::Generation { .. } => None,
        })
        .collect();
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
