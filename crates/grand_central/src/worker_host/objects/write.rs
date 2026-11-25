//! The holder's write side: a leader's write of a row is decided over the
//! worker objects, and the store follows it.
//!
//! A door hands the holder the write it would have asked of the store's
//! table: a registration's facts, a patch, or a removal. The holder takes
//! its turn and decides over the row read back
//! from the object that stands ([`Objects::row_of`]), by the rule every
//! writer of a row decides by. It builds the object of the row the change
//! leaves, tells the store the decided change
//! ([`MemoryStore::follow_objects`]), and puts the object in the registry.
//! So the object stands when the write returns, with no follow.
//! No row of the store's table is read at the head of a write: every
//! writer of a row holds the turn, so the object states the row.
//!
//! The store is told between the build and the registry's write: its write
//! files the member's count slot, which a lease books the moment a decision
//! picks the worker off the registry. A removal goes the other way round:
//! the object leaves the registry, then the store is told. A build cannot
//! fail, so the store is never told a change whose object was not built.
//!
//! A write the store does not take answers `None`: another writer moved
//! the row under the holder's turn. The store tests that under its own
//! lock, by the revision the holder read: a door and a standby's applier
//! write in the holder's turn, and a caller of the store's own writes
//! takes none.
//! The holder then reads the row that stands under the write's key in the
//! store, and that row alone, before it answers: the registry holds what
//! that writer left under the key. A removal has taken its object out by
//! then, and the row that writer left gets an object anew.
//! A change the store took and the registry did not get, because the
//! writer's thread ended between the two, is laid by that read where the
//! store refuses the key's next registration, or by a host's follow, which
//! settles each row the store's tail names against the object that stands
//! for it.
//!
//! A patch tells the store the whole row read back from the object, where
//! the store's own patch wrote the patched facts alone. The two are one
//! row while an object states what its row stated: what an object does not
//! hold as stated is kept beside it, and nothing in a controller sets a
//! fact of an object behind its build, as a gateway's refresh of a
//! worker's models does in a gateway. A change that sets one writes the
//! table at the row's next patch.
//!
//! A status is the object's word ([`mod@super::status`]). A row changed in
//! place keeps the status its table holds, and the holder tells the store
//! the status of the object laid in the standing one's place. A key's next
//! life starts in the status the rule hands on from the object that stood,
//! and its object is built in it.
//! The holder's place stays where a follow or a cut laid whole left it: the
//! next follow reads the holder's own change in the tail and finds it laid.

use std::time::Instant;

use smg_worker::registry::Generation;

use super::Objects;
use crate::{
    engine::view::WorkerId,
    store::{
        MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow,
        WorkersStore,
    },
};

impl<G: Generation> Objects<G> {
    /// Lay a registration's `facts`, by `maker`, over the object that
    /// stands under their key, by the one rule every writer of a row
    /// decides by. The change it made, or `None` where `store` did not take
    /// it: another writer moved the row under the call.
    ///
    /// A row changed in place gets a new object in the standing one's
    /// place, under its id in the registry and with its running state. A
    /// key's next life registers its object under a new id, and the old
    /// life's object is dropped first.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn upsert(
        &self,
        store: &MemoryStore,
        facts: WorkerFacts,
        maker: Registrant,
    ) -> Option<MembershipChange> {
        let _turn = self.read_to.lock();
        let standing = self.row_under(&facts.key);
        let seen = standing.as_ref().map(|row| row.revision);
        let (change, row) = facts.laid_over(standing.as_ref(), maker, Instant::now());
        self.lay_decided(store, change, &row, seen)
    }

    /// Write `patch` over the row `id` where its object was built at
    /// `revision`. The change it made, or `None` where the row moved or
    /// left, or `store` did not take the write.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn patch(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        let _turn = self.read_to.lock();
        let mut row = self.guarded(id, revision)?;
        patch.laid_on(&mut row.facts);
        row.revision += 1;
        let change = MembershipChange::Relabelled { id };
        self.lay_decided(store, change, &row, Some(revision))
    }

    /// Remove the row `id` where its object was built at `revision`. The
    /// object leaves the registry, then `store` is told. The change it
    /// made, or `None` where the row moved or left, or `store` did not take
    /// the write.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn remove(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        revision: u64,
    ) -> Option<MembershipChange> {
        let _turn = self.read_to.lock();
        let row = self.guarded(id, revision)?;
        self.take_decided(store, &row)
    }

    /// The row `id` as its object and what stands beside it state it,
    /// where the object was built at `revision`.
    fn guarded(&self, id: WorkerId, revision: u64) -> Option<WorkerRow> {
        let row = self.row(id)?;
        (row.revision == revision).then_some(row)
    }

    /// The decided `change` and the `row` it leaves, over a row read back
    /// at the revision `seen` in the holder's turn: the row's object is
    /// built, the store is told, and the object is put in the registry. A
    /// change the store does not take builds an object nobody sees
    /// ([`Self::moved`]).
    fn lay_decided(
        &self,
        store: &MemoryStore,
        change: MembershipChange,
        row: &WorkerRow,
        seen: Option<u64>,
    ) -> Option<MembershipChange> {
        let built = self.build(row);
        if !store.follow_objects(change, Some(row.clone()), seen) {
            return self.moved(store, &row.facts.key, change);
        }
        self.put(store, row, built);
        // The object's status leads. One laid in another's place took the
        // standing object's there, and the table is told it. A new object
        // is built in the status its row starts in.
        if matches!(
            change,
            MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }
        ) {
            self.tell_row(store, row.id, row.revision);
        }
        Some(change)
    }

    /// The decided removal of `row`, read back in the holder's turn: its
    /// object leaves the registry, then the store is told. Where the store
    /// does not take it, the row that stands gets an object anew
    /// ([`Self::moved`]): a new object under a new id in the registry. The
    /// leases the store counts on the row are stated on it, and what the
    /// old object alone held is not.
    fn take_decided(&self, store: &MemoryStore, row: &WorkerRow) -> Option<MembershipChange> {
        self.end(row.id);
        let change = MembershipChange::Removed { id: row.id };
        if !store.follow_objects(change, None, Some(row.revision)) {
            return self.moved(store, &row.facts.key, change);
        }
        Some(change)
    }

    /// The store did not take a write of the row under `key`: a writer that
    /// took no turn moved the row, or the registry never got a change the
    /// store took. The write's own record is read, and no other: the row
    /// that stands under the key in the store. A patch and a removal name
    /// their row by its id, and a key holds one row: the read finds the
    /// row under that id where it stands still, and the key's next life
    /// where it left. The row is laid unless its object was built from it,
    /// and an object of another life of the key gives way to it
    /// ([`Self::put`]). A row that left and came back under its id is
    /// another life too, made by another or admitted at another time: its
    /// old object is ended first, as where the tail says the row left, so
    /// the new one takes none of its running state. Where no row stands,
    /// the key's object goes. So the
    /// registry holds what stands under the key when the write answers. No
    /// other read of the store's rows is left in a write, and none of its
    /// tail.
    ///
    /// `change` is the change the store refused. The keeper forgets the
    /// row it was decided over, where the read found that row gone, and
    /// asks about no other: the read saw one record.
    fn moved(
        &self,
        store: &MemoryStore,
        key: &str,
        change: MembershipChange,
    ) -> Option<MembershipChange> {
        let read = self.keeper_clock();
        match store.by_key(key) {
            Some(row) => {
                // Another life under the id the object stands for.
                if self.remade(&row) {
                    self.end(row.id);
                }
                if !self.stands(&row) {
                    self.lay(store, &row);
                }
            }
            None => {
                if let Some(stood) = self.row_under(key) {
                    self.end(stood.id);
                }
            }
        }
        let over = match change {
            MembershipChange::Replaced { old, .. } => Some(old),
            MembershipChange::Relabelled { id }
            | MembershipChange::Resized { id }
            | MembershipChange::Removed { id } => Some(id),
            MembershipChange::Registered { .. } | MembershipChange::StatusChanged { .. } => None,
        };
        if let Some(over) = over {
            self.forget_one(read, over);
        }
        None
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use parking_lot::Mutex;
    use smg_worker::registry::{WorkerId as ObjectId, WorkerRegistry};

    use super::*;
    use crate::{
        engine::view::Role,
        store::{identity, Journal, Position, WorkerStatus, WorkersStore},
        worker_host::Members,
    };

    pub(crate) const A: &str = "http://a:8000";

    /// What a journal was told of each membership write: the change, the
    /// row it left and the row it took out.
    pub(crate) type Member = (MembershipChange, Option<WorkerRow>, Option<WorkerRow>);

    /// A journal that keeps what it was told of the membership.
    #[derive(Default)]
    struct Told(Mutex<Vec<Member>>);

    impl Journal for Told {
        fn member(
            &self,
            _: Option<Position>,
            _: u64,
            change: MembershipChange,
            row: Option<&WorkerRow>,
            left: Option<&WorkerRow>,
        ) {
            self.0.lock().push((change, row.cloned(), left.cloned()));
        }

        fn generation(&self, _: Option<Position>, _: u64) {}

        fn term_rose(&self, _: u64) {}

        fn term_followed(&self, _: Position) {}

        fn head(&self) -> Position {
            Position::default()
        }

        fn followed(&self) -> Option<u64> {
            None
        }

        fn restart(&self, _: Position) -> bool {
            true
        }
    }

    /// A replica as a controller keeps one: a journaled store, and a holder
    /// whose objects lead their rows' status, so it follows none.
    pub(crate) struct Replica {
        pub(crate) store: MemoryStore,
        pub(crate) objects: Objects<Members>,
        told: Arc<Told>,
    }

    impl Replica {
        pub(crate) fn new() -> Self {
            let told = Arc::new(Told::default());
            Self {
                store: MemoryStore::journaled(Arc::clone(&told) as _),
                objects: Objects::new(Arc::new(WorkerRegistry::new())),
                told,
            }
        }

        pub(crate) fn upsert(&self, facts: WorkerFacts) -> Option<MembershipChange> {
            self.objects.upsert(&self.store, facts, Registrant::Control)
        }

        /// The last membership write as the journal was told it.
        pub(crate) fn last(&self) -> Member {
            self.told
                .0
                .lock()
                .last()
                .cloned()
                .expect("a write was told")
        }

        pub(crate) fn object(&self, url: &str) -> Option<ObjectId> {
            self.objects.registry().get_id_by_url(url)
        }

        /// The row under `url` where the journal's last word, the object
        /// and the table agree on it, each fact and the status. The journal
        /// is told the row the write handed, which the object read back
        /// states, and the table takes it. No follow is run: the holder's
        /// place stands behind the store.
        pub(crate) fn one_row(&self, url: &str) -> WorkerRow {
            let (_, told, _) = self.last();
            let table = self.store.by_key(url).expect("the table holds the row");
            let object = self.object(url).expect("the object stands with no follow");
            let read = self.objects.row_of(&object).expect("the object's row");
            assert_eq!(told.as_ref(), Some(&read), "the journal against the object");
            assert_eq!(table, read, "the table against the object");
            let end = Some(self.store.epoch());
            assert_ne!(self.objects.read_to(), end, "no follow laid it");
            table
        }
    }

    /// A registration that states what an object holds and what stands
    /// beside it.
    pub(crate) fn facts(incarnation: u64, tier: &str) -> WorkerFacts {
        WorkerFacts {
            key: A.to_owned(),
            engine: A.to_owned(),
            incarnation,
            models: vec!["m".to_owned(), "n".to_owned()],
            dp_size: 1,
            labels: BTreeMap::from([("tier".to_owned(), tier.to_owned())]),
            zone: Some("z".to_owned()),
            vision: true,
            max_running: Some(8),
            priority: -3,
            ..WorkerFacts::default()
        }
    }

    /// Each write of a row is decided over the objects: the object stands
    /// when the write returns, with no follow, and the journal, the table
    /// and the object state one row. A row changed in place keeps its
    /// object's id in the registry, a patch moves the facts it names alone,
    /// a key's next life takes another id, and a removal takes the object
    /// out ahead of the store's word.
    #[test]
    fn a_write_of_a_row_leaves_its_object_standing_and_one_row_in_three_places() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        // A new key.
        let first = identity(A, 1);
        let change = replica.upsert(facts(1, "gold"));
        assert_eq!(change, Some(MembershipChange::Registered { id: first }));
        let row = replica.one_row(A);
        let registered = (first, 1, WorkerStatus::Pending, &facts(1, "gold"));
        assert_eq!((row.id, row.revision, row.status, &row.facts), registered);
        let object = replica.object(A);
        // The row's own incarnation: in place, under the object's id.
        let change = replica.upsert(facts(1, "iron"));
        assert_eq!(change, Some(MembershipChange::Relabelled { id: first }));
        let relabelled = replica.one_row(A);
        assert_eq!(
            (relabelled.revision, &relabelled.facts),
            (2, &facts(1, "iron"))
        );
        assert_eq!(replica.object(A), object, "the same place in the registry");
        // A patch: the fact it names, and no other.
        let patch = WorkerPatch {
            priority: Some(7),
            ..WorkerPatch::default()
        };
        let change = objects.patch(store, first, patch, 2);
        assert_eq!(change, Some(MembershipChange::Relabelled { id: first }));
        let patched = replica.one_row(A);
        let moved = WorkerFacts {
            priority: 7,
            ..facts(1, "iron")
        };
        assert_eq!((patched.revision, &patched.facts), (3, &moved));
        assert_eq!(patched.admitted_at, row.admitted_at, "the row's own life");
        // Another incarnation: the key's next life, under another id.
        let second = identity(A, 2);
        let change = replica.upsert(facts(2, "iron"));
        let replaced = MembershipChange::Replaced {
            old: first,
            new: second,
        };
        assert_eq!(change, Some(replaced));
        let next = replica.one_row(A);
        assert_eq!((next.id, next.revision), (second, 1));
        assert_eq!(
            replica.last().2,
            Some(patched),
            "the row the write took out"
        );
        assert_ne!(replica.object(A), object, "another object, another id");
        assert_eq!(objects.len(), 1, "the old life's object left");
        // A removal: the object is out when the write returns.
        let change = objects.remove(store, second, 1);
        assert_eq!(change, Some(MembershipChange::Removed { id: second }));
        assert_eq!((replica.object(A), store.by_key(A)), (None, None));
        let (told, left, took) = replica.last();
        assert_eq!(told, MembershipChange::Removed { id: second });
        assert_eq!((left, took), (None, Some(next)));
    }

    /// The status of a row a write leaves is its object's word. A key's
    /// next life takes the status its ready worker stood in, a draining
    /// worker's starts pending, and so does another worker's. The journal,
    /// the table and the object say the same status, with no follow and
    /// behind one.
    #[test]
    fn a_keys_next_life_is_handed_the_status_its_worker_stood_in() {
        type State = fn(&mut WorkerFacts);
        let lives: [(WorkerStatus, State, WorkerStatus); 3] = [
            (WorkerStatus::Ready, |_| {}, WorkerStatus::Ready),
            (WorkerStatus::Draining, |_| {}, WorkerStatus::Pending),
            (
                WorkerStatus::Ready,
                |facts| facts.role = Role::Decode,
                WorkerStatus::Pending,
            ),
        ];
        for (stood, state, handed) in lives {
            let replica = Replica::new();
            let (store, objects) = (&replica.store, &replica.objects);
            let id = replica.upsert(facts(1, "gold")).expect("registered").id();
            // The status moves by the holder's own entries: a connect
            // signal, a drain. The object leads, and the table is told.
            objects.connect_row(store, id, 1);
            if stood == WorkerStatus::Draining {
                assert!(objects.drain_row(store, id, 1), "the row drains");
            }
            let mut next = facts(2, "gold");
            state(&mut next);
            replica.upsert(next).expect("the next life");
            let row = replica.one_row(A);
            assert_eq!(row.status, handed, "behind a row that stood {stood:?}");
            // Behind a follow the three still say one row.
            objects.follow(store);
            let object = replica.object(A).expect("the object");
            assert_eq!(replica.objects.row_of(&object), Some(row));
        }
    }

    /// A status moves by its own event alone: a row changed in place keeps
    /// the status the table holds, whatever its decided row says. A key's
    /// next life starts in the status its decided row states: the one the
    /// holder's rule handed on from the object.
    #[test]
    fn a_row_in_place_keeps_the_tables_status_and_a_next_life_takes_its_rows() {
        let replica = Replica::new();
        let store = &replica.store;
        let id = replica.upsert(facts(1, "gold")).expect("registered").id();
        replica.objects.connect_row(store, id, 1);
        // In place, decided over a row that still reads pending.
        let stale = store.by_key(A).map(|row| WorkerRow {
            status: WorkerStatus::Pending,
            revision: 2,
            ..row
        });
        let relabelled = MembershipChange::Relabelled { id };
        assert!(store.follow_objects(relabelled, stale, Some(1)));
        let status = |url: &str| store.by_key(url).map(|row| row.status);
        assert_eq!(status(A), Some(WorkerStatus::Ready));
        // The next life, decided over that same reading.
        let standing = store.by_key(A).map(|row| WorkerRow {
            status: WorkerStatus::Pending,
            ..row
        });
        let now = Instant::now();
        let (change, next) =
            facts(2, "gold").laid_over(standing.as_ref(), Registrant::Control, now);
        assert_eq!(next.status, WorkerStatus::Pending, "as the holder read it");
        assert!(store.follow_objects(change, Some(next), Some(2)));
        assert_eq!(status(A), Some(WorkerStatus::Pending));
        assert_eq!(
            replica.last().1.map(|row| row.status),
            Some(WorkerStatus::Pending)
        );
    }

    /// A patch and a removal are guarded by the revision the object was
    /// built at: under another, and under an id no object stands for, they
    /// write nothing and tell nothing.
    #[test]
    fn a_patch_or_a_removal_under_another_revision_writes_nothing() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        let id = replica.upsert(facts(1, "gold")).expect("registered").id();
        let before = (store.epoch(), replica.told.0.lock().len(), store.by_key(A));
        let patch = || WorkerPatch {
            priority: Some(7),
            ..WorkerPatch::default()
        };
        for (id, revision) in [(id, 2), (id, 0), (identity(A, 9), 1)] {
            assert_eq!(objects.patch(store, id, patch(), revision), None);
            assert_eq!(objects.remove(store, id, revision), None);
        }
        let after = (store.epoch(), replica.told.0.lock().len(), store.by_key(A));
        assert_eq!(after, before);
        assert!(replica.object(A).is_some(), "the object stands");
    }

    /// A change the store took and the registry did not get is laid by the
    /// next follow: the store's tail names the row, and the follow settles
    /// it against what stands. So is a removal the store took while the
    /// object still stood.
    #[test]
    fn a_change_the_store_took_and_the_registry_lacks_is_laid_by_the_next_follow() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        let now = Instant::now();
        let (change, row) = facts(1, "gold").laid_over(None, Registrant::Control, now);
        assert!(store.follow_objects(change, Some(row.clone()), None));
        assert_eq!(replica.object(A), None, "the store alone was told");
        objects.follow(store);
        let object = replica.object(A).expect("the follow laid it");
        assert_eq!(objects.row_of(&object), Some(row));
        let removed = MembershipChange::Removed { id: change.id() };
        assert!(store.follow_objects(removed, None, Some(1)));
        assert!(replica.object(A).is_some(), "the store alone was told");
        objects.follow(store);
        assert_eq!(replica.object(A), None);
    }

    /// No write call moves the holder's place, so a read of the store's
    /// tail reads the holder's own writes too. A row the holder removed and
    /// registered again under its incarnation stands under its old id, and
    /// the read keeps that life's object: it was built from the row that
    /// stands. So it is where a key took another incarnation and then the
    /// first one again. A row that left and came back by the store's own
    /// entries, with the facts it had, is another life admitted at another
    /// time: its object is built anew, as before.
    #[test]
    fn a_read_of_the_tail_keeps_the_object_of_a_row_the_holder_laid_again() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        // The place stands at the empty store's end: the next read is the
        // tail, with each change the holder writes from here.
        objects.follow(store);
        let id = replica.upsert(facts(1, "gold")).expect("registered").id();
        assert!(objects.remove(store, id, 1).is_some(), "the row left");
        let again = replica.upsert(facts(1, "gold"));
        assert_eq!(again, Some(MembershipChange::Registered { id }));
        let object = replica.object(A);
        assert!(object.is_some(), "the next life's object");
        assert_eq!(objects.read_to(), Some(0), "no write moved the place");
        objects.follow(store);
        assert_eq!(replica.object(A), object, "behind its own removal");
        // Another incarnation, then the first one again: its old id.
        replica.upsert(facts(2, "gold")).expect("the next life");
        let back = replica.upsert(facts(1, "gold"));
        let replaced = MembershipChange::Replaced {
            old: identity(A, 2),
            new: id,
        };
        assert_eq!(back, Some(replaced));
        let object = replica.object(A);
        objects.follow(store);
        assert_eq!(replica.object(A), object, "behind its own replacement");
        assert_eq!(objects.len(), 1);
        let read = object.as_ref().and_then(|object| objects.row_of(object));
        assert_eq!(read, store.by_key(A), "the object against the table");
        // Around the holder: the same facts, the same id, the first
        // revision again, and another time of admission.
        assert!(store.remove(id, 1).is_some(), "the store's own removal");
        let around = store.upsert_by(facts(1, "gold"), Registrant::Control);
        assert_eq!(around, MembershipChange::Registered { id });
        objects.follow(store);
        assert!(replica.object(A).is_some(), "the row's object");
        assert_ne!(replica.object(A), object, "another life, built anew");
    }

    /// A change the store took and the registry did not get is laid where
    /// the store next refuses a write of the row. A patch and a removal by
    /// the row's id find no object to guard them: each answers `None`,
    /// reads nothing and lays nothing. A registration under its key is
    /// decided over no object: the store refuses it, the object stands when
    /// it answers `None`, and the same call again is decided over it.
    #[test]
    fn a_row_with_no_object_is_laid_where_its_registration_is_refused() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        let now = Instant::now();
        let (change, row) = facts(1, "gold").laid_over(None, Registrant::Control, now);
        assert!(store.follow_objects(change, Some(row.clone()), None));
        let id = change.id();
        let patch = WorkerPatch {
            priority: Some(7),
            ..WorkerPatch::default()
        };
        assert_eq!(objects.patch(store, id, patch, 1), None);
        assert_eq!(objects.remove(store, id, 1), None);
        assert_eq!(replica.object(A), None, "neither read the store");
        let told = replica.told.0.lock().len();
        assert_eq!(replica.upsert(facts(1, "iron")), None, "refused once");
        let object = replica.object(A).expect("the refusal's read laid it");
        assert_eq!(objects.row_of(&object), Some(row));
        assert_eq!(replica.told.0.lock().len(), told, "nothing told");
        let again = replica.upsert(facts(1, "iron"));
        assert_eq!(again, Some(MembershipChange::Relabelled { id }));
        assert_eq!(replica.one_row(A).facts, facts(1, "iron"));
    }

    /// A write decided over a row another writer moved since is refused
    /// under the store's lock, by the revision the holder read: a
    /// registration in place, a patch, a key's next life and a removal.
    /// Each answers `None`, which a door answers as a row that moved, and
    /// tells the journal nothing. The registry holds what the other writer
    /// left under the key when the write answers, with no follow: the row
    /// whose object a removal took out has one anew, and where that writer
    /// left the key's next life, the old life's object is gone.
    #[test]
    fn a_write_over_a_row_another_writer_moved_is_refused_and_reads_it() {
        type Write = fn(&Replica, WorkerRow) -> Option<MembershipChange>;
        /// Another writer, outside the holder's turn: a write of the
        /// store's own, which takes no turn of the holder.
        type Other = fn(&MemoryStore, &WorkerRow);
        /// A registration of `next` decided over the row `read`.
        fn lay(replica: &Replica, next: WorkerFacts, read: &WorkerRow) -> Option<MembershipChange> {
            let now = Instant::now();
            let (change, row) = next.laid_over(Some(read), Registrant::Control, now);
            let seen = Some(read.revision);
            replica
                .objects
                .lay_decided(&replica.store, change, &row, seen)
        }
        let writes: [(&str, Write); 4] = [
            ("a registration in place", |replica, read| {
                lay(replica, facts(1, "iron"), &read)
            }),
            ("a patch", |replica, mut read| {
                let seen = Some(read.revision);
                read.facts.priority = 7;
                read.revision += 1;
                let change = MembershipChange::Relabelled { id: read.id };
                replica
                    .objects
                    .lay_decided(&replica.store, change, &read, seen)
            }),
            ("a key's next life", |replica, read| {
                lay(replica, facts(2, "gold"), &read)
            }),
            ("a removal", |replica, read| {
                replica.objects.take_decided(&replica.store, &read)
            }),
        ];
        let others: [(&str, Other); 2] = [
            ("moved in place", |store, read| {
                let late = WorkerPatch {
                    labels: Some(BTreeMap::from([("tier".to_owned(), "late".to_owned())])),
                    ..WorkerPatch::default()
                };
                let moved = store.patch(read.id, late, read.revision);
                assert!(moved.is_some(), "the late write");
            }),
            ("the key's next life", |store, _| {
                let next = store.upsert_by(facts(3, "late"), Registrant::Control);
                assert!(matches!(next, MembershipChange::Replaced { .. }));
            }),
        ];
        for (what, write) in writes {
            for (how, other) in others {
                let replica = Replica::new();
                let (store, objects) = (&replica.store, &replica.objects);
                replica.upsert(facts(1, "gold")).expect("registered");
                // The holder's read, in its turn: the row off the object.
                let turn = objects.read_to.lock();
                let read = replica.object(A).and_then(|object| objects.row_of(&object));
                let read = read.expect("the row read");
                other(store, &read);
                let told = replica.told.0.lock().len();
                let left = store.by_key(A).expect("the row that writer left");
                assert_eq!(write(&replica, read), None, "{what}, {how}");
                drop(turn);
                let tells = replica.told.0.lock().len();
                assert_eq!(tells, told, "{what}, {how}: nothing told");
                let table = store.by_key(A);
                assert_eq!(table.as_ref(), Some(&left), "{what}, {how}: the table");
                let object = replica.object(A).expect("an object stands");
                let laid = objects.row_of(&object);
                assert_eq!(laid, Some(left), "{what}, {how}: the registry");
                assert_eq!(objects.len(), 1, "{what}, {how}: the key's one object");
            }
        }
    }

    /// A row that left and came back under its id around the holder is
    /// another life, admitted at another time. A write decided over the old
    /// life's object is refused, and its read ends that object before it
    /// lays the row. So the new life's object takes another id in the
    /// registry and none of the old one's running state, here a drain, as
    /// where the tail says the row left. A later read of the tail keeps it.
    #[test]
    fn a_refused_write_ends_the_object_of_a_life_that_left() {
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        objects.follow(store);
        let id = replica.upsert(facts(1, "gold")).expect("registered").id();
        replica.upsert(facts(1, "iron")).expect("relabelled");
        assert!(objects.drain_row(store, id, 2), "the old life drains");
        let old = replica.object(A).expect("the old life's object");
        // Around the holder: the row leaves and comes back under its id,
        // at the first revision.
        assert!(store.remove(id, 2).is_some(), "the store's own removal");
        let back = store.upsert_by(facts(1, "iron"), Registrant::Control);
        assert_eq!(back, MembershipChange::Registered { id });
        let patch = WorkerPatch {
            priority: Some(7),
            ..WorkerPatch::default()
        };
        assert_eq!(objects.patch(store, id, patch, 2), None, "the old life's");
        let new = replica.object(A).expect("the new life's object");
        assert_ne!(new, old, "another object under another id");
        let row = objects.row_of(&new).expect("its row");
        let started = (row.revision, row.status);
        assert_eq!(started, (1, WorkerStatus::Pending), "no drain taken over");
        assert_eq!(Some(row), store.by_key(A), "the object against the table");
        objects.follow(store);
        assert_eq!(replica.object(A), Some(new), "the tail's read keeps it");
        assert_eq!(objects.len(), 1);
    }

    /// A refused write reads the row that stands under its own key in the
    /// store, and no other. A row of another key that the store took around
    /// the holder has no object behind that read: it gets one where its own
    /// key's registration is refused, or at its host's follow. A
    /// registration over a row that left the store around the holder is
    /// refused too, and the key's object goes: the same call again
    /// registers the key anew.
    #[test]
    fn a_refused_write_reads_the_row_under_its_own_key_alone() {
        const B: &str = "http://b:8000";
        const C: &str = "http://c:8000";
        let replica = Replica::new();
        let (store, objects) = (&replica.store, &replica.objects);
        let keyed = |key: &str| WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            ..facts(1, "gold")
        };
        let now = Instant::now();
        for key in [A, B, C] {
            let (change, row) = keyed(key).laid_over(None, Registrant::Control, now);
            assert!(store.follow_objects(change, Some(row), None));
        }
        assert_eq!(replica.upsert(facts(1, "iron")), None, "refused once");
        assert!(replica.object(A).is_some(), "the row under its key is laid");
        assert_eq!(objects.len(), 1, "no other row is read");
        let refused = objects.upsert(store, keyed(B), Registrant::Control);
        assert_eq!(refused, None, "the other key's own registration");
        assert!(replica.object(B).is_some(), "laid by its own refusal");
        assert_eq!(replica.object(C), None, "a third key's row waits");
        objects.follow(store);
        assert!(replica.object(C).is_some(), "laid by its host's follow");
        assert_eq!(objects.len(), 3);
        // The row under B leaves the store around the holder.
        let left = store.by_key(B).expect("the row");
        assert!(store.remove(left.id, left.revision).is_some());
        let again = WorkerFacts {
            priority: 7,
            ..keyed(B)
        };
        let refused = objects.upsert(store, again.clone(), Registrant::Control);
        assert_eq!(refused, None, "decided over a row that left");
        assert_eq!(replica.object(B), None, "the key's object goes");
        let change = objects.upsert(store, again, Registrant::Control);
        let registered = MembershipChange::Registered { id: left.id };
        assert_eq!(change, Some(registered), "the key registers anew");
        assert!(replica.object(B).is_some());
    }

    /// A bare address and the same address with its scheme are two keys, as
    /// in the store's table: each registers its own row and its own object,
    /// in either order. The registry also finds a bare address under the
    /// form with its scheme, and a write does not read that object as the
    /// row that stands.
    #[test]
    fn a_bare_address_and_its_scheme_form_are_two_rows_in_either_order() {
        const BARE: &str = "a:8000";
        let keyed = |key: &str| WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            ..facts(1, "gold")
        };
        for order in [[A, BARE], [BARE, A]] {
            let replica = Replica::new();
            let (store, objects) = (&replica.store, &replica.objects);
            for key in order {
                let id = identity(key, 1);
                let change = replica.upsert(keyed(key));
                let registered = MembershipChange::Registered { id };
                assert_eq!(change, Some(registered), "{order:?}");
            }
            for key in order {
                let row = store.by_key(key).expect("the key's own row");
                let object = objects.object(row.id).expect("the row's own object");
                assert_eq!(objects.row_of(&object), Some(row), "{order:?}");
            }
            assert_eq!(objects.len(), 2, "{order:?}");
        }
    }
}
