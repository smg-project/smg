//! The holder's write side: a leader's write of a row is decided over the
//! worker objects, and the store follows it.
//!
//! A door hands the holder the write it would have asked of the store's
//! table: a registration's facts, a patch, or a removal. The holder takes
//! its turn, reads the store to its end, and decides over the row read back
//! from the object that stands ([`Objects::row_of`]), by the rule every
//! writer of a row decides by. It builds the object of the row the change
//! leaves, tells the store the decided change
//! ([`MemoryStore::follow_objects`]), and puts the object in the registry.
//! So the object stands when the write returns, with no follow.
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
//! The holder then reads the store to its end again before it answers, so
//! the registry holds what that writer left. A removal has taken its
//! object out by then, and the row that writer left gets an object anew.
//! A change the store took and the registry did not get, because the
//! writer's thread ended between the two, is laid by the next follow: it
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
//! A status is the table's word still. The store hands the row a change
//! leaves the status its table holds, and the put copies it onto the object
//! where the holder follows statuses ([`Objects::with_status_followed`]).
//! The holder's place stays where its follow left it: the next follow reads
//! the holder's own change in the tail and finds it laid.

use std::time::Instant;

use smg_worker::registry::Generation;

use super::{Objects, Place};
use crate::{
    engine::view::WorkerId,
    store::{MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow},
};

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(dead_code, reason = "a controller's doors write through the holder")
)]
impl<G: Generation> Objects<G> {
    /// Lay a registration's `facts`, by `maker`, over the object that
    /// stands under their key: the write of [`WorkersStore::upsert_by`],
    /// decided here ([`WorkerFacts::laid_over`]). The change it made, or
    /// `None` where the store did not take it.
    ///
    /// A row changed in place gets a new object in the standing one's
    /// place, under its id in the registry and with its running state. A
    /// key's next life registers its object under a new id, and the old
    /// life's object is dropped first, as a follow lays the two
    /// ([`Self::put`]).
    ///
    /// [`WorkersStore::upsert_by`]: crate::store::WorkersStore::upsert_by
    pub(crate) fn upsert(
        &self,
        store: &MemoryStore,
        facts: WorkerFacts,
        maker: Registrant,
    ) -> Option<MembershipChange> {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        let standing = self.row_under(&facts.key);
        let seen = standing.as_ref().map(|row| row.revision);
        let (change, row) = facts.laid_over(standing.as_ref(), maker, Instant::now());
        self.lay_decided(store, &mut place, change, &row, seen)
    }

    /// Write `patch` over the row `id` where its object was built at
    /// `revision`: the write of [`WorkersStore::patch`], decided here. The
    /// change it made, or `None` where the row moved or left, or the store
    /// did not take the write.
    ///
    /// [`WorkersStore::patch`]: crate::store::WorkersStore::patch
    pub(crate) fn patch(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        let mut row = self.guarded(id, revision)?;
        patch.laid_on(&mut row.facts);
        row.revision += 1;
        let change = MembershipChange::Relabelled { id };
        self.lay_decided(store, &mut place, change, &row, Some(revision))
    }

    /// Remove the row `id` where its object was built at `revision`: the
    /// write of [`WorkersStore::remove`], decided here. The object leaves
    /// the registry, then the store is told. The change it made, or `None`
    /// where the row moved or left, or the store did not take the write.
    ///
    /// [`WorkersStore::remove`]: crate::store::WorkersStore::remove
    pub(crate) fn remove(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        self.guarded(id, revision)?;
        self.take_decided(store, &mut place, id, revision)
    }

    /// The row `id` as its object and what stands beside it state it,
    /// where the object was built at `revision`.
    fn guarded(&self, id: WorkerId, revision: u64) -> Option<WorkerRow> {
        let row = self.row(id)?;
        (row.revision == revision).then_some(row)
    }

    /// The decided `change` and the `row` it leaves, over a row read at
    /// the revision `seen` by the read that ended at `place`: the row's
    /// object is built, the store is told, and the object is put in the
    /// registry. A change the store does not take builds an object nobody
    /// sees ([`Self::moved`]).
    fn lay_decided(
        &self,
        store: &MemoryStore,
        place: &mut Place,
        change: MembershipChange,
        row: &WorkerRow,
        seen: Option<u64>,
    ) -> Option<MembershipChange> {
        let built = self.build(row);
        if !store.follow_objects(change, Some(row.clone()), seen) {
            return self.moved(store, place);
        }
        self.put(store, row, built);
        Some(change)
    }

    /// The decided removal of the row `id`, read at `revision` by the read
    /// that ended at `place`: its object leaves the registry, then the
    /// store is told. Where the store does not take it, the row that stands
    /// gets an object anew ([`Self::moved`]): a new object under a new id
    /// in the registry. The leases the store counts on the row are stated
    /// on it, and what the old object alone held is not.
    fn take_decided(
        &self,
        store: &MemoryStore,
        place: &mut Place,
        id: WorkerId,
        revision: u64,
    ) -> Option<MembershipChange> {
        self.end(id);
        let change = MembershipChange::Removed { id };
        if !store.follow_objects(change, None, Some(revision)) {
            return self.moved(store, place);
        }
        Some(change)
    }

    /// The store did not take a write: another writer moved the row since
    /// the read that ended at `place`. The store is read to its end again,
    /// so the registry holds what that writer left when the write answers.
    fn moved(&self, store: &MemoryStore, place: &mut Place) -> Option<MembershipChange> {
        self.follow_from(store, place);
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
        store::{identity, HealthSink, Journal, Position, WorkerStatus, WorkersStore},
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
    /// whose objects follow their rows' status.
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
                objects: Objects::new(Arc::new(WorkerRegistry::new())).with_status_followed(),
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

        /// The row under `url` where the journal's last word, the table and
        /// the object agree on it, each fact and the status. No follow is
        /// run: the holder's place stands one change behind the store.
        pub(crate) fn one_row(&self, url: &str) -> WorkerRow {
            let (_, told, _) = self.last();
            let table = self.store.by_key(url).expect("the table holds the row");
            let object = self.object(url).expect("the object stands with no follow");
            let read = self.objects.row_of(&object).expect("the object's row");
            assert_eq!(told.as_ref(), Some(&table), "the journal against the table");
            assert_eq!(read, table, "the object against the table");
            let behind = self.store.epoch().checked_sub(1);
            assert_eq!(self.objects.read_to(), behind, "no follow laid it");
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

    /// The status of a row a write leaves is the table's word. A key's next
    /// life takes the status its ready worker stood in, a draining worker's
    /// starts pending, and so does another worker's. The journal, the table
    /// and the object say the same status, with no follow and behind one.
    #[test]
    fn a_keys_next_life_is_handed_the_status_the_table_holds() {
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
            let store = &replica.store;
            let id = replica.upsert(facts(1, "gold")).expect("registered").id();
            // The table's status moves by the store's own doors: a connect
            // signal, a drain. The object has not followed either.
            assert!(matches!(
                store.connected(id, 1, &|| true),
                crate::store::HealthWrite::Moved { .. }
            ));
            if stood == WorkerStatus::Draining {
                store.set_status(id, stood, 1).expect("the row drains");
            }
            let mut next = facts(2, "gold");
            state(&mut next);
            replica.upsert(next).expect("the next life");
            let row = replica.one_row(A);
            assert_eq!(row.status, handed, "behind a row that stood {stood:?}");
            // Behind a follow the three still say one row.
            replica.objects.follow(store);
            let object = replica.object(A).expect("the object");
            assert_eq!(replica.objects.row_of(&object), Some(row));
        }
    }

    /// The store keeps its own word of a status, whatever a decided row
    /// says: the holder reads a status off an object, which may trail the
    /// table. A row changed in place keeps the table's status, and a key's
    /// next life is handed the table's.
    #[test]
    fn the_store_keeps_its_word_of_a_status_whatever_a_decided_row_says() {
        let replica = Replica::new();
        let store = &replica.store;
        let id = replica.upsert(facts(1, "gold")).expect("registered").id();
        store.connected(id, 1, &|| true);
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
        assert_eq!(status(A), Some(WorkerStatus::Ready));
        assert_eq!(
            replica.last().1.map(|row| row.status),
            Some(WorkerStatus::Ready)
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

    /// A write decided over a row another writer moved since is refused
    /// under the store's lock, by the revision the holder read: a
    /// registration in place, a patch, a key's next life and a removal.
    /// Each answers `None`, which a door answers as a row that moved, and
    /// tells the journal nothing. The registry holds what the other writer
    /// left when the write answers, with no follow of the case's: the row
    /// whose object a removal took out has one anew.
    #[test]
    fn a_write_over_a_row_another_writer_moved_is_refused_and_follows_it() {
        type Write = fn(&Replica, &mut Place, WorkerRow) -> Option<MembershipChange>;
        /// A registration of `next` decided over the row `read`.
        fn lay(
            replica: &Replica,
            place: &mut Place,
            next: WorkerFacts,
            read: &WorkerRow,
        ) -> Option<MembershipChange> {
            let now = Instant::now();
            let (change, row) = next.laid_over(Some(read), Registrant::Control, now);
            let seen = Some(read.revision);
            replica
                .objects
                .lay_decided(&replica.store, place, change, &row, seen)
        }
        let writes: [(&str, Write); 4] = [
            ("a registration in place", |replica, place, read| {
                lay(replica, place, facts(1, "iron"), &read)
            }),
            ("a patch", |replica, place, mut read| {
                let seen = Some(read.revision);
                read.facts.priority = 7;
                read.revision += 1;
                let change = MembershipChange::Relabelled { id: read.id };
                replica
                    .objects
                    .lay_decided(&replica.store, place, change, &read, seen)
            }),
            ("a key's next life", |replica, place, read| {
                lay(replica, place, facts(2, "gold"), &read)
            }),
            ("a removal", |replica, place, read| {
                replica
                    .objects
                    .take_decided(&replica.store, place, read.id, read.revision)
            }),
        ];
        for (what, write) in writes {
            let replica = Replica::new();
            let (store, objects) = (&replica.store, &replica.objects);
            let id = replica.upsert(facts(1, "gold")).expect("registered").id();
            // The holder's read: the store to its end, the row off the object.
            let mut place = objects.read_to.lock();
            objects.follow_from(store, &mut place);
            let read = replica.object(A).and_then(|object| objects.row_of(&object));
            let read = read.expect("the row read");
            // Another writer, outside the holder's turn: a write of the
            // store's own, which takes no turn of the holder.
            let late = WorkerPatch {
                labels: Some(BTreeMap::from([("tier".to_owned(), "late".to_owned())])),
                ..WorkerPatch::default()
            };
            store
                .patch(id, late, read.revision)
                .expect("the late write");
            let told = replica.told.0.lock().len();
            let left = store.by_key(A).expect("the row that writer left");
            assert_eq!(write(&replica, &mut place, read), None, "{what}");
            drop(place);
            assert_eq!(replica.told.0.lock().len(), told, "{what}: nothing told");
            assert_eq!(store.by_key(A).as_ref(), Some(&left), "{what}: the table");
            let object = replica.object(A).expect("an object stands");
            assert_eq!(objects.row_of(&object), Some(left), "{what}: the registry");
        }
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
