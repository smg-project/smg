use std::{cell::RefCell, collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;
use smg_worker::registry::{WorkerId as ObjectId, WorkerRegistry};

use super::*;
use crate::{
    engine::view::Role,
    store::{identity, Journal, LeadWord, Position, WorkerStatus, WorkersStore},
    worker_host::Members,
};

pub(crate) const A: &str = "http://a:8000";

/// What a journal was told of each membership write: the change, the
/// row it left and the row it took out.
pub(crate) type Member = (MembershipChange, Option<WorkerRow>, Option<WorkerRow>);

/// What a journal was told of each status write: the row's id, the
/// status it left, the one it took and the models it serves.
pub(crate) type Status = (WorkerId, WorkerStatus, WorkerStatus, Arc<[String]>);

/// One word of a journal: a membership write or a status write.
#[derive(Clone)]
enum Word {
    Member(Box<Member>),
    Status(Status),
}

/// A journal that keeps what it was told of the membership, in the
/// order it was told, and where the row its last membership write took
/// out stood when it was told.
#[derive(Default)]
struct Told(Mutex<Vec<Word>>, Mutex<Option<usize>>);

impl Journal for Told {
    fn member(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        let word = Word::Member(Box::new((change, row.cloned(), left.cloned())));
        self.0.lock().push(word);
        *self.1.lock() = left.map(|left| std::ptr::from_ref(left).addr());
        true
    }

    fn status(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool {
        self.0.lock().push(Word::Status((id, from, to, models)));
        true
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

    fn released(&self) {
        let hook = LET_GO.with(|held| held.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }
}

/// What [`on_let_go`] keeps for the journal to run.
type Hook = Box<dyn FnOnce()>;

thread_local! {
    static LET_GO: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// `hook` runs once on this thread, where the next write of a
/// [`Replica`]'s store has let its locks go: behind a cut's write too,
/// ahead of the holder's lay. It is taken out first: a write the hook
/// itself makes meets none.
pub(crate) fn on_let_go(hook: impl FnOnce() + 'static) {
    LET_GO.with(|held| *held.borrow_mut() = Some(Box::new(hook)));
}

/// A replica as a controller keeps one: a journaled store, and a holder
/// whose objects lead their rows' status.
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

    /// A host that tells no entry, as a gateway keeps one: the same
    /// holder over a store with no journal.
    pub(crate) fn untold() -> Self {
        Self {
            store: MemoryStore::default(),
            ..Self::new()
        }
    }

    pub(crate) fn upsert(&self, facts: WorkerFacts) -> Option<MembershipChange> {
        self.objects.upsert(&self.store, facts, Registrant::Control)
    }

    /// The last membership write as the journal was told it: a status
    /// write is another word ([`Self::statuses`]).
    pub(crate) fn last(&self) -> Member {
        let told = self.told.0.lock();
        let last = told.iter().rev().find_map(|word| match word {
            Word::Member(member) => Some(Member::clone(member)),
            Word::Status(_) => None,
        });
        last.expect("a membership write was told")
    }

    /// Each status write as the journal was told it, in its order.
    pub(crate) fn statuses(&self) -> Vec<Status> {
        let told = self.told.0.lock();
        let statuses = told.iter().filter_map(|word| match word {
            Word::Status(status) => Some(status.clone()),
            Word::Member(_) => None,
        });
        statuses.collect()
    }

    /// The row the journal's words state last of the row `id`: the row
    /// the last membership write about it left, with each status told
    /// of it since laid over it. A write about another row says nothing
    /// of this one. A status write tells no row: it names the status
    /// the row left, which is the one the journal last told of it, and
    /// the models the row serves.
    fn stated(&self, id: WorkerId) -> Option<WorkerRow> {
        let mut stated: Option<WorkerRow> = None;
        for word in self.told.0.lock().iter() {
            match word {
                Word::Member(member) => match member.0 {
                    // The key's next life took this row out.
                    MembershipChange::Replaced { old, .. } if old == id => stated = None,
                    change if change.id() == id => stated.clone_from(&member.1),
                    _ => {}
                },
                Word::Status((told, from, to, models)) if *told == id => {
                    let Some(row) = stated.as_mut() else {
                        continue;
                    };
                    assert_eq!(row.status, *from, "the status the journal last told");
                    assert_eq!(
                        row.facts.models[..],
                        models[..],
                        "a status names its models"
                    );
                    row.status = *to;
                }
                Word::Status(_) => {}
            }
        }
        stated
    }

    /// Where the row the last membership write took out stood when the
    /// journal was told it.
    fn took_at(&self) -> Option<usize> {
        *self.told.1.lock()
    }

    pub(crate) fn object(&self, url: &str) -> Option<ObjectId> {
        self.objects.registry().get_id_by_url(url)
    }

    /// The row under `url` where the journal's last word of it, the object
    /// and the table agree on it, each fact and the status. The journal
    /// is told the row the write handed, which the object read back
    /// states, and the table takes it. A status told since is the
    /// journal's word of the row's status ([`Self::stated`]).
    pub(crate) fn one_row(&self, url: &str) -> WorkerRow {
        let table = self.store.by_key(url).expect("the table holds the row");
        let told = self.stated(table.id);
        let object = self.object(url).expect("the object stands");
        let read = self.objects.row_of(&object).expect("the object's row");
        assert_eq!(told.as_ref(), Some(&read), "the journal against the object");
        assert_eq!(table, read, "the table against the object");
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

/// Have `objects` lay each row `store` holds that no object of its was
/// built from: a row written around the holder. The holder's pass over
/// a cut's rows runs over the store's, in the holder's turn.
pub(crate) fn lay_standing<G: Generation>(objects: &Objects<G>, store: &MemoryStore) {
    let _turn = objects.turn.lock();
    let (_, rows) = store.all_rows();
    objects.reconcile(store, &rows);
}

/// The cases' road to a store's table writes what the holder's write
/// calls write. One registration, its refresh in place, a patch, a
/// connect signal, the key's next life and a removal go through a
/// holder. The same six go through the cases' road of a second store.
/// The two tables hold the same rows, and the two journals were told
/// the same.
#[test]
fn the_cases_road_writes_a_table_as_the_holders_write_calls_do() {
    let replica = Replica::new();
    let (held, objects) = (&replica.store, &replica.objects);
    let told = Arc::new(Told::default());
    let alone = MemoryStore::journaled(Arc::clone(&told) as _);
    // A row but for the instant it was admitted at: each store reads
    // its own clock.
    let plain = |row: &WorkerRow| {
        let row = row.clone();
        (row.id, row.facts, row.status, row.revision, row.maker)
    };
    // A membership word by its change and its two rows, and a status
    // word as it was told: the holder hands its object's models, the
    // road hands none, and both journals are told one list.
    let words = |told: &Told| {
        let told = told.0.lock();
        let word = |word: &Word| match word {
            Word::Member(member) => {
                let (change, row, left) = &**member;
                let (row, left) = (row.as_ref().map(plain), left.as_ref().map(plain));
                (Some((*change, row, left)), None)
            }
            Word::Status(status) => (None, Some(status.clone())),
        };
        told.iter().map(word).collect::<Vec<_>>()
    };
    let same = || {
        let rows = |store: &MemoryStore| store.all_rows().1.iter().map(plain).collect::<Vec<_>>();
        assert_eq!(rows(held), rows(&alone), "the two tables");
        assert_eq!(words(&replica.told), words(&told), "the two journals");
    };
    let maker = Registrant::Control;
    // A new key, then its refresh in place.
    for tier in ["gold", "iron"] {
        let by_holder = objects.upsert(held, facts(1, tier), maker.clone());
        let by_road = alone.upsert_by(facts(1, tier), maker.clone());
        assert_eq!(by_holder, Some(by_road));
        same();
    }
    // A patch at the row's revision.
    let id = identity(A, 1);
    let revision = held.get(id).expect("the row").revision;
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    let by_holder = objects.patch(held, id, patch.clone(), revision);
    assert_eq!(by_holder, alone.patch(id, patch, revision));
    same();
    // A status: the holder's connect signal, and the road's ready.
    let revision = held.get(id).expect("the row").revision;
    objects.connect_row(held, id, revision);
    let by_road = alone.set_status(id, WorkerStatus::Ready, revision);
    assert!(by_road.is_some(), "the status moved");
    same();
    // The key's next life, then its removal.
    let by_holder = objects.upsert(held, facts(2, "iron"), maker.clone());
    assert_eq!(by_holder, Some(alone.upsert_by(facts(2, "iron"), maker)));
    same();
    let next = identity(A, 2);
    let revision = held.get(next).expect("the next life's row").revision;
    assert_eq!(
        objects.remove(held, next, revision),
        alone.remove(next, revision)
    );
    same();
    assert_eq!(told.0.lock().len(), 6, "each write was told once");
}

/// Each write of a row is decided over the objects: the object stands
/// when the write returns, and the journal, the table
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
/// the table and the object say the same status.
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
    }
}

/// A status moves by its own event alone: a row changed in place keeps
/// the status the table holds, whatever its decided row says. A key's
/// next life starts in the status its decided row states: the one the
/// holder's rule handed on from the object. The old life it takes out
/// is told in the status the table held, whatever the handed row says.
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
    assert!(store.follow_objects(relabelled, stale, Some(1), None));
    let status = |url: &str| store.by_key(url).map(|row| row.status);
    assert_eq!(status(A), Some(WorkerStatus::Ready));
    // The next life, decided over that same reading.
    let mut standing = store.by_key(A).map(|row| WorkerRow {
        status: WorkerStatus::Pending,
        ..row
    });
    let now = Instant::now();
    let (change, next) = WorkerRow::laid_over(
        facts(2, "gold"),
        standing.as_ref(),
        Registrant::Control,
        now,
    );
    assert_eq!(next.status, WorkerStatus::Pending, "as the holder read it");
    assert!(store.follow_objects(change, Some(next), Some(2), standing.as_mut()));
    assert_eq!(status(A), Some(WorkerStatus::Pending));
    let (_, laid, left) = replica.last();
    assert_eq!(laid.map(|row| row.status), Some(WorkerStatus::Pending));
    assert_eq!(left.map(|row| row.status), Some(WorkerStatus::Ready));
}

/// The row a holder's write takes out is told as the holder handed it:
/// the journal is told that very row, with the status the table held
/// laid over it. So a removal reads as draining only where the log
/// told a drain.
#[test]
fn a_row_a_write_takes_out_is_told_as_handed_in_the_status_the_table_holds() {
    let replica = Replica::new();
    let store = &replica.store;
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    replica.objects.connect_row(store, id, 1);
    // The holder's read of a row whose object drains ahead of the log.
    let mut read = store.by_key(A).map(|row| WorkerRow {
        status: WorkerStatus::Draining,
        ..row
    });
    let handed = read.as_ref().map(|row| std::ptr::from_ref(row).addr());
    let removed = MembershipChange::Removed { id };
    assert!(store.follow_objects(removed, None, Some(1), read.as_mut()));
    assert_eq!(replica.took_at(), handed, "the handed row itself");
    let (_, _, left) = replica.last();
    assert_eq!(left, read, "as the store left the handed row");
    let told = left.map(|row| row.status);
    assert_eq!(told, Some(WorkerStatus::Ready), "the table's status");
}

/// While the table stands, a debug build holds the row a holder hands
/// to the row the table held, where an entry is told. Nothing in a
/// controller sets a fact of an object behind its build. Here another
/// writer does, as a gateway's refresh of a worker's models: the row
/// read back states models the table's row does not, and the removal
/// that hands it stops.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row its holder read back is the row the table held")]
fn a_removal_that_hands_a_row_the_table_does_not_hold_stops_a_debug_build() {
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    refresh(&replica);
    objects.remove(store, id, 1);
}

/// The same for a key's next life, which hands the old life's row.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row its holder read back is the row the table held")]
fn a_next_life_that_hands_a_row_the_table_does_not_hold_stops_a_debug_build() {
    let replica = Replica::new();
    replica.upsert(facts(1, "gold")).expect("registered");
    refresh(&replica);
    replica.upsert(facts(2, "gold"));
}

/// A writer hands a row as taken out only with a write that takes one
/// out, a removal or a key's next life: one handed with a change in
/// place stops a debug build.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row is handed as taken out by a write that takes one out")]
fn a_row_handed_with_a_change_in_place_stops_a_debug_build() {
    let replica = Replica::new();
    let store = &replica.store;
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    let mut standing = store.by_key(A);
    let next = standing.clone().map(|row| WorkerRow { revision: 2, ..row });
    let relabelled = MembershipChange::Relabelled { id };
    assert!(store.follow_objects(relabelled, next, Some(1), standing.as_mut()));
}

/// A host with no journal tells no entry, so the row its holder hands
/// is held to nothing there. A gateway's refresh of a worker's models
/// sets a fact of its object behind the build: the key's next life and
/// the removal behind it are taken, in a debug build too.
#[test]
fn a_host_that_tells_no_entry_takes_a_handed_row_its_table_does_not_hold() {
    let replica = Replica::untold();
    let (store, objects) = (&replica.store, &replica.objects);
    let first = replica.upsert(facts(1, "gold")).expect("registered").id();
    refresh(&replica);
    let second = identity(A, 2);
    let replaced = MembershipChange::Replaced {
        old: first,
        new: second,
    };
    assert_eq!(replica.upsert(facts(2, "gold")), Some(replaced));
    refresh(&replica);
    let removed = MembershipChange::Removed { id: second };
    assert_eq!(objects.remove(store, second, 1), Some(removed));
    assert_eq!((replica.object(A), store.by_key(A)), (None, None));
}

/// A status is told with the models its object serves and with no row:
/// the holder reads them off the object under the hold that reads the
/// status. A row whose models change in place tells its next status
/// with the new ones, and the status told before it stands as it was
/// told.
#[test]
fn a_status_is_told_with_the_models_its_object_serves_at_the_write() {
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    objects.connect_row(store, id, 1);
    let (pending, ready) = (WorkerStatus::Pending, WorkerStatus::Ready);
    let served: Arc<[String]> = facts(1, "gold").models.into();
    let first = (id, pending, ready, served);
    assert_eq!(replica.statuses(), std::slice::from_ref(&first));
    // The row's models change in place: its object serves the new ones.
    let narrower = WorkerFacts {
        models: vec!["m".to_owned()],
        ..facts(1, "gold")
    };
    let change = replica.upsert(narrower);
    assert_eq!(change, Some(MembershipChange::Relabelled { id }));
    assert!(objects.drain_row(store, id, 2), "the row drains");
    let second = (id, ready, WorkerStatus::Draining, ["m".to_owned()].into());
    assert_eq!(replica.statuses(), [first, second]);
    assert_eq!(replica.one_row(A).status, WorkerStatus::Draining);
}

/// The list a holder hands with a status is the very list the journal
/// is told: where one is handed, the store copies no model under its
/// lock.
#[test]
fn a_status_tells_the_journal_the_very_list_its_holder_handed() {
    let replica = Replica::new();
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    let handed: Arc<[String]> = facts(1, "gold").models.into();
    let (ready, word) = (WorkerStatus::Ready, LeadWord::FREE);
    let told = replica
        .store
        .status_told(id, ready, 1, word, Some(Arc::clone(&handed)));
    assert!(told.moved().is_some(), "the store took the status");
    let statuses = replica.statuses();
    let [(.., models)] = statuses.as_slice() else {
        panic!("one status was told: {statuses:?}");
    };
    assert!(Arc::ptr_eq(models, &handed), "the handed list itself");
}

/// While the table stands, a debug build holds the models a holder
/// hands with a status to the ones the table's row serves. Here another
/// writer set the object's models behind its build, as a gateway's
/// refresh does: the status that hands the object's models stops.
#[cfg(debug_assertions)]
#[test]
#[should_panic(
    expected = "the models a holder read off its object are the ones the table's row serves"
)]
fn a_status_that_hands_models_the_table_does_not_hold_stops_a_debug_build() {
    let replica = Replica::new();
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    refresh(&replica);
    replica.objects.connect_row(&replica.store, id, 1);
}

/// The debug hold reads past the first model: a status that hands a part
/// of the models the table's row serves stops too.
#[cfg(debug_assertions)]
#[test]
#[should_panic(
    expected = "the models a holder read off its object are the ones the table's row serves"
)]
fn a_status_that_hands_a_part_of_the_tables_models_stops_a_debug_build() {
    let replica = Replica::new();
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    let part: Arc<[String]> = ["m".to_owned()].into();
    let (ready, word) = (WorkerStatus::Ready, LeadWord::FREE);
    replica.store.status_told(id, ready, 1, word, Some(part));
}

/// A host with no journal tells no entry: its holder hands its store no
/// models with a status, and none is held to the table's row. A status
/// behind a refresh of the worker's models is taken, in a debug build
/// too.
#[test]
fn a_host_that_tells_no_entry_takes_a_status_behind_a_refresh() {
    let replica = Replica::untold();
    let (store, objects) = (&replica.store, &replica.objects);
    let id = replica.upsert(facts(1, "gold")).expect("registered").id();
    refresh(&replica);
    objects.connect_row(store, id, 1);
    let status = store.by_key(A).map(|row| row.status);
    assert_eq!(status, Some(WorkerStatus::Ready));
}

/// Another writer of the registry sets the models of the object under
/// `A` behind its build.
pub(crate) fn refresh(replica: &Replica) {
    let registry = replica.objects.registry();
    let object = replica.object(A).expect("the object stands");
    let worker = registry.get(&object).expect("the worker");
    let card = openai_protocol::model_card::ModelCard::new;
    assert!(registry.refresh_models(&worker, vec![card("m"), card("late")]));
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

/// A row written around the holder has no object: the store took its
/// registration through its own entry. A patch and a removal by the
/// row's id find no object to guard them. A registration under its key
/// is decided over no object, and the store refuses it each time it is
/// made. Each answers `None`, tells nothing and lays nothing. A cut laid
/// whole lays the row's object, and the same registration is decided
/// over it.
#[test]
fn a_row_written_around_the_holder_has_no_object_until_a_cut_is_laid_whole() {
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let now = Instant::now();
    let (change, row) = WorkerRow::laid_over(facts(1, "gold"), None, Registrant::Control, now);
    store.follow(Position::default(), 1, change, Some(row.clone()));
    assert_eq!(store.by_key(A).as_ref(), Some(&row), "the store took it");
    let id = change.id();
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    assert_eq!(objects.patch(store, id, patch, 1), None);
    assert_eq!(objects.remove(store, id, 1), None);
    let told = replica.told.0.lock().len();
    for _ in 0..2 {
        assert_eq!(replica.upsert(facts(1, "iron")), None, "refused");
        assert_eq!(replica.object(A), None, "nothing is laid");
    }
    assert_eq!(replica.told.0.lock().len(), told, "nothing told");
    assert_eq!(store.by_key(A).as_ref(), Some(&row), "the table stands");
    assert!(objects.restore_from(store, store.checkpoint()));
    let object = replica.object(A).expect("the cut laid it");
    assert_eq!(objects.row_of(&object), Some(row));
    let again = replica.upsert(facts(1, "iron"));
    assert_eq!(again, Some(MembershipChange::Relabelled { id }));
    assert_eq!(replica.one_row(A).facts, facts(1, "iron"));
}

/// A write decided over a row that was written around the holder since
/// is refused under the store's lock, by the revision the holder read: a
/// registration in place, a patch, a key's next life and a removal.
/// That writer moved the row in place, or left the key's next life.
///
/// Each answers `None`, which a door answers as a row that moved, and
/// tells the journal nothing. The table holds what that writer left,
/// and the registry holds the old row as it stood: nothing is laid. A
/// removal took its object out first, and laid the row again.
#[test]
fn a_write_over_a_row_written_around_the_holder_is_refused_and_lays_nothing() {
    type Write = fn(&Replica, WorkerRow) -> Option<MembershipChange>;
    /// Another writer, outside the holder's turn: a write of the
    /// store's own, which takes no turn of the holder.
    type Other = fn(&MemoryStore, &WorkerRow);
    /// A registration of `next` decided over the row `read`.
    fn lay(replica: &Replica, next: WorkerFacts, read: &WorkerRow) -> Option<MembershipChange> {
        let now = Instant::now();
        let (change, row) = WorkerRow::laid_over(next, Some(read), Registrant::Control, now);
        let seen = Some(read.revision);
        replica
            .objects
            .lay_decided(&replica.store, change, &row, seen, None)
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
                .lay_decided(&replica.store, change, &read, seen, None)
        }),
        ("a key's next life", |replica, read| {
            lay(replica, facts(2, "gold"), &read)
        }),
        ("a removal", |replica, mut read| {
            replica.objects.take_decided(&replica.store, &mut read)
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
            let turn = objects.turn.lock();
            let stood = replica.object(A);
            let read = stood.as_ref().and_then(|object| objects.row_of(object));
            let read = read.expect("the row read");
            other(store, &read);
            let told = replica.told.0.lock().len();
            let left = store.by_key(A).expect("the row that writer left");
            let wrote = write(&replica, read.clone());
            drop(turn);
            let tells = replica.told.0.lock().len();
            let table = store.by_key(A);
            let object = replica.object(A);
            let laid = object.as_ref().and_then(|object| objects.row_of(object));
            assert_eq!(wrote, None, "{what}, {how}: refused");
            assert_eq!(tells, told, "{what}, {how}: nothing told");
            assert_eq!(table.as_ref(), Some(&left), "{what}, {how}: the table");
            // The old row stands as it stood. A removal took its
            // object out and laid the row again.
            assert_eq!(laid, Some(read), "{what}, {how}: the registry");
            let anew = what == "a removal";
            assert_eq!(object != stood, anew, "{what}, {how}: its object");
        }
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
