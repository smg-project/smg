use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Barrier, Mutex,
    },
    thread,
};

use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    settings::UnhealthyRemovalScope,
    worker::WorkerStatus as ObjectStatus,
};
use tonic::{Code, Status};

use super::{MembershipCalls, Position, ReplicaMembershipWriter};
use crate::{
    engine::view::{Role, WireModes, WorkerId},
    service::{
        controller::{
            decision_slots::tests::holder,
            election::LeasePlace,
            principal::{tests::told, Principal},
            replication::{journaled, tests::journaled_from},
        },
        proto::{self, register_worker_request::Mode, worker_event::Event},
        replication,
    },
    store::{
        self, identity, GatewayId, MemoryStore, Reachability, Registrant, WorkerFacts, WorkerRow,
        WorkerStatus, WorkersStore,
    },
    worker_host::{
        tests::{
            check_event_signal, check_expectations, check_guard, check_health_scope, check_maker,
            check_one_row, check_provider, check_rank_rows, check_refresh, lay_standing, FindOne,
            Rows, StoreDoor,
        },
        MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Precondition, Remover,
        RowRevision, Stated,
    },
};

mod audit;
mod changes;
mod connect_signal;
mod lives;
mod makers;
mod objects;
mod provider;
mod ranks;
mod registration;
mod term;
mod turn;

const URL: &str = "grpc://10.0.0.5:8000";
const TERM: u64 = 7;

/// The peer the door's calls are admitted from.
fn operator() -> Principal {
    told("operator-a")
}

/// The fence of a replica that leads under `term`, as the service hands
/// it to a call.
fn under(term: u64) -> impl Fn() -> Result<u64, Status> {
    move || Ok(term)
}

/// A door over a journaled store that stands in [`TERM`], its count
/// starting at zero so a case's table reads [`life`]: the term's start
/// is the log's first place, so a case's first write is its second.
fn door() -> (Arc<MemoryStore>, MembershipCalls) {
    door_in(TERM)
}

/// The same under another term. The door decides over a holder of its
/// own, built as the controller builds one.
fn door_in(term: u64) -> (Arc<MemoryStore>, MembershipCalls) {
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), term);
    let objects = holder(&store);
    (
        Arc::clone(&store),
        MembershipCalls::based(store, feed.log, 0, objects),
    )
}

/// A life's number at a door whose count starts at zero.
fn life(term: u64, epoch: u64) -> u64 {
    super::life(term, 0, epoch)
}

/// The id of the life the door numbers under [`TERM`] in a store that
/// stands at `epoch`.
fn numbered(url: &str, epoch: u64) -> WorkerId {
    identity(url, life(TERM, epoch))
}

fn spec(url: &str) -> proto::WorkerSpec {
    proto::WorkerSpec {
        url: url.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        models: vec!["model-a".to_owned()],
        ..Default::default()
    }
}

fn unchecked(url: &str) -> proto::WorkerSpec {
    proto::WorkerSpec {
        health_policy: Some(proto::HealthPolicy {
            disabled: true,
            ..Default::default()
        }),
        ..spec(url)
    }
}

fn ask(spec: proto::WorkerSpec, mode: Mode) -> proto::RegisterWorkerRequest {
    proto::RegisterWorkerRequest {
        spec: Some(spec),
        mode: mode as i32,
        wait_ready: false,
    }
}

/// The event's row, for the three kinds that carry one.
fn row_of(event: &proto::WorkerEvent) -> proto::WorkerRow {
    match event.event.clone().expect("an event") {
        Event::Registered(row) | Event::Relabelled(row) => row,
        Event::Replaced(replaced) => replaced.new.expect("the new row"),
        other => panic!("no row in {other:?}"),
    }
}

/// The door's log as its store's journal, for a case that stands behind
/// a membership write. It hands every call on and keeps each membership
/// change it is told. At the first release behind one it runs the hook
/// the case armed, once.
struct BehindAMembershipWrite {
    log: Arc<replication::Log>,
    /// The membership changes told since the case last took them.
    told: Mutex<Vec<store::MembershipChange>>,
    /// Whether a membership change was told and its release has not come.
    owed: AtomicBool,
}

thread_local! {
    /// What [`BehindAMembershipWrite`] runs on this thread at the first
    /// release behind a membership change.
    static BEHIND: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

impl store::Journal for BehindAMembershipWrite {
    fn member(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        change: store::MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        let goes_on = self.log.member(at, term, epoch, change, row, left);
        self.told.lock().expect("the case's own lock").push(change);
        self.owed.store(true, Ordering::Relaxed);
        goes_on
    }

    fn status(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool {
        self.log.status(at, term, epoch, id, from, to, models)
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        self.log.generation(at, generation);
    }

    fn term_rose(&self, term: u64) {
        self.log.term_rose(term);
    }

    fn term_followed(&self, at: Position) {
        self.log.term_followed(at);
    }

    fn head(&self) -> Position {
        self.log.head()
    }

    fn followed(&self) -> Option<u64> {
        self.log.followed()
    }

    fn released(&self) {
        self.log.released();
        if self.owed.swap(false, Ordering::Relaxed) {
            let hook = BEHIND.with(|held| held.borrow_mut().take());
            if let Some(hook) = hook {
                hook();
            }
        }
    }

    fn restart(&self, at: Position) -> bool {
        self.log.restart(at).is_ok()
    }
}

/// A door's patch and removal are decided over the worker objects, as
/// its registration is: each runs the store's one body, and a removal's
/// object has left the registry when the store hears of it. So it is for
/// a control call and for a removal by the replica's own writer.
///
/// The store's journal is the witness. The one body alone tells it a
/// membership change, so one change told with the release behind it is
/// that body's run: one for the patch and one for each removal. A patch's
/// call takes the write's hold a second time, for its object's status.
/// That hold tells no membership change, so the case does not count it.
#[test]
fn a_doors_patch_and_removal_are_decided_over_the_worker_objects() {
    use std::{cell::Cell, rc::Rc};

    use store::MembershipChange::{Relabelled, Removed};

    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(BehindAMembershipWrite {
        log: Arc::clone(&feed.log),
        told: Mutex::default(),
        owed: AtomicBool::new(false),
    });
    let handed = Arc::clone(&journal);
    let store = Arc::new(MemoryStore::journaled(handed).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let calls = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
    let door = Arc::new(calls);
    // The membership changes `write` told the journal, and whether the
    // first release behind one came with the object of the row `gone`
    // out of the registry.
    let decided = |gone: Option<WorkerId>, write: &dyn Fn()| {
        let ran = Rc::new(Cell::new(false));
        let (released, objects) = (Rc::clone(&ran), Arc::clone(&objects));
        let told = || std::mem::take(&mut *journal.told.lock().expect("the case's own lock"));
        // What was told ahead of `write` is another call's.
        drop(told());
        BEHIND.set(Some(Box::new(move || {
            let stands = gone.and_then(|gone| objects.object(gone));
            assert_eq!(stands, None, "the object leaves first");
            released.set(true);
        })));
        write();
        (told(), ran.get())
    };
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let row = store.by_key(URL).expect("the row");
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id.0,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: row.revision,
    };
    let patched = decided(None, &|| {
        door.patch(&operator(), under(TERM), &patch)
            .expect("patched");
    });
    assert_eq!(
        patched,
        (vec![Relabelled { id: row.id }], true),
        "a patch call"
    );
    let row = store.by_key(URL).expect("the row");
    let leave = proto::DeregisterWorkerRequest {
        worker_id: row.id.0,
        revision: row.revision,
        drain: false,
    };
    let removed = decided(Some(row.id), &|| {
        door.deregister(&operator(), under(TERM), &leave)
            .expect("removed");
    });
    assert_eq!(
        removed,
        (vec![Removed { id: row.id }], true),
        "a removal call"
    );
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let maker = found("pods/a");
    hosted
        .found_one(&maker, URL, &|_| found_facts())
        .expect("found");
    let made = store.by_key(URL).expect("the found row");
    let gone = decided(Some(made.id), &|| {
        hosted
            .gone(Remover::Registrant(&maker), URL, standing(&made))
            .expect("gone");
    });
    assert_eq!(
        gone,
        (vec![Removed { id: made.id }], true),
        "a removal by the replica's own writer"
    );
}

/// A spec that states `width` ranks.
fn wide(url: &str, width: u32) -> proto::WorkerSpec {
    proto::WorkerSpec {
        dp_size: width,
        ..spec(url)
    }
}

/// What an answer of several ranks made of each row.
fn made_of(event: &proto::WorkerEvent) -> Vec<proto::rank_row::Made> {
    match event.event.clone().expect("an event") {
        Event::Ranks(ranks) => ranks
            .rows
            .into_iter()
            .map(|row| row.made.expect("a case"))
            .collect(),
        other => panic!("no ranks in {other:?}"),
    }
}

/// The door's log as its store's journal, which leaves its term as a
/// membership write lands: a lead lost inside one call.
struct LosesTheLead {
    log: Arc<replication::Log>,
    /// The membership writes the lead still holds for, a status write
    /// among them. The write that finds none left loses it, once.
    holds: AtomicUsize,
}

impl LosesTheLead {
    /// One write was told: the lead holds for one write fewer, and the
    /// write that finds none left loses it.
    fn told(&self) {
        match self.holds.load(Ordering::Relaxed) {
            0 => {
                let next = Position {
                    seq: 40,
                    term: TERM + 1,
                };
                assert!(self.log.restart(next).is_ok(), "a place the log takes");
                self.holds.store(usize::MAX, Ordering::Relaxed);
            }
            more => self.holds.store(more - 1, Ordering::Relaxed),
        }
    }
}

impl store::Journal for LosesTheLead {
    fn member(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        change: store::MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        let goes_on = self.log.member(at, term, epoch, change, row, left);
        self.told();
        goes_on
    }

    fn status(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool {
        let goes_on = self.log.status(at, term, epoch, id, from, to, models);
        self.told();
        goes_on
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        self.log.generation(at, generation);
    }

    fn term_rose(&self, term: u64) {
        self.log.term_rose(term);
    }

    fn term_followed(&self, at: Position) {
        self.log.term_followed(at);
    }

    fn head(&self) -> Position {
        self.log.head()
    }

    fn followed(&self) -> Option<u64> {
        self.log.followed()
    }

    fn released(&self) {
        self.log.released();
    }

    fn restart(&self, at: Position) -> bool {
        self.log.restart(at).is_ok()
    }
}

/// A provider's maker for one worker.
fn found(key: &str) -> Registrant {
    Registrant::provider("cluster", key).expect("a whole maker")
}

fn found_facts() -> WorkerFacts {
    WorkerFacts {
        key: URL.to_owned(),
        engine: URL.to_owned(),
        wire_modes: WireModes::GRPC,
        models: vec!["model-a".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

fn standing(row: &WorkerRow) -> RowRevision {
    RowRevision {
        row: row.id,
        revision: row.revision,
    }
}

impl Rows for ReplicaMembershipWriter {
    fn rows(
        &self,
        maker: &Registrant,
        expect: Precondition,
        rows: &[Stated<'_>],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        ReplicaMembershipWriter::rows(self, maker, expect, rows)
    }
}

fn hosted_door() -> (Arc<MemoryStore>, Box<dyn Rows>) {
    let (store, door) = door();
    (
        store,
        Box::new(ReplicaMembershipWriter::new(
            Arc::new(door),
            operator(),
            under(TERM),
        )),
    )
}

/// A door whose registration `request`, under [`TERM`], lost its lead
/// as a membership write of its own landed, the one behind the `held`
/// writes its lead held for: the store, the door and what the call
/// answered. The log leads no term behind it.
fn lost_its_lead_at(
    held: usize,
    request: &proto::RegisterWorkerRequest,
) -> (
    Arc<MemoryStore>,
    MembershipCalls,
    Result<proto::WorkerEvent, Status>,
) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(held),
    });
    let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    let answer = door.register(&operator(), under(TERM), request);
    (store, door, answer)
}

/// A term that ends by a later take of the store ([`LeavesAtATelling`]).
const BY_A_RISE: usize = 1;
/// A term that ends by the start of a later leader's term, which the store
/// follows ([`LeavesAtATelling`]).
const BY_A_FOLLOWED_TERM: usize = 2;

/// The door's log as its store's journal, which leaves [`TERM`] in a
/// telling of the kind a case arms it for, a status's or a row's: the
/// next one, or the one behind the rows' tellings a case lets by. It
/// leaves behind every ask its writer makes ahead of the write, and
/// ahead of the place the log gives the event.
struct LeavesAtATelling {
    log: Arc<replication::Log>,
    /// How the term ends at that telling, [`BY_A_RISE`] or
    /// [`BY_A_FOLLOWED_TERM`], and zero once it has.
    armed: AtomicUsize,
    /// Whether the telling it leaves at is a row's write and no status:
    /// a registration's, a patch's, a removal's.
    at_a_row: AtomicBool,
    /// The tellings of a row it lets by ahead of that one: the ranks a
    /// registration writes before the rank its term ends at.
    lets_by: AtomicUsize,
    /// The tellings of a row it was handed, whatever came of each.
    rows: AtomicUsize,
}

impl LeavesAtATelling {
    /// Whether this telling of a row is one a case lets by: one less is
    /// left to let by then.
    fn lets_one_by(&self) -> bool {
        let one_less = |left: usize| left.checked_sub(1);
        self.lets_by
            .try_update(Ordering::Relaxed, Ordering::Relaxed, one_less)
            .is_ok()
    }

    /// Ends [`TERM`] the way a case armed, once.
    fn leave(&self) {
        match self.armed.swap(0, Ordering::Relaxed) {
            BY_A_RISE => store::Journal::term_rose(self.log.as_ref(), TERM + 1),
            BY_A_FOLLOWED_TERM => {
                let next = Position {
                    seq: self.log.head().seq + 1,
                    term: TERM + 1,
                };
                let guard = self.log.adopt(next).expect("the next place");
                guard.mark();
            }
            _ => {}
        }
    }
}

impl store::Journal for LeavesAtATelling {
    fn member(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        change: store::MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        self.rows.fetch_add(1, Ordering::Relaxed);
        if self.at_a_row.load(Ordering::Relaxed) && !self.lets_one_by() {
            self.leave();
        }
        self.log.member(at, term, epoch, change, row, left)
    }

    fn status(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool {
        if !self.at_a_row.load(Ordering::Relaxed) {
            self.leave();
        }
        self.log.status(at, term, epoch, id, from, to, models)
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        self.log.generation(at, generation);
    }

    fn term_rose(&self, term: u64) {
        self.log.term_rose(term);
    }

    fn term_followed(&self, at: Position) {
        self.log.term_followed(at);
    }

    fn head(&self) -> Position {
        self.log.head()
    }

    fn followed(&self) -> Option<u64> {
        self.log.followed()
    }

    fn released(&self) {
        self.log.released();
    }

    fn restart(&self, at: Position) -> bool {
        self.log.restart(at).is_ok()
    }
}

/// A door under [`TERM`] over a store whose journal leaves that term in
/// the telling of a status, once a case arms it: the store, the door and
/// the journal.
fn leaving() -> (Arc<MemoryStore>, MembershipCalls, Arc<LeavesAtATelling>) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LeavesAtATelling {
        log: Arc::clone(&feed.log),
        armed: AtomicUsize::new(0),
        at_a_row: AtomicBool::new(false),
        lets_by: AtomicUsize::new(0),
        rows: AtomicUsize::new(0),
    });
    let told: Arc<dyn store::Journal> = Arc::<LeavesAtATelling>::clone(&journal);
    let store = Arc::new(MemoryStore::journaled(told).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    (store, door, journal)
}

/// A connect signal whose term ends in its own telling writes nothing.
/// Every ask of the door's word held: the term ended behind them, by a
/// later take or by the start of a term the store follows, ahead of the
/// place the log would give the status's event. The log takes no event
/// and nothing moved. The row is pending on the object and in the table,
/// the epoch counts the registration's write alone, and the log's head is
/// the place that ended the term. The call that waits answers the row so.
fn a_connect_signal_whose_term_ends_at_its_telling(ends: usize) {
    let (store, door, journal) = leaving();
    journal.armed.store(ends, Ordering::Relaxed);
    let waits = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(URL), Mode::Add)
    };
    let told = door
        .register(&operator(), under(TERM), &waits)
        .expect("registered");
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the signal reached its telling: {ends}");
    let row = row_of(&told);
    assert_eq!(row.status, proto::WorkerStatus::Pending as i32, "{ends}");
    let id = WorkerId(row.id);
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending),
        "{ends}"
    );
    assert_eq!(store.epoch(), 1, "the registration's write alone: {ends}");
    // The term's start, the registration's write, then the event that
    // ended the term: no status stands behind it.
    let ended = Position {
        seq: 3,
        term: TERM + 1,
    };
    assert_eq!(journal.log.head(), ended, "{ends}");
}

/// A connect signal whose term a later take ends, in the signal's own
/// telling, moves nothing.
#[test]
fn a_connect_signal_whose_term_a_later_take_ends_at_its_telling_moves_nothing() {
    a_connect_signal_whose_term_ends_at_its_telling(BY_A_RISE);
}

/// A connect signal whose term the start of a followed term ends, in the
/// signal's own telling, moves nothing.
#[test]
fn a_connect_signal_whose_term_a_followed_term_ends_at_its_telling_moves_nothing() {
    a_connect_signal_whose_term_ends_at_its_telling(BY_A_FOLLOWED_TERM);
}

/// A drain whose term ends in its own telling writes nothing, as a
/// connect signal there writes nothing. The call answers that the store
/// does not lead the term, never that the row moved.
fn a_drain_whose_term_ends_at_its_telling(ends: usize) {
    let (store, door, journal) = leaving();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    let (epoch, head) = (store.epoch(), journal.log.head());
    journal.armed.store(ends, Ordering::Relaxed);
    let refused = door
        .deregister(&operator(), under(TERM), &drain)
        .expect_err("the term ended at the telling");
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the drain reached its telling: {ends}");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM),
        "{ends}"
    );
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending),
        "{ends}"
    );
    assert_eq!(store.epoch(), epoch, "{ends}");
    let ended = Position {
        seq: head.seq + 1,
        term: TERM + 1,
    };
    assert_eq!(journal.log.head(), ended, "{ends}");
}

/// A drain whose term a later take ends, in the drain's own telling,
/// moves nothing and is refused as not leading.
#[test]
fn a_drain_whose_term_a_later_take_ends_at_its_telling_is_refused_as_not_leading() {
    a_drain_whose_term_ends_at_its_telling(BY_A_RISE);
}

/// A drain whose term the start of a followed term ends, in the drain's
/// own telling, moves nothing and is refused as not leading.
#[test]
fn a_drain_whose_term_a_followed_term_ends_at_its_telling_is_refused_as_not_leading() {
    a_drain_whose_term_ends_at_its_telling(BY_A_FOLLOWED_TERM);
}

/// [`leaving`] with a journal that leaves the term in the telling of a
/// row's write, once a case arms it.
fn leaving_at_a_row() -> (Arc<MemoryStore>, MembershipCalls, Arc<LeavesAtATelling>) {
    let made = leaving();
    made.2.at_a_row.store(true, Ordering::Relaxed);
    made
}

/// What a call whose term ended in its own telling leaves. The journal
/// reached the telling, and the call answers that the store does not lead
/// the term, never that the row moved. The epoch stands at `epoch`, and
/// the log's head is the place behind `head` that ended the term: the log
/// took no event of the call.
fn left_at_its_telling(
    (store, journal): (&MemoryStore, &LeavesAtATelling),
    refused: &Status,
    (epoch, head): (u64, Position),
    said: &str,
) {
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the write reached its telling: {said}");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM),
        "{said}"
    );
    assert_eq!(store.epoch(), epoch, "{said}");
    let ended = Position {
        seq: head.seq + 1,
        term: TERM + 1,
    };
    assert_eq!(journal.log.head(), ended, "{said}");
}

/// What a registration under test makes of its URL.
#[derive(Clone, Copy, Debug)]
enum Lays {
    /// The URL's first row.
    First,
    /// A change in place of the row that stands.
    InPlace,
    /// The URL's next life.
    NextLife,
}

/// A registration's write whose term ends in its own telling writes
/// nothing, whatever it would make of its URL. The door's asks of its
/// term held: the term ended behind them, by a later take or by the start
/// of a term the store follows, ahead of the place the log would give the
/// row's event. Nothing moved: the row under the URL reads as it stood, in
/// the table and off the objects. Nothing of the write is left behind
/// either: under a term the store leads, the same registration is
/// written.
fn a_registration_whose_term_ends_at_its_telling(lays: Lays, ends: usize) {
    let (store, door, journal) = leaving_at_a_row();
    let said = format!("{lays:?}, {ends}");
    let request = match lays {
        Lays::First => ask(spec(URL), Mode::Add),
        Lays::InPlace => {
            let labelled = proto::WorkerSpec {
                labels: [("tier".to_owned(), "gold".to_owned())].into(),
                ..spec(URL)
            };
            ask(labelled, Mode::Upsert)
        }
        Lays::NextLife => ask(spec(URL), Mode::Replace),
    };
    if !matches!(lays, Lays::First) {
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
    }
    let stood = store.by_key(URL);
    let at = (store.epoch(), journal.log.head());
    journal.armed.store(ends, Ordering::Relaxed);
    let refused = door
        .register(&operator(), under(TERM), &request)
        .expect_err("the term ended at the telling");
    left_at_its_telling((&store, &journal), &refused, at, &said);
    assert_eq!(
        (store.by_key(URL), door.row_under(URL)),
        (stood.clone(), stood.clone()),
        "{said}"
    );
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let mark = journal.log.head();
    let told = door
        .register(&operator(), under(TERM + 2), &request)
        .expect("written under the term the store leads");
    let written = store.by_key(URL).expect("the row");
    let told = row_of(&told);
    assert_eq!(
        (written.id.0, written.revision),
        (told.id, told.revision),
        "{said}"
    );
    assert_ne!(Some(written), stood, "{said}");
    // The journal was told the epoch the write's step then gave.
    let mut tail = Vec::new();
    journal
        .log
        .read(mark, replication::Log::BATCH, &mut tail)
        .expect("the log holds the mark");
    let epochs: Vec<u64> = tail
        .iter()
        .filter_map(|entry| match &entry.event {
            replication::Event::Member { epoch, .. } => Some(*epoch),
            _ => None,
        })
        .collect();
    assert_eq!(epochs, [store.epoch()], "{said}");
}

/// A registration's write whose term a later take ends, in the write's
/// own telling, moves nothing and is refused as not leading.
#[test]
fn a_registrations_write_whose_term_a_later_take_ends_at_its_telling_moves_nothing() {
    for lays in [Lays::First, Lays::InPlace, Lays::NextLife] {
        a_registration_whose_term_ends_at_its_telling(lays, BY_A_RISE);
    }
}

/// A registration's write whose term the start of a followed term ends,
/// in the write's own telling, moves nothing and is refused as not
/// leading.
#[test]
fn a_registrations_write_whose_term_a_followed_term_ends_at_its_telling_moves_nothing() {
    for lays in [Lays::First, Lays::InPlace, Lays::NextLife] {
        a_registration_whose_term_ends_at_its_telling(lays, BY_A_FOLLOWED_TERM);
    }
}

/// A registration of several ranks hands each rank's write the call's
/// term. One whose term ends in its second rank's telling is refused at
/// that rank: the log takes no event of it, and the rank behind it is not
/// tried. The rank written before it stands, as behind a lead lost
/// between two ranks.
#[test]
fn a_rank_whose_term_ends_at_its_telling_is_refused_and_the_rank_before_it_stands() {
    let (store, door, journal) = leaving_at_a_row();
    let mark = journal.log.head();
    journal.lets_by.store(1, Ordering::Relaxed);
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect_err("the term ended at the second rank's telling");
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the second rank's write reached its telling");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    let written = |rank: u32| store.by_key(&format!("{URL}@{rank}"));
    let first = written(0).expect("the first rank's row");
    assert_eq!((written(1), written(2)), (None, None));
    // The second rank's telling was the last: the third's never came.
    assert_eq!(journal.rows.load(Ordering::Relaxed), 2);
    // The whole tail, batch by batch: a batch ends where a term does.
    let (mut from, mut batch, mut rows) = (mark, Vec::new(), Vec::new());
    loop {
        journal
            .log
            .read(from, replication::Log::BATCH, &mut batch)
            .expect("the log holds the place");
        let Some(last) = batch.last() else { break };
        from = last.at;
        rows.extend(batch.iter().filter_map(|entry| match &entry.event {
            replication::Event::Member { change, .. } => Some(*change),
            _ => None,
        }));
    }
    let told = store::MembershipChange::Registered { id: first.id };
    assert_eq!(rows, [told], "the first rank's event alone");
    let ended = Position {
        seq: mark.seq + 2,
        term: TERM + 1,
    };
    assert_eq!(journal.log.head(), ended, "the term's end stands last");
}

/// A narrower registration takes out the rank it no longer states, and
/// that removal goes under the call's term, as each rank's write does.
/// One whose term ends in the removal's own telling is refused there:
/// the rank stands, and the ranks written before it stay.
#[test]
fn a_surplus_ranks_removal_whose_term_ends_at_its_telling_takes_no_row_out() {
    let (store, door, journal) = leaving_at_a_row();
    door.register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect("registered");
    // The two ranks that stay are written first, each in place.
    journal.lets_by.store(2, Ordering::Relaxed);
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Upsert))
        .expect_err("the term ended at the removal's telling");
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the removal reached its telling");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    let width = |rank: u32| {
        store
            .by_key(&format!("{URL}@{rank}"))
            .map(|row| row.facts.dp_size)
    };
    assert_eq!((width(0), width(1), width(2)), (Some(2), Some(2), Some(3)));
}

/// A patch whose term ends in its own telling writes nothing, as a
/// registration's write there writes nothing: the row reads as it stood,
/// in the table and off its object. Under a term the store leads, the
/// same patch is written.
fn a_patch_whose_term_ends_at_its_telling(ends: usize) {
    let (store, door, journal) = leaving_at_a_row();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let stood = store.get(id);
    let at = (store.epoch(), journal.log.head());
    journal.armed.store(ends, Ordering::Relaxed);
    let refused = door
        .patch(&operator(), under(TERM), &patch)
        .expect_err("the term ended at the telling");
    left_at_its_telling((&store, &journal), &refused, at, &ends.to_string());
    assert_eq!(
        (store.get(id), store.by_key(URL), door.row(id)),
        (stood.clone(), stood.clone(), stood),
        "{ends}"
    );
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    door.patch(&operator(), under(TERM + 2), &patch)
        .expect("patched under the term the store leads");
    let patched = store.get(id).expect("the row");
    assert_eq!(
        (patched.revision, patched.facts.labels.len()),
        (row.revision + 1, 1),
        "{ends}"
    );
}

/// A patch whose term a later take ends, in the patch's own telling,
/// moves nothing and is refused as not leading.
#[test]
fn a_patch_whose_term_a_later_take_ends_at_its_telling_moves_nothing() {
    a_patch_whose_term_ends_at_its_telling(BY_A_RISE);
}

/// A patch whose term the start of a followed term ends, in the patch's
/// own telling, moves nothing and is refused as not leading.
#[test]
fn a_patch_whose_term_a_followed_term_ends_at_its_telling_moves_nothing() {
    a_patch_whose_term_ends_at_its_telling(BY_A_FOLLOWED_TERM);
}

/// A removal whose term ends in its own telling takes no row out. The
/// table holds the row as it stood. Its object left the registry ahead of
/// the store's word, and it is laid again from the row read off it: the
/// row reads off the objects as it stood. Under a term the store leads,
/// the same removal takes the row.
fn a_removal_whose_term_ends_at_its_telling(ends: usize) {
    let (store, door, journal) = leaving_at_a_row();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    let removal = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: false,
    };
    let stood = store.get(id);
    let at = (store.epoch(), journal.log.head());
    journal.armed.store(ends, Ordering::Relaxed);
    let refused = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("the term ended at the telling");
    left_at_its_telling((&store, &journal), &refused, at, &ends.to_string());
    assert_eq!(
        (store.get(id), store.by_key(URL), door.row(id)),
        (stood.clone(), stood.clone(), stood),
        "{ends}"
    );
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    door.deregister(&operator(), under(TERM + 2), &removal)
        .expect("removed under the term the store leads");
    assert_eq!((store.get(id), door.row(id)), (None, None), "{ends}");
}

/// A removal whose term a later take ends, in the removal's own telling,
/// takes no row out and is refused as not leading.
#[test]
fn a_removal_whose_term_a_later_take_ends_at_its_telling_takes_no_row_out() {
    a_removal_whose_term_ends_at_its_telling(BY_A_RISE);
}

/// A removal whose term the start of a followed term ends, in the
/// removal's own telling, takes no row out and is refused as not leading.
#[test]
fn a_removal_whose_term_a_followed_term_ends_at_its_telling_takes_no_row_out() {
    a_removal_whose_term_ends_at_its_telling(BY_A_FOLLOWED_TERM);
}

/// The object of the row `id` moved alone to `to`, as a probe's verdict
/// leaves it ahead of its telling: the table holds the row's status as it
/// was still. What comes back is that telling, made when a case calls it:
/// the store's answer to the move, told for the object the verdict moved.
fn moved_alone(
    door: &MembershipCalls,
    id: WorkerId,
    to: ObjectStatus,
) -> impl Fn(&MemoryStore) -> store::HealthWrite + '_ {
    let object = door.objects.object(id).expect("the row's object");
    let registry = door.objects.registry();
    let standing = registry.get(&object).expect("it stands");
    let (built, left) = (standing.revision(), standing.status());
    registry.apply_if_revision(&object, built, |_| ((), Some(to)));
    move |store| {
        door.objects
            .tell_status(store, &object, built, (left, to), &|| true)
    }
}

/// The status the holder tells behind a change in place goes under the
/// term of its call. A probe's verdict readied this row's object, and the
/// table was not told yet. A patch lays the row in place, and the holder
/// then tells the object's status. The term ends in that status's
/// telling, by a later take. The patch stands. No status is written
/// under a term the call was not admitted under, and the object takes
/// the status its row holds.
#[test]
fn a_status_told_behind_a_change_in_place_goes_under_its_calls_term() {
    let (store, door, journal) = leaving();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    // The verdict's own telling never comes in this case.
    let _telling = moved_alone(&door, id, ObjectStatus::Ready);
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let (epoch, head) = (store.epoch(), journal.log.head());
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    let patched = row_of(
        &door
            .patch(&operator(), under(TERM), &patch)
            .expect("patched"),
    );
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the status reached its telling");
    assert_eq!(patched.revision, row.revision + 1);
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending)
    );
    assert_eq!(store.epoch(), epoch + 1, "the patch alone");
    // The patch's event, then the event that ended the term: no status
    // stands behind it.
    let ended = Position {
        seq: head.seq + 2,
        term: TERM + 1,
    };
    assert_eq!(journal.log.head(), ended);
}

/// The same status under a term that holds is written under it. A
/// probe's verdict readied this row's object, and the table was not told
/// yet. A patch lays the row in place, and the holder tells the object's
/// status with the call's term, which the store leads still. The log
/// takes both events in that term, and the table reads ready as the
/// object does.
#[test]
fn a_status_told_behind_a_change_in_place_is_written_under_a_term_that_holds() {
    let (store, door, journal) = leaving();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    // The verdict's own telling never comes in this case.
    let _telling = moved_alone(&door, id, ObjectStatus::Ready);
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let (epoch, head) = (store.epoch(), journal.log.head());
    door.patch(&operator(), under(TERM), &patch)
        .expect("patched");
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let ready = Some(WorkerStatus::Ready);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (ready, ready)
    );
    assert_eq!(store.epoch(), epoch + 2, "the patch, then the status");
    let told = Position {
        seq: head.seq + 2,
        term: TERM,
    };
    assert_eq!(journal.log.head(), told);
}

/// A removal the store refuses for its term lays the row's object again
/// in the status the table holds, whatever that status is. A verdict
/// readied this row, and the table was told. A later verdict took the
/// row's object out of the ready ones, and the table holds the row ready
/// still: that verdict's telling had not come. The removal's term ends in
/// its own telling. The store laid its table's status over the handed row
/// for the log's event, and the row comes back in it: the object laid
/// again stands ready, as its row does. The late verdict's telling then
/// comes, for the object that left, and writes nothing. The next verdict
/// moves the object laid again, and the table is told.
#[test]
fn a_removal_refused_for_its_term_lays_its_object_again_in_the_tables_status() {
    let (store, door, journal) = leaving_at_a_row();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    let moved = |from, to| store::HealthWrite::Moved { from, to };
    let readied = moved_alone(&door, id, ObjectStatus::Ready);
    assert_eq!(
        readied(&store),
        moved(WorkerStatus::Pending, WorkerStatus::Ready)
    );
    let late = moved_alone(&door, id, ObjectStatus::NotReady);
    let removal = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: false,
    };
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    let refused = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("the term ended at the telling");
    assert_eq!(refused.code(), Code::Unavailable);
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let read = || (status(store.get(id)), status(door.row(id)));
    let ready = Some(WorkerStatus::Ready);
    assert_eq!(read(), (ready, ready));
    // The late verdict's telling names the object that left the registry:
    // no object stands under its id, and nothing is written.
    assert_eq!(late(&store), store::HealthWrite::Stale);
    assert_eq!(read(), (ready, ready));
    // The next verdict moves the object laid again, and its telling lands
    // under the term the log leads since.
    let next = moved_alone(&door, id, ObjectStatus::NotReady);
    assert_eq!(
        next(&store),
        moved(WorkerStatus::Ready, WorkerStatus::NotReady)
    );
    let out = Some(WorkerStatus::NotReady);
    assert_eq!(read(), (out, out));
}

/// The replica's own writer asks its turn's term behind a write the
/// store did not take, as a control call does. Its registration's term
/// ends in the write's own telling: nothing is written, and the write is
/// refused as a turn that does not lead, never as a row that moved.
#[test]
fn a_replicas_own_registration_whose_term_ends_at_its_telling_is_refused_as_not_leading() {
    let (store, door, journal) = leaving_at_a_row();
    let maker = found("pods/a");
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    assert_eq!(
        hosted.found_one(&maker, URL, &|_| found_facts()),
        Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned()
        ))
    );
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the write reached its telling");
    assert_eq!((store.by_key(URL), store.epoch()), (None, 0));
}

/// The replica's own removal asks its turn's term behind a write the
/// store did not take, as its registration does. A removal whose term
/// ends in its own telling takes no row out, and is refused as a turn
/// that does not lead, never as a row that moved.
#[test]
fn a_replicas_own_removal_whose_term_ends_at_its_telling_is_refused_as_not_leading() {
    let (store, door, journal) = leaving_at_a_row();
    let maker = found("pods/a");
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    hosted
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL).expect("the registered row");
    let (seen, epoch) = (standing(&row), store.epoch());
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    assert_eq!(
        hosted.gone(Remover::Registrant(&maker), URL, seen),
        Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned()
        ))
    );
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the write reached its telling");
    assert_eq!((store.by_key(URL), store.epoch()), (Some(row), epoch));
}

/// The replica's own refresh of a worker's labels is a patch of its
/// turn, under the turn's term. One whose term ends in the patch's own
/// telling writes nothing, and is refused as a turn that does not lead,
/// never as a row that moved.
#[test]
fn a_replicas_own_refresh_whose_term_ends_at_its_telling_is_refused_as_not_leading() {
    const HTTP: &str = "http://10.0.0.7:8000";
    let (store, door, journal) = leaving_at_a_row();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let built = |tier: &str| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(HTTP).model(ModelCard::new("m"));
        Arc::new(worker.label("tier", tier).build())
    };
    let first = built("gold");
    hosted
        .found_built(&Registrant::Control, Precondition::Absent, &first, None)
        .expect("registered");
    let row = store.by_key(HTTP).expect("the registered row");
    let epoch = store.epoch();
    journal.armed.store(BY_A_RISE, Ordering::Relaxed);
    assert_eq!(
        hosted.refreshed(standing(&row), &built("silver"), &first),
        Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned()
        ))
    );
    let armed = journal.armed.load(Ordering::Relaxed);
    assert_eq!(armed, 0, "the patch reached its telling");
    assert_eq!((store.by_key(HTTP), store.epoch()), (Some(row), epoch));
}

pub(crate) use crate::document::operator::test_support::Logged;
