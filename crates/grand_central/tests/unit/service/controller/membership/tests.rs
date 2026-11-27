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

const URL: &str = "grpc://10.0.0.5:8000";
const TERM: u64 = 7;
/// The bound of a base and of an epoch by its number, 2^39: a case reads
/// it here and not through the door's constant, so that constant's value
/// stands on a case.
const BOUND: u64 = 1 << 39;
/// The copies of one call that [`at_once`] lets go together.
const CALLERS: usize = 4;

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

/// The replica's worker objects hold a call's write when the call
/// answers. No task follows the store here, and a decision follows
/// nothing: each is the in-process front's over the list the holder
/// lends as the door's last call left it. A worker registered through
/// the door is named by a decision behind the call. One whose
/// deregistration was acknowledged is named by none behind it, and
/// neither is one whose drain was acknowledged: a drain is written on
/// the worker object first. The replica's own writer shares the door's
/// turn: a worker it drains is named by none behind its answer.
#[tokio::test]
async fn a_decision_behind_an_acknowledged_call_reads_the_fleet_the_call_left() {
    use std::collections::BTreeSet;

    use crate::service::controller::decision_slots::tests::{frame, lent_pick, preset};

    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let door = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
    let door = Arc::new(door);
    store
        .apply(preset("round_robin", 1))
        .expect("above the unbound start");
    // The workers twelve decisions name, with no follow of the case's.
    let named = || -> BTreeSet<WorkerId> {
        (1..=12)
            .filter_map(|seed| lent_pick(&store, &objects, frame(seed, u64::from(seed))))
            .collect()
    };
    let ids: Vec<WorkerId> = ["w0", "w1", "w2"]
        .iter()
        .map(|name| {
            let asked = ask(unchecked(&format!("grpc://{name}:8000")), Mode::Add);
            let told = door
                .register(&operator(), under(TERM), &asked)
                .expect("the door takes the worker");
            WorkerId(row_of(&told).id)
        })
        .collect();
    assert_eq!(named(), ids.iter().copied().collect());
    let leave = |id: WorkerId, drain: bool| {
        let row = WorkersStore::get(&*store, id).expect("the row stands");
        let request = proto::DeregisterWorkerRequest {
            worker_id: id.0,
            revision: row.revision,
            drain,
        };
        door.deregister(&operator(), under(TERM), &request)
            .expect("the door acknowledges");
    };
    // An acknowledged deregistration: no decision behind it names the
    // worker.
    leave(ids[0], false);
    assert_eq!(named(), BTreeSet::from([ids[1], ids[2]]));
    // And an acknowledged drain.
    leave(ids[1], true);
    assert_eq!(named(), BTreeSet::from([ids[2]]));
    // The last worker is drained through the replica's own writer,
    // over the same door: no decision behind its answer names a worker.
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let last = WorkersStore::get(&*store, ids[2]).expect("the row stands");
    let by = Remover::Registrant(&last.maker);
    assert_eq!(
        hosted.drained(by, "grpc://w2:8000", standing(&last)),
        Ok(MembershipWriteOutcome::Draining(ids[2]))
    );
    assert_eq!(named(), BTreeSet::new());
}

/// A call's write of a row builds the row's worker object where it is
/// written, and what the table still leads reaches the object behind
/// the write: the registry tells a registration, then the status the
/// connect signal moved, as the log tells the two. So it is for a call
/// of one row, for each rank of a wide call, and for a row the
/// replica's own writer lays. A door that wrote the table alone would
/// build each object behind its connect signal, ready from the start,
/// and the registry would tell no status.
#[test]
fn a_call_builds_a_rows_object_at_its_write_and_its_status_follows() {
    use smg_worker::registry::event::WorkerEvent;

    use crate::store::HealthPolicy;

    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let mut events = objects.registry().subscribe_events();
    let door = Arc::new(MembershipCalls::based(store, feed.log, 0, objects));
    // What the registry told since the last read, in its order.
    let mut told = move || -> Vec<&'static str> {
        let mut told = Vec::new();
        while let Ok(event) = events.try_recv() {
            told.push(match event {
                WorkerEvent::Registered { .. } => "registered",
                WorkerEvent::Replaced { .. } => "replaced",
                WorkerEvent::Removed { .. } => "removed",
                WorkerEvent::StatusChanged { .. } => "status",
            });
        }
        told
    };
    door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("one row");
    assert_eq!(told(), ["registered", "status"]);
    let ranks = proto::WorkerSpec {
        dp_size: 2,
        ..unchecked("grpc://wide:8000")
    };
    door.register(&operator(), under(TERM), &ask(ranks, Mode::Add))
        .expect("two ranks");
    // A rank's connect signal is written behind its own row: the first
    // rank's status reaches its object ahead of the second rank's write.
    assert_eq!(told(), ["registered", "status", "registered", "status"]);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let own = "grpc://own:8000";
    let wrote = hosted.found_one(&found("pods/a"), own, &|_| WorkerFacts {
        key: own.to_owned(),
        engine: own.to_owned(),
        health: HealthPolicy {
            probes: false,
            ..HealthPolicy::default()
        },
        ..found_facts()
    });
    assert!(
        matches!(wrote, Ok(MembershipWriteOutcome::Registered(_))),
        "{wrote:?}"
    );
    assert_eq!(told(), ["registered", "status"]);
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

#[test]
fn a_registration_writes_the_row_its_caller_stated_and_answers_it_pending() {
    let (store, door) = door();
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let id = numbered(URL, 0);
    let stored = store.get(id).expect("the row is in the store");
    assert_eq!(
        told,
        proto::WorkerEvent {
            seq: 2,
            membership_epoch: 1,
            event: Some(Event::Registered(super::worker_wire::row(&stored))),
            term: TERM,
        }
    );
    let row = row_of(&told);
    assert_eq!(
        (row.id, row.incarnation, row.revision, row.status),
        (id.0, life(TERM, 0), 1, proto::WorkerStatus::Pending as i32)
    );
    assert_eq!(
        (stored.status, stored.revision, store.epoch()),
        (WorkerStatus::Pending, 1, 1)
    );
    // No mode reads as an add.
    let other = "grpc://10.0.0.6:8000";
    let unset = proto::RegisterWorkerRequest {
        spec: Some(spec(other)),
        ..Default::default()
    };
    let told = door
        .register(&operator(), under(TERM), &unset)
        .expect("registered");
    assert!(matches!(told.event, Some(Event::Registered(_))));
    assert_eq!((told.membership_epoch, told.seq, told.term), (2, 3, TERM));
}

#[test]
fn a_second_add_is_refused_and_writes_nothing() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let refused = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect_err("the URL stands");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!(
        refused.message(),
        format!(
            "the URL is registered as worker {:016x} at revision 1",
            numbered(URL, 0).0
        )
    );
    assert!(!refused.message().contains("10.0.0.5"));
    assert_eq!(store.epoch(), 1);
}

/// A call reads the row that stands off the worker objects, and no
/// row of the table. An add over a row that a writer outside the
/// holder left in the table is decided over no object. The store
/// refuses it, the holder lays nothing, and the call is answered as a
/// row that moved, the add again too. A cut laid whole lays the row,
/// and the add is then refused as an add over any row that stands.
/// None writes.
#[test]
fn an_add_over_a_row_written_around_the_holder_is_refused_and_lays_nothing() {
    // A row that names its engine, as every writer of the crate lays
    // one, and a row that names none: its ranks are asked under its
    // key, and it has none.
    for engine in [URL, ""] {
        let (store, door) = door();
        let facts = WorkerFacts {
            key: URL.to_owned(),
            engine: engine.to_owned(),
            incarnation: 1,
            wire_modes: WireModes::GRPC,
            models: vec!["model-a".to_owned()],
            ..WorkerFacts::default()
        };
        store.upsert_by(facts, Registrant::Control);
        let stood = (store.epoch(), store.by_key(URL));
        let refused = |code: Code, why: &str| {
            let refused = door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect_err(why);
            assert_eq!(refused.code(), code, "engine {engine:?}: {why}");
            assert_eq!((store.epoch(), store.by_key(URL)), stood);
        };
        for _ in 0..2 {
            refused(Code::Aborted, "decided over no object");
            assert!(door.objects.is_empty(), "a refused add lays nothing");
        }
        // The holder's pass over a cut's rows, run over the store's:
        // this store leads a term and refuses a restore of its own cut.
        lay_standing(&door.objects, &store);
        refused(Code::AlreadyExists, "the URL stands");
        assert_eq!(door.objects.len(), 1, "the cut laid it");
    }
}

/// A row that names no engine dials its key, and its engine's ranks
/// are asked under that key. Here such a row stands beside a rank that
/// names the key as its engine, both written through the holder's own
/// calls with facts no call of the door states: a
/// plain registration of the URL is refused as over any engine that
/// stands as ranks.
#[test]
fn a_row_that_names_no_engine_is_asked_for_its_ranks_under_its_key() {
    let (store, door) = door();
    let plain = WorkerFacts {
        key: URL.to_owned(),
        engine: String::new(),
        incarnation: 1,
        wire_modes: WireModes::GRPC,
        models: vec!["model-a".to_owned()],
        ..WorkerFacts::default()
    };
    let rank = WorkerFacts {
        key: format!("{URL}@1"),
        engine: URL.to_owned(),
        dp_rank: 1,
        dp_size: 2,
        ..plain.clone()
    };
    for facts in [plain, rank] {
        let laid = door.objects.upsert(&store, facts, Registrant::Control);
        laid.expect("a row through the holder");
    }
    let refused = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect_err("its engine stands as ranks");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, super::AS_RANKS)
    );
}

#[test]
fn a_replace_wants_a_standing_row_and_makes_its_next_incarnation() {
    let (store, door) = door();
    let absent = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect_err("nothing to replace");
    assert_eq!(absent.code(), Code::NotFound);
    // What a new row names is no rule of a call that makes none: a
    // REPLACE that leaves the wire and the role unsaid is not found
    // either, and so is one that leaves the role alone unsaid.
    let bare = proto::WorkerSpec {
        url: URL.to_owned(),
        models: vec!["model-a".to_owned()],
        ..Default::default()
    };
    let roleless = proto::WorkerSpec {
        wire: proto::WireMode::Grpc as i32,
        ..bare.clone()
    };
    for unsaid in [bare.clone(), roleless] {
        let absent = door
            .register(&operator(), under(TERM), &ask(unsaid, Mode::Replace))
            .expect_err("nothing to replace");
        assert_eq!(
            (absent.code(), absent.message()),
            (Code::NotFound, "no worker is registered under the URL")
        );
    }
    // A field past its bounds is refused as itself, found or not, and
    // whatever the call leaves unsaid: a field read ahead of the wire
    // (the models) and one read after it (the labels), each with the
    // wire and the role stated and with both left unsaid.
    let wide = |over: proto::WorkerSpec| proto::WorkerSpec {
        models: vec!["m".repeat(257)],
        ..over
    };
    let labelled = |over: proto::WorkerSpec| proto::WorkerSpec {
        labels: (0..129)
            .map(|n| (format!("label-{n}"), String::new()))
            .collect(),
        ..over
    };
    let models = "spec.models: a model id is at most 256 bytes";
    let labels = "spec.labels: a row holds at most 128 labels";
    for (past, field) in [
        (wide(spec(URL)), models),
        (wide(bare.clone()), models),
        (labelled(spec(URL)), labels),
        (labelled(bare.clone()), labels),
    ] {
        let refused = door
            .register(&operator(), under(TERM), &ask(past, Mode::Replace))
            .expect_err("past a bound");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::InvalidArgument, field)
        );
    }
    // An UPSERT of the same would make a row, and is held to its names.
    let unnamed = door
        .register(&operator(), under(TERM), &ask(bare, Mode::Upsert))
        .expect_err("a new row names its wire");
    assert_eq!(unnamed.code(), Code::InvalidArgument);
    assert_eq!(store.epoch(), 0);

    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let old = store.get(numbered(URL, 0)).expect("the first life");
    door.connect_row(old.id, old.revision, TERM);
    let old = store.get(old.id).expect("the first life, ready");

    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect("replaced");
    // Numbered at the epoch the two writes above left.
    let new = store.get(numbered(URL, 2)).expect("the second life");
    assert_eq!(store.get(old.id), None);
    assert_eq!(
        told.event,
        Some(Event::Replaced(proto::Replaced {
            old: Some(super::worker_wire::row(&old)),
            new: Some(super::worker_wire::row(&new)),
            // The same role, wire and models: the status came over.
            state_carried: true,
        }))
    );
    assert_eq!((new.status, new.revision), (WorkerStatus::Ready, 1));
}

#[test]
fn many_short_model_ids_register_and_a_list_past_the_row_size_is_refused() {
    let (store, door) = door();
    let heavy = proto::WorkerSpec {
        models: vec!["m".repeat(256); 64],
        ..spec(URL)
    };
    let status = door
        .register(&operator(), under(TERM), &ask(heavy, Mode::Add))
        .expect_err("past the row's size");
    assert_eq!(
        (status.code(), status.message()),
        (
            Code::InvalidArgument,
            "spec: a row is at most 16384 bytes on the wire"
        )
    );
    assert_eq!(store.epoch(), 0);

    let short = proto::WorkerSpec {
        models: (0..129).map(|n| n.to_string()).collect(),
        ..spec(URL)
    };
    let wrote = door
        .register(&operator(), under(TERM), &ask(short.clone(), Mode::Add))
        .expect("the ids fit");
    assert_eq!(row_of(&wrote).models, short.models);
    assert_eq!(
        store.by_key(URL).expect("the row").facts.models,
        short.models
    );
}

#[test]
fn a_replacement_says_carried_only_where_the_status_came_over() {
    let (store, door) = door();
    let waited = |spec, mode| proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(spec, mode)
    };
    door.register(&operator(), under(TERM), &waited(unchecked(URL), Mode::Add))
        .expect("registered");
    // The same role, wire and models: the ready status came over.
    let told = door
        .register(
            &operator(),
            under(TERM),
            &waited(unchecked(URL), Mode::Replace),
        )
        .expect("replaced");
    let Some(Event::Replaced(same)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(same.state_carried);
    // Another model: a fresh row, ready by its own connect signal, so
    // nothing came over.
    let other = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..unchecked(URL)
    };
    let told = door
        .register(&operator(), under(TERM), &waited(other, Mode::Replace))
        .expect("replaced");
    let Some(Event::Replaced(fresh)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!fresh.state_carried);
    assert_eq!(
        fresh.new.map(|new| new.status),
        Some(proto::WorkerStatus::Ready as i32)
    );
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn an_upsert_of_equal_facts_writes_nothing_and_answers_the_row_unmoved() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("registered");
    let stood = store.get(numbered(URL, 0)).expect("the row");
    for _ in 0..3 {
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
            .expect("answered");
        assert_eq!(
            told,
            proto::WorkerEvent {
                seq: 2,
                membership_epoch: 1,
                event: Some(Event::Relabelled(super::worker_wire::row(&stood))),
                term: TERM,
            }
        );
    }
    assert_eq!(store.epoch(), 1);
    assert_eq!(store.get(stood.id), Some(stood));
}

#[test]
fn an_upsert_changes_a_standing_row_in_place_or_makes_a_new_incarnation() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let first = numbered(URL, 0);

    // A label: the same worker, its facts moved.
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    let told = door
        .register(
            &operator(),
            under(TERM),
            &ask(labelled.clone(), Mode::Upsert),
        )
        .expect("relabelled");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let row = row_of(&told);
    assert_eq!(
        (row.id, row.incarnation, row.revision),
        (first.0, life(TERM, 0), 2)
    );
    assert_eq!(row.labels.get("tier").map(String::as_str), Some("gold"));
    assert_eq!(told.membership_epoch, 2);

    // The role: another worker under the URL, so a new incarnation.
    let decode = proto::WorkerSpec {
        role: proto::Role::Decode as i32,
        ..labelled
    };
    let told = door
        .register(&operator(), under(TERM), &ask(decode.clone(), Mode::Upsert))
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a change of role is a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    let new = replaced.new.expect("the new row");
    assert_eq!(
        (new.id, new.incarnation, new.revision),
        (numbered(URL, 2).0, life(TERM, 2), 1)
    );
    assert_eq!(replaced.old.map(|old| old.id), Some(first.0));
    assert_eq!(store.get(first), None);

    // The models alone, then the wire alone, then what the worker
    // states for a request's hard demands alone, then whether it takes
    // images alone: each is another worker under the URL too.
    let remodelled = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..decode
    };
    let rewired = proto::WorkerSpec {
        wire: proto::WireMode::Http as i32,
        ..remodelled.clone()
    };
    let stating = proto::WorkerSpec {
        demand_facts: Some(proto::DemandFacts {
            runtime: 7,
            ..Default::default()
        }),
        ..rewired.clone()
    };
    let seeing = proto::WorkerSpec {
        vision: Some(true),
        ..stating.clone()
    };
    let changes = [
        ("models", remodelled),
        ("wire", rewired),
        ("demand facts", stating.clone()),
        ("whether it takes images", seeing.clone()),
    ];
    for (what, changed) in changes {
        let stood = store.by_key(URL).expect("the row");
        let told = door
            .register(&operator(), under(TERM), &ask(changed, Mode::Upsert))
            .expect("replaced");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a change of {what} is a replacement: {told:?}");
        };
        assert_eq!(replaced.old.map(|old| old.id), Some(stood.id.0), "{what}");
        assert_eq!(store.get(stood.id), None, "{what}");
    }

    // The row holds the block and the bit, and the same registration
    // again finds equal facts: it writes nothing. Neither does one that
    // states the block and leaves the bit out, since a bit left out
    // keeps the row's.
    let stood = store.by_key(URL).expect("the row");
    assert_eq!(stood.facts.demand_facts.runtime, 7);
    assert!(stood.facts.vision);
    for again in [seeing, stating] {
        let told = door
            .register(&operator(), under(TERM), &ask(again, Mode::Upsert))
            .expect("answered");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        assert_eq!(store.by_key(URL).as_ref(), Some(&stood));
    }
}

#[test]
fn an_upsert_of_the_same_models_in_another_order_is_made_in_place() {
    let (store, door) = door();
    let listing = |models: &[&str]| proto::WorkerSpec {
        models: models.iter().map(|&model| model.to_owned()).collect(),
        ..unchecked(URL)
    };
    let two = listing(&["model-a", "model-b"]);
    let told = door
        .register(&operator(), under(TERM), &ask(two.clone(), Mode::Add))
        .expect("registered");
    let first = row_of(&told);
    // Unchecked, the row is ready by its connect signal: a state to
    // carry.
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
    // The same two in the other order, then one of them named twice:
    // the same worker each time, and the row tells the stated list.
    for stated in [
        &["model-b", "model-a"][..],
        &["model-a", "model-b", "model-a"][..],
    ] {
        let told = door
            .register(
                &operator(),
                under(TERM),
                &ask(listing(stated), Mode::Upsert),
            )
            .expect("reordered");
        let Some(Event::Relabelled(row)) = told.event else {
            panic!("a reorder is made in place: {told:?}");
        };
        assert_eq!((row.id, row.incarnation), (first.id, first.incarnation));
        assert_eq!(row.models, stated);
        assert_eq!(row.status, proto::WorkerStatus::Ready as i32);
    }
    // A replacement that names the same set carries the status too.
    let told = door
        .register(&operator(), under(TERM), &ask(two, Mode::Replace))
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(replaced.state_carried);
}

#[test]
fn a_registration_with_its_checks_off_is_ready_at_once() {
    let (store, door) = door();
    // Not waited for: the answer is the pending row, and the store has
    // moved it to ready by the next read, at the revision the answer
    // gave: a status change moves none.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.revision, told.membership_epoch),
        (proto::WorkerStatus::Pending as i32, 1, 1)
    );
    let stored = store.get(numbered(URL, 0)).expect("the row");
    assert_eq!(
        (stored.status, stored.revision, store.epoch()),
        (WorkerStatus::Ready, 1, 2)
    );

    // Waited for: the answer is the row after its connect signal.
    let other = "grpc://10.0.0.6:8000";
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(other), Mode::Add)
    };
    let told = door
        .register(&operator(), under(TERM), &waited)
        .expect("registered");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.revision, told.membership_epoch),
        (proto::WorkerStatus::Ready as i32, 1, 4)
    );
    assert_eq!(row.health, proto::HealthFlag::Healthy as u32);

    // A checked row that turns its checks off is ready from that write.
    let third = "grpc://10.0.0.7:8000";
    door.register(&operator(), under(TERM), &ask(spec(third), Mode::Add))
        .expect("registered");
    door.register(
        &operator(),
        under(TERM),
        &ask(unchecked(third), Mode::Upsert),
    )
    .expect("relabelled");
    let stored = store.by_key(third).expect("the row");
    assert_eq!(stored.status, WorkerStatus::Ready);
}

#[test]
fn a_wait_for_a_checked_worker_and_a_request_out_of_order_are_refused_by_name() {
    let (store, door) = door();
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(spec(URL), Mode::Add)
    };
    let bare = proto::RegisterWorkerRequest::default();
    let unknown = proto::RegisterWorkerRequest {
        mode: 9,
        ..ask(spec(URL), Mode::Add)
    };
    let wide = ask(
        proto::WorkerSpec {
            dp_size: 65,
            ..spec(URL)
        },
        Mode::Add,
    );
    for (request, path) in [
        (waited, "wait_ready: "),
        (bare, "spec: "),
        (unknown, "mode: "),
        (wide, "spec.dp_size: "),
    ] {
        let refused = door
            .register(&operator(), under(TERM), &request)
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
        assert!(
            refused.message().starts_with(path),
            "{path}: {}",
            refused.message()
        );
    }
    assert_eq!(store.epoch(), 0);
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

/// A registration that states several ranks writes one row a rank under
/// the rank's URL and answers each as its write left it, at the one
/// place of the call's last write. Stated again it writes nothing and
/// answers each rank unmoved.
#[test]
fn a_registration_of_several_ranks_writes_a_row_a_rank_and_answers_each() {
    use proto::rank_row::Made;
    let (store, door) = door();
    let told = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect("registered");
    assert_eq!((told.membership_epoch, told.seq, told.term), (3, 4, TERM));
    let made = made_of(&told);
    assert_eq!(made.len(), 3);
    for (rank, made) in (0..3_u16).zip(&made) {
        let stored = store
            .by_key(&format!("{URL}@{rank}"))
            .expect("the rank's row");
        assert_eq!(*made, Made::Registered(super::worker_wire::row(&stored)));
        let facts = &stored.facts;
        assert_eq!(
            (facts.engine.as_str(), facts.dp_rank, facts.dp_size),
            (URL, rank, 3)
        );
        assert_eq!((stored.status, stored.revision), (WorkerStatus::Pending, 1));
    }

    let again = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Upsert))
        .expect("unmoved");
    assert_eq!(
        (again.membership_epoch, again.seq, store.epoch()),
        (3, 4, 3)
    );
    for (rank, made) in made_of(&again).iter().enumerate() {
        let stored = store
            .by_key(&format!("{URL}@{rank}"))
            .expect("the rank's row");
        assert_eq!(*made, Made::Relabelled(super::worker_wire::row(&stored)));
    }
}

/// The modes hold for the engine. An add is refused where its ranks
/// stand or the engine stands in another form. A replacement wants every
/// stated rank and makes each its next life. An upsert lays each rank,
/// and the engine's rows it does not state leave with it.
#[test]
fn the_modes_of_a_registration_of_several_ranks_hold_for_the_engine() {
    use proto::rank_row::Made;
    let (store, door) = door();
    let register =
        |spec: proto::WorkerSpec, mode| door.register(&operator(), under(TERM), &ask(spec, mode));
    let rank = |rank: u16| store.by_key(&format!("{URL}@{rank}"));

    let none = register(wide(URL, 2), Mode::Replace).expect_err("no rank stands");
    assert_eq!(none.code(), Code::NotFound);
    register(spec(URL), Mode::Add).expect("the plain row");
    let other_form = register(wide(URL, 2), Mode::Add).expect_err("the plain row stands");
    assert_eq!(other_form.code(), Code::AlreadyExists);
    assert_eq!(store.epoch(), 1);

    // A worker that comes back ranked loses its plain row.
    let plain = store.by_key(URL).expect("the plain row");
    let told = register(wide(URL, 2), Mode::Upsert).expect("ranked");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Registered(_), Made::Registered(_), Made::Removed(left)]
                if left.worker_id == plain.id.0
        ),
        "{made:?}"
    );
    assert!(store.by_key(URL).is_none());
    assert_eq!((store.epoch(), told.membership_epoch), (4, 4));

    let stands = register(wide(URL, 2), Mode::Add).expect_err("a rank stands");
    assert_eq!(stands.code(), Code::AlreadyExists);
    let short = register(wide(URL, 3), Mode::Replace).expect_err("rank two is absent");
    assert_eq!(short.code(), Code::NotFound);
    assert_eq!(store.epoch(), 4);

    // A replacement is the next life of each rank.
    let old = [0, 1].map(|at| rank(at).expect("a rank"));
    let told = register(wide(URL, 2), Mode::Replace).expect("replaced");
    for (old, made) in old.iter().zip(made_of(&told)) {
        let Made::Replaced(replaced) = made else {
            panic!("{made:?}");
        };
        assert_eq!(replaced.old.map(|row| row.id), Some(old.id.0));
        assert_ne!(replaced.new.map(|row| row.id), Some(old.id.0));
        assert!(!replaced.state_carried);
    }

    // Wider, then narrower: the rank it no longer states leaves.
    let told = register(wide(URL, 3), Mode::Upsert).expect("wider");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [
                Made::Relabelled(_),
                Made::Relabelled(_),
                Made::Registered(_)
            ]
        ),
        "{made:?}"
    );
    let third = rank(2).expect("rank two");
    let told = register(wide(URL, 2), Mode::Upsert).expect("narrower");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(_), Made::Relabelled(_), Made::Removed(left)]
                if left.worker_id == third.id.0
        ),
        "{made:?}"
    );
    assert!(rank(2).is_none());
    assert_eq!(rank(0).map(|row| row.facts.dp_size), Some(2));
}

/// A registration of one rank for an engine that stands as ranks is
/// refused by name in every mode and writes nothing: a plain row beside
/// its ranks would be the engine twice. The way off is a rank at a time.
#[test]
fn a_registration_of_one_rank_over_an_engine_that_stands_as_ranks_is_refused() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
        .expect("ranks");
    // Under the engine's URL and under a rank's own: one rank laid
    // alone would offer what the engine's other rank does not.
    let rank = format!("{URL}@0");
    for mode in [Mode::Add, Mode::Replace, Mode::Upsert] {
        for url in [URL, rank.as_str()] {
            let refused = door
                .register(&operator(), under(TERM), &ask(spec(url), mode))
                .expect_err("the engine stands as ranks");
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::FailedPrecondition, super::AS_RANKS),
                "{mode:?} {url}"
            );
        }
    }
    assert_eq!(store.epoch(), 2);

    for at in 0..2 {
        let row = store.by_key(&format!("{URL}@{at}")).expect("a rank");
        let leave = proto::DeregisterWorkerRequest {
            worker_id: row.id.0,
            revision: row.revision,
            drain: false,
        };
        door.deregister(&operator(), under(TERM), &leave)
            .expect("the rank leaves");
    }
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("the plain row");
    assert!(store.by_key(URL).is_some());
}

/// A one-rank replacement or upsert over ranks another maker laid is
/// told who made them, ahead of the engine's form, as the wide call
/// tells it. An add asks no maker.
#[test]
fn a_registration_of_one_rank_over_another_makers_ranks_is_told_the_maker_first() {
    let (store, door) = door();
    door.upsert(
        &operator(),
        &found(URL),
        TERM,
        &ask(wide(URL, 2), Mode::Add),
    )
    .expect("a provider's ranks");
    let rank = format!("{URL}@0");
    let register = |spec: proto::WorkerSpec, mode| {
        door.register(&operator(), under(TERM), &ask(spec, mode))
            .expect_err("ranks of another maker")
    };
    for mode in [Mode::Replace, Mode::Upsert] {
        for asked in [spec(URL), spec(&rank), wide(URL, 2)] {
            let url = asked.url.clone();
            let refused = register(asked, mode);
            assert_eq!(refused.code(), Code::PermissionDenied, "{mode:?} {url}");
        }
    }
    for url in [URL, rank.as_str()] {
        let refused = register(spec(url), Mode::Add);
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::FailedPrecondition, super::AS_RANKS),
            "{url}"
        );
    }
    assert_eq!(store.epoch(), 2);
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

/// A wide call that loses its lead between two ranks is refused. The
/// rank written before the refusal stands, and its write leaves its
/// audit line ahead of the refusal's.
#[test]
fn a_term_lost_between_two_ranks_leaves_the_first_rank_and_its_audit_line() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(0),
    });
    let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));

    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    assert!(store.by_key(&format!("{URL}@0")).is_some());
    assert!(store.by_key(&format!("{URL}@1")).is_none());

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let [written, refusal] = lines.as_slice() else {
        panic!("one write's line and the refusal's: {logged}");
    };
    assert!(
        written.contains("membership write") && written.contains("made=\"registered\""),
        "{written}"
    );
    assert!(refusal.contains("membership call refused"), "{refusal}");
    assert!(!logged.contains(URL), "{logged}");
}

/// A door whose journal loses the lead when told, with `width` ranks of
/// [`URL`] registered under a lead that holds.
fn losing(width: u32) -> (Arc<MemoryStore>, MembershipCalls, Arc<LosesTheLead>) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(usize::MAX),
    });
    let handed = Arc::clone(&journal);
    let store = Arc::new(MemoryStore::journaled(handed).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    door.register(&operator(), under(TERM), &ask(wide(URL, width), Mode::Add))
        .expect("the ranks registered under a held lead");
    (store, door, journal)
}

/// A lead lost behind a rank's write in place refuses the next rank's.
/// The term is asked ahead of every write, whatever it makes of its
/// row.
#[test]
fn a_term_lost_behind_a_write_in_place_stops_the_next_ranks_write() {
    let (store, door, journal) = losing(2);
    journal.holds.store(0, Ordering::Relaxed);
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..wide(URL, 2)
    };
    let refused = door
        .register(&operator(), under(TERM), &ask(relabelled, Mode::Upsert))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    let labels = |rank: u32| {
        let row = store.by_key(&format!("{URL}@{rank}")).expect("the rank");
        row.facts.labels.len()
    };
    assert_eq!((labels(0), labels(1)), (1, 0));
}

/// A lead lost behind the last rank that stays refuses the removal of
/// the rank a narrower registration no longer states.
#[test]
fn a_term_lost_ahead_of_a_removal_leaves_the_rank_that_would_leave() {
    let (store, door, journal) = losing(3);
    journal.holds.store(1, Ordering::Relaxed);
    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Upsert))
        .expect_err("the lead was lost ahead of the removal");
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

/// A door whose lead was lost behind the first of an add's three ranks
/// and came back under a later term: the next leader's door, over the
/// row the first attempt left.
fn led_again_behind_a_lost_add() -> (Arc<MemoryStore>, MembershipCalls) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(0),
    });
    let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    let lost = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (lost.code(), lost.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    (store, door)
}

/// An add that lost its lead behind its first rank is the same add at
/// the next leader's door. The rank it wrote answers unmoved, the
/// others are registered, and each rank has its one row. Called once
/// more, the engine stands whole: the add is refused by the first
/// rank's row.
#[test]
fn an_add_called_again_behind_a_lost_lead_lays_the_ranks_it_left() {
    use proto::rank_row::Made;
    let (store, door) = led_again_behind_a_lost_add();
    let first = store.by_key(&format!("{URL}@0")).expect("the first rank");
    let add = || door.register(&operator(), under(TERM + 2), &ask(wide(URL, 3), Mode::Add));

    let told = add().expect("the same add");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(stood), Made::Registered(_), Made::Registered(_)]
                if stood.id == first.id.0
        ),
        "{made:?}"
    );
    assert_eq!(store.by_key(&format!("{URL}@0")).as_ref(), Some(&first));
    assert_eq!((store.epoch(), store.all_rows().1.len()), (3, 3));

    let whole = add().expect_err("the engine stands whole");
    assert_eq!(whole.code(), Code::AlreadyExists);
    assert_eq!(
        whole.message(),
        format!(
            "the URL is registered as worker {:016x} at revision {}",
            first.id.0, first.revision
        )
    );
    assert_eq!(store.epoch(), 3);
}

/// The same add with another label than the standing rank's is refused
/// by that rank's row and writes nothing: an add lays the rest by what
/// stands alone.
#[test]
fn an_add_called_again_with_another_label_is_refused_and_writes_nothing() {
    let (store, door) = led_again_behind_a_lost_add();
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..wide(URL, 3)
    };
    let refused = door
        .register(&operator(), under(TERM + 2), &ask(labelled, Mode::Add))
        .expect_err("the first rank stands with other labels");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
}

/// A registration states at most 64 ranks. At the bound, with every rank
/// replaced by rows near their size bound, the answer stays inside a
/// 4 MiB frame: a replacement tells two rows.
#[test]
fn a_registration_states_at_most_64_ranks_and_its_widest_answer_fits_a_frame() {
    use prost::Message as _;
    let (store, door) = door();
    let past = door
        .register(&operator(), under(TERM), &ask(wide(URL, 65), Mode::Add))
        .expect_err("past the bound");
    assert_eq!(
        (past.code(), past.message()),
        (
            Code::InvalidArgument,
            "spec.dp_size: a registration states at most 64 ranks"
        )
    );
    assert_eq!(store.epoch(), 0);

    let heavy = proto::WorkerSpec {
        models: vec!["m".repeat(256); 60],
        ..wide(URL, 64)
    };
    door.register(&operator(), under(TERM), &ask(heavy.clone(), Mode::Add))
        .expect("64 ranks");
    let told = door
        .register(&operator(), under(TERM), &ask(heavy, Mode::Replace))
        .expect("64 ranks replaced");
    let made = made_of(&told);
    assert_eq!(made.len(), 64);
    assert!(made
        .iter()
        .all(|made| matches!(made, proto::rank_row::Made::Replaced(_))));
    let bytes = told.encoded_len();
    assert!(bytes > 1 << 20, "{bytes}");
    assert!(bytes < 4 << 20, "{bytes}");
}

/// Each entry of the answer is the event a watcher is told of that row:
/// the log's entries of one call, as a watch tells them, read beside
/// the answer. A rank whose checks are off is answered as its write left
/// it, and its connect signal is the watcher's next event. The answer's
/// place is the call's last write.
#[test]
fn each_entry_of_an_answer_of_several_ranks_is_the_event_a_watcher_is_told() {
    use proto::rank_row::Made;
    let (_store, door) = door();
    let watched = |after: Position| {
        let mut entries = Vec::new();
        door.log
            .read(after, 64, &mut entries)
            .expect("the log holds the call's writes");
        entries
            .iter()
            .filter_map(super::worker_wire::member)
            .collect::<Vec<_>>()
    };
    let entries = |told: &proto::WorkerEvent| match told.event.clone().expect("an event") {
        Event::Ranks(ranks) => ranks.rows,
        other => panic!("no ranks in {other:?}"),
    };
    let ranked = |mode| {
        let spec = proto::WorkerSpec {
            dp_size: 2,
            ..unchecked(URL)
        };
        let before = door.place();
        let told = door
            .register(&operator(), under(TERM), &ask(spec, mode))
            .expect("the ranks are laid");
        let events = watched(before);
        let last = events.last().expect("the call's last write");
        assert_eq!(
            (told.seq, told.membership_epoch, told.term),
            (last.seq, last.membership_epoch, last.term)
        );
        // The connect signals are the watcher's own events.
        let rows: Vec<_> = events
            .iter()
            .filter_map(|event| event.event.clone())
            .filter(|event| !matches!(event, Event::Status(_)))
            .collect();
        let answered = entries(&told);
        assert_eq!(rows.len(), answered.len());
        for (event, entry) in rows.into_iter().zip(&answered) {
            assert_eq!(super::worker_wire::rank_row(event).as_ref(), Some(entry));
        }
        answered
    };

    // A worker that comes back ranked: two rows registered, the plain
    // row removed.
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("the plain row");
    let made = ranked(Mode::Upsert);
    assert!(
        matches!(
            made.iter()
                .map(|entry| entry.made.as_ref())
                .collect::<Vec<_>>()[..],
            [
                Some(Made::Registered(_)),
                Some(Made::Registered(_)),
                Some(Made::Removed(_))
            ]
        ),
        "{made:?}"
    );
    // A replacement of ranks that are ready hands their status on.
    let made = ranked(Mode::Replace);
    assert_eq!(made.len(), 2);
    for entry in &made {
        let Some(Made::Replaced(replaced)) = &entry.made else {
            panic!("{entry:?}");
        };
        assert!(replaced.state_carried, "{replaced:?}");
    }
}

#[test]
fn an_answer_carries_the_place_of_its_calls_write_under_that_places_term() {
    let (store, door) = door();
    // Not waited: the registration as its write left it. The connect
    // signal's write is the next place.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    assert_eq!((told.seq, told.term, told.membership_epoch), (2, TERM, 1));
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Pending as i32);
    // Waited: the row after its connect signal, at that write's place.
    let other = "grpc://10.0.0.6:8000";
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(other), Mode::Add)
    };
    let told = door
        .register(&operator(), under(TERM), &waited)
        .expect("registered");
    assert_eq!((told.seq, told.term, told.membership_epoch), (5, TERM, 4));
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Ready as i32);
    // A call that writes nothing answers the place the log stands at:
    // here the start of the next term.
    store::LeasesStore::set_term(store.as_ref(), TERM + 1);
    let told = door
        .register(
            &operator(),
            under(TERM + 1),
            &ask(unchecked(other), Mode::Upsert),
        )
        .expect("answered");
    assert_eq!(
        (told.seq, told.term, told.membership_epoch),
        (6, TERM + 1, 4)
    );
    // And the next write is made under that term.
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row_of(&told).id,
        revision: row_of(&told).revision,
        drain: true,
    };
    let told = door
        .deregister(&operator(), under(TERM + 1), &drain)
        .expect("draining");
    assert_eq!(
        (told.seq, told.term, told.membership_epoch),
        (7, TERM + 1, 5)
    );
}

/// A provider's maker for one worker.
fn found(key: &str) -> Registrant {
    Registrant::provider("cluster", key).expect("a whole maker")
}

#[test]
fn a_row_keeps_the_maker_its_registration_gave_it() {
    let (store, door) = door();
    // A control call's row says no maker on the wire.
    let by_hand = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(by_hand.maker, None);
    let held = store.get(WorkerId(by_hand.id)).expect("the row");
    assert_eq!(held.maker, Registrant::Control);
    // A provider's registration makes the row its own, and the answer
    // shows whose it is.
    let maker = found("pods/a");
    let other = "grpc://10.0.0.6:8000";
    let made = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert))
            .expect("found"),
    );
    let shown = Some(proto::Maker {
        provider: "cluster".to_owned(),
        key: "pods/a".to_owned(),
    });
    assert_eq!(made.maker, shown);
    // Its refresh in place and its new incarnation keep the maker. The
    // refresh comes from a maker made anew: two makers are one where
    // they agree whole, whichever copy a caller holds.
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(other)
    };
    let anew = found("pods/a");
    let in_place = row_of(
        &door
            .upsert(&operator(), &anew, TERM, &ask(relabelled, Mode::Upsert))
            .expect("refreshed"),
    );
    assert_eq!((in_place.id, &in_place.maker), (made.id, &shown));
    let again = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Replace))
            .expect("replaced"),
    );
    assert_ne!(again.id, made.id);
    assert_eq!(again.maker, shown);
    let held = store.get(WorkerId(again.id)).expect("the row");
    assert_eq!(held.maker, maker);
}

#[test]
fn a_caller_that_did_not_make_a_row_changes_nothing_of_it() {
    let (store, door) = door();
    let maker = found("pods/a");
    let made = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
            .expect("found"),
    );
    let epoch = store.epoch();
    let refused = |answer: Result<proto::WorkerEvent, Status>, whose: &str| {
        let status = answer.expect_err("refused");
        assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
        assert!(status.message().contains(whose), "{status:?}");
        assert!(!status.message().contains("10.0.0"), "no URL: {status:?}");
    };
    // A control call on the provider's row: a registration over it, with
    // other facts and with equal ones, a patch, a drain and a removal.
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    for request in [
        ask(spec(URL), Mode::Replace),
        ask(relabelled, Mode::Upsert),
        ask(spec(URL), Mode::Upsert),
    ] {
        let answer = door.register(&operator(), under(TERM), &request);
        refused(answer, "provider cluster");
    }
    let patch = proto::PatchWorkerRequest {
        worker_id: made.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: made.revision,
    };
    refused(
        door.patch(&operator(), under(TERM), &patch),
        "provider cluster",
    );
    for drain in [true, false] {
        let leave = proto::DeregisterWorkerRequest {
            worker_id: made.id,
            revision: made.revision,
            drain,
        };
        refused(
            door.deregister(&operator(), under(TERM), &leave),
            "provider cluster",
        );
    }
    // The maker is tested ahead of the revision: a caller that did not
    // make the row is told so whatever revision it read.
    let stale = proto::PatchWorkerRequest {
        revision: made.revision + 7,
        ..patch.clone()
    };
    refused(
        door.patch(&operator(), under(TERM), &stale),
        "provider cluster",
    );
    // The same provider under another key is another worker's maker.
    let another = found("pods/b");
    refused(
        door.upsert(&operator(), &another, TERM, &ask(spec(URL), Mode::Upsert)),
        "provider cluster",
    );
    // Nothing was written, and the maker's own call goes through.
    assert_eq!(store.epoch(), epoch);
    let held = store.get(WorkerId(made.id)).expect("the row");
    assert_eq!(held.revision, made.revision);
    door.relabel(&operator(), &maker, &patch)
        .expect("the maker patches its row");

    // The other way round: a provider changes nothing of a control
    // call's row.
    let other = "grpc://10.0.0.6:8000";
    let by_hand = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
            .expect("registered"),
    );
    refused(
        door.upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert)),
        "a control call",
    );
    let leave = proto::DeregisterWorkerRequest {
        worker_id: by_hand.id,
        revision: by_hand.revision,
        drain: false,
    };
    refused(
        door.remove(&operator(), &maker, TERM, &leave),
        "a control call",
    );
    door.deregister(&operator(), under(TERM), &leave)
        .expect("the control call removes its row");
}

#[test]
fn a_row_holds_its_maker_inside_the_wires_bound() {
    use super::worker_wire;

    let (store, door) = door();
    let maker = found("pods/a");
    // Three labels at a value's bound and one that fills. The widest
    // fill a row of this maker has room for is registered; one byte more
    // is refused by the row's name, and so is a patch of the row at its
    // brim.
    let stated = |fill: usize| proto::WorkerSpec {
        labels: ["a", "b", "c"]
            .map(|key| (key.to_owned(), "v".repeat(4_096)))
            .into_iter()
            .chain([("fill".to_owned(), "v".repeat(fill))])
            .collect(),
        ..spec(URL)
    };
    let fits = |fill: usize| {
        worker_wire::registration(&stated(fill), |_| None).is_ok_and(|(facts, _)| {
            worker_wire::room(&facts, &facts.labels, &maker, "spec").is_ok()
        })
    };
    let widest = (0..=4_096usize)
        .rev()
        .find(|&fill| fits(fill))
        .expect("a fill the row has room for");
    let epoch = store.epoch();
    let past = door.upsert(
        &operator(),
        &maker,
        TERM,
        &ask(stated(widest + 1), Mode::Upsert),
    );
    let status = past.expect_err("past the bound");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().starts_with("spec:"), "{status:?}");
    assert_eq!(store.epoch(), epoch);
    let brim = row_of(
        &door
            .upsert(
                &operator(),
                &maker,
                TERM,
                &ask(stated(widest), Mode::Upsert),
            )
            .expect("at the bound"),
    );
    let more = proto::PatchWorkerRequest {
        worker_id: brim.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: brim.revision,
    };
    let status = door
        .relabel(&operator(), &maker, &more)
        .expect_err("past the bound");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().starts_with("labels:"), "{status:?}");
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

/// A standby builds its rows from the log. A fact the log's row does not
/// carry would leave it another row than its leader holds.
#[test]
fn a_fact_the_log_does_not_carry_is_refused_by_its_name_and_nothing_is_written() {
    use super::super::worker_wire;
    use crate::{
        engine::DemandFacts,
        store::{pairing::PairingDescriptor, EncodeHandoff, EncodeOffer},
    };
    type State = fn(&mut WorkerFacts);

    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let maker = found("pods/a");
    let epoch = store.epoch();
    let lost: [(&str, State); 7] = [
        ("row.pairing", |facts| {
            facts.pairing = Some(Arc::new(PairingDescriptor {
                protocol: Some("pd".to_owned()),
                ..PairingDescriptor::default()
            }));
        }),
        // An empty connector, which the wire's row reads as none.
        ("row.encode", |facts| {
            facts.encode = EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                connector: Some(String::new()),
                ..EncodeOffer::default()
            };
        }),
        // A connector's one bound is the row's.
        ("row", |facts| {
            facts.encode = EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                connector: Some("a".repeat(WorkerFacts::ROW_BYTES)),
                ..EncodeOffer::default()
            };
        }),
        ("row.zone", |facts| facts.zone = Some(String::new())),
        ("row.max_running", |facts| facts.max_running = Some(0)),
        ("row.kv_block_size", |facts| facts.kv_block_size = Some(0)),
        ("row.bootstrap_port", |facts| facts.bootstrap_port = Some(0)),
    ];
    for (fact, state) in lost {
        let answer = hosted.found_one(&maker, URL, &|_| {
            let mut facts = found_facts();
            state(&mut facts);
            facts
        });
        let Err(MembershipWriteRefused::Past(past)) = answer else {
            panic!("{fact}: wrong answer: {answer:?}")
        };
        assert_eq!(past.fact, fact);
    }
    assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

    // What the log carries is taken, and a standby reads it back whole:
    // what the worker states for a request's hard demands among it,
    // whether it takes images, its end of the encode handoff, its priority
    // and a drain window of its own. The offer's connector is the store's
    // text, held to no rule of its words: one that holds a blank and a
    // control character, past a name's bound.
    let stated = DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 32_768,
        decoding: 3,
    };
    let offered = EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: true,
        connector: Some(format!("connector a\u{7}{}", "b".repeat(256))),
        ..EncodeOffer::default()
    };
    let stating = |_: Option<&WorkerFacts>| WorkerFacts {
        demand_facts: stated,
        vision: true,
        encode: offered.clone(),
        priority: 5,
        drain_settle_secs: Some(30),
        ..found_facts()
    };
    hosted.found_one(&maker, URL, &stating).expect("found");
    let row = store.by_key(URL).expect("the found row");
    assert_eq!(row.facts.demand_facts, stated);
    assert!(row.facts.vision);
    assert_eq!(row.facts.encode, offered);
    let stated = (row.facts.priority, row.facts.drain_settle_secs);
    assert_eq!(stated, (5, Some(30)));
    let back = worker_wire::told_row(&worker_wire::row(&row), std::time::Instant::now());
    assert_eq!(back.map(|back| back.facts), Ok(row.facts));
}

/// The store reads a row as host-local by its reach alone: a row of the
/// host-local wire that names no gateway would stand in every gateway's
/// pool. A store held in process serves one gateway.
#[test]
fn a_host_local_wire_names_its_gateway_at_the_controllers_door() {
    let stated = |wire: WireModes, from: Reachability| {
        move |_: Option<&WorkerFacts>| WorkerFacts {
            wire_modes: wire,
            reachable_from: from,
            ..found_facts()
        }
    };
    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let maker = found("pods/a");
    let epoch = store.epoch();
    for wire in [WireModes::ZMQ, WireModes::GRPC.union(WireModes::ZMQ)] {
        let answer = hosted.found_one(&maker, URL, &stated(wire, Reachability::Any));
        let Err(MembershipWriteRefused::Past(past)) = answer else {
            panic!("{wire:?}: wrong answer: {answer:?}")
        };
        assert_eq!(past.fact, "row.wire");
    }
    assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

    // A row that names its gateway is in that gateway's pool alone.
    let one = Reachability::Gateway(GatewayId(7));
    hosted
        .found_one(&maker, URL, &stated(WireModes::ZMQ, one))
        .expect("found");
    let row = store.by_key(URL).expect("the found row");
    assert_eq!(row.facts.reachable_from, one);
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

#[test]
fn a_registration_is_laid_over_what_it_expects() {
    check_expectations(hosted_door);
}

#[test]
fn a_found_row_states_its_event_signal() {
    check_event_signal(hosted_door);
}

#[test]
fn a_hosted_door_checks_and_writes_the_whole_registration() {
    check_rank_rows(hosted_door).expect("rank registrations at the hosted door");
}

#[test]
fn a_hosted_door_keeps_the_write_of_one_row() {
    check_one_row(hosted_door);
}

#[test]
fn a_removal_takes_only_the_row_its_asker_saw() {
    check_guard(hosted_door).expect("both registrations stand in turn");
}

#[test]
fn a_refresh_writes_its_workers_labels_and_no_other_fact() {
    check_refresh(hosted_door).expect("the makers and their rows");
}

#[test]
fn the_hosts_own_source_removes_its_rows_under_any_key() {
    check_provider(hosted_door).expect("the makers and their rows");
}

#[test]
fn health_removes_inside_its_scope_alone() {
    check_health_scope(hosted_door).expect("the makers and their rows");
}

#[test]
fn a_removal_leaves_the_rows_maker_as_it_was() {
    check_maker(hosted_door).expect("the maker and its row");
}

#[test]
fn a_provider_acting_as_maker_takes_only_its_own_guarded_row() {
    let (store, door) = door();
    let door = Arc::new(door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let maker = found("pods/a");
    hosted
        .found_one(&maker, URL, &|_| found_facts())
        .expect("found");
    let made = store.by_key(URL).expect("the found row");
    let other = "grpc://10.0.0.6:8000";
    door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    let epoch = store.epoch();
    let denied = |answer: Result<MembershipWriteOutcome, MembershipWriteRefused>, whose: &str| {
        let Err(MembershipWriteRefused::NotRegistrant(why)) = answer else {
            panic!("wrong refusal: {answer:?}")
        };
        assert!(why.contains(whose), "{why}");
        assert!(!why.contains("10.0.0"), "{why}");
        assert!(!why.contains("pods/"), "{why}");
    };
    // Another maker's row stays: a control call's, and this provider's
    // own under another key, which is told so.
    let by = Remover::Registrant(&maker);
    let seen = standing(&made);
    let control = store.by_key(other).expect("the control row");
    denied(hosted.gone(by, other, standing(&control)), "a control call");
    denied(
        hosted.gone(Remover::Registrant(&found("pods/b")), URL, seen),
        "provider cluster's under another key",
    );
    // A caller of another provider is told whose the row is, and
    // nothing of a key.
    let stranger = Registrant::provider("nodes", "pods/a").expect("a whole maker");
    denied(
        hosted.gone(Remover::Registrant(&stranger), URL, seen),
        "provider cluster's: its worker",
    );
    // No row under the URL is not found, and a replica that does not
    // lead refuses by name. None of the five wrote.
    assert_eq!(
        hosted.gone(by, "grpc://10.0.0.9:8000", seen),
        Err(MembershipWriteRefused::Absent)
    );
    let follower = || -> Result<u64, Status> {
        Err(Status::failed_precondition("this replica does not lead"))
    };
    let follower = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), follower);
    assert_eq!(
        follower.gone(by, URL, seen),
        Err(MembershipWriteRefused::NotLeading(
            "this replica does not lead".to_owned()
        ))
    );
    assert_eq!(store.epoch(), epoch);
    // The maker's own call takes the row away and answers it removed.
    assert_eq!(
        hosted.gone(by, URL, seen),
        Ok(MembershipWriteOutcome::Removed(made.id))
    );
    assert!(store.get(made.id).is_none());
    assert_eq!(store.epoch(), epoch + 1);
    // The old guard still names the previous row at this URL.
    hosted
        .found_one(&found("pods/c"), URL, &|_| found_facts())
        .expect("found again");
    let again = store.by_key(URL).expect("the new row");
    assert_ne!(again.id, made.id);
    assert_eq!(
        hosted.gone(by, URL, seen),
        Err(MembershipWriteRefused::Moved)
    );
    assert_eq!(store.by_key(URL), Some(again));
}

#[test]
fn a_providers_write_on_a_replica_that_does_not_lead_is_refused_and_writes_nothing() {
    use std::sync::atomic::AtomicUsize;

    let (store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM))
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL);
    let seen = standing(row.as_ref().expect("the registered row"));
    let by = Remover::Registrant(&maker);
    let epoch = store.epoch();
    let at = door.place();
    let reads = Arc::new(AtomicUsize::new(0));
    let fence_reads = Arc::clone(&reads);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), move || {
        assert!(door.turn.is_locked(), "the fence is read inside the turn");
        fence_reads.fetch_add(1, Ordering::Relaxed);
        Err(Status::failed_precondition("this replica does not lead"))
    });
    let refused = Err(MembershipWriteRefused::NotLeading(
        "this replica does not lead".to_owned(),
    ));
    assert_eq!(
        hosted.found_one(&maker, URL, &|_| panic!(
            "a follower does not ask for facts"
        )),
        refused
    );
    assert_eq!(hosted.drained(by, URL, seen), refused);
    assert_eq!(hosted.gone(by, URL, seen), refused);
    assert_eq!(reads.load(Ordering::Relaxed), 3);
    assert_eq!(store.by_key(URL), row);
    assert_eq!(store.epoch(), epoch);
    assert_eq!(hosted.door.place(), at);
}

/// A provider's write on the replica that leads is made in the door's
/// turn: the fence is read and the row's facts are stated while the turn
/// is held, and the call holds it no more once it answers.
#[test]
fn a_providers_write_on_the_replica_that_leads_is_made_in_the_doors_turn() {
    let (store, door) = door();
    let door = Arc::new(door);
    let fenced = Arc::clone(&door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), move || {
        assert!(fenced.turn.is_locked(), "the fence is read inside the turn");
        Ok(TERM)
    });
    let maker = found("pods/a");
    let stated = AtomicBool::new(false);
    hosted
        .found_one(&maker, URL, &|_| {
            assert!(
                door.turn.is_locked(),
                "the facts are stated inside the turn"
            );
            stated.store(true, Ordering::Relaxed);
            found_facts()
        })
        .expect("registered");
    assert!(
        stated.load(Ordering::Relaxed),
        "the write asked for the facts"
    );
    assert!(!door.turn.is_locked(), "the call let the turn go");
    assert!(store.by_key(URL).is_some(), "the row stands");
}

#[test]
fn a_providers_registration_numbers_its_life_under_the_term_the_log_leads() {
    use super::{ReplicaTurn, Turn};
    use crate::store::membership::FactPastBound;

    let (store, door) = door_in(TERM + 1);
    let door = Arc::new(door);
    let maker = found("pods/a");
    let stale = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let refusal = Err(MembershipWriteRefused::NotLeading(
        super::LEADS_NO_TERM.to_owned(),
    ));
    assert_eq!(stale.found_one(&maker, URL, &|_| found_facts()), refusal);
    assert_eq!(store.epoch(), 0);

    let leading = ReplicaMembershipWriter::new(door, operator(), under(TERM + 1));
    let number = life(TERM + 1, 0);
    assert_eq!(
        leading.found_one(&maker, URL, &|_| found_facts()),
        Ok(MembershipWriteOutcome::Registered(identity(URL, number)))
    );
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(row.facts.incarnation, number);
    let replaced = || WorkerFacts {
        role: Role::Decode,
        ..found_facts()
    };
    assert_eq!(stale.found_one(&maker, URL, &|_| replaced()), refusal);
    assert_eq!(store.by_key(URL), Some(row.clone()));
    let new = life(TERM + 1, 1);
    assert_eq!(
        leading.found_one(&maker, URL, &|_| replaced()),
        Ok(MembershipWriteOutcome::Another {
            old: row.id,
            new: identity(URL, new)
        })
    );
    assert_eq!(
        store
            .by_key(URL)
            .expect("the replaced row")
            .facts
            .incarnation,
        new
    );

    let turn = ReplicaTurn {
        host: &leading,
        term: TERM + 1,
    };
    let heavy = WorkerFacts {
        models: vec!["m".repeat(256); 64],
        ..found_facts()
    };
    assert_eq!(
        turn.room(&heavy, &maker),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "row",
            why: "a row is at most 16384 bytes on the wire",
        }))
    );

    let (store, door) = door_in(TERM);
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let state = |_: Option<&WorkerFacts>| {
        store::LeasesStore::set_term(store.as_ref(), TERM + 1);
        found_facts()
    };
    assert_eq!(hosted.found_one(&maker, URL, &state), refusal);
    assert_eq!(store.all_rows(), (0, Vec::new()));
    // A life is asked a row at a time: once the term has moved the turn
    // makes none, at whichever row of a registration asks it.
    let turn = ReplicaTurn {
        host: &hosted,
        term: TERM,
    };
    assert_eq!(
        turn.life(),
        Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned()
        ))
    );
}

#[test]
fn the_two_doors_answer_one_row_alike() {
    let (replica, door) = door();
    let local = Arc::new(MemoryStore::default());
    let maker = found("pods/a");
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let lone = StoreDoor::new(Arc::clone(&local));
    let write = |door: &dyn Rows, store: &MemoryStore| {
        let by = Remover::Registrant(&maker);
        assert_eq!(
            door.gone(
                by,
                URL,
                RowRevision {
                    row: WorkerId(0),
                    revision: 0
                }
            ),
            Err(MembershipWriteRefused::Absent)
        );
        let registered = door
            .found_one(&maker, URL, &|_| found_facts())
            .expect("registered");
        let first = store.by_key(URL).expect("the first row");
        assert_eq!(registered, MembershipWriteOutcome::Registered(first.id));
        assert_eq!(
            door.found_one(&maker, URL, &|_| found_facts()),
            Ok(MembershipWriteOutcome::Unmoved(first.id))
        );
        let refresh = || WorkerFacts {
            max_running: Some(8),
            ..found_facts()
        };
        assert_eq!(
            door.found_one(&maker, URL, &|_| refresh()),
            Ok(MembershipWriteOutcome::InPlace(first.id))
        );
        let refreshed = store.by_key(URL).expect("the refreshed row");
        let wrote = door.found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..refresh()
        });
        let fresh = store.by_key(URL).expect("the new row");
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::Another {
                old: first.id,
                new: fresh.id
            })
        );
        let seen = standing(&fresh);
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Draining(fresh.id))
        );
        let draining = store.by_key(URL).expect("the draining row");
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Unmoved(fresh.id))
        );
        assert_eq!(
            door.gone(by, URL, seen),
            Ok(MembershipWriteOutcome::Removed(fresh.id))
        );
        assert_eq!(store.by_key(URL), None);
        [first, refreshed, fresh, draining].map(|row| {
            assert_eq!(row.id, identity(URL, row.facts.incarnation));
            (
                WorkerFacts {
                    incarnation: 0,
                    ..row.facts
                },
                row.revision,
                row.status,
                row.maker,
            )
        })
    };
    assert_eq!(write(&lone, &local), write(&hosted, &replica));
    assert_eq!(local.epoch(), replica.epoch());
}

#[test]
fn an_add_over_a_standing_row_is_already_there_whoever_made_it() {
    let (store, door) = door();
    let maker = found("pods/a");
    door.upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
        .expect("found");
    let other = "grpc://10.0.0.6:8000";
    door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    let epoch = store.epoch();
    let there = |answer: Result<proto::WorkerEvent, Status>| {
        let status = answer.expect_err("refused");
        assert_eq!(status.code(), Code::AlreadyExists, "{status:?}");
    };
    // An add asks no maker: a control call's and a provider's read the
    // same over a provider's row and over a control call's.
    for url in [URL, other] {
        there(door.register(&operator(), under(TERM), &ask(spec(url), Mode::Add)));
        there(door.upsert(&operator(), &maker, TERM, &ask(spec(url), Mode::Add)));
    }
    assert_eq!(store.epoch(), epoch);
}

#[test]
fn a_patch_merges_under_the_revision_its_caller_read() {
    let (store, door) = door();
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(labelled, Mode::Add))
            .expect("registered"),
    );
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: vec!["tier".to_owned()],
        overload: None,
        revision: row.revision,
    };
    let told = door
        .patch(&operator(), under(TERM), &patch)
        .expect("patched");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let now = row_of(&told);
    assert_eq!(now.labels, [("lane".to_owned(), "b".to_owned())].into());
    assert_eq!(
        (now.revision, told.membership_epoch, told.seq, told.term),
        (2, 2, 3, TERM)
    );

    // The same patch under the revision it read: the row moved.
    let stale = door
        .patch(&operator(), under(TERM), &patch)
        .expect_err("a stale revision");
    assert_eq!(stale.code(), Code::FailedPrecondition);
    assert_eq!(stale.message(), "the row is at revision 2");

    // Under the row's revision it changes nothing and writes nothing.
    let again = proto::PatchWorkerRequest {
        revision: 2,
        ..patch.clone()
    };
    let told = door
        .patch(&operator(), under(TERM), &again)
        .expect("answered");
    assert_eq!(
        (row_of(&told).revision, told.membership_epoch, told.seq),
        (2, 2, 3)
    );
    assert_eq!(store.epoch(), 2);

    let nobody = proto::PatchWorkerRequest {
        worker_id: 1,
        ..patch
    };
    let absent = door
        .patch(&operator(), under(TERM), &nobody)
        .expect_err("no such worker");
    assert_eq!(absent.code(), Code::NotFound);
}

#[test]
fn a_row_past_a_bound_is_refused_by_name_and_nothing_is_written() {
    fn labelled(count: usize) -> proto::WorkerSpec {
        let labels = (0..count).map(|n| (n.to_string(), "v".repeat(4_096)));
        proto::WorkerSpec {
            labels: labels.collect(),
            ..spec(URL)
        }
    }
    let (store, door) = door();
    // A registration: by its field, and by its spec for the row whole.
    let long = ask(spec(&"u".repeat(2_049)), Mode::Add);
    let heavy = ask(labelled(4), Mode::Add);
    for (request, path) in [(long, "spec.url: "), (heavy, "spec: ")] {
        let refused = door
            .register(&operator(), under(TERM), &request)
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
        assert!(
            refused.message().starts_with(path),
            "{path}: {}",
            refused.message()
        );
    }
    assert_eq!(store.epoch(), 0);
    // A patch: by its labels, for the row it would make. The row stands.
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(labelled(3), Mode::Add))
            .expect("registered"),
    );
    let fourth = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("3".to_owned(), "v".repeat(4_096))].into(),
        revision: row.revision,
        ..Default::default()
    };
    let refused = door
        .patch(&operator(), under(TERM), &fourth)
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().starts_with("labels: "),
        "{}",
        refused.message()
    );
    let stands = store.get(numbered(URL, 0)).expect("the row");
    assert_eq!(
        (stands.revision, stands.facts.labels.len(), store.epoch()),
        (1, 3, 1)
    );
}

#[test]
fn a_drain_answers_the_status_change_and_a_removal_the_row_as_it_last_stood() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let stale = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision + 1,
        drain: false,
    };
    let refused = door
        .deregister(&operator(), under(TERM), &stale)
        .expect_err("a stale revision");
    assert_eq!(refused.code(), Code::FailedPrecondition);

    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    let told = door
        .deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    assert_eq!(
        told.event,
        Some(Event::Status(proto::StatusChanged {
            worker_id: row.id,
            old: proto::WorkerStatus::Pending as i32,
            new: proto::WorkerStatus::Draining as i32,
            health: proto::HealthFlag::Draining as u32,
        }))
    );
    assert_eq!((told.membership_epoch, told.seq), (2, 3));
    // A second drain writes nothing and answers the row as it stands,
    // under the revision its registration gave: no status change moved
    // it.
    let told = door
        .deregister(&operator(), under(TERM), &drain)
        .expect("draining still");
    let stood = store.get(WorkerId(row.id)).expect("the row stands");
    assert_eq!(
        told.event,
        Some(Event::Relabelled(super::worker_wire::row(&stood)))
    );
    assert_eq!(
        (
            stood.status,
            stood.revision,
            told.membership_epoch,
            told.seq
        ),
        (WorkerStatus::Draining, row.revision, 2, 3)
    );

    let removal = proto::DeregisterWorkerRequest {
        drain: false,
        ..drain
    };
    let told = door
        .deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    assert_eq!(
        told.event,
        Some(Event::Removed(proto::Removed {
            worker_id: row.id,
            registration_key: URL.to_owned(),
            revision: 1,
            draining: true,
        }))
    );
    assert_eq!((told.membership_epoch, told.seq, store.epoch()), (3, 4, 3));
    assert_eq!(store.by_key(URL), None);
    let gone = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("gone");
    assert_eq!(gone.code(), Code::NotFound);
}

#[test]
fn a_call_reads_the_fence_once_it_holds_the_doors_turn() {
    let (_store, door) = door();
    // The fence's reader finds the turn taken: the call that reads it
    // holds it.
    let held = || {
        assert!(door.turn.try_lock().is_none(), "read outside the turn");
        Ok(TERM)
    };
    let row = row_of(
        &door
            .register(&operator(), held, &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let row = row_of(&door.patch(&operator(), held, &patch).expect("patched"));
    let removal = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: false,
    };
    door.deregister(&operator(), held, &removal)
        .expect("removed");
}

#[test]
fn a_call_that_lost_the_lead_while_it_waited_its_turn_is_refused_and_writes_nothing() {
    let (store, door) = door();
    let leads = AtomicBool::new(true);
    let fence = || {
        if leads.load(Ordering::SeqCst) {
            Ok(TERM)
        } else {
            Err(Status::failed_precondition("this replica does not lead"))
        }
    };
    let request = ask(spec(URL), Mode::Add);
    // Another call holds the turn. This one waits behind it, and the
    // lead is lost before its turn comes.
    let turn = door.turn.lock();
    let answer = thread::scope(|scope| {
        let waiting = scope.spawn(|| door.register(&operator(), fence, &request));
        leads.store(false, Ordering::SeqCst);
        drop(turn);
        waiting.join().expect("the call returns")
    });
    let refused = answer.expect_err("the lead was lost before the call's turn");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!((store.epoch(), store.by_key(URL)), (0, None));
}

/// A drain goes with the word of the term its call was admitted under.
/// Here the fence names the term and the store takes the next one ahead
/// of the write, as a call finds it that waited for a lock through the
/// end of its term. The drain writes no status, on the object or in the
/// table, and the call answers that the replica does not lead, never
/// that the row moved. Under the term the log leads, the same drain is
/// written.
#[test]
fn a_drain_whose_call_lost_its_term_writes_nothing_and_is_refused_as_not_leading() {
    let (store, door) = door();
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
    let lost = || {
        store::LeasesStore::set_term(store.as_ref(), TERM + 1);
        Ok(TERM)
    };
    let epoch = store.epoch();
    let refused = door
        .deregister(&operator(), lost, &drain)
        .expect_err("the term ended ahead of the write");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::LEADS_NO_TERM)
    );
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending)
    );
    assert_eq!(store.epoch(), epoch);
    door.deregister(&operator(), under(TERM + 1), &drain)
        .expect("draining");
    assert_eq!(status(store.get(id)), Some(WorkerStatus::Draining));
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

/// A connect signal goes with the word of its call's term too. This
/// registration loses its lead as its own write lands, so the signal
/// for its row, whose checks are off, writes nothing: the row stays
/// pending on the object and in the table, and the call that waits
/// answers it so. The registration sent again states equal facts and
/// writes no row. Under a term the log leads it sends the signal: the
/// row is ready under its id, at the revision it stood at.
#[test]
fn a_connect_signal_whose_call_lost_its_term_leaves_the_row_pending() {
    let waits = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(URL), Mode::Add)
    };
    let (store, door, told) = lost_its_lead_at(0, &waits);
    let row = row_of(&told.expect("registered"));
    assert_eq!(row.status, proto::WorkerStatus::Pending as i32);
    let id = WorkerId(row.id);
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending)
    );
    assert_eq!(store.epoch(), 1, "the registration's write alone");
    // The registration sent again finds the row unmoved, and its signal
    // has the word of its own term. The log leads a later term now: a
    // call admitted under the one that ended sends nothing.
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let again = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(URL), Mode::Upsert)
    };
    let ended = door
        .register(&operator(), under(TERM), &again)
        .expect("unmoved");
    assert_eq!(row_of(&ended), row);
    assert_eq!((status(store.get(id)), store.epoch()), (pending, 1));
    // One admitted under the term the log leads sends the signal, and
    // as it waits it answers the row behind it: ready, with no row
    // written and no life made.
    let told = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("unmoved");
    assert!(matches!(told.event, Some(Event::Relabelled(_))), "{told:?}");
    let ready = row_of(&told);
    assert_eq!(
        (ready.id, ready.revision, ready.status),
        (row.id, row.revision, proto::WorkerStatus::Ready as i32)
    );
    let stored = store.get(id).expect("the row");
    assert_eq!(
        (stored.status, stored.revision),
        (WorkerStatus::Ready, row.revision)
    );
    assert_eq!(status(door.row(id)), Some(WorkerStatus::Ready));
    assert_eq!(
        (store.epoch(), store.all_rows().1.len()),
        (2, 1),
        "the status alone"
    );
    // Sent once more it finds the row ready and writes nothing.
    let once_more = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("unmoved");
    assert_eq!((row_of(&once_more), store.epoch()), (ready, 2));
}

/// A call that does not wait answers the row as it stood, and the
/// signal it sent again readies the row behind its answer, as a
/// registration that wrote answers its row ahead of the signal. The
/// registration is sent again as an upsert: the same add is refused
/// where its row stands, as any add is, and sends no signal.
#[test]
fn an_equal_registration_that_does_not_wait_answers_the_row_ahead_of_its_signal() {
    let (store, door, _) = lost_its_lead_at(0, &ask(unchecked(URL), Mode::Add));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let stood = store.by_key(URL).expect("the row");
    let refused = door
        .register(
            &operator(),
            under(TERM + 2),
            &ask(unchecked(URL), Mode::Add),
        )
        .expect_err("the row stands");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!(store.get(stood.id).as_ref(), Some(&stood));
    let told = door
        .register(
            &operator(),
            under(TERM + 2),
            &ask(unchecked(URL), Mode::Upsert),
        )
        .expect("unmoved");
    assert_eq!(
        told.event,
        Some(Event::Relabelled(super::worker_wire::row(&stood)))
    );
    assert_eq!(told.membership_epoch, 1);
    let ready = store.get(stood.id).expect("the row");
    assert_eq!(
        (ready.status, ready.revision, store.epoch()),
        (WorkerStatus::Ready, stood.revision, 2)
    );
}

/// A registration that changes its row in place asks no term ahead of
/// its signal, so the signal's own word is all that stands between a
/// call whose term ended and the status. The log leads a later term,
/// and a call admitted under the one that ended relabels the waiting
/// row: the row's write lands, which asks no term under the lock yet,
/// and the signal asks the call's term and writes nothing. The
/// same registration under the term the log leads finds the row
/// unmoved and sends it.
#[test]
fn a_change_in_place_sends_its_signal_under_the_term_of_its_call() {
    let (store, door, _) = lost_its_lead_at(0, &ask(unchecked(URL), Mode::Add));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let stood = store.by_key(URL).expect("the row");
    let relabelled = || {
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..unchecked(URL)
        };
        ask(labelled, Mode::Upsert)
    };
    door.register(&operator(), under(TERM), &relabelled())
        .expect("in place");
    let row = store.get(stood.id).expect("the row");
    assert_eq!(
        (row.status, row.revision, row.facts.labels.len()),
        (WorkerStatus::Pending, stood.revision + 1, 1)
    );
    door.register(&operator(), under(TERM + 2), &relabelled())
        .expect("unmoved");
    let ready = store.get(stood.id).expect("the row");
    assert_eq!(
        (ready.status, ready.revision),
        (WorkerStatus::Ready, row.revision)
    );
}

/// The host door's routine sends the signal again too, for a rank
/// that stands unmoved. This add of three ranks, their checks off,
/// lost its lead as its first rank's write landed: the rank stands
/// pending, its signal not written, and the call was refused at the
/// next. The same add called again lays the two ranks left and sends
/// the first its signal, which asks its own term: a call admitted under
/// the term that ended sends nothing. It waits, so it answers each
/// rank behind its signal, the unmoved one too. An add writes only
/// where a rank is left to write: the next case has the engine whole.
#[test]
fn an_add_called_again_sends_an_unmoved_ranks_connect_signal_again() {
    use proto::rank_row::Made;
    let ranks = || proto::WorkerSpec {
        dp_size: 3,
        ..unchecked(URL)
    };
    let (store, door, lost) = lost_its_lead_at(0, &ask(ranks(), Mode::Add));
    let lost = lost.expect_err("the lead was lost at the second rank");
    assert_eq!(lost.code(), Code::Unavailable);
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let first = store.by_key(&format!("{URL}@0")).expect("the first rank");
    assert_eq!(
        (first.status, store.epoch(), store.all_rows().1.len()),
        (WorkerStatus::Pending, 1, 1)
    );
    let again = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(ranks(), Mode::Add)
    };
    // The log leads a later term. A call admitted under the one that
    // ended is refused ahead of its first rank, and sends nothing.
    let ended = door
        .register(&operator(), under(TERM), &again)
        .expect_err("its term ended");
    assert_eq!(ended.code(), Code::Unavailable);
    assert_eq!(store.get(first.id).as_ref(), Some(&first));
    let told = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("the same add");
    let made = made_of(&told);
    let ready = proto::WorkerStatus::Ready as i32;
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(stood), Made::Registered(second), Made::Registered(third)]
                if (stood.id, stood.revision) == (first.id.0, first.revision)
                    && [stood.status, second.status, third.status] == [ready; 3]
        ),
        "{made:?}"
    );
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 3);
    for row in rows {
        assert_eq!(row.status, WorkerStatus::Ready, "{}", row.facts.key);
    }
    assert_eq!(
        store.get(first.id).map(|row| row.revision),
        Some(first.revision)
    );
}

/// An add sent again over an engine that stands whole is refused ahead
/// of every write and sends no signal, as an add over one row is: an
/// add changes no row that stands. This add of two ranks, their checks
/// off, lost its lead as its last rank's write landed, so both ranks
/// stand and the last waits for its signal. The same add is refused
/// and the rank waits on. An upsert of equal facts sends the signal.
/// It does not wait, so it answers the rank as it stood, ahead of the
/// signal.
#[test]
fn an_add_over_an_engine_that_stands_whole_sends_no_signal_and_an_upsert_sends_it() {
    use proto::rank_row::Made;
    let ranks = || proto::WorkerSpec {
        dp_size: 2,
        ..unchecked(URL)
    };
    // The lead holds for the first rank's row and its signal, and is
    // lost as the last rank's row lands.
    let (store, door, told) = lost_its_lead_at(2, &ask(ranks(), Mode::Add));
    assert_eq!(made_of(&told.expect("both ranks registered")).len(), 2);
    let status = |rank: u32| {
        let row = store.by_key(&format!("{URL}@{rank}"));
        row.map(|row| row.status)
    };
    let (ready, pending) = (Some(WorkerStatus::Ready), Some(WorkerStatus::Pending));
    assert_eq!((status(0), status(1), store.epoch()), (ready, pending, 3));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let whole = door
        .register(&operator(), under(TERM + 2), &ask(ranks(), Mode::Add))
        .expect_err("the engine stands whole");
    assert_eq!(whole.code(), Code::AlreadyExists);
    assert_eq!((status(1), store.epoch()), (pending, 3));
    let waiting = store.by_key(&format!("{URL}@1")).expect("the last rank");
    let told = door
        .register(&operator(), under(TERM + 2), &ask(ranks(), Mode::Upsert))
        .expect("unmoved");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(first), Made::Relabelled(last)]
                if first.status == proto::WorkerStatus::Ready as i32
                    && *last == super::worker_wire::row(&waiting)
        ),
        "{made:?}"
    );
    assert_eq!((status(0), status(1), store.epoch()), (ready, ready, 4));
    assert_eq!(
        store.get(waiting.id).map(|row| row.revision),
        Some(waiting.revision)
    );
}

/// A checked row takes no connect signal from equal facts, pending as
/// it stands: a prober's verdict readies it, and the registration sent
/// again writes nothing. So it is for a call of one row and for each
/// rank of a wide call, which the host door's routine lays.
#[test]
fn a_pending_row_whose_checks_are_on_takes_no_signal_from_equal_facts() {
    let (store, door) = door();
    for stated in [spec(URL), wide("grpc://10.0.0.6:8000", 2)] {
        door.register(&operator(), under(TERM), &ask(stated.clone(), Mode::Add))
            .expect("registered");
        let epoch = store.epoch();
        door.register(&operator(), under(TERM), &ask(stated, Mode::Upsert))
            .expect("unmoved");
        assert_eq!(store.epoch(), epoch);
    }
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let held = door.row(row.id).map(|row| row.status);
        assert_eq!(
            (row.status, held),
            (WorkerStatus::Pending, Some(WorkerStatus::Pending)),
            "{}",
            row.facts.key
        );
    }
}

/// The replica's own drain asks its turn's term, as a control call's
/// does. Where the term ended ahead of the write, the drain writes
/// nothing and is refused as a turn that does not lead.
#[test]
fn a_replicas_own_drain_whose_term_ended_is_refused_as_not_leading() {
    let (store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM))
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL).expect("the registered row");
    let (seen, epoch) = (standing(&row), store.epoch());
    let taken = Arc::clone(&store);
    let hosted = ReplicaMembershipWriter::new(door, operator(), move || {
        store::LeasesStore::set_term(taken.as_ref(), TERM + 1);
        Ok(TERM)
    });
    assert_eq!(
        hosted.drained(Remover::Registrant(&maker), URL, seen),
        Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned()
        ))
    );
    assert_eq!((store.by_key(URL), store.epoch()), (Some(row), epoch));
}

/// The holder's two public entries hand a word that always holds, as
/// before the door handed its own: a host with no term to end calls
/// them, and they write whatever a log leads. Over a log that left the
/// term, the door's two write nothing under it, and the public two
/// write.
#[test]
fn the_holders_public_connect_signal_and_drain_write_under_no_term() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let (id, revision) = (WorkerId(row.id), row.revision);
    store::LeasesStore::set_term(store.as_ref(), TERM + 1);
    let status = || store.get(id).map(|row| row.status);
    door.connect_row(id, revision, TERM);
    assert!(!door.drain_row(id, revision, TERM));
    assert_eq!(status(), Some(WorkerStatus::Pending));
    door.objects.connect_row(&store, id, revision);
    assert_eq!(status(), Some(WorkerStatus::Ready));
    assert!(door.objects.drain_row(&store, id, revision));
    assert_eq!(status(), Some(WorkerStatus::Draining));
}

/// A term that ends by a later take of the store ([`LeavesAtATelling`]).
const BY_A_RISE: usize = 1;
/// A term that ends by the start of a later leader's term, which the store
/// follows ([`LeavesAtATelling`]).
const BY_A_FOLLOWED_TERM: usize = 2;

/// The door's log as its store's journal, which leaves [`TERM`] in the
/// telling of the next status once a case arms it: behind every ask of
/// the writer's word, and ahead of the place the log gives the event.
struct LeavesAtATelling {
    log: Arc<replication::Log>,
    /// How the term ends at the next status told, [`BY_A_RISE`] or
    /// [`BY_A_FOLLOWED_TERM`], and zero once it has.
    armed: AtomicUsize,
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
        match self.armed.swap(0, Ordering::Relaxed) {
            BY_A_RISE => self.log.term_rose(TERM + 1),
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

#[test]
fn a_registration_that_states_no_policy_is_checked_whatever_stood_under_its_url() {
    let (store, door) = door();
    let status = |url: &str| store.by_key(url).map(|row| row.status);
    door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    assert_eq!(status(URL), Some(WorkerStatus::Ready));
    // The URL's next worker (another model: no status comes over)
    // states no policy. Nobody said "do not check" of it: a wait for
    // its handshake is refused as any checked worker's is, and it is
    // pending until a verdict.
    let other = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..spec(URL)
    };
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(other.clone(), Mode::Replace)
    };
    let refused = door
        .register(&operator(), under(TERM), &waited)
        .expect_err("no handshake runs");
    assert_eq!(refused.code(), Code::InvalidArgument);
    let told = door
        .register(&operator(), under(TERM), &ask(other, Mode::Replace))
        .expect("replaced");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.health_policy.map(|policy| policy.disabled)),
        (proto::WorkerStatus::Pending as i32, Some(false))
    );
    assert_eq!(status(URL), Some(WorkerStatus::Pending));

    // In place too: an upsert that states no policy over an unchecked
    // row turns its checks on and leaves its status to the prober.
    let third = "grpc://10.0.0.7:8000";
    door.register(&operator(), under(TERM), &ask(unchecked(third), Mode::Add))
        .expect("registered");
    let told = door
        .register(&operator(), under(TERM), &ask(spec(third), Mode::Upsert))
        .expect("in place");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.health_policy.map(|policy| policy.disabled)),
        (proto::WorkerStatus::Ready as i32, Some(false))
    );
}

#[test]
fn a_drain_ends_with_the_registration_it_named() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered"),
    );
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    // A patch is the same registration: draining still.
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: row.revision,
    };
    let told = door
        .relabel(&operator(), &Registrant::Control, &patch)
        .expect("patched");
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Draining as i32);
    // A registration over a draining row is the URL's next incarnation,
    // by an upsert as by a replacement: the drain is not handed on, and
    // an unchecked worker is ready from its own.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Upsert))
        .expect("registered anew");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    let anew = store.by_key(URL).expect("the new row");
    assert_eq!(anew.status, WorkerStatus::Ready);
    let drain = proto::DeregisterWorkerRequest {
        worker_id: anew.id.0,
        revision: anew.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining again");
    // The URL's next incarnation is another registration: the drain is
    // not handed on, and an unchecked worker is ready from its own.
    let told = door
        .register(
            &operator(),
            under(TERM),
            &ask(unchecked(URL), Mode::Replace),
        )
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn a_rows_revision_counts_its_stated_facts_and_no_status_change() {
    let (store, door) = door();
    // An unchecked worker turns ready behind its registration's answer
    // and is drained: neither moves the revision that answer gave.
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered"),
    );
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    assert_eq!(
        store.by_key(URL).map(|row| (row.status, row.revision)),
        Some((WorkerStatus::Draining, row.revision))
    );
    // A patch states a fact: the revision moves, the one before it is
    // refused and the new one removes the row.
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let patched = row_of(
        &door
            .patch(&operator(), under(TERM), &patch)
            .expect("patched"),
    );
    assert_eq!(patched.revision, row.revision + 1);
    let stale = proto::DeregisterWorkerRequest {
        drain: false,
        ..drain
    };
    let refused = door
        .deregister(&operator(), under(TERM), &stale)
        .expect_err("the facts moved");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    let removal = proto::DeregisterWorkerRequest {
        revision: patched.revision,
        ..stale
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
}

pub(crate) use crate::document::operator::test_support::Logged;

#[test]
fn every_call_leaves_its_audit_line_written_unmoved_or_refused() {
    // A subscriber of the case's own keeps this thread's lines. A
    // second one stays registered beside it: with one alone a line is
    // on or off by the thread that reaches it first, and a case beside
    // this one reaches it under none.
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    let add = ask(spec(URL), Mode::Add);
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &add)
            .expect("registered"),
    );
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("unmoved");
    door.register(&operator(), under(TERM), &add)
        .expect_err("the URL stands");
    let nothing = proto::PatchWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        ..Default::default()
    };
    door.patch(&operator(), under(TERM), &nothing)
        .expect("unmoved");
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    for answer in ["draining", "draining still"] {
        door.deregister(&operator(), under(TERM), &drain)
            .expect(answer);
    }
    // A line a call, in the calls' order: what a write made, `nothing`
    // where a call wrote none, a refusal by its code.
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let want = [
        ("membership write", "RegisterWorker", "made=\"registered\""),
        ("membership write", "RegisterWorker", "made=\"nothing\""),
        (
            "membership call refused",
            "RegisterWorker",
            "code=AlreadyExists",
        ),
        ("membership write", "PatchWorker", "made=\"nothing\""),
        ("membership write", "DeregisterWorker", "made=\"status\""),
        ("membership write", "DeregisterWorker", "made=\"nothing\""),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (kind, call, made)) in lines.iter().zip(want) {
        let call = format!("call=\"{call}\" by=operator-a");
        assert!(
            line.contains(kind) && line.contains(&call) && line.contains(made),
            "{line}"
        );
    }
    // No line holds the URL: it may carry a credential.
    assert!(!logged.contains(URL), "{logged}");
}

/// A registration that states several ranks leaves a line a row: what
/// the call made of each rank, `nothing` where it moved none, and each
/// row that left.
#[test]
fn a_registration_of_several_ranks_leaves_an_audit_line_a_row() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    for (width, mode) in [(3, Mode::Add), (3, Mode::Upsert), (2, Mode::Upsert)] {
        door.register(&operator(), under(TERM), &ask(wide(URL, width), mode))
            .expect("laid");
    }
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let made: Vec<&str> = logged
        .lines()
        .map(|line| {
            let call = "call=\"RegisterWorker\" by=operator-a";
            assert!(
                line.contains("membership write") && line.contains(call),
                "{line}"
            );
            ["registered", "relabelled", "nothing", "removed"]
                .into_iter()
                .find(|made| line.contains(&format!("made=\"{made}\"")))
                .expect("what the call made of the row")
        })
        .collect();
    assert_eq!(
        made,
        [
            "registered",
            "registered",
            "registered",
            "nothing",
            "nothing",
            "nothing",
            "relabelled",
            "relabelled",
            "removed"
        ]
    );
    // No line holds the URL: it may carry a credential.
    assert!(!logged.contains(URL), "{logged}");
}

#[test]
fn a_removal_health_asked_leaves_an_audit_line_that_names_health() {
    use tracing::{
        subscriber::{self, NoSubscriber},
        Dispatch,
    };

    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = Dispatch::new(NoSubscriber::default());
    let _kept = subscriber::set_default(subscriber);

    let (store, door) = hosted_door();
    let key = "private/health-key";
    let maker = found(key);
    door.found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL).expect("the row health saw");
    let seen = RowRevision {
        row: row.id,
        revision: row.revision,
    };
    assert_eq!(
        door.drained(Remover::Health(UnhealthyRemovalScope::Disabled), URL, seen),
        Err(MembershipWriteRefused::OutOfScope)
    );
    let by = Remover::Health(UnhealthyRemovalScope::DiscoveryOwned);
    assert_eq!(
        door.drained(by, URL, seen),
        Ok(MembershipWriteOutcome::Draining(row.id))
    );
    assert_eq!(
        door.gone(by, URL, seen),
        Ok(MembershipWriteOutcome::Removed(row.id))
    );

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let calls: Vec<_> = logged.lines().skip(1).collect();
    assert_eq!(calls.len(), 3, "{logged}");
    assert!(
        calls.iter().all(|line| line.contains("asker=\"health\"")),
        "{logged}"
    );
    assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
    assert!(calls[1].contains("made=\"status\""), "{logged}");
    assert!(calls[2].contains("made=\"removed\""), "{logged}");
    assert!(!logged.contains(URL) && !logged.contains(key), "{logged}");
}

/// A refresh leaves one audit line a call, under its own name. One that
/// states another fact than the labels is refused in a refresh's
/// words, never in a removal's, and no line holds the URL.
#[test]
fn a_refresh_leaves_its_audit_line_in_its_own_words() {
    const HTTP: &str = "http://10.0.0.5:8000";
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (store, door) = hosted_door();
    let built = |model: &str, tier: &str| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(HTTP).model(ModelCard::new(model));
        Arc::new(worker.label("tier", tier).build())
    };
    let first = built("m", "gold");
    door.found_built(&Registrant::Control, Precondition::Absent, &first, None)
        .expect("registered");
    let row = store.by_key(HTTP).expect("the registered row");
    let seen = standing(&row);
    let stray = door.refreshed(seen, &built("other", "gold"), &first);
    assert_eq!(stray, Err(MembershipWriteRefused::NotLabelsAlone));
    let wrote = door.refreshed(seen, &built("m", "silver"), &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let calls: Vec<_> = logged.lines().skip(1).collect();
    assert_eq!(calls.len(), 2, "{logged}");
    assert!(
        calls
            .iter()
            .all(|line| line.contains("worker label refresh")),
        "{logged}"
    );
    assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
    assert!(
        calls[0].contains(MembershipWriteRefused::NOT_LABELS_ALONE),
        "{logged}"
    );
    assert!(calls[1].contains("made=\"relabelled\""), "{logged}");
    assert!(!logged.contains(HTTP), "{logged}");
}

/// The controller's door reads a built worker's facts alone. Its log
/// carries no pairing descriptor and its room test refuses a row that
/// states one. So a worker that reports a transport and a version is
/// registered and refreshed here, and its row states no descriptor.
#[test]
fn a_worker_that_reports_pairing_facts_is_laid_with_no_descriptor() {
    const HTTP: &str = "http://10.0.0.6:8000";
    let (store, door) = hosted_door();
    let built = |version: &str| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(HTTP)
            .model(ModelCard::new("m"))
            .runtime_type(smg_worker::worker::RuntimeType::Sglang)
            .label("disaggregation_transfer_backend", "nixl");
        Arc::new(worker.label("version", version).build())
    };
    let first = built("1");
    let offered = first.pd_pairing();
    assert_eq!(
        (offered.transport(), offered.version()),
        (Some("nixl"), Some("1"))
    );
    let wrote = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
    assert!(wrote.is_ok(), "{wrote:?}");
    let row = store.by_key(HTTP).expect("the registered row");
    assert_eq!(row.facts.pairing, None);

    let wrote = door.refreshed(standing(&row), &built("2"), &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let refreshed = store.by_key(HTTP).expect("the refreshed row");
    assert_eq!(refreshed.facts.pairing, None);
    let version = refreshed.facts.labels.get("version");
    assert_eq!(version.map(String::as_str), Some("2"));
}

/// A registration may state the default's own window in so many words:
/// its row keeps the number, and the object built from the row waits it.
/// A refresh reads that object back as its row states it, so it moves the
/// row's labels, as it does for a row that states another window or none.
#[test]
fn a_row_that_states_the_default_window_takes_a_label_refresh() {
    use smg_worker::worker::HealthCheckConfig;

    let default = u32::try_from(HealthCheckConfig::default().drain_settle_secs)
        .expect("the default window in a row's width");
    let (store, door) = door();
    let door = Arc::new(door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    for (host, window) in [(1, None), (2, Some(default + 40)), (3, Some(default))] {
        let url = format!("grpc://10.0.1.{host}:8000");
        let stated = proto::WorkerSpec {
            drain_settle_secs: window,
            ..unchecked(&url)
        };
        door.register(&operator(), under(TERM), &ask(stated, Mode::Add))
            .expect("one row");
        let row = store.by_key(&url).expect("the registered row");
        assert_eq!(row.facts.drain_settle_secs, window);
        // What a refresh rebuilt: the row's object under another label.
        let first = crate::worker_host::object(&row);
        let relabelled = WorkerRow {
            facts: WorkerFacts {
                labels: [("tier".to_owned(), "silver".to_owned())].into(),
                ..row.facts.clone()
            },
            ..row.clone()
        };
        let built = crate::worker_host::object(&relabelled);
        let wrote = hosted.refreshed(standing(&row), &built, &first);
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::InPlace(row.id)),
            "a row that states {window:?}"
        );
        let refreshed = store.by_key(&url).expect("the refreshed row");
        assert_eq!(refreshed.facts.drain_settle_secs, window);
        let tier = refreshed.facts.labels.get("tier");
        assert_eq!(tier.map(String::as_str), Some("silver"));
    }
}

#[test]
fn every_write_of_a_provider_leaves_its_audit_line() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let key = "private/provider-key";
    let maker = found(key);
    let rank_urls = ["grpc://10.0.0.5:8000@0", "grpc://10.0.0.5:8000@1"];
    let states = [0, 1].map(|rank: u16| {
        let key = rank_urls[usize::from(rank)];
        move |_: Option<&WorkerFacts>| WorkerFacts {
            key: key.to_owned(),
            dp_rank: rank,
            dp_size: 2,
            ..found_facts()
        }
    });
    let ranks = [0, 1].map(|rank| Stated {
        url: rank_urls[rank],
        state: &states[rank],
    });
    hosted
        .rows(&maker, Precondition::Any, &ranks)
        .expect("registered ranks");
    let [first, second] = rank_urls.map(|url| store.by_key(url).expect("a rank").id.0);
    hosted
        .rows(&maker, Precondition::Any, &ranks)
        .expect("unmoved ranks");

    hosted
        .rows(
            &maker,
            Precondition::Any,
            &[Stated {
                url: URL,
                state: &|_| found_facts(),
            }],
        )
        .expect("registered plain and retired ranks");
    let id = store.by_key(URL).expect("the registered row").id.0;
    let refreshed = || WorkerFacts {
        max_running: Some(8),
        ..found_facts()
    };
    hosted
        .found_one(&maker, URL, &|_| refreshed())
        .expect("relabelled");
    hosted
        .found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..refreshed()
        })
        .expect("replaced");
    let row = store.by_key(URL).expect("the replaced row");
    let new = row.id.0;
    let seen = standing(&row);
    let by = Remover::Registrant(&maker);
    hosted.drained(by, URL, seen).expect("drained");
    hosted.drained(by, URL, seen).expect("unmoved");
    hosted
        .gone(Remover::Registrant(&found("another/key")), URL, seen)
        .expect_err("another maker");
    hosted
        .gone(Remover::Provider("cluster"), URL, seen)
        .expect("gone");
    hosted.gone(by, URL, seen).expect_err("already gone");

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let registration = "worker registration";
    let want = [
        (registration, "registered", Some((first, 1, 2))),
        (registration, "registered", Some((second, 2, 3))),
        (registration, "nothing", Some((first, 2, 3))),
        (registration, "nothing", Some((second, 2, 3))),
        (registration, "registered", Some((id, 3, 4))),
        (registration, "removed", Some((first, 4, 5))),
        (registration, "removed", Some((second, 5, 6))),
        (registration, "relabelled", Some((id, 6, 7))),
        (registration, "replaced", Some((new, 7, 8))),
        ("worker drain", "status", Some((new, 8, 9))),
        ("worker drain", "nothing", Some((new, 8, 9))),
        (
            "worker removal",
            "code=PermissionDenied why=\"the row is provider cluster's under another key",
            None,
        ),
        ("worker removal", "removed", Some((new, 9, 10))),
        (
            "worker removal",
            "code=NotFound why=\"no worker is registered under the URL\"",
            None,
        ),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (call, detail, place)) in lines.iter().zip(want) {
        assert!(
            line.contains(&format!("call=\"{call}\" by=operator-a")),
            "{line}"
        );
        assert_eq!(
            line.contains("asker=\"cluster\""),
            call != registration,
            "{line}"
        );
        if let Some((worker, epoch, seq)) = place {
            let fields = format!("worker={worker:016x} made=\"{detail}\" membership_epoch={epoch} seq={seq} term={TERM}");
            assert!(
                line.contains("membership write") && line.contains(&fields),
                "{line}"
            );
        } else {
            assert!(
                line.contains("membership call refused") && line.contains(detail),
                "{line}"
            );
        }
    }
    for private in [URL, key, "another/key"] {
        assert!(!logged.contains(private), "{logged}");
    }
}

/// A provider's write on a replica that does not lead leaves a refusal's
/// audit line under the call's own name: the code and the words that
/// call's read of the fence answered, and an asker on the drain's line and
/// on the removal's alone. No line holds the URL or the provider's key.
#[test]
fn a_write_refused_where_the_replica_does_not_lead_leaves_its_audit_line() {
    // What the fence answers at each read, another code or other words
    // each time: a line carries the code and the words of its own call.
    const READS: [(Code, &str); 3] = [
        (Code::FailedPrecondition, "this replica does not lead"),
        (Code::Unavailable, "the lease of this replica ran out"),
        (Code::FailedPrecondition, "this replica follows another"),
    ];
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    let reads = AtomicUsize::new(0);
    let follower = ReplicaMembershipWriter::new(Arc::new(door), operator(), move || {
        let (code, words) = READS[reads.fetch_add(1, Ordering::Relaxed)];
        Err(Status::new(code, words))
    });
    let key = "private/provider-key";
    let maker = found(key);
    let refused = |read: usize| {
        let (_, words) = READS[read];
        Err(MembershipWriteRefused::NotLeading(words.to_owned()))
    };
    assert_eq!(
        follower.found_one(&maker, URL, &|_| found_facts()),
        refused(0)
    );
    let seen = RowRevision {
        row: WorkerId(1),
        revision: 1,
    };
    let by = Remover::Provider("cluster");
    assert_eq!(follower.drained(by, URL, seen), refused(1));
    assert_eq!(follower.gone(by, URL, seen), refused(2));

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let calls = [
        ("worker registration", false),
        ("worker drain", true),
        ("worker removal", true),
    ];
    assert_eq!(lines.len(), calls.len(), "{logged}");
    for ((line, (call, asked)), (code, words)) in lines.iter().zip(calls).zip(READS) {
        let refusal = format!("call=\"{call}\" by=operator-a code={code:?} why=\"{words}\"");
        assert!(
            line.contains("membership call refused") && line.contains(&refusal),
            "{line}"
        );
        assert_eq!(line.contains("asker="), asked, "{line}");
        assert!(!asked || line.contains("asker=\"cluster\""), "{line}");
    }
    for private in [URL, key] {
        assert!(!logged.contains(private), "{logged}");
    }
}

/// A refusal's audit line is written once its call has let the door's
/// turn go, at both doors: the line of a control call the fence refuses,
/// of a provider's write the fence refuses, and of a provider's write the
/// door refuses under a term that leads.
#[test]
fn a_refusals_audit_line_is_written_once_the_turn_is_let_go() {
    let (_store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    // The row stands before the lines are kept: a write's own line is
    // written inside the turn.
    let leader = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    leader
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");

    // The lines' writer reads the door's turn as each line reaches it.
    let lines = Logged::default();
    let held = Arc::new(AtomicUsize::new(0));
    let (writer, watched, seen) = (lines.clone(), Arc::clone(&door), Arc::clone(&held));
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || {
            seen.fetch_add(usize::from(watched.turn.is_locked()), Ordering::Relaxed);
            writer.clone()
        })
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let follows = || -> Result<u64, Status> {
        Err(Status::failed_precondition("this replica does not lead"))
    };
    door.register(&operator(), follows, &ask(spec(URL), Mode::Add))
        .expect_err("the fence refuses the control call");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), follows)
        .found_one(&maker, URL, &|_| found_facts())
        .expect_err("the fence refuses the provider's write");
    assert_eq!(
        leader.found(&maker, Precondition::Absent, URL, &|_| found_facts()),
        Err(MembershipWriteRefused::Stands)
    );

    // A line a refusal, in the calls' order, each told by its call and
    // its code.
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let want = [
        ("RegisterWorker", "FailedPrecondition"),
        ("worker registration", "FailedPrecondition"),
        ("worker registration", "AlreadyExists"),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (call, code)) in lines.iter().zip(want) {
        let refusal = format!("call=\"{call}\" by=operator-a code={code} ");
        assert!(
            line.contains("membership call refused") && line.contains(&refusal),
            "{line}"
        );
    }
    assert_eq!(
        held.load(Ordering::Relaxed),
        0,
        "a line arrived with the turn held: {logged}"
    );
}

#[test]
fn a_registration_states_its_bootstrap_port_and_a_change_of_it_is_made_in_place() {
    let (store, door) = door();
    let with = |bootstrap_port| proto::WorkerSpec {
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port,
            ..Default::default()
        }),
        ..spec(URL)
    };
    let port =
        |told: &proto::WorkerEvent| row_of(told).pairing.map(|pairing| pairing.bootstrap_port);
    let told = door
        .register(&operator(), under(TERM), &ask(with(8_998), Mode::Add))
        .expect("registered");
    let first = row_of(&told);
    assert_eq!(port(&told), Some(8_998));
    assert_eq!(
        store.by_key(URL).and_then(|row| row.facts.bootstrap_port),
        Some(8_998)
    );

    // Another port: the same worker, its facts moved.
    let told = door
        .register(&operator(), under(TERM), &ask(with(9_001), Mode::Upsert))
        .expect("relabelled");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let moved = row_of(&told);
    assert_eq!((moved.id, moved.revision), (first.id, first.revision + 1));
    assert_eq!(port(&told), Some(9_001));

    // A registration that states no descriptor leaves the port: equal
    // facts, nothing written.
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("answered");
    assert_eq!(port(&told), Some(9_001));
    assert_eq!((told.membership_epoch, store.epoch()), (2, 2));

    // A REPLACE that states no descriptor is the worker's next
    // incarnation and keeps the port that stands.
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect("replaced");
    let next = row_of(&told);
    assert_ne!(next.id, first.id, "a life of its own");
    assert_eq!(port(&told), Some(9_001));

    // Removed and registered again, the worker tells no port.
    let removal = proto::DeregisterWorkerRequest {
        worker_id: next.id,
        revision: next.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered again");
    assert_eq!(row_of(&told).pairing, None);

    // A worker registered with no port tells no descriptor.
    let other = "grpc://10.0.0.6:8000";
    let told = door
        .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    assert_eq!(row_of(&told).pairing, None);
}

#[test]
fn a_life_is_numbered_by_the_term_above_the_stores_epoch() {
    assert_eq!(life(7, 0), (7 << 40) | 1);
    assert_eq!(life(7, 12), (7 << 40) | 13);
    assert_eq!(super::life(7, 5, 12), (7 << 40) | 18, "from its base");
    // The highest base with the highest epoch fills the low bits and
    // carries nothing into the term's.
    let highest = super::life(7, BOUND - 1, BOUND - 1);
    assert_eq!(highest, (7 << 40) | ((1 << 40) - 1));
    assert!(
        highest < life(8, 0),
        "a later term stands above every count of an earlier one"
    );

    // Two leaders that take the same URL first number two lives: the
    // pair read at one names no row at the other.
    let (_, earlier) = door_in(7);
    let (store, later) = door_in(8);
    let first = row_of(
        &earlier
            .register(&operator(), under(7), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let second = row_of(
        &later
            .register(&operator(), under(8), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(
        (first.incarnation, second.incarnation),
        (life(7, 0), life(8, 0))
    );
    assert_ne!(first.id, second.id);
    assert_eq!(first.revision, second.revision);
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    let refused = later
        .deregister(&operator(), under(8), &removal)
        .expect_err("another leader's registration");
    assert_eq!(refused.code(), Code::NotFound);
    assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(second.id));
}

#[test]
fn a_life_is_minted_only_under_a_term_its_log_leads() {
    let (store, door) = door();
    let refused = |told: Result<proto::WorkerEvent, Status>| {
        let status = told.expect_err("no life under a term the log does not lead");
        assert_eq!(
            (status.code(), status.message()),
            (Code::Unavailable, super::LEADS_NO_TERM)
        );
    };
    // The call was admitted under a term the log has left, or one it
    // has not entered: the two terms are compared at the mint.
    for admitted in [TERM - 1, TERM + 1] {
        let told = door.register(&operator(), under(admitted), &ask(spec(URL), Mode::Add));
        refused(told);
    }
    // The store follows another leader's history at the admitted term:
    // a number under it could be one that leader gave another worker.
    door.log
        .restart(Position {
            seq: 40,
            term: TERM + 1,
        })
        .expect("a place the log takes");
    let told = door.register(&operator(), under(TERM + 1), &ask(spec(URL), Mode::Add));
    refused(told);
    assert_eq!(store.epoch(), 0, "a refused registration writes nothing");
    assert!(store.by_key(URL).is_none());
    // A take above every term the log has seen starts a lead again.
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let row = row_of(
        &door
            .register(&operator(), under(TERM + 2), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(row.incarnation, life(TERM + 2, 0));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a term past its 24 bits")]
fn a_term_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = life(1 << 24, 0);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "an epoch past its 39 bits")]
fn an_epoch_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = life(7, BOUND);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a base past its 39 bits")]
fn a_base_past_its_bits_is_asserted_in_a_debug_build() {
    let _ = super::life(7, BOUND, 0);
}

#[test]
fn a_run_draws_its_base_below_the_bound_and_apart_from_another_runs() {
    let drawn: Vec<u64> = (0..64)
        .map(|_| {
            let (store, feed) = journaled(&LeasePlace::Process, 1);
            let objects = holder(&store);
            MembershipCalls::new(store, feed.log, objects).base
        })
        .collect();
    assert!(drawn.iter().all(|&base| base < BOUND), "{drawn:?}");
    let mut apart = drawn.clone();
    apart.sort_unstable();
    apart.dedup();
    assert_eq!(apart.len(), drawn.len(), "{drawn:?}");
}

#[test]
fn two_runs_under_one_term_number_a_urls_lives_apart() {
    // A controller with no shared lease leads under term one at every
    // start: each run's door draws where its count starts.
    let run = || {
        let (store, feed) = journaled(&LeasePlace::Process, 1);
        store::LeasesStore::set_term(store.as_ref(), 1);
        let door = MembershipCalls::new(Arc::clone(&store), feed.log, holder(&store));
        let first = row_of(
            &door
                .register(&operator(), under(1), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        (store, door, first)
    };
    let (_, earlier, removed) = run();
    let (store, later, standing) = run();
    assert_ne!(removed.incarnation, standing.incarnation);
    assert_ne!(removed.id, standing.id);
    // Inside one run the count keeps its order: a later life has the
    // higher number.
    let next = row_of(
        &earlier
            .register(&operator(), under(1), &ask(spec(URL), Mode::Replace))
            .expect("replaced"),
    );
    assert!(next.incarnation > removed.incarnation);
    // A removal read at the earlier run names no row of the later one.
    let removal = proto::DeregisterWorkerRequest {
        worker_id: removed.id,
        revision: removed.revision,
        drain: false,
    };
    let refused = later
        .deregister(&operator(), under(1), &removal)
        .expect_err("the earlier run's registration");
    assert_eq!(refused.code(), Code::NotFound);
    assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(standing.id));
}

#[test]
fn a_removal_that_arrives_twice_never_takes_the_urls_next_registration() {
    let (store, door) = door();
    let first = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    // The worker registers again before the removal's retry lands (its
    // first answer was lost): a new row, at the revision the first had.
    let second = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered again"),
    );
    assert_eq!(second.revision, first.revision);
    assert_ne!(second.id, first.id, "a life of its own");
    assert!(second.incarnation > first.incarnation);

    let retried = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("the retry names a registration that left");
    assert_eq!(retried.code(), Code::NotFound);
    let stands = store.by_key(URL).expect("the second registration");
    assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
}

#[test]
fn a_patch_under_a_registration_that_left_never_writes_the_urls_next_one() {
    let (store, door) = door();
    let first = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let removal = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    let second = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered again"),
    );
    // A patch its caller made of the first registration, arriving late.
    let late = proto::PatchWorkerRequest {
        worker_id: first.id,
        labels: [("lane".to_owned(), "late".to_owned())].into(),
        revision: first.revision,
        ..Default::default()
    };
    let refused = door
        .patch(&operator(), under(TERM), &late)
        .expect_err("the patch names a registration that left");
    assert_eq!(refused.code(), Code::NotFound);
    let stands = store.by_key(URL).expect("the second registration");
    assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
    assert!(stands.facts.labels.is_empty());
}

/// The answers of [`CALLERS`] copies of one call let go at one instant.
fn at_once<T: Send>(call: impl Fn() -> Result<T, Status> + Sync) -> Vec<Result<T, Status>> {
    let (ready, go) = (Barrier::new(CALLERS + 1), AtomicBool::new(false));
    thread::scope(|scope| {
        let calls: Vec<_> = (0..CALLERS)
            .map(|_| {
                scope.spawn(|| {
                    ready.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    call()
                })
            })
            .collect();
        ready.wait();
        go.store(true, Ordering::Release);
        calls
            .into_iter()
            .map(|call| call.join().expect("a caller"))
            .collect()
    })
}

/// What the callers that were refused were told.
fn refusals<T>(answers: &[Result<T, Status>]) -> Vec<Code> {
    answers
        .iter()
        .filter_map(|answer| answer.as_ref().err().map(Status::code))
        .collect()
}

#[test]
fn calls_at_once_take_their_turn_at_the_door() {
    // A call's read, its write and its read back are one step under the
    // door's lock. Of the copies of a call let go together one lands and
    // each other is told what it would be told after it, round after
    // round: an ADD that the URL stands, a patch that the row moved, a
    // removal that the row left.
    for round in 0..300 {
        let (store, door) = door();
        let added = at_once(|| door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add)));
        assert_eq!(
            refusals(&added),
            [Code::AlreadyExists; CALLERS - 1],
            "round {round}: {added:?}"
        );
        let row = store.by_key(URL).expect("the row");
        assert_eq!((store.epoch(), row.revision), (1, 1), "round {round}");

        let patch = proto::PatchWorkerRequest {
            worker_id: row.id.0,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            revision: row.revision,
            ..Default::default()
        };
        let patched = at_once(|| door.patch(&operator(), under(TERM), &patch));
        assert_eq!(
            refusals(&patched),
            [Code::FailedPrecondition; CALLERS - 1],
            "round {round}: {patched:?}"
        );

        let removal = proto::DeregisterWorkerRequest {
            worker_id: row.id.0,
            revision: 2,
            drain: false,
        };
        let removed = at_once(|| door.deregister(&operator(), under(TERM), &removal));
        assert_eq!(
            refusals(&removed),
            [Code::NotFound; CALLERS - 1],
            "round {round}: {removed:?}"
        );
        assert_eq!(
            (store.epoch(), store.by_key(URL)),
            (3, None),
            "round {round}"
        );
    }
}
