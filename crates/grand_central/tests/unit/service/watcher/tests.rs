use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::registry::WorkerRegistry;

use super::{WatchPositionLost, Watcher};
use crate::{
    engine::view::WorkerId,
    service::{
        controller::worker_wire,
        proto::{self, worker_event::Event},
        replication::Log,
    },
    store::{
        GatewayId, LeaseRecord, LeaseState, LeasesStore, MembershipChange, MemoryStore, PickRecord,
        Position, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
    },
    tracking::lease::{Lease, LeaseId},
    worker_host::{Members, Objects},
};

const TERM: u64 = 1;

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation,
        models: vec!["m".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// A controller's side of a watch: its store, and the place of the last
/// event it told.
struct Told {
    store: MemoryStore,
    seq: u64,
}

impl Told {
    fn new() -> Self {
        Self {
            store: MemoryStore::new(),
            seq: 100,
        }
    }

    fn row(&self, id: WorkerId) -> WorkerRow {
        WorkersStore::get(&self.store, id).expect("the row")
    }

    /// The watch's event for the write the store just took. `before` is
    /// the row the write replaced or took out.
    fn event(
        &mut self,
        change: MembershipChange,
        before: Option<&WorkerRow>,
    ) -> proto::WorkerEvent {
        // Writes a watch is not told take places between two of its
        // events: the places a watcher is told skip.
        self.seq += 3;
        let event = match change {
            MembershipChange::Registered { id } => worker_wire::registered(&self.row(id)),
            MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
                worker_wire::relabelled(&self.row(id))
            }
            MembershipChange::Replaced { new, .. } => {
                worker_wire::replaced(before.expect("the old row"), &self.row(new), false)
            }
            MembershipChange::Removed { .. } => worker_wire::removed(before.expect("the last row")),
            MembershipChange::StatusChanged { id, from, to } => {
                worker_wire::status_changed(id, from, to)
            }
        };
        worker_wire::event(event, self.seq, WorkersStore::epoch(&self.store), TERM)
    }

    /// The snapshot of the store as it stands, `rows` a page.
    fn pages(&self, rows: usize) -> Vec<proto::WorkerEvent> {
        let cut = self.store.checkpoint();
        let mut chunks: Vec<&[WorkerRow]> = cut.rows.chunks(rows).collect();
        if chunks.is_empty() {
            chunks.push(&[]);
        }
        let last = chunks.len() - 1;
        chunks
            .iter()
            .enumerate()
            .map(|(n, chunk)| {
                let page = proto::WorkerSnapshot {
                    workers: chunk.iter().map(worker_wire::row).collect(),
                    last: n == last,
                };
                worker_wire::event(Event::Snapshot(page), self.seq, cut.epoch, TERM)
            })
            .collect()
    }
}

/// What two stores must agree on: the epoch, and each row's id, facts,
/// status and revision. An admission instant crosses as an age.
fn held(store: &MemoryStore) -> (u64, Vec<(WorkerId, WorkerFacts, WorkerStatus, u64)>) {
    let (epoch, mut rows) = WorkersStore::all_rows(store);
    rows.sort_by_key(|row| row.id.0);
    let rows = rows
        .into_iter()
        .map(|row| (row.id, row.facts, row.status, row.revision))
        .collect();
    (epoch, rows)
}

/// A watcher of `store` over a holder of its own, which follows no
/// status of the store's tail: it lays each status the watch tells.
fn over(store: Arc<MemoryStore>) -> Watcher<Members> {
    let objects = Objects::new(Arc::new(WorkerRegistry::new()));
    Watcher::new(store, Arc::new(objects))
}

/// The watcher's worker objects state the rows its store holds, each
/// read back from its object, and no other object stands. Nothing
/// follows the store here: the watcher's own writes laid them.
#[track_caller]
fn laid(watcher: &Watcher<Members>) {
    let (_, mut rows) = WorkersStore::all_rows(&**watcher.store());
    rows.sort_by_key(|row| row.id.0);
    let objects = watcher.objects();
    let read = |row: &WorkerRow| {
        let object = objects.object(row.id)?;
        objects.row_of(&object)
    };
    let stated: Vec<_> = rows.iter().map(read).collect();
    let rows: Vec<_> = rows.into_iter().map(Some).collect();
    assert_eq!(stated, rows, "each row's object states it");
    assert_eq!(objects.len(), rows.len(), "no other object stands");
}

fn watching(told: &Told, rows: usize) -> Watcher<Members> {
    let mut watcher = over(Arc::new(MemoryStore::new()));
    let request = watcher.request();
    assert_eq!((request.resume_seq, request.resume_term), (0, 0));
    assert!(request.models.is_empty(), "every model");
    assert!(!request.with_needs, "no generation's needs");
    for page in told.pages(rows) {
        assert_eq!(watcher.take(&page, Instant::now()), Ok(()));
    }
    assert_eq!(held(watcher.store()), held(&told.store));
    laid(&watcher);
    watcher
}

/// A watcher holds the controller's rows after a snapshot in pages and
/// after each kind of change the tail tells, and its next request asks
/// for the tail after the last place it took. Its worker objects state
/// the same rows when the snapshot's last page and each change return.
#[test]
fn a_watcher_follows_a_fleet_from_a_snapshot_through_every_kind_of_change() {
    let mut told = Told::new();
    let a = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1)).id();
    let b = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1)).id();
    let mut watcher = watching(&told, 1);
    let now = Instant::now();
    let mut follows = |told: &mut Told, change: MembershipChange, before: Option<&WorkerRow>| {
        let event = told.event(change, before);
        assert_eq!(watcher.take(&event, now), Ok(()), "{change:?}");
        assert_eq!(held(watcher.store()), held(&told.store), "{change:?}");
        laid(&watcher);
    };

    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
    follows(&mut told, change, None);

    let revision = told.row(a).revision;
    let change = WorkersStore::set_status(&told.store, a, WorkerStatus::Ready, revision);
    follows(&mut told, change.expect("a status change"), None);

    let patch = WorkerPatch {
        labels: Some(BTreeMap::from([("zone".to_owned(), "a".to_owned())])),
        priority: None,
        overload: None,
    };
    let revision = told.row(b).revision;
    let change = WorkersStore::patch(&told.store, b, patch, revision);
    follows(&mut told, change.expect("a patch"), None);

    let old = told.row(a);
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 2));
    assert!(matches!(change, MembershipChange::Replaced { .. }));
    follows(&mut told, change, Some(&old));

    let last = told.row(b);
    let change = WorkersStore::remove(&told.store, b, last.revision);
    follows(&mut told, change.expect("a removal"), Some(&last));

    // A generation's needs write no row, and their place is taken.
    told.seq += 2;
    let needs = Event::Needs(proto::NeedsSummary::default());
    let needs = worker_wire::event(needs, told.seq, WorkersStore::epoch(&told.store), TERM);
    assert_eq!(watcher.take(&needs, now), Ok(()));
    assert_eq!(held(watcher.store()), held(&told.store));

    assert_eq!(watcher.snapshots(), 1, "the tail alone behind one snapshot");
    let request = watcher.request();
    assert_eq!((request.resume_seq, request.resume_term), (told.seq, TERM));
    assert!(request.models.is_empty(), "every model");
}

/// An event that does not land where the store stands loses the
/// watcher its place, and the next snapshot restores it. The event
/// lays no object, and the snapshot keeps the object of a row that
/// stands through it.
#[test]
fn a_change_that_does_not_land_loses_the_place_and_a_snapshot_restores_it() {
    let mut told = Told::new();
    let first = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1)).id();
    let mut watcher = watching(&told, 8);
    let object = |watcher: &Watcher<Members>| {
        let objects = watcher.objects();
        let id = objects.object(first).expect("the first row's object");
        objects.registry().get(&id).expect("it stands")
    };
    let stood = object(&watcher);
    let now = Instant::now();
    // The watcher is not told this one.
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
    let _untold = told.event(change, None);
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
    let event = told.event(change, None);
    assert_eq!(watcher.take(&event, now), Err(WatchPositionLost::Apart));
    assert_eq!(watcher.at(), None);
    laid(&watcher);
    assert_eq!(watcher.request().resume_seq, 0);
    for page in told.pages(2) {
        assert_eq!(watcher.take(&page, now), Ok(()));
    }
    assert_eq!(held(watcher.store()), held(&told.store));
    laid(&watcher);
    assert!(
        Arc::ptr_eq(&stood, &object(&watcher)),
        "the object that stood"
    );
    assert_eq!(watcher.request().resume_seq, told.seq);
    assert_eq!(watcher.snapshots(), 2);
}

#[test]
fn a_lease_filed_between_watch_snapshots_stays_on_file() {
    let told = Told::new();
    let id = told.store.upsert(facts("grpc://10.0.0.1:8000", 1)).id();
    let mut watcher = watching(&told, 8);
    let store = Arc::clone(watcher.store());
    let leases: &dyn LeasesStore = &*store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = LeaseId(1);
    leases
        .issue(LeaseRecord {
            lease: Lease {
                id: lease,
                ..Lease::default()
            },
            gateway,
            request: 1,
            target: id,
            picked: PickRecord::default(),
            decode: None,
            untold: false,
            term: 0,
            issued_at: now,
            dispatch_deadline: now + Duration::from_secs(1),
            state: LeaseState::ISSUED,
        })
        .expect("filed");
    assert_eq!(leases.live(gateway), 1);

    let word = worker_wire::start_again(Position::default());
    assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
    assert_eq!(watcher.request().resume_seq, 0);
    assert_eq!(watcher.take(&told.pages(8)[0], now), Ok(()));
    assert_eq!(watcher.snapshots(), 2);
    assert_eq!(leases.live(gateway), 1);
    assert!(leases.get(gateway, lease).is_some());
}

/// The controller's word, an event of the tail with no snapshot behind
/// it or inside one, a page of another snapshot or at a place held, and
/// an event of no known kind each lose the place.
#[test]
fn the_controllers_word_and_an_event_out_of_its_order_lose_the_place() {
    let mut told = Told::new();
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
    let now = Instant::now();
    let pages = told.pages(1);
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
    let event = told.event(change, None);

    let mut watcher = over(Arc::new(MemoryStore::new()));
    assert_eq!(
        watcher.take(&event, now),
        Err(WatchPositionLost::Order),
        "no snapshot"
    );
    assert_eq!(watcher.request().resume_seq, 0);
    assert_eq!(watcher.take(&pages[0], now), Ok(()));
    assert_eq!(
        watcher.take(&event, now),
        Err(WatchPositionLost::Order),
        "inside one"
    );
    // A page of a snapshot at another place, and one of another epoch.
    let (seq, epoch) = (pages[0].seq, pages[0].membership_epoch);
    for (seq, epoch) in [(seq + 1, epoch), (seq, epoch + 1)] {
        assert_eq!(watcher.request().resume_seq, 0);
        assert_eq!(watcher.take(&pages[0], now), Ok(()));
        let page = proto::WorkerSnapshot {
            workers: Vec::new(),
            last: true,
        };
        let other = worker_wire::event(Event::Snapshot(page), seq, epoch, TERM);
        assert_eq!(
            watcher.take(&other, now),
            Err(WatchPositionLost::Order),
            "another's"
        );
    }
    // A page at a place held.
    let mut watcher = watching(&told, 8);
    let again = told.pages(8);
    assert_eq!(
        watcher.take(&again[0], now),
        Err(WatchPositionLost::Order),
        "held"
    );
    // An event of a kind this build does not know.
    let mut watcher = watching(&told, 8);
    let unknown = proto::WorkerEvent {
        seq: told.seq + 1,
        term: TERM,
        ..proto::WorkerEvent::default()
    };
    let lost = watcher.take(&unknown, now);
    assert!(
        matches!(lost, Err(WatchPositionLost::Unread { path: "event", .. })),
        "{lost:?}"
    );

    // The answer of a registration that states several ranks rides no
    // watch: a watcher that meets it lets the stream go by its name.
    let mut watcher = watching(&told, 8);
    let ranks = worker_wire::event(
        Event::Ranks(proto::RankRows::default()),
        told.seq + 1,
        WorkersStore::epoch(&told.store),
        TERM,
    );
    assert_eq!(
        watcher.take(&ranks, now),
        Err(WatchPositionLost::Unread {
            path: "membership.ranks",
            why: "a registration's answer rides no watch",
        })
    );
    assert_eq!(watcher.at(), None);

    let mut watcher = watching(&told, 8);
    let word = worker_wire::start_again(Position::default());
    assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
    assert_eq!(watcher.at(), None);
    assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
}

/// A stream cut inside a snapshot starts the snapshot again, and a row
/// or a change the conversion refuses loses the place.
#[test]
fn a_cut_snapshot_starts_again_and_a_refused_row_loses_the_place() {
    let mut told = Told::new();
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
    let now = Instant::now();
    let mut watcher = over(Arc::new(MemoryStore::new()));
    assert_eq!(watcher.take(&told.pages(1)[0], now), Ok(()));
    assert_eq!(watcher.request().resume_seq, 0);
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
    let _told_to_others = told.event(change, None);
    for page in told.pages(2) {
        assert_eq!(watcher.take(&page, now), Ok(()));
    }
    assert_eq!(held(watcher.store()), held(&told.store));

    let blank = proto::WorkerSnapshot {
        workers: vec![proto::WorkerRow::default()],
        last: true,
    };
    let page = worker_wire::event(Event::Snapshot(blank), told.seq + 1, 9, TERM);
    let mut watcher = watching(&told, 8);
    let word = worker_wire::start_again(Position::default());
    assert_eq!(watcher.take(&word, now), Err(WatchPositionLost::Required));
    assert_eq!(watcher.request().resume_seq, 0);
    let lost = watcher.take(&page, now);
    assert!(
        matches!(lost, Err(WatchPositionLost::Unread { .. })),
        "{lost:?}"
    );
    assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
    let later = told.pages(8);
    assert_eq!(
        watcher.take(&later[0], now),
        Err(WatchPositionLost::Order),
        "let go"
    );

    let mut watcher = watching(&told, 8);
    let blank = Event::Registered(proto::WorkerRow::default());
    let blank = worker_wire::event(blank, told.seq + 1, 9, TERM);
    let lost = watcher.take(&blank, now);
    assert!(
        matches!(lost, Err(WatchPositionLost::Unread { .. })),
        "{lost:?}"
    );
    assert_eq!(watcher.at(), None);
    assert_eq!(held(watcher.store()), held(&told.store), "the rows stand");
}

/// A stream that lost the place is let go: the pages it goes on to
/// tell are refused, so no part of a snapshot changes the store, and
/// the stream the next request opens is taken from its first page.
#[test]
fn a_stream_that_lost_the_place_tells_nothing_more() {
    let told = Told::new();
    for n in 1..=3 {
        WorkersStore::upsert(&told.store, facts(&format!("grpc://10.0.0.{n}:8000"), 1));
    }
    let now = Instant::now();
    let mut watcher = over(Arc::new(MemoryStore::new()));
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.4:8000", 1));
    let before = held(watcher.store());
    let pages = told.pages(1);
    let blank = proto::WorkerSnapshot {
        workers: vec![proto::WorkerRow::default()],
        last: false,
    };
    let (seq, epoch) = (pages[0].seq, pages[0].membership_epoch);
    let blank = worker_wire::event(Event::Snapshot(blank), seq, epoch, TERM);
    assert_eq!(watcher.take(&pages[0], now), Ok(()));
    let lost = watcher.take(&blank, now);
    assert!(
        matches!(lost, Err(WatchPositionLost::Unread { .. })),
        "{lost:?}"
    );
    for page in &pages[2..] {
        assert_eq!(watcher.take(page, now), Err(WatchPositionLost::Order));
    }
    assert_eq!(watcher.at(), None);
    assert_eq!(held(watcher.store()), before, "the rows stand");

    assert_eq!(watcher.request().resume_seq, 0);
    for page in &pages {
        assert_eq!(watcher.take(page, now), Ok(()));
    }
    assert_eq!(held(watcher.store()), held(&told.store));
}

/// The watcher loses its place when the store's journal refuses a
/// snapshot. The store keeps its rows.
#[test]
fn a_snapshot_a_stores_journal_refuses_loses_the_place() {
    let told = Told::new();
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
    let store = Arc::new(MemoryStore::journaled(Arc::new(Log::new(8))));
    let mut watcher = over(Arc::clone(&store));
    // A place under no term, which a log takes from nobody.
    let mut page = told.pages(8).remove(0);
    page.term = 0;
    assert_eq!(
        watcher.take(&page, Instant::now()),
        Err(WatchPositionLost::Apart)
    );
    assert_eq!(watcher.at(), None);
    assert_eq!(held(&store).1, Vec::new(), "the store stays empty");
    assert!(watcher.objects().is_empty(), "and no object is laid");
}

/// A watcher that followed one registration behind its snapshot, with
/// the controller's side and that event.
fn followed() -> (Told, Watcher<Members>, proto::WorkerEvent) {
    let mut told = Told::new();
    WorkersStore::upsert(&told.store, facts("grpc://10.0.0.1:8000", 1));
    let mut watcher = watching(&told, 8);
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.2:8000", 1));
    let event = told.event(change, None);
    assert_eq!(watcher.take(&event, Instant::now()), Ok(()));
    (told, watcher, event)
}

/// An event of the tail at a place not past the last loses the place:
/// the last event told again, a generation's needs behind it, and an
/// event under a term below the place's. So does a change at a new
/// place stamped with the epoch the store holds already: it moves no
/// epoch, so it did not land.
#[test]
fn a_place_not_past_the_last_and_an_epoch_held_already_lose_the_place() {
    let now = Instant::now();
    let lost =
        |watcher: &mut Watcher<Members>, event: &proto::WorkerEvent, by: WatchPositionLost| {
            let rows = held(watcher.store());
            assert_eq!(watcher.take(event, now), Err(by));
            assert_eq!(watcher.at(), None);
            assert_eq!(held(watcher.store()), rows, "the rows stand");
            laid(watcher);
            assert_eq!(watcher.request().resume_seq, 0);
        };

    let (_, mut watcher, event) = followed();
    lost(&mut watcher, &event, WatchPositionLost::Order);

    let (told, mut watcher, _) = followed();
    let epoch = WorkersStore::epoch(&told.store);
    let needs = Event::Needs(proto::NeedsSummary::default());
    let needs = worker_wire::event(needs, told.seq - 1, epoch, TERM);
    lost(&mut watcher, &needs, WatchPositionLost::Order);

    let (mut told, mut watcher, _) = followed();
    let change = WorkersStore::upsert(&told.store, facts("grpc://10.0.0.3:8000", 1));
    let mut older = told.event(change, None);
    older.term = TERM - 1;
    lost(&mut watcher, &older, WatchPositionLost::Order);

    let (_, mut watcher, event) = followed();
    let mut again = event;
    again.seq += 1;
    lost(&mut watcher, &again, WatchPositionLost::Apart);
}
