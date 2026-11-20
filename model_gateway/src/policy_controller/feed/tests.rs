//! Cases of the gateway's membership feed.

use std::{
    error::Error,
    future::Future,
    pin::pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

use parking_lot::Mutex;
use smg_policy_controller::{
    store::{
        HealthPolicy, Maker, MemoryStore, WorkerFacts, WorkerStatus as RowStatus, WorkersStore,
    },
    view::{Role, WireModes, WorkerId},
    worker_host::{Expect, Membership, Refused, Remover, Standing, StoreDoor, Wrote},
};
use smg_worker::{
    connection::engine::EngineConnector,
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    registry::event::WorkerEvent,
    worker::WorkerStatus,
};

use super::{feed, Feed, FeedDoor};
use crate::{
    config::RouterConfig,
    policy_controller::{
        host::{Ask, Host, Unplaced},
        start,
    },
    routers::{
        common::snapshot::WorkerRegistry,
        grpc::backend_client::{BackendConnector, BasicWorkerBuilder},
    },
};

const URL: &str = "http://worker:8000";
const MODEL: &str = "model-a";

fn bed(store: &Arc<MemoryStore>) -> (Arc<WorkerRegistry>, Feed) {
    let registry = Arc::new(WorkerRegistry::new());
    let feed = feed(store, &registry, BackendConnector::shared());
    (registry, feed)
}

fn stated() -> WorkerFacts {
    WorkerFacts {
        key: URL.to_owned(),
        engine: URL.to_owned(),
        role: Role::Single,
        wire_modes: WireModes::HTTP,
        models: vec![MODEL.to_owned()],
        dp_size: 1,
        max_running: Some(64),
        health: HealthPolicy {
            probes: false,
            ..HealthPolicy::default()
        },
        ..WorkerFacts::default()
    }
}

fn register(door: &dyn Membership) -> Result<WorkerId, String> {
    register_with(door, stated)
}

fn register_with(door: &dyn Membership, facts: fn() -> WorkerFacts) -> Result<WorkerId, String> {
    match door.found(&Maker::Control, Expect::Nothing, URL, &|_| facts()) {
        Ok(Wrote::Registered(id)) => Ok(id),
        other => Err(format!("expected a registration, got {other:?}")),
    }
}

fn probed() -> WorkerFacts {
    WorkerFacts {
        health: HealthPolicy {
            failure_threshold: 2,
            success_threshold: 2,
            probes: true,
            ..HealthPolicy::default()
        },
        ..stated()
    }
}

fn start_store() -> Result<Arc<MemoryStore>, Box<dyn Error>> {
    let store = start(&RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    })?;
    store.ok_or_else(|| "the controller was picked and no store started".into())
}

fn ask(id: &[u8]) -> Ask<'_> {
    Ask::of(id, MODEL)
}

#[test]
fn a_write_at_the_door_puts_the_workers_object_in_the_registry() {
    // A connector of the case's own: the object's is the one handed over,
    // and no stock one by name.
    let handed: Arc<dyn EngineConnector> = Arc::new(BackendConnector);
    let registry = Arc::new(WorkerRegistry::new());
    let feed = feed(
        &Arc::new(MemoryStore::default()),
        &registry,
        Arc::clone(&handed),
    );
    let row = register(&*feed.door).expect("the worker registers");
    let worker = registry.get_by_url(URL).expect("the worker stands");
    let object = feed.objects.object(row).expect("the row has an object");
    assert_eq!(registry.get_id_by_url(URL), Some(object));
    let connector = worker
        .engine_connector()
        .expect("the worker has a connector");
    assert!(Arc::ptr_eq(&connector, &handed));
}

#[tokio::test]
async fn a_written_row_is_in_force_for_the_next_decision() {
    let store = start_store().expect("the store starts");
    let (_, feed) = bed(&store);
    let host = Host::new(store, Arc::clone(&feed.objects));
    let row = register(&*feed.door).expect("the worker registers");
    let placed = host
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the next decision picks the worker");
    assert_eq!(placed.id(), row);
}

#[tokio::test]
async fn a_failed_probe_is_written_on_the_row_and_the_object_mirrors_it() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let host = Host::new(Arc::clone(&store), Arc::clone(&feed.objects));
    let row = register_with(&*feed.door, probed).expect("the worker registers");
    let object = feed.objects.object(row).expect("the row has an object");
    let worker = registry.get(&object).expect("the worker stands");
    let revision = worker.revision();
    let sink = feed.health_sink();

    let assert_statuses = |row_status, object_status| {
        assert_eq!(store.by_key(URL).map(|row| row.status), Some(row_status));
        assert_eq!(worker.status(), object_status);
    };
    assert_statuses(RowStatus::Pending, WorkerStatus::Pending);
    assert_eq!(
        sink.connected(&object, revision),
        Some((WorkerStatus::Pending, WorkerStatus::Ready))
    );
    assert_statuses(RowStatus::Ready, WorkerStatus::Ready);
    let placed = host
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the ready row takes a decision");
    assert_eq!(placed.id(), row);
    drop(placed);

    let probe = |passed, launched| sink.probe_outcome(&object, passed, revision, launched);
    assert_eq!(
        probe(false, WorkerStatus::Ready),
        ProbeAnswer::Stood(WorkerStatus::Ready)
    );
    assert_statuses(RowStatus::Ready, WorkerStatus::Ready);
    assert_eq!(
        probe(false, WorkerStatus::Ready),
        ProbeAnswer::Moved {
            old: WorkerStatus::Ready,
            new: WorkerStatus::NotReady
        }
    );
    assert_statuses(RowStatus::NotReady, WorkerStatus::NotReady);
    assert!(matches!(
        host.ask(&ask(b"down"), &[], Instant::now()).await,
        Err(Unplaced::Shed { .. })
    ));

    assert_eq!(
        probe(true, WorkerStatus::NotReady),
        ProbeAnswer::Stood(WorkerStatus::NotReady)
    );
    assert_statuses(RowStatus::NotReady, WorkerStatus::NotReady);
    assert_eq!(
        probe(true, WorkerStatus::NotReady),
        ProbeAnswer::Moved {
            old: WorkerStatus::NotReady,
            new: WorkerStatus::Ready
        }
    );
    assert_statuses(RowStatus::Ready, WorkerStatus::Ready);
    let placed = host
        .place(&ask(b"back"), Instant::now())
        .await
        .expect("the recovered row takes a decision");
    assert_eq!(placed.id(), row);
}

#[test]
fn a_worker_with_no_row_keeps_the_registrys_health() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let classic = WorkerRegistry::new();
    let pending = || -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(URL)
                .status(WorkerStatus::Pending)
                .build(),
        )
    };
    let (worker, reference) = (pending(), pending());
    let id = registry
        .register(Arc::clone(&worker))
        .expect("a worker without a row");
    let reference_id = classic
        .register(Arc::clone(&reference))
        .expect("a classic worker");
    let epoch = store.epoch();
    let sink = feed.health_sink();

    assert_eq!(
        sink.connected(&id, worker.revision()),
        classic.connected(&reference_id, reference.revision())
    );
    assert_eq!(worker.status(), WorkerStatus::Ready);
    for _ in 0..2 {
        assert_eq!(
            sink.probe_outcome(&id, true, worker.revision(), WorkerStatus::Ready),
            classic.probe_outcome(
                &reference_id,
                true,
                reference.revision(),
                WorkerStatus::Ready
            )
        );
        assert_eq!(worker.status(), reference.status());
    }
    assert!(store.snapshot().1.is_empty());
    assert_eq!(store.epoch(), epoch);
}

#[test]
fn a_rows_object_takes_no_step_of_the_registrys_own() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let row = register_with(&*feed.door, probed).expect("the worker registers");
    let object = feed.objects.object(row).expect("the row has an object");
    let worker = registry.get(&object).expect("the worker stands");
    let pass = |sink: &dyn HealthSink| {
        sink.probe_outcome(&object, true, worker.revision(), WorkerStatus::Pending)
    };

    // Leave the object one pass short of ready while the row has counted none.
    assert_eq!(
        pass(registry.as_ref()),
        ProbeAnswer::Stood(WorkerStatus::Pending)
    );
    assert_eq!(
        pass(feed.health_sink().as_ref()),
        ProbeAnswer::Stood(WorkerStatus::Pending)
    );
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(RowStatus::Pending)
    );
    assert_eq!(worker.status(), WorkerStatus::Pending);
    assert_eq!(
        worker
            .runtime_state()
            .expect("the worker's state")
            .total_pending_probes(),
        1
    );
    assert_eq!(
        pass(registry.as_ref()),
        ProbeAnswer::Moved {
            old: WorkerStatus::Pending,
            new: WorkerStatus::Ready
        }
    );
}

#[test]
fn settle_brings_a_row_written_past_the_door_into_the_registry() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    // A follower of a served controller writes the store itself.
    let row = store.upsert(stated()).id();
    assert!(registry.get_by_url(URL).is_none());
    feed.settle();
    let object = feed.objects.object(row).expect("the row has an object");
    assert_eq!(registry.get_id_by_url(URL), Some(object));
}

/// The store shows a replacement inside its own write, and with probes
/// off the door readies it there, ahead of the listener. A door with no
/// listener leaves that state standing: the registry holds the old object.
#[tokio::test]
async fn a_replacement_takes_no_request_until_its_object_is_laid() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let host = Host::new(Arc::clone(&store), Arc::clone(&feed.objects));
    register(&*feed.door).expect("the worker registers");
    let old = registry.get_by_url(URL).expect("the first object stands");

    let replaced =
        StoreDoor::new(Arc::clone(&store)).found(&Maker::Control, Expect::Anything, URL, &|_| {
            WorkerFacts {
                models: vec![MODEL.to_owned(), "model-b".to_owned()],
                ..stated()
            }
        });
    assert!(
        matches!(replaced, Ok(Wrote::Another { .. })),
        "{replaced:?}"
    );
    let stale = registry.get_by_url(URL).expect("the old object stands");
    assert!(Arc::ptr_eq(&stale, &old));

    let gap = host
        .place(&ask(b"gap"), Instant::now())
        .await
        .expect_err("a row with no object takes no request");
    assert!(matches!(gap, Unplaced::NoTargetHolds), "{gap:?}");

    feed.settle();
    let placed = host
        .place(&ask(b"laid"), Instant::now())
        .await
        .expect("the replacement's object takes the request");
    assert!(!Arc::ptr_eq(placed.worker(), &old));
}

struct TurnAtRegistration {
    feed: Feed,
    held: OnceLock<bool>,
}

impl Wake for TurnAtRegistration {
    fn wake(self: Arc<Self>) {
        self.held
            .get_or_init(|| self.feed.turn.try_lock().is_none());
    }
}

#[test]
fn a_settle_holds_the_feeds_turn() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let at_registration = Arc::new(TurnAtRegistration {
        feed: feed.clone(),
        held: OnceLock::new(),
    });
    let waker = Waker::from(Arc::clone(&at_registration));
    let mut events = registry.subscribe_events();
    let mut event = pin!(events.recv());
    // Read the turn on the thread that lays the row's object.
    assert!(event
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());

    store.upsert(stated());
    feed.settle();
    assert_eq!(at_registration.held.get(), Some(&true));
}

#[test]
fn a_second_write_of_the_same_facts_moves_nothing() {
    let (registry, feed) = bed(&Arc::new(MemoryStore::default()));
    let row = register(&*feed.door).expect("the worker registers");
    let before = registry.get_by_url(URL).expect("the worker stands");
    assert_eq!(
        feed.door
            .found(&Maker::Control, Expect::Anything, URL, &|_| stated()),
        Ok(Wrote::Unmoved(row))
    );
    let after = registry.get_by_url(URL).expect("the worker still stands");
    assert!(Arc::ptr_eq(&before, &after));
}

#[tokio::test]
async fn a_drained_row_takes_no_decision_and_a_gone_row_leaves_the_registry() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let host = Host::new(store, Arc::clone(&feed.objects));
    let row = register(&*feed.door).expect("the worker registers");
    let before = host
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the worker is picked before draining");
    assert_eq!(before.id(), row);
    drop(before);

    let object = feed.objects.object(row).expect("the row has an object");
    let seen = feed.objects.standing(&object).expect("the object's row");
    let by = Remover::Maker(&Maker::Control);
    assert_eq!(feed.door.drained(by, URL, seen), Ok(Wrote::Draining(row)));
    let drained = host
        .place(&ask(b"next"), Instant::now())
        .await
        .expect_err("a draining worker takes no decision");
    assert!(matches!(drained, Unplaced::Shed { .. }), "{drained:?}");
    assert!(registry.get_by_url(URL).is_some());

    assert_eq!(feed.door.gone(by, URL, seen), Ok(Wrote::Removed(row)));
    assert!(registry.get_by_url(URL).is_none());
    assert!(feed.objects.object(row).is_none());
}

#[test]
fn a_refused_write_tells_nobody() {
    let (registry, feed) = bed(&Arc::new(MemoryStore::default()));
    register(&*feed.door).expect("the worker registers");
    let before = registry.get_by_url(URL).expect("the worker stands");
    let another = Maker::provider("pool", "second").expect("a maker");
    assert!(matches!(
        feed.door
            .found(&another, Expect::Anything, URL, &|_| stated()),
        Err(Refused::NotMaker(_))
    ));
    let after = registry.get_by_url(URL).expect("the worker still stands");
    assert!(Arc::ptr_eq(&before, &after));
}

struct AtRegistration {
    store: Arc<MemoryStore>,
    members: OnceLock<Option<usize>>,
}

impl Wake for AtRegistration {
    fn wake(self: Arc<Self>) {
        self.members
            .get_or_init(|| self.store.world().members(MODEL, Role::Single));
    }
}

#[test]
fn the_registry_tells_of_a_registration_before_the_store_publishes_it() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let at_registration = Arc::new(AtRegistration {
        store: Arc::clone(&store),
        members: OnceLock::new(),
    });
    let waker = Waker::from(Arc::clone(&at_registration));
    let mut events = registry.subscribe_events();
    let mut event = pin!(events.recv());
    // Read the published world on the thread sending the registry event.
    assert!(event
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());

    register(&*feed.door).expect("the worker registers");
    assert!(matches!(
        event.poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(WorkerEvent::Registered { .. }))
    ));
    assert_eq!(at_registration.members.get(), Some(&None));
    assert_eq!(store.world().members(MODEL, Role::Single), Some(1));
}

#[test]
fn a_membership_write_holds_the_feed_turn_through_its_listener() {
    let calls = Arc::new(AtomicUsize::new(0));
    let store = Arc::new(MemoryStore::default());
    let door = Arc::new_cyclic(|door: &Weak<FeedDoor>| {
        let door = Weak::clone(door);
        let calls = Arc::clone(&calls);
        FeedDoor {
            door: StoreDoor::new(Arc::clone(&store)).telling(move || {
                let door = door.upgrade().expect("the caller holds the door");
                assert!(door.turn.try_lock().is_none());
                calls.fetch_add(1, Ordering::Relaxed);
            }),
            turn: Arc::new(Mutex::new(())),
        }
    });
    let row = register(&*door).expect("the worker registers");
    let seen = Standing {
        row,
        revision: store.get(row).expect("the worker's row").revision,
    };
    let by = Remover::Maker(&Maker::Control);
    assert_eq!(door.drained(by, URL, seen), Ok(Wrote::Draining(row)));
    assert_eq!(door.gone(by, URL, seen), Ok(Wrote::Removed(row)));
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}
