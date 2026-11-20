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
    store::{HealthPolicy, Maker, MemoryStore, WorkerFacts, WorkersStore},
    view::{Role, WireModes, WorkerId},
    worker_host::{Membership, Refused, Remover, Standing, StoreDoor, Wrote},
};
use smg_worker::registry::event::WorkerEvent;

use super::{feed, Feed, FeedDoor};
use crate::{
    config::RouterConfig,
    policy_controller::{
        host::{Ask, Host, Unplaced},
        start,
    },
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BackendConnector},
};

const URL: &str = "http://worker:8000";
const MODEL: &str = "model-a";

fn bed(store: &Arc<MemoryStore>) -> (Arc<WorkerRegistry>, Feed) {
    let registry = Arc::new(WorkerRegistry::new());
    let feed = feed(store, &registry);
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
    match door.found(&Maker::Control, URL, &|_| stated()) {
        Ok(Wrote::Registered(id)) => Ok(id),
        other => Err(format!("expected a registration, got {other:?}")),
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
    Ask {
        id,
        model: MODEL,
        streaming: false,
        headers: None,
        rid: None,
        tried: &[],
        hashes: None,
    }
}

#[test]
fn a_write_at_the_door_puts_the_workers_object_in_the_registry() {
    let (registry, feed) = bed(&Arc::new(MemoryStore::default()));
    let row = register(&*feed.door).expect("the worker registers");
    let worker = registry.get_by_url(URL).expect("the worker stands");
    let object = feed.objects.object(row).expect("the row has an object");
    assert_eq!(registry.get_id_by_url(URL), Some(object));
    let connector = worker
        .engine_connector()
        .expect("the worker has a connector");
    assert!(Arc::ptr_eq(&connector, &BackendConnector::shared()));
}

#[tokio::test]
async fn a_written_row_is_in_force_for_the_next_decision() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let host = Host::new(store, registry);
    let row = register(&*feed.door).expect("the worker registers");
    let placed = host
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the next decision picks the worker");
    assert_eq!(placed.id(), row);
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
        feed.door.found(&Maker::Control, URL, &|_| stated()),
        Ok(Wrote::Unmoved(row))
    );
    let after = registry.get_by_url(URL).expect("the worker still stands");
    assert!(Arc::ptr_eq(&before, &after));
}

#[tokio::test]
async fn a_drained_row_takes_no_decision_and_a_gone_row_leaves_the_registry() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let host = Host::new(store, Arc::clone(&registry));
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
        feed.door.found(&another, URL, &|_| stated()),
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
