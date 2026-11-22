//! Cases of the feed of a gateway that follows a controller.

use std::{
    error::Error,
    future::Future,
    pin::pin,
    sync::{Arc, OnceLock},
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

use smg_policy_controller::{
    engine::{LoadRow, Role, WireModes, WorkerId},
    store::{
        HealthPolicy, LoadSink, LoadSource, MemoryStore, Overload, Registrant, WorkerFacts,
        WorkersStore,
    },
    worker_host::{
        MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Precondition,
        StoreMembershipWriter,
    },
};
use smg_worker::{identity::basic::Worker, registry::event::WorkerEvent};

use super::{following, Feed};
use crate::{
    config::RouterConfig,
    policy_controller::{
        engine::{Ask, PolicyEngine, Unplaced},
        start,
    },
    routers::{
        common::{
            snapshot::WorkerRegistry,
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        grpc::backend_client::BackendConnector,
    },
};

const URL: &str = "http://worker:8000";
const MODEL: &str = "model-a";

/// A registry, and the feed over `store` that keeps it.
fn bed(store: &Arc<MemoryStore>) -> (Arc<WorkerRegistry>, Feed) {
    let registry = Arc::new(WorkerRegistry::new());
    let feed = following(store, &registry, BackendConnector::shared());
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

/// Writes the worker's row at the store's own door, as `facts` states it.
/// With probes off that door readies the row. Nothing settles the feed:
/// the case does, as the follower does behind a write of its own.
fn write(
    store: &Arc<MemoryStore>,
    expect: Precondition,
    facts: &dyn Fn() -> WorkerFacts,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    StoreMembershipWriter::new(Arc::clone(store))
        .found(&Registrant::Control, expect, URL, &|_| facts())
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
fn a_settle_brings_a_row_the_follower_wrote_into_the_registry() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    // A follower of a served controller writes the store itself.
    let row = store.upsert(stated()).id();
    assert!(registry.get_by_url(URL).is_none());
    feed.settle();
    let object = feed.objects.object(row).expect("the row has an object");
    assert_eq!(registry.get_id_by_url(URL), Some(object));
}

/// The pool-size gauge's sample for the group of `model`, or none where
/// the group never had a worker.
fn pool_size(
    handle: &metrics_exporter_prometheus::PrometheusHandle,
    model: &str,
) -> Option<String> {
    let model = format!("model=\"{model}\"");
    handle
        .render()
        .lines()
        .find(|line| line.starts_with("smg_worker_pool_size{") && line.contains(&model))
        .and_then(|line| line.rsplit(' ').next().map(str::to_owned))
}

/// A row this gateway follows moves the pool-size gauge with no workflow
/// step beside it. The settle that lays its object counts it, and the
/// settle that takes the object out leaves the group at zero.
#[test]
fn a_followed_row_moves_the_pool_size_gauge_as_it_joins_and_as_it_leaves() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let store = Arc::new(MemoryStore::default());
        let (registry, feed) = bed(&store);
        // A follower of a served controller writes the store itself.
        let row = store.upsert(stated()).id();
        feed.settle();
        assert!(feed.objects.object(row).is_some());
        assert_eq!(pool_size(&handle, MODEL).as_deref(), Some("1"));

        let stands = store.by_key(URL).expect("the row");
        assert!(store.remove(stands.id, stands.revision).is_some());
        feed.settle();
        assert!(registry.get_by_url(URL).is_none());
        assert_eq!(pool_size(&handle, MODEL).as_deref(), Some("0"));
    });
}

/// The store shows a replacement inside its own write, and with probes
/// off its door readies it there. Until the feed settles, the registry
/// holds the old object and the replacement takes no request.
#[tokio::test]
async fn a_replacement_takes_no_request_until_its_object_is_laid() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let policy_engine = PolicyEngine::new(Arc::clone(&store), Arc::clone(&feed.objects));
    let first = write(&store, Precondition::Absent, &stated);
    assert!(
        matches!(first, Ok(MembershipWriteOutcome::Registered(_))),
        "{first:?}"
    );
    feed.settle();
    let old = registry.get_by_url(URL).expect("the first object stands");

    let replaced = write(&store, Precondition::Any, &|| WorkerFacts {
        models: vec![MODEL.to_owned(), "model-b".to_owned()],
        ..stated()
    });
    assert!(
        matches!(replaced, Ok(MembershipWriteOutcome::Another { .. })),
        "{replaced:?}"
    );
    let stale = registry.get_by_url(URL).expect("the old object stands");
    assert!(Arc::ptr_eq(&stale, &old));

    let gap = policy_engine
        .place(&ask(b"gap"), Instant::now())
        .await
        .expect_err("a row with no object takes no request");
    assert!(matches!(gap, Unplaced::NoTargetHolds), "{gap:?}");

    feed.settle();
    let placed = policy_engine
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

struct AtRegistration {
    store: Arc<MemoryStore>,
    members: OnceLock<Option<usize>>,
}

impl Wake for AtRegistration {
    fn wake(self: Arc<Self>) {
        self.members
            .get_or_init(|| self.store.snapshot().members(MODEL, Role::Single));
    }
}

/// A settle lays a row's object before the store publishes the row: at the
/// registry's word of the registration no decision can read the row yet.
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

    let wrote = write(&store, Precondition::Absent, &stated);
    assert!(
        matches!(wrote, Ok(MembershipWriteOutcome::Registered(_))),
        "{wrote:?}"
    );
    feed.settle();
    assert!(matches!(
        event.poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(WorkerEvent::Registered { .. }))
    ));
    assert_eq!(at_registration.members.get(), Some(&None));
    assert_eq!(store.snapshot().members(MODEL, Role::Single), Some(1));
}

/// [`stated`] with a limit of five queued requests, at which the row is
/// vetoed.
fn limited() -> WorkerFacts {
    WorkerFacts {
        overload: Overload {
            waiting_requests: Some(5),
            ..Overload::default()
        },
        ..stated()
    }
}

/// A started store with one row under [`limited`], and the row's worker
/// object as the feed laid it.
struct Loads {
    store: Arc<MemoryStore>,
    registry: Arc<WorkerRegistry>,
    feed: Feed,
    row: WorkerId,
    worker: Arc<dyn Worker>,
}

impl Loads {
    fn new() -> Result<Self, Box<dyn Error>> {
        let store = start_store()?;
        let (registry, feed) = bed(&store);
        let wrote = write(&store, Precondition::Absent, &limited);
        let Ok(MembershipWriteOutcome::Registered(row)) = wrote else {
            return Err(format!("expected a registration, got {wrote:?}").into());
        };
        feed.settle();
        let worker = registry
            .get_by_url(URL)
            .ok_or("the registry holds no worker of the row")?;
        Ok(Self {
            store,
            registry,
            feed,
            row,
            worker,
        })
    }

    /// Write a record of rank zero with `queued` requests waiting on the
    /// row, received at `at`. The feed has no load sink of its own: the
    /// case writes the store as a collector's sink would.
    fn pushed(&self, queued: u32, at: Instant) {
        let record = LoadRow {
            queued,
            ..LoadRow::default()
        };
        LoadSink::record(&*self.store, self.row, 0, record, at, LoadSource::Stream);
    }
}

#[test]
fn a_settle_writes_the_stores_overload_verdict_on_a_replaced_object() {
    let loads = Loads::new().expect("the bed stands");
    loads.pushed(6, Instant::now());
    loads.feed.settle();
    assert!(loads.worker.is_overloaded());
    // The row's facts move, and the next settle replaces its object. The
    // registry releases the verdict the first carried, and that settle
    // writes it on the second.
    let moved = write(&loads.store, Precondition::Any, &|| WorkerFacts {
        max_running: Some(32),
        ..limited()
    });
    assert!(moved.is_ok(), "{moved:?}");
    loads.feed.settle();
    let second = loads.registry.get_by_url(URL).expect("the row's object");
    assert!(!Arc::ptr_eq(&loads.worker, &second));
    assert!(second.is_overloaded());
    assert_eq!(loads.registry.overloaded_worker_count(MODEL), 1);
}

/// A route that picks by no routing policy reads the flag the feed mirrors,
/// as it reads the monitor's under the classic policies. Where every
/// candidate reads overloaded the selector still hands out the least loaded
/// one, and under `--worker-overload-shed` it refuses instead.
#[tokio::test]
async fn a_classic_selection_reads_the_stores_overload_verdict() {
    let loads = Loads::new().expect("the bed stands");
    let selector = WorkerSelector::new(&loads.registry);
    let request = SelectWorkerRequest {
        model_id: MODEL,
        ..Default::default()
    };
    loads.pushed(6, Instant::now());
    loads.feed.settle();
    assert!(loads.worker.is_overloaded());

    let picked = selector.select_worker(&request).await;
    assert!(picked.is_ok_and(|picked| Arc::ptr_eq(&picked, &loads.worker)));

    loads.registry.set_overload_shed(true);
    let refused = selector.select_worker(&request).await;
    assert_eq!(
        refused.err().map(|shed| shed.status()),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
}
