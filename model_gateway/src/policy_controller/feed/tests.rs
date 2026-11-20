//! Cases of the gateway's membership feed, its load sink and its
//! publication loop.

use std::{
    error::Error,
    future::Future,
    pin::pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

use openai_protocol::model_card::ModelCard;
use parking_lot::Mutex;
use smg_policy_controller::{
    store::{
        Ask as ViewAsk, GatewayId, HealthPolicy, LoadSink as StoreLoadSink, Maker, MemoryStore,
        Overload, ViewBuffers, WorkerFacts, WorkerStatus as RowStatus, WorkersStore,
        PUBLISH_WINDOW,
    },
    view::{Role, WireModes, WorkerId},
    worker_host::{
        Expect, Membership, Objects, Over, Refused, Remover, Standing, StoreDoor, Wrote,
    },
};
use smg_worker::{
    connection::engine::EngineConnector,
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    load::port::{LoadSink, PollMode, PolledMember},
    registry::event::WorkerEvent,
    runtime::StallReason,
    worker::{SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus},
};
use tokio::{task::yield_now, time::sleep_until};

use super::{feed, Feed, FeedDoor};
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
    let policy_engine = PolicyEngine::new(store, Arc::clone(&feed.objects));
    let row = register(&*feed.door).expect("the worker registers");
    let placed = policy_engine
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the next decision picks the worker");
    assert_eq!(placed.id(), row);
}

#[tokio::test]
async fn a_failed_probe_is_written_on_the_row_and_the_object_mirrors_it() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let policy_engine = PolicyEngine::new(Arc::clone(&store), Arc::clone(&feed.objects));
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
    let placed = policy_engine
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
        policy_engine.ask(&ask(b"down"), &[], Instant::now()).await,
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
    let placed = policy_engine
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

/// Whether the next decision reads the pool's worker as stalled.
fn stalled(store: &MemoryStore) -> bool {
    let ask = ViewAsk {
        gateway: GatewayId::LOCAL,
        now: Instant::now(),
        pinned: None,
        affinity: false,
        signal_cells: false,
    };
    store.deciding(|world| {
        let mut view = ViewBuffers::default();
        let built = world.build(MODEL, Role::Single, &ask, &mut view).is_some();
        built && view.candidates.iter().any(|row| row.health.stalled)
    })
}

/// The health loop tells the feed's sink each move of a worker's liveness
/// veto. The adapter writes it as the row's stalled bit, and the next
/// decision reads it there.
#[test]
fn a_move_of_a_rows_liveness_veto_is_written_on_the_row() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let row = register(&*feed.door).expect("the worker registers");
    let object = feed.objects.object(row).expect("the row has an object");
    let worker = registry.get(&object).expect("the worker stands");
    let sink = feed.health_sink();
    assert!(!stalled(&store));

    assert!(worker.set_stall(Some(StallReason::Unreachable)));
    sink.liveness_moved(&object, worker.revision());
    assert!(stalled(&store));
    assert!(worker.set_stall(None));
    sink.liveness_moved(&object, worker.revision());
    assert!(!stalled(&store));
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

/// A replacement the door hands the holder moves both pool groups in one
/// step. The only worker of a model, moved to another model, leaves the
/// old model's gauge at zero.
#[test]
fn a_replacement_at_the_door_moves_the_gauge_of_both_groups() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let (_, feed) = bed(&Arc::new(MemoryStore::default()));
        for (model, left) in [("m1", None), ("m2", Some("m1"))] {
            let new: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(URL)
                    .model(ModelCard::new(model))
                    .build(),
            );
            feed.door
                .found_built(&Maker::Control, Expect::Anything, &new, None)
                .expect("the door takes the registration");
            assert_eq!(pool_size(&handle, model).as_deref(), Some("1"), "{model}");
            if let Some(left) = left {
                assert_eq!(pool_size(&handle, left).as_deref(), Some("0"), "{left}");
            }
        }
    });
}

/// The store shows a replacement inside its own write, and with probes
/// off the door readies it there, ahead of the listener. A door with no
/// listener leaves that state standing: the registry holds the old object.
#[tokio::test]
async fn a_replacement_takes_no_request_until_its_object_is_laid() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let policy_engine = PolicyEngine::new(Arc::clone(&store), Arc::clone(&feed.objects));
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

/// A built worker handed to the feed's door is the row's object when its
/// write returns. So is the next one, whose equal facts move no row. A
/// guarded write passes only over the object that stands.
#[test]
fn a_built_worker_is_the_rows_object_when_its_write_returns() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let built = || -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(URL).model(ModelCard::new(MODEL));
        Arc::new(worker.api_key("key").build())
    };
    let held = |built: &Arc<dyn Worker>| {
        let held = registry.get_by_url(URL);
        held.is_some_and(|held| Arc::ptr_eq(&held, built))
    };

    let first = built();
    let wrote = feed
        .door
        .found_built(&Maker::Control, Expect::Anything, &first, None);
    assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
    assert!(held(&first));

    let second = built();
    let wrote = feed
        .door
        .found_built(&Maker::Control, Expect::Anything, &second, None);
    assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
    assert!(held(&second));
    assert_eq!(second.api_key().map(String::as_str), Some("key"));

    // A guarded write names the object it replaces. The first is gone, so
    // a write over it is refused as moved though the row never moved.
    let third = built();
    let guarded = |over| {
        feed.door.found_built(
            &Maker::Control,
            Expect::Anything,
            &third,
            Some(Over::Updated(over)),
        )
    };
    assert_eq!(guarded(&first), Err(Refused::Moved));
    assert!(held(&second));
    assert!(matches!(guarded(&second), Ok(Wrote::Unmoved(_))));
    assert!(held(&third));
    assert_eq!(feed.objects.unhanded(), 0);
}

/// A worker the health loop's refresh rebuilt is the row's object when
/// its write returns, whether its labels moved the row or not. Equal labels
/// move no row, and the object is laid all the same. A refresh rebuilt
/// from an object that stands no more is refused as moved.
#[test]
fn a_refreshed_worker_is_the_rows_object_when_its_write_returns() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let built = |tier: &str| -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(URL).model(ModelCard::new(MODEL));
        Arc::new(worker.api_key("key").label("tier", tier).build())
    };
    let held = |built: &Arc<dyn Worker>| {
        let held = registry.get_by_url(URL);
        held.is_some_and(|held| Arc::ptr_eq(&held, built))
    };
    let seen = || {
        let row = store.by_key(URL).expect("the row");
        Standing {
            row: row.id,
            revision: row.revision,
        }
    };

    let first = built("gold");
    let wrote = feed
        .door
        .found_built(&Maker::Control, Expect::Anything, &first, None);
    assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");

    let equal = built("gold");
    let wrote = feed.door.refreshed(seen(), &equal, &first);
    assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
    assert!(held(&equal));

    let late = feed.door.refreshed(seen(), &built("silver"), &first);
    assert_eq!(late, Err(Refused::Moved));
    assert!(held(&equal));

    let moved = built("silver");
    let wrote = feed.door.refreshed(seen(), &moved, &equal);
    assert!(matches!(wrote, Ok(Wrote::InPlace(_))), "{wrote:?}");
    assert!(held(&moved));
}

/// A worker's ranks, each built with its own key and, where one is
/// named, a tier's label.
fn ranks(width: usize, gold: Option<usize>) -> Vec<Arc<dyn Worker>> {
    (0..width)
        .map(|rank| -> Arc<dyn Worker> {
            let worker = BasicWorkerBuilder::new(URL)
                .model(ModelCard::new(MODEL))
                .dp_config(rank, width)
                .api_key(format!("key-{rank}"));
            let worker = if gold == Some(rank) {
                worker.label("tier", "gold")
            } else {
                worker
            };
            Arc::new(worker.build())
        })
        .collect()
}

/// Whether each of `built` is the object the registry holds under its URL.
fn stand(registry: &WorkerRegistry, built: &[Arc<dyn Worker>]) -> Vec<bool> {
    built
        .iter()
        .map(|built| {
            let held = registry.get_by_url(built.url());
            held.is_some_and(|held| Arc::ptr_eq(&held, built))
        })
        .collect()
}

/// A worker of three ranks registered in one turn of the feed's door
/// leaves the three objects its registration built, each with its key.
#[test]
fn three_ranks_registered_in_one_turn_leave_three_handed_objects() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let built = ranks(3, None);
    let wrote = feed
        .door
        .found_rows(&Maker::Control, Expect::Nothing, &built, None);
    assert!(
        matches!(
            wrote.as_deref(),
            Ok([
                Wrote::Registered(_),
                Wrote::Registered(_),
                Wrote::Registered(_)
            ])
        ),
        "{wrote:?}"
    );
    assert_eq!(stand(&registry, &built), [true; 3]);
    assert_eq!(built[2].api_key().map(String::as_str), Some("key-2"));
    assert_eq!(feed.objects.unhanded(), 0);

    // One rank alone is not the engine's width: the write is refused
    // whole, and the object on offer for it is taken back.
    let alone = ranks(3, None);
    let refused = feed
        .door
        .found_rows(&Maker::Control, Expect::Anything, &alone[1..2], None);
    assert!(matches!(refused, Err(Refused::Past(_))), "{refused:?}");
    assert_eq!(stand(&registry, &built), [true; 3]);
    assert_eq!(stand(&registry, &alone), [false; 3]);
    assert_eq!(feed.objects.unhanded(), 0);
}

/// The same three ranks registered again with one rank's label changed
/// lay each row with its own object, the unmoved rows too.
#[test]
fn ranks_registered_again_lay_each_row_with_its_own_object() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let first = ranks(3, None);
    feed.door
        .found_rows(&Maker::Control, Expect::Nothing, &first, None)
        .expect("three ranks registered");
    let again = ranks(3, Some(1));
    let wrote = feed
        .door
        .found_rows(&Maker::Control, Expect::Anything, &again, None);
    assert!(
        matches!(
            wrote.as_deref(),
            Ok([Wrote::Unmoved(_), Wrote::InPlace(_), Wrote::Unmoved(_)])
        ),
        "{wrote:?}"
    );
    assert_eq!(stand(&registry, &again), [true; 3]);
    assert_eq!(stand(&registry, &first), [false; 3]);
    assert_eq!(feed.objects.unhanded(), 0);
}

/// A width that shrinks to two ends the third rank's object, and the two
/// ranks that stay are laid with the objects of the narrower registration.
#[test]
fn a_width_that_shrinks_ends_the_object_of_the_rank_that_left() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let wide = ranks(3, None);
    feed.door
        .found_rows(&Maker::Control, Expect::Nothing, &wide, None)
        .expect("three ranks registered");
    let narrow = ranks(2, None);
    let wrote = feed
        .door
        .found_rows(&Maker::Control, Expect::Anything, &narrow, None);
    assert!(
        matches!(wrote.as_deref(), Ok([_, _, Wrote::Removed(_)])),
        "{wrote:?}"
    );
    assert_eq!(stand(&registry, &narrow), [true; 2]);
    assert!(registry.get_by_url(wide[2].url()).is_none());
    assert_eq!(store.snapshot().1.len(), 2);
    assert_eq!(feed.objects.unhanded(), 0);
}

/// A worker that serves several engines over one connection states their
/// count and no rank. It is one row under its own URL, and the object its
/// registration built stands for it.
#[test]
fn a_worker_that_states_a_width_and_no_rank_is_one_row_with_its_object() {
    let store = Arc::new(MemoryStore::default());
    let (registry, feed) = bed(&store);
    let built: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new(URL)
            .model(ModelCard::new(MODEL))
            .zmq_engine_group(4)
            .build(),
    )];
    assert_eq!((built[0].dp_rank(), built[0].dp_size()), (None, Some(4)));
    let wrote = feed
        .door
        .found_rows(&Maker::Control, Expect::Nothing, &built, None);
    assert!(
        matches!(wrote.as_deref(), Ok([Wrote::Registered(_)])),
        "{wrote:?}"
    );
    assert_eq!(stand(&registry, &built), [true]);
    assert_eq!(store.snapshot().1.len(), 1);
    assert_eq!(feed.objects.unhanded(), 0);
}

#[tokio::test]
async fn a_drained_row_takes_no_decision_and_a_gone_row_leaves_the_registry() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let policy_engine = PolicyEngine::new(store, Arc::clone(&feed.objects));
    let row = register(&*feed.door).expect("the worker registers");
    let before = policy_engine
        .place(&ask(b"first"), Instant::now())
        .await
        .expect("the worker is picked before draining");
    assert_eq!(before.id(), row);
    drop(before);

    let object = feed.objects.object(row).expect("the row has an object");
    let seen = feed.objects.standing(&object).expect("the object's row");
    let by = Remover::Maker(&Maker::Control);
    assert_eq!(feed.door.drained(by, URL, seen), Ok(Wrote::Draining(row)));
    let drained = policy_engine
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
            objects: Arc::new(Objects::new(Arc::new(WorkerRegistry::new()))),
            store: Arc::clone(&store),
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

/// The interval in which the cases' collector hears every ready rank.
const PERIOD: Duration = Duration::from_secs(1);

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

/// A report of rank zero with `queued` requests waiting.
fn report(queued: i32) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            num_waiting_reqs: queued,
            ..SchedulerLoadSnapshot::default()
        }],
        ..WorkerLoadResponse::default()
    }
}

/// A started store with one row under [`limited`], the row's worker
/// object, and the feed's load sink.
struct Loads {
    store: Arc<MemoryStore>,
    registry: Arc<WorkerRegistry>,
    feed: Feed,
    row: WorkerId,
    worker: Arc<dyn Worker>,
    sink: Arc<dyn LoadSink>,
}

impl Loads {
    fn new() -> Result<Self, Box<dyn Error>> {
        let store = start_store()?;
        let (registry, feed) = bed(&store);
        let row = register_with(&*feed.door, limited)?;
        let worker = registry
            .get_by_url(URL)
            .ok_or("the registry holds no worker of the row")?;
        Ok(Self {
            sink: feed.load_sink(PERIOD),
            store,
            registry,
            feed,
            row,
            worker,
        })
    }

    /// Say a pushed record of `queued` waiting requests, received at `at`.
    fn pushed(&self, queued: i32, at: Instant) {
        self.sink.record(&self.worker, 0, &report(queued), at, at);
    }

    /// The row's overload bit in the view a decision is handed. `None`
    /// where the view lists no such row.
    fn vetoed(&self) -> Option<bool> {
        let mut view = ViewBuffers::default();
        let ask = ViewAsk {
            gateway: GatewayId::LOCAL,
            now: Instant::now(),
            pinned: None,
            affinity: false,
            signal_cells: false,
        };
        self.store
            .build_view(MODEL, Role::Single, &ask, &mut view)?;
        let row = view.candidates.iter().find(|row| row.id == self.row)?;
        Some(row.health.overloaded)
    }
}

#[test]
fn a_pushed_record_is_written_on_the_row_and_a_decision_reads_it() {
    let loads = Loads::new().expect("the bed stands");
    assert_eq!(loads.vetoed(), Some(false));
    loads.pushed(6, Instant::now());
    let sample = StoreLoadSink::load(&*loads.store, loads.row, 0).expect("the row holds a sample");
    assert_eq!(sample.row.queued, 6);
    // No window stood open, so the store showed the record by itself.
    assert_eq!(loads.vetoed(), Some(true));
}

#[test]
fn a_worker_with_no_row_writes_no_load() {
    let loads = Loads::new().expect("the bed stands");
    let stranger: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://stranger:8000")
            .model(ModelCard::new(MODEL))
            .status(WorkerStatus::Ready)
            .build(),
    );
    loads
        .registry
        .register(Arc::clone(&stranger))
        .expect("the worker registers");
    let shown = loads.store.world();
    let now = Instant::now();
    let answer = report(6);
    loads.sink.record(&stranger, 0, &answer, now, now);
    let polled = [PolledMember {
        worker: &stranger,
        answer: Some(&answer),
        on_file: None,
    }];
    assert!(loads.sink.report(&polled, now).newer.is_empty());
    // A worker with no row is not asked either.
    assert_eq!(
        loads.sink.poll_mode(&stranger, now, PERIOD),
        PollMode::Suppressed
    );
    assert_eq!(StoreLoadSink::ranks(&*loads.store, loads.row), 0);
    assert_eq!(loads.store.publish_due(), None);
    assert!(Arc::ptr_eq(&shown, &loads.store.world()));
}

#[tokio::test(start_paused = true)]
async fn the_loop_shows_a_waiting_write_at_its_due_instant() {
    let loads = Loads::new().expect("the bed stands");
    let _publications = loads.feed.publications();
    let now = Instant::now();
    // The first record opens the store's window and is shown by itself.
    loads.pushed(6, now);
    assert_eq!(loads.vetoed(), Some(true));
    // The loop reads no instant and waits for a wake.
    yield_now().await;
    // The second record lands inside the window, so it waits.
    loads.pushed(0, now + PUBLISH_WINDOW / 4);
    assert_eq!(loads.vetoed(), Some(true));
    // No later write comes: the loop shows it when the window ends.
    sleep_until((now + 2 * PUBLISH_WINDOW).into()).await;
    assert_eq!(loads.vetoed(), Some(false));
}

#[tokio::test(start_paused = true)]
async fn the_loop_writes_the_stores_overload_verdict_on_the_workers_object() {
    let loads = Loads::new().expect("the bed stands");
    let _publications = loads.feed.publications();
    let over = || loads.registry.overloaded_worker_count(MODEL);
    let now = Instant::now();
    // The store shows the first record by itself. The object's flag does
    // not move at the record: the loop writes it at its wake.
    loads.pushed(6, now);
    assert_eq!(loads.vetoed(), Some(true));
    assert!(!loads.worker.is_overloaded());
    yield_now().await;
    assert!(loads.worker.is_overloaded());
    assert_eq!(over(), 1);
    // The queue drains inside the window. The verdict waits for the
    // window's end, and the flag with it.
    loads.pushed(0, now + PUBLISH_WINDOW / 4);
    yield_now().await;
    assert!(loads.worker.is_overloaded());
    // The loop's publication shows the verdict cleared and clears the flag.
    sleep_until((now + 2 * PUBLISH_WINDOW).into()).await;
    assert_eq!(loads.vetoed(), Some(false));
    assert!(!loads.worker.is_overloaded());
    assert_eq!(over(), 0);
}

#[test]
fn a_settle_writes_the_stores_overload_verdict_on_a_replaced_object() {
    let loads = Loads::new().expect("the bed stands");
    loads.pushed(6, Instant::now());
    loads.feed.settle();
    assert!(loads.worker.is_overloaded());
    // The row's facts move at the door, and the holder replaces its
    // object. The registry releases the verdict the first carried, and
    // the write's own settle writes it on the second.
    let moved = || WorkerFacts {
        max_running: Some(32),
        ..limited()
    };
    let door = &loads.feed.door;
    door.found(&Maker::Control, Expect::Anything, URL, &|_| moved())
        .expect("the row moves in place");
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

#[tokio::test(start_paused = true)]
async fn a_forgotten_workers_eviction_is_shown_by_the_loop() {
    let loads = Loads::new().expect("the bed stands");
    let _publications = loads.feed.publications();
    let now = Instant::now();
    loads.pushed(6, now);
    assert_eq!(loads.vetoed(), Some(true));
    yield_now().await;
    // The collector forgets a worker that left `Ready`. The store judges
    // its engine again and names no instant, so the verdict waits.
    loads.sink.forget(&loads.worker);
    assert_eq!(loads.vetoed(), Some(true));
    sleep_until((now + 2 * PUBLISH_WINDOW).into()).await;
    assert_eq!(loads.vetoed(), Some(false));
}

#[tokio::test(start_paused = true)]
async fn a_resets_eviction_is_shown_by_the_loop() {
    let loads = Loads::new().expect("the bed stands");
    let _publications = loads.feed.publications();
    let now = Instant::now();
    loads.pushed(6, now);
    yield_now().await;
    // The row left `Ready`. The collector lost the event, and resets.
    let row = WorkersStore::get(&*loads.store, loads.row).expect("the row");
    loads
        .store
        .set_status(loads.row, RowStatus::NotReady, row.revision)
        .expect("the row stands");
    loads.sink.reset();
    assert_eq!(StoreLoadSink::ranks(&*loads.store, loads.row), 0);
    assert!(loads.store.publish_due().is_some());
    sleep_until((now + 2 * PUBLISH_WINDOW).into()).await;
    assert_eq!(loads.store.publish_due(), None);
}

/// What the registry's listener reads on the thread that lays a row's
/// object: whether the feed's turn is held, and the members a decision
/// sees.
struct AtObject {
    feed: Feed,
    read: OnceLock<(bool, Option<usize>)>,
}

impl Wake for AtObject {
    fn wake(self: Arc<Self>) {
        self.read.get_or_init(|| {
            (
                self.feed.turn.try_lock().is_none(),
                self.feed.store.world().members(MODEL, Role::Single),
            )
        });
    }
}

#[tokio::test(start_paused = true)]
async fn the_loop_settles_a_row_written_past_the_door_under_the_feeds_turn() {
    let store = start_store().expect("the store starts");
    let (registry, feed) = bed(&store);
    let _publications = feed.publications();
    // The loop reads no instant and waits for a wake.
    yield_now().await;
    let at_object = Arc::new(AtObject {
        feed: feed.clone(),
        read: OnceLock::new(),
    });
    let waker = Waker::from(Arc::clone(&at_object));
    let mut events = registry.subscribe_events();
    let mut event = pin!(events.recv());
    assert!(event
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());

    // A write past the door leaves a build owed, and the store wakes the
    // loop. No load is written, and nobody settles.
    store.upsert(stated());
    assert!(registry.get_by_url(URL).is_none());
    sleep_until((Instant::now() + PUBLISH_WINDOW).into()).await;
    // The loop laid the object under the turn, then showed the row.
    assert_eq!(at_object.read.get(), Some(&(true, None)));
    assert!(registry.get_by_url(URL).is_some());
    assert_eq!(store.world().members(MODEL, Role::Single), Some(1));
}

#[tokio::test(start_paused = true)]
async fn the_loop_ends_when_its_answer_is_dropped() {
    let loads = Loads::new().expect("the bed stands");
    let publications = loads.feed.publications();
    yield_now().await;
    drop(publications);
    let now = Instant::now();
    loads.pushed(6, now);
    loads.pushed(0, now + PUBLISH_WINDOW / 4);
    sleep_until((now + 2 * PUBLISH_WINDOW).into()).await;
    // The second record still waits: nobody shows it.
    assert_eq!(loads.vetoed(), Some(true));
}
