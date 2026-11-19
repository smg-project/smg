//! Cases of the host over rows written by hand through the store's own
//! door, beside workers registered by hand.

use std::{
    collections::BTreeSet,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::{
    decision::ShedReason,
    lease::{ReleaseReason, TransportClass},
    store::{LeasesStore, MemoryStore, WorkerFacts, WorkerStatus as RowStatus, WorkersStore},
    view::Role,
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};

use super::{Ask, End, Held, Host, Unplaced, GATEWAY};
use crate::{
    config::RouterConfig,
    policy_controller::start,
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
};

const MODEL: &str = "model-a";

/// The store a start makes under the controller with one ready row a URL,
/// a registry with one worker a URL, and the host over the two.
fn bed(urls: &[&str]) -> (Host, Arc<MemoryStore>, Vec<Arc<dyn Worker>>) {
    let config = RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the default routing")
        .expect("the controller was picked");
    let registry = Arc::new(WorkerRegistry::new());
    let mut workers = Vec::new();
    for url in urls {
        let facts = WorkerFacts {
            key: (*url).to_string(),
            engine: (*url).to_string(),
            incarnation: 1,
            role: Role::Single,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            max_running: Some(64),
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        store.publish_now();
        let row = WorkersStore::get(&*store, id).expect("the row");
        if row.status != RowStatus::Ready {
            store
                .set_status(id, RowStatus::Ready, row.revision)
                .expect("the row stands");
        }
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(*url)
                .model(ModelCard::new(MODEL))
                .status(WorkerStatus::Ready)
                .build(),
        );
        registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        workers.push(worker);
    }
    store.publish_now();
    (Host::new(Arc::clone(&store), registry), store, workers)
}

/// A first attempt's ask of the request `id`.
fn first(id: &[u8]) -> Ask<'_> {
    Ask {
        id,
        model: MODEL,
        streaming: false,
        tried: &[],
    }
}

/// A plain end with status 200.
fn ok() -> End {
    End {
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// The gateway's live leases, read until they number `expected` or the
/// patience ends: a report is applied by whoever holds the lease's file.
fn live(store: &MemoryStore, expected: usize) -> usize {
    let patience = Instant::now() + Duration::from_secs(5);
    loop {
        let live = LeasesStore::live(store, GATEWAY);
        if live == expected || Instant::now() > patience {
            return live;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[tokio::test]
async fn a_pick_lands_on_the_gateways_worker_of_its_row() {
    let (host, store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let placed = host
        .place(&first(b"r1"), Instant::now())
        .await
        .expect("a worker");
    let row = WorkersStore::get(&*store, placed.id()).expect("the target's row");
    assert_eq!(placed.worker().url(), row.facts.key);
}

#[tokio::test]
async fn a_drawn_pick_is_not_the_same_worker_for_every_request() {
    let (host, _store, _workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    let mut picked = BTreeSet::new();
    for turn in 0..64u8 {
        let placed = host
            .place(&first(&[turn]), Instant::now())
            .await
            .expect("a worker");
        picked.insert(placed.worker().url().to_owned());
    }
    // Each ask draws its own seed. Under one seed the default profile's
    // draw lands every request on the same worker of an unchanged pool.
    assert_eq!(picked.len(), 2, "64 requests landed on {picked:?}");
}

#[tokio::test]
async fn a_worker_the_gateway_cannot_send_to_is_passed_over() {
    let (host, _store, workers) = bed(&["http://w0:8000", "http://w1:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    for turn in 0..8u8 {
        let placed = host
            .place(&first(&[turn]), Instant::now())
            .await
            .expect("the other worker");
        assert_eq!(placed.worker().url(), "http://w1:8000");
    }
}

#[tokio::test]
async fn a_store_with_no_row_sheds_by_the_controllers_reason() {
    let (host, _store, _workers) = bed(&[]);
    let unplaced = host
        .place(&first(b"r1"), Instant::now())
        .await
        .expect_err("no row, no worker");
    assert!(
        matches!(
            unplaced,
            Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            }
        ),
        "{unplaced:?}"
    );
}

#[tokio::test]
async fn an_attempt_the_gateway_can_send_nowhere_ends_after_its_asks() {
    let (host, _store, workers) = bed(&["http://w0:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    let unplaced = host.place(&first(b"r1"), Instant::now()).await.err();
    assert_eq!(unplaced, Some(Unplaced::NoTargetHolds));
}

#[tokio::test]
async fn the_commit_gives_the_lease_up_and_names_its_misses() {
    let (host, store, workers) = bed(&["http://w0:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);
    let now = Instant::now();
    let asked = host.ask(&first(b"r1"), &[], now).await.expect("a decision");
    assert_eq!(live(&store, 1), 1);
    let mut missed = Vec::new();
    assert!(host.commit(asked, &mut missed, now).is_none());
    assert_eq!(missed.len(), 1, "the one target is a miss: {missed:?}");
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_lease_ends_once_by_its_completion_and_by_a_drop() {
    let (host, store, _workers) = bed(&["http://w0:8000"]);
    let now = Instant::now();
    let mut placed = host.place(&first(b"r1"), now).await.expect("a worker");
    assert_eq!(live(&store, 1), 1);
    placed.dispatched(now);
    placed.completed(ok(), now);
    assert_eq!(live(&store, 0), 0);
    // Said once: nothing more is owed, and a release says nothing.
    placed.released(ReleaseReason::ClientGone, now);
    drop(placed);
    // A placement dropped before any send gives its lease up.
    drop(host.place(&first(b"r2"), now).await.expect("a worker"));
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_held_request_dropped_after_its_send_ends_its_lease_and_its_load() {
    let (host, store, workers) = bed(&["http://w0:8000"]);
    let now = Instant::now();
    let placed = host.place(&first(b"r1"), now).await.expect("a worker");
    let guard = WorkerLoadGuard::with_key(Arc::clone(placed.worker()), None);
    let mut held = Held { placed, guard };
    held.placed.dispatched(now);
    assert_eq!((live(&store, 1), workers[0].load()), (1, 1));
    // A response dropped with no end said: the lease completes as cancelled
    // and the worker's load is given back.
    drop(held);
    assert_eq!((live(&store, 0), workers[0].load()), (0, 0));
}
