//! The gateway's engine object over the engine of this process: an ask is
//! decided over the list the registry publishes for its route's pool, and a
//! target is the worker at its place in that list.

use std::{cell::Cell, sync::Arc, time::Instant};

use axum::http::{HeaderMap, HeaderValue};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        worker_id, Phase, Plain, PolicyEngine as OwnEngine, RequestMaker, ShedReason, WireModes,
        WorkerId,
    },
    tracking::TransportClass,
};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    test_support::no_health_check,
};

use super::{mark, Ask, End, Made, PolicyEngine, Prompt, Unplaced, Workers};
use crate::routers::{
    common::snapshot::{RoutingPool, WorkerRegistry},
    grpc::backend_client::BasicWorkerBuilder,
};

const MODEL: &str = "model-a";

/// The object over the engine of the preset `name`, with `workers` ready
/// HTTP workers of [`MODEL`] in its registry.
fn own(name: &str, workers: usize) -> (PolicyEngine, Vec<Arc<dyn Worker>>) {
    own_dialed(name, &vec![ConnectionMode::Http; workers])
}

/// [`own`] with one ready worker a connection mode of `modes`, in their
/// order.
fn own_dialed(name: &str, modes: &[ConnectionMode]) -> (PolicyEngine, Vec<Arc<dyn Worker>>) {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let registry = Arc::new(WorkerRegistry::new());
    let listed = modes
        .iter()
        .enumerate()
        .map(|(at, &mode)| joined(&registry, at, mode))
        .collect();
    let engine = Arc::new(OwnEngine::new(spec));
    (PolicyEngine::own(engine, registry), listed)
}

/// A ready worker of [`MODEL`] dialed on `mode`, the `at`-th, registered in
/// `registry`.
fn joined(registry: &WorkerRegistry, at: usize, mode: ConnectionMode) -> Arc<dyn Worker> {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("http://worker-{at}:8000"))
            .model(ModelCard::new(MODEL))
            .connection_mode(mode)
            .health_config(no_health_check())
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    assert!(registry.register(Arc::clone(&worker)).is_some());
    worker
}

/// A regular attempt under round robin is placed by the engine of this
/// process over the registry's list for the ask's pool. The placement holds
/// the worker object the registry lists, each worker of the pool takes its
/// turn, and no store stands behind the object.
#[tokio::test]
async fn a_regular_attempt_is_placed_over_the_registrys_own_list() {
    let (policy_engine, workers) = own("round_robin", 3);
    assert!(policy_engine.door.store().is_none());
    let mut placed_on = vec![0_u32; workers.len()];
    for _ in 0..6 {
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let now = Instant::now();
        let mut placed = policy_engine.place(&ask, now).await.expect("a worker");
        let at = workers
            .iter()
            .position(|worker| Arc::ptr_eq(worker, placed.worker()));
        placed_on[at.expect("a worker the registry lists")] += 1;
        placed.dispatched(now);
    }
    assert_eq!(placed_on, [2, 2, 2]);
    assert_eq!(policy_engine.faults(), 0);
}

/// An ask is plain where it is a first attempt that names nothing tried
/// or missed and asks no decode again. Its plain word states its phase, its
/// wires and the size its request would state: the ask's own, and the
/// hashing's count where the ask states none. No request is made for it.
#[test]
fn a_first_ask_is_plain_and_states_the_size_its_request_would() {
    let (policy_engine, _workers) = own("round_robin", 1);
    let id = mark();
    let tokens: Vec<u32> = (0..40).collect();
    let hashes = policy_engine.hashes(MODEL, &Prompt::of(Some(&tokens), None));
    let ask = Ask {
        wires: WireModes::HTTP,
        hashes: Some(&hashes),
        ..Ask::of(&id, MODEL)
    };
    let plain = |ask: &Ask<'_>, missed: &[WorkerId]| {
        let made = Made {
            ask,
            missed,
            members: None,
            source: Cell::new(None),
        };
        made.plain()
    };
    assert_eq!(
        plain(&ask, &[]),
        Some(Plain {
            phase: Phase::Single,
            wires: WireModes::HTTP,
            prompt_tokens: 40,
        })
    );
    let stated = Ask {
        prompt_tokens: 7,
        ..ask
    };
    assert_eq!(
        plain(&stated, &[]).map(|plain| plain.prompt_tokens),
        Some(7)
    );
    let tried = [WorkerId(1)];
    let retry = Ask {
        tried: &tried,
        ..ask
    };
    assert_eq!(plain(&retry, &[]), None);
    assert_eq!(plain(&ask, &tried), None, "a target missed at this attempt");
    let numbered = Ask {
        attempt: Some(1),
        ..ask
    };
    assert_eq!(plain(&numbered, &[]), None);
    let again = Ask {
        decode_again: true,
        ..ask
    };
    assert_eq!(plain(&again, &[]), None);
}

/// An ask that states no pool is decided over no worker. A retry names the
/// worker its failed attempt was sent to, by the id its placement carried,
/// and is never placed there again.
#[tokio::test]
async fn an_ask_with_no_pool_has_no_worker_and_a_retry_passes_the_worker_it_tried() {
    let (policy_engine, _workers) = own("round_robin", 3);
    let id = mark();
    let unplaced = policy_engine
        .place(&Ask::of(&id, MODEL), Instant::now())
        .await
        .err();
    assert!(
        matches!(
            unplaced,
            Some(Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "none",
                ..
            })
        ),
        "{unplaced:?}"
    );
    let ask = Ask {
        pool: Some(RoutingPool::HttpRegular),
        ..Ask::of(&id, MODEL)
    };
    let first = policy_engine
        .place(&ask, Instant::now())
        .await
        .expect("a worker");
    let tried = [first.id()];
    let failed = Arc::clone(first.worker());
    drop(first);
    for _ in 0..6 {
        let retry = Ask {
            tried: &tried,
            ..ask
        };
        let placed = policy_engine
            .place(&retry, Instant::now())
            .await
            .expect("another worker");
        assert!(!Arc::ptr_eq(placed.worker(), &failed));
    }
}

/// Under round robin no stage takes a word of a request, so a first attempt
/// is answered a place. Its placement holds no lease and its reports say
/// nothing. It names its worker where a route asks: by the id a whole
/// decision names the same worker by, which a retry states as tried, and by
/// the rank the worker's registration states. A retry's whole decision is
/// leased nothing either.
#[tokio::test]
async fn a_place_holds_no_lease_and_names_its_worker_where_asked() {
    let (policy_engine, _workers) = own("round_robin", 3);
    let id = mark();
    let ask = Ask {
        pool: Some(RoutingPool::HttpRegular),
        prompt_tokens: 9,
        ..Ask::of(&id, MODEL)
    };
    let now = Instant::now();
    let mut placed = policy_engine.place(&ask, now).await.expect("a worker");
    assert!(placed.lease().is_none());
    assert_eq!(placed.id(), worker_id(placed.worker().as_ref()));
    assert_eq!(placed.rank(), 0);
    placed.dispatched(now);
    placed.completed(
        End {
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
        now,
    );
    let tried = [placed.id()];
    let retry = Ask {
        tried: &tried,
        ..ask
    };
    let again = policy_engine
        .place(&retry, now)
        .await
        .expect("another worker");
    assert!(again.lease().is_none(), "no stage takes a word of it");
    assert_eq!(again.id(), worker_id(again.worker().as_ref()));
    assert_ne!(again.id(), placed.id());
    assert_eq!(policy_engine.faults(), 0);
}

/// Under a document a stage of which takes a word of a request, as least
/// load takes the dispatch, a placement holds its decision's lease, which
/// states the request's size.
#[tokio::test]
async fn a_placement_is_leased_where_a_stage_takes_a_word() {
    let (policy_engine, _workers) = own("least_load", 3);
    let id = mark();
    let ask = Ask {
        pool: Some(RoutingPool::HttpRegular),
        prompt_tokens: 9,
        ..Ask::of(&id, MODEL)
    };
    let now = Instant::now();
    let mut placed = policy_engine.place(&ask, now).await.expect("a worker");
    assert_eq!(placed.lease_tokens(), Some(9));
    assert_eq!(placed.id(), worker_id(placed.worker().as_ref()));
    placed.dispatched(now);
}

/// A route's rule that refuses a worker picked narrows the attempt to the
/// workers it holds: the request is placed on one of them while one stands,
/// however few of the pool they are, and they take such requests in turn.
/// Of 64 requests over 16 workers each worker is picked four times; the two
/// the rule holds keep theirs and share the other 56 evenly. A rule that
/// holds no worker leaves the attempt none.
#[tokio::test]
async fn a_routes_rule_narrows_the_attempt_to_the_workers_it_holds() {
    let (policy_engine, workers) = own("round_robin", 16);
    let held = [Arc::clone(&workers[3]), Arc::clone(&workers[11])];
    let holds = |worker: &dyn Worker| held.iter().any(|held| held.url() == worker.url());
    let mut placed_on = [0_u32; 2];
    for _ in 0..64 {
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let placed = policy_engine
            .place_where(&ask, &holds, Instant::now())
            .await
            .expect("a worker the rule holds");
        let at = held
            .iter()
            .position(|held| Arc::ptr_eq(held, placed.worker()));
        placed_on[at.expect("a worker the rule holds")] += 1;
    }
    assert_eq!(placed_on, [32, 32]);
    assert_eq!(policy_engine.faults(), 0);
    // Under a document that leases its decisions too: the lease of the
    // decision whose targets the rule refused is given up within the
    // attempt.
    for name in ["round_robin", "least_load"] {
        let (policy_engine, _workers) = own(name, 3);
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let unplaced = policy_engine
            .place_where(&ask, |_| false, Instant::now())
            .await
            .err();
        assert!(
            matches!(unplaced, Some(Unplaced::NoTargetHolds)),
            "{name}: {unplaced:?}"
        );
        assert_eq!(policy_engine.faults(), 0, "{name}");
    }
}

/// An ask that states the workers it may take is decided over them alone,
/// whatever pool it names, and they take its requests in turn.
#[tokio::test]
async fn an_ask_that_states_its_workers_is_decided_over_them_alone() {
    let (policy_engine, workers) = own("round_robin", 8);
    let among: Arc<[Arc<dyn Worker>]> = [Arc::clone(&workers[1]), Arc::clone(&workers[6])].into();
    let mut placed_on = [0_u32; 2];
    for _ in 0..32 {
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            among: Some(&among),
            ..Ask::of(&id, MODEL)
        };
        let placed = policy_engine
            .place(&ask, Instant::now())
            .await
            .expect("a worker of the list");
        let at = among
            .iter()
            .position(|listed| Arc::ptr_eq(listed, placed.worker()));
        placed_on[at.expect("a worker of the list")] += 1;
    }
    assert_eq!(placed_on, [16, 16]);
}

/// An ask held to one wire, over a pool whose workers are dialed on two, is
/// decided over the pool's workers on that wire alone, and they take its
/// requests in turn. An ask that dials both wires is decided over the pool.
#[tokio::test]
async fn an_ask_held_to_one_wire_is_decided_over_the_workers_on_it() {
    use ConnectionMode::{Grpc, Zmq};
    let (policy_engine, workers) = own_dialed("round_robin", &[Grpc, Zmq, Grpc, Zmq]);
    let mut placed_on = [0_u32; 4];
    for wires in [WireModes::GRPC, WireModes::GRPC.union(WireModes::ZMQ)] {
        for _ in 0..8 {
            let id = mark();
            let ask = Ask {
                pool: Some(RoutingPool::GrpcPipelineRegular),
                wires,
                ..Ask::of(&id, MODEL)
            };
            let placed = policy_engine
                .place(&ask, Instant::now())
                .await
                .expect("a worker on the wire");
            let at = workers
                .iter()
                .position(|worker| Arc::ptr_eq(worker, placed.worker()));
            placed_on[at.expect("a worker the registry lists")] += 1;
        }
    }
    assert_eq!(placed_on, [6, 2, 6, 2]);
}

/// A pool none of whose workers is ready leaves the attempt no worker,
/// under a document that leases its decisions and under one that does not.
#[tokio::test]
async fn a_pool_with_no_ready_worker_leaves_the_attempt_none() {
    for name in ["round_robin", "least_load"] {
        let (policy_engine, workers) = own(name, 3);
        for worker in &workers {
            worker.set_status(WorkerStatus::NotReady);
        }
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let unplaced = policy_engine.place(&ask, Instant::now()).await.err();
        assert!(
            matches!(
                unplaced,
                Some(Unplaced::Shed { .. } | Unplaced::NoTargetHolds)
            ),
            "{name}: {unplaced:?}"
        );
        assert_eq!(policy_engine.faults(), 0, "{name}");
    }
}

/// The registry `policy_engine` decides over.
fn registry_of(policy_engine: &PolicyEngine) -> &WorkerRegistry {
    match &policy_engine.workers {
        Workers::Listed(registry) => registry,
        _ => panic!("the object of an engine of this process"),
    }
}

/// The places in `workers` of the workers that `asks` requests of one
/// routing key are placed on, each place once. Every request is placed.
async fn places_of_one_key(
    policy_engine: &PolicyEngine,
    workers: &[Arc<dyn Worker>],
    asks: usize,
) -> Vec<usize> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        HeaderValue::from_static("conversation"),
    );
    let mut places = Vec::new();
    for _ in 0..asks {
        let id = mark();
        let ask = Ask {
            headers: Some(&headers),
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let now = Instant::now();
        let placed = policy_engine.place(&ask, now).await;
        let mut placed = placed.expect("placed, not refused");
        let at = workers
            .iter()
            .position(|worker| Arc::ptr_eq(worker, placed.worker()))
            .expect("a worker the registry lists");
        placed.dispatched(now);
        if !places.contains(&at) {
            places.push(at);
        }
    }
    places
}

/// No request builds a list's hash ring and none waits for one: here
/// nothing fills a ring but the case's own call, so a request that waited
/// would never return. A keyed request ahead of its list's ring is placed,
/// not refused, as one that carries no key: one key is handed more workers
/// than one. Once the ring stands the key keeps one worker. A worker that
/// joins makes another list, whose ring is owed again: the key's requests
/// are placed meanwhile, and behind the fill the key keeps the worker it
/// had, or the one that joined.
#[tokio::test]
async fn a_keyed_request_ahead_of_its_lists_ring_is_placed_and_keeps_one_worker_behind_it() {
    let (policy_engine, mut workers) = own("consistent_hashing", 8);
    assert!(!policy_engine.rings_stand(MODEL));
    let ahead = places_of_one_key(&policy_engine, &workers, 24).await;
    assert!(ahead.len() > 1, "one key, one worker: a ring placed it");
    assert!(
        !policy_engine.rings_stand(MODEL),
        "a request built the ring"
    );

    assert!(policy_engine.fill_rings(MODEL));
    assert!(policy_engine.rings_stand(MODEL));
    let kept = places_of_one_key(&policy_engine, &workers, 24).await;
    assert_eq!(kept.len(), 1, "the key moved: {kept:?}");

    workers.push(joined(registry_of(&policy_engine), 8, ConnectionMode::Http));
    assert!(!policy_engine.rings_stand(MODEL), "another list");
    let ahead = places_of_one_key(&policy_engine, &workers, 24).await;
    assert!(ahead.len() > 1, "one key, one worker: a ring placed it");
    assert!(policy_engine.fill_rings(MODEL));
    let behind = places_of_one_key(&policy_engine, &workers, 24).await;
    assert!(
        behind == kept || behind == [8],
        "the key had {kept:?} and has {behind:?}"
    );
}
