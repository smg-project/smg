//! The gateway's engine object over the engine of this process: an ask is
//! decided over the list the registry publishes for its route's pool, and a
//! target is the worker at its place in that list.

use std::{cell::Cell, sync::Arc, time::Instant};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        Phase, Plain, PolicyEngine as OwnEngine, RequestMaker, ShedReason, WireModes, WorkerId,
    },
};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    test_support::no_health_check,
};

use super::{mark, Ask, Made, PolicyEngine, Prompt, Unplaced};
use crate::routers::{
    common::snapshot::{RoutingPool, WorkerRegistry},
    grpc::backend_client::BasicWorkerBuilder,
};

const MODEL: &str = "model-a";

/// The object over the engine of the preset `name`, with `workers` ready
/// HTTP workers of [`MODEL`] in its registry.
fn own(name: &str, workers: usize) -> (PolicyEngine, Vec<Arc<dyn Worker>>) {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let registry = Arc::new(WorkerRegistry::new());
    let listed = (0..workers)
        .map(|at| {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{at}:8000"))
                    .model(ModelCard::new(MODEL))
                    .connection_mode(ConnectionMode::Http)
                    .health_config(no_health_check())
                    .build(),
            );
            worker.set_status(WorkerStatus::Ready);
            assert!(registry.register(Arc::clone(&worker)).is_some());
            worker
        })
        .collect();
    let engine = Arc::new(OwnEngine::new(spec));
    (PolicyEngine::own(engine, registry), listed)
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
