//! A worker's zone, from the registration that states it to a decision of
//! the policy engine in this process.

use std::{error::Error, sync::Arc, time::Instant};

use openai_protocol::{
    model_card::ModelCard,
    worker::{WorkerSpec, WorkerStatus},
};
use serde_json::json;
use smg_policy_controller::{
    document::PolicySpec,
    engine::{Candidate, Constraints, Filter, RequestFacts},
    stages::{Lifecycle, StageRegistry},
};
use smg_worker::{identity::basic::Worker, test_support::no_health_check};

use super::{
    engine::{mark, Ask, PolicyEngine},
    start_in_process,
};
use crate::{
    config::RouterConfig,
    routers::{
        common::snapshot::{RoutingPool, WorkerRegistry},
        grpc::backend_client::BasicWorkerBuilder,
    },
};

type Case = Result<(), Box<dyn Error>>;

const MODEL: &str = "model-a";

/// A filter of a host's own build: it keeps the workers of the first zone
/// the document lists.
#[derive(Debug)]
struct InFirstZone;

impl Lifecycle for InFirstZone {}

impl Filter for InFirstZone {
    fn name(&self) -> &'static str {
        "in-first-zone"
    }

    fn keep(&self, _: &RequestFacts<'_>, worker: &Candidate, _: &Constraints<'_>) -> bool {
        worker.zone == 1
    }
}

/// A ready worker of [`MODEL`], built from `registration`: the JSON an
/// operator posts to register a worker.
fn registered(
    registry: &WorkerRegistry,
    registration: serde_json::Value,
) -> Result<Arc<dyn Worker>, Box<dyn Error>> {
    let spec: WorkerSpec = serde_json::from_value(registration)?;
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .model(ModelCard::new(MODEL))
            .health_config(no_health_check())
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    registry
        .register(Arc::clone(&worker))
        .ok_or("the registry took the worker")?;
    Ok(worker)
}

/// With the setting on, a document that lists zones is decided by the engine
/// in this process over the zones the workers registered with. Six workers
/// register, two in each of two zones and two in none, and a filter of the
/// host's keeps the first listed zone: every request lands on that zone's
/// two workers, in turn, and on no other.
///
/// Before the worker's spec held a zone the document started as well, every
/// worker read as in no zone, and the filter left nobody: each request was
/// shed.
#[tokio::test]
async fn a_document_that_reads_a_zone_is_decided_over_the_zones_the_workers_registered() -> Case {
    let mut document = PolicySpec::preset("round_robin")?;
    document.view.zones = vec!["east".to_owned(), "west".to_owned()];
    let profile = document
        .profiles
        .get_mut("round_robin")
        .ok_or("the round robin profile")?;
    profile.filter.push("in-first-zone".into());
    let config = RouterConfig {
        policy_controller: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    };
    let mut stages = StageRegistry::builtin();
    stages.filter("in-first-zone", || Box::new(InFirstZone))?;
    let started = start_in_process(&config, Some(stages))?.ok_or("the setting starts an engine")?;
    assert!(
        started.decides_whole(),
        "the routers ask the engine for every request of the document"
    );

    let registry = Arc::new(WorkerRegistry::new());
    let zones = [
        Some("east"),
        Some("west"),
        None,
        Some("east"),
        Some("west"),
        None,
    ];
    let mut workers = Vec::new();
    for (at, zone) in zones.into_iter().enumerate() {
        let mut registration = json!({ "url": format!("http://worker-{at}:8000") });
        if let Some(zone) = zone {
            registration["zone"] = json!(zone);
        }
        workers.push(registered(&registry, registration)?);
    }

    let policy_engine = PolicyEngine::own(Arc::clone(&started.engine), registry);
    let mut placed_on = vec![0_u32; workers.len()];
    for _ in 0..8 {
        let id = mark();
        let ask = Ask {
            pool: Some(RoutingPool::HttpRegular),
            ..Ask::of(&id, MODEL)
        };
        let now = Instant::now();
        let mut placed = policy_engine
            .place(&ask, now)
            .await
            .map_err(|unplaced| format!("a request was not placed: {unplaced:?}"))?;
        let at = workers
            .iter()
            .position(|worker| Arc::ptr_eq(worker, placed.worker()))
            .ok_or("a worker the registry lists")?;
        placed_on[at] += 1;
        placed.dispatched(now);
    }
    assert_eq!(placed_on, [4, 0, 0, 4, 0, 0]);
    assert_eq!(policy_engine.faults(), 0);
    Ok(())
}
