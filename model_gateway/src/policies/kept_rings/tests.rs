//! The cases of the kept rings: what the ring of a generation holds, that a
//! kept ring moves a fill's cost and never its content, who fills a
//! generation's ring, and what the warm task keeps and drops.

use std::sync::Barrier;

use openai_protocol::{model_card::ModelCard, worker::WorkerType};
use tokio::runtime::Handle;

use super::*;
use crate::{
    policies::{ConsistentHashingPolicy, LoadBalancingPolicy, SelectWorkerInfo},
    worker::{test_support::no_health_check, BasicWorkerBuilder, ConnectionMode, Worker},
};

fn worker_serving(url: &str, model_ids: &[&str]) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .models(
                model_ids
                    .iter()
                    .map(|id| ModelCard::new(*id))
                    .collect::<Vec<_>>(),
            )
            .health_config(no_health_check())
            .build(),
    )
}

fn url(n: usize) -> String {
    format!("http://w{n}:8080")
}

/// A registry whose `model` is served by `members` workers.
fn registry_of(members: usize) -> WorkerRegistry {
    let registry = WorkerRegistry::new();
    for n in 0..members {
        registry
            .register(worker_serving(&url(n), &["model"]))
            .expect("a worker of the model");
    }
    registry
}

/// The ring a request reads for `model_id` holds `urls` and no other.
fn assert_ring_urls(rings: &KeptRings, registry: &WorkerRegistry, model_id: &str, urls: &[&str]) {
    let ring = rings.of(model_id, &registry.get_routing_snapshot(model_id));
    if urls.is_empty() {
        assert!(ring.is_none(), "{model_id}: nobody serves it, no ring");
        return;
    }
    let ring = ring.expect("a served model has a ring");
    assert_eq!(ring.worker_count(), urls.len(), "model {model_id}");
    for &url in urls {
        assert_eq!(
            ring.find_healthy_url("key", |candidate| candidate == url),
            Some(url),
            "the ring of {model_id} holds {url}"
        );
    }
}

/// Two rings place alike: as many workers and positions, and one worker for
/// each of a spread of keys.
fn assert_same_placements(left: &HashRing, right: &HashRing) {
    assert_eq!(left.worker_count(), right.worker_count());
    assert_eq!(left.len(), right.len());
    for key in 0..512 {
        let key = format!("key-{key}");
        assert_eq!(
            left.find_healthy_url(&key, |_| true),
            right.find_healthy_url(&key, |_| true),
            "{key}"
        );
    }
}

/// Runs two writers of `registry` at once.
fn written_at_once(
    registry: &WorkerRegistry,
    first: impl FnOnce(&WorkerRegistry) + Send,
    second: impl FnOnce(&WorkerRegistry) + Send,
) {
    let start = &Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            start.wait();
            first(registry);
        });
        scope.spawn(move || {
            start.wait();
            second(registry);
        });
    });
}

/// Yields to the runtime, the warm task's turn each time, until `holds`.
async fn until(what: &str, mut holds: impl FnMut() -> bool) {
    for _ in 0..100_000 {
        if holds() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("the warm task never got to: {what}");
}

#[test]
fn a_models_ring_holds_its_workers_and_the_model_less_ring_every_worker_once() {
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    registry
        .register(worker_serving("http://w1:8080", &["model-a", "model-b"]))
        .expect("the first worker");
    registry
        .register(worker_serving("http://w2:8080", &["model-b"]))
        .expect("the second worker");

    assert_ring_urls(&rings, &registry, "model-a", &["http://w1:8080"]);
    assert_ring_urls(
        &rings,
        &registry,
        "model-b",
        &["http://w1:8080", "http://w2:8080"],
    );
    // A worker of two models takes one share of the model-less ring.
    assert_ring_urls(
        &rings,
        &registry,
        UNKNOWN_MODEL_ID,
        &["http://w1:8080", "http://w2:8080"],
    );
}

#[test]
fn the_model_less_ring_places_as_the_ring_of_the_only_model() {
    let registry = registry_of(2);
    let rings = KeptRings::default();
    let model_less = rings
        .of(
            UNKNOWN_MODEL_ID,
            &registry.get_routing_snapshot(UNKNOWN_MODEL_ID),
        )
        .expect("a request that names no model has a ring");
    let of_the_model = rings
        .of("model", &registry.get_routing_snapshot("model"))
        .expect("the model's ring");
    assert_same_placements(&model_less, &of_the_model);
}

#[test]
fn a_ring_follows_removals_and_a_model_nobody_serves_has_none() {
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    registry
        .register(worker_serving("http://w1:8080", &["model-a"]))
        .expect("the first worker");
    registry
        .register(worker_serving("http://w2:8080", &["model-b"]))
        .expect("the second worker");

    registry.remove_by_url("http://w2:8080");
    assert_ring_urls(&rings, &registry, "model-b", &[]);
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &["http://w1:8080"]);

    registry.remove_by_url("http://w1:8080");
    assert_ring_urls(&rings, &registry, "model-a", &[]);
    // An empty registry has no ring to place on.
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &[]);
}

#[test]
fn a_replacement_moves_a_worker_between_the_rings_of_its_models() {
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    let first = registry
        .register(worker_serving("http://first:8080", &["old", "kept"]))
        .expect("the first worker");
    registry
        .register(worker_serving("http://second:8080", &["old"]))
        .expect("the second worker");
    assert_ring_urls(
        &rings,
        &registry,
        "old",
        &["http://first:8080", "http://second:8080"],
    );

    assert!(registry.replace(
        &first,
        worker_serving("http://first:8080", &["new", "kept"])
    ));
    assert_ring_urls(&rings, &registry, "old", &["http://second:8080"]);
    assert_ring_urls(&rings, &registry, "new", &["http://first:8080"]);
    assert_ring_urls(&rings, &registry, "kept", &["http://first:8080"]);
    assert_ring_urls(
        &rings,
        &registry,
        UNKNOWN_MODEL_ID,
        &["http://first:8080", "http://second:8080"],
    );
}

/// A ring is read off the generation, so two writers at once leave the rings
/// of the membership they leave, in whatever order they published.
#[test]
fn writers_at_once_leave_the_rings_of_the_membership_they_leave() {
    let rings = KeptRings::default();

    // Two registrations.
    let registry = WorkerRegistry::new();
    written_at_once(
        &registry,
        |registry| {
            registry
                .register(worker_serving("http://first:8080", &["model-a", "model-b"]))
                .expect("the first worker");
        },
        |registry| {
            registry
                .register(worker_serving("http://second:8080", &["model-b"]))
                .expect("the second worker");
        },
    );
    let both = ["http://first:8080", "http://second:8080"];
    assert_ring_urls(&rings, &registry, "model-a", &["http://first:8080"]);
    assert_ring_urls(&rings, &registry, "model-b", &both);
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &both);

    // The removals of a model's two workers: the last one is not restored.
    let registry = WorkerRegistry::new();
    let ids = both.map(|url| registry.register(worker_serving(url, &["model"])));
    let [first, second] = ids.map(|id| id.expect("a worker"));
    written_at_once(
        &registry,
        |registry| {
            registry.remove(&first).expect("the removal");
        },
        |registry| {
            registry.remove(&second).expect("the removal");
        },
    );
    assert_ring_urls(&rings, &registry, "model", &[]);
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &[]);

    // A removal and a registration: the new worker is not erased.
    let registry = WorkerRegistry::new();
    let old = registry
        .register(worker_serving("http://old:8080", &["model"]))
        .expect("the old worker");
    written_at_once(
        &registry,
        |registry| {
            registry.remove(&old).expect("the removal");
        },
        |registry| {
            registry
                .register(worker_serving("http://new:8080", &["model"]))
                .expect("the new worker");
        },
    );
    assert_ring_urls(&rings, &registry, "model", &["http://new:8080"]);
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &["http://new:8080"]);

    // A replacement that changes a worker's models and the removal of the
    // last other worker of one of them.
    let registry = WorkerRegistry::new();
    let first = registry
        .register(worker_serving("http://first:8080", &["old", "kept"]))
        .expect("the first worker");
    let second = registry
        .register(worker_serving("http://second:8080", &["old"]))
        .expect("the second worker");
    written_at_once(
        &registry,
        |registry| {
            assert!(registry.replace(
                &first,
                worker_serving("http://first:8080", &["new", "kept"])
            ));
        },
        |registry| {
            registry.remove(&second).expect("the removal");
        },
    );
    assert_ring_urls(&rings, &registry, "old", &[]);
    assert_ring_urls(&rings, &registry, "new", &["http://first:8080"]);
    assert_ring_urls(&rings, &registry, "kept", &["http://first:8080"]);
    assert_ring_urls(&rings, &registry, UNKNOWN_MODEL_ID, &["http://first:8080"]);
}

#[test]
fn an_encode_worker_registered_beside_a_prefill_one_stays_selectable() {
    let worker = |url: &str, worker_type| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("model"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        )
    };
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    let encode = worker("http://encode:8080", WorkerType::Encode);
    let (prefill, registered) = (
        worker("http://prefill:8080", WorkerType::Prefill),
        Arc::clone(&encode),
    );
    written_at_once(
        &registry,
        |registry| {
            registry.register(prefill).expect("the prefill worker");
        },
        |registry| {
            registry.register(registered).expect("the encode worker");
        },
    );

    assert!(encode.is_healthy_and_eligible());
    let info = SelectWorkerInfo {
        routing_key: Some("image-hash"),
        hash_ring: rings.of("model", &registry.get_routing_snapshot("model")),
        ..Default::default()
    };
    assert_eq!(
        ConsistentHashingPolicy::new().select_worker(&[encode], &info),
        Some(0),
        "a registered, healthy encode worker stays selectable"
    );
    assert_eq!(info.hash_ring.expect("the model's ring").worker_count(), 2);
}

/// A kept ring is the fill's starting point alone: whichever ring a
/// generation's cell is filled from, it holds the ring a whole build makes
/// of the generation's members.
#[test]
fn a_ring_filled_from_any_kept_ring_is_the_ring_of_its_generation() {
    let whole = HashRing::new((0..8).map(url));
    let starts = [
        None,
        Some(HashRing::new(["http://elsewhere:1", "http://elsewhere:2"])),
        Some(HashRing::new((0..3).map(url))),
        Some(HashRing::new((0..8).map(url))),
        Some(HashRing::new((0..64).map(url))),
    ];
    for kept in starts {
        let generation = registry_of(8).get_routing_snapshot("model");
        let filled = generation.hash_ring(|| kept.map(Arc::new));
        assert_same_placements(filled, &whole);
    }
}

#[test]
fn a_request_fills_the_ring_of_a_generation_once_and_keeps_none() {
    let registry = registry_of(2);
    let rings = KeptRings::default();
    let generation = registry.get_routing_snapshot("model");

    assert!(!generation.hash_ring_is_filled());
    let first = rings.of("model", &generation).expect("the model's ring");
    let again = rings
        .of("model", &registry.get_routing_snapshot("model"))
        .expect("the model's ring");
    assert!(Arc::ptr_eq(&first, &again), "one ring a generation");
    // With no warm task every generation's first reader builds its ring
    // whole: nothing was kept to reconcile from.
    assert!(rings.kept.is_empty(), "only the warm task keeps a ring");
}

/// A refresh of model cards publishes the generation again over the same
/// members and sends no event, so no task wakes for it: the next request
/// takes the ring the task kept, with nothing hashed and no ring built.
#[test]
fn a_generation_published_again_over_its_members_reads_the_kept_ring() {
    let registry = registry_of(4);
    let rings = KeptRings::default();
    rings.warm(&registry, "model");
    let warmed = kept(&rings, "model").expect("the task's step keeps the ring");
    let member = registry
        .get_routing_snapshot("model")
        .first()
        .cloned()
        .expect("a member of the model");

    assert!(registry.refresh_models(
        &member,
        vec![ModelCard::new("model").with_context_length(8_192)]
    ));
    let published_again = registry.get_routing_snapshot("model");
    assert!(!published_again.hash_ring_is_filled(), "a new generation");

    let read = rings
        .of("model", &published_again)
        .expect("the model's ring");
    assert!(
        Arc::ptr_eq(&read, &warmed),
        "the same members in the same order: the kept ring is this generation's"
    );
}

/// The task's step and the requests of a fresh generation race for its
/// cell: one of them fills it, and all read that one ring.
#[test]
fn readers_that_race_the_task_for_a_generation_read_the_one_ring_it_holds() {
    const READERS: usize = 7;
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    for round in 0..32 {
        registry
            .register(worker_serving(&url(round), &["model"]))
            .expect("a worker of the model");
        let generation = registry.get_routing_snapshot("model");
        let start = Barrier::new(READERS + 1);
        let read: Vec<Arc<HashRing>> = std::thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                rings.warm(&registry, "model");
            });
            let readers: Vec<_> = (0..READERS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        rings.of("model", &generation).expect("the model's ring")
                    })
                })
                .collect();
            readers
                .into_iter()
                .map(|reader| reader.join().expect("a reader"))
                .collect()
        });

        let kept = rings
            .kept
            .get("model")
            .map(|ring| Arc::clone(&ring))
            .expect("the task's step keeps the ring");
        for ring in &read {
            assert!(Arc::ptr_eq(ring, &kept), "round {round}");
        }
        assert_same_placements(&kept, &HashRing::new((0..=round).map(url)));
    }
}

/// The ring the task keeps for `model_id`.
fn kept(rings: &KeptRings, model_id: &str) -> Option<Arc<HashRing>> {
    rings.kept.get(model_id).map(|ring| Arc::clone(&ring))
}

#[tokio::test]
async fn the_task_fills_the_ring_of_a_generation_before_a_request_reads_it() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());

    registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("a worker of the model");
    let filled = |model_id: &str| {
        registry
            .get_routing_snapshot(model_id)
            .hash_ring_is_filled()
    };
    until("the generation's ring", || filled("model")).await;
    until("the model-less ring", || filled(UNKNOWN_MODEL_ID)).await;

    let read = rings
        .of("model", &registry.get_routing_snapshot("model"))
        .expect("the model's ring");
    let kept_ring = kept(&rings, "model").expect("the task keeps the ring it filled");
    assert!(
        Arc::ptr_eq(&read, &kept_ring),
        "a request reads the task's ring"
    );
}

/// Registrations that wait together are one fill, of the generation that
/// stands: the generations between them are never read, and build no ring.
#[tokio::test]
async fn the_task_fills_a_wave_once_from_the_ring_it_kept() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("the first worker");
    until("the first ring", || kept(&rings, "model").is_some()).await;

    let mut between = Vec::new();
    for n in 1..64 {
        registry
            .register(worker_serving(&url(n), &["model"]))
            .expect("a worker of the wave");
        between.push(registry.get_routing_snapshot("model"));
    }
    let standing = between.pop().expect("the generation that stands");
    until("the wave's ring", || standing.hash_ring_is_filled()).await;

    assert!(between
        .iter()
        .all(|generation| !generation.hash_ring_is_filled()));
    let ring = kept(&rings, "model").expect("the ring of the generation that stands");
    assert_same_placements(&ring, &HashRing::new((0..64).map(url)));
}

#[tokio::test]
async fn the_task_drops_the_ring_of_a_model_whose_last_worker_leaves() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    let id = registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("the model's one worker");
    until("the model's ring", || kept(&rings, "model").is_some()).await;
    until("the model-less ring", || {
        kept(&rings, UNKNOWN_MODEL_ID).is_some()
    })
    .await;

    registry.remove(&id).expect("the removal");
    until("no ring kept", || rings.kept.is_empty()).await;
}

/// A lag loses events: the task then fills every model the registry holds
/// and drops the kept ring of a model that left, whose removal it never
/// heard.
#[tokio::test]
async fn after_a_lag_the_task_fills_what_stands_and_drops_what_left() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    registry
        .register(worker_serving("http://stays:8080", &["stays"]))
        .expect("the worker that stays");
    let leaves = registry
        .register(worker_serving("http://leaves:8080", &["leaves"]))
        .expect("the worker that leaves");
    until("both rings", || {
        kept(&rings, "stays").is_some() && kept(&rings, "leaves").is_some()
    })
    .await;

    // More events than the registry's channel holds, the task never run
    // between them: the oldest, the removal of the model's last worker, is
    // lost to every listener.
    let mut witness = registry.subscribe_events();
    registry.remove(&leaves).expect("the removal");
    for _ in 0..2048 {
        let id = registry
            .register(worker_serving("http://wave:8080", &["stays"]))
            .expect("a worker of the wave");
        registry.remove(&id).expect("its removal");
    }
    assert!(
        matches!(witness.try_recv(), Err(TryRecvError::Lagged(_))),
        "the burst must overflow the channel, or this case holds nothing"
    );

    until("the ring of the model that left is dropped", || {
        kept(&rings, "leaves").is_none()
    })
    .await;
    until("the generation that stands is filled", || {
        registry.get_routing_snapshot("stays").hash_ring_is_filled()
    })
    .await;
    assert_ring_urls(&rings, &registry, "stays", &["http://stays:8080"]);
    assert_eq!(kept(&rings, "stays").expect("kept").worker_count(), 1);
}

/// The task holds neither the registry nor the rings: it ends with the
/// registry, and at the first event after the rings are gone.
#[tokio::test]
async fn the_task_ends_with_the_registry_or_the_rings() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    assert_eq!(Arc::weak_count(&rings), 1, "the task's own handle");
    drop(registry);
    until("its end with the registry", || Arc::weak_count(&rings) == 0).await;

    let registry = Arc::new(WorkerRegistry::new());
    let handles = Arc::weak_count(&registry);
    rings.keep_warm(&registry, &Handle::current());
    assert_eq!(Arc::weak_count(&registry), handles + 1);
    drop(rings);
    registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("a worker of the model");
    until("its end with the rings", || {
        Arc::weak_count(&registry) == handles
    })
    .await;
}
