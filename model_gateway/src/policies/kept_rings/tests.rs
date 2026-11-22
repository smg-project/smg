//! The cases of the kept rings: what the ring of a generation holds, who
//! fills it, what a request reads while it is filled, and what the warm
//! task keeps and drops.

use std::{sync::Barrier, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerType};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    test_support::no_health_check,
};
use tokio::runtime::Handle;

use super::*;
use crate::{
    policies::{ConsistentHashingPolicy, LoadBalancingPolicy, SelectWorkerInfo},
    routers::grpc::backend_client::BasicWorkerBuilder,
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

/// Waits until `holds`, the warm task and its fills running meanwhile.
async fn until(what: &str, mut holds: impl FnMut() -> bool) {
    let bound = Instant::now() + Duration::from_secs(60);
    while !holds() {
        assert!(Instant::now() < bound, "the warm task never got to: {what}");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
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
        let filled = generation.fill_hash_ring(kept.map(Arc::new));
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
/// takes the ring the task kept, with nothing hashed and no ring built, and
/// leaves it in the generation's cell.
#[test]
fn a_generation_published_again_over_its_members_reads_the_kept_ring() {
    let registry = registry_of(4);
    let rings = KeptRings::default();
    rings.warm(&registry, "model");
    let _task = Warming::start(&rings.warming);
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
    assert!(
        published_again.hash_ring_is_filled(),
        "the next read is of a filled ring"
    );
}

/// A replacement puts its worker last in the model's list and leaves it
/// where it stood in the ring, and the model-less list is in no order of
/// the ring's. A generation published again still takes the kept ring: the
/// ring is known by the list it was filled over, not by its own order.
#[test]
fn a_generation_published_again_takes_the_kept_ring_whatever_order_the_ring_keeps() {
    let registry = WorkerRegistry::new();
    let first = registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("the first worker");
    for n in 1..4 {
        registry
            .register(worker_serving(&url(n), &["model"]))
            .expect("a worker of the model");
    }
    let everything = || Changed {
        everything: true,
        models: Vec::new(),
    };
    let rings = KeptRings::default();
    rings.fill(&registry, everything());
    assert!(registry.replace(&first, worker_serving(&url(0), &["model"])));
    rings.fill(&registry, everything());
    let _task = Warming::start(&rings.warming);
    let listed: Vec<String> = registry
        .get_routing_snapshot("model")
        .iter()
        .map(|worker| worker.url().to_string())
        .collect();
    let warmed = kept(&rings, "model").expect("the task's step keeps the ring");
    assert_eq!(listed.last(), Some(&url(0)), "the replacement stands last");
    assert!(!warmed.is_over(&listed), "and where it stood in the ring");

    let member = registry
        .get_routing_snapshot("model")
        .first()
        .cloned()
        .expect("a member of the model");
    assert!(registry.refresh_models(
        &member,
        vec![ModelCard::new("model").with_context_length(8_192)]
    ));
    for model_id in ["model", UNKNOWN_MODEL_ID] {
        let published_again = registry.get_routing_snapshot(model_id);
        assert!(
            !published_again.hash_ring_is_filled(),
            "{model_id}: a new generation"
        );
        let read = rings.of(model_id, &published_again).expect("a ring");
        let warmed = kept(&rings, model_id).expect("the ring the task kept");
        assert!(Arc::ptr_eq(&read, &warmed), "{model_id}: the kept ring");
        assert!(
            published_again.hash_ring_is_filled(),
            "{model_id}: the next read is of a filled ring"
        );
    }
}

/// A generation over other members at equal count reads the kept ring and
/// leaves its cell empty: the kept ring is not its ring. The task's fill
/// then puts in the ring of its own members.
#[test]
fn a_generation_over_other_members_at_equal_count_takes_no_kept_ring() {
    let registry = registry_of(2);
    let rings = KeptRings::default();
    rings.warm(&registry, "model");
    let _task = Warming::start(&rings.warming);
    let warmed = kept(&rings, "model").expect("the task's step keeps the ring");

    registry.remove_by_url(&url(1));
    registry
        .register(worker_serving(&url(2), &["model"]))
        .expect("a worker in its place");
    let other_members = registry.get_routing_snapshot("model");
    assert_eq!(other_members.len(), warmed.worker_count());

    let read = rings.of("model", &other_members).expect("the kept ring");
    assert!(Arc::ptr_eq(&read, &warmed), "placed on the kept ring");
    assert!(
        !other_members.hash_ring_is_filled(),
        "the kept ring is not this generation's: the cell waits for the fill"
    );

    rings.warm(&registry, "model");
    let both = [url(0), url(2)];
    assert_ring_urls(
        &rings,
        &registry,
        "model",
        &both.each_ref().map(String::as_str),
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
            .map(|kept| Arc::clone(&kept.ring))
            .expect("the task's step keeps the ring");
        for ring in &read {
            assert!(Arc::ptr_eq(ring, &kept), "round {round}");
        }
        assert_same_placements(&kept, &HashRing::new((0..=round).map(url)));
    }
}

/// The ring the task keeps for `model_id`.
fn kept(rings: &KeptRings, model_id: &str) -> Option<Arc<HashRing>> {
    rings.kept.get(model_id).map(|kept| Arc::clone(&kept.ring))
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
    // The fill runs on a thread of its own: it writes the generation's cell
    // first and keeps the ring after it.
    until("the wave's ring is kept", || {
        kept(&rings, "model").is_some_and(|ring| ring.worker_count() == 64)
    })
    .await;
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
    assert_eq!(
        rings.warming.load(Ordering::Relaxed),
        0,
        "no task lives: a request fills again"
    );

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

/// While the task lives a request fills no ring: ahead of the task it reads
/// the ring kept before the membership change, and leaves the generation's
/// cell to the task.
#[tokio::test]
async fn a_request_ahead_of_the_task_reads_the_kept_ring_and_fills_none() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("the first worker");
    until("the first ring", || kept(&rings, "model").is_some()).await;
    let first = kept(&rings, "model").expect("the first ring");

    // The task takes no turn between this registration and the read.
    registry
        .register(worker_serving(&url(1), &["model"]))
        .expect("the second worker");
    let generation = registry.get_routing_snapshot("model");
    let read = rings.of("model", &generation).expect("the kept ring");
    assert!(
        Arc::ptr_eq(&read, &first),
        "the ring of the membership before"
    );
    assert_eq!(read.worker_count(), 1);
    assert!(!generation.hash_ring_is_filled(), "the fill is the task's");

    until("the generation's ring", || generation.hash_ring_is_filled()).await;
    let read = rings.of("model", &generation).expect("the model's ring");
    assert_same_placements(&read, &HashRing::new((0..2).map(url)));
}

/// A model's first requests, before the task has kept a ring for it, read
/// no ring and are placed with none: no request builds a first ring.
#[tokio::test]
async fn a_models_first_requests_read_no_ring_until_the_task_keeps_one() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());

    // The task takes no turn between this registration and the read.
    registry
        .register(worker_serving(&url(0), &["model"]))
        .expect("the model's first worker");
    let generation = registry.get_routing_snapshot("model");
    assert!(
        rings.of("model", &generation).is_none(),
        "no ring is kept yet"
    );
    assert!(!generation.hash_ring_is_filled(), "the fill is the task's");

    until("the first ring", || generation.hash_ring_is_filled()).await;
    assert_ring_urls(&rings, &registry, "model", &[url(0).as_str()]);
}

/// A task counts as living from its start to its end, and a runtime that
/// ends drops it: the requests then fill the rings of their generations
/// again, where they would read a ring nobody renews.
#[test]
fn a_task_that_ended_with_its_runtime_leaves_the_fills_to_the_requests() {
    let registry = Arc::new(registry_of(1));
    let rings = Arc::new(KeptRings::default());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    rings.keep_warm(&registry, runtime.handle());
    let generation = registry.get_routing_snapshot("model");
    assert!(
        rings.of("model", &generation).is_none(),
        "a task lives: the fill is its own"
    );

    // The task never ran, and ends here with its runtime.
    drop(runtime);
    assert_ring_urls(&rings, &registry, "model", &[url(0).as_str()]);
    assert!(generation.hash_ring_is_filled(), "the request filled it");
}

/// A replacement names the models of the worker it replaced too: the task
/// drops the kept ring of a model the replacement emptied, and fills the
/// ring of the model the worker serves now.
#[tokio::test]
async fn the_task_follows_a_replacement_that_moves_a_worker_to_another_model() {
    let registry = Arc::new(WorkerRegistry::new());
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());
    let id = registry
        .register(worker_serving(&url(0), &["old"]))
        .expect("the worker");
    until("the ring of its model", || kept(&rings, "old").is_some()).await;

    assert!(registry.replace(&id, worker_serving(&url(0), &["new"])));
    until("the ring of its new model", || {
        kept(&rings, "new").is_some()
    })
    .await;
    until("no ring of the model it left", || {
        kept(&rings, "old").is_none()
    })
    .await;
}

/// What the registry held before the task listened sent it no event: its
/// first pass fills the ring of every model the registry holds.
#[tokio::test]
async fn the_tasks_first_pass_fills_what_the_registry_held_before_it() {
    let registry = Arc::new(registry_of(3));
    let rings = Arc::new(KeptRings::default());
    rings.keep_warm(&registry, &Handle::current());

    let filled = |model_id: &str| {
        registry
            .get_routing_snapshot(model_id)
            .hash_ring_is_filled()
    };
    until("the model's ring", || filled("model")).await;
    until("the model-less ring", || filled(UNKNOWN_MODEL_ID)).await;
    assert_eq!(kept(&rings, "model").expect("kept").worker_count(), 3);
}

/// Beside the task's step the requests of a fresh generation read the ring
/// kept before it or the generation's own, and build none: only a ring of
/// the task is ever read.
#[test]
fn requests_beside_the_tasks_step_read_the_kept_ring_or_the_generations_own() {
    const READERS: usize = 7;
    let registry = WorkerRegistry::new();
    let rings = KeptRings::default();
    let _task = Warming::start(&rings.warming);
    let mut before: Option<Arc<HashRing>> = None;
    for round in 0..32 {
        registry
            .register(worker_serving(&url(round), &["model"]))
            .expect("a worker of the model");
        let generation = registry.get_routing_snapshot("model");
        let start = Barrier::new(READERS + 1);
        let read: Vec<Option<Arc<HashRing>>> = std::thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                rings.warm(&registry, "model");
            });
            let readers: Vec<_> = (0..READERS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        rings.of("model", &generation)
                    })
                })
                .collect();
            readers
                .into_iter()
                .map(|reader| reader.join().expect("a reader"))
                .collect()
        });

        let own = kept(&rings, "model").expect("the task's step keeps the ring");
        for ring in read {
            let is =
                |other: &Arc<HashRing>| ring.as_ref().is_some_and(|ring| Arc::ptr_eq(ring, other));
            assert!(
                is(&own) || before.as_ref().map_or_else(|| ring.is_none(), is),
                "round {round}"
            );
        }
        assert_same_placements(&own, &HashRing::new((0..=round).map(url)));
        before = Some(own);
    }
}

/// A lag can be met while the waiting events are drained, with no wait
/// before it: every model is then filled, as after a lag met in the wait.
#[test]
fn a_lag_met_in_the_drain_names_every_model() {
    let registry = registry_of(1);
    let mut events = registry.subscribe_events();
    for _ in 0..2048 {
        let id = registry
            .register(worker_serving("http://wave:8080", &["model"]))
            .expect("a worker of the wave");
        registry.remove(&id).expect("its removal");
    }

    let mut changed = Changed::default();
    changed.drain(&mut events);
    assert!(changed.everything, "the burst overflows the channel");
}

/// Where one model holds every worker the model-less ring is over that
/// model's members: its fill starts from the model's ring and hashes
/// nothing again.
#[test]
fn the_model_less_fill_starts_from_the_ring_of_the_only_model() {
    let everything = || Changed {
        everything: true,
        models: Vec::new(),
    };
    let registry = registry_of(4);
    let rings = KeptRings::default();
    rings.fill(&registry, everything());
    let of_the_model = kept(&rings, "model").expect("the model's ring");
    let model_less = kept(&rings, UNKNOWN_MODEL_ID).expect("the model-less ring");
    assert!(model_less.shares_positions_with(&of_the_model));

    // With a second model the model-less ring is over more than either's.
    registry
        .register(worker_serving("http://other:8080", &["other"]))
        .expect("a worker of another model");
    rings.fill(&registry, everything());
    let of_the_model = kept(&rings, "model").expect("the model's ring");
    let model_less = kept(&rings, UNKNOWN_MODEL_ID).expect("the model-less ring");
    assert_eq!(model_less.worker_count(), 5);
    assert!(!model_less.shares_positions_with(&of_the_model));
}

/// The rings are current where a request placed now walks the ring of its
/// own membership: always with no task, since the request fills that ring,
/// and with a task once its fill has followed the membership change.
#[tokio::test]
async fn the_rings_are_current_once_the_task_has_followed_a_change() {
    let registry = Arc::new(registry_of(1));
    let rings = Arc::new(KeptRings::default());
    assert!(rings.are_current(&registry), "a request fills its own ring");

    rings.keep_warm(&registry, &Handle::current());
    assert!(!rings.are_current(&registry), "the first pass is owed");
    until("the first pass", || rings.are_current(&registry)).await;

    // The task takes no turn between this registration and the read.
    registry
        .register(worker_serving(&url(1), &["model"]))
        .expect("a second worker");
    assert!(!rings.are_current(&registry), "the fill is owed");
    until("the fill", || rings.are_current(&registry)).await;
    let both = [url(0), url(1)];
    assert_ring_urls(
        &rings,
        &registry,
        "model",
        &both.each_ref().map(String::as_str),
    );
}

/// A refresh of model cards publishes the generations again and sends no
/// event, so the task does not come. The rings still read current with no
/// request in between: the reader takes the kept rings as a request would.
#[test]
fn the_rings_read_current_after_a_card_refresh_with_no_request() {
    let registry = registry_of(4);
    let rings = KeptRings::default();
    rings.fill(
        &registry,
        Changed {
            everything: true,
            models: Vec::new(),
        },
    );
    let _task = Warming::start(&rings.warming);
    assert!(rings.are_current(&registry), "the task's step filled them");

    let member = registry
        .get_routing_snapshot("model")
        .first()
        .cloned()
        .expect("a member of the model");
    assert!(registry.refresh_models(
        &member,
        vec![ModelCard::new("model").with_context_length(8_192)]
    ));
    assert!(
        !registry.get_routing_snapshot("model").hash_ring_is_filled(),
        "a new generation"
    );
    assert!(
        rings.are_current(&registry),
        "the kept rings are the rings of the generations published again"
    );
    for model_id in ["model", UNKNOWN_MODEL_ID] {
        assert!(
            registry
                .get_routing_snapshot(model_id)
                .hash_ring_is_filled(),
            "{model_id}: its ring stands in its cell"
        );
    }
}

/// The task alone writes a kept ring, from the generation that stands when
/// it fills, so a kept ring never steps back to an earlier membership. A
/// request can hold an earlier generation, and writes no kept ring.
#[test]
fn a_kept_ring_never_steps_back_and_a_request_writes_none() {
    let registry = registry_of(2);
    let rings = KeptRings::default();
    let earlier = registry.get_routing_snapshot("model");
    registry
        .register(worker_serving(&url(2), &["model"]))
        .expect("a third worker");
    rings.warm(&registry, "model");
    let standing = kept(&rings, "model").expect("the ring of the generation that stands");
    assert_eq!(standing.worker_count(), 3);

    // A request that still holds the earlier generation reads the kept
    // ring while a task lives, and fills its own generation's ring with
    // none: either way the kept ring stays the one that stands.
    for (task_lives, on_ring) in [(true, 3), (false, 2)] {
        let _task = task_lives.then(|| Warming::start(&rings.warming));
        let read = rings.of("model", &earlier).expect("a ring");
        assert_eq!(read.worker_count(), on_ring);
        let kept = kept(&rings, "model").expect("a kept ring");
        assert!(Arc::ptr_eq(&kept, &standing), "a request writes none");
    }
}
