//! The gateway's cases over the registry: the routing pools of its value and
//! their window floors, what the workers of a generation say together, what a
//! refresh of a worker's cards leaves a reader, and the end of a ZMQ worker's
//! handshake at its removal (the handshake is the gateway's connector's).

use std::{
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_worker::{
    identity::basic::BasicWorker,
    test_support::{no_health_check, worker_with_model_aliases},
};

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

/// A gRPC worker of `model` whose card names `window` as its context
/// window.
fn worker_with_a_window(url: &str, model: &str, window: u32) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .models(vec![ModelCard::new(model).with_context_length(window)])
            .build(),
    )
}

/// The floor of a regular pool whose narrowest window is `window`.
fn a_regular_floor_of(window: u32) -> WindowFloor {
    WindowFloor {
        budget: Some(window),
        prefill: None,
    }
}

/// A reader of a pool's floor for `model_id`: the narrowest window any
/// worker names, as the budget's. The pipeline's own rule is its context
/// check's to hold; these cases hold what the snapshot keeps of a reading.
fn the_narrowest_window(model_id: &str) -> impl Fn(&[Arc<dyn Worker>]) -> WindowFloor + '_ {
    move |workers| WindowFloor {
        budget: workers
            .iter()
            .filter_map(|worker| worker.context_length(model_id))
            .min(),
        prefill: None,
    }
}

/// Whether a worker of the list names `window` for qwen: an answer read
/// from the cards, as a snapshot keeps them.
fn a_worker_names(window: u32) -> impl Fn(&[Arc<dyn Worker>]) -> bool {
    move |workers| {
        workers
            .iter()
            .any(|worker| worker.context_length("qwen") == Some(window))
    }
}

#[test]
fn a_refresh_that_changes_a_card_is_read_by_the_next_kept_answer() {
    let registry = WorkerRegistry::new();
    let worker = worker_with_a_window("grpc://127.0.0.1:8401", "qwen", 8_192);
    registry.register(Arc::clone(&worker)).unwrap();
    let floor = |snapshot: &ModelWorkerSnapshot| {
        snapshot.window_floor(
            RoutingPool::GrpcPipelineRegular,
            "qwen",
            the_narrowest_window("qwen"),
        )
    };
    let model = registry.model_routing_snapshot("qwen").unwrap();
    let global = registry.global_generation();
    assert_eq!(floor(&model), Some(a_regular_floor_of(8_192)));
    assert_eq!(floor(&global), Some(a_regular_floor_of(8_192)));
    assert!(!model.harmony(a_worker_names(4_096)));

    assert!(registry.refresh_models(
        &worker,
        vec![ModelCard::new("qwen").with_context_length(4_096)],
    ));
    // The snapshots drawn before the refresh keep what they answered; the
    // ones a request draws now read the cards again.
    assert_eq!(floor(&model), Some(a_regular_floor_of(8_192)));
    assert_eq!(floor(&global), Some(a_regular_floor_of(8_192)));
    assert!(!model.harmony(a_worker_names(4_096)));
    let model = registry.model_routing_snapshot("qwen").unwrap();
    assert_eq!(floor(&model), Some(a_regular_floor_of(4_096)));
    assert_eq!(
        floor(&registry.global_generation()),
        Some(a_regular_floor_of(4_096))
    );
    assert!(model.harmony(a_worker_names(4_096)));
}

/// A reader of what a list's workers say together for qwen: the reasoning
/// parser the first card names, with nothing else said, counted in
/// `readings`. The stage's own rules are its to hold; the case below holds
/// what the snapshot keeps of a reading.
fn the_reasoning_parser_named(
    readings: &AtomicUsize,
) -> impl Fn(&[Arc<dyn Worker>]) -> ModelLabels + '_ {
    move |workers| {
        readings.fetch_add(1, Ordering::Relaxed);
        ModelLabels {
            tool_parser: None,
            reasoning_parser: workers.iter().find_map(|worker| {
                let card = worker.models().into_iter().find(|card| card.id == "qwen")?;
                card.reasoning_parser
            }),
            grammar_start: GrammarStart::Unknown,
            vocabulary: TagVocabulary::Current,
        }
    }
}

#[test]
fn what_the_workers_say_together_is_read_once_for_a_membership() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:8401")
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .models(vec![ModelCard::new("qwen").with_reasoning_parser("qwen3")])
            .build(),
    );
    registry.register(Arc::clone(&worker)).unwrap();
    let readings = AtomicUsize::new(0);
    let read = the_reasoning_parser_named(&readings);
    let parser = |snapshot: &ModelWorkerSnapshot, model_id: &str| {
        snapshot
            .labels(model_id, &read)
            .map(|labels| labels.reasoning_parser.clone())
    };
    let model = registry.model_routing_snapshot("qwen").unwrap();
    for _request in 0..3 {
        assert_eq!(parser(&model, "qwen"), Some(Some("qwen3".to_string())));
    }
    assert_eq!(readings.load(Ordering::Relaxed), 1);
    // Another spelling than the kept one is the caller's to read, and so is
    // the snapshot every unknown model shares: neither reads nor keeps.
    assert_eq!(parser(&model, "Qwen"), None);
    assert_eq!(
        parser(&registry.get_routing_snapshot("nobody"), "nobody"),
        None
    );
    assert_eq!(readings.load(Ordering::Relaxed), 1);

    assert!(registry.refresh_models(
        &worker,
        vec![ModelCard::new("qwen").with_reasoning_parser("deepseek_r1")],
    ));
    // The snapshot drawn before the refresh keeps what it answered; the one
    // a request draws now reads the cards again, once.
    assert_eq!(parser(&model, "qwen"), Some(Some("qwen3".to_string())));
    let model = registry.model_routing_snapshot("qwen").unwrap();
    for _request in 0..3 {
        assert_eq!(
            parser(&model, "qwen"),
            Some(Some("deepseek_r1".to_string()))
        );
    }
    assert_eq!(readings.load(Ordering::Relaxed), 2);
}

#[test]
fn a_refresh_publishes_the_snapshot_of_every_model_that_holds_the_worker() {
    let registry = WorkerRegistry::new();
    let cards = |window| {
        vec![
            ModelCard::new("qwen").with_context_length(window),
            ModelCard::new("llama").with_context_length(window),
        ]
    };
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:8401")
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .models(cards(8_192))
            .build(),
    );
    registry.register(Arc::clone(&worker)).unwrap();
    let floor = |model_id: &str| {
        registry
            .model_routing_snapshot(model_id)
            .unwrap()
            .window_floor(
                RoutingPool::GrpcPipelineRegular,
                model_id,
                the_narrowest_window(model_id),
            )
    };
    for model_id in ["qwen", "llama"] {
        assert_eq!(floor(model_id), Some(a_regular_floor_of(8_192)));
    }

    assert!(registry.refresh_models(&worker, cards(4_096)));
    // One worker under two models: each model's snapshot is published
    // again, so the next answer of each is read from the new cards.
    for model_id in ["qwen", "llama"] {
        assert_eq!(
            floor(model_id),
            Some(a_regular_floor_of(4_096)),
            "{model_id}"
        );
    }
}

#[test]
fn a_pools_window_floor_is_kept_with_the_spelling_it_was_read_for() {
    let registry = WorkerRegistry::new();
    let card = |window| {
        ModelCard::new("qwen")
            .with_alias("qwen-latest")
            .with_context_length(window)
    };
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:8401")
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .models(vec![card(8_192)])
            .build(),
    );
    registry.register(Arc::clone(&worker)).unwrap();
    let snapshot = registry.model_routing_snapshot("qwen").unwrap();
    let pool = RoutingPool::GrpcPipelineRegular;
    assert_eq!(
        snapshot.window_floor(pool, "qwen", the_narrowest_window("qwen")),
        Some(a_regular_floor_of(8_192))
    );
    // Another spelling of the model is not answered from the kept figure,
    // and takes no place: the first spelling stays.
    assert_eq!(
        snapshot.window_floor(pool, "qwen-latest", the_narrowest_window("qwen-latest")),
        None
    );
    assert_eq!(
        snapshot.window_floor(pool, "qwen", the_narrowest_window("qwen")),
        Some(a_regular_floor_of(8_192))
    );
    // Kept, not read again: cards written past the registry leave this
    // snapshot's figure as it was read.
    worker.set_models(vec![card(4_096)]);
    assert_eq!(worker.context_length("qwen"), Some(4_096));
    assert_eq!(
        snapshot.window_floor(pool, "qwen", the_narrowest_window("qwen")),
        Some(a_regular_floor_of(8_192))
    );
    // A pool with no worker binds nothing under any spelling, and keeps none.
    for spelling in ["qwen", "qwen-latest"] {
        assert_eq!(
            snapshot.window_floor(
                RoutingPool::GrpcDecode,
                spelling,
                the_narrowest_window(spelling)
            ),
            Some(WindowFloor::default())
        );
    }
}

#[test]
fn a_refresh_that_repeats_the_cards_publishes_nothing() {
    let registry = WorkerRegistry::new();
    let worker = worker_with_a_window("grpc://127.0.0.1:8401", "qwen", 8_192);
    registry.register(Arc::clone(&worker)).unwrap();
    let model = registry.model_routing_snapshot("qwen").unwrap();
    let global = registry.global_generation();

    // The cards of its registration, then the cards of an earlier refresh.
    for window in [8_192, 4_096, 4_096] {
        let changed = worker.context_length("qwen") != Some(window);
        let model = registry.model_routing_snapshot("qwen").unwrap();
        let global = registry.global_generation();
        assert_eq!(
            registry.refresh_models(
                &worker,
                vec![ModelCard::new("qwen").with_context_length(window)],
            ),
            changed
        );
        assert_eq!(
            Arc::ptr_eq(&model, &registry.model_routing_snapshot("qwen").unwrap()),
            !changed
        );
        assert_eq!(
            Arc::ptr_eq(&global, &registry.global_generation()),
            !changed
        );
    }
    assert!(!Arc::ptr_eq(
        &model,
        &registry.model_routing_snapshot("qwen").unwrap()
    ));
    assert!(!Arc::ptr_eq(&global, &registry.global_generation()));
}

#[test]
fn a_refresh_keeps_the_list_its_order_and_its_handles() {
    let registry = WorkerRegistry::new();
    let workers = [
        "grpc://127.0.0.1:8401",
        "grpc://127.0.0.1:8402",
        "grpc://127.0.0.1:8403",
    ]
    .map(|url| worker_with_a_window(url, "qwen", 8_192));
    for worker in &workers {
        registry.register(Arc::clone(worker)).unwrap();
    }
    let list = registry.get_by_model("qwen");
    let fleet = registry.get_routing_workers();

    // The first and the middle worker: neither moves to the list's end. The
    // second refresh renames the model: the worker stays where its
    // registration indexed it and is found there by its handle.
    for (refreshed, card) in [
        (
            &workers[1],
            ModelCard::new("qwen").with_context_length(4_096),
        ),
        (
            &workers[0],
            ModelCard::new("llama").with_context_length(4_096),
        ),
    ] {
        let before = registry.model_routing_snapshot("qwen").unwrap();
        assert!(registry.refresh_models(refreshed, vec![card]));
        let after = registry.model_routing_snapshot("qwen").unwrap();
        assert!(!Arc::ptr_eq(&before, &after));
        assert!(Arc::ptr_eq(&list, &registry.get_by_model("qwen")));
        assert!(registry.get_by_model("llama").is_empty());
        let fleet_now = registry.get_routing_workers();
        assert_eq!(fleet.len(), fleet_now.len());
        assert!(fleet
            .iter()
            .zip(fleet_now.iter())
            .all(|(before, after)| Arc::ptr_eq(before, after)));
    }
    assert!(list
        .iter()
        .zip(&workers)
        .all(|(listed, registered)| Arc::ptr_eq(listed, registered)));
}

#[test]
fn routing_pool_reuses_uniform_model_snapshot() {
    let registry = WorkerRegistry::new();
    for url in ["http://worker1:8080", "http://worker2:8080"] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("llama-3"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .health_config(no_health_check())
                    .build(),
            ))
            .unwrap();
    }

    let all = registry.get_by_model("llama-3");
    let first = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    let second = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);

    assert!(Arc::ptr_eq(&all, &first));
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn routing_pool_is_cached_and_replaced_with_membership_snapshot() {
    let registry = WorkerRegistry::new();
    let http_id = registry
        .register(Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .model(ModelCard::new("llama-3"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://worker2:8080")
                .model(ModelCard::new("llama-3"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();

    let first = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    let second = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].url(), "http://worker1:8080");
    assert!(Arc::ptr_eq(&first, &second));

    registry
        .transition_status(&http_id, WorkerStatus::NotReady)
        .unwrap();
    let after_health_change = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&first, &after_health_change));
    assert!(!after_health_change[0].is_available());

    registry.remove(&http_id).unwrap();
    let after_remove = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert!(after_remove.is_empty());
    assert!(!Arc::ptr_eq(&first, &after_remove));
    assert_eq!(first.len(), 1, "held snapshots remain immutable");
}

#[test]
fn wildcard_routing_pool_reuses_global_snapshot() {
    let registry = WorkerRegistry::new();
    let http_worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .model(ModelCard::new("model-a"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(no_health_check())
            .build(),
    );
    let http_id = registry.register(http_worker.clone()).unwrap();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://worker2:8080")
                .model(ModelCard::new("model-b"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();

    let before = Arc::strong_count(&http_worker);
    let first = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    let after_first = Arc::strong_count(&http_worker);
    let second = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);

    assert_eq!(first.len(), 1);
    assert_eq!(first[0].url(), "http://worker1:8080");
    assert!(Arc::ptr_eq(&first, &second));
    assert!(after_first > before);
    assert_eq!(Arc::strong_count(&http_worker), after_first);

    registry
        .transition_status(&http_id, WorkerStatus::NotReady)
        .unwrap();
    let after_health_change = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&first, &after_health_change));
    assert!(!after_health_change[0].is_available());

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .model(ModelCard::new("model-a"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    assert!(registry.replace(&http_id, replacement));
    let after_replace = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert!(after_replace.is_empty());
    assert!(!Arc::ptr_eq(&first, &after_replace));
    assert_eq!(
        registry
            .get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::GrpcPipelineRegular)
            .len(),
        2
    );
    assert_eq!(first.len(), 1, "held snapshots remain immutable");
}

#[test]
fn concurrent_global_snapshot_updates_preserve_all_workers() {
    let registry = Arc::new(WorkerRegistry::new());
    std::thread::scope(|scope| {
        for i in 0..8 {
            let registry = Arc::clone(&registry);
            scope.spawn(move || {
                registry
                    .register(Arc::new(
                        BasicWorkerBuilder::new(format!("http://worker{i}:8080"))
                            .worker_type(WorkerType::Regular)
                            .connection_mode(ConnectionMode::Http)
                            .health_config(no_health_check())
                            .build(),
                    ))
                    .unwrap();
            });
        }
    });

    let workers = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert_eq!(workers.len(), 8);
    assert_eq!(
        workers
            .iter()
            .map(|worker| worker.url())
            .collect::<HashSet<_>>()
            .len(),
        8
    );
}

/// The writer replaces the prefill worker before the decode worker on
/// every generation, so in every membership state that ever truly
/// existed, generation(prefill) >= generation(decode). A rebuild scan
/// that interleaved with the writer could publish the reverse — an old
/// prefill paired with a newer decode — which is exactly the torn pair a
/// live PD selection must never draw from.
#[test]
fn global_snapshot_never_publishes_a_torn_generation() {
    use std::sync::atomic::AtomicBool;

    fn pd_worker(url: &str, worker_type: WorkerType, generation: usize) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("model"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .label("generation", generation.to_string())
                .health_config(no_health_check())
                .build(),
        )
    }

    fn generation(pool: &[Arc<dyn Worker>]) -> usize {
        pool[0]
            .metadata()
            .spec
            .labels
            .get("generation")
            .and_then(|generation| generation.parse().ok())
            .expect("every test worker carries a generation label")
    }

    let registry = WorkerRegistry::new();
    registry.register_or_replace(pd_worker("grpc://prefill:1", WorkerType::Prefill, 0));
    registry.register_or_replace(pd_worker("grpc://decode:1", WorkerType::Decode, 0));

    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for generation in 1..=300 {
                registry.register_or_replace(pd_worker(
                    "grpc://prefill:1",
                    WorkerType::Prefill,
                    generation,
                ));
                registry.register_or_replace(pd_worker(
                    "grpc://decode:1",
                    WorkerType::Decode,
                    generation,
                ));
            }
            done.store(true, Ordering::Release);
        });
        scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let snapshot = registry.get_routing_snapshot(UNKNOWN_MODEL_ID);
                let prefill = snapshot.pool(RoutingPool::GrpcPrefill);
                let decode = snapshot.pool(RoutingPool::GrpcDecode);
                if prefill.is_empty() || decode.is_empty() {
                    continue;
                }
                assert!(
                    generation(prefill) >= generation(decode),
                    "torn snapshot: decode generation passed prefill"
                );
            }
        });
    });
}

#[test]
fn routing_pools_filter_regular_workers_by_transport() {
    let registry = WorkerRegistry::new();
    let (sglang, tokenspeed) = (RuntimeType::Sglang, RuntimeType::TokenSpeed);
    let workers = [
        (
            "http://regular",
            WorkerType::Regular,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://regular",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://regular",
            WorkerType::Regular,
            ConnectionMode::Zmq,
            sglang,
        ),
        (
            "http://prefill",
            WorkerType::Prefill,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://prefill",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://prefill",
            WorkerType::Prefill,
            ConnectionMode::Zmq,
            sglang,
        ),
        (
            "zmq://ts-prefill",
            WorkerType::Prefill,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
        (
            "http://decode",
            WorkerType::Decode,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://decode",
            WorkerType::Decode,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://ts-decode",
            WorkerType::Decode,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
        (
            "grpc://encode",
            WorkerType::Encode,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "http://encode",
            WorkerType::Encode,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "zmq://ts-encode",
            WorkerType::Encode,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
    ];
    for (url, worker_type, connection_mode, runtime_type) in workers {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("model"))
                    .worker_type(worker_type)
                    .connection_mode(connection_mode)
                    .runtime_type(runtime_type)
                    .health_config(no_health_check())
                    .build(),
            ))
            .unwrap();
    }

    let urls = |pool| {
        registry
            .get_routing_pool("model", pool)
            .iter()
            .map(|worker| worker.url().to_string())
            .collect::<HashSet<_>>()
    };

    assert_eq!(
        urls(RoutingPool::HttpRegular),
        HashSet::from(["http://regular".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcPipelineRegular),
        HashSet::from(["grpc://regular".to_string(), "zmq://regular".to_string()])
    );
    // The disaggregated pools hold gRPC workers and the direct-ZMQ engines
    // whose wire carries the rendezvous and the encode request: a ZMQ
    // prefill worker on another wire that slipped past registration
    // must not appear.
    assert_eq!(
        urls(RoutingPool::GrpcPrefill),
        HashSet::from(["grpc://prefill".to_string(), "zmq://ts-prefill".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcDecode),
        HashSet::from(["grpc://decode".to_string(), "zmq://ts-decode".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcEncode),
        HashSet::from(["grpc://encode".to_string(), "zmq://ts-encode".to_string()])
    );
    // The HTTP PD pools are HTTP-only: the router proxies plain HTTP to
    // the selected URL, so gRPC and ZMQ workers must never be selectable.
    assert_eq!(
        urls(RoutingPool::HttpPrefill),
        HashSet::from(["http://prefill".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::HttpDecode),
        HashSet::from(["http://decode".to_string()])
    );
}

#[test]
fn routing_pool_alias_reuses_canonical_snapshot() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_with_model_aliases(
            "http://worker:8080",
            "canonical-model",
            &["model-alias"],
            WorkerType::Regular,
        ))
        .unwrap();

    let canonical = registry.get_routing_pool("canonical-model", RoutingPool::HttpRegular);
    let alias = registry.get_routing_pool("model-alias", RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&canonical, &alias));
}

/// A ZMQ worker whose handshake driver is in flight, holding the socket
/// binds derived from `dir`. The handshake takes an OS-assigned loopback
/// port: the port derived from a temporary ipc path is a hash into a band
/// other listeners on the host may hold, and a collision there fails the
/// handshake bind after the data-plane sockets are already bound.
fn connecting_zmq_worker(dir: &std::path::Path) -> Arc<BasicWorker> {
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("ipc://{}", dir.join("ts0.ipc").display()))
            .connection_mode(ConnectionMode::Zmq)
            .zmq_handshake_address("tcp://127.0.0.1:0")
            .health_config(no_health_check())
            .build(),
    );
    assert!(
        !worker.zmq_health_check(),
        "worker is not ready until the handshake lands"
    );
    assert!(
        worker.zmq_connect_abort.load_full().is_some(),
        "probe must start the background handshake driver"
    );
    worker
}

/// A removed worker's ZMQ handshake driver must not outlive its registry
/// entry: it holds the worker's socket binds until it lands, which would
/// fail a re-registration at the same URL.
#[tokio::test]
async fn remove_aborts_the_zmq_handshake_driver() {
    let dir = tempfile::tempdir().unwrap();
    let registry = WorkerRegistry::new();
    let worker = connecting_zmq_worker(dir.path());

    let id = registry
        .register(worker.clone() as Arc<dyn Worker>)
        .unwrap();
    registry.remove(&id).expect("worker removed");

    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "removal must abort the in-flight handshake driver"
    );
}

/// Same for a replacement: it brings its own backend-client slot, so the
/// old instance's driver could only collide with the new worker's connect.
#[tokio::test]
async fn replace_aborts_the_replaced_workers_zmq_handshake_driver() {
    let dir = tempfile::tempdir().unwrap();
    let registry = WorkerRegistry::new();
    let worker = connecting_zmq_worker(dir.path());
    let url = worker.url().to_string();

    let id = registry
        .register(worker.clone() as Arc<dyn Worker>)
        .unwrap();
    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .build(),
    );
    assert!(registry.replace(&id, replacement));

    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "replacement must abort the replaced worker's handshake driver"
    );
}

/// The registry holds no generation for a model it never saw, where the
/// routing read answers the empty one: a reader that asks whether a model
/// has a generation is told of none that nobody published.
#[test]
fn a_model_the_registry_never_saw_has_no_generation() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_with_a_window("grpc://127.0.0.1:8431", "qwen", 8_192))
        .unwrap();

    assert!(registry.model_routing_snapshot("qwen").is_some());
    assert!(registry.model_routing_snapshot("never-seen").is_none());
    assert!(registry.get_routing_snapshot("never-seen").is_empty());
}
