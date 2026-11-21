//! One request's routing, head to head, over one fleet of workers a model,
//! every one ready unless the cell says otherwise. Two paths:
//!
//! - `classic`: the classic policies over the worker registry.
//! - `object`: the policy engine asked where the classic placement's pass
//!   asks its policy. Everything is built as `classic` builds it (the
//!   registry, the policies a registration leaves, each worker's completion
//!   sink, the load feed) and the request runs the same function, pass and
//!   all; the one difference is the policy the registry hands the pass,
//!   which asks the engine.
//! - `records`: the policy engine's whole decision over records written for
//!   that one decision from the gateway's own data: the registry's list, the
//!   worker objects and the load feed. No classic policy is built and no
//!   classic placement pass runs: the engine of the document the routing
//!   settings give for the policy decides, and its stages are told the
//!   request's issue, its dispatch and its end. Every policy the `--policy`
//!   flag accepts is asked by its name.
//!
//! A cell is one path, one route, one policy, one fleet size and one count
//! of callers. Each caller routes requests back to back: the placement, the
//! worker's load guard and what its drop tells the model's policy.
//! The cell prints its decisions a second over every caller and one caller's
//! time a decision. The callers are threads that stand for the whole cell,
//! and each makes its first decisions untimed. The gateway's metrics
//! recorder runs, as in a serving process, and is drained between the timed
//! stretches. The binary runs on the gateway's stock allocator, as the
//! serving binary does.
//!
//! `--not-ready N` leaves one worker in `N` not ready: a request is then
//! placed among the others, and a turn that lands on a worker that is not
//! ready is part of the cell.
//! `--ready N` leaves `N` workers of the fleet ready and every other not
//! ready: with `1`, nearly every request is placed by a pass over the pool.
//! `--overloaded true` puts every worker over its overload thresholds, each
//! with requests in flight and one with the fewest: every request is then
//! steered to that worker by the rule for a pool with no worker under its
//! thresholds.
//!
//! `--pair true` makes the cell a pair gateway's: the fleet is the cell's
//! size of prefill workers and as many decode workers, on the gRPC wire,
//! and a request is placed as a pair by the classic pair placement, its
//! two side policies the classic ones or the objects that ask the engine.
//! `--not-ready` and `--ready` then hold for each side.
//!
//! `--keys N` gives each request one of `N` routing keys in turn, in the
//! header `x-smg-routing-key`: a policy that places by a key places by it,
//! on both paths. `0`, the default, sends no key.
//!
//! `--prompts N` gives each request one of `N` prompts in turn, each of
//! 1,024 tokens: the first 32 are the same in every prompt, as a chat
//! template's head is, and the rest are the prompt's own. `1`, the default,
//! sends one prompt over and over.
//!
//! `--models N` serves `N` models, each with a fleet of its own of the
//! cell's size: a caller's requests name the models in turn, as the
//! requests of a gateway that serves many models do, so no two requests in
//! a row are for one model.
//!
//! No Criterion harness: a cell runs a fixed count of decisions, so the same
//! binary counts instructions under callgrind (two counts, one difference).
//!
//! ```text
//! cargo bench -p smg --bench decision_paths -- \
//!     --path classic --route grpc --policy round_robin --workers 64 --callers 1 --decisions 2000000
//! ```
#![expect(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::panic,
    reason = "benchmark code: a setup failure has no caller to answer, and the cell's line is its result"
)]

use std::{
    collections::HashMap,
    hint::black_box,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

use axum::http::{HeaderMap, HeaderValue};
use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerLoadResponse, WorkerStatus},
};
use smg::{
    config::{
        types::{PolicyConfig, RoutingMode},
        RouterConfig,
    },
    observability::metrics::{start_prometheus, PrometheusConfig},
    policies::PolicyRegistry,
    routers::{
        common::{
            bench_support::{
                attach_loads, policies_asking_the_engine, route_classic, route_pair, route_records,
                set_pair_sides, EngineOverRecords, Route,
            },
            snapshot::WorkerRegistry,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    load::state::{one_rank_load, TestFeed},
};

// The gateway's binary runs on its stock allocator under its own run-time
// options (`src/main.rs`). The bench declares the same one, so a path that
// allocates is timed on the allocator that ships.
smg::stock_allocator!();

const MODEL: &str = "bench-model";

/// Decisions a caller makes between two drains of the recorder.
const STRETCH: u64 = 1 << 16;

/// Decisions a caller makes before its first timed stretch.
const CALLER_WARM: u64 = 256;

/// Deterministic pseudo-random fleet state (SplitMix64).
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Cell {
    path: String,
    route: Route,
    policy: String,
    workers: usize,
    callers: usize,
    decisions: u64,
    reports: bool,
    /// One worker in this many is not ready; `0` for a fleet all ready.
    not_ready: usize,
    /// This many workers of the fleet are ready and every other is not;
    /// `0` for a fleet all ready.
    ready: usize,
    /// Every worker is over its overload thresholds.
    overloaded: bool,
    /// The models served, each with a fleet of `workers` workers.
    models: usize,
    /// A pair gateway: `workers` prefill workers and as many decode
    /// workers a model, and a request placed as a pair.
    pair: bool,
    /// The routing keys the requests carry in turn; `0` for none.
    keys: usize,
    /// The prompts the requests carry in turn.
    prompts: usize,
}

fn cell() -> Cell {
    let mut cell = Cell {
        path: "classic".to_owned(),
        route: Route::Grpc,
        policy: "round_robin".to_owned(),
        workers: 64,
        callers: 1,
        decisions: 1_000_000,
        reports: true,
        not_ready: 0,
        ready: 0,
        overloaded: false,
        models: 1,
        pair: false,
        keys: 0,
        prompts: 1,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        // Cargo's own word to a bench binary.
        if flag == "--bench" {
            continue;
        }
        let value = args
            .next()
            .unwrap_or_else(|| panic!("{flag} takes a value"));
        match flag.as_str() {
            "--path" => cell.path = value,
            "--route" => {
                cell.route = match value.as_str() {
                    "http" => Route::Http,
                    "grpc" => Route::Grpc,
                    other => panic!("no route {other}"),
                }
            }
            "--policy" => cell.policy = value,
            "--workers" => cell.workers = value.parse().unwrap(),
            "--callers" => cell.callers = value.parse().unwrap(),
            "--decisions" => cell.decisions = value.parse().unwrap(),
            "--reports" => cell.reports = value.parse().unwrap(),
            "--not-ready" => cell.not_ready = value.parse().unwrap(),
            "--ready" => cell.ready = value.parse().unwrap(),
            "--overloaded" => cell.overloaded = value.parse().unwrap(),
            "--models" => cell.models = value.parse().unwrap(),
            "--pair" => cell.pair = value.parse().unwrap(),
            "--keys" => cell.keys = value.parse().unwrap(),
            "--prompts" => cell.prompts = value.parse().unwrap(),
            other => panic!("no flag {other}"),
        }
    }
    cell
}

fn policy_config(name: &str) -> PolicyConfig {
    // The cache-aware policy has no form of one word: its thresholds are
    // the ones the command line gives it with every switch left out.
    if name == "cache_aware" {
        return PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864);
    }
    serde_json::from_value(serde_json::json!({ "type": name }))
        .unwrap_or_else(|error| panic!("no policy {name}: {error}"))
}

/// The ids of the models a cell serves: the one model's, or one for each of
/// several.
fn model_ids(models: usize) -> Vec<String> {
    match models {
        0 | 1 => vec![MODEL.to_owned()],
        _ => (0..models).map(|at| format!("{MODEL}-{at}")).collect(),
    }
}

/// `count` ready workers of `model` on `route`'s wire, with no health
/// checker behind them. The one model's workers are named `worker-<n>`, and
/// one of several models' by their model.
fn fleet(route: Route, model: &str, count: usize) -> Vec<Arc<dyn Worker>> {
    let name = if model == MODEL { "worker" } else { model };
    (0..count)
        .map(|i| {
            let (url, mode) = match route {
                Route::Http => (format!("http://{name}-{i:04}:8000"), ConnectionMode::Http),
                Route::Grpc => (format!("grpc://{name}-{i:04}:9000"), ConnectionMode::Grpc),
            };
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(model))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(mode)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

/// One side of a pair fleet of `model`: `count` ready workers of `role` on
/// the gRPC wire, with no health checker behind them.
fn side(model: &str, role: WorkerType, count: usize) -> Vec<Arc<dyn Worker>> {
    let name = if role == WorkerType::Prefill {
        "prefill"
    } else {
        "decode"
    };
    (0..count)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://{model}-{name}-{i:04}:9000"))
                    .model(ModelCard::new(model))
                    .worker_type(role)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Vllm)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

/// Leaves the last worker of each `one_in` of the fleet not ready; `0`
/// leaves the fleet as it is.
fn leave_not_ready(fleet: &[Arc<dyn Worker>], one_in: usize) {
    if one_in == 0 {
        return;
    }
    for worker in fleet.iter().skip(one_in - 1).step_by(one_in) {
        worker.set_status(WorkerStatus::NotReady);
    }
}

/// Leaves `ready` workers of the fleet ready, spread evenly over it, and
/// every other not ready; `0` leaves the fleet as it is.
fn leave_ready(fleet: &[Arc<dyn Worker>], ready: usize) {
    if ready == 0 {
        return;
    }
    for worker in fleet {
        worker.set_status(WorkerStatus::NotReady);
    }
    for at in 0..ready.min(fleet.len()) {
        fleet[at * fleet.len() / ready].set_status(WorkerStatus::Ready);
    }
}

/// Puts every worker of the fleet over its overload thresholds, each with
/// requests in flight: its place in the fleet plus one, but the worker in
/// the middle, which holds none.
fn overload(fleet: &[Arc<dyn Worker>]) {
    for (at, worker) in fleet.iter().enumerate() {
        worker.set_overloaded(true);
        if at != fleet.len() / 2 {
            for _ in 0..=at {
                worker.increment_load();
            }
        }
    }
}

/// One report a worker: queued token-work, running requests, KV pressure
/// and a live rate on three workers of four.
fn reports(fleet: &[Arc<dyn Worker>]) -> Vec<(Arc<dyn Worker>, WorkerLoadResponse)> {
    fleet
        .iter()
        .enumerate()
        .map(|(i, worker)| {
            let draw = |salt: u64| mix(i as u64 * 16 + salt);
            let mut report = one_rank_load(
                (draw(1) % 8_192) as i32,
                (draw(2) % 95) as f64 / 100.0,
                if draw(3) % 4 == 0 {
                    0.0
                } else {
                    500.0 + (draw(3) % 3_500) as f64
                },
            );
            report.loads[0].num_running_reqs = (draw(4) % 12) as i32;
            (Arc::clone(worker), report)
        })
        .collect()
}

/// A path's one routing call, shared by the callers: the request's tokens
/// and its turn among its caller's requests, which names its model and its
/// routing key.
type Routing = Box<dyn Fn(&[u32], u64) -> bool + Send + Sync>;

/// The header a routing key travels in.
const ROUTING_KEY: &str = "x-smg-routing-key";

/// The headers of `keys` requests, each with a routing key of its own.
fn keyed_headers(keys: usize) -> Vec<HeaderMap> {
    (0..keys)
        .map(|key| {
            let mut headers = HeaderMap::new();
            let value = HeaderValue::from_str(&format!("session-{key:06}")).unwrap();
            headers.insert(ROUTING_KEY, value);
            headers
        })
        .collect()
}

/// The shared head of every prompt, in tokens.
const PROMPT_HEAD: usize = 32;

/// `prompts` prompts of 1,024 tokens: a head every prompt shares, then
/// tokens of the prompt's own. One prompt is the tokens `0..1024`.
fn prompts(prompts: usize) -> Vec<Vec<u32>> {
    (0..prompts.max(1))
        .map(|prompt| {
            (0..1_024usize)
                .map(|at| {
                    if prompt == 0 || at < PROMPT_HEAD {
                        at as u32
                    } else {
                        (mix((prompt * 1_024 + at) as u64) % 100_000) as u32
                    }
                })
                .collect()
        })
        .collect()
}

/// Who decides a cell's requests: the classic placement over a registry's
/// policies (the classic ones, or the objects that ask the engine), or the
/// policy engine's whole decision over records.
enum Decider {
    Policies(Arc<PolicyRegistry>),
    Records(EngineOverRecords),
}

/// A cell's path, built: a registry with the fleet, the policies a
/// registration leaves (the model's policy and each worker's completion
/// sink), and the load feed the monitor would publish. On the `object` path
/// the registry's policies ask the policy engine of the same policy. A
/// pair cell's registry holds the two side policies a pair gateway's start
/// sets, and each model's fleet is its two sides. On the `records` path no
/// policy registry is built: the engine reads the feed itself.
fn build(cell: &Cell) -> (Routing, Option<TestFeed>) {
    let registry = Arc::new(WorkerRegistry::new());
    let mut config = RouterConfig {
        policy_controller: true,
        policy: policy_config(&cell.policy),
        ..RouterConfig::default()
    };
    if cell.pair {
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
    }
    let policies = match cell.path.as_str() {
        "classic" => {
            let policies = PolicyRegistry::new(policy_config(&cell.policy));
            if cell.pair {
                set_pair_sides(&policies, &config);
            }
            Some(Arc::new(policies))
        }
        "object" => Some(Arc::new(policies_asking_the_engine(&config).unwrap())),
        "records" => {
            assert!(!cell.pair, "the records path places single requests");
            None
        }
        other => panic!("no path {other}"),
    };
    let models = model_ids(cell.models);
    let mut every = Vec::with_capacity(models.len() * cell.workers);
    for model in &models {
        let fleets = if cell.pair {
            vec![
                side(model, WorkerType::Prefill, cell.workers),
                side(model, WorkerType::Decode, cell.workers),
            ]
        } else {
            vec![fleet(cell.route, model, cell.workers)]
        };
        for fleet in fleets {
            for worker in &fleet {
                registry.register(Arc::clone(worker)).unwrap();
                if let Some(policies) = &policies {
                    policies.on_worker_added(model, None);
                    worker.set_completion_sink(Some(policies.completion_sink()));
                }
            }
            leave_not_ready(&fleet, cell.not_ready);
            leave_ready(&fleet, cell.ready);
            if cell.overloaded {
                overload(&fleet);
            }
            every.extend(fleet);
        }
    }
    // The feed's receiver goes to whoever decides: the registry's policies,
    // or the engine.
    let mut receiver = None;
    let feed = cell.reports.then(|| {
        let feed = TestFeed::attached_to(|rx| receiver = rx);
        let loads: HashMap<String, WorkerLoadResponse> = reports(&every)
            .into_iter()
            .map(|(worker, report)| (worker.url().to_owned(), report))
            .collect();
        feed.publish(&loads);
        feed
    });
    let decider = match policies {
        Some(policies) => {
            if cell.reports {
                attach_loads(&policies, receiver);
            }
            Decider::Policies(policies)
        }
        None => Decider::Records(EngineOverRecords::of(&config, receiver).unwrap()),
    };
    let (route, pair) = (cell.route, cell.pair);
    assert!(
        !(pair && cell.keys != 0),
        "a pair cell sends no routing key"
    );
    let keys = keyed_headers(cell.keys);
    let routing: Routing = Box::new(move |tokens, turn| {
        let model = &models[(turn % models.len() as u64) as usize];
        let headers = match keys.len() {
            0 => None,
            count => keys.get((turn % count as u64) as usize),
        };
        match &decider {
            Decider::Policies(policies) if pair => {
                route_pair(&registry, policies, model, Some(tokens))
            }
            Decider::Policies(policies) => {
                route_classic(&registry, policies, model, route, Some(tokens), headers)
            }
            Decider::Records(engine) => {
                route_records(&registry, engine, model, route, Some(tokens), headers)
            }
        }
    });
    (routing, feed)
}

fn main() {
    let cell = cell();
    let recorder = start_prometheus(PrometheusConfig::default());
    let prompts = prompts(cell.prompts);
    let tokens_at = |turn: u64| &prompts[(turn % prompts.len() as u64) as usize][..];
    let (routing, _feed) = build(&cell);
    // The path's first sight of the pool, untimed: what a registration and
    // the first request after it build once.
    let warm = (cell.workers as u64 * 8).max(4_096) * cell.models.max(1) as u64;
    for turn in 0..warm {
        assert!(
            routing(tokens_at(turn), turn),
            "a fleet with a ready worker takes every request"
        );
    }
    recorder.run_upkeep();

    // The callers stand for the whole cell, as a serving process's threads
    // do: each makes its first decisions untimed (the buffers a path keeps
    // a thread, its metric handles), then one stretch at each start. A
    // stretch of zero decisions ends them.
    let stretch = AtomicU64::new(0);
    let start = Barrier::new(cell.callers + 1);
    let done = Barrier::new(cell.callers + 1);
    let busy_nanos = AtomicU64::new(0);
    let placed = AtomicU64::new(0);
    let mut wall = Duration::ZERO;
    thread::scope(|scope| {
        for _ in 0..cell.callers {
            scope.spawn(|| {
                for turn in 0..CALLER_WARM * cell.models.max(1) as u64 {
                    black_box(routing(black_box(tokens_at(turn)), turn));
                }
                loop {
                    start.wait();
                    let decisions = stretch.load(Ordering::Relaxed);
                    if decisions == 0 {
                        break;
                    }
                    let started = Instant::now();
                    let mut took_placed = 0u64;
                    for turn in 0..decisions {
                        took_placed += u64::from(black_box(routing(
                            black_box(tokens_at(turn)),
                            black_box(turn),
                        )));
                    }
                    let took = started.elapsed();
                    busy_nanos.fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
                    placed.fetch_add(took_placed, Ordering::Relaxed);
                    done.wait();
                }
            });
        }
        let mut left = cell.decisions;
        while left > 0 {
            let decisions = left.min(STRETCH);
            stretch.store(decisions, Ordering::Relaxed);
            start.wait();
            let started = Instant::now();
            done.wait();
            wall += started.elapsed();
            recorder.run_upkeep();
            left -= decisions;
        }
        stretch.store(0, Ordering::Relaxed);
        start.wait();
    });
    let busy = Duration::from_nanos(busy_nanos.into_inner());
    let placed = placed.into_inner();
    let total = cell.decisions * cell.callers as u64;
    assert_eq!(
        placed, total,
        "a fleet with a ready worker takes every request"
    );
    let per_second = total as f64 / wall.as_secs_f64();
    let nanos = busy.as_nanos() as f64 / total as f64;
    println!(
        "path={} route={:?} policy={} workers={} callers={} reports={} not_ready={} ready={} overloaded={} models={} pair={} keys={} prompts={} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
        cell.path,
        cell.route,
        cell.policy,
        cell.workers,
        cell.callers,
        cell.reports,
        cell.not_ready,
        cell.ready,
        cell.overloaded,
        cell.models,
        cell.pair,
        cell.keys,
        cell.prompts,
        total,
        wall.as_secs_f64(),
        per_second,
        nanos
    );
}
