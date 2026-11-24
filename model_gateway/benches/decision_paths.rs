//! One request's routing under the classic policies or under the policy
//! engine, over one fleet of workers a model, every one ready unless the
//! cell says otherwise.
//!
//! `--path classic`, the default, routes by the classic policies over the
//! worker registry. `--path engine` routes the same cell by the policy
//! engine, over what a gateway builds with the policy controller on: a
//! registry that makes no classic policy object, workers with no
//! completion sink, and the engine of the document the cell's routing
//! settings mean, started by the gateway's own start over the cell's
//! registry and, where the cell publishes reports, its load feed. The
//! same routing calls then place by it. The engine's task is spawned and
//! not waited for, as at a gateway's start: a cell's first decisions,
//! which are untimed, may run ahead of its first step. The bench keeps no
//! cache-event monitor, so the engine is handed no index of cache events.
//! Where the document reads the prefixes sent to each worker, as
//! `cache_aware`'s does, the start lends the engine the gateway's token
//! trees.
//! The paths `object` and `records` are refused by name: they are no
//! longer part of this bench.
//!
//! A cell is one route, one policy, one fleet size and one count of
//! callers. Each caller routes requests back to back: the placement, the
//! worker's load guard, and what the request's end tells the model's
//! policy or the engine.
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
//! and a request is placed as a pair: by the classic pair placement over
//! its two side policies, or by the engine, whose document plans the pair
//! alone. `--not-ready` and `--ready` then hold for each side.
//!
//! `--keys N` gives each request one of `N` routing keys in turn, in the
//! header `x-smg-routing-key`: a policy that places by a key places by it.
//! `0`, the default, sends no key.
//!
//! `--prompts N` gives each request one of `N` prompts in turn, each of
//! 1,024 tokens: the first 32 are the same in every prompt, as a chat
//! template's head is, and the rest are the prompt's own. `1`, the default,
//! sends one prompt over and over.
//!
//! `--grow N` makes each caller hold a conversation: every request carries
//! the conversation so far and `N` tokens more, from 1,024 tokens up to
//! 2,048, and then the caller begins another behind the head every prompt
//! shares. `0`, the default, sends the prompts in turn. The result line
//! then counts the turns that carried a conversation on (`turns`) and those
//! placed on the worker the turn before went to (`returned`), with their
//! share (`return_rate`): what a policy that places by the prompt's prefix
//! holds a conversation to. A policy with no memory of the prompt returns
//! by chance, one turn in the fleet's size, and by its load reports where
//! the cell publishes them: a cell read for this column also runs with
//! `--reports false`.
//!
//! `--publish-ms N` publishes the fleet's reports again, as a new load
//! snapshot, every `N` milliseconds while the callers run: the load monitor
//! publishes pushed reports at most once in 20 milliseconds. `0`, the
//! default, publishes once before the first request. The result line counts
//! the snapshots published while the callers ran (`publications`).
//!
//! The result line's `alloc_bytes_per_decision` is what the callers'
//! threads allocated in their timed decisions, by the allocator's own count
//! of each thread, over the decisions made.
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
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
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
    app_context::AppContextBuilder,
    config::{
        types::{PolicyConfig, RoutingMode},
        RouterConfig,
    },
    observability::metrics::{start_prometheus, PrometheusConfig},
    policies::PolicyRegistry,
    routers::{
        common::{
            bench_support::{
                attach_loads, route_classic, route_pair, set_pair_sides, thread_allocated_bytes,
                Route,
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
    /// Milliseconds between two publications of the fleet's reports while
    /// the callers run; `0` for the one publication before the first
    /// request.
    publish_ms: u64,
    /// The tokens a request adds to its caller's conversation; `0` for the
    /// prompts in turn.
    grow: usize,
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
        publish_ms: 0,
        grow: 0,
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
            "--publish-ms" => cell.publish_ms = value.parse().unwrap(),
            "--grow" => cell.grow = value.parse().unwrap(),
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
    // Nor has the bucket policy: the command line's two spill thresholds
    // and its seconds between two adjustments of the boundaries.
    if name == "bucket" {
        return PolicyConfig::Bucket {
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        };
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
/// routing key. It hands back the worker the request went to, as a number
/// a caller compares with its last request's, and `None` where the request
/// had no worker.
type Routing = Box<dyn Fn(&[u32], u64) -> Option<usize> + Send + Sync>;

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

/// A caller's conversation under `--grow`: every request carries the
/// conversation so far and the tokens its turn adds. Past its full length
/// the caller begins a new conversation behind the head every prompt shares.
struct Conversation {
    tokens: Vec<u32>,
    len: usize,
    /// The last token made: each is one more, from a start of the caller's
    /// own, so no two callers' conversations hold the same block.
    last: u32,
    grow: usize,
}

impl Conversation {
    /// A conversation's tokens at its first request, and at its last.
    const BEGINS: usize = 1_024;
    const FULL: usize = 2_048;

    fn new(caller: u32, grow: usize) -> Self {
        Self {
            tokens: vec![0; Self::FULL],
            len: 0,
            last: (caller + 1) << 22,
            grow,
        }
    }

    /// The next request's tokens, and whether the request carries the
    /// caller's last one on: `false` for a conversation's first request.
    fn turn(&mut self) -> (&[u32], bool) {
        let begins = self.len == 0 || self.len + self.grow > Self::FULL;
        let (from, to) = if begins {
            for (at, token) in self.tokens.iter_mut().enumerate().take(PROMPT_HEAD) {
                *token = at as u32;
            }
            (PROMPT_HEAD, Self::BEGINS)
        } else {
            (self.len, self.len + self.grow)
        };
        for token in &mut self.tokens[from..to] {
            self.last = self.last.wrapping_add(1);
            *token = self.last;
        }
        self.len = to;
        (&self.tokens[..to], !begins)
    }
}

/// The load feed of a cell and the reports it publishes: one a worker.
struct Feed {
    feed: TestFeed,
    loads: HashMap<String, WorkerLoadResponse>,
}

/// A cell's path as `build` leaves it: the routing call, the load feed
/// where the cell publishes reports, and the engine's runtime on its path.
struct Built {
    routing: Routing,
    feed: Option<Feed>,
    /// The runtime the policy engine's task runs on, kept for the cell's
    /// life: `None` on the classic path, which starts no engine.
    runtime: Option<tokio::runtime::Runtime>,
}

/// A cell's path, built: a registry with the fleet, what a registration
/// leaves (under the classic policies the model's policy and each worker's
/// completion sink, under the engine neither), and the load feed the
/// monitor would publish. A classic pair cell's registry holds the two
/// side policies a pair gateway's start sets, and each model's fleet is
/// its two sides. On the engine's path the engine is started last.
fn build(cell: &Cell) -> Built {
    let registry = Arc::new(WorkerRegistry::new());
    let mut config = RouterConfig {
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
        // The cell's workers are on the gRPC wire, and its settings say so:
        // a pair gateway of one router over gRPC plans the pair alone, where
        // one over HTTP plans single requests beside it. Only the engine's
        // document reads this.
        config.connection_mode = ConnectionMode::Grpc;
    }
    let policies = match cell.path.as_str() {
        "classic" => {
            let policies = PolicyRegistry::new(policy_config(&cell.policy));
            if cell.pair {
                set_pair_sides(&policies, &config);
            }
            Arc::new(policies)
        }
        "engine" => {
            // The registry a gateway builds with the policy controller on:
            // it makes no classic policy object, and a pair's two side
            // policies are not set on it.
            config.policy_controller = true;
            Arc::new(PolicyRegistry::of_config(&config))
        }
        gone @ ("object" | "records") => {
            panic!("the path {gone} was removed from this bench: it runs the classic and the engine path")
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
                policies.on_worker_added(model, None);
                // The sink a registration gives a worker: the registry's own
                // under the classic policies, none where the engine places.
                worker.set_completion_sink(policies.worker_completion_sink());
            }
            leave_not_ready(&fleet, cell.not_ready);
            leave_ready(&fleet, cell.ready);
            if cell.overloaded {
                overload(&fleet);
            }
            every.extend(fleet);
        }
    }
    // The feed's receiver goes to the registry's policies, and to the engine
    // on its path.
    let mut receiver = None;
    let feed = cell.reports.then(|| {
        let feed = TestFeed::attached_to(|rx| receiver = rx);
        let loads: HashMap<String, WorkerLoadResponse> = reports(&every)
            .into_iter()
            .map(|(worker, report)| (worker.url().to_owned(), report))
            .collect();
        feed.publish(&loads);
        Feed { feed, loads }
    });
    let engine_loads = receiver.clone();
    if cell.reports {
        attach_loads(&policies, receiver);
    }
    // The engine path: the gateway's own start, with the policy controller
    // on in the cell's settings, behind the fleet's registration and, where
    // the cell publishes reports, the feed's first publication. The engine's
    // task runs on a runtime of its own thread, kept for the cell's life,
    // and its first step is not waited for.
    let runtime = (cell.path == "engine").then(|| {
        config.policy_controller = true;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        {
            let _entered = runtime.enter();
            AppContextBuilder::start_policy_engine_for_bench(
                &config,
                &policies,
                &registry,
                engine_loads,
            )
            .unwrap();
        }
        runtime
    });
    let (route, pair) = (cell.route, cell.pair);
    assert!(
        !(pair && cell.keys != 0),
        "a pair cell sends no routing key"
    );
    assert!(
        !(cell.grow > 0 && cell.models > 1),
        "a growing conversation is one model's"
    );
    let keys = keyed_headers(cell.keys);
    let routing: Routing = Box::new(move |tokens, turn| {
        let model = &models[(turn % models.len() as u64) as usize];
        let headers = match keys.len() {
            0 => None,
            count => keys.get((turn % count as u64) as usize),
        };
        if pair {
            route_pair(&registry, &policies, model, Some(tokens))
        } else {
            route_classic(&registry, &policies, model, route, Some(tokens), headers)
        }
    });
    Built {
        routing,
        feed,
        runtime,
    }
}

fn main() {
    let cell = cell();
    let recorder = start_prometheus(PrometheusConfig::default());
    let prompts = prompts(cell.prompts);
    let tokens_at = |turn: u64| &prompts[(turn % prompts.len() as u64) as usize][..];
    let Built {
        routing,
        feed,
        runtime: _engine_runtime,
    } = build(&cell);
    // The path's first sight of the pool, untimed: what a registration and
    // the first request after it build once.
    let warm = (cell.workers as u64 * 8).max(4_096) * cell.models.max(1) as u64;
    for turn in 0..warm {
        assert!(
            routing(tokens_at(turn), turn).is_some(),
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
    let allocated = AtomicU64::new(0);
    let publications = AtomicU64::new(0);
    let published = AtomicBool::new(false);
    let callers = AtomicU32::new(0);
    let (turns, returned) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut wall = Duration::ZERO;
    thread::scope(|scope| {
        // The monitor's side: the fleet's reports published again as a new
        // snapshot, each sampled at its publication.
        if let (Some(feed), true) = (&feed, cell.publish_ms > 0) {
            let (publications, published) = (&publications, &published);
            let every = Duration::from_millis(cell.publish_ms);
            scope.spawn(move || {
                while !published.load(Ordering::Relaxed) {
                    thread::sleep(every);
                    feed.feed.publish(&feed.loads);
                    publications.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        for _ in 0..cell.callers {
            scope.spawn(|| {
                // A caller's own conversation, where the cell's requests
                // grow, and the worker its last request went to.
                let caller = callers.fetch_add(1, Ordering::Relaxed);
                let mut conversation =
                    (cell.grow > 0).then(|| Conversation::new(caller, cell.grow));
                let mut last = None;
                for turn in 0..CALLER_WARM * cell.models.max(1) as u64 {
                    black_box(routing(black_box(tokens_at(turn)), turn));
                }
                loop {
                    start.wait();
                    let decisions = stretch.load(Ordering::Relaxed);
                    if decisions == 0 {
                        break;
                    }
                    let allocated_before = thread_allocated_bytes();
                    let started = Instant::now();
                    let mut took_placed = 0u64;
                    if let Some(conversation) = &mut conversation {
                        let (mut carried, mut back) = (0u64, 0u64);
                        for turn in 0..decisions {
                            let (tokens, carries) = conversation.turn();
                            let went = black_box(routing(black_box(tokens), black_box(turn)));
                            took_placed += u64::from(went.is_some());
                            carried += u64::from(carries);
                            back += u64::from(carries && went == last);
                            last = went;
                        }
                        turns.fetch_add(carried, Ordering::Relaxed);
                        returned.fetch_add(back, Ordering::Relaxed);
                    } else {
                        for turn in 0..decisions {
                            let went = routing(black_box(tokens_at(turn)), black_box(turn));
                            took_placed += u64::from(black_box(went).is_some());
                        }
                    }
                    let took = started.elapsed();
                    if let (Some(before), Some(after)) =
                        (allocated_before, thread_allocated_bytes())
                    {
                        allocated.fetch_add(after - before, Ordering::Relaxed);
                    }
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
        published.store(true, Ordering::Relaxed);
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
    let allocated_each = allocated.into_inner() as f64 / total as f64;
    let (turns, returned) = (turns.into_inner(), returned.into_inner());
    let return_rate = if turns == 0 {
        0.0
    } else {
        returned as f64 / turns as f64
    };
    println!(
        "path={} route={:?} policy={} workers={} callers={} reports={} not_ready={} ready={} overloaded={} models={} pair={} keys={} prompts={} grow={} turns={} returned={} return_rate={:.4} publish_ms={} publications={} alloc_bytes_per_decision={:.2} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
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
        cell.grow,
        turns,
        returned,
        return_rate,
        cell.publish_ms,
        publications.into_inner(),
        allocated_each,
        total,
        wall.as_secs_f64(),
        per_second,
        nanos
    );
}
