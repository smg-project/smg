//! One request's routing, head to head, over one fleet of workers of one
//! model, every one ready unless the cell says otherwise. Three paths:
//!
//! - `classic`: the classic policies over the worker registry.
//! - `engine`: the policy engine over that same registry. Everything is
//!   built as `classic` builds it (the registry, the policies a
//!   registration leaves, each worker's completion sink, the load feed) and
//!   the request runs the same function; the one difference is the engine
//!   set to the placement.
//! - `controller`: the policy controller over its own store in this
//!   process, registered and reported to as that mode does it.
//!
//! A cell is one path, one route, one policy, one fleet size and one count
//! of callers. Each caller routes requests back to back: the placement, the
//! worker's load guard and what the path says of the send and of the end.
//! The cell prints its decisions a second over every caller and one caller's
//! time a decision. The gateway's metrics recorder runs, as in a serving
//! process, and is drained between the timed stretches. The binary runs on
//! the gateway's stock allocator, as the serving binary does.
//!
//! `--not-ready N` leaves one worker in `N` not ready, on the `classic` and
//! `engine` paths: a request is then placed among the others, and a turn or
//! a draw that lands on a worker that is not ready is part of the cell.
//!
//! `--draw-from N` turns the pool size from which the engine's decision
//! draws its worker first, for the run that sets that size. The binary
//! reads the size from a word the flag writes, one load a decision on the
//! `engine` path, where a product build reads a constant.
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
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerLoadResponse, WorkerStatus},
};
use smg::{
    config::{types::PolicyConfig, RouterConfig},
    observability::metrics::{start_prometheus, PrometheusConfig},
    policies::PolicyRegistry,
    routers::{
        common::{
            bench_support::{
                attach_loads, host_policy_engine, route_classic, ControllerPath, Route,
            },
            snapshot::WorkerRegistry,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};
use smg_policy_controller::engine::turn_live_draw_from;
use smg_worker::{
    identity::basic::{ConnectionMode, Worker, WorkerType},
    load::state::{one_rank_load, TestFeed},
};

// The gateway's binary runs on its stock allocator under its own run-time
// options (`src/main.rs`). The bench declares the same one, so a path that
// allocates is timed on the allocator that ships.
smg::stock_allocator!();

const MODEL: &str = "bench-model";

/// Decisions a caller makes between two drains of the recorder.
const STRETCH: u64 = 1 << 16;

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
    draw_from: Option<usize>,
    /// One worker in this many is not ready; `0` for a fleet all ready.
    not_ready: usize,
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
        draw_from: None,
        not_ready: 0,
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
            "--draw-from" => cell.draw_from = Some(value.parse().unwrap()),
            "--not-ready" => cell.not_ready = value.parse().unwrap(),
            other => panic!("no flag {other}"),
        }
    }
    cell
}

fn policy_config(name: &str) -> PolicyConfig {
    serde_json::from_value(serde_json::json!({ "type": name }))
        .unwrap_or_else(|error| panic!("no policy {name}: {error}"))
}

/// `count` ready workers of one model on `route`'s wire, with no health
/// checker behind them.
fn fleet(route: Route, count: usize) -> Vec<Arc<dyn Worker>> {
    (0..count)
        .map(|i| {
            let (url, mode) = match route {
                Route::Http => (format!("http://worker-{i:04}:8000"), ConnectionMode::Http),
                Route::Grpc => (format!("grpc://worker-{i:04}:9000"), ConnectionMode::Grpc),
            };
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(MODEL))
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

/// A path's one routing call, shared by the callers.
type Routing = Box<dyn Fn(&[u32]) -> bool + Send + Sync>;

/// What a path's timer does about once a second in a serving process.
type Upkeep = Box<dyn Fn()>;

/// The classic path: a registry with the fleet, the policies a registration
/// leaves (the model's policy and each worker's completion sink), and the
/// load feed the monitor would publish. With `engine`, the same, and the
/// policy engine of the same policy set to the placement.
fn classic(cell: &Cell, engine: bool) -> (Routing, Option<TestFeed>) {
    let registry = Arc::new(WorkerRegistry::new());
    let policies = Arc::new(PolicyRegistry::new(policy_config(&cell.policy)));
    if engine {
        let config = RouterConfig {
            policy_controller: true,
            policy: policy_config(&cell.policy),
            ..RouterConfig::default()
        };
        host_policy_engine(&policies, &config).unwrap();
    }
    let fleet = fleet(cell.route, cell.workers);
    for worker in &fleet {
        registry.register(Arc::clone(worker)).unwrap();
        policies.on_worker_added(MODEL, None);
        worker.set_completion_sink(Some(policies.completion_sink()));
    }
    leave_not_ready(&fleet, cell.not_ready);
    let feed = cell.reports.then(|| {
        let feed = TestFeed::attached_to(|rx| attach_loads(&policies, rx));
        let loads: HashMap<String, WorkerLoadResponse> = reports(&fleet)
            .into_iter()
            .map(|(worker, report)| (worker.url().to_owned(), report))
            .collect();
        feed.publish(&loads);
        feed
    });
    let route = cell.route;
    let routing: Routing =
        Box::new(move |tokens| route_classic(&registry, &policies, MODEL, route, Some(tokens)));
    (routing, feed)
}

/// The controller's path: the store, the feed and the host a gateway builds
/// under the setting, the fleet registered through the feed's door, and the
/// reports said through the feed's load sink.
fn controller(cell: &Cell) -> (Routing, Upkeep) {
    let config = RouterConfig {
        policy_controller: true,
        policy: policy_config(&cell.policy),
        ..RouterConfig::default()
    };
    assert_eq!(
        cell.not_ready, 0,
        "--not-ready is for the classic and engine paths"
    );
    let path = Arc::new(ControllerPath::start(&config, MODEL).unwrap());
    let fleet = fleet(cell.route, cell.workers);
    path.register(&fleet).unwrap();
    assert_eq!(path.registry().len(), cell.workers, "every worker stands");
    if cell.reports {
        path.report_loads(&reports(&fleet));
    }
    let route = cell.route;
    let swept = Arc::clone(&path);
    (
        Box::new(move |tokens| path.route(route, Some(tokens))),
        Box::new(move || swept.upkeep()),
    )
}

fn main() {
    let cell = cell();
    let recorder = start_prometheus(PrometheusConfig::default());
    let tokens: Vec<u32> = (0..1_024).collect();
    if let Some(rows) = cell.draw_from {
        turn_live_draw_from(rows);
    }
    let (routing, upkeep, _feed): (Routing, Upkeep, Option<TestFeed>) = match cell.path.as_str() {
        "classic" | "engine" => {
            let (routing, feed) = classic(&cell, cell.path == "engine");
            (routing, Box::new(|| {}), feed)
        }
        "controller" => {
            let (routing, upkeep) = controller(&cell);
            (routing, upkeep, None)
        }
        other => panic!("no path {other}"),
    };
    // Every caller's first decisions, untimed: the buffers a path keeps a
    // thread, the metric handles and the policies' first sight of the pool.
    let warm = (cell.workers as u64 * 8).max(4_096);
    for _ in 0..warm {
        assert!(
            routing(&tokens),
            "a fleet with a ready worker takes every request"
        );
    }
    recorder.run_upkeep();
    upkeep();

    let mut left = cell.decisions;
    let mut wall = Duration::ZERO;
    let mut busy = Duration::ZERO;
    let mut placed = 0u64;
    while left > 0 {
        let stretch = left.min(STRETCH);
        let start = Barrier::new(cell.callers + 1);
        let (stretch_wall, stretch_busy, stretch_placed) = thread::scope(|scope| {
            let callers: Vec<_> = (0..cell.callers)
                .map(|_| {
                    let (routing, tokens, start) = (&routing, &tokens, &start);
                    scope.spawn(move || {
                        start.wait();
                        let started = Instant::now();
                        let mut placed = 0u64;
                        for _ in 0..stretch {
                            placed += u64::from(black_box(routing(black_box(tokens))));
                        }
                        (started.elapsed(), placed)
                    })
                })
                .collect();
            start.wait();
            let started = Instant::now();
            let mut busy = Duration::ZERO;
            let mut placed = 0;
            for caller in callers {
                let (took, took_placed) = caller.join().unwrap();
                busy += took;
                placed += took_placed;
            }
            (started.elapsed(), busy, placed)
        });
        wall += stretch_wall;
        busy += stretch_busy;
        placed += stretch_placed;
        recorder.run_upkeep();
        upkeep();
        left -= stretch;
    }
    let total = cell.decisions * cell.callers as u64;
    assert_eq!(
        placed, total,
        "a fleet with a ready worker takes every request"
    );
    let per_second = total as f64 / wall.as_secs_f64();
    let nanos = busy.as_nanos() as f64 / total as f64;
    println!(
        "path={} route={:?} policy={} workers={} callers={} reports={} not_ready={} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
        cell.path,
        cell.route,
        cell.policy,
        cell.workers,
        cell.callers,
        cell.reports,
        cell.not_ready,
        total,
        wall.as_secs_f64(),
        per_second,
        nanos
    );
}
