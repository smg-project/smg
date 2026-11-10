//! The policy engine's seam under the gateway's prefill admission: the
//! decisions per second and their latency under 64 concurrent admissions
//! through the real prefill-admission FIFO (head-only selection, a per-worker
//! capacity window), for three shapes:
//!
//! - `legacy_under_lock`: today's path, the policy over worker handles inside
//!   the admission callback (under the lock);
//! - `engine_under_lock`: the engine's view, decision and commit inside the
//!   same callback;
//! - `engine_decided`: the decision outside the lock, the commit under it
//!   (`PrefillAdmission::admit_decided`).
//!
//! Sections (`PD_SECTIONS`, default both): `fifo` (a window that never binds:
//! the lock hold alone) and `window` (a binding window: queueing and
//! commit misses). The pair and decode-room sections arrive with the pair seam.
//! Fleets from `PD_FLEETS` (default 8,64,512), `PD_TASKS` tasks (64) on eight
//! runtime threads, `PD_PER_TASK` admissions each (2000). Output: TSV rows on
//! stdout: `section  shape  policy  workers  p50_ns  p99_ns  per_s  misses
//! rejects`.
#![expect(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmark code: panicking on setup failure is expected, println is the TSV report, eprintln the progress"
)]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use openai_protocol::worker::{HealthCheckConfig, WorkerLoadResponse};
use smg::{
    config::PolicyConfig,
    policies::{
        engine_host::HostRequest, Decider, LeastLoadPolicy, LoadBalancingPolicy, RoundRobinPolicy,
        SelectWorkerInfo, WorkerRole,
    },
    worker::{
        acquire_prefill, acquire_prefill_decided,
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, CapacityWindow, PrefillAdmission, PrefillCandidateError,
        PrefillSelectionContext, Worker, WorkerType,
    },
};
use smg_policy_controller::{Decision, Lease, Phase, Side, Target, WorkerId};

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn mix(mut v: u64) -> u64 {
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    v ^ (v >> 31)
}

// ───────────────────────── fleets ─────────────────────────

fn pool(n: usize, kind: WorkerType, prefix: &str) -> Vec<Arc<dyn Worker>> {
    (0..n)
        .map(|i| {
            let builder = BasicWorkerBuilder::new(format!("http://{prefix}-{i:03}:8000"))
                .worker_type(kind)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                });
            Arc::new(builder.build()) as Arc<dyn Worker>
        })
        .collect()
}

fn reports_for(pool: &[Arc<dyn Worker>]) -> HashMap<String, WorkerLoadResponse> {
    pool.iter()
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
            (worker.url().to_string(), report)
        })
        .collect()
}

// ───────────────────────── measurement ─────────────────────────

#[derive(Default)]
struct Counters {
    misses: AtomicU64,
    rejects: AtomicU64,
}

struct Row {
    p50: u64,
    p99: u64,
    per_s: f64,
}

fn summarize(mut lat: Vec<u64>, total: Duration) -> Row {
    lat.sort_unstable();
    let pct = |p: f64| lat[((lat.len() as f64 - 1.0) * p) as usize];
    Row {
        p50: pct(0.5),
        p99: pct(0.99),
        per_s: lat.len() as f64 / total.as_secs_f64(),
    }
}

fn print_row(section: &str, shape: &str, policy: &str, workers: usize, row: &Row, c: &Counters) {
    println!(
        "{section}\t{shape}\t{policy}\t{workers}\t{}\t{}\t{:.0}\t{}\t{}",
        row.p50,
        row.p99,
        row.per_s,
        c.misses.load(Ordering::Relaxed),
        c.rejects.load(Ordering::Relaxed)
    );
}

/// Run `op` from `tasks` tasks `per_task` times each on an eight-thread
/// runtime; the latency of every call, the wall time of the whole.
#[expect(
    clippy::disallowed_methods,
    reason = "benchmark tasks are joined before the cell ends"
)]
fn run_concurrent<F, Fut>(tasks: usize, per_task: usize, op: F) -> Row
where
    F: Fn(usize) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let op = Arc::new(op);
    runtime.block_on(async {
        let start = Instant::now();
        let handles: Vec<_> = (0..tasks)
            .map(|t| {
                let op = Arc::clone(&op);
                tokio::spawn(async move {
                    let mut lat = Vec::with_capacity(per_task);
                    for k in 0..per_task {
                        let t0 = Instant::now();
                        op(t * per_task + k).await;
                        lat.push(t0.elapsed().as_nanos() as u64);
                    }
                    lat
                })
            })
            .collect();
        let mut all = Vec::with_capacity(tasks * per_task);
        for handle in handles {
            all.extend(handle.await.unwrap());
        }
        summarize(all, start.elapsed())
    })
}

#[derive(Debug)]
enum Verdict {
    AtCapacity,
    Unavailable,
}

impl PrefillCandidateError for Verdict {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Verdict::AtCapacity)
    }
}

struct Policies {
    name: &'static str,
    legacy: Arc<dyn LoadBalancingPolicy>,
    decider: Arc<Decider>,
    _feed: Option<TestFeed>,
}

/// The legacy policy and the engine under one configuration; under
/// `least_load` both read one load feed carrying a report per worker.
fn policies(name: &str, pool: &[Arc<dyn Worker>]) -> Policies {
    let (legacy, config): (Arc<dyn LoadBalancingPolicy>, PolicyConfig) = match name {
        "least_load" => (
            Arc::new(LeastLoadPolicy::new()),
            PolicyConfig::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
        ),
        _ => (Arc::new(RoundRobinPolicy::new()), PolicyConfig::RoundRobin),
    };
    let decider = Arc::new(Decider::from_configs(&config, &config, &config).unwrap());
    let feed = (name == "least_load").then(|| {
        let feed = TestFeed::attached_to(legacy.as_ref());
        feed.publish(&reports_for(pool));
        decider.set_load_receiver(Some(feed.receiver()));
        feed
    });
    Policies {
        name: match name {
            "least_load" => "least_load",
            _ => "round_robin",
        },
        legacy,
        decider,
        _feed: feed,
    }
}

// ───────────────────────── the prefill side under the FIFO ─────────────────────────

/// Today's selection at the queue head: the capacity pre-filter, then the
/// policy over the open handles.
fn legacy_prefill(
    policy: &dyn LoadBalancingPolicy,
    pool: &[Arc<dyn Worker>],
    capacity: Option<&PrefillSelectionContext<'_>>,
) -> Result<Arc<dyn Worker>, Verdict> {
    let open: Vec<Arc<dyn Worker>> = pool
        .iter()
        .filter(|w| capacity.is_none_or(|c| c.has_capacity(w)))
        .cloned()
        .collect();
    if open.is_empty() {
        return Err(Verdict::AtCapacity);
    }
    let info = SelectWorkerInfo {
        role: WorkerRole::Prefill,
        ..SelectWorkerInfo::default()
    };
    let idx = policy
        .select_worker(&open, &info)
        .ok_or(Verdict::Unavailable)?;
    Ok(Arc::clone(&open[idx]))
}

/// The engine's decision over the prefill pool with the window bits:
/// the prefill profile filters on capacity before it scores.
fn engine_prefill_decision(
    decider: &Decider,
    pool: &[Arc<dyn Worker>],
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    excluded: &[WorkerId],
) -> Result<(Decision, Lease), Verdict> {
    let request = HostRequest {
        model: "m",
        // The legacy arm prices a request of unknown size at the mean prefill
        // (1,024 tokens); the engine credits what the request says, so both
        // arms carry the same token-work per selection.
        prompt_tokens: 1024,
        phase: Phase::Prefill,
        has_capacity: Some(has_capacity),
        ..HostRequest::default()
    };
    let (decision, lease) = decider.decide(&request, pool, excluded);
    match decision {
        Decision::Pick(_) => Ok((decision, lease)),
        Decision::Shed(shed) if shed.emptied_by == "has-capacity" => Err(Verdict::AtCapacity),
        Decision::Shed(_) | Decision::Queue(_) => Err(Verdict::Unavailable),
    }
}

/// The commit of a decision under the window as the lock reads it: a miss
/// records the refused targets for the re-decision and counts.
fn engine_prefill_commit(
    decider: &Decider,
    pool: &[Arc<dyn Worker>],
    (decision, lease): (Decision, Lease),
    capacity: Option<&PrefillSelectionContext<'_>>,
    counters: &Counters,
    missed: &std::sync::Mutex<Vec<WorkerId>>,
) -> Result<Arc<dyn Worker>, Verdict> {
    match decider.commit(&lease, &decision, Side::Prefill, |t: &Target| {
        capacity.is_none_or(|c| c.has_capacity(&pool[t.position as usize]))
    }) {
        Some(target) => Ok(Arc::clone(&pool[target.position as usize])),
        None => {
            counters.misses.fetch_add(1, Ordering::Relaxed);
            if let Some(pick) = decision.pick() {
                missed
                    .lock()
                    .unwrap()
                    .extend(pick.targets.iter().map(|t| t.id));
            }
            Err(Verdict::AtCapacity)
        }
    }
}

fn section_prefill(
    section: &str,
    fleets: &[usize],
    tasks: usize,
    per_task: usize,
    window: usize,
    names: &[String],
) {
    for &n in fleets {
        let prefills = pool(n, WorkerType::Prefill, "prefill");
        for name in names {
            let p = Arc::new(policies(name, &prefills));
            let queue = tasks.max(1);
            for shape in ["legacy_under_lock", "engine_under_lock", "engine_decided"] {
                let admission = Arc::new(PrefillAdmission::new(
                    window,
                    queue,
                    Duration::from_secs(30),
                ));
                let counters = Arc::new(Counters::default());
                let prefills = prefills.clone();
                let p = Arc::clone(&p);
                let policy_name = p.name;
                let c = Arc::clone(&counters);
                let row = match shape {
                    "legacy_under_lock" => run_concurrent(tasks, per_task, move |_| {
                        let admission = Arc::clone(&admission);
                        let prefills = prefills.clone();
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let result = acquire_prefill(
                                Some(&admission),
                                None,
                                |w: &Arc<dyn Worker>| w,
                                |capacity| legacy_prefill(p.legacy.as_ref(), &prefills, capacity),
                            )
                            .await;
                            match result {
                                Ok((_, guard)) => drop(guard),
                                Err(_) => {
                                    c.rejects.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }),
                    "engine_under_lock" => run_concurrent(tasks, per_task, move |_| {
                        let admission = Arc::clone(&admission);
                        let prefills = prefills.clone();
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let missed: std::sync::Mutex<Vec<WorkerId>> =
                                std::sync::Mutex::new(Vec::new());
                            let result = acquire_prefill(
                                Some(&admission),
                                None,
                                |w: &Arc<dyn Worker>| w,
                                |capacity| {
                                    let has = |w: &Arc<dyn Worker>| {
                                        capacity.is_none_or(|c| c.has_capacity(w))
                                    };
                                    let proposal = engine_prefill_decision(
                                        &p.decider,
                                        &prefills,
                                        &has,
                                        &missed.lock().unwrap().clone(),
                                    )?;
                                    engine_prefill_commit(
                                        &p.decider, &prefills, proposal, capacity, &c, &missed,
                                    )
                                },
                            )
                            .await;
                            match result {
                                Ok((_, guard)) => drop(guard),
                                Err(_) => {
                                    c.rejects.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }),
                    _ => run_concurrent(tasks, per_task, move |_| {
                        let admission = Arc::clone(&admission);
                        let prefills = prefills.clone();
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let missed: std::sync::Mutex<Vec<WorkerId>> =
                                std::sync::Mutex::new(Vec::new());
                            let result = acquire_prefill_decided(
                                Some(&admission),
                                None,
                                |w: &Arc<dyn Worker>| w,
                                // the pool is the model's healthy prefills
                                || Ok(()),
                                |window: &CapacityWindow| {
                                    let has = |w: &Arc<dyn Worker>| window.has_capacity(w);
                                    engine_prefill_decision(
                                        &p.decider,
                                        &prefills,
                                        &has,
                                        &missed.lock().unwrap().clone(),
                                    )
                                },
                                |capacity, proposal| {
                                    engine_prefill_commit(
                                        &p.decider, &prefills, proposal, capacity, &c, &missed,
                                    )
                                },
                            )
                            .await;
                            match result {
                                Ok((_, guard)) => drop(guard),
                                Err(_) => {
                                    c.rejects.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }),
                };
                print_row(section, shape, policy_name, n, &row, &counters);
            }
        }
    }
}

fn main() {
    let fleets: Vec<usize> = env_list("PD_FLEETS", "8,64,512")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let sections = env_list("PD_SECTIONS", "fifo,window");
    let names = env_list("PD_POLICIES", "round_robin,least_load");
    let tasks: usize = env_num("PD_TASKS", 64);
    let per_task: usize = env_num("PD_PER_TASK", 2_000);
    let window: usize = env_num("PD_WINDOW", 4_096);
    let binding: usize = env_num("PD_WINDOW_BINDING", 1);
    println!("section\tshape\tpolicy\tworkers\tp50_ns\tp99_ns\tper_s\tmisses\trejects");
    for section in &sections {
        match section.as_str() {
            "fifo" => section_prefill("fifo", &fleets, tasks, per_task, window, &names),
            "window" => section_prefill("window", &fleets, tasks, per_task, binding, &names),
            other => eprintln!("unknown section {other}"),
        }
    }
}
