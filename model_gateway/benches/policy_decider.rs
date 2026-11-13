//! The policy engine's seam under the gateway's prefill admission: the
//! decisions per second and their latency under 64 concurrent admissions
//! through the real prefill-admission FIFO (head-only selection, a per-worker
//! capacity window), for two shapes:
//!
//! - `legacy_under_lock`: today's path, the policy over worker handles inside
//!   the admission callback (under the lock);
//! - `engine_under_lock`: the engine's view, decision and commit inside the
//!   same callback.
//!
//! Sections (`PD_SECTIONS`, default all five): `fifo` (a window that never
//! binds: the lock hold alone), `window` (a binding window: queueing and
//! commit misses), `rooms` (the decode room claim with no shared lock: the
//! room pre-filter and the atomic claim against the engine's room bit and the
//! claim as the commit), `pair` (the pair plan under the FIFO: the prefill
//! selection and its partner's room under the lock against the engine's pair
//! decision under it, prefill `i` partnered with the decodes
//! `j` where `j % prefills == i`) and `pair_all` (the same with every prefill
//! partnered with every decode through one shared list, the homogeneous
//! fleet's shape). Fleets from `PD_FLEETS` (default 8,64,512; the pair
//! sections run a quarter as many prefills, two at least),
//! `PD_TASKS` tasks (64) on eight runtime threads, `PD_PER_TASK` admissions
//! each (2000), `PD_ROOMS` rooms per decode (4). Output: TSV rows on stdout:
//! `section  shape  policy  workers  p50_ns  p99_ns  per_s  misses  rejects`.
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
    config::{PolicyConfig, RoutingKeyOverrideConfig},
    policies::{
        engine_host::{HostRequest, PairPartners},
        Decider, LeastLoadPolicy, LoadBalancingPolicy, RoundRobinPolicy, SelectWorkerInfo,
        WorkerRole,
    },
    worker::{
        acquire_prefill,
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, PrefillAdmission, PrefillCandidateError, PrefillSelectionContext,
        Worker, WorkerType,
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

/// A decode fleet whose engines run `rooms` requests at once: the label the
/// gateway's room gate reads.
fn decode_pool(n: usize, rooms: usize) -> Vec<Arc<dyn Worker>> {
    (0..n)
        .map(|i| {
            let mut labels = HashMap::new();
            labels.insert("max_running_requests".to_string(), rooms.to_string());
            let builder = BasicWorkerBuilder::new(format!("http://decode-{i:03}:8000"))
                .worker_type(WorkerType::Decode)
                .labels(labels)
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
/// `least_load` both read one load feed carrying a report per worker of
/// every pool given.
fn policies(name: &str, pools: &[&[Arc<dyn Worker>]]) -> Policies {
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
    let decider = Arc::new(
        Decider::from_configs(
            &config,
            &config,
            &config,
            &RoutingKeyOverrideConfig::default(),
        )
        .unwrap(),
    );
    let feed = (name == "least_load").then(|| {
        let feed = TestFeed::attached_to(legacy.as_ref());
        let mut reports = HashMap::new();
        for pool in pools {
            reports.extend(reports_for(pool));
        }
        feed.publish(&reports);
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
            let p = Arc::new(policies(name, &[&prefills]));
            let queue = tasks.max(1);
            for shape in ["legacy_under_lock", "engine_under_lock"] {
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
                    _ => run_concurrent(tasks, per_task, move |_| {
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
                };
                print_row(section, shape, policy_name, n, &row, &counters);
            }
        }
    }
}

// ───────────────────────── the decode room claim ─────────────────────────

/// The room gate's free-room read, as the decode pick's pre-filter asks it.
fn has_free_room(decode: &dyn Worker) -> bool {
    decode
        .max_running_requests()
        .is_none_or(|window| decode.pd_admitted() + decode.pd_waiting_rooms() < usize::from(window))
}

/// The decode room claim with no shared lock: today's shape (the room
/// pre-filter, the policy over the handles, the atomic claim, a lost claim
/// re-selected up to four times) against the engine's (the room bit in the
/// view as a soft filter, the claim as the commit over the ordered targets).
fn section_rooms(fleets: &[usize], tasks: usize, per_task: usize, rooms: usize, names: &[String]) {
    for &n in fleets {
        let decodes = decode_pool(n, rooms);
        let window = rooms;
        for name in names {
            let p = Arc::new(policies(name, &[&decodes]));
            for shape in ["legacy_claim", "engine_claim"] {
                let counters = Arc::new(Counters::default());
                let decodes = decodes.clone();
                let p = Arc::clone(&p);
                let policy_name = p.name;
                let c = Arc::clone(&counters);
                let row = if shape == "legacy_claim" {
                    run_concurrent(tasks, per_task, move |_| {
                        let decodes = decodes.clone();
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            for _ in 0..4 {
                                let with_room: Vec<Arc<dyn Worker>> = decodes
                                    .iter()
                                    .filter(|d| has_free_room(d.as_ref()))
                                    .cloned()
                                    .collect();
                                let candidates = if with_room.is_empty() {
                                    &decodes
                                } else {
                                    &with_room
                                };
                                let info = SelectWorkerInfo {
                                    role: WorkerRole::Decode,
                                    ..SelectWorkerInfo::default()
                                };
                                let Some(idx) = p.legacy.select_worker(candidates, &info) else {
                                    c.rejects.fetch_add(1, Ordering::Relaxed);
                                    return;
                                };
                                if candidates[idx].try_admit_pd(1, window) {
                                    candidates[idx].release_pd(1);
                                    return;
                                }
                                c.misses.fetch_add(1, Ordering::Relaxed);
                            }
                            c.rejects.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                } else {
                    run_concurrent(tasks, per_task, move |_| {
                        let decodes = decodes.clone();
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let request = HostRequest {
                                model: "m",
                                prompt_tokens: 1024,
                                phase: Phase::Decode,
                                ..HostRequest::default()
                            };
                            let (decision, lease) = p.decider.decide(&request, &decodes, &[]);
                            let claimed =
                                p.decider
                                    .commit(&lease, &decision, Side::Decode, |t: &Target| {
                                        decodes[t.position as usize].try_admit_pd(1, window)
                                    });
                            match claimed {
                                Some(t) => decodes[t.position as usize].release_pd(1),
                                None => {
                                    c.misses.fetch_add(1, Ordering::Relaxed);
                                    c.rejects.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    })
                };
                print_row("rooms", shape, policy_name, n, &row, &counters);
            }
        }
    }
}

// ───────────────────────── the pair plan under the FIFO ─────────────────────────

type PairArms = (Arc<dyn Worker>, Arc<dyn Worker>);

/// What the engine's pair decision hands the commit: the decision, its lease
/// and the decode rows the targets' pair sides index.
type PairPlan = (Decision, Lease, Vec<Arc<dyn Worker>>);

/// Prefill `i`'s partner list; prefills of one pairing descriptor share one
/// list, as the gateway's pair index shares it.
type Partners = Arc<Vec<Arc<[Arc<dyn Worker>]>>>;

/// The counted prompt every pair request carries; both sides book it.
const PROMPT_TOKENS: u32 = 1024;

/// How the pair sections partner the prefills with the decodes.
#[derive(Clone, Copy)]
enum Topology {
    /// Prefill `i` with the decodes `j` where `j % prefills == i`: distinct
    /// lists, the engine's best case.
    Partitioned,
    /// Every prefill with every decode through one shared list: the
    /// homogeneous fleet.
    AllToAll,
}

impl Topology {
    fn section(self) -> &'static str {
        match self {
            Self::Partitioned => "pair",
            Self::AllToAll => "pair_all",
        }
    }

    fn partners(self, prefills: usize, decodes: &[Arc<dyn Worker>]) -> Partners {
        Arc::new(match self {
            Self::Partitioned => (0..prefills)
                .map(|i| {
                    decodes
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| j % prefills == i)
                        .map(|(_, d)| Arc::clone(d))
                        .collect()
                })
                .collect(),
            Self::AllToAll => {
                let all: Arc<[Arc<dyn Worker>]> = decodes.iter().cloned().collect();
                vec![all; prefills]
            }
        })
    }
}

/// Today's pair selection at the queue head: the capacity pre-filter, the
/// prefill policy over the open prefills, the room pre-filter over its
/// partners, the decode policy over them.
fn legacy_pair(
    policy: &dyn LoadBalancingPolicy,
    prefills: &[Arc<dyn Worker>],
    partners_of: &[Arc<[Arc<dyn Worker>]>],
    capacity: Option<&PrefillSelectionContext<'_>>,
) -> Result<PairArms, Verdict> {
    let open: Vec<usize> = (0..prefills.len())
        .filter(|&i| capacity.is_none_or(|c| c.has_capacity(&prefills[i])))
        .collect();
    if open.is_empty() {
        return Err(Verdict::AtCapacity);
    }
    let open_workers: Vec<Arc<dyn Worker>> =
        open.iter().map(|&i| Arc::clone(&prefills[i])).collect();
    let mut info = SelectWorkerInfo {
        role: WorkerRole::Prefill,
        ..SelectWorkerInfo::default()
    };
    let pi = policy
        .select_worker(&open_workers, &info)
        .ok_or(Verdict::Unavailable)?;
    let partners: &[Arc<dyn Worker>] = &partners_of[open[pi]];
    let with_room: Vec<Arc<dyn Worker>> = partners
        .iter()
        .filter(|d| has_free_room(d.as_ref()))
        .cloned()
        .collect();
    let decodes: &[Arc<dyn Worker>] = if with_room.is_empty() {
        partners
    } else {
        &with_room
    };
    info.role = WorkerRole::Decode;
    let di = policy
        .select_worker(decodes, &info)
        .ok_or(Verdict::Unavailable)?;
    Ok((Arc::clone(&open_workers[pi]), Arc::clone(&decodes[di])))
}

/// The engine's pair decision over the prefills with the window bits and
/// their partners with the room bits.
fn engine_pair_decision(
    decider: &Decider,
    prefills: &[Arc<dyn Worker>],
    partners_of: &Partners,
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    excluded: &[WorkerId],
) -> Result<PairPlan, Verdict> {
    let list = |i: usize| &partners_of[i];
    let request = HostRequest {
        model: "m",
        prompt_tokens: PROMPT_TOKENS,
        phase: Phase::Prefill,
        ..HostRequest::default()
    };
    let partners = PairPartners {
        list: &list,
        open: &|_| true,
        lift_decode: false,
        affinity: None,
    };
    let (decision, lease, decodes) =
        decider.decide_pair(&request, prefills, has_capacity, partners, excluded);
    match decision {
        Decision::Pick(_) => Ok((decision, lease, decodes)),
        Decision::Shed(shed) if shed.emptied_by == "has-capacity" => Err(Verdict::AtCapacity),
        Decision::Shed(_) | Decision::Queue(_) => Err(Verdict::Unavailable),
    }
}

/// The commit of a pair decision under the window as the lock reads it: the
/// first target whose prefill has capacity, its decode side the pair's
/// decode; a miss records the refused targets' ids for the re-decision.
fn engine_pair_commit(
    decider: &Decider,
    prefills: &[Arc<dyn Worker>],
    (decision, lease, decodes): PairPlan,
    capacity: Option<&PrefillSelectionContext<'_>>,
    counters: &Counters,
    missed: &std::sync::Mutex<Vec<WorkerId>>,
) -> Result<PairArms, Verdict> {
    let committed = decider.commit_pair(&lease, PROMPT_TOKENS, &decision, |t: &Target| {
        capacity.is_none_or(|c| c.has_capacity(&prefills[t.position as usize])) && t.pair.is_some()
    });
    match committed {
        Some(t) => {
            let decode = t
                .pair
                .map(|pair| &decodes[pair.decode_position as usize])
                .unwrap();
            Ok((
                Arc::clone(&prefills[t.position as usize]),
                Arc::clone(decode),
            ))
        }
        None => {
            counters.misses.fetch_add(1, Ordering::Relaxed);
            if let Some(pick) = decision.pick() {
                let mut missed = missed.lock().unwrap();
                for t in pick.targets.iter() {
                    missed.push(t.id);
                    missed.extend(t.pair.map(|pair| pair.decode_id));
                }
            }
            Err(Verdict::AtCapacity)
        }
    }
}

/// The pair plan under the FIFO: a quarter as many prefills as decodes (two
/// at least), partnered by `topology`, the window never binding (the lock
/// hold and the view build alone).
fn section_pair(
    topology: Topology,
    fleets: &[usize],
    tasks: usize,
    per_task: usize,
    window: usize,
    names: &[String],
) {
    for &n in fleets {
        let prefill_count = (n / 4).max(2);
        let prefills = pool(prefill_count, WorkerType::Prefill, "prefill");
        let decodes = decode_pool(n, 4_096);
        let partners_of = topology.partners(prefill_count, &decodes);
        for name in names {
            let p = Arc::new(policies(name, &[&prefills, &decodes]));
            for shape in ["legacy_under_lock", "engine_under_lock"] {
                let admission = Arc::new(PrefillAdmission::new(
                    window,
                    tasks.max(1),
                    Duration::from_secs(30),
                ));
                let counters = Arc::new(Counters::default());
                let prefills = prefills.clone();
                let partners_of = Arc::clone(&partners_of);
                let p = Arc::clone(&p);
                let policy_name = p.name;
                let c = Arc::clone(&counters);
                let row = match shape {
                    "legacy_under_lock" => run_concurrent(tasks, per_task, move |_| {
                        let admission = Arc::clone(&admission);
                        let prefills = prefills.clone();
                        let partners_of = Arc::clone(&partners_of);
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let result = acquire_prefill(
                                Some(&admission),
                                None,
                                |pair: &PairArms| &pair.0,
                                |capacity| {
                                    legacy_pair(
                                        p.legacy.as_ref(),
                                        &prefills,
                                        &partners_of,
                                        capacity,
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
                        let partners_of = Arc::clone(&partners_of);
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let missed: std::sync::Mutex<Vec<WorkerId>> =
                                std::sync::Mutex::new(Vec::new());
                            let result = acquire_prefill(
                                Some(&admission),
                                None,
                                |pair: &PairArms| &pair.0,
                                |capacity| {
                                    let has = |w: &Arc<dyn Worker>| {
                                        capacity.is_none_or(|c| c.has_capacity(w))
                                    };
                                    let proposal = engine_pair_decision(
                                        &p.decider,
                                        &prefills,
                                        &partners_of,
                                        &has,
                                        &missed.lock().unwrap().clone(),
                                    )?;
                                    engine_pair_commit(
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
                print_row(topology.section(), shape, policy_name, n, &row, &counters);
            }
        }
    }
}

fn main() {
    let fleets: Vec<usize> = env_list("PD_FLEETS", "8,64,512")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let sections = env_list("PD_SECTIONS", "fifo,window,rooms,pair,pair_all");
    let names = env_list("PD_POLICIES", "round_robin,least_load");
    let tasks: usize = env_num("PD_TASKS", 64);
    let per_task: usize = env_num("PD_PER_TASK", 2_000);
    let window: usize = env_num("PD_WINDOW", 4_096);
    let binding: usize = env_num("PD_WINDOW_BINDING", 1);
    let rooms: usize = env_num("PD_ROOMS", 4);
    println!("section\tshape\tpolicy\tworkers\tp50_ns\tp99_ns\tper_s\tmisses\trejects");
    for section in &sections {
        match section.as_str() {
            "fifo" => section_prefill("fifo", &fleets, tasks, per_task, window, &names),
            "window" => section_prefill("window", &fleets, tasks, per_task, binding, &names),
            "rooms" => section_rooms(&fleets, tasks, per_task, rooms, &names),
            "pair" => section_pair(
                Topology::Partitioned,
                &fleets,
                tasks,
                per_task,
                window,
                &names,
            ),
            "pair_all" => {
                section_pair(Topology::AllToAll, &fleets, tasks, per_task, window, &names);
            }
            other => eprintln!("unknown section {other}"),
        }
    }
}
