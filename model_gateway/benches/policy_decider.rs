//! The policy engine's seam under the gateway's admission mechanics: the
//! decisions per second and their latency under 64 concurrent admissions
//! through the real prefill-admission FIFO (head-only selection, a per-worker
//! capacity window) and the decode-room claims, for three shapes:
//!
//! - `legacy_under_lock`: today's path, the policy over worker handles inside
//!   the admission callback (under the lock);
//! - `engine_under_lock`: the engine's view, decision and commit inside the
//!   same callback (what `placement::select_pair` does with the switch on);
//! - `engine_decided`: the decision outside the lock, the commit under it
//!   (`PrefillAdmission::admit_decided`).
//!
//! Sections (`PD_SECTIONS`, default all): `fifo` (a window that never binds:
//! the lock hold alone), `window` (a binding window: waits and commit misses),
//! `rooms` (decode-room claims, no shared lock), `pair` (the prefill and
//! decode pick together under the FIFO). Fleets from `PD_FLEETS` (default
//! 8,64,512), `PD_TASKS` tasks (64) on eight runtime threads, `PD_PER_TASK`
//! admissions each (2000). Output: TSV rows on stdout:
//! `section  shape  policy  workers  p50_ns  p99_ns  per_s  misses  waits  rejects`.
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
        engine_host::SingleRequest, Decider, LeastLoadPolicy, LoadBalancingPolicy,
        RoundRobinPolicy, SelectWorkerInfo, WorkerRole,
    },
    worker::{
        acquire_prefill, acquire_prefill_decided,
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, CapacityWindow, PrefillAdmission, PrefillCandidateError,
        PrefillSelectionContext, Worker, WorkerType,
    },
};
use smg_policy::{Decision, Lease, Phase, Side, Target, WorkerId};

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

fn pool(n: usize, kind: WorkerType, prefix: &str, window: Option<usize>) -> Vec<Arc<dyn Worker>> {
    (0..n)
        .map(|i| {
            let mut builder = BasicWorkerBuilder::new(format!("http://{prefix}-{i:03}:8000"))
                .worker_type(kind)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                });
            if let Some(window) = window {
                builder = builder.label("max_running_requests", window.to_string());
            }
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
    waits: AtomicU64,
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
        "{section}\t{shape}\t{policy}\t{workers}\t{}\t{}\t{:.0}\t{}\t{}\t{}",
        row.p50,
        row.p99,
        row.per_s,
        c.misses.load(Ordering::Relaxed),
        c.waits.load(Ordering::Relaxed),
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
    let decider = Arc::new(Decider::from_configs(&config, Some(&config), Some(&config)).unwrap());
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

/// The engine's decision over the prefill pool with the window bits.
fn engine_prefill_decision(
    decider: &Decider,
    pool: &[Arc<dyn Worker>],
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    excluded: &[WorkerId],
) -> Result<(Decision, Lease), Verdict> {
    let request = SingleRequest {
        model: "m",
        phase: Phase::Prefill,
        has_capacity: Some(has_capacity),
        ..SingleRequest::default()
    };
    let (decision, lease) = decider.decide_single(&request, pool, excluded);
    match decision {
        Decision::Pick(_) => Ok((decision, lease)),
        Decision::Shed(shed) if shed.emptied_by == "has-capacity" => Err(Verdict::AtCapacity),
        Decision::Shed(_) => Err(Verdict::Unavailable),
    }
}

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
            missed
                .lock()
                .unwrap()
                .extend(Decider::target_ids(&decision));
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
        let prefills = pool(n, WorkerType::Prefill, "prefill", None);
        for name in names {
            let p = Arc::new(policies(name, &[&prefills]));
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

// ───────────────────────── decode rooms ─────────────────────────

fn has_free_room(decode: &dyn Worker) -> bool {
    decode
        .max_running_requests()
        .is_none_or(|window| decode.pd_admitted() + decode.pd_waiting_rooms() < usize::from(window))
}

fn section_rooms(fleets: &[usize], tasks: usize, per_task: usize, rooms: usize, names: &[String]) {
    for &n in fleets {
        let decodes = pool(n, WorkerType::Decode, "decode", Some(rooms));
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
                            // Today's shape: the room pre-filter, the policy
                            // over the handles, the atomic claim; a lost
                            // claim re-selects, up to four times.
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
                            // The engine's shape: the room bit in the view
                            // (a soft filter), the claim as the commit over
                            // the ordered targets.
                            let request = SingleRequest {
                                model: "m",
                                phase: Phase::Decode,
                                ..SingleRequest::default()
                            };
                            let (decision, lease) =
                                p.decider.decide_single(&request, &decodes, &[]);
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

fn section_pair(fleets: &[usize], tasks: usize, per_task: usize, window: usize, names: &[String]) {
    for &n in fleets {
        let prefill_count = (n / 4).max(2);
        let prefills = pool(prefill_count, WorkerType::Prefill, "prefill", None);
        let decodes = pool(n, WorkerType::Decode, "decode", Some(4096));
        // Prefill `i` partners with the decodes `j` where `j % prefills == i`.
        let partners_of: Arc<Vec<Vec<Arc<dyn Worker>>>> = Arc::new(
            (0..prefill_count)
                .map(|i| {
                    decodes
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| j % prefill_count == i)
                        .map(|(_, d)| Arc::clone(d))
                        .collect()
                })
                .collect(),
        );
        for name in names {
            let p = Arc::new(policies(name, &[&prefills, &decodes]));
            for shape in ["legacy_under_lock", "engine_under_lock", "engine_decided"] {
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
                                |pair: &(Arc<dyn Worker>, Arc<dyn Worker>)| &pair.0,
                                |capacity| {
                                    let open: Vec<usize> = (0..prefills.len())
                                        .filter(|&i| {
                                            capacity.is_none_or(|c| c.has_capacity(&prefills[i]))
                                        })
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
                                    let pi = p
                                        .legacy
                                        .select_worker(&open_workers, &info)
                                        .ok_or(Verdict::Unavailable)?;
                                    let partners = &partners_of[open[pi]];
                                    let with_room: Vec<Arc<dyn Worker>> = partners
                                        .iter()
                                        .filter(|d| has_free_room(d.as_ref()))
                                        .cloned()
                                        .collect();
                                    let decodes = if with_room.is_empty() {
                                        partners
                                    } else {
                                        &with_room
                                    };
                                    info.role = WorkerRole::Decode;
                                    let di = p
                                        .legacy
                                        .select_worker(decodes, &info)
                                        .ok_or(Verdict::Unavailable)?;
                                    Ok((Arc::clone(&open_workers[pi]), Arc::clone(&decodes[di])))
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
                    "engine_under_lock" => run_concurrent(tasks, per_task, move |_| {
                        let admission = Arc::clone(&admission);
                        let prefills = prefills.clone();
                        let partners_of = Arc::clone(&partners_of);
                        let p = Arc::clone(&p);
                        let c = Arc::clone(&c);
                        async move {
                            let result = acquire_prefill(
                                Some(&admission),
                                None,
                                |pair: &(Arc<dyn Worker>, Arc<dyn Worker>)| &pair.0,
                                |capacity| {
                                    let has = |w: &Arc<dyn Worker>| {
                                        capacity.is_none_or(|c| c.has_capacity(w))
                                    };
                                    let partners = |i: usize| partners_of[i].clone();
                                    let request = SingleRequest {
                                        model: "m",
                                        phase: Phase::Prefill,
                                        ..SingleRequest::default()
                                    };
                                    let (decision, lease, decode_rows) = p.decider.decide_pair(
                                        &request,
                                        &prefills,
                                        &has,
                                        &partners,
                                        false,
                                        &[],
                                    );
                                    if let Decision::Shed(shed) = &decision {
                                        return Err(if shed.emptied_by == "has-capacity" {
                                            Verdict::AtCapacity
                                        } else {
                                            Verdict::Unavailable
                                        });
                                    }
                                    match p.decider.commit(
                                        &lease,
                                        &decision,
                                        Side::Prefill,
                                        |t: &Target| {
                                            has(&prefills[t.position as usize])
                                                && t.decode.is_some()
                                        },
                                    ) {
                                        Some(t) => Ok((
                                            Arc::clone(&prefills[t.position as usize]),
                                            Arc::clone(
                                                &decode_rows[t.decode.unwrap().position as usize],
                                            ),
                                        )),
                                        None => {
                                            c.misses.fetch_add(1, Ordering::Relaxed);
                                            Err(Verdict::AtCapacity)
                                        }
                                    }
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
                            let result = acquire_prefill_decided(
                                Some(&admission),
                                None,
                                |pair: &(Arc<dyn Worker>, Arc<dyn Worker>)| &pair.0,
                                |window: &CapacityWindow| {
                                    let has = |w: &Arc<dyn Worker>| window.has_capacity(w);
                                    let partners = |i: usize| partners_of[i].clone();
                                    let request = SingleRequest {
                                        model: "m",
                                        phase: Phase::Prefill,
                                        ..SingleRequest::default()
                                    };
                                    let (decision, lease, decode_rows) = p.decider.decide_pair(
                                        &request,
                                        &prefills,
                                        &has,
                                        &partners,
                                        false,
                                        &[],
                                    );
                                    if let Decision::Shed(shed) = &decision {
                                        return Err(if shed.emptied_by == "has-capacity" {
                                            Verdict::AtCapacity
                                        } else {
                                            Verdict::Unavailable
                                        });
                                    }
                                    Ok((decision, lease, decode_rows))
                                },
                                |capacity, (decision, lease, decode_rows)| {
                                    let has = |w: &Arc<dyn Worker>| {
                                        capacity.is_none_or(|c| c.has_capacity(w))
                                    };
                                    match p.decider.commit(
                                        &lease,
                                        &decision,
                                        Side::Prefill,
                                        |t: &Target| {
                                            has(&prefills[t.position as usize])
                                                && t.decode.is_some()
                                        },
                                    ) {
                                        Some(t) => Ok((
                                            Arc::clone(&prefills[t.position as usize]),
                                            Arc::clone(
                                                &decode_rows[t.decode.unwrap().position as usize],
                                            ),
                                        )),
                                        None => {
                                            c.misses.fetch_add(1, Ordering::Relaxed);
                                            Err(Verdict::AtCapacity)
                                        }
                                    }
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
                print_row("pair", shape, policy_name, n, &row, &counters);
            }
        }
    }
}

fn main() {
    let fleets: Vec<usize> = env_list("PD_FLEETS", "8,64,512")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let sections = env_list("PD_SECTIONS", "fifo,window,rooms,pair");
    let names = env_list("PD_POLICIES", "round_robin,least_load");
    let tasks: usize = env_num("PD_TASKS", 64);
    let per_task: usize = env_num("PD_PER_TASK", 2_000);
    let window: usize = env_num("PD_WINDOW", 4_096);
    let binding: usize = env_num("PD_WINDOW_BINDING", 1);
    let rooms: usize = env_num("PD_ROOMS", 4);
    println!("section\tshape\tpolicy\tworkers\tp50_ns\tp99_ns\tper_s\tmisses\twaits\trejects");
    for section in &sections {
        match section.as_str() {
            "fifo" => section_prefill("fifo", &fleets, tasks, per_task, window, &names),
            "window" => section_prefill("window", &fleets, tasks, per_task, binding, &names),
            "rooms" => section_rooms(&fleets, tasks, per_task, rooms, &names),
            "pair" => section_pair(&fleets, tasks, per_task, window, &names),
            other => eprintln!("unknown section {other}"),
        }
    }
}
