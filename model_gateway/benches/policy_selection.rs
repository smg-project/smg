//! Selection-policy stage cost: what one request pays inside `WorkerSelectionPolicy::select`
//! for each policy, what the host pays to gather the inputs from a 128-worker overlap map
//! before calling it, and what one least-load decision pays over a fed pool (the expected-wait
//! scorer that least_load, power_of_two and cache-aware share).
//!
//! Run with: cargo bench --bench policy_selection
#![expect(
    clippy::unwrap_used,
    reason = "benchmark code: panicking on setup failure is expected"
)]

use std::{collections::HashMap, hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openai_protocol::worker::{HealthCheckConfig, WorkerLoadResponse};
use smg::{
    policies::{
        cost::{build, CandidateInputs, RequestInputs, POLICY_NAMES},
        LeastLoadPolicy, LoadBalancingPolicy, SelectWorkerInfo,
    },
    worker::{
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, Worker, WorkerType,
    },
};

const BLOCK_SIZE: usize = 16;
const PROMPT_TOKENS: usize = 4_096;

/// Deterministic pseudo-random fleet state (SplitMix64).
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Fleet {
    urls: Vec<String>,
    /// Overlap in blocks for the workers that hold part of the prompt (the indexer's map).
    overlap: HashMap<u32, u32>,
    prefix_hashes: Vec<u64>,
}

fn fleet(workers: usize) -> Fleet {
    let urls: Vec<String> = (0..workers)
        .map(|i| format!("http://worker-{i:03}:8000"))
        .collect();
    let overlap = (0..workers as u32)
        .filter(|&i| mix(u64::from(i)).is_multiple_of(3))
        .map(|i| (i, (mix(u64::from(i) + 7) % 256) as u32 + 1))
        .collect();
    let prefix_hashes = (0..PROMPT_TOKENS / BLOCK_SIZE)
        .map(|i| mix(i as u64 + 1_000))
        .collect();
    Fleet {
        urls,
        overlap,
        prefix_hashes,
    }
}

fn gather<'a>(fleet: &'a Fleet, all_workers: bool) -> Vec<CandidateInputs<'a>> {
    fleet
        .urls
        .iter()
        .enumerate()
        .filter_map(|(idx, url)| {
            let overlap = fleet.overlap.get(&(idx as u32)).copied().unwrap_or(0);
            if overlap == 0 && !all_workers {
                return None;
            }
            Some(CandidateInputs {
                idx,
                url,
                device_blocks: f64::from(overlap),
                effective_score: f64::from(overlap),
            })
        })
        .collect()
}

fn request(fleet: &Fleet) -> RequestInputs<'_> {
    RequestInputs {
        prompt_tokens: PROMPT_TOKENS,
        block_size: BLOCK_SIZE,
        request_blocks: PROMPT_TOKENS / BLOCK_SIZE,
        avg_load: 8.0,
        prefix_hashes: Some(&fleet.prefix_hashes),
    }
}

/// The policy stage alone: inputs are prepared once, `select` runs per iteration.
fn bench_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_selection/select");
    for workers in [8, 32, 128] {
        let fleet = fleet(workers);
        for name in POLICY_NAMES {
            let policy = build(name, 0.0).unwrap();
            let inputs = gather(&fleet, policy.needs().all_workers);
            let req = request(&fleet);
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(BenchmarkId::new(*name, workers), &inputs, |b, inputs| {
                b.iter(|| black_box(policy.select(black_box(&req), black_box(inputs))));
            });
        }
    }
    group.finish();
}

/// Gather plus select: per iteration, build the candidate inputs from the 128-worker overlap
/// map (what the cache-aware host does), then select.
fn bench_gather_and_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_selection/gather_and_select");
    let fleet = fleet(128);
    for name in POLICY_NAMES {
        let policy = build(name, 0.0).unwrap();
        let all_workers = policy.needs().all_workers;
        let req = request(&fleet);
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new(*name, 128), |b| {
            b.iter(|| {
                let inputs = gather(black_box(&fleet), all_workers);
                black_box(policy.select(black_box(&req), &inputs))
            });
        });
    }
    group.finish();
}

/// A pool of `workers` ready regular workers (no health checker behind them), each with a
/// published report (queued token-work, running requests, KV pressure, a live rate on three of
/// four) and in-flight requests on the router, under a least-load policy whose tally has credited
/// every worker once.
fn fed_least_load(workers: usize) -> (LeastLoadPolicy, TestFeed, Vec<Arc<dyn Worker>>) {
    let policy = LeastLoadPolicy::new();
    let feed = TestFeed::attached_to(&policy);
    let pool: Vec<Arc<dyn Worker>> = (0..workers)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i:03}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let reports: HashMap<String, WorkerLoadResponse> = pool
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
            for _ in 0..draw(5) % 4 {
                worker.increment_load();
            }
            (worker.url().to_string(), report)
        })
        .collect();
    feed.publish(&reports);
    // The tally knows every worker: the credit of the first decision on each
    // is the one allocation a decision ever makes for it.
    let info = SelectWorkerInfo::default();
    for _ in 0..workers * 4 {
        black_box(policy.select_worker(&pool, &info));
    }
    (policy, feed, pool)
}

/// One least-load decision over a fed pool: the eligibility walk, the expected-wait score of
/// every candidate from its report and its since-report credit, the argmin with its tie draw,
/// and the winner's credit. The request carries 1,024 token ids.
fn bench_least_load_decision(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_load/decision");
    let tokens: Vec<u32> = (0..1_024).collect();
    for workers in [8, 32, 128] {
        let (policy, _feed, pool) = fed_least_load(workers);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("workers", workers), &workers, |b, _| {
            b.iter(|| black_box(policy.select_worker(black_box(&pool), black_box(&info))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_select,
    bench_gather_and_select,
    bench_least_load_decision
);
criterion_main!(benches);
