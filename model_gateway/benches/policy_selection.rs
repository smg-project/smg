//! Selection-policy stage cost: what one request pays inside `WorkerSelectionPolicy::select`
//! for each policy, and what the host pays to gather the inputs from a 128-worker overlap map
//! before calling it.
//!
//! Run with: cargo bench --bench policy_selection
#![expect(
    clippy::unwrap_used,
    reason = "benchmark code: panicking on setup failure is expected"
)]

use std::{collections::HashMap, hint::black_box};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg::policies::cost::{build, CandidateInputs, RequestInputs, POLICY_NAMES};

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

criterion_group!(benches, bench_select, bench_gather_and_select);
criterion_main!(benches);
