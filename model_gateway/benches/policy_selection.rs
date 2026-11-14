//! Selection-policy stage cost: what one request pays inside `WorkerSelectionPolicy::select`
//! for each policy, what the host pays to gather the inputs from a 128-worker overlap map
//! before calling it, what one least-load decision pays over a fed pool (the expected-wait
//! scorer that least_load, power_of_two and cache-aware share), what one request pays inside
//! the two ring policies over a ready pool with its ring, what one pick of the simple
//! policies (power_of_two, random, round_robin, passthrough) pays over the same pool, what
//! a keyed request pays for a sticky hit, under the routing-key override and in the manual
//! policy, what one regular placement pays through the routers' single-worker selection
//! (`regular_placement`: round robin and cache-aware over an event index, 128 and 1,024
//! workers), and what one prefill-decode pair placement pays through the routers' pair
//! selection (`pair_placement`: round robin and cache-aware over an event index, 64 and 512
//! workers a side, untraced and inside a cache-trace capture). The two placement groups
//! refuse to run under a process trace switch (`SMG_CACHE_TRACE`, `SMG_CACHE_TRACE_HEADER`),
//! which captures every placement, the cells labelled untraced too.
//!
//! Run with: cargo bench --bench policy_selection
#![expect(
    clippy::unwrap_used,
    reason = "benchmark code: panicking on setup failure is expected"
)]

use std::{collections::HashMap, hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use kv_index::{compute_content_hash, request_prefix_hashes};
use openai_protocol::worker::{HealthCheckConfig, WorkerLoadResponse};
use smg::{
    config::types::{ManualAssignmentMode, PolicyConfig, RoutingKeyOverrideConfig},
    policies::{
        cost::{build, CandidateInputs, RequestInputs, POLICY_NAMES},
        CacheAwareConfig, CacheAwarePolicy, ConsistentHashingPolicy, LeastLoadPolicy,
        LoadBalancingPolicy, ManualPolicy, PassthroughPolicy, PolicyRegistry, PowerOfTwoPolicy,
        PrefixHashConfig, PrefixHashPolicy, RandomPolicy, RoundRobinPolicy, SelectWorkerInfo,
    },
    routers::common::bench_support::{process_trace_configured, PairPath, RegularPath},
    worker::{
        kv_event_monitor::bench_support::IndexFeed,
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, ConnectionMode, HashRing, KvEventMonitor, KvIndex, ModelCard,
        RuntimeType, Worker, WorkerRegistry, WorkerType,
    },
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch,
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
/// four) and in-flight requests on the router, fed to `policy`, which has picked from the pool
/// often enough to know every worker.
fn fed_pool(
    policy: &dyn LoadBalancingPolicy,
    workers: usize,
) -> (
    TestFeed,
    Vec<Arc<dyn Worker>>,
    HashMap<String, WorkerLoadResponse>,
) {
    let feed = TestFeed::attached_to(policy);
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
    // A registered worker carries a published state, which the routing reads
    // answer from; the pool reads like one.
    for worker in &pool {
        worker.republish_state();
    }
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
    (feed, pool, reports)
}

/// One least-load decision over a fed pool: the eligibility walk, the expected-wait score of
/// every candidate from its report and its since-report credit, the argmin with its tie draw,
/// and the winner's credit. The request carries 1,024 token ids. Four workers take the decision's
/// held-guard shape, the larger pools its split shape.
fn bench_least_load_decision(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_load/decision");
    let tokens: Vec<u32> = (0..1_024).collect();
    for workers in [4, 8, 32, 128] {
        let policy = LeastLoadPolicy::new();
        let (_feed, pool, _reports) = fed_pool(&policy, workers);
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

/// The first least-load decision after a fresh report, the regime in which the tally releases:
/// the previous decision's dispatch is at or before the new report's sample, so the winner's
/// credit is read and released in the same decision. The report is republished before each
/// timed decision, outside the timing. Four workers take the held-guard shape, eight the split
/// one.
fn bench_least_load_after_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_load/after_report");
    let tokens: Vec<u32> = (0..1_024).collect();
    for workers in [4, 8] {
        let policy = LeastLoadPolicy::new();
        let (feed, pool, reports) = fed_pool(&policy, workers);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("workers", workers), &workers, |b, _| {
            b.iter_batched(
                || feed.publish(&reports),
                |()| black_box(policy.select_worker(black_box(&pool), black_box(&info))),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// `workers` ready regular workers (no health checker behind them) and the ring built over their
/// URLs, as the registry builds it.
fn ring_pool(workers: usize) -> (Vec<Arc<dyn Worker>>, Arc<HashRing>) {
    let pool = ready_pool(workers);
    let ring = Arc::new(HashRing::new(pool.iter().map(|worker| worker.url())));
    (pool, ring)
}

/// One request of each ring policy over a ready pool with its ring: consistent_hashing placing a
/// session key, prefix_hash placing a 1,024-token prompt over the boundaries 64 and 256 (the
/// deeper level's owner is under its load margin, so one ring lookup places it). Every worker is
/// eligible, so the ring's answer is the pick.
fn bench_ring_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_policies/select");
    let tokens: Vec<u32> = (0..1_024).collect();
    for workers in [8, 32, 128] {
        let (pool, ring) = ring_pool(workers);
        let consistent = ConsistentHashingPolicy::new();
        let keyed = SelectWorkerInfo {
            routing_key: Some("session-0001"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("consistent_hashing", workers),
            &workers,
            |b, _| {
                b.iter(|| black_box(consistent.select_worker(black_box(&pool), black_box(&keyed))));
            },
        );
        let prefix = PrefixHashPolicy::new(PrefixHashConfig {
            cache_boundaries: vec![64, 256],
            ..Default::default()
        });
        let prompt = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("prefix_hash", workers),
            &workers,
            |b, _| {
                b.iter(|| black_box(prefix.select_worker(black_box(&pool), black_box(&prompt))));
            },
        );
    }
    group.finish();
}

/// One pick of each simple policy over the fed pool with every worker eligible: power_of_two
/// (its pair drawn, the pair scored), random, round_robin (its rotation keyed by the eligible
/// set) and passthrough (the first eligible worker). The request carries 1,024 token ids.
fn bench_simple_policy_pick(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_policies/pick");
    let tokens: Vec<u32> = (0..1_024).collect();
    for workers in [8, 32, 128] {
        let policies: [(&str, Box<dyn LoadBalancingPolicy>); 4] = [
            ("power_of_two", Box::new(PowerOfTwoPolicy::new())),
            ("random", Box::new(RandomPolicy::new())),
            ("round_robin", Box::new(RoundRobinPolicy::new())),
            ("passthrough", Box::new(PassthroughPolicy::new())),
        ];
        for (name, policy) in &policies {
            let (_feed, pool, _reports) = fed_pool(policy.as_ref(), workers);
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            };
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(BenchmarkId::new(*name, workers), &workers, |b, _| {
                b.iter(|| black_box(policy.select_worker(black_box(&pool), black_box(&info))));
            });
        }
    }
    group.finish();
}

/// `workers` ready regular workers (no health checker behind them).
fn ready_pool(workers: usize) -> Vec<Arc<dyn Worker>> {
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
    // A registered worker carries a published state, which the routing reads
    // answer from; the pool reads like one.
    for worker in &pool {
        worker.republish_state();
    }
    pool
}

/// A keyed hit over a ready pool: sixteen conversations pinned once, then one request per
/// iteration under a key already pinned, through the routing-key override (delegating its
/// assignments to a round-robin registry) and through the manual policy on its own.
fn bench_keyed_sticky_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyed_sticky/hit");
    let keys: Vec<String> = (0..16).map(|i| format!("conversation-{i:04}")).collect();
    for workers in [8, 32, 128] {
        let pool = ready_pool(workers);
        let model_id = pool[0].model_id().to_string();
        let registry = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::Delegate,
                ..Default::default()
            },
        );
        let policy = registry.get_default_policy();
        let manual = ManualPolicy::new();
        let infos: Vec<SelectWorkerInfo> = keys
            .iter()
            .map(|key| SelectWorkerInfo {
                rid_key: Some(key),
                ..Default::default()
            })
            .collect();
        for info in &infos {
            black_box(registry.select_worker_for_model(&policy, &model_id, &pool, info));
            black_box(manual.select_worker(&pool, info));
        }
        group.throughput(Throughput::Elements(1));
        let mut next = 0usize;
        group.bench_with_input(BenchmarkId::new("override", workers), &workers, |b, _| {
            b.iter(|| {
                let info = &infos[next % infos.len()];
                next += 1;
                black_box(registry.select_worker_for_model(
                    &policy,
                    &model_id,
                    black_box(&pool),
                    black_box(info),
                ))
            });
        });
        let mut next = 0usize;
        group.bench_with_input(BenchmarkId::new("manual", workers), &workers, |b, _| {
            b.iter(|| {
                let info = &infos[next % infos.len()];
                next += 1;
                black_box(manual.select_worker(black_box(&pool), black_box(info)))
            });
        });
    }
    group.finish();
}

/// The model of the regular fleet.
const REGULAR_MODEL: &str = "regular-model";
/// The model of the pair fleet.
const PAIR_MODEL: &str = "pair-model";
/// Blocks of a placement request's prompt.
const PLACEMENT_PROMPT_BLOCKS: usize = 4;

/// A placement group measures what its labels say only in a process with no trace switch:
/// a switch captures every placement, the cells labelled untraced too.
fn refuse_a_process_trace(group: &str) {
    assert!(
        !process_trace_configured(),
        "{group}: unset SMG_CACHE_TRACE and SMG_CACHE_TRACE_HEADER; a process trace switch \
         captures every placement, the untraced cells too"
    );
}

/// The stored event of one prompt, as an engine publishes a chain.
fn stored(tokens: &[u32]) -> KvEventBatch {
    let contents: Vec<_> = tokens
        .chunks(BLOCK_SIZE)
        .map(compute_content_hash)
        .collect();
    let blocks = tokens
        .chunks(BLOCK_SIZE)
        .zip(request_prefix_hashes(&contents))
        .map(|(chunk, prefix)| KvBlock {
            block_hash: prefix.0 as i64,
            token_ids: chunk.to_vec(),
            block_size: BLOCK_SIZE as i32,
            ..Default::default()
        })
        .collect();
    KvEventBatch {
        events: vec![KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks,
                ..Default::default()
            })),
        }],
        ..Default::default()
    }
}

/// One prompt a worker of a placement fleet's pool.
fn placement_prompts(count: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|p| {
            (0..BLOCK_SIZE * PLACEMENT_PROMPT_BLOCKS)
                .map(|t| (p * 1_000 + t + 1) as u32)
                .collect()
        })
        .collect()
}

/// An event monitor whose index of `model` holds prompt `p` on worker `p` of each pool.
fn holders_monitor(
    model: &str,
    pools: &[Vec<Arc<dyn Worker>>],
    prompts: &[Vec<u32>],
) -> Arc<KvEventMonitor> {
    let index = Arc::new(KvIndex::chain());
    for pool in pools {
        for (p, worker) in pool.iter().enumerate() {
            let mut feed = IndexFeed::new(&index, worker.url()).unwrap();
            feed.apply(&index, &stored(&prompts[p]));
        }
    }
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.set_index(model, Arc::clone(&index));
    monitor.set_block_size(model, BLOCK_SIZE);
    monitor
}

/// A regular fleet of one model, `workers` workers over the gRPC wire, registered as the
/// gateway registers them, under one default policy: round robin, or cache-aware (the pair
/// fleet's configuration) over one event index in which prompt `p` is held by worker `p`.
/// The prompts come with it.
fn regular_fleet(workers: usize, cache_aware: bool) -> (RegularPath, Vec<Vec<u32>>) {
    let registry = Arc::new(WorkerRegistry::new());
    let pool: Vec<Arc<dyn Worker>> = (0..workers)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://regular-{i:04}:9000"))
                    .model(ModelCard::new(REGULAR_MODEL))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    for worker in &pool {
        registry.register(Arc::clone(worker)).unwrap();
    }
    let prompts = placement_prompts(workers);
    let policies = if cache_aware {
        let policies = PolicyRegistry::new(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 10_000));
        policies.set_kv_event_monitor(Some(holders_monitor(REGULAR_MODEL, &[pool], &prompts)));
        policies
    } else {
        PolicyRegistry::new(PolicyConfig::RoundRobin)
    };
    (
        RegularPath::new(registry, Arc::new(policies), REGULAR_MODEL),
        prompts,
    )
}

/// One regular placement through the routers' own single-worker selection: the model's pool
/// drawn from the registry and one policy pick over it, with no filter, no wire pin and no
/// tried engine. Under round robin, and under cache-aware where each request's prompt has
/// one holder (an index lookup and a hit), at 128 and 1,024 workers. The
/// `cache_aware_filtered` cells run the cache-aware placement under a candidate filter that
/// refuses nobody, as both routers hand one on a request's first attempt.
fn bench_regular_placement(c: &mut Criterion) {
    refuse_a_process_trace("regular_placement");
    let mut group = c.benchmark_group("regular_placement");
    for workers in [128, 1_024] {
        for (policy, cache_aware) in [("round_robin", false), ("cache_aware", true)] {
            let (path, prompts) = regular_fleet(workers, cache_aware);
            group.throughput(Throughput::Elements(1));
            let mut next = 0usize;
            group.bench_with_input(BenchmarkId::new(policy, workers), &workers, |b, _| {
                b.iter(|| {
                    let tokens = &prompts[next % prompts.len()];
                    next += 1;
                    black_box(path.place(Some(black_box(tokens))).unwrap())
                });
            });
            if !cache_aware {
                continue;
            }
            let mut next = 0usize;
            group.bench_with_input(
                BenchmarkId::new("cache_aware_filtered", workers),
                &workers,
                |b, _| {
                    b.iter(|| {
                        let tokens = &prompts[next % prompts.len()];
                        next += 1;
                        black_box(path.place_filtered(Some(black_box(tokens))).unwrap())
                    });
                },
            );
        }
    }
    group.finish();
}

/// A prefill-decode fleet of one model, `side` workers a role over the gRPC wire, every
/// prefill paired with every decode, registered as the gateway registers them, with one
/// policy object a role: round robin, or cache-aware over one event index in which prompt
/// `p` is held by prefill `p` and by decode `p`. The prompts come with it.
fn pair_fleet(side: usize, cache_aware: bool) -> (PairPath, Vec<Vec<u32>>) {
    let registry = Arc::new(WorkerRegistry::new());
    let mut pools: Vec<Vec<Arc<dyn Worker>>> = Vec::new();
    for (host, role) in [
        ("prefill", WorkerType::Prefill),
        ("decode", WorkerType::Decode),
    ] {
        let pool: Vec<Arc<dyn Worker>> = (0..side)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://{host}-{i:03}:9000"))
                        .model(ModelCard::new(PAIR_MODEL))
                        .worker_type(role)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Vllm)
                        .kv_connector("NixlConnector")
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        for worker in &pool {
            registry.register(Arc::clone(worker)).unwrap();
        }
        pools.push(pool);
    }
    let prompts = placement_prompts(side);
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    if cache_aware {
        let monitor = holders_monitor(PAIR_MODEL, &pools, &prompts);
        let policy = |pool: &[Arc<dyn Worker>]| {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                eviction_interval_secs: 0,
                block_size: BLOCK_SIZE,
                ..Default::default()
            });
            policy.init_workers(pool);
            policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
            Arc::new(policy)
        };
        policies.set_prefill_policy(policy(&pools[0]));
        policies.set_decode_policy(policy(&pools[1]));
    } else {
        policies.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        policies.set_decode_policy(Arc::new(RoundRobinPolicy::new()));
    }
    (PairPath::new(registry, policies, PAIR_MODEL), prompts)
}

/// One pair placement through the routers' own pair selection, with no admission window: the
/// open prefills and the picked prefill's open partners gathered for the request, one policy
/// pick over each. Under round robin, and under cache-aware where each request's prompt has
/// one holder a role (an index lookup and a hit on both sides), at 64 and 512 workers a side,
/// untraced and inside a cache-trace capture.
fn bench_pair_placement(c: &mut Criterion) {
    refuse_a_process_trace("pair_placement");
    let mut group = c.benchmark_group("pair_placement");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for side in [64, 512] {
        for (policy, cache_aware) in [("round_robin", false), ("cache_aware", true)] {
            let (path, prompts) = pair_fleet(side, cache_aware);
            for (mode, traced) in [("untraced", false), ("traced", true)] {
                group.throughput(Throughput::Elements(1));
                let mut next = 0usize;
                group.bench_with_input(
                    BenchmarkId::new(format!("{policy}/{mode}"), side),
                    &side,
                    |b, _| {
                        b.iter(|| {
                            let tokens = &prompts[next % prompts.len()];
                            next += 1;
                            let pair =
                                runtime.block_on(path.place(Some(black_box(tokens)), traced));
                            black_box(pair.unwrap())
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_select,
    bench_gather_and_select,
    bench_least_load_decision,
    bench_least_load_after_report,
    bench_ring_policies,
    bench_simple_policy_pick,
    bench_keyed_sticky_hit,
    bench_regular_placement,
    bench_pair_placement
);
criterion_main!(benches);
