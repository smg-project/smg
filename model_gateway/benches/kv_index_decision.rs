//! T6, the routing decision's cost: what one request pays in `cache_aware`'s
//! `select_worker` (block hashing, the KV index lookup, the selection) against
//! an index populated to 128 workers, for the positional indexer and the chain
//! index (`--kv-index`).
//!
//! The index is fed through the KV event monitor's own apply path: the mock
//! engine's vLLM-shaped stream (`mock_streams`, the exactness tests' generator)
//! into every worker, then a synthetic fleet state shaped like Mooncake
//! conversation traffic (sessions of 8 to 512 blocks, most starting with one of
//! a few shared chat-template prefixes, each session resident on one to four
//! workers, every worker filled to its block budget). Requests are new turns on
//! resident sessions, prefixes of them, and novel prompts on a shared prefix.
//!
//! Two measurements per backend: the criterion sample of one decision, and the
//! contract's condition, a sustained loop at `T6_RATE` decisions per second
//! (default 10,000) for `T6_SECS` seconds (default 12, the first 2 discarded)
//! on whatever core the process is pinned to, reporting p50/p99/p999 over the
//! decisions with the lookup's and the hashing's own distributions beside
//! them. `T6_WORKERS` (128) and `T6_BLOCKS_PER_WORKER` (8192) size the fleet;
//! `T6_INDEX=positional|chain` runs one backend alone, which is how the index's
//! RSS delta is measured (the second backend in a process reuses the first's
//! freed pages and reads zero).
//!
//! Run with (one pinned core, outside the measurement set):
//!   taskset -c 100 cargo bench -p smg --bench kv_index_decision
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "benchmark code: panicking on setup failure is expected, eprintln is the report"
)]

use std::{
    fs,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::Criterion;
use kv_index::{compute_content_hash, compute_request_content_hashes, request_prefix_hashes};
use openai_protocol::worker::HealthCheckConfig;
use smg::{
    config::KvIndexKind,
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
    worker::{
        kv_event_monitor::bench_support::IndexFeed, BasicWorkerBuilder, KvEventMonitor, KvIndex,
        Worker, WorkerType,
    },
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch,
};

/// The exactness tests' stream generator, by path: the mock engine's streams
/// through the relay's decoder and normalizer.
#[path = "../src/worker/kv_index_backend/mock_streams.rs"]
#[expect(
    dead_code,
    reason = "the bench takes the payloads; the exactness tests the checkpoints and prompts too"
)]
mod mock_streams;

const BLOCK: usize = 16;
/// Workers whose model id is empty route under this key.
const MODEL: &str = "unknown";

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// SplitMix64: a deterministic fleet and request stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Log-uniform in `[lo, hi]`.
    fn log_uniform(&mut self, lo: usize, hi: usize) -> usize {
        let (lo_f, hi_f) = ((lo as f64).ln(), (hi as f64).ln());
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (lo_f + (hi_f - lo_f) * unit).exp().round() as usize
    }
}

/// Token ids of block `position` of stream `stream`: distinct per pair.
fn tokens(stream: u64, position: usize) -> Vec<u32> {
    (0..BLOCK as u32)
        .map(|i| {
            let word = stream
                .wrapping_mul(0x2545_f491_4f6c_dd1d)
                .wrapping_add(position as u64 * 0x9e37_79b9)
                .wrapping_add(u64::from(i));
            (word >> 7) as u32 & 0x3_ffff
        })
        .collect()
}

/// A resident chain: its token blocks and the engine hashes an engine would
/// publish (the chain hash of the contents so far, shared by every holder).
struct Chain {
    token_blocks: Vec<Vec<u32>>,
    hashes: Vec<i64>,
}

impl Chain {
    fn new(token_blocks: Vec<Vec<u32>>) -> Self {
        let contents: Vec<_> = token_blocks
            .iter()
            .map(|tokens| compute_content_hash(tokens))
            .collect();
        let hashes = request_prefix_hashes(&contents)
            .into_iter()
            .map(|prefix| prefix.0 as i64)
            .collect();
        Self {
            token_blocks,
            hashes,
        }
    }

    fn tokens(&self, blocks: usize) -> Vec<u32> {
        self.token_blocks[..blocks].concat()
    }

    /// Stores of at most 16 blocks each, chained by parent, as vLLM publishes.
    fn stores(&self) -> Vec<KvCacheEvent> {
        (0..self.token_blocks.len())
            .step_by(16)
            .map(|start| {
                let end = (start + 16).min(self.token_blocks.len());
                KvCacheEvent {
                    event_id: 0,
                    data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                        blocks: (start..end)
                            .map(|i| KvBlock {
                                block_hash: self.hashes[i],
                                token_ids: self.token_blocks[i].clone(),
                                block_size: BLOCK as i32,
                                ..Default::default()
                            })
                            .collect(),
                        parent_block_hash: (start > 0).then(|| self.hashes[start - 1]),
                        ..Default::default()
                    })),
                }
            })
            .collect()
    }
}

/// The fleet's sessions: shared chat-template prefixes, then conversations.
struct Sessions {
    prefixes: Vec<Vec<Vec<u32>>>,
    chains: Vec<Chain>,
}

impl Sessions {
    fn generate(rng: &mut Rng, count: usize) -> Self {
        let prefixes: Vec<Vec<Vec<u32>>> = (0..8)
            .map(|p| {
                let len = rng.log_uniform(16, 64);
                (0..len).map(|i| tokens(1_000 + p, i)).collect()
            })
            .collect();
        let chains = (0..count)
            .map(|s| {
                let mut token_blocks = if rng.below(10) < 6 {
                    prefixes[rng.below(prefixes.len())].clone()
                } else {
                    Vec::new()
                };
                let body = rng.log_uniform(8, 512);
                token_blocks.extend((0..body).map(|i| tokens(10_000 + s as u64, i)));
                Chain::new(token_blocks)
            })
            .collect();
        Self { prefixes, chains }
    }
}

/// The engine stream every worker starts from: the mock engine's vLLM-shaped
/// run, as the relay forwards it.
fn engine_batches() -> Vec<KvEventBatch> {
    mock_streams::Stream::generate(mock_streams::Shape::Vllm, 1, 320).normalized()
}

fn rss_mb() -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

/// A populated gateway: the policy with its monitor and index, the workers,
/// and the request stream.
struct Setup {
    policy: CacheAwarePolicy,
    index: Arc<KvIndex>,
    workers: Vec<Arc<dyn Worker>>,
    requests: Vec<Vec<u32>>,
    memberships: usize,
    index_mb: f64,
    fill_secs: f64,
}

fn setup(kind: KvIndexKind, worker_count: usize, blocks_per_worker: usize) -> Setup {
    let rss_before = rss_mb();
    let started = Instant::now();
    // One seed for both backends: the same fleet, the same requests.
    let mut rng = Rng(1);
    let sessions = Sessions::generate(&mut rng, worker_count * 48);
    let engine_stream = engine_batches();

    let index = Arc::new(KvIndex::new(kind, 64));
    let workers: Vec<Arc<dyn Worker>> = (0..worker_count)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://worker-{i:03}:9000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    // Every session lives on one to four workers; each worker is filled to
    // its budget from the sessions assigned to it, the engine stream first.
    let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); worker_count];
    for (s, _) in sessions.chains.iter().enumerate() {
        let holders = 1 + rng.below(4);
        for _ in 0..holders {
            assigned[rng.below(worker_count)].push(s);
        }
    }
    let mut memberships = 0usize;
    for (w, worker) in workers.iter().enumerate() {
        let mut feed = IndexFeed::new(&index, worker.url()).unwrap();
        for batch in &engine_stream {
            feed.apply(&index, batch);
        }
        let mut budget = blocks_per_worker;
        let mut sessions_here = assigned[w].clone();
        let mut next_unique = 0u64;
        while budget > 0 {
            let chain = match sessions_here.pop() {
                Some(s) => &sessions.chains[s],
                None => {
                    // A chain only this worker holds.
                    next_unique += 1;
                    let len = rng.log_uniform(8, 256).min(budget.max(8));
                    let stream = 1_000_000 + (w as u64) * 1_000_000 + next_unique;
                    let token_blocks = (0..len).map(|i| tokens(stream, i)).collect();
                    let chain = Chain::new(token_blocks);
                    feed.apply(
                        &index,
                        &KvEventBatch {
                            events: chain.stores(),
                            ..Default::default()
                        },
                    );
                    budget = budget.saturating_sub(len);
                    continue;
                }
            };
            feed.apply(
                &index,
                &KvEventBatch {
                    events: chain.stores(),
                    ..Default::default()
                },
            );
            budget = budget.saturating_sub(chain.token_blocks.len());
        }
        memberships += index.worker_block_count(feed.worker_id());
    }
    let index_mb = rss_mb() - rss_before;

    let monitor = Arc::new(KvEventMonitor::with_kind(kind, None));
    monitor.set_index(MODEL, Arc::clone(&index));
    monitor.set_block_size(MODEL, BLOCK);
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        block_size: BLOCK,
        ..Default::default()
    });
    policy.init_workers(&workers);
    policy.set_kv_event_monitor(Some(monitor));

    // Requests: a new turn on a resident session (its chain plus 1 to 16 new
    // blocks), a prefix of a resident session, or a novel prompt on a shared
    // chat-template prefix.
    let requests: Vec<Vec<u32>> = (0..8192)
        .map(|r| {
            let draw = rng.below(10);
            if draw < 7 {
                let chain = &sessions.chains[rng.below(sessions.chains.len())];
                let mut turn = chain.tokens(chain.token_blocks.len());
                let more = 1 + rng.below(16);
                for i in 0..more {
                    turn.extend(tokens(5_000_000 + r as u64, i));
                }
                turn
            } else if draw < 9 {
                let chain = &sessions.chains[rng.below(sessions.chains.len())];
                let cut = 1 + rng.below(chain.token_blocks.len());
                chain.tokens(cut)
            } else {
                let mut novel = sessions.prefixes[rng.below(sessions.prefixes.len())].concat();
                for i in 0..rng.log_uniform(4, 256) {
                    novel.extend(tokens(6_000_000 + r as u64, i));
                }
                novel
            }
        })
        .collect();
    Setup {
        policy,
        index,
        workers,
        requests,
        memberships,
        index_mb,
        fill_secs: started.elapsed().as_secs_f64(),
    }
}

fn percentile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[rank] as f64 / 1000.0
}

struct Dist {
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    max: f64,
    mean: f64,
    count: usize,
}

fn dist(mut samples: Vec<u64>) -> Dist {
    samples.sort_unstable();
    let count = samples.len();
    let mean = samples.iter().sum::<u64>() as f64 / count.max(1) as f64 / 1000.0;
    Dist {
        p50: percentile(&samples, 0.5),
        p90: percentile(&samples, 0.9),
        p99: percentile(&samples, 0.99),
        p999: percentile(&samples, 0.999),
        max: percentile(&samples, 1.0),
        mean,
        count,
    }
}

/// The contract's condition: one decision every `1 / rate` seconds on this
/// core, for `secs` seconds, the first two discarded as warm-up. Returns the
/// decision latencies and the achieved rate.
fn sustained(
    setup: &Setup,
    rate: u64,
    secs: u64,
    mut work: impl FnMut(&Setup, &[u32]),
) -> (Vec<u64>, f64) {
    let interval = Duration::from_nanos(1_000_000_000 / rate.max(1));
    let warmup = Duration::from_secs(2.min(secs));
    let total = Duration::from_secs(secs);
    let start = Instant::now();
    let mut next = start;
    let mut samples = Vec::with_capacity((rate * secs) as usize);
    let mut decisions = 0u64;
    let mut r = 0usize;
    loop {
        while Instant::now() < next {
            std::hint::spin_loop();
        }
        let now = Instant::now();
        if now - start >= total {
            break;
        }
        let request = &setup.requests[r % setup.requests.len()];
        r += 1;
        let t0 = Instant::now();
        work(setup, request);
        let elapsed = t0.elapsed();
        decisions += 1;
        if now - start >= warmup {
            samples.push(elapsed.as_nanos() as u64);
        }
        next += interval;
        if next < Instant::now() - interval {
            // Fell behind by more than a tick: resume the schedule from now.
            next = Instant::now();
        }
    }
    (samples, decisions as f64 / start.elapsed().as_secs_f64())
}

fn decide(setup: &Setup, tokens: &[u32]) {
    let info = SelectWorkerInfo {
        tokens: Some(tokens),
        ..Default::default()
    };
    black_box(setup.policy.select_worker(&setup.workers, &info));
}

fn lookup(setup: &Setup, tokens: &[u32]) {
    let hashes = compute_request_content_hashes(tokens, BLOCK);
    black_box(setup.index.find_matches(&hashes, false));
}

fn hashing(_: &Setup, tokens: &[u32]) {
    black_box(compute_request_content_hashes(tokens, BLOCK));
}

fn report(kind: KvIndexKind, setup: &Setup, rate: u64, secs: u64) {
    let (decision, achieved) = sustained(setup, rate, secs, decide);
    let (lookup_samples, _) = sustained(setup, rate, secs.min(4), lookup);
    let (hash_samples, _) = sustained(setup, rate, secs.min(4), hashing);
    let decision = dist(decision);
    let lookup = dist(lookup_samples);
    let hash = dist(hash_samples);
    let mut lens: Vec<usize> = setup.requests.iter().map(Vec::len).collect();
    lens.sort_unstable();
    eprintln!(
        "| {kind:?} | {workers} | {memberships} | {index_mb:.0} | {fill:.1} | {achieved:.0} | {n} | \
         {d50:.2} | {d90:.2} | {d99:.2} | {d999:.2} | {dmax:.1} | {l50:.2} | {l99:.2} | {lshare:.0}% | \
         {h50:.2} | {h99:.2} | {tok50} | {tok99} |",
        workers = setup.workers.len(),
        memberships = setup.memberships,
        index_mb = setup.index_mb,
        fill = setup.fill_secs,
        n = decision.count,
        d50 = decision.p50,
        d90 = decision.p90,
        d99 = decision.p99,
        d999 = decision.p999,
        dmax = decision.max,
        l50 = lookup.p50,
        l99 = lookup.p99,
        lshare = 100.0 * lookup.mean / decision.mean,
        h50 = hash.p50,
        h99 = hash.p99,
        tok50 = lens[lens.len() / 2],
        tok99 = lens[lens.len() * 99 / 100],
    );
}

fn main() {
    let workers = env_or("T6_WORKERS", 128) as usize;
    let blocks_per_worker = env_or("T6_BLOCKS_PER_WORKER", 8192) as usize;
    let rate = env_or("T6_RATE", 10_000);
    let secs = env_or("T6_SECS", 12);
    // A Prometheus recorder, so the decision pays for its metrics as in the gateway.
    let _recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("recorder");
    let mut criterion = Criterion::default().configure_from_args();
    eprintln!(
        "| index | workers | memberships | index MB | fill s | achieved/s | decisions | dec p50 us | p90 | p99 | p999 | max | \
         lookup p50 | lookup p99 | lookup share | hash p50 | hash p99 | req tokens p50 | p99 |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    // `T6_INDEX=positional|chain` runs one backend in its own process: the second
    // backend in one process reuses the pages the first freed, so its RSS
    // delta reads zero.
    let only = std::env::var("T6_INDEX")
        .ok()
        .and_then(|value| KvIndexKind::parse(&value));
    for kind in [KvIndexKind::Positional, KvIndexKind::Chain] {
        if only.is_some_and(|only| only != kind) {
            continue;
        }
        let setup = setup(kind, workers, blocks_per_worker);
        let mut r = 0usize;
        criterion.bench_function(&format!("decision/{kind:?}"), |b| {
            b.iter(|| {
                let request = &setup.requests[r % setup.requests.len()];
                r += 1;
                decide(&setup, request);
            });
        });
        report(kind, &setup, rate, secs);
        drop(setup);
    }
    criterion.final_summary();
}
