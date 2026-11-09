//! T6, the routing decision's cost: what one request pays in `cache_aware`'s
//! `select_worker` (block hashing, the KV index lookup, the selection) against
//! an index populated to 128 workers.
//!
//! The index is fed through the KV event monitor's own apply path: the mock
//! engine's vLLM-shaped stream (`mock_streams`, the exactness tests' generator)
//! into every worker, then a synthetic fleet state shaped like Mooncake
//! conversation traffic (sessions of 8 to 512 blocks, most starting with one of
//! a few shared chat-template prefixes, each session resident on one to four
//! workers, every worker filled to its block budget). Requests are new turns on
//! resident sessions, prefixes of them, and novel prompts on a shared prefix.
//!
//! Two measurements: the criterion sample of one decision, and the
//! contract's condition, a sustained loop at `T6_RATE` decisions per second
//! (default 10,000) for `T6_SECS` seconds (default 12, the first 2 discarded)
//! on whatever core the process is pinned to, reporting p50/p99/p999 over the
//! decisions with the lookup's and the hashing's own distributions beside
//! them. `T6_WORKERS` (128) and `T6_BLOCKS_PER_WORKER` (8192) size the fleet.
//!
//! The decision's configuration comes from the environment too, one table row
//! per run. `T6_SELECTION` names the selection policy (`cache-aware-default`).
//! `T6_TEMPERATURE`, `T6_DECAY` and `T6_ACCOUNTING_MS` set
//! `--selection-temperature`, `--overlap-decay` and
//! `--selection-accounting-ttl-ms` (all 0). `T6_FEED=1` publishes a load
//! report for every worker and puts requests in flight on them, so the
//! expected wait scores the candidates as it does over engines that report.
//! `T6_INDEX=tree` routes on the approximate token tree of a fleet without KV
//! events, filled by routing every session once. `T6_HYBRID=1` makes every
//! worker a hybrid model's (a sliding-window tail and a state block stored
//! beside each chain), so every holder's overlap passes the cache groups'
//! rule in the decision. `T6_THREADS` (1) runs that
//! many decision loops on the one policy, each at `T6_RATE`, the row taking
//! the latencies of all of them. A configuration other than the default has
//! its own criterion id (`decision/chain/temperature-0.1/fed`), so criterion's
//! change report compares runs of one configuration.
//!
//! Run with (one pinned core, outside the measurement set; a core per loop
//! with `T6_THREADS`):
//!   taskset -c 100 cargo bench -p smg --bench kv_index_decision
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "benchmark code: panicking on setup failure is expected, eprintln is the report"
)]

use std::{
    collections::HashMap,
    fs,
    hint::black_box,
    str::FromStr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use criterion::Criterion;
use kv_index::{compute_content_hash, compute_request_content_hashes, request_prefix_hashes};
use openai_protocol::worker::{HealthCheckConfig, WorkerLoadResponse};
use smg::{
    policies::{cost, CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
    worker::{
        kv_event_monitor::bench_support::IndexFeed,
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, KvEventMonitor, KvIndex, Worker, WorkerType,
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

fn env_or<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The decision under measurement, read from the environment (see the
/// module docs).
struct Variant {
    selection: String,
    temperature: f32,
    decay: f32,
    accounting_ms: u64,
    feed: bool,
    tree: bool,
    threads: usize,
    /// Every worker is a hybrid model's: beside each chain's main-group
    /// stores, its sliding-window group's tail and its state group's last
    /// block, so every holder's overlap passes the cache groups' rule.
    hybrid: bool,
}

impl Variant {
    fn from_env() -> Self {
        let selection = env_or("T6_SELECTION", cost::DEFAULT_POLICY.to_string());
        let temperature = env_or("T6_TEMPERATURE", 0.0);
        // The policy would fall back to the default on an unknown name.
        cost::build(&selection, temperature).expect("T6_SELECTION names a selection policy");
        let index = env_or("T6_INDEX", "chain".to_string());
        let tree = ["chain", "tree"]
            .iter()
            .position(|known| *known == index)
            .expect("T6_INDEX is chain or tree")
            == 1;
        Self {
            selection,
            temperature,
            decay: env_or("T6_DECAY", 0.0),
            accounting_ms: env_or("T6_ACCOUNTING_MS", 0),
            feed: env_or("T6_FEED", 0u8) != 0,
            tree,
            threads: env_or("T6_THREADS", 1usize).max(1),
            hybrid: env_or("T6_HYBRID", 0u8) != 0,
        }
    }

    fn index(&self) -> &'static str {
        if self.tree {
            "tree"
        } else {
            "chain"
        }
    }

    /// The criterion id: `decision/chain` (or `decision/tree`) for the
    /// default policy configuration, as before, with each setting that
    /// differs from it appended, so criterion compares a run with the last
    /// run of the same configuration.
    fn criterion_id(&self) -> String {
        let mut id = format!("decision/{}", self.index());
        if self.selection != cost::DEFAULT_POLICY {
            id.push_str(&format!("/{}", self.selection));
        }
        if self.temperature > 0.0 {
            id.push_str(&format!("/temperature-{}", self.temperature));
        }
        if self.decay > 0.0 {
            id.push_str(&format!("/decay-{}", self.decay));
        }
        if self.accounting_ms > 0 {
            id.push_str(&format!("/accounting-{}ms", self.accounting_ms));
        }
        if self.feed {
            id.push_str("/fed");
        }
        if self.hybrid {
            id.push_str("/hybrid");
        }
        id
    }
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

    /// The chain's stores and, for a hybrid model's worker, the stores of its
    /// other cache groups after them.
    fn events(&self, variant: &Variant) -> Vec<KvCacheEvent> {
        let mut events = self.stores();
        if variant.hybrid {
            events.extend(self.group_stores());
        }
        events
    }

    /// The stores of a hybrid model's other cache groups for this chain, as
    /// an engine publishes them under its default retention: the
    /// sliding-window group (four blocks of window) keeps the four blocks
    /// ending at the chain's end, the state group the last block; hashes
    /// only, no tokens.
    fn group_stores(&self) -> Vec<KvCacheEvent> {
        let len = self.token_blocks.len();
        let store = |group: u32, kind: &str, window: Option<u32>, from: usize| KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: (from..len)
                    .map(|i| KvBlock {
                        block_hash: self.hashes[i],
                        token_ids: Vec::new(),
                        block_size: BLOCK as i32,
                        ..Default::default()
                    })
                    .collect(),
                group_idx: Some(group),
                kv_cache_spec_kind: Some(kind.to_string()),
                kv_cache_spec_sliding_window: window,
                ..Default::default()
            })),
        };
        vec![
            store(
                1,
                "sliding_window",
                Some(4 * BLOCK as u32),
                len.saturating_sub(4),
            ),
            store(2, "mamba", None, len.saturating_sub(1)),
        ]
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
    /// The published load reports, while the run lasts (`T6_FEED`).
    _feed: Option<TestFeed>,
}

/// A load report for every worker (queued token-work, running requests, KV
/// pressure, a live rate on three of four) and up to three requests in flight
/// on each, as `policy_selection`'s fed pool has them.
fn feed_loads(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], rng: &mut Rng) -> TestFeed {
    let feed = TestFeed::attached_to(policy);
    let reports: HashMap<String, WorkerLoadResponse> = workers
        .iter()
        .map(|worker| {
            let live_rate = if rng.below(4) == 0 {
                0.0
            } else {
                500.0 + rng.below(3_500) as f64
            };
            let mut report = one_rank_load(
                rng.below(8_192) as i32,
                rng.below(95) as f64 / 100.0,
                live_rate,
            );
            report.loads[0].num_running_reqs = rng.below(12) as i32;
            for _ in 0..rng.below(4) {
                worker.increment_load();
            }
            (worker.url().to_string(), report)
        })
        .collect();
    feed.publish(&reports);
    feed
}

fn setup(worker_count: usize, blocks_per_worker: usize, variant: &Variant) -> Setup {
    let rss_before = rss_mb();
    let started = Instant::now();
    // One seed: the same fleet and the same requests, run to run.
    let mut rng = Rng(1);
    let sessions = Sessions::generate(&mut rng, worker_count * 48);
    let engine_stream = engine_batches();

    let index = Arc::new(KvIndex::chain());
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
    // A registered worker carries a published state, which the routing reads
    // answer from; the fleet reads like one.
    for worker in &workers {
        worker.republish_state();
    }
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
                            events: chain.events(variant),
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
                    events: chain.events(variant),
                    ..Default::default()
                },
            );
            budget = budget.saturating_sub(chain.token_blocks.len());
        }
        memberships += index.worker_block_count(feed.worker_id());
    }
    let mut index_mb = rss_mb() - rss_before;

    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        block_size: BLOCK,
        overlap_decay: variant.decay,
        selection_temperature: variant.temperature,
        selection_policy: Some(variant.selection.clone()),
        selection_accounting_ttl_ms: variant.accounting_ms,
        ..Default::default()
    });
    policy.init_workers(&workers);
    if variant.tree {
        // The index is filled all the same, so both fleets draw the same
        // requests; the tree fleet's decision never reads it. Every session
        // routed once: the tree holds each where the policy placed it, as a
        // fleet without KV events learns its sessions.
        let rss_before_tree = rss_mb();
        memberships = 0;
        for chain in &sessions.chains {
            let tokens = chain.tokens(chain.token_blocks.len());
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            };
            black_box(policy.select_worker(&workers, &info));
        }
        index_mb = rss_mb() - rss_before_tree;
    } else {
        let monitor = Arc::new(KvEventMonitor::new());
        monitor.set_index(MODEL, Arc::clone(&index));
        monitor.set_block_size(MODEL, BLOCK);
        policy.set_kv_event_monitor(Some(monitor));
    }
    // Its own draws, so a fed run decides over the same requests.
    let feed = variant
        .feed
        .then(|| feed_loads(&policy, &workers, &mut Rng(2)));

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
        _feed: feed,
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
    offset: usize,
    mut work: impl FnMut(&Setup, &[u32]),
) -> (Vec<u64>, f64) {
    let interval = Duration::from_nanos(1_000_000_000 / rate.max(1));
    let warmup = Duration::from_secs(2.min(secs));
    let total = Duration::from_secs(secs);
    let start = Instant::now();
    let mut next = start;
    let mut samples = Vec::with_capacity((rate * secs) as usize);
    let mut decisions = 0u64;
    let mut r = offset;
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

/// [`sustained`] decisions on `threads` loops at once over the one policy,
/// each at `rate` and from its own place in the request stream: the
/// latencies of all of them, and the rate they achieved together.
fn sustained_decisions(setup: &Setup, threads: usize, rate: u64, secs: u64) -> (Vec<u64>, f64) {
    if threads == 1 {
        return sustained(setup, rate, secs, 0, decide);
    }
    thread::scope(|scope| {
        let loops: Vec<_> = (0..threads)
            .map(|t| scope.spawn(move || sustained(setup, rate, secs, t * 977, decide)))
            .collect();
        let mut samples = Vec::new();
        let mut achieved = 0.0;
        for run in loops {
            let (latencies, rate) = run.join().unwrap();
            samples.extend(latencies);
            achieved += rate;
        }
        (samples, achieved)
    })
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

fn report(setup: &Setup, variant: &Variant, rate: u64, secs: u64) {
    let (decision, achieved) = sustained_decisions(setup, variant.threads, rate, secs);
    let decision = dist(decision);
    // The index lookup is the chain fleet's; the tree fleet's is inside the
    // decision.
    let lookup = if variant.tree {
        "- | - | -".to_string()
    } else {
        let lookup = dist(sustained(setup, rate, secs.min(4), 0, lookup).0);
        format!(
            "{:.2} | {:.2} | {:.0}%",
            lookup.p50,
            lookup.p99,
            100.0 * lookup.mean / decision.mean
        )
    };
    let hash = dist(sustained(setup, rate, secs.min(4), 0, hashing).0);
    let mut lens: Vec<usize> = setup.requests.iter().map(Vec::len).collect();
    lens.sort_unstable();
    eprintln!(
        "| {index} | {selection} | {temperature} | {decay} | {accounting} | {feed} | {threads} | \
         {workers} | {memberships} | {index_mb:.0} | {fill:.1} | {achieved:.0} | {n} | \
         {d50:.2} | {d90:.2} | {d99:.2} | {d999:.2} | {dmax:.1} | {lookup} | \
         {h50:.2} | {h99:.2} | {tok50} | {tok99} |",
        index = variant.index(),
        selection = variant.selection,
        temperature = variant.temperature,
        decay = variant.decay,
        accounting = variant.accounting_ms,
        feed = if variant.feed { "fed" } else { "dark" },
        threads = variant.threads,
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
        h50 = hash.p50,
        h99 = hash.p99,
        tok50 = lens[lens.len() / 2],
        tok99 = lens[lens.len() * 99 / 100],
    );
}

fn main() {
    let workers = env_or("T6_WORKERS", 128usize);
    let blocks_per_worker = env_or("T6_BLOCKS_PER_WORKER", 8192usize);
    let rate = env_or("T6_RATE", 10_000u64);
    let secs = env_or("T6_SECS", 12u64);
    let variant = Variant::from_env();
    // A Prometheus recorder, so the decision pays for its metrics as in the gateway.
    let _recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("recorder");
    let mut criterion = Criterion::default().configure_from_args();
    eprintln!(
        "| index | selection | temperature | decay | accounting ms | loads | threads | workers | memberships | \
         index MB | fill s | achieved/s | decisions | dec p50 us | p90 | p99 | p999 | max | \
         lookup p50 | lookup p99 | lookup share | hash p50 | hash p99 | req tokens p50 | p99 |"
    );
    eprintln!(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
    let setup = setup(workers, blocks_per_worker, &variant);
    let mut r = 0usize;
    criterion.bench_function(&variant.criterion_id(), |b| {
        b.iter(|| {
            let request = &setup.requests[r % setup.requests.len()];
            r += 1;
            decide(&setup, request);
        });
    });
    report(&setup, &variant, rate, secs);
    drop(setup);
    criterion.final_summary();
}
