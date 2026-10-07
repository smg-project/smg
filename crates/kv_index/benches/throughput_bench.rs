//! Trace-driven throughput benchmark for the chain index.
//!
//! Measures block throughput (blocks/sec):
//! - Synthetic trace generation with shared prefixes and multi-turn sessions
//! - Concurrent tokio task replay with timing-accurate pacing
//! - Block throughput metric: total_blocks = request_blocks + event_blocks
//! - Sweep mode: compress trace into progressively shorter durations to find peak
//!
//! Run: `cargo bench -p kv-index --bench throughput_bench -- --help`

// Benchmark binary — println is the intended output mechanism.
#![expect(clippy::print_stdout)]
// Benchmark binary — panicking on task join failure is acceptable.
#![expect(clippy::expect_used)]
// Benchmark binary — tokio::spawn is required for concurrent benchmark replay.
#![expect(clippy::disallowed_methods)]

use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use clap::Parser;
use kv_index::{
    compute_content_hash, ChainBlockMap, ContentHash, OverlapScores, SequenceHash,
    ShardedChainIndex, StoredBlock,
};
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Worker slots per chain index shard: the index's own ceiling.
const CHAIN_WORKERS_PER_SHARD: usize = 1024;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "throughput_bench",
    about = "Block throughput benchmark for the chain index"
)]
struct Args {
    /// Ignored — passed by `cargo bench` harness.
    #[arg(long, hide = true)]
    bench: bool,

    /// Number of workers (concurrent replay tasks). Must be >= 1.
    #[arg(long, default_value_t = 256)]
    num_workers: usize,

    /// Number of content-hash blocks per find_matches request.
    #[arg(long, default_value_t = 128)]
    blocks_per_request: usize,

    /// Number of blocks drawn from shared prefix pool per request.
    #[arg(long, default_value_t = 32)]
    shared_prefix_blocks: usize,

    /// Number of blocks per apply_stored event.
    #[arg(long, default_value_t = 64)]
    event_blocks: usize,

    /// Number of synthetic conversation sessions.
    /// Higher values generate more total blocks, needed to saturate the indexer.
    /// With 200K sessions × 5 turns × (128 + 64) blocks = 192M total blocks.
    #[arg(long, default_value_t = 200_000)]
    num_sessions: usize,

    /// Turns per session (multi-turn conversation depth).
    #[arg(long, default_value_t = 5)]
    requests_per_session: usize,

    /// Base benchmark duration in milliseconds (used for sweep max).
    #[arg(long, default_value_t = 60_000)]
    benchmark_duration_ms: u64,

    /// Disable sweep mode (run a single duration instead of peak-finding).
    #[arg(long)]
    no_sweep: bool,

    /// Minimum sweep duration in milliseconds.
    #[arg(long, default_value_t = 10)]
    sweep_min_ms: u64,

    /// Number of sweep steps.
    #[arg(long, default_value_t = 20)]
    sweep_steps: usize,

    /// Exclude event blocks from throughput (only count request blocks).
    #[arg(long)]
    no_count_events: bool,

    /// Worker duplication factor (simulates TP replicas).
    #[arg(long, default_value_t = 1)]
    duplication_factor: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Path to a Mooncake JSONL trace file. When provided, replaces synthetic generation.
    /// Format: one JSON object per line with at minimum a "hash_ids" array of u64.
    /// Optional fields: "timestamp" (ms), "output_length".
    #[arg(long)]
    trace_path: Option<String>,

    /// Factor to stretch each trace request's hash sequence length.
    #[arg(long, default_value_t = 1)]
    trace_length_factor: usize,

    /// How many times to duplicate the raw trace with offset hash_ids.
    #[arg(long, default_value_t = 1)]
    trace_duplication_factor: usize,
}

// ---------------------------------------------------------------------------
// Trace types
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum TraceEntry {
    Request {
        content_hashes: Vec<ContentHash>,
    },
    Event {
        blocks: Vec<StoredBlock>,
        parent_seq_hash: Option<SequenceHash>,
    },
}

#[derive(Clone)]
struct TimedEntry {
    timestamp_us: u64,
    entry: TraceEntry,
}

/// Per-task state for processing trace entries against the indexer.
struct TaskState {
    worker_blocks: ChainBlockMap,
    req_blocks: u64,
    evt_blocks: u64,
    errors: u64,
    latencies: Vec<u64>,
    count_events: bool,
}

impl TaskState {
    fn new(count_events: bool) -> Self {
        Self {
            worker_blocks: ChainBlockMap::default(),
            req_blocks: 0,
            evt_blocks: 0,
            errors: 0,
            latencies: Vec::new(),
            count_events,
        }
    }

    #[inline]
    fn process(&mut self, entry: &TraceEntry, indexer: &ShardedChainIndex, worker_id: u32) {
        match entry {
            TraceEntry::Request { content_hashes } => {
                let start = Instant::now();
                let _scores: OverlapScores = indexer.find_matches(content_hashes, false);
                let elapsed_ns = start.elapsed().as_nanos() as u64;
                self.latencies.push(elapsed_ns);
                self.req_blocks += content_hashes.len() as u64;
            }
            TraceEntry::Event {
                blocks,
                parent_seq_hash,
            } => {
                if indexer
                    .apply_stored(worker_id, blocks, *parent_seq_hash, &mut self.worker_blocks)
                    .is_ok()
                {
                    if self.count_events {
                        self.evt_blocks += blocks.len() as u64;
                    }
                } else {
                    self.errors += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic trace generation
// ---------------------------------------------------------------------------

fn generate_shared_prefix_pool(rng: &mut StdRng, pool_size: usize) -> Vec<ContentHash> {
    (0..pool_size)
        .map(|i| {
            let token_id = rng.random::<u32>().wrapping_add(i as u32);
            compute_content_hash(&[token_id])
        })
        .collect()
}

/// Generate per-worker traces with merged request + event timelines.
fn generate_traces(args: &Args) -> Vec<Vec<TimedEntry>> {
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Build a shared prefix pool that requests draw from.
    let prefix_pool = generate_shared_prefix_pool(&mut rng, args.shared_prefix_blocks * 10);

    let total_requests = args.num_sessions * args.requests_per_session;
    // Spread requests evenly across the benchmark duration.
    let inter_arrival_us = if total_requests > 1 {
        (args.benchmark_duration_ms * 1000) / total_requests as u64
    } else {
        0
    };

    // Assign sessions round-robin to workers.
    let mut worker_traces: Vec<Vec<TimedEntry>> =
        (0..args.num_workers).map(|_| Vec::new()).collect();

    let mut global_seq_counter: u64 = 0;
    let mut timestamp_us: u64 = 0;

    for session_id in 0..args.num_sessions {
        let worker_id = session_id % args.num_workers;

        // Session-specific prefix: draw a contiguous slice from the pool.
        let prefix_range = prefix_pool.len().saturating_sub(args.shared_prefix_blocks);
        let prefix_start = if prefix_range == 0 {
            0
        } else {
            (session_id * 7) % prefix_range
        };
        let prefix_end = (prefix_start + args.shared_prefix_blocks).min(prefix_pool.len());
        let session_prefix = &prefix_pool[prefix_start..prefix_end];

        let mut prev_event_last_seq: Option<SequenceHash> = None;

        for turn in 0..args.requests_per_session {
            // --- Request (find_matches) ---
            let mut content_hashes = Vec::with_capacity(args.blocks_per_request);

            // Shared prefix portion grows with turn depth (simulating multi-turn cache hits)
            let prefix_blocks = session_prefix
                .len()
                .min(args.blocks_per_request)
                .min((turn + 1) * (args.shared_prefix_blocks / args.requests_per_session).max(1));
            content_hashes.extend_from_slice(&session_prefix[..prefix_blocks]);

            // Fill remaining with unique hashes
            while content_hashes.len() < args.blocks_per_request {
                let token_id = rng.random::<u32>();
                content_hashes.push(compute_content_hash(&[token_id]));
            }

            worker_traces[worker_id].push(TimedEntry {
                timestamp_us,
                entry: TraceEntry::Request { content_hashes },
            });

            // --- Event (apply_stored) shortly after the request ---
            timestamp_us += inter_arrival_us / 4;

            let blocks: Vec<StoredBlock> = (0..args.event_blocks)
                .map(|_| {
                    global_seq_counter += 1;
                    StoredBlock {
                        seq_hash: SequenceHash(global_seq_counter),
                        content_hash: compute_content_hash(&[rng.random::<u32>()]),
                    }
                })
                .collect();

            let parent = if turn > 0 { prev_event_last_seq } else { None };

            prev_event_last_seq = blocks.last().map(|b| b.seq_hash);

            worker_traces[worker_id].push(TimedEntry {
                timestamp_us,
                entry: TraceEntry::Event {
                    blocks,
                    parent_seq_hash: parent,
                },
            });

            timestamp_us += inter_arrival_us - (inter_arrival_us / 4);
        }
    }

    for trace in &mut worker_traces {
        trace.sort_by_key(|e| e.timestamp_us);
    }

    worker_traces
}

// ---------------------------------------------------------------------------
// Mooncake trace loading
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Clone)]
struct MooncakeRequest {
    #[serde(default)]
    timestamp: u64,
    hash_ids: Vec<u64>,
    #[expect(dead_code)]
    #[serde(default)]
    output_length: u64,
}

/// Convert a Mooncake hash_id (u64) to our ContentHash.
///
/// Passes both halves of the u64 as two u32 tokens to avoid truncation
/// when hash_ids exceed u32 range after expand/duplicate transforms.
fn content_hash_from_id(id: u64) -> ContentHash {
    compute_content_hash(&[id as u32, (id >> 32) as u32])
}

fn load_mooncake_trace(path: &str) -> Vec<MooncakeRequest> {
    let file = File::open(path).expect("failed to open trace file");
    let reader = BufReader::new(file);
    let mut requests = Vec::new();
    for line in reader.lines() {
        let line = line.expect("failed to read trace line");
        if line.trim().is_empty() {
            continue;
        }
        let req: MooncakeRequest = serde_json::from_str(&line).expect("failed to parse trace line");
        requests.push(req);
    }
    requests
}

/// Stretch each request's hash sequence by the given factor.
///
/// Each hash `h` becomes `factor` consecutive hashes:
/// `h * factor`, `h * factor + 1`, ..., `h * factor + (factor - 1)`.
fn expand_trace_lengths(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }
    requests
        .into_iter()
        .map(|mut req| {
            req.hash_ids = req
                .hash_ids
                .iter()
                .flat_map(|&h| {
                    let base = h * factor as u64;
                    (0..factor as u64).map(move |offset| base + offset)
                })
                .collect();
            req
        })
        .collect()
}

/// Duplicate traces with offset hash_ids, creating `factor` structurally
/// identical copies of the prefix tree with disjoint hash spaces.
fn duplicate_traces(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }
    let max_hash_id = requests
        .iter()
        .flat_map(|r| r.hash_ids.iter().copied())
        .max()
        .unwrap_or(0);
    let offset_base = max_hash_id + 1;
    let mut out = Vec::with_capacity(requests.len() * factor);
    for r in &requests {
        for d in 0..factor {
            let offset = offset_base * d as u64;
            out.push(MooncakeRequest {
                hash_ids: r.hash_ids.iter().map(|&h| h + offset).collect(),
                ..r.clone()
            });
        }
    }
    out
}

/// Randomly partition a flat request list across `num_workers` worker buckets.
fn partition_trace(
    requests: Vec<MooncakeRequest>,
    num_workers: usize,
    seed: u64,
) -> Vec<Vec<MooncakeRequest>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut traces: Vec<Vec<MooncakeRequest>> = (0..num_workers).map(|_| Vec::new()).collect();
    for req in requests {
        traces[rng.random_range(0..num_workers)].push(req);
    }
    traces
}

/// Convert Mooncake per-worker request traces into our TimedEntry format.
///
/// For each request:
/// - Request entry: `find_matches` with content_hashes derived from hash_ids
/// - Event entry: `apply_stored` with StoredBlocks matching those content_hashes,
///   populating the cache so future requests with overlapping prefixes find matches
fn convert_mooncake_traces(worker_traces: Vec<Vec<MooncakeRequest>>) -> Vec<Vec<TimedEntry>> {
    let global_min_ts = worker_traces
        .iter()
        .flat_map(|t| t.iter())
        .map(|r| r.timestamp)
        .min()
        .unwrap_or(0);

    let mut result = Vec::with_capacity(worker_traces.len());

    for (worker_idx, requests) in worker_traces.into_iter().enumerate() {
        let mut entries = Vec::with_capacity(requests.len() * 2);
        // Use worker_idx in high bits to ensure globally unique seq_hashes.
        let mut seq_counter: u64 = (worker_idx as u64) << 40;
        let mut prev_last_seq: Option<SequenceHash> = None;

        for req in &requests {
            // Mooncake timestamps are in milliseconds; convert to microseconds.
            let timestamp_us = req.timestamp.saturating_sub(global_min_ts) * 1000;

            let content_hashes: Vec<ContentHash> = req
                .hash_ids
                .iter()
                .map(|&id| content_hash_from_id(id))
                .collect();

            // Request: find_matches
            entries.push(TimedEntry {
                timestamp_us,
                entry: TraceEntry::Request {
                    content_hashes: content_hashes.clone(),
                },
            });

            // Event: apply_stored (populate cache for future prefix matches)
            let blocks: Vec<StoredBlock> = content_hashes
                .iter()
                .map(|&ch| {
                    seq_counter += 1;
                    StoredBlock {
                        seq_hash: SequenceHash(seq_counter),
                        content_hash: ch,
                    }
                })
                .collect();

            let parent = prev_last_seq;
            prev_last_seq = blocks.last().map(|b| b.seq_hash);

            entries.push(TimedEntry {
                timestamp_us: timestamp_us + 1,
                entry: TraceEntry::Event {
                    blocks,
                    parent_seq_hash: parent,
                },
            });
        }

        entries.sort_by_key(|e| e.timestamp_us);
        result.push(entries);
    }

    result
}

/// Rescale trace timestamps to fit within a target duration.
fn rescale_traces(traces: &[Vec<TimedEntry>], target_duration_ms: u64) -> Vec<Vec<TimedEntry>> {
    let max_ts = traces
        .iter()
        .flat_map(|t| t.iter())
        .map(|e| e.timestamp_us)
        .max()
        .unwrap_or(1);

    if max_ts == 0 {
        return traces.to_vec();
    }

    let target_us = target_duration_ms * 1000;
    let scale = target_us as f64 / max_ts as f64;

    traces
        .iter()
        .map(|worker_trace| {
            worker_trace
                .iter()
                .map(|entry| TimedEntry {
                    timestamp_us: (entry.timestamp_us as f64 * scale) as u64,
                    entry: entry.entry.clone(),
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct BenchmarkResults {
    total_request_blocks: u64,
    total_event_blocks: u64,
    total_blocks: u64,
    total_errors: u64,
    actual_duration: Duration,
    block_throughput: f64,
    offered_block_throughput: f64,
    latency_p99_us: f64,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    assert!(args.num_workers > 0, "--num-workers must be >= 1");
    let sweep = !args.no_sweep;

    println!("=== Throughput Benchmark ===");
    if let Some(ref path) = args.trace_path {
        println!(
            "Trace: {path} | Workers: {}, Seed: {}",
            args.num_workers, args.seed,
        );
        if args.trace_length_factor > 1 || args.trace_duplication_factor > 1 {
            println!(
                "Trace transforms: length {}x, duplication {}x",
                args.trace_length_factor, args.trace_duplication_factor,
            );
        }
    } else {
        println!(
            "Workers: {}, Blocks/req: {}, Event blocks: {}, Duplication: {}x",
            args.num_workers, args.blocks_per_request, args.event_blocks, args.duplication_factor,
        );
        println!(
            "Sessions: {}, Turns/session: {}, Seed: {}",
            args.num_sessions, args.requests_per_session, args.seed,
        );
    }

    let gen_start = Instant::now();
    let base_traces = if let Some(ref path) = args.trace_path {
        let path = path.as_str();
        println!("\nLoading Mooncake trace from {path}...");
        let requests = load_mooncake_trace(path);
        println!("  Loaded {} requests", requests.len());
        let requests = expand_trace_lengths(requests, args.trace_length_factor);
        let requests = duplicate_traces(requests, args.trace_duplication_factor);
        let worker_traces = partition_trace(requests, args.num_workers, args.seed);
        convert_mooncake_traces(worker_traces)
    } else {
        println!("\nGenerating synthetic traces...");
        generate_traces(&args)
    };
    let gen_elapsed = gen_start.elapsed();

    let total_entries: usize = base_traces.iter().map(|t| t.len()).sum();
    let total_req_blocks: u64 = base_traces
        .iter()
        .flat_map(|t| t.iter())
        .filter_map(|e| match &e.entry {
            TraceEntry::Request { content_hashes } => Some(content_hashes.len() as u64),
            TraceEntry::Event { .. } => None,
        })
        .sum();
    let total_evt_blocks: u64 = base_traces
        .iter()
        .flat_map(|t| t.iter())
        .filter_map(|e| match &e.entry {
            TraceEntry::Event { blocks, .. } => Some(blocks.len() as u64),
            TraceEntry::Request { .. } => None,
        })
        .sum();

    println!(
        "Generated {total_entries} trace entries in {:.1}ms ({} request blocks, {} event blocks)",
        gen_elapsed.as_secs_f64() * 1000.0,
        total_req_blocks * args.duplication_factor as u64,
        total_evt_blocks * args.duplication_factor as u64,
    );

    if sweep {
        run_sweep(&args, &base_traces).await;
    } else {
        let traces = rescale_traces(&base_traces, args.benchmark_duration_ms);
        let result = run_benchmark(&args, traces).await;
        print_results(&result);
    }
}

async fn run_sweep(args: &Args, base_traces: &[Vec<TimedEntry>]) {
    let durations = compute_sweep_durations(
        args.sweep_min_ms,
        args.benchmark_duration_ms,
        args.sweep_steps,
    );
    let durations_high_to_low: Vec<u64> = durations.into_iter().rev().collect();

    let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();

    for &dur_ms in &durations_high_to_low {
        println!("\n--- Sweep: benchmark_duration_ms = {dur_ms} ---");

        let traces = rescale_traces(base_traces, dur_ms);
        let result = run_benchmark(args, traces).await;
        print_results(&result);
        results.push((dur_ms, result));
    }

    println!("\n{}", "=".repeat(80));
    println!("SWEEP SUMMARY");
    println!("{}", "=".repeat(80));
    println!(
        "{:<12} | {:<18} | {:<18}",
        "Duration", "Offered", "Block Throughput"
    );
    println!("{}", "-".repeat(60));

    let mut peak_throughput = 0.0f64;
    let mut peak_dur = 0u64;

    for (dur_ms, result) in &results {
        let dur_label = if *dur_ms >= 1000 {
            format!("{:.1}s", *dur_ms as f64 / 1000.0)
        } else {
            format!("{dur_ms}ms")
        };

        let is_peak = result.block_throughput > peak_throughput;
        if is_peak {
            peak_throughput = result.block_throughput;
            peak_dur = *dur_ms;
        }

        println!(
            "{:<12} | {:<18} | {:<18}{}",
            dur_label,
            format_throughput(result.offered_block_throughput),
            format_throughput(result.block_throughput),
            if is_peak && results.len() > 1 {
                " <-- peak"
            } else {
                ""
            },
        );
    }

    println!("{}", "-".repeat(60));
    println!(
        "Peak: {} block ops/sec at {peak_dur}ms duration",
        format_throughput(peak_throughput),
    );
}

async fn run_benchmark(args: &Args, traces: Vec<Vec<TimedEntry>>) -> BenchmarkResults {
    // A chain index shard holds up to 1,024 workers and the replay interns one
    // worker per task, so the index gets as many shards as the tasks need; the
    // gateway's own index is this type at one shard.
    let num_total_workers = args.num_workers * args.duplication_factor;
    let shards = num_total_workers.div_ceil(CHAIN_WORKERS_PER_SHARD).max(1);
    let indexer = Arc::new(ShardedChainIndex::new(shards, CHAIN_WORKERS_PER_SHARD));
    for w in 0..num_total_workers {
        indexer
            .intern_worker(&format!("worker-{w}"))
            .expect("worker id space exhausted");
    }

    let traces: Vec<Arc<Vec<TimedEntry>>> = traces.into_iter().map(Arc::new).collect();

    let max_ts_us = traces
        .iter()
        .flat_map(|t| t.iter())
        .map(|e| e.timestamp_us)
        .max()
        .unwrap_or(1);
    let trace_duration_ms = max_ts_us / 1000;

    let total_request_blocks = Arc::new(AtomicU64::new(0));
    let total_event_blocks = Arc::new(AtomicU64::new(0));
    let total_errors = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(parking_lot::Mutex::new(Vec::new()));

    let wall_start = Instant::now();

    let mut tasks = Vec::new();
    for replica in 0..args.duplication_factor {
        for (worker_idx, worker_trace) in traces.iter().enumerate() {
            let indexer = indexer.clone();
            let trace = worker_trace.clone();
            let req_blocks = total_request_blocks.clone();
            let evt_blocks = total_event_blocks.clone();
            let err_count = total_errors.clone();
            let latencies = latencies.clone();
            let worker_id = (worker_idx + replica * args.num_workers) as u32;
            let count_events = !args.no_count_events;

            tasks.push(tokio::spawn(async move {
                let mut state = TaskState::new(count_events);

                let base_time = tokio::time::Instant::now();

                for entry in trace.iter() {
                    // Pace to this entry's timestamp before processing
                    let target = base_time + Duration::from_micros(entry.timestamp_us);
                    if target > tokio::time::Instant::now() {
                        tokio::time::sleep_until(target).await;
                    }

                    state.process(&entry.entry, &indexer, worker_id);
                }

                req_blocks.fetch_add(state.req_blocks, Ordering::Relaxed);
                evt_blocks.fetch_add(state.evt_blocks, Ordering::Relaxed);
                err_count.fetch_add(state.errors, Ordering::Relaxed);
                latencies.lock().extend(state.latencies);
            }));
        }
    }

    for task in tasks {
        task.await.expect("benchmark task panicked");
    }

    let actual_duration = wall_start.elapsed();
    let req_blocks = total_request_blocks.load(Ordering::Relaxed);
    let evt_blocks = total_event_blocks.load(Ordering::Relaxed);
    let errors = total_errors.load(Ordering::Relaxed);
    let total_blocks = req_blocks + evt_blocks;

    let block_throughput = total_blocks as f64 / actual_duration.as_secs_f64();
    let offered_block_throughput = if trace_duration_ms > 0 {
        total_blocks as f64 / (trace_duration_ms as f64 / 1000.0)
    } else {
        0.0
    };

    let mut lats = latencies.lock().clone();
    lats.sort_unstable();
    let latency_p99_us = if lats.is_empty() {
        0.0
    } else {
        let p99_idx = lats.len().saturating_sub(1) * 99 / 100;
        lats[p99_idx] as f64 / 1000.0
    };

    BenchmarkResults {
        total_request_blocks: req_blocks,
        total_event_blocks: evt_blocks,
        total_blocks,
        total_errors: errors,
        actual_duration,
        block_throughput,
        offered_block_throughput,
        latency_p99_us,
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn compute_sweep_durations(min_ms: u64, max_ms: u64, steps: usize) -> Vec<u64> {
    if steps <= 1 {
        return vec![max_ms];
    }
    let safe_min = min_ms.max(1);
    let safe_max = max_ms.max(1);
    let log_min = (safe_min as f64).ln();
    let log_max = (safe_max as f64).ln();
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            let log_val = log_min + t * (log_max - log_min);
            log_val.exp().round() as u64
        })
        .collect()
}

fn format_throughput(throughput: f64) -> String {
    if throughput >= 1_000_000_000.0 {
        format!("{:.1}B", throughput / 1_000_000_000.0)
    } else if throughput >= 1_000_000.0 {
        format!("{:.1}M", throughput / 1_000_000.0)
    } else if throughput >= 1_000.0 {
        format!("{:.1}K", throughput / 1_000.0)
    } else {
        format!("{throughput:.0}")
    }
}

fn print_results(result: &BenchmarkResults) {
    println!(
        "  Duration: {:.2}s | Blocks: {} req + {} evt = {} total",
        result.actual_duration.as_secs_f64(),
        result.total_request_blocks,
        result.total_event_blocks,
        result.total_blocks,
    );
    println!(
        "  Block throughput: {:.0} block ops/sec ({})",
        result.block_throughput,
        format_throughput(result.block_throughput),
    );
    println!(
        "  Offered: {} block ops/sec | Latency p99: {:.1}us",
        format_throughput(result.offered_block_throughput),
        result.latency_p99_us,
    );
    if result.total_errors > 0 {
        println!("  Errors: {} apply_stored failures", result.total_errors);
    }
}
