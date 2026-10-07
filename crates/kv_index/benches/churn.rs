//! Churn bench: hours of a soak's block-LRU churn on shared chains, compressed to the index's
//! own speed, with the figures that tell fragmentation from everything else.
//!
//! The generator (`kv_index::churn`) runs workers with block-LRU caches over a shared chain
//! pool: every request is a lookup over a random prefix of a random chain, routed to the worker
//! with the longest prefix, which stores what it lacks and evicts past its capacity in the
//! configured order (the mock's tail-first order frees suffixes; hash order frees middle
//! stretches). One thread drives it: fragmentation is a property of the event sequence, not of
//! concurrency, and the lookups' service time is what is measured.
//!
//! Every `--sample-every-requests` requests a row is recorded: lookup p50/p99/p999, runs walked
//! per lookup (mean and p99), runs live, blocks live and the mean run length, the split counters
//! by cause, the mergeable adjacent pairs found by a debug walk (a run with one child, equal
//! coverage, no prefix holder of its own on the child), arena and header bytes, and the events
//! applied; and the index is checked against the reference indexer on a sample of queries. The
//! series goes to the file named by `--json`, when one is given.

#![expect(clippy::print_stdout)]

use std::{
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use kv_index::{
    churn::{Churn, ChurnConfig, FreeOrder, StepReport},
    ReferenceIndexer, ShardedChainIndex,
};
use serde_json::json;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FreeOrderArg {
    /// The mock's order: equal-age blocks freed from the chain's tail (suffix evictions).
    TailFirst,
    /// Equal-age blocks freed by hash: stretches anywhere, holes included.
    Hash,
}

#[derive(Parser, Debug)]
#[command(
    about = "Block-LRU churn on shared chains against the chain index, with the fragmentation series"
)]
struct Args {
    #[arg(long, default_value = "128")]
    workers: usize,
    #[arg(long, default_value = "4000")]
    chains: usize,
    #[arg(long, default_value = "16")]
    min_chain_len: usize,
    #[arg(long, default_value = "256")]
    max_chain_len: usize,
    /// Probability that a chain shares a prefix with an earlier one (where runs branch).
    #[arg(long, default_value = "0.7")]
    share: f64,
    /// Blocks a worker keeps before evicting.
    #[arg(long, default_value = "20000")]
    cache_blocks: usize,
    #[arg(long, value_enum, default_value = "tail-first")]
    free_order: FreeOrderArg,
    /// Blocks a request decodes after its prompt (private tails stored block by block).
    #[arg(long, default_value = "4")]
    decode_blocks: usize,
    /// Probability that a request decodes at all.
    #[arg(long, default_value = "1.0")]
    decode_probability: f64,
    /// Restart (clear) one worker every this many requests and refill it cold; 0 = never.
    #[arg(long, default_value = "0")]
    restart_every: u64,
    /// Requests routed to a restarted worker whatever the holders.
    #[arg(long, default_value = "2000")]
    refill_requests: u64,
    #[arg(long, default_value = "1")]
    shards: usize,
    /// Worker slots per shard.
    #[arg(long, default_value = "256")]
    max_workers: usize,
    #[arg(long, default_value = "2000000")]
    requests: u64,
    /// Stretch the run over this many minutes of wall time (sleeping between requests) instead
    /// of running at the index's speed.
    #[arg(long)]
    minutes: Option<f64>,
    #[arg(long, default_value = "100000")]
    sample_every_requests: u64,
    /// Queries checked against the reference indexer at every sample.
    #[arg(long, default_value = "200")]
    exact_samples: usize,
    /// Queries checked against the reference indexer at the end.
    #[arg(long, default_value = "5000")]
    final_exact_samples: usize,
    #[arg(long, default_value = "20261006")]
    seed: u64,
    /// Write the sample series and the run's figures to this file; without it nothing is
    /// written (the bench also runs under `cargo test --all-targets`, in the crate directory).
    #[arg(long)]
    json: Option<PathBuf>,
    /// Passed by `cargo bench`; ignored.
    #[arg(long, hide = true)]
    bench: bool,
}

fn percentile(sorted: &[u64], num: usize, den: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = sorted.len().saturating_mul(num).div_ceil(den).max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Default)]
struct Window {
    lookup_ns: Vec<u64>,
    walked: Vec<u64>,
    stored: u64,
    removed: u64,
    healed: u64,
    holders: u64,
    decoded: u64,
}

impl Window {
    fn add(&mut self, r: &StepReport) {
        self.lookup_ns.push(r.lookup_ns);
        self.walked.push(r.runs_walked as u64);
        self.stored += r.stored_blocks as u64;
        self.removed += r.removed_blocks as u64;
        self.healed += r.healed_blocks as u64;
        self.decoded += r.decoded_blocks as u64;
        self.holders += r.holders as u64;
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = ChurnConfig {
        workers: args.workers,
        chains: args.chains,
        min_chain_len: args.min_chain_len,
        max_chain_len: args.max_chain_len,
        share: args.share,
        cache_blocks: args.cache_blocks,
        free_order: match args.free_order {
            FreeOrderArg::TailFirst => FreeOrder::TailFirst,
            FreeOrderArg::Hash => FreeOrder::Hash,
        },
        decode_blocks: args.decode_blocks,
        decode_probability: args.decode_probability,
        restart_every: args.restart_every,
        refill_requests: args.refill_requests,
        seed: args.seed,
    };
    let index = ShardedChainIndex::new(args.shards.max(1), args.max_workers);
    let mut reference = ReferenceIndexer::new();
    let mut churn = Churn::new(cfg.clone(), &index);
    let divergence_points = churn.pool.divergence_points();
    let pool_blocks: usize = churn.pool.chains.iter().map(|c| c.contents.len()).sum();
    println!(
        "churn: {} workers, {} chains ({} blocks, {} divergence points), cache {} blocks per worker, free order {:?}, {} shards, {} requests{}",
        args.workers,
        args.chains,
        pool_blocks,
        divergence_points,
        args.cache_blocks,
        cfg.free_order,
        args.shards.max(1),
        args.requests,
        args.minutes.map_or(String::new(), |m| format!(" over {m} minutes")),
    );
    let pace = args
        .minutes
        .map(|m| Duration::from_secs_f64(m * 60.0 / args.requests.max(1) as f64));
    let started = Instant::now();
    let mut window = Window::default();
    let mut series = Vec::new();
    let mut exact_failures = Vec::new();
    for request in 1..=args.requests {
        let report = churn.step(&index, Some(&mut reference));
        window.add(&report);
        if let Some(pace) = pace {
            if request % 1000 == 0 {
                let due = started + pace * request as u32;
                let now = Instant::now();
                if due > now {
                    std::thread::sleep(due - now);
                }
            }
        }
        if request % args.sample_every_requests == 0 || request == args.requests {
            let stats = index.stats();
            let (mergeable_pairs, mergeable_blocks) = index.debug_mergeable();
            let exact = churn.check_exact(&index, &reference, args.exact_samples);
            if let Err(message) = &exact {
                exact_failures.push(format!("request {request}: {message}"));
            }
            let mut lookups = std::mem::take(&mut window.lookup_ns);
            lookups.sort_unstable();
            let mut walked = std::mem::take(&mut window.walked);
            walked.sort_unstable();
            let n = lookups.len().max(1) as f64;
            let row = json!({
                "requests": request,
                "elapsed_s": started.elapsed().as_secs_f64(),
                "lookup_p50_us": percentile(&lookups, 50, 100) as f64 / 1e3,
                "lookup_p99_us": percentile(&lookups, 99, 100) as f64 / 1e3,
                "lookup_p999_us": percentile(&lookups, 999, 1000) as f64 / 1e3,
                "lookup_mean_us": lookups.iter().sum::<u64>() as f64 / n / 1e3,
                "runs_walked_mean": walked.iter().sum::<u64>() as f64 / n,
                "runs_walked_p99": percentile(&walked, 99, 100),
                "holders_mean": window.holders as f64 / n,
                "runs_live": stats.runs_live,
                "blocks_live": stats.blocks_live,
                "mean_run_len": stats.blocks_live as f64 / stats.runs_live.max(1) as f64,
                "memberships": index.current_size(),
                "distinct_blocks": index.entry_count(),
                "splits_by_branch": stats.splits_by_branch,
                "splits_by_hole": stats.splits_by_hole,
                "splits_by_mid_run_store": stats.splits_by_mid_run_store,
                "splits_by_prefix_holders": stats.splits_by_prefix_holders,
                "runs_died": stats.runs_died,
                "mergeable_pairs": mergeable_pairs,
                "mergeable_blocks": mergeable_blocks,
                "partial_entries": stats.partial_entries,
                "max_partials": stats.max_partials,
                "child_entries": stats.child_entries,
                "child_tombstones": stats.child_tombstones,
                "arena_bytes": stats.arena_bytes,
                "header_bytes": stats.header_bytes,
                "slab_bytes": stats.slab_bytes,
                "stored_blocks": window.stored,
                "removed_blocks": window.removed,
                "healed_blocks": window.healed,
                "decoded_blocks": window.decoded,
                "restarts": churn.restarts,
                "exact": exact.is_ok(),
            });
            println!(
                "{:>9} req {:7.1}s lookup p50 {:6.2} p99 {:7.2} us walked mean {:5.2} p99 {:3} | runs live {:7} blocks live {:9} mean len {:6.1} | splits branch {} hole {} mid-run {} holders {} died {} mergeable {} ({} blocks) | stored {} removed {} healed {} | exact {}",
                request,
                row["elapsed_s"].as_f64().unwrap_or(0.0),
                row["lookup_p50_us"].as_f64().unwrap_or(0.0),
                row["lookup_p99_us"].as_f64().unwrap_or(0.0),
                row["runs_walked_mean"].as_f64().unwrap_or(0.0),
                row["runs_walked_p99"].as_u64().unwrap_or(0),
                stats.runs_live,
                stats.blocks_live,
                row["mean_run_len"].as_f64().unwrap_or(0.0),
                stats.splits_by_branch,
                stats.splits_by_hole,
                stats.splits_by_mid_run_store,
                stats.splits_by_prefix_holders,
                stats.runs_died,
                mergeable_pairs,
                mergeable_blocks,
                window.stored,
                window.removed,
                window.healed,
                exact.is_ok(),
            );
            std::io::stdout().flush().ok();
            series.push(row);
            window = Window::default();
        }
    }
    let final_exact = churn.check_exact(&index, &reference, args.final_exact_samples);
    if let Err(message) = &final_exact {
        exact_failures.push(format!("final: {message}"));
    }
    let result = json!({
        "harness": "smg-churn",
        "config": {
            "workers": args.workers,
            "chains": args.chains,
            "min_chain_len": args.min_chain_len,
            "max_chain_len": args.max_chain_len,
            "share": args.share,
            "cache_blocks": args.cache_blocks,
            "free_order": format!("{:?}", cfg.free_order).to_lowercase(),
            "decode_blocks": args.decode_blocks,
            "decode_probability": args.decode_probability,
            "restart_every": args.restart_every,
            "refill_requests": args.refill_requests,
            "shards": args.shards.max(1),
            "requests": args.requests,
            "minutes": args.minutes,
            "seed": args.seed,
            "pool_blocks": pool_blocks,
            "divergence_points": divergence_points,
        },
        "series": series,
        "exact_failures": exact_failures,
        "final_exact": final_exact.is_ok(),
        "elapsed_s": started.elapsed().as_secs_f64(),
    });
    if let Some(out) = &args.json {
        std::fs::write(out, serde_json::to_vec_pretty(&result)?)?;
    }
    println!(
        "churn done in {:.1} s; final exactness {}{}",
        started.elapsed().as_secs_f64(),
        if final_exact.is_ok() { "ok" } else { "FAILED" },
        if exact_failures.is_empty() {
            String::new()
        } else {
            format!(
                "; {} checkpoint failures, first: {}",
                exact_failures.len(),
                exact_failures[0]
            )
        }
    );
    anyhow::ensure!(
        exact_failures.is_empty(),
        "{} exactness failures, first: {}",
        exact_failures.len(),
        exact_failures[0]
    );
    Ok(())
}
