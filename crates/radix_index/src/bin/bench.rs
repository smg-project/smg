//! Synthetic scale bench: fill the engine to N entries, report RSS per
//! entry, sustained apply rate, and query latency percentiles. This is
//! the only source for production sizing claims — the sim rig's fleet is
//! ~0.3% of production entry count.
//!
//! Usage:
//!   radix-index-bench [--holders 2000] [--blocks-per-holder 9000]
//!     [--queries 20000] [--query-depth 78]

use std::time::Instant;

use radix_index::{
    placement_chain, wire_hash::content_hash as compute_content_hash, Engine, EngineConfig,
    KeyspaceKey, SymbolKind, UpdateMsg, WireEvent,
};

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let holders: usize = parse_flag(&args, "--holders").unwrap_or(2000);
    let blocks_per_holder: usize = parse_flag(&args, "--blocks-per-holder").unwrap_or(9000);
    let queries: usize = parse_flag(&args, "--queries").unwrap_or(20_000);
    let query_depth: usize = parse_flag(&args, "--query-depth").unwrap_or(78);

    let keyspace = KeyspaceKey {
        model: "bench".into(),
        symbol_kind: SymbolKind::Tokens,
        block_size: 128,
    };
    let engine = Engine::new(EngineConfig::default());
    let rss_before = rss_kib();

    // Fill: each holder gets a distinct chain, stored in event batches of
    // 8 blocks (the realistic event-shape), timing the apply path.
    let batch = 8usize;
    let fill_start = Instant::now();
    let mut applied_blocks = 0usize;
    for h in 0..holders {
        let hashes: Vec<_> = (0..blocks_per_holder as u64)
            .map(|i| compute_content_hash(&[h as u32, (i >> 32) as u32, i as u32]))
            .collect();
        let chain = placement_chain(&hashes);
        for (i, window) in chain.chunks(batch).enumerate() {
            let parent = if i == 0 {
                None
            } else {
                Some(chain[i * batch - 1].seq_hash)
            };
            let update = UpdateMsg {
                keyspace: keyspace.clone(),
                holder: format!("grpc://10.0.0.{}:{}", h % 250, 9000 + h / 250),
                epoch: 1,
                seq: (i + 1) as u64,
                events: vec![WireEvent::Stored {
                    parent,
                    blocks: window.to_vec(),
                }],
                added: None,
                dropped: false,
            };
            engine.apply(&update);
            applied_blocks += window.len();
        }
    }
    let fill_secs = fill_start.elapsed().as_secs_f64();
    let rss_after = rss_kib();
    let entries = engine.entry_count();

    // Queries: warm prefixes of the filled chains at the requested depth.
    let mut latencies = Vec::with_capacity(queries);
    for q in 0..queries {
        let h = q % holders;
        let hashes: Vec<_> = (0..query_depth as u64)
            .map(|i| compute_content_hash(&[h as u32, (i >> 32) as u32, i as u32]))
            .collect();
        let start = Instant::now();
        let scores = engine.find_matches(&keyspace, &hashes);
        latencies.push(start.elapsed().as_secs_f64() * 1e6);
        assert!(!scores.is_empty(), "warm query must match");
    }
    latencies.sort_by(f64::total_cmp);
    let pct = |p: f64| latencies[((latencies.len() as f64 * p) as usize).min(latencies.len() - 1)];

    println!("entries {entries}");
    println!(
        "rss_total_mib {:.1}  rss_bytes_per_entry {:.1}",
        (rss_after - rss_before) as f64 / 1024.0,
        (rss_after - rss_before) as f64 * 1024.0 / entries.max(1) as f64
    );
    println!(
        "apply_blocks_per_sec {:.0} (filled {applied_blocks} blocks in {fill_secs:.1}s)",
        applied_blocks as f64 / fill_secs
    );
    println!(
        "query_us p50 {:.1}  p99 {:.1}  max {:.1}  (depth {query_depth}, {queries} queries)",
        pct(0.50),
        pct(0.99),
        pct(1.0)
    );
}
