//! Cross-structure comparison: the gateway's TokenTree (per-token
//! radix) and StringTree (per-character radix) vs the new block-hash
//! RadixTree, on one shared token corpus — build rate, resident
//! bytes, match latency, and MATCHING RESOLUTION (what each promises
//! vs what an engine paging KV in fixed blocks can physically reuse).
//!
//! Block size is a swept axis (64/128/256/512), exercising the same
//! knob the wire exposes end to end (gateway --kv-indexer-block-size,
//! bridge --block-size, keyspace-keyed on the service).
//!
//! One structure per process for clean RSS deltas:
//!   RADIX_COMPARE_SIDE=token|string|radix64|radix128|radix256|radix512
//!
//! Run: RADIX_COMPARE_SIDE=... cargo test -p radix-tree --release \
//!   --test tree_compare -- --ignored --nocapture

mod common;

use std::time::Instant;

use common::Rng;
use kv_index::{compute_request_content_hashes, RadixTree as _, StringTree, TokenTree};
use radix_tree::{Config, HolderId, OverlapScratch, RadixTree};

const TENANTS: usize = 64;
const FAMILIES: usize = 400;
/// Shared prefix length in tokens, log-uniform.
const SHARED_TOKENS: (u32, u32) = (512, 8192);
const TAIL_TOKENS: (u32, u32) = (128, 1024);
const TENANTS_PER_FAMILY: (usize, usize) = (1, 6);
const TOKEN_SPACE: u64 = 150_000;
/// The engine's actual page size: the amount of matching resolution
/// that is PHYSICALLY reusable, whatever the index promises.
const ENGINE_PAGE: u32 = 256;

fn log_uniform(rng: &mut Rng, lo: u32, hi: u32) -> u32 {
    let (llo, lhi) = ((lo as f64).ln(), (hi as f64).ln());
    let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    (llo + u * (lhi - llo)).exp().round() as u32
}

struct Corpus {
    /// (tokens of shared prefix, member tenants)
    families: Vec<(Vec<u32>, Vec<usize>)>,
    /// (tenant, full sequence = family prefix + private tail, family)
    inserts: Vec<(usize, Vec<u32>, usize)>,
    /// (family, query length clamped to family prefix, tenant known
    /// to hold it) — the TRUE cached-prefix length equals the query
    /// length by construction.
    queries: Vec<(usize, u32, usize)>,
}

fn build_corpus(seed: u64) -> Corpus {
    let mut rng = Rng::new(seed);
    let mut families = Vec::with_capacity(FAMILIES);
    for _ in 0..FAMILIES {
        let len = log_uniform(&mut rng, SHARED_TOKENS.0, SHARED_TOKENS.1);
        let tokens: Vec<u32> = (0..len)
            .map(|_| (rng.next() % TOKEN_SPACE) as u32)
            .collect();
        let n = TENANTS_PER_FAMILY.0 + rng.below(TENANTS_PER_FAMILY.1 - TENANTS_PER_FAMILY.0 + 1);
        let mut members: Vec<usize> = (0..n).map(|_| rng.below(TENANTS)).collect();
        members.sort_unstable();
        members.dedup();
        families.push((tokens, members));
    }
    let mut inserts = Vec::new();
    for (fi, (tokens, members)) in families.iter().enumerate() {
        for &tenant in members {
            let tail_len =
                TAIL_TOKENS.0 + (rng.next() % (TAIL_TOKENS.1 - TAIL_TOKENS.0 + 1) as u64) as u32;
            let mut seq = tokens.clone();
            seq.extend((0..tail_len).map(|_| (rng.next() % TOKEN_SPACE) as u32));
            inserts.push((tenant, seq, fi));
        }
    }
    let mut queries = Vec::new();
    for (fi, (tokens, members)) in families.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        for _ in 0..8 {
            let d = 1 + rng.below(tokens.len()) as u32;
            queries.push((fi, d, members[rng.below(members.len())]));
        }
    }
    Corpus {
        families,
        inserts,
        queries,
    }
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

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

/// Tokens -> 4-hex-char words, the string-side rendering of the same
/// corpus (per-character tree granularity, 4 chars per token).
fn to_text(tokens: &[u32]) -> String {
    let mut s = String::with_capacity(tokens.len() * 4);
    for &t in tokens {
        s.push_str(&format!("{:04x}", t & 0xFFFF));
    }
    s
}

fn placement_chain_keys(hashes: &[kv_index::ContentHash]) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(hashes.len());
    let mut prev = kv_index::SequenceHash(0);
    for (i, &h) in hashes.iter().enumerate() {
        let seq = if i == 0 {
            kv_index::SequenceHash(h.0)
        } else {
            kv_index::chain_prefix_hash(prev, h)
        };
        out.push((seq.0, h.0));
        prev = seq;
    }
    out
}

#[allow(clippy::large_enum_variant)] // bench-local, one instance per process
enum Side {
    Token(TokenTree),
    String(StringTree),
    Radix {
        tree: RadixTree,
        ids: Vec<HolderId>,
        block: u32,
        out: Vec<radix_tree::Overlap>,
        qscratch: OverlapScratch,
    },
}

#[test]
#[ignore = "comparison benchmark; run --release --ignored --nocapture with RADIX_COMPARE_SIDE"]
fn tree_compare() {
    let side_name = std::env::var("RADIX_COMPARE_SIDE").unwrap_or_else(|_| "radix256".into());
    let corpus = build_corpus(20260902);
    let total_tokens: u64 = corpus.inserts.iter().map(|(_, s, _)| s.len() as u64).sum();
    println!(
        "side: {side_name}; corpus: {} families, {} inserts, {:.1}M tokens",
        corpus.families.len(),
        corpus.inserts.len(),
        total_tokens as f64 / 1e6
    );

    let rss_before = rss_kib();
    let mut side = match side_name.as_str() {
        "token" => Side::Token(TokenTree::new()),
        "string" => Side::String(StringTree::new()),
        other => {
            let block: u32 = other
                .strip_prefix("radix")
                .and_then(|b| b.parse().ok())
                .expect("radix<block>");
            let mut tree = RadixTree::new(Config::default());
            let ids = (0..TENANTS)
                .map(|t| tree.create_holder(&format!("tenant-{t}")))
                .collect();
            Side::Radix {
                tree,
                ids,
                block,
                out: Vec::new(),
                qscratch: OverlapScratch::default(),
            }
        }
    };

    // Pre-render side-specific insert inputs OUTSIDE the timed region.
    enum Prepared {
        Tokens(usize, Vec<u32>),
        Text(usize, String),
        Blocks(usize, Vec<(u64, u64)>),
    }
    let prepared: Vec<Prepared> = corpus
        .inserts
        .iter()
        .map(|(tenant, seq, _)| match &side {
            Side::Token(_) => Prepared::Tokens(*tenant, seq.clone()),
            Side::String(_) => Prepared::Text(*tenant, to_text(seq)),
            Side::Radix { block, .. } => Prepared::Blocks(
                *tenant,
                placement_chain_keys(&compute_request_content_hashes(seq, *block as usize)),
            ),
        })
        .collect();

    let build_start = Instant::now();
    for item in &prepared {
        match (&mut side, item) {
            (Side::Token(t), Prepared::Tokens(tenant, seq)) => {
                t.insert_tokens(seq, &format!("tenant-{tenant}"));
            }
            (Side::String(t), Prepared::Text(tenant, text)) => {
                t.insert_text(text, &format!("tenant-{tenant}"));
            }
            (Side::Radix { tree, ids, .. }, Prepared::Blocks(tenant, blocks)) => {
                tree.store(ids[*tenant], None, blocks).expect("store");
            }
            _ => unreachable!(),
        }
    }
    let build = build_start.elapsed();
    let rss_after = rss_kib();
    println!(
        "build: {:.2}s -> {:.2}M tokens/s; memory: {:.1} MiB -> {:.2} B/token",
        build.as_secs_f64(),
        total_tokens as f64 / build.as_secs_f64() / 1e6,
        (rss_after - rss_before) as f64 / 1024.0,
        (rss_after - rss_before) as f64 * 1024.0 / total_tokens as f64
    );

    // Queries: latency + promised-vs-physical matching resolution.
    // True cached prefix = query length L (by construction). Physical
    // ceiling = floor(L / ENGINE_PAGE) * ENGINE_PAGE.
    let mut prepared_queries = Vec::new();
    for &(fi, len, tenant) in &corpus.queries {
        let toks = &corpus.families[fi].0[..len as usize];
        match &side {
            Side::Token(_) => prepared_queries.push((Prepared::Tokens(tenant, toks.to_vec()), len)),
            Side::String(_) => prepared_queries.push((Prepared::Text(tenant, to_text(toks)), len)),
            Side::Radix { block, .. } => prepared_queries.push((
                Prepared::Blocks(
                    tenant,
                    placement_chain_keys(&compute_request_content_hashes(toks, *block as usize)),
                ),
                len,
            )),
        }
    }
    let mut lat: Vec<u64> = Vec::with_capacity(prepared_queries.len());
    let mut promised_minus_true = 0i64;
    let mut promised_minus_physical = 0i64;
    let mut n = 0i64;
    for (q, true_len) in &prepared_queries {
        let physical = (*true_len / ENGINE_PAGE) * ENGINE_PAGE;
        let t = Instant::now();
        let matched_tokens: u32 = match (&mut side, q) {
            (Side::Token(tree), Prepared::Tokens(_, toks)) => {
                tree.prefix_match_with_counts(toks).matched_token_count as u32
            }
            (Side::String(tree), Prepared::Text(_, text)) => {
                (tree.prefix_match_with_counts(text).matched_char_count as u32) / 4
            }
            (
                Side::Radix {
                    tree,
                    ids,
                    block,
                    out,
                    qscratch,
                },
                Prepared::Blocks(tenant, blocks),
            ) => {
                let chain: Vec<u64> = blocks.iter().map(|&(_, c)| c).collect();
                tree.overlap(&chain, qscratch, out);
                let (idx, _) = ids[*tenant].parts();
                out.iter()
                    .find(|o| o.holder.parts().0 == idx)
                    .map_or(0, |o| o.depth * *block)
            }
            _ => unreachable!(),
        };
        lat.push(t.elapsed().as_nanos() as u64);
        promised_minus_true += matched_tokens as i64 - *true_len as i64;
        promised_minus_physical += matched_tokens as i64 - physical as i64;
        n += 1;
    }
    lat.sort_unstable();
    println!(
        "match: n={} p50={}ns p99={}ns; promised-vs-true avg {:+.1} tokens; promised-vs-physical(page {ENGINE_PAGE}) avg {:+.1} tokens",
        n,
        percentile(&lat, 0.50),
        percentile(&lat, 0.99),
        promised_minus_true as f64 / n as f64,
        promised_minus_physical as f64 / n as f64,
    );
}
