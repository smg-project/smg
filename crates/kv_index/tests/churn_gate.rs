//! Gate-sized churn: a few hundred thousand events of block-LRU churn with decode tails, at the
//! index's own speed, checked against the reference indexer and held to the content's shape.
//!
//! The churn bench (`benches/churn.rs`) runs the same generator for the series; this is the part
//! of it that fails a gate: exactness under churn always, and the bound on runs live that keeps
//! fragmentation from coming back silently where a divergence inside a run hangs a child off the
//! offset and leaves the run whole.

use kv_index::{
    churn::{Churn, ChurnConfig, FreeOrder},
    ReferenceIndexer, ShardedChainIndex,
};

fn config(seed: u64) -> ChurnConfig {
    ChurnConfig {
        workers: 32,
        chains: 600,
        min_chain_len: 16,
        max_chain_len: 128,
        share: 0.7,
        cache_blocks: 3000,
        free_order: FreeOrder::TailFirst,
        decode_blocks: 2,
        decode_probability: 0.5,
        restart_every: 7000,
        refill_requests: 300,
        seed,
    }
}

fn requests() -> u64 {
    std::env::var("KV_INDEX_CHURN_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000)
}

/// Every lookup during and after the churn agrees with the reference, under one shard and two.
#[test]
fn churn_with_decode_tails_and_restarts_is_exact() {
    for shards in [1usize, 2] {
        let index = ShardedChainIndex::new(shards, 64);
        let mut reference = ReferenceIndexer::new();
        let mut churn = Churn::new(config(7 + shards as u64), &index);
        let total = requests();
        for request in 1..=total {
            let report = churn.step(&index, Some(&mut reference));
            assert!(
                report.lookup_ns < 10_000_000_000,
                "a lookup took {} ns",
                report.lookup_ns
            );
            if request % 5000 == 0 {
                churn
                    .check_exact(&index, &reference, 100)
                    .unwrap_or_else(|why| panic!("shards {shards}, request {request}: {why}"));
            }
        }
        churn
            .check_exact(&index, &reference, 1000)
            .unwrap_or_else(|why| panic!("shards {shards}, at the end: {why}"));
        let stats = index.stats();
        assert_eq!(
            stats.splits_by_branch, 0,
            "a divergence inside a run hangs a child off it and splits nothing"
        );
        assert!(churn.restarts > 0, "workers restarted");
        assert!(index.current_size() > 0);
    }
}

/// Runs live stay within a small multiple of the content's shape: one run per chain and per
/// divergence point, plus the live private decode tails, each a run of its own. Before a
/// divergence inside a run stopped splitting it, every prompt end left a boundary behind and the
/// count ran away with the requests; this keeps that from coming back.
#[test]
fn runs_live_stay_within_the_content_shape() {
    let index = ShardedChainIndex::new(1, 64);
    let mut churn = Churn::new(config(11), &index);
    let total = requests() * 4;
    let shape = churn.pool.chains.len() + churn.pool.divergence_points();
    for request in 1..=total {
        churn.step(&index, None);
        if request % 20_000 == 0 || request == total {
            let runs_live = index.stats().runs_live;
            let bound = 2 * shape + churn.live_decode_blocks();
            assert!(
                runs_live <= bound,
                "request {request}: runs live {runs_live} above the content's shape ({shape} chains and branch points, {} live decode blocks; bound {bound})",
                churn.live_decode_blocks()
            );
        }
    }
}
