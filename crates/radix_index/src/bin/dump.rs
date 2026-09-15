//! `radix-index-dump`: pull one replica's full state and print a per-holder
//! summary (block count and an order-independent digest of the block set)
//! as JSON. Two replicas that have converged print identical digests; the
//! harness diffs them after a partition heals, a replica rejoins, or a
//! run ends.
//!
//! The digest is a commutative fold over content hashes, so it compares
//! SETS of blocks regardless of the order the snapshot streams them.

use std::collections::BTreeMap;

use radix_index::{
    cli::{parse_flag, validate_flags},
    proto::{self, radix_index_client::RadixIndexClient},
};
use tokio_stream::StreamExt;

const KNOWN_FLAGS: &[&str] = &["--connect"];

#[derive(Default)]
struct HolderSummary {
    blocks: u64,
    xor: u64,
    sum: u64,
    event_fed: bool,
    dropped: bool,
}

fn fold(summary: &mut HolderSummary, content_hash: u64) {
    summary.blocks += 1;
    summary.xor ^= content_hash;
    summary.sum = summary
        .sum
        .wrapping_add(content_hash.wrapping_mul(0x9E37_79B9_7F4A_7C15));
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    validate_flags(&args, KNOWN_FLAGS, &[]);
    let url: String =
        parse_flag(&args, "--connect").unwrap_or_else(|| "http://127.0.0.1:40000".into());
    let mut client = RadixIndexClient::connect(url.clone())
        .await
        .unwrap_or_else(|e| panic!("connect {url}: {e}"))
        .max_decoding_message_size(64 * 1024 * 1024);
    let mut stream = client
        .pull(proto::PullRequest {})
        .await
        .unwrap_or_else(|e| panic!("pull {url}: {e}"))
        .into_inner();
    let mut holders: BTreeMap<String, HolderSummary> = BTreeMap::new();
    let mut updates = 0u64;
    while let Some(update) = stream.next().await {
        let update = update.unwrap_or_else(|e| panic!("pull stream {url}: {e}"));
        updates += 1;
        let ks = update.keyspace.as_ref();
        let key = format!(
            "{}|{}|{}|{}",
            ks.map(|k| k.model.clone()).unwrap_or_default(),
            ks.map_or(0, |k| k.block_size),
            ks.map_or(0, |k| k.symbol_kind),
            update.holder
        );
        let entry = holders.entry(key).or_default();
        if let Some(added) = &update.added {
            entry.event_fed |= added.event_fed;
        }
        entry.dropped |= update.dropped;
        for event in &update.events {
            if let Some(proto::event::Kind::Stored(stored)) = &event.kind {
                for block in &stored.blocks {
                    fold(entry, block.content_hash);
                }
            }
        }
    }
    let total_blocks: u64 = holders.values().map(|h| h.blocks).sum();
    println!("{{");
    println!("  \"source\": {url:?},");
    println!("  \"updates\": {updates},");
    println!("  \"total_blocks\": {total_blocks},");
    println!("  \"holders\": {{");
    let n = holders.len();
    for (i, (key, h)) in holders.iter().enumerate() {
        println!(
            "    {key:?}: {{\"blocks\": {}, \"digest\": \"{:016x}{:016x}\", \"event_fed\": {}, \"dropped\": {}}}{}",
            h.blocks,
            h.xor,
            h.sum,
            h.event_fed,
            h.dropped,
            if i + 1 < n { "," } else { "" }
        );
    }
    println!("  }}");
    println!("}}");
}
