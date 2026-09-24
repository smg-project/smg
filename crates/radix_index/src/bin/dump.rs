//! `radix-index-dump`: pull one replica's full state and print a per-holder
//! summary (block count and an order-independent digest of the block set)
//! as JSON. Two replicas that have converged print identical digests; the
//! harness diffs them after a partition heals, a replica rejoins, or a
//! run ends.
//!
//! The digest is a commutative fold over BOTH of each block's
//! identities — the position-chained `seq_hash` and the
//! position-independent `content_hash` — so it compares CHAINS, not
//! merely block sets, regardless of the order the snapshot streams them.

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

fn fold(summary: &mut HolderSummary, block: &proto::Block) {
    summary.blocks += 1;
    // BOTH identities. `content_hash` is position-INDEPENDENT, so folding
    // it alone compares block multisets and is blind to where those
    // blocks sit in the chain — and that gap is reachable: bootstrap
    // equivalence is scoped to gap-free holders, so a gapped holder
    // brought up by a Pull lands the same content set at different
    // positions than its source (same shape when a chunk whose parent is
    // missing gets re-anchored on one replica but not the other). Two
    // dumps would print identical digests while the replicas genuinely
    // disagree and every routing query against them scores differently.
    // `seq_hash` is the position-chained identity the tree is keyed on
    // and `find_matches` traverses, and it is deterministic across
    // replicas by construction, so folding it adds no false divergence.
    // The rotate stops the two from cancelling on a block whose
    // identities coincide, which would erase that block from the digest.
    let mixed = block.seq_hash ^ block.content_hash.rotate_left(32);
    summary.xor ^= mixed;
    // XOR cancels in pairs, and a repeated block within one holder is
    // ordinary (a prompt that repeats a chunk); the multiplied running
    // sum keeps multiplicity. Both stay order-independent.
    summary.sum = summary
        .sum
        .wrapping_add(mixed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
}

/// One JSON string literal, quoted and escaped. Rust's `{:?}` is NOT
/// JSON — it renders a control byte as `\u{7f}`, which every JSON parser
/// rejects — and the model and holder names here arrive from the wire.
/// One odd byte published by a worker would make this dump unparsable
/// exactly when the harness is diffing two replicas to decide whether
/// they converged. (No serde dependency in this crate; the output is
/// these few lines.)
fn json_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    out.push('"');
    for c in raw.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Every remaining C0 control needs the six-character form;
            // JSON has no short escape for them.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
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
                    fold(entry, block);
                }
            }
        }
    }
    let total_blocks: u64 = holders.values().map(|h| h.blocks).sum();
    println!("{{");
    println!("  \"source\": {},", json_string(&url));
    println!("  \"updates\": {updates},");
    println!("  \"total_blocks\": {total_blocks},");
    println!("  \"holders\": {{");
    let n = holders.len();
    for (i, (key, h)) in holders.iter().enumerate() {
        println!(
            "    {}: {{\"blocks\": {}, \"digest\": \"{:016x}{:016x}\", \"event_fed\": {}, \"dropped\": {}}}{}",
            json_string(key),
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
