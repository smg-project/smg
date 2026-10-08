//! The relay's publisher-side cost and footprint with the live-block record
//! ([`engine_servicer::kv_state`]) next to the history and the normalizer's
//! own per-hash record, at the fleet's scale and above: eight data-parallel
//! ranks holding 676k blocks each (the 8B workers' pool token count taken as
//! a block count, sixteen times their 42k blocks of 16 tokens).
//!
//! Printed once at the start: the resident set after filling the state
//! without and with the live-block record (bytes per live block), and the
//! end-to-end rate of a real [`KvEventRelay`] absorbing a ZMQ publisher.
//! Then criterion times one incoming batch on the publisher task's path
//! (msgpack decode, normalize, history push; plus the record) in steady
//! state, every batch storing a 64-block chain and evicting one, so the live
//! set stays at its size: the throughput is reported in blocks per second.
//!
//! Run: `cargo bench -p engine-servicer --bench kv_relay_apply`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::{
    fs,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use engine_servicer::{
    kv_events::{KvEventRelay, RelayConfig, DEFAULT_HISTORY_BATCHES, DEFAULT_HISTORY_BYTES},
    kv_history::History,
    kv_state::LiveState,
    kv_wire::{Normalizer, WireBatch},
};
use engine_zmq_client::codec::TrailingTolerant;
use serde_json::json;
use zeromq::{prelude::*, PubSocket, ZmqMessage};

const RANKS: i32 = 8;
/// Live blocks per rank: 676,144 / 64 chains of 64 blocks.
const CHAINS_PER_RANK: u64 = 676_144 / BLOCKS_PER_BATCH as u64;
const BLOCKS_PER_BATCH: usize = 64;
const BLOCK_SIZE: usize = 16;
const TOKENS_PER_BATCH: usize = BLOCKS_PER_BATCH * BLOCK_SIZE;
/// Batches the ZMQ run publishes.
const ZMQ_BATCHES: u64 = 20_000;

fn hashes(rank: i32, chain: u64) -> Vec<i64> {
    let first = chain * BLOCKS_PER_BATCH as u64 + 1;
    (first..first + BLOCKS_PER_BATCH as u64)
        .map(|hash| (hash as i64) | (i64::from(rank) << 48))
        .collect()
}

fn token(seed: u64) -> u32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0x7FFF) as u32
}

/// A vLLM-style publisher batch on `rank`: one `BlockStored` of chain
/// `chain` (64 blocks, 1,024 tokens, the engine's event keys) and, when
/// `evict` names a chain, one `BlockRemoved` of its 64 hashes.
fn payload(rank: i32, chain: u64, evict: Option<u64>) -> Vec<u8> {
    let stored = hashes(rank, chain);
    let token_ids: Vec<u32> = (0..TOKENS_PER_BATCH as u64)
        .map(|i| token(chain * 4_096 + i))
        .collect();
    let mut events = vec![json!({
        "type": "BlockStored",
        "block_hashes": stored,
        "parent_block_hash": null,
        "token_ids": token_ids,
        "block_size": BLOCK_SIZE,
        "lora_id": null,
        "medium": "GPU",
        "lora_name": null,
        "group_idx": 0,
        "kv_cache_spec_kind": "full_attention",
    })];
    if let Some(old) = evict {
        events.push(json!({
            "type": "BlockRemoved",
            "block_hashes": hashes(rank, old),
            "medium": "GPU",
            "group_idx": 0,
        }));
    }
    rmp_serde::to_vec_named(&json!([1_700_000_000.0, events, rank])).expect("encodes")
}

/// The publisher task's per-batch work, with or without the live-block record.
struct Path {
    normalizer: Normalizer,
    history: History,
    state: Option<LiveState>,
    event_id: u64,
    seq: u64,
}

impl Path {
    fn new(with_state: bool) -> Self {
        Self {
            normalizer: Normalizer::new(),
            history: History::new(DEFAULT_HISTORY_BATCHES, DEFAULT_HISTORY_BYTES),
            state: with_state.then(LiveState::new),
            event_id: 0,
            seq: 0,
        }
    }

    fn apply(&mut self, payload: &[u8]) {
        let wire = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(payload)
            .expect("decodes")
            .0;
        self.seq += 1;
        let batch = Arc::new(
            self.normalizer
                .normalize_batch(wire, self.seq, &mut self.event_id),
        );
        self.history.push(self.seq, Arc::clone(&batch));
        if let Some(state) = &mut self.state {
            state.apply(&batch);
        }
    }

    /// Every rank at its full live set.
    fn fill(&mut self) {
        for chain in 0..CHAINS_PER_RANK {
            for rank in 0..RANKS {
                self.apply(&payload(rank, chain, None));
            }
        }
    }
}

/// Resident set in bytes, from `/proc/self/status`.
fn rss_bytes() -> u64 {
    let status = fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map_or(0, |kib| kib * 1024)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Fill the publisher path without and with the record; print what each
/// adds to the resident set.
fn footprint() -> (Path, Path) {
    let live_blocks = u64::from(RANKS as u32) * CHAINS_PER_RANK * BLOCKS_PER_BATCH as u64;
    let baseline = rss_bytes();
    let started = Instant::now();
    let mut without = Path::new(false);
    without.fill();
    let fill_without = started.elapsed();
    let after_without = rss_bytes();
    let started = Instant::now();
    let mut with = Path::new(true);
    with.fill();
    let fill_with = started.elapsed();
    let after_with = rss_bytes();
    let history_and_normalizer = after_without.saturating_sub(baseline);
    let record = after_with
        .saturating_sub(after_without)
        .saturating_sub(history_and_normalizer);
    eprintln!(
        "footprint: {} ranks x {} live blocks ({} batches of {} blocks per rank)\n  \
         history + normalizer record: {:.0} MiB ({:.0} B per live block), filled in {:.1?} \
         ({:.0} blocks/s)\n  \
         live-block record on top:    {:.0} MiB ({:.0} B per live block), filled in {:.1?} \
         ({:.0} blocks/s)\n  \
         record entries {} (copies {}), history {} batches / {:.0} MiB encoded",
        RANKS,
        CHAINS_PER_RANK * BLOCKS_PER_BATCH as u64,
        CHAINS_PER_RANK,
        BLOCKS_PER_BATCH,
        mib(history_and_normalizer),
        history_and_normalizer as f64 / live_blocks as f64,
        fill_without,
        live_blocks as f64 / fill_without.as_secs_f64(),
        mib(record),
        record as f64 / live_blocks as f64,
        fill_with,
        live_blocks as f64 / fill_with.as_secs_f64(),
        with.state.as_ref().map_or(0, LiveState::entries),
        with.state.as_ref().map_or(0, LiveState::blocks),
        with.history.len(),
        mib(with.history.bytes() as u64),
    );
    (without, with)
}

/// A real relay behind a ZMQ publisher, absorbing `ZMQ_BATCHES` batches of
/// 64 blocks as fast as the publisher sends them (a ring of 512 chains, so
/// the record sees repeats as capped copies).
fn relay_over_zmq() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let relay = KvEventRelay::new(RelayConfig {
            endpoint,
            replay_endpoint: None,
            topic: "kv".to_string(),
            history_batches: DEFAULT_HISTORY_BATCHES,
            history_bytes: DEFAULT_HISTORY_BYTES,
            replay_timeout: Duration::from_secs(2),
            load_tick: Duration::from_millis(100),
            heartbeat_interval: Duration::from_secs(1),
            heartbeat_backoff: Duration::from_secs(5),
        });
        relay.start();
        let frame = |seq: u64, body: &[u8]| {
            let mut message = ZmqMessage::from(b"kv".to_vec());
            message.push_back(seq.to_be_bytes().to_vec().into());
            message.push_back(body.to_vec().into());
            message
        };
        let ring: Vec<Vec<u8>> = (0..512u64).map(|chain| payload(0, chain, None)).collect();
        // The subscription lands a moment after the connect.
        for _ in 0..500 {
            publisher.send(frame(0, &ring[0])).await.expect("publish");
            if relay.counts().relayed >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(relay.counts().relayed, 1, "the relay joined the publisher");
        let started = Instant::now();
        for seq in 1..=ZMQ_BATCHES {
            publisher
                .send(frame(seq, &ring[(seq % 512) as usize]))
                .await
                .expect("publish");
        }
        let published = started.elapsed();
        let counts = loop {
            let counts = relay.counts();
            if counts.relayed + counts.gap_batches_lost > ZMQ_BATCHES
                || started.elapsed() > Duration::from_secs(120)
            {
                break counts;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        };
        let elapsed = started.elapsed();
        let blocks = (counts.relayed - 1) * BLOCKS_PER_BATCH as u64;
        eprintln!(
            "relay over zmq: {} batches of {} blocks published in {:.2?}, relayed {} in {:.2?} \
             ({:.0} batches/s, {:.0} blocks/s); publisher gaps {}, batches lost {}",
            ZMQ_BATCHES,
            BLOCKS_PER_BATCH,
            published,
            counts.relayed - 1,
            elapsed,
            (counts.relayed - 1) as f64 / elapsed.as_secs_f64(),
            blocks as f64 / elapsed.as_secs_f64(),
            counts.publisher_gaps,
            counts.gap_batches_lost,
        );
    });
}

/// Steady state: each batch stores a new chain on the next rank and evicts
/// that rank's oldest, keeping every rank at its live set.
fn bench_apply(c: &mut Criterion, name: &str, mut path: Path) {
    let mut next = [CHAINS_PER_RANK; RANKS as usize];
    let mut rank = 0usize;
    let mut group = c.benchmark_group("relay_apply");
    group.throughput(Throughput::Elements(BLOCKS_PER_BATCH as u64));
    group.sample_size(30);
    group.bench_function(name, |b| {
        b.iter_batched(
            || {
                let r = rank;
                rank = (rank + 1) % RANKS as usize;
                let chain = next[r];
                next[r] += 1;
                payload(r as i32, chain, Some(chain - CHAINS_PER_RANK))
            },
            |payload| path.apply(&payload),
            BatchSize::SmallInput,
        );
    });
    group.finish();
    if let Some(state) = &path.state {
        assert_eq!(
            state.blocks(),
            u64::from(RANKS as u32) * CHAINS_PER_RANK * BLOCKS_PER_BATCH as u64,
            "the live set stayed at its size"
        );
    }
}

fn benches(c: &mut Criterion) {
    relay_over_zmq();
    let (without, with) = footprint();
    bench_apply(c, "decode_normalize_push", without);
    bench_apply(c, "decode_normalize_push_record", with);
}

criterion_group!(relay, benches);
criterion_main!(relay);
