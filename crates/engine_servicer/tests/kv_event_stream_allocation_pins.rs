//! What a `SubscribeKvEvents` stream allocates per relayed batch, at one and
//! at two subscribers of one relay, over a ZMQ publisher of 64-block batches
//! (the batch `benches/kv_relay_apply.rs` publishes). The relay's own work per
//! batch (decode, normalize, history, state) is the no-subscriber figure;
//! every subscriber adds what its stream does with the batch before it goes
//! out: the last stream to claim a batch takes it as it is, a further stream
//! copies it. Printed and pinned, so a change in what a subscriber costs per
//! batch fails a test rather than moving a bench inside its noise. A separate
//! executable, so it can declare the counting allocator.
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "a test that stops on a broken setup and prints the counts it pins"
)]

mod common;

use std::{sync::Arc, time::Duration};

use common::counting_alloc::{self, Counting};
use engine_servicer::kv_events::{
    KvEventRelay, RankSource, RelayConfig, DEFAULT_HISTORY_BATCHES, DEFAULT_HISTORY_BYTES,
};
use futures::StreamExt;
use serde_json::json;
use smg_grpc_client::common_proto;
use zeromq::{prelude::*, PubSocket, ZmqMessage};

#[global_allocator]
static GLOBAL: Counting = Counting;

const BLOCKS_PER_BATCH: usize = 64;
const BLOCK_SIZE: usize = 16;
const TOKENS_PER_BATCH: usize = BLOCKS_PER_BATCH * BLOCK_SIZE;
/// Batches counted at each subscriber count.
const BATCHES: u64 = 64;
/// The allocation calls the one stream of a relay adds per relayed batch: the
/// slot the batch rides the live channel in and the stream's own bookkeeping,
/// nothing of the batch itself (3 measured, plus one for the
/// window's rounding). Before the relay handed the batch over, every stream
/// copied it to attach its load record: one call for the event list, one for
/// the block list, one per block's token ids and one per string the stored
/// event carries, 70 in all over the relay's own 247.
const TAKER_CEILING: usize = 4;
/// What a second stream of the same relay adds: one copy of the batch and its
/// own bookkeeping (68 measured, plus two).
const COPIER_CEILING: usize = 70;

fn hashes(chain: u64) -> Vec<i64> {
    let first = chain * BLOCKS_PER_BATCH as u64 + 1;
    (first..first + BLOCKS_PER_BATCH as u64)
        .map(|hash| hash as i64)
        .collect()
}

fn token(seed: u64) -> u32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0x7FFF) as u32
}

/// A vLLM-style publisher batch on rank 0: one `BlockStored` of chain
/// `chain` (64 blocks, 1,024 tokens, the engine's event keys).
fn payload(chain: u64) -> Vec<u8> {
    let token_ids: Vec<u32> = (0..TOKENS_PER_BATCH as u64)
        .map(|i| token(chain * 4_096 + i))
        .collect();
    let events = vec![json!({
        "type": "BlockStored",
        "block_hashes": hashes(chain),
        "parent_block_hash": null,
        "token_ids": token_ids,
        "block_size": BLOCK_SIZE,
        "lora_id": null,
        "medium": "GPU",
        "lora_name": null,
        "group_idx": 0,
        "kv_cache_spec_kind": "full_attention",
    })];
    rmp_serde::to_vec_named(&json!([1_700_000_000.0, events, 0])).expect("encodes")
}

fn frame(seq: u64, body: &[u8]) -> ZmqMessage {
    let mut message = ZmqMessage::from(b"kv".to_vec());
    message.push_back(seq.to_be_bytes().to_vec().into());
    message.push_back(body.to_vec().into());
    message
}

/// A relay joined to a publisher of this test's own, with the sequence the
/// next published batch takes.
struct Bed {
    publisher: PubSocket,
    relay: Arc<KvEventRelay>,
    next_seq: u64,
    ring: Vec<Vec<u8>>,
}

impl Bed {
    async fn start() -> Self {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let relay = KvEventRelay::new(RelayConfig {
            sources: vec![RankSource {
                rank: 0,
                endpoint,
                replay_endpoint: None,
            }],
            topic: "kv".to_string(),
            history_batches: DEFAULT_HISTORY_BATCHES,
            history_bytes: DEFAULT_HISTORY_BYTES,
            replay_timeout: Duration::from_secs(2),
            load_tick: Duration::from_millis(100),
            heartbeat_interval: Duration::from_secs(1),
            heartbeat_backoff: Duration::from_secs(5),
        });
        relay.start();
        let ring: Vec<Vec<u8>> = (0..512u64).map(payload).collect();
        // The subscription lands a moment after the connect.
        for _ in 0..500 {
            publisher.send(frame(0, &ring[0])).await.expect("publish");
            if relay.counts().relayed >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(relay.counts().relayed, 1, "the relay joined the publisher");
        Self {
            publisher,
            relay,
            next_seq: 1,
            ring,
        }
    }

    /// Publish the next batch; its sequence.
    async fn publish(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let body = &self.ring[(seq % 512) as usize];
        self.publisher
            .send(frame(seq, body))
            .await
            .expect("publish");
        seq
    }

    /// Wait until the relay has taken every batch published so far.
    async fn drained(&self) {
        for _ in 0..2_000 {
            if self.relay.counts().relayed >= self.next_seq {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!(
            "the relay took {} of {} batches",
            self.relay.counts().relayed,
            self.next_seq
        );
    }
}

type Stream =
    futures::stream::BoxStream<'static, Result<common_proto::KvEventBatch, tonic::Status>>;

/// Read `stream` until the batch of sequence `seq` arrives (the history and
/// the heartbeats before it are passed over).
async fn until(stream: &mut Stream, seq: u64) {
    loop {
        let batch = stream.next().await.expect("stream open").expect("a batch");
        if batch.sequence_number == seq && !batch.events.is_empty() {
            return;
        }
    }
}

/// Publish [`BATCHES`] batches and hand each to every stream; the allocation
/// calls of every thread over the window.
async fn calls_per_window(bed: &mut Bed, streams: &mut [Stream]) -> usize {
    // A sync batch after the subscribe: the streams are past their history.
    let sync = bed.publish().await;
    for stream in streams.iter_mut() {
        until(stream, sync).await;
    }
    bed.drained().await;
    let before = counting_alloc::total_calls();
    for _ in 0..BATCHES {
        let seq = bed.publish().await;
        for stream in streams.iter_mut() {
            until(stream, seq).await;
        }
    }
    if streams.is_empty() {
        bed.drained().await;
    }
    counting_alloc::total_calls() - before
}

#[test]
fn a_subscriber_stream_allocates_this_much_per_relayed_batch() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let mut per_batch = Vec::new();
    runtime.block_on(async {
        let mut bed = Bed::start().await;
        let mut streams: Vec<Stream> = Vec::new();
        for subscribers in 0..=2usize {
            while streams.len() < subscribers {
                let stream = bed
                    .relay
                    .subscribe(common_proto::SubscribeKvEventsRequest::default())
                    .expect("subscribed");
                streams.push(Box::pin(stream));
            }
            let calls = calls_per_window(&mut bed, &mut streams).await;
            per_batch.push(calls / BATCHES as usize);
            eprintln!(
                "allocation calls per relayed batch of {BLOCKS_PER_BATCH} blocks at {subscribers} \
                 subscriber(s): {}",
                calls / BATCHES as usize
            );
        }
    });
    let relay_alone = per_batch[0];
    let first = per_batch[1].saturating_sub(relay_alone);
    let second = per_batch[2].saturating_sub(per_batch[1]);
    eprintln!(
        "per batch: the relay alone {relay_alone}; the first subscriber adds {first}, the second \
         {second}"
    );
    assert!(
        first <= TAKER_CEILING,
        "the one stream of a relay adds {first} allocation calls per batch (ceiling \
         {TAKER_CEILING}: the batch goes out as it is, uncopied)"
    );
    assert!(
        second <= COPIER_CEILING,
        "a second stream adds {second} allocation calls per batch (ceiling {COPIER_CEILING}: \
         one copy of the batch)"
    );
}
