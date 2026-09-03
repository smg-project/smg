//! Event-bridge core: worker `SubscribeKvEvents` streams -> hash-only
//! index Updates -> one Publish stream to the index. The binary in
//! `bin/bridge.rs` is a thin flag-parsing shell over these; tests drive
//! them in-process.
//!
//! Reconnect semantics mirror the gateway monitor's: resume from the
//! last applied seq; a gap or a backend loss signal (DataLoss /
//! OutOfRange) bumps the holder's EPOCH and restarts from zero — the
//! epoch bump is what makes the restart safe to relay.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::StreamExt;
use smg_grpc_client::{common_proto, tokenspeed_scheduler::TokenSpeedSchedulerClient};
use tokio::sync::mpsc;

use crate::proto::{self, radix_index_client::RadixIndexClient};

/// Cross-task view of each holder's (epoch, seq) watermark as the
/// INDEX has acked it — the holder's STORED values, not an echo of our
/// own sends. `run_publisher` writes it from acks; worker loops consult
/// it so a restarted bridge (local epoch back at 1) adopts PAST a
/// surviving index's state instead of feeding dead-on-arrival updates
/// (liveness review's latent-bug finding — `PublishAck.epoch` exists
/// on the wire precisely for this and was ignored).
#[derive(Clone, Default)]
pub struct EpochLedger(Arc<Mutex<HashMap<String, (u64, u64)>>>);

impl EpochLedger {
    pub fn observe(&self, holder: &str, epoch: u64, seq: u64) {
        let mut map = self.0.lock().expect("epoch ledger lock");
        let entry = map.entry(holder.to_string()).or_insert((0, 0));
        // Lexicographic max: a higher epoch supersedes outright; within
        // one epoch the seq watermark only advances.
        *entry = (*entry).max((epoch, seq));
    }

    /// The holder's acked (epoch, seq) watermark; (0, 0) if never acked.
    pub fn known(&self, holder: &str) -> (u64, u64) {
        self.0
            .lock()
            .expect("epoch ledger lock")
            .get(holder)
            .copied()
            .unwrap_or((0, 0))
    }
}

/// Publisher-side digest state: chains this client has ESTABLISHED
/// with the index (sent full, not since missed), so a re-publish can
/// send a `{tip, len}` digest instead of the full block chain. The
/// stored full `Update` is the replay source: a `digest_miss_tip` ack
/// or a reconnect resends it in full — so a digest is NEVER a silent
/// under-match. Keyed by (holder, tip): the engine confirms digests
/// PER HOLDER, so a tip-only key would treat holder B's first publish
/// of a chain holder A established as "established" — B's digest then
/// misses forever while the miss-resend replays A's update.
/// Bounded; eviction just forces a future full re-send.
#[derive(Clone, Default)]
pub struct DigestCache(Arc<Mutex<HashMap<(String, u64), proto::Update>>>);

/// Cap on established chains retained per client process. Past it,
/// eviction forces full re-sends — correctness holds, cost rises.
const DIGEST_CACHE_CAP: usize = 131_072;

impl DigestCache {
    /// Decide how to publish `full` (whose chain tip is `tip`): return
    /// `Some(digest)` if the chain is already established (send that
    /// instead), or `None` to send `full` as-is (and record it).
    pub fn plan(&self, tip: u64, len: u32, full: &proto::Update) -> Option<proto::Update> {
        let mut map = self.0.lock().expect("digest cache lock");
        let key = (full.holder.clone(), tip);
        if map.contains_key(&key) {
            return Some(proto::Update {
                keyspace: full.keyspace.clone(),
                holder: full.holder.clone(),
                epoch: full.epoch,
                seq: 0,
                events: vec![proto::Event {
                    kind: Some(proto::event::Kind::StoredDigest(proto::StoredDigest {
                        parent_seq_hash: None,
                        tip_seq_hash: tip,
                        len,
                    })),
                }],
                added: None,
                dropped: false,
            });
        }
        if map.len() >= DIGEST_CACHE_CAP {
            if let Some(victim) = map.keys().next().cloned() {
                map.remove(&victim);
            }
        }
        map.insert(key, full.clone());
        None
    }

    /// The full chain to resend for a holder's missed digest tip, if
    /// retained. `None` (tip evicted or reset since) is logged by the
    /// caller: the next request's re-publish re-establishes the chain
    /// full, so the under-match is bounded to one turn, never permanent.
    pub fn resend(&self, holder: &str, tip: u64) -> Option<proto::Update> {
        self.0
            .lock()
            .expect("digest cache lock")
            .get(&(holder.to_string(), tip))
            .cloned()
    }

    /// Forget everything: after a reconnect the peer may be a different
    /// replica (or a restarted one) that does not hold these chains, so
    /// the next publishes must re-establish with full sends.
    pub fn reset(&self) {
        self.0.lock().expect("digest cache lock").clear();
    }
}

pub fn keyspace(model: &str, block_size: u32) -> proto::Keyspace {
    keyspace_with_kind(model, block_size, proto::SymbolKind::Tokens)
}

/// Keyspace for an explicit symbol kind. `Tokens` is the token-tree
/// keyspace every token-native path uses; `Bytes` is the separate,
/// server-isolated keyspace for string-mode (raw-byte) placements.
pub fn keyspace_with_kind(
    model: &str,
    block_size: u32,
    symbol_kind: proto::SymbolKind,
) -> proto::Keyspace {
    proto::Keyspace {
        model: model.to_string(),
        symbol_kind: symbol_kind as i32,
        block_size,
        hash_scheme: crate::wire_hash::HASH_SCHEME_V1,
    }
}

pub fn convert_batch(
    batch: &common_proto::KvEventBatch,
    model: &str,
    block_size: u32,
    holder: &str,
    epoch: u64,
) -> proto::Update {
    let events = batch
        .events
        .iter()
        .filter_map(|event| event.data.as_ref())
        .map(|data| {
            let kind = match data {
                common_proto::kv_cache_event::Data::Stored(stored) => {
                    proto::event::Kind::Stored(proto::Stored {
                        parent_seq_hash: stored.parent_block_hash.map(|p| p as u64),
                        blocks: stored
                            .blocks
                            .iter()
                            .map(|b| proto::Block {
                                seq_hash: b.block_hash as u64,
                                content_hash: crate::wire_hash::content_hash(&b.token_ids).0,
                            })
                            .collect(),
                    })
                }
                common_proto::kv_cache_event::Data::Removed(removed) => {
                    proto::event::Kind::Removed(proto::Removed {
                        seq_hashes: removed.block_hashes.iter().map(|&h| h as u64).collect(),
                    })
                }
                common_proto::kv_cache_event::Data::Cleared(_) => proto::event::Kind::Cleared(true),
            };
            proto::Event { kind: Some(kind) }
        })
        .collect();
    proto::Update {
        keyspace: Some(keyspace(model, block_size)),
        holder: holder.to_string(),
        epoch,
        seq: batch.sequence_number,
        events,
        added: None,
        dropped: false,
    }
}

/// Is this feed generation dead on arrival at the index — and if so, the
/// epoch to restart at? `known` is the holder's acked (epoch, seq)
/// watermark from the ledger.
///
/// - Index on a STRICTLY higher epoch: a previous life of this bridge
///   (or another authority) advanced the holder, so our sends are
///   Deduped. Adopt one past it.
/// - Same epoch but the index's seq cursor is AHEAD of what we sent:
///   bridge and worker both restarted into an old generation, and the
///   dedup watermark silently swallows our fresh low seqs. Only a new
///   epoch (implicit clear + replay from zero) restarts losslessly.
///
/// In steady state acks TRAIL our sends (kepoch == local_epoch and
/// kseq <= local_seq), so this never self-triggers — a `>=` epoch
/// compare here once bumped the epoch after nearly every acked batch,
/// wiping and refeeding the holder forever.
fn adopt_epoch(local_epoch: u64, local_seq: u64, known: (u64, u64)) -> Option<u64> {
    let (kepoch, kseq) = known;
    if kepoch > local_epoch {
        return Some(kepoch + 1);
    }
    if kepoch == local_epoch && kseq > local_seq {
        return Some(local_epoch + 1);
    }
    None
}

/// One worker's subscription loop: resume on plain failures, epoch-bump
/// on loss signals or sequence gaps. Runs until the publish channel
/// closes or the worker reports Unimplemented.
pub async fn worker_loop(
    worker: String,
    model: String,
    block_size: u32,
    out: mpsc::Sender<proto::Update>,
    ledger: EpochLedger,
) {
    let mut epoch: u64 = 1;
    let mut last_seq: u64 = 0;
    loop {
        // Adopt past whatever the index has acked for this holder: a
        // stale local generation means every update we send is dead on
        // arrival. Adoption is a new generation, so replay from zero
        // (the resubscribe below starts at `last_seq`).
        if let Some(adopted) = adopt_epoch(epoch, last_seq, ledger.known(&worker)) {
            epoch = adopted;
            last_seq = 0;
        }
        let Ok(client) = TokenSpeedSchedulerClient::connect(&worker).await else {
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        };
        let mut stream = match client.subscribe_kv_events(last_seq).await {
            Ok(stream) => stream,
            Err(status) => {
                match status.code() {
                    // Terminal per the monitor contract.
                    tonic::Code::Unimplemented => {
                        tracing::warn!(%worker, "KV events unimplemented; bridge exits for this worker");
                        return;
                    }
                    // Cursor lost: new generation, replay from zero.
                    tonic::Code::OutOfRange | tonic::Code::DataLoss => {
                        epoch += 1;
                        last_seq = 0;
                    }
                    _ => {}
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        };
        while let Some(batch) = stream.next().await {
            let Ok(batch) = batch else { break };
            if last_seq > 0 && batch.sequence_number <= last_seq {
                continue; // duplicate replay
            }
            if last_seq > 0 && batch.sequence_number > last_seq + 1 {
                // Gap: the ring may have wrapped; new generation.
                epoch += 1;
                last_seq = 0;
                break;
            }
            last_seq = batch.sequence_number;
            let update = convert_batch(&batch, &model, block_size, &worker, epoch);
            if out.send(update).await.is_err() {
                return; // publisher gone; process exiting
            }
            // Mid-stream adoption: acks arrive async, and a healthy
            // stream never reconnects on its own — without this check
            // a stale-generation bridge would keep feeding deduped
            // updates forever. Steady-state acks trail `last_seq`, so
            // this only fires when the index is genuinely ahead of us.
            if adopt_epoch(epoch, last_seq, ledger.known(&worker)).is_some() {
                break; // outer loop adopts and resubscribes from zero
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// The publish pump: drain `rx` into one (re)connected Publish stream to
/// `index`. The receiver persists across reconnects, so no update is
/// lost inside the bridge. Returns when all worker loops have ended.
pub async fn run_publisher(rx: mpsc::Receiver<proto::Update>, index: String, ledger: EpochLedger) {
    run_publisher_with_digest(rx, index, ledger, None).await
}

/// As `run_publisher`, plus optional publisher-side digest support:
/// on reconnect the cache is reset (the peer may not hold prior
/// chains), and a `digest_miss_tip` ack resends that chain in full.
pub async fn run_publisher_with_digest(
    mut rx: mpsc::Receiver<proto::Update>,
    index: String,
    ledger: EpochLedger,
    digest: Option<DigestCache>,
) {
    loop {
        let Ok(client) = RadixIndexClient::connect(index.clone()).await else {
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        };
        let mut client = client
            .max_decoding_message_size(64 * 1024 * 1024)
            .max_encoding_message_size(64 * 1024 * 1024);
        let (fwd_tx, fwd_rx) = mpsc::channel::<proto::Update>(1024);
        let outbound = tokio_stream::wrappers::ReceiverStream::new(fwd_rx);
        let mut acks = match client.publish(tonic::Request::new(outbound)).await {
            Ok(response) => response.into_inner(),
            Err(error) => {
                tracing::warn!(%error, "publish stream failed; retrying");
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        };
        // New connection: the peer may be a different or restarted
        // replica, so no chain can be assumed established.
        if let Some(cache) = &digest {
            cache.reset();
        }
        loop {
            tokio::select! {
                item = rx.recv() => match item {
                    Some(update) => {
                        if fwd_tx.send(update).await.is_err() {
                            break;
                        }
                    }
                    // All worker loops ended (fleet torn down).
                    None => return,
                },
                ack = acks.next() => match ack {
                    Some(Ok(ack)) => {
                        ledger.observe(&ack.holder, ack.epoch, ack.applied_seq);
                        // A digest the index could not confirm: resend
                        // the chain in full — never a silent under-match.
                        if let (Some(cache), Some(tip)) = (&digest, ack.digest_miss_tip) {
                            match cache.resend(&ack.holder, tip) {
                                Some(full) => {
                                    if fwd_tx.send(full).await.is_err() {
                                        break;
                                    }
                                }
                                None => {
                                    // Evicted/reset since the digest was
                                    // queued: unrecoverable HERE, but the
                                    // next request's publish re-establishes
                                    // the chain full (the cache no longer
                                    // plans a digest for it) — bounded to
                                    // one turn, and never silent.
                                    tracing::warn!(
                                        holder = %ack.holder,
                                        tip,
                                        "digest miss with no retained chain; next publish re-establishes full"
                                    );
                                }
                            }
                        }
                    }
                    Some(Err(_)) | None => break,
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Epoch adoption arithmetic. Two rules, and one absence: adopt one
    /// past a STRICTLY higher acked epoch; adopt on a same-epoch acked
    /// seq AHEAD of our sends (the both-restarted collision the dedup
    /// watermark would otherwise silently swallow); and NEVER
    /// self-trigger in steady state — a `>=` epoch compare here once
    /// bumped the epoch after nearly every acked batch, wiping and
    /// refeeding the holder forever.
    #[test]
    fn adopt_epoch_fires_on_stale_generations_only() {
        // Restarted bridge (local 1) vs a surviving index whose holder
        // is stored at epoch 7 -> adopt 8.
        assert_eq!(adopt_epoch(1, 0, (7, 42)), Some(8));
        // Both bridge and worker restarted into an old generation: same
        // epoch, but the index's seq cursor (100) is ahead of what this
        // generation has sent (1) -> our low seqs are being swallowed;
        // start a fresh epoch.
        assert_eq!(adopt_epoch(1, 1, (1, 100)), Some(2));
        // STEADY STATE never self-triggers: acks trail our sends.
        assert_eq!(adopt_epoch(5, 40, (5, 40)), None, "ack caught up");
        assert_eq!(adopt_epoch(5, 40, (5, 38)), None, "acks lagging");
        // We are already ahead of everything acked: keep our epoch.
        assert_eq!(adopt_epoch(8, 0, (7, 500)), None);
        // Nothing acked yet while we are at the initial epoch: keep it.
        assert_eq!(adopt_epoch(1, 0, (0, 0)), None);
    }

    /// The ledger tracks the running lexicographic max (epoch, seq) per
    /// holder — a later lower/reordered ack must not lower it.
    #[test]
    fn epoch_ledger_keeps_the_running_max() {
        let ledger = EpochLedger::default();
        assert_eq!(ledger.known("w1"), (0, 0), "unknown holder is zero");
        ledger.observe("w1", 7, 10);
        ledger.observe("w1", 3, 999); // stale epoch: must not lower
        assert_eq!(ledger.known("w1"), (7, 10));
        ledger.observe("w1", 7, 8); // reordered seq: must not lower
        assert_eq!(ledger.known("w1"), (7, 10));
        ledger.observe("w1", 7, 25);
        assert_eq!(ledger.known("w1"), (7, 25));
        ledger.observe("w1", 9, 1); // higher epoch supersedes outright
        assert_eq!(ledger.known("w1"), (9, 1));
        assert_eq!(ledger.known("w2"), (0, 0), "holders are independent");
    }

    fn full_update(tip: u64, holder: &str) -> proto::Update {
        proto::Update {
            keyspace: Some(keyspace("m", 4)),
            holder: holder.to_string(),
            epoch: 1,
            seq: 0,
            events: vec![proto::Event {
                kind: Some(proto::event::Kind::Stored(proto::Stored {
                    parent_seq_hash: None,
                    blocks: vec![proto::Block {
                        seq_hash: tip,
                        content_hash: tip,
                    }],
                })),
            }],
            added: None,
            dropped: false,
        }
    }

    /// Digest lifecycle: first publish of a chain sends full (None) and
    /// records it; a re-publish sends a `{tip, len}` digest; the recorded
    /// full is retained for resend-on-miss; and a reset (reconnect) forces
    /// the next publish to re-establish full — so a digest is never a
    /// silent under-match.
    #[test]
    fn digest_cache_establishes_digests_resends_and_resets() {
        let cache = DigestCache::default();
        let tip = 0xABCDu64;
        let full = full_update(tip, "w1");

        // First time: send full, record it.
        assert!(cache.plan(tip, 1, &full).is_none(), "first publish is full");
        // Established: a re-publish is a digest carrying {tip, len}.
        let digest = cache.plan(tip, 1, &full).expect("re-publish is a digest");
        match digest.events.as_slice() {
            [proto::Event {
                kind: Some(proto::event::Kind::StoredDigest(d)),
            }] => {
                assert_eq!(d.tip_seq_hash, tip);
                assert_eq!(d.len, 1);
                assert_eq!(d.parent_seq_hash, None);
            }
            other => panic!("expected a single StoredDigest event, got {other:?}"),
        }
        assert_eq!(digest.seq, 0, "placements/digests are unsequenced");

        // Resend recovers the full chain for a missed tip; unknown -> None.
        let resent = cache.resend("w1", tip).expect("retained full for resend");
        assert!(matches!(
            resent.events.as_slice(),
            [proto::Event {
                kind: Some(proto::event::Kind::Stored(_))
            }]
        ));
        assert!(
            cache.resend("w1", 0xDEAD).is_none(),
            "unknown tip is not resendable"
        );

        // Keying is PER HOLDER: the same chain published for a second
        // holder is NOT established (the engine confirms per holder), so
        // its first publish must go out full — and a miss for w2's tip
        // must never replay w1's update.
        let full_w2 = full_update(tip, "w2");
        assert!(
            cache.plan(tip, 1, &full_w2).is_none(),
            "same tip under a new holder re-establishes full"
        );
        assert_eq!(cache.resend("w2", tip).expect("w2 retained").holder, "w2");

        // Reconnect reset: the peer may not hold prior chains, so the
        // next publish must re-establish full.
        cache.reset();
        assert!(
            cache.plan(tip, 1, &full).is_none(),
            "after reset the chain re-establishes full"
        );
    }

    /// `convert_batch` maps the worker's Removed and Cleared events, not
    /// just Stored — a removed/cleared holder must be told to the index or
    /// it keeps scoring gone blocks. Only Stored was exercised before.
    #[test]
    fn convert_batch_maps_removed_and_cleared() {
        let batch = common_proto::KvEventBatch {
            sequence_number: 5,
            timestamp: 0.0,
            events: vec![
                common_proto::KvCacheEvent {
                    event_id: 1,
                    data: Some(common_proto::kv_cache_event::Data::Removed(
                        common_proto::KvBlocksRemoved {
                            block_hashes: vec![10, 20],
                            cache_level: None,
                        },
                    )),
                },
                common_proto::KvCacheEvent {
                    event_id: 2,
                    data: Some(common_proto::kv_cache_event::Data::Cleared(
                        common_proto::KvCacheCleared {},
                    )),
                },
            ],
            dp_rank: Some(0),
        };
        let update = convert_batch(&batch, "m", 4, "w1", 3);
        assert_eq!(update.seq, 5);
        assert_eq!(update.epoch, 3);
        assert_eq!(update.holder, "w1");
        match &update.events[0].kind {
            Some(proto::event::Kind::Removed(r)) => assert_eq!(r.seq_hashes, vec![10u64, 20]),
            other => panic!("expected Removed, got {other:?}"),
        }
        match &update.events[1].kind {
            Some(proto::event::Kind::Cleared(_)) => {}
            other => panic!("expected Cleared, got {other:?}"),
        }
    }
}
