//! Gateway-side client: one persistent Subscribe stream for queries
//! (query_id-correlated, caller-enforced deadline) and one Publish
//! stream for fire-and-forget placements. Both reconnect forever in
//! background drivers; a query during an outage resolves Disconnected
//! and the caller falls through to its local fallback.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};

use crate::{
    bridge,
    engine::placement_chain,
    proto::{self, radix_index_client::RadixIndexClient},
    ContentHash,
};

/// What a routing-time query resolved to; the caller maps this onto its
/// fallback ladder and metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryOutcome {
    /// Per-holder (url, matched_blocks), descending.
    Scores(Vec<(String, u32)>),
    /// The index answered with no overlap.
    Empty,
    /// Deadline elapsed; the late answer is dropped by id.
    Timeout,
    /// No live stream (index down / reconnecting).
    Disconnected,
}

/// Bound on a lifecycle send into the publish queue. Generous against a
/// momentary full queue; small enough that an unreachable index cannot
/// wedge the worker-removal workflow.
const LIFECYCLE_SEND_DEADLINE: Duration = Duration::from_secs(2);

struct PendingQuery {
    query: proto::Query,
    reply: oneshot::Sender<proto::Match>,
}

pub struct RemoteIndex {
    queries: mpsc::Sender<PendingQuery>,
    placements: mpsc::Sender<proto::Update>,
    next_id: AtomicU64,
    /// Flipped by the subscribe driver on stream up/down. While false, a
    /// query resolves Disconnected immediately instead of burning its
    /// deadline on a stream that cannot answer.
    connected: Arc<AtomicBool>,
    /// `Some` when digest publishing is enabled: a re-publish of an
    /// established chain sends a `{tip, len}` digest instead of its
    /// blocks (the index confirms it with one lookup, or misses and
    /// the publisher resends full). Opt-in — off, every publish is a
    /// full chain, byte-identical to the pre-digest client.
    digest: Option<bridge::DigestCache>,
}

impl RemoteIndex {
    /// Lazy client: drivers connect (and reconnect) in the background.
    pub fn connect(url: String) -> Arc<Self> {
        // Digest publishing is opt-in (RADIX_CLIENT_DIGEST=1) until the
        // sim rig validates it end to end, mirroring how the whole
        // remote index sits behind --kv-indexer-url.
        let digest_on = std::env::var("RADIX_CLIENT_DIGEST").as_deref() == Ok("1");
        Self::connect_with(url, digest_on)
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "client-lifetime driver tasks; the owner holds the Arc for the process lifetime"
    )]
    pub fn connect_with(url: String, digest_on: bool) -> Arc<Self> {
        let (query_tx, query_rx) = mpsc::channel::<PendingQuery>(4096);
        let (placement_tx, placement_rx) = mpsc::channel::<proto::Update>(65_536);
        let connected = Arc::new(AtomicBool::new(false));
        let digest = digest_on.then(bridge::DigestCache::default);
        tokio::spawn(subscribe_driver(
            url.clone(),
            query_rx,
            Arc::clone(&connected),
        ));
        tokio::spawn(bridge::run_publisher_with_digest(
            placement_rx,
            url,
            bridge::EpochLedger::default(),
            digest.clone(),
        ));
        Arc::new(Self {
            queries: query_tx,
            placements: placement_tx,
            next_id: AtomicU64::new(1),
            connected,
            digest,
        })
    }

    /// Whether the subscribe stream is currently established.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Overlap query with a hard deadline. Never blocks longer than
    /// `deadline`; every non-Scores outcome is a signal to fall back.
    /// Token keyspace — the token-native routing paths.
    pub async fn query(
        &self,
        model: &str,
        block_size: u32,
        content_hashes: Vec<u64>,
        deadline: Duration,
    ) -> QueryOutcome {
        self.query_kind(
            model,
            block_size,
            content_hashes,
            deadline,
            proto::SymbolKind::Tokens,
        )
        .await
    }

    /// Overlap query against the string-mode (`Bytes`) keyspace — for
    /// HTTP requests routed on raw-text byte chunks rather than tokens.
    pub async fn query_bytes(
        &self,
        model: &str,
        byte_block: u32,
        content_hashes: Vec<u64>,
        deadline: Duration,
    ) -> QueryOutcome {
        self.query_kind(
            model,
            byte_block,
            content_hashes,
            deadline,
            proto::SymbolKind::Bytes,
        )
        .await
    }

    async fn query_kind(
        &self,
        model: &str,
        block_size: u32,
        content_hashes: Vec<u64>,
        deadline: Duration,
        symbol_kind: proto::SymbolKind,
    ) -> QueryOutcome {
        if !self.is_connected() {
            return QueryOutcome::Disconnected;
        }
        let query_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply_tx, reply_rx) = oneshot::channel();
        let pending = PendingQuery {
            query: proto::Query {
                query_id,
                keyspace: Some(bridge::keyspace_with_kind(model, block_size, symbol_kind)),
                content_hashes,
            },
            reply: reply_tx,
        };
        if self.queries.try_send(pending).is_err() {
            return QueryOutcome::Disconnected;
        }
        match tokio::time::timeout(deadline, reply_rx).await {
            Ok(Ok(answer)) => {
                let scores: Vec<(String, u32)> = answer
                    .scores
                    .into_iter()
                    .map(|s| (s.holder, s.matched_blocks))
                    .collect();
                if scores.is_empty() {
                    QueryOutcome::Empty
                } else {
                    QueryOutcome::Scores(scores)
                }
            }
            Ok(Err(_)) => QueryOutcome::Disconnected,
            Err(_) => QueryOutcome::Timeout,
        }
    }

    /// Fire-and-forget placement: the request's block-hash chain now
    /// (probably) resides on `holder`. Never blocks; dropped on overflow
    /// (the next turn re-places it). Token keyspace.
    pub fn publish_placement(
        &self,
        model: &str,
        block_size: u32,
        holder: &str,
        content_hashes: &[u64],
    ) {
        self.publish_placement_kind(
            model,
            block_size,
            holder,
            content_hashes,
            proto::SymbolKind::Tokens,
        );
    }

    /// String-mode placement against the `Bytes` keyspace — the raw-text
    /// byte chain now (probably) resides on `holder`.
    pub fn publish_placement_bytes(
        &self,
        model: &str,
        byte_block: u32,
        holder: &str,
        content_hashes: &[u64],
    ) {
        self.publish_placement_kind(
            model,
            byte_block,
            holder,
            content_hashes,
            proto::SymbolKind::Bytes,
        );
    }

    fn publish_placement_kind(
        &self,
        model: &str,
        block_size: u32,
        holder: &str,
        content_hashes: &[u64],
        symbol_kind: proto::SymbolKind,
    ) {
        if content_hashes.is_empty() {
            return;
        }
        let hashes: Vec<ContentHash> = content_hashes.iter().copied().map(ContentHash).collect();
        let chain = placement_chain(&hashes);
        let tip = chain.last().expect("non-empty").seq_hash.0;
        let blocks = chain
            .into_iter()
            .map(|b| proto::Block {
                seq_hash: b.seq_hash.0,
                content_hash: b.content_hash.0,
            })
            .collect();
        let update = proto::Update {
            keyspace: Some(bridge::keyspace_with_kind(model, block_size, symbol_kind)),
            holder: holder.to_string(),
            // Placements are unsequenced (seq 0) and epoch-constant: an
            // event-fed holder rejects them regardless, and inferred-only
            // holders never bump epochs.
            epoch: 1,
            seq: 0,
            events: vec![proto::Event {
                kind: Some(proto::event::Kind::Stored(proto::Stored {
                    parent_seq_hash: None,
                    blocks,
                })),
            }],
            added: None,
            dropped: false,
        };
        // With digest enabled, an already-established chain publishes as
        // a tiny {tip, len} digest; the run_publisher ack loop resends
        // `update` in full on a miss. The digest cache is keyed by tip
        // hash alone, with no symbol-kind qualifier, so only the Tokens
        // path uses it; string-mode (`Bytes`) placements always send the
        // full chain and never touch the shared cache.
        let to_send = match (&self.digest, symbol_kind) {
            (Some(cache), proto::SymbolKind::Tokens) => cache
                .plan(tip, content_hashes.len() as u32, &update)
                .unwrap_or(update),
            _ => update,
        };
        let _ = self.placements.try_send(to_send);
    }

    /// Fleet-membership lifecycle: soft-retire `holder` (stop scoring
    /// it; state expires by TTL). Published by the gateway's
    /// worker-removal workflow — the same signal that purges the
    /// local indexer. Awaited under a DEADLINE (unlike placements'
    /// fire-and-forget): a lifecycle signal should not be droppable on
    /// a momentarily full queue, but the caller is the worker-removal
    /// workflow, and an advisory routing index must never wedge the
    /// control plane behind an unreachable index — on timeout the
    /// engine's silence backstop (`event_ttl`) is the designed
    /// self-heal for the lost signal.
    pub async fn publish_dropped(&self, model: &str, block_size: u32, holder: &str) {
        self.send_lifecycle(lifecycle(model, block_size, holder, true))
            .await;
    }

    /// Fleet-membership lifecycle: (re)announce `holder` — heals a
    /// same-URL rejoin out of a standing soft-retire.
    pub async fn publish_added(&self, model: &str, block_size: u32, holder: &str) {
        self.send_lifecycle(lifecycle(model, block_size, holder, false))
            .await;
    }

    async fn send_lifecycle(&self, update: proto::Update) {
        let holder = update.holder.clone();
        let dropped = update.dropped;
        if tokio::time::timeout(LIFECYCLE_SEND_DEADLINE, self.placements.send(update))
            .await
            .is_err()
        {
            tracing::warn!(
                %holder,
                dropped,
                "remote-index lifecycle signal timed out (queue full / index unreachable); \
                 the engine's TTL backstop covers the lost signal"
            );
        }
    }
}

fn lifecycle(model: &str, block_size: u32, holder: &str, dropped: bool) -> proto::Update {
    proto::Update {
        keyspace: Some(bridge::keyspace(model, block_size)),
        holder: holder.to_string(),
        // Control payloads apply independent of the epoch gate, so the
        // constant epoch is safe even against a bridge-advanced holder.
        epoch: 1,
        seq: 0,
        events: Vec::new(),
        added: (!dropped).then_some(proto::Added {
            capacity_blocks: 0,
            event_fed: false,
            metadata: Vec::new(),
        }),
        dropped,
    }
}

/// Owns the Subscribe bidi stream and the pending-answer map; reconnects
/// forever. On disconnect all pending replies drop (callers resolve
/// Disconnected); late answers for abandoned ids are discarded by the
/// map lookup.
async fn subscribe_driver(
    url: String,
    mut queries: mpsc::Receiver<PendingQuery>,
    connected: Arc<AtomicBool>,
) {
    loop {
        let Ok(client) = RadixIndexClient::connect(url.clone()).await else {
            drain_disconnected(&mut queries);
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        };
        let mut client = client
            .max_decoding_message_size(64 * 1024 * 1024)
            .max_encoding_message_size(64 * 1024 * 1024);
        let (fwd_tx, fwd_rx) = mpsc::channel::<proto::Query>(1024);
        let outbound = tokio_stream::wrappers::ReceiverStream::new(fwd_rx);
        let mut answers = match client.subscribe(tonic::Request::new(outbound)).await {
            Ok(response) => response.into_inner(),
            Err(_) => {
                drain_disconnected(&mut queries);
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        };
        connected.store(true, Ordering::Relaxed);
        let mut pending: HashMap<u64, oneshot::Sender<proto::Match>> = HashMap::new();
        // A query whose answer is shed by the server (try_send Full) or
        // otherwise lost never gets a matching Match, so its entry would
        // linger for the life of the connection — an unbounded map under
        // sustained shedding. Sweep entries whose caller already timed
        // out (receiver dropped => Sender::is_closed) on a slow tick.
        let mut evict = tokio::time::interval(PENDING_EVICT_INTERVAL);
        evict.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                item = queries.recv() => match item {
                    Some(PendingQuery { query, reply }) => {
                        let id = query.query_id;
                        if fwd_tx.send(query).await.is_err() {
                            break; // stream gone; reply drops -> Disconnected
                        }
                        pending.insert(id, reply);
                    }
                    None => return, // client dropped
                },
                answer = answers.next() => match answer {
                    Some(Ok(answer)) => {
                        if let Some(reply) = pending.remove(&answer.query_id) {
                            let _ = reply.send(answer);
                        }
                    }
                    _ => break, // stream error/closed; reconnect
                },
                _ = evict.tick() => {
                    evict_timed_out(&mut pending);
                }
            }
        }
        // Pending replies drop here -> callers resolve Disconnected;
        // new queries fast-fail on the flag until the stream is back.
        connected.store(false, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// While there is no live stream, immediately fail queued queries so
/// callers hit their fallback instead of their deadline.
fn drain_disconnected(queries: &mut mpsc::Receiver<PendingQuery>) {
    while let Ok(PendingQuery { reply, .. }) = queries.try_recv() {
        drop(reply);
    }
}

/// How often the subscribe driver sweeps abandoned pending-answer slots.
/// A query's caller deadline is single-digit milliseconds, so a 1s sweep
/// keeps the map bounded by roughly one interval of genuinely in-flight
/// queries even when every answer is being shed.
const PENDING_EVICT_INTERVAL: Duration = Duration::from_secs(1);

/// Drop pending slots whose caller has already timed out (the receiver
/// was dropped, so the `oneshot::Sender` reports closed). Bounds the map
/// against lost/shed answers that would otherwise never remove their id.
/// Returns the number evicted.
fn evict_timed_out(pending: &mut HashMap<u64, oneshot::Sender<proto::Match>>) -> usize {
    let before = pending.len();
    pending.retain(|_, reply| !reply.is_closed());
    before - pending.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pending-answer map must shed slots whose caller has timed out
    /// (receiver dropped) while keeping still-live ones — otherwise a
    /// shed/lost answer leaks its id for the life of the connection.
    #[test]
    fn evict_timed_out_drops_only_abandoned_slots() {
        let mut pending: HashMap<u64, oneshot::Sender<proto::Match>> = HashMap::new();

        // Two callers time out (receiver dropped); one is still waiting.
        let (tx_gone_a, rx_a) = oneshot::channel::<proto::Match>();
        let (tx_gone_b, rx_b) = oneshot::channel::<proto::Match>();
        let (tx_live, _rx_live) = oneshot::channel::<proto::Match>();
        drop(rx_a);
        drop(rx_b);
        pending.insert(1, tx_gone_a);
        pending.insert(2, tx_gone_b);
        pending.insert(3, tx_live);

        assert_eq!(
            evict_timed_out(&mut pending),
            2,
            "two abandoned slots evicted"
        );
        assert_eq!(pending.len(), 1);
        assert!(pending.contains_key(&3), "the live caller's slot is kept");

        // Idempotent: nothing left to evict.
        assert_eq!(evict_timed_out(&mut pending), 0);
    }
}
