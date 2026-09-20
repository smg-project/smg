//! The gRPC surface: Publish (apply + best-effort peer relay),
//! Subscribe (query stream), Pull (state as synthetic Updates for
//! replica bootstrap). Relay and bootstrap speak the same Update
//! vocabulary as publishers, so replicas copy — they never agree.

use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tonic::{transport::Server, Request, Response, Status, Streaming};

use crate::{
    engine::{Applied, ApplyOutcome, Engine, HolderDigest, KeyspaceKey, SymbolKind},
    proto::{
        self,
        radix_index_client::RadixIndexClient,
        radix_index_server::{RadixIndex, RadixIndexServer},
    },
    ContentHash, UpdateMsg, WireEvent,
};

type AckStream = Pin<Box<dyn Stream<Item = Result<proto::PublishAck, Status>> + Send>>;
type MatchStream = Pin<Box<dyn Stream<Item = Result<proto::Match, Status>> + Send>>;
type PullStream = Pin<Box<dyn Stream<Item = Result<proto::Update, Status>> + Send>>;

/// Bound on the per-peer relay queue; overflowing drops updates
/// (divergence is bounded by TTL + re-placement, and a wedged peer
/// must not wedge ingest). Overridable via RADIX_RELAY_QUEUE for
/// overflow drills.
const RELAY_QUEUE: usize = 65_536;

fn relay_queue_len() -> usize {
    parse_relay_queue(std::env::var("RADIX_RELAY_QUEUE").ok().as_deref())
}

/// Unset keeps the default; a value that IS set must be a positive
/// integer. Fatal on anything else, like the flag parser in
/// [`crate::cli`]: a typo'd bound (`64k`) that fell back to 65_536 would
/// leave an overflow drill measuring the default depth and reporting a
/// false negative, and `0` reaches `mpsc::channel(0)`, which panics
/// inside service construction rather than here at startup.
fn parse_relay_queue(raw: Option<&str>) -> usize {
    let Some(raw) = raw else {
        return RELAY_QUEUE;
    };
    let len = raw.trim().parse::<usize>().unwrap_or(0);
    assert!(
        len > 0,
        "RADIX_RELAY_QUEUE must be a positive integer, got {raw:?}"
    );
    len
}

/// Process counters for the metrics endpoint. Shared between the gRPC
/// service and the admin listener.
#[derive(Debug, Default)]
pub struct ServiceStats {
    pub applies: AtomicU64,
    pub queries: AtomicU64,
    pub relay_dropped: AtomicU64,
    /// Engine time for one applied BATCH (the write path's own cost, not
    /// including transport or queueing). Observed once per batch, never
    /// divided across its members: one apply stalling 30 ms behind a
    /// keyspace write lock inside a 256-update batch has to land in the
    /// 25/50 ms buckets, but dividing first files all 256 observations
    /// under 250 µs — and batches only reach 256 UNDER write pressure,
    /// so the divisor is largest exactly when latency is worst and p99
    /// would flatten as the system degrades. Per-update cost stays
    /// exact as `_sum / applies_total`, with the tail still readable.
    pub apply_batch_latency: LatencyHistogram,
    /// Engine time per answered query (the read path's own cost).
    pub query_latency: LatencyHistogram,
    /// Anti-entropy rounds completed against a peer, holders replaced
    /// from a peer because it was provably ahead, and rounds that failed
    /// before they could compare anything. The failure count is the only
    /// outward sign that the backstop is broken: `anti_entropy_rounds`
    /// moves only after the digests call succeeds, so a wrong peer URL
    /// or a peer too old to serve `Digests` otherwise looks exactly like
    /// a converged fleet.
    pub anti_entropy_rounds: AtomicU64,
    pub anti_entropy_holders_pulled: AtomicU64,
    pub anti_entropy_failures: AtomicU64,
    /// Flipped true once the bootstrap pull (if any) has completed; the
    /// admin listener's /readyz reports it.
    pub ready: AtomicBool,
}

/// Upper bounds of the latency buckets, in microseconds. Spans the
/// tens-of-µs an in-memory apply or query takes to the tens of ms a
/// lock convoy would show. ENGINE time only: a gateway's query deadline
/// is a round-trip budget (serialization, network, queueing behind other
/// work on the shared runtime, the ack hop back), and those are the
/// parts that miss it under load — so this histogram can read entirely
/// under 2 ms while the caller is timing out, and "over deadline" is
/// only answerable from the client side.
const LATENCY_BUCKETS_US: [u64; 12] = [
    10, 25, 50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000,
];

/// A fixed-bucket, lock-free latency histogram in the Prometheus shape
/// (cumulative `le` buckets plus `+Inf`, `_sum` in seconds, `_count`).
#[derive(Debug, Default)]
pub struct LatencyHistogram {
    buckets: [AtomicU64; LATENCY_BUCKETS_US.len() + 1],
    sum_ns: AtomicU64,
    count: AtomicU64,
}

impl LatencyHistogram {
    pub fn observe(&self, elapsed: Duration) {
        // Nanoseconds, not `as_micros()`: truncation drops up to 1 µs per
        // observation, and anything genuinely sub-µs truncates to zero —
        // `_sum` would stay flat while `_count` climbed and the derived
        // average would render as exactly 0, which reads as a broken
        // exporter rather than as a fast path. Bucketing compares in ns
        // too, so an 11 µs sample cannot round down into the 10 µs edge.
        let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let idx = LATENCY_BUCKETS_US
            .iter()
            .position(|&bound| ns <= bound.saturating_mul(1_000))
            .unwrap_or(LATENCY_BUCKETS_US.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Prometheus text for a histogram named `name` (seconds).
    pub fn render(&self, name: &str) -> String {
        let mut out = format!("# TYPE {name} histogram\n");
        let mut cumulative = 0u64;
        for (idx, bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            cumulative += self.buckets[idx].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{le=\"{}\"}} {cumulative}\n",
                *bound as f64 / 1_000_000.0
            ));
        }
        cumulative += self.buckets[LATENCY_BUCKETS_US.len()].load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!(
            "{name}_sum {}\n{name}_count {}\n",
            self.sum_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.count.load(Ordering::Relaxed)
        ));
        out
    }
}

pub struct IndexService {
    engine: Arc<Engine>,
    stats: Arc<ServiceStats>,
    relay: Vec<mpsc::Sender<proto::Update>>,
    /// Staleness injection for the experiment's sweep: delay applied
    /// before Stored / Removed events land in the engine. Zero = off.
    delay_stored: Duration,
    delay_removed: Duration,
}

impl IndexService {
    /// `peers`: sibling replica endpoints to relay Publishes to (empty
    /// for single-replica runs or when publishers fan out themselves).
    /// Relay is async and best-effort: a wedged peer drops updates, and
    /// epoch/seq dedup plus TTL/re-placement bound the divergence.
    pub fn new(engine: Arc<Engine>, peers: Vec<String>) -> Self {
        Self::with_delays(engine, peers, Duration::ZERO, Duration::ZERO)
    }

    pub fn with_delays(
        engine: Arc<Engine>,
        peers: Vec<String>,
        delay_stored: Duration,
        delay_removed: Duration,
    ) -> Self {
        Self::with_stats(
            engine,
            Arc::new(ServiceStats::default()),
            peers,
            delay_stored,
            delay_removed,
        )
    }

    pub fn with_stats(
        engine: Arc<Engine>,
        stats: Arc<ServiceStats>,
        peers: Vec<String>,
        delay_stored: Duration,
        delay_removed: Duration,
    ) -> Self {
        let relay = peers.into_iter().map(spawn_relay).collect();
        Self {
            engine,
            stats,
            relay,
            delay_stored,
            delay_removed,
        }
    }
}

/// One background relay: queue -> (re)connected Publish stream to `peer`.
#[expect(
    clippy::disallowed_methods,
    reason = "service-lifetime task; the index process is its own supervisor"
)]
fn spawn_relay(peer: String) -> mpsc::Sender<proto::Update> {
    let (tx, mut rx) = mpsc::channel::<proto::Update>(relay_queue_len());
    tokio::spawn(async move {
        loop {
            match RadixIndexClient::connect(peer.clone()).await {
                Ok(client) => {
                    let mut client = client
                        .max_decoding_message_size(64 * 1024 * 1024)
                        .max_encoding_message_size(64 * 1024 * 1024);
                    let (fwd_tx, fwd_rx) = mpsc::channel::<proto::Update>(1024);
                    let outbound = tokio_stream::wrappers::ReceiverStream::new(fwd_rx);
                    let mut acks = match client.publish(Request::new(outbound)).await {
                        Ok(response) => response.into_inner(),
                        Err(error) => {
                            tracing::warn!(%peer, %error, "relay publish failed; retrying");
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            continue;
                        }
                    };
                    loop {
                        tokio::select! {
                            item = rx.recv() => match item {
                                Some(update) => {
                                    if fwd_tx.send(update).await.is_err() {
                                        break; // stream torn down; reconnect
                                    }
                                }
                                None => return, // service dropped
                            },
                            ack = acks.next() => {
                                if ack.is_none() {
                                    break; // peer closed; reconnect
                                }
                            }
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%peer, %error, "relay connect failed; retrying");
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
    tx
}

#[tonic::async_trait]
impl RadixIndex for IndexService {
    type PublishStream = AckStream;
    type SubscribeStream = MatchStream;
    type PullStream = PullStream;
    type PullHoldersStream = PullStream;

    #[expect(
        clippy::disallowed_methods,
        reason = "per-stream task, bounded by the stream's lifetime"
    )]
    async fn publish(
        &self,
        request: Request<Streaming<proto::Update>>,
    ) -> Result<Response<Self::PublishStream>, Status> {
        let mut inbound = request.into_inner();
        let engine = Arc::clone(&self.engine);
        let relay = self.relay.clone();
        let delays = (self.delay_stored, self.delay_removed);
        let (tx, rx) = mpsc::channel::<Result<proto::PublishAck, Status>>(1024);
        // Staleness injection is a constant LAG, not per-update service
        // time: updates flow through an unbounded FIFO stamped with an
        // apply deadline, and a drainer applies each at its deadline —
        // per-stream order (and so per-holder seq order) is preserved,
        // and throughput is unaffected. Zero-delay legs skip the queue.
        // BOUNDED: when the applier lags, this fills, the inbound
        // task blocks on send, tonic stops reading, and HTTP/2 flow
        // control pushes back on the publisher — instead of an
        // unbounded queue quietly absorbing an OOM (audit finding).
        // The decoded form rides along with the wire form: the applier
        // needs the first and the relay forwards the second, and decoding
        // on the inbound side is what lets a malformed update be rejected
        // to the publisher's face instead of normalised into a keyspace
        // it never meant to write.
        let (delayed_tx, mut delayed_rx) =
            mpsc::channel::<(tokio::time::Instant, proto::Update, UpdateMsg)>(65_536);
        let apply_engine = Arc::clone(&engine);
        let apply_relay = relay.clone();
        let apply_stats = Arc::clone(&self.stats);
        let ack_tx = tx.clone();
        let reject_tx = tx.clone();
        tokio::spawn(async move {
            // Drain whatever is already queued and apply it in one pass:
            // consecutive SEQUENCED (event-feed) updates go through
            // apply_batch — one keyspace write-lock per run instead of
            // per update — so the shared-lock routing queries get real
            // gaps between write bursts instead of ping-ponging against
            // a per-event writer (the multi-writer event-path fix).
            // Placement/control (seq 0) stay on per-update apply to keep
            // their read-lock fast paths.
            const MAX_BATCH: usize = 256;
            while let Some(first) = delayed_rx.recv().await {
                let mut batch = vec![first];
                while batch.len() < MAX_BATCH {
                    match delayed_rx.try_recv() {
                        Ok(item) => batch.push(item),
                        Err(_) => break,
                    }
                }
                // Honor the LATEST injected staleness deadline in the
                // batch (fault-drill legs); zero-delay batches fall
                // through immediately. Deadlines are not monotonic
                // within a batch — `delay` is picked per update from its
                // own event kinds, so a zero-delay Removed update can
                // follow a 200 ms Stored one — and taking the last
                // element's deadline would drain the delayed update
                // early, injecting less staleness than the leg
                // configured and quietly understating what the sweep
                // measures.
                if let Some(deadline) = batch.iter().map(|(deadline, ..)| *deadline).max() {
                    tokio::time::sleep_until(deadline).await;
                }
                let (protos, msgs): (Vec<proto::Update>, Vec<UpdateMsg>) = batch
                    .into_iter()
                    .map(|(_, proto, msg)| (proto, msg))
                    .unzip();
                let mut results: Vec<Applied> = Vec::with_capacity(msgs.len());
                let apply_started = std::time::Instant::now();
                let mut k = 0;
                while k < msgs.len() {
                    if msgs[k].seq != 0 {
                        let mut m = k + 1;
                        while m < msgs.len() && msgs[m].seq != 0 {
                            m += 1;
                        }
                        results.extend(apply_engine.apply_batch(&msgs[k..m]));
                        k = m;
                    } else {
                        results.push(apply_engine.apply(&msgs[k]));
                        k += 1;
                    }
                }
                apply_stats
                    .applies
                    .fetch_add(msgs.len() as u64, Ordering::Relaxed);
                // ONE observation for the whole batch (see
                // `ServiceStats::apply_batch_latency`): dividing the cost
                // across the batch erases the convoy the histogram exists
                // to show, and the divided form also cost 3 atomic
                // read-modify-writes per update on three cache lines
                // shared by every publisher stream — cross-core
                // contention added straight to the write path.
                apply_stats
                    .apply_batch_latency
                    .observe(apply_started.elapsed());

                let mut closed = false;
                for (idx, msg) in msgs.iter().enumerate() {
                    let applied = results[idx];
                    // Find the missed digest's tip ANYWHERE in the batch:
                    // an events.first()-only probe loses the tip on a
                    // mixed batch, acking a miss with no tip the
                    // publisher can resend — a silent under-match.
                    let digest_miss_tip = (applied.outcome == ApplyOutcome::DigestMiss)
                        .then(|| {
                            msg.events.iter().find_map(|event| match event {
                                WireEvent::StoredDigest { tip, .. } => Some(tip.0),
                                _ => None,
                            })
                        })
                        .flatten();
                    // Relay ONLY state-changing applies (echo dies in one
                    // hop; bounded O(K^2) fan-out).
                    if applied.changed {
                        for peer in &apply_relay {
                            if peer.try_send(protos[idx].clone()).is_err() {
                                apply_stats.relay_dropped.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    // The ack carries the holder's STORED epoch (from the
                    // apply), never an echo of the publisher's own — the
                    // EpochLedger's restart adoption is only sound against
                    // the index's real epoch.
                    let ack = proto::PublishAck {
                        holder: msg.holder.clone(),
                        epoch: applied.epoch,
                        applied_seq: applied.last_seq,
                        digest_miss_tip,
                    };
                    // Acks are advisory: drop when the publisher is not
                    // reading rather than wedging the applier.
                    match ack_tx.try_send(Ok(ack)) {
                        Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            closed = true;
                            break;
                        }
                    }
                }
                if closed {
                    break;
                }
            }
        });
        tokio::spawn(async move {
            while let Some(update) = inbound.next().await {
                let Ok(update) = update else { break };
                // Hash-scheme gate: a publisher on a scheme this build
                // cannot serve would poison the keyspace with hashes
                // that match nothing. A wrong scheme is a permanent
                // publisher misconfiguration, not a per-update
                // condition, so FAIL THE STREAM: silently dropping
                // updates would leave the publisher streaming into a
                // black hole with its acked watermark never moving —
                // indistinguishable, from its side, from a healthy feed.
                let scheme = update.keyspace.as_ref().map_or(0, |k| k.hash_scheme);
                if !crate::wire_hash::scheme_supported(scheme) {
                    tracing::warn!(scheme, holder = %update.holder, "unsupported hash scheme; failing the publish stream");
                    let _ = reject_tx
                        .send(Err(Status::failed_precondition(format!(
                            "unsupported hash scheme {scheme}; this build serves scheme {}",
                            crate::wire_hash::HASH_SCHEME_V1
                        ))))
                        .await;
                    break;
                }
                // Same reasoning as the scheme gate, one step further in:
                // a missing keyspace, an unknown symbol kind or an event
                // with no kind set used to be normalised away, which put
                // the publisher's state in a keyspace none of its queries
                // read while its acks kept arriving. Fail the stream so
                // the misconfiguration is visible at the publisher.
                let msg = match UpdateMsg::try_from(&update) {
                    Ok(msg) => msg,
                    Err(error) => {
                        tracing::warn!(%error, holder = %update.holder, "malformed update; failing the publish stream");
                        let _ = reject_tx
                            .send(Err(Status::invalid_argument(error.to_string())))
                            .await;
                        break;
                    }
                };
                let mut delay = Duration::ZERO;
                for event in &update.events {
                    match event.kind.as_ref() {
                        Some(proto::event::Kind::Stored(_)) => delay = delay.max(delays.0),
                        Some(proto::event::Kind::Removed(_)) => delay = delay.max(delays.1),
                        _ => {}
                    }
                }
                let deadline = tokio::time::Instant::now() + delay;
                if delayed_tx.send((deadline, update, msg)).await.is_err() {
                    break;
                }
            }
        });
        drop(tx);
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "per-stream task, bounded by the stream's lifetime"
    )]
    async fn subscribe(
        &self,
        request: Request<Streaming<proto::Query>>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let mut inbound = request.into_inner();
        let engine = Arc::clone(&self.engine);
        let stats = Arc::clone(&self.stats);
        let (tx, rx) = mpsc::channel::<Result<proto::Match, Status>>(1024);
        tokio::spawn(async move {
            while let Some(query) = inbound.next().await {
                let Ok(query) = query else { break };
                stats.queries.fetch_add(1, Ordering::Relaxed);
                let scheme = query.keyspace.as_ref().map_or(0, |k| k.hash_scheme);
                if !crate::wire_hash::scheme_supported(scheme) {
                    tracing::warn!(scheme, "unsupported hash scheme; empty answer");
                    let answer = proto::Match {
                        query_id: query.query_id,
                        scores: Vec::new(),
                    };
                    if tx.send(Ok(answer)).await.is_err() {
                        break;
                    }
                    continue;
                }
                let keyspace = query.keyspace.as_ref();
                let key = KeyspaceKey {
                    model: keyspace.map(|k| k.model.clone()).unwrap_or_default(),
                    symbol_kind: match keyspace.map(|k| k.symbol_kind) {
                        Some(k) if k == proto::SymbolKind::Bytes as i32 => SymbolKind::Bytes,
                        _ => SymbolKind::Tokens,
                    },
                    block_size: keyspace.map(|k| k.block_size).unwrap_or_default(),
                };
                // Cap query length: callers send request-sized
                // chains; anything longer is an abuse/DoS shape that
                // would run under the engine lock (audit finding).
                const MAX_QUERY_BLOCKS: usize = 16_384;
                if query.content_hashes.len() > MAX_QUERY_BLOCKS {
                    let _ = tx.try_send(Ok(proto::Match {
                        query_id: query.query_id,
                        scores: Vec::new(),
                    }));
                    continue;
                }
                let hashes: Vec<ContentHash> = query
                    .content_hashes
                    .iter()
                    .copied()
                    .map(ContentHash)
                    .collect();
                let query_started = std::time::Instant::now();
                let scores = engine.find_matches(&key, &hashes);
                stats.query_latency.observe(query_started.elapsed());
                let answer = proto::Match {
                    query_id: query.query_id,
                    scores: scores
                        .into_iter()
                        .map(|s| proto::HolderScore {
                            holder: s.holder,
                            matched_blocks: s.matched_blocks,
                            total_blocks: s.total_blocks,
                            event_fed: s.event_fed,
                            intervals: s
                                .intervals
                                .into_iter()
                                .map(|(start, end)| proto::Interval { start, end })
                                .collect(),
                            lane_meta: s.lane_meta,
                        })
                        .collect(),
                };
                // Answers are deadline-bound on the caller: when the
                // gateway stops reading, drop answers rather than
                // blocking this task off the inbound stream
                // (head-of-line deadlock shape, audit finding).
                match tx.try_send(Ok(answer)) {
                    Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                    Err(mpsc::error::TrySendError::Closed(_)) => break,
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    /// The peer-visible half of anti-entropy: every holder's (epoch,
    /// seq, block count, set digest) in one unary response, so a replica
    /// can decide what it needs without streaming any state. A full-index
    /// fold (see [`Engine::holder_digests_yielding`]) taking one read
    /// lock per holder, never a long hold, and yielding between holders
    /// so a large index does not stall the runtime worker.
    async fn digests(
        &self,
        _request: Request<proto::PullRequest>,
    ) -> Result<Response<proto::DigestsResponse>, Status> {
        let holders = self
            .engine
            .holder_digests_yielding()
            .await
            .into_iter()
            .map(|d| proto::HolderDigest {
                keyspace: Some(keyspace_to_proto(&d.keyspace)),
                holder: d.holder,
                epoch: d.epoch,
                last_seq: d.last_seq,
                event_fed: d.event_fed,
                blocks: d.blocks,
                digest_xor: d.digest_xor,
                digest_sum: d.digest_sum,
            })
            .collect();
        Ok(Response::new(proto::DigestsResponse { holders }))
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "per-stream producer task, bounded by the stream's lifetime"
    )]
    async fn pull_holders(
        &self,
        request: Request<proto::PullHoldersRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        let engine = Arc::clone(&self.engine);
        let refs = request.into_inner().holders;
        let (tx, rx) = mpsc::channel::<Result<proto::Update, Status>>(32);
        tokio::spawn(async move {
            for r in refs {
                let key = keyspace_from_proto(r.keyspace.as_ref());
                for update in engine.snapshot_holder(&key, &r.holder) {
                    if tx.send(Ok(proto::Update::from(&update))).await.is_err() {
                        return;
                    }
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    /// Bootstrap stream, produced LAZILY one holder at a time as the
    /// puller drains: the serving replica never materializes a second
    /// copy of its whole index (at production sizing that transient was
    /// a multi-GB spike on the HEALTHY replica whenever a sibling
    /// restarted — the wrong direction for a fault-tolerance story).
    #[expect(
        clippy::disallowed_methods,
        reason = "per-stream producer task, bounded by the stream's lifetime"
    )]
    async fn pull(
        &self,
        _request: Request<proto::PullRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        let engine = Arc::clone(&self.engine);
        let (tx, rx) = mpsc::channel::<Result<proto::Update, Status>>(32);
        tokio::spawn(async move {
            for key in engine.snapshot_keys() {
                for holder in engine.snapshot_holders(&key) {
                    for update in engine.snapshot_holder(&key, &holder) {
                        if tx.send(Ok(proto::Update::from(&update))).await.is_err() {
                            return; // puller went away
                        }
                    }
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }
}

fn keyspace_to_proto(key: &KeyspaceKey) -> proto::Keyspace {
    proto::Keyspace {
        model: key.model.clone(),
        symbol_kind: match key.symbol_kind {
            SymbolKind::Tokens => proto::SymbolKind::Tokens as i32,
            SymbolKind::Bytes => proto::SymbolKind::Bytes as i32,
        },
        block_size: key.block_size,
        hash_scheme: crate::wire_hash::HASH_SCHEME_V1,
    }
}

fn keyspace_from_proto(ks: Option<&proto::Keyspace>) -> KeyspaceKey {
    KeyspaceKey {
        model: ks.map(|k| k.model.clone()).unwrap_or_default(),
        symbol_kind: match ks.map(|k| k.symbol_kind) {
            Some(k) if k == proto::SymbolKind::Bytes as i32 => SymbolKind::Bytes,
            _ => SymbolKind::Tokens,
        },
        block_size: ks.map(|k| k.block_size).unwrap_or_default(),
    }
}

/// Holders our own sweeper retired recently, so a round does not read a
/// peer's not-yet-swept copy as "we are behind" and resurrect them.
///
/// Retirement ([`Engine::sweep_idle`]) removes the holder entry outright
/// on a local, clock-driven decision, so two replicas never make it at
/// the same instant — they differ by up to a sweep interval plus skew.
/// Absence carries no watermark, so without this the replica that swept
/// first pulls the holder straight back (a zero-block holder still
/// produces a snapshot chunk, which recreates the entry with a fresh
/// idle clock, and if the pull beats the peer's own TTL clear the stale
/// blocks come back too and answer queries again). Retirement would be
/// undone instead of propagated: a dead worker's entry survives
/// indefinitely and the placement-only keyspace it pins is never
/// collected.
///
/// Nothing but retirement removes a holder entry, so a name that was in
/// our digests last round and is gone this round was retired here — no
/// extra bookkeeping in the engine is needed to notice it.
#[derive(Debug)]
pub struct RetiredHolders {
    /// How long a retirement suppresses a pull. Must outrun the peer's
    /// own sweep plus a round of its own; past it, whatever the peer
    /// still carries is a live holder rather than our retirement, and a
    /// worker that really came back is pulled normally.
    window: Duration,
    seen: HashSet<(KeyspaceKey, String)>,
    tombstones: HashMap<(KeyspaceKey, String), std::time::Instant>,
}

impl RetiredHolders {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            seen: HashSet::new(),
            tombstones: HashMap::new(),
        }
    }

    /// Fold one round's local digests in and return the holders that are
    /// still tombstoned. The first call only records what we hold, so a
    /// cold replica pulls everything the peer has, as it must.
    pub fn observe(&mut self, local: &[HolderDigest]) -> HashSet<(KeyspaceKey, String)> {
        let now = std::time::Instant::now();
        let window = self.window;
        let current: HashSet<(KeyspaceKey, String)> = local
            .iter()
            .map(|d| (d.keyspace.clone(), d.holder.clone()))
            .collect();
        for gone in self.seen.difference(&current) {
            self.tombstones.insert(gone.clone(), now);
        }
        self.tombstones
            .retain(|_, stamped| now.duration_since(*stamped) < window);
        self.seen = current;
        self.tombstones.keys().cloned().collect()
    }
}

/// The holders to pull from a peer, given both sides' digests, ignoring
/// recent local retirements (see [`plan_anti_entropy_excluding`]).
pub fn plan_anti_entropy(
    local: &[HolderDigest],
    remote: &[HolderDigest],
) -> Vec<(KeyspaceKey, String)> {
    plan_anti_entropy_excluding(local, remote, &HashSet::new())
}

/// The holders to pull from a peer, given both sides' digests and the
/// holders we retired recently (`retired`, from [`RetiredHolders`]).
///
/// A peer is *provably ahead* on a holder when it carries a higher
/// (epoch, seq) watermark, or the same watermark over a block set that
/// wins the digest tie-break: sequenced (event-fed) state is
/// deterministic per watermark, so a different set at the same seq means
/// one side applied something the other never saw (a relay lost to a
/// partition, a wedged peer, a bootstrap that landed between two
/// batches). At an equal watermark neither side is authoritative, so the
/// digest itself decides, under a total order both replicas evaluate
/// identically — exactly one of them pulls. A bare "differs, so pull" is
/// symmetric in the direction that never converges: two rounds racing on
/// independent timers swap the two states instead of agreeing, and each
/// swap is a clear plus a full snapshot replace under the write lock.
///
/// Placement-fed holders (seq 0) are unsequenced and inferred — replicas
/// copy them, never agree on them by design (TTL and re-placement bound
/// that divergence), so they are pulled only when missing entirely.
/// Missing, however, is not the same as behind: a holder absent because
/// we retired it seconds ago is skipped, or anti-entropy would undo
/// retirements instead of propagating them. Ties on watermark AND digest
/// are converged and skipped; a peer BEHIND us is its problem to notice
/// on its own round.
pub fn plan_anti_entropy_excluding(
    local: &[HolderDigest],
    remote: &[HolderDigest],
    retired: &HashSet<(KeyspaceKey, String)>,
) -> Vec<(KeyspaceKey, String)> {
    let mut mine: HashMap<(&KeyspaceKey, &str), &HolderDigest> = HashMap::new();
    for d in local {
        mine.insert((&d.keyspace, d.holder.as_str()), d);
    }
    let mut pull = Vec::new();
    for theirs in remote {
        match mine.get(&(&theirs.keyspace, theirs.holder.as_str())) {
            None => {
                let id = (theirs.keyspace.clone(), theirs.holder.clone());
                if !retired.contains(&id) {
                    pull.push(id);
                }
            }
            Some(ours) => {
                let ahead = (theirs.epoch, theirs.last_seq) > (ours.epoch, ours.last_seq);
                let same_mark = (theirs.epoch, theirs.last_seq) == (ours.epoch, ours.last_seq);
                // Total order over the digest: which set wins does not
                // matter, agreeing on a winner does.
                let theirs_wins = (theirs.blocks, theirs.digest_xor, theirs.digest_sum)
                    > (ours.blocks, ours.digest_xor, ours.digest_sum);
                let sequenced = theirs.event_fed || ours.event_fed;
                if ahead || (same_mark && theirs_wins && sequenced) {
                    pull.push((theirs.keyspace.clone(), theirs.holder.clone()));
                }
            }
        }
    }
    pull
}

/// One anti-entropy round against `peer`: fetch its digests, plan, and
/// replace every planned holder wholesale from the peer's snapshot.
/// Returns the number of holders replaced. `retired` carries this
/// replica's recent retirements across rounds (see [`RetiredHolders`]).
pub async fn anti_entropy_round(
    engine: &Engine,
    peer: &str,
    stats: &ServiceStats,
    retired: &mut RetiredHolders,
) -> Result<usize, tonic::Status> {
    let outcome = anti_entropy_round_inner(engine, peer, stats, retired).await;
    if outcome.is_err() {
        // Counted separately because `anti_entropy_rounds` only moves
        // after the digests call succeeds: a round that never got to
        // compare anything would otherwise leave no trace at all, and
        // this is the only repair path once the missed deltas are gone.
        stats.anti_entropy_failures.fetch_add(1, Ordering::Relaxed);
    }
    outcome
}

async fn anti_entropy_round_inner(
    engine: &Engine,
    peer: &str,
    stats: &ServiceStats,
    retired: &mut RetiredHolders,
) -> Result<usize, tonic::Status> {
    let mut client = RadixIndexClient::connect(peer.to_string())
        .await
        .map_err(|e| tonic::Status::unavailable(e.to_string()))?
        .max_decoding_message_size(64 * 1024 * 1024)
        .max_encoding_message_size(64 * 1024 * 1024);
    let remote: Vec<HolderDigest> = client
        .digests(Request::new(proto::PullRequest {}))
        .await?
        .into_inner()
        .holders
        .into_iter()
        .map(|d| HolderDigest {
            keyspace: keyspace_from_proto(d.keyspace.as_ref()),
            holder: d.holder,
            epoch: d.epoch,
            last_seq: d.last_seq,
            event_fed: d.event_fed,
            blocks: d.blocks,
            digest_xor: d.digest_xor,
            digest_sum: d.digest_sum,
        })
        .collect();
    let local = engine.holder_digests_yielding().await;
    let tombstoned = retired.observe(&local);
    let plan = plan_anti_entropy_excluding(&local, &remote, &tombstoned);
    stats.anti_entropy_rounds.fetch_add(1, Ordering::Relaxed);
    if plan.is_empty() {
        return Ok(0);
    }
    let refs = plan
        .iter()
        .map(|(key, holder)| proto::HolderRef {
            keyspace: Some(keyspace_to_proto(key)),
            holder: holder.clone(),
        })
        .collect();
    let mut stream = client
        .pull_holders(Request::new(proto::PullHoldersRequest { holders: refs }))
        .await?
        .into_inner();
    // Buffer a holder's chunks, then clear and refill it back to back.
    // Clearing on the first chunk and applying the rest as they arrive
    // off the wire is safe for `Pull` (bootstrap runs before the replica
    // serves, behind a 503 /readyz) but not here: anti-entropy repairs a
    // LIVE replica, a holder above the snapshot chunk size arrives as
    // several messages, and every query landing in that window saw the
    // holder rebuilt only as far as the network had got. A peer that
    // died mid-stream left it truncated for good — the snapshot has
    // already stamped the peer's watermark on it, so a placement-fed
    // holder is not even re-planned next round. The cost is one holder's
    // chunks in memory, which the peer materialized anyway to send them.
    let mut replaced = 0usize;
    let mut chunks: Vec<UpdateMsg> = Vec::new();
    while let Some(update) = stream.next().await {
        // A peer that sends a chunk we cannot read is a version or a
        // corruption problem, not a per-holder one. Abort the round and
        // let the next one re-plan rather than writing half a repair.
        let msg = UpdateMsg::try_from(&update?)
            .map_err(|error| tonic::Status::invalid_argument(error.to_string()))?;
        // `pull_holders` emits one holder's chunks consecutively, so the
        // first chunk of the next holder completes the previous one.
        let boundary = chunks
            .first()
            .is_some_and(|head| head.keyspace != msg.keyspace || head.holder != msg.holder);
        if boundary {
            replace_holder(engine, &chunks);
            replaced += 1;
            chunks.clear();
        }
        chunks.push(msg);
    }
    if !chunks.is_empty() {
        replace_holder(engine, &chunks);
        replaced += 1;
    }
    stats
        .anti_entropy_holders_pulled
        .fetch_add(replaced as u64, Ordering::Relaxed);
    if replaced > 0 {
        tracing::info!(peer, replaced, "anti-entropy replaced diverged holders");
    }
    Ok(replaced)
}

/// Swap one holder's state for the peer's snapshot: clear (so blocks the
/// peer removed while we were apart do not survive), then apply every
/// buffered chunk. Empty input is a peer that retired the holder between
/// its digests and our pull — leave ours alone.
fn replace_holder(engine: &Engine, chunks: &[UpdateMsg]) {
    let Some(head) = chunks.first() else {
        return;
    };
    engine.clear_holder(&head.keyspace, &head.holder);
    for chunk in chunks {
        engine.apply_snapshot(chunk);
    }
}

/// Consecutive failed rounds against one peer before the log line moves
/// from `debug!` to `warn!`, and how often it repeats after that. A
/// partition heals on its own and must not be noisy; a wrong peer URL or
/// a peer too old to serve `Digests` never recovers, and a backstop that
/// has never run cannot be left to one log line from a week ago.
const FAILURES_BEFORE_WARN: u32 = 3;
const FAILURES_BETWEEN_WARNS: u32 = 40;

/// How many anti-entropy periods a local retirement suppresses a pull
/// for. The peer needs its own sweep plus a round of its own to reach
/// the same decision, and until it does its copy is not evidence that we
/// are behind.
const RETIRE_TOMBSTONE_PERIODS: u32 = 3;

/// The window above, floored so the peer's sweep actually fits inside
/// it. The two intervals are configured independently, so a short
/// anti-entropy period against the default sweep (1 s against 5 s) gave
/// a 3 s window against a ~5 s lag: whoever swept first pulled the dead
/// holder back with a fresh idle clock, and it cycled between the
/// replicas instead of retiring.
fn tombstone_window(interval: Duration, sweep_interval: Duration) -> Duration {
    (interval * RETIRE_TOMBSTONE_PERIODS).max(sweep_interval + interval)
}

/// Service-lifetime anti-entropy: every `interval`, one round per peer.
/// `Duration::ZERO` disables it.
#[expect(
    clippy::disallowed_methods,
    reason = "service-lifetime task; the index process is its own supervisor"
)]
pub fn spawn_anti_entropy(
    engine: Arc<Engine>,
    peers: Vec<String>,
    interval: Duration,
    sweep_interval: Duration,
    stats: Arc<ServiceStats>,
) {
    if interval.is_zero() || peers.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut retired = RetiredHolders::new(tombstone_window(interval, sweep_interval));
        let mut failures = vec![0u32; peers.len()];
        loop {
            tick.tick().await;
            for (idx, peer) in peers.iter().enumerate() {
                match anti_entropy_round(&engine, peer, &stats, &mut retired).await {
                    Ok(_) => failures[idx] = 0,
                    Err(error) => {
                        failures[idx] = failures[idx].saturating_add(1);
                        let consecutive = failures[idx];
                        if consecutive == FAILURES_BEFORE_WARN
                            || consecutive.is_multiple_of(FAILURES_BETWEEN_WARNS)
                        {
                            tracing::warn!(
                                peer,
                                %error,
                                consecutive,
                                "anti-entropy has not completed a round against this peer"
                            );
                        } else {
                            tracing::debug!(peer, %error, "anti-entropy round skipped");
                        }
                    }
                }
            }
        }
    });
}

/// Bootstrap: pull the full state from `peer` and apply it before
/// serving. Returns Ok(applied_count); a connect failure is Ok(0) so a
/// lone first replica can boot cold.
///
/// ALL OR NOTHING: on failure the engine is left empty, so the caller's
/// choice is between a full snapshot and a cold start, never a truncated
/// one. Callers go ready either way (the pull is an optimisation, not a
/// correctness requirement), and a cold replica is repaired by
/// anti-entropy's "we have never held this" arm — a truncated one is
/// already stamped with the peer's watermark, so nothing re-plans it and
/// it under-matches until TTL or the next placement.
///
/// The engine must therefore be empty on entry, as it is at startup.
pub async fn bootstrap_from(engine: &Engine, peer: &str) -> Result<usize, tonic::Status> {
    let Ok(client) = RadixIndexClient::connect(peer.to_string()).await else {
        // Not an error (a lone first replica boots cold by design), but
        // never silent: a wrong or not-yet-up peer is otherwise
        // indistinguishable from a healthy empty pull.
        tracing::warn!(peer, "bootstrap peer unreachable; starting cold");
        return Ok(0);
    };
    let mut client = client
        .max_decoding_message_size(64 * 1024 * 1024)
        .max_encoding_message_size(64 * 1024 * 1024);
    let outcome = bootstrap_stream(engine, &mut client).await;
    if outcome.is_err() {
        engine.clear_all();
    }
    outcome
}

async fn bootstrap_stream(
    engine: &Engine,
    client: &mut RadixIndexClient<tonic::transport::Channel>,
) -> Result<usize, tonic::Status> {
    let mut stream = client
        .pull(Request::new(proto::PullRequest {}))
        .await?
        .into_inner();
    let mut applied = 0usize;
    while let Some(update) = stream.next().await {
        let update = update?;
        // Snapshot reconstruction, NOT a live feed: `apply_snapshot`
        // bypasses seq-dedup so a holder spanning several chunks (all
        // carrying the same last_seq) reconstructs in full instead of
        // being truncated to the first chunk.
        let msg = UpdateMsg::try_from(&update)
            .map_err(|error| tonic::Status::invalid_argument(error.to_string()))?;
        engine.apply_snapshot(&msg);
        applied += 1;
    }
    Ok(applied)
}

/// Serve the index on `addr` until the process exits.
pub async fn serve(
    engine: Arc<Engine>,
    addr: std::net::SocketAddr,
    peers: Vec<String>,
    sweep_interval: Duration,
) -> Result<(), tonic::transport::Error> {
    serve_with_delays(
        engine,
        addr,
        peers,
        sweep_interval,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await
}

/// [`serve`] with peer anti-entropy every `anti_entropy_interval`
/// (see [`spawn_anti_entropy`]); `Duration::ZERO` disables it.
pub async fn serve_with_anti_entropy(
    engine: Arc<Engine>,
    addr: std::net::SocketAddr,
    peers: Vec<String>,
    sweep_interval: Duration,
    anti_entropy_interval: Duration,
) -> Result<(), tonic::transport::Error> {
    serve_until(
        engine,
        addr,
        peers,
        sweep_interval,
        anti_entropy_interval,
        Duration::ZERO,
        Duration::ZERO,
        Arc::new(ServiceStats::default()),
        std::future::pending::<()>(),
    )
    .await
}

/// [`serve`] with staleness injection (the experiment's sweep knob).
pub async fn serve_with_delays(
    engine: Arc<Engine>,
    addr: std::net::SocketAddr,
    peers: Vec<String>,
    sweep_interval: Duration,
    delay_stored: Duration,
    delay_removed: Duration,
) -> Result<(), tonic::transport::Error> {
    serve_until(
        engine,
        addr,
        peers,
        sweep_interval,
        Duration::ZERO, // anti-entropy off: these entry points exercise relay alone
        delay_stored,
        delay_removed,
        Arc::new(ServiceStats::default()),
        std::future::pending::<()>(),
    )
    .await
}

/// The full server: gRPC on `addr`, idle sweeper on `sweep_interval`,
/// graceful stop when `shutdown` resolves (in-flight streams get to
/// finish; publishers and gateways reconnect to a sibling replica).
#[expect(
    clippy::disallowed_methods,
    reason = "service-lifetime sweeper; the index process is its own supervisor"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "top-level composition point mirroring the binary's flags"
)]
pub async fn serve_until(
    engine: Arc<Engine>,
    addr: std::net::SocketAddr,
    peers: Vec<String>,
    sweep_interval: Duration,
    anti_entropy_interval: Duration,
    delay_stored: Duration,
    delay_removed: Duration,
    stats: Arc<ServiceStats>,
    shutdown: impl std::future::Future<Output = ()> + Send,
) -> Result<(), tonic::transport::Error> {
    // Rejected here, not only in the CLI: the sweeper below is detached,
    // so a zero interval panics tokio's timer inside a task nobody joins
    // and the server keeps serving with TTL clearing, retirement and
    // keyspace GC all silently off.
    assert!(
        !sweep_interval.is_zero(),
        "sweep_interval must be non-zero (a zero interval panics tokio's timer inside the detached sweeper and silently disables TTL/retire)"
    );
    spawn_anti_entropy(
        Arc::clone(&engine),
        peers.clone(),
        anti_entropy_interval,
        sweep_interval,
        Arc::clone(&stats),
    );
    let sweeper = Arc::clone(&engine);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(sweep_interval);
        loop {
            tick.tick().await;
            sweeper.sweep_idle();
        }
    });
    Server::builder()
        // Explicit, generous decode cap (chunked snapshots keep real
        // messages far below it; the default 4MiB was a silent
        // bootstrap killer at scale — audit finding).
        .add_service(
            RadixIndexServer::new(IndexService::with_stats(
                engine,
                stats,
                peers,
                delay_stored,
                delay_removed,
            ))
            .max_decoding_message_size(64 * 1024 * 1024)
            .max_encoding_message_size(64 * 1024 * 1024),
        )
        .serve_with_shutdown(addr, shutdown)
        .await
}

/// How long an accepted admin connection has to send its request line.
const ADMIN_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Admin plane on its own port: `/metrics` (Prometheus text),
/// `/healthz` (liveness: the process answers), `/readyz` (readiness:
/// bootstrap finished — 503 until then). Deliberately handwritten over
/// a plain TCP listener: three fixed GET routes don't justify an HTTP
/// framework dependency in this crate.
#[expect(
    clippy::disallowed_methods,
    reason = "service-lifetime admin listener; the index process is its own supervisor"
)]
pub async fn serve_admin(
    engine: Arc<Engine>,
    stats: Arc<ServiceStats>,
    addr: std::net::SocketAddr,
) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind(addr).await?;
    loop {
        let (mut socket, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => {
                // Back off instead of busy-spinning on accept errors
                // (fd exhaustion would otherwise pin a core).
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let engine = Arc::clone(&engine);
        let stats = Arc::clone(&stats);
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            // Bound the first read: a peer that connects and then sends
            // nothing holds this task and its fd for the life of the
            // process, and nothing else caps how many admin connections
            // accumulate — a scrape target is trivially easy to point a
            // half-open connection at.
            let Ok(Ok(n)) = tokio::time::timeout(ADMIN_READ_TIMEOUT, socket.read(&mut buf)).await
            else {
                return;
            };
            let request = String::from_utf8_lossy(&buf[..n]);
            let path = request.split_whitespace().nth(1).unwrap_or("/");
            let (status, body) = match path {
                "/metrics" => (200, render_metrics(&engine, &stats)),
                "/healthz" => (200, "ok\n".to_string()),
                "/readyz" => {
                    if stats.ready.load(Ordering::Relaxed) {
                        (200, "ready\n".to_string())
                    } else {
                        (503, "bootstrapping\n".to_string())
                    }
                }
                _ => (404, "not found\n".to_string()),
            };
            let reason = match status {
                200 => "OK",
                503 => "Service Unavailable",
                _ => "Not Found",
            };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: text/plain; version=0.0.4\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

/// Prometheus text for the admin plane. The apply histogram is per
/// BATCH so the write-lock tail survives; the per-update average is
/// `radix_index_apply_batch_duration_seconds_sum / radix_index_applies_total`.
fn render_metrics(engine: &Engine, stats: &ServiceStats) -> String {
    let gauges = engine.stats();
    format!(
        concat!(
            "# TYPE radix_index_keyspaces gauge\n",
            "radix_index_keyspaces {}\n",
            "# TYPE radix_index_holders gauge\n",
            "radix_index_holders {}\n",
            "# TYPE radix_index_event_fed_holders gauge\n",
            "radix_index_event_fed_holders {}\n",
            "# TYPE radix_index_dropped_holders gauge\n",
            "radix_index_dropped_holders {}\n",
            "# TYPE radix_index_blocks gauge\n",
            "radix_index_blocks {}\n",
            "# TYPE radix_index_applies_total counter\n",
            "radix_index_applies_total {}\n",
            "# TYPE radix_index_queries_total counter\n",
            "radix_index_queries_total {}\n",
            "# TYPE radix_index_relay_dropped_total counter\n",
            "radix_index_relay_dropped_total {}\n",
            "# TYPE radix_index_anti_entropy_rounds_total counter\n",
            "radix_index_anti_entropy_rounds_total {}\n",
            "# TYPE radix_index_anti_entropy_holders_pulled_total counter\n",
            "radix_index_anti_entropy_holders_pulled_total {}\n",
            "# TYPE radix_index_anti_entropy_failures_total counter\n",
            "radix_index_anti_entropy_failures_total {}\n",
        ),
        gauges.keyspaces,
        gauges.holders,
        gauges.event_fed_holders,
        gauges.dropped_holders,
        gauges.blocks,
        stats.applies.load(Ordering::Relaxed),
        stats.queries.load(Ordering::Relaxed),
        stats.relay_dropped.load(Ordering::Relaxed),
        stats.anti_entropy_rounds.load(Ordering::Relaxed),
        stats.anti_entropy_holders_pulled.load(Ordering::Relaxed),
        stats.anti_entropy_failures.load(Ordering::Relaxed),
    ) + &stats
        .apply_batch_latency
        .render("radix_index_apply_batch_duration_seconds")
        + &stats
            .query_latency
            .render("radix_index_query_duration_seconds")
        + &engine
            .cut_latency()
            .render("radix_index_capacity_cut_duration_seconds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineConfig, SequenceHash, WireBlock};

    fn keyspace() -> KeyspaceKey {
        KeyspaceKey {
            model: "m".into(),
            symbol_kind: SymbolKind::Tokens,
            block_size: 4,
        }
    }

    fn digest(
        holder: &str,
        epoch: u64,
        seq: u64,
        event_fed: bool,
        blocks: u64,
        x: u64,
    ) -> HolderDigest {
        HolderDigest {
            keyspace: keyspace(),
            holder: holder.into(),
            epoch,
            last_seq: seq,
            event_fed,
            blocks,
            digest_xor: x,
            digest_sum: x.wrapping_mul(3),
        }
    }

    /// One chunk of a holder's snapshot, shaped like `snapshot_holder`
    /// emits them: every chunk carries the holder's true `seq`, and all
    /// but the first are parent-linked to the previous chunk's tail.
    fn chunk(holder: &str, parent: Option<u64>, blocks: &[(u64, u64)], seq: u64) -> UpdateMsg {
        UpdateMsg {
            keyspace: keyspace(),
            holder: holder.into(),
            epoch: 1,
            seq,
            events: vec![WireEvent::Stored {
                parent: parent.map(SequenceHash),
                blocks: blocks
                    .iter()
                    .map(|&(seq_hash, content)| WireBlock {
                        seq_hash: SequenceHash(seq_hash),
                        content_hash: ContentHash(content),
                    })
                    .collect(),
            }],
            added: None,
            dropped: false,
        }
    }

    #[test]
    fn anti_entropy_pulls_only_where_the_peer_is_provably_ahead() {
        let local = vec![
            digest("a", 1, 10, true, 5, 0xA), // converged
            digest("b", 1, 10, true, 5, 0xB), // peer ahead on seq
            digest("c", 1, 10, true, 5, 0xC), // same seq, peer's set wins the tie
            digest("d", 1, 0, false, 5, 0xD), // placement-fed, differs: by design
            digest("e", 2, 3, true, 5, 0xE),  // we are ahead: their round's job
            digest("g", 1, 10, true, 9, 0x9), // same seq, OUR set wins the tie
        ];
        let remote = vec![
            digest("a", 1, 10, true, 5, 0xA),
            digest("b", 1, 12, true, 6, 0xB1),
            digest("c", 1, 10, true, 6, 0xC1),
            digest("d", 1, 0, false, 7, 0xD1),
            digest("e", 1, 9, true, 5, 0xE1),
            digest("f", 1, 0, false, 2, 0xF), // missing locally: pulled
            digest("g", 1, 10, true, 8, 0x8),
        ];
        let plan: Vec<String> = plan_anti_entropy(&local, &remote)
            .into_iter()
            .map(|(_, h)| h)
            .collect();
        assert_eq!(plan, vec!["b", "c", "f"]);
    }

    #[test]
    fn a_tied_watermark_is_broken_the_same_way_on_both_replicas() {
        // Each side runs the rule against its own view. Without a total
        // order over the digest both classify the other as ahead, so two
        // rounds on independent timers swap the states instead of
        // agreeing — and keep swapping, one full holder replace apiece.
        let a = vec![digest("w", 1, 10, true, 5, 0xA)];
        let b = vec![digest("w", 1, 10, true, 5, 0xB)];
        let a_pulls = plan_anti_entropy(&a, &b).len();
        let b_pulls = plan_anti_entropy(&b, &a).len();
        assert_eq!(a_pulls + b_pulls, 1, "exactly one side may pull");
    }

    #[test]
    fn a_holder_we_just_retired_is_not_pulled_back_from_the_peer() {
        let mut retired = RetiredHolders::new(Duration::from_secs(60));
        let before = vec![digest("w7", 1, 0, false, 4, 0x7)];
        assert!(retired.observe(&before).is_empty(), "nothing retired yet");
        // Our sweeper retired w7; the peer has not swept yet, so it still
        // carries the holder. Pulling it back would restart w7's TTL
        // clock here and pin the placement-only keyspace forever.
        let tombstoned = retired.observe(&[]);
        let plan = plan_anti_entropy_excluding(&[], &before, &tombstoned);
        assert!(plan.is_empty(), "retirement must propagate, not be undone");
    }

    #[test]
    fn a_holder_we_have_never_held_is_still_pulled() {
        // A cold replica, and every genuinely new worker, must still
        // arrive through the `None` arm.
        let mut retired = RetiredHolders::new(Duration::from_secs(60));
        let remote = vec![digest("w8", 1, 0, false, 4, 0x8)];
        let tombstoned = retired.observe(&[]);
        assert_eq!(
            plan_anti_entropy_excluding(&[], &remote, &tombstoned).len(),
            1
        );
    }

    #[test]
    fn the_tombstone_window_outlasts_a_sweep_slower_than_the_round() {
        // The two intervals are configured independently. Three rounds
        // of 1 s is 3 s, but the peer needs its own 5 s sweep plus a
        // round before it can agree — inside that gap the replica that
        // swept first pulls the dead holder back with a fresh idle
        // clock and it cycles between the two.
        assert_eq!(
            tombstone_window(Duration::from_secs(1), Duration::from_secs(5)),
            Duration::from_secs(6)
        );
        // Where the round already dominates, nothing changes.
        assert_eq!(
            tombstone_window(Duration::from_secs(15), Duration::from_secs(5)),
            Duration::from_secs(45)
        );
    }

    #[test]
    fn a_tombstone_expires_so_a_worker_that_came_back_is_pulled_again() {
        // Past the window the peer has had its own sweep and a round of
        // its own; whatever it still carries is live state, not our
        // retirement. A zero window expires everything immediately.
        let mut retired = RetiredHolders::new(Duration::ZERO);
        let before = vec![digest("w7", 1, 0, false, 4, 0x7)];
        retired.observe(&before);
        let tombstoned = retired.observe(&[]);
        assert!(tombstoned.is_empty());
        assert_eq!(
            plan_anti_entropy_excluding(&[], &before, &tombstoned).len(),
            1
        );
    }

    #[test]
    fn a_holder_swap_clears_once_and_keeps_every_chunk() {
        let engine = Engine::new(EngineConfig::default());
        // State from before the repair, including blocks the peer no
        // longer has.
        engine.apply_snapshot(&chunk("w1", None, &[(11, 101), (12, 102)], 7));
        // The peer's answer for one holder, chunked as any holder above
        // SNAPSHOT_CHUNK blocks arrives.
        let chunks = vec![
            chunk("w1", None, &[(21, 201), (22, 202)], 9),
            chunk("w1", Some(22), &[(23, 203)], 9),
        ];
        replace_holder(&engine, &chunks);
        let digests = engine.holder_digests();
        assert_eq!(digests.len(), 1);
        // 3: the clear ran ONCE ahead of the chunks, so neither the
        // peer's removals survived (5) nor did the second chunk wipe the
        // first (1).
        assert_eq!(digests[0].blocks, 3, "{digests:?}");
        assert_eq!(digests[0].last_seq, 9);
    }

    #[test]
    fn a_swap_of_an_unknown_holder_is_a_no_op() {
        // A peer that retired the holder between its digests and our
        // pull sends no chunks; ours must be left alone.
        let engine = Engine::new(EngineConfig::default());
        engine.apply_snapshot(&chunk("w1", None, &[(11, 101)], 7));
        replace_holder(&engine, &[]);
        assert_eq!(engine.holder_digests()[0].blocks, 1);
    }

    #[test]
    fn latency_histogram_buckets_are_cumulative_and_span_the_engine_range() {
        let h = LatencyHistogram::default();
        h.observe(Duration::from_micros(7)); // <= 10 µs
        h.observe(Duration::from_micros(1_500)); // <= 2 ms
        h.observe(Duration::from_micros(3_000)); // > 2 ms, <= 5 ms
        h.observe(Duration::from_secs(1)); // +Inf
        let text = h.render("x");
        assert!(text.contains("x_bucket{le=\"0.00001\"} 1\n"));
        assert!(text.contains("x_bucket{le=\"0.002\"} 2\n"), "{text}");
        assert!(text.contains("x_bucket{le=\"0.005\"} 3\n"));
        assert!(text.contains("x_bucket{le=\"+Inf\"} 4\n"));
        assert!(text.contains("x_count 4\n"));
    }

    #[test]
    fn latency_histogram_keeps_sub_microsecond_observations() {
        let h = LatencyHistogram::default();
        for _ in 0..1_000 {
            h.observe(Duration::from_nanos(400));
        }
        let text = h.render("x");
        // Truncating to whole µs would have filed every one of these as
        // 0: `_sum` flat, `_count` climbing, and the derived average
        // rendering as exactly zero.
        assert!(text.contains("x_sum 0.0004\n"), "{text}");
        assert!(text.contains("x_count 1000\n"));
    }

    #[test]
    fn latency_histogram_bucketing_does_not_round_down_into_the_floor() {
        let h = LatencyHistogram::default();
        h.observe(Duration::from_nanos(10_001)); // just over the 10 µs edge
        let text = h.render("x");
        assert!(text.contains("x_bucket{le=\"0.00001\"} 0\n"), "{text}");
        assert!(text.contains("x_bucket{le=\"0.000025\"} 1\n"), "{text}");
    }

    #[test]
    fn relay_queue_falls_back_only_when_unset() {
        assert_eq!(parse_relay_queue(None), RELAY_QUEUE);
        assert_eq!(parse_relay_queue(Some("128")), 128);
    }

    #[test]
    #[should_panic(expected = "RADIX_RELAY_QUEUE")]
    fn relay_queue_rejects_a_malformed_value() {
        // Falling back to the default here would leave an overflow drill
        // measuring the 65k default and reporting a false negative.
        let _ = parse_relay_queue(Some("64k"));
    }

    #[test]
    #[should_panic(expected = "RADIX_RELAY_QUEUE")]
    fn relay_queue_rejects_zero() {
        // `mpsc::channel(0)` panics inside service construction.
        let _ = parse_relay_queue(Some("0"));
    }
}
