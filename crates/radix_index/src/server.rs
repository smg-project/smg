//! The gRPC surface: Publish (apply + best-effort peer relay),
//! Subscribe (query stream), Pull (state as synthetic Updates for
//! replica bootstrap). Relay and bootstrap speak the same Update
//! vocabulary as publishers, so replicas copy — they never agree.

use std::{
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
    engine::{Applied, ApplyOutcome, Engine, KeyspaceKey, SymbolKind},
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
    std::env::var("RADIX_RELAY_QUEUE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(RELAY_QUEUE)
}

/// Process counters for the metrics endpoint. Shared between the gRPC
/// service and the admin listener.
#[derive(Debug, Default)]
pub struct ServiceStats {
    pub applies: AtomicU64,
    pub queries: AtomicU64,
    pub relay_dropped: AtomicU64,
    /// Flipped true once the bootstrap pull (if any) has completed; the
    /// admin listener's /readyz reports it.
    pub ready: AtomicBool,
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
        let (delayed_tx, mut delayed_rx) =
            mpsc::channel::<(tokio::time::Instant, proto::Update)>(65_536);
        let apply_engine = Arc::clone(&engine);
        let apply_relay = relay.clone();
        let apply_stats = Arc::clone(&self.stats);
        let ack_tx = tx.clone();
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
                // Honor the latest injected staleness deadline in the
                // batch (fault-drill legs); zero-delay batches fall
                // through immediately.
                if let Some((deadline, _)) = batch.last() {
                    tokio::time::sleep_until(*deadline).await;
                }
                let msgs: Vec<UpdateMsg> = batch.iter().map(|(_, u)| UpdateMsg::from(u)).collect();
                let mut results: Vec<Applied> = Vec::with_capacity(msgs.len());
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
                            if peer.try_send(batch[idx].1.clone()).is_err() {
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
                // that match nothing — reject loudly instead.
                let scheme = update.keyspace.as_ref().map_or(0, |k| k.hash_scheme);
                if !crate::wire_hash::scheme_supported(scheme) {
                    tracing::warn!(scheme, holder = %update.holder, "unsupported hash scheme; update dropped");
                    continue;
                }
                let mut delay = Duration::ZERO;
                for event in &update.events {
                    match event.kind.as_ref() {
                        Some(proto::event::Kind::Stored(_)) => delay = delay.max(delays.0),
                        Some(proto::event::Kind::Removed(_)) => delay = delay.max(delays.1),
                        _ => {}
                    }
                }
                let deadline = tokio::time::Instant::now() + delay;
                if delayed_tx.send((deadline, update)).await.is_err() {
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
                let scores = engine.find_matches(&key, &hashes);
                let answer = proto::Match {
                    query_id: query.query_id,
                    scores: scores
                        .into_iter()
                        .map(|s| proto::HolderScore {
                            holder: s.holder,
                            matched_blocks: s.matched_blocks,
                            total_blocks: s.total_blocks,
                            event_fed: s.event_fed,
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

    async fn pull(
        &self,
        _request: Request<proto::PullRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        let snapshot = self.engine.snapshot();
        let updates = snapshot
            .iter()
            .map(|u| Ok(proto::Update::from(u)))
            .collect::<Vec<_>>();
        Ok(Response::new(Box::pin(futures::stream::iter(updates))))
    }
}

/// Bootstrap: pull the full state from `peer` and apply it before
/// serving. Returns Ok(applied_count); a connect failure is Ok(0) so a
/// lone first replica can boot cold.
pub async fn bootstrap_from(engine: &Engine, peer: &str) -> Result<usize, tonic::Status> {
    let Ok(client) = RadixIndexClient::connect(peer.to_string()).await else {
        return Ok(0);
    };
    let mut client = client
        .max_decoding_message_size(64 * 1024 * 1024)
        .max_encoding_message_size(64 * 1024 * 1024);
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
        engine.apply_snapshot(&UpdateMsg::from(&update));
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
    delay_stored: Duration,
    delay_removed: Duration,
    stats: Arc<ServiceStats>,
    shutdown: impl std::future::Future<Output = ()> + Send,
) -> Result<(), tonic::transport::Error> {
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
            let Ok(n) = socket.read(&mut buf).await else {
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
        ),
        gauges.keyspaces,
        gauges.holders,
        gauges.event_fed_holders,
        gauges.dropped_holders,
        gauges.blocks,
        stats.applies.load(Ordering::Relaxed),
        stats.queries.load(Ordering::Relaxed),
        stats.relay_dropped.load(Ordering::Relaxed),
    )
}
