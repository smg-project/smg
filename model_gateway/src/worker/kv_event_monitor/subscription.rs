//! The subscription task of one worker: connect, read the stream, reconnect
//! with backoff, hand the pushed load records on, and take the worker's
//! blocks out of the index when it leaves.

use std::{
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use smg_grpc_client::common_proto::{kv_cache_event, EngineLoad, KvEventBatch};
use tokio::sync::{oneshot, Semaphore};
use tracing::{debug, error, info, warn};

use super::{
    admission::{BatchOutcome, WorkerStreamState},
    apply::{WorkerIndexCounters, WorkerIndexState},
    KvBlockSize, KvEventMonitor,
};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::backend_client::BackendClient,
    worker::{
        kv_event_recovery::ResyncReason, kv_index_backend::KvIndex, liveness,
        monitor::WorkerMonitor, Worker,
    },
};

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Resolves when the worker is heard from (see `Worker::contact_wake`), or
/// never for a worker without a notifier.
async fn woken(wake: Option<&Arc<tokio::sync::Notify>>) {
    match wake {
        Some(notify) => notify.notified().await,
        None => std::future::pending().await,
    }
}

/// Take a contact noted before now (`notify_one` keeps one until it is
/// awaited), so that only a contact after this point ends a wait: the
/// connect of a stream that then failed was a contact, and says nothing
/// about the worker now.
fn drain_contact(wake: Option<&Arc<tokio::sync::Notify>>) {
    if let Some(notify) = wake {
        let mut notified = std::pin::pin!(notify.notified());
        // Consumes a stored permit without waiting; a waiter it registered
        // instead goes away with the future.
        let _ = notified.as_mut().enable();
    }
}

/// How long a subscription call may take to answer with headers. A port that
/// accepts but does not serve yet (an engine still starting) hangs the call.
const SUBSCRIBE_DEADLINE: Duration = Duration::from_secs(2);

/// A contact with the worker while a subscription call is pending (a poll
/// answered, a probe passed) says the worker serves now: a call still without
/// an answer this long after the contact is abandoned and retried. A live
/// server answers in milliseconds, so the contacts of a healthy worker (every
/// token it streams is one) never cut a call short.
const SUBSCRIBE_RETRY_GRACE: Duration = Duration::from_millis(500);

/// The least a reconnect waits, contact or not: a server that keeps closing
/// the stream of a worker that is otherwise talking must not be hammered.
const RECONNECT_FLOOR: Duration = Duration::from_millis(INITIAL_RECONNECT_DELAY_MS);

/// Maximum backoff between subscription attempts. Kept short: a worker that
/// restarts is healthy again within a few seconds, and until the stream is
/// back the blocks it stores are invisible to routing (the servicers resume
/// after the cursor and never resend them). A connect attempt is cheap.
const MAX_RECONNECT_DELAY_MS: u64 = 5_000;

/// Index cleanup is CPU-bound and can touch many blocks. Keep it
/// off Tokio workers and bound concurrent purges during fleet-wide drains.
pub(super) const MAX_CONCURRENT_INDEX_REMOVALS: usize = 4;
pub(super) static INDEX_REMOVAL_PERMITS: Semaphore =
    Semaphore::const_new(MAX_CONCURRENT_INDEX_REMOVALS);

/// Result of processing a stream connection to completion.
enum StreamResult {
    /// Stream closed normally (server-side).
    Ended,
    /// Stream produced an error.
    Error(tonic::Status),
    /// Detected a gap in sequence numbers.
    GapDetected { expected: u64, received: u64 },
}

/// The label a stream error is counted under: its gRPC status code.
fn error_class(code: tonic::Code) -> &'static str {
    match code {
        tonic::Code::Ok => "ok",
        tonic::Code::Cancelled => "cancelled",
        tonic::Code::Unknown => "unknown",
        tonic::Code::InvalidArgument => "invalid_argument",
        tonic::Code::DeadlineExceeded => "deadline_exceeded",
        tonic::Code::NotFound => "not_found",
        tonic::Code::AlreadyExists => "already_exists",
        tonic::Code::PermissionDenied => "permission_denied",
        tonic::Code::ResourceExhausted => "resource_exhausted",
        tonic::Code::FailedPrecondition => "failed_precondition",
        tonic::Code::Aborted => "aborted",
        tonic::Code::OutOfRange => "out_of_range",
        tonic::Code::Unimplemented => "unimplemented",
        tonic::Code::Internal => "internal",
        tonic::Code::Unavailable => "unavailable",
        tonic::Code::DataLoss => "data_loss",
        tonic::Code::Unauthenticated => "unauthenticated",
    }
}

/// What a round of a worker's subscription asks the loop to do next.
enum Round {
    /// Start the next round at once.
    Again,
    /// Sleep the reconnect backoff first, then grow it.
    Backoff,
    /// The worker is leaving: end the task.
    Shutdown,
    /// The backend serves no KV events: end the task and say so.
    Unsupported,
}

/// How a subscribe call under its deadline came out.
#[expect(
    clippy::large_enum_variant,
    reason = "one outcome per connect; the stream is handed on at once, never kept in the enum"
)]
enum Subscribed {
    /// The server answered with a stream.
    Stream(tonic::Streaming<KvEventBatch>),
    /// The server refused, or did not answer within the deadline.
    Refused(tonic::Status),
    /// The worker was heard from on another path while the call hung; a
    /// fresh call will land.
    Contacted,
    /// The worker left while the call was pending.
    Shutdown,
}

/// How a stream was drained.
enum Drained {
    /// The stream ended as `result` says; `applied` is whether the index
    /// applied a batch of it.
    Ended { result: StreamResult, applied: bool },
    /// The worker left while the stream was open.
    Shutdown,
}

/// The reconnect backoff: the delay the next wait sleeps, doubled after each
/// wait up to the cap, and back at the initial delay once a stream delivered
/// a batch or the worker asked for a fresh start.
struct Backoff {
    delay_ms: u64,
}

impl Backoff {
    const fn new() -> Self {
        Self {
            delay_ms: INITIAL_RECONNECT_DELAY_MS,
        }
    }

    fn delay(&self) -> Duration {
        Duration::from_millis(self.delay_ms)
    }

    fn delay_ms(&self) -> u64 {
        self.delay_ms
    }

    fn grow(&mut self) {
        self.delay_ms = (self.delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
    }

    fn reset(&mut self) {
        self.delay_ms = INITIAL_RECONNECT_DELAY_MS;
    }
}

/// What one worker's subscription task holds for its whole life, borrowed by
/// every round: the worker, its index handles and its contact source.
struct WorkerSubscription<'a> {
    worker: &'a Arc<dyn Worker>,
    worker_url: &'a str,
    worker_id: u32,
    indexer: &'a KvIndex,
    block_sizes: &'a DashMap<String, KvBlockSize>,
    model_id: &'a str,
    load_sink: Option<&'a Weak<WorkerMonitor>>,
    wake: Option<&'a Arc<tokio::sync::Notify>>,
}

/// Sleep `delay`, unless the worker leaves first: `true` when it did. A
/// contact with the worker ends the sleep early, but never before
/// `RECONNECT_FLOOR`, and only one made after the sleep began: the connect
/// of a stream that then failed does not count.
async fn sleep_or_shutdown(
    delay: Duration,
    wake: Option<&Arc<tokio::sync::Notify>>,
    shutdown_rx: &mut oneshot::Receiver<()>,
) -> bool {
    drain_contact(wake);
    tokio::select! {
        () = tokio::time::sleep(delay) => false,
        () = async {
            tokio::time::sleep(delay.min(RECONNECT_FLOOR)).await;
            woken(wake).await;
        } => false,
        _ = &mut *shutdown_rx => true,
    }
}

/// Subscribe to the worker's KV events from `start_seq`, with `deadline` on
/// the server's answer. A live server answers a subscription with headers
/// at once; one that does not within the deadline is unreachable (a
/// half-open connection behind a partition), and must not hang the loop. A
/// contact with the worker on another path (a poll, a probe) while the call
/// hangs ends it after a grace period, so a fresh call lands; the worker's
/// removal ends it at once (without this arm a call that never answers,
/// retried at every contact, would keep the task alive past the removal).
async fn subscribe_with_deadline(
    backend_client: &BackendClient,
    start_seq: u64,
    wake: Option<&Arc<tokio::sync::Notify>>,
    shutdown_rx: &mut oneshot::Receiver<()>,
    deadline: Duration,
) -> Subscribed {
    tokio::select! {
        attempt = tokio::time::timeout(
            deadline,
            backend_client.subscribe_kv_events(start_seq),
        ) => match attempt {
            Ok(Ok(stream)) => Subscribed::Stream(stream),
            Ok(Err(status)) => Subscribed::Refused(status),
            Err(_) => Subscribed::Refused(tonic::Status::unavailable(format!(
                "SubscribeKvEvents did not answer within {deadline:?}"
            ))),
        },
        () = async {
            woken(wake).await;
            tokio::time::sleep(SUBSCRIBE_RETRY_GRACE).await;
        } => Subscribed::Contacted,
        _ = &mut *shutdown_rx => Subscribed::Shutdown,
    }
}

impl KvEventMonitor {
    /// Learn the model's `block_size` from the stored blocks of a batch.
    ///
    /// The first stored block replaces a provisional size (a `WorkerSpec`
    /// seed): the event stream reflects the backend's actual page size.
    /// After that a block only raises the size. Hybrid engines publish,
    /// beside their full blocks, partial entries at their hash granularity
    /// (a fraction of the block, for a prefix boundary a later request can
    /// resume at); a request chunked at that size would never match a full
    /// block, so a smaller block never lowers the size of record.
    /// `CacheAwarePolicy` uses the size to chunk request tokens into blocks
    /// for overlap scoring.
    fn learn_block_size(
        block_sizes: &DashMap<String, KvBlockSize>,
        model_id: &str,
        batch: &KvEventBatch,
    ) {
        for event in &batch.events {
            let Some(kv_cache_event::Data::Stored(stored)) = &event.data else {
                continue;
            };
            let Some(block) = stored.blocks.first() else {
                continue;
            };
            let Ok(tokens) = usize::try_from(block.block_size) else {
                continue;
            };
            if tokens == 0 {
                continue;
            }
            // The common case, a block of the size of record, costs one shared
            // read; the write lock is taken only to change the size.
            if block_sizes
                .get(model_id)
                .is_some_and(|size| size.from_events && tokens <= size.tokens)
            {
                continue;
            }
            let mut entry = block_sizes
                .entry(model_id.to_string())
                .or_insert(KvBlockSize {
                    tokens: 0,
                    from_events: false,
                });
            if entry.from_events && tokens <= entry.tokens {
                continue;
            }
            let previous = (entry.tokens > 0).then_some(entry.tokens);
            *entry = KvBlockSize {
                tokens,
                from_events: true,
            };
            Metrics::set_kv_block_size(model_id, tokens);
            info!(
                model_id = %model_id,
                block_size = tokens,
                previous,
                "Learned block_size from KV event"
            );
        }
    }

    /// Take the worker's blocks out of the index when its subscription ends,
    /// on the blocking pool and under a fleet-wide bound, and log what its
    /// stream carried that the index did not take as is.
    async fn remove_indexer_worker(
        indexer: Arc<KvIndex>,
        worker_id: u32,
        worker_url: &str,
        worker_blocks: WorkerIndexState,
    ) {
        let Ok(permit) = INDEX_REMOVAL_PERMITS.acquire().await else {
            error!(worker_id, "Index cleanup semaphore closed");
            return;
        };
        let WorkerIndexState {
            blocks, counters, ..
        } = worker_blocks;
        if counters != WorkerIndexCounters::default() {
            debug!(
                worker_id,
                ?counters,
                "KV events the index did not take as is"
            );
        }
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            indexer.remove_worker(worker_id, blocks);
        })
        .await;

        if let Err(error) = result {
            error!(worker_id, %error, "Index worker cleanup task failed");
        }
        Metrics::set_kv_index_blocks(worker_url, 0);
    }

    /// Main subscription loop for a single worker.
    ///
    /// Owns the worker's index state and takes its blocks out of the index on
    /// exit. Exits when `shutdown_rx` fires or the backend returns
    /// `Unimplemented`. Each round ([`Self::subscription_round`]) gets the
    /// backend client, subscribes under the deadline and drains one stream,
    /// then says what comes next; this loop is the one place that sleeps the
    /// reconnect backoff and the one place that ends the task.
    pub(super) async fn subscription_loop(
        worker: Arc<dyn Worker>,
        worker_url: String,
        indexer: Arc<KvIndex>,
        block_sizes: Arc<DashMap<String, KvBlockSize>>,
        model_id: String,
        mut shutdown_rx: oneshot::Receiver<()>,
        load_sink: Option<Weak<WorkerMonitor>>,
    ) {
        let worker_id = match indexer.intern_worker(&worker_url) {
            Ok(id) => id,
            Err(e) => {
                error!(
                    worker_url = %worker_url,
                    error = %e,
                    "Failed to intern worker; KV events from this worker will \
                     not feed cache-aware routing"
                );
                Metrics::record_kv_event_subscription_failure(&worker_url, "intern_failed");
                return;
            }
        };
        let mut state = WorkerStreamState::of_rank(worker.dp_rank());
        // A contact with the worker (a poll answered, a probe passed) ends the
        // reconnect backoff early: a worker that is back gets its stream back
        // at once instead of after the remaining delay.
        let wake = worker.contact_wake();
        let subscription = WorkerSubscription {
            worker: &worker,
            worker_url: &worker_url,
            worker_id,
            indexer: &indexer,
            block_sizes: &block_sizes,
            model_id: &model_id,
            load_sink: load_sink.as_ref(),
            wake: wake.as_ref(),
        };
        let mut backoff = Backoff::new();

        let unsupported = loop {
            match Self::subscription_round(
                &subscription,
                &mut state,
                &mut backoff,
                &mut shutdown_rx,
            )
            .await
            {
                Round::Again => {}
                Round::Backoff => {
                    if sleep_or_shutdown(backoff.delay(), wake.as_ref(), &mut shutdown_rx).await {
                        break false;
                    }
                    backoff.grow();
                }
                Round::Shutdown => break false,
                Round::Unsupported => break true,
            }
        };
        Self::remove_indexer_worker(Arc::clone(&indexer), worker_id, &worker_url, state.index)
            .await;
        if unsupported {
            Metrics::set_kv_events_unavailable(&worker_url, true);
        }
    }

    /// One round of a worker's subscription: the backend client, the
    /// subscribe call under its deadline, the stream drained to its end;
    /// returns what the loop does next.
    async fn subscription_round(
        sub: &WorkerSubscription<'_>,
        state: &mut WorkerStreamState,
        backoff: &mut Backoff,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Round {
        let worker_url = sub.worker_url;
        let backend_client = match sub.worker.get_backend_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                // HTTP workers are filtered in on_worker_added, so this should
                // be unreachable. Retry defensively rather than exiting and
                // leaving a stale entry in worker_handles.
                warn!(
                    worker_url = %worker_url,
                    delay_ms = backoff.delay_ms(),
                    "Worker has no backend client yet, retrying"
                );
                return Round::Backoff;
            }
            Err(e) => {
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to get backend client, retrying"
                );
                return Round::Backoff;
            }
        };

        let start_seq = state.resume_sequence();
        let subscribed = subscribe_with_deadline(
            &backend_client,
            start_seq,
            sub.wake,
            shutdown_rx,
            SUBSCRIBE_DEADLINE,
        )
        .await;
        let stream = match subscribed {
            Subscribed::Stream(stream) => {
                info!(
                    worker_url = %worker_url,
                    start_seq,
                    "KV event stream connected"
                );
                Metrics::record_kv_event_subscription(worker_url);
                // The backoff is not reset here: a connect is no proof of
                // a working stream (one that fails on its first message
                // would be retried at the initial delay forever); it is
                // reset once the stream has delivered a batch.
                state.reconnected();
                liveness::on_contact(sub.worker);
                stream
            }
            Subscribed::Contacted => return Round::Again,
            Subscribed::Shutdown => return Round::Shutdown,
            Subscribed::Refused(e) => {
                // If the backend doesn't implement SubscribeKvEvents (some engines don't),
                // stop retrying — this RPC will never succeed.
                if e.code() == tonic::Code::Unimplemented {
                    warn!(
                        worker_url = %worker_url,
                        "Backend does not implement SubscribeKvEvents, \
                         disabling KV event subscription for this worker; \
                         cache-aware routing sees nothing of its cache"
                    );
                    return Round::Unsupported;
                }
                if e.code() == tonic::Code::OutOfRange {
                    warn!(
                        worker_url = %worker_url,
                        start_seq,
                        "KV event replay cursor expired; clearing worker state and requesting a current snapshot"
                    );
                    Self::reset_worker(
                        sub.indexer,
                        sub.worker_id,
                        state,
                        worker_url,
                        ResyncReason::OutOfRange,
                    );
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    liveness::on_contact_failed(sub.worker, "kv subscribe");
                }
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to subscribe to KV events, retrying"
                );
                return Round::Backoff;
            }
        };

        let (result, applied) = match Self::drain_stream(sub, stream, state, shutdown_rx).await {
            Drained::Ended { result, applied } => (result, applied),
            Drained::Shutdown => return Round::Shutdown,
        };
        if applied {
            // The stream worked: its failure starts the backoff over.
            backoff.reset();
        }
        if state.abandon_snapshot() {
            warn!(
                worker_url = %worker_url,
                "KV event stream ended during a relay snapshot; the next subscription \
                 starts over from zero"
            );
        }
        match result {
            StreamResult::Ended => {
                info!(
                    worker_url = %worker_url,
                    resume_from = state.resume_sequence(),
                    delay_ms = backoff.delay_ms(),
                    "KV event stream ended, reconnecting"
                );
                // Backoff to avoid tight reconnect loop if server keeps
                // closing the stream cleanly (e.g., rolling connections).
                Round::Backoff
            }
            StreamResult::Error(e) => {
                Metrics::record_kv_event_stream_error(worker_url, error_class(e.code()));
                if e.code() == tonic::Code::DataLoss {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        resume_from = state.resume_sequence(),
                        "KV event subscriber fell behind; clearing worker state and requesting a current snapshot"
                    );
                    Self::reset_worker(
                        sub.indexer,
                        sub.worker_id,
                        state,
                        worker_url,
                        ResyncReason::DataLoss,
                    );
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    liveness::on_contact_failed(sub.worker, "kv stream");
                }
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    resume_from = state.resume_sequence(),
                    delay_ms = backoff.delay_ms(),
                    "KV event stream error, reconnecting"
                );
                Round::Backoff
            }
            StreamResult::GapDetected { expected, received } => {
                warn!(
                    worker_url = %worker_url,
                    expected,
                    received,
                    "Sequence gap detected, reconnecting for replay from seq {expected}"
                );
                // No backoff: gap replay is a normal recovery path, and the
                // rank state asks for it once; if the server skips ahead
                // again the gap is settled instead of retried.
                Round::Again
            }
        }
    }

    /// Drain `stream` to its end: each batch applied to the index (a batch
    /// the index applied is the stream working), each batch's load record
    /// forwarded to the monitor only if its rank belongs to this worker; the
    /// worker's removal ends the stream at once.
    async fn drain_stream(
        sub: &WorkerSubscription<'_>,
        stream: tonic::Streaming<KvEventBatch>,
        state: &mut WorkerStreamState,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Drained {
        let mut applied = false;
        let on_batch = |batch: &KvEventBatch| {
            applied = true;
            liveness::on_contact(sub.worker);
            Self::learn_block_size(sub.block_sizes, sub.model_id, batch);
        };
        // The load record on a batch is a poll of this worker, received now.
        let on_load = |batch: &KvEventBatch, load: &EngineLoad| {
            let dp_rank = batch.dp_rank.unwrap_or(0);
            if dp_rank < 0
                || sub
                    .worker
                    .dp_rank()
                    .is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank))
            {
                return;
            }
            liveness::on_contact(sub.worker);
            if let Some(monitor) = sub.load_sink.and_then(Weak::upgrade) {
                monitor.apply_pushed_load(sub.worker, dp_rank, load, Instant::now());
            }
        };
        let result = tokio::select! {
            result = Self::process_stream(
                stream, sub.worker_url, sub.worker_id, sub.indexer, state, on_batch, on_load,
            ) => result,
            _ = &mut *shutdown_rx => return Drained::Shutdown,
        };
        Drained::Ended { result, applied }
    }

    /// Process batches from a single stream connection.
    async fn process_stream(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        mut on_batch: impl FnMut(&KvEventBatch),
        mut on_load: impl FnMut(&KvEventBatch, &EngineLoad),
    ) -> StreamResult {
        use tokio_stream::StreamExt;

        while let Some(result) = stream.next().await {
            let batch = match result {
                Ok(batch) => batch,
                Err(e) => return StreamResult::Error(e),
            };
            if let Some(load) = &batch.load {
                on_load(&batch, load);
                if Self::is_load_only(&batch) {
                    Metrics::record_kv_event_batch(worker_url, "load_only");
                    continue;
                }
            }
            if let BatchOutcome::Gap { expected, received } =
                Self::admit_batch(&batch, worker_url, worker_id, indexer, state, &mut on_batch)
            {
                return StreamResult::GapDetected { expected, received };
            }
        }

        StreamResult::Ended
    }

    /// A batch that carries only a load record (`EngineLoad.load_only`): no
    /// events, the last sequence repeated; it never enters admission.
    pub(super) fn is_load_only(batch: &KvEventBatch) -> bool {
        batch.load.as_ref().is_some_and(|load| load.load_only)
    }
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, pin::Pin, sync::Mutex};

    use futures::Stream;
    use kv_index::{compute_content_hash, SequenceHash, StoredBlock};
    use openai_protocol::worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerType};
    use smg_grpc_client::{
        common_proto as common,
        tokenspeed_scheduler::tokenspeed_proto::{
            self as ts,
            token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
        },
    };
    use tonic::{transport::Server, Request, Response, Status};

    use super::*;
    use crate::worker::{worker::StallReason, BasicWorkerBuilder};

    /// One batch of stored events, one block per event, `(hash, block_size)` each.
    fn stored_batch(blocks: &[(i64, usize)]) -> KvEventBatch {
        KvEventBatch {
            sequence_number: 1,
            events: blocks
                .iter()
                .map(|&(hash, block_size)| common::KvCacheEvent {
                    event_id: hash as u64,
                    data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
                        blocks: vec![common::KvBlock {
                            block_hash: hash,
                            token_ids: (0..block_size as u32).collect(),
                            block_size: block_size as i32,
                            ..Default::default()
                        }],
                        ..Default::default()
                    })),
                })
                .collect(),
            ..Default::default()
        }
    }

    fn learned(sizes: &DashMap<String, KvBlockSize>) -> Option<usize> {
        sizes.get("m").map(|size| size.tokens)
    }

    #[test]
    fn a_partial_entry_never_sets_the_block_size_below_the_full_block() {
        let sizes: DashMap<String, KvBlockSize> = DashMap::new();
        // The engine's first batch names a partial entry before a full block.
        KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(1, 64), (2, 128)]));
        assert_eq!(learned(&sizes), Some(128));
        // A partial entry on its own never lowers the size.
        KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(3, 64)]));
        assert_eq!(learned(&sizes), Some(128));
        // A larger block raises it.
        KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(4, 256)]));
        assert_eq!(learned(&sizes), Some(256));
    }

    #[test]
    fn the_first_stored_block_replaces_a_provisional_seed() {
        let sizes: DashMap<String, KvBlockSize> = DashMap::new();
        sizes.insert(
            "m".to_string(),
            KvBlockSize {
                tokens: 1152,
                from_events: false,
            },
        );
        KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(1, 128)]));
        assert_eq!(
            sizes.get("m").map(|size| *size),
            Some(KvBlockSize {
                tokens: 128,
                from_events: true,
            })
        );
        // Empty blocks and batches without stores change nothing.
        KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(2, 0)]));
        KvEventMonitor::learn_block_size(&sizes, "m", &KvEventBatch::default());
        assert_eq!(learned(&sizes), Some(128));
    }

    /// What the test scheduler's `SubscribeKvEvents` does.
    enum Subscribe {
        /// Accepts the call and never answers it, as an engine still
        /// starting behind an open port does.
        Hang,
        /// Answers at once with a stream that fails on its first message,
        /// as a server does whose first message the client cannot decode;
        /// each call's arrival is noted.
        FailFirstMessage(Arc<Mutex<Vec<Instant>>>),
        /// Answers `UNIMPLEMENTED`, as a backend whose KV event publisher is
        /// off does.
        Unimplemented,
        /// Notes the request, then never answers it, like `Hang`.
        RecordRequest(Arc<Mutex<Option<common::SubscribeKvEventsRequest>>>),
        /// Answers at once with a stream that yields the script's items,
        /// then ends.
        Stream(Vec<Result<KvEventBatch, Status>>),
        /// Answers at once with a stream that yields nothing and never ends.
        Open,
    }

    struct TestScheduler(Subscribe);

    type ServerStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

    #[tonic::async_trait]
    impl TokenSpeedScheduler for TestScheduler {
        type GenerateStream = ServerStream<ts::GenerateResponse>;
        type SubscribeKvEventsStream = ServerStream<KvEventBatch>;
        type GetTokenizerStream = ServerStream<common::GetTokenizerChunk>;

        async fn generate(
            &self,
            _: Request<ts::GenerateRequest>,
        ) -> Result<Response<Self::GenerateStream>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn health_check(
            &self,
            _: Request<ts::HealthCheckRequest>,
        ) -> Result<Response<ts::HealthCheckResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn abort(
            &self,
            _: Request<ts::AbortRequest>,
        ) -> Result<Response<ts::AbortResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn get_model_info(
            &self,
            _: Request<ts::GetModelInfoRequest>,
        ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn get_server_info(
            &self,
            _: Request<ts::GetServerInfoRequest>,
        ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn get_loads(
            &self,
            _: Request<ts::GetLoadsRequest>,
        ) -> Result<Response<ts::GetLoadsResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn subscribe_kv_events(
            &self,
            request: Request<common::SubscribeKvEventsRequest>,
        ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
            match &self.0 {
                Subscribe::Hang => std::future::pending().await,
                Subscribe::RecordRequest(seen) => {
                    *seen.lock().unwrap() = Some(request.into_inner());
                    std::future::pending().await
                }
                Subscribe::Unimplemented => {
                    Err(Status::unimplemented("KV cache events not enabled"))
                }
                Subscribe::Stream(script) => Ok(Response::new(Box::pin(futures::stream::iter(
                    script.clone(),
                )))),
                Subscribe::Open => Ok(Response::new(Box::pin(futures::stream::pending()))),
                Subscribe::FailFirstMessage(calls) => {
                    calls.lock().unwrap().push(Instant::now());
                    Ok(Response::new(Box::pin(futures::stream::once(async {
                        Err(Status::out_of_range(
                            "the first message is larger than the decode limit",
                        ))
                    }))))
                }
            }
        }

        async fn flush_cache(
            &self,
            _: Request<common::FlushCacheRequest>,
        ) -> Result<Response<common::FlushCacheResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn start_profile(
            &self,
            _: Request<common::StartProfileRequest>,
        ) -> Result<Response<common::ProfileResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn stop_profile(
            &self,
            _: Request<common::StopProfileRequest>,
        ) -> Result<Response<common::ProfileResponse>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }

        async fn get_tokenizer(
            &self,
            _: Request<common::GetTokenizerRequest>,
        ) -> Result<Response<Self::GetTokenizerStream>, Status> {
            Err(Status::unimplemented("test scheduler"))
        }
    }

    async fn wait_until_listening(addr: SocketAddr) {
        for _ in 0..100 {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("test scheduler did not start listening on {addr}");
    }

    fn grpc_worker(addr: SocketAddr) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://{addr}"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// The test scheduler served on a free port, listening.
    async fn serve(behaviour: Subscribe) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let port = portpicker::pick_unused_port().expect("a free port");
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test scheduler lives as long as the test"
        )]
        let server = tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(TokenSpeedSchedulerServer::new(TestScheduler(behaviour)))
                .serve(addr)
                .await;
        });
        wait_until_listening(addr).await;
        (addr, server)
    }

    /// A gRPC worker at `addr` and its backend client.
    async fn connected(addr: SocketAddr) -> (Arc<dyn Worker>, Arc<BackendClient>) {
        let worker = grpc_worker(addr);
        let client = worker
            .get_backend_client()
            .await
            .expect("a backend client")
            .expect("a gRPC worker has one");
        (worker, client)
    }

    /// The subscribe call under its deadline: a server that never answers is
    /// refused as unavailable at the deadline; a contact with the worker
    /// while the call hangs ends it for a fresh one; the worker's removal
    /// ends it at once; the server's own refusal is passed through.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_subscribe_call_ends_at_its_deadline_a_contact_or_the_removal() {
        let (addr, server) = serve(Subscribe::Hang).await;
        let (worker, client) = connected(addr).await;
        let wake = worker.contact_wake();
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();

        let Subscribed::Refused(status) = subscribe_with_deadline(
            &client,
            0,
            wake.as_ref(),
            &mut shutdown_rx,
            Duration::from_millis(100),
        )
        .await
        else {
            panic!("a call that never answers is refused at the deadline");
        };
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(
            status.message().contains("did not answer within"),
            "{status}"
        );

        let contact = {
            let worker = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "the contact lands while the call hangs and is awaited below"
            )]
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                worker.note_contact();
            })
        };
        assert!(matches!(
            subscribe_with_deadline(
                &client,
                0,
                wake.as_ref(),
                &mut shutdown_rx,
                Duration::from_secs(5)
            )
            .await,
            Subscribed::Contacted
        ));
        contact.await.unwrap();

        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
        shutdown_tx.send(()).unwrap();
        assert!(matches!(
            subscribe_with_deadline(
                &client,
                0,
                wake.as_ref(),
                &mut shutdown_rx,
                Duration::from_secs(5)
            )
            .await,
            Subscribed::Shutdown
        ));
        server.abort();

        let (addr, server) = serve(Subscribe::Unimplemented).await;
        let (_worker, client) = connected(addr).await;
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let Subscribed::Refused(status) =
            subscribe_with_deadline(&client, 0, None, &mut shutdown_rx, Duration::from_secs(5))
                .await
        else {
            panic!("the server's refusal is passed through");
        };
        assert_eq!(status.code(), tonic::Code::Unimplemented);
        server.abort();
    }

    /// A stream drained to its end says how it ended: closed by the server
    /// after a load-only batch (nothing applied, the load record was a
    /// contact with the worker), failed with the server's status, or cut by
    /// the worker's removal while open.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drained_stream_says_how_it_ended() {
        let load_only = KvEventBatch {
            load: Some(EngineLoad {
                load_only: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let behaviours = [
            Subscribe::Stream(vec![Ok(load_only)]),
            Subscribe::Stream(vec![Err(Status::data_loss("the subscriber fell behind"))]),
            Subscribe::Open,
        ];
        for (which, behaviour) in behaviours.into_iter().enumerate() {
            let (addr, server) = serve(behaviour).await;
            let (worker, client) = connected(addr).await;
            let indexer = KvIndex::chain();
            let worker_id = indexer.intern_worker(worker.url()).unwrap();
            let block_sizes = DashMap::new();
            let wake = worker.contact_wake();
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                worker_id,
                indexer: &indexer,
                block_sizes: &block_sizes,
                model_id: "m",
                load_sink: None,
                wake: wake.as_ref(),
            };
            let mut state = WorkerStreamState::default();
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
            let Subscribed::Stream(stream) =
                subscribe_with_deadline(&client, 0, None, &mut shutdown_rx, Duration::from_secs(5))
                    .await
            else {
                panic!("case {which}: the server answers with a stream");
            };
            drain_contact(wake.as_ref());
            if which == 2 {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the removal lands while the stream is open"
                )]
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let _ = shutdown_tx.send(());
                });
            }
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                KvEventMonitor::drain_stream(&sub, stream, &mut state, &mut shutdown_rx),
            )
            .await
            .unwrap_or_else(|_| panic!("case {which}: the drain ends"));
            match which {
                0 => {
                    assert!(
                        matches!(
                            drained,
                            Drained::Ended {
                                result: StreamResult::Ended,
                                applied: false
                            }
                        ),
                        "case 0: a stream closed after a load-only batch"
                    );
                    // The load record was a contact with the worker.
                    tokio::time::timeout(Duration::from_millis(100), woken(wake.as_ref()))
                        .await
                        .expect("case 0: the load record counted as a contact");
                }
                1 => {
                    let Drained::Ended {
                        result: StreamResult::Error(status),
                        applied: false,
                    } = drained
                    else {
                        panic!("case 1: a stream that fails ends with its status");
                    };
                    assert_eq!(status.code(), tonic::Code::DataLoss);
                }
                _ => assert!(
                    matches!(drained, Drained::Shutdown),
                    "case 2: the removal cuts an open stream"
                ),
            }
            server.abort();
        }
    }

    /// Load-only batches contact only their eligible rank, while KV events
    /// still contact the subscriber even when their rank is foreign (another
    /// rank's batch is the stream working; it is not indexed, see
    /// `a_rank_worker_indexes_its_ranks_batches_only`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pushed_load_contact_respects_virtual_rank_without_filtering_events() {
        for (pinned, rank, load_only, contact) in [
            (Some(1), Some(0), true, false),
            (Some(1), Some(-1), true, false),
            (Some(1), Some(1), true, true),
            (Some(1), None, true, false),
            (Some(0), None, true, true),
            (None, Some(2), true, true),
            (None, Some(-1), true, false),
            (Some(1), Some(0), false, true),
        ] {
            let batch = KvEventBatch {
                dp_rank: rank,
                load: Some(EngineLoad {
                    load_only,
                    ..Default::default()
                }),
                ..if load_only {
                    KvEventBatch::default()
                } else {
                    stored_batch(&[(42, 16)])
                }
            };
            let (addr, server) = serve(Subscribe::Stream(vec![Ok(batch)])).await;
            let (_transport_worker, client) = connected(addr).await;
            let mut builder = BasicWorkerBuilder::new(format!("grpc://{addr}"));
            if let Some(pinned) = pinned {
                builder = builder.dp_config(pinned, 8);
            }
            let worker: Arc<dyn Worker> = Arc::new(builder.build());
            let indexer = KvIndex::chain();
            let worker_id = indexer.intern_worker(worker.url()).unwrap();
            let block_sizes = DashMap::new();
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                worker_id,
                indexer: &indexer,
                block_sizes: &block_sizes,
                model_id: "m",
                load_sink: None,
                wake: None,
            };
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) =
                subscribe_with_deadline(&client, 0, None, &mut shutdown_rx, Duration::from_secs(5))
                    .await
            else {
                panic!("the server answers with a stream");
            };
            worker.set_stall(Some(StallReason::Unreachable));
            let mut state = WorkerStreamState::default();
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                KvEventMonitor::drain_stream(&sub, stream, &mut state, &mut shutdown_rx),
            )
            .await
            .expect("the finite stream ends");
            assert!(matches!(
                drained,
                Drained::Ended { result: StreamResult::Ended, applied } if applied != load_only
            ));
            assert_eq!(
                worker.stall_reason().is_none(),
                contact,
                "pinned={pinned:?}, rank={rank:?}, load_only={load_only}"
            );
            server.abort();
        }
    }

    /// Under `--dp-aware` an engine's stream names every rank's batches; a
    /// rank worker indexes its rank's and the unranked ones, never another
    /// rank's, so its overlap is its own cache, while every rank's batch
    /// moves its cursor, so the resubscribe cursor is the unpinned worker's.
    /// An unexpanded worker indexes them all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rank_worker_indexes_its_ranks_batches_only() {
        // Three blocks of distinct content (the chain index is keyed by the
        // tokens, not the engine's hash), numbered per admission cursor: a
        // batch that names no rank runs on rank 0's cursor.
        let ranked = |hash: i64, first_token: u32, sequence_number: u64, dp_rank: Option<i32>| {
            let mut batch = stored_batch(&[(hash, 16)]);
            if let Some(kv_cache_event::Data::Stored(stored)) = batch.events[0].data.as_mut() {
                stored.blocks[0].token_ids = (first_token..first_token + 16).collect();
            }
            batch.sequence_number = sequence_number;
            batch.dp_rank = dp_rank;
            batch
        };
        let batches = || {
            vec![
                Ok(ranked(42, 1_000, 1, Some(0))),
                Ok(ranked(43, 2_000, 1, Some(1))),
                Ok(ranked(44, 3_000, 2, None)),
            ]
        };
        for (pinned, indexed) in [(Some(1), 2), (Some(0), 2), (None, 3)] {
            let (addr, server) = serve(Subscribe::Stream(batches())).await;
            let (_transport_worker, client) = connected(addr).await;
            let mut builder = BasicWorkerBuilder::new(format!("grpc://{addr}"));
            if let Some(pinned) = pinned {
                builder = builder.dp_config(pinned, 2);
            }
            let worker: Arc<dyn Worker> = Arc::new(builder.build());
            let indexer = KvIndex::chain();
            let worker_id = indexer.intern_worker(worker.url()).unwrap();
            let block_sizes = DashMap::new();
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                worker_id,
                indexer: &indexer,
                block_sizes: &block_sizes,
                model_id: "m",
                load_sink: None,
                wake: None,
            };
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) =
                subscribe_with_deadline(&client, 0, None, &mut shutdown_rx, Duration::from_secs(5))
                    .await
            else {
                panic!("the server answers with a stream");
            };
            let mut state = WorkerStreamState::of_rank(worker.dp_rank());
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                KvEventMonitor::drain_stream(&sub, stream, &mut state, &mut shutdown_rx),
            )
            .await
            .expect("the finite stream ends");
            assert!(matches!(
                drained,
                Drained::Ended {
                    result: StreamResult::Ended,
                    applied: true
                }
            ));
            assert_eq!(
                indexer.worker_block_count(worker_id),
                indexed,
                "pinned={pinned:?}"
            );
            assert_eq!(
                state.resume_sequence(),
                2,
                "rank 0's cursor moved by rank 0's and the unranked batch, pinned={pinned:?}"
            );
            server.abort();
        }
    }

    /// A subscribe call that never answers, retried at every contact with
    /// the worker, must still see the worker's removal: with no shutdown arm
    /// on the call the task lived on and `on_worker_removed` never returned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn removal_ends_a_subscribe_call_that_never_answers() {
        let port = portpicker::pick_unused_port().expect("a free port");
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test scheduler lives as long as the test"
        )]
        let server = tokio::spawn(
            Server::builder()
                .add_service(TokenSpeedSchedulerServer::new(TestScheduler(
                    Subscribe::Hang,
                )))
                .serve(addr),
        );
        wait_until_listening(addr).await;

        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor.on_worker_added(&worker).await;
        // Let the task connect and park in the subscribe call.
        tokio::time::sleep(Duration::from_millis(500)).await;
        // The worker keeps being heard from on other paths; every contact
        // retries the parked call well inside its deadline.
        let pinger = {
            let worker = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "the contact source is stopped at the end of the test"
            )]
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    worker.note_contact();
                }
            })
        };

        tokio::time::timeout(
            Duration::from_secs(5),
            monitor.on_worker_removed(worker.url()),
        )
        .await
        .expect("removal returns while the subscribe call hangs");

        pinger.abort();
        server.abort();
    }

    /// The subscription asks the servicer for load records before the
    /// publisher's first batch: this gateway admits a `load_only` batch to
    /// no cursor, and the silence of an engine frozen on the first request
    /// it ever receives reaches the wedge rule through no other record.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_subscription_asks_for_load_records_before_the_first_batch() {
        let port = portpicker::pick_unused_port().expect("a free port");
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let seen = Arc::new(Mutex::new(None));
        #[expect(
            clippy::disallowed_methods,
            reason = "the test scheduler lives as long as the test"
        )]
        let server = tokio::spawn(
            Server::builder()
                .add_service(TokenSpeedSchedulerServer::new(TestScheduler(
                    Subscribe::RecordRequest(Arc::clone(&seen)),
                )))
                .serve(addr),
        );
        wait_until_listening(addr).await;
        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor.on_worker_added(&worker).await;
        let deadline = Instant::now() + Duration::from_secs(5);
        let request = loop {
            if let Some(request) = *seen.lock().unwrap() {
                break request;
            }
            assert!(
                Instant::now() < deadline,
                "the subscribe call never reached the scheduler"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert!(request.load_records_before_first_batch, "{request:?}");
        assert_eq!(request.start_sequence_number, 0);
        tokio::time::timeout(
            Duration::from_secs(5),
            monitor.on_worker_removed(worker.url()),
        )
        .await
        .expect("removal returns while the subscribe call hangs");
        server.abort();
    }

    /// A stream that fails on its first message is a failed subscription,
    /// not a working one: the delay before the next attempt doubles from
    /// attempt to attempt instead of restarting at the initial delay on
    /// every connect, the worker's own answer to the subscribe call does
    /// not end the wait early, and the failure is counted by its code.
    #[test]
    fn a_stream_that_fails_on_its_first_message_backs_off_and_is_counted() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let port = portpicker::pick_unused_port().expect("a free port");
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        // One thread, so the subscription task's metrics land on this
        // thread's recorder.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the test scheduler lives as long as the test"
                )]
                let server = tokio::spawn(
                    Server::builder()
                        .add_service(TokenSpeedSchedulerServer::new(TestScheduler(
                            Subscribe::FailFirstMessage(Arc::clone(&calls)),
                        )))
                        .serve(addr),
                );
                wait_until_listening(addr).await;
                let worker = grpc_worker(addr);
                let monitor = KvEventMonitor::new();
                monitor.on_worker_added(&worker).await;
                tokio::time::sleep(Duration::from_millis(1_900)).await;
                monitor.on_worker_removed(worker.url()).await;
                server.abort();
            });
        });

        // Attempts at about 0, 100, 300, 700 and 1,500 ms: five in 1.9 s,
        // where a backoff reset on every connect made nineteen.
        let calls = calls.lock().unwrap().clone();
        assert!(
            (3..=6).contains(&calls.len()),
            "{} subscribe calls in 1.9 s",
            calls.len()
        );
        let gaps: Vec<Duration> = calls.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(
            gaps.windows(2).all(|pair| pair[1] > pair[0]),
            "the gaps do not grow: {gaps:?}"
        );
        assert!(
            gaps.last().unwrap() >= &Duration::from_millis(600),
            "{gaps:?}"
        );

        let rendered = handle.render();
        let line = rendered
            .lines()
            .find(|line| line.starts_with("smg_kv_event_stream_errors_total{"))
            .unwrap_or_else(|| panic!("no stream error counted:\n{rendered}"));
        assert!(
            line.contains(&format!("worker=\"grpc://{addr}\"")),
            "{line}"
        );
        assert!(line.contains("error=\"out_of_range\""), "{line}");
        let errors: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
        assert!(
            errors >= 3.0 && errors as usize <= calls.len(),
            "{errors} errors for {} calls",
            calls.len()
        );
    }

    /// A backend that serves no KV events (an engine whose publisher is off)
    /// left one WARN line as the only trace of a cache-aware router routing
    /// blind: the worker carries a gauge from the answer until it leaves.
    #[test]
    fn a_backend_without_kv_events_is_a_gauge_until_the_worker_leaves() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn gauge(handle: &PrometheusHandle, worker_url: &str) -> Option<f64> {
            let prefix = format!("smg_kv_events_unavailable{{worker=\"{worker_url}\"}}");
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&prefix))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        // The subscription task publishes from the runtime's thread, which
        // must be the one holding the local recorder.
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let port = portpicker::pick_unused_port().expect("a free port");
                let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the test scheduler lives as long as the test"
                )]
                let server = tokio::spawn(
                    Server::builder()
                        .add_service(TokenSpeedSchedulerServer::new(TestScheduler(
                            Subscribe::Unimplemented,
                        )))
                        .serve(addr),
                );
                wait_until_listening(addr).await;

                let worker = grpc_worker(addr);
                let monitor = KvEventMonitor::new();
                monitor.on_worker_added(&worker).await;
                for _ in 0..500 {
                    if gauge(&handle, worker.url()) == Some(1.0) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                assert_eq!(
                    gauge(&handle, worker.url()),
                    Some(1.0),
                    "the answer raises the gauge"
                );

                monitor.on_worker_removed(worker.url()).await;
                assert_eq!(
                    gauge(&handle, worker.url()),
                    Some(0.0),
                    "the worker's removal lowers it"
                );
                server.abort();
            });
        });
    }

    #[tokio::test]
    async fn a_removed_workers_blocks_leave_the_index_off_the_runtime() {
        let indexer = Arc::new(KvIndex::chain());
        let worker_id = indexer.intern_worker("http://w1:8000").unwrap();
        let mut worker_blocks = WorkerIndexState::default();
        indexer
            .apply_stored(
                worker_id,
                &[StoredBlock {
                    seq_hash: SequenceHash(1),
                    content_hash: compute_content_hash(&[1, 2, 3]),
                }],
                None,
                &mut worker_blocks.blocks,
            )
            .unwrap();

        KvEventMonitor::remove_indexer_worker(
            Arc::clone(&indexer),
            worker_id,
            "grpc://w1:9000",
            worker_blocks,
        )
        .await;

        assert_eq!(indexer.current_size(), 0);
    }
}
