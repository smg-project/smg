//! The subscription task of one worker: connect, read the stream, reconnect
//! with backoff, hand the pushed load records on, say to the sink what the
//! stream brought, and say to it that the worker left.

use std::{
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use smg_grpc_client::common_proto::{kv_cache_event, EngineLoad, KvEventBatch, KvSnapshotChunk};
use tokio::sync::{oneshot, Semaphore};
use tracing::{error, info, warn};

use super::{
    port::{BatchFacts, BatchOutcome, KvEventSink, ResyncReason, SnapshotChunk},
    KvEventMonitor, PushedLoadSink, SubscriptionKind,
};
use crate::worker::{
    connection::engine::EngineConnection, health::liveness, identity::endpoint::Endpoint,
    metrics::Metrics, Worker,
};

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Resolves when the worker is heard from (see `WorkerRuntime::contact_wake`), or
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
    /// The stream ended as `result` says; `applied` is whether the sink took
    /// a batch of it.
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

/// What a worker's stream feeds: a sink (its KV events, under the sink's id
/// for the worker, with the load records that ride them), or the worker
/// monitor alone (its load records, asked for with `loads_only`; a servicer
/// that predates the field streams its events too, which are dropped).
enum Role<'a, S: KvEventSink> {
    Index { sink: &'a S, id: S::Id },
    Loads,
}

impl<S: KvEventSink> Clone for Role<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: KvEventSink> Copy for Role<'_, S> {}

impl<S: KvEventSink> Role<'_, S> {
    fn kind(self) -> SubscriptionKind {
        match self {
            Self::Index { .. } => SubscriptionKind::Events,
            Self::Loads => SubscriptionKind::Loads,
        }
    }

    /// The cursor to send when (re)subscribing: rank 0's last applied
    /// sequence as the sink holds it (the servicers replay rank 0's
    /// publisher), 0 where the sink has no cursor for it or the stream feeds
    /// no sink.
    fn resume_sequence(self) -> u64 {
        match self {
            Self::Index { sink, id } => sink.cursor(id, 0).map_or(0, |cursor| cursor.applied),
            Self::Loads => 0,
        }
    }
}

/// What one worker's subscription task holds for its whole life, borrowed by
/// every round: the worker, what its stream feeds and its contact source.
struct WorkerSubscription<'a, S: KvEventSink> {
    worker: &'a Arc<dyn Worker>,
    worker_url: &'a str,
    role: Role<'a, S>,
    model_id: &'a str,
    load_sink: Option<&'a Weak<dyn PushedLoadSink>>,
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

/// Subscribe to the worker's KV events from `start_seq` (`Events`), or to
/// its load records alone (`Loads`), with `deadline` on the server's answer.
/// A live server answers a subscription with headers
/// at once; one that does not within the deadline is unreachable (a
/// half-open connection behind a partition), and must not hang the loop. A
/// contact with the worker on another path (a poll, a probe) while the call
/// hangs ends it after a grace period, so a fresh call lands; the worker's
/// removal ends it at once (without this arm a call that never answers,
/// retried at every contact, would keep the task alive past the removal).
async fn subscribe_with_deadline(
    backend_client: &dyn EngineConnection,
    start_seq: u64,
    kind: SubscriptionKind,
    wake: Option<&Arc<tokio::sync::Notify>>,
    shutdown_rx: &mut oneshot::Receiver<()>,
    deadline: Duration,
) -> Subscribed {
    let call = async {
        match kind {
            SubscriptionKind::Events => backend_client.subscribe_kv_events(start_seq).await,
            SubscriptionKind::Loads => backend_client.subscribe_load_records().await,
        }
    };
    tokio::select! {
        attempt = tokio::time::timeout(deadline, call) => match attempt {
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
    /// Tell the sink the worker left, when its subscription ends: on the
    /// blocking pool and under a fleet-wide bound, since the worker's blocks
    /// leave the sink's index inside the call.
    async fn remove_from_sink<S>(sink: Arc<S>, id: S::Id, worker_url: &str)
    where
        S: KvEventSink + 'static,
    {
        let Ok(permit) = INDEX_REMOVAL_PERMITS.acquire().await else {
            error!(
                worker_url = %Endpoint::redacted_str(worker_url),
                "Index cleanup semaphore closed"
            );
            return;
        };
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            sink.removed(id);
        })
        .await;

        if let Err(error) = result {
            error!(
                worker_url = %Endpoint::redacted_str(worker_url),
                %error,
                "Index worker cleanup task failed"
            );
        }
    }

    /// Main subscription loop for a single worker.
    ///
    /// With a `feed` (an events subscription: the sink and its id for the
    /// worker) it says to the sink what the stream brings and, on exit, that
    /// the worker left; without one it streams the worker's load records
    /// alone. Exits when `shutdown_rx`
    /// fires or the backend returns `Unimplemented`. Each round
    /// ([`Self::subscription_round`]) gets the backend client, subscribes
    /// under the deadline and drains one stream, then says what comes next;
    /// this loop is the one place that sleeps the reconnect backoff and the
    /// one place that ends the task.
    pub(super) async fn subscription_loop<S>(
        worker: Arc<dyn Worker>,
        worker_url: String,
        feed: Option<(Arc<S>, S::Id)>,
        model_id: String,
        mut shutdown_rx: oneshot::Receiver<()>,
        load_sink: Option<Weak<dyn PushedLoadSink>>,
    ) where
        S: KvEventSink<Batch = KvEventBatch> + 'static,
    {
        let role = match &feed {
            Some((sink, id)) => Role::Index {
                sink: Arc::as_ref(sink),
                id: *id,
            },
            None => Role::Loads,
        };
        // A contact with the worker (a poll answered, a probe passed) ends the
        // reconnect backoff early: a worker that is back gets its stream back
        // at once instead of after the remaining delay.
        let wake = worker.runtime_state().map(|state| state.contact_wake());
        let subscription = WorkerSubscription {
            worker: &worker,
            worker_url: &worker_url,
            role,
            model_id: &model_id,
            load_sink: load_sink.as_ref(),
            wake: wake.as_ref(),
        };
        let mut backoff = Backoff::new();

        let unsupported = loop {
            match Self::subscription_round(&subscription, &mut backoff, &mut shutdown_rx).await {
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
        let events = feed.is_some();
        if let Some((sink, id)) = feed {
            Self::remove_from_sink(sink, id, &worker_url).await;
        }
        // The gauge is the index's: a loads-only subscription a backend refuses
        // leaves the worker on the poll, which is the designed fallback.
        if unsupported && events {
            Metrics::set_kv_events_unavailable(&worker_url, true);
        }
    }

    /// One round of a worker's subscription: the backend client, the
    /// subscribe call under its deadline, the stream drained to its end;
    /// returns what the loop does next.
    async fn subscription_round<S: KvEventSink<Batch = KvEventBatch>>(
        sub: &WorkerSubscription<'_, S>,
        backoff: &mut Backoff,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Round {
        let worker_url = sub.worker_url;
        let backend_client = match sub.worker.engine_connection().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                // HTTP workers are filtered in on_worker_added, so this should
                // be unreachable. Retry defensively rather than exiting and
                // leaving a stale entry in worker_handles.
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    delay_ms = backoff.delay_ms(),
                    "Worker has no backend client yet, retrying"
                );
                return Round::Backoff;
            }
            Err(e) => {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to get backend client, retrying"
                );
                return Round::Backoff;
            }
        };

        let start_seq = sub.role.resume_sequence();
        // The streams are the connection's own calls: one whose wire carries
        // none refuses them as unimplemented, and is read as an engine that
        // serves none.
        let subscribed = subscribe_with_deadline(
            backend_client.as_ref(),
            start_seq,
            sub.role.kind(),
            sub.wake,
            shutdown_rx,
            SUBSCRIBE_DEADLINE,
        )
        .await;
        let stream = match subscribed {
            Subscribed::Stream(stream) => {
                info!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    start_seq,
                    "KV event stream connected"
                );
                Metrics::record_kv_event_subscription(worker_url);
                // The backoff is not reset here: a connect is no proof of
                // a working stream (one that fails on its first message
                // would be retried at the initial delay forever); it is
                // reset once the stream has delivered a batch.
                if let Role::Index { sink, id } = sub.role {
                    sink.reconnected(id);
                }
                liveness::on_contact(sub.worker);
                stream
            }
            Subscribed::Contacted => return Round::Again,
            Subscribed::Shutdown => return Round::Shutdown,
            Subscribed::Refused(e) => {
                // If the backend doesn't implement SubscribeKvEvents (some engines don't),
                // stop retrying — this RPC will never succeed.
                if e.code() == tonic::Code::Unimplemented {
                    match sub.role {
                        // The one line an operator gets for a worker whose
                        // cache the cache-aware policy cannot see: it names
                        // the model and carries the servicer's own reason
                        // (the publisher is off, or the engine cannot serve
                        // this model's cache groups). The model itself stays
                        // on its event index while any other worker of it
                        // publishes; only when none does is it routed on the
                        // approximate prefix tree.
                        Role::Index { .. } => warn!(
                            worker_url = %Endpoint::redacted_str(worker_url),
                            model_id = sub.model_id,
                            reason = e.message(),
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this worker; \
                             cache-aware routing sees nothing of this worker's \
                             cache, and the model falls back to the approximate \
                             prefix tree only when no worker of it publishes \
                             events"
                        ),
                        Role::Loads => info!(
                            worker_url = %Endpoint::redacted_str(worker_url),
                            "Backend does not implement SubscribeKvEvents, \
                             disabling the load stream for this worker; its load \
                             stays polled"
                        ),
                    }
                    return Round::Unsupported;
                }
                if e.code() == tonic::Code::OutOfRange {
                    warn!(
                        worker_url = %Endpoint::redacted_str(worker_url),
                        start_seq,
                        "KV event replay cursor expired; clearing worker state and requesting a current snapshot"
                    );
                    if let Role::Index { sink, id } = sub.role {
                        sink.resync(id, None, ResyncReason::OutOfRange);
                    }
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    liveness::on_contact_failed(sub.worker, "kv subscribe");
                }
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to subscribe to KV events, retrying"
                );
                return Round::Backoff;
            }
        };

        // The chunks still to come of a relay's snapshot the stream is
        // inside, by this round's own count.
        let mut snapshot = None;
        let (result, applied) =
            match Self::drain_stream(sub, stream, &mut snapshot, shutdown_rx).await {
                Drained::Ended { result, applied } => (result, applied),
                Drained::Shutdown => return Round::Shutdown,
            };
        if applied {
            // The stream worked: its failure starts the backoff over.
            backoff.reset();
        }
        if snapshot.is_some() {
            if let Role::Index { sink, id } = sub.role {
                sink.resync(id, None, ResyncReason::SnapshotAbandoned);
            }
            warn!(
                worker_url = %Endpoint::redacted_str(worker_url),
                "KV event stream ended during a relay snapshot; the next subscription \
                 starts over from zero"
            );
        }
        match result {
            StreamResult::Ended => {
                info!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    resume_from = sub.role.resume_sequence(),
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
                        worker_url = %Endpoint::redacted_str(worker_url),
                        error = %e,
                        resume_from = sub.role.resume_sequence(),
                        "KV event subscriber fell behind; clearing worker state and requesting a current snapshot"
                    );
                    if let Role::Index { sink, id } = sub.role {
                        sink.resync(id, None, ResyncReason::DataLoss);
                    }
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    liveness::on_contact_failed(sub.worker, "kv stream");
                }
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    error = %e,
                    resume_from = sub.role.resume_sequence(),
                    delay_ms = backoff.delay_ms(),
                    "KV event stream error, reconnecting"
                );
                Round::Backoff
            }
            StreamResult::GapDetected { expected, received } => {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
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

    /// The load record on a batch is a poll of this worker, received now:
    /// forwarded to the monitor only if its rank belongs to this worker.
    fn forward_record<S: KvEventSink>(
        sub: &WorkerSubscription<'_, S>,
        batch: &KvEventBatch,
        load: &EngineLoad,
    ) {
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
        if let Some(sink) = sub.load_sink.and_then(Weak::upgrade) {
            sink.push_load(sub.worker, dp_rank, load, Instant::now());
        }
    }

    /// Drain `stream` to its end: on an events subscription each batch goes
    /// to the sink (a batch the sink took is the stream working, and a
    /// contact with the worker) and each batch's load record is forwarded;
    /// on a loads-only one every batch is the stream working, its record
    /// forwarded and its events, if any, dropped. `snapshot` is left at the
    /// chunks still to come of a relay's snapshot the stream ended inside.
    /// The worker's removal ends the stream at once.
    async fn drain_stream<S: KvEventSink<Batch = KvEventBatch>>(
        sub: &WorkerSubscription<'_, S>,
        stream: tonic::Streaming<KvEventBatch>,
        snapshot: &mut Option<u32>,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Drained {
        let mut applied = false;
        let on_load =
            |batch: &KvEventBatch, load: &EngineLoad| Self::forward_record(sub, batch, load);
        let result = match sub.role {
            Role::Index { sink, id } => {
                let on_batch = || {
                    applied = true;
                    liveness::on_contact(sub.worker);
                };
                tokio::select! {
                    result = Self::process_stream(
                        stream, sub.worker_url, sink, id, snapshot, on_batch, on_load,
                    ) => result,
                    _ = &mut *shutdown_rx => return Drained::Shutdown,
                }
            }
            Role::Loads => {
                let on_batch = |_: &KvEventBatch| applied = true;
                tokio::select! {
                    result = Self::process_records(stream, sub.worker_url, on_batch, on_load) => result,
                    _ = &mut *shutdown_rx => return Drained::Shutdown,
                }
            }
        };
        Drained::Ended { result, applied }
    }

    /// Drain a loads-only stream: every record to `on_load`; a batch of
    /// events (a servicer that predates `loads_only`) is dropped, counted
    /// `not_indexed`. Nothing is admitted and no gap is looked for: the
    /// stream has no cursor.
    async fn process_records(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        mut on_batch: impl FnMut(&KvEventBatch),
        mut on_load: impl FnMut(&KvEventBatch, &EngineLoad),
    ) -> StreamResult {
        use tokio_stream::StreamExt;

        while let Some(result) = stream.next().await {
            let batch = match result {
                Ok(batch) => batch,
                Err(e) => return StreamResult::Error(e),
            };
            on_batch(&batch);
            if let Some(load) = &batch.load {
                on_load(&batch, load);
            }
            let disposition = if Self::is_load_only(&batch) {
                "load_only"
            } else {
                "not_indexed"
            };
            Metrics::record_kv_event_batch(worker_url, disposition);
        }

        StreamResult::Ended
    }

    /// Process batches from a single stream connection: every batch that is
    /// more than a load record goes to the sink with its header facts, a
    /// snapshot's start said ahead of its first chunk. A batch the sink took
    /// is `on_batch`'s; a gap ends the stream, for a replay.
    async fn process_stream<S: KvEventSink<Batch = KvEventBatch>>(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        sink: &S,
        id: S::Id,
        snapshot: &mut Option<u32>,
        mut on_batch: impl FnMut(),
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
            if let Some(chunk) = &batch.snapshot {
                if count_snapshot_chunk(snapshot, chunk) {
                    sink.resync(id, None, ResyncReason::Snapshot);
                }
            }
            match sink.admit(id, batch_rank(&batch), batch_facts(&batch), &batch) {
                BatchOutcome::Applied => on_batch(),
                BatchOutcome::Skipped => {}
                BatchOutcome::Gap { expected, received } => {
                    return StreamResult::GapDetected { expected, received };
                }
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

/// The header facts of a batch, as a sink's cursor reads them.
pub(super) fn batch_facts(batch: &KvEventBatch) -> BatchFacts {
    BatchFacts {
        sequence: batch.sequence_number,
        clears: batch.events.iter().any(|event| {
            matches!(&event.data, Some(kv_cache_event::Data::Cleared(cleared)) if cleared.ownership.is_none())
        }),
        snapshot: batch.snapshot.as_ref().map(|chunk| SnapshotChunk {
            unknown_before: chunk.unknown_before > 0,
        }),
    }
}

/// The rank a batch names, 0 for a batch that names none.
pub(super) fn batch_rank(batch: &KvEventBatch) -> u16 {
    batch
        .dp_rank
        .and_then(|rank| u16::try_from(rank).ok())
        .unwrap_or(0)
}

/// Count one chunk of a relay's snapshot on its stream. `open` is the chunks
/// still to come of the snapshot the stream is inside, `None` outside one: a
/// first chunk, or a chunk outside a snapshot, opens one of the count the
/// chunk says, and the chunk that fills the count closes it. A stream that
/// ends with a snapshot open has brought a partial live set, which the
/// collector says to the sink. Returns whether the chunk opened a snapshot:
/// the start the collector says ahead of the chunk.
fn count_snapshot_chunk(open: &mut Option<u32>, chunk: &KvSnapshotChunk) -> bool {
    let (opens, left) = match *open {
        Some(left) if chunk.index != 0 => (false, left),
        _ => (true, chunk.count.max(1)),
    };
    *open = (left > 1).then(|| left - 1);
    opens
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, pin::Pin, sync::Mutex};

    use dashmap::DashMap;
    use futures::Stream;
    use openai_protocol::{
        model_card::ModelCard,
        worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerType},
    };
    use smg_grpc_client::{
        common_proto as common,
        tokenspeed_scheduler::tokenspeed_proto::{
            self as ts,
            token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
        },
    };
    use tonic::{transport::Server, Request, Response, Status};
    use tracing_test::traced_test;

    use super::{
        super::{
            index_sink::{IndexSink, OpenRow},
            port::KvCursor,
            KvBlockSize,
        },
        *,
    };
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyRegistry,
        routers::grpc::{backend_client::BackendClient, test_engine::serve_on_loopback},
        worker::{
            identity::basic::StallReason, kv::index_backend::KvIndex, BasicWorkerBuilder,
            WorkerMonitor, WorkerRegistry, UNKNOWN_MODEL_ID,
        },
    };

    /// A sink over one fresh index with a row open in it, as an events
    /// subscription starts: for a worker that stands for `rank` of its
    /// engine, or for the whole of it.
    fn open_row(worker_url: &str, rank: Option<usize>) -> (Arc<KvIndex>, Arc<IndexSink>, OpenRow) {
        let index = Arc::new(KvIndex::chain());
        let sink = Arc::new(IndexSink::new(Arc::new(DashMap::new())));
        let row = sink
            .open(Arc::clone(&index), "m", worker_url, rank)
            .expect("the index interns the worker");
        (index, sink, row)
    }

    /// A sink that keeps what the collector said to it, in order, and
    /// answers from a script: every batch taken, the cursor at `applied`.
    #[derive(Default)]
    struct Said {
        calls: Mutex<Vec<String>>,
        applied: u64,
    }

    impl Said {
        fn note(&self, call: String) {
            self.calls.lock().unwrap().push(call);
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl KvEventSink for Said {
        type Batch = KvEventBatch;
        type Id = u8;

        fn admit(&self, id: u8, rank: u16, facts: BatchFacts, _: &KvEventBatch) -> BatchOutcome {
            let chunk = if facts.snapshot.is_some() {
                " chunk"
            } else {
                ""
            };
            self.note(format!(
                "admit {id} rank {rank} seq {}{chunk}",
                facts.sequence
            ));
            BatchOutcome::Applied
        }

        fn reconnected(&self, id: u8) {
            self.note(format!("reconnected {id}"));
        }

        fn resync(&self, id: u8, rank: Option<u16>, reason: ResyncReason) {
            self.note(format!("resync {id} {rank:?} {}", reason.as_str()));
        }

        fn cursor(&self, id: u8, rank: u16) -> Option<KvCursor> {
            self.note(format!("cursor {id} rank {rank}"));
            Some(KvCursor {
                applied: self.applied,
                ..KvCursor::default()
            })
        }

        fn removed(&self, id: u8) {
            self.note(format!("removed {id}"));
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            None
        }

        fn set_block_size(&self, _: &str, _: u32) {}
    }

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

    /// The reconnect delay a failing stream waits doubles from the initial
    /// delay to the cap, stays at the cap once there, and a reset returns
    /// it to the initial delay.
    #[test]
    fn the_reconnect_backoff_doubles_to_its_cap_and_a_reset_starts_over() {
        let mut backoff = Backoff::new();
        assert_eq!(backoff.delay_ms(), INITIAL_RECONNECT_DELAY_MS);
        let mut delays = Vec::new();
        for _ in 0..8 {
            backoff.grow();
            delays.push(backoff.delay_ms());
        }
        assert_eq!(
            delays,
            [200, 400, 800, 1_600, 3_200, 5_000, 5_000, 5_000],
            "doubled from 100 ms up to the 5 s cap, then held there"
        );
        assert_eq!(
            backoff.delay(),
            Duration::from_millis(MAX_RECONNECT_DELAY_MS)
        );
        backoff.reset();
        assert_eq!(
            backoff.delay_ms(),
            INITIAL_RECONNECT_DELAY_MS,
            "a reset returns the initial delay"
        );
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

    /// A rank worker of a data-parallel engine at `addr`: rank `rank` of `size`.
    fn rank_worker(addr: SocketAddr, rank: usize, size: usize) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://{addr}"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .dp_config(rank, size)
                .build(),
        )
    }

    /// The test scheduler served on a loopback port bound before it is
    /// handed out, as its address and its server task.
    async fn serve(behaviour: Subscribe) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let router =
            Server::builder().add_service(TokenSpeedSchedulerServer::new(TestScheduler(behaviour)));
        serve_on_loopback(router).await
    }

    /// A gRPC worker at `addr` and its backend client.
    async fn connected(addr: SocketAddr) -> (Arc<dyn Worker>, Arc<BackendClient>) {
        let worker = grpc_worker(addr);
        let client = worker
            .engine_connection()
            .await
            .expect("a backend client")
            .expect("a gRPC worker has one")
            .downcast_arc::<BackendClient>()
            .expect("the gateway's client");
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
        let wake = worker.runtime_state().map(|state| state.contact_wake());
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();

        let Subscribed::Refused(status) = subscribe_with_deadline(
            client.as_ref(),
            0,
            SubscriptionKind::Events,
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
                worker.runtime_state().unwrap().note_contact();
            })
        };
        assert!(matches!(
            subscribe_with_deadline(
                client.as_ref(),
                0,
                SubscriptionKind::Events,
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
                client.as_ref(),
                0,
                SubscriptionKind::Events,
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
        let Subscribed::Refused(status) = subscribe_with_deadline(
            client.as_ref(),
            0,
            SubscriptionKind::Events,
            None,
            &mut shutdown_rx,
            Duration::from_secs(5),
        )
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
            let (_index, sink, row) = open_row(worker.url(), None);
            let (_, id) = row.feed();
            let wake = worker.runtime_state().map(|state| state.contact_wake());
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                role: Role::Index {
                    sink: sink.as_ref(),
                    id,
                },
                model_id: "m",
                load_sink: None,
                wake: wake.as_ref(),
            };
            let mut snapshot = None;
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
            let Subscribed::Stream(stream) = subscribe_with_deadline(
                client.as_ref(),
                0,
                SubscriptionKind::Events,
                None,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
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
                KvEventMonitor::drain_stream(&sub, stream, &mut snapshot, &mut shutdown_rx),
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

    /// A loads-only stream: every record of the worker's own rank reaches the
    /// worker monitor as a pushed load (another rank's is dropped), a batch of events from a
    /// servicer that predates the request field is dropped rather than
    /// indexed, and any batch counts as the stream working.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_loads_only_stream_feeds_the_monitor_and_drops_events() {
        let record = |running: u32, rank: Option<i32>, load_only: bool| KvEventBatch {
            sequence_number: 7,
            dp_rank: rank,
            events: if load_only {
                Vec::new()
            } else {
                stored_batch(&[(1, 16)]).events
            },
            load: Some(EngineLoad {
                running_requests: running,
                load_only,
                ..Default::default()
            }),
            ..Default::default()
        };
        let script = vec![
            Ok(record(3, Some(0), true)),
            // An old servicer's event batch: its record is read, its events are not.
            Ok(record(5, Some(0), false)),
            // Another rank's record is not this worker's.
            Ok(record(9, Some(1), true)),
        ];
        let (addr, server) = serve(Subscribe::Stream(script)).await;
        let worker = rank_worker(addr, 0, 2);
        let client = worker
            .engine_connection()
            .await
            .expect("a backend client")
            .expect("a gRPC worker has one")
            .downcast_arc::<BackendClient>()
            .expect("the gateway's client");
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::clone(&worker));
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            1,
            false,
            false,
        ));
        let sink = Arc::downgrade(&monitor);
        let sink: Weak<dyn PushedLoadSink> = sink;
        let wake = worker.runtime_state().map(|state| state.contact_wake());
        let sub = WorkerSubscription {
            worker: &worker,
            worker_url: worker.url(),
            role: Role::<IndexSink>::Loads,
            model_id: "m",
            load_sink: Some(&sink),
            wake: wake.as_ref(),
        };
        let mut snapshot = None;
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let Subscribed::Stream(stream) = subscribe_with_deadline(
            client.as_ref(),
            0,
            SubscriptionKind::Loads,
            None,
            &mut shutdown_rx,
            Duration::from_secs(5),
        )
        .await
        else {
            panic!("the server answers with a stream");
        };
        let drained = tokio::time::timeout(
            Duration::from_secs(5),
            KvEventMonitor::drain_stream(&sub, stream, &mut snapshot, &mut shutdown_rx),
        )
        .await
        .expect("the drain ends");
        assert!(
            matches!(
                drained,
                Drained::Ended {
                    result: StreamResult::Ended,
                    applied: true
                }
            ),
            "a loads-only stream that delivered is the stream working"
        );
        monitor.flush_pushed_for_test();
        let snapshot = monitor.load_snapshot();
        let report = snapshot
            .get(worker.url())
            .expect("the worker's pushed report is in the snapshot");
        assert_eq!(report.version, "pushed");
        // A rank worker keeps its own rank's records (the event batch carried
        // rank 0's newer one); another rank's record is not its own.
        let ranks: Vec<(i32, i32)> = report
            .loads
            .iter()
            .map(|load| (load.dp_rank, load.num_running_reqs))
            .collect();
        assert_eq!(
            ranks,
            vec![(0, 5)],
            "rank 0's newer record from the event batch; rank 1's dropped"
        );
        server.abort();
    }

    /// A loads-only subscription asks the servicer for its records alone,
    /// from the subscription on, and creates no index for the model.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_loads_only_subscription_asks_for_records_alone_and_keeps_no_index() {
        let seen = Arc::new(Mutex::new(None));
        let (addr, server) = serve(Subscribe::RecordRequest(Arc::clone(&seen))).await;
        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
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
        assert!(request.loads_only, "{request:?}");
        assert!(request.load_records_before_first_batch, "{request:?}");
        assert_eq!(request.start_sequence_number, 0);
        assert!(
            monitor.get_indexer(worker.model_id()).is_none()
                && monitor.get_indexer(UNKNOWN_MODEL_ID).is_none(),
            "a loads-only subscription builds no index"
        );
        tokio::time::timeout(
            Duration::from_secs(5),
            monitor.on_worker_removed(worker.url()),
        )
        .await
        .expect("removal returns while the subscribe call hangs");
        server.abort();
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
            let (_index, sink, row) = open_row(worker.url(), None);
            let (_, id) = row.feed();
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                role: Role::Index {
                    sink: sink.as_ref(),
                    id,
                },
                model_id: "m",
                load_sink: None,
                wake: None,
            };
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) = subscribe_with_deadline(
                client.as_ref(),
                0,
                SubscriptionKind::Events,
                None,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
            .await
            else {
                panic!("the server answers with a stream");
            };
            worker.set_stall(Some(StallReason::Unreachable));
            let mut snapshot = None;
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                KvEventMonitor::drain_stream(&sub, stream, &mut snapshot, &mut shutdown_rx),
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
            let (indexer, sink, row) = open_row(worker.url(), worker.dp_rank());
            let worker_id = indexer.intern_worker(worker.url()).unwrap();
            let (_, id) = row.feed();
            let sub = WorkerSubscription {
                worker: &worker,
                worker_url: worker.url(),
                role: Role::Index {
                    sink: sink.as_ref(),
                    id,
                },
                model_id: "m",
                load_sink: None,
                wake: None,
            };
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) = subscribe_with_deadline(
                client.as_ref(),
                0,
                SubscriptionKind::Events,
                None,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
            .await
            else {
                panic!("the server answers with a stream");
            };
            let mut snapshot = None;
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                KvEventMonitor::drain_stream(&sub, stream, &mut snapshot, &mut shutdown_rx),
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
                sub.role.resume_sequence(),
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
        let (addr, server) = serve(Subscribe::Hang).await;

        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
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
                    worker.runtime_state().unwrap().note_contact();
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
        let seen = Arc::new(Mutex::new(None));
        let (addr, server) = serve(Subscribe::RecordRequest(Arc::clone(&seen))).await;
        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
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
    /// not end the wait early, and the failure is counted by its code. The
    /// doubling is read from the delays the subscription schedules (its
    /// retry lines), not from the wall-clock gaps between the connects,
    /// which a loaded host stretches unevenly; each gap is held to the delay
    /// slept before it, a floor load can only raise.
    #[traced_test]
    #[test]
    fn a_stream_that_fails_on_its_first_message_backs_off_and_is_counted() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let calls = Arc::new(Mutex::new(Vec::new()));
        // One thread, so the subscription task's metrics land on this
        // thread's recorder.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let addr = metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let (addr, server) = serve(Subscribe::FailFirstMessage(Arc::clone(&calls))).await;
                let worker = grpc_worker(addr);
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                // Four connects are three failed streams, each with its
                // delay slept before the next connect.
                let deadline = Instant::now() + Duration::from_secs(30);
                while calls.lock().unwrap().len() < 4 {
                    assert!(Instant::now() < deadline, "no fourth subscribe call");
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                monitor.on_worker_removed(worker.url()).await;
                server.abort();
                addr
            })
        });

        let calls = calls.lock().unwrap().clone();
        // The delay scheduled after each failed stream doubles up to the cap,
        // and each was slept: the gap before a connect is at least the delay
        // the retry line before it scheduled. A loaded host only stretches a
        // gap, so the floor reads the retry lines rather than a fixed value.
        // A backoff that restarted on every connect would schedule the
        // initial delay each time, and fail here.
        let gaps: Vec<Duration> = calls.windows(2).map(|pair| pair[1] - pair[0]).collect();
        logs_assert(|lines: &[&str]| {
            let delays = lines
                .iter()
                .filter(|line| line.contains("KV event stream error, reconnecting"))
                .map(|line| {
                    line.split("delay_ms=")
                        .nth(1)
                        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                        .and_then(|digits| digits.parse::<u64>().ok())
                        .ok_or_else(|| format!("no delay_ms on the retry line: {line}"))
                })
                .collect::<Result<Vec<u64>, String>>()?;
            if delays.len() < 3 {
                return Err(format!(
                    "{} retry lines, three or more expected",
                    delays.len()
                ));
            }
            if delays[0] != INITIAL_RECONNECT_DELAY_MS
                || !delays
                    .windows(2)
                    .all(|pair| pair[1] == (pair[0] * 2).min(MAX_RECONNECT_DELAY_MS))
            {
                return Err(format!("the scheduled delays do not double: {delays:?}"));
            }
            if let Some((gap, delay)) = gaps
                .iter()
                .zip(&delays)
                .find(|(gap, delay)| **gap < Duration::from_millis(**delay))
            {
                return Err(format!(
                    "a connect came {gap:?} after the one before it, under the {delay} ms \
                     its retry line scheduled: gaps {gaps:?}, delays {delays:?}"
                ));
            }
            Ok(())
        });

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

    /// The subscription takes the contact wake from its worker's own state:
    /// a worker heard from on another path while its stream is in backoff
    /// is retried at the floor, not after the delay. Without the wake a
    /// connect never comes sooner after the one before it than the delay
    /// slept between them, however loaded the host, so one that does is the
    /// wake's doing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_contact_with_the_worker_ends_its_subscriptions_backoff_early() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (addr, server) = serve(Subscribe::FailFirstMessage(Arc::clone(&calls))).await;
        let worker = grpc_worker(addr);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        // The worker is heard from on another path until six connects are in.
        let deadline = Instant::now() + Duration::from_secs(10);
        while calls.lock().unwrap().len() < 6 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
            worker.runtime_state().unwrap().note_contact();
        }
        monitor.on_worker_removed(worker.url()).await;
        server.abort();

        let calls = calls.lock().unwrap().clone();
        let gaps: Vec<Duration> = calls.windows(2).map(|pair| pair[1] - pair[0]).collect();
        // The delay scheduled before each connect after the first, in the
        // loop's own order: no stream here delivers a batch, so none resets.
        let mut backoff = Backoff::new();
        let delays: Vec<Duration> = std::iter::repeat_with(|| {
            let delay = backoff.delay();
            backoff.grow();
            delay
        })
        .take(gaps.len())
        .collect();
        assert!(
            gaps.iter().zip(&delays).any(|(gap, delay)| gap < delay),
            "no connect came before its scheduled delay: gaps {gaps:?}, delays {delays:?}"
        );
    }

    /// A backend that serves no KV events (an engine whose publisher is off)
    /// leaves one WARN line, which names the model, carries the servicer's
    /// own reason and says that the model reaches the approximate tree only
    /// when no worker of it publishes; the worker carries a gauge from the
    /// answer until it leaves.
    #[traced_test]
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
                let (addr, server) = serve(Subscribe::Unimplemented).await;

                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://{addr}"))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::TokenSpeed)
                        .model(ModelCard::new("hybrid-m"))
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                );
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
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
                // The WARN line itself names the model, carries the servicer's
                // reason and says when the model reaches the tree; the
                // subscription's start line names the model too, so the
                // checks are scoped to the refusal's line.
                logs_assert(|lines: &[&str]| {
                    let line = lines
                        .iter()
                        .find(|line| line.contains("does not implement SubscribeKvEvents"))
                        .ok_or_else(|| "no refusal line was logged".to_string())?;
                    for wanted in [
                        "WARN",
                        "hybrid-m",
                        "KV cache events not enabled",
                        "approximate prefix tree only when no worker of it publishes events",
                    ] {
                        if !line.contains(wanted) {
                            return Err(format!("the refusal line lacks {wanted:?}: {line}"));
                        }
                    }
                    Ok(())
                });

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

    /// A worker whose connection is not the gateway's client carries no
    /// event stream: its subscription is refused as unimplemented, the
    /// worker's gauge rises, and no second connect is asked for.
    #[test]
    fn a_connection_that_is_not_the_gateways_client_carries_no_event_stream() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        use crate::worker::{
            connection::engine::EngineConnector, kv::event_monitor::Slot,
            test_support::CountingConnector,
        };

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
                let connector = Arc::new(CountingConnector::default());
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new("grpc://no-engine:9000")
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .model(ModelCard::new("m"))
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .engine_connector(Arc::clone(&connector) as Arc<dyn EngineConnector>)
                        .build(),
                );
                let prefix = format!("smg_kv_events_unavailable{{worker=\"{}\"}}", worker.url());
                let gauge = || {
                    handle
                        .render()
                        .lines()
                        .find(|line| line.starts_with(&prefix))
                        .and_then(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
                };
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&worker, SubscriptionKind::Events)
                    .await;
                for _ in 0..500 {
                    if gauge() == Some(1.0) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                assert_eq!(gauge(), Some(1.0), "the refusal raises the gauge");
                assert_eq!(connector.connects(), 1, "no second connect");
                // The connect count cannot tell a second subscription over the
                // kept handle from an end: the loop's own task says it ended.
                let mut ended = false;
                for _ in 0..500 {
                    ended = match monitor.worker_handles.lock().await.get(worker.url()) {
                        Some(Slot::Active(sub)) => sub.handle.is_finished(),
                        Some(Slot::Removing(_)) => false,
                        None => true,
                    };
                    if ended {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                assert!(ended, "the refusal ends the loop");
                monitor.on_worker_removed(worker.url()).await;
            });
        });
    }

    #[tokio::test]
    async fn a_removed_workers_blocks_leave_the_index_off_the_runtime() {
        let (indexer, sink, row) = open_row("http://w1:8000", None);
        let (_, id) = row.feed();
        let batch = stored_batch(&[(1, 16)]);
        assert_eq!(
            sink.admit(id, batch_rank(&batch), batch_facts(&batch), &batch),
            BatchOutcome::Applied
        );
        assert_ne!(indexer.current_size(), 0, "the stored block is indexed");

        KvEventMonitor::remove_from_sink(sink, id, "grpc://w1:9000").await;

        assert_eq!(indexer.current_size(), 0);
    }

    /// A subscription of `worker` that feeds `sink` under id 7.
    fn said_subscription<'a>(
        worker: &'a Arc<dyn Worker>,
        sink: &'a Said,
    ) -> WorkerSubscription<'a, Said> {
        WorkerSubscription {
            worker,
            worker_url: worker.url(),
            role: Role::Index { sink, id: 7 },
            model_id: "m",
            load_sink: None,
            wake: None,
        }
    }

    /// The sequence a round subscribes from is the sink's cursor for rank 0:
    /// the collector keeps none of its own, so a reconnect resumes wherever
    /// the sink stands.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_round_subscribes_from_the_sinks_cursor() {
        let seen = Arc::new(Mutex::new(None));
        let (addr, server) = serve(Subscribe::RecordRequest(Arc::clone(&seen))).await;
        let worker = grpc_worker(addr);
        let sink = Said {
            applied: 41,
            ..Said::default()
        };
        let sub = said_subscription(&worker, &sink);
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
        let mut backoff = Backoff::new();
        let request = async {
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
            // The scheduler never answers: the worker's removal ends the round.
            let _ = shutdown_tx.send(());
            request
        };
        let (round, request) = tokio::join!(
            KvEventMonitor::subscription_round(&sub, &mut backoff, &mut shutdown_rx),
            request,
        );
        assert!(matches!(round, Round::Shutdown | Round::Backoff));
        assert_eq!(request.start_sequence_number, 41);
        assert_eq!(sink.calls(), ["cursor 7 rank 0"]);
        server.abort();
    }

    /// What crosses the port from the collector's side, call by call, held
    /// against a sink that only keeps what it was told: a round reads the
    /// sink's cursor first; a connection, each batch with its rank and its
    /// sequence, a snapshot's start, a snapshot the stream ended inside and
    /// a server's lost history are said to the sink; and the worker's
    /// leaving is said once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_collector_says_what_its_stream_brought_to_the_sink() {
        let ranked = KvEventBatch {
            sequence_number: 5,
            dp_rank: Some(1),
            ..stored_batch(&[(1, 16)])
        };
        let chunk = KvEventBatch {
            sequence_number: 6,
            snapshot: Some(KvSnapshotChunk {
                index: 0,
                count: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let scripts = [
            (
                vec![Ok(ranked), Ok(chunk)],
                vec![
                    "reconnected 7",
                    "admit 7 rank 1 seq 5",
                    "resync 7 None snapshot",
                    "admit 7 rank 0 seq 6 chunk",
                    "resync 7 None snapshot_abandoned",
                ],
                false,
            ),
            (
                vec![Err(Status::data_loss("the subscriber fell behind"))],
                vec!["reconnected 7", "resync 7 None data_loss"],
                true,
            ),
        ];
        for (script, expected, again) in scripts {
            let (addr, server) = serve(Subscribe::Stream(script)).await;
            let worker = grpc_worker(addr);
            let sink = Said::default();
            let sub = said_subscription(&worker, &sink);
            let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
            let round =
                KvEventMonitor::subscription_round(&sub, &mut Backoff::new(), &mut shutdown_rx)
                    .await;
            if again {
                assert!(matches!(round, Round::Again), "{expected:?}");
            } else {
                assert!(matches!(round, Round::Backoff), "{expected:?}");
            }
            let calls = sink.calls();
            assert_eq!(calls[0], "cursor 7 rank 0");
            // A log line of the round reads the cursor again where a
            // subscriber is installed: the reads are left out here.
            let said: Vec<String> = calls
                .into_iter()
                .filter(|call| !call.starts_with("cursor"))
                .collect();
            assert_eq!(said, expected);
            server.abort();
        }

        let sink = Arc::new(Said::default());
        KvEventMonitor::remove_from_sink(Arc::clone(&sink), 7, "grpc://w1:9000").await;
        assert_eq!(sink.calls(), ["removed 7"]);
    }

    /// A snapshot's start is said once, ahead of its first chunk: the chunks
    /// that follow in its count say nothing more, and a count the stream
    /// filled is no abandoned snapshot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_snapshots_start_is_said_once_ahead_of_its_first_chunk() {
        let chunk = |sequence_number, index| KvEventBatch {
            sequence_number,
            snapshot: Some(KvSnapshotChunk {
                index,
                count: 3,
                ..Default::default()
            }),
            ..Default::default()
        };
        let live = KvEventBatch {
            sequence_number: 9,
            ..stored_batch(&[(1, 16)])
        };
        let script = vec![Ok(chunk(6, 0)), Ok(chunk(7, 1)), Ok(chunk(8, 2)), Ok(live)];
        let (addr, server) = serve(Subscribe::Stream(script)).await;
        let worker = grpc_worker(addr);
        let sink = Said::default();
        let sub = said_subscription(&worker, &sink);
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let round =
            KvEventMonitor::subscription_round(&sub, &mut Backoff::new(), &mut shutdown_rx).await;
        assert!(matches!(round, Round::Backoff));
        let said: Vec<String> = sink
            .calls()
            .into_iter()
            .filter(|call| !call.starts_with("cursor"))
            .collect();
        assert_eq!(
            said,
            [
                "reconnected 7",
                "resync 7 None snapshot",
                "admit 7 rank 0 seq 6 chunk",
                "admit 7 rank 0 seq 7 chunk",
                "admit 7 rank 0 seq 8 chunk",
                "admit 7 rank 0 seq 9",
            ]
        );
        server.abort();
    }
}
