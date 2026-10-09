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
    KvEventMonitor,
};
use crate::{
    observability::metrics::Metrics,
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

/// Positional-index cleanup is CPU-bound and can touch many blocks. Keep it
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

impl KvEventMonitor {
    /// Learn `block_size` from the first `KvBlock` in a stored event.
    ///
    /// Called once per model when the first stored event arrives, providing
    /// ground truth from the backend. `CacheAwarePolicy` uses this to chunk
    /// request tokens into blocks for overlap scoring.
    ///
    /// Overwrites any provisional value seeded from `WorkerSpec` since the
    /// event stream reflects the backend's actual page size.
    fn learn_block_size(
        block_sizes: &DashMap<String, usize>,
        model_id: &str,
        learned: &mut bool,
        batch: &KvEventBatch,
    ) {
        if *learned {
            return;
        }
        for event in &batch.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if let Some(block) = stored.blocks.first() {
                    if block.block_size > 0 {
                        let bs = block.block_size as usize;
                        block_sizes.insert(model_id.to_string(), bs);
                        info!(
                            model_id = %model_id,
                            block_size = bs,
                            "Learned block_size from KV event"
                        );
                        *learned = true;
                        return;
                    }
                }
            }
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
            error!(worker_id, "Positional-index cleanup semaphore closed");
            return;
        };
        let WorkerIndexState {
            blocks, counters, ..
        } = worker_blocks;
        if counters != WorkerIndexCounters::default() {
            debug!(
                worker_id,
                ?counters,
                "KV events the positional index did not take as is"
            );
        }
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            indexer.remove_worker(worker_id, blocks);
        })
        .await;

        if let Err(error) = result {
            error!(worker_id, %error, "Positional-index worker cleanup task failed");
        }
        Metrics::set_kv_index_blocks(worker_url, 0);
    }

    /// Main subscription loop for a single worker.
    ///
    /// Owns the worker's index state and takes its blocks out of the index on
    /// exit. Exits when `shutdown_rx` fires or the backend returns
    /// `Unimplemented`.
    pub(super) async fn subscription_loop(
        worker: Arc<dyn Worker>,
        worker_url: String,
        indexer: Arc<KvIndex>,
        block_sizes: Arc<DashMap<String, usize>>,
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
        let mut state = WorkerStreamState::default();
        // A contact with the worker (a poll answered, a probe passed) ends the
        // reconnect backoff early: a worker that is back gets its stream back
        // at once instead of after the remaining delay.
        let wake = worker.contact_wake();
        let mut reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
        let mut block_size_learned = false;

        /// Sleep with shutdown check. Returns `true` if shutdown was signaled.
        /// A contact with the worker ends the sleep early, but never before
        /// `RECONNECT_FLOOR`, and only one made after the sleep began: the
        /// connect of a stream that then failed does not count.
        macro_rules! sleep_or_shutdown {
            ($delay:expr, $rx:expr) => {{
                let delay: Duration = $delay;
                drain_contact(wake.as_ref());
                tokio::select! {
                    _ = tokio::time::sleep(delay) => false,
                    () = async {
                        tokio::time::sleep(delay.min(RECONNECT_FLOOR)).await;
                        woken(wake.as_ref()).await;
                    } => false,
                    _ = &mut *$rx => true,
                }
            }};
        }

        loop {
            let backend_client = match worker.get_backend_client().await {
                Ok(Some(client)) => client,
                Ok(None) => {
                    // HTTP workers are filtered in on_worker_added, so this should
                    // be unreachable. Retry defensively rather than exiting and
                    // leaving a stale entry in worker_handles.
                    warn!(
                        worker_url = %worker_url,
                        delay_ms = reconnect_delay_ms,
                        "Worker has no backend client yet, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to get backend client, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let start_seq = state.resume_sequence();
            // A live server answers a subscription with headers at once; one
            // that does not within the deadline is unreachable (a half-open
            // connection behind a partition), and must not hang the loop.
            let subscribed = tokio::select! {
                attempt = tokio::time::timeout(
                    SUBSCRIBE_DEADLINE,
                    backend_client.subscribe_kv_events(start_seq),
                ) => attempt.unwrap_or_else(|_| {
                    Err(tonic::Status::unavailable(format!(
                        "SubscribeKvEvents did not answer within {SUBSCRIBE_DEADLINE:?}"
                    )))
                }),
                // The worker was heard from on another path (a poll, a probe)
                // and this call still hangs: a fresh one will land.
                () = async {
                    woken(wake.as_ref()).await;
                    tokio::time::sleep(SUBSCRIBE_RETRY_GRACE).await;
                } => continue,
                // The worker left while the call was pending. Without this arm
                // a call that never answers, retried at every contact, would
                // keep the task alive past the worker's removal.
                _ = &mut shutdown_rx => {
                    Self::remove_indexer_worker(
                        Arc::clone(&indexer),
                        worker_id,
                        &worker_url,
                        state.index,
                    )
                    .await;
                    return;
                }
            };
            let stream = match subscribed {
                Ok(stream) => {
                    info!(
                        worker_url = %worker_url,
                        start_seq,
                        "KV event stream connected"
                    );
                    Metrics::record_kv_event_subscription(&worker_url);
                    // The backoff is not reset here: a connect is no proof of
                    // a working stream (one that fails on its first message
                    // would be retried at the initial delay forever); it is
                    // reset once the stream has delivered a batch.
                    state.reconnected();
                    liveness::on_contact(&worker);
                    stream
                }
                Err(e) => {
                    // If the backend doesn't implement SubscribeKvEvents (e.g. vLLM),
                    // stop retrying — this RPC will never succeed.
                    if e.code() == tonic::Code::Unimplemented {
                        warn!(
                            worker_url = %worker_url,
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this worker; \
                             cache-aware routing sees nothing of its cache"
                        );
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        Metrics::set_kv_events_unavailable(&worker_url, true);
                        return;
                    }
                    if e.code() == tonic::Code::OutOfRange {
                        warn!(
                            worker_url = %worker_url,
                            start_seq,
                            "KV event replay cursor expired; clearing worker state and requesting a current snapshot"
                        );
                        Self::reset_worker(
                            &indexer,
                            worker_id,
                            &mut state,
                            &worker_url,
                            ResyncReason::OutOfRange,
                        );
                        reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                        continue;
                    }
                    if liveness::is_transport_failure(e.code()) {
                        liveness::on_contact_failed(&worker, "kv subscribe");
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to subscribe to KV events, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let mut applied = false;
            let on_batch = |batch: &KvEventBatch| {
                applied = true;
                liveness::on_contact(&worker);
                Self::learn_block_size(&block_sizes, &model_id, &mut block_size_learned, batch);
            };
            // The load record on a batch is a poll of this worker, received now.
            let on_load = |batch: &KvEventBatch, load: &EngineLoad| {
                liveness::on_contact(&worker);
                if let Some(monitor) = load_sink.as_ref().and_then(Weak::upgrade) {
                    monitor.apply_pushed_load(
                        &worker,
                        batch.dp_rank.unwrap_or(0),
                        load,
                        Instant::now(),
                    );
                }
            };
            let stream_result = tokio::select! {
                result = Self::process_stream(
                    stream, &worker_url, worker_id, &indexer, &mut state, on_batch, on_load,
                ) => result,
                _ = &mut shutdown_rx => {
                    Self::remove_indexer_worker(
                        Arc::clone(&indexer),
                        worker_id,
                        &worker_url,
                        state.index,
                    )
                    .await;
                    return;
                }
            };

            if applied {
                // The stream worked: its failure starts the backoff over.
                reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
            }
            if state.abandon_snapshot() {
                warn!(
                    worker_url = %worker_url,
                    "KV event stream ended during a relay snapshot; the next subscription \
                     starts over from zero"
                );
            }
            match stream_result {
                StreamResult::Ended => {
                    info!(
                        worker_url = %worker_url,
                        resume_from = state.resume_sequence(),
                        delay_ms = reconnect_delay_ms,
                        "KV event stream ended, reconnecting"
                    );
                    // Backoff to avoid tight reconnect loop if server keeps
                    // closing the stream cleanly (e.g., rolling connections).
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
                StreamResult::Error(e) => {
                    Metrics::record_kv_event_stream_error(&worker_url, error_class(e.code()));
                    if e.code() == tonic::Code::DataLoss {
                        warn!(
                            worker_url = %worker_url,
                            error = %e,
                            resume_from = state.resume_sequence(),
                            "KV event subscriber fell behind; clearing worker state and requesting a current snapshot"
                        );
                        Self::reset_worker(
                            &indexer,
                            worker_id,
                            &mut state,
                            &worker_url,
                            ResyncReason::DataLoss,
                        );
                        reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                        continue;
                    }
                    if liveness::is_transport_failure(e.code()) {
                        liveness::on_contact_failed(&worker, "kv stream");
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        resume_from = state.resume_sequence(),
                        delay_ms = reconnect_delay_ms,
                        "KV event stream error, reconnecting"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        Self::remove_indexer_worker(
                            Arc::clone(&indexer),
                            worker_id,
                            &worker_url,
                            state.index,
                        )
                        .await;
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
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
                }
            }
        }
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
    use crate::worker::BasicWorkerBuilder;

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
            _: Request<common::SubscribeKvEventsRequest>,
        ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
            match &self.0 {
                Subscribe::Hang => std::future::pending().await,
                Subscribe::Unimplemented => {
                    Err(Status::unimplemented("KV cache events not enabled"))
                }
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
        let monitor = KvEventMonitor::new(None);
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
                let monitor = KvEventMonitor::new(None);
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
                let monitor = KvEventMonitor::new(None);
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
        let indexer = Arc::new(KvIndex::positional(64));
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
