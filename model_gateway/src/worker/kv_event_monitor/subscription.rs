//! The subscription task of one engine: connect, read the engine's one
//! stream, reconnect with backoff, hand each rank's batches and pushed load
//! records to the rank worker that stands for it, take rank workers in and
//! out as they register and leave, and take a leaving worker's blocks out of
//! the index.

use std::{
    collections::HashMap,
    pin::pin,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use smg_grpc_client::common_proto::{kv_cache_event, EngineLoad, KvEventBatch};
use tokio::sync::{mpsc, oneshot, watch, Mutex, Notify, Semaphore};
use tracing::{debug, error, info, warn};

use super::{
    admission::{BatchOutcome, EngineStreamState, Member},
    apply::{WorkerIndexCounters, WorkerIndexState},
    KvBlockSize, KvEventMonitor, Slot, SubscriptionKind, Subscriptions,
};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::backend_client::BackendClient,
    worker::{
        endpoint::Endpoint, kv_event_recovery::ResyncReason, kv_index_backend::KvIndex, liveness,
        monitor::WorkerMonitor, Worker,
    },
};

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Resolves when one of the engine's workers is heard from (see
/// `Worker::contact_wake`), or never for workers without a notifier.
async fn woken(wakes: &[Arc<Notify>]) {
    if wakes.is_empty() {
        std::future::pending::<()>().await;
    }
    let notified = wakes.iter().map(|wake| Box::pin(wake.notified()));
    futures::future::select_all(notified).await;
}

/// Take a contact noted before now (`notify_one` keeps one until it is
/// awaited), so that only a contact after this point ends a wait: the
/// connect of a stream that then failed was a contact, and says nothing
/// about the worker now.
fn drain_contact(wakes: &[Arc<Notify>]) {
    for wake in wakes {
        let mut notified = pin!(wake.notified());
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

/// How long the first connect of a data-parallel engine waits for its other
/// rank workers. A registration adds an engine's ranks one after another in
/// one pass, so the stream opens once for all of them; a rank that joins
/// after the stream delivered starts the engine over from zero instead.
const RANKS_JOIN_GRACE: Duration = Duration::from_secs(1);

/// Index cleanup is CPU-bound and can touch many blocks. Keep it
/// off Tokio workers and bound concurrent purges during fleet-wide drains.
pub(super) const MAX_CONCURRENT_INDEX_REMOVALS: usize = 4;
pub(super) static INDEX_REMOVAL_PERMITS: Semaphore =
    Semaphore::const_new(MAX_CONCURRENT_INDEX_REMOVALS);

/// What the monitor tells an engine's task about the workers on its stream.
pub(super) enum MemberChange {
    /// A worker registered under the engine's address joins the stream.
    Join(Arc<dyn Worker>),
    /// The worker at `url` leaves: its blocks leave the index, then the
    /// removal `id` left in its slot is lifted and `done` closes.
    Leave {
        url: String,
        id: u64,
        done: watch::Sender<()>,
    },
}

/// Result of processing a stream connection to completion.
enum StreamResult {
    /// Stream closed normally (server-side).
    Ended,
    /// Stream produced an error.
    Error(tonic::Status),
    /// Detected a gap in sequence numbers.
    GapDetected { expected: u64, received: u64 },
    /// A rank worker joined after the stream had delivered: the engine
    /// starts over from zero so the relay's history reaches its rank too.
    Rejoin,
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

/// What a round of an engine's subscription asks the loop to do next.
enum Round {
    /// Start the next round at once.
    Again,
    /// Sleep the reconnect backoff first, then grow it.
    Backoff,
    /// The engine's last worker is leaving: end the task.
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
    /// A worker of the engine was heard from on another path while the call
    /// hung; a fresh call will land.
    Contacted,
    /// The engine's last worker left while the call was pending.
    Shutdown,
}

/// How a stream was drained.
enum Drained {
    /// The stream ended as `result` says; `applied` is whether the index
    /// applied a batch of it.
    Ended { result: StreamResult, applied: bool },
    /// The engine's last worker left while the stream was open.
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

/// What an engine's stream feeds: the model's index (its KV events, with the
/// load records that ride them), or the worker monitor alone (its load
/// records, asked for with `loads_only`; a servicer that predates the field
/// streams its events too, which are dropped).
#[derive(Clone, Copy)]
enum Role<'a> {
    Index {
        indexer: &'a Arc<KvIndex>,
        block_sizes: &'a DashMap<String, KvBlockSize>,
    },
    Loads,
}

impl Role<'_> {
    fn kind(self) -> SubscriptionKind {
        match self {
            Self::Index { .. } => SubscriptionKind::Events,
            Self::Loads => SubscriptionKind::Loads,
        }
    }
}

/// What one engine's subscription task holds for its whole life, borrowed by
/// every round: the engine's address, what its stream feeds, and the
/// monitor's maps a leaving member's removal completes in, held as `Arc`s
/// so a removal runs in a task of its own while the stream goes on.
struct EngineStream<'a> {
    address: &'a str,
    role: Role<'a>,
    model_id: &'a str,
    load_sink: Option<&'a Weak<WorkerMonitor>>,
    slots: Arc<Mutex<Subscriptions>>,
    indexers: Arc<DashMap<String, Arc<KvIndex>>>,
    block_sizes: Arc<DashMap<String, KvBlockSize>>,
}

/// What the monitor hands an engine's task at its spawn.
pub(super) struct EngineTask {
    pub(super) address: String,
    pub(super) model_id: String,
    /// The model's index for an events subscription; `None` streams the
    /// engine's load records alone.
    pub(super) indexer: Option<Arc<KvIndex>>,
    pub(super) block_sizes: Arc<DashMap<String, KvBlockSize>>,
    pub(super) load_sink: Option<Weak<WorkerMonitor>>,
    /// The worker whose registration opened the subscription.
    pub(super) first: Arc<dyn Worker>,
    pub(super) changes: mpsc::UnboundedReceiver<MemberChange>,
    pub(super) shutdown_rx: oneshot::Receiver<()>,
    pub(super) slots: Arc<Mutex<Subscriptions>>,
    pub(super) indexers: Arc<DashMap<String, Arc<KvIndex>>>,
}

/// The contact notifiers of the engine's workers, for a wait that a contact
/// on another path ends early.
fn wakes_of(state: &EngineStreamState) -> Vec<Arc<Notify>> {
    state
        .members
        .iter()
        .filter_map(|member| member.worker.contact_wake())
        .collect()
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

    /// Take a worker's blocks out of the index when it leaves its engine's
    /// stream, on the blocking pool and under a fleet-wide bound, and log
    /// what its stream carried that the index did not take as is.
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

    /// A worker joins the engine's stream: interned in the model's index on
    /// an events subscription (a worker the index has no id left for joins
    /// without one and feeds nothing). A member that joins after the stream
    /// delivered has missed its rank's history: the next round starts the
    /// engine over from zero.
    fn member_joined(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        worker: Arc<dyn Worker>,
    ) {
        let worker_id = match sub.role {
            Role::Index { indexer, .. } => match indexer.intern_worker(worker.url()) {
                Ok(worker_id) => Some(worker_id),
                Err(e) => {
                    error!(
                        worker_url = %Endpoint::redacted_str(worker.url()),
                        error = %e,
                        "Failed to intern worker; KV events from this worker will \
                         not feed cache-aware routing"
                    );
                    Metrics::record_kv_event_subscription_failure(worker.url(), "intern_failed");
                    None
                }
            },
            Role::Loads => None,
        };
        let late = worker_id.is_some() && state.streamed();
        if late {
            state.rejoin = true;
            state
                .joined_late
                .push(worker.dp_rank().and_then(|rank| i32::try_from(rank).ok()));
        }
        info!(
            worker_url = %Endpoint::redacted_str(worker.url()),
            engine = %Endpoint::redacted_str(sub.address),
            rank = worker.dp_rank(),
            members = state.members.len() + 1,
            late,
            "Worker joins its engine's KV event subscription"
        );
        state.members.push(Member::new(worker, worker_id));
    }

    /// The worker at `url` leaves the engine's stream: its blocks leave the
    /// index, the removal `id` it left in its slot is lifted (with the
    /// model's index when it was the model's last events member), and `done`
    /// closes last, so the remover and an add waiting on it find the slot
    /// free. A member's cleanup runs in a task of its own: the blocks leave
    /// under the fleet-wide removal permits and on the blocking pool, and
    /// the stream is polled for the engine's other members meanwhile. A
    /// worker that is not on the stream (the task is ending and took its
    /// members out itself) completes here.
    async fn member_left(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        url: &str,
        id: u64,
        done: watch::Sender<()>,
    ) {
        let Some(member) = state.leave(url) else {
            // The task took this member out itself: the remover's write,
            // made under the caller's handle on the removed worker.
            let mut subs = sub.slots.lock().await;
            Self::complete_member_removal_locked(
                &mut subs,
                &sub.indexers,
                &sub.block_sizes,
                url,
                id,
                sub.model_id,
                sub.role.kind(),
            );
            drop(subs);
            drop(done);
            return;
        };
        info!(
            worker_url = %Endpoint::redacted_str(url),
            engine = %Endpoint::redacted_str(sub.address),
            members = state.members.len(),
            "Worker leaves its engine's KV event subscription"
        );
        let indexed = match (sub.role, member.worker_id) {
            (Role::Index { indexer, .. }, Some(worker_id)) => {
                Some((Arc::clone(indexer), worker_id))
            }
            _ => None,
        };
        // The member's handle on its worker rides through the cleanup's last
        // write: the series store forgets a removed address once no handle on
        // its workers is left, and a write by name alone would recreate it.
        let Member { index, worker, .. } = member;
        let slots = Arc::clone(&sub.slots);
        let indexers = Arc::clone(&sub.indexers);
        let block_sizes = Arc::clone(&sub.block_sizes);
        let (url, model_id, kind) = (url.to_owned(), sub.model_id.to_owned(), sub.role.kind());
        #[expect(
            clippy::disallowed_methods,
            reason = "a leaving worker's index cleanup: the remover and a same-URL add wait on \
                      `done`, and at a shutdown the index goes with the process"
        )]
        tokio::spawn(async move {
            if let Some((indexer, worker_id)) = indexed {
                Self::remove_indexer_worker(indexer, worker_id, &url, index).await;
            }
            Self::complete_member_removal(
                &slots,
                &indexers,
                &block_sizes,
                &worker,
                id,
                &model_id,
                kind,
            )
            .await;
            // The cleanup's last write is made: the worker may go.
            drop(worker);
            drop(done);
        });
    }

    /// Apply one membership change to the engine's state.
    async fn apply_change(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        change: MemberChange,
    ) {
        match change {
            MemberChange::Join(worker) => Self::member_joined(sub, state, worker),
            MemberChange::Leave { url, id, done } => {
                Self::member_left(sub, state, &url, id, done).await;
            }
        }
    }

    /// Main subscription loop of one engine.
    ///
    /// With an index (an events subscription) it owns every member's index
    /// state and takes a member's blocks out of the index when it leaves;
    /// without one it streams the engine's load records alone. Members join
    /// and leave through `changes` at every point of a round without the
    /// stream restarting for them. Exits when `shutdown_rx` fires (the
    /// engine's last worker left, or the monitor stops) or the backend
    /// returns `Unimplemented`. Each round ([`Self::subscription_round`])
    /// gets the backend client, subscribes under the deadline and drains one
    /// stream, then says what comes next; this loop is the one place that
    /// sleeps the reconnect backoff and the one place that ends the task.
    pub(super) async fn subscription_loop(task: EngineTask) {
        let EngineTask {
            address,
            model_id,
            indexer,
            block_sizes,
            load_sink,
            first,
            mut changes,
            mut shutdown_rx,
            slots,
            indexers,
        } = task;
        let role = match &indexer {
            Some(indexer) => Role::Index {
                indexer,
                block_sizes: &block_sizes,
            },
            None => Role::Loads,
        };
        let sub = EngineStream {
            address: &address,
            role,
            model_id: &model_id,
            load_sink: load_sink.as_ref(),
            slots,
            indexers,
            block_sizes: Arc::clone(&block_sizes),
        };
        // The first connect of a data-parallel engine waits for its other
        // ranks: a registration adds them one after another, and one stream
        // opened for all of them is the point of the engine's subscription.
        let expected = first
            .dp_size()
            .filter(|_| first.dp_rank().is_some())
            .unwrap_or(1);
        let mut state = EngineStreamState::new(&address, Vec::new());
        Self::member_joined(&sub, &mut state, first);
        // A shutdown taken inside the grace spends the receiver (a oneshot
        // polled again after it fired panics), so the end is carried past
        // the rounds to the exit path below.
        let mut ended = false;
        if expected > 1 {
            let mut grace = pin!(tokio::time::sleep(RANKS_JOIN_GRACE));
            while state.members.len() < expected {
                tokio::select! {
                    biased;
                    _ = &mut shutdown_rx => {
                        ended = true;
                        break;
                    }
                    change = changes.recv() => match change {
                        Some(change) => Self::apply_change(&sub, &mut state, change).await,
                        None => {
                            ended = true;
                            break;
                        }
                    },
                    () = &mut grace => break,
                }
            }
            // Nothing was streamed yet: the members present start together.
            state.rejoin = false;
        }
        let mut backoff = Backoff::new();

        let unsupported = loop {
            if ended {
                break false;
            }
            // Changes that landed between rounds, before the next connect.
            while let Ok(change) = changes.try_recv() {
                Self::apply_change(&sub, &mut state, change).await;
            }
            if shutdown_rx.try_recv().is_ok() {
                break false;
            }
            if state.members.is_empty() {
                // Every worker left; the monitor ends the task, or another
                // worker of the engine joins meanwhile.
                tokio::select! {
                    biased;
                    _ = &mut shutdown_rx => break false,
                    change = changes.recv() => match change {
                        Some(change) => {
                            Self::apply_change(&sub, &mut state, change).await;
                            continue;
                        }
                        None => break false,
                    },
                }
            }
            match Self::subscription_round(
                &sub,
                &mut state,
                &mut backoff,
                &mut changes,
                &mut shutdown_rx,
            )
            .await
            {
                Round::Again => {}
                Round::Backoff => {
                    if Self::sleep_or_shutdown(
                        &sub,
                        &mut state,
                        backoff.delay(),
                        &mut changes,
                        &mut shutdown_rx,
                    )
                    .await
                    {
                        break false;
                    }
                    backoff.grow();
                }
                Round::Shutdown => break false,
                Round::Unsupported => break true,
            }
        };
        // No change is taken from here on: the channel closes first, so a
        // removal that follows completes on the monitor's side (its send
        // fails) and an add waits for this task's end and starts the next.
        // What was queued before the close is taken below: the members'
        // blocks leave the index with the task, and a removal that was still
        // queued completes here.
        changes.close();
        let mut leaves: HashMap<String, (u64, watch::Sender<()>)> = HashMap::new();
        while let Ok(change) = changes.try_recv() {
            match change {
                MemberChange::Leave { url, id, done } => {
                    leaves.insert(url, (id, done));
                }
                MemberChange::Join(worker) => {
                    debug!(worker_url = %Endpoint::redacted_str(worker.url()), "KV event subscription of the engine is ending; the worker joins no stream");
                }
            }
        }
        // The gauge is the index's: a loads-only subscription a backend
        // refuses leaves the worker on the poll, which is the designed
        // fallback. Raised under the slot lock for the members still on
        // their slots, so a removal that completed on the monitor's side
        // meanwhile (which lowered it) is not raised again.
        if unsupported && indexer.is_some() {
            let subs = sub.slots.lock().await;
            for member in &state.members {
                if leaves.contains_key(&member.url) {
                    continue;
                }
                if matches!(subs.workers.get(&member.url), Some(Slot::Member { address, .. }) if address == sub.address)
                {
                    Metrics::set_kv_events_unavailable(&member.url, true);
                }
            }
        }
        for member in std::mem::take(&mut state.members) {
            if let (Role::Index { indexer, .. }, Some(worker_id)) = (role, member.worker_id) {
                Self::remove_indexer_worker(
                    Arc::clone(indexer),
                    worker_id,
                    &member.url,
                    member.index,
                )
                .await;
            }
            if let Some((id, done)) = leaves.remove(&member.url) {
                Self::member_left(&sub, &mut state, &member.url, id, done).await;
            }
        }
        for (url, (id, done)) in leaves {
            Self::member_left(&sub, &mut state, &url, id, done).await;
        }
    }

    /// Sleep `delay`, unless the engine's last worker leaves first: `true`
    /// when it did. A contact with one of the engine's workers ends the
    /// sleep early, but never before `RECONNECT_FLOOR`, and only one made
    /// after the sleep began: the connect of a stream that then failed does
    /// not count. A worker joining or leaving meanwhile is taken in at once;
    /// a join that wants the engine started over ends the sleep.
    async fn sleep_or_shutdown(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        delay: Duration,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> bool {
        let mut sleep = pin!(tokio::time::sleep(delay));
        let mut floor = pin!(tokio::time::sleep(delay.min(RECONNECT_FLOOR)));
        let mut wakes = wakes_of(state);
        drain_contact(&wakes);
        loop {
            tokio::select! {
                biased;
                _ = &mut *shutdown_rx => return true,
                change = changes.recv() => match change {
                    Some(change) => {
                        Self::apply_change(sub, state, change).await;
                        if state.rejoin {
                            return false;
                        }
                        wakes = wakes_of(state);
                    }
                    None => return true,
                },
                () = &mut sleep => return false,
                () = async {
                    (&mut floor).await;
                    woken(&wakes).await;
                } => return false,
            }
        }
    }

    /// Subscribe to the engine's KV events from `start_seq` (`Events`), or
    /// to its load records alone (`Loads`), with `deadline` on the server's
    /// answer. A live server answers a subscription with headers at once;
    /// one that does not within the deadline is unreachable (a half-open
    /// connection behind a partition), and must not hang the loop. A contact
    /// with one of the engine's workers on another path (a poll, a probe)
    /// while the call hangs ends it after a grace period, so a fresh call
    /// lands; the engine's last removal ends it at once (without this arm a
    /// call that never answers, retried at every contact, would keep the
    /// task alive past the removal). A worker joining or leaving while the
    /// call is pending is taken in without ending it.
    async fn subscribe_with_deadline(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        backend_client: &BackendClient,
        start_seq: u64,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
        deadline: Duration,
    ) -> Subscribed {
        let kind = sub.role.kind();
        let call = async {
            match kind {
                SubscriptionKind::Events => backend_client.subscribe_kv_events(start_seq).await,
                SubscriptionKind::Loads => backend_client.subscribe_load_records().await,
            }
        };
        let mut call = pin!(tokio::time::timeout(deadline, call));
        let mut wakes = wakes_of(state);
        loop {
            tokio::select! {
                biased;
                _ = &mut *shutdown_rx => return Subscribed::Shutdown,
                change = changes.recv() => match change {
                    Some(change) => {
                        Self::apply_change(sub, state, change).await;
                        if state.rejoin {
                            // A worker joined late: the call pending here
                            // would resume past its rank's history, so it
                            // is dropped and the next round starts from zero.
                            return Subscribed::Contacted;
                        }
                        wakes = wakes_of(state);
                    }
                    None => return Subscribed::Shutdown,
                },
                attempt = &mut call => return match attempt {
                    Ok(Ok(stream)) => Subscribed::Stream(stream),
                    Ok(Err(status)) => Subscribed::Refused(status),
                    Err(_) => Subscribed::Refused(tonic::Status::unavailable(format!(
                        "SubscribeKvEvents did not answer within {deadline:?}"
                    ))),
                },
                () = async {
                    woken(&wakes).await;
                    tokio::time::sleep(SUBSCRIBE_RETRY_GRACE).await;
                } => return Subscribed::Contacted,
            }
        }
    }

    /// One round of an engine's subscription: the backend client, the
    /// subscribe call under its deadline, the stream drained to its end;
    /// returns what the loop does next.
    async fn subscription_round(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        backoff: &mut Backoff,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Round {
        let address = sub.address;
        let rejoined = state.rejoin;
        if rejoined {
            // A rank worker joined after the stream delivered: the engine
            // starts over from zero, so the relay's history or snapshot
            // reaches its rank too; the members already streaming keep
            // their state and skip the replay up to their cursors.
            if let Role::Index { indexer, .. } = sub.role {
                info!(
                    engine = %Endpoint::redacted_str(address),
                    members = state.members.len(),
                    "A worker joined the engine's KV event subscription late; resubscribing from zero for its rank"
                );
                Self::rejoin_members(indexer, state);
            }
            state.rejoin = false;
            state.joined_late.clear();
        }
        let Some(first) = state.members.first() else {
            return Round::Shutdown;
        };
        let backend_client = match first.worker.get_backend_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                // HTTP workers are filtered in on_worker_added, so this should
                // be unreachable. Retry defensively rather than exiting and
                // leaving a stale entry in the subscription map.
                warn!(
                    engine = %Endpoint::redacted_str(address),
                    delay_ms = backoff.delay_ms(),
                    "Worker has no backend client yet, retrying"
                );
                return Round::Backoff;
            }
            Err(e) => {
                warn!(
                    engine = %Endpoint::redacted_str(address),
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to get backend client, retrying"
                );
                return Round::Backoff;
            }
        };

        // From zero while a late-joined rank owes its history (this round's
        // joiners included), whether or not the last attempt delivered it.
        let start_seq = if state.starts_from_zero() {
            0
        } else {
            state.resume_sequence()
        };
        let subscribed = Self::subscribe_with_deadline(
            sub,
            state,
            &backend_client,
            start_seq,
            changes,
            shutdown_rx,
            SUBSCRIBE_DEADLINE,
        )
        .await;
        let stream = match subscribed {
            Subscribed::Stream(stream) => {
                if state.rejoin {
                    // A worker joined late while the call was pending: this
                    // stream would be drained without its rank's history,
                    // so it is dropped and the next round starts from zero.
                    drop(stream);
                    return Round::Again;
                }
                info!(
                    engine = %Endpoint::redacted_str(address),
                    start_seq,
                    members = state.members.len(),
                    "KV event stream connected"
                );
                // The backoff is not reset here: a connect is no proof of
                // a working stream (one that fails on its first message
                // would be retried at the initial delay forever); it is
                // reset once the stream has delivered a batch.
                state.reconnected();
                for member in &state.members {
                    Metrics::record_kv_event_subscription(&member.url);
                    liveness::on_contact(&member.worker);
                }
                stream
            }
            Subscribed::Contacted => return Round::Again,
            Subscribed::Shutdown => return Round::Shutdown,
            Subscribed::Refused(e) => {
                // If the backend doesn't implement SubscribeKvEvents (some engines don't),
                // stop retrying — this RPC will never succeed.
                if e.code() == tonic::Code::Unimplemented {
                    match sub.role {
                        // The one line an operator gets for an engine whose
                        // cache the cache-aware policy cannot see: it names
                        // the model and carries the servicer's own reason
                        // (the publisher is off, or the engine cannot serve
                        // this model's cache groups). The model itself stays
                        // on its event index while any other worker of it
                        // publishes; only when none does is it routed on the
                        // approximate prefix tree.
                        Role::Index { .. } => warn!(
                            engine = %Endpoint::redacted_str(address),
                            model_id = sub.model_id,
                            reason = e.message(),
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this engine; \
                             cache-aware routing sees nothing of this engine's \
                             cache, and the model falls back to the approximate \
                             prefix tree only when no worker of it publishes \
                             events"
                        ),
                        Role::Loads => info!(
                            engine = %Endpoint::redacted_str(address),
                            "Backend does not implement SubscribeKvEvents, \
                             disabling the load stream for this engine; its load \
                             stays polled"
                        ),
                    }
                    return Round::Unsupported;
                }
                if e.code() == tonic::Code::OutOfRange {
                    warn!(
                        engine = %Endpoint::redacted_str(address),
                        start_seq,
                        "KV event replay cursor expired; clearing the engine's index state and requesting a current snapshot"
                    );
                    if let Role::Index { indexer, .. } = sub.role {
                        Self::reset_members(indexer, state, ResyncReason::OutOfRange);
                    }
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    for member in &state.members {
                        liveness::on_contact_failed(&member.worker, "kv subscribe");
                    }
                }
                warn!(
                    engine = %Endpoint::redacted_str(address),
                    error = %e,
                    delay_ms = backoff.delay_ms(),
                    "Failed to subscribe to KV events, retrying"
                );
                return Round::Backoff;
            }
        };

        let (result, applied) =
            match Self::drain_stream(sub, stream, state, changes, shutdown_rx).await {
                Drained::Ended { result, applied } => (result, applied),
                Drained::Shutdown => return Round::Shutdown,
            };
        if applied {
            // The stream worked: its failure starts the backoff over.
            backoff.reset();
        }
        if state.abandon_snapshot() {
            warn!(
                engine = %Endpoint::redacted_str(address),
                "KV event stream ended during a relay snapshot; the next subscription \
                 starts over from zero"
            );
        }
        match result {
            StreamResult::Ended => {
                info!(
                    engine = %Endpoint::redacted_str(address),
                    resume_from = state.resume_sequence(),
                    delay_ms = backoff.delay_ms(),
                    "KV event stream ended, reconnecting"
                );
                // Backoff to avoid tight reconnect loop if server keeps
                // closing the stream cleanly (e.g., rolling connections).
                Round::Backoff
            }
            StreamResult::Error(e) => {
                for member in &state.members {
                    Metrics::record_kv_event_stream_error(&member.url, error_class(e.code()));
                }
                if e.code() == tonic::Code::DataLoss {
                    warn!(
                        engine = %Endpoint::redacted_str(address),
                        error = %e,
                        resume_from = state.resume_sequence(),
                        "KV event subscriber fell behind; clearing the engine's index state and requesting a current snapshot"
                    );
                    if let Role::Index { indexer, .. } = sub.role {
                        Self::reset_members(indexer, state, ResyncReason::DataLoss);
                    }
                    backoff.reset();
                    return Round::Again;
                }
                if liveness::is_connection_code(e.code()) {
                    for member in &state.members {
                        liveness::on_contact_failed(&member.worker, "kv stream");
                    }
                }
                warn!(
                    engine = %Endpoint::redacted_str(address),
                    error = %e,
                    resume_from = state.resume_sequence(),
                    delay_ms = backoff.delay_ms(),
                    "KV event stream error, reconnecting"
                );
                Round::Backoff
            }
            StreamResult::GapDetected { expected, received } => {
                warn!(
                    engine = %Endpoint::redacted_str(address),
                    expected,
                    received,
                    "Sequence gap detected, reconnecting for replay from seq {expected}"
                );
                // No backoff: gap replay is a normal recovery path, and the
                // rank state asks for it once; if the server skips ahead
                // again the gap is settled instead of retried.
                Round::Again
            }
            // The reset and the subscribe from zero are the next round's.
            StreamResult::Rejoin => Round::Again,
        }
    }

    /// The load record on a batch is a poll of the rank worker it names,
    /// received now: forwarded to the monitor for every member whose rank it
    /// is (an unranked record is rank 0's; a whole-engine member takes them
    /// all).
    fn forward_record(
        sub: &EngineStream<'_>,
        state: &EngineStreamState,
        batch: &KvEventBatch,
        load: &EngineLoad,
    ) {
        let rank = batch.dp_rank.unwrap_or(0);
        if rank < 0 {
            return;
        }
        let monitor = sub.load_sink.and_then(Weak::upgrade);
        for member in state.members.iter().filter(|m| m.takes_record(rank)) {
            liveness::on_contact(&member.worker);
            if let Some(monitor) = &monitor {
                monitor.apply_pushed_load(&member.worker, rank, load, Instant::now());
            }
        }
    }

    /// Drain `stream` to its end: on an events subscription each batch is
    /// applied to the index state of the member whose rank it is (a batch
    /// the index applied is the stream working) and each batch's load record
    /// forwarded; on a loads-only one every batch is the stream working, its
    /// record forwarded and its events, if any, dropped. The engine's last
    /// removal ends the stream at once; a member joining or leaving is taken
    /// in between batches.
    async fn drain_stream(
        sub: &EngineStream<'_>,
        stream: tonic::Streaming<KvEventBatch>,
        state: &mut EngineStreamState,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Drained {
        let mut applied = false;
        let result = match sub.role {
            Role::Index {
                indexer,
                block_sizes,
            } => {
                let on_batch = |batch: &KvEventBatch| {
                    #[cfg(test)]
                    tests::panic_if_armed(sub.address);
                    applied = true;
                    Self::learn_block_size(block_sizes, sub.model_id, batch);
                };
                Self::process_stream(stream, sub, indexer, state, changes, shutdown_rx, on_batch)
                    .await
            }
            Role::Loads => {
                let on_batch = |_: &KvEventBatch| applied = true;
                Self::process_records(stream, sub, state, changes, shutdown_rx, on_batch).await
            }
        };
        match result {
            Some(result) => Drained::Ended { result, applied },
            None => Drained::Shutdown,
        }
    }

    /// The next item of a stream, or a membership change taken in meanwhile:
    /// `None` when the engine's last removal ends the stream, `Some(None)`
    /// when the stream has nothing more.
    async fn next_batch(
        stream: &mut tonic::Streaming<KvEventBatch>,
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Option<Option<Result<KvEventBatch, tonic::Status>>> {
        use tokio_stream::StreamExt;

        loop {
            tokio::select! {
                biased;
                _ = &mut *shutdown_rx => return None,
                change = changes.recv() => match change {
                    Some(change) => {
                        Self::apply_change(sub, state, change).await;
                        if state.rejoin {
                            // The round ends; the next one subscribes from zero.
                            return Some(None);
                        }
                    }
                    None => return None,
                },
                item = stream.next() => return Some(item),
            }
        }
    }

    /// Drain a loads-only stream: every record to the member whose rank it
    /// is; a batch of events (a servicer that predates `loads_only`) is
    /// dropped, counted `not_indexed`. Nothing is admitted and no gap is
    /// looked for: the stream has no cursor.
    async fn process_records(
        mut stream: tonic::Streaming<KvEventBatch>,
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
        mut on_batch: impl FnMut(&KvEventBatch),
    ) -> Option<StreamResult> {
        loop {
            let batch =
                match Self::next_batch(&mut stream, sub, state, changes, shutdown_rx).await? {
                    Some(Ok(batch)) => batch,
                    Some(Err(e)) => return Some(StreamResult::Error(e)),
                    None => return Some(StreamResult::Ended),
                };
            on_batch(&batch);
            if let Some(load) = &batch.load {
                Self::forward_record(sub, state, &batch, load);
            }
            let disposition = if Self::is_load_only(&batch) {
                "load_only"
            } else {
                "not_indexed"
            };
            Self::count_records_batch(state, &batch, disposition);
        }
    }

    /// Count a loads-only stream's batch under the members whose rank its
    /// record is (every member for a record no member stands for).
    fn count_records_batch(
        state: &EngineStreamState,
        batch: &KvEventBatch,
        disposition: &'static str,
    ) {
        let rank = batch.dp_rank.unwrap_or(0);
        let taken = state.members.iter().any(|m| m.takes_record(rank));
        for member in state
            .members
            .iter()
            .filter(|m| !taken || m.takes_record(rank))
        {
            Metrics::record_kv_event_batch(&member.url, disposition);
        }
    }

    /// Process batches from a single stream connection: a batch's record
    /// goes to its rank's member (a contact with that worker), a batch of
    /// events is a contact with each of the engine's workers (the engine
    /// answered) and goes into the index state of the member whose rank it
    /// is.
    async fn process_stream(
        mut stream: tonic::Streaming<KvEventBatch>,
        sub: &EngineStream<'_>,
        indexer: &KvIndex,
        state: &mut EngineStreamState,
        changes: &mut mpsc::UnboundedReceiver<MemberChange>,
        shutdown_rx: &mut oneshot::Receiver<()>,
        mut on_batch: impl FnMut(&KvEventBatch),
    ) -> Option<StreamResult> {
        loop {
            let batch =
                match Self::next_batch(&mut stream, sub, state, changes, shutdown_rx).await? {
                    Some(Ok(batch)) => batch,
                    Some(Err(e)) => return Some(StreamResult::Error(e)),
                    None if state.rejoin => return Some(StreamResult::Rejoin),
                    None => return Some(StreamResult::Ended),
                };
            if let Some(load) = &batch.load {
                Self::forward_record(sub, state, &batch, load);
                if Self::is_load_only(&batch) {
                    Self::count_records_batch(state, &batch, "load_only");
                    continue;
                }
            }
            // A batch of events is the engine's stream working, whichever
            // rank's events they are; a record alone contacts its rank's
            // members only (`forward_record`).
            for member in &state.members {
                liveness::on_contact(&member.worker);
            }
            if let BatchOutcome::Gap { expected, received } =
                Self::admit_batch(&batch, indexer, state, &mut on_batch)
            {
                return Some(StreamResult::GapDetected { expected, received });
            }
        }
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
    use tokio_stream::wrappers::UnboundedReceiverStream;
    use tonic::{transport::Server, Request, Response, Status};
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::PolicyRegistry,
        routers::grpc::test_engine::serve_on_loopback,
        worker::{basic::StallReason, BasicWorkerBuilder, WorkerRegistry, UNKNOWN_MODEL_ID},
    };

    /// What a test's engine subscription holds: the monitor's maps a
    /// removal completes in (shared with the removal's task), the model's
    /// index for an events subscription (`None`: a loads-only one) and the
    /// load sink.
    struct Harness {
        slots: Arc<tokio::sync::Mutex<Subscriptions>>,
        indexers: Arc<DashMap<String, Arc<KvIndex>>>,
        block_sizes: Arc<DashMap<String, KvBlockSize>>,
        indexer: Option<Arc<KvIndex>>,
        load_sink: Option<Weak<WorkerMonitor>>,
    }

    impl Harness {
        fn new(indexer: Option<Arc<KvIndex>>, load_sink: Option<Weak<WorkerMonitor>>) -> Self {
            Self {
                slots: Arc::new(tokio::sync::Mutex::new(Subscriptions::default())),
                indexers: Arc::new(DashMap::new()),
                block_sizes: Arc::new(DashMap::new()),
                indexer,
                load_sink,
            }
        }

        /// An events subscription into `indexer`.
        fn events(indexer: Arc<KvIndex>) -> Self {
            Self::new(Some(indexer), None)
        }

        /// A loads-only subscription feeding `load_sink`.
        fn loads(load_sink: Option<Weak<WorkerMonitor>>) -> Self {
            Self::new(None, load_sink)
        }

        fn sub<'a>(&'a self, address: &'a str) -> EngineStream<'a> {
            EngineStream {
                address,
                role: match &self.indexer {
                    Some(indexer) => Role::Index {
                        indexer,
                        block_sizes: &self.block_sizes,
                    },
                    None => Role::Loads,
                },
                model_id: "m",
                load_sink: self.load_sink.as_ref(),
                slots: Arc::clone(&self.slots),
                indexers: Arc::clone(&self.indexers),
                block_sizes: Arc::clone(&self.block_sizes),
            }
        }

        /// The engine's state with `workers` as its members, each interned
        /// in the index on an events subscription.
        fn state(&self, workers: &[&Arc<dyn Worker>]) -> EngineStreamState {
            let members = workers
                .iter()
                .map(|worker| {
                    let worker_id = self
                        .indexer
                        .as_ref()
                        .map(|indexer| indexer.intern_worker(worker.url()).unwrap());
                    Member::new(Arc::clone(worker), worker_id)
                })
                .collect();
            let address = workers.first().map_or("", |worker| worker.base_url());
            EngineStreamState::new(address, members)
        }
    }

    /// A subscribe call of the test engine `sub` with `state`, under
    /// `deadline`, with no membership change arriving.
    async fn subscribe(
        sub: &EngineStream<'_>,
        state: &mut EngineStreamState,
        client: &BackendClient,
        shutdown_rx: &mut oneshot::Receiver<()>,
        deadline: Duration,
    ) -> Subscribed {
        let (_changes_tx, mut changes) = mpsc::unbounded_channel();
        KvEventMonitor::subscribe_with_deadline(
            sub,
            state,
            client,
            0,
            &mut changes,
            shutdown_rx,
            deadline,
        )
        .await
    }

    /// `stream` drained to its end for the test engine `sub` with `state`,
    /// with no membership change arriving.
    async fn drain(
        sub: &EngineStream<'_>,
        stream: tonic::Streaming<KvEventBatch>,
        state: &mut EngineStreamState,
        shutdown_rx: &mut oneshot::Receiver<()>,
    ) -> Drained {
        let (_changes_tx, mut changes) = mpsc::unbounded_channel();
        KvEventMonitor::drain_stream(sub, stream, state, &mut changes, shutdown_rx).await
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
        /// Answers the first call at once with a stream the test feeds (its
        /// items in order, open until the sender drops); every later call
        /// hangs, like `Hang`.
        Fed(Feed),
        /// Notes every request, answers at once with a stream that yields
        /// the script's items and then stays open.
        Counted {
            calls: Arc<Mutex<Vec<common::SubscribeKvEventsRequest>>>,
            script: Vec<Result<KvEventBatch, Status>>,
        },
        /// Notes every request and answers each call with the next step of
        /// the sequence (the last step answers every call after it).
        Sequence {
            calls: Arc<Mutex<Vec<common::SubscribeKvEventsRequest>>>,
            steps: Arc<Mutex<std::collections::VecDeque<Step>>>,
        },
    }

    /// The batches of a `Subscribe::Fed` stream, taken by the first call.
    type Feed = Arc<Mutex<Option<mpsc::UnboundedReceiver<Result<KvEventBatch, Status>>>>>;

    /// One call's answer in a `Subscribe::Sequence`.
    #[derive(Clone)]
    enum Step {
        /// A stream of the script's items that then ends.
        Ends(Vec<Result<KvEventBatch, Status>>),
        /// A stream of the script's items that then stays open.
        Stays(Vec<Result<KvEventBatch, Status>>),
        /// A stream that yields nothing and never ends.
        Open,
        /// The call is refused as unimplemented.
        Unimplemented,
        /// The call waits for the gate, then answers with an open stream.
        Gate(Arc<Notify>),
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
                Subscribe::Fed(batches) => {
                    let taken = batches.lock().unwrap().take();
                    match taken {
                        Some(batches) => Ok(Response::new(Box::pin(UnboundedReceiverStream::new(
                            batches,
                        )))),
                        None => std::future::pending().await,
                    }
                }
                Subscribe::Counted { calls, script } => {
                    calls.lock().unwrap().push(request.into_inner());
                    Ok(Response::new(Box::pin(futures::StreamExt::chain(
                        futures::stream::iter(script.clone()),
                        futures::stream::pending(),
                    ))))
                }
                Subscribe::Sequence { calls, steps } => {
                    calls.lock().unwrap().push(request.into_inner());
                    let step = {
                        let mut steps = steps.lock().unwrap();
                        if steps.len() > 1 {
                            steps.pop_front().expect("a step")
                        } else {
                            steps.front().cloned().expect("a step")
                        }
                    };
                    match step {
                        Step::Ends(script) => {
                            Ok(Response::new(Box::pin(futures::stream::iter(script))))
                        }
                        Step::Stays(script) => {
                            Ok(Response::new(Box::pin(futures::StreamExt::chain(
                                futures::stream::iter(script),
                                futures::stream::pending(),
                            ))))
                        }
                        Step::Open => Ok(Response::new(Box::pin(futures::stream::pending()))),
                        Step::Unimplemented => {
                            Err(Status::unimplemented("KV cache events not enabled"))
                        }
                        Step::Gate(gate) => {
                            gate.notified().await;
                            Ok(Response::new(Box::pin(futures::stream::pending())))
                        }
                    }
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
        let harness = Harness::events(Arc::new(KvIndex::chain()));
        let sub = harness.sub(worker.base_url());
        let mut state = harness.state(&[&worker]);
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();

        let Subscribed::Refused(status) = subscribe(
            &sub,
            &mut state,
            &client,
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
            subscribe(
                &sub,
                &mut state,
                &client,
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
            subscribe(
                &sub,
                &mut state,
                &client,
                &mut shutdown_rx,
                Duration::from_secs(5)
            )
            .await,
            Subscribed::Shutdown
        ));
        server.abort();

        let (addr, server) = serve(Subscribe::Unimplemented).await;
        let (worker, client) = connected(addr).await;
        let sub = harness.sub(worker.base_url());
        let mut state = harness.state(&[&worker]);
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let Subscribed::Refused(status) = subscribe(
            &sub,
            &mut state,
            &client,
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
            let harness = Harness::events(Arc::new(KvIndex::chain()));
            let sub = harness.sub(worker.base_url());
            let mut state = harness.state(&[&worker]);
            let wakes = wakes_of(&state);
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
            let Subscribed::Stream(stream) = subscribe(
                &sub,
                &mut state,
                &client,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
            .await
            else {
                panic!("case {which}: the server answers with a stream");
            };
            drain_contact(&wakes);
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
                drain(&sub, stream, &mut state, &mut shutdown_rx),
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
                    tokio::time::timeout(Duration::from_millis(100), woken(&wakes))
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
            .get_backend_client()
            .await
            .expect("a backend client")
            .expect("a gRPC worker has one");
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::clone(&worker));
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            1,
            false,
            false,
        ));
        let harness = Harness::loads(Some(Arc::downgrade(&monitor)));
        let sub = harness.sub(worker.base_url());
        let mut state = harness.state(&[&worker]);
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let Subscribed::Stream(stream) = subscribe(
            &sub,
            &mut state,
            &client,
            &mut shutdown_rx,
            Duration::from_secs(5),
        )
        .await
        else {
            panic!("the server answers with a stream");
        };
        let drained = tokio::time::timeout(
            Duration::from_secs(5),
            drain(&sub, stream, &mut state, &mut shutdown_rx),
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
        monitor.flush_pushed();
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
        assert_eq!(
            state.members[0].index.counters,
            WorkerIndexCounters::default(),
            "nothing was admitted on a loads-only stream"
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
            let harness = Harness::events(Arc::new(KvIndex::chain()));
            let sub = harness.sub(worker.base_url());
            let mut state = harness.state(&[&worker]);
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) = subscribe(
                &sub,
                &mut state,
                &client,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
            .await
            else {
                panic!("the server answers with a stream");
            };
            worker.set_stall(Some(StallReason::Unreachable));
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                drain(&sub, stream, &mut state, &mut shutdown_rx),
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
            let indexer = Arc::new(KvIndex::chain());
            let harness = Harness::events(Arc::clone(&indexer));
            let sub = harness.sub(worker.base_url());
            let mut state = harness.state(&[&worker]);
            let worker_id = state.members[0].worker_id.expect("interned");
            let (_keep, mut shutdown_rx) = oneshot::channel();
            let Subscribed::Stream(stream) = subscribe(
                &sub,
                &mut state,
                &client,
                &mut shutdown_rx,
                Duration::from_secs(5),
            )
            .await
            else {
                panic!("the server answers with a stream");
            };
            let drained = tokio::time::timeout(
                Duration::from_secs(5),
                drain(&sub, stream, &mut state, &mut shutdown_rx),
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
    /// which a loaded host stretches unevenly.
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
                tokio::time::sleep(Duration::from_millis(1_900)).await;
                monitor.on_worker_removed(worker.url()).await;
                server.abort();
                addr
            })
        });

        // Attempts at about 0, 100, 300, 700 and 1,500 ms: five in 1.9 s,
        // where a backoff reset on every connect made nineteen.
        let calls = calls.lock().unwrap().clone();
        assert!(
            (3..=6).contains(&calls.len()),
            "{} subscribe calls in 1.9 s",
            calls.len()
        );
        // The delay scheduled after each failed stream doubles up to the cap.
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
            Ok(())
        });
        // The last wait was slept: a loaded host only stretches it.
        let gaps: Vec<Duration> = calls.windows(2).map(|pair| pair[1] - pair[0]).collect();
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

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        for _ in 0..500 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    /// The workers on the engine's subscription at `address`, as the monitor
    /// counts them; `None` when the engine has no subscription. Read under
    /// the lock: a busy map must not read as an absent engine.
    async fn engine_members(monitor: &KvEventMonitor, address: &str) -> Option<usize> {
        let subs = monitor.subscriptions.lock().await;
        subs.engines.get(address).map(|engine| engine.members)
    }

    /// Polls `engine_members` until it reads `expected`, five seconds at most.
    async fn wait_for_members(
        monitor: &KvEventMonitor,
        address: &str,
        what: &str,
        expected: Option<usize>,
    ) {
        for _ in 0..500 {
            if engine_members(monitor, address).await == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    /// The first connect of a data-parallel engine waits for its other ranks
    /// (`RANKS_JOIN_GRACE`): rank workers registered a moment after the
    /// first one share its connect instead of joining a stream that has
    /// already delivered, which would start the engine over (a second
    /// subscribe call from zero).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rank_workers_arriving_within_the_grace_share_the_first_connect() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (addr, server) = serve(Subscribe::Counted {
            calls: Arc::clone(&calls),
            script: vec![Ok(ranked_batch(0))],
        })
        .await;
        let workers: Vec<Arc<dyn Worker>> = (0..4).map(|rank| rank_worker(addr, rank, 4)).collect();
        let address = workers[0].base_url().to_string();
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&workers[0], SubscriptionKind::Events)
            .await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            calls.lock().unwrap().is_empty(),
            "the first connect waits for the other ranks"
        );
        for worker in &workers[1..] {
            monitor
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        }
        wait_until("the engine's stream connects", || {
            !calls.lock().unwrap().is_empty()
        })
        .await;
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "one subscribe call: the later ranks joined before the connect"
        );
        assert_eq!(engine_members(&monitor, &address).await, Some(4));
        monitor.stop().await;
        server.abort();
    }

    /// How a two-rank engine with one rank on the task ends inside the grace.
    enum GraceEnd {
        /// The rank is removed (a registration rolled back, a fast health
        /// removal) before the other rank registers.
        Removed,
        /// The monitor stops.
        Stopped,
    }

    /// Adds `ranks` ranks of an engine one rank wider, so the first connect
    /// waits in its grace; ends the engine inside the grace by `end`, and
    /// returns the subscribe calls made and the panics counted
    /// (`smg_kv_event_subscription_failures_total{reason="panic"}` summed
    /// over every member's series, plus the orphan series).
    fn engine_ended_within_the_grace(ranks: usize, end: GraceEnd) -> (usize, f64) {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let calls = Arc::new(Mutex::new(Vec::new()));
        // One thread, so the task's metrics land on this thread's recorder.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let (addr, server) = serve(Subscribe::Counted {
                    calls: Arc::clone(&calls),
                    script: vec![Ok(ranked_batch(0))],
                })
                .await;
                let workers: Vec<Arc<dyn Worker>> = (0..ranks)
                    .map(|rank| rank_worker(addr, rank, ranks + 1))
                    .collect();
                let address = workers[0].base_url().to_string();
                let monitor = KvEventMonitor::new();
                for worker in &workers {
                    monitor
                        .on_worker_added(worker, SubscriptionKind::Events)
                        .await;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                match end {
                    GraceEnd::Removed => {
                        for worker in &workers {
                            monitor.on_worker_removed(worker.url()).await;
                        }
                    }
                    GraceEnd::Stopped => monitor.stop().await,
                }
                wait_for_members(&monitor, &address, "the engine's entry is gone", None).await;
                server.abort();
            });
        });
        let panics = handle
            .render()
            .lines()
            .filter(|line| {
                (line.starts_with("smg_kv_event_subscription_failures_total{")
                    && line.contains("reason=\"panic\""))
                    || line.starts_with("smg_kv_event_subscription_orphan_panics_total ")
            })
            .filter_map(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
            .sum();
        let calls = calls.lock().unwrap().len();
        (calls, panics)
    }

    /// The engine address whose next batch panics its task, armed by the
    /// test that reads where a panic with members on the task is counted.
    static PANIC_ON_BATCH_AT: parking_lot::Mutex<Option<String>> = parking_lot::Mutex::new(None);

    /// The hook `drain_stream` runs on every batch under the tests alone.
    pub(super) fn panic_if_armed(address: &str) {
        assert_ne!(
            PANIC_ON_BATCH_AT.lock().as_deref(),
            Some(address),
            "a test's panic on a batch, with members on the task"
        );
    }

    /// Arms the hook for one engine address until dropped.
    struct ArmedPanic;

    impl ArmedPanic {
        fn arm(address: &str) -> Self {
            *PANIC_ON_BATCH_AT.lock() = Some(address.to_owned());
            Self
        }
    }

    impl Drop for ArmedPanic {
        fn drop(&mut self) {
            *PANIC_ON_BATCH_AT.lock() = None;
        }
    }

    /// The grace ends with the engine when its one rank is removed inside
    /// it: the task leaves through its exit path. A shutdown taken in the
    /// grace spends the receiver, and a round that polled it again was a
    /// caught panic counted as a failure, the exit path skipped.
    #[test]
    fn a_rank_removed_within_the_grace_ends_the_engine_without_a_panic() {
        let (calls, panics) = engine_ended_within_the_grace(1, GraceEnd::Removed);
        assert_eq!(calls, 0, "no connect: the engine ended inside its grace");
        assert_eq!(panics, 0.0, "a removal inside the grace is no panic");
    }

    /// A stop inside the grace ends the task the same way: the exit path
    /// takes the member present, nothing is counted as a panic.
    #[test]
    fn a_stop_within_the_grace_ends_the_engine_without_a_panic() {
        let (calls, panics) = engine_ended_within_the_grace(1, GraceEnd::Stopped);
        assert_eq!(calls, 0, "no connect: the engine ended inside its grace");
        assert_eq!(panics, 0.0, "a stop inside the grace is no panic");
    }

    /// With two ranks on the task, a stop inside the grace ends it with both
    /// members on the stream. `stop` has taken their slots first, so a panic
    /// there would land on the orphan series, as with one rank; none does.
    #[test]
    fn a_stop_within_the_grace_with_two_ranks_counts_no_panic() {
        let (calls, panics) = engine_ended_within_the_grace(2, GraceEnd::Stopped);
        assert_eq!(calls, 0, "no connect: the engine ended inside its grace");
        assert_eq!(panics, 0.0, "a stop inside the grace is no panic");
    }

    /// A panic with members on the task is counted under each member and
    /// not on the orphan series: the hook panics the task on its first
    /// batch, with both ranks of a two-rank engine on it.
    #[test]
    fn a_panic_with_members_on_the_task_counts_under_each_member() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let members: Vec<String> = metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let (addr, server) = serve(Subscribe::Counted {
                    calls: Arc::new(Mutex::new(Vec::new())),
                    script: vec![Ok(ranked_batch(0))],
                })
                .await;
                let workers: Vec<Arc<dyn Worker>> =
                    (0..2).map(|rank| rank_worker(addr, rank, 2)).collect();
                let address = workers[0].base_url().to_string();
                let _armed = ArmedPanic::arm(&address);
                let monitor = KvEventMonitor::new();
                for worker in &workers {
                    monitor
                        .on_worker_added(worker, SubscriptionKind::Events)
                        .await;
                }
                wait_for_members(&monitor, &address, "the engine's entry is gone", None).await;
                server.abort();
                workers
                    .iter()
                    .map(|worker| worker.url().to_string())
                    .collect()
            })
        });
        let rendered = handle.render();
        for member in &members {
            let counted = rendered.lines().any(|line| {
                line.starts_with("smg_kv_event_subscription_failures_total{")
                    && line.contains("reason=\"panic\"")
                    && line.contains(member.as_str())
                    && line.ends_with(" 1")
            });
            assert!(counted, "the panic counts once under {member}:\n{rendered}");
        }
        assert!(
            !rendered.contains("smg_kv_event_subscription_orphan_panics_total"),
            "no orphan count with members on the task:\n{rendered}"
        );
    }

    /// A rank worker that joins while a reconnect's subscribe call is
    /// pending is a late join too: the stream that call returns is dropped
    /// and the next round starts the engine over from zero for the rank,
    /// instead of draining a stream that resumed past its history.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rank_joining_during_a_pending_reconnect_starts_the_engine_over() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(Notify::new());
        let steps = std::collections::VecDeque::from([
            Step::Ends(vec![Ok(ranked_batch(0))]),
            Step::Gate(Arc::clone(&gate)),
            Step::Open,
        ]);
        let (addr, server) = serve(Subscribe::Sequence {
            calls: Arc::clone(&calls),
            steps: Arc::new(Mutex::new(steps)),
        })
        .await;
        let first = rank_worker(addr, 0, 2);
        let second = rank_worker(addr, 1, 2);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        // The first stream delivers rank 0's batch and ends; the reconnect's
        // call is held at the gate.
        wait_until("the reconnect's call is pending", || {
            calls.lock().unwrap().len() == 2
        })
        .await;
        monitor
            .on_worker_added(&second, SubscriptionKind::Events)
            .await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        gate.notify_one();
        wait_until(
            "the engine resubscribes from zero for the late rank",
            || calls.lock().unwrap().len() == 3,
        )
        .await;
        let requests = calls.lock().unwrap().clone();
        assert_eq!(
            requests[1].start_sequence_number, 1,
            "the reconnect resumed past rank 0's batch"
        );
        assert_eq!(
            requests[2].start_sequence_number, 0,
            "the late join starts from zero"
        );
        monitor.stop().await;
        server.abort();
    }

    /// A subscribe attempt that starts the engine over for a late rank but
    /// delivers nothing does not spend the decision: the next round asks for
    /// zero again instead of resuming from the first rank's cursor, so the
    /// late rank's history still arrives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_attempt_after_a_late_join_starts_from_zero_again() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let steps = std::collections::VecDeque::from([
            Step::Stays(vec![Ok(ranked_batch_at(0, 1)), Ok(ranked_batch_at(0, 2))]),
            Step::Ends(vec![]),
            Step::Stays(vec![
                Ok(ranked_batch_at(0, 1)),
                Ok(ranked_batch_at(0, 2)),
                Ok(ranked_batch_at(0, 3)),
                Ok(ranked_batch_at(1, 1)),
            ]),
        ]);
        let (addr, server) = serve(Subscribe::Sequence {
            calls: Arc::clone(&calls),
            steps: Arc::new(Mutex::new(steps)),
        })
        .await;
        let first = rank_worker(addr, 0, 2);
        let second = rank_worker(addr, 1, 2);
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("rank 0's cursor stands at 2", || {
            index
                .worker_id(first.url())
                .is_some_and(|id| index.worker_block_count(id) == 2)
        })
        .await;
        monitor
            .on_worker_added(&second, SubscriptionKind::Events)
            .await;
        // The from-zero attempt ends without a batch; the round after it
        // asks for zero again and delivers both ranks' history.
        wait_until("the third subscribe call", || {
            calls.lock().unwrap().len() == 3
        })
        .await;
        let requests = calls.lock().unwrap().clone();
        assert_eq!(
            requests[1].start_sequence_number, 0,
            "the late join starts from zero"
        );
        assert_eq!(
            requests[2].start_sequence_number, 0,
            "an attempt that delivered nothing leaves the late rank's history owed"
        );
        wait_until("both ranks' history lands", || {
            index
                .worker_id(second.url())
                .is_some_and(|id| index.worker_block_count(id) == 1)
                && index
                    .worker_id(first.url())
                    .is_some_and(|id| index.worker_block_count(id) == 3)
        })
        .await;
        monitor.stop().await;
        server.abort();
    }

    /// A round that resumes from the cursor ends the replay skip: after a
    /// from-zero stream delivered the late rank's history but ended before
    /// the first rank's replay reached its cursor, the next round resumes
    /// from that cursor, and a sequence below it is a restarted publisher
    /// again (its index rebuilt), not the replay.
    #[test]
    fn a_resumed_round_ends_the_replay_skip() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let calls = Arc::new(Mutex::new(Vec::new()));
                let steps = std::collections::VecDeque::from([
                    Step::Stays(vec![Ok(ranked_batch_at(0, 1)), Ok(ranked_batch_at(0, 2))]),
                    Step::Ends(vec![Ok(ranked_batch_at(1, 1))]),
                    Step::Stays(vec![
                        Ok(ranked_batch_at(0, 1)),
                        Ok(ranked_batch_at(0, 2)),
                        Ok(ranked_batch_at(0, 3)),
                    ]),
                ]);
                let (addr, server) = serve(Subscribe::Sequence {
                    calls: Arc::clone(&calls),
                    steps: Arc::new(Mutex::new(steps)),
                })
                .await;
                let first = rank_worker(addr, 0, 2);
                let second = rank_worker(addr, 1, 2);
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&first, SubscriptionKind::Events)
                    .await;
                let index = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                wait_until("rank 0's cursor stands at 2", || {
                    index
                        .worker_id(first.url())
                        .is_some_and(|id| index.worker_block_count(id) == 2)
                })
                .await;
                monitor
                    .on_worker_added(&second, SubscriptionKind::Events)
                    .await;
                // The from-zero stream delivers the late rank's batch and ends
                // before rank 0's replay; the next round resumes from rank 0's
                // cursor, where the relay sends a restarted publisher's 1, 2, 3.
                wait_until("the third subscribe call", || {
                    calls.lock().unwrap().len() == 3
                })
                .await;
                let requests = calls.lock().unwrap().clone();
                assert_eq!(
                    requests[1].start_sequence_number, 0,
                    "the late join starts from zero"
                );
                assert_eq!(
                    requests[2].start_sequence_number, 2,
                    "the history delivered, the round resumes from rank 0's cursor"
                );
                wait_until("the restarted publisher's batches land", || {
                    index
                        .worker_id(first.url())
                        .is_some_and(|id| index.worker_block_count(id) == 3)
                })
                .await;
                assert_eq!(
                    resyncs(&handle, first.url(), Some("publisher_restart")),
                    1.0,
                    "a sequence below the cursor on a resumed stream is a restart, not the replay"
                );
                monitor.stop().await;
                server.abort();
            });
        });
    }

    /// A rank that joins late and publishes nothing owes its history for one
    /// completed round from zero: once that stream reached live for the
    /// first rank (a batch past its cursor) the joiner's window, if any,
    /// has followed or follows through the refused cursor's restart, so the
    /// next round resumes from the cursor (restart detection back) instead
    /// of starting the engine over at every reconnect while the rank idles.
    #[test]
    fn an_idle_late_rank_costs_one_round_from_zero() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let calls = Arc::new(Mutex::new(Vec::new()));
                // The first stream moves rank 0's cursor to 2. After the join
                // the stream from zero replays rank 0 past that cursor with
                // nothing of rank 1 and ends; the next one ends at once too.
                let steps = std::collections::VecDeque::from([
                    Step::Stays(vec![Ok(ranked_batch_at(0, 1)), Ok(ranked_batch_at(0, 2))]),
                    Step::Ends(vec![
                        Ok(ranked_batch_at(0, 1)),
                        Ok(ranked_batch_at(0, 2)),
                        Ok(ranked_batch_at(0, 3)),
                    ]),
                    Step::Ends(vec![]),
                    Step::Open,
                ]);
                let (addr, server) = serve(Subscribe::Sequence {
                    calls: Arc::clone(&calls),
                    steps: Arc::new(Mutex::new(steps)),
                })
                .await;
                let first = rank_worker(addr, 0, 2);
                let second = rank_worker(addr, 1, 2);
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&first, SubscriptionKind::Events)
                    .await;
                let index = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                wait_until("rank 0's cursor stands at 2", || {
                    index
                        .worker_id(first.url())
                        .is_some_and(|id| index.worker_block_count(id) == 2)
                })
                .await;
                monitor
                    .on_worker_added(&second, SubscriptionKind::Events)
                    .await;
                wait_until("the fourth subscribe call", || {
                    calls.lock().unwrap().len() == 4
                })
                .await;
                let requests = calls.lock().unwrap().clone();
                assert_eq!(
                    requests[1].start_sequence_number, 0,
                    "the late join starts from zero"
                );
                assert_eq!(
                    requests[2].start_sequence_number, 3,
                    "the replay walked past rank 0's cursor: the round resumes from it"
                );
                assert_eq!(
                    requests[3].start_sequence_number, 3,
                    "and the rounds keep resuming while the late rank idles"
                );
                assert_eq!(
                    index
                        .worker_id(first.url())
                        .map(|id| index.worker_block_count(id)),
                    Some(3),
                    "rank 0 skipped the replay and took the batch past its cursor"
                );
                assert_eq!(
                    index
                        .worker_id(second.url())
                        .map(|id| index.worker_block_count(id)),
                    Some(0),
                    "the idle rank is interned with nothing"
                );
                assert_eq!(
                    resyncs(&handle, second.url(), Some("rank_joined")),
                    1.0,
                    "the join is the joiner's one resync"
                );
                assert_eq!(
                    resyncs(&handle, first.url(), None),
                    0.0,
                    "no resync of the first rank"
                );
                monitor.stop().await;
                server.abort();
            });
        });
    }

    /// A member slot of a task that ended (an engine without KV events)
    /// does not reach the engine's next task: the old member's removal
    /// leaves the successor's count alone, and its re-add joins the
    /// successor instead of being skipped as already active.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_ended_tasks_member_does_not_touch_its_successors_subscription() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let steps = std::collections::VecDeque::from([Step::Unimplemented, Step::Open]);
        let (addr, server) = serve(Subscribe::Sequence {
            calls: Arc::clone(&calls),
            steps: Arc::new(Mutex::new(steps)),
        })
        .await;
        let workers: Vec<Arc<dyn Worker>> = (0..3).map(|rank| rank_worker(addr, rank, 3)).collect();
        let address = workers[0].base_url().to_string();
        let monitor = KvEventMonitor::new();
        for worker in &workers[..2] {
            monitor
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        }
        // The first task's one call is refused as unimplemented and the
        // task ends; its two member slots stay.
        wait_for_members(&monitor, &address, "the first task ends", None).await;
        assert_eq!(calls.lock().unwrap().len(), 1);
        monitor
            .on_worker_added(&workers[2], SubscriptionKind::Events)
            .await;
        wait_until("the successor connects", || {
            calls.lock().unwrap().len() == 2
        })
        .await;
        assert_eq!(engine_members(&monitor, &address).await, Some(1));

        monitor.on_worker_removed(workers[0].url()).await;
        assert_eq!(
            engine_members(&monitor, &address).await,
            Some(1),
            "an ended task's member leaves the successor's count alone"
        );
        monitor
            .on_worker_added(&workers[1], SubscriptionKind::Events)
            .await;
        wait_for_members(
            &monitor,
            &address,
            "the stale member joins the successor",
            Some(2),
        )
        .await;
        assert_eq!(calls.lock().unwrap().len(), 2, "no restart for the join");
        monitor.stop().await;
        server.abort();
    }

    /// One stored block of rank `rank`, its tokens unique to the rank, as
    /// the first batch of that rank's publisher.
    fn ranked_batch(rank: i32) -> KvEventBatch {
        let mut batch = stored_batch(&[(100 + i64::from(rank), 16)]);
        if let Some(kv_cache_event::Data::Stored(stored)) = batch.events[0].data.as_mut() {
            let first = 1_000 * (rank as u32 + 1);
            stored.blocks[0].token_ids = (first..first + 16).collect();
        }
        batch.dp_rank = Some(rank);
        batch
    }

    /// `ranked_batch` numbered `seq`, its one block distinct per sequence
    /// number, so a rank's batches add up in the index.
    fn ranked_batch_at(rank: i32, seq: u64) -> KvEventBatch {
        let mut batch = ranked_batch(rank);
        batch.sequence_number = seq;
        let hash = 100 + i64::from(rank) + 10 * seq as i64;
        if let Some(kv_cache_event::Data::Stored(stored)) = batch.events[0].data.as_mut() {
            stored.blocks[0].block_hash = hash;
            let first = 1_000 * (rank as u32 + 1) + 16 * seq as u32;
            stored.blocks[0].token_ids = (first..first + 16).collect();
        }
        batch.events[0].event_id = hash as u64;
        batch
    }

    /// The rank workers of a data-parallel engine share one subscription:
    /// four rank workers registered one after another open one stream and
    /// are all interned in the model's index. A rank worker's removal takes
    /// its blocks out and leaves the others' stream running, with no new
    /// subscribe call; the last removal ends the subscription.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_data_parallel_engine_opens_one_subscription_for_its_ranks() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (addr, server) = serve(Subscribe::Counted {
            calls: Arc::clone(&calls),
            script: Vec::new(),
        })
        .await;
        let workers: Vec<Arc<dyn Worker>> = (0..4).map(|rank| rank_worker(addr, rank, 4)).collect();
        let address = workers[0].base_url().to_string();
        let monitor = KvEventMonitor::new();
        for worker in &workers {
            monitor
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        }
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("every rank worker is interned", || {
            workers
                .iter()
                .all(|worker| index.worker_id(worker.url()).is_some())
        })
        .await;
        wait_until("the engine's stream connects", || {
            !calls.lock().unwrap().is_empty()
        })
        .await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "one subscribe call for four rank workers"
        );
        assert_eq!(engine_members(&monitor, &address).await, Some(4));

        monitor.on_worker_removed(workers[1].url()).await;
        assert!(
            index.worker_id(workers[1].url()).is_none(),
            "the removed rank's id is released"
        );
        for worker in [&workers[0], &workers[2], &workers[3]] {
            assert!(
                index.worker_id(worker.url()).is_some(),
                "the other ranks keep their ids"
            );
        }
        assert_eq!(engine_members(&monitor, &address).await, Some(3));
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "a rank worker's removal restarts nothing"
        );

        for worker in [&workers[0], &workers[2], &workers[3]] {
            monitor.on_worker_removed(worker.url()).await;
        }
        wait_for_members(
            &monitor,
            &address,
            "the last removal ends the engine's subscription",
            None,
        )
        .await;
        assert!(
            monitor.get_indexer(UNKNOWN_MODEL_ID).is_none(),
            "the model's index went with its last worker"
        );
        assert_eq!(calls.lock().unwrap().len(), 1);
        server.abort();
    }

    /// A rank worker's removal takes its blocks out of the index under the
    /// fleet-wide removal permits, which other engines' drains may hold.
    /// That wait is the leaving rank's alone: the engine's stream goes on
    /// for its other ranks, whose next batch lands while every permit is
    /// held, and the removal completes once a permit is free.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_sibling_ranks_batch_lands_while_a_removed_ranks_cleanup_waits() {
        let (feed, batches) = mpsc::unbounded_channel();
        let (addr, server) = serve(Subscribe::Fed(Arc::new(Mutex::new(Some(batches))))).await;
        let workers: Vec<Arc<dyn Worker>> = (0..2).map(|rank| rank_worker(addr, rank, 2)).collect();
        let address = workers[0].base_url().to_string();
        let monitor = Arc::new(KvEventMonitor::new());
        for worker in &workers {
            monitor
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        }
        let index = monitor
            .get_indexer(UNKNOWN_MODEL_ID)
            .expect("the model's index");
        wait_until("both rank workers are interned", || {
            workers
                .iter()
                .all(|worker| index.worker_id(worker.url()).is_some())
        })
        .await;
        let sibling = index
            .worker_id(workers[1].url())
            .expect("rank 1 is interned");
        feed.send(Ok(ranked_batch_at(1, 1)))
            .expect("the stream is open");
        wait_until("rank 1's first batch lands", || {
            index.worker_block_count(sibling) == 1
        })
        .await;

        // Every removal permit is held, as other engines' drains hold them.
        let permits = INDEX_REMOVAL_PERMITS
            .acquire_many(MAX_CONCURRENT_INDEX_REMOVALS as u32)
            .await
            .expect("the removal semaphore is open");
        #[expect(clippy::disallowed_methods, reason = "the removal is awaited below")]
        let removal = tokio::spawn({
            let monitor = Arc::clone(&monitor);
            let url = workers[0].url().to_string();
            async move { monitor.on_worker_removed(&url).await }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!removal.is_finished(), "the removal waits for a permit");

        feed.send(Ok(ranked_batch_at(1, 2)))
            .expect("the stream is open");
        wait_until(
            "rank 1's next batch lands while rank 0's cleanup waits for a permit",
            || index.worker_block_count(sibling) == 2,
        )
        .await;
        assert!(
            !removal.is_finished(),
            "the cleanup still waits for a permit"
        );
        assert!(
            index.worker_id(workers[0].url()).is_some(),
            "rank 0's id is held until its blocks are out"
        );

        drop(permits);
        removal.await.expect("removal task");
        assert!(
            index.worker_id(workers[0].url()).is_none(),
            "the removed rank's id is released"
        );
        assert_eq!(
            index.worker_block_count(sibling),
            2,
            "the sibling keeps its blocks"
        );
        assert_eq!(engine_members(&monitor, &address).await, Some(1));
        monitor.stop().await;
        server.abort();
    }

    /// On the engine's one stream a batch enters the index state of the
    /// rank worker whose rank it names and no other's; a batch of a rank no
    /// worker stands for enters none.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn each_ranks_batches_reach_its_own_workers_index_only() {
        let script = (0..4)
            .map(|rank| Ok(ranked_batch(rank)))
            .chain(std::iter::once(Ok(ranked_batch(7))))
            .collect();
        let (addr, server) = serve(Subscribe::Stream(script)).await;
        let (_transport_worker, client) = connected(addr).await;
        let workers: Vec<Arc<dyn Worker>> = (0..4).map(|rank| rank_worker(addr, rank, 4)).collect();
        let indexer = Arc::new(KvIndex::chain());
        let harness = Harness::events(Arc::clone(&indexer));
        let sub = harness.sub(workers[0].base_url());
        let members: Vec<&Arc<dyn Worker>> = workers.iter().collect();
        let mut state = harness.state(&members);
        let (_keep, mut shutdown_rx) = oneshot::channel();
        let Subscribed::Stream(stream) = subscribe(
            &sub,
            &mut state,
            &client,
            &mut shutdown_rx,
            Duration::from_secs(5),
        )
        .await
        else {
            panic!("the server answers with a stream");
        };
        let drained = tokio::time::timeout(
            Duration::from_secs(5),
            drain(&sub, stream, &mut state, &mut shutdown_rx),
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
        for (rank, member) in state.members.iter().enumerate() {
            assert_eq!(
                indexer.worker_block_count(member.worker_id.expect("interned")),
                1,
                "rank {rank} holds its own batch's block and no other's"
            );
        }
        assert_eq!(indexer.current_size(), 4, "rank 7's block entered no index");
        assert_eq!(
            state.resume_sequence(),
            1,
            "rank 0's cursor stands at its one batch"
        );
        assert_eq!(state.ranks.len(), 5, "every rank's cursor moved once");
        server.abort();
    }

    /// On the engine's one loads-only stream each rank's record reaches the
    /// rank worker that stands for it and no other; a record of a rank no
    /// worker stands for reaches none.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn each_ranks_load_record_reaches_its_own_worker_only() {
        let record = |running: u32, rank: i32| KvEventBatch {
            sequence_number: 0,
            dp_rank: Some(rank),
            load: Some(EngineLoad {
                running_requests: running,
                load_only: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let script = vec![Ok(record(9, 1)), Ok(record(3, 0)), Ok(record(1, 7))];
        let (addr, server) = serve(Subscribe::Stream(script)).await;
        let (_transport_worker, client) = connected(addr).await;
        let workers: Vec<Arc<dyn Worker>> = (0..2).map(|rank| rank_worker(addr, rank, 2)).collect();
        let registry = Arc::new(WorkerRegistry::new());
        for worker in &workers {
            registry.register(Arc::clone(worker));
        }
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            1,
            false,
            false,
        ));
        let harness = Harness::loads(Some(Arc::downgrade(&monitor)));
        let sub = harness.sub(workers[0].base_url());
        let members: Vec<&Arc<dyn Worker>> = workers.iter().collect();
        let mut state = harness.state(&members);
        let (_keep, mut shutdown_rx) = oneshot::channel::<()>();
        let Subscribed::Stream(stream) = subscribe(
            &sub,
            &mut state,
            &client,
            &mut shutdown_rx,
            Duration::from_secs(5),
        )
        .await
        else {
            panic!("the server answers with a stream");
        };
        let drained = tokio::time::timeout(
            Duration::from_secs(5),
            drain(&sub, stream, &mut state, &mut shutdown_rx),
        )
        .await
        .expect("the drain ends");
        assert!(matches!(
            drained,
            Drained::Ended {
                result: StreamResult::Ended,
                applied: true
            }
        ));
        monitor.flush_pushed();
        let snapshot = monitor.load_snapshot();
        let ranks_of = |worker: &Arc<dyn Worker>| -> Vec<(i32, i32)> {
            snapshot
                .get(worker.url())
                .expect("the rank worker's pushed report is in the snapshot")
                .loads
                .iter()
                .map(|load| (load.dp_rank, load.num_running_reqs))
                .collect()
        };
        assert_eq!(ranks_of(&workers[0]), vec![(0, 3)], "rank 0's record alone");
        assert_eq!(ranks_of(&workers[1]), vec![(1, 9)], "rank 1's record alone");
        server.abort();
    }

    /// A second add of a rank worker already on its engine's stream changes
    /// nothing: no join, no new subscribe call, the member count stands.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_second_add_of_a_rank_worker_changes_nothing() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (addr, server) = serve(Subscribe::Counted {
            calls: Arc::clone(&calls),
            script: Vec::new(),
        })
        .await;
        let workers: Vec<Arc<dyn Worker>> = (0..2).map(|rank| rank_worker(addr, rank, 2)).collect();
        let address = workers[0].base_url().to_string();
        let monitor = KvEventMonitor::new();
        for worker in &workers {
            monitor
                .on_worker_added(worker, SubscriptionKind::Events)
                .await;
        }
        wait_until("the engine's stream connects", || {
            !calls.lock().unwrap().is_empty()
        })
        .await;
        monitor
            .on_worker_added(&workers[0], SubscriptionKind::Events)
            .await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(engine_members(&monitor, &address).await, Some(2));
        assert_eq!(calls.lock().unwrap().len(), 1);
        monitor.stop().await;
        server.abort();
    }

    /// The worker's resyncs of one reason, or of every reason, as the
    /// recorder renders them.
    fn resyncs(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
        worker_url: &str,
        reason: Option<&str>,
    ) -> f64 {
        let prefix = match reason {
            Some(reason) => {
                format!("smg_kv_event_resyncs_total{{worker=\"{worker_url}\",reason=\"{reason}\"}}")
            }
            None => format!("smg_kv_event_resyncs_total{{worker=\"{worker_url}\","),
        };
        handle
            .render()
            .lines()
            .filter(|line| line.starts_with(&prefix))
            .filter_map(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
            .sum()
    }

    /// A rank worker that joins after the engine's stream delivered has
    /// missed its rank's history: the engine resubscribes from zero, counted
    /// as a `rank_joined` resync of the joining worker alone; the workers
    /// already streaming keep their blocks and skip the replay up to their
    /// cursors (a replay from below the cursor is not a publisher restart),
    /// take what follows it, and the new worker is interned beside them.
    #[test]
    fn a_rank_worker_joining_late_starts_the_engine_over_from_zero() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let calls = Arc::new(Mutex::new(Vec::new()));
                // The first stream moves rank 0's cursor to 2; the stream
                // after the join replays rank 0 from the start (below that
                // cursor), goes past it, and carries the joiner's first batch.
                let steps = std::collections::VecDeque::from([
                    Step::Stays(vec![Ok(ranked_batch_at(0, 1)), Ok(ranked_batch_at(0, 2))]),
                    Step::Stays(vec![
                        Ok(ranked_batch_at(0, 1)),
                        Ok(ranked_batch_at(0, 2)),
                        Ok(ranked_batch_at(0, 3)),
                        Ok(ranked_batch_at(1, 1)),
                    ]),
                ]);
                let (addr, server) = serve(Subscribe::Sequence {
                    calls: Arc::clone(&calls),
                    steps: Arc::new(Mutex::new(steps)),
                })
                .await;
                let first = rank_worker(addr, 0, 2);
                let second = rank_worker(addr, 1, 2);
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&first, SubscriptionKind::Events)
                    .await;
                // The first connect waits for the second rank, then goes
                // ahead without it; its batches move rank 0's cursor to 2.
                let index = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                wait_until("the engine's stream connects and delivers", || {
                    index
                        .worker_id(first.url())
                        .is_some_and(|id| index.worker_block_count(id) == 2)
                })
                .await;
                assert_eq!(calls.lock().unwrap().len(), 1);
                assert_eq!(resyncs(&handle, first.url(), None), 0.0);

                monitor
                    .on_worker_added(&second, SubscriptionKind::Events)
                    .await;
                wait_until("the engine resubscribes for the late rank", || {
                    calls.lock().unwrap().len() == 2
                })
                .await;
                let requests = calls.lock().unwrap().clone();
                assert_eq!(
                    requests[1].start_sequence_number, 0,
                    "the second subscription starts from zero"
                );
                wait_until(
                    "the joiner's batch and rank 0's batch past the replay land",
                    || {
                        index
                            .worker_id(second.url())
                            .is_some_and(|id| index.worker_block_count(id) == 1)
                            && index
                                .worker_id(first.url())
                                .is_some_and(|id| index.worker_block_count(id) == 3)
                    },
                )
                .await;
                assert_eq!(
                    resyncs(&handle, second.url(), Some("rank_joined")),
                    1.0,
                    "the join is the joiner's resync"
                );
                assert_eq!(
                    resyncs(&handle, first.url(), None),
                    0.0,
                    "the replay below rank 0's cursor is skipped, not read as a publisher restart"
                );
                monitor.stop().await;
                server.abort();
            });
        });
    }

    /// A rank worker re-registered under its URL while its sibling streams:
    /// the old object's membership ends on the engine's stream (its blocks
    /// out of the index, the object released) and the new object joins that
    /// stream as a late rank, so the engine starts over from zero for its
    /// rank alone, counted as the joiner's resync; the sibling keeps its
    /// blocks, skips the replay up to its cursor and takes what follows,
    /// and the rank's replayed batches are indexed under the new object
    /// (the index hands a freed id out again, so the id itself says nothing).
    #[test]
    fn a_rank_re_registered_beside_a_streaming_sibling_rejoins_its_engines_stream() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let calls = Arc::new(Mutex::new(Vec::new()));
                // The first stream moves both cursors to 1; the stream after
                // the re-registration replays both ranks from the start, goes
                // past rank 0's cursor and brings rank 1 a second batch.
                let steps = std::collections::VecDeque::from([
                    Step::Stays(vec![Ok(ranked_batch_at(0, 1)), Ok(ranked_batch_at(1, 1))]),
                    Step::Stays(vec![
                        Ok(ranked_batch_at(0, 1)),
                        Ok(ranked_batch_at(1, 1)),
                        Ok(ranked_batch_at(1, 2)),
                        Ok(ranked_batch_at(0, 2)),
                    ]),
                ]);
                let (addr, server) = serve(Subscribe::Sequence {
                    calls: Arc::clone(&calls),
                    steps: Arc::new(Mutex::new(steps)),
                })
                .await;
                let first = rank_worker(addr, 0, 2);
                let second = rank_worker(addr, 1, 2);
                let monitor = KvEventMonitor::new();
                monitor
                    .on_worker_added(&first, SubscriptionKind::Events)
                    .await;
                monitor
                    .on_worker_added(&second, SubscriptionKind::Events)
                    .await;
                let index = monitor
                    .get_indexer(UNKNOWN_MODEL_ID)
                    .expect("the model's index");
                wait_until("the engine's stream delivers both ranks' first batches", || {
                    [&first, &second].into_iter().all(|worker| {
                        index
                            .worker_id(worker.url())
                            .is_some_and(|id| index.worker_block_count(id) == 1)
                    })
                })
                .await;
                assert_eq!(calls.lock().unwrap().len(), 1);

                // The registry replaced rank 1's object under its URL.
                let replacement = rank_worker(addr, 1, 2);
                monitor
                    .on_worker_added(&replacement, SubscriptionKind::Events)
                    .await;
                wait_until("the engine resubscribes for the re-registered rank", || {
                    calls.lock().unwrap().len() == 2
                })
                .await;
                assert_eq!(
                    calls.lock().unwrap()[1].start_sequence_number,
                    0,
                    "the stream after the re-registration starts from zero"
                );
                wait_until(
                    "rank 1's replayed batches land under the new object and rank 0 goes past its cursor",
                    || {
                        index
                            .worker_id(replacement.url())
                            .is_some_and(|id| index.worker_block_count(id) == 2)
                            && index
                                .worker_id(first.url())
                                .is_some_and(|id| index.worker_block_count(id) == 2)
                    },
                )
                .await;
                wait_until("the old object is released", || {
                    Arc::strong_count(&second) == 1
                })
                .await;
                assert_eq!(
                    engine_members(&monitor, first.base_url()).await,
                    Some(2)
                );
                assert_eq!(
                    resyncs(&handle, replacement.url(), Some("rank_joined")),
                    1.0,
                    "the restart is the re-registered rank's resync"
                );
                assert_eq!(
                    resyncs(&handle, first.url(), None),
                    0.0,
                    "the sibling keeps its blocks: the replay below its cursor is no restart"
                );
                monitor.stop().await;
                server.abort();
            });
        });
    }
}
