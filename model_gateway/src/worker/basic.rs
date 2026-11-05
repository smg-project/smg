use std::{
    any::Any,
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use arc_swap::{ArcSwap, ArcSwapOption};
use async_trait::async_trait;
// Re-export protocol types as the canonical types for the gateway
pub use openai_protocol::worker::{ConnectionMode, ProfileOptions, RuntimeType, WorkerType};
use openai_protocol::{
    model_card::ModelCard,
    model_type::Endpoint,
    worker::{ProviderType, WorkerInfo, WorkerModels, WorkerStatus},
};
use smg_grpc_client::common_proto;
use tokio::{
    sync::{mpsc, Notify, OnceCell},
    task::AbortHandle,
    time,
};

pub(crate) use super::runtime::COLD_PREFILL_TOKENS_PER_SEC;
use super::{
    engine_admin::{
        admin_backend_client, admin_grpc_result, admin_http_post, FLUSH_HTTP_TIMEOUT,
        PROFILE_HTTP_TIMEOUT,
    },
    event::WorkerConnected,
    http_client::LazyHttpClient,
    pd_pairing::PdPairing,
    runtime::{routing_state_series, CIRCUIT_OPEN_CODE, ROUTING_STATE_SERIES},
    CircuitBreaker, ResolvedResilience, WorkerError, WorkerResult,
};
pub use super::{
    load::{WorkerLoadGuard, WorkerRoutingKeyLoad},
    metadata::WorkerMetadata,
    runtime::{RoutingState, StallReason, WorkerRuntime},
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::grpc::{backend_client::BackendClient, client::GrpcClient, zmq_client},
};

/// Connect the ZMQ backend for this worker URL. All connect mechanics —
/// address derivation, socket-dir prep, EOS resolution, handshake — live in
/// the client layer ([`zmq_client::connect_for_worker`]); the worker layer
/// wraps the client into its slot and error type, after checking the role
/// each engine declared at the handshake against `worker_type`.
#[expect(
    clippy::too_many_arguments,
    reason = "the worker's connection inputs, captured by the detached connect driver"
)]
async fn connect_zmq_backend(
    base_url: String,
    model_id: String,
    runtime: RuntimeType,
    worker_type: WorkerType,
    handshake_override: Option<String>,
    engine_count: zmq_client::EngineCount,
    startup: Duration,
    request_timeout: Option<Duration>,
) -> WorkerResult<Arc<BackendClient>> {
    zmq_client::connect_for_worker(
        &base_url,
        model_id,
        runtime,
        handshake_override.as_deref(),
        engine_count,
        startup,
        request_timeout,
    )
    .await
    .and_then(|client| {
        client.check_declared_role(worker_type)?;
        Ok(Arc::new(BackendClient::Zmq(client)))
    })
    .map_err(|reason| WorkerError::ConnectionFailed {
        url: base_url,
        reason,
    })
}

/// Default bootstrap port for PD disaggregation (used by SGLang and vLLM Mooncake)
pub const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

/// vLLM Mooncake KV connector name
pub const MOONCAKE_CONNECTOR: &str = "MooncakeConnector";

/// vLLM NIXL KV connector name
pub const NIXL_CONNECTOR: &str = "NixlConnector";

/// vLLM MoRI-IO KV connector name
pub const MORIIO_CONNECTOR: &str = "MoRIIOConnector";

/// Worker label naming a MoRI-IO engine's transfer mode, `read` or `write`.
pub const MORIIO_MODE_LABEL: &str = "moriio_mode";

/// Context window advertised for `model_id` in `models`: the matching card's
/// (aliases included), else the primary card's. Borrows only, so it is safe on
/// the per-request path.
fn context_length_from(models: &WorkerModels, model_id: &str) -> Option<u32> {
    models
        .find(model_id)
        .or_else(|| models.primary())
        .and_then(|card| card.context_length)
}

/// The labels that name a worker's running window, in the order the reader
/// (`Worker::max_running_requests`) prefers them.
pub(crate) const WINDOW_LABELS: [&str; 2] = ["max_num_seqs", "max_running_requests"];

/// A worker's labels after a merge: `current` (the discovered labels at
/// registration, the worker's own on a label update) with `by_hand` (the
/// spec's own labels, the request's) on top. A window the operator set by
/// hand is the worker's only window: a window under the other name is
/// dropped, since the reader prefers `max_num_seqs` to `max_running_requests`
/// and would otherwise take the engine's figure over the operator's, on
/// every worker, expanded per rank or not. Registration, the admin label
/// update and its early cap check all merge through here.
pub(crate) fn merge_labels(
    current: &HashMap<String, String>,
    by_hand: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut labels = current.clone();
    if WINDOW_LABELS.iter().any(|key| by_hand.contains_key(*key)) {
        for key in WINDOW_LABELS {
            if !by_hand.contains_key(key) {
                labels.remove(key);
            }
        }
    }
    for (key, value) in by_hand {
        labels.insert(key.clone(), value.clone());
    }
    labels
}

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug + 'static {
    /// Downcast support for same-URL replace state sharing.
    fn as_any(&self) -> &dyn Any;

    /// Get the worker's URL
    fn url(&self) -> &str;
    /// The URL as a request's success path shows it (the routed-worker
    /// response header, the cache-trace header, the request-sent event): the
    /// registered URL byte for byte when it has no userinfo, query or
    /// fragment, otherwise `Endpoint::redacted_str`'s form (see
    /// `Endpoint::redacted_url`). Computed once at construction
    /// (`WorkerMetadata::redacted_url`).
    fn redacted_url(&self) -> &str {
        &self.metadata().redacted_url
    }
    /// Get the worker's API key
    fn api_key(&self) -> Option<&String>;
    /// Get the worker's type (Regular, Prefill, or Decode)
    /// Returns a reference to avoid cloning on every access
    fn worker_type(&self) -> &WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    /// Returns a reference to avoid cloning on every access
    fn connection_mode(&self) -> &ConnectionMode;

    /// Get the worker's lifecycle status.
    fn status(&self) -> WorkerStatus;

    /// Get the current monotonic worker revision.
    ///
    /// Same-URL `replace()` increments the revision so stale probe outcomes
    /// can be discarded without mutating the newly installed worker object.
    fn revision(&self) -> u64 {
        0
    }

    /// Epochs reported during metadata discovery, invalidated after a health
    /// failure. These are observations, not proof against an undetected restart.
    fn cache_trace_epochs(&self) -> Vec<String> {
        Vec::new()
    }

    /// Set the worker's lifecycle status.
    fn set_status(&self, status: WorkerStatus);

    /// Adopt shared mutable runtime state from a previous worker object.
    ///
    /// Used by same-URL `replace()` so in-flight traffic and counters remain
    /// attached to a single shared runtime across the old and new objects.
    fn inherit_shared_state_from(&self, _other: &dyn Worker) -> bool {
        false
    }

    /// Check if the worker is currently healthy (status == Ready).
    ///
    /// This is a routing predicate — returns true only for `Ready` workers.
    /// A `Pending` worker is not "unhealthy", just unverified.
    fn is_healthy(&self) -> bool {
        self.status() == WorkerStatus::Ready
    }

    /// Perform an async health check on the worker.
    ///
    /// Pure probe — does not mutate worker status, does not increment
    /// counters beyond `consecutive_*` totals exposed via the accessors
    /// below. The state machine lives in `WorkerManager`, which reads the
    /// counters and applies transitions via `WorkerRegistry::transition_status()`.
    async fn check_health_async(&self) -> WorkerResult<()>;

    // ── Health check counter accessors (used by WorkerManager state machine) ──

    /// Increment `consecutive_failures` and return the new value.
    fn consecutive_failures_increment(&self) -> usize;

    /// Reset `consecutive_failures` to 0.
    fn consecutive_failures_reset(&self);

    /// Increment `consecutive_successes` and return the new value.
    fn consecutive_successes_increment(&self) -> usize;

    /// Reset `consecutive_successes` to 0.
    fn consecutive_successes_reset(&self);

    /// Read `total_pending_probes` (lifetime probe attempts in Pending state).
    fn total_pending_probes(&self) -> usize;

    /// Increment `total_pending_probes` and return the new value.
    fn total_pending_probes_increment(&self) -> usize;

    /// Reset `total_pending_probes` to 0 (called when promoting Pending → Ready).
    fn total_pending_probes_reset(&self);

    /// Get the current load (number of active requests)
    fn load(&self) -> usize;

    /// Increment the load counter
    fn increment_load(&self);

    /// Increment the load counter only when the resulting value does not exceed `max`.
    ///
    /// Implementations backed by an atomic counter should override this method
    /// with a compare-and-update operation. This fallback preserves the limit
    /// for existing external implementations, but can reject usable capacity
    /// during concurrent attempts.
    fn try_increment_load(&self, max: usize) -> bool {
        self.increment_load();
        if self.load() <= max {
            true
        } else {
            self.decrement_load();
            false
        }
    }

    /// Decrement the load counter
    fn decrement_load(&self);

    /// The request-completion observer installed on this worker, if any
    /// (see [`RequestCompletionSink`]).
    fn completion_sink(&self) -> Option<Arc<dyn RequestCompletionSink>> {
        None
    }

    /// Install (or clear) the request-completion observer.
    fn set_completion_sink(&self, _sink: Option<Arc<dyn RequestCompletionSink>>) {}

    /// Claim `count` PD bootstrap rooms while the worker's claimed total
    /// stays within `window`; a refusal claims nothing.
    ///
    /// The PD admission gate reserves through this rather than comparing
    /// [`Worker::load`] to the window: a read-then-send lets two dispatches
    /// both take the last free slot, which is exactly the over-window burst
    /// the gate exists to prevent.
    ///
    /// The defaults admit unconditionally and track nothing. They exist for
    /// implementations that carry no shared runtime (the language bindings,
    /// test doubles): the gate only reaches a worker that reports a running
    /// window, and any implementation that does report one must override
    /// these three with a real claim, or the window is not enforced (and
    /// the three queue methods after them with a real order, or the wait
    /// is not first come, first served).
    fn try_admit_pd(&self, _count: usize, _window: usize) -> bool {
        true
    }

    /// Release `count` rooms claimed by [`Worker::try_admit_pd`].
    fn release_pd(&self, _count: usize) {}

    /// Rooms currently claimed on this worker.
    fn pd_admitted(&self) -> usize {
        0
    }

    /// Take a place at the back of this worker's admission queue for a
    /// dispatch that needs `rooms`. The gate lets it claim only while no older
    /// place needs as few rooms ([`Worker::pd_may_claim`]), which is what
    /// hands a freed room to the request that has waited longest for one;
    /// the place is given back through [`Worker::pd_dequeue`] however the
    /// wait ends.
    fn pd_enqueue(&self, _rooms: usize) -> u64 {
        0
    }

    /// Whether `place` may claim: no older place still waiting needs as few
    /// rooms as it does, or fewer. The default never queues, so it always may.
    fn pd_may_claim(&self, _place: u64) -> bool {
        true
    }

    /// Give `place` back, whether its dispatch claimed, was shed, or stopped
    /// waiting early; the places behind it move up.
    fn pd_dequeue(&self, _place: u64) {}

    /// The notifier of `place`, fired when it is worth trying again: rooms
    /// were released, or a place ahead of it left. `None` (the default, or a
    /// place already given back) leaves the gate to its retry interval.
    fn pd_wake(&self, _place: u64) -> Option<Arc<Notify>> {
        None
    }

    /// Get the current routing-key load cardinality.
    fn routing_key_load(&self) -> usize;

    /// In-flight requests for one routing key on this worker.
    fn routing_key_inflight(&self, routing_key: &str) -> usize;

    /// Increment the routing-key load tracker for an active key.
    fn increment_routing_key_load(&self, routing_key: &str);

    /// Decrement the routing-key load tracker for a completed key.
    fn decrement_routing_key_load(&self, routing_key: &str);

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// The PD pairing descriptor placement compares across a prefill and a
    /// decode.
    fn pd_pairing(&self) -> &PdPairing {
        &self.metadata().pd_pairing
    }

    /// Worker-reported in-flight capacity, if available.
    ///
    /// Reads the running-window label populated by the metadata discovery
    /// pipeline (Step 4 of the worker lifecycle). Returns `None` when the
    /// worker hasn't reported a value or reports zero (zero is meaningless
    /// for capacity accounting).
    ///
    /// Engines spell the same window two ways — TokenSpeed and vLLM
    /// advertise `max_num_seqs`, SGLang advertises `max_running_requests` —
    /// and both are the count of requests the scheduler will run at once, so
    /// both are read here rather than leaving TokenSpeed workers looking like
    /// non-reporters. `max_num_seqs` wins when a worker reports both: it is
    /// the authoritative name for the engines that use it, and it is the
    /// order `smg_grpc_servicer.tokenspeed.loads::running_window` reports in,
    /// so the label and the `GetLoads` report cannot disagree.
    ///
    /// `WorkerCapacity` uses this to derive total fleet capacity when
    /// every worker reports; falls back to a configured per-worker
    /// estimate otherwise. The PD admission gate uses it as the decode
    /// side's admission bound.
    fn max_running_requests(&self) -> Option<u16> {
        let labels = &self.metadata().spec.labels;
        WINDOW_LABELS
            .iter()
            .find_map(|key| labels.get(*key))
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            // A figure wider than the count (an explicit flag above 65535 on
            // a large engine) saturates: a window that wide is the widest the
            // gate and the capacity sum can use, not no window at all.
            .map(|n| u16::try_from(n).unwrap_or(u16::MAX))
    }

    /// Whether this worker can serve the Realtime API (WS/WebRTC/REST
    /// relay). Reads the `realtime` label (`"true"`) populated via
    /// discovery, worker registration, or static config. Defaults to
    /// `false` so only explicitly-marked workers receive realtime traffic.
    fn is_realtime_capable(&self) -> bool {
        self.metadata()
            .spec
            .labels
            .get("realtime")
            .is_some_and(|v| v == "true")
    }

    /// Get the current circuit breaker state for observability/debugging.
    fn circuit_breaker_state(&self) -> super::circuit_breaker::CircuitState;

    /// Check whether the current circuit breaker state allows execution.
    fn circuit_breaker_can_execute(&self) -> bool;

    /// Record a request outcome against the circuit breaker.
    fn record_circuit_breaker_outcome(&self, success: bool);

    /// Check if the worker is available (healthy + circuit closed/half-open +
    /// not vetoed by the absolute overload guard).
    fn is_available(&self) -> bool {
        self.is_healthy()
            && self.circuit_breaker_can_execute()
            && !self.is_overloaded()
            && self.stall_reason().is_none()
    }

    /// [`Self::is_healthy`] fused with the overload veto. For the hash policies,
    /// which route on health alone and never consult the circuit breaker;
    /// `BasicWorker` overrides it to read both under a single runtime guard.
    fn is_healthy_and_eligible(&self) -> bool {
        self.is_healthy() && !self.is_overloaded() && self.stall_reason().is_none()
    }

    /// Publish the worker's routing state as the `smg_worker_state` series:
    /// 1 on the state and reason the routing decision reads now, 0 on the
    /// pair it left; nothing when the pair did not move. Called where a fact
    /// of the decision moves (the status, the overload veto, the liveness
    /// veto, the circuit breaker). A no-op for a worker without the series.
    fn publish_state(&self) {}

    /// [`Self::publish_state`] for an address whose series were dropped and
    /// come back (a registration): the current pair is set whether or not it
    /// was the one published before.
    fn republish_state(&self) {}

    /// Whether the absolute overload guard currently vetoes this worker.
    ///
    /// Written only by the load monitor, once per ingested load report, and
    /// always `false` while overload protection is unconfigured.
    fn is_overloaded(&self) -> bool {
        false
    }

    /// Set the overload veto, returning `true` when the flag actually changed.
    ///
    /// Route writes through [`WorkerRegistry::set_worker_overloaded`](crate::worker::WorkerRegistry::set_worker_overloaded) instead of
    /// calling this directly: the per-model counters and the
    /// `smg_workers_overloaded` gauge move only on transitions.
    fn set_overloaded(&self, _overloaded: bool) -> bool {
        false
    }

    /// Why the liveness tracker vetoes this worker, if it does.
    fn stall_reason(&self) -> Option<StallReason> {
        None
    }

    /// Set or clear the liveness veto, returning `true` when it changed.
    /// Route writes through the `liveness` module, which logs and counts the
    /// transition.
    fn set_stall(&self, _reason: Option<StallReason>) -> bool {
        false
    }

    /// Record a successful interaction with the worker (a poll answered, a
    /// probe passed, an event batch, a response).
    fn note_contact(&self) {}

    /// Record a token or a completion from the worker.
    fn note_token_progress(&self) {}

    /// [`Self::note_token_progress`] at `now_ms` on the gateway clock
    /// (`liveness::now_ms`): a stamp the caller read once for every note of
    /// the same message.
    fn note_token_progress_at(&self, _now_ms: u64) {
        self.note_token_progress();
    }

    /// Record a transport failure (a poll, probe or stream that failed on the
    /// connection); cleared by the next contact.
    fn note_transport_failure(&self) {}

    /// Whether a transport failure happened since the last contact.
    fn transport_failure_pending(&self) -> bool {
        false
    }

    /// Time since the last successful contact.
    fn contact_age(&self) -> Duration {
        Duration::ZERO
    }

    /// Time since the last token or completion.
    fn token_progress_age(&self) -> Duration {
        Duration::ZERO
    }

    /// Record a load record pushed on the worker's own KV-event stream: from
    /// here on the worker is expected to be heard from at the stream's
    /// cadence (see the `liveness` module).
    fn note_load_record(&self) {}

    /// Whether the worker's stream has ever pushed a load record.
    fn pushes_load_records(&self) -> bool {
        false
    }

    /// Record whether the worker's latest load record reports its engine
    /// silent on its in-flight requests for the wedge bound (see the
    /// `liveness` module).
    fn note_engine_stall(&self, _stalled: bool) {}

    /// Whether the latest load record reported the engine stalled.
    fn engine_stalled(&self) -> bool {
        false
    }

    /// The prefill rate observed on this worker, tokens per second; zero
    /// until a window of first tokens has been seen.
    fn prefill_rate_tps(&self) -> u64 {
        0
    }
    /// Record the start of a request whose responses the gateway sees one by
    /// one (a streaming generation to this worker over gRPC): the pile the
    /// wedged rule counts, and the start of its clock when a run begins.
    /// Requests without that signal (HTTP, a PD side, a non-streaming
    /// generation) never form a pile.
    fn note_tracked_started(&self) {}
    /// Record the end of a tracked request.
    fn note_tracked_ended(&self) {}
    /// Requests in flight whose responses the gateway sees one by one.
    fn tracked_load(&self) -> usize {
        0
    }
    /// Record the first token or completion of a stream from this worker:
    /// from here until its end the stream is one the worker is producing for.
    fn note_output_started(&self) {}
    /// Record the end of a stream that had produced output.
    fn note_output_ended(&self) {}
    /// How long the worker has produced nothing while it still holds streams
    /// it had been producing for: `Some(age)` when at least one stream past
    /// its first token or completion is open and the last token or
    /// completion from the worker was `age` ago; `None` when no such stream
    /// is open (an idle worker, one whose streams all ended, one whose
    /// in-flight work has shown nothing yet: nothing to judge it by), or for
    /// a worker that does not track it. The signal a disaggregated request
    /// reads on its decode partner when its prefill side's silence bound
    /// passes on the parallel dispatch shape, where the prefill's first
    /// response waits for the decode side. The age is that of the last
    /// response the gateway read, not of the engine's last production: a
    /// stream whose reader stops polling (a client that stays connected and
    /// stops reading its bounded channel) keeps its open count and ages, so
    /// a decode whose only open producing streams belong to such readers
    /// can read as silent while a frozen prefill is the cause; the two
    /// conditions together are the limit of this signal.
    fn output_silence(&self) -> Option<Duration> {
        None
    }

    /// Store the engine's reported waiting-queue depth, returning the previous
    /// value.
    fn swap_waiting_reqs(&self, _waiting: i64) -> i64 {
        0
    }

    /// Store the in-flight count seen by the liveness sweep, returning the
    /// previous sample.
    fn swap_load_sample(&self, _load: usize) -> usize {
        0
    }

    /// A notifier fired on every contact with the worker, for loops that back
    /// off from it and should retry as soon as it is heard from again.
    fn contact_wake(&self) -> Option<Arc<Notify>> {
        None
    }

    /// Ask the health manager to promote the worker now, on the strength of a
    /// successful contact, instead of at its next scheduled probe.
    fn signal_connected(&self) {}

    /// Close the circuit breaker: a liveness veto was cleared by a contact,
    /// so the failures that opened it belong to the outage. It reopens on
    /// fresh failures like any closed breaker. A health promotion goes
    /// through [`Self::readmit_circuit_breaker`] instead: an open breaker is
    /// put on trial, not closed.
    fn reset_circuit_breaker(&self) {}

    /// The worker's health returned: put its open circuit breaker on trial
    /// (half-open) now, as its timeout would, or forget the failure streak a
    /// still-closed breaker banked during the outage. Returns whether an
    /// open breaker went on trial; a worker without a breaker never does.
    fn readmit_circuit_breaker(&self) -> bool {
        false
    }

    /// Record that the worker just became routable (promotion to Ready, or a
    /// liveness veto cleared); the warm-up slice counts from here.
    fn note_admitted(&self) {}

    /// Time since the worker last became routable; `Duration::MAX` for a
    /// worker that does not track it (never warming).
    fn admitted_age(&self) -> Duration {
        Duration::MAX
    }

    /// Blocks the worker's index has gained since its current admission, for
    /// the warm-up slice; `indexed` is what the index holds for it now. A
    /// worker that does not track admissions reports the count itself.
    fn warmup_growth(&self, indexed: usize) -> usize {
        indexed
    }

    /// Gateway clock (`liveness::now_ms`) before which a thin worker with
    /// requests in flight takes no further diverted hit (see
    /// `CacheAwarePolicy::warmup_divert`); zero for a worker that does not
    /// track it.
    fn divert_until_ms(&self) -> u64 {
        0
    }

    /// A hit was diverted to this thin worker; the next one waits until
    /// `until_ms` while it has anything in flight.
    fn note_diverted(&self, _until_ms: u64) {}

    /// A request of `tokens` prompt tokens was dispatched: work the engine
    /// still has to prefill before its first token (see
    /// [`Self::prefill_backlog`]). `timed` says the gateway will see that
    /// first token on its own (a streaming generation), so the prompt's
    /// prefill can be timed for the worker's rate; a non-streaming
    /// generation answers once, at its end, and its prompt counts in the
    /// backlog only.
    fn note_prefill_started(&self, _tokens: u64, _timed: bool) {}

    /// [`Self::note_prefill_started`] at `now_ms` on the gateway clock: the
    /// dispatch's one stamp.
    fn note_prefill_started_at(&self, tokens: u64, timed: bool, _now_ms: u64) {
        self.note_prefill_started(tokens, timed);
    }

    /// The request's first token came back, with `cached` of its prompt
    /// tokens reported served from the engine's prefix cache (`Some`; zero
    /// when the engine reports no count), or its stream ended or was dropped
    /// before one (`None`): its prompt is no longer pending prefill. `timed`
    /// as at its start.
    fn note_prefill_ended(&self, _tokens: u64, _timed: bool, _cached: Option<u64>) {}

    /// [`Self::note_prefill_ended`] at `now_ms` on the gateway clock: the
    /// first response's one stamp.
    fn note_prefill_ended_at(&self, tokens: u64, timed: bool, cached: Option<u64>, _now_ms: u64) {
        self.note_prefill_ended(tokens, timed, cached);
    }

    /// How long the engine may still need before the first token of its
    /// in-flight work is due: the prompt tokens dispatched and not yet
    /// answered, over the prefill rate observed on this worker (a cold prior
    /// until one is). Zero for a worker that does not track it.
    fn prefill_backlog(&self) -> Duration {
        Duration::ZERO
    }

    /// [`Self::prefill_backlog`] with `tokens` more prompt tokens about to
    /// be dispatched: how long the engine may need before the first token of
    /// a prompt of that length is due, behind what is already pending. Zero
    /// for a worker that does not track it.
    fn prefill_eta(&self, _tokens: u64) -> Duration {
        Duration::ZERO
    }

    /// How long ago a prompt dispatched to the worker was last answered with
    /// its first token (or, for a non-streaming generation, its one answer):
    /// the worker's latest first response, the evidence that it answers at
    /// all. `None` for a worker that has answered none since the gateway
    /// started, or does not track it. A handoff is not an answer (see
    /// [`Self::prefill_handed_off_ago`]).
    fn prefill_answered_ago(&self) -> Option<Duration> {
        None
    }

    /// A decode side showed a handoff of this prefill worker's: on the
    /// parallel dispatch shape, its first output, which the decode engine
    /// produces from the KV the worker handed off. Evidence for
    /// `liveness::prefill_side_reprieve` only: no prompt is released,
    /// nothing is timed, no run is answered.
    fn note_prefill_handoff(&self) {}

    /// How long ago a decode side last showed a handoff of this worker's
    /// ([`Self::note_prefill_handoff`]); `None` for a worker that has shown
    /// none since the gateway started, or does not track it.
    fn prefill_handed_off_ago(&self) -> Option<Duration> {
        None
    }

    /// How long the worker's current run of pending prompts (since its
    /// pending backlog last left zero) has waited for its first answer,
    /// plus the time earlier runs since its latest first response held
    /// prompts and ended unanswered (given up on, or dropped): its silence,
    /// the spells it held nothing left out. `None` while nothing is pending,
    /// once the run has had an answer, or for a worker that does not track
    /// its runs.
    fn prefill_unanswered_run(&self) -> Option<Duration> {
        None
    }

    /// Whether the engine behind the worker is on a prompt shape it has not
    /// shown it runs: a prompt pending on the worker, or one of `tokens`
    /// about to be dispatched, longer than any the worker has answered since
    /// the gateway started, on a worker that has answered at least one. A
    /// worker that has answered nothing has shown no shape to go by, and is
    /// judged like any other; so is a worker that does not track it
    /// (`false`).
    fn prefill_cold_shape(&self, _tokens: u64) -> bool {
        false
    }

    /// One-shot routing snapshot for the per-request O(workers) selection loops:
    /// reads status, load, processed and the overload veto together so the hot
    /// path takes one `ArcSwap` guard per backing cell per worker instead of one
    /// per accessor (that guard traffic is a large share of routing CPU at
    /// scale). `BasicWorker` overrides this to share the runtime guard.
    fn routing_state(&self) -> RoutingState {
        RoutingState {
            healthy: self.is_healthy(),
            can_execute: self.circuit_breaker_can_execute(),
            load: self.load(),
            processed: self.processed_requests(),
            overloaded: self.is_overloaded(),
            stalled: self.stall_reason().is_some(),
        }
    }

    /// Record the outcome of a request based on the HTTP status code.
    ///
    /// Statuses in the per-worker `capacity_status_codes` set (default: 429)
    /// record nothing at all — neither failure nor success. Any other status
    /// is a circuit-breaker failure when it appears in `retryable_status_codes`,
    /// which by default leaves 408, 500, 502, 503 and 504 tripping the breaker.
    /// 429 is in that set too, but the capacity check returns before it is read.
    /// Callers just pass the status — no need to interpret it.
    ///
    /// For transport/connection errors where no HTTP response is received,
    /// pass the status code returned to the client (e.g., 502 for a send
    /// error, 504 for a timeout).
    fn record_outcome(&self, status_code: u16) {
        // Capacity pushback (429 by default) is a routing signal, not a
        // worker fault: the request is retried elsewhere, but no
        // circuit-breaker sample is recorded in either direction — opening
        // the breaker on backpressure would amplify a load spike into
        // unavailability, and crediting a success would close a half-open
        // breaker on a request the worker refused.
        if self
            .resilience()
            .capacity_status_codes
            .contains(&status_code)
        {
            return;
        }
        self.record_circuit_breaker_outcome(!self.is_breaker_failure(status_code));
    }

    /// Whether `status_code` counts as a circuit-breaker failure for this
    /// worker: a status in its `retryable_status_codes` that is not capacity
    /// pushback. The per-request accounting in
    /// `routers::common::attempt_ledger` asks this before charging, so a
    /// request's retries charge a worker once.
    fn is_breaker_failure(&self, status_code: u16) -> bool {
        let resilience = self.resilience();
        !resilience.capacity_status_codes.contains(&status_code)
            && resilience.retryable_status_codes.contains(&status_code)
    }

    /// Get the resolved resilience config for this worker.
    fn resilience(&self) -> &ResolvedResilience;

    /// Get the per-worker HTTP client.
    fn http_client(&self) -> &reqwest::Client;

    /// Strong handle to the worker's HTTP client, if one was materialized.
    /// Must not force a deferred client into existence; cache-fed workers
    /// return the shared handle so a replacement worker keeps the cache
    /// entry alive.
    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>>;

    /// Strong handle to the worker's backend client (gRPC or ZMQ), if its
    /// lazy connect landed. Never drives the connect: a worker still
    /// connecting, or an HTTP worker, has none.
    fn backend_client_if_initialized(&self) -> Option<Arc<BackendClient>> {
        None
    }

    // ── Metadata convenience delegates ──────────────────────────────
    //
    // These default impls forward to the canonical implementation on
    // [`WorkerMetadata`] so callers can write `worker.foo()` instead
    // of the longer `worker.metadata().foo()`. Adding a new metadata
    // accessor means adding it to `impl WorkerMetadata` first and
    // then forwarding it here. Implementors of `Worker` should never
    // need to override these — `BasicWorker` and the FFI workers
    // both rely on the defaults.

    /// Get the bootstrap hostname for PD mode.
    fn bootstrap_host(&self) -> &str {
        self.metadata().bootstrap_host()
    }

    /// Get the bootstrap port for PD mode.
    fn bootstrap_port(&self) -> Option<u16> {
        self.metadata().bootstrap_port()
    }

    /// Get the base URL without any DP rank suffix.
    fn base_url(&self) -> &str {
        self.metadata().base_url()
    }

    /// Compose an endpoint URL for a specific route.
    fn endpoint_url(&self, route: &str) -> String {
        self.metadata().endpoint_url(route)
    }

    /// Whether the router speaks HTTP/2 prior knowledge to this worker.
    fn http2(&self) -> bool {
        self.metadata().http2()
    }

    /// Check if this worker is DP-aware.
    fn is_dp_aware(&self) -> bool {
        self.metadata().is_dp_aware()
    }

    /// Get DP rank if this is a DP-aware worker.
    fn dp_rank(&self) -> Option<usize> {
        self.metadata().dp_rank()
    }

    /// Get DP size if this worker is part of a DP group.
    fn dp_size(&self) -> Option<usize> {
        self.metadata().dp_size()
    }

    /// The KV transfer engine id the worker's engine currently reports: the
    /// spec's, unless a recovery re-read it (see
    /// [`Self::refresh_kv_engine_id`]). A PD handoff must be minted for the
    /// engine process that is there now, not the one discovered at
    /// registration (#2491).
    fn kv_engine_id(&self) -> Option<String> {
        self.metadata().spec.kv_engine_id.clone()
    }

    /// Replace the engine id after a re-read. Returns `true` when it changed.
    fn refresh_kv_engine_id(&self, _kv_engine_id: Option<String>) -> bool {
        false
    }

    /// Whether the engine id in force has been confirmed by the engine since
    /// the worker last recovered. `false` after a re-read that failed,
    /// expired or returned no id, so the next probe tries again instead of
    /// the worker serving with a possibly stale id until its next outage.
    fn kv_engine_id_confirmed(&self) -> bool {
        true
    }

    /// Record the outcome of a re-read (see [`Self::kv_engine_id_confirmed`]).
    fn set_kv_engine_id_confirmed(&self, _confirmed: bool) {}

    /// Transform a request for DP-aware routing.
    ///
    /// When the worker has a `dp_rank`, injects `data_parallel_rank`
    /// into the request body. Otherwise returns the request unchanged.
    ///
    /// Any override that edits the body must also override
    /// [`Worker::mutates_request`] to return `true`.
    fn prepare_request(&self, req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        self.metadata().prepare_request(req)
    }

    /// Whether [`Worker::prepare_request`] rewrites the body. The HTTP proxy
    /// path skips the `serde_json::Value` round-trip when this is `false`.
    fn mutates_request(&self) -> bool {
        self.metadata().mutates_request()
    }

    /// Whether [`Worker::prepare_request`] is the built-in edit, the
    /// `data_parallel_rank` insert for a DP-aware worker, and nothing else.
    /// The HTTP proxy path then applies that edit on raw JSON slices instead
    /// of round-tripping the body through `serde_json::Value`. An
    /// implementation with its own `prepare_request` keeps the default,
    /// `false`, so its edits and errors still run.
    fn uses_builtin_prepare_request(&self) -> bool {
        false
    }

    /// Get the model ID this worker serves.
    fn model_id(&self) -> &str {
        self.metadata().model_id()
    }

    /// Get the priority of this worker (higher value = higher priority).
    fn priority(&self) -> u32 {
        self.metadata().priority()
    }

    /// Get the cost factor of this worker (baseline = 1.0).
    fn cost(&self) -> f32 {
        self.metadata().cost()
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    fn default_provider(&self) -> Option<&ProviderType> {
        self.metadata().default_provider()
    }

    /// Get the provider for a specific model. Priority:
    /// `ModelCard.provider` > `worker.default_provider()`.
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.metadata().provider_for_model(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to LLM capabilities if the model is not registered.
    fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        self.metadata().supports_endpoint(model_id, endpoint)
    }

    /// Check if this worker supports a specific model.
    ///
    /// `BasicWorker` overrides this to consult its lazy-discovered
    /// `models_override`; the default delegates to the underlying
    /// [`WorkerMetadata::supports_model`].
    fn supports_model(&self, model_id: &str) -> bool {
        self.metadata().supports_model(model_id)
    }

    /// Get all models this worker can serve.
    fn models(&self) -> Vec<ModelCard> {
        self.metadata().spec.models.all().to_vec()
    }

    /// Whether any model card of this worker satisfies `pred`: the same
    /// cards [`Self::models`] returns, read in place. For a check on the
    /// request path, which [`Self::models`] would answer by cloning every
    /// card.
    fn any_model(&self, pred: &mut dyn FnMut(&ModelCard) -> bool) -> bool {
        self.metadata().spec.models.all().iter().any(pred)
    }

    /// Context window (in tokens) this worker advertises for `model_id`, from
    /// the matching model card (falling back to the primary card). `None`
    /// when the worker never advertised one; callers then leave the length
    /// check to the engine.
    ///
    /// `BasicWorker` overrides this to consult its lazy-discovered
    /// `models_override`, as `supports_model` does.
    fn context_length(&self, model_id: &str) -> Option<u32> {
        context_length_from(&self.metadata().spec.models, model_id)
    }

    /// Set models for this worker (for lazy discovery).
    /// Default implementation does nothing - only BasicWorker supports this.
    fn set_models(&self, _models: Vec<ModelCard>) {
        // Default: no-op. BasicWorker overrides this.
    }

    /// Check if models have been discovered for this worker.
    /// Returns true if models were set via set_models() or if metadata has models.
    fn has_models_discovered(&self) -> bool {
        !self.metadata().spec.models.is_wildcard()
    }

    /// Get or create a gRPC client for this worker
    /// Returns None for HTTP workers, Some(client) for gRPC workers
    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>>;

    /// Reset the gRPC client connection (for reconnection scenarios)
    /// No-op for HTTP workers
    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        Ok(())
    }

    /// Release background work owned by this worker instance.
    ///
    /// Called by the registry when the worker leaves it (removal) or is
    /// superseded by a replacement. `BasicWorker` aborts the detached ZMQ
    /// handshake driver: it holds the handshake and data-plane socket binds
    /// until it lands, so an orphaned driver would keep them for up to the
    /// connect timeout and collide with a same-URL re-registration.
    fn abort_background_tasks(&self) {}
    async fn grpc_health_check(&self) -> WorkerResult<bool>;
    /// Liveness check for a ZMQ worker. Unlike gRPC there is no health RPC on
    /// the raw wire: liveness is local (handshake completed and the engine has
    /// not signalled `ENGINE_CORE_DEAD`).
    async fn zmq_health_check(&self) -> WorkerResult<bool>;
    async fn http_health_check(&self) -> WorkerResult<bool>;

    // ── Admin operations ────────────────────────────────────────────
    //
    // Dual-path dispatch on connection mode, mirroring
    // `check_health_async`: HTTP workers receive a POST to the engine's
    // native admin endpoint, gRPC workers receive the corresponding RPC.
    // Backends without the RPC surface the dispatcher's `Unimplemented`
    // status as an `OperationFailed` error.

    /// Flush the KV cache on this worker's backend.
    async fn flush_cache(&self) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/flush_cache"),
                    self.api_key(),
                    None,
                    "flush_cache",
                    FLUSH_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                let client = admin_backend_client(
                    self.url(),
                    "flush_cache",
                    self.get_backend_client().await?,
                )?;
                let result = client.flush_cache(0.0).await;
                admin_grpc_result(
                    self.url(),
                    "flush_cache",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }

    /// Start a profiling run on this worker's backend.
    async fn start_profile(&self, options: &ProfileOptions) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                let body =
                    serde_json::to_value(options).map_err(|e| WorkerError::OperationFailed {
                        url: self.url().to_string(),
                        operation: "start_profile".to_string(),
                        reason: format!("failed to serialize profile options: {e}"),
                    })?;
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/start_profile"),
                    self.api_key(),
                    Some(body),
                    "start_profile",
                    PROFILE_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                let client = admin_backend_client(
                    self.url(),
                    "start_profile",
                    self.get_backend_client().await?,
                )?;
                let req = common_proto::StartProfileRequest {
                    output_dir: options.output_dir.clone(),
                    start_step: options.start_step,
                    num_steps: options.num_steps,
                    activities: options.activities.clone().unwrap_or_default(),
                    with_stack: options.with_stack,
                    record_shapes: options.record_shapes,
                    profile_by_stage: options.profile_by_stage,
                };
                let result = client.start_profile(req).await;
                admin_grpc_result(
                    self.url(),
                    "start_profile",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }

    /// Stop the in-flight profiling run on this worker's backend and
    /// export traces.
    async fn stop_profile(&self) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/stop_profile"),
                    self.api_key(),
                    None,
                    "stop_profile",
                    PROFILE_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                let client = admin_backend_client(
                    self.url(),
                    "stop_profile",
                    self.get_backend_client().await?,
                )?;
                let result = client.stop_profile().await;
                admin_grpc_result(
                    self.url(),
                    "stop_profile",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }
}

/// Extension trait for model_gateway-specific ConnectionMode methods.
pub(crate) trait ConnectionModeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl ConnectionModeExt for ConnectionMode {
    fn as_metric_label(&self) -> &'static str {
        match self {
            ConnectionMode::Http => metrics_labels::CONNECTION_HTTP,
            ConnectionMode::Grpc => metrics_labels::CONNECTION_GRPC,
            ConnectionMode::Zmq => metrics_labels::CONNECTION_ZMQ,
        }
    }
}

/// Extension trait for model_gateway-specific WorkerType methods.
pub(crate) trait WorkerTypeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl WorkerTypeExt for WorkerType {
    fn as_metric_label(&self) -> &'static str {
        match self {
            WorkerType::Regular => metrics_labels::WORKER_REGULAR,
            WorkerType::Prefill => metrics_labels::WORKER_PREFILL,
            WorkerType::Decode => metrics_labels::WORKER_DECODE,
            WorkerType::Encode => metrics_labels::WORKER_ENCODE,
        }
    }
}

/// Observer of request completions on a worker. The request's load guard
/// notifies it when it drops, which is the one point every path that holds
/// worker load (HTTP, gRPC, both PD sides, admitted prefill) passes through on
/// success, error and client disconnect alike. The policy registry installs
/// one on each worker it learns about, so policies that book state at
/// dispatch (reservations, bookings) see the request end.
pub trait RequestCompletionSink: Send + Sync + fmt::Debug {
    fn request_completed(&self, worker: &dyn Worker);
}

/// Basic worker implementation
pub struct BasicWorker {
    pub metadata: WorkerMetadata,
    pub runtime: ArcSwap<WorkerRuntime>,
    pub circuit_breaker: ArcSwap<CircuitBreaker>,
    /// Lazily initialized backend client (gRPC or ZMQ) for local workers.
    /// The inner `OnceCell` keeps reads lock-free after initialization and
    /// dedupes concurrent connects; the outer `ArcSwap` lets the ZMQ health
    /// probe evict a dead client by swapping in a fresh cell (gRPC never
    /// swaps — a failed gRPC worker is removed and re-added instead).
    pub backend_client: Arc<ArcSwap<OnceCell<Arc<BackendClient>>>>,
    /// Guards the one-shot background ZMQ handshake driver so the health probe
    /// never cancels a long (model-load) handshake. Self-clears on failure to
    /// allow a retry. Unused for HTTP/gRPC.
    pub zmq_connect_started: Arc<AtomicBool>,
    /// Abort handle for that driver, so removing or replacing the worker
    /// releases the sockets the in-flight handshake has bound instead of
    /// leaving them held until it times out. `None` until a driver is spawned.
    pub zmq_connect_abort: Arc<ArcSwapOption<AbortHandle>>,
    /// Wakes the manager the instant the ZMQ handshake completes so it can
    /// promote the worker without waiting for the next health poll. Set only
    /// for ZMQ workers built through the registration path; `None` elsewhere
    /// (HTTP/gRPC, tests), where promotion stays poll-driven.
    pub connect_signal_tx: Option<mpsc::UnboundedSender<WorkerConnected>>,
    /// Runtime-mutable models override (for lazy discovery).
    /// When not `Wildcard`, overrides metadata.models for routing decisions.
    /// Uses `ArcSwap` for lock-free reads on the hot path (`supports_model`).
    pub models_override: Arc<ArcSwap<WorkerModels>>,
    /// The KV transfer engine id in force, seeded from the spec and replaced
    /// when a recovered engine reports a new one (see
    /// [`Worker::refresh_kv_engine_id`]). A metadata update shares this state
    /// with the old worker. A fresh discovery starts from its own spec.
    pub kv_engine_id: Arc<ArcSwapOption<String>>,
    /// Set while a recovery re-read of the engine id has not succeeded yet
    /// (see [`Worker::kv_engine_id_confirmed`]).
    pub kv_engine_id_unconfirmed: Arc<AtomicBool>,
    /// Worker-directed HTTP client, shared across same-config workers, built
    /// on first use (see [`LazyHttpClient`]).
    pub http_client: Arc<LazyHttpClient>,
    /// Resolved resilience config (retry + circuit breaker settings).
    pub resilience: ResolvedResilience,
    /// Request-completion observer, installed by the policy registry; shared
    /// by clones so a DP-rank view reports to the same sink.
    pub completion_sink: Arc<std::sync::RwLock<Option<Arc<dyn RequestCompletionSink>>>>,
}

impl Clone for BasicWorker {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            completion_sink: Arc::clone(&self.completion_sink),
            runtime: ArcSwap::from(self.runtime.load_full()),
            circuit_breaker: ArcSwap::from(self.circuit_breaker.load_full()),
            backend_client: Arc::clone(&self.backend_client),
            zmq_connect_started: Arc::clone(&self.zmq_connect_started),
            zmq_connect_abort: Arc::clone(&self.zmq_connect_abort),
            connect_signal_tx: self.connect_signal_tx.clone(),
            models_override: Arc::clone(&self.models_override),
            kv_engine_id: Arc::new(ArcSwapOption::new(self.kv_engine_id.load_full())),
            kv_engine_id_unconfirmed: Arc::new(AtomicBool::new(
                self.kv_engine_id_unconfirmed.load(Ordering::Relaxed),
            )),
            http_client: Arc::clone(&self.http_client),
            resilience: self.resilience.clone(),
        }
    }
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let runtime = self.runtime.load();
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("status", &runtime.status())
            .field("revision", &runtime.revision())
            .field("circuit_breaker_state", &self.circuit_breaker_state())
            .field("grpc_client", &"<OnceCell>")
            .finish()
    }
}

impl BasicWorker {
    /// Keep the live KV state across a metadata update.
    pub(crate) fn share_kv_engine_state(&mut self, previous: &BasicWorker) {
        self.kv_engine_id = Arc::clone(&previous.kv_engine_id);
        self.kv_engine_id_unconfirmed = Arc::clone(&previous.kv_engine_id_unconfirmed);
    }

    /// The circuit breaker configuration this worker runs with: the gateway
    /// defaults plus its own overrides, as resolved at registration.
    pub(crate) fn circuit_breaker_config(&self) -> super::circuit_breaker::CircuitBreakerConfig {
        self.circuit_breaker.load().config().clone()
    }

    fn update_running_requests_metrics(&self) {
        let load = self.load();
        Metrics::set_worker_requests_active(self.url(), load);
    }

    fn shared_runtime(&self) -> Arc<WorkerRuntime> {
        self.runtime.load_full()
    }

    /// Start the one-shot background ZMQ handshake driver for `cell` unless one
    /// is already in flight.
    ///
    /// The handshake is never driven inline: it can take as long as a model
    /// load, so any caller awaiting it (request pipeline, load monitor, health
    /// probe) would block far past its own deadline. Callers peek the cell and
    /// report unavailable until the driver lands.
    fn spawn_zmq_connect_driver(&self, cell: &Arc<OnceCell<Arc<BackendClient>>>) {
        if self.zmq_connect_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let cell = Arc::clone(cell);
        let started = Arc::clone(&self.zmq_connect_started);
        let base_url = self.metadata.base_url().to_string();
        let model_id = self.metadata.model_id().to_string();
        let url = self.metadata.spec.url.clone();
        let runtime = self.metadata.spec.runtime_type;
        let worker_type = self.metadata.spec.worker_type;
        let handshake_override = self.metadata.spec.zmq_handshake_address.clone();
        let engine_count = self.metadata.zmq_engine_count().map_or(
            zmq_client::EngineCount::Declared,
            zmq_client::EngineCount::Exactly,
        );
        let startup = self.metadata.zmq_startup_timeout;
        let request_timeout = self.metadata.zmq_request_timeout;
        // Capture the readiness signal and the revision at hand-off. The
        // manager only promotes if this revision still matches, so a
        // same-URL replacement racing the handshake is discarded.
        let signal_tx = self.connect_signal_tx.clone();
        let revision = self.revision();
        #[expect(
            clippy::disallowed_methods,
            reason = "detached one-shot handshake driver; the OnceCell dedupes with the request path and the guard self-clears on failure to allow a retry"
        )]
        // Detached: only the AbortHandle is kept, so the task keeps running
        // until it completes or the worker leaves the registry.
        let handle = tokio::spawn(async move {
            match cell
                .get_or_try_init(|| {
                    connect_zmq_backend(
                        base_url,
                        model_id,
                        runtime,
                        worker_type,
                        handshake_override,
                        engine_count,
                        startup,
                        request_timeout,
                    )
                })
                .await
            {
                Ok(_) => {
                    // Handshake landed: wake the manager to promote now
                    // rather than on the next poll. A dropped signal (no
                    // manager, or receiver gone) is harmless — polling
                    // still promotes on the success threshold.
                    if let Some(tx) = &signal_tx {
                        let _ = tx.send(WorkerConnected { url, revision });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "ZMQ backend handshake failed for {}: {e}; will retry on the next health probe",
                        super::endpoint::Endpoint::redacted_str(&url)
                    );
                    started.store(false, Ordering::SeqCst);
                }
            }
        });
        self.zmq_connect_abort
            .store(Some(Arc::new(handle.abort_handle())));
    }

    /// Whether the breaker permits execution. A breaker that moves from open
    /// to half-open on this read (its timeout) moves the worker's routing
    /// state with it, and the breaker has no view of that state: the series
    /// is published here, only when the one published was `circuit_open`.
    fn breaker_allows(&self, rt: &WorkerRuntime) -> bool {
        let allows = self.circuit_breaker.load().can_execute();
        if allows && rt.published_state() == CIRCUIT_OPEN_CODE {
            self.publish_state_under(rt, false);
        }
        allows
    }

    /// [`Worker::publish_state`] under the runtime guard the caller holds:
    /// the pair the facts give now is set to 1 and the pair published before
    /// it to 0, when they differ; with `forget_previous` (a registration,
    /// whose address's series were dropped) the pair is set whether or not
    /// it was the one published before. One publish at a time, from the
    /// facts to the last write: every fact is stored before its publish is
    /// called, so the last publish in lock order reads every fact that
    /// moved, and the series end on the pair the facts give. The breaker is
    /// read as stored: its timeout's move belongs to the routing decision's
    /// read ([`Self::breaker_allows`]), which publishes it.
    fn publish_state_under(&self, rt: &WorkerRuntime, forget_previous: bool) {
        let _publish = rt.publish_lock();
        if forget_previous {
            rt.swap_published_state(0);
        }
        let index = routing_state_series(
            rt.status(),
            rt.is_overloaded(),
            rt.stall_reason(),
            self.circuit_breaker.load().peek_state() == super::circuit_breaker::CircuitState::Open,
        );
        let code = index as u8 + 1;
        let previous = rt.swap_published_state(code);
        if previous == code {
            return;
        }
        let (state, reason) = ROUTING_STATE_SERIES[index];
        Metrics::set_worker_state(self.url(), state, reason, true);
        if let Some(left) = (previous as usize).checked_sub(1) {
            let (state, reason) = ROUTING_STATE_SERIES[left];
            Metrics::set_worker_state(self.url(), state, reason, false);
        }
    }

    fn install_shared_state_from_basic(&self, other: &BasicWorker) {
        let shared_runtime = other.shared_runtime();
        shared_runtime.bump_revision();
        let shared_status = shared_runtime.status();
        self.runtime.store(shared_runtime);
        Metrics::set_worker_health(self.url(), shared_status == WorkerStatus::Ready);

        let existing_cb = self.circuit_breaker.load();
        let other_cb = other.circuit_breaker.load_full();
        if other_cb.config() == existing_cb.config() {
            // Building this worker reset the URL's breaker state gauge to
            // closed; the adopted breaker is what monitoring must show.
            other_cb.publish_metrics();
            self.circuit_breaker.store(other_cb);
        }
        self.publish_state();

        self.adopt_backend_client_from(other);
    }

    /// Adopt the replaced worker's backend-client cell so a same-URL
    /// replacement (a metadata-only update, say) keeps talking over the
    /// connection the old worker already established. Decisive for ZMQ: SMG
    /// binds the sockets and the engine handshakes only at its own startup, so
    /// a replacement starting from an empty cell would unbind the live sockets,
    /// rebind, and wait for a HELLO that never arrives.
    ///
    /// `zmq_connect_started` is deliberately left cleared rather than copied:
    /// the shared cell already dedupes a handshake in flight, and a cleared
    /// guard lets the replacement retry (and signal under its own revision) if
    /// that handshake fails.
    fn adopt_backend_client_from(&self, other: &BasicWorker) {
        // A transport or runtime change means a different wire protocol, and a
        // replacement that arrived with its own client keeps it.
        if self.metadata.spec.connection_mode != other.metadata.spec.connection_mode
            || self.metadata.spec.runtime_type != other.metadata.spec.runtime_type
            || self.backend_client.load().get().is_some()
        {
            return;
        }
        self.backend_client.store(other.backend_client.load_full());
    }
}

impl BasicWorker {
    /// The model cards this worker answers for, read in place: the lazily
    /// discovered list when one was set, the metadata's cards otherwise. The
    /// one place that chooses, so `models` and `any_model` cannot drift
    /// apart.
    fn with_model_cards<R>(&self, read: impl FnOnce(&[ModelCard]) -> R) -> R {
        let overridden = self.models_override.load();
        if overridden.is_wildcard() {
            read(self.metadata.spec.models.all())
        } else {
            read(overridden.all())
        }
    }
}

#[async_trait]
impl Worker for BasicWorker {
    /// [`BasicWorker`] keeps the trait's default `prepare_request`, so the
    /// HTTP proxy path may apply the DP-rank edit on raw slices.
    fn uses_builtin_prepare_request(&self) -> bool {
        true
    }

    fn kv_engine_id(&self) -> Option<String> {
        self.kv_engine_id.load_full().map(|id| (*id).clone())
    }

    fn refresh_kv_engine_id(&self, kv_engine_id: Option<String>) -> bool {
        let previous = self.kv_engine_id.load_full();
        if previous.as_deref() == kv_engine_id.as_ref() {
            return false;
        }
        self.kv_engine_id.store(kv_engine_id.map(Arc::new));
        true
    }

    fn kv_engine_id_confirmed(&self) -> bool {
        !self.kv_engine_id_unconfirmed.load(Ordering::Relaxed)
    }

    fn set_kv_engine_id_confirmed(&self, confirmed: bool) {
        self.kv_engine_id_unconfirmed
            .store(!confirmed, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self) -> &str {
        &self.metadata.spec.url
    }

    fn redacted_url(&self) -> &str {
        &self.metadata.redacted_url
    }

    fn api_key(&self) -> Option<&String> {
        self.metadata.spec.api_key.as_ref()
    }

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.spec.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.spec.connection_mode
    }

    fn status(&self) -> WorkerStatus {
        self.runtime.load().status()
    }

    fn revision(&self) -> u64 {
        self.runtime.load().revision()
    }

    fn cache_trace_epochs(&self) -> Vec<String> {
        if !self.runtime.load().cache_trace_epochs_valid() {
            return Vec::new();
        }
        self.metadata
            .spec
            .labels
            .get("cache_trace_epochs")
            .and_then(|epochs| serde_json::from_str(epochs).ok())
            .unwrap_or_default()
    }

    fn set_status(&self, status: WorkerStatus) {
        self.runtime.load().set_status(status);
        Metrics::set_worker_health(self.url(), status == WorkerStatus::Ready);
        self.publish_state();
    }

    fn inherit_shared_state_from(&self, other: &dyn Worker) -> bool {
        let Some(other) = other.as_any().downcast_ref::<BasicWorker>() else {
            return false;
        };
        self.install_shared_state_from_basic(other);
        true
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        if self.metadata.health_config.disable_health_check {
            return Ok(());
        }

        let probe_ok = match &self.metadata.spec.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc => self.grpc_health_check().await?,
            ConnectionMode::Zmq => self.zmq_health_check().await?,
        };

        if probe_ok {
            Ok(())
        } else {
            Err(WorkerError::HealthCheckFailed {
                url: self.metadata.spec.url.clone(),
                reason: "health probe returned non-success".to_string(),
            })
        }
    }

    fn consecutive_failures_increment(&self) -> usize {
        self.runtime.load().consecutive_failures_increment()
    }

    fn consecutive_failures_reset(&self) {
        self.runtime.load().consecutive_failures_reset();
    }

    fn consecutive_successes_increment(&self) -> usize {
        self.runtime.load().consecutive_successes_increment()
    }

    fn consecutive_successes_reset(&self) {
        self.runtime.load().consecutive_successes_reset();
    }

    fn total_pending_probes(&self) -> usize {
        self.runtime.load().total_pending_probes()
    }

    fn total_pending_probes_increment(&self) -> usize {
        self.runtime.load().total_pending_probes_increment()
    }

    fn total_pending_probes_reset(&self) {
        self.runtime.load().total_pending_probes_reset();
    }

    fn load(&self) -> usize {
        self.runtime.load().load()
    }

    fn increment_load(&self) {
        self.runtime.load().increment_load();
        self.update_running_requests_metrics();
    }

    fn try_increment_load(&self, max: usize) -> bool {
        let incremented = self.runtime.load().try_increment_load(max);
        if incremented {
            self.update_running_requests_metrics();
        }
        incremented
    }

    fn decrement_load(&self) {
        if !self.runtime.load().try_decrement_load() {
            tracing::warn!(
                worker_url = %self.metadata.redacted_url,
                "Attempted to decrement load counter that is already at 0"
            );
        }
        self.update_running_requests_metrics();
    }

    fn try_admit_pd(&self, count: usize, window: usize) -> bool {
        self.runtime.load().try_admit_pd(count, window)
    }

    fn release_pd(&self, count: usize) {
        self.runtime.load().release_pd(count);
    }

    fn pd_admitted(&self) -> usize {
        self.runtime.load().pd_admitted()
    }

    fn pd_enqueue(&self, rooms: usize) -> u64 {
        self.runtime.load().pd_enqueue(rooms)
    }

    fn pd_may_claim(&self, place: u64) -> bool {
        self.runtime.load().pd_may_claim(place)
    }

    fn pd_dequeue(&self, place: u64) {
        self.runtime.load().pd_dequeue(place);
    }

    fn pd_wake(&self, place: u64) -> Option<Arc<Notify>> {
        self.runtime.load().pd_wake(place)
    }

    fn routing_key_load(&self) -> usize {
        self.runtime.load().routing_key_load()
    }

    fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.runtime.load().routing_key_inflight(routing_key)
    }

    fn increment_routing_key_load(&self, routing_key: &str) {
        self.runtime.load().increment_routing_key_load(routing_key);
    }

    fn decrement_routing_key_load(&self, routing_key: &str) {
        self.runtime.load().decrement_routing_key_load(routing_key);
    }

    fn processed_requests(&self) -> usize {
        self.runtime.load().processed_requests()
    }

    fn increment_processed(&self) {
        self.runtime.load().increment_processed();
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn circuit_breaker_state(&self) -> super::circuit_breaker::CircuitState {
        self.circuit_breaker.load().state()
    }

    fn circuit_breaker_can_execute(&self) -> bool {
        self.breaker_allows(&self.runtime.load())
    }

    fn is_overloaded(&self) -> bool {
        self.runtime.load().is_overloaded()
    }

    fn set_overloaded(&self, overloaded: bool) -> bool {
        let changed = self.runtime.load().set_overloaded(overloaded);
        if changed {
            self.publish_state();
        }
        changed
    }

    fn stall_reason(&self) -> Option<StallReason> {
        self.runtime.load().stall_reason()
    }

    fn set_stall(&self, reason: Option<StallReason>) -> bool {
        let changed = self.runtime.load().set_stall(reason);
        if changed {
            self.publish_state();
        }
        changed
    }

    fn note_contact(&self) {
        self.runtime.load().note_contact();
    }

    fn note_token_progress(&self) {
        self.runtime.load().note_token_progress();
    }

    fn note_token_progress_at(&self, now_ms: u64) {
        self.runtime.load().note_token_progress_at(now_ms);
    }

    fn note_transport_failure(&self) {
        self.runtime.load().note_transport_failure();
    }

    fn transport_failure_pending(&self) -> bool {
        self.runtime.load().transport_failure_pending()
    }

    fn contact_age(&self) -> Duration {
        self.runtime.load().contact_age()
    }

    fn token_progress_age(&self) -> Duration {
        self.runtime.load().token_progress_age()
    }

    fn note_load_record(&self) {
        self.runtime.load().note_load_record();
    }

    fn pushes_load_records(&self) -> bool {
        self.runtime.load().pushes_load_records()
    }

    fn note_engine_stall(&self, stalled: bool) {
        self.runtime.load().note_engine_stall(stalled);
    }

    fn engine_stalled(&self) -> bool {
        self.runtime.load().engine_stalled()
    }

    fn prefill_rate_tps(&self) -> u64 {
        self.runtime.load().prefill_rate_tps()
    }

    fn swap_waiting_reqs(&self, waiting: i64) -> i64 {
        self.runtime.load().swap_waiting_reqs(waiting)
    }

    fn swap_load_sample(&self, load: usize) -> usize {
        self.runtime.load().swap_load_sample(load)
    }

    fn contact_wake(&self) -> Option<Arc<Notify>> {
        Some(self.runtime.load().contact_wake())
    }

    fn signal_connected(&self) {
        if let Some(tx) = &self.connect_signal_tx {
            let _ = tx.send(WorkerConnected {
                url: self.url().to_string(),
                revision: self.revision(),
            });
        }
    }

    fn reset_circuit_breaker(&self) {
        self.circuit_breaker.load().reset();
    }

    fn readmit_circuit_breaker(&self) -> bool {
        let breaker = self.circuit_breaker.load();
        if breaker.try_half_open() {
            return true;
        }
        breaker.clear_failure_streak();
        false
    }

    fn note_admitted(&self) {
        self.runtime.load().note_admitted();
    }

    fn admitted_age(&self) -> Duration {
        self.runtime.load().admitted_age()
    }

    fn warmup_growth(&self, indexed: usize) -> usize {
        self.runtime.load().warmup_growth(indexed)
    }

    fn divert_until_ms(&self) -> u64 {
        self.runtime.load().divert_until_ms()
    }

    fn note_diverted(&self, until_ms: u64) {
        self.runtime.load().note_diverted(until_ms);
    }

    fn note_prefill_started(&self, tokens: u64, timed: bool) {
        self.runtime.load().note_prefill_started(tokens, timed);
    }

    fn note_prefill_started_at(&self, tokens: u64, timed: bool, now_ms: u64) {
        // 0 is the runtime's unset stamp, as the plain note guards too.
        self.runtime
            .load()
            .note_prefill_started_at(tokens, timed, now_ms.max(1));
    }

    fn note_prefill_ended(&self, tokens: u64, timed: bool, cached: Option<u64>) {
        self.runtime
            .load()
            .note_prefill_ended(tokens, timed, cached);
    }

    fn note_prefill_ended_at(&self, tokens: u64, timed: bool, cached: Option<u64>, now_ms: u64) {
        self.runtime
            .load()
            .note_prefill_ended_at(tokens, timed, cached, now_ms.max(1));
    }

    fn prefill_backlog(&self) -> Duration {
        self.runtime.load().prefill_backlog()
    }

    fn prefill_eta(&self, tokens: u64) -> Duration {
        self.runtime.load().prefill_eta(tokens)
    }

    fn prefill_answered_ago(&self) -> Option<Duration> {
        self.runtime.load().prefill_answered_ago()
    }

    fn note_prefill_handoff(&self) {
        self.runtime.load().note_prefill_handoff();
    }

    fn prefill_handed_off_ago(&self) -> Option<Duration> {
        self.runtime.load().prefill_handed_off_ago()
    }

    fn prefill_unanswered_run(&self) -> Option<Duration> {
        self.runtime.load().prefill_unanswered_run()
    }

    fn prefill_cold_shape(&self, tokens: u64) -> bool {
        self.runtime.load().prefill_cold_shape(tokens)
    }

    fn note_tracked_started(&self) {
        self.runtime.load().note_tracked_started();
    }

    fn note_tracked_ended(&self) {
        self.runtime.load().note_tracked_ended();
    }

    fn tracked_load(&self) -> usize {
        self.runtime.load().tracked_load()
    }

    fn note_output_started(&self) {
        self.runtime.load().note_output_started();
    }

    fn note_output_ended(&self) {
        self.runtime.load().note_output_ended();
    }

    fn output_silence(&self) -> Option<Duration> {
        self.runtime.load().output_silence()
    }

    fn is_available(&self) -> bool {
        // Same two guards the pre-veto version took (`is_healthy` +
        // `circuit_breaker_can_execute`): the vetoes ride the runtime guard.
        let rt = self.runtime.load();
        rt.status() == WorkerStatus::Ready
            && !rt.is_overloaded()
            && rt.stall_reason().is_none()
            && self.breaker_allows(&rt)
    }

    fn is_healthy_and_eligible(&self) -> bool {
        let rt = self.runtime.load();
        rt.status() == WorkerStatus::Ready && !rt.is_overloaded() && rt.stall_reason().is_none()
    }

    fn completion_sink(&self) -> Option<Arc<dyn RequestCompletionSink>> {
        self.completion_sink
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_completion_sink(&self, sink: Option<Arc<dyn RequestCompletionSink>>) {
        *self
            .completion_sink
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = sink;
    }

    fn routing_state(&self) -> RoutingState {
        // One runtime guard covers status + load + processed + the overload veto
        // (all live in `self.runtime`); the circuit breaker is a separate
        // ArcSwap, so it needs its own guard.
        let rt = self.runtime.load();
        RoutingState {
            healthy: rt.status() == WorkerStatus::Ready,
            can_execute: self.breaker_allows(&rt),
            load: rt.load(),
            processed: rt.processed_requests(),
            overloaded: rt.is_overloaded(),
            stalled: rt.stall_reason().is_some(),
        }
    }

    fn record_circuit_breaker_outcome(&self, success: bool) {
        let breaker = self.circuit_breaker.load();
        // The state as stored, before and after: the timeout's move from
        // open to half-open belongs to the routing decision's read, as
        // before this series existed, not to a record that happens to land
        // after the timeout.
        let before = breaker.peek_state();
        breaker.record_outcome(success);
        let after = breaker.peek_state();
        // A trip or a close moved the routing state. A record that finds the
        // breaker open publishes as well: its request was in flight across
        // the trip, and a selection's read may have moved the breaker to
        // half-open, and the series to available, between the peek and the
        // record (open, half-open, open reads as no move); the publish
        // returns early under the lock when the series already says
        // circuit_open. A record that left a closed or half-open breaker
        // where it was publishes nothing, so the steady path takes no lock.
        if after != before || after == super::circuit_breaker::CircuitState::Open {
            self.publish_state();
        }
    }

    fn publish_state(&self) {
        self.publish_state_under(&self.runtime.load(), false);
    }

    fn republish_state(&self) {
        self.publish_state_under(&self.runtime.load(), true);
    }

    fn resilience(&self) -> &ResolvedResilience {
        &self.resilience
    }

    fn http_client(&self) -> &reqwest::Client {
        self.http_client.client()
    }

    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.http_client.handle_if_initialized()
    }

    fn backend_client_if_initialized(&self) -> Option<Arc<BackendClient>> {
        self.backend_client.load_full().get().cloned()
    }

    fn supports_model(&self, model_id: &str) -> bool {
        let overridden = self.models_override.load();
        if !overridden.is_wildcard() {
            return overridden.supports(model_id);
        }
        self.metadata.supports_model(model_id)
    }

    fn models(&self) -> Vec<ModelCard> {
        self.with_model_cards(<[ModelCard]>::to_vec)
    }

    fn any_model(&self, pred: &mut dyn FnMut(&ModelCard) -> bool) -> bool {
        self.with_model_cards(|cards| cards.iter().any(pred))
    }

    fn context_length(&self, model_id: &str) -> Option<u32> {
        let overridden = self.models_override.load();
        if overridden.is_wildcard() {
            context_length_from(&self.metadata.spec.models, model_id)
        } else {
            context_length_from(&overridden, model_id)
        }
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        tracing::debug!(
            "Setting {} models for worker {} via lazy discovery",
            models.len(),
            self.metadata.redacted_url
        );
        self.models_override
            .store(Arc::new(WorkerModels::from(models)));
    }

    fn has_models_discovered(&self) -> bool {
        !self.models_override.load().is_wildcard() || !self.metadata.spec.models.is_wildcard()
    }

    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>> {
        match self.metadata.spec.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc => {
                // OnceCell provides lock-free reads after initialization.
                // get_or_try_init only acquires internal lock on first call.
                // The initializer's future is boxed: this method's own future
                // (boxed once per call by the trait) would otherwise carry the
                // connect path's whole state machine on every call, although
                // only the first call per worker runs it. The cold path pays
                // its allocation once; every later call's future holds a
                // pointer where it held kilobytes.
                let cell = self.backend_client.load_full();
                let client = cell
                    .get_or_try_init(|| {
                        Box::pin(async {
                            let runtime_str = self.metadata.spec.runtime_type.to_string();
                            tracing::info!(
                                "Lazily initializing gRPC client ({}) for worker: {}",
                                runtime_str,
                                self.metadata.redacted_url
                            );
                            // DP-expanded workers carry a `{base}@{rank}` URL; connect to the base
                            match GrpcClient::connect(self.metadata.base_url(), &runtime_str).await
                            {
                                Ok(client) => {
                                    tracing::info!(
                                        "Successfully connected gRPC client ({}) for worker: {}",
                                        runtime_str,
                                        self.metadata.redacted_url
                                    );
                                    Ok(Arc::new(BackendClient::Grpc(client)))
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to connect gRPC client for worker {}: {}",
                                        self.metadata.redacted_url,
                                        e
                                    );
                                    Err(WorkerError::ConnectionFailed {
                                        url: self.metadata.spec.url.clone(),
                                        reason: format!("Failed to connect to gRPC server: {e}"),
                                    })
                                }
                            }
                        })
                    })
                    .await?;
                Ok(Some(Arc::clone(client)))
            }
            ConnectionMode::Zmq => {
                // SMG binds the handshake + data-plane sockets; the
                // operator-launched engine dials them. Peek only: the handshake
                // is owned by the background driver (it can outlast any
                // caller's deadline — see spawn_zmq_connect_driver), so fail
                // fast instead of blocking a request or a load poll behind it.
                // Kicking the driver off here also covers workers whose health
                // checks are disabled, where no probe ever runs.
                let cell = self.backend_client.load_full();
                if let Some(client) = cell.get() {
                    return Ok(Some(Arc::clone(client)));
                }
                self.spawn_zmq_connect_driver(&cell);
                Err(WorkerError::ConnectionFailed {
                    url: self.metadata.spec.url.clone(),
                    reason: "ZMQ backend handshake has not completed yet".to_string(),
                })
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        // Intentional no-op: a failed gRPC worker is removed and re-added rather
        // than reconnected in place. (The ZMQ path differs: its health probe
        // evicts a dead client and reconnects — see zmq_health_check.)
        tracing::debug!(
            "reset_grpc_client called for {} (no-op for gRPC workers)",
            self.metadata.redacted_url
        );
        Ok(())
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);
        let maybe = self.get_backend_client().await?;
        let Some(backend_client) = maybe else {
            tracing::error!(
                "Worker {} is not a gRPC worker but connection mode is gRPC",
                self.metadata.redacted_url
            );
            return Ok(false);
        };

        match time::timeout(timeout, backend_client.health_check()).await {
            Ok(Ok(resp)) => {
                tracing::debug!(
                    "gRPC health OK for {}: healthy={}",
                    self.metadata.redacted_url,
                    resp.healthy
                );
                Ok(resp.healthy)
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    "gRPC health RPC error for {}: {err:?}",
                    self.metadata.redacted_url
                );
                Ok(false)
            }
            Err(_) => {
                tracing::warn!("gRPC health timed out for {}", self.metadata.redacted_url);
                Ok(false)
            }
        }
    }

    async fn zmq_health_check(&self) -> WorkerResult<bool> {
        // Never drive (or cancel) the handshake from the probe: model load can
        // take far longer than the health timeout, and cancelling it mid-flight
        // would rebind the sockets on every probe and never connect. Peek the
        // cached client instead — if it isn't there yet, kick the handshake off
        // once in the background and report not-ready until it lands. Thereafter
        // liveness is local (the connector marks the client closed on
        // ENGINE_CORE_DEAD or a transport failure); there is no health RPC on
        // the raw ZMQ wire.
        let cell = self.backend_client.load_full();
        if let Some(backend_client) = cell.get() {
            if backend_client.is_alive() {
                return Ok(true);
            }
            // The engine behind this client died, and a ZMQ client cannot
            // reconnect in place (liveness is latched). Evict the dead client
            // and reset the handshake guard so the next probe re-runs
            // connect_zmq_backend, rebinding the sockets for a replacement
            // engine to dial into. Eviction cannot race an in-flight connect:
            // a cell only becomes non-empty once its connect finished, and
            // compare_and_swap keeps a lagging probe from clobbering a fresh
            // cell another probe already swapped in.
            tracing::warn!(
                "ZMQ engine for {} is no longer alive; evicting the dead client and rebinding on a later probe",
                self.metadata.redacted_url
            );
            self.backend_client
                .compare_and_swap(&cell, Arc::new(OnceCell::new()));
            self.zmq_connect_started.store(false, Ordering::SeqCst);
            return Ok(false);
        }
        self.spawn_zmq_connect_driver(&cell);
        Ok(false)
    }

    fn abort_background_tasks(&self) {
        if let Some(handle) = self.zmq_connect_abort.swap(None) {
            handle.abort();
            // The bound sockets are released with the cancelled future; reset
            // the guard so any instance still sharing this state can retry.
            self.zmq_connect_started.store(false, Ordering::SeqCst);
        }
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        let health_url = format!("{}{}", self.base_url(), self.metadata.health_endpoint);

        let mut req = self.http_client.client().get(&health_url).timeout(timeout);
        if let Some(api_key) = &self.metadata.spec.api_key {
            req = req.bearer_auth(api_key);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    Ok(true)
                } else {
                    tracing::warn!(
                        "HTTP health check returned non-success status for {}: {}",
                        super::endpoint::Endpoint::redacted_str(&health_url),
                        status
                    );
                    Ok(false)
                }
            }
            Err(err) => {
                tracing::warn!(
                    "HTTP health check failed for {}: {err:?}",
                    super::endpoint::Endpoint::redacted_str(&health_url)
                );
                Ok(false)
            }
        }
    }
}

/// Helper to convert Worker trait object to WorkerInfo struct.
///
/// Both `is_healthy` and `status` are derived from the same atomic snapshot
/// to avoid TOCTOU between the two fields. The `status` field exposes the
/// real lifecycle state (Pending/Ready/NotReady/Failed) so API consumers
/// can distinguish "starting up" from "broken" — `is_healthy` collapses
/// everything to a routability bool for backwards compatibility.
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    let metadata = worker.metadata();
    let spec = metadata.spec.clone();
    let status = worker.status();
    // Only PD workers pair; a regular worker's descriptor would be noise.
    let pd_pairing = matches!(spec.worker_type, WorkerType::Prefill | WorkerType::Decode)
        .then(|| metadata.pd_pairing.key());

    WorkerInfo {
        id: worker.url().to_string(),
        model_id: spec.models.primary().map(|m| m.id.clone()),
        spec,
        is_healthy: status == WorkerStatus::Ready,
        status: Some(status),
        load: worker.load(),
        http2: metadata.http2,
        pd_pairing,
        stalled: worker
            .stall_reason()
            .map(|reason| reason.as_str().to_string()),
        engine_load: None,
        job_status: None,
    }
}

#[cfg(test)]
mod tests;
