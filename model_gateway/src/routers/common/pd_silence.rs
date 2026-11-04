//! The silence bound of a disaggregated prefill side, one judge for the gRPC
//! pipeline and the HTTP prefill-decode router.
//!
//! A prefill side that produces nothing is given up on within a bound
//! counted from its dispatch ([`PrefillBound`], sized by the liveness
//! module from the worker's observed prefill rate and the prompt), instead
//! of holding the request, its decode partner and the prefill admission
//! slot until the request's own timeout. Past the deadline the side is
//! judged again a bound at a time ([`judge_prefill_side`]): a worker that
//! answered another side, or whose handoff a decode side showed, within the
//! bound is backed up, not frozen, and a worker whose current run of prompts
//! has not answered yet gets the time its backlog takes
//! ([`crate::worker::liveness::prefill_side_reprieve`]). The verdict
//! ([`SilenceVerdict`]) charges one worker once, where the bound passed: the
//! prefill, or the decode partner that had stalled mid-stream on the parallel
//! dispatch shape. Each transport awaits its own call under the bound
//! ([`await_prefill_within`]) and answers the client with the shared
//! silent-side 504 ([`SilenceVerdict::answer`]).

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use axum::{http::StatusCode, response::Response};
use tracing::warn;

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::common::{
        attempt_ledger::mark_failed_side,
        pd_failure::{PdSide, SidePhase},
    },
    worker::{
        liveness::{self, SideReprieve},
        Worker,
    },
};

/// The message a prefill side's silence verdict starts with (see
/// [`prefill_side_silent`]); it tells the verdict from an engine's own
/// `DEADLINE_EXCEEDED`.
pub(crate) const PREFILL_SILENCE_MESSAGE: &str = "prefill side produced nothing within";

/// The message of the verdict that puts a prefill side's silence on its
/// stalled decode partner (see [`decode_partner_silent`]).
pub(crate) const DECODE_SILENCE_MESSAGE: &str = "decode worker produced nothing within";

/// Whether the rule is on: a prefill side gets a bound unless the wedge
/// threshold is zero ([`liveness::prefill_side_bound`]). For what a router
/// spends only because a bound reads it (the HTTP router's prompt estimate).
pub(crate) fn applies() -> bool {
    !liveness::budgets().wedge.is_zero()
}

/// The bound on one prefill side's silence, counted from its dispatch (see
/// [`liveness::prefill_side_bound`]); the dispatch instant stays with it,
/// as a reprieve past the deadline may count from there, and so does the
/// worker's backlog once a reprieve has read it, for both phases of the
/// side (see [`liveness::prefill_side_reprieve`]).
#[derive(Clone, Debug)]
pub(crate) struct PrefillBound {
    pub(crate) dispatched: Instant,
    pub(crate) deadline: Instant,
    pub(crate) bound: Duration,
    /// The silence a stalled decode partner is judged by (see
    /// [`stalled_decode_partner`]): a decode side's bound for the same
    /// prompt, which counts its prefill time once where `bound` counts it
    /// twice (see [`liveness::decode_side_bound`]).
    pub(crate) partner_bound: Duration,
    pub(crate) backlog: Arc<OnceLock<Duration>>,
}

impl PrefillBound {
    /// The bound for a side of `prompt_tokens` dispatched to `worker` at
    /// `dispatched`; `None` when the rule is off.
    pub(crate) fn at(
        worker: &Arc<dyn Worker>,
        prompt_tokens: u64,
        dispatched: Instant,
    ) -> Option<Self> {
        let partner_bound = liveness::decode_side_bound(worker, prompt_tokens)?;
        Self::of(
            liveness::prefill_side_bound(worker, prompt_tokens)?,
            dispatched,
        )
        .map(|bound| Self {
            partner_bound,
            ..bound
        })
    }

    /// A bound of `bound` from `dispatched`, a stalled decode partner judged
    /// by the same. A bound past the clock's range never expires: `None`,
    /// and the side stays unbounded.
    pub(crate) fn of(bound: Duration, dispatched: Instant) -> Option<Self> {
        let deadline = dispatched.checked_add(bound)?;
        Some(Self {
            dispatched,
            deadline,
            bound,
            partner_bound: bound,
            backlog: Arc::default(),
        })
    }
}

/// What a prefill side's silence verdict was judged on (see
/// [`liveness::SideReprieve::Silent`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillSilence {
    /// The side's own bound.
    pub(crate) bound: Duration,
    /// How long the side had waited since its dispatch.
    pub(crate) waited: Duration,
    /// How long its worker had answered nothing (`None`: never).
    pub(crate) silent: Option<Duration>,
    /// What that silence was judged against: the bound, or the time the
    /// backlog of a run that had not answered takes.
    pub(crate) window: Duration,
}

/// The verdict on a prefill side that produced nothing within its bound, in
/// `phase` (`dispatch`: the call returned no stream; `first_response`: the
/// stream yielded nothing), while its worker answered no other side for its
/// window either: the bound, or the time the backlog of a run that had not
/// answered takes (see [`liveness::prefill_side_reprieve`]). One WARN naming
/// the worker, the request, how long the side waited, how long the worker
/// had answered nothing against which window and how long ago it last
/// answered anything or showed a decode side a handoff (a handoff only
/// reprieves, so the silence a verdict names is the worker's own), the
/// worker-error counter under the timeout class, and
/// the status the side's failure is reported with. Nothing else bounds this
/// wait below the request's own timeout: the engine's health probes keep
/// passing, the queue timeout ended at worker selection, and the admission
/// gate only gates the dispatch. The attempt fails here; the retry layer
/// takes the request to another prefill worker when one is available (the
/// answer is [`SilenceVerdict::answer`]).
pub(crate) fn prefill_side_silent(
    worker: &Arc<dyn Worker>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
    silence: PrefillSilence,
    phase: &'static str,
) -> SilenceVerdict {
    let millis = |span: Duration| u64::try_from(span.as_millis()).unwrap_or(u64::MAX);
    let ago = |ago: Option<Duration>| {
        ago.map_or_else(|| "never".to_string(), |ago| format!("{ago:.1?} ago"))
    };
    let worker_last_answer = ago(worker.prefill_answered_ago());
    let worker_last_handoff = ago(worker.prefill_handed_off_ago());
    warn!(
        request_id = %request_id,
        worker = worker.redacted_url(),
        worker_type = metrics_labels::WORKER_PREFILL,
        phase,
        prompt_tokens,
        bound_ms = millis(silence.bound),
        waited_ms = millis(silence.waited),
        worker_silent_ms = silence.silent.map(millis),
        window_ms = millis(silence.window),
        worker_last_answer = %worker_last_answer,
        worker_last_handoff = %worker_last_handoff,
        "PD prefill side produced nothing past its silence bound, and its worker answered no other side for its silence window; giving up on it"
    );
    Metrics::record_worker_error(
        metrics_labels::WORKER_PREFILL,
        connection,
        metrics_labels::ERROR_TIMEOUT,
    );
    let PrefillSilence {
        bound,
        waited,
        silent,
        window,
    } = silence;
    let worker_silent = match silent {
        Some(silent) => {
            format!("its worker answered nothing for {silent:.1?} of a {window:.1?} window")
        }
        None => format!("its worker never answered (a {window:.1?} window)"),
    };
    SilenceVerdict {
        charged: PdSide::Prefill,
        message: format!(
            "{PREFILL_SILENCE_MESSAGE} {waited:.1?} of its dispatch ({phase}, bound {bound:.1?}): {worker_silent}"
        ),
    }
}

/// The decode partner of a prefill side whose silence bound passed, when the
/// silence is the partner's: on the parallel dispatch shape the prefill
/// side's first response waits for the decode side, so a decode worker that
/// stopped producing makes every prefill side look silent and the prefill
/// worker answer no side at all. A decode worker that still holds streams it
/// had been producing for and has produced nothing for `bound` (a decode
/// side's bound for the prompt, see [`PrefillBound`]) has stalled
/// ([`Worker::output_silence`]); one that is still producing, or
/// holds no such stream (its streams ended, or its in-flight work has shown
/// nothing yet: a frozen prefill in front of an idle decode looks like that,
/// and the prefill keeps the verdict as before), is not read as the cause.
/// `None` without a partner (the sequential shape sends the decode side
/// after the prefill answered, so a silent prefill there is its own).
pub(crate) fn stalled_decode_partner(
    partner: Option<&Arc<dyn Worker>>,
    bound: Duration,
) -> Option<(Arc<dyn Worker>, Duration)> {
    let decode = partner?;
    let silence = decode
        .output_silence()
        .filter(|silence| *silence >= bound)?;
    Some((Arc::clone(decode), silence))
}

/// The verdict on a prefill side that produced nothing within its bound
/// while its decode partner had stalled mid-stream (see
/// [`stalled_decode_partner`]): one WARN naming the decode worker, the
/// prefill worker and the request, the worker-error counter under the
/// timeout class for the DECODE worker, and the status the attempt fails
/// with, which the retry layer may take to another decode worker (the
/// answer is [`SilenceVerdict::answer`]). Nothing is charged to the prefill
/// worker, which was waiting on its partner.
pub(crate) fn decode_partner_silent(
    prefill: &Arc<dyn Worker>,
    decode: &Arc<dyn Worker>,
    connection: &'static str,
    request_id: &str,
    bound: Duration,
    silence: Duration,
    phase: &'static str,
) -> SilenceVerdict {
    warn!(
        request_id = %request_id,
        worker = decode.redacted_url(),
        worker_type = metrics_labels::WORKER_DECODE,
        prefill_worker = prefill.redacted_url(),
        phase,
        bound_ms = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX),
        decode_silence_ms = u64::try_from(silence.as_millis()).unwrap_or(u64::MAX),
        "PD decode worker has produced nothing for the silence bound on the streams it was serving, and the prefill side waits on it; giving up on the pair and charging the decode worker"
    );
    Metrics::record_worker_error(
        metrics_labels::WORKER_DECODE,
        connection,
        metrics_labels::ERROR_TIMEOUT,
    );
    SilenceVerdict {
        charged: PdSide::Decode,
        message: format!(
            "{DECODE_SILENCE_MESSAGE} {bound:.1?} of the prefill side's dispatch ({phase}): the prefill \
             side waited on it (decode silent for {silence:.1?})"
        ),
    }
}

/// One check of a prefill side past its deadline, in `phase`: how much
/// longer it waits (see [`liveness::prefill_side_reprieve`]), or the
/// verdict that takes its place. Once its worker has answered nothing
/// within the side's bound, a decode `partner` that had stalled mid-stream
/// takes the verdict at once (see [`stalled_decode_partner`]; the parallel
/// shape passes the decode worker, the sequential shape none), whatever the
/// worker's backlog; otherwise the prefill's verdict, once the worker has
/// been silent for its window.
pub(crate) fn judge_prefill_side(
    worker: &Arc<dyn Worker>,
    partner: Option<&Arc<dyn Worker>>,
    bound: &PrefillBound,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
    phase: &'static str,
) -> Result<Duration, SilenceVerdict> {
    let waited = bound.dispatched.elapsed();
    let standing = liveness::prefill_side_reprieve(worker, bound.bound, waited, &bound.backlog);
    if standing.past_bound() {
        if let Some((decode, silence)) = stalled_decode_partner(partner, bound.partner_bound) {
            return Err(decode_partner_silent(
                worker,
                &decode,
                connection,
                request_id,
                bound.partner_bound,
                silence,
                phase,
            ));
        }
    }
    match standing {
        SideReprieve::WithinBound(more) | SideReprieve::FirstAnswerDue(more) => Ok(more),
        SideReprieve::Silent { silent, window } => Err(prefill_side_silent(
            worker,
            connection,
            request_id,
            prompt_tokens,
            PrefillSilence {
                bound: bound.bound,
                waited,
                silent,
                window,
            },
            phase,
        )),
    }
}

/// The verdict on a prefill side given up on: the side whose worker it
/// charged (the prefill, or the decode partner that had stalled) and the
/// words of the answer, which open with the verdict's message
/// ([`PREFILL_SILENCE_MESSAGE`], [`DECODE_SILENCE_MESSAGE`]) so a
/// transport's own deadline error is told from it.
#[derive(Debug)]
pub(crate) struct SilenceVerdict {
    pub(crate) charged: PdSide,
    pub(crate) message: String,
}

impl SilenceVerdict {
    /// The client's answer for the verdict: the charged side's silent-side
    /// 504 ([`silent_side_answer`]), `worker` being that side's worker.
    pub(crate) fn answer(&self, worker: &dyn Worker) -> Response {
        silent_side_answer(self.charged, &self.message, worker)
    }
}

/// The client answer for a side given up on for its silence: a 504 under the
/// side's own silent code, replayable on another worker of its kind. The
/// side's worker is marked on the response so the retry window closes on it
/// alone (its partner stays selectable) and the answer stands when no other
/// worker of the side's kind is available (see
/// `attempt_ledger::FailedSide`).
pub(crate) fn silent_side_answer(side: PdSide, message: &str, worker: &dyn Worker) -> Response {
    let mut response = side.answer(SidePhase::Silent, StatusCode::GATEWAY_TIMEOUT, message);
    mark_failed_side(&mut response, worker);
    response
}

/// Await a prefill side's `dispatch` (its call, up to the point the
/// transport counts as the side's answer, or the first item of the stream
/// the call returned; `phase` names which in the verdict, `dispatch` or
/// `first_response`) within `bound`. Past the deadline
/// the side waits on while its worker keeps answering other sides, or while
/// the worker's current run of prompts has not answered yet and has not
/// gone unanswered for the time its backlog takes ([`judge_prefill_side`]);
/// once it is judged silent the verdict takes its place and the call stays
/// with the caller, still pending: the gRPC pipeline retires it off the
/// request path, the HTTP router drops it, which closes the connection. A
/// stalled decode `partner` takes the verdict where the shape has one.
/// Without a bound the call is awaited as it is.
#[expect(
    clippy::too_many_arguments,
    reason = "a side's identity and its bound: plumbing the two transports share, not state"
)]
pub(crate) async fn await_prefill_within<T, F>(
    bound: Option<&PrefillBound>,
    mut dispatch: Pin<&mut F>,
    worker: &Arc<dyn Worker>,
    partner: Option<&Arc<dyn Worker>>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
    phase: &'static str,
) -> Result<T, SilenceVerdict>
where
    F: Future<Output = T> + ?Sized,
{
    let Some(bound) = bound else {
        return Ok(dispatch.await);
    };
    let mut deadline = bound.deadline;
    loop {
        if let Ok(result) = tokio::time::timeout_at(deadline.into(), dispatch.as_mut()).await {
            return Ok(result);
        }
        let more = judge_prefill_side(
            worker,
            partner,
            bound,
            connection,
            request_id,
            prompt_tokens,
            phase,
        )?;
        deadline = Instant::now() + more;
    }
}
