//! Gateway-side context-window check: reject an input the selected engine
//! cannot hold before it is dispatched.
//!
//! The gRPC path tokenizes at the gateway and ships token ids, so the count
//! here is exactly what the engine would see. Without this check an over-long
//! prompt travels to the engine, which rejects it, and on the disaggregated
//! path that rejection comes back as `500 prefill_worker_failed_to_start`: a
//! server error for a client mistake, blamed on a healthy worker (#2380).
//!
//! Regular gRPC and PD selection share these checks to exclude workers whose
//! advertised windows cannot hold the request. The post-selection guard runs
//! at ingress before any client is acquired, including on other pipeline modes.

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::common::GenerationRequest;
use smg_worker::identity::basic::{ConnectionMode, Worker, WorkerType};
use tracing::debug;

use crate::routers::{
    common::snapshot::{ModelWorkerSnapshot, RoutingPool, WindowFloor},
    error,
    grpc::context::{PreparationOutput, RequestType, WorkerSelection},
};

/// The window that binds `worker` for `model_id`, and whether the worker
/// holds the completion budget. `None` where nothing binds it: another
/// connection mode than gRPC (a direct-ZMQ worker of the shared pipeline is
/// checked after selection), or no known window. A prefill worker is held to
/// the input alone: in a disaggregated request it runs one token and its
/// decode partner holds the budget.
///
/// The one reading of a worker that a candidate's answer and a list's floor
/// both stand on.
fn bound_window(worker: &dyn Worker, model_id: &str) -> Option<(u32, bool)> {
    if !matches!(worker.connection_mode(), ConnectionMode::Grpc) {
        return None;
    }
    let holds_the_budget = !matches!(worker.worker_type(), WorkerType::Prefill);
    worker
        .context_length(model_id)
        .map(|window| (window, holds_the_budget))
}

/// The window floor of `workers` for `model_id`, one window read a worker:
/// the narrowest window of each class [`bound_window`] names. What a
/// membership snapshot keeps for a pool
/// ([`ModelWorkerSnapshot::window_floor`]).
fn window_floor_of(workers: &[Arc<dyn Worker>], model_id: &str) -> WindowFloor {
    let mut floor = WindowFloor::default();
    for worker in workers {
        let Some((window, holds_the_budget)) = bound_window(worker.as_ref(), model_id) else {
            continue;
        };
        let kept = if holds_the_budget {
            &mut floor.budget
        } else {
            &mut floor.prefill
        };
        *kept = Some(kept.map_or(window, |narrowest| narrowest.min(window)));
    }
    floor
}

/// Error code for an input that does not fit the model's window; matches the
/// OpenAI API's code for the same condition.
pub(crate) const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

/// The existing gateway checks, retained through retries. Input and output
/// are independent bounds: adding them would change SGLang's clamping policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ContextRequirements {
    input_tokens: usize,
    scoring_position: bool,
    output_tokens: Option<u32>,
}

impl ContextRequirements {
    /// Capture exact preparation counts and the request's completion budget.
    pub(crate) fn new(prep: &PreparationOutput, request: &RequestType) -> Self {
        Self {
            input_tokens: prep.max_input_token_count(),
            scoring_position: matches!(prep, PreparationOutput::Decisions { .. }),
            output_tokens: output_budget(request),
        }
    }

    /// The demand of a request of `input_tokens` with a completion budget of
    /// `budget` tokens, for a benchmark that prepares no request.
    #[cfg(any(test, feature = "test-util"))]
    pub(crate) fn of_counts(input_tokens: usize, budget: u32) -> Self {
        Self {
            input_tokens,
            scoring_position: false,
            output_tokens: Some(budget),
        }
    }

    /// The demand of a request of `input_tokens`, with or without a scoring
    /// position and a completion budget, for a recorder that prepares no
    /// request.
    #[cfg(test)]
    pub(crate) fn of_demand(
        input_tokens: usize,
        scoring_position: bool,
        output_tokens: Option<u32>,
    ) -> Self {
        Self {
            input_tokens,
            scoring_position,
            output_tokens,
        }
    }

    /// The request's completion budget as [`Self::new`] captured it: what a
    /// selection states to Grand Central as the answer's limit. `None`
    /// where the body names none, and for requirements that were not
    /// captured (the encode-prefill-decode mode's default).
    pub(crate) fn output_tokens(self) -> Option<u32> {
        self.output_tokens
    }

    /// Unknown windows remain eligible; no character estimate or token sum is used.
    /// Direct-ZMQ workers in the shared pipeline retain post-selection validation.
    /// A prefill worker is held to the input alone: in a disaggregated request
    /// it runs one token (`sanitize_sampling_for_prefill`) and its decode
    /// partner holds the completion budget, so a prefill window narrower than
    /// the decode window must not refuse a budget the decode side can hold.
    /// A demand that binds nobody reads no window.
    pub(crate) fn admits(self, worker: &dyn Worker, model_id: &str) -> bool {
        self.binds_nobody()
            || bound_window(worker, model_id)
                .is_none_or(|(window, holds_the_budget)| self.holds(window, holds_the_budget))
    }

    /// Whether every worker of the list `floor` was read from admits this
    /// demand: the narrowest window of each class answers for its class.
    fn held_by_all(self, floor: WindowFloor) -> bool {
        floor.budget.is_none_or(|window| self.holds(window, true))
            && floor.prefill.is_none_or(|window| self.holds(window, false))
    }

    /// What a worker of `pools` can still refuse of this demand: the demand
    /// itself, or nothing where the narrowest windows `snapshot` keeps for
    /// those pools hold it. A selection that draws its candidates from the
    /// same snapshot then asks no worker about a request every window holds.
    ///
    /// Candidates narrowed to a wire are a part of their pool, so the pool's
    /// floor errs toward asking them, never toward a missed refusal. A floor
    /// kept for another spelling of the model id leaves the demand whole. A
    /// floor read before a worker's cards changed outside the registry is
    /// stale: the request it lets through unasked still meets the
    /// post-selection guards ([`enforce_context_length`],
    /// [`enforce_output_budget`]) and fails as a 400, never as a wrong
    /// dispatch.
    pub(crate) fn left_by(
        self,
        snapshot: &ModelWorkerSnapshot,
        pools: &[RoutingPool],
        model_id: &str,
    ) -> Self {
        let held = self.binds_nobody()
            || pools.iter().all(|&pool| {
                snapshot
                    .window_floor(pool, model_id, |workers| window_floor_of(workers, model_id))
                    .is_some_and(|floor| self.held_by_all(floor))
            });
        if held {
            Self::default()
        } else {
            self
        }
    }

    /// Whether no window refuses this demand: no input and no budget.
    pub(crate) fn binds_nobody(self) -> bool {
        self == Self::default()
    }

    /// Whether a window of `window` tokens holds this demand: the input, and
    /// the budget where the worker holds it. The one rule behind a
    /// candidate's answer and a list's. Both bounds grow with the window.
    fn holds(self, window: u32, holds_the_budget: bool) -> bool {
        input_fits(self.input_tokens, window, self.scoring_position)
            && (!holds_the_budget || self.output_tokens.is_none_or(|tokens| tokens <= window))
    }

    /// Whether the input fits a window of `limit` tokens: the input bound of
    /// [`Self::admits`], for a caller judging a whole pair by its tighter window.
    pub(crate) fn input_fits(self, limit: u32) -> bool {
        input_fits(self.input_tokens, limit, self.scoring_position)
    }

    /// Preserve the existing 400 and diagnostic for a known insufficient window.
    pub(crate) fn enforce(self, limit: u32, model_id: &str) -> Result<(), Response> {
        enforce_input(self.input_tokens, limit, self.scoring_position, model_id)?;
        enforce_output(self.output_tokens, limit, model_id)
    }

    /// [`Self::enforce`] for a disaggregated pair the request did not fit:
    /// the input against the widest window a pair offers it (`input_limit`,
    /// the tighter of a pair's two sides), the budget against the widest
    /// decode window (`decode_limit`); an unknown window binds nothing.
    pub(crate) fn enforce_for_pair(
        self,
        input_limit: Option<u32>,
        decode_limit: Option<u32>,
        model_id: &str,
    ) -> Result<(), Response> {
        if let Some(limit) = input_limit {
            enforce_input(self.input_tokens, limit, self.scoring_position, model_id)?;
        }
        if let Some(limit) = decode_limit {
            enforce_output(self.output_tokens, limit, model_id)?;
        }
        Ok(())
    }
}

/// Whether the input fits, reserving the scoring position without arithmetic
/// that can overflow even for a synthetic `usize::MAX` token count.
fn input_fits(input_tokens: usize, limit: u32, scoring_position: bool) -> bool {
    if scoring_position {
        input_tokens < limit as usize
    } else {
        input_tokens <= limit as usize
    }
}

/// The shared input check used by selection and the post-selection guard.
fn enforce_input(
    input_tokens: usize,
    limit: u32,
    scoring_position: bool,
    model_id: &str,
) -> Result<(), Response> {
    if input_fits(input_tokens, limit, scoring_position) {
        return Ok(());
    }
    if scoring_position {
        return Err(error::bad_request(CONTEXT_LENGTH_EXCEEDED,
            format!("Decision prompt has {input_tokens} tokens and cannot fit the {limit}-token context window with its scoring position intact")));
    }
    Err(context_length_exceeded(input_tokens, limit, model_id))
}

/// Reject the request when its longest input exceeds the context window the
/// selected worker(s) advertise for `model_id`.
///
/// Only a *known* window is enforced: a worker that never advertised one
/// leaves the check to the engine, as before. The check is on the input
/// alone (strictly longer than the window), the one condition every engine
/// rejects; whether `input + max_tokens` must also fit is engine-specific
/// (vLLM rejects, SGLang clamps) and stays with the engine.
pub(crate) fn enforce_context_length(
    prep: &PreparationOutput,
    workers: &WorkerSelection,
    model_id: &str,
) -> Result<(), Response> {
    if matches!(prep, PreparationOutput::Decisions { .. }) {
        return enforce_decisions_context_length(prep.max_input_token_count(), workers, model_id);
    }
    let Some(limit) = selection_context_length(workers, model_id) else {
        return Ok(());
    };
    enforce_input(prep.max_input_token_count(), limit, false, model_id)
}

/// The refusal for an input of `input_tokens` tokens that does not fit a
/// window of `limit` tokens: the pipeline's, so an answer the router gives
/// by itself refuses an oversized prompt as the pipeline would.
pub(crate) fn context_length_exceeded(input_tokens: usize, limit: u32, model_id: &str) -> Response {
    debug!(
        function = "enforce_context_length",
        input_tokens, limit, model_id, "Rejecting input longer than the model's context window"
    );
    error::bad_request(
        CONTEXT_LENGTH_EXCEEDED,
        format!(
            "This model's maximum context length is {limit} tokens. However, your request has \
             {input_tokens} input tokens. Please reduce the length of the input."
        ),
    )
}

/// The smallest context window any of `workers` advertises for `model_id`,
/// for a check made before a worker is selected: an input within it fits
/// whichever of them a request lands on. `None` when no worker advertises
/// one; the check is then left to the engine, as it is after selection.
pub(crate) fn smallest_context_length(workers: &[Arc<dyn Worker>], model_id: &str) -> Option<u32> {
    workers
        .iter()
        .filter_map(|worker| worker.context_length(model_id))
        .min()
}

/// Reject prompts reaching SGLang's advertised context boundary. The engine
/// can impose a smaller KV capacity limit; response processing also verifies
/// the scored prompt length to detect any scheduler-side truncation.
pub(crate) fn enforce_decisions_context_length(
    input_tokens: usize,
    workers: &WorkerSelection,
    model_id: &str,
) -> Result<(), Response> {
    let Some(limit) = selection_context_length(workers, model_id) else {
        return Ok(());
    };
    enforce_input(input_tokens, limit, true, model_id)
}

/// Reject a completion budget the model's window could never hold.
///
/// A `max_tokens` (or its route's equivalent) above the window is a client
/// mistake on every route and under every provider profile, and on a lane
/// whose engine has no frontend of its own behind the gateway (the direct
/// ZMQ lane, the Rust servicer) nothing after this stage refuses it: the
/// engine runs to the window and stops with a length finish. As above, only
/// a known window is enforced; whether `input + max_tokens` must also fit
/// stays engine-specific (one engine refuses it, another clamps it) and with
/// the engine.
/// Like the input check this runs on the gRPC pipeline only: the HTTP router
/// forwards the body and leaves the budget to the engine.
pub(crate) fn enforce_output_budget(
    request_type: &RequestType,
    workers: &WorkerSelection,
    model_id: &str,
) -> Result<(), Response> {
    let Some(max_tokens) = output_budget(request_type) else {
        return Ok(());
    };
    let Some(limit) = output_context_length(workers, model_id) else {
        return Ok(());
    };
    enforce_output(Some(max_tokens), limit, model_id)
}

/// The window that holds the completion: the selected worker's, or on a
/// disaggregated pair the decode side's (the prefill side runs one token,
/// `sanitize_sampling_for_prefill`, whatever its own window). `None` when
/// that side advertises none.
fn output_context_length(workers: &WorkerSelection, model_id: &str) -> Option<u32> {
    match workers {
        WorkerSelection::Single { worker } => worker.context_length(model_id),
        WorkerSelection::Disaggregated { decode, .. } => decode.context_length(model_id),
    }
}

/// The completion budget the request names, on the routes that carry one:
/// the body's one reader answers it
/// ([`GenerationRequest::max_output_tokens`]), which the HTTP routes read
/// too, so the two wires state one limit.
pub(crate) fn output_budget(request_type: &RequestType) -> Option<u32> {
    match request_type {
        RequestType::Chat(request) => request.max_output_tokens(),
        RequestType::Completion(request) => request.max_output_tokens(),
        RequestType::Messages(request) => request.max_output_tokens(),
        RequestType::Responses(request) => request.max_output_tokens(),
        RequestType::Generate(request) => request.max_output_tokens(),
        RequestType::Embedding(_)
        | RequestType::Classify(_)
        | RequestType::Decisions(_)
        | RequestType::Transcription { .. } => None,
    }
}

/// The shared output check; absent and zero budgets retain their old meaning.
fn enforce_output(max_tokens: Option<u32>, limit: u32, model_id: &str) -> Result<(), Response> {
    let Some(max_tokens) = max_tokens else {
        return Ok(());
    };
    if max_tokens <= limit {
        return Ok(());
    }
    debug!(
        function = "enforce_output_budget",
        max_tokens,
        limit,
        model_id,
        "Rejecting a completion budget larger than the model's context window"
    );
    Err(error::bad_request(
        CONTEXT_LENGTH_EXCEEDED,
        format!(
            "This model's maximum context length is {limit} tokens. However, you requested \
             {max_tokens} tokens for the completion. Please reduce max_tokens."
        ),
    ))
}

/// Tightest context window across the selected sides, `None` if no side
/// advertises one. A disaggregated prompt must fit both the prefill and the
/// decode engine, so the smaller window governs the input (the completion
/// budget is the decode side's, `output_context_length`). Encode workers
/// process media, not the token prompt, and are not consulted.
pub(crate) fn selection_context_length(workers: &WorkerSelection, model_id: &str) -> Option<u32> {
    match workers {
        WorkerSelection::Single { worker } => worker.context_length(model_id),
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => pair_context_length(prefill.as_ref(), &**decode, model_id),
    }
}

/// Tightest known bound of two token-processing sides. An unknown side adds no
/// bound, but cannot erase the known constraint on its partner.
pub(super) fn pair_context_length(
    prefill: &dyn Worker,
    decode: &dyn Worker,
    model_id: &str,
) -> Option<u32> {
    match (
        prefill.context_length(model_id),
        decode.context_length(model_id),
    ) {
        (Some(prefill), Some(decode)) => Some(prefill.min(decode)),
        (prefill, decode) => prefill.or(decode),
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/stages/context_length/tests.rs"]
mod tests;
