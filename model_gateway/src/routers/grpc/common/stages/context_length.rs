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
    /// selection states to the policy engine as the answer's limit. `None`
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
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use openai_protocol::{
        chat::ChatCompletionRequest, model_card::ModelCard, validated::Normalizable,
        worker::HealthCheckConfig,
    };
    use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType};

    use super::*;
    use crate::routers::{
        common::snapshot::{WorkerRegistry, WorkerRegistryExt},
        error::HEADER_X_SMG_ERROR_CODE,
        grpc::{backend_client::BasicWorkerBuilder, context::CompletionItem},
    };

    const MODEL: &str = "ctx-test-model";

    fn worker(worker_type: WorkerType, context_length: Option<u32>) -> Arc<dyn Worker> {
        kind(worker_type, ConnectionMode::Grpc, context_length)
    }

    fn kind(
        worker_type: WorkerType,
        mode: ConnectionMode,
        context_length: Option<u32>,
    ) -> Arc<dyn Worker> {
        let mut card = ModelCard::new(MODEL);
        if let Some(len) = context_length {
            card = card.with_context_length(len);
        }
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:1/{worker_type:?}"))
                .worker_type(worker_type)
                .connection_mode(mode)
                .runtime_type(RuntimeType::Sglang)
                .model(card)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// What the kept floors of a snapshot's pools leave of a demand:
    /// nothing where every window of every pool holds it; the demand whole
    /// where a window of one pool refuses it, and where a floor was kept for
    /// another spelling of the model id.
    #[test]
    fn a_demand_is_left_whole_unless_every_pools_floor_holds_it() {
        let registry = WorkerRegistry::new();
        for (port, worker_type, window) in [
            (8861, WorkerType::Prefill, 64),
            (8862, WorkerType::Decode, 256),
        ] {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Sglang)
                        .model(
                            ModelCard::new(MODEL)
                                .with_alias("ctx-test-alias")
                                .with_context_length(window),
                        )
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .unwrap();
        }
        let snapshot = registry.get_routing_snapshot(MODEL);
        let demand = |input_tokens, budget| ContextRequirements {
            input_tokens,
            scoring_position: false,
            output_tokens: Some(budget),
        };
        let nothing = ContextRequirements::default();
        let (prefill, decode) = (RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode);

        // Both floors hold it: the prefill window the input, the decode
        // window the input and the budget.
        let held = demand(64, 256);
        assert_eq!(held.left_by(&snapshot, &[prefill, decode], MODEL), nothing);
        // The prefill window refuses the input, which the decode window holds.
        let long = demand(65, 256);
        assert_eq!(long.left_by(&snapshot, &[prefill, decode], MODEL), long);
        assert_eq!(long.left_by(&snapshot, &[decode, prefill], MODEL), long);
        assert_eq!(long.left_by(&snapshot, &[decode], MODEL), nothing);
        // The decode window refuses the budget, which binds no prefill worker.
        let wide = demand(64, 257);
        assert_eq!(wide.left_by(&snapshot, &[prefill, decode], MODEL), wide);
        assert_eq!(wide.left_by(&snapshot, &[prefill], MODEL), nothing);
        // A floor kept for another spelling answers nothing: the workers
        // are asked.
        assert_eq!(
            held.left_by(&snapshot, &[prefill, decode], "ctx-test-alias"),
            held
        );
        // A pool with no worker binds nothing, and a demand that binds
        // nobody is left as it is.
        assert_eq!(
            long.left_by(&snapshot, &[RoutingPool::GrpcEncode], MODEL),
            nothing
        );
        assert_eq!(
            nothing.left_by(&snapshot, &[prefill, decode], "ctx-test-alias"),
            nothing
        );
    }

    fn single(context_length: Option<u32>) -> WorkerSelection {
        WorkerSelection::Single {
            worker: worker(WorkerType::Regular, context_length).into(),
        }
    }

    fn pd(prefill: Option<u32>, decode: Option<u32>) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: worker(WorkerType::Prefill, prefill),
            decode: worker(WorkerType::Decode, decode).into(),
            runtime_type: RuntimeType::Sglang,
        }
    }

    fn input(tokens: usize) -> PreparationOutput {
        PreparationOutput::Generate {
            original_text: None,
            token_ids: vec![7; tokens],
        }
    }

    fn assert_rejected(result: Result<(), Response>) {
        let response = result.expect_err("over-long input must be rejected");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response
                .headers()
                .get(HEADER_X_SMG_ERROR_CODE)
                .and_then(|v| v.to_str().ok()),
            Some(CONTEXT_LENGTH_EXCEEDED)
        );
    }

    fn chat(model: &str, max_tokens: Option<u32>) -> RequestType {
        let mut body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        RequestType::Chat(Arc::new(
            serde_json::from_value(body).expect("chat request deserializes"),
        ))
    }

    fn completion(max_tokens: u32) -> RequestType {
        RequestType::Completion(Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL, "prompt": "hi", "max_tokens": max_tokens
            }))
            .expect("completion request deserializes"),
        ))
    }

    fn messages(max_tokens: u32) -> RequestType {
        RequestType::Messages(Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .expect("messages request deserializes"),
        ))
    }

    fn responses(max_output_tokens: u32) -> RequestType {
        RequestType::Responses(Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL, "input": "hi", "max_output_tokens": max_output_tokens
            }))
            .expect("responses request deserializes"),
        ))
    }

    fn generate(max_new_tokens: u32) -> RequestType {
        RequestType::Generate(Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL, "text": "hi", "sampling_params": {"max_new_tokens": max_new_tokens}
            }))
            .expect("generate request deserializes"),
        ))
    }

    fn embedding() -> RequestType {
        RequestType::Embedding(Arc::new(
            serde_json::from_value(serde_json::json!({"model": MODEL, "input": "hi"}))
                .expect("embedding request deserializes"),
        ))
    }

    /// Selection shares both the budget bound and the overflow-safe scoring
    /// boundary with the existing post-selection checks.
    #[test]
    fn context_requirements_preserve_budget_and_scoring_boundaries() {
        for budget in [None, Some(0), Some(16)] {
            let request = chat("ordinary-model", budget);
            let requirement = ContextRequirements::new(&input(16), &request);
            assert!(requirement.enforce(16, MODEL).is_ok());
        }
        let request = chat("ordinary-model", Some(17));
        assert_rejected(ContextRequirements::new(&input(16), &request).enforce(16, MODEL));
        let requirement = ContextRequirements {
            input_tokens: usize::MAX,
            scoring_position: true,
            output_tokens: Some(u32::MAX),
        };
        assert_rejected(requirement.enforce(u32::MAX, MODEL));
        let requirement = ContextRequirements {
            input_tokens: 16,
            scoring_position: true,
            output_tokens: None,
        };
        assert_rejected(requirement.enforce(16, MODEL));
        assert!(requirement.enforce(17, MODEL).is_ok());
    }

    /// A budget the window could never hold is refused whatever the model's
    /// provider profile: the check the z.ai profile had is every profile's.
    #[test]
    fn a_budget_beyond_the_window_is_a_400_context_length_exceeded() {
        for model in [
            "glm-5.3-flash",
            "zai-org/GLM-5.3-Flash",
            "gpt-4o",
            "kimi-k3",
            "MiniMax-M3",
        ] {
            assert_rejected(enforce_output_budget(
                &chat(model, Some(10_000_000)),
                &single(Some(131_072)),
                MODEL,
            ));
            assert_rejected(enforce_output_budget(
                &chat(model, Some(131_073)),
                &pd(Some(1_000_000), Some(131_072)),
                MODEL,
            ));
        }
    }

    /// An omitted Flash budget leaves a smaller worker window available.
    #[test]
    fn glm53_flash_chat_omitted_budget_preserves_the_worker_window() {
        let mut request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        request.normalize();
        assert_eq!(request.output_token_cap(), None);
        let request = RequestType::Chat(Arc::new(request));
        assert!(enforce_output_budget(&request, &single(Some(32_768)), MODEL).is_ok());
    }

    #[test]
    fn a_budget_within_the_window_or_without_a_window_passes() {
        for model in ["glm-5.3-flash", "gpt-4o"] {
            assert!(
                enforce_output_budget(&chat(model, Some(131_072)), &single(Some(131_072)), MODEL)
                    .is_ok(),
                "{model}"
            );
            assert!(
                enforce_output_budget(&chat(model, None), &single(Some(131_072)), MODEL).is_ok(),
                "{model}"
            );
            assert!(
                enforce_output_budget(&chat(model, Some(10_000_000)), &single(None), MODEL).is_ok(),
                "{model}"
            );
        }
    }

    /// A route's request with the budget it names.
    type Budgeted = fn(u32) -> RequestType;

    /// Every route that names a completion budget is bounded by the window
    /// (the budget alone: the input's share stays with the engine); a route
    /// without one names nothing to bound.
    #[test]
    fn every_route_that_names_a_budget_is_bounded_by_the_window() {
        let routes: [(&str, Budgeted); 4] = [
            ("completion", completion),
            ("messages", messages),
            ("responses", responses),
            ("generate", generate),
        ];
        for (route, request) in routes {
            assert_rejected(enforce_output_budget(
                &request(17),
                &single(Some(16)),
                MODEL,
            ));
            assert!(
                enforce_output_budget(&request(16), &single(Some(16)), MODEL).is_ok(),
                "{route}: a budget that fits"
            );
            assert!(
                enforce_output_budget(&request(17), &single(None), MODEL).is_ok(),
                "{route}: an unknown window"
            );
        }
        assert!(enforce_output_budget(&embedding(), &single(Some(16)), MODEL).is_ok());
    }

    #[tokio::test]
    async fn budget_rejection_body_names_the_limit_and_the_budget() {
        let response =
            enforce_output_budget(&chat("any-model", Some(40)), &single(Some(16)), MODEL)
                .expect_err("a budget beyond the window must be rejected");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED);
        let message = json["error"]["message"].as_str().expect("message");
        assert!(message.contains("16 tokens"), "{message}");
        assert!(
            message.contains("40 tokens for the completion"),
            "{message}"
        );
        assert!(message.contains("max_tokens"), "{message}");
    }

    #[test]
    fn input_within_the_window_passes() {
        assert!(enforce_context_length(&input(8), &single(Some(16)), MODEL).is_ok());
    }

    /// The window is a capacity, not a strict bound: a prompt that exactly
    /// fills it is not "longer than" it, and whether the engine can still
    /// generate from it is the engine's call (embeddings, for one, can).
    #[test]
    fn input_exactly_at_the_window_passes() {
        assert!(enforce_context_length(&input(16), &single(Some(16)), MODEL).is_ok());
    }

    #[test]
    fn input_over_the_window_is_a_400_context_length_exceeded() {
        assert_rejected(enforce_context_length(&input(17), &single(Some(16)), MODEL));
    }

    #[tokio::test]
    async fn rejection_body_names_the_limit_and_the_actual_count() {
        let response = enforce_context_length(&input(40), &single(Some(16)), MODEL)
            .expect_err("over-long input must be rejected");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED);
        let message = json["error"]["message"].as_str().expect("message");
        assert!(message.contains("16 tokens"), "{message}");
        assert!(message.contains("40 input tokens"), "{message}");
        assert!(
            !message.contains("worker"),
            "the client mistake must not be blamed on a worker: {message}"
        );
    }

    /// No advertised window means no gateway opinion: the engine keeps
    /// enforcing its own limit exactly as before this check existed.
    #[test]
    fn unknown_window_is_not_enforced() {
        assert!(enforce_context_length(&input(1_000_000), &single(None), MODEL).is_ok());
        assert!(enforce_context_length(&input(1_000_000), &pd(None, None), MODEL).is_ok());
    }

    /// A disaggregated prompt has to fit both engines; the tighter one wins.
    #[test]
    fn disaggregated_uses_the_tighter_side() {
        assert!(enforce_context_length(&input(10), &pd(Some(16), Some(16)), MODEL).is_ok());
        assert_rejected(enforce_context_length(
            &input(10),
            &pd(Some(16), Some(8)),
            MODEL,
        ));
        assert_rejected(enforce_context_length(
            &input(10),
            &pd(Some(8), Some(16)),
            MODEL,
        ));
    }

    #[test]
    fn disaggregated_with_one_unknown_side_uses_the_known_one() {
        assert!(enforce_context_length(&input(10), &pd(None, Some(16)), MODEL).is_ok());
        assert_rejected(enforce_context_length(
            &input(10),
            &pd(Some(8), None),
            MODEL,
        ));
        assert_rejected(enforce_context_length(
            &input(10),
            &pd(None, Some(8)),
            MODEL,
        ));
    }

    /// Each batched prompt is its own engine request: a batch of short prompts
    /// whose *sum* exceeds the window is fine, one long prompt anywhere in the
    /// batch is not.
    #[test]
    fn batched_completion_is_bounded_per_item() {
        let batch = |lens: &[usize]| PreparationOutput::Completion {
            items: lens
                .iter()
                .map(|&len| CompletionItem {
                    text: String::new(),
                    token_ids: vec![7; len],
                })
                .collect(),
            joined_routing_text: None,
        };
        assert!(enforce_context_length(&batch(&[10, 10, 10]), &single(Some(16)), MODEL).is_ok());
        assert_rejected(enforce_context_length(
            &batch(&[4, 17]),
            &single(Some(16)),
            MODEL,
        ));
    }

    /// The window is looked up for the model actually requested, so a worker
    /// serving several cards with different windows is bounded by the right one.
    #[test]
    fn window_is_resolved_for_the_requested_model_card() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1/multi")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Sglang)
                .models(vec![
                    ModelCard::new("wide").with_context_length(64),
                    ModelCard::new("narrow").with_context_length(8),
                ])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let selection = WorkerSelection::Single {
            worker: worker.into(),
        };
        assert!(enforce_context_length(&input(10), &selection, "wide").is_ok());
        assert_rejected(enforce_context_length(&input(10), &selection, "narrow"));
        // An id no card matches falls back to the primary (first) card.
        assert!(enforce_context_length(&input(10), &selection, "unlisted").is_ok());
    }

    /// A prefill worker is held to the input alone: its decode partner holds
    /// the completion budget. The same budget on a regular or a decode worker
    /// with the same window is refused, and the input still binds the
    /// prefill side.
    #[test]
    fn a_prefill_worker_is_held_to_the_input_alone() {
        let requirements = ContextRequirements::new(&input(1024), &completion(8192));
        assert!(requirements.admits(worker(WorkerType::Prefill, Some(4096)).as_ref(), MODEL));
        assert!(!requirements.admits(worker(WorkerType::Decode, Some(4096)).as_ref(), MODEL));
        assert!(!requirements.admits(worker(WorkerType::Regular, Some(4096)).as_ref(), MODEL));
        assert!(requirements.admits(worker(WorkerType::Decode, Some(32768)).as_ref(), MODEL));
        let over_long = ContextRequirements::new(&input(5000), &completion(1));
        assert!(
            !over_long.admits(worker(WorkerType::Prefill, Some(4096)).as_ref(), MODEL),
            "the input still binds the prefill side"
        );
    }

    /// On a disaggregated pair the budget is bound by the decode side's
    /// window and the input by the tighter of the two.
    #[test]
    fn the_pair_budget_is_bound_by_the_decode_window() {
        let pair = pd(Some(4096), Some(32768));
        assert!(enforce_output_budget(&completion(8192), &pair, MODEL).is_ok());
        assert_rejected(enforce_output_budget(&completion(40000), &pair, MODEL));
        assert!(enforce_context_length(&input(1024), &pair, MODEL).is_ok());
        assert_rejected(enforce_context_length(&input(5000), &pair, MODEL));
        // The selection-time refusal of a pair nothing fits names the same windows.
        assert_rejected(
            ContextRequirements::new(&input(1024), &completion(40000)).enforce_for_pair(
                Some(4096),
                Some(32768),
                MODEL,
            ),
        );
        assert!(ContextRequirements::new(&input(1024), &completion(8192))
            .enforce_for_pair(Some(4096), Some(32768), MODEL)
            .is_ok());
        assert_rejected(
            ContextRequirements::new(&input(5000), &completion(1)).enforce_for_pair(
                Some(4096),
                None,
                MODEL,
            ),
        );
    }

    #[test]
    fn a_lists_floor_answers_as_its_workers_do() {
        // Every kind of worker the rule tells apart, alone, in pairs (a
        // minimum over a class is decided between two) and in triples of
        // the two classes, against every demand around the windows' edges:
        // each worker answers as the rule spelled out below, and the
        // floor's answer is the workers' own, both ways.
        //
        // The rule: only a gRPC worker with a known window can refuse. It
        // refuses an input that does not fit, and a budget above its
        // window unless it is a prefill worker.
        let refuses = |demand: &ContextRequirements, worker: &dyn Worker| {
            let Some(window) = worker.context_length(MODEL) else {
                return false;
            };
            if *worker.connection_mode() != ConnectionMode::Grpc {
                return false;
            }
            let input_over = if demand.scoring_position {
                demand.input_tokens >= window as usize
            } else {
                demand.input_tokens > window as usize
            };
            let budget_over = *worker.worker_type() != WorkerType::Prefill
                && demand.output_tokens.is_some_and(|tokens| tokens > window);
            input_over || budget_over
        };
        let windows = [None, Some(32), Some(64), Some(65), Some(256), Some(257)];
        let modes = [
            ConnectionMode::Grpc,
            ConnectionMode::Http,
            ConnectionMode::Zmq,
        ];
        let types = [
            WorkerType::Regular,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
        ];
        let mut kinds = Vec::new();
        for window in windows {
            for mode in modes {
                for worker_type in types {
                    kinds.push(kind(worker_type, mode, window));
                }
            }
        }
        let mut pools: Vec<Vec<Arc<dyn Worker>>> = Vec::new();
        for (i, first) in kinds.iter().enumerate() {
            pools.push(vec![Arc::clone(first)]);
            for second in &kinds[i + 1..] {
                pools.push(vec![Arc::clone(first), Arc::clone(second)]);
            }
        }
        // Three workers: a floor that kept the last one alone would pass
        // some pairs. The gRPC kinds of both classes, in every order.
        let bound: Vec<Arc<dyn Worker>> = [None, Some(32), Some(64), Some(257)]
            .into_iter()
            .flat_map(|window| {
                [WorkerType::Regular, WorkerType::Prefill]
                    .map(|worker_type| kind(worker_type, ConnectionMode::Grpc, window))
            })
            .collect();
        for first in &bound {
            for second in &bound {
                for third in &bound {
                    pools.push(vec![
                        Arc::clone(first),
                        Arc::clone(second),
                        Arc::clone(third),
                    ]);
                }
            }
        }
        let mut demands = Vec::new();
        for input_tokens in [0, 63, 64, 65, 256, 300] {
            for scoring_position in [false, true] {
                for output_tokens in [None, Some(1), Some(255), Some(256), Some(257)] {
                    demands.push(ContextRequirements {
                        input_tokens,
                        scoring_position,
                        output_tokens,
                    });
                }
            }
        }
        let (mut held, mut refused) = (0usize, 0usize);
        for pool in &pools {
            let floor = window_floor_of(pool, MODEL);
            for demand in &demands {
                for worker in pool {
                    assert_eq!(
                        demand.admits(worker.as_ref(), MODEL),
                        !refuses(demand, worker.as_ref()),
                        "{demand:?} at {:?}",
                        (
                            worker.worker_type(),
                            worker.connection_mode(),
                            worker.context_length(MODEL)
                        )
                    );
                }
                let every_worker_admits = pool
                    .iter()
                    .all(|worker| demand.admits(worker.as_ref(), MODEL));
                assert_eq!(
                    demand.held_by_all(floor),
                    every_worker_admits,
                    "{demand:?} over {floor:?}: {:?}",
                    pool.iter()
                        .map(|worker| (
                            worker.worker_type(),
                            worker.connection_mode(),
                            worker.context_length(MODEL)
                        ))
                        .collect::<Vec<_>>()
                );
                if every_worker_admits {
                    held += 1;
                } else {
                    refused += 1;
                }
            }
        }
        // The cross product reaches both answers many times over.
        assert!(
            held > 1_000 && refused > 1_000,
            "{held} held, {refused} refused"
        );
    }
}
