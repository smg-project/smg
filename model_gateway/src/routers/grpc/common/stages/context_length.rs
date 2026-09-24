//! Gateway-side context-window check: reject an input the selected engine
//! cannot hold before it is dispatched.
//!
//! The gRPC path tokenizes at the gateway and ships token ids, so the count
//! here is exactly what the engine would see. Without this check an over-long
//! prompt travels to the engine, which rejects it, and on the disaggregated
//! path that rejection comes back as `500 prefill_worker_failed_to_start`: a
//! server error for a client mistake, blamed on a healthy worker (#2380).
//!
//! Runs once at ingress, after worker selection (the window is a property of
//! the chosen worker's model card) and before any client is acquired.

use axum::response::Response;
use openai_protocol::profile::ProviderProfile;
use tracing::debug;

use crate::routers::{
    error,
    grpc::context::{PreparationOutput, RequestType, WorkerSelection},
};

/// Error code for an input that does not fit the model's window; matches the
/// OpenAI API's code for the same condition.
pub(crate) const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

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
    let Some(limit) = selection_context_length(workers, model_id) else {
        return Ok(());
    };
    let input_tokens = prep.max_input_token_count();
    if input_tokens <= limit as usize {
        return Ok(());
    }
    debug!(
        function = "enforce_context_length",
        input_tokens, limit, model_id, "Rejecting input longer than the model's context window"
    );
    Err(error::bad_request(
        CONTEXT_LENGTH_EXCEEDED,
        format!(
            "This model's maximum context length is {limit} tokens. However, your request has \
             {input_tokens} input tokens. Please reduce the length of the input."
        ),
    ))
}

/// Reject a chat completion budget the model's window could never hold.
///
/// Only under the z.ai profile, whose vendor rejects such a request; every
/// other profile keeps the engine's own policy (vLLM rejects at admission,
/// SGLang clamps), as the input check above leaves `input + max_tokens`
/// to the engine. As above, only a known window is enforced, and like the
/// input check this runs on the gRPC pipeline only: the HTTP router forwards
/// the body and leaves the budget to the engine.
pub(crate) fn enforce_output_budget(
    request_type: &RequestType,
    workers: &WorkerSelection,
    model_id: &str,
) -> Result<(), Response> {
    let RequestType::Chat(request) = request_type else {
        return Ok(());
    };
    if ProviderProfile::for_model(&request.model) != ProviderProfile::Zai {
        return Ok(());
    }
    let Some(limit) = selection_context_length(workers, model_id) else {
        return Ok(());
    };
    #[expect(deprecated, reason = "the request may predate normalization")]
    let Some(max_tokens) = request.max_completion_tokens.or(request.max_tokens) else {
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

/// Tightest context window across the selected legs, `None` if no leg
/// advertises one. A disaggregated prompt must fit both the prefill and the
/// decode engine, so the smaller window governs. Encode workers process media,
/// not the token prompt, and are not consulted.
fn selection_context_length(workers: &WorkerSelection, model_id: &str) -> Option<u32> {
    match workers {
        WorkerSelection::Single { worker } => worker.context_length(model_id),
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => match (
            prefill.context_length(model_id),
            decode.context_length(model_id),
        ) {
            (Some(prefill), Some(decode)) => Some(prefill.min(decode)),
            (prefill, decode) => prefill.or(decode),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        routers::{error::HEADER_X_SMG_ERROR_CODE, grpc::context::CompletionItem},
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
    };

    const MODEL: &str = "ctx-test-model";

    fn worker(worker_type: WorkerType, context_length: Option<u32>) -> Arc<dyn Worker> {
        let mut card = ModelCard::new(MODEL);
        if let Some(len) = context_length {
            card = card.with_context_length(len);
        }
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:1/{worker_type:?}"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Sglang)
                .model(card)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn single(context_length: Option<u32>) -> WorkerSelection {
        WorkerSelection::Single {
            worker: worker(WorkerType::Regular, context_length),
        }
    }

    fn pd(prefill: Option<u32>, decode: Option<u32>) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: worker(WorkerType::Prefill, prefill),
            decode: worker(WorkerType::Decode, decode),
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

    #[test]
    fn a_zai_budget_beyond_the_window_is_a_400_context_length_exceeded() {
        assert_rejected(enforce_output_budget(
            &chat("glm-5.3-flash", Some(10_000_000)),
            &single(Some(131_072)),
            MODEL,
        ));
        assert_rejected(enforce_output_budget(
            &chat("zai-org/GLM-5.3-Flash", Some(131_073)),
            &pd(Some(1_000_000), Some(131_072)),
            MODEL,
        ));
    }

    #[test]
    fn a_zai_budget_within_the_window_or_without_a_window_passes() {
        assert!(enforce_output_budget(
            &chat("glm-5.3-flash", Some(131_072)),
            &single(Some(131_072)),
            MODEL
        )
        .is_ok());
        assert!(
            enforce_output_budget(&chat("glm-5.3-flash", None), &single(Some(131_072)), MODEL)
                .is_ok()
        );
        assert!(enforce_output_budget(
            &chat("glm-5.3-flash", Some(10_000_000)),
            &single(None),
            MODEL
        )
        .is_ok());
    }

    #[test]
    fn other_profiles_leave_the_budget_to_the_engine() {
        for model in ["gpt-4o", "kimi-k3", "MiniMax-M3"] {
            assert!(
                enforce_output_budget(
                    &chat(model, Some(10_000_000)),
                    &single(Some(131_072)),
                    MODEL
                )
                .is_ok(),
                "{model}"
            );
        }
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
    fn disaggregated_uses_the_tighter_leg() {
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
    fn disaggregated_with_one_unknown_leg_uses_the_known_one() {
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
        let selection = WorkerSelection::Single { worker };
        assert!(enforce_context_length(&input(10), &selection, "wide").is_ok());
        assert_rejected(enforce_context_length(&input(10), &selection, "narrow"));
        // An id no card matches falls back to the primary (first) card.
        assert!(enforce_context_length(&input(10), &selection, "unlisted").is_ok());
    }
}
