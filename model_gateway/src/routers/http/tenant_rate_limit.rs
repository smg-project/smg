//! Tenant rate limiting on the regular HTTP router: what a request reserves,
//! and how its answer resolves the reservation.
//!
//! `Router::route_typed_request` reserves once per logical request, before
//! any dispatch (every attempt dispatches under that one reservation), and
//! resolves it once, on the one response its no-retry path and its retry
//! loop converge on ([`resolve`]). This router has no tokenizer, so the
//! reserve-time estimate is the request's own token ids when it is
//! pre-tokenized and 1.3 tokens per word of its text otherwise, the
//! heuristic `pd_router.rs` ranks DP workers by ([`estimated_input_tokens`]).
//! A buffered success settles the counts the worker reports in its JSON
//! body; a streamed success is not parsed and keeps its estimate, as do a
//! buffered success whose usage cannot be read and every failure.

use axum::{
    body::{to_bytes, Body},
    response::Response,
};
use openai_protocol::common::GenerationRequest;
use serde_json::Value;

use crate::{
    rate_limit::{ReservationAttachment, UsageSettlement},
    routers::{common::rate_limit_cell::RateLimitCell, error},
    worker::GuardedStream,
};

/// Estimated tokens per whitespace-separated word of a request's text.
const TOKENS_PER_WORD: f64 = 1.3;

/// The reserve-time estimate of `request`'s input tokens: the count of its
/// own token ids when it is pre-tokenized, otherwise 1.3 per word of its
/// routing text. `routing_text` is the text routing already extracted; it is
/// absent when routing keys on tokens, and then a request that carries no
/// ids of its own was routed on the client's `x-smg-routing-tokens` hint. A
/// hint says nothing the gateway can hold the tenant to, so the estimate
/// reads the body's text instead.
pub(super) fn estimated_input_tokens<T: GenerationRequest>(
    request: &T,
    routing_text: Option<&str>,
) -> u32 {
    if let Some(ids) = request.routing_tokens() {
        return u32::try_from(ids.len()).unwrap_or(u32::MAX);
    }
    let words = match routing_text {
        Some(text) => text.split_whitespace().count(),
        None => request
            .extract_text_for_routing()
            .split_whitespace()
            .count(),
    };
    (words as f64 * TOKENS_PER_WORD).ceil() as u32
}

/// Resolve `cell`'s reservation on the answer of a request to `route`, once:
/// a buffered success settles the usage its body reports (or closes at the
/// estimate when the body reports none it can read), a streamed success
/// hands the reservation to its body, which keeps the estimate when it
/// drops, and any other answer closes it at the estimate. The answer is
/// returned as the client would have had it.
pub(super) async fn resolve(
    cell: &RateLimitCell,
    route: &str,
    is_stream: bool,
    response: Response,
    max_body_bytes: usize,
) -> Response {
    if !response.status().is_success() {
        cell.close_unsettled().await;
        return response;
    }
    if is_stream {
        return match cell.take_for_streaming_handoff() {
            Some(handle) => hold_in_body(response, ReservationAttachment::new(handle)),
            None => response,
        };
    }

    // The buffered answer is already in memory, read capped at the same
    // limit (`Router::forward_worker_response`): this takes the bytes back.
    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, max_body_bytes).await {
        Ok(bytes) => bytes,
        Err(e) => {
            cell.close_unsettled().await;
            return error::internal_error(
                "read_response_body_failed",
                format!("Failed to get response body: {e}"),
            );
        }
    };
    match usage_from_body(route, &bytes) {
        Some(usage) => cell.settle(usage).await,
        None => cell.close_unsettled().await,
    }
    Response::from_parts(parts, Body::from(bytes))
}

/// `response` with `attachment` riding in its body's own stream
/// ([`GuardedStream`]), dropped with the body at its end or when the client
/// leaves. The relay builds its body per dispatch attempt, before the
/// request's answer is known, so the attachment goes over the answer the
/// attempts converged on: one stream layer more per frame, on a tenant's
/// streamed answer only.
fn hold_in_body(response: Response, attachment: ReservationAttachment) -> Response {
    let (parts, body) = response.into_parts();
    let body = Body::from_stream(GuardedStream::new(body.into_data_stream(), attachment));
    Response::from_parts(parts, body)
}

/// The usage a worker's buffered JSON answer to `route` reports: SGLang's
/// `meta_info` on `/generate` ([`generate_usage`]), and on every other route
/// a `usage` object in the OpenAI shape (`prompt_tokens` and
/// `completion_tokens`: chat, completions, embeddings, classify) or the
/// Anthropic one (`input_tokens` and `output_tokens`: messages, Responses).
/// `None` for a body that is not JSON or reports neither pair whole. The
/// engine's cached and reasoning split is not read; it takes no part in the
/// true-up.
fn usage_from_body(route: &str, body: &[u8]) -> Option<UsageSettlement> {
    let value: Value = serde_json::from_slice(body).ok()?;
    if route == "/generate" {
        return generate_usage(&value);
    }
    let usage = value.get("usage")?;
    let (input, output) = token_pair(usage, "prompt_tokens", "completion_tokens")
        .or_else(|| token_pair(usage, "input_tokens", "output_tokens"))?;
    Some(UsageSettlement {
        actual_input_tokens: input,
        completion_tokens: output,
        ..UsageSettlement::default()
    })
}

/// The usage of SGLang's native `/generate` answer: one object, or an array
/// of them for a request of several generations, each with its own
/// `meta_info.prompt_tokens` and `meta_info.completion_tokens`. The prompt
/// is counted once, at its largest (the generations of one `n>1` request
/// share it), and the completions sum: each is a generation of its own.
fn generate_usage(value: &Value) -> Option<UsageSettlement> {
    let items = match value {
        Value::Array(items) if !items.is_empty() => items.as_slice(),
        Value::Array(_) => return None,
        item => std::slice::from_ref(item),
    };
    let mut usage = UsageSettlement::default();
    for item in items {
        let (prompt, completion) =
            token_pair(item.get("meta_info")?, "prompt_tokens", "completion_tokens")?;
        usage.actual_input_tokens = usage.actual_input_tokens.max(prompt);
        usage.completion_tokens = usage.completion_tokens.checked_add(completion)?;
    }
    Some(usage)
}

/// The two counts `object` reports under `input` and `output`, when both
/// are there as `u32`s.
fn token_pair(object: &Value, input: &str, output: &str) -> Option<(u32, u32)> {
    let count = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
    };
    Some((count(input)?, count(output)?))
}

#[cfg(test)]
mod tests {
    use openai_protocol::{chat::ChatCompletionRequest, generate::GenerateRequest};
    use serde_json::json;

    use super::*;

    fn usage(input: u32, output: u32) -> UsageSettlement {
        UsageSettlement {
            actual_input_tokens: input,
            completion_tokens: output,
            ..UsageSettlement::default()
        }
    }

    fn parse(route: &str, body: Value) -> Option<UsageSettlement> {
        usage_from_body(route, body.to_string().as_bytes())
    }

    #[test]
    fn each_routes_usage_shape_is_read() {
        let openai =
            json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}});
        for route in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/classify",
        ] {
            assert_eq!(parse(route, openai.clone()), Some(usage(7, 3)), "{route}");
        }
        let anthropic = json!({"usage": {"input_tokens": 5, "output_tokens": 2}});
        for route in ["/v1/messages", "/v1/responses"] {
            assert_eq!(
                parse(route, anthropic.clone()),
                Some(usage(5, 2)),
                "{route}"
            );
        }
        assert_eq!(
            parse(
                "/generate",
                json!({"text": "x", "meta_info": {"prompt_tokens": 4, "completion_tokens": 9}})
            ),
            Some(usage(4, 9))
        );
    }

    /// A batch counts its shared prompt once, at its largest, and sums the
    /// completions.
    #[test]
    fn a_generate_batch_counts_the_prompt_once_and_sums_the_completions() {
        let batch = json!([
            {"meta_info": {"prompt_tokens": 4, "completion_tokens": 9}},
            {"meta_info": {"prompt_tokens": 6, "completion_tokens": 1}},
        ]);
        assert_eq!(parse("/generate", batch), Some(usage(6, 10)));
    }

    #[test]
    fn usage_that_cannot_be_read_whole_is_none() {
        for (route, body) in [
            ("/v1/chat/completions", json!({"choices": []})),
            ("/v1/chat/completions", json!({"usage": null})),
            (
                "/v1/chat/completions",
                json!({"usage": {"prompt_tokens": 7}}),
            ),
            (
                "/v1/chat/completions",
                json!({"usage": {"prompt_tokens": -1, "completion_tokens": 3}}),
            ),
            (
                "/v1/messages",
                json!({"usage": {"input_tokens": "5", "output_tokens": 2}}),
            ),
            (
                "/v1/completions",
                json!({"usage": {"prompt_tokens": u64::from(u32::MAX) + 1, "completion_tokens": 0}}),
            ),
            (
                "/generate",
                json!({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            ),
            ("/generate", json!([])),
            (
                "/generate",
                json!([{"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}}, {}]),
            ),
            (
                "/generate",
                json!([
                    {"meta_info": {"prompt_tokens": 1, "completion_tokens": u32::MAX}},
                    {"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}},
                ]),
            ),
        ] {
            assert_eq!(parse(route, body.clone()), None, "{route}: {body}");
        }
        assert_eq!(usage_from_body("/v1/chat/completions", b"not json"), None);
    }

    fn generate(body: Value) -> GenerateRequest {
        serde_json::from_value(body).unwrap()
    }

    /// Text is estimated at 1.3 tokens per word, rounded up; a pre-tokenized
    /// request at its own ids, whatever text routing saw.
    #[test]
    fn the_estimate_is_the_requests_own_ids_or_its_words() {
        let text = generate(json!({"model": "m", "text": "one two three four five six seven"}));
        assert_eq!(
            estimated_input_tokens(&text, Some("one two three four five six seven")),
            10
        );
        assert_eq!(estimated_input_tokens(&text, Some("")), 0);

        let ids = generate(json!({"model": "m", "input_ids": [1, 2, 3, 4, 5]}));
        assert_eq!(estimated_input_tokens(&ids, None), 5);
    }

    /// Routing on the client's tokens hint extracts no text; the estimate
    /// reads the body's own, not the hint's length.
    #[test]
    fn a_routing_hint_does_not_set_the_estimate() {
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "one two three four five six seven"}],
        }))
        .unwrap();
        assert_eq!(estimated_input_tokens(&chat, None), 10);
    }
}
