//! Tenant rate limiting on the routers that relay a request as HTTP: what a
//! request reserves, and how its answer resolves the reservation.
//!
//! The regular HTTP router (`Router::route_typed_request`), the
//! disaggregated HTTP router (`PDRouter::execute_dual_dispatch`) and the
//! external routers' gateway side (`ExternalRouterAdapter::route_reserved`)
//! each [`reserve`] once per logical request, before any dispatch (every
//! attempt dispatches under that one reservation), and [`resolve`] it once,
//! on the one response their no-retry path and their retry loop converge
//! on. These routers have no tokenizer, so the reserve-time estimate
//! ([`TenantCharge::estimated_input_tokens`]) counts every token id a
//! pre-tokenized input carries and 1.3 tokens per word of every text input
//! otherwise, the heuristic `pd_router.rs` ranks DP workers by. A buffered
//! success settles the counts the worker reports in its JSON body; a
//! streamed success is not parsed and keeps its estimate, as do a buffered
//! success whose usage cannot be read and every failure.

use axum::{
    body::{to_bytes, Body},
    response::Response,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage},
    classify::ClassifyRequest,
    common::{GenerationRequest, InputIds},
    completion::CompletionRequest,
    decisions::DecisionsRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::{
        CountMessageTokensRequest, CreateMessageRequest, InputContent, InputContentBlock,
        ToolResultContent, ToolResultContentBlock,
    },
    rerank::RerankRequest,
    responses::{
        CustomToolCallOutputContent, CustomToolInputContentPart, ResponseContentPart,
        ResponseInput, ResponseInputOutputItem, ResponsesRequest, StringOrContentParts,
    },
    systemone::SystemOneRequest,
};
use serde_json::Value;

#[cfg(test)]
use crate::routers::common::request_lease::test_probe::DropProbeRequest;
use crate::{
    middleware::TenantRequestMeta,
    rate_limit::{RateLimitManager, ReservationAttachment, UsageSettlement},
    routers::{
        common::{rate_limit_cell::RateLimitCell, retry::mark_non_retryable},
        error,
    },
    worker::GuardedStream,
};

/// Estimated tokens per whitespace-separated word of a request's text.
const TOKENS_PER_WORD: f64 = 1.3;

/// What a typed request charges its tenant: the input it reserves before
/// dispatch, and the shape its `/generate` answer reports usage in. Every
/// request type the HTTP routers and the external routers take implements
/// it; the routes that never reserve take the defaults, which they never
/// ask for.
pub(crate) trait TenantCharge: GenerationRequest {
    /// The reserve-time estimate of the request's input tokens. By default,
    /// 1.3 per word of its routing text -- `routing_text` when routing
    /// already extracted it, the body's own otherwise -- and of the input
    /// that text leaves out ([`Self::unrouted_words`]). `routing_text` is
    /// absent when routing keys on tokens, and for a request that carries no
    /// ids of its own that is the client's `x-smg-routing-tokens` hint: a
    /// hint says nothing the gateway can hold the tenant to, so the estimate
    /// reads the body.
    fn estimated_input_tokens(&self, routing_text: Option<&str>) -> u32 {
        let words = match routing_text {
            Some(text) => word_count(text),
            None => word_count(&self.extract_text_for_routing()),
        };
        tokens_for_words(words + self.unrouted_words())
    }

    /// The words of the request's input that its routing text leaves out.
    fn unrouted_words(&self) -> usize {
        0
    }

    /// How many answers each of the request's prompts gets: the entries of
    /// a `/generate` batch answer that repeat one prompt's count.
    fn samples_per_prompt(&self) -> u32 {
        1
    }
}

/// The arguments of the assistant's tool calls in the history.
impl TenantCharge for ChatCompletionRequest {
    fn unrouted_words(&self) -> usize {
        self.messages
            .iter()
            .filter_map(|message| match message {
                ChatMessage::Assistant {
                    tool_calls: Some(calls),
                    ..
                } => Some(calls),
                _ => None,
            })
            .flatten()
            .filter_map(|call| call.function.arguments.as_deref())
            .map(word_count)
            .sum()
    }
}

impl TenantCharge for CompletionRequest {}

/// The input of each `tool_use` block in the history and the text of each
/// `tool_result`, a string or text blocks; not their images or documents.
impl TenantCharge for CreateMessageRequest {
    fn unrouted_words(&self) -> usize {
        self.messages
            .iter()
            .filter_map(|message| match &message.content {
                InputContent::Blocks(blocks) => Some(blocks),
                InputContent::String(_) => None,
            })
            .flatten()
            .map(|block| match block {
                InputContentBlock::ToolUse(tool_use) => json_words(&tool_use.input),
                InputContentBlock::ToolResult(result) => match &result.content {
                    Some(ToolResultContent::String(text)) => word_count(text),
                    Some(ToolResultContent::Blocks(blocks)) => blocks
                        .iter()
                        .map(|block| match block {
                            ToolResultContentBlock::Text(text) => word_count(&text.text),
                            _ => 0,
                        })
                        .sum(),
                    None => 0,
                },
                _ => 0,
            })
            .sum()
    }
}

/// Every token id of every sequence: routing keys on the first sequence of
/// a batch only, but each is a prompt of its own. Without ids, the words
/// of its text. A request of `n` samples prefills each prompt once.
impl TenantCharge for GenerateRequest {
    fn estimated_input_tokens(&self, _routing_text: Option<&str>) -> u32 {
        let ids = match &self.input_ids {
            Some(InputIds::Single(ids)) => ids.len(),
            Some(InputIds::Batch(sequences)) => sequences.iter().map(Vec::len).sum(),
            None => 0,
        };
        if ids > 0 {
            return u32::try_from(ids).unwrap_or(u32::MAX);
        }
        tokens_for_words(self.text.as_deref().map_or(0, word_count))
    }

    fn samples_per_prompt(&self) -> u32 {
        self.sampling_params
            .as_ref()
            .and_then(|params| params.n)
            .unwrap_or(1)
            .max(1)
    }
}

/// The instructions, and the function and custom tool calls in the history
/// with their outputs. Images, files and the built-in tools' items are not
/// counted.
impl TenantCharge for ResponsesRequest {
    fn unrouted_words(&self) -> usize {
        let instructions = self.instructions.as_deref().map_or(0, word_count);
        let ResponseInput::Items(items) = &self.input else {
            return instructions;
        };
        let history: usize = items
            .iter()
            .map(|item| match item {
                ResponseInputOutputItem::FunctionToolCall {
                    arguments, output, ..
                } => word_count(arguments) + output.as_deref().map_or(0, word_count),
                ResponseInputOutputItem::FunctionCallOutput { output, .. } => match output {
                    StringOrContentParts::String(text) => word_count(text),
                    StringOrContentParts::Array(parts) => parts
                        .iter()
                        .filter_map(ResponseContentPart::text)
                        .map(word_count)
                        .sum(),
                },
                ResponseInputOutputItem::CustomToolCall { input, .. } => word_count(input),
                ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
                    CustomToolCallOutputContent::Text(text) => word_count(text),
                    CustomToolCallOutputContent::Parts(parts) => parts
                        .iter()
                        .map(|part| match part {
                            CustomToolInputContentPart::InputText { text } => word_count(text),
                            _ => 0,
                        })
                        .sum(),
                },
                _ => 0,
            })
            .sum();
        instructions + history
    }
}

impl TenantCharge for EmbeddingRequest {
    fn estimated_input_tokens(&self, _routing_text: Option<&str>) -> u32 {
        input_value_tokens(&self.input)
    }
}

impl TenantCharge for ClassifyRequest {
    fn estimated_input_tokens(&self, _routing_text: Option<&str>) -> u32 {
        input_value_tokens(&self.input)
    }
}

/// The system instruction, which routing text leaves out.
impl TenantCharge for InteractionsRequest {
    fn unrouted_words(&self) -> usize {
        self.system_instruction.as_deref().map_or(0, word_count)
    }
}

// Routes that never reserve.
impl TenantCharge for CountMessageTokensRequest {}
impl TenantCharge for DecisionsRequest {}
impl TenantCharge for SystemOneRequest {}
impl TenantCharge for RerankRequest {}

#[cfg(test)]
impl TenantCharge for DropProbeRequest {}

/// The input of an embeddings or classify request in each shape it takes: a
/// string, an array of strings, or token ids, one sequence or several. Each
/// id counts as a token and each word of text as 1.3.
fn input_value_tokens(input: &Value) -> u32 {
    let (mut words, mut ids) = (0usize, 0usize);
    let mut count = |value: &Value| match value {
        Value::String(text) => words += word_count(text),
        Value::Number(_) => ids += 1,
        Value::Array(sequence) => ids += sequence.iter().filter(|id| id.is_number()).count(),
        _ => {}
    };
    match input {
        Value::Array(items) => items.iter().for_each(&mut count),
        single => count(single),
    }
    u32::try_from(ids)
        .unwrap_or(u32::MAX)
        .saturating_add(tokens_for_words(words))
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The words of a JSON value: of its strings and its keys, and one for each
/// number, boolean and null.
fn json_words(value: &Value) -> usize {
    match value {
        Value::String(text) => word_count(text),
        Value::Array(items) => items.iter().map(json_words).sum(),
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| word_count(key) + json_words(value))
            .sum(),
        Value::Null | Value::Bool(_) | Value::Number(_) => 1,
    }
}

/// 1.3 tokens per word, rounded up.
fn tokens_for_words(words: usize) -> u32 {
    (words as f64 * TOKENS_PER_WORD).ceil() as u32
}

/// One logical request's reservation: taken by [`reserve`] before any
/// dispatch, resolved by [`resolve`] on the answer the dispatch paths
/// converge on.
pub(crate) struct TenantReservation {
    cell: RateLimitCell,
    /// The request's own ([`TenantCharge::samples_per_prompt`]), for the
    /// resolve of a `/generate` answer of several entries.
    samples_per_prompt: u32,
}

/// Reserve `request`'s estimate for `tenant_meta` under `manager`, once per
/// logical request and before any dispatch: every attempt dispatches under
/// this one reservation. With a limiter and a tenant to charge, either the
/// reservation or the denial, which is the request's answer, never retried;
/// `(None, None)` for a request outside the limiter (no limiter configured,
/// or a route that charges no tenant). `routing_text` is the request's
/// routing text when routing already extracted it
/// ([`TenantCharge::estimated_input_tokens`]).
pub(crate) async fn reserve<T: TenantCharge>(
    manager: Option<&RateLimitManager>,
    tenant_meta: Option<&TenantRequestMeta>,
    request: &T,
    model_id: &str,
    routing_text: Option<&str>,
) -> (Option<TenantReservation>, Option<Response>) {
    let (Some(manager), Some(tenant_meta)) = (manager, tenant_meta) else {
        return (None, None);
    };
    let cell = RateLimitCell::new();
    let estimate = request.estimated_input_tokens(routing_text);
    match cell.reserve(manager, tenant_meta, model_id, estimate).await {
        Ok(()) => (
            Some(TenantReservation {
                cell,
                samples_per_prompt: request.samples_per_prompt(),
            }),
            None,
        ),
        Err(mut denied) => {
            mark_non_retryable(&mut denied);
            (None, Some(denied))
        }
    }
}

/// Resolve `reservation` on the answer of a request to `route`, once: a
/// buffered success settles the usage its body reports (or closes at the
/// estimate when the body reports none it can read), a streamed success
/// hands the reservation to its body, which keeps the estimate when it
/// drops, and any other answer closes it at the estimate. The answer is
/// returned as the client would have had it; an answer outside the limiter
/// (no reservation) passes through. `max_body_bytes` bounds the re-read of
/// a buffered answer: the bound the router read it under, or none for a
/// router that holds the body whole already.
pub(crate) async fn resolve(
    reservation: Option<&TenantReservation>,
    route: &str,
    is_stream: bool,
    response: Response,
    max_body_bytes: usize,
) -> Response {
    let Some(TenantReservation {
        cell,
        samples_per_prompt,
    }) = reservation
    else {
        return response;
    };
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
    match usage_from_body(route, *samples_per_prompt, &bytes) {
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
/// `meta_info` on `/generate` ([`generate_usage`]), Anthropic's `usage` on
/// `/v1/messages` ([`messages_input_tokens`]), the Interactions `usage`
/// totals on `/v1/interactions` ([`interactions_usage`]), and on every
/// other route a `usage` object in the OpenAI shape (`prompt_tokens` and
/// `completion_tokens`: chat, completions) or the Responses one
/// (`input_tokens` and `output_tokens`). Embeddings and classify report
/// `prompt_tokens` alone: they generate nothing, and a `completion_tokens`
/// they do report must read whole. `None` for a body that is not JSON or
/// reports none of these whole. Which of the input the engine served from
/// its cache is not recorded; which of the output it counted as reasoning
/// is recorded on `/v1/interactions` only.
fn usage_from_body(route: &str, samples_per_prompt: u32, body: &[u8]) -> Option<UsageSettlement> {
    let value: Value = serde_json::from_slice(body).ok()?;
    if route == "/generate" {
        return generate_usage(&value, samples_per_prompt);
    }
    let usage = value.get("usage")?;
    let (input, output) = match route {
        "/v1/embeddings" | "/v1/classify" => (
            as_u32(usage.get("prompt_tokens")?)?,
            count_or_zero(usage, "completion_tokens")?,
        ),
        "/v1/messages" => (
            messages_input_tokens(usage)?,
            as_u32(usage.get("output_tokens")?)?,
        ),
        "/v1/interactions" => return interactions_usage(usage),
        _ => token_pair(usage, "prompt_tokens", "completion_tokens")
            .or_else(|| token_pair(usage, "input_tokens", "output_tokens"))?,
    };
    Some(UsageSettlement {
        actual_input_tokens: input,
        completion_tokens: output,
        ..UsageSettlement::default()
    })
}

/// The usage of SGLang's native `/generate` answer: one object, or an array
/// of them for a batch or a request of `n` samples, an entry per sample of
/// each prompt, each with its own `meta_info.prompt_tokens` and
/// `meta_info.completion_tokens`. Every completion is a generation of its
/// own and sums; each prompt is prefilled once, but every one of its
/// `samples_per_prompt` entries repeats its count, so the prompts' sum is
/// the entries' sum over that. `None` for an answer whose entries do not
/// come `samples_per_prompt` to a prompt.
fn generate_usage(value: &Value, samples_per_prompt: u32) -> Option<UsageSettlement> {
    let items = match value {
        Value::Array(items) if !items.is_empty() => items.as_slice(),
        Value::Array(_) => return None,
        item => std::slice::from_ref(item),
    };
    let samples = u64::from(samples_per_prompt.max(1));
    if !(items.len() as u64).is_multiple_of(samples) {
        return None;
    }
    let (mut prompts, mut completions) = (0u64, 0u64);
    for item in items {
        let (prompt, completion) =
            token_pair(item.get("meta_info")?, "prompt_tokens", "completion_tokens")?;
        prompts += u64::from(prompt);
        completions += u64::from(completion);
    }
    Some(UsageSettlement {
        actual_input_tokens: u32::try_from(prompts.div_ceil(samples)).ok()?,
        completion_tokens: u32::try_from(completions).ok()?,
        ..UsageSettlement::default()
    })
}

/// The usage of an Interactions answer. The prompt is `total_input_tokens`
/// and the `total_tool_use_tokens` of its tool use, which count the
/// tool-use prompts (the tool results the model is fed back), input like
/// the rest. The generation is `total_output_tokens` and the
/// `total_thought_tokens` of its thinking, as the OpenAI `completion_tokens`
/// and the Responses `output_tokens` of the other routes include their
/// reasoning; the thought count is kept as the reasoning share
/// (`total_tokens` sums all four). A count the answer leaves out is zero,
/// as an answer that is all thought leaves `total_output_tokens` out; one
/// that is not a `u32`, or a `total_input_tokens` left out, makes the usage
/// unreadable.
fn interactions_usage(usage: &Value) -> Option<UsageSettlement> {
    let thought = count_or_zero(usage, "total_thought_tokens")?;
    Some(UsageSettlement {
        actual_input_tokens: as_u32(usage.get("total_input_tokens")?)?
            .checked_add(count_or_zero(usage, "total_tool_use_tokens")?)?,
        completion_tokens: count_or_zero(usage, "total_output_tokens")?.checked_add(thought)?,
        reasoning_tokens: thought,
        ..UsageSettlement::default()
    })
}

/// The whole prompt of an Anthropic Messages `usage`: its `input_tokens`
/// count only the input after the last cache breakpoint, and the prompt is
/// those, the `cache_read_input_tokens` read from the cache and the
/// `cache_creation_input_tokens` written to it. SGLang's Messages adapter
/// reports its cached prefix the same way, out of `input_tokens`. Unlike
/// Responses, whose `input_tokens` include their cached tokens.
fn messages_input_tokens(usage: &Value) -> Option<u32> {
    as_u32(usage.get("input_tokens")?)?
        .checked_add(count_or_zero(usage, "cache_read_input_tokens")?)?
        .checked_add(count_or_zero(usage, "cache_creation_input_tokens")?)
}

/// The two counts `object` reports under `input` and `output`, when both
/// are there as `u32`s.
fn token_pair(object: &Value, input: &str, output: &str) -> Option<(u32, u32)> {
    Some((as_u32(object.get(input)?)?, as_u32(object.get(output)?)?))
}

/// The count `object` reports under `key`, zero when it reports none (absent
/// or null); `None` for one that is not a `u32`.
fn count_or_zero(object: &Value, key: &str) -> Option<u32> {
    match object.get(key) {
        None | Some(Value::Null) => Some(0),
        Some(count) => as_u32(count),
    }
}

fn as_u32(count: &Value) -> Option<u32> {
    count.as_u64().and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
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
        usage_from_body(route, 1, body.to_string().as_bytes())
    }

    fn parse_generate(samples_per_prompt: u32, body: Value) -> Option<UsageSettlement> {
        usage_from_body("/generate", samples_per_prompt, body.to_string().as_bytes())
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
        let interactions = json!({"usage": {
            "total_input_tokens": 6, "total_output_tokens": 1, "total_tokens": 7,
        }});
        assert_eq!(parse("/v1/interactions", interactions), Some(usage(6, 1)));
        // A thinking model's thought tokens are generated too: charged with
        // the output and kept as the reasoning share; its tool-use prompts
        // are input. An answer that is all thought leaves the output count
        // out: zero, not unreadable.
        let thinking = json!({"usage": {
            "total_input_tokens": 6, "total_output_tokens": 1, "total_thought_tokens": 40,
            "total_tool_use_tokens": 3, "total_tokens": 50,
        }});
        assert_eq!(
            parse("/v1/interactions", thinking),
            Some(UsageSettlement {
                reasoning_tokens: 40,
                ..usage(9, 41)
            })
        );
        let all_thought = json!({"usage": {
            "total_input_tokens": 6, "total_thought_tokens": 40, "total_tokens": 46,
        }});
        assert_eq!(
            parse("/v1/interactions", all_thought),
            Some(UsageSettlement {
                reasoning_tokens: 40,
                ..usage(6, 40)
            })
        );
        assert_eq!(
            parse(
                "/generate",
                json!({"text": "x", "meta_info": {"prompt_tokens": 4, "completion_tokens": 9}})
            ),
            Some(usage(4, 9))
        );
    }

    /// The usage embeddings and classify report generates nothing: their
    /// `prompt_tokens` alone settles, where a generating route's would not.
    /// Anthropic's `input_tokens` leave out the cached prefix, read or
    /// written, which the prompt adds back; Responses' include theirs.
    #[test]
    fn a_messages_prompt_counts_its_cache_reads_and_writes() {
        let cached = json!({"usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
            "output_tokens": 7,
        }});
        assert_eq!(parse("/v1/messages", cached), Some(usage(1000, 7)));

        let read_only = json!({"usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": null,
            "output_tokens": 7,
        }});
        assert_eq!(parse("/v1/messages", read_only), Some(usage(950, 7)));

        let responses = json!({"usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 900},
            "output_tokens": 7,
            "total_tokens": 1007,
        }});
        assert_eq!(parse("/v1/responses", responses), Some(usage(1000, 7)));
    }

    #[test]
    fn embeddings_and_classify_settle_their_prompt_tokens_alone() {
        for usage_object in [
            json!({"prompt_tokens": 8, "total_tokens": 8}),
            json!({"prompt_tokens": 8, "completion_tokens": null, "total_tokens": 8}),
        ] {
            let body = json!({"usage": usage_object});
            for route in ["/v1/embeddings", "/v1/classify"] {
                assert_eq!(parse(route, body.clone()), Some(usage(8, 0)), "{route}");
            }
            assert_eq!(parse("/v1/chat/completions", body), None);
        }
    }

    /// Every entry of an `n=1` batch is a prompt of its own: the prompts sum
    /// with the completions.
    #[test]
    fn a_generate_batch_sums_its_prompts_and_its_completions() {
        let batch = json!([
            {"meta_info": {"prompt_tokens": 2, "completion_tokens": 9}},
            {"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}},
        ]);
        assert_eq!(parse_generate(1, batch), Some(usage(6, 10)));
    }

    /// The `n` samples of one prompt repeat its count: it is counted once,
    /// per prompt of a batch, and every sample's completion sums.
    #[test]
    fn the_samples_of_a_prompt_count_it_once() {
        let one_prompt = json!([
            {"meta_info": {"prompt_tokens": 4, "completion_tokens": 9}},
            {"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}},
        ]);
        assert_eq!(parse_generate(2, one_prompt), Some(usage(4, 10)));

        let two_prompts = json!([
            {"meta_info": {"prompt_tokens": 2, "completion_tokens": 1}},
            {"meta_info": {"prompt_tokens": 2, "completion_tokens": 2}},
            {"meta_info": {"prompt_tokens": 5, "completion_tokens": 3}},
            {"meta_info": {"prompt_tokens": 5, "completion_tokens": 4}},
        ]);
        assert_eq!(parse_generate(2, two_prompts), Some(usage(7, 10)));
    }

    /// An answer that does not hold `n` entries to each prompt is not one
    /// this request asked for.
    #[test]
    fn a_generate_answer_short_of_its_samples_is_none() {
        let entry = json!({"meta_info": {"prompt_tokens": 4, "completion_tokens": 1}});
        assert_eq!(parse_generate(2, entry.clone()), None);
        assert_eq!(
            parse_generate(2, json!([entry.clone(), entry.clone(), entry])),
            None
        );
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
                "/v1/messages",
                json!({"usage": {"input_tokens": 5, "cache_read_input_tokens": "9", "output_tokens": 2}}),
            ),
            (
                "/v1/messages",
                json!({"usage": {"input_tokens": u32::MAX, "cache_creation_input_tokens": 1, "output_tokens": 2}}),
            ),
            (
                "/v1/completions",
                json!({"usage": {"prompt_tokens": u64::from(u32::MAX) + 1, "completion_tokens": 0}}),
            ),
            ("/v1/embeddings", json!({"usage": {"total_tokens": 8}})),
            (
                "/v1/embeddings",
                json!({"usage": {"prompt_tokens": 8, "completion_tokens": "0"}}),
            ),
            ("/v1/classify", json!({"usage": {"prompt_tokens": -8}})),
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
            (
                "/generate",
                json!([
                    {"meta_info": {"prompt_tokens": u32::MAX, "completion_tokens": 1}},
                    {"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}},
                ]),
            ),
        ] {
            assert_eq!(parse(route, body.clone()), None, "{route}: {body}");
        }
        assert_eq!(
            usage_from_body("/v1/chat/completions", 1, b"not json"),
            None
        );
    }

    const TEN_WORDS: &str = "one two three four five six seven eight nine ten";

    fn request<T: serde::de::DeserializeOwned>(body: Value) -> T {
        serde_json::from_value(body).unwrap()
    }

    /// Text is estimated at 1.3 tokens per word, rounded up; a pre-tokenized
    /// request at its own ids, whatever text routing saw.
    #[test]
    fn the_estimate_is_the_requests_own_ids_or_its_words() {
        let text: GenerateRequest =
            request(json!({"model": "m", "text": "one two three four five six seven"}));
        assert_eq!(text.estimated_input_tokens(None), 10);

        let ids: GenerateRequest = request(json!({"model": "m", "input_ids": [1, 2, 3, 4, 5]}));
        assert_eq!(ids.estimated_input_tokens(None), 5);

        let chat: ChatCompletionRequest = request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "one two three four five six seven"}],
        }));
        assert_eq!(chat.estimated_input_tokens(Some("one two three")), 4);
        assert_eq!(chat.estimated_input_tokens(Some("")), 0);
    }

    /// Routing keys on the first sequence of a batch; the estimate counts
    /// them all.
    #[test]
    fn a_generate_batch_reserves_every_sequence() {
        let batch: GenerateRequest =
            request(json!({"model": "m", "input_ids": [[1, 2], [3, 4, 5, 6]]}));
        assert_eq!(batch.routing_tokens().map(<[i32]>::len), Some(2));
        assert_eq!(batch.estimated_input_tokens(None), 6);

        let leading_empty: GenerateRequest =
            request(json!({"model": "m", "input_ids": [[], [3, 4, 5]]}));
        assert_eq!(leading_empty.estimated_input_tokens(Some("3 4 5")), 3);
    }

    #[test]
    fn a_generate_request_has_its_n_samples_per_prompt() {
        for (sampling_params, samples) in [
            (json!(null), 1),
            (json!({}), 1),
            (json!({"n": 3}), 3),
            (json!({"n": 0}), 1),
        ] {
            let generate: GenerateRequest = request(
                json!({"model": "m", "text": "hi", "sampling_params": sampling_params.clone()}),
            );
            assert_eq!(generate.samples_per_prompt(), samples, "{sampling_params}");
        }
        let chat: ChatCompletionRequest = request(json!({
            "model": "m", "n": 3, "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(chat.samples_per_prompt(), 1);
    }

    /// Embeddings and classify take text or token ids, one input or several:
    /// each id is a token, each word 1.3.
    #[test]
    fn embeddings_and_classify_reserve_each_input_shape() {
        for (input, tokens) in [
            (json!("one two three four five six seven"), 10),
            (json!(["one two", "three"]), 4),
            (json!([1, 2, 3, 4]), 4),
            (json!([[1, 2], [3]]), 3),
            (json!([]), 0),
        ] {
            let embedding: EmbeddingRequest = request(json!({"model": "m", "input": input}));
            assert_eq!(embedding.estimated_input_tokens(None), tokens, "{input}");
            let classify: ClassifyRequest = request(json!({"model": "m", "input": input}));
            assert_eq!(classify.estimated_input_tokens(None), tokens, "{input}");
        }
    }

    /// The instructions and the function calls and their outputs count with
    /// the messages routing reads.
    #[test]
    fn a_responses_request_reserves_its_instructions_and_function_calls() {
        let instructed: ResponsesRequest = request(json!({
            "model": "m",
            "instructions": TEN_WORDS,
            "input": "hello world",
        }));
        assert_eq!(instructed.estimated_input_tokens(Some("hello world")), 16);

        let tool_turn: ResponsesRequest = request(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "weather in Paris"},
                {"type": "function_call", "call_id": "c1", "name": "weather",
                 "arguments": "{\"city\": \"Paris\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny and warm"},
            ],
        }));
        assert_eq!(tool_turn.extract_text_for_routing(), "weather in Paris");
        assert_eq!(tool_turn.estimated_input_tokens(None), 11);
    }

    #[test]
    fn an_interactions_request_reserves_its_system_instruction() {
        let instructed: InteractionsRequest = request(json!({
            "model": "m",
            "system_instruction": TEN_WORDS,
            "input": "hello world",
        }));
        assert_eq!(instructed.extract_text_for_routing(), "hello world");
        assert_eq!(instructed.estimated_input_tokens(None), 16);
    }

    /// A custom tool's input and its output, a string or text parts.
    #[test]
    fn a_responses_request_reserves_its_custom_tool_calls() {
        for output in [
            json!(TEN_WORDS),
            json!([{"type": "input_text", "text": TEN_WORDS}]),
        ] {
            let custom: ResponsesRequest = request(json!({
                "model": "m",
                "input": [
                    {"type": "message", "role": "user", "content": "hello world"},
                    {"type": "custom_tool_call", "call_id": "c1", "name": "grep", "input": TEN_WORDS},
                    {"type": "custom_tool_call_output", "call_id": "c1", "output": output},
                ],
            }));
            assert_eq!(custom.extract_text_for_routing(), "hello world");
            // 2 + 10 + 10 words: ceil(22 * 1.3).
            assert_eq!(custom.estimated_input_tokens(None), 29);
        }
    }

    /// The arguments of the assistant's tool calls, whatever text routing saw.
    #[test]
    fn a_chat_history_reserves_its_tool_call_arguments() {
        let chat: ChatCompletionRequest = request(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather in Paris"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"query\": \"one two three four five six seven eight nine\"}",
                    },
                }]},
            ],
        }));
        assert_eq!(chat.extract_text_for_routing(), "weather in Paris");
        // 3 words of text and 10 of arguments: ceil(13 * 1.3).
        assert_eq!(chat.estimated_input_tokens(None), 17);
        assert_eq!(chat.estimated_input_tokens(Some("weather in Paris")), 17);
    }

    /// A tool result's text, a string or text blocks, and a tool use's input.
    #[test]
    fn a_messages_history_reserves_its_tool_uses_and_results() {
        for content in [
            json!(TEN_WORDS),
            json!([{"type": "text", "text": TEN_WORDS}]),
        ] {
            let result: CreateMessageRequest = request(json!({
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hello world"},
                    {"type": "tool_result", "tool_use_id": "t1", "content": content},
                ]}],
            }));
            assert_eq!(result.extract_text_for_routing(), "hello world");
            // 2 + 10 words: ceil(12 * 1.3).
            assert_eq!(result.estimated_input_tokens(None), 16);
        }

        let tool_use: CreateMessageRequest = request(json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "weather in Paris"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "weather",
                     "input": {"city": "Paris", "days": 3}},
                ]},
            ],
        }));
        assert_eq!(tool_use.extract_text_for_routing(), "weather in Paris");
        // 3 words of text; two keys, a word and a number of input: ceil(7 * 1.3).
        assert_eq!(tool_use.estimated_input_tokens(None), 10);
    }

    /// Routing on the client's tokens hint extracts no text; the estimate
    /// reads the body's own, not the hint's length.
    #[test]
    fn a_routing_hint_does_not_set_the_estimate() {
        let chat: ChatCompletionRequest = request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "one two three four five six seven"}],
        }));
        assert_eq!(chat.estimated_input_tokens(None), 10);
    }
}
