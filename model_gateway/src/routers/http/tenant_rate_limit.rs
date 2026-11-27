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
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    classify::ClassifyRequest,
    common::{GenerationRequest, InputIds},
    completion::{CompletionPrompt, CompletionRequest},
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
use serde::Deserialize;
use serde_json::{value::RawValue, Value};

#[cfg(test)]
use crate::routers::common::request_lease::test_probe::DropProbeRequest;
use crate::{
    middleware::TenantRequestMeta,
    rate_limit::{RateLimitManager, ReservationAttachment, UsageSettlement},
    routers::{
        common::{
            rate_limit_cell::RateLimitCell, retry::mark_non_retryable, stream::GuardedStream,
        },
        error,
    },
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
        tokens_for_words(routed_words(self, routing_text) + self.unrouted_words())
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

/// What the assistant's turns in the history carry beside their content and
/// reasoning: a refusal, and the arguments of their tool calls and of a
/// deprecated function call, by the words of their JSON. A call's name is not
/// counted.
impl TenantCharge for ChatCompletionRequest {
    /// The routing text of a request with a part that is no text carries
    /// that part's marker, which is no word of the prompt: such a request
    /// is counted from its body.
    fn estimated_input_tokens(&self, routing_text: Option<&str>) -> u32 {
        let unmarked = routing_text.filter(|_| {
            self.messages.iter().all(|message| {
                let content = match message {
                    ChatMessage::System { content, .. }
                    | ChatMessage::User { content, .. }
                    | ChatMessage::Tool { content, .. }
                    | ChatMessage::Developer { content, .. }
                    | ChatMessage::Root { content, .. } => Some(content),
                    ChatMessage::Assistant { content, .. } => content.as_ref(),
                    ChatMessage::Function { .. } => None,
                };
                match content {
                    Some(MessageContent::Parts(parts)) => {
                        parts.iter().all(|part| part.text().is_some())
                    }
                    _ => true,
                }
            })
        });
        tokens_for_words(routed_words(self, unmarked) + self.unrouted_words())
    }

    fn unrouted_words(&self) -> usize {
        self.messages
            .iter()
            .map(|message| match message {
                ChatMessage::Assistant {
                    refusal,
                    tool_calls,
                    function_call,
                    ..
                } => {
                    let arguments: usize = tool_calls
                        .iter()
                        .flatten()
                        .map(|call| &call.function)
                        .chain(function_call.as_deref())
                        .filter_map(|function| function.arguments.as_deref())
                        .map(argument_words)
                        .sum();
                    arguments + refusal.as_deref().map_or(0, word_count)
                }
                _ => 0,
            })
            .sum()
    }
}

/// A prompt of ids at its own count, every list of a batch (the `/generate`
/// rule); the words of its text otherwise. The routing text of ids renders
/// each id as a word, which the word estimate would charge at 1.3.
impl TenantCharge for CompletionRequest {
    fn estimated_input_tokens(&self, routing_text: Option<&str>) -> u32 {
        let ids = match &self.prompt {
            CompletionPrompt::TokenIds(ids) => ids.len(),
            CompletionPrompt::TokenIdBatches(batches) => batches.iter().map(Vec::len).sum(),
            CompletionPrompt::Text(_) | CompletionPrompt::Texts(_) => 0,
        };
        if ids > 0 {
            return u32::try_from(ids).unwrap_or(u32::MAX);
        }
        tokens_for_words(routed_words(self, routing_text) + self.unrouted_words())
    }
}

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
/// with their outputs, a function call's arguments by the words of their
/// JSON. Images, files and the built-in tools' items are not counted.
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
                } => argument_words(arguments) + output.as_deref().map_or(0, word_count),
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

/// The words of the routing text, or of the body's own text when routing
/// keyed on tokens.
fn routed_words(request: &(impl GenerationRequest + ?Sized), routing_text: Option<&str>) -> usize {
    match routing_text {
        Some(text) => word_count(text),
        None => word_count(&request.extract_text_for_routing()),
    }
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

/// The words of a call's arguments: of their JSON when they parse, as a tool
/// use's input counts, since JSON written without spaces (as clients send it,
/// and as an object-form `arguments` is kept) is one word of text. Arguments
/// that do not parse, as a call cut off mid-way leaves them, count the runs
/// between spaces and JSON punctuation, so a cut-off compact object still
/// counts its keys and values.
fn argument_words(arguments: &str) -> usize {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => json_words(&value),
        Err(_) => arguments
            .split(|c: char| c.is_whitespace() || "{}[]:,\"".contains(c))
            .filter(|word| !word.is_empty())
            .count(),
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

/// What a tenant reservation charges for one request, with the limiter and
/// the tenant it was gated on: the reserve-time estimate of the request's
/// input tokens and the answers each prompt gets, read from the request
/// before any dispatch ([`estimate`]), so the reservation itself borrows
/// nothing of the request across its await and cannot be asked for with a
/// limiter, a tenant and an estimate that were not gated together.
#[derive(Clone, Copy)]
pub(crate) struct TenantEstimate<'a> {
    manager: &'a RateLimitManager,
    tenant_meta: &'a TenantRequestMeta,
    estimated_input_tokens: u32,
    samples_per_prompt: u32,
}

/// The charge of `request` under the tenant rate limiter: `None` for a
/// request outside it (no limiter configured, or a route that charges no
/// tenant), so the estimate is read only when a reservation will use it.
/// `routing_text` is the request's routing text when routing already
/// extracted it ([`TenantCharge::estimated_input_tokens`]).
pub(crate) fn estimate<'a, T: TenantCharge>(
    manager: Option<&'a RateLimitManager>,
    tenant_meta: Option<&'a TenantRequestMeta>,
    request: &T,
    routing_text: Option<&str>,
) -> Option<TenantEstimate<'a>> {
    Some(TenantEstimate {
        manager: manager?,
        tenant_meta: tenant_meta?,
        estimated_input_tokens: request.estimated_input_tokens(routing_text),
        samples_per_prompt: request.samples_per_prompt(),
    })
}

/// Reserve `estimate` once per logical request, before any dispatch: every
/// attempt dispatches under this one reservation. Either the reservation or
/// the denial, which is the request's answer, never retried; `(None, None)`
/// for a request outside the limiter (no estimate: no limiter configured, or
/// a route that charges no tenant).
pub(crate) async fn reserve(
    estimate: Option<TenantEstimate<'_>>,
    model_id: &str,
) -> (Option<TenantReservation>, Option<Response>) {
    let Some(estimate) = estimate else {
        return (None, None);
    };
    let cell = RateLimitCell::new();
    match cell
        .reserve(
            estimate.manager,
            estimate.tenant_meta,
            model_id,
            estimate.estimated_input_tokens,
        )
        .await
    {
        Ok(()) => (
            Some(TenantReservation {
                cell,
                samples_per_prompt: estimate.samples_per_prompt,
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
    if route == "/generate" {
        let value: Value = serde_json::from_slice(body).ok()?;
        return generate_usage(&value, samples_per_prompt);
    }
    // The answer's `usage` object alone is read into a value; the rest of
    // the document (the choices, the text) is scanned past, not built.
    #[derive(Deserialize)]
    struct UsageOnly<'a> {
        #[serde(borrow)]
        usage: Option<&'a RawValue>,
    }
    let usage: Value = serde_json::from_str(
        serde_json::from_slice::<UsageOnly<'_>>(body)
            .ok()?
            .usage?
            .get(),
    )
    .ok()?;
    let usage = &usage;
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
#[path = "../../../tests/unit/routers/http/tenant_rate_limit/tests.rs"]
mod tests;
