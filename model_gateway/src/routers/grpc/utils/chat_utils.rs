//! Chat message processing, tool constraints, and shared utilities for gRPC routers.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, OnceLock},
    time::Duration,
};

use anyhow::anyhow;
use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};
use bytes::Bytes;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateParams, PreparedTools, ThinkingKeyName,
    },
    traits::{Encoding, PromptEncoding, Tokenizer},
    StopRules, StopSequenceConfig, StopSequenceDecoder,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{
        ContentPart, FunctionCallResponse, StringOrArray, Tool, ToolCall, ToolChoice,
        ToolChoiceValue,
    },
    generate::GenerateFinishReason,
    profile::ProviderProfile,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::{sync::Semaphore, task::yield_now};
use tool_parser::types::ToolCallItem;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::prepared_tools::prepare_tools;
use crate::{
    middleware::ReportedRefusal,
    observability::log_throttle::KeyedLogThrottle,
    routers::{
        common::sse,
        error,
        grpc::{
            context::RequestContext,
            multimodal::{anchor_rendered_markers, PlaceholderTokens},
            ProcessedMessages,
        },
    },
    workflow::TokenizerLoadFailure,
};

/// Type alias for the SSE channel sender used across streaming endpoints.
///
/// Re-exports the bounded [`crate::routers::common::sse::SseSender`] so the gRPC
/// streaming paths share one definition with the rest of the gateway and cannot
/// drift back to an unbounded channel. Bounded => `send` is async and applies
/// backpressure to a slow client.
pub(crate) type SseSender = sse::SseSender;

/// The body of an SSE error event, `{"error":{"message":...,"type":...}}`
/// (with a `code` when the failure has one of its own), serialized straight
/// into its frame: the message is written through `Display` into the JSON
/// escaper, so the frame is the one allocation.
#[derive(Serialize)]
#[serde(bound = "M: fmt::Display")]
struct SseErrorFrame<'a, M: fmt::Display> {
    error: SseErrorBody<'a, M>,
}

#[derive(Serialize)]
#[serde(bound = "M: fmt::Display")]
struct SseErrorBody<'a, M: fmt::Display> {
    #[serde(serialize_with = "serialize_display")]
    message: M,
    #[serde(rename = "type")]
    error_type: &'a str,
    /// The code the HTTP answer to the same failure carries, for a failure
    /// that has one (`constraint_failed`, `reader_panicked`); absent
    /// otherwise, so the frames clients have been reading do not change.
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'a str>,
}

fn serialize_display<M: fmt::Display, S: serde::Serializer>(
    message: &M,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(message)
}

/// Send an SSE error event with a typed error body:
/// `data: {"error":{"message":"...","type":"..."}}\n\n`, the message
/// JSON-escaped (quotes, newlines, control characters).
///
/// `async` because the sender is bounded: awaits if the buffer is full so the
/// terminal error frame still applies backpressure rather than being dropped.
pub(crate) async fn send_error_sse(tx: &SseSender, message: impl fmt::Display, error_type: &str) {
    send_error_sse_with_code(tx, message, error_type, None).await;
}

/// [`send_error_sse`] with the failure's own `code` in the body when it has
/// one: `data: {"error":{"message":"...","type":"...","code":"..."}}\n\n`.
pub(crate) async fn send_error_sse_with_code(
    tx: &SseSender,
    message: impl fmt::Display,
    error_type: &str,
    code: Option<&str>,
) {
    const PREFIX: &[u8] = b"data: ";
    let mut frame = Vec::with_capacity(256);
    frame.extend_from_slice(PREFIX);
    let body = SseErrorFrame {
        error: SseErrorBody {
            message,
            error_type,
            code,
        },
    };
    // Unreachable in practice: a `Vec` writer never fails, and serde_json
    // panics (as `to_string` did) on a `Display` that errors rather than
    // returning the error; the fallback keeps the frame well-formed should
    // either ever change.
    if serde_json::to_writer(&mut frame, &body).is_err() {
        frame.truncate(PREFIX.len());
        frame.extend_from_slice(
            br#"{"error":{"message":"the error message could not be serialized","type":"internal_error"}}"#,
        );
    }
    frame.extend_from_slice(b"\n\n");
    let _ = tx.send(Ok(Bytes::from(frame))).await;
}

/// Resolve tokenizer from registry and cache it in request context.
///
/// This is a helper to avoid duplicating tokenizer resolution logic across
/// preparation stages (chat, generate, embedding).
///
/// Returns the tokenizer Arc, which is also cached in `ctx.state.tokenizer`.
pub(crate) fn resolve_tokenizer(
    ctx: &mut RequestContext,
    stage_name: &str,
) -> Result<Arc<dyn Tokenizer>, Box<Response>> {
    let model_id = ctx.input.model_id.as_str();

    let tokenizer = ctx
        .components
        .tokenizer_registry
        .get(model_id)
        .ok_or_else(|| {
            let served = ctx.components.worker_registry.contains_model(model_id);
            let failure = ctx.components.tokenizer_load_failures.get(model_id);
            Box::new(missing_tokenizer_response(
                stage_name,
                model_id,
                served,
                ctx.components.tokenizer_autoload,
                failure.as_ref(),
            ))
        })?;

    // Cache tokenizer in context for reuse in response processing stage
    ctx.state.tokenizer = Some(tokenizer.clone());

    Ok(tokenizer)
}

/// Lines per second the tokenizer-window WARN admits per model, and the
/// models the throttle tracks: the allowance of the worker-selection
/// refusal line, for the same reason (a client retrying at once through the
/// window is one condition, not a line per request).
const TOKENIZER_NOT_READY_LINES_PER_SECOND: u32 = 10;
const TOKENIZER_NOT_READY_LOG_MODELS: usize = 64;

static TOKENIZER_NOT_READY_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    TOKENIZER_NOT_READY_LINES_PER_SECOND,
    Duration::from_secs(1),
    TOKENIZER_NOT_READY_LOG_MODELS,
);

/// The client-visible code of [`tokenizer_not_ready`].
pub(crate) const TOKENIZER_NOT_READY_CODE: &str = "tokenizer_not_ready";

/// Seconds a client is asked to wait before retrying through the window: a
/// tokenizer registration takes a fraction of a second on a warm cache.
const TOKENIZER_NOT_READY_RETRY_AFTER_SECS: u32 = 1;

/// The 503 for a served model whose tokenizer the gateway has not registered
/// yet: the window between a worker turning healthy and the end of its
/// tokenizer load, which `/readiness` reports as not ready ("tokenizer not
/// yet registered"). A planned, transient state, so the answer is the
/// retryable class with a `Retry-After`, like a pool with no available
/// worker, not a 500. One WARN per request, capped per model at
/// [`TOKENIZER_NOT_READY_LINES_PER_SECOND`] a second (the overflow is counted
/// as `suppressed` on the next admitted line); the response is marked
/// [`ReportedRefusal`], so the pipeline's and the access log's lines for it
/// stay at DEBUG (three ERROR lines per request were the storm of the 500).
/// A served model whose tokenizer load was skipped or failed for good is
/// answered [`tokenizer_load_failed`] instead.
pub(crate) fn tokenizer_not_ready(function: &str, model_id: &str) -> Response {
    let mut response = error::service_unavailable(
        TOKENIZER_NOT_READY_CODE,
        format!("Tokenizer for model '{model_id}' is not registered yet; retry shortly"),
    );
    response.headers_mut().insert(
        RETRY_AFTER,
        HeaderValue::from(TOKENIZER_NOT_READY_RETRY_AFTER_SECS),
    );
    response.extensions_mut().insert(ReportedRefusal);
    if let Some(admitted) = TOKENIZER_NOT_READY_LOG.admit(model_id) {
        warn!(
            function = %function,
            model = %model_id,
            suppressed = admitted.suppressed,
            "Tokenizer not yet registered for a served model"
        );
    }
    response
}

/// The client-visible code of [`tokenizer_load_failed`].
pub(crate) const TOKENIZER_LOAD_FAILED_CODE: &str = "tokenizer_load_failed";

/// The 503 for a served model whose tokenizer the gateway will not register
/// on its own: its load was skipped (no path to load it from) or failed for
/// good, or its tokenizer was removed, as the registration recorded for the
/// model (a load in flight is pending, not this). The envelope
/// of [`tokenizer_not_ready`] without the `Retry-After`: a retry changes
/// nothing until a tokenizer registers for the model (a worker re-registering
/// with a usable path, an operator's `POST /v1/tokenizers`), which clears
/// the record. The message names the reason class, not the path or the
/// error: the gateway's log has those, in the one line written when the
/// model entered the state, not per request. Marked [`ReportedRefusal`], so
/// the pipeline's and the access log's lines for it stay at DEBUG.
pub(crate) fn tokenizer_load_failed(
    function: &str,
    model_id: &str,
    failure: &TokenizerLoadFailure,
) -> Response {
    debug!(
        function = %function,
        model = %model_id,
        reason = failure.reason(),
        "Tokenizer load failed for a served model"
    );
    let mut response = error::service_unavailable(
        TOKENIZER_LOAD_FAILED_CODE,
        format!(
            "Tokenizer for model '{model_id}' could not be loaded: {}",
            failure.reason()
        ),
    );
    response.extensions_mut().insert(ReportedRefusal);
    response
}

/// The error for a model without a registered tokenizer: a model nobody
/// serves is the client's mistake (404); a served model whose tokenizer load
/// was skipped or failed for good, or whose tokenizer was removed, gets none
/// on its own (503 without a retry hint, [`tokenizer_load_failed`]); a
/// served model whose tokenizer the
/// gateway is still loading is not ready yet (503 with a `Retry-After`, the
/// state `/readiness` reports); a served model with no tokenizer when the
/// gateway does not load tokenizers itself (`--disable-tokenizer-autoload`)
/// is a gateway fault (500): nothing will register one.
pub(crate) fn missing_tokenizer_response(
    function: &str,
    model_id: &str,
    served: bool,
    tokenizer_autoload: bool,
    failure: Option<&TokenizerLoadFailure>,
) -> Response {
    if !served {
        debug!(function = %function, model = %model_id, "Unknown model");
        return error::model_not_found(model_id);
    }
    if let Some(failure) = failure {
        return tokenizer_load_failed(function, model_id, failure);
    }
    if tokenizer_autoload {
        return tokenizer_not_ready(function, model_id);
    }
    error!(
        function = %function,
        model = %model_id,
        "Tokenizer not found for model"
    );
    error::internal_error(
        "tokenizer_not_found",
        format!("Tokenizer not found for model: {model_id}"),
    )
}

/// Below this input size (in bytes) a prompt is tokenized on the runtime
/// thread that holds the request; from this size up the encode runs on the
/// blocking pool, bounded by [`encode_permits`], so no worker is held for
/// longer than about one request's own share of work: a single prompt by its
/// own size, a batch (a completion with several prompts, as text or as ids, a
/// tokenize call, a Decisions request's prompts and label probes) by what the
/// call has already encoded or decoded in place, through [`InPlaceBudget`]. A
/// prompt of ids counts its ids' bytes, four each. An input under the
/// break-even size ([`OFFLOAD_BREAK_EVEN_BYTES`]) never takes the pool: once
/// the call's share is spent it yields the worker instead.
///
/// The bound is where the encode outweighs its offload. An offloaded encode
/// costs the request a round trip through the pool: the pool thread's wake,
/// the request task re-scheduled through the inject queue when the encode
/// ends, and the parked worker those wakes notify. On the gateway's CPU bed
/// (the repository's `gateway-perf` cells: eight runtime threads, 500 chat
/// requests per second against scripted engines) the round trip costs the
/// same whatever the prompt's length: about 55 k instructions, 3.7 context
/// switches and, on a virtual machine, 0.25 ms of CPU time per request more
/// than the same encode in place, and the first token leaves 0.2 ms later.
/// The encode itself costs about 70 k instructions and 10-20 us per
/// kilobyte of prompt with a byte-level BPE tokenizer: from one kilobyte up
/// the work outweighs the trip in instructions, and at 16 KiB (about 3,000
/// tokens, 0.2-0.3 ms) it still holds a worker for no longer than one short
/// request's own CPU time (0.3 ms). Above that the pool takes it. The
/// earlier bound of 512 bytes charged the round trip to every ordinary
/// conversation: a system prompt and a few turns render to 1-10 KiB.
const ENCODE_OFFLOAD_MIN_BYTES: usize = 16 * 1024;

/// Below this input size the pool's round trip costs more than the work it
/// would move (the measurement above: a trip is about 55 k instructions, an
/// encode about 70 k per kilobyte), so such an input never takes the pool:
/// when the call's in-place share is spent it yields the worker once instead
/// and starts the next share.
const OFFLOAD_BREAK_EVEN_BYTES: usize = 1024;

/// Bounds how many CPU-bound encodes run concurrently on the blocking pool.
/// tokio's blocking pool is otherwise unbounded (grows to 512 threads), so under
/// a burst of large prompts the offloaded encodes would oversubscribe the CPU
/// and starve the very request runtime this offload is meant to protect. Sized
/// to the host's available parallelism.
fn encode_permits() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        Semaphore::new(n)
    })
}

/// Whether a prompt of `len` bytes is encoded on the blocking pool: at or above
/// [`ENCODE_OFFLOAD_MIN_BYTES`]; a shorter one runs inline, so the offload's
/// round trip does not dominate it. The one rule [`offload`] and
/// [`encode_blocking`] read.
fn offloads(len: usize) -> bool {
    len >= ENCODE_OFFLOAD_MIN_BYTES
}

/// Run CPU-bound tokenizer work, an encode or a decode, off the async worker
/// threads so it cannot stall the runtime, bounded by [`encode_permits`] so
/// concurrent offloaded jobs cannot oversubscribe the CPU. Inputs below the
/// threshold ([`offloads`]) run inline to avoid the offload round-trip
/// dominating.
async fn offload<T, F>(len: usize, work: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    if !offloads(len) {
        return work();
    }
    pool(work).await
}

/// Run `work` on the blocking pool behind the [`encode_permits`].
async fn pool<T, F>(work: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    pool_on(encode_permits(), work).await
}

/// Run `work` on the blocking pool behind one of `permits`. The permit rides
/// in the blocking closure, held until the work ends whatever happens to this
/// future: a pass ended by another input's error, or dropped by a client
/// gone, still counts its running jobs against the bound.
async fn pool_on<T, F>(permits: &'static Semaphore, work: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    let permit = permits
        .acquire()
        .await
        .map_err(|e| anyhow!("encode semaphore closed: {e}"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
    .map_err(|e| anyhow!("tokenizer task failed: {e}"))?
}

/// Tokenize `text` through the offload policy of [`offload`]: a prompt below
/// the threshold ([`offloads`]) encodes in place, from the caller's own text;
/// one at or above it is copied once, for the blocking task, which outlives
/// the borrow.
pub(crate) async fn encode_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    text: &str,
    add_special_tokens: bool,
) -> anyhow::Result<Encoding> {
    if !offloads(text.len()) {
        return tokenizer.encode(text, add_special_tokens);
    }
    let text = text.to_string();
    pool(move || tokenizer.encode(&text, add_special_tokens)).await
}

/// What one call has encoded or decoded on the runtime thread since it last
/// yielded it: the call's share. A call places its inputs one after another
/// against it, so it holds a worker for about one input at the bound before
/// it lets go: an input that would take the share over the bound goes to the
/// pool if it is worth the trip ([`OFFLOAD_BREAK_EVEN_BYTES`]), and a smaller
/// one has the call yield the worker once and start the next share.
#[derive(Default)]
pub(crate) struct InPlaceBudget {
    in_place: usize,
}

/// Where an input of a call runs, by [`InPlaceBudget::place`].
enum Placement {
    /// On the calling thread, within the share.
    InPlace,
    /// On the blocking pool.
    Pool,
    /// On the calling thread, once the call has yielded it: the share is spent.
    AfterYield,
}

impl InPlaceBudget {
    /// Where an input of `len` bytes runs, the share charged when in place.
    fn place(&mut self, len: usize) -> Placement {
        if offloads(len) {
            return Placement::Pool;
        }
        let total = self.in_place + len;
        if !offloads(total) {
            self.in_place = total;
            Placement::InPlace
        } else if len >= OFFLOAD_BREAK_EVEN_BYTES {
            Placement::Pool
        } else {
            Placement::AfterYield
        }
    }

    /// The next share, opened by an input of `len` bytes after the yield.
    fn next_share(&mut self, len: usize) {
        self.in_place = len;
    }
}

/// Tokenize one `text` of a call's sequence through the offload policy, the
/// call's [`InPlaceBudget`] placing it: in place from the caller's own text,
/// as [`encode_blocking`] does below the bound; on the pool from one copy; or
/// in place once the call has yielded the worker. A call that holds several
/// inputs at once places them in one pass with [`encode_texts_blocking`].
pub(crate) async fn encode_blocking_in_batch(
    tokenizer: Arc<dyn Tokenizer>,
    text: &str,
    add_special_tokens: bool,
    budget: &mut InPlaceBudget,
) -> anyhow::Result<Encoding> {
    match budget.place(text.len()) {
        Placement::InPlace => tokenizer.encode(text, add_special_tokens),
        Placement::AfterYield => {
            yield_now().await;
            budget.next_share(text.len());
            tokenizer.encode(text, add_special_tokens)
        }
        Placement::Pool => {
            let text = text.to_string();
            pool(move || tokenizer.encode(&text, add_special_tokens)).await
        }
    }
}

/// Tokenize a call's `texts` as one pass over them: each is placed once by
/// the call's [`InPlaceBudget`]; the texts in place are encoded one after
/// another on the calling thread, the call yielding the worker once per spent
/// share, and the texts worth the trip run on the pool beside them, started
/// with the pass's first poll and joined at its end. The encodings in the
/// texts' order.
pub(crate) async fn encode_texts_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    texts: &[String],
    add_special_tokens: bool,
) -> anyhow::Result<Vec<Encoding>> {
    place_all(
        texts.iter(),
        |text| text.len(),
        |text| tokenizer.encode(text.as_str(), add_special_tokens),
        |text| {
            let (tokenizer, text) = (tokenizer.clone(), text.to_string());
            Box::new(move || tokenizer.encode(&text, add_special_tokens))
        },
    )
    .await
}

/// Decode a call's prompts given as ids, `lists`, as one pass over them, as
/// [`encode_texts_blocking`] does for texts: a prompt of ids leaves the worker
/// thread at the same size a text prompt does, four bytes an id. The texts in
/// the lists' order.
pub(crate) async fn decode_ids_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    lists: Vec<Vec<u32>>,
    skip_special_tokens: bool,
) -> anyhow::Result<Vec<String>> {
    place_all(
        lists,
        |ids| ids.len() * size_of::<u32>(),
        |ids| tokenizer.decode(ids, skip_special_tokens),
        |ids| {
            let tokenizer = tokenizer.clone();
            Box::new(move || tokenizer.decode(&ids, skip_special_tokens))
        },
    )
    .await
}

/// The pool's job for an input a call does not run in place.
type PoolJob<T> = Box<dyn FnOnce() -> anyhow::Result<T> + Send + 'static>;

/// One pass of a call over its `inputs` through the offload policy: `len` is
/// an input's bytes for the budget, `in_place` runs one on the calling
/// thread, `job` makes the pool's job of one. The pass is planned first, every
/// input placed and the pool's jobs made; then the pool's join and the
/// in-place pass run together, the join polled first, so the pool's jobs
/// start on the pass's first poll and the in-place inputs run one after
/// another, the call yielding the worker once per spent share. The outputs in
/// the inputs' order; the first error ends the pass, and a pool job already
/// running finishes on its thread, its permit held until then ([`pool_on`]).
async fn place_all<I, T>(
    inputs: impl IntoIterator<Item = I>,
    len: impl Fn(&I) -> usize,
    in_place: impl Fn(&I) -> anyhow::Result<T>,
    job: impl Fn(I) -> PoolJob<T>,
) -> anyhow::Result<Vec<T>>
where
    T: Send + 'static,
{
    let mut budget = InPlaceBudget::default();
    let mut outputs: Vec<Option<T>> = Vec::new();
    // The in-place inputs in order, each with whether the call yields before it.
    let mut placed = Vec::new();
    let mut pooled = Vec::new();
    for (index, input) in inputs.into_iter().enumerate() {
        let len = len(&input);
        match budget.place(len) {
            Placement::InPlace => placed.push((index, input, false)),
            Placement::AfterYield => {
                budget.next_share(len);
                placed.push((index, input, true));
            }
            Placement::Pool => {
                let job = job(input);
                pooled.push(async move { pool(job).await.map(|output| (index, output)) });
            }
        }
        outputs.push(None);
    }
    let in_place_pass = async {
        let mut done = Vec::with_capacity(placed.len());
        for (index, input, yields_first) in placed {
            if yields_first {
                yield_now().await;
            }
            done.push((index, in_place(&input)?));
        }
        anyhow::Ok(done)
    };
    let (from_pool, from_here) =
        futures::future::try_join(futures::future::try_join_all(pooled), in_place_pass).await?;
    for (index, output) in from_pool.into_iter().chain(from_here) {
        outputs[index] = Some(output);
    }
    outputs
        .into_iter()
        .map(|output| output.ok_or_else(|| anyhow!("an input of the pass was left unplaced")))
        .collect()
}

/// Tokenize a rendered chat prompt the way its renderer said to: a deferred
/// encode runs as the job the renderer prepared, anything else encodes `text`.
/// Both take the same offload policy as [`encode_blocking`].
pub(crate) async fn encode_prompt_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    text: &str,
    encoding: PromptEncoding,
) -> anyhow::Result<Encoding> {
    match encoding {
        PromptEncoding::FromText => encode_blocking(tokenizer, text, false).await,
        PromptEncoding::Deferred(job) => offload(text.len(), move || job.run()).await,
    }
}

/// Hand the template the tool-call arguments of the assistant turns in the
/// shape the engines hand theirs: an object (what the Transformers docs ask
/// of a tool-use template). The arguments' JSON text is parsed; an object is
/// kept; absent arguments are an empty object.
pub(crate) fn process_tool_call_arguments(messages: &mut [Value]) {
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in tool_calls {
            let Some(function) = call.get_mut("function").and_then(Value::as_object_mut) else {
                continue;
            };
            let arguments = match function.get("arguments") {
                Some(Value::Object(_)) => continue,
                Some(Value::String(text)) => {
                    arguments_object(function.get("name").and_then(Value::as_str), text)
                }
                _ => json!({}),
            };
            function.insert("arguments".to_string(), arguments);
        }
    }
}

/// The object an `arguments` text holds, read as the engines read it: empty
/// text is a call without arguments; text that is not JSON, or JSON that is
/// not an object, is an empty object too, with a warning. A malformed
/// `arguments` text lives in the conversation's history, so refusing it
/// would fail every later turn of that conversation.
fn arguments_object(tool: Option<&str>, text: &str) -> Value {
    if text.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(text) {
        Ok(object @ Value::Object(_)) => object,
        Ok(_) => {
            tracing::warn!(
                tool,
                bytes = text.len(),
                "tool call arguments in history are JSON but not an object; rendered as an \
                 empty object"
            );
            json!({})
        }
        Err(error) => {
            tracing::warn!(
                tool,
                bytes = text.len(),
                %error,
                "tool call arguments in history are not valid JSON; rendered as an empty object"
            );
            json!({})
        }
    }
}

/// Process messages based on content format for ANY message type
#[cfg(test)]
pub(crate) fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
) -> Result<Vec<Value>, String> {
    process_content_format_with_order(
        messages,
        content_format,
        placeholder_tokens,
        MediaPartOrder::MediaFirst,
    )
}

pub(super) const REASONING_EFFORT_KEY: &str = "reasoning_effort";
const TOOL_CHOICE_KEY: &str = "tool_choice";
const RESPONSE_FORMAT_KEY: &str = "response_format";

/// The template kwarg both prompt paths tell the tool choice through: the
/// chat value as sent, so a renderer reads one shape under
/// `/v1/chat/completions`, `/v1/messages` and `count_tokens`.
pub(crate) fn insert_tool_choice_kwarg(
    kwargs: &mut HashMap<String, Value>,
    tool_choice: &ToolChoice,
) {
    if let Ok(value) = serde_json::to_value(tool_choice) {
        kwargs.insert(TOOL_CHOICE_KEY.to_string(), value);
    }
}

/// Merge request-level reasoning, tool, and output-format controls with any
/// request `chat_template_kwargs`, forwarding each verbatim. The chat template
/// owns interpretation (level→value mapping, `tool_choice`/`response_format`
/// rendering, defaulting, and validation); an explicit `chat_template_kwargs`
/// entry wins, except for `reasoning_effort`: the request-level effort
/// (`thinking.effort`, else the top-level field) owns the key, as the
/// engine's own server renders it, written over an entry as the request's
/// provider profile hands it to the template
/// (`ProviderProfile::template_reasoning_effort`, given the thinking
/// preference the prompt renders with: the template's own `thinking_mode`
/// switch among the kwargs where the template reads that key, `thinking_key`
/// being the tokenizer's detected thinking key as the backed plan gates its
/// mode kwargs, else the typed toggle or the profile's reading of the effort),
/// and the entry removed when the profile hands the template none.
fn build_chat_template_kwargs(
    request: &ChatCompletionRequest,
    provider: ProviderProfile,
    thinking_key: Option<ThinkingKeyName>,
) -> HashMap<String, Value> {
    let request_kwargs = request
        .chat_template_kwargs
        .as_ref()
        .map_or(0, HashMap::len);
    let explicit = matches!(thinking_key, Some(ThinkingKeyName::ThinkingMode))
        .then(|| provider.thinking_from_template_kwargs(request.chat_template_kwargs.as_ref()))
        .flatten();
    let reasoning_effort = provider.template_reasoning_effort(
        request.effective_reasoning_effort(),
        explicit.or_else(|| request.template_thinking_for(provider)),
    );
    // A plain request (no tool choice, no response format, no kwargs, no
    // effort) renders with no kwargs at all: its map stays unallocated.
    let mut combined = if request.tool_choice.is_some()
        || request.response_format.is_some()
        || request_kwargs > 0
        || reasoning_effort.is_some()
    {
        HashMap::with_capacity(3 + request_kwargs)
    } else {
        HashMap::new()
    };
    if let Some(tool_choice) = &request.tool_choice {
        insert_tool_choice_kwarg(&mut combined, tool_choice);
    }
    if let Some(response_format) = &request.response_format {
        if let Ok(value) = serde_json::to_value(response_format) {
            combined.insert(RESPONSE_FORMAT_KEY.to_string(), value);
        }
    }
    if let Some(template_kwargs) = &request.chat_template_kwargs {
        combined.extend(template_kwargs.clone());
    }
    if request.effective_reasoning_effort().is_some() {
        match reasoning_effort {
            Some(reasoning_effort) => {
                combined.insert(
                    REASONING_EFFORT_KEY.to_string(),
                    Value::String(reasoning_effort.to_string()),
                );
            }
            None => {
                combined.remove(REASONING_EFFORT_KEY);
            }
        }
    }
    combined
}

/// The thinking preference the prompt is rendered with: the typed toggle,
/// else the effective effort as the request's provider profile reads it
/// (`ChatCompletionRequest::template_thinking_for`, with the profile the
/// preparation stage derived once for the request).
fn resolve_template_thinking(
    request: &ChatCompletionRequest,
    provider: ProviderProfile,
) -> Option<bool> {
    request.template_thinking_for(provider)
}

/// gRPC backends require content parts the gateway knows how to render, and
/// a refusal part only where the contract defines it: on an assistant turn.
pub(crate) fn validate_chat_content_parts(messages: &[ChatMessage]) -> Result<(), String> {
    for message in messages {
        let content = match message {
            ChatMessage::System { content, .. }
            | ChatMessage::Root { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_ref(),
            ChatMessage::Function { .. } => None,
        };
        let assistant = matches!(message, ChatMessage::Assistant { .. });
        if let Some(MessageContent::Parts(parts)) = content {
            for part in parts {
                match part {
                    ContentPart::Unknown(fields) => {
                        let type_name = fields
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("<missing>");
                        return Err(format!(
                            "Unsupported chat content part type {type_name:?} for gRPC backends"
                        ));
                    }
                    ContentPart::Refusal { .. } if !assistant => {
                        return Err(
                            "Unsupported chat content part type \"refusal\" outside an assistant \
                             message for gRPC backends"
                                .to_string(),
                        );
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// The content the engine's chat utils hand a template for a message without
/// one (`null` or absent, as a tool-call-only assistant turn comes): the empty
/// string under the string format, the empty parts list under the OpenAI
/// format. The engine converts every message's content before it copies the
/// tool calls or the reasoning beside it, so a turn carrying either converts
/// the same way.
pub(crate) fn empty_content(content_format: ChatTemplateContentFormat) -> Value {
    match content_format {
        ChatTemplateContentFormat::String => Value::String(String::new()),
        ChatTemplateContentFormat::OpenAI => Value::Array(Vec::new()),
    }
}

fn process_content_format_with_order(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {e}"))?;

            if let Some(obj) = message_json.as_object_mut() {
                // The engines' chat utils build an assistant turn for the
                // template from its content, tool calls, reasoning and name. A
                // replayed refusal, audio reference or deprecated function call
                // is kept for an upstream and left out here, so the prompt is
                // the one the engine itself renders.
                if obj.get("role").and_then(Value::as_str) == Some("assistant") {
                    for key in ["refusal", "audio", "function_call"] {
                        obj.remove(key);
                    }
                }

                // skip_serializing_none omits content when None (a tool-call-only
                // assistant turn, the OpenAI shape), and a function result with
                // `content: null` keeps its null on the wire. The engine's chat
                // utils hand such a turn to the template as empty content in the
                // template's own format, never as none, which a template that
                // tests, splits or concatenates every message's content cannot
                // take.
                if matches!(
                    obj.get("role").and_then(|v| v.as_str()),
                    Some("assistant" | "function")
                ) && obj.get("content").is_none_or(Value::is_null)
                {
                    obj.insert("content".to_string(), empty_content(content_format));
                }

                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(
                        content_value,
                        content_format,
                        placeholder_tokens,
                        media_order,
                    )?;
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format.
///
/// For `String` templates, every media URL is replaced with its
/// modality-specific structural anchor when placeholders are configured. The
/// public preprocessing bindings do not have model metadata, so their legacy
/// `None` path continues to omit media parts.
///
/// Where a message's media goes relative to its text is the template's
/// business when the template takes parts (`OpenAI` format): it walks them
/// and writes each anchor where its part sits, so the parts reach it in the
/// request's order, as the engine's own server hands them to such a
/// template. A template that takes one string per message cannot place them
/// itself; there the media is written before the text by default (the
/// engine's own server's placement for such a template) and a renderer
/// whose string form is positional opts into the request's order
/// (`MediaPartOrder::Authored`).
fn transform_content_field(
    content_value: &mut Value,
    content_format: ChatTemplateContentFormat,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
) -> Result<(), String> {
    let Some(content_array) = content_value.as_array() else {
        return Ok(()); // Not multimodal, keep as-is
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Replace media parts with placeholders. The default order builds
            // separate media/text buckets for the front placement; the
            // authored order appends each rendered part as it is encountered.
            let mut media_parts: Vec<String> = Vec::new();
            let mut text_parts: Vec<String> = Vec::new();
            let mut authored_parts: Vec<String> = Vec::new();
            for part in content_array {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                match obj.get("type").and_then(|t| t.as_str()) {
                    // A refusal part is the turn's text, as the engines' own
                    // servers render it.
                    Some(type_name @ ("text" | "refusal")) => {
                        if let Some(t) = obj.get(type_name).and_then(|t| t.as_str()) {
                            match media_order {
                                MediaPartOrder::MediaFirst => text_parts.push(t.to_string()),
                                MediaPartOrder::Authored => authored_parts.push(t.to_string()),
                            }
                        }
                    }
                    Some(type_name @ ("image_url" | "video_url" | "audio_url" | "input_audio")) => {
                        let modality = modality_for_chat_part(type_name).ok_or_else(|| {
                            format!("unsupported media content part type: {type_name}")
                        })?;
                        let Some(tokens) = placeholder_tokens else {
                            continue;
                        };
                        let placeholder = tokens.get(modality).ok_or_else(|| {
                            format!(
                                "missing {modality} placeholder for string-format chat template"
                            )
                        })?;
                        match media_order {
                            MediaPartOrder::MediaFirst => {
                                media_parts.push(placeholder.to_string());
                            }
                            MediaPartOrder::Authored => {
                                authored_parts.push(placeholder.to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }

            let ordered: Vec<String> = match media_order {
                MediaPartOrder::MediaFirst => media_parts.into_iter().chain(text_parts).collect(),
                MediaPartOrder::Authored => authored_parts,
            };
            if !ordered.is_empty() {
                *content_value = Value::String(ordered.join("\n"));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            // Replace media URLs with simple type placeholders. The parts keep
            // the request's order whatever `media_order` says: the template
            // walks them and writes each anchor where its part sits (a text
            // part before an image renders before it), which is how the
            // engine's own server hands them to such a template. A refusal
            // part becomes a text part, as that server hands it over.
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| {
                            let type_str = obj.get("type")?.as_str()?;
                            match type_str {
                                "image_url" => Some(json!({"type": "image"})),
                                "video_url" => Some(json!({"type": "video"})),
                                "audio_url" | "input_audio" => Some(json!({"type": "audio"})),
                                "refusal" => {
                                    Some(json!({"type": "text", "text": obj.get("refusal")?}))
                                }
                                _ => None,
                            }
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }

    Ok(())
}

/// A popped assistant message's prefill: its string content, or its text parts joined in order.
fn prefill_text(content: Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|part| part.get("text")?.as_str())
                .collect(),
        ),
        _ => None,
    }
}

fn modality_for_chat_part(type_name: &str) -> Option<Modality> {
    match type_name {
        "image_url" | "image" => Some(Modality::Image),
        "audio_url" | "input_audio" | "audio" => Some(Modality::Audio),
        "video_url" | "video" => Some(Modality::Video),
        _ => None,
    }
}

/// Filter tools based on tool_choice (generic helper)
///
/// Returns filtered tools if filtering is needed, otherwise returns None.
/// Used by both Chat API and Responses API (Harmony) for constraint generation.
pub(crate) fn filter_tools_by_tool_choice(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
) -> Option<Vec<Tool>> {
    tool_choice.and_then(|choice| choice.narrow_tools(tools))
}

/// Filter ChatCompletionRequest by tool_choice
///
/// Returns a reference to the original request if no filtering needed,
/// otherwise returns a cloned request with filtered tools.
///
/// Note: Tool existence is validated earlier in ChatCompletionRequest::validate(),
/// so this function assumes tool_choice references valid tools.
pub(crate) fn filter_chat_request_by_tool_choice(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    if let Some(tools) = &body.tools {
        if let Some(filtered_tools) = filter_tools_by_tool_choice(tools, body.tool_choice.as_ref())
        {
            let mut filtered_body = body.clone();
            filtered_body.tools = Some(filtered_tools.into());
            return std::borrow::Cow::Owned(filtered_body);
        }
    }

    // No filtering needed - return original request
    std::borrow::Cow::Borrowed(body)
}

/// The request whose tools the template renders. A named `tool_choice`
/// renders every tool, as the engine's own server renders it (the named call
/// is forced by the tool constraint, built from `callable_tools`); an
/// `allowed_tools` choice renders its subset; any other request renders as
/// it is. Under `none` the tools stay declared, as the engine's own server
/// declares them (its prompt keeps the definitions and the choice reaches the
/// template as a kwarg, `build_chat_template_kwargs`): the model answers the
/// same prompt whichever way the choice is set, so a session that toggles the
/// choice between turns keeps its prefix. A call the model writes anyway is
/// not delivered where Symphony reads the response (the reader's forbidden
/// calls, `regular::symphony`); the legacy parsers run no tool parser under
/// `none`, as they never did, so there such markup stays in the content. The
/// Responses API keeps withholding the request's function tools under `none`
/// (`regular::responses::conversions::chat_tools`, for its replayed tool-loop
/// histories), so the same request renders a different prompt through it: a
/// divergence kept until that API's rule is decided on its own.
pub fn request_as_rendered(
    request: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    match request.tool_choice {
        Some(ToolChoice::Function { .. }) => std::borrow::Cow::Borrowed(request),
        _ => filter_chat_request_by_tool_choice(request),
    }
}

/// Process chat messages and apply template (shared by both routers)
/// Requires HuggingFace tokenizer with chat template support
///
/// Returns the flat prompt only. A renderer that must encode the prompt itself
/// (Kimi-K3) cannot be honored through this entry point; the gRPC pipeline
/// uses `process_chat_messages_with_placeholders` and encodes what it
/// returns, and a caller outside it takes
/// [`process_chat_messages_with_encoding`].
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
    image_placeholder: Option<&str>,
) -> Result<ProcessedMessages, String> {
    let (processed, encoding) =
        process_chat_messages_with_encoding(request, tokenizer, image_placeholder)?;
    if matches!(encoding, PromptEncoding::Deferred(_)) {
        static WARNED: OnceLock<()> = OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                "the tokenizer's renderer encodes prompts itself; a flat encode of \
                 ProcessedMessages.text may not reproduce its token ids (marker strings \
                 in message text, BPE merges across attribute-piece boundaries)"
            );
        });
    }
    Ok(processed)
}

/// [`process_chat_messages`] with how the tokenize step must encode the
/// prompt: [`PromptEncoding::FromText`] for every flat renderer, or the
/// deferred encode a segment-aware renderer prepared, whose ids are the
/// prompt the gateway sends.
pub fn process_chat_messages_with_encoding(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
    image_placeholder: Option<&str>,
) -> Result<(ProcessedMessages, PromptEncoding), String> {
    // Bindings call this entry point without the gRPC preparation stage.
    validate_chat_content_parts(&request.messages)?;
    let placeholder_tokens = image_placeholder.map(|token| {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, token.to_string());
        placeholders
    });
    process_chat_messages_with_placeholders(
        request,
        tokenizer,
        placeholder_tokens.as_ref(),
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
}

/// Render the chat prompt. The second element says how the tokenize step must
/// encode it: [`PromptEncoding::FromText`] for every flat renderer, or the
/// deferred encode a segment-aware renderer prepared.
pub(crate) fn process_chat_messages_with_placeholders(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
    placeholder_tokens: Option<&PlaceholderTokens>,
    media_order: MediaPartOrder,
    provider: ProviderProfile,
) -> Result<(ProcessedMessages, PromptEncoding), String> {
    let rendered = {
        // Get content format and transform messages accordingly: the format
        // of the template this request renders through, which a checkpoint
        // may pick by whether the request declares tools.
        let content_format =
            tokenizer.chat_template_content_format_for_request(request.tools.is_some());
        let mut transformed_messages = process_content_format_with_order(
            &request.messages,
            content_format,
            placeholder_tokens,
            media_order,
        )?;

        // Process tool call arguments in assistant messages. Renderers that
        // parse `arguments` strings themselves with the reference's tolerance,
        // and templates that write them as the text the request sent, get
        // them as written; every other template gets the parsed object the
        // Transformers docs expect.
        if !tokenizer.renderer_capabilities().raw_tool_call_arguments {
            process_tool_call_arguments(&mut transformed_messages);
        }

        // The tools as the template reads them, prepared once per distinct
        // tool set and shared by every request that carries the same tools.
        let prepared_tools = request.tools.as_deref().map(prepare_tools).transpose()?;

        let combined_template_kwargs =
            build_chat_template_kwargs(request, provider, tokenizer.thinking_key_name());

        let final_template_kwargs = if combined_template_kwargs.is_empty() {
            None
        } else {
            Some(&combined_template_kwargs)
        };

        let continues_final_assistant = request.continues_final_assistant();
        // Renderers that continue a trailing assistant message natively
        // (Jinja templates, cut right after the message as transformers
        // does; DeepSeek-V4.1: rendered without EOS and without a generation
        // header) keep the message and are called without a generation
        // prompt; other templates get the message popped and its content
        // appended after the generation prompt as a prefix.
        let native_continuation = continues_final_assistant
            && tokenizer
                .renderer_capabilities()
                .native_assistant_continuation;

        let params = ChatTemplateParams {
            // A continued final message decides its header by the renderer
            // (none natively, the emulation's otherwise); any other request's
            // header is its own `add_generation_prompt`, `true` when absent.
            add_generation_prompt: if continues_final_assistant {
                !native_continuation
            } else {
                request.adds_generation_prompt()
            },
            continue_final_message: native_continuation,
            tools: prepared_tools.as_deref().map(PreparedTools::tools),
            prepared_tools: prepared_tools.as_deref(),
            template_kwargs: final_template_kwargs,
            // The tokenizer applies the toggle under the template's own key.
            thinking: resolve_template_thinking(request, provider),
            ..Default::default()
        };

        // Handle assistant prefix for continue_final_message
        let assistant_prefix = if continues_final_assistant && !native_continuation {
            // Pop the last message to render it as the prefix. A trailing
            // assistant role implies a non-empty list, so the `else` arm is
            // only defensive.
            let Some(mut last_msg) = transformed_messages.pop() else {
                return Ok((
                    ProcessedMessages {
                        text: String::new(),
                        unbilled_prompt_tokens: 0,
                        continued_final_message: false,
                        generation_prompt: false,
                    },
                    PromptEncoding::FromText,
                ));
            };
            last_msg
                .get_mut("content")
                .map(Value::take)
                .and_then(prefill_text)
        } else {
            None
        };

        // Apply chat template with the (now possibly shorter) list of messages.
        // The prefill goes into the same call so the text and its encoding are
        // produced together and cannot drift apart.
        tokenizer
            .apply_chat_template_with_encoding_owned(
                transformed_messages,
                params,
                assistant_prefix.as_deref(),
            )
            .map_err(|e| format!("Failed to apply chat template: {e}"))?
    };

    let text = anchor_rendered_markers(placeholder_tokens, &rendered.encoding, rendered.text);
    Ok((
        ProcessedMessages {
            text,
            unbilled_prompt_tokens: rendered.unbilled_prompt_tokens,
            continued_final_message: rendered.continued_final_message,
            generation_prompt: rendered.generation_prompt,
        },
        rendered.encoding,
    ))
}

/// The stop rules of a request, built once: its stop strings (visible when `no_stop_trim`,
/// hidden otherwise), the tokenizer's end-of-sequence ids (unless `ignore_eos`) and its stop
/// token ids. A stream whose request asks several choices copies them into each choice's
/// decoder ([`StopSequenceDecoder::with_rules`]) instead of compiling them per choice.
pub fn stop_rules(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    no_stop_trim: bool,
    ignore_eos: bool,
) -> StopRules {
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };
    let mut config = StopSequenceConfig::default();
    // The stop strings are visible when no_stop_trim is set, hidden otherwise.
    if no_stop_trim {
        config.visible_stop_sequences = stop_sequences;
    } else {
        config.stop_sequences = stop_sequences;
    }

    // Collect stop token IDs: EOS from tokenizer (unless ignore_eos) + user-provided.
    // EOS tokens come from generation_config.json and are stripped at the token ID
    // level before decoding, matching vllm/sglang behavior.
    // When ignore_eos=true, EOS tokens are not added — the backend continues past EOS.
    let eos_ids = if ignore_eos {
        &[] as &[u32]
    } else {
        tokenizer.eos_token_ids()
    };
    for &token_id in eos_ids
        .iter()
        .chain(stop_token_ids.map(|ids| ids.as_slice()).unwrap_or_default())
    {
        config = if no_stop_trim {
            config.with_visible_stop_token(token_id)
        } else {
            config.with_stop_token(token_id)
        };
    }
    StopRules::new(config)
}

/// Create a StopSequenceDecoder from stop parameters: the request's rules
/// ([`stop_rules`]) and a decode state of its own.
pub fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
    ignore_eos: bool,
) -> StopSequenceDecoder {
    StopSequenceDecoder::with_rules(
        tokenizer.clone(),
        stop_rules(tokenizer, stop, stop_token_ids, no_stop_trim, ignore_eos),
        skip_special_tokens,
    )
}

/// With parallel tool calls turned off (`parallel_tool_calls: false`, or
/// Messages `disable_parallel_tool_use: true`) a response returns only its
/// first tool call; the calls a model writes after it are dropped.
pub(crate) fn keep_first_tool_call(calls: &mut Option<Vec<ToolCall>>) {
    if let Some(calls) = calls.as_mut().filter(|calls| calls.len() > 1) {
        debug!(
            dropped = calls.len() - 1,
            "parallel tool calls are off: returning the first call"
        );
        calls.truncate(1);
    }
}

/// The streamed form of [`keep_first_tool_call`].
pub(crate) fn keep_first_tool_call_items(items: &mut Vec<ToolCallItem>) {
    items.retain(|item| item.tool_index == 0);
}

/// Parse tool calls from JSON schema constrained response
pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: Option<&ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            // Specific function: Parse parameters directly
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: generate_tool_call_id(
                            model,
                            &function.name,
                            0,
                            history_tool_calls_count,
                        ),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
            // Required mode: Parse array of tool calls
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: generate_tool_call_id(
                                    model,
                                    &name,
                                    i,
                                    history_tool_calls_count,
                                ),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string:
/// - Kimi-K3 (XTML): `{name}_{history_count + tool_index}` — an opaque ordinal
///   that keeps counting across turns, so a second turn's three calls are
///   `3,4,5` rather than restarting at `0`. The `_` separator matches the
///   reference K3 server's id shape (`Bash_0`); K2's `:` is protocol-visible
///   inside the prompt, but K3 never renders the id and matches tool results
///   back to calls by opaque id equality, so the separator is free to differ.
///   The history offset keeps ids unique over a conversation (OpenAI treats
///   `tool_call_id` as globally unique, and clients correlate by it).
/// - Kimi-K2: `functions.{name}:{history_count + tool_index}` (globally unique).
/// - others: `call_{24-char-uuid}`.
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    // Case-insensitive substring checks without allocation.
    let is_kimi = model
        .as_bytes()
        .windows(4) // "kimi".len()
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if !is_kimi {
        // Standard OpenAI format: call_{24-char-uuid}
        return format!("call_{}", &Uuid::now_v7().simple().to_string()[..24]);
    }

    let is_k3 = model
        .as_bytes()
        .windows(2) // "k3".len()
        .any(|window| window.eq_ignore_ascii_case(b"k3"));

    if is_k3 {
        // Kimi-K3 (XTML) opaque format: {name}_{global_index}.
        format!("{tool_name}_{}", history_count + tool_index)
    } else {
        // Kimi-K2 format: functions.{name}:{global_index}.
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    }
}

/// The completion API's finish reason from the engine's: a local stop beats
/// the engine's reason (which is "length" when stop strings are enforced
/// gateway-side rather than by the backend); the API's words `stop`,
/// `length` and `content_filter` pass, and so do the engines' own signals
/// `abort`, `error` and `failed`, which the chat paths pass through too and
/// clients read as the failure they are; each as a plain word or as a JSON
/// object's `type`. Anything else (an empty reason, an unmapped type, an
/// unrecognized word) is `stop`, logged once with the raw value. One table
/// for the streamed and the non-streamed completion.
pub(crate) fn completion_finish_reason(local_stop: bool, engine: &str) -> &'static str {
    if local_stop {
        return "stop";
    }
    let known = |word: &str| -> Option<&'static str> {
        match word {
            "stop" => Some("stop"),
            "length" => Some("length"),
            "content_filter" => Some("content_filter"),
            "abort" => Some("abort"),
            "error" => Some("error"),
            "failed" => Some("failed"),
            _ => None,
        }
    };
    if let Some(reason) = known(engine) {
        return reason;
    }
    if let Ok(json) = serde_json::from_str::<Value>(engine) {
        if let Some(reason) = json.get("type").and_then(Value::as_str).and_then(known) {
            return reason;
        }
    }
    if !engine.is_empty() {
        warn!(
            unexpected_finish_reason = engine,
            "Unrecognized finish_reason from backend, defaulting to stop"
        );
    }
    "stop"
}

/// Parse finish_reason string into GenerateFinishReason enum
///
/// Uses serde to deserialize the finish_reason, which handles all tagged variants automatically.
/// The GenerateFinishReason enum is tagged with `#[serde(tag = "type", rename_all = "lowercase")]`,
/// so it expects JSON objects like:
/// - `{"type":"stop"}` -> Stop
/// - `{"type":"length","length":100}` -> Length { length: 100 }
/// - Any other JSON -> Other(...)
///
/// For backward compatibility, also handles simple string "stop" -> Stop
pub(crate) fn parse_finish_reason(
    reason_str: &str,
    completion_tokens: u32,
) -> GenerateFinishReason {
    if reason_str == "stop" {
        return GenerateFinishReason::Stop {
            finish_type: openai_protocol::generate::GenerateFinishType::Stop,
        };
    }

    if reason_str == "length" {
        return GenerateFinishReason::Length {
            finish_type: openai_protocol::generate::GenerateFinishType::Length,
            length: completion_tokens,
        };
    }

    match serde_json::from_str::<GenerateFinishReason>(reason_str) {
        Ok(finish_reason) => finish_reason,
        Err(_) => match serde_json::from_str::<Value>(reason_str) {
            Ok(json_value) => GenerateFinishReason::Other(json_value),
            Err(_) => GenerateFinishReason::Other(Value::String(reason_str.to_string())),
        },
    }
}

#[cfg(test)]
mod tests;
