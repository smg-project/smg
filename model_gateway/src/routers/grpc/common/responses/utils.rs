//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    common::{Tool, Usage},
    responses::{
        CustomTool, IncompleteDetails, IncompleteReason, InputTokensDetails, NamespaceTool,
        OutputTokensDetails, ResponseStatus, ResponseTool, ResponseUsage, ResponsesRequest,
        ResponsesResponse,
    },
};
use serde_json::{json, to_value, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseId, ResponseStorage, ResponseStorageError, StoredResponse,
};
use smg_mcp::{McpOrchestrator, McpServerBinding};
use tracing::{debug, error, warn};

use super::background;
use crate::{
    observability::metrics::Metrics,
    routers::{
        common::{
            mcp_utils::{ensure_request_mcp_routing, DEFAULT_MAX_ITERATIONS},
            openai_bridge,
            persistence_utils::{
                persist_conversation_items_if_open, persist_conversation_items_owned, PersistError,
                PersistStep,
            },
            snapshot::WorkerRegistry,
        },
        error,
    },
};

/// Supply failure details when an engine reports only a terminal finish reason.
/// A structured upstream error, when available, takes precedence over this fallback.
pub(crate) fn generation_failure_error(finish_reason: Option<&str>) -> Option<Value> {
    matches!(finish_reason, Some("failed" | "error")).then(|| {
        serde_json::json!({
            "code": "server_error",
            "message": "Upstream generation failed",
        })
    })
}

/// Map a Chat finish reason to a generated function-call item's status: a call whose arguments
/// are no JSON value is `incomplete` whatever the finish, since a client must not run it as a
/// whole one: cut by the token limit or an abort, or ended by the model and not read whole (a
/// tagged syntax that refused bytes inside the call writes no close for it; a JSON syntax carries
/// the bytes as written from where they stopped being JSON), the turn ending on the call
/// (`tool_calls`) all the same; a failed or cancelled generation stays `in_progress`.
pub(crate) fn function_call_status(finish_reason: Option<&str>, arguments: &str) -> &'static str {
    match finish_reason {
        Some("failed" | "error" | "cancelled") => "in_progress",
        _ if serde_json::from_str::<Value>(arguments).is_err() => "incomplete",
        _ => "completed",
    }
}

/// Whether a turn's calls hold one that is no whole call, cut or not read whole: a call whose
/// status under the finish reason reads `incomplete` (`function_call_status`), read on the
/// function-shaped arguments of every call before a custom tool's call becomes its
/// `custom_tool_call` item, which carries no status on the wire.
pub(crate) fn cut_call<'a>(
    finish_reason: Option<&str>,
    arguments: impl IntoIterator<Item = &'a str>,
) -> bool {
    arguments
        .into_iter()
        .any(|arguments| function_call_status(finish_reason, arguments) == "incomplete")
}

/// The response's status for the Chat finish reason, with why it is incomplete: a `length`
/// finish is the token limit; a call that is no whole call under any other finish (`cut_call`,
/// read on the calls' function-shaped arguments) leaves the response incomplete with no stated
/// reason; `cancelled` is the gateway's own finish for a stream stopped before its end (the
/// client left a stored stream), `cancelled` with no reason. Every
/// builder of a Responses object reads its status here: the non-streaming conversions,
/// regular and Harmony, the stored object of a regular streamed turn, and the emitter for its
/// terminal event and stored object under the finish it holds, whether chat chunks or a
/// Harmony stream gave it.
pub(crate) fn response_status(
    finish_reason: Option<&str>,
    cut_call: bool,
) -> (ResponseStatus, Option<IncompleteDetails>) {
    match finish_reason {
        Some("failed" | "error") => (ResponseStatus::Failed, None),
        Some("cancelled") => (ResponseStatus::Cancelled, None),
        Some("length") => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: IncompleteReason::MaxOutputTokens,
            }),
        ),
        _ if cut_call => (ResponseStatus::Incomplete, None),
        _ => (ResponseStatus::Completed, None),
    }
}

/// The MCP tool calls a request may make in all: its `max_tool_calls`
/// capped by the loop's safety limit, the safety limit when it names none.
pub(crate) fn tool_call_limit(max_tool_calls: Option<usize>) -> usize {
    max_tool_calls.map_or(DEFAULT_MAX_ITERATIONS, |user_max| {
        user_max.min(DEFAULT_MAX_ITERATIONS)
    })
}

/// The server calls a request may still run once `executed` ran: its cap (see
/// `tool_call_limit`) less them.
pub(crate) fn tool_call_allowance(max_tool_calls: Option<usize>, executed: usize) -> usize {
    tool_call_limit(max_tool_calls).saturating_sub(executed)
}

/// Whether a Harmony loop generation is past the loop's safety net. The call
/// limit (see `tool_call_limit`) already bounds the loop, since every turn
/// that continues it runs at least one server call, so the net allows that
/// many generations plus the one that answers on the last result; the final
/// turn after the request's cap is always the last generation and is exempt.
pub(crate) fn iteration_exceeds_safety_limit(iteration_count: usize, final_turn: bool) -> bool {
    iteration_count > DEFAULT_MAX_ITERATIONS + 1 && !final_turn
}

/// Adds one model call's usage to a response's total. Usage is per model
/// request, so a tool loop reports the sum of its calls; a call that reported
/// none leaves the total as it is.
pub(crate) fn add_model_call_usage(total: &mut Option<ResponseUsage>, usage: Option<&Usage>) {
    let Some(usage) = usage else {
        return;
    };
    let total = total.get_or_insert_with(|| ResponseUsage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        input_tokens_details: InputTokensDetails::default(),
        output_tokens_details: OutputTokensDetails::default(),
    });
    total.input_tokens = total.input_tokens.saturating_add(usage.prompt_tokens);
    total.output_tokens = total.output_tokens.saturating_add(usage.completion_tokens);
    total.total_tokens = total.total_tokens.saturating_add(usage.total_tokens);
    if let Some(details) = &usage.prompt_tokens_details {
        total.input_tokens_details.cached_tokens = total
            .input_tokens_details
            .cached_tokens
            .saturating_add(details.cached_tokens);
    }
    if let Some(tokens) = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
    {
        total.output_tokens_details.reasoning_tokens = total
            .output_tokens_details
            .reasoning_tokens
            .saturating_add(tokens);
    }
}

/// Why a generated batch of server calls must stop after its permitted prefix
/// executes: the request's own `max_tool_calls`, or the loop's internal safety
/// limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum McpToolCallLimit {
    User,
    Safety,
}

/// The cap a batch of `batch` server calls overflows once `executed` calls
/// ran: `None` when the batch fits the remaining allowance, else the cap and
/// the length of the batch's permitted prefix. The cap is the request's own
/// when it names one at or under the safety limit (a user cap wins when both
/// are equal), else the safety limit.
pub(crate) fn tool_call_overflow(
    max_tool_calls: Option<usize>,
    executed: usize,
    batch: usize,
) -> Option<(McpToolCallLimit, usize)> {
    let remaining = tool_call_allowance(max_tool_calls, executed);
    if batch <= remaining {
        return None;
    }
    let limit = if max_tool_calls.is_some_and(|limit| limit <= DEFAULT_MAX_ITERATIONS) {
        McpToolCallLimit::User
    } else {
        McpToolCallLimit::Safety
    };
    Some((limit, remaining))
}

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared.
///
/// Checks if the request declares MCP tools or builtin tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`) and,
/// if so, validates that the MCP clients can be created and connected.
///
/// Returns Ok((has_mcp_tools, mcp_servers)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &openai_bridge::FormatRegistry,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<McpServerBinding>), Response> {
    // Check for explicit MCP tools (must error if connection fails)
    let has_explicit_mcp_tools = tools
        .map(|t| t.iter().any(|tool| matches!(tool, ResponseTool::Mcp(_))))
        .unwrap_or(false);

    // Check for builtin tools that MAY have MCP routing configured.
    //
    // `ImageGeneration` is included here because gpt-oss via the
    // harmony pipeline, and Qwen/Llama via the regular pipeline, both
    // dispatch hosted `image_generation` calls through the same MCP
    // routing path — the only difference is how the tool is advertised in
    // the prompt. Without this arm, the short-circuit below would return
    // `(false, Vec::new())`, the MCP loop would never be entered, and the
    // registered `image_generation` MCP server would receive zero
    // dispatches.
    let has_builtin_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| openai_bridge::builtin_type_for_response_tool(tool).is_some())
        })
        .unwrap_or(false);

    // Only process if we have MCP or builtin tools
    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok((false, Vec::new()));
    }

    if let Some(tools) = tools {
        // TODO: Thread real request headers through the gRPC responses path if/when
        // gRPC MCP flows need the same forwarded-header preservation contract.
        if let Some(mcp_servers) =
            ensure_request_mcp_routing(mcp_orchestrator, format_registry, tools).await?
        {
            return Ok((true, mcp_servers));
        }
        // Builtin tools without MCP routing - pass through to model
        debug!(
            function = "ensure_mcp_connection",
            "No MCP routing configured for builtin tools, passing through to model"
        );
    }

    Ok((false, Vec::new()))
}

/// Refuse tools that disallow direct calls, including namespace members.
/// This router can offer tools to the model only for direct calls.
pub(crate) fn refuse_tools_without_a_direct_caller(
    tools: Option<&[ResponseTool]>,
) -> Option<Response> {
    let refused_at = |idx: usize, tool: &ResponseTool| match tool {
        ResponseTool::Namespace(namespace) => {
            namespace
                .tools
                .iter()
                .enumerate()
                .find_map(|(member_idx, member)| {
                    let direct = match member {
                        NamespaceTool::Function(ft) => ft.allows_direct_calls(),
                        NamespaceTool::Custom(ct) => ct.allows_direct_calls(),
                    };
                    (!direct).then(|| format!("tools[{idx}].tools[{member_idx}]"))
                })
        }
        _ => (!tool.allows_direct_calls()).then(|| format!("tools[{idx}]")),
    };
    let at = tools
        .unwrap_or_default()
        .iter()
        .enumerate()
        .find_map(|(idx, tool)| refused_at(idx, tool))?;
    Some(error::bad_request(
        "tool_not_supported",
        format!(
            "Invalid '{at}.allowed_callers': this backend runs no programmatic tool calls, so a tool must allow the 'direct' caller."
        ),
    ))
}

/// Validate that workers are available for the requested model.
///
/// Runs on the client-supplied name, before the pipeline canonicalizes it, so
/// it has to accept aliases as well as canonical model IDs. `contains_model`
/// covers both; listing `get_models()` and testing membership would reject
/// every alias here.
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    if !worker_registry.contains_model(model) {
        return Some(error::model_not_found(model));
    }

    None
}

/// Extract function tools from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts function tools because MCP tools are exposed to the model as
///   function tools (via `convert_mcp_tools_to_response_tools()`), and those are used to
///   generate structural constraints in the Harmony preparation stage.
///
/// - **Regular router**: Extracts function tools during the initial conversion from
///   ResponsesRequest to ChatCompletionRequest. MCP tools are merged later by the tool loop.
///
/// Custom tools (`{"type": "custom"}`) are downgraded to function tools with a
/// single `input: string` parameter; `chat_to_responses` maps the calls back.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .flat_map(|tool| match tool {
            ResponseTool::Function(ft) => vec![Tool {
                tool_type: "function".to_string(),
                function: ft.chat_function(),
            }],
            ResponseTool::Custom(ct) => vec![Tool {
                tool_type: "function".to_string(),
                function: custom_tool_as_function(ct),
            }],
            ResponseTool::Namespace(namespace) => namespace
                .tools
                .iter()
                .map(|member| {
                    let mut function = match member {
                        NamespaceTool::Function(ft) => ft.chat_function(),
                        NamespaceTool::Custom(ct) => custom_tool_as_function(ct),
                    };
                    function.name = format!("{}.{}", namespace.name, function.name);
                    Tool {
                        tool_type: "function".to_string(),
                        function,
                    }
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Downgrade a Responses custom tool to a chat function tool whose single
/// `input` string parameter carries the free-form payload.
pub(crate) fn custom_tool_as_function(ct: &CustomTool) -> openai_protocol::common::Function {
    openai_protocol::common::Function {
        name: ct.name.clone(),
        description: ct.description.clone(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The raw payload to pass to this custom tool."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        })),
        strict: Some(false),
        extra: Default::default(),
    }
}

/// Recover a custom tool's raw `input` from the `{"input": "..."}` arguments
/// of its downgraded function call, preserving the raw `arguments` string if
/// the model emitted plain text or another JSON shape instead.
pub(crate) fn custom_tool_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|v| v.get("input")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| arguments.to_owned())
}

/// Collect the names of all declared custom tools (top-level and namespaced),
/// used to map function-shaped calls back to `custom_tool_call` items.
pub(crate) fn custom_tool_names(
    response_tools: Option<&[ResponseTool]>,
) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let Some(tools) = response_tools else {
        return names;
    };
    for tool in tools {
        match tool {
            ResponseTool::Custom(ct) => {
                names.insert(ct.name.clone());
            }
            ResponseTool::Namespace(namespace) => {
                for member in &namespace.tools {
                    if let NamespaceTool::Custom(ct) = member {
                        names.insert(format!("{}.{}", namespace.name, ct.name));
                    }
                }
            }
            _ => {}
        }
    }
    names
}

/// Version tag for the opaque `reasoning.encrypted_content` payload.
const REASONING_BLOB_PREFIX: &str = "smg-reasoning-v1.";

/// Encode reasoning text as the opaque `encrypted_content` blob a `store=false`
/// client hands back on its next turn. The gateway is stateless, so the blob
/// carries the text itself (version-tagged base64): opaque, not confidential.
pub(crate) fn encode_reasoning_content(text: &str) -> String {
    format!("{REASONING_BLOB_PREFIX}{}", BASE64_STANDARD.encode(text))
}

/// Recover reasoning text from a blob produced by [`encode_reasoning_content`].
/// Blobs from other producers yield `None`.
pub(crate) fn decode_reasoning_content(blob: &str) -> Option<String> {
    let payload = blob.strip_prefix(REASONING_BLOB_PREFIX)?;
    String::from_utf8(BASE64_STANDARD.decode(payload).ok()?).ok()
}

/// Whether `name` is `namespace` and `member` joined by a dot, read at the
/// dot without building the joined text.
fn is_namespaced(name: &str, namespace: &str, member: &str) -> bool {
    name.strip_prefix(namespace)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some_and(|rest| rest == member)
}

/// Recover structured identity only for a declared namespace member.
/// Literal top-level names (including dots) take precedence over namespace matches.
pub(crate) fn resolve_function_identity(
    tools: Option<&[ResponseTool]>,
    name: &str,
) -> (String, Option<String>) {
    let tools = tools.unwrap_or_default();
    if !tools.iter().any(|tool| match tool {
        ResponseTool::Function(ft) => ft.function.name == name,
        ResponseTool::Custom(ct) => ct.name == name,
        _ => false,
    }) {
        for tool in tools {
            if let ResponseTool::Namespace(namespace) = tool {
                for member in &namespace.tools {
                    let member_name = match member {
                        NamespaceTool::Function(ft) => &ft.function.name,
                        NamespaceTool::Custom(ct) => &ct.name,
                    };
                    if is_namespaced(name, &namespace.name, member_name) {
                        return (member_name.clone(), Some(namespace.name.clone()));
                    }
                }
            }
        }
    }
    (name.to_string(), None)
}

/// Synthetic same-name members used by namespace routing regression tests.
#[cfg(test)]
pub(crate) fn namespace_test_request() -> ResponsesRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": "Check the weather",
        "tools": [
            {"type": "namespace", "name": "weather", "description": "Weather tools", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
            ]},
            {"type": "namespace", "name": "travel", "description": "Travel tools", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
            ]}
        ]
    })).unwrap()
}

/// The usage the serialization-path tests feed in: 12 input and 7 output
/// tokens, 3 of them cached and 2 reasoning; `assert_responses_api_usage`
/// reads it back off the wire.
#[cfg(test)]
pub(crate) fn responses_api_test_usage() -> Usage {
    Usage::from_counts(12, 7)
        .with_cached_tokens(3)
        .with_reasoning_tokens(2)
}

/// The Responses usage object as the tests of the serialization paths read it
/// off the wire: `responses_api_test_usage` under the Responses names only (no
/// chat-style `prompt_tokens` / `completion_tokens`).
#[cfg(test)]
pub(crate) fn assert_responses_api_usage(wire: &Value) {
    let usage = wire.get("usage").expect("usage should be present");

    assert_eq!(usage.get("input_tokens"), Some(&serde_json::json!(12)));
    assert_eq!(usage.get("output_tokens"), Some(&serde_json::json!(7)));
    assert_eq!(usage.get("total_tokens"), Some(&serde_json::json!(19)));
    assert_eq!(
        usage.pointer("/input_tokens_details/cached_tokens"),
        Some(&serde_json::json!(3))
    );
    assert_eq!(
        usage.pointer("/output_tokens_details/reasoning_tokens"),
        Some(&serde_json::json!(2))
    );
    assert!(usage.get("prompt_tokens").is_none());
    assert!(usage.get("completion_tokens").is_none());
}

/// A response the client expects stored (`store` unset or true) whose record
/// could not be written: nothing of it is stored, so it can be neither
/// retrieved nor continued. A foreground request is answered with it, so the
/// client knows the response was not kept instead of finding its id unknown.
#[derive(Debug)]
pub(crate) struct NotStored {
    /// The store was not available (the same write can succeed later); a
    /// record the response itself makes unstorable is not retried.
    pub retryable: bool,
    pub message: String,
}

impl NotStored {
    fn new(retryable: bool, error: impl std::fmt::Display) -> Self {
        Self {
            retryable,
            message: format!("The response was not stored: {error}"),
        }
    }

    /// The error code: the store-class `storage_unavailable` for a store that
    /// is not available, `store_response_failed` otherwise.
    pub(crate) fn code(&self) -> &'static str {
        if self.retryable {
            "storage_unavailable"
        } else {
            "store_response_failed"
        }
    }

    /// The `error` object of the `response.failed` terminal event.
    pub(crate) fn error_object(&self) -> Value {
        json!({"code": self.code(), "message": self.message})
    }

    /// The answer of a buffered request: 503 for a store that is not
    /// available, 500 otherwise (`error.type` server_error).
    pub(crate) fn into_response(self) -> Response {
        if self.retryable {
            error::service_unavailable(self.code(), self.message)
        } else {
            error::internal_error(self.code(), self.message)
        }
    }
}

/// The responses behind a `previous_response_id`, oldest first, as the store
/// holds them (shared, read in place), or the refusal every loader answers: `previous_response_not_found` for an id the
/// store does not know, an empty chain, or a chain ending in a record its
/// client asked not to store (a background run's record, kept for its
/// polling window only: a chain built on it would break when it goes, so it
/// is no previous response, as the public API answers for an unstored id),
/// or a chain with no record the caller's `tenant` may see (the chain ends at
/// the first record of another tenant, as it would at a missing one);
/// `load_previous_response_chain_failed` for a store that failed (a 503
/// `storage_unavailable` when it timed out, see `error::storage_failure`).
pub(crate) async fn previous_response_chain(
    storage: &Arc<dyn ResponseStorage>,
    prev_id_str: &str,
    tenant: Option<&str>,
) -> Result<Vec<Arc<StoredResponse>>, Response> {
    let prev_id = ResponseId::from(prev_id_str);
    match storage.get_response_chain_for(&prev_id, tenant).await {
        Ok(chain) if !chain.responses.is_empty() => Ok(chain.responses),
        Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => Err(error::not_found(
            "previous_response_not_found",
            format!("Previous response with id '{prev_id_str}' not found."),
        )),
        Err(e) => {
            error!(
                prev_id = %prev_id_str,
                error = %e,
                "Failed to load previous response chain from storage"
            );
            Err(error::storage_failure(
                "load_previous_response_chain_failed",
                format!("Failed to load previous response chain for {prev_id_str}"),
                &e,
            ))
        }
    }
}

/// Whether `request` is a foreground response the client expects stored
/// (`store` unset or true, not `background`): its persist is part of the
/// answer, where a background response's outcome is in the record its client
/// polls and an unstored response stores nothing.
pub(crate) fn stored_foreground(request: &ResponsesRequest) -> bool {
    request.store.unwrap_or(true) && !request.background.unwrap_or(false)
}

/// Persist response to storage if store=true; an unstored background response
/// keeps its result in the record its client polls, for the polling window
/// only (see [`background::finish_unstored`]).
///
/// Answers `Err` for a foreground response the client expects stored whose
/// record could not be written (see [`NotStored`]); the caller fails the
/// request with it. A background response's failures are logged and counted
/// here (its record keeps what it had), as is a record that was stored but
/// whose items could not be linked into the conversation, and an unstored
/// response has nothing to fail: those answer `Ok`.
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations. The record is
/// scoped to the caller's `tenant`.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    tenant: Option<&str>,
    request_context: Option<StorageRequestContext>,
) -> Result<(), NotStored> {
    if !original_request.store.unwrap_or(true) {
        // Nothing of an unstored response reaches the conversation and
        // history side; a background one still has the record its client
        // polls, and the result goes there.
        if original_request.background.unwrap_or(false) {
            background::finish_unstored(&response_storage, response, request_context).await;
        }
        return Ok(());
    }
    let background = original_request.background.unwrap_or(false);
    let mode = if background {
        "background"
    } else {
        "foreground"
    };
    let mut response_json = match to_value(response) {
        Ok(mut json) => {
            exact_sampling_numbers(&mut json, response);
            json
        }
        Err(e) => {
            warn!(response_id = %response.id, error = %e, "Failed to serialize the response for storage");
            Metrics::record_response_not_stored(mode, "record");
            return if background {
                Ok(())
            } else {
                Err(NotStored::new(false, e))
            };
        }
    };
    // A background response's record may have been closed while the work
    // ran (a cancel, here or on another gateway sharing the store): its
    // terminal object is stored with one conditional write, so
    // a closed record is never rewritten by a late result. The write is
    // retried at the status-watch cadence within the terminal write window
    // while the store answers with an error: the result of a run that
    // outlived a store blip is stored once the store is back, and a record
    // closed meanwhile refuses it then. Only the write of the response record
    // is retried: once it landed, the conversation linking runs once (as
    // before), so a retry can neither store the record twice nor link the
    // items twice. A foreground response is written once, as before: its
    // client is waiting for the answer. A background stream attached through
    // the Harmony paths receives its terminal event after this persist, so
    // during a store outage that event can wait for up to the window; the
    // regular streaming paths emit it first.
    let every = background::STATUS_WATCH;
    let deadline = tokio::time::Instant::now() + background::terminal_write_window(every);
    let mut attempts = 0u32;
    loop {
        let persisted = if background {
            persist_conversation_items_if_open(
                conversation_storage.clone(),
                conversation_item_storage.clone(),
                response_storage.clone(),
                &response_json,
                original_request,
                tenant,
                request_context.clone(),
            )
            .await
        } else {
            // Only a background persist comes round again: the store takes the tree.
            debug_assert!(
                !response_json.is_null(),
                "a foreground persist runs once; its tree is taken"
            );
            persist_conversation_items_owned(
                conversation_storage.clone(),
                conversation_item_storage.clone(),
                response_storage.clone(),
                std::mem::take(&mut response_json),
                original_request,
                tenant,
                request_context.clone(),
            )
            .await
            .map(|()| true)
        };
        match persisted {
            Ok(true) => {
                debug!("Persisted response: {}", response.id);
                return Ok(());
            }
            Ok(false) => {
                debug!(
                    response_id = %response.id,
                    "Background response record already closed; the result is not stored"
                );
                return Ok(());
            }
            Err(e)
                if background
                    && e.step == PersistStep::ResponseRecord
                    && e.retryable
                    && tokio::time::Instant::now() < deadline =>
            {
                if attempts == 0 {
                    warn!(
                        response_id = %response.id,
                        error = %e,
                        "Failed to persist the background response; retrying while the store is unreachable"
                    );
                } else {
                    debug!(response_id = %response.id, attempts, error = %e, "The background response persist still fails");
                }
                attempts += 1;
                tokio::time::sleep(every).await;
            }
            Err(e)
                if matches!(
                    e.step,
                    PersistStep::ConversationLookup | PersistStep::Linking
                ) =>
            {
                warn!(
                    response_id = %response.id,
                    error = %e,
                    "Response stored; its items were not linked into the conversation"
                );
                Metrics::record_response_not_stored(mode, "linking");
                return Ok(());
            }
            Err(e) => {
                warn!(response_id = %response.id, error = %e, "Failed to persist response; nothing of it is stored");
                Metrics::record_response_not_stored(mode, "record");
                return if background {
                    Ok(())
                } else {
                    Err(not_stored(&e))
                };
            }
        }
    }
}

/// [`NotStored`] for a persist that failed before the record was written.
fn not_stored(error: &PersistError) -> NotStored {
    NotStored::new(error.retryable, error)
}

/// `value` as a JSON number the way the `f32` prints (`0.7`), not widened
/// to 64 bits (`0.699999988079071`); the text is written on the stack.
pub(crate) fn exact_f32(value: f32) -> Value {
    let mut text = [0u8; 32];
    let mut rest = &mut text[..];
    let exact = match serde_json::to_writer(&mut rest, &value) {
        Ok(()) => {
            let left = rest.len();
            serde_json::from_slice(&text[..text.len() - left]).ok()
        }
        Err(_) => None,
    };
    // An f32 prints in at most 16 bytes, so this never falls back in
    // practice; if it did, the widened number is kept.
    exact.unwrap_or_else(|| Value::from(value))
}

/// Writes `temperature` and `top_p` into the JSON tree of a response as
/// the response holds them: the plain conversion widens the two 32-bit
/// numbers to 64 bits.
pub(crate) fn exact_sampling_numbers(raw: &mut Value, response: &ResponsesResponse) {
    for (key, value) in [
        ("temperature", response.temperature),
        ("top_p", response.top_p),
    ] {
        if let (Some(slot), Some(value)) = (raw.get_mut(key), value) {
            *slot = exact_f32(value);
        }
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/utils/tests.rs"]
mod tests;
