//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    common::Tool,
    responses::{CustomTool, NamespaceTool, ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde_json::{json, to_value, Value};
use smg_data_connector::{
    is_unstored, visible_chain, ConversationItemStorage, ConversationStorage,
    RequestContext as StorageRequestContext, ResponseId, ResponseStorage, ResponseStorageError,
    StoredResponse,
};
use smg_mcp::{McpOrchestrator, McpServerBinding};
use tracing::{debug, error, warn};

use super::background;
use crate::{
    observability::metrics::Metrics,
    routers::{
        common::{
            mcp_utils::ensure_request_mcp_client,
            openai_bridge,
            persistence_utils::{persist_conversation_items, PersistError, PersistStep},
        },
        error,
    },
    worker::WorkerRegistry,
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

/// Map a Chat finish reason to a generated function-call item's status.
/// On token-limit truncation, preserve complete arguments and mark partial JSON incomplete.
pub(crate) fn function_call_status(finish_reason: Option<&str>, arguments: &str) -> &'static str {
    match finish_reason {
        Some("failed" | "error") => "in_progress",
        Some("length") if serde_json::from_str::<Value>(arguments).is_err() => "incomplete",
        _ => "completed",
    }
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
        match ensure_request_mcp_client(mcp_orchestrator, format_registry, tools).await {
            Some(mcp_servers) => {
                return Ok((true, mcp_servers));
            }
            None => {
                // No MCP servers available
                if has_explicit_mcp_tools {
                    // Explicit MCP tools MUST have working connections
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                // Builtin tools without MCP routing - pass through to model
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok((false, Vec::new()));
            }
        }
    }

    Ok((false, Vec::new()))
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
                function: ft.function.clone(),
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
                        NamespaceTool::Function(ft) => ft.function.clone(),
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
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The raw payload to pass to this custom tool."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        }),
        strict: Some(false),
        extra: Default::default(),
    }
}

/// Recover a custom tool's raw `input` from the `{"input": "..."}` arguments
/// of its downgraded function call.
pub(crate) fn custom_tool_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|v| v.get("input")?.as_str().map(str::to_owned))
        .unwrap_or_default()
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

/// Recover structured identity only for a declared namespace member.
/// Literal top-level names (including dots) take precedence over namespace matches.
pub(crate) fn resolve_function_identity(
    tools: Option<&[ResponseTool]>,
    name: &str,
) -> (String, Option<String>) {
    let tools = tools.unwrap_or_default();
    if !tools
        .iter()
        .any(|tool| matches!(tool, ResponseTool::Function(ft) if ft.function.name == name))
    {
        for tool in tools {
            if let ResponseTool::Namespace(namespace) = tool {
                for member in &namespace.tools {
                    if let NamespaceTool::Function(ft) = member {
                        if name == format!("{}.{}", namespace.name, ft.function.name) {
                            return (ft.function.name.clone(), Some(namespace.name.clone()));
                        }
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

/// The responses behind a `previous_response_id`, oldest first, or the
/// refusal every loader answers: `previous_response_not_found` for an id the
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
) -> Result<Vec<StoredResponse>, Response> {
    let prev_id = ResponseId::from(prev_id_str);
    match storage.get_response_chain(&prev_id, None).await {
        Ok(chain)
            if !chain.responses.is_empty() && !chain.responses.last().is_some_and(is_unstored) =>
        {
            let responses = visible_chain(chain.responses, tenant);
            if responses.is_empty() {
                return Err(error::bad_request(
                    "previous_response_not_found",
                    format!("Previous response with id '{prev_id_str}' not found."),
                ));
            }
            Ok(responses)
        }
        Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => Err(error::bad_request(
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
    // A background response's record may have been closed while the work
    // ran (a cancel, here or on another gateway sharing the store), or
    // deleted: a terminal record is not rewritten by a late result and a
    // deleted one is not re-created by it.
    if background
        && background::is_closed(&response_storage, &response.id, request_context.clone()).await
    {
        debug!(
            response_id = %response.id,
            "Background response record already closed or deleted; the result is not stored"
        );
        return Ok(());
    }

    let response_json = match to_value(response) {
        Ok(json) => json,
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
    // A background response's terminal write is retried at the status-watch
    // cadence within the terminal write window while the store answers with
    // an error: the result of a run that outlived a store blip is stored once
    // the store is back. Only the write of the response record is retried:
    // once it landed, the conversation linking runs once (as before), so a
    // retry can neither store the record twice nor link the items twice. A
    // foreground response is written once, as before: its client is waiting
    // for the answer. A background stream attached through the Harmony paths
    // receives its terminal event after this persist, so during a store
    // outage that event can wait for up to the window; the regular streaming
    // paths emit it first.
    let every = background::STATUS_WATCH;
    let deadline = tokio::time::Instant::now() + background::terminal_write_window(every);
    let mut attempts = 0u32;
    loop {
        match persist_conversation_items(
            conversation_storage.clone(),
            conversation_item_storage.clone(),
            response_storage.clone(),
            &response_json,
            original_request,
            tenant,
            request_context.clone(),
        )
        .await
        {
            Ok(()) => {
                debug!("Persisted response: {}", response.id);
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
                // A cancel or a delete that landed meanwhile wins: a closed
                // record is not rewritten by the result that waited for the
                // store, a deleted one not re-created.
                if background::is_closed(&response_storage, &response.id, request_context.clone())
                    .await
                {
                    debug!(
                        response_id = %response.id,
                        "Background response record was closed or deleted while its result waited for the store; the result is not stored"
                    );
                    return Ok(());
                }
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
#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use openai_protocol::{
        model_card::ModelCard, responses::ResponseStatus, worker::HealthCheckConfig,
    };
    use serde_json::json;
    use smg_data_connector::{
        ListParams, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage, NewConversation, ResponseId, ResponseStorageError, SortOrder,
        StoredResponse,
    };

    use super::*;
    use crate::worker::{BasicWorkerBuilder, UNKNOWN_MODEL_ID};

    /// The terminal object of `request` with one output message.
    fn completed_response(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
        let message = serde_json::from_value(json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }))
        .unwrap();
        ResponsesResponse::builder(id, &request.model)
            .copy_from_request(request)
            .created_at(chrono::Utc::now().timestamp())
            .completed_at(chrono::Utc::now().timestamp())
            .status(ResponseStatus::Completed)
            .add_output(message)
            .build()
    }

    /// A chain ending in a record its client asked not to store is no
    /// previous response (the loaders of both model families call this; the
    /// item-reference resolver shares its predicate); a stored chain is
    /// returned, an unknown id refused the same way.
    #[tokio::test]
    async fn an_unstored_record_is_no_previous_response() {
        let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
        for (id, store) in [("resp_unstored", false), ("resp_stored", true)] {
            let mut response = StoredResponse::new(None);
            response.id = ResponseId::from(id);
            response.raw_response =
                json!({"id": id, "store": store, "status": "completed", "output": []});
            storage.store_response(response).await.unwrap();
        }

        let refused = previous_response_chain(&storage, "resp_unstored", None)
            .await
            .expect_err("an unstored record is no previous response");
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let chain = previous_response_chain(&storage, "resp_stored", None)
            .await
            .expect("a stored record is a previous response");
        assert_eq!(chain.len(), 1);
        let unknown = previous_response_chain(&storage, "resp_unknown", None)
            .await
            .expect_err("an unknown id is refused");
        assert_eq!(unknown.status(), StatusCode::BAD_REQUEST);
    }

    /// `background: true` with `store: false`, the public API's probe: the
    /// result is written over the record the client polls and nothing is
    /// linked to the conversation; a foreground `store: false` response
    /// stores nothing at all.
    #[tokio::test]
    async fn an_unstored_background_response_finishes_in_its_record_and_the_history_keeps_nothing()
    {
        let conversation_storage: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let item_storage: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let response_storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
        let conversation = conversation_storage
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "background": true,
            "store": false,
            "conversation": conversation.id.0,
            "max_output_tokens": 64
        }))
        .unwrap();
        background::store_record(
            &response_storage,
            &background::queued_response("resp_u", &request),
            &request,
            None,
            None,
        )
        .await
        .unwrap();

        persist_response_if_needed(
            conversation_storage.clone(),
            item_storage.clone(),
            response_storage.clone(),
            &completed_response("resp_u", &request),
            &request,
            None,
            None,
        )
        .await
        .expect("nothing is reported to the caller here");

        let stored = response_storage
            .get_response(&ResponseId::from("resp_u"))
            .await
            .unwrap()
            .expect("the record is polled");
        assert_eq!(
            stored.raw_response["status"], "completed",
            "{}",
            stored.raw_response
        );
        assert_eq!(stored.raw_response["store"], false);
        assert_eq!(
            stored.raw_response["output"][0]["content"][0]["text"],
            "hello"
        );
        let items = item_storage
            .list_items(
                &conversation.id,
                ListParams {
                    limit: 10,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(
            items.is_empty(),
            "nothing is linked to the conversation: {items:?}"
        );

        let mut foreground = request;
        foreground.background = None;
        persist_response_if_needed(
            conversation_storage,
            item_storage,
            response_storage.clone(),
            &completed_response("resp_f", &foreground),
            &foreground,
            None,
            None,
        )
        .await
        .expect("nothing is reported to the caller here");
        assert!(response_storage
            .get_response(&ResponseId::from("resp_f"))
            .await
            .unwrap()
            .is_none());
    }

    /// A response store that is down: every call answers a storage error.
    struct DownResponseStorage;

    #[async_trait::async_trait]
    impl ResponseStorage for DownResponseStorage {
        async fn store_response(
            &self,
            _response: StoredResponse,
        ) -> Result<ResponseId, ResponseStorageError> {
            Err(ResponseStorageError::StorageError(
                "connection refused".to_string(),
            ))
        }

        async fn get_response(
            &self,
            _response_id: &ResponseId,
        ) -> Result<Option<StoredResponse>, ResponseStorageError> {
            Err(ResponseStorageError::StorageError(
                "connection refused".to_string(),
            ))
        }

        async fn delete_response(
            &self,
            _response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
            Err(ResponseStorageError::StorageError(
                "connection refused".to_string(),
            ))
        }

        async fn list_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
            _limit: Option<usize>,
        ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
            Err(ResponseStorageError::StorageError(
                "connection refused".to_string(),
            ))
        }

        async fn delete_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
        ) -> Result<usize, ResponseStorageError> {
            Err(ResponseStorageError::StorageError(
                "connection refused".to_string(),
            ))
        }
    }

    /// A foreground response the client expects stored whose record could
    /// not be written is reported to the caller with the store-class code; an
    /// unstored request has nothing to report.
    #[tokio::test]
    async fn a_foreground_stored_response_whose_record_was_not_written_is_reported() {
        let conversation_storage: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let item_storage: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let down: Arc<dyn ResponseStorage> = Arc::new(DownResponseStorage);
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "max_output_tokens": 64
        }))
        .unwrap();
        assert!(stored_foreground(&request));

        let not_stored = persist_response_if_needed(
            conversation_storage.clone(),
            item_storage.clone(),
            down.clone(),
            &completed_response("resp_s", &request),
            &request,
            None,
            None,
        )
        .await
        .expect_err("the record was not written");
        assert!(not_stored.retryable, "{not_stored:?}");
        assert_eq!(not_stored.code(), "storage_unavailable");
        assert!(
            not_stored.message.contains("connection refused"),
            "{}",
            not_stored.message
        );
        let error = not_stored.error_object();
        assert_eq!(error["code"], "storage_unavailable");
        let answer = not_stored.into_response();
        assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);

        let unstorable = NotStored::new(false, "no output");
        assert_eq!(unstorable.code(), "store_response_failed");
        assert_eq!(
            unstorable.into_response().status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );

        let mut unstored = request.clone();
        unstored.store = Some(false);
        assert!(!stored_foreground(&unstored));
        persist_response_if_needed(
            conversation_storage,
            item_storage,
            down,
            &completed_response("resp_n", &unstored),
            &unstored,
            None,
            None,
        )
        .await
        .expect("nothing to store, nothing to report");
    }

    /// A background response deleted while its work ran (its record is gone
    /// when the result arrives): the result is not stored under the id again
    /// and nothing of it reaches the conversation; the delete holds.
    #[tokio::test]
    async fn a_background_result_whose_record_was_deleted_is_not_stored() {
        let conversation_storage: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let item_storage: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let response_storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
        let conversation = conversation_storage
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "background": true,
            "conversation": conversation.id.0,
            "max_output_tokens": 64
        }))
        .unwrap();
        background::store_record(
            &response_storage,
            &background::queued_response("resp_d", &request),
            &request,
            None,
            None,
        )
        .await
        .unwrap();
        response_storage
            .delete_response(&ResponseId::from("resp_d"))
            .await
            .unwrap();

        persist_response_if_needed(
            conversation_storage,
            item_storage.clone(),
            response_storage.clone(),
            &completed_response("resp_d", &request),
            &request,
            None,
            None,
        )
        .await
        .expect("a deleted record is not a store failure");

        assert!(
            response_storage
                .get_response(&ResponseId::from("resp_d"))
                .await
                .unwrap()
                .is_none(),
            "the deleted record is not re-created by the result"
        );
        let items = item_storage
            .list_items(
                &conversation.id,
                ListParams {
                    limit: 10,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(
            items.is_empty(),
            "nothing is linked to the conversation: {items:?}"
        );
    }

    fn registry_with_aliased_worker() -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .model(ModelCard::new("canonical-model").with_alias("model-alias"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register_or_replace(Arc::new(worker));
        registry
    }

    #[test]
    fn worker_availability_accepts_alias_and_preserves_unknown_rejection() {
        let registry = registry_with_aliased_worker();

        assert!(validate_worker_availability(&registry, "canonical-model").is_none());
        assert!(validate_worker_availability(&registry, "model-alias").is_none());

        let response = validate_worker_availability(&registry, UNKNOWN_MODEL_ID)
            .expect("unknown model should remain rejected for Responses");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn worker_availability_rejects_alias_once_its_worker_is_gone() {
        let registry = registry_with_aliased_worker();
        let worker_id = registry.get_id_by_url("http://worker:8080").unwrap();
        assert!(registry.remove(&worker_id).is_some());

        let response = validate_worker_availability(&registry, "model-alias")
            .expect("alias must stop resolving with no workers behind it");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
    #[test]
    fn namespace_function_identity_roundtrips_and_preserves_literal_names() {
        let mut request: ResponsesRequest = namespace_test_request();
        let tools = extract_tools_from_response_tools(request.tools.as_deref());
        assert_eq!(
            tools
                .iter()
                .map(|t| t.function.name.as_str())
                .collect::<Vec<_>>(),
            vec!["weather.lookup", "travel.lookup"]
        );
        for namespace in ["weather", "travel"] {
            assert_eq!(
                resolve_function_identity(request.tools.as_deref(), &format!("{namespace}.lookup")),
                ("lookup".into(), Some(namespace.into()))
            );
        }
        for name in ["lookup", "unknown.lookup"] {
            assert_eq!(
                resolve_function_identity(request.tools.as_deref(), name),
                (name.into(), None)
            );
        }
        request.tools.as_mut().unwrap().push(
            serde_json::from_value(
                serde_json::json!({"type":"function","name":"weather.lookup","parameters":{}}),
            )
            .unwrap(),
        );
        assert_eq!(
            resolve_function_identity(request.tools.as_deref(), "weather.lookup"),
            ("weather.lookup".into(), None)
        );
    }
}
