//! Streaming infrastructure for /v1/responses endpoint

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use http::header::{HeaderValue, CONTENT_TYPE};
use openai_protocol::{
    chat::ChatCompletionStreamResponse,
    common::{Usage, UsageInfo},
    event_types::{
        ContentPartEvent, FunctionCallEvent, McpEvent, OutputItemEvent, OutputTextEvent,
        ResponseEvent,
    },
    responses::{
        IncludeField, ResponseOutputItem, ResponseStatus, ResponsesRequest, ResponsesResponse,
        ResponsesUsage,
    },
};
use serde_json::json;
use smg_mcp::{self as mcp};
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;
use uuid::Uuid;

use crate::routers::{
    common::{
        openai_bridge::{self, descriptor, ResponseFormat},
        sse::{SseReceiver, SseSender},
    },
    grpc::harmony::responses::ToolResult,
};

/// Item-id-prefix discriminator for non-format kinds. Format-driven items
/// derive their prefix from `descriptor(format).id_prefix` instead — keeping
/// the wire-shape mapping in one place.
pub(crate) enum OutputItemKind {
    Message,
    McpListTools,
    FunctionCall,
    Reasoning,
}

impl OutputItemKind {
    fn id_prefix(&self) -> &'static str {
        match self {
            Self::Message => "msg",
            Self::McpListTools => "mcpl",
            Self::FunctionCall => "fc",
            Self::Reasoning => "rs",
        }
    }
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
    item_data: Option<serde_json::Value>,
}

/// Streaming state for one in-flight chat tool-call being translated into a
/// Responses `function_call` output item.
struct ToolCallStreamItem {
    /// `delta.tool_calls[].index` from the chat stream.
    chat_index: u32,
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    added_emitted: bool,
}

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.output_item.done
/// - response.completed
/// - response.content_part.added
/// - response.content_part.done
/// - response.output_text.delta
/// - response.output_text.done
/// - response.mcp_list_tools.in_progress
/// - response.mcp_list_tools.completed
/// - response.mcp_call.in_progress
/// - response.mcp_call_arguments.delta
/// - response.mcp_call_arguments.done
/// - response.mcp_call.completed
/// - response.mcp_call.failed
/// - response.web_search_call.in_progress
/// - response.web_search_call.searching
/// - response.web_search_call.completed
/// - response.function_call_arguments.delta
/// - response.function_call_arguments.done
pub(crate) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    pub response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    accumulated_text: String,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    // Output item tracking
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>,
    current_item_id: Option<String>,
    original_request: Option<ResponsesRequest>,
    tool_call_items: Vec<ToolCallStreamItem>,
    /// In-flight reasoning item: opened on the first reasoning delta, closed
    /// before the first message/tool item or on finish.
    reasoning_item: Option<ReasoningStreamItem>,
}

/// Streaming state for the reasoning output item of the current turn.
struct ReasoningStreamItem {
    output_index: usize,
    item_id: String,
    text: String,
}

impl ResponseStreamEventEmitter {
    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
        let message_id = format!("msg_{}", Uuid::now_v7());

        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            message_id,
            accumulated_text: String::new(),
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
            original_request: None,
            tool_call_items: Vec::new(),
            reasoning_item: None,
        }
    }

    /// Set the original request for including all fields in response.completed
    pub fn set_original_request(&mut self, request: ResponsesRequest) {
        self.original_request = Some(request);
    }

    /// Update tool call output items with tool execution results.
    ///
    /// Replaces each matched item's stored payload with the authoritative
    /// `ResponseOutputItem` produced by `ResponseTransformer::transform`
    /// (carried on `ToolResult::output_item`). That transformer is the
    /// single source of truth for per-tool shape — e.g. `mcp_call`
    /// receives `output: string`, `web_search_call` receives
    /// `{status, action, ...}`, and `image_generation_call` receives
    /// `result: base64`. The streaming emitter's initial
    /// `output_item.added` carries a partial stub (tool-call id + arguments);
    /// this call overwrites that stub once the MCP result is in hand, so
    /// `response.completed.response.output` sees the same item a
    /// non-streaming response would emit.
    ///
    /// Matching is by the original `call_id` the stub stored, since that
    /// is the only identifier common to every tool-call item type.
    pub(crate) fn update_mcp_call_outputs(&mut self, tool_results: &[ToolResult]) {
        for tool_result in tool_results {
            let Ok(item_value) = serde_json::to_value(&tool_result.output_item) else {
                warn!(
                    call_id = %tool_result.call_id,
                    "Failed to serialize transformed tool output item; keeping streaming stub"
                );
                continue;
            };
            for item_state in &mut self.output_items {
                if let Some(ref mut item_data) = item_state.item_data {
                    if item_data.get("call_id").and_then(|c| c.as_str())
                        == Some(&tool_result.call_id)
                    {
                        *item_data = item_value;
                        break;
                    }
                }
            }
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    pub fn emit_created(&mut self) -> serde_json::Value {
        json!({
            "type": ResponseEvent::CREATED,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": []
            }
        })
    }

    pub fn emit_in_progress(&mut self) -> serde_json::Value {
        json!({
            "type": ResponseEvent::IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    pub fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": ContentPartEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "output_text",
                "text": ""
            }
        })
    }

    pub fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": OutputTextEvent::DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta
        })
    }

    pub fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": OutputTextEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    pub fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": ContentPartEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "output_text",
                "text": self.accumulated_text.clone()
            }
        })
    }

    // INVARIANT: this method is terminal — it drains internal state via `take()`
    // and must only be called once per emitter lifetime.
    pub fn emit_completed(&mut self, usage: Option<&serde_json::Value>) -> serde_json::Value {
        // Build output array from tracked items
        let output: Vec<serde_json::Value> = self
            .output_items
            .iter_mut()
            .filter_map(|item| {
                if item.status == ItemStatus::Completed {
                    item.item_data.take()
                } else {
                    None
                }
            })
            .collect();

        // If no items were tracked (legacy path), fall back to generic message
        let output = if output.is_empty() {
            vec![json!({
                "id": std::mem::take(&mut self.message_id),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": std::mem::take(&mut self.accumulated_text)
                }]
            })]
        } else {
            output
        };

        // Build base response object
        let mut response_obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output
        });

        // Add usage if provided
        if let Some(usage_val) = usage {
            response_obj["usage"] = usage_val.clone();
        }

        // Add all original request fields if available
        if let Some(ref req) = self.original_request {
            Self::add_optional_field(&mut response_obj, "instructions", req.instructions.as_ref());
            Self::add_optional_field(
                &mut response_obj,
                "max_output_tokens",
                req.max_output_tokens.as_ref(),
            );
            Self::add_optional_field(
                &mut response_obj,
                "max_tool_calls",
                req.max_tool_calls.as_ref(),
            );
            Self::add_optional_field(
                &mut response_obj,
                "previous_response_id",
                req.previous_response_id.as_ref(),
            );
            Self::add_optional_field(&mut response_obj, "reasoning", req.reasoning.as_ref());
            Self::add_optional_field(&mut response_obj, "temperature", req.temperature.as_ref());
            Self::add_optional_field(&mut response_obj, "top_p", req.top_p.as_ref());
            Self::add_optional_field(&mut response_obj, "truncation", req.truncation.as_ref());
            Self::add_optional_field(&mut response_obj, "user", req.user.as_ref());

            response_obj["parallel_tool_calls"] = json!(req.parallel_tool_calls.unwrap_or(true));
            response_obj["store"] = json!(req.store.unwrap_or(true));
            let empty_tools = vec![];
            let empty_metadata = Default::default();
            response_obj["tools"] = json!(req.tools.as_ref().unwrap_or(&empty_tools));
            response_obj["metadata"] = json!(req.metadata.as_ref().unwrap_or(&empty_metadata));

            // tool_choice: serialize if present, otherwise use "auto"
            if let Some(ref tc) = req.tool_choice {
                response_obj["tool_choice"] = json!(tc);
            } else {
                response_obj["tool_choice"] = json!("auto");
            }
        }

        json!({
            "type": ResponseEvent::COMPLETED,
            "sequence_number": self.next_sequence(),
            "response": response_obj
        })
    }

    /// Convert tool entries to JSON values using the shared bridge builder.
    fn tool_entries_to_json(
        tools: &[mcp::ToolEntry],
    ) -> Result<Vec<serde_json::Value>, serde_json::Error> {
        openai_bridge::build_mcp_tool_infos(tools)
            .into_iter()
            .map(serde_json::to_value)
            .collect()
    }

    /// Helper to add optional fields to JSON object
    fn add_optional_field<T: serde::Serialize>(
        obj: &mut serde_json::Value,
        key: &str,
        value: Option<&T>,
    ) {
        if let Some(val) = value {
            obj[key] = json!(val);
        }
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    pub fn emit_mcp_list_tools_in_progress(&mut self, output_index: usize) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "output_index": output_index
        })
    }

    pub fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        tool_items: &[serde_json::Value],
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_COMPLETED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "tools": tool_items
        })
    }

    pub fn emit_mcp_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_FAILED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Generic Tool Call Event Emission (based on ResponseFormat)
    // ========================================================================

    /// Emit a tool call event with the specified event type.
    /// This is the internal helper used by all tool call event methods.
    fn emit_tool_event(
        &mut self,
        event_type: &'static str,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": event_type,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_tool_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> serde_json::Value {
        let event_type = descriptor(response_format).in_progress_event;
        self.emit_tool_event(event_type, output_index, item_id)
    }

    /// Emit the searching/interpreting/generating event; `None` for formats
    /// with no intermediate phase.
    pub fn emit_tool_call_searching(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> Option<serde_json::Value> {
        let event_type = descriptor(response_format).searching_event?;
        Some(self.emit_tool_event(event_type, output_index, item_id))
    }

    /// Emit a `response.image_generation_call.partial_image` event. Returns
    /// `None` for formats with no partial-image frame.
    #[expect(
        dead_code,
        reason = "partial_image emission is wired by per-router integrations"
    )]
    pub fn emit_image_generation_partial_image(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
        partial_image_index: u32,
        partial_image_b64: &str,
    ) -> Option<serde_json::Value> {
        let event_type = descriptor(response_format).partial_image_event?;
        Some(json!({
            "type": event_type,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "partial_image_index": partial_image_index,
            "partial_image_b64": partial_image_b64
        }))
    }

    pub fn emit_tool_call_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> serde_json::Value {
        let event_type = descriptor(response_format).completed_event;
        self.emit_tool_event(event_type, output_index, item_id)
    }

    pub fn type_str_for_format(response_format: Option<&ResponseFormat>) -> &'static str {
        match response_format {
            Some(format) => descriptor(*format).type_str,
            None => "function_call",
        }
    }

    // ========================================================================
    // Function Call Event Emission Methods
    // ========================================================================

    pub fn emit_function_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub fn emit_function_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Restore declared namespace identities for streamed function items.
    fn normalize_function_item<'a>(
        &self,
        item: &'a serde_json::Value,
    ) -> std::borrow::Cow<'a, serde_json::Value> {
        if item["type"] == "function_call" && item["namespace"].is_null() {
            if let Some(name) = item["name"].as_str() {
                let tools = self
                    .original_request
                    .as_ref()
                    .and_then(|req| req.tools.as_deref());
                let (name, namespace) = super::utils::resolve_function_identity(tools, name);
                if let Some(namespace) = namespace {
                    let mut item = item.clone();
                    item["name"] = json!(name);
                    item["namespace"] = json!(namespace);
                    return std::borrow::Cow::Owned(item);
                }
            }
        }
        std::borrow::Cow::Borrowed(item)
    }

    /// Emit response.output_item.added event
    pub fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        let item = self.normalize_function_item(item);
        json!({
            "type": OutputItemEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Emit response.output_item.done event
    pub fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        let item = self.normalize_function_item(item).into_owned();
        // Store the item data for later use in emit_completed
        self.store_output_item_data(output_index, item.clone());

        json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::now_v7().simple())
    }

    /// Allocate next output index and track item, deriving the item-id from
    /// `id_prefix` (e.g. `"msg"`, `"mcp"`, `"ws"` — without trailing `_`).
    pub fn allocate_output_index_with_prefix(&mut self, id_prefix: &str) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
            item_data: None,
        });

        (index, id)
    }

    /// Convenience: allocate for a non-format kind.
    pub fn allocate_output_index(&mut self, kind: OutputItemKind) -> (usize, String) {
        self.allocate_output_index_with_prefix(kind.id_prefix())
    }

    /// Convenience: allocate for a `ResponseFormat`-driven item, falling back
    /// to `function_call` (`"fc"`) when no format applies. Mirrors the prior
    /// `output_item_type_for_format` mapping but reads the prefix straight
    /// off the format descriptor.
    pub fn allocate_output_index_for_format(
        &mut self,
        response_format: Option<ResponseFormat>,
    ) -> (usize, String) {
        let prefix = response_format
            .map(|f| descriptor(f).id_prefix)
            .unwrap_or("fc");
        self.allocate_output_index_with_prefix(prefix)
    }

    /// Mark output item as completed and store its data
    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Store output item data when emitting output_item.done
    pub fn store_output_item_data(&mut self, output_index: usize, item_data: serde_json::Value) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.item_data = Some(item_data);
        }
    }

    /// Finalize and return the complete ResponsesResponse
    ///
    /// This constructs the final ResponsesResponse from all accumulated output items
    /// for persistence. Should be called after streaming is complete.
    /// Reads non-destructively so `emit_completed()` can still drain state afterwards.
    pub fn finalize(&self, usage: Option<Usage>) -> ResponsesResponse {
        // Build output array from tracked items (clone — emit_completed drains later)
        let output: Vec<ResponseOutputItem> = self
            .output_items
            .iter()
            .filter_map(|item| {
                item.item_data
                    .as_ref()
                    .and_then(|data| serde_json::from_value(data.clone()).ok())
            })
            .collect();

        // Convert Usage to ResponsesUsage
        let responses_usage = usage.map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: u.prompt_tokens_details.clone(),
            };
            ResponsesUsage::Modern(usage_info.to_response_usage())
        });

        // Build response using builder
        ResponsesResponse::builder(&self.response_id, &self.model)
            .created_at(self.created_at as i64)
            .status(ResponseStatus::Completed)
            .output(output)
            .maybe_copy_from_request(self.original_request.as_ref())
            .maybe_usage(responses_usage)
            .build()
    }

    /// Emit reasoning item wrapper events (added + done)
    ///
    /// Reasoning items in OpenAI format are simple placeholders emitted between tool iterations.
    /// They don't have streaming content - just wrapper events with empty/null content.
    pub async fn emit_reasoning_item(
        &mut self,
        tx: &SseSender,
        reasoning_content: Option<String>,
    ) -> Result<(), String> {
        // Allocate output index and generate ID
        let (output_index, item_id) = self.allocate_output_index(OutputItemKind::Reasoning);

        // Build reasoning item structure. `content` is an array of typed
        // parts on the wire (a bare string breaks clients walking the
        // content list).
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": reasoning_content
                .map(|text| json!([{ "type": "reasoning_text", "text": text }]))
                .unwrap_or(json!([])),
            "encrypted_content": null,
            "status": null
        });

        // Emit output_item.added
        let added_event = self.emit_output_item_added(output_index, &item);
        self.send_event(&added_event, tx).await?;

        // Immediately emit output_item.done (no streaming for reasoning)
        let done_event = self.emit_output_item_done(output_index, &item);
        self.send_event(&done_event, tx).await?;

        // Mark as completed
        self.complete_output_item(output_index);

        Ok(())
    }

    fn include_encrypted_reasoning(&self) -> bool {
        self.original_request
            .as_ref()
            .and_then(|r| r.include.as_deref())
            .is_some_and(|f| f.contains(&IncludeField::ReasoningEncryptedContent))
    }

    fn is_custom_tool(&self, name: &str) -> bool {
        let tools = self
            .original_request
            .as_ref()
            .and_then(|r| r.tools.as_deref());
        super::utils::custom_tool_names(tools).contains(name)
    }

    fn emit_reasoning_text_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.reasoning_text.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": 0,
            "delta": delta
        })
    }

    fn emit_reasoning_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        text: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.reasoning_text.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": 0,
            "text": text
        })
    }

    /// Close the in-flight reasoning item, if any: `reasoning_text.done`
    /// followed by `output_item.done` carrying the full text.
    async fn close_reasoning_item(&mut self, tx: &SseSender) -> Result<(), String> {
        let Some(reasoning) = self.reasoning_item.take() else {
            return Ok(());
        };
        let event = self.emit_reasoning_text_done(
            reasoning.output_index,
            &reasoning.item_id,
            &reasoning.text,
        );
        self.send_event(&event, tx).await?;

        let mut item = json!({
            "id": reasoning.item_id,
            "type": "reasoning",
            "summary": [],
            "content": [{ "type": "reasoning_text", "text": reasoning.text }],
            "status": "completed"
        });
        if self.include_encrypted_reasoning() {
            item["encrypted_content"] =
                json!(super::utils::encode_reasoning_content(&reasoning.text));
        }
        let event = self.emit_output_item_done(reasoning.output_index, &item);
        self.send_event(&event, tx).await?;
        self.complete_output_item(reasoning.output_index);
        Ok(())
    }

    /// Process a chunk and emit appropriate events
    pub async fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &SseSender,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            // Reasoning streams as its own output item, opened on the first
            // delta so it precedes the answer (OpenAI order).
            if let Some(reasoning) = choice
                .delta
                .reasoning_content
                .as_deref()
                .filter(|r| !r.is_empty())
            {
                if self.reasoning_item.is_none() {
                    let (output_index, item_id) =
                        self.allocate_output_index(OutputItemKind::Reasoning);
                    let item = json!({
                        "id": item_id,
                        "type": "reasoning",
                        "summary": [],
                        "content": [],
                        "status": "in_progress"
                    });
                    let event = self.emit_output_item_added(output_index, &item);
                    self.send_event(&event, tx).await?;
                    self.reasoning_item = Some(ReasoningStreamItem {
                        output_index,
                        item_id,
                        text: String::new(),
                    });
                }
                if let Some(item) = self.reasoning_item.as_mut() {
                    item.text.push_str(reasoning);
                    let (output_index, item_id) = (item.output_index, item.item_id.clone());
                    let event = self.emit_reasoning_text_delta(output_index, &item_id, reasoning);
                    self.send_event(&event, tx).await?;
                }
            }
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
                        self.close_reasoning_item(tx).await?;
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemKind::Message);

                        // Build message item structure
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        // Emit output_item.added
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx).await?;
                        self.has_emitted_output_item_added = true;

                        // Store for subsequent events
                        self.current_item_id = Some(item_id);
                        self.current_message_output_index = Some(output_index);
                    }

                    // output_index and item_id are always set in the block above
                    // when current_item_id was None and we allocated new ones
                    if let (Some(output_index), Some(item_id)) = (
                        self.current_message_output_index,
                        self.current_item_id.clone(),
                    ) {
                        let content_index = 0; // Single content part for now

                        // Emit content_part.added before first delta
                        if !self.has_emitted_content_part_added {
                            let event =
                                self.emit_content_part_added(output_index, &item_id, content_index);
                            self.send_event(&event, tx).await?;
                            self.has_emitted_content_part_added = true;
                        }

                        // Emit text delta
                        let event =
                            self.emit_text_delta(content, output_index, &item_id, content_index);
                        self.send_event(&event, tx).await?;
                    }
                }
            }

            // Translate chat tool-call deltas into Responses function_call
            // events; without this the streamed call is dropped entirely.
            if let Some(tool_calls) = &choice.delta.tool_calls {
                for tc in tool_calls {
                    let pos = match self
                        .tool_call_items
                        .iter()
                        .position(|i| i.chat_index == tc.index)
                    {
                        Some(pos) => pos,
                        None => {
                            self.close_reasoning_item(tx).await?;
                            let (output_index, item_id) =
                                self.allocate_output_index(OutputItemKind::FunctionCall);
                            self.tool_call_items.push(ToolCallStreamItem {
                                chat_index: tc.index,
                                output_index,
                                item_id,
                                call_id: tc
                                    .id
                                    .clone()
                                    .unwrap_or_else(|| format!("call_{}", Uuid::now_v7())),
                                name: String::new(),
                                arguments: String::new(),
                                added_emitted: false,
                            });
                            self.tool_call_items.len() - 1
                        }
                    };
                    let function = tc.function.as_ref();
                    let name_delta = function
                        .and_then(|f| f.name.as_deref())
                        .filter(|n| !n.is_empty());
                    let args_delta = function
                        .and_then(|f| f.arguments.as_deref())
                        .filter(|a| !a.is_empty());

                    let item = &mut self.tool_call_items[pos];
                    if let (true, Some(name)) = (item.name.is_empty(), name_delta) {
                        item.name = name.to_string();
                    }
                    if let Some(args) = args_delta {
                        item.arguments.push_str(args);
                    }
                    let emit_added = !item.added_emitted
                        && (!item.name.is_empty() || !item.arguments.is_empty());
                    item.added_emitted |= emit_added;
                    let (output_index, item_id, call_id, name) = (
                        item.output_index,
                        item.item_id.clone(),
                        item.call_id.clone(),
                        item.name.clone(),
                    );
                    let custom = self.is_custom_tool(&name);

                    if emit_added {
                        let item = if custom {
                            json!({
                                "id": item_id,
                                "type": "custom_tool_call",
                                "call_id": call_id,
                                "name": name,
                                "input": "",
                            })
                        } else {
                            json!({
                                "id": item_id,
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": "",
                                "status": "in_progress",
                            })
                        };
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx).await?;
                    }
                    // Custom tool calls carry their payload as the raw `input`
                    // string of the final item, not as streamed JSON arguments.
                    if let Some(delta) = args_delta.filter(|_| !custom) {
                        let event =
                            self.emit_function_call_arguments_delta(output_index, &item_id, delta);
                        self.send_event(&event, tx).await?;
                    }
                }
            }

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                self.close_reasoning_item(tx).await?;
                if reason == "stop" || reason == "length" || reason == "tool_calls" {
                    if let (Some(output_index), Some(item_id)) = (
                        self.current_message_output_index,
                        self.current_item_id.clone(),
                    ) {
                        let content_index = 0;

                        // Emit closing events
                        if self.has_emitted_content_part_added {
                            let event = self.emit_text_done(output_index, &item_id, content_index);
                            self.send_event(&event, tx).await?;
                            let event =
                                self.emit_content_part_done(output_index, &item_id, content_index);
                            self.send_event(&event, tx).await?;
                        }

                        if self.has_emitted_output_item_added {
                            // Build complete message item for output_item.done
                            let item = json!({
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": std::mem::take(&mut self.accumulated_text)
                                }]
                            });
                            let event = self.emit_output_item_done(output_index, &item);
                            self.send_event(&event, tx).await?;
                        }

                        // Mark item as completed
                        self.complete_output_item(output_index);
                    }
                }
                if reason == "tool_calls" {
                    // Close every streamed tool-call item: arguments (or custom
                    // input) done, then output_item.done, as the non-streaming
                    // path pairs them.
                    for item in std::mem::take(&mut self.tool_call_items) {
                        let full = if self.is_custom_tool(&item.name) {
                            let input = super::utils::custom_tool_input(&item.arguments);
                            for (event_type, field) in [
                                ("response.custom_tool_call_input.delta", "delta"),
                                ("response.custom_tool_call_input.done", "input"),
                            ] {
                                let event = json!({
                                    "type": event_type,
                                    "sequence_number": self.next_sequence(),
                                    "output_index": item.output_index,
                                    "item_id": item.item_id,
                                    field: input,
                                });
                                self.send_event(&event, tx).await?;
                            }
                            json!({
                                "id": item.item_id,
                                "type": "custom_tool_call",
                                "call_id": item.call_id,
                                "name": item.name,
                                "input": input,
                            })
                        } else {
                            let event = self.emit_function_call_arguments_done(
                                item.output_index,
                                &item.item_id,
                                &item.arguments,
                            );
                            self.send_event(&event, tx).await?;
                            json!({
                                "id": item.item_id,
                                "type": "function_call",
                                "call_id": item.call_id,
                                "name": item.name,
                                "arguments": item.arguments,
                                "status": "completed",
                            })
                        };
                        let event = self.emit_output_item_done(item.output_index, &full);
                        self.send_event(&event, tx).await?;
                        self.complete_output_item(item.output_index);
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn send_event(
        &self,
        event: &serde_json::Value,
        tx: &SseSender,
    ) -> Result<(), String> {
        let event_json =
            serde_json::to_string(event).map_err(|e| format!("Failed to serialize event: {e}"))?;

        // Extract event type from the JSON for SSE event field
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        // Format as SSE with event: field
        let sse_message = format!("event: {event_type}\ndata: {event_json}\n\n");

        if tx.send(Ok(Bytes::from(sse_message))).await.is_err() {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }

    /// Send event and log any errors (typically client disconnect)
    ///
    /// This is a convenience method for streaming scenarios where client
    /// disconnection is expected and should be logged but not fail the operation.
    /// Returns true if sent successfully, false if client disconnected.
    pub async fn send_event_best_effort(&self, event: &serde_json::Value, tx: &SseSender) -> bool {
        match self.send_event(event, tx).await {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// Emit an error event
    ///
    /// Creates and sends an error event with the given error message.
    /// Uses OpenAI's error event format.
    /// Use this for terminal errors that should abort the streaming response.
    pub async fn emit_error(&mut self, error_msg: &str, error_code: Option<&str>, tx: &SseSender) {
        let event = json!({
            "type": "error",
            "code": error_code.unwrap_or("internal_error"),
            "message": error_msg,
            "param": null,
            "sequence_number": self.next_sequence()
        });
        let sse_data = match serde_json::to_string(&event) {
            Ok(json) => format!("data: {json}\n\n"),
            Err(_) => "data: {\"type\":\"error\",\"code\":\"internal_error\",\"message\":\"serialization failed\",\"param\":null}\n\n".to_string(),
        };
        let _ = tx.send(Ok(Bytes::from(sse_data))).await;
    }

    /// Emit the full mcp_list_tools output-item sequence.
    ///
    /// Allocates an output index, builds the tool-list JSON, then emits the four
    /// standard events (output_item.added, mcp_list_tools.in_progress,
    /// mcp_list_tools.completed, output_item.done) and marks the item complete.
    ///
    /// `server_label` is taken as an explicit parameter so callers can iterate
    /// over multiple MCP servers without mutating the emitter's own label.
    pub async fn emit_mcp_list_tools_sequence(
        &mut self,
        server_label: &str,
        tools: &[mcp::ToolEntry],
        tx: &SseSender,
    ) -> Result<(), String> {
        let (output_index, item_id) = self.allocate_output_index(OutputItemKind::McpListTools);

        // Build per-tool JSON items
        let tool_items = Self::tool_entries_to_json(tools).unwrap_or_else(|e| {
            warn!("Failed to serialize McpToolInfo to JSON: {e}");
            Vec::new()
        });

        // In-progress item (empty tools)
        let item_in_progress = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "in_progress",
            "tools": []
        });

        // Emit output_item.added
        let event = self.emit_output_item_added(output_index, &item_in_progress);
        self.send_event(&event, tx).await?;

        // Emit mcp_list_tools.in_progress
        let event = self.emit_mcp_list_tools_in_progress(output_index);
        self.send_event(&event, tx).await?;

        // Emit mcp_list_tools.completed
        let event = self.emit_mcp_list_tools_completed(output_index, &tool_items);
        self.send_event(&event, tx).await?;

        // Completed item (with tools populated)
        let item_done = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "completed",
            "tools": tool_items
        });

        // Emit output_item.done (also stores item data internally)
        let event = self.emit_output_item_done(output_index, &item_done);
        self.send_event(&event, tx).await?;

        self.complete_output_item(output_index);

        Ok(())
    }
}

/// Build a Server-Sent Events (SSE) response
///
/// Creates a Response with proper SSE headers and streaming body.
#[expect(
    clippy::expect_used,
    reason = "Response::builder with static headers and valid status code is infallible"
)]
pub(crate) fn build_sse_response(rx: SseReceiver) -> Response {
    let stream = ReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        )
        .header("Cache-Control", HeaderValue::from_static("no-cache"))
        .header("Connection", HeaderValue::from_static("keep-alive"))
        .body(Body::from_stream(stream))
        .expect("infallible: static headers and valid status code")
}

/// Attach `server_label` to an MCP tool-call JSON item.
///
/// Only sets the field when `response_format` indicates a passthrough (mcp_call)
/// type and a server label is provided, since built-in tool types
/// (web_search_call, etc.) do not carry a server label.
pub(crate) fn attach_mcp_server_label(
    item: &mut serde_json::Value,
    server_label: Option<&str>,
    response_format: Option<&ResponseFormat>,
) {
    if let (Some(label), Some(ResponseFormat::Passthrough)) = (server_label, response_format) {
        item["server_label"] = json!(label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalized_streaming_response_serializes_responses_api_usage() {
        let emitter =
            ResponseStreamEventEmitter::new("resp_test".to_string(), "test-model".to_string(), 1);

        let usage = Usage::from_counts(12, 7)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2);
        let wire = serde_json::to_value(emitter.finalize(Some(usage)))
            .expect("finalized response should serialize");
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
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use crate::routers::grpc::common::responses::utils::namespace_test_request;
    #[test]
    fn namespace_stream_events_and_completed_output_agree() {
        let request: ResponsesRequest = namespace_test_request();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        emitter.set_original_request(request);
        let (index, id) = emitter.allocate_output_index(OutputItemKind::FunctionCall);
        let item = json!({"id":id,"type":"function_call","call_id":"call_test","name":"weather.lookup","arguments":"{}","status":"completed"});
        for event in [
            emitter.emit_output_item_added(index, &item),
            emitter.emit_output_item_done(index, &item),
        ] {
            assert_eq!(event["item"]["name"], "lookup");
            assert_eq!(event["item"]["namespace"], "weather");
        }
        emitter.complete_output_item(index);
        let event = emitter.emit_completed(None);
        assert_eq!(event["response"]["output"][0]["name"], "lookup");
        assert_eq!(event["response"]["output"][0]["namespace"], "weather");
    }
}

#[cfg(test)]
mod process_chunk_tests {
    use tokio::sync::mpsc;

    use super::*;
    use crate::routers::grpc::common::responses::utils::decode_reasoning_content;

    fn chunk(
        delta: serde_json::Value,
        finish_reason: Option<&str>,
    ) -> ChatCompletionStreamResponse {
        serde_json::from_value(json!({
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }))
        .unwrap()
    }

    /// Drive `chunks` through a fresh emitter; return the emitted events in
    /// order plus the `output` array of `response.completed`.
    async fn stream(
        request: serde_json::Value,
        chunks: &[ChatCompletionStreamResponse],
    ) -> (Vec<serde_json::Value>, serde_json::Value) {
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        emitter.set_original_request(serde_json::from_value(request).unwrap());
        let (tx, mut rx) = mpsc::channel(256);
        for chunk in chunks {
            emitter.process_chunk(chunk, &tx).await.unwrap();
        }
        drop(tx);
        let mut events = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
            events.push(serde_json::from_str(data).unwrap());
        }
        let completed = emitter.emit_completed(None);
        (events, completed["response"]["output"].clone())
    }

    fn types(events: &[serde_json::Value]) -> Vec<&str> {
        events.iter().map(|e| e["type"].as_str().unwrap()).collect()
    }

    #[tokio::test]
    async fn reasoning_then_function_call_stream_as_paired_items() {
        let (events, output) = stream(
            json!({"model": "test-model", "input": "hi", "include": ["reasoning.encrypted_content"]}),
            &[
                chunk(json!({"reasoning_content": "thi"}), None),
                chunk(json!({"reasoning_content": "nk"}), None),
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}),
                    None,
                ),
                chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;

        assert_eq!(
            types(&events),
            [
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
            ]
        );
        let seq: Vec<u64> = events
            .iter()
            .map(|e| e["sequence_number"].as_u64().unwrap())
            .collect();
        assert!(seq.windows(2).all(|w| w[0] < w[1]));

        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(
            output[0]["content"],
            json!([{"type": "reasoning_text", "text": "think"}])
        );
        assert_eq!(
            decode_reasoning_content(output[0]["encrypted_content"].as_str().unwrap()).as_deref(),
            Some("think")
        );
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["call_id"], "call_1");
        assert_eq!(output[1]["name"], "get_weather");
        assert_eq!(output[1]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(output[1]["status"], "completed");
        // Each output_item.done carries the item response.completed reports.
        assert_eq!(events[4]["item"], output[0]);
        assert_eq!(events[9]["item"], output[1]);
        assert_eq!(events[8]["arguments"], "{\"city\":\"Paris\"}");
    }

    #[tokio::test]
    async fn custom_tool_call_streams_as_custom_input() {
        let (events, output) = stream(
            json!({"model": "test-model", "input": "hi",
                "tools": [{"type": "custom", "name": "emit_command"}]}),
            &[
                chunk(
                    json!({"tool_calls": [{"index": 0, "id": "call_c", "type": "function",
                        "function": {"name": "emit_command", "arguments": "{\"input\": \"pwd\"}"}}]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        )
        .await;

        assert_eq!(
            types(&events),
            [
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
            ]
        );
        assert_eq!(events[0]["item"]["type"], "custom_tool_call");
        assert_eq!(events[1]["delta"], "pwd");
        assert_eq!(events[2]["input"], "pwd");
        assert_eq!(output[0]["type"], "custom_tool_call");
        assert_eq!(output[0]["input"], "pwd");
        assert_eq!(output[0]["call_id"], "call_c");
        assert_eq!(events[3]["item"], output[0]);
    }

    #[tokio::test]
    async fn text_after_reasoning_closes_the_reasoning_item_first() {
        let (events, output) = stream(
            json!({"model": "test-model", "input": "hi"}),
            &[
                chunk(json!({"reasoning_content": "why"}), None),
                chunk(json!({"content": "Hel"}), None),
                chunk(json!({"content": "lo"}), None),
                chunk(json!({}), Some("stop")),
            ],
        )
        .await;

        assert_eq!(
            types(&events),
            [
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            ]
        );
        assert_eq!(output[0]["type"], "reasoning");
        assert!(output[0].get("encrypted_content").is_none());
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "Hello");
    }
}
