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
        ResponseOutputItem, ResponseStatus, ResponsesRequest, ResponsesResponse, ResponsesUsage,
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

    /// Emit response.output_item.added event
    pub fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
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

        // Build reasoning item structure
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": reasoning_content,
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

    /// Process a chunk and emit appropriate events
    pub async fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &SseSender,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
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

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                if reason == "stop" || reason == "length" {
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
