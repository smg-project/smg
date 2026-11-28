//! Streaming tool call handling for MCP interception.

use std::collections::{HashMap, HashSet};

use openai_protocol::event_types::{
    is_function_call_type, FunctionCallEvent, OutputItemEvent, ResponseEvent,
};
use serde_json::Value;
use tracing::warn;

use crate::{
    openai::responses::{extract_output_index, get_event_type, StreamingResponseAccumulator},
    openai_bridge,
};

/// True for item types whose upstream `output_item.done` must be suppressed
/// — the tool loop emits its own umbrella AFTER `<type>.completed` to
/// satisfy the spec invariant that the umbrella is the LAST event for a
/// given item (see `.claude/_audit/openai-responses-api-spec.md`
/// §streaming). `mcp_call` is excluded; that umbrella is synthesized
/// here, not forwarded.
fn is_tool_call_item_type(item_type: &str) -> bool {
    is_function_call_type(item_type) || openai_bridge::is_hosted_tool_call_item_type(item_type)
}

/// Action to take based on streaming event processing
#[derive(Debug)]
pub(crate) enum StreamAction {
    /// Pass this event to the client.
    Forward,
    /// Accumulate the event for tool execution, do not send downstream.
    Buffer,
    /// Drop the upstream event from the wire without triggering tool
    /// execution.
    ///
    /// Used for the R6.7c native-passthrough suppression: when the upstream
    /// OpenAI cloud API emits a hosted tool-call item (e.g.
    /// `image_generation_call`) directly and the accompanying umbrella
    /// `output_item.done` arrives BEFORE `response.<type>.completed`,
    /// forwarding the umbrella would violate the spec ordering invariant
    /// (see `.claude/_audit/openai-responses-api-spec.md` §streaming). The
    /// handler drops the duplicated/mis-ordered envelope; the subsequent
    /// `<type>.completed` still reaches the client and upstream re-emits the
    /// correctly-ordered umbrella later on the wire. Unlike
    /// [`StreamAction::ExecuteTools`], this variant does NOT kick the tool
    /// loop — native passthrough items are not tracked in the handler's
    /// pending-call registry, so there is nothing to dispatch.
    Drop,
    /// The upstream signalled tool-call completion — run the MCP tool now.
    ///
    /// `forward_triggering_event` distinguishes two upstream shapes:
    ///
    /// * When the upstream emits `function_call_arguments.done` we want to
    ///   forward the event (it becomes `mcp_call_arguments.done`, a sub-event
    ///   that belongs BEFORE `response.<type>.completed`).
    /// * When the upstream skips the delta/done arguments events and signals
    ///   tool completion with `output_item.done` for the `function_call`
    ///   item, forwarding it would emit a duplicate umbrella
    ///   `response.output_item.done` event that fires BEFORE the sub-event
    ///   `response.<type>.completed` — violating the spec sub-event ordering
    ///   (spec §streaming: the umbrella `output_item.done` is always the
    ///   LAST event for a given item). The tool_loop's
    ///   `send_tool_call_completion_events` emits the correct umbrella
    ///   `output_item.done` in its proper position.
    ExecuteTools { forward_triggering_event: bool },
}

/// Maps upstream output indices to sequential downstream indices
#[derive(Debug, Default)]
pub(crate) struct OutputIndexMapper {
    next_index: usize,
    // Map upstream output_index -> remapped output_index
    assigned: HashMap<usize, usize>,
}

impl OutputIndexMapper {
    pub fn with_start(next_index: usize) -> Self {
        Self {
            next_index,
            assigned: HashMap::new(),
        }
    }

    pub fn ensure_mapping(&mut self, upstream_index: usize) -> usize {
        *self.assigned.entry(upstream_index).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        })
    }

    pub fn lookup(&self, upstream_index: usize) -> Option<usize> {
        self.assigned.get(&upstream_index).copied()
    }

    pub fn allocate_synthetic(&mut self) -> usize {
        let assigned = self.next_index;
        self.next_index += 1;
        assigned
    }

    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

/// Represents a function call being accumulated across delta events
#[derive(Debug, Clone)]
pub(crate) struct FunctionCallInProgress {
    pub call_id: String,
    pub name: String,
    pub arguments_buffer: String,
    pub item_id: Option<String>,
    pub output_index: usize,
    pub last_obfuscation: Option<String>,
    pub assigned_output_index: Option<usize>,
}

impl FunctionCallInProgress {
    pub fn new(call_id: String, output_index: usize) -> Self {
        Self {
            call_id,
            name: String::new(),
            arguments_buffer: String::new(),
            item_id: None,
            output_index,
            last_obfuscation: None,
            assigned_output_index: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.name.is_empty()
    }

    pub fn effective_output_index(&self) -> usize {
        self.assigned_output_index.unwrap_or(self.output_index)
    }
}

/// Handles streaming responses with MCP tool call interception
pub(crate) struct StreamingToolHandler {
    /// Accumulator for response persistence
    pub accumulator: StreamingResponseAccumulator,
    /// Function calls being built from deltas
    pub pending_calls: Vec<FunctionCallInProgress>,
    /// Manage output_index remapping so they increment per item
    output_index_mapper: OutputIndexMapper,
    /// Original response id captured from the first response.created event
    pub original_response_id: Option<String>,
    /// Output indices whose `output_item.added` carried a known tool-call
    /// item type that the MCP-dispatch registry does NOT track (i.e. upstream
    /// emitted a native hosted-tool item directly — `image_generation_call`,
    /// `web_search_call`, `code_interpreter_call`, `file_search_call` — rather
    /// than wrapping it as a `function_call` that [`Self::handle_output_item_added`]
    /// would register in `pending_calls`). Recorded so the
    /// `output_item.done` gate can recognise the R6.7c "native passthrough"
    /// path and drop the duplicate/mis-ordered umbrella without kicking the
    /// tool loop.
    native_passthrough_tool_call_indices: HashSet<usize>,
}

impl StreamingToolHandler {
    pub fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
            native_passthrough_tool_call_indices: HashSet::new(),
        }
    }

    pub fn ensure_output_index(&mut self, upstream_index: usize) -> usize {
        self.output_index_mapper.ensure_mapping(upstream_index)
    }

    pub fn mapped_output_index(&self, upstream_index: usize) -> Option<usize> {
        self.output_index_mapper.lookup(upstream_index)
    }

    pub fn allocate_synthetic_output_index(&mut self) -> usize {
        self.output_index_mapper.allocate_synthetic()
    }

    pub fn next_output_index(&self) -> usize {
        self.output_index_mapper.next_index()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.original_response_id
            .as_deref()
            .or_else(|| self.accumulator.original_response_id())
    }

    /// Process an SSE event given as its name and data payload: the payload
    /// is parsed here, once, and handed to [`Self::process_parsed`]. An
    /// unparsable payload reaches the accumulator (which logs it, as it
    /// always did) and is forwarded.
    pub fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        match serde_json::from_str::<Value>(data) {
            Ok(parsed) => self.process_parsed(event_name, &parsed),
            Err(_) => {
                self.accumulator.ingest_event(event_name, data);
                StreamAction::Forward
            }
        }
    }

    /// Process an SSE event the reader has parsed already and determine
    /// what action to take. The accumulator records the same value, so the
    /// stored record holds exactly what the reader relayed, a data payload
    /// spanning lines included, and the reader's one parse serves its own
    /// checks, this gate and the record.
    pub fn process_parsed(&mut self, event_name: Option<&str>, parsed: &Value) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_parsed(event_name, parsed);

        match get_event_type(event_name, parsed) {
            ResponseEvent::CREATED => {
                if self.original_response_id.is_none() {
                    self.original_response_id = parsed
                        .get("response")
                        .and_then(|v| v.get("id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                StreamAction::Forward
            }
            ResponseEvent::COMPLETED => StreamAction::Forward,
            OutputItemEvent::ADDED => self.handle_output_item_added(parsed),
            FunctionCallEvent::ARGUMENTS_DELTA => self.handle_arguments_delta(parsed),
            FunctionCallEvent::ARGUMENTS_DONE => self.handle_arguments_done(parsed),
            OutputItemEvent::DELTA => self.process_output_delta(parsed),
            OutputItemEvent::DONE => {
                let done_output_index = extract_output_index(parsed);
                if let Some(output_index) = done_output_index {
                    self.ensure_output_index(output_index);
                }
                // Only suppress the umbrella `output_item.done` when it is
                // closing a tool-call item that we are intercepting — an
                // unrelated `output_item.done` for a message, reasoning
                // item, or mcp_list_tools block must reach the client
                // untouched even when a pending tool call is queued.
                //
                // The gate covers every tool-call item type that emits its
                // own structured `<type>.completed` sub-event (function_call
                // and the hosted builtins: image_generation_call,
                // web_search_call, code_interpreter_call, file_search_call).
                // When OpenAI cloud passthrough streams a native hosted tool
                // directly (not wrapped as function_call), the upstream
                // umbrella `output_item.done` fires BEFORE
                // `<type>.completed`, and forwarding it breaks the spec
                // ordering invariant.
                let is_tool_call_done = parsed
                    .get("item")
                    .and_then(|item| item.get("type"))
                    .and_then(|value| value.as_str())
                    .is_some_and(is_tool_call_item_type);
                // R6.7b MCP-dispatch guard: require the done event's
                // `output_index` to match one of our pending calls. Without
                // this, a mixed stream that carries a function_call the
                // handler intercepts plus a native hosted-tool item at a
                // different output_index could have its hosted-tool
                // `output_item.done` suppressed via the wrong branch — the
                // tool loop only re-emits umbrellas for items it
                // dispatched, so routing a passthrough item through the
                // MCP-dispatch branch would drop it permanently. The
                // output_index match ensures the MCP-dispatch branch only
                // kicks in for items tracked by `pending_calls`.
                let belongs_to_pending_call = done_output_index
                    .is_some_and(|idx| self.pending_calls.iter().any(|c| c.output_index == idx));
                // R6.7c native-passthrough guard: an upstream
                // `output_item.added` for a known hosted tool-call item
                // type that the MCP-dispatch path does NOT track
                // registered this output_index in
                // `native_passthrough_tool_call_indices`. When the
                // matching `output_item.done` arrives — mis-ordered BEFORE
                // `response.<type>.completed` on the cloud path — we drop
                // the upstream envelope rather than forwarding it, so the
                // wire order the client sees satisfies the spec invariant
                // that `output_item.done` is the LAST event for a given
                // item. Unlike the MCP-dispatch branch, this path does NOT
                // kick the tool loop (there is nothing to dispatch — the
                // item was not intercepted).
                //
                // We `take()` the index out of the set (via `remove`) so
                // that only the FIRST `output_item.done` for a
                // passthrough item is dropped. Upstream re-emits a
                // correctly-ordered umbrella after `<type>.completed`,
                // and that second envelope must be forwarded so the
                // client sees the terminal `output_item.done`. Without
                // this one-shot behaviour both envelopes would be
                // dropped and the client would never receive a final
                // umbrella for the passthrough item.
                if is_tool_call_done && belongs_to_pending_call && self.has_complete_calls() {
                    // MCP-dispatch path (R6.7b): suppress the upstream
                    // umbrella event and kick the tool loop — it will emit
                    // its own `output_item.done` at the correct position,
                    // AFTER `response.<type>.completed`.
                    StreamAction::ExecuteTools {
                        forward_triggering_event: false,
                    }
                } else if is_tool_call_done
                    && done_output_index
                        .is_some_and(|idx| self.native_passthrough_tool_call_indices.remove(&idx))
                {
                    // Native passthrough path (R6.7c): drop the upstream
                    // umbrella without running the tool loop. The
                    // downstream `<type>.completed` event still reaches
                    // the client, and upstream's subsequent
                    // correctly-ordered `output_item.done` is forwarded
                    // (the index was removed above so this branch will
                    // not match a second time for the same item).
                    StreamAction::Drop
                } else {
                    StreamAction::Forward
                }
            }
            _ => StreamAction::Forward,
        }
    }

    fn handle_output_item_added(&mut self, parsed: &Value) -> StreamAction {
        let cached_output_index = extract_output_index(parsed);
        if let Some(output_index) = cached_output_index {
            self.ensure_output_index(output_index);
        }

        let Some(item) = parsed.get("item") else {
            return StreamAction::Forward;
        };
        let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
            return StreamAction::Forward;
        };

        if !is_function_call_type(item_type) {
            // R6.7c: when upstream emits a hosted tool-call item directly
            // (e.g. `image_generation_call` from OpenAI cloud native
            // passthrough), record the output_index so the matching
            // `output_item.done` gate can recognise the item as native
            // passthrough rather than an MCP-dispatched call. This is the
            // counterpart to the `belongs_to_pending_call` guard added in
            // R6.7b — together they let the gate distinguish
            // MCP-dispatch (tracked in `pending_calls`) from native
            // passthrough (tracked here) without confusing the two.
            if is_tool_call_item_type(item_type) {
                if let Some(output_index) = cached_output_index {
                    self.native_passthrough_tool_call_indices
                        .insert(output_index);
                }
            }
            return StreamAction::Forward;
        }

        let Some(output_index) = cached_output_index else {
            warn!(
                "Missing output_index in function_call added event, \
                 forwarding without processing for tool execution"
            );
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let call = self.get_or_create_call(output_index, item);
        call.call_id = call_id.to_string();
        call.name = name.to_string();
        call.item_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .map(|id| id.to_string());
        call.assigned_output_index = Some(assigned_index);

        StreamAction::Forward
    }

    fn handle_arguments_delta(&mut self, parsed: &Value) -> StreamAction {
        let Some(output_index) = extract_output_index(parsed) else {
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);

        if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
            if let Some(call) = self.find_call_mut(output_index) {
                call.arguments_buffer.push_str(delta);
                if let Some(obfuscation) = parsed.get("obfuscation").and_then(|v| v.as_str()) {
                    call.last_obfuscation = Some(obfuscation.to_string());
                }
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }
        StreamAction::Forward
    }

    fn handle_arguments_done(&mut self, parsed: &Value) -> StreamAction {
        if let Some(output_index) = extract_output_index(parsed) {
            let assigned_index = self.ensure_output_index(output_index);
            if let Some(call) = self.find_call_mut(output_index) {
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }

        if self.has_complete_calls() {
            // Forward the triggering event — `function_call_arguments.done`
            // becomes `mcp_call_arguments.done` which is a sub-event
            // belonging BEFORE `response.<type>.completed` and therefore
            // safe to forward inline here.
            StreamAction::ExecuteTools {
                forward_triggering_event: true,
            }
        } else {
            StreamAction::Forward
        }
    }

    fn find_call_mut(&mut self, output_index: usize) -> Option<&mut FunctionCallInProgress> {
        self.pending_calls
            .iter_mut()
            .find(|c| c.output_index == output_index)
    }

    /// Process output delta events to detect and accumulate function calls
    fn process_output_delta(&mut self, event: &Value) -> StreamAction {
        let output_index = extract_output_index(event).unwrap_or(0);
        let assigned_index = self.ensure_output_index(output_index);

        let delta = match event.get("delta") {
            Some(d) => d,
            None => return StreamAction::Forward,
        };

        let item_type = delta.get("type").and_then(|v| v.as_str());

        if item_type.is_some_and(is_function_call_type) {
            // Get or create function call for this output index
            let call = self.get_or_create_call(output_index, delta);
            call.assigned_output_index = Some(assigned_index);

            if let Some(call_id) = delta.get("call_id").and_then(|v| v.as_str()) {
                call.call_id = call_id.to_string();
            }
            if let Some(item_id) = delta.get("id").and_then(|v| v.as_str()) {
                call.item_id = Some(item_id.to_string());
            }
            if let Some(name) = delta.get("name").and_then(|v| v.as_str()) {
                call.name.push_str(name);
            }
            if let Some(args) = delta.get("arguments").and_then(|v| v.as_str()) {
                call.arguments_buffer.push_str(args);
            }

            if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
                call.last_obfuscation = Some(obfuscation.to_string());
            }

            return StreamAction::Buffer;
        }

        StreamAction::Forward
    }

    fn get_or_create_call(
        &mut self,
        output_index: usize,
        delta: &Value,
    ) -> &mut FunctionCallInProgress {
        if let Some(pos) = self
            .pending_calls
            .iter()
            .position(|c| c.output_index == output_index)
        {
            return &mut self.pending_calls[pos];
        }

        let call_id = delta
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut call = FunctionCallInProgress::new(call_id, output_index);
        if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
            call.last_obfuscation = Some(obfuscation.to_string());
        }

        self.pending_calls.push(call);
        #[expect(
            clippy::expect_used,
            reason = "just pushed an element; last_mut is infallible"
        )]
        self.pending_calls
            .last_mut()
            .expect("Just pushed to pending_calls, must have at least one element")
    }

    fn has_complete_calls(&self) -> bool {
        !self.pending_calls.is_empty() && self.pending_calls.iter().all(|c| c.is_complete())
    }

    pub fn take_pending_calls(&mut self) -> Vec<FunctionCallInProgress> {
        std::mem::take(&mut self.pending_calls)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/mcp/tool_handler/tests.rs"]
mod tests;
