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
    /// than wrapping it as a `function_call` that [`handle_output_item_added`]
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

    pub fn snapshot_final_response(&self) -> Option<Value> {
        self.accumulator.snapshot_final_response()
    }

    /// Process an SSE event and determine what action to take
    pub fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_block(&format!(
            "{}data: {}",
            event_name
                .map(|n| format!("event: {n}\n"))
                .unwrap_or_default(),
            data
        ));

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return StreamAction::Forward,
        };

        match get_event_type(event_name, &parsed) {
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
            OutputItemEvent::ADDED => self.handle_output_item_added(&parsed),
            FunctionCallEvent::ARGUMENTS_DELTA => self.handle_arguments_delta(&parsed),
            FunctionCallEvent::ARGUMENTS_DONE => self.handle_arguments_done(&parsed),
            OutputItemEvent::DELTA => self.process_output_delta(&parsed),
            OutputItemEvent::DONE => {
                let done_output_index = extract_output_index(&parsed);
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
                // `output_item.done` suppressed via the wrong arm — the
                // tool loop only re-emits umbrellas for items it
                // dispatched, so routing a passthrough item through the
                // MCP-dispatch arm would drop it permanently. The
                // output_index match ensures the MCP-dispatch arm only
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
                // item. Unlike the MCP-dispatch arm, this path does NOT
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
mod tests {
    //! Regression tests for streaming event ordering on image-generation-
    //! style built-in tool calls.
    //!
    //! The bug: when the upstream signalled tool completion with a direct
    //! `response.output_item.done` event (no preceding
    //! `response.function_call_arguments.done`), the streaming layer used
    //! to forward that upstream event to the client. The tool loop then
    //! emitted its own umbrella `output_item.done` AFTER emitting the
    //! `<tool>.completed` sub-event — producing the wire sequence
    //!
    //!     response.output_item.added
    //!     response.<tool>.in_progress
    //!     response.output_item.done       ← duplicate (upstream-forwarded)
    //!     response.<tool>.generating
    //!     response.<tool>.completed
    //!     response.output_item.done       ← tool-loop synthesized
    //!
    //! which violates the spec invariant that `response.output_item.done`
    //! is the LAST event emitted for a given item (see
    //! `.claude/_audit/openai-responses-api-spec.md` §streaming, events
    //! L1054-L1072).

    use openai_protocol::event_types::ItemType;

    use super::*;

    /// Feed an `output_item.added` for a function_call item so the handler
    /// has a complete pending call registered.
    fn bootstrap_function_call_added(handler: &mut StreamingToolHandler) {
        let data = r#"{
          "type": "response.output_item.added",
          "output_index": 0,
          "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "image_generation"
          }
        }"#;
        let _ = handler.process_event(Some("response.output_item.added"), data);
    }

    #[test]
    fn output_item_done_triggers_execute_tools_without_forwarding() {
        // When the upstream signals tool-call completion via
        // `output_item.done` for a function_call item (no preceding
        // `function_call_arguments.done`), the handler must ask the
        // caller NOT to forward the triggering event — the tool loop
        // will emit its own umbrella `output_item.done` at the correct
        // position after `response.<tool>.completed`.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "image_generation",
            "arguments": "{}",
            "status": "completed"
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                !forward_triggering_event,
                "output_item.done must be suppressed to avoid a duplicate umbrella event"
            ),
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    }

    /// Assert that an `output_item.done` for `item_type` is suppressed (the
    /// gate returns `ExecuteTools { forward_triggering_event: false }`) when
    /// a tool call is already pending. The gate must match every hosted
    /// tool-call item type — OpenAI cloud passthrough of native hosted
    /// tools (which emits the item directly with its hosted-tool type) would
    /// otherwise slip through and send a duplicate umbrella event to the
    /// wire.
    fn assert_output_item_done_suppressed_for_hosted_tool(item_type: &str) {
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = format!(
            r#"{{
              "type": "response.output_item.done",
              "output_index": 0,
              "item": {{
                "type": "{item_type}",
                "id": "hosted_test",
                "status": "completed"
              }}
            }}"#
        );

        let action = handler.process_event(Some("response.output_item.done"), &done_event);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                !forward_triggering_event,
                "output_item.done for {item_type} must be suppressed to avoid a duplicate umbrella event"
            ),
            other => panic!("expected ExecuteTools for {item_type}, got {other:?}"),
        }
    }

    #[test]
    fn output_item_done_suppressed_for_image_generation_call() {
        // OpenAI cloud passthrough emits the `image_generation_call` item
        // directly — the umbrella `output_item.done` that upstream sends
        // before `response.image_generation_call.completed` must be
        // suppressed so the tool loop's umbrella lands at the right spot.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::IMAGE_GENERATION_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_web_search_call() {
        // Hosted `web_search_call` items emitted directly by upstream
        // must also trigger suppression — the tool loop emits
        // `response.web_search_call.completed` followed by its own
        // `output_item.done` at the correct position.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::WEB_SEARCH_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_code_interpreter_call() {
        // Hosted `code_interpreter_call` items share the ordering
        // contract with the other tool-call item types.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::CODE_INTERPRETER_CALL);
    }

    #[test]
    fn output_item_done_suppressed_for_file_search_call() {
        // Hosted `file_search_call` items share the ordering contract
        // with the other tool-call item types.
        assert_output_item_done_suppressed_for_hosted_tool(ItemType::FILE_SEARCH_CALL);
    }

    #[test]
    fn output_item_done_for_unobserved_hosted_tool_output_index_forwards() {
        // When the handler has seen NEITHER a function_call_added (MCP
        // dispatch path) NOR an output_item.added for a hosted tool-call
        // type (native passthrough path) at the done event's
        // output_index, the gate must forward. Dropping the umbrella
        // without having observed the item's start would silently
        // swallow an `output_item.done` that no one else is going to
        // re-emit.
        //
        // This scenario models a truly spurious/unexpected
        // `output_item.done` whose matching `output_item.added` never
        // arrived — the safe action is to forward it rather than drop.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        // Pending function_call at output_index 0 (MCP dispatch).
        bootstrap_function_call_added(&mut handler);

        // `output_item.done` at output_index 1 with NO preceding
        // `output_item.added` — neither MCP-dispatch nor native
        // passthrough tracking recorded this index.
        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 1,
          "item": {
            "type": "web_search_call",
            "id": "ws_passthrough",
            "status": "completed",
            "action": {"type": "search"}
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        assert!(
            matches!(action, StreamAction::Forward),
            "output_item.done for an index the handler never observed via output_item.added \
             must forward — suppressing here would drop the umbrella permanently; got {action:?}"
        );
    }

    #[test]
    fn arguments_done_triggers_execute_tools_with_forwarding() {
        // Forwarding is safe for `function_call_arguments.done` because it
        // becomes `mcp_call_arguments.done` — a sub-event that belongs
        // BEFORE `response.<tool>.completed` per spec sub-event ordering.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let args_done = r#"{
          "type": "response.function_call_arguments.done",
          "output_index": 0,
          "item_id": "fc_test",
          "arguments": "{}"
        }"#;

        let action =
            handler.process_event(Some("response.function_call_arguments.done"), args_done);

        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                forward_triggering_event,
                "function_call_arguments.done forwards as mcp_call_arguments.done"
            ),
            other => panic!("expected ExecuteTools, got {other:?}"),
        }
    }

    #[test]
    fn output_item_done_for_unrelated_item_forwards_even_with_pending_call() {
        // Regression for PR #1369 review (chatgpt-codex-connector P1):
        // suppression must be gated on the done event's item type. When a
        // function_call is queued AND an unrelated `output_item.done` for a
        // message / reasoning / mcp_list_tools block arrives, the umbrella
        // event for that sibling item must pass through untouched so the
        // client sees its completion. Only the function_call's own
        // `output_item.done` should be suppressed (the tool loop will emit
        // the correct umbrella at its proper position).
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 1,
          "item": {
            "type": "message",
            "id": "msg_sibling",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}]
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        assert!(
            matches!(action, StreamAction::Forward),
            "sibling output_item.done must forward even while a function_call is pending; got {action:?}"
        );
    }

    #[test]
    fn output_item_done_without_pending_calls_forwards() {
        // Non-function-call umbrella `output_item.done` events (messages,
        // mcp_list_tools items, etc.) must pass through unchanged. This
        // prevents regressing the path where the assistant emits a plain
        // text message after tool execution completes.
        let mut handler = StreamingToolHandler::with_starting_index(0);

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "message",
            "id": "msg_test",
            "status": "completed",
            "role": "assistant",
            "content": []
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);

        assert!(
            matches!(action, StreamAction::Forward),
            "expected Forward for an umbrella output_item.done on a non-tool item, got {action:?}"
        );
    }

    // ========================================================================
    // R6.7c — native passthrough suppression.
    //
    // When OpenAI cloud passthrough streams a hosted tool-call item
    // directly (e.g. `image_generation_call` emitted end-to-end rather
    // than wrapped as a `function_call` we intercept), R6.7b's
    // `belongs_to_pending_call` guard short-circuits the gate because
    // `handle_output_item_added` only registers `function_call` /
    // `function_tool_call` types in `pending_calls`. The duplicate
    // umbrella then reached the wire mis-ordered before
    // `response.<type>.completed` and broke the spec invariant.
    //
    // R6.7c extends the gate with an OR-branch: an `output_item.added`
    // for a hosted tool-call type records the output_index in
    // `native_passthrough_tool_call_indices`, and the matching
    // `output_item.done` returns `StreamAction::Drop` so the caller
    // drops the upstream envelope without kicking the tool loop. The
    // R6.7b MCP-dispatch arm is untouched — both paths coexist.
    // ========================================================================

    /// Feed an `output_item.added` for a hosted tool-call type so the
    /// handler records the index in `native_passthrough_tool_call_indices`.
    fn bootstrap_native_hosted_tool_added(
        handler: &mut StreamingToolHandler,
        item_type: &str,
        output_index: usize,
        item_id: &str,
    ) {
        let added_event = format!(
            r#"{{
              "type": "response.output_item.added",
              "output_index": {output_index},
              "item": {{
                "type": "{item_type}",
                "id": "{item_id}",
                "status": "in_progress"
              }}
            }}"#
        );
        let _ = handler.process_event(Some("response.output_item.added"), &added_event);
    }

    /// Assert that an `output_item.done` for a hosted tool-call item at
    /// `output_index` is dropped (the gate returns `StreamAction::Drop`)
    /// after the corresponding upstream `output_item.added` was observed.
    /// This is the R6.7c native-passthrough case: no MCP dispatch, no
    /// pending call — just upstream emitting the item directly and the
    /// gate needing to drop the mis-ordered umbrella.
    fn assert_native_passthrough_done_drops(item_type: &str) {
        let mut handler = StreamingToolHandler::with_starting_index(0);
        // Only upstream's `output_item.added` — no MCP-dispatched function
        // call is pending.
        bootstrap_native_hosted_tool_added(&mut handler, item_type, 0, "native_test");

        let done_event = format!(
            r#"{{
              "type": "response.output_item.done",
              "output_index": 0,
              "item": {{
                "type": "{item_type}",
                "id": "native_test",
                "status": "completed"
              }}
            }}"#
        );

        let action = handler.process_event(Some("response.output_item.done"), &done_event);

        assert!(
            matches!(action, StreamAction::Drop),
            "output_item.done for native-passthrough {item_type} must be dropped — \
             upstream emits a duplicate/mis-ordered umbrella, dropping preserves wire \
             ordering; got {action:?}"
        );
    }

    #[test]
    fn output_item_done_dropped_for_native_passthrough_image_generation_call() {
        // R6.7c: OpenAI cloud passthrough emits `image_generation_call`
        // directly. With R6.7b alone the mis-ordered umbrella leaked to
        // the wire because the item was not tracked in `pending_calls`.
        // The R6.7c arm drops it based on the observed `output_item.added`.
        assert_native_passthrough_done_drops(ItemType::IMAGE_GENERATION_CALL);
    }

    #[test]
    fn output_item_done_dropped_for_native_passthrough_web_search_call() {
        assert_native_passthrough_done_drops(ItemType::WEB_SEARCH_CALL);
    }

    #[test]
    fn output_item_done_dropped_for_native_passthrough_code_interpreter_call() {
        assert_native_passthrough_done_drops(ItemType::CODE_INTERPRETER_CALL);
    }

    #[test]
    fn output_item_done_dropped_for_native_passthrough_file_search_call() {
        assert_native_passthrough_done_drops(ItemType::FILE_SEARCH_CALL);
    }

    #[test]
    fn native_passthrough_gate_coexists_with_mcp_dispatch_gate() {
        // R6.7c invariant: both arms of the gate must fire correctly when
        // a single stream carries an MCP-dispatched function_call at one
        // output_index AND a native-passthrough hosted tool-call at
        // another. The function_call's done goes through the
        // `ExecuteTools` arm (R6.7b); the hosted tool-call's done goes
        // through the `Drop` arm (R6.7c). Neither re-narrows the other.
        let mut handler = StreamingToolHandler::with_starting_index(0);

        // MCP-dispatch path: function_call at output_index 0.
        bootstrap_function_call_added(&mut handler);

        // Native passthrough path: image_generation_call at output_index 1.
        bootstrap_native_hosted_tool_added(
            &mut handler,
            ItemType::IMAGE_GENERATION_CALL,
            1,
            "ig_native",
        );

        // First done — hosted-tool at output_index 1 should Drop.
        let passthrough_done = r#"{
          "type": "response.output_item.done",
          "output_index": 1,
          "item": {
            "type": "image_generation_call",
            "id": "ig_native",
            "status": "completed"
          }
        }"#;
        let action = handler.process_event(Some("response.output_item.done"), passthrough_done);
        assert!(
            matches!(action, StreamAction::Drop),
            "native passthrough hosted-tool done must Drop; got {action:?}"
        );

        // Second done — function_call at output_index 0 should
        // ExecuteTools (the MCP-dispatch arm). This confirms R6.7c did
        // not accidentally funnel the function_call into the new branch.
        let function_call_done = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "image_generation",
            "arguments": "{}",
            "status": "completed"
          }
        }"#;
        let action = handler.process_event(Some("response.output_item.done"), function_call_done);
        match action {
            StreamAction::ExecuteTools {
                forward_triggering_event,
            } => assert!(
                !forward_triggering_event,
                "MCP-dispatch arm must still suppress the upstream umbrella"
            ),
            other => panic!(
                "function_call done must go through MCP-dispatch arm (ExecuteTools), got {other:?}"
            ),
        }
    }

    #[test]
    fn native_passthrough_done_drops_even_when_pending_function_call_complete() {
        // R6.7c must not depend on `pending_calls` being empty. When an
        // MCP-dispatched function_call is in-flight (has_complete_calls()
        // is true) AND a native passthrough hosted-tool done arrives at
        // a different output_index, the passthrough path must still
        // Drop — `belongs_to_pending_call` for the done's own index is
        // false, so the R6.7b arm cleanly declines and the R6.7c arm
        // fires.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_function_call_added(&mut handler); // complete pending call @ idx 0
        bootstrap_native_hosted_tool_added(&mut handler, ItemType::WEB_SEARCH_CALL, 2, "ws_native");

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 2,
          "item": {
            "type": "web_search_call",
            "id": "ws_native",
            "status": "completed",
            "action": {"type": "search"}
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.done"), done_event);
        assert!(
            matches!(action, StreamAction::Drop),
            "native passthrough done at a non-pending-call index must Drop even when a \
             function_call is pending elsewhere; got {action:?}"
        );
    }

    #[test]
    fn output_item_added_for_native_passthrough_still_forwards() {
        // R6.7c records the output_index for later drop-gating but must
        // still forward the `output_item.added` itself so the client
        // sees the item come into existence. Only the mis-ordered
        // `output_item.done` is dropped.
        let mut handler = StreamingToolHandler::with_starting_index(0);

        let added_event = r#"{
          "type": "response.output_item.added",
          "output_index": 0,
          "item": {
            "type": "image_generation_call",
            "id": "ig_native",
            "status": "in_progress"
          }
        }"#;

        let action = handler.process_event(Some("response.output_item.added"), added_event);
        assert!(
            matches!(action, StreamAction::Forward),
            "native passthrough output_item.added must still forward; got {action:?}"
        );
    }

    #[test]
    fn native_passthrough_second_output_item_done_forwards_after_first_dropped() {
        // R6.7c one-shot invariant (CodeRabbit P1 fix): the first
        // (mis-ordered) upstream `output_item.done` for a native
        // passthrough item is dropped, but upstream then re-emits a
        // correctly-ordered `output_item.done` after
        // `response.<type>.completed`. That second envelope MUST be
        // forwarded so the client sees the terminal umbrella for the
        // item — otherwise the spec's "output_item.done is the LAST
        // event" invariant fails on the client side.
        //
        // The implementation removes the passthrough index from
        // `native_passthrough_tool_call_indices` on the first drop, so
        // the second done falls through to the `Forward` arm.
        let mut handler = StreamingToolHandler::with_starting_index(0);
        bootstrap_native_hosted_tool_added(
            &mut handler,
            ItemType::IMAGE_GENERATION_CALL,
            0,
            "ig_native",
        );

        let done_event = r#"{
          "type": "response.output_item.done",
          "output_index": 0,
          "item": {
            "type": "image_generation_call",
            "id": "ig_native",
            "status": "completed"
          }
        }"#;

        // First done (mis-ordered, arrives BEFORE `<type>.completed`) —
        // must be dropped.
        let first_action = handler.process_event(Some("response.output_item.done"), done_event);
        assert!(
            matches!(first_action, StreamAction::Drop),
            "first output_item.done for native passthrough must Drop; got {first_action:?}"
        );

        // Second done (correctly-ordered, arrives AFTER
        // `<type>.completed`) — must be forwarded so the client sees
        // the terminal umbrella.
        let second_action = handler.process_event(Some("response.output_item.done"), done_event);
        assert!(
            matches!(second_action, StreamAction::Forward),
            "second output_item.done for native passthrough must Forward after the first \
             dropped envelope consumed the passthrough-index entry; got {second_action:?}"
        );
    }

    #[test]
    fn is_tool_call_item_type_covers_function_call_and_hosted_builtins() {
        // Pin the suppression-gate set: function_call variants + hosted
        // builtins. `mcp_call` is NOT included — those umbrellas are
        // emitted by the tool loop, not forwarded from upstream.
        // Adding a new hosted format must flip this set automatically via
        // openai_bridge::is_hosted_tool_call_item_type — no edit here.
        for ty in [
            ItemType::FUNCTION_CALL,
            ItemType::FUNCTION_TOOL_CALL,
            ItemType::WEB_SEARCH_CALL,
            ItemType::CODE_INTERPRETER_CALL,
            ItemType::FILE_SEARCH_CALL,
            ItemType::IMAGE_GENERATION_CALL,
        ] {
            assert!(
                is_tool_call_item_type(ty),
                "{ty} must be classified as a tool-call item type"
            );
        }
        for ty in [
            ItemType::MCP_CALL,
            ItemType::MCP_LIST_TOOLS,
            ItemType::FUNCTION_CALL_OUTPUT,
            "message",
            "reasoning",
        ] {
            assert!(
                !is_tool_call_item_type(ty),
                "{ty} must NOT be classified as a tool-call item type"
            );
        }
    }
}
