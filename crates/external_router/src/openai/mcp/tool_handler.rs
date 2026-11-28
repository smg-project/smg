//! Streaming tool call handling for MCP interception.

use std::collections::{HashMap, HashSet};

use openai_protocol::event_types::{
    is_function_call_type, FunctionCallEvent, OutputItemEvent, OutputTextEvent, ResponseEvent,
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
    ///
    /// Also used for the upstream's own `output_item.done` of a call the
    /// tool loop took already (the relay reads a turn on after a
    /// hand-over): the loop writes that call's umbrella itself.
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
    /// Upstream output indices of the calls [`Self::take_pending_calls`]
    /// handed to the tool loop. The loop writes each call's
    /// `output_item.done` itself, so the upstream's own umbrella for such a
    /// call, read when the relay goes on with the turn, is dropped.
    dispatched_tool_call_indices: HashSet<usize>,
    /// The items other than function calls that the upstream opened behind
    /// a hand-over and whose closing event the client has not read, by
    /// upstream output index: each as announced, the text streamed for it
    /// since laid into its parts. A turn cut or stopped there closes them
    /// (see [`Self::take_open_items`]).
    open_items: Vec<(usize, Value)>,
    /// The upstream output indices of the items other than function calls
    /// that the upstream closed behind a hand-over. The loop keeps those
    /// items for the final response, each at the output index the client
    /// read for it (see [`Self::take_closed_items`]).
    closed_items: Vec<usize>,
}

/// The text deltas of an item other than a function call: the event, the
/// item's list of parts, the event's key for the part's index, the part's
/// type and the part's key for its text.
const OPEN_ITEM_TEXT_DELTAS: [(&str, &str, &str, &str, &str); 4] = [
    (
        OutputTextEvent::DELTA,
        "content",
        "content_index",
        "output_text",
        "text",
    ),
    (
        "response.refusal.delta",
        "content",
        "content_index",
        "refusal",
        "refusal",
    ),
    (
        "response.reasoning_summary_text.delta",
        "summary",
        "summary_index",
        "summary_text",
        "text",
    ),
    (
        "response.reasoning_text.delta",
        "content",
        "content_index",
        "reasoning_text",
        "text",
    ),
];

impl StreamingToolHandler {
    pub fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
            native_passthrough_tool_call_indices: HashSet::new(),
            dispatched_tool_call_indices: HashSet::new(),
            open_items: Vec::new(),
            closed_items: Vec::new(),
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
                        .is_some_and(|idx| self.dispatched_tool_call_indices.contains(&idx))
                {
                    // The tool loop took this call and wrote its
                    // `output_item.done` itself: the upstream's own
                    // umbrella, read after the hand-over, is dropped and
                    // does not kick the loop again.
                    StreamAction::Drop
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
                    // The client reads this closing event: a cut has
                    // nothing to close for the item. A closing event
                    // dropped above (a hosted tool call's first) leaves
                    // its item open.
                    if let Some(output_index) = done_output_index {
                        self.open_items.retain(|(open, _)| *open != output_index);
                    }
                    // Behind a hand-over an item that is no function call
                    // reaches the client closed: the loop keeps it for the
                    // final response, among its own items. The accumulator
                    // holds the item; its index is noted here.
                    if !self.dispatched_tool_call_indices.is_empty() {
                        if let (Some(upstream_index), Some(item)) =
                            (done_output_index, parsed.get("item"))
                        {
                            let is_function_call = item
                                .get("type")
                                .and_then(Value::as_str)
                                .is_some_and(is_function_call_type);
                            if !is_function_call {
                                self.closed_items.push(upstream_index);
                            }
                        }
                    }
                    StreamAction::Forward
                }
            }
            event_type => {
                if !self.open_items.is_empty() {
                    self.record_open_item_text(event_type, parsed);
                }
                StreamAction::Forward
            }
        }
    }

    /// Lays one text delta into the open item it belongs to, in the part
    /// the event names, so a turn cut there closes the item with what was
    /// streamed. A part is opened by its first delta, in order.
    fn record_open_item_text(&mut self, event_type: &str, parsed: &Value) {
        let Some(&(_, list, index_key, part_type, text_key)) =
            OPEN_ITEM_TEXT_DELTAS.iter().find(|row| row.0 == event_type)
        else {
            return;
        };
        let (Some(output_index), Some(delta)) = (
            extract_output_index(parsed),
            parsed.get("delta").and_then(Value::as_str),
        ) else {
            return;
        };
        let Some(parts) = self
            .open_items
            .iter_mut()
            .find(|(open, _)| *open == output_index)
            .and_then(|(_, item)| item.as_object_mut())
            .map(|item| item.entry(list).or_insert_with(|| Value::Array(Vec::new())))
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        let part_index = parsed
            .get(index_key)
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(0);
        if part_index == parts.len() {
            let mut part = serde_json::Map::new();
            part.insert("type".to_string(), Value::String(part_type.to_string()));
            part.insert(text_key.to_string(), Value::String(String::new()));
            if part_type == "output_text" {
                part.insert("annotations".to_string(), Value::Array(Vec::new()));
            }
            parts.push(Value::Object(part));
        }
        if let Some(Value::String(text)) = parts
            .get_mut(part_index)
            .and_then(|part| part.get_mut(text_key))
        {
            text.push_str(delta);
        }
    }

    /// Takes the items other than function calls that the upstream closed
    /// behind a hand-over, each under the output index the client read for
    /// it and as the upstream closed it. They move out of the accumulator:
    /// a final response built from this handler (the stop at the tool-call
    /// limit) lists each once, where the loop places it.
    pub fn take_closed_items(&mut self) -> Vec<(usize, Value)> {
        let closed_items = std::mem::take(&mut self.closed_items);
        closed_items
            .into_iter()
            .filter_map(|upstream_index| {
                let item = self.accumulator.take_output_item(upstream_index)?;
                let output_index = self
                    .mapped_output_index(upstream_index)
                    .unwrap_or(upstream_index);
                Some((output_index, item))
            })
            .collect()
    }

    /// Takes the items the upstream left open behind a hand-over, each
    /// under the output index the client reads and closed the way a cut
    /// turn closes it: the item as announced with the text streamed for
    /// it, `incomplete` (a hosted tool call: `failed`, the status the skip
    /// road gives a call of its type that did not run). An item the
    /// upstream closed by an event the client did not read (a hosted tool
    /// call's first `output_item.done`, dropped) is taken as the upstream
    /// closed it and moves out of the accumulator.
    pub fn take_open_items(&mut self) -> Vec<(usize, Value)> {
        let open_items = std::mem::take(&mut self.open_items);
        open_items
            .into_iter()
            .map(|(upstream_index, mut item)| {
                if let Some(closed) = self.accumulator.take_output_item(upstream_index) {
                    item = closed;
                } else {
                    let status = match item.get("type").and_then(Value::as_str) {
                        Some(item_type) if is_tool_call_item_type(item_type) => "failed",
                        _ => "incomplete",
                    };
                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("status".to_string(), Value::String(status.to_string()));
                    }
                }
                let output_index = self
                    .mapped_output_index(upstream_index)
                    .unwrap_or(upstream_index);
                (output_index, item)
            })
            .collect()
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
            // Behind a hand-over the turn can be cut: an item that opens
            // there is kept as announced until the upstream closes it.
            if !self.dispatched_tool_call_indices.is_empty() {
                if let Some(output_index) = cached_output_index {
                    self.open_items.push((output_index, item.clone()));
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
        self.dispatched_tool_call_indices
            .extend(self.pending_calls.iter().map(|call| call.output_index));
        std::mem::take(&mut self.pending_calls)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/mcp/tool_handler/tests.rs"]
mod tests;
