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
use serde_json::json;

use super::*;
use crate::test_support::counting_alloc;

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

    let action = handler.process_event(Some("response.function_call_arguments.done"), args_done);

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
// R6.7b MCP-dispatch branch is untouched — both paths coexist.
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
    // The R6.7c branch drops it based on the observed `output_item.added`.
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
    // R6.7c invariant: both branches of the gate must fire correctly when
    // a single stream carries an MCP-dispatched function_call at one
    // output_index AND a native-passthrough hosted tool-call at
    // another. The function_call's done goes through the
    // `ExecuteTools` branch (R6.7b); the hosted tool-call's done goes
    // through the `Drop` branch (R6.7c). Neither re-narrows the other.
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
    // ExecuteTools (the MCP-dispatch branch). This confirms R6.7c did
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
            "MCP-dispatch branch must still suppress the upstream umbrella"
        ),
        other => panic!(
            "function_call done must go through the MCP-dispatch branch (ExecuteTools), got {other:?}"
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
    // false, so the R6.7b branch cleanly declines and the R6.7c branch
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
    // the second done falls through to the `Forward` branch.
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

/// A relayed event whose data payload spans lines (an upstream that
/// indents or splits its JSON over `data:` lines, which the SSE reader
/// joins) reaches the stored record whole: the record is built from the
/// value the relay parsed, not from a block rebuilt around the payload.
#[test]
fn an_event_whose_data_spans_lines_reaches_the_record_whole() {
    let mut handler = StreamingToolHandler::with_starting_index(0);
    let created = "{\n  \"type\": \"response.created\",\n  \"response\": {\"id\": \"resp_lines\", \"output\": []}\n}";

    let action = handler.process_event(Some("response.created"), created);

    assert!(matches!(action, StreamAction::Forward), "{action:?}");
    assert_eq!(handler.original_response_id(), Some("resp_lines"));
    let record = handler
        .accumulator
        .into_final_response()
        .expect("the created event is on record");
    assert_eq!(record["id"], json!("resp_lines"));
}

/// The relay hands the accumulator the event it parsed: a relayed text
/// delta and a relayed `response.completed` of four output items each
/// cost the one parse, with no block rebuilt and no second parse behind
/// it. The counts are printed for the Numbers table.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
fn a_relayed_event_is_parsed_once() {
    let mut handler = StreamingToolHandler::with_starting_index(0);
    let delta = serde_json::to_string(&json!({
        "type": "response.output_text.delta",
        "sequence_number": 7,
        "item_id": "msg_0123456789abcdef0123456789abcdef",
        "output_index": 0,
        "content_index": 0,
        "delta": "a dozen words of answer text that a model streams per step ",
    }))
    .expect("a delta event");
    let items: Vec<Value> = (0..4)
        .map(|i| {
            json!({"type": "message", "id": format!("msg_{i}"), "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": format!("paragraph {i}: a few sentences of answer text, long enough to live on the heap"), "annotations": []}]})
        })
        .collect();
    let completed = serde_json::to_string(&json!({
        "type": "response.completed",
        "sequence_number": 8,
        "response": {"id": "resp_pinned", "object": "response", "status": "completed", "model": "m",
            "output": items, "usage": {"input_tokens": 120, "output_tokens": 480, "total_tokens": 600}},
    }))
    .expect("a completed event");

    let before = counting_alloc::calls();
    let delta_action = handler.process_event(Some("response.output_text.delta"), &delta);
    let delta_calls = counting_alloc::calls() - before;
    let before = counting_alloc::calls();
    let completed_action = handler.process_event(Some("response.completed"), &completed);
    let completed_calls = counting_alloc::calls() - before;

    eprintln!(
        "a relayed text delta: {delta_calls} allocation calls; a relayed response.completed of four output items: {completed_calls} allocation calls"
    );
    assert!(
        matches!(delta_action, StreamAction::Forward),
        "{delta_action:?}"
    );
    assert!(
        matches!(completed_action, StreamAction::Forward),
        "{completed_action:?}"
    );
    assert_eq!(
        handler
            .accumulator
            .into_final_response()
            .map(|r| r["id"].clone()),
        Some(json!("resp_pinned"))
    );
    assert_eq!(
        delta_calls, PINNED_DELTA_CALLS,
        "the relayed delta's allocation calls changed"
    );
    assert_eq!(
        completed_calls, PINNED_COMPLETED_CALLS,
        "the relayed response.completed's allocation calls changed"
    );
}

/// Allocation calls of the relayed text delta above, today.
const PINNED_DELTA_CALLS: usize = 13;
/// Allocation calls of the relayed `response.completed` above, today.
const PINNED_COMPLETED_CALLS: usize = 204;
