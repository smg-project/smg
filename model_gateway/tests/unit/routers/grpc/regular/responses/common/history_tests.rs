//! The tool loop is rebuilt on the loaded request: a stored turn's tool
//! calls and outputs stay in front of the calls this loop executes.

use openai_protocol::{chat::ChatMessage, common::ConversationRef};

use super::{super::conversions, *};
use crate::routers::grpc::common::responses::{
    history::tests::{function_call_output, function_call_output_input, shape, Stores},
    load_history, ReplayInto,
};

/// The request the rebuild tests load: one completed user message and
/// nothing else, so a rebuilt request's items read as that message
/// followed by the calls recorded since.
fn one_message_request() -> ResponsesRequest {
    ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_1".to_string(),
            role: "user".to_string(),
            content: vec![responses::ResponseContentPart::InputText {
                text: "What is 6*7?".to_string(),
            }],
            status: Some("completed".to_string()),
            phase: None,
        }]),
        ..Default::default()
    }
}

/// Records in the state the executed call `call_<tag>` of the tool
/// `search`, answered `found`: what the loop records between a response
/// and the rebuild that follows it. The call id is what `shape` prints,
/// so the tests assert it (two tags tell a second rebuild's calls
/// apart); the output item's id `fc_<tag>` is read by none of them.
fn record_search_call(state: &mut ToolLoopState, request: &ResponsesRequest, tag: &str) {
    state.record_call(
        request.tools.as_deref(),
        format!("call_{tag}"),
        "search".to_string(),
        "{}".to_string(),
        "found".to_string(),
        ResponseOutputItem::new_function_tool_call(
            format!("fc_{tag}"),
            format!("call_{tag}"),
            "search".to_string(),
            "{}".to_string(),
            None,
            "completed".to_string(),
        ),
    );
}

/// Turn 2 answers the call a stored turn issued: the engine sees the
/// question, the call and its answer on the first iteration, and the
/// request rebuilt after a call this loop executed still carries them,
/// in front of that call.
#[tokio::test]
async fn the_tool_loop_rebuilds_on_the_loaded_history() {
    let stores = Stores::with_conversation().await;
    let turn_1 = ResponsesRequest {
        input: ResponseInput::Text("What is 6*7?".to_string()),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    stores
        .store_turn("resp_1", &turn_1, function_call_output())
        .await;
    let turn_2 = ResponsesRequest {
        input: function_call_output_input(),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    let loaded = load_history(stores.history(), &turn_2, None, ReplayInto::Chat)
        .await
        .expect("the conversation loads");

    let chat = conversions::responses_to_chat(&loaded).expect("the loaded history converts");
    let turns: Vec<&str> = chat
        .messages
        .iter()
        .map(|message| match message {
            ChatMessage::User { .. } => "user",
            ChatMessage::Assistant {
                tool_calls: Some(_),
                ..
            } => "assistant+tool_calls",
            ChatMessage::Tool { .. } => "tool",
            _ => "other",
        })
        .collect();
    assert_eq!(turns, ["user", "assistant+tool_calls", "tool"]);

    let mut state = ToolLoopState::default();
    record_search_call(&mut state, &loaded, "B");
    let next = build_next_request(&mut state, loaded);

    let ResponseInput::Items(items) = &next.input else {
        panic!("the next request's input is an item list");
    };
    assert_eq!(
        shape(items),
        [
            "message:user",
            "function_call:call_A",
            "function_call_output:call_A",
            "function_call:call_B+output",
        ]
    );
    assert_eq!(next.conversation, None);
    assert_eq!(next.store, Some(false));
}

/// The rebuild carries the loaded items themselves: the loaded message's
/// content buffer sits at the same address in the next request (moved,
/// not copied), the call this loop executed is moved in behind it, and
/// the state keeps no copy of the input. Pinned so the next change that
/// clones the input into the state, or rebuilds the request from that
/// copy, fails here instead of restoring a per-request copy of a whole
/// thread that the bed's counts would show only as drift.
#[test]
fn the_rebuild_moves_the_loaded_items_instead_of_copying_them() {
    let loaded = one_message_request();
    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded request's input is an item list");
    };
    let ResponseInputOutputItem::Message { content, .. } = &items[0] else {
        panic!("the loaded message leads the loaded input");
    };
    let content_before = content.as_ptr();

    let mut state = ToolLoopState::default();
    record_search_call(&mut state, &loaded, "B");
    let next = build_next_request(&mut state, loaded);

    let ResponseInput::Items(items) = &next.input else {
        panic!("the next request's input is an item list");
    };
    assert_eq!(
        shape(items),
        ["message:user", "function_call:call_B+output"]
    );
    let ResponseInputOutputItem::Message { content, .. } = &items[0] else {
        panic!("the loaded message leads the next request's input");
    };
    assert!(
        std::ptr::eq(content.as_ptr(), content_before),
        "the loaded message's content was copied into the next request instead of moved"
    );
    assert!(
        state.calls_to_carry.is_empty(),
        "the executed call was copied into the next request instead of moved"
    );
}

/// An earlier iteration's calls stay in every later request: the second
/// rebuild takes the first rebuild's output as it is and moves only the
/// calls recorded since in behind it, so the loaded items are followed by
/// both calls in order and the state carries nothing over. Pinned so a
/// rebuild from anything but the previous request (a fresh copy of the
/// loaded input, say) fails here instead of dropping the earlier calls.
#[test]
fn calls_recorded_in_earlier_iterations_stay_in_every_later_request() {
    let loaded = one_message_request();
    let mut state = ToolLoopState::default();

    record_search_call(&mut state, &loaded, "B");
    let second = build_next_request(&mut state, loaded);
    record_search_call(&mut state, &second, "C");
    let third = build_next_request(&mut state, second);

    let ResponseInput::Items(items) = &third.input else {
        panic!("the third request's input is an item list");
    };
    assert_eq!(
        shape(items),
        [
            "message:user",
            "function_call:call_B+output",
            "function_call:call_C+output",
        ]
    );
    assert!(
        state.calls_to_carry.is_empty(),
        "a call recorded before a build stayed in the state"
    );
}

/// A chain longer than the store keeps reloads as its newest turns, and
/// the oldest turn kept opens with the result of a call the dropped turn
/// issued: the loader leaves that result out, so a long tool loop keeps
/// converting under the contract profiles instead of being refused, from
/// then on, for a call the client never sent in the request.
#[tokio::test]
async fn a_chain_cut_at_the_stores_depth_cap_converts_without_the_dropped_calls_result() {
    use openai_protocol::responses::StringOrContentParts;
    use serde_json::json;

    let stores = Stores::with_conversation().await;
    let call = |k: usize| {
        json!([{
            "type": "function_call", "id": format!("fc_{k}"), "call_id": format!("call_{k}"),
            "name": "calc", "arguments": "{}", "status": "completed"
        }])
    };
    let result = |k: usize| {
        ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
            id: None,
            call_id: Some(format!("call_{k}")),
            caller: None,
            name: None,
            namespace: None,
            output: StringOrContentParts::String("ok".to_string()),
            status: None,
        }])
    };
    // Turn 1 asks and gets call_1; turn k answers call_{k-1} and gets
    // call_k: 101 turns, one more than the memory store's chain keeps.
    stores
        .store_turn(
            "resp_1",
            &ResponsesRequest {
                model: "deepseek-flash".to_string(),
                input: ResponseInput::Text("Start.".to_string()),
                ..Default::default()
            },
            call(1),
        )
        .await;
    for k in 2..=101 {
        stores
            .store_turn(
                &format!("resp_{k}"),
                &ResponsesRequest {
                    model: "deepseek-flash".to_string(),
                    input: result(k - 1),
                    previous_response_id: Some(format!("resp_{}", k - 1)),
                    ..Default::default()
                },
                call(k),
            )
            .await;
    }
    let turn_102 = ResponsesRequest {
        model: "deepseek-flash".to_string(),
        input: result(101),
        previous_response_id: Some("resp_101".to_string()),
        ..Default::default()
    };

    let loaded = load_history(stores.history(), &turn_102, None, ReplayInto::Chat)
        .await
        .expect("the chain loads");

    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    let shapes = shape(items);
    // resp_2 to resp_101 were kept; resp_2's input, the result of call_1,
    // answers a call that is gone and is left out.
    assert_eq!(
        shapes.first().map(String::as_str),
        Some("function_call:call_2")
    );
    assert!(!shapes.iter().any(|s| s == "function_call_output:call_1"));
    assert_eq!(shapes.len(), 200);
    assert_eq!(
        shapes.last().map(String::as_str),
        Some("function_call_output:call_101")
    );
    let chat = conversions::responses_to_chat(&loaded)
        .expect("the reloaded chain converts under the contract");
    assert!(matches!(
        chat.messages.last(),
        Some(ChatMessage::Tool { tool_call_id, .. }) if tool_call_id == "call_101"
    ));
}

/// A client's own tool result with no call before it, sent with a
/// `previous_response_id`, is not the loader's to drop: it stays after the
/// chain's items in the loaded input, and the contract profile refuses it
/// as the public API does. (The loader judges the chain's items alone.)
#[tokio::test]
async fn a_clients_own_result_after_a_chain_keeps_its_refusal() {
    use openai_protocol::responses::StringOrContentParts;
    use serde_json::json;

    let stores = Stores::with_conversation().await;
    stores
        .store_turn(
            "resp_1",
            &ResponsesRequest {
                model: "deepseek-flash".to_string(),
                input: ResponseInput::Text("Start.".to_string()),
                ..Default::default()
            },
            json!([{
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "calc", "arguments": "{}", "status": "completed"
            }]),
        )
        .await;
    let turn_2 = ResponsesRequest {
        model: "deepseek-flash".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
            id: None,
            call_id: Some("call_unknown".to_string()),
            caller: None,
            name: None,
            namespace: None,
            output: StringOrContentParts::String("ok".to_string()),
            status: None,
        }]),
        previous_response_id: Some("resp_1".to_string()),
        ..Default::default()
    };

    let loaded = load_history(stores.history(), &turn_2, None, ReplayInto::Chat)
        .await
        .expect("the chain loads");

    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    let shapes = shape(items);
    assert_eq!(
        shapes.last().map(String::as_str),
        Some("function_call_output:call_unknown"),
        "{shapes:?}"
    );
    let refused = conversions::responses_to_chat(&loaded)
        .expect_err("a result no call before it issued is a malformed history")
        .to_string();
    assert!(refused.contains("call_unknown"), "{refused}");
}

/// A stored turn keeps its `item_reference`s as the client sent them, so
/// the call a stored result answers may sit behind a reference: the
/// loader resolves the references first and judges the results after,
/// and the pair reaches the engine intact.
#[tokio::test]
async fn a_stored_result_whose_call_sits_behind_a_reference_keeps_its_call() {
    use openai_protocol::responses::StringOrContentParts;
    use serde_json::json;

    let stores = Stores::with_conversation().await;
    // resp_1: the question, answered by call_1 (item fc_1).
    stores
        .store_turn(
            "resp_1",
            &ResponsesRequest {
                model: "deepseek-flash".to_string(),
                input: ResponseInput::Text("Start.".to_string()),
                ..Default::default()
            },
            json!([{
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "calc", "arguments": "{}", "status": "completed"
            }]),
        )
        .await;
    // resp_2, no chain: the client references the call and sends its
    // result; the turn is stored with the reference unresolved.
    let reference: ResponseInputOutputItem =
        serde_json::from_value(json!({"type": "item_reference", "id": "fc_1"}))
            .expect("an item reference");
    stores
        .store_turn(
            "resp_2",
            &ResponsesRequest {
                model: "deepseek-flash".to_string(),
                input: ResponseInput::Items(vec![
                    reference,
                    ResponseInputOutputItem::FunctionCallOutput {
                        id: None,
                        call_id: Some("call_1".to_string()),
                        caller: None,
                        name: None,
                        namespace: None,
                        output: StringOrContentParts::String("42".to_string()),
                        status: None,
                    },
                ]),
                ..Default::default()
            },
            json!([{
                "type": "message", "id": "msg_2", "role": "assistant",
                "content": [{"type": "output_text", "text": "42 it is.", "annotations": []}]
            }]),
        )
        .await;
    let turn_3 = ResponsesRequest {
        model: "deepseek-flash".to_string(),
        input: ResponseInput::Text("Thanks.".to_string()),
        previous_response_id: Some("resp_2".to_string()),
        ..Default::default()
    };

    let loaded = load_history(stores.history(), &turn_3, None, ReplayInto::Chat)
        .await
        .expect("the chain loads");

    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    assert_eq!(
        shape(items),
        [
            "function_call:call_1",
            "function_call_output:call_1",
            "message:assistant",
            "message:user"
        ]
    );
    let chat =
        conversions::responses_to_chat(&loaded).expect("the pair converts under the contract");
    assert!(chat.messages.iter().any(|message| matches!(
        message,
        ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "call_1"
    )));
}
