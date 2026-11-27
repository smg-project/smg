use axum::http;
use openai_protocol::{common::ConversationRef, responses::StringOrContentParts};
use serde_json::json;
use smg_data_connector::{
    AddedAtSequence, ConversationItemId, MemoryConversationItemStorage, MemoryConversationStorage,
    MemoryResponseStorage, NewConversation, NewConversationItem,
};

use super::*;
use crate::routers::common::persistence_utils::persist_conversation_items;

/// The stores of one gateway, in memory, with the conversation `conv_1`.
pub(crate) struct Stores {
    pub responses: Arc<dyn ResponseStorage>,
    pub conversations: Arc<dyn ConversationStorage>,
    pub conversation_items: Arc<dyn ConversationItemStorage>,
}

impl Stores {
    pub(crate) async fn with_conversation() -> Self {
        let stores = Self {
            responses: Arc::new(MemoryResponseStorage::new()),
            conversations: Arc::new(MemoryConversationStorage::new()),
            conversation_items: Arc::new(MemoryConversationItemStorage::new()),
        };
        stores
            .conversations
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_1")),
                metadata: None,
                tenant: None,
            })
            .await
            .expect("the conversation is created");
        stores
    }

    pub(crate) fn history(&self) -> HistoryStores<'_> {
        HistoryStores {
            responses: &self.responses,
            conversations: self.conversations.as_ref(),
            conversation_items: self.conversation_items.as_ref(),
        }
    }

    /// Stores a turn as the gateway persists one: `request` answered by
    /// a response of `output` items under `response_id`.
    pub(crate) async fn store_turn(
        &self,
        response_id: &str,
        request: &ResponsesRequest,
        output: Value,
    ) {
        let response = json!({
            "id": response_id,
            "object": "response",
            "model": "m",
            "status": "completed",
            "output": output,
        });
        persist_conversation_items(
            self.conversations.clone(),
            self.conversation_items.clone(),
            self.responses.clone(),
            &response,
            request,
            None,
            None,
        )
        .await
        .expect("the turn is stored");
    }
}

/// Turn 1 of the tests: the user's question, answered by one function
/// call the client is to run.
pub(crate) fn function_call_output() -> Value {
    json!([{
        "type": "function_call", "id": "fc_1", "call_id": "call_A",
        "name": "calc", "arguments": "{\"expr\":\"6*7\"}", "status": "completed"
    }])
}

/// Turn 2 of the tests: the client answers call_A.
pub(crate) fn function_call_output_input() -> ResponseInput {
    ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
        id: None,
        call_id: Some("call_A".to_string()),
        caller: None,
        name: None,
        namespace: None,
        output: StringOrContentParts::String("42".to_string()),
        status: None,
    }])
}

/// The kinds of the items, with the call id of a tool item, for the
/// order assertions.
pub(crate) fn shape(items: &[ResponseInputOutputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| match item {
            ResponseInputOutputItem::Message { role, .. } => format!("message:{role}"),
            ResponseInputOutputItem::FunctionToolCall {
                call_id, output, ..
            } => format!(
                "function_call:{call_id}{}",
                if output.is_some() { "+output" } else { "" }
            ),
            ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => {
                format!("function_call_output:{}", call_id.as_deref().unwrap_or(""))
            }
            ResponseInputOutputItem::Reasoning { .. } => "reasoning".to_string(),
            other => serde_json::to_value(other).expect("an item serializes")["type"]
                .as_str()
                .unwrap_or("?")
                .to_string(),
        })
        .collect()
}

/// The part types of a loaded message item, for the replay assertions.
fn part_types(item: &ResponseInputOutputItem) -> Vec<String> {
    let ResponseInputOutputItem::Message { content, .. } = item else {
        panic!("a message item: {item:?}");
    };
    content
        .iter()
        .map(|part| {
            serde_json::to_value(part).expect("a part serializes")["type"]
                .as_str()
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

/// A stored `developer` turn whose message carried an `input_image` with
/// a URL replays without it on the chat path too: the developer turn
/// folded into the system text when it was answered, and the fold refuses
/// such a part now, so a continuation refused on a turn the client cannot
/// edit would end the conversation. The request's own developer item keeps
/// its parts and meets the refusal itself.
#[tokio::test]
async fn a_stored_developer_image_is_left_out_of_the_replay() {
    let stores = Stores::with_conversation().await;
    let developer_turn: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "message", "id": "msg_d1", "role": "developer", "content": [
            {"type": "input_text", "text": "Describe what you see."},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ]
    }))
    .expect("a developer message item");
    let turn_1 = ResponsesRequest {
        input: ResponseInput::Items(vec![developer_turn.clone()]),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    stores
        .store_turn(
            "resp_1",
            &turn_1,
            json!([{"type": "message", "id": "msg_1", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Nothing to see.", "annotations": []}]}]),
        )
        .await;

    for into in [ReplayInto::Chat, ReplayInto::Harmony] {
        for continuation in [
            ResponsesRequest {
                input: ResponseInput::Items(vec![developer_turn.clone()]),
                previous_response_id: Some("resp_1".to_string()),
                ..Default::default()
            },
            ResponsesRequest {
                input: ResponseInput::Items(vec![developer_turn.clone()]),
                conversation: Some(ConversationRef::Id("conv_1".to_string())),
                ..Default::default()
            },
        ] {
            let loaded = load_history(stores.history(), &continuation, None, into)
                .await
                .expect("the history loads");
            let ResponseInput::Items(items) = &loaded.input else {
                panic!("the loaded input is an item list");
            };
            assert_eq!(
                shape(items),
                [
                    "message:developer",
                    "message:assistant",
                    "message:developer"
                ]
            );
            assert_eq!(
                part_types(&items[0]),
                ["input_text"],
                "{into:?}: the stored developer turn replays its text alone"
            );
            assert_eq!(
                part_types(&items[2]),
                ["input_text", "input_image"],
                "{into:?}: the request's own input keeps every part"
            );
        }
    }
}

/// A stored turn whose message carried a part the pipeline never took
/// (an `input_file`; an `input_image` without a URL; on the Harmony path
/// any `input_image`) replays without that part, on the chain and in the
/// conversation alike: the turn was answered without it, and a chain
/// cannot be edited. The request's own input is not filtered.
#[tokio::test]
async fn a_stored_part_the_pipeline_never_carried_is_left_out_of_the_replay() {
    let stores = Stores::with_conversation().await;
    let user_turn: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "message", "id": "msg_u1", "role": "user", "content": [
            {"type": "input_text", "text": "Read this and look at that."},
            {"type": "input_file", "filename": "notes.txt", "file_data": "aGVsbG8="},
            {"type": "input_image", "file_id": "file_1"},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ]
    }))
    .expect("a user message item");
    let turn_1 = ResponsesRequest {
        input: ResponseInput::Items(vec![user_turn.clone()]),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    stores
        .store_turn(
            "resp_1",
            &turn_1,
            json!([{"type": "message", "id": "msg_1", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Read.", "annotations": []}]}]),
        )
        .await;

    for (into, replayed) in [
        (ReplayInto::Chat, vec!["input_text", "input_image"]),
        (ReplayInto::Harmony, vec!["input_text"]),
    ] {
        for continuation in [
            ResponsesRequest {
                input: ResponseInput::Items(vec![user_turn.clone()]),
                previous_response_id: Some("resp_1".to_string()),
                ..Default::default()
            },
            ResponsesRequest {
                input: ResponseInput::Items(vec![user_turn.clone()]),
                conversation: Some(ConversationRef::Id("conv_1".to_string())),
                ..Default::default()
            },
        ] {
            let loaded = load_history(stores.history(), &continuation, None, into)
                .await
                .expect("the history loads");
            let ResponseInput::Items(items) = &loaded.input else {
                panic!("the loaded input is an item list");
            };
            assert_eq!(
                shape(items),
                ["message:user", "message:assistant", "message:user"]
            );
            assert_eq!(part_types(&items[0]), replayed, "{into:?}: the stored turn");
            assert_eq!(
                part_types(&items[2]),
                ["input_text", "input_file", "input_image", "input_image"],
                "{into:?}: the request's own input keeps every part"
            );
        }
    }
}

fn turn_1_in_conversation() -> ResponsesRequest {
    ResponsesRequest {
        input: ResponseInput::Text("What is 6*7?".to_string()),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_conversation_reload_keeps_the_stored_turns_function_call_and_the_output_in_order() {
    let stores = Stores::with_conversation().await;
    stores
        .store_turn("resp_1", &turn_1_in_conversation(), function_call_output())
        .await;
    let turn_2 = ResponsesRequest {
        input: function_call_output_input(),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };

    let loaded = load_history(stores.history(), &turn_2, None, ReplayInto::Chat)
        .await
        .expect("the conversation loads");

    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    assert_eq!(
        shape(items),
        [
            "message:user",
            "function_call:call_A",
            "function_call_output:call_A"
        ]
    );
    assert!(matches!(
        &items[1],
        ResponseInputOutputItem::FunctionToolCall { name, arguments, .. }
            if name == "calc" && arguments == "{\"expr\":\"6*7\"}"
    ));
}

/// A stored row of a kind no conversion replays (a hosted image
/// generation the gateway wrote, the client's approval turn) is left out
/// when the history loads, through `conversation` and through
/// `previous_response_id` alike, so the continuation goes on instead of
/// failing on the row until the client deletes it.
#[tokio::test]
async fn stored_kinds_no_conversion_replays_are_left_out_of_the_history() {
    let stores = Stores::with_conversation().await;
    stores
        .store_turn(
            "resp_1",
            &turn_1_in_conversation(),
            json!([
                {"type": "image_generation_call", "id": "ig_1", "status": "completed",
                 "result": "aGVsbG8="},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Drawn.", "annotations": []}]}
            ]),
        )
        .await;
    let approval: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true
    }))
    .expect("an approval response item");
    // Turn 2 continues the conversation and the chain alike, so both
    // loaders walk past the rows they do not replay.
    let turn_2 = ResponsesRequest {
        input: ResponseInput::Items(vec![approval]),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        previous_response_id: Some("resp_1".to_string()),
        ..Default::default()
    };
    stores
        .store_turn(
            "resp_2",
            &turn_2,
            json!([
                {"type": "message", "id": "msg_2", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Approved.", "annotations": []}]}
            ]),
        )
        .await;
    let expected = [
        "message:user",
        "message:assistant",
        "message:assistant",
        "message:user",
    ];

    let turn_3 = ResponsesRequest {
        input: ResponseInput::Text("And now?".to_string()),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    let loaded = load_history(stores.history(), &turn_3, None, ReplayInto::Chat)
        .await
        .expect("the conversation loads past the rows it does not replay");
    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    assert_eq!(shape(items), expected);

    let turn_3 = ResponsesRequest {
        input: ResponseInput::Text("And now?".to_string()),
        previous_response_id: Some("resp_2".to_string()),
        ..Default::default()
    };
    let loaded = load_history(stores.history(), &turn_3, None, ReplayInto::Chat)
        .await
        .expect("the chain loads past the items it does not replay");
    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    assert_eq!(shape(items), expected);
}

/// A stored hosted call replays its output; failing that, its failure
/// as text (never a successful empty result for a call that failed); a
/// call that has not run replays no result.
#[test]
fn a_stored_hosted_call_replays_its_output_else_its_failure() {
    let execution = McpToolCallError::execution("wiki is down");
    let http = McpToolCallError::HttpError {
        code: 503,
        message: "unavailable".to_string(),
    };
    assert_eq!(
        hosted_call_result(Some("found"), None, Some("completed")).as_deref(),
        Some("found")
    );
    assert_eq!(
        hosted_call_result(Some("found"), Some(&execution), Some("failed")).as_deref(),
        Some("found"),
        "an output that is there is the result"
    );
    assert_eq!(
        hosted_call_result(Some(""), Some(&execution), Some("failed")).as_deref(),
        Some("The tool call failed: wiki is down")
    );
    assert_eq!(
        hosted_call_result(None, Some(&http), None).as_deref(),
        Some("The tool call failed (HTTP 503): unavailable")
    );
    assert_eq!(
        hosted_call_result(None, None, Some("failed")).as_deref(),
        Some("The tool call failed.")
    );
    assert_eq!(hosted_call_result(None, None, Some("in_progress")), None);
    assert_eq!(
        hosted_call_result(Some(""), None, Some("completed")).as_deref(),
        Some(""),
        "a completed call with an empty output keeps it"
    );
}

#[tokio::test]
async fn a_previous_response_reload_keeps_the_stored_turns_function_call_and_the_output_in_order() {
    let stores = Stores::with_conversation().await;
    let turn_1 = ResponsesRequest {
        input: ResponseInput::Text("What is 6*7?".to_string()),
        ..Default::default()
    };
    stores
        .store_turn("resp_1", &turn_1, function_call_output())
        .await;
    let turn_2 = ResponsesRequest {
        input: function_call_output_input(),
        previous_response_id: Some("resp_1".to_string()),
        ..Default::default()
    };

    let loaded = load_history(stores.history(), &turn_2, None, ReplayInto::Chat)
        .await
        .expect("the chain loads");

    assert_eq!(loaded.previous_response_id, None);
    let ResponseInput::Items(items) = &loaded.input else {
        panic!("the loaded input is an item list");
    };
    assert_eq!(
        shape(items),
        [
            "message:user",
            "function_call:call_A",
            "function_call_output:call_A"
        ]
    );
}

/// Turn 3 after the client's answer was stored: the whole exchange comes
/// back, the call before its output, through both continuations.
#[tokio::test]
async fn a_stored_function_call_output_reloads_after_its_call() {
    let stores = Stores::with_conversation().await;
    stores
        .store_turn("resp_1", &turn_1_in_conversation(), function_call_output())
        .await;
    let turn_2 = ResponsesRequest {
        input: function_call_output_input(),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        previous_response_id: Some("resp_1".to_string()),
        ..Default::default()
    };
    stores
        .store_turn(
            "resp_2",
            &turn_2,
            json!([{
                "type": "message", "id": "msg_a2", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "42", "annotations": []}]
            }]),
        )
        .await;
    let expected = [
        "message:user",
        "function_call:call_A",
        "function_call_output:call_A",
        "message:assistant",
        "message:user",
    ];

    for turn_3 in [
        ResponsesRequest {
            input: ResponseInput::Text("Thanks".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        },
        ResponsesRequest {
            input: ResponseInput::Text("Thanks".to_string()),
            previous_response_id: Some("resp_2".to_string()),
            ..Default::default()
        },
    ] {
        let loaded = load_history(stores.history(), &turn_3, None, ReplayInto::Chat)
            .await
            .expect("the history loads");
        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(shape(items), expected, "{:?}", turn_3.conversation);
    }
}

/// A function call added through the conversations API is stored whole,
/// its id on the row: it reloads under that id.
#[tokio::test]
async fn a_function_call_added_to_the_conversation_reloads_under_the_rows_id() {
    let stores = Stores::with_conversation().await;
    let conv_id = ConversationId::from("conv_1");
    let row = stores
        .conversation_items
        .create_item(NewConversationItem {
            id: None,
            response_id: None,
            item_type: "function_call".to_string(),
            role: None,
            content: json!({
                "type": "function_call", "call_id": "call_B", "name": "calc",
                "arguments": "{}", "status": "completed"
            }),
            status: Some("completed".to_string()),
            tenant: None,
        })
        .await
        .expect("the row is created");
    stores
        .conversation_items
        .link_item(&conv_id, &row.id, chrono::Utc::now())
        .await
        .expect("the row is linked");
    let items = replayed_items(&stores, "go on").await;
    assert_eq!(shape(&items), ["function_call:call_B", "message:user"]);
    assert_eq!(input_item_id(&items[0]), Some(row.id.0.as_str()));
}

/// A message the client added to the conversation itself (no response
/// wrote it) keeps every part when the history loads, so a generation
/// over that conversation meets the conversion's refusal instead of a
/// silent trim; the same parts in a turn a stored response wrote are
/// left out as never carried.
#[tokio::test]
async fn a_conversation_item_the_client_added_keeps_its_parts_for_the_refusal() {
    let stores = Stores::with_conversation().await;
    let conv_id = ConversationId::from("conv_1");
    let parts = json!([
        {"type": "input_text", "text": "Read this."},
        {"type": "input_file", "filename": "notes.txt", "file_data": "aGVsbG8="},
        {"type": "input_image", "file_id": "file_1"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
    ]);
    for (response_id, expected) in [
        (
            None,
            vec!["input_text", "input_file", "input_image", "input_image"],
        ),
        (
            Some("resp_0".to_string()),
            vec!["input_text", "input_image"],
        ),
    ] {
        let row = stores
            .conversation_items
            .create_item(NewConversationItem {
                id: None,
                response_id: response_id.clone(),
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: parts.clone(),
                status: Some("completed".to_string()),
                tenant: None,
            })
            .await
            .expect("the row is created");
        stores
            .conversation_items
            .link_item(&conv_id, &row.id, chrono::Utc::now())
            .await
            .expect("the row is linked");
        let items = replayed_items(&stores, "go on").await;
        let loaded = items
            .iter()
            .find(|item| input_item_id(item) == Some(row.id.0.as_str()))
            .expect("the row reloads");
        assert_eq!(part_types(loaded), expected, "response_id {response_id:?}");
    }
}

/// The items a request that continues `conv_1` with the message `text`
/// replays: the loaded history's input, an item list.
async fn replayed_items(stores: &Stores, text: &str) -> Vec<ResponseInputOutputItem> {
    let request = ResponsesRequest {
        input: ResponseInput::Text(text.to_string()),
        conversation: Some(ConversationRef::Id("conv_1".to_string())),
        ..Default::default()
    };
    let loaded = load_history(stores.history(), &request, None, ReplayInto::Chat)
        .await
        .expect("the conversation loads");
    let ResponseInput::Items(items) = loaded.input else {
        panic!("the loaded input is an item list");
    };
    items
}

/// Links `rows` to `conv_1` in order, each stamped later than the one
/// before, as the conversations API does.
async fn link_rows(stores: &Stores, rows: Vec<NewConversationItem>) {
    let conv_id = ConversationId::from("conv_1");
    let mut stamps = AddedAtSequence::new();
    for row in rows {
        let created = stores
            .conversation_items
            .create_item(row)
            .await
            .expect("the row is created");
        stores
            .conversation_items
            .link_item(&conv_id, &created.id, stamps.next_added_at())
            .await
            .expect("the row is linked");
    }
}

/// Message `n` of a conversation: a user's for an odd `n`, the
/// assistant's for an even one, its text `item n`.
fn message_row(n: usize) -> NewConversationItem {
    let (role, content) = if n % 2 == 1 {
        (
            "user",
            json!([{"type": "input_text", "text": format!("item {n}")}]),
        )
    } else {
        (
            "assistant",
            json!([{"type": "output_text", "text": format!("item {n}"), "annotations": []}]),
        )
    };
    NewConversationItem {
        id: None,
        response_id: None,
        item_type: "message".to_string(),
        role: Some(role.to_string()),
        content,
        status: Some("completed".to_string()),
        tenant: None,
    }
}

fn call_row(call_id: &str) -> NewConversationItem {
    NewConversationItem {
        id: None,
        response_id: None,
        item_type: "function_call".to_string(),
        role: None,
        content: json!({
            "type": "function_call", "call_id": call_id, "name": "calc",
            "arguments": "{}", "status": "completed"
        }),
        status: Some("completed".to_string()),
        tenant: None,
    }
}

fn output_row(call_id: &str) -> NewConversationItem {
    NewConversationItem {
        id: None,
        response_id: None,
        item_type: "function_call_output".to_string(),
        role: None,
        content: json!({"type": "function_call_output", "call_id": call_id, "output": "42"}),
        status: Some("completed".to_string()),
        tenant: None,
    }
}

/// The text of each message item, for the window assertions.
fn texts(items: &[ResponseInputOutputItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match item {
            ResponseInputOutputItem::Message { content, .. } => {
                content.iter().find_map(|part| match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                    _ => None,
                })
            }
            _ => None,
        })
        .collect()
}

/// A call and its output replay together or not at all: when the caller
/// may not see one of the two, the other goes with it, as the output of
/// a call before a cut window does. A second output of the call takes
/// nothing with it: the call keeps the output the caller may see.
#[tokio::test]
async fn a_tool_row_the_caller_may_not_see_takes_its_pair_out_of_the_replay() {
    for hide_call in [true, false] {
        let stores = Stores::with_conversation().await;
        let (mut call, mut output) = (call_row("call_other"), output_row("call_other"));
        let hidden = if hide_call { &mut call } else { &mut output };
        hidden.tenant = Some("other".to_string());
        link_rows(&stores, vec![message_row(1), call, output]).await;
        let items = replayed_items(&stores, "item 2").await;
        assert_eq!(
            shape(&items),
            ["message:user", "message:user"],
            "hide_call {hide_call}"
        );
    }

    let stores = Stores::with_conversation().await;
    let mut second = output_row("call_own");
    second.tenant = Some("other".to_string());
    let rows = vec![
        message_row(1),
        call_row("call_own"),
        output_row("call_own"),
        second,
    ];
    link_rows(&stores, rows).await;
    assert_eq!(
        shape(&replayed_items(&stores, "item 2").await),
        [
            "message:user",
            "function_call:call_own",
            "function_call_output:call_own",
            "message:user"
        ]
    );
}

/// A call the conversation holds as an `item_reference` row pairs as a
/// stored call does: it replays with its output behind a cut and beside
/// a row the caller may not see, and not at all when its output is such
/// a row. An output before the reference, of an older call of the id
/// that the cut left out, stays out, behind another reference too.
#[tokio::test]
async fn a_call_the_conversation_holds_by_reference_pairs_with_its_output() {
    let pair = ["function_call:call_A", "function_call_output:call_A"];
    for (case, messages, hidden, older_output, tool_items) in [
        ("a cut window", 110, "", false, &pair[..]),
        ("another row is hidden", 1, "message", false, &pair[..]),
        ("the output is hidden", 1, "output", false, &pair[..0]),
        ("an older output in a cut window", 110, "", true, &pair[..]),
    ] {
        let stores = Stores::with_conversation().await;
        let first = ResponsesRequest {
            input: ResponseInput::Text("q".to_string()),
            ..Default::default()
        };
        stores
            .store_turn("resp_1", &first, function_call_output())
            .await;
        let reference = |id: &str| NewConversationItem {
            id: Some(ConversationItemId::from(id)),
            response_id: None,
            item_type: "item_reference".to_string(),
            role: None,
            content: json!({"type": "item_reference", "id": id}),
            status: None,
            tenant: None,
        };
        let mut rows: Vec<_> = (1..=messages).map(message_row).collect();
        let mut output = output_row("call_A");
        match hidden {
            "message" => rows[0].tenant = Some("other".to_string()),
            "output" => output.tenant = Some("other".to_string()),
            _ => {}
        }
        if older_output {
            // Behind another reference, to the message before it: the
            // first reference of the window is not the call's.
            rows[messages - 1].id = Some(ConversationItemId::from("msg_last"));
            rows.push(NewConversationItem {
                id: None,
                ..reference("msg_last")
            });
            rows.push(output_row("call_A"));
        }
        rows.extend([reference("fc_1"), output]);
        if messages > 100 {
            // First in the cut window, before its turn boundary: a
            // reference to nothing is dropped there, never looked up.
            rows.insert(rows.len() - 99, reference("gone"));
        }
        link_rows(&stores, rows).await;
        let kinds = shape(&replayed_items(&stores, "next").await);
        let replayed: Vec<&str> = kinds
            .iter()
            .map(String::as_str)
            .filter(|kind| kind.starts_with("function_call"))
            .collect();
        assert_eq!(replayed, tool_items, "{case}: {kinds:?}");
    }
}

/// A conversation of 120 items replays its newest 100, oldest first,
/// then the request's input: the first 20 are not in the prompt.
#[tokio::test]
async fn a_conversation_longer_than_the_limit_replays_its_newest_items() {
    let stores = Stores::with_conversation().await;
    link_rows(&stores, (1..=120).map(message_row).collect()).await;
    let items = replayed_items(&stores, "item 121").await;
    let expected: Vec<String> = (21..=121).map(|n| format!("item {n}")).collect();
    assert_eq!(texts(&items), expected);
}

/// A conversation one item longer than the limit, whose first items are
/// two calls with their outputs: its window begins at the first output,
/// inside a turn whose request is before the window. The window replays
/// from its first client message; the output of the call before it and
/// the call with its output after it go with the turn the cut split.
#[tokio::test]
async fn a_cut_window_begins_at_its_first_client_message() {
    let stores = Stores::with_conversation().await;
    let mut rows = vec![
        call_row("call_A"),
        output_row("call_A"),
        call_row("call_B"),
        output_row("call_B"),
    ];
    rows.extend((5..=101).map(message_row));
    link_rows(&stores, rows).await;
    let items = replayed_items(&stores, "item 102").await;
    let kinds = shape(&items);
    assert_eq!(kinds[0], "message:user", "{kinds:?}");
    assert!(
        !kinds.iter().any(|kind| kind.starts_with("function_call")),
        "{kinds:?}"
    );
    // the calls and outputs are the tail of the turn the cut split: the
    // messages replay, then the request
    assert_eq!(items.len(), 101 - 4 + 1);
    let expected: Vec<String> = (5..=102).map(|n| format!("item {n}")).collect();
    assert_eq!(texts(&items), expected);
}

/// A conversation of exactly the limit is not cut: every item replays,
/// an output whose call is not among them included (the window did not
/// begin after its call; a builder that refuses such an output says so).
#[tokio::test]
async fn a_conversation_of_exactly_the_limit_replays_whole() {
    let stores = Stores::with_conversation().await;
    let mut rows = vec![output_row("call_A")];
    rows.extend((2..=100).map(message_row));
    link_rows(&stores, rows).await;
    let items = replayed_items(&stores, "item 101").await;
    let kinds = shape(&items);
    assert_eq!(kinds[0], "function_call_output:call_A", "{kinds:?}");
    assert_eq!(items.len(), 100 + 1);
    assert_eq!(kinds.last().map(String::as_str), Some("message:user"));
}

/// A message row keeps the phase label it was stored with, and a reasoning
/// row added through the conversations API (bare content) comes back as
/// the reasoning item.
#[test]
fn message_and_reasoning_rows_rebuild_their_items() {
    let row = |item_type: &str, role: Option<&str>, content: Value| ConversationItem {
        id: ConversationItemId::from(format!("{item_type}_1").as_str()),
        response_id: None,
        item_type: item_type.to_string(),
        role: role.map(str::to_string),
        content,
        status: Some("completed".to_string()),
        created_at: chrono::Utc::now(),
        tenant: None,
    };

    let message = conversation_item_as_input(row(
        "message",
        Some("assistant"),
        json!({"content": [{"type": "output_text", "text": "hi", "annotations": []}], "phase": "final_answer"}),
    ))
    .expect("a message row rebuilds");
    assert!(matches!(
        message,
        ResponseInputOutputItem::Message { ref id, ref role, phase: Some(openai_protocol::responses::MessagePhase::FinalAnswer), .. }
            if id == "message_1" && role == "assistant"
    ));

    let reasoning = conversation_item_as_input(row(
        "reasoning",
        None,
        json!([{"type": "reasoning_text", "text": "thinking"}]),
    ))
    .expect("a reasoning row rebuilds");
    assert!(matches!(
        reasoning,
        ResponseInputOutputItem::Reasoning { ref id, ref content, .. }
            if id.as_deref() == Some("reasoning_1") && content.len() == 1
    ));

    // a row with no input form is left out, not an error
    assert!(conversation_item_as_input(row(
        "file_search_call",
        None,
        json!({"type": "file_search_call", "status": "completed"})
    ))
    .is_none());
}

fn stored_response_with_message(item_id: &str, text: &str) -> StoredResponse {
    let mut stored = StoredResponse::new(None);
    stored.raw_response = json!({
        "id": "resp_stored",
        "output": [{
            "type": "message", "id": item_id, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }]
    });
    stored
}

fn reference(id: &str) -> ResponseInputOutputItem {
    serde_json::from_value(json!({"type": "item_reference", "id": id})).unwrap()
}

#[tokio::test]
async fn item_reference_resolves_a_stored_output_item() {
    let storage = MemoryResponseStorage::new();
    storage
        .store_response(stored_response_with_message("msg_stored", "earlier answer"))
        .await
        .unwrap();
    let user = normalize_input_item(
        serde_json::from_value(json!({"role": "user", "content": "continue"})).unwrap(),
    );

    let resolved = resolve_item_references(&storage, vec![reference("msg_stored"), user], None)
        .await
        .expect("a stored output item resolves");

    assert_eq!(resolved.len(), 2);
    match &resolved[0] {
        ResponseInputOutputItem::Message {
            id, role, content, ..
        } => {
            assert_eq!(id, "msg_stored");
            assert_eq!(role, "assistant");
            assert!(matches!(
                &content[0],
                ResponseContentPart::OutputText { text, .. } if text == "earlier answer"
            ));
        }
        other => panic!("expected the stored message, got {other:?}"),
    }
}

#[tokio::test]
async fn an_item_reference_to_an_unstored_response_is_not_found() {
    let storage = MemoryResponseStorage::new();
    let mut unstored = stored_response_with_message("msg_unstored", "a background answer");
    unstored.raw_response["store"] = json!(false);
    storage.store_response(unstored).await.unwrap();

    let refused = resolve_item_references(&storage, vec![reference("msg_unstored")], None)
        .await
        .expect_err("an unstored response's item is no stored item");

    assert_eq!(refused.status(), http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn item_reference_prefers_an_item_already_loaded() {
    let storage = MemoryResponseStorage::new();
    let loaded = ResponseInputOutputItem::Message {
        id: "msg_loaded".to_string(),
        role: "assistant".to_string(),
        content: vec![ResponseContentPart::OutputText {
            text: "from the chain".to_string(),
            annotations: vec![],
            logprobs: None,
        }],
        status: Some("completed".to_string()),
        phase: None,
    };

    let resolved = resolve_item_references(&storage, vec![loaded, reference("msg_loaded")], None)
        .await
        .expect("an item already in the list resolves without storage");

    assert_eq!(resolved.len(), 2);
    assert_eq!(input_item_id(&resolved[1]), Some("msg_loaded"));
    assert!(!matches!(
        resolved[1],
        ResponseInputOutputItem::ItemReference { .. }
    ));
}

#[tokio::test]
async fn item_reference_to_a_stored_item_with_no_input_form_is_400() {
    let storage = MemoryResponseStorage::new();
    let mut stored = StoredResponse::new(None);
    stored.raw_response = json!({
        "id": "resp_stored",
        "output": [{"type": "file_search_call", "id": "fs_1", "status": "completed"}]
    });
    storage.store_response(stored).await.unwrap();

    let response = resolve_item_references(&storage, vec![reference("fs_1")], None)
        .await
        .expect_err("a stored item without an input form is refused");

    assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn item_reference_finds_a_loaded_custom_tool_call_by_its_id() {
    let storage = MemoryResponseStorage::new();
    let loaded: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1",
        "name": "lookup", "input": "weather"
    }))
    .unwrap();

    let resolved = resolve_item_references(&storage, vec![loaded, reference("ctc_1")], None)
        .await
        .expect("an item already in the list resolves without storage");

    assert_eq!(resolved.len(), 2);
    assert_eq!(input_item_id(&resolved[1]), Some("ctc_1"));
}

/// An output item of another tenant's response is as unknown as a missing
/// one; the owner's reference resolves.
#[tokio::test]
async fn item_reference_to_another_tenants_output_item_is_404() {
    let storage = MemoryResponseStorage::new();
    let mut stored = stored_response_with_message("msg_a", "a's answer");
    stored.tenant = Some("header:tenant-a".to_string());
    storage.store_response(stored).await.unwrap();

    for caller in [Some("header:tenant-b"), None] {
        let response = resolve_item_references(&storage, vec![reference("msg_a")], caller)
            .await
            .expect_err("another tenant's item is refused");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND, "{caller:?}");
    }
    let resolved =
        resolve_item_references(&storage, vec![reference("msg_a")], Some("header:tenant-a"))
            .await
            .expect("the owner's reference resolves");
    assert_eq!(input_item_id(&resolved[0]), Some("msg_a"));
}

#[tokio::test]
async fn item_reference_to_an_unknown_id_is_404() {
    let storage = MemoryResponseStorage::new();

    let response = resolve_item_references(&storage, vec![reference("msg_missing")], None)
        .await
        .expect_err("an unknown item is refused");

    assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
}
/// Reasoning step `k` of a turn, as the conversations API stores one.
fn reasoning_row(k: usize) -> NewConversationItem {
    NewConversationItem {
        id: None,
        response_id: None,
        item_type: "reasoning".to_string(),
        role: None,
        content: json!([{"type": "reasoning_text", "text": format!("step {k}")}]),
        status: Some("completed".to_string()),
        tenant: None,
    }
}

/// A conversation of 150 messages replays its newest 100, items 51 to
/// 150 in order, then the request: the cut lands on a user message, a
/// turn boundary, so the window replays whole.
#[tokio::test]
async fn a_cut_that_lands_on_a_turn_boundary_replays_the_newest_items_whole() {
    let stores = Stores::with_conversation().await;
    link_rows(&stores, (1..=150).map(message_row).collect()).await;
    let items = replayed_items(&stores, "item 151").await;
    let expected: Vec<String> = (51..=151).map(|n| format!("item {n}")).collect();
    assert_eq!(texts(&items), expected);
}

/// The rows of a conversation of agent turns, each the user's message,
/// the model's call, its output and the model's answer: 37 whole turns,
/// then a user message and an open call, 150 rows.
fn agent_turn_rows() -> Vec<NewConversationItem> {
    let mut rows = Vec::new();
    for turn in 1..=37 {
        rows.push(message_row(4 * turn - 3));
        rows.push(call_row(&format!("call_{turn}")));
        rows.push(output_row(&format!("call_{turn}")));
        rows.push(message_row(4 * turn));
    }
    rows.push(message_row(149));
    rows.push(call_row("call_38"));
    rows
}

/// A cut that lands inside a turn (row 51 is the output of turn 13's
/// call): the window begins at the next turn boundary, the user's
/// message of turn 14 (row 53); the output and the answer of turn 13
/// before it are not replayed.
#[tokio::test]
async fn a_cut_window_that_begins_inside_a_turn_starts_at_the_next_turn_boundary() {
    let stores = Stores::with_conversation().await;
    link_rows(&stores, agent_turn_rows()).await;
    let items = replayed_items(&stores, "item 151").await;
    let kinds = shape(&items);
    assert_eq!(
        &kinds[..4],
        [
            "message:user",
            "function_call:call_14",
            "function_call_output:call_14",
            "message:assistant"
        ],
        "{kinds:?}"
    );
    assert!(
        !kinds.contains(&"function_call_output:call_13".to_string()),
        "{kinds:?}"
    );
    // rows 53 to 150, then the request
    assert_eq!(items.len(), 150 - 52 + 1);
    let mut expected: Vec<String> = (14..=37)
        .flat_map(|turn| [4 * turn - 3, 4 * turn])
        .map(|n| format!("item {n}"))
        .collect();
    expected.extend(["item 149".to_string(), "item 151".to_string()]);
    assert_eq!(texts(&items), expected);
}

/// One turn longer than the window, the model's reasoning, call and
/// output step after step: the window begins at the first call whose
/// output it holds, with the reasoning that led to it; the output of the
/// step before, whose call is not in the window, is left out.
#[tokio::test]
async fn a_cut_window_with_no_user_message_starts_at_a_call_with_its_output() {
    let stores = Stores::with_conversation().await;
    let mut rows = vec![message_row(1)];
    for step in 1..=50 {
        rows.push(reasoning_row(step));
        rows.push(call_row(&format!("call_{step}")));
        rows.push(output_row(&format!("call_{step}")));
    }
    link_rows(&stores, rows).await;
    let items = replayed_items(&stores, "go on").await;
    let kinds = shape(&items);
    // row 52 of 151 is the output of step 17: the window begins at step 18
    assert_eq!(
        &kinds[..3],
        [
            "reasoning",
            "function_call:call_18",
            "function_call_output:call_18"
        ],
        "{kinds:?}"
    );
    assert!(
        !kinds.contains(&"function_call_output:call_17".to_string()),
        "{kinds:?}"
    );
    assert_eq!(kinds.last().map(String::as_str), Some("message:user"));
    // steps 18 to 50 whole, then the request
    assert_eq!(items.len(), 33 * 3 + 1);
}

/// A first turn that declares tools inline via `additional_tools` keeps
/// them in the stored input; a second turn continuing via
/// either history mechanism with only a `function_call_output` recovers
/// those tools even when the client sends an empty top-level list.
#[tokio::test]
async fn continuations_preserve_inline_additional_tools() {
    let stores = Stores::with_conversation().await;
    let turn_1_req: ResponsesRequest = serde_json::from_value(json!({
        "model": "test-model",
        "store": true,
        "conversation": "conv_1",
        "input": [
            {
                "type": "additional_tools",
                "id": "at_1",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "Run shell command",
                        "parameters": {"type": "object", "properties": {}}
                    }]
                }]
            },
            {
                "type": "agent_message",
                "id": "am_1",
                "author": "/root",
                "recipient": "/root/worker",
                "content": [{"type": "input_text", "text": "Check status"}]
            }
        ]
    }))
    .unwrap();
    let turn_1_resp =
        openai_protocol::responses::ResponsesResponse::builder("resp_turn_1", "test-model")
            .output(vec![
                openai_protocol::responses::ResponseOutputItem::FunctionToolCall {
                    id: Some("fc_1".to_string()),
                    call_id: "call_1".to_string(),
                    name: "exec_command".to_string(),
                    namespace: None,
                    arguments: "{}".to_string(),
                    output: None,
                    status: "completed".to_string(),
                },
            ])
            .build();

    super::super::utils::persist_response_if_needed(
        stores.conversations.clone(),
        stores.conversation_items.clone(),
        stores.responses.clone(),
        &turn_1_resp,
        &turn_1_req,
        None,
        None,
    )
    .await
    .expect("turn 1 persists");

    for (field, id) in [
        ("previous_response_id", "resp_turn_1"),
        ("conversation", "conv_1"),
    ] {
        let body = json!({
            (field): id,
            "model": "test-model",
            "tools": [],
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}]
        });
        let mut turn_2: ResponsesRequest = serde_json::from_value(body).unwrap();

        turn_2.normalize();
        let loaded = load_history(stores.history(), &turn_2, None, ReplayInto::Chat)
            .await
            .expect("turn 2 reloads turn 1 history");
        assert_eq!(serde_json::to_value(&loaded.tool_choice).unwrap(), "auto");
        let tools = loaded.tools.as_deref().expect("lifted tools from turn 1");
        assert_eq!(tools.len(), 1);
        assert!(matches!(
            &tools[0],
            openai_protocol::responses::ResponseTool::Function(ft) if ft.function.name == "exec_command"
        ));
        let ResponseInput::Items(items) = &loaded.input else {
            panic!("expected items");
        };
        assert_eq!(items.len(), 3);
        assert!(matches!(
            &items[0],
            ResponseInputOutputItem::Message { role, .. } if role == "user"
        ));
    }
}
