use std::sync::Arc;

use rustls::crypto::ring;
use serde_json::json;
use smg_data_connector::{
    AddedAtSequence, ConversationItemId, ConversationStorage, MemoryConversationItemStorage,
    MemoryConversationStorage, MemoryResponseStorage, NewConversation, NewConversationItem,
};
use smg_mcp::{McpConfig, McpOrchestrator};

use super::*;
use crate::openai::context::{ResponsesComponents, SharedComponents};

/// The tests' HTTP client, the process-level TLS crypto provider
/// installed first; the loader under test sends nothing with it.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
fn client() -> reqwest::Client {
    let _ = ring::default_provider().install_default();
    reqwest::Client::new()
}

/// The components of one router over in-memory stores, with the
/// conversation `conv_1`.
async fn components() -> ResponsesComponents {
    let conversations = Arc::new(MemoryConversationStorage::new());
    conversations
        .create_conversation(NewConversation {
            id: Some(ConversationId::from("conv_1")),
            metadata: None,
            tenant: None,
        })
        .await
        .expect("the conversation is created");
    ResponsesComponents {
        shared: Arc::new(SharedComponents { client: client() }),
        mcp_orchestrator: Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("an orchestrator without servers"),
        ),
        mcp_format_registry: Default::default(),
        response_storage: Arc::new(MemoryResponseStorage::new()),
        conversation_storage: conversations,
        conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
    }
}

/// Links `rows` to `conv_1` in order, each stamped later than the one
/// before, as the conversations API does.
async fn link_rows(components: &ResponsesComponents, rows: Vec<NewConversationItem>) {
    let conv_id = ConversationId::from("conv_1");
    let mut stamps = AddedAtSequence::new();
    for row in rows {
        let created = components
            .conversation_item_storage
            .create_item(row)
            .await
            .expect("the row is created");
        components
            .conversation_item_storage
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
        item_type: ItemType::FUNCTION_CALL.to_string(),
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
        item_type: ItemType::FUNCTION_CALL_OUTPUT.to_string(),
        role: None,
        content: json!({"type": "function_call_output", "call_id": call_id, "output": "42"}),
        status: Some("completed".to_string()),
        tenant: None,
    }
}

/// The kinds of the items, with the call id of a tool item, and the text
/// of a message, for the window assertions.
fn shape(items: &[ResponseInputOutputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| match item {
            ResponseInputOutputItem::Message { role, content, .. } => {
                let text = content
                    .iter()
                    .find_map(|part| match part {
                        ResponseContentPart::InputText { text }
                        | ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .unwrap_or("");
                format!("{role}:{text}")
            }
            ResponseInputOutputItem::FunctionToolCall { call_id, .. } => {
                format!("function_call:{call_id}")
            }
            ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => {
                format!("function_call_output:{}", call_id.as_deref().unwrap_or(""))
            }
            other => format!("{other:?}"),
        })
        .collect()
}

async fn loaded_input(
    components: &ResponsesComponents,
    text: &str,
) -> Vec<ResponseInputOutputItem> {
    let mut request = ResponsesRequest {
        input: ResponseInput::Text(text.to_string()),
        ..Default::default()
    };
    load_input_history(components, Some("conv_1"), &mut request, "m", None)
        .await
        .expect("the conversation loads");
    match request.input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(_) => panic!("the loaded input is an item list"),
    }
}

#[tokio::test]
async fn external_conversation_replays_codex_tool_declarations_and_agent_messages() {
    let components = components().await;
    let first_input = json!([
        {"type":"additional_tools","id":"at_1","role":"developer","tools":[
            {"type":"function","name":"exec_command","parameters":{"type":"object"}}
        ]},
        {"type":"agent_message","id":"amsg_1","author":"/root","recipient":"/root/worker",
         "internal_chat_message_metadata_passthrough":{"turn_id":"turn_1","create_time":1795400000.5},
         "content":[{"type":"input_text","text":"Message Type: NEW_TASK\nTask name: /root/worker\nSender: /root\nPayload:\n"},
                    {"type":"encrypted_content","encrypted_content":"opaque-agent-message"}]}
    ]);
    let first = serde_json::from_value(json!({
        "model":"m", "conversation":"conv_1", "input":first_input
    }))
    .expect("the first request parses");
    crate::persistence_utils::persist_conversation_items(
        components.conversation_storage.clone(),
        components.conversation_item_storage.clone(),
        components.response_storage.clone(),
        &json!({"id":"resp_first","status":"completed","output":[]}),
        &first,
        None,
        None,
    )
    .await
    .expect("the first turn is stored");

    let replayed = loaded_input(&components, "Report the results.").await;
    let wire = serde_json::to_value(replayed).expect("the follow-up input serializes");
    assert_eq!(wire[0], first_input[0]);
    assert_eq!(wire[1], first_input[1]);
    assert_eq!(wire[2]["content"][0]["text"], "Report the results.");
}

/// A conversation of 120 items replays its newest 100, oldest first,
/// then the request's input: the first 20 are not sent.
#[tokio::test]
async fn a_conversation_longer_than_the_limit_replays_its_newest_items() {
    let components = components().await;
    link_rows(&components, (1..=120).map(message_row).collect()).await;

    let items = loaded_input(&components, "item 121").await;

    let expected: Vec<String> = (21..=121)
        .map(|n| format!("{}:item {n}", if n % 2 == 1 { "user" } else { "assistant" }))
        .collect();
    assert_eq!(shape(&items), expected);
}

/// A call and its output replay together or not at all: when the caller
/// may not see one of the two, the other goes with it. A second output
/// of the call takes nothing with it: the call keeps the output the
/// caller may see.
#[tokio::test]
async fn a_tool_row_the_caller_may_not_see_takes_its_pair_out_of_the_replay() {
    for hide_call in [true, false] {
        let components = components().await;
        let (mut call, mut output) = (call_row("call_other"), output_row("call_other"));
        let hidden = if hide_call { &mut call } else { &mut output };
        hidden.tenant = Some("other".into());
        link_rows(&components, vec![message_row(1), call, output]).await;
        assert_eq!(
            shape(&loaded_input(&components, "item 2").await),
            ["user:item 1", "user:item 2"],
            "hide_call {hide_call}"
        );
    }

    let components = components().await;
    let mut second = output_row("call_own");
    second.tenant = Some("other".into());
    let rows = vec![
        message_row(1),
        call_row("call_own"),
        output_row("call_own"),
        second,
    ];
    link_rows(&components, rows).await;
    assert_eq!(
        shape(&loaded_input(&components, "item 2").await),
        [
            "user:item 1",
            "function_call:call_own",
            "function_call_output:call_own",
            "user:item 2"
        ]
    );
}

/// A conversation one item longer than the limit, whose first items are
/// two calls with their outputs: its window begins at the first output,
/// inside a turn whose request is before the window. The window replays
/// from its first client message; the output of the call before it and
/// the call with its output after it go with the turn the cut split.
#[tokio::test]
async fn a_cut_window_begins_at_its_first_client_message() {
    let components = components().await;
    let mut rows = vec![
        call_row("call_A"),
        output_row("call_A"),
        call_row("call_B"),
        output_row("call_B"),
    ];
    rows.extend((5..=101).map(message_row));
    link_rows(&components, rows).await;

    let items = loaded_input(&components, "item 102").await;

    let kinds = shape(&items);
    assert_eq!(kinds[0], "user:item 5", "{kinds:?}");
    assert!(
        !kinds.iter().any(|kind| kind.starts_with("function_call")),
        "{kinds:?}"
    );
    // the calls and outputs are the tail of the turn the cut split: the
    // messages replay, then the request
    assert_eq!(items.len(), 101 - 4 + 1);
    assert_eq!(kinds.last().map(String::as_str), Some("user:item 102"));
}

/// A conversation of exactly the limit is not cut: every item replays,
/// an output whose call is not among them included (the window did not
/// begin after its call).
#[tokio::test]
async fn a_conversation_of_exactly_the_limit_replays_whole() {
    let components = components().await;
    let mut rows = vec![output_row("call_A")];
    rows.extend((2..=100).map(message_row));
    link_rows(&components, rows).await;

    let items = loaded_input(&components, "item 101").await;

    let kinds = shape(&items);
    assert_eq!(kinds[0], "function_call_output:call_A", "{kinds:?}");
    assert_eq!(items.len(), 100 + 1);
    assert_eq!(kinds.last().map(String::as_str), Some("user:item 101"));
}
/// A cut that lands inside a turn (row 51 of 150 is the output of turn
/// 13's call): the window begins at the next turn boundary, the user's
/// message of turn 14 (row 53); the output and the answer of turn 13
/// before it are not sent.
#[tokio::test]
async fn a_cut_window_that_begins_inside_a_turn_starts_at_the_next_turn_boundary() {
    let components = components().await;
    let mut rows = Vec::new();
    for turn in 1..=37 {
        rows.push(message_row(4 * turn - 3));
        rows.push(call_row(&format!("call_{turn}")));
        rows.push(output_row(&format!("call_{turn}")));
        rows.push(message_row(4 * turn));
    }
    rows.push(message_row(149));
    rows.push(call_row("call_38"));
    link_rows(&components, rows).await;

    let items = loaded_input(&components, "item 151").await;

    let kinds = shape(&items);
    assert_eq!(
        &kinds[..4],
        [
            "user:item 53",
            "function_call:call_14",
            "function_call_output:call_14",
            "assistant:item 56"
        ],
        "{kinds:?}"
    );
    assert!(
        !kinds.contains(&"assistant:item 52".to_string()),
        "{kinds:?}"
    );
    assert!(
        !kinds.contains(&"function_call_output:call_13".to_string()),
        "{kinds:?}"
    );
    // rows 53 to 150, then the request
    assert_eq!(items.len(), 150 - 52 + 1);
    assert_eq!(kinds.last().map(String::as_str), Some("user:item 151"));
}

/// A stored row replays as the item it was (a message by its role and
/// content, a call as itself); a reasoning row, a row of an unknown type
/// and a message whose content does not read are not replayed.
#[test]
fn a_stored_row_replays_as_its_item_or_not_at_all() {
    let row = |item_type: &str, content: Value| ConversationItem {
        id: ConversationItemId::from("item_1"),
        response_id: None,
        item_type: item_type.to_string(),
        role: Some("assistant".to_string()),
        content,
        status: Some("completed".to_string()),
        created_at: chrono::Utc::now(),
        tenant: None,
    };
    let message = stored_item_as_input(row(
        "message",
        json!([{"type": "output_text", "text": "item 1", "annotations": []}]),
    ))
    .expect("a message replays");
    assert_eq!(shape(&[message]), vec!["assistant:item 1"]);
    let call = stored_item_as_input(row(
        ItemType::FUNCTION_CALL,
        json!({"type": "function_call", "call_id": "call_A", "name": "calc", "arguments": "{}", "status": "completed"}),
    ))
    .expect("a call replays");
    assert_eq!(shape(&[call]), vec!["function_call:call_A"]);
    assert!(stored_item_as_input(row("reasoning", json!({}))).is_none());
    assert!(stored_item_as_input(row("mystery", json!({}))).is_none());
    assert!(stored_item_as_input(row("message", json!({"text": "no content array"}))).is_none());
}
