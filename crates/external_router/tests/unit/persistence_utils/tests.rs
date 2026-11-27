use chrono::Utc;

use super::*;

fn stored_item(item_type: &str, content: Value) -> ConversationItem {
    ConversationItem {
        id: ConversationItemId::from("ig_replay_1"),
        response_id: None,
        item_type: item_type.to_string(),
        role: None,
        content,
        status: Some("completed".to_string()),
        created_at: Utc::now(),
        tenant: None,
    }
}

#[test]
fn item_to_json_restores_top_level_fields_for_whole_item_store() {
    let stored = stored_item(
        "image_generation_call",
        json!({
            "id": "ig_replay_1",
            "type": "image_generation_call",
            "status": "completed",
            "revised_prompt": "cat",
        }),
    );

    let serialized = item_to_json(stored);
    assert_eq!(serialized["type"], json!("image_generation_call"));
    assert_eq!(serialized["revised_prompt"], json!("cat"));
    assert!(
        serialized.get("content").is_none(),
        "non-message whole-item store must hoist top-level fields, not \
             nest the original item under `content`: {serialized}"
    );
}

#[test]
fn item_to_json_falls_back_to_legacy_content_wrap_for_array_content() {
    // Pre-fix rows for non-listed types may have stored `[]`. Preserve
    // the legacy wrapping so old data still round-trips.
    let stored = stored_item("image_generation_call", json!([]));
    let serialized = item_to_json(stored);
    assert_eq!(serialized["content"], json!([]));
}

#[test]
fn replayed_image_generation_input_item_is_stored_whole() {
    let input_item = json!({
        "id": "ig_replay_1",
        "type": "image_generation_call",
        "status": "completed",
        "revised_prompt": "cat",
    });

    let new_item = item_to_new_conversation_item(input_item, None, /* is_input */ true, None);
    assert_eq!(new_item.item_type, "image_generation_call");
    assert_eq!(
        new_item
            .content
            .get("revised_prompt")
            .and_then(|v| v.as_str()),
        Some("cat"),
    );
    assert_eq!(
        new_item.content.get("type").and_then(|v| v.as_str()),
        Some("image_generation_call"),
    );
}

#[test]
fn replayed_message_input_item_still_extracts_content() {
    let input_item = json!({
        "id": "msg_1",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
        "status": "completed",
    });

    let new_item = item_to_new_conversation_item(input_item, None, /* is_input */ true, None);
    assert_eq!(new_item.item_type, "message");
    assert_eq!(
        new_item.content,
        json!([{"type": "input_text", "text": "hi"}])
    );
}

/// The object `item_to_json` answers for each stored content shape, byte
/// for byte: the listed fields of a tool item in the list's order, a
/// message's content with its phase hoisted, a legacy message array, the
/// fields of an item stored whole (its own id, type and status left to the
/// record's), the legacy `content` wrap of a row that is not an object.
#[test]
fn item_to_json_writes_the_same_bytes_for_every_content_shape() {
    let item = |id: &str, item_type: &str, role: Option<&str>, content: Value| ConversationItem {
        id: ConversationItemId::from(id),
        response_id: None,
        item_type: item_type.to_string(),
        role: role.map(str::to_string),
        content,
        status: Some("completed".to_string()),
        created_at: Utc::now(),
        tenant: None,
    };
    let cases = [
        (
            item(
                "mcp_1",
                "mcp_call",
                None,
                json!({
                    "server_label": "s", "name": "n", "arguments": "{}", "output": "o", "extra": 1
                }),
            ),
            r#"{"id":"mcp_1","type":"mcp_call","name":"n","arguments":"{}","output":"o","server_label":"s","status":"completed"}"#,
        ),
        (
            item(
                "msg_1",
                "message",
                Some("user"),
                json!({"content": [{"type": "input_text", "text": "hi"}], "phase": "commentary"}),
            ),
            r#"{"id":"msg_1","type":"message","role":"user","content":[{"type":"input_text","text":"hi"}],"phase":"commentary","status":"completed"}"#,
        ),
        (
            item(
                "msg_2",
                "message",
                Some("assistant"),
                json!([{"type": "output_text", "text": "hello", "annotations": []}]),
            ),
            r#"{"id":"msg_2","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}],"status":"completed"}"#,
        ),
        (
            item(
                "ig_1",
                "image_generation_call",
                None,
                json!({
                    "id": "ig_1", "type": "image_generation_call", "status": "completed",
                    "revised_prompt": "cat", "result": null
                }),
            ),
            r#"{"id":"ig_1","type":"image_generation_call","revised_prompt":"cat","result":null,"status":"completed"}"#,
        ),
        (
            item("ws_1", "web_search_call", None, json!([])),
            r#"{"id":"ws_1","type":"web_search_call","content":[],"status":"completed"}"#,
        ),
    ];
    for (item, expected) in cases {
        let written = serde_json::to_string(&item_to_json(item)).unwrap();
        assert_eq!(written, expected);
    }
}

/// The content each of a turn's items is stored under, byte for byte,
/// through the persist and back from the store (a tool item whole, a
/// message with a phase as its content beside the phase, a plain message
/// as its content array), and the objects the API answers for them.
#[tokio::test]
async fn persisted_items_store_the_same_bytes_for_every_shape() {
    use smg_data_connector::{
        ListParams, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage, NewConversation, SortOrder,
    };

    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let responses: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let conversation = conversations
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "conversation": conversation.id.0,
    }))
    .unwrap();
    let response = json!({
        "id": "resp_bytes",
        "object": "response",
        "model": "m",
        "status": "completed",
        "output": [
            {
                "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
                "arguments": "{\"a\":1}", "status": "completed"
            },
            {
                "type": "message", "id": "msg_p", "role": "assistant", "status": "completed",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "thinking", "annotations": []}]
            },
            {
                "type": "message", "id": "msg_f", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }
        ]
    });
    persist_conversation_items(
        conversations,
        items.clone(),
        responses,
        &response,
        &request,
        None,
        None,
    )
    .await
    .unwrap();

    let stored = items
        .list_items(
            &conversation.id,
            ListParams {
                limit: 20,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    let contents: Vec<String> = stored
        .iter()
        .map(|item| serde_json::to_string(&item.content).unwrap())
        .collect();
    assert_eq!(
        contents,
        [
            r#"[{"type":"input_text","text":"hi"}]"#,
            r#"{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":"{\"a\":1}","status":"completed"}"#,
            r#"{"content":[{"type":"output_text","text":"thinking","annotations":[]}],"phase":"commentary"}"#,
            r#"[{"type":"output_text","text":"hello","annotations":[]}]"#,
        ]
    );
    let answered: Vec<String> = stored
        .into_iter()
        .skip(1)
        .map(|item| serde_json::to_string(&item_to_json(item)).unwrap())
        .collect();
    assert_eq!(
        answered,
        [
            r#"{"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":1}","status":"completed"}"#,
            r#"{"id":"msg_p","type":"message","role":"assistant","content":[{"type":"output_text","text":"thinking","annotations":[]}],"phase":"commentary","status":"completed"}"#,
            r#"{"id":"msg_f","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}],"status":"completed"}"#,
        ]
    );
}

/// A turn that echoes an item id held by another tenant links nothing: the
/// item under the id is that tenant's, and the link would read it through
/// this conversation. The item's own tenant links it, as the contract says.
#[tokio::test]
async fn an_echoed_id_held_by_another_tenant_is_refused_before_the_link() {
    use smg_data_connector::{ListParams, MemoryConversationItemStorage, SortOrder};

    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let echoed = |tenant: &str| NewConversationItem {
        id: Some(ConversationItemId::from("msg_theirs")),
        response_id: None,
        item_type: "message".to_string(),
        role: Some("user".to_string()),
        content: json!([{"type": "input_text", "text": tenant}]),
        status: None,
        tenant: Some(tenant.to_string()),
    };
    items.create_item(echoed("tenant-a")).await.unwrap();
    let conversation = ConversationId::from("conv_b");

    let err = create_and_link_item(&items, Some(&conversation), echoed("tenant-b"), Utc::now())
        .await
        .unwrap_err();
    assert!(err.contains("held by another tenant"), "{err}");
    let linked = items
        .list_items(
            &conversation,
            ListParams {
                limit: 20,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert!(linked.is_empty(), "nothing is linked: {linked:?}");

    create_and_link_item(&items, Some(&conversation), echoed("tenant-a"), Utc::now())
        .await
        .unwrap();
}

/// A background record closed with the object this persist writes (the
/// status and the output items) holds an earlier attempt's write that
/// landed while its answer was lost: the persist goes on to link the
/// items, for a result and for the run's own `failed` object alike. A
/// record closed with anything else (a cancel from elsewhere, another
/// status, other items, a `failed` without items that could be a failure
/// written elsewhere) refuses the persist and links nothing.
#[tokio::test]
async fn a_record_that_holds_this_runs_result_is_linked_and_a_close_from_elsewhere_is_not() {
    use smg_data_connector::{
        ListParams, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage, NewConversation, SortOrder,
    };

    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let responses: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let conversation = conversations
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "background": true,
        "conversation": conversation.id.0,
    }))
    .unwrap();
    let result = |id: &str, status: &str, items: &[&str]| {
        let output: Vec<Value> = items
            .iter()
            .map(|item| {
                json!({
                    "type": "message",
                    "id": item,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}]
                })
            })
            .collect();
        json!({
            "id": id,
            "object": "response",
            "model": "m",
            "status": status,
            "background": true,
            "output": output
        })
    };
    // What the store holds under each id when the persist runs.
    for (id, status, items) in [
        ("resp_own", "completed", &["msg_own"][..]),
        ("resp_bare", "completed", &[]),
        ("resp_failed_own", "failed", &["msg_failed"]),
        ("resp_cancelled", "cancelled", &["msg_own"]),
        ("resp_other_status", "completed", &["msg_own"]),
        ("resp_other_items", "completed", &["msg_theirs"]),
        ("resp_failed_bare", "failed", &[]),
    ] {
        let mut stored = build_stored_response(result(id, status, items), &request);
        stored.id = ResponseId::from(id);
        responses.store_response(stored).await.unwrap();
    }
    let linked = |id: &'static str| {
        let items = items.clone();
        let conversation_id = conversation.id.clone();
        async move {
            items
                .list_items(
                    &conversation_id,
                    ListParams {
                        limit: 20,
                        order: SortOrder::Asc,
                        after: None,
                    },
                )
                .await
                .unwrap()
                .into_iter()
                .filter(|item| item.response_id.as_deref() == Some(id))
                .count()
        }
    };
    let persist = |id: &'static str, status: &'static str, output: &'static [&'static str]| {
        let (conversations, items, responses) =
            (conversations.clone(), items.clone(), responses.clone());
        let request = request.clone();
        async move {
            persist_conversation_items_if_open(
                conversations,
                items,
                responses,
                &result(id, status, output),
                &request,
                None,
                None,
            )
            .await
            .unwrap()
        }
    };

    assert!(
        persist("resp_own", "completed", &["msg_own"]).await,
        "the record holds this run's result: stored"
    );
    assert_eq!(
        linked("resp_own").await,
        2,
        "the input and the output are linked"
    );
    assert!(
        persist("resp_bare", "completed", &[]).await,
        "a result without output items is still this run's"
    );
    assert_eq!(linked("resp_bare").await, 1, "the input is linked");
    assert!(
        persist("resp_failed_own", "failed", &["msg_failed"]).await,
        "the run's own failed object, with its items, is this run's write"
    );
    assert_eq!(linked("resp_failed_own").await, 2);
    for (id, status, items, why) in [
        (
            "resp_cancelled",
            "completed",
            &["msg_own"][..],
            "a cancel from elsewhere refuses the result",
        ),
        (
            "resp_other_status",
            "incomplete",
            &["msg_own"],
            "another status is not this write",
        ),
        (
            "resp_other_items",
            "completed",
            &["msg_own"],
            "other output items are not this write",
        ),
        (
            "resp_failed_bare",
            "failed",
            &[],
            "a failed object without items could be a failure written elsewhere",
        ),
    ] {
        assert!(!persist(id, status, items).await, "{why}");
        assert_eq!(linked(id).await, 0, "{why}");
    }
    assert!(is_result_status("completed") && is_result_status("incomplete"));
    assert!(!is_result_status("cancelled") && !is_result_status("failed"));
}
