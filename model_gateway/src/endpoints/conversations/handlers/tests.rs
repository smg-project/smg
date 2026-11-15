use serde_json::Value;
use smg_data_connector::{MemoryConfig, MemoryConversationItemStorage, MemoryConversationStorage};

use super::*;
use crate::middleware::test_support::json_of;

/// A client-supplied item id longer than the stores' id columns is refused
/// at the handler, before any store is asked, so every backend answers the
/// same 400 under the public API's code; an id at the bound is kept and
/// linked under that id.
#[tokio::test]
async fn an_item_id_over_the_bound_is_refused_before_the_store_and_one_at_the_bound_is_kept() {
    let conversation_storage: Arc<dyn ConversationStorage> =
        Arc::new(MemoryConversationStorage::new());
    let item_storage: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::new());
    let conversation = conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
            tenant: None,
        })
        .await
        .unwrap();
    let conv_id = conversation.id.0.clone();
    let items = |id: &str| json!({"items": [{"type": "message", "role": "user", "content": "x", "id": id}]});

    // 65 and 64 characters: one over the stores' 64-character id columns, one at the bound.
    let over = format!("msg_{}", "a".repeat(61));
    assert_eq!(over.len(), 65);
    let response = create_conversation_items(
        &conversation_storage,
        &item_storage,
        None,
        &conv_id,
        items(&over),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "string_above_max_length", "{json}");
    assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    assert_eq!(json["error"]["param"], "items[].id", "{json}");
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("64"),
        "the message names the bound: {json}"
    );

    // The bound holds for the whole batch before any store is asked: an over-long
    // id on the second item leaves the first item unstored and unlinked.
    let batch = json!({"items": [
        {"type": "message", "role": "user", "content": "first", "id": "msg_first"},
        {"type": "message", "role": "user", "content": "second", "id": over}
    ]});
    let response =
        create_conversation_items(&conversation_storage, &item_storage, None, &conv_id, batch)
            .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let listed = item_storage
        .list_items(
            &ConversationId::from(conv_id.as_str()),
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert!(
        listed.is_empty(),
        "nothing is stored for a refused batch: {listed:?}"
    );
    // The store itself, not the conversation's links: the old per-item check
    // created the first item before refusing the second, linking none.
    assert!(
        item_storage
            .get_item(&ConversationItemId::from("msg_first"))
            .await
            .unwrap()
            .is_none(),
        "an earlier item of a refused batch is not stored"
    );
    let over_id = ConversationItemId::from(over.as_str());
    assert!(
        item_storage.get_item(&over_id).await.unwrap().is_none(),
        "nothing is stored under the refused id"
    );
    assert!(!item_storage
        .is_item_linked(&conversation.id, &over_id)
        .await
        .unwrap());

    let at_bound = format!("msg_{}", "a".repeat(60));
    assert_eq!(at_bound.len(), 64);
    let response = create_conversation_items(
        &conversation_storage,
        &item_storage,
        None,
        &conv_id,
        items(&at_bound),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(item_storage
        .is_item_linked(
            &conversation.id,
            &ConversationItemId::from(at_bound.as_str())
        )
        .await
        .unwrap());
}

/// A conversation created with `items` lists them, in the order given, as
/// a conversation created empty and given the same items does; the
/// accepted-but-unimplemented types answer the add-items route's
/// `warnings`; a batch the add-items route refuses before any store (over
/// the bound, an unsupported type, a malformed field) creates no
/// conversation, and one the store refuses leaves none behind.
#[tokio::test]
async fn a_conversation_created_with_items_lists_them_as_the_add_items_route_stores_them() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let two = json!([
        {"type": "message", "role": "user", "content": "first"},
        {"type": "message", "role": "assistant", "content": "second"}
    ]);
    let shape = |listed: &Value| -> Vec<Value> {
        listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| json!([item["type"], item["role"], item["content"], item["status"]]))
            .collect()
    };

    let response = create_conversation(
        &conversations,
        &items,
        None,
        json!({"items": two, "metadata": {"a": "b"}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let created = json_of(response).await;
    assert_eq!(created["object"], "conversation", "{created}");
    assert_eq!(created["metadata"]["a"], "b", "{created}");
    assert!(created.get("warnings").is_none(), "{created}");
    let with_items = created["id"].as_str().unwrap().to_string();
    let listed = json_of(
        list_conversation_items(
            &conversations,
            &items,
            None,
            &with_items,
            None,
            Some("asc"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(listed["data"].as_array().map(Vec::len), Some(2), "{listed}");
    assert_eq!(listed["data"][0]["content"][0]["text"], "first", "{listed}");
    assert_eq!(
        listed["data"][1]["content"][0]["text"], "second",
        "{listed}"
    );

    let with_null = create_conversation(&conversations, &items, None, json!({"items": null})).await;
    assert_eq!(
        with_null.status(),
        StatusCode::OK,
        "a null items field is an absent one"
    );
    let empty = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let then_added = empty["id"].as_str().unwrap().to_string();
    let added = create_conversation_items(
        &conversations,
        &items,
        None,
        &then_added,
        json!({"items": two}),
    )
    .await;
    assert_eq!(added.status(), StatusCode::OK);
    let listed_after = json_of(
        list_conversation_items(
            &conversations,
            &items,
            None,
            &then_added,
            None,
            Some("asc"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(
        shape(&listed),
        shape(&listed_after),
        "{listed}\n{listed_after}"
    );

    let response = create_conversation(
        &conversations,
        &items,
        None,
        json!({"items": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let warned = json_of(response).await;
    assert_eq!(
        warned["warnings"].as_array().map(Vec::len),
        Some(1),
        "{warned}"
    );

    let before = conversations.stored_record_count();
    let items_before = items.stored_record_count();
    let too_many: Vec<Value> = (0..=MAX_ITEMS_PER_REQUEST)
        .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
        .collect();
    for (body, status) in [
        (json!({"items": too_many}), StatusCode::BAD_REQUEST),
        (
            json!({"items": [{"type": "bogus"}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"items": [{"type": "message", "content": "no role"}]}),
            StatusCode::BAD_REQUEST,
        ),
        (json!({"items": "first"}), StatusCode::BAD_REQUEST),
        (
            json!({"items": [{"type": "item_reference", "id": "msg_nowhere"}]}),
            StatusCode::NOT_FOUND,
        ),
    ] {
        let response = create_conversation(&conversations, &items, None, body.clone()).await;
        assert_eq!(response.status(), status, "{body}");
        assert_eq!(
            conversations.stored_record_count(),
            before,
            "a refused create creates nothing: {body}"
        );
        assert_eq!(
            items.stored_record_count(),
            items_before,
            "a refused create stores no item: {body}"
        );
    }

    // An id the caller already holds is adopted without a parse, as the
    // add-items route adopts it: no role is asked of it.
    let held = listed["data"][0]["id"].as_str().unwrap().to_string();
    let adopted = create_conversation(
        &conversations,
        &items,
        None,
        json!({"items": [{"id": held}]}),
    )
    .await;
    assert_eq!(adopted.status(), StatusCode::OK);
}

/// A batch refused after its first item (an unknown reference, an
/// unsupported type) stores nothing on either route: the whole batch is
/// prepared before the first write, so a corrected retry under the same id
/// is a fresh write with the retry's content.
#[tokio::test]
async fn a_refused_batch_stores_nothing_on_either_route() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let empty = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let conversation = empty["id"].as_str().unwrap().to_string();
    let conversations_before = conversations.stored_record_count();
    let items_before = items.stored_record_count();
    let first = json!({"id": "msg_first", "type": "message", "role": "user", "content": "A"});

    for (second, status) in [
        (
            json!({"type": "item_reference", "id": "msg_nowhere"}),
            StatusCode::NOT_FOUND,
        ),
        (json!({"type": "bogus"}), StatusCode::BAD_REQUEST),
    ] {
        let body = json!({"items": [first, second]});
        let on_create = create_conversation(&conversations, &items, None, body.clone()).await;
        assert_eq!(on_create.status(), status, "{body}");
        let on_add =
            create_conversation_items(&conversations, &items, None, &conversation, body.clone())
                .await;
        assert_eq!(on_add.status(), status, "{body}");
        assert_eq!(
            conversations.stored_record_count(),
            conversations_before,
            "no conversation: {body}"
        );
        assert_eq!(items.stored_record_count(), items_before, "no item: {body}");
    }

    let retried = create_conversation_items(
        &conversations,
        &items,
        None,
        &conversation,
        json!({"items": [{"id": "msg_first", "type": "message", "role": "user", "content": "B"}]}),
    )
    .await;
    assert_eq!(retried.status(), StatusCode::OK);
    let listed = json_of(
        list_conversation_items(
            &conversations,
            &items,
            None,
            &conversation,
            None,
            Some("asc"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(listed["data"][0]["content"][0]["text"], "B", "{listed}");
}

/// A conversation's `metadata` is bounded as the public API bounds it, on
/// create and on update: 16 pairs, 64-character keys, 512-character
/// values, strings only; every refusal carries the public code under
/// `metadata` and stores nothing; metadata at the bounds is kept; a body
/// that is not an object is refused before any field is read.
#[tokio::test]
async fn metadata_is_bounded_as_the_public_api_bounds_it_on_create_and_update() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let error_of = |response: Response| async move { json_of(response).await["error"].clone() };
    let one = |key: String, value: Value| {
        let mut map = serde_json::Map::new();
        map.insert(key, value);
        Value::Object(map)
    };

    let before = conversations.stored_record_count();
    let seventeen: serde_json::Map<String, Value> =
        (0..17).map(|i| (format!("k{i}"), json!("v"))).collect();
    for (metadata, code) in [
        (json!({"k": "v".repeat(513)}), "string_above_max_length"),
        (one("k".repeat(65), json!("v")), "string_above_max_length"),
        (Value::Object(seventeen), "object_above_max_properties"),
        (json!({"k": {"nested": 1}}), "invalid_type"),
        (json!({"n": 5}), "invalid_type"),
        (json!("text"), "invalid_type"),
    ] {
        let response =
            create_conversation(&conversations, &items, None, json!({"metadata": metadata})).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{metadata}");
        let error = error_of(response).await;
        assert_eq!(error["code"], code, "{error}");
        assert_eq!(error["param"], "metadata", "{error}");
        assert_eq!(
            conversations.stored_record_count(),
            before,
            "a refused create creates nothing: {metadata}"
        );
    }
    let response = create_conversation(&conversations, &items, None, json!([])).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = error_of(response).await;
    assert_eq!(error["code"], "invalid_type", "{error}");
    assert!(error["param"].is_null(), "{error}");
    assert_eq!(conversations.stored_record_count(), before);

    let at_the_bounds: serde_json::Map<String, Value> = (0..16)
        .map(|i| (format!("{i:0>64}"), json!("v".repeat(512))))
        .collect();
    let response = create_conversation(
        &conversations,
        &items,
        None,
        json!({"metadata": Value::Object(at_the_bounds.clone())}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let created = json_of(response).await;
    assert_eq!(
        created["metadata"],
        Value::Object(at_the_bounds.clone()),
        "{created}"
    );
    let conv_id = created["id"].as_str().unwrap().to_string();

    let key0 = format!("{:0>64}", 0);
    for (body, code) in [
        (
            json!({"metadata": one(key0.clone(), json!("v".repeat(513)))}),
            "string_above_max_length",
        ),
        (
            json!({"metadata": one(key0.clone(), json!(1))}),
            "invalid_type",
        ),
        (
            json!({"metadata": {"seventeenth": "v"}}),
            "object_above_max_properties",
        ),
        (json!({"metadata": "text"}), "invalid_type"),
        (json!([]), "invalid_type"),
    ] {
        let response = update_conversation(&conversations, None, &conv_id, body.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(error_of(response).await["code"], code, "{body}");
    }
    let kept = json_of(get_conversation(&conversations, None, &conv_id).await).await;
    assert_eq!(
        kept["metadata"],
        Value::Object(at_the_bounds),
        "a refused update changes nothing: {kept}"
    );
    let response = update_conversation(
        &conversations,
        None,
        &conv_id,
        json!({"metadata": one(key0, Value::Null)}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let updated = json_of(response).await;
    assert_eq!(
        updated["metadata"].as_object().map(serde_json::Map::len),
        Some(15),
        "{updated}"
    );

    // A pair stored before these rules held (a number) does not block an
    // unrelated patch: the key and value rules are read on the patch's own
    // pairs, the pair count on the merged object.
    let stored = conversations
        .create_conversation(NewConversation {
            id: None,
            metadata: json!({"n": 5}).as_object().cloned(),
            tenant: None,
        })
        .await
        .unwrap();
    let response = update_conversation(
        &conversations,
        None,
        &stored.id.0,
        json!({"metadata": {"a": "b"}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let patched = json_of(response).await;
    assert_eq!(patched["metadata"], json!({"n": 5, "a": "b"}), "{patched}");
    let sixteen_more: serde_json::Map<String, Value> =
        (0..16).map(|i| (format!("k{i}"), json!("v"))).collect();
    let response = update_conversation(
        &conversations,
        None,
        &stored.id.0,
        json!({"metadata": Value::Object(sixteen_more)}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_of(response).await["error"]["code"],
        "object_above_max_properties",
        "the pair count is the merged object's"
    );
}

/// A message item's `role`, `status` and content parts are refused outside
/// the public API's sets, in its words and under the field named, before
/// any store is asked, on the add-items route and on create alike; an item
/// inside the sets (a developer message, an incomplete status, the typed
/// parts) is stored and listed as sent, and an item of another type keeps
/// its status.
#[tokio::test]
async fn a_message_items_role_status_and_parts_are_refused_outside_the_public_sets() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let created = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let conv_id = created["id"].as_str().unwrap().to_string();
    let message = |fields: Value| {
        let mut item = json!({"type": "message", "role": "user", "content": "x"});
        item.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        json!({"items": [item]})
    };

    let before = conversations.stored_record_count();
    for (body, code, param) in [
        (
            message(json!({"role": "bogus"})),
            "invalid_value",
            "items[].role",
        ),
        (message(json!({"role": 42})), "invalid_type", "items[].role"),
        (
            message(json!({"status": "flying"})),
            "invalid_value",
            "items[].status",
        ),
        (
            message(json!({"status": 3})),
            "invalid_value",
            "items[].status",
        ),
        (
            message(json!({"content": [{"type": "bogus_part", "x": 1}]})),
            "invalid_value",
            "items[].content[0].type",
        ),
        (
            message(
                json!({"content": [{"type": "input_text", "text": "ok"}, {"text": "no type"}]}),
            ),
            "missing_required_parameter",
            "items[].content[1].type",
        ),
        (
            message(json!({"content": [{"type": "input_text"}]})),
            "missing_required_parameter",
            "items[].content[0].text",
        ),
        (
            message(json!({"content": [{"type": "refusal", "refusal": 7}]})),
            "invalid_type",
            "items[].content[0]",
        ),
        (
            message(json!({"content": [{"type": 7}]})),
            "invalid_type",
            "items[].content[0].type",
        ),
        (
            message(json!({"content": ["a bare string"]})),
            "invalid_type",
            "items[].content[0]",
        ),
        // An error inside a nested value names the part, never a field of it
        // that is fine or that it does not have.
        (
            message(json!({"content": [
                {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "ultra"}
            ]})),
            "invalid_value",
            "items[].content[0]",
        ),
        (
            message(json!({"content": [
                {"type": "output_text", "text": "x", "annotations": [{"type": "url_citation", "url": "u"}]}
            ]})),
            "missing_required_parameter",
            "items[].content[0]",
        ),
        (
            message(json!({"content": [
                {"type": "output_text", "text": "x", "annotations": [{"type": "bogus"}]}
            ]})),
            "invalid_value",
            "items[].content[0]",
        ),
        // A nested variant that carries the part's own type name is still the
        // part's error, not its type's.
        (
            message(json!({"content": [
                {"type": "output_text", "text": "x", "annotations": [{"type": "output_text"}]}
            ]})),
            "invalid_value",
            "items[].content[0]",
        ),
        (
            message(json!({"content": {"text": "x"}})),
            "invalid_type",
            "items[].content",
        ),
        (
            message(json!({"content": null})),
            "invalid_type",
            "items[].content",
        ),
    ] {
        let response =
            create_conversation_items(&conversations, &items, None, &conv_id, body.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        let error = json_of(response).await["error"].clone();
        assert_eq!(error["code"], code, "{body}: {error}");
        assert_eq!(error["param"], param, "{body}: {error}");
        assert_eq!(error["type"], "invalid_request_error", "{body}: {error}");
        let response = create_conversation(&conversations, &items, None, body.clone()).await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "on create: {body}"
        );
        assert_eq!(
            conversations.stored_record_count(),
            before,
            "a refused create creates nothing: {body}"
        );
    }
    let response =
        create_conversation_items(&conversations, &items, None, &conv_id, json!([])).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_of(response).await["error"]["code"], "invalid_type");

    // A batch refused at its second item under these rules stores nothing
    // of its first: the whole batch is prepared before any write.
    let items_before = items.stored_record_count();
    let mixed = json!({"items": [
        {"id": "msg_first", "type": "message", "role": "user", "content": "ok"},
        {"type": "message", "role": "user", "content": "x", "status": "flying"}
    ]});
    let response = create_conversation_items(&conversations, &items, None, &conv_id, mixed).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        items.stored_record_count(),
        items_before,
        "nothing of a refused batch is stored"
    );
    let listed = json_of(
        list_conversation_items(&conversations, &items, None, &conv_id, None, None, None).await,
    )
    .await;
    assert!(
        listed["data"].as_array().unwrap().is_empty(),
        "nothing is stored for a refused batch: {listed}"
    );

    let accepted = json!({"items": [
        {"type": "message", "role": "developer", "content": "rules", "status": "incomplete"},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/a.png"}
        ]},
        {"type": "message", "role": "assistant", "status": "in_progress", "content": [
            {"type": "output_text", "text": "hello"},
            {"type": "refusal", "refusal": "no"}
        ]},
        {"type": "web_search_call", "id": "ws_1", "status": "searching"}
    ]});
    let response =
        create_conversation_items(&conversations, &items, None, &conv_id, accepted).await;
    assert_eq!(response.status(), StatusCode::OK);
    let added = json_of(response).await;
    assert_eq!(added["data"].as_array().map(Vec::len), Some(4), "{added}");
    assert_eq!(added["data"][0]["status"], "incomplete", "{added}");
    assert_eq!(
        added["data"][1]["content"][1]["type"], "input_image",
        "{added}"
    );
    assert_eq!(added["data"][2]["content"][1]["refusal"], "no", "{added}");
    assert_eq!(
        added["data"][3]["status"], "searching",
        "an item of another type keeps its status: {added}"
    );
}

const A: Option<&str> = Some("header:tenant-a");
const B: Option<&str> = Some("header:tenant-b");

/// A conversation created by tenant a with one item; the stores and the ids.
async fn conversation_of_a() -> (
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
    String,
    String,
) {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let created = json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
    let conv_id = created["id"].as_str().unwrap().to_string();
    let added = json_of(
        create_conversation_items(
            &conversations,
            &items,
            A,
            &conv_id,
            json!({"items": [{"type": "message", "role": "user", "content": "secret"}]}),
        )
        .await,
    )
    .await;
    let item_id = added["data"][0]["id"].as_str().unwrap().to_string();
    (conversations, items, conv_id, item_id)
}

/// A conversation created under a tenant is read, listed, written into and
/// deleted by that tenant only; another tenant, and a caller without one,
/// get the answer of an unknown id on every route and change nothing.
#[tokio::test]
async fn another_tenants_conversation_answers_not_found_on_every_route() {
    let (conversations, items, conv_id, item_id) = conversation_of_a().await;
    let stored = conversations
        .get_conversation(&ConversationId::from(conv_id.as_str()))
        .await
        .unwrap()
        .expect("created");
    assert_eq!(
        stored.tenant.as_deref(),
        A,
        "the conversation carries its tenant"
    );

    for caller in [B, None] {
        let answers = [
            get_conversation(&conversations, caller, &conv_id).await,
            update_conversation(
                &conversations,
                caller,
                &conv_id,
                json!({"metadata": {"k": "v"}}),
            )
            .await,
            list_conversation_items(&conversations, &items, caller, &conv_id, None, None, None)
                .await,
            create_conversation_items(
                &conversations,
                &items,
                caller,
                &conv_id,
                json!({"items": [{"type": "message", "role": "user", "content": "intruder"}]}),
            )
            .await,
            get_conversation_item(&conversations, &items, caller, &conv_id, &item_id, None).await,
            delete_conversation_item(&conversations, &items, caller, &conv_id, &item_id).await,
            delete_conversation(&conversations, caller, &conv_id).await,
        ];
        for answer in answers {
            assert_eq!(answer.status(), StatusCode::NOT_FOUND, "{caller:?}");
        }
    }

    // Nothing changed for its owner: the one item is still there, the
    // conversation too, and the owner's own calls work.
    let listed = json_of(
        list_conversation_items(&conversations, &items, A, &conv_id, None, None, None).await,
    )
    .await;
    assert_eq!(listed["data"].as_array().unwrap().len(), 1);
    assert_eq!(
        get_conversation_item(&conversations, &items, A, &conv_id, &item_id, None)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        delete_conversation_item(&conversations, &items, A, &conv_id, &item_id)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        delete_conversation(&conversations, A, &conv_id)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        get_conversation(&conversations, A, &conv_id).await.status(),
        StatusCode::NOT_FOUND
    );
}

/// An item of another tenant is as unknown as a missing one to an
/// `item_reference`, and an item arriving under that id is refused rather
/// than served or overwritten; the owner references its own item freely.
#[tokio::test]
async fn another_tenants_item_cannot_be_referenced_or_adopted() {
    let (conversations, items, _conv_a, item_a) = conversation_of_a().await;
    let created = json_of(create_conversation(&conversations, &items, B, json!({})).await).await;
    let conv_b = created["id"].as_str().unwrap().to_string();

    let reference = json!({"items": [{"type": "item_reference", "id": item_a}]});
    let answer = create_conversation_items(&conversations, &items, B, &conv_b, reference).await;
    assert_eq!(answer.status(), StatusCode::NOT_FOUND);
    let adopt = json!({"items": [{"type": "message", "id": item_a, "role": "user", "content": "mine now"}]});
    let answer = create_conversation_items(&conversations, &items, B, &conv_b, adopt).await;
    assert_eq!(answer.status(), StatusCode::BAD_REQUEST);
    // a's item is untouched and still a's.
    let stored = items
        .get_item(&ConversationItemId::from(item_a.as_str()))
        .await
        .unwrap()
        .expect("still there");
    assert_eq!(stored.tenant.as_deref(), A);
    assert_eq!(stored.content[0]["text"], "secret");
    let listed = json_of(
        list_conversation_items(&conversations, &items, B, &conv_b, None, None, None).await,
    )
    .await;
    assert_eq!(listed["data"].as_array().map(Vec::len), Some(0));

    // The owner references its own item into a second conversation of its own.
    let created = json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
    let conv_a2 = created["id"].as_str().unwrap().to_string();
    let reference = json!({"items": [{"type": "item_reference", "id": item_a}]});
    let answer = create_conversation_items(&conversations, &items, A, &conv_a2, reference).await;
    assert_eq!(answer.status(), StatusCode::OK);
}

/// Without tenant identity nothing changes: a conversation created by a
/// caller without a tenant is served to every caller, as before.
#[tokio::test]
async fn a_conversation_without_a_tenant_is_served_to_every_caller() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let created = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let conv_id = created["id"].as_str().unwrap().to_string();
    for caller in [None, A] {
        assert_eq!(
            get_conversation(&conversations, caller, &conv_id)
                .await
                .status(),
            StatusCode::OK,
            "{caller:?}"
        );
        assert_eq!(
            list_conversation_items(&conversations, &items, caller, &conv_id, None, None, None)
                .await
                .status(),
            StatusCode::OK,
            "{caller:?}"
        );
    }
    assert_eq!(
        delete_conversation(&conversations, B, &conv_id)
            .await
            .status(),
        StatusCode::OK
    );
}

/// The CRUD helpers answer with the gateway's envelope, an object an SDK
/// decodes (`type` from the public vocabulary, a string `code`, `param`),
/// not a bare string under `error`.
#[tokio::test]
async fn error_helpers_build_the_object_envelope() {
    for (response, status, code) in [
        (
            bad_request("bad"),
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            not_found("Conversation not found"),
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            store_error(
                "Failed to get conversation",
                &smg_data_connector::ResponseStorageError::Timeout(
                    "the redis get_conversation call exceeded its bound of 5 s".to_string(),
                ),
            ),
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
        ),
        (
            store_error(
                "Failed to get conversation",
                &smg_data_connector::ResponseStorageError::StorageError("db error".to_string()),
            ),
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
        ),
    ] {
        assert_eq!(response.status(), status);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let error = json["error"]
            .as_object()
            .unwrap_or_else(|| panic!("error must be an object: {json}"));
        assert_eq!(error["code"], code);
        assert_eq!(error["type"], route_error::error_type_for_status(status));
        assert!(error["message"].is_string());
        assert!(error["param"].is_null());
    }
}

/// A reference to an item the same batch adds, and a new id given twice in
/// one batch, store the item once and list it once on both routes: the
/// batch's own ids are known before any write, so no store is asked for an
/// item that is not written yet; an accepted-but-unimplemented type warns
/// once.
#[tokio::test]
async fn a_reference_to_an_item_of_the_same_batch_and_a_repeated_id_store_the_item_once() {
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> = Arc::new(MemoryConversationItemStorage::new());
    let empty = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let conversation = empty["id"].as_str().unwrap().to_string();
    let message = |id: &str| json!({"id": id, "type": "message", "role": "user", "content": "x"});
    let ids_of = |listed: &Value| -> Vec<String> {
        listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["id"].as_str().unwrap().to_string())
            .collect()
    };

    // The create route: a reference to the item the batch itself adds.
    let before = items.stored_record_count().unwrap();
    let created = create_conversation(
        &conversations,
        &items,
        None,
        json!({"items": [message("msg_a"), {"type": "item_reference", "id": "msg_a"}]}),
    )
    .await;
    assert_eq!(created.status(), StatusCode::OK);
    let created = json_of(created).await;
    assert!(created.get("warnings").is_none(), "{created}");
    let with_items = created["id"].as_str().unwrap().to_string();
    assert_eq!(items.stored_record_count(), Some(before + 1));
    let listed = json_of(
        list_conversation_items(
            &conversations,
            &items,
            None,
            &with_items,
            None,
            Some("asc"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(ids_of(&listed), vec!["msg_a"], "{listed}");

    // The add-items route: the same new id twice, then an accepted-but-unimplemented
    // type referenced by its own batch (one warning, not two).
    for (batch, id, warnings) in [
        (json!([message("msg_b"), message("msg_b")]), "msg_b", 0),
        (
            json!([
                {"id": "ws_c", "type": "web_search_call", "status": "completed"},
                {"type": "item_reference", "id": "ws_c"}
            ]),
            "ws_c",
            1,
        ),
    ] {
        let before = items.stored_record_count().unwrap();
        let added = create_conversation_items(
            &conversations,
            &items,
            None,
            &conversation,
            json!({"items": batch}),
        )
        .await;
        assert_eq!(added.status(), StatusCode::OK, "{id}");
        let added = json_of(added).await;
        assert_eq!(added["data"].as_array().map(Vec::len), Some(2), "{added}");
        assert_eq!(added["data"][1]["id"], id, "{added}");
        assert_eq!(
            added["warnings"].as_array().map(Vec::len).unwrap_or(0),
            warnings,
            "{added}"
        );
        assert_eq!(items.stored_record_count(), Some(before + 1), "{id}");
    }
    let listed = json_of(
        list_conversation_items(
            &conversations,
            &items,
            None,
            &conversation,
            None,
            Some("asc"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(ids_of(&listed), vec!["msg_b", "ws_c"], "{listed}");
}

/// A reference resolved before the writes can name an item a bounded
/// store drops to make room for the batch's own new items. Every reference
/// is read again after the writes: one that is gone fails the batch before
/// anything is linked, on both routes, with the answer of a missing
/// reference.
#[tokio::test]
async fn a_reference_the_batch_s_own_writes_evict_fails_the_batch_before_the_link() {
    for route in ["create", "add-items"] {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let two_items = MemoryConfig {
            max_items: 2,
            ..MemoryConfig::default()
        };
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::with_config(&two_items));
        let stored = |id: &str| NewConversationItem {
            id: Some(ConversationItemId::from(id)),
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([{"type": "input_text", "text": id}]),
            status: None,
            tenant: None,
        };
        items.create_item(stored("msg_one")).await.unwrap();
        items.create_item(stored("msg_two")).await.unwrap();
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        let conversations_before = conversations.stored_record_count();

        // The new item takes the store over its bound: one of the two
        // stored items goes, and the batch references both.
        let batch = json!({"items": [
            {"id": "msg_new", "type": "message", "role": "user", "content": "new"},
            {"type": "item_reference", "id": "msg_one"},
            {"type": "item_reference", "id": "msg_two"}
        ]});
        let response = match route {
            "create" => create_conversation(&conversations, &items, None, batch).await,
            _ => {
                create_conversation_items(&conversations, &items, None, &conversation, batch).await
            }
        };
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{route}");
        let body = json_of(response).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("msg_one") || message.contains("msg_two"),
            "{route}: the answer names the evicted reference: {body}"
        );
        let kept = [
            items
                .get_item(&ConversationItemId::from("msg_one"))
                .await
                .unwrap(),
            items
                .get_item(&ConversationItemId::from("msg_two"))
                .await
                .unwrap(),
        ];
        assert_eq!(
            kept.iter().filter(|item| item.is_some()).count(),
            1,
            "{route}: one stored item went to make room"
        );
        assert_eq!(
            conversations.stored_record_count(),
            conversations_before,
            "{route}: no conversation is left behind"
        );
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &conversation,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(
            listed["data"].as_array().map(Vec::len),
            Some(0),
            "{route}: {listed}"
        );
        assert!(
            !items
                .is_item_linked(
                    &ConversationId::from(conversation.as_str()),
                    &ConversationItemId::from("msg_new")
                )
                .await
                .unwrap(),
            "{route}: nothing of the batch is linked"
        );
    }
}

/// An item adopted under its own id is not a reference: when the batch's
/// writes evict the stored item, the request's own content is written
/// under that id (the answer the write-as-you-go order gave), on both
/// routes; a bare id that is gone answers as a missing item.
#[tokio::test]
async fn an_adopted_id_the_batch_s_own_writes_evict_takes_the_request_s_content() {
    let two_items = MemoryConfig {
        max_items: 2,
        ..MemoryConfig::default()
    };
    let stored = |id: &str| NewConversationItem {
        id: Some(ConversationItemId::from(id)),
        response_id: None,
        item_type: "message".to_string(),
        role: Some("user".to_string()),
        content: json!([{"type": "input_text", "text": "stored"}]),
        status: None,
        tenant: None,
    };
    for route in ["create", "add-items"] {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::with_config(&two_items));
        // The oldest record goes first: the two stored items get distinct stamps.
        items.create_item(stored("msg_oldest")).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        items.create_item(stored("msg_other")).await.unwrap();
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        let batch = json!({"items": [
            {"type": "message", "role": "user", "content": "new"},
            {"id": "msg_oldest", "type": "message", "role": "user", "content": "own"}
        ]});
        let response = match route {
            "create" => create_conversation(&conversations, &items, None, batch).await,
            _ => {
                create_conversation_items(&conversations, &items, None, &conversation, batch).await
            }
        };
        assert_eq!(response.status(), StatusCode::OK, "{route}");
        let body = json_of(response).await;
        let listed_in = if route == "create" {
            body["id"].as_str().unwrap().to_string()
        } else {
            assert_eq!(
                body["data"][1]["content"][0]["text"], "own",
                "{route}: {body}"
            );
            conversation.clone()
        };
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &listed_in,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        let texts: Vec<&str> = listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(texts, vec!["new", "own"], "{route}: {listed}");
        assert_eq!(listed["data"][1]["id"], "msg_oldest", "{route}: {listed}");
    }

    // A bare id (nothing the parser takes) adopted and then evicted by the
    // batch's write answers as a missing item, not a reference.
    let conversations: Arc<dyn ConversationStorage> = Arc::new(MemoryConversationStorage::new());
    let items: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::with_config(&two_items));
    items.create_item(stored("msg_oldest")).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    items.create_item(stored("msg_other")).await.unwrap();
    let empty = json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
    let conversation = empty["id"].as_str().unwrap().to_string();
    let response = create_conversation_items(
        &conversations,
        &items,
        None,
        &conversation,
        json!({"items": [
            {"type": "message", "role": "user", "content": "new"},
            {"id": "msg_oldest"}
        ]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = json_of(response).await;
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("msg_oldest") && !message.contains("Referenced"),
        "{body}"
    );
}
/// An adopted id the batch's writes evict is written once with the
/// request's own content, and every entry of the answer that names it (the
/// adoption, a reference to it, a repeat of it) lists the item as the
/// listing does, on both routes.
#[tokio::test]
async fn a_recreated_adopted_id_lists_the_same_in_every_entry_of_the_answer() {
    let two_items = MemoryConfig {
        max_items: 2,
        ..MemoryConfig::default()
    };
    let stored = |id: &str| NewConversationItem {
        id: Some(ConversationItemId::from(id)),
        response_id: None,
        item_type: "message".to_string(),
        role: Some("user".to_string()),
        content: json!([{"type": "input_text", "text": "stored"}]),
        status: None,
        tenant: None,
    };
    for route in ["create", "add-items"] {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::with_config(&two_items));
        // The oldest record goes first: the two stored items get distinct stamps.
        items.create_item(stored("msg_oldest")).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        items.create_item(stored("msg_other")).await.unwrap();
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        // The new item evicts the oldest; the batch then adopts that id, references it and
        // adopts it again.
        let batch = json!({"items": [
            {"type": "message", "role": "user", "content": "new"},
            {"id": "msg_oldest", "type": "message", "role": "user", "content": "own"},
            {"type": "item_reference", "id": "msg_oldest"},
            {"id": "msg_oldest", "type": "message", "role": "user", "content": "own again"}
        ]});
        let response = match route {
            "create" => create_conversation(&conversations, &items, None, batch).await,
            _ => {
                create_conversation_items(&conversations, &items, None, &conversation, batch).await
            }
        };
        assert_eq!(response.status(), StatusCode::OK, "{route}");
        let body = json_of(response).await;
        let listed_in = if route == "create" {
            body["id"].as_str().unwrap().to_string()
        } else {
            let texts: Vec<&str> = body["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
                .collect();
            assert_eq!(texts, vec!["new", "own", "own", "own"], "{route}: {body}");
            conversation.clone()
        };
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &listed_in,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        let texts: Vec<&str> = listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(texts, vec!["new", "own"], "{route}: {listed}");
        assert_eq!(listed["data"][1]["id"], "msg_oldest", "{route}: {listed}");
        assert!(
            items
                .get_item(&ConversationItemId::from("msg_other"))
                .await
                .unwrap()
                .is_none(),
            "{route}: the other stored item went to make room for the recreation"
        );
    }
}

/// The post-write re-check applies the prepare step's visibility rule: when
/// the batch's own writes evicted an adopted or referenced item and another
/// caller's write took the id before the re-check, the item now under the id
/// is refused as the prepare step refuses it (an adoption's 400, a
/// reference's 404) and nothing of the batch is linked; the other caller's
/// item is untouched.
#[tokio::test]
async fn an_id_another_caller_took_before_the_re_check_is_refused_before_the_link() {
    for (case, entry, status) in [
        (
            "adoption",
            json!({"id": "msg_o", "type": "message", "role": "user", "content": "own"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "reference",
            json!({"type": "item_reference", "id": "msg_o"}),
            StatusCode::NOT_FOUND,
        ),
    ] {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let two_items = MemoryConfig {
            max_items: 2,
            ..MemoryConfig::default()
        };
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::with_config(&two_items));
        let stored = |id: &str, tenant: Option<&str>| NewConversationItem {
            id: Some(ConversationItemId::from(id)),
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([{"type": "input_text", "text": id}]),
            status: None,
            tenant: tenant.map(str::to_string),
        };
        items.create_item(stored("msg_o", A)).await.unwrap();
        items.create_item(stored("msg_u", A)).await.unwrap();
        let created =
            json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
        let conv_id = created["id"].as_str().unwrap().to_string();
        let conversation = ConversationId::from(conv_id.as_str());

        // The batch is prepared while msg_o is a's own item.
        let batch = [
            json!({"id": "msg_new", "type": "message", "role": "user", "content": "new"}),
            entry,
        ];
        let prepared = match prepare_items(&items, Some(&conversation), &batch, A).await {
            Ok(prepared) => prepared,
            Err(response) => panic!("{case}: prepared: {}", json_of(response).await),
        };
        // Between the prepare and the writes: a write evicts msg_o (the
        // oldest record of the bounded store) and another caller's write
        // takes the id.
        items.create_item(stored("msg_filler", A)).await.unwrap();
        items.create_item(stored("msg_o", B)).await.unwrap();

        let response = store_items(&items, &conversation, prepared, A)
            .await
            .expect_err("the id now holds another caller's item");
        assert_eq!(response.status(), status, "{case}");
        let body = json_of(response).await;
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("msg_o"),
            "{case}: the answer names the id: {body}"
        );
        for id in ["msg_o", "msg_new"] {
            assert!(
                !items
                    .is_item_linked(&conversation, &ConversationItemId::from(id))
                    .await
                    .unwrap(),
                "{case}: {id} is not linked"
            );
        }
        let listed = json_of(
            list_conversation_items(&conversations, &items, A, &conv_id, None, Some("asc"), None)
                .await,
        )
        .await;
        assert_eq!(
            listed["data"].as_array().map(Vec::len),
            Some(0),
            "{case}: {listed}"
        );
        let other = items
            .get_item(&ConversationItemId::from("msg_o"))
            .await
            .unwrap()
            .expect("the other caller's item stands");
        assert_eq!(other.tenant.as_deref(), B, "{case}");
        assert_eq!(other.content[0]["text"], "msg_o", "{case}");
    }
}
