//! Integration tests for the Redis backend against a live Redis server.
//!
//! Opt-in (`#[ignore]`) like the Postgres ones, since they need a real server.
//! To run locally:
//!
//!   docker run -d -p 6379:6379 redis:7
//!   DATA_CONNECTOR_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
//!       cargo test -p data-connector --test redis_integration -- --ignored

use chrono::{TimeDelta, Utc};
use data_connector::{
    create_storage, ConversationItemId, HistoryBackend, ListParams, NewConversation,
    NewConversationItem, RedisConfig, SortOrder, StorageBundle, StorageFactoryConfig,
    StoredResponse,
};
use serde_json::json;

fn test_redis_url() -> Option<String> {
    std::env::var("DATA_CONNECTOR_TEST_REDIS_URL").ok()
}

async fn redis_bundle(url: &str) -> Result<StorageBundle, String> {
    let redis_cfg = RedisConfig {
        url: url.to_string(),
        pool_max: 4,
        schema: None,
        retention_days: Some(1),
    };
    create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        hook: None,
    })
    .await
}

/// A response with two output items whose ids are unique to this run (the
/// server keeps the keys of earlier runs).
fn response_with_output_items() -> (StoredResponse, String, String) {
    let mut stored = StoredResponse::new(None);
    let msg_id = format!("msg_{}", stored.id.0);
    let rs_id = format!("rs_{}", stored.id.0);
    stored.input = json!([{"type": "message", "role": "user", "content": "hi"}]);
    stored.raw_response = json!({
        "id": stored.id.0,
        "output": [
            {"type": "reasoning", "id": rs_id, "summary": []},
            {"type": "message", "id": msg_id, "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "hello", "annotations": []}]}
        ]
    });
    (stored, msg_id, rs_id)
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn find_response_by_output_item_follows_the_keys_written_at_store_time() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let resp = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage")
        .response_storage;

    let (stored, msg_id, rs_id) = response_with_output_items();
    let response_id = resp.store_response(stored).await.expect("store response");

    for item_id in [&msg_id, &rs_id] {
        let found = resp
            .find_response_by_output_item(item_id)
            .await
            .expect("lookup by output item id");
        assert_eq!(found.map(|r| r.id), Some(response_id.clone()), "{item_id}");
    }
    let unknown = resp
        .find_response_by_output_item(&format!("msg_unknown_{}", response_id.0))
        .await
        .expect("lookup of an unknown id");
    assert!(unknown.is_none());

    // The keys go with the response.
    resp.delete_response(&response_id)
        .await
        .expect("delete response");
    let gone = resp
        .find_response_by_output_item(&msg_id)
        .await
        .expect("lookup after delete");
    assert!(gone.is_none());
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn deleting_the_responses_of_an_identifier_drops_their_output_item_keys() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let resp = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage")
        .response_storage;

    let (mut stored, msg_id, _) = response_with_output_items();
    let identifier = format!("sid_{}", stored.id.0);
    stored.safety_identifier = Some(identifier.clone());
    resp.store_response(stored).await.expect("store response");
    assert!(resp
        .find_response_by_output_item(&msg_id)
        .await
        .expect("lookup by output item id")
        .is_some());

    let deleted = resp
        .delete_identifier_responses(&identifier)
        .await
        .expect("delete the identifier's responses");
    assert_eq!(deleted, 1);
    let gone = resp
        .find_response_by_output_item(&msg_id)
        .await
        .expect("lookup after delete");
    assert!(gone.is_none());
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn a_second_link_of_an_item_moves_it_and_the_item_lists_once() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let bundle = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage");
    let conversation = bundle
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");
    let items = bundle.conversation_item_storage;

    // Two items, their ids unique to this run (the server keeps the keys of earlier runs).
    let nonce = &conversation.id.0;
    let ids: Vec<ConversationItemId> = ["a", "b"]
        .iter()
        .map(|s| ConversationItemId::from(format!("msg_{s}_{nonce}").as_str()))
        .collect();
    for id in &ids {
        items
            .create_item(NewConversationItem {
                id: Some(id.clone()),
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([]),
                status: Some("completed".to_string()),
            })
            .await
            .expect("create item");
    }
    let base = Utc::now();
    let at = |seconds: i64| base + TimeDelta::seconds(seconds);
    items
        .link_item(&conversation.id, &ids[0], at(0))
        .await
        .expect("link item");
    items
        .link_item(&conversation.id, &ids[1], at(1))
        .await
        .expect("link item");

    // The sorted set holds one member per item: a second link of the first item moves it to
    // the new score instead of adding a second entry. The gateway refuses a duplicate before
    // it gets here, so a client sees the same 400 on every backend.
    items
        .link_item(&conversation.id, &ids[0], at(2))
        .await
        .expect("link item again");

    let list = |order: SortOrder| {
        let items = items.clone();
        let conversation_id = conversation.id.clone();
        let params = ListParams {
            limit: 10,
            order,
            after: None,
        };
        async move {
            items
                .list_items(&conversation_id, params)
                .await
                .expect("list items")
                .into_iter()
                .map(|item| item.id)
                .collect::<Vec<_>>()
        }
    };
    assert_eq!(list(SortOrder::Asc).await, [ids[1].clone(), ids[0].clone()]);

    // One member per item: the delete takes the item out of the conversation.
    items
        .delete_item(&conversation.id, &ids[0])
        .await
        .expect("delete item");
    assert!(!items
        .is_item_linked(&conversation.id, &ids[0])
        .await
        .expect("is linked"));
    assert_eq!(list(SortOrder::Asc).await, std::slice::from_ref(&ids[1]));
}
