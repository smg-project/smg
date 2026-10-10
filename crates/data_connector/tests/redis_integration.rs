//! Integration tests for the Redis backend against a live Redis server.
//!
//! Opt-in (`#[ignore]`) like the Postgres ones, since they need a real server.
//! To run locally:
//!
//!   docker run -d -p 6379:6379 redis:7
//!   DATA_CONNECTOR_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
//!       cargo test -p data-connector --test redis_integration -- --ignored

use data_connector::{
    create_storage, HistoryBackend, RedisConfig, StorageBundle, StorageFactoryConfig,
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
