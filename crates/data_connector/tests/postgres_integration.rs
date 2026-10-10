//! Integration tests for the Postgres backend against a live PostgreSQL instance.
//!
//! Regression coverage for https://github.com/smg-project/smg/issues/1930: the
//! `responses.input`/`responses.raw_response`, `conversations.metadata`, and
//! `conversation_items.content` columns are native Postgres `JSON` columns.
//! `tokio-postgres` panics with `WrongType` if a `JSON` column is fetched as
//! `String` instead of `serde_json::Value` — these tests exercise the real
//! read paths (`get_response`, `get_response_chain`, `get_conversation`,
//! `get_item`) against a real server, since that panic can't be triggered
//! through a unit test with no database.
//!
//! These stay opt-in (`#[ignore]`) rather than running under plain
//! `cargo test`, since they need a real database. CI provisions a shared
//! Postgres instance (`postgres-db` in `agentic-services.yaml`), creates a
//! database unique to the run, and runs these with `--ignored` at the end
//! of the `Run Rust tests` step. To run locally:
//!
//!   docker run -d -e POSTGRES_PASSWORD=test -e POSTGRES_DB=smg_test \
//!       -p 5432:5432 postgres:16
//!   DATA_CONNECTOR_TEST_POSTGRES_URL=postgres://postgres:test@localhost:5432/smg_test \
//!       cargo test -p data-connector --test postgres_integration -- --ignored --test-threads=1
//!
//! `--test-threads=1` matters: each test independently calls `create_storage`,
//! which runs `CREATE TABLE IF NOT EXISTS` and schema migrations as a side
//! effect. Postgres doesn't serialize concurrent first-time DDL on the same
//! table name, so running these in parallel against a fresh database races.

use data_connector::{
    create_storage, HistoryBackend, NewConversation, NewConversationItem, PostgresConfig,
    SchemaConfig, StorageBundle, StorageFactoryConfig, StoredResponse,
};
use serde_json::json;

fn test_db_url() -> Option<String> {
    std::env::var("DATA_CONNECTOR_TEST_POSTGRES_URL").ok()
}

async fn postgres_bundle(db_url: &str) -> Result<StorageBundle, String> {
    let postgres_cfg = PostgresConfig {
        db_url: db_url.to_string(),
        pool_max: 4,
        // The shared test database starts with no migrations applied; opt in
        // to auto-migrate so `create_storage` doesn't require a human to run
        // the pending migrations by hand first.
        schema: Some(SchemaConfig {
            auto_migrate: true,
            ..Default::default()
        }),
    };
    create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Postgres,
        oracle: None,
        postgres: Some(&postgres_cfg),
        redis: None,
        hook: None,
    })
    .await
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn store_and_get_response_round_trips_json_columns() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let resp = bundle.response_storage;

    // Mirrors the issue's repro: store an initial response, then a follow-up
    // that continues from it via `previous_response_id`.
    let mut first = StoredResponse::new(None);
    first.input = json!(["Remember the token PG-HISTORY-TEST."]);
    first.raw_response = json!({"output": [{"type": "message", "content": "ok"}]});
    let first_id = resp
        .store_response(first.clone())
        .await
        .expect("store first response");

    let mut second = StoredResponse::new(Some(first_id.clone()));
    second.input = json!(["What token did I ask you to remember?"]);
    second.raw_response = json!({"output": [{"type": "message", "content": "PG-HISTORY-TEST"}]});
    let second_id = resp
        .store_response(second.clone())
        .await
        .expect("store follow-up response");

    // This is the exact panic path from issue #1930: fetching a response
    // whose `input`/`raw_response` are native Postgres `JSON` columns must
    // decode cleanly instead of panicking with `WrongType`.
    let fetched_first = resp
        .get_response(&first_id)
        .await
        .expect("get_response must not panic on JSON columns")
        .expect("first response should exist");
    assert_eq!(fetched_first.input, first.input);
    assert_eq!(fetched_first.raw_response, first.raw_response);

    let fetched_second = resp
        .get_response(&second_id)
        .await
        .expect("get_response must not panic on JSON columns")
        .expect("second response should exist");
    assert_eq!(fetched_second.previous_response_id, Some(first_id));
    assert_eq!(fetched_second.input, second.input);
    assert_eq!(fetched_second.raw_response, second.raw_response);
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn get_response_chain_walks_json_columns() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let resp = bundle.response_storage;

    let mut r1 = StoredResponse::new(None);
    r1.input = json!(["first"]);
    let id1 = resp.store_response(r1).await.expect("store r1");

    let mut r2 = StoredResponse::new(Some(id1.clone()));
    r2.input = json!(["second"]);
    let id2 = resp.store_response(r2).await.expect("store r2");

    let mut r3 = StoredResponse::new(Some(id2.clone()));
    r3.input = json!(["third"]);
    let id3 = resp.store_response(r3).await.expect("store r3");

    // The default `get_response_chain` implementation walks
    // `previous_response_id` links via repeated `get_response()` calls, so
    // this exercises the same JSON-column decoding for every hop.
    let chain = resp
        .get_response_chain(&id3, None)
        .await
        .expect("get_response_chain must not panic on JSON columns");
    let ids: Vec<_> = chain.responses.iter().map(|r| r.id.clone()).collect();
    assert_eq!(ids, vec![id1, id2, id3]);
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn conversation_metadata_round_trips_json_column() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let conv = bundle.conversation_storage;

    let mut metadata = serde_json::Map::new();
    metadata.insert("key".to_string(), json!("value"));

    let created = conv
        .create_conversation(NewConversation {
            id: None,
            metadata: Some(metadata.clone()),
        })
        .await
        .expect("create_conversation");

    let fetched = conv
        .get_conversation(&created.id)
        .await
        .expect("get_conversation must not panic on the JSON metadata column")
        .expect("conversation should exist");
    assert_eq!(fetched.metadata, Some(metadata));

    // `update_conversation` has the same String-vs-Value bind bug as
    // `create_conversation` did — cover both setting new metadata and
    // clearing it back to SQL NULL.
    let mut updated_metadata = serde_json::Map::new();
    updated_metadata.insert("key".to_string(), json!("updated"));
    updated_metadata.insert("nested".to_string(), json!({"count": 2}));

    let updated = conv
        .update_conversation(&created.id, Some(updated_metadata.clone()))
        .await
        .expect("update_conversation must not fail serializing the JSON metadata column")
        .expect("conversation should exist");
    assert_eq!(updated.metadata, Some(updated_metadata.clone()));

    let fetched_after_update = conv
        .get_conversation(&created.id)
        .await
        .expect("get_conversation must not panic after update")
        .expect("conversation should exist");
    assert_eq!(fetched_after_update.metadata, Some(updated_metadata));

    let cleared = conv
        .update_conversation(&created.id, None)
        .await
        .expect("update_conversation must not fail clearing the JSON metadata column")
        .expect("conversation should exist");
    assert_eq!(cleared.metadata, None);

    let fetched_after_clear = conv
        .get_conversation(&created.id)
        .await
        .expect("get_conversation must not panic after clearing metadata")
        .expect("conversation should exist");
    assert_eq!(fetched_after_clear.metadata, None);
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn conversation_item_content_round_trips_json_column() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let items = bundle.conversation_item_storage;

    let content = json!([{"type": "text", "text": "hello"}]);
    let item = items
        .create_item(NewConversationItem {
            id: None,
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: content.clone(),
            status: Some("completed".to_string()),
        })
        .await
        .expect("create_item");

    let fetched = items
        .get_item(&item.id)
        .await
        .expect("get_item must not panic on the JSON content column")
        .expect("item should exist");
    assert_eq!(fetched.content, content);
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn find_response_by_output_item_queries_the_stored_output() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let resp = bundle.response_storage;

    // The lookup behind `item_reference` input items: a stored response is
    // found by the id of any of its output items, through the JSON column.
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

    resp.delete_response(&response_id)
        .await
        .expect("delete response");
    let gone = resp
        .find_response_by_output_item(&msg_id)
        .await
        .expect("lookup after delete");
    assert!(gone.is_none());
}
