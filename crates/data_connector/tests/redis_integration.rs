//! Integration tests for the Redis backend against a live Redis server.
//!
//! Opt-in (`#[ignore]`) like the Postgres ones, since they need a real server.
//! To run locally:
//!
//!   docker run -d -p 6379:6379 redis:7
//!   DATA_CONNECTOR_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
//!       cargo test -p data-connector --test redis_integration -- --ignored

use std::time::{Duration, Instant};

use chrono::{TimeDelta, Utc};
use data_connector::{
    create_storage, ConditionalStore, ConversationItemId, HistoryBackend, ListParams,
    NewConversation, NewConversationItem, RedisConfig, ResponseId, SchemaConfig, SortOrder,
    StorageBundle, StorageFactoryConfig, StoredResponse,
};
use serde_json::json;

mod common;
use common::Relay;

fn test_redis_url() -> Option<String> {
    std::env::var("DATA_CONNECTOR_TEST_REDIS_URL").ok()
}

async fn redis_bundle(url: &str) -> Result<StorageBundle, String> {
    let redis_cfg = RedisConfig {
        url: url.to_string(),
        pool_max: 4,
        op_timeout_secs: RedisConfig::default_op_timeout_secs(),
        schema: None,
        retention_days: Some(1),
    };
    create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        memory: None,
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
        .delete_identifier_responses(&identifier, None)
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
            tenant: None,
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
                tenant: None,
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

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn deleting_a_conversation_answers_once_whether_it_existed() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let bundle = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage");
    let conversations = bundle.conversation_storage;
    let created = conversations
        .create_conversation(NewConversation::default())
        .await
        .expect("create conversation");

    // The delete pipeline removes the conversation key and its item index
    // (two counts); the store answers whether the conversation existed.
    let deleted = conversations
        .delete_conversation(&created.id)
        .await
        .expect("delete an existing conversation");
    assert!(deleted, "the conversation existed");
    let again = conversations
        .delete_conversation(&created.id)
        .await
        .expect("delete a conversation that is gone");
    assert!(!again, "a second delete finds nothing");
    assert!(conversations
        .get_conversation(&created.id)
        .await
        .expect("lookup after delete")
        .is_none());
}

/// A store that accepts the connection and then stops answering (a frozen or
/// partitioned server, a stalled proxy) fails the call at the configured
/// bound, with an error that names the backend, the call and the bound; once
/// the store answers again, the same pool serves the next call.
#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn a_store_that_stops_answering_fails_the_call_at_its_bound() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let (relay, relayed_url) = Relay::in_front_of(&url, 6379).expect("start the relay");
    // The bound as an operator configures it; one second here.
    let redis_cfg: RedisConfig = serde_json::from_value(json!({
        "url": relayed_url,
        "pool_max": 4,
        "op_timeout_secs": 1,
        "retention_days": 1,
    }))
    .expect("a Redis config with a call bound");
    let resp = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        memory: None,
        hook: None,
    })
    .await
    .expect("failed to initialize Redis storage")
    .response_storage;
    let mut stored = StoredResponse::new(None);
    stored.raw_response = json!({"id": stored.id.0, "status": "completed"});
    let id = resp.store_response(stored).await.expect("store response");

    relay.hold();
    let started = Instant::now();
    let outcome = tokio::time::timeout(Duration::from_secs(10), resp.get_response(&id)).await;
    let elapsed = started.elapsed();
    relay.release();
    let result = outcome.unwrap_or_else(|_| {
        panic!("the read was still waiting on the store after 10 s: nothing bounds a store call")
    });
    let err = result.expect_err("a store that does not answer must fail the call");
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(5),
        "the call failed after {elapsed:?}; the bound is 1 s"
    );
    let message = err.to_string();
    assert!(
        message.contains("redis get_response") && message.contains("bound of 1 s"),
        "{message}"
    );

    // The store answers again: the pool serves the next call.
    let read_back = tokio::time::timeout(Duration::from_secs(10), resp.get_response(&id))
        .await
        .expect("the read after the release must return")
        .expect("the read after the release must succeed");
    assert_eq!(read_back.map(|r| r.id), Some(id));
}

/// A store a few hundred milliseconds away still answers: the configured
/// call bound, not the client library's own defaults, decides how slow is
/// too slow.
#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn a_store_a_few_hundred_milliseconds_away_still_answers() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let (relay, relayed_url) = Relay::in_front_of(&url, 6379).expect("start the relay");
    // 400 ms on each direction: about 0.8 s per round trip, connecting included.
    relay.delay(Duration::from_millis(400));
    // The default bound.
    let redis_cfg: RedisConfig = serde_json::from_value(json!({
        "url": relayed_url,
        "pool_max": 4,
        "retention_days": 1,
    }))
    .expect("a Redis config with the default call bound");
    let resp = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        memory: None,
        hook: None,
    })
    .await
    .expect("failed to initialize Redis storage")
    .response_storage;
    let mut stored = StoredResponse::new(None);
    stored.raw_response = json!({"id": stored.id.0, "status": "completed"});
    let started = Instant::now();
    let id = resp
        .store_response(stored)
        .await
        .expect("a store a few hundred milliseconds away must take the write");
    let read_back = resp
        .get_response(&id)
        .await
        .expect("a store a few hundred milliseconds away must answer the read");
    assert_eq!(read_back.map(|r| r.id), Some(id));
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "two calls took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let bundle = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage");
    let resp = bundle.response_storage;

    // Three responses under one safety identifier (unique to this run): tenant
    // a's, tenant b's and one stored under no tenant.
    let mut of_a = StoredResponse::new(None);
    let identifier = format!("sid_{}", of_a.id.0);
    of_a.safety_identifier = Some(identifier.clone());
    of_a.tenant = Some("header:a".to_string());
    of_a.input = json!(["a"]);
    of_a.raw_response = json!({"output": []});
    let mut of_b = StoredResponse::new(None);
    of_b.safety_identifier = Some(identifier.clone());
    of_b.tenant = Some("header:b".to_string());
    of_b.input = json!(["b"]);
    of_b.raw_response = json!({"output": []});
    let mut of_nobody = StoredResponse::new(None);
    of_nobody.safety_identifier = Some(identifier.clone());
    of_nobody.input = json!(["nobody"]);
    of_nobody.raw_response = json!({"output": []});
    let id_a = resp.store_response(of_a).await.expect("store a's");
    let id_b = resp.store_response(of_b).await.expect("store b's");
    let id_nobody = resp
        .store_response(of_nobody)
        .await
        .expect("store the untenanted one");

    // The tenant field round-trips.
    let fetched = resp.get_response(&id_a).await.unwrap().expect("a's record");
    assert_eq!(fetched.tenant.as_deref(), Some("header:a"));
    let fetched = resp
        .get_response(&id_nobody)
        .await
        .unwrap()
        .expect("the untenanted record");
    assert_eq!(fetched.tenant, None);

    // The listing by safety identifier is scoped at the store.
    let ids = |responses: Vec<StoredResponse>| {
        let mut ids: Vec<String> = responses.into_iter().map(|r| r.id.0).collect();
        ids.sort();
        ids
    };
    let mut expected = vec![id_a.0.clone(), id_nobody.0.clone()];
    expected.sort();
    let listed = resp
        .list_identifier_responses(&identifier, Some("header:a"), None)
        .await
        .expect("list as a");
    assert_eq!(ids(listed), expected, "a's record and the untenanted one");
    let listed = resp
        .list_identifier_responses(&identifier, None, Some(10))
        .await
        .expect("list without a tenant");
    assert_eq!(ids(listed), vec![id_nobody.0.clone()]);

    // So is the deletion: a's call takes a's record and the untenanted one, b's stays.
    let deleted = resp
        .delete_identifier_responses(&identifier, Some("header:a"))
        .await
        .expect("delete as a");
    assert_eq!(deleted, 2);
    assert!(resp.get_response(&id_a).await.unwrap().is_none());
    assert!(resp.get_response(&id_b).await.unwrap().is_some());
    let left = resp
        .list_identifier_responses(&identifier, Some("header:b"), None)
        .await
        .expect("list as b");
    assert_eq!(ids(left), vec![id_b.0.clone()]);

    // A conversation keeps its tenant through a read and a metadata update.
    let conversations = bundle.conversation_storage;
    let created = conversations
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
            tenant: Some("header:a".to_string()),
        })
        .await
        .expect("create conversation");
    let fetched = conversations
        .get_conversation(&created.id)
        .await
        .expect("get conversation")
        .expect("created");
    assert_eq!(fetched.tenant.as_deref(), Some("header:a"));
    let metadata = json!({"topic": "tenants"}).as_object().cloned();
    let updated = conversations
        .update_conversation(&created.id, metadata.clone())
        .await
        .expect("update conversation")
        .expect("exists");
    assert_eq!(updated.tenant.as_deref(), Some("header:a"));
    assert_eq!(updated.metadata, metadata);
    // The conversation is left in place: the Redis delete of a conversation is
    // covered on its own, and the server keeps the keys of earlier runs anyway.
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let bundle = redis_bundle(&url).await.expect("create storage");
    let storage = bundle.response_storage;

    let base = StoredResponse::new(None);
    let record = |status: &str| {
        let mut stored = base.clone();
        stored.raw_response =
            json!({"id": stored.id.0, "background": true, "status": status, "output": []});
        stored
    };
    // Missing: nothing is written.
    assert!(matches!(
        storage
            .store_response_if_open(record("in_progress"))
            .await
            .expect("conditional store"),
        ConditionalStore::Missing
    ));
    assert!(storage.get_response(&base.id).await.expect("get").is_none());

    // Open: replaced.
    storage
        .store_response(record("queued"))
        .await
        .expect("store");
    assert!(matches!(
        storage
            .store_response_if_open(record("in_progress"))
            .await
            .expect("conditional store"),
        ConditionalStore::Stored
    ));
    let stored = storage
        .get_response(&base.id)
        .await
        .expect("get")
        .expect("stored");
    assert_eq!(stored.raw_response["status"], "in_progress");
    assert!(matches!(
        storage
            .store_response_if_open(record("cancelled"))
            .await
            .expect("conditional store"),
        ConditionalStore::Stored
    ));

    // Closed: refused, the record handed back as it is.
    let ConditionalStore::Closed(current) = storage
        .store_response_if_open(record("completed"))
        .await
        .expect("conditional store")
    else {
        panic!("a cancelled record is closed");
    };
    assert_eq!(current.raw_response["status"], "cancelled");
    let stored = storage
        .get_response(&base.id)
        .await
        .expect("get")
        .expect("stored");
    assert_eq!(stored.raw_response["status"], "cancelled");
    storage.delete_response(&base.id).await.expect("delete");
}

/// Under a schema that skips the object column there is no status to check:
/// the conditional store writes while the record exists and answers
/// `Missing` only for a record that is gone, as the SQL stores' plain update
/// does for the same schema (it answered `Missing` for every record before
/// and wrote nothing). Under the default schema a hash that lost its object
/// field counts as open the same way, as a NULL object column does in the
/// SQL stores.
#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn a_conditional_store_without_an_object_column_writes_while_the_record_exists() {
    let Some(url) = test_redis_url() else {
        return;
    };
    let mut schema = SchemaConfig::default();
    schema
        .responses
        .skip_columns
        .insert("raw_response".to_string());
    let redis_cfg = RedisConfig {
        url: url.clone(),
        pool_max: 4,
        op_timeout_secs: RedisConfig::default_op_timeout_secs(),
        schema: Some(schema),
        retention_days: Some(1),
    };
    let resp = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis_cfg),
        memory: None,
        hook: None,
    })
    .await
    .expect("failed to initialize Redis storage")
    .response_storage;

    let mut queued = StoredResponse::new(None);
    queued.model = Some("m".to_string());
    queued.raw_response = json!({"id": queued.id.0, "status": "queued"});
    let id = resp
        .store_response(queued.clone())
        .await
        .expect("store response");
    let mut result = queued.clone();
    result.model = Some("m2".to_string());
    result.raw_response = json!({"id": id.0, "status": "completed"});
    assert!(
        matches!(
            resp.store_response_if_open(result)
                .await
                .expect("conditional store"),
            ConditionalStore::Stored
        ),
        "the record exists: the write lands"
    );
    let read = resp
        .get_response(&id)
        .await
        .expect("read")
        .expect("the record");
    assert_eq!(read.model.as_deref(), Some("m2"), "the write landed");
    assert!(read.raw_response.is_null(), "no object column");
    // Nothing closes a record without a status: a further write lands too,
    // and a transition has nothing to move.
    assert!(matches!(
        resp.store_response_if_open(queued.clone())
            .await
            .expect("conditional store"),
        ConditionalStore::Stored
    ));
    assert!(matches!(
        resp.update_response_object_if_open(&id, json!({"id": id.0, "status": "cancelled"}))
            .await
            .expect("conditional update"),
        ConditionalStore::Stored
    ));
    let mut unknown = queued.clone();
    unknown.id = ResponseId::new();
    assert!(
        matches!(
            resp.store_response_if_open(unknown)
                .await
                .expect("conditional store"),
            ConditionalStore::Missing
        ),
        "a record that is gone takes nothing"
    );
    resp.delete_response(&id).await.expect("delete response");

    // The default schema: a hash that lost its object field is open, not missing.
    let resp = redis_bundle(&url)
        .await
        .expect("failed to initialize Redis storage")
        .response_storage;
    let mut queued = StoredResponse::new(None);
    queued.raw_response = json!({"id": queued.id.0, "status": "queued"});
    let id = resp
        .store_response(queued.clone())
        .await
        .expect("store response");
    let client = redis::Client::open(url.as_str()).expect("a client for the test's own commands");
    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .expect("connect");
    let removed: i64 = redis::cmd("HDEL")
        .arg(format!("response:{}", id.0))
        .arg("raw_response")
        .query_async(&mut conn)
        .await
        .expect("drop the object field");
    assert_eq!(removed, 1);
    let mut completed = queued.clone();
    completed.raw_response = json!({"id": id.0, "status": "completed"});
    assert!(
        matches!(
            resp.store_response_if_open(completed)
                .await
                .expect("conditional store"),
            ConditionalStore::Stored
        ),
        "a hash without the object field is open"
    );
    let read = resp
        .get_response(&id)
        .await
        .expect("read")
        .expect("the record");
    assert_eq!(read.raw_response["status"], "completed");
    resp.delete_response(&id).await.expect("delete response");
}
