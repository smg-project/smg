//! The contract every history backend keeps, as one scenario run on each
//! store: on the in-process memory store always, and on a live Redis,
//! Postgres or Oracle when its url is set (opt-in, `#[ignore]`, as the
//! backend suites are):
//!
//!   DATA_CONNECTOR_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
//!   DATA_CONNECTOR_TEST_POSTGRES_URL=postgres://postgres:test@localhost:5432/smg_test \
//!   DATA_CONNECTOR_TEST_ORACLE_URL=oracle://user:password@localhost:1521/FREEPDB1 \
//!       cargo test -p data-connector --test conformance -- --ignored --test-threads=1
//!
//! The Oracle url names the user, the password and the connect descriptor
//! (`host:port/service`); the Oracle client library must be on the library
//! path for that one.

// A test binary: a step that fails is the test's failure, said where it failed.
#![allow(clippy::expect_used, clippy::panic)]

use chrono::{TimeDelta, Utc};
use data_connector::{
    create_storage, ConversationItem, ConversationItemId, HistoryBackend, ListParams, MemoryConfig,
    NewConversation, NewConversationItem, OracleConfig, PostgresConfig, RedisConfig, SchemaConfig,
    SortOrder, StorageBundle, StorageFactoryConfig,
};
use serde_json::{json, Value};

/// A turn whose input echoes an item of an earlier response under the id it
/// came with, created and linked item by item as the response persist does:
/// a client that appends the `function_call` of the earlier response and its
/// `function_call_output` to its next input. The echoed item is created again
/// under its id, with the new turn's response id and the client's copy of its
/// content; the turn's other items under fresh ids. Every backend answers the
/// stored item for the echoed id, untouched, and links the turn whole; an
/// item created without an id gets a fresh one.
async fn an_echoed_item_is_the_stored_item_and_its_turn_links_whole(
    bundle: StorageBundle,
    nonce: &str,
) {
    let items = bundle.conversation_item_storage;
    let conversation = bundle
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
            tenant: None,
        })
        .await
        .expect("create conversation");
    let conversation_id = conversation.id.clone();
    // Ids unique to this run (a live store keeps the keys of earlier runs),
    // within the 64 characters of the id column.
    let id = |name: &str| ConversationItemId::from(format!("{name}_{nonce}").as_str());
    let new_item = |id: Option<ConversationItemId>,
                    item_type: &str,
                    role: Option<&str>,
                    content: Value,
                    response_id: &str| NewConversationItem {
        id,
        response_id: Some(response_id.to_string()),
        item_type: item_type.to_string(),
        role: role.map(str::to_string),
        content,
        status: Some("completed".to_string()),
        tenant: None,
    };
    let base = Utc::now();
    let at = |seconds: i64| base + TimeDelta::seconds(seconds);

    // The first turn: the response `resp_1` ended with a tool call.
    let call = id("fc");
    let call_content = json!({
        "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":1}"
    });
    items
        .create_item(new_item(
            Some(call.clone()),
            "function_call",
            None,
            call_content.clone(),
            "resp_1",
        ))
        .await
        .expect("create the first turn's tool call");
    items
        .link_item(&conversation_id, &call, at(0))
        .await
        .expect("link the first turn's tool call");
    let stored = items
        .get_item(&call)
        .await
        .expect("get the stored tool call")
        .expect("the tool call is stored");

    // The second turn, in the persist's order: the user's message, the echoed
    // tool call (the client's copy re-serialized the arguments; the persist
    // stamps it with the new response's id), its output, the answer.
    let echoed = json!({
        "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\": 1}"
    });
    let turn = [
        (
            id("msg"),
            new_item(
                Some(id("msg")),
                "message",
                Some("user"),
                json!([{"type": "input_text", "text": "and then?"}]),
                "resp_2",
            ),
        ),
        (
            call.clone(),
            new_item(Some(call.clone()), "function_call", None, echoed, "resp_2"),
        ),
        (
            id("fco"),
            new_item(
                Some(id("fco")),
                "function_call_output",
                None,
                json!({"type": "function_call_output", "call_id": "call_1", "output": "42"}),
                "resp_2",
            ),
        ),
        (
            id("out"),
            new_item(
                Some(id("out")),
                "message",
                Some("assistant"),
                json!([{"type": "output_text", "text": "42"}]),
                "resp_2",
            ),
        ),
    ];
    let mut created: Vec<ConversationItem> = Vec::new();
    for (item_id, item) in turn {
        let answered = items.create_item(item).await.unwrap_or_else(|e| {
            panic!(
                "create item {} of the turn ({} linked before it): {e}",
                created.len() + 1,
                created.len()
            )
        });
        assert_eq!(answered.id, item_id);
        items
            .link_item(&conversation_id, &item_id, at(1 + created.len() as i64))
            .await
            .expect("link the turn's item");
        created.push(answered);
    }

    // The echoed id answered the stored item: the content and the response
    // that produced it, not the client's copy under the new response.
    let answered = &created[1];
    assert_eq!(answered.content, call_content, "the stored content");
    assert_eq!(
        answered.response_id.as_deref(),
        Some("resp_1"),
        "the response that produced the item"
    );
    assert_eq!(answered.created_at, stored.created_at, "the stored stamp");
    let read = items
        .get_item(&call)
        .await
        .expect("get the tool call")
        .expect("the tool call is still stored");
    assert_eq!(read.content, call_content);
    assert_eq!(read.response_id.as_deref(), Some("resp_1"));

    // The turn is linked whole, the echoed item at its new place (a second
    // link moves an item).
    let listed: Vec<ConversationItemId> = items
        .list_items(
            &conversation_id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .expect("list the conversation")
        .into_iter()
        .map(|item| item.id)
        .collect();
    assert_eq!(listed, [id("msg"), call, id("fco"), id("out")]);

    // An item created without an id is a new one each time.
    let first = items
        .create_item(new_item(None, "message", Some("user"), json!([]), "resp_3"))
        .await
        .expect("create an item without an id");
    let second = items
        .create_item(new_item(None, "message", Some("user"), json!([]), "resp_3"))
        .await
        .expect("create another item without an id");
    assert_ne!(first.id, second.id);
}

/// A tag unique to this run, for the ids on a live store.
fn nonce() -> String {
    ulid::Ulid::new().to_string()
}

async fn bundle(config: StorageFactoryConfig<'_>) -> StorageBundle {
    create_storage(config)
        .await
        .unwrap_or_else(|e| panic!("initialize the storage: {e}"))
}

#[tokio::test]
async fn the_memory_store_keeps_the_item_contract() {
    let memory = MemoryConfig::default();
    let bundle = bundle(StorageFactoryConfig {
        backend: &HistoryBackend::Memory,
        oracle: None,
        postgres: None,
        redis: None,
        memory: Some(&memory),
        hook: None,
    })
    .await;
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, "memory").await;
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn the_redis_store_keeps_the_item_contract() {
    let Some(url) = std::env::var("DATA_CONNECTOR_TEST_REDIS_URL").ok() else {
        return;
    };
    let redis = RedisConfig {
        url,
        pool_max: 4,
        op_timeout_secs: RedisConfig::default_op_timeout_secs(),
        schema: None,
        retention_days: Some(1),
    };
    let bundle = bundle(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis),
        memory: None,
        hook: None,
    })
    .await;
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn the_postgres_store_keeps_the_item_contract() {
    let Some(db_url) = std::env::var("DATA_CONNECTOR_TEST_POSTGRES_URL").ok() else {
        return;
    };
    let postgres = PostgresConfig {
        db_url,
        pool_max: 4,
        op_timeout_secs: PostgresConfig::default_op_timeout_secs(),
        schema: Some(SchemaConfig {
            auto_migrate: true,
            ..Default::default()
        }),
    };
    let bundle = bundle(StorageFactoryConfig {
        backend: &HistoryBackend::Postgres,
        oracle: None,
        postgres: Some(&postgres),
        redis: None,
        memory: None,
        hook: None,
    })
    .await;
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}

#[tokio::test]
#[ignore = "requires a live Oracle database; set DATA_CONNECTOR_TEST_ORACLE_URL and run with -- --ignored"]
async fn the_oracle_store_keeps_the_item_contract() {
    let Some(url) = std::env::var("DATA_CONNECTOR_TEST_ORACLE_URL").ok() else {
        return;
    };
    let url = url::Url::parse(&url)
        .expect("DATA_CONNECTOR_TEST_ORACLE_URL is oracle://user:password@host:port/service");
    let host = url.host_str().expect("a host in the Oracle url");
    let service = url.path().trim_start_matches('/');
    let connect_descriptor = match url.port() {
        Some(port) => format!("{host}:{port}/{service}"),
        None => format!("{host}/{service}"),
    };
    let oracle = OracleConfig {
        wallet_path: None,
        connect_descriptor,
        external_auth: false,
        username: url.username().to_string(),
        password: url.password().unwrap_or_default().to_string(),
        pool_min: OracleConfig::default_pool_min(),
        pool_max: 4,
        pool_timeout_secs: OracleConfig::default_pool_timeout_secs(),
        schema: Some(SchemaConfig {
            auto_migrate: true,
            ..Default::default()
        }),
    };
    let bundle = bundle(StorageFactoryConfig {
        backend: &HistoryBackend::Oracle,
        oracle: Some(&oracle),
        postgres: None,
        redis: None,
        memory: None,
        hook: None,
    })
    .await;
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}
