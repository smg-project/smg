//! The contract every history backend keeps, as scenarios run on each
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
    create_storage, ConditionalStore, ConversationItem, ConversationItemId, HistoryBackend,
    ListParams, MemoryConfig, NewConversation, NewConversationItem, OracleConfig, PostgresConfig,
    RedisConfig, SchemaConfig, SortOrder, StorageBundle, StorageFactoryConfig, StoredResponse,
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

/// A second link of an item the conversation already has, through either
/// link path, moves it to the new stamp instead of adding a second entry or
/// being refused: every backend keeps one link per (conversation, item), so
/// the item lists once, at its new place, in both orders, and its delete
/// takes it out of the conversation. The gateway refuses a duplicate before
/// it gets here, so a client sees the same 400 on every backend.
async fn a_second_link_of_an_item_moves_it_and_the_item_lists_once(bundle: StorageBundle) {
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

    // Two items, their ids unique to this run through the conversation's (a
    // live store keeps the keys of earlier runs) and within the 64 characters
    // of the id column.
    let ids: Vec<ConversationItemId> = ["a", "b"]
        .iter()
        .map(|s| ConversationItemId::from(format!("msg_{s}_{}", conversation.id.0).as_str()))
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

    // One link per (conversation, item): a second link of the first item,
    // through either link path, moves it to the new stamp.
    items
        .link_item(&conversation.id, &ids[0], at(2))
        .await
        .expect("a second link moves the item");
    items
        .link_items(&conversation.id, &[(ids[0].clone(), at(3))])
        .await
        .expect("a second batch link moves the item");

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
    assert_eq!(
        list(SortOrder::Desc).await,
        [ids[0].clone(), ids[1].clone()]
    );

    // One link per item: the delete takes the item out of the conversation.
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

/// A response and a conversation keep the tenant they were stored under, and
/// the queries by safety identifier are scoped at the store: a caller of a
/// tenant lists and deletes that tenant's records and the untenanted ones,
/// never another tenant's; a caller of no tenant the untenanted ones alone.
async fn responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped(
    bundle: StorageBundle,
) {
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

    // The tenant round-trips.
    let fetched = resp
        .get_response(&id_a)
        .await
        .expect("get a's record")
        .expect("a's record");
    assert_eq!(fetched.tenant.as_deref(), Some("header:a"));
    let fetched = resp
        .get_response(&id_nobody)
        .await
        .expect("get the untenanted record")
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
    assert!(resp
        .get_response(&id_a)
        .await
        .expect("get a's record")
        .is_none());
    assert!(resp
        .get_response(&id_b)
        .await
        .expect("get b's record")
        .is_some());
    let left = resp
        .list_identifier_responses(&identifier, Some("header:b"), None)
        .await
        .expect("list as b");
    assert_eq!(ids(left), vec![id_b.0.clone()]);

    // A conversation keeps its tenant and its creation time through a read
    // and a metadata update, and its delete answers that it existed.
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
    assert_eq!(updated.created_at, fetched.created_at);
    assert!(conversations
        .delete_conversation(&created.id)
        .await
        .expect("delete conversation"));
}

/// The conditional store of a background response's transition: a missing
/// record is left missing, an open one is replaced, a closed one is refused
/// and handed back as it is.
async fn a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one(
    bundle: StorageBundle,
) {
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

/// A tag unique to this run, for the ids on a live store.
fn nonce() -> String {
    ulid::Ulid::new().to_string()
}

async fn bundle(config: StorageFactoryConfig<'_>) -> StorageBundle {
    create_storage(config)
        .await
        .unwrap_or_else(|e| panic!("initialize the storage: {e}"))
}

/// The in-process memory store, under its default bounds.
async fn memory_store() -> StorageBundle {
    let memory = MemoryConfig::default();
    bundle(StorageFactoryConfig {
        backend: &HistoryBackend::Memory,
        oracle: None,
        postgres: None,
        redis: None,
        memory: Some(&memory),
        hook: None,
    })
    .await
}

/// The live Redis `DATA_CONNECTOR_TEST_REDIS_URL` names, with a retention of
/// one day; `None` when the url is not set.
async fn redis_store() -> Option<StorageBundle> {
    let url = std::env::var("DATA_CONNECTOR_TEST_REDIS_URL").ok()?;
    let redis = RedisConfig {
        url,
        pool_max: 4,
        op_timeout_secs: RedisConfig::default_op_timeout_secs(),
        schema: None,
        retention_days: Some(1),
    };
    Some(
        bundle(StorageFactoryConfig {
            backend: &HistoryBackend::Redis,
            oracle: None,
            postgres: None,
            redis: Some(&redis),
            memory: None,
            hook: None,
        })
        .await,
    )
}

/// The live Postgres `DATA_CONNECTOR_TEST_POSTGRES_URL` names, migrated to
/// the current schema on start; `None` when the url is not set.
async fn postgres_store() -> Option<StorageBundle> {
    let db_url = std::env::var("DATA_CONNECTOR_TEST_POSTGRES_URL").ok()?;
    let postgres = PostgresConfig {
        db_url,
        pool_max: 4,
        op_timeout_secs: PostgresConfig::default_op_timeout_secs(),
        schema: Some(SchemaConfig {
            auto_migrate: true,
            ..Default::default()
        }),
    };
    Some(
        bundle(StorageFactoryConfig {
            backend: &HistoryBackend::Postgres,
            oracle: None,
            postgres: Some(&postgres),
            redis: None,
            memory: None,
            hook: None,
        })
        .await,
    )
}

/// The live Oracle `DATA_CONNECTOR_TEST_ORACLE_URL` names
/// (`oracle://user:password@host:port/service`), migrated to the current
/// schema on start; `None` when the url is not set.
async fn oracle_store() -> Option<StorageBundle> {
    let url = std::env::var("DATA_CONNECTOR_TEST_ORACLE_URL").ok()?;
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
    Some(
        bundle(StorageFactoryConfig {
            backend: &HistoryBackend::Oracle,
            oracle: Some(&oracle),
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        })
        .await,
    )
}

#[tokio::test]
async fn the_memory_store_keeps_the_item_contract() {
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(memory_store().await, "memory")
        .await;
}

#[tokio::test]
async fn the_memory_store_keeps_the_link_contract() {
    a_second_link_of_an_item_moves_it_and_the_item_lists_once(memory_store().await).await;
}

#[tokio::test]
async fn the_memory_store_keeps_the_tenant_contract() {
    responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped(
        memory_store().await,
    )
    .await;
}

#[tokio::test]
async fn the_memory_store_keeps_the_conditional_store_contract() {
    a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one(
        memory_store().await,
    )
    .await;
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn the_redis_store_keeps_the_item_contract() {
    let Some(bundle) = redis_store().await else {
        return;
    };
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn the_redis_store_keeps_the_link_contract() {
    let Some(bundle) = redis_store().await else {
        return;
    };
    a_second_link_of_an_item_moves_it_and_the_item_lists_once(bundle).await;
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn the_redis_store_keeps_the_tenant_contract() {
    let Some(bundle) = redis_store().await else {
        return;
    };
    responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped(bundle)
        .await;
}

#[tokio::test]
#[ignore = "requires a live Redis server; set DATA_CONNECTOR_TEST_REDIS_URL and run with -- --ignored"]
async fn the_redis_store_keeps_the_conditional_store_contract() {
    let Some(bundle) = redis_store().await else {
        return;
    };
    a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one(bundle).await;
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn the_postgres_store_keeps_the_item_contract() {
    let Some(bundle) = postgres_store().await else {
        return;
    };
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn the_postgres_store_keeps_the_link_contract() {
    let Some(bundle) = postgres_store().await else {
        return;
    };
    a_second_link_of_an_item_moves_it_and_the_item_lists_once(bundle).await;
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn the_postgres_store_keeps_the_tenant_contract() {
    let Some(bundle) = postgres_store().await else {
        return;
    };
    responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped(bundle)
        .await;
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn the_postgres_store_keeps_the_conditional_store_contract() {
    let Some(bundle) = postgres_store().await else {
        return;
    };
    a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one(bundle).await;
}

#[tokio::test]
#[ignore = "requires a live Oracle database; set DATA_CONNECTOR_TEST_ORACLE_URL and run with -- --ignored"]
async fn the_oracle_store_keeps_the_item_contract() {
    let Some(bundle) = oracle_store().await else {
        return;
    };
    an_echoed_item_is_the_stored_item_and_its_turn_links_whole(bundle, &nonce()).await;
}

#[tokio::test]
#[ignore = "requires a live Oracle database; set DATA_CONNECTOR_TEST_ORACLE_URL and run with -- --ignored"]
async fn the_oracle_store_keeps_the_link_contract() {
    let Some(bundle) = oracle_store().await else {
        return;
    };
    a_second_link_of_an_item_moves_it_and_the_item_lists_once(bundle).await;
}

#[tokio::test]
#[ignore = "requires a live Oracle database; set DATA_CONNECTOR_TEST_ORACLE_URL and run with -- --ignored"]
async fn the_oracle_store_keeps_the_tenant_contract() {
    let Some(bundle) = oracle_store().await else {
        return;
    };
    responses_and_conversations_keep_their_tenant_and_the_identifier_queries_are_scoped(bundle)
        .await;
}

#[tokio::test]
#[ignore = "requires a live Oracle database; set DATA_CONNECTOR_TEST_ORACLE_URL and run with -- --ignored"]
async fn the_oracle_store_keeps_the_conditional_store_contract() {
    let Some(bundle) = oracle_store().await else {
        return;
    };
    a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one(bundle).await;
}
