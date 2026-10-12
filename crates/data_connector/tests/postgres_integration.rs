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

use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use chrono::{TimeDelta, Utc};
use data_connector::{
    create_storage, ConversationItemId, HistoryBackend, ListParams, NewConversation,
    NewConversationItem, PostgresConfig, SchemaConfig, SortOrder, StorageBundle,
    StorageFactoryConfig, StoredResponse, TableConfig,
};
use serde_json::json;
use tokio::task::JoinSet;

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
async fn storing_under_an_existing_id_replaces_the_row() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let resp = bundle.response_storage;

    // A background response is stored as `queued` before its work starts and
    // once more with its terminal object, under the one id its client polls.
    let mut queued = StoredResponse::new(None);
    queued.input = json!(["write a story"]);
    queued.raw_response = json!({"id": queued.id.0, "status": "queued", "output": []});
    let id = resp
        .store_response(queued.clone())
        .await
        .expect("store the queued record");

    let mut done = queued;
    done.raw_response = json!({
        "id": id.0,
        "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "content": "once upon a time"}]
    });
    resp.store_response(done.clone())
        .await
        .expect("the second store must replace the row, not fail on the primary key");

    let fetched = resp
        .get_response(&id)
        .await
        .expect("get_response")
        .expect("the record stays under its id");
    assert_eq!(fetched.raw_response, done.raw_response);
    assert_eq!(fetched.input, done.input);
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

/// A responses table a deployment brings itself, with its own column names:
/// the store's `CREATE TABLE IF NOT EXISTS` leaves it as it is, so it has
/// only what its owner gave it. `unique_id` decides whether the mapped id
/// column carries a unique index. Returns the connection and the task that
/// drives it (the connection ends with the set).
async fn create_configured_responses_table(
    db_url: &str,
    table: &str,
    unique_id: bool,
) -> Result<(tokio_postgres::Client, JoinSet<()>), String> {
    let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls)
        .await
        .map_err(|e| format!("connect to the test database: {e}"))?;
    let mut driver = JoinSet::new();
    driver.spawn(async move {
        let _ = connection.await;
    });
    let id_def = if unique_id {
        "response_key VARCHAR(64) PRIMARY KEY"
    } else {
        "response_key VARCHAR(64)"
    };
    client
        .batch_execute(&format!(
            "DROP TABLE IF EXISTS {table}; \
             CREATE TABLE {table} ({id_def}, conversation_id VARCHAR(64), \
             previous_response_id VARCHAR(64), input JSON, created_at TIMESTAMPTZ, \
             safety_identifier VARCHAR(128), model VARCHAR(128), payload JSON)"
        ))
        .await
        .map_err(|e| format!("create the configured responses table: {e}"))?;
    Ok((client, driver))
}

/// Storage over the configured table, with `id` mapped to `response_key`
/// and `raw_response` to `payload`.
async fn configured_table_bundle(db_url: &str, table: &str) -> Result<StorageBundle, String> {
    let mut responses = TableConfig {
        table: table.to_string(),
        ..Default::default()
    };
    responses
        .columns
        .insert("id".to_string(), "response_key".to_string());
    responses
        .columns
        .insert("raw_response".to_string(), "payload".to_string());
    let postgres_cfg = PostgresConfig {
        db_url: db_url.to_string(),
        pool_max: 4,
        schema: Some(SchemaConfig {
            auto_migrate: true,
            responses,
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

/// A background response is stored as `queued` and once more with its
/// terminal object under the one id, on a table the store did not create:
/// the second store replaces the row whether or not the id column has a
/// unique index, and concurrent first stores of one id leave one row.
async fn a_configured_table_replaces_the_row_under_its_id(unique_id: bool) -> Result<(), String> {
    let Some(db_url) = test_db_url() else {
        return Ok(());
    };
    let table = if unique_id {
        "configured_responses_unique"
    } else {
        "configured_responses_plain"
    };
    let (admin, _driver) = create_configured_responses_table(&db_url, table, unique_id).await?;
    let resp = configured_table_bundle(&db_url, table)
        .await?
        .response_storage;

    let mut queued = StoredResponse::new(None);
    queued.input = json!(["write a story"]);
    queued.raw_response = json!({"id": queued.id.0, "status": "queued", "output": []});
    let id = resp
        .store_response(queued.clone())
        .await
        .map_err(|e| format!("store the queued record: {e}"))?;

    let mut done = queued;
    done.raw_response = json!({
        "id": id.0,
        "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "content": "once upon a time"}]
    });
    resp.store_response(done.clone()).await.map_err(|e| {
        format!(
            "the second store must replace the row, with or without a unique index on the id: {e}"
        )
    })?;

    let fetched = resp
        .get_response(&id)
        .await
        .map_err(|e| format!("get_response: {e}"))?
        .ok_or_else(|| "the record stays under its id".to_string())?;
    assert_eq!(fetched.raw_response, done.raw_response);
    assert_eq!(fetched.input, done.input);
    let count_sql = format!("SELECT count(*) FROM {table} WHERE response_key = $1");
    let rows: i64 = admin
        .query_one(&count_sql, &[&id.0])
        .await
        .map_err(|e| format!("count the rows under the id: {e}"))?
        .get(0);
    assert_eq!(rows, 1, "one row under the id after two stores");

    // Eight first stores of one fresh id at once.
    let fresh = StoredResponse::new(None);
    let mut writers = JoinSet::new();
    for writer in 0..8 {
        let resp = resp.clone();
        let mut record = fresh.clone();
        record.raw_response = json!({"id": fresh.id.0, "status": "queued", "writer": writer});
        writers.spawn(async move { resp.store_response(record).await });
    }
    while let Some(stored) = writers.join_next().await {
        stored
            .map_err(|e| format!("the store task: {e}"))?
            .map_err(|e| format!("a concurrent first store: {e}"))?;
    }
    let rows: i64 = admin
        .query_one(&count_sql, &[&fresh.id.0])
        .await
        .map_err(|e| format!("count the rows under the fresh id: {e}"))?
        .get(0);
    assert_eq!(rows, 1, "one row under an id eight writers stored at once");
    Ok(())
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_configured_table_without_a_unique_id_replaces_the_row_under_its_id() {
    a_configured_table_replaces_the_row_under_its_id(false)
        .await
        .expect("a configured table without a unique index on the id");
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_configured_table_with_a_unique_id_replaces_the_row_under_its_id() {
    a_configured_table_replaces_the_row_under_its_id(true)
        .await
        .expect("a configured table with a unique index on the id");
}

/// What a relay in front of the database counted: the server's ReadyForQuery
/// messages (one per round trip the client waited for) and the client's
/// statement Close messages (a statement prepared for one use is closed
/// afterwards; a cached one is not).
struct RoundTripMeter {
    ready_for_query: Arc<AtomicUsize>,
    closes: Arc<AtomicUsize>,
}

/// Copies one direction of a relayed connection and counts the messages of
/// kind `wanted`, before the bytes go on, so a count is in place when the
/// receiver acts on them. The client's first message (the startup) has no
/// type byte; every later message of either side is a type byte and a
/// length. (The store connects without TLS, so no SSL negotiation precedes
/// the startup message.)
fn pump(
    mut from: TcpStream,
    mut to: TcpStream,
    startup_first: bool,
    wanted: u8,
    counter: Arc<AtomicUsize>,
) {
    let mut chunk = [0u8; 65536];
    let mut buf: Vec<u8> = Vec::new();
    let mut startup_pending = startup_first;
    loop {
        let n = match from.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.extend_from_slice(&chunk[..n]);
        let mut at = 0;
        loop {
            if startup_pending {
                if buf.len() < at + 4 {
                    break;
                }
                let len =
                    u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
                if buf.len() < at + len {
                    break;
                }
                at += len;
                startup_pending = false;
                continue;
            }
            if buf.len() < at + 5 {
                break;
            }
            let len =
                u32::from_be_bytes([buf[at + 1], buf[at + 2], buf[at + 3], buf[at + 4]]) as usize;
            if buf.len() < at + 1 + len {
                break;
            }
            if buf[at] == wanted {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            at += 1 + len;
        }
        buf.drain(..at);
        if to.write_all(&chunk[..n]).is_err() {
            break;
        }
    }
    let _ = to.shutdown(Shutdown::Write);
}

/// Starts a relay on a free local port in front of the database `db_url`
/// names and returns the url that goes through it, with its meter. The url
/// shape is `scheme://[user:password@]host[:port][/...]`.
fn relayed_database(db_url: &str) -> Result<(String, RoundTripMeter), String> {
    let scheme_end = db_url
        .find("://")
        .ok_or_else(|| format!("no scheme in the database url {db_url}"))?
        + 3;
    let rest = &db_url[scheme_end..];
    let authority_end = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let (userinfo, host_port) = match authority.rfind('@') {
        Some(i) => (&authority[..=i], &authority[i + 1..]),
        None => ("", authority),
    };
    let upstream = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:5432")
    };
    let upstream_addr = upstream
        .to_socket_addrs()
        .map_err(|e| format!("resolve {upstream}: {e}"))?
        .next()
        .ok_or_else(|| format!("{upstream} resolves to nothing"))?;
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("bind the relay: {e}"))?;
    let local = listener
        .local_addr()
        .map_err(|e| format!("the relay's address: {e}"))?;
    let ready_for_query = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let (ready, closed) = (ready_for_query.clone(), closes.clone());
    thread::spawn(move || {
        for client in listener.incoming() {
            let Ok(client) = client else { break };
            let Ok(server) = TcpStream::connect(upstream_addr) else {
                break;
            };
            let (Ok(client_rx), Ok(server_rx)) = (client.try_clone(), server.try_clone()) else {
                break;
            };
            let closed = closed.clone();
            thread::spawn(move || pump(client_rx, server, true, b'C', closed));
            let ready = ready.clone();
            thread::spawn(move || pump(server_rx, client, false, b'Z', ready));
        }
    });
    let url = format!(
        "{}{userinfo}{local}{}",
        &db_url[..scheme_end],
        &rest[authority_end..]
    );
    Ok((
        url,
        RoundTripMeter {
            ready_for_query,
            closes,
        },
    ))
}

/// A store costs the round trips of its path and nothing more, measured on
/// the wire: `expected` per store, after the first store on the connection
/// prepared the statement, and no statement closed between stores.
async fn a_store_costs_the_round_trips_of_its_path(
    unique_id: bool,
    expected: usize,
) -> Result<(), String> {
    let Some(db_url) = test_db_url() else {
        return Ok(());
    };
    let table = if unique_id {
        "configured_responses_unique_relayed"
    } else {
        "configured_responses_plain_relayed"
    };
    let (_admin, _driver) = create_configured_responses_table(&db_url, table, unique_id).await?;
    let (relayed_url, meter) = relayed_database(&db_url)?;
    let resp = configured_table_bundle(&relayed_url, table)
        .await?
        .response_storage;

    let mut first = StoredResponse::new(None);
    first.raw_response = json!({"id": first.id.0, "status": "queued"});
    resp.store_response(first.clone())
        .await
        .map_err(|e| format!("the first store: {e}"))?;
    let ready_before = meter.ready_for_query.load(Ordering::SeqCst);
    let closes_before = meter.closes.load(Ordering::SeqCst);

    // A fresh id, then the first one replaced: both cost the same.
    let mut second = StoredResponse::new(None);
    second.raw_response = json!({"id": second.id.0, "status": "queued"});
    resp.store_response(second)
        .await
        .map_err(|e| format!("the store of a fresh id: {e}"))?;
    first.raw_response = json!({"id": first.id.0, "status": "completed"});
    resp.store_response(first)
        .await
        .map_err(|e| format!("the store that replaces the first record: {e}"))?;

    let round_trips = meter.ready_for_query.load(Ordering::SeqCst) - ready_before;
    let closes = meter.closes.load(Ordering::SeqCst) - closes_before;
    if round_trips != 2 * expected || closes != 0 {
        return Err(format!(
            "two stores took {round_trips} round trips and closed {closes} statements; \
             expected {} and none",
            2 * expected
        ));
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_store_is_one_round_trip_with_a_unique_index_on_the_id() {
    a_store_costs_the_round_trips_of_its_path(true, 1)
        .await
        .expect("one round trip per store on a table with a unique index on the id");
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_store_is_four_round_trips_without_a_unique_index_on_the_id() {
    a_store_costs_the_round_trips_of_its_path(false, 4)
        .await
        .expect("four round trips per store on a table without a unique index on the id");
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn conversation_items_linked_a_microsecond_apart_list_in_link_order() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let conversation = bundle
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");
    let items = bundle.conversation_item_storage;

    // Ids in the reverse of the link order, unique to this run (the database is shared): a
    // backend that tied on the stamp would list them by id, backwards. The stamps are one
    // microsecond apart, the resolution of the column.
    let nonce = &conversation.id.0;
    let ids: Vec<ConversationItemId> = ["z", "m", "a"]
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
    for (i, id) in ids.iter().enumerate() {
        items
            .link_item(
                &conversation.id,
                id,
                base + TimeDelta::microseconds(i as i64),
            )
            .await
            .expect("link item");
    }

    let list = |order: SortOrder, after: Option<&ConversationItemId>| {
        let items = items.clone();
        let conversation_id = conversation.id.clone();
        let params = ListParams {
            limit: 10,
            order,
            after: after.map(|id| id.0.clone()),
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

    assert_eq!(list(SortOrder::Asc, None).await, ids);
    let mut reversed = ids.clone();
    reversed.reverse();
    assert_eq!(list(SortOrder::Desc, None).await, reversed);
    assert_eq!(list(SortOrder::Asc, Some(&ids[0])).await, ids[1..]);
    assert_eq!(
        list(SortOrder::Desc, Some(&ids[2])).await,
        [ids[1].clone(), ids[0].clone()]
    );
}

#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_second_link_of_an_item_is_refused_by_the_key_and_the_item_lists_once() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let bundle = postgres_bundle(&db_url)
        .await
        .expect("failed to initialize Postgres storage");
    let conversation = bundle
        .conversation_storage
        .create_conversation(NewConversation {
            id: None,
            metadata: None,
        })
        .await
        .expect("create conversation");
    let items = bundle.conversation_item_storage;

    // One item, its id unique to this run (the database is shared) and within the 64
    // characters of the id column.
    let id = ConversationItemId::from(format!("msg_r_{}", conversation.id.0).as_str());
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
    let first = Utc::now();
    items
        .link_item(&conversation.id, &id, first)
        .await
        .expect("link item");
    assert!(items
        .is_item_linked(&conversation.id, &id)
        .await
        .expect("is linked"));

    // The links table keys one row per (conversation, item): a second link of the same item is
    // refused by the database. The gateway refuses a duplicate before it gets here, so a client
    // sees the same 400 on every backend rather than this error.
    let again = items
        .link_item(&conversation.id, &id, first + TimeDelta::seconds(1))
        .await;
    assert!(
        again.is_err(),
        "a second link of the same item is refused by the key"
    );

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
    assert_eq!(list(SortOrder::Asc).await, std::slice::from_ref(&id));

    // One link per item: the delete takes the item out of the conversation.
    items
        .delete_item(&conversation.id, &id)
        .await
        .expect("delete item");
    assert!(!items
        .is_item_linked(&conversation.id, &id)
        .await
        .expect("is linked"));
    assert!(list(SortOrder::Asc).await.is_empty());
}

/// A database of its own on the test server, for the tests of a first start:
/// created from the test url under a unique name, dropped at the end.
struct ScratchDatabase {
    admin_url: String,
    name: String,
    url: String,
}

impl ScratchDatabase {
    async fn create(db_url: &str, tag: &str) -> Result<Self, String> {
        let name = format!(
            "smg_scratch_{tag}_{}",
            ulid::Ulid::new().to_string().to_lowercase()
        );
        let url = database_url(db_url, &name)?;
        let scratch = Self {
            admin_url: db_url.to_string(),
            name,
            url,
        };
        scratch
            .admin(&format!("CREATE DATABASE \"{}\"", scratch.name))
            .await?;
        Ok(scratch)
    }

    /// Runs one statement on the test database (`CREATE` / `DROP DATABASE`
    /// run outside a transaction, on a connection of their own).
    async fn admin(&self, sql: &str) -> Result<(), String> {
        let (client, connection) = tokio_postgres::connect(&self.admin_url, tokio_postgres::NoTls)
            .await
            .map_err(|e| format!("connect to the test database: {e}"))?;
        let mut driver = JoinSet::new();
        driver.spawn(async move {
            let _ = connection.await;
        });
        let result = client
            .batch_execute(sql)
            .await
            .map_err(|e| format!("{sql}: {e}"));
        drop(client);
        driver.join_all().await;
        result
    }

    /// The schema versions the database records, in order.
    async fn versions(&self) -> Result<Vec<i32>, String> {
        let (client, connection) = tokio_postgres::connect(&self.url, tokio_postgres::NoTls)
            .await
            .map_err(|e| format!("connect to the scratch database: {e}"))?;
        let mut driver = JoinSet::new();
        driver.spawn(async move {
            let _ = connection.await;
        });
        let rows = client
            .query("SELECT version FROM _schema_versions ORDER BY version", &[])
            .await
            .map_err(|e| format!("read the schema versions: {e}"));
        drop(client);
        driver.join_all().await;
        Ok(rows?.iter().map(|row| row.get::<_, i32>(0)).collect())
    }

    /// The indexes of the database's current schema, by name.
    async fn indexes(&self) -> Result<Vec<String>, String> {
        let (client, connection) = tokio_postgres::connect(&self.url, tokio_postgres::NoTls)
            .await
            .map_err(|e| format!("connect to the scratch database: {e}"))?;
        let mut driver = JoinSet::new();
        driver.spawn(async move {
            let _ = connection.await;
        });
        let rows = client
            .query(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() ORDER BY indexname",
                &[],
            )
            .await
            .map_err(|e| format!("read the indexes: {e}"));
        drop(client);
        driver.join_all().await;
        Ok(rows?.iter().map(|row| row.get::<_, String>(0)).collect())
    }

    async fn drop(self) {
        let _ = self
            .admin(&format!(
                "DROP DATABASE IF EXISTS \"{}\" WITH (FORCE)",
                self.name
            ))
            .await;
    }
}

/// Six gateways start together on `scratch` (`auto_migrate` on, as a
/// deployment that migrates at start-up): the start-up failures, and the
/// versions the database records afterwards.
async fn six_gateways_start_together(
    scratch: &ScratchDatabase,
) -> (Vec<String>, Result<Vec<i32>, String>) {
    let mut starts = JoinSet::new();
    for _ in 0..6 {
        let url = scratch.url.clone();
        starts.spawn(async move {
            let postgres_cfg: PostgresConfig = serde_json::from_value(json!({
                "db_url": url,
                "pool_max": 2,
                "schema": {"auto_migrate": true},
            }))
            .map_err(|e| format!("a Postgres config: {e}"))?;
            create_storage(StorageFactoryConfig {
                backend: &HistoryBackend::Postgres,
                oracle: None,
                postgres: Some(&postgres_cfg),
                redis: None,
                hook: None,
            })
            .await
            .map(drop)
        });
    }
    let outcomes = starts.join_all().await;
    let versions = scratch.versions().await;
    (
        outcomes.into_iter().filter_map(Result::err).collect(),
        versions,
    )
}

/// `db_url` with its database name replaced by `name`.
fn database_url(db_url: &str, name: &str) -> Result<String, String> {
    let scheme_end = db_url
        .find("://")
        .ok_or_else(|| format!("no scheme in the database url {db_url}"))?
        + 3;
    let rest = &db_url[scheme_end..];
    let path_start = rest
        .find('/')
        .ok_or_else(|| format!("no database name in the url {db_url}"))?;
    let query_start = rest[path_start..]
        .find('?')
        .map_or(rest.len(), |q| path_start + q);
    Ok(format!(
        "{}{}/{name}{}",
        &db_url[..scheme_end],
        &rest[..path_start],
        &rest[query_start..]
    ))
}

/// A first start on an empty database comes up without `auto_migrate`: the
/// tables are created in their current shape, so nothing is pending, and the
/// version table records every migration as applied.
#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn a_fresh_database_bootstraps_at_the_latest_version_without_auto_migrate() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    let scratch = ScratchDatabase::create(&db_url, "fresh")
        .await
        .expect("create a scratch database");
    let postgres_cfg: PostgresConfig = serde_json::from_value(json!({
        "db_url": scratch.url,
        "pool_max": 4,
        "schema": {"auto_migrate": false},
    }))
    .expect("a Postgres config without auto_migrate");
    let outcome = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Postgres,
        oracle: None,
        postgres: Some(&postgres_cfg),
        redis: None,
        hook: None,
    })
    .await;
    let versions = scratch.versions().await;
    let indexes = scratch.indexes().await;
    let served = match &outcome {
        Ok(bundle) => Some(
            bundle
                .conversation_storage
                .create_conversation(NewConversation {
                    id: None,
                    metadata: None,
                })
                .await,
        ),
        Err(_) => None,
    };
    drop(outcome);
    scratch.drop().await;
    served
        .expect("a fresh database must come up without auto_migrate")
        .expect("the fresh store serves");
    assert_eq!(versions.expect("read the versions"), vec![1, 2, 3]);
    let indexes = indexes.expect("read the indexes");
    for index in ["conv_item_links_conv_idx", "responses_safety_idx"] {
        assert!(
            indexes.iter().any(|name| name == index),
            "{index} is created by the bootstrap: {indexes:?}"
        );
    }
}

/// Gateways starting together on an empty database all come up: the
/// bootstrap (tables, versions, migrations) runs once under the schema lock
/// and the others find it done.
#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn gateways_starting_together_on_an_empty_database_all_come_up() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    for round in 0..3 {
        let scratch = ScratchDatabase::create(&db_url, &format!("race{round}"))
            .await
            .expect("create a scratch database");
        let (failures, versions) = six_gateways_start_together(&scratch).await;
        scratch.drop().await;
        assert!(
            failures.is_empty(),
            "round {round}: {} of 6 starts failed: {failures:?}",
            failures.len()
        );
        assert_eq!(
            versions.expect("read the versions"),
            vec![1, 2, 3],
            "round {round}"
        );
    }
}

/// The bootstrap reads the schema version in a snapshot taken after its lock
/// wait whatever the server's default isolation: under a `repeatable read`
/// default a gateway queued behind another's bootstrap would otherwise read
/// the versions table as it was before that bootstrap and apply the
/// migrations again to tables that already have their final shape.
#[tokio::test]
#[ignore = "requires a live Postgres database; set DATA_CONNECTOR_TEST_POSTGRES_URL and run with -- --ignored"]
async fn gateways_starting_together_come_up_under_a_repeatable_read_default() {
    let Some(db_url) = test_db_url() else {
        return;
    };
    for round in 0..3 {
        let scratch = ScratchDatabase::create(&db_url, &format!("rr{round}"))
            .await
            .expect("create a scratch database");
        scratch
            .admin(&format!(
                "ALTER DATABASE \"{}\" SET default_transaction_isolation TO 'repeatable read'",
                scratch.name
            ))
            .await
            .expect("set the database's default isolation");
        let (failures, versions) = six_gateways_start_together(&scratch).await;
        scratch.drop().await;
        assert!(
            failures.is_empty(),
            "round {round}: {} of 6 starts failed under a repeatable read default: {failures:?}",
            failures.len()
        );
        assert_eq!(
            versions.expect("read the versions"),
            vec![1, 2, 3],
            "round {round}"
        );
    }
}
