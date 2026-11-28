use serde_json::json;

use super::*;
use crate::test_support::RefusedAddress;

#[test]
fn an_insert_numbers_one_placeholder_per_column() {
    assert_eq!(
        insert_sql("conversations", &["id", "created_at", "tenant"]),
        "INSERT INTO conversations (id, created_at, tenant) VALUES ($1, $2, $3)"
    );
}

/// The link table is read from the schema: its owner qualifies the
/// table, its column overrides rename the key columns.
#[test]
fn the_link_table_comes_from_the_schema_with_its_key_columns() {
    let mut schema = SchemaConfig::default();
    let (table, col_conv, col_item) = link_table(&schema);
    assert_eq!(
        (table.as_str(), col_conv, col_item),
        ("conversation_item_links", "conversation_id", "item_id")
    );

    schema.owner = Some("app".to_string());
    schema
        .conversation_item_links
        .columns
        .insert("item_id".to_string(), "item".to_string());
    let (table, _, col_item) = link_table(&schema);
    assert_eq!(
        (table.as_str(), col_item),
        ("app.\"conversation_item_links\"", "item")
    );
}

/// The replacement's literals are dollar-quoted, so the index expression,
/// the sweep and the lookup read the same under
/// `standard_conforming_strings = off` as under `on`: an ordinary literal
/// with a backslash in it is re-read under `off` (the pattern refused, the
/// replacement a space).
#[test]
fn the_nul_replacement_s_literals_read_the_same_under_either_string_setting() {
    let document = nul_safe_json_document("raw_response");
    assert_eq!(
        document,
        "replace(raw_response::text, $$\\u0000$$, $$\\u0020$$)"
    );
    assert!(!document.contains("'\\"), "{document}");
}

/// The sweep's statement reads `store` and `status` through the very
/// expressions its index is built on (a drift between the two would leave
/// the planner without the index, and the sweep scanning the table), and
/// the lookup by output item reads the document through the same NUL-safe
/// form.
#[test]
fn the_sweep_s_predicate_is_the_index_s_expressions() {
    let schema = SchemaConfig::default();
    let ddl = PostgresResponseStorage::unstored_index_ddl(&schema)
        .expect("the default schema keeps the object column");
    let sweep = PostgresResponseStorage::unstored_sweep_sql(&schema)
        .expect("the default schema keeps the object column");
    let col_raw = schema.responses.col("raw_response");
    let status = nul_safe_json_text(col_raw, "status");
    let store = nul_safe_json_text(col_raw, "store");
    assert!(ddl.contains(&format!("(({status}))")), "{ddl}");
    assert!(ddl.contains(&format!("WHERE {store} = 'false'")), "{ddl}");
    assert!(
        sweep.contains(&format!("WHERE {store} = 'false'")),
        "{sweep}"
    );
    assert!(sweep.contains(&format!("AND {status} IN (")), "{sweep}");
    assert!(!sweep.contains("::jsonb"), "{sweep}");
    let lookup = find_by_output_item_sql(&schema, "SELECT id FROM responses")
        .expect("the default schema keeps the object column");
    assert!(
        lookup.contains(&format!("{}::jsonb @>", nul_safe_json_document(col_raw))),
        "{lookup}"
    );
}

/// The earlier index is dropped concurrently (a plain drop holds an
/// ACCESS EXCLUSIVE lock on the live table to the end of its
/// transaction), under the schema's owner when it has one, as are the
/// indexes a start looks up before asking for them.
#[test]
fn the_earlier_index_is_dropped_concurrently() {
    let schema = SchemaConfig::default();
    assert_eq!(
        PostgresResponseStorage::legacy_unstored_index_drop_ddl(&schema),
        "DROP INDEX CONCURRENTLY IF EXISTS responses_unstored_idx;"
    );
    let owned = SchemaConfig {
        owner: Some("history".to_string()),
        ..Default::default()
    };
    assert_eq!(
        PostgresResponseStorage::legacy_unstored_index_drop_ddl(&owned),
        "DROP INDEX CONCURRENTLY IF EXISTS history.responses_unstored_idx;"
    );
    assert_eq!(
        PostgresConversationItemStorage::links_index_name(&owned),
        "history.conv_item_links_conv_idx"
    );
    assert_eq!(
        PostgresResponseStorage::safety_index_name(&schema),
        "responses_safety_idx"
    );
}

#[test]
fn parse_metadata_none_returns_ok_none() {
    assert!(PostgresConversationStorage::parse_metadata(None)
        .unwrap()
        .is_none());
}

#[test]
fn parse_metadata_json_null_returns_ok_none() {
    assert!(
        PostgresConversationStorage::parse_metadata(Some(Value::Null))
            .unwrap()
            .is_none()
    );
}

#[test]
fn parse_metadata_object_round_trips() {
    let obj = json!({"key": "value"}).as_object().unwrap().clone();
    let parsed =
        PostgresConversationStorage::parse_metadata(Some(Value::Object(obj.clone()))).unwrap();
    assert_eq!(parsed, Some(obj));
}

#[test]
fn parse_metadata_non_object_is_error() {
    assert!(PostgresConversationStorage::parse_metadata(Some(json!("not an object"))).is_err());
}

/// The start-up contact's failure carries its cause chain: the driver's
/// kind alone (`error connecting to server`) says nothing of why.
#[tokio::test]
async fn the_start_up_contact_keeps_the_cause_of_a_refused_connection() {
    // A port nothing listens on: bound and never listening, held for the
    // case, so the OS picked it, no service of the host is assumed absent,
    // and no other test's bind can take it meanwhile.
    let refused = RefusedAddress::reserve();
    let store = PostgresStore::new(PostgresConfig {
        db_url: format!("postgres://u:p@127.0.0.1:{}/db", refused.port()),
        pool_max: 1,
        op_timeout_secs: 2,
        schema: None,
    })
    .expect("a store");
    let err = store
        .contact(Duration::from_secs(5))
        .await
        .expect_err("nothing listens on the closed port");
    match err {
        StoreContactError::Failed(cause) => {
            assert!(cause.contains("refused"), "{cause}");
        }
        StoreContactError::NoAnswer => {
            panic!("a refused connection is a failure with its cause, not silence")
        }
    }
}

/// The bootstrap creates every table `IF NOT EXISTS`, in the schema's
/// names, and the index on its own so a legacy table can gain the column
/// first.
#[test]
fn bootstrap_ddl_creates_the_tables_if_they_do_not_exist() {
    let schema = SchemaConfig::default();
    for ddl in [
        PostgresConversationStorage::table_ddl(&schema),
        PostgresConversationItemStorage::table_ddl(&schema),
        PostgresResponseStorage::table_ddl(&schema),
    ] {
        assert!(ddl.starts_with("CREATE TABLE IF NOT EXISTS "), "{ddl}");
    }
    assert!(PostgresResponseStorage::table_ddl(&schema).contains("safety_identifier VARCHAR(128)"));
    assert_eq!(
        PostgresResponseStorage::index_ddl(&schema).as_deref(),
        Some("CREATE INDEX IF NOT EXISTS responses_safety_idx ON responses (safety_identifier);")
    );
    let mut skipped = SchemaConfig::default();
    skipped
        .responses
        .skip_columns
        .insert("safety_identifier".to_string());
    assert!(PostgresResponseStorage::index_ddl(&skipped).is_none());
    // The links index is a statement of its own, run after the migrations.
    assert_eq!(
        PostgresConversationItemStorage::index_ddl(&schema),
        "CREATE INDEX IF NOT EXISTS conv_item_links_conv_idx ON conversation_item_links (conversation_id, added_at);"
    );
    assert!(
        !PostgresConversationItemStorage::table_ddl(&schema).contains("CREATE INDEX"),
        "the tables' DDL carries no index"
    );
}

#[test]
fn link_sql_moves_an_item_the_conversation_already_has() {
    assert_eq!(
        link_sql(
            "conversation_item_links",
            &["conversation_id", "item_id", "added_at", "tenant"]
        ),
        "INSERT INTO conversation_item_links (conversation_id, item_id, added_at, tenant) \
             VALUES ($1, $2, $3, $4) ON CONFLICT (conversation_id, item_id) \
             DO UPDATE SET added_at = EXCLUDED.added_at, tenant = EXCLUDED.tenant"
    );
    // The key alone leaves nothing to move.
    assert_eq!(
        link_sql("l", &["conversation_id", "item_id"]),
        "INSERT INTO l (conversation_id, item_id) VALUES ($1, $2) \
             ON CONFLICT (conversation_id, item_id) DO NOTHING"
    );
}

#[test]
fn update_if_open_sql_updates_the_row_under_its_id_only_while_its_status_is_open() {
    let open = open_predicate("raw_response");
    assert_eq!(
        open,
        "COALESCE(replace(raw_response::text, $$\\u0000$$, $$\\u0020$$)::json->>'status', '') \
             NOT IN ('completed', 'incomplete', 'failed', 'cancelled')"
    );
    assert_eq!(
        update_if_open_sql("responses", &["id", "input", "raw_response"], Some(&open)),
        "UPDATE responses SET id = $1, input = $2, raw_response = $3 WHERE id = $1 AND \
             COALESCE(replace(raw_response::text, $$\\u0000$$, $$\\u0020$$)::json->>'status', '') \
             NOT IN ('completed', 'incomplete', 'failed', 'cancelled')"
    );
    assert_eq!(
        update_if_open_sql("r", &["response_key", "input"], None),
        "UPDATE r SET response_key = $1, input = $2 WHERE response_key = $1"
    );
}

#[test]
fn insert_if_absent_sql_leaves_a_stored_row_as_it_is() {
    assert_eq!(
        insert_if_absent_sql("conversation_items", &["id", "item_type", "content"]),
        "INSERT INTO conversation_items (id, item_type, content) VALUES ($1, $2, $3) \
             ON CONFLICT DO NOTHING"
    );
}

#[test]
fn upsert_sql_replaces_the_row_under_its_id_in_one_statement() {
    let sql = upsert_sql("responses", &["id", "input", "raw_response"]);
    assert_eq!(
        sql,
        "INSERT INTO responses (id, input, raw_response) VALUES ($1, $2, $3) \
             ON CONFLICT (id) DO UPDATE SET input = EXCLUDED.input, raw_response = EXCLUDED.raw_response"
    );
    // The mapped id is the first column; with no other column there is
    // nothing to replace.
    assert_eq!(
        upsert_sql("r", &["response_key"]),
        "INSERT INTO r (response_key) VALUES ($1) ON CONFLICT (response_key) DO NOTHING"
    );
}

#[test]
fn locked_merge_sql_updates_the_rows_under_the_id_and_inserts_when_none_moved() {
    let sql = locked_merge_sql("responses", &["id", "input", "raw_response"]);
    assert_eq!(
        sql,
        "WITH upd AS (UPDATE responses SET id = $1, input = $2, raw_response = $3 \
             WHERE id = $1 RETURNING 1) \
             INSERT INTO responses (id, input, raw_response) SELECT $1, $2, $3 \
             WHERE NOT EXISTS (SELECT 1 FROM upd)"
    );
    // One statement, and nothing in it needs a unique index on the id
    // column (a configured table that pre-exists may have none).
    assert!(!sql.contains("ON CONFLICT"));
    // The SET list is never empty: the id is assigned to itself.
    assert_eq!(
        locked_merge_sql("r", &["response_key"]),
        "WITH upd AS (UPDATE r SET response_key = $1 WHERE response_key = $1 RETURNING 1) \
             INSERT INTO r (response_key) SELECT $1 WHERE NOT EXISTS (SELECT 1 FROM upd)"
    );
    assert_eq!(lock_sql(-42), "SELECT pg_advisory_xact_lock(-42)");
}

#[test]
fn the_tenant_scope_condition_serves_untenanted_rows_and_the_callers_own() {
    let s = TableConfig::with_table("responses");
    assert_eq!(
        tenant_scope_sql(&s, 2).as_deref(),
        Some(" AND (tenant IS NULL OR tenant = $2)")
    );
    let mut skipped = TableConfig::with_table("responses");
    skipped.skip_columns.insert("tenant".to_string());
    assert_eq!(tenant_scope_sql(&skipped, 2), None);
}

#[test]
fn store_lock_key_is_stable_and_tells_ids_apart() {
    assert_eq!(store_lock_key("resp_1"), store_lock_key("resp_1"));
    assert_ne!(store_lock_key("resp_1"), store_lock_key("resp_2"));
    // Pinned FNV-1a 64 values: gateways of different builds share one
    // database, so the key of an id must never change.
    assert_eq!(store_lock_key(""), 0xcbf2_9ce4_8422_2325_u64 as i64);
    assert_eq!(store_lock_key("a"), 0xaf63_dc4c_8601_ec8c_u64 as i64);
}

/// The two answers that mean the statement the server was handed is
/// stale (its cached plan no longer fits, raised by the server's
/// cached-plan revalidation; it no longer exists, whatever raised it),
/// and no other, make a helper prepare the statement afresh and call
/// again: `0A000` from any other routine is the broad class it is (a
/// feature the server does not support) and is answered as it came.
#[test]
fn a_stale_plan_and_a_missing_statement_are_the_stale_states() {
    assert!(is_stale_statement_state(
        &SqlState::FEATURE_NOT_SUPPORTED,
        Some("RevalidateCachedQuery")
    ));
    assert!(is_stale_statement_state(
        &SqlState::INVALID_SQL_STATEMENT_NAME,
        None
    ));
    assert!(is_stale_statement_state(
        &SqlState::INVALID_SQL_STATEMENT_NAME,
        Some("FetchPreparedStatement")
    ));
    assert!(!is_stale_statement_state(
        &SqlState::FEATURE_NOT_SUPPORTED,
        None
    ));
    assert!(!is_stale_statement_state(
        &SqlState::FEATURE_NOT_SUPPORTED,
        Some("infer_arbiter_indexes")
    ));
    assert!(!is_stale_statement_state(&SqlState::UNIQUE_VIOLATION, None));
    assert!(!is_stale_statement_state(
        &SqlState::IN_FAILED_SQL_TRANSACTION,
        None
    ));
    assert!(!is_stale_statement_state(&SqlState::UNDEFINED_COLUMN, None));
    assert!(!is_stale_statement_state(&SqlState::QUERY_CANCELED, None));
}

#[test]
fn a_schema_without_the_object_column_has_no_open_condition() {
    let mut s = TableConfig::with_table("responses");
    s.columns
        .insert("raw_response".to_string(), "payload".to_string());
    assert_eq!(
        open_predicate_for(&s).as_deref(),
        Some(open_predicate("payload").as_str()),
        "the mapped column is checked"
    );
    s.skip_columns.insert("raw_response".to_string());
    assert_eq!(
        open_predicate_for(&s),
        None,
        "no object column: the conditional store is a plain update"
    );
}
