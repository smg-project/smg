use std::{sync::mpsc, time::Instant};

use super::*;

/// A pooled connect a bound gave up on runs on a thread of its own, not
/// in the runtime's blocking pool: a runtime that shuts down after the
/// verdict does not wait for it, so a drain ends in the process's exit
/// while the connect's thread still sleeps.
#[test]
fn a_pooled_connect_the_bound_gave_up_on_does_not_hold_the_runtime_s_shutdown() {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime of the test's own");
        let limiter = Arc::new(Semaphore::new(1));
        let verdict = runtime.block_on(async move {
            let permit = limiter.acquire_owned().await.expect("a permit");
            tokio::time::timeout(
                Duration::from_secs(1),
                connect_on_own_thread(permit, || {
                    std::thread::sleep(Duration::from_secs(30));
                    7u8
                }),
            )
            .await
        });
        assert!(verdict.is_err(), "the bound gives the verdict: {verdict:?}");
        let dropping = Instant::now();
        drop(runtime);
        let _ = done_tx.send(dropping.elapsed());
    });
    let took = done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the runtime shut down while the connect's thread still ran");
    assert!(took < Duration::from_secs(5), "the drop took {took:?}");
}

/// The connect permit rides on the connect's thread, not on the waiting
/// future: once a bound has dropped the wait, the limiter stays one permit
/// short until the stalled connect's thread returns, so the next connect
/// waits for it; then the permit comes back with the thread. (A permit
/// held on the async side would come back with the dropped future and
/// let the next connect start while the stalled one still ran.)
#[tokio::test]
async fn a_stalled_connect_keeps_its_permit_after_the_bound_dropped_the_wait() {
    let limiter = Arc::new(Semaphore::new(1));
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let permit = limiter.clone().acquire_owned().await.expect("a permit");
    let gave_up = tokio::time::timeout(
        Duration::from_millis(100),
        connect_on_own_thread(permit, move || {
            let _ = release_rx.recv();
            7u8
        }),
    )
    .await;
    assert!(
        gave_up.is_err(),
        "the bound gives up on the stalled connect: {gave_up:?}"
    );
    // The waiting future is gone; the thread runs on and holds the permit.
    assert!(
        limiter.try_acquire().is_err(),
        "the stalled connect's thread holds the only permit after the bound dropped the wait"
    );
    release_tx
        .send(())
        .expect("the connect's thread waits for the release");
    // The permit comes back when the thread returns; its answer goes to a
    // receiver that is gone.
    let next = tokio::time::timeout(Duration::from_secs(10), limiter.acquire())
        .await
        .expect("the permit came back with the thread");
    assert!(next.is_ok(), "the limiter is open again: {next:?}");
}

/// The identifier filter binds the identifier first and the tenant scope
/// second, and leaves the scope out when the table keeps no tenant.
#[test]
fn the_identifier_filter_scopes_by_tenant_only_when_the_table_keeps_one() {
    let mut s = SchemaConfig::default().responses;
    let identifier = "user-1".to_string();
    let tenant = Some("t".to_string());
    let scope = || IdentifierScope {
        identifier: &identifier,
        tenant: &tenant,
    };
    let (filter, binds) = identifier_filter(&s, scope());
    assert_eq!(
        filter,
        "safety_identifier = :1 AND (tenant IS NULL OR tenant = :2)"
    );
    assert_eq!(binds.len(), 2);
    assert!(
        std::ptr::addr_eq(binds[0], &identifier),
        "the identifier is :1"
    );
    assert!(std::ptr::addr_eq(binds[1], &tenant), "the tenant is :2");

    s.skip_columns.insert("tenant".to_string());
    let (filter, binds) = identifier_filter(&s, scope());
    assert_eq!(filter, "safety_identifier = :1");
    assert_eq!(binds.len(), 1);
    assert!(std::ptr::addr_eq(binds[0], &identifier));
}

/// The responses row binds the id whatever the table skips, the other
/// logical fields unless skipped, then the extra columns, one bind per
/// column.
#[test]
fn the_responses_row_keeps_the_id_skips_the_skipped_and_appends_the_extras() {
    let mut s = SchemaConfig::default().responses;
    s.skip_columns.insert("id".to_string());
    s.skip_columns.insert("model".to_string());
    let row = ResponseRow::new(StoredResponse {
        id: ResponseId("resp_1".to_string()),
        previous_response_id: Some(ResponseId("resp_0".to_string())),
        input: serde_json::json!([]),
        created_at: Utc::now(),
        safety_identifier: None,
        model: Some("m".to_string()),
        conversation_id: None,
        raw_response: serde_json::json!({"id": "resp_1"}),
        tenant: None,
    })
    .expect("the JSON fields serialize");
    let extras = vec![("region", Some("eu".to_string()))];
    let (columns, binds) = row.columns_and_binds(&s, &extras);
    assert_eq!(
        columns,
        [
            "id",
            "previous_response_id",
            "input",
            "created_at",
            "safety_identifier",
            "conversation_id",
            "raw_response",
            "tenant",
            "region",
        ]
    );
    assert_eq!(binds.len(), columns.len());
}

#[test]
fn output_item_lookup_binds_the_id_into_the_json_path_and_takes_one_row() {
    let schema = SchemaConfig::default();
    let sql = find_by_output_item_sql(&schema, &build_response_select_base(&schema))
        .expect("raw_response is stored by default");
    assert!(
        sql.contains(
            "JSON_EXISTS(raw_response, '$.output[*]?(@.id == $iid)' PASSING :1 AS \"iid\")"
        ),
        "{sql}"
    );
    assert!(
        sql.ends_with(" ORDER BY created_at DESC) WHERE ROWNUM <= 1"),
        "{sql}"
    );
}

#[test]
fn output_item_lookup_needs_the_raw_response_column() {
    let mut schema = SchemaConfig::default();
    schema
        .responses
        .skip_columns
        .insert("raw_response".to_string());
    assert!(find_by_output_item_sql(&schema, "SELECT 1 FROM dual").is_none());
}

#[test]
fn link_sql_merges_on_the_conversation_and_item_key() {
    let (sql, binds) = link_sql(
        "links",
        &["conversation_id", "item_id", "added_at", "tenant"],
    );
    assert_eq!(
        sql,
        "MERGE INTO links t USING (SELECT :v1 AS conversation_id, :v2 AS item_id, \
             :v3 AS added_at FROM dual) s \
             ON (t.conversation_id = s.conversation_id AND t.item_id = s.item_id) \
             WHEN MATCHED THEN UPDATE SET t.added_at = s.added_at, t.tenant = :v4 \
             WHEN NOT MATCHED THEN INSERT (conversation_id, item_id, added_at, tenant) \
             VALUES (s.conversation_id, s.item_id, s.added_at, :v4)"
    );
    assert_eq!(binds, ["v1", "v2", "v3", "v4"]);
    // An extra column may be a CLOB: its value is never bound inside the
    // source (ORA-01461 above 4,000 bytes there), as in `store_response_sql`.
    let source = &sql[sql.find("USING (").unwrap()..sql.find(") s ON").unwrap()];
    assert!(!source.contains(":v4"), "{source}");
    // The key alone leaves nothing to move.
    let (sql, _) = link_sql("l", &["conversation_id", "item_id"]);
    assert!(!sql.contains("WHEN MATCHED"), "{sql}");
    assert!(
        sql.ends_with("VALUES (s.conversation_id, s.item_id)"),
        "{sql}"
    );
}

#[test]
fn link_items_sql_merges_the_batch_in_one_statement() {
    let sql = link_items_sql(
        "links",
        &["conversation_id", "item_id", "added_at", "tenant"],
        2,
    );
    assert_eq!(
        sql,
        "MERGE INTO links t USING (\
             SELECT :cid AS conversation_id, :iid0 AS item_id, :ts0 AS added_at FROM dual \
             UNION ALL \
             SELECT :cid AS conversation_id, :iid1 AS item_id, :ts1 AS added_at FROM dual) s \
             ON (t.conversation_id = s.conversation_id AND t.item_id = s.item_id) \
             WHEN MATCHED THEN UPDATE SET t.added_at = s.added_at, t.tenant = :ex0 \
             WHEN NOT MATCHED THEN INSERT (conversation_id, item_id, added_at, tenant) \
             VALUES (s.conversation_id, s.item_id, s.added_at, :ex0)"
    );
    let source = &sql[sql.find("USING (").unwrap()..sql.find(") s ON").unwrap()];
    assert!(!source.contains(":ex0"), "{source}");
}

#[test]
fn a_batch_keeps_the_last_stamp_of_a_repeated_item() {
    let t = |secs: i64| DateTime::<Utc>::from_timestamp(secs, 0).unwrap();
    let rows = last_stamp_per_item(vec![
        ("a".to_string(), t(1)),
        ("b".to_string(), t(2)),
        ("a".to_string(), t(3)),
        ("c".to_string(), t(4)),
        ("b".to_string(), t(5)),
    ]);
    assert_eq!(
        rows,
        vec![
            ("a".to_string(), t(3)),
            ("b".to_string(), t(5)),
            ("c".to_string(), t(4)),
        ]
    );
}

#[test]
fn update_if_open_sql_updates_the_row_under_its_id_only_while_its_status_is_open() {
    let open = oracle_open_predicate("raw_response");
    let (sql, binds) =
        update_if_open_sql("responses", &["id", "input", "raw_response"], Some(&open));
    assert_eq!(binds, ["v1", "v2", "v3"]);
    assert_eq!(
        sql,
        "UPDATE responses SET input = :v2, raw_response = :v3 WHERE id = :v1 AND \
             COALESCE(JSON_VALUE(raw_response, '$.status'), ' ') NOT IN ('completed', \
             'incomplete', 'failed', 'cancelled')"
    );
    let (plain, _) = update_if_open_sql("r", &["response_key", "input"], None);
    assert_eq!(plain, "UPDATE r SET input = :v2 WHERE response_key = :v1");
}

#[test]
fn create_item_sql_inserts_under_a_free_id_and_leaves_a_stored_row_as_it_is() {
    let (sql, binds) = create_item_sql("conversation_items", &["id", "item_type", "content"]);
    assert_eq!(
        sql,
        "MERGE INTO conversation_items t USING (SELECT :v1 AS id FROM dual) s ON (t.id = s.id) \
             WHEN NOT MATCHED THEN INSERT (id, item_type, content) VALUES (:v1, :v2, :v3)"
    );
    assert_eq!(binds, ["v1", "v2", "v3"]);
    // The content may be a CLOB: bound in INSERT VALUES, never inside the
    // source (ORA-01461 above 4,000 bytes there), as `store_response_sql` binds.
    let source = &sql[sql.find("USING (").unwrap()..sql.find(") s ON").unwrap()];
    assert!(!source.contains(":v3"), "{source}");
}

#[test]
fn store_response_sql_merges_on_the_id_and_binds_the_values_outside_the_source() {
    let (sql, binds) = store_response_sql("responses", &["id", "input", "raw_response"]);
    assert_eq!(
        sql,
        "MERGE INTO responses t USING (SELECT :v1 AS id FROM dual) s ON (t.id = s.id) \
             WHEN MATCHED THEN UPDATE SET t.input = :v2, t.raw_response = :v3 \
             WHEN NOT MATCHED THEN INSERT (id, input, raw_response) VALUES (:v1, :v2, :v3)"
    );
    assert_eq!(
        binds,
        ["v1", "v2", "v3"],
        "one bind per column, in the columns' order"
    );
    // The CLOB-sized values never sit in the USING source (ORA-01461 above
    // 4,000 bytes there); they are bound where an INSERT binds them.
    let source = &sql[sql.find("USING (").unwrap()..sql.find(") s ON").unwrap()];
    assert!(
        !source.contains(":v2") && !source.contains(":v3"),
        "{source}"
    );
    let (sql, binds) = store_response_sql("r", &["id"]);
    assert_eq!(
        sql,
        "MERGE INTO r t USING (SELECT :v1 AS id FROM dual) s ON (t.id = s.id) \
             WHEN NOT MATCHED THEN INSERT (id) VALUES (:v1)"
    );
    assert_eq!(binds, ["v1"]);
}

#[test]
fn a_schema_without_the_object_column_has_no_open_condition() {
    let mut s = TableConfig::with_table("responses");
    s.columns
        .insert("raw_response".to_string(), "payload".to_string());
    assert_eq!(
        oracle_open_predicate_for(&s).as_deref(),
        Some(oracle_open_predicate("payload").as_str()),
        "the mapped column is checked"
    );
    s.skip_columns.insert("raw_response".to_string());
    assert_eq!(
        oracle_open_predicate_for(&s),
        None,
        "no object column: the conditional store is a plain update"
    );
}
