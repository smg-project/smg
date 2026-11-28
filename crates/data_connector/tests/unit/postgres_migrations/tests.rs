use super::*;
use crate::schema::{ColumnDef, TableConfig};

#[test]
fn postgres_history_migrations_cover_only_core_history_schema() {
    let versions: Vec<u32> = POSTGRES_HISTORY_MIGRATIONS
        .iter()
        .map(|migration| migration.version)
        .collect();
    assert_eq!(versions, vec![1, 2, 3, 4]);
}

#[test]
fn postgres_migrations_are_strictly_increasing() {
    // Versions must be strictly increasing and unique; a numbering gap is allowed.
    for pair in POSTGRES_HISTORY_MIGRATIONS.windows(2) {
        assert!(
            pair[1].version > pair[0].version,
            "migration versions must strictly increase: {} then {}",
            pair[0].version,
            pair[1].version
        );
    }
}

#[test]
fn pg_v1_up_generates_add_column_if_not_exists() {
    let schema = SchemaConfig::default();
    let stmts = pg_v1_up(&schema);
    assert_eq!(stmts.len(), 1);
    assert!(stmts[0].contains("IF NOT EXISTS"), "got: {}", stmts[0]);
}

#[test]
fn pg_v1_up_skipped_returns_empty() {
    let schema = SchemaConfig {
        responses: TableConfig {
            skip_columns: ["safety_identifier".to_string()].into_iter().collect(),
            ..TableConfig::with_table("responses")
        },
        ..Default::default()
    };
    let stmts = pg_v1_up(&schema);
    assert!(stmts.is_empty());
}

#[test]
fn pg_v2_up_generates_drop_column_if_exists() {
    let schema = SchemaConfig::default();
    let stmts = pg_v2_up(&schema);
    assert_eq!(stmts.len(), 1);
    assert!(stmts[0].contains("IF EXISTS"), "got: {}", stmts[0]);
}

#[test]
fn pg_v2_up_skipped_when_column_maps_to_user_id() {
    let mut schema = SchemaConfig::default();
    schema
        .responses
        .columns
        .insert("safety_identifier".to_string(), "user_id".to_string());
    let stmts = pg_v2_up(&schema);
    assert!(stmts.is_empty(), "should skip drop when user_id is mapped");
}

#[test]
fn pg_v2_up_skipped_when_extra_column_is_user_id() {
    let mut schema = SchemaConfig::default();
    schema.responses.extra_columns.insert(
        "user_id".to_string(),
        ColumnDef {
            sql_type: "VARCHAR(128)".to_string(),
            default_value: None,
        },
    );
    let stmts = pg_v2_up(&schema);
    assert!(
        stmts.is_empty(),
        "should skip drop when user_id is an extra column"
    );
}

#[test]
fn pg_v4_up_adds_the_tenant_column_to_responses_and_conversations() {
    let stmts = pg_v4_up(&SchemaConfig::default());
    assert_eq!(
        stmts,
        vec![
            "ALTER TABLE responses ADD COLUMN IF NOT EXISTS tenant VARCHAR(255)",
            "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tenant VARCHAR(255)",
            "ALTER TABLE conversation_items ADD COLUMN IF NOT EXISTS tenant VARCHAR(255)",
        ]
    );
    // A mapped column name and a skipped table are respected.
    let mut schema = SchemaConfig::default();
    schema
        .responses
        .columns
        .insert("tenant".to_string(), "tenant_key".to_string());
    schema
        .conversations
        .skip_columns
        .insert("tenant".to_string());
    let stmts = pg_v4_up(&schema);
    assert_eq!(
        stmts,
        vec![
            "ALTER TABLE responses ADD COLUMN IF NOT EXISTS tenant_key VARCHAR(255)",
            "ALTER TABLE conversation_items ADD COLUMN IF NOT EXISTS tenant VARCHAR(255)",
        ]
    );
}

#[test]
fn pg_v3_up_generates_one_drop_statement() {
    let schema = SchemaConfig::default();
    let stmts = pg_v3_up(&schema);
    assert_eq!(stmts.len(), 1);
    let stmt = &stmts[0];
    assert!(stmt.contains("DROP COLUMN IF EXISTS output"));
    assert!(stmt.contains("DROP COLUMN IF EXISTS metadata"));
    assert!(stmt.contains("DROP COLUMN IF EXISTS instructions"));
    assert!(stmt.contains("DROP COLUMN IF EXISTS tool_calls"));
}

#[test]
fn pg_v3_up_skips_when_output_is_used_by_another_field() {
    let mut schema = SchemaConfig::default();
    // Another field maps to physical column "output"
    schema
        .responses
        .columns
        .insert("safety_identifier".to_string(), "output".to_string());
    let stmts = pg_v3_up(&schema);
    assert_eq!(stmts.len(), 1);
    assert!(
        !stmts[0].contains("EXISTS output"),
        "should skip output when another field maps to it: {stmts:?}"
    );
    assert!(stmts[0].contains("metadata"));
}

#[test]
fn pg_v3_up_skips_extra_column_named_metadata() {
    let mut schema = SchemaConfig::default();
    schema.responses.extra_columns.insert(
        "metadata".to_string(),
        ColumnDef {
            sql_type: "JSON".to_string(),
            default_value: None,
        },
    );
    let stmts = pg_v3_up(&schema);
    assert_eq!(stmts.len(), 1);
    assert!(
        !stmts[0].contains("metadata"),
        "should skip metadata when it's an extra column: {stmts:?}"
    );
    assert!(stmts[0].contains("output"));
}

#[test]
fn pg_v3_up_drops_mapped_physical_column_name() {
    let mut schema = SchemaConfig::default();
    schema
        .responses
        .columns
        .insert("output".to_string(), "resp_output".to_string());
    let stmts = pg_v3_up(&schema);
    assert_eq!(stmts.len(), 1);
    assert!(
        stmts[0].contains("resp_output"),
        "should drop mapped physical column: {stmts:?}"
    );
    assert!(
        !stmts[0].contains("EXISTS output"),
        "should not use logical name: {stmts:?}"
    );
}
