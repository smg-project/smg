use serde_json::Value;

use crate::{
    core::ConversationMetadata,
    hooks::ExtraColumns,
    schema::{SchemaConfig, TableConfig},
};

/// An error with its sources after it: the database driver's errors print
/// only their kind (`db error`) and keep the server's message in the source,
/// and the pool wraps the driver's error once more. A source whose text the
/// error already printed is not repeated.
pub(crate) fn describe_error(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.ends_with(&cause_text) {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    text
}

/// Logical column names for the responses table, in canonical SELECT order.
///
/// Shared between Oracle and Postgres backends to build dynamic SELECT queries.
/// The order here doesn't affect correctness (both backends read by name, not
/// position), but having a single source prevents accidental divergence.
pub(super) const RESPONSE_COLUMNS: &[&str] = &[
    "id",
    "conversation_id",
    "previous_response_id",
    "input",
    "created_at",
    "safety_identifier",
    "model",
    "raw_response",
    "tenant",
];

/// Build the `SELECT col1, col2, ... FROM table` base query for responses.
///
/// Used by Oracle and Postgres to pre-build the SELECT prefix at construction
/// time, avoiding repeated string formatting on every query. Respects
/// `skip_columns` (omits them). Extra columns are write-side enrichment only
/// and are NOT included in SELECT.
pub(super) fn build_response_select_base(schema: &SchemaConfig) -> String {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());
    let cols: Vec<&str> = RESPONSE_COLUMNS
        .iter()
        .filter(|&&logical| !s.is_skipped(logical))
        .map(|&logical| s.col(logical))
        .collect();
    format!("SELECT {} FROM {table}", cols.join(", "))
}

/// Generate DDL column definitions for extra columns.
///
/// Returns e.g. `["TENANT_ID VARCHAR(128)", "EXPIRES_AT TIMESTAMP"]`.
/// Sorted for deterministic DDL generation.
pub(super) fn extra_column_defs(tc: &TableConfig) -> Vec<String> {
    let names = sorted_extra_column_names(tc);
    names
        .iter()
        .filter_map(|name| {
            tc.extra_columns
                .get(*name)
                .map(|def| format!("{name} {}", def.sql_type))
        })
        .collect()
}

/// Get extra column names sorted alphabetically for deterministic SQL generation.
pub(super) fn sorted_extra_column_names(tc: &TableConfig) -> Vec<&str> {
    let mut names: Vec<&str> = tc.extra_columns.keys().map(String::as_str).collect();
    names.sort_unstable();
    names
}

/// Convert a `serde_json::Value` to a SQL-bindable string representation.
///
/// Used by backends to bind extra column values (from hooks or defaults)
/// as text parameters in SQL statements.
pub(super) fn value_to_sql_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Resolve extra column values for a write operation.
///
/// For each extra column defined in the table config, resolves the value from:
/// 1. Hook-provided `ExtraColumns` (highest priority)
/// 2. `ColumnDef.default_value` from schema config
/// 3. `None` (if neither provides a non-null value)
///
/// Returns `(column_name, resolved_value)` pairs in sorted order.
pub(super) fn resolve_extra_column_values<'a>(
    tc: &'a TableConfig,
    hook_extra: &ExtraColumns,
) -> Vec<(&'a str, Option<String>)> {
    sorted_extra_column_names(tc)
        .into_iter()
        .map(|name| {
            let val = hook_extra
                .get(name)
                .filter(|v| !v.is_null())
                .map(value_to_sql_string)
                .or_else(|| {
                    tc.extra_columns
                        .get(name)
                        .and_then(|def| def.default_value.as_ref())
                        .filter(|v| !v.is_null())
                        .map(value_to_sql_string)
                });
            (name, val)
        })
        .collect()
}

/// Parse raw JSON string into `ConversationMetadata` (`JsonMap<String, Value>`).
///
/// Shared across Postgres, Redis, and Oracle conversation storage backends.
/// Returns `Ok(None)` for `None`, empty strings, and the literal `"null"`.
pub(super) fn parse_conversation_metadata(
    raw: Option<String>,
) -> Result<Option<ConversationMetadata>, String> {
    match raw {
        Some(s) if !s.is_empty() => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("null") {
                return Ok(None);
            }
            serde_json::from_str::<ConversationMetadata>(s)
                .map(Some)
                .map_err(|e| e.to_string())
        }
        _ => Ok(None),
    }
}

fn parse_json_or<F>(raw: Option<String>, default: F) -> Result<Value, String>
where
    F: FnOnce() -> Value,
{
    raw.filter(|s| !s.is_empty()).map_or_else(
        || Ok(default()),
        |s| serde_json::from_str(&s).map_err(|e| e.to_string()),
    )
}

pub(super) fn parse_raw_response(raw: Option<String>) -> Result<Value, String> {
    parse_json_or(raw, || Value::Null)
}

pub(super) fn parse_json_value(raw: Option<String>) -> Result<Value, String> {
    parse_json_or(raw, || Value::Array(Vec::new()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn describe_error_prints_the_sources_once() {
        #[derive(Debug)]
        struct Layer(&'static str, Option<Box<Layer>>, bool);
        impl std::fmt::Display for Layer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match (&self.1, self.2) {
                    (Some(inner), true) => write!(f, "{}: {inner}", self.0),
                    _ => f.write_str(self.0),
                }
            }
        }
        impl std::error::Error for Layer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.1
                    .as_deref()
                    .map(|inner| inner as &(dyn std::error::Error + 'static))
            }
        }
        let server = Layer("ERROR: relation \"responses\" already exists", None, false);
        let driver = Layer("db error", Some(Box::new(server)), false);
        assert_eq!(
            describe_error(&driver),
            "db error: ERROR: relation \"responses\" already exists"
        );
        // A wrapper that already printed its source does not repeat it.
        let server = Layer("timed out", None, false);
        let pool = Layer(
            "Error occurred while creating a new object",
            Some(Box::new(server)),
            true,
        );
        assert_eq!(
            describe_error(&pool),
            "Error occurred while creating a new object: timed out"
        );
    }

    #[test]
    fn parse_raw_response_handles_null() {
        assert_eq!(parse_raw_response(None).unwrap(), Value::Null);
    }

    #[test]
    fn parse_raw_response_round_trips() {
        let payload = json!({"id": "abc"}).to_string();
        let parsed = parse_raw_response(Some(payload)).unwrap();
        assert_eq!(parsed["id"], "abc");
    }

    #[test]
    fn parse_conversation_metadata_none_returns_ok_none() {
        assert!(parse_conversation_metadata(None).unwrap().is_none());
    }

    #[test]
    fn parse_conversation_metadata_empty_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some(String::new()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_null_string_returns_ok_none() {
        assert!(parse_conversation_metadata(Some("null".to_string()))
            .unwrap()
            .is_none());
        // Also test case-insensitive
        assert!(parse_conversation_metadata(Some("NULL".to_string()))
            .unwrap()
            .is_none());
        assert!(parse_conversation_metadata(Some("Null".to_string()))
            .unwrap()
            .is_none());
    }

    #[test]
    fn parse_conversation_metadata_valid_json_object() {
        let payload = json!({"key": "value", "count": 42}).to_string();
        let parsed = parse_conversation_metadata(Some(payload))
            .unwrap()
            .expect("should be Some");
        assert_eq!(parsed.get("key").expect("key should exist"), "value");
        assert_eq!(
            parsed
                .get("count")
                .expect("count should exist")
                .as_i64()
                .expect("should be i64"),
            42
        );
    }

    #[test]
    fn parse_conversation_metadata_invalid_json_returns_err() {
        let result = parse_conversation_metadata(Some("not json".to_string()));
        assert!(result.is_err());
    }

    // ── Extra column helpers ─────────────────────────────────────────────

    #[test]
    fn extra_column_defs_empty_by_default() {
        let tc = TableConfig::with_table("t");
        assert!(extra_column_defs(&tc).is_empty());
    }

    #[test]
    fn extra_column_defs_generates_sql() {
        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );
        tc.extra_columns.insert(
            "EXPIRES_AT".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TIMESTAMP".to_string(),
                default_value: None,
            },
        );
        let defs = extra_column_defs(&tc);
        assert_eq!(defs.len(), 2);
        // Sorted alphabetically
        assert_eq!(defs[0], "EXPIRES_AT TIMESTAMP");
        assert_eq!(defs[1], "TENANT_ID VARCHAR(128)");
    }

    #[test]
    fn sorted_extra_column_names_returns_sorted() {
        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "z_col".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        tc.extra_columns.insert(
            "a_col".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        let names = sorted_extra_column_names(&tc);
        assert_eq!(names, vec!["a_col", "z_col"]);
    }

    // ── resolve_extra_column_values ──────────────────────────────────────

    #[test]
    fn resolve_extra_values_prefers_hook_over_default() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("default_tenant")),
            },
        );

        let mut hook = ExtraColumns::new();
        hook.insert("TENANT_ID".to_string(), json!("hook_tenant"));

        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].0, "TENANT_ID");
        assert_eq!(resolved[0].1, Some("hook_tenant".to_string()));
    }

    #[test]
    fn resolve_extra_values_falls_back_to_default() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("default_tenant")),
            },
        );

        let hook = ExtraColumns::new(); // empty
        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, Some("default_tenant".to_string()));
    }

    #[test]
    fn resolve_extra_values_returns_none_when_no_value() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );

        let hook = ExtraColumns::new(); // empty
        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, None);
    }

    #[test]
    fn resolve_extra_values_skips_null_hook_values() {
        use crate::hooks::ExtraColumns;

        let mut tc = TableConfig::with_table("t");
        tc.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("fallback")),
            },
        );

        let mut hook = ExtraColumns::new();
        hook.insert("TENANT_ID".to_string(), Value::Null);

        let resolved = resolve_extra_column_values(&tc, &hook);
        assert_eq!(resolved[0].1, Some("fallback".to_string()));
    }

    // ── build_response_select_base with skip/extra ──────────────────────

    #[test]
    fn select_base_skips_columns() {
        let mut cfg = SchemaConfig::default();
        cfg.responses
            .skip_columns
            .insert("raw_response".to_string());
        cfg.responses
            .skip_columns
            .insert("safety_identifier".to_string());
        let sql = build_response_select_base(&cfg);
        assert!(!sql.contains("raw_response"));
        assert!(!sql.contains("safety_identifier"));
        assert!(sql.contains("id")); // not skipped
    }

    #[test]
    fn select_base_excludes_extra_columns() {
        let mut cfg = SchemaConfig::default();
        cfg.responses.extra_columns.insert(
            "tenant_id".to_string(),
            crate::schema::ColumnDef {
                sql_type: "TEXT".to_string(),
                default_value: None,
            },
        );
        let sql = build_response_select_base(&cfg);
        // Extra columns are write-side only — NOT included in SELECT
        assert!(!sql.contains("tenant_id"));
        assert!(sql.contains("id")); // core cols still there
    }

    // ── Schema config drives Oracle-compatible SQL ───

    /// SchemaConfig with custom names generates Oracle-compatible SQL.
    #[test]
    fn schema_config_produces_oracle_compatible_sql() {
        // Default table/column names with an Oracle owner prefix; Oracle uses
        // unquoted identifiers, which are case-insensitive.
        let mut cfg = SchemaConfig {
            owner: Some("ADMIN".to_string()),
            ..SchemaConfig::default()
        };

        // Uppercase for Oracle catalog compatibility (uppercases table names)
        cfg.uppercase_for_oracle();

        let sql = build_response_select_base(&cfg);

        // Should produce Oracle-style qualified table with all standard columns.
        // Column names are lowercase (Oracle treats unquoted identifiers as
        // case-insensitive, so `id` and `ID` resolve to the same column).
        assert!(
            sql.contains("ADMIN.\"RESPONSES\""),
            "missing qualified table: {sql}"
        );
        for col in RESPONSE_COLUMNS {
            assert!(sql.contains(col), "missing standard column '{col}': {sql}");
        }
    }

    /// Adding a TENANT_ID extra column works purely through config.
    #[test]
    fn schema_config_adds_tenant_column_without_forking() {
        let mut cfg = SchemaConfig {
            owner: Some("ADMIN".to_string()),
            ..SchemaConfig::default()
        };

        cfg.responses.extra_columns.insert(
            "TENANT_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR2(128)".to_string(),
                default_value: Some(json!("default_tenant")),
            },
        );

        cfg.uppercase_for_oracle();

        // DDL includes the extra column
        let defs = extra_column_defs(&cfg.responses);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0], "TENANT_ID VARCHAR2(128)");

        // Extra columns resolve with hook value > default > None
        let mut hook_extra = ExtraColumns::new();
        hook_extra.insert("TENANT_ID".to_string(), json!("acme-corp"));
        let resolved = resolve_extra_column_values(&cfg.responses, &hook_extra);
        assert_eq!(resolved[0].0, "TENANT_ID");
        assert_eq!(resolved[0].1, Some("acme-corp".to_string()));

        // Without hook value, falls back to default
        let empty_hook = ExtraColumns::new();
        let resolved = resolve_extra_column_values(&cfg.responses, &empty_hook);
        assert_eq!(resolved[0].1, Some("default_tenant".to_string()));
    }

    /// Proves that skip_columns lets you omit columns that a particular
    /// deployment doesn't use — another common fork reason.
    #[test]
    fn schema_config_skips_unused_columns_without_forking() {
        let mut cfg = SchemaConfig::default();
        // This deployment doesn't use safety_identifier or raw_response
        cfg.responses
            .skip_columns
            .insert("safety_identifier".to_string());
        cfg.responses
            .skip_columns
            .insert("raw_response".to_string());

        let sql = build_response_select_base(&cfg);
        assert!(
            !sql.contains("safety_identifier"),
            "should be skipped: {sql}"
        );
        assert!(!sql.contains("raw_response"), "should be skipped: {sql}");
        // Other columns still present
        assert!(sql.contains("id"), "id should remain: {sql}");
        assert!(sql.contains("model"), "model should remain: {sql}");
    }

    /// Proves that column name remapping works — another fork reason when
    /// an existing database uses different naming conventions.
    #[test]
    fn schema_config_remaps_column_names_without_forking() {
        let mut cfg = SchemaConfig::default();
        // This DB uses "resp_id" instead of "id" and "user_input" instead of "input"
        cfg.responses
            .columns
            .insert("id".to_string(), "resp_id".to_string());
        cfg.responses
            .columns
            .insert("input".to_string(), "user_input".to_string());

        let sql = build_response_select_base(&cfg);
        assert!(sql.contains("resp_id"), "should use remapped name: {sql}");
        assert!(
            sql.contains("user_input"),
            "should use remapped name: {sql}"
        );
    }

    // ── Combined feature tests ────────────────────────────────────────────

    #[test]
    fn skip_columns_and_extra_columns_together() {
        use crate::{hooks::ExtraColumns, schema::ColumnDef};

        let mut cfg = SchemaConfig::default();
        cfg.responses
            .skip_columns
            .insert("raw_response".to_string());
        cfg.responses
            .skip_columns
            .insert("safety_identifier".to_string());
        cfg.responses.extra_columns.insert(
            "TENANT_ID".to_string(),
            ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );
        cfg.responses.extra_columns.insert(
            "AUDIT_TS".to_string(),
            ColumnDef {
                sql_type: "TIMESTAMP".to_string(),
                default_value: None,
            },
        );

        let sql = build_response_select_base(&cfg);

        // Skipped columns must be absent
        assert!(
            !sql.contains("raw_response"),
            "raw_response should be skipped: {sql}"
        );
        assert!(
            !sql.contains("safety_identifier"),
            "safety_identifier should be skipped: {sql}"
        );

        // Extra columns are write-side only — NOT in SELECT
        assert!(
            !sql.contains("TENANT_ID"),
            "TENANT_ID should not be in SELECT: {sql}"
        );
        assert!(
            !sql.contains("AUDIT_TS"),
            "AUDIT_TS should not be in SELECT: {sql}"
        );

        // Core columns that are not skipped must still be present
        for col in &["id", "input", "model", "conversation_id"] {
            assert!(
                sql.contains(col),
                "core column '{col}' should remain: {sql}"
            );
        }

        // extra_column_defs returns both extra columns
        let defs = extra_column_defs(&cfg.responses);
        assert_eq!(defs.len(), 2);
        // Sorted alphabetically: AUDIT_TS before TENANT_ID
        assert_eq!(defs[0], "AUDIT_TS TIMESTAMP");
        assert_eq!(defs[1], "TENANT_ID VARCHAR(128)");

        // resolve_extra_column_values works with hook values for both
        let mut hook = ExtraColumns::new();
        hook.insert("TENANT_ID".to_string(), json!("acme"));
        hook.insert("AUDIT_TS".to_string(), json!("2025-01-01T00:00:00Z"));
        let resolved = resolve_extra_column_values(&cfg.responses, &hook);
        assert_eq!(resolved.len(), 2);
        // Sorted: AUDIT_TS first, TENANT_ID second
        assert_eq!(resolved[0].0, "AUDIT_TS");
        assert_eq!(resolved[0].1, Some("2025-01-01T00:00:00Z".to_string()));
        assert_eq!(resolved[1].0, "TENANT_ID");
        assert_eq!(resolved[1].1, Some("acme".to_string()));
    }

    #[test]
    fn column_remapping_with_skip_and_extra() {
        use crate::schema::ColumnDef;

        let mut cfg = SchemaConfig::default();
        // Rename table
        cfg.responses.table = "my_responses".to_string();
        // Remap columns
        cfg.responses
            .columns
            .insert("id".to_string(), "resp_id".to_string());
        cfg.responses
            .columns
            .insert("input".to_string(), "user_input".to_string());
        // Skip a column
        cfg.responses
            .skip_columns
            .insert("safety_identifier".to_string());
        // Add extra column
        cfg.responses.extra_columns.insert(
            "TENANT_ID".to_string(),
            ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: None,
            },
        );

        let sql = build_response_select_base(&cfg);

        // Table name should be the renamed one
        assert!(
            sql.contains("my_responses"),
            "should use renamed table: {sql}"
        );

        // Remapped column names
        assert!(
            sql.contains("resp_id"),
            "should use remapped 'resp_id': {sql}"
        );
        assert!(
            sql.contains("user_input"),
            "should use remapped 'user_input': {sql}"
        );
        // Original names should not appear (they were remapped)
        // Note: "id" is a substring of other column names, so we check more carefully
        // by looking at the column list portion. "input" could be substring of "user_input".
        // We check that the original logical names aren't used as standalone columns.
        let cols_part = sql
            .strip_prefix("SELECT ")
            .unwrap()
            .split(" FROM ")
            .next()
            .unwrap();
        let col_list: Vec<&str> = cols_part.split(", ").collect();
        assert!(
            !col_list.contains(&"id"),
            "should not contain bare 'id': {col_list:?}"
        );
        assert!(
            !col_list.contains(&"input"),
            "should not contain bare 'input': {col_list:?}"
        );

        // Skipped column absent
        assert!(
            !sql.contains("safety_identifier"),
            "safety_identifier should be skipped: {sql}"
        );

        // Extra column DDL still works
        let defs = extra_column_defs(&cfg.responses);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0], "TENANT_ID VARCHAR(128)");
    }

    #[test]
    fn extra_columns_in_insert_column_list() {
        use crate::{hooks::ExtraColumns, schema::ColumnDef};

        let mut cfg = SchemaConfig::default();
        // Remap id → resp_id
        cfg.responses
            .columns
            .insert("id".to_string(), "resp_id".to_string());
        // Skip raw_response
        cfg.responses
            .skip_columns
            .insert("raw_response".to_string());
        // Extra column with default
        cfg.responses.extra_columns.insert(
            "TENANT_ID".to_string(),
            ColumnDef {
                sql_type: "VARCHAR(128)".to_string(),
                default_value: Some(json!("unknown")),
            },
        );

        let s = &cfg.responses;

        // Build column names list the way INSERT code does in postgres.rs/oracle.rs
        let mut col_names: Vec<&str> = Vec::new();
        for &logical in RESPONSE_COLUMNS {
            if !s.is_skipped(logical) {
                col_names.push(s.col(logical));
            }
        }
        let hook_extra = ExtraColumns::new(); // empty hook
        let extra = resolve_extra_column_values(s, &hook_extra);
        for (name, _val) in &extra {
            col_names.push(name);
        }

        // Remapped: "resp_id" present, not "id"
        assert!(
            col_names.contains(&"resp_id"),
            "should contain remapped 'resp_id': {col_names:?}"
        );
        assert!(
            !col_names.contains(&"id"),
            "should not contain original 'id': {col_names:?}"
        );

        // Skipped: "raw_response" absent
        assert!(
            !col_names.contains(&"raw_response"),
            "should not contain skipped 'raw_response': {col_names:?}"
        );

        // Extra column appended
        assert!(
            col_names.contains(&"TENANT_ID"),
            "should contain extra 'TENANT_ID': {col_names:?}"
        );

        // Extra column value resolves to the default "unknown"
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].0, "TENANT_ID");
        assert_eq!(extra[0].1, Some("unknown".to_string()));
    }

    #[test]
    fn value_to_sql_string_conversions() {
        // String → string as-is
        assert_eq!(value_to_sql_string(&json!("hello")), "hello");

        // Number → "42"
        assert_eq!(value_to_sql_string(&json!(42)), "42");

        // Bool → "true"
        assert_eq!(value_to_sql_string(&json!(true)), "true");

        // Null → "null"
        assert_eq!(value_to_sql_string(&json!(null)), "null");

        // Object → JSON string
        let obj = json!({"key": "value"});
        let result = value_to_sql_string(&obj);
        // serde_json::to_string produces compact JSON
        assert_eq!(result, r#"{"key":"value"}"#);
    }
}
