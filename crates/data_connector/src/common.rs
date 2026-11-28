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
/// The one list of the table's columns: the Oracle and Postgres backends build
/// their SELECT base from it (both read by name, so the order is a convention),
/// and the schema validation reads it as the core columns an extra column may
/// not shadow and a skipped column has to be one of.
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
#[path = "../tests/unit/common/tests.rs"]
mod tests;
