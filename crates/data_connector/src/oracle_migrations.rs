//! Oracle-specific schema migrations.
//!
//! Each migration is a function that generates Oracle DDL from [`SchemaConfig`],
//! so it respects custom table/column names. PL/SQL exception handling ensures
//! idempotency (safe to re-run if a previous attempt partially completed).

use crate::{schema::SchemaConfig, versioning::Migration};

const ORACLE_V1: Migration = Migration {
    version: 1,
    description: "Add safety_identifier column to responses",
    up: oracle_v1_up,
};
const ORACLE_V2: Migration = Migration {
    version: 2,
    description: "Remove legacy user_id column from responses",
    up: oracle_v2_up,
};
const ORACLE_V3: Migration = Migration {
    version: 3,
    description: "Drop redundant output, metadata, instructions, tool_calls columns from responses",
    up: oracle_v3_up,
};
const ORACLE_V4: Migration = Migration {
    version: 4,
    description: "Add tenant column to responses, conversations and conversation items",
    up: oracle_v4_up,
};

/// Core history-backend migrations required by the SQL response/conversation
/// storage path during normal gateway startup.
pub(crate) static ORACLE_HISTORY_MIGRATIONS: [Migration; 4] =
    [ORACLE_V1, ORACLE_V2, ORACLE_V3, ORACLE_V4];

fn oracle_v1_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    if s.is_skipped("safety_identifier") {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    let col = s.col("safety_identifier");
    // PL/SQL block: ORA-01430 = "column already exists" (idempotent)
    vec![format!(
        "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD ({col} VARCHAR2(128))'; \
         EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1430 THEN RAISE; END IF; END;"
    )]
}

/// The nullable `tenant` column of responses, conversations and conversation
/// items: the tenant a record is scoped to. Rows from before the column stay
/// NULL and are served to every caller, as they were.
fn oracle_v4_up(schema: &SchemaConfig) -> Vec<String> {
    [
        &schema.responses,
        &schema.conversations,
        &schema.conversation_items,
    ]
    .into_iter()
    .filter(|s| !s.is_skipped("tenant"))
    .map(|s| {
        let table = s.qualified_table(schema.owner.as_deref());
        let col = s.col("tenant");
        // PL/SQL block: ORA-01430 = "column already exists" (idempotent)
        format!(
            "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} ADD ({col} VARCHAR2(255))'; \
                 EXCEPTION WHEN OTHERS THEN IF SQLCODE != -1430 THEN RAISE; END IF; END;"
        )
    })
    .collect()
}

fn oracle_v2_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    // Don't drop USER_ID if a configured column maps to that name
    // or if it's defined as an extra column.
    if s.columns
        .values()
        .any(|v| v.eq_ignore_ascii_case("USER_ID"))
        || s.extra_columns
            .keys()
            .any(|k| k.eq_ignore_ascii_case("USER_ID"))
    {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    // PL/SQL block: ORA-00904 = "invalid identifier" (column doesn't exist)
    vec![format!(
        "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP (USER_ID)'; \
         EXCEPTION WHEN OTHERS THEN IF SQLCODE != -904 THEN RAISE; END IF; END;"
    )]
}

/// Drop the four redundant columns (output, metadata, instructions, tool_calls)
/// that are now fully covered by `raw_response`.
fn oracle_v3_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());

    // Resolve each redundant field to its physical column name (uppercased for Oracle).
    // Skip if another field maps to the same physical name or it's an extra column.
    // Drop one column per statement so a missing column doesn't block dropping others.
    let redundant = ["output", "metadata", "instructions", "tool_calls"];

    redundant
        .iter()
        .filter_map(|&field| {
            let col = s.col(field).to_uppercase();
            let mapped_by_non_redundant_field = s.columns.iter().any(|(k, v)| {
                !k.eq_ignore_ascii_case(field)
                    && !redundant.iter().any(|r| k.eq_ignore_ascii_case(r))
                    && v.eq_ignore_ascii_case(&col)
            });
            let used_as_extra = s.extra_columns.keys().any(|k| k.eq_ignore_ascii_case(&col));
            if mapped_by_non_redundant_field || used_as_extra {
                None
            } else {
                // PL/SQL block: ORA-00904 = "invalid identifier" (column doesn't exist)
                Some(format!(
                    "BEGIN EXECUTE IMMEDIATE 'ALTER TABLE {table} DROP ({col})'; \
                     EXCEPTION WHEN OTHERS THEN IF SQLCODE != -904 THEN RAISE; END IF; END;"
                ))
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../tests/unit/oracle_migrations/tests.rs"]
mod tests;
