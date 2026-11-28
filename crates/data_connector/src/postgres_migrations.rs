//! Postgres-specific schema migrations.
//!
//! Each migration is a function that generates Postgres DDL from [`SchemaConfig`],
//! so it respects custom table/column names. `IF NOT EXISTS` / `IF EXISTS`
//! clauses ensure idempotency.

use crate::{schema::SchemaConfig, versioning::Migration};

const POSTGRES_V1: Migration = Migration {
    version: 1,
    description: "Add safety_identifier column to responses",
    up: pg_v1_up,
};
const POSTGRES_V2: Migration = Migration {
    version: 2,
    description: "Remove legacy user_id column from responses",
    up: pg_v2_up,
};
const POSTGRES_V3: Migration = Migration {
    version: 3,
    description: "Drop redundant output, metadata, instructions, tool_calls columns from responses",
    up: pg_v3_up,
};
const POSTGRES_V4: Migration = Migration {
    version: 4,
    description: "Add tenant column to responses, conversations and conversation items",
    up: pg_v4_up,
};

/// Core history-backend migrations required by the SQL response/conversation
/// storage path during normal gateway startup.
pub(crate) static POSTGRES_HISTORY_MIGRATIONS: [Migration; 4] =
    [POSTGRES_V1, POSTGRES_V2, POSTGRES_V3, POSTGRES_V4];

fn pg_v1_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    if s.is_skipped("safety_identifier") {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    let col = s.col("safety_identifier");
    vec![format!(
        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} VARCHAR(128)"
    )]
}

fn pg_v2_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    // Don't drop user_id if a configured column maps to that name
    // or if it's defined as an extra column.
    if s.columns
        .values()
        .any(|v| v.eq_ignore_ascii_case("user_id"))
        || s.extra_columns
            .keys()
            .any(|k| k.eq_ignore_ascii_case("user_id"))
    {
        return vec![];
    }
    let table = s.qualified_table(schema.owner.as_deref());
    vec![format!("ALTER TABLE {table} DROP COLUMN IF EXISTS user_id")]
}

/// Drop the four redundant columns (output, metadata, instructions, tool_calls)
/// that are now fully covered by `raw_response`.
fn pg_v3_up(schema: &SchemaConfig) -> Vec<String> {
    let s = &schema.responses;
    let table = s.qualified_table(schema.owner.as_deref());

    // Resolve each redundant field to its physical column name, then drop it.
    // Skip if another field maps to the same physical name or it's an extra column.
    let redundant = ["output", "metadata", "instructions", "tool_calls"];

    let cols_to_drop: Vec<_> = redundant
        .iter()
        .filter_map(|&field| {
            let col = s.col(field);
            let mapped_by_non_redundant_field = s.columns.iter().any(|(k, v)| {
                !k.eq_ignore_ascii_case(field)
                    && !redundant.iter().any(|r| k.eq_ignore_ascii_case(r))
                    && v.eq_ignore_ascii_case(col)
            });
            let used_as_extra = s.extra_columns.keys().any(|k| k.eq_ignore_ascii_case(col));
            if mapped_by_non_redundant_field || used_as_extra {
                None
            } else {
                Some(format!("DROP COLUMN IF EXISTS {col}"))
            }
        })
        .collect();

    if cols_to_drop.is_empty() {
        return vec![];
    }

    vec![format!("ALTER TABLE {table} {}", cols_to_drop.join(", "))]
}

/// The nullable `tenant` column of responses, conversations and conversation
/// items: the tenant a record is scoped to. Rows from before the column stay
/// NULL and are served to every caller, as they were.
fn pg_v4_up(schema: &SchemaConfig) -> Vec<String> {
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
        format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} VARCHAR(255)")
    })
    .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../tests/unit/postgres_migrations/tests.rs"]
mod tests;
