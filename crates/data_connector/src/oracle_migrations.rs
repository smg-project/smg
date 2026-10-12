//! Oracle-specific schema migrations.
//!
//! Each migration is a function that generates Oracle DDL from [`SchemaConfig`],
//! so it respects custom table/column names. PL/SQL exception handling ensures
//! idempotency (safe to re-run if a previous attempt partially completed).
#![cfg_attr(not(test), allow(dead_code))]

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
mod tests {
    use super::*;
    use crate::schema::TableConfig;

    #[test]
    fn oracle_history_migrations_cover_only_core_history_schema() {
        let versions: Vec<u32> = ORACLE_HISTORY_MIGRATIONS
            .iter()
            .map(|migration| migration.version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3, 4]);
    }

    #[test]
    fn oracle_v4_up_adds_the_tenant_column_to_responses_and_conversations() {
        let mut schema = SchemaConfig::default();
        schema.uppercase_for_oracle();
        let stmts = oracle_v4_up(&schema);
        assert_eq!(stmts.len(), 3);
        assert!(
            stmts[2].contains("ALTER TABLE CONVERSATION_ITEMS ADD (tenant VARCHAR2(255))"),
            "got: {}",
            stmts[2]
        );
        // An unmapped logical column keeps its spelling, as migration 1's
        // safety_identifier does: Oracle folds the unquoted name to upper case.
        assert!(
            stmts[0].contains("ALTER TABLE RESPONSES ADD (tenant VARCHAR2(255))"),
            "got: {}",
            stmts[0]
        );
        assert!(
            stmts[1].contains("ALTER TABLE CONVERSATIONS ADD (tenant VARCHAR2(255))"),
            "got: {}",
            stmts[1]
        );
        assert!(stmts.iter().all(|s| s.contains("-1430")), "idempotent");
        // skip_columns hold logical (lower-case) names.
        schema
            .conversations
            .skip_columns
            .insert("tenant".to_string());
        assert_eq!(oracle_v4_up(&schema).len(), 2);
    }

    #[test]
    fn oracle_migrations_are_strictly_increasing() {
        // Versions must be strictly increasing and unique; a numbering gap is allowed.
        for pair in ORACLE_HISTORY_MIGRATIONS.windows(2) {
            assert!(
                pair[1].version > pair[0].version,
                "migration versions must strictly increase: {} then {}",
                pair[0].version,
                pair[1].version
            );
        }
    }

    #[test]
    fn oracle_v1_up_generates_plsql_add_column() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v1_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("ADD"), "got: {}", stmts[0]);
        assert!(stmts[0].contains("SQLCODE"), "got: {}", stmts[0]);
    }

    #[test]
    fn oracle_v1_up_skipped_returns_empty() {
        let schema = SchemaConfig {
            responses: TableConfig {
                skip_columns: ["safety_identifier".to_string()].into_iter().collect(),
                ..TableConfig::with_table("responses")
            },
            ..Default::default()
        };
        let stmts = oracle_v1_up(&schema);
        assert!(stmts.is_empty());
    }

    #[test]
    fn oracle_v2_up_generates_plsql_drop_column() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v2_up(&schema);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("DROP"), "got: {}", stmts[0]);
        assert!(stmts[0].contains("USER_ID"), "got: {}", stmts[0]);
    }

    #[test]
    fn oracle_v2_up_skipped_when_column_maps_to_user_id() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "USER_ID".to_string());
        let stmts = oracle_v2_up(&schema);
        assert!(stmts.is_empty(), "should skip drop when USER_ID is mapped");
    }

    #[test]
    fn oracle_v2_up_skipped_when_extra_column_is_user_id() {
        let mut schema = SchemaConfig::default();
        schema.responses.extra_columns.insert(
            "USER_ID".to_string(),
            crate::schema::ColumnDef {
                sql_type: "VARCHAR2(128)".to_string(),
                default_value: None,
            },
        );
        let stmts = oracle_v2_up(&schema);
        assert!(
            stmts.is_empty(),
            "should skip drop when USER_ID is an extra column"
        );
    }

    #[test]
    fn oracle_v3_up_generates_per_column_plsql_drops() {
        let schema = SchemaConfig::default();
        let stmts = oracle_v3_up(&schema);
        assert_eq!(stmts.len(), 4);
        assert!(stmts[0].contains("OUTPUT"), "got: {}", stmts[0]);
        assert!(stmts[1].contains("METADATA"), "got: {}", stmts[1]);
        assert!(stmts[2].contains("INSTRUCTIONS"), "got: {}", stmts[2]);
        assert!(stmts[3].contains("TOOL_CALLS"), "got: {}", stmts[3]);
        for stmt in &stmts {
            assert!(stmt.contains("SQLCODE"), "got: {stmt}");
        }
    }

    #[test]
    fn oracle_v3_up_skips_when_output_is_used_by_another_field() {
        let mut schema = SchemaConfig::default();
        schema
            .responses
            .columns
            .insert("safety_identifier".to_string(), "OUTPUT".to_string());
        let stmts = oracle_v3_up(&schema);
        assert_eq!(stmts.len(), 3, "expected 3 statements (OUTPUT skipped)");
        for stmt in &stmts {
            assert!(
                !stmt.contains("DROP (OUTPUT)"),
                "should skip OUTPUT when mapped: {stmt}"
            );
        }
    }

    /// Oracle pre-12.2 rejects unquoted identifiers over 30 chars with
    /// ORA-00972. This test scans every identifier emitted by every migration
    /// and fails loudly if anything crosses the limit, so future contributors
    /// can't accidentally reintroduce the bug.
    ///
    /// Note: we deliberately do NOT strip the EXECUTE IMMEDIATE literal
    /// content. Oracle DDL identifiers (table / column / constraint / index
    /// names) live INSIDE those literals, so stripping would make the test
    /// check only outer PL/SQL wrapper keywords (BEGIN, EXCEPTION, SQLCODE,
    /// all ≤11 chars) and silently miss real violations. Doubled-quote
    /// string values like `''completed''` tokenize to short words
    /// (`completed` = 9 chars) that never trip the 30-char limit.
    #[test]
    fn all_oracle_migration_identifiers_are_within_30_chars() {
        let schema = SchemaConfig::default();
        let all: Vec<String> = ORACLE_HISTORY_MIGRATIONS
            .iter()
            .flat_map(|m| (m.up)(&schema))
            .collect();

        fn is_ident_char(c: char) -> bool {
            c.is_ascii_alphanumeric() || c == '_'
        }
        let mut violations: Vec<String> = Vec::new();
        for stmt in &all {
            let mut token = String::new();
            for c in stmt.chars().chain(std::iter::once(' ')) {
                if is_ident_char(c) {
                    token.push(c);
                } else {
                    if token.len() > 30 && token.starts_with(|ch: char| !ch.is_ascii_digit()) {
                        violations.push(format!(
                            "identifier `{}` ({} chars) in: {}",
                            token,
                            token.len(),
                            stmt.chars().take(80).collect::<String>()
                        ));
                    }
                    token.clear();
                }
            }
        }
        assert!(
            violations.is_empty(),
            "Oracle identifiers must be ≤30 chars (pre-12.2 limit, ORA-00972). \
             Violations:\n  {}",
            violations.join("\n  ")
        );
    }

    /// Meta-test: plant a 31-char identifier inside an EXECUTE IMMEDIATE
    /// literal and confirm the guard above catches it. Protects against
    /// anyone accidentally reintroducing literal-stripping (which would make
    /// the guard silently useless because real DDL identifiers live INSIDE
    /// the literal).
    #[test]
    fn identifier_length_guard_catches_planted_violation() {
        fn is_ident_char(c: char) -> bool {
            c.is_ascii_alphanumeric() || c == '_'
        }
        let planted = "BEGIN EXECUTE IMMEDIATE \
            'CREATE TABLE AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA (x INT)'; \
            EXCEPTION WHEN OTHERS THEN RAISE; END;";
        let mut hit_long = false;
        let mut token = String::new();
        for c in planted.chars().chain(std::iter::once(' ')) {
            if is_ident_char(c) {
                token.push(c);
            } else {
                if token.len() > 30 && token.starts_with(|ch: char| !ch.is_ascii_digit()) {
                    hit_long = true;
                }
                token.clear();
            }
        }
        assert!(
            hit_long,
            "guard regressed — must detect >30-char identifiers inside EXECUTE IMMEDIATE"
        );
    }
}
