use super::*;

// ── Default config ────────────────────────────────────────────────────

#[test]
fn default_config_matches_hardcoded_names() {
    let cfg = SchemaConfig::default();
    assert_eq!(cfg.conversations.table, "conversations");
    assert_eq!(cfg.responses.table, "responses");
    assert_eq!(cfg.conversation_items.table, "conversation_items");
    assert_eq!(cfg.conversation_item_links.table, "conversation_item_links");
    assert!(cfg.owner.is_none());
}

#[test]
fn default_config_validates_successfully() {
    SchemaConfig::default()
        .validate()
        .expect("default config should be valid");
}

// ── col() ─────────────────────────────────────────────────────────────

#[test]
fn col_returns_field_name_when_no_override() {
    let tc = TableConfig::with_table("t");
    assert_eq!(tc.col("id"), "id");
    assert_eq!(tc.col("created_at"), "created_at");
}

#[test]
fn col_returns_override_when_configured() {
    let mut tc = TableConfig::with_table("t");
    tc.columns
        .insert("id".to_string(), "CONVERSATION_ID".to_string());
    assert_eq!(tc.col("id"), "CONVERSATION_ID");
    // Non-overridden field still returns itself
    assert_eq!(tc.col("created_at"), "created_at");
}

// ── qualified_table() ─────────────────────────────────────────────────

#[test]
fn qualified_table_without_owner() {
    let tc = TableConfig::with_table("conversations");
    assert_eq!(tc.qualified_table(None), "conversations");
}

#[test]
fn qualified_table_with_owner() {
    let tc = TableConfig::with_table("CONVERSATIONS");
    assert_eq!(tc.qualified_table(Some("ADMIN")), "ADMIN.\"CONVERSATIONS\"");
}

// ── uppercase_for_oracle() ──────────────────────────────────────────

#[test]
fn uppercase_for_oracle_converts_defaults() {
    let mut cfg = SchemaConfig::default();
    cfg.uppercase_for_oracle();
    assert_eq!(cfg.conversations.table, "CONVERSATIONS");
    assert_eq!(cfg.responses.table, "RESPONSES");
    assert_eq!(cfg.conversation_items.table, "CONVERSATION_ITEMS");
    assert_eq!(cfg.conversation_item_links.table, "CONVERSATION_ITEM_LINKS");
    cfg.validate().expect("uppercased config should be valid");
}

#[test]
fn uppercase_for_oracle_converts_custom_table_and_columns() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.table = "my_convos".to_string();
    cfg.conversations
        .columns
        .insert("id".to_string(), "conv_id".to_string());
    cfg.uppercase_for_oracle();
    assert_eq!(cfg.conversations.table, "MY_CONVOS");
    assert_eq!(cfg.conversations.col("id"), "CONV_ID");
}

// ── validate() ────────────────────────────────────────────────────────

#[test]
fn validate_accepts_valid_identifiers() {
    let mut cfg = SchemaConfig {
        owner: Some("ADMIN_01".to_string()),
        ..Default::default()
    };
    cfg.conversations.table = "MY_CONVERSATIONS".to_string();
    cfg.conversations
        .columns
        .insert("id".to_string(), "CONV_ID".to_string());
    cfg.validate().expect("should be valid");
}

#[test]
fn validate_rejects_empty_table_name_with_helpful_message() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.table = String::new();
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("conversations.table") && err.contains("table name is required"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_overly_long_identifier() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.table = "a".repeat(129);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("exceeds maximum length"), "unexpected: {err}");
}

#[test]
fn validate_rejects_special_characters() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.table = "table;DROP".to_string();
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("conversations.table") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_dots_in_owner() {
    let cfg = SchemaConfig {
        owner: Some("ADMIN.SCHEMA".to_string()),
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("owner") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_invalid_column_override_key() {
    let mut cfg = SchemaConfig::default();
    cfg.responses
        .columns
        .insert("bad;col".to_string(), "GOOD_COL".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("columns key") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_invalid_column_override_value() {
    let mut cfg = SchemaConfig::default();
    cfg.responses
        .columns
        .insert("id".to_string(), "bad column!".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("columns value") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

// ── Serde roundtrip ───────────────────────────────────────────────────

#[test]
fn serde_roundtrip_default() {
    let cfg = SchemaConfig::default();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let restored: SchemaConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg, restored);
}

#[test]
fn serde_roundtrip_custom() {
    let mut cfg = SchemaConfig {
        owner: Some("ADMIN".to_string()),
        ..Default::default()
    };
    cfg.conversations.table = "CONVERSATIONS".to_string();
    cfg.conversations
        .columns
        .insert("id".to_string(), "CONVERSATION_ID".to_string());

    let json = serde_json::to_string(&cfg).expect("serialize");
    let restored: SchemaConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg, restored);
}

#[test]
fn serde_deserialize_empty_object_uses_defaults() {
    let cfg: SchemaConfig = serde_json::from_str("{}").expect("deserialize empty");
    assert_eq!(cfg, SchemaConfig::default());
}

// ── version / auto_migrate serde ────────────────────────────────────

#[test]
fn serde_roundtrip_with_version_and_auto_migrate() {
    let cfg = SchemaConfig {
        version: Some(3),
        auto_migrate: false,
        ..Default::default()
    };
    let json = serde_json::to_string(&cfg).expect("serialize");
    let restored: SchemaConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.version, Some(3));
    assert!(!restored.auto_migrate);
}

#[test]
fn serde_defaults_version_none_and_auto_migrate_false() {
    let cfg: SchemaConfig = serde_json::from_str("{}").expect("deserialize empty");
    assert_eq!(cfg.version, None);
    assert!(!cfg.auto_migrate);
}

#[test]
fn serde_version_none_is_omitted_from_json() {
    let cfg = SchemaConfig::default();
    let json = serde_json::to_string(&cfg).expect("serialize");
    assert!(
        !json.contains("version"),
        "version:None should be skipped: {json}"
    );
}

// ── is_skipped() ─────────────────────────────────────────────────────

#[test]
fn is_skipped_returns_false_by_default() {
    let tc = TableConfig::with_table("t");
    assert!(!tc.is_skipped("id"));
    assert!(!tc.is_skipped("safety_identifier"));
}

#[test]
fn is_skipped_returns_true_for_configured_fields() {
    let mut tc = TableConfig::with_table("t");
    tc.skip_columns.insert("safety_identifier".to_string());
    tc.skip_columns.insert("raw_response".to_string());
    assert!(tc.is_skipped("safety_identifier"));
    assert!(tc.is_skipped("raw_response"));
    assert!(!tc.is_skipped("id"));
}

// ── extra_columns ────────────────────────────────────────────────────

#[test]
fn extra_columns_default_is_empty() {
    let tc = TableConfig::with_table("t");
    assert!(tc.extra_columns.is_empty());
}

#[test]
fn validate_accepts_valid_extra_columns() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "EXPIRES_AT".to_string(),
        ColumnDef {
            sql_type: "TIMESTAMP".to_string(),
            default_value: None,
        },
    );
    cfg.validate().expect("should be valid");
}

#[test]
fn validate_rejects_invalid_extra_column_name() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "bad;col".to_string(),
        ColumnDef {
            sql_type: "TEXT".to_string(),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("extra_columns key") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_empty_sql_type() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "MY_COL".to_string(),
        ColumnDef {
            sql_type: String::new(),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("sql_type must not be empty"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_invalid_skip_column_name() {
    let mut cfg = SchemaConfig::default();
    cfg.responses.skip_columns.insert("bad;col".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("skip_columns") && err.contains("invalid identifier"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_accepts_valid_skip_columns() {
    let mut cfg = SchemaConfig::default();
    cfg.responses
        .skip_columns
        .insert("safety_identifier".to_string());
    cfg.validate().expect("should be valid");
}

#[test]
fn validate_rejects_skip_id() {
    let mut cfg = SchemaConfig::default();
    cfg.responses.skip_columns.insert("id".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("cannot skip 'id'") && err.contains("primary key"),
        "unexpected: {err}"
    );
}

/// Each table's own key is refused, not `id` everywhere: the links table
/// has no `id`, its key is (`conversation_id`, `item_id`), and a skip of
/// either is a configuration error at start-up instead of passing
/// validation and being ignored by every backend, which build the links
/// statements from those columns unconditionally.
#[test]
fn validate_rejects_skip_of_each_tables_own_key_columns() {
    for (table, key) in [
        ("conversations", "id"),
        ("conversation_items", "id"),
        ("conversation_item_links", "conversation_id"),
        ("conversation_item_links", "item_id"),
    ] {
        let mut cfg = SchemaConfig::default();
        let tc = match table {
            "conversations" => &mut cfg.conversations,
            "conversation_items" => &mut cfg.conversation_items,
            _ => &mut cfg.conversation_item_links,
        };
        tc.skip_columns.insert(key.to_string());
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains(&format!("{table}.skip_columns: cannot skip '{key}'"))
                && err.contains("primary key"),
            "unexpected: {err}"
        );
    }

    // `id` is not a column of the links table at all: unknown, not a key.
    let mut cfg = SchemaConfig::default();
    cfg.conversation_item_links
        .skip_columns
        .insert("id".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("not a recognized column"), "unexpected: {err}");
}

/// The links table has no skippable column: `added_at` is in its DDL,
/// its insert, its index and its listing order on every backend, so a
/// skip of it passed validation and changed nothing; it is refused at
/// start-up now, while a skip of a column the other tables' statements
/// consult still passes.
#[test]
fn validate_rejects_every_skip_on_the_links_table() {
    let mut cfg = SchemaConfig::default();
    cfg.conversation_item_links
        .skip_columns
        .insert("added_at".to_string());
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("conversation_item_links.skip_columns: cannot skip 'added_at'")
            && err.contains("keeps every column"),
        "unexpected: {err}"
    );

    let mut cfg = SchemaConfig::default();
    cfg.conversation_items
        .skip_columns
        .insert("created_at".to_string());
    cfg.validate()
        .expect("a skip the items statements consult still passes");
}

// ── validate_sql_type ─────────────────────────────────────────────────

#[test]
fn validate_accepts_valid_sql_types() {
    let mut cfg = SchemaConfig::default();
    for sql_type in [
        "VARCHAR(128)",
        "TEXT",
        "TIMESTAMP",
        "TIMESTAMP WITH TIME ZONE",
        "NUMBER(10,2)",
        "VARCHAR2(256)",
        "BIGINT",
    ] {
        cfg.conversations.extra_columns.insert(
            "TEST_COL".to_string(),
            ColumnDef {
                sql_type: sql_type.to_string(),
                default_value: None,
            },
        );
        cfg.validate()
            .unwrap_or_else(|e| panic!("sql_type '{sql_type}' should be valid: {e}"));
    }
}

#[test]
fn validate_rejects_sql_injection_in_sql_type() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "MY_COL".to_string(),
        ColumnDef {
            sql_type: "TEXT); DROP TABLE responses; --".to_string(),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("sql_type") && err.contains("invalid"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_overly_long_sql_type() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "MY_COL".to_string(),
        ColumnDef {
            sql_type: "A".repeat(65),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("sql_type") && err.contains("exceeds maximum length"),
        "unexpected: {err}"
    );
}

// ── uppercase_for_oracle with extra_columns ──────────────────────────

#[test]
fn uppercase_for_oracle_converts_extra_column_keys() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "expires_at".to_string(),
        ColumnDef {
            sql_type: "TIMESTAMP".to_string(),
            default_value: None,
        },
    );
    cfg.uppercase_for_oracle();
    assert!(cfg.conversations.extra_columns.contains_key("EXPIRES_AT"));
    assert!(!cfg.conversations.extra_columns.contains_key("expires_at"));
}

#[test]
fn uppercase_for_oracle_preserves_skip_column_names_lowercase() {
    let mut cfg = SchemaConfig::default();
    cfg.responses
        .skip_columns
        .insert("safety_identifier".to_string());
    cfg.uppercase_for_oracle();
    // skip_columns must stay lowercase — backends call is_skipped("safety_identifier")
    assert!(cfg.responses.skip_columns.contains("safety_identifier"));
    assert!(!cfg.responses.skip_columns.contains("SAFETY_IDENTIFIER"));
}

// ── Serde roundtrip with extra_columns and skip_columns ──────────────

#[test]
fn serde_roundtrip_with_extra_and_skip() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "EXPIRES_AT".to_string(),
        ColumnDef {
            sql_type: "TIMESTAMP".to_string(),
            default_value: Some(Value::String("2099-01-01".to_string())),
        },
    );
    cfg.responses
        .skip_columns
        .insert("safety_identifier".to_string());

    let json = serde_json::to_string(&cfg).expect("serialize");
    let restored: SchemaConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg, restored);
}

#[test]
fn validate_rejects_extra_column_shadowing_core_column() {
    let mut cfg = SchemaConfig::default();
    cfg.responses.extra_columns.insert(
        "CREATED_AT".to_string(),
        ColumnDef {
            sql_type: "TIMESTAMP".to_string(),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("shadows a core column"), "unexpected: {err}");

    // Also check a conversation core column
    let mut cfg2 = SchemaConfig::default();
    cfg2.conversations.extra_columns.insert(
        "metadata".to_string(),
        ColumnDef {
            sql_type: "TEXT".to_string(),
            default_value: None,
        },
    );
    let err2 = cfg2.validate().unwrap_err();
    assert!(err2.contains("shadows a core column"), "unexpected: {err2}");
}

#[test]
fn validate_rejects_unknown_skip_column() {
    let mut cfg = SchemaConfig::default();
    cfg.responses
        .skip_columns
        .insert("safty_identifier".to_string()); // typo
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("not a recognized column"), "unexpected: {err}");
}

#[test]
fn validate_rejects_case_colliding_extra_columns() {
    let mut cfg = SchemaConfig::default();
    cfg.conversations.extra_columns.insert(
        "tenant_id".to_string(),
        ColumnDef {
            sql_type: "TEXT".to_string(),
            default_value: None,
        },
    );
    cfg.conversations.extra_columns.insert(
        "TENANT_ID".to_string(),
        ColumnDef {
            sql_type: "TEXT".to_string(),
            default_value: None,
        },
    );
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("case-insensitive collision"),
        "unexpected: {err}"
    );
}
