//! Postgres storage implementation using PostgresStore helper

use std::{str::FromStr, sync::Arc};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use serde_json::Value;
use tokio_postgres::{NoTls, Row};

use crate::{
    common::{build_response_select_base, extra_column_defs, resolve_extra_column_values},
    config::PostgresConfig,
    context::current_extra_columns,
    core::{
        make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemResult, ConversationItemStorage, ConversationItemStorageError,
        ConversationMetadata, ConversationResult, ConversationStorage, ConversationStorageError,
        ListParams, NewConversation, NewConversationItem, ResponseId, ResponseResult,
        ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
    },
    postgres_migrations::POSTGRES_HISTORY_MIGRATIONS,
    schema::SchemaConfig,
};

// ── Store ────────────────────────────────────────────────────────────────

pub(crate) struct PostgresStore {
    pool: Pool,
    pub(crate) schema: Arc<SchemaConfig>,
}

impl PostgresStore {
    pub fn new(config: PostgresConfig) -> Result<Self, String> {
        let schema = config.schema.clone().unwrap_or_default();
        schema.validate()?;
        let schema = Arc::new(schema);

        let pg_config = tokio_postgres::Config::from_str(config.db_url.as_str())
            .map_err(|e| format!("Invalid PostgreSQL connection URL: {e}"))?;
        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let mgr = Manager::from_config(pg_config, NoTls, mgr_config);
        let pool = Pool::builder(mgr)
            .max_size(config.pool_max)
            .build()
            .map_err(|e| format!("Failed to build PostgreSQL connection pool: {e}"))?;

        Ok(Self { pool, schema })
    }

    /// Run versioned schema migrations after tables have been created.
    pub(crate) async fn run_migrations(&self) -> Result<Vec<u32>, String> {
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|e| format!("failed to get connection for migrations: {e}"))?;
        crate::versioning::run_postgres_migrations(
            &mut client,
            &self.schema,
            &POSTGRES_HISTORY_MIGRATIONS,
            self.schema.version,
            self.schema.auto_migrate,
        )
        .await
    }

    /// Create indexes that may have been deferred during init because
    /// migration-added columns did not yet exist.
    pub(crate) async fn ensure_response_indexes(&self) -> Result<(), String> {
        let s = &self.schema.responses;
        if s.is_skipped("safety_identifier") {
            return Ok(());
        }
        let table = s.qualified_table(self.schema.owner.as_deref());
        let col = s.col("safety_identifier");
        let idx_ddl =
            format!("CREATE INDEX IF NOT EXISTS responses_safety_idx ON {table} ({col});");
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| format!("failed to get connection for index creation: {e}"))?;
        client
            .batch_execute(&idx_ddl)
            .await
            .map_err(|e| format!("failed to create response index: {e}"))?;
        Ok(())
    }
}

impl Clone for PostgresStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            schema: self.schema.clone(),
        }
    }
}

/// Name of a JSON value's type, for error messages that must not echo
/// caller-supplied content (metadata is arbitrary and may hold secrets/PII).
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(super) struct PostgresConversationStorage {
    store: PostgresStore,
}

impl PostgresConversationStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ConversationStorageError> {
        let s = &store.schema.conversations;
        let table = s.qualified_table(store.schema.owner.as_deref());

        let mut col_defs = vec![format!("{} VARCHAR(64) PRIMARY KEY", s.col("id"))];
        if !s.is_skipped("created_at") {
            col_defs.push(format!("{} TIMESTAMPTZ", s.col("created_at")));
        }
        if !s.is_skipped("metadata") {
            col_defs.push(format!("{} JSON", s.col("metadata")));
        }
        col_defs.extend(extra_column_defs(s));

        let ddl = format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", ")
        );

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&ddl)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(Self { store })
    }

    /// Postgres stores metadata as a native `JSON` column, so it round-trips as
    /// `serde_json::Value` rather than the text that Oracle/Redis persist —
    /// `tokio-postgres` rejects decoding a `JSON` column as `String`.
    fn parse_metadata(
        metadata: Option<Value>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        match metadata {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(map)) => Ok(Some(map)),
            // Don't interpolate the value itself: metadata is arbitrary
            // caller-supplied JSON, and this error is rendered via Display
            // into logs/API responses.
            Some(other) => Err(ConversationStorageError::StorageError(format!(
                "expected JSON object for conversation metadata, got JSON {}",
                json_type_name(&other)
            ))),
        }
    }
}

#[async_trait]
impl ConversationStorage for PostgresConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.as_str();
        let created_at: DateTime<Utc> = conversation.created_at;
        // Bind the JSON column as `Value`, not a serialized `String`: Postgres
        // reports the placeholder's type as `json`/`jsonb` from the target
        // column, and `tokio-postgres` rejects a `String` value for that
        // parameter type instead of implicitly casting it.
        let metadata_value: Option<Value> = conversation.metadata.clone().map(Value::Object);

        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());

        let mut col_names: Vec<&str> = vec![s.col("id")];
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![&id_str];
        if !s.is_skipped("created_at") {
            col_names.push(s.col("created_at"));
            params.push(&created_at);
        }
        if !s.is_skipped("metadata") {
            col_names.push(s.col("metadata"));
            params.push(&metadata_value);
        }

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        let extra_cols: Vec<(&str, Option<String>)> = resolve_extra_column_values(s, &hook_extra);
        for (name, val) in &extra_cols {
            col_names.push(*name);
            params.push(val);
        }

        let placeholders: String = (1..=params.len())
            .map(|i| format!("${i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders})",
            col_names.join(", ")
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &params)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let mut select_cols = vec![col_id.to_string()];
        if !s.is_skipped("created_at") {
            select_cols.push(s.col("created_at").to_string());
        }
        if !s.is_skipped("metadata") {
            select_cols.push(s.col("metadata").to_string());
        }

        let sql = format!(
            "SELECT {} FROM {table} WHERE {col_id} = $1",
            select_cols.join(", ")
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&id.0.as_str()])
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let id_str: String = row.get(s.col("id"));
        let created_at: DateTime<Utc> = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            row.get(s.col("created_at"))
        };
        // Use `try_get` (not `get`) so schema drift or malformed data in the
        // `JSON` column returns an error instead of panicking the request task.
        let metadata_json: Option<Value> = if s.is_skipped("metadata") {
            None
        } else {
            row.try_get(s.col("metadata")).map_err(|e| {
                ConversationStorageError::StorageError(format!(
                    "failed to decode metadata column: {e}"
                ))
            })?
        };
        let metadata = Self::parse_metadata(metadata_json)?;
        Ok(Some(Conversation::with_parts(
            ConversationId(id_str),
            created_at,
            metadata,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        if s.is_skipped("metadata") {
            // Nothing to update — just verify the row exists
            let sql = format!("SELECT 1 FROM {table} WHERE {col_id} = $1");
            let rows = client
                .query(&sql, &[&id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
            if rows.is_empty() {
                return Ok(None);
            }
            let created_at = Utc::now();
            return Ok(Some(Conversation::with_parts(
                ConversationId(id.0.clone()),
                created_at,
                metadata,
            )));
        }

        // Bind the JSON column as `Value`, not a serialized `String` (see
        // `create_conversation` for why `tokio-postgres` requires this).
        let metadata_value: Option<Value> = metadata.clone().map(Value::Object);
        let col_meta = s.col("metadata");

        let (_, created_at) = if s.is_skipped("created_at") {
            let sql = format!("UPDATE {table} SET {col_meta} = $1 WHERE {col_id} = $2");
            let rows_affected = client
                .execute(&sql, &[&metadata_value, &id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
            if rows_affected == 0 {
                return Ok(None);
            }
            (sql, Utc::now())
        } else {
            let col_created = s.col("created_at");
            let sql = format!(
                "UPDATE {table} SET {col_meta} = $1 WHERE {col_id} = $2 RETURNING {col_created}"
            );
            let rows = client
                .query(&sql, &[&metadata_value, &id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
            if rows.is_empty() {
                return Ok(None);
            }
            let created_at: DateTime<Utc> = rows[0].get(col_created);
            (sql, created_at)
        };

        Ok(Some(Conversation::with_parts(
            ConversationId(id.0.clone()),
            created_at,
            metadata,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let rows_deleted = client
            .execute(
                &format!("DELETE FROM {table} WHERE {col_id} = $1"),
                &[&id.0.as_str()],
            )
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        Ok(rows_deleted > 0)
    }
}

pub(super) struct PostgresConversationItemStorage {
    store: PostgresStore,
}

impl PostgresConversationItemStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ConversationItemStorageError> {
        let schema = &store.schema;
        let si = &schema.conversation_items;
        let sl = &schema.conversation_item_links;
        let items_table = si.qualified_table(schema.owner.as_deref());
        let links_table = sl.qualified_table(schema.owner.as_deref());

        // ── conversation_items DDL ──
        let mut item_col_defs = vec![format!("{} VARCHAR(64) PRIMARY KEY", si.col("id"))];
        let item_core_cols: [(&str, &str); 6] = [
            ("response_id", "VARCHAR(64)"),
            ("item_type", "VARCHAR(32) NOT NULL"),
            ("role", "VARCHAR(32)"),
            ("content", "JSON"),
            ("status", "VARCHAR(32)"),
            ("created_at", "TIMESTAMPTZ"),
        ];
        for (logical, sql_type) in &item_core_cols {
            if !si.is_skipped(logical) {
                item_col_defs.push(format!("{} {sql_type}", si.col(logical)));
            }
        }
        item_col_defs.extend(extra_column_defs(si));

        // ── conversation_item_links DDL ──
        let col_conv_id = sl.col("conversation_id");
        let col_item_id = sl.col("item_id");
        let col_added_at = sl.col("added_at");

        let mut link_col_defs = vec![
            format!("{col_conv_id} VARCHAR(64)"),
            format!("{col_item_id} VARCHAR(64) NOT NULL"),
            format!("{col_added_at} TIMESTAMPTZ"),
        ];
        link_col_defs.extend(extra_column_defs(sl));
        link_col_defs.push(format!(
            "CONSTRAINT pk_conv_item_link PRIMARY KEY ({col_conv_id}, {col_item_id})"
        ));

        let ddl = format!(
            "CREATE TABLE IF NOT EXISTS {items_table} ({});\n\
             CREATE TABLE IF NOT EXISTS {links_table} ({});\n\
             CREATE INDEX IF NOT EXISTS conv_item_links_conv_idx ON {links_table} ({col_conv_id}, {col_added_at});",
            item_col_defs.join(", "),
            link_col_defs.join(", "),
        );

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&ddl)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(Self { store })
    }
}

#[async_trait]
impl ConversationItemStorage for PostgresConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let NewConversationItem {
            id: opt_id,
            response_id,
            item_type,
            role,
            content,
            status,
        } = item;
        let id = opt_id.unwrap_or_else(|| make_item_id(&item_type));
        let created_at = Utc::now();

        let si = &self.store.schema.conversation_items;
        let table = si.qualified_table(self.store.schema.owner.as_deref());

        // Build dynamic column/param lists, respecting skip_columns
        let mut col_names: Vec<&str> = vec![si.col("id")];
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![&id.0];
        if !si.is_skipped("response_id") {
            col_names.push(si.col("response_id"));
            params.push(&response_id);
        }
        if !si.is_skipped("item_type") {
            col_names.push(si.col("item_type"));
            params.push(&item_type);
        }
        if !si.is_skipped("role") {
            col_names.push(si.col("role"));
            params.push(&role);
        }
        if !si.is_skipped("content") {
            col_names.push(si.col("content"));
            // Bind the JSON column as `Value`, not a serialized `String`
            // (see `create_conversation` for why `tokio-postgres` requires
            // this for a `json`/`jsonb`-typed parameter).
            params.push(&content);
        }
        if !si.is_skipped("status") {
            col_names.push(si.col("status"));
            params.push(&status);
        }
        if !si.is_skipped("created_at") {
            col_names.push(si.col("created_at"));
            params.push(&created_at);
        }

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        let extra_cols: Vec<(&str, Option<String>)> = resolve_extra_column_values(si, &hook_extra);
        for (name, val) in &extra_cols {
            col_names.push(*name);
            params.push(val);
        }

        let placeholders: String = (1..=params.len())
            .map(|i| format!("${i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders})",
            col_names.join(", ")
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &params)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
        })
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());

        let mut col_names: Vec<&str> = vec![
            sl.col("conversation_id"),
            sl.col("item_id"),
            sl.col("added_at"),
        ];
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
            vec![&conversation_id.0, &item_id.0, &added_at];

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        let extra_cols: Vec<(&str, Option<String>)> = resolve_extra_column_values(sl, &hook_extra);
        for (name, val) in &extra_cols {
            col_names.push(*name);
            params.push(val);
        }

        let placeholders: String = (1..=params.len())
            .map(|i| format!("${i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders})",
            col_names.join(", ")
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &params)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let schema = &self.store.schema;
        let si = &schema.conversation_items;
        let sl = &schema.conversation_item_links;
        let items_table = si.qualified_table(schema.owner.as_deref());
        let links_table = sl.qualified_table(schema.owner.as_deref());

        let l_conv_id = sl.col("conversation_id");
        let l_item_id = sl.col("item_id");
        let l_added_at = sl.col("added_at");
        let i_id = si.col("id");

        let cid = conversation_id.0.as_str();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);

        let after_key: Option<(DateTime<Utc>, String)> = if let Some(ref aid) = params.after {
            let client = self
                .store
                .pool
                .get()
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            let cursor_sql = format!(
                "SELECT {l_added_at} FROM {links_table} WHERE {l_conv_id} = $1 AND {l_item_id} = $2"
            );
            let rows = client
                .query(&cursor_sql, &[&cid, &aid.as_str()])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            if rows.is_empty() {
                None
            } else {
                let row = &rows[0];
                let ts: DateTime<Utc> = row.get(0);
                Some((ts, aid.clone()))
            }
        } else {
            None
        };

        // Build select columns from items table (prefixed with i.), respecting skip_columns
        let mut select_cols = vec![format!("i.{}", si.col("id"))];
        for field in &[
            "response_id",
            "item_type",
            "role",
            "content",
            "status",
            "created_at",
        ] {
            if !si.is_skipped(field) {
                select_cols.push(format!("i.{}", si.col(field)));
            }
        }

        let mut sql = format!(
            "SELECT {} FROM {links_table} l JOIN {items_table} i ON i.{i_id} = l.{l_item_id} \
             WHERE l.{l_conv_id} = $1",
            select_cols.join(", "),
        );

        if let Some((_ts, _iid)) = &after_key {
            if order_desc {
                sql.push_str(&format!(
                    " AND (l.{l_added_at} < $2 OR (l.{l_added_at} = $2 AND l.{l_item_id} < $3))"
                ));
            } else {
                sql.push_str(&format!(
                    " AND (l.{l_added_at} > $2 OR (l.{l_added_at} = $2 AND l.{l_item_id} > $3))"
                ));
            }
        }
        if order_desc {
            sql.push_str(&format!(
                " ORDER BY l.{l_added_at} DESC, l.{l_item_id} DESC"
            ));
        } else {
            sql.push_str(&format!(" ORDER BY l.{l_added_at} ASC, l.{l_item_id} ASC"));
        }
        if after_key.is_some() {
            sql.push_str(" LIMIT $4");
        } else {
            sql.push_str(" LIMIT $2");
        }

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let rows = if let Some((ts, iid)) = &after_key {
            client
                .query(&sql, &[&cid, ts, iid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
        } else {
            client
                .query(&sql, &[&cid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
        };

        let mut out = Vec::new();
        for row in rows {
            out.push(build_item_from_row(&row, si)?);
        }
        Ok(out)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> Result<Option<ConversationItem>, ConversationItemStorageError> {
        let si = &self.store.schema.conversation_items;
        let table = si.qualified_table(self.store.schema.owner.as_deref());
        let col_id = si.col("id");

        let mut select_cols = vec![si.col("id").to_string()];
        for field in &[
            "response_id",
            "item_type",
            "role",
            "content",
            "status",
            "created_at",
        ] {
            if !si.is_skipped(field) {
                select_cols.push(si.col(field).to_string());
            }
        }

        let sql = format!(
            "SELECT {} FROM {table} WHERE {col_id} = $1",
            select_cols.join(", "),
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(build_item_from_row(&rows[0], si)?))
        }
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());
        let col_conv = sl.col("conversation_id");
        let col_item = sl.col("item_id");

        let sql = format!("SELECT COUNT(*) FROM {table} WHERE {col_conv} = $1 AND {col_item} = $2");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let row = client
            .query_one(&sql, &[&conversation_id.0.as_str(), &item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let count: i64 = row.get(0);
        Ok(count > 0)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let sl = &self.store.schema.conversation_item_links;
        let table = sl.qualified_table(self.store.schema.owner.as_deref());
        let col_conv = sl.col("conversation_id");
        let col_item = sl.col("item_id");

        let sql = format!("DELETE FROM {table} WHERE {col_conv} = $1 AND {col_item} = $2");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        client
            .execute(&sql, &[&conversation_id.0.as_str(), &item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        Ok(())
    }
}

/// Parse a single `tokio_postgres::Row` into a `ConversationItem`, respecting
/// the `is_skipped` guards configured in `TableConfig`.
fn build_item_from_row(
    row: &Row,
    si: &crate::schema::TableConfig,
) -> Result<ConversationItem, ConversationItemStorageError> {
    let id: String = row.get(si.col("id"));
    let response_id: Option<String> = if si.is_skipped("response_id") {
        None
    } else {
        row.get(si.col("response_id"))
    };
    let item_type: String = if si.is_skipped("item_type") {
        String::new()
    } else {
        row.get(si.col("item_type"))
    };
    let role: Option<String> = if si.is_skipped("role") {
        None
    } else {
        row.get(si.col("role"))
    };
    let content: Value = if si.is_skipped("content") {
        Value::Null
    } else {
        // `content` is a native Postgres `JSON` column; it decodes as
        // `serde_json::Value`, not `String` (see `build_response_from_row`).
        // Use `try_get` (not `get`) so schema drift or malformed data
        // returns an error instead of panicking the request task.
        let content_raw: Option<Value> = row.try_get(si.col("content")).map_err(|e| {
            ConversationItemStorageError::StorageError(format!(
                "failed to decode content column: {e}"
            ))
        })?;
        content_raw.unwrap_or(Value::Null)
    };
    let status: Option<String> = if si.is_skipped("status") {
        None
    } else {
        row.get(si.col("status"))
    };
    let created_at: DateTime<Utc> = if si.is_skipped("created_at") {
        Utc::now()
    } else {
        row.get(si.col("created_at"))
    };

    Ok(ConversationItem {
        id: ConversationItemId(id),
        response_id,
        item_type,
        role,
        content,
        status,
        created_at,
    })
}

pub(super) struct PostgresResponseStorage {
    store: PostgresStore,
    select_base: String,
}

impl PostgresResponseStorage {
    pub async fn new(store: PostgresStore) -> Result<Self, ResponseStorageError> {
        let schema = &store.schema;
        let s = &schema.responses;
        let table = s.qualified_table(schema.owner.as_deref());

        // Build DDL column definitions, filtering out skip_columns and appending extras
        let mut col_defs = vec![format!("{} VARCHAR(64) PRIMARY KEY", s.col("id"))];
        let core_cols: [(&str, &str); 7] = [
            ("conversation_id", "VARCHAR(64)"),
            ("previous_response_id", "VARCHAR(64)"),
            ("input", "JSON"),
            ("created_at", "TIMESTAMPTZ"),
            ("safety_identifier", "VARCHAR(128)"),
            ("model", "VARCHAR(128)"),
            ("raw_response", "JSON"),
        ];
        for (logical, sql_type) in &core_cols {
            if !s.is_skipped(logical) {
                col_defs.push(format!("{} {sql_type}", s.col(logical)));
            }
        }
        col_defs.extend(extra_column_defs(s));

        let table_ddl = format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", "),
        );

        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        client
            .batch_execute(&table_ddl)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Index creation is separate so legacy tables missing migrated
        // columns (e.g. safety_identifier) don't block startup. Indexes
        // are retried after migrations via ensure_response_indexes().
        if !s.is_skipped("safety_identifier") {
            let idx_ddl = format!(
                "CREATE INDEX IF NOT EXISTS responses_safety_idx ON {table} ({});",
                s.col("safety_identifier")
            );
            if let Err(e) = client.batch_execute(&idx_ddl).await {
                tracing::debug!("deferred response index creation (column may not exist yet): {e}");
            }
        }

        let select_base = build_response_select_base(&store.schema);
        Ok(Self { store, select_base })
    }

    pub fn build_response_from_row(
        row: &Row,
        schema: &SchemaConfig,
    ) -> Result<StoredResponse, ResponseStorageError> {
        let s = &schema.responses;

        let id: String = row.get(s.col("id"));
        let conversation_id: Option<String> = if s.is_skipped("conversation_id") {
            None
        } else {
            row.get(s.col("conversation_id"))
        };
        let previous: Option<String> = if s.is_skipped("previous_response_id") {
            None
        } else {
            row.get(s.col("previous_response_id"))
        };
        // `input` and `raw_response` are native Postgres `JSON` columns, so they
        // decode as `serde_json::Value` — `tokio-postgres` panics if a `JSON`
        // column is fetched as `String` (see WrongType in issue #1930). Use
        // `try_get` (not `get`) so schema drift or malformed data returns an
        // error instead of panicking the request task.
        let input_json: Option<Value> = if s.is_skipped("input") {
            None
        } else {
            row.try_get(s.col("input")).map_err(|e| {
                ResponseStorageError::StorageError(format!("failed to decode input column: {e}"))
            })?
        };
        let created_at: DateTime<Utc> = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            row.get(s.col("created_at"))
        };
        let safety_identifier: Option<String> = if s.is_skipped("safety_identifier") {
            None
        } else {
            row.get(s.col("safety_identifier"))
        };
        let model: Option<String> = if s.is_skipped("model") {
            None
        } else {
            row.get(s.col("model"))
        };
        let raw_response_json: Option<Value> = if s.is_skipped("raw_response") {
            None
        } else {
            row.try_get(s.col("raw_response")).map_err(|e| {
                ResponseStorageError::StorageError(format!(
                    "failed to decode raw_response column: {e}"
                ))
            })?
        };

        let previous_response_id = previous.map(ResponseId);
        let raw_response = raw_response_json.unwrap_or(Value::Null);
        let input = input_json.unwrap_or_else(|| Value::Array(Vec::new()));

        Ok(StoredResponse {
            id: ResponseId(id),
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
        })
    }
}

/// The statement that stores a response: an insert that becomes an update of
/// every other column when a row with the same id exists (the first column
/// is the id, the table's primary key), so a record written as `queued`
/// is replaced by its terminal object under the one id.
fn store_response_sql(table: &str, col_names: &[&str]) -> String {
    let placeholders: String = (1..=col_names.len())
        .map(|i| format!("${i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let id_col = col_names.first().copied().unwrap_or("id");
    let updates: Vec<String> = col_names
        .iter()
        .skip(1)
        .map(|col| format!("{col} = EXCLUDED.{col}"))
        .collect();
    if updates.is_empty() {
        return format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders}) ON CONFLICT ({id_col}) DO NOTHING",
            col_names.join(", ")
        );
    }
    format!(
        "INSERT INTO {table} ({}) VALUES ({placeholders}) ON CONFLICT ({id_col}) DO UPDATE SET {}",
        col_names.join(", "),
        updates.join(", ")
    )
}

#[async_trait]
impl ResponseStorage for PostgresResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let StoredResponse {
            id: response_id,
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
        } = response;
        let previous_id = previous_response_id.map(|r| r.0);

        let s = &self.store.schema.responses;
        let table = s.qualified_table(self.store.schema.owner.as_deref());

        // Build dynamic column/param lists, skipping configured skip_columns
        let mut col_names: Vec<&str> = vec![s.col("id")];
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![&response_id.0];

        if !s.is_skipped("conversation_id") {
            col_names.push(s.col("conversation_id"));
            params.push(&conversation_id);
        }
        if !s.is_skipped("previous_response_id") {
            col_names.push(s.col("previous_response_id"));
            params.push(&previous_id);
        }
        if !s.is_skipped("input") {
            col_names.push(s.col("input"));
            params.push(&input);
        }
        if !s.is_skipped("created_at") {
            col_names.push(s.col("created_at"));
            params.push(&created_at);
        }
        if !s.is_skipped("safety_identifier") {
            col_names.push(s.col("safety_identifier"));
            params.push(&safety_identifier);
        }
        if !s.is_skipped("model") {
            col_names.push(s.col("model"));
            params.push(&model);
        }
        if !s.is_skipped("raw_response") {
            col_names.push(s.col("raw_response"));
            params.push(&raw_response);
        }

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        let extra_cols: Vec<(&str, Option<String>)> = resolve_extra_column_values(s, &hook_extra);
        for (name, val) in &extra_cols {
            col_names.push(*name);
            params.push(val);
        }

        let sql = store_response_sql(&table, &col_names);

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let insert_count = client
            .execute(&sql, &params)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        tracing::debug!(rows_affected = insert_count, "Response stored in Postgres");
        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let col_id = self.store.schema.responses.col("id");
        let sql = format!("{} WHERE {col_id} = $1", self.select_base);

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows = client
            .query(&sql, &[&response_id.0.as_str()])
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        if rows.is_empty() {
            return Ok(None);
        }
        Self::build_response_from_row(&rows[0], &self.store.schema).map(Some)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let s = &self.store.schema.responses;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        client
            .execute(
                &format!("DELETE FROM {table} WHERE {col_id} = $1"),
                &[&response_id.0.as_str()],
            )
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let s = &self.store.schema.responses;

        // safety_identifier must exist to filter by it
        if s.is_skipped("safety_identifier") {
            return Ok(vec![]);
        }

        let col_safety = s.col("safety_identifier");

        let order_clause = if s.is_skipped("created_at") {
            String::new()
        } else {
            format!(" ORDER BY {} DESC", s.col("created_at"))
        };

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows = if let Some(l) = limit {
            let l_i64: i64 = l as i64;
            let sql = format!(
                "{} WHERE {col_safety} = $1{order_clause} LIMIT $2",
                self.select_base,
            );
            client
                .query(&sql, &[&identifier, &l_i64])
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?
        } else {
            let sql = format!("{} WHERE {col_safety} = $1{order_clause}", self.select_base,);
            client
                .query(&sql, &[&identifier])
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?
        };

        let schema = &self.store.schema;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(Self::build_response_from_row(&row, schema)?);
        }

        Ok(out)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let s = &self.store.schema.responses;

        // safety_identifier must exist to filter by it
        if s.is_skipped("safety_identifier") {
            return Ok(0);
        }

        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_safety = s.col("safety_identifier");

        let sql = format!("DELETE FROM {table} WHERE {col_safety} = $1");

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let rows_deleted = client
            .execute(&sql, &[&identifier])
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        Ok(rows_deleted as usize)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

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
    #[test]
    fn store_response_sql_replaces_an_existing_row_by_id() {
        let sql = store_response_sql("responses", &["id", "input", "raw_response"]);
        assert_eq!(
            sql,
            "INSERT INTO responses (id, input, raw_response) VALUES ($1, $2, $3) \
             ON CONFLICT (id) DO UPDATE SET input = EXCLUDED.input, raw_response = EXCLUDED.raw_response"
        );
        let sql = store_response_sql("r", &["id"]);
        assert_eq!(
            sql,
            "INSERT INTO r (id) VALUES ($1) ON CONFLICT (id) DO NOTHING"
        );
    }
}
