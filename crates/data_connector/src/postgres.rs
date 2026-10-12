//! Postgres storage implementation using PostgresStore helper

use std::{str::FromStr, sync::Arc};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{ClientWrapper, Manager, ManagerConfig, Pool, RecyclingMethod};
use serde_json::Value;
use tokio_postgres::{types::ToSql, IsolationLevel, NoTls, Row};

use crate::{
    common::{
        build_response_select_base, describe_error, extra_column_defs, resolve_extra_column_values,
    },
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
    versioning::{
        postgres_create_versions_table, postgres_table_exists, run_postgres_migrations,
        stamp_postgres_versions, POSTGRES_SCHEMA_LOCK_ID, VERSIONS_TABLE,
    },
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

    /// Brings the database to the schema this build expects, once per
    /// database and in turn across gateways: one transaction under the schema
    /// advisory lock creates the versions table and the history tables (`IF
    /// NOT EXISTS`), records every migration as applied when the responses
    /// table did not exist before (a first start creates the tables in their
    /// final shape, so nothing is pending and no `auto_migrate` is needed),
    /// applies the pending migrations otherwise (or refuses when
    /// `auto_migrate` is off), and creates the indexes last, once the columns
    /// they cover exist. Gateways starting together on an empty database all
    /// come up: the next one waits for the first's commit and finds the
    /// schema in place.
    pub(crate) async fn bootstrap(&self) -> Result<(), String> {
        let mut client = self.pool.get().await.map_err(|e| {
            format!(
                "failed to get a connection for the schema bootstrap: {}",
                describe_error(&e)
            )
        })?;
        // READ COMMITTED whatever the server's default isolation: the version
        // rows are read after the lock wait, in a snapshot taken then. Under a
        // REPEATABLE READ default the snapshot would be the lock statement's
        // own, taken before the wait, and a gateway queued behind another's
        // bootstrap would read the versions table as it was before that
        // bootstrap (the tables there, the versions not) and refuse, or apply
        // the migrations again.
        let mut tx = client
            .build_transaction()
            .isolation_level(IsolationLevel::ReadCommitted)
            .start()
            .await
            .map_err(|e| {
                format!(
                    "failed to begin the schema bootstrap: {}",
                    describe_error(&e)
                )
            })?;
        tx.execute(
            "SELECT pg_advisory_xact_lock($1)",
            &[&POSTGRES_SCHEMA_LOCK_ID],
        )
        .await
        .map_err(|e| format!("failed to acquire the schema lock: {}", describe_error(&e)))?;
        tx.batch_execute(&postgres_create_versions_table())
            .await
            .map_err(|e| {
                format!(
                    "failed to create {VERSIONS_TABLE} table: {}",
                    describe_error(&e)
                )
            })?;

        let schema = &self.schema;
        let responses_table = schema.responses.qualified_table(schema.owner.as_deref());
        let fresh = !postgres_table_exists(&tx, &responses_table).await?;
        for ddl in [
            PostgresConversationStorage::table_ddl(schema),
            PostgresConversationItemStorage::table_ddl(schema),
            PostgresResponseStorage::table_ddl(schema),
        ] {
            tx.batch_execute(&ddl).await.map_err(|e| {
                format!(
                    "failed to create the history tables: {}",
                    describe_error(&e)
                )
            })?;
        }

        if fresh {
            stamp_postgres_versions(&tx, &POSTGRES_HISTORY_MIGRATIONS).await?;
            tracing::info!(
                schema_version = POSTGRES_HISTORY_MIGRATIONS.last().map_or(0, |m| m.version),
                "fresh database: history tables created at the latest schema version"
            );
        } else {
            run_postgres_migrations(
                &tx,
                schema,
                &POSTGRES_HISTORY_MIGRATIONS,
                schema.version,
                schema.auto_migrate,
            )
            .await?;
        }

        // The indexes last, so the share lock a `CREATE INDEX IF NOT EXISTS`
        // takes on its table is not held while the migrations wait for their
        // own locks: the links index on columns the table always has, then
        // the safety-identifier index, whose column exists once the
        // migrations ran. A table the operator declared up to date without
        // the column keeps starting without that index, as before.
        tx.batch_execute(&PostgresConversationItemStorage::index_ddl(schema))
            .await
            .map_err(|e| {
                format!(
                    "failed to create the conversation item links index: {}",
                    describe_error(&e)
                )
            })?;
        if let Some(index_ddl) = PostgresResponseStorage::index_ddl(schema) {
            let index = tx.savepoint("response_index").await.map_err(|e| {
                format!("failed to open the index savepoint: {}", describe_error(&e))
            })?;
            match index.batch_execute(&index_ddl).await {
                Ok(()) => index.commit().await.map_err(|e| {
                    format!(
                        "failed to release the index savepoint: {}",
                        describe_error(&e)
                    )
                })?,
                Err(e) => {
                    tracing::warn!(
                        "response index not created (the safety_identifier column may be missing): {}",
                        describe_error(&e)
                    );
                    index.rollback().await.map_err(|e| {
                        format!(
                            "failed to roll back the index savepoint: {}",
                            describe_error(&e)
                        )
                    })?;
                }
            }
        }

        tx.commit().await.map_err(|e| {
            format!(
                "failed to commit the schema bootstrap: {}",
                describe_error(&e)
            )
        })
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

// ── Cached statements ────────────────────────────────────────────────────
//
// A statement passed as text costs two round trips per call: the parse and
// the execute. Prepared once and cached on the connection (the pool keeps the
// cache with the connection), every later call is one round trip, which is
// what a store a few milliseconds away multiplies by.

/// Runs `sql` as a statement prepared once per connection.
async fn query_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<Vec<Row>, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    client.query(&statement, params).await
}

/// Runs `sql` as a statement prepared once per connection; one row.
async fn query_one_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<Row, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    client.query_one(&statement, params).await
}

/// Runs `sql` as a statement prepared once per connection; the rows affected.
async fn execute_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<u64, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    client.execute(&statement, params).await
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
    pub fn new(store: PostgresStore) -> Self {
        Self { store }
    }

    /// The conversations table in its current shape, created by the store's
    /// bootstrap (`IF NOT EXISTS`).
    pub(crate) fn table_ddl(schema: &SchemaConfig) -> String {
        let s = &schema.conversations;
        let table = s.qualified_table(schema.owner.as_deref());

        let mut col_defs = vec![format!("{} VARCHAR(64) PRIMARY KEY", s.col("id"))];
        if !s.is_skipped("created_at") {
            col_defs.push(format!("{} TIMESTAMPTZ", s.col("created_at")));
        }
        if !s.is_skipped("metadata") {
            col_defs.push(format!("{} JSON", s.col("metadata")));
        }
        col_defs.extend(extra_column_defs(s));

        format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", ")
        )
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
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&id_str];
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
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
        let rows = query_cached(&client, &sql, &[&id.0.as_str()])
            .await
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;

        if s.is_skipped("metadata") {
            // Nothing to update — just verify the row exists
            let sql = format!("SELECT 1 FROM {table} WHERE {col_id} = $1");
            let rows = query_cached(&client, &sql, &[&id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
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
            let rows_affected = execute_cached(&client, &sql, &[&metadata_value, &id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
            if rows_affected == 0 {
                return Ok(None);
            }
            (sql, Utc::now())
        } else {
            let col_created = s.col("created_at");
            let sql = format!(
                "UPDATE {table} SET {col_meta} = $1 WHERE {col_id} = $2 RETURNING {col_created}"
            );
            let rows = query_cached(&client, &sql, &[&metadata_value, &id.0.as_str()])
                .await
                .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
        let rows_deleted = execute_cached(
            &client,
            &format!("DELETE FROM {table} WHERE {col_id} = $1"),
            &[&id.0.as_str()],
        )
        .await
        .map_err(|e| ConversationStorageError::StorageError(describe_error(&e)))?;
        Ok(rows_deleted > 0)
    }
}

pub(super) struct PostgresConversationItemStorage {
    store: PostgresStore,
}

impl PostgresConversationItemStorage {
    pub fn new(store: PostgresStore) -> Self {
        Self { store }
    }

    /// The conversation items and links tables in their current shape,
    /// created by the store's bootstrap (`IF NOT EXISTS`); the links index is
    /// `index_ddl`.
    pub(crate) fn table_ddl(schema: &SchemaConfig) -> String {
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

        format!(
            "CREATE TABLE IF NOT EXISTS {items_table} ({});\n\
             CREATE TABLE IF NOT EXISTS {links_table} ({});",
            item_col_defs.join(", "),
            link_col_defs.join(", "),
        )
    }

    /// The index on the links table's conversation and time columns, created
    /// by the bootstrap after the migrations, so the share lock it takes on
    /// the table is not held while they run.
    pub(crate) fn index_ddl(schema: &SchemaConfig) -> String {
        let sl = &schema.conversation_item_links;
        let links_table = sl.qualified_table(schema.owner.as_deref());
        format!(
            "CREATE INDEX IF NOT EXISTS conv_item_links_conv_idx ON {links_table} ({}, {});",
            sl.col("conversation_id"),
            sl.col("added_at")
        )
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
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&id.0];
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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&conversation_id.0, &item_id.0, &added_at];

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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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

        let after_key: Option<(DateTime<Utc>, String)> =
            if let Some(ref aid) = params.after {
                let client =
                    self.store.pool.get().await.map_err(|e| {
                        ConversationItemStorageError::StorageError(describe_error(&e))
                    })?;
                let cursor_sql = format!(
                "SELECT {l_added_at} FROM {links_table} WHERE {l_conv_id} = $1 AND {l_item_id} = $2"
            );
                let rows = query_cached(&client, &cursor_sql, &[&cid, &aid.as_str()])
                    .await
                    .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        let rows = if let Some((ts, iid)) = &after_key {
            query_cached(&client, &sql, &[&cid, ts, iid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?
        } else {
            query_cached(&client, &sql, &[&cid, &limit])
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?
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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        let rows = query_cached(&client, &sql, &[&item_id.0.as_str()])
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        let row = query_one_cached(
            &client,
            &sql,
            &[&conversation_id.0.as_str(), &item_id.0.as_str()],
        )
        .await
        .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
        execute_cached(
            &client,
            &sql,
            &[&conversation_id.0.as_str(), &item_id.0.as_str()],
        )
        .await
        .map_err(|e| ConversationItemStorageError::StorageError(describe_error(&e)))?;
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
    store_path: StorePath,
}

impl PostgresResponseStorage {
    /// The store over tables the bootstrap created; reads how the responses
    /// table takes a store (one statement under a unique id, or a locked
    /// merge).
    pub async fn new(store: PostgresStore) -> Result<Self, ResponseStorageError> {
        let schema = &store.schema;
        let s = &schema.responses;
        let table = s.qualified_table(schema.owner.as_deref());
        let client = store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let store_path = store_path_for(&client, &table, s.col("id")).await;
        let select_base = build_response_select_base(&store.schema);
        Ok(Self {
            store,
            select_base,
            store_path,
        })
    }

    /// The responses table in its current shape, created by the store's
    /// bootstrap (`IF NOT EXISTS`).
    pub(crate) fn table_ddl(schema: &SchemaConfig) -> String {
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

        format!(
            "CREATE TABLE IF NOT EXISTS {table} ({});",
            col_defs.join(", "),
        )
    }

    /// The index on the safety identifier, created by the bootstrap after
    /// the migrations (a legacy table gains the column in v1), unless the
    /// column is skipped.
    pub(crate) fn index_ddl(schema: &SchemaConfig) -> Option<String> {
        let s = &schema.responses;
        if s.is_skipped("safety_identifier") {
            return None;
        }
        let table = s.qualified_table(schema.owner.as_deref());
        Some(format!(
            "CREATE INDEX IF NOT EXISTS responses_safety_idx ON {table} ({});",
            s.col("safety_identifier")
        ))
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

/// The query behind [`ResponseStorage::find_response_by_output_item`]:
/// `raw_response` is a native `JSON` column, so each row is cast to `jsonb`
/// and asked whether its `output` array contains an item with the id bound as
/// `$1`. `None` when the schema skips the column (nothing to search).
fn find_by_output_item_sql(schema: &SchemaConfig, select_base: &str) -> Option<String> {
    let s = &schema.responses;
    if s.is_skipped("raw_response") {
        return None;
    }
    let col_raw = s.col("raw_response");
    let order = if s.is_skipped("created_at") {
        String::new()
    } else {
        format!(" ORDER BY {} DESC", s.col("created_at"))
    };
    Some(format!(
        "{select_base} WHERE {col_raw}::jsonb @> jsonb_build_object('output', \
         jsonb_build_array(jsonb_build_object('id', $1::text))){order} LIMIT 1"
    ))
}

/// How a response is stored under its id, decided once at start-up from the
/// catalog (`store_path_for`). The table the store creates has a primary key
/// on the id, so it takes the one-statement path; a configured table that
/// pre-exists without a unique index on its mapped id column takes the
/// locked path, which needs none. Either way a record written as `queued`
/// is replaced by its terminal object under the one id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorePath {
    /// `INSERT ... ON CONFLICT (id) DO UPDATE`: one statement, no lock, the
    /// unique index as the arbiter; one round trip once the statement is
    /// prepared on the connection.
    Upsert,
    /// One transaction: a transaction-scoped advisory lock on the id, then
    /// one statement that updates every column of the rows under the id and
    /// inserts when none moved (four round trips: begin, lock, statement,
    /// commit). The lock serializes the stores of one id across every
    /// gateway sharing the database: the statement takes its snapshot once
    /// the lock is held, so two first stores of an id cannot both insert.
    LockedMerge,
}

/// Whether the mapped id column of a table carries an index that
/// `INSERT ... ON CONFLICT (id)` can use as its arbiter: unique, valid, not
/// deferrable, on that one column alone, without a predicate or an
/// expression. `$1` is the table as configured, `$2` the id column; an
/// unquoted column name is stored lower-cased, so both spellings are tried.
const UNIQUE_ID_INDEX_SQL: &str = "SELECT EXISTS (\
    SELECT 1 FROM pg_catalog.pg_index i \
    JOIN pg_catalog.pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] \
    WHERE i.indrelid = to_regclass($1) AND i.indisunique AND i.indisvalid AND i.indimmediate \
    AND i.indnkeyatts = 1 AND i.indpred IS NULL AND i.indexprs IS NULL \
    AND (a.attname::text = $2 OR a.attname::text = lower($2)))";

/// Reads the catalog once and names the path in the log. A failed read (a
/// server without these catalog columns, a missing privilege) takes the
/// locked path, which works on any table.
async fn store_path_for(
    client: &deadpool_postgres::Client,
    table: &str,
    id_col: &str,
) -> StorePath {
    let probe = client
        .query_one(UNIQUE_ID_INDEX_SQL, &[&table, &id_col])
        .await
        .map(|row| row.get::<_, bool>(0));
    match probe {
        Ok(true) => {
            tracing::info!(
                table,
                id_column = id_col,
                "Postgres response store: the id column has a unique index; a store is one INSERT ... ON CONFLICT statement"
            );
            StorePath::Upsert
        }
        Ok(false) => {
            tracing::warn!(
                table,
                id_column = id_col,
                "Postgres response store: no unique index on the id column; every store takes an advisory lock and an UPDATE-then-INSERT transaction (four round trips, and a table scan while no index covers the column). A unique index on {table} ({id_col}) makes a store one statement"
            );
            StorePath::LockedMerge
        }
        Err(e) => {
            tracing::warn!(
                table,
                id_column = id_col,
                error = %e,
                "Postgres response store: could not read the catalog for a unique index on the id column; stores take the locked UPDATE-then-INSERT path"
            );
            StorePath::LockedMerge
        }
    }
}

fn param_placeholders(n: usize) -> String {
    (1..=n)
        .map(|i| format!("${i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The statement of the upsert path: the row is inserted, or every other
/// column of the row under the id (the first column) is replaced;
/// `DO NOTHING` when the id is the only column.
fn upsert_sql(table: &str, col_names: &[&str]) -> String {
    let placeholders = param_placeholders(col_names.len());
    let id_col = col_names.first().copied().unwrap_or("id");
    let columns = col_names.join(", ");
    let updates: Vec<String> = col_names
        .iter()
        .skip(1)
        .map(|col| format!("{col} = EXCLUDED.{col}"))
        .collect();
    if updates.is_empty() {
        return format!(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT ({id_col}) DO NOTHING"
        );
    }
    format!(
        "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT ({id_col}) DO UPDATE SET {}",
        updates.join(", ")
    )
}

/// The data statement of the locked path, run once the advisory lock is
/// held: an UPDATE of every column of the rows under the id, then the INSERT
/// when no row moved, as one statement. Neither part needs a unique index.
/// The id is assigned to itself so every parameter takes its type from the
/// SET list (the INSERT's select list reuses them) and the list is never
/// empty.
fn locked_merge_sql(table: &str, col_names: &[&str]) -> String {
    let placeholders = param_placeholders(col_names.len());
    let id_col = col_names.first().copied().unwrap_or("id");
    let assignments: Vec<String> = col_names
        .iter()
        .enumerate()
        .map(|(i, col)| format!("{col} = ${}", i + 1))
        .collect();
    format!(
        "WITH upd AS (UPDATE {table} SET {} WHERE {id_col} = $1 RETURNING 1) \
         INSERT INTO {table} ({}) SELECT {placeholders} WHERE NOT EXISTS (SELECT 1 FROM upd)",
        assignments.join(", "),
        col_names.join(", ")
    )
}

/// The lock statement of the locked path, sent through the simple query
/// protocol (one round trip, nothing to prepare): the key is a locally
/// computed integer.
fn lock_sql(key: i64) -> String {
    format!("SELECT pg_advisory_xact_lock({key})")
}

/// The key of the transaction-scoped advisory lock a store of `id` holds on
/// the locked path (FNV-1a over the id's bytes): stores of one id serialize
/// across every gateway sharing the database, so two first stores of an id
/// cannot both insert without a unique index on the id column.
fn store_lock_key(id: &str) -> i64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let hash = id.bytes().fold(OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    });
    hash as i64
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
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&response_id.0];

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

        let mut client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let store_err =
            |e: tokio_postgres::Error| ResponseStorageError::StorageError(describe_error(&e));
        // The statement is prepared once per pooled connection (the cache
        // keeps it), so a store on the upsert path is one round trip.
        let rows = match self.store_path {
            StorePath::Upsert => {
                let stmt = client
                    .prepare_cached(&upsert_sql(&table, &col_names))
                    .await
                    .map_err(store_err)?;
                client.execute(&stmt, &params).await.map_err(store_err)?
            }
            StorePath::LockedMerge => {
                // One READ COMMITTED transaction: the statement takes its
                // snapshot once the lock is held, so it sees the row of every
                // store of this id that finished before it, and inserts only
                // when there is none.
                let tx = client
                    .build_transaction()
                    .isolation_level(IsolationLevel::ReadCommitted)
                    .start()
                    .await
                    .map_err(store_err)?;
                tx.batch_execute(&lock_sql(store_lock_key(&response_id.0)))
                    .await
                    .map_err(store_err)?;
                let stmt = tx
                    .prepare_cached(&locked_merge_sql(&table, &col_names))
                    .await
                    .map_err(store_err)?;
                let inserted = tx.execute(&stmt, &params).await.map_err(store_err)?;
                tx.commit().await.map_err(store_err)?;
                inserted
            }
        };
        tracing::debug!(
            path = ?self.store_path,
            rows,
            "Response stored in Postgres"
        );
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
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let rows = query_cached(&client, &sql, &[&response_id.0.as_str()])
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        if rows.is_empty() {
            return Ok(None);
        }
        Self::build_response_from_row(&rows[0], &self.store.schema).map(Some)
    }

    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        let Some(sql) = find_by_output_item_sql(&self.store.schema, &self.select_base) else {
            return Ok(None);
        };

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let rows = query_cached(&client, &sql, &[&item_id])
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        match rows.first() {
            Some(row) => Self::build_response_from_row(row, &self.store.schema).map(Some),
            None => Ok(None),
        }
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
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        execute_cached(
            &client,
            &format!("DELETE FROM {table} WHERE {col_id} = $1"),
            &[&response_id.0.as_str()],
        )
        .await
        .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
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
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let rows = if let Some(l) = limit {
            let l_i64: i64 = l as i64;
            let sql = format!(
                "{} WHERE {col_safety} = $1{order_clause} LIMIT $2",
                self.select_base,
            );
            query_cached(&client, &sql, &[&identifier, &l_i64])
                .await
                .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?
        } else {
            let sql = format!("{} WHERE {col_safety} = $1{order_clause}", self.select_base,);
            query_cached(&client, &sql, &[&identifier])
                .await
                .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?
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
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
        let rows_deleted = execute_cached(&client, &sql, &[&identifier])
            .await
            .map_err(|e| ResponseStorageError::StorageError(describe_error(&e)))?;
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
        assert!(
            PostgresResponseStorage::table_ddl(&schema).contains("safety_identifier VARCHAR(128)")
        );
        assert_eq!(
            PostgresResponseStorage::index_ddl(&schema).as_deref(),
            Some(
                "CREATE INDEX IF NOT EXISTS responses_safety_idx ON responses (safety_identifier);"
            )
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
    fn store_lock_key_is_stable_and_tells_ids_apart() {
        assert_eq!(store_lock_key("resp_1"), store_lock_key("resp_1"));
        assert_ne!(store_lock_key("resp_1"), store_lock_key("resp_2"));
        // Pinned FNV-1a 64 values: gateways of different builds share one
        // database, so the key of an id must never change.
        assert_eq!(store_lock_key(""), 0xcbf2_9ce4_8422_2325_u64 as i64);
        assert_eq!(store_lock_key("a"), 0xaf63_dc4c_8601_ec8c_u64 as i64);
    }
}
