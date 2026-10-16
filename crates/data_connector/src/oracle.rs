//! Oracle storage implementation using OracleStore helper.

use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use oracle::{sql_type::ToSql, Connection, Connector, Row};
use serde_json::Value;

use super::core::{
    make_item_id, sql_string_list, ConditionalStore, Conversation, ConversationId,
    ConversationItem, ConversationItemId, ConversationItemStorage, ConversationItemStorageError,
    ConversationMetadata, ConversationStorage, ConversationStorageError, Failure, ListParams,
    NewConversation, NewConversationItem, ResponseId, ResponseStorage, ResponseStorageError,
    SortOrder, StoredResponse, TERMINAL_RESPONSE_STATUSES,
};
use crate::{
    common::{
        build_response_select_base, extra_column_defs, parse_json_value, parse_raw_response,
        resolve_extra_column_values,
    },
    config::OracleConfig,
    context::current_extra_columns,
    oracle_migrations::ORACLE_HISTORY_MIGRATIONS,
    schema::SchemaConfig,
    versioning::{
        bootstrap_oracle_schema, oracle_recorded_version, run_oracle_migrations, Migration,
        OracleSchemaSteps,
    },
};

// ============================================================================
// PART 1: OracleStore Helper + Common Utilities
// ============================================================================

/// Schema initializer function signature for Oracle storage backends.
pub(crate) type SchemaInitFn = fn(&Connection, &SchemaConfig) -> Result<(), String>;

/// Shared Oracle connection pool infrastructure.
///
/// This helper eliminates ~540 LOC of duplication across storage implementations.
/// It handles connection pooling, error mapping, and client configuration.
pub(crate) struct OracleStore {
    pool: Pool<OracleConnectionManager>,
    pub(crate) schema: Arc<SchemaConfig>,
}

impl OracleStore {
    /// Create a connection pool, with the schema brought to this build's
    /// shape first.
    ///
    /// Accepts a list of schema initializers that run on a single connection
    /// before the pool is created, ensuring all tables exist. The database's
    /// recorded schema version is checked against this build's migrations
    /// before any of them runs, so a database a newer build migrated forward
    /// is refused untouched (see [`bootstrap_oracle_schema`]).
    pub fn new(config: &OracleConfig, init_schemas: &[SchemaInitFn]) -> Result<Self, String> {
        // Extract and validate schema config.
        // Oracle folds unquoted identifiers to uppercase, so existing tables
        // and columns are CONVERSATIONS, CONV_ID, etc.  Uppercase all
        // configured names so that quoted references match reality.
        let mut schema = config.schema.clone().unwrap_or_default();
        schema.uppercase_for_oracle();
        schema.validate()?;
        let schema = Arc::new(schema);

        // Configure Oracle client (wallet env vars, etc.)
        configure_oracle_env(config)?;

        // Initialize schemas using a single connection
        let conn = connect_oracle(
            config.external_auth,
            &config.username,
            &config.password,
            &config.connect_descriptor,
        )
        .map_err(map_oracle_error)?;

        // The recorded schema version against this build's migrations before
        // any history table is touched; then the tables, the migrations and
        // the indexes on the columns they added.
        bootstrap_oracle_schema(
            &OracleSchemaSetup {
                conn: &conn,
                schema: schema.as_ref(),
                init_schemas,
            },
            &ORACLE_HISTORY_MIGRATIONS,
            schema.version,
        )?;

        drop(conn);

        // Create connection pool
        let config_arc = Arc::new(config.clone());
        let manager = OracleConnectionManager {
            params: Arc::new(OracleConnectParams::from_config(&config_arc)),
        };

        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder
            .build()
            .map_err(|e| format!("Failed to build Oracle pool: {e}"))?;

        Ok(Self { pool, schema })
    }

    /// Execute function with a connection from the pool
    /// Runs `func` on a pooled connection. A connection the pool cannot give
    /// is the store being unavailable; what `func` answers is the store's
    /// own failure (classifying the statement's errors further is left to
    /// the caller's text).
    pub async fn execute<F, T>(&self, func: F) -> Result<T, Failure>
    where
        F: FnOnce(&Connection) -> Result<T, String> + Send + 'static,
        T: Send + 'static,
    {
        let connection = self.pool.get().await.map_err(|e| Failure::pool(&e))?;

        tokio::task::spawn_blocking(move || {
            let result = func(&connection);
            drop(connection);
            result
        })
        .await
        .map_err(|e| Failure::Storage(format!("Task execution failed: {e}")))?
        .map_err(Failure::Storage)
    }
}

/// The start-up steps of [`OracleStore::new`] on its connection, in the order
/// [`bootstrap_oracle_schema`] runs them.
struct OracleSchemaSetup<'a> {
    conn: &'a Connection,
    schema: &'a SchemaConfig,
    init_schemas: &'a [SchemaInitFn],
}

impl OracleSchemaSteps for OracleSchemaSetup<'_> {
    fn recorded_version(&self) -> Result<Option<u32>, String> {
        oracle_recorded_version(self.conn, self.schema)
    }

    fn create_tables(&self) -> Result<(), String> {
        for init_schema in self.init_schemas {
            init_schema(self.conn, self.schema)?;
        }
        Ok(())
    }

    fn migrate(
        &self,
        migrations: &[Migration],
        config_version: Option<u32>,
    ) -> Result<Vec<u32>, String> {
        run_oracle_migrations(
            self.conn,
            self.schema,
            migrations,
            config_version,
            self.schema.auto_migrate,
        )
    }
}

impl Clone for OracleStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            schema: self.schema.clone(),
        }
    }
}

// Error mapping helper
pub(crate) fn map_oracle_error(err: oracle::Error) -> String {
    if let Some(db_err) = err.db_error() {
        format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        )
    } else {
        err.to_string()
    }
}

/// Validate Oracle wallet path and set `TNS_ADMIN` environment variable.
///
/// # Thread-safety note
///
/// `std::env::set_var` is not thread-safe and is marked unsafe in Rust 2024 edition.
/// This function is called once during `OracleStore::new()` initialization, before the
/// connection pool is created and before any worker threads are spawned.  When migrating
/// to edition 2024, this call will need to be wrapped in `unsafe` (requires removing the
/// workspace `unsafe_code = "deny"` lint or adding a targeted `#[expect]`), or the
/// environment variable should be set by the process launcher instead.
pub(crate) fn configure_oracle_env(config: &OracleConfig) -> Result<(), String> {
    if let Some(wallet_path) = &config.wallet_path {
        let path = Path::new(wallet_path);

        if !path.is_dir() {
            return Err(format!(
                "Oracle wallet path '{wallet_path}' is not a directory"
            ));
        }

        if !path.join("tnsnames.ora").exists() && !path.join("sqlnet.ora").exists() {
            return Err(format!(
                "Oracle wallet path '{wallet_path}' is missing tnsnames.ora or sqlnet.ora"
            ));
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

// Connection parameters
#[derive(Clone)]
pub(crate) struct OracleConnectParams {
    pub username: String,
    pub password: String,
    pub connect_descriptor: String,
    pub external_auth: bool,
}

impl OracleConnectParams {
    pub fn from_config(config: &OracleConfig) -> Self {
        Self {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
            external_auth: config.external_auth,
        }
    }
}

impl std::fmt::Debug for OracleConnectParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectParams")
            .field("username", &self.username)
            .field("connect_descriptor", &self.connect_descriptor)
            .field("external_auth", &self.external_auth)
            .finish()
    }
}

// Connection manager (same for all stores)
#[derive(Clone)]
struct OracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

impl std::fmt::Debug for OracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for OracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = Result<Connection, oracle::Error>> + Send {
        let params = self.params.clone();
        async move {
            let mut conn = connect_oracle(
                params.external_auth,
                &params.username,
                &params.password,
                &params.connect_descriptor,
            )?;
            conn.set_autocommit(true);
            Ok(conn)
        }
    }

    #[expect(clippy::manual_async_fn)]
    fn recycle(
        &self,
        conn: &mut Connection,
        _: &Metrics,
    ) -> impl std::future::Future<Output = RecycleResult<Self::Error>> + Send {
        async move { conn.ping().map_err(RecycleError::Backend) }
    }
}

fn connect_oracle(
    external_auth: bool,
    username: &str,
    password: &str,
    connect_descriptor: &str,
) -> Result<Connection, oracle::Error> {
    if external_auth {
        Connector::new("", "", connect_descriptor)
            .external_auth(true)
            .connect()
    } else {
        Connection::connect(username, password, connect_descriptor)
    }
}

// ============================================================================
// PART 2: OracleConversationStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct OracleConversationStorage {
    store: OracleStore,
}

impl OracleConversationStorage {
    pub fn new(store: OracleStore) -> Self {
        Self { store }
    }

    pub(crate) fn init_schema(conn: &Connection, schema: &SchemaConfig) -> Result<(), String> {
        let s = &schema.conversations;
        // Table and column names are already uppercased by OracleStore::new().
        let table = s.qualified_table(schema.owner.as_deref());

        let exists: i64 = conn
            .query_row_as(
                &format!(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = '{}'",
                    s.table
                ),
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists == 0 {
            let mut col_defs = vec![format!("{} VARCHAR2(64) PRIMARY KEY", s.col("id"))];
            if !s.is_skipped("created_at") {
                col_defs.push(format!("{} TIMESTAMP WITH TIME ZONE", s.col("created_at")));
            }
            if !s.is_skipped("metadata") {
                col_defs.push(format!("{} CLOB", s.col("metadata")));
            }
            if !s.is_skipped("tenant") {
                col_defs.push(format!("{} VARCHAR2(255)", s.col("tenant")));
            }
            col_defs.extend(extra_column_defs(s));

            conn.execute(
                &format!("CREATE TABLE {table} ({})", col_defs.join(", ")),
                &[],
            )
            .map_err(map_oracle_error)?;
        }

        Ok(())
    }

    fn parse_metadata(
        raw: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        crate::common::parse_conversation_metadata(raw)
            .map_err(ConversationStorageError::StorageError)
    }
}

#[async_trait]
impl ConversationStorage for OracleConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.clone();
        let created_at = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let tenant = conversation.tenant.clone();
        let schema = self.store.schema.clone();
        // Capture extra columns before spawn_blocking (task-locals don't propagate)
        let hook_extra = current_extra_columns().unwrap_or_default();

        self.store
            .execute(move |conn| {
                let s = &schema.conversations;
                let table = s.qualified_table(schema.owner.as_deref());

                let mut columns: Vec<&str> = vec![s.col("id")];
                let mut params: Vec<&dyn ToSql> = vec![&id_str];
                if !s.is_skipped("created_at") {
                    columns.push(s.col("created_at"));
                    params.push(&created_at);
                }
                if !s.is_skipped("metadata") {
                    columns.push(s.col("metadata"));
                    params.push(&metadata_json);
                }
                if !s.is_skipped("tenant") {
                    columns.push(s.col("tenant"));
                    params.push(&tenant);
                }

                // Append extra columns from hooks or defaults
                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(s, &hook_extra);
                for (name, val) in &extra_cols {
                    columns.push(*name);
                    params.push(val);
                }

                let placeholders: Vec<String> =
                    (1..=params.len()).map(|i| format!(":{i}")).collect();
                let sql = format!(
                    "INSERT INTO {table} ({}) VALUES ({})",
                    columns.join(", "),
                    placeholders.join(", ")
                );
                conn.execute(&sql, &params[..])
                    .map(|_| ())
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::from_failure)?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let lookup = id.0.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let s = &schema.conversations;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_id = s.col("id");

                let mut select_cols = vec![col_id.to_string()];
                if !s.is_skipped("created_at") {
                    select_cols.push(s.col("created_at").to_string());
                }
                if !s.is_skipped("metadata") {
                    select_cols.push(s.col("metadata").to_string());
                }
                if !s.is_skipped("tenant") {
                    select_cols.push(s.col("tenant").to_string());
                }

                let sql = format!(
                    "SELECT {} FROM {table} WHERE {col_id} = :1",
                    select_cols.join(", ")
                );
                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&lookup]).map_err(map_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_oracle_error)?;
                    let id: String = row.get(s.col("id")).map_err(map_oracle_error)?;
                    let created_at: DateTime<Utc> = if s.is_skipped("created_at") {
                        Utc::now()
                    } else {
                        row.get(s.col("created_at")).map_err(map_oracle_error)?
                    };
                    let metadata_raw: Option<String> = if s.is_skipped("metadata") {
                        None
                    } else {
                        row.get(s.col("metadata")).map_err(map_oracle_error)?
                    };
                    let metadata = Self::parse_metadata(metadata_raw).map_err(|e| e.to_string())?;
                    let tenant: Option<String> = if s.is_skipped("tenant") {
                        None
                    } else {
                        row.get(s.col("tenant")).map_err(map_oracle_error)?
                    };
                    Ok(Some(Conversation::with_parts(
                        ConversationId(id),
                        created_at,
                        metadata,
                        tenant,
                    )))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(ConversationStorageError::from_failure)
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        // The row as stored (its creation time and its tenant come back as
        // they are), then the one column the update moves.
        let Some(current) = self.get_conversation(id).await? else {
            return Ok(None);
        };
        if self.store.schema.conversations.is_skipped("metadata") {
            return Ok(Some(Conversation {
                metadata,
                ..current
            }));
        }
        let id_str = id.0.clone();
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;
        let schema = self.store.schema.clone();

        let rows = self
            .store
            .execute(move |conn| {
                let s = &schema.conversations;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_id = s.col("id");
                let col_meta = s.col("metadata");
                let sql = format!("UPDATE {table} SET {col_meta} = :1 WHERE {col_id} = :2");
                let stmt = conn
                    .execute(&sql, &[&metadata_json, &id_str])
                    .map_err(map_oracle_error)?;
                stmt.row_count().map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::from_failure)?;
        if rows == 0 {
            return Ok(None);
        }
        Ok(Some(Conversation {
            metadata,
            ..current
        }))
    }

    async fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<bool, ConversationStorageError> {
        let id_str = id.0.clone();
        let schema = self.store.schema.clone();

        let res = self
            .store
            .execute(move |conn| {
                let s = &schema.conversations;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_id = s.col("id");

                conn.execute(
                    &format!("DELETE FROM {table} WHERE {col_id} = :1"),
                    &[&id_str],
                )
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::from_failure)?;

        Ok(res
            .row_count()
            .map_err(|e| ConversationStorageError::StorageError(map_oracle_error(e)))?
            > 0)
    }
}

// ============================================================================
// PART 3: OracleConversationItemStorage
// ============================================================================

#[derive(Clone)]
pub(super) struct OracleConversationItemStorage {
    store: OracleStore,
}

impl OracleConversationItemStorage {
    pub fn new(store: OracleStore) -> Self {
        Self { store }
    }

    pub(crate) fn init_schema(conn: &Connection, schema: &SchemaConfig) -> Result<(), String> {
        let si = &schema.conversation_items;
        // Table and column names are already uppercased by OracleStore::new().
        let si_table = si.qualified_table(schema.owner.as_deref());

        let exists_items: i64 = conn
            .query_row_as(
                &format!(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = '{}'",
                    si.table
                ),
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists_items == 0 {
            let mut col_defs = vec![format!("{} VARCHAR2(64) PRIMARY KEY", si.col("id"))];
            let item_core_cols: [(&str, &str); 7] = [
                ("response_id", "VARCHAR2(64)"),
                ("item_type", "VARCHAR2(32) NOT NULL"),
                ("role", "VARCHAR2(32)"),
                ("content", "CLOB"),
                ("status", "VARCHAR2(32)"),
                ("created_at", "TIMESTAMP WITH TIME ZONE"),
                ("tenant", "VARCHAR2(255)"),
            ];
            for (logical, sql_type) in &item_core_cols {
                if !si.is_skipped(logical) {
                    col_defs.push(format!("{} {sql_type}", si.col(logical)));
                }
            }
            col_defs.extend(extra_column_defs(si));

            conn.execute(
                &format!("CREATE TABLE {si_table} ({})", col_defs.join(", ")),
                &[],
            )
            .map_err(map_oracle_error)?;
        }

        let sl = &schema.conversation_item_links;
        let sl_table = sl.qualified_table(schema.owner.as_deref());

        let exists_links: i64 = conn
            .query_row_as(
                &format!(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = '{}'",
                    sl.table
                ),
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists_links == 0 {
            let col_cid = sl.col("conversation_id");
            let col_iid = sl.col("item_id");
            let col_added = sl.col("added_at");

            let pk_name = format!("PK_{}", sl.table);
            let idx_name = format!("{}_CONV_IDX", sl.table);

            let mut col_defs = vec![
                format!("{col_cid} VARCHAR2(64) NOT NULL"),
                format!("{col_iid} VARCHAR2(64) NOT NULL"),
                format!("{col_added} TIMESTAMP WITH TIME ZONE"),
            ];
            col_defs.extend(extra_column_defs(sl));
            col_defs.push(format!(
                "CONSTRAINT {pk_name} PRIMARY KEY ({col_cid}, {col_iid})"
            ));

            conn.execute(
                &format!("CREATE TABLE {sl_table} ({})", col_defs.join(", ")),
                &[],
            )
            .map_err(map_oracle_error)?;

            conn.execute(
                &format!("CREATE INDEX {idx_name} ON {sl_table} ({col_cid}, {col_added})"),
                &[],
            )
            .map_err(map_oracle_error)?;
        }

        Ok(())
    }
}

#[async_trait]
impl ConversationItemStorage for OracleConversationItemStorage {
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
            tenant,
        } = item;
        let id = opt_id.unwrap_or_else(|| make_item_id(&item_type));
        let created_at = Utc::now();
        let content_json = serde_json::to_string(&content)?;

        let id_str = id.0.clone();
        let cl_response_id = response_id.clone();
        let cl_item_type = item_type.clone();
        let cl_role = role.clone();
        let cl_status = status.clone();
        let cl_tenant = tenant.clone();
        let schema = self.store.schema.clone();
        let hook_extra = current_extra_columns().unwrap_or_default();

        self.store
            .execute(move |conn| {
                let si = &schema.conversation_items;
                let table = si.qualified_table(schema.owner.as_deref());

                let mut columns: Vec<&str> = vec![si.col("id")];
                let mut params: Vec<&dyn ToSql> = vec![&id_str];
                if !si.is_skipped("response_id") {
                    columns.push(si.col("response_id"));
                    params.push(&cl_response_id);
                }
                if !si.is_skipped("item_type") {
                    columns.push(si.col("item_type"));
                    params.push(&cl_item_type);
                }
                if !si.is_skipped("role") {
                    columns.push(si.col("role"));
                    params.push(&cl_role);
                }
                if !si.is_skipped("content") {
                    columns.push(si.col("content"));
                    params.push(&content_json);
                }
                if !si.is_skipped("status") {
                    columns.push(si.col("status"));
                    params.push(&cl_status);
                }
                if !si.is_skipped("created_at") {
                    columns.push(si.col("created_at"));
                    params.push(&created_at);
                }
                if !si.is_skipped("tenant") {
                    columns.push(si.col("tenant"));
                    params.push(&cl_tenant);
                }

                // Append extra columns from hooks or defaults
                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(si, &hook_extra);
                for (name, val) in &extra_cols {
                    columns.push(*name);
                    params.push(val);
                }

                let placeholders: Vec<String> =
                    (1..=params.len()).map(|i| format!(":{i}")).collect();
                let sql = format!(
                    "INSERT INTO {table} ({}) VALUES ({})",
                    columns.join(", "),
                    placeholders.join(", ")
                );
                conn.execute(&sql, &params[..]).map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)?;

        Ok(ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
            tenant,
        })
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> Result<(), ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let schema = self.store.schema.clone();
        let hook_extra = current_extra_columns().unwrap_or_default();

        self.store
            .execute(move |conn| {
                let sl = &schema.conversation_item_links;
                let table = sl.qualified_table(schema.owner.as_deref());

                let mut columns: Vec<&str> = vec![
                    sl.col("conversation_id"),
                    sl.col("item_id"),
                    sl.col("added_at"),
                ];
                let mut params: Vec<&dyn ToSql> = vec![&cid, &iid, &added_at];

                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(sl, &hook_extra);
                for (name, val) in &extra_cols {
                    columns.push(*name);
                    params.push(val);
                }

                let (sql, binds) = link_sql(&table, &columns);
                let named: Vec<(&str, &dyn ToSql)> = binds
                    .iter()
                    .map(String::as_str)
                    .zip(params.iter().copied())
                    .collect();
                conn.execute_named(&sql, &named).map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }

    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> Result<(), ConversationItemStorageError> {
        if items.is_empty() {
            return Ok(());
        }

        let cid = conversation_id.0.clone();
        let schema = self.store.schema.clone();
        let hook_extra = current_extra_columns().unwrap_or_default();
        // One source row per item: a repeated id keeps its last stamp, as the
        // other backends' per-item links leave it (two rows under one key
        // would fail the MERGE).
        let owned_items =
            last_stamp_per_item(items.iter().map(|(id, ts)| (id.0.clone(), *ts)).collect());

        self.store
            .execute(move |conn| {
                let sl = &schema.conversation_item_links;
                let table = sl.qualified_table(schema.owner.as_deref());

                let mut columns: Vec<&str> = vec![
                    sl.col("conversation_id"),
                    sl.col("item_id"),
                    sl.col("added_at"),
                ];

                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(sl, &hook_extra);
                for (name, _) in &extra_cols {
                    columns.push(*name);
                }

                let sql = link_items_sql(&table, &columns, owned_items.len());

                let mut params_vec: Vec<(&str, &dyn ToSql)> = vec![("cid", &cid)];

                // We need stable references, so collect param names
                let iid_names: Vec<String> =
                    (0..owned_items.len()).map(|i| format!("iid{i}")).collect();
                let ts_names: Vec<String> =
                    (0..owned_items.len()).map(|i| format!("ts{i}")).collect();
                let ex_names: Vec<String> =
                    (0..extra_cols.len()).map(|i| format!("ex{i}")).collect();

                for (idx, (iid, ts)) in owned_items.iter().enumerate() {
                    params_vec.push((&iid_names[idx], iid));
                    params_vec.push((&ts_names[idx], ts));
                }
                for (ei, (_, val)) in extra_cols.iter().enumerate() {
                    params_vec.push((&ex_names[ei], val));
                }

                conn.execute_named(&sql, &params_vec)
                    .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> Result<Vec<ConversationItem>, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);
        let after_id = params.after.clone();
        let schema = self.store.schema.clone();

        // Single query with optional LEFT JOIN to resolve cursor in one round trip
        self.store
            .execute({
                move |conn| {
                    let si = &schema.conversation_items;
                    let sl = &schema.conversation_item_links;
                    let si_table = si.qualified_table(schema.owner.as_deref());
                    let sl_table = sl.qualified_table(schema.owner.as_deref());
                    let si_col_id = si.col("id");
                    let sl_col_cid = sl.col("conversation_id");
                    let sl_col_iid = sl.col("item_id");
                    let sl_col_added = sl.col("added_at");

                    // Build SELECT column list dynamically, respecting skip_columns
                    let mut select_cols = vec![format!("i.{si_col_id}")];
                    for field in &[
                        "response_id",
                        "item_type",
                        "role",
                        "content",
                        "status",
                        "created_at",
                        "tenant",
                    ] {
                        if !si.is_skipped(field) {
                            select_cols.push(format!("i.{}", si.col(field)));
                        }
                    }

                    let mut sql = format!(
                        "SELECT {} FROM {sl_table} l \
                         JOIN {si_table} i ON i.{si_col_id} = l.{sl_col_iid}",
                        select_cols.join(", "),
                    );

                    if after_id.is_some() {
                        // LEFT JOIN to the cursor row to resolve its added_at inline
                        sql.push_str(&format!(
                            " LEFT JOIN {sl_table} c \
                             ON c.{sl_col_cid} = :cid AND c.{sl_col_iid} = :after_id"
                        ));
                    }

                    sql.push_str(&format!(" WHERE l.{sl_col_cid} = :cid"));

                    if after_id.is_some() {
                        // c.item_id IS NULL handles "cursor not found" → returns all items
                        if order_desc {
                            sql.push_str(&format!(
                                " AND (c.{sl_col_iid} IS NULL \
                                 OR l.{sl_col_added} < c.{sl_col_added} \
                                 OR (l.{sl_col_added} = c.{sl_col_added} AND l.{sl_col_iid} < c.{sl_col_iid}))"
                            ));
                        } else {
                            sql.push_str(&format!(
                                " AND (c.{sl_col_iid} IS NULL \
                                 OR l.{sl_col_added} > c.{sl_col_added} \
                                 OR (l.{sl_col_added} = c.{sl_col_added} AND l.{sl_col_iid} > c.{sl_col_iid}))"
                            ));
                        }
                    }

                    if order_desc {
                        sql.push_str(&format!(
                            " ORDER BY l.{sl_col_added} DESC, l.{sl_col_iid} DESC"
                        ));
                    } else {
                        sql.push_str(&format!(
                            " ORDER BY l.{sl_col_added} ASC, l.{sl_col_iid} ASC"
                        ));
                    }
                    sql.push_str(" FETCH NEXT :limit ROWS ONLY");

                    let mut params_vec: Vec<(&str, &dyn ToSql)> = vec![("cid", &cid)];
                    if let Some(ref aid) = after_id {
                        params_vec.push(("after_id", aid));
                    }
                    params_vec.push(("limit", &limit));

                    let rows_iter = conn
                        .query_named(&sql, &params_vec)
                        .map_err(map_oracle_error)?;

                    let mut items = Vec::new();
                    for row_res in rows_iter {
                        let row = row_res.map_err(map_oracle_error)?;
                        items.push(build_item_from_oracle_row(&row, si)?);
                    }
                    Ok(items)
                }
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> Result<Option<ConversationItem>, ConversationItemStorageError> {
        let iid = item_id.0.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let si = &schema.conversation_items;
                let table = si.qualified_table(schema.owner.as_deref());
                let col_id = si.col("id");

                let mut select_cols = vec![col_id.to_string()];
                for field in &[
                    "response_id",
                    "item_type",
                    "role",
                    "content",
                    "status",
                    "created_at",
                    "tenant",
                ] {
                    if !si.is_skipped(field) {
                        select_cols.push(si.col(field).to_string());
                    }
                }

                let sql = format!(
                    "SELECT {} FROM {table} WHERE {col_id} = :1",
                    select_cols.join(", "),
                );
                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&iid]).map_err(map_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_oracle_error)?;
                    Ok(Some(build_item_from_oracle_row(&row, si)?))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<bool, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let sl = &schema.conversation_item_links;
                let table = sl.qualified_table(schema.owner.as_deref());
                let col_cid = sl.col("conversation_id");
                let col_iid = sl.col("item_id");

                let sql =
                    format!("SELECT COUNT(*) FROM {table} WHERE {col_cid} = :1 AND {col_iid} = :2");
                let count: i64 = conn
                    .query_row_as(&sql, &[&cid, &iid])
                    .map_err(map_oracle_error)?;
                Ok(count > 0)
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<(), ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let sl = &schema.conversation_item_links;
                let table = sl.qualified_table(schema.owner.as_deref());
                let col_cid = sl.col("conversation_id");
                let col_iid = sl.col("item_id");

                conn.execute(
                    &format!("DELETE FROM {table} WHERE {col_cid} = :1 AND {col_iid} = :2"),
                    &[&cid, &iid],
                )
                .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::from_failure)
    }
}

// ============================================================================
// PART 4: OracleResponseStorage
// ============================================================================

/// Parse a single `oracle::Row` into a `ConversationItem`, respecting
/// the `is_skipped` guards configured in `TableConfig`.
fn build_item_from_oracle_row(
    row: &Row,
    si: &crate::schema::TableConfig,
) -> Result<ConversationItem, String> {
    let id: String = row.get(si.col("id")).map_err(map_oracle_error)?;
    let response_id: Option<String> = if si.is_skipped("response_id") {
        None
    } else {
        row.get(si.col("response_id")).map_err(map_oracle_error)?
    };
    let item_type: String = if si.is_skipped("item_type") {
        String::new()
    } else {
        row.get(si.col("item_type")).map_err(map_oracle_error)?
    };
    let role: Option<String> = if si.is_skipped("role") {
        None
    } else {
        row.get(si.col("role")).map_err(map_oracle_error)?
    };
    let content: Value = if si.is_skipped("content") {
        Value::Null
    } else {
        let raw: Option<String> = row.get(si.col("content")).map_err(map_oracle_error)?;
        match raw {
            Some(s) => serde_json::from_str(&s).map_err(|e| e.to_string())?,
            None => Value::Null,
        }
    };
    let status: Option<String> = if si.is_skipped("status") {
        None
    } else {
        row.get(si.col("status")).map_err(map_oracle_error)?
    };
    let created_at: DateTime<Utc> = if si.is_skipped("created_at") {
        Utc::now()
    } else {
        row.get(si.col("created_at")).map_err(map_oracle_error)?
    };
    let tenant: Option<String> = if si.is_skipped("tenant") {
        None
    } else {
        row.get(si.col("tenant")).map_err(map_oracle_error)?
    };

    Ok(ConversationItem {
        id: ConversationItemId(id),
        response_id,
        item_type,
        role,
        content,
        status,
        created_at,
        tenant,
    })
}

#[derive(Clone)]
pub(super) struct OracleResponseStorage {
    store: OracleStore,
    select_base: String,
}

/// The query behind `find_response_by_output_item`: `raw_response` holds the
/// response as JSON text, so `JSON_EXISTS` asks whether its `output` array has
/// an item whose id equals the bind variable `:1` (passed into the path as
/// `$iid`, never spliced into it). `None` when the schema skips the column.
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
        "SELECT * FROM ({select_base} WHERE JSON_EXISTS({col_raw}, \
         '$.output[*]?(@.id == $iid)' PASSING :1 AS \"iid\"){order}) WHERE ROWNUM <= 1"
    ))
}

impl OracleResponseStorage {
    pub fn new(store: OracleStore) -> Self {
        let select_base = build_response_select_base(&store.schema);
        Self { store, select_base }
    }

    pub(crate) fn init_schema(conn: &Connection, schema: &SchemaConfig) -> Result<(), String> {
        let s = &schema.responses;
        // Table and column names are already uppercased by OracleStore::new().
        let table = s.qualified_table(schema.owner.as_deref());

        let exists: i64 = conn
            .query_row_as(
                &format!(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = '{}'",
                    s.table
                ),
                &[],
            )
            .map_err(map_oracle_error)?;

        if exists == 0 {
            let mut col_defs = vec![format!("{} VARCHAR2(64) PRIMARY KEY", s.col("id"))];
            let core_cols: [(&str, &str); 8] = [
                ("conversation_id", "VARCHAR2(64)"),
                ("previous_response_id", "VARCHAR2(64)"),
                ("input", "CLOB"),
                ("created_at", "TIMESTAMP WITH TIME ZONE"),
                ("safety_identifier", "VARCHAR2(128)"),
                ("model", "VARCHAR2(128)"),
                ("raw_response", "CLOB"),
                ("tenant", "VARCHAR2(255)"),
            ];
            for (logical, sql_type) in &core_cols {
                if !s.is_skipped(logical) {
                    col_defs.push(format!("{} {sql_type}", s.col(logical)));
                }
            }
            col_defs.extend(extra_column_defs(s));

            conn.execute(
                &format!("CREATE TABLE {table} ({})", col_defs.join(", ")),
                &[],
            )
            .map_err(map_oracle_error)?;

            // The index behind `delete_unstored_older_than`: the object's status
            // over the unstored records only (NULL, hence absent, for the rest).
            // Built with the table, while it is empty; an existing table gets it
            // from its operator (see the release note), and a build that fails
            // (a busy object, an identifier too long for the release, no
            // temporary space) leaves the sweep a scan rather than failing the
            // start, as on Postgres.
            if !s.is_skipped("raw_response") {
                let col_raw = s.col("raw_response");
                let unstored_idx = format!("{}_UNSTORED_IDX", s.table);
                if let Err(e) = create_index_if_missing(
                    conn,
                    &s.table,
                    &unstored_idx,
                    &format!(
                        "CREATE INDEX {unstored_idx} ON {table}({})",
                        unstored_status_expr(col_raw)
                    ),
                ) {
                    tracing::warn!(
                        "unstored response index not created (the sweep of unstored background records scans the table): {e}"
                    );
                }
            }
        }

        if !s.is_skipped("previous_response_id") {
            let prev = s.col("previous_response_id");
            let prev_idx = format!("{}_PREV_IDX", s.table);
            create_index_if_missing(
                conn,
                &s.table,
                &prev_idx,
                &format!("CREATE INDEX {prev_idx} ON {table}({prev})"),
            )?;
        }

        if !s.is_skipped("safety_identifier") {
            let safety = s.col("safety_identifier");
            let user_idx = format!("{}_USER_IDX", s.table);
            create_index_if_missing(
                conn,
                &s.table,
                &user_idx,
                &format!("CREATE INDEX {user_idx} ON {table}({safety})"),
            )?;
        }

        Ok(())
    }

    fn build_response_from_row(row: &Row, schema: &SchemaConfig) -> Result<StoredResponse, String> {
        let s = &schema.responses;
        let col_id = s.col("id");
        let col_created = s.col("created_at");

        let id: String = row.get(col_id).map_err(map_oracle_error)?;
        let created_at: DateTime<Utc> = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            row.get(col_created).map_err(map_oracle_error)?
        };

        let previous: Option<String> = if s.is_skipped("previous_response_id") {
            None
        } else {
            row.get(s.col("previous_response_id"))
                .map_err(map_oracle_error)?
        };
        let input_json: Option<String> = if s.is_skipped("input") {
            None
        } else {
            row.get(s.col("input")).map_err(map_oracle_error)?
        };
        let safety_identifier: Option<String> = if s.is_skipped("safety_identifier") {
            None
        } else {
            row.get(s.col("safety_identifier"))
                .map_err(map_oracle_error)?
        };
        let model: Option<String> = if s.is_skipped("model") {
            None
        } else {
            row.get(s.col("model")).map_err(map_oracle_error)?
        };
        let conversation_id: Option<String> = if s.is_skipped("conversation_id") {
            None
        } else {
            row.get(s.col("conversation_id"))
                .map_err(map_oracle_error)?
        };
        let raw_response_json: Option<String> = if s.is_skipped("raw_response") {
            None
        } else {
            row.get(s.col("raw_response")).map_err(map_oracle_error)?
        };
        let tenant: Option<String> = if s.is_skipped("tenant") {
            None
        } else {
            row.get(s.col("tenant")).map_err(map_oracle_error)?
        };

        let previous_response_id = previous.map(ResponseId);
        let raw_response = parse_raw_response(raw_response_json)?;
        let input = parse_json_value(input_json)?;

        Ok(StoredResponse {
            id: ResponseId(id),
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
            tenant,
        })
    }
}

/// The condition of the tenant scope on the rows of a query, with the
/// caller's tenant bound as `:bind`: the rows stored under no tenant and
/// those stored under the caller's (see
/// [`tenant_scope_allows`](crate::core::tenant_scope_allows)). `None` when
/// the schema skips the column: every row is served then, as before.
fn tenant_scope_sql(s: &crate::schema::TableConfig, bind: usize) -> Option<String> {
    if s.is_skipped("tenant") {
        return None;
    }
    let col = s.col("tenant");
    Some(format!(" AND ({col} IS NULL OR {col} = :{bind})"))
}

/// The statement that stores a response, with the names of its binds in the
/// columns' order: a merge on the id (the first column, the table's primary
/// key) that inserts a new row or updates every other column of the
/// existing one, so a record written as `queued` is replaced by its terminal
/// object under the one id. Only the id is bound inside the USING source;
/// the values are bound in UPDATE SET and INSERT VALUES, where a string
/// above 4,000 bytes (the JSON of `input` or `raw_response`) is accepted
/// for a CLOB column as it is in an INSERT, while the same bind in a
/// subquery's select list fails with ORA-01461. Named binds, so the id's
/// two occurrences are bound once. The MERGE needs no unique index on the
/// id: its one-row source matches the target rows through the ON clause
/// (every row under the id is updated, a row is inserted when none
/// matches), so a configured table that pre-exists without the primary key
/// the store would create works too.
fn store_response_sql(table: &str, columns: &[&str]) -> (String, Vec<String>) {
    let binds: Vec<String> = (1..=columns.len()).map(|i| format!("v{i}")).collect();
    let id_col = columns.first().copied().unwrap_or("id");
    let id_bind = binds.first().cloned().unwrap_or_else(|| "v1".to_string());
    let updates: Vec<String> = columns
        .iter()
        .zip(&binds)
        .skip(1)
        .map(|(col, bind)| format!("t.{col} = :{bind}"))
        .collect();
    let values: Vec<String> = binds.iter().map(|bind| format!(":{bind}")).collect();
    let matched = if updates.is_empty() {
        String::new()
    } else {
        format!(" WHEN MATCHED THEN UPDATE SET {}", updates.join(", "))
    };
    let sql = format!(
        "MERGE INTO {table} t USING (SELECT :{id_bind} AS {id_col} FROM dual) s \
         ON (t.{id_col} = s.{id_col}){matched} \
         WHEN NOT MATCHED THEN INSERT ({}) VALUES ({})",
        columns.join(", "),
        values.join(", ")
    );
    (sql, binds)
}

/// The statement of a link: a MERGE on the (conversation, item) key, the
/// first two columns, so the row takes the new stamp and the new values of
/// any extra column when it exists and is inserted when it does not. A second
/// link of an item moves it, the contract of
/// [`ConversationItemStorage::link_item`], where the key used to refuse the
/// insert. Named binds, one per column, so an extra column's two occurrences
/// (UPDATE SET and INSERT VALUES) are bound once.
fn link_sql(table: &str, columns: &[&str]) -> (String, Vec<String>) {
    let binds: Vec<String> = (1..=columns.len()).map(|i| format!("v{i}")).collect();
    let placeholders: Vec<String> = binds.iter().map(|bind| format!(":{bind}")).collect();
    let keyed = placeholders.len().min(LINK_KEY_COLUMNS);
    let (key_binds, extra_binds) = placeholders.split_at(keyed);
    (
        merge_links_sql(table, columns, &[key_binds.to_vec()], extra_binds),
        binds,
    )
}

/// The statement of a batch link: one MERGE whose source is the batch's rows
/// (ids and stamps, far under the source's size limit), keyed as [`link_sql`]
/// is. The binds are `:cid` (the conversation, once), `:iid<n>` and `:ts<n>`
/// per row, `:ex<n>` per extra column (one value for the whole batch, bound
/// outside the source). The batch must hold distinct item ids: two source
/// rows under one key are refused by the MERGE (ORA-30926, or ORA-00001 when
/// both insert), so the caller keeps the last stamp of a repeated id.
fn link_items_sql(table: &str, columns: &[&str], rows: usize) -> String {
    let extra = columns.len().saturating_sub(LINK_KEY_COLUMNS);
    let key_rows: Vec<Vec<String>> = (0..rows)
        .map(|idx| {
            vec![
                ":cid".to_string(),
                format!(":iid{idx}"),
                format!(":ts{idx}"),
            ]
        })
        .collect();
    let extra_binds: Vec<String> = (0..extra).map(|ei| format!(":ex{ei}")).collect();
    merge_links_sql(table, columns, &key_rows, &extra_binds)
}

/// The columns a link's source row carries: the key (conversation, item) and
/// the stamp. Any further column is an extra column of the table.
const LINK_KEY_COLUMNS: usize = 3;

/// One MERGE into the links `table` with one source row per entry of
/// `key_rows` (one bind per key column, in the columns' order), matched on
/// the first two columns: the stamp and the extra columns of a matched row
/// take the new values, an unmatched row is inserted. The extra columns'
/// values (`extra_binds`, one per column after the first three) are bound in
/// UPDATE SET and INSERT VALUES, outside the source: a string above 4,000
/// bytes is accepted there for a CLOB column, as in an INSERT, while the same
/// bind in a subquery's select list fails with ORA-01461 (the rule
/// [`store_response_sql`] follows); an extra column is user-configured and
/// may well be a CLOB.
fn merge_links_sql(
    table: &str,
    columns: &[&str],
    key_rows: &[Vec<String>],
    extra_binds: &[String],
) -> String {
    let keyed = columns.len().min(LINK_KEY_COLUMNS);
    let (key_columns, extra_columns) = columns.split_at(keyed);
    let source: Vec<String> = key_rows
        .iter()
        .map(|binds| {
            let selected: Vec<String> = key_columns
                .iter()
                .zip(binds)
                .map(|(col, bind)| format!("{bind} AS {col}"))
                .collect();
            format!("SELECT {} FROM dual", selected.join(", "))
        })
        .collect();
    let on: Vec<String> = key_columns
        .iter()
        .take(2)
        .map(|col| format!("t.{col} = s.{col}"))
        .collect();
    let updates: Vec<String> = key_columns
        .iter()
        .skip(2)
        .map(|col| format!("t.{col} = s.{col}"))
        .chain(
            extra_columns
                .iter()
                .zip(extra_binds)
                .map(|(col, bind)| format!("t.{col} = {bind}")),
        )
        .collect();
    let matched = if updates.is_empty() {
        String::new()
    } else {
        format!(" WHEN MATCHED THEN UPDATE SET {}", updates.join(", "))
    };
    let values: Vec<String> = key_columns
        .iter()
        .map(|col| format!("s.{col}"))
        .chain(extra_binds.iter().cloned())
        .collect();
    format!(
        "MERGE INTO {table} t USING ({}) s ON ({}){matched} \
         WHEN NOT MATCHED THEN INSERT ({}) VALUES ({})",
        source.join(" UNION ALL "),
        on.join(" AND "),
        columns.join(", "),
        values.join(", ")
    )
}

/// The batch with one row per item id, the last stamp of a repeated id kept
/// in the position of its first occurrence: the result of linking the items
/// one by one, which the other backends give a repeated id.
fn last_stamp_per_item(items: Vec<(String, DateTime<Utc>)>) -> Vec<(String, DateTime<Utc>)> {
    let mut positions: HashMap<String, usize> = HashMap::new();
    let mut rows: Vec<(String, DateTime<Utc>)> = Vec::with_capacity(items.len());
    for (id, ts) in items {
        match positions.get(&id) {
            Some(&at) => rows[at].1 = ts,
            None => {
                positions.insert(id.clone(), rows.len());
                rows.push((id, ts));
            }
        }
    }
    rows
}

/// The statement of `store_response_if_open`, with the names of its binds
/// in the columns' order: an UPDATE of every column but the id of the row
/// under the id, applied only while the stored object's status is not
/// terminal (`open`; none when the schema skips the column, so the update
/// is plain). The values are bound by name as in `store_response_sql`.
fn update_if_open_sql(table: &str, columns: &[&str], open: Option<&str>) -> (String, Vec<String>) {
    let binds: Vec<String> = (1..=columns.len()).map(|i| format!("v{i}")).collect();
    let id_col = columns.first().copied().unwrap_or("id");
    let id_bind = binds.first().cloned().unwrap_or_else(|| "v1".to_string());
    let updates: Vec<String> = columns
        .iter()
        .zip(&binds)
        .skip(1)
        .map(|(col, bind)| format!("{col} = :{bind}"))
        .collect();
    let condition = open.map(|open| format!(" AND {open}")).unwrap_or_default();
    let sql = format!(
        "UPDATE {table} SET {} WHERE {id_col} = :{id_bind}{condition}",
        updates.join(", ")
    );
    (sql, binds)
}

/// The condition of `update_if_open_sql` over the stored object's JSON text
/// (a record without a status counts as open).
fn oracle_open_predicate(col_raw: &str) -> String {
    format!(
        "COALESCE(JSON_VALUE({col_raw}, '$.status'), ' ') NOT IN ({})",
        sql_string_list(&TERMINAL_RESPONSE_STATUSES)
    )
}

#[async_trait]
impl ResponseStorage for OracleResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let StoredResponse {
            id,
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
            tenant,
        } = response;

        let return_id = id.clone();

        let response_id_str = id.0;
        let previous_id = previous_response_id.map(|r| r.0);
        let json_input = serde_json::to_string(&input)?;
        let json_raw_response = serde_json::to_string(&raw_response)?;
        let schema = self.store.schema.clone();
        // Capture extra columns before spawn_blocking (task-locals don't propagate)
        let hook_extra = current_extra_columns().unwrap_or_default();

        self.store
            .execute(move |conn| {
                let s = &schema.responses;
                let table = s.qualified_table(schema.owner.as_deref());

                // Build column list and params dynamically, respecting skip_columns
                let logical_fields: &[(&str, &dyn ToSql)] = &[
                    ("id", &response_id_str),
                    ("previous_response_id", &previous_id),
                    ("input", &json_input),
                    ("created_at", &created_at),
                    ("safety_identifier", &safety_identifier),
                    ("model", &model),
                    ("conversation_id", &conversation_id),
                    ("raw_response", &json_raw_response),
                    ("tenant", &tenant),
                ];

                let mut columns = Vec::new();
                let mut params: Vec<&dyn ToSql> = Vec::new();
                for &(logical, val) in logical_fields {
                    // Always include id (primary key), skip others if configured
                    if logical != "id" && s.is_skipped(logical) {
                        continue;
                    }
                    columns.push(s.col(logical));
                    params.push(val);
                }

                // Append extra columns from hooks or defaults
                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(s, &hook_extra);
                for (name, val) in &extra_cols {
                    columns.push(*name);
                    params.push(val);
                }

                let (sql, binds) = store_response_sql(&table, &columns);
                let named: Vec<(&str, &dyn ToSql)> = binds
                    .iter()
                    .map(String::as_str)
                    .zip(params.iter().copied())
                    .collect();
                conn.execute_named(&sql, &named)
                    .map(|_| ())
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)?;

        Ok(return_id)
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        let StoredResponse {
            id,
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
            tenant,
        } = response;
        let read_back = id.clone();
        let response_id_str = id.0;
        let previous_id = previous_response_id.map(|r| r.0);
        let json_input = serde_json::to_string(&input)?;
        let json_raw_response = serde_json::to_string(&raw_response)?;
        let schema = self.store.schema.clone();
        let hook_extra = current_extra_columns().unwrap_or_default();

        let updated = self
            .store
            .execute(move |conn| {
                let s = &schema.responses;
                let table = s.qualified_table(schema.owner.as_deref());
                let logical_fields: &[(&str, &dyn ToSql)] = &[
                    ("id", &response_id_str),
                    ("previous_response_id", &previous_id),
                    ("input", &json_input),
                    ("created_at", &created_at),
                    ("safety_identifier", &safety_identifier),
                    ("model", &model),
                    ("conversation_id", &conversation_id),
                    ("raw_response", &json_raw_response),
                    ("tenant", &tenant),
                ];
                let mut columns = Vec::new();
                let mut params: Vec<&dyn ToSql> = Vec::new();
                for &(logical, val) in logical_fields {
                    if logical != "id" && s.is_skipped(logical) {
                        continue;
                    }
                    columns.push(s.col(logical));
                    params.push(val);
                }
                let extra_cols: Vec<(&str, Option<String>)> =
                    resolve_extra_column_values(s, &hook_extra);
                for (name, val) in &extra_cols {
                    columns.push(*name);
                    params.push(val);
                }
                let open = (!s.is_skipped("raw_response"))
                    .then(|| oracle_open_predicate(s.col("raw_response")));
                let (sql, binds) = update_if_open_sql(&table, &columns, open.as_deref());
                let named: Vec<(&str, &dyn ToSql)> = binds
                    .iter()
                    .map(String::as_str)
                    .zip(params.iter().copied())
                    .collect();
                let stmt = conn.execute_named(&sql, &named).map_err(map_oracle_error)?;
                stmt.row_count().map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)?;
        if updated > 0 {
            return Ok(ConditionalStore::Stored);
        }
        Ok(match self.get_response(&read_back).await? {
            Some(current) => ConditionalStore::Closed(Box::new(current)),
            None => ConditionalStore::Missing,
        })
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        let read_back = id.clone();
        let id_str = id.0.clone();
        let json_raw_response = serde_json::to_string(&raw_response)?;
        let schema = self.store.schema.clone();
        let updated = self
            .store
            .execute(move |conn| {
                let s = &schema.responses;
                if s.is_skipped("raw_response") {
                    // No object column: nothing a transition could move.
                    return Ok(1);
                }
                let table = s.qualified_table(schema.owner.as_deref());
                let open = oracle_open_predicate(s.col("raw_response"));
                let (sql, binds) =
                    update_if_open_sql(&table, &[s.col("id"), s.col("raw_response")], Some(&open));
                let params: Vec<&dyn ToSql> = vec![&id_str, &json_raw_response];
                let named: Vec<(&str, &dyn ToSql)> =
                    binds.iter().map(String::as_str).zip(params).collect();
                let stmt = conn.execute_named(&sql, &named).map_err(map_oracle_error)?;
                stmt.row_count().map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)?;
        if updated > 0 {
            return Ok(ConditionalStore::Stored);
        }
        Ok(match self.get_response(&read_back).await? {
            Some(current) => ConditionalStore::Closed(Box::new(current)),
            None => ConditionalStore::Missing,
        })
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.clone();
        let select_base = self.select_base.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let col_id = schema.responses.col("id");
                let sql = format!("{select_base} WHERE {col_id} = :1");
                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&id]).map_err(map_oracle_error)?;
                match rows.next() {
                    Some(row) => {
                        let row = row.map_err(map_oracle_error)?;
                        Self::build_response_from_row(&row, &schema).map(Some)
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(ResponseStorageError::from_failure)
    }

    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let schema = self.store.schema.clone();
        let Some(sql) = find_by_output_item_sql(&schema, &self.select_base) else {
            return Ok(None);
        };
        let item_id = item_id.to_string();

        self.store
            .execute(move |conn| {
                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&item_id]).map_err(map_oracle_error)?;
                match rows.next() {
                    Some(row) => {
                        let row = row.map_err(map_oracle_error)?;
                        Self::build_response_from_row(&row, &schema).map(Some)
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(ResponseStorageError::from_failure)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        let id = response_id.0.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let s = &schema.responses;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_id = s.col("id");

                conn.execute(&format!("DELETE FROM {table} WHERE {col_id} = :1"), &[&id])
                    .map(|_| ())
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        let identifier = identifier.to_string();
        let tenant = tenant.map(str::to_string);
        let select_base = self.select_base.clone();
        let schema = self.store.schema.clone();

        self.store
            .execute(move |conn| {
                let s = &schema.responses;

                // If safety_identifier is skipped, we cannot filter by identifier
                if s.is_skipped("safety_identifier") {
                    return Ok(Vec::new());
                }

                let col_safety = s.col("safety_identifier");
                let mut filter = format!("{col_safety} = :1");
                let mut binds: Vec<&dyn ToSql> = vec![&identifier];
                if let Some(scope) = tenant_scope_sql(s, binds.len() + 1) {
                    filter.push_str(&scope);
                    binds.push(&tenant);
                }

                let sql = if s.is_skipped("created_at") {
                    // No created_at column — omit ORDER BY
                    if let Some(limit) = limit {
                        format!(
                            "SELECT * FROM ({select_base} WHERE {filter}) \
                             WHERE ROWNUM <= {limit}"
                        )
                    } else {
                        format!("{select_base} WHERE {filter}")
                    }
                } else {
                    let col_created = s.col("created_at");
                    if let Some(limit) = limit {
                        format!(
                            "SELECT * FROM ({select_base} WHERE {filter} \
                             ORDER BY {col_created} DESC) WHERE ROWNUM <= {limit}"
                        )
                    } else {
                        format!("{select_base} WHERE {filter} ORDER BY {col_created} DESC")
                    }
                };

                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&binds[..]).map_err(map_oracle_error)?;
                let mut results = Vec::new();

                for row in &mut rows {
                    let row = row.map_err(map_oracle_error)?;
                    results.push(Self::build_response_from_row(&row, &schema)?);
                }

                Ok(results)
            })
            .await
            .map_err(ResponseStorageError::from_failure)
    }

    async fn delete_unstored_older_than(
        &self,
        cutoff: DateTime<Utc>,
    ) -> Result<usize, ResponseStorageError> {
        let schema = self.store.schema.clone();
        // The object (its `store`, `status` and `completed_at`) is what the
        // query reads; without the column there is nothing to read.
        if schema.responses.is_skipped("raw_response") {
            return Ok(0);
        }
        let cutoff_secs: i64 = cutoff.timestamp();

        let affected = self
            .store
            .execute(move |conn| {
                let s = &schema.responses;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_raw = s.col("raw_response");
                let ended = if s.is_skipped("created_at") {
                    format!("TO_NUMBER(JSON_VALUE({col_raw}, '$.completed_at'))")
                } else {
                    format!(
                        "COALESCE(TO_NUMBER(JSON_VALUE({col_raw}, '$.completed_at')), \
                         ROUND((CAST({} AS DATE) - DATE '1970-01-01') * 86400))",
                        s.col("created_at")
                    )
                };

                // The predicate is the expression the table's unstored index
                // holds (see the bootstrap), so the sweep reads those rows only.
                conn.execute(
                    &format!(
                        "DELETE FROM {table} WHERE {} IN ('completed', 'incomplete', 'failed', 'cancelled') \
                         AND {ended} < :1",
                        unstored_status_expr(col_raw)
                    ),
                    &[&cutoff_secs],
                )
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)?;

        let deleted = affected
            .row_count()
            .map_err(|e| ResponseStorageError::StorageError(map_oracle_error(e)))?
            as usize;
        Ok(deleted)
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        let identifier = identifier.to_string();
        let tenant = tenant.map(str::to_string);
        let schema = self.store.schema.clone();

        let schema_check = self.store.schema.clone();
        if schema_check.responses.is_skipped("safety_identifier") {
            return Ok(0);
        }

        let affected = self
            .store
            .execute(move |conn| {
                let s = &schema.responses;
                let table = s.qualified_table(schema.owner.as_deref());
                let col_safety = s.col("safety_identifier");
                let mut filter = format!("{col_safety} = :1");
                let mut binds: Vec<&dyn ToSql> = vec![&identifier];
                if let Some(scope) = tenant_scope_sql(s, binds.len() + 1) {
                    filter.push_str(&scope);
                    binds.push(&tenant);
                }

                conn.execute(&format!("DELETE FROM {table} WHERE {filter}"), &binds[..])
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::from_failure)?;

        let deleted = affected
            .row_count()
            .map_err(|e| ResponseStorageError::StorageError(map_oracle_error(e)))?
            as usize;
        Ok(deleted)
    }
}

/// The status of an unstored response object, NULL for a stored one: the
/// expression the unstored index holds and the sweep's predicate reads.
fn unstored_status_expr(col_raw: &str) -> String {
    format!(
        "CASE WHEN JSON_VALUE({col_raw}, '$.store') = 'false' THEN JSON_VALUE({col_raw}, '$.status') END"
    )
}

fn create_index_if_missing(
    conn: &Connection,
    table_upper: &str,
    index_name: &str,
    ddl: &str,
) -> Result<(), String> {
    let count: i64 = conn
        .query_row_as(
            &format!(
                "SELECT COUNT(*) FROM user_indexes \
                 WHERE table_name = '{table_upper}' AND index_name = :1"
            ),
            &[&index_name],
        )
        .map_err(map_oracle_error)?;

    if count == 0 {
        if let Err(err) = conn.execute(ddl, &[]) {
            if let Some(db_err) = err.db_error() {
                // ORA-00955: name is already used by an existing object
                // ORA-01408: such column list already indexed
                // ORA-00904: invalid identifier (column not yet added by migration)
                if db_err.code() != 955 && db_err.code() != 1408 && db_err.code() != 904 {
                    return Err(map_oracle_error(err));
                }
            } else {
                return Err(map_oracle_error(err));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
