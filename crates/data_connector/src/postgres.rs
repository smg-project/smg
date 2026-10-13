//! Postgres storage implementation using PostgresStore helper

use std::{str::FromStr, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{ClientWrapper, Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use serde_json::Value;
use tokio_postgres::{error::SqlState, types::ToSql, IsolationLevel, NoTls, Row, Statement};

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
    factory::StoreContactError,
    postgres_migrations::POSTGRES_HISTORY_MIGRATIONS,
    schema::SchemaConfig,
    versioning::{
        postgres_create_versions_table, postgres_current_version, postgres_table_exists,
        postgres_table_exists_sql, run_postgres_migrations, schema_ahead_check,
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

        let mut pg_config = tokio_postgres::Config::from_str(config.db_url.as_str())
            .map_err(|e| format!("Invalid PostgreSQL connection URL: {e}"))?;
        // Each call as a whole is bounded by the `bounded` wrappers the
        // factory installs. The pool and the driver bound their own steps
        // (connecting, unless the URL sets its own `connect_timeout`; waiting
        // for a pooled connection; creating one; recycling one) at twice that
        // bound: backstops that clean up after a call the wrapper has already
        // answered, never the first to fire.
        let backstop = config.op_timeout().saturating_mul(2);
        if pg_config.get_connect_timeout().is_none() {
            pg_config.connect_timeout(backstop);
        }
        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let mgr = Manager::from_config(pg_config, NoTls, mgr_config);
        let pool = Pool::builder(mgr)
            .max_size(config.pool_max)
            .wait_timeout(Some(backstop))
            .create_timeout(Some(backstop))
            .recycle_timeout(Some(backstop))
            .runtime(Runtime::Tokio1)
            .build()
            .map_err(|e| format!("Failed to build PostgreSQL connection pool: {e}"))?;

        Ok(Self { pool, schema })
    }

    /// The start-up contact: one connection and one round trip within
    /// `bound`, so a store that refuses the connection, or accepts it and never
    /// answers, is a verdict before the gateway serves.
    pub(crate) async fn contact(&self, bound: Duration) -> Result<(), StoreContactError> {
        let round_trip = async {
            let client = self.pool.get().await.map_err(|e| e.to_string())?;
            client
                .simple_query("SELECT 1")
                .await
                .map_err(|e| e.to_string())?;
            Ok::<(), String>(())
        };
        match tokio::time::timeout(bound, round_trip).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(cause)) => Err(StoreContactError::Failed(cause)),
            Err(_elapsed) => Err(StoreContactError::NoAnswer),
        }
    }

    /// Brings the database to the schema this build expects, once per
    /// database and in turn across gateways: one transaction under the schema
    /// advisory lock creates the versions table, refuses a database whose
    /// recorded version is ahead of the migrations this build knows (before
    /// any history table is touched, see `schema_ahead_check`), creates the
    /// history tables (`IF NOT EXISTS`), records every migration as applied
    /// when the responses table did not exist before (a first start creates
    /// the tables in their final shape, so nothing is pending and no
    /// `auto_migrate` is needed), applies the pending migrations otherwise
    /// (or refuses when `auto_migrate` is off), and creates the indexes
    /// last, once the columns they cover exist. Gateways starting together
    /// on an empty database all come up: the next one waits for the first's
    /// commit and finds the schema in place. The index an earlier release
    /// built for the sweep of unstored records goes after the commit,
    /// concurrently (see [`Self::drop_legacy_unstored_index`]).
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
        // The store's version against this build's migrations first, before
        // any table is touched: a store a newer build migrated forward is
        // refused here whatever the responses table looks like (a table that
        // a newer migration renamed or dropped would otherwise read as a
        // fresh database and be re-created in its legacy shape).
        let current = postgres_current_version(&*tx).await?;
        schema_ahead_check(
            "postgres",
            current,
            &POSTGRES_HISTORY_MIGRATIONS,
            schema.version,
        )?;
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
            // The unstored-record index is built here, on the table just created
            // and still empty: on a populated table the build would hold a share
            // lock on it inside this transaction, blocking every replica's writes
            // and the other gateways waiting on the schema lock. An existing
            // table gets it from its operator, concurrently (see the release
            // note); the sweep works without it, as a scan.
            if let Some(index_ddl) = PostgresResponseStorage::unstored_index_ddl(schema) {
                let index = tx.savepoint("unstored_index").await.map_err(|e| {
                    format!(
                        "failed to open the unstored index savepoint: {}",
                        describe_error(&e)
                    )
                })?;
                match index.batch_execute(&index_ddl).await {
                    Ok(()) => index.commit().await.map_err(|e| {
                        format!(
                            "failed to release the unstored index savepoint: {}",
                            describe_error(&e)
                        )
                    })?,
                    Err(e) => {
                        tracing::warn!(
                            "unstored response index not created (the sweep of unstored background records scans the table): {}",
                            describe_error(&e)
                        );
                        index.rollback().await.map_err(|e| {
                            format!(
                                "failed to roll back the unstored index savepoint: {}",
                                describe_error(&e)
                            )
                        })?;
                    }
                }
            }
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
        // the column keeps starting without that index, as before. An index
        // the catalog already has is not asked for again: the statement
        // takes its share lock before it finds the index in place, and a
        // share lock waiting behind an open write queues every later write
        // on the table behind it (a start next to another's concurrent drop
        // would wait that way too), so a start on a database that has its
        // indexes takes no lock on the live tables.
        let links_index = PostgresConversationItemStorage::links_index_name(schema);
        if !postgres_index_exists(&tx, &links_index).await? {
            tx.batch_execute(&PostgresConversationItemStorage::index_ddl(schema))
                .await
                .map_err(|e| {
                    format!(
                        "failed to create the conversation item links index: {}",
                        describe_error(&e)
                    )
                })?;
        }
        let safety_index = PostgresResponseStorage::safety_index_name(schema);
        if let Some(index_ddl) = PostgresResponseStorage::index_ddl(schema) {
            if !postgres_index_exists(&tx, &safety_index).await? {
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
        }

        tx.commit().await.map_err(|e| {
            format!(
                "failed to commit the schema bootstrap: {}",
                describe_error(&e)
            )
        })?;

        // The index an earlier release built over `::jsonb` (it refused every
        // row whose object carries the escape `\u0000`) goes after the commit,
        // outside any transaction and concurrently: a plain `DROP INDEX`
        // takes an ACCESS EXCLUSIVE lock on the table, which on a database
        // other replicas are serving queues behind every open transaction on
        // the table and every later statement behind it, held to the commit.
        // An existing table gets the replacement from its operator,
        // concurrently too (see the release note).
        self.drop_legacy_unstored_index(&client).await
    }

    /// Drops the index an earlier release built under
    /// [`LEGACY_UNSTORED_INDEX`] when the database still has it: `DROP INDEX
    /// CONCURRENTLY`, which blocks no read or write on the table and waits
    /// for the transactions using it to end instead (it cannot run inside a
    /// transaction block, so this runs on `client` after the bootstrap's
    /// commit). Nothing is sent once the index is gone.
    async fn drop_legacy_unstored_index(
        &self,
        client: &tokio_postgres::Client,
    ) -> Result<(), String> {
        let index = PostgresResponseStorage::legacy_unstored_index_name(&self.schema);
        let present: bool = client
            .query_one(postgres_table_exists_sql(), &[&index])
            .await
            .map_err(|e| {
                format!(
                    "failed to check for the earlier unstored response index: {}",
                    describe_error(&e)
                )
            })?
            .get(0);
        if !present {
            return Ok(());
        }
        tracing::info!(
            index = %index,
            "dropping the earlier unstored response index concurrently (the drop waits for the transactions using the table to end, and blocks none)"
        );
        client
            .batch_execute(&PostgresResponseStorage::legacy_unstored_index_drop_ddl(
                &self.schema,
            ))
            .await
            .map_err(|e| {
                format!(
                    "failed to drop the earlier unstored response index: {}",
                    describe_error(&e)
                )
            })
    }
}

/// Whether the catalog `tx` sees has the index `index` (owner-qualified when
/// the schema has an owner): `to_regclass` resolves an index as it does a
/// table.
async fn postgres_index_exists(
    tx: &tokio_postgres::Transaction<'_>,
    index: &str,
) -> Result<bool, String> {
    let row = tx
        .query_one(postgres_table_exists_sql(), &[&index])
        .await
        .map_err(|e| {
            format!(
                "failed to check for the {index} index: {}",
                describe_error(&e)
            )
        })?;
    Ok(row.get(0))
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
//
// A cached statement can go stale on its connection: the type of a column it
// selects altered under it (a migration a newer build runs during a rolling
// upgrade) makes the server answer every later call with SQLSTATE `0A000`
// ("cached plan must not change result type"), and a statement the server
// deallocated answers `26000`. The pool does not notice either (its recycling
// only checks that the connection is open), so a helper drops such an entry
// from the connection's cache and makes the call once more on a statement
// prepared afresh; the connection is whole again for every later call.

/// Whether `error` says the statement the server was handed is stale: its
/// cached plan no longer fits the table (`0A000` raised by the server's
/// cached-plan revalidation; the class holds unrelated answers too, which
/// stay as they are) or it no longer exists on the server (`26000`).
fn is_stale_statement(error: &tokio_postgres::Error) -> bool {
    error
        .as_db_error()
        .is_some_and(|db| is_stale_statement_state(db.code(), db.routine()))
}

/// The server routine that raises `0A000` for a cached plan whose result no
/// longer fits ("cached plan must not change result type"); the routine field
/// is sent whatever the server's message locale, the message is not.
const CACHED_PLAN_REVALIDATION: &str = "RevalidateCachedQuery";

fn is_stale_statement_state(code: &SqlState, routine: Option<&str>) -> bool {
    *code == SqlState::INVALID_SQL_STATEMENT_NAME
        || (*code == SqlState::FEATURE_NOT_SUPPORTED && routine == Some(CACHED_PLAN_REVALIDATION))
}

/// The statement for `sql` prepared afresh after a stale answer: the entry
/// goes from the connection's cache (the statement it held is closed on the
/// server when its last handle goes) and the next prepare fills it again.
async fn fresh_statement(
    client: &ClientWrapper,
    sql: &str,
) -> Result<Statement, tokio_postgres::Error> {
    drop(client.statement_cache.remove(sql, &[]));
    client.prepare_cached(sql).await
}

/// Runs `sql` as a statement prepared once per connection.
async fn query_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<Vec<Row>, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    match client.query(&statement, params).await {
        Err(error) if is_stale_statement(&error) => {
            let statement = fresh_statement(client, sql).await?;
            client.query(&statement, params).await
        }
        result => result,
    }
}

/// Runs `sql` as a statement prepared once per connection; one row.
async fn query_one_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<Row, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    match client.query_one(&statement, params).await {
        Err(error) if is_stale_statement(&error) => {
            let statement = fresh_statement(client, sql).await?;
            client.query_one(&statement, params).await
        }
        result => result,
    }
}

/// Runs `sql` as a statement prepared once per connection; the rows affected.
async fn execute_cached(
    client: &ClientWrapper,
    sql: &str,
    params: &[&(dyn ToSql + Sync)],
) -> Result<u64, tokio_postgres::Error> {
    let statement = client.prepare_cached(sql).await?;
    match client.execute(&statement, params).await {
        Err(error) if is_stale_statement(&error) => {
            let statement = fresh_statement(client, sql).await?;
            client.execute(&statement, params).await
        }
        result => result,
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
        if !s.is_skipped("tenant") {
            col_defs.push(format!("{} VARCHAR(255)", s.col("tenant")));
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

    /// The columns a conversation is read with: the id, then its creation
    /// time, metadata and tenant when the schema keeps them.
    fn row_columns(s: &crate::schema::TableConfig) -> Vec<String> {
        let mut columns = vec![s.col("id").to_string()];
        for name in ["created_at", "metadata", "tenant"] {
            if !s.is_skipped(name) {
                columns.push(s.col(name).to_string());
            }
        }
        columns
    }

    /// The conversation a row read with [`Self::row_columns`] holds.
    fn conversation_from_row(
        s: &crate::schema::TableConfig,
        row: &Row,
    ) -> Result<Conversation, ConversationStorageError> {
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
        let tenant: Option<String> = if s.is_skipped("tenant") {
            None
        } else {
            row.get(s.col("tenant"))
        };
        Ok(Conversation::with_parts(
            ConversationId(id_str),
            created_at,
            metadata,
            tenant,
        ))
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
        if !s.is_skipped("tenant") {
            col_names.push(s.col("tenant"));
            params.push(&conversation.tenant);
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
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(ConversationStorageError::from_pg)?;
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");
        let sql = format!(
            "SELECT {} FROM {table} WHERE {col_id} = $1",
            Self::row_columns(s).join(", ")
        );

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &[&id.0.as_str()])
            .await
            .map_err(ConversationStorageError::from_pg)?;
        match rows.first() {
            Some(row) => Self::conversation_from_row(s, row).map(Some),
            None => Ok(None),
        }
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        if s.is_skipped("metadata") {
            // Nothing to move in the table: the row as stored, with the
            // metadata the caller holds.
            return Ok(self
                .get_conversation(id)
                .await?
                .map(|current| Conversation {
                    metadata,
                    ..current
                }));
        }
        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_id = s.col("id");
        let col_meta = s.col("metadata");
        // One statement: the update, returning the row's other columns (its
        // creation time, its tenant) for the answer; no row returned = no
        // row matched.
        let sql = format!(
            "UPDATE {table} SET {col_meta} = $1 WHERE {col_id} = $2 RETURNING {}",
            Self::row_columns(s).join(", ")
        );

        // Bind the JSON column as `Value`, not a serialized `String` (see
        // `create_conversation` for why `tokio-postgres` requires this).
        let metadata_value: Option<Value> = metadata.map(Value::Object);
        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &[&metadata_value, &id.0.as_str()])
            .await
            .map_err(ConversationStorageError::from_pg)?;
        match rows.first() {
            Some(row) => Self::conversation_from_row(s, row).map(Some),
            None => Ok(None),
        }
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
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;
        let rows_deleted = execute_cached(
            &client,
            &format!("DELETE FROM {table} WHERE {col_id} = $1"),
            &[&id.0.as_str()],
        )
        .await
        .map_err(ConversationStorageError::from_pg)?;
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
        let item_core_cols: [(&str, &str); 7] = [
            ("response_id", "VARCHAR(64)"),
            ("item_type", "VARCHAR(32) NOT NULL"),
            ("role", "VARCHAR(32)"),
            ("content", "JSON"),
            ("status", "VARCHAR(32)"),
            ("created_at", "TIMESTAMPTZ"),
            ("tenant", "VARCHAR(255)"),
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
            "CREATE INDEX IF NOT EXISTS {LINKS_INDEX} ON {links_table} ({}, {});",
            sl.col("conversation_id"),
            sl.col("added_at")
        )
    }

    /// The name of the links index, under the schema's owner when it has one.
    pub(crate) fn links_index_name(schema: &SchemaConfig) -> String {
        qualified_index(schema, LINKS_INDEX)
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
            tenant,
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
        if !si.is_skipped("tenant") {
            col_names.push(si.col("tenant"));
            params.push(&tenant);
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
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(ConversationItemStorageError::from_pg)?;
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

        let sql = link_sql(&table, &col_names);

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        execute_cached(&client, &sql, &params)
            .await
            .map_err(ConversationItemStorageError::from_pg)?;
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
                .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
            let cursor_sql = format!(
                "SELECT {l_added_at} FROM {links_table} WHERE {l_conv_id} = $1 AND {l_item_id} = $2"
            );
            let rows = query_cached(&client, &cursor_sql, &[&cid, &aid.as_str()])
                .await
                .map_err(ConversationItemStorageError::from_pg)?;
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
            "tenant",
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
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        let rows = if let Some((ts, iid)) = &after_key {
            query_cached(&client, &sql, &[&cid, ts, iid, &limit])
                .await
                .map_err(ConversationItemStorageError::from_pg)?
        } else {
            query_cached(&client, &sql, &[&cid, &limit])
                .await
                .map_err(ConversationItemStorageError::from_pg)?
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
            "tenant",
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
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &[&item_id.0.as_str()])
            .await
            .map_err(ConversationItemStorageError::from_pg)?;
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
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        let row = query_one_cached(
            &client,
            &sql,
            &[&conversation_id.0.as_str(), &item_id.0.as_str()],
        )
        .await
        .map_err(ConversationItemStorageError::from_pg)?;
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
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        execute_cached(
            &client,
            &sql,
            &[&conversation_id.0.as_str(), &item_id.0.as_str()],
        )
        .await
        .map_err(ConversationItemStorageError::from_pg)?;
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
    let tenant: Option<String> = if si.is_skipped("tenant") {
        None
    } else {
        row.get(si.col("tenant"))
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
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
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
        let core_cols: [(&str, &str); 8] = [
            ("conversation_id", "VARCHAR(64)"),
            ("previous_response_id", "VARCHAR(64)"),
            ("input", "JSON"),
            ("created_at", "TIMESTAMPTZ"),
            ("safety_identifier", "VARCHAR(128)"),
            ("model", "VARCHAR(128)"),
            ("raw_response", "JSON"),
            ("tenant", "VARCHAR(255)"),
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
            "CREATE INDEX IF NOT EXISTS {SAFETY_INDEX} ON {table} ({});",
            s.col("safety_identifier")
        ))
    }

    /// The name of the safety-identifier index, under the schema's owner
    /// when it has one.
    pub(crate) fn safety_index_name(schema: &SchemaConfig) -> String {
        qualified_index(schema, SAFETY_INDEX)
    }

    /// The index behind [`ResponseStorage::delete_unstored_older_than`]: the
    /// object's status, over the unstored records only, so a sweep reads
    /// those rows and no other. Its expressions are the sweep's predicate,
    /// read through [`nul_safe_json_text`]: an index over `::jsonb` refused
    /// every row whose object carries the escape `\u0000` (a model's or a
    /// client's text can), so that form is not used.
    pub(crate) fn unstored_index_ddl(schema: &SchemaConfig) -> Option<String> {
        let s = &schema.responses;
        if s.is_skipped("raw_response") {
            return None;
        }
        let table = s.qualified_table(schema.owner.as_deref());
        let col_raw = s.col("raw_response");
        let status = nul_safe_json_text(col_raw, "status");
        let store = nul_safe_json_text(col_raw, "store");
        Some(format!(
            "CREATE INDEX IF NOT EXISTS {UNSTORED_INDEX} ON {table} (({status})) WHERE {store} = 'false';"
        ))
    }

    /// The statement behind [`ResponseStorage::delete_unstored_older_than`]
    /// (the cutoff bound as `$1`, epoch seconds): the unstored records in a
    /// terminal status whose `completed_at` (or, without one, `created_at`)
    /// lies before it. Its `store` and `status` reads are the index's own
    /// expressions (see [`Self::unstored_index_ddl`]; a unit test holds them
    /// together), so the planner matches the index. `None` when the schema
    /// skips the object column (nothing to read).
    pub(crate) fn unstored_sweep_sql(schema: &SchemaConfig) -> Option<String> {
        let s = &schema.responses;
        if s.is_skipped("raw_response") {
            return None;
        }
        let table = s.qualified_table(schema.owner.as_deref());
        let col_raw = s.col("raw_response");
        let store = nul_safe_json_text(col_raw, "store");
        let status = nul_safe_json_text(col_raw, "status");
        let completed_at = nul_safe_json_text(col_raw, "completed_at");
        let ended = if s.is_skipped("created_at") {
            format!("({completed_at})::bigint")
        } else {
            format!(
                "COALESCE(({completed_at})::bigint, EXTRACT(EPOCH FROM {})::bigint)",
                s.col("created_at")
            )
        };
        Some(format!(
            "DELETE FROM {table} WHERE {store} = 'false' \
             AND {status} IN ('completed', 'incomplete', 'failed', 'cancelled') \
             AND {ended} < $1"
        ))
    }

    /// The name of the index an earlier release built over `::jsonb`,
    /// [`LEGACY_UNSTORED_INDEX`], under the schema's owner when it has one.
    pub(crate) fn legacy_unstored_index_name(schema: &SchemaConfig) -> String {
        qualified_index(schema, LEGACY_UNSTORED_INDEX)
    }

    /// The statement that drops the index an earlier release built over
    /// `::jsonb` under [`LEGACY_UNSTORED_INDEX`], valid or not: with it on
    /// the table no row whose object carries `\u0000` can be written.
    /// Concurrently, so no read or write on the table waits on the drop;
    /// the statement cannot run inside a transaction block.
    pub(crate) fn legacy_unstored_index_drop_ddl(schema: &SchemaConfig) -> String {
        format!(
            "DROP INDEX CONCURRENTLY IF EXISTS {};",
            Self::legacy_unstored_index_name(schema)
        )
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

        let tenant: Option<String> = if s.is_skipped("tenant") {
            None
        } else {
            row.get(s.col("tenant"))
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
            tenant,
        })
    }
}

/// The condition of the tenant scope on the rows of a query, with the
/// caller's tenant bound as `$param`: the rows stored under no tenant and
/// those stored under the caller's (see
/// [`tenant_scope_allows`](crate::core::tenant_scope_allows)). `None` when
/// the schema skips the column: every row is served then, as before.
fn tenant_scope_sql(s: &crate::schema::TableConfig, param: usize) -> Option<String> {
    if s.is_skipped("tenant") {
        return None;
    }
    let col = s.col("tenant");
    Some(format!(" AND ({col} IS NULL OR {col} = ${param})"))
}

/// The query behind [`ResponseStorage::find_response_by_output_item`]:
/// `raw_response` is a native `JSON` column, so each row's document is read
/// through [`nul_safe_json_document`] (the cast fails on a row whose object
/// carries `\u0000`, and the predicate runs on every row the scan visits),
/// cast to `jsonb` and asked whether its `output` array contains an item with
/// the id bound as `$1`. `None` when the schema skips the column (nothing to
/// search).
fn find_by_output_item_sql(schema: &SchemaConfig, select_base: &str) -> Option<String> {
    let s = &schema.responses;
    if s.is_skipped("raw_response") {
        return None;
    }
    let document = nul_safe_json_document(s.col("raw_response"));
    let order = if s.is_skipped("created_at") {
        String::new()
    } else {
        format!(" ORDER BY {} DESC", s.col("created_at"))
    };
    Some(format!(
        "{select_base} WHERE {document}::jsonb @> jsonb_build_object('output', \
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

/// The statement of a link: the row is inserted, or, for an item the
/// conversation already has, the row under the (conversation, item) key (the
/// first two columns, the table's primary key) takes the new stamp and the
/// new values of any extra column. A second link of an item moves it, the
/// contract of [`ConversationItemStorage::link_item`], where the key used to
/// refuse the insert.
fn link_sql(table: &str, col_names: &[&str]) -> String {
    let placeholders = param_placeholders(col_names.len());
    let columns = col_names.join(", ");
    let key = col_names
        .iter()
        .take(2)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    let updates: Vec<String> = col_names
        .iter()
        .skip(2)
        .map(|col| format!("{col} = EXCLUDED.{col}"))
        .collect();
    if updates.is_empty() {
        return format!(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT ({key}) DO NOTHING"
        );
    }
    format!(
        "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT ({key}) DO UPDATE SET {}",
        updates.join(", ")
    )
}

/// The name of the conversation item links index.
const LINKS_INDEX: &str = "conv_item_links_conv_idx";
/// The name of the safety-identifier index on the responses table.
const SAFETY_INDEX: &str = "responses_safety_idx";
/// The name of the index the unstored-record sweep reads.
const UNSTORED_INDEX: &str = "responses_unstored_text_idx";
/// The name an earlier release gave that index, built over `::jsonb`.
const LEGACY_UNSTORED_INDEX: &str = "responses_unstored_idx";

/// `index` under the schema's owner when it has one: the name the catalog
/// resolves, as `to_regclass` and `DROP INDEX` take it.
fn qualified_index(schema: &SchemaConfig, index: &str) -> String {
    match schema.owner.as_deref() {
        Some(owner) => format!("{owner}.{index}"),
        None => index.to_string(),
    }
}

/// The JSON document in `col_raw` as text every stored object survives: the
/// column's text with the escape `\u0000` replaced by `\u0020` (a space), to
/// be parsed as `json` or `jsonb` after. A `::jsonb` cast, and `->>` on a
/// `json` value, both fail on an object that carries that escape anywhere,
/// and an index expression is evaluated for every row written and a lookup's
/// predicate for every row scanned; the replacement keeps the text the same
/// length and a literal backslash sequence in it valid, which replacing with
/// nothing would not. The two literals are dollar-quoted: an ordinary literal
/// with a backslash in it reads differently under
/// `standard_conforming_strings = off` (an installation setting the server
/// still accepts: the pattern is refused at parse time, the replacement
/// becomes a space), and the index on a database, the sweep and the lookup
/// must mean the same on every installation.
fn nul_safe_json_document(col_raw: &str) -> String {
    format!("replace({col_raw}::text, $$\\u0000$$, $$\\u0020$$)")
}

/// `field` of the JSON object in `col_raw` as text, read through
/// [`nul_safe_json_document`].
fn nul_safe_json_text(col_raw: &str, field: &str) -> String {
    format!("{}::json->>'{field}'", nul_safe_json_document(col_raw))
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
            tenant,
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
        if !s.is_skipped("tenant") {
            col_names.push(s.col("tenant"));
            params.push(&tenant);
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
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        let store_err = ResponseStorageError::from_pg;
        // The statement is prepared once per pooled connection (the cache
        // keeps it), so a store on the upsert path is one round trip.
        let rows = match self.store_path {
            StorePath::Upsert => execute_cached(&client, &upsert_sql(&table, &col_names), &params)
                .await
                .map_err(store_err)?,
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
                let sql = locked_merge_sql(&table, &col_names);
                let stmt = tx.prepare_cached(&sql).await.map_err(store_err)?;
                // A stale statement fails the transaction (no second try
                // inside it); its entry goes, so the next store on this
                // connection prepares the statement afresh.
                let inserted = tx
                    .execute(&stmt, &params)
                    .await
                    .inspect_err(|error| {
                        if is_stale_statement(error) {
                            drop(tx.statement_cache.remove(&sql, &[]));
                        }
                    })
                    .map_err(store_err)?;
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
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &[&response_id.0.as_str()])
            .await
            .map_err(ResponseStorageError::from_pg)?;
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
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &[&item_id])
            .await
            .map_err(ResponseStorageError::from_pg)?;
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
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        execute_cached(
            &client,
            &format!("DELETE FROM {table} WHERE {col_id} = $1"),
            &[&response_id.0.as_str()],
        )
        .await
        .map_err(ResponseStorageError::from_pg)?;
        Ok(())
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let s = &self.store.schema.responses;

        // safety_identifier must exist to filter by it
        if s.is_skipped("safety_identifier") {
            return Ok(vec![]);
        }

        let col_safety = s.col("safety_identifier");
        let limit_i64: Option<i64> = limit.map(|l| l as i64);

        let mut sql = format!("{} WHERE {col_safety} = $1", self.select_base);
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&identifier];
        if let Some(scope) = tenant_scope_sql(s, params.len() + 1) {
            sql.push_str(&scope);
            params.push(&tenant);
        }
        if !s.is_skipped("created_at") {
            sql.push_str(&format!(" ORDER BY {} DESC", s.col("created_at")));
        }
        if let Some(limit) = &limit_i64 {
            sql.push_str(&format!(" LIMIT ${}", params.len() + 1));
            params.push(limit);
        }

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        let rows = query_cached(&client, &sql, &params)
            .await
            .map_err(ResponseStorageError::from_pg)?;

        let schema = &self.store.schema;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(Self::build_response_from_row(&row, schema)?);
        }

        Ok(out)
    }

    async fn delete_unstored_older_than(&self, cutoff: DateTime<Utc>) -> ResponseResult<usize> {
        // The object (its `store`, `status` and `completed_at`) is what the
        // query reads; without the column there is nothing to read.
        let Some(sql) = Self::unstored_sweep_sql(&self.store.schema) else {
            return Ok(0);
        };

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let cutoff_secs: i64 = cutoff.timestamp();
        let rows_deleted = client
            .execute(&sql, &[&cutoff_secs])
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        Ok(rows_deleted as usize)
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize> {
        let s = &self.store.schema.responses;

        // safety_identifier must exist to filter by it
        if s.is_skipped("safety_identifier") {
            return Ok(0);
        }

        let table = s.qualified_table(self.store.schema.owner.as_deref());
        let col_safety = s.col("safety_identifier");

        let mut sql = format!("DELETE FROM {table} WHERE {col_safety} = $1");
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&identifier];
        if let Some(scope) = tenant_scope_sql(s, params.len() + 1) {
            sql.push_str(&scope);
            params.push(&tenant);
        }

        let client = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        let rows_deleted = execute_cached(&client, &sql, &params)
            .await
            .map_err(ResponseStorageError::from_pg)?;
        Ok(rows_deleted as usize)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The replacement's literals are dollar-quoted, so the index expression,
    /// the sweep and the lookup read the same under
    /// `standard_conforming_strings = off` as under `on`: an ordinary literal
    /// with a backslash in it is re-read under `off` (the pattern refused, the
    /// replacement a space).
    #[test]
    fn the_nul_replacement_s_literals_read_the_same_under_either_string_setting() {
        let document = nul_safe_json_document("raw_response");
        assert_eq!(
            document,
            "replace(raw_response::text, $$\\u0000$$, $$\\u0020$$)"
        );
        assert!(!document.contains("'\\"), "{document}");
    }

    /// The sweep's statement reads `store` and `status` through the very
    /// expressions its index is built on (a drift between the two would leave
    /// the planner without the index, and the sweep scanning the table), and
    /// the lookup by output item reads the document through the same NUL-safe
    /// form.
    #[test]
    fn the_sweep_s_predicate_is_the_index_s_expressions() {
        let schema = SchemaConfig::default();
        let ddl = PostgresResponseStorage::unstored_index_ddl(&schema)
            .expect("the default schema keeps the object column");
        let sweep = PostgresResponseStorage::unstored_sweep_sql(&schema)
            .expect("the default schema keeps the object column");
        let col_raw = schema.responses.col("raw_response");
        let status = nul_safe_json_text(col_raw, "status");
        let store = nul_safe_json_text(col_raw, "store");
        assert!(ddl.contains(&format!("(({status}))")), "{ddl}");
        assert!(ddl.contains(&format!("WHERE {store} = 'false'")), "{ddl}");
        assert!(
            sweep.contains(&format!("WHERE {store} = 'false'")),
            "{sweep}"
        );
        assert!(sweep.contains(&format!("AND {status} IN (")), "{sweep}");
        assert!(!sweep.contains("::jsonb"), "{sweep}");
        let lookup = find_by_output_item_sql(&schema, "SELECT id FROM responses")
            .expect("the default schema keeps the object column");
        assert!(
            lookup.contains(&format!("{}::jsonb @>", nul_safe_json_document(col_raw))),
            "{lookup}"
        );
    }

    /// The earlier index is dropped concurrently (a plain drop holds an
    /// ACCESS EXCLUSIVE lock on the live table to the end of its
    /// transaction), under the schema's owner when it has one, as are the
    /// indexes a start looks up before asking for them.
    #[test]
    fn the_earlier_index_is_dropped_concurrently() {
        let schema = SchemaConfig::default();
        assert_eq!(
            PostgresResponseStorage::legacy_unstored_index_drop_ddl(&schema),
            "DROP INDEX CONCURRENTLY IF EXISTS responses_unstored_idx;"
        );
        let owned = SchemaConfig {
            owner: Some("history".to_string()),
            ..Default::default()
        };
        assert_eq!(
            PostgresResponseStorage::legacy_unstored_index_drop_ddl(&owned),
            "DROP INDEX CONCURRENTLY IF EXISTS history.responses_unstored_idx;"
        );
        assert_eq!(
            PostgresConversationItemStorage::links_index_name(&owned),
            "history.conv_item_links_conv_idx"
        );
        assert_eq!(
            PostgresResponseStorage::safety_index_name(&schema),
            "responses_safety_idx"
        );
    }

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
    fn link_sql_moves_an_item_the_conversation_already_has() {
        assert_eq!(
            link_sql(
                "conversation_item_links",
                &["conversation_id", "item_id", "added_at", "tenant"]
            ),
            "INSERT INTO conversation_item_links (conversation_id, item_id, added_at, tenant) \
             VALUES ($1, $2, $3, $4) ON CONFLICT (conversation_id, item_id) \
             DO UPDATE SET added_at = EXCLUDED.added_at, tenant = EXCLUDED.tenant"
        );
        // The key alone leaves nothing to move.
        assert_eq!(
            link_sql("l", &["conversation_id", "item_id"]),
            "INSERT INTO l (conversation_id, item_id) VALUES ($1, $2) \
             ON CONFLICT (conversation_id, item_id) DO NOTHING"
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
    fn the_tenant_scope_condition_serves_untenanted_rows_and_the_callers_own() {
        let s = crate::schema::TableConfig::with_table("responses");
        assert_eq!(
            tenant_scope_sql(&s, 2).as_deref(),
            Some(" AND (tenant IS NULL OR tenant = $2)")
        );
        let mut skipped = crate::schema::TableConfig::with_table("responses");
        skipped.skip_columns.insert("tenant".to_string());
        assert_eq!(tenant_scope_sql(&skipped, 2), None);
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

    /// The two answers that mean the statement the server was handed is
    /// stale (its cached plan no longer fits, raised by the server's
    /// cached-plan revalidation; it no longer exists, whatever raised it),
    /// and no other, make a helper prepare the statement afresh and call
    /// again: `0A000` from any other routine is the broad class it is (a
    /// feature the server does not support) and is answered as it came.
    #[test]
    fn a_stale_plan_and_a_missing_statement_are_the_stale_states() {
        assert!(is_stale_statement_state(
            &SqlState::FEATURE_NOT_SUPPORTED,
            Some("RevalidateCachedQuery")
        ));
        assert!(is_stale_statement_state(
            &SqlState::INVALID_SQL_STATEMENT_NAME,
            None
        ));
        assert!(is_stale_statement_state(
            &SqlState::INVALID_SQL_STATEMENT_NAME,
            Some("FetchPreparedStatement")
        ));
        assert!(!is_stale_statement_state(
            &SqlState::FEATURE_NOT_SUPPORTED,
            None
        ));
        assert!(!is_stale_statement_state(
            &SqlState::FEATURE_NOT_SUPPORTED,
            Some("infer_arbiter_indexes")
        ));
        assert!(!is_stale_statement_state(&SqlState::UNIQUE_VIOLATION, None));
        assert!(!is_stale_statement_state(
            &SqlState::IN_FAILED_SQL_TRANSACTION,
            None
        ));
        assert!(!is_stale_statement_state(&SqlState::UNDEFINED_COLUMN, None));
        assert!(!is_stale_statement_state(&SqlState::QUERY_CANCELED, None));
    }
}
