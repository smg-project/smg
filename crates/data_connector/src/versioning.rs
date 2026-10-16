//! Schema versioning and migration infrastructure.
//!
//! Replaces ad-hoc `ALTER TABLE` / column-existence checks with a tracked,
//! per-backend migration system.  Each backend defines its own migration list
//! (DDL syntax differs across databases) and migrations reference
//! [`SchemaConfig`] so they work correctly even with custom table/column names.
//!
//! SMG migrations are intentionally forward-only. The runner records applied
//! versions and can auto-apply pending `up` steps, but it does not model or
//! execute down-migrations.
//!
//! # Version tracking
//!
//! A `_schema_versions` table records which migrations have been applied.
//! On startup the runner reads the current version and applies any pending
//! migrations in order.
//!
//! # Safety
//!
//! `auto_migrate` defaults to `false`. When pending migrations are detected
//! and auto-migration is off, startup **fails** with the exact SQL statements
//! needed so the operator can review and apply them manually.
//!
//! # Configuration
//!
//! ```yaml
//! oracle:
//!   schema:
//!     version: 2           # "my schema is already at v2, skip 1-2"
//!     auto_migrate: true    # opt in to automatic migration
//! ```

use tokio_postgres::GenericClient;

use crate::schema::SchemaConfig;

// ── Types ──────────────────────────────────────────────────────────────────

/// A single forward-only schema migration.
///
/// Migrations are functions (not static SQL strings) so they can reference
/// [`SchemaConfig`] for table/column names. `up` returns a `Vec<String>`
/// because some migrations require multiple DDL statements (e.g. ALTER TABLE
/// followed by CREATE INDEX).
pub struct Migration {
    /// Monotonically increasing version number (1, 2, 3, …).
    pub version: u32,
    /// Human-readable description for the version log.
    pub description: &'static str,
    /// Generate the forward-migration DDL statements.
    pub up: fn(&SchemaConfig) -> Vec<String>,
}

// ── Versions table DDL ─────────────────────────────────────────────────────

/// Name of the schema-versions tracking table.
pub const VERSIONS_TABLE: &str = "_schema_versions";

/// The advisory lock key under which a gateway creates and migrates the
/// Postgres history schema (`"smg_migr"`). The whole bootstrap of a
/// gateway runs in one transaction that holds it, so gateways starting
/// together on one database take turns; the key is the one earlier builds
/// hold at session level while they migrate, and the two kinds exclude each
/// other.
pub(crate) const POSTGRES_SCHEMA_LOCK_ID: i64 = 0x736D_675F_6D69_6772;

/// Oracle-qualified name for the versions table (always quoted since `_` prefix
/// is invalid for unquoted Oracle identifiers).
fn oracle_versions_table(schema: &SchemaConfig) -> String {
    match &schema.owner {
        Some(owner) => format!("{owner}.\"{VERSIONS_TABLE}\""),
        None => format!("\"{VERSIONS_TABLE}\""),
    }
}

/// Oracle DDL for creating the versions tracking table.
pub fn oracle_create_versions_table(schema: &SchemaConfig) -> String {
    let table = oracle_versions_table(schema);
    format!(
        "CREATE TABLE {table} (\
         version NUMBER(10) NOT NULL PRIMARY KEY, \
         description VARCHAR2(512), \
         applied_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL)"
    )
}

/// Postgres DDL for creating the versions tracking table.
pub fn postgres_create_versions_table() -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {VERSIONS_TABLE} (\
         version INTEGER NOT NULL PRIMARY KEY, \
         description VARCHAR(512), \
         applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
    )
}

// ── Pending-migration error ───────────────────────────────────────────────

/// Build an actionable error message listing pending migrations and their SQL.
fn pending_migrations_error(
    backend: &str,
    current: u32,
    pending: &[&Migration],
    schema: &SchemaConfig,
) -> String {
    let mut msg = format!(
        "Schema migration required (current version: {current}, \
         latest version: {}).\n\n\
         The following migrations need to be applied:\n",
        pending.last().map_or(current, |m| m.version)
    );

    let versions_insert = if backend == "oracle" {
        format!(
            "INSERT INTO {} (version, description) VALUES",
            oracle_versions_table(schema),
        )
    } else {
        format!("INSERT INTO {VERSIONS_TABLE} (version, description) VALUES")
    };

    for m in pending {
        msg.push_str(&format!("\n  v{}: {}\n", m.version, m.description));
        let stmts = (m.up)(schema);
        for stmt in &stmts {
            if !stmt.is_empty() {
                msg.push_str(&format!("    {stmt}\n"));
            }
        }
        // Include the version-tracking INSERT so operators record the
        // migration after applying the DDL manually.
        msg.push_str(&format!(
            "    {versions_insert} ({}, '{}');\n",
            m.version, m.description,
        ));
    }

    msg.push_str(&format!(
        "\nTo apply automatically, set `auto_migrate: true` in your {backend} schema config \
         (the YAML file given by `--schema-config`, or the `{backend}.schema` block of the \
         gateway's config), or set the environment variable `DB_AUTO_MIGRATE=true`.\n\
         To skip, set `version: {}` in that schema config to mark your schema as already up to date.",
        pending.last().map_or(current, |m| m.version)
    ));

    msg
}

// ── A schema ahead of this build ───────────────────────────────────────────

/// Refuses a store whose schema version is ahead of the migrations this
/// build knows: an older gateway behind a store that a newer build migrated
/// forward (a rollback of the gateway image is the usual case) would start,
/// then fail its first read of a changed column. The schema config's
/// `version` at or above the store's version is the operator's word that this
/// build may run against it; it is logged once.
pub(crate) fn schema_ahead_check(
    backend: &str,
    current: u32,
    migrations: &[Migration],
    config_version: Option<u32>,
) -> Result<(), String> {
    if schema_ahead_override(backend, current, migrations, config_version)? {
        tracing::warn!(
            backend,
            schema_version = current,
            latest_known = migrations.last().map_or(0, |m| m.version),
            "the schema is ahead of the migrations this build knows; running on the schema config's `version`"
        );
    }
    Ok(())
}

/// The verdict of [`schema_ahead_check`] without its log line: `Ok(false)`
/// when the schema is not ahead of this build, `Ok(true)` when it is and the
/// schema config's `version` lets this build run on it, the refusal
/// otherwise. For a re-check after the check that logged.
pub(crate) fn schema_ahead_override(
    backend: &str,
    current: u32,
    migrations: &[Migration],
    config_version: Option<u32>,
) -> Result<bool, String> {
    let latest = migrations.last().map_or(0, |m| m.version);
    if current <= latest {
        return Ok(false);
    }
    if config_version.is_some_and(|version| version >= current) {
        return Ok(true);
    }
    Err(format!(
        "The {backend} schema is at version {current}, ahead of the latest migration this build knows \
         (v{latest}). This build does not know the shape of a newer schema: upgrade the gateway to the \
         build that migrated the database, or set `version: {current}` in the {backend} schema config to \
         run this build against it anyway."
    ))
}

// ── Oracle helpers ─────────────────────────────────────────────────────────

/// What an Oracle store's start-up does to its database, as the steps
/// [`bootstrap_oracle_schema`] orders. The store runs them on its connection;
/// the unit tests run them on a fake that records their order.
pub(crate) trait OracleSchemaSteps {
    /// The highest recorded schema version, `None` when the versions table
    /// does not exist. A read: nothing is created before the verdict on it.
    fn recorded_version(&self) -> Result<Option<u32>, String>;
    /// Creates the history tables and indexes that do not exist yet.
    fn create_tables(&self) -> Result<(), String>;
    /// Creates the versions table when it is missing, applies the pending
    /// `migrations` (or refuses to start on them; `config_version` is the
    /// schema config's `version`) and returns the versions it applied.
    fn migrate(
        &self,
        migrations: &[Migration],
        config_version: Option<u32>,
    ) -> Result<Vec<u32>, String>;
}

/// Brings an Oracle database to the schema this build expects, in the order
/// that leaves a database this build does not understand untouched: the
/// recorded version against this build's migrations first, so a database a
/// newer build migrated forward is refused before any history table is
/// created or indexed (a table a newer migration renamed or dropped would
/// otherwise be re-created in its legacy shape, and an index on a column it
/// dropped would fail with the server's error instead of the refusal, see
/// [`schema_ahead_check`]); a database without a versions table is at version
/// 0, which no build is behind. Then the history tables, then the migrations,
/// then the tables once more when migrations were applied, so that the
/// indexes on the columns they added are created in this start rather than
/// the next. Oracle DDL commits as it runs, so there is no transaction to roll
/// the steps back into: the order is what keeps the refused database as it
/// was.
pub(crate) fn bootstrap_oracle_schema(
    steps: &impl OracleSchemaSteps,
    migrations: &[Migration],
    config_version: Option<u32>,
) -> Result<(), String> {
    if let Some(current) = steps.recorded_version()? {
        schema_ahead_check("oracle", current, migrations, config_version)?;
    }
    steps.create_tables()?;
    let applied = steps.migrate(migrations, config_version)?;
    if !applied.is_empty() {
        // The initializers once more: the indexes on the columns the
        // migrations added (e.g. the safety identifier's, v1) are created in
        // this start rather than the next.
        steps.create_tables()?;
    }
    Ok(())
}

/// Run pending Oracle migrations on a synchronous `oracle::Connection`: the
/// `migrate` step of [`bootstrap_oracle_schema`], which checked the recorded
/// version against this build's migrations before the tables were created.
///
/// Returns the list of newly applied version numbers.
pub fn run_oracle_migrations(
    conn: &oracle::Connection,
    schema: &SchemaConfig,
    migrations: &[Migration],
    config_version: Option<u32>,
    auto_migrate: bool,
) -> Result<Vec<u32>, String> {
    // Ensure the versions table exists (needed to check current version)
    ensure_oracle_versions_table(conn, schema)?;

    let current = oracle_current_version(conn, schema)?;
    // A newer build may have migrated the database between the start-up's
    // early check (`bootstrap_oracle_schema`, before the tables were created)
    // and this read -- a rolling update starts old and new builds together,
    // and Oracle has no schema lock to serialize them -- so the version is
    // judged again here, before this build serves on it. The override's
    // warning was logged by the early check; this one only refuses.
    schema_ahead_override("oracle", current, migrations, config_version)?;
    let skip_up_to = config_version.unwrap_or(0);
    let effective_start = current.max(skip_up_to);

    let pending: Vec<&Migration> = migrations
        .iter()
        .filter(|m| m.version > effective_start)
        .collect();

    if pending.is_empty() {
        tracing::debug!(current_version = effective_start, "schema is up to date");
        return Ok(Vec::new());
    }

    // When auto_migrate is off, fail with actionable info
    if !auto_migrate {
        return Err(pending_migrations_error(
            "oracle",
            effective_start,
            &pending,
            schema,
        ));
    }

    tracing::info!(
        current_version = effective_start,
        pending = pending.len(),
        "applying schema migrations"
    );

    let versions_table = oracle_versions_table(schema);

    let mut applied = Vec::new();
    for migration in pending {
        // NOTE: Oracle DDL implicitly commits. If the DDL below succeeds but
        // the subsequent INSERT into _schema_versions fails (for a reason
        // other than ORA-00001), the schema change is committed without a
        // version record.  Next startup will re-attempt the migration, so
        // all Oracle migrations MUST be idempotent (e.g. use PL/SQL
        // EXCEPTION handlers to tolerate "column already exists").
        let stmts = (migration.up)(schema);
        for stmt in &stmts {
            if stmt.is_empty() {
                continue;
            }
            conn.execute(stmt, &[]).map_err(|e| {
                format!(
                    "migration v{} ({}) failed: {}",
                    migration.version, migration.description, e
                )
            })?;
        }
        // Record the applied migration.
        // ORA-00001 (unique constraint) means another instance already
        // applied this migration concurrently — safe to skip since the
        // DDL statements above are idempotent.
        match conn.execute(
            &format!("INSERT INTO {versions_table} (version, description) VALUES (:1, :2)"),
            &[&migration.version, &migration.description],
        ) {
            Ok(_) => {
                conn.commit().map_err(|e| format!("commit failed: {e}"))?;
            }
            Err(e) if e.db_error().is_some_and(|de| de.code() == 1) => {
                tracing::info!(
                    version = migration.version,
                    "migration already applied by another instance, skipping"
                );
                continue;
            }
            Err(e) => {
                return Err(format!(
                    "failed to record migration v{}: {}",
                    migration.version, e
                ));
            }
        }

        tracing::info!(
            version = migration.version,
            description = migration.description,
            "applied migration"
        );
        applied.push(migration.version);
    }

    let final_version = oracle_current_version(conn, schema)?;
    tracing::info!(
        schema_version = final_version,
        "schema version after migrations"
    );

    Ok(applied)
}

/// The query that counts the `_schema_versions` table in Oracle's catalog:
/// `all_tables` under the configured owner (the catalog holds owners in
/// upper case), `user_tables` for the current user's schema otherwise.
///
/// The table is always created with a quoted identifier since `_` is not
/// valid as the first character of an unquoted Oracle identifier. This
/// preserves the lowercase name in the catalog, which is the name looked up.
pub(crate) fn oracle_versions_table_exists_sql(schema: &SchemaConfig) -> String {
    match &schema.owner {
        Some(owner) => format!(
            "SELECT COUNT(*) FROM all_tables WHERE owner = '{}' AND table_name = '{VERSIONS_TABLE}'",
            owner.to_ascii_uppercase()
        ),
        None => {
            format!("SELECT COUNT(*) FROM user_tables WHERE table_name = '{VERSIONS_TABLE}'")
        }
    }
}

/// Whether the `_schema_versions` table exists in the Oracle database `conn`
/// is connected to.
fn oracle_versions_table_exists(
    conn: &oracle::Connection,
    schema: &SchemaConfig,
) -> Result<bool, String> {
    let present: i64 = conn
        .query_row_as(&oracle_versions_table_exists_sql(schema), &[])
        .map_err(|e| format!("failed to check for {VERSIONS_TABLE} table: {e}"))?;
    Ok(present > 0)
}

/// The highest schema version an Oracle database records, `None` when it has
/// no versions table (no build has versioned it yet): the `recorded_version`
/// step of [`bootstrap_oracle_schema`], a read that creates nothing.
pub(crate) fn oracle_recorded_version(
    conn: &oracle::Connection,
    schema: &SchemaConfig,
) -> Result<Option<u32>, String> {
    if !oracle_versions_table_exists(conn, schema)? {
        return Ok(None);
    }
    oracle_current_version(conn, schema).map(Some)
}

/// Ensure the `_schema_versions` table exists in Oracle (see
/// [`oracle_versions_table_exists_sql`] for the lookup and the quoting).
fn ensure_oracle_versions_table(
    conn: &oracle::Connection,
    schema: &SchemaConfig,
) -> Result<(), String> {
    if !oracle_versions_table_exists(conn, schema)? {
        let ddl = oracle_create_versions_table(schema);
        if let Err(err) = conn.execute(&ddl, &[]) {
            // ORA-00955: name is already used — another instance created
            // the table between our check and this CREATE. Safe to ignore.
            if err.db_error().is_some_and(|de| de.code() == 955) {
                tracing::debug!("versions table created by concurrent instance, proceeding");
            } else {
                return Err(format!("failed to create {VERSIONS_TABLE} table: {err}"));
            }
        }
        conn.commit().map_err(|e| format!("commit failed: {e}"))?;
    }
    Ok(())
}

/// Read the current (highest) schema version from the Oracle versions table.
fn oracle_current_version(conn: &oracle::Connection, schema: &SchemaConfig) -> Result<u32, String> {
    let versions_table = oracle_versions_table(schema);
    let row: Option<i64> = conn
        .query_row_as_named(&format!("SELECT MAX(version) FROM {versions_table}"), &[])
        .map_err(|e| format!("failed to read current schema version: {e}"))?;

    Ok(row.unwrap_or(0) as u32)
}

// ── Postgres helpers ───────────────────────────────────────────────────────

/// Run pending Postgres migrations inside `tx`, the caller's bootstrap
/// transaction (which holds [`POSTGRES_SCHEMA_LOCK_ID`] and created the
/// versions table). Every pending migration and its version record go in
/// that one transaction: a failure leaves the database as it was. Returns the
/// newly applied version numbers.
pub async fn run_postgres_migrations(
    tx: &tokio_postgres::Transaction<'_>,
    schema: &SchemaConfig,
    migrations: &[Migration],
    config_version: Option<u32>,
    auto_migrate: bool,
) -> Result<Vec<u32>, String> {
    let current = postgres_current_version(tx).await?;
    let skip_up_to = config_version.unwrap_or(0);
    let effective_start = current.max(skip_up_to);

    let pending: Vec<&Migration> = migrations
        .iter()
        .filter(|m| m.version > effective_start)
        .collect();

    if pending.is_empty() {
        tracing::debug!(current_version = effective_start, "schema is up to date");
        return Ok(Vec::new());
    }

    // When auto_migrate is off, fail with actionable info
    if !auto_migrate {
        return Err(pending_migrations_error(
            "postgres",
            effective_start,
            &pending,
            schema,
        ));
    }

    tracing::info!(
        current_version = effective_start,
        pending = pending.len(),
        "applying schema migrations"
    );

    let mut applied = Vec::new();
    for migration in pending {
        // DDL in Postgres is transactional (unlike Oracle): the statements
        // and the version record commit with the bootstrap or not at all.
        let stmts = (migration.up)(schema);
        for stmt in &stmts {
            if stmt.is_empty() {
                continue;
            }
            tx.batch_execute(stmt).await.map_err(|e| {
                format!(
                    "migration v{} ({}) failed: {}",
                    migration.version,
                    migration.description,
                    crate::common::describe_error(&e)
                )
            })?;
        }

        // Record the applied migration
        tx.execute(
            &format!("INSERT INTO {VERSIONS_TABLE} (version, description) VALUES ($1, $2)"),
            &[&(migration.version as i32), &migration.description],
        )
        .await
        .map_err(|e| {
            format!(
                "failed to record migration v{}: {}",
                migration.version,
                crate::common::describe_error(&e)
            )
        })?;

        tracing::info!(
            version = migration.version,
            description = migration.description,
            "applied migration"
        );
        applied.push(migration.version);
    }

    let final_version = postgres_current_version(tx).await?;
    tracing::info!(
        schema_version = final_version,
        "schema version after migrations"
    );

    Ok(applied)
}

/// The statement that records one migration as applied unless a record
/// exists: a fresh database is stamped with every migration at once, and a
/// database another gateway stamped first is left as it is.
pub fn postgres_stamp_version_sql() -> String {
    format!(
        "INSERT INTO {VERSIONS_TABLE} (version, description) VALUES ($1, $2) \
         ON CONFLICT (version) DO NOTHING"
    )
}

/// Records every migration in `migrations` as applied, inside `tx`: the
/// tables a first start just created are in their final shape, so the
/// migrations that describe that shape have nothing left to do.
pub async fn stamp_postgres_versions(
    tx: &tokio_postgres::Transaction<'_>,
    migrations: &[Migration],
) -> Result<(), String> {
    let sql = postgres_stamp_version_sql();
    for migration in migrations {
        tx.execute(&sql, &[&(migration.version as i32), &migration.description])
            .await
            .map_err(|e| {
                format!(
                    "failed to record migration v{} on the fresh database: {}",
                    migration.version,
                    crate::common::describe_error(&e)
                )
            })?;
    }
    Ok(())
}

/// The query that tells whether a (possibly schema-qualified) table exists,
/// as `$1`, resolved against the connection's search path.
pub fn postgres_table_exists_sql() -> &'static str {
    "SELECT to_regclass($1) IS NOT NULL"
}

/// Whether `table` exists in the database `tx` is connected to.
pub async fn postgres_table_exists(
    tx: &tokio_postgres::Transaction<'_>,
    table: &str,
) -> Result<bool, String> {
    let row = tx
        .query_one(postgres_table_exists_sql(), &[&table])
        .await
        .map_err(|e| {
            format!(
                "failed to check for the {table} table: {}",
                crate::common::describe_error(&e)
            )
        })?;
    Ok(row.get(0))
}

/// Read the current (highest) schema version from the Postgres versions table.
pub(crate) async fn postgres_current_version(client: &impl GenericClient) -> Result<u32, String> {
    let row = client
        .query_one(
            &format!("SELECT COALESCE(MAX(version), 0) FROM {VERSIONS_TABLE}"),
            &[],
        )
        .await
        .map_err(|e| {
            format!(
                "failed to read current schema version: {}",
                crate::common::describe_error(&e)
            )
        })?;

    let version: i32 = row.get(0);
    Ok(version as u32)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::schema::TableConfig;

    /// The start-up steps on a database that records `recorded` (`None`: no
    /// versions table) and whose migrate step applies `applies`; `ran` is
    /// the order the steps ran in.
    struct RecordedSteps {
        recorded: Option<u32>,
        applies: Vec<u32>,
        ran: RefCell<Vec<&'static str>>,
    }

    impl RecordedSteps {
        fn on(recorded: Option<u32>, applies: Vec<u32>) -> Self {
            Self {
                recorded,
                applies,
                ran: RefCell::new(Vec::new()),
            }
        }

        fn ran(&self) -> Vec<&'static str> {
            self.ran.borrow().clone()
        }
    }

    impl OracleSchemaSteps for RecordedSteps {
        fn recorded_version(&self) -> Result<Option<u32>, String> {
            self.ran.borrow_mut().push("read the version");
            Ok(self.recorded)
        }

        fn create_tables(&self) -> Result<(), String> {
            self.ran.borrow_mut().push("create the tables");
            Ok(())
        }

        fn migrate(
            &self,
            _migrations: &[Migration],
            _config_version: Option<u32>,
        ) -> Result<Vec<u32>, String> {
            self.ran.borrow_mut().push("migrate");
            Ok(self.applies.clone())
        }
    }

    fn three_migrations() -> Vec<Migration> {
        (1..=3)
            .map(|version| Migration {
                version,
                description: "test",
                up: |_| vec![],
            })
            .collect()
    }

    /// A database whose recorded version is ahead of the migrations this
    /// build knows is refused before any history table is created or
    /// indexed: the version is the only thing the start-up read.
    #[test]
    fn an_oracle_database_ahead_of_this_build_is_refused_before_its_tables_are_touched() {
        let steps = RecordedSteps::on(Some(99), vec![]);
        let message = bootstrap_oracle_schema(&steps, &three_migrations(), None)
            .expect_err("a database ahead of this build must be refused at start-up");
        assert!(
            message.contains("version 99") && message.contains("ahead of the latest migration"),
            "{message}"
        );
        assert_eq!(
            steps.ran(),
            ["read the version"],
            "the refusal must come before the tables are touched"
        );
    }

    /// The schema config's `version` at the database's version is the
    /// operator's word that this build may run against it: the start-up goes
    /// on, in its order.
    #[test]
    fn the_version_setting_lets_this_build_start_on_a_newer_oracle_database() {
        let steps = RecordedSteps::on(Some(99), vec![]);
        bootstrap_oracle_schema(&steps, &three_migrations(), Some(99))
            .expect("`version: 99` lets the build run against the newer database");
        assert_eq!(
            steps.ran(),
            ["read the version", "create the tables", "migrate"]
        );
    }

    /// A database without a versions table (nothing is ahead of it) or at a
    /// version this build knows gets its tables, then its migrations, then
    /// its tables once more only when a migration was applied (the indexes
    /// on the columns it added).
    #[test]
    fn an_oracle_database_this_build_knows_gets_its_tables_then_its_migrations() {
        let settled = ["read the version", "create the tables", "migrate"];
        let migrated = [
            "read the version",
            "create the tables",
            "migrate",
            "create the tables",
        ];
        for (recorded, applies, expected) in [
            (None, vec![], &settled[..]),
            (Some(3), vec![], &settled[..]),
            (Some(0), vec![1, 2, 3], &migrated[..]),
            (Some(1), vec![2, 3], &migrated[..]),
        ] {
            let steps = RecordedSteps::on(recorded, applies);
            bootstrap_oracle_schema(&steps, &three_migrations(), None)
                .expect("a database this build knows comes up");
            assert_eq!(steps.ran(), expected, "recorded version {recorded:?}");
        }
    }

    /// The version is read from the catalog's lowercase, quoted name: under
    /// the configured owner (upper-cased, as the catalog holds it) or in the
    /// current user's schema.
    #[test]
    fn the_versions_table_lookup_follows_the_owner() {
        let no_owner = SchemaConfig::default();
        assert_eq!(
            oracle_versions_table_exists_sql(&no_owner),
            "SELECT COUNT(*) FROM user_tables WHERE table_name = '_schema_versions'"
        );
        let with_owner = SchemaConfig {
            owner: Some("admin".to_string()),
            ..Default::default()
        };
        assert_eq!(
            oracle_versions_table_exists_sql(&with_owner),
            "SELECT COUNT(*) FROM all_tables WHERE owner = 'ADMIN' AND table_name = '_schema_versions'"
        );
    }

    /// The re-check in `run_oracle_migrations` is the ahead check's verdict
    /// without its log: not ahead, ahead under the config's `version`, or
    /// the same refusal.
    #[test]
    fn the_ahead_verdict_is_the_same_with_and_without_the_log() {
        let migrations = three_migrations();
        assert_eq!(
            schema_ahead_override("oracle", 3, &migrations, None),
            Ok(false)
        );
        assert_eq!(
            schema_ahead_override("oracle", 4, &migrations, Some(4)),
            Ok(true)
        );
        let refused = schema_ahead_override("oracle", 4, &migrations, None)
            .expect_err("a schema ahead of this build is refused");
        assert_eq!(
            schema_ahead_check("oracle", 4, &migrations, None),
            Err(refused.clone())
        );
        assert!(
            refused.contains("version 4") && refused.contains("ahead of the latest migration"),
            "{refused}"
        );
    }

    #[test]
    fn oracle_versions_table_name_is_always_quoted() {
        let no_owner = SchemaConfig::default();
        assert_eq!(oracle_versions_table(&no_owner), "\"_schema_versions\"");

        let with_owner = SchemaConfig {
            owner: Some("ADMIN".to_string()),
            ..Default::default()
        };
        assert_eq!(
            oracle_versions_table(&with_owner),
            "ADMIN.\"_schema_versions\""
        );
    }

    #[test]
    fn oracle_versions_table_ddl_without_owner() {
        let schema = SchemaConfig::default();
        let ddl = oracle_create_versions_table(&schema);
        assert!(
            ddl.contains("\"_schema_versions\""),
            "must be quoted for Oracle: {ddl}"
        );
        assert!(ddl.contains("PRIMARY KEY"));
    }

    #[test]
    fn oracle_versions_table_ddl_with_owner() {
        let schema = SchemaConfig {
            owner: Some("ADMIN".to_string()),
            ..Default::default()
        };
        let ddl = oracle_create_versions_table(&schema);
        assert!(ddl.contains("ADMIN.\"_schema_versions\""), "got: {ddl}");
    }

    /// A fresh database is stamped with every migration; a stamp that meets a
    /// record (another gateway, a table re-created under a stamped database)
    /// leaves it as it is.
    #[test]
    fn fresh_database_stamp_leaves_an_existing_record_as_it_is() {
        let sql = postgres_stamp_version_sql();
        assert!(sql.starts_with("INSERT INTO _schema_versions"), "{sql}");
        assert!(sql.ends_with("ON CONFLICT (version) DO NOTHING"), "{sql}");
        assert_eq!(
            postgres_table_exists_sql(),
            "SELECT to_regclass($1) IS NOT NULL"
        );
    }

    #[test]
    fn postgres_versions_table_ddl() {
        let ddl = postgres_create_versions_table();
        assert!(ddl.contains("IF NOT EXISTS"));
        assert!(ddl.contains("_schema_versions"));
        assert!(ddl.contains("PRIMARY KEY"));
    }

    #[test]
    fn migration_up_respects_schema_config() {
        let schema = SchemaConfig {
            owner: Some("ADMIN".to_string()),
            responses: TableConfig {
                table: "MY_RESPONSES".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let m = Migration {
            version: 1,
            description: "test",
            up: |s| {
                let t = s.responses.qualified_table(s.owner.as_deref());
                vec![format!("ALTER TABLE {t} ADD COLUMN x INT")]
            },
        };
        let stmts = (m.up)(&schema);
        assert!(
            stmts[0].contains("ADMIN.\"MY_RESPONSES\""),
            "got: {}",
            stmts[0]
        );
    }

    #[test]
    fn pending_migrations_error_includes_sql_and_hints() {
        let schema = SchemaConfig::default();
        let migrations = [
            Migration {
                version: 1,
                description: "add col_x",
                up: |_| vec!["ALTER TABLE t ADD COLUMN x INT".to_string()],
            },
            Migration {
                version: 2,
                description: "drop col_y",
                up: |_| vec!["ALTER TABLE t DROP COLUMN y".to_string()],
            },
        ];
        let pending: Vec<&Migration> = migrations.iter().collect();
        let err = pending_migrations_error("postgres", 0, &pending, &schema);

        assert!(err.contains("v1: add col_x"), "should list v1: {err}");
        assert!(err.contains("v2: drop col_y"), "should list v2: {err}");
        assert!(
            err.contains("ALTER TABLE t ADD COLUMN x INT"),
            "should include SQL: {err}"
        );
        assert!(
            err.contains("INSERT INTO _schema_versions"),
            "should include version INSERT: {err}"
        );
        assert!(
            err.contains("auto_migrate: true"),
            "should hint auto_migrate: {err}"
        );
        assert!(
            err.contains("version: 2"),
            "should hint version skip: {err}"
        );
    }

    /// The refusal names every way to opt in: the YAML key with the flag
    /// that loads its file, and the environment variable.
    #[test]
    fn pending_migrations_error_names_every_way_to_opt_in() {
        let schema = SchemaConfig::default();
        let migrations = [Migration {
            version: 1,
            description: "test",
            up: |_| vec!["SELECT 1".to_string()],
        }];
        let pending: Vec<&Migration> = migrations.iter().collect();
        let err = pending_migrations_error("postgres", 0, &pending, &schema);
        for needle in [
            "auto_migrate: true",
            "--schema-config",
            "DB_AUTO_MIGRATE=true",
        ] {
            assert!(
                err.contains(needle),
                "the refusal should name {needle}: {err}"
            );
        }
    }

    /// A store at or below this build's latest migration passes; one ahead
    /// of it is refused with the versions and the override named, unless the
    /// schema config's `version` reaches the store's.
    #[test]
    fn a_schema_ahead_of_this_build_is_refused_unless_the_version_setting_reaches_it() {
        let migrations: Vec<Migration> = (1..=3)
            .map(|version| Migration {
                version,
                description: "test",
                up: |_| vec![],
            })
            .collect();
        assert!(schema_ahead_check("postgres", 0, &migrations, None).is_ok());
        assert!(schema_ahead_check("postgres", 3, &migrations, None).is_ok());
        let err = schema_ahead_check("postgres", 99, &migrations, None)
            .expect_err("a schema ahead of the build is refused");
        for needle in [
            "version 99",
            "(v3)",
            "`version: 99`",
            "postgres schema config",
        ] {
            assert!(
                err.contains(needle),
                "the refusal should name {needle}: {err}"
            );
        }
        assert!(schema_ahead_check("oracle", 99, &migrations, Some(50)).is_err());
        assert!(schema_ahead_check("oracle", 99, &migrations, Some(99)).is_ok());
        assert!(schema_ahead_check("oracle", 99, &migrations, Some(100)).is_ok());
    }

    #[test]
    fn pending_migrations_error_shows_current_version() {
        let schema = SchemaConfig::default();
        let migrations = [Migration {
            version: 3,
            description: "test",
            up: |_| vec!["SELECT 1".to_string()],
        }];
        let pending: Vec<&Migration> = migrations.iter().collect();
        let err = pending_migrations_error("oracle", 2, &pending, &schema);

        assert!(
            err.contains("current version: 2"),
            "should show current version: {err}"
        );
        assert!(
            err.contains("latest version: 3"),
            "should show target version: {err}"
        );
    }
}
