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

/// The prelude the two runners share: the migrations above the recorded
/// version are pending, the schema config's `version` is a floor under
/// it, nothing pending is `None`, and pending migrations without
/// automatic migration are the refusal that names them.
#[test]
fn pending_migrations_follow_the_recorded_version_the_floor_and_the_automatic_setting() {
    let schema = SchemaConfig::default();
    let migrations = three_migrations();
    let versions = |pending: Option<Vec<&Migration>>| -> Option<Vec<u32>> {
        pending.map(|pending| pending.iter().map(|m| m.version).collect())
    };

    // Nothing pending: the store is at the last version this build knows.
    let none = pending_migrations("postgres", 3, &migrations, None, false, &schema);
    assert_eq!(versions(none.unwrap()), None);

    // Pending with automatic migration: the ones above the recorded version.
    let above = pending_migrations("postgres", 1, &migrations, None, true, &schema);
    assert_eq!(versions(above.unwrap()), Some(vec![2, 3]));

    // The floor above the recorded version takes the version's place.
    let floored = pending_migrations("oracle", 1, &migrations, Some(2), true, &schema);
    assert_eq!(versions(floored.unwrap()), Some(vec![3]));

    // Pending without automatic migration: the refusal, counted from the
    // floor and naming only the migrations above it.
    let refused = pending_migrations("oracle", 1, &migrations, Some(2), false, &schema);
    let Err(err) = refused else {
        panic!("pending migrations without automatic migration are a refusal");
    };
    assert!(err.contains("current version: 2"), "{err}");
    assert!(
        err.contains("v3: test") && !err.contains("v2: test"),
        "{err}"
    );
    assert!(err.contains("oracle schema config"), "{err}");
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
