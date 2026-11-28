use super::*;

// ── validate_bounds ─────────────────────────────────────────────────

/// The factory's check holds the bounds only: a connection string the
/// driver understands but the CLI's URL check does not (a key=value
/// string, a socket path) passes it, and a zero bound does not.
#[test]
fn validate_bounds_leaves_the_connection_string_to_the_driver() {
    let postgres = PostgresConfig {
        db_url: "host=/var/run/postgresql dbname=smg user=smg".to_string(),
        pool_max: 4,
        op_timeout_secs: 5,
        schema: None,
    };
    assert!(postgres.validate_bounds().is_ok());
    assert!(postgres.validate().is_err());
    let zero = PostgresConfig {
        op_timeout_secs: 0,
        ..postgres
    };
    assert!(zero
        .validate_bounds()
        .unwrap_err()
        .contains("op_timeout_secs"));

    let redis = RedisConfig {
        url: "unix:///run/redis.sock".to_string(),
        pool_max: 4,
        op_timeout_secs: 5,
        retention_days: Some(1),
        schema: None,
    };
    assert!(redis.validate_bounds().is_ok());
    assert!(redis.validate().is_err());
    let zero = RedisConfig {
        pool_max: 0,
        ..redis
    };
    assert!(zero.validate_bounds().unwrap_err().contains("pool_max"));
}

// ── PostgresConfig::validate ────────────────────────────────────────

#[test]
fn postgres_valid_url_succeeds() {
    let cfg = PostgresConfig {
        db_url: "postgres://user:pass@localhost:5432/mydb".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        schema: None,
    };
    cfg.validate()
        .expect("valid postgres URL should pass validation");
}

#[test]
fn postgres_postgresql_scheme_succeeds() {
    let cfg = PostgresConfig {
        db_url: "postgresql://user:pass@localhost/mydb".to_string(),
        pool_max: 8,
        op_timeout_secs: 5,
        schema: None,
    };
    cfg.validate()
        .expect("postgresql:// scheme should also be accepted");
}

#[test]
fn postgres_empty_url_fails() {
    let cfg = PostgresConfig {
        db_url: "  ".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        schema: None,
    };
    let err = cfg.validate().expect_err("empty URL should fail");
    assert!(
        err.contains("not be empty"),
        "unexpected error message: {err}"
    );
}

#[test]
fn postgres_non_postgres_scheme_fails() {
    let cfg = PostgresConfig {
        db_url: "mysql://user:pass@localhost/mydb".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        schema: None,
    };
    let err = cfg.validate().expect_err("mysql scheme should be rejected");
    assert!(
        err.contains("unsupported URL scheme"),
        "unexpected error message: {err}"
    );
}

#[test]
fn postgres_missing_host_fails() {
    // `postgres:///mydb` is a valid URL with no host
    let cfg = PostgresConfig {
        db_url: "postgres:///mydb".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        schema: None,
    };
    let err = cfg.validate().expect_err("missing host should fail");
    assert!(
        err.contains("must have a host"),
        "unexpected error message: {err}"
    );
}

#[test]
fn postgres_missing_database_name_fails() {
    let cfg = PostgresConfig {
        db_url: "postgres://user:pass@localhost".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        schema: None,
    };
    let err = cfg
        .validate()
        .expect_err("missing database name should fail");
    assert!(
        err.contains("database name"),
        "unexpected error message: {err}"
    );
}

#[test]
fn postgres_pool_max_zero_fails() {
    let cfg = PostgresConfig {
        db_url: "postgres://user:pass@localhost/mydb".to_string(),
        pool_max: 0,
        op_timeout_secs: 5,
        schema: None,
    };
    let err = cfg.validate().expect_err("pool_max=0 should fail");
    assert!(
        err.contains("greater than 0"),
        "unexpected error message: {err}"
    );
}

// ── RedisConfig::validate ───────────────────────────────────────────

#[test]
fn redis_valid_url_succeeds() {
    let cfg = RedisConfig {
        url: "redis://:password@localhost:6379/0".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        retention_days: Some(30),
        schema: None,
    };
    cfg.validate()
        .expect("valid redis URL should pass validation");
}

#[test]
fn redis_rediss_scheme_succeeds() {
    let cfg = RedisConfig {
        url: "rediss://:password@redis.example.com:6380".to_string(),
        pool_max: 8,
        op_timeout_secs: 5,
        retention_days: None,
        schema: None,
    };
    cfg.validate()
        .expect("rediss:// scheme should also be accepted");
}

#[test]
fn redis_empty_url_fails() {
    let cfg = RedisConfig {
        url: String::new(),
        pool_max: 16,
        op_timeout_secs: 5,
        retention_days: Some(30),
        schema: None,
    };
    let err = cfg.validate().expect_err("empty URL should fail");
    assert!(
        err.contains("not be empty"),
        "unexpected error message: {err}"
    );
}

#[test]
fn redis_non_redis_scheme_fails() {
    let cfg = RedisConfig {
        url: "http://localhost:6379".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        retention_days: Some(30),
        schema: None,
    };
    let err = cfg.validate().expect_err("http scheme should be rejected");
    assert!(
        err.contains("unsupported URL scheme"),
        "unexpected error message: {err}"
    );
}

#[test]
fn redis_missing_host_fails() {
    let cfg = RedisConfig {
        url: "redis:///0".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        retention_days: Some(30),
        schema: None,
    };
    let err = cfg.validate().expect_err("missing host should fail");
    assert!(
        err.contains("must have a host"),
        "unexpected error message: {err}"
    );
}

#[test]
fn redis_pool_max_zero_fails() {
    let cfg = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        pool_max: 0,
        op_timeout_secs: 5,
        retention_days: Some(30),
        schema: None,
    };
    let err = cfg.validate().expect_err("pool_max=0 should fail");
    assert!(
        err.contains("greater than 0"),
        "unexpected error message: {err}"
    );
}

// ── pool_max ────────────────────────────────────────────────────────

/// A configuration that leaves `pool_max` out parses to the crate's
/// default for either store, as one without `op_timeout_secs` does: the
/// CLI passes the field, a file or a binding built by hand need not.
#[test]
fn pool_max_defaults_to_sixteen_when_left_out() {
    let pg: PostgresConfig =
        serde_json::from_value(serde_json::json!({"db_url": "postgres://u:p@h/d"}))
            .expect("a Postgres config without pool_max");
    assert_eq!(pg.pool_max, PostgresConfig::default_pool_max());
    assert_eq!(pg.pool_max, 16);
    let redis: RedisConfig = serde_json::from_value(serde_json::json!({"url": "redis://h"}))
        .expect("a Redis config without pool_max");
    assert_eq!(redis.pool_max, 16);

    let pg: PostgresConfig =
        serde_json::from_value(serde_json::json!({"db_url": "postgres://u:p@h/d", "pool_max": 4}))
            .expect("a Postgres config with pool_max");
    assert_eq!(pg.pool_max, 4, "a set value is kept");
}

// ── op_timeout_secs ─────────────────────────────────────────────────

#[test]
fn op_timeout_defaults_to_five_seconds_and_reads_the_configured_value() {
    let pg: PostgresConfig =
        serde_json::from_value(serde_json::json!({"db_url": "postgres://u:p@h/d", "pool_max": 4}))
            .expect("a config without the field");
    assert_eq!(pg.op_timeout_secs, 5);
    assert_eq!(pg.op_timeout(), Duration::from_secs(5));
    let redis: RedisConfig = serde_json::from_value(serde_json::json!({"url": "redis://h"}))
        .expect("a config without the field");
    assert_eq!(redis.op_timeout_secs, 5);
    let oracle: OracleConfig =
        serde_json::from_value(oracle_json(None)).expect("a config without the field");
    assert_eq!(oracle.op_timeout_secs, 5);
    assert_eq!(oracle.op_timeout(), Duration::from_secs(5));

    let pg: PostgresConfig = serde_json::from_value(
        serde_json::json!({"db_url": "postgres://u:p@h/d", "pool_max": 4, "op_timeout_secs": 12}),
    )
    .expect("a config with the field");
    assert_eq!(pg.op_timeout(), Duration::from_secs(12));
    let redis: RedisConfig =
        serde_json::from_value(serde_json::json!({"url": "redis://h", "op_timeout_secs": 3}))
            .expect("a config with the field");
    assert_eq!(redis.op_timeout(), Duration::from_secs(3));
    let oracle: OracleConfig =
        serde_json::from_value(oracle_json(Some(9))).expect("a config with the field");
    assert_eq!(oracle.op_timeout(), Duration::from_secs(9));
}

/// An Oracle configuration with the connection fields alone, and the
/// bound when given: the Oracle key is the same key as the other two.
fn oracle_json(op_timeout_secs: Option<u64>) -> serde_json::Value {
    let mut config = serde_json::json!({
        "connect_descriptor": "tcps://h:1522/s",
        "username": "u",
        "password": "p",
    });
    if let Some(secs) = op_timeout_secs {
        config["op_timeout_secs"] = serde_json::json!(secs);
    }
    config
}

#[test]
fn a_zero_op_timeout_is_refused() {
    let pg = PostgresConfig {
        db_url: "postgres://user:pass@localhost:5432/mydb".to_string(),
        pool_max: 16,
        op_timeout_secs: 0,
        schema: None,
    };
    let err = pg.validate().expect_err("op_timeout_secs=0 should fail");
    assert!(
        err.contains("op_timeout_secs"),
        "unexpected error message: {err}"
    );
    let redis = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        pool_max: 16,
        op_timeout_secs: 0,
        retention_days: Some(30),
        schema: None,
    };
    let err = redis.validate().expect_err("op_timeout_secs=0 should fail");
    assert!(
        err.contains("op_timeout_secs"),
        "unexpected error message: {err}"
    );
    let oracle: OracleConfig =
        serde_json::from_value(oracle_json(Some(0))).expect("a config with a zero bound");
    let err = oracle
        .validate_bounds()
        .expect_err("op_timeout_secs=0 should fail");
    assert!(
        err.contains("op_timeout_secs"),
        "unexpected error message: {err}"
    );
}

#[test]
fn redis_retention_zero_fails() {
    let cfg = RedisConfig {
        url: "redis://localhost:6379/0".to_string(),
        pool_max: 16,
        op_timeout_secs: 5,
        retention_days: Some(0),
        schema: None,
    };
    let err = cfg.validate().expect_err("retention_days=0 should fail");
    assert!(
        err.contains("retention_days must be at least 1"),
        "unexpected error message: {err}"
    );
}

// ── HistoryBackend ──────────────────────────────────────────────────

#[test]
fn history_backend_default_is_memory() {
    assert_eq!(HistoryBackend::default(), HistoryBackend::Memory);
}

#[test]
fn history_backend_serde_roundtrip_memory() {
    let backend = HistoryBackend::Memory;
    let json = serde_json::to_string(&backend).expect("serialize Memory");
    assert_eq!(json, r#""memory""#);
    let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Memory");
    assert_eq!(deserialized, HistoryBackend::Memory);
}

#[test]
fn history_backend_serde_roundtrip_none() {
    let backend = HistoryBackend::None;
    let json = serde_json::to_string(&backend).expect("serialize None");
    assert_eq!(json, r#""none""#);
    let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize None");
    assert_eq!(deserialized, HistoryBackend::None);
}

#[test]
fn history_backend_serde_roundtrip_oracle() {
    let backend = HistoryBackend::Oracle;
    let json = serde_json::to_string(&backend).expect("serialize Oracle");
    assert_eq!(json, r#""oracle""#);
    let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Oracle");
    assert_eq!(deserialized, HistoryBackend::Oracle);
}

#[test]
fn history_backend_serde_roundtrip_postgres() {
    let backend = HistoryBackend::Postgres;
    let json = serde_json::to_string(&backend).expect("serialize Postgres");
    assert_eq!(json, r#""postgres""#);
    let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Postgres");
    assert_eq!(deserialized, HistoryBackend::Postgres);
}

#[test]
fn history_backend_serde_roundtrip_redis() {
    let backend = HistoryBackend::Redis;
    let json = serde_json::to_string(&backend).expect("serialize Redis");
    assert_eq!(json, r#""redis""#);
    let deserialized: HistoryBackend = serde_json::from_str(&json).expect("deserialize Redis");
    assert_eq!(deserialized, HistoryBackend::Redis);
}
