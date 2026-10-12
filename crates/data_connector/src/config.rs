//! Storage backend configuration types.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::schema::SchemaConfig;

/// History backend configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum HistoryBackend {
    #[default]
    Memory,
    None,
    Oracle,
    Postgres,
    Redis,
}

/// Oracle history backend configuration
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleConfig {
    /// ATP wallet or TLS config files directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wallet_path: Option<String>,
    /// DSN (e.g. `tcps://host:port/service`)
    pub connect_descriptor: String,
    #[serde(default)]
    pub external_auth: bool,
    pub username: String,
    pub password: String,
    #[serde(default = "default_pool_min")]
    pub pool_min: usize,
    #[serde(default = "default_pool_max")]
    pub pool_max: usize,
    #[serde(default = "default_pool_timeout_secs")]
    pub pool_timeout_secs: u64,
    /// Optional schema customization (table names, column names, extra columns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<SchemaConfig>,
}

impl OracleConfig {
    pub fn default_pool_min() -> usize {
        default_pool_min()
    }

    pub fn default_pool_max() -> usize {
        default_pool_max()
    }

    pub fn default_pool_timeout_secs() -> u64 {
        default_pool_timeout_secs()
    }
}

fn default_pool_min() -> usize {
    1
}

fn default_pool_max() -> usize {
    16
}

fn default_pool_timeout_secs() -> u64 {
    30
}

impl std::fmt::Debug for OracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &self.connect_descriptor)
            .field("external_auth", &self.external_auth)
            .field("username", &self.username)
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .field("schema", &self.schema)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostgresConfig {
    // Database connection URL,
    // postgres://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
    pub db_url: String,
    // Database pool max size
    pub pool_max: usize,
    /// Bound, in seconds, on one call to the store: waiting for a pooled
    /// connection, connecting, and the operation itself. A call that exceeds
    /// it fails as a timeout, which the API answers as `503
    /// storage_unavailable`, instead of holding its request for as long as
    /// the client waits. Must be > 0.
    #[serde(default = "default_op_timeout_secs")]
    pub op_timeout_secs: u64,
    /// Optional schema customization (table names, column names, extra columns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<SchemaConfig>,
}

impl PostgresConfig {
    pub fn default_pool_max() -> usize {
        16
    }

    pub fn default_op_timeout_secs() -> u64 {
        default_op_timeout_secs()
    }

    /// The bound on one call to the store.
    pub fn op_timeout(&self) -> Duration {
        Duration::from_secs(self.op_timeout_secs)
    }

    /// The numeric invariants every entry path must hold (the CLI, the
    /// Python bindings, a config built by hand), checked by the factory
    /// before any connection. The connection string's shape is left to the
    /// driver here: [`Self::validate`] checks it for the CLI's URLs, while a
    /// socket path or a key=value string the driver understands passes.
    pub fn validate_bounds(&self) -> Result<(), String> {
        if self.pool_max == 0 {
            return Err("pool_max must be greater than 0".to_string());
        }
        if self.op_timeout_secs == 0 {
            return Err("op_timeout_secs must be greater than 0".to_string());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let s = self.db_url.trim();
        if s.is_empty() {
            return Err("db_url should not be empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid db_url: {e}"))?;

        let scheme = url.scheme();
        if scheme != "postgres" && scheme != "postgresql" {
            return Err(format!("unsupported URL scheme: {scheme}"));
        }

        if url.host().is_none() {
            return Err("db_url must have a host".to_string());
        }

        let path = url.path();
        let dbname = path
            .strip_prefix('/')
            .filter(|p| !p.is_empty())
            .map(|s| s.to_string());
        if dbname.is_none() {
            return Err("db_url must include a database name".to_string());
        }

        self.validate_bounds()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedisConfig {
    // Redis connection URL
    // redis://[:password@]host[:port][/db]
    pub url: String,
    // Connection pool max size
    #[serde(default = "default_redis_pool_max")]
    pub pool_max: usize,
    /// Bound, in seconds, on one call to the store: waiting for a pooled
    /// connection, connecting, and the operation itself. A call that exceeds
    /// it fails as a timeout, which the API answers as `503
    /// storage_unavailable`, instead of holding its request for as long as
    /// the client waits. Must be > 0.
    #[serde(default = "default_op_timeout_secs")]
    pub op_timeout_secs: u64,
    /// Optional schema customization (key prefix, field names, extra fields).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<SchemaConfig>,
    // Data retention in days. If None, data persists indefinitely.
    #[serde(default = "default_redis_retention_days")]
    pub retention_days: Option<u64>,
}

fn default_redis_pool_max() -> usize {
    16
}

/// The default bound on one store call, in seconds: a few seconds, so a store
/// that stopped answering degrades the stateful endpoints instead of pinning
/// their requests, while a slow store still works.
fn default_op_timeout_secs() -> u64 {
    5
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default function must match field type Option<u64>"
)]
fn default_redis_retention_days() -> Option<u64> {
    Some(30)
}

impl RedisConfig {
    pub fn default_op_timeout_secs() -> u64 {
        default_op_timeout_secs()
    }

    /// The bound on one call to the store.
    pub fn op_timeout(&self) -> Duration {
        Duration::from_secs(self.op_timeout_secs)
    }

    /// The numeric invariants every entry path must hold (the CLI, the
    /// Python bindings, a config built by hand), checked by the factory
    /// before any connection. The connection string's shape is left to the
    /// driver here: [`Self::validate`] checks it for the CLI's URLs, while a
    /// socket path the driver understands passes.
    pub fn validate_bounds(&self) -> Result<(), String> {
        if self.pool_max == 0 {
            return Err("pool_max must be greater than 0".to_string());
        }
        if self.op_timeout_secs == 0 {
            return Err("op_timeout_secs must be greater than 0".to_string());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let s = self.url.trim();
        if s.is_empty() {
            return Err("redis url should not be empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid redis url: {e}"))?;

        let scheme = url.scheme();
        if scheme != "redis" && scheme != "rediss" {
            return Err(format!("unsupported URL scheme: {scheme}"));
        }

        if url.host().is_none() {
            return Err("redis url must have a host".to_string());
        }

        self.validate_bounds()
    }
}

#[cfg(test)]
mod tests {
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

    // ── op_timeout_secs ─────────────────────────────────────────────────

    #[test]
    fn op_timeout_defaults_to_five_seconds_and_reads_the_configured_value() {
        let pg: PostgresConfig = serde_json::from_value(
            serde_json::json!({"db_url": "postgres://u:p@h/d", "pool_max": 4}),
        )
        .expect("a config without the field");
        assert_eq!(pg.op_timeout_secs, 5);
        assert_eq!(pg.op_timeout(), Duration::from_secs(5));
        let redis: RedisConfig = serde_json::from_value(serde_json::json!({"url": "redis://h"}))
            .expect("a config without the field");
        assert_eq!(redis.op_timeout_secs, 5);

        let pg: PostgresConfig = serde_json::from_value(
            serde_json::json!({"db_url": "postgres://u:p@h/d", "pool_max": 4, "op_timeout_secs": 12}),
        )
        .expect("a config with the field");
        assert_eq!(pg.op_timeout(), Duration::from_secs(12));
        let redis: RedisConfig =
            serde_json::from_value(serde_json::json!({"url": "redis://h", "op_timeout_secs": 3}))
                .expect("a config with the field");
        assert_eq!(redis.op_timeout(), Duration::from_secs(3));
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
        let deserialized: HistoryBackend =
            serde_json::from_str(&json).expect("deserialize Postgres");
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
}
