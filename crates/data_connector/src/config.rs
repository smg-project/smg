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

/// Bounds of the in-memory history backend. Its records live in the
/// gateway's own memory, so it keeps at most `max_responses` stored responses
/// and `max_items` conversation items and drops a record older than
/// `retention_secs`: the oldest go first, checked on every write. The
/// defaults hold a default deployment at a few hundred MB (a stored response
/// of a short exchange takes about 12 KB, a conversation item about 1.5 KB).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct MemoryConfig {
    /// Most stored responses kept; the oldest go first.
    pub max_responses: usize,
    /// Most conversation items kept; the oldest go first, unlinked from every
    /// conversation that lists them.
    pub max_items: usize,
    /// How long a record is kept, in seconds; `None` (or 0) keeps a record
    /// until a count bound drops it. Conversations follow the retention only,
    /// counted from their last use (created, read or updated); a background
    /// response still running is not dropped by a count bound.
    pub retention_secs: Option<u64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_responses: 10_000,
            max_items: 100_000,
            retention_secs: Some(24 * 60 * 60),
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_responses == 0 || self.max_items == 0 {
            return Err(
                "memory history: max_responses and max_items must be at least 1".to_string(),
            );
        }
        Ok(())
    }

    /// The retention as a duration; `None` for no retention.
    pub fn retention(&self) -> Option<Duration> {
        self.retention_secs
            .filter(|secs| *secs > 0)
            .map(Duration::from_secs)
    }
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

impl OracleConfig {
    /// The pool's wait for a connection, and the bound on the start-up
    /// connection; `None` when `pool_timeout_secs` is 0 (no wait bound).
    pub fn pool_timeout(&self) -> Option<Duration> {
        (self.pool_timeout_secs > 0).then(|| Duration::from_secs(self.pool_timeout_secs))
    }

    pub fn default_pool_min() -> usize {
        default_pool_min()
    }

    pub fn default_pool_max() -> usize {
        default_pool_max()
    }

    pub fn default_pool_timeout_secs() -> u64 {
        default_pool_timeout_secs()
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
    /// before any connection, as for the other remote backends; the pool's
    /// minimum and the connect descriptor are checked where the config is
    /// built.
    pub fn validate_bounds(&self) -> Result<(), String> {
        if self.pool_max == 0 {
            return Err("pool_max must be greater than 0".to_string());
        }
        if self.op_timeout_secs == 0 {
            return Err("op_timeout_secs must be greater than 0".to_string());
        }
        Ok(())
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
            .field("op_timeout_secs", &self.op_timeout_secs)
            .field("schema", &self.schema)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostgresConfig {
    /// Database connection URL,
    /// `postgres://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]`.
    pub db_url: String,
    /// Most connections in the pool; 16 unless set. Must be > 0.
    #[serde(default = "PostgresConfig::default_pool_max")]
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
    /// Redis connection URL, `redis://[:password@]host[:port][/db]`.
    pub url: String,
    /// Most connections in the pool; 16 unless set. Must be > 0.
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
    /// Data retention in days, at least 1; 30 unless set. `None` keeps every
    /// record for good.
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
        // A retention of 0 days would expire every record the moment it is
        // written; persistent records are spelled by leaving the retention unset.
        if self.retention_days == Some(0) {
            return Err(
                "retention_days must be at least 1; persistent records are spelled -1 on the command line and in Python (unset in Rust)".to_string(),
            );
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
#[path = "../tests/unit/config/tests.rs"]
mod tests;
