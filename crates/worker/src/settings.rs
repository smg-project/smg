//! The settings vocabulary of the worker's own code: the values a host's
//! configuration names and the worker's health, pairing and connection code
//! reads. The gateway's configuration module re-exports each at the path it
//! had there.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[cfg(feature = "runtime")]
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::{health::overload::OverloadThresholds, worker::HealthCheckConfig};

/// Which Failed workers the health checker removes. Mesh-imported workers
/// are never removed under any scope: their owner manages them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnhealthyRemovalScope {
    /// None: a Failed worker stays registered and probed, and rejoins in
    /// place when its engine answers again.
    Disabled,
    /// The workers a discovery provider registered, which it registers again
    /// once their engine returns. Static and API-added workers stay.
    DiscoveryOwned,
    /// Every local worker, static and API-added ones included, though
    /// nothing re-adds those.
    AllLocal,
}

/// How strictly PD placement pairs a prefill with a decode on their KV
/// transfer protocol (#2483). A descriptor component an engine does not report
/// is "unknown".
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PdPairingMode {
    /// Compare nothing of the KV handoff: placement behaves as if no
    /// descriptor existed. The gRPC wire still keeps two known runtimes
    /// apart, since its requests are in the runtime's own message type.
    Off,
    /// A known difference in runtime, transport or KV layout refuses the
    /// pair; unknown components and engine versions pair with anything.
    #[default]
    Lenient,
    /// Runtime, transport and KV layout must be known on both sides and
    /// agree, and reported engine versions must match.
    Strict,
}

impl PdPairingMode {
    /// Number of modes, for per-mode caches.
    pub const COUNT: usize = 3;

    /// A dense index for per-mode caches.
    pub const fn index(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Lenient => 1,
            Self::Strict => 2,
        }
    }

    /// Parse the CLI spelling (`off` / `lenient` / `strict`).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "lenient" => Some(Self::Lenient),
            "strict" => Some(Self::Strict),
            _ => None,
        }
    }

    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Lenient => "lenient",
            Self::Strict => "strict",
        }
    }
}

/// The default per-stage silence bound of a ZMQ engine's startup handshake:
/// an hour, since it alone covers the engine's init between HELLO and READY.
pub const DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS: u64 = 3600;

/// Retry configuration for request handling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// What a host fixes for every worker registration it runs, read once from
/// its own configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistrationSettings {
    /// Whether worker-directed clients are tuned for HTTP/2 upstreams. An HTTP
    /// worker that declares no version is then asked for HTTP/2 with prior
    /// knowledge beside HTTP/1.1.
    pub upstream_http2: bool,
    /// Timeout of one health probe, in seconds, for a worker whose own health
    /// settings name none.
    pub health_check_timeout_secs: u64,
    /// How long a starting worker is polled before its registration fails, in
    /// seconds.
    pub startup_timeout_secs: u64,
    /// How long a worker is left alone before its first probe, in seconds.
    pub startup_delay_secs: u64,
    /// The pause between two probes of a starting worker, in seconds.
    pub startup_check_interval_secs: u64,
    /// Model identity for a ZMQ worker whose spec and labels name no model.
    pub model_path: Option<String>,
    /// Aliases to add to the card of the canonical model they name.
    pub model_aliases: HashMap<String, String>,
    /// Retry settings for a worker before its own overrides.
    pub retry: RetryConfig,
    /// Whether a worker retries requests unless its spec sets otherwise.
    pub retries_enabled: bool,
    /// Circuit breaker settings for a worker before its own overrides.
    #[cfg(feature = "runtime")]
    pub circuit_breaker: CircuitBreakerConfig,
    /// Whether a worker uses its circuit breaker unless its spec sets otherwise.
    pub circuit_breaker_enabled: bool,
    /// Overload thresholds for signals the worker leaves unset.
    pub overload_defaults: OverloadThresholds,
    /// Health probe settings for a worker before its own overrides.
    pub health: HealthCheckConfig,
    /// Path a worker's HTTP health probe requests.
    pub health_endpoint: String,
    /// How long a stage of a worker's ZMQ startup handshake may stay silent.
    pub zmq_startup_timeout_secs: u64,
    /// Request timeout used for the ZMQ link when the worker has none of its own.
    pub request_timeout_secs: u64,
}
