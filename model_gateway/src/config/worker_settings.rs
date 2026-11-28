//! The settings the worker modules take, filled from the gateway's
//! configuration.
//!
//! Overload protection, liveness and the worker-directed HTTP clients each
//! take a small value of their own; the methods here are where a
//! [`RouterConfig`] becomes one.

use std::time::Duration;

use smg_worker::{
    connection::http_client::WorkerHttpClientSettings,
    health::{
        circuit_breaker::CircuitBreakerConfig, liveness::Budgets, overload::OverloadThresholds,
    },
    settings::RegistrationSettings,
};

use super::types::RouterConfig;

impl RouterConfig {
    /// Gateway-level thresholds: the `--worker-overload-*` values, which
    /// default to [`DEFAULT_WAITING_REQUESTS`] and
    /// [`DEFAULT_TOKEN_USAGE_CEILING`]. The overload switch in force
    /// (`RouterConfig::overload_words_in_force`) switches both off whatever
    /// the thresholds say: `--disable-worker-overload-protection`
    /// (`worker_overload_protection: false`), or, where Grand
    /// Central runs a supplied document, that document's
    /// `overload.enabled: false`. Per-worker `overload` blocks still apply
    /// on top.
    ///
    /// [`DEFAULT_WAITING_REQUESTS`]: smg_worker::health::overload::DEFAULT_WAITING_REQUESTS
    /// [`DEFAULT_TOKEN_USAGE_CEILING`]: smg_worker::health::overload::DEFAULT_TOKEN_USAGE_CEILING
    pub fn worker_overload_thresholds(&self) -> OverloadThresholds {
        if !self.overload_words_in_force().protection {
            return OverloadThresholds::default();
        }
        OverloadThresholds {
            waiting_requests: self.worker_overload_waiting_requests,
            token_usage: self.worker_overload_token_usage,
        }
    }

    /// Every liveness budget this configuration carries: the stall, wedge and
    /// stale thresholds and the warm-up slice. The one place the budgets are
    /// read from a configuration; `smg_worker::health::liveness::configure_from`
    /// applies them.
    pub fn worker_liveness_budgets(&self) -> Budgets {
        Budgets {
            stall: Duration::from_secs(self.worker_stall_secs),
            wedge: Duration::from_secs(self.worker_wedge_secs),
            stale: Duration::from_secs(self.worker_stale_secs),
            warmup_secs: Duration::from_secs(self.worker_warmup_secs),
            warmup_share: self.worker_warmup_share,
            warmup_blocks: self.worker_warmup_blocks,
            warmup_thin_ratio: self.worker_warmup_thin_ratio,
            warmup_divert_every: self.worker_warmup_divert_every,
        }
    }

    /// What every worker-directed HTTP client takes from this configuration:
    /// the TLS material, the HTTP/2 tuning switch and the two timeouts a
    /// worker's own pool settings may replace.
    pub fn worker_http_client_settings(&self) -> WorkerHttpClientSettings {
        WorkerHttpClientSettings {
            client_identity: self.client_identity.clone(),
            ca_certificates: self.ca_certificates.clone(),
            upstream_http2: self.upstream_http2,
            request_timeout_secs: self.request_timeout_secs,
            pool_idle_timeout_secs: self.upstream_pool_idle_timeout_secs,
        }
    }

    /// The settings the worker's detection and creation steps read.
    pub fn registration_settings(&self) -> RegistrationSettings {
        let breaker = self.effective_circuit_breaker_config();
        RegistrationSettings {
            upstream_http2: self.upstream_http2,
            health_check_timeout_secs: self.health_check.timeout_secs,
            startup_timeout_secs: self.worker_startup_timeout_secs,
            startup_delay_secs: self.worker_startup_delay_secs,
            startup_check_interval_secs: self.worker_startup_check_interval_secs,
            model_path: self.model_path.clone(),
            model_aliases: self.model_aliases.clone(),
            retry: self.effective_retry_config(),
            retries_enabled: !self.disable_retries,
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: breaker.failure_threshold,
                success_threshold: breaker.success_threshold,
                timeout_duration: Duration::from_secs(breaker.timeout_duration_secs),
                window_duration: Duration::from_secs(breaker.window_duration_secs),
            },
            circuit_breaker_enabled: !self.disable_circuit_breaker,
            overload_defaults: self.worker_overload_thresholds(),
            health: self.health_check.to_protocol_config(),
            health_endpoint: self.health_check.endpoint.clone(),
            zmq_startup_timeout_secs: self.zmq_startup_timeout_secs,
            request_timeout_secs: self.request_timeout_secs,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/worker_settings/tests.rs"]
mod tests;
