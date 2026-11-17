//! The settings the worker modules take, filled from the gateway's
//! configuration.
//!
//! Overload protection, liveness and the worker-directed HTTP clients each
//! take a small value of their own; the methods here are where a
//! [`RouterConfig`] becomes one.

use std::time::Duration;

use super::types::RouterConfig;
use crate::worker::{
    connection::http_client::WorkerHttpClientSettings,
    health::{liveness::Budgets, overload::OverloadThresholds},
};

impl RouterConfig {
    /// Gateway-level thresholds: the `--worker-overload-*` values, which
    /// default to [`DEFAULT_WAITING_REQUESTS`] and
    /// [`DEFAULT_TOKEN_USAGE_CEILING`]. `--disable-worker-overload-protection`
    /// (`worker_overload_protection: false`) switches both off whatever the
    /// thresholds say; per-worker `overload` blocks still apply on top.
    ///
    /// [`DEFAULT_WAITING_REQUESTS`]: crate::worker::health::overload::DEFAULT_WAITING_REQUESTS
    /// [`DEFAULT_TOKEN_USAGE_CEILING`]: crate::worker::health::overload::DEFAULT_TOKEN_USAGE_CEILING
    pub fn worker_overload_thresholds(&self) -> OverloadThresholds {
        if !self.worker_overload_protection {
            return OverloadThresholds::default();
        }
        OverloadThresholds {
            waiting_requests: self.worker_overload_waiting_requests,
            token_usage: self.worker_overload_token_usage,
        }
    }

    /// Every liveness budget this configuration carries: the stall, wedge and
    /// stale thresholds and the warm-up slice. The one place the budgets are
    /// read from a configuration; `worker::configure_liveness` applies them.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::health::overload::{DEFAULT_TOKEN_USAGE_CEILING, DEFAULT_WAITING_REQUESTS};

    fn gateway_config(
        protection: bool,
        waiting: Option<usize>,
        token: Option<f64>,
    ) -> RouterConfig {
        RouterConfig {
            worker_overload_protection: protection,
            worker_overload_waiting_requests: waiting,
            worker_overload_token_usage: token,
            ..RouterConfig::default()
        }
    }

    /// The gateway defaults enable protection on both signals.
    #[test]
    fn protection_is_on_by_default_on_both_signals() {
        let thresholds = RouterConfig::default().worker_overload_thresholds();
        assert_eq!(
            thresholds,
            OverloadThresholds {
                waiting_requests: Some(DEFAULT_WAITING_REQUESTS),
                token_usage: Some(DEFAULT_TOKEN_USAGE_CEILING),
            }
        );
        assert!(thresholds.is_enabled());
        assert!(!RouterConfig::default().worker_overload_shed);
    }

    /// Explicit thresholds replace the defaults; disabling protection switches
    /// both signals off whatever the thresholds say.
    #[test]
    fn explicit_thresholds_replace_the_defaults_and_disabling_wins() {
        let explicit = gateway_config(true, Some(64), Some(0.5)).worker_overload_thresholds();
        assert_eq!(explicit.token_usage, Some(0.5));
        assert_eq!(explicit.waiting_requests, Some(64));

        let one_signal = gateway_config(true, None, Some(0.5)).worker_overload_thresholds();
        assert_eq!(
            one_signal,
            OverloadThresholds {
                waiting_requests: None,
                token_usage: Some(0.5),
            }
        );

        let off = gateway_config(false, Some(8), Some(0.8)).worker_overload_thresholds();
        assert!(!off.is_enabled());
    }

    /// Each liveness budget comes from its own field.
    #[test]
    fn each_liveness_budget_reads_its_own_field() {
        let config = RouterConfig {
            worker_stall_secs: 11,
            worker_wedge_secs: 12,
            worker_stale_secs: 13,
            worker_warmup_secs: 14,
            worker_warmup_share: 0.15,
            worker_warmup_blocks: 16,
            worker_warmup_thin_ratio: 0.17,
            worker_warmup_divert_every: 18,
            ..RouterConfig::default()
        };
        assert_eq!(
            config.worker_liveness_budgets(),
            Budgets {
                stall: Duration::from_secs(11),
                wedge: Duration::from_secs(12),
                stale: Duration::from_secs(13),
                warmup_secs: Duration::from_secs(14),
                warmup_share: 0.15,
                warmup_blocks: 16,
                warmup_thin_ratio: 0.17,
                warmup_divert_every: 18,
            }
        );
    }

    /// Each setting of the worker-directed clients comes from its own field.
    #[test]
    fn each_http_client_setting_reads_its_own_field() {
        let config = RouterConfig {
            client_identity: Some(b"identity".to_vec()),
            ca_certificates: vec![b"ca".to_vec()],
            upstream_http2: true,
            request_timeout_secs: 21,
            upstream_pool_idle_timeout_secs: 22,
            ..RouterConfig::default()
        };
        let settings = config.worker_http_client_settings();
        assert_eq!(settings.client_identity.as_deref(), Some(&b"identity"[..]));
        assert_eq!(settings.ca_certificates, vec![b"ca".to_vec()]);
        assert!(settings.upstream_http2);
        assert_eq!(settings.request_timeout_secs, 21);
        assert_eq!(settings.pool_idle_timeout_secs, 22);
    }
}
