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
    /// (`worker_overload_protection: false`), or, where the policy
    /// controller runs a supplied document, that document's
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
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::HttpPoolConfig;
    use smg_policy_controller::document::{OverloadSpec, PolicySpec};
    use smg_worker::{
        connection::http_client::WorkerHttpClientCache,
        health::overload::{DEFAULT_TOKEN_USAGE_CEILING, DEFAULT_WAITING_REQUESTS},
    };

    use super::*;
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    #[test]
    fn the_registration_settings_carry_what_the_creation_reads() {
        use crate::config::{
            CircuitBreakerConfig as RouterCircuitBreakerConfig, HealthCheckConfig, RetryConfig,
        };

        let config = RouterConfig {
            model_path: Some("/models/a".to_string()),
            model_aliases: [("alias".to_string(), "a".to_string())].into(),
            retry: RetryConfig {
                max_retries: 11,
                initial_backoff_ms: 19,
                max_backoff_ms: 29,
                backoff_multiplier: 2.0,
                jitter_factor: 0.1,
            },
            disable_retries: true,
            circuit_breaker: RouterCircuitBreakerConfig {
                failure_threshold: 13,
                success_threshold: 7,
                timeout_duration_secs: 21,
                window_duration_secs: 39,
            },
            disable_circuit_breaker: true,
            worker_overload_protection: true,
            worker_overload_waiting_requests: Some(12),
            worker_overload_token_usage: Some(0.4),
            health_check: HealthCheckConfig {
                endpoint: "/live".to_string(),
                timeout_secs: 11,
                check_interval_secs: 17,
                failure_threshold: 5,
                success_threshold: 4,
                disable_health_check: true,
                drain_settle_secs: 9,
                ..HealthCheckConfig::default()
            },
            zmq_startup_timeout_secs: 43,
            request_timeout_secs: 47,
            ..RouterConfig::default()
        };
        let settings = config.registration_settings();
        let breaker = config.effective_circuit_breaker_config();

        assert_eq!(settings.model_path, config.model_path);
        assert_eq!(settings.model_aliases, config.model_aliases);
        assert_eq!(settings.retry, config.effective_retry_config());
        assert_eq!(settings.retries_enabled, !config.disable_retries);
        assert_eq!(
            settings.circuit_breaker,
            CircuitBreakerConfig {
                failure_threshold: breaker.failure_threshold,
                success_threshold: breaker.success_threshold,
                timeout_duration: Duration::from_secs(breaker.timeout_duration_secs),
                window_duration: Duration::from_secs(breaker.window_duration_secs),
            }
        );
        assert_eq!(
            settings.circuit_breaker_enabled,
            !config.disable_circuit_breaker
        );
        assert_eq!(
            settings.overload_defaults,
            config.worker_overload_thresholds()
        );
        assert_eq!(settings.health, config.health_check.to_protocol_config());
        assert_eq!(settings.health_endpoint, config.health_check.endpoint);
        assert_eq!(
            settings.zmq_startup_timeout_secs,
            config.zmq_startup_timeout_secs
        );
        assert_eq!(settings.request_timeout_secs, config.request_timeout_secs);
    }

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

    /// Where the policy controller runs a supplied document, the document's
    /// switch decides the default thresholds. Bindings that switch the
    /// overload block off leave the worker objects no default threshold
    /// whatever the two threshold settings say, as disabling protection
    /// does with no document, and a block that is on leaves the two
    /// settings standing. With the option off the document is not read.
    #[test]
    fn a_supplied_documents_switch_decides_the_default_thresholds() {
        let under = |enabled: bool, policy_controller: bool| {
            let mut document = PolicySpec::preset("round_robin").expect("a stock policy");
            document.bindings[0].overload = Some(OverloadSpec {
                enabled,
                ..OverloadSpec::default()
            });
            RouterConfig {
                policy_controller,
                policy_spec: Some(document),
                ..gateway_config(true, Some(32), Some(0.9))
            }
        };
        let stated = OverloadThresholds {
            waiting_requests: Some(32),
            token_usage: Some(0.9),
        };
        assert_eq!(under(true, true).worker_overload_thresholds(), stated);
        assert_eq!(
            under(false, true).worker_overload_thresholds(),
            OverloadThresholds::default()
        );
        assert_eq!(
            under(false, true).registration_settings().overload_defaults,
            OverloadThresholds::default()
        );
        assert_eq!(under(false, false).worker_overload_thresholds(), stated);
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

    #[test]
    fn entry_lives_exactly_as_long_as_worker_handles() {
        let cache =
            WorkerHttpClientCache::new(RouterConfig::default().worker_http_client_settings());
        let handle = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let probe = Arc::downgrade(&handle);
        let worker_a = BasicWorkerBuilder::new("http://a:1")
            .http_client(handle.clone())
            .build();
        let worker_b = BasicWorkerBuilder::new("http://b:1")
            .http_client(handle)
            .build();

        drop(worker_a);
        assert!(
            probe.upgrade().is_some(),
            "entry survives while another worker holds it"
        );

        drop(worker_b);
        assert!(
            probe.upgrade().is_none(),
            "last worker drop releases the entry"
        );

        // The dead entry cannot be upgraded, so the next get rebuilds.
        let rebuilt = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        assert!(probe.upgrade().is_none());
        drop(rebuilt);
    }
}
