//! Resolved per-worker resilience configuration.
//!
//! Merges router-level defaults with per-worker overrides from `ResilienceUpdate`.
//! Created at worker construction time — immutable after that.

use std::{collections::HashSet, time::Duration};

use openai_protocol::worker::ResilienceUpdate;

use crate::{config::types::RetryConfig, worker::circuit_breaker::CircuitBreakerConfig};

/// Default retryable HTTP status codes.
pub const DEFAULT_RETRYABLE_STATUS_CODES: &[u16] = &[408, 429, 500, 502, 503, 504];

/// Default capacity-pushback status codes: retried elsewhere but recorded as
/// neither success nor failure by the circuit breaker. 429 is unambiguous
/// backpressure; 503 stays a fault by default (indistinguishable from an
/// outage) and can be added per worker for backends that use it as pushback.
pub const DEFAULT_CAPACITY_STATUS_CODES: &[u16] = &[429];

/// Fully resolved resilience configuration for a worker.
///
/// Created by merging `RouterConfig` defaults with `WorkerSpec.resilience` overrides.
/// Immutable after construction.
#[derive(Debug, Clone)]
pub struct ResolvedResilience {
    /// Effective retry config.
    pub retry: RetryConfig,
    /// Whether retries are enabled.
    pub retry_enabled: bool,
    /// Whether circuit breaker is enabled.
    pub circuit_breaker_enabled: bool,
    /// Set of HTTP status codes considered retryable.
    pub retryable_status_codes: HashSet<u16>,
    /// Statuses treated as capacity pushback: retryable, but excluded from
    /// circuit-breaker accounting entirely.
    pub capacity_status_codes: HashSet<u16>,
}

impl Default for ResolvedResilience {
    fn default() -> Self {
        Self {
            retry: RetryConfig::default(),
            retry_enabled: true,
            circuit_breaker_enabled: true,
            retryable_status_codes: DEFAULT_RETRYABLE_STATUS_CODES.iter().copied().collect(),
            capacity_status_codes: DEFAULT_CAPACITY_STATUS_CODES.iter().copied().collect(),
        }
    }
}

/// Resolve per-worker resilience config by merging router defaults with worker overrides.
pub fn resolve_resilience(
    base_retry: &RetryConfig,
    base_cb: &CircuitBreakerConfig,
    base_retry_enabled: bool,
    base_cb_enabled: bool,
    overrides: &ResilienceUpdate,
) -> (ResolvedResilience, CircuitBreakerConfig) {
    // Resolve retry config
    let retry = RetryConfig {
        max_retries: overrides.max_retries.unwrap_or(base_retry.max_retries),
        initial_backoff_ms: overrides
            .initial_backoff_ms
            .unwrap_or(base_retry.initial_backoff_ms),
        max_backoff_ms: overrides
            .max_backoff_ms
            .unwrap_or(base_retry.max_backoff_ms),
        backoff_multiplier: overrides
            .backoff_multiplier
            .unwrap_or(base_retry.backoff_multiplier),
        jitter_factor: overrides.jitter_factor.unwrap_or(base_retry.jitter_factor),
    };

    // Resolve circuit breaker config
    let cb_config = CircuitBreakerConfig {
        failure_threshold: overrides
            .cb_failure_threshold
            .unwrap_or(base_cb.failure_threshold),
        success_threshold: overrides
            .cb_success_threshold
            .unwrap_or(base_cb.success_threshold),
        timeout_duration: overrides
            .cb_timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(base_cb.timeout_duration),
        window_duration: overrides
            .cb_window_secs
            .map(Duration::from_secs)
            .unwrap_or(base_cb.window_duration),
    };

    // Resolve enabled flags (per-worker overrides router-level)
    let retry_enabled = overrides
        .disable_retry
        .map(|d| !d)
        .unwrap_or(base_retry_enabled);

    let cb_enabled = overrides
        .disable_circuit_breaker
        .map(|d| !d)
        .unwrap_or(base_cb_enabled);

    // Both sets replace their defaults wholesale, and the capacity set is
    // deliberately not unioned into the retryable one. Retryability is not
    // decided here: every retry path goes through
    // `routers::common::retry::is_retryable_status`, a router-global rule that
    // always covers 429, so a partial override here cannot make capacity
    // pushback non-retryable. `retryable_status_codes` has exactly one
    // consumer - circuit-breaker failure classification in
    // `Worker::record_outcome` - and that check is unreachable for capacity
    // codes, which return early just above it. Merging the sets would add no
    // behaviour and would break the documented replace-the-default contract.
    let retryable_status_codes = overrides
        .retryable_status_codes
        .as_ref()
        .map(|codes| codes.iter().copied().collect())
        .unwrap_or_else(|| DEFAULT_RETRYABLE_STATUS_CODES.iter().copied().collect());

    let capacity_status_codes = overrides
        .capacity_status_codes
        .as_ref()
        .map(|codes| codes.iter().copied().collect())
        .unwrap_or_else(|| DEFAULT_CAPACITY_STATUS_CODES.iter().copied().collect());

    let resolved = ResolvedResilience {
        retry,
        retry_enabled,
        circuit_breaker_enabled: cb_enabled,
        retryable_status_codes,
        capacity_status_codes,
    };

    (resolved, cb_config)
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;
    use crate::routers::common::retry::is_retryable_status;

    #[test]
    fn test_default_resilience() {
        let resolved = ResolvedResilience::default();
        assert!(resolved.retry_enabled);
        assert!(resolved.circuit_breaker_enabled);
        assert_eq!(resolved.retryable_status_codes.len(), 6);
        assert!(resolved.retryable_status_codes.contains(&429));
        assert!(resolved.retryable_status_codes.contains(&503));
    }

    #[test]
    fn test_resolve_with_no_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate::default();

        let (resolved, cb_config) =
            resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retry.max_retries, base_retry.max_retries);
        assert_eq!(cb_config.failure_threshold, base_cb.failure_threshold);
        assert!(resolved.retry_enabled);
        assert!(resolved.circuit_breaker_enabled);
    }

    #[test]
    fn test_resolve_with_retry_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            max_retries: Some(10),
            initial_backoff_ms: Some(200),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retry.max_retries, 10);
        assert_eq!(resolved.retry.initial_backoff_ms, 200);
        // Non-overridden fields keep base values
        assert_eq!(resolved.retry.max_backoff_ms, base_retry.max_backoff_ms);
    }

    #[test]
    fn test_resolve_disable_retry() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            disable_retry: Some(true),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert!(!resolved.retry_enabled);
    }

    #[test]
    fn test_resolve_worker_enables_when_router_disables() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            disable_retry: Some(false),
            ..Default::default()
        };

        // Router has retries disabled, but worker explicitly enables
        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, false, true, &overrides);

        assert!(resolved.retry_enabled);
    }

    #[test]
    fn test_resolve_custom_retryable_codes() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            retryable_status_codes: Some(vec![502, 503]),
            ..Default::default()
        };

        let (resolved, _) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(resolved.retryable_status_codes.len(), 2);
        assert!(resolved.retryable_status_codes.contains(&502));
        assert!(resolved.retryable_status_codes.contains(&503));
        assert!(!resolved.retryable_status_codes.contains(&429));
    }

    #[test]
    fn test_resolve_cb_overrides() {
        let base_retry = RetryConfig::default();
        let base_cb = CircuitBreakerConfig::default();
        let overrides = ResilienceUpdate {
            cb_failure_threshold: Some(10),
            cb_timeout_secs: Some(60),
            ..Default::default()
        };

        let (_, cb_config) = resolve_resilience(&base_retry, &base_cb, true, true, &overrides);

        assert_eq!(cb_config.failure_threshold, 10);
        assert_eq!(cb_config.timeout_duration, Duration::from_secs(60));
        // Non-overridden
        assert_eq!(cb_config.success_threshold, base_cb.success_threshold);
    }

    #[test]
    fn capacity_codes_default_to_429_only() {
        let resolved = ResolvedResilience::default();
        assert!(resolved.capacity_status_codes.contains(&429));
        assert_eq!(resolved.capacity_status_codes.len(), 1);
        // 503 stays a plain retryable fault by default.
        assert!(!resolved.capacity_status_codes.contains(&503));
        assert!(resolved.retryable_status_codes.contains(&503));
    }

    #[test]
    fn capacity_codes_override_replaces_default() {
        // The override omits the default 429 so this pins replacement, not
        // union: a superset override would pass either way.
        let overrides = ResilienceUpdate {
            capacity_status_codes: Some(vec![503]),
            ..Default::default()
        };
        let (resolved, _cb) = resolve_resilience(
            &RetryConfig::default(),
            &CircuitBreakerConfig::default(),
            true,
            true,
            &overrides,
        );
        assert!(resolved.capacity_status_codes.contains(&503));
        assert!(!resolved.capacity_status_codes.contains(&429));
        assert_eq!(resolved.capacity_status_codes.len(), 1);
    }

    #[test]
    fn partial_retryable_override_leaves_capacity_pushback_retryable() {
        // Pins why the capacity set is not merged into the retryable set: a
        // per-worker retryable override that drops 429 cannot make pushback
        // non-retryable, because the retry gate is router-global.
        let overrides = ResilienceUpdate {
            retryable_status_codes: Some(vec![500, 502]),
            ..Default::default()
        };
        let (resolved, _cb) = resolve_resilience(
            &RetryConfig::default(),
            &CircuitBreakerConfig::default(),
            true,
            true,
            &overrides,
        );

        // The override replaces the default set verbatim - 429 is not merged in.
        assert!(!resolved.retryable_status_codes.contains(&429));
        // Yet 429 is still retried, because retryability never consults this set.
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        // And it still records no circuit-breaker sample, because
        // `record_outcome` returns on the capacity set before it ever reads
        // `retryable_status_codes`.
        assert!(resolved.capacity_status_codes.contains(&429));
    }
}
