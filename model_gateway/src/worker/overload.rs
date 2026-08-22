//! Absolute per-worker overload protection.
//!
//! The predicate is evaluated once per ingested load report, never per request;
//! the verdict is latched into routing state, which selection already reads.

use openai_protocol::worker::{OverloadUpdate, WorkerLoadResponse};

use crate::config::types::RouterConfig;

/// Token-usage ceiling applied when protection is enabled by
/// `--worker-overload-protection` alone. KV token usage means the same thing
/// on every engine; a waiting-requests default would be workload-dependent, so
/// that signal stays unset unless configured.
pub const DEFAULT_TOKEN_USAGE_CEILING: f64 = 0.9;

/// Decision-log branch: every worker selection could have used is overloaded,
/// so the request is shed immediately instead of queued.
pub const BRANCH_ALL_OVERLOADED_SHED: &str = "all_overloaded_shed";

/// Decision-log branch: the single worker already chosen crossed the threshold
/// between selection and dispatch. Distinct from the fleet-wide shed above —
/// here every other worker may well be idle.
pub const BRANCH_OVERLOADED_AT_DISPATCH: &str = "overloaded_at_dispatch";

/// Shed detected while assembling the candidate pool.
pub const STAGE_SELECTION: &str = "selection";

/// Shed detected by the dispatch-time re-check of the chosen worker.
pub const STAGE_DISPATCH: &str = "dispatch";

/// Absolute thresholds above which a worker is vetoed from routing.
///
/// Both fields unset disables the feature entirely: the flag is never written,
/// so every routing path behaves exactly as it did before.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OverloadThresholds {
    /// Queued (waiting) requests summed across DP ranks, `>= 1` when set.
    pub waiting_requests: Option<usize>,
    /// Mean KV-cache token usage across DP ranks, in `(0.0, 1.0]` when set.
    /// `f64` to match the reported signal exactly: widening an `f32` threshold
    /// would put `0.8` just above the `0.8` an engine reports.
    pub token_usage: Option<f64>,
}

impl OverloadThresholds {
    /// Whether any signal is configured. Gates both the monitor's polling
    /// requirement and every write to the flag.
    pub const fn is_enabled(&self) -> bool {
        self.waiting_requests.is_some() || self.token_usage.is_some()
    }

    /// Gateway-level thresholds: the explicit `--worker-overload-*` values,
    /// with the token ceiling defaulted to [`DEFAULT_TOKEN_USAGE_CEILING`]
    /// when `--worker-overload-protection` enables the feature without one.
    /// Everything unset with the flag off disables protection — exact #2220
    /// behavior.
    pub fn from_gateway_config(config: &RouterConfig) -> Self {
        Self {
            waiting_requests: config.worker_overload_waiting_requests,
            token_usage: config.worker_overload_token_usage.or_else(|| {
                config
                    .worker_overload_protection
                    .then_some(DEFAULT_TOKEN_USAGE_CEILING)
            }),
        }
    }

    /// Per-worker effective thresholds: each signal takes the worker's spec
    /// override, else the gateway value. Resolved once at registration; the
    /// monitor's ingestion predicate reads the result per report.
    pub fn resolve(overrides: &OverloadUpdate, gateway: Self) -> Self {
        Self {
            waiting_requests: overrides.waiting_requests.or(gateway.waiting_requests),
            token_usage: overrides.token_usage.or(gateway.token_usage),
        }
    }

    /// Evaluate the veto for one load report. O(DP ranks), i.e. O(1) per report.
    ///
    /// Both comparisons are `>=` so that the excluded ends of the validated
    /// ranges (`0` waiting requests, `0.0` token usage) are exactly the values
    /// that would mark every worker overloaded unconditionally.
    pub fn is_overloaded(&self, load: &WorkerLoadResponse) -> bool {
        if let Some(threshold) = self.waiting_requests {
            if load.total_waiting_reqs() >= threshold as i64 {
                return true;
            }
        }
        if let Some(threshold) = self.token_usage {
            if load.effective_token_usage() >= threshold {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::SchedulerLoadSnapshot;

    use super::*;

    fn response(waiting: i32, token_usage: f64) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_reqs: waiting,
                token_usage,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn unset_thresholds_never_veto() {
        let thresholds = OverloadThresholds::default();
        assert!(!thresholds.is_enabled());
        assert!(!thresholds.is_overloaded(&response(10_000, 1.0)));
    }

    #[test]
    fn waiting_requests_vetoes_at_and_above_threshold() {
        let thresholds = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: None,
        };
        assert!(!thresholds.is_overloaded(&response(7, 0.0)));
        assert!(thresholds.is_overloaded(&response(8, 0.0)));
        assert!(thresholds.is_overloaded(&response(9, 0.0)));
    }

    #[test]
    fn token_usage_vetoes_at_and_above_threshold() {
        let thresholds = OverloadThresholds {
            waiting_requests: None,
            token_usage: Some(0.9),
        };
        assert!(!thresholds.is_overloaded(&response(0, 0.89)));
        assert!(thresholds.is_overloaded(&response(0, 0.9)));
        assert!(thresholds.is_overloaded(&response(0, 0.95)));
    }

    #[test]
    fn signals_are_independent_and_either_vetoes() {
        let thresholds = OverloadThresholds {
            waiting_requests: Some(4),
            token_usage: Some(0.8),
        };
        assert!(!thresholds.is_overloaded(&response(3, 0.7)));
        assert!(thresholds.is_overloaded(&response(4, 0.1)));
        assert!(thresholds.is_overloaded(&response(0, 0.8)));
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

    /// The flag alone enables protection with the engine-universal token
    /// ceiling and no waiting-requests threshold.
    #[test]
    fn protection_flag_alone_defaults_the_token_ceiling() {
        let thresholds = OverloadThresholds::from_gateway_config(&gateway_config(true, None, None));
        assert_eq!(
            thresholds,
            OverloadThresholds {
                waiting_requests: None,
                token_usage: Some(DEFAULT_TOKEN_USAGE_CEILING),
            }
        );
        assert!(thresholds.is_enabled());
    }

    /// Explicit thresholds override the flag's default; without either, the
    /// feature stays exactly off (#2220 back-compat).
    #[test]
    fn explicit_thresholds_override_the_flag_default() {
        let explicit =
            OverloadThresholds::from_gateway_config(&gateway_config(true, Some(8), Some(0.5)));
        assert_eq!(explicit.token_usage, Some(0.5));
        assert_eq!(explicit.waiting_requests, Some(8));

        let no_flag =
            OverloadThresholds::from_gateway_config(&gateway_config(false, Some(8), None));
        assert_eq!(
            no_flag,
            OverloadThresholds {
                waiting_requests: Some(8),
                token_usage: None,
            }
        );

        let off = OverloadThresholds::from_gateway_config(&gateway_config(false, None, None));
        assert!(!off.is_enabled());
    }

    /// Per-signal resolution: a worker override beats the gateway default for
    /// its signal only; the other signal keeps the gateway value.
    #[test]
    fn resolve_applies_worker_overrides_per_signal() {
        let gateway = OverloadThresholds {
            waiting_requests: Some(16),
            token_usage: Some(DEFAULT_TOKEN_USAGE_CEILING),
        };
        let resolved = OverloadThresholds::resolve(
            &OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.5),
            },
            gateway,
        );
        assert_eq!(
            resolved,
            OverloadThresholds {
                waiting_requests: Some(16),
                token_usage: Some(0.5),
            }
        );

        // A spec block alone enables the worker even with protection off.
        let solo = OverloadThresholds::resolve(
            &OverloadUpdate {
                waiting_requests: Some(4),
                token_usage: None,
            },
            OverloadThresholds::default(),
        );
        assert!(solo.is_enabled());
        assert_eq!(solo.waiting_requests, Some(4));
        assert_eq!(solo.token_usage, None);

        // No block, protection off: resolution stays disabled.
        assert!(!OverloadThresholds::resolve(
            &OverloadUpdate::default(),
            OverloadThresholds::default()
        )
        .is_enabled());
    }

    #[test]
    fn token_usage_averages_across_dp_ranks() {
        let thresholds = OverloadThresholds {
            waiting_requests: None,
            token_usage: Some(0.9),
        };
        let load = WorkerLoadResponse {
            loads: vec![
                SchedulerLoadSnapshot {
                    token_usage: 1.0,
                    ..Default::default()
                },
                SchedulerLoadSnapshot {
                    token_usage: 0.5,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Mean 0.75 — one saturated rank does not veto the whole worker.
        assert!(!thresholds.is_overloaded(&load));
    }
}
