//! Absolute per-worker overload protection.
//!
//! The predicate is evaluated once per ingested load report, never per request;
//! the verdict is latched into routing state, which selection already reads.
//!
//! On by default, steering only: a worker over either threshold is left out
//! of selection while another worker is under them, and when every worker is
//! over them the request goes to the least-loaded of them (see
//! `routers::common::overload`). Refusing requests with a 503 instead is the
//! opt-in `--worker-overload-shed`.

#[cfg(feature = "runtime")]
use tracing::info;

#[cfg(feature = "runtime")]
use crate::identity::basic::Worker;
use crate::worker::{OverloadUpdate, WorkerLoadResponse};

/// Default KV token-usage ceiling (mean across DP ranks). On the replay
/// harness this is the setting that keeps `cache_aware`'s hit rate while
/// cutting its tail; the hardware fleet sits near 0.9 under load, so a
/// higher ceiling never fires there.
pub const DEFAULT_TOKEN_USAGE_CEILING: f64 = 0.8;

/// Default waiting-requests ceiling (summed across DP ranks): the signal that
/// trips first on the hardware fleet, where the KV pools fill slowly and the
/// queue forms in front of the engine.
pub const DEFAULT_WAITING_REQUESTS: usize = 8;

/// Decision-log branch: every worker selection could have used is overloaded,
/// so the request is shed immediately instead of queued.
pub const BRANCH_ALL_OVERLOADED_SHED: &str = "all_overloaded_shed";

/// Decision-log branch: the single worker already chosen crossed the threshold
/// between selection and dispatch. Distinct from the fleet-wide shed above —
/// here every other worker may well be idle.
pub const BRANCH_OVERLOADED_AT_DISPATCH: &str = "overloaded_at_dispatch";

/// Decision-log branch: every worker selection could have used is over the
/// thresholds and shedding is off, so the request goes to the least-loaded
/// of them instead of being refused (a cache-aware prefill pool of a
/// prefill/decode pair, or a regular request that consistent_hashing places
/// by its routing key, is placed by its policy instead).
pub const BRANCH_ALL_OVERLOADED_FALLBACK: &str = "all_overloaded_fallback";

/// Decision-log branch: every worker selection could have used is vetoed and
/// the only ones left ready are vetoed by the liveness tracker, so the request
/// goes to the least-loaded of them instead of being refused: a veto steers,
/// it never empties the pool.
pub const BRANCH_ALL_STALLED_FALLBACK: &str = "all_stalled_fallback";

/// Decision-log branch: the decode side of a disaggregated pair was already
/// running its full engine window and no slot freed inside the admission wait.
/// The pair is not overloaded by the threshold predicate above — the gateway
/// simply has nowhere to put another bootstrap room.
pub const BRANCH_PD_ADMISSION_SHED: &str = "pd_admission_shed";

/// Shed detected while assembling the candidate pool.
pub const STAGE_SELECTION: &str = "selection";

/// Shed detected by the dispatch-time re-check of the chosen worker.
pub const STAGE_DISPATCH: &str = "dispatch";

/// Shed detected by the PD admission gate, which runs between selection and
/// the paired dispatch.
pub const STAGE_PD_ADMISSION: &str = "pd_admission";

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

/// What the line of an overload veto's edge states of the worker beside
/// the verdict: its load figures as the veto's writer read them, and its
/// thresholds.
///
/// The line does not say that a figure crossed a threshold. A writer that
/// judges the worker's own report against the worker's own thresholds
/// states what it judged on. A writer that mirrors a verdict judged
/// elsewhere states the worker's figures and thresholds as it holds them,
/// and they need not explain the verdict: such a judge may weigh several
/// ranks together, or hold them against thresholds the worker does not
/// name. The line's `source` says who judged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverloadReading {
    /// Queued requests, as the writer read them.
    pub waiting_requests: i64,
    /// KV pressure, as the writer read it.
    pub kv_pressure: f64,
    /// The worker's thresholds, as the writer holds them.
    pub thresholds: OverloadThresholds,
}

/// Write the INFO line of an overload veto's edge on `worker`: the verdict,
/// the worker's figures and thresholds ([`OverloadReading`]) and who judged
/// (`source`). A worker vetoed by overload reads as "unavailable" in a
/// refusal line exactly like one with an open breaker, and the breaker and
/// the liveness veto log their edges already. Every writer of the veto
/// calls this, so the line reads alike whoever judged. It keeps the load
/// monitor's log target, where it was first written.
#[cfg(feature = "runtime")]
pub fn log_overload_edge(
    worker: &dyn Worker,
    overloaded: bool,
    reading: &OverloadReading,
    source: &'static str,
) {
    info!(
        target: "smg_worker::load::monitor",
        worker_url = %worker.redacted_url(),
        overloaded,
        waiting_requests = reading.waiting_requests,
        kv_pressure = reading.kv_pressure,
        waiting_threshold = ?reading.thresholds.waiting_requests,
        kv_pressure_threshold = ?reading.thresholds.token_usage,
        source,
        "Worker overload veto {}",
        if overloaded { "set" } else { "cleared" }
    );
}

impl OverloadThresholds {
    /// Whether any signal is configured. Gates both the monitor's polling
    /// requirement and every write to the flag.
    pub const fn is_enabled(&self) -> bool {
        self.waiting_requests.is_some() || self.token_usage.is_some()
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
            if load.effective_kv_pressure() >= threshold {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::SchedulerLoadSnapshot;

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
