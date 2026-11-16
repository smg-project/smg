//! Prometheus metrics for the priority scheduler.
//!
//! Split to match design §9: this module holds the **operational** counters
//! and the queue-wait histogram (recorded on the admission / queue paths),
//! plus the **capacity / autoscaling** gauge setters. The gauges are
//! point-in-time state (inflight, queue depth, utilization, capacity
//! pressure) refreshed by the scheduler's sampler task, so the hot
//! admission path only does cheap counter increments.
//!
//! All names carry the `smg_` prefix to match the rest of the gateway's
//! metrics. Class/outcome labels are `&'static str` (no per-request
//! allocation).

use std::{sync::Arc, time::Duration};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

use super::Class;

const ADMIT_TOTAL: &str = "smg_scheduler_admit_total";
const QUEUE_WAIT_SECONDS: &str = "smg_scheduler_queue_wait_seconds";
const PREEMPTION_TOTAL: &str = "smg_scheduler_preemption_total";
const CLAMP_TOTAL: &str = "smg_scheduler_clamp_total";
const UNKNOWN_PRIORITY_TOTAL: &str = "smg_scheduler_unknown_priority_value_total";
const STARVATION_PROMOTION_TOTAL: &str = "smg_scheduler_starvation_promotion_total";

// Capacity / autoscaling gauges, refreshed by the sampler task.
const INFLIGHT: &str = "smg_scheduler_inflight";
const QUEUE_DEPTH: &str = "smg_scheduler_queue_depth";
const UTILIZATION: &str = "smg_scheduler_utilization";
const QUEUE_SIZE_LIMIT: &str = "smg_scheduler_queue_size_limit";
const CLASS_CAPACITY_PRESSURE: &str = "smg_scheduler_class_capacity_pressure";

/// `outcome` label values for [`record_admit`].
pub mod outcome {
    /// Admitted (fast path or after queueing — not distinguished).
    pub const ADMITTED: &str = "admitted";
    /// Per-class queue was at its limit.
    pub const REJECTED_QUEUE_FULL: &str = "rejected_queue_full";
    /// Queued waiter aged past `queue_timeout`.
    pub const REJECTED_QUEUE_TIMEOUT: &str = "rejected_queue_timeout";
    /// The request was admitted but then preempted before producing a byte.
    pub const PREEMPTED: &str = "preempted";
    /// The caller's client disconnected before admission completed.
    pub const CLIENT_CANCELLED: &str = "client_cancelled";
}

/// Register descriptions. Called once from `observability::metrics::init_metrics`.
pub fn describe() {
    describe_counter!(
        ADMIT_TOTAL,
        "Priority-scheduler admission outcomes by class and outcome"
    );
    describe_histogram!(
        QUEUE_WAIT_SECONDS,
        "Time a request spent queued before admission, timeout, or cancel"
    );
    describe_counter!(
        PREEMPTION_TOTAL,
        "Successful preemptions by victim class and preempting class"
    );
    describe_counter!(
        CLAMP_TOTAL,
        "Requests whose priority was clamped below the requested class by tenant policy"
    );
    describe_counter!(
        UNKNOWN_PRIORITY_TOTAL,
        "Requests with an unrecognized priority header value (treated as default)"
    );
    describe_counter!(
        STARVATION_PROMOTION_TOTAL,
        "Queued waiters admitted via the starvation override path"
    );
    describe_gauge!(INFLIGHT, "Current in-flight request count per class");
    describe_gauge!(QUEUE_DEPTH, "Current queued waiter count per class");
    describe_gauge!(
        UTILIZATION,
        "Total in-flight requests divided by backend capacity (0.0-1.0+)"
    );
    describe_gauge!(QUEUE_SIZE_LIMIT, "Configured queue limit per class");
    describe_gauge!(
        CLASS_CAPACITY_PRESSURE,
        "Normalized 0.0-1.0 per-class pressure (max of queue and slot pressure)"
    );
}

/// Record the outcome of an admission attempt for `class`.
pub fn record_admit(class: Class, outcome: &'static str) {
    counter!(ADMIT_TOTAL, "class" => class.as_str(), "outcome" => outcome).increment(1);
}

/// Record the time a request waited in a class queue.
pub fn record_queue_wait(class: Class, wait: Duration) {
    histogram!(QUEUE_WAIT_SECONDS, "class" => class.as_str()).record(wait.as_secs_f64());
}

/// Record a successful preemption.
pub fn record_preemption(victim_class: Class, by_class: Class) {
    counter!(
        PREEMPTION_TOTAL,
        "victim_class" => victim_class.as_str(),
        "by_class" => by_class.as_str()
    )
    .increment(1);
}

/// Record a priority clamp (only when the effective class is below the
/// requested class). `tenant` is the key as the scheduler's bounded label
/// set interned it (`SchedulerState::tenant_labels`, sized from
/// `tenant_metric_top_n`): the keys are client-chosen, so a tenant past the
/// cap counts under `other` rather than minting a series of its own.
pub fn record_clamp(requested: Class, effective: Class, tenant: Arc<str>) {
    counter!(
        CLAMP_TOTAL,
        "tenant" => tenant,
        "requested_class" => requested.as_str(),
        "effective_class" => effective.as_str()
    )
    .increment(1);
}

/// Record an unrecognized priority header value under the tenant that sent
/// it, the actionable dimension (it tells ops which tenant is mis-setting
/// the priority header). `tenant` is the scheduler's bounded label, as for
/// [`record_clamp`]: a tenant past the cap counts under `other`.
pub fn record_unknown_priority(tenant: Arc<str>) {
    counter!(UNKNOWN_PRIORITY_TOTAL, "tenant" => tenant).increment(1);
}

/// Record a starvation-override promotion.
pub fn record_starvation_promotion(class: Class) {
    counter!(STARVATION_PROMOTION_TOTAL, "class" => class.as_str()).increment(1);
}

/// Set the in-flight gauge for a class (sampler).
pub fn set_inflight(class: Class, count: u16) {
    gauge!(INFLIGHT, "class" => class.as_str()).set(f64::from(count));
}

/// Set the queue-depth gauge for a class (sampler).
pub fn set_queue_depth(class: Class, depth: usize) {
    gauge!(QUEUE_DEPTH, "class" => class.as_str()).set(depth as f64);
}

/// Set the overall utilization gauge: total in-flight / capacity (sampler).
pub fn set_utilization(utilization: f64) {
    gauge!(UTILIZATION).set(utilization);
}

/// Set the queue-size-limit gauge for a class (sampler).
pub fn set_queue_size_limit(class: Class, limit: usize) {
    gauge!(QUEUE_SIZE_LIMIT, "class" => class.as_str()).set(limit as f64);
}

/// Set the normalized capacity-pressure gauge for a class (sampler).
pub fn set_class_capacity_pressure(class: Class, pressure: f64) {
    gauge!(CLASS_CAPACITY_PRESSURE, "class" => class.as_str()).set(pressure);
}

#[cfg(test)]
mod tests {
    use smg_external_router::metrics::BoundedLabels;

    use super::*;
    use crate::observability::metrics::{interner_size, test_support::render_with_recorder};

    /// Forty distinct tenant keys at a cap of 32 render 32 tenant series of
    /// their own and one `other` series carrying the rest; the process-wide
    /// interner, which never frees, does not grow with them.
    #[test]
    fn tenants_past_the_cap_count_under_other_and_grow_no_interner() {
        let labels = BoundedLabels::new(32);
        let interned_before = interner_size();
        let rendered = render_with_recorder(|| {
            for tenant in 0..40 {
                record_unknown_priority(labels.intern(&format!("header:tenant-{tenant}")));
            }
        });
        let series: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with(&format!("{UNKNOWN_PRIORITY_TOTAL}{{")))
            .collect();
        assert_eq!(series.len(), 33, "32 tenants and the sentinel:\n{rendered}");
        let other = series
            .iter()
            .find(|line| line.contains("tenant=\"other\""))
            .expect("the sentinel series");
        assert!(other.ends_with(" 8"), "the eight past the cap: {other}");
        // Other tests of the process intern labels of their own meanwhile;
        // forty tenant keys through the never-freed interner would be forty.
        assert!(
            interner_size().saturating_sub(interned_before) < 40,
            "the tenant keys never reach the process-wide interner"
        );
    }
}
