//! The discovery series: what the providers and the reconciler record.

use std::time::Duration;

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

#[cfg(any(test, feature = "kubernetes"))]
pub(super) const DISCOVERY_KUBERNETES: &str = "kubernetes";
#[cfg(any(test, feature = "file"))]
pub(super) const DISCOVERY_FILE: &str = "file";
#[cfg(any(test, feature = "slurm"))]
pub(super) const DISCOVERY_SLURM: &str = "slurm";
#[cfg(any(test, feature = "consul"))]
pub(super) const DISCOVERY_CONSUL: &str = "consul";

pub(super) const REGISTRATION_SUCCESS: &str = "success";
pub(super) const REGISTRATION_FAILED: &str = "failed";
pub(super) const DEREGISTRATION_RECONCILED: &str = "reconciled";

/// `reason` values of `smg_discovery_snapshot_errors_total`: the source could
/// not be read, or was read and rejected as invalid.
#[cfg(any(feature = "consul", feature = "file", feature = "slurm"))]
pub(super) const SNAPSHOT_ERROR_UNREADABLE: &str = "unreadable";
#[cfg(any(feature = "consul", feature = "file", feature = "slurm"))]
pub(super) const SNAPSHOT_ERROR_INVALID: &str = "invalid";

/// `reason` values of `smg_discovery_skipped_total`, for a worker a provider
/// left out of an otherwise good snapshot: its record could not be used, or
/// one started earlier at its address.
#[cfg(any(feature = "consul", feature = "slurm"))]
pub(super) const DISCOVERY_SKIP_INVALID: &str = "invalid";
#[cfg(any(feature = "consul", feature = "slurm"))]
pub(super) const DISCOVERY_SKIP_CONFLICT: &str = "conflict";

/// Describe the discovery series to the installed recorder. A host calls it
/// once, where it describes its own.
pub fn describe_series() {
    describe_counter!(
        "smg_discovery_registrations_total",
        "Worker registration attempts by source and result"
    );
    describe_counter!(
        "smg_discovery_deregistrations_total",
        "Worker deregistration events by source and reason"
    );
    describe_histogram!(
        "smg_discovery_sync_duration_seconds",
        "Discovery sync duration by source"
    );
    describe_gauge!(
        "smg_discovery_workers_discovered",
        "Workers known via discovery by source"
    );
    describe_counter!(
        "smg_discovery_snapshot_errors_total",
        "Discovery source observations rejected, by source and reason"
    );
    describe_counter!(
        "smg_discovery_skipped_total",
        "Discovered workers left out of an otherwise good snapshot, by source and reason"
    );
}

/// Record worker registration attempt
pub(super) fn record_registration(source: &'static str, result: &'static str) {
    counter!(
        "smg_discovery_registrations_total",
        "source" => source,
        "result" => result
    )
    .increment(1);
}

/// Record worker deregistration
pub(super) fn record_deregistration(source: &'static str, reason: &'static str) {
    counter!(
        "smg_discovery_deregistrations_total",
        "source" => source,
        "reason" => reason
    )
    .increment(1);
}

/// Record discovery sync duration
pub(super) fn record_sync_duration(source: &'static str, duration: Duration) {
    histogram!(
        "smg_discovery_sync_duration_seconds",
        "source" => source
    )
    .record(duration.as_secs_f64());
}

/// Set workers discovered count
pub(super) fn set_workers_discovered(source: &'static str, count: usize) {
    gauge!(
        "smg_discovery_workers_discovered",
        "source" => source
    )
    .set(count as f64);
}

/// Record a discovered worker a provider left out of a snapshot it
/// otherwise applied
#[cfg(any(feature = "consul", feature = "slurm"))]
pub(super) fn record_skip(source: &'static str, reason: &'static str) {
    counter!(
        "smg_discovery_skipped_total",
        "source" => source,
        "reason" => reason
    )
    .increment(1);
}

/// Record a discovery source observation that was rejected; the provider
/// keeps reconciling its last good snapshot meanwhile
#[cfg(any(feature = "consul", feature = "file", feature = "slurm"))]
pub(super) fn record_snapshot_error(source: &'static str, reason: &'static str) {
    counter!(
        "smg_discovery_snapshot_errors_total",
        "source" => source,
        "reason" => reason
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    /// Every family this file describes is recorded in it: its name stands
    /// once in a description and once in a recorder. The gateway's metric
    /// catalogue holds the gateway's own families to the same promise.
    #[test]
    fn every_described_family_is_recorded() {
        let source = include_str!("series.rs");
        let (code, _cases) = source.split_once("#[cfg(test)]").unwrap_or((source, ""));
        let named: Vec<&str> = code
            .split('"')
            .filter(|text| text.starts_with("smg_discovery_"))
            .collect();
        assert_eq!(named.len(), 12, "six families, each named twice: {named:?}");
        for family in &named {
            let times = named.iter().filter(|other| other == &family).count();
            assert_eq!(times, 2, "{family}");
        }
    }
}
