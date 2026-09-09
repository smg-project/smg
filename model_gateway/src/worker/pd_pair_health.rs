//! Pair-level health for PD placement (#2483).
//!
//! The pairing descriptor ([`super::pd_pairing`]) keeps a prefill and a
//! decode apart when their labels say a KV handoff cannot work. This table is
//! the defence for the mismatch the labels miss: an engine-side failure at
//! the rendezvous, or before the decode's first response, counts against the
//! (prefill, decode) pair, and a run of consecutive failures quarantines the
//! pair for a bounded time. Placement steers around a quarantined pair while
//! another compatible pair is open and keeps serving through it when none
//! is: the quarantine is a hint drawn from failures, not a veto. The decode's
//! first response is the proof the handoff completed and clears the pair.
//!
//! Like the PD admission wait, the thresholds and the table are
//! process-global: the dispatch paths that observe a rendezvous carry no
//! handle on the registry.

use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        LazyLock,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tracing::{debug, warn};

use crate::observability::metrics::Metrics;

/// Consecutive failures that quarantine a pair. `0` disables the table.
pub const DEFAULT_PD_PAIR_QUARANTINE_FAILURES: u32 = 3;
/// How long a quarantined pair is steered around. `0` disables the table.
pub const DEFAULT_PD_PAIR_QUARANTINE_SECS: u64 = 30;

static FAILURES: AtomicU32 = AtomicU32::new(DEFAULT_PD_PAIR_QUARANTINE_FAILURES);
static QUARANTINE_SECS: AtomicU64 = AtomicU64::new(DEFAULT_PD_PAIR_QUARANTINE_SECS);
/// Prefill URL, then decode URL: two levels so a lookup borrows `&str`
/// instead of allocating a pair key on the placement path.
static PAIRS: LazyLock<DashMap<String, DashMap<String, PairState>>> = LazyLock::new(DashMap::new);

#[derive(Debug, Default)]
struct PairState {
    consecutive_failures: u32,
    quarantined_until: Option<Instant>,
}

impl PairState {
    fn quarantined(&self, now: Instant) -> bool {
        self.quarantined_until.is_some_and(|until| now < until)
    }
}

/// Latch the thresholds at startup; either `0` disables quarantining.
pub fn configure(failures: u32, quarantine_secs: u64) {
    FAILURES.store(failures, Ordering::Relaxed);
    QUARANTINE_SECS.store(quarantine_secs, Ordering::Relaxed);
}

fn enabled() -> bool {
    FAILURES.load(Ordering::Relaxed) > 0 && QUARANTINE_SECS.load(Ordering::Relaxed) > 0
}

/// Whether a leg's failure says something about the pair rather than about
/// one worker: an engine-side error other than an overload shed or an
/// unavailable worker, which the per-worker circuit breaker and overload
/// protection already own.
pub fn pair_attributable(status: u16) -> bool {
    status >= 500 && status != 503
}

/// Whether placement should steer around this pair right now.
pub fn is_quarantined(prefill: &str, decode: &str) -> bool {
    if !enabled() || PAIRS.is_empty() {
        return false;
    }
    let now = Instant::now();
    PAIRS.get(prefill).is_some_and(|by_decode| {
        by_decode
            .get(decode)
            .is_some_and(|state| state.quarantined(now))
    })
}

/// A rendezvous or first-response failure on the pair. Returns `true` when
/// this failure put the pair into quarantine.
pub fn record_failure(prefill: &str, decode: &str) -> bool {
    if !enabled() {
        return false;
    }
    let now = Instant::now();
    let by_decode = PAIRS.entry(prefill.to_string()).or_default();
    let mut state = by_decode.entry(decode.to_string()).or_default();
    state.consecutive_failures = state.consecutive_failures.saturating_add(1);
    if state.consecutive_failures < FAILURES.load(Ordering::Relaxed) || state.quarantined(now) {
        return false;
    }
    let quarantine = Duration::from_secs(QUARANTINE_SECS.load(Ordering::Relaxed));
    state.quarantined_until = Some(now + quarantine);
    warn!(
        prefill,
        decode,
        consecutive_failures = state.consecutive_failures,
        quarantine_secs = quarantine.as_secs(),
        "Quarantining PD pair after consecutive rendezvous failures"
    );
    // Both guards must be released before the count walks the table.
    drop(state);
    drop(by_decode);
    Metrics::record_pd_pair_quarantine(quarantined_count());
    true
}

/// The decode leg answered: the handoff completed, so the pair is healthy.
pub fn record_success(prefill: &str, decode: &str) {
    if !enabled() || PAIRS.is_empty() {
        return;
    }
    let Some(by_decode) = PAIRS.get(prefill) else {
        return;
    };
    let Some((_, state)) = by_decode.remove(decode) else {
        return;
    };
    drop(by_decode);
    if state.quarantined_until.is_some() {
        debug!(
            prefill,
            decode, "PD pair cleared quarantine after a successful handoff"
        );
        Metrics::set_pd_pairs_quarantined(quarantined_count());
    }
}

/// Drop every pair the worker took part in (the worker was removed).
pub fn forget(url: &str) {
    if PAIRS.is_empty() {
        return;
    }
    PAIRS.remove(url);
    for by_decode in PAIRS.iter() {
        by_decode.remove(url);
    }
    PAIRS.retain(|_, by_decode| !by_decode.is_empty());
}

/// Pairs currently in quarantine.
pub fn quarantined_count() -> usize {
    let now = Instant::now();
    PAIRS
        .iter()
        .map(|by_decode| {
            by_decode
                .iter()
                .filter(|state| state.quarantined(now))
                .count()
        })
        .sum()
}

/// Serialises tests that change the thresholds or count the whole table;
/// tests that only touch their own URLs need not take it.
#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_is_quarantined_after_consecutive_failures_and_cleared_by_a_success() {
        let _guard = test_guard();
        let (p, d) = ("grpc://health-p:1", "grpc://health-d:1");
        for _ in 1..DEFAULT_PD_PAIR_QUARANTINE_FAILURES {
            assert!(!record_failure(p, d));
            assert!(!is_quarantined(p, d));
        }
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
        // Only the failing pair is affected.
        assert!(!is_quarantined(p, "grpc://health-d:2"));
        assert!(!is_quarantined("grpc://health-p:2", d));
        // Already quarantined: another failure does not re-quarantine.
        assert!(!record_failure(p, d));

        record_success(p, d);
        assert!(!is_quarantined(p, d));
        forget(p);
    }

    #[test]
    fn a_quarantine_expires_and_a_removed_worker_is_forgotten() {
        let _guard = test_guard();
        configure(1, 1);
        let (p, d) = ("grpc://expiry-p:1", "grpc://expiry-d:1");
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
        std::thread::sleep(Duration::from_millis(1100));
        assert!(!is_quarantined(p, d));
        // The count persists, so the next failure re-quarantines at once.
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));

        forget(d);
        assert!(!is_quarantined(p, d));
        assert!(!record_failure(p, d) || quarantined_count() >= 1);
        forget(p);
        configure(
            DEFAULT_PD_PAIR_QUARANTINE_FAILURES,
            DEFAULT_PD_PAIR_QUARANTINE_SECS,
        );
    }

    #[test]
    fn disabled_thresholds_record_nothing() {
        let _guard = test_guard();
        configure(0, DEFAULT_PD_PAIR_QUARANTINE_SECS);
        let (p, d) = ("grpc://off-p:1", "grpc://off-d:1");
        for _ in 0..5 {
            assert!(!record_failure(p, d));
        }
        assert!(!is_quarantined(p, d));
        configure(
            DEFAULT_PD_PAIR_QUARANTINE_FAILURES,
            DEFAULT_PD_PAIR_QUARANTINE_SECS,
        );
    }

    #[test]
    fn only_engine_side_errors_count_against_a_pair() {
        assert!(pair_attributable(500));
        assert!(pair_attributable(504));
        assert!(!pair_attributable(503));
        assert!(!pair_attributable(429));
        assert!(!pair_attributable(400));
        assert!(!pair_attributable(200));
    }
}
