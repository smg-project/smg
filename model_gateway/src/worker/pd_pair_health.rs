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
//! handle on the registry. The placement path pays nothing while no pair is
//! quarantined: `QUARANTINED` gates every lookup.

use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
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
/// instead of allocating a pair key on the placement path. An inner map is
/// dropped as soon as it empties, so the table is empty whenever no pair has
/// a pending failure.
static PAIRS: LazyLock<DashMap<String, DashMap<String, PairState>>> = LazyLock::new(DashMap::new);
/// Pairs whose `quarantined_until` is set. Counts an expired quarantine
/// until a lookup on that pair clears it, so it never undercounts.
static QUARANTINED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default)]
struct PairState {
    consecutive_failures: u32,
    /// When the last failure was recorded: a run older than one quarantine
    /// window is not consecutive any more.
    last_failure: Option<Instant>,
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

/// Pairs currently counted as quarantined (the `smg_pd_pairs_quarantined`
/// gauge). An expired quarantine stays counted until the next lookup on
/// that pair clears it, which placement does on its next decision.
pub fn quarantined_count() -> usize {
    QUARANTINED.load(Ordering::Relaxed)
}

fn uncount(pairs: usize) {
    let _ = QUARANTINED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(pairs))
    });
    Metrics::set_pd_pairs_quarantined(quarantined_count());
}

/// Whether placement should steer around this pair right now.
pub fn is_quarantined(prefill: &str, decode: &str) -> bool {
    if !enabled() || quarantined_count() == 0 {
        return false;
    }
    let now = Instant::now();
    let Some(by_decode) = PAIRS.get(prefill) else {
        return false;
    };
    match by_decode
        .get(decode)
        .and_then(|state| state.quarantined_until)
    {
        Some(until) if now < until => true,
        // Expired since it was entered: clear the flag so the count and
        // gauge follow. The failure count stays, so the next failure
        // re-quarantines at once.
        Some(_) => {
            if let Some(mut state) = by_decode.get_mut(decode) {
                if state.quarantined_until.is_some_and(|until| until <= now) {
                    state.quarantined_until = None;
                    drop(state);
                    uncount(1);
                }
            }
            false
        }
        None => false,
    }
}

/// A rendezvous or first-response failure on the pair. Returns `true` when
/// this failure put the pair into quarantine.
pub fn record_failure(prefill: &str, decode: &str) -> bool {
    if !enabled() {
        return false;
    }
    let now = Instant::now();
    let quarantine = Duration::from_secs(QUARANTINE_SECS.load(Ordering::Relaxed));
    let by_decode = PAIRS.entry(prefill.to_string()).or_default();
    let mut state = by_decode.entry(decode.to_string()).or_default();
    // Failures count as a run only within one quarantine window of each
    // other: an unrelated failure hours later does not extend a run. A pair
    // that already reached the threshold keeps its count, so the one probe
    // placement lets through after expiry re-quarantines it at once.
    let aged_out = state
        .last_failure
        .is_some_and(|last| now.duration_since(last) > quarantine);
    if aged_out && state.consecutive_failures < FAILURES.load(Ordering::Relaxed) {
        state.consecutive_failures = 0;
    }
    state.last_failure = Some(now);
    state.consecutive_failures = state.consecutive_failures.saturating_add(1);
    if state.consecutive_failures < FAILURES.load(Ordering::Relaxed) || state.quarantined(now) {
        return false;
    }
    // An expired quarantine nobody looked up yet is still counted.
    let counted = state.quarantined_until.is_some();
    state.quarantined_until = Some(now + quarantine);
    let consecutive_failures = state.consecutive_failures;
    drop(state);
    drop(by_decode);
    if !counted {
        QUARANTINED.fetch_add(1, Ordering::Relaxed);
    }
    warn!(
        prefill,
        decode,
        consecutive_failures,
        quarantine_secs = quarantine.as_secs(),
        "Quarantining PD pair after consecutive rendezvous failures"
    );
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
    let removed = by_decode.remove(decode).map(|(_, state)| state);
    let emptied = by_decode.is_empty();
    drop(by_decode);
    if emptied {
        PAIRS.remove_if(prefill, |_, by_decode| by_decode.is_empty());
    }
    if removed.is_some_and(|state| state.quarantined_until.is_some()) {
        debug!(
            prefill,
            decode, "PD pair cleared quarantine after a successful handoff"
        );
        uncount(1);
    }
}

/// Drop every pair the worker took part in (the worker was removed).
pub fn forget(url: &str) {
    if PAIRS.is_empty() {
        return;
    }
    let flagged = |state: &PairState| state.quarantined_until.is_some();
    let mut cleared = 0;
    if let Some((_, by_decode)) = PAIRS.remove(url) {
        cleared += by_decode.iter().filter(|state| flagged(state)).count();
    }
    for by_decode in PAIRS.iter() {
        if let Some((_, state)) = by_decode.remove(url) {
            cleared += usize::from(flagged(&state));
        }
    }
    PAIRS.retain(|_, by_decode| !by_decode.is_empty());
    if cleared > 0 {
        uncount(cleared);
    }
}

/// Serialises tests that touch the table; they assert on its global state.
#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Pairs with a pending failure count, quarantined or not.
#[cfg(test)]
pub(crate) fn tracked_pairs() -> usize {
    PAIRS.iter().map(|by_decode| by_decode.len()).sum()
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
        assert_eq!(quarantined_count(), 1);
        // Only the failing pair is affected.
        assert!(!is_quarantined(p, "grpc://health-d:2"));
        assert!(!is_quarantined("grpc://health-p:2", d));
        // Already quarantined: another failure does not re-quarantine.
        assert!(!record_failure(p, d));
        assert_eq!(quarantined_count(), 1);

        record_success(p, d);
        assert!(!is_quarantined(p, d));
        assert_eq!(quarantined_count(), 0);
        // A success on the only pending pair leaves the table empty, so the
        // placement fast path is back.
        assert_eq!(tracked_pairs(), 0);
    }

    #[test]
    fn a_quarantine_expires_and_a_removed_worker_is_forgotten() {
        let _guard = test_guard();
        configure(1, 1);
        let (p, d) = ("grpc://expiry-p:1", "grpc://expiry-d:1");
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
        assert_eq!(quarantined_count(), 1);
        std::thread::sleep(Duration::from_millis(1100));
        // The lookup that sees the expiry clears the count.
        assert!(!is_quarantined(p, d));
        assert_eq!(quarantined_count(), 0);
        // The failure count persists, so the next failure re-quarantines.
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
        assert_eq!(quarantined_count(), 1);

        // Removing the decode drops the pair and its quarantine.
        forget(d);
        assert!(!is_quarantined(p, d));
        assert_eq!(quarantined_count(), 0);
        assert_eq!(tracked_pairs(), 0);

        // Removing the prefill drops every pair it led.
        assert!(record_failure(p, d));
        forget(p);
        assert_eq!(quarantined_count(), 0);
        assert_eq!(tracked_pairs(), 0);
        configure(
            DEFAULT_PD_PAIR_QUARANTINE_FAILURES,
            DEFAULT_PD_PAIR_QUARANTINE_SECS,
        );
    }

    #[test]
    fn failures_further_apart_than_the_window_are_not_a_run() {
        let _guard = test_guard();
        configure(2, 1);
        let (p, d) = ("grpc://stale-p:1", "grpc://stale-d:1");
        assert!(!record_failure(p, d));
        std::thread::sleep(Duration::from_millis(1100));
        // The old failure has aged out: this one starts a new run.
        assert!(!record_failure(p, d));
        assert!(!is_quarantined(p, d));
        // Two within the window do quarantine.
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
        // Past the threshold the count is kept: the one probe placement
        // lets through after expiry re-quarantines at once.
        std::thread::sleep(Duration::from_millis(1100));
        assert!(!is_quarantined(p, d));
        assert!(record_failure(p, d));
        assert!(is_quarantined(p, d));
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
        assert_eq!(tracked_pairs(), 0);
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
