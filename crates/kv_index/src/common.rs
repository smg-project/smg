//! Common types shared between radix tree implementations.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Once,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use once_cell::sync::Lazy;
use rustc_hash::FxBuildHasher;
use tracing::warn;

/// Interned tenant identifier for efficient comparison and storage.
///
/// Using `Arc<str>` allows O(1) cloning and pointer-based equality checks
/// for frequently accessed tenant identifiers.
pub type TenantId = Arc<str>;

/// The tenant string intern pool both trees share: one `Arc<str>` per tenant
/// name, whichever tree saw it first.
static TENANT_INTERN_POOL: Lazy<DashMap<Arc<str>, (), FxBuildHasher>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));

/// The interned `TenantId` for `tenant`: the pooled `Arc<str>` when the name
/// was seen before, otherwise the one the pool holds after this call entered
/// it (one `Arc` per name even when two callers race on it).
#[inline]
pub(super) fn intern_tenant(tenant: &str) -> TenantId {
    if let Some(entry) = TENANT_INTERN_POOL.get(tenant) {
        return Arc::clone(entry.key());
    }
    // A miss goes through the entry: when two callers race on a new name,
    // both get the key the pool holds and the loser's allocation is dropped.
    let entry = TENANT_INTERN_POOL.entry(Arc::from(tenant)).or_insert(());
    Arc::clone(entry.key())
}

/// Trait for prefix match results.
///
/// This trait provides a common interface for accessing match results
/// from both string-based and token-based radix trees.
pub trait MatchResult {
    /// Get the tenant that owns the matched prefix.
    fn tenant(&self) -> &TenantId;

    /// Get the number of units (chars or tokens) that matched.
    fn matched_count(&self) -> usize;

    /// Get the total number of units in the input.
    fn input_count(&self) -> usize;

    /// Calculate the cache hit ratio (matched / input).
    ///
    /// Returns 0.0 if input is empty, otherwise returns a value in [0.0, 1.0].
    fn hit_ratio(&self) -> f64 {
        let input = self.input_count();
        if input == 0 {
            0.0
        } else {
            self.matched_count() as f64 / input as f64
        }
    }
}

/// Candidate cap for a match result's `matched_tenants`: enough for
/// pressure selection to discriminate, small enough to bound the per-match
/// allocation.
pub const MATCHED_TENANTS_CAP: usize = 8;

/// Origin of the seconds an access stamp carries: the process start, with the
/// wall-clock seconds since [`STAMP_ORIGIN_UNIX_SECS`] at that moment. Seconds
/// then advance by the monotonic clock, so a stamp never runs backwards with
/// the wall clock, yet reads about the same in every process.
static STAMP_EPOCH: Lazy<(Instant, u64)> = Lazy::new(|| {
    let unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    (
        Instant::now(),
        unix_secs.saturating_sub(STAMP_ORIGIN_UNIX_SECS),
    )
});

/// 2025-01-01T00:00:00Z. Counting from here keeps the seconds below 2^31 (a
/// shifted stamp below `i64::MAX`, as eviction's priority cast needs) until
/// 2093.
const STAMP_ORIGIN_UNIX_SECS: u64 = 1_735_689_600;

/// The clock's current seconds, counted from 1: a stamp whose seconds are 0
/// (the root's) carries no time.
fn clock_seconds() -> u64 {
    let (started, offset) = *STAMP_EPOCH;
    offset + started.elapsed().as_secs() + 1
}

/// [`clock_seconds`] as of the stamp clock's last tick; 0 until the clock
/// starts. A stamp reads this one atomic instead of the clock, which on the
/// request path cost about 8% of a match.
static STAMP_SECONDS: AtomicU64 = AtomicU64::new(0);
static STAMP_CLOCK: Once = Once::new();

/// How far [`STAMP_SECONDS`] may lag the clock.
const STAMP_TICK: Duration = Duration::from_millis(250);

/// Seconds since [`STAMP_ORIGIN_UNIX_SECS`], counted from 1, to within a
/// [`STAMP_TICK`]. Inline, as both trees call it on every stamp from other
/// modules: the hot path is the one load.
#[inline]
pub(super) fn stamp_seconds() -> u64 {
    match STAMP_SECONDS.load(Ordering::Relaxed) {
        0 => start_stamp_clock(),
        secs => secs,
    }
}

/// The first stamp's path: start the ticker thread.
#[cold]
#[inline(never)]
fn start_stamp_clock() -> u64 {
    STAMP_CLOCK.call_once(|| {
        STAMP_SECONDS.store(clock_seconds(), Ordering::Relaxed);
        let ticker = thread::Builder::new()
            .name("kv-index-stamp-clock".into())
            .spawn(|| loop {
                thread::sleep(STAMP_TICK);
                STAMP_SECONDS.store(clock_seconds(), Ordering::Relaxed);
            });
        if let Err(err) = ticker {
            warn!(%err, "stamp clock thread failed to start; access stamps keep their first second");
        }
    });
    STAMP_SECONDS.load(Ordering::Relaxed)
}

/// An access stamp: the seconds in the high 32 bits, `sequence`'s low 32
/// bits below. Stamps order by second, then by sequence within it, so
/// eviction compares them as it compared bare sequence values; the seconds
/// give a stamp's age.
#[inline]
pub(super) fn compose_stamp(sequence: u64) -> u64 {
    (stamp_seconds() << 32) | (sequence & 0xFFFF_FFFF)
}

/// Seconds since `stamp` was taken; `None` for a stamp that carries no time
/// (0, an attach no access has stamped, or a stamp from before stamps carried
/// seconds).
#[inline]
pub(super) fn stamp_age_secs(stamp: u64) -> Option<u64> {
    let secs = stamp >> 32;
    (secs != 0).then(|| stamp_seconds().saturating_sub(secs))
}
