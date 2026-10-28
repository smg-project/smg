//! Common types shared between radix tree implementations.

use std::sync::Arc;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use rustc_hash::FxBuildHasher;

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
