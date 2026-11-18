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

/// How often an eviction asks its `keep_going`: every this many nodes walked
/// or entries taken off its heap, so a stop lands within a fraction of a pass
/// (a pass over a memory-sized tree walks for seconds). The value may change
/// between releases.
pub const CANCEL_CHECK_EVERY: usize = 4096;

/// Passes in a row a bounded eviction runs that evict nothing (every entry
/// they kept went stale under a concurrent insert) before it gives up until
/// the next eviction.
pub(crate) const MAX_IDLE_PASSES: usize = 2;

/// What a bounded eviction did (`evict_tenant_by_size_bounded` on either
/// tree): its passes, the (tenant, leaf) entries it evicted, its longest
/// pass, and whether `keep_going` stopped it before the budget was met.
///
/// An eviction ends in one of three ways: the tree is at or under budget;
/// `keep_going` stopped it (`cancelled`); or it stopped over budget without
/// `cancelled`, because its passes reached the limit its first walk set,
/// because passes in a row evicted nothing (each kept entry had gone stale
/// under concurrent inserts), or because a pass that left no leaf out ran
/// out of entries while concurrent inserts kept the tree over budget. On the
/// token tree that last one can also happen single-threaded: an evicted node
/// another tenant keeps does not take up its parent in that pass (see the
/// token tree's `evict_tenant_by_size_bounded`). The next eviction carries
/// on from there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BoundedEviction {
    pub passes: usize,
    pub evicted: usize,
    pub longest_pass: std::time::Duration,
    pub cancelled: bool,
}

/// What one pass of a bounded eviction did.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Pass {
    /// (tenant, leaf) entries it evicted (freeing chars or tokens).
    pub evicted: usize,
    /// Kept entries a concurrent change had left no longer a leaf.
    pub stale: usize,
    /// (tenant, node) pairs its walk saw: every entry the tree could evict,
    /// promoted parents included.
    pub pairs: usize,
    /// Whether its walk left leaves out, past `max_entries`.
    pub truncated: bool,
    /// Whether `keep_going` stopped it.
    pub cancelled: bool,
}

/// The pass loop of `evict_tenant_by_size_bounded`, the same on both trees.
/// `keep_going` is asked before every pass; another pass runs only after one
/// that left leaves out, left the tree over budget and made progress
/// (evicted, or found entries a concurrent insert had moved), and at most
/// [`MAX_IDLE_PASSES`] in a row evict nothing. The passes are also limited
/// to what the first walk saw: its (tenant, node) pairs in `max_entries`
/// steps, plus the final pass and one for concurrent change, so a caller
/// whose `keep_going` never stops it still ends under sustained inserts.
/// Each pass is a whole walk, so the limit bounds the passes, not the time:
/// `keep_going` remains the caller's time bound. `max_entries` is at least 1
/// (the trees' wrappers clamp it).
pub(crate) fn evict_in_passes(
    max_entries: usize,
    keep_going: &dyn Fn() -> bool,
    over_budget: impl Fn() -> bool,
    mut pass: impl FnMut() -> Pass,
) -> BoundedEviction {
    let mut done = BoundedEviction::default();
    let mut limit = usize::MAX;
    let mut idle = 0;
    loop {
        if !keep_going() {
            done.cancelled = true;
            break;
        }
        let started = std::time::Instant::now();
        let this = pass();
        done.passes += 1;
        done.evicted += this.evicted;
        done.longest_pass = done.longest_pass.max(started.elapsed());
        if this.cancelled {
            done.cancelled = true;
            break;
        }
        if done.passes == 1 {
            limit = this.pairs.div_ceil(max_entries).saturating_add(2);
        }
        idle = if this.evicted == 0 { idle + 1 } else { 0 };
        let again = this.truncated && over_budget() && (this.evicted > 0 || this.stale > 0);
        if !again || idle >= MAX_IDLE_PASSES || done.passes >= limit {
            break;
        }
    }
    done
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// [`evict_in_passes`] over scripted passes, the tree over budget
    /// throughout; past the script a pass leaves nothing out, which ends it.
    fn scripted(max_entries: usize, script: &[Pass]) -> BoundedEviction {
        let next = Cell::new(0);
        evict_in_passes(
            max_entries,
            &|| true,
            || true,
            || {
                next.set(next.get() + 1);
                script.get(next.get() - 1).copied().unwrap_or_default()
            },
        )
    }

    const EVICTING: Pass = Pass {
        evicted: 1,
        stale: 0,
        pairs: 100,
        truncated: true,
        cancelled: false,
    };
    const STALE_ONLY: Pass = Pass {
        evicted: 0,
        stale: 1,
        ..EVICTING
    };

    /// A pass whose kept entries all went stale (a concurrent insert gave
    /// them children) evicted nothing, yet the tree is over budget and other
    /// leaves were left out: the next pass walks again and finds them.
    #[test]
    fn a_stale_only_pass_walks_again() {
        let last = Pass {
            truncated: false,
            ..EVICTING
        };
        let done = scripted(10, &[STALE_ONLY, last]);
        assert_eq!((done.passes, done.evicted), (2, 1));
    }

    /// A pass that kept every leaf has nothing left to find: no other pass,
    /// though it evicted and the tree is over budget.
    #[test]
    fn a_pass_that_left_nothing_out_is_the_last() {
        let whole = Pass {
            truncated: false,
            ..EVICTING
        };
        assert_eq!(scripted(10, &[whole, EVICTING]).passes, 1);
    }

    /// Stale-only passes in a row end the eviction at the idle bound; an
    /// evicting pass between them starts the count again.
    #[test]
    fn idle_passes_in_a_row_are_bounded() {
        assert_eq!(scripted(10, &[STALE_ONLY; 5]).passes, MAX_IDLE_PASSES);
        let mixed = [STALE_ONLY, EVICTING, STALE_ONLY, STALE_ONLY, STALE_ONLY];
        assert_eq!(scripted(10, &mixed).passes, 1 + MAX_IDLE_PASSES + 1);
    }

    /// Passes that keep evicting under sustained inserts end at the limit
    /// the first walk set: its pairs in `max_entries` steps, plus two.
    #[test]
    fn the_passes_end_at_the_first_walks_limit() {
        let first = Pass {
            pairs: 5,
            ..EVICTING
        };
        let mut script = vec![first];
        script.extend([EVICTING; 20]);
        assert_eq!(scripted(2, &script).passes, 5_usize.div_ceil(2) + 2);
    }
}
