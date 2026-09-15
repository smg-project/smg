//! Targeted API contract tests: the behaviors the differential harness
//! can't reach because the model deliberately has no ids, config
//! bounds, or lifecycle. Every case runs against BOTH cores — the two
//! implement one contract, and the chaos fuzz does not cover live-name
//! idempotency, retirement accounting, stale-id semantics, exact
//! truncation order, or `ChainTooLong` atomicity.

use radix_tree::{
    Config, FlatTree, HolderId, Overlap, OverlapScratch, RadixTree, Stats, StoreError, StoreOutcome,
};

/// The public surface both cores share, so each case is written once.
trait Core {
    fn with_config(cfg: Config) -> Self;
    fn create_holder(&mut self, name: &str) -> HolderId;
    fn retire_holder(&mut self, id: HolderId);
    fn holder_name(&self, id: HolderId) -> Option<&str>;
    fn holder_blocks(&self, id: HolderId) -> u64;
    fn store(
        &mut self,
        id: HolderId,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
    ) -> Result<StoreOutcome, StoreError>;
    fn remove(&mut self, id: HolderId, keys: &[u64]) -> u32;
    fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64;
    fn enumerate(&self, id: HolderId) -> Vec<(u32, u64, u64)>;
    fn overlap(&self, chain: &[u64], scratch: &mut OverlapScratch, out: &mut Vec<Overlap>);
    fn stats(&self) -> Stats;
}

macro_rules! impl_core {
    ($t:ty) => {
        impl Core for $t {
            fn with_config(cfg: Config) -> Self {
                <$t>::new(cfg)
            }
            fn create_holder(&mut self, name: &str) -> HolderId {
                <$t>::create_holder(self, name)
            }
            fn retire_holder(&mut self, id: HolderId) {
                <$t>::retire_holder(self, id)
            }
            fn holder_name(&self, id: HolderId) -> Option<&str> {
                <$t>::holder_name(self, id)
            }
            fn holder_blocks(&self, id: HolderId) -> u64 {
                <$t>::holder_blocks(self, id)
            }
            fn store(
                &mut self,
                id: HolderId,
                parent: Option<u64>,
                blocks: &[(u64, u64)],
            ) -> Result<StoreOutcome, StoreError> {
                <$t>::store(self, id, parent, blocks)
            }
            fn remove(&mut self, id: HolderId, keys: &[u64]) -> u32 {
                <$t>::remove(self, id, keys)
            }
            fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64 {
                <$t>::truncate_tail(self, id, keep)
            }
            fn enumerate(&self, id: HolderId) -> Vec<(u32, u64, u64)> {
                <$t>::enumerate(self, id).collect()
            }
            fn overlap(&self, chain: &[u64], scratch: &mut OverlapScratch, out: &mut Vec<Overlap>) {
                <$t>::overlap(self, chain, scratch, out)
            }
            fn stats(&self) -> Stats {
                <$t>::stats(self)
            }
        }
    };
}

impl_core!(FlatTree);
impl_core!(RadixTree);

/// One `#[test]` per core for each generic case below.
macro_rules! both_cores {
    ($($case:ident),* $(,)?) => {
        $(
            mod $case {
                #[test]
                fn flat() {
                    super::$case::<super::FlatTree>();
                }
                #[test]
                fn chain() {
                    super::$case::<super::RadixTree>();
                }
            }
        )*
    };
}

both_cores!(
    stale_id_fails_loudly_never_aliases,
    retire_releases_everything_bounded_under_churn,
    truncate_tail_is_forest_wide_prefix_closed_and_deterministic,
    chain_too_long_is_terminal_and_atomic,
    create_holder_is_idempotent_per_live_name,
    lineage_exactness_content_coincidence_never_over_matches,
);

fn tree<C: Core>() -> C {
    C::with_config(Config::default())
}

fn stale_id_fails_loudly_never_aliases<C: Core>() {
    let mut t = tree::<C>();
    let a = t.create_holder("a");
    t.store(a, None, &[(1, 10), (2, 20)]).expect("store");
    t.retire_holder(a);
    // Slot recycled by a different holder.
    let b = t.create_holder("b");
    t.store(b, None, &[(3, 30)]).expect("store");
    assert_eq!(a.parts().0, b.parts().0, "test premise: slot reused");
    // Every operation through the stale id is a loud no-op.
    assert_eq!(t.store(a, None, &[(4, 40)]), Err(StoreError::UnknownHolder));
    assert_eq!(t.remove(a, &[3]), 0);
    assert_eq!(t.holder_blocks(a), 0);
    assert_eq!(t.holder_name(a), None);
    assert_eq!(t.enumerate(a).len(), 0);
    assert_eq!(t.truncate_tail(a, 0), 0);
    // The recycled holder was never touched.
    assert_eq!(t.holder_blocks(b), 1);
    assert_eq!(t.holder_name(b), Some("b"));
}

fn retire_releases_everything_bounded_under_churn<C: Core>() {
    let mut t = tree::<C>();
    let mut baseline = None;
    for cycle in 0..200u64 {
        let h = t.create_holder(&format!("pod-{cycle}"));
        let blocks: Vec<(u64, u64)> = (0..64)
            .map(|i| (cycle * 1000 + i + 1, cycle * 2000 + i + 1))
            .collect();
        t.store(h, None, &blocks).expect("store");
        t.retire_holder(h);
        let est = t.stats().bytes_estimate;
        match baseline {
            None => baseline = Some(est),
            Some(b) => assert!(
                est <= b,
                "state grew under churn: cycle {cycle}, {est} > {b}"
            ),
        }
        assert_eq!(t.stats().holders, 0);
        assert_eq!(t.stats().holder_blocks, 0);
        assert_eq!(t.stats().distinct_entries, 0);
    }
}

fn truncate_tail_is_forest_wide_prefix_closed_and_deterministic<C: Core>() {
    let mut t = tree::<C>();
    let h = t.create_holder("h");
    // Two chains: positions 0..4 and 0..2.
    t.store(h, None, &[(10, 1), (11, 2), (12, 3), (13, 4)])
        .expect("chain A");
    t.store(h, None, &[(20, 5), (21, 6)]).expect("chain B");
    assert_eq!(t.holder_blocks(h), 6);
    // keep=3: drops A@3, A@2, then the position-1 tie (A@1 key 11 vs
    // B@1 key 21) resolved by HIGHEST key first per the (pos, key)
    // order -> 21 goes first.
    let dropped = t.truncate_tail(h, 3);
    assert_eq!(dropped, 3);
    assert_eq!(t.enumerate(h), vec![(0, 10, 1), (0, 20, 5), (1, 11, 2)]);
    // Prefix-closed: every remaining position p>0 has its position
    // p-1 present on the same chain (A kept 0,1; B kept 0).
    // Queries still answer the kept prefixes exactly.
    let mut out = Vec::new();
    let mut sc = OverlapScratch::default();
    t.overlap(&[1, 2, 3], &mut sc, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].depth, 2);
    t.overlap(&[5, 6], &mut sc, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].depth, 1);
}

fn chain_too_long_is_terminal_and_atomic<C: Core>() {
    let mut t = C::with_config(Config { max_chain_len: 4 });
    let h = t.create_holder("h");
    t.store(h, None, &[(1, 1), (2, 2), (3, 3)]).expect("fits");
    // Extending 3 + 2 > 4: rejected whole.
    assert_eq!(
        t.store(h, Some(3), &[(4, 4), (5, 5)]),
        Err(StoreError::ChainTooLong)
    );
    assert_eq!(t.holder_blocks(h), 3);
    // Exactly at the bound is fine.
    t.store(h, Some(3), &[(4, 4)]).expect("at bound");
    assert_eq!(t.holder_blocks(h), 4);
}

fn create_holder_is_idempotent_per_live_name<C: Core>() {
    let mut t = tree::<C>();
    let a1 = t.create_holder("a");
    let a2 = t.create_holder("a");
    assert_eq!(a1, a2);
    t.retire_holder(a1);
    let a3 = t.create_holder("a");
    assert_ne!(a1, a3, "new generation after retire");
    assert_eq!(t.holder_blocks(a3), 0);
}

fn lineage_exactness_content_coincidence_never_over_matches<C: Core>() {
    let mut t = tree::<C>();
    let h = t.create_holder("h");
    // Chain X: contents [7, 8]; chain Y: contents [9, 8] — content 8
    // appears at position 1 under BOTH lineages.
    t.store(h, None, &[(1, 7), (2, 8)]).expect("X");
    t.store(h, None, &[(3, 9), (4, 8)]).expect("Y");
    let mut out = Vec::new();
    let mut sc = OverlapScratch::default();
    // Query [7, 8]: depth 2 via X only.
    t.overlap(&[7, 8], &mut sc, &mut out);
    assert_eq!((out[0].holder, out[0].depth), (h, 2));
    // Query [9, 8]: depth 2 via Y only.
    t.overlap(&[9, 8], &mut sc, &mut out);
    assert_eq!((out[0].holder, out[0].depth), (h, 2));
    // Query [5, 8]: content 8 exists at position 1 (twice!) but no
    // chain has lineage [5] -> depth 0, no answer at all.
    t.overlap(&[5, 8], &mut sc, &mut out);
    assert!(out.is_empty(), "lineage-blind positional match leaked");
}

// ---- evict_oldest: chain-core only (recency needs per-chain state the
// flat core never had; the flat core is the differential reference and
// stays on truncate_tail) ----

fn chain_tree() -> RadixTree {
    RadixTree::new(Config::default())
}

/// Keys of the surviving blocks, as a set.
fn surviving_keys(t: &RadixTree, h: HolderId) -> std::collections::BTreeSet<u64> {
    t.enumerate(h).map(|(_, k, _)| k).collect()
}

#[test]
fn evict_oldest_drops_least_recent_chains_whole_and_frees_their_slots() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    // Four disjoint root chains stored in order A, B, C, D.
    for (i, base) in [(0u64, 100u64), (1, 200), (2, 300), (3, 400)] {
        t.store(
            h,
            None,
            &[
                (base, i * 10 + 1),
                (base + 1, i * 10 + 2),
                (base + 2, i * 10 + 3),
            ],
        )
        .expect("store");
    }
    assert_eq!(t.holder_blocks(h), 12);
    assert_eq!(t.live_chain_count(), 4);
    let dropped = t.evict_oldest(h, 6);
    assert_eq!(dropped, 6);
    assert_eq!(t.holder_blocks(h), 6);
    assert_eq!(
        surviving_keys(&t, h),
        [300, 301, 302, 400, 401, 402].into_iter().collect(),
        "the two oldest chains go, the two newest stay whole"
    );
    // The holder was the only member: both evicted chains are freed,
    // not left as empty heads.
    assert_eq!(t.live_chain_count(), 2);
    t.audit().expect("audit");
    let mut out = Vec::new();
    let mut sc = OverlapScratch::default();
    t.overlap(&[31, 32, 33], &mut sc, &mut out);
    assert_eq!((out[0].holder, out[0].depth), (h, 3));
    t.overlap(&[1, 2, 3], &mut sc, &mut out);
    assert!(out.is_empty(), "evicted chain still answers");
}

#[test]
fn evict_oldest_keeps_a_parent_as_young_as_its_youngest_child() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    // P (tick 1), then children A (tick 2) and B (tick 3) forking off
    // P's tip, then an unrelated root R (tick 4).
    t.store(h, None, &[(1, 1), (2, 2)]).expect("P");
    t.store(h, Some(2), &[(3, 3), (4, 4)]).expect("A");
    t.store(h, Some(2), &[(5, 5), (6, 6)]).expect("B");
    t.store(h, None, &[(7, 7), (8, 8)]).expect("R");
    assert_eq!(t.holder_blocks(h), 8);
    // keep=4: A is the least recent subtree. P's own tick is the
    // oldest of all, but it inherits B's recency, so B (deeper, same
    // effective recency) goes before P and P survives with R.
    assert_eq!(t.evict_oldest(h, 4), 4);
    assert_eq!(surviving_keys(&t, h), [1, 2, 7, 8].into_iter().collect());
    t.audit().expect("audit");
    let mut out = Vec::new();
    let mut sc = OverlapScratch::default();
    t.overlap(&[1, 2, 5, 6], &mut sc, &mut out);
    assert_eq!(
        (out[0].holder, out[0].depth),
        (h, 2),
        "P still answers, B is gone"
    );
    // keep=3 from here: P (2 blocks) is the final victim and is
    // trimmed deepest-first, never dropped past `keep`.
    assert_eq!(t.evict_oldest(h, 3), 1);
    assert_eq!(surviving_keys(&t, h), [1, 7, 8].into_iter().collect());
    t.audit().expect("audit");
}

#[test]
fn evict_oldest_treats_a_duplicate_store_as_a_touch() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    t.store(h, None, &[(1, 1), (2, 2)]).expect("A");
    t.store(h, None, &[(3, 3), (4, 4)]).expect("B");
    // Republishing A verbatim applies nothing but refreshes it.
    let again = t.store(h, None, &[(1, 1), (2, 2)]).expect("A again");
    assert_eq!(again.applied, 0);
    assert_eq!(t.evict_oldest(h, 2), 2);
    assert_eq!(
        surviving_keys(&t, h),
        [1, 2].into_iter().collect(),
        "B was least recent"
    );
    t.audit().expect("audit");
}

#[test]
fn evict_oldest_treats_a_shared_lock_duplicate_walk_as_a_touch() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    t.store(h, None, &[(1, 1), (2, 2), (3, 3)]).expect("A");
    t.store(h, None, &[(4, 4), (5, 5)]).expect("B");
    // The plain predicate is pure: A stays the least recent.
    assert_eq!(t.dup_prefix(h, None, &[(1, 1), (2, 2)]), (2, true));
    // The touching walk refreshes A (a covered prefix is enough).
    assert_eq!(t.dup_prefix_touch(h, None, &[(1, 1), (2, 2)]), (2, true));
    assert_eq!(t.evict_oldest(h, 3), 2);
    assert_eq!(
        surviving_keys(&t, h),
        [1, 2, 3].into_iter().collect(),
        "B was least recent"
    );
    // `covered_touch` is the same publish signal for the fully
    // resident predicate: B republished through it outlives A.
    t.store(h, None, &[(4, 4), (5, 5)]).expect("B back");
    assert!(
        t.covered(h, None, &[(1, 1), (2, 2), (3, 3)]),
        "pure predicate"
    );
    assert!(t.covered_touch(h, None, &[(4, 4), (5, 5)]));
    assert_eq!(t.evict_oldest(h, 2), 3);
    assert_eq!(surviving_keys(&t, h), [4, 5].into_iter().collect());
    t.store(h, None, &[(1, 1), (2, 2), (3, 3)]).expect("A back");
    // An uncovered walk touches only what it covered: a walk that
    // fails on B's missing tail still counts for B's resident prefix.
    t.store(h, None, &[(4, 4), (5, 5)]).expect("B again");
    t.store(h, None, &[(6, 6)]).expect("C");
    assert_eq!(t.dup_prefix_touch(h, None, &[(1, 1), (9, 9)]), (1, false));
    assert_eq!(t.evict_oldest(h, 4), 2, "B goes, A (touched) and C stay");
    assert_eq!(surviving_keys(&t, h), [1, 2, 3, 6].into_iter().collect());
    t.audit().expect("audit");
}

#[test]
fn evict_oldest_ignores_stale_ids_and_holders_within_keep() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    t.store(h, None, &[(1, 1), (2, 2)]).expect("A");
    assert_eq!(t.evict_oldest(h, 2), 0);
    assert_eq!(t.evict_oldest(h, 5), 0);
    t.retire_holder(h);
    assert_eq!(t.evict_oldest(h, 0), 0, "stale id is a no-op");
    let other = t.create_holder("o");
    assert_eq!(t.evict_oldest(other, 0), 0, "empty holder");
}

#[test]
fn evict_oldest_leaves_other_holders_and_shared_chains_intact() {
    let mut t = chain_tree();
    let a = t.create_holder("a");
    let b = t.create_holder("b");
    // Both hold the same chain; `a` also holds an older private one.
    t.store(a, None, &[(10, 10), (11, 11)]).expect("a private");
    t.store(a, None, &[(1, 1), (2, 2), (3, 3)])
        .expect("a shared");
    t.store(b, None, &[(1, 1), (2, 2), (3, 3)])
        .expect("b shared");
    assert_eq!(t.live_chain_count(), 2);
    assert_eq!(t.evict_oldest(a, 3), 2);
    assert_eq!(surviving_keys(&t, a), [1, 2, 3].into_iter().collect());
    assert_eq!(t.holder_blocks(b), 3, "other holder untouched");
    // Evict all of `a`'s shared coverage: the chain survives for `b`.
    assert_eq!(t.evict_oldest(a, 0), 3);
    assert_eq!(t.live_chain_count(), 1);
    assert_eq!(t.holder_blocks(b), 3);
    let mut out = Vec::new();
    let mut sc = OverlapScratch::default();
    t.overlap(&[1, 2, 3], &mut sc, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!((out[0].holder, out[0].depth), (b, 3));
    t.audit().expect("audit");
}

/// Randomized: stores with shared prefixes (same prefix => same keys),
/// evictions to arbitrary `keep`. Survivors must be exactly `keep`,
/// prefix-closed along every stored sequence, and the tree must audit
/// clean after every step.
#[test]
fn evict_oldest_is_exact_and_prefix_closed_under_churn() {
    fn key_of(prefix: &[u64]) -> u64 {
        prefix.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
            (h ^ c).wrapping_mul(0x0100_0000_01b3)
        }) | 1
    }
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut t = chain_tree();
    let mut h = t.create_holder("h");
    // A second holder sharing the same prefixes, never evicted: its
    // coverage must be untouched by everything below.
    let bystander = t.create_holder("bystander");
    let mut bystander_keys = std::collections::BTreeSet::new();
    let mut sequences: Vec<Vec<(u64, u64)>> = Vec::new();
    for round in 0..1200 {
        // A sequence of 2..=9 contents from an alphabet of 4 per
        // position, so prefixes collide often and fork everywhere.
        let len = 2 + (next() % 8) as usize;
        let mut contents = Vec::with_capacity(len);
        let mut seq = Vec::with_capacity(len);
        for pos in 0..len {
            contents.push(1 + (pos as u64) * 16 + next() % 4);
            seq.push((key_of(&contents), contents[pos]));
        }
        t.store(h, None, &seq).expect("store");
        if round % 5 == 0 {
            t.store(bystander, None, &seq).expect("bystander store");
            bystander_keys.extend(seq.iter().map(|(k, _)| *k));
        }
        sequences.push(seq);
        // The event-feed path: drop the deepest surviving key of a
        // random stored sequence (a leaf eviction), which empties the
        // holder's coverage on a chain without going through
        // `evict_oldest` and can leave a chain's coverage at a fork.
        if round % 3 == 1 {
            let alive = surviving_keys(&t, h);
            let s = &sequences[(next() as usize) % sequences.len()];
            if let Some(at) = s.iter().rposition(|(k, _)| alive.contains(k)) {
                let k = s[at].0;
                // A leaf of the holder's whole coverage, not just of
                // this sequence: no other sequence that shares the
                // prefix through `k` (same key at the same position)
                // still holds anything deeper, or the remove itself
                // would open the gap the check below looks for.
                let is_leaf = sequences.iter().all(|o| {
                    o.get(at).is_none_or(|(ok, _)| *ok != k)
                        || o[at + 1..].iter().all(|(dk, _)| !alive.contains(dk))
                });
                if is_leaf {
                    assert_eq!(t.remove(h, &[k]), 1);
                }
            }
        }
        // Epoch bump and slot recycling: a fresh holder in a possibly
        // recycled slot must start with no recency state at all.
        if round % 250 == 249 {
            if round % 500 == 249 {
                t.clear(h);
            } else {
                t.retire_holder(h);
                h = t.create_holder("h");
            }
            assert_eq!(t.holder_blocks(h), 0);
            assert_eq!(t.evict_oldest(h, 0), 0);
            sequences.clear();
            t.audit()
                .unwrap_or_else(|e| panic!("audit after reset: {e}"));
        }
        if round % 7 == 6 {
            let total = t.holder_blocks(h);
            let keep = next() % (total + 1);
            let dropped = t.evict_oldest(h, keep);
            assert_eq!(dropped, total - keep);
            assert_eq!(t.holder_blocks(h), keep);
            t.audit()
                .unwrap_or_else(|e| panic!("audit after evict: {e}"));
            let alive = surviving_keys(&t, h);
            for s in &sequences {
                let mut seen_gap = false;
                for (k, _) in s {
                    let present = alive.contains(k);
                    assert!(
                        !(present && seen_gap),
                        "round {round}: key {k:#x} survives below an evicted ancestor"
                    );
                    seen_gap |= !present;
                }
            }
            assert_eq!(
                surviving_keys(&t, bystander),
                bystander_keys,
                "round {round}: the bystander lost or gained coverage"
            );
        }
    }
}

/// A lineage forked at every position: a 6,000-block chain whose
/// every prefix also has a one-block sibling, so the holder covers
/// ~6,000 chains at depths up to 6,000. The ancestor walk in
/// `evict_oldest` must stay linear here (the un-memoized version
/// visited ~18M parent links; a 65k-position lineage would visit 2
/// billion) and the result must be exactly the recency order: the
/// siblings were stored oldest-first, the spine last.
#[test]
fn evict_oldest_is_linear_on_a_lineage_forked_at_every_position() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    let n = 6_000u64;
    // Spine keys 1..=n, contents 1..=n, stored one block at a time so
    // every position is a chain tip when its sibling is stored.
    t.store(h, None, &[(1, 1)]).expect("root");
    for i in 2..=n {
        t.store(h, Some(i - 1), &[(i, i)]).expect("extend");
        // Sibling fork at position i-1 (a different content), stored
        // BEFORE the spine grows past it: an older, deeper-forked chain.
        t.store(h, Some(i - 1), &[(1_000_000 + i, 1_000_000 + i)])
            .expect("fork");
    }
    // Refresh the spine as one publish: every spine chain is now the
    // youngest; the siblings are the oldest in ascending i.
    let spine: Vec<(u64, u64)> = (1..=n).map(|i| (i, i)).collect();
    assert_eq!(t.dup_prefix_touch(h, None, &spine), (n as u32, true));
    let total = t.holder_blocks(h);
    assert_eq!(total, 2 * n - 1);
    let started = std::time::Instant::now();
    assert_eq!(t.evict_oldest(h, n), n - 1);
    let took = started.elapsed();
    assert_eq!(
        surviving_keys(&t, h),
        (1..=n).collect(),
        "spine kept whole, every sibling gone"
    );
    t.audit().expect("audit");
    assert!(
        took.as_secs() < 5,
        "evict_oldest took {took:?} on a {n}-deep forked lineage"
    );
}

/// Concurrent touching walks race on the same chains through `&self`:
/// the recency hint must keep the latest tick (max, not last-writer),
/// the tree must audit clean, and the eviction that follows must be
/// exact.
#[test]
fn evict_oldest_survives_concurrent_touches() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    let a: Vec<(u64, u64)> = (1..=8).map(|i| (i, i)).collect();
    let b: Vec<(u64, u64)> = (101..=108).map(|i| (i, i)).collect();
    t.store(h, None, &b).expect("B (older)");
    t.store(h, None, &a).expect("A");
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..2_000 {
                    assert!(t.covered_touch(h, None, &a));
                    assert_eq!(t.dup_prefix_touch(h, None, &a[..4]), (4, true));
                }
            });
        }
    });
    // A single touch of B after the race is the newest tick of all:
    // B must now outlive A regardless of the order the racing A
    // touches landed in.
    assert!(t.covered_touch(h, None, &b));
    t.audit().expect("audit");
    assert_eq!(t.evict_oldest(h, 8), 8);
    assert_eq!(surviving_keys(&t, h), (101..=108).collect());
}

/// The memory regression: a holder fed fresh chains past capacity and
/// cut back on a 2x hysteresis must not accumulate chain slots.
/// Depth-ordered truncation left every chain head alive forever;
/// whole-chain eviction keeps live chains proportional to the kept
/// blocks.
#[test]
fn evict_oldest_keeps_chain_slots_bounded_under_placement_churn() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    let capacity = 1_000u64;
    let chain_len = 10u64;
    let mut peak_chains = 0usize;
    for i in 0..10_000u64 {
        let base = (i + 1) * 1_000;
        let blocks: Vec<(u64, u64)> = (0..chain_len).map(|p| (base + p, base + p)).collect();
        t.store(h, None, &blocks).expect("store");
        if t.holder_blocks(h) > capacity * 2 {
            t.evict_oldest(h, capacity);
        }
        peak_chains = peak_chains.max(t.live_chain_count());
    }
    let kept_chains = (capacity * 2 / chain_len) as usize;
    assert!(
        peak_chains <= kept_chains + 1,
        "live chains peaked at {peak_chains}, bound {}",
        kept_chains + 1
    );
    t.evict_oldest(h, capacity);
    assert_eq!(t.live_chain_count(), (capacity / chain_len) as usize);
    t.audit().expect("audit");
}

/// Cost of one capacity cut on a placement-shaped forest (ignored; run
/// with `--release -- --ignored --nocapture`): 120 holders sharing a
/// 16-block system prefix, each holding private session chains of
/// 30..100 blocks up to 2x a 4,688-block capacity, plus a hot set of
/// 256 chains shared by 8 holders each. Times `evict_oldest` and
/// `truncate_tail` on identically built trees.
#[test]
#[ignore]
fn bench_capacity_cut_on_a_placement_shaped_forest() {
    fn build(shared_hot: bool) -> (RadixTree, Vec<HolderId>) {
        let mut t = chain_tree();
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let holders: Vec<HolderId> = (0..120)
            .map(|i| t.create_holder(&format!("w{i}")))
            .collect();
        let system: Vec<(u64, u64)> = (1..=16).map(|i| (i, i)).collect();
        let hot: Vec<Vec<(u64, u64)>> = (0..256)
            .map(|c| {
                (0..64)
                    .map(|p| (10_000_000 + c * 1000 + p, 10_000_000 + c * 1000 + p))
                    .collect()
            })
            .collect();
        let mut key = 1_000_000u64;
        for (hi, &h) in holders.iter().enumerate() {
            t.store(h, None, &system).expect("system prefix");
            let mut n = 16u64;
            while n < 9_376 {
                if shared_hot && next() % 10 < 3 {
                    let c = &hot[(next() as usize) % hot.len()];
                    let out = t.store(h, Some(16), c).expect("hot chain");
                    n += u64::from(out.applied);
                } else {
                    let len = 30 + next() % 71;
                    let chain: Vec<(u64, u64)> = (0..len)
                        .map(|p| (key + p, key + p + (hi as u64) * 7))
                        .collect();
                    key += len;
                    t.store(h, Some(16), &chain).expect("session chain");
                    n += len;
                }
            }
        }
        (t, holders)
    }
    for shared_hot in [false, true] {
        for method in ["evict_oldest", "truncate_tail"] {
            let (mut t, holders) = build(shared_hot);
            let before = t.stats();
            let mut worst = std::time::Duration::ZERO;
            let started = std::time::Instant::now();
            let mut dropped = 0u64;
            for &h in &holders {
                let one = std::time::Instant::now();
                dropped += if method == "evict_oldest" {
                    t.evict_oldest(h, 4_688)
                } else {
                    t.truncate_tail(h, 4_688)
                };
                worst = worst.max(one.elapsed());
            }
            let total = started.elapsed();
            let after = t.stats();
            eprintln!(
                "{method:13} shared_hot={shared_hot}: 120 cuts in {total:?} (mean {:?}, worst {worst:?}); dropped {dropped}; blocks {} -> {}; live chains {} -> {}",
                total / 120,
                before.holder_blocks,
                after.holder_blocks,
                before.distinct_entries,
                after.distinct_entries,
            );
            t.audit().expect("audit");
        }
    }
}

// ---- coverage: per-holder runs along the query path (chain core) ----

fn runs_of(t: &RadixTree, h: HolderId, query: &[u64]) -> Vec<(u32, u32)> {
    let mut sc = OverlapScratch::default();
    let mut out = Vec::new();
    t.coverage(query, &mut sc, &mut out);
    out.iter()
        .filter(|r| r.holder == h)
        .map(|r| (r.start, r.end))
        .collect()
}

#[test]
fn coverage_reports_holes_and_mid_path_runs_that_overlap_cannot() {
    let mut t = chain_tree();
    let full = t.create_holder("w#full");
    let window = t.create_holder("w#window");
    let mamba = t.create_holder("w#mamba");
    let chain: Vec<(u64, u64)> = (1..=12).map(|i| (i, 100 + i)).collect();
    let query: Vec<u64> = chain.iter().map(|&(_, c)| c).collect();
    // Full attention holds everything; the window lane evicted its
    // first four blocks; the mamba lane keeps checkpoints at 4 and 8.
    t.store(full, None, &chain).expect("full");
    t.store(window, None, &chain).expect("window");
    assert_eq!(t.remove(window, &[1, 2, 3, 4]), 4);
    t.store(mamba, None, &chain).expect("mamba");
    assert_eq!(t.remove(mamba, &[1, 2, 3, 5, 6, 7, 9, 10, 11, 12]), 10);

    assert_eq!(runs_of(&t, full, &query), vec![(0, 12)]);
    assert_eq!(runs_of(&t, window, &query), vec![(4, 12)]);
    assert_eq!(runs_of(&t, mamba, &query), vec![(3, 4), (7, 8)]);
    // The depth answer sees only the full lane.
    let mut sc = OverlapScratch::default();
    let mut out = Vec::new();
    t.overlap(&query, &mut sc, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!((out[0].holder, out[0].depth), (full, 12));
    // Runs stop where the query diverges from the stored chain.
    let mut diverged = query.clone();
    diverged[6] = 999;
    assert_eq!(runs_of(&t, full, &diverged), vec![(0, 6)]);
    assert_eq!(runs_of(&t, window, &diverged), vec![(4, 6)]);
    assert_eq!(runs_of(&t, mamba, &diverged), vec![(3, 4)]);
    // A query that matches nothing yields no runs; an empty query too.
    assert!(runs_of(&t, full, &[42, 43]).is_empty());
    assert!(runs_of(&t, full, &[]).is_empty());
    t.audit().expect("audit");
}

#[test]
fn coverage_is_sorted_per_holder_and_reuses_its_scratch_across_queries() {
    let mut t = chain_tree();
    let a = t.create_holder("a");
    let b = t.create_holder("b");
    let long: Vec<(u64, u64)> = (1..=8).map(|i| (i, 100 + i)).collect();
    let short: Vec<(u64, u64)> = (11..=13).map(|i| (i, 200 + i)).collect();
    t.store(a, None, &long).expect("a long");
    t.store(b, None, &long).expect("b long");
    t.store(b, None, &short).expect("b short");
    t.remove(b, &[3, 4]);
    let mut sc = OverlapScratch::default();
    let mut out = Vec::new();
    let q_long: Vec<u64> = long.iter().map(|&(_, c)| c).collect();
    let q_short: Vec<u64> = short.iter().map(|&(_, c)| c).collect();
    for _ in 0..3 {
        t.coverage(&q_long, &mut sc, &mut out);
        let got: Vec<(HolderId, u32, u32)> =
            out.iter().map(|r| (r.holder, r.start, r.end)).collect();
        assert_eq!(got, vec![(a, 0, 8), (b, 0, 2), (b, 4, 8)]);
        assert!(out
            .iter()
            .all(|r| r.total_blocks == if r.holder == a { 8 } else { 9 }));
        t.coverage(&q_short, &mut sc, &mut out);
        let got: Vec<(HolderId, u32, u32)> =
            out.iter().map(|r| (r.holder, r.start, r.end)).collect();
        assert_eq!(
            got,
            vec![(b, 0, 3)],
            "stale runs must not leak between queries"
        );
    }
    t.retire_holder(a);
    t.coverage(&q_long, &mut sc, &mut out);
    assert!(
        out.iter().all(|r| r.holder == b),
        "retired holders are not reported"
    );
}

/// Randomized: colliding prefixes across holders, random leaf removes,
/// then every stored sequence is queried and each holder's runs must
/// equal the positions where that holder still holds the query's key.
#[test]
fn coverage_matches_the_per_position_key_map_under_churn() {
    fn key_of(prefix: &[u64]) -> u64 {
        prefix.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
            (h ^ c).wrapping_mul(0x0100_0000_01b3)
        }) | 1
    }
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut t = chain_tree();
    let holders: Vec<HolderId> = (0..6).map(|i| t.create_holder(&format!("h{i}"))).collect();
    let mut sequences: Vec<Vec<(u64, u64)>> = Vec::new();
    for round in 0..600 {
        let len = 2 + (next() % 10) as usize;
        let mut contents = Vec::with_capacity(len);
        let mut seq = Vec::with_capacity(len);
        for pos in 0..len {
            contents.push(1 + (pos as u64) * 8 + next() % 3);
            seq.push((key_of(&contents), contents[pos]));
        }
        let h = holders[(next() as usize) % holders.len()];
        t.store(h, None, &seq).expect("store");
        sequences.push(seq);
        if round % 4 == 3 {
            // Remove a random key from a random holder: holes anywhere.
            let s = &sequences[(next() as usize) % sequences.len()];
            let (k, _) = s[(next() as usize) % s.len()];
            let h = holders[(next() as usize) % holders.len()];
            t.remove(h, &[k]);
        }
        if round % 25 == 24 {
            t.audit().unwrap_or_else(|e| panic!("audit: {e}"));
            let mut sc = OverlapScratch::default();
            let mut out = Vec::new();
            for s in &sequences {
                let query: Vec<u64> = s.iter().map(|&(_, c)| c).collect();
                t.coverage(&query, &mut sc, &mut out);
                for &h in &holders {
                    // Expected runs from the key map: position p is held
                    // iff the holder has the key of query[..=p] at p.
                    let mut expected = Vec::new();
                    for (p, (k, _)) in s.iter().enumerate() {
                        if t.position_of(h, *k) == Some(p as u32) {
                            match expected.last_mut() {
                                Some((_, end)) if *end == p as u32 => *end += 1,
                                _ => expected.push((p as u32, p as u32 + 1)),
                            }
                        }
                    }
                    let got: Vec<(u32, u32)> = out
                        .iter()
                        .filter(|r| r.holder == h)
                        .map(|r| (r.start, r.end))
                        .collect();
                    assert_eq!(got, expected, "round {round}: holder {h:?} on {query:?}");
                }
            }
        }
    }
}

#[test]
fn path_contents_walks_forks_back_to_the_root_for_held_and_holed_coverage() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    // Root [c1..c4], fork at position 1 with contents [c5, c6], and a
    // hole: the holder drops keys 1 and 5 afterwards.
    t.store(h, None, &[(1, 101), (2, 102), (3, 103), (4, 104)])
        .expect("root");
    t.store(h, Some(2), &[(5, 105), (6, 106)]).expect("fork");
    assert_eq!(t.path_contents(h, 4), Some(vec![101, 102, 103, 104]));
    assert_eq!(t.path_contents(h, 6), Some(vec![101, 102, 105, 106]));
    t.remove(h, &[1, 5]);
    // Holes do not hide the path: the chain keeps its contents.
    assert_eq!(t.path_contents(h, 6), Some(vec![101, 102, 105, 106]));
    assert_eq!(t.path_contents(h, 3), Some(vec![101, 102, 103]));
    assert_eq!(t.path_contents(h, 5), None, "a removed key has no position");
    t.retire_holder(h);
    assert_eq!(t.path_contents(h, 6), None, "stale id");
}

#[test]
fn runs_split_by_lineage_and_carry_the_uncovered_head_of_the_path() {
    let mut t = chain_tree();
    let h = t.create_holder("h");
    let other = t.create_holder("o");
    t.store(h, None, &[(1, 101), (2, 102), (3, 103), (4, 104), (5, 105)])
        .expect("root");
    t.store(h, Some(2), &[(6, 106), (7, 107), (8, 108)])
        .expect("fork at 1");
    // Another holder shares the root but that must not change h's runs.
    t.store(other, None, &[(1, 101), (2, 102), (3, 103)])
        .expect("other");
    // Holes: drop position 0 on the root path and position 3 on the fork.
    t.remove(h, &[1, 7]);
    let runs = t.runs(h);
    let view: Vec<(u32, u32, Vec<u64>, Vec<u64>)> = runs
        .iter()
        .map(|r| (r.start, r.end, r.keys.clone(), r.path.clone()))
        .collect();
    assert_eq!(view.len(), runs.len());
    // Every run's keys sit at its positions with the path's contents.
    for r in &runs {
        assert_eq!(r.path.len(), r.end as usize);
        for (i, k) in r.keys.iter().enumerate() {
            assert_eq!(t.position_of(h, *k), Some(r.start + i as u32));
        }
    }
    // The union of held positions is exactly the key map: 6 keys.
    let held: usize = runs.iter().map(|r| r.keys.len()).sum();
    assert_eq!(held, 6);
    // Both lineages are present: one path ends in 105, one in 108.
    let ends: std::collections::BTreeSet<u64> =
        runs.iter().map(|r| *r.path.last().unwrap()).collect();
    assert!(ends.contains(&105) && ends.contains(&108), "{runs:?}");
    // A run that starts past 0 carries its uncovered head.
    assert!(runs.iter().any(|r| r.start > 0 && r.path[0] == 101));
    assert!(t.runs(other).iter().all(|r| r.start == 0));
    t.retire_holder(h);
    assert!(t.runs(h).is_empty());
}
