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
    let h = t.create_holder("h");
    let mut sequences: Vec<Vec<(u64, u64)>> = Vec::new();
    for round in 0..400 {
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
        sequences.push(seq);
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
        }
    }
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
