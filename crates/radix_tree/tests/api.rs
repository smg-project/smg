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
