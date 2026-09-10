//! The model-referee differential harness, R0 stage:
//! model vs oracle. When the R1 core exists it joins as the third
//! side with the HARD assertion `RadixTree == model`; until then this
//! file proves the model, the oracle adapter, and the generator agree
//! on the contract's terms:
//!
//! - the oracle never under-matches the model (an under-match has no
//!   quirk explanation and fails the run — §10.1's "unclassifiable");
//! - every oracle over-match position is tagged into three classes:
//!   gap-bridged (right block present, an earlier position missing),
//!   cross-lineage (content coincidence under another chain), or
//!   absent (pure skip overshoot) — §6's quirk surface, measured;
//! - both sides accept/reject every store identically;
//! - the model itself converges under §7-scoped reordering.

mod common;

use std::collections::BTreeMap;

use common::{
    model::{Model, StoreResult},
    oracle::Oracle,
    workload, Op,
};
use radix_tree::{Config, FlatTree, HolderId, OverlapScratch, RadixTree, StoreError};

/// The implementations under test (flat R1 and chain-native R3),
/// driven by the harness Op protocol. Every observable is asserted
/// for BOTH against the model.
enum Core {
    Flat(FlatTree),
    Chain(RadixTree),
}

struct Subject {
    core: Core,
    ids: Vec<HolderId>,
    scratch: Vec<radix_tree::Overlap>,
    qscratch: OverlapScratch,
}

impl Subject {
    fn new_flat(holders: usize) -> Self {
        let mut tree = FlatTree::new(Config::default());
        let ids = (0..holders)
            .map(|h| tree.create_holder(&format!("holder-{h}")))
            .collect();
        Self {
            core: Core::Flat(tree),
            ids,
            scratch: Vec::new(),
            qscratch: OverlapScratch::default(),
        }
    }

    fn new_chain(holders: usize) -> Self {
        let mut tree = RadixTree::new(Config::default());
        let ids = (0..holders)
            .map(|h| tree.create_holder(&format!("holder-{h}")))
            .collect();
        Self {
            core: Core::Chain(tree),
            ids,
            scratch: Vec::new(),
            qscratch: OverlapScratch::default(),
        }
    }

    /// Returns false when the store was rejected (must match model).
    fn apply(&mut self, op: &Op) -> bool {
        match op {
            Op::Store {
                holder,
                parent,
                blocks,
            } => {
                let r = match &mut self.core {
                    Core::Flat(t) => t.store(self.ids[*holder], *parent, blocks),
                    Core::Chain(t) => t.store(self.ids[*holder], *parent, blocks),
                };
                match r {
                    Ok(_) => true,
                    Err(StoreError::ParentNotFound) => false,
                    Err(e) => panic!("unexpected store error {e:?}"),
                }
            }
            Op::Remove { holder, keys } => {
                match &mut self.core {
                    Core::Flat(t) => t.remove(self.ids[*holder], keys),
                    Core::Chain(t) => t.remove(self.ids[*holder], keys),
                };
                true
            }
            Op::Clear { holder } => {
                match &mut self.core {
                    Core::Flat(t) => t.clear(self.ids[*holder]),
                    Core::Chain(t) => t.clear(self.ids[*holder]),
                }
                true
            }
        }
    }

    fn overlap(&mut self, query: &[u64]) -> BTreeMap<usize, u32> {
        let scratch = &mut self.scratch;
        match &mut self.core {
            Core::Flat(t) => t.overlap(query, &mut self.qscratch, scratch),
            Core::Chain(t) => t.overlap(query, &mut self.qscratch, scratch),
        }
        // Shape check on the RAW Vec before the BTreeMap collapse: every
        // holder must appear at most once. The map projection would hide a
        // merge-loop regression that emitted a holder twice (a downstream
        // consumer iterating the Vec would then double-count).
        let mut out = BTreeMap::new();
        for o in scratch.iter() {
            assert!(
                out.insert(o.holder.parts().0 as usize, o.depth).is_none(),
                "overlap emitted holder {} twice in the raw Vec",
                o.holder.parts().0
            );
        }
        out
    }

    fn holder_blocks(&self, h: usize) -> u64 {
        match &self.core {
            Core::Flat(t) => t.holder_blocks(self.ids[h]),
            Core::Chain(t) => t.holder_blocks(self.ids[h]),
        }
    }

    fn enumerate(&self, h: usize) -> Vec<(u32, u64, u64)> {
        match &self.core {
            Core::Flat(t) => t.enumerate(self.ids[h]).collect(),
            Core::Chain(t) => t.enumerate(self.ids[h]).collect(),
        }
    }

    fn distinct_entries(&self) -> u64 {
        match &self.core {
            Core::Flat(t) => t.stats().distinct_entries,
            Core::Chain(t) => t.stats().distinct_entries,
        }
    }

    fn audit(&self) -> Result<(), String> {
        match &self.core {
            Core::Flat(t) => t.audit(),
            Core::Chain(t) => t.audit(),
        }
    }
}

struct Census {
    queries: usize,
    holder_answers: usize,
    over_matches: usize,
    /// Excess position holds the right block, an EARLIER position is
    /// missing: the oracle bridged a gap (jump landing / retain
    /// guard) — the quirk M2 measured as prediction error.
    excess_gap_bridged: usize,
    /// Excess position holds the content under a DIFFERENT lineage:
    /// cross-chain coincidence (Single-entry lineage skip / retain
    /// guard on coincident content).
    excess_cross_lineage: usize,
    /// Excess position holds nothing matching: pure skip overshoot.
    excess_absent: usize,
}

/// Drive one workload through all three sides with periodic
/// checkpoints. RadixTree == model is the HARD gate; the oracle is
/// the >=-side sanity anchor with the quirk census.
fn run_differential(seed: u64, cfg: &workload::Config) -> Census {
    let wl = workload::generate(seed, cfg);
    let mut model = Model::new(wl.holders);
    let mut oracle = Oracle::new(wl.holders);
    let mut subjects = [
        Subject::new_flat(wl.holders),
        Subject::new_chain(wl.holders),
    ];
    let mut census = Census {
        queries: 0,
        holder_answers: 0,
        over_matches: 0,
        excess_gap_bridged: 0,
        excess_cross_lineage: 0,
        excess_absent: 0,
    };
    let checkpoint_every = (wl.ops.len() / 8).max(1);
    for (i, op) in wl.ops.iter().enumerate() {
        let model_outcome = model.apply(op);
        let oracle_ok = oracle.apply(op);
        if let Op::Store { .. } = op {
            let model_ok = !matches!(model_outcome, Some(StoreResult::ParentNotFound));
            assert_eq!(
                model_ok, oracle_ok,
                "oracle store acceptance diverged at op {i} (seed {seed}): {op:?}"
            );
        }
        for (ci, subject) in subjects.iter_mut().enumerate() {
            let subject_ok = subject.apply(op);
            if let Op::Store { .. } = op {
                let model_ok = !matches!(model_outcome, Some(StoreResult::ParentNotFound));
                assert_eq!(
                    model_ok, subject_ok,
                    "core{ci} store acceptance diverged at op {i} (seed {seed}): {op:?}"
                );
            }
        }
        if i % checkpoint_every == 0 || i + 1 == wl.ops.len() {
            checkpoint(
                seed,
                i,
                &model,
                &oracle,
                &mut subjects,
                &wl.queries,
                &mut census,
            );
        }
    }
    // Terminal per-holder accounting, enumeration, and distinct-entry
    // parity, for BOTH cores.
    for (ci, subject) in subjects.iter().enumerate() {
        for h in 0..wl.holders {
            assert_eq!(
                subject.holder_blocks(h),
                model.holder_blocks(h),
                "core{ci} holder_blocks diverged for holder {h} (seed {seed})"
            );
            assert_eq!(
                subject.enumerate(h),
                model.enumerate(h),
                "core{ci} enumerate diverged for holder {h} (seed {seed})"
            );
        }
        assert_eq!(
            subject.distinct_entries(),
            model.distinct_entries(),
            "core{ci} distinct_entries diverged (seed {seed})"
        );
    }
    census
}

fn checkpoint(
    seed: u64,
    at: usize,
    model: &Model,
    oracle: &Oracle,
    subjects: &mut [Subject; 2],
    queries: &[Vec<u64>],
    census: &mut Census,
) {
    for (ci, subject) in subjects.iter().enumerate() {
        subject
            .audit()
            .unwrap_or_else(|e| panic!("core{ci} audit failed (seed {seed}, op {at}): {e}"));
    }
    for query in queries {
        census.queries += 1;
        let want = model.overlap(query);
        // THE gate: both implementations equal the model, always.
        for (ci, subject) in subjects.iter_mut().enumerate() {
            let subject_map = subject.overlap(query);
            assert_eq!(
                subject_map,
                want,
                "core{ci} != model (seed {seed}, op {at}, query len {})",
                query.len()
            );
            for o in subject.scratch.iter() {
                let h = o.holder.parts().0 as usize;
                assert_eq!(
                    o.total_blocks,
                    model.holder_blocks(h),
                    "core{ci} total_blocks diverged for holder {h} (seed {seed}, op {at})"
                );
            }
        }
        let got = oracle.overlap(query);
        // Under-match anywhere = unclassifiable = fail.
        for (&holder, &depth) in &want {
            let oracle_depth = got.get(&holder).copied().unwrap_or(0);
            assert!(
                oracle_depth >= depth,
                "oracle under-matches model (seed {seed}, op {at}, holder {holder}): \
                 oracle {oracle_depth} < model {depth} on query len {}",
                query.len()
            );
        }
        census.holder_answers += want.len();
        // Over-matches: every excess position must classify.
        for (&holder, &oracle_depth) in &got {
            let model_depth = want.get(&holder).copied().unwrap_or(0);
            if oracle_depth > model_depth {
                census.over_matches += 1;
                for p in model_depth..oracle_depth {
                    if model.holds_lineage_true_at(holder, p, query) {
                        census.excess_gap_bridged += 1;
                    } else if model.holds_content_at(holder, p, query[p as usize]) {
                        census.excess_cross_lineage += 1;
                    } else {
                        census.excess_absent += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn model_vs_oracle_content_unique() {
    let mut total_over = 0usize;
    for seed in 1..=8u64 {
        let census = run_differential(seed, &workload::Config::small());
        total_over += census.over_matches;
        println!(
            "seed {seed}: {} queries, {} holder answers, {} over-matches \
             (excess positions: {} gap-bridged / {} cross-lineage / {} absent)",
            census.queries,
            census.holder_answers,
            census.over_matches,
            census.excess_gap_bridged,
            census.excess_cross_lineage,
            census.excess_absent
        );
    }
    println!("content-unique total over-matches: {total_over}");
}

#[test]
fn model_vs_oracle_with_coincidence() {
    let mut total_over = 0usize;
    for seed in 100..=107u64 {
        let census = run_differential(seed, &workload::Config::with_coincidence());
        total_over += census.over_matches;
    }
    println!("coincidence total over-matches: {total_over}");
}

/// §7: the model's observable state is identical under any
/// §7-scoped reordering of the same multiset.
#[test]
fn model_converges_within_scope() {
    for seed in 1..=6u64 {
        let wl = workload::generate(seed, &workload::Config::small());
        let mut reference = Model::new(wl.holders);
        for op in &wl.ops {
            reference.apply(op);
        }
        for reorder_seed in [7u64, 8, 9] {
            let ops = workload::reinterleave(reorder_seed, &wl.ops, wl.holders);
            let mut other = Model::new(wl.holders);
            let mut subjects = [
                Subject::new_flat(wl.holders),
                Subject::new_chain(wl.holders),
            ];
            for op in &ops {
                other.apply(op);
                for subject in subjects.iter_mut() {
                    subject.apply(op);
                }
            }
            for query in &wl.queries {
                let want = reference.overlap(query);
                assert_eq!(
                    want,
                    other.overlap(query),
                    "model diverged under scoped reordering (seed {seed}/{reorder_seed})"
                );
                for (ci, subject) in subjects.iter_mut().enumerate() {
                    assert_eq!(
                        subject.overlap(query),
                        want,
                        "core{ci} diverged under scoped reordering (seed {seed}/{reorder_seed})"
                    );
                }
            }
            for h in 0..wl.holders {
                assert_eq!(reference.holder_blocks(h), other.holder_blocks(h));
                for subject in subjects.iter() {
                    assert_eq!(subject.holder_blocks(h), reference.holder_blocks(h));
                }
            }
        }
    }
}

/// §4: a rejected store applies nothing, on both sides.
#[test]
fn rejected_store_applies_nothing() {
    let mut model = Model::new(1);
    let mut oracle = Oracle::new(1);
    let mut subjects = [Subject::new_flat(1), Subject::new_chain(1)];
    let seeded = Op::Store {
        holder: 0,
        parent: None,
        blocks: vec![(11, 101), (12, 102)],
    };
    model.apply(&seeded);
    oracle.apply(&seeded);
    for subject in subjects.iter_mut() {
        subject.apply(&seeded);
    }
    let before_model = model.overlap(&[101, 102]);
    let before_oracle = oracle.overlap(&[101, 102]);
    let before_subject: Vec<_> = subjects
        .iter_mut()
        .map(|s| s.overlap(&[101, 102]))
        .collect();
    let bad = Op::Store {
        holder: 0,
        parent: Some(999),
        blocks: vec![(13, 103)],
    };
    assert_eq!(model.apply(&bad), Some(StoreResult::ParentNotFound));
    assert!(!oracle.apply(&bad));
    for subject in subjects.iter_mut() {
        assert!(!subject.apply(&bad));
    }
    assert_eq!(model.overlap(&[101, 102]), before_model);
    assert_eq!(oracle.overlap(&[101, 102]), before_oracle);
    for (subject, before) in subjects.iter_mut().zip(&before_subject) {
        assert_eq!(subject.overlap(&[101, 102]), *before);
        assert_eq!(subject.holder_blocks(0), 2);
    }
    assert_eq!(model.holder_blocks(0), 2);
}
