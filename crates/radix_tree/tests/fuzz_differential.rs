//! Campaign C1/C3/C4: wide-config differential fuzz plus
//! out-of-contract chaos.
//!
//! In-contract mode: randomized workload configs far beyond the
//! normative shapes, RadixTree == model at every checkpoint, full
//! `audit()` at every checkpoint (and after EVERY op on small
//! workloads).
//!
//! Chaos mode: deliberately violates every §7 precondition — random
//! parents (including keys inside the same batch and never-seen
//! keys), keys reused across positions and chains, interleaved
//! retires/recreates, operations through stale ids, truncates at
//! random keeps. The contract here is: NO panics, `audit()` holds
//! after EVERY op, stale ids are loud no-ops, and a bit-for-bit
//! replay of the same sequence lands in an identical observable
//! state (determinism).
//!
//! `fuzz_quick` (32+8 seeds) always runs. The campaign entry point:
//!   RADIX_FUZZ_SEEDS=10000 cargo test -p radix-tree --release \
//!     --test fuzz_differential -- --ignored --nocapture

mod common;

use std::collections::BTreeMap;

use common::{
    model::{Model, StoreResult},
    workload::{self, Config as WlConfig},
    Op, Rng,
};
use radix_tree::{Config, FlatTree, HolderId, OverlapScratch, RadixTree, StoreError};

/// `wide` raises the sharing cap to the bench's H=64 width — model
/// scans at that width are too slow for the always-on debug quick
/// suite, so wide configs belong to the release campaign entry point
/// (audit finding: correctness past ~27 holders/block was previously
/// never model-gated ANYWHERE).
fn random_config(rng: &mut Rng, wide: bool) -> WlConfig {
    let holders = 2 + rng.below(255);
    let share_cap = if wide { 64 } else { 24 };
    WlConfig {
        holders,
        families: 1 + rng.below(64),
        family_len: {
            let lo = 1 + rng.below(64);
            (lo, lo + 1 + rng.below(448))
        },
        holders_per_family: {
            let lo = 1 + rng.below(4);
            (lo, lo + rng.below(holders.min(share_cap)))
        },
        tail_len: (1 + rng.below(8), 9 + rng.below(56)),
        store_batch: 1 + rng.below(16),
        duplicate_pct: rng.below(41) as u64,
        gap_pct: rng.below(41) as u64,
        clear_pct: rng.below(31) as u64,
        coincidence_pct: rng.below(51) as u64,
    }
}

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
    fn apply(&mut self, op: &Op) -> Option<(u32, u32)> {
        match op {
            Op::Store {
                holder,
                parent,
                blocks,
            } => {
                // covered() is the shared-lock duplicate fast path's
                // load-bearing predicate: gate it as an exact no-op
                // oracle on EVERY store of every seed.
                let predicted = match &self.core {
                    Core::Flat(t) => Some(t.covered(self.ids[*holder], *parent, blocks)),
                    Core::Chain(t) => Some(t.covered(self.ids[*holder], *parent, blocks)),
                };
                let r = match &mut self.core {
                    Core::Flat(t) => t.store(self.ids[*holder], *parent, blocks),
                    Core::Chain(t) => t.store(self.ids[*holder], *parent, blocks),
                };
                if let Some(predicted) = predicted {
                    let no_op = matches!(&r, Ok(o) if o.applied == 0);
                    assert_eq!(
                        predicted, no_op,
                        "covered() disagreed with store outcome {r:?}"
                    );
                }
                match r {
                    Ok(o) => Some((o.applied, o.duplicates)),
                    Err(StoreError::ParentNotFound) => None,
                    Err(e) => panic!("unexpected store error {e:?}"),
                }
            }
            Op::Remove { holder, keys } => {
                match &mut self.core {
                    Core::Flat(t) => t.remove(self.ids[*holder], keys),
                    Core::Chain(t) => t.remove(self.ids[*holder], keys),
                };
                Some((0, 0))
            }
            Op::Clear { holder } => {
                match &mut self.core {
                    Core::Flat(t) => t.clear(self.ids[*holder]),
                    Core::Chain(t) => t.clear(self.ids[*holder]),
                }
                Some((0, 0))
            }
        }
    }
    fn overlap(&mut self, query: &[u64]) -> BTreeMap<usize, u32> {
        let scratch = &mut self.scratch;
        match &mut self.core {
            Core::Flat(t) => t.overlap(query, &mut self.qscratch, scratch),
            Core::Chain(t) => t.overlap(query, &mut self.qscratch, scratch),
        }
        let mut out = BTreeMap::new();
        for o in scratch.iter() {
            out.insert(o.holder.parts().0 as usize, o.depth);
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
    fn dup_prefix(&self, h: usize, parent: Option<u64>, blocks: &[(u64, u64)]) -> (u32, bool) {
        match &self.core {
            Core::Flat(t) => t.dup_prefix(self.ids[h], parent, blocks),
            Core::Chain(t) => t.dup_prefix(self.ids[h], parent, blocks),
        }
    }
    fn position_of(&self, h: usize, key: u64) -> Option<u32> {
        match &self.core {
            Core::Flat(t) => t.position_of(self.ids[h], key),
            Core::Chain(t) => t.position_of(self.ids[h], key),
        }
    }
    fn audit(&self) -> Result<(), String> {
        match &self.core {
            Core::Flat(t) => t.audit(),
            Core::Chain(t) => t.audit(),
        }
    }
}

fn run_one_in_contract(seed: u64, wide: bool) {
    let mut rng = Rng::new(seed ^ 0xF00D);
    let cfg = random_config(&mut rng, wide);
    let wl = workload::generate(seed, &cfg);
    let audit_every_op = wl.ops.len() < 4000;
    let mut model = Model::new(wl.holders);
    let mut subjects = [
        Subject::new_flat(wl.holders),
        Subject::new_chain(wl.holders),
    ];
    let checkpoint_every = (wl.ops.len() / 6).max(1);
    for (i, op) in wl.ops.iter().enumerate() {
        if let Op::Store {
            holder,
            parent,
            blocks,
        } = op
        {
            // dup_prefix parity: the engine's split-apply re-anchors a
            // store at blocks[run-1] and applies only the suffix, so a
            // wrong run means silently corrupted placements.
            let expect = model.dup_prefix(*holder, *parent, blocks);
            for subject in subjects.iter() {
                assert_eq!(
                    subject.dup_prefix(*holder, *parent, blocks),
                    expect,
                    "dup_prefix diverged from model"
                );
                for &(key, _) in blocks.iter() {
                    assert_eq!(
                        subject.position_of(*holder, key),
                        model.position_of(*holder, key),
                        "position_of diverged from model"
                    );
                }
            }
        }
        let model_outcome = model.apply(op);
        for (ci, subject) in subjects.iter_mut().enumerate() {
            let subject_out = subject.apply(op);
            if let Op::Store { .. } = op {
                // StoreOutcome PARITY, not just acceptance: the relay
                // gate depends on applied counts (audit finding — the
                // payload was discarded everywhere).
                match (&model_outcome, &subject_out) {
                    (Some(StoreResult::ParentNotFound), None) => {}
                    (
                        Some(StoreResult::Applied {
                            applied,
                            duplicates,
                        }),
                        Some((a, d)),
                    ) => {
                        assert_eq!(
                            (*applied, *duplicates),
                            (*a, *d),
                            "core{ci} StoreOutcome diverged: seed {seed} op {i}"
                        );
                    }
                    other => panic!("core{ci} acceptance diverged: seed {seed} op {i}: {other:?}"),
                }
            }
            if audit_every_op {
                subject
                    .audit()
                    .unwrap_or_else(|e| panic!("core{ci} audit failed: seed {seed} op {i}: {e}"));
            }
        }
        if i % checkpoint_every == 0 || i + 1 == wl.ops.len() {
            for (ci, subject) in subjects.iter_mut().enumerate() {
                if !audit_every_op {
                    subject.audit().unwrap_or_else(|e| {
                        panic!("core{ci} audit failed: seed {seed} op {i}: {e}")
                    });
                }
                for query in &wl.queries {
                    assert_eq!(
                        subject.overlap(query),
                        model.overlap(query),
                        "core{ci} != model: seed {seed} op {i}"
                    );
                }
            }
        }
    }
    for (ci, subject) in subjects.iter().enumerate() {
        for h in 0..wl.holders {
            assert_eq!(
                subject.holder_blocks(h),
                model.holder_blocks(h),
                "core{ci} holder_blocks diverged: seed {seed} holder {h}"
            );
            // Terminal ENUMERATE parity: per-block position/key/
            // content — off-query-path corruption was invisible to
            // the whole campaign without this (audit finding).
            assert_eq!(
                subject.enumerate(h),
                model.enumerate(h),
                "core{ci} enumerate diverged: seed {seed} holder {h}"
            );
        }
        assert_eq!(
            subject.distinct_entries(),
            model.distinct_entries(),
            "core{ci} distinct_entries diverged: seed {seed}"
        );
    }
}

/// Uniform surface over both cores for the generic chaos driver.
trait CoreApi {
    fn create_holder(&mut self, name: &str) -> HolderId;
    fn dup_prefix(&self, id: HolderId, parent: Option<u64>, blocks: &[(u64, u64)]) -> (u32, bool);
    fn position_of(&self, id: HolderId, key: u64) -> Option<u32>;
    fn store(
        &mut self,
        id: HolderId,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
    ) -> Result<radix_tree::StoreOutcome, StoreError>;
    fn remove(&mut self, id: HolderId, keys: &[u64]) -> u32;
    fn clear(&mut self, id: HolderId);
    fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64;
    fn retire_holder(&mut self, id: HolderId);
    fn holder_blocks(&self, id: HolderId) -> u64;
    fn holder_name(&self, id: HolderId) -> Option<&str>;
    fn overlap(&self, q: &[u64], sc: &mut OverlapScratch, out: &mut Vec<radix_tree::Overlap>);
    fn audit(&self) -> Result<(), String>;
    fn distinct_entries(&self) -> u64;
}

macro_rules! impl_core_api {
    ($t:ty) => {
        impl CoreApi for $t {
            fn create_holder(&mut self, name: &str) -> HolderId {
                <$t>::create_holder(self, name)
            }
            fn dup_prefix(
                &self,
                id: HolderId,
                parent: Option<u64>,
                blocks: &[(u64, u64)],
            ) -> (u32, bool) {
                <$t>::dup_prefix(self, id, parent, blocks)
            }
            fn position_of(&self, id: HolderId, key: u64) -> Option<u32> {
                <$t>::position_of(self, id, key)
            }
            fn store(
                &mut self,
                id: HolderId,
                parent: Option<u64>,
                blocks: &[(u64, u64)],
            ) -> Result<radix_tree::StoreOutcome, StoreError> {
                <$t>::store(self, id, parent, blocks)
            }
            fn remove(&mut self, id: HolderId, keys: &[u64]) -> u32 {
                <$t>::remove(self, id, keys)
            }
            fn clear(&mut self, id: HolderId) {
                <$t>::clear(self, id)
            }
            fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64 {
                <$t>::truncate_tail(self, id, keep)
            }
            fn retire_holder(&mut self, id: HolderId) {
                <$t>::retire_holder(self, id)
            }
            fn holder_blocks(&self, id: HolderId) -> u64 {
                <$t>::holder_blocks(self, id)
            }
            fn holder_name(&self, id: HolderId) -> Option<&str> {
                <$t>::holder_name(self, id)
            }
            fn overlap(
                &self,
                q: &[u64],
                sc: &mut OverlapScratch,
                out: &mut Vec<radix_tree::Overlap>,
            ) {
                <$t>::overlap(self, q, sc, out)
            }
            fn audit(&self) -> Result<(), String> {
                <$t>::audit(self)
            }
            fn distinct_entries(&self) -> u64 {
                <$t>::stats(self).distinct_entries
            }
        }
    };
}
impl_core_api!(FlatTree);
impl_core_api!(RadixTree);

/// Chaos: arbitrary op soup. Contract: no panic, audit always green,
/// stale-id ops are loud no-ops, replay is deterministic — for BOTH
/// cores, each against the total model.
fn run_one_chaos(seed: u64) {
    #[derive(Clone, Debug)]
    enum COp {
        Store {
            slot: usize,
            parent_pick: u64,
            blocks: Vec<(u64, u64)>,
        },
        Remove {
            slot: usize,
            keys: Vec<u64>,
        },
        Clear {
            slot: usize,
        },
        Truncate {
            slot: usize,
            keep: u64,
        },
        Retire {
            slot: usize,
        },
        Recreate {
            slot: usize,
        },
        StaleProbe {
            slot: usize,
        },
        Query {
            probe: Vec<u64>,
        },
    }
    let mut rng = Rng::new(seed ^ 0xC4A05);
    let slots = 2 + rng.below(12);
    let ops_count = 400 + rng.below(1200);
    // Key pool encourages collisions across positions and holders.
    let key_pool: Vec<u64> = (0..64).map(|_| rng.next() | 1).collect();
    let content_pool: Vec<u64> = (0..48).map(|_| rng.next() | 1).collect();
    let mut script = Vec::with_capacity(ops_count);
    for _ in 0..ops_count {
        let slot = rng.below(slots);
        script.push(match rng.below(100) {
            0..=44 => {
                let len = 1 + rng.below(12);
                let blocks: Vec<(u64, u64)> = (0..len)
                    .map(|_| {
                        (
                            key_pool[rng.below(key_pool.len())],
                            content_pool[rng.below(content_pool.len())],
                        )
                    })
                    .collect();
                COp::Store {
                    slot,
                    parent_pick: rng.next(),
                    blocks,
                }
            }
            45..=59 => COp::Remove {
                slot,
                keys: (0..1 + rng.below(6))
                    .map(|_| key_pool[rng.below(key_pool.len())])
                    .collect(),
            },
            60..=66 => COp::Clear { slot },
            67..=74 => COp::Truncate {
                slot,
                keep: rng.below(40) as u64,
            },
            75..=81 => COp::Retire { slot },
            82..=88 => COp::Recreate { slot },
            89..=92 => COp::StaleProbe { slot },
            _ => COp::Query {
                probe: (0..1 + rng.below(16))
                    .map(|_| content_pool[rng.below(content_pool.len())])
                    .collect(),
            },
        });
    }

    fn chaos_run<T: CoreApi>(
        seed: u64,
        slots: usize,
        key_pool: &[u64],
        script: &[COp],
        mut tree: T,
    ) -> (Vec<BTreeMap<usize, u32>>, u64) {
        // The model is total (defined on arbitrary inputs), so chaos
        // now runs under the same hard referee gate as the
        // in-contract fuzz — keyed by chaos slot.
        let mut model = Model::with_max(slots, 64);
        let mut ids: Vec<Option<HolderId>> = (0..slots)
            .map(|s| Some(tree.create_holder(&format!("chaos-{s}"))))
            .collect();
        let mut stale: Vec<HolderId> = Vec::new();
        let mut answers = Vec::new();
        let mut scratch = Vec::new();
        let mut qscratch = OverlapScratch::default();
        for (i, op) in script.iter().enumerate() {
            match op {
                COp::Store {
                    slot,
                    parent_pick,
                    blocks,
                } => {
                    if let Some(id) = ids[*slot] {
                        // Parent: none / a pooled key / a never-seen key.
                        let parent = match parent_pick % 3 {
                            0 => None,
                            1 => Some(key_pool[(*parent_pick as usize / 3) % key_pool.len()]),
                            _ => Some(parent_pick | 1),
                        };
                        // Chaos-side covered() gate: subject and model
                        // must agree on the no-op prediction too.
                        let predicted_pair = tree.dup_prefix(id, parent, blocks);
                        assert_eq!(
                            predicted_pair,
                            model.dup_prefix(*slot, parent, blocks),
                            "dup_prefix diverged from model (chaos)"
                        );
                        for &(key, _) in blocks.iter() {
                            assert_eq!(
                                tree.position_of(id, key),
                                model.position_of(*slot, key),
                                "position_of diverged from model (chaos)"
                            );
                        }
                        let predicted = predicted_pair.1;
                        let subject = tree.store(id, parent, blocks);
                        let modeled = model.store(*slot, parent, blocks);
                        let no_op = matches!(&subject, Ok(o) if o.applied == 0);
                        assert_eq!(predicted, no_op, "covered() wrong vs chaos store");
                        let pair = (
                            subject.is_ok(),
                            !matches!(
                                modeled,
                                StoreResult::ParentNotFound | StoreResult::ChainTooLong
                            ),
                        );
                        assert!(
                            pair.0 == pair.1,
                            "store acceptance diverged (seed {seed} op {i}): {subject:?} vs {modeled:?}"
                        );
                    }
                }
                COp::Remove { slot, keys } => {
                    if let Some(id) = ids[*slot] {
                        let a = tree.remove(id, keys);
                        let b = model.remove(*slot, keys);
                        assert_eq!(a, b, "remove count diverged (seed {seed} op {i})");
                    }
                }
                COp::Clear { slot } => {
                    if let Some(id) = ids[*slot] {
                        tree.clear(id);
                        model.clear(*slot);
                    }
                }
                COp::Truncate { slot, keep } => {
                    if let Some(id) = ids[*slot] {
                        let a = tree.truncate_tail(id, *keep);
                        let b = model.truncate_tail(*slot, *keep);
                        assert_eq!(a, b, "truncate count diverged (seed {seed} op {i})");
                    }
                }
                COp::Retire { slot } => {
                    if let Some(id) = ids[*slot].take() {
                        tree.retire_holder(id);
                        model.clear(*slot);
                        stale.push(id);
                    }
                }
                COp::Recreate { slot } => {
                    if ids[*slot].is_none() {
                        ids[*slot] = Some(tree.create_holder(&format!("chaos-{slot}-re{i}")));
                    }
                }
                COp::StaleProbe { slot } => {
                    // Every stale id must be a loud no-op forever.
                    if let Some(&old) = stale.last() {
                        assert_eq!(
                            tree.store(old, None, &[(1, 1)]),
                            Err(StoreError::UnknownHolder),
                            "stale id accepted a store (seed {seed} op {i} slot {slot})"
                        );
                        assert_eq!(tree.remove(old, &[1]), 0);
                        assert_eq!(tree.holder_blocks(old), 0);
                        assert_eq!(tree.holder_name(old), None);
                    }
                }
                COp::Query { probe } => {
                    tree.overlap(probe, &mut qscratch, &mut scratch);
                    let mut by_slot = BTreeMap::new();
                    for o in scratch.iter() {
                        // Map subject holder ids back to chaos slots;
                        // an unmapped id would be a stale-holder leak.
                        let slot = ids
                            .iter()
                            .position(|x| *x == Some(o.holder))
                            .unwrap_or_else(|| {
                                panic!("answer for unknown holder (seed {seed} op {i})")
                            });
                        by_slot.insert(slot, o.depth);
                    }
                    assert_eq!(
                        by_slot,
                        model.overlap(probe),
                        "chaos subject != model (seed {seed} op {i})"
                    );
                    answers.push(by_slot);
                }
            }
            tree.audit()
                .unwrap_or_else(|e| panic!("chaos audit failed: seed {seed} op {i}: {e}"));
            for (slot, id) in ids.iter().enumerate() {
                if let Some(id) = id {
                    assert_eq!(
                        tree.holder_blocks(*id),
                        model.holder_blocks(slot),
                        "chaos holder_blocks diverged (seed {seed} op {i} slot {slot})"
                    );
                }
            }
        }
        let distinct = tree.distinct_entries();
        assert_eq!(
            distinct,
            model.distinct_entries(),
            "chaos distinct_entries diverged (seed {seed})"
        );
        (answers, distinct)
    }
    // Determinism per core, and cross-core agreement, all vs model.
    let flat_a = chaos_run(
        seed,
        slots,
        &key_pool,
        &script,
        FlatTree::new(Config { max_chain_len: 64 }),
    );
    let flat_b = chaos_run(
        seed,
        slots,
        &key_pool,
        &script,
        FlatTree::new(Config { max_chain_len: 64 }),
    );
    assert_eq!(flat_a, flat_b, "flat chaos replay diverged (seed {seed})");
    let chain_a = chaos_run(
        seed,
        slots,
        &key_pool,
        &script,
        RadixTree::new(Config { max_chain_len: 64 }),
    );
    assert_eq!(
        flat_a, chain_a,
        "flat vs chain chaos diverged (seed {seed})"
    );
}

#[test]
fn fuzz_quick() {
    for seed in 1..=32u64 {
        run_one_in_contract(seed, false);
    }
    for seed in 1..=8u64 {
        run_one_chaos(seed);
    }
}

#[test]
#[ignore = "campaign entry point; set RADIX_FUZZ_SEEDS"]
fn fuzz_campaign() {
    let seeds: u64 = std::env::var("RADIX_FUZZ_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let start: u64 = std::env::var("RADIX_FUZZ_START")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    for seed in start..start + seeds {
        // Every 4th campaign seed runs the wide-sharing (H<=64)
        // distribution under the full model gate.
        run_one_in_contract(seed, seed % 4 == 0);
        if seed % 3 == 0 {
            run_one_chaos(seed);
        }
        if (seed - start) % 250 == 249 {
            println!("fuzz: {} seeds green", seed - start + 1);
        }
    }
    println!("fuzz campaign: {seeds} seeds green");
}
