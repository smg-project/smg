//! The implementations under test, the flat (R1) and the chain-native
//! (R3) core, behind one handle driven by the harness's dense holder
//! indices. Every observable is asserted for BOTH against the model.
//! Each differential binary keeps its own `apply` policy over these
//! primitives: acceptance only, or the full store outcome behind the
//! `covered()` no-op gate.

use std::collections::BTreeMap;

use radix_tree::{
    Config, FlatTree, HolderId, Overlap, OverlapScratch, RadixTree, StoreError, StoreOutcome,
};

enum Core {
    Flat(FlatTree),
    Chain(RadixTree),
}

pub struct Subject {
    core: Core,
    ids: Vec<HolderId>,
    scratch: Vec<Overlap>,
    qscratch: OverlapScratch,
}

impl Subject {
    pub fn new_flat(holders: usize) -> Self {
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

    pub fn new_chain(holders: usize) -> Self {
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

    /// The core's own store outcome; the binary's `apply` decides what
    /// counts as accepted.
    pub fn store(
        &mut self,
        holder: usize,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
    ) -> Result<StoreOutcome, StoreError> {
        match &mut self.core {
            Core::Flat(t) => t.store(self.ids[holder], parent, blocks),
            Core::Chain(t) => t.store(self.ids[holder], parent, blocks),
        }
    }

    pub fn remove(&mut self, holder: usize, keys: &[u64]) -> u32 {
        match &mut self.core {
            Core::Flat(t) => t.remove(self.ids[holder], keys),
            Core::Chain(t) => t.remove(self.ids[holder], keys),
        }
    }

    pub fn clear(&mut self, holder: usize) {
        match &mut self.core {
            Core::Flat(t) => t.clear(self.ids[holder]),
            Core::Chain(t) => t.clear(self.ids[holder]),
        }
    }

    /// The read-only no-op prediction for `store(holder, parent, blocks)`.
    pub fn covered(&self, holder: usize, parent: Option<u64>, blocks: &[(u64, u64)]) -> bool {
        match &self.core {
            Core::Flat(t) => t.covered(self.ids[holder], parent, blocks),
            Core::Chain(t) => t.covered(self.ids[holder], parent, blocks),
        }
    }

    pub fn dup_prefix(&self, h: usize, parent: Option<u64>, blocks: &[(u64, u64)]) -> (u32, bool) {
        match &self.core {
            Core::Flat(t) => t.dup_prefix(self.ids[h], parent, blocks),
            Core::Chain(t) => t.dup_prefix(self.ids[h], parent, blocks),
        }
    }

    pub fn position_of(&self, h: usize, key: u64) -> Option<u32> {
        match &self.core {
            Core::Flat(t) => t.position_of(self.ids[h], key),
            Core::Chain(t) => t.position_of(self.ids[h], key),
        }
    }

    pub fn overlap(&mut self, query: &[u64]) -> BTreeMap<usize, u32> {
        let scratch = &mut self.scratch;
        match &mut self.core {
            Core::Flat(t) => t.overlap(query, &mut self.qscratch, scratch),
            Core::Chain(t) => t.overlap(query, &mut self.qscratch, scratch),
        }
        // Shape check on the RAW Vec before the BTreeMap collapse: every
        // holder must appear at most once. The map projection would hide a
        // merge-loop regression that emitted a holder twice (a downstream
        // consumer iterating the Vec would then double-count), exactly
        // where the widest seed space runs.
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

    /// The raw answers of the last `overlap` call (the per-holder totals
    /// ride on them).
    pub fn last_overlaps(&self) -> &[Overlap] {
        &self.scratch
    }

    pub fn holder_blocks(&self, h: usize) -> u64 {
        match &self.core {
            Core::Flat(t) => t.holder_blocks(self.ids[h]),
            Core::Chain(t) => t.holder_blocks(self.ids[h]),
        }
    }

    pub fn enumerate(&self, h: usize) -> Vec<(u32, u64, u64)> {
        match &self.core {
            Core::Flat(t) => t.enumerate(self.ids[h]).collect(),
            Core::Chain(t) => t.enumerate(self.ids[h]).collect(),
        }
    }

    pub fn distinct_entries(&self) -> u64 {
        match &self.core {
            Core::Flat(t) => t.stats().distinct_entries,
            Core::Chain(t) => t.stats().distinct_entries,
        }
    }

    pub fn audit(&self) -> Result<(), String> {
        match &self.core {
            Core::Flat(t) => t.audit(),
            Core::Chain(t) => t.audit(),
        }
    }
}
