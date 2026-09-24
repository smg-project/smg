//! The differential oracle: `kv_index::PositionalIndexer` behind the
//! same glue the service engine holds today (per-holder caller-owned
//! reverse maps, worker interning, id->holder resolution), driven by
//! the harness's operation stream.

use std::collections::BTreeMap;

use kv_index::{ContentHash, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap};

use super::Op;

pub struct Oracle {
    index: PositionalIndexer,
    /// worker_id by holder index (dense).
    ids: Vec<u32>,
    /// The engine-side caller-owned reverse maps, one per holder.
    blocks: Vec<WorkerBlockMap>,
}

impl Oracle {
    pub fn new(holder_count: usize) -> Self {
        let index = PositionalIndexer::new(64);
        let mut ids = Vec::with_capacity(holder_count);
        let mut blocks = Vec::with_capacity(holder_count);
        for h in 0..holder_count {
            let id = index
                .intern_worker(&format!("holder-{h}"))
                .expect("id space");
            ids.push(id);
            blocks.push(WorkerBlockMap::default());
        }
        Self { index, ids, blocks }
    }

    /// Apply one op the way the engine does. Returns false when the
    /// oracle rejected a store (parent not found / not tracked) — the
    /// model must have rejected it too.
    pub fn apply(&mut self, op: &Op) -> bool {
        match op {
            Op::Store {
                holder,
                parent,
                blocks,
            } => {
                let stored: Vec<StoredBlock> = blocks
                    .iter()
                    .map(|&(key, content)| StoredBlock {
                        seq_hash: SequenceHash(key),
                        content_hash: ContentHash(content),
                    })
                    .collect();
                self.index
                    .apply_stored(
                        self.ids[*holder],
                        &stored,
                        parent.map(SequenceHash),
                        &mut self.blocks[*holder],
                    )
                    .is_ok()
            }
            Op::Remove { holder, keys } => {
                let seqs: Vec<SequenceHash> = keys.iter().copied().map(SequenceHash).collect();
                self.index
                    .apply_removed(self.ids[*holder], &seqs, &mut self.blocks[*holder]);
                true
            }
            Op::Clear { holder } => {
                self.index
                    .apply_cleared(self.ids[*holder], &mut self.blocks[*holder]);
                true
            }
        }
    }

    /// Full holder->depth map for one query (depth-0 absent).
    pub fn overlap(&self, query: &[u64]) -> BTreeMap<usize, u32> {
        let hashes: Vec<ContentHash> = query.iter().copied().map(ContentHash).collect();
        let scores = self.index.find_matches(&hashes, false);
        let mut out = BTreeMap::new();
        for (holder, &id) in self.ids.iter().enumerate() {
            if let Some(&depth) = scores.scores.get(&id) {
                if depth > 0 {
                    out.insert(holder, depth);
                }
            }
        }
        out
    }
}
