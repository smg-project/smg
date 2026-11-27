//! The reference indexer behind the [`KvIndex`](super::KvIndex) surface:
//! interning and the per-worker membership set the chain index keeps in
//! its map, over `kv_index`'s single-threaded model.

use std::collections::{BTreeSet, HashMap, HashSet};

use kv_index::{
    ApplyError, ContentHash, OverlapScores, ReferenceIndexer, SequenceHash, StoredBlock,
};
use parking_lot::Mutex;

use super::{GroupViews, WorkerBlocks};

/// The engine hashes a worker holds, mirrored from the reference so the
/// monitor's copy counting sees the same `contains_key` answers.
#[derive(Default)]
pub struct ReferenceBlocks(HashSet<SequenceHash>);

impl ReferenceBlocks {
    pub fn hashes(&self) -> Vec<SequenceHash> {
        self.0.iter().copied().collect()
    }

    pub fn contains(&self, seq_hash: SequenceHash) -> bool {
        self.0.contains(&seq_hash)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Default)]
struct Inner {
    index: ReferenceIndexer,
    names: HashMap<String, u32>,
    next: u32,
}

#[derive(Default)]
pub struct ReferenceBackend {
    inner: Mutex<Inner>,
    pub groups: GroupViews,
}

impl ReferenceBackend {
    pub fn intern_worker(&self, worker: &str) -> u32 {
        let mut inner = self.inner.lock();
        if let Some(&id) = inner.names.get(worker) {
            return id;
        }
        let id = inner.next;
        inner.next += 1;
        inner.names.insert(worker.to_string(), id);
        id
    }

    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        self.inner.lock().names.get(worker).copied()
    }

    pub fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut WorkerBlocks,
    ) -> Result<(), ApplyError> {
        self.inner
            .lock()
            .index
            .apply_stored(worker, blocks, parent)?;
        let mirror = held.reference.get_or_insert_with(ReferenceBlocks::default);
        mirror.0.extend(blocks.iter().map(|block| block.seq_hash));
        Ok(())
    }

    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
        self.inner.lock().index.apply_removed(worker, hashes);
        if let Some(mirror) = held.reference.as_mut() {
            for hash in hashes {
                mirror.0.remove(hash);
            }
        }
    }

    pub fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
        self.inner.lock().index.apply_cleared(worker);
        held.reference = None;
    }

    pub fn remove_worker(&self, worker: u32) {
        let mut inner = self.inner.lock();
        inner.index.remove_worker(worker);
        inner.names.retain(|_, id| *id != worker);
    }

    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        let inner = self.inner.lock();
        let scored = if early_exit {
            inner
                .index
                .find_matches(&content_hashes[..content_hashes.len().min(1)])
        } else {
            inner.index.find_matches(content_hashes)
        };
        let mut out = OverlapScores::default();
        for (worker, score) in scored {
            out.scores.insert(worker, score);
        }
        out
    }

    pub fn worker_block_count(&self, worker: u32) -> usize {
        self.inner.lock().index.worker_block_count(worker)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().index.blocks().is_empty()
    }

    pub fn current_size(&self) -> usize {
        self.inner.lock().index.blocks().len()
    }

    pub fn entry_count(&self) -> usize {
        let inner = self.inner.lock();
        inner
            .index
            .blocks()
            .into_iter()
            .map(|(_, position, content, prefix)| (position, content, prefix))
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        self.inner.lock().index.blocks()
    }
}
