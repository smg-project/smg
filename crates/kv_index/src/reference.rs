//! A single-threaded reference model of what [`PositionalIndexer`](crate::PositionalIndexer)
//! promises, for the exactness guardrail (`docs/kv-router-leap.md`, section 3).
//!
//! Nothing here is meant to be fast. Every structure is the most literal one that states the
//! contract:
//!
//! - a worker holds a set of blocks; each block is identified by the engine's sequence hash and
//!   sits at a position with a content hash and a prefix hash (the router's chain hash over the
//!   content hashes up to and including that position);
//! - a store places its blocks after the parent block of the same worker (position 0 when there
//!   is no parent) and fails exactly like the production indexer when the parent is unknown;
//! - a removal forgets the named blocks of that worker; a clear or a worker removal forgets all
//!   of them;
//! - a lookup scores a worker with the length of the longest prefix of the request for which the
//!   worker holds, at every position `i`, a block whose content hash and prefix hash equal the
//!   request's at `i`; the first position without such a block ends the prefix; workers with an
//!   empty prefix are not reported.
//!
//! The chain hash is the crate's own [`chain_prefix_hash`], not a copy of it, so both indexers
//! always agree on what a prefix hash is.

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::event_tree::{chain_prefix_hash, ApplyError, ContentHash, SequenceHash, StoredBlock};

/// One worker's blocks, indexed two ways: by the engine hash (how removals name them) and by
/// `(position, content hash)` (how lookups find them), each position holding every prefix hash
/// that reaches it.
#[derive(Default, Clone)]
struct WorkerBlocks {
    by_engine_hash: FxHashMap<SequenceHash, (usize, ContentHash, SequenceHash)>,
    by_position: FxHashMap<(usize, ContentHash), FxHashSet<SequenceHash>>,
}

impl WorkerBlocks {
    fn insert(
        &mut self,
        engine_hash: SequenceHash,
        position: usize,
        content: ContentHash,
        prefix: SequenceHash,
    ) {
        // A hash stored again at another place (a store without its parent followed by the
        // whole chain) is held at the new place only, as the production indexers hold it.
        if let Some(old) = self
            .by_engine_hash
            .insert(engine_hash, (position, content, prefix))
        {
            if old != (position, content, prefix) {
                let (old_position, old_content, old_prefix) = old;
                if let Some(prefixes) = self.by_position.get_mut(&(old_position, old_content)) {
                    prefixes.remove(&old_prefix);
                    if prefixes.is_empty() {
                        self.by_position.remove(&(old_position, old_content));
                    }
                }
            }
        }
        self.by_position
            .entry((position, content))
            .or_default()
            .insert(prefix);
    }

    fn remove(&mut self, engine_hash: SequenceHash) {
        let Some((position, content, prefix)) = self.by_engine_hash.remove(&engine_hash) else {
            return;
        };
        // Another block of this worker may reach the same (position, content) through the same
        // chain only if it has the same engine hash, so the prefix is gone with this block.
        if let Some(prefixes) = self.by_position.get_mut(&(position, content)) {
            prefixes.remove(&prefix);
            if prefixes.is_empty() {
                self.by_position.remove(&(position, content));
            }
        }
    }

    fn holds(&self, position: usize, content: ContentHash, prefix: SequenceHash) -> bool {
        self.by_position
            .get(&(position, content))
            .is_some_and(|prefixes| prefixes.contains(&prefix))
    }
}

/// The reference indexer. Workers are the `u32` ids the production indexer interns.
#[derive(Default, Clone)]
pub struct ReferenceIndexer {
    workers: BTreeMap<u32, WorkerBlocks>,
}

/// The chain of prefix hashes of a request: position 0 is the bare content hash, every later
/// position chains the previous prefix with its content hash.
pub fn request_prefix_hashes(content_hashes: &[ContentHash]) -> Vec<SequenceHash> {
    let mut chain = Vec::with_capacity(content_hashes.len());
    for (position, &content) in content_hashes.iter().enumerate() {
        let prefix = if position == 0 {
            SequenceHash(content.0)
        } else {
            chain_prefix_hash(chain[position - 1], content)
        };
        chain.push(prefix);
    }
    chain
}

impl ReferenceIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `blocks` for `worker` after `parent`, with the production indexer's placement and
    /// error rules.
    pub fn apply_stored(
        &mut self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let held = self.workers.entry(worker).or_default();
        let (start, mut previous_prefix) = match parent {
            Some(parent_hash) => {
                if held.by_engine_hash.is_empty() {
                    return Err(ApplyError::WorkerNotTracked);
                }
                let Some(&(parent_position, _, parent_prefix)) =
                    held.by_engine_hash.get(&parent_hash)
                else {
                    return Err(ApplyError::ParentBlockNotFound);
                };
                (parent_position + 1, Some(parent_prefix))
            }
            None => (0, None),
        };
        for (offset, block) in blocks.iter().enumerate() {
            let position = start + offset;
            let prefix = match previous_prefix {
                Some(previous) => chain_prefix_hash(previous, block.content_hash),
                None => SequenceHash(block.content_hash.0),
            };
            held.insert(block.seq_hash, position, block.content_hash, prefix);
            previous_prefix = Some(prefix);
        }
        Ok(())
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored, as in production.
    pub fn apply_removed(&mut self, worker: u32, engine_hashes: &[SequenceHash]) {
        if let Some(held) = self.workers.get_mut(&worker) {
            for &engine_hash in engine_hashes {
                held.remove(engine_hash);
            }
        }
    }

    /// Forget every block of `worker` (the engine cleared its cache).
    pub fn apply_cleared(&mut self, worker: u32) {
        self.workers.remove(&worker);
    }

    /// Forget every block of `worker` (the worker left).
    pub fn remove_worker(&mut self, worker: u32) {
        self.workers.remove(&worker);
    }

    /// Score every worker by the length of the longest matching prefix of the request, omitting
    /// workers that match nothing.
    pub fn find_matches(&self, content_hashes: &[ContentHash]) -> BTreeMap<u32, u32> {
        let prefixes = request_prefix_hashes(content_hashes);
        let mut scores = BTreeMap::new();
        for (&worker, held) in &self.workers {
            let matched = content_hashes
                .iter()
                .zip(&prefixes)
                .enumerate()
                .take_while(|(position, (&content, &prefix))| {
                    held.holds(*position, content, prefix)
                })
                .count();
            if matched > 0 {
                scores.insert(worker, matched as u32);
            }
        }
        scores
    }

    /// Every block every worker holds, as `(worker, position, content hash, prefix hash)`.
    pub fn blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        self.workers
            .iter()
            .flat_map(|(&worker, held)| {
                held.by_engine_hash
                    .values()
                    .map(move |&(position, content, prefix)| (worker, position, content, prefix))
            })
            .collect()
    }

    /// Number of blocks held by `worker`.
    pub fn worker_block_count(&self, worker: u32) -> usize {
        self.workers
            .get(&worker)
            .map_or(0, |held| held.by_engine_hash.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(contents: &[u64]) -> Vec<StoredBlock> {
        let hashes: Vec<ContentHash> = contents.iter().map(|&c| ContentHash(c)).collect();
        let prefixes = request_prefix_hashes(&hashes);
        hashes
            .iter()
            .zip(prefixes)
            .map(|(&content_hash, prefix)| StoredBlock {
                seq_hash: prefix,
                content_hash,
            })
            .collect()
    }

    #[test]
    fn scores_the_longest_exact_prefix_only() {
        let mut reference = ReferenceIndexer::new();
        let blocks = chain(&[1, 2, 3, 4]);
        reference.apply_stored(7, &blocks, None).expect("store");
        let query: Vec<ContentHash> = [1, 2, 3, 4].iter().map(|&c| ContentHash(c)).collect();
        assert_eq!(reference.find_matches(&query).get(&7), Some(&4));
        let diverged: Vec<ContentHash> = [1, 9, 3, 4].iter().map(|&c| ContentHash(c)).collect();
        assert_eq!(reference.find_matches(&diverged).get(&7), Some(&1));
        let other: Vec<ContentHash> = [5, 2, 3].iter().map(|&c| ContentHash(c)).collect();
        assert!(reference.find_matches(&other).is_empty());
    }

    #[test]
    fn stores_after_the_parent_and_rejects_unknown_parents() {
        let mut reference = ReferenceIndexer::new();
        let head = chain(&[1, 2]);
        reference.apply_stored(1, &head, None).expect("head");
        let tail = chain(&[1, 2, 3, 4]);
        reference
            .apply_stored(1, &tail[2..], Some(head[1].seq_hash))
            .expect("tail after parent");
        let query: Vec<ContentHash> = [1, 2, 3, 4].iter().map(|&c| ContentHash(c)).collect();
        assert_eq!(reference.find_matches(&query).get(&1), Some(&4));
        assert!(matches!(
            reference.apply_stored(1, &tail[2..], Some(SequenceHash(0xdead))),
            Err(ApplyError::ParentBlockNotFound)
        ));
        assert!(matches!(
            reference.apply_stored(2, &tail[2..], Some(head[1].seq_hash)),
            Err(ApplyError::WorkerNotTracked)
        ));
    }

    #[test]
    fn removals_truncate_the_match() {
        let mut reference = ReferenceIndexer::new();
        let blocks = chain(&[1, 2, 3, 4]);
        reference.apply_stored(3, &blocks, None).expect("store");
        reference.apply_removed(3, &[blocks[3].seq_hash, blocks[2].seq_hash]);
        let query: Vec<ContentHash> = [1, 2, 3, 4].iter().map(|&c| ContentHash(c)).collect();
        assert_eq!(reference.find_matches(&query).get(&3), Some(&2));
        assert_eq!(reference.blocks().len(), 2);
        reference.apply_cleared(3);
        assert!(reference.find_matches(&query).is_empty());
        assert!(reference.blocks().is_empty());
    }
}
