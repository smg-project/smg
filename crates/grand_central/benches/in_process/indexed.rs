//! A cell whose host keeps an index of cache events.
//!
//! The index holds one block, for the list's first worker and of no prompt,
//! and the host lends no tree. A turn hands token ids, so the index names it
//! alone: the engine's record of sent prefixes is neither read nor written,
//! and a worker the index holds nothing for is named nothing. No worker's
//! stream feeds the index here. So no turn finds its conversation in it, and
//! a turn goes back to its worker by the profile's other stages alone.
//!
//! The bench publishes its load reports once, unless `--publish-ms` says
//! how often. Where nothing names a turn's worker, the load stages then
//! send every turn of a long run to one worker. That reads as a return and
//! is no memory of a conversation: read this cell's `return_rate` under
//! `--publish-ms 5`, or over a short run.

use std::sync::Arc;

use kv_index::{ChainBlockMap, ChainIndex, ContentHash, SequenceHash, StoredBlock};
use smg_grand_central::engine::{CacheIndex, CacheIndexes, InProcess};

/// The host's index of cache events, one for every model.
struct Keeps(Arc<ChainIndex>);

impl CacheIndexes for Keeps {
    fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
        Some(Arc::clone(&self.0) as Arc<dyn CacheIndex>)
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        None
    }
}

/// What the cell makes behind its engine: the host's index, with one block
/// stored for the list's first worker as its event stream would store it.
pub(super) fn keeps(engine: &InProcess) {
    let index = ChainIndex::new();
    let first = index.intern_worker(&super::address(0)).expect("a number");
    let block = StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: ContentHash(1),
    };
    index
        .apply_stored(first, &[block], None, &mut ChainBlockMap::new())
        .expect("the block is stored");
    engine
        .affinity()
        .set_indexes(Arc::new(Keeps(Arc::new(index))));
}
