//! Two cells whose host keeps an index of cache events.
//!
//! The index of the cell named `indexed` holds one block, for the list's
//! first worker and of no prompt,
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
//!
//! The index of the cell named `held` is filled behind its engine, ahead of
//! the list's first request, as the workers' streams of events would have
//! filled it ([`hold`]): every conversation of the run whole, each for one
//! worker, round the list, a worker's first one from the head the prompts
//! share and a later one behind the head it holds; then the head alone for a
//! worker no conversation reached. Nothing is stored once a request is
//! asked, so the engine reads the same sizes in every table of the list,
//! the first one too, and what a turn is named moves neither with the clock
//! nor with the picks. The index names a turn the worker that holds its
//! conversation, as deep as its prompt, and, up to the cap, workers that
//! hold the head alone, which every worker does whatever the run's
//! decisions.

use std::sync::{Arc, OnceLock};

use kv_index::{
    compute_request_content_hashes, ChainBlockMap, ChainIndex, ContentHash, SequenceHash,
    StoredBlock,
};
use parking_lot::Mutex;
use smg_grand_central::engine::{CacheIndex, CacheIndexes, InProcess};

use super::{mix, RING};

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

/// What the host of the cell named `held` keeps beside its index: for each
/// worker of the list its number there and the map of the blocks stored for
/// it, and the tokens of one block.
struct Held {
    index: Arc<ChainIndex>,
    workers: Vec<(u32, Mutex<ChainBlockMap>)>,
    block: usize,
}

/// The index of the cell named `held`: made behind its engine ([`holds`]),
/// filled ahead of the list's first request ([`hold`]).
static HELD: OnceLock<Held> = OnceLock::new();

/// What the cell named `held` makes behind its engine: the host's index,
/// made for the list's length, with a number for every worker of the list.
pub(super) fn holds(engine: &InProcess) {
    let list = RING.get().expect("the run's list");
    let index = Arc::new(ChainIndex::with_max_workers(list.0.len()));
    let numbered = |address: &String| {
        let number = index.intern_worker(address);
        let number = number.expect("a number for every worker: an index numbers 1,024");
        (number, Mutex::new(ChainBlockMap::new()))
    };
    let held = Held {
        workers: list.0.iter().map(numbered).collect(),
        block: engine.spec().block_size().max(1) as usize,
        index: Arc::clone(&index),
    };
    assert!(HELD.set(held).is_ok(), "one index a run");
    engine.affinity().set_indexes(Arc::new(Keeps(index)));
}

/// Stores the prompt `full` for the list's worker at `place`, counted round
/// the list, as that worker's stream would have stored it: each block named
/// by the hash of its tokens and by one of the prompt up to it, behind the
/// last block of the prompt the worker holds already, which is the head's
/// once the worker holds the head; a prompt the worker holds whole stores
/// nothing. An index declines a block it has no room for: the run stops here
/// then, ahead of its callers, which would wait at their start for one that
/// stopped alone.
pub(super) fn hold(place: usize, full: &[u32]) {
    let held = HELD.get().expect("the cell's index");
    let (number, map) = &held.workers[place % held.workers.len()];
    let mut sequence = 0;
    let named = |content_hash: ContentHash| {
        sequence = mix(sequence ^ content_hash.0);
        StoredBlock {
            seq_hash: SequenceHash(sequence),
            content_hash,
        }
    };
    let hashes = compute_request_content_hashes(full, held.block);
    let blocks: Vec<StoredBlock> = hashes.into_iter().map(named).collect();
    let mut map = map.lock();
    let known = blocks.partition_point(|block| map.contains_key(block.seq_hash));
    if known == blocks.len() {
        return;
    }
    let parent = known.checked_sub(1).map(|last| blocks[last].seq_hash);
    let stored = held
        .index
        .apply_stored(*number, &blocks[known..], parent, &mut map);
    stored.expect("the index takes the prompt: fewer callers or decisions");
}
