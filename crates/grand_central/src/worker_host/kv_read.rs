//! What the engine over worker lists reads of a host's cache events: the
//! index the host's event streams write into its KV tables
//! ([`KvTables`]), lent through the engine's own traits. No block is copied,
//! and no stream is read twice.
//!
//! The engine names a worker by its address, and the tables name a row by
//! its id, which holds the registration's incarnation. So a worker's number
//! is found from its address through the registry and the holder's record
//! beside the object: the worker that stands under the address now, which
//! is the one a request to the address reaches. The registry is asked as
//! the holder asks it where it lends a list ([`Objects::lent`]).
//!
//! A row's number is its lane's place in the model's index, held from the
//! lane's making: a row that asks for cache events has its number ahead of
//! its stream, as a gateway's index numbers a worker where its stream's
//! task opens its place. The index holds no block of such a row until its
//! stream writes. On a pair's side the engine reads a worker with no block
//! as one the record of sent prefixes speaks for. A single request of token
//! ids is decided by the index alone, and names such a worker by nothing.
//! An index that holds no block is none to the engine: it is lent all the
//! same, and answers the engine's question of it in constant time
//! ([`CacheIndex::holds_a_block`]).
//!
//! # Locks
//!
//! A read here takes, one after the other and never one under another: the
//! registry's own read for the address's id, the holder's entry under its
//! read lock for the row beside the object, then the tables' table lock for
//! the row's place. No lane's lock is taken, so a read waits for no batch
//! a worker's stream writes. The holder's write turn takes its entry and
//! then the tables (the table's lock, then a lane's), never the other way,
//! and it asks no list and calls nothing of the engine. A decision holds no lock of the store while it reads here:
//! it decides over a loaded pointer. A served decision already makes the
//! first two reads, once a generation, where it asks the holder for its
//! lists ([`Objects::lent`]).
//!
//! # A number taken again
//!
//! An index id is free for the next row once its row leaves, and the engine
//! maps ids to the workers of a list once a table, which stands up to a
//! second. The chain index counts each freed id it hands to the next row,
//! and a lending answers that count ([`CacheIndex::numbers_reused`]). The
//! engine reads a table anew once the count moved, and a lookup it moved
//! under names nobody. So an id taken again inside a table's second never
//! reads as the worker that held it when the table was read.
//!
//! # Limits
//!
//! - The engine asks one index a list, by its first worker's model
//!   ([`CacheIndexes::index`]), and a row is indexed under its first model.

use std::sync::{Arc, Weak};

use smg_worker::registry::Generation;

use super::{kv_tables::LentIndex, KvTables, Objects};
use crate::engine::{overlap, CacheIndex, CacheIndexes};

/// A host's indexes of cache events as the engine over worker lists asks
/// for them: the tables' own, a model each.
pub(crate) struct EventIndexes<G> {
    tables: Arc<KvTables>,
    /// Held weakly: an engine keeps these for its life, and the holder
    /// holds the registry whose lists hold that engine's tables.
    objects: Weak<Objects<G>>,
}

impl<G: Generation> EventIndexes<G> {
    /// The indexes of `tables`, read by the addresses of the worker objects
    /// `objects` holds. `tables` are the ones that holder was handed
    /// ([`Objects::with_kv_tables`]): a host that keeps both hands both, so
    /// no reader is left with tables nothing writes.
    pub(crate) fn new(tables: Arc<KvTables>, objects: &Arc<Objects<G>>) -> Self {
        debug_assert!(
            objects
                .kv_tables()
                .is_some_and(|held| Arc::ptr_eq(&held, &tables)),
            "the tables lent are the holder's own"
        );
        Self {
            tables,
            objects: Arc::downgrade(objects),
        }
    }
}

impl<G: Generation> CacheIndexes for EventIndexes<G> {
    /// A lending of the model's index, made anew at each call: the engine
    /// asks once a table it reads. It is lent whenever the model has one,
    /// and says itself whether it holds a block
    /// ([`CacheIndex::holds_a_block`]): the engine reads one that holds
    /// none as no index, and reads its table again once that word moved.
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        let lent = self.tables.lend(model)?;
        let objects = Weak::clone(&self.objects);
        Some(Arc::new(EventIndex { lent, objects }))
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.tables.block_size(model)
    }
}

/// One model's index, lent for one table of the engine's.
struct EventIndex<G> {
    lent: LentIndex,
    objects: Weak<Objects<G>>,
}

impl<G: Generation> CacheIndex for EventIndex<G> {
    fn worker_number(&self, address: &str) -> Option<u32> {
        let objects = self.objects.upgrade()?;
        // The registry first: no lock of the holder is held across it.
        let object = objects.registry().get_id_by_url(address)?;
        let row = objects.standing(&object)?.row;
        self.lent.number_of(row)
    }

    /// The count of the chain the lanes write: it moves where a row takes
    /// the id another row freed, ahead of that row's first block.
    fn numbers_reused(&self) -> u64 {
        self.lent.ids_reused()
    }

    /// The chain's own test of its root, in constant time: no lock, and no
    /// walk of the rows.
    fn holds_a_block(&self) -> bool {
        !self.lent.is_empty()
    }

    fn worker_blocks(&self, number: u32) -> usize {
        self.lent.blocks(number)
    }

    fn deepest_into(
        &self,
        block_hashes: &[u64],
        block_size: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    ) {
        let (chain, groups) = self.lent.chain();
        overlap::deepest_within_groups(
            chain,
            groups,
            block_hashes,
            block_size,
            within,
            cap,
            below,
            kept,
        );
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/kv_read/tests.rs"]
mod tests;
