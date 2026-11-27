//! The holder's read side: a door reads the record that stands off the
//! worker objects.
//!
//! A door reads a record before it writes one: the row under a URL, the
//! row under an id, an engine's rows. The holder takes its turn
//! and reads the record back from the object that stands
//! ([`Objects::row_of`]). That is the record the holder's write decides
//! over ([`mod@super::write`]), so a door's test and the write behind it
//! read one record. The two lookups stand here once: the write side asks
//! them inside the turn it holds.
//!
//! The turn makes a door's read exact. A
//! writer of a row holds the turn from its read to its put, so a door that
//! waited behind it reads the object that writer left, and never the
//! registry one write behind the table.
//!
//! A worker object's status leads its row's ([`mod@super::status`]): a
//! record a door reads carries the status its object states, and no read
//! here asks the store's table for a row.
//!
//! An engine's rows are found by the holder's index of ranks. It lists a
//! row under the address it dials: every row that names the engine, and
//! the row under the engine's own URL that names none. The read keeps the
//! rows that name the engine, which is what a door asks.
//!
//! A first copy is cut here too: every row at one place of the journal,
//! for a watch's start, a standby's restore and an operator's read
//! ([`Objects::checkpoint`]), and the few rows of an operator's lone read
//! ([`Objects::rows_serving`]). A row's facts are read off its object in
//! the holder's turn. Its status is the one the journal told
//! ([`MemoryStore::told`]), not its object's: an object stands in a
//! status its log has not told yet between a driver's move and the tell,
//! and a copy holds the log's word at the copy's place. A cut takes no
//! object's mutation lock. A row a writer left with no turn of the holder
//! is in no first copy until a cut laid whole lays it.

use parking_lot::MutexGuard;
use smg_worker::registry::Generation;

use super::{Laid, Objects};
use crate::{
    engine::view::WorkerId,
    store::{memory::Told, MemoryStore, StoreCheckpoint, WorkerRow},
};

/// The row `laid` stands for, with the status the journal told of it.
/// `None` where the journal told none: the row left the store's table.
fn told_row(laid: &Laid, told: &Told) -> Option<WorkerRow> {
    let mut row = laid.remainder.row(laid.object.as_ref());
    row.status = told.status(row.id)?;
    Some(row)
}

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(dead_code, reason = "a controller's doors read through the holder")
)]
impl<G: Generation> Objects<G> {
    /// The record under the row id `id`, as a door reads it: in the
    /// holder's turn, off the object. `None` where no object stands for
    /// the row.
    pub(crate) fn record(&self, id: WorkerId) -> Option<WorkerRow> {
        let _turn = self.turn.lock();
        self.row(id)
    }

    /// The record under the registration key `key`, as [`Self::record`]
    /// reads one.
    pub(crate) fn record_under(&self, key: &str) -> Option<WorkerRow> {
        let _turn = self.turn.lock();
        self.row_under(key)
    }

    /// Every record whose facts name `engine`, as [`Self::record`] reads
    /// one: its plain row, or each rank that stands. In no order. Asked
    /// once a registration at a door: no decision reads it.
    pub(crate) fn records_of(&self, engine: &str) -> Vec<WorkerRow> {
        let _turn = self.turn.lock();
        let dialing: Vec<WorkerId> = {
            let held = self.held.read();
            let ranked = held.ranks.get(engine).into_iter().flatten();
            ranked.map(|&(_, row)| row).collect()
        };
        dialing
            .into_iter()
            .filter_map(|row| self.row(row))
            .filter(|row| row.facts.engine == engine)
            .collect()
    }

    /// Every row at one place of the journal, cut from the worker objects:
    /// what a watch starts from, what a standby is restored from and what
    /// an operator reads whole. The rows come by registration key.
    ///
    /// In the holder's turn. Every
    /// writer of a row's facts holds that turn from its decision to the
    /// journal's telling, so the objects state each row the journal told
    /// and none it did not. A row's status is the one the journal told,
    /// read with the place and the epoch in one hold of the store's
    /// membership lock ([`MemoryStore::told`]). A status tell takes no
    /// turn of the holder: it changes no fact, and the copy holds its
    /// status where the place holds its event.
    ///
    /// The store's lock is held for the place and the statuses, and the
    /// holder's turn for the rows: a door's write and a standby's applier
    /// wait for the copy there, and no decision does.
    pub(crate) fn checkpoint(&self, store: &MemoryStore) -> StoreCheckpoint {
        let (turn, told) = self.told_turn(store);
        let mut rows: Vec<WorkerRow> = {
            let held = self.held.read();
            let laid = held.rows.values();
            laid.filter_map(|laid| told_row(laid, &told)).collect()
        };
        // Sorted after the turn: a writer waits for the copy alone.
        drop(turn);
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        StoreCheckpoint {
            at: told.at,
            epoch: told.epoch,
            rows,
        }
    }

    /// The first `limit` rows in key order of the ones that serve one of
    /// `models`, with the count that serve one; no model asked is every
    /// row. What an operator's read that tells a few rows takes, where
    /// [`Self::checkpoint`] cuts every row. Read as that cut is: in the
    /// holder's turn, each row with its told status. Every object's
    /// models are asked, and only the rows told are built. A model matches
    /// by the rule a watch and a shared snapshot filter by, an equal id
    /// (`serves`, in the operators' snapshot rules): the two change
    /// together. The rows name no write of the journal: no reader starts
    /// a tail from them.
    pub(crate) fn rows_serving(
        &self,
        store: &MemoryStore,
        models: &[String],
        limit: usize,
    ) -> (Vec<WorkerRow>, usize) {
        let mut asked: Vec<&str> = models.iter().map(String::as_str).collect();
        asked.sort_unstable();
        let serves = |laid: &&Laid| {
            let serves = |id: &str| asked.binary_search(&id).is_ok();
            asked.is_empty() || laid.object.any_model(&mut |card| serves(&card.id))
        };
        let (turn, told) = self.told_turn(store);
        let held = self.held.read();
        let mut serving: Vec<&Laid> = held.rows.values().filter(serves).collect();
        let count = serving.len();
        if limit < count {
            if let Some(last) = limit.checked_sub(1) {
                serving.select_nth_unstable_by(last, |a, b| a.object.url().cmp(b.object.url()));
            }
            serving.truncate(limit);
        }
        let mut rows: Vec<WorkerRow> = serving
            .into_iter()
            .filter_map(|laid| told_row(laid, &told))
            .collect();
        // Sorted after the turn, as the cut's rows are.
        drop(held);
        drop(turn);
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (rows, count)
    }

    /// The holder's turn for a first copy, with the statuses the journal
    /// told: one read of `store`, inside the turn. Every writer of a row's
    /// facts holds the turn, so the objects state each row the journal
    /// told at the place this read takes, and no row of the store's table
    /// is read. A status tell, a generation's apply and a term's start
    /// take no turn: none changes a row's facts, and the told read takes
    /// the place in the hold that takes the statuses. A change handed to
    /// the store around the holder ([`MemoryStore::follow`]) takes none
    /// either: no product code hands one so. A row written so is in no
    /// copy until a cut laid whole lays it.
    fn told_turn(&self, store: &MemoryStore) -> (MutexGuard<'_, ()>, Told) {
        let turn = self.turn.lock();
        let told = store.told();
        (turn, told)
    }

    /// The row `id` as its object and what stands beside it state it. It
    /// takes no turn: a door's read and the holder's write ask it inside
    /// theirs.
    pub(super) fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.row_of(&self.object(id)?)
    }

    /// The row under `key`, as [`Self::row`] reads one. Under the exact
    /// key, as the store's table finds a row: the registry's other lookup
    /// also finds a bare address under the form with its scheme, and that
    /// is another key.
    pub(super) fn row_under(&self, key: &str) -> Option<WorkerRow> {
        let object = self.registry.get_id_by_url_or_reservation(key)?;
        self.row_of(&object)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/read/tests.rs"]
mod tests;
