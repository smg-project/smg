//! A pair's decision by each side's draw, as the records' own
//! ([`PoolRecords::decide_pair_drawn`]), with the inputs of one side
//! ([`DrawnRecords`]).
//!
//! The decision reads both lists' fixed rows and writes a row as a
//! single's draw does ([`Singly`]): private items of the parent module,
//! which this child reads as they are. Nothing of the parent is widened
//! for it.

use std::sync::Arc;

use smg_worker::{identity::basic::Worker, runtime::published_word};

use super::{Live, PoolRecords, Singly};
use crate::engine::{
    decision::Decision,
    framework::{Engine, Scratch},
    view::{Candidate, Constraints, RequestFacts},
};

/// One side of a pair whose records are
/// written as its draw asks about them: its list's records, the workers
/// the list was made from, how they are read, and the stamp a pair writes
/// on a record beside a single's.
pub(crate) type DrawnRecords<'a> = (
    &'a PoolRecords,
    &'a [Arc<dyn Worker>],
    &'a Live<'a>,
    &'a dyn Fn(&mut Candidate),
);

impl PoolRecords {
    /// A pair's decision where each side's
    /// records are written as its draw asks about them ([`Singly`], the
    /// writer of a single's draw), two a side for a pick of two, whatever the
    /// lists hold. Asked where both lists stand and neither is read at rest
    /// or holds an engine's ranks, under the engine's own question
    /// ([`Engine::pair_drawn_for`]); the caller says that the request states
    /// nothing beside its lists, and answers two questions of the two lists:
    /// whether its host's admission refuses no row (`admits`, asked once at
    /// most: it may read its host) and whether every row of a side pairs with
    /// every row of the other (`pairs`). A side's stamp is called behind the
    /// yes of `admits` alone. `None` leaves the decision to the whole pass,
    /// which writes every record itself.
    #[cfg(feature = "in-process")]
    pub(crate) fn decide_pair_drawn(
        engine: &Engine,
        facts: &RequestFacts<'_>,
        cx: &Constraints<'_>,
        [prefills, decodes]: [DrawnRecords<'_>; 2],
        (admits, pairs): (&mut dyn FnMut() -> bool, &dyn Fn() -> bool),
        rows: [&mut Vec<Candidate>; 2],
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let (prefills, prefill_workers, prefill_live, prefill_stamp) = prefills;
        let (decodes, decode_workers, decode_live, decode_stamp) = decodes;
        if prefill_live.rest || prefills.has_ranks() || decodes.has_ranks() {
            return None;
        }
        let unsaid = prefills.facts(prefill_live, false);
        let drawn = engine.pair_drawn_for(facts, &unsaid, [prefills.len(), decodes.len()], cx)?;
        // The lists are asked first, by one load each where no published state
        // moved: a list with a worker that is not available leaves here before
        // the host is asked, and its whole pass asks the host itself.
        let word = published_word();
        if !(prefills.stands(prefill_workers, word) && decodes.stands(decode_workers, word)) {
            return None;
        }
        // The host's admission is asked behind the engine's own question and
        // the lists', so that a pair no draw is asked of pays none of it, and
        // ahead of the classes: a pair with a refused row leaves here.
        if !admits() {
            return None;
        }
        // The classes are asked last: the question reads every row's class.
        if !pairs() {
            return None;
        }
        let [prefill_rows, decode_rows] = rows;
        if prefill_rows.len() < prefills.len() {
            prefill_rows.resize(prefills.len(), Candidate::default());
        }
        if decode_rows.len() < decodes.len() {
            decode_rows.resize(decodes.len(), Candidate::default());
        }
        let prefill_singly = Singly::new(prefills, prefill_workers, prefill_live);
        let decode_singly = Singly::new(decodes, decode_workers, decode_live);
        let mut write_prefill = |fixed: &Candidate, row: &mut Candidate| {
            prefill_singly.write(fixed, row);
            prefill_stamp(row);
        };
        let mut write_decode = |fixed: &Candidate, row: &mut Candidate| {
            decode_singly.write(fixed, row);
            decode_stamp(row);
        };
        let decision = engine.decide_pair_drawn(
            drawn,
            facts,
            &unsaid,
            (
                &prefills.fixed,
                &mut prefill_rows[..prefills.len()],
                &mut write_prefill,
            ),
            (
                &decodes.fixed,
                &mut decode_rows[..decodes.len()],
                &mut write_decode,
            ),
            cx,
            scratch,
        );
        prefill_singly.paid();
        decode_singly.paid();
        decision
    }
}
