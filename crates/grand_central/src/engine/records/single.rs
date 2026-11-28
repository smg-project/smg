//! One decision of a single request over a list's records ([`decide`]),
//! its two draws, methods of the records ([`PoolRecords::draw`],
//! [`PoolRecords::claim`]), and the scorers of a pick made alone, asked of
//! its one row (`PoolRecords::score_picked`).
//!
//! The decision takes its buffers by the parent's borrow ([`with_buffers`])
//! and each draw writes its rows as a read of one worker does ([`Singly`]),
//! over the list's fixed rows: private items of the parent module, which
//! this child reads as they are. Nothing of the parent is widened for it.

use std::sync::Arc;

use smg_worker::{identity::basic::Worker, runtime::published_word};

use super::{vetoes, with_buffers, Buffers, Live, PoolRecords, Singly};
use crate::engine::{
    decision::Decision,
    framework::{Drawn, Engine, Scratch},
    view::{Candidate, Constraints, PoolFacts, RequestFacts},
};

impl PoolRecords {
    /// [`Self::read`] for the scorers of a profile whose picker picked
    /// first ([`Engine::score_picked`]): the record of the worker at
    /// `position` is written at its own place in the calling thread's
    /// rows, as a draw writes the row it lands on ([`Self::draw`]), and
    /// each scorer of the profile at `profile` is asked of that one place.
    /// The look-up the write made is paid as a draw's are.
    #[cfg(feature = "in-process")]
    pub(crate) fn score_picked(
        &self,
        engine: &Engine,
        profile: u16,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        facts: &RequestFacts<'_>,
        position: u32,
    ) {
        with_buffers(|buffers| {
            let rows = &mut buffers.rows;
            if rows.len() < self.len() {
                rows.resize(self.len(), Candidate::default());
            }
            let rows = &mut rows[..self.len()];
            let singly = Singly::new(&self.reported, workers, live);
            let mut write = |fixed: &Candidate, row: &mut Candidate| singly.write(fixed, row);
            engine.score_picked(profile, facts, &self.fixed, rows, &mut write, position);
            self.paid(workers, live, singly.looked_up());
        });
    }

    /// [`decide`]'s plain draw over the list ([`Engine::decide_drawn`]):
    /// the engine draws over the list's fixed rows, and a worker's record
    /// is written when the draw asks about the worker, as [`Self::read`]
    /// writes it. The look-ups the draw made are paid toward the
    /// conversion of the list's reports at its end ([`super::LookUps`]).
    ///
    /// `rows` is the buffer the decision lends: it grows to the list's
    /// length here where it is shorter.
    // In line at its one caller: a plain draw's decision pays no call for
    // it.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "one decision's inputs and the calling thread's two buffers"
    )]
    fn draw(
        &self,
        engine: &Engine,
        drawn: Drawn,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        facts: &RequestFacts<'_>,
        unsaid: &PoolFacts<'_>,
        cx: &Constraints<'_>,
        rows: &mut Vec<Candidate>,
        scratch: &mut Scratch,
    ) -> Decision {
        if rows.len() < self.len() {
            rows.resize(self.len(), Candidate::default());
        }
        let rows = &mut rows[..self.len()];
        let singly = Singly::new(&self.reported, workers, live);
        let mut write = |fixed: &Candidate, row: &mut Candidate| singly.write(fixed, row);
        let decision = engine.decide_drawn(
            drawn,
            facts,
            unsaid,
            &self.fixed,
            rows,
            &mut write,
            cx,
            scratch,
        );
        self.paid(workers, live, singly.looked_up());
        decision
    }

    /// [`decide`]'s draw for a key with no pin under a sessions rule
    /// ([`Engine::decide_claimed`]), as the plain draw runs
    /// ([`Self::draw`]): a worker's record is written when the draw asks
    /// about the worker, and the look-ups are paid at its end. `None`
    /// where the whole decision answers, as it does over a list where two
    /// rows carry one id ([`Self::place_of`]): the rule's cap is asked of
    /// every row of an id, and a draw keeps a few rows.
    ///
    /// Out of line: a draw that claims nothing holds none of this.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "one decision's inputs and the calling thread's two buffers"
    )]
    fn claim(
        &self,
        engine: &Engine,
        drawn: Drawn,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        facts: &RequestFacts<'_>,
        unsaid: &PoolFacts<'_>,
        cx: &Constraints<'_>,
        rows: &mut Vec<Candidate>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        if self.places.is_empty() {
            return None;
        }
        if rows.len() < self.len() {
            rows.resize(self.len(), Candidate::default());
        }
        let rows = &mut rows[..self.len()];
        let singly = Singly::new(&self.reported, workers, live);
        let mut write = |fixed: &Candidate, row: &mut Candidate| singly.write(fixed, row);
        let claimed = engine.decide_claimed(
            drawn,
            facts,
            unsaid,
            &self.fixed,
            rows,
            &mut write,
            cx,
            scratch,
        );
        self.paid(workers, live, singly.looked_up());
        claimed
    }
}

/// One decision of a single request over a gateway's own data: the records
/// of `workers`, the list `pool` was made from, read as `live` says and put
/// to the engine's whole decision ([`Engine::decide`]), which runs the
/// binding's rules and its profile's stages as it does over any records.
///
/// The records written are the ones the decision reads. Where the
/// request's profile draws first and its picker is one of the crate's own,
/// the picker draws over the list and a worker's record is written when
/// the draw asks about the worker: two workers for a pick of two, whatever
/// the pool's size. The rows the request names join them. The draws over a
/// list convert its reports once they looked up as many as the list holds,
/// and copy the rows from there. Every other
/// decision is handed every worker's record ([`PoolRecords::write`]), and
/// so is a draw that kept no worker. Both give the pick the whole write
/// gives: a stage that answers row by row reads no other row.
///
/// `facts` are the request's, as its host made them. The decision's instant
/// is written into them here ([`RequestFacts::now_micros`]): the one `live`
/// was made at ([`Live::now_micros`]).
///
/// A target of the decision names its worker by position, the worker's
/// place in `workers`, and by id ([`PoolRecords::ids`]).
///
/// Where `live` carries the list's own word, as the engine's decision of a
/// single request over one list hands it, the pool's facts lend it to the
/// decision ([`PoolFacts::word`]): a picker that takes turns counts a
/// draw's turn in it.
///
/// A standing veto `cx` states beside the records
/// ([`Constraints::stalled`]) is written into them where the request's
/// profile shows its rows to a plugin: the stalled bit, and the breaker
/// bit where the veto holds. The engine declines such a profile over a
/// mask beside its rows.
///
/// The records and the engine's scratch are the calling thread's own.
pub fn decide(
    engine: &Engine,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
) -> Decision {
    facts.now_micros = live.now_micros;
    let facts = &*facts;
    let decide = |buffers: &mut Buffers| {
        if live.unread {
            buffers.scratch.unread(true);
        }
        let Buffers {
            rows,
            whole,
            stands,
            scratch,
        } = buffers;
        // No record says yet whether it carries a report: a draw reads that
        // fact of no pool.
        let unsaid = pool.facts(live, false);
        // Under the draw's floor of every profile ([`DRAW_FROM`]) a draw is
        // asked only where no filter refuses it a row: of a request that
        // names no worker tried or excluded, is held to no places and
        // stands under no veto of its host, over a list whose every worker
        // publishes its routing state available ([`PoolRecords::stands`]):
        // one load where no published state moved. The whole pass draws
        // among the rows it keeps and a draw among all of them, so the two
        // hand out one pick of a seed only where every row is kept. Every
        // other decision under that floor is written whole and read by the
        // pass: no refused row costs it a draw, and its pick is the pass's
        // own. The request and the list are asked before the draw is: a
        // decision the pass makes resolves its request's binding once.
        // A request with a session key is written whole under that floor:
        // under a sessions rule a profile that draws two is asked no draw
        // of a key ([`Engine::drawn_for`]'s last test), and the question
        // would resolve the request's binding a second time. So is a list
        // that holds an engine's ranks ([`PoolRecords::has_ranks`]): a
        // profile that decides over the engines is asked no draw of it.
        let drawn = if engine.asks_standing(pool.len())
            && !(facts.tried.is_empty()
                && cx.excluded.is_empty()
                && cx.places.is_none()
                && cx.stalled.is_empty()
                && facts.session_key.is_none()
                && !pool.has_ranks()
                && pool.stands(workers, published_word()))
        {
            None
        } else {
            engine.drawn_for(facts, &unsaid, pool.len())
        };
        if let Some(drawn) = drawn.filter(|drawn| !drawn.claims()) {
            return pool.draw(
                engine, drawn, workers, live, facts, &unsaid, cx, rows, scratch,
            );
        } else if let Some(decision) = drawn.and_then(|drawn| {
            // A key with no pin takes the list's turn, and the turn's
            // worker is claimed for it. Where the whole decision answers
            // by another way it is asked below, over every record.
            pool.claim(
                engine, drawn, workers, live, facts, &unsaid, cx, rows, scratch,
            )
        }) {
            return decision;
        }
        let (rows, pool_facts) = pool.write_over::<true, false>(workers, live, whole, stands);
        if engine.turns_a_set() {
            scratch.whole(pool.whole_rows(rows));
        }
        let written;
        let cx = if cx.stalled.is_empty() {
            cx
        } else {
            written = vetoes::written(engine, facts.model, facts.phase, rows, cx);
            &written
        };
        engine.decide(facts, &pool_facts, rows, cx, scratch)
    };
    with_buffers(decide)
}
