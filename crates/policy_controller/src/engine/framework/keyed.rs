//! A keyed decision over rows read one at a time.
//!
//! A decision that reads its pool whole is handed every row, written for it
//! ([`Engine::decide`]). A host that keeps its workers where it always kept
//! them pays for each row it writes, and a keyed request asks about few of
//! them: its key's pin is one row, and a ring's walk is the key's owner and
//! the workers behind it. Here such a request is decided over rows its host
//! reads when the decision asks for one ([`PoolRows`]), and no other row is
//! read.
//!
//! The answer is the whole decision's: the same targets in the same order,
//! the same session verdict and the same mark of a lifted veto. Its trace
//! counts no survivors, as a draw's counts none. Wherever the rows read
//! cannot say what the whole decision would answer, nothing is decided here
//! and the host asks the whole decision, which reads every row. No request
//! is refused here and none is shed.

use super::{
    places, shadow_index, slot, trace_ordered, vetoed, Engine, Entry, Filter, Marks, Plan, Profile,
};
use crate::{
    engine::{
        decision::{Decision, Pick, Targets, Trace, MAX_TARGETS},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{Candidate, Constraints, PoolFacts, RequestFacts, WorkerId},
    },
    stages::{filters::NotOverloaded, pickers::target_of},
};

/// A pool's rows as a keyed decision reads them: one at a time, by place,
/// from wherever their host keeps them.
///
/// A row read here is the row a whole decision of the same instant is handed
/// at that place: its fixed fields, its health and its counts as they stand,
/// and `position` its place.
pub trait PoolRows {
    /// How many rows the pool holds.
    fn len(&self) -> usize;

    /// Whether the pool holds no row.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The row at `place`; `None` past the pool's end.
    fn read(&self, place: u32) -> Option<Candidate>;

    /// The place of the row whose id is `id`; `None` where the pool holds no
    /// such row. A session's pin names its worker by id. In a pool where
    /// two rows carry one id a pin names neither of them: such a pool
    /// answers `None` for every id, and its pins are followed by the whole
    /// decision.
    fn place_of(&self, id: WorkerId) -> Option<u32>;

    /// The generation of the document the rows are read under: the trace's
    /// ([`Trace::generation`]).
    fn generation(&self) -> u64;
}

impl Engine {
    /// The decision of a keyed request of the single plan, over rows read
    /// one at a time: [`Self::decide`]'s answer for it, at the cost of the
    /// rows the key names.
    ///
    /// Two requests are decided here:
    ///
    /// - **A session's pin.** Under a binding's sessions rule, a request
    ///   whose key holds a pin goes to the pin's worker: the first of the
    ///   key's pins whose row every filter of the profile keeps, where that
    ///   row is under the rule's cap. One row is read for a pin, whatever
    ///   the profile's scorers are: the whole decision takes a standing
    ///   pin behind its filters and asks no scorer.
    /// - **A ring's walk.** Under a profile whose one scorer places a keyed
    ///   request by itself ([`Scorer::own_by_place`]), the request goes to
    ///   the workers its key reaches on the ring its caller lent
    ///   ([`RequestFacts::ring`]), the owner first. A key the whole
    ///   decision keeps on its worker over the overload veto is walked
    ///   with that veto read as lifted: the binding steers, the request
    ///   carries a session key and the profile is of the one shape that
    ///   keeps a key.
    ///
    /// A session key under a sessions rule is the rule's alone: with no pin
    /// that stands it is left to the whole decision, which may claim it,
    /// and no ring is walked for it.
    ///
    /// `None` wherever the whole decision would answer from more than these
    /// rows, and the host then asks it ([`Self::decide`]): a key with no pin
    /// that stands, a pin at its cap, a request that is lent no ring or
    /// states no place on it, a walk whose scorer says its rule ran out
    /// (the stock ring's does with a row of the pool not asked, once eight
    /// rows were refused or its ring ended short of four kept), a kept key
    /// whose every row read is over its thresholds (the whole decision
    /// says whether the veto left the pool nobody), a binding with a shadow,
    /// a profile that opens a scope or draws, a profile whose filters
    /// cannot be asked of one row alone ([`Filter::row_wise`]), a request
    /// that carries an affinity column, a request its host holds to places
    /// of the pool ([`Constraints::places`]), and every plan but the single
    /// one.
    ///
    /// A row is read here by its place, a pin's and a walk's alike,
    /// whatever places the request may take. The whole decision opens its
    /// set with those places, so a request held to any is its to answer.
    ///
    /// A pick made here states no holding, as [`Self::decide`]'s states
    /// none. A host that feeds a scorer the request's holders, and keeps
    /// for each pick the worker a prompt was sent to, does both around the
    /// whole decision. For a pin followed here it asks no scorer, and what
    /// it keeps is by the rule's own word of a request it places
    /// (`rule_placed`): nothing where the rule keeps the request from the
    /// profile, as a gateway's own sticky map keeps a pinned request from
    /// its policy, and where the request is the profile's own, what it
    /// keeps behind the whole decision's pick.
    ///
    /// The session table is read as the whole decision reads it: a pin
    /// followed renews its key's idle time, and a request not decided here
    /// writes nothing.
    ///
    /// The rows are taken for regular workers': a key keeps its worker over
    /// the overload veto among those alone, and the rows read here do not
    /// say the pool's roles. A host whose pool holds a worker of another
    /// role asks the whole decision.
    ///
    /// [`Scorer::own_by_place`]: super::Scorer::own_by_place
    pub fn decide_keyed<R: PoolRows>(
        &self,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
    ) -> Option<Decision> {
        self.decide_keyed_among(facts, rows, cx, true)
    }

    /// [`Self::decide_keyed`] for a host that says whether every row of the
    /// pool is a regular worker's (`regular`): over any other pool the
    /// key's walk reads the overload veto as it stands.
    pub(crate) fn decide_keyed_among<R: PoolRows>(
        &self,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
        regular: bool,
    ) -> Option<Decision> {
        // No binding of the phase pins a key or walks a ring: one test.
        if !self.keyed[slot(facts.phase)] {
            return None;
        }
        let (entry, index, profile) = self.resolve(facts.model, facts.phase)?;
        let Entry {
            binding,
            placing,
            vetoed,
            sides_placed: _,
        } = entry;
        if binding.plan != Plan::Single {
            return None;
        }
        // A column of another length is refused by the whole decision. It
        // is asked behind the binding: beside the first test the two fold
        // into one branch, and every request then reads its column's length.
        if !facts.affinity.is_empty() {
            return None;
        }
        let shadow = shadow_index(binding, facts.phase);
        let at = usize::from(index);
        let marks = self.marks[at];
        // A shadow reads the rows its live decision was handed, and a
        // stage's scope is opened over them: the whole decision's, each.
        if shadow.is_some() || marks.scoped() {
            return None;
        }
        // A request held to places is the whole decision's, which opens
        // its set with them: no row is read by its place here.
        if places::stated(cx) {
            return None;
        }
        if let Some(rule) = &binding.sessions {
            if !facts.session_key.is_none() {
                return self.follow_pin(index, rule, facts, rows, cx);
            }
        }
        // No rule governs the key. A picker that leaves the placement to
        // its rule's assignment places over every row, and so does a
        // profile whose scorer places nothing by itself. A profile that
        // draws is the draw's, over rows or over engines. A request that is
        // lent no ring, or states no place on it, is placed over every row.
        let draws = marks.first() & (Marks::DRAWS_FIRST | Marks::JOINED | Marks::ENGINES) != 0;
        let unplaced = facts.ring.is_none() || facts.ring_positions.is_empty();
        if marks.by_assignment() || !marks.owns() || draws || unplaced {
            return None;
        }
        // The kept key's form: its walk reads the overload veto as lifted.
        let kept =
            marks.keyed() && regular && vetoed::keeps_its_key(binding, *placing, *vetoed, facts);
        self.walk_ring(index, profile, facts, rows, cx, kept)
    }

    /// A keyed request under a sessions rule, where one of its key's pins
    /// stands: the pick of that pin, as the rule's whole decision makes it
    /// ([`Self::decide`]). That decision narrows the pool by the profile's
    /// filters and takes the first of the key's pins among the rows kept. A
    /// row every filter keeps is among them, and while one is, no soft
    /// filter steps aside: so the first pin whose own row every filter
    /// keeps is that pin.
    ///
    /// The pins are read first with nothing written. The read that renews
    /// the key's idle time follows once a pin's row is kept, where the
    /// whole decision makes it too: behind filters that kept a row. Pins
    /// that moved between the two reads are the whole decision's to read
    /// again.
    fn follow_pin<R: PoolRows>(
        &self,
        index: u16,
        rule: &SessionRule,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
    ) -> Option<Decision> {
        let key = facts.session();
        let test = self.row_test_at(index, facts, cx)?;
        let pins = self.sessions.pins_at(key, facts.now_micros);
        let pinned = pins
            .iter()
            .filter(|pin| **pin != WorkerId(0))
            .filter_map(|pin| rows.read(rows.place_of(*pin)?))
            .find(|row| test.keeps(row))?;
        // At its cap a pin puts the request to the stages again, over every
        // row the filters keep.
        if Self::at_cap(rule, &pinned) {
            return None;
        }
        if self.sessions.lookup(key, facts.now_micros, rule) != pins {
            return None;
        }
        let mut targets = Targets::new();
        targets.push(target_of(&pinned));
        // The rule's pin placed the request, as in the whole decision, and
        // the pick carries the rule's word of it on both paths. The engine
        // over worker lists reads that word behind this entry's answer as
        // behind the whole decision's: by it a followed pin is told to the
        // stages or kept from them, and its prompt handed to the record or
        // not, the same on both.
        targets.by_rule(rule.rule_placed);
        Some(Decision::Pick(Pick {
            targets,
            trace: trace(index, facts, rows.generation(), [(pinned.id, 1.0)]),
            session_verdict: SessionVerdict::Pinned,
        }))
    }

    /// A keyed request of a profile whose one scorer places it by itself:
    /// the scorer's walk over the workers it names, each found in the pool
    /// by its id, every filter of the profile asked about each row read,
    /// and the rows kept handed out in the walk's order. The trace scores
    /// each a step under the one before it, by the scorer's share of the
    /// profile's weights: the whole decision's scores of the stock ring's
    /// walk, to the bit. How many rows are read is the scorer's rule: the
    /// stock ring's walk reads eleven at most.
    ///
    /// `kept` is the form of a key kept on its worker over the overload
    /// veto ([`Engine::keep`]): the filters read that veto as lifted. The
    /// whole decision then marks a pick the lift alone made possible, which
    /// it knows by a pass over every row. A row handed out here that stands
    /// under the veto as it is says the pool was not emptied, and the pick
    /// carries no mark. Where none does, the whole decision answers.
    fn walk_ring<R: PoolRows>(
        &self,
        index: u16,
        profile: &Profile,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
        kept: bool,
    ) -> Option<Decision> {
        let weighted = profile.scorers.first()?;
        let stage = &weighted.scorer;
        // The stage's share of the profile's weights, as the whole pass
        // folds it into a score: a weight's share of itself need not be
        // one.
        let share = weighted.sole_share();
        let lifted = Constraints {
            overload_lifted: true,
            ..*cx
        };
        let test = self.row_test_at(index, facts, if kept { &lifted } else { cx })?;
        let mut targets = Targets::new();
        let mut steps = [(WorkerId(0), 0.0); MAX_TARGETS];
        // Whether a row handed out stands under the veto as it is.
        let mut stands = false;
        let mut keep = |place: u32| {
            let Some(row) = rows.read(place) else {
                return false;
            };
            if !test.keeps(&row) {
                return false;
            }
            if let Some(step) = steps.get_mut(targets.len()) {
                let rank = targets.len() as f32 / MAX_TARGETS as f32;
                *step = (row.id, share * (1.0 - rank));
            }
            stands |= NotOverloaded.keep(facts, &row, cx);
            targets.push(target_of(&row));
            true
        };
        let place_of = |id| rows.place_of(id);
        let handed = stage.own_by_place(facts, rows.len(), &place_of, &mut keep);
        if handed == 0 || handed != targets.len() || (kept && !stands) {
            return None;
        }
        Some(Decision::Pick(Pick {
            targets,
            trace: trace(
                index,
                facts,
                rows.generation(),
                steps.into_iter().take(handed),
            ),
            session_verdict: SessionVerdict::None,
        }))
    }
}

/// The trace of a decision made over the rows it read alone
/// ([`trace_ordered`]): `best` names the rows handed out, best first, each
/// with its score, and no survivor is counted, as behind a draw. The pool's
/// facts such a decision holds are the generation its rows are read under.
fn trace(
    profile: u16,
    facts: &RequestFacts<'_>,
    generation: u64,
    best: impl IntoIterator<Item = (WorkerId, f32)>,
) -> Trace {
    let pool = PoolFacts {
        generation,
        ..PoolFacts::default()
    };
    trace_ordered(profile, facts, &pool, best)
}

#[cfg(test)]
mod tests;
