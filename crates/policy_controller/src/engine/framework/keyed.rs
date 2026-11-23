//! A keyed decision over rows read one at a time.
//!
//! A decision that reads its pool whole is handed every row, written for it
//! ([`Engine::decide`]). A host that keeps its workers where it always kept
//! them pays for each row it writes, and a keyed request asks about few of
//! them: its key's pin is one row. Here such a request is decided over rows
//! its host reads when the decision asks for one ([`PoolRows`]), and no
//! other row is read.
//!
//! The answer is the whole decision's: the same targets in the same order,
//! the same session verdict and the same mark of a lifted veto. Its trace
//! counts no survivors, as a draw's counts none. Wherever the rows read
//! cannot say what the whole decision would answer, nothing is decided here
//! and the host asks the whole decision, which reads every row. No request
//! is refused here and none is shed.

use super::{places, shadow_index, slot, trace_ordered, Engine, Plan};
use crate::{
    engine::{
        decision::{Decision, Pick, Targets, Trace},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{Candidate, Constraints, PoolFacts, RequestFacts, WorkerId},
    },
    stages::pickers::target_of,
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
    /// One request is decided here: under a binding's sessions rule, a
    /// request whose key holds a pin goes to the pin's worker, the first of
    /// the key's pins whose row every filter of the profile keeps, where
    /// that row is under the rule's cap. One row is read for a pin, whatever
    /// the profile's scorers are: the whole decision takes a standing pin
    /// behind its filters and asks no scorer.
    ///
    /// `None` wherever the whole decision would answer from more than these
    /// rows, and the host then asks it ([`Self::decide`]): a key with no pin
    /// that stands, a pin at its cap, a binding with a shadow, a profile
    /// that opens a scope, a profile whose filters cannot be asked of one
    /// row alone ([`Filter::row_wise`]), a request that carries an affinity
    /// column, a request its host holds to places of the pool
    /// ([`Constraints::places`]), and every plan but the single one.
    ///
    /// A pin is read here by its place, whatever places the request may
    /// take. The whole decision opens its set with those places, so a
    /// request held to any is its to answer.
    ///
    /// A pick made here states no holding, as [`Self::decide`]'s states
    /// none. A host that feeds a scorer the request's holders, and keeps
    /// for each pick the worker a prompt was sent to, does both around the
    /// whole decision: for a pin followed here it names no holder and
    /// keeps nothing, as a gateway's own sticky map keeps a pinned request
    /// from its policy.
    ///
    /// The session table is read as the whole decision reads it: a pin
    /// followed renews its key's idle time, and a request not decided here
    /// writes nothing. No decision lent pins of record
    /// ([`Constraints::pins`]) comes here: the entry's one caller, the engine
    /// over worker lists, reads none from its host.
    ///
    /// [`Filter::row_wise`]: super::Filter::row_wise
    pub fn decide_keyed<R: PoolRows>(
        &self,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
    ) -> Option<Decision> {
        // No binding of the phase pins a key: one test.
        if !self.keyed[slot(facts.phase)] {
            return None;
        }
        let (entry, index, _) = self.resolve(facts.model, facts.phase)?;
        let binding = &entry.binding;
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
        // No rule governs the key: the whole decision places it.
        None
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
        if Self::at_cap(rule, facts, &pinned) {
            return None;
        }
        if self.sessions.lookup(key, facts.now_micros, rule) != pins {
            return None;
        }
        let mut targets = Targets::new();
        targets.push(target_of(&pinned));
        // The rule's pin placed the request, as in the whole decision. The
        // engine over worker lists writes no record behind this entry's
        // answer: the word is here so a followed pin says the same on both
        // paths.
        targets.by_rule(rule.rule_placed);
        Some(Decision::Pick(Pick {
            targets,
            trace: trace(index, facts, rows.generation(), [(pinned.id, 1.0)]),
            session_verdict: SessionVerdict::Pinned,
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
