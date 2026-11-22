//! A keyed decision over rows read one at a time.
//!
//! A decision that reads its pool whole is handed every row, written for it
//! ([`Engine::decide`]). A host that keeps its workers where it always kept
//! them pays for each row it writes, and a keyed request asks about few of
//! them: its key's pin is one row, and a ring's walk is the key's owner and
//! the members behind it. Here such a request is decided over rows its host
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
    shadow_index, slot, trace_ordered, vetoed, Engine, Entry, Filter, Marks, Plan, Profile,
};
use crate::{
    engine::{
        decision::{Decision, Pick, Targets, Trace, MAX_TARGETS},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{Candidate, Constraints, PoolFacts, RequestFacts, WorkerId},
    },
    policies::consistent_hashing::ListRing,
    stages::{filters::NotOverloaded, pickers::target_of},
};

/// A pool's rows as a keyed decision reads them: one at a time, by place,
/// from wherever their host keeps them.
///
/// A row read here is the row a whole decision of the same instant is handed
/// at that place: its fixed fields, its health and its counts as they stand,
/// and `position` its place.
///
/// `'p` is how long the pool's list stands for the decision: the ring it
/// lends ([`Self::ring`]) is borrowed that long, whatever holds the rows'
/// reader.
pub trait PoolRows<'p> {
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

    /// The host's own hash ring as this pool's list reads it, where the
    /// host keeps the rings its workers are placed on: asked for a request
    /// whose binding places by a ring, and for no other. It names this
    /// pool's rows by their places and no other list's
    /// ([`RequestFacts::ring`]). `None` where the host keeps none, and the
    /// ring stage then places by its own ring, over every row; `None` too
    /// where the host keeps the rings and holds none for the request yet
    /// ([`Self::keeps_a_ring`]).
    fn ring(&self) -> Option<&'p ListRing<'p>>;

    /// Whether the host keeps the rings its workers are placed on, filled
    /// off the path of a request. Such a host that hands none
    /// ([`Self::ring`]) holds none for the request yet, and a request that
    /// reads a ring there is placed by the host's rule for a request with
    /// no ring ([`RequestFacts::ring_awaited`]): no decision builds a ring,
    /// and none waits for one.
    fn keeps_a_ring(&self) -> bool {
        false
    }
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
    ///   row is under the rule's cap. One row is read for a pin.
    /// - **A ring's walk.** Under a profile whose one scorer places a keyed
    ///   request by itself ([`Scorer::own_by_place`]), the request goes to
    ///   its key's owner and the members behind it, over the ring of the
    ///   pool's list. Where the binding steers and the request carries a
    ///   session key, the walk reads the overload veto as lifted, as the
    ///   whole decision does for a kept key.
    ///
    /// A request of the single plan whose profile reads a ring, or whose
    /// shadow's does, is lent its host's as the pool reads it
    /// ([`PoolRows::ring`]) where its facts carry none: it is written into
    /// `facts` ([`RequestFacts::ring`]) and stands there for the whole
    /// decision and the shadow that follow a `None`, which then read that
    /// ring too.
    ///
    /// `None` wherever the whole decision would answer from more than these
    /// rows, and the host then asks it ([`Self::decide`]): a key with no pin
    /// that stands, a pin at its cap, a walk that met too many refused
    /// rows, a kept key whose every row read is over its thresholds (the
    /// whole decision says whether the veto left the pool nobody), a
    /// binding with a shadow, a profile that opens a scope or draws, a
    /// profile whose filters cannot be asked of one row alone
    /// ([`Filter::row_wise`]) or whose scorer states what its pick holds
    /// of the request ([`Self::decide_stated`]), a request that carries an
    /// affinity column, and every plan but the single one.
    ///
    /// The session table is read as the whole decision reads it: a pin
    /// followed renews its key's idle time, and a request not decided here
    /// writes nothing.
    ///
    /// [`Scorer::own_by_place`]: super::Scorer::own_by_place
    pub fn decide_keyed<'a, R: PoolRows<'a>>(
        &self,
        facts: &mut RequestFacts<'a>,
        rows: &R,
        cx: &Constraints<'_>,
    ) -> Option<Decision> {
        // No binding of the phase pins a key or reads a ring: one test.
        if !self.keyed[slot(facts.phase)] {
            return None;
        }
        // A column of another length is refused by the whole decision.
        if !facts.affinity.is_empty() {
            return None;
        }
        let (entry, index, profile) = self.resolve(facts.model, facts.phase)?;
        let Entry {
            binding,
            placing,
            vetoed,
        } = entry;
        if binding.plan != Plan::Single {
            return None;
        }
        // The ring a stage of the binding reads, its profile's or its
        // shadow's: the pool's own, from here on and for the decisions
        // that follow.
        let shadow = shadow_index(binding, facts.phase);
        let reads_ring = |profile: u16| {
            let needs = self.profile_needs.get(usize::from(profile));
            needs.is_some_and(|needs| needs.ring)
        };
        if facts.ring.is_none() && (reads_ring(index) || shadow.is_some_and(reads_ring)) {
            facts.ring = rows.ring();
            // A host that keeps the rings and hands none holds none for
            // this request yet: the ring stage places by the host's rule
            // for a request with no ring, in the whole decision that
            // follows.
            facts.ring_awaited = facts.ring.is_none() && rows.keeps_a_ring();
        }
        let at = usize::from(index);
        let marks = self.marks[at];
        // A shadow reads the rows its live decision was handed, a stage's
        // scope is opened over them, and a scorer that states what its
        // pick holds reads them for it: the whole decision's, each one.
        if shadow.is_some() || marks.scoped() || self.stating[at].is_some() {
            return None;
        }
        let facts = &*facts;
        if let Some(rule) = &binding.sessions {
            if !facts.session_key.is_none() {
                return self.follow_pin(index, rule, facts, rows, cx);
            }
        }
        // No rule governs the key. A picker that leaves the placement to
        // its rule's assignment places over every row, and so does a
        // profile whose scorer places nothing by itself. A profile that
        // draws is the draw's, over rows or over engines.
        let draws = marks.first() & (Marks::DRAWS_FIRST | Marks::JOINED | Marks::ENGINES) != 0;
        if marks.by_assignment() || !marks.owns() || draws || facts.ring_key() == 0 {
            return None;
        }
        // The kept key's form: its walk reads the overload veto as lifted.
        let kept = marks.keyed() && vetoed::keeps_its_key(binding, *placing, *vetoed, facts);
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
    fn follow_pin<'p, R: PoolRows<'p>>(
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
        Some(Decision::Pick(Pick {
            targets,
            trace: trace(index, facts, rows.generation(), [(pinned.id, 1.0)]),
            session_verdict: SessionVerdict::Pinned,
        }))
    }

    /// A keyed request of a profile whose one scorer places it by itself:
    /// the scorer's walk over the rows it names by place, every filter of
    /// the profile asked about each row read, and the rows kept handed out
    /// in the walk's order, each a step under the one before it, as the
    /// whole decision's scores of a ring's walk stand.
    ///
    /// `kept` is the form of a key kept on its worker over the overload
    /// veto ([`Engine::keep`]): the filters read that veto as lifted. The
    /// whole decision then marks a pick the lift alone made possible, which
    /// it knows by a pass over every row. A row handed out here that stands
    /// under the veto as it is says the pool was not emptied, and the pick
    /// carries no mark. Where none does, the whole decision answers.
    fn walk_ring<'p, R: PoolRows<'p>>(
        &self,
        index: u16,
        profile: &Profile,
        facts: &RequestFacts<'_>,
        rows: &R,
        cx: &Constraints<'_>,
        kept: bool,
    ) -> Option<Decision> {
        let stage = &profile.scorers.first()?.scorer;
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
                *step = (row.id, 1.0 - targets.len() as f32 / MAX_TARGETS as f32);
            }
            stands |= NotOverloaded.keep(facts, &row, cx);
            targets.push(target_of(&row));
            true
        };
        let mut places = [0; MAX_TARGETS];
        let handed = stage.own_by_place(facts, rows.len(), &mut keep, &mut places);
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
