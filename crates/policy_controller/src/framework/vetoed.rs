//! The all-vetoed rule: what a decision does when the overload veto leaves
//! it no worker. Every usable worker of the pool is over its thresholds, so
//! the veto that steers requests away from a busy worker has nowhere to
//! steer them. A binding answers in one of two ways ([`AllVetoed`]): the
//! request goes to the least-vetoed worker, or it is shed. One function
//! holds the rule ([`emptied`]). The filter half of a pass asks it where a
//! hard filter leaves nobody, by the word the pass's set carries, so a pass
//! that keeps a worker holds nothing for the rule and runs none of it.
//!
//! The liveness veto is the second soft veto, and it is lifted last. A pool
//! whose every usable worker is doubted, or over its thresholds too, goes
//! to the fewest requests in flight as well, the workers over their
//! thresholds alone first. A doubt is not an outage. The pick's trace says
//! which veto it lifted ([`Lifted`]).
//!
//! A profile that places a request by its session key reads the same word
//! ahead of its pass ([`keeps_its_key`]): where the binding steers, a keyed
//! request is decided with the veto lifted ([`Engine::keep`]), so the key
//! keeps its worker while that worker is over its thresholds. A sessions
//! rule with an in-flight cap keeps the key only where the host wrote the
//! cap's count ([`RequestFacts::session_counted`]). Under a sessions rule
//! the key's pin is asked first, with the veto as it stands, and the lifted
//! pass places a key no pin takes. Where the veto as it stands leaves the
//! key's pool nobody, the kept pick says the lifted veto in its trace as the
//! rule's own pick does ([`Lifted::Overload`]): a host counts both from the
//! one mark.

use super::{
    engines::run_owned, filter, keep_folded, leads, run_single, single_emptied, trace_of,
    Assignment, Binding, Buffers, Engine, Filter, Picker, Plan, Profile, Scorer, Scratch,
    SessionRule, Set,
};
use crate::{
    builtins::{Best, Healthy, LeastRequests, NotOverloaded},
    decision::{Decision, Lifted, Pick, Targets, Trace},
    sessions::SessionVerdict,
    view::{Candidate, Constraints, Health, PoolFacts, RequestFacts},
};

/// What a binding does with a pool whose every usable worker the overload
/// veto refuses: the document's `overload.shed_when_all_vetoed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum AllVetoed {
    /// The request goes to the least-vetoed worker. The default: a fleet
    /// over its thresholds on every worker is still a fleet, and a refusal
    /// would turn a load signal into an outage.
    #[default]
    Steer,
    /// The request is shed as saturated. The word is the overload veto's
    /// alone: a doubted worker under its thresholds still takes a request
    /// no other worker can.
    Shed,
    /// No rule is asked: a pool a veto empties is shed at the stage that
    /// emptied it. A set's word, never a binding's: a side of a pair and
    /// the rule's own passes carry it ([`Set::shedding`]).
    Unasked,
}

impl Engine {
    /// Makes the binding at `at`, in the spec's order, shed a pool the
    /// overload veto empties ([`AllVetoed::Shed`]). The validator calls it
    /// for a binding whose document says so; every other binding steers.
    /// The index is the validator's own, taken where the binding is built:
    /// one the engine does not hold is a bug, and a debug build refuses it.
    /// A host that builds its bindings itself has no door to the word yet:
    /// such an engine steers.
    pub(crate) fn shed_when_all_vetoed(&mut self, at: usize) {
        let entry = self.bindings.get_mut(at);
        debug_assert!(entry.is_some(), "binding {at} is not built");
        if let Some(entry) = entry {
            entry.vetoed = AllVetoed::Shed;
        }
    }

    /// Marks the profile at `at`, in the spec's order, as placing a request
    /// by its session key. The validator calls it for a profile the plain
    /// ring alone places: its only scorer, with no bound, under the picker
    /// that takes the best score. A decision reads the mark where it reads
    /// the profile's others ([`keeps_its_key`]). A profile whose picker
    /// leaves the placement to its rule takes no mark: no stage of it
    /// places a key. An engine built by hand marks nothing: its host lifts
    /// the veto for the requests it wants kept on their worker
    /// ([`Constraints::overload_lifted`]). The index is the validator's
    /// own: one the engine does not hold is a bug, and a debug build
    /// refuses it.
    pub(crate) fn places_by_key(&mut self, at: usize) {
        let marks = self.marks.get_mut(at);
        debug_assert!(marks.is_some(), "profile {at} is not built");
        if let Some(marks) = marks {
            *marks = marks.with_key();
        }
    }

    /// The decision of a keyed request that keeps its key's worker
    /// ([`keeps_its_key`]). Under no sessions rule the profile runs with the
    /// overload veto lifted, so the key's worker takes the request over its
    /// thresholds. Under a rule the key's pin is asked first, with the veto
    /// as it stands: a pin over its thresholds yields, and the profile then
    /// places the key with the veto lifted. The run is never a draw: the
    /// ring's scorer is shown the whole view, so a profile that holds it
    /// does not draw first. It is the scorer's walk over a pool whose host
    /// lists its engines ([`run_owned`]), the veto lifted in it, and the
    /// profile's whole pass over any other and where the walk gives no
    /// answer: the rows handed are the pass's either way. The lifted run
    /// stands in for the all-vetoed rule's overload tier: a pass that reads
    /// the veto as it stands is put to no rule ([`Set::shedding`]), and a
    /// pool the lifted run leaves nobody is put to the liveness tier alone.
    ///
    /// A pick the lift alone made possible says so: where the veto as it
    /// stands leaves the pool nobody ([`all_vetoed`]), the trace carries the
    /// mark the rule's own pick carries ([`Lifted::Overload`]), under a rule
    /// and under none.
    ///
    /// Out of line: a kept key is the rare decision, and the engine's
    /// decision path keeps its code for every other.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    pub(super) fn keep(
        &self,
        profile: &Profile,
        index: u16,
        rule: Option<&SessionRule>,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        if let Some(rule) = rule {
            return self.decide_session::<true>(
                profile,
                index,
                rule,
                Assignment::Delegate,
                facts,
                pool,
                set,
                cx,
                scratch,
            );
        }
        let lifted = Constraints {
            overload_lifted: true,
            ..*cx
        };
        let buf = &mut scratch.single;
        // The rule's own mark of this decision, read back below: none until
        // one of its tiers picks.
        buf.lifted = Lifted::None;
        // Over a pool whose host lists its engines the ring places the key
        // by its walk, the veto lifted in it as in the pass: the rows it
        // hands are the whole pass's, and `buf` holds the rows it kept.
        let walked = if self.marks[usize::from(index)].owns() {
            run_owned(self, index, facts, pool, set.candidates, &lifted, buf)
        } else {
            None
        };
        let mut decision = match walked {
            Some(decision) => decision,
            None => run_single(profile, index, facts, pool, set, &lifted, buf),
        };
        if let Decision::Pick(pick) = &mut decision {
            // A pick of the rule's own carries its mark already. The rule
            // left that mark in `buf`, and it is read there: read through
            // the pick, it had the decision built aside and copied whole
            // to the caller's place.
            if buf.lifted == Lifted::None && all_vetoed(profile, facts, set, cx, buf) {
                pick.trace.lifted = Lifted::Overload;
            }
        }
        decision
    }
}

/// Whether a keyed request keeps its key's worker over the overload veto:
/// a request of the single plan that carries a session key, under a binding
/// that steers ([`AllVetoed::Steer`]). A decision asks it for a profile the
/// plain ring alone places ([`Engine::places_by_key`]) and then
/// decides the request with the veto lifted ([`Engine::keep`]). The key's
/// worker takes the request while it is over its thresholds, beside free
/// workers too, as a session keeps its owner. A binding that sheds keeps
/// the veto: an owner over its thresholds passes the key on, and a pool
/// over on every worker is shed. So does a sessions rule whose own
/// assignment places a key no pin takes: no stage is asked where that key
/// goes, and nothing is lifted. So does a rule with an in-flight cap under
/// a host that writes no count of the key's requests into its rows
/// ([`RequestFacts::session_counted`]): a kept key leaves its worker at the
/// cap alone, and a cap that reads no count hands no key on.
pub(super) fn keeps_its_key(
    binding: &Binding,
    placing: Assignment,
    rule: AllVetoed,
    facts: &RequestFacts<'_>,
) -> bool {
    rule == AllVetoed::Steer
        && binding.plan == Plan::Single
        && !facts.session_key.is_none()
        && match &binding.sessions {
            None => true,
            Some(sessions) => {
                placing == Assignment::Delegate
                    && (sessions.inflight_cap == 0 || facts.session_counted)
            }
        }
}

/// The profile's filters again with the overload veto lifted, for a keyed
/// request no pin of its sessions rule took under the veto as it stands.
/// Answers for a pool the lifted pass leaves nobody, with the liveness
/// tier's pick or the shed; `buf` holds the lifted pass's survivors
/// otherwise, and the placement runs over them.
pub(super) fn lifted(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Decision> {
    let lifted = Constraints {
        overload_lifted: true,
        ..*cx
    };
    let emptied_by = filter(profile, facts, &lifted, buf, set)?;
    Some(single_emptied(
        index,
        facts,
        pool,
        set.candidates,
        emptied_by,
        buf,
    ))
}

/// Whether the overload veto as it stands leaves a kept key's pool nobody.
/// The lifted pass just placed the key, or the ring's walk did: `buf` holds
/// that pass's survivors, or the rows the walk kept, each one of them, and
/// `cx` is the decision's own, the veto standing in it. A survivor
/// under its thresholds stands without the lift, so the pool was not
/// emptied. The filters run again with the veto standing only over a pool
/// whose survivors are all over: a soft filter of the lifted pass may have
/// passed over a worker under its thresholds. That pass overwrites `buf`,
/// and the caller reads nothing of it afterwards.
fn all_vetoed(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> bool {
    let candidates = set.candidates;
    let stands = |&row: &u32| NotOverloaded.keep(facts, &candidates[row as usize], cx);
    !buf.kept.iter().any(stands) && filter(profile, facts, cx, buf, set.shedding()).is_some()
}

/// Whether the veto of the stage `veto` stood in the pass the stage
/// `emptied_by` ended: that filter is one of the profile's hard filters, at
/// or before that stage. A pool emptied before a veto ran is no nearer a
/// worker with the veto lifted.
fn veto_ran(profile: &Profile, veto: &str, emptied_by: &str) -> bool {
    let mut seen = false;
    for hard in profile.filters.iter().filter(|f| !f.soft()) {
        seen |= hard.name() == veto;
        if hard.name() == emptied_by {
            return seen;
        }
    }
    false
}

/// The rule, asked by the filter half where a hard filter leaves a pass
/// nobody: the stage whose shed stands, with `buf.kept` empty, or the stage
/// `emptied_by` with the rows the rule kept in `buf.kept`, for the plan's
/// pick among them ([`steered`]), and the veto lifted for them in
/// `buf.lifted`.
///
/// Two vetoes are soft, and they are lifted in one order: the overload
/// veto, then the liveness veto. A worker over its thresholds is alive and
/// busy; a doubted one may be gone. Each lift runs the profile's filters
/// once more, and the rows such a pass keeps are the usable ones. So a hard
/// filter listed behind a veto does not hide such a pool, and the answer
/// does not turn on the order a document lists its filters in. The retry's
/// tried list is set aside for those passes, as the gateway's fallback
/// sets it aside: a worker an earlier attempt tried is taken again when it
/// holds the fewest requests. The host's list of workers it just failed to
/// reach stays: the soft filter passes over such a worker while another
/// usable one stands, and takes it as the last resort.
///
/// The overload veto is lifted under [`AllVetoed::Steer`], where it stood
/// in the emptied pass and a row reads as over its thresholds. Under a veto
/// its host lifted none does, and that tier ends there. The liveness veto
/// is lifted next, under either word of the binding, where the doubt alone
/// refuses a row ([`doubted_alone`], [`mask_doubts`]) and the health stage
/// stood in the pass that ended empty. That pass reads the rows with no
/// doubt ([`undoubted`]) and is otherwise as the pass before it: a steering
/// binding has the overload veto lifted there too, a binding that sheds
/// keeps it. So under a binding that sheds a doubted worker under its
/// thresholds takes the request. A pool that pass leaves nobody is shed at
/// the stage that ended it, as a pool with no doubt in it is: as saturated
/// where that stage is the overload veto's. A document that lists the
/// health stage ahead of that veto, as every preset does, sheds a pool
/// over on every ready worker so. Nothing harder is lifted: a worker that
/// is not ready, behind an open breaker or draining stays refused, and a
/// doubt on such a worker opens no pass. So does a row a hold of the host
/// stands on, whichever way the host says it: by the breaker bit beside
/// the stalled one on a row it wrote, or among its held rows beside lent
/// ones ([`Constraints::held`]). The lifted pass leaves such a row both
/// bits and keeps the held rows masked, so a stage reads it as vetoed in
/// every pass.
///
/// A set put to no rule is shed where it emptied ([`AllVetoed::Unasked`]):
/// each side of a pair, and the rule's own passes, so the rule is asked
/// once a decision. A pool nobody is usable in with its vetoes lifted is
/// shed at the stage that says why.
///
/// Cold and out of line: a pass whose filters keep a row never comes here.
#[cold]
#[inline(never)]
pub(super) fn emptied(
    emptied_by: &'static str,
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    set: Set<'_>,
) -> &'static str {
    if set.vetoed == AllVetoed::Unasked {
        return emptied_by;
    }
    let candidates = set.candidates;
    let steers = set.vetoed == AllVetoed::Steer;
    // The stage that ended the last pass. Each pass is put to no rule:
    // where it leaves nobody, its own stage stands.
    let mut ended = emptied_by;
    // A pool with no row over its thresholds gains nobody when the overload
    // veto is lifted: that tier is one look at the whole set.
    if steers
        && veto_ran(profile, NotOverloaded.name(), emptied_by)
        && candidates
            .iter()
            .any(|row| !NotOverloaded.keep(facts, row, cx))
    {
        let whole = RequestFacts {
            tried: &[],
            ..*facts
        };
        let pass = Constraints {
            overload_lifted: true,
            ..*cx
        };
        match overload_pass(profile, &whole, &pass, buf, set.shedding()) {
            None => {
                buf.lifted = Lifted::Overload;
                return emptied_by;
            }
            Some(stage) => ended = stage,
        }
    }
    // A doubt is a row's own bit or the host's mask beside the rows, and
    // the tier opens for a row the doubt alone refuses. A doubt on a row a
    // harder veto refuses too gains nobody when it is lifted. Nor does a
    // row a hold of the host stands on: it stays masked.
    let own = candidates.iter().any(|row| doubted_alone(row.health));
    // The two lengths answer for the rows only where every held row is a
    // row of the mask: the store's view keeps that, and a host that fills
    // the constraints by hand owes it.
    debug_assert!(
        cx.held.iter().all(|held| cx.stalled.contains(held)),
        "a held row is no row of the mask"
    );
    let masked = cx.stalled.len() > cx.held.len() && mask_doubts(candidates, cx);
    if (own || masked) && veto_ran(profile, Healthy.name(), ended) {
        return doubted(emptied_by, own, profile, facts, cx, buf, set);
    }
    ended
}

/// The pass that lifts the overload veto, over `set` under `cx`, the lift
/// in it: `None` where a row stands, the rows in `buf.kept`, or the stage
/// that left nobody.
///
/// Where a whole pass handed its fold with the set ([`Set::fold`]) and the
/// set is the plan's whole pool, the pass is one pass of the test
/// ([`keep_folded`]) where a front row passes it ([`leads`]), and the
/// stages run only where it keeps no row. The fold is the one the engine
/// made at its build: no decision makes one. The pass that lifts the
/// liveness veto is staged.
#[inline(always)]
fn overload_pass(
    profile: &Profile,
    whole: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    set: Set<'_>,
) -> Option<&'static str> {
    if set.plain() {
        if let Some(tests) = set.fold {
            let tests = tests.asked(whole, cx);
            if leads(tests, whole, set.candidates, cx) {
                // The fold's pass asks for every buffer's room, as the
                // whole pass does: no pass of another kind behind it
                // grows a list.
                buf.reserve(set.candidates.len());
                if keep_folded(tests, whole, set.candidates, cx, &mut buf.kept) {
                    return None;
                }
            }
        }
    }
    filter(profile, whole, cx, buf, set)
}

/// The pass of [`emptied`] that lifts the liveness veto: the profile's
/// filters once more, over the rows with no doubt ([`undoubted`]) where a
/// row's own bit is set (`own`), with the host's mask cut to the rows a
/// hold stands on. `emptied_by` with the rows the pass kept in `buf.kept`
/// and the lift said in `buf.lifted`, or the stage that left it nobody.
///
/// The overload veto is lifted in this pass under a steering binding,
/// whether its own tier ran or not: that tier runs under no other.
///
/// Out of line: the overload tier's pass, the one a pool over its
/// thresholds on every worker takes, keeps nothing of this one in hand.
#[cold]
#[inline(never)]
fn doubted(
    emptied_by: &'static str,
    own: bool,
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    set: Set<'_>,
) -> &'static str {
    let whole = RequestFacts {
        tried: &[],
        ..*facts
    };
    let pass = Constraints {
        overload_lifted: cx.overload_lifted || set.vetoed == AllVetoed::Steer,
        stalled: cx.held,
        ..*cx
    };
    let stage = if own {
        let mut rows = std::mem::take(&mut buf.undoubted);
        undoubted(set.candidates, &mut rows);
        let lifted = Set {
            candidates: &rows,
            ..set.shedding()
        };
        let stage = filter(profile, &whole, &pass, buf, lifted);
        buf.undoubted = rows;
        stage
    } else {
        filter(profile, &whole, &pass, buf, set.shedding())
    };
    match stage {
        None => {
            buf.lifted = Lifted::Liveness;
            emptied_by
        }
        Some(stage) => stage,
    }
}

/// The rows as they read with no doubt, into `rows`: each row of
/// `candidates` with its stalled bit cleared where its breaker bit is
/// clear, in the same order, so a stage handed a row reads the lift
/// whatever it reads of the row. A row that carries both bits is one a hold
/// of the host stands on, where the host wrote its rows. It keeps both, as
/// a held row stays in the mask beside lent rows. The pass that lifts the
/// liveness veto is shown these where a row's own bit is set. Rows the
/// host's mask alone doubts are read as they are, with the mask cut to the
/// rows a hold stands on. A stage that answers from its own scope answers
/// as it did: the scope was opened over the decision's rows. One pool of
/// rows a scratch, sized at the first pass that needs it.
fn undoubted(candidates: &[Candidate], rows: &mut Vec<Candidate>) {
    rows.clear();
    rows.extend(candidates.iter().map(|row| Candidate {
        health: Health {
            stalled: row.health.stalled && row.health.breaker_open,
            ..row.health
        },
        ..*row
    }));
}

/// Whether the liveness veto alone refuses a row of these bits: the row
/// carries the doubt and stands once it is lifted. The health stage's own
/// test of the row with no doubt on it
/// ([`Health::routable_ignoring_overload`]), so the two cannot part.
const fn doubted_alone(health: Health) -> bool {
    health.stalled
        && Health {
            stalled: false,
            ..health
        }
        .routable_ignoring_overload()
}

/// Whether the host's mask alone refuses a row: a row of the mask no hold
/// stands on whose own bits refuse nothing. Both lists are in order, so a
/// held row is found by a search. Asked only of a pool the rule was put
/// to, and only where the mask is longer than the held list.
fn mask_doubts(candidates: &[Candidate], cx: &Constraints<'_>) -> bool {
    cx.stalled.iter().any(|&at| {
        cx.held.binary_search(&at).is_err()
            && candidates
                .get(at as usize)
                .is_some_and(|row| row.health.routable_ignoring_overload())
    })
}

/// The rule's pick among the rows its lifted pass kept (`buf.kept`): the
/// least-vetoed worker, which is the fewest requests in flight, ties drawn
/// by the decision's seed, the next fewest as alternates. The pick is the
/// engine's rule and no stage of the profile makes it, so a document cannot
/// put a scorer where a safety default stands. It pins no session key, and
/// its trace says the veto the rule lifted for those rows ([`Lifted`], as
/// [`emptied`] left it in `buf.lifted`): the mark a host counts its steered
/// picks from, each tier apart.
pub(super) fn steered(
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
) -> Decision {
    // Sized and not cleared: the scorer writes every score it is handed.
    buf.scores.resize(buf.kept.len(), 0.0);
    let band = LeastRequests.score(facts, candidates, &buf.kept, &mut buf.scores);
    let mut targets = Targets::new();
    Best.pick(
        facts,
        candidates,
        &buf.kept,
        &buf.scores,
        band,
        &mut targets,
    );
    let trace = Trace {
        lifted: buf.lifted,
        ..trace_of(index, facts, pool, candidates, &buf.kept, &buf.scores)
    };
    Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, Weak,
    };

    use crate::{
        builtins::{Best, Healthy, Manual, NotOverloaded},
        counts::tests::Lent,
        decision::{Decision, Lifted, ShedReason},
        framework::{
            tests::Stored, Binding, DecisionScope, Engine, Filter, Marks, PairPools, Picker, Plan,
            Profile, Scorer, Scratch, TieBand, Weighted, DRAW_FROM, WRITTEN_ROWS,
        },
        lease::{Lease, Lifecycle, Side},
        ring::ConsistentHash,
        sessions::{SessionRule, SessionVerdict},
        spec::{Assignment, Needs, OverloadSpec, PolicySpec, Recovery},
        view::{
            Candidate, Constraints, Engines, Health, Identity, NamedAffinity, NamedHolder, Phase,
            PoolFacts, RequestFacts, WorkerId,
        },
    };

    /// A ready worker over its thresholds, holding `in_flight` requests.
    fn over(position: u32, in_flight: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
            key: 100 + position,
            health: Health {
                healthy: true,
                overloaded: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        }
    }

    /// The engine of one preset; `shed` writes the binding's
    /// `overload.shed_when_all_vetoed`.
    fn engine_of(preset: &str, shed: bool) -> Engine {
        let mut spec = PolicySpec::preset(preset).expect("the preset's form");
        if shed {
            spec.bindings
                .first_mut()
                .expect("the preset's binding")
                .overload = Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            });
        }
        spec.validate().expect("the preset builds")
    }

    /// The round robin preset with its one profile's hard filters replaced.
    fn engine_filtering(filters: &[&str], shed: bool) -> Engine {
        let mut spec = PolicySpec::preset("round_robin").expect("the preset's form");
        for profile in spec.profiles.values_mut() {
            profile.filter = filters.iter().map(ToString::to_string).collect();
        }
        if shed {
            spec.bindings
                .first_mut()
                .expect("the preset's binding")
                .overload = Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            });
        }
        spec.validate().expect("the document builds")
    }

    /// A request of the cases' host, which writes a key's counts into its
    /// rows.
    fn facts(seed: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            seed,
            session_counted: true,
            ..RequestFacts::default()
        }
    }

    fn decide(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
        engine.decide(
            facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    /// Refuses every row and counts the rows it was asked about.
    #[derive(Debug)]
    struct Refuses(Arc<AtomicUsize>);
    impl Filter for Refuses {
        fn name(&self) -> &'static str {
            "refuses"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Keeps every row and counts the rows it was asked about.
    #[derive(Debug)]
    struct Counts(Arc<AtomicUsize>);
    impl Filter for Counts {
        fn name(&self) -> &'static str {
            "counts"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            true
        }
    }

    /// The positions a pick names, best first.
    fn order(decision: &Decision) -> Vec<u32> {
        let pick = decision.pick().expect("a pick");
        pick.targets.iter().map(|t| t.position).collect()
    }

    /// The reason and the stage of a shed.
    fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
        match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not a shed: {other:?}"),
        }
    }

    /// Every preset whose stages this build carries steers a pool that is
    /// over its thresholds on every worker to the fewest requests in flight,
    /// whatever its own stages would pick, and names the next fewest as
    /// alternates.
    #[test]
    fn a_pool_over_on_every_worker_goes_to_the_fewest_in_flight_under_every_preset() {
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        for preset in [
            "random",
            "round_robin",
            "passthrough",
            "power_of_two",
            "least_load",
        ] {
            let engine = engine_of(preset, false);
            for seed in 1..=8 {
                let decision = decide(&engine, &facts(seed), &pool);
                assert_eq!(order(&decision), vec![1, 2, 0], "{preset}, seed {seed}");
                let pick = decision.pick().expect("a pick");
                assert_eq!(pick.trace.survivors.get(), Some(3), "{preset}");
                assert_eq!(pick.session_verdict, SessionVerdict::None, "{preset}");
            }
        }
    }

    /// The steer reads the fewest requests through the request's facts: a
    /// pool whose counts stand in a table lent beside its rows, each row
    /// reading zero, is steered as the pool that carries them, under
    /// presets whose own stages read no count.
    #[test]
    fn a_steered_pool_goes_to_the_fewest_in_flight_by_a_lent_table() {
        let mut pool = [over(0, 3), over(1, 1), over(2, 2)];
        let lent = Lent::take(&mut pool);
        assert!(pool.iter().all(|row| row.in_flight == 0));
        for preset in ["random", "round_robin", "passthrough"] {
            let engine = engine_of(preset, false);
            for seed in 1..=8 {
                let decision = decide(&engine, &lent.beside(&facts(seed)), &pool);
                assert_eq!(order(&decision), vec![1, 2, 0], "{preset}, seed {seed}");
            }
        }
    }

    /// Under the binding's word the same pool is shed as saturated, by the
    /// overload filter's name.
    #[test]
    fn the_same_pool_is_shed_as_saturated_under_the_bindings_word() {
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        for preset in ["random", "round_robin", "least_load", "manual"] {
            let decision = decide(&engine_of(preset, true), &facts(1), &pool);
            assert_eq!(
                shed_of(&decision),
                (ShedReason::Saturated, "not-overloaded"),
                "{preset}"
            );
        }
    }

    /// A hard filter listed behind the overload filter does not hide an
    /// all-vetoed pool: the one worker with a free slot is over its
    /// thresholds and the free worker has no slot, so the later filter empties
    /// the pass, and the request still goes to the usable worker. The answer
    /// is the same with the two filters listed the other way round. Where no
    /// worker has a slot, lifting the veto gains nothing and the shed names
    /// the filter that says why, also when the veto itself emptied the first
    /// pass. A binding that sheds answers at the stage that emptied its
    /// pass, as before.
    #[test]
    fn a_hard_filter_behind_the_veto_does_not_hide_an_all_vetoed_pool() {
        let mut pool = [over(0, 4), over(1, 0)];
        pool[0].has_capacity = true;
        pool[1].health.overloaded = false;
        let behind = ["healthy", "not-overloaded", "has-capacity", "not-tried"];
        let ahead = ["healthy", "has-capacity", "not-overloaded", "not-tried"];
        for filters in [behind, ahead] {
            let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
            assert_eq!(order(&decision), vec![0], "{filters:?}");
        }
        assert_eq!(
            shed_of(&decide(&engine_filtering(&behind, true), &facts(1), &pool)),
            (ShedReason::NoEndpoints, "has-capacity")
        );
        assert_eq!(
            shed_of(&decide(&engine_filtering(&ahead, true), &facts(1), &pool)),
            (ShedReason::Saturated, "not-overloaded")
        );
        pool[0].has_capacity = false;
        for filters in [behind, ahead] {
            let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
            assert_eq!(
                shed_of(&decision),
                (ShedReason::NoEndpoints, "has-capacity"),
                "{filters:?}"
            );
        }
        // Every worker over its thresholds and none with a slot: the veto
        // empties the first pass, the later filter empties the second, and
        // the shed names that filter, not the veto.
        pool[1].health.overloaded = true;
        assert_eq!(
            shed_of(&decide(&engine_filtering(&behind, false), &facts(1), &pool)),
            (ShedReason::NoEndpoints, "has-capacity")
        );
    }

    /// The law the example above is one instance of: the overload veto is the
    /// one filter that steps aside, so the pool is what the hard filters keep
    /// less the vetoed, or what the hard filters keep when that leaves
    /// nobody. Under every order of three hard filters (one of them the
    /// veto) the steer is the same two workers, fewest in flight first; and
    /// with one usable worker under its thresholds every order picks it.
    #[test]
    fn the_pick_is_the_same_under_every_order_of_the_hard_filters() {
        let orders = [
            ["healthy", "not-overloaded", "has-capacity"],
            ["healthy", "has-capacity", "not-overloaded"],
            ["not-overloaded", "healthy", "has-capacity"],
            ["not-overloaded", "has-capacity", "healthy"],
            ["has-capacity", "healthy", "not-overloaded"],
            ["has-capacity", "not-overloaded", "healthy"],
        ];
        // Usable and over its thresholds: rows 0 and 3. Row 1 is free with no
        // slot; row 2 is free with a slot and draining.
        let mut pool = [over(0, 4), over(1, 0), over(2, 0), over(3, 6)];
        pool[0].has_capacity = true;
        pool[1].health.overloaded = false;
        pool[2].health.overloaded = false;
        pool[2].has_capacity = true;
        pool[2].health.draining = true;
        pool[3].has_capacity = true;
        for hard in orders {
            let filters = [hard[0], hard[1], hard[2], "not-tried"];
            let engine = engine_filtering(&filters, false);
            for seed in 1..=4 {
                let decision = decide(&engine, &facts(seed), &pool);
                assert_eq!(order(&decision), vec![0, 3], "{hard:?}, seed {seed}");
            }
        }
        // Row 1 gains a slot: one usable worker under its thresholds.
        pool[1].has_capacity = true;
        for hard in orders {
            let filters = [hard[0], hard[1], hard[2], "not-tried"];
            let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
            assert_eq!(order(&decision), vec![1], "{hard:?}");
        }
    }

    /// Equal counts are one tie, drawn by the decision's seed over the tied
    /// rows alone: a row that holds more is never drawn.
    #[test]
    fn a_tie_is_drawn_by_the_seed_over_the_tied_rows_alone() {
        let engine = engine_of("round_robin", false);
        let pool = [over(0, 2), over(1, 2), over(2, 5)];
        let mut firsts = [0u32; 3];
        for seed in 1..=600 {
            let first = order(&decide(&engine, &facts(seed), &pool))[0];
            firsts[first as usize] += 1;
        }
        assert_eq!(firsts[2], 0, "{firsts:?}");
        assert!(firsts[0] > 200 && firsts[1] > 200, "{firsts:?}");
    }

    /// A worker that is down, draining or behind an open breaker is not
    /// usable: it is never the steer, however few requests it holds. A
    /// doubted worker is the last to be taken, once no worker over its
    /// thresholds alone is left.
    #[test]
    fn a_worker_another_veto_refuses_is_never_the_steer() {
        let engine = engine_of("least_load", false);
        let mut pool = [over(0, 0), over(1, 0), over(2, 0), over(3, 0), over(4, 7)];
        pool[0].health.healthy = false;
        pool[1].health.draining = true;
        pool[2].health.stalled = true;
        pool[3].health.breaker_open = true;
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![4]);
        // With no worker over its thresholds alone the doubted one is taken.
        pool[4].health.healthy = false;
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2]);
        // With no usable row left the stage that refused them names the shed.
        pool[2].health.healthy = false;
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "healthy")
        );
    }

    /// A worker under its thresholds takes the request by the profile's own
    /// stages: the rule runs only when the overload filter leaves nobody.
    #[test]
    fn a_worker_under_its_thresholds_keeps_the_profiles_own_pick() {
        let engine = engine_of("passthrough", false);
        let mut pool = [over(0, 0), over(1, 9), over(2, 0)];
        pool[1].health.overloaded = false;
        assert_eq!(order(&decide(&engine, &facts(1), &pool))[0], 1);
    }

    /// The retry's tried list does not narrow the steer: a worker an earlier
    /// attempt tried is taken again when it holds the fewest requests.
    #[test]
    fn a_tried_worker_is_taken_again_when_it_holds_the_fewest() {
        let engine = engine_of("round_robin", false);
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        let tried = [WorkerId(101)];
        let retry = RequestFacts {
            tried: &tried,
            attempt: 1,
            ..facts(1)
        };
        assert_eq!(order(&decide(&engine, &retry, &pool)), vec![1, 2, 0]);
    }

    /// The host's list of workers it just failed to reach stays in the
    /// lifted pass: the steer passes over such a worker while another usable
    /// one stands, whatever it holds, and takes it as the last resort.
    #[test]
    fn a_worker_the_host_failed_to_reach_is_the_steers_last_resort() {
        let engine = engine_of("round_robin", false);
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        let steer = |unreachable: &[WorkerId]| {
            let cx = Constraints {
                excluded: unreachable,
                ..Constraints::default()
            };
            let decision = engine.decide(
                &facts(1),
                &PoolFacts::default(),
                &pool,
                &cx,
                &mut Scratch::new(),
            );
            order(&decision)
        };
        assert_eq!(steer(&[WorkerId(101)]), vec![2, 0]);
        assert_eq!(steer(&[WorkerId(101), WorkerId(102)]), vec![0]);
        let every = [WorkerId(100), WorkerId(101), WorkerId(102)];
        assert_eq!(steer(&every), vec![1, 2, 0]);
    }

    /// Under every preset a retry is steered past the workers its host
    /// failed to reach: three attempts over a pool over on every worker
    /// take its three workers by their requests in flight, and each pick
    /// says the lifted veto.
    #[test]
    fn a_retry_leaves_the_workers_the_host_failed_to_reach() {
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        let mut read = Vec::new();
        for preset in [
            "round_robin",
            "least_load",
            "random",
            "power_of_two",
            "cache_aware",
            "consistent_hashing",
            "passthrough",
        ] {
            let engine = engine_of(preset, false);
            let mut failed: Vec<WorkerId> = Vec::new();
            let mut firsts = Vec::new();
            for attempt in 0..3u64 {
                let cx = Constraints {
                    excluded: &failed,
                    ..Constraints::default()
                };
                let retry = RequestFacts {
                    tried: &failed,
                    ..facts(1 + attempt)
                };
                let decision = engine.decide(
                    &retry,
                    &PoolFacts::default(),
                    &pool,
                    &cx,
                    &mut Scratch::new(),
                );
                let first = order(&decision)[0];
                assert_eq!(
                    decision.trace().lifted,
                    Lifted::Overload,
                    "{preset} attempt {attempt}"
                );
                firsts.push(first);
                failed.push(WorkerId(100 + u64::from(first)));
            }
            read.push((preset, firsts));
        }
        assert!(
            read.iter().all(|(_, firsts)| firsts == &[1, 2, 0]),
            "{read:?}"
        );
    }

    /// A steered pick says the veto the rule lifted, the mark a host counts
    /// such picks from. A pick the profile's own stages made says
    /// none, and so does the shed of a binding that sheds.
    #[test]
    fn a_steered_pick_says_the_veto_it_lifted() {
        let engine = engine_of("round_robin", false);
        let steered = decide(&engine, &facts(1), &[over(0, 3), over(1, 1)]);
        assert_eq!(steered.trace().lifted, Lifted::Overload);
        let mut under = over(1, 1);
        under.health.overloaded = false;
        let kept = decide(&engine, &facts(1), &[over(0, 3), under]);
        assert_eq!(kept.trace().lifted, Lifted::None);
        let shedding = engine_of("round_robin", true);
        let shed = decide(&shedding, &facts(1), &[over(0, 3), over(1, 1)]);
        assert_eq!(shed.trace().lifted, Lifted::None);
    }

    /// A binding the engine does not hold cannot be told to shed: the index
    /// is the validator's own, and a debug build refuses any other.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "is not built"))]
    fn a_binding_the_engine_does_not_hold_is_refused_in_a_debug_build() {
        let mut engine = engine_of("round_robin", false);
        engine.shed_when_all_vetoed(7);
    }

    /// A host's lift puts the decision outside the veto: the filter keeps
    /// every row and the profile's own stages pick.
    #[test]
    fn a_hosts_lift_keeps_the_profiles_own_pick() {
        let engine = engine_of("passthrough", false);
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        let lifted = Constraints {
            overload_lifted: true,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &lifted,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision)[0], 0, "the first survivor, as listed");
    }

    /// A pool a picker draws from ends in the same rule. The rows the draw
    /// names are refused as every other row is, the whole pass keeps nobody
    /// either, and the binding's word answers: the fewest in flight, or the
    /// shed.
    #[test]
    fn a_pool_a_picker_draws_from_ends_in_the_same_rule() {
        let rows = u32::try_from(DRAW_FROM).expect("a small count") + 4;
        // The fewest requests in flight stand on the last row.
        let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
        for preset in [
            "random",
            "round_robin",
            "passthrough",
            "power_of_two",
            "least_load",
        ] {
            let steering = engine_of(preset, false);
            let shedding = engine_of(preset, true);
            for seed in 1..=8 {
                let steered = decide(&steering, &facts(seed), &pool);
                assert_eq!(
                    order(&steered).first(),
                    Some(&(rows - 1)),
                    "{preset}, seed {seed}"
                );
                let pick = steered.pick().expect("a pick");
                assert_eq!(pick.trace.survivors.get(), Some(rows), "{preset}");
                assert_eq!(
                    shed_of(&decide(&shedding, &facts(seed), &pool)),
                    (ShedReason::Saturated, "not-overloaded"),
                    "{preset}, seed {seed}"
                );
            }
        }
    }

    /// A request that names rows ends in the same rule. Under a picker the
    /// rows named join, the draw is asked for such a request: the rows it
    /// names are refused as every other row is, the whole pass keeps nobody
    /// either, and the binding's word answers: the fewest in flight, which
    /// no name stands on, or the shed.
    #[test]
    fn a_request_that_names_rows_ends_in_the_same_rule() {
        let rows = u32::try_from(DRAW_FROM).expect("a small count") + 4;
        // The fewest requests in flight stand on the last row; the host
        // names two others.
        let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
        static HASHES: [u64; 16] = [0; 16];
        let holders = [
            NamedHolder {
                row: 3,
                overlap_blocks: 12,
                predicted_blocks: 0,
            },
            NamedHolder {
                row: 7,
                overlap_blocks: 8,
                predicted_blocks: 0,
            },
        ];
        let steering = engine_of("cache_aware", false);
        let shedding = engine_of("cache_aware", true);
        for seed in 1..=8 {
            let naming = RequestFacts {
                block_hashes: &HASHES,
                prompt_tokens: 256,
                named: Some(NamedAffinity {
                    holders: &holders,
                    ..NamedAffinity::default()
                }),
                ..facts(seed)
            };
            let steered = decide(&steering, &naming, &pool);
            assert_eq!(order(&steered).first(), Some(&(rows - 1)), "seed {seed}");
            let pick = steered.pick().expect("a pick");
            assert_eq!(pick.trace.survivors.get(), Some(rows), "seed {seed}");
            assert_eq!(
                shed_of(&decide(&shedding, &naming, &pool)),
                (ShedReason::Saturated, "not-overloaded"),
                "seed {seed}"
            );
        }
    }

    /// The word is the binding's and not the profile's: two bindings that
    /// share one profile answer each by its own, over a pool the whole pass
    /// decides and over one a picker draws from. The binding that sheds
    /// stands first and then second, so the place the validator names for
    /// it is read.
    #[test]
    fn two_bindings_of_one_profile_answer_each_by_its_own_word() {
        let sheds = || {
            Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            })
        };
        let drawn = u32::try_from(DRAW_FROM).expect("a small count") + 4;
        for shedding_first in [true, false] {
            let mut spec = PolicySpec::preset("random").expect("the preset's form");
            let wildcard = spec.bindings.first_mut().expect("the preset's binding");
            let mut named = wildcard.clone();
            named.matcher.model = "named".into();
            if shedding_first {
                named.overload = sheds();
            } else {
                wildcard.overload = sheds();
            }
            spec.bindings.insert(0, named);
            let engine = spec.validate().expect("the document builds");
            let (steers, strict) = if shedding_first {
                ("m", "named")
            } else {
                ("named", "m")
            };
            for rows in [3, drawn] {
                // The fewest requests in flight stand on the last row.
                let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
                let steered = RequestFacts {
                    model: steers,
                    ..facts(1)
                };
                assert_eq!(
                    order(&decide(&engine, &steered, &pool)).first(),
                    Some(&(rows - 1)),
                    "{rows} rows, the shedding binding first: {shedding_first}"
                );
                let shed = RequestFacts {
                    model: strict,
                    ..facts(1)
                };
                assert_eq!(
                    shed_of(&decide(&engine, &shed, &pool)),
                    (ShedReason::Saturated, "not-overloaded"),
                    "{rows} rows, the shedding binding first: {shedding_first}"
                );
            }
        }
    }

    /// The pair plan's shed over `prefills` and one free decode. The prefill
    /// side is filtered by `veto` and then by a filter that refuses every
    /// row, under the host's mask over the prefills and the rows of it the
    /// host holds (`masks`). With the shed, the count of the prefills that
    /// filter was asked about.
    fn pairs_side_shed(
        veto: Box<dyn Filter>,
        prefills: [Candidate; 2],
        masks: (&[u32], &[u32]),
    ) -> ((ShedReason, &'static str), usize) {
        let asked = Arc::new(AtomicUsize::new(0));
        let prefill_side = Profile {
            name: "counted".into(),
            filters: vec![veto, Box::new(Refuses(Arc::clone(&asked)))],
            scorers: vec![],
            picker: Box::new(Best),
        };
        let decode_side = Profile {
            name: "plain".into(),
            filters: vec![Box::new(NotOverloaded)],
            scorers: vec![],
            picker: Box::new(Best),
        };
        let engine = Engine::new(
            vec![prefill_side, decode_side],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let paired = |mut worker: Candidate| {
            worker.pair_class = 1;
            worker
        };
        let prefills = prefills.map(paired);
        let mut free = paired(over(0, 0));
        free.health.overloaded = false;
        let decodes = [free];
        let decision = engine.decide_pair(
            &facts(1),
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints {
                stalled: masks.0,
                held: masks.1,
                ..Constraints::default()
            },
            &Constraints::default(),
            &mut Scratch::new(),
        );
        (shed_of(&decision), asked.load(Ordering::Relaxed))
    }

    /// Each side of a pair sheds a pool the overload veto empties: the pair
    /// plan does not enter the rule. Under a binding that steers, a prefill
    /// side over its thresholds on every worker is shed at the veto, and the
    /// filter listed behind the veto is asked about no prefill.
    #[test]
    fn a_pairs_side_sheds_at_the_veto_and_is_filtered_once() {
        let none: (&[u32], &[u32]) = (&[], &[]);
        let side = pairs_side_shed(Box::new(NotOverloaded), [over(0, 1), over(1, 2)], none);
        assert_eq!(side, ((ShedReason::Saturated, "not-overloaded"), 0));
    }

    /// An engine of one profile: `veto` and a filter that refuses every row,
    /// the veto listed first when `veto_first`. With it, the count of the
    /// rows the refusing filter was asked about.
    fn counted(veto: Box<dyn Filter>, veto_first: bool) -> (Engine, Arc<AtomicUsize>) {
        counted_among(|refuses| {
            if veto_first {
                vec![veto, refuses]
            } else {
                vec![refuses, veto]
            }
        })
    }

    /// An engine of one profile under the filters `listed` gives, among
    /// them a filter that refuses every row. With it, the count of the rows
    /// that filter was asked about.
    fn counted_among(
        listed: impl FnOnce(Box<dyn Filter>) -> Vec<Box<dyn Filter>>,
    ) -> (Engine, Arc<AtomicUsize>) {
        let asked = Arc::new(AtomicUsize::new(0));
        let filters = listed(Box::new(Refuses(Arc::clone(&asked))));
        let engine = Engine::new(
            vec![Profile {
                name: "counted".into(),
                filters,
                scorers: vec![],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        (engine, asked)
    }

    /// A shed the veto had no part in is not filtered a second time: a pool
    /// emptied before the veto ran, a decision whose host lifted the veto
    /// itself and a pool with no row over its thresholds cost the one pass
    /// they cost without the rule. A pool with one row over its thresholds
    /// is filtered again, and sheds at the stage behind the veto.
    #[test]
    fn a_shed_the_veto_had_no_part_in_costs_one_pass() {
        let counted = |veto_first: bool| counted(Box::new(NotOverloaded), veto_first);
        let pool = [over(0, 1), over(1, 2)];
        // Listed ahead of the veto: the veto never ran.
        let (engine, asked) = counted(false);
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
        // Listed behind it, under a host's lift: the veto was off already.
        let (engine, asked) = counted(true);
        let lifted = Constraints {
            overload_lifted: true,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &lifted,
            &mut Scratch::new(),
        );
        assert_eq!(shed_of(&decision), (ShedReason::NoEndpoints, "refuses"));
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
        // Listed behind it, with nobody over its thresholds: the veto ran
        // and refused no row.
        let (engine, asked) = counted(true);
        let mut under = pool;
        for worker in &mut under {
            worker.health.overloaded = false;
        }
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &under)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), under.len());
        // One row over its thresholds: the veto refused it, so the lifted
        // pass asks the filter behind the veto about both rows.
        let (engine, asked) = counted(true);
        let mut mixed = under;
        mixed[0].health.overloaded = true;
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &mixed)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), 1 + mixed.len());
    }

    /// The by-assignment path (a picker that leaves the placement to its
    /// rule, a request with no key) is steered as the plain path is.
    #[test]
    fn a_request_placed_by_assignment_is_steered_too() {
        let pool = [over(0, 3), over(1, 1), over(2, 2)];
        let decision = decide(&engine_of("manual", false), &facts(1), &pool);
        assert_eq!(order(&decision), vec![1, 2, 0]);
    }

    /// The keyed path: a key whose pool is over on every worker goes to the
    /// fewest in flight and the steer pins nothing, so the key's pin stands
    /// for the decision after the pool recovers.
    #[test]
    fn a_keyed_request_is_steered_and_the_steer_pins_nothing() {
        let engine = engine_of("manual", false);
        let place = |pool: &[Candidate], now: u64| {
            let keyed = RequestFacts {
                session_key: Identity::plain("k1"),
                now_micros: now,
                ..facts(now)
            };
            let decision = decide(&engine, &keyed, pool);
            let pick = *decision.pick().expect("a pick");
            let target = *pick.targets.first().expect("a target");
            let lease = Lease {
                issued_micros: now,
                session: keyed.session(),
                session_verdict: pick.session_verdict,
                ..Lease::default()
            };
            engine.on_dispatched(&lease, &target, Side::Single, 0);
            (target.position, pick.session_verdict)
        };
        let mut pool = [over(0, 1), over(1, 2), over(2, 3)];
        for c in &mut pool {
            c.health.overloaded = false;
        }
        let (home, _) = place(&pool, 1_000);
        for c in &mut pool {
            c.health.overloaded = true;
        }
        pool[home as usize].in_flight = 9;
        let (steered, verdict) = place(&pool, 2_000);
        assert_ne!(steered, home, "the fewest in flight, not the pin");
        assert_eq!(verdict, SessionVerdict::None);
        for c in &mut pool {
            c.health.overloaded = false;
        }
        assert_eq!(place(&pool, 3_000).0, home, "the pin stands");
    }

    /// An encode item of the `epd` plan is steered behind the plan's own
    /// rules: an encoder the pair cannot use is not usable.
    #[test]
    fn an_encode_item_is_steered_among_the_encoders_the_plan_admits() {
        let engine = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin")
            .expect("the presets' form")
            .validate()
            .expect("the presets build");
        let mut pool = [over(0, 0), over(1, 4), over(2, 2)];
        pool[1].ec_compatible = true;
        pool[2].ec_compatible = true;
        let item = RequestFacts {
            phase: Phase::Encode,
            ..facts(1)
        };
        assert_eq!(order(&decide(&engine, &item, &pool)), vec![2, 1]);
    }

    /// A ready worker under its thresholds, holding `in_flight` requests.
    fn free(position: u32, in_flight: u32) -> Candidate {
        let mut worker = over(position, in_flight);
        worker.health.overloaded = false;
        worker
    }

    fn keyed(key: &'static str) -> RequestFacts<'static> {
        RequestFacts {
            session_key: Identity::plain(key),
            ..facts(1)
        }
    }

    /// The ring's order for the request's key over `pool`, found by probes
    /// and never written as places: the pick with every worker ready and
    /// under its thresholds, then with the members found so far not ready.
    fn ring_order(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Vec<u32> {
        let mut probe = pool.to_vec();
        for worker in &mut probe {
            worker.health.overloaded = false;
        }
        let mut walk = Vec::with_capacity(pool.len());
        while walk.len() < pool.len() {
            let member = order(&decide(engine, facts, &probe))[0];
            walk.push(member);
            probe[member as usize].health.healthy = false;
        }
        walk
    }

    /// A profile that places by the session key keeps a keyed request on
    /// its key's worker over the veto, where the binding steers: the owner
    /// over its thresholds takes the key beside free workers, and takes it
    /// when every worker is over, whoever holds the fewest requests.
    #[test]
    fn a_key_keeps_its_ring_owner_over_the_veto() {
        let engine = engine_of("consistent_hashing", false);
        let request = keyed("k1");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine, &request, &pool);
        let owner = ring[0] as usize;
        pool[owner].health.overloaded = true;
        pool[owner].in_flight = 9;
        assert_eq!(
            order(&decide(&engine, &request, &pool))[0],
            ring[0],
            "beside free workers"
        );
        for worker in &mut pool {
            worker.health.overloaded = true;
        }
        let decision = decide(&engine, &request, &pool);
        assert_eq!(order(&decision)[0], ring[0], "every worker over");
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.trace.survivors.get(), Some(4));
    }

    /// A binding that sheds keeps the veto for a keyed request: an owner
    /// over its thresholds passes the key to the ring's next member, and a
    /// pool over on every worker is shed as saturated.
    #[test]
    fn a_binding_that_sheds_keeps_the_veto_for_a_key() {
        let engine = engine_of("consistent_hashing", true);
        let request = keyed("k1");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine, &request, &pool);
        pool[ring[0] as usize].health.overloaded = true;
        assert_eq!(order(&decide(&engine, &request, &pool))[0], ring[1]);
        for worker in &mut pool {
            worker.health.overloaded = true;
        }
        assert_eq!(
            shed_of(&decide(&engine, &request, &pool)),
            (ShedReason::Saturated, "not-overloaded")
        );
    }

    /// A kept key's pick says the veto it stands by. With every worker over
    /// its thresholds the veto as it stands leaves the pool nobody, and the
    /// trace carries the mark the rule's own pick carries: a host counts
    /// both from it. Beside a worker under its thresholds the owner over its
    /// own takes the key as a session keeps its owner, and the trace says no
    /// veto. So it does beside a worker under its thresholds that the retry's
    /// list passes over: the veto did not empty that pool.
    #[test]
    fn a_kept_keys_pick_says_the_veto_over_an_all_vetoed_pool_alone() {
        let engine = engine_of("consistent_hashing", false);
        let request = keyed("k1");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine, &request, &pool);
        let (owner, next) = (ring[0] as usize, ring[1] as usize);
        pool[owner].health.overloaded = true;
        let beside = decide(&engine, &request, &pool);
        assert_eq!(order(&beside)[0], ring[0]);
        assert_eq!(beside.trace().lifted, Lifted::None, "beside free workers");
        for worker in &mut pool {
            worker.health.overloaded = true;
        }
        let whole = decide(&engine, &request, &pool);
        assert_eq!(order(&whole)[0], ring[0]);
        assert_eq!(whole.trace().lifted, Lifted::Overload, "every worker over");
        pool[next].health.overloaded = false;
        let tried = [pool[next].id];
        let retry = RequestFacts {
            tried: &tried,
            attempt: 1,
            ..request
        };
        let passed = decide(&engine, &retry, &pool);
        assert_eq!(order(&passed)[0], ring[0]);
        assert_eq!(passed.trace().lifted, Lifted::None, "a tried worker under");
    }

    /// No pass of a kept key is put to the all-vetoed rule: its lifted run
    /// stands in for it. With every worker over its thresholds the profile
    /// runs once with the veto lifted, and the pass that asks whether the
    /// veto emptied the pool stops at the veto: a filter listed behind it is
    /// asked about each worker once.
    #[test]
    fn a_kept_keys_pool_is_put_to_no_rule() {
        /// Keeps every row and counts the rows it was asked about.
        #[derive(Debug)]
        struct Counts(Arc<AtomicUsize>);
        impl Filter for Counts {
            fn name(&self) -> &'static str {
                "counts"
            }
            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed);
                true
            }
        }
        let asked = Arc::new(AtomicUsize::new(0));
        let mut engine = Engine::new(
            vec![Profile {
                name: "counted".into(),
                filters: vec![
                    Box::new(NotOverloaded),
                    Box::new(Counts(Arc::clone(&asked))),
                ],
                scorers: vec![],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        engine.places_by_key(0);
        let pool = [over(0, 1), over(1, 2)];
        let decision = decide(&engine, &keyed("k1"), &pool);
        assert_eq!(decision.trace().lifted, Lifted::Overload);
        assert_eq!(asked.load(Ordering::Relaxed), 2);
    }

    /// The lift is the veto's alone: an owner that is not ready, or that an
    /// earlier attempt tried, passes the key to the ring's next member under
    /// either word of the binding.
    #[test]
    fn a_key_whose_owner_is_unusable_walks_the_ring() {
        for shed in [false, true] {
            let engine = engine_of("consistent_hashing", shed);
            let request = keyed("k1");
            let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
            let ring = ring_order(&engine, &request, &pool);
            let owner = ring[0] as usize;
            pool[owner].health.overloaded = true;
            let tried = [pool[owner].id];
            let retry = RequestFacts {
                tried: &tried,
                attempt: 1,
                ..request
            };
            assert_eq!(
                order(&decide(&engine, &retry, &pool))[0],
                ring[1],
                "tried, shed {shed}"
            );
            let excluded = Constraints {
                excluded: &tried,
                ..Constraints::default()
            };
            let decision = engine.decide(
                &request,
                &PoolFacts::default(),
                &pool,
                &excluded,
                &mut Scratch::new(),
            );
            assert_eq!(order(&decision)[0], ring[1], "unreached, shed {shed}");
            pool[owner].health.healthy = false;
            assert_eq!(
                order(&decide(&engine, &request, &pool))[0],
                ring[1],
                "not ready, shed {shed}"
            );
        }
    }

    /// The lift is the mark's alone: a request with no key under the ring's
    /// profile, and a keyed request under a profile with no ring, meet the
    /// veto as every request does. A pool over on every worker goes to the
    /// fewest in flight, and a worker under its thresholds takes the request
    /// when there is one.
    #[test]
    fn a_request_the_ring_does_not_place_gets_no_lift() {
        for (preset, request) in [
            ("consistent_hashing", facts(1)),
            ("round_robin", keyed("k1")),
            ("least_load", keyed("k1")),
        ] {
            let engine = engine_of(preset, false);
            let mut pool = [over(0, 3), over(1, 1), over(2, 2), over(3, 4)];
            assert_eq!(order(&decide(&engine, &request, &pool))[0], 1, "{preset}");
            pool[3].health.overloaded = false;
            assert_eq!(
                order(&decide(&engine, &request, &pool)),
                vec![3],
                "{preset}"
            );
        }
    }

    /// The lift is the single plan's: an encode item of the `epd` plan is
    /// keyed by its content hash, and under the ring's profile it meets the
    /// veto as every item does. Encoders over on every worker take it by
    /// the fewest in flight, whichever of them the ring names.
    #[test]
    fn an_encode_item_under_the_ring_keeps_the_veto() {
        let engine = PolicySpec::epd_presets("consistent_hashing", "round_robin", "round_robin")
            .expect("the presets' form")
            .validate()
            .expect("the presets build");
        let mut pool = [over(0, 3), over(1, 1), over(2, 2), over(3, 4)];
        for worker in &mut pool {
            worker.ec_compatible = true;
        }
        for key in ["k1", "k2", "k3", "k4"] {
            let item = RequestFacts {
                phase: Phase::Encode,
                session_key: Identity::plain(key),
                ..facts(1)
            };
            assert_eq!(order(&decide(&engine, &item, &pool))[0], 1, "{key}");
        }
    }

    /// An engine built by hand marks no profile: its ring stage meets the
    /// veto, and the host's lift keeps the key on its worker.
    #[test]
    fn an_engine_built_by_hand_lifts_nothing_for_a_key() {
        let engine = Engine::new(
            vec![Profile {
                name: "ring".into(),
                filters: vec![Box::new(NotOverloaded)],
                scorers: vec![Weighted {
                    scorer: Box::new(ConsistentHash::new(None)),
                    weight: 1.0,
                }],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let request = keyed("k1");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let owner = order(&decide(&engine, &request, &pool))[0];
        pool[owner as usize].health.overloaded = true;
        assert_ne!(order(&decide(&engine, &request, &pool))[0], owner);
        let lifted = Constraints {
            overload_lifted: true,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &request,
            &PoolFacts::default(),
            &pool,
            &lifted,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision)[0], owner);
    }

    /// The ring's profile under the manual preset's sessions rule, with the
    /// header that rule reads admitted by the view. No gateway composes the
    /// two: the composition is this engine's own.
    fn ring_under_a_rule(assignment: Assignment) -> Engine {
        ring_under(assignment, None)
    }

    /// The same composition under a rule that delegates, with the rule's
    /// in-flight cap set: the manual preset's own rule has none.
    fn ring_under_a_cap(cap: u16) -> Engine {
        ring_under(Assignment::Delegate, Some(cap))
    }

    fn ring_under(assignment: Assignment, cap: Option<u16>) -> Engine {
        let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
        let manual = PolicySpec::preset("manual").expect("the preset's form");
        let mut rule = manual
            .bindings
            .first()
            .and_then(|binding| binding.sessions.clone())
            .expect("the manual preset's rule");
        rule.assignment = assignment;
        if let Some(cap) = cap {
            rule.inflight_cap = cap;
        }
        spec.view.headers = manual.view.headers.clone();
        spec.bindings
            .first_mut()
            .expect("the preset's binding")
            .sessions = Some(rule);
        spec.validate().expect("the document builds")
    }

    /// The worker a request keyed `k1` is sent to, dispatched there so its
    /// pin is written.
    fn dispatched(engine: &Engine, pool: &[Candidate], now: u64) -> u32 {
        let request = RequestFacts {
            session_key: Identity::plain("k1"),
            now_micros: now,
            ..facts(now)
        };
        let decision = decide(engine, &request, pool);
        let pick = *decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: now,
            session: request.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
        target.position
    }

    /// A pin does not borrow the ring's lift. Under a sessions rule over the
    /// ring's profile the pin's test reads the veto as it stands: a pinned
    /// worker over its thresholds yields, and the ring's own worker for the
    /// key takes the request over its thresholds, beside free workers.
    #[test]
    fn a_pin_over_its_thresholds_yields_and_the_rings_worker_takes_the_key() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        // The key's ring order, read on an engine of the same profile with
        // no rule.
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let (owner, next) = (ring[0] as usize, ring[1] as usize);
        // The key's worker is not ready: the ring's next member takes the
        // key and is pinned.
        pool[owner].health.healthy = false;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[1], "the pin");
        // The worker is back and free: the pin stands.
        pool[owner].health.healthy = true;
        assert_eq!(dispatched(&engine, &pool, 1_500), ring[1], "the pin stands");
        // The worker and the pin are both over their thresholds beside two
        // free workers.
        pool[owner].health.overloaded = true;
        pool[next].health.overloaded = true;
        assert_eq!(
            dispatched(&engine, &pool, 2_000),
            ring[0],
            "the pin yields and the ring's worker takes the key"
        );
    }

    /// A pool over on every worker keeps no pin standing: the pin's test
    /// reads the veto as it stands, under which nobody is kept, and no rule
    /// answers that pass. The ring's own worker takes the key, and the trace
    /// says the lifted veto.
    #[test]
    fn a_pin_yields_in_a_pool_over_on_every_worker() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        // The key's worker is not ready: the ring's next member is pinned.
        pool[owner].health.healthy = false;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[1], "the pin");
        pool[owner].health.healthy = true;
        for worker in &mut pool {
            worker.health.overloaded = true;
        }
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let whole = decide(&engine, &request, &pool);
        assert_eq!(order(&whole)[0], ring[0], "the pin yields");
        assert_eq!(whole.trace().lifted, Lifted::Overload);
    }

    /// Under a sessions rule the lifted placement says the veto the same
    /// way: a key no pin takes, placed in a pool over on every worker,
    /// carries the mark, and beside a worker under its thresholds it carries
    /// none.
    #[test]
    fn a_key_placed_under_a_rule_says_the_veto_over_an_all_vetoed_pool_alone() {
        let mut pool = [over(0, 4), over(1, 4), over(2, 4), over(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let request = RequestFacts {
            now_micros: 1_000,
            ..keyed("k1")
        };
        let whole = decide(&ring_under_a_rule(Assignment::Delegate), &request, &pool);
        assert_eq!(order(&whole)[0], ring[0]);
        assert_eq!(whole.trace().lifted, Lifted::Overload, "every worker over");
        pool[ring[1] as usize].health.overloaded = false;
        let beside = decide(&ring_under_a_rule(Assignment::Delegate), &request, &pool);
        assert_eq!(order(&beside)[0], ring[0]);
        assert_eq!(beside.trace().lifted, Lifted::None, "beside a free worker");
    }

    /// The lift is the ring's: a rule whose own assignment places a key no
    /// pin takes asks no stage where the key goes, and nothing is lifted.
    /// The key's worker over its thresholds is passed over for the fewest
    /// in flight under theirs.
    #[test]
    fn a_rule_that_places_a_key_itself_lifts_nothing() {
        let engine = ring_under_a_rule(Assignment::MinLoad);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        pool[ring[0] as usize].health.overloaded = true;
        pool[ring[3] as usize].in_flight = 1;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[3]);
    }

    /// The key's mark moves a profile's scope word to the key's own, where
    /// both plans still find a scope to open, and it leaves the marks of a
    /// profile placed by assignment as they are.
    #[test]
    fn the_keys_mark_keeps_a_scope_to_open() {
        let keyed = Marks(Marks::SCOPED).with_key();
        assert!(keyed.scoped() && keyed.keyed_scope());
        assert!(!keyed.opens_first() && !keyed.asks_nothing_first());
        let bare = Marks(Marks::DRAWS_FIRST).with_key();
        assert!(!bare.scoped() && !bare.keyed_scope());
        assert_eq!(bare.ahead(), Marks::DRAWS_FIRST);
        assert!(!bare.asks_nothing_first());
        let placed = Marks(Marks::SCOPED | Marks::BY_ASSIGNMENT);
        assert_eq!(placed.with_key(), placed);
    }

    /// Counts a scope's opens and closes.
    #[derive(Debug, Default)]
    struct Scope {
        opened: AtomicUsize,
        closed: AtomicUsize,
    }

    impl DecisionScope for Scope {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
            self.opened.fetch_add(1, Ordering::Relaxed);
        }
        fn close(&self) {
            self.closed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Keeps every row and holds a scope.
    #[derive(Debug)]
    struct Scoped(Arc<Scope>);

    impl Filter for Scoped {
        fn name(&self) -> &'static str {
            "scoped"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self.0.as_ref())
        }
    }

    /// The ring's profile built by hand with a scope to open and the
    /// picker given, marked as the validator marks the ring's profile.
    fn scoped_ring(picker: Box<dyn Picker>) -> (Engine, Arc<Scope>) {
        let scope = Arc::new(Scope::default());
        let mut engine = Engine::new(
            vec![Profile {
                name: "ring".into(),
                filters: vec![
                    Box::new(NotOverloaded),
                    Box::new(Scoped(Arc::clone(&scope))),
                ],
                scorers: vec![Weighted {
                    scorer: Box::new(ConsistentHash::new(None)),
                    weight: 1.0,
                }],
                picker,
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        engine.places_by_key(0);
        (engine, scope)
    }

    /// The key's mark keeps the profile's scope: each decision opens and
    /// closes it once, a keyed request's lifted run and a request with no
    /// key alike.
    #[test]
    fn a_profile_under_the_keys_mark_opens_its_scope_once() {
        let (engine, scope) = scoped_ring(Box::new(Best));
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let owner = order(&decide(&engine, &keyed("k1"), &pool))[0];
        pool[owner as usize].health.overloaded = true;
        assert_eq!(
            order(&decide(&engine, &keyed("k1"), &pool))[0],
            owner,
            "the lift"
        );
        assert!(decide(&engine, &facts(1), &pool).pick().is_some());
        assert_eq!(scope.opened.load(Ordering::Relaxed), 3);
        assert_eq!(scope.closed.load(Ordering::Relaxed), 3);
    }

    /// A profile under the key's mark that shows its rows declines what
    /// the scope's own arm declines: a mask or a pin beside the rows says
    /// they were not written for the decision, and the key's arm opens no
    /// scope over them, for a request with a key and for one with none.
    #[test]
    fn a_profile_under_the_keys_mark_declines_rows_not_written_for_it() {
        let (engine, scope) = scoped_ring(Box::new(Best));
        let pool = [free(0, 4), free(1, 4)];
        let key = keyed("k1");
        let none = facts(1);
        let pinned = RequestFacts {
            pinned_at: 1,
            ..key
        };
        let masked = Constraints {
            stalled: &[0],
            ..Constraints::default()
        };
        let written = Constraints::default();
        let run = |facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
            engine.decide(facts, &PoolFacts::default(), &pool, cx, &mut Scratch::new())
        };
        for (facts, cx, beside) in [
            (&key, &masked, "a mask"),
            (&pinned, &written, "a pin"),
            (&none, &masked, "a mask and no key"),
        ] {
            assert_eq!(
                shed_of(&run(facts, cx)),
                (ShedReason::PolicyDeclined, WRITTEN_ROWS),
                "{beside}"
            );
        }
        assert_eq!(scope.opened.load(Ordering::Relaxed), 0);
        assert!(run(&key, &written).pick().is_some());
        assert_eq!(scope.opened.load(Ordering::Relaxed), 1);
    }

    /// A picker that leaves the placement to its rule takes no mark: its
    /// profile's scope is opened as it is without a ring.
    #[test]
    fn a_profile_placed_by_assignment_takes_no_mark() {
        let (engine, scope) = scoped_ring(Box::new(Manual));
        let pool = [free(0, 4), free(1, 4)];
        assert!(decide(&engine, &keyed("k1"), &pool).pick().is_some());
        assert_eq!(scope.opened.load(Ordering::Relaxed), 1);
        assert_eq!(scope.closed.load(Ordering::Relaxed), 1);
    }

    /// A pool over on every worker keeps no pin standing, and the ring's
    /// own worker takes a key under a rule there: the lifted profile answers
    /// where the all-vetoed rule answers a profile with no ring. A pool the
    /// lifted pass leaves nobody is shed as it is under no rule.
    #[test]
    fn a_key_under_a_rule_goes_to_the_rings_worker_over_a_pool_over_on_every_worker() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let ringed = engine_of("consistent_hashing", false);
        let mut pool = [over(0, 4), over(1, 4), over(2, 4), over(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        // The fewest in flight is not the key's worker.
        pool[ring[2] as usize].in_flight = 1;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0]);
        for worker in &mut pool {
            worker.health.healthy = false;
        }
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        assert_eq!(
            shed_of(&decide(&engine, &request, &pool)),
            shed_of(&decide(&ringed, &request, &pool))
        );
    }

    /// Two decisions of one key agree before either is dispatched: the
    /// second reads the first one's claim as a pin. With the ring's worker
    /// over its thresholds that pin yields, and the ring sends the key to
    /// the same worker again: the decision follows the pin it read and
    /// claims nothing, so only the first decision's own dispatch elsewhere
    /// withdraws the claim.
    #[test]
    fn a_second_decision_follows_the_first_ones_claim_under_the_ring() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let request = RequestFacts {
            now_micros: 1_000,
            ..keyed("k1")
        };
        let owner = pool[ring[0] as usize].id;
        let verdicts: Vec<_> = [false, false, true, true]
            .into_iter()
            .map(|over_its_thresholds| {
                pool[ring[0] as usize].health.overloaded = over_its_thresholds;
                let decision = decide(&engine, &request, &pool);
                let pick = *decision.pick().expect("a pick");
                assert_eq!(
                    pick.targets.first().map(|target| target.position),
                    Some(ring[0])
                );
                pick.session_verdict
            })
            .collect();
        let (claimed, pinned) = (SessionVerdict::Claimed(owner), SessionVerdict::Pinned);
        assert_eq!(verdicts, [claimed, pinned, pinned, pinned]);
    }

    /// A concurrent decision, staged: the engine it runs on and the pool it
    /// sees.
    #[derive(Debug, Default)]
    struct Gap {
        engine: OnceLock<Weak<Engine>>,
        staged: Mutex<Vec<Candidate>>,
    }

    /// Keeps every row. The first time it is asked under the lifted veto it
    /// runs the request once more, on the same engine and over the staged
    /// pool, lent the pins of record its caller was lent: another
    /// decision's whole run, in the gap between a decision's read of the
    /// pins and its own claim.
    #[derive(Debug)]
    struct InTheGap(Arc<Gap>);

    impl Filter for InTheGap {
        fn name(&self) -> &'static str {
            "in_the_gap"
        }
        fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, cx: &Constraints<'_>) -> bool {
            if cx.overload_lifted {
                let staged = std::mem::take(&mut *self.0.staged.lock().expect("the staged pool"));
                if let Some(engine) = self.0.engine.get().and_then(Weak::upgrade) {
                    if !staged.is_empty() {
                        let lent = Constraints {
                            pins: cx.pins,
                            ..Constraints::default()
                        };
                        let mut scratch = Scratch::new();
                        let _ = engine.decide(
                            facts,
                            &PoolFacts::default(),
                            &staged,
                            &lent,
                            &mut scratch,
                        );
                    }
                }
            }
            true
        }
    }

    /// The ring's profile built by hand with `last` behind the health stage
    /// and the overload veto, marked as the validator marks the ring's
    /// profile, under `rule`.
    fn ring_behind(last: Box<dyn Filter>, rule: Option<SessionRule>) -> Engine {
        let mut engine = Engine::new(
            vec![Profile {
                name: "ring".into(),
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded), last],
                scorers: vec![Weighted {
                    scorer: Box::new(ConsistentHash::new(None)),
                    weight: 1.0,
                }],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: rule,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        engine.places_by_key(0);
        engine
    }

    /// [`ring_behind`] with [`InTheGap`] as that filter.
    fn ring_with_a_gap(rule: Option<SessionRule>) -> (Arc<Engine>, Arc<Gap>) {
        let gap = Arc::new(Gap::default());
        let engine = Arc::new(ring_behind(Box::new(InTheGap(Arc::clone(&gap))), rule));
        gap.engine
            .set(Arc::downgrade(&engine))
            .expect("the engine is set once");
        (engine, gap)
    }

    /// The claim's lead reads the survivors the placement ran over. Another
    /// decision of the key, which sees the key's worker down, claims the
    /// ring's next worker between this decision's read of the pins and its
    /// own claim. This decision sees that worker over its thresholds: the
    /// lifted pass keeps it, and the decision follows the claim onto it.
    /// Read over the workers the standing veto kept, the lead would not
    /// find the claimed worker, and the two decisions would part.
    #[test]
    fn a_claim_made_in_the_gap_is_followed_onto_a_worker_the_lift_keeps() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 0,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let (engine, gap) = ring_with_a_gap(Some(rule));
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let mut theirs = pool.to_vec();
        theirs[ring[0] as usize].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        pool[ring[1] as usize].health.overloaded = true;
        let decision = decide(&engine, &keyed("k1"), &pool);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "the other decision ran"
        );
        assert_eq!(order(&decision)[..2], [ring[1], ring[0]]);
        assert_eq!(
            decision.pick().expect("a pick").session_verdict,
            SessionVerdict::Pinned,
            "the claim of record is the other decision's"
        );
    }

    /// The validator marks the profile that holds the ring wherever it
    /// stands in the document, and no other. A keyed request of the ring's
    /// binding keeps its worker over the veto; one of a binding whose
    /// profile scores without the ring meets the veto, though the worker
    /// over its thresholds holds the fewest requests.
    #[test]
    fn the_mark_lands_on_the_rings_profile_and_no_other() {
        let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
        let scored = PolicySpec::preset("least_load").expect("the preset's form");
        // First by name, so the ring's profile is not the document's first.
        spec.profiles
            .insert("a_least_load".into(), scored.profiles["least_load"].clone());
        let mut binding = spec
            .bindings
            .first()
            .cloned()
            .expect("the preset's binding");
        binding.matcher.model = "scored".into();
        binding.profile = Some("a_least_load".into());
        spec.bindings.insert(0, binding);
        let engine = spec.validate().expect("the document builds");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let owner = ring_order(&engine, &keyed("k1"), &pool)[0];
        pool[owner as usize].health.overloaded = true;
        pool[owner as usize].in_flight = 1;
        assert_eq!(order(&decide(&engine, &keyed("k1"), &pool))[0], owner);
        let other = RequestFacts {
            model: "scored",
            ..keyed("k1")
        };
        assert_ne!(order(&decide(&engine, &other, &pool))[0], owner);
    }

    /// The ring's preset with its one profile's scorers and picker replaced,
    /// under the prefix hash preset's view, which admits the prefix form.
    fn ring_shaped(score: serde_json::Value, pick: serde_json::Value) -> Engine {
        let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
        spec.view = PolicySpec::preset("prefix_hash")
            .expect("the preset's form")
            .view;
        for profile in spec.profiles.values_mut() {
            profile.score = serde_json::from_value(score.clone()).expect("a score list");
            profile.pick = serde_json::from_value(pick.clone()).expect("a picker");
        }
        spec.validate().expect("the document builds")
    }

    /// The mark is the plain ring's only where the ring alone decides the
    /// pick: the profile's only scorer, with no bound, under the picker that
    /// takes the best score. A ring under a bound, in its prefix form,
    /// beside another scorer, under a picker that turns or under one that
    /// samples takes no mark: a keyed request of such a profile meets the
    /// veto, and its key's worker over its thresholds is never its pick,
    /// though that worker holds the fewest requests.
    #[test]
    fn a_ring_that_does_not_decide_the_pick_alone_takes_no_mark() {
        use serde_json::json;
        let plain = json!(["consistent-hash"]);
        let shapes = [
            ("the preset's own shape", plain.clone(), json!("best"), true),
            (
                "a bound",
                json!([{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}]),
                json!("best"),
                false,
            ),
            (
                "the prefix form",
                json!([{"consistent-hash": {"prefix_tokens": 256}}]),
                json!("best"),
                false,
            ),
            (
                "another scorer beside it",
                json!(["consistent-hash", "least-load"]),
                json!("best"),
                false,
            ),
            (
                "a picker that turns",
                plain.clone(),
                json!("round-robin"),
                false,
            ),
            (
                "a picker that samples",
                plain,
                json!({"best": {"sample": 8}}),
                false,
            ),
        ];
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let owner = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool)[0];
        pool[owner as usize].health.overloaded = true;
        pool[owner as usize].in_flight = 0;
        for (shape, score, pick, marked) in shapes {
            let engine = ring_shaped(score, pick);
            for seed in 1..=8 {
                let request = RequestFacts {
                    seed,
                    ..keyed("k1")
                };
                let first = order(&decide(&engine, &request, &pool))[0];
                assert_eq!(first == owner, marked, "{shape}, seed {seed}");
            }
        }
    }

    /// A ring under a bound keeps the veto for a keyed request. Its owner
    /// past the bound is passed over and no row is ranked, so the picker
    /// draws among the survivors: a worker over its thresholds is not one
    /// of them, whatever it holds, and the pick says no lifted veto.
    #[test]
    fn a_bounded_rings_draw_leaves_out_a_worker_over_its_thresholds() {
        use serde_json::json;
        let engine = ring_shaped(
            json!([{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}]),
            json!("best"),
        );
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        // The owner holds requests past the bound; the ring's next member is
        // over its thresholds and holds the fewest.
        pool[ring[0] as usize].in_flight = 20;
        pool[ring[1] as usize].health.overloaded = true;
        pool[ring[1] as usize].in_flight = 2;
        for seed in 1..=32 {
            let request = RequestFacts {
                seed,
                ..keyed("k1")
            };
            let decision = decide(&engine, &request, &pool);
            assert_ne!(order(&decision)[0], ring[1], "seed {seed}");
            assert_eq!(decision.trace().lifted, Lifted::None, "seed {seed}");
        }
    }

    /// A kept key the lifted pass puts back on its pin follows that pin and
    /// claims nothing. The pin over its thresholds yields under the veto as
    /// it stands, the ring names the same worker with the veto lifted, and
    /// the pin stands as it was read: when the host dispatches the decision
    /// to its alternate, the key's worker is still its first pin, and the
    /// key returns there (the manual preset's rule recovers the original).
    #[test]
    fn a_kept_key_put_back_on_its_pin_claims_nothing() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        // The key is served on its ring worker: the pin is confirmed.
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        // The pin is over its thresholds beside free workers.
        pool[owner].health.overloaded = true;
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let decision = decide(&engine, &request, &pool);
        let pick = *decision.pick().expect("a pick");
        assert_eq!(order(&decision)[..2], ring[..2], "the ring's order");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned, "no claim");
        // The host dispatches the decision to its alternate.
        let alternate = *pick.targets.iter().nth(1).expect("an alternate");
        let lease = Lease {
            issued_micros: 2_000,
            session: request.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &alternate, Side::Single, 0);
        // The worker is under its thresholds again: the key is still its.
        pool[owner].health.overloaded = false;
        assert_eq!(dispatched(&engine, &pool, 3_000), ring[0], "the pin stood");
    }

    /// A kept key the lifted pass puts back on its pin of record follows
    /// that pin as it follows a pin the table holds. The table holds no pin
    /// of the key, the host's record names the key's ring worker, and that
    /// worker is over its thresholds: the read at a miss passes it over
    /// under the veto as it stands, the ring names it with the veto lifted,
    /// and the pin is written back with no claim. When the host dispatches
    /// the decision to its alternate, the worker is still the key's first
    /// pin, and the key returns there.
    #[test]
    fn a_kept_key_put_back_on_its_pin_of_record_claims_nothing() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        // The table holds nothing of the key. The record names its worker,
        // over its thresholds beside free workers.
        let record = Stored::of(request.session_key.hash, pool[owner].id);
        pool[owner].health.overloaded = true;
        let decision = engine.decide(
            &request,
            &PoolFacts::default(),
            &pool,
            &Constraints {
                pins: Some(&record),
                ..Constraints::default()
            },
            &mut Scratch::new(),
        );
        let pick = *decision.pick().expect("a pick");
        assert_eq!(order(&decision)[..2], ring[..2], "the ring's order");
        let verdict = pick.session_verdict;
        // The host dispatches the decision to its alternate.
        let alternate = *pick.targets.iter().nth(1).expect("an alternate");
        let lease = Lease {
            issued_micros: 2_000,
            session: request.session(),
            session_verdict: verdict,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &alternate, Side::Single, 0);
        // The worker is under its thresholds again: the key is still its.
        pool[owner].health.overloaded = false;
        let back = dispatched(&engine, &pool, 3_000);
        assert_eq!(
            (verdict, back),
            (SessionVerdict::Pinned, ring[0]),
            "no claim, and the pin stood"
        );
    }

    /// The rule's in-flight cap holds for a pin the lifted pass lands on as
    /// it holds for a pin read first: the lift is of the overload veto, and
    /// of no other limit. A pin that carries the cap's requests of the key
    /// puts the request to the ring again. Under its thresholds the ring
    /// places the key on its pin. Over them the veto as it stands drops
    /// the pin, and the ring's next worker takes the key. With every
    /// worker over its thresholds the key stays on its pin, by the lift.
    #[test]
    fn a_followed_pin_at_the_rules_cap_puts_the_key_to_the_ring_again() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let verdict = |decision: &Decision| decision.pick().expect("a pick").session_verdict;
        // Under the cap a pin over its thresholds keeps the key.
        pool[owner].session_in_flight = 1;
        pool[owner].health.overloaded = true;
        let kept = decide(&engine, &request, &pool);
        assert_eq!(order(&kept)[0], ring[0], "under the cap");
        assert_eq!(verdict(&kept), SessionVerdict::Pinned);
        // The pin carries two requests of the key, the rule's cap.
        pool[owner].session_in_flight = 2;
        pool[owner].health.overloaded = false;
        // Under its thresholds the ring places the key on its pin again.
        let under = decide(&engine, &request, &pool);
        assert_eq!(order(&under)[0], ring[0], "under its thresholds");
        assert_eq!(verdict(&under), SessionVerdict::Pinned);
        // Over them the veto as it stands drops the pin: the ring's next
        // worker takes the key.
        pool[owner].health.overloaded = true;
        let over = decide(&engine, &request, &pool);
        assert_eq!(order(&over)[0], ring[1], "over its thresholds");
        assert_eq!(verdict(&over), SessionVerdict::Respill);
        assert_eq!(over.trace().lifted, Lifted::None);
        // Every worker over its thresholds: the lifted pass places the key
        // on its pin again, and the placement stands by the lift.
        for candidate in &mut pool {
            candidate.health.overloaded = true;
        }
        let all = decide(&engine, &request, &pool);
        assert_eq!(order(&all)[0], ring[0], "every worker over");
        assert_eq!(verdict(&all), SessionVerdict::Pinned);
        assert_eq!(all.trace().lifted, Lifted::Overload);
        // Every worker at the cap as well: the key stays on its pin.
        for candidate in &mut pool {
            candidate.session_in_flight = 2;
        }
        let stays = decide(&engine, &request, &pool);
        assert_eq!(order(&stays)[0], ring[0], "every worker at the cap");
        assert_eq!(verdict(&stays), SessionVerdict::Pinned);
    }

    /// The rule's in-flight cap holds for a pin of record the lifted pass
    /// lands on as it holds for a pin the table held. The record names the
    /// key's ring worker, over its thresholds and carrying the cap's
    /// requests of the key: the pin is written back and hands the key to
    /// the ring's next worker, where a claim would have kept it.
    #[test]
    fn a_followed_pin_of_record_at_the_rules_cap_hands_the_key_on() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let record = Stored::of(request.session_key.hash, pool[owner].id);
        pool[owner].health.overloaded = true;
        pool[owner].session_in_flight = 2;
        let decision = engine.decide(
            &request,
            &PoolFacts::default(),
            &pool,
            &Constraints {
                pins: Some(&record),
                ..Constraints::default()
            },
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision)[0], ring[1], "handed on at the cap");
        assert_eq!(
            decision.pick().expect("a pick").session_verdict,
            SessionVerdict::Respill
        );
        assert_eq!(decision.trace().lifted, Lifted::None);
    }

    /// A pin of record on another worker than the ring's stands through a
    /// pool over its thresholds on every worker. The ring's worker takes
    /// the request with the veto lifted, a claim behind the record's pin
    /// written back, and once the pool is under its thresholds the key
    /// returns to its recorded worker (the manual preset's rule recovers
    /// the original).
    #[test]
    fn a_pin_of_record_off_the_rings_worker_stands_through_a_pool_over_on_every_worker() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [over(0, 4), over(1, 4), over(2, 4), over(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let recorded = pool[ring[1] as usize].id;
        let record = Stored::of(request.session_key.hash, recorded);
        let lent = Constraints {
            pins: Some(&record),
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let over_all = engine.decide(&request, &PoolFacts::default(), &pool, &lent, &mut scratch);
        assert_eq!(
            order(&over_all)[0],
            ring[0],
            "the ring's worker, by the lift"
        );
        assert_eq!(over_all.trace().lifted, Lifted::Overload);
        assert_eq!(
            engine
                .sessions()
                .pins(request.session())
                .map(|pins| pins[0]),
            Some(recorded),
            "the record's pin is the key's first"
        );
        // The pool is under its thresholds again: the key is its record's.
        for worker in &mut pool {
            worker.health.overloaded = false;
        }
        let later = RequestFacts {
            now_micros: 3_000,
            ..keyed("k1")
        };
        let back = engine.decide(&later, &PoolFacts::default(), &pool, &lent, &mut scratch);
        assert_eq!(order(&back)[0], ring[1], "the recorded worker");
        assert_eq!(
            back.pick().expect("a pick").session_verdict,
            SessionVerdict::Pinned
        );
    }

    /// A pin of record on a worker the lifted pass drops as well (it is not
    /// ready) is passed over, as the plain form passes it: nothing is
    /// written back, and the ring's worker takes the key as a fresh one.
    #[test]
    fn a_pin_of_record_the_lifted_pass_drops_is_passed_over() {
        let engine = ring_under_a_rule(Assignment::Delegate);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let down = ring[1] as usize;
        let record = Stored::of(request.session_key.hash, pool[down].id);
        pool[down].health.healthy = false;
        let lent = Constraints {
            pins: Some(&record),
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(&request, &PoolFacts::default(), &pool, &lent, &mut scratch);
        let taker = pool[ring[0] as usize].id;
        assert_eq!(order(&decision)[0], ring[0], "the ring's worker");
        assert_eq!(
            decision.pick().expect("a pick").session_verdict,
            SessionVerdict::Claimed(taker)
        );
        assert_eq!(
            engine
                .sessions()
                .pins(request.session())
                .map(|pins| pins[0]),
            Some(taker),
            "nothing is written back"
        );
    }

    /// The write-back does not take another request's claim for this
    /// decision's own read. Another decision of the key, which sees the
    /// recorded worker down, claims the ring's next worker in the gap
    /// between this decision's read of the pins and its write-back. The
    /// door then writes nothing, and the pins that stand are not this
    /// decision's: its claim sees the move and follows the other's worker.
    /// Taken for its own, they would let it claim the recorded worker over
    /// the standing claim, and the two decisions would part.
    #[test]
    fn a_claim_made_ahead_of_the_write_back_is_followed() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 0,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let (engine, gap) = ring_with_a_gap(Some(rule));
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let request = keyed("k1");
        let recorded = ring[0] as usize;
        let record = Stored::of(request.session_key.hash, pool[recorded].id);
        let mut theirs = pool.to_vec();
        theirs[recorded].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        pool[recorded].health.overloaded = true;
        let lent = Constraints {
            pins: Some(&record),
            ..Constraints::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(&request, &PoolFacts::default(), &pool, &lent, &mut scratch);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "the other decision ran"
        );
        assert_eq!(order(&decision)[..2], [ring[1], ring[0]]);
        assert_eq!(
            decision.pick().expect("a pick").session_verdict,
            SessionVerdict::Pinned,
            "the claim of record is the other decision's"
        );
    }

    /// A claim made behind the record is not this decision's own read
    /// either. Another decision of the key, lent the same record, writes
    /// it back first and claims the ring's next worker behind it: it sees
    /// the ring's first worker down. Where the rule recovers the original
    /// the record still leads the pins that stand. This decision's sight
    /// is the record alone, so its claim sees the pin behind it and
    /// follows the other's worker. Taken whole, the pins would let it
    /// claim a second worker, and the record's pin would be dropped.
    #[test]
    fn a_claim_made_behind_the_record_is_followed() {
        for recovery in [Recovery::Original, Recovery::Latest] {
            let rule = SessionRule {
                assignment: Assignment::Delegate,
                inflight_cap: 0,
                ttl_micros: None,
                sweep_micros: 0,
                recovery,
            };
            let (ringed, _) = ring_with_a_gap(None);
            let (engine, gap) = ring_with_a_gap(Some(rule));
            let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
            let ring = ring_order(&ringed, &keyed("k1"), &pool);
            let request = keyed("k1");
            // The record names the ring's third worker, over its thresholds
            // in both views: each decision's lifted pass keeps it.
            let recorded = ring[2] as usize;
            let record = Stored::of(request.session_key.hash, pool[recorded].id);
            pool[recorded].health.overloaded = true;
            let mut theirs = pool.to_vec();
            theirs[ring[0] as usize].health.healthy = false;
            *gap.staged.lock().expect("the staged pool") = theirs;
            let lent = Constraints {
                pins: Some(&record),
                ..Constraints::default()
            };
            let mut scratch = Scratch::new();
            let decision =
                engine.decide(&request, &PoolFacts::default(), &pool, &lent, &mut scratch);
            assert!(
                gap.staged.lock().expect("the staged pool").is_empty(),
                "the other decision ran"
            );
            assert_eq!(order(&decision)[..2], [ring[1], ring[0]], "{recovery:?}");
            assert_eq!(
                decision.pick().expect("a pick").session_verdict,
                SessionVerdict::Pinned,
                "{recovery:?}: the claim of record is the other decision's"
            );
            let pins = engine.sessions().pins(request.session()).expect("the pins");
            assert!(pins.contains(&pool[recorded].id), "{recovery:?}: {pins:?}");
            assert!(
                pins.contains(&pool[ring[1] as usize].id),
                "{recovery:?}: {pins:?}"
            );
        }
    }

    /// The ring's scorer. The first time it is asked it runs the request
    /// once more, on the same engine and over the staged pool: another
    /// decision's whole run between a decision's read of the pins and its
    /// own claim, under a profile that lifts no veto.
    #[derive(Debug)]
    struct ScoresInTheGap(Arc<Gap>, ConsistentHash);

    impl Lifecycle for ScoresInTheGap {}

    impl Scorer for ScoresInTheGap {
        fn name(&self) -> &'static str {
            "scores_in_the_gap"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let staged = std::mem::take(&mut *self.0.staged.lock().expect("the staged pool"));
            if let Some(engine) = self.0.engine.get().and_then(Weak::upgrade) {
                if !staged.is_empty() {
                    let mut scratch = Scratch::new();
                    let _ = engine.decide(
                        facts,
                        &PoolFacts::default(),
                        &staged,
                        &Constraints::default(),
                        &mut scratch,
                    );
                }
            }
            self.1.score(facts, candidates, kept, out)
        }
    }

    /// The ring's stages built by hand under `rule`, with [`ScoresInTheGap`]
    /// as the scorer and no mark of the validator's: a keyed request of
    /// this profile meets the veto as it stands.
    fn plain_ring_with_a_gap(rule: SessionRule) -> (Arc<Engine>, Arc<Gap>) {
        let gap = Arc::new(Gap::default());
        let engine = Arc::new(Engine::new(
            vec![Profile {
                name: "ring".into(),
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                scorers: vec![Weighted {
                    scorer: Box::new(ScoresInTheGap(Arc::clone(&gap), ConsistentHash::new(None))),
                    weight: 1.0,
                }],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: Some(rule),
                overload: None,
            }],
            1,
            Needs::default(),
        ));
        gap.engine
            .set(Arc::downgrade(&engine))
            .expect("the engine is set once");
        (engine, gap)
    }

    /// One decision of the key `k1` with another's in its gap, the pair on
    /// an engine of its own: the other sees the ring's first worker down
    /// and claims the ring's second. `shape` sets this decision's view of
    /// four free workers. The lead, the verdict and the trace's lift.
    fn followed(
        (engine, gap): (Arc<Engine>, Arc<Gap>),
        ring: &[u32],
        shape: impl Fn(&mut [Candidate]),
    ) -> (u32, SessionVerdict, Lifted) {
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let mut theirs = pool.to_vec();
        theirs[ring[0] as usize].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        shape(&mut pool);
        let decision = decide(&engine, &keyed("k1"), &pool);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "the other decision ran"
        );
        let verdict = decision.pick().expect("a pick").session_verdict;
        (order(&decision)[0], verdict, decision.trace().lifted)
    }

    /// Another request's claim is a pin of the key: the rule's in-flight
    /// cap holds for it as for the pin read first, whether the decision
    /// placed the key on that worker or follows the claim onto it. Under
    /// the cap the decision follows the claim onto a worker over its
    /// thresholds.
    /// Where that worker carries the cap's requests of the key, the request
    /// is put to the ring again by the veto as it stands. A survivor under
    /// the cap takes the key. A survivor at the cap takes the request and
    /// no pin moves, though the lifted pass would keep a worker under the
    /// cap. Placed on the claimed worker again, the key stays on the claim.
    #[test]
    fn a_followed_claim_at_the_rules_cap_hands_the_key_on() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 2,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let (claimed, last) = (ring[1] as usize, ring[3] as usize);
        // This decision sees the claimed worker over its thresholds, with
        // `held` requests of the key: the lifted pass alone keeps it.
        let at = |held: u16| {
            move |pool: &mut [Candidate]| {
                pool[claimed].health.overloaded = true;
                pool[claimed].session_in_flight = held;
            }
        };
        let (pinned, handed) = (SessionVerdict::Pinned, SessionVerdict::Respill);
        let none = SessionVerdict::None;
        let under = followed(ring_with_a_gap(Some(rule)), &ring, at(1));
        assert_eq!(under, (ring[1], pinned, Lifted::None), "under the cap");
        let on = followed(ring_with_a_gap(Some(rule)), &ring, at(2));
        assert_eq!(on, (ring[0], handed, Lifted::None), "at the cap");
        // Every worker the standing veto keeps carries the cap as well.
        // The ring's last, over its thresholds, is under it.
        let lifted = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
            pool[claimed].health.overloaded = true;
            pool[last].health.overloaded = true;
            pool[last].session_in_flight = 0;
        });
        assert_eq!(lifted, (ring[0], none, Lifted::None), "no veto is lifted");
        let stays = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
            pool[claimed].health.overloaded = true;
        });
        assert_eq!(stays, (ring[0], none, Lifted::None), "no worker under it");
        // This decision sees the ring's first worker down as the other
        // does: its own placement is the claimed worker.
        let same = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
            pool[ring[0] as usize].health.healthy = false;
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(same, (ring[1], pinned, Lifted::None), "placed on the claim");
    }

    /// The plain form follows a claim made in its gap under the same cap.
    /// The followed worker is among the survivors the standing veto kept.
    /// At the cap the ring places the request again. A survivor under the
    /// cap takes the key. One at the cap takes the request and no pin
    /// moves, though a worker over its thresholds is under the cap: this
    /// form lifts no veto.
    #[test]
    fn a_followed_claim_at_the_cap_hands_the_key_on_under_the_standing_veto() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 2,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let (claimed, last) = (ring[1] as usize, ring[3] as usize);
        let (pinned, handed) = (SessionVerdict::Pinned, SessionVerdict::Respill);
        let none = SessionVerdict::None;
        let under = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            pool[claimed].session_in_flight = 1;
        });
        assert_eq!(under, (ring[1], pinned, Lifted::None), "under the cap");
        let on = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(on, (ring[0], handed, Lifted::None), "at the cap");
        let stays = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
        });
        assert_eq!(stays, (ring[0], none, Lifted::None), "no survivor under it");
        // The ring's last is under the cap and over its thresholds.
        let refused = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
            pool[last].health.overloaded = true;
            pool[last].session_in_flight = 0;
        });
        assert_eq!(refused, (ring[0], none, Lifted::None), "no veto is lifted");
        // This decision sees the ring's first worker down as the other
        // does: its own placement is the claimed worker.
        let same = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            pool[ring[0] as usize].health.healthy = false;
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(same, (ring[1], pinned, Lifted::None), "placed on the claim");
    }

    /// A key handed on at the cap reads the veto as it stands: from a pin
    /// over its thresholds it passes the ring's next worker where that
    /// worker is over its thresholds beside free ones. Where the free
    /// workers carry the cap too, the first of them still takes the
    /// request and no pin moves: the lifted pass's survivors decide only
    /// where the standing veto keeps no worker.
    #[test]
    fn a_key_handed_on_at_the_cap_passes_a_worker_over_its_thresholds() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let (owner, next) = (ring[0] as usize, ring[1] as usize);
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        // The pin carries the cap's two requests of the key, and the ring's
        // next worker is over its thresholds beside two free ones.
        pool[owner].session_in_flight = 2;
        pool[next].health.overloaded = true;
        // Under its thresholds the ring places the key on its pin again.
        let under = decide(&engine, &request, &pool);
        assert_eq!(order(&under)[0], ring[0], "the pin under its thresholds");
        pool[owner].health.overloaded = true;
        let over = decide(&engine, &request, &pool);
        assert_eq!(order(&over)[0], ring[2], "the pin over its thresholds");
        assert_eq!(over.trace().lifted, Lifted::None);
        // The free workers carry the cap too. The veto as it stands still
        // keeps them: the ring's first of them takes the request, no veto
        // is lifted, and no pin moves to a worker at the cap.
        pool[ring[2] as usize].session_in_flight = 2;
        pool[ring[3] as usize].session_in_flight = 2;
        let standing = decide(&engine, &request, &pool);
        assert_eq!(order(&standing)[0], ring[2], "a free worker at the cap");
        let pick = standing.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None);
        assert_eq!(standing.trace().lifted, Lifted::None);
    }

    /// What a filter that turns was asked, and whether it turned.
    #[derive(Debug, Default)]
    struct Turn {
        asked: AtomicUsize,
        turned: AtomicBool,
    }

    /// A filter whose answer turns within a decision. It keeps every row
    /// until it is asked under the lifted veto, and from then on it refuses
    /// every row it is asked about under the veto as it stands.
    #[derive(Debug)]
    struct Turns(Arc<Turn>);

    impl Filter for Turns {
        fn name(&self) -> &'static str {
            "turns"
        }
        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, cx: &Constraints<'_>) -> bool {
            self.0.asked.fetch_add(1, Ordering::Relaxed);
            if cx.overload_lifted {
                self.0.turned.store(true, Ordering::Relaxed);
            }
            cx.overload_lifted || !self.0.turned.load(Ordering::Relaxed)
        }
    }

    /// The hand-on's pass under the veto as it stands is put to no rule. It
    /// runs where the same pass kept a worker earlier in the decision, so
    /// it ends empty only under a filter whose answer turned since. The
    /// lifted pass's survivors then decide and the trace says the lift, and
    /// the emptied pass is not run again with the veto lifted: the filter
    /// is asked about the rows of the decision's four passes and no more.
    #[test]
    fn a_hand_ons_pass_a_filter_emptied_is_put_to_no_rule() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 2,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
        };
        let turn = Arc::new(Turn::default());
        let engine = ring_behind(Box::new(Turns(Arc::clone(&turn))), Some(rule));
        let (ringed, _) = ring_with_a_gap(None);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        // The pin carries the cap's two requests of the key, over its
        // thresholds: the lifted pass lands on it and puts the request to
        // the ring again.
        pool[owner].session_in_flight = 2;
        pool[owner].health.overloaded = true;
        turn.turned.store(false, Ordering::Relaxed);
        turn.asked.store(0, Ordering::Relaxed);
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let decision = decide(&engine, &request, &pool);
        // The lifted pass's survivors hold the pin, and the ring places the
        // key on it again.
        assert_eq!(order(&decision)[0], ring[0]);
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        assert_eq!(decision.trace().lifted, Lifted::Overload);
        // Two passes under the veto as it stands reach the filter with the
        // three workers under their thresholds, two lifted passes with all.
        assert_eq!(turn.asked.load(Ordering::Relaxed), 2 * 3 + 2 * 4);
    }

    /// A kept key leaves its worker at the rule's cap alone, so a capped
    /// rule keeps no key where the host writes no count of the key's
    /// requests: the veto stands for the key as for any request, and a pin
    /// over its thresholds yields it to the ring's next worker. The same
    /// request with its count written is kept, and a rule with no cap reads
    /// no count.
    #[test]
    fn a_capped_rule_keeps_no_key_where_the_host_writes_no_count() {
        let free_pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(
            &engine_of("consistent_hashing", false),
            &keyed("k1"),
            &free_pool,
        );
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        pool[ring[0] as usize].health.overloaded = true;
        // The key's pin on its ring worker, then a request of the key with
        // that worker over its thresholds beside three free ones.
        let asked = |engine: &Engine, counted: bool| {
            assert_eq!(dispatched(engine, &free_pool, 1_000), ring[0], "the pin");
            let request = RequestFacts {
                now_micros: 2_000,
                session_counted: counted,
                ..keyed("k1")
            };
            let decision = decide(engine, &request, &pool);
            (order(&decision)[0], decision.trace().lifted)
        };
        assert_eq!(
            asked(&ring_under_a_cap(2), false),
            (ring[1], Lifted::None),
            "no count: the veto stands for the key"
        );
        assert_eq!(
            asked(&ring_under_a_cap(2), true),
            (ring[0], Lifted::None),
            "the count written: the key is kept"
        );
        assert_eq!(
            asked(&ring_under_a_rule(Assignment::Delegate), false),
            (ring[0], Lifted::None),
            "no cap: no count is read"
        );
    }

    /// A profile the engine does not hold cannot be marked: the index is
    /// the validator's own, and a debug build refuses any other.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "is not built"))]
    fn a_profile_the_engine_does_not_hold_is_refused_in_a_debug_build() {
        let mut engine = engine_of("consistent_hashing", false);
        engine.places_by_key(7);
    }

    /// A ready worker under its thresholds that the liveness veto doubts.
    fn doubted(position: u32, in_flight: u32) -> Candidate {
        let mut worker = free(position, in_flight);
        worker.health.stalled = true;
        worker
    }

    /// A ready worker over its thresholds that the liveness veto doubts.
    fn doubted_over(position: u32, in_flight: u32) -> Candidate {
        let mut worker = over(position, in_flight);
        worker.health.stalled = true;
        worker
    }

    /// A pool doubted on every ready worker goes to the fewest requests in
    /// flight, under every preset and under either word of the binding: a
    /// doubt is not an outage.
    #[test]
    fn a_pool_doubted_on_every_worker_goes_to_the_fewest_in_flight() {
        let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
        for preset in [
            "random",
            "round_robin",
            "passthrough",
            "power_of_two",
            "least_load",
        ] {
            for shed in [false, true] {
                let decision = decide(&engine_of(preset, shed), &facts(1), &pool);
                assert_eq!(order(&decision), vec![1, 2, 0], "{preset}, shed {shed}");
                let pick = decision.pick().expect("a pick");
                assert_eq!(pick.session_verdict, SessionVerdict::None, "{preset}");
            }
        }
    }

    /// The overload veto is lifted first: a worker over its thresholds
    /// alone takes the request before any doubted one, whatever it holds.
    /// With no such worker the doubted ones are taken by their load, over
    /// their thresholds or not.
    #[test]
    fn a_worker_over_alone_is_taken_before_the_doubted_ones() {
        let engine = engine_of("least_load", false);
        let pool = [doubted_over(0, 0), doubted(1, 3), over(2, 5)];
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2]);
        let pool = [doubted_over(0, 2), doubted(1, 1)];
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![1, 0]);
        let pool = [doubted_over(0, 0), doubted(1, 5)];
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![0, 1]);
    }

    /// A binding that sheds keeps the overload veto in the lifted pass: a
    /// doubted worker under its thresholds takes the request, a doubted one
    /// over them does not, and a pool over on every ready worker is shed as
    /// saturated, doubted or not.
    #[test]
    fn a_binding_that_sheds_takes_a_doubted_worker_under_its_thresholds() {
        let engine = engine_of("least_load", true);
        let pool = [over(0, 0), doubted(1, 3), doubted(2, 1)];
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2, 1]);
        let pool = [doubted_over(0, 0), doubted(1, 3)];
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![1]);
        for pool in [
            vec![over(0, 3), doubted_over(1, 1)],
            vec![doubted_over(0, 3), doubted_over(1, 1), doubted_over(2, 2)],
        ] {
            assert_eq!(
                shed_of(&decide(&engine, &facts(1), &pool)),
                (ShedReason::Saturated, "not-overloaded")
            );
        }
    }

    /// A worker that is down, draining or behind an open breaker is not
    /// usable with both vetoes lifted: it is never the steer, and a pool of
    /// such workers alone is shed at the health stage.
    #[test]
    fn a_doubted_pool_never_steers_to_a_worker_another_veto_refuses() {
        let engine = engine_of("least_load", false);
        let mut pool = [doubted(0, 0), doubted(1, 0), doubted(2, 0), doubted(3, 7)];
        pool[0].health.healthy = false;
        pool[1].health.draining = true;
        pool[2].health.breaker_open = true;
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![3]);
        pool[3].health.healthy = false;
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "healthy")
        );
    }

    /// Equal loads among doubted workers are drawn by the decision's seed,
    /// over the tied rows alone.
    #[test]
    fn a_tie_among_the_doubted_is_drawn_by_the_seed() {
        let engine = engine_of("round_robin", false);
        let pool = [doubted(0, 2), doubted(1, 2), doubted(2, 5)];
        let mut firsts = [0u32; 3];
        for seed in 1..=600 {
            let first = order(&decide(&engine, &facts(seed), &pool))[0];
            firsts[first as usize] += 1;
        }
        assert_eq!(firsts[2], 0, "{firsts:?}");
        assert!(firsts[0] > 200 && firsts[1] > 200, "{firsts:?}");
    }

    /// A host's own lift of the overload veto ends the overload tier alone:
    /// a doubted pool under it is still steered. The lifted pass is as its
    /// host made it, so under a binding that sheds a host's lift stands
    /// there too, and a doubted worker over its thresholds is taken.
    #[test]
    fn a_hosts_lift_leaves_the_liveness_tier_its_own() {
        let pool = [doubted_over(0, 3), doubted_over(1, 1), doubted(2, 2)];
        let lifted = Constraints {
            overload_lifted: true,
            ..Constraints::default()
        };
        for shed in [false, true] {
            let decision = engine_of("least_load", shed).decide(
                &facts(1),
                &PoolFacts::default(),
                &pool,
                &lifted,
                &mut Scratch::new(),
            );
            assert_eq!(order(&decision), vec![1, 2, 0], "shed {shed}");
        }
        // With no lift of the host's, a binding that sheds keeps the veto.
        let shedding = engine_of("least_load", true);
        assert_eq!(order(&decide(&shedding, &facts(1), &pool)), vec![2]);
    }

    /// Affinity outlives a load signal, not a doubt. A key whose ring
    /// worker is doubted passes to the ring's next member, over its
    /// thresholds or not. With every worker doubted the key goes to the
    /// fewest requests in flight, whichever worker the ring names, and no
    /// pin is written: the key is its ring's again when the doubt ends.
    /// Under a binding that sheds, a doubted ring worker under its
    /// thresholds takes its key once every other worker is over.
    #[test]
    fn a_key_with_every_worker_doubted_goes_to_the_fewest_in_flight() {
        let engine = engine_of("consistent_hashing", false);
        let request = keyed("k1");
        let mut pool = [free(0, 5), free(1, 4), free(2, 3), free(3, 2)];
        let ring = ring_order(&engine, &request, &pool);
        let (owner, next) = (ring[0] as usize, ring[1] as usize);
        pool[owner].health.stalled = true;
        let first = order(&decide(&engine, &request, &pool))[0];
        assert_eq!(first, ring[1], "beside free workers");
        for worker in &mut pool {
            worker.health.overloaded = !worker.health.stalled;
        }
        let first = order(&decide(&engine, &request, &pool))[0];
        assert_eq!(first, ring[1], "beside workers over their thresholds");
        for worker in &mut pool {
            worker.health.overloaded = false;
            worker.health.stalled = true;
        }
        pool[owner].in_flight = 9;
        pool[next].in_flight = 8;
        let fewest = (0..pool.len())
            .min_by_key(|&i| pool[i].in_flight)
            .expect("a worker");
        let decision = decide(&engine, &request, &pool);
        assert_eq!(order(&decision)[0] as usize, fewest, "every worker doubted");
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None);

        let shedding = engine_of("consistent_hashing", true);
        let mut pool = [over(0, 5), over(1, 4), over(2, 3), over(3, 2)];
        let ring = ring_order(&shedding, &request, &pool);
        let owner = ring[0] as usize;
        pool[owner].health.overloaded = false;
        pool[owner].health.stalled = true;
        assert_eq!(order(&decide(&shedding, &request, &pool)), vec![ring[0]]);
    }

    /// A shed no doubt had a part in is not filtered again for the liveness
    /// veto: a pool with no doubted row, and a doubted pool emptied before
    /// the health stage ran, cost the one pass they cost without the tier.
    /// A doubted pool behind the health stage costs the tier's one pass
    /// more, and its shed names the stage that refused the rows with no
    /// doubt on them.
    #[test]
    fn a_shed_no_doubt_had_a_part_in_costs_no_further_pass() {
        // No doubted row: the health stage keeps both and the next refuses.
        let pool = [free(0, 1), free(1, 2)];
        let (engine, asked) = counted(Box::new(Healthy), true);
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
        // Doubted rows refused ahead of the health stage: it never ran.
        let pool = [doubted(0, 1), doubted(1, 2)];
        let (engine, asked) = counted(Box::new(Healthy), false);
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
        // Doubted rows behind it: the first pass ends at the health stage
        // and the tier's pass asks the next stage about each row once.
        let (engine, asked) = counted(Box::new(Healthy), true);
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    }

    /// The rule is asked once a decision: its own passes are put to no rule.
    /// A pool over and doubted on every worker ends its first pass at the
    /// overload veto, and the overload lift's pass at the health stage. The
    /// liveness tier's pass alone reaches the filter listed behind both,
    /// and asks it about each row once.
    #[test]
    fn the_rules_own_passes_are_put_to_no_rule() {
        let pool = [doubted_over(0, 1), doubted_over(1, 2)];
        let (engine, asked) =
            counted_among(|refuses| vec![Box::new(NotOverloaded), Box::new(Healthy), refuses]);
        assert_eq!(
            shed_of(&decide(&engine, &facts(1), &pool)),
            (ShedReason::NoEndpoints, "refuses")
        );
        assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    }

    /// A side of a pair is put to the liveness tier no more than to the
    /// overload tier: a prefill side doubted on every worker is shed at the
    /// health stage, and the filter listed behind that stage is asked about
    /// no prefill.
    #[test]
    fn a_pairs_side_doubted_on_every_worker_sheds_at_the_health_stage() {
        let none: (&[u32], &[u32]) = (&[], &[]);
        let side = pairs_side_shed(Box::new(Healthy), [doubted(0, 1), doubted(1, 2)], none);
        assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), 0));
    }

    /// A hold changes nothing on a side of a pair, which is put to no tier.
    /// A prefill side the host holds on every row, and one with a doubted
    /// row among held ones, are shed at the health stage, and the filter
    /// listed behind that stage is asked about no prefill. No tier reads a
    /// side's held rows: a rule put to a pair's side brings them with it,
    /// in the same change.
    #[test]
    fn a_pairs_side_under_a_hold_sheds_at_the_health_stage() {
        for held in [&[0, 1][..], &[0]] {
            let masks: (&[u32], &[u32]) = (&[0, 1], held);
            let side = pairs_side_shed(Box::new(Healthy), [free(0, 1), free(1, 2)], masks);
            assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), 0));
        }
    }

    /// The liveness tier follows the pass the overload lift ended. Where a
    /// document lists the overload filter ahead of the health stage, a pool
    /// over and doubted on every worker ends its first pass before the
    /// health stage ran; the overload lift's pass ends at the health stage,
    /// and the tier takes it from there. A binding that sheds lifts no
    /// overload veto, and that pool is shed as saturated.
    #[test]
    fn the_liveness_tier_follows_the_pass_the_overload_lift_ended() {
        let filters = ["not-overloaded", "healthy", "not-tried"];
        let pool = [doubted_over(0, 3), doubted_over(1, 1), doubted_over(2, 2)];
        let steering = engine_filtering(&filters, false);
        assert_eq!(order(&decide(&steering, &facts(1), &pool)), vec![1, 2, 0]);
        let shedding = engine_filtering(&filters, true);
        assert_eq!(
            shed_of(&decide(&shedding, &facts(1), &pool)),
            (ShedReason::Saturated, "not-overloaded")
        );
    }

    /// The retry's tried list does not narrow the liveness tier either: a
    /// doubted worker an earlier attempt tried is taken again when it holds
    /// the fewest requests, whether its own bit doubts it or the host's
    /// mask. A worker the host just failed to reach is passed over while
    /// another doubted one stands.
    #[test]
    fn a_doubted_worker_tried_is_taken_again_when_it_holds_the_fewest() {
        let engine = engine_of("round_robin", true);
        let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
        let tried = [WorkerId(101)];
        let retry = RequestFacts {
            tried: &tried,
            attempt: 1,
            ..facts(1)
        };
        assert_eq!(order(&decide(&engine, &retry, &pool)), vec![1, 2, 0]);
        // The pass that leaves the host's mask out sets the list aside too.
        let masked = Constraints {
            stalled: &[0, 1, 2],
            ..Constraints::default()
        };
        let bare = [free(0, 3), free(1, 1), free(2, 2)];
        let decision = engine.decide(
            &retry,
            &PoolFacts::default(),
            &bare,
            &masked,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision), vec![1, 2, 0]);
        let excluded = Constraints {
            excluded: &tried,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &excluded,
            &mut Scratch::new(),
        );
        assert_eq!(order(&decision), vec![2, 0]);
    }

    /// The pass that lifts the liveness veto hands its rows back to the
    /// scratch: the next doubted decision finds them sized.
    #[test]
    fn the_lifted_pass_hands_its_rows_back_to_the_scratch() {
        let engine = engine_of("least_load", true);
        let pool = [doubted(0, 3), doubted(1, 1), doubted(2, 2)];
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        assert_eq!(order(&decision), vec![1, 2, 0]);
        assert!(scratch.single.undoubted.capacity() >= pool.len());
    }

    /// The trace says which veto the rule lifted: the overload veto for a
    /// worker over its thresholds alone, the liveness veto for a doubted
    /// one, under either word of the binding, and none for a pick no lift
    /// made.
    #[test]
    fn a_steered_pick_says_which_veto_was_lifted() {
        let mark =
            |engine: &Engine, pool: &[Candidate]| decide(engine, &facts(1), pool).trace().lifted;
        let steering = engine_of("least_load", false);
        assert_eq!(
            mark(&steering, &[over(0, 1), doubted(1, 0)]),
            Lifted::Overload
        );
        assert_eq!(
            mark(&steering, &[doubted(0, 1), doubted_over(1, 0)]),
            Lifted::Liveness
        );
        assert_eq!(mark(&steering, &[free(0, 1), doubted(1, 0)]), Lifted::None);
        let shedding = engine_of("least_load", true);
        assert_eq!(
            mark(&shedding, &[over(0, 1), doubted(1, 3)]),
            Lifted::Liveness
        );
    }

    /// A doubt the host hands beside the rows is the same doubt: a pool its
    /// mask covers whole is steered as one the rows' own bits doubt, and no
    /// row is copied for it. A row with its own bit beside masked rows is
    /// read with no doubt too, and a row the mask leaves out stands with no
    /// lift.
    #[test]
    fn a_pool_the_hosts_mask_doubts_is_steered_too() {
        let engine = engine_of("least_load", true);
        let ask = |pool: &[Candidate], stalled: &[u32], scratch: &mut Scratch| {
            let masked = Constraints {
                stalled,
                ..Constraints::default()
            };
            engine.decide(&facts(1), &PoolFacts::default(), pool, &masked, scratch)
        };
        let mut scratch = Scratch::new();
        let decision = ask(
            &[free(0, 3), free(1, 1), free(2, 2)],
            &[0, 1, 2],
            &mut scratch,
        );
        assert_eq!(order(&decision), vec![1, 2, 0]);
        assert_eq!(decision.trace().lifted, Lifted::Liveness);
        assert_eq!(scratch.single.undoubted.capacity(), 0, "no row is copied");
        let decision = ask(
            &[free(0, 3), doubted(1, 1), free(2, 2)],
            &[0, 2],
            &mut scratch,
        );
        assert_eq!(order(&decision), vec![1, 2, 0]);
        assert_eq!(decision.trace().lifted, Lifted::Liveness);
        let decision = ask(&[free(0, 0), free(1, 5)], &[0], &mut scratch);
        assert_eq!(order(&decision), vec![1]);
        assert_eq!(decision.trace().lifted, Lifted::None);
    }

    /// A row a hold of the host stands on is not doubted: the tier lifts the
    /// host's mask and leaves its held rows masked. A pool held on every
    /// row is shed at the health stage under either word of the binding, as
    /// with no tier. One doubted row among held ones takes the request
    /// whatever it holds. So it does where the host wrote its rows: there a
    /// held row carries the breaker bit beside the stalled one.
    #[test]
    fn a_row_the_host_holds_stays_refused_through_the_liveness_tier() {
        let pool = [free(0, 0), free(1, 1), free(2, 9), free(3, 2)];
        let ask = |engine: &Engine, pool: &[Candidate], stalled: &[u32], held: &[u32]| {
            let masked = Constraints {
                stalled,
                held,
                ..Constraints::default()
            };
            let mut scratch = Scratch::new();
            engine.decide(
                &facts(1),
                &PoolFacts::default(),
                pool,
                &masked,
                &mut scratch,
            )
        };
        let mut written = pool;
        for row in &mut written {
            row.health.stalled = true;
            row.health.breaker_open = row.position != 2;
        }
        for shed in [false, true] {
            let engine = engine_of("least_load", shed);
            let whole = ask(&engine, &pool, &[0, 1, 2, 3], &[0, 1, 2, 3]);
            assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
            for one in [
                ask(&engine, &pool, &[0, 1, 2, 3], &[0, 1, 3]),
                ask(&engine, &written, &[], &[]),
            ] {
                assert_eq!(order(&one), vec![2], "shed {shed}");
                assert_eq!(one.trace().lifted, Lifted::Liveness);
            }
        }
    }

    /// Keeps every row, and writes down what a stage that knows no lift
    /// reads of each: whether the row's stalled bit or the host's mask
    /// refuses it.
    #[derive(Debug)]
    struct Shown(Arc<Mutex<Vec<bool>>>);
    impl Filter for Shown {
        fn name(&self) -> &'static str {
            "shown"
        }
        fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
            let mut shown = self.0.lock().expect("no reader panics");
            shown.push(row.health.stalled || cx.masks(row.position));
            true
        }
    }

    /// A stage of a host's own that reads the stalled bit and the mask, and
    /// knows no lift, refuses a held row in every pass: the first, and the
    /// lifted one, where a doubted row reads clear. Rows the host wrote and
    /// rows it lent beside its lists show the stage the same.
    #[test]
    fn a_held_row_reads_as_vetoed_to_a_stage_in_every_pass() {
        let shown = Arc::new(Mutex::new(Vec::new()));
        let stage: Box<dyn Filter> = Box::new(Shown(Arc::clone(&shown)));
        let (engine, _) = counted_among(|_| vec![stage, Box::new(Healthy)]);
        let lent = [free(0, 0), free(1, 1)];
        let mut written = lent;
        written[0].health.stalled = true;
        written[0].health.breaker_open = true;
        written[1].health.stalled = true;
        let hosts: [(&[Candidate], &[u32], &[u32]); 2] =
            [(&lent, &[0, 1], &[0]), (&written, &[], &[])];
        for (pool, stalled, held) in hosts {
            let masked = Constraints {
                stalled,
                held,
                ..Constraints::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide(
                &facts(1),
                &PoolFacts::default(),
                pool,
                &masked,
                &mut scratch,
            );
            assert_eq!(order(&decision), vec![1]);
            assert_eq!(decision.trace().lifted, Lifted::Liveness);
            // Row by row: the first pass, then the lifted one.
            let passes = std::mem::take(&mut *shown.lock().expect("no reader panics"));
            assert_eq!(passes, [true, true, true, false]);
        }
    }

    /// The tier runs where it would gain a row and nowhere else: over a mask
    /// of held rows alone it makes no pass. A filter listed ahead of the
    /// health stage is asked about each row once there, and once more by the
    /// tier's pass where one row of the mask is doubted.
    #[test]
    fn a_mask_of_held_rows_alone_costs_no_further_pass() {
        let pool = [free(0, 0), free(1, 1), free(2, 9), free(3, 2)];
        let asked = Arc::new(AtomicUsize::new(0));
        let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
        let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
        let ask = |held: &[u32]| {
            let masked = Constraints {
                stalled: &[0, 1, 2, 3],
                held,
                ..Constraints::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide(
                &facts(1),
                &PoolFacts::default(),
                &pool,
                &masked,
                &mut scratch,
            );
            (decision, asked.swap(0, Ordering::Relaxed))
        };
        let (whole, passes) = ask(&[0, 1, 2, 3]);
        assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
        assert_eq!(passes, pool.len());
        let (one, passes) = ask(&[0, 1, 3]);
        assert_eq!(order(&one), vec![2]);
        assert_eq!(passes, 2 * pool.len());
    }

    /// A doubt on a row a harder veto refuses too opens no tier: lifting it
    /// gains nobody. A pool with no ready worker but the doubted one, which
    /// is not ready, draining or behind an open breaker, costs the one pass
    /// it costs with no doubt in it, by the host's mask and by the row's own
    /// bit alike. A doubt alone on a ready row costs the tier's pass, and
    /// that row takes the request.
    #[test]
    fn a_doubt_on_a_row_a_harder_veto_refuses_costs_no_further_pass() {
        let asked = Arc::new(AtomicUsize::new(0));
        let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
        let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
        let ask = |pool: &[Candidate], stalled: &[u32]| {
            let masked = Constraints {
                stalled,
                ..Constraints::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide(
                &facts(1),
                &PoolFacts::default(),
                pool,
                &masked,
                &mut scratch,
            );
            (decision, asked.swap(0, Ordering::Relaxed))
        };
        // No row is ready but the second, which each case refuses its way.
        let pool_of = |refuse: fn(&mut Health)| {
            let mut pool = [free(0, 0), free(1, 0), free(2, 0)];
            pool[0].health.healthy = false;
            pool[2].health.healthy = false;
            refuse(&mut pool[1].health);
            pool
        };
        let harder: [fn(&mut Health); 3] = [
            |health| health.healthy = false,
            |health| health.draining = true,
            |health| health.breaker_open = true,
        ];
        for refuse in harder {
            let mut pool = pool_of(refuse);
            let (masked, passes) = ask(&pool, &[1]);
            assert_eq!(shed_of(&masked), (ShedReason::NoEndpoints, "healthy"));
            assert_eq!(passes, pool.len());
            pool[1].health.stalled = true;
            let (own, passes) = ask(&pool, &[]);
            assert_eq!(shed_of(&own), (ShedReason::NoEndpoints, "healthy"));
            assert_eq!(passes, pool.len());
        }
        let mut pool = pool_of(|_| {});
        let (masked, passes) = ask(&pool, &[1]);
        assert_eq!(order(&masked), vec![1]);
        assert_eq!(passes, 2 * pool.len());
        pool[1].health.stalled = true;
        let (own, passes) = ask(&pool, &[]);
        assert_eq!(order(&own), vec![1]);
        assert_eq!(passes, 2 * pool.len());
    }

    /// A kept key in a pool doubted on every worker is steered to the fewest
    /// requests in flight, under a sessions rule too: the pick says the
    /// liveness veto and claims no pin. So it does where every worker is
    /// over its thresholds as well: the lift that kept a row there is the
    /// liveness veto's, and the kept run's own mark does not write over it.
    #[test]
    fn a_kept_key_in_a_doubted_pool_is_steered_and_claims_nothing() {
        let request = keyed("k1");
        let under = [doubted(0, 5), doubted(1, 4), doubted(2, 1), doubted(3, 2)];
        let over = [
            doubted_over(0, 5),
            doubted_over(1, 4),
            doubted_over(2, 1),
            doubted_over(3, 2),
        ];
        for pool in [under, over] {
            for engine in [
                engine_of("consistent_hashing", false),
                ring_under_a_rule(Assignment::Delegate),
            ] {
                let decision = decide(&engine, &request, &pool);
                assert_eq!(order(&decision)[0], 2);
                assert_eq!(decision.trace().lifted, Lifted::Liveness);
                let pick = decision.pick().expect("a pick");
                assert_eq!(pick.session_verdict, SessionVerdict::None);
            }
        }
    }

    /// A kept key reaches the liveness tier over a pool whose host lists its
    /// engines as over one that lists none. The ring's walk places the key
    /// while a row it reads stands. With every worker doubted the walk
    /// keeps no row and the whole pass decides: the key goes to the fewest
    /// requests in flight by the lift of the liveness veto and claims no
    /// pin, with the workers under their thresholds and over them.
    #[test]
    fn a_kept_key_in_a_doubted_pool_is_steered_past_the_rings_walk() {
        let engine = engine_of("consistent_hashing", false);
        let request = keyed("k1");
        let (listed, ends) = ([0, 1, 2, 3], [1, 2, 3, 4]);
        let listing = PoolFacts {
            pool_size: 4,
            membership_epoch: 1,
            generation: 1,
            engines: Engines::new(0, &listed, &ends),
            ..PoolFacts::default()
        };
        let decide = |pool: &[Candidate]| {
            let cx = Constraints::default();
            engine.decide(&request, &listing, pool, &cx, &mut Scratch::new())
        };
        // No worker doubted: the walk alone decides, and counts no survivor.
        let calm = [free(0, 5), free(1, 4), free(2, 1), free(3, 2)];
        let walked = decide(&calm);
        assert!(walked.trace().survivors.get().is_none(), "by the walk");
        let under = [doubted(0, 5), doubted(1, 4), doubted(2, 1), doubted(3, 2)];
        let over = [
            doubted_over(0, 5),
            doubted_over(1, 4),
            doubted_over(2, 1),
            doubted_over(3, 2),
        ];
        for pool in [under, over] {
            let decision = decide(&pool);
            assert_eq!(order(&decision)[0], 2);
            assert_eq!(decision.trace().lifted, Lifted::Liveness);
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.session_verdict, SessionVerdict::None);
        }
    }

    /// An encode item of the `epd` plan is put to the liveness tier as it is
    /// put to the overload tier, behind the plan's own rules: encoders
    /// doubted on every worker take it by the fewest requests in flight, and
    /// an encoder the pair cannot use stays unusable.
    #[test]
    fn an_encode_item_over_doubted_encoders_is_steered_among_those_the_plan_admits() {
        let engine = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin")
            .expect("the presets' form")
            .validate()
            .expect("the presets build");
        let mut pool = [doubted(0, 0), doubted(1, 4), doubted(2, 2)];
        pool[1].ec_compatible = true;
        pool[2].ec_compatible = true;
        let item = RequestFacts {
            phase: Phase::Encode,
            ..facts(1)
        };
        let decision = decide(&engine, &item, &pool);
        assert_eq!(order(&decision), vec![2, 1]);
        assert_eq!(decision.trace().lifted, Lifted::Liveness);
    }
}
