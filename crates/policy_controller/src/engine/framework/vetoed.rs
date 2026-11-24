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
//! which veto it lifted ([`LiftedVeto`]).
//!
//! A pair's decode side is put to the rule: its set carries the pair
//! binding's word ([`Set::ruled`]), so partners whose every usable worker
//! is vetoed are steered or shed as a single request's pool is, over their
//! own rows. A pair's first decision asks the rule for the overload veto
//! alone, and its prefill side is put to no rule. A pair whose pool still
//! fails, vetoed on every usable worker, is paired by load ([`paired`]).
//!
//! A profile that places a request by its session key reads the same word
//! ahead of its pass ([`keeps_its_key`]): where the binding steers, a keyed
//! request is decided with the veto lifted ([`Engine::keep`]), so the key
//! keeps its worker while that worker is over its thresholds. A sessions
//! rule with an in-flight cap keeps the key only where the host holds the
//! cap's count ([`RequestFacts::session_counted`]). Under a sessions rule
//! the key's pin is asked first, with the veto as it stands, and the lifted
//! pass places a key no pin takes.
//!
//! A pair's prefill profile that weighs load against affinity reads the
//! word the same way ([`Engine::weighing_lift`]): where the binding steers,
//! its run has the veto lifted, so a session keeps the prefill that holds
//! its prefix while that prefill is over its thresholds.
//!
//! The mark of such a lifted run has one meaning. Where the veto as it
//! stands leaves the pool nobody ([`all_vetoed`]), the pick exists by the
//! lift, and it says the lifted veto in its trace as the rule's own pick
//! does ([`LiftedVeto::Overload`]): a kept key's pick, and a pair placed by
//! its weighing prefill profile. A host counts them all from the one mark.
//! A pick beside a worker under its thresholds carries no mark.

use super::{
    classed, engines::run_owned, filter, keep_folded, leads, places, reason_of, run_single, shed,
    single_emptied, trace_of, Assignment, Binding, Buffers, Engine, Filter, PairTable, Picker,
    Plan, Profile, Scorer, Scratch, SessionRule, Set,
};
use crate::{
    engine::{
        decision::{Decision, LiftedVeto, Pick, Survivors, Targets, Trace},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Health, PoolFacts, RequestFacts, Role},
    },
    stages::{
        filters::{HasCapacity, Healthy, NotOverloaded, NotTried},
        pickers::Best,
        scorers::LeastRequests,
    },
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
    /// emptied it. A set's word, never a binding's. The rule's own passes
    /// carry it ([`Set::shedding`]), and so does a pass of a pair that says
    /// no binding's word ([`Set::side`]): a prefill side's, one that only
    /// names a shed's stage, a shadow's run, an encode item's.
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
    /// mark the rule's own pick carries ([`LiftedVeto::Overload`]), under a
    /// rule and under none.
    ///
    /// `KEEPS` is the keeping entry's form ([`Engine::decide_keeping`]):
    /// under a rule the decision also keeps what it read of the key.
    ///
    /// Out of line: a kept key is the rare decision, and the engine's
    /// decision path keeps its code for every other.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    pub(super) fn keep<const KEEPS: bool>(
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
            return self.decide_session::<true, KEEPS>(
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
        buf.lifted = LiftedVeto::None;
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
            if buf.lifted == LiftedVeto::None && all_vetoed(profile, facts, set, cx, buf) {
                pick.trace.lifted = LiftedVeto::Overload;
            }
        }
        decision
    }
}

/// Whether a keyed request keeps its key's worker over the overload veto:
/// a request of the single plan that carries a session key, under a binding
/// that steers ([`AllVetoed::Steer`]). A decision asks it for a profile the
/// plain ring alone places ([`Engine::places_by_key`]) over a pool of
/// regular workers ([`regular`]) and then
/// decides the request with the veto lifted ([`Engine::keep`]). The key's
/// worker takes the request while it is over its thresholds, beside free
/// workers too, as a session keeps its owner. A binding that sheds keeps
/// the veto: an owner over its thresholds passes the key on, and a pool
/// over on every worker is shed. So does a sessions rule whose own
/// assignment places a key no pin takes: no stage is asked where that key
/// goes, and nothing is lifted. So does a rule with an in-flight cap under
/// a host that holds no count of the key's requests
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

/// Whether every candidate the request may take is a regular worker. A key
/// keeps its worker over the overload veto among regular workers alone, as
/// the gateway's own placement lifts the veto for a key: a worker of
/// another role over its thresholds passes the key on.
pub(super) fn regular(candidates: &[Candidate], cx: &Constraints<'_>) -> bool {
    places::rows(candidates, cx).all(|row| row.role == Role::Single)
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

/// Whether the overload veto as it stands leaves nobody in a pool a lifted
/// run just placed a request in: a kept key's pool ([`Engine::keep`]), or
/// a pair's prefills under a profile that weighs load against affinity
/// ([`Engine::decide_pair`]). The lifted pass just placed the request, or
/// the ring's walk did: `buf` holds that pass's survivors, or the rows the
/// walk kept, each one of them, and `cx` is the decision's own, the veto
/// standing in it. A survivor under its thresholds stands without the
/// lift, so the pool was not emptied. The filters run again with the veto
/// standing only over a pool whose survivors are all over: a soft filter of
/// the lifted pass may have passed over a worker under its thresholds. That
/// pass overwrites `buf`, and the caller reads nothing of it afterwards.
pub(super) fn all_vetoed(
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
/// filter listed behind a veto does not hide such a pool. The retry's tried
/// list is set aside for those passes, as the gateway's fallback sets it
/// aside: a worker an earlier attempt tried is taken again when it holds
/// the fewest requests. On a retry one pass reads the list first: the one
/// that lifts the overload veto over a chosen prefill's partners. The
/// profile's soft filter then keeps to the partners the request has not
/// tried while one is left, as the gateway's pair selection does. That pass
/// stands where it keeps a partner, unless that filter stood aside in it.
/// The partners it keeps then fail the filter's own test, each tried or
/// one the host just failed to reach, and the pass with the list set aside
/// decides, as it does where the pass keeps nobody. Under a profile that
/// lists no such filter the pass stands as its hard filters leave it. A
/// request that names no tried worker takes the pass with the list set
/// aside alone. The pair presets list the soft filter behind the room
/// filter, the order the gateway narrows in. The host's list of workers it
/// just failed to reach stays in every pass: the soft filter passes over
/// such a worker while another usable one stands, and takes it as the last
/// resort.
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
/// the rule's own passes, so no pass of the rule asks it again, and a
/// pair's passes that say no binding's word. A single request asks the
/// rule once. A pair asks it of its decode side: over the decode pool where
/// no decode is open, then over the chosen prefill's partners. A pool nobody
/// is usable in with its vetoes lifted is shed at the stage that says why.
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
        let pass = lifted_cx(cx, true, false);
        // On a retry a chosen prefill's partners keep to the ones the
        // request has not tried while one is left: their pass reads the
        // tried list first. It stands where it keeps a partner, unless the
        // profile's soft filter stood aside in it: the partners it keeps
        // then fail that filter's own test. Under a profile that lists the
        // filter they pass it all or none, so `all` and `any` are one
        // answer here. A profile that lists no such filter has none that
        // stood aside. With no tried list the pass is the one below, and
        // is not run.
        let untried = |&at: &u32| NotTried.keep(facts, &candidates[at as usize], &pass);
        let listed = || profile.filters.iter().any(|f| f.name() == NotTried.name());
        if set.class.is_some()
            && !facts.tried.is_empty()
            && filter(profile, facts, &pass, buf, set.shedding()).is_none()
            && (!listed() || buf.kept.iter().all(untried))
        {
            buf.lifted = LiftedVeto::Overload;
            return emptied_by;
        }
        let whole = RequestFacts {
            tried: &[],
            ..*facts
        };
        match overload_pass(profile, &whole, &pass, buf, set.shedding()) {
            None => {
                buf.lifted = LiftedVeto::Overload;
                return emptied_by;
            }
            Some(stage) => ended = stage,
        }
    }
    // A doubt is a row's own bit or the host's mask beside the rows, and
    // the tier opens for a row the doubt alone refuses. A doubt on a row a
    // harder veto refuses too gains nobody when it is lifted. Nor does a
    // row a hold of the host stands on, whichever doubt it carries: it
    // stays masked, and both tests pass it by.
    let own = candidates
        .iter()
        .any(|row| doubted_alone(row.health) && cx.held.binary_search(&row.position).is_err());
    // The two lengths answer for the rows only where every held row is a
    // row of the mask, named once: a host that fills the constraints
    // owes it.
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
    let pass = lifted_cx(cx, set.vetoed == AllVetoed::Steer, true);
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
            buf.lifted = LiftedVeto::Liveness;
            emptied_by
        }
        Some(stage) => stage,
    }
}

/// The constraints of a pass that lifts a veto: the overload veto where the
/// binding steers (`steers`) or the host lifted it, and with `doubt` the
/// liveness veto, the host's mask cut to the rows a hold stands on.
fn lifted_cx<'a>(cx: &Constraints<'a>, steers: bool, doubt: bool) -> Constraints<'a> {
    Constraints {
        overload_lifted: cx.overload_lifted || steers,
        stalled: if doubt { cx.held } else { cx.stalled },
        ..*cx
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
/// its trace says the veto the rule lifted for those rows ([`LiftedVeto`], as
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

/// One side of a pair as the pair's default reads it ([`paired`]): the
/// side's profile and its index, the request as the side's stages read it,
/// the side's rows and the constraints over them.
#[derive(Clone, Copy)]
pub(super) struct Half<'a> {
    profile: &'a Profile,
    index: u16,
    facts: &'a RequestFacts<'a>,
    rows: &'a [Candidate],
    cx: &'a Constraints<'a>,
    /// Whether the admission limit is set aside where a row is judged: the
    /// prefill side's, whose limit is asked behind the pick.
    limit_aside: bool,
}

impl<'a> Half<'a> {
    pub(super) const fn new(
        profile: &'a Profile,
        index: u16,
        facts: &'a RequestFacts<'a>,
        rows: &'a [Candidate],
        cx: &'a Constraints<'a>,
    ) -> Self {
        Self {
            profile,
            index,
            facts,
            rows,
            cx,
            limit_aside: false,
        }
    }

    /// Whether the side's hard filters keep `row` under `cx`, on the
    /// request's own facts. On the prefill side the admission limit is set
    /// aside: it is asked of the one prefill the default takes, behind its
    /// pick. No soft filter is asked: a room or the built-in filter of tried
    /// workers steers no default. The row's class is not asked: a row
    /// outside any pairing stands in its pool, and pairs with nobody
    /// ([`paired`]).
    fn stands(&self, row: &Candidate, cx: &Constraints<'_>) -> bool {
        let hard = self.profile.filters.iter().filter(|f| !f.soft());
        let mut asked = hard.filter(|f| !(self.limit_aside && f.name() == HasCapacity.name()));
        asked.all(|f| f.keep(self.facts, row, cx))
    }

    /// The side's rows the request may take ([`places::rows`]).
    pub(super) fn places(&self) -> impl Iterator<Item = &'a Candidate> {
        places::rows(self.rows, self.cx)
    }

    /// Whether `row` stands but for the side's admission limit: the side's
    /// hard filters keep it with the limit set aside, under the side's own
    /// constraints. A pair's decision judges the limit over the prefills
    /// that stand so and have an open partner, and names a refusal by the
    /// partners of every prefill that stands so ([`Engine::decide_pair`]).
    pub(super) fn but_for_the_limit(&self, row: &Candidate) -> bool {
        let aside = Self {
            limit_aside: true,
            ..*self
        };
        aside.stands(row, aside.cx)
    }

    /// The side's rows that stand in the pass that lifts `lifted`
    /// ([`lifted_cx`]), into `buf.kept`, the mark left in `buf.lifted`. The
    /// set opens with the places the side's host states, as every pass of
    /// the rule does ([`places::open`]). With the liveness veto lifted, rows
    /// that carry a doubt are read with none ([`undoubted`]).
    fn usable(&self, lifted: LiftedVeto, steers: bool, buf: &mut Buffers) {
        let doubt = lifted == LiftedVeto::Liveness;
        let pass = lifted_cx(self.cx, steers, doubt);
        // The rows are written again only where one carries a doubt.
        let own = doubt && self.rows.iter().any(|row| row.health.stalled);
        let mut rows = std::mem::take(&mut buf.undoubted);
        if own {
            undoubted(self.rows, &mut rows);
        }
        let read = if own { &rows[..] } else { self.rows };
        buf.kept.clear();
        places::open(&mut buf.kept, read.len(), self.cx);
        buf.kept
            .retain(|&at| self.stands(&read[at as usize], &pass));
        buf.undoubted = rows;
        buf.lifted = lifted;
    }
}

/// The pair's default: the pair a refused pool goes to, where every usable
/// worker of the pool that failed is vetoed. It is the steering default of
/// the gateway's own pair selection
/// (`model_gateway/src/routers/common/placement.rs:1084-1172` on main). Each
/// usable prefill stands with the usable partner that holds the fewest
/// requests, and the pair of the least load is taken: the prefill's load
/// first, then its partner's. Equal pairs are drawn by the prefill side's
/// seed, and equal partners by the decode side's. Pairs the overload veto
/// alone refuses come first, under a binding that steers. The liveness veto
/// is lifted only where nothing else pairs, and never onto a worker over
/// its thresholds under a binding that sheds. A hold stays in the
/// constraints of every pass ([`lifted_cx`]). A side's workers are those
/// at the places its host states, in each pass and where its pool is judged
/// (`:774-786`, `:1097-1104`). Every pass reads the request's own facts. No
/// built-in hard filter reads the retry's tried list, so the default takes
/// a tried worker again, as the gateway's does. The pair's trace is the
/// prefill's: it says the veto lifted and counts the prefills its pass kept.
///
/// Asked where the prefill side's own run shed ([`Engine::decide_pair`]),
/// with the prefill classes that have an open partner in `scratch.classes`.
/// It answers for a prefill pool no worker of which stands (`:1395-1413`
/// there) and for a decode pool no worker of which stands (`:1438-1465`).
/// It answers `None`, and the caller's refusal stands, where:
/// - a prefill with an open partner stands with its limit aside, under the
///   constraints its profile's own run read. Under a prefill profile that
///   reads affinity (`weighs`) and a binding that steers, that run had the
///   overload veto lifted ([`Engine::weighing_lift`]), so a prefill over
///   its thresholds alone stands there (`:1327-1346`): the run's own
///   refusal is the answer, its admission limit's among them;
/// - a prefill and a decode both stand as they are, with no lift: the
///   prefills fail for want of a partner, and the decode pool is not vetoed
///   on every worker. A weighing profile's pool is judged so too, as it was
///   before the lift (`:1439-1447`).
///   The prefills judged are those with a decode to pair with anywhere in
///   the decode pool, in or outside the request's places: the gateway's
///   list of them is the pool's own and no request's
///   (`model_gateway/src/routers/common/pair_index.rs:99-103`, `:143-146`).
///   The decodes are those at the places, a decode outside any pairing
///   among them;
/// - nothing pairs.
///
/// The admission limit has the last word on the prefill taken, where the
/// prefill profile lists it as a hard filter (`:1148-1153`). At its limit
/// the request is shed under the limit's stage, and that refusal's trace
/// names no lifted veto: no request was steered.
pub(super) fn paired(
    [prefill, decode]: [Half<'_>; 2],
    table: &PairTable<'_>,
    (steers, weighs): (bool, bool),
    pool: &PoolFacts<'_>,
    scratch: &mut Scratch,
) -> Option<Decision> {
    let prefill = Half {
        limit_aside: true,
        ..prefill
    };
    let lift = lifted_cx(prefill.cx, steers && weighs, false);
    let open = |row: &&Candidate| classed(&scratch.classes, row.pair_class);
    let mut partnered = prefill.places().filter(open);
    // The pools as the gateway judges them: the prefills at the places that
    // have a decode to pair with, in whatever state and at whatever place,
    // and every decode at the places.
    let pairs = |row: &&Candidate| {
        let classes = table.of(&row.pair_class);
        let mut decodes = decode.rows.iter();
        decodes.any(|partner| classed(classes, partner.pair_class))
    };
    let mut prefills = prefill.places().filter(pairs);
    let stands = |side: &Half<'_>, row: &Candidate| side.stands(row, side.cx);
    if partnered.any(|row| prefill.stands(row, &lift))
        || (prefills.any(|row| stands(&prefill, row))
            && decode.places().any(|row| stands(&decode, row)))
    {
        return None;
    }
    let tiers = [LiftedVeto::Overload, LiftedVeto::Liveness];
    for lifted in tiers.into_iter().skip(usize::from(!steers)) {
        prefill.usable(lifted, steers, &mut scratch.single);
        decode.usable(lifted, steers, &mut scratch.decode);
        // Each usable prefill with the fewest requests among its usable
        // partners. The prefills of the least pair stay. A pair's rows hold
        // their own counts, and no side is lent any ([`PairPools::side`]):
        // they are read off the rows.
        let (prefills, decodes) = (&mut scratch.single.kept, &scratch.decode.kept);
        let loads = |at: &u32| {
            let row = &prefill.rows[*at as usize];
            let classes = table.of(&row.pair_class);
            let partners = decodes.iter().map(|&at| &decode.rows[at as usize]);
            let partners = partners.filter(|partner| classed(classes, partner.pair_class));
            let fewest = partners.map(|partner| partner.in_flight).min()?;
            Some((row.in_flight, fewest))
        };
        let Some(least) = prefills.iter().filter_map(loads).min() else {
            continue;
        };
        let usable = Survivors::counted(prefills.len() as u32);
        prefills.retain(|at| loads(at) == Some(least));
        let buf = &mut scratch.single;
        let Decision::Pick(mut taken) =
            steered(prefill.index, prefill.facts, pool, prefill.rows, buf)
        else {
            return None;
        };
        // The trace counts the prefills the pass kept, not the least pair's.
        taken.trace.survivors = usable;
        let first = *taken.targets.first()?;
        let row = prefill.rows.get(first.position as usize)?;
        let mut hard = prefill.profile.filters.iter().filter(|f| !f.soft());
        let limit = hard.find(|f| f.name() == HasCapacity.name());
        if let Some(limit) = limit.filter(|limit| !limit.keep(prefill.facts, row, prefill.cx)) {
            let trace = Trace {
                lifted: LiftedVeto::None,
                ..taken.trace
            };
            return Some(shed(reason_of(limit.name()), limit.name(), trace));
        }
        // The decode: the taken prefill's usable partner of the fewest
        // requests.
        let (classes, buf) = (table.of(&row.pair_class), &mut scratch.decode);
        buf.kept
            .retain(|&at| classed(classes, decode.rows[at as usize].pair_class));
        let mut pair = steered(decode.index, decode.facts, pool, decode.rows, buf);
        if let Decision::Pick(pick) = &mut pair {
            pick.targets.pair_under(&first, SessionVerdict::None);
            pick.trace = taken.trace;
        }
        return Some(pair);
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, Weak,
    };

    use crate::{
        document::spec::{Assignment, Needs, OverloadSpec, PolicySpec, Recovery, RulePlaced},
        engine::{
            decision::{Decision, LiftedVeto, ShedReason, Survivors},
            framework::{
                Binding, DecisionScope, Engine, Filter, FirstPartners, Marks, PairPools, PairSide,
                PairTable, Picker, Plan, Profile, Scorer, Scratch, TieBand, WeightedScorer,
                DRAW_FROM, WRITTEN_ROWS,
            },
            sticky_sessions::{SessionRule, SessionVerdict},
            view::{
                AffinityHolder, AffinityRows, Candidate, Constraints, Engines, Health, Identity,
                Phase, PoolFacts, RequestFacts, WorkerId,
            },
        },
        policies::{consistent_hashing::ConsistentHash, manual::Manual},
        stages::{
            filters::{HasCapacity, HasRoom, Healthy, NotOverloaded, NotTried},
            lifecycle::Lifecycle,
            pickers::Best,
        },
        tracking::{
            in_flight::tests::Lent,
            lease::{Lease, Side},
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
            profile.filter = filters.iter().map(|&name| name.into()).collect();
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
                    LiftedVeto::Overload,
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
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        let mut under = over(1, 1);
        under.health.overloaded = false;
        let kept = decide(&engine, &facts(1), &[over(0, 3), under]);
        assert_eq!(kept.trace().lifted, LiftedVeto::None);
        let shedding = engine_of("round_robin", true);
        let shed = decide(&shedding, &facts(1), &[over(0, 3), over(1, 1)]);
        assert_eq!(shed.trace().lifted, LiftedVeto::None);
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
            AffinityHolder {
                row: 3,
                overlap_blocks: 12,
                predicted_blocks: 0,
            },
            AffinityHolder {
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
                named: Some(AffinityRows {
                    holders: &holders,
                    ..AffinityRows::default()
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
                shadow: None,
                decode_shadow: None,
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

    /// A pair's prefill pool nobody is usable in is shed at the stage that
    /// emptied it. The prefill side's own run is put to no rule, and the
    /// pair's default asks the profile's filters once more with each veto
    /// lifted: the filter listed behind the veto refuses both prefills in
    /// each of its two passes.
    #[test]
    fn a_pairs_prefill_pool_with_no_usable_worker_sheds_at_the_veto() {
        let none: (&[u32], &[u32]) = (&[], &[]);
        let side = pairs_side_shed(Box::new(NotOverloaded), [over(0, 1), over(1, 2)], none);
        assert_eq!(side, ((ShedReason::Saturated, "not-overloaded"), 4));
    }

    /// A pair's engine built by hand: each side filters by the health stage
    /// and the overload veto and takes the best of equals, so a side the
    /// rule steers is the only one whose pick the requests in flight
    /// decide. `shed` makes its binding shed a pool the veto empties. Each
    /// side's shadow is a twin of its profile.
    fn pair_engine(shed: bool) -> Engine {
        pair_engine_of(shed, [vetoes(), vetoes()])
    }

    /// The health stage and the overload veto: a side's filters.
    fn vetoes() -> Vec<Box<dyn Filter>> {
        vec![Box::new(Healthy), Box::new(NotOverloaded)]
    }

    /// [`pair_engine`] with its live sides' filters given: the prefill
    /// side's, then the decode side's.
    fn pair_engine_of(shed: bool, [prefill, decode]: [Vec<Box<dyn Filter>>; 2]) -> Engine {
        pair_engine_holding(shed, [prefill, decode], [None, None])
    }

    /// A scorer that stands in for a stage that weighs load against
    /// affinity. The row at `holder` scores highest, as the worker that
    /// holds a session's prefix does, and `weighs` says whether the stage
    /// declares that it reads affinity.
    #[derive(Clone, Copy, Debug)]
    struct Holds {
        holder: u32,
        weighs: bool,
    }

    impl Lifecycle for Holds {}

    impl Scorer for Holds {
        fn name(&self) -> &'static str {
            "holds"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (score, &row) in out.iter_mut().zip(kept) {
                let holds = candidates[row as usize].position == self.holder;
                *score = if holds { 1.0 } else { 0.0 };
            }
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            Needs {
                affinity: self.weighs,
                ..Needs::default()
            }
        }
    }

    /// [`pair_engine_of`] with a scorer on the prefill side: the live
    /// prefill profile's, then its shadow's. `None` leaves that profile
    /// with no scorer.
    fn pair_engine_holding(
        shed: bool,
        [prefill, decode]: [Vec<Box<dyn Filter>>; 2],
        [live, shadow]: [Option<Holds>; 2],
    ) -> Engine {
        let side = |name: &str, filters, holds: Option<Holds>| Profile {
            name: name.into(),
            filters,
            scorers: holds
                .map(|holds| WeightedScorer {
                    scorer: Box::new(holds),
                    weight: 1.0,
                })
                .into_iter()
                .collect(),
            picker: Box::new(Best),
        };
        let mut engine = Engine::new(
            vec![
                side("prefill", prefill, live),
                side("decode", decode, None),
                side("twin", vetoes(), None),
                side("prefill-twin", vetoes(), shadow),
            ],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                shadow: Some(3),
                decode_shadow: Some(2),
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        if shed {
            engine.shed_when_all_vetoed(0);
        }
        engine
    }

    /// `worker` in the pair class `class`: classes pair by their number.
    fn classed(mut worker: Candidate, class: u32) -> Candidate {
        worker.pair_class = class;
        worker
    }

    /// The pair `engine` decides over `prefills` and `decodes`.
    fn decide_pair(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        engine.decide_pair(
            &facts(1),
            &PoolFacts::default(),
            pools,
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    /// The pair a pick names first: its prefill's position, then its decode's.
    fn pair_of(decision: &Decision) -> (u32, u32) {
        let pick = decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target");
        let decode = first.pair.expect("a pair's target").decode_position;
        (first.position, decode)
    }

    /// Under a binding that steers, a pair's decode side over its thresholds
    /// on every worker goes to the decode that holds the fewest requests
    /// among the chosen prefill's partners. No decode is open, so the rule
    /// first says which decodes it keeps and the prefill is chosen for
    /// them: a prefill whose class holds no decode is passed over. A doubted
    /// decode among them is not kept, however few requests it holds. The
    /// gateway's own pair selection answers so
    /// (`model_gateway/src/routers/common/placement.rs:1315-1321`,
    /// `:1338-1341`, `:1571-1578`).
    #[test]
    fn a_pairs_decode_side_over_on_every_worker_goes_to_the_fewest_in_flight() {
        let engine = pair_engine(false);
        let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
        let mut decodes = [over(0, 3), over(1, 1), over(2, 2)].map(|worker| classed(worker, 1));
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(pair_of(&decision), (1, 1));
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
        decodes[1].health.stalled = true;
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(pair_of(&decision), (1, 2));
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
    }

    /// Under a binding that sheds, a pair's decode side over its thresholds
    /// on every worker is shed as saturated, as the gateway's own selection
    /// lifts nothing where it sheds (`placement.rs:1314-1321` there).
    #[test]
    fn a_binding_that_sheds_sheds_a_pairs_decode_side_over_on_every_worker() {
        let (prefills, decodes) = ([classed(free(0, 1), 1)], [classed(over(0, 1), 1)]);
        let decision = decide_pair(&pair_engine(true), &prefills, &decodes);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::Saturated, "not-overloaded")
        );
    }

    /// Under a binding that sheds, a doubted decode under its thresholds
    /// beside a busy one takes the pair. The rule keeps that decode past its
    /// doubt, and a decode kept so opens no prefill: the prefill side sheds,
    /// and the pair's default pairs the free prefill with it, the liveness
    /// veto lifted and no worker over its thresholds taken. The gateway's own
    /// pair selection places that pair by its fallback
    /// (`model_gateway/src/routers/common/placement.rs:1443-1458`,
    /// `:1138-1147`).
    #[test]
    fn a_binding_that_sheds_pairs_past_a_decodes_doubt() {
        let prefills = [classed(free(0, 0), 1)];
        let decodes = [classed(over(0, 1), 1), classed(doubted(1, 0), 1)];
        let decision = decide_pair(&pair_engine(true), &prefills, &decodes);
        assert_eq!(pair_of(&decision), (0, 1));
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
    }

    /// What a pair's first decision asks of its rule: the overload veto of
    /// a decode pool, and no more. Decodes of one class all over stay
    /// unasked while a decode of another class is open. A decode pool
    /// doubted on every worker opens no prefill, and a prefill pool over on
    /// every worker is put to no rule: each goes to the pair's default,
    /// which says the veto it lifted.
    #[test]
    fn a_pairs_first_decision_answers_for_a_decode_pool_over_alone() {
        let engine = pair_engine(false);
        let prefills = [classed(free(0, 0), 1), classed(free(1, 5), 2)];
        let decodes = [
            classed(over(0, 0), 1),
            classed(over(1, 0), 1),
            classed(free(2, 9), 2),
        ];
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(pair_of(&decision), (1, 2));
        assert_eq!(decision.trace().lifted, LiftedVeto::None);
        let (idle, busy) = ([classed(free(0, 0), 1)], [classed(over(0, 0), 1)]);
        let suspect = [classed(doubted(0, 0), 1)];
        let (doubt, load) = (LiftedVeto::Liveness, LiftedVeto::Overload);
        for (prefills, decodes, lifted) in [(&idle, &suspect, doubt), (&busy, &idle, load)] {
            let paired = decide_pair(&engine, prefills, decodes);
            assert_eq!((pair_of(&paired), paired.trace().lifted), ((0, 0), lifted));
        }
    }

    /// A prefill pool refused on every worker goes to the pair of the least
    /// load: the prefill's requests first, then its partner's. Pairs the
    /// overload veto alone refuses come first: a doubted prefill that holds
    /// fewer is passed by. A binding that sheds takes no worker over its
    /// thresholds: the doubted prefill takes the request, and a pool over on
    /// every worker is shed as saturated. The gateway's own default pairs so
    /// (`model_gateway/src/routers/common/placement.rs:1093-1147`, asked at
    /// `:1395-1413`).
    #[test]
    fn a_refused_prefill_pool_goes_to_the_pair_of_the_least_load() {
        let prefills = [
            classed(over(0, 2), 1),
            classed(over(1, 1), 2),
            classed(over(2, 1), 3),
            classed(doubted(3, 0), 3),
        ];
        let decodes = [free(0, 0), free(1, 5), free(2, 4)];
        let decodes = [0, 1, 2].map(|at| classed(decodes[at], at as u32 + 1));
        let steered = decide_pair(&pair_engine(false), &prefills, &decodes);
        assert_eq!(pair_of(&steered), (2, 2));
        // The pair's trace is the prefill side's: its profile, the veto
        // lifted, and the three prefills the lifted pass kept.
        let trace = steered.trace();
        assert_eq!((trace.profile, trace.lifted), (0, LiftedVeto::Overload));
        assert_eq!(trace.survivors, Survivors::counted(3));
        let shedding = pair_engine(true);
        let doubted = decide_pair(&shedding, &prefills, &decodes);
        assert_eq!(pair_of(&doubted), (3, 2));
        assert_eq!(doubted.trace().lifted, LiftedVeto::Liveness);
        let refused = decide_pair(&shedding, &prefills[..3], &decodes);
        assert_eq!(shed_of(&refused), (ShedReason::Saturated, "not-overloaded"));
    }

    /// A free prefill no decode pairs with is no prefill of the pool the
    /// default judges: beside it, the prefills that have a partner, all over
    /// their thresholds, pair by load. The gateway's index lists the
    /// prefills with a compatible decode, and its default is judged over
    /// them (`model_gateway/src/routers/common/pair_index.rs:99-103`,
    /// `placement.rs:1395-1404` beside it).
    #[test]
    fn a_free_prefill_no_decode_pairs_with_is_not_of_the_refused_pool() {
        let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 2)];
        let decodes = [classed(free(0, 0), 1)];
        let paired = decide_pair(&pair_engine(false), &prefills, &decodes);
        assert_eq!(pair_of(&paired), (0, 0));
        assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
    }

    /// The pair's default keeps each side to the places its host states. A
    /// worker outside them is no worker of the pool the default judges,
    /// however free, and no pass takes it: a free prefill outside the
    /// places, and a free decode outside them. The gateway's own default
    /// judges the workers the request may use and pairs among them
    /// (`model_gateway/src/routers/common/placement.rs:774-786`,
    /// `:1097-1104`).
    ///
    /// The prefills it judges are those with a decode to pair with anywhere
    /// in the pool. A free prefill whose one partner is outside the decodes'
    /// places stands among them, so beside a free decode at the places the
    /// request is refused, as on main: the gateway lists a prefill's
    /// partners for the pool and no request (`pair_index.rs:99-103`,
    /// `:143-146` there), finds no pool vetoed on every worker and refuses
    /// too (`placement.rs:1438-1447`, `:1094-1096`).
    #[test]
    fn a_pairs_default_keeps_to_each_sides_places() {
        let engine = pair_engine(false);
        let held = |prefills: &[Candidate], decodes: &[Candidate], places: [Option<&[u32]>; 2]| {
            let pools = PairPools {
                prefills,
                decodes,
                ..PairPools::default()
            };
            let [cx, decode_cx] = places.map(|places| Constraints {
                places,
                ..Constraints::default()
            });
            let (req, pool, scratch) = (facts(1), PoolFacts::default(), &mut Scratch::new());
            let paired = engine.decide_pair(&req, &pool, pools, &cx, &decode_cx, scratch);
            let lifted = paired.trace().lifted;
            let pair = paired.pick().map(|_| (pair_of(&paired), lifted));
            pair.ok_or_else(|| shed_of(&paired))
        };
        let (first, every): (Option<&[u32]>, Option<&[u32]>) = (Some(&[0]), None);
        let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 1)];
        let pair = held(&prefills, &[classed(free(0, 0), 1)], [first, every]);
        assert_eq!(pair, Ok(((0, 0), LiftedVeto::Overload)));
        let decodes = [classed(doubted(0, 1), 1), classed(free(1, 0), 1)];
        let pair = held(&[classed(free(0, 0), 1)], &decodes, [every, first]);
        assert_eq!(pair, Ok(((0, 0), LiftedVeto::Liveness)));
        let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 2)];
        let decodes = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
        let refused = held(&prefills, &decodes, [every, first]);
        assert_eq!(refused, Err((ShedReason::NoEndpoints, "partnered")));
    }

    /// A pool the liveness veto refuses on every worker pairs by load under
    /// either word of the binding: the prefill pool, and the decode pool
    /// beside a standing prefill. A worker a hold stands on is never taken,
    /// however few requests it holds. The gateway's own default lifts the
    /// liveness veto so (`model_gateway/src/routers/common/placement.rs:1138-1147`,
    /// asked at `:1395-1413` and `:1438-1465`).
    #[test]
    fn a_pool_doubted_on_every_worker_pairs_by_load() {
        let mut suspect = [doubted(0, 2), doubted(1, 1), doubted(2, 0)];
        suspect[2].health.breaker_open = true;
        let suspect = suspect.map(|worker| classed(worker, 1));
        let idle = [classed(free(0, 0), 1)];
        let pools = [
            (&suspect[..], &idle[..], (1, 0)),
            (&idle[..], &suspect[..], (0, 1)),
        ];
        for shed in [false, true] {
            for (prefills, decodes, pair) in pools {
                let paired = decide_pair(&pair_engine(shed), prefills, decodes);
                assert_eq!(pair_of(&paired), pair, "shedding: {shed}");
                assert_eq!(paired.trace().lifted, LiftedVeto::Liveness);
            }
        }
    }

    /// The admission limit has the last word on a pair's default. A free
    /// prefill at its limit keeps the limit's refusal beside a prefill a
    /// veto refuses: the request waits, and no veto is lifted for it. Where
    /// the default takes a prefill at its limit the request is shed under
    /// the limit too, and the next prefill is not taken in its place. The
    /// gateway's own selection answers both so
    /// (`model_gateway/src/routers/common/placement.rs:1467-1480`,
    /// `:1148-1153`).
    #[test]
    fn the_admission_limit_has_the_last_word_on_a_pairs_default() {
        let mut filters = vetoes();
        filters.push(Box::new(HasCapacity));
        let engine = pair_engine_of(false, [filters, vetoes()]);
        let room = |mut worker: Candidate| {
            worker.has_capacity = true;
            classed(worker, 1)
        };
        let decodes = [classed(free(0, 0), 1)];
        let full = (ShedReason::NoEndpoints, "has-capacity");
        for vetoed in [over(1, 0), doubted(1, 0)] {
            let prefills = [classed(free(0, 4), 1), room(vetoed)];
            assert_eq!(shed_of(&decide_pair(&engine, &prefills, &decodes)), full);
        }
        // So it does where every decode is over its thresholds and the rule
        // opened their class: the free prefill stands at its limit with an
        // open partner, and the busy prefill with a slot is not taken. The
        // gateway judges the limit over the prefills with an open partner
        // (`:2515-2528` there on main `7526c6664e`).
        let busy = [classed(over(0, 0), 1)];
        let prefills = [classed(free(0, 4), 1), room(over(1, 0))];
        assert_eq!(shed_of(&decide_pair(&engine, &prefills, &busy)), full);
        // Two prefills over their thresholds: the less loaded is taken, at
        // its limit and then with a slot.
        let busier = room(over(1, 5));
        let limited = decide_pair(&engine, &[classed(over(0, 4), 1), busier], &decodes);
        assert_eq!(shed_of(&limited), full);
        // That refusal names no lifted veto: no request was steered.
        assert_eq!(limited.trace().lifted, LiftedVeto::None);
        let paired = decide_pair(&engine, &[room(over(0, 4)), busier], &decodes);
        assert_eq!(pair_of(&paired), (0, 0));
        // The limit is set aside on the prefill side alone. A decode whose
        // own profile lists the limit, and that is at it, pairs with nobody.
        let mut both = [vetoes(), vetoes()];
        for side in &mut both {
            side.push(Box::new(HasCapacity));
        }
        let alone = [room(over(0, 4))];
        let refused = decide_pair(&pair_engine_of(false, both), &alone, &decodes);
        assert_eq!(shed_of(&refused), (ShedReason::Saturated, "not-overloaded"));
    }

    /// The pools of a fleet some of whose partners go first. Both prefill
    /// classes pair with both decode classes, and prefill class 1 is paired
    /// with decode class 1 first. Prefill class 2 has no first partner. A
    /// gateway hands such classes for the workers of one known runtime
    /// beside workers whose runtime is unknown.
    fn first_pools<'a>(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> PairPools<'a> {
        PairPools {
            prefills,
            decodes,
            table: PairTable::new(&[2, 4], &[1, 2, 1, 2]),
            first: FirstPartners::new(&[1, 1], &[1]),
            ..PairPools::default()
        }
    }

    /// The pair `engine` decides over [`first_pools`].
    fn decide_first(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let pools = first_pools(prefills, decodes);
        engine.decide_pair(&facts(1), &pool, pools, &cx, &cx, &mut Scratch::new())
    }

    /// While a prefill has a first partner open, the pair is made among
    /// such prefills and their first partners, whatever stands beside
    /// them, and a request waits where the admission limit alone holds
    /// such a prefill back. A prefill another stage refuses keeps no
    /// request: the pair is then made among every prefill with an open
    /// partner, as it is where no first partner is open. The gateway's own
    /// selection orders its candidates so, asks the limit of them, and
    /// takes the pick's decode among its partners of that order
    /// (`model_gateway/src/routers/common/placement.rs:2479-2502`,
    /// `:2532-2545`, `:2615-2618` on main `30f636706`).
    #[test]
    fn a_pair_is_made_among_first_partners_while_one_is_open() {
        let mut filters = vetoes();
        filters.push(Box::new(HasCapacity));
        let engine = pair_engine_of(false, [filters, vetoes()]);
        let room = |mut worker: Candidate, class: u32| {
            worker.has_capacity = true;
            classed(worker, class)
        };
        // Place 0 of each side holds a worker with no first partner, and
        // place 1 the two workers that pair first.
        let decodes = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
        let prefills = [room(free(0, 0), 2), room(free(1, 0), 1)];
        assert_eq!(pair_of(&decide_first(&engine, &prefills, &decodes)), (1, 1));
        // With no first partner stated the same lists pair by the picker
        // alone, which takes the other prefill.
        let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
        let (pools, scratch) = (first_pools(&prefills, &decodes), &mut Scratch::new());
        let plain = PairPools {
            first: FirstPartners::default(),
            ..pools
        };
        let unordered = engine.decide_pair(&req, &pool, plain, &cx, &cx, scratch);
        assert_eq!(pair_of(&unordered), (0, 1));
        // Each side's shadow is shown what its live side was shown: the
        // prefills with a first partner open, and that pair's first
        // partners. The decodes stand the other way round here, so the
        // picker alone takes neither worker of the pair.
        let turned = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
        let pools = first_pools(&prefills, &turned);
        let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
        assert_eq!(pair_of(&live), (1, 0));
        let pick = live.pick().expect("a pair");
        for (side, shown) in [(PairSide::Prefill, 1), (PairSide::Decode, 0)] {
            let twin = engine.shadow_side(0, side, &req, &pool, pools, &cx, &cx, pick, scratch);
            assert_eq!(
                twin.map(|shadow| order(&shadow)[0]),
                Some(shown),
                "{side:?}"
            );
        }
        // Under no table a class pairs with its own number, and no first
        // partner is read, whatever an earlier decision marked in the
        // buffer.
        let bare = PairPools {
            table: PairTable::default(),
            ..first_pools(&prefills, &decodes)
        };
        let numbered = engine.decide_pair(&req, &pool, bare, &cx, &cx, scratch);
        assert_eq!(pair_of(&numbered), (0, 0));
        // At its admission limit the prefill with a first partner open
        // keeps the request: it waits, beside a prefill with a slot and an
        // open partner.
        let full = [room(free(0, 0), 2), classed(free(1, 0), 1)];
        assert_eq!(
            shed_of(&decide_first(&engine, &full, &decodes)),
            (ShedReason::NoEndpoints, "has-capacity")
        );
        // Refused by another stage it keeps none, with a slot or at its
        // limit: the pair is made among every prefill with an open partner.
        let mut down = free(1, 0);
        down.health.healthy = false;
        for refused in [over(1, 0), down] {
            for prefill in [room(refused, 1), classed(refused, 1)] {
                let next = decide_first(&engine, &[room(free(0, 0), 2), prefill], &decodes);
                assert_eq!(pair_of(&next).0, 0);
            }
        }
        // No first partner is open: a prefill pairs with any open partner
        // of its, the one with first partners too.
        let closed = [classed(free(0, 0), 2), classed(over(1, 0), 1)];
        let alone = [classed(free(0, 0), 2), room(free(1, 0), 1)];
        assert_eq!(pair_of(&decide_first(&engine, &alone, &closed)), (1, 0));
    }

    /// The decode of a pair made among first partners is one of them under
    /// the binding's all-vetoed rule too: with every decode over its
    /// thresholds the rule opens both classes, and the first partner is
    /// taken though the other decode holds fewer requests
    /// (`model_gateway/src/routers/common/placement.rs:2454-2456`,
    /// `:2615-2618` on main `30f636706`). The pair's default reads no
    /// first partner: over a prefill pool refused whole it takes the pair
    /// of the least load among every partner, as the gateway's default
    /// does (`:1804-1813` there).
    #[test]
    fn a_first_pairs_decode_is_a_first_partner_under_the_rule() {
        let engine = pair_engine(false);
        let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
        let busy = [classed(over(0, 0), 2), classed(over(1, 5), 1)];
        let steered = decide_first(&engine, &prefills, &busy);
        assert_eq!(pair_of(&steered), (1, 1));
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        // The default: the prefill of the least load with the partner of
        // the fewest requests, which is no first partner of it.
        let refused = [classed(over(0, 5), 2), classed(over(1, 0), 1)];
        let decodes = [classed(free(0, 0), 2), classed(free(1, 3), 1)];
        let paired = decide_first(&engine, &refused, &decodes);
        assert_eq!(pair_of(&paired), (1, 0));
        assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
    }

    /// A decode asked again behind its prefill is put to the test a pair's
    /// first decision makes, at its own instant: it is one of the held
    /// prefill's first partners while one of them is open, or opens where
    /// the rule lifts the overload veto for partners that are all over
    /// their thresholds, and any partner of the prefill else. The gateway's
    /// own selection asks no decode again, so these answers are the
    /// engine's own.
    #[test]
    fn a_decode_asked_again_is_a_first_partner_while_one_is_open() {
        let engine = pair_engine(false);
        let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
        let asked = |held: usize, decodes: &[Candidate]| {
            let (pool, cx) = (PoolFacts::default(), Constraints::default());
            let (pools, id) = (first_pools(&prefills, decodes), prefills[held].id);
            engine.decide_decode(&facts(1), &pool, pools, id, None, &cx, &mut Scratch::new())
        };
        let open = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
        assert_eq!(pair_of(&asked(1, &open)), (1, 1));
        // The first partner over its thresholds beside an open partner:
        // the open one, with no veto lifted.
        let busy = [classed(free(0, 0), 2), classed(over(1, 0), 1)];
        let other = asked(1, &busy);
        assert_eq!(pair_of(&other), (1, 0));
        assert_eq!(other.trace().lifted, LiftedVeto::None);
        // Every partner over its thresholds: the rule's lift opens the
        // first partner, which holds more requests than the other.
        let over_all = [classed(over(0, 0), 2), classed(over(1, 5), 1)];
        let steered = asked(1, &over_all);
        assert_eq!(pair_of(&steered), (1, 1));
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        // A first partner the lifted veto does not open leaves the rule
        // every partner.
        let mut down = over(1, 5);
        down.health.healthy = false;
        let steered = asked(1, &[classed(over(0, 0), 2), classed(down, 1)]);
        assert_eq!(pair_of(&steered), (1, 0));
        // The prefill with no first partner is asked among all its
        // partners: the fewest requests under the rule.
        let steered = asked(0, &over_all);
        assert_eq!(pair_of(&steered), (0, 0));
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        // The shadow of a decode asked again reads the partners that call
        // read, the held prefill's first partners. It is asked with the
        // decodes both ways round: the picker alone takes one place of the
        // two, whichever class stands there.
        let turned = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
        let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
        for (decodes, first) in [(&open, 1), (&turned, 0)] {
            let (pools, held) = (first_pools(&prefills, decodes), prefills[1].id);
            let scratch = &mut Scratch::new();
            let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
            assert_eq!(pair_of(&again), (1, first));
            let live = again.pick().expect("a decode");
            let later = engine.shadow_decode_later(0, &req, &pool, pools, held, live, &cx, scratch);
            assert_eq!(later.map(|shadow| order(&shadow)[0]), Some(first));
        }
    }

    /// The pools of a first list that is no partner list of the table: the
    /// table pairs class 1 with class 1 alone, and the first list names
    /// class 2.
    #[cfg(debug_assertions)]
    fn stray_pools<'a>(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> PairPools<'a> {
        PairPools {
            prefills,
            decodes,
            table: PairTable::new(&[1], &[1]),
            first: FirstPartners::new(&[1], &[2]),
            ..PairPools::default()
        }
    }

    /// First partners that do not stand within the pair table stop a debug
    /// build where a decision takes them: at a pair's first decision.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "the first partners do not stand within the pair table: Err(NoPartner)"
    )]
    fn first_partners_outside_the_table_stop_a_debug_build_at_a_pairs_decision() {
        let engine = pair_engine(false);
        let (prefills, decodes) = ([classed(free(0, 0), 1)], [classed(free(0, 0), 1)]);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let pools = stray_pools(&prefills, &decodes);
        let pair = engine.decide_pair(&facts(1), &pool, pools, &cx, &cx, &mut Scratch::new());
        panic!("first partners outside the table answered {pair:?}");
    }

    /// And at a decode asked again behind its prefill, which reads the held
    /// prefill's first list as it is handed.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "the first partners do not stand within the pair table: Err(NoPartner)"
    )]
    fn first_partners_outside_the_table_stop_a_debug_build_at_a_decode_asked_again() {
        let engine = pair_engine(false);
        let (prefills, decodes) = ([classed(free(0, 0), 1)], [classed(free(0, 0), 1)]);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let (pools, held) = (stray_pools(&prefills, &decodes), prefills[0].id);
        let scratch = &mut Scratch::new();
        let again = engine.decide_decode(&facts(1), &pool, pools, held, None, &cx, scratch);
        panic!("first partners outside the table answered {again:?}");
    }

    /// A prefill profile that reads affinity is taken to weigh load against
    /// it itself. Under a binding that steers, its run has the overload
    /// veto lifted: a prefill over its thresholds alone with an open
    /// partner is placed by the profile, where a pick by load would move a
    /// session off the worker that holds its prefix. No prefill stood
    /// without the lift here, so the pair's trace names the lifted veto.
    /// The default answers for such a profile everywhere else: under a
    /// binding that sheds, and where every prefill is doubted. The gateway
    /// keeps its default from such a policy's prefill the same way, and
    /// places that prefill by the policy with the veto lifted
    /// (`model_gateway/src/routers/common/placement.rs:1327-1346`, at main
    /// `b8c15d40a`).
    #[test]
    fn a_profile_that_reads_affinity_places_a_busy_prefill_itself() {
        #[derive(Debug)]
        struct ReadsAffinity;
        impl Filter for ReadsAffinity {
            fn name(&self) -> &'static str {
                "reads-affinity"
            }
            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
            fn needs(&self) -> Needs {
                Needs {
                    affinity: true,
                    ..Needs::default()
                }
            }
        }
        let weighing = |shed| {
            let mut filters = vetoes();
            filters.push(Box::new(ReadsAffinity));
            pair_engine_of(shed, [filters, vetoes()])
        };
        let decodes = [classed(free(0, 0), 1)];
        let busy = [classed(over(0, 1), 1), classed(doubted(1, 0), 1)];
        let placed = decide_pair(&weighing(false), &busy, &decodes);
        assert_eq!(pair_of(&placed), (0, 0));
        assert_eq!(placed.trace().lifted, LiftedVeto::Overload);
        let suspect = [classed(doubted(0, 2), 1), busy[1]];
        for (engine, prefills) in [(weighing(true), &busy), (weighing(false), &suspect)] {
            let paired = decide_pair(&engine, prefills, &decodes);
            assert_eq!(pair_of(&paired), (1, 0));
            assert_eq!(paired.trace().lifted, LiftedVeto::Liveness);
        }
    }

    /// A pair engine whose prefill profile holds the two vetoes and a stage
    /// that prefers the prefill at `holder` ([`Holds`]): `weighs` says
    /// whether that stage reads affinity. `shed` makes its binding shed.
    fn holding(shed: bool, holder: u32, weighs: bool) -> Engine {
        let live = Some(Holds { holder, weighs });
        pair_engine_holding(shed, [vetoes(), vetoes()], [live, None])
    }

    /// Under a binding that steers, a prefill profile that weighs load
    /// against affinity keeps a session on the prefill that holds its
    /// prefix while that prefill is over its thresholds, beside a free
    /// prefill too: its run has the overload veto lifted, and its own
    /// stages weigh the load. The pick stands beside a prefill under its
    /// thresholds, so the pair's trace names no lifted veto. A binding that
    /// sheds keeps the veto, and so does a profile none of whose stages
    /// reads affinity: each passes the busy holder over. The gateway's own
    /// pair selection lifts the veto so
    /// (`model_gateway/src/routers/common/placement.rs:2375-2394`; its case
    /// `an_overloaded_prefill_holding_the_prefix_keeps_the_session_under_cache_aware`
    /// at `:5881`, at main `55c35100d`).
    #[test]
    fn a_weighing_prefill_profile_keeps_a_session_on_its_busy_holder() {
        let prefills = [classed(over(0, 3), 1), classed(free(1, 0), 1)];
        let decodes = [classed(free(0, 0), 1)];
        let kept = decide_pair(&holding(false, 0, true), &prefills, &decodes);
        assert_eq!(pair_of(&kept), (0, 0));
        assert_eq!(kept.trace().lifted, LiftedVeto::None);
        for engine in [holding(true, 0, true), holding(false, 0, false)] {
            let moved = decide_pair(&engine, &prefills, &decodes);
            assert_eq!(pair_of(&moved), (1, 0));
            assert_eq!(moved.trace().lifted, LiftedVeto::None);
        }
    }

    /// Where every prefill is over its thresholds, a weighing profile under
    /// a binding that steers places the pair itself: the prefill its stages
    /// prefer, not the least loaded. No prefill stood without the lift, so
    /// the pair's trace says the lifted overload veto, as a kept key's pick
    /// does over a pool vetoed on every worker. A profile that reads no
    /// affinity leaves such a pool to the pair's default, which takes the
    /// least loaded, and a binding that sheds sheds it as saturated. The
    /// gateway's own selection places such a pool by its prefill policy
    /// (`model_gateway/src/routers/common/placement.rs:5853`, the case
    /// `an_all_overloaded_prefill_pool_is_placed_by_the_prefill_policy`, at
    /// main `55c35100d`).
    #[test]
    fn a_weighing_prefill_profile_places_a_pool_over_on_every_prefill() {
        let prefills = [classed(over(0, 5), 1), classed(over(1, 1), 1)];
        let decodes = [classed(free(0, 0), 1)];
        let placed = decide_pair(&holding(false, 0, true), &prefills, &decodes);
        assert_eq!(pair_of(&placed), (0, 0));
        assert_eq!(placed.trace().lifted, LiftedVeto::Overload);
        let by_load = decide_pair(&holding(false, 0, false), &prefills, &decodes);
        assert_eq!(pair_of(&by_load), (1, 0));
        assert_eq!(by_load.trace().lifted, LiftedVeto::Overload);
        let shed = decide_pair(&holding(true, 0, true), &prefills, &decodes);
        assert_eq!(shed_of(&shed), (ShedReason::Saturated, "not-overloaded"));
    }

    /// The lifted constraints of a weighing profile's run are a second
    /// value: the pair's default judges the pool as it stands, with no
    /// lift. Every prefill is over its thresholds, their partners are over
    /// too, and the one free decode partners none of them. No prefill has
    /// an open partner, so the prefill's run sheds and the default pairs by
    /// load. Judged under the lift, the prefills would stand beside the
    /// free decode, and the default would leave the refusal. The gateway
    /// judges its default over such a pool "as it was before the lift"
    /// (`model_gateway/src/routers/common/placement.rs:2486-2490`, at main
    /// `55c35100d`).
    #[test]
    fn a_weighing_profiles_pool_is_judged_for_the_default_with_no_lift() {
        let prefills = [classed(over(0, 2), 1), classed(over(1, 1), 1)];
        let partners = [classed(over(0, 4), 1), classed(over(1, 0), 1)];
        let decodes = [partners[0], partners[1], classed(free(2, 0), 2)];
        let paired = decide_pair(&holding(false, 0, true), &prefills, &decodes);
        assert_eq!(pair_of(&paired), (1, 1));
        assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
    }

    /// A weighing profile's refusal is its lifted run's own. The one
    /// prefill is over its thresholds and at its admission limit: the
    /// lifted run ends at the limit, the default leaves that refusal, and
    /// the shed names the limit. Named with the veto standing, the stage
    /// would be the veto. No request was steered, so the trace names no
    /// lifted veto. The gateway refuses a lifted prefill at its limit the
    /// same way
    /// (`model_gateway/src/routers/common/placement.rs:2590-2595`, at main
    /// `55c35100d`).
    ///
    /// The prefills that run could take are counted under the lift too.
    /// Where the one prefill is over its thresholds and its only partner
    /// is not ready, it is a routable prefill with no decode to go to, and
    /// the shed names the decode side's stage. Counted with the veto
    /// standing it would be no routable prefill, and the shed would name
    /// the veto.
    #[test]
    fn a_weighing_profiles_refusal_names_the_stage_its_lifted_run_ended_at() {
        let mut filters = vetoes();
        filters.push(Box::new(HasCapacity));
        let live = Some(Holds {
            holder: 0,
            weighs: true,
        });
        let engine = pair_engine_holding(false, [filters, vetoes()], [live, None]);
        let full = decide_pair(
            &engine,
            &[classed(over(0, 1), 1)],
            &[classed(free(0, 0), 1)],
        );
        assert_eq!(shed_of(&full), (ShedReason::NoEndpoints, "has-capacity"));
        assert_eq!(full.trace().lifted, LiftedVeto::None);
        let mut down = classed(free(0, 0), 1);
        down.health.healthy = false;
        let alone = decide_pair(&holding(false, 0, true), &[classed(over(0, 1), 1)], &[down]);
        assert_eq!(shed_of(&alone), (ShedReason::NoEndpoints, "healthy"));
    }

    /// The prefill's shadow is lifted by its own profile's word, as the
    /// live run is by the live profile's: a shadow that weighs load against
    /// affinity keeps the busy holder beside a live profile that passes it
    /// over, and a shadow that reads no affinity passes it over beside a
    /// live profile that keeps it. A pair the weighing lift alone placed
    /// gets no shadow on either side: its trace names the lifted veto, and
    /// the prefill buffer carries the mark.
    #[test]
    fn a_prefills_shadow_is_lifted_by_its_own_profiles_word() {
        let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
        let decodes = [classed(free(0, 0), 1)];
        let said = |prefills: &[Candidate], live: bool, shadow: bool| {
            let sides = [live, shadow].map(|weighs| Some(Holds { holder: 0, weighs }));
            let engine = pair_engine_holding(false, [vetoes(), vetoes()], sides);
            let pools = PairPools {
                prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let scratch = &mut Scratch::new();
            let pair = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
            let pick = pair.pick().expect("a pair");
            let twins = PairSide::BOTH.map(|side| {
                let twin = engine.shadow_side(0, side, &req, &pool, pools, &cx, &cx, pick, scratch);
                twin.map(|shadow| order(&shadow)[0])
            });
            (pair_of(&pair).0, pick.trace.lifted, twins)
        };
        // A busy holder beside a free prefill: no pick exists by the lift.
        let beside = [classed(over(0, 3), 1), classed(free(1, 0), 1)];
        let none = LiftedVeto::None;
        assert_eq!(said(&beside, true, true), (0, none, [Some(0), Some(0)]));
        assert_eq!(said(&beside, true, false), (0, none, [Some(1), Some(0)]));
        assert_eq!(said(&beside, false, true), (1, none, [Some(0), Some(0)]));
        assert_eq!(said(&beside, false, false), (1, none, [Some(1), Some(0)]));
        // Every prefill over: the pick exists by the lift, and no shadow
        // is said beside it.
        let alone = [classed(over(0, 3), 1)];
        let lifted = LiftedVeto::Overload;
        assert_eq!(said(&alone, true, true), (0, lifted, [None, None]));
    }

    /// The pair's default asks each side's hard filters alone
    /// ([`Half::stands`]). The one decode has no room, so a soft filter of
    /// its profile refuses it, and the default still takes it for the
    /// prefill pool it answers for: a room steers no default, as no soft
    /// filter does.
    #[test]
    fn a_pairs_default_asks_no_soft_filter() {
        let mut roomed = vetoes();
        roomed.push(Box::new(HasRoom));
        let engine = pair_engine_of(false, [vetoes(), roomed]);
        let paired = decide_pair(
            &engine,
            &[classed(over(0, 1), 1)],
            &[classed(free(0, 0), 1)],
        );
        assert_eq!(pair_of(&paired), (0, 0));
        assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
    }

    /// Each side of the pair's default is steered by its own facts. Equal
    /// pairs are drawn by the prefill side's seed, which is the request's,
    /// and equal partners by the decode side's own draw sequence: seed by
    /// seed, the pair is the worker the rule draws over each side's rows
    /// alone under that side's seed. The pair's trace is the prefill's: it
    /// is made under the prefill profile and carries the request's seed.
    #[test]
    fn a_pairs_default_steers_each_side_by_its_own_facts() {
        let (engine, alone) = (pair_engine(false), engine_of("round_robin", false));
        let prefills = [classed(over(0, 2), 1), classed(over(1, 2), 1)];
        let decodes = [classed(over(0, 3), 1), classed(over(1, 3), 1)];
        let first = |seed: u64, pool: &[Candidate]| order(&decide(&alone, &facts(seed), pool))[0];
        let mut parted = 0;
        for seed in 1..=64 {
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let (pool, cx) = (PoolFacts::default(), Constraints::default());
            let scratch = &mut Scratch::new();
            let paired = engine.decide_pair(&facts(seed), &pool, pools, &cx, &cx, scratch);
            let own = crate::Draw::new(seed).next_u64();
            let drawn = (first(seed, &prefills), first(own, &decodes));
            assert_eq!(pair_of(&paired), drawn, "seed {seed}");
            assert_eq!(paired.trace().seed, seed);
            assert_eq!(paired.trace().made_under(), Some(0));
            parted += u32::from(first(seed, &decodes) != drawn.1);
        }
        // The decode side's sequence is no copy of the request's.
        assert!(parted > 0, "the two sides drew alike under every seed");
    }

    /// A free decode keeps its pool from the rule, whoever it pairs with: a
    /// decode outside any pairing, or one whose own prefill is not ready.
    /// The gateway's own pair selection lifts the veto for the decode side
    /// only where no decode of the whole pool is available
    /// (`model_gateway/src/routers/common/placement.rs:1315-1321`, `:1341`),
    /// and refuses these pairs. Its default answers for no such pool either:
    /// the decode pool is not vetoed on every worker (`:1094`, `:1443-1448`).
    #[test]
    fn a_free_decode_keeps_its_pool_from_the_rule() {
        let engine = pair_engine(false);
        let refused = |prefills: &[Candidate], decodes: &[Candidate]| {
            shed_of(&decide_pair(&engine, prefills, decodes))
        };
        let saturated = (ShedReason::Saturated, "not-overloaded");
        let outside = [classed(over(0, 0), 1), free(1, 0)];
        assert_eq!(refused(&[classed(free(0, 0), 1)], &outside), saturated);
        let mut prefills = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
        prefills[1].health.healthy = false;
        let decodes = [classed(over(0, 0), 1), classed(free(1, 0), 2)];
        assert_eq!(refused(&prefills, &decodes), saturated);
    }

    /// A room narrows a prefill's partners, and hides no prefill. Both
    /// decodes are over their thresholds alone: one is full and its prefill
    /// free, the other has room and its prefill is at its limit. The pair is
    /// the free prefill with its full decode, whichever of the two classes
    /// stands first among the decodes. Between two partners of one prefill
    /// the one with a room takes the pair. The gateway's own pair
    /// selection makes it so: a prefill at its limit leaves before the
    /// policy picks, and the rooms are asked among the picked prefill's
    /// partners (`model_gateway/src/routers/common/placement.rs:1467-1480`,
    /// `:1520-1521`, `:1551-1578`).
    #[test]
    fn a_room_narrows_a_prefills_partners_and_hides_no_prefill() {
        let (mut limit, mut room) = (vetoes(), vetoes());
        limit.push(Box::new(HasCapacity));
        room.push(Box::new(HasRoom));
        let engine = pair_engine_of(false, [limit, room]);
        // The pair where the prefill at `under` is under its limit and the
        // other prefill's decode has the room.
        let placed = |under: usize| {
            let mut prefills = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
            prefills[under].has_capacity = true;
            let mut decodes = [classed(over(0, 0), 1), classed(over(1, 0), 2)];
            decodes[1 - under].has_room = true;
            let decision = decide_pair(&engine, &prefills, &decodes);
            (pair_of(&decision), decision.trace().lifted)
        };
        assert_eq!(placed(0), ((0, 0), LiftedVeto::Overload));
        assert_eq!(placed(1), ((1, 1), LiftedVeto::Overload));
        // The partner with a room, though the full one holds fewer requests.
        let mut prefill = classed(free(0, 0), 1);
        prefill.has_capacity = true;
        let mut partners = [classed(over(0, 0), 1), classed(over(1, 3), 1)];
        partners[1].has_room = true;
        let decision = decide_pair(&engine, &[prefill], &partners);
        assert_eq!(pair_of(&decision), (0, 1));
    }

    /// Three partners of class 1 over their thresholds, holding `in_flight`
    /// requests in their order, and their ids. The ids are their own: a
    /// tried list names a decode and no prefill.
    fn partners(in_flight: [u32; 3]) -> ([Candidate; 3], [WorkerId; 3]) {
        let mut decodes = [0, 1, 2].map(|at| classed(over(at, in_flight[at as usize]), 1));
        for decode in &mut decodes {
            decode.id = WorkerId(200 + u64::from(decode.position));
        }
        (decodes, decodes.map(|decode| decode.id))
    }

    /// The decodes a request is given among `decodes`, one prefill's
    /// partners, best first. It tried `tried`, and its host just failed to
    /// reach `gone`. A first decision names them. A decode asked again
    /// behind the prefill names the same over a scratch of its own, and
    /// both say the lifted overload veto.
    fn given(
        engine: &Engine,
        decodes: &[Candidate],
        tried: &[WorkerId],
        gone: &[WorkerId],
    ) -> Vec<u32> {
        let prefills = [classed(free(0, 0), 1)];
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            ..PairPools::default()
        };
        // A request that names a tried worker says its retry number.
        let req = RequestFacts {
            tried,
            attempt: u8::from(!tried.is_empty()),
            ..facts(1)
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let reached = Constraints {
            excluded: gone,
            ..Constraints::default()
        };
        let first = engine.decide_pair(&req, &pool, pools, &cx, &reached, &mut Scratch::new());
        let (held, fresh) = (prefills[0].id, &mut Scratch::new());
        let again = engine.decide_decode(&req, &pool, pools, held, None, &reached, fresh);
        let named = |decision: &Decision| {
            assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
            let targets = decision.pick().expect("a pick").targets.iter();
            let decodes = targets.map(|target| target.pair.expect("a pair").decode_position);
            decodes.collect::<Vec<_>>()
        };
        assert_eq!(named(&first), named(&again));
        named(&first)
    }

    /// The pass that opens the prefills sets the retry's tried list aside,
    /// and the partners' pass does where reading the list leaves nobody: a
    /// hard filter that reads the list hides no prefill and refuses no lone
    /// partner. The gateway's own pair selection steps aside from a tried
    /// decode only while another partner is left
    /// (`model_gateway/src/routers/common/placement.rs:1562-1570`), and
    /// places this pair.
    ///
    /// The partners' pass sets the list aside too where the profile's
    /// `not-tried` stood aside in it. Behind the hard filter that filter
    /// keeps a partner the host just failed to reach alone, and the request
    /// goes to the tried partners the host can reach. Under a profile that
    /// lists no `not-tried` no filter stood aside and none reads the host's
    /// list: the hard filter's pass stands as it keeps, and no tried
    /// partner comes back beside an unreachable one. The gateway keeps one
    /// list for a tried worker and one it failed to reach (`was_tried`,
    /// `:1177-1179` there) and takes the partner outside it (`:1568-1570`).
    #[test]
    fn a_tried_decode_still_opens_its_prefill() {
        /// Refuses a worker an earlier attempt tried: a hard filter.
        #[derive(Debug)]
        struct Untried;
        impl Filter for Untried {
            fn name(&self) -> &'static str {
                "untried"
            }
            fn keep(&self, req: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
                !req.tried.contains(&row.id)
            }
        }
        let mut reads_tried = vetoes();
        reads_tried.push(Box::new(Untried));
        let engine = pair_engine_of(false, [vetoes(), reads_tried]);
        let (prefills, mut decodes) = ([classed(free(0, 0), 1)], [classed(over(0, 1), 1)]);
        // An id of its own: the tried list names the decode and no prefill.
        decodes[0].id = WorkerId(200);
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let tried = [decodes[0].id];
        let req = RequestFacts {
            tried: &tried,
            ..facts(1)
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let decision = engine.decide_pair(&req, &pool, pools, &cx, &cx, &mut Scratch::new());
        assert_eq!(pair_of(&decision), (0, 0));
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
        // The pair's default reads the request's own facts. The same
        // partner, free and tried, stays refused by the hard filter as on
        // main: no veto refuses it, so none is lifted for it.
        let mut idle = decodes;
        idle[0].health.overloaded = false;
        let pools = PairPools {
            decodes: &idle,
            ..pools
        };
        let decision = engine.decide_pair(&req, &pool, pools, &cx, &cx, &mut Scratch::new());
        assert_eq!(shed_of(&decision), (ShedReason::NoEndpoints, "untried"));
        // Two tried partners beside one the host just failed to reach,
        // under `not-tried` behind the hard filter: the tried ones.
        let (decodes, ids) = partners([0, 1, 2]);
        let mut listed = vetoes();
        listed.push(Box::new(Untried));
        listed.push(Box::new(NotTried));
        let listed = pair_engine_of(false, [vetoes(), listed]);
        assert_eq!(given(&listed, &decodes, &ids[..2], &ids[2..]), [0, 1]);
        // Under the hard filter alone its pass stands on the two partners
        // it keeps. The host just failed to reach the first, which is the
        // tried one, and the third: the second takes the request, as in
        // the gateway, and the first does not come back.
        let gone = [ids[0], ids[2]];
        assert_eq!(given(&engine, &decodes, &ids[..1], &gone), [1, 2]);
        // That profile reads no host list: an unreachable second partner
        // stays ahead of the third by its load, where the gateway's one
        // list leaves the third alone.
        assert_eq!(given(&engine, &decodes, &ids[..1], &ids[1..2]), [1, 2]);
    }

    /// A retry over partners all over their thresholds keeps to the ones the
    /// request has not tried while one is left. The fewest in flight among
    /// those takes it, and the next fewest of them stand behind it.
    /// With every partner tried the list narrows nobody. A tried partner
    /// with a room stays ahead of untried ones with none. The gateway's own
    /// pair selection orders a pick's partners so: the rooms, the untried
    /// ones, the least loaded
    /// (`model_gateway/src/routers/common/placement.rs:1561`, `:1568-1570`,
    /// `:1575-1578`, `narrow_to` at `:266-282`). It names one pair
    /// (`:1621-1625`), so the alternates are the engine's own. Where the
    /// list names every partner but one the host just failed to reach, that
    /// one stays the last resort: the gateway's retry keeps one list, so
    /// that answer is the engine's own. So is a decode asked again behind
    /// its prefill: the gateway asks no decode again. So is the answer
    /// under a document that lists `not-tried` ahead of the room filter,
    /// an order the gateway has not: a partner the host just failed to
    /// reach takes no retry there from tried ones the host can reach.
    #[test]
    fn a_retry_keeps_to_the_untried_partners_while_one_is_left() {
        // The decode side's filters in the pair presets' order.
        let mut reads_tried = vetoes();
        reads_tried.push(Box::new(HasRoom));
        reads_tried.push(Box::new(NotTried));
        let engine = pair_engine_of(false, [vetoes(), reads_tried]);
        // The lightest untried partner stands last among them.
        let (decodes, ids) = partners([0, 2, 1]);
        let mut roomy = decodes;
        roomy[0].has_room = true;
        // Where the list moves no answer: none, every partner tried, every
        // partner but one the host failed to reach, a tried one with a room.
        assert_eq!(given(&engine, &decodes, &[], &[]), [0, 2, 1]);
        assert_eq!(given(&engine, &decodes, &ids, &[]), [0, 2, 1]);
        assert_eq!(given(&engine, &decodes, &ids[1..], &ids[..1]), [2, 1]);
        assert_eq!(given(&engine, &roomy, &ids[..1], &[]), [0]);
        // Where it does: the untried partners while one is left, the
        // fewest in flight first, and no tried one behind them.
        assert_eq!(given(&engine, &decodes, &ids[..1], &[]), [2, 1]);
        assert_eq!(given(&engine, &decodes, &ids[..2], &[]), [2]);
        // A document may list `not-tried` ahead of the room filter. The
        // one partner with a room is one the host just failed to reach,
        // and the others are tried: the pass that reads the list keeps
        // that one alone, and the tried ones take the retry. In the
        // presets' order the room goes first, as in the gateway.
        let mut reversed = vetoes();
        reversed.push(Box::new(NotTried));
        reversed.push(Box::new(HasRoom));
        let reversed = pair_engine_of(false, [vetoes(), reversed]);
        assert_eq!(given(&reversed, &roomy, &ids[1..], &ids[..1]), [2, 1]);
        assert_eq!(given(&engine, &roomy, &ids[1..], &ids[..1]), [0]);
    }

    /// Under a profile that lists no `not-tried`, a filter of the host's
    /// that reads the tried list answers on the request's own facts in the
    /// partners' pass. One that keeps the tried partner alone leaves the
    /// retry on it, however busy, and the pass with the list set aside is
    /// not run. The gateway's own selection steps aside from a tried worker
    /// while another is left
    /// (`model_gateway/src/routers/common/placement.rs:1485-1494`,
    /// `:1562-1570`), and asks no filter of a host's: this answer is the
    /// engine's own.
    #[test]
    fn a_hosts_filter_that_keeps_a_tried_partner_stands_on_a_retry() {
        /// Keeps the workers an earlier attempt tried, and every worker of
        /// a request that tried none: a hard filter.
        #[derive(Debug)]
        struct Again;
        impl Filter for Again {
            fn name(&self) -> &'static str {
                "again"
            }
            fn keep(&self, req: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
                req.tried.is_empty() || req.tried.contains(&row.id)
            }
        }
        let mut sticky = vetoes();
        sticky.push(Box::new(Again));
        let engine = pair_engine_of(false, [vetoes(), sticky]);
        let (decodes, ids) = partners([0, 1, 2]);
        assert_eq!(given(&engine, &decodes, &[], &[]), [0, 1, 2]);
        assert_eq!(given(&engine, &decodes, &ids[2..], &[]), [2]);
    }

    /// A decode asked for a held prefill is put to the rule, both its tiers.
    /// Partners over their thresholds on every worker go to the one that
    /// holds the fewest requests, and under a binding that sheds they are
    /// shed as saturated. Partners doubted on every worker go to the fewest
    /// requests under either word, and never to one a hold stands on. A
    /// request that names no tried worker takes one pass under the lifted
    /// overload veto: with no doubt among its partners, a filter behind the
    /// veto is asked about each once, and its refusal stands. The gateway's
    /// own selection asks no decode again, so these answers are the
    /// engine's own.
    #[test]
    fn a_decode_asked_for_a_held_prefill_is_put_to_the_rule() {
        let prefills = [classed(free(0, 0), 1)];
        let asked = |engine: &Engine, decodes: &[Candidate]| {
            let pools = PairPools {
                prefills: &prefills,
                decodes,
                ..PairPools::default()
            };
            engine.decide_decode(
                &facts(1),
                &PoolFacts::default(),
                pools,
                prefills[0].id,
                None,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let busy = [over(0, 2), over(1, 1)].map(|worker| classed(worker, 1));
        let steered = asked(&pair_engine(false), &busy);
        assert_eq!(pair_of(&steered), (0, 1));
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        assert_eq!(
            shed_of(&asked(&pair_engine(true), &busy)),
            (ShedReason::Saturated, "not-overloaded")
        );
        // The request names no tried worker: one pass under the lifted veto.
        let count = Arc::new(AtomicUsize::new(0));
        let mut refused = vetoes();
        refused.push(Box::new(Refuses(Arc::clone(&count))));
        let shed = asked(&pair_engine_of(false, [vetoes(), refused]), &busy);
        assert_eq!(shed_of(&shed), (ShedReason::NoEndpoints, "refuses"));
        assert_eq!(count.load(Ordering::Relaxed), busy.len());
        let mut suspect = [doubted(0, 2), doubted(1, 1), doubted(2, 0)];
        suspect[2].health.breaker_open = true;
        let suspect = suspect.map(|worker| classed(worker, 1));
        for shed in [false, true] {
            let steered = asked(&pair_engine(shed), &suspect);
            assert_eq!(pair_of(&steered), (0, 1), "shedding: {shed}");
            assert_eq!(steered.trace().lifted, LiftedVeto::Liveness);
        }
    }

    /// A pair the rule placed gets no decode shadow, at its first decision
    /// and where its decode is asked again, under either lift: a shadow's
    /// run says no binding's word, and a twin would shed the partners its
    /// live side was steered among. Beside a pair no rule placed the twin
    /// names the live decode.
    #[test]
    fn a_pair_the_rule_placed_gets_no_decode_shadow() {
        let (engine, prefills) = (pair_engine(false), [classed(free(0, 0), 1)]);
        let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
        let shadows = |made: [fn(u32, u32) -> Candidate; 2]| {
            let decodes = [made[0](0, 2), made[1](1, 1)].map(|worker| classed(worker, 1));
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let (held, side, scratch) = (prefills[0].id, PairSide::Decode, &mut Scratch::new());
            let first = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
            // A first decision's pair past a doubt is the default's, and no
            // shadow is said beside it either.
            let twin = first.pick().map(|live| {
                let said = engine.shadow_side(0, side, &req, &pool, pools, &cx, &cx, live, scratch);
                said.map(|shadow| order(&shadow)[0])
            });
            let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
            let live = again.pick().expect("a decode");
            let later = engine.shadow_decode_later(0, &req, &pool, pools, held, live, &cx, scratch);
            let later = later.map(|shadow| order(&shadow)[0]);
            (twin, later, live.trace.lifted == LiftedVeto::Liveness)
        };
        assert_eq!(shadows([free, over]), (Some(Some(0)), Some(0), false));
        assert_eq!(shadows([over, over]), (Some(None), None, false));
        assert_eq!(shadows([doubted, doubted]), (Some(None), None, true));
    }

    /// A pair the default made gets no shadow on either side: a twin
    /// decides by its profile where the live side was taken by load, and
    /// its stages shed a pool the vetoes refuse.
    /// Beside a pair whose decode side alone the rule steered, the prefill's
    /// shadow speaks: its live side was put to no rule. One scratch serves
    /// both pairs, the default's first.
    #[test]
    fn a_pair_the_default_made_gets_no_shadow() {
        let (engine, scratch) = (pair_engine(false), &mut Scratch::new());
        let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
        let mut said = |prefill: Candidate, decode: Candidate| {
            let (prefills, decodes) = ([classed(prefill, 1)], [classed(decode, 1)]);
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
            assert_eq!(pair_of(&live), (0, 0));
            assert_eq!(live.trace().lifted, LiftedVeto::Overload);
            let pick = live.pick().expect("a pair");
            PairSide::BOTH.map(|side| {
                let twin = engine.shadow_side(0, side, &req, &pool, pools, &cx, &cx, pick, scratch);
                twin.map(|shadow| order(&shadow)[0])
            })
        };
        // The default's pair: every prefill is over its thresholds.
        assert_eq!(said(over(0, 1), free(0, 0)), [None, None]);
        // The rule's pair: every decode is over, and the prefill side open.
        assert_eq!(said(free(0, 0), over(0, 1)), [Some(0), None]);
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
                shadow: None,
                decode_shadow: None,
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
        assert_eq!(
            beside.trace().lifted,
            LiftedVeto::None,
            "beside free workers"
        );
        for worker in &mut pool {
            worker.health.overloaded = true;
        }
        let whole = decide(&engine, &request, &pool);
        assert_eq!(order(&whole)[0], ring[0]);
        assert_eq!(
            whole.trace().lifted,
            LiftedVeto::Overload,
            "every worker over"
        );
        pool[next].health.overloaded = false;
        let tried = [pool[next].id];
        let retry = RequestFacts {
            tried: &tried,
            attempt: 1,
            ..request
        };
        let passed = decide(&engine, &retry, &pool);
        assert_eq!(order(&passed)[0], ring[0]);
        assert_eq!(
            passed.trace().lifted,
            LiftedVeto::None,
            "a tried worker under"
        );
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
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        engine.places_by_key(0);
        let pool = [over(0, 1), over(1, 2)];
        let decision = decide(&engine, &keyed("k1"), &pool);
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
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
                scorers: vec![WeightedScorer {
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
                shadow: None,
                decode_shadow: None,
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
        let binding = spec.bindings.first_mut().expect("the preset's binding");
        // The rule names the key's header, so the ring's own sources go: a
        // binding names them once.
        binding.routing_key = None;
        binding.sessions = Some(rule);
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
    /// says the lifted veto. The pin stands ahead of that worker's claim, and
    /// once the pool is under its thresholds the key returns to it (the
    /// manual preset's rule recovers the original).
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
        assert_eq!(whole.trace().lifted, LiftedVeto::Overload);
        assert_eq!(
            engine
                .sessions()
                .pins(request.session())
                .map(|pins| pins[0]),
            Some(pool[ring[1] as usize].id),
            "the pin is the key's first"
        );
        // The pool is under its thresholds again: the key is its pin's.
        for worker in &mut pool {
            worker.health.overloaded = false;
        }
        let later = RequestFacts {
            now_micros: 3_000,
            ..keyed("k1")
        };
        let back = decide(&engine, &later, &pool);
        assert_eq!(order(&back)[0], ring[1], "the pinned worker");
        assert_eq!(
            back.pick().expect("a pick").session_verdict,
            SessionVerdict::Pinned
        );
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
        assert_eq!(
            whole.trace().lifted,
            LiftedVeto::Overload,
            "every worker over"
        );
        pool[ring[1] as usize].health.overloaded = false;
        let beside = decide(&ring_under_a_rule(Assignment::Delegate), &request, &pool);
        assert_eq!(order(&beside)[0], ring[0]);
        assert_eq!(
            beside.trace().lifted,
            LiftedVeto::None,
            "beside a free worker"
        );
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
                scorers: vec![WeightedScorer {
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
                shadow: None,
                decode_shadow: None,
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
    /// pool: another decision's whole run, in the gap between a decision's
    /// read of the pins and its own claim.
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
                scorers: vec![WeightedScorer {
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
                shadow: None,
                decode_shadow: None,
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
            rule_placed: RulePlaced::Told,
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

    /// A client's key places like a session's and keeps nothing: the request
    /// goes to the ring's worker for the key, and off it where that worker
    /// is over its thresholds, which a session's key stays on.
    #[test]
    fn a_client_key_places_on_the_ring_and_lifts_no_veto() {
        let engine = PolicySpec::preset("consistent_hashing")
            .expect("the preset's form")
            .validate()
            .expect("the document builds");
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let as_client = |session: &RequestFacts<'static>| RequestFacts {
            session_key: Identity::NONE,
            client_key: session.session_key.hash,
            ..*session
        };
        // Eight keys: one key alone passes a tie's draw one time in four.
        for key in ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"] {
            let session = keyed(key);
            let owner = ring_order(&engine, &session, &pool)[0];
            let placed = order(&decide(&engine, &as_client(&session), &pool))[0];
            assert_eq!(placed, owner, "{key}");
        }
        let session = keyed("k1");
        let client = as_client(&session);
        let owner = ring_order(&engine, &session, &pool)[0];
        pool[owner as usize].health.overloaded = true;
        pool[owner as usize].in_flight = 1;
        assert_eq!(order(&decide(&engine, &session, &pool))[0], owner);
        assert_ne!(order(&decide(&engine, &client, &pool))[0], owner);
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
            assert_eq!(decision.trace().lifted, LiftedVeto::None, "seed {seed}");
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

    /// A pin the lifted pass lands on leads no worker that carries the cap's
    /// requests of the key. A dispatch on an alternate writes that worker
    /// among the key's pins, and the cap bounds where a pin may go. The free
    /// alternates stay behind the pin.
    #[test]
    fn a_kept_key_on_its_pin_names_no_alternate_at_the_rules_cap() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        // The pin over its thresholds and under the cap. The ring's next
        // worker carries the cap's requests of the key.
        pool[owner].health.overloaded = true;
        pool[owner].session_in_flight = 1;
        pool[ring[1] as usize].session_in_flight = 2;
        let one = decide(&engine, &request, &pool);
        let verdict = one.pick().expect("a pick").session_verdict;
        assert_eq!(verdict, SessionVerdict::Pinned);
        assert_eq!(
            order(&one),
            [ring[0], ring[2], ring[3]],
            "the free alternates alone"
        );
        // Every other worker carries them: the pin alone.
        for &other in &ring[1..] {
            pool[other as usize].session_in_flight = 2;
        }
        assert_eq!(order(&decide(&engine, &request, &pool)), [ring[0]]);
    }

    /// No dispatch of a kept key's decision writes a worker at the cap as
    /// the key's failover. Every worker but the pin carries the cap's
    /// requests of the key, and the host dispatches the decision's last
    /// target. Once the pin is down and the requests have ended, the key
    /// follows no failover: the ring places it anew.
    #[test]
    fn a_kept_keys_dispatch_writes_no_failover_at_the_rules_cap() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let owner = ring[0] as usize;
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        pool[owner].health.overloaded = true;
        pool[owner].session_in_flight = 1;
        for &other in &ring[1..] {
            pool[other as usize].session_in_flight = 2;
        }
        let decision = decide(&engine, &request, &pool);
        let pick = *decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        let last = *pick.targets.as_slice().last().expect("a target");
        let lease = Lease {
            issued_micros: 2_000,
            session: request.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &last, Side::Single, 0);
        pool[owner].health.healthy = false;
        for candidate in &mut pool {
            candidate.session_in_flight = 0;
        }
        assert_eq!(
            dispatched(&engine, &pool, 3_000),
            ring[1],
            "no failover was written at the cap"
        );
    }

    /// A followed claim leads no worker that carries the cap's requests of
    /// the key either. This decision's own placement, the ring's first
    /// worker, carries them: it leaves the targets behind the claimed
    /// worker, where a dispatch would write it among the key's pins.
    #[test]
    fn a_followed_claim_names_no_alternate_at_the_rules_cap() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 2,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
            rule_placed: RulePlaced::Told,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let (engine, gap) = ring_with_a_gap(Some(rule));
        // The other decision sees the ring's first worker down and claims
        // the ring's second.
        let mut theirs = pool.to_vec();
        theirs[ring[0] as usize].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        pool[ring[0] as usize].session_in_flight = 2;
        let decision = decide(&engine, &keyed("k1"), &pool);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "the other decision ran"
        );
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        assert_eq!(
            order(&decision),
            [ring[1], ring[2], ring[3]],
            "the free alternates alone"
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
        assert_eq!(over.trace().lifted, LiftedVeto::None);
        // Every worker over its thresholds: the lifted pass places the key
        // on its pin again, and the placement stands by the lift.
        for candidate in &mut pool {
            candidate.health.overloaded = true;
        }
        let all = decide(&engine, &request, &pool);
        assert_eq!(order(&all)[0], ring[0], "every worker over");
        assert_eq!(verdict(&all), SessionVerdict::Pinned);
        assert_eq!(all.trace().lifted, LiftedVeto::Overload);
        // Every worker at the cap as well: the key stays on its pin.
        for candidate in &mut pool {
            candidate.session_in_flight = 2;
        }
        let stays = decide(&engine, &request, &pool);
        assert_eq!(order(&stays)[0], ring[0], "every worker at the cap");
        assert_eq!(verdict(&stays), SessionVerdict::Pinned);
    }

    /// A claim made in the gap beside a pin that stands is followed too,
    /// and the pin stays. The key's pin is the ring's third worker, over
    /// its thresholds in both views: each decision's lifted pass keeps it.
    /// Another decision of the key, which sees the ring's first worker
    /// down, claims the ring's next worker between this decision's read of
    /// the pins and its own claim. This decision's claim sees the newcomer
    /// beside the pin it read and follows the other's worker, under either
    /// recovery. A second claim there would drop the pin.
    #[test]
    fn a_claim_made_in_the_gap_beside_a_standing_pin_is_followed() {
        for recovery in [Recovery::Original, Recovery::Latest] {
            let rule = SessionRule {
                assignment: Assignment::Delegate,
                inflight_cap: 0,
                ttl_micros: None,
                sweep_micros: 0,
                recovery,
                rule_placed: RulePlaced::Told,
            };
            let (ringed, _) = ring_with_a_gap(None);
            let (engine, gap) = ring_with_a_gap(Some(rule));
            let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
            let ring = ring_order(&ringed, &keyed("k1"), &pool);
            let request = keyed("k1");
            // The key's pin is the ring's third worker, over its thresholds
            // in both views: each decision's lifted pass keeps it.
            let pinned = ring[2] as usize;
            let table = engine.sessions();
            table.lookup(request.session(), request.now_micros, &rule);
            table.dispatched(request.session(), SessionVerdict::Pinned, pool[pinned].id);
            pool[pinned].health.overloaded = true;
            let mut theirs = pool.to_vec();
            theirs[ring[0] as usize].health.healthy = false;
            *gap.staged.lock().expect("the staged pool") = theirs;
            let decision = decide(&engine, &request, &pool);
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
            assert!(pins.contains(&pool[pinned].id), "{recovery:?}: {pins:?}");
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
                scorers: vec![WeightedScorer {
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
                shadow: None,
                decode_shadow: None,
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
    ) -> (u32, SessionVerdict, LiftedVeto) {
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
            rule_placed: RulePlaced::Told,
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
        assert_eq!(under, (ring[1], pinned, LiftedVeto::None), "under the cap");
        let on = followed(ring_with_a_gap(Some(rule)), &ring, at(2));
        assert_eq!(on, (ring[0], handed, LiftedVeto::None), "at the cap");
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
        assert_eq!(
            lifted,
            (ring[0], none, LiftedVeto::None),
            "no veto is lifted"
        );
        let stays = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
            pool[claimed].health.overloaded = true;
        });
        assert_eq!(
            stays,
            (ring[0], none, LiftedVeto::None),
            "no worker under it"
        );
        // This decision sees the ring's first worker down as the other
        // does: its own placement is the claimed worker.
        let same = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
            pool[ring[0] as usize].health.healthy = false;
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(
            same,
            (ring[1], pinned, LiftedVeto::None),
            "placed on the claim"
        );
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
            rule_placed: RulePlaced::Told,
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
        assert_eq!(under, (ring[1], pinned, LiftedVeto::None), "under the cap");
        let on = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(on, (ring[0], handed, LiftedVeto::None), "at the cap");
        let stays = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
        });
        assert_eq!(
            stays,
            (ring[0], none, LiftedVeto::None),
            "no survivor under it"
        );
        // The ring's last is under the cap and over its thresholds.
        let refused = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            for candidate in pool.iter_mut() {
                candidate.session_in_flight = 2;
            }
            pool[last].health.overloaded = true;
            pool[last].session_in_flight = 0;
        });
        assert_eq!(
            refused,
            (ring[0], none, LiftedVeto::None),
            "no veto is lifted"
        );
        // This decision sees the ring's first worker down as the other
        // does: its own placement is the claimed worker.
        let same = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
            pool[ring[0] as usize].health.healthy = false;
            pool[claimed].session_in_flight = 2;
        });
        assert_eq!(
            same,
            (ring[1], pinned, LiftedVeto::None),
            "placed on the claim"
        );
    }

    /// A key handed on from another request's claim at the rule's cap moves
    /// its pin where it is decided, in either form, as a respill from a pin
    /// read first does: the worker taken leads the key's pins ahead of its
    /// dispatch, and the claimed worker stands behind it.
    #[test]
    fn a_key_handed_on_from_a_followed_claim_fronts_its_worker_where_it_is_decided() {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 2,
            ttl_micros: None,
            sweep_micros: 0,
            recovery: Recovery::Original,
            rule_placed: RulePlaced::Told,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let (taken, claimed) = (ring[0] as usize, ring[1] as usize);
        let forms = [
            ("kept", ring_with_a_gap(Some(rule))),
            ("plain", plain_ring_with_a_gap(rule)),
        ];
        for (form, (engine, gap)) in forms {
            // The other decision sees the ring's first worker down and
            // claims the ring's second. This one sees the claimed worker
            // with the cap's two requests of the key.
            let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
            let mut theirs = pool.to_vec();
            theirs[taken].health.healthy = false;
            *gap.staged.lock().expect("the staged pool") = theirs;
            pool[claimed].session_in_flight = 2;
            let request = keyed("k1");
            let decision = decide(&engine, &request, &pool);
            assert!(
                gap.staged.lock().expect("the staged pool").is_empty(),
                "{form}: the other decision ran"
            );
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.session_verdict, SessionVerdict::Respill, "{form}");
            assert_eq!(order(&decision)[0], ring[0], "{form}");
            assert_eq!(
                engine.sessions().pins(request.session()),
                Some([pool[taken].id, pool[claimed].id]),
                "{form}: ahead of its dispatch"
            );
        }
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
        // The free workers carry the cap too. The veto as it stands still
        // keeps them: the ring's first of them takes the request, no veto
        // is lifted, and no pin moves to a worker at the cap. Asked ahead
        // of the respill below, which moves the pin where it is decided.
        pool[ring[2] as usize].session_in_flight = 2;
        pool[ring[3] as usize].session_in_flight = 2;
        let standing = decide(&engine, &request, &pool);
        assert_eq!(order(&standing)[0], ring[2], "a free worker at the cap");
        let pick = standing.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None);
        assert_eq!(standing.trace().lifted, LiftedVeto::None);
        // The free workers under the cap: the key passes the ring's next
        // worker, over its thresholds, to the first of them.
        pool[ring[2] as usize].session_in_flight = 0;
        pool[ring[3] as usize].session_in_flight = 0;
        let over = decide(&engine, &request, &pool);
        assert_eq!(order(&over)[0], ring[2], "the pin over its thresholds");
        assert_eq!(over.trace().lifted, LiftedVeto::None);
    }

    /// A kept key handed on from the pin its lifted pass landed on moves
    /// that pin where it is decided, as a respill from a pin read first
    /// does: the worker taken leads the key's pins ahead of its dispatch,
    /// and the key's next request, decided ahead of that dispatch, follows
    /// it there as a pin.
    #[test]
    fn a_kept_key_handed_on_at_the_cap_fronts_its_worker_where_it_is_decided() {
        let engine = ring_under_a_cap(2);
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&engine_of("consistent_hashing", false), &keyed("k1"), &pool);
        let (owner, next) = (ring[0] as usize, ring[1] as usize);
        assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
        let request = RequestFacts {
            now_micros: 2_000,
            ..keyed("k1")
        };
        let pins = || engine.sessions().pins(request.session());
        assert_eq!(pins(), Some([pool[owner].id, WorkerId(0)]), "the pin read");
        // The pin carries the cap's two requests of the key, over its
        // thresholds: the veto as it stands drops it, the lifted pass lands
        // on it, and the ring's next worker takes the key.
        pool[owner].session_in_flight = 2;
        pool[owner].health.overloaded = true;
        let handed = decide(&engine, &request, &pool);
        assert_eq!(order(&handed)[0], ring[1]);
        let pick = handed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        assert_eq!(
            pins(),
            Some([pool[next].id, pool[owner].id]),
            "ahead of its dispatch"
        );
        // The key's next request, decided ahead of that dispatch.
        let later = decide(&engine, &request, &pool);
        assert_eq!(order(&later)[0], ring[1]);
        let pick = later.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
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
            rule_placed: RulePlaced::Told,
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
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
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
            (ring[1], LiftedVeto::None),
            "no count: the veto stands for the key"
        );
        assert_eq!(
            asked(&ring_under_a_cap(2), true),
            (ring[0], LiftedVeto::None),
            "the count written: the key is kept"
        );
        assert_eq!(
            asked(&ring_under_a_rule(Assignment::Delegate), false),
            (ring[0], LiftedVeto::None),
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

    /// A pair's prefill pool doubted on every worker, with nobody usable in
    /// it, is shed at the health stage. The pair's default lifts the doubt
    /// in its last pass alone: the filter listed behind that stage is asked
    /// about both prefills there, and about none before.
    #[test]
    fn a_pairs_prefill_pool_doubted_with_no_usable_worker_sheds_at_the_health_stage() {
        let none: (&[u32], &[u32]) = (&[], &[]);
        let side = pairs_side_shed(Box::new(Healthy), [doubted(0, 1), doubted(1, 2)], none);
        assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), 2));
    }

    /// A hold stands in every pass of a pair's default. A prefill pool the
    /// host holds on every row is shed at the health stage, and the filter
    /// listed behind that stage is asked about no prefill. With a doubted
    /// row among held ones it is asked about that row alone.
    #[test]
    fn a_pairs_prefill_pool_under_a_hold_sheds_at_the_health_stage() {
        for (held, asked) in [(&[0, 1][..], 0), (&[0], 1)] {
            let masks: (&[u32], &[u32]) = (&[0, 1], held);
            let side = pairs_side_shed(Box::new(Healthy), [free(0, 1), free(1, 2)], masks);
            assert_eq!(side, ((ShedReason::NoEndpoints, "healthy"), asked));
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
            LiftedVeto::Overload
        );
        assert_eq!(
            mark(&steering, &[doubted(0, 1), doubted_over(1, 0)]),
            LiftedVeto::Liveness
        );
        assert_eq!(
            mark(&steering, &[free(0, 1), doubted(1, 0)]),
            LiftedVeto::None
        );
        let shedding = engine_of("least_load", true);
        assert_eq!(
            mark(&shedding, &[over(0, 1), doubted(1, 3)]),
            LiftedVeto::Liveness
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
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
        assert_eq!(scratch.single.undoubted.capacity(), 0, "no row is copied");
        let decision = ask(
            &[free(0, 3), doubted(1, 1), free(2, 2)],
            &[0, 2],
            &mut scratch,
        );
        assert_eq!(order(&decision), vec![1, 2, 0]);
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
        let decision = ask(&[free(0, 0), free(1, 5)], &[0], &mut scratch);
        assert_eq!(order(&decision), vec![1]);
        assert_eq!(decision.trace().lifted, LiftedVeto::None);
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
                assert_eq!(one.trace().lifted, LiftedVeto::Liveness);
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
            assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
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

    /// A held row that carries a doubt of its own gains nobody either: the
    /// hold stands through the tier, on a lent row with its leader's bit
    /// as on any other. A pool held on every row costs its one pass
    /// whether or not a row carries the bit. A row that carries it with no
    /// hold on it still opens the tier and takes the request.
    #[test]
    fn a_held_row_that_carries_its_own_doubt_costs_no_further_pass() {
        let mut pool = [free(0, 0), free(1, 1), free(2, 9)];
        pool[0].health.stalled = true;
        pool[1].health.stalled = true;
        let asked = Arc::new(AtomicUsize::new(0));
        let counts: Box<dyn Filter> = Box::new(Counts(Arc::clone(&asked)));
        let (engine, _) = counted_among(|_| vec![counts, Box::new(Healthy)]);
        let ask = |stalled: &[u32], held: &[u32]| {
            let masked = Constraints {
                stalled,
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
        let (whole, passes) = ask(&[0, 1, 2], &[0, 1, 2]);
        assert_eq!(shed_of(&whole), (ShedReason::NoEndpoints, "healthy"));
        assert_eq!(passes, pool.len());
        // Row 1 carries the bit and no hold: the tier opens for it alone.
        let (one, passes) = ask(&[0, 2], &[0, 2]);
        assert_eq!(order(&one), vec![1]);
        assert_eq!(one.trace().lifted, LiftedVeto::Liveness);
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
                assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
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
            assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
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
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
    }
}
