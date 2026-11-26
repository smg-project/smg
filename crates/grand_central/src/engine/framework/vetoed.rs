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
//! its prefix while that prefill is over its thresholds. A profile weighs
//! where one of its scorers says so of itself
//! ([`Scorer::balances_load_against_affinity`]): a stage that reads the
//! affinity column and has no gate of its own lifts nothing.
//!
//! The mark of such a lifted run has one meaning. Where the veto as it
//! stands leaves the pool nobody ([`all_vetoed`]), the pick exists by the
//! lift, and it says the lifted veto in its trace as the rule's own pick
//! does ([`LiftedVeto::Overload`]): a kept key's pick, and a pair placed by
//! its weighing prefill profile. A host counts them all from the one mark.
//! A pick beside a worker under its thresholds carries no mark.

use super::{
    classed, engines::run_owned, filter, keep_folded, leads, places, reason_of, run_single, shed,
    single_emptied, trace_of, Assignment, Binding, Buffers, Engine, Filter, PairSide, PairTable,
    Picker, Plan, Profile, Scorer, Scratch, SessionRule, Set,
};
use crate::{
    engine::{
        decision::{Decision, LiftedVeto, Pick, Placing, Survivors, Targets, Trace},
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
    /// no binding's word ([`Set::side`]): a prefill side's and its shadow's,
    /// one that only names a shed's stage, an encode item's.
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
/// `candidates` with no doubt on it ([`with_no_doubt`]), in the same
/// order, so a stage handed a row reads the lift whatever it reads of the
/// row. The pass that lifts the liveness veto is shown these where a row's
/// own bit is set. Rows the host's mask alone doubts are read as they are,
/// with the mask cut to the rows a hold stands on. A stage that answers
/// from its own scope answers as it did: the scope was opened over the
/// decision's rows. One pool of rows a scratch, sized at the first pass
/// that needs it.
fn undoubted(candidates: &[Candidate], rows: &mut Vec<Candidate>) {
    rows.clear();
    rows.extend(candidates.iter().map(with_no_doubt));
}

/// A row as it reads with no doubt: its stalled bit cleared where its
/// breaker bit is clear. A row that carries both bits is one a hold of the
/// host stands on, where the host wrote its rows. It keeps both, as a held
/// row stays in the mask beside lent rows.
fn with_no_doubt(row: &Candidate) -> Candidate {
    Candidate {
        health: Health {
            stalled: row.health.stalled && row.health.breaker_open,
            ..row.health
        },
        ..*row
    }
}

/// Whether the gateway's own pair selection would read `rows` as wholly
/// over their thresholds: one of them is ready and over them, and none
/// that is ready is under them (`all_overloaded`,
/// `model_gateway/src/routers/common/overload.rs:58-74`, on main
/// `56773d28d`). A pair's decision says the list it shed over by this
/// count ([`Engine::decide_pair`]).
///
/// That selection counts a worker under a liveness doubt as ready. So a
/// row is ready here as the pass that lifts the liveness veto reads it
/// ([`doubted`]): by the health stage's own test of the row with no doubt
/// on it ([`with_no_doubt`]), under that pass's constraints
/// ([`lifted_cx`]), in which a hold of the host stays. A row is over its
/// thresholds by the overload veto's own test, under the decision's
/// constraints. Each test is its stage's, so the count and the stages
/// cannot part.
///
/// A row under a doubt and under its thresholds keeps its list from being
/// wholly over them, though the passes of the decision that shed kept no
/// such row. Under a veto its host lifted no row is over its thresholds,
/// and no list is wholly over them.
///
/// Cold and out of line: asked behind a pair's saturated shed alone.
#[cold]
#[inline(never)]
pub(super) fn wholly_over<'a>(
    rows: impl Iterator<Item = &'a Candidate>,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
) -> bool {
    let lifted = lifted_cx(cx, false, true);
    let mut over = false;
    for row in rows {
        if !Healthy.keep(facts, &with_no_doubt(row), &lifted) {
            continue;
        }
        if NotOverloaded.keep(facts, row, cx) {
            return false;
        }
        over = true;
    }
    over
}

/// Whether `row`, a prefill, has a decode to pair with anywhere in
/// `decodes`, in whatever state and at whatever place. The gateway's list
/// of the prefills that pair is the pool's own and no request's: it holds
/// no prefill that pairs with no decode
/// (`model_gateway/src/routers/common/pair_index.rs:158-161`, on main
/// `56773d28d`).
pub(super) fn pairs_in(row: &Candidate, table: &PairTable<'_>, decodes: &[Candidate]) -> bool {
    let classes = table.of(&row.pair_class);
    let mut decodes = decodes.iter();
    decodes.any(|partner| classed(classes, partner.pair_class))
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
        ..trace_of(index, facts, pool, candidates, &buf.kept, buf.traced())
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

    /// Whether each hard filter of the side reads a row alone
    /// ([`Filter::row_wise`]) and holds no decision scope. Such a filter
    /// answers alike behind the decision it stood in. Any other may answer
    /// by a scope that is closed by then, or by what it kept of being
    /// asked. A prefill side's shadow asks this of the live decode profile
    /// before its default asks that profile's filters
    /// ([`Engine::shadow_side`]).
    pub(super) fn reads_rows_alone(&self) -> bool {
        let mut hard = self.profile.filters.iter().filter(|f| !f.soft());
        hard.all(|f| f.row_wise() && f.scope().is_none())
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

    /// Whether one of the side's rows of `classes` stands but for the
    /// side's admission limit ([`Self::but_for_the_limit`]): the limit
    /// alone left the side's run nobody among them. A pair's decision sheds
    /// such a request under the limit's stage, on which its host waits for
    /// room ([`Engine::decide_pair`]). A prefill side's shadow asks the
    /// same test ([`Engine::shadow_side`]).
    ///
    /// In line at each caller: as a call, it cost a pair's decision that
    /// never asks it 3 instructions.
    #[inline(always)]
    pub(super) fn waits_among(&self, classes: &[u32]) -> bool {
        let listed = |row: &&Candidate| classed(classes, row.pair_class);
        let mut among = self.places().filter(listed);
        among.any(|row| self.but_for_the_limit(row))
    }

    /// The side's rows that stand in the pass that lifts `lifted`
    /// ([`lifted_cx`]), into `buf.kept`, the mark left in `buf.lifted`. The
    /// set opens with the places the side's host states, as every pass of
    /// the rule does ([`places::open`]). With the liveness veto lifted, rows
    /// that carry a doubt are read with none ([`undoubted`]).
    pub(super) fn usable(&self, lifted: LiftedVeto, steers: bool, buf: &mut Buffers) {
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
/// (`model_gateway/src/routers/common/placement.rs:1084-1172`, at main
/// `b8c15d40a` as every line of that file cited here). Each
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
///   weighs load against affinity (`weighs`, [`Engine::weighs`]) and a
///   binding that steers, that run had the
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
///
/// Two steps make it, and a side's shadow asks its own step alone
/// ([`Engine::shadow_side`]): the prefill ([`paired_prefill`]), then that
/// prefill's decode ([`paired_decode`]).
pub(super) fn paired(
    sides: [Half<'_>; 2],
    table: &PairTable<'_>,
    words: (bool, bool),
    pool: &PoolFacts<'_>,
    scratch: &mut Scratch,
) -> Option<Decision> {
    let [prefill, decode] = sides;
    let step = paired_prefill::<false>(sides, table, words, pool, scratch)?;
    let Decision::Pick(taken) = &step else {
        // The admission limit's shed of the prefill taken.
        return Some(step);
    };
    let first = *taken.targets.first()?;
    let class = prefill.rows.get(first.position as usize)?.pair_class;
    let mut pair = paired_decode::<false>(&decode, table, class, pool, &mut scratch.decode);
    if let Decision::Pick(pick) = &mut pair {
        pick.targets
            .pair_under(&first, SessionVerdict::None, Placing::Stages);
        pick.trace = taken.trace;
    }
    Some(pair)
}

/// The first step of the pair's default ([`paired`]): the prefill it takes,
/// which is the prefill of the least pair, or the admission limit's shed of
/// that prefill. `None` where the default answers nothing. The pass that
/// paired leaves its mark in both buffers ([`Half::usable`]) and the
/// decodes it kept in `scratch.decode.kept`, where the second step reads
/// them ([`paired_decode`]).
///
/// A prefill side's shadow whose own run shed asks this step alone
/// ([`shadow_prefill`]).
///
/// No line reads `SHADOW`. It gives the default and a shadow a copy of
/// this step each, so the default's copy is compiled into its one caller.
/// One copy for both costs the default's decision about 190 instructions:
/// the bench cell `pair_default` counts them.
pub(super) fn paired_prefill<const SHADOW: bool>(
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
    let pairs = |row: &&Candidate| pairs_in(row, table, decode.rows);
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
        // their own counts: they are read off the rows.
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
            // The prefill the default took is at its limit: the prefill
            // side's refusal.
            let refused = shed(reason_of(limit.name()), limit.name(), trace);
            return Some(refused.of_side(PairSide::Prefill));
        }
        return Some(Decision::Pick(taken));
    }
    None
}

/// The scratch of a pair's decision while the default's first step reads
/// it for a prefill side's shadow ([`shadow_prefill`]). Its end puts back
/// what that step changed and the decode's shadow reads: the classes, and
/// the mark on the prefill's buffer. It runs on a return and on a panic.
struct PutBack<'a> {
    scratch: &'a mut Scratch,
    /// The mark the pair's decision left on the prefill's buffer.
    mark: LiftedVeto,
}

impl Drop for PutBack<'_> {
    fn drop(&mut self) {
        let scratch = &mut *self.scratch;
        if scratch.among_first {
            std::mem::swap(&mut scratch.classes, &mut scratch.firsts);
        }
        scratch.single.lifted = self.mark;
    }
}

/// The default's first step ([`paired_prefill`]) for a prefill side's
/// shadow whose own run shed ([`Engine::shadow_side`]), with its profile in
/// the prefill's place.
///
/// The step reads every open class. Among first partners the pair's
/// decision left those in `scratch.firsts`, so the two lists are swapped
/// for the call. The step also writes the prefill buffer's mark, which
/// held `mark` until then.
///
/// The scratch is left as the pair's decision left it ([`PutBack`]), where
/// the step panics too. A host catches each side's shadow by itself, so
/// the decode's shadow runs on this scratch behind a prefill's that
/// panicked.
pub(super) fn shadow_prefill(
    sides: [Half<'_>; 2],
    table: &PairTable<'_>,
    words: (bool, bool),
    pool: &PoolFacts<'_>,
    scratch: &mut Scratch,
    mark: LiftedVeto,
) -> Option<Decision> {
    if scratch.among_first {
        std::mem::swap(&mut scratch.classes, &mut scratch.firsts);
    }
    let back = PutBack { scratch, mark };
    paired_prefill::<true>(sides, table, words, pool, &mut *back.scratch)
}

/// The second step of the pair's default ([`paired`]): the decode, which is
/// the usable partner that holds the fewest requests of the prefill of
/// class `class`. `buf.kept` holds the side's rows that stand in the pass
/// that paired ([`Half::usable`]). The pick is the rule's ([`steered`]):
/// its targets name decodes, and the caller pairs them with the prefill.
///
/// A decode side's shadow beside the default's pair asks this step alone,
/// for the live pair's prefill and behind its own profile's usable rows.
///
/// No line reads `SHADOW`. It gives each caller its own copy, as the first
/// step's does ([`paired_prefill`]), and the bench cell `pair_default`
/// counts what that saves.
pub(super) fn paired_decode<const SHADOW: bool>(
    decode: &Half<'_>,
    table: &PairTable<'_>,
    class: u32,
    pool: &PoolFacts<'_>,
    buf: &mut Buffers,
) -> Decision {
    let classes = table.of(&class);
    buf.kept
        .retain(|&at| classed(classes, decode.rows[at as usize].pair_class));
    steered(decode.index, decode.facts, pool, decode.rows, buf)
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/framework/vetoed/tests.rs"]
mod tests;
