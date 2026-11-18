//! The all-vetoed rule: what a decision does when the overload veto leaves
//! it no worker. Every usable worker of the pool is over its thresholds, so
//! the veto that steers requests away from a busy worker has nowhere to
//! steer them. A binding answers in one of two ways ([`AllVetoed`]): the
//! request goes to the least-vetoed worker, or it is shed. One function
//! holds the rule ([`emptied`]). The filter half of a pass asks it where a
//! hard filter leaves nobody, by the word the pass's set carries, so a pass
//! that keeps a worker holds nothing for the rule and runs none of it.
//!
//! A profile that places a request by its session key reads the same word
//! ahead of its pass ([`keeps_its_key`]): where the binding steers, a keyed
//! request is decided with the veto lifted ([`Engine::keep`]), so the key
//! keeps its worker while that worker is over its thresholds. Under a
//! sessions rule the key's pin is asked first, with the veto as it stands,
//! and the lifted pass places a key no pin takes. Where the veto as it
//! stands leaves the key's pool nobody, the kept pick says the lifted veto in
//! its trace as the rule's own pick does ([`Lifted::Overload`]): a host
//! counts both from the one mark.

use super::{
    filter, run_single, single_emptied, trace_of, Assignment, Binding, Buffers, Engine, Filter,
    Picker, Plan, Profile, Scorer, Scratch, SessionRule, Set,
};
use crate::{
    builtins::{Best, LeastRequests, NotOverloaded},
    decision::{Decision, Lifted, Pick, Targets, Trace},
    sessions::SessionVerdict,
    view::{Candidate, Constraints, PoolFacts, RequestFacts},
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
    /// The request is shed as saturated.
    Shed,
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
    /// places the key with the veto lifted. The run is the profile's whole
    /// pass and never a draw: the ring's scorer is shown the whole view, so
    /// a profile that holds it does not draw first. The set it is handed
    /// sheds ([`Set::shedding`]): no pass of a kept key is put to the
    /// all-vetoed rule, which the lifted run stands in for.
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
        let mut decision = run_single(profile, index, facts, pool, set, &lifted, buf);
        if let Decision::Pick(pick) = &mut decision {
            if all_vetoed(profile, facts, set, cx, buf) {
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
/// goes, and nothing is lifted.
pub(super) fn keeps_its_key(
    binding: &Binding,
    placing: Assignment,
    rule: AllVetoed,
    facts: &RequestFacts<'_>,
) -> bool {
    rule == AllVetoed::Steer
        && binding.plan == Plan::Single
        && !facts.session_key.is_none()
        && (binding.sessions.is_none() || placing == Assignment::Delegate)
}

/// The profile's filters again with the overload veto lifted, for a keyed
/// request no pin of its sessions rule took under the veto as it stands.
/// Answers the shed of a pool the lifted pass leaves nobody; `buf` holds
/// the lifted pass's survivors otherwise, and the placement runs over them.
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
/// The lifted pass just placed the key: `buf` holds that pass's survivors,
/// and `cx` is the decision's own, the veto standing in it. A survivor
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
    !buf.kept.iter().any(stands) && filter(profile, facts, cx, buf, set).is_some()
}

/// Whether the overload veto stood in the pass the stage `emptied_by`
/// ended: the overload filter is one of the profile's hard filters, at or
/// before that stage. A pool emptied before the veto ran is no nearer a
/// worker with the veto lifted.
fn veto_ran(profile: &Profile, emptied_by: &str) -> bool {
    let veto = NotOverloaded.name();
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
/// pick among them ([`steered`]).
///
/// Under [`AllVetoed::Steer`], when the overload veto stood in that pass,
/// the profile's filters run again with the veto lifted. The rows they keep
/// are the usable ones, every one over its thresholds. So a hard filter
/// listed behind the overload filter does not hide such a pool, and the
/// answer does not turn on the order a document lists its filters in. The
/// retry's tried list is set aside for that pass, as the gateway's fallback
/// sets it aside: a worker an earlier attempt tried is taken again when it
/// holds the fewest requests. The host's list of workers it just failed to
/// reach stays: the soft filter passes over such a worker while another
/// usable one stands, and takes it as the last resort.
///
/// The shed stands for a set that sheds ([`AllVetoed::Shed`]: a binding's
/// word, each side of a pair, and the rule's own second pass), for a pool
/// emptied before the veto ran and for a pool with no row over its
/// thresholds, as under a veto its host lifted: none of them is filtered
/// a second time. A pool nobody is usable in even with the veto lifted is
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
    if set.vetoed == AllVetoed::Shed || !veto_ran(profile, emptied_by) {
        return emptied_by;
    }
    // A pool with no row over its thresholds gains nobody when the veto is
    // lifted: the shed stands, for this one look at the whole set. Under a
    // veto its host lifted no row reads as over.
    if !set
        .candidates
        .iter()
        .any(|row| !NotOverloaded.keep(facts, row, cx))
    {
        return emptied_by;
    }
    let lifted = Constraints {
        overload_lifted: true,
        ..*cx
    };
    let whole = RequestFacts {
        tried: &[],
        ..*facts
    };
    // The second pass is put to no rule: its set sheds, so the rule is asked
    // once a decision. Where that pass leaves nobody either, its own stage
    // stands. The rows it keeps are the rule's, behind the stage the veto's
    // pass named.
    filter(profile, &whole, &lifted, buf, set.shedding()).unwrap_or(emptied_by)
}

/// The rule's pick among the rows its lifted pass kept (`buf.kept`): the
/// least-vetoed worker, which is the fewest requests in flight, ties drawn
/// by the decision's seed, the next fewest as alternates. The pick is the
/// engine's rule and no stage of the profile makes it, so a document cannot
/// put a scorer where a safety default stands. It pins no session key, and
/// its trace says the veto it lifted ([`Lifted::Overload`]): the mark a host
/// counts its steered picks from.
pub(super) fn steered(
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
) -> Decision {
    buf.scores.clear();
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
        lifted: Lifted::Overload,
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
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, Weak,
    };

    use crate::{
        builtins::{Best, Healthy, Manual, NotOverloaded},
        counts::tests::Lent,
        decision::{Decision, Lifted, ShedReason},
        framework::{
            Binding, DecisionScope, Engine, Filter, Marks, PairPools, Picker, Plan, Profile,
            Scratch, Weighted, DRAW_FROM, WRITTEN_ROWS,
        },
        lease::{Lease, Lifecycle, Side},
        ring::ConsistentHash,
        sessions::{SessionRule, SessionVerdict},
        spec::{Assignment, Needs, OverloadSpec, PolicySpec, Recovery},
        view::{
            Candidate, Constraints, Health, Identity, NamedAffinity, NamedHolder, Phase, PoolFacts,
            RequestFacts, WorkerId,
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

    fn facts(seed: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            seed,
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

    /// A worker that is down, draining, stalled or behind an open breaker is
    /// not usable: it is never the steer, however few requests it holds.
    #[test]
    fn a_worker_another_veto_refuses_is_never_the_steer() {
        let engine = engine_of("least_load", false);
        let mut pool = [over(0, 0), over(1, 0), over(2, 0), over(3, 0), over(4, 7)];
        pool[0].health.healthy = false;
        pool[1].health.draining = true;
        pool[2].health.stalled = true;
        pool[3].health.breaker_open = true;
        assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![4]);
        // With no usable row left the stage that refused them names the shed.
        pool[4].health.healthy = false;
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

    /// Each side of a pair sheds a pool the overload veto empties: the pair
    /// plan does not enter the rule. Under a binding that steers, a prefill
    /// side over its thresholds on every worker is shed at the veto, and the
    /// filter listed behind the veto is asked about no prefill.
    #[test]
    fn a_pairs_side_sheds_at_the_veto_and_is_filtered_once() {
        let asked = Arc::new(AtomicUsize::new(0));
        let prefill_side = Profile {
            name: "counted".into(),
            filters: vec![
                Box::new(NotOverloaded),
                Box::new(Refuses(Arc::clone(&asked))),
            ],
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
        let prefills = [paired(over(0, 1)), paired(over(1, 2))];
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
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert_eq!(
            shed_of(&decision),
            (ShedReason::Saturated, "not-overloaded")
        );
        assert_eq!(asked.load(Ordering::Relaxed), 0);
    }

    /// A shed the veto had no part in is not filtered a second time: a pool
    /// emptied before the veto ran, a decision whose host lifted the veto
    /// itself and a pool with no row over its thresholds cost the one pass
    /// they cost without the rule. A pool with one row over its thresholds
    /// is filtered again, and sheds at the stage behind the veto.
    #[test]
    fn a_shed_the_veto_had_no_part_in_costs_one_pass() {
        let counted = |veto_first: bool| {
            let asked = Arc::new(AtomicUsize::new(0));
            let refuses: Box<dyn Filter> = Box::new(Refuses(Arc::clone(&asked)));
            let veto: Box<dyn Filter> = Box::new(NotOverloaded);
            let filters = if veto_first {
                vec![veto, refuses]
            } else {
                vec![refuses, veto]
            };
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
        };
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
        assert_eq!(bare.draw(), Marks::DRAWS_FIRST);
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
                        let _ = decide(&engine, facts, &staged);
                    }
                }
            }
            true
        }
    }

    /// The ring's profile built by hand with [`InTheGap`] among its filters,
    /// marked as the validator marks the ring's profile, under `rule`.
    fn ring_with_a_gap(rule: Option<SessionRule>) -> (Arc<Engine>, Arc<Gap>) {
        let gap = Arc::new(Gap::default());
        let mut engine = Engine::new(
            vec![Profile {
                name: "ring".into(),
                filters: vec![
                    Box::new(Healthy),
                    Box::new(NotOverloaded),
                    Box::new(InTheGap(Arc::clone(&gap))),
                ],
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
        let engine = Arc::new(engine);
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

    /// The rule's in-flight cap holds for a pin the lifted pass lands on as
    /// it holds for a pin read first: the lift is of the overload veto, and
    /// of no other limit. A pin that carries the cap's requests of the key
    /// hands the key to the ring's next worker, over its thresholds as under
    /// them. With every worker at the cap the key stays on its pin.
    #[test]
    fn a_followed_pin_at_the_rules_cap_hands_the_key_on() {
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
        // Under its thresholds the pin hands the key on.
        let under = decide(&engine, &request, &pool);
        assert_eq!(order(&under)[0], ring[1], "under its thresholds");
        assert_eq!(verdict(&under), SessionVerdict::Respill);
        // Over them it hands the key on as well.
        pool[owner].health.overloaded = true;
        let over = decide(&engine, &request, &pool);
        assert_eq!(order(&over)[0], ring[1], "over its thresholds");
        assert_eq!(verdict(&over), SessionVerdict::Respill);
        assert_eq!(over.trace().lifted, Lifted::None);
        // Every worker over its thresholds: the placement stands by the lift.
        for candidate in &mut pool {
            candidate.health.overloaded = true;
        }
        let all = decide(&engine, &request, &pool);
        assert_eq!(order(&all)[0], ring[1], "every worker over");
        assert_eq!(verdict(&all), SessionVerdict::Respill);
        assert_eq!(all.trace().lifted, Lifted::Overload);
        // Every worker at the cap: the key stays on its pin.
        for candidate in &mut pool {
            candidate.session_in_flight = 2;
        }
        let stays = decide(&engine, &request, &pool);
        assert_eq!(order(&stays)[0], ring[0], "no worker under the cap");
        assert_eq!(verdict(&stays), SessionVerdict::Pinned);
    }

    /// A key handed on at the cap reads the veto as it stands, from a pin
    /// over its thresholds as from one under them: it passes the ring's next
    /// worker where that worker is over its thresholds beside free ones.
    /// Where the standing veto keeps no worker under the cap, the lifted
    /// pass's survivors decide, and the trace says the lift.
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
        let under = decide(&engine, &request, &pool);
        assert_eq!(order(&under)[0], ring[2], "the pin under its thresholds");
        pool[owner].health.overloaded = true;
        let over = decide(&engine, &request, &pool);
        assert_eq!(order(&over)[0], ring[2], "the pin over its thresholds");
        assert_eq!(over.trace().lifted, Lifted::None);
        // The free workers carry the cap too: the one worker under it is
        // over its thresholds, and the key goes there by the lift.
        pool[ring[2] as usize].session_in_flight = 2;
        pool[ring[3] as usize].session_in_flight = 2;
        let lifted = decide(&engine, &request, &pool);
        assert_eq!(order(&lifted)[0], ring[1], "one worker under the cap");
        let pick = lifted.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        assert_eq!(lifted.trace().lifted, Lifted::Overload);
    }

    /// A profile the engine does not hold cannot be marked: the index is
    /// the validator's own, and a debug build refuses any other.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "is not built"))]
    fn a_profile_the_engine_does_not_hold_is_refused_in_a_debug_build() {
        let mut engine = engine_of("consistent_hashing", false);
        engine.places_by_key(7);
    }
}
