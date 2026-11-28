//! A pair's decision: how the engine places a request on a prefill and a
//! decode. The functions are the engine's own, in a block of this module.
//!
//! [`Engine::decide_pair`] decides a pair, both its sides, and
//! [`Engine::decide_decode`] the decode side for a held prefill.
//! [`Engine::decide_over_an_empty_list`] answers for a pair over two
//! lists of which one holds no row. [`Engine::weighs`] says whether a
//! profile weighs load against affinity by a gate of its own. Under a
//! binding that steers, [`Engine::weighing_lift`] gives such a profile
//! its constraints with the overload veto lifted.
//!
//! One side of a decision can be made once more with that side's shadow
//! profile: [`Engine::shadow_side`] for a side of a pair's decision, and
//! [`Engine::shadow_decode_later`] for a decode decided after its
//! prefill. [`Engine::shadows_side`] says whether a pair binding names a
//! shadow for a side that the build found free to run.
//! [`Engine::side_apart`] says whether the build found that a side's
//! named shadow would be handed another column than its live side's pool
//! carries.
//!
//! Under `in-process`, `Engine::decide_pair_drawn` tries a pair's
//! decision in which each side is asked by its draw.
//! `Engine::pairs_drawn` says whether some binding's pair may be asked
//! so, and `Engine::pair_drawn_for` whether the request's own may.
//!
//! This module is a child of the framework's because these functions read
//! private fields of the engine and call private functions of that module
//! as they stand.

use std::cell::OnceCell;

use super::{
    carries_a_column, classed, counts_no_place, filter, held_to, keeps_its_first, live_sides,
    number_not_counted, places, reason_of, rows_not_written, run_profile, run_side, run_side_lent,
    shed, target_missed, vetoed, with_lent, AllVetoed, Buffers, Count, Engine, Filter, Half,
    PairPools, Scratch, Set, ShedOver,
};
#[cfg(feature = "in-process")]
use super::{run_side_drawn_written, DrawnSide, Marks, PairDrawn};
// The two imports under `cfg(doc)` hold names that only doc links of this
// module use.
#[cfg(doc)]
use super::{side_reads_another_column, side_shadows, PairTable, Picker, Scorer};
#[cfg(doc)]
use crate::{document::spec::Needs, tracking::lease::Lease};
#[cfg(feature = "in-process")]
use crate::{
    engine::{decision::Survivors, view::Phase},
    Draw,
};
use crate::{
    engine::{
        decision::{
            Decision, LiftedVeto, PairSide, Pick, Placing, Shed, ShedReason, Trace,
            AFFINITY_COLUMN, BINDING, NO_STAGE, PARTNERED, PARTNERS, PREFILL_GONE,
        },
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, PoolFacts, RequestFacts, WorkerId},
    },
    policies::consistent_hashing::LentRing,
    stages::{
        filters::{HasCapacity, NotOverloaded},
        pickers::target_of,
    },
};

impl Engine {
    /// The `pd-pair` plan, and the pair half of `epd`: the prefill profile
    /// over `prefills`, then the decode profile over the decodes of the
    /// classes the chosen prefill's class pairs with ([`PairTable`]). A
    /// prefill whose partners leave no decode is passed over for the next
    /// prefill target; the targets list carries the chosen pair first and the
    /// alternate decodes of the same prefill after it. Pair class `0` is
    /// outside any pairing on either side: such a decode partners nobody and
    /// such a prefill is dropped at the `partnered` stage, so a host that
    /// leaves the classes unset sheds instead of pairing all with all. Each
    /// side's affinity column rides its pool ([`PairPools`]); the request
    /// carries none here and names no rows, and either is refused under
    /// the stage name `affinity-column`.
    ///
    /// A side whose profile shows its rows to a plugin is handed rows
    /// written for the decision ([`Self::reads_for`]). A mask beside that
    /// side's rows ([`Constraints::stalled`]) is declined under the stage
    /// name `written-rows` before either scope opens, as [`Self::decide`]
    /// declines it. A pair side carries no pin.
    ///
    /// Under the binding's sessions rule each side the rule covers follows
    /// its key's pin by the side's own session. A side whose picker leaves
    /// the placement to the rule is placed by the assignment. The pick
    /// states the prefill side's verdict, and its first target's pair the
    /// decode side's
    /// ([`Pair::decode_verdict`](crate::engine::decision::Pair::decode_verdict)).
    /// A claim of the prefill side names the prefill the pair formed on,
    /// before any dispatch, and a decision that sheds leaves none.
    /// A prefill side's pin at the cap moves to that prefill where the
    /// pair forms, and a decision that forms no pair moves none.
    ///
    /// The decode side is put to its binding's all-vetoed rule for the
    /// overload veto, as the pool of a single request is ([`Self::decide`]).
    /// Where no decode is open and some are over their thresholds alone, the
    /// rule lifts that veto once for the pool, and every decode the hard
    /// filters then keep opens its prefills. A prefill is chosen among them
    /// by its own profile, its partners go to the one of them that holds the
    /// fewest requests, and the pair's trace says the lifted veto
    /// ([`LiftedVeto::Overload`]). On a retry the profile's soft filter keeps
    /// that pick to the partners the request has not tried while one is
    /// left. Under a binding that sheds, such a pool opens no prefill, and
    /// the request is shed by the one question of a pair's shed, below.
    /// The prefill side's own run is put to no rule. Under a binding that
    /// steers, a prefill profile that weighs load against affinity runs with
    /// the overload veto lifted (`Engine::weighing_lift`), and a pair no
    /// prefill of which stood without the lift says that veto in its trace.
    ///
    /// Where that run sheds, a pool whose every usable worker is vetoed
    /// goes to the pair's default: the usable prefill and partner of the
    /// least load, the pairs the overload veto alone refuses first, with the
    /// admission limit asked of the prefill taken. A prefill pool refused
    /// whole pairs so, and so does a decode pool the liveness veto refuses
    /// on every worker. Each side's workers are those at the places its
    /// host states.
    ///
    /// The prefill admission limit is judged over the prefills that still
    /// have a partner to go to. Where one of them stands but for its limit,
    /// the limit alone left the run nobody, and the request is shed under
    /// the limit's stage, on which its host waits for room. Outside the
    /// pair's default, a prefill at its limit with no open partner makes no
    /// request wait: its refusal is named by what its partners' own stages
    /// say, as the refusal of a prefill under its limit is.
    ///
    /// Behind the pair's default and its question of saturation, where no
    /// prefill at the request's places stands but for its limit, the limit
    /// names no shed: its name is the word a host waits on for room. The
    /// prefill side's stage is read there with the limit set aside
    /// (`vetoed::Half::emptied_limit_aside`): a prefill at its limit stays
    /// in that pass until another filter refuses it. So where the prefill
    /// side's profile lists the limit among its filters changes no answer
    /// there, under filters whose word on a row does not turn on how
    /// often they are asked. The gateway's own pair selection asks its
    /// overload shed ahead of its capacity test (`pair_pass` in
    /// `model_gateway/src/routers/common/placement.rs`).
    ///
    /// Where the prefill side's run sheds, no default pairs and no prefill
    /// waits on its limit alone, one question says whether the request is
    /// shed as saturated: the count of the one list the gateway's own pair
    /// selection judges (`vetoed::Half::count`). That list is every decode
    /// at the request's places, a decode that pairs with no prefill too,
    /// where a prefill at those places that pairs with a decode of the pool
    /// stands as it is, its limit aside and with no lift. Else it is those
    /// prefills. Under a binding that sheds, the shed is saturated exactly
    /// where that list is wholly over its thresholds, and the decision
    /// says the list (`ShedOver`). Under a binding that steers no such
    /// shed is saturated. A picker that declines keeps its own reason.
    /// Every other shed there has no endpoint, under
    /// the name of the stage that left a pass nobody. Where that stage is
    /// the overload veto, whose name is the word for saturation, the shed
    /// takes the name of the stage that refused a worker with room of the
    /// list judged (`vetoed::Half::refused_room`): the health stage for a
    /// worker under a liveness doubt, and `partnered` for one that stands
    /// and pairs with nobody that stands. The gateway's own pair selection
    /// judges so (`pair_pass` and `failure_from` in
    /// `model_gateway/src/routers/common/placement.rs`: the pass's two
    /// branches, where no prefill is eligible and where none is open).
    ///
    /// The question is asked over whole lists alone. Behind a part of a
    /// list (`PairPools::part`: a hard demand its door judged, or a
    /// request the encode pass is laid for) it is not asked. The shed
    /// keeps the name of the stage that left a pass nobody: it is
    /// saturated where that stage is the overload veto. It says its list
    /// there as it did before the question stood: the prefills that pair
    /// where no prefill stands, and else the decode list whole, each
    /// where wholly over its thresholds. That selection judges another
    /// list there than this decision is handed (`admitted_pool` in the
    /// same file), so the question over the part would answer for
    /// another list.
    ///
    /// Each shed of that question names the side that selection refuses
    /// for ([`Shed::side`](crate::engine::decision::Shed::side)). Three
    /// tests of the lists whole stand ahead of its question in
    /// `pair_pass`, in this order. A prefill pool that holds no worker is
    /// refused for the prefill side. A decode pool that holds none is
    /// refused for the decode side. So is a pool no prefill of which
    /// pairs with any decode, by its index's refusal (`PairingRefusal` in
    /// `model_gateway/src/routers/common/pair_index.rs`). Else the side is
    /// the list the question judges. Behind a part the same tests name
    /// the side over the part, and it may be another than that
    /// selection's role: that selection tests its lists whole.
    ///
    /// Where the host states first partners ([`PairPools::first`]), the
    /// pair is made among the prefills with a first partner open and those
    /// partners, while there is one. The prefill profile is shown those
    /// prefills alone, the admission limit is judged over them, and the
    /// decode is one of the pick's first partners. Where the profile's own
    /// stages refuse each such prefill, the pair is made among every
    /// prefill with an open partner, as it is where no first partner is
    /// open. The pair's default reads every partner alike. The gateway's
    /// own pair selection orders its candidates so
    /// (`model_gateway/src/routers/common/placement.rs:2479-2502`,
    /// `:2532-2545`, `:2615-2618`, and its default at `:1804-1813`, on main
    /// `30f636706`).
    ///
    /// Where the decode side states a number
    /// ([`Constraints::target_worker`] of `decode_cx`), the pair's decode
    /// is the one that number reaches among the partners of the prefill
    /// chosen, at the decode side's places: its first partners where the
    /// pair is made among those. Where the number reaches none, or the
    /// decode it reaches is shed for want of an endpoint or as over its
    /// thresholds, the pair is shed under `target-worker` and the partners
    /// of no other prefill are asked. The pair's default reads no number.
    /// A number beside no places of the decode side is declined under that
    /// name. The gateway's own pair selection counts the open partners of
    /// its one pick so, and reads no header for its default
    /// (`model_gateway/src/routers/common/placement.rs:2884-2887`,
    /// `:2914-2920`, `:2729-2747`, on main `4b41f39a7`).
    pub fn decide_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // The word of the list a shed is made over is this decision's: one
        // an earlier decision left goes here ([`Scratch::shed_over`]).
        scratch.shed_over = None;
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        // A pair's columns ride its two pools. One on the request, or rows
        // named on it, name neither pool: refused, where they would be
        // dropped in silence and both sides scored as if nothing were held.
        if carries_a_column(facts) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        if counts_no_place(decode_cx) {
            return number_not_counted(bare, bare.profile);
        }
        // Each side's stages read their own pool and its own column.
        let (Some((prefills, prefill_facts)), Some((decodes, decode_facts))) = (
            pools.side(PairSide::Prefill, facts),
            pools.side(PairSide::Decode, facts),
        ) else {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        };
        // One binding for both sides: the prefill's lookup, whose decode
        // profile is the pair's decode side. A binding of the single plan
        // declines a pair decision: the host and the spec disagree.
        let Some(entry) = self.pair_entry(facts.model) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        // The binding's word for a pool a soft veto empties: the live
        // decode side is put to the all-vetoed rule by it ([`Set::ruled`]).
        let (binding, vetoed) = (&entry.binding, entry.vetoed);
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        let (Some(prefill_profile), Some(decode_profile)) = (
            self.profiles.get(usize::from(binding.profile)),
            self.profiles.get(usize::from(decode_index)),
        ) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        // A plugin reads a row's own fields: a side that shows its rows is
        // handed rows written for the decision. A mask beside that side's
        // rows says they are not, and the decision is declined before
        // either scope opens. A side that shows nothing reads no mask here.
        let prefill_shown = self.marks[usize::from(binding.profile)].scoped();
        let decode_shown = self.marks[usize::from(decode_index)].scoped();
        if (prefill_shown && !cx.stalled.is_empty())
            || (decode_shown && !decode_cx.stalled.is_empty())
        {
            // The side whose shown rows stand beside a mask is the one that
            // declines: the prefill's first.
            let prefill = prefill_shown && !cx.stalled.is_empty();
            let declining = if prefill {
                binding.profile
            } else {
                decode_index
            };
            return rows_not_written(bare, declining);
        }
        // One scope per side: the WIT view has one candidate list, so the
        // prefill stages read the prefill pool and the decode stages the
        // decode pool, through every decode run and the partner passes.
        let _prefill_opened =
            prefill_shown.then(|| prefill_profile.open(&prefill_facts, prefills, pool));
        let _decode_opened =
            decode_shown.then(|| decode_profile.open(&decode_facts, decodes, pool));
        // The prefill pool narrows to the prefills with a partner the decode
        // profile's hard filters keep, before the prefill picker truncates to
        // its targets: a pair that exists is found. A decode of class 0 is
        // outside any pairing and partners nobody.
        debug_assert!(
            pools.table.validate().is_ok(),
            "the pair table is not valid: {}",
            pools
                .table
                .validate()
                .map_or_else(|clause| clause.to_string(), |()| String::new())
        );
        // A decode outside its side's places partners nobody.
        let mut open = places::rows(decodes, decode_cx)
            .filter(|d| {
                d.pair_class != 0
                    && decode_profile
                        .filters
                        .iter()
                        .filter(|f| !f.soft())
                        .all(|f| f.keep(&decode_facts, d, decode_cx))
            })
            .map(|d| d.pair_class);
        pools
            .table
            .partnered(open.by_ref(), &mut scratch.classes, &mut scratch.marks);
        // The pass may end ahead of the last decode. A filter the engine
        // does not hold itself is asked about every decode all the same,
        // in the pass's order: what it keeps of its asks stands.
        let asked_whole = self.row_tests[usize::from(decode_index)].is_none();
        if asked_whole {
            open.for_each(drop);
        }
        // No prefill class has an open decode. The decode pool is put to
        // the binding's all-vetoed rule once. Where the overload veto alone
        // left it nobody, every decode the hard filters keep with that veto
        // lifted opens its prefills. A soft filter of that pass hides no
        // class: a room narrows a prefill's partners, not the prefills. The
        // decode run of the chosen prefill then meets its partners emptied
        // the same way, and the rule picks among them. Decodes the rule
        // keeps past a doubt open no prefill: the prefill side sheds, and
        // that pool goes to the pair's default below.
        if scratch.classes.is_empty() {
            let whole = Set::side(decodes, None, None).ruled(vetoed);
            let buf = &mut scratch.decode;
            let emptied = filter(decode_profile, &decode_facts, decode_cx, buf, whole);
            let lifted = !buf.kept.is_empty() && buf.lifted == LiftedVeto::Overload;
            if emptied.is_some() && lifted {
                // The facts and the constraints of the rule's own pass
                // ([`vetoed::emptied`]).
                let all = RequestFacts {
                    tried: &[],
                    ..decode_facts
                };
                let pass = Constraints {
                    overload_lifted: true,
                    ..*decode_cx
                };
                let hard = || decode_profile.filters.iter().filter(|f| !f.soft());
                // Here too a decode outside its side's places partners
                // nobody: the rule's pass above opened with those places.
                // So does a decode of class 0.
                let mut kept = places::rows(decodes, decode_cx)
                    .filter(|d| d.pair_class != 0 && hard().all(|f| f.keep(&all, d, &pass)))
                    .map(|d| d.pair_class);
                pools
                    .table
                    .partnered(kept.by_ref(), &mut scratch.classes, &mut scratch.marks);
                if asked_whole {
                    kept.for_each(drop);
                }
            }
        }
        // The prefill buffer's mark of a lifted veto is cleared for this
        // decision. The pair's default alone writes it again, by the tier
        // it paired in ([`vetoed::paired`]): each side's shadow reads there
        // whether the default made the pair ([`Self::shadow_side`]).
        scratch.single.lifted = LiftedVeto::None;
        // A prefill profile that weighs load against affinity runs with the
        // overload veto lifted ([`Self::weighing_lift`]): its run, the
        // admission limit's tests and the count of its routable prefills
        // read `run_cx`. The lifted constraints are a second value beside
        // the stated ones, which the pair's default still reads: it judges
        // the pool as it stands with no lift.
        let lifted_run = self.weighing_lift(binding.profile, vetoed, cx);
        let run_cx = lifted_run.as_ref().unwrap_or(cx);
        debug_assert!(
            pools.first.within(&pools.table).is_ok(),
            "the first partners do not stand within the pair table: {:?}",
            pools.first.within(&pools.table)
        );
        // While a prefill has a first partner open, the pair is made among
        // those prefills and their first partners ([`FirstPartners`]): the
        // prefill profile is shown their classes alone.
        let mut firsts = pools
            .first
            .open(&pools.table, &scratch.marks, &mut scratch.firsts);
        let shown: &[u32] = if firsts {
            &scratch.firsts
        } else {
            &scratch.classes
        };
        // The prefill side's run over the classes it is shown, the first
        // partners' or every open partner's: by its rule or its picker's
        // assignment where one places the side ([`Self::placed_prefill`]),
        // else by its profile. The classes are chosen ahead of that test,
        // where the first partners' answer stands, and the test is asked
        // again where the second run needs it. Chosen behind the test, the
        // classes were chosen by value, and a pair under no rule paid 9
        // instructions for it.
        let mut prefill =
            if self.places_side(entry, binding.profile, PairSide::Prefill, &prefill_facts) {
                self.placed_prefill(
                    firsts,
                    binding,
                    prefill_profile,
                    &prefill_facts,
                    pool,
                    prefills,
                    run_cx,
                    scratch,
                )
            } else {
                run_profile(
                    prefill_profile,
                    binding.profile,
                    &prefill_facts,
                    pool,
                    Set::side(prefills, None, Some(shown)),
                    run_cx,
                    &mut scratch.single,
                )
            };
        if firsts {
            let stands = match &prefill {
                Decision::Pick(_) => true,
                // One of those prefills that stands but for its admission
                // limit keeps the request for them: it waits, though a
                // prefill with no first partner open has room. Where none
                // stands, the pairs with any open partner are next.
                Decision::Shed(refused) if refused.reason != ShedReason::PolicyDeclined => {
                    let takes = Half::new(
                        prefill_profile,
                        binding.profile,
                        &prefill_facts,
                        prefills,
                        run_cx,
                    );
                    if takes.waits_among(&scratch.firsts) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, refused.trace)
                            .of_side(PairSide::Prefill);
                    }
                    false
                }
                // A picker's decline passes through, as the prefill side's.
                _ => return prefill.of_side(PairSide::Prefill),
            };
            if stands {
                // The classes the prefill profile was shown stand in
                // `classes`, as a side's shadow reads them.
                std::mem::swap(&mut scratch.classes, &mut scratch.firsts);
            } else {
                firsts = false;
                prefill = if self.places_side(
                    entry,
                    binding.profile,
                    PairSide::Prefill,
                    &prefill_facts,
                ) {
                    self.placed_prefill(
                        false,
                        binding,
                        prefill_profile,
                        &prefill_facts,
                        pool,
                        prefills,
                        run_cx,
                        scratch,
                    )
                } else {
                    run_profile(
                        prefill_profile,
                        binding.profile,
                        &prefill_facts,
                        pool,
                        Set::side(prefills, None, Some(&scratch.classes)),
                        run_cx,
                        &mut scratch.single,
                    )
                };
            }
        }
        // Whether the pair is made among first partners, left for the
        // decode's shadow: it reads the partners this decision's decode
        // side reads ([`Self::shadow_side`]).
        scratch.among_first = firsts;
        // The prefill's pick is read where its decision stands: a pick is
        // 256 bytes, and taking it out copies them.
        let Decision::Pick(prefill_pick) = &prefill else {
            // A prefill pool its filters emptied goes to the pair's default,
            // then to the admission limit's shed, for a prefill with an open
            // partner alone, and then to the one question that says whether
            // the request is shed as saturated: the count of the list the
            // gateway's own pair selection judges, asked over whole lists
            // alone. Every other shed is named by a stage. It is the
            // prefill side's where no prefill stands but for its admission
            // limit, read with that limit set aside, and "none" where that
            // pass ends at no stage. It is "partnered" for a class with no
            // decode at all and for a routable prefill of class 0, and else
            // what the decode pool's own stages say about the partners of
            // the prefills that stand so (those were dropped by the
            // partnered test alone). Where those stages name none, the
            // prefill's own shed stands. Over whole lists each such shed
            // has no endpoint, and where its stage is the overload veto,
            // the list judged names it again ([`Half::refused_room`]).
            // Behind a part a pass the overload veto ended is shed as
            // saturated. A picker's decline passes through.
            if let Decision::Shed(prefill_shed) = &prefill {
                if prefill_shed.reason != ShedReason::PolicyDeclined {
                    // A refused pool vetoed on every usable worker goes to
                    // the pair of the least load ([`vetoed::paired`]). The
                    // default reads the classes with an open partner, so it
                    // is asked before they are written over.
                    let (first, second) = (binding.profile, decode_index);
                    let sides = [
                        Half::new(prefill_profile, first, &prefill_facts, prefills, cx),
                        Half::new(decode_profile, second, &decode_facts, decodes, decode_cx),
                    ];
                    // The decision says the list it sheds over as saturated
                    // under a binding that sheds a list wholly over its
                    // thresholds, and under no other ([`ShedOver`]).
                    let says = vetoed == AllVetoed::Shed;
                    let words = (vetoed == AllVetoed::Steer, self.weighs(first));
                    if let Some(pair) = vetoed::paired(sides, &pools.table, words, pool, scratch) {
                        return pair;
                    }
                    // The admission limit is judged over the prefills with
                    // an open partner. Where one of them stands but for its
                    // limit, the limit alone left the run nobody: the shed
                    // is the limit's, and a host waits on it. Asked before
                    // the classes with an open partner are written over,
                    // of a side that reads the constraints the run itself
                    // read: the default above read the stated ones.
                    let takes = Half::new(prefill_profile, first, &prefill_facts, prefills, run_cx);
                    if takes.waits_among(&scratch.classes) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, prefill_shed.trace)
                            .of_side(PairSide::Prefill);
                    }
                    // No stage placed the pair, no default paired it and no
                    // prefill waits on its limit alone. The prefills the
                    // request may take that stand, the limit set aside: one
                    // outside its side's places is no routable prefill of
                    // it, and one at its limit with no open partner is named
                    // by its partners' stage as any other. A prefill is
                    // routable under the constraints the run itself read.
                    // Their classes are written over the classes with an
                    // open partner, which nothing below reads.
                    let has_partner = |c: &Candidate| vetoed::pairs_in(c, &pools.table, decodes);
                    // Whether a prefill of the pool pairs with a decode of
                    // it at all, in whatever state, asked once: where none
                    // does, nothing below looks for a prefill's partner
                    // again.
                    let any_pairs = prefills.iter().any(&has_partner);
                    let pairs = |c: &Candidate| any_pairs && has_partner(c);
                    // Whether one of them that pairs with a decode of the
                    // pool stands in the pool as it is with no lift: behind
                    // a weighing profile's lifted run the stated
                    // constraints are asked of it once more, as the default
                    // above read them.
                    let unlifted =
                        |c: &Candidate| lifted_run.is_none() || sides[0].but_for_the_limit(c);
                    let (mut routable, mut pair_stands) = (0usize, false);
                    scratch.classes.clear();
                    for c in takes.places().filter(|c| takes.but_for_the_limit(c)) {
                        routable += 1;
                        if c.pair_class != 0 {
                            scratch.classes.push(c.pair_class);
                        }
                        pair_stands = pair_stands || (pairs(c) && unlifted(c));
                    }
                    scratch.classes.sort_unstable();
                    scratch.classes.dedup();
                    // One question says whether the shed is saturation, the
                    // gateway's own pair selection's (`pair_pass`,
                    // `model_gateway/src/routers/common/placement.rs`,
                    // where no prefill is eligible or open). It judges one
                    // list: every decode at the request's places, a decode
                    // that pairs with no prefill too, where a prefill at
                    // those places that pairs with a decode of the pool
                    // stands; and else those prefills. Under a binding that
                    // sheds, the shed is saturated exactly where that list
                    // is wholly over its thresholds ([`Half::count`]), and
                    // the decision says the list. Under a binding that
                    // steers no shed here is saturated: that selection
                    // sheds for overload under its shed option alone
                    // (`failure_from` there). The count reads the
                    // list's rows once more, by the health stage's and the
                    // overload veto's tests, behind a shed alone. For the
                    // prefills a decode that pairs with each is looked for
                    // too, at worst every decode once a prefill.
                    let judged = || {
                        if pair_stands {
                            (sides[1].count(sides[1].places()), ShedOver::Decodes)
                        } else {
                            let paired = sides[0].places().filter(|&c| pairs(c));
                            (sides[0].count(paired), ShedOver::Prefills)
                        }
                    };
                    // Every shed below names one side, the role that
                    // selection refuses for. It asks three tests of its
                    // lists whole ahead of its question, in this order: a
                    // prefill pool that holds no worker is refused for the
                    // prefill's role, a decode pool that holds none for the
                    // decode's, and a pool no prefill of which pairs with
                    // any decode for the decode's too. Else the side is the
                    // list the question judges.
                    let side = if pair_stands {
                        PairSide::Decode
                    } else if prefills.is_empty() || any_pairs {
                        PairSide::Prefill
                    } else {
                        PairSide::Decode
                    };
                    // The question is asked over whole lists alone. Behind a
                    // part ([`PairPools::part`]) that selection judges a
                    // wider list than this decision is handed: the
                    // workers on the request's wire, and every worker
                    // where none is on it (`admitted_pool` there). So no
                    // shed behind a part is made saturated by the count,
                    // and a pass the overload veto ended keeps its name
                    // below.
                    let part = pools.part;
                    let asked = (says && !part).then(&judged);
                    if let Some((Count::WhollyOver(_), over)) = asked {
                        let veto = NotOverloaded.name();
                        scratch.shed_over = Some(over);
                        return shed(reason_of(veto), veto, prefill_shed.trace).of_side(side);
                    }
                    // Else the shed takes a first name. Where no prefill of
                    // the request's stands but for its limit it is the
                    // stage that leaves the prefill side's pass nobody, the
                    // limit aside, and `none` where that pass ends at no
                    // stage. Where those that stand pair with no decode it
                    // is `partnered`. Else it is what the decode pool's own
                    // stages say about the partners of the prefills that
                    // stand, and the prefill's own shed stands where they
                    // name none. A first name that is the overload veto's
                    // sheds as saturated behind a part, and over whole
                    // lists gives way to the name the list judged gives
                    // it, below. Every other shed has no endpoint.
                    let named = if routable == 0 {
                        // The limit's name says a wait for room, and no
                        // prefill here stands but for its limit. So the
                        // side's stage is read by a pass that sets the
                        // limit aside, wherever the profile lists it: a
                        // prefill at its limit stays in that pass until
                        // another filter refuses it.
                        let buf = &mut scratch.single;
                        Some(takes.emptied_limit_aside(buf).unwrap_or(NO_STAGE))
                    } else if scratch.classes.is_empty() {
                        Some(PARTNERED)
                    } else {
                        // Their partners are the decodes of the classes they
                        // pair with. An empty decode pool is that class for
                        // every prefill: its "none" leaves the prefill's own
                        // shed standing.
                        pools.table.listed(&mut scratch.classes, &mut scratch.marks);
                        let partners = Set::side(decodes, None, Some(&scratch.classes));
                        let buf = &mut scratch.decode;
                        filter(decode_profile, &decode_facts, decode_cx, buf, partners)
                            .filter(|&stage| stage != NO_STAGE)
                    };
                    let reason = named.map_or(prefill_shed.reason, reason_of);
                    // A pass the overload veto ended names that veto, the
                    // engine's word for saturation, which the count did not
                    // bear. The shed has no endpoint, under the stage that
                    // refused a worker with room of the list judged
                    // ([`Half::refused_room`]). Behind a part no list was
                    // judged, and the pass's name stands. The decision
                    // says its list there as it did before the question
                    // stood, where a stage named the shed: the prefills
                    // that pair where no prefill stands, and else the
                    // decode list whole, each where wholly over its
                    // thresholds.
                    if reason == ShedReason::Saturated && part && says && named.is_some() {
                        let (read, over) = if routable == 0 {
                            let paired = takes.places().filter(|&c| pairs(c));
                            (takes.count(paired), ShedOver::Prefills)
                        } else {
                            (sides[1].count(sides[1].places()), ShedOver::Decodes)
                        };
                        if let Count::WhollyOver(_) = read {
                            scratch.shed_over = Some(over);
                        }
                    }
                    if reason == ShedReason::Saturated && !part {
                        let (read, over) = asked.unwrap_or_else(&judged);
                        let judged = &sides[usize::from(over == ShedOver::Decodes)];
                        let stage = judged.refused_room(read);
                        return shed(ShedReason::NoEndpoints, stage, prefill_shed.trace)
                            .of_side(side);
                    }
                    if let Some(stage) = named {
                        return shed(reason, stage, prefill_shed.trace).of_side(side);
                    }
                    // The prefill's own shed stands, where the decodes at
                    // the request's places hold nobody or keep a partner.
                    return prefill.of_side(side);
                }
            }
            // A picker's decline passes through, as the prefill side's.
            return prefill.of_side(PairSide::Prefill);
        };
        // A pick the weighing lift alone made possible says so, as a kept
        // key's pick does ([`Self::keep`]): where the veto as it stands
        // leaves the prefills nobody, the pair's trace names the lifted
        // overload veto. The prefill buffer takes no mark: the pair is the
        // profile's, and each side's shadow runs beside it
        // ([`Self::shadow_side`]). A pick beside a prefill under its
        // thresholds carries no mark.
        let lift_placed = lifted_run.is_some() && {
            let shown = Set::side(prefills, None, Some(&scratch.classes));
            let buf = &mut scratch.single;
            vetoed::all_vetoed(prefill_profile, &prefill_facts, shown, cx, buf)
        };
        let mut last_shed: Option<Shed> = None;
        for prefill_target in prefill_pick.targets.iter() {
            // A target's position is the host's; a row outside the slice
            // costs that target its turn, never the process.
            let Some(class) = prefills
                .get(prefill_target.position as usize)
                .map(|c| c.pair_class)
            else {
                continue;
            };
            // The decode profile runs over the chosen prefill's partners
            // alone: the decodes of the classes its class pairs with. A
            // class is a per-candidate fact and the table the decision's
            // own, so no host callback is needed to narrow the set.
            let partners = pools.partners(&class, firsts);
            let partners = Set::side(decodes, Some(partners), None).ruled(vetoed);
            let mut decision =
                if self.places_side(entry, decode_index, PairSide::Decode, &decode_facts) {
                    self.placed_side::<true>(
                        binding,
                        decode_index,
                        decode_profile,
                        &decode_facts,
                        pool,
                        partners,
                        decode_cx,
                        scratch,
                    )
                } else {
                    run_profile(
                        decode_profile,
                        decode_index,
                        &decode_facts,
                        pool,
                        partners,
                        decode_cx,
                        &mut scratch.decode,
                    )
                };
            match &mut decision {
                // The decode's pick becomes the pair's where it stands: its
                // targets name the prefill, its trace is the prefill's, and
                // that decision is the one returned.
                Decision::Pick(paired) => {
                    // Every field of a pick is named here, with no rest:
                    // a field added later fails this build until the pair
                    // says whose value it keeps.
                    let Pick {
                        targets,
                        trace,
                        session_verdict,
                    } = paired;
                    // Each side's own verdict and its own word of who
                    // placed it: the decode side's verdict in the first
                    // target's pair and its word in every pair, the
                    // prefill side's as the pick's.
                    let placed = prefill_pick.targets.placing();
                    targets.pair_under(prefill_target, *session_verdict, placed);
                    // The pair's trace is the prefill's, with the mark of
                    // the veto the rule lifted for the decode side, or of
                    // the overload veto the prefill's own run was placed
                    // past. The decode side's own mark stands where it names
                    // one. A first decision puts that side to the rule's
                    // overload tier alone, so the two marks differ only
                    // under a filter whose `retain` drops a row its `keep`
                    // keeps, which [`Filter::retain`] forbids: the liveness
                    // mark of such a decode side is kept.
                    let lifted = trace.lifted;
                    *trace = prefill_pick.trace;
                    trace.lifted = match lifted {
                        LiftedVeto::None if lift_placed => LiftedVeto::Overload,
                        lifted => lifted,
                    };
                    *session_verdict = prefill_pick.session_verdict;
                    // The prefill side's run claimed the prefill it placed
                    // first. Where the pair formed on another, the claim
                    // moves to that one here, ahead of any dispatch: the
                    // key's next decision follows a prefill that pairs.
                    // Where the table moves none, the claim stands
                    // unconfirmed no longer or that prefill is a pin
                    // already: the pick states a pin's verdict, and its
                    // dispatch moves the pins as a pin's alternate does.
                    if let SessionVerdict::Claimed(claimed) = prefill_pick.session_verdict {
                        if claimed != prefill_target.id {
                            let (key, paired) = (prefill_facts.session(), prefill_target.id);
                            let at = prefill_facts.now_micros;
                            *session_verdict = self.sessions.move_claim(key, claimed, paired, at);
                        }
                    }
                    // The prefill side's run met its key's pin at the cap
                    // and placed the request again, and moved no pin
                    // ([`Self::moved`]). The pin moves here, where the
                    // pair forms and ahead of any dispatch, to the
                    // prefill the request goes to. A prefill passed over
                    // is a pin at no instant, so no pin is taken back,
                    // and none another request was sent to meanwhile.
                    if prefill_pick.session_verdict == SessionVerdict::Respill {
                        let (key, at) = (prefill_facts.session(), prefill_facts.now_micros);
                        self.sessions.respilled(key, prefill_target.id, at);
                    }
                    return decision;
                }
                Decision::Shed(shed) => {
                    last_shed = Some(*shed);
                    // The decode a request names is counted behind the one
                    // prefill: where it is shed, no other prefill's
                    // partners are asked.
                    if decode_cx.target_worker.is_some() {
                        break;
                    }
                }
                // A queue returns with the prefill side's claim standing.
                // No stage issues one today.
                Decision::Queue(_) => return decision,
            }
        }
        // No prefill paired: the claim the prefill side's run made is
        // withdrawn, and the key's next request is placed again.
        if let SessionVerdict::Claimed(claimed) = prefill_pick.session_verdict {
            let (key, at) = (prefill_facts.session(), prefill_facts.now_micros);
            self.sessions.move_claim(key, claimed, WorkerId(0), at);
        }
        // A prefill was picked and no decode pairs with it: the decode side
        // is the one left with nobody.
        match last_shed {
            // The shed of the one decode a request names takes the
            // target's name.
            Some(last) if decode_cx.target_worker.is_some() => target_missed(Decision::Shed(last)),
            Some(last) => Decision::Shed(last),
            None => shed(ShedReason::NoEndpoints, PARTNERS, prefill_pick.trace),
        }
        .of_side(PairSide::Decode)
    }

    /// Whether some binding's pair may be asked by its draw: what a pair's
    /// decision asks first, at one load ([`Self::pair_drawn_for`] asks the
    /// rest of the request's own binding).
    #[cfg(feature = "in-process")]
    pub(crate) const fn pairs_drawn(&self) -> bool {
        self.pairs_drawn
    }

    /// Whether each side of a pair may be
    /// asked by its draw ([`run_side_drawn_written`]), which writes the records
    /// it asks about and no other. Both profiles are a pick of two whose
    /// filters fold into one test of a row, each list holds its profile's
    /// floor of rows, and the request states nothing the pair's whole
    /// decision reads beside its two picks. Asked before a record is
    /// written; answers the two profiles.
    #[cfg(feature = "in-process")]
    pub(crate) fn pair_drawn_for(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        rows: [usize; 2],
        cx: &Constraints<'_>,
    ) -> Option<PairDrawn> {
        if carries_a_column(facts) || counts_no_place(cx) || !facts.session_key.is_none() {
            return None;
        }
        let entry = self.pair_entry(facts.model)?;
        if !entry.pair_drawn {
            return None;
        }
        let binding = &entry.binding;
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        for (index, rows) in [(binding.profile, rows[0]), (decode_index, rows[1])] {
            let at = usize::from(index);
            if rows < self.written_from.get(at).copied().unwrap_or(usize::MAX)
                || (self.marks[at].first() & (Marks::ENGINES | Marks::OWNS) != 0
                    && !pool.engines.is_empty())
            {
                return None;
            }
        }
        if self
            .weighing_lift(binding.profile, entry.vetoed, cx)
            .is_some()
        {
            return None;
        }
        Some(PairDrawn {
            profiles: [binding.profile, decode_index],
        })
    }

    /// A pair's decision where each side is
    /// asked by its draw: [`Self::pair_drawn_for`] said yes, both lists
    /// stand, and every row of a side pairs with every row of the other.
    /// The prefill by the draw of its profile, then the decode by the draw
    /// of its own, joined as [`Self::decide_pair`] joins its two picks.
    /// `None` where a side's draw answers no pick: the caller then decides
    /// whole, over records it writes itself.
    #[cfg(feature = "in-process")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair's facts, each side's rows with its writer, and the engine's scratch"
    )]
    pub(crate) fn decide_pair_drawn(
        &self,
        drawn: PairDrawn,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        prefills: DrawnSide<'_>,
        decodes: DrawnSide<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let PairDrawn {
            profiles: [prefill_index, decode_index],
        } = drawn;
        scratch.shed_over = None;
        scratch.single.lifted = LiftedVeto::None;
        let prefill_facts = RequestFacts {
            phase: Phase::Prefill,
            named: None,
            signal_cells: None,
            decode_ring_positions: None,
            ..*facts
        };
        let decode_facts = RequestFacts {
            phase: Phase::Decode,
            seed: Draw::new(facts.seed).next_u64(),
            named: None,
            signal_cells: None,
            ring_positions: facts.decode_ring_positions.unwrap_or(facts.ring_positions),
            decode_ring_positions: None,
            ..*facts
        };
        let (fixed, rows, write) = prefills;
        let prefill_rows = fixed.len();
        let prefill = run_side_drawn_written(
            self,
            prefill_index,
            &prefill_facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.single,
        );
        let prefill_pick = prefill?;
        let prefill_target = prefill_pick.targets.iter().next()?;
        let (fixed, rows, write) = decodes;
        let mut decision = Decision::Pick(run_side_drawn_written(
            self,
            decode_index,
            &decode_facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.decode,
        )?);
        let Decision::Pick(paired) = &mut decision else {
            return None;
        };
        let Pick {
            targets,
            trace,
            session_verdict,
        } = paired;
        let placed = prefill_pick.targets.placing();
        targets.pair_under(prefill_target, *session_verdict, placed);
        let lifted = trace.lifted;
        *trace = prefill_pick.trace;
        // The whole pass counts the prefills it keeps, and a pair that
        // draws is asked only where it keeps every one of them.
        trace.survivors = Survivors::counted(prefill_rows as u32);
        trace.lifted = lifted;
        *session_verdict = prefill_pick.session_verdict;
        Some(decision)
    }

    /// A pair's answer over two lists of which one holds no row: one side
    /// alone is no pair, and the engine answers over no candidate, under
    /// the request's own constraints `cx` for both sides. `pool` holds
    /// the prefill list's facts, and `prefills` says whether that list
    /// holds a row. The refusal names the side of the list with none, the
    /// prefills' first. The engine read neither list, so where its refusal
    /// names a side it names the prefills': where that list holds a row,
    /// the list with none is the decodes'.
    ///
    /// A pair's decision in process answers so ahead of the request's
    /// demand. An encode request's answers so where its demand left the
    /// other list nobody, with an item behind its encoders' own answers:
    /// its caller reads the answer of its pair alone. Cold and out of
    /// line: its code stays off the path of a pair both of whose lists
    /// hold a row, and so does the arm that calls it.
    #[cold]
    #[inline(never)]
    pub(crate) fn decide_over_an_empty_list(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        prefills: bool,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let none = PoolFacts {
            generation: pool.generation,
            ..PoolFacts::default()
        };
        let mut decision = self.decide_pair(facts, &none, PairPools::default(), cx, cx, scratch);
        if let Decision::Shed(shed) = &mut decision {
            if shed.side.is_some() && prefills {
                shed.side = Some(PairSide::Decode);
            }
        }
        decision
    }

    /// Whether the profile at `index` weighs load against affinity by a
    /// gate of its own: one of its scorers says so
    /// ([`Scorer::balances_load_against_affinity`]), read into the
    /// profile's marks where the engine is built. The one reader of that
    /// mark. A pair's decision asks it twice of its prefill profile: for
    /// the lifted run ([`Self::weighing_lift`]) and for the word its
    /// default is handed ([`vetoed::paired`]). What a stage is fed is
    /// another word ([`Needs::affinity`]): a stage that reads the column
    /// and has no gate lifts nothing.
    pub(crate) fn weighs(&self, index: u16) -> bool {
        let marks = self.marks.get(usize::from(index));
        marks.is_some_and(|marks| marks.weighs())
    }

    /// The constraints a pair's prefill profile at `index` runs under where
    /// it weighs load against affinity: `cx` with the overload veto lifted.
    /// `None` where the stated ones stand: under a binding that sheds
    /// (`rule`), and under a profile none of whose scorers says that it
    /// weighs ([`Self::weighs`]).
    ///
    /// A scorer that weighs has a gate of its own, which moves a request
    /// off a holder loaded well past the pool. The veto would move a
    /// session off the prefill that holds its prefix each time that prefill
    /// crosses a threshold, and under sustained load the prefills cross
    /// them in turn. So such a profile places among the prefills over their
    /// thresholds too, as the gateway's own pair selection leaves the veto
    /// off a prefill pool whose policy says the same word of itself
    /// (`model_gateway/src/routers/common/placement.rs:2575-2577`, at main
    /// `82a135b1b`). The lift is the constraint a kept key's run is given
    /// ([`Self::keep`]). A profile whose stages only read the affinity
    /// column keeps the veto: a need says what a stage is fed, and nothing
    /// of a gate.
    fn weighing_lift<'a>(
        &self,
        index: u16,
        rule: AllVetoed,
        cx: &Constraints<'a>,
    ) -> Option<Constraints<'a>> {
        let weighs = rule == AllVetoed::Steer && self.weighs(index);
        // Every pair's decision asks, so the copy is made only where the
        // lift stands.
        if !weighs {
            return None;
        }
        Some(Constraints {
            overload_lifted: true,
            ..*cx
        })
    }

    /// Whether the pair binding at `at` names a shadow for `side` that
    /// the build found free to run beside the live profile
    /// ([`side_shadows`]). That shadow is then due beside each decision
    /// the pair's profiles make, and a host says one that
    /// [`Self::shadow_side`] does not run as skipped.
    pub(crate) fn shadows_side(&self, at: u32, side: PairSide) -> bool {
        let sides = self.shadow_sides.get(at as usize);
        sides.is_some_and(|sides| sides[side.index()].is_some())
    }

    /// Whether the shadow the pair binding at `at` names for `side` is
    /// not asked: its profile would be handed another column than the
    /// live side's pool carries ([`side_reads_another_column`]). A host
    /// says such a side's shadow as skipped.
    pub(crate) fn side_apart(&self, at: u32, side: PairSide) -> bool {
        let sides = self.side_apart.get(at as usize);
        sides.is_some_and(|sides| sides[side.index()])
    }

    /// One side of a pair's decision, made once more with that side's
    /// shadow profile over the rows its live side decided over. The
    /// prefill's shadow reads the prefills the live prefill was shown,
    /// those with a partner: their classes stand in `scratch` as the
    /// pair's decision left them, so a caller hands the scratch that
    /// decision wrote. The decode's shadow reads the partners of the
    /// live pair's prefill, never of a prefill of its own: its first
    /// partners where the pair was made among them, by the word that
    /// decision left in the scratch ([`Scratch::among_first`]). `at` is
    /// the pair binding's position ([`Self::shadow_for`]) and `live` the
    /// pair's pick, made over the same `pools`. The run writes the
    /// scratch and the shadow profile's stages, and nothing a commit
    /// reads.
    ///
    /// A side's shadow is lent the ring its request lends by the rule of a
    /// single's shadow ([`Self::shadow`]), held against the word its
    /// side's list was stated by. Where the pair is lent one list of
    /// places for its two sides ([`Self::states_key_place`]), that is
    /// both live sides' word: a shadow that reads a ring by another place
    /// than either live side states is lent none. Where each side is
    /// stated its own places ([`RequestFacts::decode_ring_positions`]),
    /// it is the side's own live profile's word: a shadow that reads the
    /// ring as its live side does is lent the ring with its side's
    /// places ([`PairPools::side`]), and one that reads it as the other
    /// side does is lent none. A shadow lent none places on its own ring
    /// ([`Self::unlent`]).
    ///
    /// A shadow that reads a ring where neither live side does is stated
    /// its own place on the ring the request lends
    /// ([`Self::side_ring_profile`]), asked of `lends` behind the filter
    /// half of its run ([`run_side_lent`]): a shadow whose set the filters
    /// emptied, which the all-vetoed rule steers or sheds, asks nothing.
    /// Neither does the pair's default that a side's shadow follows, where
    /// no filter of the profile declares the ring need: no step of it
    /// scores.
    ///
    /// A shadow profile with a filter that declares the ring need
    /// ([`Filter::needs`]) asks ahead of the filter half of its run,
    /// whatever its filters then keep, and once a decision: a prefill's
    /// shadow that runs again reads the answer of its first run. The
    /// steps of the pair's default that a side's shadow follows show its
    /// filters that answer too. The prefill's shadow reads the answer of
    /// its run there. The decode's, which makes no run of its own beside
    /// a pair the default made, asks for it there.
    ///
    /// The decode's shadow is put to the binding's all-vetoed rule by the
    /// binding's word, as its live side is ([`Set::ruled`]): where a veto
    /// leaves the live prefill's partners nobody, the rule steers the
    /// shadow or sheds it as it does that side. The prefill's shadow is
    /// put to no all-vetoed rule, as its live side is put to none.
    ///
    /// A side's shadow follows the pair's default as its live side does
    /// ([`vetoed::paired`]). It starts where the pair's decision ended, on
    /// the scratch that decision left. A prefill's shadow is shown the
    /// classes its live side was shown last. Where the live side shed among
    /// first partners and ran again over every open class, the shadow's
    /// one run reads every open class.
    ///
    /// A prefill's shadow that sheds goes the live decision's way
    /// ([`Self::decide_pair`]), in this order:
    ///
    /// - Among first partners, the admission limit's test of the prefills
    ///   shown. Where one of them waits on its limit, the shed is the
    ///   limit's and names the prefill side. The test reads the facts the
    ///   shadow's run showed its filters.
    /// - Among first partners, its run over every open class.
    /// - The default's first step, which answers its prefill
    ///   ([`vetoed::shadow_prefill`]). The live decode profile stands on
    ///   the other side.
    ///
    /// Where the default answers nothing, the run's own refusal stands: a
    /// shadow is compared by its outcome.
    ///
    /// The default asks the live decode profile's hard filters behind the
    /// decision they stood in. It is asked only where each of them reads a
    /// row alone and holds no decision scope ([`Half::reads_rows_alone`]).
    /// Another filter may answer otherwise there, and the side says `None`.
    ///
    /// The mark on the prefill's buffer says a pair the default made.
    /// Beside such a pair the decode's shadow is the default's second step
    /// for the live pair's prefill ([`vetoed::paired_decode`]). It reads
    /// the decodes its own profile's hard filters keep in the pass that
    /// paired. Where they keep no partner of that prefill, it sheds under
    /// `partners`, as a decode side with no partner does.
    ///
    /// The prefill's shadow leaves the classes and the mark as it found
    /// them, where it panics too. The decode's shadow runs behind it on
    /// one scratch.
    ///
    /// A pair that a weighing prefill profile's lifted run placed
    /// ([`Self::weighing_lift`]) is a pair of the profiles. Each shadow
    /// runs beside it as beside any other. A prefill's shadow that does
    /// not weigh asks the default there.
    ///
    /// `None` where the binding names no shadow for the side, where the
    /// build did not find that profile free to run beside the live one
    /// ([`side_shadows`]), and where it would be handed another column
    /// than the live side's pool carries ([`Self::side_apart`]).
    ///
    /// `None` as well beside a side its binding's sessions rule or its
    /// picker's assignment placed ([`Self::places_side`]): a shadow's run
    /// is put to neither, and its bare profile set beside a key's pin
    /// would say nothing true of the profile. A request with no key under
    /// a rule is its profiles', and its sides' shadows run.
    ///
    /// `None` too for a prefill's shadow whose default may not be asked,
    /// as said above, and only where the default would be asked. Its own
    /// pick stands, and so does its limit's shed.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair's entry and its inputs, beside the side and the live pick"
    )]
    #[cfg_attr(
        not(any(feature = "in-process", test)),
        expect(
            dead_code,
            reason = "the one caller is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn shadow_side<'l>(
        &self,
        at: u32,
        side: PairSide,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        live: &Pick,
        scratch: &mut Scratch,
        lends: &dyn Fn(u16) -> Option<(&'l dyn LentRing, &'l [u64])>,
    ) -> Option<Decision> {
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        let entry = self.bindings.get(at as usize)?;
        let lives = live_sides(&entry.binding);
        // A side its binding's sessions rule or its picker's assignment
        // placed: a shadow's run is put to neither.
        if self.places_side(entry, lives[side.index()], side, facts) {
            return None;
        }
        // The place stated on a lent ring is the two live sides' where
        // one list is both sides', and the side's own live profile's
        // where each side is stated its own.
        let unlent = self.unlent(index, held_to(&lives, side, facts), facts);
        let facts = unlent.as_ref().unwrap_or(facts);
        let (prefills, prefill_facts) = pools.side(PairSide::Prefill, facts)?;
        let (decodes, decode_facts) = pools.side(PairSide::Decode, facts)?;
        // The shadow's own place, asked behind the filter half, or ahead
        // of it where a filter of the profile declares the ring need. It
        // is asked once: a run of the side that follows reads the answer.
        let asked = OnceCell::new();
        let lent = || *asked.get_or_init(|| self.side_ring_profile(at, side).and_then(lends));
        let ahead = self.marks[usize::from(index)].ring_filter();
        // What the hard filters asked outside a run of the side are shown:
        // the lender's answer where a filter of the profile declares the
        // ring need, as in that run.
        let ahead_lent = || if ahead { lent() } else { None };
        // The mark the pair's decision left on the prefill's buffer: the
        // tier the pair's default paired in, and none beside a pair the
        // profiles made ([`Self::decide_pair`]).
        let defaulted = scratch.single.lifted;
        let steers = entry.vetoed == AllVetoed::Steer;
        Some(match side {
            PairSide::Prefill => {
                // The prefills the live prefill profile was shown: the
                // classes with a partner stand in the scratch as the
                // pair's decision left them ([`Self::decide_pair`]). The
                // shadow's own runs ask no stage of a live profile again.
                // Its decision's scope closed with it, and a stage asked
                // behind it answers by its fallback. The default below
                // asks the live decode profile's hard filters, and only
                // where each reads a row alone. The shadow's run is lifted
                // by its own profile's word, as the live run is by the
                // live profile's ([`Self::weighing_lift`]).
                let lifted_run = self.weighing_lift(index, entry.vetoed, cx);
                let run_cx = lifted_run.as_ref().unwrap_or(cx);
                // A refusal the pair's decision goes on from: a shed that is
                // no picker's decline.
                let refused = |run: &Decision| match run {
                    Decision::Shed(shed) => shed.reason != ShedReason::PolicyDeclined,
                    Decision::Pick(_) | Decision::Queue(_) => false,
                };
                // The shadow's own run over the prefills of `classes`. It asks
                // for its place behind its filter half, or ahead of it
                // ([`run_side_lent`]).
                let own = |classes: &[u32], buf: &mut Buffers| {
                    let set = Set::side(prefills, None, Some(classes));
                    run_side_lent(
                        profile,
                        index,
                        &prefill_facts,
                        pool,
                        set,
                        run_cx,
                        buf,
                        None,
                        ahead,
                        &lent,
                    )
                };
                let mut run = own(&scratch.classes, &mut scratch.single);
                // A run that sheds goes the live decision's way. Among
                // first partners, one of the prefills shown that stands
                // but for its admission limit keeps the request under the
                // limit's stage. Else the side runs again over every open
                // class, which stand in `firsts` behind that decision's
                // swap. The limit's test asks the hard filters on the facts
                // the run showed them: with what the lender answered that
                // run where one of them declares the ring need. The answer
                // stands, and the lender is asked nothing more.
                let firsts_shed = scratch.among_first && refused(&run);
                let waits = firsts_shed && {
                    let stated = with_lent(&prefill_facts, ahead_lent());
                    let shown = stated.as_ref().unwrap_or(&prefill_facts);
                    let takes = Half::new(profile, index, shown, prefills, run_cx);
                    takes.waits_among(&scratch.classes)
                };
                if waits {
                    // The limit's shed names the prefill side, as the live
                    // decision's does.
                    let limit = HasCapacity.name();
                    run = shed(reason_of(limit), limit, *run.trace()).of_side(PairSide::Prefill);
                } else if firsts_shed {
                    run = own(&scratch.firsts, &mut scratch.single);
                }
                // A run that still sheds goes to the pair's default for
                // its prefill, the live decode profile on the other side.
                // The default reads every open class and the constraints
                // as they are stated. The decode side's facts are this
                // run's, with no ring where the shadow is lent none: that
                // step asks the decode profile's hard filters alone, and
                // none of the crate's own reads a ring.
                if !waits && refused(&run) {
                    let live_decode = lives[PairSide::Decode.index()];
                    let live_decode_profile = self.profiles.get(usize::from(live_decode))?;
                    // The shadow's own hard filters are asked there on the
                    // facts its run showed them, as the limit's test asks
                    // them.
                    let stated = with_lent(&prefill_facts, ahead_lent());
                    let shown = stated.as_ref().unwrap_or(&prefill_facts);
                    let sides = [
                        Half::new(profile, index, shown, prefills, cx),
                        Half::new(
                            live_decode_profile,
                            live_decode,
                            &decode_facts,
                            decodes,
                            decode_cx,
                        ),
                    ];
                    // Those filters stood in the pair's decision, which is
                    // over. One that reads more than a row, or that holds
                    // a decision scope, may answer otherwise behind it:
                    // the side says none.
                    if !sides[1].reads_rows_alone() {
                        return None;
                    }
                    // The step leaves the classes and the mark as it found
                    // them: the decode's shadow reads both behind this one.
                    let words = (steers, self.weighs(index));
                    let table = &pools.table;
                    let taken =
                        vetoed::shadow_prefill(sides, table, words, pool, scratch, defaulted);
                    run = taken.unwrap_or(run);
                }
                run
            }
            PairSide::Decode => {
                let prefill = live.targets.first()?;
                let class = prefills.get(prefill.position as usize)?.pair_class;
                if defaulted == LiftedVeto::None {
                    // The partners of the live pair's prefill, put to the
                    // binding's all-vetoed rule as the live decode side's
                    // are: its first partners where the pair's decision
                    // read those, by the word it left
                    // ([`Scratch::among_first`]).
                    let firsts = scratch.among_first;
                    let partners = Set::side(decodes, Some(pools.partners(&class, firsts)), None);
                    run_side_lent(
                        profile,
                        index,
                        &decode_facts,
                        pool,
                        partners.ruled(entry.vetoed),
                        decode_cx,
                        &mut scratch.decode,
                        None,
                        ahead,
                        &lent,
                    )
                } else {
                    // The pair's default made the live pair: its decode is
                    // the live prefill's usable partner of the fewest
                    // requests, in the pass of the tier the mark names.
                    // The shadow's is that step's answer over the decodes
                    // its own profile's hard filters keep in that pass,
                    // every partner alike. Where one of them declares the
                    // ring need, they read what the shadow's lender
                    // answers, as the filters of its own run do. The
                    // lender is asked here: no run of the shadow's stands
                    // beside such a pair.
                    let stated = with_lent(&decode_facts, ahead_lent());
                    let shown = stated.as_ref().unwrap_or(&decode_facts);
                    let half = Half::new(profile, index, shown, decodes, decode_cx);
                    half.usable(defaulted, steers, &mut scratch.decode);
                    let (table, buf) = (&pools.table, &mut scratch.decode);
                    match vetoed::paired_decode::<true>(&half, table, class, pool, buf) {
                        // The shadow's filters kept no partner of the live
                        // prefill in that pass, and the step's pick names
                        // nobody. The shadow sheds as a decode side with
                        // no partner does, and no veto was lifted for it.
                        Decision::Pick(pick) if pick.targets.is_empty() => {
                            let trace = Trace {
                                lifted: LiftedVeto::None,
                                ..pick.trace
                            };
                            shed(ShedReason::NoEndpoints, PARTNERS, trace).of_side(PairSide::Decode)
                        }
                        step => step,
                    }
                }
            }
        })
    }

    /// The decode side of a decode decided after its prefill
    /// ([`Self::decide_decode`]), made once more with the side's shadow
    /// profile over the rows that call decided over: the decodes that
    /// pair with the held `prefill`, on the call's own draw sequence. They
    /// are put to the binding's all-vetoed rule as that call's are, its
    /// liveness tier included ([`Set::ruled`]).
    /// `at` is the pair binding's position ([`Self::pair_position`]). The
    /// run writes the scratch and the shadow profile's stages, and
    /// nothing a commit reads. The shadow is lent the ring its request
    /// lends by the rule of a side's at the pair's decision
    /// ([`Self::shadow_side`]): held against both live sides, or against
    /// the decode's own alone where each side is stated its own places
    /// ([`Self::unlent`]). And it is asked its own place of `lends` as a
    /// side's is there: behind the filter half of its run, or ahead of it
    /// where a filter of the profile declares the ring need.
    ///
    /// `first_decode` is the decode the pair's first decision named, as
    /// that call is handed it. A shadow profile that keeps such a decode
    /// ([`keeps_its_first`]) answers it while its own filters keep it, and
    /// its picker takes no turn there, as the live profile's takes none: a
    /// rotation's shadow stays a turn a request beside its live side. The
    /// test is the shadow profile's own: one that keeps no such decode
    /// decides afresh, whatever its live side keeps.
    ///
    /// `None` where the binding names no shadow for the decode side that
    /// is free to run ([`side_shadows`]), where that shadow would be handed
    /// another column ([`Self::side_apart`]), and where the held prefill
    /// left the rows. The same beside a decode its sessions rule or its
    /// picker's assignment placed ([`Self::places_side`]).
    ///
    /// The run follows that call over the same `scratch`: the call left
    /// there whether it read the held prefill's first partners
    /// ([`Scratch::among_first`]), and the shadow reads the same partners
    /// by that word.
    #[expect(
        clippy::too_many_arguments,
        reason = "the decode entry and its inputs, beside the binding"
    )]
    #[cfg_attr(
        not(any(feature = "in-process", test)),
        expect(
            dead_code,
            reason = "the one caller is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn shadow_decode_later<'l>(
        &self,
        at: u32,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        prefill: WorkerId,
        first_decode: Option<(WorkerId, u16)>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
        lends: &dyn Fn(u16) -> Option<(&'l dyn LentRing, &'l [u64])>,
    ) -> Option<Decision> {
        let side = PairSide::Decode;
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) {
            return None;
        }
        // A decode its binding's sessions rule or its picker's assignment
        // placed, as at the pair's decision.
        let entry = self.bindings.get(at as usize)?;
        let lives = live_sides(&entry.binding);
        if self.places_side(entry, lives[side.index()], side, facts) {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        // The place stated on a lent ring, by the rule of a side's at the
        // pair's first decision.
        let unlent = self.unlent(index, held_to(&lives, side, facts), facts);
        let facts = unlent.as_ref().unwrap_or(facts);
        let (decodes, mut decode_facts) = pools.side(side, facts)?;
        let prefill = pools.prefills.iter().find(|row| row.id == prefill)?;
        // The call's own draw sequence, as its live decode's.
        decode_facts.seed = facts.seed;
        // The partners the live call read: the held prefill's first
        // partners where it read those, by the word it left.
        let class = prefill.pair_class;
        let firsts = scratch.among_first;
        let partners = Set::side(decodes, Some(pools.partners(&class, firsts)), None);
        Some(run_side_lent(
            profile,
            index,
            &decode_facts,
            pool,
            partners.ruled(entry.vetoed),
            decode_cx,
            &mut scratch.decode,
            first_decode.filter(|_| keeps_its_first(profile)),
            self.marks[usize::from(index)].ring_filter(),
            &|| self.side_ring_profile(at, side).and_then(lends),
        ))
    }

    /// Decides the decode side for a held prefill. The model's pair binding
    /// runs its decode profile over that prefill's partners in `pools`.
    /// Each picked target names the held prefill and one of those decodes.
    /// A missing prefill is shed under `prefill-gone`, whatever the call's
    /// constraints state: it is looked for before they are read. A shed
    /// made once the decode profile resolves names that profile. `pool`
    /// holds the facts of the decode pool, where the profile places. The
    /// decodes are held to the places `decode_cx` states. The held
    /// prefill is found by its id, in or outside any places. Partners that
    /// are all over their thresholds, or doubted, are put to the binding's
    /// all-vetoed rule, both its tiers: the rule picks afresh among them,
    /// whatever decode the first decision named. Over partners all over
    /// their thresholds the profile's soft filter keeps that pick to the
    /// ones the request has not tried while one is left, as the rule's
    /// pick in a first decision does. A pair's first decision lifts the
    /// overload veto alone by the rule, and a pool it still refuses goes to
    /// the pair's default, which asks no soft filter
    /// ([`Self::decide_pair`]). Here the prefill has run, and a refusal
    /// fails its request.
    ///
    /// Where the host states first partners ([`PairPools::first`]), the
    /// partners are the held prefill's first partners while one of them is
    /// open at this call's instant, by the test a first decision makes, and
    /// every partner of the prefill else.
    ///
    /// `first_decode` is the worker and the rank of the decode the pair's
    /// first decision named, where the caller holds them. Under a decode
    /// profile that holds no scorer and whose picker takes turns, the call
    /// answers that decode first while the profile's filters keep it, the
    /// decodes they keep behind it as its alternates, and the picker is not
    /// asked: a second pick would turn a rotation a second time for one
    /// request, and move the request for no reason. A scorer moves no pick
    /// of a picker that takes turns and reads no score
    /// ([`Picker::picks_alone`]): a profile that holds one beside such a
    /// picker keeps that decode as well. Its scorers read the decodes the
    /// filters keep, as ahead of a pick, and the trace holds their scores.
    /// Where a filter refuses that decode, or none of the partners is that
    /// worker, the picker picks among the decodes the filters keep. So it
    /// does under every other profile, and for a caller that hands none.
    ///
    /// A number `decode_cx` states ([`Constraints::target_worker`]) is
    /// counted among those partners at the decode side's places, as a
    /// pair's first decision counts it ([`Self::decide_pair`]): the call
    /// answers the decode it reaches, is shed under `target-worker` where
    /// that decode cannot be named, and is declined under that name where
    /// the number stands beside no places.
    ///
    /// Under a sessions rule the call may claim its decode for the decode
    /// side's key, at this call's `now_micros`. A host that asks this entry
    /// itself tells the side it issues under a lease that states that
    /// instant, or none ([`Lease::issued_micros`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "the request and the pair's pools, beside the pair the call is held by"
    )]
    pub fn decide_decode(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        prefill: WorkerId,
        first_decode: Option<(WorkerId, u16)>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        if carries_a_column(facts) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        let Some((decodes, mut decode_facts)) = pools.side(PairSide::Decode, facts) else {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        };
        let Some(entry) = self.pair_entry(facts.model) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        let binding = &entry.binding;
        let index = binding.decode_profile.unwrap_or(binding.profile);
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        // The held prefill is looked for before any constraint is read: a
        // call whose prefill left the lists says so, whatever it states.
        let Some(prefill) = pools.prefills.iter().find(|row| row.id == prefill) else {
            // The decode profile is resolved, so the shed names it.
            let under = Trace {
                profile: index,
                ..bare
            };
            // The held prefill is the side left with nobody.
            return shed(ShedReason::NoEndpoints, PREFILL_GONE, under).of_side(PairSide::Prefill);
        };
        let shown = self.marks[usize::from(index)].scoped();
        if shown && !decode_cx.stalled.is_empty() {
            return rows_not_written(bare, index);
        }
        if counts_no_place(decode_cx) {
            return number_not_counted(bare, index);
        }
        // This call has one draw sequence, seeded by its own request.
        decode_facts.seed = facts.seed;
        let _opened = shown.then(|| profile.open(&decode_facts, decodes, pool));
        debug_assert!(
            pools.first.within(&pools.table).is_ok(),
            "the first partners do not stand within the pair table: {:?}",
            pools.first.within(&pools.table)
        );
        // The held prefill's partners are put to the binding's all-vetoed
        // rule, its liveness tier included. They are its first partners
        // while one of those is open at this instant, by the test a pair's
        // first decision makes ([`FirstPartners`]): one the profile's hard
        // filters keep, or, where they keep no partner of the prefill, one
        // they keep with the overload veto lifted as the rule lifts it.
        // Else every partner.
        let class = prefill.pair_class;
        let first = pools.first.of(class);
        let firsts = !first.is_empty() && {
            let hard = || profile.filters.iter().filter(|f| !f.soft());
            let open = |among: &[u32], facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
                let kept = |d: &Candidate| hard().all(|f| f.keep(facts, d, cx));
                places::rows(decodes, decode_cx).any(|d| classed(among, d.pair_class) && kept(d))
            };
            // The facts and the constraints of the rule's own pass
            // ([`vetoed::emptied`]).
            let all = RequestFacts {
                tried: &[],
                ..decode_facts
            };
            let pass = Constraints {
                overload_lifted: true,
                ..*decode_cx
            };
            open(first, &decode_facts, decode_cx)
                || (!open(pools.table.of(&class), &decode_facts, decode_cx)
                    && open(first, &all, &pass))
        };
        // Left for the decode's shadow, which reads the same partners
        // ([`Self::shadow_decode_later`]).
        scratch.among_first = firsts;
        let partners = pools.partners(&class, firsts);
        let partners = Set::side(decodes, Some(partners), None).ruled(entry.vetoed);
        let mut decision = if self.places_side(entry, index, PairSide::Decode, &decode_facts) {
            // The rule's run reads no `first_decode`: a session's first
            // decision claimed its decode in the session table, and the
            // key's pin answers this call.
            self.placed_side::<true>(
                binding,
                index,
                profile,
                &decode_facts,
                pool,
                partners,
                decode_cx,
                scratch,
            )
        } else {
            run_side(
                profile,
                index,
                &decode_facts,
                pool,
                partners,
                decode_cx,
                &mut scratch.decode,
                first_decode.filter(|_| keeps_its_first(profile)),
            )
        };
        // Each target names the held prefill with one decode, as the
        // targets of a pair's first decision do. They are turned where the
        // decision stands, and its trace stays the decode's: taken out and
        // built beside it, the pick was copied twice. The decode side's
        // verdict goes into the first target's pair and its word of who
        // placed it into every pair, and the pick states neither of a
        // prefill: this call decided none. A host that holds the pair's
        // first pick keeps that pick's word of the prefill side beside
        // this one.
        if let Decision::Pick(pick) = &mut decision {
            let decode = std::mem::take(&mut pick.session_verdict);
            pick.targets
                .pair_under(&target_of(prefill), decode, Placing::Stages);
            return decision;
        }
        // The decode's run named no worker: the refusal is the decode side's.
        if decode_cx.target_worker.is_some() {
            decision = target_missed(decision);
        }
        decision.of_side(PairSide::Decode)
    }
}
