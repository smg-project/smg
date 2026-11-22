//! A request of a side's phase that is no side of a pair.
//!
//! A document binds a model and a phase to a plan. A pair plan decides some
//! phases together, at the pair's prefill ([`Plan::pairs`]). A request of
//! any other phase is decided by itself, whatever its phase says: one
//! profile over the one list its host hands. An engine over stored records
//! decides such a request the same way, over the pool of the request's
//! phase.
//!
//! So a host that hands ONE list asks for one worker of it, and the engine
//! answers where the document agrees. A prefill under the plan of one
//! profile is decided by itself. A prefill under a pair plan is its pair's
//! first side: over one list it has no decode to pair with, and it is
//! decided over no worker.
//!
//! [`Plan::pairs`]: crate::engine::framework::Plan::pairs

use super::PolicyEngine;
use crate::engine::view::Phase;

impl PolicyEngine {
    /// Whether a made request of `phase` for `model` is decided by itself
    /// over the one list its host hands: a single request, and a request
    /// of a phase its binding's plan pairs no side of. A request that no
    /// binding matches is one too: the whole decision declines it by name.
    #[inline]
    pub(super) fn lone(&self, model: &str, phase: Phase) -> bool {
        phase == Phase::Single || !self.pairs(model, phase)
    }

    /// Whether the plan `model` is bound to for `phase` decides that phase
    /// as a side of a pair. Out of line: a single request never asks.
    #[cold]
    fn pairs(&self, model: &str, phase: Phase) -> bool {
        let binding = self.spec().engine().binding(model, phase);
        binding.is_some_and(|binding| binding.plan.pairs(phase))
    }
}

#[cfg(test)]
mod tests {
    //! The classic router takes one worker of the prefill pool for a count
    //! of tokens, by its single selection
    //! (`model_gateway/src/routers/http/pd_router.rs:3169-3188`,
    //! `model_gateway/src/routers/common/placement.rs:375`). The engine over
    //! stored records decides a request of a phase its plan pairs no side
    //! of as a single over that phase's pool
    //! (`engine/store_backed_engine.rs`, `entry_and_plan`, with its case
    //! `a_lone_prefill_is_held_to_the_input_and_a_single_to_both_bounds`).

    use std::{cell::Cell, sync::Arc, time::Instant};

    use smg_worker::identity::basic::Worker;

    use super::super::{
        pair::{PairAdmission, PairLists},
        single::tests::{pool, request, Asks},
        PoolMembers,
    };
    use crate::{
        document::spec::{AppliedSpec, PolicySpec, ShadowSpec, PRESETS},
        engine::{
            decision::{Decision, ShadowKind, ShedReason, Target},
            decision_engine::{
                DecideOutcome, DecisionEngine, DecisionRefused, HeldDecision, HeldDecode, Request,
                RequestMaker, WorkerLists,
            },
            demands::{DemandFacts, Demands},
            pair_classes::PartnerClasses,
            policy_engine::PolicyEngine,
            view::{FixedFacts, Phase},
        },
        store::GatewayId,
    };

    fn engine_of(document: &PolicySpec) -> PolicyEngine {
        let spec = AppliedSpec::validate(document);
        PolicyEngine::new(spec.expect("the document validates"))
    }

    /// A first attempt of `phase` that states nothing but its seed.
    fn stating<'a>(phase: Phase, seed: u64) -> Request<'a> {
        let mut stated = request();
        stated.facts.phase = phase;
        stated.facts.seed = seed;
        stated
    }

    /// What an entry answers, in the terms two entries are compared by: the
    /// targets' places with their decodes', a shed's reason and stage, or
    /// the refusal.
    #[derive(Debug, PartialEq)]
    enum Answer {
        Picked(Vec<(u32, Option<u32>)>),
        Shed(ShedReason, &'static str),
        Refused(DecisionRefused),
    }

    fn answer(decided: Result<Decision, DecisionRefused>) -> Answer {
        match decided {
            Ok(Decision::Shed(shed)) => Answer::Shed(shed.reason, shed.emptied_by),
            Ok(decision) => {
                let pick = decision.pick().expect("a pick");
                let place = |target: &Target| {
                    let decode = target.pair.map(|pair| pair.decode_position);
                    (target.position, decode)
                };
                Answer::Picked(pick.targets.iter().map(place).collect())
            }
            Err(refused) => Answer::Refused(refused),
        }
    }

    /// The whole decision the leasing entry answers, with its lease.
    fn leased(
        engine: &PolicyEngine,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let decided = engine.decide_under(GatewayId::LOCAL, maker, now)?;
        Ok(decided.whole().expect("no place").0)
    }

    /// A host with room everywhere: every prefill has capacity and every
    /// decode a room.
    #[derive(Debug)]
    struct Open;

    impl PairAdmission for Open {
        fn has_capacity(&self, _: usize) -> bool {
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    /// A request stated ahead with the lists its host hands for it, and how
    /// many times the engine made it.
    struct Hands<'a> {
        request: Request<'a>,
        lists: WorkerLists<'a>,
        made: Cell<u32>,
    }

    impl<'a> Hands<'a> {
        fn of(request: Request<'a>, lists: WorkerLists<'a>) -> Self {
            Self {
                request,
                lists,
                made: Cell::new(0),
            }
        }
    }

    impl RequestMaker for Hands<'_> {
        fn model(&self) -> &str {
            self.request.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.made.set(self.made.get() + 1);
            self.request
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.lists
        }
    }

    /// An engine of the stock pair document with two prefills and two
    /// decodes of its own, every prefill paired with every decode.
    struct Paired {
        engine: PolicyEngine,
        members: [PoolMembers; 2],
        classes: PartnerClasses,
    }

    impl Paired {
        fn new() -> Self {
            let document = PolicySpec::pair_presets("round_robin", "round_robin");
            let engine = engine_of(&document.expect("the pair presets"));
            let all = pool(4);
            let side = |from: usize| -> Arc<[Arc<dyn Worker>]> {
                all[from..from + 2].iter().map(Arc::clone).collect()
            };
            let members = [engine.pool_members(&side(0)), engine.pool_members(&side(2))];
            let both = [0, 1];
            let classes = PartnerClasses::number(2, [both.as_slice(), both.as_slice()]);
            Self {
                engine,
                members,
                classes,
            }
        }

        /// The lists a host hands: none, the prefills alone, or the pair's
        /// two.
        fn lists(&self, count: usize) -> WorkerLists<'_> {
            let [prefill, decode] = &self.members;
            match count {
                0 => WorkerLists::None,
                1 => WorkerLists::Single(prefill),
                _ => WorkerLists::Pair(PairLists {
                    prefill,
                    decode,
                    classes: &self.classes,
                    admission: &Open,
                    prefill_places: None,
                    decode_places: None,
                }),
            }
        }
    }

    /// Under the plan of one profile a request of a side's phase is
    /// decided by itself over the one list its host hands, by both kinds
    /// of entry: of two engines made alike, each names the same workers
    /// for the same request, under every stock policy and each side's
    /// phase.
    #[test]
    fn a_phase_its_plan_pairs_no_side_of_is_decided_by_itself_at_both_entries() {
        for policy in PRESETS {
            let document = PolicySpec::preset(policy).expect("a built-in preset");
            for phase in [Phase::Prefill, Phase::Decode, Phase::Encode] {
                let (leasing, bare) = (engine_of(&document), engine_of(&document));
                let list = pool(8);
                let (theirs, ours) = (leasing.pool_members(&list), bare.pool_members(&list));
                for seed in 0..8 {
                    let now = Instant::now();
                    let asks = Asks::of(stating(phase, seed), Some(&theirs));
                    let named =
                        answer(leased(&leasing, &asks, now).map(|outcome| outcome.decision));
                    assert!(
                        matches!(&named, Answer::Picked(targets) if !targets.is_empty()),
                        "{policy}, {phase:?}, seed {seed}: {named:?}"
                    );
                    let asks = Asks::of(stating(phase, seed), Some(&ours));
                    let alone = bare.decide_without_lease(&asks, now);
                    let alone = answer(alone.map(|(decision, _)| decision));
                    assert_eq!(alone, named, "{policy}, {phase:?}, seed {seed}");
                }
            }
        }
    }

    /// Both kinds of entry answer every phase over every shape of lists
    /// alike, under a pair plan: of two engines made alike, the same pick,
    /// the same shed under the same stage, or the same refusal, and each
    /// makes its request once. A side of the pair is never decided by
    /// itself: a prefill over one list has no decode to pair with and is
    /// decided over no worker, and a decode sent on its own is refused. A
    /// single and an encode request are sides of no pair under this plan:
    /// over one list each is decided by itself, and the binding, which
    /// names no profile for it, declines it. Over no list each is decided
    /// over no worker, and an encode request that hands two is refused.
    ///
    /// A request for a model no binding matches is a side of no pair
    /// either: over one list it is decided by itself and declined by name,
    /// whatever its phase.
    #[test]
    fn both_entries_answer_every_phase_over_every_shape_of_lists_alike() {
        let (leasing, bare) = (Paired::new(), Paired::new());
        let none = Answer::Shed(ShedReason::NoEndpoints, "none");
        let declined = Answer::Shed(ShedReason::PolicyDeclined, "binding");
        let refused = Answer::Refused(DecisionRefused::PairSide);
        let now = Instant::now();
        for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
            for lists in 0..3 {
                let asks = Hands::of(stating(phase, 3), bare.lists(lists));
                let alone = bare.engine.decide_without_lease(&asks, now);
                let alone = answer(alone.map(|(decision, _)| decision));
                assert_eq!(asks.made.get(), 1, "{phase:?}, {lists}: made once");
                let asks = Hands::of(stating(phase, 3), leasing.lists(lists));
                let theirs = leased(&leasing.engine, &asks, now);
                let theirs = answer(theirs.map(|outcome| outcome.decision));
                assert_eq!(asks.made.get(), 1, "{phase:?}, {lists}: made once");
                assert_eq!(alone, theirs, "{phase:?} over {lists} lists");
                let wanted = match (phase, lists) {
                    (Phase::Prefill, 2) => {
                        assert!(matches!(alone, Answer::Picked(_)), "{alone:?}");
                        continue;
                    }
                    (Phase::Single | Phase::Encode, 1) => &declined,
                    (Phase::Single | Phase::Prefill, _) | (Phase::Encode, 0) => &none,
                    (Phase::Decode | Phase::Encode, _) => &refused,
                };
                assert_eq!(&alone, wanted, "{phase:?} over {lists} lists");
            }
        }
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        document.bindings[0].matcher.model = "another-*".into();
        let engine = engine_of(&document);
        let list = pool(2);
        let members = engine.pool_members(&list);
        for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
            let asks = Asks::of(stating(phase, 3), Some(&members));
            let alone = engine.decide_without_lease(&asks, now);
            let alone = answer(alone.map(|(decision, _)| decision));
            let theirs = answer(leased(&engine, &asks, now).map(|outcome| outcome.decision));
            assert_eq!((&alone, &theirs), (&declined, &declined), "{phase:?}");
        }
    }

    /// A prefill decided by itself is held to the input alone: its
    /// completion's bound and the way its constraint was decided are a
    /// decode's to hold, and a host states both bounds whatever the phase.
    /// A single is held to both. Both kinds of entry answer alike, as the
    /// engine over stored records answers
    /// (`a_lone_prefill_is_held_to_the_input_and_a_single_to_both_bounds`).
    #[test]
    fn a_prefill_decided_by_itself_is_held_to_the_input_alone() {
        let document = PolicySpec::preset("random").expect("a built-in preset");
        let engine = engine_of(&document);
        let list = pool(1);
        let states = DemandFacts {
            window: 8,
            decoding: 1,
            ..DemandFacts::default()
        };
        let members = engine.pool_members_with_fixed_facts(&list, |_| FixedFacts {
            demands: states,
            ..FixedFacts::default()
        });
        let window = |input_window, output_window| Demands {
            input_window,
            output_window,
            ..Demands::default()
        };
        let decoded = Demands {
            decoding: 2,
            ..Demands::default()
        };
        let met = Answer::Picked(vec![(0, None)]);
        let now = Instant::now();
        for (phase, demands, meets) in [
            (Phase::Prefill, window(4, 64), true),
            (Phase::Prefill, window(16, 64), false),
            (Phase::Prefill, decoded, true),
            (Phase::Single, window(4, 64), false),
            (Phase::Single, window(4, 8), true),
            (Phase::Single, decoded, false),
        ] {
            let mut stated = stating(phase, 7);
            stated.constraints.demands = demands;
            let asks = Asks::of(stated, Some(&members));
            let alone = engine.decide_without_lease(&asks, now);
            let alone = answer(alone.map(|(decision, _)| decision));
            let theirs = answer(leased(&engine, &asks, now).map(|outcome| outcome.decision));
            assert_eq!(alone, theirs, "{phase:?} {demands:?}");
            assert_eq!(alone == met, meets, "{phase:?} {demands:?}: {alone:?}");
            if !meets {
                assert!(
                    matches!(alone, Answer::Shed(ShedReason::NoEndpoints, _)),
                    "{phase:?} {demands:?}: {alone:?}"
                );
            }
        }
    }

    /// The decode decided again is answered alike by both kinds of entry
    /// over every shape of lists: over no pair's lists no worker, and over
    /// the pair's lists the held prefill with the decode the first decision
    /// named, which a rotation would not name again by itself.
    #[test]
    fn both_decode_entries_answer_every_shape_of_lists_alike() {
        let (leasing, bare) = (Paired::new(), Paired::new());
        let now = Instant::now();
        let later = || {
            let mut stated = stating(Phase::Prefill, 5);
            stated.constraints.decode_later = true;
            stated
        };
        let first = bare
            .engine
            .decide_without_lease(&Hands::of(later(), bare.lists(2)), now);
        let (first, _) = first.expect("a decision");
        let held = *first
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target");
        let outcome = leased(&leasing.engine, &Hands::of(later(), leasing.lists(2)), now);
        let outcome = outcome.expect("a decision");
        let lease = outcome.lease.expect("the pair's lease").lease;
        let view = HeldDecision {
            decision: &outcome.decision,
            lease: &lease,
            index: 0,
            target: &held,
            decode: HeldDecode {
                session: outcome.decode_session,
                tokens: lease.tokens,
                chosen: true,
            },
        };
        let none = Answer::Shed(ShedReason::NoEndpoints, "none");
        for lists in 0..3 {
            let asks = Hands::of(later(), bare.lists(lists));
            let again = bare.engine.decide_decode_without_lease(&asks, &held, now);
            let again = answer(again);
            assert_eq!(asks.made.get(), 1, "{lists}: made once");
            let asks = Hands::of(later(), leasing.lists(lists));
            let issued = leasing
                .engine
                .decide_decode_under(GatewayId::LOCAL, &asks, &view, now);
            let issued = answer(issued.map(|outcome| outcome.decision));
            assert_eq!(again, issued, "over {lists} lists");
            if lists < 2 {
                assert_eq!(again, none, "over {lists} lists");
                continue;
            }
            let Answer::Picked(targets) = again else {
                panic!("no pick: {again:?}");
            };
            let kept = held.pair.map(|pair| pair.decode_position);
            assert_eq!(targets.first(), Some(&(held.position, kept)));
        }
    }

    /// A panic under an entry that issues no lease costs its request alone
    /// and is counted, as under the leasing entry: here the maker panics
    /// where its request is made.
    #[test]
    fn a_panic_under_an_entry_that_issues_no_lease_is_caught_and_counted() {
        struct Panics;

        impl RequestMaker for Panics {
            fn model(&self) -> &str {
                "model-a"
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                panic!("the case's own panic, where a request is made")
            }
        }

        let paired = Paired::new();
        let now = Instant::now();
        let faulted = Some(DecisionRefused::Faulted);
        assert_eq!(
            paired.engine.decide_without_lease(&Panics, now).err(),
            faulted
        );
        assert_eq!(paired.engine.faults(), 1);
        let held = Target::default();
        let again = paired
            .engine
            .decide_decode_without_lease(&Panics, &held, now);
        assert_eq!(again.err(), faulted);
        assert_eq!(paired.engine.faults(), 2);
    }

    /// A binding matches by phase, and a request decided by itself runs the
    /// shadow of ITS binding. Here the binding of the prefill phase names a
    /// shadow and the binding behind it names none: a prefill's shadow
    /// decides and is counted under the first binding, before a single
    /// request of the same list and after it, and a single's runs none.
    #[test]
    fn a_request_decided_by_itself_runs_the_shadow_of_its_own_phases_binding() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let twin = document
            .profiles
            .values()
            .next()
            .expect("a profile")
            .clone();
        document.profiles.insert("twin".into(), twin);
        let mut rest = document.bindings[0].clone();
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        document.bindings[0].matcher.phase = vec![Phase::Prefill];
        rest.matcher.phase = Vec::new();
        document.bindings.push(rest);
        let engine = engine_of(&document);
        let list = pool(4);
        let members = engine.pool_members(&list);
        let counted = |binding: u32| -> u64 {
            let kinds = ShadowKind::ALL.map(|kind| engine.shadowed(binding, kind));
            kinds.iter().sum()
        };
        let now = Instant::now();
        let ask = |phase: Phase, seed: u64| {
            let asks = Asks::of(stating(phase, seed), Some(&members));
            let decided = engine.decide_under(GatewayId::LOCAL, &asks, now);
            assert!(decided.is_ok(), "{phase:?}: {decided:?}");
        };
        ask(Phase::Prefill, 1);
        assert_eq!((counted(0), counted(1)), (1, 0), "the prefill's binding");
        ask(Phase::Single, 2);
        assert_eq!((counted(0), counted(1)), (1, 0), "a single names no shadow");
        ask(Phase::Prefill, 3);
        assert_eq!((counted(0), counted(1)), (2, 0), "behind a single too");
        // The entry that issues no lease answers the same binding's word,
        // and counts none.
        let word = |phase: Phase| {
            let asks = Asks::of(stating(phase, 4), Some(&members));
            let decided = engine.decide_without_lease(&asks, now);
            decided.expect("a decision").1.map(|said| said.binding)
        };
        assert_eq!((word(Phase::Prefill), word(Phase::Single)), (Some(0), None));
        assert_eq!((counted(0), counted(1)), (2, 0), "the caller counts it");
    }
}
