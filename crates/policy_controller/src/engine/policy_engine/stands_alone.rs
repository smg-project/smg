//! A request of a side's phase that is no side of a pair.
//!
//! A document binds a model and a phase to a plan. A pair plan decides some
//! phases together, at the pair's prefill ([`Plan::pairs`]). A request of
//! any other phase stands alone, whatever its phase says: one profile
//! decides it over the one list its host hands, by the binding its own
//! phase resolves to.
//!
//! So a host that hands ONE list asks for one worker of it, and the engine
//! answers where the document agrees. A prefill under a plan of one profile
//! stands alone. A prefill under a pair plan is its pair's first side: over
//! one list it has no decode to pair with, and it is decided over no
//! worker. A decode or an encode request that is a pair's side has no
//! decision of its own.
//!
//! [`Plan::pairs`]: crate::engine::framework::Plan::pairs

use super::Loaded;
use crate::engine::view::Phase;

impl Loaded<'_, '_> {
    /// Whether a made request of `model` at `phase` stands alone: a single
    /// request, and a request of a phase that its binding's plan decides
    /// with no pair. One profile decides it over the one list its host
    /// hands. A request that no binding matches stands alone too: over a
    /// list the whole decision declines it by name, and over none it is
    /// decided over no worker, as a single request is.
    #[inline]
    pub(super) fn stands_alone(self, model: &str, phase: Phase) -> bool {
        phase == Phase::Single || !self.pairs(model, phase)
    }

    /// Whether the plan `model` is bound to for `phase`, by that phase's
    /// own lookup, decides the phase as a side of a pair. Out of line: a
    /// single request never asks.
    #[cold]
    fn pairs(self, model: &str, phase: Phase) -> bool {
        self.document.spec.engine().pairs(model, phase)
    }
}

#[cfg(test)]
mod tests {
    //! What the classic router answers where these cases hold a new answer
    //! of the engine. For a count of tokens it takes one worker of the
    //! prefill pool by its single selection over that pool, the model's
    //! policy over the one list and no decode beside it
    //! (`route_messages_count_tokens` in
    //! `model_gateway/src/routers/http/pd_router.rs`, and
    //! `classic_single_of` in
    //! `model_gateway/src/routers/common/placement.rs`). It asks no
    //! decode and no encoder alone: a pair is selected together
    //! (`select_pair` in the same file).

    use std::{cell::Cell, sync::Arc, time::Instant};

    use smg_worker::identity::basic::Worker;

    use super::super::{
        pair::{PairAdmission, PairLists},
        single::tests::{pool, request, Asks},
        PolicyEngine,
    };
    use crate::{
        document::spec::{AppliedSpec, PolicySpec, ShadowSpec, PRESETS},
        engine::{
            decision::{Decision, ShadowKind, ShedReason, Target},
            decision_engine::{
                DecideOutcome, DecisionEngine, DecisionRefused, Request, RequestMaker, WorkerLists,
            },
            demands::{DemandFacts, Demands},
            list_cell::tests::PublishedList,
            pair_classes::PartnerClasses,
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
    #[derive(Clone, Debug, PartialEq)]
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

    /// What both entries of ONE engine answer `maker`'s request at `now`:
    /// the entry that issues no lease first, then the leasing entry. For an
    /// answer that no earlier request moves; a policy that takes turns is
    /// asked through [`alike`].
    fn both(engine: &PolicyEngine, maker: &impl RequestMaker, now: Instant) -> (Answer, Answer) {
        let bare = engine.decide_without_lease(maker, now);
        let bare = answer(bare.map(|(decision, _)| decision));
        let theirs = answer(leased(engine, maker, now).map(|outcome| outcome.decision));
        (bare, theirs)
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
    /// many times the engine made it. It states no plain request.
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

    /// An engine of `document` with two prefills and two decodes of its
    /// own, every prefill paired with every decode.
    struct Paired {
        engine: PolicyEngine,
        sides: [PublishedList; 2],
        classes: PartnerClasses,
    }

    impl Paired {
        fn under(document: &PolicySpec) -> Self {
            let all = pool(4);
            let side = |from: usize| -> Arc<[Arc<dyn Worker>]> {
                all[from..from + 2].iter().map(Arc::clone).collect()
            };
            let both = [0, 1];
            Self {
                engine: engine_of(document),
                sides: [PublishedList::new(side(0)), PublishedList::new(side(2))],
                classes: PartnerClasses::number(2, [both.as_slice(), both.as_slice()]),
            }
        }

        /// The lists a host hands: none, the prefills alone, or the pair's
        /// two.
        fn lists(&self, count: usize) -> WorkerLists<'_> {
            let [prefill, decode] = &self.sides;
            match count {
                0 => WorkerLists::None,
                1 => WorkerLists::Single(prefill.lent()),
                _ => WorkerLists::Pair(PairLists {
                    prefill: prefill.lent(),
                    decode: decode.lent(),
                    classes: &self.classes,
                    admission: &Open,
                    prefill_places: None,
                    decode_places: None,
                    prefill_stalled: &[],
                    prefill_held: &[],
                    decode_stalled: &[],
                    decode_held: &[],
                }),
            }
        }
    }

    /// What two engines made alike answer a request of `phase` over `lists`
    /// lists: the entry that issues no lease on `bare`, the leasing entry on
    /// `leasing`. The two answer alike, and each makes its request once.
    fn alike(bare: &Paired, leasing: &Paired, phase: Phase, lists: usize, now: Instant) -> Answer {
        let asks = Hands::of(stating(phase, 3), bare.lists(lists));
        let alone = bare.engine.decide_without_lease(&asks, now);
        let alone = answer(alone.map(|(decision, _)| decision));
        assert_eq!(asks.made.get(), 1, "{phase:?}, {lists}: made once");
        let asks = Hands::of(stating(phase, 3), leasing.lists(lists));
        let theirs = leased(&leasing.engine, &asks, now);
        let theirs = answer(theirs.map(|outcome| outcome.decision));
        assert_eq!(asks.made.get(), 1, "{phase:?}, {lists}: made once");
        assert_eq!(alone, theirs, "{phase:?} over {lists} lists");
        alone
    }

    /// Under a plan of one profile a request of a side's phase stands
    /// alone: both entries decide it over the one list its host hands. Of
    /// two engines made alike, each names the same workers for the same
    /// request, under every stock policy and each side's phase. The classic
    /// router's own such request is the count of tokens, one worker of the
    /// prefill pool by the model's policy.
    #[test]
    fn a_phase_its_plan_pairs_no_side_of_stands_alone_at_both_entries() {
        for policy in PRESETS {
            let document = PolicySpec::preset(policy).expect("a built-in preset");
            for phase in [Phase::Prefill, Phase::Decode, Phase::Encode] {
                let (leasing, bare) = (engine_of(&document), engine_of(&document));
                let list = pool(8);
                let theirs = PublishedList::new(Arc::clone(&list));
                let ours = PublishedList::new(Arc::clone(&list));
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

    /// Both entries answer every phase alike over no list, over one and
    /// over a pair's two, under a pair plan: the same pick, the same shed
    /// under the same stage, or the same refusal, and each makes its
    /// request once. An encode request's three lists have an entry of
    /// their own, with its own cases. A side
    /// of the pair never stands alone: a prefill over one list has no
    /// decode to pair with and is decided over no worker, and a decode sent
    /// on its own is refused, as the classic router asks no decode alone.
    /// A single and an encode request are sides of no pair under this
    /// plan: over one list each stands alone, and the binding, which names
    /// no profile for it, declines it. Over no list each is decided over no
    /// worker, and an encode request that hands two is refused.
    ///
    /// Under the `epd` plan every phase but the single is a side. Over one
    /// list and over none, an encode request is refused as a decode is, a
    /// prefill is decided over no worker, and a single is as under the
    /// pair plan.
    ///
    /// A request for a model no binding matches is a side of no pair
    /// either: over one list it stands alone and is declined by name,
    /// whatever its phase.
    #[test]
    fn both_entries_answer_every_phase_over_every_shape_of_lists_alike() {
        let none = Answer::Shed(ShedReason::NoEndpoints, "none");
        let declined = Answer::Shed(ShedReason::PolicyDeclined, "binding");
        let refused = Answer::Refused(DecisionRefused::PairSide);
        let now = Instant::now();
        let pair =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        let epd = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin");
        let epd = epd.expect("the presets under the epd plan");
        // The plan's document, how many shapes of lists it is asked over,
        // and whether an encode request is a side of its pair.
        for (document, shapes, encodes) in [(&pair, 3, false), (&epd, 2, true)] {
            let (leasing, bare) = (Paired::under(document), Paired::under(document));
            for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
                for lists in 0..shapes {
                    let alone = alike(&bare, &leasing, phase, lists, now);
                    let wanted = match (phase, lists) {
                        (Phase::Prefill, 2) => {
                            assert!(matches!(alone, Answer::Picked(_)), "{alone:?}");
                            continue;
                        }
                        (Phase::Single, 1) => &declined,
                        (Phase::Encode, 1) if !encodes => &declined,
                        (Phase::Single | Phase::Prefill, _) => &none,
                        (Phase::Encode, 0) if !encodes => &none,
                        (Phase::Decode | Phase::Encode, _) => &refused,
                    };
                    assert_eq!(
                        &alone, wanted,
                        "{phase:?} over {lists} lists, an encode side: {encodes}"
                    );
                }
            }
        }
        let mut document = pair;
        document.bindings[0].matcher.model = "another-*".into();
        let engine = engine_of(&document);
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
            let asks = Hands::of(stating(phase, 3), WorkerLists::Single(members.lent()));
            let answers = both(&engine, &asks, now);
            assert_eq!(answers, (declined.clone(), declined.clone()), "{phase:?}");
        }
    }

    /// The decode phase's own lookup says whether a decode stands alone. A
    /// binding of one profile listed ahead of the pair's for the decode
    /// phase decides a decode asked alone over one list, at both entries.
    /// The prefill lookup still resolves to the pair: a prefill over one
    /// list is its pair's first side, over no worker, and over the pair's
    /// two lists the pair is decided. The classic router has no such
    /// request: it asks no decode alone.
    #[test]
    fn a_binding_of_one_profile_ahead_of_the_pairs_decides_a_decode_alone() {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        let mut ahead = document.bindings[0].clone();
        ahead.matcher.phase = vec![Phase::Decode];
        ahead.plan = None;
        ahead.profiles.clear();
        ahead.profile = Some("decode".into());
        document.bindings.insert(0, ahead);
        let (leasing, bare) = (Paired::under(&document), Paired::under(&document));
        let none = Answer::Shed(ShedReason::NoEndpoints, "none");
        let now = Instant::now();
        let ask = |phase, lists| alike(&bare, &leasing, phase, lists, now);
        let alone = ask(Phase::Decode, 1);
        assert!(
            matches!(&alone, Answer::Picked(targets) if targets.iter().all(|at| at.1.is_none())),
            "one worker of the one list: {alone:?}"
        );
        assert_eq!(ask(Phase::Decode, 0), none, "no list, no worker");
        assert_eq!(ask(Phase::Prefill, 1), none, "its pair's first side");
        let pair = ask(Phase::Prefill, 2);
        assert!(
            matches!(&pair, Answer::Picked(targets) if targets.iter().all(|at| at.1.is_some())),
            "the pair, under the pair binding: {pair:?}"
        );
    }

    /// A prefill that stands alone is held to the input alone: its
    /// completion's bound and the way its constraint was decided are a
    /// decode's to hold, and a host states both bounds whatever the phase.
    /// A single is held to both, and so are a decode and an encode request
    /// that stand alone. Both entries answer alike. The classic router's
    /// count of tokens generates nothing, and its prefill worker is asked
    /// for no completion.
    #[test]
    fn a_prefill_that_stands_alone_is_held_to_the_input_alone() {
        let document = PolicySpec::preset("random").expect("a built-in preset");
        let engine = engine_of(&document);
        let list = pool(1);
        let states = DemandFacts {
            window: 8,
            decoding: 1,
            ..DemandFacts::default()
        };
        let members = PublishedList::with_facts(Arc::clone(&list), |_| FixedFacts {
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
            (Phase::Decode, window(4, 64), false),
            (Phase::Decode, window(4, 8), true),
            (Phase::Encode, window(4, 64), false),
            (Phase::Encode, decoded, false),
        ] {
            let mut stated = stating(phase, 7);
            stated.constraints.demands = demands;
            let asks = Hands::of(stated, WorkerLists::Single(members.lent()));
            let (alone, theirs) = both(&engine, &asks, now);
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

    /// A binding matches by phase, and a request that stands alone runs the
    /// shadow of ITS binding. Here the binding of the prefill phase names a
    /// shadow and the binding behind it names none: a prefill's shadow
    /// decides and is counted under the first binding, before a single
    /// request of the same list and after it, and a single's runs none.
    #[test]
    fn a_request_that_stands_alone_runs_the_shadow_of_its_own_phases_binding() {
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
        let members = PublishedList::new(Arc::clone(&list));
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
