//! A request's budget, as the engine over worker lists shows it to a stage.
//!
//! A host in process states no budget by `0`: the decision of such a
//! request is unbounded ([`Request::facts`](crate::engine::Request::facts)).
//! A stage reads the number as the time left
//! ([`RequestFacts::budget_micros`](crate::engine::view::RequestFacts::budget_micros)),
//! as a controller's frame needs it read, whose `0` says none is left: a
//! plugin's stage passes such a request by and counts a failure of its own.
//! So this engine shows a stage the widest value where the host stated
//! none. It does so where it takes the request its maker makes
//! ([`Loaded::made`](super::Loaded::made)), which every entry
//! that makes one calls: a single's and a pair's, with a lease and without.
//!
//! The two calls a host makes with no engine above its lists turn a `0`
//! the same way, in the host's own request and beside the instant they
//! write there ([`PairLists::decide`](super::pair::PairLists::decide),
//! [`EncodeLists::decide`](super::encode::EncodeLists::decide)).

/// `stated`, the budget a host in process states for its request, as a
/// stage is shown it: the widest value where the host stated none, and
/// `stated` otherwise. Only a stage and the floor of a shadow read it: no
/// lease and no record keeps a request's budget.
#[inline]
pub(super) const fn shown(stated: u32) -> u32 {
    match stated {
        0 => u32::MAX,
        stated => stated,
    }
}

#[cfg(test)]
mod tests {
    //! What the gateway's own routing answers here: it states no budget for
    //! a selection and has no stage that reads one, so it passes no request
    //! by for want of time (`model_gateway/src/policies/mod.rs:486`, the
    //! facts a selection carries).

    use std::{sync::Arc, time::Instant};

    use parking_lot::Mutex;
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        test_support::{no_health_check, ready_worker},
        worker::{WorkerSpec, WorkerStatus, WorkerType},
    };

    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
            framework::{DecisionScope, Filter},
            list_cell::tests::PublishedList,
            view::{Candidate, Constraints, Identity, Phase, PoolFacts, RequestFacts},
            EncodeItems, EncodeLists, EncodePlaced, EncoderFacts, PairAdmission, PairLists,
            PartnerClasses, PlaceRequest, PolicyEngine,
        },
        stages::{registry::StageRegistry, Lifecycle},
        store::GatewayId,
    };

    const MODEL: &str = "model-a";

    /// A budget a host states, in microseconds.
    const STATED: u32 = 25_000;

    /// The budget each open of a scope was shown, in order.
    type Noted = Arc<Mutex<Vec<u32>>>;

    /// A filter of a host's own build that opens a scope over its rows, as
    /// a plugin's stage does. It keeps every row and notes the budget its
    /// request shows it.
    #[derive(Debug)]
    struct Notes(Noted);

    impl Lifecycle for Notes {}

    impl DecisionScope for Notes {
        fn open(&self, facts: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
            self.0.lock().push(facts.budget_micros);
        }

        fn close(&self) {}
    }

    impl Filter for Notes {
        fn name(&self) -> &'static str {
            "notes"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    /// A filter of a host's own build that opens no scope: the engine asks
    /// no stage of an encode side that opens one. It keeps every row and
    /// notes the budget the row's request shows it.
    #[derive(Debug)]
    struct NotesRows(Noted);

    impl Lifecycle for NotesRows {}

    impl Filter for NotesRows {
        fn name(&self) -> &'static str {
            "notes-rows"
        }

        fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            self.0.lock().push(facts.budget_micros);
            true
        }
    }

    /// The engine of `document` with a stage that notes its budget in every
    /// profile, and what that stage noted.
    fn noting(document: PolicySpec) -> (PolicyEngine, Noted) {
        let (engine, noted, _) = noting_each_side(document);
        (engine, noted)
    }

    /// [`noting`], with what the encode side's stage noted told apart: the
    /// stage of the profile named `encode` opens no scope and notes at each
    /// row it is asked for.
    fn noting_each_side(mut document: PolicySpec) -> (PolicyEngine, Noted, Noted) {
        let (noted, rows) = (Noted::default(), Noted::default());
        let (kept, kept_rows) = (Arc::clone(&noted), Arc::clone(&rows));
        let mut catalog = StageRegistry::builtin();
        catalog
            .filter("notes", move || Box::new(Notes(Arc::clone(&kept))))
            .expect("a free name");
        catalog
            .filter("notes-rows", move || {
                Box::new(NotesRows(Arc::clone(&kept_rows)))
            })
            .expect("a free name");
        document.generation = 1;
        for (name, profile) in &mut document.profiles {
            let stage = if name == "encode" {
                "notes-rows"
            } else {
                "notes"
            };
            profile.filter.push(stage.into());
        }
        let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
        let engine = PolicyEngine::new(spec.expect("the catalog holds the stages"));
        (engine, noted, rows)
    }

    /// Every budget a stage was shown since the last call was `shown`, and
    /// at least one stage was shown one.
    #[track_caller]
    fn all_shown(noted: &Noted, shown: u32, entry: &str) {
        let seen = std::mem::take(&mut *noted.lock());
        assert!(!seen.is_empty(), "{entry}: no stage opened");
        assert!(
            seen.iter().all(|&micros| micros == shown),
            "{entry}: {seen:?}"
        );
    }

    /// A request of `phase` that states `budget_micros` and nothing else.
    fn stating(phase: Phase, budget_micros: u32) -> Request<'static> {
        Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                phase,
                budget_micros,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    /// A request stated ahead, with the lists its host hands for it.
    struct Hands<'a>(Request<'a>, WorkerLists<'a>);

    impl RequestMaker for Hands<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.1
        }
    }

    fn fleet(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    /// Two ready workers of `worker_type`.
    fn side(name: &str, worker_type: WorkerType) -> Arc<[Arc<dyn Worker>]> {
        let ready = |at: usize| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
                worker_type,
                ..WorkerSpec::new(format!("grpc://{name}-{at}:9000"))
            })
            .health_config(no_health_check())
            .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker)
        };
        (0..2).map(ready).collect()
    }

    /// Every prefill has capacity and every decode a room.
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

    /// The turning itself: none stated is the widest, and a stated budget
    /// is shown as stated, the widest one too.
    #[test]
    fn a_budget_is_shown_as_stated_and_none_as_the_widest() {
        assert_eq!(super::shown(0), u32::MAX);
        for stated in [1, STATED, u32::MAX] {
            assert_eq!(super::shown(stated), stated);
        }
    }

    /// A single request that states no budget shows a stage the widest
    /// value at both kinds of entry, where the stage was shown `0` and a
    /// plugin's stage read its time as spent. A stated budget is shown as
    /// stated. The host's own request is not written.
    #[test]
    fn a_single_request_with_no_budget_shows_a_stage_the_widest_at_both_kinds_of_entry() {
        let preset = PolicySpec::preset("random").expect("a built-in preset");
        let (engine, noted) = noting(preset);
        let list = fleet(3);
        let members = PublishedList::new(Arc::clone(&list));
        for (stated, shown) in [(0, u32::MAX), (STATED, STATED)] {
            let asks = Hands(
                stating(Phase::Single, stated),
                WorkerLists::Single(members.lent()),
            );
            let now = Instant::now();
            let bare = engine.decide_without_lease(&asks, now);
            let bare = bare.expect("a decision").0;
            assert!(bare.pick().is_some(), "no pick: {bare:?}");
            all_shown(&noted, shown, "no lease");
            let leased = engine.decide_under(GatewayId::LOCAL, &asks, now);
            let leased = leased.expect("a decision").whole().expect("no place").0;
            assert!(leased.decision.pick().is_some(), "no pick: {leased:?}");
            all_shown(&noted, shown, "a lease");
            assert_eq!(asks.0.facts.budget_micros, stated);
        }
    }

    /// A pair that states no budget shows each side's stage the widest
    /// value: at its first decision with a lease and without, and at its
    /// decode asked again behind the prefill. A stated budget is shown as
    /// stated.
    #[test]
    fn a_pair_with_no_budget_shows_each_side_the_widest_at_both_kinds_of_entry() {
        let presets = PolicySpec::pair_presets("round_robin", "round_robin");
        let (engine, noted) = noting(presets.expect("the pair presets"));
        let prefills = side("prefill", WorkerType::Prefill);
        let decodes = side("decode", WorkerType::Decode);
        let members = [
            PublishedList::new(Arc::clone(&prefills)),
            PublishedList::new(Arc::clone(&decodes)),
        ];
        let classes = PartnerClasses::number(2, [&[0u32, 1][..], &[0, 1][..]]);
        let lists = PairLists {
            prefill: members[0].lent(),
            decode: members[1].lent(),
            classes: &classes,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        for (stated, shown) in [(0, u32::MAX), (STATED, STATED)] {
            let asks = Hands(stating(Phase::Prefill, stated), WorkerLists::Pair(lists));
            let now = Instant::now();
            let bare = engine.decide_without_lease(&asks, now);
            let bare = bare.expect("a decision").0;
            let held = *bare
                .pick()
                .expect("a pair")
                .targets
                .first()
                .expect("a target");
            all_shown(&noted, shown, "a pair, no lease");
            let again = engine.decide_decode_without_lease(&asks, &held, now);
            let again = again.expect("a decision");
            assert!(again.pick().is_some(), "no pick: {again:?}");
            all_shown(&noted, shown, "a decode asked again");
            let leased = engine.decide_under(GatewayId::LOCAL, &asks, now);
            let leased = leased.expect("a decision").whole().expect("no place").0;
            assert!(leased.decision.pick().is_some(), "no pick: {leased:?}");
            all_shown(&noted, shown, "a pair, a lease");
        }
    }

    /// The two calls a host makes with no engine above its lists turn a
    /// request's `0` too, in the host's own request: a pair's two lists and
    /// an encode request's three show each side's stage the widest value,
    /// where each was shown `0`. A stated budget is shown as stated.
    ///
    /// An encode request with no item is its pair's alone, and no stage of
    /// the encode side is asked for it: that side's stage is read under a
    /// request with an item.
    #[test]
    fn the_calls_over_a_hosts_lists_show_a_stage_the_widest_with_no_engine_above() {
        let preset = "power_of_two";
        let presets = PolicySpec::epd_presets(preset, preset, preset);
        let (engine, noted, rows) = noting_each_side(presets.expect("the presets"));
        let members = [0, 1, 2].map(|list| {
            let ready = |at| ready_worker(&format!("http://list-{list}-worker-{at}:8000"), MODEL);
            let workers: Arc<[Arc<dyn Worker>]> = (0..2).map(ready).collect();
            PublishedList::new(workers)
        });
        let classes = PartnerClasses::number(2, [&[0u32, 1][..], &[0, 1][..]]);
        let pair = PairLists {
            prefill: members[0].lent(),
            decode: members[1].lent(),
            classes: &classes,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        // Each encoder hands to both prefills.
        let hands = [EncoderFacts {
            class: 1,
            ..EncoderFacts::default()
        }; 2];
        let lists = EncodeLists {
            pair,
            encoders: members[2].lent(),
            encoders_places: None,
            encoders_stalled: &[],
            encoders_held: &[],
            facts: &hands,
            prefill_classes: &[1, 1],
        };
        let item = [Identity::plain("item")];
        let now = Instant::now();
        for (stated, shown) in [(0, u32::MAX), (STATED, STATED)] {
            let mut theirs = stating(Phase::Prefill, stated);
            let decision = pair.decide(&engine, None, now, &mut theirs, None);
            assert!(decision.pick().is_some(), "no pick: {decision:?}");
            all_shown(&noted, shown, "a pair's lists");
            assert_eq!(theirs.facts.budget_micros, shown);

            let mut theirs = stating(Phase::Prefill, stated);
            let (none, mut placed) = (EncodeItems::default(), EncodePlaced::new());
            let decision = lists.decide(&engine, None, now, &mut theirs, none, &mut placed);
            assert!(decision.pick().is_some(), "no pick: {decision:?}");
            all_shown(&noted, shown, "an encode request's lists");
            assert_eq!(theirs.facts.budget_micros, shown);
            assert!(rows.lock().is_empty(), "no item, no encode side");

            let mut theirs = stating(Phase::Prefill, stated);
            let one = EncodeItems {
                keys: &item,
                ..EncodeItems::default()
            };
            let decision = lists.decide(&engine, None, now, &mut theirs, one, &mut placed);
            assert!(decision.pick().is_some(), "no pick: {decision:?}");
            all_shown(&noted, shown, "an encode request's pair, with an item");
            all_shown(&rows, shown, "an encode request's encoders");
        }
    }

    /// The placing call states no budget unless its caller writes one
    /// ([`PlaceRequest::of`]): its stage is shown the widest value, and a
    /// budget the caller writes as written.
    #[test]
    fn a_placing_call_with_no_budget_shows_a_stage_the_widest() {
        let preset = PolicySpec::preset("random").expect("a built-in preset");
        let (engine, noted) = noting(preset);
        let engine: &'static PolicyEngine = Box::leak(Box::new(engine));
        let list = fleet(3);
        let members = PublishedList::new(Arc::clone(&list));
        let mut request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
        assert_eq!(request.budget_micros, 0);
        drop(engine.place(&request).expect("a placement"));
        all_shown(&noted, u32::MAX, "a place, no budget");
        request.budget_micros = STATED;
        drop(engine.place(&request).expect("a placement"));
        all_shown(&noted, STATED, "a place, a budget");
    }
}
