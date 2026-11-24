//! A gateway's standing vetoes, written into the records of a decision
//! that shows them to a plugin.
//!
//! A host that decides for many gateways hands one list's records to each
//! of them. It states the asking gateway's standing vetoes beside the
//! records: the places they hold back ([`Constraints::stalled`]) and those
//! of them no rule lifts ([`Constraints::held`]). The crate's own stages
//! read a record's bit and the mask beside it.
//!
//! A plugin is shown each record as the record's own fields, and no mask.
//! The engine declines a profile that shows its rows to a plugin over rows
//! that carry a mask beside them, under the stage name `written-rows`. So
//! for such a profile the vetoes are written into the records
//! ([`written`]), in the form of a row written for its decision: the
//! stalled bit on each masked record, and the breaker bit beside it where
//! the veto holds ([`Health`](crate::engine::view::Health)). The decision
//! is then put to constraints that state no mask.
//!
//! The records are those the decision wrote for itself, in its thread's
//! own buffer ([`PoolRecords::write`](super::PoolRecords::write)): no
//! other decision reads them.

use crate::engine::{
    framework::Engine,
    view::{Candidate, Constraints, Phase},
};

/// The constraints a decision of `phase` for `model` is put to over
/// `rows`, every record of its list written for it, where its host states
/// a standing veto beside them. Where the decision's profile shows its
/// rows to a plugin ([`Engine::reads_for`]), the vetoes are written into
/// `rows` and the constraints state no mask. Under any other profile
/// `rows` stand as written and the constraints are `cx`'s own.
///
/// A record a hold stands on carries both bits, as the record of a worker
/// whose breaker refuses execution does. A place past the list's end
/// names no record.
///
/// Cold and out of line: a decision under no veto asks one test of its
/// constraints and never comes here.
#[cold]
#[inline(never)]
pub(crate) fn written<'a>(
    engine: &Engine,
    model: &str,
    phase: Phase,
    rows: &mut [Candidate],
    cx: &Constraints<'a>,
) -> Constraints<'a> {
    let (_, shown, _) = engine.reads_for(model, phase);
    if !shown {
        return *cx;
    }
    for &place in cx.stalled {
        if let Some(row) = rows.get_mut(place as usize) {
            row.health.stalled = true;
        }
    }
    for &place in cx.held {
        if let Some(row) = rows.get_mut(place as usize) {
            row.health.stalled = true;
            row.health.breaker_open = true;
        }
    }
    Constraints {
        stalled: &[],
        held: &[],
        ..*cx
    }
}

#[cfg(test)]
mod tests {
    //! The classic router keeps a doubt of a worker on the worker's own
    //! object, in its one process, and selects no worker that carries one
    //! (`crates/worker/src/identity/basic.rs:438-443`). It keeps no veto of
    //! one gateway among several, and it has no stage that is shown its rows.

    use std::{sync::Arc, time::Instant};

    use parking_lot::Mutex;
    use smg_worker::{identity::basic::Worker, test_support::ready_worker};

    use super::{
        super::{decide, with_shadow, DecisionClock, Live, PoolRecords},
        written,
    };
    use crate::{
        document::spec::{AppliedSpec, PolicySpec, ShadowSpec, PRESETS},
        engine::{
            decision::{Decision, ShadowKind, ShedReason},
            decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
            framework::{DecisionScope, Filter},
            list_cell::tests::PublishedList,
            policy_engine::PolicyEngine,
            shadow::Shadow,
            view::{Candidate, Constraints, Phase, PoolFacts, RequestFacts},
        },
        stages::{registry::StageRegistry, Lifecycle},
        store::GatewayId,
    };

    const MODEL: &str = "model-a";

    /// What a scope was shown when it last opened: each place whose own
    /// stalled bit was set, with its breaker bit.
    type Seen = Arc<Mutex<Vec<(u32, bool)>>>;

    /// A filter of a host's own build that opens a scope over the rows, as
    /// a plugin's stage does. It keeps every row and notes the vetoes the
    /// rows themselves carried.
    #[derive(Debug)]
    struct Shown(Seen);

    impl Lifecycle for Shown {}

    impl DecisionScope for Shown {
        fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], _: &PoolFacts<'_>) {
            let vetoed = rows.iter().filter(|row| row.health.stalled);
            *self.0.lock() = vetoed
                .map(|row| (row.position, row.health.breaker_open))
                .collect();
        }

        fn close(&self) {}
    }

    impl Filter for Shown {
        fn name(&self) -> &'static str {
            "shown"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    /// The stages a document may name here: the crate's own, and one that
    /// is shown its rows. Beside them, what that stage last saw.
    fn catalog() -> (StageRegistry, Seen) {
        let seen = Seen::default();
        let noted = Arc::clone(&seen);
        let mut catalog = StageRegistry::builtin();
        catalog
            .filter("shown", move || Box::new(Shown(Arc::clone(&noted))))
            .expect("a free name");
        (catalog, seen)
    }

    /// `document` as it stands, and with a stage that is shown its rows in
    /// every profile, beside what that stage last saw.
    fn with_a_shown_stage(mut document: PolicySpec) -> (AppliedSpec, AppliedSpec, Seen) {
        document.generation = 1;
        let plain = AppliedSpec::validate(&document).expect("the document validates");
        let (catalog, seen) = catalog();
        for profile in document.profiles.values_mut() {
            profile.filter.push("shown".into());
        }
        let shown = AppliedSpec::validate_in(&document, &catalog, &[]);
        (plain, shown.expect("the catalog holds the stage"), seen)
    }

    fn preset(policy: &str) -> PolicySpec {
        PolicySpec::preset(policy).expect("a built-in preset")
    }

    fn fleet(workers: usize) -> Vec<Arc<dyn Worker>> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    fn single(seed: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: MODEL,
            phase: Phase::Single,
            seed,
            ..RequestFacts::default()
        }
    }

    /// One decision over the records of `workers` under `spec`, for a
    /// request of `seed` whose host states `cx`.
    fn decided(
        spec: &AppliedSpec,
        workers: &[Arc<dyn Worker>],
        seed: u64,
        cx: &Constraints<'_>,
    ) -> Decision {
        let clock = DecisionClock::new();
        let pool = PoolRecords::new(workers, spec, &clock);
        let live = Live::at(spec.engine(), &clock, Instant::now(), None, None);
        decide(spec.engine(), &pool, workers, &live, &mut single(seed), cx)
    }

    /// A request stated ahead, with the one list its host hands for it.
    struct Hands<'a>(Request<'a>, &'a PublishedList);

    impl RequestMaker for Hands<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            WorkerLists::Single(self.1.lent())
        }
    }

    /// The places a decision names, in its order: none for a shed.
    fn places(decision: &Decision) -> Vec<u32> {
        decision.pick().map_or_else(Vec::new, |pick| {
            pick.targets.iter().map(|target| target.position).collect()
        })
    }

    /// Under a profile that shows its rows to no plugin the records stand
    /// as written and the mask stays beside them: the crate's own stages
    /// read both. Under one that shows them, the records carry the bits
    /// and no mask is left. A place past the list's end names no record.
    #[test]
    fn the_vetoes_are_written_for_a_profile_that_shows_its_rows_and_for_no_other() {
        let (plain, shown, _) = with_a_shown_stage(preset("random"));
        let vetoed = Constraints {
            stalled: &[1, 3, 9],
            held: &[3, 9],
            ..Constraints::default()
        };
        let mut rows = [Candidate::default(); 4];
        let bits =
            |rows: &[Candidate; 4]| rows.map(|row| (row.health.stalled, row.health.breaker_open));
        let beside = written(plain.engine(), MODEL, Phase::Single, &mut rows, &vetoed);
        assert_eq!((beside.stalled, beside.held), (vetoed.stalled, vetoed.held));
        assert_eq!(bits(&rows), [(false, false); 4]);
        let inside = written(shown.engine(), MODEL, Phase::Single, &mut rows, &vetoed);
        assert!(inside.stalled.is_empty() && inside.held.is_empty());
        let (free, doubted, held) = ((false, false), (true, false), (true, true));
        assert_eq!(bits(&rows), [free, doubted, free, held]);
    }

    /// A standing veto its host states beside the records is written into
    /// them for a profile that shows its rows: the stage is shown each
    /// vetoed record's own stalled bit, with the breaker bit where the
    /// veto holds. The request is decided, and it is handed no vetoed
    /// worker. Before, the engine declined it under `written-rows`.
    #[test]
    fn a_veto_beside_rows_a_plugin_is_shown_is_written_into_them() {
        let (_, shown, seen) = with_a_shown_stage(preset("random"));
        let workers = fleet(4);
        let vetoed = Constraints {
            stalled: &[1, 3],
            held: &[3],
            ..Constraints::default()
        };
        for seed in 0..32 {
            let named = places(&decided(&shown, &workers, seed, &vetoed));
            assert!(!named.is_empty(), "seed {seed}: a pick, not a decline");
            assert!(
                named.iter().all(|place| [0, 2].contains(place)),
                "seed {seed}: {named:?}"
            );
            assert_eq!(*seen.lock(), [(1, false), (3, true)], "seed {seed}");
        }
        // Under no veto the stage is shown none.
        let free = places(&decided(&shown, &workers, 0, &Constraints::default()));
        assert!(!free.is_empty(), "a pick");
        assert!(seen.lock().is_empty());
    }

    /// Records with the vetoes written into them are decided as the same
    /// records with the mask beside them: under each stock policy, a
    /// document whose stage is shown its rows names the workers the stock
    /// document names, for the same request under the same vetoes. Where
    /// every worker is vetoed the binding's rule for such a pool decides
    /// in both, and it never hands a worker whose veto holds.
    #[test]
    fn rows_written_under_a_veto_are_decided_as_the_mask_beside_them_is() {
        let masks: [(&[u32], &[u32]); 4] = [
            (&[1], &[]),
            (&[0, 2], &[2]),
            (&[0, 1, 2, 3], &[]),
            (&[0, 1, 2, 3], &[0, 3]),
        ];
        for policy in PRESETS {
            for (stalled, held) in masks {
                // Two engines made alike, so a rotation turns alike.
                let (plain, shown, _) = with_a_shown_stage(preset(policy));
                let workers = fleet(4);
                let cx = Constraints {
                    stalled,
                    held,
                    ..Constraints::default()
                };
                for seed in 0..16 {
                    let beside = places(&decided(&plain, &workers, seed, &cx));
                    let written = places(&decided(&shown, &workers, seed, &cx));
                    let under = format!("{policy}, {stalled:?} held {held:?}, seed {seed}");
                    assert!(!written.is_empty(), "{under}: a pick");
                    assert_eq!(written, beside, "{under}");
                    assert!(written.iter().all(|place| !held.contains(place)), "{under}");
                }
            }
        }
    }

    /// A decision followed by its binding's shadow takes the vetoes written
    /// too: the live profile, which shows its rows, decides under the
    /// veto. The shadow profile, the stock one, reads the same records
    /// under the same constraints and names the same first worker.
    #[test]
    fn a_decision_with_a_shadow_behind_it_is_decided_under_a_veto_too() {
        let (catalog, seen) = catalog();
        let mut document = preset("random");
        document.generation = 1;
        let stock = document.profiles.values().next().expect("a profile");
        let stock = stock.clone();
        for profile in document.profiles.values_mut() {
            profile.filter.push("shown".into());
        }
        document.profiles.insert("twin".into(), stock);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        let shown = AppliedSpec::validate_in(&document, &catalog, &[]);
        let shown = shown.expect("the catalog holds the stage");
        let workers = fleet(4);
        let clock = DecisionClock::new();
        let pool = PoolRecords::new(&workers, &shown, &clock);
        let engine = shown.engine();
        let vetoed = Constraints {
            stalled: &[0, 2],
            held: &[2],
            ..Constraints::default()
        };
        for seed in 0..16 {
            let shadow = engine.shadow_for(MODEL, Phase::Single);
            let shadow = shadow.expect("the binding names a shadow");
            let live = Live::at(engine, &clock, Instant::now(), None, None);
            let mut facts = single(seed);
            let (decision, said) =
                with_shadow::decide(engine, &pool, &workers, &live, &mut facts, &vetoed, shadow);
            let named = places(&decision);
            assert!(
                !named.is_empty() && named.iter().all(|place| [1, 3].contains(place)),
                "seed {seed}: {named:?}"
            );
            assert_eq!(*seen.lock(), [(0, false), (2, true)], "seed {seed}");
            let Shadow::Said(said) = said else {
                panic!("seed {seed}: the shadow decides, not {said:?}");
            };
            assert_eq!(said.kind, ShadowKind::Agreed, "seed {seed}");
        }
    }

    /// The engine over worker lists decides a request whose host states a
    /// standing veto, under a document with a stage that is shown its
    /// rows, at both kinds of entry: a pick that names no vetoed worker,
    /// where both declined it under `written-rows`.
    #[test]
    fn a_request_under_a_veto_is_decided_over_a_list_at_both_kinds_of_entry() {
        let (_, shown, seen) = with_a_shown_stage(preset("random"));
        let engine = PolicyEngine::new(shown);
        let list: Arc<[Arc<dyn Worker>]> = fleet(4).into();
        let members = PublishedList::new(Arc::clone(&list));
        let declined = (ShedReason::PolicyDeclined, "written-rows");
        for seed in 0..16 {
            let stated = Request {
                id: b"request",
                facts: single(seed),
                constraints: Constraints {
                    stalled: &[1, 2],
                    held: &[1],
                    ..Constraints::default()
                },
            };
            let asks = Hands(stated, &members);
            let now = Instant::now();
            let bare = engine.decide_without_lease(&asks, now);
            let bare = bare.expect("a decision").0;
            let leased = engine.decide_under(GatewayId::LOCAL, &asks, now);
            let leased = leased.expect("a decision").whole().expect("no place").0;
            for decision in [&bare, &leased.decision] {
                if let Decision::Shed(shed) = decision {
                    assert_ne!((shed.reason, shed.emptied_by), declined, "seed {seed}");
                }
                let named = places(decision);
                assert!(
                    !named.is_empty() && named.iter().all(|place| [0, 3].contains(place)),
                    "seed {seed}: {named:?}"
                );
                assert_eq!(*seen.lock(), [(1, true), (2, false)], "seed {seed}");
            }
        }
    }
}
