use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};

use super::{BoundPlugin, Fallback, PluginFilter, PluginPicker, PluginScorer};
use crate::{
    document::spec::{Needs, PolicySpec, SpecError},
    engine::{
        decision::{Decision, Pick, Target},
        framework::{DecisionScope, Scorer, TieBand},
        view::{Candidate, Health, Phase, PoolFacts, RequestFacts, WorkerId},
    },
    plugins::shared::config::Failure,
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    tracking::lease::{
        EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal,
        TransportClass,
    },
};

/// A plugin that declares nothing and can answer scores alone.
#[derive(Debug, Default)]
struct Quiet(Option<TieBand>);

impl DecisionScope for Quiet {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {}
}

impl BoundPlugin for Quiet {
    fn name(&self) -> &'static str {
        "quiet"
    }

    fn failure(&self) -> Failure {
        Failure::Open
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn observes(&self) -> EventSet {
        EventSet::NONE
    }

    fn take_observer(&self) -> bool {
        false
    }

    fn retain(&self, _: &RequestFacts<'_>, _: &mut Vec<u32>) -> Result<(), Fallback> {
        Err(Fallback)
    }

    fn keep(&self, _: &RequestFacts<'_>, _: u32) -> Result<bool, Fallback> {
        Err(Fallback)
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[u32], out: &mut [f32]) -> Result<TieBand, Fallback> {
        let band = self.0.ok_or(Fallback)?;
        out.fill(0.5);
        Ok(band)
    }

    fn pick(&self, _: &RequestFacts<'_>, _: &[u32], _: &[f32]) -> Result<u32, Fallback> {
        Err(Fallback)
    }

    fn observe_issued(&self, _: &Lease, _: &Decision) {}

    fn observe_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {}

    fn observe_completed(&self, _: &Lease, _: &Outcome) {}

    fn observe_released(&self, _: &Lease, _: ReleaseReason) {}

    fn observe_membership(&self, _: &Membership<'_>) {}

    fn observe_signal(&self, _: &Signal<'_>) {}
}

/// A bound plugin that answers no stage and counts the events it is
/// told, by kind.
#[derive(Debug, Default)]
struct Told {
    observes: EventSet,
    observer_taken: AtomicBool,
    told: [AtomicU32; 6],
}

impl Told {
    fn listing(observes: EventSet) -> Arc<Self> {
        Arc::new(Self {
            observes,
            ..Self::default()
        })
    }

    fn note(&self, kind: EventKind) {
        let at = EventKind::ALL
            .iter()
            .position(|known| *known == kind)
            .expect("a kind of the six");
        self.told[at].fetch_add(1, Ordering::Relaxed);
    }

    /// How often each kind was told, in the order of [`EventKind::ALL`].
    fn told(&self) -> [u32; 6] {
        std::array::from_fn(|at| self.told[at].load(Ordering::Relaxed))
    }
}

impl DecisionScope for Told {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {}
}

impl BoundPlugin for Told {
    fn name(&self) -> &'static str {
        "told"
    }

    fn failure(&self) -> Failure {
        Failure::Open
    }

    fn needs(&self) -> Needs {
        Needs::default()
    }

    fn observes(&self) -> EventSet {
        self.observes
    }

    fn take_observer(&self) -> bool {
        !self.observer_taken.swap(true, Ordering::Relaxed)
    }

    fn retain(&self, _: &RequestFacts<'_>, _: &mut Vec<u32>) -> Result<(), Fallback> {
        Err(Fallback)
    }

    fn keep(&self, _: &RequestFacts<'_>, _: u32) -> Result<bool, Fallback> {
        Err(Fallback)
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[u32], _: &mut [f32]) -> Result<TieBand, Fallback> {
        Err(Fallback)
    }

    fn pick(&self, _: &RequestFacts<'_>, _: &[u32], _: &[f32]) -> Result<u32, Fallback> {
        Err(Fallback)
    }

    fn observe_issued(&self, _: &Lease, _: &Decision) {
        self.note(EventKind::Issued);
    }

    fn observe_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.note(EventKind::Dispatched);
    }

    fn observe_completed(&self, _: &Lease, _: &Outcome) {
        self.note(EventKind::Completed);
    }

    fn observe_released(&self, _: &Lease, _: ReleaseReason) {
        self.note(EventKind::Released);
    }

    fn observe_membership(&self, _: &Membership<'_>) {
        self.note(EventKind::Membership);
    }

    fn observe_signal(&self, _: &Signal<'_>) {
        self.note(EventKind::Signal);
    }
}

/// A fallback scorer whose state the lifecycle's calls keep: it counts
/// them.
#[derive(Debug)]
struct Kept(Arc<AtomicU32>);

impl Kept {
    const BAND: TieBand = TieBand(0.625);

    fn count(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl Lifecycle for Kept {
    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.count();
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.count();
    }

    fn on_completed(&self, _: &Lease, _: &Outcome) {
        self.count();
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.count();
    }

    fn on_membership(&self, _: &Membership<'_>) {
        self.count();
    }

    fn on_signal(&self, _: &Signal<'_>) {
        self.count();
    }

    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl Scorer for Kept {
    fn name(&self) -> &'static str {
        "kept"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        Self::BAND
    }
}

/// Every lifecycle call once, in the order of [`EventKind::ALL`].
fn tell(stage: &dyn Lifecycle) {
    let lease = Lease::default();
    stage.on_decision_issued(&lease, &Decision::Pick(Pick::default()));
    stage.on_dispatched(&lease, &Target::default(), Side::Single, 0);
    stage.on_completed(
        &lease,
        &Outcome {
            side: Side::Single,
            served: WorkerId(1),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
    );
    stage.on_released(&lease, ReleaseReason::ClientGone);
    stage.on_membership(&Membership::Added(WorkerId(1)));
    stage.on_signal(&Signal::Health {
        worker: WorkerId(1),
        health: Health::default(),
    });
}

/// A host names a decision's rows in the affinity column's place only
/// for a profile whose every stage answers row by row, and a plugin's
/// stage does not. A profile that holds one, as a filter or as a scorer
/// beside the affinity stage, is named no rows: its plugin reads the
/// column whole. A profile of row-wise stages that reads the column and
/// takes no sample is named every holder.
#[test]
fn a_profile_that_holds_a_plugins_stage_is_named_no_rows() {
    // The sample of the profile that decides, and the cap a host names
    // its rows under.
    let named = |catalog: &StageRegistry, stages: &str| {
        let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{{stages},"pick":{{"best":{{"sample":16}}}}}}}}}}"#
            ))
            .expect("a well-formed document");
        let engine = document
            .validate_in(catalog, &[])
            .expect("a document of this build");
        (
            engine.sample_for("m", Phase::Single),
            engine.affinity_rows_cap("m", Phase::Single),
        )
    };
    let stock = StageRegistry::builtin();
    assert_eq!(named(&stock, r#""score":["prefix-affinity"]"#), (16, 16));
    // A catalog holds a name once, whatever the stage's kind: one a
    // shape.
    let mut filtering = StageRegistry::builtin();
    filtering
        .filter("quiet", || {
            Box::new(PluginFilter::new(Arc::new(Quiet::default()), None))
        })
        .expect("a free name, the stage's own");
    let filtered = r#""filter":["quiet"],"score":["prefix-affinity"]"#;
    assert_eq!(named(&filtering, filtered), (0, 0));
    let mut scoring = StageRegistry::builtin();
    scoring
        .scorer("quiet", || {
            Box::new(PluginScorer::new(Arc::new(Quiet::default()), None))
        })
        .expect("a free name, the stage's own");
    let scored = r#""score":["prefix-affinity","quiet"]"#;
    assert_eq!(named(&scoring, scored), (0, 0));
}

/// A plugin's rule is its own, and its stage of each kind says
/// nothing of what it may refuse by: the trait's own answer stands,
/// that it may refuse by a report of leases, and the host of its
/// decision applies what its gateways reported first.
#[test]
fn a_plugins_stage_of_each_kind_is_taken_as_one_that_may_refuse_by_reports() {
    use crate::engine::framework::{Filter, Picker};

    assert!(PluginFilter::new(Arc::new(Quiet::default()), None).refuses_by_reports());
    assert!(PluginScorer::new(Arc::new(Quiet::default()), None).refuses_by_reports());
    assert!(PluginPicker::new(Arc::new(Quiet::default()), None).refuses_by_reports());
}

/// A plugin's scorer says nothing of a gate of its own: the trait's own
/// answer stands, that it weighs no load against affinity. So a guest
/// lifts the overload veto for no profile, whatever it is fed: its
/// adapter answers no behind a fallback that reads the affinity column
/// and says the word of itself too.
#[test]
fn a_plugins_scorer_says_no_gate_of_its_own() {
    use crate::{
        document::spec::PrefixAffinityParams,
        policies::cache_aware::{Gates, PrefixAffinity},
    };

    let bare = PluginScorer::new(Arc::new(Quiet::default()), None);
    assert!(!bare.balances_load_against_affinity());
    let params = PrefixAffinityParams::default();
    let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, None));
    let fallback: Box<dyn Scorer> = Box::new(stage);
    assert!(fallback.balances_load_against_affinity());
    let behind = PluginScorer::new(Arc::new(Quiet::default()), Some(fallback));
    assert!(behind.needs().affinity);
    assert!(!behind.balances_load_against_affinity());
}

/// The scorer's adapter returns the band its plugin answered, and its
/// fallback's band where it takes the fallback.
#[test]
fn the_scorer_adapter_returns_the_band_its_plugin_answered() {
    let facts = RequestFacts::default();
    let candidates = [Candidate::default()];
    let kept = [0];
    let mut out = [0.0];
    let band = TieBand(0.3125);
    let scorer = PluginScorer::new(Arc::new(Quiet(Some(band))), None);
    assert_eq!(scorer.score(&facts, &candidates, &kept, &mut out), band);

    let fallback = Box::new(Kept(Arc::default()));
    let scorer = PluginScorer::new(Arc::new(Quiet::default()), Some(fallback));
    assert_eq!(
        scorer.score(&facts, &candidates, &kept, &mut out),
        Kept::BAND
    );
}

/// The forwarding adapter hands the plugin the kinds its row lists and
/// no other, each once: none for a row that lists nothing, every kind
/// for one that lists them all, and each kind alone its own call.
#[test]
fn an_adapter_hands_the_kinds_the_row_lists_and_no_other() {
    let alone = EventKind::EVERY.map(|kind| EventSet::of([kind]));
    let two = EventSet::of([EventKind::Completed, EventKind::Signal]);
    for listed in [EventSet::NONE, EventSet::ALL, two]
        .into_iter()
        .chain(alone)
    {
        let wanted = EventKind::EVERY.map(|kind| u32::from(listed.contains(kind)));
        let scored = Told::listing(listed);
        tell(&PluginScorer::new(Arc::clone(&scored), None));
        assert_eq!(scored.told(), wanted, "the scorer, {listed:?}");
        let picked = Told::listing(listed);
        tell(&PluginPicker::new(Arc::clone(&picked), None));
        assert_eq!(picked.told(), wanted, "the picker, {listed:?}");
    }
}

/// One adapter of a build forwards, the first made, and every scorer and
/// picker of a plugin that hears says it needs the lifecycle's calls,
/// the forwarding one or not: a host that makes none refuses each. A
/// plugin that hears nothing leaves that word to its fallback.
#[test]
fn a_plugin_that_hears_needs_the_lifecycle_whichever_adapter_forwards() {
    let hearing = Told::listing(EventSet::of([EventKind::Released]));
    let first = PluginScorer::new(Arc::clone(&hearing), None);
    let second = PluginPicker::new(Arc::clone(&hearing), None);
    tell(&second);
    assert_eq!(
        hearing.told(),
        [0; 6],
        "the second adapter forwards nothing"
    );
    tell(&first);
    assert_eq!(hearing.told(), [0, 0, 0, 1, 0, 0]);
    assert!(first.needs_lifecycle() && second.needs_lifecycle());

    let unlisted = Told::listing(EventSet::NONE);
    assert!(!PluginScorer::new(Arc::clone(&unlisted), None).needs_lifecycle());
    assert!(!PluginPicker::new(Arc::clone(&unlisted), None).needs_lifecycle());
    let kept: Box<dyn Scorer> = Box::new(Kept(Arc::default()));
    assert!(PluginScorer::new(unlisted, Some(kept)).needs_lifecycle());
}

/// A fallback is told every event, whatever the plugin's row lists.
#[test]
fn a_fallback_is_told_every_event_whatever_the_row_lists() {
    for listed in [
        EventSet::NONE,
        EventSet::of([EventKind::Issued]),
        EventSet::ALL,
    ] {
        let calls = Arc::new(AtomicU32::new(0));
        let fallback: Box<dyn Scorer> = Box::new(Kept(Arc::clone(&calls)));
        tell(&PluginScorer::new(Told::listing(listed), Some(fallback)));
        assert_eq!(calls.load(Ordering::Relaxed), 6, "{listed:?}");
    }
}

/// A document that names a hearing plugin's scorer is refused by a host
/// that makes no lifecycle call, by the stage's field. One that hears
/// nothing is built there, and a host that makes the calls builds the
/// hearing one.
#[test]
fn a_document_naming_a_plugin_that_hears_is_refused_where_the_host_makes_no_call() {
    let catalog_of = |listed: EventSet| {
        let plugin = Told::listing(listed);
        let mut catalog = StageRegistry::builtin();
        catalog
            .scorer("told", move || {
                Box::new(PluginScorer::new(Arc::clone(&plugin), None))
            })
            .expect("a free name, the stage's own");
        catalog
    };
    let document = || -> PolicySpec {
        serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["told"],"pick":"best"}}}"#,
        )
        .expect("a well-formed document")
    };
    let hearing = EventSet::of([EventKind::Signal]);

    let refused = document()
        .validate_in(&catalog_of(hearing).without_lifecycle(), &[])
        .err();
    assert_eq!(
        refused,
        Some(SpecError::NotInBuild {
            path: "profiles.p.score[0].told".to_string(),
            what: "the lifecycle this stage needs",
        })
    );
    let unheard = document()
        .validate_in(&catalog_of(EventSet::NONE).without_lifecycle(), &[])
        .expect("a plugin that hears nothing needs no call");
    assert!(!unheard.needs_lifecycle());
    let served = document()
        .validate_in(&catalog_of(hearing), &[])
        .expect("the host makes the calls");
    assert!(served.needs_lifecycle());
}
