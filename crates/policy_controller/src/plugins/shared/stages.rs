//! A bound plugin as the framework's stages: a filter, a scorer and a
//! picker, each with the fallback stage its row names. The three adapters of
//! one profile share the bound plugin's decision (their [`DecisionScope`]);
//! the first scorer or picker made in an engine build, whichever of the
//! build's profiles binds the plugin, forwards the lifecycle, so an event
//! reaches the plugin once per engine. It forwards the kinds of event the
//! plugin's row lists and no other: a row lists none by default, and a
//! plugin whose row lists any has its scorer and picker say they need the
//! lifecycle's calls, forwarding or not. A fallback is a built-in stage: the
//! engine opens the scopes of a profile's own stages, not a fallback's. Each
//! adapter declares the plugin's needs and its fallback's, so the profile's
//! needs hold them whichever stage the plugin is bound as. The adapters are
//! generic over the plugin's kind: what a kind bound is their
//! [`BoundPlugin`], declared here beside its only readers.

use std::sync::Arc;

use super::config::Failure;
use crate::{
    document::spec::Needs,
    engine::{
        decision::{Decision, Target, Targets},
        framework::{DecisionScope, Filter, Picker, Scorer, TieBand},
        view::{Candidate, Constraints, RequestFacts},
    },
    stages::{lifecycle::Lifecycle, pickers::Best},
    tracking::lease::{
        EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal,
    },
};

/// The three stages a profile may bind a plugin as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Filter,
    Score,
    Pick,
}

/// The stage takes its fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fallback;

/// A plugin bound to one profile, as its three stage adapters read it: the
/// decision the engine opens through [`DecisionScope`] answers the filter,
/// the scorer and the picker, and one adapter of an engine build forwards
/// the lifecycle, one `observe_` call per event of a kind the plugin's row
/// lists and no call for an event of another. This crate's plugin kinds
/// implement it and these adapters alone call it. It is public for the
/// adapters' bounds and no public path names it: a host builds the adapters
/// from what a kind bound and never writes this name.
pub trait BoundPlugin: DecisionScope + 'static {
    /// The plugin's name: the trace and a shed carry it as the stage.
    fn name(&self) -> &'static str;
    /// What a stage does when the plugin does not answer and its row names
    /// no fallback stage.
    fn failure(&self) -> Failure;
    /// What the host must feed for the plugin's stages to mean anything.
    fn needs(&self) -> Needs;
    /// The kinds of lifecycle event the plugin's row lists: the adapter
    /// that forwards the lifecycle hands those and no other.
    fn observes(&self) -> EventSet;
    /// Claims the lifecycle for the adapter that asks: `true` for the first
    /// of an engine build, so an event reaches the plugin once per build.
    fn take_observer(&self) -> bool;
    /// The profile binds the plugin as `stage`: an adapter says so when it
    /// is made. A kind asked stage by stage ignores it; a kind asked once
    /// per decision names its frame's stages from it.
    fn bound_to(&self, stage: Stage) {
        let _ = stage;
    }
    /// The filter's answer: `kept` keeps the positions the plugin kept.
    fn retain(&self, facts: &RequestFacts<'_>, kept: &mut Vec<u32>) -> Result<(), Fallback>;
    /// One row of the filter's answer.
    fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback>;
    /// The scorer's answer. `out[i]` is the score of `kept[i]`. The result
    /// carries the plugin's tie band, or [`TieBand::NORMALIZED`] if the
    /// plugin left it unset.
    fn score(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        out: &mut [f32],
    ) -> Result<TieBand, Fallback>;
    /// The picker's answer over the survivors (`scores[i]` belongs to
    /// `kept[i]`): one surviving position.
    fn pick(&self, facts: &RequestFacts<'_>, kept: &[u32], scores: &[f32])
        -> Result<u32, Fallback>;
    fn observe_issued(&self, lease: &Lease, decision: &Decision);
    fn observe_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    );
    fn observe_completed(&self, lease: &Lease, outcome: &Outcome);
    fn observe_released(&self, lease: &Lease, reason: ReleaseReason);
    fn observe_membership(&self, change: &Membership<'_>);
    fn observe_signal(&self, signal: &Signal<'_>);
}

/// What one adapter needs from the host: its plugin's declared needs and
/// its fallback's, since either answers the call. The same for each stage
/// kind: a plugin that reads the affinity facts as the filter alone or as
/// the picker alone is a reader as one that scores is.
fn stage_needs<B: BoundPlugin>(bound: &B, fallback: Option<Needs>) -> Needs {
    let own = bound.needs();
    fallback.map_or(own, |fallback| own.or(fallback))
}

/// What one scorer or picker forwards of the lifecycle: the kinds the
/// plugin's row lists where the adapter is its build's observer, none where
/// another adapter is.
fn forwarded<B: BoundPlugin>(bound: &B) -> EventSet {
    if bound.take_observer() {
        bound.observes()
    } else {
        EventSet::NONE
    }
}

/// The plugin's `filter` stage. Without a fallback, a failure keeps every
/// row under `failure: open` and empties the pool under `closed`. A filter
/// is not a lifecycle stage of the framework (the engine forwards the
/// lifecycle to scorers and pickers), so a filter-only binding observes
/// nothing, whatever its row lists.
#[derive(Debug)]
pub struct PluginFilter<B> {
    bound: Arc<B>,
    fallback: Option<Box<dyn Filter>>,
    soft: bool,
}

impl<B: BoundPlugin> PluginFilter<B> {
    pub fn new(bound: Arc<B>, fallback: Option<Box<dyn Filter>>) -> Self {
        bound.bound_to(Stage::Filter);
        Self {
            bound,
            fallback,
            soft: false,
        }
    }

    /// A soft filter steps aside when it would empty the pool.
    pub fn soft(mut self, soft: bool) -> Self {
        self.soft = soft;
        self
    }
}

impl<B: BoundPlugin> Filter for PluginFilter<B> {
    fn name(&self) -> &'static str {
        self.bound.name()
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, cx: &Constraints<'_>) -> bool {
        match self.bound.keep(facts, candidate.position) {
            Ok(keep) => keep,
            Err(Fallback) => match &self.fallback {
                Some(fallback) => fallback.keep(facts, candidate, cx),
                None => self.bound.failure() == Failure::Open,
            },
        }
    }

    fn soft(&self) -> bool {
        self.soft
    }

    fn needs(&self) -> Needs {
        stage_needs(&*self.bound, self.fallback.as_ref().map(|f| f.needs()))
    }

    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        match self.bound.retain(facts, kept) {
            Ok(()) => {}
            Err(Fallback) => match &self.fallback {
                Some(fallback) => fallback.retain(facts, candidates, cx, kept),
                None => {
                    if self.bound.failure() == Failure::Closed {
                        kept.clear();
                    }
                }
            },
        }
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.bound.as_ref())
    }
}

/// The plugin's `score` stage. Without a fallback, a failure writes a flat
/// score, which weighs nothing in the combination.
#[derive(Debug)]
pub struct PluginScorer<B> {
    bound: Arc<B>,
    fallback: Option<Box<dyn Scorer>>,
    /// The kinds of event this adapter forwards.
    observes: EventSet,
}

impl<B: BoundPlugin> PluginScorer<B> {
    pub fn new(bound: Arc<B>, fallback: Option<Box<dyn Scorer>>) -> Self {
        bound.bound_to(Stage::Score);
        let observes = forwarded(&*bound);
        Self {
            bound,
            fallback,
            observes,
        }
    }
}

impl<B: BoundPlugin> Scorer for PluginScorer<B> {
    fn name(&self) -> &'static str {
        self.bound.name()
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        match self.bound.score(facts, kept, out) {
            Ok(band) => band,
            Err(Fallback) => match &self.fallback {
                Some(fallback) => fallback.score(facts, candidates, kept, out),
                None => {
                    out.fill(0.0);
                    TieBand::NORMALIZED
                }
            },
        }
    }

    fn needs(&self) -> Needs {
        stage_needs(&*self.bound, self.fallback.as_ref().map(|f| f.needs()))
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.bound.as_ref())
    }
}

/// The plugin's `pick` stage: the plugin names one surviving position, the
/// host ranks the alternates behind it by the combined scores. Without a
/// fallback, a failure or a declined pick ranks by the scores under
/// `failure: open` and declines (a shed naming the plugin) under `closed`.
#[derive(Debug)]
pub struct PluginPicker<B> {
    bound: Arc<B>,
    fallback: Option<Box<dyn Picker>>,
    /// The kinds of event this adapter forwards.
    observes: EventSet,
}

impl<B: BoundPlugin> PluginPicker<B> {
    pub fn new(bound: Arc<B>, fallback: Option<Box<dyn Picker>>) -> Self {
        bound.bound_to(Stage::Pick);
        let observes = forwarded(&*bound);
        Self {
            bound,
            fallback,
            observes,
        }
    }
}

impl<B: BoundPlugin> Picker for PluginPicker<B> {
    fn name(&self) -> &'static str {
        self.bound.name()
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        if let Ok(position) = self.bound.pick(facts, kept, scores) {
            if let Some(c) = candidates.get(position as usize) {
                let lead = Target {
                    position: c.position,
                    id: c.id,
                    dp_rank: c.dp_rank,
                    pair: None,
                };
                let mut ranked = Targets::new();
                Best.pick(facts, candidates, kept, scores, band, &mut ranked);
                out.push(lead);
                // By position: the rows of one worker's ranks share its id.
                for alternate in ranked.iter().filter(|t| t.position != lead.position) {
                    out.push(*alternate);
                }
                return;
            }
        }
        match &self.fallback {
            Some(fallback) => fallback.pick(facts, candidates, kept, scores, band, out),
            None => {
                if self.bound.failure() == Failure::Open {
                    Best.pick(facts, candidates, kept, scores, band, out);
                }
            }
        }
    }

    fn needs(&self) -> Needs {
        stage_needs(&*self.bound, self.fallback.as_ref().map(|f| f.needs()))
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.bound.as_ref())
    }
}

macro_rules! forward_lifecycle {
    ($ty:ident) => {
        impl<B: BoundPlugin> Lifecycle for $ty<B> {
            fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
                if self.observes.contains(EventKind::Issued) {
                    self.bound.observe_issued(lease, decision);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_decision_issued(lease, decision);
                }
            }

            fn on_dispatched(
                &self,
                lease: &Lease,
                target: &Target,
                side: Side,
                since_decision_micros: u32,
            ) {
                if self.observes.contains(EventKind::Dispatched) {
                    self.bound
                        .observe_dispatched(lease, target, side, since_decision_micros);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_dispatched(lease, target, side, since_decision_micros);
                }
            }

            fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
                if self.observes.contains(EventKind::Completed) {
                    self.bound.observe_completed(lease, outcome);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_completed(lease, outcome);
                }
            }

            fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
                if self.observes.contains(EventKind::Released) {
                    self.bound.observe_released(lease, reason);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_released(lease, reason);
                }
            }

            fn on_membership(&self, change: &Membership<'_>) {
                if self.observes.contains(EventKind::Membership) {
                    self.bound.observe_membership(change);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_membership(change);
                }
            }

            fn on_signal(&self, signal: &Signal<'_>) {
                if self.observes.contains(EventKind::Signal) {
                    self.bound.observe_signal(signal);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_signal(signal);
                }
            }

            // A plugin that hears any event keeps state by it, whichever
            // adapter of the build forwards. One that hears none declares
            // no such need: its fallback's word stands.
            fn needs_lifecycle(&self) -> bool {
                !self.bound.observes().is_empty()
                    || self
                        .fallback
                        .as_ref()
                        .is_some_and(|fallback| fallback.needs_lifecycle())
            }

            // The events the plugin observes, and its fallback's.
            fn hears(&self) -> EventSet {
                self.fallback.as_ref().map_or(self.observes, |fallback| {
                    self.observes.union(fallback.hears())
                })
            }
        }
    };
}

forward_lifecycle!(PluginScorer);
forward_lifecycle!(PluginPicker);

#[cfg(test)]
mod tests {
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

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[u32],
            out: &mut [f32],
        ) -> Result<TieBand, Fallback> {
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

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[u32],
            _: &mut [f32],
        ) -> Result<TieBand, Fallback> {
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

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
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
}
