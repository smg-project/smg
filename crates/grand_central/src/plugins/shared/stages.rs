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
#[path = "../../../tests/unit/plugins/shared/stages/tests.rs"]
mod tests;
