//! A bound plugin as the framework's stages: a filter, a scorer and a
//! picker, each with the fallback stage its row names. The three adapters of
//! one profile share the bound plugin's decision (their [`DecisionScope`]);
//! the first scorer or picker made in an engine build, whichever of the
//! build's profiles binds the plugin, forwards the lifecycle, so an event
//! reaches the plugin once per engine. A fallback is a built-in stage: the
//! engine opens the scopes of a profile's own stages, not a fallback's. The
//! scorer declares the plugin's needs (or its fallback's), which the engine's
//! summary folds in; the framework's filters and pickers declare none, so a
//! binding without a scorer leaves them to the host ([`super::Plugin::needs`]).

use std::sync::Arc;

use super::{
    config::Failure,
    plugin::{Bound, Fallback},
};
use crate::{
    builtins::Best,
    decision::{Decision, Target, Targets},
    framework::{DecisionScope, Filter, Picker, Scorer, TieBand},
    lease::{Lease, Lifecycle, Membership, Outcome, ReleaseReason, Side, Signal},
    spec::Needs,
    view::{Candidate, Constraints, RequestFacts},
};

/// The plugin's `filter` stage. Without a fallback, a failure keeps every
/// row under `failure: open` and empties the pool under `closed`. A filter
/// is not a lifecycle stage of the framework (the engine forwards the
/// lifecycle to scorers and pickers), so a filter-only binding observes
/// nothing.
#[derive(Debug)]
pub struct WasmFilter {
    bound: Arc<Bound>,
    fallback: Option<Box<dyn Filter>>,
    soft: bool,
}

impl WasmFilter {
    pub fn new(bound: Arc<Bound>, fallback: Option<Box<dyn Filter>>) -> Self {
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

impl Filter for WasmFilter {
    fn name(&self) -> &'static str {
        self.bound.plugin().name()
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, cx: &Constraints<'_>) -> bool {
        match self.bound.keep(facts, candidate.position) {
            Ok(keep) => keep,
            Err(Fallback) => match &self.fallback {
                Some(fallback) => fallback.keep(facts, candidate, cx),
                None => self.bound.plugin().failure() == Failure::Open,
            },
        }
    }

    fn soft(&self) -> bool {
        self.soft
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
                    if self.bound.plugin().failure() == Failure::Closed {
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
pub struct WasmScorer {
    bound: Arc<Bound>,
    fallback: Option<Box<dyn Scorer>>,
    observes: bool,
}

impl WasmScorer {
    pub fn new(bound: Arc<Bound>, fallback: Option<Box<dyn Scorer>>) -> Self {
        let observes = bound.take_observer();
        Self {
            bound,
            fallback,
            observes,
        }
    }
}

impl Scorer for WasmScorer {
    fn name(&self) -> &'static str {
        self.bound.plugin().name()
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        match self.bound.score(facts, kept, out) {
            Ok(()) => TieBand::NORMALIZED,
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
        let own = self.bound.plugin().needs();
        match &self.fallback {
            Some(fallback) => own.or(fallback.needs()),
            None => own,
        }
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.bound.as_ref())
    }
}

/// The plugin's `pick` stage: the guest names one surviving position, the
/// host ranks the alternates behind it by the combined scores. Without a
/// fallback, a failure or a declined pick ranks by the scores under
/// `failure: open` and declines (a shed naming the plugin) under `closed`.
#[derive(Debug)]
pub struct WasmPicker {
    bound: Arc<Bound>,
    fallback: Option<Box<dyn Picker>>,
    observes: bool,
}

impl WasmPicker {
    pub fn new(bound: Arc<Bound>, fallback: Option<Box<dyn Picker>>) -> Self {
        let observes = bound.take_observer();
        Self {
            bound,
            fallback,
            observes,
        }
    }
}

impl Picker for WasmPicker {
    fn name(&self) -> &'static str {
        self.bound.plugin().name()
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
                if self.bound.plugin().failure() == Failure::Open {
                    Best.pick(facts, candidates, kept, scores, band, out);
                }
            }
        }
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self.bound.as_ref())
    }
}

macro_rules! forward_lifecycle {
    ($ty:ident) => {
        impl Lifecycle for $ty {
            fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
                if self.observes {
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
                if self.observes {
                    self.bound
                        .observe_dispatched(lease, target, side, since_decision_micros);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_dispatched(lease, target, side, since_decision_micros);
                }
            }

            fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
                if self.observes {
                    self.bound.observe_completed(lease, outcome);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_completed(lease, outcome);
                }
            }

            fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
                if self.observes {
                    self.bound.observe_released(lease, reason);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_released(lease, reason);
                }
            }

            fn on_membership(&self, change: &Membership<'_>) {
                if self.observes {
                    self.bound.observe_membership(change);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_membership(change);
                }
            }

            fn on_signal(&self, signal: &Signal) {
                if self.observes {
                    self.bound.observe_signal(signal);
                }
                if let Some(fallback) = &self.fallback {
                    fallback.on_signal(signal);
                }
            }
        }
    };
}

forward_lifecycle!(WasmScorer);
forward_lifecycle!(WasmPicker);
