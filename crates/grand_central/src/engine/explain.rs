//! A seat's holder counts picks and sheds before it takes the ring lock.
//! The lock holds a push or a scan of the seat's recent decisions.
//!
//! The engine in a gateway's own process counts on each calling thread and
//! keeps one ring a document (`InProcess::explained`).
//!
//! Its child modules:
//! - `json`: a kept decision as JSON, under the contract's field names.
//! - `shadow`: a binding's shadow profile next to a live decision.

#![cfg_attr(
    not(any(test, feature = "in-process")),
    expect(
        dead_code,
        reason = "Only an engine built under `in-process` keeps sampled records: a served controller's, and the one in a gateway's own process."
    )
)]

use std::{collections::VecDeque, time::Instant};

use crate::{
    document::spec::PlanName,
    engine::{
        decision::{Decision, ShadowKind},
        explain::shadow::Shadowed,
        framework::{Engine, Plan},
        view::{fingerprint, WorkerId},
    },
    store::GatewayId,
};

#[cfg(feature = "in-process")]
mod json;
pub(crate) mod shadow;

/// The maximum records a decider holds.
const SLOTS: usize = 1_024;

/// The decision a door returned, kept by gateway and request fingerprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionRecord {
    /// The decision and its trace, which carries the generation. A request
    /// that one pick of a picker decided in a gateway's own process is kept
    /// as that pick: its trace is a drawn pick's, with one target and no
    /// survivor counted.
    pub decision: Decision,
    /// The position in that generation of the binding the decision ran
    /// under, if one matched. A decode decided after its prefill ran
    /// under its pair's binding, whatever phase the call stated.
    pub binding: Option<u32>,
    /// The instant the request was taken.
    pub at: Instant,
    /// What the binding's shadow said beside the decision, where one
    /// was due.
    pub shadow: Option<ShadowKind>,
    /// The first target of the shadow's pick.
    pub shadow_target: Option<WorkerId>,
    /// What the decode side's shadow said beside a pair's decode, where
    /// one was due. A pair's prefill side reads in `shadow`.
    pub decode_shadow: Option<ShadowKind>,
    /// The first target of the decode side's shadow.
    pub decode_shadow_target: Option<WorkerId>,
    gateway: GatewayId,
    request: u64,
}

/// The names of a sampled decision, from the generation that decided.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecisionNames {
    /// The binding's plan, spelled as in the policy document.
    pub plan: &'static str,
    /// The profile named by the trace. This and the stage names are empty
    /// where the decision was made ahead of every profile, or the trace's
    /// profile is outside the binding.
    pub profile: String,
    /// The profile's picker.
    pub picker: String,
    /// The profile's filters, in their declared order.
    pub filters: Vec<String>,
    /// The profile's scorers, in their declared order.
    pub scorers: Vec<String>,
}

impl DecisionRecord {
    /// Reads names if `engine` has the recorded generation number and binding.
    #[must_use]
    pub fn names(&self, engine: &Engine) -> Option<DecisionNames> {
        let trace = self.decision.trace();
        if trace.generation != engine.generation() {
            return None;
        }
        let binding = engine.binding_at(self.binding?)?;
        let plan = match binding.plan {
            Plan::Single => PlanName::Single,
            Plan::PdPair => PlanName::PdPair,
            Plan::Epd { .. } => PlanName::Epd,
        };
        let mut names = DecisionNames {
            plan: plan.name(),
            ..DecisionNames::default()
        };
        let Some(at) = trace.made_under() else {
            return Some(names);
        };
        if let Some(profile) = engine.profile_under(binding, at) {
            names.profile.clone_from(&profile.name);
            profile.picker.name().clone_into(&mut names.picker);
            names.filters = profile
                .filters
                .iter()
                .map(|f| f.name().to_owned())
                .collect();
            names.scorers = profile
                .scorers
                .iter()
                .map(|w| w.scorer.name().to_owned())
                .collect();
        }
        Some(names)
    }
}

/// A decider's count toward its next sample, kept under one term and
/// generation. An engine that has no term counts under the number of its
/// document.
#[derive(Clone, Copy, Default)]
pub(crate) struct SampleCount {
    term: u64,
    generation: u64,
    count: u32,
}

impl SampleCount {
    /// A count that has counted nothing: what a thread's own count starts
    /// as.
    #[cfg(feature = "in-process")]
    pub(crate) const NEW: Self = Self {
        term: 0,
        generation: 0,
        count: 0,
    };

    /// Counts a pick or shed, restarting under a new term or generation.
    /// Returns true when the count reaches the sampling rate.
    pub(crate) fn keeps(&mut self, term: u64, engine: &Engine, decision: &Decision) -> bool {
        matches!(decision, Decision::Pick(_) | Decision::Shed(_)) && self.steps(term, engine)
    }

    /// Counts one pick or shed its caller holds, as [`Self::keeps`] counts
    /// one: for a pick that is counted ahead of the decision that names it.
    pub(crate) fn steps(&mut self, term: u64, engine: &Engine) -> bool {
        let generation = engine.generation();
        if self.term != term || self.generation != generation {
            *self = Self {
                term,
                generation,
                count: 0,
            };
        }
        self.count += 1;
        if self.count != engine.explain_sample() {
            return false;
        }
        self.count = 0;
        true
    }
}

/// A seat's recent decisions. The seat's holder counts separately and
/// takes the ring lock to push. A reader takes it for one scan.
#[derive(Debug, Default)]
pub(crate) struct Samples {
    records: VecDeque<DecisionRecord>,
    term: u64,
}

impl Samples {
    /// Pushes one record, allocating the ring on its first write. A new
    /// lead starts with an empty ring. `binding` is the position of the
    /// binding the decision ran under. The argument behind it is the
    /// instant the request was taken, the gateway that asked and the
    /// request's id as stated.
    #[cold]
    pub(crate) fn push(
        &mut self,
        term: u64,
        binding: Option<u32>,
        (at, gateway, id): (Instant, GatewayId, &[u8]),
        decision: &Decision,
        shadow: Option<Shadowed>,
        sides: [Option<Shadowed>; 2],
    ) {
        debug_assert!(
            matches!(decision, Decision::Pick(_) | Decision::Shed(_)),
            "a record is a pick's or a shed's: the count keeps no other"
        );
        if self.term != term {
            *self = Self {
                term,
                ..Self::default()
            };
        }
        if self.records.capacity() == 0 {
            self.records = VecDeque::with_capacity(SLOTS);
        }
        if self.records.len() == SLOTS {
            self.records.pop_front();
        }
        // A single's shadow, or a pair's by its two sides.
        let [prefill, decode] = sides;
        self.records.push_back(DecisionRecord {
            decision: *decision,
            binding,
            at,
            shadow: shadow.or(prefill).map(|said| said.kind),
            shadow_target: shadow.or(prefill).and_then(|said| said.first),
            decode_shadow: decode.map(|said| said.kind),
            decode_shadow_target: decode.and_then(|said| said.first),
            gateway,
            request: fingerprint(id),
        });
    }

    /// The last record of this key at the seat, from the lead being read.
    pub(crate) fn find(
        &self,
        term: u64,
        gateway: GatewayId,
        request: u64,
    ) -> Option<DecisionRecord> {
        if self.term != term {
            return None;
        }
        self.records
            .iter()
            .rev()
            .find(|record| record.gateway == gateway && record.request == request)
            .copied()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/explain/tests.rs"]
mod tests;
