//! The cache-aware decision as it was before the policy layer, expressed as a policy.
//!
//! Only workers with a positive effective (decayed) overlap are candidates; the picker returns
//! the top effective-score group: the exact maximum at temperature zero, or the softmax-sampled
//! score group otherwise. The host then resolves that group exactly as it always has: the
//! per-request pressure gate, then LeastLoad's expected-wait selection and credit. An empty group
//! is the miss path. Decisions are therefore identical to the pre-policy code; the tests in
//! `cache_aware.rs` compare the two at zero temperature and under a temperature.

use super::{
    inputs::{CandidateInputs, RequestInputs},
    policy::{Needs, Pick, WorkerFilter, WorkerPicker, WorkerScorer, WorkerSelectionPolicy},
    softmax::sample_by_score_temperature,
};

pub const POLICY_NAME: &str = "cache-aware-default";

#[derive(Debug)]
struct PositiveOverlap;

impl WorkerFilter for PositiveOverlap {
    fn keep(&self, _request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) -> bool {
        candidate.effective_score > 0.0
    }
}

/// Cost is the negated affinity, so the generic "lowest cost" reading of the costs agrees with
/// the picker below (which ranks on the score directly to keep the draw byte-identical).
#[derive(Debug)]
struct NegatedAffinity;

impl WorkerScorer for NegatedAffinity {
    fn score(
        &self,
        _request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &mut [f64],
    ) {
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost += -candidate.effective_score;
        }
    }
}

#[derive(Debug)]
struct AffinityGroupPicker {
    temperature: f32,
}

impl WorkerPicker for AffinityGroupPicker {
    fn pick(
        &self,
        _request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        _costs: &[f64],
    ) -> Pick {
        if candidates.is_empty() {
            return Pick::None;
        }
        let scores: Vec<f64> = candidates.iter().map(|c| c.effective_score).collect();
        let selected = if self.temperature > 0.0 {
            sample_by_score_temperature(&scores, self.temperature)
        } else {
            // `max_by` keeps the last of equal maxima, as the pre-policy code did; the group
            // expansion below makes which one immaterial.
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
        };
        let Some(selected) = selected else {
            return Pick::None;
        };
        let selected_score = scores[selected];
        Pick::Group(
            (0..scores.len())
                .filter(|&i| scores[i] == selected_score)
                .collect(),
        )
    }
}

pub(super) fn policy(temperature: f32) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        POLICY_NAME,
        Needs::default(),
        vec![Box::new(PositiveOverlap)],
        vec![Box::new(NegatedAffinity)],
        Box::new(AffinityGroupPicker { temperature }),
    )
}
