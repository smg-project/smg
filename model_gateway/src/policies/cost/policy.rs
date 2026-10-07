//! The filter / score / pick pipeline and its result.

use std::fmt::Debug;

use super::inputs::{CandidateInputs, RequestInputs};

/// Drops candidates before scoring. Every filter must keep a candidate for it to be scored.
pub trait WorkerFilter: Send + Sync + Debug {
    fn keep(&self, request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) -> bool;
}

/// Adds to each candidate's cost. Lower is better; costs start at zero and scorers are additive,
/// so a policy can combine independent terms. A scorer sees all candidates at once so it can
/// normalise against the fleet (minimum backlog, warmest overlap).
pub trait WorkerScorer: Send + Sync + Debug {
    fn score(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &mut [f64],
    );
}

/// Turns the scored candidates into a decision. A picker that keeps its own in-flight
/// accounting (size-weighted reservations released at completion) learns about the host's final
/// dispatch and about completions through the hooks below; all default to no-ops.
pub trait WorkerPicker: Send + Sync + Debug {
    fn pick(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &[f64],
    ) -> Pick;

    /// The host dispatched `request` to `candidate`, after its own gate and credit. Not every
    /// pick becomes a dispatch, so reservations belong here rather than in `pick`.
    fn on_dispatch(&self, _request: &RequestInputs<'_>, _candidate: &CandidateInputs<'_>) {}

    /// A request on `url` finished, successfully or not.
    fn on_request_complete(&self, _url: &str) {}

    /// The router holds `in_flight` requests on `url` right now; a picker
    /// keeping per-dispatch reservations releases any beyond that count (see
    /// `LoadBalancingPolicy::reconcile_in_flight`).
    fn reconcile_in_flight(&self, _url: &str, _in_flight: usize) {}

    /// `url` left the fleet.
    fn on_worker_removed(&self, _url: &str) {}
}

/// A picker's decision, in positions of the candidate slice the policy was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pick {
    /// No candidate qualifies; the host runs its miss path (expected-wait over the fleet).
    None,
    /// One worker. The host credits it and dispatches.
    Final(usize),
    /// An affinity group the host resolves with its pressure gate and expected-wait selector;
    /// this is how the pre-policy cache-aware decision is expressed exactly.
    Group(Vec<usize>),
}

/// Inputs a policy needs the host to gather before calling it. The default policy needs none of
/// them, so its hot path gathers exactly what the pre-policy decision did.
#[derive(Debug, Clone, Copy, Default)]
pub struct Needs {
    /// Chain hashes of the prompt's blocks (`RequestInputs::prefix_hashes`).
    pub prefix_hashes: bool,
    /// Every eligible worker, not only those with a positive overlap.
    pub all_workers: bool,
}

/// A named selection policy: zero or more filters, zero or more scorers, one picker.
#[derive(Debug)]
pub struct WorkerSelectionPolicy {
    name: &'static str,
    needs: Needs,
    filters: Vec<Box<dyn WorkerFilter>>,
    scorers: Vec<Box<dyn WorkerScorer>>,
    picker: Box<dyn WorkerPicker>,
}

impl WorkerSelectionPolicy {
    pub fn new(
        name: &'static str,
        needs: Needs,
        filters: Vec<Box<dyn WorkerFilter>>,
        scorers: Vec<Box<dyn WorkerScorer>>,
        picker: Box<dyn WorkerPicker>,
    ) -> Self {
        Self {
            name,
            needs,
            filters,
            scorers,
            picker,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn needs(&self) -> Needs {
        self.needs
    }

    /// Tell the picker which candidate the host finally dispatched to.
    pub fn on_dispatch(&self, request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) {
        self.picker.on_dispatch(request, candidate);
    }

    /// Tell the picker a request on `url` completed.
    pub fn on_request_complete(&self, url: &str) {
        self.picker.on_request_complete(url);
    }

    /// Tell the picker how many requests the router holds on `url`.
    pub fn reconcile_in_flight(&self, url: &str, in_flight: usize) {
        self.picker.reconcile_in_flight(url, in_flight);
    }

    /// Tell the picker `url` left the fleet.
    pub fn on_worker_removed(&self, url: &str) {
        self.picker.on_worker_removed(url);
    }

    /// Run the pipeline. Picks are positions in `candidates` as given by the caller, whatever
    /// the filters removed.
    pub fn select(&self, request: &RequestInputs<'_>, candidates: &[CandidateInputs<'_>]) -> Pick {
        if candidates.is_empty() {
            return Pick::None;
        }
        let kept: Vec<usize> = (0..candidates.len())
            .filter(|&i| {
                self.filters
                    .iter()
                    .all(|filter| filter.keep(request, &candidates[i]))
            })
            .collect();
        if kept.is_empty() {
            return Pick::None;
        }
        let compact: Vec<CandidateInputs<'_>>;
        let view: &[CandidateInputs<'_>] = if kept.len() == candidates.len() {
            candidates
        } else {
            compact = kept.iter().map(|&i| candidates[i].clone()).collect();
            &compact
        };
        let mut costs = vec![0.0f64; view.len()];
        for scorer in &self.scorers {
            scorer.score(request, view, &mut costs);
        }
        match self.picker.pick(request, view, &costs) {
            Pick::None => Pick::None,
            Pick::Final(i) => Pick::Final(kept[i]),
            Pick::Group(group) => Pick::Group(group.into_iter().map(|i| kept[i]).collect()),
        }
    }
}
