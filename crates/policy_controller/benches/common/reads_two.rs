//! The scorer the signal benches register: it reads two signals of every row.
use smg_policy_controller::{
    document::Needs,
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    load::SignalColumn,
    stages::Lifecycle,
};

/// Scores a row by the two values it reads: the less of both, the better.
#[derive(Debug)]
pub struct ReadsTwo(pub SignalColumn, pub SignalColumn);

impl Lifecycle for ReadsTwo {}

impl Scorer for ReadsTwo {
    fn name(&self) -> &'static str {
        "reads-two"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&index, score) in kept.iter().zip(out) {
            let row = &candidates[index as usize];
            let used = self.0.read(facts, row).unwrap_or(0.0);
            let pending = self.1.read(facts, row).unwrap_or(0.0);
            *score = 1.0 / (1.0 + (used + pending) as f32);
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs::NONE.with_signal_cells()
    }
}
