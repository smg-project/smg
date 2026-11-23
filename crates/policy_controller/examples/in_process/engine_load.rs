//! A stage of the program's own, added in code: a scorer that weighs a
//! data-parallel engine as a whole.
//!
//! A stage is handed one record for each rank of such an engine, and each
//! record states which engine of its list it is a rank of
//! ([`Candidate::engine`]). [`EngineLoad`] adds up the requests in flight
//! over the ranks of each engine and scores highest the ranks of the engine
//! that holds the fewest. A request then goes to an idle engine before it
//! goes to an idle rank of a busy one.
//!
//! The sums count every rank the stage is handed, the ones a filter dropped
//! too: a rank that is refused for its load still loads its engine.
//!
//! A program adds the stage to the catalog it validates its document over
//! ([`stages`]) and names it in a profile's `score` list
//! ([`decided_by_engine_load`]).

use std::cell::RefCell;

use smg_policy_controller::{
    document::{PickSpec, PolicySpec, ScoreSpec},
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    stages::{Lifecycle, StageRegistry, StageRegistryError},
};

/// The name a document's `score` list names the scorer by.
pub const ENGINE_LOAD: &str = "engine-load";

/// Scores each record by the requests in flight on its whole engine: `1`
/// where the engine holds none, `0` for the engine that holds the most
/// among the engines of the records it scores.
#[derive(Debug)]
pub struct EngineLoad;

thread_local! {
    /// The sums of the decision under way, one an engine. A thread keeps
    /// them between its decisions, so a decision allocates nothing once
    /// its thread has decided over as many engines.
    static HELD: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

impl Lifecycle for EngineLoad {}

impl Scorer for EngineLoad {
    fn name(&self) -> &'static str {
        ENGINE_LOAD
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let engine_of = |row: u32| candidates[row as usize].engine as usize;
        HELD.with_borrow_mut(|held| {
            // An engine's place is below the count of its list's workers,
            // so the sums are as many at most.
            let engines = candidates.iter().map(|row| row.engine as usize + 1);
            held.clear();
            held.resize(engines.max().unwrap_or(0), 0);
            // Every rank handed counts, kept or not.
            for row in candidates {
                held[row.engine as usize] += facts.in_flight(row);
            }
            let most = kept.iter().map(|&row| held[engine_of(row)]).max();
            let most = most.unwrap_or(0).max(1);
            for (&row, score) in kept.iter().zip(out) {
                *score = 1.0 - held[engine_of(row)] as f32 / most as f32;
            }
        });
        TieBand::NORMALIZED
    }
}

/// The catalog a program validates its document over: the built-in stages
/// and [`EngineLoad`].
///
/// # Errors
///
/// The catalog holds the scorer's name already.
pub fn stages() -> Result<StageRegistry, StageRegistryError> {
    let mut stages = StageRegistry::builtin();
    stages.scorer(ENGINE_LOAD, || Box::new(EngineLoad))?;
    Ok(stages)
}

/// Writes the scorer into every profile of `document`, under the picker
/// that takes the best score. The profiles' filters stay.
pub fn decided_by_engine_load(document: &mut PolicySpec) {
    for profile in document.profiles.values_mut() {
        profile.score.push(ScoreSpec::from(ENGINE_LOAD));
        profile.pick = PickSpec::from("best");
    }
}
