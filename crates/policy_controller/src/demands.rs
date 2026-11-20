//! A request's hard demands: what it must have of the worker that takes it.
//!
//! A hard demand is the request's ([`Demands`]) and its fact is the row's
//! ([`DemandFacts`]). The door keeps the demands before any stage runs, in
//! the pass that keeps its gateway's wires: a row that fails one is no
//! candidate, and no stage, no rule and no plugin is shown it. No document
//! writes a filter for a demand, so none can leave one out.
//!
//! A worker's facts are stated by its registration
//! ([`WorkerFacts::demand_facts`](crate::store::WorkerFacts::demand_facts))
//! and stand in a column beside its pool's rows, one entry a row
//! ([`World::demand_facts`](crate::store::World::demand_facts)): the
//! candidate row carries none of them. The crate compares what the two
//! sides state and derives nothing: it names no label and no runtime.
//!
//! A request that demands nothing reads no column and pays one test.

/// What a request demands of the worker that takes it. The default demands
/// nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demands {
    /// The runtime the request stays on: a retry states the number its
    /// first attempt's worker registered ([`DemandFacts::runtime`]), and
    /// only a worker that registered the same number takes it. `0` demands
    /// none.
    pub runtime: u32,
}

impl Demands {
    /// Whether the request demands nothing: its door reads no fact then.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.runtime == 0
    }

    /// Whether a worker that states `facts` meets every demand.
    #[must_use]
    pub const fn met_by(self, facts: DemandFacts) -> bool {
        self.runtime == 0 || self.runtime == facts.runtime
    }
}

/// What a worker states for the hard demands of a request, by its
/// registration: one entry a row, in a column beside the rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DemandFacts {
    /// The worker's runtime as a number its registering host names: equal
    /// numbers are one runtime, and the crate reads no more of it. `0`
    /// states none, and meets no demand for one.
    pub runtime: u32,
}

#[cfg(test)]
mod tests {
    use super::{DemandFacts, Demands};

    /// A request that demands nothing is met by every worker, and one that
    /// demands a runtime by the workers that registered its number alone: a
    /// worker that states no runtime meets no demand for one.
    #[test]
    fn a_runtime_demand_is_met_by_the_same_number_alone() {
        let (first, other, none) = (
            DemandFacts { runtime: 7 },
            DemandFacts { runtime: 8 },
            DemandFacts::default(),
        );
        let nothing = Demands::default();
        assert!(nothing.is_none());
        assert!([first, other, none]
            .iter()
            .all(|facts| nothing.met_by(*facts)));
        let stays = Demands { runtime: 7 };
        assert!(!stays.is_none());
        assert_eq!(
            [first, other, none].map(|facts| stays.met_by(facts)),
            [true, false, false]
        );
    }
}
