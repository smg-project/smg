//! A request's hard demands: what it must have of the worker that takes it.
//!
//! A hard demand is the request's ([`Demands`]) and its fact is the row's
//! ([`DemandFacts`]). The door keeps the demands before any stage runs, in
//! the pass that keeps its gateway's wires: a row that fails one is no
//! candidate, and no stage, no rule and no plugin is shown it. No document
//! writes a filter for a demand, so none can leave one out.
//!
//! A worker's facts are stated by its registration. Whether it takes images
//! is the bit its row carries
//! ([`WorkerFacts::vision`](crate::store::WorkerFacts::vision)), the one
//! fact a pairing plan reads too. The rest
//! ([`WorkerFacts::demand_facts`](crate::store::WorkerFacts::demand_facts))
//! stand in a column beside its pool's rows, one entry a row
//! ([`World::demand_facts`](crate::store::World::demand_facts)): the
//! candidate row carries none of them. The crate compares what the two
//! sides state and derives nothing: it names no label and no runtime.
//!
//! A request that demands nothing reads no column and pays one test.
//!
//! A pool the demands leave nobody is shed with no endpoint, under the
//! stage of one demand ([`Unmet::stage`]). A row is judged by the demand it
//! fails first, in the order a host answers for them ([`Unmet`]), and the
//! pool is shed under the last of those: where a host cuts a pool one
//! demand after another, that is the demand that took its last worker.

/// Where a request's media by reference may be processed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum References {
    /// The request carries no media by reference.
    #[default]
    None,
    /// By a worker that processes media by reference, whichever pipeline
    /// it runs.
    Any,
    /// By a worker that processes it by its registering host's own
    /// pipeline, and by no other.
    Host,
    /// By a worker that processes it by a pipeline of its own, and by no
    /// other.
    Worker,
}

/// What a request demands of the worker that takes it. The default demands
/// nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demands {
    /// The runtime the request stays on: a retry states the number its
    /// first attempt's worker registered ([`DemandFacts::runtime`]), and
    /// only a worker that registered the same number takes it. `0` demands
    /// none.
    pub runtime: u32,
    /// Whether the request needs vision: its media holds pixels, and only a
    /// worker that takes images
    /// ([`WorkerFacts::vision`](crate::store::WorkerFacts::vision)) takes
    /// the request.
    pub vision: bool,
    /// Where the request's media by reference may be processed.
    pub references: References,
}

impl Demands {
    /// Whether the request demands nothing: its door reads no fact then.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.runtime == 0 && !self.vision && matches!(self.references, References::None)
    }

    /// The demand a worker fails first, in the order of [`Unmet`]: `vision`
    /// is whether it takes images, the bit its row carries, and `facts`
    /// what it states beside its row. `None` where it meets every demand.
    #[must_use]
    pub const fn unmet_by(self, vision: bool, facts: DemandFacts) -> Option<Unmet> {
        if self.runtime != 0 && self.runtime != facts.runtime {
            return Some(Unmet::Runtime);
        }
        if self.vision && !vision {
            return Some(Unmet::Vision);
        }
        let host = match self.references {
            References::None => return None,
            References::Any => facts.host_pipeline,
            References::Host => true,
            References::Worker => false,
        };
        if !facts.references {
            Some(Unmet::References)
        } else if facts.host_pipeline != host {
            Some(Unmet::Pipeline)
        } else {
            None
        }
    }

    /// Whether a worker meets every demand ([`Self::unmet_by`]).
    #[must_use]
    pub const fn met_by(self, vision: bool, facts: DemandFacts) -> bool {
        self.unmet_by(vision, facts).is_none()
    }
}

/// What a worker states for the hard demands of a request beside its row,
/// by its registration: one entry a row, in a column beside the rows. The
/// default states nothing, and meets no demand that reads it. Whether the
/// worker takes images is no entry of the column: its row carries that bit
/// ([`WorkerFacts::vision`](crate::store::WorkerFacts::vision)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DemandFacts {
    /// The worker's runtime as a number its registering host names: equal
    /// numbers are one runtime, and the crate reads no more of it. `0`
    /// states none, and meets no demand for one.
    pub runtime: u32,
    /// Whether the worker processes media by reference.
    pub references: bool,
    /// Whether it processes that media by its registering host's own
    /// pipeline, and not by one of its own. Read only beside
    /// [`Self::references`].
    pub host_pipeline: bool,
}

/// The stage a pool is shed under when none of its rows processes media by
/// reference ([`Unmet::References`]).
pub(crate) const REFERENCES: &str = "references";

/// The stage a pool is shed under when its rows that process media by
/// reference all run the other pipeline than the request names
/// ([`Unmet::Pipeline`]).
pub(crate) const PIPELINE: &str = "pipeline";

/// The demand a worker fails first. The order is the order a host answers
/// for them: a pool is cut to its request's runtime, then to the workers
/// that take images, then to those that process media by reference, then
/// to the pipeline the request names. Further demands add their cases: a
/// host's match keeps an arm for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Unmet {
    /// Another runtime than the request stays on, or none stated.
    Runtime,
    /// The worker takes no images.
    Vision,
    /// The worker processes no media by reference.
    References,
    /// The worker processes media by reference by the other pipeline.
    Pipeline,
}

impl Unmet {
    /// The stage a pool this demand left nobody is shed under. A runtime is
    /// the rest of a request's wire, and is shed as its gateway's wires
    /// are. Vision is shed under the word of the `epd` plan's vision rule.
    /// An encode pair's decision names two stages of its own, by the side
    /// that lacks it.
    #[must_use]
    pub const fn stage(self) -> &'static str {
        match self {
            Self::Runtime => crate::door::WIRES,
            Self::Vision => crate::framework::VISION,
            Self::References => REFERENCES,
            Self::Pipeline => PIPELINE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DemandFacts, Demands, References, Unmet};

    /// A request that demands nothing is met by every worker, and one that
    /// demands a runtime by the workers that registered its number alone: a
    /// worker that states no runtime meets no demand for one.
    #[test]
    fn a_runtime_demand_is_met_by_the_same_number_alone() {
        let of = |runtime: u32| DemandFacts {
            runtime,
            ..DemandFacts::default()
        };
        let (first, other, none) = (of(7), of(8), DemandFacts::default());
        let nothing = Demands::default();
        assert!(nothing.is_none());
        assert!([first, other, none]
            .iter()
            .all(|facts| nothing.met_by(false, *facts)));
        let stays = Demands {
            runtime: 7,
            ..Demands::default()
        };
        assert!(!stays.is_none());
        assert_eq!(
            [first, other, none].map(|facts| stays.met_by(false, facts)),
            [true, false, false]
        );
    }

    /// Each media demand is a demand, and a worker is judged by the one it
    /// fails first: its runtime, then vision, then media by reference, then
    /// the pipeline that processes it. Vision is judged by the bit of the
    /// worker's row, and the rest by what it states beside it.
    #[test]
    fn a_worker_is_judged_by_the_demand_it_fails_first() {
        let worker = |vision: bool, references: bool, host_pipeline: bool| {
            let facts = DemandFacts {
                runtime: 7,
                references,
                host_pipeline,
            };
            (vision, facts)
        };
        let asks = |vision: bool, references: References| Demands {
            runtime: 7,
            vision,
            references,
        };
        for demands in [
            asks(true, References::None),
            asks(false, References::Any),
            asks(false, References::Host),
            asks(false, References::Worker),
        ] {
            assert!(!demands.is_none(), "{demands:?}");
        }
        let (plain, host, own) = (
            worker(true, false, false),
            worker(true, true, true),
            worker(false, true, false),
        );
        let table = [
            (asks(true, References::None), plain, None),
            (asks(true, References::None), own, Some(Unmet::Vision)),
            (asks(false, References::Any), plain, Some(Unmet::References)),
            (asks(false, References::Any), host, None),
            (asks(false, References::Any), own, None),
            (asks(false, References::Host), host, None),
            (asks(false, References::Host), own, Some(Unmet::Pipeline)),
            (asks(false, References::Worker), host, Some(Unmet::Pipeline)),
            (asks(false, References::Worker), own, None),
            (
                asks(false, References::Worker),
                plain,
                Some(Unmet::References),
            ),
            // Vision is judged before media by reference.
            (asks(true, References::Any), own, Some(Unmet::Vision)),
            (asks(true, References::Host), host, None),
        ];
        for (demands, (vision, facts), unmet) in table {
            let judged = demands.unmet_by(vision, facts);
            assert_eq!(judged, unmet, "{demands:?} {vision} {facts:?}");
            assert_eq!(demands.met_by(vision, facts), unmet.is_none());
        }
        // The runtime is judged before any of them.
        let elsewhere = DemandFacts {
            runtime: 8,
            ..DemandFacts::default()
        };
        let every = asks(true, References::Host);
        assert_eq!(every.unmet_by(true, elsewhere), Some(Unmet::Runtime));
    }

    /// The demands stand in the order a host answers for them, and each has
    /// the stage its pool is shed under.
    #[test]
    fn the_demands_stand_in_a_hosts_order_each_with_its_stage() {
        let order = [
            Unmet::Runtime,
            Unmet::Vision,
            Unmet::References,
            Unmet::Pipeline,
        ];
        assert!(order.is_sorted());
        assert_eq!(
            order.map(Unmet::stage),
            ["wires", "vision", "references", "pipeline"]
        );
    }
}
