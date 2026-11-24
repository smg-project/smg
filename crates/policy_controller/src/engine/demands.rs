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
//! stand in a column beside its pool's rows, one entry a row, which a pool
//! keeps where a worker of it states one: the candidate row carries none of
//! them. The crate compares what the two sides state and derives nothing: it
//! names no label and no runtime.
//!
//! A request that demands nothing reads no column and pays one test.
//!
//! A request's context window is two bounds: what its input takes, and
//! what its completion may ([`Demands::input_window`],
//! [`Demands::output_window`]). Each is held against a worker's window
//! alone ([`DemandFacts::window`]), and the two are not added. A worker
//! that states no window holds a request of any size.
//!
//! A retry whose constraint was decided for its first worker says how that
//! worker decides one, as a number ([`Demands::decoding`]), and only a
//! worker that states the same number takes it
//! ([`DemandFacts::decoding`]). The numbers are the registering host's:
//! the crate holds no rule for them.
//!
//! A pool the demands leave nobody is shed with no endpoint, under the
//! stage of one demand ([`UnmetDemand::stage`]). A row is judged by the demand
//! it fails first, in the order a host answers for them ([`UnmetDemand`]), and
//! the pool is shed under the last of those: where a host cuts a pool one
//! demand after another, that is the demand that took its last worker.

use crate::engine::{decision, framework, view::Role};

/// Where a request's media by reference may be processed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum References {
    /// The request carries no media by reference.
    #[default]
    None,
    /// By a worker that processes media by reference, whichever pipeline
    /// it runs.
    Any,
    /// By a worker that processes it by the pipeline of the caller that
    /// registered it, and by no other.
    Caller,
    /// By a worker that processes it by a pipeline of its own, and by no
    /// other.
    Worker,
}

/// What a request demands of the worker that takes it. The default demands
/// nothing. Further demands add their fields: a host writes the rest of the
/// type as `..Demands::default()`.
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
    /// The context window the request's input needs, in tokens: the count
    /// of its prompt, and one more where the request scores the position
    /// behind it. Only a worker whose window is as wide
    /// ([`DemandFacts::window`]), or that states none, takes the request.
    /// `0` demands none.
    pub input_window: u32,
    /// The context window the request's completion needs, in tokens: its
    /// budget. It is held against a worker's window alone, as the input
    /// is. A prefill worker is not held to it ([`Self::held_by`]): a host
    /// states it whatever the request's phase. `0` demands none.
    pub output_window: u32,
    /// How the request's constraint was decided: a retry states the number
    /// of its first attempt's worker ([`DemandFacts::decoding`]), and only
    /// a worker that states the same number takes it. A prefill worker is
    /// not held to it ([`Self::held_by`]). `0` demands none.
    pub decoding: u8,
}

impl Demands {
    /// Whether the request demands nothing: its door reads no fact then.
    #[must_use]
    pub const fn is_none(self) -> bool {
        // Every field by its name and no rest: a demand added later does
        // not build until this test reads it.
        let Self {
            runtime,
            vision,
            references,
            input_window,
            output_window,
            decoding,
        } = self;
        // One compare for the numbers, one for the bit and the byte.
        (runtime | input_window | output_window) == 0
            && (vision as u8 | decoding) == 0
            && matches!(references, References::None)
    }

    /// The demand a worker fails first, in the order of [`UnmetDemand`]:
    /// `vision` is whether it takes images, the bit its row carries, and
    /// `facts` what it states beside its row. `None` where it meets every
    /// demand.
    #[must_use]
    pub const fn unmet_by(self, vision: bool, facts: DemandFacts) -> Option<UnmetDemand> {
        if self.runtime != 0 && self.runtime != facts.runtime {
            return Some(UnmetDemand::Runtime);
        }
        if self.vision && !vision {
            return Some(UnmetDemand::Vision);
        }
        let host = match self.references {
            References::None => None,
            References::Any => Some(facts.host_pipeline),
            References::Caller => Some(true),
            References::Worker => Some(false),
        };
        if let Some(host) = host {
            if !facts.references {
                return Some(UnmetDemand::References);
            }
            if facts.host_pipeline != host {
                return Some(UnmetDemand::Pipeline);
            }
        }
        // A window that is stated holds each bound alone.
        let held = facts.window == 0
            || (self.input_window <= facts.window && self.output_window <= facts.window);
        if !held {
            return Some(UnmetDemand::Window);
        }
        if self.decoding != 0 && self.decoding != facts.decoding {
            return Some(UnmetDemand::Decoding);
        }
        None
    }

    /// What a worker of `role`'s pool is held to: the request's demands,
    /// less its completion's bound and the way its constraint was decided
    /// for a prefill worker. A prefill runs the input: the decode beside it
    /// holds the budget and generates the tokens a constraint holds. The
    /// door reads a request's demands through this where it judges rows:
    /// the role is the door's to know.
    #[must_use]
    pub const fn held_by(self, role: Role) -> Self {
        match role {
            Role::Prefill => Self {
                output_window: 0,
                decoding: 0,
                ..self
            },
            Role::Single | Role::Decode | Role::Encode => self,
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
/// default states nothing: it meets no demand for a runtime, for media by
/// reference or for a way to decide a constraint, and it holds a request of
/// any size.
/// Whether the worker takes
/// images is no entry of the column: its row carries that bit
/// ([`WorkerFacts::vision`](crate::store::WorkerFacts::vision)). Further
/// demands add their fields: a host writes the rest of the type as
/// `..DemandFacts::default()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DemandFacts {
    /// The worker's runtime as a number its registering host names: equal
    /// numbers are one runtime, and the crate reads no more of it. A host
    /// may give one number to each runtime and transport it dials: a
    /// retry's demand ([`Demands::runtime`]) then holds the request to
    /// both. `0` states none, and meets no demand for one.
    pub runtime: u32,
    /// Whether the worker processes media by reference.
    pub references: bool,
    /// Whether it processes that media by its registering host's own
    /// pipeline, and not by one of its own. Read only beside
    /// [`Self::references`].
    pub host_pipeline: bool,
    /// The worker's context window in tokens, where its registering host
    /// holds a request to it. `0` states none, and the worker takes a
    /// request of any size.
    pub window: u32,
    /// How the worker decides a request's constraint, as a number its
    /// registering host names: equal numbers decide a constraint alike,
    /// and the crate reads no more of it. `0` states none, and meets no
    /// demand for one.
    pub decoding: u8,
}

/// The stage a pool is shed under when none of its rows processes media by
/// reference ([`UnmetDemand::References`]).
pub(crate) const REFERENCES: &str = "references";

/// The stage a pool is shed under when its rows that process media by
/// reference all run the other pipeline than the request names
/// ([`UnmetDemand::Pipeline`]).
pub(crate) const PIPELINE: &str = "pipeline";

/// The stage a pool is shed under when each of its rows states a context
/// window that holds less than the request needs ([`UnmetDemand::Window`]).
pub(crate) const CONTEXT_WINDOW: &str = "context-window";

/// The stage a pool is shed under when none of its rows decides a
/// constraint as the request's was decided ([`UnmetDemand::Decoding`]).
pub(crate) const DECODING: &str = "decoding";

/// The demand a worker fails first. The order is the order a host answers
/// for them: a pool is cut to its request's runtime, then to the workers
/// that take images, then to those that process media by reference, then
/// to the pipeline the request names, then to the windows that hold the
/// request, then to the workers that decide its constraint as it was
/// decided. Further demands add their cases: a host's match keeps an arm
/// for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum UnmetDemand {
    /// Another runtime than the request stays on, or none stated.
    Runtime,
    /// The worker takes no images.
    Vision,
    /// The worker processes no media by reference.
    References,
    /// The worker processes media by reference by the other pipeline.
    Pipeline,
    /// The worker's context window holds less than the request's input or
    /// its completion needs.
    Window,
    /// The worker decides a constraint another way than the request's was
    /// decided, or states no way.
    Decoding,
}

impl UnmetDemand {
    /// The stage a pool this demand left nobody is shed under. A runtime is
    /// the rest of a request's wire, and is shed as its gateway's wires
    /// are. Vision is shed under the word of the `epd` plan's vision rule.
    /// An encode pair's decision names two stages of its own, by the side
    /// that lacks it.
    #[must_use]
    pub const fn stage(self) -> &'static str {
        match self {
            Self::Runtime => decision::WIRES,
            Self::Vision => framework::VISION,
            Self::References => REFERENCES,
            Self::Pipeline => PIPELINE,
            Self::Window => CONTEXT_WINDOW,
            Self::Decoding => DECODING,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DemandFacts, Demands, References, Role, UnmetDemand};

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
                ..DemandFacts::default()
            };
            (vision, facts)
        };
        let asks = |vision: bool, references: References| Demands {
            runtime: 7,
            vision,
            references,
            ..Demands::default()
        };
        for demands in [
            asks(true, References::None),
            asks(false, References::Any),
            asks(false, References::Caller),
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
            (asks(true, References::None), own, Some(UnmetDemand::Vision)),
            (
                asks(false, References::Any),
                plain,
                Some(UnmetDemand::References),
            ),
            (asks(false, References::Any), host, None),
            (asks(false, References::Any), own, None),
            (asks(false, References::Caller), host, None),
            (
                asks(false, References::Caller),
                own,
                Some(UnmetDemand::Pipeline),
            ),
            (
                asks(false, References::Worker),
                host,
                Some(UnmetDemand::Pipeline),
            ),
            (asks(false, References::Worker), own, None),
            (
                asks(false, References::Worker),
                plain,
                Some(UnmetDemand::References),
            ),
            // Vision is judged before media by reference.
            (asks(true, References::Any), own, Some(UnmetDemand::Vision)),
            (asks(true, References::Caller), host, None),
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
        let every = asks(true, References::Caller);
        assert_eq!(every.unmet_by(true, elsewhere), Some(UnmetDemand::Runtime));
    }

    /// A window demand is two bounds, the input's and the completion's, and
    /// a worker's window holds each alone: the two are not added. A worker
    /// that states no window holds a request of any size. The window is
    /// judged last, behind the pipeline of media by reference.
    #[test]
    fn a_window_holds_each_bound_alone_and_none_stated_holds_any_request() {
        let window = |window: u32| DemandFacts {
            window,
            ..DemandFacts::default()
        };
        let asks = |input_window: u32, output_window: u32| Demands {
            input_window,
            output_window,
            ..Demands::default()
        };
        assert!(asks(0, 0).is_none());
        assert!(!asks(1, 0).is_none() && !asks(0, 1).is_none());
        // The request's two bounds, the worker's window, whether it holds.
        let table = [
            ((100, 0), 100, true),
            ((101, 0), 100, false),
            ((0, 100), 100, true),
            ((0, 101), 100, false),
            ((100, 100), 100, true),
            ((60, 60), 100, true),
            ((150, 150), 200, true),
            ((150, 150), 100, false),
            ((u32::MAX, u32::MAX), 0, true),
        ];
        for ((input, output), stated, held) in table {
            let unmet = asks(input, output).unmet_by(false, window(stated));
            let judged = if held {
                None
            } else {
                Some(UnmetDemand::Window)
            };
            assert_eq!(unmet, judged, "{input} {output} {stated}");
        }
        let both = Demands {
            references: References::Caller,
            ..asks(101, 0)
        };
        let own = DemandFacts {
            references: true,
            ..window(100)
        };
        assert_eq!(both.unmet_by(false, own), Some(UnmetDemand::Pipeline));
        let host = DemandFacts {
            host_pipeline: true,
            ..own
        };
        assert_eq!(both.unmet_by(false, host), Some(UnmetDemand::Window));
    }

    /// A pool is shed under the largest of the demands its workers fail
    /// first. A worker of another runtime that holds the request, beside
    /// one of the request's runtime whose window does not: the pool is
    /// shed under the window.
    #[test]
    fn a_pool_is_shed_under_the_largest_demand_its_workers_fail_first() {
        let long = Demands {
            runtime: 7,
            input_window: 101,
            ..Demands::default()
        };
        let elsewhere = DemandFacts {
            runtime: 8,
            ..DemandFacts::default()
        };
        let narrow = DemandFacts {
            runtime: 7,
            window: 100,
            ..DemandFacts::default()
        };
        let mix = [
            long.unmet_by(false, elsewhere),
            long.unmet_by(false, narrow),
        ];
        assert_eq!(mix, [Some(UnmetDemand::Runtime), Some(UnmetDemand::Window)]);
        assert_eq!(mix.into_iter().max(), Some(Some(UnmetDemand::Window)));
    }

    /// A retry's constraint is met by the workers that state its number
    /// alone: a worker that states another way, or none, meets no demand
    /// for one. It is judged last, behind the window.
    #[test]
    fn a_decided_constraint_is_met_by_the_same_number_alone() {
        let decides = |decoding: u8| DemandFacts {
            decoding,
            ..DemandFacts::default()
        };
        let retry = Demands {
            decoding: 5,
            ..Demands::default()
        };
        assert!(!retry.is_none());
        assert_eq!(
            [decides(5), decides(7), decides(0)].map(|facts| retry.unmet_by(false, facts)),
            [
                None,
                Some(UnmetDemand::Decoding),
                Some(UnmetDemand::Decoding)
            ]
        );
        assert!(Demands::default().met_by(false, decides(7)));
        let long = Demands {
            input_window: 101,
            ..retry
        };
        let narrow = DemandFacts {
            window: 100,
            ..decides(7)
        };
        assert_eq!(long.unmet_by(false, narrow), Some(UnmetDemand::Window));
    }

    /// The test of a request that demands nothing reads every field: the
    /// default demands nothing, and a request that states one demand alone
    /// demands something, whichever it is.
    #[test]
    fn a_request_that_states_one_demand_alone_demands_something() {
        let nothing = Demands::default();
        assert!(nothing.is_none());
        let mut alone = [nothing; 8];
        alone[0].runtime = 1;
        alone[1].vision = true;
        alone[2].references = References::Any;
        alone[3].references = References::Caller;
        alone[4].references = References::Worker;
        alone[5].input_window = 1;
        alone[6].output_window = 1;
        alone[7].decoding = 1;
        for demands in alone {
            assert!(!demands.is_none(), "{demands:?}");
        }
    }

    /// A prefill worker is held to the input alone: the completion's
    /// bound is its decode's to hold, and so is the way a constraint is
    /// decided. Every other role is held to all of it, and no other demand
    /// moves with the role.
    #[test]
    fn a_prefill_is_held_to_the_input_alone() {
        let asked = Demands {
            runtime: 7,
            vision: true,
            references: References::Caller,
            input_window: 40,
            output_window: 400,
            decoding: 5,
        };
        let input_alone = Demands {
            output_window: 0,
            decoding: 0,
            ..asked
        };
        assert_eq!(asked.held_by(Role::Prefill), input_alone);
        for role in [Role::Single, Role::Decode, Role::Encode] {
            assert_eq!(asked.held_by(role), asked, "{role:?}");
        }
        let budget = Demands {
            output_window: 400,
            ..Demands::default()
        };
        assert!(budget.held_by(Role::Prefill).is_none());
    }

    /// The demands stand in the order a host answers for them, and each has
    /// the stage its pool is shed under.
    #[test]
    fn the_demands_stand_in_a_hosts_order_each_with_its_stage() {
        let order = [
            UnmetDemand::Runtime,
            UnmetDemand::Vision,
            UnmetDemand::References,
            UnmetDemand::Pipeline,
            UnmetDemand::Window,
            UnmetDemand::Decoding,
        ];
        assert!(order.is_sorted());
        assert_eq!(
            order.map(UnmetDemand::stage),
            [
                "wires",
                "vision",
                "references",
                "pipeline",
                "context-window",
                "decoding"
            ]
        );
    }
}
