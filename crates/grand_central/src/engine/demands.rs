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
//!
//! An encode request's encoders are the one list no judge cuts ahead of its
//! decision. The encode pass tests an encoder for a request once, by its
//! media, its wire and its role, and names what a fleet lacks by lifting
//! one of them at a time. A demand enters that test a part at a time: the
//! runtime with the wire, the references and their pipeline with the
//! items', vision with the request's pixels. So the name follows the
//! pass's order whether a request states a fact as a demand or as its own.

use crate::engine::{decision, view::Role};

/// Where a request's media by reference may be processed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
/// nothing. Further demands add their fields: a host starts from
/// `Demands::default()` and assigns the demands it states, since no literal
/// of the record builds outside this crate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
        if !self.on_runtime(facts) {
            return Some(UnmetDemand::Runtime);
        }
        if self.vision && !vision {
            return Some(UnmetDemand::Vision);
        }
        if let Some(unmet) = self.references_unmet_by(facts) {
            return Some(unmet);
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

    /// Whether a worker is on the runtime the request stays on, the first
    /// test of [`Self::unmet_by`] alone. An encode pass reads it beside the
    /// wires its request states: a runtime is the rest of a request's wire.
    #[must_use]
    pub(crate) const fn on_runtime(self, facts: DemandFacts) -> bool {
        self.runtime == 0 || self.runtime == facts.runtime
    }

    /// The demand for media by reference a worker fails, alone:
    /// [`UnmetDemand::References`] where it processes none,
    /// [`UnmetDemand::Pipeline`] where it processes them by the other
    /// pipeline than the request names. `None` where it meets the demand,
    /// and where the request names none.
    #[must_use]
    pub(crate) const fn references_unmet_by(self, facts: DemandFacts) -> Option<UnmetDemand> {
        let host = match self.references {
            References::None => return None,
            References::Any => facts.host_pipeline,
            References::Caller => true,
            References::Worker => false,
        };
        if !facts.references {
            return Some(UnmetDemand::References);
        }
        if facts.host_pipeline != host {
            return Some(UnmetDemand::Pipeline);
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

    /// What an encoder is held to where a pair takes the request beside
    /// it: these demands less both windows and the way the constraint was
    /// decided. Such an encoder runs neither the input nor the completion:
    /// it takes the request's media, and the pair's two workers hold the
    /// rest ([`Self::beside_encoders`]). An encode request that stands
    /// alone has no pair beside it, and its one list is held as its role
    /// is ([`Self::held_by`]).
    #[must_use]
    pub const fn beside_pair(self) -> Self {
        Self {
            input_window: 0,
            output_window: 0,
            decoding: 0,
            ..self
        }
    }

    /// What a side of a pair is held to where an encoder takes the
    /// request's media: these demands less the media's. Whether the request
    /// needs vision and where its references may be processed are the
    /// encoders' to meet ([`Self::beside_pair`]): the pair's two workers
    /// take what the encoders hand on.
    #[must_use]
    pub const fn beside_encoders(self) -> Self {
        Self {
            vision: false,
            references: References::None,
            ..self
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
/// demands add their fields: a host starts from the default and assigns
/// what it knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
    /// An encode request's encoders are shed under none of these: the
    /// encode pass keeps their part of a demand in its one test of an
    /// encoder, and names what they lack in its own order.
    #[must_use]
    pub const fn stage(self) -> &'static str {
        match self {
            Self::Runtime => decision::WIRES,
            Self::Vision => decision::VISION,
            Self::References => decision::REFERENCES,
            Self::Pipeline => decision::PIPELINE,
            Self::Window => decision::CONTEXT_WINDOW,
            Self::Decoding => decision::DECODING,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/demands/tests.rs"]
mod tests;
